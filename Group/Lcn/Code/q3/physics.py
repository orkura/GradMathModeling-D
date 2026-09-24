"""DEM geometry and bidirectional radio budgets for problem 3.

Coordinates are metres in a local affine WGS84 tangent approximation.  This
explicit modelling approximation preserves straight lines in the supplied
geographic raster, so every intersected original DEM pixel is inspected.
Terrain is piecewise constant at its pixel elevation; touching a pixel counts
as intersecting it.  Results certify this raster model, not subpixel terrain.
The GeoTIFF PixelIsPoint half-pixel convention is handled by rasterio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import math

import numpy as np
import rasterio


FREQUENCY_MHZ = 2400.0
OBSTRUCTION_LOSS_DB = 10.0
# min(Pt_a + G_a + G_b, Pt_b + G_b + G_a) - Lsys - (Psens + M)
# Values in 通信链路参数.xlsx: Lsys=3, Psens=-98, M=8.
LINK_BUDGET_DB = {"direct": 122.0, "access": 116.0, "backhaul": 126.0}


def fspl(distance_m: float) -> float:
    """Free-space loss in dB, with MHz/km units from appendix 3."""
    if distance_m < 0:
        raise ValueError("Distance must be nonnegative")
    if distance_m == 0:
        return -math.inf
    return 32.45 + 20 * math.log10(FREQUENCY_MHZ) + 20 * math.log10(distance_m / 1000)


def _clip_to_cell(vertices: np.ndarray, col: int, row: int) -> np.ndarray:
    """Intersect a projected LOS triangle with one closed raster square."""
    polygon = [vertex for vertex in vertices]
    for axis, bound, sense in ((0, col, 1), (0, col + 1, -1), (1, row, 1), (1, row + 1, -1)):
        if not polygon:
            break
        output = []
        previous = polygon[-1]
        previous_inside = sense * (previous[axis] - bound) >= 0
        for current in polygon:
            current_inside = sense * (current[axis] - bound) >= 0
            if current_inside != previous_inside:
                fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
                output.append(previous + fraction * (current - previous))
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        polygon = output
    return np.asarray(polygon)


class Terrain:
    """Read-only geometry over the original north-up EPSG:4326 GeoTIFF.

    ``xy`` and ``lonlat`` accept scalars or broadcastable numpy arrays.
    Positions passed to ray methods are local metric (x, y[, altitude_m]).
    Out-of-DEM or nodata rays raise ValueError rather than being declared safe.
    """

    def __init__(self, dem_path: str | Path):
        self.path = Path(dem_path)
        with rasterio.open(self.path) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                raise ValueError("Expected the original EPSG:4326 geographic DEM")
            if abs(src.transform.b) > 1e-15 or abs(src.transform.d) > 1e-15:
                raise ValueError("Rotated raster is not supported")
            self.dem = src.read(1).astype(float)
            self.data = self.dem
            self.transform = src.transform
            self.nodata = src.nodata
            self.geographic_bounds = tuple(src.bounds)
        self.height, self.width = self.dem.shape
        west, south, east, north = self.geographic_bounds
        self.lon0, self.lat0 = (west + east) / 2, (south + north) / 2
        # WGS84 meridian / prime-vertical radii at the DEM centre.
        lat = math.radians(self.lat0)
        a, e2 = 6378137.0, 6.6943799901413165e-3
        w = math.sqrt(1 - e2 * math.sin(lat) ** 2)
        self.metres_per_lon = math.pi / 180 * a / w * math.cos(lat)
        self.metres_per_lat = math.pi / 180 * a * (1 - e2) / w**3
        self._x0, self._y0 = self.xy(self.transform.c, self.transform.f)
        self._dx = self.transform.a * self.metres_per_lon
        self._dy = self.transform.e * self.metres_per_lat
        self.resolution_m = (abs(self._dx), abs(self._dy))
        lo, hi = self.xy(west, south), self.xy(east, north)
        self.bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

    def xy(self, lon, lat) -> np.ndarray:
        return np.stack(np.broadcast_arrays(
            (np.asarray(lon) - self.lon0) * self.metres_per_lon,
            (np.asarray(lat) - self.lat0) * self.metres_per_lat,
        ), axis=-1)

    def lonlat(self, x, y) -> np.ndarray:
        return np.stack(np.broadcast_arrays(
            np.asarray(x) / self.metres_per_lon + self.lon0,
            np.asarray(y) / self.metres_per_lat + self.lat0,
        ), axis=-1)

    def _pixel_xy(self, xy) -> np.ndarray:
        a = np.asarray(xy, dtype=float)
        return (a - (self._x0, self._y0)) / (self._dx, self._dy)

    def contains(self, x: float, y: float) -> bool:
        col, row = self._pixel_xy((x, y))
        return bool(0 <= col < self.width and 0 <= row < self.height)

    def _elevations(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        if np.any(rows < 0) or np.any(rows >= self.height) or np.any(cols < 0) or np.any(cols >= self.width):
            raise ValueError("Geometry leaves the DEM coverage")
        z = self.dem[rows, cols]
        invalid = ~np.isfinite(z)
        if self.nodata is not None:
            invalid |= z == self.nodata
        if np.any(invalid):
            raise ValueError("Geometry crosses DEM nodata")
        return z

    def ground(self, x: float, y: float) -> float:
        """Containing pixel elevation (node operations use supplied node height)."""
        col, row = np.floor(self._pixel_xy((x, y))).astype(int)
        return float(self._elevations(np.array([row]), np.array([col]))[0])

    def ray_cells(self, a_xy: Sequence[float], b_xy: Sequence[float]):
        """Return (rows, cols, entry_t, exit_t) for every touched pixel.

        Crossing times come from the exact affine grid boundaries, not distance
        sampling.  Zero-duration entries retain pixels touched at a corner;
        both sides of a ray running on a boundary are conservatively included.
        """
        start, end = self._pixel_xy(a_xy), self._pixel_xy(b_xy)
        delta = end - start
        for p in (start, end):
            if not (0 <= p[0] < self.width and 0 <= p[1] < self.height):
                raise ValueError("Ray endpoint leaves the DEM coverage")
        events = [np.array([0.0, 1.0])]
        for axis in (0, 1):
            if abs(delta[axis]) > 1e-12:
                lo, hi = sorted((start[axis], end[axis]))
                edges = np.arange(math.floor(lo) + 1, math.ceil(hi), dtype=float)
                if len(edges):
                    events.append((edges - start[axis]) / delta[axis])
        times = np.unique(np.clip(np.concatenate(events), 0, 1))
        t0, t1 = times[:-1], times[1:]
        midpoint = start + ((t0 + t1) / 2)[:, None] * delta
        ix = np.floor(midpoint).astype(int)
        cols, rows = [ix[:, 0]], [ix[:, 1]]
        entries, exits = [t0], [t1]
        # Grid-aligned rays touch both neighbouring strips throughout.
        for axis in (0, 1):
            if abs(delta[axis]) <= 1e-12 and abs(start[axis] - round(start[axis])) <= 1e-9:
                old_cols, old_rows = np.concatenate(cols), np.concatenate(rows)
                old_t0, old_t1 = np.concatenate(entries), np.concatenate(exits)
                cols.append(old_cols - (axis == 0))
                rows.append(old_rows - (axis == 1))
                entries.append(old_t0)
                exits.append(old_t1)
        # Every crossing includes all pixels sharing that precise point.
        pts = start + times[:, None] * delta
        base = np.floor(pts + 1e-10).astype(int)
        on_col = np.abs(pts[:, 0] - np.round(pts[:, 0])) < 1e-9
        on_row = np.abs(pts[:, 1] - np.round(pts[:, 1])) < 1e-9
        for dc, dr in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
            keep = np.ones(len(times), dtype=bool)
            if dc:
                keep &= on_col
            if dr:
                keep &= on_row
            cols.append(base[keep, 0] + dc)
            rows.append(base[keep, 1] + dr)
            entries.append(times[keep])
            exits.append(times[keep])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        entry, leave = np.concatenate(entries), np.concatenate(exits)
        # Outer raster boundary has no adjoining pixel outside the dataset.
        keep = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        return rows[keep], cols[keep], entry[keep], leave[keep]

    def max_along(self, a_xy: Sequence[float], b_xy: Sequence[float]) -> float:
        rows, cols, _, _ = self.ray_cells(a_xy, b_xy)
        return float(np.max(self._elevations(rows, cols)))

    def blocked(self, a_xyz: Sequence[float], b_xyz: Sequence[float]) -> bool:
        a, b = np.asarray(a_xyz, dtype=float), np.asarray(b_xyz, dtype=float)
        rows, cols, entry, leave = self.ray_cells(a[:2], b[:2])
        dz = b[2] - a[2]
        low_z = a[2] + dz * (entry if dz >= 0 else leave)
        return bool(np.any(self._elevations(rows, cols) > low_z + 1e-7))

    def link_margin(self, a_xyz, b_xyz, kind: str = "direct") -> float:
        """Positive margin means the bidirectional link is available."""
        a, b = np.asarray(a_xyz, dtype=float), np.asarray(b_xyz, dtype=float)
        budget = LINK_BUDGET_DB[kind]
        distance = float(np.linalg.norm(a - b))
        unobstructed_margin = budget - fspl(distance)
        # Obstruction cannot change the pass/fail decision in these ranges,
        # but exact margin still includes the actual obstruction status.
        return unobstructed_margin - OBSTRUCTION_LOSS_DB * self.blocked(a, b)

    def link(self, a_xyz, b_xyz, kind: str = "direct", margin_db: float = 0.0) -> bool:
        a, b = np.asarray(a_xyz, dtype=float), np.asarray(b_xyz, dtype=float)
        raw = LINK_BUDGET_DB[kind] - fspl(float(np.linalg.norm(a - b))) - margin_db
        if raw >= OBSTRUCTION_LOSS_DB:
            return True
        if raw < 0:
            return False
        return not self.blocked(a, b)

    def _swept_los_clear(self, a: np.ndarray, b: np.ndarray, anchor: np.ndarray) -> bool:
        """Prove every ray anchor -> [a,b] is unobstructed in the raster model.

        The swept sightlines form a 3-D triangle.  Each raster cell intersecting
        its projection is first checked against a lower bound of the triangle
        plane over the whole cell.  Ambiguous cells are clipped against the
        projected triangle, then tested at all intersection vertices, where a
        linear height function attains its extrema.  This is a finite geometry
        certificate rather than a temporal sample test.
        """
        vertices = np.array([self._pixel_xy(p[:2]) for p in (a, b, anchor)])
        matrix = np.column_stack((vertices, np.ones(3)))
        determinant = np.linalg.det(matrix)
        if abs(determinant) < 1e-9:
            return not (self.blocked(a, anchor) or self.blocked(b, anchor) or self.blocked(a, b))
        zcoef = np.linalg.solve(matrix, np.array([a[2], b[2], anchor[2]]))
        minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
        if minimum[0] < 0 or minimum[1] < 0 or maximum[0] >= self.width or maximum[1] >= self.height:
            raise ValueError("Swept radio geometry leaves DEM")
        # Retain adjacent cells when the projected triangle just touches their
        # boundary.  A one-cell padding is harmless after the intersection test.
        lo = np.maximum(np.floor(minimum).astype(int) - 1, 0)
        hi = np.minimum(np.floor(maximum).astype(int) + 1, (self.width - 1, self.height - 1))
        cc, rr = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        cols, rows = cc.ravel(), rr.ravel()
        # Separating-axis test: bbox already supplies the square's two axes;
        # the remaining three axes are the triangle-edge normals.
        overlap = np.ones(len(rows), dtype=bool)
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = second - first
            normal = np.array([-edge[1], edge[0]])
            projections = vertices @ normal
            cell_low = normal[0] * cols + normal[1] * rows + min(0.0, normal[0]) + min(0.0, normal[1])
            cell_high = normal[0] * cols + normal[1] * rows + max(0.0, normal[0]) + max(0.0, normal[1])
            overlap &= (cell_low <= projections.max() + 1e-9) & (cell_high >= projections.min() - 1e-9)
        rows, cols = rows[overlap], cols[overlap]
        plane_low = zcoef[0] * cols + zcoef[1] * rows + zcoef[2] + min(0.0, zcoef[0]) + min(0.0, zcoef[1])
        elevations = self._elevations(rows, cols)
        for index in np.flatnonzero(elevations > plane_low + 1e-7):
            polygon = _clip_to_cell(vertices, int(cols[index]), int(rows[index]))
            if polygon.size:
                lowest = float(np.min(polygon @ zcoef[:2] + zcoef[2]))
                if elevations[index] > lowest + 1e-7:
                    return False
        return True

    def segment_link_lower_bound(self, a_xyz, b_xyz, anchor_xyz, kind: str = "direct") -> float:
        """Certified lower bound of margin over a straight trajectory segment.

        Maximum distance to a fixed anchor is attained at one endpoint.  The
        full 10 dB obstruction penalty always gives a valid bound; a swept DEM
        triangle can sharpen it to the unobstructed bound.  Negative output is
        inconclusive, so subdivide or use another simultaneously active relay.
        """
        a, b, anchor = [np.asarray(p, dtype=float) for p in (a_xyz, b_xyz, anchor_xyz)]
        farthest = max(float(np.linalg.norm(a - anchor)), float(np.linalg.norm(b - anchor)))
        raw_margin = LINK_BUDGET_DB[kind] - fspl(farthest)
        lower = raw_margin - OBSTRUCTION_LOSS_DB
        if lower >= 0 or raw_margin < 0:
            return lower
        if self._swept_los_clear(a, b, anchor):
            return raw_margin
        return lower

    def metadata(self) -> dict:
        return {
            "dem_shape": [self.height, self.width],
            "geographic_bounds": list(self.geographic_bounds),
            "resolution_m": list(self.resolution_m),
            "coordinate_model": "local affine WGS84 metric approximation at DEM centre",
            "origin_lon_lat": [self.lon0, self.lat0],
            "terrain_model": "original raster cells, constant elevation, all touched cells inspected",
            "radio_frequency_mhz": FREQUENCY_MHZ,
            "obstruction_loss_db": OBSTRUCTION_LOSS_DB,
            "link_budget_db": LINK_BUDGET_DB,
        }

    def candidate_relay_positions(
        self, nodes: dict, step_m: float = 500.0,
        agl_values: Sequence[float] = (150.0, 300.0),
        max_candidates: int = 40,
    ) -> list[dict]:
        """Screen finite relay sites against depot-to-node trajectory samples.

        Includes up to three sites covering every sampled direct-blind point
        for each node, plus five sites with the largest aggregate coverage.
        ``coverage`` names nodes with complete *sampled* coverage, never a
        continuous-time certificate.  A caller must certify its actual routes
        with ``segment_link_lower_bound`` before accepting a schedule.
        ``candidate_search`` records nodes lacking a complete sampled option.
        Height values must already respect the attached relay height limit.
        """
        if step_m <= 0 or max_candidates < 1 or not agl_values or min(agl_values) < 0:
            raise ValueError("Invalid relay candidate settings")
        depot = nodes["O01"]
        origin = self.xy(depot["lon"], depot["lat"])
        gateway = np.r_[origin, depot["z"] + 20.0]
        points, labels = [], []
        sample_spacing = 50.0
        for identifier, node in nodes.items():
            if identifier == "O01":
                continue
            xy = self.xy(node["lon"], node["lat"])
            altitude = node["z"] + node.get("work_agl", 30.0)
            cruise = max(self.max_along(origin, xy) + 50.0, altitude, depot["z"])
            vertices = [np.r_[origin, depot["z"]], np.r_[origin, cruise], np.r_[xy, cruise], np.r_[xy, altitude]]
            for a, b in zip(vertices[:-1], vertices[1:]):
                count = max(2, math.ceil(np.linalg.norm(b - a) / sample_spacing) + 1)
                for p in np.linspace(a, b, count):
                    if not self.link(p, gateway, "direct"):
                        points.append(p)
                        labels.append(identifier)
        if not points:
            self.candidate_search = {"grid_sites": 0, "sampled_blind_points": 0, "missing_nodes": [], "sample_spacing_m": sample_spacing}
            return []
        points, labels = np.asarray(points), np.asarray(labels)
        targets = {label: labels == label for label in np.unique(labels)}
        low = points[:, :2].min(axis=0) - 2 * step_m
        high = points[:, :2].max(axis=0) + 2 * step_m
        xy_sites = [(float(x), float(y)) for x in np.arange(low[0], high[0], step_m) for y in np.arange(low[1], high[1], step_m)]
        xy_sites += [tuple(self.xy(n["lon"], n["lat"])) for n in nodes.values()]
        candidates = []
        for x, y in xy_sites:
            if not self.contains(x, y):
                continue
            ground = self.ground(x, y)
            for agl in agl_values:
                position = np.array([x, y, ground + agl])
                if not self.link(position, gateway, "backhaul"):
                    continue
                distances = np.linalg.norm(points - position, axis=1)
                raw_margin = LINK_BUDGET_DB["access"] - (32.45 + 20 * math.log10(FREQUENCY_MHZ) + 20 * np.log10(np.maximum(distances, 1e-9) / 1000))
                mask = raw_margin >= OBSTRUCTION_LOSS_DB
                for index in np.flatnonzero((raw_margin >= 0) & ~mask):
                    mask[index] = not self.blocked(points[index], position)
                coverage = [label for label, target in targets.items() if np.all(mask[target])]
                distance = float(np.linalg.norm(position[:2] - origin))
                candidates.append({
                    "position": position.tolist(), "lon": float(self.lonlat(x, y)[0]),
                    "lat": float(self.lonlat(x, y)[1]), "agl": float(agl),
                    "coverage": coverage, "sampled_points_covered": int(mask.sum()),
                    "sampled_points_total": len(points),
                    "screening_cost": distance + 2 * max(0.0, position[2] - depot["z"]),
                })
        options = {}
        for label in targets:
            eligible = [i for i, candidate in enumerate(candidates) if label in candidate["coverage"]]
            options[label] = sorted(eligible, key=lambda i: (candidates[i]["screening_cost"], -candidates[i]["sampled_points_covered"]))[:3]
        chosen = []
        # First preserve one option for every feasible node, then alternatives.
        for rank in range(3):
            for label in sorted(options):
                if len(options[label]) > rank and options[label][rank] not in chosen:
                    chosen.append(options[label][rank])
        global_best = sorted(range(len(candidates)), key=lambda i: (-candidates[i]["sampled_points_covered"], candidates[i]["screening_cost"]))[:5]
        chosen += [i for i in global_best if i not in chosen]
        chosen = chosen[:max_candidates]
        result = []
        for number, index in enumerate(chosen, 1):
            item = candidates[index].copy()
            item["id"] = f"SITE-{number:03d}"
            result.append(item)
        represented = {label for item in result for label in item["coverage"]}
        self.candidate_search = {
            "grid_sites": len(xy_sites) * len(agl_values),
            "backhaul_feasible_sites": len(candidates),
            "selected_sites": len(result),
            "sampled_blind_points": len(points),
            "sample_spacing_m": sample_spacing,
            "grid_spacing_m": step_m,
            "agl_values_m": list(agl_values),
            "missing_nodes": sorted(set(targets) - represented),
            "coverage_kind": "sampled screening only; continuous route certification required",
        }
        return result
