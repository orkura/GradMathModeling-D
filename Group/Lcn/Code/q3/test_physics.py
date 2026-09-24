"""Reproducible geometry checks: python Code/q3/test_physics.py.

The independent line/rectangle oracle checks all touched pixels.  Random
cross-checks are implementation evidence, not substitutes for the analytic
continuous-segment certificate used by the solver.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
import unittest

import numpy as np
import rasterio

try:
    from .physics import Terrain, _clip_to_cell, fspl
except ImportError:
    from physics import Terrain, _clip_to_cell, fspl


def synthetic_terrain(seed=81024):
    terrain = Terrain.__new__(Terrain)
    terrain._x0 = terrain._y0 = 0.0
    terrain._dx = terrain._dy = 1.0
    terrain.width, terrain.height = 8, 7
    terrain.dem = np.random.default_rng(seed).uniform(0, 100, (7, 8))
    terrain.nodata = None
    return terrain


def rectangle_oracle(a, b, width=8, height=7):
    """Independently clip a segment against each closed pixel rectangle."""
    found = set()
    delta = b - a
    for row, col in itertools.product(range(height), range(width)):
        low, high = 0.0, 1.0
        for axis, left, right in ((0, col, col + 1), (1, row, row + 1)):
            if abs(delta[axis]) < 1e-12:
                if not left - 1e-10 <= a[axis] <= right + 1e-10:
                    low = 2.0
                    break
            else:
                enter, leave = sorted(((left - a[axis]) / delta[axis], (right - a[axis]) / delta[axis]))
                low, high = max(low, enter), min(high, leave)
        if low <= high + 1e-10:
            found.add((row, col))
    return found


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.terrain = synthetic_terrain()

    def test_all_touched_pixels_against_independent_oracle(self):
        rng = np.random.default_rng(81024)
        rays = [(rng.uniform([0, 0], [7.999, 6.999]), rng.uniform([0, 0], [7.999, 6.999])) for _ in range(500)]
        rays += [(np.array(a, float), np.array(b, float)) for a, b in [
            ((1, 1), (6, 6)), ((2, 1), (2, 6)), ((1, 3), (7, 3)),
            ((3, 3), (3, 3)), ((0, 0), (7, 6)),
        ]]
        for a, b in rays:
            rows, cols, entry, leave = self.terrain.ray_cells(a, b)
            expected = rectangle_oracle(a, b)
            self.assertEqual(set(zip(rows, cols)), expected)
            self.assertTrue(np.all(entry <= leave + 1e-12))
            self.assertAlmostEqual(self.terrain.max_along(a, b), max(self.terrain.dem[r, c] for r, c in expected))

    def test_obstruction_is_symmetric(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            a = rng.uniform([0, 0, 20], [7.99, 6.99, 200])
            b = rng.uniform([0, 0, 20], [7.99, 6.99, 200])
            self.assertEqual(self.terrain.blocked(a, b), self.terrain.blocked(b, a))

    def test_hidden_obstruction_inside_swept_triangle(self):
        terrain = self.terrain
        terrain.dem[:] = 0
        terrain.dem[4, 2] = 201
        anchor = np.array([0.5, 0.5, 200.0])
        a, b = np.array([0.5, 6.5, 200.0]), np.array([6.5, 6.5, 200.0])
        self.assertFalse(terrain.blocked(a, anchor))
        self.assertFalse(terrain.blocked(b, anchor))
        self.assertTrue(terrain.blocked((a + b) / 2, anchor))
        self.assertFalse(terrain._swept_los_clear(a, b, anchor))
        terrain.dem[4, 2] = 199
        self.assertTrue(terrain._swept_los_clear(a, b, anchor))

    def test_swept_certificates_cross_checked_on_random_rays(self):
        rng = np.random.default_rng(907)
        certified = 0
        for _ in range(100):
            a, b, anchor = [rng.uniform([0, 0, 50], [7.99, 6.99, 200]) for _ in range(3)]
            if self.terrain._swept_los_clear(a, b, anchor):
                certified += 1
                for fraction in np.linspace(0, 1, 51):
                    self.assertFalse(self.terrain.blocked(a + fraction * (b - a), anchor))
        self.assertGreater(certified, 10)

    def test_triangle_clipping_retains_cell_boundary(self):
        polygon = _clip_to_cell(np.array([[0, 0], [2, 0], [0, 2]], float), 1, 1)
        self.assertGreater(polygon.size, 0)
        np.testing.assert_allclose(polygon, np.ones_like(polygon))

    def test_no_data_and_out_of_bounds_fail_closed(self):
        with self.assertRaises(ValueError):
            self.terrain.max_along((-0.1, 1), (3, 3))
        self.terrain.dem[3, 3] = math.nan
        with self.assertRaises(ValueError):
            self.terrain.max_along((3.2, 3.2), (3.8, 3.8))

    def test_free_space_loss_units(self):
        self.assertAlmostEqual(fspl(1000), 32.45 + 20 * math.log10(2400))
        self.assertAlmostEqual(fspl(2000) - fspl(1000), 20 * math.log10(2))
        self.assertEqual(fspl(0), -math.inf)


class SuppliedDemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[4]
        paths = list((root / "D题" / "数据").rglob("*30米DEM.tif"))
        if len(paths) != 1:
            raise unittest.SkipTest("Original DEM not available")
        cls.terrain = Terrain(paths[0])

    def test_original_raster_and_point_pixel_convention(self):
        terrain = self.terrain
        self.assertEqual(terrain.dem.shape, (1309, 1486))
        with rasterio.open(terrain.path) as source:
            self.assertEqual(source.tags().get("AREA_OR_POINT"), "Point")
            np.testing.assert_array_equal(terrain.dem, source.read(1))
        # The original tie point (109.032777..., 23.224722...) is the
        # centre of pixel [0,0], not its north-west corner.
        expected = [109.0326388888889, 22.86125, 109.44541666666667, 23.22486111111111]
        np.testing.assert_allclose(terrain.geographic_bounds, expected, rtol=0, atol=2e-12)

    def test_coordinate_round_trip_and_supplied_depot(self):
        terrain = self.terrain
        lon, lat = 109.2308517, 23.0085095
        xy = terrain.xy(lon, lat)
        np.testing.assert_allclose(terrain.lonlat(*xy), (lon, lat), rtol=0, atol=1e-12)
        self.assertAlmostEqual(terrain.ground(*xy), 128.7255401611328, places=6)
        # Keep the attachment's depot operating altitude 127.7 m separate
        # from the containing DEM pixel: it is not silently overwritten.
        self.assertGreater(abs(terrain.ground(*xy) - 127.7), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
