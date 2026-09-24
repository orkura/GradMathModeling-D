"""Independent Q2 audit: transport physics, box deadlines and reusable resources.

No optimizer or Q3/Q4 solution is imported. Flight paths are reconstructed from
the supplied node elevations and all intersected original DEM cells. All times
are seconds, positions metres and energies kWh. This audits feasibility of the
specified raster model, not heuristic optimality or subpixel terrain safety.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

import numpy as np


TIME_EPS = 1e-5
VALUE_EPS = 1e-5
ENERGY_EPS = 1e-6
POSITION_EPS = 1e-4
GRAVITY = 9.80665


class _Audit:
    def __init__(self):
        self.errors: list[dict] = []
        self.checks: Counter = Counter()
        self.failed: Counter = Counter()
        self.error_count = 0

    def check(self, condition, name, subject, detail):
        self.checks[name] += 1
        if bool(condition):
            return True
        self.error_count += 1
        self.failed[name] += 1
        if len(self.errors) < 200:
            self.errors.append({"check": name, "subject": subject, "detail": detail})
        return False

    def close(self, actual, expected, name, subject, tolerance=VALUE_EPS):
        try:
            actual = float(actual)
        except (TypeError, ValueError):
            actual = math.nan
        return self.check(math.isfinite(actual) and math.isfinite(expected)
                          and abs(actual - expected) <= tolerance,
                          name, subject, f"actual={actual}; expected={expected}; tolerance={tolerance}")


def charge_time(soc: float, full_time: float) -> float:
    """Remaining charge time under the appendix's two-stage linear rule."""
    if not math.isfinite(soc) or not 0 <= soc <= 1:
        raise ValueError("SOC must be finite and lie in [0,1]")
    return full_time * (0.65 * max(0.0, 0.9 - soc) / 0.9
                        + 0.35 * min(0.1, 1.0 - soc) / 0.1)


def _nonfinite(value: Any, location="") -> list[str]:
    """Find non-finite numeric values before any geometry or event arithmetic."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _nonfinite(v, f"{location}.{k}")]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in _nonfinite(v, f"{location}[{i}]")]
    if isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
        return [location]
    return []


def _leg(a, b, par, terrain, start, load):
    altitude = max(float(terrain.max_along(a[:2], b[:2])) + 50.0, a[2], b[2])
    distance = float(np.linalg.norm(b[:2] - a[:2]))
    ascent, descent = altitude - a[2], altitude - b[2]
    vertices = [a, np.array([a[0], a[1], altitude]),
                np.array([b[0], b[1], altitude]), b]
    durations = [ascent / par["up_speed"], distance / par["speed"],
                 descent / par["down_speed"]]
    expected = []
    time = start
    for phase, p0, p1, duration in zip(("climb", "cruise", "descent"),
                                      vertices, vertices[1:], durations):
        if duration > TIME_EPS:
            expected.append({"phase": phase, "t0": time, "t1": time + duration,
                             "p0": p0.tolist(), "p1": p1.tolist(), "load": load})
        time += duration
    effective_range = par["range_empty"] - (par["range_empty"] - par["range_full"]) * (
        max(0.0, load) / par["payload"]) ** 1.5
    if effective_range <= 0 or not math.isfinite(effective_range):
        raise ValueError("nonpositive or non-finite effective range")
    energy = (par["energy"] * distance / effective_range
              + (par["mass_empty"] + load) * GRAVITY * ascent / (par["eta_up"] * 3.6e6))
    return expected, time, energy


def _compare_segments(audit, trip, expected):
    tid = trip["id"]
    supplied = trip["segments"]
    active = []
    for i, seg in enumerate(supplied):
        label = f"{tid}:{i}"
        audit.check(seg["t1"] >= seg["t0"], "nonnegative_segment_duration", label,
                    f"t0={seg['t0']}; t1={seg['t1']}")
        if seg["t1"] - seg["t0"] > TIME_EPS:
            active.append(seg)
    audit.check(len(active) == len(expected), "segment_count", tid,
                f"actual={len(active)}; expected={len(expected)}")
    for i, (actual, want) in enumerate(zip(active, expected)):
        label = f"{tid}:{i}"
        audit.check(actual["phase"] == want["phase"], "flight_phase", label,
                    f"actual={actual['phase']}; expected={want['phase']}")
        for key in ("t0", "t1"):
            audit.close(actual[key], want[key], "segment_time", label, TIME_EPS)
        for key in ("p0", "p1"):
            got = np.asarray(actual[key], dtype=float)
            audit.check(got.shape == (3,) and np.allclose(got, want[key], rtol=0, atol=POSITION_EPS),
                        "segment_geometry", label, f"{key}: actual={actual[key]}; expected={want[key]}")
        audit.close(actual.get("load"), want["load"], "segment_load", label)


def _resource_usage(audit, occupation, name):
    """Half-open intervals: an exact end/start equality permits reuse.

    Resource separation uses strict floating-point comparisons; no tolerance is
    allowed to hide a positive overlap. Reconstruction tolerances apply only
    to comparisons of serialized physical quantities, not event ordering.
    """
    events = []
    for identifier, rows in occupation.items():
        last_end, last_trip = -math.inf, ""
        for begin, end, trip in sorted(rows):
            audit.check(begin >= last_end, name + "_conflict", identifier,
                        f"{trip} starts at {begin}; {last_trip} releases at {last_end}")
            if end > last_end:
                last_end, last_trip = end, trip
            events.extend([(begin, 1), (end, -1)])
    count = peak = 0
    for _, delta in sorted(events):
        count += delta
        peak = max(peak, count)
    return {"resources_used": len(occupation), "peak_occupied": peak,
            "occupations": {key: [list(row) for row in sorted(rows)]
                            for key, rows in sorted(occupation.items())}}


def verify(result: dict[str, Any], data: dict[str, Any], terrain) -> dict[str, Any]:
    """Independently reproduce transport schedules; intentionally no radio audit."""
    audit = _Audit()
    trips = result.get("trips", [])
    boxes = {box["id"]: box for box in data["boxes"]}
    nodes = {key: np.array([*terrain.xy(node["lon"], node["lat"]),
                            node["z"] + node.get("work_agl", 0.0)], dtype=float)
             for key, node in data["nodes"].items()}
    counts = Counter(boxid for trip in trips for boxid in trip.get("box_ids", []))
    audit.check(set(counts) == set(boxes), "box_inventory", "all",
                f"missing={sorted(set(boxes)-set(counts))}; unknown={sorted(set(counts)-set(boxes))}")
    for boxid, count in counts.items():
        audit.check(count == 1, "box_once", boxid, f"occurrences={count}")
    identifiers = [trip.get("id") for trip in trips]
    audit.check(len(identifiers) == len(set(identifiers)), "unique_trip_ids", "all", str(identifiers))
    occupations = {"transport_drone": defaultdict(list), "transport_battery": defaultdict(list)}
    delivery_times, returns, socs = {}, [], []
    total_energy = weighted_tardiness = max_tardiness = 0.0
    hard_violations = late_boxes = multi_stop = 0
    cycles = []
    required = {"id", "type", "box_ids", "route", "drone", "battery", "start", "takeoff",
                "return", "energy", "soc", "mass", "volume", "segments", "deliveries"}
    for trip in trips:
        tid = str(trip.get("id", "<missing>"))
        if not audit.check(required <= trip.keys(), "trip_schema", tid,
                           f"missing={sorted(required-trip.keys())}"):
            continue
        bad = _nonfinite(trip)
        if not audit.check(not bad, "finite_trip_values", tid, str(bad)):
            continue
        if not audit.check(trip["type"] in data["types"], "type", tid, str(trip["type"])):
            continue
        par = data["types"][trip["type"]]
        audit.check(trip["drone"] in par["drones"], "drone_inventory", tid, str(trip["drone"]))
        audit.check(trip["battery"] in par["battery_ids"], "battery_inventory", tid, str(trip["battery"]))
        audit.check(trip["start"] >= 0, "nonnegative_start", tid, str(trip["start"]))
        selected = [boxes[b] for b in trip["box_ids"] if b in boxes]
        if not audit.check(len(selected) == len(trip["box_ids"]) and bool(selected),
                           "valid_trip_boxes", tid, str(trip["box_ids"])):
            continue
        load = sum(box["mass"] for box in selected)
        volume = sum(box["volume"] for box in selected)
        audit.close(trip["mass"], load, "initial_mass", tid)
        audit.close(trip["volume"], volume, "initial_volume", tid)
        audit.check(load <= par["payload"] + VALUE_EPS, "payload", tid,
                    f"load={load}; limit={par['payload']}")
        audit.check(volume <= par["volume"] + VALUE_EPS, "volume", tid,
                    f"volume={volume}; limit={par['volume']}")
        route = trip["route"]
        if not audit.check(set(route) == {box["node"] for box in selected}
                           and len(route) == len(set(route)) and set(route) <= nodes.keys(),
                           "route_destinations", tid, str(route)):
            continue
        multi_stop += int(len(route) > 1)
        audit.check(set(trip["deliveries"]) == set(trip["box_ids"]), "delivery_records", tid,
                    "Every assigned box needs exactly one delivery completion record")
        # Reconstruct in local sortie time, then translate once. Accumulating
        # absolute times at every phase can invent a 1-ulp overlap at exact
        # resource-release boundaries, although the physical interval is equal.
        takeoff_offset = par["prepare"] + par["load_per_box"] * len(selected)
        audit.close(trip["takeoff"], trip["start"] + takeoff_offset,
                    "preparation_loading", tid, TIME_EPS)
        time, energy, expected, current = takeoff_offset, 0.0, [], nodes["O01"]
        try:
            for destination in [*route, "O01"]:
                segments, time, leg_energy = _leg(current, nodes[destination], par, terrain, time, load)
                expected.extend(segments)
                energy += leg_energy
                current = nodes[destination]
                if destination == "O01":
                    continue
                drop = [box for box in selected if box["node"] == destination]
                service = par["service_base"] + par["service_per_box"] * len(drop)
                expected.append({"phase": "delivery", "t0": time, "t1": time + service,
                                 "p0": current.tolist(), "p1": current.tolist(), "load": load})
                time += service
                completed = trip["start"] + time
                for box in drop:
                    bid = box["id"]
                    audit.close(trip["deliveries"].get(bid), completed, "delivery_completion", bid, TIME_EPS)
                    deadline = min(box["due"] if box.get("medical", False) else math.inf,
                                   box["first_deadline"] if box.get("first", False) else math.inf)
                    meets = completed <= deadline + TIME_EPS
                    hard_violations += int(not meets)
                    audit.check(meets, "hard_deadline", bid, f"delivered={completed}; deadline={deadline}")
                    late = max(0.0, completed - box["due"])
                    weighted_tardiness += box["weight"] * late
                    max_tardiness = max(max_tardiness, late)
                    late_boxes += int(late > TIME_EPS)
                    delivery_times[bid] = completed
                load -= sum(box["mass"] for box in drop)
        except (ValueError, ArithmeticError) as exc:
            audit.check(False, "physical_reconstruction", tid, str(exc))
            continue
        time += trip["start"]
        for segment in expected:
            segment["t0"] += trip["start"]
            segment["t1"] += trip["start"]
        audit.close(trip["return"], time, "return_time", tid, TIME_EPS)
        _compare_segments(audit, trip, expected)
        audit.close(trip["energy"], energy, "transport_energy", tid, ENERGY_EPS)
        soc = 1.0 - energy / par["energy"]
        audit.close(trip["soc"], soc, "transport_soc", tid)
        audit.check(soc >= par["reserve"] - VALUE_EPS and soc <= 1 + VALUE_EPS,
                    "transport_reserve", tid, f"SOC={soc}; reserve={par['reserve']}")
        total_energy += energy
        socs.append(soc)
        returns.append(time)
        recharge = charge_time(min(1.0, max(0.0, soc)), par["charge_full"])
        release = time + recharge
        occupations["transport_drone"][trip["drone"]].append((trip["start"], time, tid))
        occupations["transport_battery"][trip["battery"]].append((trip["start"], release, tid))
        cycles.append({"trip_id": tid, "type": trip["type"], "drone": trip["drone"],
                       "battery": trip["battery"], "start": trip["start"], "return": time,
                       "charge_s": recharge, "battery_ready": release, "soc": soc})
    resources = {key: _resource_usage(audit, intervals, key) for key, intervals in occupations.items()}
    resources["cycles"] = cycles
    metrics = {"boxes_delivered": len(delivery_times), "transport_sorties": len(trips),
               "transport_energy_kwh": total_energy, "total_energy_kwh": total_energy,
               "makespan_s": max(returns, default=0.0), "transport_makespan_s": max(returns, default=0.0),
               "weighted_tardiness": weighted_tardiness, "weighted_tardiness_s": weighted_tardiness,
               "hard_deadline_violations": hard_violations, "min_transport_soc": min(socs, default=1.0),
               "late_boxes": late_boxes, "max_tardiness_s": max_tardiness,
               "last_delivery_s": max(delivery_times.values(), default=0.0), "multi_stop_sorties": multi_stop}
    supplied_metrics = result.get("metrics", {})
    audit.check(not _nonfinite(supplied_metrics), "finite_summary", "metrics", str(_nonfinite(supplied_metrics)))
    for name, expected in metrics.items():
        if name in supplied_metrics:
            audit.close(supplied_metrics[name], expected, "summary_metric", name,
                        ENERGY_EPS if "energy" in name else VALUE_EPS)
    return {"passed": audit.error_count == 0, "error_count": audit.error_count,
            "errors": audit.errors, "check_count": sum(audit.checks.values()), "checks": dict(audit.checks),
            "failed_checks": dict(audit.failed), "metrics": metrics,
            "resources": resources, "resource_usage": resources,
            "delivery_times": delivery_times,
            "tolerances": {"time_s": TIME_EPS, "energy_kwh": ENERGY_EPS,
                           "position_m": POSITION_EPS, "scalar": VALUE_EPS,
                           "resource_overlap_s": 0.0},
            "scope": "Independent transport-only Q2 feasibility audit under the original raster/affine-coordinate model; no communications requirement and no optimality certificate."}
