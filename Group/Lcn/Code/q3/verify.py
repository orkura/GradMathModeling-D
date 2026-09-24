"""Independent schedule audit for the supplied problem-3 instance.

Reconstructs flight/time/energy from inputs instead of trusting summary fields.
Radio verification has two separate layers: a 2 s point check and conservative
whole-interval certificates under the original piecewise-constant DEM model.
An unproved interval is a failed audit, not a claimed communications outage.
No optimization routines are imported and no input or result is modified.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import math
from typing import Any

import numpy as np


EPS = 1e-5
ENERGY_EPS = 1e-6
GRAVITY = 9.80665


def charge_time(soc: float, full_time: float) -> float:
    """Appendix 2: 0--90% takes 65%, 90--100% takes 35% of full time."""
    if not -EPS <= soc <= 1 + EPS:
        raise ValueError(f"SOC outside [0,1]: {soc}")
    soc = min(1.0, max(0.0, soc))
    return full_time * (0.65 * max(0.0, 0.9 - soc) / 0.9
                        + 0.35 * min(0.1, 1.0 - soc) / 0.1)


def _position(segment: dict, time: float) -> np.ndarray:
    dt = segment["t1"] - segment["t0"]
    fraction = 0.0 if dt <= EPS else (time - segment["t0"]) / dt
    return np.asarray(segment["p0"], dtype=float) + fraction * (
        np.asarray(segment["p1"], dtype=float) - segment["p0"])


def _xyz(node: dict, terrain) -> np.ndarray:
    xy = terrain.xy(node["lon"], node["lat"])
    return np.array([*xy, node["z"] + node.get("work_agl", 0.0)], dtype=float)


class _Audit:
    def __init__(self):
        self.errors: list[dict] = []
        self.error_count = 0
        self.checks: Counter = Counter()
        self.failed: Counter = Counter()

    def check(self, condition, check: str, subject: str, detail: str) -> bool:
        self.checks[check] += 1
        if bool(condition):
            return True
        self.error_count += 1
        self.failed[check] += 1
        if len(self.errors) < 150:
            self.errors.append({"check": check, "subject": subject, "detail": detail})
        return False

    def close(self, actual: float, expected: float, check: str,
              subject: str, tolerance: float = EPS):
        return self.check(math.isfinite(actual) and abs(actual - expected) <= tolerance,
                          check, subject, f"actual={actual:.9g}; expected={expected:.9g}")


def _resource_conflicts(audit: _Audit, occupations: dict, category: str) -> dict:
    """Check every named resource; count concurrent occupation after charging."""
    events = []
    for resource, intervals in occupations.items():
        latest_end = -math.inf
        latest_id = ""
        for begin, end, identifier in sorted(intervals):
            audit.check(begin >= latest_end - EPS, category, resource,
                        f"{identifier} starts {begin:.6f} before {latest_id} releases {latest_end:.6f}")
            if end > latest_end:
                latest_end, latest_id = end, identifier
            events += [(begin, 1), (end, -1)]
    current = peak = 0
    for _, change in sorted(events):
        current += change
        peak = max(peak, current)
    return {"resources_used": len(occupations), "peak_occupied": peak,
            "occupations": {key: [list(x) for x in sorted(value)]
                            for key, value in occupations.items()}}


def _leg(a: np.ndarray, b: np.ndarray, parameters: dict, terrain,
         time: float, load: float, relay: bool = False):
    """Independently reconstruct appendix-2 geometry, times and energies."""
    height = max(terrain.max_along(a[:2], b[:2]) + 50.0, a[2], b[2])
    ascent, descent = height - a[2], height - b[2]
    distance = float(np.linalg.norm(a[:2] - b[:2]))
    vertices = [a, np.array([a[0], a[1], height]),
                np.array([b[0], b[1], height]), b]
    durations = [ascent / parameters["up_speed"], distance / parameters["speed"],
                 descent / parameters["down_speed"]]
    segments = []
    for phase, p0, p1, duration in zip(("climb", "cruise", "descent"),
                                       vertices, vertices[1:], durations):
        if duration > EPS:
            segments.append({"phase": phase, "t0": time, "t1": time + duration,
                             "p0": p0.tolist(), "p1": p1.tolist(), "load": load})
        time += duration
    if relay:
        horizontal = parameters["power_cruise"] * durations[1] / 3600.0
        mass = parameters["mass"]
    else:
        effective_range = parameters["range_empty"] - (
            parameters["range_empty"] - parameters["range_full"]) * (
                max(0.0, load) / parameters["payload"]) ** 1.5
        if effective_range <= 0:
            raise ValueError("Non-positive effective range")
        horizontal = parameters["energy"] * distance / effective_range
        mass = parameters["mass_empty"] + load
    energy = horizontal + mass * GRAVITY * ascent / (parameters["eta_up"] * 3.6e6)
    return segments, time, energy


def _compare_segments(audit: _Audit, trip: dict, expected: list, transport: bool):
    supplied = trip.get("segments", [])
    valid = True
    for index, seg in enumerate(supplied):
        arrays = [*seg["p0"], *seg["p1"], seg["t0"], seg["t1"]]
        valid &= audit.check(all(math.isfinite(float(v)) for v in arrays)
                             and seg["t1"] >= seg["t0"] - EPS,
                             "finite_segments", trip["id"], f"segment {index}")
    actual = [x for x in supplied if x["t1"] - x["t0"] > EPS]
    valid &= audit.check(len(actual) == len(expected), "segment_count", trip["id"],
                         f"actual={len(actual)}; expected={len(expected)}")
    for i, (got, want) in enumerate(zip(actual, expected)):
        label = f"{trip['id']}:{i}"
        audit.check(got["phase"] == want["phase"], "flight_phase", label,
                    f"actual={got['phase']}; expected={want['phase']}")
        for key in ("t0", "t1"):
            audit.close(float(got[key]), want[key], "flight_time", label)
        for key in ("p0", "p1"):
            audit.check(np.allclose(got[key], want[key], rtol=0, atol=1e-4),
                        "flight_geometry", label, f"{key}: actual={got[key]}; expected={want[key]}")
        if transport:
            audit.close(float(got.get("load", math.nan)), want["load"], "segment_load", label)
    return valid


def verify(result: dict[str, Any], data: dict[str, Any], terrain) -> dict[str, Any]:
    """Return JSON-compatible independent checks, errors and reproduced metrics.

    A passed result certifies the affine metric / constant-DEM model, including
    H=max(maximum DEM+50 m, endpoint operating heights). It does not certify
    real subpixel terrain or global optimality of the optimization heuristic.
    """
    audit = _Audit()
    trips, relays = result.get("trips", []), result.get("relay_trips", [])
    boxes = {box["id"]: box for box in data["boxes"]}
    nodes = {name: _xyz(value, terrain) for name, value in data["nodes"].items()}
    depot = nodes["O01"]
    gateway = depot + np.array([0, 0, data["links"]["gateway_height"]])
    delivered = Counter(boxid for t in trips for boxid in t.get("box_ids", []))
    audit.check(set(delivered) == set(boxes), "box_inventory", "all boxes",
                f"missing={sorted(set(boxes)-set(delivered))}; unknown={sorted(set(delivered)-set(boxes))}")
    for boxid, count in delivered.items():
        audit.check(count == 1, "box_once", boxid, f"occurrences={count}")
    ids = [t["id"] for t in trips + relays]
    audit.check(len(ids) == len(set(ids)), "unique_trip_ids", "all trips", "Trip identifiers must be unique")
    occupations = {key: defaultdict(list) for key in (
        "transport_drone", "transport_battery", "relay_drone", "relay_component")}
    energy_transport = energy_relay = weighted_tardiness = max_tardiness = 0.0
    delivery_times = {}
    for trip in trips:
        tid = trip["id"]
        if not audit.check(trip["type"] in data["types"], "type", tid, str(trip["type"])):
            continue
        par = data["types"][trip["type"]]
        audit.check(trip["drone"] in par["drones"], "drone_inventory", tid, str(trip["drone"]))
        audit.check(trip["battery"] in par["battery_ids"], "battery_inventory", tid, str(trip["battery"]))
        audit.check(trip["start"] >= -EPS, "nonnegative_start", tid, str(trip["start"]))
        selected = [boxes[b] for b in trip["box_ids"] if b in boxes]
        if len(selected) != len(trip["box_ids"]):
            continue
        audit.check(bool(selected), "nonempty_transport_trip", tid, "No boxes assigned")
        load, volume = sum(b["mass"] for b in selected), sum(b["volume"] for b in selected)
        audit.check(load <= par["payload"] + EPS, "payload", tid, f"load={load}; limit={par['payload']}")
        audit.check(volume <= par["volume"] + EPS, "volume", tid, f"volume={volume}; limit={par['volume']}")
        route = trip["route"]
        if not audit.check(set(route) == {b["node"] for b in selected}
                           and len(route) == len(set(route)), "route_destinations", tid,
                           "Route must visit each assigned destination exactly once"):
            continue
        audit.check(set(trip["deliveries"]) == set(trip["box_ids"]), "delivery_records", tid,
                    "Delivery keys differ from assigned box ids")
        takeoff = trip["start"] + par["prepare"] + par["load_per_box"] * len(selected)
        audit.close(trip["takeoff"], takeoff, "preparation_loading", tid)
        time, energy, expected, current = takeoff, 0.0, [], depot
        for destination in [*route, "O01"]:
            leg, time, e = _leg(current, nodes[destination], par, terrain, time, load)
            expected.extend(leg)
            energy += e
            current = nodes[destination]
            if destination == "O01":
                continue
            drop = [box for box in selected if box["node"] == destination]
            duration = par["service_base"] + par["service_per_box"] * len(drop)
            expected.append({"phase": "delivery", "t0": time, "t1": time + duration,
                             "p0": current.tolist(), "p1": current.tolist(), "load": load})
            time += duration
            for box in drop:
                actual = float(trip["deliveries"].get(box["id"], math.nan))
                audit.close(actual, time, "delivery_completion", box["id"])
                audit.check(time <= box["deadline"] + EPS, "hard_deadline", box["id"],
                            f"delivered={time:.6f}; deadline={box['deadline']}")
                late = max(0.0, time - box["due"])
                weighted_tardiness += box["weight"] * late
                max_tardiness = max(max_tardiness, late)
                delivery_times[box["id"]] = time
            load -= sum(b["mass"] for b in drop)
        audit.close(trip["return"], time, "return_time", tid)
        _compare_segments(audit, trip, expected, True)
        audit.close(trip["energy"], energy, "transport_energy", tid, ENERGY_EPS)
        soc = 1.0 - energy / par["energy"]
        audit.close(trip["soc"], soc, "transport_soc", tid)
        audit.check(soc >= par["reserve"] - EPS, "transport_reserve", tid,
                    f"SOC={soc:.9g}; reserve={par['reserve']}")
        energy_transport += energy
        occupations["transport_drone"][trip["drone"]].append((trip["start"], time, tid))
        recharge = charge_time(min(1.0, max(0.0, soc)), par["charge_full"])
        occupations["transport_battery"][trip["battery"]].append((trip["start"], time + recharge, tid))

    par = data["relay"]
    for trip in relays:
        tid, point = trip["id"], np.asarray(trip["position"], dtype=float)
        audit.check(trip["drone"] in par["drones"], "relay_drone_inventory", tid, str(trip["drone"]))
        audit.check(trip["component"] in par["component_ids"], "relay_component_inventory", tid, str(trip["component"]))
        audit.check(trip["start"] >= -EPS, "nonnegative_start", tid, str(trip["start"]))
        if not audit.check(terrain.contains(*point[:2]), "relay_dem_bounds", tid, str(point)):
            continue
        ground = terrain.ground(*point[:2])
        audit.check(-EPS <= point[2] - ground <= par["max_agl"] + EPS,
                    "relay_agl", tid, f"AGL={point[2]-ground}; upper={par['max_agl']}")
        takeoff = trip["start"] + par["prepare"]
        audit.close(trip["takeoff"], takeoff, "relay_preparation", tid)
        expected, arrival, energy = _leg(depot, point, par, terrain, takeoff, 0.0, True)
        ready = arrival + par["link_time"]
        audit.close(trip["ready"], ready, "relay_link_setup", tid)
        audit.check(trip["service_end"] >= ready - EPS, "relay_service_time", tid,
                    f"ready={ready}; service_end={trip['service_end']}")
        hover_duration = max(0.0, trip["service_end"] - arrival)
        expected.append({"phase": "hover", "t0": arrival, "t1": trip["service_end"],
                         "p0": point.tolist(), "p1": point.tolist(), "load": 0.0})
        energy += (par["power_hover"] + par["power_comm"]) * hover_duration / 3600.0
        back, end, extra = _leg(point, depot, par, terrain, trip["service_end"], 0.0, True)
        expected.extend(back)
        energy += extra
        audit.close(trip["return"], end, "relay_return_time", tid)
        _compare_segments(audit, trip, expected, False)
        audit.close(trip["energy"], energy, "relay_energy", tid, ENERGY_EPS)
        soc = 1.0 - energy / par["energy"]
        audit.close(trip["soc"], soc, "relay_soc", tid)
        audit.check(soc >= par["reserve"] - EPS, "relay_reserve", tid,
                    f"SOC={soc:.9g}; reserve={par['reserve']}")
        energy_relay += energy
        occupations["relay_drone"][trip["drone"]].append((trip["start"], end + par["turnaround"], tid))
        recharge = charge_time(min(1.0, max(0.0, soc)), par["charge_full"])
        occupations["relay_component"][trip["component"]].append((trip["start"], end + recharge, tid))

    resources = {key: _resource_conflicts(audit, value, key + "_conflict")
                 for key, value in occupations.items()}
    communication = _verify_communication(result, data, terrain, gateway, audit)
    metrics = {"boxes_delivered": len(delivery_times), "transport_sorties": len(trips),
               "relay_sorties": len(relays), "transport_energy_kwh": energy_transport,
               "relay_energy_kwh": energy_relay, "total_energy_kwh": energy_transport + energy_relay,
               "weighted_tardiness_s": weighted_tardiness, "max_tardiness_s": max_tardiness,
               "last_delivery_s": max(delivery_times.values(), default=0.0),
               "transport_makespan_s": max((t["return"] for t in trips), default=0.0),
               "joint_makespan_s": max((t["return"] for t in trips + relays), default=0.0)}
    return {"passed": audit.error_count == 0, "error_count": audit.error_count,
            "errors": audit.errors, "warnings": [], "checks": dict(audit.checks),
            "failed_checks": dict(audit.failed), "metrics": metrics,
            "communication": communication, "resource_usage": resources,
            "scope": "Independent time/energy/resource reconstruction; 2 s radio samples plus continuous interval certificates in the affine metric, piecewise-constant original DEM model. No global optimality claim."}


def _verify_communication(result, data, terrain, gateway, audit: _Audit) -> dict:
    """Recompute budgets, samples and certificates, ignoring solver pass flags."""
    links = data["links"]
    budgets = {}
    for kind, a, b in (("direct", "transport", "gateway"),
                       ("access", "transport", "relay_access"),
                       ("backhaul", "relay_backhaul", "gateway")):
        aa, bb = links[a], links[b]
        budgets[kind] = min(aa["pt"], bb["pt"]) + aa["gain"] + bb["gain"] - (
            links["system_loss"] + links["sensitivity"] + links["fade_margin"])
    frequency_term = 32.45 + 20 * math.log10(links["frequency_mhz"])
    loss_obstructed = links["obstruction_loss"]

    def raw_margin(point, anchor, kind):
        distance = float(np.linalg.norm(np.asarray(point) - anchor))
        return budgets[kind] - frequency_term - 20 * math.log10(max(distance, 1e-10) / 1000)

    @lru_cache(maxsize=150000)
    def point_margin(point, anchor, kind):
        raw = raw_margin(point, anchor, kind)
        # The full obstruction penalty is a valid lower bound. Only ambiguous
        # cases require the exact all-intersected-pixel ray test.
        if raw >= loss_obstructed:
            return raw - loss_obstructed
        if raw < 0:
            return raw
        return raw - loss_obstructed * terrain.blocked(point, anchor)

    def lower_bound(a, b, anchor, kind):
        raw = min(raw_margin(a, anchor, kind), raw_margin(b, anchor, kind))
        if raw - loss_obstructed >= 0 or raw < 0:
            return raw - loss_obstructed
        if terrain._swept_los_clear(np.asarray(a), np.asarray(b), np.asarray(anchor)):
            return raw
        return raw - loss_obstructed

    g = tuple(gateway)
    relays = {t["id"]: t for t in result.get("relay_trips", [])}
    backhaul = {rid: point_margin(tuple(r["position"]), g, "backhaul")
                for rid, r in relays.items()}
    for rid, margin in backhaul.items():
        audit.check(margin >= -EPS, "relay_backhaul", rid, f"margin={margin:.9g} dB")
    boundaries = sorted({float(t[key]) for t in relays.values() for key in ("ready", "service_end")})
    samples = outages = direct_samples = relay_samples = 0
    min_margin = math.inf
    proven = 0
    unresolved = []

    def active(t0, t1):
        return [r for r in relays.values() if r["ready"] <= t0 + EPS
                and r["service_end"] >= t1 - EPS and backhaul[r["id"]] >= -EPS]

    def certify(segment, t0, t1, candidates):
        nonlocal proven
        a, b = _position(segment, t0), _position(segment, t1)
        if lower_bound(a, b, gateway, "direct") >= -EPS:
            proven += 1
            return
        for relay in candidates:
            if lower_bound(a, b, relay["position"], "access") >= -EPS:
                proven += 1
                return
        if t1 - t0 <= 0.25 + EPS:
            unresolved.append([segment["_trip_id"], t0, t1])
            return
        middle = (t0 + t1) / 2
        certify(segment, t0, middle, candidates)
        certify(segment, middle, t1, candidates)

    for trip in result.get("trips", []):
        for source in trip["segments"]:
            segment = dict(source, _trip_id=trip["id"])
            t0, t1 = float(segment["t0"]), float(segment["t1"])
            if t1 < t0:
                continue
            # Include endpoints and each segment boundary; spacing never >2 s.
            times = np.linspace(t0, t1, max(1, math.ceil((t1 - t0) / 2)) + 1)
            for time in times:
                p = tuple(_position(segment, float(time)))
                direct = point_margin(p, g, "direct")
                samples += 1
                if direct >= -EPS:
                    direct_samples += 1
                    chosen = direct
                else:
                    margins = [min(point_margin(p, tuple(r["position"]), "access"), backhaul[r["id"]])
                               for r in active(float(time), float(time))]
                    chosen = max(margins, default=-math.inf)
                    if chosen >= -EPS:
                        relay_samples += 1
                    else:
                        outages += 1
                        audit.check(False, "sampled_communication", trip["id"],
                                    f"No link at {time:.6f}s, phase={segment['phase']}")
                min_margin = min(min_margin, chosen)
            # Ten-second partitions limit triangle extent; readiness changes
            # are exact event boundaries so one relay is active throughout.
            splits = sorted({t0, t1, *(x for x in boundaries if t0 < x < t1),
                             *np.arange(t0 + 10, t1, 10).tolist()})
            for begin, end in zip(splits, splits[1:]):
                certify(segment, begin, end, active(begin, end))

    audit.check(not unresolved, "continuous_communication", "all flight intervals",
                f"{len(unresolved)} intervals have no conservative certificate down to 0.25s; sample outages={outages}")

    def record_certificate(segment, begin, end, anchor, kind):
        """A merged record is a union of intervals, with the same provider.

        A single swept triangle may be too conservative for a long merged row.
        Repartitioning preserves continuous coverage and does not substitute
        an unrecorded relay or a point-sampling argument for the stated link.
        """
        margin = lower_bound(_position(segment, begin), _position(segment, end), anchor, kind)
        if margin >= -EPS or end - begin <= 0.25 + EPS:
            return margin
        middle = (begin + end) / 2
        first = record_certificate(segment, begin, middle, anchor, kind)
        if first < -EPS:
            return first
        return min(first, record_certificate(segment, middle, end, anchor, kind))

    records = result.get("communication", [])
    if records:
        by_trip = defaultdict(list)
        for block in records:
            by_trip[block["trip_id"]].append(block)
        for trip in result.get("trips", []):
            blocks = sorted(by_trip[trip["id"]], key=lambda b: (b["t0"], b["t1"]))
            cursor = float(trip["takeoff"])
            for block in blocks:
                audit.close(float(block["t0"]), cursor, "communication_record_contiguity", trip["id"])
                audit.check(block["t1"] >= block["t0"] - EPS, "communication_record_duration", trip["id"], str(block))
                cursor = float(block["t1"])
                for segment in trip["segments"]:
                    a, b = max(block["t0"], segment["t0"]), min(block["t1"], segment["t1"])
                    if b <= a + EPS:
                        continue
                    mode = block["mode"]
                    if mode == "direct":
                        margin = record_certificate(segment, a, b, gateway, "direct")
                    elif mode == "relay" and block.get("relay_trip_id") in relays:
                        r = relays[block["relay_trip_id"]]
                        audit.check(r["ready"] <= a + EPS and r["service_end"] >= b - EPS,
                                    "record_relay_active", trip["id"], str(block))
                        margin = min(record_certificate(segment, a, b, r["position"], "access"), backhaul[r["id"]])
                        # Records designate the guaranteed provider; direct
                        # priority is checked separately at sampled instants.
                    else:
                        margin = -math.inf
                    audit.check(margin >= -EPS, "communication_record_certificate", trip["id"],
                                f"mode={mode}, interval=[{a},{b}], margin={margin}")
            audit.close(cursor, float(trip["return"]), "communication_record_complete", trip["id"])
    return {"sample_step_s": 2.0, "samples": samples, "interruptions": outages,
            "direct_samples": direct_samples, "relay_samples": relay_samples,
            "continuous_certified": not unresolved, "certified_intervals": proven,
            "uncertified_intervals_count": len(unresolved), "uncertified_intervals": unresolved[:50],
            "min_sample_margin_lower_bound_db": min_margin if math.isfinite(min_margin) else None,
            "budgets_db": budgets, "certificate_min_interval_s": 0.25,
            "evidence": "2 s samples are finite checks; interval bounds separately certify all times under the original DEM raster model."}
