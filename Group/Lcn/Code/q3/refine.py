"""CP-SAT timing refinement for an already selected Q3 task structure.

Box batches, route order, transport type, relay site and relay association stay
fixed. Integer-second start times and cumulative resource capacities remove the
construction heuristic's append-only scheduling restriction. Objective bounds
refer only to this fixed, conservatively rounded scheduling model.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
import time

from .solver import charge_time, make_communication, metrics, shift_segments


def _ceil(value: float) -> int:
    return int(math.ceil(float(value) - 1e-9))


def _floor(value: float) -> int:
    return int(math.floor(float(value) + 1e-9))


def _color(trips, identifiers, duration, field):
    """Interval graphs need only their maximum overlap number of colors."""
    available = {identifier: -math.inf for identifier in identifiers}
    for trip in sorted(trips, key=lambda x: (x["start"], x["id"])):
        free = [identifier for identifier in sorted(available)
                if available[identifier] <= trip["start"] + 1e-7]
        if not free:
            raise RuntimeError(f"CP cumulative allocation cannot be colored for {field}")
        chosen = free[0]
        trip[field] = chosen
        available[chosen] = trip["start"] + duration(trip)


def refine_schedule(result, data, model, time_limit_s=45, seed=20260924):
    """Return an improved schedule, or None if the bounded search finds none.

    All durations occupying resources are rounded up, hard deadlines down.
    Relay service ends are extended by two seconds before constructing the
    model, including the corresponding hover/communications energy and charge.
    Returned floating-point trajectories are reconstructed from the original
    physical model; callers must run the independent verifier before delivery.
    """
    from ortools.sat.python import cp_model

    begun = time.perf_counter()
    old_transport = result["trips"]
    old_relays = result["relay_trips"]
    if not old_transport or time_limit_s <= 0:
        return None
    horizon = _ceil(max(t["return"] for t in old_transport + old_relays) + 600.0)
    problem = cp_model.CpModel()
    transport_starts, relay_starts, plans, relay_templates = {}, {}, {}, {}
    transport_occupations = {key: {"drone": [], "battery": []} for key in data["types"]}
    relay_occupations = {"drone": [], "component": []}
    flight_ends = []

    for trip in old_transport:
        tid, par = trip["id"], data["types"][trip["type"]]
        plan = model.plan(trip["type"], trip["box_ids"], trip["route"])
        if plan is None:
            return None
        plans[tid] = plan
        duration = _ceil(plan["duration"])
        start = problem.new_int_var(0, horizon - duration, f"start_{tid}")
        transport_starts[tid] = start
        flight_ends.append(start + duration)
        transport_occupations[trip["type"]]["drone"].append(
            problem.new_fixed_size_interval_var(start, duration, f"drone_{tid}"))
        occupied = _ceil(plan["duration"] + charge_time(plan["soc"], par["charge_full"]))
        transport_occupations[trip["type"]]["battery"].append(
            problem.new_fixed_size_interval_var(start, occupied, f"battery_{tid}"))
        problem.add_hint(start, min(horizon-duration, max(0, _ceil(trip["start"]))))
        for boxid, offset in plan["deliveries"].items():
            deadline = model.boxes[boxid]["deadline"]
            if math.isfinite(deadline):
                problem.add(start + _ceil(offset) <= _floor(deadline))

    par = data["relay"]
    for trip in old_relays:
        rid = trip["id"]
        service_end = trip["service_end"] - trip["start"] + 2.0
        template = model.relay_trip(trip["site_id"], trip["drone"], trip["component"],
                                    0.0, service_end, rid)
        if template is None:
            return None
        relay_templates[rid] = template
        duration = _ceil(template["return_"])
        start = problem.new_int_var(0, horizon-duration, f"start_{rid}")
        relay_starts[rid] = start
        flight_ends.append(start + duration)
        drone_duration = _ceil(template["return_"] + par["turnaround"])
        component_duration = _ceil(template["return_"] + charge_time(template["soc"], par["charge_full"]))
        relay_occupations["drone"].append(
            problem.new_fixed_size_interval_var(start, drone_duration, f"drone_{rid}"))
        relay_occupations["component"].append(
            problem.new_fixed_size_interval_var(start, component_duration, f"component_{rid}"))
        problem.add_hint(start, min(horizon-duration, max(0, _ceil(trip["start"]))))

    for type_id, occupations in transport_occupations.items():
        spec = data["types"][type_id]
        for key, capacity in (("drone", len(spec["drones"])), ("battery", len(spec["battery_ids"]))):
            intervals = occupations[key]
            if intervals:
                problem.add_cumulative(intervals, [1] * len(intervals), capacity)
    for key, capacity in (("drone", len(par["drones"])), ("component", len(par["component_ids"]))):
        intervals = relay_occupations[key]
        if intervals:
            problem.add_cumulative(intervals, [1] * len(intervals), capacity)

    radio_bounds = {}
    for trip in old_transport:
        tid, rid = trip["id"], trip.get("relay_trip_id")
        needs, _, pieces = model.radio(plans[tid])
        required = [piece for piece in pieces if not piece["direct"]]
        if not needs:
            continue
        if not required or rid not in relay_templates:
            return None
        first = min(piece["t0"] for piece in required)
        last = max(piece["t1"] for piece in required)
        template = relay_templates[rid]
        problem.add(relay_starts[rid] + _ceil(template["ready"])
                    <= transport_starts[tid] + _floor(first))
        problem.add(transport_starts[tid] + _ceil(last)
                    <= relay_starts[rid] + _floor(template["service_end"]))
        radio_bounds[tid] = [first, last]

    weights = {boxid: Fraction(str(box["weight"])).limit_denominator(10000)
               for boxid, box in model.boxes.items()}
    scale = math.lcm(*(fraction.denominator for fraction in weights.values()))
    penalties = []
    for trip in old_transport:
        tid = trip["id"]
        for boxid, offset in plans[tid]["deliveries"].items():
            late = problem.new_int_var(0, horizon, f"late_{boxid}")
            problem.add_max_equality(late, [0, transport_starts[tid] + _ceil(offset)
                                           - _floor(model.boxes[boxid]["due"])])
            penalties.append(int(weights[boxid] * scale) * late)
    tardiness = sum(penalties)
    makespan = problem.new_int_var(0, horizon, "joint_makespan")
    problem.add_max_equality(makespan, flight_ends)
    problem.minimize(tardiness)

    def configured_solver(limit):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(limit)
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = int(seed)
        solver.parameters.log_search_progress = False
        return solver

    def snapshot(solver):
        return ({tid: solver.value(var) for tid, var in transport_starts.items()},
                {rid: solver.value(var) for rid, var in relay_starts.items()})

    first_solver = configured_solver(time_limit_s / 2)
    first_status = first_solver.solve(problem)
    first_info = {"status": first_solver.status_name(first_status),
                  "wall_time_s": first_solver.wall_time,
                  "objective_bound": first_solver.best_objective_bound}
    if first_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    best_tardiness = int(round(first_solver.objective_value))
    first_info["objective"] = best_tardiness
    starts_t, starts_r = snapshot(first_solver)
    problem.add(tardiness <= best_tardiness)
    problem.minimize(makespan)
    problem.clear_hints()
    for tid, variable in transport_starts.items():
        problem.add_hint(variable, starts_t[tid])
    for rid, variable in relay_starts.items():
        problem.add_hint(variable, starts_r[rid])
    second_solver = configured_solver(time_limit_s / 2)
    second_status = second_solver.solve(problem)
    second_info = {"status": second_solver.status_name(second_status),
                   "wall_time_s": second_solver.wall_time,
                   "objective_bound": second_solver.best_objective_bound}
    if second_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        starts_t, starts_r = snapshot(second_solver)
        second_info["objective"] = second_solver.objective_value

    refined = deepcopy(result)
    refined.pop("validation", None)
    refined["trips"] = []
    for old in old_transport:
        plan, start = plans[old["id"]], float(starts_t[old["id"]])
        trip = {**old, **{k: value for k, value in plan.items()
                         if k not in ("duration", "segments", "deliveries")}}
        trip.update(start=start, takeoff=start+plan["takeoff"],
                    segments=shift_segments(plan["segments"], start),
                    deliveries={b: start+t for b, t in plan["deliveries"].items()})
        trip["return"] = start + plan["duration"]
        refined["trips"].append(trip)
    refined["relay_trips"] = []
    for old in old_relays:
        rid, start = old["id"], float(starts_r[old["id"]])
        new = model.relay_trip(old["site_id"], old["drone"], old["component"], start,
                               start+relay_templates[rid]["service_end"], rid)
        if new is None:
            return None
        new["return"] = new.pop("return_")
        refined["relay_trips"].append(new)

    for type_id, spec in data["types"].items():
        tasks = [t for t in refined["trips"] if t["type"] == type_id]
        _color(tasks, spec["drones"], lambda t: _ceil(t["return"]-t["start"]), "drone")
        _color(tasks, spec["battery_ids"],
               lambda t: _ceil(t["return"]-t["start"] + charge_time(t["soc"], spec["charge_full"])), "battery")
    _color(refined["relay_trips"], par["drones"],
           lambda t: _ceil(t["return"]-t["start"] + par["turnaround"]), "drone")
    _color(refined["relay_trips"], par["component_ids"],
           lambda t: _ceil(t["return"]-t["start"] + charge_time(t["soc"], par["charge_full"])), "component")
    refined["communication"] = make_communication(refined, model)
    refined["metrics"] = metrics(refined, model)
    before, after = result["metrics"], refined["metrics"]
    if (after["weighted_tardiness"], after["makespan_s"], after["total_energy_kwh"]) >= (
            before["weighted_tardiness"], before["makespan_s"], before["total_energy_kwh"]):
        return None
    refined["refinement"] = {
        "method": "CP-SAT fixed-structure timing and cumulative resource allocation",
        "time_grid_s": 1, "relay_service_extension_s": 2, "time_limit_s": time_limit_s,
        "seed": seed, "workers": 1, "horizon_s": horizon, "weight_integer_scale": scale,
        "phase_1_weighted_tardiness": first_info, "phase_2_makespan": second_info,
        "runtime_s": time.perf_counter()-begun, "previous_metrics": before,
        "fixed": ["box batches", "route order", "transport types", "relay sites", "relay associations"],
        "scope": "Bounds and statuses apply only to this fixed-structure, conservatively rounded timing model, not the full Q3 optimization problem.",
        "relay_requirement_offsets": radio_bounds,
    }
    return refined
