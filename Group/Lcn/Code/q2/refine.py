"""Single-worker integer-second CP-SAT refinement of fixed Q2 trip structures."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
import time

from q3.solver import charge_time, shift_segments
from .solver import metrics, objective


def _ceil(value):
    """Exact ceiling: do not erase a real floating-point interval overlap."""
    return int(math.ceil(float(value)))


def _color(trips, ids, duration, field):
    available = {identifier: -math.inf for identifier in sorted(ids)}
    for trip in sorted(trips, key=lambda t: (t["start"], t["id"])):
        free = [identifier for identifier in available if available[identifier] <= trip["start"]]
        if not free:
            raise RuntimeError(f"Cannot color cumulative {field} allocation")
        chosen = free[0]
        trip[field] = chosen
        available[chosen] = trip["start"]+duration(trip)


def refine_schedule(result, data, model, time_limit_s=30.0, seed=20260924):
    from ortools.sat.python import cp_model

    begun = time.perf_counter()
    metadata = {"method": "CP-SAT fixed-structure timing and interval coloring", "time_grid_s": 1,
                "time_limit_s": time_limit_s, "seed": seed, "workers": 1,
                "fixed": ["box batches", "route order", "transport type"],
                "scope": "Bounds apply only to this fixed, conservatively rounded timing model; no full-problem optimum claim."}
    if not result["trips"] or time_limit_s <= 0:
        return None, {**metadata, "status": "SKIPPED"}
    problem = cp_model.CpModel()
    horizon = _ceil(max(trip["return"] for trip in result["trips"])+600)
    starts, plans, ends = {}, {}, []
    intervals = {tid: {"drone": [], "battery": []} for tid in sorted(data["types"])}
    penalties = []
    weights = {bid: Fraction(str(box["weight"])).limit_denominator(10000) for bid, box in model.boxes.items()}
    scale = math.lcm(*(w.denominator for w in weights.values()))
    for old in result["trips"]:
        tid, spec = old["id"], data["types"][old["type"]]
        plan = model.plan(old["type"], old["box_ids"], old["route"])
        if plan is None:
            return None, {**metadata, "status": "INVALID_STRUCTURE"}
        plans[tid] = plan
        duration = _ceil(plan["duration"])
        start = problem.new_int_var(0, horizon-duration, f"start_{tid}")
        starts[tid] = start
        ends.append(start+duration)
        intervals[old["type"]]["drone"].append(problem.new_fixed_size_interval_var(start, duration, f"drone_{tid}"))
        occupied = _ceil(plan["duration"]+charge_time(plan["soc"], spec["charge_full"]))
        intervals[old["type"]]["battery"].append(problem.new_fixed_size_interval_var(start, occupied, f"battery_{tid}"))
        problem.add_hint(start, min(horizon-duration, max(0, _ceil(old["start"]))))
        for bid, offset in plan["deliveries"].items():
            box = model.boxes[bid]
            if math.isfinite(box["deadline"]):
                problem.add(start+_ceil(offset) <= math.floor(box["deadline"]))
            late = problem.new_int_var(0, horizon, f"late_{bid}")
            problem.add_max_equality(late, [0, start+_ceil(offset)-math.floor(box["due"])])
            penalties.append(int(weights[bid]*scale)*late)
    for tid, resources in intervals.items():
        spec = data["types"][tid]
        for resource, capacity in (("drone", len(spec["drones"])), ("battery", len(spec["battery_ids"]))):
            if resources[resource]:
                problem.add_cumulative(resources[resource], [1]*len(resources[resource]), capacity)
    tardiness = sum(penalties)
    makespan = problem.new_int_var(0, horizon, "makespan")
    problem.add_max_equality(makespan, ends)

    def configured(limit):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(limit)
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = int(seed)
        return solver

    problem.minimize(tardiness)
    first = configured(time_limit_s/2)
    first_status = first.solve(problem)
    first_info = {"status": first.status_name(first_status), "wall_time_s": first.wall_time,
                  "objective_bound": first.best_objective_bound}
    metadata["phase_1_weighted_tardiness"] = first_info
    metadata.update(horizon_s=horizon, weight_integer_scale=scale)
    if first_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        metadata.update(status="NO_FEASIBLE_REFINEMENT", runtime_s=time.perf_counter()-begun)
        return None, metadata
    first_info["objective"] = first.objective_value
    chosen_starts = {tid: first.value(start) for tid, start in starts.items()}
    problem.add(tardiness <= int(round(first.objective_value)))
    problem.minimize(makespan)
    problem.clear_hints()
    for tid, start in starts.items():
        problem.add_hint(start, chosen_starts[tid])
    second = configured(time_limit_s/2)
    second_status = second.solve(problem)
    second_info = {"status": second.status_name(second_status), "wall_time_s": second.wall_time,
                   "objective_bound": second.best_objective_bound}
    metadata["phase_2_makespan"] = second_info
    if second_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        second_info["objective"] = second.objective_value
        chosen_starts = {tid: second.value(start) for tid, start in starts.items()}
    refined = deepcopy(result)
    refined.pop("validation", None)
    refined["trips"] = []
    for old in result["trips"]:
        plan, start = plans[old["id"]], float(chosen_starts[old["id"]])
        trip = {key: value for key, value in plan.items() if key not in ("duration", "segments", "deliveries")}
        trip.update(id=old["id"], start=start, takeoff=start+plan["takeoff"],
                    segments=shift_segments(plan["segments"], start),
                    deliveries={bid: start+offset for bid, offset in plan["deliveries"].items()})
        trip["return"] = start+plan["duration"]
        refined["trips"].append(trip)
    for tid, spec in data["types"].items():
        tasks = [trip for trip in refined["trips"] if trip["type"] == tid]
        _color(tasks, spec["drones"], lambda t: _ceil(t["return"]-t["start"]), "drone")
        _color(tasks, spec["battery_ids"], lambda t: _ceil(t["return"]-t["start"]
                                                       + charge_time(t["soc"], spec["charge_full"])), "battery")
    refined["metrics"] = metrics(refined, model)
    accepted = objective(refined) < objective(result)
    metadata.update(status="IMPROVED" if accepted else "NO_IMPROVEMENT",
                    runtime_s=time.perf_counter()-begun, previous_metrics=result["metrics"],
                    candidate_metrics=refined["metrics"])
    refined["refinement"] = metadata
    return (refined if accepted else None), metadata
