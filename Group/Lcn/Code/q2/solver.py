"""Bounded deterministic Q2 batching, routing and resource scheduling search.

The shared Q3 ``Model`` supplies only transport geometry and energy primitives.
No Q3 result, relay site, radio calculation or communication constraint is used.
This is a reproducible feasible-schedule search, not a global optimality proof.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import itertools
import math
import random
import time

import numpy as np

from q3.solver import Model, charge_time, shift_segments


class TransportModel(Model):
    """Memoize immutable transport plans without constructing relay geometry."""

    def __init__(self, data, terrain):
        super().__init__(data, terrain, [])
        self.plan_cache = {}

    def plan(self, type_id, box_ids, route):
        destinations = {self.boxes[bid]["node"] for bid in box_ids}
        effective = tuple(node for node in route if node in destinations)
        key = (type_id, tuple(sorted(box_ids)), effective)
        if key not in self.plan_cache:
            self.plan_cache[key] = super().plan(type_id, key[1], effective)
        return self.plan_cache[key]


def objective(result):
    m = result["metrics"]
    return (m["weighted_tardiness"], m["makespan_s"],
            m["total_energy_kwh"], m["transport_sorties"])


def metrics(result, model):
    trips = result["trips"]
    deliveries = {bid: t for trip in trips for bid, t in trip["deliveries"].items()}
    late = {bid: max(0.0, t-model.boxes[bid]["due"]) for bid, t in deliveries.items()}
    return {
        "boxes_delivered": len(deliveries), "transport_sorties": len(trips),
        "total_energy_kwh": sum(trip["energy"] for trip in trips),
        "transport_energy_kwh": sum(trip["energy"] for trip in trips),
        "makespan_s": max((trip["return"] for trip in trips), default=0.0),
        "last_delivery_s": max(deliveries.values(), default=0.0),
        "weighted_tardiness": sum(model.boxes[bid]["weight"]*t for bid, t in late.items()),
        "late_boxes": sum(t > 1e-6 for t in late.values()),
        "max_tardiness_s": max(late.values(), default=0.0),
        "hard_deadline_violations": sum(t > model.boxes[bid]["deadline"]+1e-6
                                        for bid, t in deliveries.items()),
        "min_transport_soc": min((trip["soc"] for trip in trips), default=1.0),
        "multi_stop_sorties": sum(len(trip["route"]) > 1 for trip in trips),
        "sorties_by_type": {tid: sum(trip["type"] == tid for trip in trips)
                            for tid in sorted(model.data["types"])},
    }


def _initial_state(data):
    return {
        "trips": [],
        "drone_free": {drone: 0.0 for t in data["types"].values() for drone in t["drones"]},
        "battery_free": {battery: 0.0 for t in data["types"].values() for battery in t["battery_ids"]},
    }


def _place(model, state, plan, settings):
    spec = model.data["types"][plan["type"]]
    drone = min(spec["drones"], key=lambda d: (state["drone_free"][d], d))
    battery = min(spec["battery_ids"], key=lambda b: (state["battery_free"][b], b))
    start = max(state["drone_free"][drone], state["battery_free"][battery])
    if any(start+t > model.boxes[bid]["deadline"]+1e-7 for bid, t in plan["deliveries"].items()):
        return None
    tardiness = sum(model.boxes[bid]["weight"]*max(0.0, start+t-model.boxes[bid]["due"])
                    for bid, t in plan["deliveries"].items())
    hard = sum(math.isfinite(model.boxes[bid]["deadline"]) for bid in plan["box_ids"])
    count = len(plan["box_ids"])+settings["hard_bonus"]*hard
    score = (start+plan["duration"]+settings["wait_weight"]*start
             + settings["energy_weight"]*plan["energy"]
             + settings["late_weight"]*tardiness) / count**settings["packing_power"]
    return {"plan": plan, "start": start, "drone": drone, "battery": battery, "score": score}


def _commit(model, state, placement):
    plan, start = placement["plan"], placement["start"]
    trip = {key: value for key, value in plan.items() if key not in ("duration", "segments", "deliveries")}
    trip.update(id=f"T{len(state['trips'])+1:03d}", start=start,
                takeoff=start+plan["takeoff"], drone=placement["drone"], battery=placement["battery"],
                segments=shift_segments(plan["segments"], start),
                deliveries={bid: start+t for bid, t in plan["deliveries"].items()})
    trip["return"] = start+plan["duration"]
    state["trips"].append(trip)
    state["drone_free"][trip["drone"]] = trip["return"]
    state["battery_free"][trip["battery"]] = trip["return"]+charge_time(
        trip["soc"], model.data["types"][trip["type"]]["charge_full"])
    return trip


def _candidates(model, remaining, seed, settings, rng):
    bynode = defaultdict(list)
    for bid in sorted(remaining):
        bynode[model.boxes[bid]["node"]].append(bid)
    seednode = model.boxes[seed]["node"]
    nearby = sorted((node for node in bynode if node != seednode),
                    key=lambda node: (float(np.linalg.norm(model.nodes[node][:2]-model.nodes[seednode][:2])), node))
    nearby = nearby[:settings["neighbors"]]
    routes = [(seednode,)]
    for size in range(2, settings["max_stops"]+1):
        for others in itertools.combinations(nearby, size-1):
            routes.extend(itertools.permutations((seednode, *others)))
    noise = {bid: rng.random() for bid in sorted(remaining)}
    unique = {}
    for route in routes:
        pool = [bid for node in route for bid in bynode[node] if bid != seed]
        if settings["hard_first"]:
            pool.sort(key=lambda bid: (model.boxes[bid]["deadline"], model.boxes[bid]["due"],
                                       -model.boxes[bid]["weight"]+settings["noise"]*noise[bid], bid))
        else:
            pool.sort(key=lambda bid: (min(model.boxes[bid]["deadline"], model.boxes[bid]["due"]),
                                       -model.boxes[bid]["weight"]+settings["noise"]*noise[bid], bid))
        for tid in sorted(model.data["types"]):
            batch = [seed]
            first = model.plan(tid, batch, route)
            if first is None:
                continue
            unique[(tid, tuple(first["box_ids"]), tuple(first["route"]))] = first
            for bid in pool:
                trial = model.plan(tid, batch+[bid], route)
                if trial is not None:
                    batch.append(bid)
                    unique[(tid, tuple(trial["box_ids"]), tuple(trial["route"]))] = trial
    ranked = sorted(unique.values(), key=lambda p: (
        (p["duration"]+settings["energy_weight"]*p["energy"])/len(p["box_ids"])**settings["packing_power"],
        p["type"], tuple(p["route"]), tuple(p["box_ids"])))
    # Retain all single-stop alternatives to protect urgent partial batches.
    return ranked[:settings["plan_limit"]]+[p for p in ranked[settings["plan_limit"]:] if len(p["route"]) == 1]


def construct(model, settings, seed):
    rng = random.Random(seed)
    remaining = set(model.boxes)
    state = _initial_state(model.data)
    ranks = {bid: rng.random() for bid in sorted(remaining)}
    nominal = {}
    for bid in sorted(remaining):
        choices = [plan["deliveries"][bid] for tid in sorted(model.data["types"])
                   if (plan := model.plan(tid, [bid], [model.boxes[bid]["node"]])) is not None]
        if not choices:
            return None, {"reason": "Box has no feasible solo transport plan", "box": bid}
        nominal[bid] = min(choices)

    def priority(bid):
        box = model.boxes[bid]
        if settings["priority"] == "hard":
            return (box["deadline"], box["due"], -box["weight"]+settings["noise"]*ranks[bid], bid)
        target = min(box["deadline"], box["due"])
        slack = target-nominal[bid] if settings["priority"] == "slack" else target
        return (slack, -nominal[bid]+settings["noise"]*ranks[bid], -box["weight"], bid)

    while remaining:
        seedbox = min(remaining, key=priority)
        proposals = []
        for plan in _candidates(model, remaining, seedbox, settings, rng):
            placed = _place(model, state, plan, settings)
            if placed is not None:
                proposals.append(placed)
        if not proposals:
            return None, {"reason": "No feasible placement in bounded candidate family", "seed_box": seedbox,
                          "scheduled_boxes": len(model.boxes)-len(remaining), "transport_sorties": len(state["trips"])}
        proposals.sort(key=lambda p: (p["score"], p["start"], p["plan"]["type"],
                                     tuple(p["plan"]["route"]), tuple(p["plan"]["box_ids"])))
        chosen = proposals[0]
        if settings["lookahead"]:
            # Necessary individual feasibility only, not a claim of joint feasibility.
            for proposal in proposals[:settings["lookahead_limit"]]:
                trial = deepcopy(state)
                _commit(model, trial, proposal)
                future = remaining-set(proposal["plan"]["box_ids"])
                hard = sorted((bid for bid in future if math.isfinite(model.boxes[bid]["deadline"])),
                              key=lambda bid: (model.boxes[bid]["deadline"], bid))
                viable = True
                for bid in hard:
                    node = model.boxes[bid]["node"]
                    if not any((solo := model.plan(tid, [bid], [node])) is not None
                               and _place(model, trial, solo, settings) is not None
                               for tid in sorted(model.data["types"])):
                        viable = False
                        break
                if viable:
                    chosen = proposal
                    break
        trip = _commit(model, state, chosen)
        remaining.difference_update(trip["box_ids"])
    result = {"trips": state["trips"], "config": settings, "seed": seed}
    result["metrics"] = metrics(result, model)
    return result, None


def _settings(i, max_stops, neighbors):
    return {"max_stops": min(max_stops, 1+i % max_stops), "neighbors": neighbors,
            "plan_limit": 32, "packing_power": [0.75, 1.0, 1.25, 1.5][(i//3) % 4],
            "energy_weight": [20.0, 80.0, 160.0][i % 3], "wait_weight": [0.0, 0.5][(i//6) % 2],
            "late_weight": [2.0, 8.0, 20.0][(i//4) % 3], "hard_bonus": [0.0, 0.5][(i//12) % 2],
            "noise": 0.0 if i < 12 else [3.0, 10.0, 30.0][i % 3],
            "priority": ["target", "slack", "hard"][i//4 % 3], "hard_first": i % 2 == 1,
            "lookahead": True, "lookahead_limit": 24}


def solve(data, terrain, config=None):
    config = config or {}
    starts, seed = int(config.get("starts", 48)), int(config.get("seed", 20260924))
    max_stops, neighbors = int(config.get("max_stops", 3)), int(config.get("neighbors", 4))
    refinement_budget = float(config.get("refine_seconds", 120.0))
    if starts < 1 or not 1 <= max_stops <= 4 or not 0 <= neighbors <= 6 or refinement_budget < 0:
        raise ValueError("Require starts >= 1, 1 <= max_stops <= 4, 0 <= neighbors <= 6 and refinement >= 0")
    model, history, solutions = TransportModel(data, terrain), [], []
    for i in range(starts):
        settings = _settings(i, max_stops, neighbors)
        begun = time.perf_counter()
        candidate, failure = construct(model, settings, seed+i)
        row = {"label": f"S{i+1:02d}", "seed": seed+i, "config": settings,
               "feasible": candidate is not None, "runtime_s": time.perf_counter()-begun}
        if candidate is None:
            row.update(failure)
            print(f"Q2 search {i+1}/{starts}: rejected ({failure['reason']})", flush=True)
        else:
            row.update(candidate["metrics"])
            solutions.append(candidate)
            print(f"Q2 search {i+1}/{starts}: {row['transport_sorties']} sorties; "
                  f"delay {row['weighted_tardiness']:.3f}; finish {row['makespan_s']/60:.2f} min", flush=True)
        history.append(row)
    if not solutions:
        raise RuntimeError(f"No complete feasible Q2 schedule in bounded search: {history}")
    solutions.sort(key=objective)
    refinement_history = []
    if refinement_budget > 0:
        from .refine import refine_schedule
        templates = []
        for candidate in [*solutions[:2], min(solutions, key=lambda r: r["metrics"]["total_energy_kwh"]),
                          min(solutions, key=lambda r: r["metrics"]["transport_sorties"])]:
            signature = tuple((t["type"], tuple(t["box_ids"]), tuple(t["route"])) for t in candidate["trips"])
            if not any(signature == sig for sig, _ in templates):
                templates.append((signature, candidate))
        for j, (_, candidate) in enumerate(templates):
            print(f"Q2 CP-SAT timing refinement {j+1}/{len(templates)}", flush=True)
            refined, metadata = refine_schedule(candidate, data, model,
                                                time_limit_s=refinement_budget/len(templates), seed=seed+j)
            refinement_history.append(metadata)
            if refined is not None:
                refined["metrics"] = metrics(refined, model)
                solutions.append(refined)
                history.append({"label": f"CP{j+1}", "feasible": True, **refined["metrics"], "refinement": metadata})
        solutions.sort(key=objective)
    from .verify import verify
    for candidate in solutions:
        audit = verify(candidate, data, terrain)
        if audit["passed"]:
            candidate.update(validation=audit, search_history=history, refinement_attempts=refinement_history,
                             terrain=terrain.metadata(), search_bounds={
                "starts": starts, "seed": seed, "max_stops": max_stops, "neighbors": neighbors,
                "greedy_batch_prefixes": True, "retained_ranked_plans": 32,
                "all_single_stop_candidates_retained": True, "refinement_total_seconds": refinement_budget,
                "cached_plan_count": len(model.plan_cache), "global_optimality_proven": False,
                "objective_order": ["weighted_tardiness", "makespan_s", "total_energy_kwh", "transport_sorties"],
                "communication_constraints": False,
            })
            return candidate
        print(f"Q2 independent audit rejected candidate: {audit['errors'][:3]}", flush=True)
    raise RuntimeError("All complete Q2 schedules failed independent verification")
