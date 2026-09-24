"""Exact Q4 partition enumeration over the frozen, verified Q3 schedule.

Transport batches, types, routes and every timestamp remain fixed.  A relay
sortie is an indivisible task and is never copied across groups.  Homogeneous
resource identities may be reassigned, so an interval graph's maximum overlap
gives the minimum resource count.  All intervals are half open and retain the
original floating-point times without rounding or overlap tolerances.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import heapq
import math
from typing import Iterable, Iterator, Sequence


RESOURCE_KEYS = (
    "drone_A", "drone_B", "drone_C", "battery_A", "battery_B", "battery_C",
    "relay", "module",
)


def peak_intervals(intervals: Iterable[Sequence[float]]) -> int:
    """Return exact maximum overlap of finite half-open ``[start, end)`` pairs.

    End events precede start events at the same timestamp.  Empty intervals
    occupy no resource.  Invalid or non-finite intervals raise ValueError.
    """
    events = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("An interval must contain exactly start and end")
        start, end = map(float, interval)
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f"Invalid interval: {interval!r}")
        if end > start:
            events.extend(((start, 1), (end, -1)))
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return peak


def iter_partitions(n: int, k: int) -> Iterator[tuple[int, ...]]:
    """Yield all unlabeled nonempty k-way partitions as restricted-growth codes.

    Code position is the atom index; values are canonical group indices
    ``0, ..., k-1``.  Codes are yielded in lexicographic order, with no label
    permutations.  ``iter_partitions(0, 0)`` yields the empty partition.
    """
    if not isinstance(n, int) or not isinstance(k, int) or n < 0 or k < 0:
        raise ValueError("n and k must be nonnegative integers")
    if n == 0:
        if k == 0:
            yield ()
        return
    if k == 0 or k > n:
        return
    code = [0] * n

    def visit(position: int, largest: int):
        if k - (largest + 1) > n - position:
            return
        if position == n:
            if largest + 1 == k:
                yield tuple(code)
            return
        for group in range(min(largest + 1, k - 1) + 1):
            code[position] = group
            yield from visit(position + 1, max(largest, group))

    yield from visit(1, 0)


def charge_time(soc: float, full_time: float) -> float:
    """Appendix-2 full recharge time; power resources may charge in parallel."""
    soc, full_time = float(soc), float(full_time)
    if not math.isfinite(soc) or not 0 <= soc <= 1:
        raise ValueError(f"SOC outside [0,1]: {soc}")
    if not math.isfinite(full_time) or full_time < 0:
        raise ValueError(f"Invalid full charge time: {full_time}")
    return full_time * (
        0.65 * max(0.0, 0.9 - soc) / 0.9
        + 0.35 * min(0.1, 1.0 - soc) / 0.1
    )


def _cv(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = math.fsum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _inventory(data: dict) -> dict[str, int]:
    result = {}
    for kind in ("A", "B", "C"):
        spec = data["types"][kind]
        result[f"drone_{kind}"] = len(spec["drones"])
        result[f"battery_{kind}"] = len(spec["battery_ids"])
    result["relay"] = len(data["relay"]["drones"])
    result["module"] = len(data["relay"]["component_ids"])
    return {key: result[key] for key in RESOURCE_KEYS}


def _fixed_tasks(q3: dict, data: dict) -> tuple[list[dict], list[dict], list[str]]:
    nodes = sorted(identifier for identifier, node in data["nodes"].items()
                   if identifier != "O01" and not node.get("is_depot", False))
    known_nodes = set(nodes)
    boxes = {box["id"]: box for box in data["boxes"]}
    transports, relays = q3["trips"], q3["relay_trips"]
    by_transport = {trip["id"]: trip for trip in transports}
    by_relay = {trip["id"]: trip for trip in relays}
    ids = [trip["id"] for trip in transports + relays]
    if len(set(ids)) != len(ids):
        raise ValueError("Q3 task IDs must be globally unique")
    assigned = Counter(box_id for trip in transports for box_id in trip["box_ids"])
    if set(assigned) != set(boxes) or any(count != 1 for count in assigned.values()):
        raise ValueError("Frozen Q3 must deliver every input box exactly once")

    # Retain explicit sortie associations and communication-record references.
    # Neither actual direct-link preference nor grouping may erase a frozen
    # relay service relation.
    supported_by = defaultdict(set)
    for trip in transports:
        rid = trip.get("relay_trip_id")
        if rid is not None:
            if rid not in by_relay:
                raise ValueError(f"Unknown relay {rid} for transport {trip['id']}")
            supported_by[rid].add(trip["id"])
    for block in q3.get("communication", []):
        if block.get("mode") == "relay":
            tid, rid = block.get("trip_id"), block.get("relay_trip_id")
            if tid not in by_transport or rid not in by_relay:
                raise ValueError(f"Invalid frozen communication association: {block}")
            supported_by[rid].add(tid)

    def times(trip):
        start, end = float(trip["start"]), float(trip["return"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid frozen task time: {trip['id']}")
        return start, end

    tasks, edges, transport_nodes = [], [], {}
    for trip in sorted(transports, key=lambda item: item["id"]):
        kind = trip["type"]
        if kind not in ("A", "B", "C"):
            raise ValueError(f"Unknown transport type: {kind}")
        route = list(trip["route"])
        destinations = sorted(set(route))
        if not destinations or not set(destinations) <= known_nodes:
            raise ValueError(f"Invalid transport destination: {trip['id']}")
        if set(destinations) != {boxes[box_id]["node"] for box_id in trip["box_ids"]}:
            raise ValueError(f"Q3 route and box destinations disagree: {trip['id']}")
        transport_nodes[trip["id"]] = destinations
        start, end = times(trip)
        recharge = charge_time(trip["soc"], data["types"][kind]["charge_full"])
        tasks.append({
            "id": trip["id"], "kind": "transport", "type": kind,
            "nodes": destinations, "box_ids": list(trip["box_ids"]),
            "start": start, "return": end, "work_s": end - start,
            "intervals": {f"drone_{kind}": [start, end],
                          f"battery_{kind}": [start, end + recharge]},
        })
        edges.append({"kind": "transport", "task_id": trip["id"], "nodes": destinations})

    relay_spec = data["relay"]
    for trip in sorted(relays, key=lambda item: item["id"]):
        destinations = sorted({node for tid in supported_by[trip["id"]]
                               for node in transport_nodes[tid]})
        if not destinations:
            raise ValueError(f"Relay task has no frozen service association: {trip['id']}")
        start, end = times(trip)
        recharge = charge_time(trip["soc"], relay_spec["charge_full"])
        tasks.append({
            "id": trip["id"], "kind": "relay", "type": "R",
            "nodes": destinations, "box_ids": [],
            "start": start, "return": end, "work_s": end - start,
            "intervals": {"relay": [start, end + float(relay_spec["turnaround"])],
                          "module": [start, end + recharge]},
        })
        edges.append({"kind": "relay", "task_id": trip["id"], "nodes": destinations})
    return tasks, edges, nodes


def _atoms(nodes: list[str], edges: list[dict]) -> list[dict]:
    parents = {node: node for node in nodes}

    def root(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for edge in edges:
        first = edge["nodes"][0]
        for node in edge["nodes"][1:]:
            a, b = sorted((root(first), root(node)))
            parents[b] = a
    connected = defaultdict(list)
    for node in nodes:
        connected[root(node)].append(node)
    ordered = sorted((sorted(component) for component in connected.values()), key=tuple)
    return [{"id": f"C{number:02d}", "nodes": component}
            for number, component in enumerate(ordered, 1)]


def _requirements(tasks: Sequence[dict]) -> dict[str, int]:
    return {key: peak_intervals([task["intervals"][key] for task in tasks
                                 if key in task["intervals"]]) for key in RESOURCE_KEYS}


def _evaluate(code: tuple[int, ...], identifier: str, atoms: list[dict], tasks: list[dict],
              boxes: dict, inventory: dict, baseline: dict) -> dict:
    k = max(code) + 1
    groups = []
    for group_index in range(k):
        included = [atom for atom, label in zip(atoms, code) if label == group_index]
        nodes = sorted(node for atom in included for node in atom["nodes"])
        node_set = set(nodes)
        own_tasks = [task for task in tasks if set(task["nodes"]) <= node_set]
        box_ids = [bid for task in own_tasks for bid in task["box_ids"]]
        groups.append({
            "id": f"G{group_index + 1}", "atom_ids": [atom["id"] for atom in included],
            "nodes": nodes, "task_ids": sorted(task["id"] for task in own_tasks),
            "resources": _requirements(own_tasks),
            "work_s": math.fsum(task["work_s"] for task in own_tasks),
            "box_count": len(box_ids), "mass_kg": math.fsum(boxes[bid]["mass"] for bid in box_ids),
        })
    if sum(len(group["task_ids"]) for group in groups) != len(tasks):
        raise RuntimeError("A frozen task was lost or copied during grouping")
    totals = {key: sum(group["resources"][key] for group in groups) for key in RESOURCE_KEYS}
    shortages = {key: max(0, totals[key] - inventory[key]) for key in RESOURCE_KEYS}
    surplus = {key: max(0, inventory[key] - totals[key]) for key in RESOURCE_KEYS}
    added = {key: totals[key] - baseline["resources"][key] for key in RESOURCE_KEYS}
    if any(value < 0 for value in added.values()):
        raise RuntimeError("Independent groups cannot require less than pooled peak resources")
    shortage_total, total_resources = sum(shortages.values()), sum(totals.values())
    work_cv, box_cv = _cv([group["work_s"] for group in groups]), _cv([group["box_count"] for group in groups])
    return {
        "id": identifier, "k": k, "groups": groups, "totals": totals,
        "shortages": shortages, "surplus": surplus, "added_vs_pool": added,
        "total_resources": total_resources, "shortage_total": shortage_total,
        "work_cv": work_cv, "box_cv": box_cv,
        "score": [shortage_total, total_resources, work_cv],
    }


def _pareto_ids(candidates: Sequence[dict]) -> list[str]:
    def dominates(left, right):
        return all(a <= b for a, b in zip(left["score"], right["score"])) and any(
            a < b for a, b in zip(left["score"], right["score"]))
    return [candidate["id"] for candidate in candidates
            if not any(dominates(other, candidate) for other in candidates)]


def _allocate(candidate: dict, by_task: dict) -> None:
    """Give the selected partition a constructive minimum-resource certificate."""
    for group in candidate["groups"]:
        allocations = []
        for key in RESOURCE_KEYS:
            occupied, available = [], []
            count = 0
            intervals = sorted((by_task[tid]["intervals"][key][0],
                                by_task[tid]["intervals"][key][1], tid)
                               for tid in group["task_ids"] if key in by_task[tid]["intervals"])
            for start, end, tid in intervals:
                if end <= start:
                    continue
                while occupied and occupied[0][0] <= start:
                    _, resource = heapq.heappop(occupied)
                    heapq.heappush(available, resource)
                if available:
                    resource = heapq.heappop(available)
                else:
                    count += 1
                    resource = count
                heapq.heappush(occupied, (end, resource))
                allocations.append({
                    "task_id": tid, "resource_key": key,
                    "resource_id": f"{candidate['id']}/{group['id']}/{key}-{resource:02d}",
                    "start": start, "end": end,
                })
            if count != group["resources"][key]:
                raise RuntimeError(f"Coloring does not attain peak requirement: {group['id']}/{key}")
        group["allocations"] = allocations


def solve(q3: dict, data: dict) -> dict:
    """Enumerate and evaluate every 2- and 3-way partition of frozen-task atoms.

    The first objective is total inventory shortage, then total resource count,
    then the population coefficient of variation of group workload.  Workload
    is the sum of transport and relay preparation-to-return durations.  Charging
    and relay turnaround affect resource occupation, not this workload metric.
    No input object is changed and no file is written.
    """
    tasks, edges, nodes = _fixed_tasks(q3, data)
    atoms, inventory = _atoms(nodes, edges), _inventory(data)
    baseline_resources = _requirements(tasks)
    baseline = {"resources": baseline_resources, "total_resources": sum(baseline_resources.values())}
    boxes = {box["id"]: box for box in data["boxes"]}
    by_task = {task["id"]: task for task in tasks}
    partitions = {}
    for k in (2, 3):
        candidates = [_evaluate(code, f"K{k}-{number:03d}", atoms, tasks, boxes, inventory, baseline)
                      for number, code in enumerate(iter_partitions(len(atoms), k), 1)]
        selected = balanced = None
        if candidates:
            selected = deepcopy(min(candidates, key=lambda item: (*item["score"], item["id"])))
            balanced = deepcopy(min(candidates, key=lambda item: (
                item["work_cv"], item["shortage_total"], item["total_resources"], item["id"])))
            _allocate(selected, by_task)
            _allocate(balanced, by_task)
        partitions[str(k)] = {
            "enumerated": len(candidates),
            "zero_gap_count": sum(candidate["shortage_total"] == 0 for candidate in candidates),
            "all": candidates, "selected": selected, "balanced": balanced,
            "pareto_ids": _pareto_ids(candidates),
        }
    return {
        "resource_keys": list(RESOURCE_KEYS), "inventory": inventory,
        "atoms": atoms, "dependency_edges": edges, "tasks": tasks,
        "baseline": baseline, "partitions": partitions,
        "workload_definition": "Sum of all transport and relay preparation-to-return durations; excludes recharge and post-return turnaround.",
        "scope": "Exhaustive unlabeled partitions of frozen Q3 task atoms, without relay duplication; same-type physical resource IDs may be reassigned within each group.",
        "interval_semantics": "Half open [start,end); original floating-point times retained; end events precede starts at equal times.",
    }
