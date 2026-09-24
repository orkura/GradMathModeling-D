"""Independent, read-only audit of the frozen-task Q4 partition enumeration.

This module does not import q4.model. It checks overlap by counting active
intervals at every start, connected components by graph traversal, and complete
partitions by generating all labelled assignments then removing permutations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
import math


RESOURCE_KEYS = ["drone_A", "drone_B", "drone_C", "battery_A", "battery_B",
                 "battery_C", "relay", "module"]


def _charge(soc, full):
    if not 0 <= soc <= 1:
        raise ValueError("Invalid source SOC")
    return full * (0.65 * max(0.0, 0.9-soc) / 0.9
                   + 0.35 * min(0.1, 1.0-soc) / 0.1)


def _peak_at_starts(intervals):
    """Strict half-open comparisons; no event sweep and no time rounding."""
    return max((sum(start <= time < end for start, end in intervals)
                for time, _ in intervals), default=0)


def _cv(values):
    if not values or math.fsum(values) == 0:
        return 0.0
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((x-mean)**2 for x in values) / len(values)) / mean


def _canonical(groups):
    return tuple(sorted(tuple(sorted(nodes)) for nodes in groups))


def _all_partitions(atoms, k):
    """Independent exhaustive K^m assignments, canonicalized by node sets."""
    found = set()
    for labels in product(range(k), repeat=len(atoms)):
        if len(set(labels)) != k:
            continue
        groups = [[] for _ in range(k)]
        for atom, label in zip(atoms, labels):
            groups[label].extend(atom)
        found.add(_canonical(groups))
    return found


class _Audit:
    def __init__(self):
        self.errors = []
        self.count = 0
        self.error_count = 0
        self.checked = Counter()
        self.failed = Counter()

    def check(self, condition, name, subject, detail):
        self.count += 1
        self.checked[name] += 1
        if bool(condition):
            return True
        self.error_count += 1
        self.failed[name] += 1
        if len(self.errors) < 150:
            self.errors.append({"check": name, "subject": subject, "detail": detail})
        return False

    def close(self, value, expected, name, subject):
        return self.check(isinstance(value, (float, int)) and math.isfinite(value)
                          and abs(value-expected) <= 1e-8,
                          name, subject, f"actual={value}; expected={expected}")


def _source(q3, data):
    boxes = {b["id"]: b for b in data["boxes"]}
    nodes = sorted(n for n, value in data["nodes"].items()
                   if n != "O01" and not value.get("is_depot", False))
    trip_nodes = {t["id"]: sorted({boxes[b]["node"] for b in t["box_ids"]})
                  for t in q3["trips"]}
    relay_support = defaultdict(set)
    for trip in q3["trips"]:
        if trip.get("relay_trip_id"):
            relay_support[trip["relay_trip_id"]].add(trip["id"])
    for record in q3.get("communication", []):
        if record.get("mode") == "relay":
            relay_support[record["relay_trip_id"]].add(record["trip_id"])
    tasks = {}
    for trip in q3["trips"]:
        tid, kind = trip["id"], trip["type"]
        start, end = trip["start"], trip["return"]
        tasks[tid] = dict(id=tid, kind="transport", type=kind, nodes=trip_nodes[tid],
                          box_ids=list(trip["box_ids"]), start=start, return_=end,
                          work_s=end-start, intervals={f"drone_{kind}": [start, end],
                          f"battery_{kind}": [start, end+_charge(trip["soc"], data["types"][kind]["charge_full"])]})
    for trip in q3["relay_trips"]:
        tid, start, end = trip["id"], trip["start"], trip["return"]
        supported_nodes = sorted({n for t in relay_support[tid] for n in trip_nodes[t]})
        tasks[tid] = dict(id=tid, kind="relay", type="R", nodes=supported_nodes,
                          box_ids=[], start=start, return_=end, work_s=end-start,
                          intervals={"relay": [start, end+data["relay"]["turnaround"]],
                          "module": [start, end+_charge(trip["soc"], data["relay"]["charge_full"])]})
    for task in tasks.values():
        task["return"] = task.pop("return_")
    adjacency = {node: set() for node in nodes}
    for task in tasks.values():
        for first in task["nodes"]:
            adjacency[first].update(task["nodes"])
    remaining, components = set(nodes), []
    while remaining:
        stack, component = [min(remaining)], set()
        while stack:
            node = stack.pop()
            if node not in component:
                component.add(node)
                stack.extend(adjacency[node]-component)
        remaining.difference_update(component)
        components.append(sorted(component))
    inventory = {}
    for kind in "ABC":
        inventory[f"drone_{kind}"] = len(data["types"][kind]["drones"])
        inventory[f"battery_{kind}"] = len(data["types"][kind]["battery_ids"])
    inventory["relay"] = len(data["relay"]["drones"])
    inventory["module"] = len(data["relay"]["component_ids"])
    return boxes, nodes, tasks, sorted(components), inventory


def _requirements(tasks):
    return {key: _peak_at_starts([task["intervals"][key] for task in tasks
                                  if key in task["intervals"]]) for key in RESOURCE_KEYS}


def _evaluate(groups, tasks, boxes, inventory, baseline):
    group_truth = {}
    for group in groups:
        members = set(group)
        owned = [task for task in tasks.values() if set(task["nodes"]) <= members]
        boxids = [b for task in owned for b in task["box_ids"]]
        group_truth[tuple(group)] = {
            "task_ids": sorted(task["id"] for task in owned),
            "resources": _requirements(owned),
            "work_s": math.fsum(t["work_s"] for t in owned),
            "box_count": len(boxids), "mass_kg": math.fsum(boxes[b]["mass"] for b in boxids),
        }
    totals = {key: sum(g["resources"][key] for g in group_truth.values()) for key in RESOURCE_KEYS}
    shortage = {key: max(0, totals[key]-inventory[key]) for key in RESOURCE_KEYS}
    surplus = {key: max(0, inventory[key]-totals[key]) for key in RESOURCE_KEYS}
    added = {key: totals[key]-baseline[key] for key in RESOURCE_KEYS}
    work_cv = _cv([g["work_s"] for g in group_truth.values()])
    box_cv = _cv([g["box_count"] for g in group_truth.values()])
    return {"groups": group_truth, "totals": totals, "shortages": shortage,
            "surplus": surplus, "added_vs_pool": added, "total_resources": sum(totals.values()),
            "shortage_total": sum(shortage.values()), "work_cv": work_cv, "box_cv": box_cv,
            "score": [sum(shortage.values()), sum(totals.values()), work_cv]}


def _allocations(audit, candidate, tasks):
    owners = {}
    for group in candidate["groups"]:
        gid = f"{candidate['id']}/{group['id']}"
        records = group.get("allocations", [])
        expected = Counter((tid, key) for tid in group["task_ids"] if tid in tasks
                           for key in tasks[tid]["intervals"])
        actual = Counter((a["task_id"], a["resource_key"]) for a in records)
        audit.check(actual == expected, "allocation_task_coverage", gid,
                    "Each task must receive exactly one allocation for each required resource type")
        by_entity = defaultdict(list)
        for allocation in records:
            tid, key, resource = allocation["task_id"], allocation["resource_key"], allocation["resource_id"]
            if not audit.check(tid in tasks and key in tasks[tid]["intervals"],
                               "allocation_type", gid, f"Invalid task/resource: {tid}/{key}"):
                continue
            begin, end = tasks[tid]["intervals"][key]
            audit.check(allocation["start"] == begin, "allocation_start", gid,
                        f"{tid}: original start must be inherited exactly")
            audit.close(allocation["end"], end, "allocation_end", f"{gid}/{tid}/{key}")
            owner_key = (key, resource)
            audit.check(owner_key not in owners or owners[owner_key] == group["id"],
                        "resource_not_shared_across_groups", gid, str(resource))
            owners[owner_key] = group["id"]
            by_entity[owner_key].append((begin, end, tid))
        for (key, resource), intervals in by_entity.items():
            for i, (a, b, first) in enumerate(intervals):
                for c, d, second in intervals[i+1:]:
                    audit.check(not (a < d and c < b), "resource_overlap", gid,
                                f"{resource} overlaps {first} and {second}")
        for key in RESOURCE_KEYS:
            used = len({resource for resource_key, resource in by_entity if resource_key == key})
            audit.check(used == group["resources"].get(key), "allocation_attains_peak", gid,
                        f"{key}: allocated={used}; claimed={group['resources'].get(key)}")


def verify(result, q3, data):
    """Recompute all resources, dependency components and exhaustive choices."""
    audit = _Audit()
    boxes, nodes, tasks, atoms, inventory = _source(q3, data)
    audit.check(result.get("resource_keys") == RESOURCE_KEYS, "resource_keys", "result", "Resource order differs")
    audit.check(result.get("inventory") == inventory, "inventory", "result", "Inventory differs from raw data")
    frozen_boxes = Counter(b for t in q3["trips"] for b in t["box_ids"])
    audit.check(frozen_boxes == Counter({b: 1 for b in boxes}), "source_box_coverage", "Q3", "Q3 must deliver every input box once")
    audit.check(len(tasks) == len(q3["trips"])+len(q3["relay_trips"]),
                "source_task_ids", "Q3", "Frozen task IDs must be globally unique")
    for trip in q3["trips"]:
        audit.check(set(trip["route"]) == {boxes[b]["node"] for b in trip["box_ids"]},
                    "source_route_destinations", trip["id"], "Frozen route and box destinations disagree")
    actual_atoms = result.get("atoms", [])
    audit.check(Counter(tuple(sorted(a["nodes"])) for a in actual_atoms) == Counter(map(tuple, atoms)),
                "dependency_components", "atoms", "Atoms differ from independently traversed dependencies")
    atom_lookup = {a["id"]: set(a["nodes"]) for a in actual_atoms}
    audit.check(len(atom_lookup) == len(actual_atoms), "atom_ids", "atoms", "Duplicate atom ID")
    expected_edges = Counter((t["kind"], t["id"], tuple(t["nodes"])) for t in tasks.values())
    actual_edges = Counter((e["kind"], e["task_id"], tuple(sorted(e["nodes"]))) for e in result.get("dependency_edges", []))
    audit.check(actual_edges == expected_edges, "dependency_edges", "edges", "Dependency records differ from frozen tasks")
    emitted = result.get("tasks", [])
    audit.check(Counter(t["id"] for t in emitted) == Counter(tasks.keys()), "frozen_task_coverage", "tasks", "A task is missing or duplicated")
    for actual in emitted:
        tid = actual["id"]
        if tid not in tasks:
            continue
        expected = tasks[tid]
        for key in ("kind", "type", "nodes", "box_ids", "start", "return"):
            audit.check(actual.get(key) == expected[key], "frozen_task_inheritance", tid,
                        f"{key} differs from Q3")
        audit.close(actual.get("work_s"), expected["work_s"], "frozen_task_work", tid)
        actual_intervals = actual.get("intervals", {})
        audit.check(set(actual_intervals) == set(expected["intervals"]), "task_resource_types", tid, "Wrong occupied resource types")
        for key, (begin, end) in expected["intervals"].items():
            if key not in actual_intervals:
                continue
            audit.check(actual_intervals[key][0] == begin, "frozen_interval_start", tid, key)
            audit.close(actual_intervals[key][1], end, "frozen_interval_end", f"{tid}/{key}")
    baseline = _requirements(list(tasks.values()))
    audit.check(result.get("baseline", {}).get("resources") == baseline, "pooled_baseline", "baseline", str(baseline))
    audit.check(result.get("baseline", {}).get("total_resources") == sum(baseline.values()), "pooled_total", "baseline", str(sum(baseline.values())))
    summary = {}

    def check_candidate(candidate, k, truths, with_allocations=False):
        cid = candidate["id"]
        groups = candidate["groups"]
        audit.check(candidate.get("k") == k and len(groups) == k, "group_count", cid, f"Expected {k} groups")
        audit.check(len({g["id"] for g in groups}) == len(groups), "group_ids", cid, "Duplicate group ID")
        members = Counter(n for group in groups for n in group["nodes"])
        audit.check(members == Counter({n: 1 for n in nodes}), "node_partition", cid, "Nodes must be covered exactly once")
        locations = {n: group["id"] for group in groups for n in group["nodes"]}
        for task in tasks.values():
            audit.check(len({locations.get(n) for n in task["nodes"]}) == 1,
                        "dependency_not_split", cid, f"Frozen task {task['id']} crosses groups")
        all_task_ids = Counter(tid for g in groups for tid in g["task_ids"])
        audit.check(all_task_ids == Counter(tasks.keys()), "group_task_coverage", cid,
                    "Each transport and each shared relay must appear in exactly one group")
        for group in groups:
            atom_ids = group["atom_ids"]
            inferred = set().union(*(atom_lookup.get(a, set()) for a in atom_ids))
            audit.check(bool(group["nodes"]) and set(group["nodes"]) == inferred,
                        "group_atom_consistency", f"{cid}/{group['id']}", "Group must be a nonempty union of listed atoms")
        audit.check(Counter(a for g in groups for a in g["atom_ids"]) == Counter(atom_lookup.keys()),
                    "atom_partition", cid, "Every atom ID must occur exactly once across groups")
        canonical = _canonical([g["nodes"] for g in groups])
        if not audit.check(canonical in truths, "valid_atom_partition", cid, "Partition is outside independent exhaustive set"):
            return canonical, None
        truth = truths[canonical]
        for group in groups:
            target = truth["groups"][tuple(sorted(group["nodes"]))]
            for key in ("task_ids", "resources", "box_count"):
                audit.check(group.get(key) == target[key], "group_"+key, f"{cid}/{group['id']}",
                            f"actual={group.get(key)}; expected={target[key]}")
            for key in ("work_s", "mass_kg"):
                audit.close(group.get(key), target[key], "group_"+key, f"{cid}/{group['id']}")
        for key in ("totals", "shortages", "surplus", "added_vs_pool", "total_resources", "shortage_total"):
            audit.check(candidate.get(key) == truth[key], key, cid, f"actual={candidate.get(key)}; expected={truth[key]}")
        for key in ("work_cv", "box_cv"):
            audit.close(candidate.get(key), truth[key], key, cid)
        audit.check(len(candidate.get("score", [])) == 3 and all(abs(a-b) <= 1e-8 for a,b in zip(candidate["score"], truth["score"])),
                    "lex_score", cid, f"Expected {truth['score']}")
        if with_allocations:
            _allocations(audit, candidate, tasks)
        return canonical, truth

    for k in (2, 3):
        section = result.get("partitions", {}).get(str(k), {})
        possible = _all_partitions(atoms, k)
        truths = {partition: _evaluate(partition, tasks, boxes, inventory, baseline) for partition in possible}
        candidates = section.get("all", [])
        ids = [c["id"] for c in candidates]
        audit.check(len(ids) == len(set(ids)), "candidate_ids", f"K{k}", "Duplicate candidate ID")
        observed, independently_scored = [], {}
        for candidate in candidates:
            canonical, truth = check_candidate(candidate, k, truths)
            observed.append(canonical)
            if truth is not None:
                independently_scored[candidate["id"]] = truth
        audit.check(Counter(observed) == Counter({p: 1 for p in possible}), "exhaustive_enumeration", f"K{k}",
                    f"Expected each of {len(possible)} partitions once; emitted={len(candidates)}")
        audit.check(section.get("enumerated") == len(possible), "enumeration_count", f"K{k}", str(len(possible)))
        zero = sum(truth["shortage_total"] == 0 for truth in truths.values())
        audit.check(section.get("zero_gap_count") == zero, "zero_gap_count", f"K{k}", str(zero))
        if independently_scored:
            best_id = min(independently_scored, key=lambda cid: (*independently_scored[cid]["score"], cid))
            balanced_id = min(independently_scored, key=lambda cid: (
                independently_scored[cid]["work_cv"], independently_scored[cid]["shortage_total"],
                independently_scored[cid]["total_resources"], cid))
            for key, expected_id in (("selected", best_id), ("balanced", balanced_id)):
                chosen = section.get(key)
                if audit.check(isinstance(chosen, dict), "selected_present", f"K{k}/{key}", "Missing selected candidate"):
                    check_candidate(chosen, k, truths, with_allocations=True)
                    audit.check(chosen["id"] == expected_id, key+"_optimal", f"K{k}", f"Expected {expected_id}")
                    original = next((c for c in candidates if c["id"] == chosen["id"]), None)
                    audit.check(original is not None and _canonical([g["nodes"] for g in original["groups"]])
                                == _canonical([g["nodes"] for g in chosen["groups"]]),
                                "selection_matches_enumeration", f"K{k}/{key}", "Selected ID refers to another partition")
            pareto = []
            for cid, truth in independently_scored.items():
                dominated = False
                for other, alternative in independently_scored.items():
                    if other == cid:
                        continue
                    a, b = alternative["score"], truth["score"]
                    if max(x-y for x,y in zip(a,b)) <= 0 and min(x-y for x,y in zip(a,b)) < 0:
                        dominated = True
                        break
                if not dominated:
                    pareto.append(cid)
            audit.check(Counter(section.get("pareto_ids", [])) == Counter(pareto), "pareto_frontier", f"K{k}", str(pareto))
        summary[str(k)] = {"expected_partitions": len(possible), "zero_gap_count": zero,
                           "minimum_shortage_total": min((t["shortage_total"] for t in truths.values()), default=None)}
    return {"passed": audit.error_count == 0, "check_count": audit.count,
            "error_count": audit.error_count, "errors": audit.errors,
            "checks": dict(audit.checked), "failed_checks": dict(audit.failed),
            "atom_count": len(atoms), "task_count": len(tasks), "partitions": summary,
            "independent_baseline": baseline,
            "scope": "Reconstructed from frozen Q3 and raw resource data; exact half-open time comparisons, active-at-start peak counts, graph traversal and all labelled assignments independently validate the no-relay-duplication model."}
