"""Hand-computed boundaries and fault-injection tests for Q4 validation."""

from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import unittest

from q3.data import load_data
from q4.model import solve
from q4.verify import _charge, _peak_at_starts, verify


class IntervalTests(unittest.TestCase):
    def test_resource_returned_exactly_at_next_start_can_be_reused(self):
        self.assertEqual(_peak_at_starts([(0.0, 10.0), (10.0, 20.0)]), 1)

    def test_even_tiny_actual_overlap_is_not_erased_by_tolerance(self):
        self.assertEqual(_peak_at_starts([(0.0, 10.0+1e-12), (10.0, 20.0)]), 2)

    def test_recharging_needs_an_extra_battery_but_not_a_drone(self):
        recharge = _charge(0.9, 100.0)
        self.assertAlmostEqual(recharge, 35.0)
        self.assertEqual(_peak_at_starts([(0.0, 10.0), (10.0, 20.0)]), 1)
        self.assertEqual(_peak_at_starts([(0.0, 10.0+recharge), (10.0, 20.0+recharge)]), 2)
        self.assertAlmostEqual(_charge(0.0, 100.0), 100.0)
        self.assertAlmostEqual(_charge(0.95, 100.0), 17.5)
        self.assertEqual(_charge(1.0, 100.0), 0.0)


class FullScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[4]
        cls.data = load_data(root)
        cls.q3 = json.loads((root / "Group/Lcn/Results/Q3/solution.json").read_text(encoding="utf-8"))
        cls.good = solve(cls.q3, cls.data)

    def setUp(self):
        self.result = deepcopy(self.good)

    def failures(self):
        outcome = verify(self.result, self.q3, self.data)
        self.assertFalse(outcome["passed"])
        return outcome["failed_checks"]

    def test_original_scene_passes_complete_independent_enumeration(self):
        outcome = verify(self.result, self.q3, self.data)
        self.assertTrue(outcome["passed"], outcome["errors"])
        self.assertEqual(outcome["atom_count"], 6)
        self.assertEqual(outcome["task_count"], 41)
        self.assertEqual(outcome["partitions"]["2"]["expected_partitions"], 31)
        self.assertEqual(outcome["partitions"]["3"]["expected_partitions"], 90)
        self.assertEqual(outcome["partitions"]["2"]["minimum_shortage_total"], 2)
        self.assertEqual(outcome["partitions"]["3"]["minimum_shortage_total"], 5)

    def test_missing_candidate_cannot_claim_exhaustive_search(self):
        self.result["partitions"]["3"]["all"].pop()
        self.assertIn("exhaustive_enumeration", self.failures())

    def test_shared_relay_destinations_cannot_be_split(self):
        relay = next(t for t in self.result["tasks"] if t["kind"] == "relay" and len(t["nodes"]) > 1)
        groups = self.result["partitions"]["2"]["selected"]["groups"]
        node = relay["nodes"][0]
        source = next(g for g in groups if node in g["nodes"])
        destination = next(g for g in groups if g is not source)
        source["nodes"].remove(node)
        destination["nodes"].append(node)
        self.assertIn("dependency_not_split", self.failures())

    def test_wrong_entity_assignment_causing_overlap_is_rejected(self):
        found = False
        for group in self.result["partitions"]["2"]["selected"]["groups"]:
            for first, second in combinations(group["allocations"], 2):
                if first["resource_key"] == second["resource_key"] and (
                        first["start"] < second["end"] and second["start"] < first["end"]):
                    second["resource_id"] = first["resource_id"]
                    found = True
                    break
            if found:
                break
        self.assertTrue(found, "The actual schedule must contain concurrent resource occupations")
        self.assertIn("resource_overlap", self.failures())

    def test_one_physical_resource_cannot_belong_to_two_groups(self):
        first, second = self.result["partitions"]["2"]["selected"]["groups"]
        found = False
        for a in first["allocations"]:
            for b in second["allocations"]:
                if a["resource_key"] == b["resource_key"]:
                    b["resource_id"] = a["resource_id"]
                    found = True
                    break
            if found:
                break
        self.assertTrue(found)
        self.assertIn("resource_not_shared_across_groups", self.failures())

    def test_lost_resource_allocation_is_rejected(self):
        self.result["partitions"]["2"]["selected"]["groups"][0]["allocations"].pop()
        self.assertIn("allocation_task_coverage", self.failures())

    def test_frozen_q3_task_cannot_be_shifted(self):
        self.result["tasks"][0]["start"] += 1.0
        self.assertIn("frozen_task_inheritance", self.failures())

    def test_wrong_claimed_group_peak_is_rejected(self):
        group = self.result["partitions"]["2"]["all"][0]["groups"][0]
        group["resources"]["drone_A"] += 1
        self.assertIn("group_resources", self.failures())

    def test_pareto_claim_is_independently_recomputed(self):
        self.result["partitions"]["3"]["pareto_ids"] = []
        self.assertIn("pareto_frontier", self.failures())


if __name__ == "__main__":
    unittest.main()
