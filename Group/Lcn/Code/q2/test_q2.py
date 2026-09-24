"""Small analytic fixtures for Q2: no original/Q3 solution is required."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q2.verify import charge_time, verify


class FlatTerrain:
    def xy(self, lon, lat):
        return float(lon), float(lat)

    def max_along(self, a, b):
        return 0.0


def fixture():
    par = {"id": "A", "mass_empty": 10.0, "payload": 10.0, "volume": 1.0,
           "speed": 10.0, "range_empty": 20000.0, "range_full": 10000.0,
           "energy": 5.0, "reserve": 0.2, "prepare": 20.0, "load_per_box": 5.0,
           "service_base": 10.0, "service_per_box": 5.0, "up_speed": 2.0,
           "down_speed": 2.0, "eta_up": 0.8, "eta_down": 0.0,
           "drones": ["A01", "A02"], "battery_ids": ["BAT-A-01", "BAT-A-02"],
           "batteries": 2, "charge_full": 1000.0}
    nodes = {"O01": {"lon": 0, "lat": 0, "z": 0, "work_agl": 0},
             "S001": {"lon": 1000, "lat": 0, "z": 0, "work_agl": 30},
             "S002": {"lon": 2000, "lat": 0, "z": 0, "work_agl": 30}}
    boxes = [{"id": "B1", "node": "S001", "mass": 2.0, "volume": 0.1,
              "medical": False, "first": False, "first_deadline": math.inf,
              "due": 1000.0, "deadline": math.inf, "weight": 2.0},
             {"id": "B2", "node": "S002", "mass": 4.0, "volume": 0.2,
              "medical": False, "first": False, "first_deadline": math.inf,
              "due": 1000.0, "deadline": math.inf, "weight": 3.0}]
    return {"types": {"A": par}, "nodes": nodes, "boxes": boxes}


def manual_trip(data, boxids, route, start=0.0, identifier="T001", drone="A01", battery="BAT-A-01"):
    """Serialize the analytic flat, east-west fixture; never calls solver/audit helpers."""
    selected = [box for box in data["boxes"] if box["id"] in boxids]
    load = sum(box["mass"] for box in selected)
    mass = load
    takeoff = start + 20 + 5 * len(selected)
    time, x, z, energy = takeoff, 0, 0, 0.0
    segments, deliveries = [], {}
    for destination in [*route, "O01"]:
        node = data["nodes"][destination]
        target_x, target_z = node["lon"], node["z"] + node["work_agl"]
        distance = abs(target_x - x)
        climb = 50 - z
        # The test world is flat with 50 m cruise height and 2 m/s vertical speed.
        steps = [("climb", [x, 0, z], [x, 0, 50], climb / 2),
                 ("cruise", [x, 0, 50], [target_x, 0, 50], distance / 10),
                 ("descent", [target_x, 0, 50], [target_x, 0, target_z], (50 - target_z) / 2)]
        for phase, p0, p1, duration in steps:
            segments.append({"phase": phase, "t0": time, "t1": time + duration,
                             "p0": p0, "p1": p1, "load": load})
            time += duration
        energy += 5 * distance / (20000 - 10000 * (load / 10) ** 1.5)
        energy += (10 + load) * 9.80665 * climb / (0.8 * 3600000)
        x, z = target_x, target_z
        if destination != "O01":
            drop = [box for box in selected if box["node"] == destination]
            service = 10 + 5 * len(drop)
            segments.append({"phase": "delivery", "t0": time, "t1": time + service,
                             "p0": [x, 0, z], "p1": [x, 0, z], "load": load})
            time += service
            deliveries.update({box["id"]: time for box in drop})
            load -= sum(box["mass"] for box in drop)
    return {"id": identifier, "type": "A", "box_ids": list(boxids), "route": list(route),
            "drone": drone, "battery": battery, "start": start, "takeoff": takeoff,
            "return": time, "energy": energy, "soc": 1 - energy / 5,
            "mass": mass, "volume": sum(box["volume"] for box in selected),
            "segments": segments, "deliveries": deliveries}


class Q2AuditTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.terrain = FlatTerrain()
        self.trip = manual_trip(self.data, ["B1", "B2"], ["S001", "S002"])

    def audit(self, trips=None, data=None, **extra):
        return verify({"trips": trips if trips is not None else [self.trip], **extra},
                      data or self.data, self.terrain)

    def assertFails(self, audit, check):
        self.assertFalse(audit["passed"])
        self.assertIn(check, audit["failed_checks"])

    def one_box(self):
        self.data["boxes"] = self.data["boxes"][:1]
        self.trip = manual_trip(self.data, ["B1"], ["S001"])

    def test_two_stop_analytic_time_energy_and_unloading(self):
        result = self.audit()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["delivery_times"], {"B1": 180.0, "B2": 315.0})
        self.assertEqual(result["metrics"]["makespan_s"], 550.0)
        expected = (5000 / (20000 - 10000 * 0.6 ** 1.5)
                    + 5000 / (20000 - 10000 * 0.4 ** 1.5) + 0.5
                    + (16 * 50 + 14 * 20 + 10 * 20) * 9.80665 / 2880000)
        self.assertAlmostEqual(result["metrics"]["total_energy_kwh"], expected, places=12)
        self.assertEqual(result["metrics"]["multi_stop_sorties"], 1)
        self.assertEqual([s["load"] for s in self.trip["segments"] if s["phase"] == "cruise"], [6, 4, 0])

    def test_shared_battery_can_move_between_same_type_airframes(self):
        first = manual_trip(self.data, ["B1"], ["S001"])
        # Hand-computed recharge uses both charging regimes for this fixture.
        soc = first["soc"]
        refill = 1000 * (0.65 * max(0, 0.9 - soc) / 0.9 + 0.35 * min(0.1, 1 - soc) / 0.1)
        second = manual_trip(self.data, ["B2"], ["S002"], first["return"] + refill,
                             identifier="T002", drone="A02")
        result = self.audit([first, second])
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["resources"]["transport_battery"]["resources_used"], 1)
        self.assertEqual(result["resources"]["transport_drone"]["resources_used"], 2)

    def test_charging_blocks_battery_after_airframe_returns(self):
        first = manual_trip(self.data, ["B1"], ["S001"])
        second = manual_trip(self.data, ["B2"], ["S002"], first["return"], identifier="T002", drone="A02")
        self.assertFails(self.audit([first, second]), "transport_battery_conflict")

    def test_exact_airframe_release_allows_immediate_new_battery(self):
        first = manual_trip(self.data, ["B1"], ["S001"])
        second = manual_trip(self.data, ["B2"], ["S002"], first["return"], identifier="T002", battery="BAT-A-02")
        result = self.audit([first, second])
        self.assertTrue(result["passed"], result["errors"])

    def test_tiny_positive_airframe_overlap_is_rejected(self):
        first = manual_trip(self.data, ["B1"], ["S001"])
        second = manual_trip(self.data, ["B2"], ["S002"], first["return"] - 1e-9,
                             identifier="T002", battery="BAT-A-02")
        self.assertFails(self.audit([first, second]), "transport_drone_conflict")

    def test_two_stage_charging_boundaries(self):
        self.assertAlmostEqual(charge_time(0, 1000), 1000)
        self.assertAlmostEqual(charge_time(0.9, 1000), 350)
        self.assertAlmostEqual(charge_time(0.95, 1000), 175)
        self.assertEqual(charge_time(1, 1000), 0)
        with self.assertRaises(ValueError):
            charge_time(math.nan, 1000)

    def test_medical_due_is_hard(self):
        self.one_box()
        self.data["boxes"][0].update(medical=True, due=170)
        result = self.audit()
        self.assertFails(result, "hard_deadline")
        self.assertEqual(result["metrics"]["hard_deadline_violations"], 1)

    def test_first_box_deadline_is_hard_for_nonmedical_boxes(self):
        self.one_box()
        self.data["boxes"][0].update(first=True, first_deadline=170)
        self.assertFails(self.audit(), "hard_deadline")

    def test_ordinary_due_has_weighted_penalty_without_infeasibility(self):
        self.one_box()
        self.data["boxes"][0]["due"] = 170
        result = self.audit()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["metrics"]["weighted_tardiness"], 10)
        self.assertEqual(result["metrics"]["late_boxes"], 1)
        self.assertEqual(result["metrics"]["max_tardiness_s"], 5)

    def test_duplicate_and_missing_boxes_are_rejected(self):
        self.assertFails(self.audit([self.trip, copy.deepcopy(self.trip)]), "box_once")
        first = manual_trip(self.data, ["B1"], ["S001"])
        self.assertFails(self.audit([first]), "box_inventory")

    def test_unknown_or_wrong_type_resource_is_rejected(self):
        self.trip["battery"] = "BAT-B-01"
        self.trip["drone"] = "B01"
        result = self.audit()
        self.assertFails(result, "battery_inventory")
        self.assertFails(result, "drone_inventory")

    def test_nan_inf_in_physical_values_and_summary_are_rejected(self):
        for field, bad in [("return", math.nan), ("energy", math.inf), ("start", -math.inf)]:
            with self.subTest(field=field):
                trip = copy.deepcopy(self.trip)
                trip[field] = bad
                self.assertFails(self.audit([trip]), "finite_trip_values")
        self.assertFails(self.audit(metrics={"makespan_s": math.nan}), "finite_summary")

    def test_forged_energy_load_delivery_and_summary_are_rejected(self):
        self.trip["energy"] += 1
        self.trip["segments"][5]["load"] = 6
        self.trip["deliveries"]["B2"] = 1
        result = self.audit(metrics={"makespan_s": 1})
        for check in ["transport_energy", "segment_load", "delivery_completion", "summary_metric"]:
            self.assertFails(result, check)

    def test_payload_volume_and_reserve_checked_independently(self):
        self.data["types"]["A"].update(payload=5, volume=0.25, reserve=0.99)
        result = self.audit()
        for check in ["payload", "volume", "transport_reserve"]:
            self.assertFails(result, check)

    def test_original_terrain_is_reconsulted(self):
        class RidgeTerrain(FlatTerrain):
            def max_along(self, a, b):
                return 100.0
        result = verify({"trips": [self.trip]}, self.data, RidgeTerrain())
        for check in ["segment_geometry", "return_time", "transport_energy"]:
            self.assertFails(result, check)

    def test_audit_does_not_modify_input_schedule_or_parameters(self):
        result = {"trips": [self.trip]}
        previous_result, previous_data = copy.deepcopy(result), copy.deepcopy(self.data)
        verify(result, self.data, self.terrain)
        self.assertEqual(result, previous_result)
        self.assertEqual(self.data, previous_data)


if __name__ == "__main__":
    unittest.main()
