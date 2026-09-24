"""Fault-injection checks for the independent Q3 audit (no optimizer needed)."""

from copy import deepcopy
import math
import unittest

import numpy as np

from verify import charge_time, verify


class FlatTerrain:
    """A fully known synthetic geometry, in metres rather than geographic units."""

    def xy(self, x, y):
        return np.array([x, y], dtype=float)

    def max_along(self, _a, _b):
        return 0.0

    def contains(self, _x, _y):
        return True

    def ground(self, _x, _y):
        return 0.0

    def blocked(self, _a, _b):
        return False

    def _swept_los_clear(self, _a, _b, _anchor):
        return True


def scenario():
    """A manually timed out-and-back flight with a 25-second delivery."""
    parameter = dict(mass_empty=10.0, payload=5.0, volume=1.0, energy=1.0,
                     range_empty=10000.0, range_full=5000.0, speed=10.0,
                     up_speed=10.0, down_speed=10.0, eta_up=0.8,
                     reserve=0.1, prepare=20.0, load_per_box=5.0,
                     service_base=20.0, service_per_box=5.0,
                     charge_full=100.0, drones=["A-1"], battery_ids=["A-BAT-1"])
    data = {"nodes": {"O01": dict(lon=0.0, lat=0.0, z=0.0, work_agl=0.0),
                      "S001": dict(lon=1000.0, lat=0.0, z=0.0, work_agl=30.0)},
            "boxes": [dict(id="BOX-1", node="S001", mass=1.0, volume=0.1,
                           deadline=math.inf, due=200.0, weight=2.0)],
            "types": {"A": parameter}, "relay": {},
            "links": dict(frequency_mhz=2400.0, obstruction_loss=10.0,
                          system_loss=3.0, sensitivity=-98.0, fade_margin=8.0,
                          gateway_height=10.0,
                          transport=dict(pt=30.0, gain=5.0),
                          gateway=dict(pt=40.0, gain=5.0),
                          relay_access=dict(pt=30.0, gain=5.0),
                          relay_backhaul=dict(pt=40.0, gain=5.0))}
    definitions = [
        (25, 30, [0, 0, 0], [0, 0, 50], "climb", 1),
        (30, 130, [0, 0, 50], [1000, 0, 50], "cruise", 1),
        (130, 132, [1000, 0, 50], [1000, 0, 30], "descent", 1),
        (132, 157, [1000, 0, 30], [1000, 0, 30], "delivery", 1),
        (157, 159, [1000, 0, 30], [1000, 0, 50], "climb", 0),
        (159, 259, [1000, 0, 50], [0, 0, 50], "cruise", 0),
        (259, 264, [0, 0, 50], [0, 0, 0], "descent", 0),
    ]
    energy = (1000 / (10000 - 5000 * 0.2**1.5) + 0.1
              + (11 * 50 + 10 * 20) * 9.80665 / (0.8 * 3.6e6))
    trip = dict(id="T-1", type="A", drone="A-1", battery="A-BAT-1",
                start=0.0, takeoff=25.0, return_=264.0, box_ids=["BOX-1"],
                route=["S001"], deliveries={"BOX-1": 157.0}, energy=energy, soc=1-energy,
                segments=[dict(zip(("t0", "t1", "p0", "p1", "phase", "load"), x))
                          for x in definitions])
    trip["return"] = trip.pop("return_")
    return {"trips": [trip], "relay_trips": [], "communication": []}, data


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.result, self.data = scenario()
        self.terrain = FlatTerrain()

    def audit(self):
        return verify(self.result, self.data, self.terrain)

    def test_known_feasible_flight_passes(self):
        outcome = self.audit()
        self.assertTrue(outcome["passed"], outcome["errors"])
        self.assertTrue(outcome["communication"]["continuous_certified"])
        self.assertEqual(outcome["metrics"]["last_delivery_s"], 157.0)

    def test_two_stage_charge_boundaries(self):
        self.assertAlmostEqual(charge_time(0, 100), 100)
        self.assertAlmostEqual(charge_time(0.9, 100), 35)
        self.assertAlmostEqual(charge_time(0.95, 100), 17.5)
        self.assertAlmostEqual(charge_time(1, 100), 0)
        with self.assertRaises(ValueError):
            charge_time(-0.1, 100)

    def test_self_consistent_but_false_energy_is_rejected(self):
        self.result["trips"][0]["energy"] = 0.01
        self.result["trips"][0]["soc"] = 0.99
        self.assertIn("transport_energy", self.audit()["failed_checks"])

    def test_delivery_deadline_uses_end_of_service(self):
        self.data["boxes"][0]["deadline"] = 156.0
        self.assertIn("hard_deadline", self.audit()["failed_checks"])

    def test_battery_is_unavailable_while_drone_can_fly_again(self):
        second = deepcopy(self.result["trips"][0])
        second.update(id="T-2", box_ids=["BOX-2"], deliveries={"BOX-2": 421.0})
        for key in ("start", "takeoff", "return"):
            second[key] += 264.0
        for segment in second["segments"]:
            segment["t0"] += 264.0
            segment["t1"] += 264.0
        self.result["trips"].append(second)
        self.data["boxes"].append(dict(self.data["boxes"][0], id="BOX-2"))
        failures = self.audit()["failed_checks"]
        self.assertIn("transport_battery_conflict", failures)
        self.assertNotIn("transport_drone_conflict", failures)

    def test_lost_box_is_rejected(self):
        self.result["trips"] = []
        self.assertIn("box_inventory", self.audit()["failed_checks"])

    def test_communication_failure_is_not_hidden_by_flags(self):
        self.data["links"]["gateway"]["pt"] = -60.0
        self.result["communication"] = [dict(trip_id="T-1", phase="all", t0=25.0,
                                             t1=264.0, mode="direct", certified=True)]
        outcome = self.audit()
        self.assertFalse(outcome["passed"])
        self.assertGreater(outcome["communication"]["interruptions"], 0)
        self.assertFalse(outcome["communication"]["continuous_certified"])


if __name__ == "__main__":
    unittest.main()
