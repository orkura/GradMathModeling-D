"""Small exact oracle and end-to-end corruption detection."""
import copy
import unittest
from q1.solver import dynamic_program, solve
from q1.verify import verify
from q3.data import load_data
from q3.physics import Terrain


class DPTests(unittest.TestCase):
    def test_priority_changes_batching(self):
        p=[dict(counts=[1],energy_kwh=1.,duration_s=2.),
           dict(counts=[2],energy_kwh=3.,duration_s=3.)]
        self.assertEqual(dynamic_program([2],p,(0,1,2))[0],(1,3.,3.))
        self.assertEqual(dynamic_program([2],p,(1,0,2))[0],(2,2.,4.))

    def test_no_feasible_single_box(self):
        with self.assertRaises(ValueError):
            dynamic_program([1],[],(0,1,2))

    def test_empty_demand(self):
        self.assertEqual(dynamic_program([0,0],[],(0,1,2))[0],(0,0,0))


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_data();cls.terrain=Terrain(cls.data['dem_path'])
        cls.result=solve(cls.data,cls.terrain)

    def test_original_and_independent_milp(self):
        audit=verify(self.result,self.data,self.terrain)
        self.assertTrue(audit['passed'],audit['errors'])
        self.assertEqual(self.result['metrics']['sorties'],18)
        self.assertEqual(len(audit['milp_certificates']),15)

    def test_reject_duplicate_box(self):
        r=copy.deepcopy(self.result)
        r['trips'][0]['box_ids'].append(r['trips'][0]['box_ids'][0])
        self.assertFalse(verify(r,self.data,self.terrain,False)['passed'])

    def test_reject_energy_tampering(self):
        r=copy.deepcopy(self.result);r['trips'][0]['energy_kwh']-=.1
        self.assertFalse(verify(r,self.data,self.terrain,False)['passed'])

    def test_reject_omitted_pattern(self):
        r=copy.deepcopy(self.result);r['areas'][0]['patterns'].pop()
        self.assertFalse(verify(r,self.data,self.terrain,False)['passed'])

    def test_reject_missed_terrain(self):
        r=copy.deepcopy(self.result);r['areas'][0]['geometry']['dem_max_m']-=1
        self.assertFalse(verify(r,self.data,self.terrain,False)['passed'])


if __name__=='__main__':
    unittest.main()
