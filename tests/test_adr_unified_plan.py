"""No kernels, meshes or simulations: validate portable campaign planning."""
import importlib.util
from copy import deepcopy
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('adr_unified_plan', ROOT/'scripts/advection_diffusion_reaction/campaigns/unified/adr_unified_plan.py')
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


class UnifiedPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = plan.read_inventory(ROOT/'run_configs/adr_unified_l5/inventory.json')
        cls.plan = plan.build_plan(cls.inventory)

    def test_all_original_systems_preserved(self):
        original = {s['id']: s for s in self.inventory['systems']}
        current = {s['id']: s for s in self.plan['systems'] if s['level'] != 'L5'}
        self.assertEqual(set(original), set(current))
        self.assertEqual(len(original), 107)
        for key, s in original.items():
            self.assertEqual(s['problem'], current[key]['problem'])
            self.assertEqual({c['id'] for c in s['candidates']}, {c['id'] for c in current[key]['candidates']})

    def test_every_class_has_large_l5(self):
        originals = {plan.problem_class(s) for s in self.inventory['systems']}
        l5 = [s for s in self.plan['systems'] if s['level'] == 'L5']
        self.assertEqual(originals, {plan.problem_class(s) for s in l5})
        for s in l5:
            self.assertEqual(s['problem']['p'], 6)
            self.assertGreater(s.get('triangles') or s['target_triangles'], 390000)
            self.assertNotIn('archived_operator_sha256', s)

    def test_no_tiny_l5_or_reduced_cap(self):
        for kwargs in ({'l5_triangles': 150000}, {'maxiter': 1000}):
            with self.assertRaises(ValueError):
                plan.build_plan(self.inventory, **kwargs)

    def test_memory_grows_with_restart(self):
        from copy import deepcopy
        s = next(s for s in self.plan['systems'] if s['level'] == 'L5')
        choice = deepcopy(s['candidates'][0])
        before = plan.memory_estimate(s, choice)['solver_device_bytes']
        choice['solver']['restart'] = 500
        self.assertGreater(plan.memory_estimate(s, choice)['solver_device_bytes'], before)

    def test_no_direct_solver_and_both_pmg_policies(self):
        for s in self.plan['systems']:
            policies = {c['solver'].get('policy') for c in s['candidates']}
            self.assertTrue({'standard', 'robust'} <= policies)
            self.assertFalse(any('pardiso' in c['solver']['family'] for c in s['candidates']))

    def test_backend_rewrites_nested_scopes_without_mutation(self):
        original = {'solver': {'solver': 'FGMRES', 'max_iters': 2000,
                    'preconditioner': {'bsr_spmv_backend': 'cusparse_generic',
                    'levels': [{'bsr_spmv_backend': 'cusparse_generic', 'sweeps': 4}]}}}
        saved = deepcopy(original)
        updated = plan.configure_amgx_backend(original, 'legacy')
        expected = deepcopy(original)
        expected['solver']['bsr_spmv_backend'] = 'legacy'
        nested = expected['solver']['preconditioner']
        nested['bsr_spmv_backend'] = nested['levels'][0]['bsr_spmv_backend'] = 'legacy'
        self.assertEqual(updated, expected)
        self.assertEqual(original, saved)
        with self.assertRaises(ValueError):
            plan.configure_amgx_backend(original, 'automatic')

    def test_legacy_preserves_coverage_and_all_numerical_settings(self):
        saved = deepcopy(self.inventory)
        legacy = plan.build_plan(self.inventory, amgx_backend='legacy')
        self.assertEqual(self.inventory, saved)
        self.assertEqual(legacy['coverage'], self.plan['coverage'])
        self.assertEqual(legacy['coverage']['solver_jobs'], 947)
        self.assertNotEqual(plan.fingerprint(legacy), plan.fingerprint(self.plan))
        for before, after in zip(self.plan['systems'], legacy['systems'], strict=True):
            self.assertEqual(before['id'], after['id'])
            for generic, old in zip(before['candidates'], after['candidates'], strict=True):
                self.assertEqual(generic['id'], old['id'])
                expected = deepcopy(generic['solver'])
                if 'amgx_config' in expected:
                    expected['amgx_config'] = plan.configure_amgx_backend(expected['amgx_config'], 'legacy')
                    self.assertNotEqual(generic['effective_solver_sha256'], old['effective_solver_sha256'])
                self.assertEqual(old['solver'], expected)
                self.assertEqual(old['effective_solver_sha256'], plan.fingerprint(old['solver']))

    def test_default_keeps_archived_amgx_backend(self):
        self.assertEqual(self.plan['amgx_backend'], 'cusparse_generic')
        originals = {s['id']: s for s in self.inventory['systems']}
        for system in self.plan['systems']:
            if system['id'] not in originals:
                continue
            for old, new in zip(originals[system['id']]['candidates'], system['candidates'], strict=True):
                self.assertEqual(old['solver'].get('amgx_config'), new['solver'].get('amgx_config'))
        with self.assertRaises(ValueError):
            plan.build_plan(self.inventory, amgx_backend='automatic')


if __name__ == '__main__':
    unittest.main()
