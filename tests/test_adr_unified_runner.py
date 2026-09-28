"""CPU-only orchestration tests using tiny JSON-writing subprocesses, not solves."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
from scripts.advection_diffusion_reaction.campaigns.unified import run_adr_unified_campaign as runner
from scripts.advection_diffusion_reaction.campaigns.unified.adr_unified_worker import check_loaded_cuda_runtime, compatibility


class Common:
    @staticmethod
    def read_json(path):
        return json.loads(Path(path).read_text())

    @staticmethod
    def atomic_json(path, value):
        path = Path(path)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value))
        temp.replace(path)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = SimpleNamespace(output=Path(self.temp.name), resume=False, timeout=5., heartbeat=30.)
        self.spec = dict(stage='solve', case='test', maxiter=2000)

    def job(self, script, **kwargs):
        # Locate the attempt chosen by the real runner, then emit worker JSON.
        code = ("import pathlib,json; p=max(pathlib.Path("+repr(self.temp.name)+")"
                ".glob('jobs/test/attempt_*'),key=lambda p:int(p.name.split('_')[-1])); "+script)
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run_job('test', self.spec, self.args, Common, dict(os.environ),
                                  command=[sys.executable, '-B', '-c', code], **kwargs)

    def test_pass_then_resume_does_not_launch(self):
        first = self.job("(p/'result.json').write_text(json.dumps({'status':'passed'}))")
        self.assertEqual(first['status'], 'passed')
        self.args.resume = True
        second = self.job("raise RuntimeError('must not execute')")
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.args.output/'jobs/test').glob('attempt_*'))), 1)

    def test_failure_is_preserved_on_resume(self):
        first = self.job("(p/'result.json').write_text(json.dumps({'status':'numerical_failure'}))")
        self.args.resume = True
        self.assertEqual(self.job("raise RuntimeError('must not execute')"), first)

    def test_crash_is_retryable_with_attempt_history(self):
        failed = self.job("raise RuntimeError('mock crash')")
        self.assertEqual(failed['status'], 'process_error')
        self.args.resume = True
        passed = self.job("(p/'result.json').write_text(json.dumps({'status':'passed'}))")
        self.assertEqual(passed['attempt'], 2)
        self.assertTrue((self.args.output/'jobs/test/attempt_1/worker.log').exists())

    def test_stale_configuration_is_rejected(self):
        self.job("(p/'result.json').write_text(json.dumps({'status':'passed'}))")
        self.args.resume = True
        self.spec['maxiter'] = 3000
        with self.assertRaisesRegex(ValueError, 'Changed specification'):
            self.job('pass')

    def test_backend_switch_cannot_resume(self):
        self.spec['amgx_backend'] = 'cusparse_generic'
        self.job("(p/'result.json').write_text(json.dumps({'status':'passed'}))")
        self.args.resume = True
        self.spec['amgx_backend'] = 'legacy'
        with self.assertRaisesRegex(ValueError, 'Changed specification'):
            self.job('pass')

    def test_resume_rejects_missing_mesh(self):
        self.spec['stage'] = 'mesh'
        self.job("(p/'result.json').write_text(json.dumps({'status':'passed','mesh_path':'/missing/mock.npz','mesh_sha256':'bad'}))")
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, 'Prepared mesh'):
            self.job('pass')

    def test_changed_cache_rejected_before_reuse(self):
        cache = self.args.output/'cache'
        cache.mkdir()
        (cache/'marker.npy').write_bytes(b'opaque test data, not a numerical array')
        self.spec.update(stage='assemble', cache=str(cache))
        self.job("(p/'result.json').write_text(json.dumps({'status':'passed'}))")
        self.args.resume = True
        (cache/'marker.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Assembly cache'):
            self.job('pass')

    def test_timeout_is_terminal_and_process_stopped(self):
        self.args.timeout = .1
        result = self.job('import time; time.sleep(3)')
        self.assertEqual(result['status'], 'timeout')
        self.assertIsNotNone(result['returncode'])

    def test_masks_preserve_scheduler_tokens(self):
        self.assertEqual(runner.mask_for_device(1, 'GPU-a,GPU-b'), 'GPU-b')
        self.assertEqual(runner.mask_for_device(0, 'MIG-test'), 'MIG-test')
        self.assertEqual(runner.mask_for_device(2, None), '2')
        with self.assertRaises(ValueError):
            runner.mask_for_device(1, 'GPU-a')

    def test_cuda13_rejects_amu_even_for_legacy(self):
        for major, minor in ((3, 7), (6, 0), (7, 0)):
            with self.assertRaises(ValueError):
                compatibility(dict(major=major, minor=minor), 13000, 'legacy')

    def test_cuda12_cannot_silently_run_generic(self):
        with self.assertRaisesRegex(ValueError, 'No silent'):
            compatibility(dict(major=7, minor=0), 12080, 'cusparse_generic')
        compatibility(dict(major=8, minor=0), 13000, 'cusparse_generic')

    def test_legacy_cuda12_supports_p100_v100_not_k80(self):
        for major in (6, 7):
            compatibility(dict(major=major, minor=0), 12080, 'legacy')
        for runtime in (11080, 12080):
            with self.assertRaises(ValueError):
                compatibility(dict(major=3, minor=7), runtime, 'legacy')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            compatibility(dict(major=7, minor=0), 12080, 'automatic')

    def test_mixed_loaded_runtimes_rejected(self):
        check_loaded_cuda_runtime(['/cuda12/libcudart.so.12.8.57', '/amgx/libamgxsh.so'], 12080)
        with self.assertRaisesRegex(ValueError, 'Mixed CUDA'):
            check_loaded_cuda_runtime(['/cuda12/libcudart.so.12.8.57', '/cuda13/libcudart.so.13.0.48'], 12080)
        with self.assertRaisesRegex(ValueError, 'Mixed CUDA'):
            check_loaded_cuda_runtime(['/cuda13/libcudart.so.13.0.48'], 12080)

    def test_full_plan_dispatch_without_numerical_work(self):
        self.dispatch_plan('cusparse_generic')

    def test_full_legacy_plan_dispatch_without_numerical_work(self):
        self.dispatch_plan('legacy')

    def test_plan_backend_mismatch_rejected_before_preflight(self):
        args = runner.parser().parse_args(['--output', self.temp.name, '--amgx-backend', 'legacy'])
        plan = runner.build_plan(runner.read_inventory(args.inventory))
        with patch.object(runner, 'run_job') as launch, self.assertRaisesRegex(ValueError, 'backend differ'):
            runner.execute(plan, args, Common)
        launch.assert_not_called()

    def test_plan_only_cli_needs_neither_cuda_nor_branch_tree(self):
        for backend in ('cusparse_generic', 'legacy'):
            output = self.args.output/backend
            completed = subprocess.run([
                sys.executable, '-B', str(ROOT/'scripts/advection_diffusion_reaction/campaigns/unified/run_adr_unified_campaign.py'),
                '--output', str(output), '--branch-root', '/missing/branch/tree',
                '--amgx-backend', backend,
            ], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = Common.read_json(output/'plan.json')
            self.assertEqual(plan['amgx_backend'], backend)
            self.assertEqual(plan['coverage']['solver_jobs'], 947)
            self.assertFalse((output/'jobs').exists())

    def dispatch_plan(self, backend):
        args = runner.parser().parse_args(['--output', self.temp.name, '--execute', '--amgx-backend', backend])
        plan = runner.build_plan(runner.read_inventory(args.inventory), amgx_backend=backend)
        calls, artifacts = [], {}
        class MemoryCommon(Common):
            @staticmethod
            def atomic_json(path, value):
                artifacts[str(path)] = value
        def fake_job(key, spec, *unused, **kwargs):
            calls.append((key, spec))
            if spec['stage'] == 'preflight':
                self.assertEqual(spec['amgx_backend'], backend)
                for config in spec['amgx_configs']:
                    self.assertEqual(config['solver']['bsr_spmv_backend'], backend)
                return dict(status='passed', selected=dict(device=0, name='mock GPU', total_bytes=80*2**30,
                            selection_fp64_flops=10e12), cuda_runtime=12080 if backend == 'legacy' else 13000,
                            cupy='mock', library_sha256={})
            if spec['stage'] == 'mesh':
                s = spec['system']
                triangles = s.get('triangles') or s['target_triangles']
                return dict(status='passed', mesh_path='/mock/mesh.npz', mesh_sha256='mock',
                            triangles=triangles, trace_dofs=triangles*10, n=500)
            if spec['stage'] == 'assemble':
                return dict(status='passed', operator_sha256='mock')
            self.assertEqual(spec['expected_operator_sha256'], 'mock')
            self.assertEqual(spec['reference_max_dofs'], 0)
            self.assertEqual(spec['maxiter'], 2000)
            self.assertGreater(spec['memory_estimate']['solver_device_bytes'], 0)
            self.assertNotIn('pardiso', spec['family'])
            self.assertEqual(spec['amgx_backend'], backend)
            if 'amgx_config' in spec:
                self.assertEqual(spec['amgx_config']['solver']['bsr_spmv_backend'], backend)
            return dict(status='passed')
        with patch.object(runner, 'run_job', fake_job), patch.object(runner, 'source_hashes', return_value={}), \
             patch.dict(os.environ, {'NUMBA_DISABLE_JIT': '0'}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.execute(plan, args, MemoryCommon), 0)
        solve_calls = [s for _, s in calls if s['stage'] == 'solve']
        self.assertEqual(len(solve_calls), plan['coverage']['solver_jobs'])
        self.assertEqual(len(calls), len({key for key, _ in calls}))
        summary = artifacts[str(args.output/'summary.json')]
        self.assertEqual(len({r['system_id'] for r in summary}), 128)
        self.assertEqual({r['amgx_backend'] for r in summary}, {backend})


if __name__ == '__main__':
    unittest.main()
