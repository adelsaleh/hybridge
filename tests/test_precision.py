"""Regression checks for process precision and the experimental GPU runner."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / '.cache' / 'pyamgx-fp32'


def _run(code: str, precision: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, HYBRIDGE_PRECISION=precision,
               NUMBA_CACHE_DIR=str(tmp_path / f'numba-{precision}'))
    return subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT,
                          text=True, capture_output=True, timeout=180)


@pytest.mark.parametrize('precision,mode,itemsize', [('float32', 'dFFI', 4), ('float64', 'dDDI', 8)])
def test_precision_geometry_fields_and_mixed_array_guard(precision, mode, itemsize, tmp_path):
    code = f'''
import numpy as np
from hybridge.runtime.precision import REAL_DTYPE, REAL_ITEMSIZE, AMGX_MODE, check_real_arrays
from hybridge.core.mesh import rectangle_mesh, _mesh_cache_files
from hybridge.core.space import DGSpace
mesh = rectangle_mesh(nx=1, ny=1)
space = DGSpace(mesh, 2)
assert mesh.node_coords.dtype == np.dtype({precision!r})
assert space.reference.Krf_quads.dtype == np.dtype({precision!r})
assert space.reference.Krf_w.dtype == np.dtype({precision!r})
assert REAL_ITEMSIZE == {itemsize}
assert AMGX_MODE == {mode!r}
check_real_arrays('test', [np.ones(4, dtype=REAL_DTYPE), np.ones(4, dtype='int32')])
wrong = 'float64' if {precision!r} == 'float32' else 'float32'
try:
    check_real_arrays('test', [np.ones(4, dtype=wrong)])
except TypeError:
    pass
else:
    raise AssertionError('mixed precision accepted')
_, key = _mesh_cache_files('test', 0.1, algorithm=None, cache_key_data=None, cache_dir=None)
assert ('"precision":"float32"' in key) == ({precision!r} == 'float32')
'''
    result = _run(code, precision, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_process_precision_is_rejected(tmp_path):
    result = _run('import hybridge', 'float16', tmp_path)
    assert result.returncode != 0
    assert 'HYBRIDGE_PRECISION must be float32 or float64' in result.stderr


def _require_cuda_and_binding():
    cp = pytest.importorskip('cupy')
    try:
        assert cp.cuda.runtime.getDeviceCount() > 0
    except Exception as exc:
        pytest.skip(f'CUDA unavailable: {exc}')
    if not tuple(BINDING.glob('pyamgx*.so')):
        pytest.skip('local precision-aware PyAMGX extension has not been built')


def test_fp32_cuda_math_and_amgx_vector_round_trip(tmp_path):
    _require_cuda_and_binding()
    code = f'''
import sys
sys.path.insert(0, {str(BINDING)!r})
import cupy as cp
import numpy as np
import pyamgx
from hybridge.runtime.precision import cuda_source, real_raw_kernel
source = 'extern "C" __global__ void eval(const double* x, double* y) {{ int i=threadIdx.x; y[i]=sqrt(x[i])+1.25e-2; }}'
converted = cuda_source(source)
assert 'double' not in converted and 'sqrtf(' in converted and '1.25e-2f' in converted
x = cp.asarray([1., 4., 9.], dtype=cp.float32)
y = cp.empty_like(x)
kernel = real_raw_kernel(source, 'eval')
kernel((1,), (3,), (x, y))
cp.testing.assert_allclose(y, cp.asarray([1.0125, 2.0125, 3.0125], dtype=cp.float32), rtol=1e-6)
try:
    kernel((1,), (3,), (x.astype(cp.float64), y))
except TypeError:
    pass
else:
    raise AssertionError('kernel accepted FP64 pointer')
pyamgx.initialize()
cfg = pyamgx.Config().create_from_dict({{'config_version': 2, 'solver': {{'solver':'BICGSTAB'}}}})
rsrc = pyamgx.Resources().create_simple(cfg)
try:
    for mode, dtype in [('dFFI', np.float32), ('dDDI', np.float64)]:
        vector = pyamgx.Vector().create(rsrc, mode=mode)
        try:
            values = np.arange(14, dtype=dtype)
            vector.upload(values, block_dim=7)
            output = vector.download()
            assert output.dtype == dtype
            np.testing.assert_array_equal(output, values)
            try:
                vector.download(np.empty(2, dtype=dtype))
            except ValueError:
                pass
            else:
                raise AssertionError('undersized download accepted')
            wrong = np.float64 if dtype == np.float32 else np.float32
            try:
                vector.upload(values.astype(wrong), block_dim=7)
            except ValueError:
                pass
            else:
                raise AssertionError('mismatched vector dtype accepted')
        finally:
            vector.destroy()
finally:
    rsrc.destroy()
    cfg.destroy()
    pyamgx.finalize()
'''
    result = _run(code, 'float32', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fp32_guiding_center_p6_pipeline_matches_fp64(tmp_path):
    _require_cuda_and_binding()
    common = [sys.executable, '-m', 'scripts.guiding_center.benchmarks.run_precision_benchmark',
              '--num-steps', '2', '--mesh-size', '0.18', '--minimum-triangles', '0',
              '--verbosity', '0', '--output', str(tmp_path)]
    for precision, tolerance in [('float32', '0.002'), ('float64', '1e-8')]:
        result = subprocess.run(common + ['--precision', precision, '--rtol', tolerance,
                                         '--poisson-rtol', tolerance, '--atol', '0'],
                                cwd=ROOT, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / 'float32-audit.json').read_text())
    assert report['amgx_mode'] == 'dFFI'
    assert report['transport_amgx_config_path'].endswith('adv_rea_gpu4_hdg_fgmres_scaled_none.json')
    for stage in ('poisson-assembly', 'poisson-krylov-workspace', 'poisson-reconstruction',
                  'transport-assembly', 'transport-reconstruction', 'diagnostic-reductions'):
        assert report['pipeline'][stage]['device_arrays'] > 0, stage
    for kernel in ('assemble_advection_raw_fused_bsr', 'assemble_diffusion_raw_coop_bsr',
                   'compact_diffusion_reconstruct', 'reconstruct_advection_from_response_raw'):
        assert report['kernels'][kernel]['calls'] > 0, kernel
    with np.load(tmp_path / 'float32-fields.npz') as single, np.load(tmp_path / 'float64-fields.npz') as double:
        np.testing.assert_array_equal(single['triangles'], double['triangles'])
        for field in ('density', 'potential'):
            assert single[field].dtype == np.float32
            relative_error = np.linalg.norm(single[field].astype(np.float64) - double[field]) / np.linalg.norm(double[field])
            assert relative_error < 0.01, (field, relative_error)


def test_fp32_cli_uses_relaxed_defaults_and_respects_overrides(tmp_path):
    command = [sys.executable, '-m', 'scripts.guiding_center.run_guiding_center_cases',
               '--preset', 'diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx',
               '--precision', 'float32', '--dry-run', '--poisson-solver-rtol', '0.004']
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    for setting in ('poisson_solver_rtol: 0.004', 'transport_solver_rtol: 0.005',
                    'transport_amgx_tolerance: 0.005', 'transport_solver_atol: 0.0',
                    'order: 6', 'dt: 0.1', 'num_steps: 500',
                    'adv_rea_gpu4_hdg_fgmres_scaled_none.json'):
        assert setting in result.stdout


@pytest.mark.parametrize('precision,explicit_config', [('float32', True), ('float64', False)])
def test_transport_solver_choice_preserves_explicit_override_and_fp64(precision, explicit_config):
    command = [sys.executable, '-m', 'scripts.guiding_center.run_guiding_center_cases',
               '--preset', 'diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx',
               '--precision', precision, '--dry-run']
    if explicit_config:
        command += ['--transport-amgx-config', 'configs/amgx/adv_rea_gpu4_hdg_bicgstab_scaled_none.json']
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'adv_rea_gpu4_hdg_bicgstab_scaled_none.json' in result.stdout
    assert 'adv_rea_gpu4_hdg_fgmres_scaled_none.json' not in result.stdout
