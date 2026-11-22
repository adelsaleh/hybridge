"""Local-solve selection must preserve inputs and match an independent solve."""
import numpy as np
import pytest

from hybridge.runtime import cuda_tuning as tuning
from hybridge.linalg.gpu import tuned_batched_solve as batched


def test_selection_is_lazy_preserves_inputs_and_rejects_wrong_algorithm(monkeypatch, tmp_path):
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    device = lambda cp: dict(name='test', fp32_fp64_ratio=64)
    monkeypatch.setattr(tuning, 'device_signature', device)
    monkeypatch.setattr(batched, 'device_signature', device)
    monkeypatch.setattr(batched, 'require_cupy_device', lambda: np)
    monkeypatch.setattr(tuning, 'event_seconds', lambda cp, call: (call(), .001)[1])
    from hybridge.linalg.gpu import magma_batched
    monkeypatch.setattr(magma_batched, 'load_magma', lambda: None)
    def solve(name, a, b, *, info, **kwargs):
        x = np.linalg.solve(a.transpose(0, 2, 1), b.transpose(0, 2, 1)).transpose(0, 2, 1)
        b[...] = x + (1 if name == 'magma' else 0)
        a.fill(99)  # emulate destructive LU
        info.fill(0)
    monkeypatch.setattr(batched, 'solve_selected', solve)
    a = np.tile(np.eye(3, dtype=np.float32), (5, 1, 1))
    b = np.ones((5, 2, 3), dtype=np.float32)
    original_a, original_b, preparations = a.copy(), b.copy(), []
    tuning.clear_tuning_cache()
    name, report = batched.select_batched_solver(a, b, prepare_inputs=lambda: preparations.append(1))
    assert name == 'cublas' and preparations == [1]
    assert report['candidates'][1]['rejected'] == 'independent validation'
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(b, original_b)
    tuning.clear_tuning_cache()
    name, report = batched.select_batched_solver(a, b, prepare_inputs=lambda: preparations.append(2))
    assert report['source'] == 'disk' and preparations == [1]


def _cuda_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason='CUDA device required')
@pytest.mark.parametrize('n,nrhs', [(6, 10), (28, 22), (55, 31)])
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_gpu_selected_solver_matches_numpy_and_reuses_disk(n, nrhs, dtype, monkeypatch, tmp_path):
    import cupy as cp
    from hybridge.linalg.gpu.cublas_batched import BatchedLUWorkspace
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    tuning.clear_tuning_cache()
    rng = np.random.default_rng(41+n)
    logical = (rng.normal(size=(37, n, n)) + n*np.eye(n)).astype(dtype)
    logical[1::2] = logical[1::2, ::-1]  # pivoting required
    rhs = rng.normal(size=(37, n, nrhs)).astype(dtype)
    a = cp.asarray(np.ascontiguousarray(logical.transpose(0, 2, 1)))
    b = cp.asarray(np.ascontiguousarray(rhs.transpose(0, 2, 1)))
    name, report = batched.select_batched_solver(a, b)
    assert report['source'] == 'measured'
    if dtype == np.float64:
        assert any(row['variant'].startswith('fused:') and 'rejected' not in row
                   for row in report['candidates']), report
    np.testing.assert_array_equal(cp.asnumpy(a).transpose(0, 2, 1), logical)
    np.testing.assert_array_equal(cp.asnumpy(b).transpose(0, 2, 1), rhs)
    tuning.clear_tuning_cache()
    assert batched.select_batched_solver(a, b)[1]['source'] == 'disk'
    info = cp.empty(a.shape[0], dtype=cp.int32)
    batched.solve_selected(name, a, b, info=info, workspace=BatchedLUWorkspace())
    assert not bool(cp.any(info).item())
    tolerance = 2e-12 if dtype == np.float64 else 3e-5
    np.testing.assert_allclose(cp.asnumpy(b).transpose(0, 2, 1), np.linalg.solve(logical, rhs),
                               rtol=tolerance, atol=tolerance)
