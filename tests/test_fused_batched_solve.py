from __future__ import annotations

import numpy as np
import pytest


def _cupy_runtime_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")


def _systems(batch: int, n: int, nrhs: int, seed: int):
    """Random systems in the column-major batched layout, half of them needing row exchanges."""
    rng = np.random.default_rng(seed)
    logical = rng.standard_normal((batch, n, n)) + 0.5 * n * np.eye(n)
    # Permute the rows of odd systems so partial pivoting must exchange rows.
    for b in range(1, batch, 2):
        logical[b] = logical[b][rng.permutation(n)]
    logical[0, 0, 0] = 0.0  # a zero leading entry in the first system
    rhs = rng.standard_normal((batch, n, nrhs))
    return logical, rhs


@pytest.mark.parametrize("variant", (1, 2, 3))
@pytest.mark.parametrize("n, nrhs, config", [
    (28, 22, None), (36, 25, None), (45, 28, None), (55, 31, None),
    (55, 31, (16, 16, 2)), (55, 31, (8, 32, 1)), (45, 28, (4, 16, 2)), (28, 22, (32, 2, 4)), (10, 3, (4, 8, 1)),
])
def test_fused_solve_matches_numpy(n, nrhs, config, variant):
    import cupy as cp

    from hybridge.linalg.gpu.fused_batched_solve import solve_batched_fused

    logical, rhs = _systems(37, n, nrhs, seed=n + nrhs)
    matrices = cp.asarray(np.ascontiguousarray(logical.transpose(0, 2, 1)))
    original = matrices.copy()
    solution = cp.asarray(np.ascontiguousarray(rhs.transpose(0, 2, 1)))
    info = cp.full(logical.shape[0], -1, dtype=cp.int32)
    from hybridge.linalg.gpu.fused_batched_solve import default_config

    config = (default_config(n) if config is None else config)[:3] + (variant,)
    solve_batched_fused(matrices, solution, config=config, info=info)
    expected = np.linalg.solve(logical, rhs)
    got = cp.asnumpy(solution).transpose(0, 2, 1)
    assert np.all(cp.asnumpy(info) == 0)
    assert bool(cp.all(matrices == original))
    error = np.linalg.norm(got - expected, axis=(1, 2)) / np.linalg.norm(expected, axis=(1, 2))
    assert error.max() < 1e-12


@pytest.mark.parametrize("variant", (1, 2, 3))
def test_fused_solve_reports_singular_systems(variant):
    import cupy as cp

    from hybridge.linalg.gpu.fused_batched_solve import solve_batched_fused

    logical, rhs = _systems(4, 28, 22, seed=3)
    logical[2, :, 5] = 0.0
    matrices = cp.asarray(np.ascontiguousarray(logical.transpose(0, 2, 1)))
    solution = cp.asarray(np.ascontiguousarray(rhs.transpose(0, 2, 1)))
    info = cp.zeros(4, dtype=cp.int32)
    solve_batched_fused(matrices, solution, config=(8, 8, 6, variant), info=info)
    flags = cp.asnumpy(info)
    assert flags[2] > 0 and flags[[0, 1, 3]].tolist() == [0, 0, 0]
