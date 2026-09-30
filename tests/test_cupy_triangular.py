"""Small nonsymmetric diagnostics for explicit LU permutations and GPU reuse."""
import os

import numpy as np
import pytest

from hdgfem.backends.cupy_triangular import ReusableCuPyLUSolve, superlu_gather_indices
from scripts.guiding_center.poisson import benchmark_scipy_lu_gpu as benchmark


@pytest.fixture(params=["splu", "spilu-nodrop"])
def explicit_factors(request, tmp_path):
    capture, cache = tmp_path / "capture", tmp_path / "factors"
    benchmark.make_smoke_capture(capture)
    cache.mkdir()
    matrix, rhs, reference, _, hashes, _ = benchmark.load_capture(capture)
    args = benchmark.build_parser().parse_args([
        "--output", str(tmp_path / "output"), "--factor-cache", str(cache),
        "--factor-driver", request.param,
    ])
    benchmark.factor_cpu(args, matrix, hashes, lambda *args, **kwargs: None)
    return matrix, rhs, reference, benchmark.load_factor_cache(cache, matrix.shape)


def test_explicit_factors_preserve_original_nonsymmetric_operator(explicit_factors):
    matrix, _, _, (lower, upper, perm_r, perm_c) = explicit_factors
    rows, columns = superlu_gather_indices(perm_r, perm_c)
    assert not np.array_equal(rows, np.arange(matrix.shape[0]))
    assert not np.array_equal(columns, np.arange(matrix.shape[0]))
    np.testing.assert_allclose(
        (lower @ upper).toarray(), matrix.toarray()[rows][:, np.argsort(columns)],
        rtol=2e-13, atol=2e-13,
    )


@pytest.mark.skipif(os.environ.get("HDGFEM_TEST_GPU_LU") != "1",
                    reason="opt-in GPU diagnostic; existing kernels only")
@pytest.mark.parametrize("candidate", benchmark.METHODS)
def test_gpu_lu_reuses_analysis_with_changing_rhs(explicit_factors, candidate):
    from hdgfem.runtime.optional import require_cupy_device
    from hdgfem.backends.cupy import scipy_csr_to_cupy
    from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only

    cp = require_cupy_device()
    matrix, _, _, (lower, upper, perm_r, perm_c) = explicit_factors
    rng = np.random.default_rng(82)
    with kernel_cache_only(True):
        l_gpu, u_gpu = scipy_csr_to_cupy(lower), scipy_csr_to_cupy(upper)
        method, graph = candidate.removesuffix("-graph"), candidate.endswith("-graph")
        with ReusableCuPyLUSolve(l_gpu, u_gpu, perm_r, perm_c, method=method, graph=graph) as solver:
            analysis_ids = tuple(id(item) for item in solver.triangular)
            output = cp.empty(matrix.shape[0], dtype=cp.float64)
            for repetition in range(3):
                exact = rng.standard_normal(matrix.shape[0])
                rhs = cp.asarray(matrix @ exact)
                # Exercise the retained output, caller-owned output, and RHS alias.
                out = (None, output, rhs)[repetition]
                result = solver.solve(rhs, out=out)
                if out is not None:
                    assert result is out
                np.testing.assert_allclose(cp.asnumpy(result), exact, rtol=2e-13, atol=2e-13)
                assert analysis_ids == tuple(id(item) for item in solver.triangular)
        solver.close()  # Closing twice is harmless; solving after close is not.
        with pytest.raises(RuntimeError, match="closed"):
            solver.solve(rhs)
