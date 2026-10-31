"""AMGX retry compatibility on tiny BSR matrices; no mesh or time integration."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse




@pytest.fixture
def unpooled_bsr_system():
    """Build sparse algebra fixtures without creating a mesh or stepping time."""
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pyamgx")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")

    from hybridge.runtime.precision import REAL_DTYPE

    def build(block_size, block_rows=513):
        # More than 256 rows exposes the old large-block BJ scratch overflow.
        # Keep this sparse: the largest fixture has only 5,130 unknowns.
        rng = np.random.default_rng(104)
        diagonal = 4.0 * np.eye(block_size) + 0.1 * rng.standard_normal((block_size, block_size))
        diagonal *= np.geomspace(0.5, 2.0, block_size)[:, None]
        row_scale = np.linspace(0.8, 1.2, block_rows)
        host = scipy.sparse.kron(scipy.sparse.diags(row_scale), diagonal, format="bsr")
        offdiag = scipy.sparse.diags(
            [np.full(block_rows - 1, 0.1), np.full(block_rows - 1, -0.2)],
            [-1, 1], shape=(block_rows, block_rows),
        )
        host += scipy.sparse.kron(offdiag, np.eye(block_size), format="bsr")
        host = host.astype(REAL_DTYPE).tobsr(blocksize=(block_size, block_size))
        exact = np.linspace(-0.75, 1.0, host.shape[0], dtype=REAL_DTYPE)
        assembly = SimpleNamespace(
            matrix_format="bsr", data=cp.asarray(host.data),
            indices=cp.asarray(host.indices, dtype=cp.int32),
            indptr=cp.asarray(host.indptr, dtype=cp.int32), rhs=cp.asarray(host @ exact),
        )
        blocks = (row_scale[:, None, None] * diagonal).astype(REAL_DTYPE)
        rtol = 2.0e-5 if np.dtype(REAL_DTYPE).itemsize == 4 else 1.0e-10
        return cp, host, blocks, exact, assembly, rtol

    return build


@pytest.mark.parametrize("block_size", range(6, 11))
@pytest.mark.parametrize("backend", ("legacy", "cusparse_generic"))
def test_block_jacobi_large_block_update(unpooled_bsr_system, block_size, backend) -> None:
    """Check Dinv and both BSR multiplies, with allocations visible to memcheck."""
    from hybridge.linalg.amgx.device_solver import _solve_reduced_system_amgx_device_once

    cp, host, blocks, exact, assembly, rtol = unpooled_bsr_system(block_size)
    guess = np.cos(np.arange(exact.size) * 0.13).astype(exact.dtype)
    residual = host @ exact - host @ guess
    expected = guess + 0.8 * np.linalg.solve(
        blocks, residual.reshape(-1, block_size, 1)
    ).reshape(-1)
    config = {
        "config_version": 2, "exception_handling": 1, "determinism_flag": 1,
        # Pooling hides writes into neighboring AMGX suballocations from
        # compute-sanitizer. This setting belongs only to the regression.
        "device_mem_pool_enabled": 0,
        "solver": {
            "solver": "BLOCK_JACOBI", "bsr_spmv_backend": backend,
            "max_iters": 1, "relaxation_factor": 0.8,
            "convergence": "RELATIVE_INI_CORE", "tolerance": rtol / 10.0,
        },
    }
    _, solution = _solve_reduced_system_amgx_device_once(
        assembly, config=config, initial_guess=cp.asarray(guess), maxiter=1,
        check_rtol=rtol, scale_system=False, scalarize_bsr=False,
        raise_on_nonconvergence=False, materialize_host_solution=False,
    )
    # One BJ update should equal x + omega*D^{-1}(b-A*x), even though it has
    # not converged. A Krylov solve alone could conceal an incorrect Dinv.
    np.testing.assert_allclose(cp.asnumpy(solution), expected, rtol=rtol, atol=rtol)
    cp.testing.assert_array_equal(assembly.data, host.data)
