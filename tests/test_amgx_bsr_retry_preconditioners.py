"""AMGX retry compatibility on tiny BSR matrices; no mesh or time integration."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse


@pytest.mark.parametrize("scale_system", (False, True), ids=("unscaled", "scaled"))
@pytest.mark.parametrize("block_size", range(2, 11))
@pytest.mark.parametrize("retry_index", (0, 1), ids=("l1", "block-jacobi"))
def test_native_amgx_bsr_retry_preconditioners(block_size, retry_index, scale_system) -> None:
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pyamgx")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")

    from hdgfem.linalg.amgx.device_solver import _solve_reduced_system_amgx_device_once
    from hdgfem.runtime.precision import REAL_DTYPE
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.configuration import _make_transport_options

    retry = _make_transport_options(
        preset_by_key("euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"), "zero-flux"
    ).amgx_retry_attempts[retry_index]
    # Four nonsymmetric coupled blocks, at most 40 scalar unknowns. These cover
    # all face block dimensions requested by the runner, including 7x7.
    rng = np.random.default_rng(42)
    diagonal = 4.0 * np.eye(block_size) + 0.1 * rng.standard_normal((block_size, block_size))
    diagonal *= np.geomspace(0.5, 2.0, block_size)[:, None]
    dense = np.kron(np.eye(4), diagonal)
    dense += np.kron(np.diag(np.full(3, -0.2), 1), np.eye(block_size))
    dense += np.kron(np.diag(np.full(3, 0.1), -1), np.eye(block_size))
    dense = dense.astype(REAL_DTYPE)
    host = scipy.sparse.bsr_matrix(dense, blocksize=(block_size, block_size))
    exact = np.linspace(-0.75, 1.0, dense.shape[0], dtype=REAL_DTYPE)
    assembly = SimpleNamespace(
        matrix_format="bsr", data=cp.asarray(host.data),
        indices=cp.asarray(host.indices, dtype=cp.int32),
        indptr=cp.asarray(host.indptr, dtype=cp.int32), rhs=cp.asarray(dense @ exact),
    )
    rtol = 2.0e-5 if np.dtype(REAL_DTYPE).itemsize == 4 else 1.0e-10
    retry["config"]["solver"]["tolerance"] = rtol / 10.0
    result, solution = _solve_reduced_system_amgx_device_once(
        assembly, config=retry["config"], tolerance=rtol / 10.0, check_rtol=rtol,
        maxiter=100, scale_system=scale_system,
        scalarize_bsr=retry["scalarize_bsr"], materialize_host_solution=False,
    )

    assert result.converged and result.physical_residual_target_met
    assert result.amgx_bsr_scalarized is False
    assert result.x is None
    assert isinstance(solution, cp.ndarray)
    actual = cp.asnumpy(solution)
    assert np.linalg.norm(dense @ actual - dense @ exact) <= rtol * np.linalg.norm(dense @ exact)
    np.testing.assert_allclose(actual, exact, rtol=rtol * 10.0, atol=rtol)
    cp.testing.assert_allclose(assembly.data, host.data, rtol=np.finfo(REAL_DTYPE).eps * 3, atol=0)


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

    from hdgfem.runtime.precision import REAL_DTYPE

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
    from hdgfem.linalg.amgx.device_solver import _solve_reduced_system_amgx_device_once

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


@pytest.mark.parametrize("failure_phase", ("setup", "solve"))
def test_block_jacobi_exception_reaches_real_dilu(
    monkeypatch, unpooled_bsr_system, failure_phase,
) -> None:
    """Inject a recoverable BJ error; the very next attempt must solve via DILU."""
    import hdgfem.transport.cuda as raw_amgx
    import hdgfem.linalg.amgx.device_solver as amgx_device_solver
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.configuration import _make_transport_options

    cp, host, _, exact, assembly, rtol = unpooled_bsr_system(7, block_rows=33)
    options = _make_transport_options(
        preset_by_key("euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"), "zero-flux"
    )
    for config in (options.amgx_config, *(r["config"] for r in options.amgx_retry_attempts)):
        config["device_mem_pool_enabled"] = 0
        config["solver"]["tolerance"] = rtol / 10.0

    original_once = amgx_device_solver._solve_reduced_system_amgx_device_once
    original_method = getattr(amgx_device_solver.PyAMGXCsrDeviceSolver, failure_phase)
    calls = []
    failed_solvers = []

    def fail_block_jacobi(self, *args, **kwargs):
        if self.config_dict["solver"].get("preconditioner", {}).get("solver") == "BLOCK_JACOBI":
            if failure_phase == "setup":
                original_method(self, *args, **kwargs)
            failed_solvers.append(self)
            # Do not deliberately poison the CUDA context: a sticky device
            # memory fault cannot be recovered by any in-process fallback.
            raise RuntimeError(f"injected block-Jacobi {failure_phase} failure")
        return original_method(self, *args, **kwargs)

    def solve_once(current_assembly, **kwargs):
        solver = kwargs["config"]["solver"]
        calls.append((solver["solver"], solver.get("preconditioner", {}).get("solver")))
        if len(calls) <= 2:
            kwargs["maxiter"] = 1  # Force primary and L1 to miss the strict target.
        return original_once(current_assembly, **kwargs)

    monkeypatch.setattr(amgx_device_solver.PyAMGXCsrDeviceSolver, failure_phase, fail_block_jacobi)
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_amgx_device_once", solve_once)
    cache = {}
    try:
        result, solution = amgx_device_solver.solve_reduced_system_amgx_device(
            assembly, config=options.amgx_config, retry_attempts=options.amgx_retry_attempts,
            retry_solver_cache=cache, tolerance=rtol / 10.0, check_rtol=rtol,
            atol=rtol / 10.0, maxiter=100, initial_guess=cp.ones_like(assembly.rhs),
            materialize_host_solution=False,
        )
        assert calls == [
            ("BICGSTAB", None), ("PBICGSTAB", "JACOBI_L1"),
            ("PBICGSTAB", "BLOCK_JACOBI"), ("FGMRES", "MULTICOLOR_DILU"),
        ]
        assert len(failed_solvers) == 1 and failed_solvers[0].closed
        assert result.converged and result.physical_residual_target_met
        assert result.amgx_attempt_count == 4
        assert result.amgx_attempts[2]["error"] == f"RuntimeError: injected block-Jacobi {failure_phase} failure"
        assert result.amgx_attempts[3]["label"] == "robust-zero-scaled"
        assert result.amgx_attempts[3]["scalarized_bsr"]
        np.testing.assert_allclose(cp.asnumpy(solution), exact, rtol=10 * rtol, atol=rtol)
        cp.testing.assert_allclose(assembly.data, host.data, rtol=rtol, atol=rtol)
    finally:
        for solver in cache.values():
            solver.close(suppress_errors=True)
