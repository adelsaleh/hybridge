"""Tiny device matrix solves and retry policy checks; no time integration."""
from dataclasses import dataclass, replace

import numpy as np
import pytest
import scipy.sparse

from hdgfem.precision import REAL_DTYPE
from hdgfem.linalg.system import LinearSolveConvergenceError, SolveResult, finalize_solve_result
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime import runner


@dataclass
class Assembly:
    data: object
    indices: object
    indptr: object
    rhs: object
    matrix_format: str


@pytest.fixture
def cp():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    return cp


def system(cp, fmt="csr", block_size=7):
    rng = np.random.default_rng(73)
    n = 3 * block_size
    dense = 4*np.eye(n) + 0.1*rng.standard_normal((n, n))
    dense *= np.geomspace(0.01, 100, n)[:, None]
    dense = dense.astype(REAL_DTYPE)
    exact = np.linspace(-1, 2, n, dtype=REAL_DTYPE)
    host = scipy.sparse.csr_matrix(dense)
    if fmt == "bsr":
        host = host.tobsr(blocksize=(block_size, block_size))
    assembly = Assembly(cp.asarray(host.data), cp.asarray(host.indices), cp.asarray(host.indptr),
                        cp.asarray(dense @ exact), fmt)
    rtol = 2e-5 if REAL_DTYPE == np.float32 else 1e-11
    return assembly, dense, exact, rtol


def rejection(cp, assembly):
    norm = float(cp.linalg.norm(assembly.rhs).get())
    target = norm * 1e-12
    result = SolveResult(
        x=None, info=0, preconditioner=None, residual_norm=norm, rtol=1e-12, atol=0.0,
        rhs_norm=norm, relative_residual_norm=1.0, residual_target=target,
        solver_residual_norm=norm, solver_rhs_norm=norm,
        solver_relative_residual_norm=1.0, solver_residual_target=target,
        physical_residual_norm=norm, physical_rhs_norm=norm,
        physical_relative_residual_norm=1.0, physical_residual_target=target,
    )
    return finalize_solve_result(result, backend="pyamgx-device", backend_success=True,
                                 solution_is_finite=True, raise_on_nonconvergence=False), cp.zeros_like(assembly.rhs)


@pytest.mark.parametrize("fmt", ["csr", "bsr"])
def test_device_qr_solves_nonsymmetric_scaled_rows_without_host_solution(cp, fmt):
    from hdgfem.backends.advection_cuda import _solve_reduced_system_cusolver_qr_device_once
    assembly, dense, exact, rtol = system(cp, fmt)
    original = assembly.data.copy()
    result, solution = _solve_reduced_system_cusolver_qr_device_once(
        assembly, check_rtol=rtol, materialize_host_solution=False,
    )
    assert result.converged and result.physical_residual_target_met
    assert result.backend == "cusolver-qr-device"
    assert result.x is None and isinstance(solution, cp.ndarray)
    relative = np.linalg.norm(dense @ cp.asnumpy(solution) - dense @ exact) / np.linalg.norm(dense @ exact)
    assert relative <= rtol
    np.testing.assert_allclose(cp.asnumpy(solution), exact, atol=rtol*5, rtol=rtol*5)
    cp.testing.assert_array_equal(assembly.data, original)


def test_final_direct_retry_recovers_after_all_six_amgx_rejections(cp, monkeypatch):
    import hdgfem.backends.advection_cuda as backend
    assembly, dense, exact, rtol = system(cp, "bsr")
    config = replace(preset_by_key("euler_vortex_gas_localized_p6_50k_dt001_t50_raw_cuda_bsr"),
                     transport_direct_fallback="cusolver-qr")
    options = runner._make_transport_options(config, "zero-flux")
    calls = []
    def reject(assembly, **kwargs):
        calls.append(kwargs)
        return rejection(cp, assembly)
    monkeypatch.setattr(backend, "_solve_reduced_system_amgx_device_once", reject)
    result, solution = backend.solve_reduced_system_amgx_device(
        assembly, retry_attempts=options.amgx_retry_attempts,
        check_rtol=rtol, materialize_host_solution=False,
    )
    assert len(calls) == 6
    assert result.amgx_attempt_count == 7 and result.converged
    assert result.amgx_attempts[-1]["label"] == "direct-qr"
    assert all(not entry["success"] for entry in result.amgx_attempts[:-1])
    assert result.amgx_attempts[-1]["physical_relative_residual"] <= rtol
    np.testing.assert_allclose(cp.asnumpy(solution), exact, atol=rtol*5, rtol=rtol*5)


def test_direct_retry_is_not_called_after_an_accepted_solve(cp, monkeypatch):
    import hdgfem.backends.advection_cuda as backend
    assembly, _, _, rtol = system(cp)
    solved = backend._solve_reduced_system_cusolver_qr_device_once(assembly, check_rtol=rtol)
    monkeypatch.setattr(backend, "_solve_reduced_system_amgx_device_once", lambda *a, **k: solved)
    def forbidden(*args, **kwargs):
        pytest.fail("direct retry must be lazy")
    monkeypatch.setattr(backend, "_solve_reduced_system_cusolver_qr_device_once", forbidden)
    result, _ = backend.solve_reduced_system_amgx_device(
        assembly, retry_attempts=({"backend": "cusolver-qr", "scale_system": False},),
    )
    assert result.converged and result.amgx_attempt_count == 1


@pytest.mark.parametrize("bad_value", [0.0, np.nan])
def test_direct_solver_output_must_pass_physical_residual_and_finite_checks(cp, monkeypatch, bad_value):
    import cupyx.cusolver
    from hdgfem.backends.advection_cuda import _solve_reduced_system_cusolver_qr_device_once
    assembly, _, _, rtol = system(cp)
    monkeypatch.setattr(cupyx.cusolver, "csrlsvqr", lambda A, b, **kw: cp.full_like(b, bad_value))
    result, _ = _solve_reduced_system_cusolver_qr_device_once(assembly, check_rtol=rtol)
    assert not result.converged


def test_singular_device_matrix_is_not_accepted(cp):
    from hdgfem.backends.advection_cuda import _solve_reduced_system_cusolver_qr_device_once
    host = scipy.sparse.csr_matrix(np.array([[1., 0.], [0., 0.]], dtype=REAL_DTYPE))
    assembly = Assembly(cp.asarray(host.data), cp.asarray(host.indices), cp.asarray(host.indptr),
                        cp.array([1., 1.], dtype=REAL_DTYPE), "csr")
    with pytest.raises(UserWarning, match="singular"):
        _solve_reduced_system_cusolver_qr_device_once(assembly)


def test_exhausted_direct_retry_reports_matrix_scales(cp, monkeypatch):
    import hdgfem.backends.advection_cuda as backend
    assembly, dense, _, _ = system(cp, "bsr")
    monkeypatch.setattr(backend, "_solve_reduced_system_amgx_device_once", lambda a, **kw: rejection(cp, a))
    monkeypatch.setattr(backend, "_solve_reduced_system_cusolver_qr_device_once", lambda a, **kw: rejection(cp, a))
    with pytest.raises(LinearSolveConvergenceError) as caught:
        backend.solve_reduced_system_amgx_device(
            assembly, retry_attempts=({"backend": "cusolver-qr"},),
        )
    error = caught.value
    assert error.amgx_attempts[-1]["success"] is False
    assert error.amgx_attempts[-1]["scale_system"] == "none"
    assert error.matrix_diagnostics["matrix_zero_rows"] == 0
    assert error.matrix_diagnostics["matrix_row_l1_min"] == pytest.approx(np.abs(dense).sum(axis=1).min(), rel=1e-6)
    assert error.matrix_diagnostics["matrix_row_l1_max"] == pytest.approx(np.abs(dense).sum(axis=1).max(), rel=1e-6)


@pytest.mark.parametrize("scaling", [False, True])
def test_jacobi_and_fgmres_retries_share_configured_scaling(scaling):
    config = replace(preset_by_key("euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"),
                     transport_scale_system=scaling, transport_direct_fallback="cusolver-qr")
    runner._validate_config(config)
    retries = runner._make_transport_options(config, "zero-flux").amgx_retry_attempts
    assert len(retries) == 6
    assert all(attempt["scale_system"] == scaling for attempt in retries[:-1])
    assert retries[-1]["backend"] == "cusolver-qr"
    assert retries[-1]["scale_system"] is False


@pytest.mark.parametrize("updates", [
    {"transport_retry_policy": "none"}, {"transport_solver": "direct"},
    {"transport_assembly_backend": "numpy"}, {"transport_direct_fallback": "invalid"},
])
def test_invalid_direct_fallback_config_is_rejected(updates):
    config = replace(preset_by_key("euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"),
                     transport_direct_fallback="cusolver-qr")
    with pytest.raises(ValueError, match="direct"):
        runner._validate_config(replace(config, **updates))
