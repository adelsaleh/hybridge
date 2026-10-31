"""Tiny device matrix solves and retry policy checks; no time integration."""
from dataclasses import dataclass, replace

import numpy as np
import pytest
import scipy.sparse

from hybridge.runtime.precision import REAL_DTYPE
from hybridge.linalg.results import (
    LinearSolveConvergenceError,
    SolveResult,
    finalize_solve_result,
)


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
    from hybridge.linalg.amgx.device_solver import (
            _solve_reduced_system_cusolver_qr_device_once,
        )
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




def test_direct_retry_is_not_called_after_an_accepted_solve(cp, monkeypatch):
    import hybridge.transport.cuda as backend
    import hybridge.linalg.amgx.device_solver as amgx_device_solver
    assembly, _, _, rtol = system(cp)
    solved = amgx_device_solver._solve_reduced_system_cusolver_qr_device_once(assembly, check_rtol=rtol)
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_amgx_device_once", lambda *a, **k: solved)
    def forbidden(*args, **kwargs):
        pytest.fail("direct retry must be lazy")
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_cusolver_qr_device_once", forbidden)
    result, _ = amgx_device_solver.solve_reduced_system_amgx_device(
        assembly, retry_attempts=({"backend": "cusolver-qr", "scale_system": False},),
    )
    assert result.converged and result.amgx_attempt_count == 1


@pytest.mark.parametrize("bad_value", [0.0, np.nan])
def test_direct_solver_output_must_pass_physical_residual_and_finite_checks(cp, monkeypatch, bad_value):
    import cupyx.cusolver
    from hybridge.linalg.amgx.device_solver import (
            _solve_reduced_system_cusolver_qr_device_once,
        )
    assembly, _, _, rtol = system(cp)
    monkeypatch.setattr(cupyx.cusolver, "csrlsvqr", lambda A, b, **kw: cp.full_like(b, bad_value))
    result, _ = _solve_reduced_system_cusolver_qr_device_once(assembly, check_rtol=rtol)
    assert not result.converged


def test_singular_device_matrix_is_not_accepted(cp):
    from hybridge.linalg.amgx.device_solver import (
            _solve_reduced_system_cusolver_qr_device_once,
        )
    host = scipy.sparse.csr_matrix(np.array([[1., 0.], [0., 0.]], dtype=REAL_DTYPE))
    assembly = Assembly(cp.asarray(host.data), cp.asarray(host.indices), cp.asarray(host.indptr),
                        cp.array([1., 1.], dtype=REAL_DTYPE), "csr")
    with pytest.raises(UserWarning, match="singular"):
        _solve_reduced_system_cusolver_qr_device_once(assembly)


def test_exhausted_direct_retry_reports_matrix_scales(cp, monkeypatch):
    import hybridge.transport.cuda as backend
    import hybridge.linalg.amgx.device_solver as amgx_device_solver
    assembly, dense, _, _ = system(cp, "bsr")
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_amgx_device_once", lambda a, **kw: rejection(cp, a))
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_cusolver_qr_device_once", lambda a, **kw: rejection(cp, a))
    with pytest.raises(LinearSolveConvergenceError) as caught:
        amgx_device_solver.solve_reduced_system_amgx_device(
            assembly, retry_attempts=({"backend": "cusolver-qr"},),
        )
    error = caught.value
    assert error.amgx_attempts[-1]["success"] is False
    assert error.amgx_attempts[-1]["scale_system"] == "none"
    assert error.matrix_diagnostics["matrix_zero_rows"] == 0
    assert error.matrix_diagnostics["matrix_row_l1_min"] == pytest.approx(np.abs(dense).sum(axis=1).min(), rel=1e-6)
    assert error.matrix_diagnostics["matrix_row_l1_max"] == pytest.approx(np.abs(dense).sum(axis=1).max(), rel=1e-6)
