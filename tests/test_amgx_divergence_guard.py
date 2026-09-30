"""Exercise the already-built native guard using host arrays; no JIT or build.

Requires the patched AMGX library on LD_LIBRARY_PATH. No mesh/time integration.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import ctypes
import re

import numpy as np
import pytest
import scipy.sparse

from hdgfem.runtime.precision import AMGX_MODE, REAL_DTYPE


@pytest.fixture(scope="module")
def amgx():
    pytest.importorskip("pyamgx")
    from hdgfem.linalg.amgx.host import initialize_pyamgx_once
    return initialize_pyamgx_once()


@contextmanager
def native_solver(amgx, dense, *, block_size=1, **controls):
    # All arrays are on the host; only the existing AMGX binary runs on device.
    matrix = scipy.sparse.bsr_matrix(np.asarray(dense, dtype=REAL_DTYPE),
                                     blocksize=(block_size, block_size))
    config = {"config_version": 2, "exception_handling": 1, "determinism_flag": 1,
              "solver": {
                  "solver": "BLOCK_JACOBI", "relaxation_factor": 4.0,
                  "bsr_spmv_backend": "cusparse_generic",
                  "convergence": "RELATIVE_INI_CORE", "tolerance": 1e-11,
                  "norm": "L2", "max_iters": 1500,
                  "monitor_residual": 1, "store_res_history": 1,
                  "print_solve_stats": 1, "print_solve_stats_interval": 10,
                  "rel_div_tolerance": 1000.0, "divergence_patience": 5,
                  "divergence_grace_iters": 10,
                  **controls,
              }}
    objects = []
    def own(obj):
        objects.append(obj)
        return obj
    try:
        cfg = own(amgx.Config().create_from_dict(config))
        resources = own(amgx.Resources().create_simple(cfg))
        mat = own(amgx.Matrix().create(resources, mode=AMGX_MODE))
        mat.upload(matrix.indptr, matrix.indices, matrix.data,
                   block_dims=(block_size, block_size),
                   shape=(dense.shape[0] // block_size,) * 2)
        b = own(amgx.Vector().create(resources, mode=AMGX_MODE))
        x = own(amgx.Vector().create(resources, mode=AMGX_MODE))
        solver = own(amgx.Solver().create(resources, cfg, mode=AMGX_MODE))
        solver.setup(mat)
        def solve(rhs, guess=None):
            rhs = np.ascontiguousarray(rhs, dtype=REAL_DTYPE)
            initial = np.zeros_like(rhs) if guess is None else np.ascontiguousarray(guess, dtype=REAL_DTYPE)
            b.upload_raw(rhs.ctypes.data, rhs.size // block_size, block_size)
            x.upload_raw(initial.ctypes.data, initial.size // block_size, block_size)
            solver.solve(b, x, zero_initial_guess=guess is None)
            answer = np.empty_like(rhs)
            x.download_raw(answer.ctypes.data)
            count = solver.iterations_number
            components = 1 if controls.get("use_scalar_norm", 0) else block_size
            history = [np.linalg.norm([solver.get_residual(i, j) for j in range(components)])
                       for i in range(count + 1)]
            ctypes.CDLL(None).fflush(None)
            return SimpleNamespace(x=answer, status=solver.status, iterations=count,
                                   history=history)
        yield solve
    finally:
        for obj in reversed(objects):
            obj.destroy()
        ctypes.CDLL(None).fflush(None)


@pytest.mark.parametrize("block_size", [1, 7])
def test_explosive_growth_stops_at_fifteen_iterations_and_prints_final_row(amgx, capfd, block_size):
    dense = np.eye(2 * block_size, dtype=REAL_DTYPE)
    rhs = np.ones(dense.shape[0], dtype=REAL_DTYPE)
    with native_solver(amgx, dense, block_size=block_size) as solve:
        result = solve(rhs)
    assert result.status == "diverged"
    assert result.iterations == 15
    # For A=I and omega=4, every complete Jacobi step multiplies r by -3.
    expected = np.linalg.norm(rhs.astype(np.float64)) * 3**15
    actual = np.linalg.norm(rhs.astype(np.float64) - dense.astype(np.float64) @ result.x)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    np.testing.assert_allclose(result.history[-1], actual, rtol=2e-6)
    output = capfd.readouterr().out
    assert "early exit: confirmed residual growth at iteration 15" in output
    assert "explicit b-A*x=" in output
    rows = re.findall(r"^\s*(Ini|\d+)\s+\d+\.\d+\s+\d+\.\d+\s", output, re.M)
    assert rows == ["Ini", "0", "10", "14"]


def test_disabled_guard_reaches_limit_and_prints_last_unsampled_row(amgx, capfd):
    with native_solver(amgx, np.eye(2), rel_div_tolerance=-1, max_iters=24) as solve:
        result = solve(np.ones(2))
    assert result.status == "not_converged"
    assert result.iterations == 24
    output = capfd.readouterr().out
    assert "early exit:" not in output
    assert re.search(r"^\s*23\s+\d+\.\d+", output, re.M)


@pytest.mark.parametrize("grace,patience,expected_status,expected_iters", [
    (0, 1, "diverged", 1), (1, 1, "success", 2), (0, 2, "success", 2),
])
def test_transient_growth_obeys_grace_and_patience(amgx, grace, patience, expected_status, expected_iters):
    # Jacobi has one huge residual spike followed by an exact solution.
    dense = np.array([[1, -10000], [0, 1]], dtype=REAL_DTYPE)
    with native_solver(amgx, dense, relaxation_factor=1.0,
                       divergence_grace_iters=grace, divergence_patience=patience) as solve:
        result = solve(np.ones(2))
    assert result.status == expected_status
    assert result.iterations == expected_iters
    if expected_status == "success":
        np.testing.assert_array_equal(dense @ result.x, np.ones(2))


def test_large_initial_residual_can_converge(amgx):
    tol = 1e-5 if REAL_DTYPE == np.float32 else 1e-10
    rhs = np.array([1e10, -2e10], dtype=REAL_DTYPE)
    with native_solver(amgx, np.eye(2), relaxation_factor=0.5, tolerance=tol) as solve:
        result = solve(rhs)
    assert result.status == "success"
    assert np.linalg.norm((result.x - rhs).astype(np.float64)) <= tol * np.linalg.norm(rhs)


@pytest.mark.parametrize("solver_name", ["BICGSTAB", "PBICGSTAB"])
def test_krylov_growth_is_checked_after_a_complete_iteration(amgx, solver_name):
    # A nearly skew-symmetric operator produces a finite BiCGStab spike.
    dense = np.array([[1e-4, 1.0], [-1.0, 1e-4]], dtype=REAL_DTYPE)
    preconditioner = {"solver": "BLOCK_JACOBI", "max_iters": 1,
                      "relaxation_factor": 1.0, "monitor_residual": 0}
    with native_solver(amgx, dense, solver=solver_name, preconditioner=preconditioner,
                       divergence_grace_iters=0, divergence_patience=1) as solve:
        result = solve(np.ones(2))
    assert result.status == "diverged"
    assert result.iterations == 1
    true_residual = np.linalg.norm(np.ones(2) - dense.astype(np.float64) @ result.x)
    assert true_residual > 1000 * np.sqrt(2)
    np.testing.assert_allclose(result.history[-1], true_residual, rtol=2e-5)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_initial_residual_exits_without_iterations(amgx, bad):
    with native_solver(amgx, np.eye(2)) as solve:
        result = solve(np.array([bad, 1]))
    assert result.status == "diverged"
    assert result.iterations == 0


def test_nonfinite_recurrence_exits_during_startup(amgx):
    # r.T A r == 0 gives plain BiCGStab non-finite recurrence arithmetic.
    with native_solver(amgx, np.diag([1.0, -1.0]), solver="BICGSTAB") as solve:
        result = solve(np.ones(2))
    assert result.status == "diverged"
    assert result.iterations == 1


def test_guard_history_resets_when_a_solver_is_reused(amgx):
    with native_solver(amgx, np.eye(2)) as solve:
        first = solve(np.ones(2))
        exact = solve(np.zeros(2))
        second = solve(np.full(2, 1e-6))
    assert first.status == second.status == "diverged"
    assert first.iterations == second.iterations == 15
    assert exact.status == "success" and exact.iterations == 0


@pytest.mark.parametrize("solver_name", ["BICGSTAB", "PBICGSTAB", "FGMRES", "GMRES"])
def test_guard_preserves_successful_krylov_solves(amgx, solver_name):
    dense = np.diag(np.linspace(1, 2, 12)) + np.diag(np.full(11, 0.2), 1)
    exact = np.linspace(-1, 1, 12).astype(REAL_DTYPE)
    rhs = (dense @ exact).astype(REAL_DTYPE)
    tol = 1e-5 if REAL_DTYPE == np.float32 else 1e-10
    controls = {"solver": solver_name, "tolerance": tol / 10,
                "gmres_n_restart": 4, "max_iters": 100,
                "preconditioner": {"solver": "BLOCK_JACOBI", "max_iters": 1,
                                   "relaxation_factor": 1.0, "monitor_residual": 0}}
    if solver_name in ("GMRES", "FGMRES"):
        controls["use_scalar_norm"] = 1
    with native_solver(amgx, dense, **controls) as solve:
        result = solve(rhs)
    assert result.status == "success"
    assert np.linalg.norm(dense @ result.x - rhs) <= tol * np.linalg.norm(rhs)
