"""hybridge.linalg.direct."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
import threading
import time
from typing import Any, Literal
from hybridge.linalg.results import (
    LinearSolveError,
    SolveResult,
    _validate_finite_array,
    _validate_solver_controls,
    finalize_solve_result,
    residual_diagnostics,
    restricted_residual_diagnostics,
)
from numpy.typing import NDArray
from hybridge.runtime.precision import REAL_DTYPE
from scipy.sparse.linalg import spsolve


_PYPARDISO_LOCK = threading.RLock()


_PYPARDISO_SOLVERS: dict[tuple[int, int], Any] = {}


def _import_pypardiso():
    """Import the optional oneMKL PARDISO adapter only when requested."""
    try:
        import pypardiso
    except ImportError as exc:
        raise ImportError(
            "solver='pypardiso' requires the optional pypardiso runtime; "
            "install hybridge[pardiso] or pypardiso directly"
        ) from exc
    return pypardiso


def solve_direct_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    diagnostic_rows: NDArray | None = None,
    diagnostic_label: str | None = None,
    raise_on_nonconvergence: bool = False,
) -> SolveResult:
    """Solve an already assembled sparse system with ``spsolve``."""
    _validate_solver_controls(rtol=rtol, atol=atol)
    matrix = matrix.tocsr()
    rhs = np.asarray(rhs, dtype=REAL_DTYPE)
    if matrix.shape != (rhs.size, rhs.size):
        raise ValueError(f"matrix must have shape ({rhs.size}, {rhs.size}), got {matrix.shape}")
    _validate_finite_array(np.asarray(matrix.data), "matrix data")
    _validate_finite_array(rhs, "rhs")

    start = time.time()
    x = spsolve(matrix.tocsr(), rhs)
    solve_elapsed_seconds = time.time() - start

    physical_residual = matrix @ x - rhs
    physical_residual_norm = float(np.linalg.norm(physical_residual))
    physical_rhs_norm, physical_relative_residual_norm, physical_residual_target = residual_diagnostics(
        physical_residual_norm, rhs, rtol=rtol, atol=atol
    )
    diagnostic_residual_norm, diagnostic_rhs_norm, diagnostic_relative_residual_norm, diagnostic_residual_target = (
        restricted_residual_diagnostics(physical_residual, rhs, diagnostic_rows, rtol=rtol, atol=atol)
    )

    result = SolveResult(
        x=np.asarray(x),
        residual_norm=physical_residual_norm,
        rhs_norm=physical_rhs_norm,
        relative_residual_norm=physical_relative_residual_norm,
        residual_target=physical_residual_target,
        solver_residual_norm=physical_residual_norm,
        solver_rhs_norm=physical_rhs_norm,
        solver_relative_residual_norm=physical_relative_residual_norm,
        solver_residual_target=physical_residual_target,
        physical_residual_norm=physical_residual_norm,
        physical_rhs_norm=physical_rhs_norm,
        physical_relative_residual_norm=physical_relative_residual_norm,
        physical_residual_target=physical_residual_target,
        diagnostic_residual_label=diagnostic_label,
        diagnostic_residual_norm=diagnostic_residual_norm,
        diagnostic_rhs_norm=diagnostic_rhs_norm,
        diagnostic_relative_residual_norm=diagnostic_relative_residual_norm,
        diagnostic_residual_target=diagnostic_residual_target,
        rtol=rtol,
        atol=atol,
        info=0,
        preconditioner=None,
        total_elapsed_seconds=solve_elapsed_seconds,
        scale_elapsed_seconds=0.0,
        preconditioner_elapsed_seconds=0.0,
        solve_elapsed_seconds=solve_elapsed_seconds,
    )
    return finalize_solve_result(
        result,
        backend="scipy-direct",
        backend_info=0,
        backend_success=True,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )


def prepare_pypardiso_spd_matrix(matrix) -> scipy.sparse.csr_matrix:
    """Validate symmetry and return upper CSR storage for PARDISO mtype=2.

    Accept a full real square matrix, not an already truncated triangle. This
    shared preparation can be timed once by callers that explicitly retain a
    PyPardiso Cholesky factorization. Positive definiteness is checked by the
    subsequent native factorization, not established by this symmetry check.
    """
    if np.issubdtype(matrix.dtype, np.complexfloating):
        raise TypeError("pypardiso supports real-valued systems only")
    matrix = scipy.sparse.csr_matrix(matrix, dtype=REAL_DTYPE)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("PARDISO SPD input must be square")
    matrix.sum_duplicates()
    matrix.sort_indices()
    _validate_finite_array(matrix.data, "matrix data")
    asymmetry = matrix - matrix.T
    scale = 0.0 if matrix.nnz == 0 else float(np.max(np.abs(matrix.data)))
    defect = 0.0 if asymmetry.nnz == 0 else float(np.max(np.abs(asymmetry.data)))
    tolerance = 1.0e-11 * max(1.0, scale)
    if defect > tolerance:
        raise ValueError(
            "matrix_type='spd' requires a symmetric matrix; "
            f"max_abs_asymmetry={defect:.3e}, tolerance={tolerance:.3e}"
        )
    upper = scipy.sparse.triu(matrix, format="csr")
    upper.sum_duplicates()
    upper.sort_indices()
    return upper


def solve_pypardiso_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    diagnostic_rows: NDArray | None = None,
    diagnostic_label: str | None = None,
    raise_on_nonconvergence: bool = False,
    matrix_type: Literal["nonsymmetric", "spd"] = "nonsymmetric",
) -> SolveResult:
    """Solve a real CSR system with the optional oneMKL PARDISO backend.

    PyPardiso owns process-wide solver instances and reuses their most recent
    factorizations. Calls are serialized because those native instances are not
    safe for concurrent solves. The SPD matrix type validates symmetry and
    passes only the upper triangle to PARDISO mtype=2.
    """
    _validate_solver_controls(rtol=rtol, atol=atol)
    rhs = np.asarray(rhs)
    if np.issubdtype(matrix.dtype, np.complexfloating) or np.iscomplexobj(rhs):
        raise TypeError("pypardiso supports real-valued systems only")
    if rhs.ndim != 1:
        raise ValueError(f"rhs must be one-dimensional, got shape {rhs.shape}")
    matrix = scipy.sparse.csr_matrix(matrix, dtype=REAL_DTYPE)
    matrix.sum_duplicates()
    matrix.sort_indices()
    rhs = np.asarray(rhs, dtype=REAL_DTYPE)
    if matrix.shape != (rhs.size, rhs.size):
        raise ValueError(f"matrix must have shape ({rhs.size}, {rhs.size}), got {matrix.shape}")
    _validate_finite_array(np.asarray(matrix.data), "matrix data")
    _validate_finite_array(rhs, "rhs")

    normalized_matrix_type = str(matrix_type).lower().replace("_", "-")
    if normalized_matrix_type not in {"nonsymmetric", "spd"}:
        raise ValueError("matrix_type must be 'nonsymmetric' or 'spd'")

    native_matrix = matrix
    pardiso_mtype = 11
    backend = "pypardiso-direct"
    if normalized_matrix_type == "spd":
        native_matrix = prepare_pypardiso_spd_matrix(matrix)
        pardiso_mtype = 2
        backend = "pypardiso-spd"

    pypardiso = _import_pypardiso()
    start = time.perf_counter()
    with _PYPARDISO_LOCK:
        active_solver = pypardiso.ps
        try:
            if pardiso_mtype == 11:
                x = pypardiso.spsolve(native_matrix, rhs)
            else:
                cache_key = (id(pypardiso), pardiso_mtype)
                active_solver = _PYPARDISO_SOLVERS.get(cache_key)
                if active_solver is None:
                    active_solver = pypardiso.PyPardisoSolver(mtype=pardiso_mtype)
                    _PYPARDISO_SOLVERS[cache_key] = active_solver
                x = pypardiso.spsolve(native_matrix, rhs, solver=active_solver)
        except Exception as exc:
            try:
                active_solver.free_memory(everything=True)
            except Exception:
                pass
            if pardiso_mtype != 11:
                _PYPARDISO_SOLVERS.pop((id(pypardiso), pardiso_mtype), None)
            raise LinearSolveError(f"pypardiso solve failed: {exc}") from exc
    solve_elapsed_seconds = time.perf_counter() - start

    physical_residual = matrix @ x - rhs
    physical_residual_norm = float(np.linalg.norm(physical_residual))
    physical_rhs_norm, physical_relative_residual_norm, physical_residual_target = residual_diagnostics(
        physical_residual_norm, rhs, rtol=rtol, atol=atol
    )
    diagnostic_residual_norm, diagnostic_rhs_norm, diagnostic_relative_residual_norm, diagnostic_residual_target = (
        restricted_residual_diagnostics(physical_residual, rhs, diagnostic_rows, rtol=rtol, atol=atol)
    )

    result = SolveResult(
        x=np.asarray(x),
        residual_norm=physical_residual_norm,
        rhs_norm=physical_rhs_norm,
        relative_residual_norm=physical_relative_residual_norm,
        residual_target=physical_residual_target,
        solver_residual_norm=physical_residual_norm,
        solver_rhs_norm=physical_rhs_norm,
        solver_relative_residual_norm=physical_relative_residual_norm,
        solver_residual_target=physical_residual_target,
        physical_residual_norm=physical_residual_norm,
        physical_rhs_norm=physical_rhs_norm,
        physical_relative_residual_norm=physical_relative_residual_norm,
        physical_residual_target=physical_residual_target,
        diagnostic_residual_label=diagnostic_label,
        diagnostic_residual_norm=diagnostic_residual_norm,
        diagnostic_rhs_norm=diagnostic_rhs_norm,
        diagnostic_relative_residual_norm=diagnostic_relative_residual_norm,
        diagnostic_residual_target=diagnostic_residual_target,
        rtol=rtol,
        atol=atol,
        info=0,
        preconditioner=None,
        total_elapsed_seconds=solve_elapsed_seconds,
        scale_elapsed_seconds=0.0,
        preconditioner_elapsed_seconds=0.0,
        solve_elapsed_seconds=solve_elapsed_seconds,
    )
    return finalize_solve_result(
        result,
        backend=backend,
        backend_info=0,
        backend_success=True,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )


def clear_pypardiso_cache(*, everything: bool = True) -> None:
    """Release all optional-backend factorizations and native memory."""
    pypardiso = _import_pypardiso()
    with _PYPARDISO_LOCK:
        solvers = [pypardiso.ps, *_PYPARDISO_SOLVERS.values()]
        first_error = None
        for solver in solvers:
            try:
                solver.free_memory(everything=everything)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if everything:
            _PYPARDISO_SOLVERS.clear()
        if first_error is not None:
            raise LinearSolveError(
                f"pypardiso cache cleanup failed: {first_error}"
            ) from first_error
