"""hybridge.linalg.results."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import scipy.sparse.linalg
import time
from typing import Any, Literal, Mapping
from hybridge.runtime.precision import REAL_DTYPE
from dataclasses import dataclass

try:
    from scipy.sparse import _sparsetools
except ImportError:  # Fall back to a vectorized implementation on other SciPy builds.
    _sparsetools = None



def _verbosity_level(verbose: bool | int) -> int:
    """Normalize solver verbosity flags to levels 0 through 3."""
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    return min(3, max(0, int(verbose)))


def _solver_print(verbose: bool | int, level: int, message: str, *args) -> None:
    """Print formatted solver diagnostics when ``verbose`` reaches ``level``."""
    if _verbosity_level(verbose) >= level:
        print(message % args if args else message, flush=True)


def _validate_finite_array(values: NDArray, label: str) -> None:
    """Reject non-finite solver inputs before backend setup."""
    try:
        finite = bool(np.all(np.isfinite(values)))
    except TypeError as exc:
        raise TypeError(f"{label} must contain numeric values") from exc
    if not finite:
        raise ValueError(f"{label} contains non-finite values")


def _validate_solver_controls(
    *,
    rtol: float,
    atol: float,
    maxiter: int | None = None,
    restart: int | None = None,
) -> None:
    """Validate common tolerance and iteration controls."""
    if not np.isfinite(rtol) or float(rtol) < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    if not np.isfinite(atol) or float(atol) < 0.0:
        raise ValueError(f"atol must be finite and non-negative, got {atol}")
    if maxiter is not None and int(maxiter) <= 0:
        raise ValueError(f"maxiter must be positive when provided, got {maxiter}")
    if restart is not None and int(restart) <= 0:
        raise ValueError(f"restart must be positive when provided, got {restart}")


def _normalize_diagnostic_rows(diagnostic_rows: NDArray | None, system_size: int) -> NDArray | None:
    """Validate optional rows used for row-restricted residual diagnostics."""
    if diagnostic_rows is None:
        return None
    rows = np.asarray(diagnostic_rows)
    if rows.dtype == bool:
        if rows.shape != (system_size,):
            raise ValueError(f"boolean diagnostic_rows must have shape ({system_size},), got {rows.shape}")
        return rows
    if rows.ndim != 1:
        raise ValueError("diagnostic_rows must be a one-dimensional integer array or boolean mask")
    if rows.size == 0:
        return rows.astype(np.intp, copy=False)
    if rows.min() < 0 or rows.max() >= system_size:
        raise ValueError("diagnostic_rows contains row indices outside the system")
    return rows.astype(np.intp, copy=False)


SolveStatus = Literal["converged", "not-converged", "diverged", "stagnated", "non-finite"]


class LinearSolveError(RuntimeError):
    """Base class for configured linear-solve failures."""

    def __init__(self, message: str, *, result: "SolveResult | None" = None):
        """Initialize the instance."""
        super().__init__(message)
        self.result = result


class LinearSolveConvergenceError(LinearSolveError):
    """Raised when a backend does not produce a validated solution."""


class LinearSolveCapacityError(LinearSolveError):
    """Raised when a backend cannot proceed because device capacity is exhausted."""

    def __init__(
            self,
            message: str,
            *,
            backend: str | None = None,
            phase: str | None = None,
            memory: Mapping[str, Any] | None = None,
            result: "SolveResult | None" = None,
    ):
        """Initialize a terminal capacity failure with structured diagnostics."""
        super().__init__(message, result=result)
        self.backend = backend
        self.phase = phase
        self.memory = dict(memory or {})


class KrylovIterationCounter:
    """
    Counts Krylov iterations through SciPy's callback interface.

    For GMRES with callback_type="pr_norm", this counts inner Krylov iterations.
    For BICGSTAB/CG/CGS/MINRES/LGMRES, this counts callback calls.
    """

    def __init__(
            self,
            *,
            matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
            rhs: NDArray | None = None,
            verbose: bool | int = 0,
            solver_name: str = "KRYLOV",
    ):
        """Initialize this object."""
        self.count = 0
        self.matrix = matrix
        self.rhs = rhs
        self.verbose = verbose
        self.solver_name = solver_name
        self.rhs_norm = None if rhs is None else max(float(np.linalg.norm(rhs)), 1.0e-300)
        self.residual_history: list[float] = []
        self.elapsed_seconds = 0.0

    def __call__(self, value):
        """Execute the configured call behavior."""
        callback_start = time.perf_counter()
        self.count += 1
        residual_norm = None
        value_array = np.asarray(value)
        if value_array.ndim == 0:
            # GMRES with callback_type="pr_norm" supplies the preconditioned
            # residual norm directly.
            residual_norm = float(value_array)
        elif self.matrix is not None and self.rhs is not None:
            residual_norm = float(np.linalg.norm(self.matrix @ value_array - self.rhs))
        if residual_norm is not None:
            self.residual_history.append(residual_norm)
            if _verbosity_level(self.verbose) >= 3:
                if self.rhs_norm is not None:
                    rel = residual_norm / self.rhs_norm
                    print(
                        f"  {self.solver_name} iter={self.count} residual={residual_norm:.6e} rel={rel:.6e}",
                        flush=True,
                    )
                else:
                    print(
                        f"  {self.solver_name} iter={self.count} residual={residual_norm:.6e}",
                        flush=True,
                    )
        self.elapsed_seconds += time.perf_counter() - callback_start


@dataclass
class SolveResult:
    """Solution vector and diagnostics returned by :func:`solve_global_system`."""

    x: NDArray | None
    x_device: Any | None = None
    residual_norm: float | None = None
    info: int | None = None
    preconditioner: scipy.sparse.linalg.LinearOperator | None = None

    total_elapsed_seconds: float | None = None
    scale_elapsed_seconds: float | None = None
    preconditioner_elapsed_seconds: float | None = None
    solve_elapsed_seconds: float | None = None
    global_elapsed_seconds: float | None = None
    matrix_assembly_elapsed_seconds: float | None = None
    initial_residual_elapsed_seconds: float | None = None
    callback_elapsed_seconds: float | None = None
    residual_diagnostics_elapsed_seconds: float | None = None

    # Krylov diagnostics
    iteration_count: int | None = None
    initial_residual_norm: float | None = None
    # Backward-compatible names for the residual of the system actually solved.
    # For iterative solves with scale_system=True, these are diagnostics for the
    # diagonally scaled system D^{-1} A x = D^{-1} b, matching SciPy's stopping test.
    rhs_norm: float | None = None
    relative_residual_norm: float | None = None
    residual_target: float | None = None

    # Explicit solver-system diagnostics.
    solver_residual_norm: float | None = None
    solver_rhs_norm: float | None = None
    solver_relative_residual_norm: float | None = None
    solver_residual_target: float | None = None

    # Physical/original-system diagnostics for A x = b.
    physical_residual_norm: float | None = None
    physical_rhs_norm: float | None = None
    physical_relative_residual_norm: float | None = None
    physical_residual_target: float | None = None

    # Optional row-restricted diagnostics for systems with strongly imposed
    # rows, for example HDG boundary trace penalties.
    diagnostic_residual_label: str | None = None
    diagnostic_residual_norm: float | None = None
    diagnostic_rhs_norm: float | None = None
    diagnostic_relative_residual_norm: float | None = None
    diagnostic_residual_target: float | None = None

    rtol: float | None = None
    atol: float | None = None

    # Preconditioner-application diagnostics, if the supplied preconditioner
    # exposes these attributes. This is mainly for ElementSchwarzPreconditioner.
    preconditioner_apply_count: int | None = None
    preconditioner_apply_seconds: float | None = None
    preconditioner_local_solve_seconds: float | None = None
    preconditioner_reduce_seconds: float | None = None
    preconditioner_copy_seconds: float | None = None
    preconditioner_factor_nnz: int | None = None

    # PETSc diagnostics, when the PETSc backend is used.
    petsc_preset: str | None = None
    petsc_converged_reason: int | None = None
    petsc_residual_norm: float | None = None

    # CuPy/Cupyx diagnostics, when a GPU sparse solve is used.
    cupyx_solver: str | None = None

    # Symmetric matrix permutation diagnostics.
    permutation_elapsed_seconds: float | None = None
    permutation_size: int | None = None
    ilu_permc_spec: str | None = None

    # Backend-neutral convergence contract. ``info`` remains the normalized
    # integer compatibility field; ``backend_info`` preserves the native code.
    backend: str | None = None
    backend_info: int | str | None = None
    status: SolveStatus = "not-converged"
    converged: bool = False
    failure_reason: str | None = None
    solution_is_finite: bool | None = None
    solver_residual_is_finite: bool | None = None
    physical_residual_is_finite: bool | None = None
    solver_residual_target_met: bool | None = None
    physical_residual_target_met: bool | None = None
    stagnated: bool = False
    residual_history: tuple[float, ...] | None = None


def validate_global_system_inputs(
    row_indices: NDArray,
    col_indices: NDArray,
    matrix_values: NDArray,
    rhs: NDArray,
    system_size: int,
    initial_guess: NDArray | None = None,
) -> None:
    """Validate COO triplets, RHS, system size, and optional initial guess."""
    if system_size <= 0:
        raise ValueError(f"system_size must be positive, got {system_size}")

    if row_indices.ndim != 1:
        raise ValueError("row_indices must be a one-dimensional array")

    if col_indices.ndim != 1:
        raise ValueError("col_indices must be a one-dimensional array")

    if matrix_values.ndim != 1:
        raise ValueError("matrix_values must be a one-dimensional array")

    if row_indices.size == 0:
        raise ValueError("row_indices, col_indices, and matrix_values must be non-empty")

    if not (row_indices.size == col_indices.size == matrix_values.size):
        raise ValueError(
            "row_indices, col_indices, and matrix_values must have the same size"
        )

    if row_indices.min() < 0:
        raise ValueError(f"row index out of bounds: minimum row index is {row_indices.min()}")

    if col_indices.min() < 0:
        raise ValueError(f"column index out of bounds: minimum column index is {col_indices.min()}")

    if row_indices.max() >= system_size:
        raise ValueError(
            f"row index out of bounds: maximum row index is {row_indices.max()}, "
            f"but system size is {system_size}"
        )

    if col_indices.max() >= system_size:
        raise ValueError(
            f"column index out of bounds: maximum column index is {col_indices.max()}, "
            f"but system size is {system_size}"
        )

    if rhs.shape != (system_size,):
        raise ValueError(
            f"rhs must have shape ({system_size},), got {rhs.shape}"
        )

    if initial_guess is not None and initial_guess.shape != (system_size,):
        raise ValueError(
            f"initial_guess must have shape ({system_size},), got {initial_guess.shape}"
        )

    _validate_finite_array(matrix_values, "matrix_values")
    _validate_finite_array(rhs, "rhs")
    if initial_guess is not None:
        _validate_finite_array(initial_guess, "initial_guess")


def diagonal_scale_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    *, copy_matrix: bool = True,
) -> tuple[scipy.sparse.csr_array, NDArray]:
    """Apply left Jacobi scaling while keeping the matrix in CSR format."""
    scaled_matrix = matrix.tocsr(copy=copy_matrix)
    diagonal = scaled_matrix.diagonal()
    diagonal[diagonal == 0] = 1.0
    inverse_diagonal = 1.0 / diagonal

    row_nnz = np.diff(scaled_matrix.indptr)
    if _sparsetools is not None:
        nrows, ncols = scaled_matrix.shape
        _sparsetools.csr_scale_rows(
            nrows, ncols, scaled_matrix.indptr, scaled_matrix.indices,
            scaled_matrix.data, inverse_diagonal,
        )
    else:
        scaled_matrix.data *= np.repeat(inverse_diagonal, row_nnz)
    scaled_rhs = inverse_diagonal * rhs

    return scaled_matrix, scaled_rhs


def scale_sparse_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    mode: bool | Literal["none", "left", "symmetric"] = "none",
) -> tuple[scipy.sparse.csr_array, NDArray, NDArray | None]:
    """Scale a square sparse system and return its solution transform.

    ``"left"`` applies Jacobi row scaling and leaves the unknown unchanged.
    ``"symmetric"`` applies ``D**-1/2 A D**-1/2`` and
    ``D**-1/2 b``; the returned vector is ``D**-1/2`` and maps the scaled
    unknown back to the physical one by elementwise multiplication. Boolean
    values are accepted for compatibility, with ``True`` meaning ``"left"``.
    """
    normalized = "left" if mode is True else "none" if mode is False else str(mode).lower()
    normalized = {"on": "left", "off": "none"}.get(normalized, normalized)
    if normalized not in {"none", "left", "symmetric"}:
        raise ValueError("mode must be one of 'none', 'left', or 'symmetric'")

    csr = matrix.tocsr(copy=True)
    vector = np.ascontiguousarray(rhs, dtype=REAL_DTYPE)
    if csr.shape != (vector.size, vector.size):
        raise ValueError(f"matrix must have shape ({vector.size}, {vector.size}); got {csr.shape}")
    if normalized == "none":
        return csr, vector.copy(), None
    if normalized == "left":
        scaled_matrix, scaled_rhs = diagonal_scale_system(csr, vector, copy_matrix=False)
        return scaled_matrix, np.ascontiguousarray(scaled_rhs), None

    diagonal = np.asarray(csr.diagonal(), dtype=REAL_DTYPE)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError("symmetric scaling requires a finite, strictly positive diagonal")
    inverse_sqrt = 1.0 / np.sqrt(diagonal)
    scaling = scipy.sparse.diags(inverse_sqrt, format="csr")
    scaled_matrix = (scaling @ csr @ scaling).tocsr()
    scaled_rhs = np.ascontiguousarray(inverse_sqrt * vector)
    return scaled_matrix, scaled_rhs, np.ascontiguousarray(inverse_sqrt)


def compute_residual_norm(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    x: NDArray,
    rhs: NDArray,
) -> float:
    r"""Return :math:`\|Ax-b\|_2`."""
    return float(np.linalg.norm(matrix @ x - rhs))


def residual_diagnostics(
    residual_norm: float,
    rhs: NDArray,
    *,
    rtol: float,
    atol: float,
) -> tuple[float, float, float]:
    """Return (rhs_norm, relative_residual_norm, stopping_target).

    SciPy's Krylov stopping test is ||b - A x|| <= max(rtol*||b||, atol).
    Reporting the same target prevents warm-start solves from looking falsely
    successful or falsely suspicious when zero iterations are taken.
    """
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm > 0.0:
        relative_residual_norm = residual_norm / rhs_norm
    else:
        relative_residual_norm = residual_norm
    residual_target = max(rtol * rhs_norm, atol)
    return rhs_norm, relative_residual_norm, residual_target


def restricted_residual_diagnostics(
    residual: NDArray,
    rhs: NDArray,
    rows: NDArray | None,
    *,
    rtol: float,
    atol: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return residual diagnostics restricted to selected rows."""
    if rows is None:
        return None, None, None, None
    residual_norm = float(np.linalg.norm(residual[rows]))
    rhs_norm, relative_residual_norm, residual_target = residual_diagnostics(
        residual_norm,
        rhs[rows],
        rtol=rtol,
        atol=atol,
    )
    return residual_norm, rhs_norm, relative_residual_norm, residual_target


def residual_history_is_stagnated(
    history: Any,
    *,
    window: int = 50,
    relative_improvement: float = 1.0e-6,
) -> bool:
    """Return whether the recent finite residual history has stopped improving."""
    values = np.asarray(() if history is None else tuple(history), dtype=REAL_DTYPE)
    if values.size < int(window) or int(window) < 2:
        return False
    recent = values[-int(window):]
    if not np.all(np.isfinite(recent)) or np.any(recent < 0.0):
        return False
    baseline = float(recent[0])
    best = float(np.min(recent))
    if baseline <= 0.0:
        return bool(best <= 0.0)
    improvement = max(0.0, (baseline - best) / baseline)
    return bool(improvement <= float(relative_improvement))


def _residual_target_met(residual_norm: float | None, target: float | None, *, rtol: float, atol: float) -> bool:
    """Check a residual target, treating two zero tolerances as validation-disabled."""
    if residual_norm is None or target is None or not np.isfinite(residual_norm) or not np.isfinite(target):
        return False
    if float(rtol) == 0.0 and float(atol) == 0.0:
        return True
    return bool(float(residual_norm) <= float(target))


def _linear_solve_failure_message(result: SolveResult) -> str:
    """Build the stable diagnostic message used by convergence exceptions."""
    parts = [
        f"{result.backend or 'linear'} solve did not converge",
        f"status={result.status}",
        f"reason={result.failure_reason or 'unknown'}",
    ]
    if result.backend_info is not None:
        parts.append(f"backend_info={result.backend_info}")
    if result.iteration_count is not None:
        parts.append(f"iterations={result.iteration_count}")
    if result.solver_residual_norm is not None:
        parts.append(f"solver_residual={result.solver_residual_norm:.3e}")
    if result.solver_residual_target is not None:
        parts.append(f"solver_target={result.solver_residual_target:.3e}")
    if result.physical_residual_norm is not None:
        parts.append(f"physical_residual={result.physical_residual_norm:.3e}")
    if result.physical_residual_target is not None:
        parts.append(f"physical_target={result.physical_residual_target:.3e}")
    return ", ".join(parts)


def finalize_solve_result(
    result: SolveResult,
    *,
    backend: str,
    backend_info: int | str | None = None,
    backend_success: bool | None = None,
    solution_is_finite: bool | None = None,
    residual_history: Any = None,
    raise_on_nonconvergence: bool = False,
    stagnation_window: int = 50,
    stagnation_relative_improvement: float = 1.0e-6,
) -> SolveResult:
    """Apply one convergence contract to a backend-populated result."""
    native_info = result.info if backend_info is None else backend_info
    if backend_success is None:
        backend_success = result.info in {None, 0}
    backend_diverged = isinstance(native_info, str) and "diverg" in native_info.lower()
    if backend_diverged:
        backend_success = False

    history = tuple(float(value) for value in (() if residual_history is None else residual_history))[-64:]
    if solution_is_finite is None:
        solution_is_finite = result.x is not None and bool(np.all(np.isfinite(result.x)))

    solver_values = (
        result.solver_residual_norm,
        result.solver_rhs_norm,
        result.solver_relative_residual_norm,
        result.solver_residual_target,
    )
    physical_values = (
        result.physical_residual_norm,
        result.physical_rhs_norm,
        result.physical_relative_residual_norm,
        result.physical_residual_target,
    )
    solver_residual_is_finite = all(value is not None and np.isfinite(value) for value in solver_values)
    physical_residual_is_finite = all(value is not None and np.isfinite(value) for value in physical_values)
    rtol = 0.0 if result.rtol is None else float(result.rtol)
    atol = 0.0 if result.atol is None else float(result.atol)
    solver_target_met = _residual_target_met(
        result.solver_residual_norm,
        result.solver_residual_target,
        rtol=rtol,
        atol=atol,
    )
    physical_target_met = _residual_target_met(
        result.physical_residual_norm,
        result.physical_residual_target,
        rtol=rtol,
        atol=atol,
    )
    stagnated = (
        not solver_target_met
        and residual_history_is_stagnated(
            history,
            window=stagnation_window,
            relative_improvement=stagnation_relative_improvement,
        )
    )
    converged = bool(
        backend_success
        and solution_is_finite
        and solver_residual_is_finite
        and physical_residual_is_finite
        and solver_target_met
        and physical_target_met
    )

    result.backend = str(backend)
    result.backend_info = native_info
    result.converged = converged
    result.solution_is_finite = bool(solution_is_finite)
    result.solver_residual_is_finite = solver_residual_is_finite
    result.physical_residual_is_finite = physical_residual_is_finite
    result.solver_residual_target_met = solver_target_met
    result.physical_residual_target_met = physical_target_met
    result.stagnated = stagnated
    result.residual_history = history or None

    if converged:
        result.status = "converged"
        result.failure_reason = None
        result.info = 0
    else:
        if not solution_is_finite:
            result.status = "non-finite"
            result.failure_reason = "non-finite-solution"
        elif not solver_residual_is_finite or not physical_residual_is_finite:
            result.status = "non-finite"
            result.failure_reason = "non-finite-residual"
        elif backend_diverged:
            result.status = "diverged"
            result.failure_reason = "backend-divergence"
        elif stagnated:
            result.status = "stagnated"
            result.failure_reason = "stagnation"
        elif not backend_success:
            result.status = "not-converged"
            result.failure_reason = "backend-nonconvergence"
        else:
            result.status = "not-converged"
            result.failure_reason = "residual-target-not-met"
        result.info = int(native_info) if isinstance(native_info, (int, np.integer)) and int(native_info) != 0 else 1

    if raise_on_nonconvergence and not result.converged:
        raise LinearSolveConvergenceError(
            _linear_solve_failure_message(result),
            result=result,
        )
    return result


def refine_host_linear_solution(matrix, rhs, solution, *, solve_correction,
                                rtol: float, atol: float = 0.0,
                                max_corrections: int = 2) -> tuple[NDArray, int]:
    """Correct a host solution against the original operator with retained factors.

    ``solve_correction(residual)`` applies the caller-owned approximate inverse.
    For example, roundoff-asymmetric SPD systems can retain Cholesky factors of
    the symmetric upper-triangle interpretation, while residuals use the full
    original matrix. Inputs are not modified. Return the corrected vector and
    number of correction solves; callers must independently verify convergence.
    Include these residual evaluations and corrections in solver timings.
    """
    _validate_solver_controls(rtol=rtol, atol=atol)
    if isinstance(max_corrections, bool) or not isinstance(max_corrections, (int, np.integer)) or max_corrections < 0:
        raise ValueError("max_corrections must be a nonnegative integer")
    rhs = np.asarray(rhs, dtype=REAL_DTYPE)
    x = np.array(solution, dtype=REAL_DTYPE, copy=True)
    if rhs.ndim != 1 or x.shape != rhs.shape or matrix.shape != (rhs.size, rhs.size):
        raise ValueError("matrix, rhs and solution shapes are incompatible")
    _validate_finite_array(rhs, "rhs")
    _validate_finite_array(x, "solution")
    target = max(float(atol), float(rtol) * float(np.linalg.norm(rhs)))
    count = 0
    for _ in range(max_corrections):
        residual = rhs - matrix @ x
        _validate_finite_array(residual, "refinement residual")
        if float(np.linalg.norm(residual)) <= target:
            break
        delta = np.asarray(solve_correction(residual), dtype=REAL_DTYPE)
        if delta.shape != rhs.shape:
            raise ValueError("correction has the wrong shape")
        _validate_finite_array(delta, "refinement correction")
        x += delta
        count += 1
    return x, count
