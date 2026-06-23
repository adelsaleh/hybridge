"""Sparse global-system assembly and solve helpers.

This module is the self-contained :mod:`dgfem` version of the trace-system
solver used by the HDG assembly code.  It builds CSR matrices from COO triplets
and provides direct or Krylov solves with optional diagonal scaling and ILU
preconditioning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.sparse.linalg import LinearOperator, spilu, spsolve
try:
    from scipy.sparse import _sparsetools
except ImportError:  # Fall back to a vectorized implementation on other SciPy builds.
    _sparsetools = None
from scipy.sparse.linalg import cg, cgs, bicg, bicgstab, minres, gmres, lgmres

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_ITERATIVE_SOLVERS = {
    "BICG": bicg,
    "BICGSTAB": bicgstab,
    "CG": cg,
    "CGS": cgs,
    "GMRES": gmres,
    "LGMRES": lgmres,
    "MINRES": minres,
}


class KrylovIterationCounter:
    """
    Counts Krylov iterations through SciPy's callback interface.

    For GMRES with callback_type="pr_norm", this counts inner Krylov iterations.
    For BICGSTAB/CG/CGS/MINRES/LGMRES, this counts callback calls.
    """

    def __init__(self):
        self.count = 0

    def __call__(self, _):
        self.count += 1


@dataclass
class SolveResult:
    """Solution vector and diagnostics returned by :func:`solve_global_system`."""

    x: NDArray
    residual_norm: float | None = None
    info: int | None = None
    preconditioner: scipy.sparse.linalg.LinearOperator | None = None

    total_elapsed_seconds: float | None = None
    scale_elapsed_seconds: float | None = None
    preconditioner_elapsed_seconds: float | None = None
    solve_elapsed_seconds: float | None = None

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

    rtol: float | None = None
    atol: float | None = None

    # Preconditioner-application diagnostics, if the supplied preconditioner
    # exposes these attributes. This is mainly for ElementSchwarzPreconditioner.
    preconditioner_apply_count: int | None = None
    preconditioner_apply_seconds: float | None = None
    preconditioner_local_solve_seconds: float | None = None
    preconditioner_reduce_seconds: float | None = None
    preconditioner_copy_seconds: float | None = None


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


def assemble_global_matrix(
    row_indices: NDArray,
    col_indices: NDArray,
    matrix_values: NDArray,
    system_size: int,
) -> scipy.sparse.csr_array:
    """Assemble COO triplets into a CSR sparse array."""
    return coo_array(
        (matrix_values, (row_indices, col_indices)),
        shape=(system_size, system_size),
    ).tocsr()


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


def build_ilu_preconditioner(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    *,
    drop_tol: float = 1e-10,
    fill_factor: float = 35,
) -> scipy.sparse.linalg.LinearOperator:
    """Build a SciPy ILU preconditioner after removing explicit sparse zeros."""
    matrix_csc = matrix.tocsc()
    matrix_csc.eliminate_zeros()
    ilu_decomposition = spilu(
        matrix_csc,
        drop_tol=drop_tol,
        fill_factor=fill_factor,
    )

    return LinearOperator(matrix.shape, matvec=ilu_decomposition.solve)


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


def get_iterative_solver(
    solver_name: str,
):
    """Return the SciPy Krylov routine matching ``solver_name``."""
    solvers = _ITERATIVE_SOLVERS

    normalized_name = solver_name.upper()

    if normalized_name not in solvers:
        valid_names = ", ".join(sorted(solvers))
        raise ValueError(
            f"Unknown iterative solver {solver_name!r}. "
            f"Valid options are: {valid_names}"
        )

    return solvers[normalized_name]


def solve_direct_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> SolveResult:
    """Solve an already assembled sparse system with ``spsolve``."""
    start = time.time()
    x = spsolve(matrix.tocsr(), rhs)
    solve_elapsed_seconds = time.time() - start

    physical_residual_norm = compute_residual_norm(matrix, x, rhs)
    physical_rhs_norm, physical_relative_residual_norm, physical_residual_target = residual_diagnostics(
        physical_residual_norm, rhs, rtol=rtol, atol=atol
    )

    return SolveResult(
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
        rtol=rtol,
        atol=atol,
        info=0,
        preconditioner=None,
        total_elapsed_seconds=solve_elapsed_seconds,
        scale_elapsed_seconds=0.0,
        preconditioner_elapsed_seconds=0.0,
        solve_elapsed_seconds=solve_elapsed_seconds,
    )


def solve_iterative_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    *,
    solver_name: str,
    preconditioner: scipy.sparse.linalg.LinearOperator | Literal["ilu", None] = "ilu",
    initial_guess: NDArray | None = None,
    rtol: float = 1e-13,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int | None = None,
    ilu_drop_tol: float = 1e-10,
    ilu_fill_factor: float = 35,
    scale_system: bool = True,
    raise_on_nonconvergence: bool = False,
    verbose: bool = False,
    prepared_scaled_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
    prepared_inverse_diagonal: NDArray | None = None,
    scale_matrix_in_place: bool = False,
) -> SolveResult:
    """Solve an already assembled sparse system with a SciPy Krylov method."""
    total_start = time.time()

    solver = get_iterative_solver(solver_name)
    solver_name_upper = solver_name.upper()

    scale_start = time.time()
    scaled_in_place = False
    physical_diagonal = None
    if scale_system:
        if prepared_scaled_matrix is None and prepared_inverse_diagonal is None:
            if scale_matrix_in_place:
                physical_diagonal = matrix.diagonal().copy()
                physical_diagonal[physical_diagonal == 0] = 1.0
            scaled_matrix, scaled_rhs = diagonal_scale_system(
                matrix, rhs, copy_matrix=not scale_matrix_in_place,
            )
            scaled_in_place = scaled_matrix is matrix and scale_matrix_in_place
        elif prepared_scaled_matrix is not None and prepared_inverse_diagonal is not None:
            if prepared_scaled_matrix.shape != matrix.shape:
                raise ValueError("prepared_scaled_matrix shape does not match matrix")
            inverse_diagonal = np.asarray(prepared_inverse_diagonal)
            if inverse_diagonal.shape != rhs.shape:
                raise ValueError("prepared_inverse_diagonal shape does not match rhs")
            scaled_matrix = prepared_scaled_matrix
            scaled_rhs = inverse_diagonal * rhs
        else:
            raise ValueError("prepared scaling matrix and diagonal must be supplied together")
    else:
        # Keep a sparse format suitable for SciPy Krylov methods.
        scaled_matrix = matrix.tocsr()
        scaled_rhs = rhs
    scale_elapsed_seconds = time.time() - scale_start

    if verbose:
        if scale_system:
            logger.info(
                "Diagonal scaling completed in %.5f seconds",
                scale_elapsed_seconds,
            )
        else:
            logger.info(
                "Diagonal scaling disabled; matrix conversion completed in %.5f seconds",
                scale_elapsed_seconds,
            )

    preconditioner_elapsed_seconds = 0.0

    if isinstance(preconditioner, str) and preconditioner == "ilu":
        if verbose:
            logger.info(
                "Building ILU preconditioner with drop_tol=%g, fill_factor=%g",
                ilu_drop_tol,
                ilu_fill_factor,
            )

        preconditioner_start = time.time()

        M = build_ilu_preconditioner(
            scaled_matrix,
            drop_tol=ilu_drop_tol,
            fill_factor=ilu_fill_factor,
        )

        preconditioner_elapsed_seconds = time.time() - preconditioner_start

        if verbose:
            logger.info(
                "ILU preconditioner built in %.5f seconds",
                preconditioner_elapsed_seconds,
            )

    else:
        M = preconditioner

        if verbose:
            if M is None:
                logger.info("No preconditioner used")
            else:
                logger.info("Using supplied preconditioner")

    iteration_counter = KrylovIterationCounter()

    solver_kwargs = {
        "M": M,
        "rtol": rtol,
        "atol": atol,
        "callback": iteration_counter,
    }

    if maxiter is not None:
        solver_kwargs["maxiter"] = maxiter

    # SciPy GMRES supports callback_type. With "pr_norm", the callback receives
    # the preconditioned residual norm and is called on each inner iteration.
    # Do not pass callback_type to solvers that do not accept it.
    if solver_name_upper == "GMRES":
        solver_kwargs["callback_type"] = "pr_norm"
        if restart is not None:
            solver_kwargs["restart"] = restart

    if initial_guess is not None:
        solver_kwargs["x0"] = initial_guess

    if verbose:
        logger.info(
            "Starting %s solve with rtol=%g, atol=%g",
            solver_name,
            rtol,
            atol,
        )

    initial_residual_norm = None
    initial_physical_residual_norm = None
    if initial_guess is not None:
        # Residual of the system actually solved by SciPy.
        initial_solver_residual = scaled_matrix @ initial_guess - scaled_rhs
        initial_residual_norm = float(np.linalg.norm(initial_solver_residual))
        if scaled_in_place:
            initial_physical_residual_norm = float(
                np.linalg.norm(physical_diagonal * initial_solver_residual)
            )
        else:
            initial_physical_residual_norm = compute_residual_norm(matrix, initial_guess, rhs)
        if verbose:
            logger.info(
                "Initial residual from supplied guess: solver %.3e; physical %.3e",
                initial_residual_norm,
                initial_physical_residual_norm,
            )

    solve_start = time.time()
    x, info = solver(scaled_matrix, scaled_rhs, **solver_kwargs)
    solve_elapsed_seconds = time.time() - solve_start

    x = np.asarray(x)

    # Solver diagnostics: match the exact system and RHS used by SciPy.
    solver_residual = scaled_matrix @ x - scaled_rhs
    solver_residual_norm = float(np.linalg.norm(solver_residual))
    solver_rhs_norm, solver_relative_residual_norm, solver_residual_target = residual_diagnostics(
        solver_residual_norm, scaled_rhs, rtol=rtol, atol=atol
    )

    # Physical diagnostics: useful for original-system interpretation, but not
    # the criterion SciPy used when scale_system=True.
    if scaled_in_place:
        physical_residual_norm = float(np.linalg.norm(physical_diagonal * solver_residual))
    else:
        physical_residual_norm = compute_residual_norm(matrix, x, rhs)
    physical_rhs_norm, physical_relative_residual_norm, physical_residual_target = residual_diagnostics(
        physical_residual_norm, rhs, rtol=rtol, atol=atol
    )

    total_elapsed_seconds = time.time() - total_start

    # Optional preconditioner timing diagnostics.
    #
    # A plain scipy LinearOperator does not provide these. Our
    # ElementSchwarzPreconditioner can expose them by attaching attributes
    # to the LinearOperator or by being reachable from the solver object.
    prec_apply_count = getattr(M, "apply_count", None)
    prec_apply_seconds = getattr(M, "apply_seconds", None)
    prec_local_solve_seconds = getattr(M, "local_solve_seconds", None)
    prec_reduce_seconds = getattr(M, "reduce_seconds", None)
    prec_copy_seconds = getattr(M, "copy_seconds", None)

    if verbose:
        logger.info(
            "%s finished with info=%s in %.5f seconds; iterations=%d; "
            "solver_res %.3e; solver_rel %.3e; solver_target %.3e; "
            "physical_res %.3e; physical_rel %.3e",
            solver_name,
            info,
            solve_elapsed_seconds,
            iteration_counter.count,
            solver_residual_norm,
            solver_relative_residual_norm,
            solver_residual_target,
            physical_residual_norm,
            physical_relative_residual_norm,
        )
        logger.info(
            "Total iterative path time %.5f seconds "
            "[scale %.5f, preconditioner %.5f, solve %.5f]",
            total_elapsed_seconds,
            scale_elapsed_seconds,
            preconditioner_elapsed_seconds,
            solve_elapsed_seconds,
        )

        if prec_apply_count is not None:
            logger.info(
                "Preconditioner apply stats: count=%s, total=%.5f, local=%.5f, reduce=%.5f, copy=%.5f",
                prec_apply_count,
                0.0 if prec_apply_seconds is None else prec_apply_seconds,
                0.0 if prec_local_solve_seconds is None else prec_local_solve_seconds,
                0.0 if prec_reduce_seconds is None else prec_reduce_seconds,
                0.0 if prec_copy_seconds is None else prec_copy_seconds,
            )

    if info != 0 and raise_on_nonconvergence:
        raise RuntimeError(
            f"{solver_name} failed to converge. "
            f"info={info}, iterations={iteration_counter.count}, "
            f"solver_res={solver_residual_norm:.3e}, "
            f"solver_rel={solver_relative_residual_norm:.3e}, "
            f"solver_target={solver_residual_target:.3e}, "
            f"physical_res={physical_residual_norm:.3e}, "
            f"physical_rel={physical_relative_residual_norm:.3e}"
        )

    return SolveResult(
        x=x,
        # Backward-compatible residual fields now refer to the solver system,
        # not the unscaled physical system.
        residual_norm=solver_residual_norm,
        info=info,
        preconditioner=M,
        total_elapsed_seconds=total_elapsed_seconds,
        scale_elapsed_seconds=scale_elapsed_seconds,
        preconditioner_elapsed_seconds=preconditioner_elapsed_seconds,
        solve_elapsed_seconds=solve_elapsed_seconds,
        iteration_count=iteration_counter.count,
        initial_residual_norm=initial_residual_norm,
        rhs_norm=solver_rhs_norm,
        relative_residual_norm=solver_relative_residual_norm,
        residual_target=solver_residual_target,
        solver_residual_norm=solver_residual_norm,
        solver_rhs_norm=solver_rhs_norm,
        solver_relative_residual_norm=solver_relative_residual_norm,
        solver_residual_target=solver_residual_target,
        physical_residual_norm=physical_residual_norm,
        physical_rhs_norm=physical_rhs_norm,
        physical_relative_residual_norm=physical_relative_residual_norm,
        physical_residual_target=physical_residual_target,
        rtol=rtol,
        atol=atol,
        preconditioner_apply_count=prec_apply_count,
        preconditioner_apply_seconds=prec_apply_seconds,
        preconditioner_local_solve_seconds=prec_local_solve_seconds,
        preconditioner_reduce_seconds=prec_reduce_seconds,
        preconditioner_copy_seconds=prec_copy_seconds,
    )


def solve_global_system(
    row_indices: NDArray,
    col_indices: NDArray,
    matrix_values: NDArray,
    rhs: NDArray,
    system_size: int,
    *,
    solver: str | Literal["direct"] = "direct",
    preconditioner: scipy.sparse.linalg.LinearOperator | Literal["ilu", None] = "ilu",
    initial_guess: NDArray | None = None,
    rtol: float = 1e-13,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int | None = None,
    ilu_drop_tol: float = 1e-10,
    ilu_fill_factor: float = 35,
    scale_system: bool = True,
    raise_on_nonconvergence: bool = False,
    verbose: bool = False,
    assembled_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
    prepared_scaled_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
    prepared_inverse_diagonal: NDArray | None = None,
    scale_matrix_in_place: bool = False,
) -> SolveResult:
    """Assemble and solve a sparse linear system from COO triplets.

    Parameters
    ----------
    row_indices, col_indices, matrix_values
        One-dimensional COO representation of the global sparse matrix.
    rhs
        Right-hand side vector.
    system_size
        Number of rows and columns in the square system.
    solver
        ``"direct"`` or ``None`` uses :func:`scipy.sparse.linalg.spsolve`.
        Other names are looked up in the supported SciPy Krylov solver table.
    preconditioner
        ``"ilu"`` builds a SciPy ILU preconditioner, ``None`` disables
        preconditioning, and a supplied :class:`LinearOperator` is passed
        directly to SciPy.
    scale_system
        Apply left Jacobi scaling before iterative solves.
    scale_matrix_in_place
        Allow in-place CSR row scaling.  This avoids an extra sparse matrix copy
        on large trace systems when the caller does not need the unscaled matrix
        after the solve.
    """
    row_indices = np.asarray(row_indices)
    col_indices = np.asarray(col_indices)
    matrix_values = np.asarray(matrix_values)
    rhs = np.asarray(rhs)

    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess)

    if assembled_matrix is None:
        validate_global_system_inputs(
            row_indices=row_indices,
            col_indices=col_indices,
            matrix_values=matrix_values,
            rhs=rhs,
            system_size=system_size,
            initial_guess=initial_guess,
        )
        matrix = assemble_global_matrix(
            row_indices=row_indices,
            col_indices=col_indices,
            matrix_values=matrix_values,
            system_size=system_size,
        )
    else:
        if not scipy.sparse.issparse(assembled_matrix):
            raise TypeError("assembled_matrix must be a SciPy sparse matrix or sparse array")
        if assembled_matrix.shape != (system_size, system_size):
            raise ValueError(
                f"assembled_matrix must have shape ({system_size}, {system_size}), "
                f"got {assembled_matrix.shape}"
            )
        if rhs.shape != (system_size,):
            raise ValueError(f"rhs must have shape ({system_size},), got {rhs.shape}")
        if initial_guess is not None and initial_guess.shape != (system_size,):
            raise ValueError(
                f"initial_guess must have shape ({system_size},), got {initial_guess.shape}"
            )
        matrix = assembled_matrix

    if solver is None or solver == "direct":
        if verbose:
            logger.info("Solving global system using scipy.sparse.linalg.spsolve")

        result = solve_direct_system(matrix, rhs, rtol=rtol, atol=atol)

    else:
        if verbose:
            if isinstance(preconditioner, str) and preconditioner == "ilu":
                logger.info("Solving global system using %s with ILU preconditioner", solver)
            elif preconditioner is None:
                logger.info("Solving global system using %s without preconditioner", solver)
            else:
                logger.info("Solving global system using %s with provided preconditioner", solver)

        result = solve_iterative_system(
            matrix,
            rhs,
            solver_name=solver,
            preconditioner=preconditioner,
            initial_guess=initial_guess,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            restart=restart,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            scale_system=scale_system,
            raise_on_nonconvergence=raise_on_nonconvergence,
            verbose=verbose,
            prepared_scaled_matrix=prepared_scaled_matrix,
            prepared_inverse_diagonal=prepared_inverse_diagonal,
            scale_matrix_in_place=scale_matrix_in_place,
        )

    if verbose:
        if result.info == 0:
            logger.info(
                "Global system solved in %.5f seconds with solver_res %.3e; "
                "solver_rel %.3e; solver_target %.3e; physical_res %.3e",
                result.total_elapsed_seconds,
                result.solver_residual_norm,
                result.solver_relative_residual_norm,
                result.solver_residual_target,
                result.physical_residual_norm,
            )
        else:
            logger.warning(
                "Global system solve did not converge. info=%s, solver_res=%.3e, "
                "solver_rel=%.3e, solver_target=%.3e, physical_res=%.3e",
                result.info,
                result.solver_residual_norm,
                result.solver_relative_residual_norm,
                result.solver_residual_target,
                result.physical_residual_norm,
            )

    return result


__all__ = [
    "KrylovIterationCounter",
    "SolveResult",
    "assemble_global_matrix",
    "build_ilu_preconditioner",
    "compute_residual_norm",
    "diagonal_scale_system",
    "get_iterative_solver",
    "residual_diagnostics",
    "solve_direct_system",
    "solve_global_system",
    "solve_iterative_system",
    "validate_global_system_inputs",
]
