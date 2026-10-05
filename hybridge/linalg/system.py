"""Sparse global-system assembly and solve helpers.

This module is the self-contained :mod:`hybridge` version of the trace-system
solver used by the HDG assembly code.  It builds CSR matrices from COO triplets
and provides direct or Krylov solves with optional diagonal scaling, ILU
preconditioning, and cheap diagonal Jacobi preconditioning.
"""

from __future__ import annotations


import time
from typing import Any, Mapping, Literal

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray
from scipy.sparse import coo_array
try:
    from scipy.sparse import _sparsetools
except ImportError:  # Fall back to a vectorized implementation on other SciPy builds.
    _sparsetools = None
from hybridge.linalg.results import (
    _normalize_diagnostic_rows,
    _solver_print,
    _validate_finite_array,
    _validate_solver_controls,
)
from hybridge.linalg.amgx.host import solve_pyamgx_system
from hybridge.linalg.results import (
    LinearSolveConvergenceError,
    SolveResult,
    _linear_solve_failure_message,
    diagonal_scale_system,
    finalize_solve_result,
    residual_diagnostics,
    restricted_residual_diagnostics,
    validate_global_system_inputs,
)
from hybridge.linalg.direct import solve_direct_system, solve_pypardiso_system
from hybridge.linalg.iterative import solve_iterative_system, solve_petsc_system
from hybridge.linalg.gpu.cupyx import solve_cupyx_system


def _normalize_permutation(permutation: NDArray | None, system_size: int) -> NDArray | None:
    """Validate a symmetric matrix permutation.

    The permutation convention is ``A_perm = A[permutation][:, permutation]``.
    The solved vector is unpermuted with ``x[permutation] = x_perm``.
    """
    if permutation is None:
        return None
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.shape != (system_size,):
        raise ValueError(f"permutation must have shape ({system_size},), got {perm.shape}")
    if perm.size == 0:
        return np.ascontiguousarray(perm)
    if perm.min() < 0 or perm.max() >= system_size:
        raise ValueError("permutation contains indices outside the system")
    seen = np.zeros(system_size, dtype=bool)
    seen[perm] = True
    if not np.all(seen):
        raise ValueError("permutation must contain every system index exactly once")
    return np.ascontiguousarray(perm)


def _inverse_permutation(permutation: NDArray) -> NDArray:
    """Return the inverse of a permutation using the module convention."""
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size, dtype=permutation.dtype)
    return inverse


def _permute_diagnostic_rows(diagnostic_rows: NDArray | None, permutation: NDArray, inverse_permutation: NDArray):
    """Map row diagnostics from original row ids to permuted row ids."""
    if diagnostic_rows is None:
        return None
    rows = np.asarray(diagnostic_rows)
    if rows.dtype == bool:
        return np.ascontiguousarray(rows[permutation])
    return np.ascontiguousarray(inverse_permutation[rows.astype(np.int64)])


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


def solve_global_system(
    row_indices: NDArray,
    col_indices: NDArray,
    matrix_values: NDArray,
    rhs: NDArray,
    system_size: int,
    *,
    solver: str | Literal["direct"] = "direct",
    preconditioner: scipy.sparse.linalg.LinearOperator | Literal["ilu", "jacobi", None] = "ilu",
    initial_guess: NDArray | None = None,
    rtol: float = 1e-13,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int | None = None,
    ilu_drop_tol: float = 1e-10,
    ilu_fill_factor: float = 35,
    ilu_failure: Literal["raise", "none"] = "raise",
    ilu_permc_spec: str = "COLAMD",
    upwind_block_size: int | None = None,
    upwind_level_widths: Any | None = None,
    upwind_diagonal_regularization: float = 0.0,
    petsc_preset: str = "cg_gamg",
    petsc_levels: int | None = None,
    petsc_options: Mapping[str, Any] | None = None,
    petsc_divtol: float = 1e4,
    petsc_monitor: bool = False,
    amgx_config: Mapping[str, Any] | None = None,
    cupyx_solver: str = "bicgstab",
    scale_system: bool = True,
    raise_on_nonconvergence: bool = False,
    materialize_host_solution: bool = True,
    verbose: bool | int = 0,
    assembled_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
    prepared_scaled_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
    prepared_inverse_diagonal: NDArray | None = None,
    prepared_device_matrix: Any | None = None,
    scale_matrix_in_place: bool = False,
    permutation: NDArray | None = None,
    diagnostic_rows: NDArray | None = None,
    diagnostic_label: str | None = None,
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
        ``"pypardiso"`` or ``"pardiso"`` uses the optional oneMKL PARDISO
        host backend for general real matrices. ``"pypardiso-spd"`` and
        ``"pardiso-spd"`` validate symmetry and use its real-SPD mode.
        ``"petsc"`` uses the PETSc backend. ``"pyamgx"`` or ``"amgx"`` uses
        the optional PyAMGX backend. ``"cupyx"`` uses a Cupyx sparse
        Krylov solver selected by ``cupyx_solver``; aliases such as
        ``"cupyx_bicgstab"`` select the Cupyx method inline. Other names are
        looked up in the supported SciPy Krylov solver table.
    preconditioner
        ``"ilu"`` builds an ILU preconditioner for the selected Krylov backend
        (SciPy for CPU solvers, Cupyx for ``solver="cupyx"``). ``"jacobi"``
        builds a cheap diagonal Jacobi preconditioner for CPU solvers, ``None``
        disables preconditioning, and a supplied operator is passed directly to
        the selected backend.
    scale_system
        Apply left Jacobi scaling before iterative solves.
    ilu_failure
        Behavior when SciPy ILU factorization fails.  ``"raise"`` preserves
        the failure, while ``"none"`` continues with no preconditioner.
    ilu_permc_spec
        SuperLU column permutation used by ``spilu``.  Use ``"NATURAL"`` when
        the caller has already supplied a meaningful matrix ``permutation``.
    prepared_device_matrix
        Optional GPU sparse matrix for GPU solve backends.  For ``solver="cupyx"``
        this must be a CuPy CSR matrix matching the scaled or unscaled solve
        operator selected by ``scale_system``.  It is primarily for stateful
        solver classes that reuse a Numba-assembled trace operator across RHS
        updates.
    scale_matrix_in_place
        Allow in-place CSR row scaling.  This avoids an extra sparse matrix copy
        on large trace systems when the caller does not need the unscaled matrix
        after the solve.
    permutation
        Optional symmetric permutation with convention
        ``A_perm = A[permutation][:, permutation]`` and
        ``rhs_perm = rhs[permutation]``.  The returned solution is unpermuted
        before it is stored in :class:`SolveResult`.
    verbose
        Solver verbosity level.  ``0`` is silent, ``1`` prints concise solve
        progress, and ``2`` prints detailed CSR assembly, scaling, ILU, Krylov,
        residual, and timing diagnostics.  ``True`` is accepted as level ``1``
        for backward compatibility.
    diagnostic_rows
        Optional boolean mask or integer row indices used only for an
        additional residual diagnostic.  This is useful when some rows are
        strongly imposed with large penalties and should not dominate the
        physically meaningful relative residual.
    """
    global_start = time.perf_counter()
    _validate_solver_controls(rtol=rtol, atol=atol, maxiter=maxiter, restart=restart)
    row_indices = np.asarray(row_indices)
    col_indices = np.asarray(col_indices)
    matrix_values = np.asarray(matrix_values)
    rhs = np.asarray(rhs)

    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess)
    diagnostic_rows = _normalize_diagnostic_rows(diagnostic_rows, system_size)
    permutation = _normalize_permutation(permutation, system_size)
    if permutation is not None and prepared_device_matrix is not None:
        raise ValueError("prepared_device_matrix cannot be combined with a new matrix permutation")

    normalized_solver = "" if solver is None else str(solver).lower()
    normalized_solver_family = normalized_solver.replace("_", "-")
    solver_is_pypardiso_spd = normalized_solver_family in {"pypardiso-spd", "pardiso-spd"}
    solver_is_pypardiso = normalized_solver_family in {
        "pypardiso",
        "pardiso",
        "pypardiso-spd",
        "pardiso-spd",
    }
    solver_is_petsc = normalized_solver == "petsc"
    solver_is_pyamgx = normalized_solver in {"pyamgx", "amgx"}
    solver_is_cupyx = normalized_solver == "cupyx" or normalized_solver.startswith(("cupyx_", "cupyx-"))
    effective_cupyx_solver = cupyx_solver
    if normalized_solver.startswith(("cupyx_", "cupyx-")):
        effective_cupyx_solver = str(solver)[6:]

    matrix = None
    matrix_assembly_elapsed_seconds = 0.0
    if assembled_matrix is None:
        validate_global_system_inputs(
            row_indices=row_indices,
            col_indices=col_indices,
            matrix_values=matrix_values,
            rhs=rhs,
            system_size=system_size,
            initial_guess=initial_guess,
        )
        if not solver_is_cupyx:
            assembly_start = time.time()
            matrix = assemble_global_matrix(
                row_indices=row_indices,
                col_indices=col_indices,
                matrix_values=matrix_values,
                system_size=system_size,
            )
            matrix_assembly_elapsed_seconds = time.time() - assembly_start
            _solver_print(
                verbose,
                2,
                "  sparse CSR matrix assembled in %.5fs with nnz=%d",
                matrix_assembly_elapsed_seconds,
                matrix.nnz,
            )
        else:
            _solver_print(verbose, 2, "  deferring COO-to-CSR construction to Cupyx on the device")
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
        _validate_finite_array(np.asarray(assembled_matrix.data), "assembled_matrix data")
        _validate_finite_array(rhs, "rhs")
        if initial_guess is not None:
            _validate_finite_array(initial_guess, "initial_guess")
        matrix = assembled_matrix
        _solver_print(
            verbose,
            2,
            "  using preassembled sparse matrix with shape=%s and nnz=%d",
            matrix.shape,
            matrix.nnz,
        )

    permutation_elapsed_seconds = 0.0
    inverse_permutation = None
    if permutation is not None:
        permutation_start = time.time()
        inverse_permutation = _inverse_permutation(permutation)
        if matrix is None:
            row_indices = inverse_permutation[row_indices]
            col_indices = inverse_permutation[col_indices]
        else:
            matrix = matrix.tocsr()[permutation][:, permutation].tocsr()
        rhs = rhs[permutation]
        if initial_guess is not None:
            initial_guess = initial_guess[permutation]
        diagnostic_rows = _permute_diagnostic_rows(diagnostic_rows, permutation, inverse_permutation)
        permutation_elapsed_seconds = time.time() - permutation_start
        _solver_print(
            verbose,
            2,
            "  symmetric matrix permutation applied in %.5fs",
            permutation_elapsed_seconds,
        )

    if normalized_solver in {"", "direct"}:
        _solver_print(verbose, 1, "  solver: scipy.sparse.linalg.spsolve")

        result = solve_direct_system(
            matrix,
            rhs,
            rtol=rtol,
            atol=atol,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label=diagnostic_label,
            raise_on_nonconvergence=False,
        )

    elif solver_is_pypardiso:
        _solver_print(
            verbose,
            1,
            "  solver: pypardiso oneMKL PARDISO%s",
            " (real SPD)" if solver_is_pypardiso_spd else "",
        )
        result = solve_pypardiso_system(
            matrix,
            rhs,
            rtol=rtol,
            atol=atol,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label=diagnostic_label,
            raise_on_nonconvergence=False,
            matrix_type="spd" if solver_is_pypardiso_spd else "nonsymmetric",
        )

    elif solver_is_petsc:
        if scale_system:
            _solver_print(
                verbose,
                2,
                "  diagonal scaling enabled for PETSc comparison runs",
            )
            scale_start = time.time()
            petsc_matrix_input, petsc_rhs_input = diagonal_scale_system(
                matrix,
                rhs,
                copy_matrix=True,
            )
            scale_elapsed_seconds = time.time() - scale_start
            _solver_print(verbose, 2, "  diagonal scaling completed in %.5fs", scale_elapsed_seconds)
        else:
            scale_start = time.time()
            petsc_matrix_input = matrix.tocsr()
            petsc_rhs_input = rhs
            scale_elapsed_seconds = time.time() - scale_start
            _solver_print(
                verbose,
                2,
                "  diagonal scaling disabled; CSR conversion completed in %.5fs",
                scale_elapsed_seconds,
            )
        result = solve_petsc_system(
            petsc_matrix_input,
            petsc_rhs_input,
            preset=petsc_preset,
            levels=petsc_levels,
            options=petsc_options,
            initial_guess=initial_guess,
            rtol=rtol,
            atol=atol,
            divtol=petsc_divtol,
            maxiter=maxiter,
            use_monitor=petsc_monitor,
            raise_on_nonconvergence=False,
            verbose=verbose,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label=diagnostic_label,
        )
        if result.total_elapsed_seconds is not None:
            result.total_elapsed_seconds += scale_elapsed_seconds
        result.scale_elapsed_seconds = scale_elapsed_seconds
        physical_residual = matrix @ result.x - rhs
        physical_residual_norm = float(np.linalg.norm(physical_residual))
        physical_rhs_norm, physical_relative_residual_norm, physical_residual_target = residual_diagnostics(
            physical_residual_norm,
            rhs,
            rtol=rtol,
            atol=atol,
        )
        (
            diagnostic_residual_norm,
            diagnostic_rhs_norm,
            diagnostic_relative_residual_norm,
            diagnostic_residual_target,
        ) = restricted_residual_diagnostics(physical_residual, rhs, diagnostic_rows, rtol=rtol, atol=atol)
        result.physical_residual_norm = physical_residual_norm
        result.physical_rhs_norm = physical_rhs_norm
        result.physical_relative_residual_norm = physical_relative_residual_norm
        result.physical_residual_target = physical_residual_target
        result.diagnostic_residual_norm = diagnostic_residual_norm
        result.diagnostic_rhs_norm = diagnostic_rhs_norm
        result.diagnostic_relative_residual_norm = diagnostic_relative_residual_norm
        result.diagnostic_residual_target = diagnostic_residual_target
        result = finalize_solve_result(
            result,
            backend=result.backend or f"petsc-{petsc_preset}",
            backend_info=result.backend_info,
            backend_success=(result.petsc_converged_reason or 0) > 0,
            residual_history=result.residual_history,
            raise_on_nonconvergence=False,
        )

    elif solver_is_pyamgx:
        _solver_print(verbose, 1, "  solver: PyAMGX BICGSTAB+AMG")
        result = solve_pyamgx_system(
            matrix,
            rhs,
            initial_guess=initial_guess,
            config=amgx_config,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            scale_system=scale_system,
            raise_on_nonconvergence=False,
            verbose=verbose,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label=diagnostic_label,
        )

    elif solver_is_cupyx:
        if isinstance(preconditioner, str) and str(preconditioner).lower().replace("-", "_") in {"ilu", "host_ilu", "host_ilu_export", "cupyx_ilu1", "device_ilu1"}:
            _solver_print(verbose, 1, "  solver: Cupyx %s with %s preconditioner", str(effective_cupyx_solver).upper(), str(preconditioner))
        elif preconditioner is None:
            _solver_print(verbose, 1, "  solver: Cupyx %s without preconditioner", str(effective_cupyx_solver).upper())
        else:
            _solver_print(verbose, 1, "  solver: Cupyx %s with supplied preconditioner", str(effective_cupyx_solver).upper())
        result = solve_cupyx_system(
            matrix,
            rhs,
            row_indices=row_indices if matrix is None else None,
            col_indices=col_indices if matrix is None else None,
            matrix_values=matrix_values if matrix is None else None,
            system_size=system_size if matrix is None else None,
            cupyx_solver=effective_cupyx_solver,
            preconditioner=preconditioner,
            prepared_device_matrix=prepared_device_matrix,
            initial_guess=initial_guess,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            restart=restart,
            scale_system=scale_system,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            ilu_failure=ilu_failure,
            ilu_permc_spec=ilu_permc_spec,
            upwind_block_size=upwind_block_size,
            upwind_level_widths=upwind_level_widths,
            upwind_diagonal_regularization=upwind_diagonal_regularization,
            raise_on_nonconvergence=False,
            materialize_host_solution=materialize_host_solution,
            verbose=verbose,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label=diagnostic_label,
        )

    else:
        if isinstance(preconditioner, str) and preconditioner == "ilu":
            _solver_print(verbose, 1, "  solver: %s with ILU preconditioner", str(solver).upper())
        elif isinstance(preconditioner, str) and preconditioner == "jacobi":
            _solver_print(verbose, 1, "  solver: %s with Jacobi preconditioner", str(solver).upper())
        elif preconditioner is None:
            _solver_print(verbose, 1, "  solver: %s without preconditioner", str(solver).upper())
        else:
            _solver_print(verbose, 1, "  solver: %s with supplied preconditioner", str(solver).upper())

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
            ilu_failure=ilu_failure,
            ilu_permc_spec=ilu_permc_spec,
            scale_system=scale_system,
            raise_on_nonconvergence=False,
            verbose=verbose,
            prepared_scaled_matrix=prepared_scaled_matrix,
            prepared_inverse_diagonal=prepared_inverse_diagonal,
            scale_matrix_in_place=scale_matrix_in_place,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label=diagnostic_label,
            validate_matrix=False,
        )

    if permutation is not None:
        if result.x is not None:
            unpermuted = np.empty_like(result.x)
            unpermuted[permutation] = result.x
            result.x = unpermuted
        if result.x_device is not None:
            cupy = __import__("cupy")
            unpermuted_device = cupy.empty_like(result.x_device)
            unpermuted_device[cupy.asarray(permutation)] = result.x_device
            result.x_device = unpermuted_device
        result.permutation_elapsed_seconds = permutation_elapsed_seconds
        result.permutation_size = int(permutation.size)

    if result.converged:
        _solver_print(
            verbose,
            1,
            "  global system solved in %.5fs with solver_rel=%.3e",
            result.total_elapsed_seconds,
            result.solver_relative_residual_norm,
        )
        _solver_print(
            verbose,
            2,
            "  final diagnostics: solver_abs=%.3e, solver_target=%.3e, penalized_physical_abs=%.3e",
            result.solver_residual_norm,
            result.solver_residual_target,
            result.physical_residual_norm,
        )
        if result.diagnostic_residual_norm is not None:
            _solver_print(
                verbose,
                2,
                "  final %s residual: abs=%.3e, rel=%.3e",
                result.diagnostic_residual_label or "restricted-row",
                result.diagnostic_residual_norm,
                result.diagnostic_relative_residual_norm,
            )
    else:
        _solver_print(
            verbose,
            1,
            "  global system solve did not converge: info=%s, solver_rel=%.3e, solver_target=%.3e",
            result.info,
            result.solver_relative_residual_norm,
            result.solver_residual_target,
        )
        _solver_print(
            verbose,
            2,
            "  nonconvergence diagnostics: solver_abs=%.3e, physical_abs=%.3e, physical_rel=%.3e",
            result.solver_residual_norm,
            result.physical_residual_norm,
            result.physical_relative_residual_norm,
        )

    result.matrix_assembly_elapsed_seconds = matrix_assembly_elapsed_seconds
    result.global_elapsed_seconds = time.perf_counter() - global_start

    if raise_on_nonconvergence and not result.converged:
        raise LinearSolveConvergenceError(
            _linear_solve_failure_message(result),
            result=result,
        )
    return result


__all__ = [
    "assemble_global_matrix",
    "solve_global_system",
]
