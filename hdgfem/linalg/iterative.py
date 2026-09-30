"""hdgfem.linalg.iterative."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
import time
from typing import Any, Literal, Mapping
from hdgfem.linalg.results import (
    KrylovIterationCounter,
    SolveResult,
    _normalize_diagnostic_rows,
    _solver_print,
    _validate_finite_array,
    _validate_solver_controls,
    _verbosity_level,
    compute_residual_norm,
    diagonal_scale_system,
    finalize_solve_result,
    residual_diagnostics,
    restricted_residual_diagnostics,
)
from scipy.sparse.linalg import (
    LinearOperator,
    bicg,
    bicgstab,
    cg,
    cgs,
    gmres,
    lgmres,
    minres,
    spilu,
)
from numpy.typing import NDArray
from hdgfem.runtime.precision import REAL_DTYPE


_ITERATIVE_SOLVERS = {
    "BICG": bicg,
    "BICGSTAB": bicgstab,
    "CG": cg,
    "CGS": cgs,
    "GMRES": gmres,
    "LGMRES": lgmres,
    "MINRES": minres,
}


def build_ilu_preconditioner(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    *,
    drop_tol: float = 1e-10,
    fill_factor: float = 35,
    permc_spec: str = "COLAMD",
) -> scipy.sparse.linalg.LinearOperator:
    """Build a SciPy ILU preconditioner after removing explicit sparse zeros."""
    matrix_csc = matrix.tocsc()
    matrix_csc.eliminate_zeros()
    ilu_decomposition = spilu(
        matrix_csc,
        drop_tol=drop_tol,
        fill_factor=fill_factor,
        permc_spec=permc_spec,
    )

    apply_count = 0
    apply_seconds = 0.0

    def timed_solve(vector):
        """Apply the SuperLU factors while accumulating reusable timing data."""
        nonlocal apply_count, apply_seconds
        apply_start = time.perf_counter()
        result = ilu_decomposition.solve(vector)
        apply_seconds += time.perf_counter() - apply_start
        apply_count += 1
        operator.apply_count = apply_count
        operator.apply_seconds = apply_seconds
        return result

    operator = LinearOperator(matrix.shape, matvec=timed_solve, dtype=matrix.dtype)
    operator.apply_count = 0
    operator.apply_seconds = 0.0
    operator.factor_nnz = int(ilu_decomposition.L.nnz + ilu_decomposition.U.nnz)
    return operator


def build_jacobi_preconditioner(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
) -> scipy.sparse.linalg.LinearOperator:
    r"""Build the diagonal Jacobi preconditioner :math:`M^{-1}=D^{-1}`.

    This is the cheap preconditioner to use with symmetric Krylov solvers such
    as CG or MINRES on reduced diffusion trace systems.  The matrix diagonal
    must be strictly positive; otherwise the diagonal preconditioner would not
    be symmetric positive definite.
    """
    diagonal = np.asarray(matrix.diagonal(), dtype=REAL_DTYPE)
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("Jacobi preconditioner diagonal contains non-finite entries")
    if np.any(diagonal <= 0.0):
        raise ValueError("Jacobi preconditioner requires a strictly positive diagonal")

    inverse_diagonal = 1.0 / diagonal

    def apply(vector):
        """Apply the inverse Jacobi diagonal to a vector."""
        return inverse_diagonal * vector

    return LinearOperator(matrix.shape, matvec=apply, rmatvec=apply, dtype=REAL_DTYPE)


def _import_petsc():
    """Import PETSc lazily so SciPy-only users do not need petsc4py."""
    try:
        from petsc4py import PETSc
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "PETSc solve requested, but petsc4py is not importable in this environment."
        ) from exc
    return PETSc


def _petsc_options_context(PETSc, options: Mapping[str, Any] | None):
    """Install temporary PETSc options and return the keys to clear later."""
    if not options:
        return ()
    opts = PETSc.Options()
    keys = []
    for key, value in options.items():
        key = str(key).lstrip("-")
        opts[key] = str(value)
        keys.append(key)
    return tuple(keys)


def _clear_petsc_options(PETSc, keys) -> None:
    """Remove temporary PETSc options from the process-global options DB."""
    opts = PETSc.Options()
    for key in keys:
        try:
            del opts[key]
        except Exception:
            pass


def _configure_petsc_solver(
    PETSc,
    ksp,
    *,
    preset: str,
    levels: int | None,
    options: Mapping[str, Any] | None,
) -> tuple[str, tuple[str, ...]]:
    """Configure a PETSc KSP/PC pair and return temporary option keys."""
    normalized = preset.lower()
    pc = ksp.getPC()
    temporary_options: dict[str, Any] = {}

    if normalized == "cg_ilu":
        ksp.setType("cg")
        pc.setType("ilu")
        if levels is not None and levels > 0:
            pc.setFactorLevels(int(levels))
    elif normalized in {"bicgstab_ilu", "bcgs_ilu"}:
        ksp.setType("bcgs")
        pc.setType("ilu")
        if levels is not None and levels > 0:
            pc.setFactorLevels(int(levels))
    elif normalized in {"bicgstab_asm_ilu", "bcgs_asm_ilu"}:
        ksp.setType("bcgs")
        pc.setType("asm")
        temporary_options["sub_ksp_type"] = "preonly"
        temporary_options["sub_pc_type"] = "ilu"
        if levels is not None and levels > 0:
            temporary_options["sub_pc_factor_levels"] = int(levels)
    elif normalized == "gmres_ilu":
        ksp.setType("gmres")
        pc.setType("ilu")
        if levels is not None and levels > 0:
            pc.setFactorLevels(int(levels))
    elif normalized == "gmres_asm_ilu":
        ksp.setType("gmres")
        pc.setType("asm")
        temporary_options["sub_ksp_type"] = "preonly"
        temporary_options["sub_pc_type"] = "ilu"
        if levels is not None and levels > 0:
            temporary_options["sub_pc_factor_levels"] = int(levels)
    elif normalized in {"bicgstab_gamg", "bcgs_gamg"}:
        ksp.setType("bcgs")
        pc.setType("gamg")
        if levels is not None:
            temporary_options["pc_gamg_levels"] = int(levels)
    elif normalized == "gmres_gamg":
        ksp.setType("gmres")
        pc.setType("gamg")
        if levels is not None:
            temporary_options["pc_gamg_levels"] = int(levels)
    elif normalized == "cg_icc":
        ksp.setType("cg")
        pc.setType("icc")
        if levels is not None and levels > 0:
            pc.setFactorLevels(int(levels))
    elif normalized == "cg_hypre":
        ksp.setType("cg")
        pc.setType("hypre")
        try:
            pc.setHYPREType("boomeramg")
        except AttributeError:  # pragma: no cover - petsc4py version dependent.
            temporary_options["pc_hypre_type"] = "boomeramg"
    elif normalized == "cg_gamg":
        ksp.setType("cg")
        pc.setType("gamg")
        if levels is not None:
            temporary_options["pc_gamg_levels"] = int(levels)
    elif normalized == "lu":
        ksp.setType("preonly")
        pc.setType("lu")
    elif normalized == "mumps_lu":
        ksp.setType("preonly")
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
    else:
        raise ValueError(
            "unknown PETSc solver preset "
            f"{preset!r}; expected cg_ilu, bicgstab_ilu, bicgstab_asm_ilu, "
            "gmres_ilu, gmres_asm_ilu, bicgstab_gamg, gmres_gamg, "
            "cg_icc, cg_hypre, cg_gamg, lu, or mumps_lu"
        )

    combined_options = dict(temporary_options)
    if options:
        combined_options.update({str(key).lstrip("-"): value for key, value in options.items()})
    option_keys = _petsc_options_context(PETSc, combined_options)
    return normalized, option_keys


def _petsc_matrix_from_scipy(
    PETSc, matrix: scipy.sparse.spmatrix | scipy.sparse.sparray, *, resource_owner=None
):
    """Build a PETSc AIJ matrix from a SciPy sparse matrix."""
    petsc_matrix = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    if resource_owner is not None:
        resource_owner(petsc_matrix)
    if hasattr(petsc_matrix, "setPreallocationCOO"):
        coo = matrix.tocoo(copy=False)
        rows = np.asarray(coo.row, dtype=PETSc.IntType)
        cols = np.asarray(coo.col, dtype=PETSc.IntType)
        values = np.asarray(coo.data, dtype=REAL_DTYPE)
        petsc_matrix.setSizes(matrix.shape)
        petsc_matrix.setType(PETSc.Mat.Type.AIJ)
        petsc_matrix.setPreallocationCOO(rows, cols)
        petsc_matrix.setValuesCOO(values)
        petsc_matrix.assemble()
        return petsc_matrix

    csr = matrix.tocsr(copy=False)
    csr.sum_duplicates()
    row_pointers = np.asarray(csr.indptr, dtype=PETSc.IntType)
    column_indices = np.asarray(csr.indices, dtype=PETSc.IntType)
    values = np.asarray(csr.data, dtype=REAL_DTYPE)
    petsc_matrix.setSizes(matrix.shape)
    petsc_matrix.setType(PETSc.Mat.Type.AIJ)
    petsc_matrix.setPreallocationCSR((row_pointers, column_indices))
    petsc_matrix.setValuesCSR(row_pointers, column_indices, values)
    petsc_matrix.assemble()
    return petsc_matrix


def _solve_petsc_system_impl(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    *,
    preset: str = "cg_gamg",
    levels: int | None = None,
    options: Mapping[str, Any] | None = None,
    initial_guess: NDArray | None = None,
    rtol: float = 1e-13,
    atol: float = 0.0,
    divtol: float = 1e4,
    maxiter: int | None = None,
    use_monitor: bool = False,
    raise_on_nonconvergence: bool = False,
    verbose: bool | int = 0,
    diagnostic_rows: NDArray | None = None,
    diagnostic_label: str | None = None,
    _resource_owner: Any,
) -> SolveResult:
    """Solve a sparse system with PETSc KSP/PC presets.

    The intended diffusion-reaction path is ``preset="cg_gamg"`` on the
    boundary-eliminated trace system.  PETSc is imported lazily, so the rest of
    :mod:`hdgfem` remains usable without petsc4py.
    """
    _validate_solver_controls(rtol=rtol, atol=atol, maxiter=maxiter)
    total_start = time.time()
    import_start = time.time()
    PETSc = _import_petsc()
    petsc_import_elapsed_seconds = time.time() - import_start
    _solver_print(verbose, 2, "  PETSc runtime initialized in %.5fs", petsc_import_elapsed_seconds)
    matrix = matrix.tocsr()
    rhs = np.asarray(rhs, dtype=REAL_DTYPE)
    if matrix.shape != (rhs.size, rhs.size):
        raise ValueError(f"matrix must have shape ({rhs.size}, {rhs.size}), got {matrix.shape}")
    _validate_finite_array(np.asarray(matrix.data), "matrix data")
    _validate_finite_array(rhs, "rhs")
    diagnostic_rows = _normalize_diagnostic_rows(diagnostic_rows, rhs.size)
    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=REAL_DTYPE)
        if initial_guess.shape != rhs.shape:
            raise ValueError(f"initial_guess must have shape {rhs.shape}; got {initial_guess.shape}")
        _validate_finite_array(initial_guess, "initial_guess")

    matrix_start = time.time()
    petsc_matrix = _petsc_matrix_from_scipy(PETSc, matrix, resource_owner=_resource_owner)
    petsc_matrix_elapsed_seconds = time.time() - matrix_start
    _solver_print(
        verbose,
        2,
        "  PETSc AIJ matrix assembled in %.5fs with nnz=%d",
        petsc_matrix_elapsed_seconds,
        matrix.nnz,
    )

    b = _resource_owner(PETSc.Vec().createWithArray(rhs, comm=PETSc.COMM_WORLD))
    x = _resource_owner(b.duplicate())
    if initial_guess is not None:
        x_array = x.getArray()
        x_array[...] = initial_guess

    ksp = _resource_owner(PETSc.KSP().create(comm=PETSc.COMM_WORLD))
    ksp.setOperators(petsc_matrix)
    preset, option_keys = _configure_petsc_solver(
        PETSc,
        ksp,
        preset=preset,
        levels=levels,
        options=options,
    )
    try:
        ksp.setFromOptions()
    finally:
        _clear_petsc_options(PETSc, option_keys)

    if initial_guess is not None:
        ksp.setInitialGuessNonzero(True)
    if maxiter is None:
        ksp.setTolerances(rtol=rtol, atol=atol, divtol=divtol)
    else:
        ksp.setTolerances(rtol=rtol, atol=atol, divtol=divtol, max_it=int(maxiter))
    if use_monitor:
        ksp.setMonitor(lambda _ksp, its, rnorm: print(f"[petsc it={its}] ||r||={rnorm:g}", flush=True))

    _solver_print(verbose, 1, "  solver: PETSc preset %s", preset)
    _solver_print(
        verbose,
        2,
        "  setting up PETSc KSP/PC with rtol=%g, atol=%g, maxiter=%s",
        rtol,
        atol,
        "PETSc default" if maxiter is None else maxiter,
    )

    setup_start = time.time()
    ksp.setUp()
    setup_elapsed_seconds = time.time() - setup_start
    _solver_print(verbose, 2, "  PETSc KSP/PC setup completed in %.5fs", setup_elapsed_seconds)

    initial_residual_norm = None
    if initial_guess is not None:
        initial_residual_norm = compute_residual_norm(matrix, initial_guess, rhs)
        _solver_print(verbose, 2, "  initial residual from supplied guess: %.3e", initial_residual_norm)

    solve_start = time.time()
    try:
        ksp.solve(b, x)
    except PETSc.Error as exc:
        if preset == "mumps_lu" and "mumps" in str(exc).lower():
            raise RuntimeError(
                "PETSc MUMPS LU requested, but this PETSc build does not appear "
                "to provide MUMPS. Use petsc_preset='lu' or a Krylov preset."
            ) from exc
        raise
    solve_elapsed_seconds = time.time() - solve_start

    solution = np.asarray(x.getArray()).copy()
    residual = matrix @ solution - rhs
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm, relative_residual_norm, residual_target = residual_diagnostics(
        residual_norm,
        rhs,
        rtol=rtol,
        atol=atol,
    )
    diagnostic_residual_norm, diagnostic_rhs_norm, diagnostic_relative_residual_norm, diagnostic_residual_target = (
        restricted_residual_diagnostics(residual, rhs, diagnostic_rows, rtol=rtol, atol=atol)
    )

    reason = int(ksp.getConvergedReason())
    iteration_count = int(ksp.getIterationNumber())
    petsc_residual_norm = float(ksp.getResidualNorm())
    total_elapsed_seconds = time.time() - total_start

    _solver_print(
        verbose,
        1,
        "  PETSc %s finished in %.5fs with reason=%d, iterations=%d, solver_rel=%.3e",
        preset,
        solve_elapsed_seconds,
        reason,
        iteration_count,
        relative_residual_norm,
    )
    _solver_print(
        verbose,
        2,
        "  residuals: solver_abs=%.3e, solver_rhs=%.3e, solver_target=%.3e; petsc_reported=%.3e",
        residual_norm,
        rhs_norm,
        residual_target,
        petsc_residual_norm,
    )
    _solver_print(
        verbose,
        2,
        "  timings: petsc_init=%.5fs, petsc_matrix=%.5fs, petsc_setup=%.5fs, krylov=%.5fs, total=%.5fs",
        petsc_import_elapsed_seconds,
        petsc_matrix_elapsed_seconds,
        setup_elapsed_seconds,
        solve_elapsed_seconds,
        total_elapsed_seconds,
    )

    if reason < 0 and raise_on_nonconvergence:
        raise RuntimeError(
            f"PETSc solver failed to converge. preset={preset}, reason={reason}, "
            f"iterations={iteration_count}, residual={residual_norm:.3e}, "
            f"relative_residual={relative_residual_norm:.3e}"
        )

    return SolveResult(
        x=solution,
        residual_norm=residual_norm,
        info=0 if reason > 0 else reason,
        preconditioner=None,
        total_elapsed_seconds=total_elapsed_seconds,
        scale_elapsed_seconds=0.0,
        preconditioner_elapsed_seconds=setup_elapsed_seconds,
        solve_elapsed_seconds=solve_elapsed_seconds,
        iteration_count=iteration_count,
        initial_residual_norm=initial_residual_norm,
        rhs_norm=rhs_norm,
        relative_residual_norm=relative_residual_norm,
        residual_target=residual_target,
        solver_residual_norm=residual_norm,
        solver_rhs_norm=rhs_norm,
        solver_relative_residual_norm=relative_residual_norm,
        solver_residual_target=residual_target,
        physical_residual_norm=residual_norm,
        physical_rhs_norm=rhs_norm,
        physical_relative_residual_norm=relative_residual_norm,
        physical_residual_target=residual_target,
        diagnostic_residual_label=diagnostic_label,
        diagnostic_residual_norm=diagnostic_residual_norm,
        diagnostic_rhs_norm=diagnostic_rhs_norm,
        diagnostic_relative_residual_norm=diagnostic_relative_residual_norm,
        diagnostic_residual_target=diagnostic_residual_target,
        rtol=rtol,
        atol=atol,
        petsc_preset=preset,
        petsc_converged_reason=reason,
        petsc_residual_norm=petsc_residual_norm,
    )


def solve_petsc_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    *,
    preset: str = "cg_gamg",
    levels: int | None = None,
    options: Mapping[str, Any] | None = None,
    initial_guess: NDArray | None = None,
    rtol: float = 1e-13,
    atol: float = 0.0,
    divtol: float = 1e4,
    maxiter: int | None = None,
    use_monitor: bool = False,
    raise_on_nonconvergence: bool = False,
    verbose: bool | int = 0,
    diagnostic_rows: NDArray | None = None,
    diagnostic_label: str | None = None,
) -> SolveResult:
    """Solve with PETSc and deterministically destroy all native objects."""
    resources = []

    def own(resource):
        """Register a PETSc object for deterministic cleanup."""
        resources.append(resource)
        return resource

    try:
        result = _solve_petsc_system_impl(
            matrix,
            rhs,
            preset=preset,
            levels=levels,
            options=options,
            initial_guess=initial_guess,
            rtol=rtol,
            atol=atol,
            divtol=divtol,
            maxiter=maxiter,
            use_monitor=use_monitor,
            raise_on_nonconvergence=False,
            verbose=verbose,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label=diagnostic_label,
            _resource_owner=own,
        )
    finally:
        for resource in reversed(resources):
            try:
                resource.destroy()
            except (AttributeError, ReferenceError):
                pass

    return finalize_solve_result(
        result,
        backend=f"petsc-{result.petsc_preset or preset}",
        backend_info=result.petsc_converged_reason,
        backend_success=(result.petsc_converged_reason or 0) > 0,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )


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


def solve_iterative_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    *,
    solver_name: str,
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
    scale_system: bool = True,
    raise_on_nonconvergence: bool = False,
    verbose: bool | int = 0,
    prepared_scaled_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
    prepared_inverse_diagonal: NDArray | None = None,
    scale_matrix_in_place: bool = False,
    diagnostic_rows: NDArray | None = None,
    diagnostic_label: str | None = None,
    validate_matrix: bool = True,
) -> SolveResult:
    """Solve an already assembled sparse system with a SciPy Krylov method."""
    _validate_solver_controls(rtol=rtol, atol=atol, maxiter=maxiter, restart=restart)
    matrix = matrix.tocsr()
    rhs = np.asarray(rhs, dtype=REAL_DTYPE)
    if matrix.shape != (rhs.size, rhs.size):
        raise ValueError(f"matrix must have shape ({rhs.size}, {rhs.size}), got {matrix.shape}")
    if validate_matrix:
        _validate_finite_array(np.asarray(matrix.data), "matrix data")
    _validate_finite_array(rhs, "rhs")
    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=REAL_DTYPE)
        if initial_guess.shape != rhs.shape:
            raise ValueError(f"initial_guess must have shape {rhs.shape}, got {initial_guess.shape}")
        _validate_finite_array(initial_guess, "initial_guess")

    total_start = time.time()

    solver = get_iterative_solver(solver_name)
    solver_name_upper = solver_name.upper()
    symmetric_solver = solver_name_upper in {"CG", "MINRES"}
    if symmetric_solver and scale_system:
        raise ValueError(
            f"{solver_name_upper} requires scale_system=False because the current "
            "scaling is left row scaling, which does not preserve matrix symmetry. "
            "Use preconditioner='jacobi' for cheap symmetric diagonal preconditioning."
        )
    if symmetric_solver and isinstance(preconditioner, str) and preconditioner == "ilu":
        raise ValueError(
            f"{solver_name_upper} should not be used with the nonsymmetric ILU "
            "preconditioner. Use preconditioner='jacobi' or preconditioner=None."
        )

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

    if scale_system:
        _solver_print(verbose, 2, "  diagonal scaling completed in %.5fs", scale_elapsed_seconds)
    else:
        _solver_print(
            verbose,
            2,
            "  diagonal scaling disabled; CSR conversion completed in %.5fs",
            scale_elapsed_seconds,
        )

    preconditioner_elapsed_seconds = 0.0

    if isinstance(preconditioner, str):
        if preconditioner == "ilu":
            if ilu_failure not in {"raise", "none"}:
                raise ValueError("ilu_failure must be 'raise' or 'none'")
            _solver_print(
                verbose,
                2,
                "  building ILU preconditioner with drop_tol=%g, fill_factor=%g",
                ilu_drop_tol,
                ilu_fill_factor,
            )
            _solver_print(verbose, 2, "  SuperLU ILU column permutation: %s", ilu_permc_spec)

            preconditioner_start = time.time()
            try:
                M = build_ilu_preconditioner(
                    scaled_matrix,
                    drop_tol=ilu_drop_tol,
                    fill_factor=ilu_fill_factor,
                    permc_spec=ilu_permc_spec,
                )
            except RuntimeError as exc:
                preconditioner_elapsed_seconds = time.time() - preconditioner_start
                if ilu_failure == "none":
                    M = None
                    _solver_print(
                        verbose,
                        1,
                        "  ILU preconditioner failed in %.5fs (%s); continuing without preconditioner",
                        preconditioner_elapsed_seconds,
                        exc,
                    )
                else:
                    raise RuntimeError(
                        "ILU preconditioner failed. Try a larger fill_factor, "
                        "a smaller drop_tol, or set ilu_failure='none' to fall back "
                        "to an unpreconditioned Krylov solve."
                    ) from exc
            else:
                preconditioner_elapsed_seconds = time.time() - preconditioner_start
                _solver_print(verbose, 2, "  ILU preconditioner built in %.5fs", preconditioner_elapsed_seconds)
        elif preconditioner == "jacobi":
            _solver_print(verbose, 2, "  building Jacobi preconditioner")
            preconditioner_start = time.time()
            M = build_jacobi_preconditioner(scaled_matrix)
            preconditioner_elapsed_seconds = time.time() - preconditioner_start
            _solver_print(verbose, 2, "  Jacobi preconditioner built in %.5fs", preconditioner_elapsed_seconds)
        else:
            raise ValueError("preconditioner must be 'ilu', 'jacobi', None, or a LinearOperator")

    else:
        M = preconditioner

        if M is None:
            _solver_print(verbose, 2, "  no preconditioner used")
        else:
            _solver_print(verbose, 2, "  using supplied preconditioner")

    # For vector-valued callbacks, counting should remain O(1). Computing an
    # explicit residual here adds a full sparse matvec to every Krylov
    # iteration and can dominate otherwise short, strongly preconditioned
    # solves. Final residuals are still computed and validated below.
    iteration_counter = KrylovIterationCounter(
        verbose=verbose,
        solver_name=solver_name_upper,
    )

    solver_kwargs = {
        "M": M,
        "rtol": rtol,
        "callback": iteration_counter,
    }
    if solver_name_upper != "MINRES":
        solver_kwargs["atol"] = atol

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

    _solver_print(
        verbose,
        2,
        "  starting %s solve with rtol=%g, atol=%g, maxiter=%s",
        solver_name_upper,
        rtol,
        atol,
        "default" if maxiter is None else maxiter,
    )

    initial_residual_norm = None
    initial_physical_residual_norm = None
    initial_residual_start = time.perf_counter()
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
        _solver_print(
            verbose,
            2,
            "  initial residual from supplied guess: solver %.3e; physical %.3e",
            initial_residual_norm,
            initial_physical_residual_norm,
        )
    elif _verbosity_level(verbose) >= 3:
        zero_guess = np.zeros_like(scaled_rhs)
        initial_solver_residual = scaled_matrix @ zero_guess - scaled_rhs
        zero_initial_residual_norm = float(np.linalg.norm(initial_solver_residual))
        zero_initial_physical_residual_norm = (
            float(np.linalg.norm(physical_diagonal * initial_solver_residual))
            if scaled_in_place
            else compute_residual_norm(matrix, zero_guess, rhs)
        )
        _solver_print(
            verbose,
            3,
            "  initial residual from zero guess: solver %.3e; physical %.3e",
            zero_initial_residual_norm,
            zero_initial_physical_residual_norm,
        )
    initial_residual_elapsed_seconds = time.perf_counter() - initial_residual_start

    preconditioner_apply_count_before = int(getattr(M, "apply_count", 0) or 0)
    preconditioner_apply_seconds_before = float(getattr(M, "apply_seconds", 0.0) or 0.0)

    solve_start = time.time()
    x, info = solver(scaled_matrix, scaled_rhs, **solver_kwargs)
    solve_elapsed_seconds = time.time() - solve_start

    x = np.asarray(x)

    residual_diagnostics_start = time.perf_counter()
    # Solver diagnostics: match the exact system and RHS used by SciPy.
    solver_residual = scaled_matrix @ x - scaled_rhs
    solver_residual_norm = float(np.linalg.norm(solver_residual))
    solver_rhs_norm, solver_relative_residual_norm, solver_residual_target = residual_diagnostics(
        solver_residual_norm, scaled_rhs, rtol=rtol, atol=atol
    )

    # Physical diagnostics: useful for original-system interpretation, but not
    # the criterion SciPy used when scale_system=True.
    if scaled_in_place:
        physical_residual = physical_diagonal * solver_residual
    else:
        physical_residual = matrix @ x - rhs
    physical_residual_norm = float(np.linalg.norm(physical_residual))
    physical_rhs_norm, physical_relative_residual_norm, physical_residual_target = residual_diagnostics(
        physical_residual_norm, rhs, rtol=rtol, atol=atol
    )
    diagnostic_residual_norm, diagnostic_rhs_norm, diagnostic_relative_residual_norm, diagnostic_residual_target = (
        restricted_residual_diagnostics(physical_residual, rhs, diagnostic_rows, rtol=rtol, atol=atol)
    )
    residual_diagnostics_elapsed_seconds = time.perf_counter() - residual_diagnostics_start

    total_elapsed_seconds = time.time() - total_start

    # Optional preconditioner timing diagnostics.
    #
    # A plain scipy LinearOperator does not provide these. Our
    # ElementSchwarzPreconditioner can expose them by attaching attributes
    # to the LinearOperator or by being reachable from the solver object.
    cumulative_prec_apply_count = getattr(M, "apply_count", None)
    cumulative_prec_apply_seconds = getattr(M, "apply_seconds", None)
    prec_apply_count = (
        None
        if cumulative_prec_apply_count is None
        else int(cumulative_prec_apply_count) - preconditioner_apply_count_before
    )
    prec_apply_seconds = (
        None
        if cumulative_prec_apply_seconds is None
        else float(cumulative_prec_apply_seconds) - preconditioner_apply_seconds_before
    )
    prec_local_solve_seconds = getattr(M, "local_solve_seconds", None)
    prec_reduce_seconds = getattr(M, "reduce_seconds", None)
    prec_copy_seconds = getattr(M, "copy_seconds", None)

    _solver_print(
        verbose,
        1,
        "  %s finished in %.5fs with info=%s, iterations=%d, solver_rel=%.3e",
        solver_name_upper,
        solve_elapsed_seconds,
        info,
        iteration_counter.count,
        solver_relative_residual_norm,
    )
    _solver_print(
        verbose,
        2,
        "  residuals: solver_abs=%.3e, solver_rhs=%.3e, solver_target=%.3e; "
        "penalized_physical_abs=%.3e, penalized_physical_rhs=%.3e, penalized_physical_rel=%.3e",
        solver_residual_norm,
        solver_rhs_norm,
        solver_residual_target,
        physical_residual_norm,
        physical_rhs_norm,
        physical_relative_residual_norm,
    )
    if diagnostic_residual_norm is not None:
        _solver_print(
            verbose,
            2,
            "  %s residual: abs=%.3e, rhs=%.3e, rel=%.3e, target=%.3e",
            diagnostic_label or "restricted-row",
            diagnostic_residual_norm,
            diagnostic_rhs_norm,
            diagnostic_relative_residual_norm,
            diagnostic_residual_target,
        )
    _solver_print(
        verbose,
        2,
        "  timings: scaling=%.5fs, preconditioner=%.5fs, krylov=%.5fs, total=%.5fs",
        scale_elapsed_seconds,
        preconditioner_elapsed_seconds,
        solve_elapsed_seconds,
        total_elapsed_seconds,
    )
    _solver_print(
        verbose,
        2,
        "  iterative breakdown: initial_residual=%.5fs, callback=%.5fs, "
        "preconditioner_apply=%.5fs, final_residuals=%.5fs",
        initial_residual_elapsed_seconds,
        iteration_counter.elapsed_seconds,
        0.0 if prec_apply_seconds is None else prec_apply_seconds,
        residual_diagnostics_elapsed_seconds,
    )

    if prec_apply_count is not None:
        _solver_print(
            verbose,
            2,
            "  preconditioner applies: count=%s, total=%.5fs, local=%.5fs, reduce=%.5fs, copy=%.5fs",
            prec_apply_count,
            0.0 if prec_apply_seconds is None else prec_apply_seconds,
            0.0 if prec_local_solve_seconds is None else prec_local_solve_seconds,
            0.0 if prec_reduce_seconds is None else prec_reduce_seconds,
            0.0 if prec_copy_seconds is None else prec_copy_seconds,
        )

    result = SolveResult(
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
        initial_residual_elapsed_seconds=initial_residual_elapsed_seconds,
        callback_elapsed_seconds=iteration_counter.elapsed_seconds,
        residual_diagnostics_elapsed_seconds=residual_diagnostics_elapsed_seconds,
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
        diagnostic_residual_label=diagnostic_label,
        diagnostic_residual_norm=diagnostic_residual_norm,
        diagnostic_rhs_norm=diagnostic_rhs_norm,
        diagnostic_relative_residual_norm=diagnostic_relative_residual_norm,
        diagnostic_residual_target=diagnostic_residual_target,
        rtol=rtol,
        atol=atol,
        preconditioner_apply_count=prec_apply_count,
        preconditioner_apply_seconds=prec_apply_seconds,
        preconditioner_local_solve_seconds=prec_local_solve_seconds,
        preconditioner_reduce_seconds=prec_reduce_seconds,
        preconditioner_copy_seconds=prec_copy_seconds,
        preconditioner_factor_nnz=getattr(M, "factor_nnz", None),
        ilu_permc_spec=ilu_permc_spec if isinstance(preconditioner, str) and preconditioner == "ilu" else None,
    )
    return finalize_solve_result(
        result,
        backend=f"scipy-{solver_name_upper.lower()}",
        backend_info=int(info),
        backend_success=int(info) == 0,
        residual_history=iteration_counter.residual_history,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )
