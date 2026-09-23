"""Sparse global-system assembly and solve helpers.

This module is the self-contained :mod:`hdgfem` version of the trace-system
solver used by the HDG assembly code.  It builds CSR matrices from COO triplets
and provides direct or Krylov solves with optional diagonal scaling, ILU
preconditioning, and cheap diagonal Jacobi preconditioning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Literal

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

_ITERATIVE_SOLVERS = {
    "BICG": bicg,
    "BICGSTAB": bicgstab,
    "CG": cg,
    "CGS": cgs,
    "GMRES": gmres,
    "LGMRES": lgmres,
    "MINRES": minres,
}


def _verbosity_level(verbose: bool | int) -> int:
    """Normalize solver verbosity flags to levels 0, 1, or 2."""
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    return min(2, max(0, int(verbose)))


def _solver_print(verbose: bool | int, level: int, message: str, *args) -> None:
    """Print formatted solver diagnostics when ``verbose`` reaches ``level``."""
    if _verbosity_level(verbose) >= level:
        print(message % args if args else message, flush=True)


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

    # PETSc diagnostics, when the PETSc backend is used.
    petsc_preset: str | None = None
    petsc_converged_reason: int | None = None
    petsc_residual_norm: float | None = None

    # Symmetric matrix permutation diagnostics.
    permutation_elapsed_seconds: float | None = None
    permutation_size: int | None = None
    ilu_permc_spec: str | None = None


@dataclass(frozen=True)
class KnownDofReduction:
    """Reduced COO system obtained by eliminating prescribed dofs."""

    rows: NDArray
    cols: NDArray
    data: NDArray
    rhs: NDArray
    free_mask: NDArray
    known_mask: NDArray
    known_values: NDArray
    old_to_new: NDArray


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


def eliminate_known_dofs(
    row_indices: NDArray,
    col_indices: NDArray,
    matrix_values: NDArray,
    rhs: NDArray,
    known_mask: NDArray,
    known_values: NDArray,
) -> KnownDofReduction:
    r"""Eliminate prescribed unknowns from a COO linear system.

    Given :math:`Ax=b` and known entries :math:`x_k=g`, this returns the
    reduced free-dof system

    .. math::

        A_{ff} x_f = b_f - A_{fk} g.

    The implementation is fully vectorized over the COO triplets.  Rows whose
    unknown is prescribed are dropped, columns whose unknown is prescribed are
    accumulated into the reduced RHS, and free/free entries are remapped to
    compact reduced indices.
    """
    row_indices = np.asarray(row_indices)
    col_indices = np.asarray(col_indices)
    matrix_values = np.asarray(matrix_values, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    known_mask = np.asarray(known_mask, dtype=bool)
    known_values = np.asarray(known_values, dtype=np.float64)

    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")
    system_size = rhs.size
    if known_mask.shape != (system_size,):
        raise ValueError(f"known_mask must have shape ({system_size},), got {known_mask.shape}")
    if known_values.shape != (system_size,):
        raise ValueError(f"known_values must have shape ({system_size},), got {known_values.shape}")
    validate_global_system_inputs(
        row_indices,
        col_indices,
        matrix_values,
        rhs,
        system_size,
    )

    free_mask = ~known_mask
    old_to_new = np.full(system_size, -1, dtype=np.int64)
    old_to_new[free_mask] = np.arange(np.count_nonzero(free_mask), dtype=np.int64)

    row_is_free = free_mask[row_indices]
    col_is_free = free_mask[col_indices]
    free_free = row_is_free & col_is_free
    free_known = row_is_free & ~col_is_free

    reduced_rows = old_to_new[row_indices[free_free]]
    reduced_cols = old_to_new[col_indices[free_free]]
    reduced_data = matrix_values[free_free].copy()
    reduced_rhs = rhs[free_mask].copy()

    if np.any(free_known):
        np.add.at(
            reduced_rhs,
            old_to_new[row_indices[free_known]],
            -matrix_values[free_known] * known_values[col_indices[free_known]],
        )

    return KnownDofReduction(
        rows=np.ascontiguousarray(reduced_rows),
        cols=np.ascontiguousarray(reduced_cols),
        data=np.ascontiguousarray(reduced_data),
        rhs=np.ascontiguousarray(reduced_rhs),
        free_mask=np.ascontiguousarray(free_mask),
        known_mask=np.ascontiguousarray(known_mask),
        known_values=np.ascontiguousarray(known_values),
        old_to_new=np.ascontiguousarray(old_to_new),
    )


def expand_known_dofs(reduced_solution: NDArray, reduction: KnownDofReduction) -> NDArray:
    """Expand a reduced solution by reinserting prescribed dof values."""
    reduced_solution = np.asarray(reduced_solution, dtype=np.float64)
    expected_shape = (np.count_nonzero(reduction.free_mask),)
    if reduced_solution.shape != expected_shape:
        raise ValueError(f"reduced_solution must have shape {expected_shape}; got {reduced_solution.shape}")
    full_solution = reduction.known_values.copy()
    full_solution[reduction.free_mask] = reduced_solution
    return np.ascontiguousarray(full_solution)


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

    return LinearOperator(matrix.shape, matvec=ilu_decomposition.solve)


def build_jacobi_preconditioner(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
) -> scipy.sparse.linalg.LinearOperator:
    r"""Build the diagonal Jacobi preconditioner :math:`M^{-1}=D^{-1}`.

    This is the cheap preconditioner to use with symmetric Krylov solvers such
    as CG or MINRES on reduced diffusion trace systems.  The matrix diagonal
    must be strictly positive; otherwise the diagonal preconditioner would not
    be symmetric positive definite.
    """
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("Jacobi preconditioner diagonal contains non-finite entries")
    if np.any(diagonal <= 0.0):
        raise ValueError("Jacobi preconditioner requires a strictly positive diagonal")

    inverse_diagonal = 1.0 / diagonal

    def apply(vector):
        return inverse_diagonal * vector

    return LinearOperator(matrix.shape, matvec=apply, rmatvec=apply, dtype=np.float64)


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


def _petsc_matrix_from_scipy(PETSc, matrix: scipy.sparse.spmatrix | scipy.sparse.sparray):
    """Build a PETSc AIJ matrix from a SciPy sparse matrix."""
    petsc_matrix = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    if hasattr(petsc_matrix, "setPreallocationCOO"):
        coo = matrix.tocoo(copy=False)
        rows = np.asarray(coo.row, dtype=PETSc.IntType)
        cols = np.asarray(coo.col, dtype=PETSc.IntType)
        values = np.asarray(coo.data, dtype=np.float64)
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
    values = np.asarray(csr.data, dtype=np.float64)
    petsc_matrix.setSizes(matrix.shape)
    petsc_matrix.setType(PETSc.Mat.Type.AIJ)
    petsc_matrix.setPreallocationCSR((row_pointers, column_indices))
    petsc_matrix.setValuesCSR(row_pointers, column_indices, values)
    petsc_matrix.assemble()
    return petsc_matrix


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
    """Solve a sparse system with PETSc KSP/PC presets.

    The intended diffusion-reaction path is ``preset="cg_gamg"`` on the
    boundary-eliminated trace system.  PETSc is imported lazily, so the rest of
    :mod:`hdgfem` remains usable without petsc4py.
    """
    total_start = time.time()
    import_start = time.time()
    PETSc = _import_petsc()
    petsc_import_elapsed_seconds = time.time() - import_start
    _solver_print(verbose, 2, "  PETSc runtime initialized in %.5fs", petsc_import_elapsed_seconds)
    matrix = matrix.tocsr()
    rhs = np.asarray(rhs, dtype=np.float64)
    diagnostic_rows = _normalize_diagnostic_rows(diagnostic_rows, rhs.size)

    matrix_start = time.time()
    petsc_matrix = _petsc_matrix_from_scipy(PETSc, matrix)
    petsc_matrix_elapsed_seconds = time.time() - matrix_start
    _solver_print(
        verbose,
        2,
        "  PETSc AIJ matrix assembled in %.5fs with nnz=%d",
        petsc_matrix_elapsed_seconds,
        matrix.nnz,
    )

    b = PETSc.Vec().createWithArray(rhs, comm=PETSc.COMM_WORLD)
    x = b.duplicate()
    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=np.float64)
        if initial_guess.shape != rhs.shape:
            raise ValueError(f"initial_guess must have shape {rhs.shape}; got {initial_guess.shape}")
        x_array = x.getArray()
        x_array[...] = initial_guess

    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
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
    diagnostic_rows: NDArray | None = None,
    diagnostic_label: str | None = None,
) -> SolveResult:
    """Solve an already assembled sparse system with ``spsolve``."""
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
) -> SolveResult:
    """Solve an already assembled sparse system with a SciPy Krylov method."""
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

    iteration_counter = KrylovIterationCounter()

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
        ilu_permc_spec=ilu_permc_spec if isinstance(preconditioner, str) and preconditioner == "ilu" else None,
    )


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
    petsc_preset: str = "cg_gamg",
    petsc_levels: int | None = None,
    petsc_options: Mapping[str, Any] | None = None,
    petsc_divtol: float = 1e4,
    petsc_monitor: bool = False,
    scale_system: bool = True,
    raise_on_nonconvergence: bool = False,
    verbose: bool | int = 0,
    assembled_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
    prepared_scaled_matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None = None,
    prepared_inverse_diagonal: NDArray | None = None,
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
        ``"petsc"`` uses the PETSc backend. Other names are looked up in the
        supported SciPy Krylov solver table.
    preconditioner
        ``"ilu"`` builds a SciPy ILU preconditioner, ``"jacobi"`` builds a
        cheap diagonal Jacobi preconditioner, ``None`` disables
        preconditioning, and a supplied :class:`LinearOperator` is passed
        directly to SciPy.
    scale_system
        Apply left Jacobi scaling before iterative solves.
    ilu_failure
        Behavior when SciPy ILU factorization fails.  ``"raise"`` preserves
        the failure, while ``"none"`` continues with no preconditioner.
    ilu_permc_spec
        SuperLU column permutation used by ``spilu``.  Use ``"NATURAL"`` when
        the caller has already supplied a meaningful matrix ``permutation``.
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
    row_indices = np.asarray(row_indices)
    col_indices = np.asarray(col_indices)
    matrix_values = np.asarray(matrix_values)
    rhs = np.asarray(rhs)

    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess)
    diagnostic_rows = _normalize_diagnostic_rows(diagnostic_rows, system_size)
    permutation = _normalize_permutation(permutation, system_size)

    if assembled_matrix is None:
        validate_global_system_inputs(
            row_indices=row_indices,
            col_indices=col_indices,
            matrix_values=matrix_values,
            rhs=rhs,
            system_size=system_size,
            initial_guess=initial_guess,
        )
        assembly_start = time.time()
        matrix = assemble_global_matrix(
            row_indices=row_indices,
            col_indices=col_indices,
            matrix_values=matrix_values,
            system_size=system_size,
        )
        _solver_print(
            verbose,
            2,
            "  sparse CSR matrix assembled in %.5fs with nnz=%d",
            time.time() - assembly_start,
            matrix.nnz,
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

    solver_is_petsc = solver is not None and str(solver).lower() == "petsc"

    if solver is None or solver == "direct":
        _solver_print(verbose, 1, "  solver: scipy.sparse.linalg.spsolve")

        result = solve_direct_system(
            matrix,
            rhs,
            rtol=rtol,
            atol=atol,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label=diagnostic_label,
        )

    elif solver_is_petsc:
        if scale_system:
            _solver_print(
                verbose,
                2,
                "  SciPy left scaling is disabled for PETSc; PETSc handles scaling/preconditioning internally",
            )
        result = solve_petsc_system(
            matrix,
            rhs,
            preset=petsc_preset,
            levels=petsc_levels,
            options=petsc_options,
            initial_guess=initial_guess,
            rtol=rtol,
            atol=atol,
            divtol=petsc_divtol,
            maxiter=maxiter,
            use_monitor=petsc_monitor,
            raise_on_nonconvergence=raise_on_nonconvergence,
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
            raise_on_nonconvergence=raise_on_nonconvergence,
            verbose=verbose,
            prepared_scaled_matrix=prepared_scaled_matrix,
            prepared_inverse_diagonal=prepared_inverse_diagonal,
            scale_matrix_in_place=scale_matrix_in_place,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label=diagnostic_label,
        )

    if permutation is not None:
        unpermuted = np.empty_like(result.x)
        unpermuted[permutation] = result.x
        result.x = unpermuted
        result.permutation_elapsed_seconds = permutation_elapsed_seconds
        result.permutation_size = int(permutation.size)

    if result.info == 0:
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

    return result


__all__ = [
    "KrylovIterationCounter",
    "KnownDofReduction",
    "SolveResult",
    "assemble_global_matrix",
    "build_ilu_preconditioner",
    "build_jacobi_preconditioner",
    "compute_residual_norm",
    "diagonal_scale_system",
    "eliminate_known_dofs",
    "expand_known_dofs",
    "get_iterative_solver",
    "residual_diagnostics",
    "solve_direct_system",
    "solve_global_system",
    "solve_iterative_system",
    "solve_petsc_system",
    "validate_global_system_inputs",
]
