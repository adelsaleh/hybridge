"""hdgfem.linalg.gpu.cupyx."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
import time
from typing import Any
from hdgfem.runtime.precision import REAL_DTYPE
from hdgfem.runtime.optional import require_cupy, require_cupyx_sparse_linalg
from hdgfem.linalg.gpu.sparse import scipy_csr_to_cupy

from typing import Literal
from numpy.typing import NDArray
from hdgfem.linalg.results import (
    SolveResult,
    _normalize_diagnostic_rows,
    _solver_print,
    _validate_finite_array,
    _validate_solver_controls,
    compute_residual_norm,
    diagonal_scale_system,
    finalize_solve_result,
    residual_diagnostics,
    restricted_residual_diagnostics,
)



def build_cupyx_ilu_preconditioner(
        matrix,
        *,
        drop_tol: float,
        fill_factor: float,
        permc_spec: str | None,
):
    """Build a CuPy ILU preconditioner directly on the device."""
    linalg = require_cupyx_sparse_linalg()
    cupy = require_cupy()
    kwargs = {
        "drop_tol": float(drop_tol),
        "fill_factor": float(fill_factor),
    }
    if permc_spec is not None:
        kwargs["permc_spec"] = permc_spec
    ilu = linalg.spilu(matrix, **kwargs)
    state = {"count": 0, "seconds": 0.0}

    def matvec(vec):
        """Apply a matrix-vector product."""
        start = time.perf_counter()
        out = ilu.solve(vec)
        cupy.cuda.get_current_stream().synchronize()
        state["count"] += 1
        state["seconds"] += time.perf_counter() - start
        operator.apply_count = state["count"]
        operator.apply_seconds = state["seconds"]
        return out

    operator = linalg.LinearOperator(matrix.shape, matvec=matvec, dtype=matrix.dtype)
    operator.apply_count = 0
    operator.apply_seconds = 0.0
    return operator


def build_cupyx_exported_host_ilu_preconditioner(
        matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
        *,
        drop_tol: float,
        fill_factor: float,
        permc_spec: str | None,
        dtype=None,
):
    """Build SciPy ILU on host, then apply its factors on the GPU.

    SciPy SuperLU stores factors satisfying ``Pr A Pc = L U``.  The returned
    CuPy ``LinearOperator`` applies ``M^{-1}`` as two device sparse triangular
    solves with the exported ``L`` and ``U`` factors.  Host memory is touched
    only while constructing and transferring the factorization.
    """
    cupy = require_cupy()
    linalg = require_cupyx_sparse_linalg()
    if dtype is None:
        dtype = REAL_DTYPE

    matrix_csc = matrix.tocsc()
    matrix_csc.eliminate_zeros()
    kwargs = {
        "drop_tol": float(drop_tol),
        "fill_factor": float(fill_factor),
    }
    if permc_spec is not None:
        kwargs["permc_spec"] = permc_spec
    ilu = scipy.sparse.linalg.spilu(matrix_csc, **kwargs)

    lower = scipy_csr_to_cupy(ilu.L.tocsr(), dtype=dtype)
    upper = scipy_csr_to_cupy(ilu.U.tocsr(), dtype=dtype)
    inv_perm_r = cupy.asarray(np.argsort(np.asarray(ilu.perm_r, dtype=np.int64)), dtype=cupy.int64)
    perm_c = cupy.asarray(np.asarray(ilu.perm_c, dtype=np.int64), dtype=cupy.int64)
    cupy.cuda.get_current_stream().synchronize()

    def matvec(vec):
        """Apply a matrix-vector product."""
        rhs_perm = vec[inv_perm_r]
        y = linalg.spsolve_triangular(lower, rhs_perm, lower=True, unit_diagonal=True)
        z = linalg.spsolve_triangular(upper, y, lower=False)
        return z[perm_c]

    operator = linalg.LinearOperator(matrix.shape, matvec=matvec, dtype=dtype)
    operator.host_ilu_nnz = int(ilu.L.nnz + ilu.U.nnz)
    operator.host_ilu_fill_ratio = float(operator.host_ilu_nnz) / max(int(matrix_csc.nnz), 1)
    return operator


def solve_cupyx_csr(
        matrix,
        rhs,
        *,
        solver: str = "cg",
        preconditioner=None,
        initial_guess=None,
        rtol: float = 1e-13,
        atol: float = 0.0,
        maxiter: int | None = None,
        restart: int | None = None,
):
    """Solve a CuPy CSR system with ``cupyx.scipy.sparse.linalg``.

    Parameters are intentionally close to :func:`scipy.sparse.linalg` Krylov
    solvers.  ``matrix`` is expected to already be a CuPy sparse matrix; callers
    that assemble on the host should convert once with :func:`scipy_csr_to_cupy`
    and cache that device matrix when the operator is reused.

    CuPy has used both SciPy's modern ``rtol``/``atol`` convention and older
    ``tol``-only signatures across releases.  The wrapper first tries
    ``rtol``/``atol`` and falls back to ``tol`` so this optional backend remains
    version tolerant.
    """
    cupy = require_cupy()
    linalg = require_cupyx_sparse_linalg()
    normalized = str(solver).lower().replace("-", "_")
    aliases = {
        "bicgstab": "bicgstab",
        "bicg_stab": "bicgstab",
        "bcgs": "bicgstab",
        "cg": "cg",
        "cgs": "cgs",
        "gmres": "gmres",
    }
    solver_name = aliases.get(normalized, normalized)
    if solver_name not in {"cg", "bicgstab", "cgs", "gmres"}:
        raise ValueError("cupyx solver must be one of 'cg', 'bicgstab', 'cgs', or 'gmres'")
    solver_fn = getattr(linalg, solver_name, None)
    if solver_fn is None:
        raise RuntimeError(f"cupyx.scipy.sparse.linalg.{solver_name} is not available")

    rhs_cp = cupy.asarray(rhs, dtype=matrix.dtype)
    kwargs: dict[str, Any] = {}
    if initial_guess is not None:
        x0 = cupy.asarray(initial_guess, dtype=matrix.dtype)
        if x0.shape != rhs_cp.shape:
            raise ValueError(f"initial_guess must have shape {rhs_cp.shape}; got {x0.shape}")
        kwargs["x0"] = x0
    if maxiter is not None:
        kwargs["maxiter"] = int(maxiter)
    if restart is not None and solver_name == "gmres":
        kwargs["restart"] = int(restart)
    if preconditioner is not None:
        kwargs["M"] = preconditioner

    class _IterationCounter:
        """Callback class that counts iterative solver callback invocations."""

        def __init__(self):
            """Initialize the instance."""
            self.count = 0

        def __call__(self, *_args, **_kwargs):
            """Execute the configured call behavior."""
            self.count += 1

    counter = _IterationCounter()
    kwargs["callback"] = counter

    try:
        solution, info = solver_fn(matrix, rhs_cp, rtol=rtol, atol=atol, **kwargs)
    except TypeError:
        solution, info = solver_fn(matrix, rhs_cp, tol=rtol, **kwargs)
    cupy.cuda.get_current_stream().synchronize()
    return solution, int(info), counter.count


def _coo_diagonal(
    row_indices: NDArray,
    col_indices: NDArray,
    matrix_values: NDArray,
    system_size: int,
) -> NDArray:
    """Return the diagonal of a COO matrix without constructing CSR."""
    diagonal = np.zeros(system_size, dtype=REAL_DTYPE)
    diagonal_mask = row_indices == col_indices
    if np.any(diagonal_mask):
        np.add.at(diagonal, row_indices[diagonal_mask], matrix_values[diagonal_mask])
    return diagonal


def _coo_matvec(
    row_indices: NDArray,
    col_indices: NDArray,
    matrix_values: NDArray,
    x: NDArray,
    system_size: int,
) -> NDArray:
    """Compute ``A @ x`` from COO triplets without constructing CSR."""
    return np.bincount(
        row_indices,
        weights=matrix_values * x[col_indices],
        minlength=system_size,
    ).astype(REAL_DTYPE, copy=False)


def _coo_residual(
    row_indices: NDArray,
    col_indices: NDArray,
    matrix_values: NDArray,
    x: NDArray,
    rhs: NDArray,
    system_size: int,
) -> NDArray:
    """Compute ``A @ x - rhs`` from COO triplets without constructing CSR."""
    return _coo_matvec(row_indices, col_indices, matrix_values, x, system_size) - rhs


def solve_cupyx_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray | None,
    rhs: NDArray,
    *,
    row_indices: NDArray | None = None,
    col_indices: NDArray | None = None,
    matrix_values: NDArray | None = None,
    system_size: int | None = None,
    cupyx_solver: str = "bicgstab",
    preconditioner: Any = None,
    prepared_device_matrix: Any | None = None,
    initial_guess: NDArray | None = None,
    rtol: float = 1e-13,
    atol: float = 0.0,
    maxiter: int | None = None,
    restart: int | None = None,
    scale_system: bool = True,
    ilu_drop_tol: float = 1e-10,
    ilu_fill_factor: float = 35,
    ilu_failure: Literal["raise", "none"] = "raise",
    ilu_permc_spec: str | None = None,
    upwind_block_size: int | None = None,
    upwind_level_widths: Any | None = None,
    upwind_diagonal_regularization: float = 0.0,
    raise_on_nonconvergence: bool = False,
    materialize_host_solution: bool = True,
    verbose: bool | int = 0,
    diagnostic_rows: NDArray | None = None,
    diagnostic_label: str | None = None,
) -> SolveResult:
    """Solve a sparse system with Cupyx Krylov solvers.

    When ``matrix`` is ``None``, host-assembled COO triplets are copied to the
    GPU and converted to CSR there.  This avoids constructing a host CSR matrix
    for the Cupyx path while preserving host-side residual diagnostics via COO
    scatter operations.
    """
    _validate_solver_controls(rtol=rtol, atol=atol, maxiter=maxiter, restart=restart)
    total_start = time.time()
    from hdgfem.runtime.optional import asnumpy
    from hdgfem.linalg.gpu.sparse import scipy_coo_to_cupy_csr, scipy_csr_to_cupy

    import os

    cupy_dtype_name = os.environ.get("HDGFEM_CUPYX_DTYPE", np.dtype(REAL_DTYPE).name).lower()
    if cupy_dtype_name in {"fp32", "single"}:
        cupy_dtype_name = "float32"
    elif cupy_dtype_name in {"fp64", "double"}:
        cupy_dtype_name = "float64"
    if cupy_dtype_name not in {"float32", "float64"}:
        raise ValueError("HDGFEM_CUPYX_DTYPE must be 'float32' or 'float64'")
    cupy = __import__("cupy")
    cupy_dtype = getattr(cupy, cupy_dtype_name)

    normalized_cupyx_solver = str(cupyx_solver).lower().replace("-", "_")
    if normalized_cupyx_solver == "cg" and scale_system:
        raise ValueError(
            "Cupyx CG requires scale_system=False because the current scaling is "
            "left row scaling, which does not preserve matrix symmetry. Use "
            "cupyx_solver='bicgstab' or set scale_system=False for SPD systems."
        )

    physical_rhs = np.asarray(rhs, dtype=REAL_DTYPE)
    diagnostic_rows = _normalize_diagnostic_rows(diagnostic_rows, physical_rhs.size)
    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=REAL_DTYPE)
        if initial_guess.shape != physical_rhs.shape:
            raise ValueError(f"initial_guess must have shape {physical_rhs.shape}; got {initial_guess.shape}")

    using_host_coo = matrix is None
    if using_host_coo:
        if row_indices is None or col_indices is None or matrix_values is None or system_size is None:
            raise ValueError("row_indices, col_indices, matrix_values, and system_size are required when matrix is None")
        row_indices = np.asarray(row_indices, dtype=np.int64)
        col_indices = np.asarray(col_indices, dtype=np.int64)
        physical_values = np.asarray(matrix_values, dtype=REAL_DTYPE)
        system_size = int(system_size)
        if physical_rhs.shape != (system_size,):
            raise ValueError(f"rhs must have shape ({system_size},), got {physical_rhs.shape}")
        physical_matrix = None
    else:
        physical_matrix = matrix.tocsr()
        system_size = physical_matrix.shape[0]
        row_indices = col_indices = None
        physical_values = None

    _validate_finite_array(physical_rhs, "rhs")
    if initial_guess is not None:
        _validate_finite_array(initial_guess, "initial_guess")
    if using_host_coo:
        _validate_finite_array(physical_values, "matrix_values")
    else:
        _validate_finite_array(np.asarray(physical_matrix.data), "matrix data")

    scale_start = time.time()
    if scale_system:
        if using_host_coo:
            diagonal = _coo_diagonal(row_indices, col_indices, physical_values, system_size)
            diagonal[diagonal == 0.0] = 1.0
            inverse_diagonal = 1.0 / diagonal
            solve_values = physical_values * inverse_diagonal[row_indices]
            solve_rhs = inverse_diagonal * physical_rhs
            solve_matrix = None
        else:
            diagonal = np.asarray(physical_matrix.diagonal(), dtype=REAL_DTYPE)
            diagonal[diagonal == 0.0] = 1.0
            inverse_diagonal = 1.0 / diagonal
            solve_matrix, solve_rhs = diagonal_scale_system(physical_matrix, physical_rhs, copy_matrix=True)
            solve_values = None
    else:
        if using_host_coo:
            solve_values = physical_values
            solve_rhs = physical_rhs
            solve_matrix = None
        else:
            solve_matrix = physical_matrix
            solve_rhs = physical_rhs
            solve_values = None
    scale_elapsed_seconds = time.time() - scale_start

    matrix_start = time.time()
    if prepared_device_matrix is None:
        if using_host_coo:
            matrix_cp = scipy_coo_to_cupy_csr(
                row_indices,
                col_indices,
                solve_values,
                (system_size, system_size),
                dtype=cupy_dtype,
            )
            matrix_message = "built from host COO on device"
        else:
            matrix_cp = scipy_csr_to_cupy(solve_matrix, dtype=cupy_dtype)
            matrix_message = "copied host CSR to device"
    else:
        matrix_cp = prepared_device_matrix
        matrix_message = "reused from device cache"
    if tuple(matrix_cp.shape) != (system_size, system_size):
        raise ValueError(f"device matrix must have shape ({system_size}, {system_size}), got {matrix_cp.shape}")
    if not bool(cupy.all(cupy.isfinite(matrix_cp.data)).get()):
        raise ValueError("device matrix contains non-finite values")
    matrix_elapsed_seconds = time.time() - matrix_start
    matrix_nnz = int(getattr(matrix_cp, "nnz", -1))
    _solver_print(
        verbose,
        2,
        "  Cupyx CSR matrix %s in %.5fs with nnz=%d",
        matrix_message,
        matrix_elapsed_seconds,
        matrix_nnz,
    )

    preconditioner_operator = None
    preconditioner_elapsed_seconds = 0.0
    preconditioner_uses_ilu = False
    if isinstance(preconditioner, str):
        preconditioner_key = str(preconditioner).lower().replace("-", "_")
        if preconditioner_key in {"ilu", "host_ilu", "host_ilu_export", "cupyx_ilu1", "device_ilu1"}:
            if ilu_failure not in {"raise", "none"}:
                raise ValueError("ilu_failure must be 'raise' or 'none'")
            preconditioner_uses_ilu = True
            unit_fill = abs(float(ilu_fill_factor) - 1.0) <= 1e-12
            device_ilu = preconditioner_key in {"cupyx_ilu1", "device_ilu1"} or (preconditioner_key == "ilu" and unit_fill)
            if preconditioner_key in {"cupyx_ilu1", "device_ilu1"} and not unit_fill:
                raise ValueError("cupyx-ilu1 requires ilu_fill_factor=1.0; use host-ilu-export for other ILU strengths")

            if device_ilu:
                _solver_print(
                    verbose,
                    2,
                    "  building device Cupyx ILU(1) preconditioner with drop_tol=%g",
                    ilu_drop_tol,
                )
                preconditioner_start = time.time()
                try:
                    preconditioner_operator = build_cupyx_ilu_preconditioner(
                        matrix_cp,
                        drop_tol=ilu_drop_tol,
                        fill_factor=1.0,
                        permc_spec=ilu_permc_spec,
                    )
                except Exception as exc:
                    preconditioner_elapsed_seconds = time.time() - preconditioner_start
                    if ilu_failure == "none":
                        _solver_print(
                            verbose,
                            1,
                            "  Cupyx ILU(1) preconditioner failed in %.5fs (%s); continuing without preconditioner",
                            preconditioner_elapsed_seconds,
                            exc,
                        )
                        preconditioner_operator = None
                    else:
                        raise RuntimeError(
                            "Cupyx ILU(1) preconditioner failed. Set ilu_failure='none' "
                            "to fall back to an unpreconditioned Cupyx Krylov solve."
                        ) from exc
                else:
                    preconditioner_elapsed_seconds = time.time() - preconditioner_start
                    _solver_print(verbose, 2, "  Cupyx ILU(1) preconditioner built in %.5fs", preconditioner_elapsed_seconds)
            else:
                _solver_print(
                    verbose,
                    2,
                    "  building host ILU and exporting factors to device with drop_tol=%g, fill_factor=%g",
                    ilu_drop_tol,
                    ilu_fill_factor,
                )
                preconditioner_start = time.time()
                try:
                    if using_host_coo:
                        host_matrix = scipy.sparse.csr_array(
                            (solve_values, (row_indices, col_indices)),
                            shape=(system_size, system_size),
                        )
                    else:
                        host_matrix = solve_matrix

                    preconditioner_operator = build_cupyx_exported_host_ilu_preconditioner(
                        host_matrix,
                        drop_tol=ilu_drop_tol,
                        fill_factor=ilu_fill_factor,
                        permc_spec=ilu_permc_spec,
                        dtype=matrix_cp.dtype,
                    )
                except Exception as exc:
                    preconditioner_elapsed_seconds = time.time() - preconditioner_start
                    if ilu_failure == "none":
                        _solver_print(
                            verbose,
                            1,
                            "  Host ILU export preconditioner failed in %.5fs (%s); continuing without preconditioner",
                            preconditioner_elapsed_seconds,
                            exc,
                        )
                        preconditioner_operator = None
                    else:
                        raise RuntimeError(
                            "Host ILU export preconditioner failed. Try a larger fill_factor, "
                            "a smaller drop_tol, or set ilu_failure='none' to fall back "
                            "to an unpreconditioned Cupyx Krylov solve."
                        ) from exc
                else:
                    preconditioner_elapsed_seconds = time.time() - preconditioner_start
                    fill_ratio = getattr(preconditioner_operator, "host_ilu_fill_ratio", None)
                    if fill_ratio is None:
                        _solver_print(verbose, 2, "  Host ILU factors exported to device in %.5fs", preconditioner_elapsed_seconds)
                    else:
                        _solver_print(
                            verbose,
                            2,
                            "  Host ILU factors exported to device in %.5fs with factor_fill=%.2f",
                            preconditioner_elapsed_seconds,
                            fill_ratio,
                        )
        elif preconditioner_key == "jacobi":
            raise ValueError("Cupyx preconditioner currently supports only ILU routes or None")
        elif preconditioner_key == "upwind_block_gs":
            if upwind_block_size is None or upwind_level_widths is None:
                raise ValueError(
                    "Cupyx upwind_block_gs preconditioner requires upwind_block_size "
                    "and upwind_level_widths"
                )
            if using_host_coo:
                host_matrix = scipy.sparse.coo_array(
                    (solve_values, (row_indices, col_indices)),
                    shape=(system_size, system_size),
                ).tocsr()
            else:
                host_matrix = solve_matrix
            _solver_print(
                verbose,
                2,
                "  building Cupyx upwind block-GS preconditioner with block_size=%d",
                int(upwind_block_size),
            )
            preconditioner_start = time.time()
            from hdgfem.linalg.gpu.upwind_block_gs import (
                            build_cupy_upwind_block_gs_preconditioner,
                        )

            preconditioner_operator = build_cupy_upwind_block_gs_preconditioner(
                host_matrix,
                block_size=int(upwind_block_size),
                level_widths=upwind_level_widths,
                diagonal_regularization=upwind_diagonal_regularization,
                dtype=cupy_dtype,
            )
            preconditioner_elapsed_seconds = time.time() - preconditioner_start
            stats = getattr(preconditioner_operator, "stats", None)
            host_stats = None if stats is None else getattr(stats, "host_stats", None)
            if host_stats is not None:
                _solver_print(
                    verbose,
                    2,
                    "  Cupyx upwind block-GS built in %.5fs: levels=%d, max_width=%d, retained=%d, dropped_fraction=%.3f",
                    preconditioner_elapsed_seconds,
                    host_stats.num_levels,
                    host_stats.max_width,
                    host_stats.retained_block_couplings,
                    host_stats.dropped_coupling_fraction,
                )
        else:
            raise ValueError(
                "Cupyx preconditioner must be 'host_ilu_export', 'cupyx_ilu1', 'ilu', 'upwind_block_gs', None, or a Cupyx LinearOperator"
            )
    elif preconditioner is not None:
        preconditioner_operator = preconditioner

    initial_residual_norm = None
    if initial_guess is not None:
        if using_host_coo:
            initial_residual = _coo_residual(row_indices, col_indices, physical_values, initial_guess, physical_rhs, system_size)
            initial_residual_norm = float(np.linalg.norm(initial_residual))
        else:
            initial_residual_norm = compute_residual_norm(physical_matrix, initial_guess, physical_rhs)
        _solver_print(verbose, 2, "  initial residual from supplied guess: %.3e", initial_residual_norm)

    solve_start = time.time()
    solution_cp, info, cupyx_iteration_count = solve_cupyx_csr(
        matrix_cp,
        solve_rhs,
        solver=cupyx_solver,
        preconditioner=preconditioner_operator,
        initial_guess=initial_guess,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        restart=restart,
    )
    solution = (
        np.ascontiguousarray(asnumpy(solution_cp), dtype=REAL_DTYPE)
        if materialize_host_solution
        else None
    )
    solution_is_finite = bool(cupy.all(cupy.isfinite(solution_cp)).item())
    solve_elapsed_seconds = time.time() - solve_start

    if materialize_host_solution and using_host_coo:
        solver_residual = _coo_residual(row_indices, col_indices, solve_values, solution, solve_rhs, system_size)
        physical_residual = _coo_residual(row_indices, col_indices, physical_values, solution, physical_rhs, system_size)
    elif materialize_host_solution:
        solver_residual = solve_matrix @ solution - solve_rhs
        physical_residual = physical_matrix @ solution - physical_rhs
    else:
        solve_rhs_cp = cupy.asarray(solve_rhs, dtype=solution_cp.dtype)
        solver_residual = matrix_cp @ solution_cp - solve_rhs_cp
        if scale_system:
            inverse_diagonal_cp = cupy.asarray(inverse_diagonal, dtype=solution_cp.dtype)
            physical_residual = solver_residual / inverse_diagonal_cp
        else:
            physical_residual = solver_residual

    if materialize_host_solution:
        solver_residual_norm = float(np.linalg.norm(solver_residual))
    else:
        solver_residual_norm = float(cupy.linalg.norm(solver_residual).item())
    solver_rhs_norm, solver_relative_residual_norm, solver_residual_target = residual_diagnostics(
        solver_residual_norm,
        solve_rhs,
        rtol=rtol,
        atol=atol,
    )

    if materialize_host_solution:
        physical_residual_norm = float(np.linalg.norm(physical_residual))
    else:
        physical_residual_norm = float(cupy.linalg.norm(physical_residual).item())
    physical_rhs_norm, physical_relative_residual_norm, physical_residual_target = residual_diagnostics(
        physical_residual_norm,
        physical_rhs,
        rtol=rtol,
        atol=atol,
    )
    if diagnostic_rows is None:
        diagnostic_residual_norm = diagnostic_rhs_norm = diagnostic_relative_residual_norm = diagnostic_residual_target = None
    elif materialize_host_solution:
        diagnostic_residual_norm, diagnostic_rhs_norm, diagnostic_relative_residual_norm, diagnostic_residual_target = (
            restricted_residual_diagnostics(physical_residual, physical_rhs, diagnostic_rows, rtol=rtol, atol=atol)
        )
    else:
        diagnostic_residual_cp = physical_residual[cupy.asarray(diagnostic_rows)]
        diagnostic_residual_norm = float(cupy.linalg.norm(diagnostic_residual_cp).item())
        diagnostic_rhs_norm, diagnostic_relative_residual_norm, diagnostic_residual_target = residual_diagnostics(
            diagnostic_residual_norm,
            physical_rhs[diagnostic_rows],
            rtol=rtol,
            atol=atol,
        )

    total_elapsed_seconds = time.time() - total_start
    _solver_print(
        verbose,
        1,
        "  Cupyx %s finished in %.5fs with info=%s, solver_rel=%.3e",
        cupyx_solver,
        solve_elapsed_seconds,
        info,
        solver_relative_residual_norm,
    )
    _solver_print(
        verbose,
        2,
        "  residuals: solver_abs=%.3e, solver_target=%.3e, physical_abs=%.3e, physical_rel=%.3e",
        solver_residual_norm,
        solver_residual_target,
        physical_residual_norm,
        physical_relative_residual_norm,
    )
    _solver_print(
        verbose,
        2,
        "  timings: scale=%.5fs, cupy_matrix=%.5fs, cupyx_ilu=%.5fs, cupyx=%.5fs, total=%.5fs",
        scale_elapsed_seconds,
        matrix_elapsed_seconds,
        preconditioner_elapsed_seconds,
        solve_elapsed_seconds,
        total_elapsed_seconds,
    )

    result = SolveResult(
        x=solution,
        x_device=solution_cp,
        residual_norm=solver_residual_norm,
        info=info,
        preconditioner=preconditioner_operator,
        total_elapsed_seconds=total_elapsed_seconds,
        scale_elapsed_seconds=scale_elapsed_seconds,
        preconditioner_elapsed_seconds=preconditioner_elapsed_seconds,
        solve_elapsed_seconds=solve_elapsed_seconds,
        iteration_count=cupyx_iteration_count,
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
        preconditioner_apply_count=getattr(preconditioner_operator, "apply_count", None),
        preconditioner_apply_seconds=getattr(preconditioner_operator, "apply_seconds", None),
        preconditioner_local_solve_seconds=getattr(preconditioner_operator, "local_solve_seconds", None),
        preconditioner_reduce_seconds=getattr(preconditioner_operator, "reduce_seconds", None),
        preconditioner_copy_seconds=getattr(preconditioner_operator, "copy_seconds", None),
        cupyx_solver=str(cupyx_solver),
        ilu_permc_spec=ilu_permc_spec if preconditioner_uses_ilu else None,
    )
    return finalize_solve_result(
        result,
        backend=f"cupyx-{normalized_cupyx_solver}",
        backend_info=int(info),
        backend_success=int(info) == 0,
        solution_is_finite=solution_is_finite,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )
