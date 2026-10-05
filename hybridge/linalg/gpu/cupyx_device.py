"""Direct device-resident Cupyx solve helpers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _device_residual_stats(cupy, residual, rhs, *, rtol: float, atol: float):
    """Return absolute, relative, and target residual diagnostics from device arrays."""
    residual_norm = float(cupy.linalg.norm(residual).item())
    rhs_norm = float(cupy.linalg.norm(rhs).item())
    relative = residual_norm / rhs_norm if rhs_norm > 0.0 else residual_norm
    target = max(float(rtol) * rhs_norm, float(atol))
    return residual_norm, rhs_norm, relative, target


def solve_cupyx_device_coo(
        rows,
        cols,
        data,
        rhs,
        system_size: int,
        *,
        cupyx_solver: str = "bicgstab",
        preconditioner: Any = None,
        initial_guess=None,
        rtol: float = 1.0e-13,
        atol: float = 0.0,
        maxiter: int | None = None,
        restart: int | None = None,
        scale_system: bool = True,
        ilu_drop_tol: float = 1.0e-10,
        ilu_fill_factor: float = 1.0,
        ilu_failure: str = "raise",
        ilu_permc_spec: str | None = None,
        permutation=None,
        diagnostic_rows=None,
        diagnostic_label: str | None = None,
        materialize_host_solution: bool = False,
        raise_on_nonconvergence: bool = False,
):
    """Solve a device COO system without materializing matrix, RHS, or trace."""
    from hybridge.linalg.gpu.cupyx import build_cupyx_ilu_preconditioner, solve_cupyx_csr
    from hybridge.linalg.gpu.sparse import (
            diagonal_scale_cupy_csr_rows_in_place,
            scipy_coo_to_cupy_csr,
        )
    from hybridge.runtime.optional import require_cupy
    from hybridge.linalg.results import SolveResult, finalize_solve_result

    cupy = require_cupy()
    total_start = time.perf_counter()
    physical_matrix = scipy_coo_to_cupy_csr(
        rows,
        cols,
        data,
        (int(system_size), int(system_size)),
        dtype=cupy.float64,
    )
    physical_rhs = cupy.asarray(rhs, dtype=cupy.float64)
    if physical_rhs.shape != (int(system_size),):
        raise ValueError(f"rhs must have shape ({system_size},); got {physical_rhs.shape}")

    permutation_host = None
    diagnostic_rows_solve = diagnostic_rows
    if permutation is not None:
        permutation_host = np.asarray(permutation, dtype=np.int64)
        if permutation_host.shape != (int(system_size),):
            raise ValueError(f"permutation must have shape ({system_size},); got {permutation_host.shape}")
        permutation_cp = cupy.asarray(permutation_host)
        physical_matrix = physical_matrix[permutation_cp][:, permutation_cp].tocsr()
        physical_rhs = physical_rhs[permutation_cp]
        if diagnostic_rows is not None:
            rows_host = np.asarray(diagnostic_rows)
            if rows_host.dtype == bool:
                diagnostic_rows_solve = rows_host[permutation_host]
            else:
                inverse = np.empty_like(permutation_host)
                inverse[permutation_host] = np.arange(permutation_host.size)
                diagnostic_rows_solve = inverse[rows_host]

    solve_matrix = physical_matrix.copy() if scale_system else physical_matrix
    solve_rhs = physical_rhs.copy() if scale_system else physical_rhs
    scale_start = time.perf_counter()
    if scale_system:
        diagonal_scale_cupy_csr_rows_in_place(solve_matrix, solve_rhs)
        cupy.cuda.get_current_stream().synchronize()
    scale_elapsed = time.perf_counter() - scale_start

    preconditioner_operator = preconditioner
    preconditioner_elapsed = 0.0
    preconditioner_uses_ilu = False
    if isinstance(preconditioner, str):
        key = preconditioner.lower().replace("-", "_")
        device_ilu = key in {"cupyx_ilu1", "device_ilu1"} or (
            key == "ilu" and abs(float(ilu_fill_factor) - 1.0) <= 1.0e-12
        )
        if not device_ilu:
            raise ValueError(
                "direct device Cupyx handoff supports None, a device operator, or device ILU(1); "
                "host ILU/export requires host trace-system materialization"
            )
        preconditioner_uses_ilu = True
        preconditioner_start = time.perf_counter()
        try:
            preconditioner_operator = build_cupyx_ilu_preconditioner(
                solve_matrix,
                drop_tol=ilu_drop_tol,
                fill_factor=1.0,
                permc_spec=ilu_permc_spec,
            )
        except Exception:
            if ilu_failure != "none":
                raise
            preconditioner_operator = None
        preconditioner_elapsed = time.perf_counter() - preconditioner_start

    guess_cp = None
    if initial_guess is not None:
        guess_cp = cupy.asarray(initial_guess, dtype=cupy.float64)
        if guess_cp.shape != (int(system_size),):
            raise ValueError(f"initial_guess must have shape ({system_size},); got {guess_cp.shape}")
        if permutation_host is not None:
            guess_cp = guess_cp[cupy.asarray(permutation_host)]

    solve_start = time.perf_counter()
    solution_solve, info, iteration_count = solve_cupyx_csr(
        solve_matrix,
        solve_rhs,
        solver=cupyx_solver,
        preconditioner=preconditioner_operator,
        initial_guess=guess_cp,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        restart=restart,
    )
    solve_elapsed = time.perf_counter() - solve_start
    solver_residual = solve_matrix @ solution_solve - solve_rhs
    physical_residual = physical_matrix @ solution_solve - physical_rhs
    solver_norm, solver_rhs_norm, solver_relative, solver_target = _device_residual_stats(
        cupy, solver_residual, solve_rhs, rtol=rtol, atol=atol
    )
    physical_norm, physical_rhs_norm, physical_relative, physical_target = _device_residual_stats(
        cupy, physical_residual, physical_rhs, rtol=rtol, atol=atol
    )

    diagnostic_norm = diagnostic_rhs_norm = diagnostic_relative = diagnostic_target = None
    if diagnostic_rows_solve is not None:
        selected = cupy.asarray(diagnostic_rows_solve)
        diagnostic_norm, diagnostic_rhs_norm, diagnostic_relative, diagnostic_target = _device_residual_stats(
            cupy,
            physical_residual[selected],
            physical_rhs[selected],
            rtol=rtol,
            atol=atol,
        )

    solution = solution_solve
    if permutation_host is not None:
        solution = cupy.empty_like(solution_solve)
        solution[cupy.asarray(permutation_host)] = solution_solve
    solution_is_finite = bool(cupy.all(cupy.isfinite(solution)).item())
    solution_host = cupy.asnumpy(solution) if materialize_host_solution else None
    total_elapsed = time.perf_counter() - total_start
    result = SolveResult(
        x=None if solution_host is None else np.ascontiguousarray(solution_host),
        x_device=cupy.ascontiguousarray(solution),
        residual_norm=solver_norm,
        info=int(info),
        preconditioner=preconditioner_operator,
        total_elapsed_seconds=total_elapsed,
        scale_elapsed_seconds=scale_elapsed,
        preconditioner_elapsed_seconds=preconditioner_elapsed,
        solve_elapsed_seconds=solve_elapsed,
        iteration_count=iteration_count,
        rhs_norm=solver_rhs_norm,
        relative_residual_norm=solver_relative,
        residual_target=solver_target,
        solver_residual_norm=solver_norm,
        solver_rhs_norm=solver_rhs_norm,
        solver_relative_residual_norm=solver_relative,
        solver_residual_target=solver_target,
        physical_residual_norm=physical_norm,
        physical_rhs_norm=physical_rhs_norm,
        physical_relative_residual_norm=physical_relative,
        physical_residual_target=physical_target,
        diagnostic_residual_label=diagnostic_label,
        diagnostic_residual_norm=diagnostic_norm,
        diagnostic_rhs_norm=diagnostic_rhs_norm,
        diagnostic_relative_residual_norm=diagnostic_relative,
        diagnostic_residual_target=diagnostic_target,
        rtol=rtol,
        atol=atol,
        preconditioner_apply_count=getattr(preconditioner_operator, "apply_count", None),
        preconditioner_apply_seconds=getattr(preconditioner_operator, "apply_seconds", None),
        cupyx_solver=str(cupyx_solver),
        ilu_permc_spec=ilu_permc_spec if preconditioner_uses_ilu else None,
        permutation_elapsed_seconds=0.0 if permutation_host is not None else None,
        permutation_size=None if permutation_host is None else int(permutation_host.size),
    )
    normalized = str(cupyx_solver).lower().replace("-", "_")
    return finalize_solve_result(
        result,
        backend=f"cupyx-{normalized}",
        backend_info=int(info),
        backend_success=int(info) == 0,
        solution_is_finite=solution_is_finite,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )


__all__ = ["solve_cupyx_device_coo"]
