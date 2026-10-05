"""hybridge.diagnostics.solver."""

from __future__ import annotations

import numpy as np
from hybridge.runtime.precision import REAL_DTYPE


def relative_drift(value: float, baseline: float) -> float:
    """Return ``(value - baseline) / abs(baseline)`` with safe zero scaling."""
    scale = max(abs(float(baseline)), np.finfo(REAL_DTYPE).tiny)
    return (float(value) - float(baseline)) / scale


def result_transfer_time(result) -> float:
    """Sum timing details associated with host/device data movement."""
    details = getattr(getattr(result, "timings", None), "details", None) or {}
    return sum(
        float(value)
        for key, value in details.items()
        if "host" in str(key) or "to_device" in str(key) or "materialization" in str(key)
    )


def solver_diagnostics_snapshot(result):
    """Keep common HDG solve metrics without owning fields, matrices or factors.

    Rejected time stages still contribute iterations and timings. Their device
    systems can be released once the caller has copied any warm-start traces.
    The original result and its cached preconditioner are never mutated.
    """
    from copy import copy
    from types import SimpleNamespace

    global_solve = getattr(result, "global_solve_result", None)
    if global_solve is not None:
        global_solve = copy(global_solve)
        global_solve.x = global_solve.x_device = global_solve.preconditioner = None
    return SimpleNamespace(
        timings=result.timings, global_solve_result=global_solve,
        assembly_backend=result.assembly_backend, boundary_mode=result.boundary_mode,
        ordering_result=getattr(result, "ordering_result", None),
    )


def solver_result_metrics(prefix: str, result) -> dict[str, object]:
    """Flatten common HDG result, timing, linear-solve, and ordering metrics."""
    timings = result.timings
    global_solve = result.global_solve_result
    row: dict[str, object] = {
        f"{prefix}_assembly_backend": result.assembly_backend,
        f"{prefix}_boundary_mode": result.boundary_mode,
        f"{prefix}_time_total": timings.total,
        f"{prefix}_time_assembly": timings.assembly,
        f"{prefix}_time_solve": timings.solve,
        f"{prefix}_time_reconstruction": timings.reconstruction,
        f"{prefix}_host_device_transfer_time": result_transfer_time(result),
    }
    for attribute in ("trace_ordering", "postprocessing"):
        if hasattr(timings, attribute):
            row[f"{prefix}_time_{attribute}"] = getattr(timings, attribute)
    timing_details = getattr(timings, "details", None) or {}
    rhs_only = bool(timing_details.get("raw.assembly.rhs_only", 0.0))
    operator_reused = bool(timing_details.get("raw.assembly.operator_reused", 0.0))
    row[f"{prefix}_time_rhs_assembly"] = timings.assembly if rhs_only else 0.0
    row[f"{prefix}_time_operator_assembly"] = (
        0.0 if operator_reused else timings.assembly
    )
    for key, value in timing_details.items():
        if isinstance(value, (int, float)):
            safe_key = "".join(ch if ch.isalnum() else "_" for ch in str(key)).strip("_")
            row[f"{prefix}_detail_{safe_key}"] = float(value)
    if global_solve is not None:
        attempts = getattr(global_solve, "amgx_attempts", None)
        if attempts is not None:
            row[f"{prefix}_amgx_attempt_count"] = int(
                getattr(global_solve, "amgx_attempt_count", len(attempts))
            )
            row[f"{prefix}_amgx_attempts"] = list(attempts)
        for key, attribute in {
            "retry_seed_label": "amgx_retry_seed_label",
            "retry_seed_physical_residual": "amgx_retry_seed_physical_residual",
            "retry_seed_physical_rhs_norm": "amgx_retry_seed_physical_rhs_norm",
            "retry_seed_physical_rel_residual": (
                "amgx_retry_seed_physical_relative_residual"
            ),
            "retry_seed_physical_target": "amgx_retry_seed_physical_target",
        }.items():
            if hasattr(global_solve, attribute):
                row[f"{prefix}_{key}"] = getattr(global_solve, attribute)
        attributes = {
            "solver_residual": "solver_residual_norm",
            "solver_rhs_norm": "solver_rhs_norm",
            "solver_residual_target": "solver_residual_target",
            "solver_rel_residual": "solver_relative_residual_norm",
            "physical_residual": "physical_residual_norm",
            "physical_rhs_norm": "physical_rhs_norm",
            "physical_residual_target": "physical_residual_target",
            "physical_rel_residual": "physical_relative_residual_norm",
            "diagnostic_residual": "diagnostic_residual_norm",
            "diagnostic_residual_target": "diagnostic_residual_target",
            "diagnostic_rel_residual": "diagnostic_relative_residual_norm",
            "preconditioner_time": "preconditioner_elapsed_seconds",
            "krylov_time": "solve_elapsed_seconds",
            "matrix_csr_time": "matrix_assembly_elapsed_seconds",
            "solver_global_time": "global_elapsed_seconds",
            "scale_time": "scale_elapsed_seconds",
            "initial_residual_time": "initial_residual_elapsed_seconds",
            "callback_time": "callback_elapsed_seconds",
            "final_residual_time": "residual_diagnostics_elapsed_seconds",
            "preconditioner_apply_count": "preconditioner_apply_count",
            "permutation_time": "permutation_elapsed_seconds",
            "permutation_size": "permutation_size",
            "ilu_permc_spec": "ilu_permc_spec",
            "preconditioner_apply_time": "preconditioner_apply_seconds",
            "preconditioner_factor_nnz": "preconditioner_factor_nnz",
        }
        row[f"{prefix}_solver_iterations"] = (
            -1 if global_solve.iteration_count is None else global_solve.iteration_count
        )
        row.update({f"{prefix}_{key}": getattr(global_solve, attribute) for key, attribute in attributes.items()})
    ordering = getattr(result, "ordering_result", None)
    if ordering is not None:
        diagnostics = ordering.diagnostics
        widths = diagnostics.level_widths
        ordering_values = {
            "nodes": diagnostics.num_nodes,
            "directed_edges": diagnostics.num_directed_edges,
            "components": diagnostics.num_components,
            "largest_scc": diagnostics.largest_component_size,
            "cyclic_components": diagnostics.cyclic_components,
            "cyclic_nodes": diagnostics.cyclic_nodes,
            "levels": widths.num_levels,
            "max_level_width": widths.max_width,
            "median_level_width": widths.median_width,
            "mean_level_width": widths.mean_width,
            "top10_width_fraction": widths.top10_width_fraction,
        }
        row.update({f"{prefix}_ordering_{key}": value for key, value in ordering_values.items()})
    return row
