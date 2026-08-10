#!/usr/bin/env python3
"""End-to-end tight-tolerance validation of GPU face-dense GMRES.

The driver exercises every manufactured problem registered in
``scripts.diff_rea_cases`` and sweeps the production matvec, orthogonalization,
base preconditioner, ASM/Block-Jacobi application, and polynomial degree.
It records both mathematical validation metrics and setup/solve timings, then
ranks only configurations that pass every measured repetition.

The default matrix is deliberately thorough.  Use ``--dry-run`` to inspect its
size, or select a smaller subset while bringing up a new GPU.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any, Iterable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diff_rea_cases import CASE_BY_KEY, case_definition_by_key


PUBLIC_PRECONDITIONERS = (
    "none",
    "polynomial",
    "bj",
    "bj-polynomial",
    "asm",
    "asm-polynomial",
)

PRECONDITIONER_MAP = {
    "none": "none",
    "polynomial": "poly",
    "bj": "block_jacobi",
    "bj-polynomial": "block_jacobi_poly",
    "asm": "asm",
    "asm-polynomial": "asm_poly",
}

POLYNOMIAL_PRECONDITIONERS = {
    "polynomial",
    "bj-polynomial",
    "asm-polynomial",
}


@dataclass(frozen=True)
class SolverConfiguration:
    """One meaningful point in the solver/kernel parameter space."""

    preconditioner: str
    operator: str
    orthogonalization: str
    polynomial_degree: int | None = None
    asm_application: str | None = None
    block_jacobi_application: str | None = None

    @property
    def internal_preconditioner(self) -> str:
        return PRECONDITIONER_MAP[self.preconditioner]

    @property
    def label(self) -> str:
        fields = [self.preconditioner, f"op={self.operator}", f"orth={self.orthogonalization}"]
        if self.polynomial_degree is not None:
            fields.append(f"d={self.polynomial_degree}")
        if self.asm_application is not None:
            fields.append(f"asm={self.asm_application}")
        if self.block_jacobi_application is not None:
            fields.append(f"bj={self.block_jacobi_application}")
        return ",".join(fields)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("all", *sorted(CASE_BY_KEY)),
        default=["all"],
    )
    parser.add_argument("--orders", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument(
        "--preconditioners",
        nargs="+",
        choices=PUBLIC_PRECONDITIONERS,
        default=list(PUBLIC_PRECONDITIONERS),
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        choices=("auto", "raw", "raw_fused", "matmul"),
        default=["auto"],
    )
    parser.add_argument(
        "--asm-applications",
        nargs="+",
        choices=("auto", "raw", "fused", "matmul"),
        default=["auto"],
    )
    parser.add_argument(
        "--bj-applications",
        nargs="+",
        choices=("auto", "raw", "matmul"),
        default=["auto"],
    )
    parser.add_argument(
        "--orthogonalizations",
        nargs="+",
        choices=("mgs", "mgs2", "cgs", "cgs2"),
        default=["cgs"],
    )
    parser.add_argument(
        "--polynomial-degrees",
        nargs="+",
        type=int,
        default=[4, 8, 12, 18, 24, 32],
    )
    parser.add_argument("--structured-nx", type=int, default=32)
    parser.add_argument("--structured-ny", type=int, default=None)
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=0.30,
        help="Gmsh target size for the disk and L-shaped cases.",
    )
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--basis", default="dub_orth")
    parser.add_argument("--quadrature-extra", type=int, default=1)
    parser.add_argument("--tensor-m", type=int, default=1)
    parser.add_argument("--tensor-n", type=int, default=1)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument(
        "--boundary-mode",
        choices=("eliminate", "penalty"),
        default="eliminate",
    )
    parser.add_argument("--boundary-penalty", type=float, default=1.0e20)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument(
        "--dtype", choices=("float64",), default="float64",
        help="Tolerances below 1e-11 require the float64 production path.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--local-solver",
        choices=("cpu_inverse", "gpu_inverse", "cublas_inverse", "gpu_solve"),
        default="cublas_inverse",
    )
    parser.add_argument(
        "--polynomial-setup-orthogonalization",
        choices=("mgs", "mgs2", "cgs", "cgs2"),
        default="cgs2",
    )
    parser.add_argument("--polynomial-seed", type=int, default=1729)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--warmup-solves",
        type=int,
        default=0,
        help="Discarded full solves per configuration; useful for timing/JIT warmup.",
    )
    parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--autotune-cache", type=Path, default=None)
    parser.add_argument("--autotune-warmup", type=int, default=10)
    parser.add_argument("--autotune-repeats", type=int, default=50)
    parser.add_argument("--force-autotune", action="store_true")
    parser.add_argument(
        "--measure-cold-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force one post-warmup retune per auto configuration so the first-"
            "solve cost is measured separately from cached timings."
        ),
    )
    parser.add_argument(
        "--reference-solver",
        choices=("direct", "first-passing", "none"),
        default="direct",
        help=(
            "Independent SciPy direct trace reference for numerical validation; "
            "performance-only sweeps may use first-passing."
        ),
    )
    parser.add_argument(
        "--monitor-orthogonality",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--residual-factor", type=float, default=5.0)
    parser.add_argument("--physical-residual-factor", type=float, default=10.0)
    parser.add_argument("--trace-agreement-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--release-memory-between-configs", action="store_true")
    parser.add_argument("--max-configurations", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--require-coverage",
        action="store_true",
        help=(
            "Return a nonzero status for execution exceptions or when any "
            "case/order has no configuration whose measured repetitions all pass. "
            "Individual failed candidates remain report data."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a nonzero status if any requested candidate fails.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/diff_rea_gpu_hard_cases"),
    )
    args = parser.parse_args()

    if "all" in args.cases:
        if len(args.cases) != 1:
            parser.error("--cases all cannot be combined with explicit case names")
        args.cases = list(CASE_BY_KEY)
    if any(order < 1 for order in args.orders):
        parser.error("orders must be positive")
    if any(degree < 1 for degree in args.polynomial_degrees):
        parser.error("polynomial degrees must be positive")
    if args.structured_nx < 1 or (args.structured_ny is not None and args.structured_ny < 1):
        parser.error("structured mesh dimensions must be positive")
    if args.mesh_size <= 0.0:
        parser.error("--mesh-size must be positive")
    if args.quadrature_extra < 1:
        parser.error("--quadrature-extra must be at least one")
    if args.rtol <= 0.0 or args.atol < 0.0:
        parser.error("invalid solver tolerances")
    if args.restart < 1 or args.max_iterations < 1:
        parser.error("restart and max iterations must be positive")
    if args.repeats < 1 or args.warmup_solves < 0:
        parser.error("repeats must be positive and warmup-solves non-negative")
    if args.max_configurations is not None and args.max_configurations < 1:
        parser.error("--max-configurations must be positive")
    return args


def build_solver_configurations(args: argparse.Namespace) -> list[SolverConfiguration]:
    """Build only semantically meaningful combinations from the CLI grid."""

    configurations: list[SolverConfiguration] = []
    for preconditioner in args.preconditioners:
        degrees: Iterable[int | None] = (
            args.polynomial_degrees
            if preconditioner in POLYNOMIAL_PRECONDITIONERS
            else (None,)
        )
        asm_applications: Iterable[str | None] = (
            args.asm_applications if preconditioner.startswith("asm") else (None,)
        )
        bj_applications: Iterable[str | None] = (
            args.bj_applications if preconditioner.startswith("bj") else (None,)
        )
        for operator, orthogonalization, degree, asm_application, bj_application in itertools.product(
            args.operators,
            args.orthogonalizations,
            degrees,
            asm_applications,
            bj_applications,
        ):
            configurations.append(
                SolverConfiguration(
                    preconditioner=preconditioner,
                    operator=operator,
                    orthogonalization=orthogonalization,
                    polynomial_degree=degree,
                    asm_application=asm_application,
                    block_jacobi_application=bj_application,
                )
            )
    if args.max_configurations is not None:
        configurations = configurations[: args.max_configurations]
    return configurations


def _build_mesh(case_key: str, args: argparse.Namespace):
    from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_lshape_mesh, rectangle_mesh

    ny = args.structured_nx if args.structured_ny is None else args.structured_ny
    if case_key == "trigonometric-poisson":
        return gmsh_disc_mesh(
            args.mesh_size,
            center=(0.0, 0.0),
            radius=5.0,
            verbosity=args.gmsh_verbosity,
        )
    if case_key == "lshape-singular":
        return gmsh_lshape_mesh(
            args.mesh_size,
            corner_mesh_size=args.mesh_size / 10.0,
            corner_refine_radius=0.1,
            verbosity=args.gmsh_verbosity,
        )
    return rectangle_mesh(
        args.structured_nx,
        ny,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
    )


def _case_params(case_key: str, args: argparse.Namespace) -> dict[str, int]:
    if case_key == "tensor-sine":
        return {"m": args.tensor_m, "n": args.tensor_n}
    return {}


def _device_metadata(cp: Any, device_id: int) -> dict[str, Any]:
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return {
        "name": str(name),
        "device_id": int(device_id),
        "compute_capability": f"{int(properties['major'])}.{int(properties['minor'])}",
        "total_memory_bytes": int(properties["totalGlobalMem"]),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cupy_version": str(cp.__version__),
    }


def _orthogonality_metrics(gmres_result: Any) -> tuple[float, float, float]:
    records = gmres_result.orthogonality_records
    if not records:
        return math.nan, math.nan, math.nan
    return (
        max(float(item.frobenius_defect) for item in records),
        max(float(item.maximum_offdiagonal) for item in records),
        max(float(item.maximum_diagonal_error) for item in records),
    )


def _relative_difference(actual: np.ndarray, reference: np.ndarray) -> float:
    difference = np.linalg.norm(np.asarray(actual) - np.asarray(reference))
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).eps)
    return float(difference / denominator)


def _finite_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _autotune_candidates(diagnostics: Any, kind: str) -> str:
    result = diagnostics.autotune_result
    if result is None:
        return ""
    rows = getattr(result, f"{kind}_candidates")
    return ";".join(f"{item.name}:{item.median_ms:.9g}" for item in rows)


@dataclass(frozen=True)
class TraceReference:
    source: str
    trace: np.ndarray | None
    total_ms: float = math.nan
    solve_ms: float = math.nan
    solver_relative_residual: float = math.nan
    physical_relative_residual: float = math.nan
    primal_l2_error: float = math.nan
    flux_l2_error: float = math.nan


def _build_trace_reference(
    *,
    space: Any,
    problem: Any,
    args: argparse.Namespace,
) -> TraceReference:
    """Build an independent trace reference once per case/order."""

    if args.reference_solver != "direct":
        return TraceReference(source=args.reference_solver, trace=None)

    from hdgfem.solvers.diff_rea import solve_diffusion_reaction_hdg
    from scripts.run_diff_rea_cases import _vector_l2_error

    diffusion, reaction, source, exact = problem
    started = time.perf_counter()
    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=args.tau,
        solver="direct",
        preconditioner=None,
        solver_rtol=min(args.rtol, 1.0e-13),
        solver_atol=args.atol,
        maxiter=None,
        scale_system=False,
        local_solver_backend="numpy",
        assembly_backend="numpy",
        boundary_penalty=args.boundary_penalty,
        boundary_mode=args.boundary_mode,
        hdg_postprocess="none",
        verbose=False,
    )
    total_ms = 1.0e3 * (time.perf_counter() - started)
    solve = result.global_solve_result
    reference = TraceReference(
        source="scipy-direct",
        trace=np.array(result.trace, copy=True),
        total_ms=total_ms,
        solve_ms=1.0e3 * float(result.timings.solve),
        solver_relative_residual=_finite_or_nan(
            solve.solver_relative_residual_norm
        ),
        physical_relative_residual=_finite_or_nan(
            solve.physical_relative_residual_norm
        ),
        primal_l2_error=float(result.field.l2_error(exact)),
        flux_l2_error=float(_vector_l2_error(result.flux, problem.exact_flux)),
    )
    reference_limit = max(100.0 * args.rtol, 1.0e-11)
    if (
        not np.all(np.isfinite(reference.trace))
        or not math.isfinite(reference.solver_relative_residual)
        or not math.isfinite(reference.physical_relative_residual)
        or reference.solver_relative_residual > reference_limit
        or reference.physical_relative_residual > reference_limit
    ):
        raise RuntimeError(
            "independent direct reference failed residual/finite validation: "
            f"solver={reference.solver_relative_residual:.3e}, "
            f"physical={reference.physical_relative_residual:.3e}, "
            f"limit={reference_limit:.3e}"
        )
    return reference


def _configuration_uses_autotuning(configuration: SolverConfiguration) -> bool:
    return bool(
        configuration.operator == "auto"
        or configuration.asm_application == "auto"
        or configuration.block_jacobi_application == "auto"
    )


def _solve_once(
    *,
    cp: Any,
    case_key: str,
    order: int,
    mesh: Any,
    space: Any,
    problem: Any,
    configuration: SolverConfiguration,
    args: argparse.Namespace,
    repeat: int,
    reference: TraceReference,
    force_autotune: bool = False,
) -> tuple[dict[str, Any], np.ndarray | None]:
    from hdgfem.solvers.diff_rea import solve_diffusion_reaction_hdg
    from scripts.run_diff_rea_cases import _vector_l2_error

    diffusion, reaction, source, exact = problem
    cache_path = args.autotune_cache
    if cache_path is None:
        cache_path = args.output_prefix.parent / "diff_rea_gpu_autotune_cache.json"
    gpu_options = {
        "device_id": args.device,
        "dtype": args.dtype,
        "operator": configuration.operator,
        "preconditioner": configuration.internal_preconditioner,
        "local_solver": args.local_solver,
        "block_jacobi_application": configuration.block_jacobi_application or "raw",
        "asm_application": configuration.asm_application or "auto",
        "polynomial_degree": configuration.polynomial_degree or args.polynomial_degrees[0],
        "polynomial_seed": args.polynomial_seed,
        "polynomial_setup_orthogonalization": args.polynomial_setup_orthogonalization,
        "restart": args.restart,
        "orthogonalization": configuration.orthogonalization,
        "raise_on_failure": False,
        "autotune": args.autotune,
        "autotune_cache_file": str(cache_path),
        "autotune_use_cache": True,
        "autotune_force": bool(force_autotune),
        "autotune_warmup": args.autotune_warmup,
        "autotune_repeats": args.autotune_repeats,
        "monitor_orthogonality": args.monitor_orthogonality,
    }
    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=args.tau,
        solver="gpu_face_dense",
        preconditioner=None,
        solver_rtol=args.rtol,
        solver_atol=args.atol,
        maxiter=args.max_iterations,
        local_solver_backend="numpy",
        boundary_penalty=args.boundary_penalty,
        boundary_mode=args.boundary_mode,
        hdg_postprocess="none",
        gpu_options=gpu_options,
        verbose=False,
    )
    diagnostics = result.gpu_diagnostics
    gmres = diagnostics.gmres_result
    solve = result.global_solve_result
    needs_reference = (
        reference.source == "first-passing" and reference.trace is None
    )
    if reference.trace is None:
        trace_difference = math.nan if reference.source == "none" else 0.0
    else:
        trace_difference = _relative_difference(result.trace, reference.trace)

    primal_l2 = float(result.field.l2_error(exact))
    flux_l2 = float(_vector_l2_error(result.flux, problem.exact_flux))
    orth_fro, orth_offdiag, orth_diag = _orthogonality_metrics(gmres)
    solver_relative = float(gmres.relative_residual)
    physical_relative = float(solve.physical_relative_residual_norm)
    finite = bool(
        np.isfinite(solver_relative)
        and np.isfinite(physical_relative)
        and np.isfinite(primal_l2)
        and np.isfinite(flux_l2)
        and np.all(np.isfinite(result.trace))
    )
    residual_pass = solver_relative <= args.residual_factor * args.rtol
    physical_pass = physical_relative <= args.physical_residual_factor * args.rtol
    trace_pass = bool(
        reference.source == "none"
        or trace_difference <= args.trace_agreement_tolerance
    )
    passed = bool(gmres.converged and finite and residual_pass and physical_pass and trace_pass)
    discovered_reference = (
        np.array(result.trace, copy=True) if needs_reference and passed else None
    )

    setup_ms = 1.0e3 * (
        diagnostics.operator_setup_seconds + diagnostics.preconditioner_setup_seconds
    )
    cold_time_ms = setup_ms + 1.0e3 * (
        diagnostics.autotune_seconds + diagnostics.solve_seconds
    )
    hot_time_ms = setup_ms + 1.0e3 * diagnostics.solve_seconds
    row = {
        "case": case_key,
        "order": order,
        "repeat": repeat,
        "configuration": configuration.label,
        **asdict(configuration),
        "internal_preconditioner": configuration.internal_preconditioner,
        "device": diagnostics.device_name,
        "dtype": diagnostics.dtype,
        "boundary_mode": args.boundary_mode,
        "tau": args.tau,
        "num_elements": int(mesh.num_tri),
        "num_global_faces": int(mesh.num_edg),
        "trace_dofs": int(result.trace.size),
        "system_dofs": int(solve.x.size),
        "rtol": args.rtol,
        "atol": args.atol,
        "restart": args.restart,
        "max_iterations": args.max_iterations,
        "status": str(gmres.status),
        "termination_reason": str(gmres.termination_reason),
        "converged": bool(gmres.converged),
        "passed": passed,
        "finite": finite,
        "residual_pass": residual_pass,
        "physical_residual_pass": physical_pass,
        "trace_agreement_pass": trace_pass,
        "trace_reference_source": reference.source,
        "reference_total_ms": reference.total_ms,
        "reference_solve_ms": reference.solve_ms,
        "reference_solver_relative_residual": reference.solver_relative_residual,
        "reference_physical_relative_residual": reference.physical_relative_residual,
        "reference_primal_l2_error": reference.primal_l2_error,
        "reference_flux_l2_error": reference.flux_l2_error,
        "solver_relative_residual": solver_relative,
        "physical_relative_residual": physical_relative,
        "trace_relative_difference": trace_difference,
        "primal_l2_error": primal_l2,
        "flux_l2_error": flux_l2,
        "iterations": int(gmres.iterations),
        "restart_cycles": int(gmres.restart_cycles),
        "fallback_count": int(gmres.fallback_count),
        "matvec_count": int(gmres.matvec_count),
        "preconditioner_count": int(gmres.preconditioner_count),
        "dot_count": int(gmres.dot_count),
        "maximum_orthogonality_frobenius_defect": orth_fro,
        "maximum_orthogonality_offdiagonal": orth_offdiag,
        "maximum_orthogonality_diagonal_error": orth_diag,
        "resolved_operator": diagnostics.operator,
        "resolved_block_jacobi_application": (
            diagnostics.block_jacobi_application
        ),
        "resolved_asm_application": diagnostics.asm_application,
        "autotune_cache_hit": diagnostics.autotune_cache_hit,
        "autotune_forced": bool(force_autotune),
        "timing_sample": (
            "retuned-cold"
            if force_autotune and _configuration_uses_autotuning(configuration)
            else "cached-or-explicit"
        ),
        "autotune_operator_candidates_ms": _autotune_candidates(diagnostics, "operator"),
        "autotune_block_jacobi_candidates_ms": _autotune_candidates(
            diagnostics, "block_jacobi"
        ),
        "autotune_asm_candidates_ms": _autotune_candidates(diagnostics, "asm"),
        "face_assembly_ms": 1.0e3 * diagnostics.face_assembly_seconds,
        "autotune_ms": 1.0e3 * diagnostics.autotune_seconds,
        "operator_setup_ms": 1.0e3 * diagnostics.operator_setup_seconds,
        "preconditioner_setup_ms": 1.0e3 * diagnostics.preconditioner_setup_seconds,
        "solve_ms": 1.0e3 * diagnostics.solve_seconds,
        "transfer_to_host_ms": 1.0e3 * diagnostics.transfer_to_host_seconds,
        "hot_time_to_solution_ms": hot_time_ms,
        "cold_time_to_solution_ms": cold_time_ms,
        "end_to_end_ms": 1.0e3 * result.timings.total,
        "solve_ms_per_iteration": (
            math.nan if gmres.iterations == 0 else 1.0e3 * diagnostics.solve_seconds / gmres.iterations
        ),
        "workspace_device_bytes": int(diagnostics.workspace_device_bytes),
        "operator_workspace_bytes": int(diagnostics.operator_workspace_bytes),
        "preconditioner_workspace_bytes": int(diagnostics.preconditioner_workspace_bytes),
        "memory_pool_used_bytes": int(cp.get_default_memory_pool().used_bytes()),
        "exception_type": "",
        "exception_message": "",
    }
    return row, discovered_reference


def _error_row(
    *,
    case_key: str,
    order: int,
    configuration: SolverConfiguration,
    repeat: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "case": case_key,
        "order": order,
        "repeat": repeat,
        "configuration": configuration.label,
        **asdict(configuration),
        "internal_preconditioner": configuration.internal_preconditioner,
        "converged": False,
        "passed": False,
        "status": "exception",
        "exception_type": type(error).__name__,
        "exception_message": str(error),
    }


GROUP_FIELDS = (
    "case",
    "order",
    "configuration",
    "preconditioner",
    "operator",
    "orthogonalization",
    "polynomial_degree",
    "asm_application",
    "block_jacobi_application",
)


def _median(rows: list[dict[str, Any]], field: str) -> float:
    values = [_finite_or_nan(row.get(field)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return math.nan if not values else float(statistics.median(values))


def _finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [
        value
        for value in (_finite_or_nan(row.get(field)) for row in rows)
        if math.isfinite(value)
    ]


def _percentile(values: list[float], percentile: float) -> float:
    return math.nan if not values else float(np.percentile(values, percentile))


def _unique_text(rows: list[dict[str, Any]], field: str) -> str:
    values = sorted(
        {
            str(row[field])
            for row in rows
            if row.get(field) not in {None, ""}
        }
    )
    return ";".join(values)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in GROUP_FIELDS)
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        summary = dict(zip(GROUP_FIELDS, key, strict=True))
        solve_values = _finite_values(group, "solve_ms")
        cold_values = _finite_values(group, "cold_time_to_solution_ms")
        retuned_rows = [
            row for row in group if row.get("timing_sample") == "retuned-cold"
        ]
        retuned_cold_values = _finite_values(
            retuned_rows, "cold_time_to_solution_ms"
        )
        retuned_end_to_end_values = _finite_values(
            retuned_rows, "end_to_end_ms"
        )
        solver_residuals = _finite_values(group, "solver_relative_residual")
        physical_residuals = _finite_values(group, "physical_relative_residual")
        trace_differences = _finite_values(group, "trace_relative_difference")
        summary.update(
            {
                "repeats": len(group),
                "passed_repeats": sum(bool(row.get("passed")) for row in group),
                "all_passed": all(bool(row.get("passed")) for row in group),
                "median_iterations": _median(group, "iterations"),
                "median_solve_ms": _median(group, "solve_ms"),
                "minimum_solve_ms": min(solve_values, default=math.nan),
                "mean_solve_ms": (
                    math.nan if not solve_values else float(statistics.mean(solve_values))
                ),
                "std_solve_ms": (
                    math.nan if not solve_values else float(np.std(solve_values))
                ),
                "p90_solve_ms": _percentile(solve_values, 90.0),
                "median_hot_time_to_solution_ms": _median(group, "hot_time_to_solution_ms"),
                "median_cold_time_to_solution_ms": _median(group, "cold_time_to_solution_ms"),
                "representative_cold_time_to_solution_ms": (
                    float(statistics.median(retuned_cold_values))
                    if retuned_cold_values
                    else (float(statistics.median(cold_values)) if cold_values else math.nan)
                ),
                "median_end_to_end_ms": _median(group, "end_to_end_ms"),
                "representative_cold_end_to_end_ms": (
                    float(statistics.median(retuned_end_to_end_values))
                    if retuned_end_to_end_values
                    else _median(group, "end_to_end_ms")
                ),
                "median_autotune_ms": _median(group, "autotune_ms"),
                "retuned_autotune_ms": _median(retuned_rows, "autotune_ms"),
                "median_preconditioner_setup_ms": _median(group, "preconditioner_setup_ms"),
                "median_operator_setup_ms": _median(group, "operator_setup_ms"),
                "median_face_assembly_ms": _median(group, "face_assembly_ms"),
                "median_workspace_device_bytes": _median(group, "workspace_device_bytes"),
                "resolved_operators": _unique_text(group, "resolved_operator"),
                "resolved_block_jacobi_applications": _unique_text(
                    group, "resolved_block_jacobi_application"
                ),
                "resolved_asm_applications": _unique_text(
                    group, "resolved_asm_application"
                ),
                "worst_solver_relative_residual": max(solver_residuals, default=math.nan),
                "worst_physical_relative_residual": max(physical_residuals, default=math.nan),
                "worst_trace_relative_difference": max(trace_differences, default=math.nan),
                "median_primal_l2_error": _median(group, "primal_l2_error"),
                "median_flux_l2_error": _median(group, "flux_l2_error"),
                "reference_primal_l2_error": _median(
                    group, "reference_primal_l2_error"
                ),
                "reference_flux_l2_error": _median(
                    group, "reference_flux_l2_error"
                ),
            }
        )
        summaries.append(summary)

    for case_key, order in sorted({(row["case"], row["order"]) for row in summaries}):
        candidates = [
            row
            for row in summaries
            if row["case"] == case_key and row["order"] == order and row["all_passed"]
        ]
        for metric, rank_field in (
            ("median_solve_ms", "solve_rank"),
            ("median_hot_time_to_solution_ms", "hot_time_rank"),
            ("representative_cold_time_to_solution_ms", "cold_time_rank"),
            ("median_end_to_end_ms", "end_to_end_rank"),
        ):
            ordered = sorted(
                (row for row in candidates if math.isfinite(_finite_or_nan(row.get(metric)))),
                key=lambda row: float(row[metric]),
            )
            for rank, row in enumerate(ordered, start=1):
                row[rank_field] = rank
        for row in summaries:
            if row["case"] == case_key and row["order"] == order:
                row.setdefault("solve_rank", None)
                row.setdefault("hot_time_rank", None)
                row.setdefault("cold_time_rank", None)
                row.setdefault("end_to_end_rank", None)
    return summaries


def _best_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    for case_key, order in sorted({(row["case"], row["order"]) for row in summaries}):
        candidates = [
            row
            for row in summaries
            if row["case"] == case_key and row["order"] == order and row["all_passed"]
        ]
        entry: dict[str, Any] = {"case": case_key, "order": order}
        for metric, label in (
            ("median_solve_ms", "best_solve"),
            ("median_hot_time_to_solution_ms", "best_hot_time"),
            ("representative_cold_time_to_solution_ms", "best_cold_time"),
            ("median_end_to_end_ms", "best_end_to_end"),
        ):
            finite = [row for row in candidates if math.isfinite(_finite_or_nan(row.get(metric)))]
            winner = min(finite, key=lambda row: float(row[metric])) if finite else None
            entry[f"{label}_configuration"] = None if winner is None else winner["configuration"]
            entry[f"{label}_ms"] = math.nan if winner is None else winner[metric]
        best.append(entry)
    return best


def _uncovered_case_orders(
    summaries: list[dict[str, Any]],
) -> list[tuple[Any, Any]]:
    """Return case/order pairs without a fully passing configuration."""

    pairs = sorted({(row["case"], row["order"]) for row in summaries})
    return [
        (case_key, order)
        for case_key, order in pairs
        if not any(
            row["case"] == case_key
            and row["order"] == order
            and bool(row.get("all_passed"))
            for row in summaries
        )
    ]


def _validation_exit_code(
    *,
    strict: bool,
    require_coverage: bool,
    any_candidate_failed: bool,
    uncovered_count: int,
    exception_count: int,
) -> int:
    """Apply the requested candidate-level or coverage-level failure policy."""

    if strict and any_candidate_failed:
        return 1
    if require_coverage and (uncovered_count > 0 or exception_count > 0):
        return 1
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_outputs(
    args: argparse.Namespace,
    *,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    best: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    prefix = args.output_prefix
    raw_csv = prefix.with_name(prefix.name + "_raw.csv")
    summary_csv = prefix.with_name(prefix.name + "_summary.csv")
    json_path = prefix.with_suffix(".json")
    _write_csv(raw_csv, rows)
    _write_csv(summary_csv, summaries)
    payload = {
        "metadata": metadata,
        "arguments": vars(args),
        "rows": rows,
        "summaries": summaries,
        "best_by_case_order": best,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return raw_csv, summary_csv, json_path


def main() -> int:
    args = _parse_args()
    configurations = build_solver_configurations(args)
    total_measured = len(args.cases) * len(args.orders) * len(configurations) * args.repeats
    print("Tight-tolerance diffusion-reaction GPU validation")
    print("=" * 58)
    print(f"Cases/configurations/orders : {len(args.cases)}/{len(configurations)}/{len(args.orders)}")
    print(f"Measured solves             : {total_measured}")
    print(f"Tolerance / dtype           : {args.rtol:.1e} / {args.dtype}")
    print(f"Mesh                        : structured={args.structured_nx}x{args.structured_ny or args.structured_nx}, gmsh h={args.mesh_size:g}")
    print(f"Output prefix               : {args.output_prefix}")
    if args.dry_run:
        for index, configuration in enumerate(configurations, start=1):
            print(f"  {index:3d}. {configuration.label}")
        return 0

    from hdgfem.backends.cupy import require_cupy_device
    from hdgfem.core.space import DGSpace

    cp = require_cupy_device()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "device": _device_metadata(cp, args.device),
    }
    print(f"Device                      : {metadata['device']['name']}")
    print()

    rows: list[dict[str, Any]] = []
    failed = False
    run_index = 0

    def checkpoint() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        summaries_now = _aggregate_rows(rows)
        best_now = _best_rows(summaries_now)
        _write_outputs(
            args,
            rows=rows,
            summaries=summaries_now,
            best=best_now,
            metadata=metadata,
        )
        return summaries_now, best_now

    with cp.cuda.Device(args.device):
        for case_key in args.cases:
            case = case_definition_by_key(case_key)
            problem = case.build(**_case_params(case_key, args))
            print(f"Building mesh for {case_key} ...", flush=True)
            mesh = _build_mesh(case_key, args)
            for order in args.orders:
                quadrature = order + args.quadrature_extra
                space = DGSpace(
                    mesh,
                    order,
                    basis_type=args.basis,
                    volume_quad_1d=quadrature,
                    edge_quad_1d=quadrature,
                )
                print(
                    f"Building {args.reference_solver} reference for "
                    f"case={case_key} p={order} ...",
                    flush=True,
                )
                try:
                    reference = _build_trace_reference(
                        space=space,
                        problem=problem,
                        args=args,
                    )
                except Exception as error:
                    failed = True
                    print(
                        f"  REFERENCE EXCEPTION {type(error).__name__}: {error}",
                        flush=True,
                    )
                    traceback.print_exc()
                    for configuration in configurations:
                        for repeat in range(args.repeats):
                            rows.append(
                                _error_row(
                                    case_key=case_key,
                                    order=order,
                                    configuration=configuration,
                                    repeat=repeat,
                                    error=error,
                                )
                            )
                    checkpoint()
                    if args.fail_fast:
                        raise
                    continue
                print(
                    f"\ncase={case_key} p={order} elements={mesh.num_tri:,} faces={mesh.num_edg:,}",
                    flush=True,
                )
                for config_index, configuration in enumerate(configurations, start=1):
                    if args.release_memory_between_configs:
                        cp.get_default_memory_pool().free_all_blocks()
                    print(
                        f"  [{config_index:3d}/{len(configurations):3d}] {configuration.label}",
                        flush=True,
                    )
                    for warmup in range(args.warmup_solves):
                        try:
                            _solve_once(
                                cp=cp,
                                case_key=case_key,
                                order=order,
                                mesh=mesh,
                                space=space,
                                problem=problem,
                                configuration=configuration,
                                args=args,
                                repeat=-(warmup + 1),
                                reference=reference,
                                force_autotune=False,
                            )
                        except Exception as error:  # keep campaign context in the measured row
                            print(f"    warmup failed: {type(error).__name__}: {error}", flush=True)
                            if args.fail_fast:
                                raise
                    for repeat in range(args.repeats):
                        run_index += 1
                        try:
                            force_autotune = bool(
                                args.force_autotune
                                or (
                                    args.measure_cold_autotune
                                    and repeat == 0
                                    and args.autotune
                                    and _configuration_uses_autotuning(configuration)
                                )
                            )
                            row, discovered_trace = _solve_once(
                                cp=cp,
                                case_key=case_key,
                                order=order,
                                mesh=mesh,
                                space=space,
                                problem=problem,
                                configuration=configuration,
                                args=args,
                                repeat=repeat,
                                reference=reference,
                                force_autotune=force_autotune,
                            )
                            if discovered_trace is not None:
                                reference = TraceReference(
                                    source="first-passing-gpu",
                                    trace=discovered_trace,
                                )
                            rows.append(row)
                            failed = failed or not bool(row["passed"])
                            print(
                                f"    repeat={repeat} pass={int(row['passed'])} "
                                f"it={row['iterations']} rel={row['solver_relative_residual']:.2e} "
                                f"phys={row['physical_relative_residual']:.2e} "
                                f"solve={row['solve_ms']:.3f} ms",
                                flush=True,
                            )
                        except Exception as error:
                            failed = True
                            rows.append(
                                _error_row(
                                    case_key=case_key,
                                    order=order,
                                    configuration=configuration,
                                    repeat=repeat,
                                    error=error,
                                )
                            )
                            print(
                                f"    repeat={repeat} EXCEPTION {type(error).__name__}: {error}",
                                flush=True,
                            )
                            traceback.print_exc()
                            if args.fail_fast:
                                raise

                checkpoint()

    summaries = _aggregate_rows(rows)
    best = _best_rows(summaries)
    raw_csv, summary_csv, json_path = _write_outputs(
        args,
        rows=rows,
        summaries=summaries,
        best=best,
        metadata=metadata,
    )
    print("\nOutputs")
    print(f"  raw       : {raw_csv}")
    print(f"  summary   : {summary_csv}")
    print(f"  full JSON : {json_path}")
    print(f"  passed    : {sum(bool(row.get('passed')) for row in rows)}/{len(rows)}")
    uncovered_case_orders = _uncovered_case_orders(summaries)
    exception_count = sum(bool(row.get("exception_type")) for row in rows)
    print(f"  uncovered : {len(uncovered_case_orders)} case/order pair(s)")
    print(f"  exceptions: {exception_count}")
    return _validation_exit_code(
        strict=args.strict,
        require_coverage=args.require_coverage,
        any_candidate_failed=failed,
        uncovered_count=len(uncovered_case_orders),
        exception_count=exception_count,
    )


if __name__ == "__main__":
    raise SystemExit(main())
