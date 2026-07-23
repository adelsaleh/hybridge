#!/usr/bin/env python3
"""Run the advection-reaction HDG fast upwind-GS/Cupyx path.

This is a performance-oriented solver runner for the same manufactured case
family used by ``scripts/advection_reaction/run_adv_rea_cases.py``.  It keeps
the path deliberately narrow:

1. project the manufactured coefficients on the host DG space;
2. compute the upwind-SCC free-edge ordering;
3. assemble the reduced trace matrix with the Numba eliminated HDG kernel,
   already in upwind order, while also emitting dense edge-block COO entries;
4. build the forward upwind block-GS preconditioner from those blocks;
5. build the scaled CuPy CSR matrix from the ordered COO stream and solve with
   Cupyx BiCGSTAB;
6. copy the reduced trace back only for reconstruction and error evaluation.

The script is not a comparison harness.  Use
``experimental/check_upwind_block_gs_onfly_adv_rea.py`` for parity checks and
side-by-side reference timings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from numba import config as numba_config
    from numba import get_num_threads, njit, set_num_threads
except ImportError:  # pragma: no cover - optional dependency guard.
    get_num_threads = None
    njit = None
    numba_config = None
    set_num_threads = None

from hdgfem.assembly import matrices_numpy as hdg_mats
from hdgfem.backends.numba import (
    assemble_projected_trace_system_eliminated_numba,
    reconstruct_projected_field_numba,
)
from hdgfem.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.io.output import pretty_print_sections
from hdgfem.linalg.ordering import upwind_scc_trace_ordering
from hdgfem.linalg.system import expand_known_dofs
from hdgfem.linalg.upwind_block_gs_onfly import (
    build_forward_upwind_block_gs_from_ordered_block_coo,
    scale_ordered_trace_coo_from_block_gs,
)
from scripts.advection_reaction.adv_rea_cases import case_definition_by_key


@dataclass
class CupyxRun:
    info: int
    iterations: int
    residual_norm: float
    relative_residual: float
    residual_target: float
    m_calls: int
    m_apply_seconds: float


class StageLogger:
    """Minimal verbosity-aware wall-clock logger."""

    def __init__(self, verbosity: int):
        self.verbosity = int(verbosity)
        self.timings: dict[str, float] = {}
        self.order: list[str] = []

    def start(self, key: str, label: str, *, level: int = 1) -> float:
        if self.verbosity >= level:
            print(f"{label} ...", flush=True)
        return time.perf_counter()

    def done(self, key: str, start: float, label: str, *, level: int = 1, extra: str | None = None) -> float:
        elapsed = time.perf_counter() - start
        self.timings[key] = elapsed
        self.order.append(key)
        if self.verbosity >= level:
            suffix = "" if extra is None else f" ({extra})"
            print(f"{label} ... done in {elapsed:.5f}s{suffix}", flush=True)
        return elapsed



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="test2_legacy_gpu3")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.01)
    parser.add_argument("--mesh-type", "-mt", choices=("rectangle", "structured-rectangle"), default="rectangle")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default="legacy-lagrange")
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1.0e-13)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--cupyx-solver", choices=("bicgstab", "gmres", "cg", "cgs"), default="bicgstab")
    parser.add_argument("--gmres-restart", type=int, default=None, help="restart length for --cupyx-solver gmres")
    parser.add_argument("--check-rtol", type=float, default=1.0e-10)
    parser.add_argument("--trace-ordering-flux-tolerance", type=float, default=0.0)
    parser.add_argument("--diagonal-regularization", type=float, default=0.0)
    parser.add_argument("--apply-mode", choices=("auto", "serial", "parallel"), default="auto")
    parser.add_argument("--parallel-min-width", type=int, default=1024)
    parser.add_argument("--max-couplings-per-block", type=int, default=6)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--gmsh-num-threads", type=int, default=None)
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help="Numba worker threads; default uses all Numba-configured threads",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--plot", action="store_true", help="show numerical/exact/error plots after a successful solve")
    parser.add_argument("--plot-resolution", type=int, default=10, help="HDG/error plot sampling resolution per element")
    parser.add_argument(
        "--exact-plot-resolution",
        type=int,
        default=None,
        help="exact-solution panel resolution per element; default is twice --plot-resolution",
    )
    parser.add_argument("--hide-mesh", action="store_true", help="hide mesh overlay in plots")
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="0: summary only, 1: stage progress, 2: detailed timings",
    )
    return parser


def configure_numba_threads(requested: int | None) -> int:
    """Set and return the Numba worker thread count used by parallel kernels."""
    if get_num_threads is None or set_num_threads is None or numba_config is None:
        raise RuntimeError("this runner requires numba")

    max_threads = int(numba_config.NUMBA_NUM_THREADS)
    if requested is None:
        requested = max_threads
    requested = int(requested)
    if requested <= 0:
        raise ValueError("--numba-threads must be positive")
    if requested > max_threads:
        raise ValueError(
            f"--numba-threads={requested} exceeds Numba's configured maximum "
            f"NUMBA_NUM_THREADS={max_threads}; set the environment variable before process start"
        )
    set_num_threads(requested)
    return int(get_num_threads())


def build_mesh(args):
    if args.mesh_type == "structured-rectangle":
        return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    return gmsh_rectangle_mesh(
        args.mesh_size,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        verbosity=args.gmsh_verbosity,
        algorithm=args.gmsh_algorithm,
        num_threads=args.gmsh_num_threads,
    )


def build_case(args):
    try:
        case = case_definition_by_key(args.case)
    except ValueError:
        if args.case != "test2_legacy_gpu3":
            raise
        case = case_definition_by_key("test2")
    return case, case.build()


def active_free_edges(space: DGSpace) -> np.ndarray:
    mask = np.ones(space.mesh.num_edg, dtype=bool)
    mask[space.mesh.bnd_edges_inds] = False
    return np.flatnonzero(mask).astype(np.int64)


def project_problem(space: DGSpace, beta_x, beta_y, reaction, source):
    beta_x_h = space.project_callable(beta_x, name="beta_x_h")
    beta_y_h = space.project_callable(beta_y, name="beta_y_h")
    beta_h = (space * space).field((beta_x_h, beta_y_h), name="beta_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    source_h = space.project_callable(source, name="source_h")
    return beta_h, reaction_h, source_h


def reconstruct_full_trace(x_ordered: np.ndarray, reduction, dof_permutation: np.ndarray) -> np.ndarray:
    """Return the full trace in the natural global-edge trace layout."""
    natural_reduced = np.empty_like(x_ordered)
    natural_reduced[np.asarray(dof_permutation, dtype=np.int64)] = np.asarray(x_ordered, dtype=np.float64)
    return expand_known_dofs(natural_reduced, reduction)


def solve_cupyx(matrix_rows, matrix_cols, matrix_data, rhs, preconditioner, args, logger: StageLogger) -> tuple[CupyxRun, Any]:
    import_start = logger.start("cupyx_import", "loading CuPy/Cupyx helpers", level=2)
    from hdgfem.backends.cupy import require_cupy, scipy_coo_to_cupy_csr, solve_cupyx_csr
    from hdgfem.linalg.cupy_upwind_block_gs import cupy_upwind_block_gs_from_host_preconditioner

    cupy = require_cupy()
    logger.done("cupyx_import", import_start, "loading CuPy/Cupyx helpers", level=2)

    matrix_start = logger.start("cupyx_matrix_build", "building CuPy CSR from scaled COO", level=1)
    matrix_cp = scipy_coo_to_cupy_csr(
        matrix_rows,
        matrix_cols,
        matrix_data,
        (rhs.size, rhs.size),
        dtype=cupy.float64,
    )
    logger.done("cupyx_matrix_build", matrix_start, "building CuPy CSR from scaled COO", level=1, extra=f"nnz={matrix_cp.nnz:,}")

    rhs_start = logger.start("cupyx_rhs_h2d", "copying scaled RHS to device", level=2)
    rhs_cp = cupy.asarray(rhs, dtype=cupy.float64)
    cupy.cuda.get_current_stream().synchronize()
    logger.done("cupyx_rhs_h2d", rhs_start, "copying scaled RHS to device", level=2, extra=f"size={rhs.size:,}")

    prec_start = logger.start("cupyx_preconditioner_export", "exporting upwind-GS preconditioner to device", level=1)
    preconditioner_cp = cupy_upwind_block_gs_from_host_preconditioner(
        preconditioner,
        dtype=cupy.float64,
        warm_start=not args.no_warmup,
    )
    logger.done(
        "cupyx_preconditioner_export",
        prec_start,
        "exporting upwind-GS preconditioner to device",
        level=1,
        extra=f"retained={preconditioner.stats.retained_block_couplings:,}",
    )

    impl = getattr(preconditioner_cp, "_upwind_block_gs_impl", None)
    if impl is not None:
        impl.reset_timing()

    solve_label = f"solving scaled ordered system with Cupyx {args.cupyx_solver.upper()}"
    solve_start = logger.start("cupyx_krylov", solve_label, level=1)
    x_cp, info, iterations = solve_cupyx_csr(
        matrix_cp,
        rhs_cp,
        solver=args.cupyx_solver,
        preconditioner=preconditioner_cp,
        rtol=args.rtol,
        atol=args.atol,
        maxiter=args.maxiter,
        restart=args.gmres_restart,
    )
    logger.done(
        "cupyx_krylov",
        solve_start,
        solve_label,
        level=1,
        extra=f"info={int(info)}, iters={int(iterations)}",
    )

    residual_start = logger.start("cupyx_residual", "evaluating scaled device residual", level=1)
    residual_cp = matrix_cp @ x_cp - rhs_cp
    residual_norm = float(cupy.asnumpy(cupy.linalg.norm(residual_cp)))
    rhs_norm = float(cupy.asnumpy(cupy.linalg.norm(rhs_cp)))
    cupy.cuda.get_current_stream().synchronize()
    relative = residual_norm / rhs_norm if rhs_norm != 0.0 else residual_norm
    target = max(float(args.rtol) * rhs_norm, float(args.atol))
    logger.done("cupyx_residual", residual_start, "evaluating scaled device residual", level=1, extra=f"rel={relative:.3e}")

    return (
        CupyxRun(
            info=int(info),
            iterations=int(iterations),
            residual_norm=residual_norm,
            relative_residual=relative,
            residual_target=target,
            m_calls=int(getattr(preconditioner_cp, "apply_count", 0) or 0),
            m_apply_seconds=float(getattr(preconditioner_cp, "apply_seconds", 0.0) or 0.0),
        ),
        x_cp,
    )


def format_seconds(value: float, total: float | None = None) -> str:
    if total is None or total <= 0.0:
        return f"{value:.5f}s"
    return f"{value:.5f}s ({100.0 * value / total:.1f}%)"


def print_detail_table(logger: StageLogger, total_seconds: float) -> None:
    seen: set[str] = set()
    rows: list[tuple[str, str, str]] = []
    for key in logger.order:
        if key in seen:
            continue
        seen.add(key)
        seconds = logger.timings[key]
        rows.append((key.replace("_", " "), f"{seconds:.5f}", f"{100.0 * seconds / max(total_seconds, 1.0e-300):.1f}%"))
    widths = [max(len(row[i]) for row in [("stage", "seconds", "% total"), *rows]) for i in range(3)]
    print()
    print("HDGFEM Upwind-GS/Cupyx Detailed Timings")
    print("-" * (sum(widths) + 4))
    print(f"{'stage'.ljust(widths[0])}  {'seconds'.rjust(widths[1])}  {'% total'.rjust(widths[2])}")
    print(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}")
    for stage, seconds, percent in rows:
        print(f"{stage.ljust(widths[0])}  {seconds.rjust(widths[1])}  {percent.rjust(widths[2])}")


def dense_exact_plot_resolution(
        exact_resolution: int | str | None,
        *,
        numerical_resolution: int,
        num_elements: int,
) -> int | str | None:
    """Choose an exact-panel resolution denser than the HDG/error panels."""
    if exact_resolution is not None:
        return exact_resolution
    return max(2 * int(numerical_resolution), int(numerical_resolution) + 1)


def plot_resolution(requested_resolution: int | None) -> int:
    """Return the requested per-element HDG/error plotting grid resolution."""
    if requested_resolution is None:
        return 10
    return max(2, int(requested_resolution))


def plot_solution(field, exact, *, resolution: int, exact_resolution: int | str | None, title: str, show_mesh: bool):
    """Plot numerical, exact, and absolute-error panels for the reconstructed field."""
    from hdgfem.io.plot import (
        _resolve_exact_plot_resolution,
        plot_scalar_sample_panels_matplotlib,
        plot_solution_comparison,
        sample_callable_on_elements,
        sample_field_on_elements,
    )

    mesh = field.space.mesh
    if mesh.num_tri > 100:
        return plot_solution_comparison(
            field,
            exact,
            resolution=resolution,
            exact_resolution=exact_resolution,
            title=title,
            show_mesh=show_mesh,
        )

    reference_points, _, numerical_values = sample_field_on_elements(
        field,
        resolution=resolution,
    )
    _, _, exact_values_for_error = sample_callable_on_elements(
        mesh,
        exact,
        reference_points=reference_points,
    )
    absolute_error = np.abs(numerical_values - exact_values_for_error)
    exact_panel_resolution = _resolve_exact_plot_resolution(
        exact_resolution,
        numerical_resolution=resolution,
        num_elements=mesh.num_tri,
    )
    exact_reference_points, _, exact_display_values = sample_callable_on_elements(
        mesh,
        exact,
        resolution=exact_panel_resolution,
    )
    return plot_scalar_sample_panels_matplotlib(
        mesh,
        (
            ("Numerical solution", reference_points, numerical_values),
            ("Exact solution", exact_reference_points, exact_display_values),
            ("Absolute error", reference_points, absolute_error),
        ),
        suptitle=title,
        show_mesh=show_mesh,
        cmap="jet",
        levels=128,
        share_clim=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if njit is None:
        raise RuntimeError("this runner requires numba")

    logger = StageLogger(args.verbosity)
    numba_threads = configure_numba_threads(args.numba_threads)
    run_start = time.perf_counter()
    if args.verbosity >= 1:
        print("HDGFEM upwind-SCC + upwind-GS + Cupyx advection-reaction runner", flush=True)
        print(
            f"case={args.case}, order={args.order}, mesh_size={args.mesh_size}, "
            f"basis={args.basis}, trace_basis={args.trace_basis}, "
            f"numba_threads={numba_threads}/{numba_config.NUMBA_NUM_THREADS}, "
            f"os_cpu_count={os.cpu_count()}",
            flush=True,
        )

    mesh_start = logger.start("mesh", "building mesh")
    mesh = build_mesh(args)
    logger.done("mesh", mesh_start, "building mesh", extra=f"triangles={mesh.num_tri:,}, edges={mesh.num_edg:,}")

    space_start = logger.start("space", "building DG space")
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    logger.done("space", space_start, "building DG space", extra=f"el_dof={space.el_dof:,}, edge_dof={space.quad_data.edg_dof:,}")

    case_start = logger.start("case", "building manufactured case", level=2)
    case, (beta_x, beta_y, reaction, source, exact) = build_case(args)
    logger.done("case", case_start, "building manufactured case", level=2)

    project_start = logger.start("project", "projecting coefficients")
    beta_h, reaction_h, source_h = project_problem(space, beta_x, beta_y, reaction, source)
    logger.done("project", project_start, "projecting coefficients")

    ordering_start = logger.start("ordering_total", "building upwind-SCC ordering")
    flux_start = time.perf_counter()
    beta_dot_normal = hdg_mats.advective_boundary_normal(beta_h, space)
    logger.timings["beta_dot_normal"] = time.perf_counter() - flux_start
    logger.order.append("beta_dot_normal")
    scc_start = time.perf_counter()
    ordering = upwind_scc_trace_ordering(
        mesh,
        beta_dot_normal,
        space.quad_data.edg_dof,
        active_edges=active_free_edges(space),
        flux_tolerance=args.trace_ordering_flux_tolerance,
    )
    logger.timings["upwind_scc"] = time.perf_counter() - scc_start
    logger.order.append("upwind_scc")
    logger.done(
        "ordering_total",
        ordering_start,
        "building upwind-SCC ordering",
        extra=f"levels={ordering.diagnostics.level_widths.num_levels:,}",
    )

    assembly_start = logger.start("numba_ordered_assembly", "assembling ordered reduced trace system")
    assembly = assemble_projected_trace_system_eliminated_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        edge_order=ordering.edge_order,
        return_block_coo=True,
    )
    trace_system = assembly.trace_system
    if assembly.block_rows is None or assembly.block_cols is None or assembly.block_data is None:
        raise RuntimeError("Numba assembly did not return block COO data")
    logger.done(
        "numba_ordered_assembly",
        assembly_start,
        "assembling ordered reduced trace system",
        extra=f"triplets={trace_system.data.size:,}, block_entries={assembly.block_data.shape[0]:,}",
    )
    if args.verbosity >= 2:
        for key, value in sorted(assembly.timings.items()):
            if isinstance(value, (float, int)) and key != "block_coo_entries":
                logger.timings[f"assembly_{key}"] = float(value)
                logger.order.append(f"assembly_{key}")

    edge_dof = int(space.quad_data.edg_dof)
    num_blocks = int(trace_system.rhs.size // edge_dof)

    precond_start = logger.start("preconditioner", "building forward upwind-GS preconditioner")
    preconditioner = build_forward_upwind_block_gs_from_ordered_block_coo(
        assembly.block_rows,
        assembly.block_cols,
        assembly.block_data,
        num_blocks,
        level_widths=ordering.diagnostics.level_widths,
        diagonal_regularization=args.diagonal_regularization,
        apply_mode=args.apply_mode,
        parallel_min_width=args.parallel_min_width,
        bounded_max_couplings_per_block=args.max_couplings_per_block,
        warm_start=not args.no_warmup,
    )
    logger.done(
        "preconditioner",
        precond_start,
        "building forward upwind-GS preconditioner",
        extra=f"retained={preconditioner.stats.retained_block_couplings:,}",
    )

    scale_start = logger.start("row_scaling", "applying block-GS row scale to COO/RHS")
    scaled_data, scaled_rhs = scale_ordered_trace_coo_from_block_gs(
        trace_system.rows,
        trace_system.data,
        trace_system.rhs,
        preconditioner,
    )
    logger.done("row_scaling", scale_start, "applying block-GS row scale to COO/RHS")

    cupyx_run, x_cp = solve_cupyx(
        np.ascontiguousarray(trace_system.rows, dtype=np.int64),
        np.ascontiguousarray(trace_system.cols, dtype=np.int64),
        scaled_data,
        scaled_rhs,
        preconditioner,
        args,
        logger,
    )

    from hdgfem.backends.cupy import require_cupy

    cupy = require_cupy()
    copy_start = logger.start("solution_d2h", "copying reduced trace to host for reconstruction", level=1)
    x_ordered = np.ascontiguousarray(cupy.asnumpy(x_cp), dtype=np.float64)
    logger.done("solution_d2h", copy_start, "copying reduced trace to host for reconstruction", level=1)

    trace_start = logger.start("trace_expand", "expanding ordered reduced trace", level=2)
    if assembly.reduction is None:
        raise RuntimeError("Numba eliminated assembly did not return a boundary reduction")
    full_trace = reconstruct_full_trace(x_ordered, assembly.reduction, ordering.dof_permutation)
    logger.done("trace_expand", trace_start, "expanding ordered reduced trace", level=2)

    recon_start = logger.start("reconstruct", "reconstructing element field")
    field = reconstruct_projected_field_numba(full_trace, source_h, beta_h, reaction_h, space)
    logger.done("reconstruct", recon_start, "reconstructing element field")

    error_start = logger.start("error_eval", "evaluating errors")
    l2_error = float(field.l2_error(exact))
    values = field.values()
    points = space.mapped_quads()
    exact_values = exact(points[:, :, 0], points[:, :, 1])
    abs_error = np.abs(values - exact_values)
    linf_error = float(np.max(abs_error))
    element_max_error = np.max(abs_error, axis=1)
    avg_error = float(np.average(element_max_error))
    max_error_element = int(np.argmax(element_max_error))
    logger.done("error_eval", error_start, "evaluating errors", extra=f"L2={l2_error:.3e}")

    total_seconds = time.perf_counter() - run_start

    if args.verbosity >= 2:
        print_detail_table(logger, total_seconds)

    total = total_seconds
    timings = logger.timings
    setup_timing_items = [
        ("mesh", format_seconds(timings.get("mesh", 0.0), total), "s"),
        ("space", format_seconds(timings.get("space", 0.0), total), "s"),
        ("project", format_seconds(timings.get("project", 0.0), total), "s"),
        ("ordering", format_seconds(timings.get("ordering_total", 0.0), total), "s"),
        ("assembly", format_seconds(timings.get("numba_ordered_assembly", 0.0), total), "s"),
        ("row scaling", format_seconds(timings.get("row_scaling", 0.0), total), "s"),
        ("preconditioner", format_seconds(timings.get("preconditioner", 0.0), total), "s"),
        ("CuPy CSR build", format_seconds(timings.get("cupyx_matrix_build", 0.0), total), "s"),
        ("Cupyx solve", format_seconds(timings.get("cupyx_krylov", 0.0), total), "s"),
        ("reconstruct", format_seconds(timings.get("reconstruct", 0.0), total), "s"),
        ("error eval", format_seconds(timings.get("error_eval", 0.0), total), "s"),
        ("total measured", format_seconds(total), "s"),
    ]
    sections = [
        (
            "Run / Options",
            [
                ("case", args.case, "s"),
                ("order", args.order, "d"),
                ("basis", args.basis, "s"),
                ("trace basis", args.trace_basis, "s"),
                ("ordering", "upwind-scc", "s"),
                ("preconditioner", "forward upwind block-GS", "s"),
                ("solver", f"cupyx {args.cupyx_solver}", "s"),
                ("numba threads", numba_threads, ",d"),
            ],
        ),
        (
            "Mesh / DOF",
            [
                ("mesh", args.mesh_type, "s"),
                ("mesh size", args.mesh_size, ".5g"),
                ("h", mesh.h, ".3e"),
                ("triangles", mesh.num_tri, ",d"),
                ("edges", mesh.num_edg, ",d"),
                ("system dof", trace_system.rhs.size, ",d"),
                ("edge dof", edge_dof, "d"),
                ("triplets", trace_system.data.size, ",d"),
                ("block entries", assembly.block_data.shape[0], ",d"),
            ],
        ),
        (
            "Solver",
            [
                ("rtol", args.rtol, ".1e"),
                ("atol", args.atol, ".1e"),
                ("maxiter", args.maxiter, ",d"),
                ("gmres restart", -1 if args.gmres_restart is None else args.gmres_restart, ",d"),
                ("info", cupyx_run.info, "d"),
                ("iterations", cupyx_run.iterations, ",d"),
                ("scaled rel residual", cupyx_run.relative_residual, ".3e"),
                ("M calls", cupyx_run.m_calls, ",d"),
                ("M apply", cupyx_run.m_apply_seconds, ".5f"),
            ],
        ),
        (
            "Errors",
            [
                ("h^(p+1)", mesh.h ** (space.order + 1), ".3e"),
                ("L2 error", l2_error, ".3e"),
                ("Linf error", linf_error, ".3e"),
                ("avg max error", avg_error, ".3e"),
                ("max-error element", max_error_element, "d"),
            ],
        ),
        ("Timings", setup_timing_items),
    ]
    pretty_print_sections(sections, title="HDGFEM Upwind-GS/Cupyx Advection-Reaction Solve Summary")

    if args.json_output is not None:
        payload: dict[str, Any] = {
            "inputs": vars(args),
            "runtime": {
                "numba_threads": int(numba_threads),
                "numba_configured_threads": int(numba_config.NUMBA_NUM_THREADS),
                "os_cpu_count": None if os.cpu_count() is None else int(os.cpu_count()),
            },
            "mesh": {
                "triangles": int(mesh.num_tri),
                "edges": int(mesh.num_edg),
                "h": float(mesh.h),
            },
            "dofs": {
                "system": int(trace_system.rhs.size),
                "edge": int(edge_dof),
                "element": int(space.el_dof),
                "triplets": int(trace_system.data.size),
                "block_entries": int(assembly.block_data.shape[0]),
            },
            "ordering": asdict(ordering.diagnostics),
            "preconditioner": asdict(preconditioner.stats),
            "solver": asdict(cupyx_run),
            "errors": {
                "l2": l2_error,
                "linf": linf_error,
                "avg_max": avg_error,
                "max_error_element": max_error_element,
            },
            "timings": timings,
            "total_seconds": total_seconds,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if args.verbosity >= 1:
            print(f"\nwrote JSON summary to {args.json_output}", flush=True)

    if args.plot:
        plot_resolution_value = plot_resolution(args.plot_resolution)
        exact_plot_resolution = dense_exact_plot_resolution(
            args.exact_plot_resolution,
            numerical_resolution=plot_resolution_value,
            num_elements=mesh.num_tri,
        )
        title = (
            f"{args.case}, p={space.order}, elements={mesh.num_tri:,}, "
            f"L2={l2_error:.2e}, Cupyx {args.cupyx_solver}"
        )
        plot_start = logger.start("plot", "plotting numerical/exact/error panels", level=1)
        plot_solution(
            field,
            exact,
            resolution=plot_resolution_value,
            exact_resolution=exact_plot_resolution,
            title=title,
            show_mesh=not args.hide_mesh,
        )
        logger.done("plot", plot_start, "plotting numerical/exact/error panels", level=1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
