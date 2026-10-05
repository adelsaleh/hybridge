#!/usr/bin/env python3
"""Run advection-reaction with package-owned upwind-SCC/Cupyx block-GS."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybridge import AdvectionReactionHDGSolver, DGSpace, VectorDGField, evaluate_scalar_error
from hybridge.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
from hybridge.runtime.logging import format_elapsed_percent
from hybridge.io.output import pretty_print_sections
from hybridge.io.plot import plot_solution_comparison, resolve_field_plot_resolution
from scripts.advection_reaction.cases import case_definition_by_key


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the compatibility CLI for the package-backed upwind runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="test2_legacy_gpu3")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.01)
    parser.add_argument("--mesh-type", "-mt", choices=("rectangle", "structured-rectangle"), default="rectangle")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal"), default="legacy-lagrange")
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1.0e-13)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--cupyx-solver", choices=("bicgstab", "gmres", "cg", "cgs"), default="bicgstab")
    parser.add_argument("--gmres-restart", type=int, default=None)
    parser.add_argument("--check-rtol", type=float, default=1.0e-10)
    parser.add_argument("--trace-ordering-flux-tolerance", type=float, default=0.0)
    parser.add_argument("--diagonal-regularization", type=float, default=0.0)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--gmsh-num-threads", type=int, default=None)
    parser.add_argument("--numba-threads", type=int, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-resolution", type=int, default=10)
    parser.add_argument("--exact-plot-resolution", type=int, default=None)
    parser.add_argument("--hide-mesh", action="store_true")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def configure_numba_threads(requested: int | None) -> int:
    """Configure and return the Numba worker count."""
    try:
        from numba import config, get_num_threads, set_num_threads
    except ImportError as exc:
        raise RuntimeError("this runner requires numba") from exc
    maximum = int(config.NUMBA_NUM_THREADS)
    selected = maximum if requested is None else int(requested)
    if selected <= 0 or selected > maximum:
        raise ValueError(f"--numba-threads must be in [1, {maximum}]")
    set_num_threads(selected)
    return int(get_num_threads())


def build_mesh(args):
    """Build the selected rectangle mesh."""
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


def _case(args):
    """Resolve the manufactured case including its legacy alias."""
    try:
        case = case_definition_by_key(args.case)
    except ValueError:
        if args.case != "test2_legacy_gpu3":
            raise
        case = case_definition_by_key("test2")
    return case, case.build()



def main(argv: list[str] | None = None) -> int:
    """Execute the reusable solver and print benchmark-compatible metrics."""
    args = build_arg_parser().parse_args(argv)
    numba_threads = configure_numba_threads(args.numba_threads)
    started = time.perf_counter()
    mesh = build_mesh(args)
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    case, (beta_x, beta_y, reaction, source, exact) = _case(args)
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_h = VectorDGField((beta_x, beta_y), space, name="beta_h")
    solver = AdvectionReactionHDGSolver(
        space,
        source=source_h,
        beta=beta_h,
        reaction=reaction_h,
        boundary_condition=exact,
        solver="cupyx",
        cupyx_solver=args.cupyx_solver,
        preconditioner="upwind_block_gs",
        solver_rtol=args.rtol,
        solver_atol=args.atol,
        maxiter=args.maxiter,
        restart=args.gmres_restart,
        scale_system=False,
        boundary_mode="eliminate",
        trace_ordering="upwind-scc",
        trace_ordering_flux_tolerance=args.trace_ordering_flux_tolerance,
        upwind_diagonal_regularization=args.diagonal_regularization,
        assembly_backend="numba",
        trace_basis=args.trace_basis,
        materialize_host_solution=True,
        verbose=args.verbosity,
    )
    result = solver.solve()
    solve = result.global_solve_result
    if (
        solve is not None
        and solve.physical_relative_residual_norm is not None
        and solve.physical_relative_residual_norm > args.check_rtol
    ):
        raise RuntimeError(
            f"physical relative residual {solve.physical_relative_residual_norm:.3e} "
            f"exceeds --check-rtol={args.check_rtol:.3e}"
        )
    report = evaluate_scalar_error(result.field, exact)
    elapsed = time.perf_counter() - started
    ordering = result.ordering_result
    metrics = report.metrics
    sections = [
        ("Run / Options", [
            ("case", args.case, "s"), ("order", args.order, "d"), ("basis", args.basis, "s"),
            ("trace basis", args.trace_basis, "s"), ("ordering", "upwind-scc", "s"),
            ("preconditioner", "cupyx upwind block-GS", "s"),
            ("solver", f"cupyx {args.cupyx_solver}", "s"), ("numba threads", numba_threads, ",d"),
        ]),
        ("Mesh / Error", [
            ("triangles", mesh.num_tri, ",d"), ("edges", mesh.num_edg, ",d"),
            ("L2 error", metrics.l2, ".3e"), ("Linf error", metrics.linf, ".3e"),
            ("avg max error", metrics.mean_element_linf, ".3e"), ("max-error element", metrics.max_element, ",d"),
        ]),
        ("Solver", [
            ("iterations", -1 if solve is None or solve.iteration_count is None else solve.iteration_count, ",d"),
            ("relative residual", float("nan") if solve is None else solve.relative_residual_norm, ".3e"),
            ("ordering levels", 0 if ordering is None else ordering.diagnostics.level_widths.num_levels, ",d"),
            ("largest SCC", 0 if ordering is None else ordering.diagnostics.largest_component_size, ",d"),
        ]),
        ("Timings", [
            ("assembly", format_elapsed_percent(result.timings.assembly, elapsed), "s"),
            ("global solve", format_elapsed_percent(result.timings.solve, elapsed), "s"),
            ("reconstruction", format_elapsed_percent(result.timings.reconstruction, elapsed), "s"),
            ("total measured", f"{elapsed:.3f}s", "s"),
        ]),
    ]
    pretty_print_sections(sections, title="HYBRIDGE Upwind-SCC / Cupyx Advection-Reaction Summary")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({
            "case": case.key,
            "order": args.order,
            "triangles": mesh.num_tri,
            "l2_error": metrics.l2,
            "linf_error": metrics.linf,
            "iterations": None if solve is None else solve.iteration_count,
            "relative_residual": None if solve is None else solve.relative_residual_norm,
            "total_seconds": elapsed,
            "numba_threads": numba_threads,
            "os_cpu_count": os.cpu_count(),
        }, indent=2), encoding="utf-8")
    if args.plot:
        resolution = resolve_field_plot_resolution(
            args.plot_resolution, order=space.order, num_elements=mesh.num_tri,
        )
        plot_solution_comparison(
            result.field,
            exact,
            resolution=resolution,
            exact_resolution="auto" if args.exact_plot_resolution is None else args.exact_plot_resolution,
            title=f"{case.key}, p={space.order}, L2={metrics.l2:.2e}",
            show_mesh=not args.hide_mesh,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
