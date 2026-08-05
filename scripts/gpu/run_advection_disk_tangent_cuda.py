#!/usr/bin/env python3
"""Run the disk-tangent zero-flux advection-reaction test with raw CUDA."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem.backends.cupy import as_cupy_space, require_cupy, require_cupyx_sparse, require_pyamgx
from hdgfem.backends.raw_cuda import resolve_raw_cuda_block_size
from hdgfem.core.mesh import gmsh_disc_mesh
from hdgfem.core.space import DGSpace, VectorDGField
from hdgfem.io.output import pretty_print_sections
from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver
from scripts.advection_reaction.cases import case_definition_by_key
from hdgfem.io.plot import plot_solution_comparison
from scripts.gpu.run_advection_reaction_cuda import (
    effective_plot_resolution,
    evaluate_errors,
    evaluate_errors_device,
    plot_sampled_solution_comparison,
)


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "amgx"
DEFAULT_AMGX_CONFIG_PATH = CONFIG_DIR / "adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json"

DEFAULT_AMGX_CONFIG = {
    "config_version": 2,
    "determinism_flag": 1,
    "exception_handling": 1,
    "solver": {
        "solver": "BICGSTAB",
        "monitor_residual": 1,
        "convergence": "RELATIVE_INI_CORE",
        "tolerance": 1.0e-11,
        "max_iters": 1000,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "selector": "PMIS",
            "cycle": "W",
        },
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.03)
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--trace-basis", default="legacy-lagrange", choices=("legacy-lagrange", "legendre-modal"))
    parser.add_argument("--volume-quadrature", choices=("auto", "symmetric", "duffy"), default="symmetric")
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--error-volume-quad-1d", type=int, default=None)
    parser.add_argument("--raw-lu-mode", choices=("safe", "coop"), default="coop")
    parser.add_argument("--raw-block-size", choices=("auto", "1", "32", "64", "128"), default="auto")
    parser.add_argument("--raw-matrix-format", choices=("auto", "coo", "csr"), default="auto")
    parser.add_argument("--solver", choices=("amgx", "direct", "bicgstab"), default="amgx")
    parser.add_argument("--tolerance", type=float, default=1.0e-11)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--amgx-config", default=str(DEFAULT_AMGX_CONFIG_PATH))
    parser.add_argument("--amgx-solver", default=None, help="override the solver named in the AMGX config")
    parser.add_argument("--ilu-drop-tol", type=float, default=1.0e-5)
    parser.add_argument("--ilu-fill-factor", type=float, default=5.0)
    parser.add_argument("--ilu-permc-spec", default="COLAMD", choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"))
    parser.add_argument("--scale-system", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--materialize-host-system", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--materialize-host-solution", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evaluate-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action="store_true", help="show numerical/exact/error plots after the summary")
    parser.add_argument("--plot-resolution", "-pr", type=int, default=20, help="plot/error sampling resolution; coarse meshes use a polynomial-degree minimum")
    parser.add_argument(
        "--exact-plot-resolution",
        default="auto",
        help="exact-solution panel resolution: integer, 'auto', or 'same'",
    )
    parser.add_argument("--hide-mesh", action="store_true", help="hide mesh overlay in plots")
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--gmsh-num-threads", type=int, default=None)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--show-cupy-config", action="store_true")
    return parser


def load_amgx_config(args):
    config_path = Path(args.amgx_config).expanduser() if args.amgx_config else None
    if config_path is not None and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    elif config_path is not None:
        raise FileNotFoundError(f"AMGX config file not found: {config_path}")
    else:
        config = copy.deepcopy(DEFAULT_AMGX_CONFIG)
    solver_config = config.setdefault("solver", {})
    if args.amgx_solver is not None:
        solver_config["solver"] = str(args.amgx_solver)
    solver_config["tolerance"] = float(args.tolerance)
    solver_config["max_iters"] = int(args.maxiter)
    return config, config_path


def _fmt(value, spec):
    if spec == "s":
        return str(value)
    return format(value, spec)


def _maybe(value, default="n/a"):
    return default if value is None else value


def describe_amgx_solver(config):
    if not config:
        return "amgx"
    solver_config = config.get("solver", {})
    return str(solver_config.get("solver", "amgx"))


def describe_amgx_preconditioner(config):
    if not config:
        return "none"
    preconditioner = config.get("solver", {}).get("preconditioner")
    if preconditioner is None:
        return "none"
    if isinstance(preconditioner, str):
        return preconditioner
    if not isinstance(preconditioner, dict):
        return type(preconditioner).__name__

    parts = [str(preconditioner.get("solver", "unknown"))]
    algorithm = preconditioner.get("algorithm")
    if algorithm:
        parts.append(str(algorithm))
    smoother = preconditioner.get("smoother")
    if isinstance(smoother, dict):
        smoother = smoother.get("solver")
    if smoother:
        parts.append(str(smoother))
    cycle = preconditioner.get("cycle")
    if cycle:
        parts.append(f"{cycle}-cycle")
    presweeps = preconditioner.get("presweeps")
    postsweeps = preconditioner.get("postsweeps")
    if presweeps is not None or postsweeps is not None:
        parts.append(f"pre/post={presweeps or 0}/{postsweeps or 0}")
    return " / ".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.raw_block_size = resolve_raw_cuda_block_size(
        args.raw_block_size,
        equation="advection-reaction",
        order=args.order,
    )
    cp = require_cupy()
    require_cupyx_sparse()
    if args.solver == "amgx":
        require_pyamgx()
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    if args.show_cupy_config:
        cp.show_config()

    run_start = time.perf_counter()
    mesh_start = time.perf_counter()
    mesh = gmsh_disc_mesh(
        args.mesh_size,
        center=(0.0, 0.0),
        radius=1.0,
        verbosity=args.gmsh_verbosity,
        algorithm=args.gmsh_algorithm,
        num_threads=args.gmsh_num_threads,
    )
    mesh_time = time.perf_counter() - mesh_start

    space_start = time.perf_counter()
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quadrature=args.volume_quadrature,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    space_time = time.perf_counter() - space_start
    plot_resolution = effective_plot_resolution(args.plot_resolution, space.order, mesh.num_tri)

    case = case_definition_by_key("disk_tangent")
    beta_x, beta_y, reaction, source, exact = case.build()

    projection_start = time.perf_counter()
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_h = VectorDGField((beta_x, beta_y), space, name="beta_h")
    projection_time = time.perf_counter() - projection_start

    amgx_config = None
    amgx_path = None
    solver_name = args.solver
    solver_display_name = args.solver
    preconditioner = None
    preconditioner_display_name = "none"
    if args.solver == "amgx":
        amgx_config, amgx_path = load_amgx_config(args)
        solver_name = "amgx"
        solver_display_name = describe_amgx_solver(amgx_config)
        preconditioner_display_name = describe_amgx_preconditioner(amgx_config)
    elif args.solver == "bicgstab":
        solver_name = "BICGSTAB"
        solver_display_name = "BICGSTAB"
        preconditioner = "ilu"
        preconditioner_display_name = "ilu"
    elif args.solver == "direct":
        solver_name = "direct"
        solver_display_name = "direct"

    materialize_host_solution = args.materialize_host_solution
    if (args.evaluate_errors or args.plot) and args.solver != "amgx":
        materialize_host_solution = True if materialize_host_solution is None else materialize_host_solution

    solver = AdvectionReactionHDGSolver(
        space,
        source=source_h,
        beta=beta_h,
        reaction=reaction_h,
        solver=solver_name,
        preconditioner=preconditioner,
        solver_rtol=args.tolerance,
        maxiter=args.maxiter,
        amgx_config=amgx_config,
        ilu_drop_tol=args.ilu_drop_tol,
        ilu_fill_factor=args.ilu_fill_factor,
        ilu_permc_spec=args.ilu_permc_spec,
        scale_system=args.scale_system,
        boundary_mode="zero-flux",
        trace_ordering="none",
        assembly_backend="raw-cuda",
        trace_basis=args.trace_basis,
        raw_local_assembly="fused",
        raw_lu_mode=args.raw_lu_mode,
        raw_block_size=args.raw_block_size,
        raw_matrix_format=args.raw_matrix_format,
        materialize_host_system=args.materialize_host_system,
        materialize_host_solution=materialize_host_solution,
        verbose=args.verbosity,
    )

    solve_start = time.perf_counter()
    result = solver.solve()
    solve_call_time = time.perf_counter() - solve_start

    l2 = linf = avg_max = np.nan
    max_element = -1
    error_time = 0.0
    error_mode = "not evaluated"
    plot_samples = None
    if args.evaluate_errors or args.plot:
        error_start = time.perf_counter()
        if result.field_device is not None:
            cspace = as_cupy_space(space)
            error_result = evaluate_errors_device(
                result.field_device,
                cspace,
                exact,
                plot_resolution,
                args.error_volume_quad_1d,
                return_plot_samples=args.plot,
            )
            if args.plot:
                l2, linf, avg_max, max_element, plot_samples = error_result
            else:
                l2, linf, avg_max, max_element = error_result
            error_mode = "device"
        elif result.field is not None:
            l2, linf, avg_max, max_element = evaluate_errors(
                result.field,
                exact,
                plot_resolution,
                args.error_volume_quad_1d,
            )
            error_mode = "host"
        else:
            raise RuntimeError("error evaluation/plotting requires a host or device field")
        error_time = time.perf_counter() - error_start

    total = time.perf_counter() - run_start
    solve = result.global_solve_result
    trace_space = space.trace_space(args.trace_basis)
    global_dof = mesh.int_edges_inds.size * trace_space.edg_dof
    matrix_format = args.raw_matrix_format
    if matrix_format == "auto":
        matrix_format = "csr" if args.solver == "amgx" and not args.materialize_host_system else "coo"

    detail = result.timings.details
    setup_time = mesh_time + space_time + projection_time
    sections = [
        (
            "Run / Options",
            [
                ("case", "disk_tangent", "s"),
                ("domain", "disc", "s"),
                ("order", args.order, ",d"),
                ("basis", args.basis, "s"),
                ("trace basis", args.trace_basis, "s"),
                ("boundary mode", "zero-flux", "s"),
                ("raw LU", args.raw_lu_mode, "s"),
                ("raw block", args.raw_block_size, ",d"),
                ("matrix", matrix_format, "s"),
            ],
        ),
        (
            "Mesh / DOF",
            [
                ("mesh size", args.mesh_size, ".4f"),
                ("h", mesh.h, ".3e"),
                ("triangles", mesh.num_tri, ",d"),
                ("edges", mesh.num_edg, ",d"),
                ("interior edges", mesh.int_edges_inds.size, ",d"),
                ("global dof", global_dof, ",d"),
                ("element dof", space.el_dof, ",d"),
            ],
        ),
        (
            "Solver",
            [
                ("solver", solver_display_name, "s"),
                ("AMGX config", "none" if amgx_path is None else Path(amgx_path).name, "s"),
                ("preconditioner", preconditioner_display_name, "s"),
                ("row scaled", "on" if args.scale_system else "off", "s"),
                ("iterations", -1 if solve is None or solve.iteration_count is None else solve.iteration_count, ",d"),
                ("rel residual", np.nan if solve is None or solve.solver_relative_residual_norm is None else solve.solver_relative_residual_norm, ".3e"),
            ],
        ),
        (
            "Errors",
            [
                ("theory h^(p+1)", mesh.h ** (args.order + 1), ".3e"),
                ("L2", l2, ".3e"),
                ("Linf", linf, ".3e"),
                ("avg max", avg_max, ".3e"),
                ("max element", max_element, ",d"),
            ],
        ),
        (
            "Timings",
            [
                ("mesh", mesh_time, ".3f"),
                ("space", space_time, ".3f"),
                ("projection", projection_time, ".3f"),
                ("assembly", result.timings.assembly, ".3f"),
                ("raw kernel", detail.get("raw.assembly.raw.kernel", detail.get("raw.assembly.raw.csr_kernel", np.nan)), ".3f"),
                ("global solve", result.timings.solve, ".3f"),
                ("reconstruct", result.timings.reconstruction, ".3f"),
                ("error", error_time, ".3f"),
                ("solver call", solve_call_time, ".3f"),
                ("total", total, ".3f"),
            ],
        ),
    ]
    print()
    print("Raw-CUDA Disk-Tangent Zero-Flux Advection-Reaction Summary")
    pretty_print_sections(sections)

    if args.plot:
        plot_title = f"disk_tangent, p={space.order}, elements={mesh.num_tri:,}, L2={l2:.2e}"
        print("plotting solution comparison ... ", end="", flush=True)
        plot_start = time.perf_counter()
        if plot_samples is not None:
            plot_sampled_solution_comparison(
                mesh,
                exact,
                plot_samples,
                numerical_resolution=plot_resolution,
                exact_resolution=args.exact_plot_resolution,
                polynomial_order=space.order,
                title=plot_title,
                show_mesh=not args.hide_mesh,
            )
        else:
            if result.field is None:
                raise RuntimeError("plotting requires a device field sample or a host-materialized DGField")
            plot_solution_comparison(
                result.field,
                exact,
                resolution=plot_resolution,
                exact_resolution=args.exact_plot_resolution,
                title=plot_title,
                show_mesh=not args.hide_mesh,
            )
        print(f"done in {time.perf_counter() - plot_start:.5f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
