#!/usr/bin/env python3
"""Package-backed CUDA diffusion-reaction HDG runner.

All numerical assembly, AMGX solve, reconstruction, diagnostics, and plotting
live in :mod:`hdgfem`; this module only translates CLI options into public API
calls and presents a benchmark-friendly summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem import DGSpace, DiffusionReactionHDGSolver, evaluate_scalar_error
from hdgfem.core.mesh import (
    gmsh_disc_mesh,
    gmsh_lshape_mesh,
    gmsh_rectangle_mesh,
    gmsh_triangle_mesh,
    rectangle_mesh,
)
from hdgfem.io.comparison import plot_sampled_solution_comparison
from hdgfem.linalg.amgx.config import load_amgx_config
from hdgfem.runtime.logging import format_elapsed_percent
from hdgfem.io.output import pretty_print_sections
from hdgfem.io.plot import resolve_field_plot_resolution, resolve_postprocessed_plot_resolution
from scripts.diffusion_reaction.cases import case_definition_by_key


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "amgx"
DEFAULT_AMGX_CONFIG_PATH = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json"

AMGX_CONFIG = {
    "config_version": 2,
    "solver": {
        "solver": "PCGF",
        "tolerance": 1.0e-13,
        "max_iters": 2000,
        "convergence": "RELATIVE_INI",
        "norm": "L2",
        "monitor_residual": 1,
        "store_res_history": 1,
        "print_solve_stats": 0,
        "preconditioner": {"solver": "AMG", "algorithm": "CLASSICAL", "selector": "PMIS"},
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="trigonometric-poisson")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.05)
    parser.add_argument(
        "--mesh-type", "-mt",
        choices=("auto", "disc", "rectangle", "unit-rectangle", "triangle", "lshape", "structured-rectangle"),
        default="auto",
    )
    parser.add_argument("--disc-radius", type=float, default=None)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--volume-quadrature", choices=("auto", "symmetric", "duffy"), default="auto")
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--error-volume-quad-1d", type=int, default=None)
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default="legacy-lagrange")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="cupy")
    parser.add_argument("--raw-matrix-format", choices=("coo", "csr"), default="csr")
    parser.add_argument("--raw-block-size", choices=("auto", "1", "32", "64", "128"), default="auto")
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="explicit constant tau_d; selects explicit mode",
    )
    parser.add_argument(
        "--diffusion-stabilization-mode",
        choices=("global-length", "explicit"),
        default="global-length",
        help="global gamma_d*kappa/L_Omega (default) or explicit --tau",
    )
    parser.add_argument(
        "--diffusion-domain-length",
        default="auto",
        help="positive L_Omega or auto for 2*area/boundary-length",
    )
    parser.add_argument(
        "--diffusion-stabilization-gamma",
        type=float,
        default=1.0,
        help="positive gamma_d multiplier for global-length mode",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-postprocess-primal", action="store_true")
    parser.add_argument(
        "--postprocess-backend", choices=("auto", "host", "cupy", "raw-cuda"), default="auto",
        help="retained for CLI compatibility; postprocessing is selected through the solver API",
    )
    parser.add_argument("--plot-resolution", "-pr", type=int, default=12)
    parser.add_argument("--exact-plot-resolution", default="auto")
    parser.add_argument("--hide-mesh", action="store_true")
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--amgx-config", default=None)
    parser.add_argument("--amgx-solver", default="PCGF")
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-13)
    parser.add_argument("--scale-system", choices=("symmetric", "left", "on", "off"), default="off")
    parser.add_argument("--amgx-maxiter", type=int, default=2000)
    parser.add_argument("--show-cupy-config", action="store_true")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def build_mesh(args, case):
    """Build the case domain selected by CLI metadata."""
    domain = case.default_domain if args.mesh_type == "auto" else args.mesh_type
    log_cache = args.verbosity >= 1
    if domain == "structured-rectangle":
        return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)), domain
    if domain == "unit-rectangle":
        return gmsh_rectangle_mesh(
            args.mesh_size, xlim=(0.0, 1.0), ylim=(0.0, 1.0), verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm, log_cache=log_cache,
        ), domain
    if domain == "rectangle":
        return gmsh_rectangle_mesh(
            args.mesh_size, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm, log_cache=log_cache,
        ), domain
    if domain == "disc":
        radius = args.disc_radius if args.disc_radius is not None else (5.0 if case.key == "trigonometric-poisson" else 1.0)
        return gmsh_disc_mesh(
            args.mesh_size, center=(0.0, 0.0), radius=radius,
            verbosity=args.gmsh_verbosity, algorithm=args.gmsh_algorithm, log_cache=log_cache,
        ), domain
    if domain == "lshape":
        return gmsh_lshape_mesh(
            args.mesh_size, verbosity=args.gmsh_verbosity, algorithm=args.gmsh_algorithm, log_cache=log_cache,
        ), domain
    return gmsh_triangle_mesh(
        args.mesh_size, vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
        verbosity=args.gmsh_verbosity, algorithm=args.gmsh_algorithm, log_cache=log_cache,
    ), "triangle"


def _scale_mode(value: str) -> str:
    return {"off": "none", "on": "left"}.get(str(value).lower(), str(value).lower())


def _optional_int(value) -> str:
    return "default" if value is None else f"{int(value):,d}"


def _print_summary(
        args,
        domain,
        mesh,
        space,
        result,
        report,
        elapsed,
        config_path,
        stabilization_mode,
        tau_value,
) -> None:
    metrics = report.metrics
    solve = result.global_solve_result
    timings = result.timings
    details = timings.details or {}
    sections = [
        (
            "Run / Options",
            [
                ("case", args.case, "s"),
                ("domain", domain, "s"),
                ("backend", args.assembly_backend, "s"),
                ("basis", args.basis, "s"),
                ("trace basis", args.trace_basis, "s"),
                ("scaling", _scale_mode(args.scale_system), "s"),
                ("diffusion stabilization", stabilization_mode, "s"),
                ("tau_d", tau_value, ".6g"),
            ],
        ),
        (
            "Mesh / DOF",
            [
                ("order", args.order, ",d"),
                ("h", mesh.h, ".3e"),
                ("triangles", mesh.num_tri, ",d"),
                ("edges", mesh.num_edg, ",d"),
                ("interior edges", int(mesh.int_edges_inds.size), ",d"),
                ("global dof", int(mesh.int_edges_inds.size * space.layout.edg_dof), ",d"),
                ("volume quad 1d", _optional_int(args.volume_quad_1d), "s"),
            ],
        ),
        (
            "Solver / Error",
            [
                ("config", "embedded" if config_path is None else str(config_path), "s"),
                ("iterations", _optional_int(None if solve is None else solve.iteration_count), "s"),
                ("physical residual", float("nan") if solve is None else solve.physical_relative_residual_norm, ".3e"),
                ("L2", metrics.l2, ".3e"),
                ("Linf", metrics.linf, ".3e"),
                ("avg max", metrics.mean_element_linf, ".3e"),
                ("max element", metrics.max_element, ",d"),
            ],
        ),
        (
            "Timings",
            [
                ("assembly", format_elapsed_percent(timings.assembly, elapsed), "s"),
                ("global solve", format_elapsed_percent(timings.solve, elapsed), "s"),
                ("reconstruction", format_elapsed_percent(timings.reconstruction, elapsed), "s"),
                ("HDG postprocess", format_elapsed_percent(timings.postprocessing, elapsed), "s"),
                ("total measured", f"{elapsed:.3f}s", "s"),
            ],
        ),
    ]
    if args.verbosity >= 2 and details:
        sections.append(("Backend details", [(key, value, ".5f") for key, value in sorted(details.items())]))
    pretty_print_sections(sections, title="HDGFEM CUDA Diffusion-Reaction Solve Summary")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.plot_postprocess_primal:
        args.plot = True
    if args.assembly_backend == "raw-cuda" and args.raw_matrix_format != "csr":
        raise ValueError("the reusable raw-CUDA diffusion solve requires --raw-matrix-format csr")
    if args.assembly_backend == "raw-cuda" and args.plot_postprocess_primal:
        raise ValueError("raw-CUDA diffusion currently does not support HDG postprocessing; use --assembly-backend cupy")
    if args.show_cupy_config:
        from hdgfem.runtime.optional import require_cupy

        require_cupy().show_config()

    started = time.perf_counter()
    case = case_definition_by_key(args.case)
    problem = case.build()
    if problem.diffusion != (1.0, 0.0, 1.0):
        raise NotImplementedError("CUDA diffusion assembly currently supports identity diffusion only")
    mesh, domain = build_mesh(args, case)
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quadrature=args.volume_quadrature,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    from hdgfem.hdg.stabilization import GlobalLengthDiffusion

    stabilization_mode = args.diffusion_stabilization_mode
    if args.tau is not None:
        stabilization_mode = "explicit"
    if stabilization_mode == "explicit":
        if args.tau is None:
            raise ValueError(
                "--diffusion-stabilization-mode explicit requires --tau"
            )
        stabilization = float(args.tau)
        tau_value = stabilization
    else:
        domain_length = (
            "auto"
            if args.diffusion_domain_length == "auto"
            else float(args.diffusion_domain_length)
        )
        stabilization = GlobalLengthDiffusion(
            gamma_d=args.diffusion_stabilization_gamma,
            domain_length=domain_length,
        )
        tau_value = stabilization.resolve(problem.diffusion, space)

    config, config_path = load_amgx_config(
        args.amgx_config,
        default_path=DEFAULT_AMGX_CONFIG_PATH,
        default_config=AMGX_CONFIG,
        solver=args.amgx_solver,
        tolerance=args.amgx_tolerance,
        maxiter=args.amgx_maxiter,
    )
    solver = DiffusionReactionHDGSolver(
        space,
        diffusion=problem.diffusion,
        stabilization=stabilization,
        solver="amgx",
        solver_rtol=args.amgx_tolerance,
        maxiter=args.amgx_maxiter,
        scale_system=_scale_mode(args.scale_system),
        amgx_config=config,
        assembly_backend=args.assembly_backend,
        trace_basis=args.trace_basis,
        raw_matrix_format=args.raw_matrix_format,
        raw_block_size=args.raw_block_size,
        boundary_mode="eliminate",
        hdg_postprocess="primal" if args.plot_postprocess_primal else "none",
        verbose=args.verbosity,
    )
    solver.set_problem(problem.source, problem.reaction, problem.exact)
    result = solver.solve()

    plot_resolution = resolve_field_plot_resolution(
        args.plot_resolution, order=space.order, num_elements=mesh.num_tri,
    )
    report = evaluate_scalar_error(
        result.field,
        problem.exact,
        volume_quad_1d=args.error_volume_quad_1d,
        sample_resolution=plot_resolution,
        include_samples=args.plot,
    )
    post_samples = None
    if args.plot and result.postprocessed_field is not None:
        post_resolution = resolve_postprocessed_plot_resolution(
            plot_resolution, order=space.order, num_elements=mesh.num_tri,
        )
        post_samples = evaluate_scalar_error(
            result.postprocessed_field,
            problem.exact,
            volume_quad_1d=args.error_volume_quad_1d,
            sample_resolution=post_resolution,
            include_samples=True,
        ).samples

    elapsed = time.perf_counter() - started
    _print_summary(
        args,
        domain,
        mesh,
        space,
        result,
        report,
        elapsed,
        config_path,
        stabilization_mode,
        tau_value,
    )
    if args.plot:
        plot_sampled_solution_comparison(
            mesh,
            problem.exact,
            report.samples,
            numerical_resolution=plot_resolution,
            exact_resolution=args.exact_plot_resolution,
            polynomial_order=space.order,
            postprocessed_samples=post_samples,
            title=f"{args.case}, p={space.order}, elements={mesh.num_tri:,}, L2={report.metrics.l2:.2e}",
            show_mesh=not args.hide_mesh,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
