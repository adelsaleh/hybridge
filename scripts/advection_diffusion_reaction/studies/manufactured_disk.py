#!/usr/bin/env python3
"""Solve and plot a steady manufactured ADR problem on the unit disk."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hdgfem import (
    AdvectionDiffusionReactionHDGOptions,
    AdvectionDiffusionReactionHDGSolver,
    DGSpace,
    GlobalLengthDiffusion,
    automatic_domain_length,
    gmsh_disc_mesh,
)
from hdgfem.io.output import format_elapsed_percent, pretty_print_sections, timed_call


from scripts.advection_diffusion_reaction.cases.disk_case import (
    DEFAULT_PECLET, manufactured_adr_disk,
)


@dataclass(frozen=True)
class ManufacturedADRDiskRun:
    """Result and error diagnostics from one manufactured disk solve."""

    result: object
    problem: dict[str, Callable]
    scalar_l2_error: float
    postprocessed_scalar_l2_error: float
    diffusive_flux_l2_error: float
    total_flux_l2_error: float
    postprocessed_total_flux_l2_error: float
    peclet: float
    mesh_seconds: float = 0.0
    space_seconds: float = 0.0



def _verbosity_level(verbosity: int) -> int:
    """Validate and normalize the runner's three verbosity levels."""
    level = int(verbosity)
    if level not in {0, 1, 2}:
        raise ValueError("verbosity must be 0, 1, or 2")
    return level


def run_manufactured_adr_disk(
    *,
    peclet: float = DEFAULT_PECLET,
    mesh_size: float = 0.3,
    order: int = 3,
    basis: str = "dub_orth",
    trace_basis: str = "legacy-lagrange",
    assembly_backend: str = "numba",
    reconstruction_backend: str = "auto",
    postprocessing_backend: str = "auto",
    hdg_postprocess: str = "both",
    flux_postprocess_space: str = "l2_closest",
    solver: str = "pypardiso",
    solver_rtol: float = 1.0e-11,
    maxiter: int | None = None,
    diffusion_stabilization: float | None = None,
    diffusion_stabilization_mode: str = "global-length",
    diffusion_domain_length: float | str | None = 1.0,
    diffusion_stabilization_gamma: float = 1.0,
    diffusion_penalty_constant: float = 1.0,
    gmsh_verbosity: int = 0,
    verbosity: int = 1,
    plot: bool = False,
    plot_resolution: int = 18,
    exact_plot_resolution: int | str | None = "auto",
    show_mesh: bool = True,
) -> ManufacturedADRDiskRun:
    """Build, solve, diagnose, and optionally plot the steady disk problem."""
    verbosity = _verbosity_level(verbosity)
    problem = manufactured_adr_disk(peclet)
    if verbosity >= 2:
        print(
            "stationary ADR setup: "
            f"Pe={peclet:g}, kappa={1.0 / peclet:.6g}, p={order}, "
            f"assembly={assembly_backend}, reconstruction={reconstruction_backend}, "
            f"postprocessing={postprocessing_backend}, solver={solver}",
            flush=True,
        )

    mesh, mesh_seconds = timed_call(
        "generating unit-disk mesh",
        verbosity,
        lambda: gmsh_disc_mesh(
            mesh_size,
            radius=1.0,
            verbosity=gmsh_verbosity,
            log_cache=verbosity >= 1,
        ),
    )
    space_start = time.perf_counter()
    space = DGSpace(mesh, order, basis_type=basis)
    space_seconds = time.perf_counter() - space_start
    if verbosity:
        print(f"building DG/trace spaces ... done in {space_seconds:.5f}s", flush=True)

    normalized_tau_mode = str(diffusion_stabilization_mode).lower().replace("_", "-")
    if normalized_tau_mode not in {"inverse-h", "global-length"}:
        raise ValueError(
            "diffusion_stabilization_mode must be 'inverse-h' or 'global-length'"
        )
    effective_diffusion_stabilization = diffusion_stabilization
    if effective_diffusion_stabilization is None:
        if normalized_tau_mode == "global-length":
            effective_diffusion_stabilization = GlobalLengthDiffusion(
                gamma_d=diffusion_stabilization_gamma,
                domain_length=diffusion_domain_length,
            )
        else:
            effective_diffusion_stabilization = "inverse-h"

    normalized_solver = str(solver).lower().replace("_", "-")
    if assembly_backend == "raw-cuda" and normalized_solver not in {"amgx", "pyamgx"}:
        raise ValueError("assembly_backend='raw-cuda' requires --solver amgx")
    preconditioner = "ilu" if normalized_solver in {"bicgstab", "gmres", "cg", "cgs"} else None
    scale_system = normalized_solver not in {"direct", "pypardiso", "pardiso"}
    options = AdvectionDiffusionReactionHDGOptions(
        diffusion=1.0 / peclet,
        advection_stabilization=None,
        diffusion_stabilization=effective_diffusion_stabilization,
        diffusion_penalty_constant=diffusion_penalty_constant,
        solver=solver,
        preconditioner=preconditioner,
        solver_rtol=solver_rtol,
        maxiter=maxiter,
        scale_system=scale_system,
        assembly_backend=assembly_backend,
        reconstruction_backend=reconstruction_backend,
        postprocessing_backend=postprocessing_backend,
        boundary_mode="eliminate",
        trace_basis=trace_basis,
        raw_matrix_format="csr",
        hdg_postprocess=hdg_postprocess,
        flux_postprocess_space=flux_postprocess_space,
        verbose=verbosity,
    )
    adr_solver = AdvectionDiffusionReactionHDGSolver(
        space,
        source=problem["source"],
        beta=(problem["beta_x"], problem["beta_y"]),
        reaction=problem["reaction"],
        boundary_condition=problem["exact"],
        options=options,
    )
    result, _solve_wall = timed_call(
        "assembling and solving stationary ADR system",
        verbosity,
        adr_solver.solve,
    )

    post_scalar_error = (
        np.nan
        if result.postprocessed_field is None
        else result.postprocessed_field.l2_error(problem["exact"])
    )
    post_flux_error = (
        np.nan
        if result.postprocessed_flux is None
        else result.postprocessed_flux.l2_error(problem["exact_total_flux"])
    )
    diagnostics = ManufacturedADRDiskRun(
        result=result,
        problem=problem,
        scalar_l2_error=result.field.l2_error(problem["exact"]),
        postprocessed_scalar_l2_error=post_scalar_error,
        diffusive_flux_l2_error=result.flux.l2_error(problem["exact_diffusive_flux"]),
        total_flux_l2_error=result.total_flux.l2_error(problem["exact_total_flux"]),
        postprocessed_total_flux_l2_error=post_flux_error,
        peclet=float(peclet),
        mesh_seconds=mesh_seconds,
        space_seconds=space_seconds,
    )
    if plot:
        _plot_manufactured_run(
            diagnostics,
            plot_resolution=plot_resolution,
            exact_plot_resolution=exact_plot_resolution,
            show_mesh=show_mesh,
        )
    return diagnostics


def _plot_manufactured_run(
    run: ManufacturedADRDiskRun,
    *,
    plot_resolution: int,
    exact_plot_resolution: int | str | None,
    show_mesh: bool,
) -> None:
    """Plot raw, postprocessed, exact, and postprocessed-error panels."""
    from hdgfem.diagnostics import evaluate_scalar_error
    from hdgfem.io.comparison import plot_sampled_solution_comparison
    from hdgfem.io.plot import resolve_postprocessed_plot_resolution

    result = run.result
    field = result.field
    mesh = field.space.mesh
    resolution = resolve_postprocessed_plot_resolution(
        plot_resolution,
        order=field.space.order,
        num_elements=mesh.num_tri,
    )
    primary_samples = evaluate_scalar_error(
        field,
        run.problem["exact"],
        sample_resolution=resolution,
        include_samples=True,
    ).samples
    postprocessed_samples = (
        None
        if result.postprocessed_field is None
        else evaluate_scalar_error(
            result.postprocessed_field,
            run.problem["exact"],
            sample_resolution=resolution,
            include_samples=True,
        ).samples
    )
    plot_sampled_solution_comparison(
        mesh,
        run.problem["exact"],
        primary_samples,
        numerical_resolution=resolution,
        exact_resolution=exact_plot_resolution,
        polynomial_order=field.space.order,
        postprocessed_samples=postprocessed_samples,
        title=f"Steady ADR disk, Pe={run.peclet:g}",
        show_mesh=show_mesh,
    )


def _print_run_summary(run: ManufacturedADRDiskRun, args) -> None:
    """Print the diffusion/advection-runner-style structured solve summary."""
    result = run.result
    mesh = result.field.space.mesh
    global_solve = result.global_solve_result
    total_time = result.timings.total + run.mesh_seconds + run.space_seconds
    domain_length = None
    gamma_d = None
    if args.diffusion_stabilization is not None:
        tau_diffusion = f"{args.diffusion_stabilization:g} (constant)"
    elif args.diffusion_stabilization_mode == "global-length":
        domain_length = (
            automatic_domain_length(mesh)
            if args.diffusion_domain_length == "auto"
            else float(args.diffusion_domain_length)
        )
        gamma_d = args.diffusion_stabilization_gamma
        tau_value = gamma_d * (1.0 / args.peclet) / domain_length
        tau_diffusion = f"{tau_value:.6g} = gamma_d*kappa/L_Omega"
    else:
        penalty = args.diffusion_penalty_constant
        tau_diffusion = (
            "(p+1)^2*kappa/h_F"
            if penalty == 1.0
            else f"{penalty:g}*(p+1)^2*kappa/h_F"
        )
    run_items = [
        ("case", "steady-manufactured-disk", "s"),
        ("Pe", args.peclet, ".3g"),
        ("kappa", 1.0 / args.peclet, ".4e"),
        ("p", result.field.space.order, ",d"),
        ("triangles", mesh.num_tri, ",d"),
        ("edges", mesh.num_edg, ",d"),
        ("trace dofs", result.trace.size, ",d"),
    ]
    option_items = [
        ("assembly backend", result.assembly_backend, "s"),
        ("reconstruction backend", result.reconstruction_backend, "s"),
        ("postprocessing backend", result.postprocessing_backend, "s"),
        ("postprocess", args.hdg_postprocess, "s"),
        ("flux postprocess space", args.flux_postprocess_space, "s"),
        ("boundary mode", "eliminate/Dirichlet", "s"),
        ("trace basis", args.trace_basis, "s"),
        ("tau advection", "abs(beta.n)", "s"),
        ("tau diffusion", tau_diffusion, "s"),
    ]
    if domain_length is not None:
        option_items.extend(
            [
                ("diffusion stabilization", "global_length", "s"),
                ("domain length", domain_length, ".6g"),
                ("gamma_d", gamma_d, ".6g"),
                ("min tau_d", tau_value, ".6g"),
                ("max tau_d", tau_value, ".6g"),
            ]
        )
    normalized_solver = str(args.solver).lower()
    if normalized_solver == "bicgstab":
        preconditioner_name = "ilu"
    elif normalized_solver == "amgx":
        preconditioner_name = "MULTICOLOR_DILU (default)"
    else:
        preconditioner_name = "none"
    solver_items = [
        ("solver", args.solver, "s"),
        ("preconditioner", preconditioner_name, "s"),
    ]
    if global_solve is not None:
        solver_items.extend(
            [
                ("iterations", -1 if global_solve.iteration_count is None else global_solve.iteration_count, ",d"),
                (
                    "physical rel residual",
                    np.nan
                    if global_solve.physical_relative_residual_norm is None
                    else global_solve.physical_relative_residual_norm,
                    ".3e",
                ),
            ]
        )
    error_items = [
        ("primal L2 error", run.scalar_l2_error, ".4e"),
        ("post primal L2 error", run.postprocessed_scalar_l2_error, ".4e"),
        ("diffusive flux L2 error", run.diffusive_flux_l2_error, ".4e"),
        ("total flux L2 error", run.total_flux_l2_error, ".4e"),
        ("post total flux L2 error", run.postprocessed_total_flux_l2_error, ".4e"),
    ]
    timing_items = [
        ("mesh (s)", format_elapsed_percent(run.mesh_seconds, total_time), "s"),
        ("space (s)", format_elapsed_percent(run.space_seconds, total_time), "s"),
        ("assembly (s)", format_elapsed_percent(result.timings.assembly, total_time), "s"),
        ("global solve (s)", format_elapsed_percent(result.timings.solve, total_time), "s"),
        ("reconstruct (s)", format_elapsed_percent(result.timings.reconstruction, total_time), "s"),
        ("postprocess (s)", format_elapsed_percent(result.timings.postprocessing, total_time), "s"),
        ("total (s)", total_time, ".5f"),
    ]
    pretty_print_sections(
        [
            ("Run / mesh", run_items),
            ("Options", option_items),
            ("Solver", solver_items),
            ("Errors", error_items),
            ("Timings", timing_items),
        ],
        title="Steady Advection-Diffusion-Reaction Disk Summary",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the manufactured disk driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peclet", type=float, default=DEFAULT_PECLET)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.3)
    parser.add_argument("--order", "-o", type=int, default=3)
    parser.add_argument("--basis", default="dub_orth")
    parser.add_argument(
        "--trace-basis",
        choices=("legacy-lagrange", "legendre-modal"),
        default="legacy-lagrange",
    )
    parser.add_argument(
        "--assembly-backend",
        choices=("numpy", "numba", "raw-cuda"),
        default="numba",
    )
    parser.add_argument(
        "--reconstruction-backend",
        choices=("auto", "numpy", "numba", "raw-cuda"),
        default="auto",
    )
    parser.add_argument(
        "--postprocessing-backend",
        choices=("auto", "numba", "cupy"),
        default="auto",
        help=(
            "Numba supports both flux spaces; CuPy currently supports only "
            "RT_projection and uses host Numba for coupled primal recovery"
        ),
    )
    parser.add_argument(
        "--hdg-postprocess",
        choices=("none", "primal", "flux", "both"),
        default="both",
    )
    parser.add_argument(
        "--flux-postprocess-space",
        choices=(
            "l2_closest",
            "RT_projection",
            "full-p-plus-1",
            "rt-p",
        ),
        default="l2_closest",
        help=(
            "l2_closest full degree-p+1 recovery or the RT_projection "
            "moment reconstruction; legacy spellings remain accepted"
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("pypardiso", "pardiso", "direct", "BICGSTAB", "amgx"),
        default="pypardiso",
        help="default host solver is nonsymmetric oneMKL PARDISO; raw CUDA requires AMGX",
    )
    parser.add_argument("--solver-rtol", type=float, default=1.0e-11)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument(
        "--diffusion-stabilization",
        type=float,
        default=None,
        help=(
            "explicit constant tau_diff; overrides the default "
            "gamma_d*kappa/L_Omega rule"
        ),
    )
    parser.add_argument(
        "--diffusion-stabilization-mode",
        choices=("inverse-h", "global-length"),
        default="global-length",
        help=(
            "automatic diffusion stabilization: mesh/degree-independent "
            "gamma_d*kappa/L_Omega (default) or legacy inverse-h rule"
        ),
    )
    parser.add_argument(
        "--diffusion-domain-length",
        default="1.0",
        help=(
            "positive physical L_Omega (default 1 for the unit disk), or "
            "'auto' for 2*area/boundary-length"
        ),
    )
    parser.add_argument(
        "--diffusion-stabilization-gamma",
        type=float,
        default=1.0,
        help="positive gamma_d multiplier for global-length mode",
    )
    parser.add_argument("--diffusion-penalty-constant", type=float, default=1.0)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument(
        "--verbosity", "-v", type=int, choices=(0, 1, 2), default=1,
        help="0=summary only, 1=phase progress, 2=detailed backend/solver logging",
    )
    parser.add_argument("--quiet", action="store_true", help="equivalent to --verbosity 0")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show raw/postprocessed/exact/error panels",
    )
    parser.add_argument("--plot-resolution", type=int, default=18)
    parser.add_argument(
        "--exact-plot-resolution",
        default="auto",
        help="exact panel resolution: integer, 'auto', or 'same'",
    )
    parser.add_argument("--hide-mesh", action="store_true")
    return parser


def _exact_plot_resolution(value: str) -> int | str:
    """Normalize the exact-panel CLI resolution value."""
    policy = str(value).lower()
    if policy in {"auto", "same"}:
        return policy
    resolution = int(value)
    if resolution < 2:
        raise ValueError("exact plot resolution must be at least 2")
    return resolution


def main(argv: list[str] | None = None) -> int:
    """Run the manufactured problem from the command line."""
    args = build_arg_parser().parse_args(argv)
    verbosity = 0 if args.quiet else args.verbosity
    run = run_manufactured_adr_disk(
        peclet=args.peclet,
        mesh_size=args.mesh_size,
        order=args.order,
        basis=args.basis,
        trace_basis=args.trace_basis,
        assembly_backend=args.assembly_backend,
        reconstruction_backend=args.reconstruction_backend,
        postprocessing_backend=args.postprocessing_backend,
        hdg_postprocess=args.hdg_postprocess,
        flux_postprocess_space=args.flux_postprocess_space,
        solver=args.solver,
        solver_rtol=args.solver_rtol,
        maxiter=args.maxiter,
        diffusion_stabilization=args.diffusion_stabilization,
        diffusion_stabilization_mode=args.diffusion_stabilization_mode,
        diffusion_domain_length=(
            "auto"
            if args.diffusion_domain_length == "auto"
            else float(args.diffusion_domain_length)
        ),
        diffusion_stabilization_gamma=args.diffusion_stabilization_gamma,
        diffusion_penalty_constant=args.diffusion_penalty_constant,
        gmsh_verbosity=args.gmsh_verbosity,
        verbosity=verbosity,
        plot=args.plot,
        plot_resolution=args.plot_resolution,
        exact_plot_resolution=_exact_plot_resolution(args.exact_plot_resolution),
        show_mesh=not args.hide_mesh,
    )
    _print_run_summary(run, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
