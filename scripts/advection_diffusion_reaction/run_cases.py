#!/usr/bin/env python3
"""Run one stationary ADR case using the common catalogue and presets."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields, replace
import inspect
import json
import math
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.advection_diffusion_reaction.cases import CASE_DEFINITIONS
from scripts.advection_diffusion_reaction.presets import DEFAULT_PRESET, PRESETS, preset_by_key


def build_arg_parser():
    """Construct the CLI without importing solvers, Gmsh, CUDA or plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", default=DEFAULT_PRESET, choices=sorted(PRESETS))
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--print-preset", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print effective configuration without building coefficients or a mesh")
    parser.add_argument("--case", choices=sorted(CASE_DEFINITIONS))
    parser.add_argument("--case-param", action="append", default=[], metavar="KEY=JSON",
                        help='case parameter, e.g. peclet=20 or level="entry"; plain strings also work')
    for name in ("order", "nx", "ny", "volume-quad-1d", "edge-quad-1d", "maxiter", "plot-resolution"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("mesh-size", "solver-rtol", "solver-atol", "advection-stabilization"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--domain", choices=("auto", "square", "unit-square", "disk", "annulus"))
    parser.add_argument("--basis", choices=("dub_orth", "bernstein", "hier_C0"))
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal"))
    parser.add_argument("--volume-quadrature", choices=("auto", "duffy", "symmetric"))
    parser.add_argument("--assembly-backend", choices=("numpy", "numba", "raw-cuda"))
    parser.add_argument("--reconstruction-backend", choices=("auto", "numpy", "numba", "raw-cuda"))
    parser.add_argument("--solver", choices=("pypardiso", "amgx"))
    parser.add_argument("--diffusion-stabilization", help="global_length, inverse-h, or a positive scalar")
    parser.add_argument("--raw-matrix-format", choices=("coo", "csr", "bsr"))
    parser.add_argument("--raw-block-size", choices=("auto", "1", "32", "64", "128"))
    parser.add_argument("--threads", choices=("16", "all"))
    for name in ("scale-system", "materialize-host-solution", "plot"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None)
    for name in ("verbosity", "gmsh-verbosity"):
        parser.add_argument(f"--{name}", type=int, choices=(0, 1, 2))
    parser.add_argument("--output", type=Path, help="write a JSON result summary")
    return parser


def runtime_config(args):
    """Merge overrides and validate them before any numerical initialization."""
    config = preset_by_key(args.preset)
    updates = {item.name: getattr(args, item.name) for item in fields(config)
               if getattr(args, item.name, None) is not None}
    params = {} if args.case is not None and args.case != config.case else dict(config.case_params)
    for entry in args.case_param:
        key, separator, value = entry.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ValueError("--case-param must be KEY=JSON (plain strings are also accepted)")
        try:
            params[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            params[key.strip()] = value
    updates["case_params"] = params
    if updates.get("raw_block_size", "auto") != "auto":
        updates["raw_block_size"] = int(updates["raw_block_size"])
    tau = updates.get("diffusion_stabilization")
    if tau is not None:
        if tau.replace("-", "_") in {"global_length", "inverse_h"}:
            updates["diffusion_stabilization"] = tau.replace("_", "-")
        else:
            updates["diffusion_stabilization"] = float(tau)
    config = replace(config, **updates)
    definition = CASE_DEFINITIONS[config.case]
    inspect.signature(definition.factory).bind(**(definition.default_params | config.case_params))
    if not 0 <= config.order <= 6:
        raise ValueError("order must be between 0 and 6")
    for key in ("nx", "ny", "volume_quad_1d", "edge_quad_1d", "maxiter", "plot_resolution"):
        value = getattr(config, key)
        if value is not None and value <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("mesh_size", "solver_rtol", "solver_atol", "advection_stabilization"):
        value = getattr(config, key)
        if value is not None and (not math.isfinite(value) or value < 0 or (key == "mesh_size" and value == 0)):
            raise ValueError(f"{key} must be finite and {'positive' if key == 'mesh_size' else 'nonnegative'}")
    if isinstance(config.diffusion_stabilization, float) and (
            not math.isfinite(config.diffusion_stabilization) or config.diffusion_stabilization <= 0):
        raise ValueError("diffusion_stabilization must be finite and positive")
    if config.assembly_backend == "raw-cuda" and config.solver != "amgx":
        raise ValueError("raw-cuda requires --solver amgx (or use a tensor_cuda_* preset)")
    if config.assembly_backend == "raw-cuda" and config.reconstruction_backend not in {"auto", "raw-cuda"}:
        raise ValueError("raw-cuda assembly requires raw-cuda reconstruction")
    if config.assembly_backend != "raw-cuda" and config.reconstruction_backend == "raw-cuda":
        raise ValueError("raw-cuda reconstruction requires raw-cuda assembly")
    if config.domain != "auto" and config.domain != definition.default_domain:
        if (definition.default_domain in {"annulus", "disk"} or config.domain == "annulus"
                or config.case.startswith("stress_")):
            raise ValueError("this case requires its original geometry; use --domain auto")
    if config.plot and config.case.startswith("coefficient_"):
        raise ValueError("comparison plotting requires a manufactured exact solution")
    return config


def configuration_record(config):
    """Include resolved domain and case defaults in an inspectable configuration."""
    definition = CASE_DEFINITIONS[config.case]
    record = asdict(config)
    record["case_params"] = definition.default_params | config.case_params
    record["domain"] = definition.default_domain if config.domain == "auto" else config.domain
    record["hdg_postprocess"] = "none"
    return record


def _build_mesh(config, problem):
    from hdgfem.core.mesh import rectangle_mesh, gmsh_disc_mesh, gmsh_smooth_star_mesh

    domain = problem.domain if config.domain == "auto" else config.domain
    if domain in {"square", "unit-square"}:
        limits = (-1., 1.) if domain == "square" else (0., 1.)
        return rectangle_mesh(config.nx, config.ny, xlim=limits, ylim=limits)
    if domain == "disk":
        return gmsh_disc_mesh(config.mesh_size, radius=1., verbosity=config.gmsh_verbosity,
                              log_cache=bool(config.verbosity))
    return gmsh_smooth_star_mesh(
        config.mesh_size, radius=1., amplitude=.35, mode=9, boundary_points=360,
        hole_radius=problem.metadata["hole_radius"], verbosity=config.gmsh_verbosity,
        log_cache=bool(config.verbosity))


def run_case(config):
    """Build, solve, diagnose and optionally plot one case using package helpers."""
    from hdgfem import DGSpace, AdvectionDiffusionReactionHDGSolver, AdvectionDiffusionReactionHDGOptions
    from hdgfem.io.output import timed_call
    from hdgfem.linalg.pardiso_runtime import pardiso_thread_limit

    problem, case_seconds = timed_call("preparing analytic case", config.verbosity,
        lambda: CASE_DEFINITIONS[config.case].build(**config.case_params))
    mesh, mesh_seconds = timed_call("building mesh", config.verbosity, lambda: _build_mesh(config, problem))
    space, space_seconds = timed_call("building DG space", config.verbosity, lambda: DGSpace(
        mesh, config.order, basis_type=config.basis, volume_quadrature=config.volume_quadrature,
        volume_quad_1d=config.volume_quad_1d, edge_quad_1d=config.edge_quad_1d))
    # The public ADR API accepts callable/field velocity components, not numbers.
    beta = tuple(value if callable(value) else space.constant(value) for value in problem.beta)
    options = AdvectionDiffusionReactionHDGOptions(
        diffusion=problem.diffusion, trace_basis=config.trace_basis,
        diffusion_stabilization=config.diffusion_stabilization,
        advection_stabilization=config.advection_stabilization,
        assembly_backend=config.assembly_backend, reconstruction_backend=config.reconstruction_backend,
        solver=config.solver, solver_rtol=config.solver_rtol, solver_atol=config.solver_atol,
        maxiter=config.maxiter, scale_system=config.scale_system, hdg_postprocess="none",
        raw_matrix_format=config.raw_matrix_format, raw_block_size=config.raw_block_size,
        materialize_host_solution=config.materialize_host_solution, verbose=config.verbosity)
    solver = AdvectionDiffusionReactionHDGSolver(
        space, source=problem.source, beta=beta, reaction=problem.reaction,
        boundary_condition=problem.boundary_condition, options=options)
    context = pardiso_thread_limit(config.threads) if config.solver == "pypardiso" else nullcontext(None)
    with context as actual_threads:
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        result = solver.solve()
        wall, cpu = time.perf_counter()-wall_start, time.process_time()-cpu_start
    report = dict(config=configuration_record(config), problem_metadata=problem.metadata,
                  elements=mesh.num_tri, trace_dofs=int(result.trace.size),
                  matrix_format=result.matrix_format, diffusion_structure=result.diffusion_structure,
                  timings=dict(case=case_seconds, mesh=mesh_seconds, space=space_seconds,
                               wall_solve=wall, **asdict(result.timings)),
                  cpu=dict(mkl_max_threads=actual_threads, solve_cpu_seconds=cpu,
                           solve_wall_seconds=wall, solve_cpu_wall_ratio=cpu/max(wall, 1e-15),
                           parallel_cpu_observed=cpu > 1.05*wall,
                           measurement_scope="complete stationary solve, including preparation/JIT"),
                  scalar_l2_error=None if problem.exact is None else float(result.field.l2_error(problem.exact)),
                  diffusive_flux_l2_error=None if problem.exact_flux is None else float(result.flux.l2_error(problem.exact_flux)))
    linear = result.global_solve_result
    report["relative_residual"] = None if linear is None else linear.physical_relative_residual_norm
    if config.plot:
        from hdgfem.io.plot import plot_solution_comparison
        if problem.exact is None:
            raise ValueError("comparison plotting requires a manufactured exact solution")
        plot_solution_comparison(result.field, problem.exact, resolution=config.plot_resolution,
                                 title=f"ADR: {config.case}, p={config.order}")
    return result, report


def main(argv=None):
    """List, inspect or execute a preset, supporting both module and file invocation."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_cases or args.list_presets:
        catalogue = CASE_DEFINITIONS if args.list_cases else PRESETS
        for key, value in catalogue.items():
            print(f"{key:36s} {value.description}")
        return 0
    try:
        config = runtime_config(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.print_preset or args.dry_run:
        print(json.dumps(configuration_record(config), indent=2, allow_nan=False))
        return 0
    _, report = run_case(config)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    from hdgfem.io.output import pretty_print_sections
    items = [("elements", report["elements"], ",d"), ("trace DOFs", report["trace_dofs"], ",d"),
             ("solve wall (s)", report["timings"]["wall_solve"], ".4g")]
    if report["cpu"]["mkl_max_threads"] is not None:
        items.extend([("MKL thread limit", report["cpu"]["mkl_max_threads"], "d"),
                      ("solve CPU / wall", report["cpu"]["solve_cpu_wall_ratio"], ".3g")])
    for key in ("scalar_l2_error", "diffusive_flux_l2_error", "relative_residual"):
        if report[key] is not None:
            items.append((key.replace("_", " "), report[key], ".4e"))
    pretty_print_sections([("Stationary solve", items)], title=f"ADR: {config.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
