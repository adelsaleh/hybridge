#!/usr/bin/env python3
"""Smoke-test the reusable HDGFEM CuPy advection-reaction solver class.

The defaults intentionally mirror the advection-reaction working path in
``configs/amgx/README.md``: p6/ms0.01, ``dub_orth``, volume quadrature 12,
and the production ``BICGSTAB + classical AMG/ILU0 W-cycle`` AMGX config.
This script is separate from ``run_advection_reaction_cuda.py`` because that file is a
tuned standalone benchmark path; here we exercise the public
``AdvectionReactionHDGSolver`` API with ``assembly_backend="cupy"`` and GPU
global solvers selected through class options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem.backends.cupy import require_cupy, require_cupyx_sparse_linalg, require_pyamgx
from hdgfem.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
from hdgfem.io.config import load_amgx_config
from hdgfem.core.space import DGField, DGSpace, VectorDGField
from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver
from scripts.advection_reaction.cases import case_definition_by_key


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "amgx"
DEFAULT_AMGX_CONFIG_PATH = CONFIG_DIR / "adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json"
DEFAULT_CASE = "test2_legacy_gpu3"


def _runtime_available(solver: str) -> tuple[bool, str | None]:
    try:
        require_cupy().cuda.runtime.getDeviceCount()
        if solver == "cupyx":
            require_cupyx_sparse_linalg()
        elif solver == "amgx":
            require_pyamgx()
    except Exception as exc:
        return False, str(exc)
    return True, None


def _selected_solvers(value: str) -> list[str]:
    if value == "both":
        return ["cupyx", "amgx"]
    return [value]


def _build_mesh(args):
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


def _case_definition(case_key: str):
    try:
        return case_definition_by_key(case_key)
    except ValueError:
        if case_key == "test2_legacy_gpu3":
            return case_definition_by_key("test2")
        raise


def _build_problem(args, space: DGSpace):
    case = _case_definition(args.case)
    beta_x, beta_y, reaction, source, exact = case.build()
    source_input = DGField(source, space, name="source_h")
    reaction_input = DGField(reaction, space, name="reaction_h")
    beta_input = VectorDGField((beta_x, beta_y), space, name="beta_h")
    return source_input, beta_input, reaction_input, exact


def _solve_reference(space: DGSpace, source, beta, reaction, exact, args):
    solver = AdvectionReactionHDGSolver(
        space,
        source=source,
        beta=beta,
        reaction=reaction,
        boundary_condition=exact,
        assembly_backend="cupy",
        solver="direct",
        preconditioner=None,
        solver_rtol=args.reference_rtol,
        solver_atol=args.atol,
        maxiter=args.reference_maxiter,
        scale_system=args.scale_system,
        boundary_mode="eliminate",
        trace_ordering=args.trace_ordering,
        trace_ordering_flux_tolerance=args.trace_ordering_flux_tolerance,
        cache_local_solvers=True,
        materialize_host_solution=True,
        verbose=args.verbose,
    )
    return solver.solve(), solver


def _solve_gpu(space: DGSpace, source, beta, reaction, exact, solver_name: str, args):
    options = {
        "assembly_backend": args.assembly_backend,
        "trace_basis": args.trace_basis,
        "raw_local_assembly": args.raw_local_assembly,
        "raw_lu_mode": args.raw_lu_mode,
        "raw_block_size": args.raw_block_size,
        "raw_matrix_format": args.raw_matrix_format,
        "solver": solver_name,
        "preconditioner": None,
        "solver_rtol": args.rtol,
        "solver_atol": args.atol,
        "maxiter": args.maxiter,
        "scale_system": args.scale_system,
        "boundary_mode": "eliminate",
        "trace_ordering": args.trace_ordering,
        "trace_ordering_flux_tolerance": args.trace_ordering_flux_tolerance,
        "cache_local_solvers": True,
        "materialize_host_solution": True,
        "verbose": args.verbose,
    }
    if solver_name == "cupyx":
        options["cupyx_solver"] = args.cupyx_solver
        options["preconditioner"] = args.cupyx_preconditioner
    elif solver_name == "amgx":
        options["amgx_config"], _ = load_amgx_config(
            args.amgx_config,
            solver=args.amgx_solver,
            tolerance=args.amgx_tolerance,
            maxiter=args.maxiter,
            verbose=args.verbose,
        )

    solver = AdvectionReactionHDGSolver(
        space,
        source=source,
        beta=beta,
        reaction=reaction,
        boundary_condition=exact,
        **options,
    )
    return solver.solve(), solver


def _print_result(label: str, result, exact) -> None:
    solve = result.global_solve_result
    l2_error = result.field.l2_error(exact)
    rel = getattr(solve, "solver_relative_residual_norm", None)
    info = getattr(solve, "info", None)
    iterations = getattr(solve, "iteration_count", None)
    rel_text = "n/a" if rel is None else f"{rel:.3e}"
    print(
        f"{label:>9}: info={info}, iterations={iterations}, "
        f"solver_rel={rel_text}, L2={l2_error:.3e}, "
        f"total={result.timings.total:.3f}s",
        flush=True,
    )


def _assert_finite_result(label: str, result, exact) -> None:
    if (
        result.trace is None
        or result.field is None
        or not np.all(np.isfinite(result.trace))
        or not np.all(np.isfinite(result.field.coeffs))
        or not np.isfinite(result.field.l2_error(exact))
    ):
        raise AssertionError(f"{label} class solve produced non-finite values")


def _check_against_reference(label: str, result, reference, exact, args) -> None:
    _assert_finite_result(label, result, exact)
    trace_delta = float(np.linalg.norm(result.trace - reference.trace))
    trace_ref = float(np.linalg.norm(reference.trace))
    trace_rel = trace_delta / max(trace_ref, 1.0)
    coeff_delta = float(np.linalg.norm(result.field.coeffs - reference.field.coeffs))
    coeff_ref = float(np.linalg.norm(reference.field.coeffs))
    coeff_rel = coeff_delta / max(coeff_ref, 1.0)

    print(
        f"{label:>9}: trace_rel_delta={trace_rel:.3e}, "
        f"coeff_rel_delta={coeff_rel:.3e}",
        flush=True,
    )
    if not np.isfinite(trace_rel) or not np.isfinite(coeff_rel):
        raise AssertionError(
            f"{label} class solve produced a non-finite comparison against the configs/amgx/README.md reference run"
        )
    if trace_rel > args.compare_rtol or coeff_rel > args.compare_rtol:
        raise AssertionError(
            f"{label} class solve disagrees with the configs/amgx/README.md reference run: "
            f"trace_rel_delta={trace_rel:.3e}, coeff_rel_delta={coeff_rel:.3e}, "
            f"tolerance={args.compare_rtol:.3e}"
        )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("cupyx", "amgx", "both"), default="amgx")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="cupy")
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default="legacy-lagrange")
    parser.add_argument("--raw-local-assembly", choices=("precomputed", "fused", "split3"), default="fused")
    parser.add_argument("--raw-lu-mode", choices=("safe", "coop"), default="safe")
    parser.add_argument("--raw-block-size", choices=("auto", "1", "32", "64", "128"), default="auto")
    parser.add_argument("--raw-matrix-format", choices=("auto", "coo", "csr", "bsr"), default="auto")
    parser.add_argument("--case", default=DEFAULT_CASE, help="advection case key from the CUDA runner; test2_legacy_gpu3 aliases to test2 when absent")
    parser.add_argument("--order", "-p", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.01)
    parser.add_argument("--mesh-type", "-mt", choices=("rectangle", "structured-rectangle"), default="rectangle")
    parser.add_argument("--nx", type=int, default=128, help="structured rectangle cells in x")
    parser.add_argument("--ny", type=int, default=None, help="structured rectangle cells in y")
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--volume-quad-1d", type=int, default=12)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--gmsh-num-threads", type=int, default=None)
    parser.add_argument("--trace-ordering", choices=("none", "upwind-scc"), default="none")
    parser.add_argument("--trace-ordering-flux-tolerance", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=1.0e-12, help="package post-solve residual check tolerance")
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-14, help="AMGX config tolerance from configs/amgx/README.md")
    parser.add_argument("--reference-rtol", type=float, default=1.0e-13)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--reference-maxiter", type=int, default=None)
    parser.add_argument("--compare-rtol", type=float, default=5.0e-8)
    parser.add_argument("--reference-solver", choices=("none", "direct"), default="none", help="optional reference comparison; use direct only for reduced smoke sizes")
    parser.add_argument("--scale-system", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cupyx-solver", default="bicgstab")
    parser.add_argument("--cupyx-preconditioner", choices=("none", "ilu"), default="none")
    parser.add_argument("--amgx-solver", default="BICGSTAB")
    parser.add_argument("--amgx-config", type=Path, default=DEFAULT_AMGX_CONFIG_PATH)
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args(argv)
    if args.cupyx_preconditioner == "none":
        args.cupyx_preconditioner = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mesh = _build_mesh(args)
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quad_1d=args.volume_quad_1d,
    )
    source, beta, reaction, exact = _build_problem(args, space)

    print(
        "Reusable AdvectionReactionHDGSolver CuPy check from configs/amgx/README.md: "
        f"case={args.case}, p={args.order}, ms={args.mesh_size}, "
        f"mesh={args.mesh_type}, basis={args.basis}, volume_quad_1d={args.volume_quad_1d}, "
        f"elements={mesh.num_tri}, trace_dofs={mesh.num_edg * space.quad_data.edg_dof}",
        flush=True,
    )
    print(
        "  package backend check: "
        f"assembly_backend={args.assembly_backend}, trace_basis={args.trace_basis}, "
        f"raw_matrix_format={args.raw_matrix_format}, "
        f"AMGX tolerance={args.amgx_tolerance:.1e}, package check rtol={args.rtol:.1e}.",
        flush=True,
    )
    reference = None
    if args.reference_solver == "direct":
        reference, reference_solver = _solve_reference(space, source, beta, reaction, exact, args)
        _print_result("reference", reference, exact)
        if reference_solver.result is not reference:
            raise AssertionError("reference class did not cache its result object")
        _assert_finite_result("reference", reference, exact)
    else:
        print("reference: skipped (--reference-solver none)", flush=True)

    ran = 0
    for solver_name in _selected_solvers(args.solver):
        available, reason = _runtime_available(solver_name)
        if not available:
            message = f"{solver_name} runtime is unavailable: {reason}"
            if args.skip_unavailable:
                print(f"{solver_name:>9}: skipped ({message})", flush=True)
                continue
            raise RuntimeError(message)

        result, solver = _solve_gpu(space, source, beta, reaction, exact, solver_name, args)
        if solver.result is not result or solver.global_solve_result is not result.global_solve_result:
            raise AssertionError(f"{solver_name} class did not cache the solve result")
        if result.assembly_backend != args.assembly_backend:
            raise AssertionError(f"{solver_name} did not use the requested {args.assembly_backend} assembly backend")
        _print_result(solver_name, result, exact)
        _assert_finite_result(solver_name, result, exact)
        if reference is not None:
            _check_against_reference(solver_name, result, reference, exact, args)
        ran += 1

    if ran == 0:
        raise RuntimeError("no GPU solver checks ran")
    print("Reusable CuPy solver class check passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
