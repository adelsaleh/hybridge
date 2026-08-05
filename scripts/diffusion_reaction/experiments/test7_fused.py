r"""Experimental CLI for the hard-coded fused Numba tensor-diffusion test7 path."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import time
import sys
from argparse import ArgumentParser
from collections.abc import Iterable
from typing import Literal

import numpy as np

from scripts.diffusion_reaction.experiments.test7_fused_backend import (
    assemble_test7_tensor_trace_system_eliminated_numba,
    reconstruct_test7_tensor_local_unknowns_numba,
)
from hdgfem.core.space import DGSpace
from hdgfem.linalg.system import expand_known_dofs, solve_global_system
from hdgfem.solvers.diffusion_reaction import (
    DiffusionReactionResult,
    DiffusionReactionTimings,
    _diffusion_is_identity,
    _format_seconds,
    _timed_call,
    _verbosity_level,
    split_diffusion_unknowns,
)


def _parse_key_value_options(option_strings: Iterable[str] | None) -> dict[str, str]:
    """Parse repeated ``key=value`` CLI options into a dictionary."""
    parsed: dict[str, str] = {}
    if option_strings is None:
        return parsed
    for item in option_strings:
        if "=" not in item:
            raise ValueError(f"option {item!r} must have the form key=value")
        key, value = item.split("=", 1)
        key = key.strip().lstrip("-")
        if not key:
            raise ValueError(f"option {item!r} has an empty key")
        parsed[key] = value.strip()
    return parsed


def _test7_exact_callable(m: int = 1, n: int = 1):
    a = 0.5 * int(m) * np.pi
    b = 0.5 * int(n) * np.pi

    def exact(x, y):
        return np.sin(a * (x + 1.0)) * np.sin(b * (y + 1.0))

    return exact


def _test7_diffusion_components():
    def k11(x, y):
        return 2.0 + x**2

    def k12(x, y):
        return 0.5 * x * y

    def k22(x, y):
        return 3.0 + y**2

    return k11, k12, k22


def solve_test7_tensor_fused_hdg(
        space: DGSpace,
        *,
        m: int = 1,
        n: int = 1,
        stabilization=1.0,
        solver: str | None = "BICGSTAB",
        preconditioner="ilu",
        solver_rtol: float = 1e-13,
        solver_atol: float = 0.0,
        maxiter: int | None = None,
        scale_system: bool = True,
        petsc_preset: str = "cg_gamg",
        petsc_levels: int | None = None,
        petsc_options: dict | None = None,
        petsc_divtol: float = 1e4,
        petsc_monitor: bool = False,
        ilu_drop_tol: float = 1e-10,
        ilu_fill_factor: float = 35,
        ilu_failure: Literal["raise", "none"] = "raise",
        verbose: bool | int = True,
) -> DiffusionReactionResult:
    """Solve hard-coded test7 with the experimental fused tensor Numba kernels."""
    total_start = time.perf_counter()
    verbosity = _verbosity_level(verbose)
    if verbosity:
        print("\n----- DG FEM Diffusion-Reaction Test7 Fused HDG Solve -----")

    effective_scale_system = False if solver is not None and str(solver).lower() == "petsc" else scale_system

    assembly, trace_assembly = _timed_call(
        "assembling reduced test7 tensor trace system (numba fused)",
        verbosity,
        lambda: assemble_test7_tensor_trace_system_eliminated_numba(
            stabilization,
            space,
            m=m,
            n=n,
        ),
        multiline=verbosity >= 2,
    )
    if verbosity >= 2:
        timings = assembly.timings
        print(
            "  numba test7 fused timings: "
            f"prep={timings.get('preparation', 0.0):.5f}s, "
            f"boundary={timings.get('boundary_trace', 0.0):.5f}s, "
            f"reduction={timings.get('reduction_map', 0.0):.5f}s, "
            f"kernel={timings.get('kernel', 0.0):.5f}s, "
            f"rhs={timings.get('rhs_finalization', 0.0):.5f}s",
            flush=True,
        )

    reduction = assembly.reduction
    global_solve_result, solve_time = _timed_call(
        "solving global system",
        verbosity,
        lambda: solve_global_system(
            reduction.rows,
            reduction.cols,
            reduction.data,
            reduction.rhs,
            reduction.rhs.size,
            solver=solver,
            preconditioner=preconditioner,
            rtol=solver_rtol,
            atol=solver_atol,
            maxiter=maxiter,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            ilu_failure=ilu_failure,
            petsc_preset=petsc_preset,
            petsc_levels=petsc_levels,
            petsc_options=petsc_options,
            petsc_divtol=petsc_divtol,
            petsc_monitor=petsc_monitor,
            scale_system=effective_scale_system,
            scale_matrix_in_place=effective_scale_system,
            raise_on_nonconvergence=True,
            verbose=verbosity,
        ),
        multiline=verbosity >= 1,
    )
    trace = expand_known_dofs(global_solve_result.x, reduction)

    def reconstruct():
        unknowns = reconstruct_test7_tensor_local_unknowns_numba(
            trace,
            stabilization,
            space,
            m=m,
            n=n,
        )
        field, flux = split_diffusion_unknowns(unknowns, space)
        return unknowns, field, flux

    (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)

    timings = DiffusionReactionTimings(
        preparation=0.0,
        local_solver=0.0,
        element_boundary=0.0,
        trace_assembly=trace_assembly,
        solve=solve_time,
        reconstruction=reconstruction,
        total=time.perf_counter() - total_start,
    )
    return DiffusionReactionResult(
        field=field,
        flux=flux,
        trace=trace,
        timings=timings,
        local_unknowns=local_unknowns,
        matrix_rows=assembly.trace_system.rows,
        matrix_cols=assembly.trace_system.cols,
        matrix_data=assembly.trace_system.data,
        rhs=assembly.trace_system.rhs,
        solve_matrix_rows=reduction.rows,
        solve_matrix_cols=reduction.cols,
        solve_matrix_data=reduction.data,
        solve_rhs=reduction.rhs,
        boundary_trace=assembly.trace_system.boundary_trace,
        reduction=reduction,
        local_solver=None,
        element_boundary_mats=None,
        boundary_mode="eliminate",
        scale_system=effective_scale_system,
        assembly_backend="numba",
        global_solve_result=global_solve_result,
    )


def _summarize_solve(result: DiffusionReactionResult, exact, *, mesh, space, tau, args) -> float:
    from hdgfem.io.output import pretty_print_ncol

    l2_error = result.field.l2_error(exact)
    numerical_values = result.field.values()
    points = space.mapped_quads()
    exact_values = exact(points[:, :, 0], points[:, :, 1])
    abs_error = np.abs(numerical_values - exact_values)
    linfty_error = float(np.max(abs_error))
    element_max_error = np.max(abs_error, axis=1)
    avg_error = float(np.average(element_max_error))
    max_error_element = int(np.argmax(element_max_error))
    global_solve = result.global_solve_result
    diffusion = _test7_diffusion_components()

    items = [
        ("p", space.order, ",d"),
        ("#triangles", mesh.num_tri, ",d"),
        ("# edges", mesh.num_edg, ",d"),
        ("#global_dof", result.trace.size, ",d"),
        ("tau", tau, ".3e"),
        ("diffusion", "identity" if _diffusion_is_identity(diffusion) else "tensor", "s"),
        ("h^p", mesh.h ** (space.order + 1), ".4e"),
        ("L2 error", l2_error, ".4e"),
        ("Linf error", linfty_error, ".4e"),
        ("avg error", avg_error, ".4e"),
        ("max_err at el", max_error_element, "d"),
        ("setup time(s)", result.timings.assembly, "1.1f"),
        ("glb_solve time(s)", result.timings.solve, "1.1f"),
        ("recons time(s)", result.timings.reconstruction, "1.1f"),
        ("tot time(s)", result.timings.total, "1.1f"),
        ("solver", args.solver, "s"),
        ("preconditioner", "petsc" if str(args.solver).lower() == "petsc" else args.preconditioner, "s"),
        ("scaling", "left" if result.scale_system else "none", "s"),
        ("assembly backend", "numba", "s"),
        ("local backend", "fused", "s"),
        ("boundary mode", result.boundary_mode, "s"),
    ]
    if str(args.solver).lower() == "petsc":
        items.extend(
            [
                ("PETSc preset", args.petsc_preset, "s"),
                ("PETSc levels", -1 if args.petsc_levels is None else args.petsc_levels, ",d"),
            ]
        )
    if global_solve is not None:
        free_trace_relative_residual = global_solve.diagnostic_relative_residual_norm
        if free_trace_relative_residual is None:
            free_trace_relative_residual = global_solve.solver_relative_residual_norm
        items.extend(
            [
                ("iterations", -1 if global_solve.iteration_count is None else global_solve.iteration_count, ",d"),
                (
                    "solver rel res",
                    np.nan if global_solve.solver_relative_residual_norm is None else global_solve.solver_relative_residual_norm,
                    ".3e",
                ),
                (
                    "free trace rel res",
                    np.nan if free_trace_relative_residual is None else free_trace_relative_residual,
                    ".3e",
                ),
                (
                    "prec time(s)",
                    0.0 if global_solve.preconditioner_elapsed_seconds is None else global_solve.preconditioner_elapsed_seconds,
                    ".3f",
                ),
                (
                    "Krylov time(s)",
                    0.0 if global_solve.solve_elapsed_seconds is None else global_solve.solve_elapsed_seconds,
                    ".3f",
                ),
            ]
        )
    pretty_print_ncol(items, ncols=3, title="Diffusion-Reaction Test7 Fused Solve Summary")
    return l2_error


def _main() -> None:
    """Run the experimental hard-coded test7 fused tensor solve."""
    from hdgfem.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
    from hdgfem.io.plot import plot_solution_comparison

    parser = ArgumentParser(description="Run the experimental fused Numba tensor-diffusion test7 solver.")
    parser.add_argument("--order", "-p", type=int, default=2, help="uniform DG polynomial order")
    parser.add_argument("--m", type=int, default=1, help="test7 x-frequency multiplier")
    parser.add_argument("--n", type=int, default=1, help="test7 y-frequency multiplier")
    parser.add_argument(
        "--domain",
        default="auto",
        choices=("auto", "rectangle", "structured-rectangle"),
        help="mesh domain; auto uses rectangle",
    )
    parser.add_argument("--mesh-size", "--lc", type=float, default=0.35, help="Gmsh target mesh size")
    parser.add_argument("--nx", type=int, default=8, help="structured rectangle cells in x")
    parser.add_argument("--ny", type=int, default=None, help="structured rectangle cells in y")
    parser.add_argument("--gmsh-verbosity", type=int, default=0, help="Gmsh verbosity level")
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--volume-quad-1d", type=int, default=None, help="1D point count for collapsed volume quadrature")
    parser.add_argument("--edge-quad-1d", type=int, default=None, help="1D point count for edge quadrature")
    parser.add_argument("--tau", type=float, default=1.0, help="constant HDG stabilization")
    parser.add_argument("--solver", default="BICGSTAB", help="global trace solver; use direct for sparse direct or petsc for PETSc")
    parser.add_argument("--petsc", dest="solver", action="store_const", const="petsc", help="shortcut for --solver petsc")
    parser.add_argument(
        "--preconditioner",
        default="ilu",
        choices=("ilu", "jacobi", "none"),
        help="global trace preconditioner",
    )
    parser.add_argument("--solver-rtol", type=float, default=1e-13)
    parser.add_argument("--solver-atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument(
        "--petsc-preset",
        default="cg_gamg",
        choices=("cg_ilu", "cg_icc", "cg_hypre", "cg_gamg", "lu", "mumps_lu"),
        help="PETSc KSP/PC preset used when --solver petsc",
    )
    parser.add_argument("--petsc-levels", type=int, default=None, help="PETSc ILU/ICC fill levels or GAMG levels")
    parser.add_argument("--petsc-divtol", type=float, default=1e4, help="PETSc KSP divergence tolerance")
    parser.add_argument("--petsc-monitor", action="store_true", help="print PETSc residual monitor output")
    parser.add_argument(
        "--petsc-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="extra PETSc option without leading dash; repeatable",
    )
    parser.add_argument("--scale-system", dest="scale_system", action="store_true", default=True)
    parser.add_argument("--no-scale-system", dest="scale_system", action="store_false")
    parser.add_argument("--ilu-drop-tol", type=float, default=1e-10, help="ILU drop tolerance")
    parser.add_argument("--ilu-fill-factor", type=float, default=35.0, help="ILU fill factor")
    parser.add_argument("--ilu-failure", default="none", choices=("raise", "none"))
    parser.add_argument("--verbosity", "-v", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-resolution", type=int, default=20)
    parser.add_argument("--exact-plot-resolution", type=int, default=None)
    parser.add_argument("--hide-mesh", action="store_true")
    args = parser.parse_args()

    petsc_option_flags = (
        "--petsc-preset",
        "--petsc-levels",
        "--petsc-divtol",
        "--petsc-monitor",
        "--petsc-option",
    )
    used_petsc_options = any(
        arg == flag or arg.startswith(f"{flag}=")
        for arg in sys.argv[1:]
        for flag in petsc_option_flags
    )
    if used_petsc_options and str(args.solver).lower() != "petsc":
        parser.error("PETSc options were provided, but PETSc was not selected. Add --solver petsc or --petsc.")

    verbosity = 0 if args.quiet else max(0, int(args.verbosity))
    if verbosity >0:
        import numba as nb
        print("nb threads:", nb.get_num_threads())

    def build_mesh():
        domain = "rectangle" if args.domain == "auto" else args.domain
        if domain == "structured-rectangle":
            return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        return gmsh_rectangle_mesh(args.mesh_size, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), verbosity=args.gmsh_verbosity)

    mesh, _ = _timed_call(f"generating {args.domain} mesh", verbosity, build_mesh)
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    petsc_options = _parse_key_value_options(args.petsc_option)
    result = solve_test7_tensor_fused_hdg(
        space,
        m=args.m,
        n=args.n,
        stabilization=args.tau,
        solver=args.solver,
        preconditioner=None if args.preconditioner == "none" else args.preconditioner,
        solver_rtol=args.solver_rtol,
        solver_atol=args.solver_atol,
        maxiter=args.maxiter,
        scale_system=args.scale_system,
        petsc_preset=args.petsc_preset,
        petsc_levels=args.petsc_levels,
        petsc_options=petsc_options,
        petsc_divtol=args.petsc_divtol,
        petsc_monitor=args.petsc_monitor,
        ilu_drop_tol=args.ilu_drop_tol,
        ilu_fill_factor=args.ilu_fill_factor,
        ilu_failure=args.ilu_failure,
        verbose=verbosity,
    )

    exact = _test7_exact_callable(m=args.m, n=args.n)
    l2_error = _summarize_solve(result, exact, mesh=mesh, space=space, tau=args.tau, args=args)

    if args.plot:
        title = f"test7 fused tensor, p={space.order}, elements={mesh.num_tri}, L2={l2_error:.2e}"
        plot_solution_comparison(
            result.field,
            exact,
            resolution=args.plot_resolution,
            exact_resolution="auto" if args.exact_plot_resolution is None else args.exact_plot_resolution,
            title=title,
            show_mesh=not args.hide_mesh,
        )


__all__ = [
    "solve_test7_tensor_fused_hdg",
]


if __name__ == "__main__":
    _main()
