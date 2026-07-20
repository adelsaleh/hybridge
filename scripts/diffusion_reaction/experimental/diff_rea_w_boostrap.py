r"""Bootstrap-enabled HDG solver for scalar diffusion-reaction problems.

This module layers a coarse same-mesh initial guess on top of
:mod:`hdgfem.solvers.diff_rea`.  The base module owns the actual HDG assembly and solve;
this module only builds a lower-order trace solution, degree-elevates it to the
target trace space, and passes it as ``initial_guess`` to the normal solver.

The file name follows the requested ``diff_rea_w_boostrap`` spelling.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from math import comb
from typing import Literal

import numpy as np

from hdgfem.solvers.diff_rea import (
    DiffusionReactionResult,
    LocalSolverBackend,
    ReturnKey,
    _timed_call,
    _verbosity_level,
    impose_boundary_trace_on_guess,
    solve_diffusion_reaction_hdg as _solve_plain_diffusion_reaction_hdg,
)
from hdgfem.core.space import DGSpace


@dataclass(frozen=True)
class DiffusionReactionBootstrapResult(DiffusionReactionResult):
    """Result returned by the bootstrap-enabled diffusion-reaction wrapper."""

    bootstrap_order: int | None = None


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


def bernstein_degree_elevation_matrix(source_order: int, target_order: int) -> np.ndarray:
    r"""Return the exact 1D Bernstein degree-elevation matrix.

    The matrix :math:`E \in \mathbb{R}^{(p+1)\times(P+1)}` satisfies
    :math:`c_P = c_p E`, where ``source_order`` is :math:`p` and
    ``target_order`` is :math:`P`.  HDG trace unknowns use a 1D Bernstein edge
    basis, so this gives an exact same-mesh trace prolongation from a coarse
    bootstrap solve to the target trace space.
    """
    source_order = int(source_order)
    target_order = int(target_order)
    if source_order < 0 or target_order < 0:
        raise ValueError("source_order and target_order must be nonnegative")
    if source_order > target_order:
        raise ValueError("source_order must be <= target_order for degree elevation")
    if source_order == target_order:
        return np.eye(source_order + 1, dtype=np.float64)

    degree_gap = target_order - source_order
    elevation = np.zeros((source_order + 1, target_order + 1), dtype=np.float64)
    for i in range(source_order + 1):
        start = i
        stop = i + degree_gap
        for j in range(start, stop + 1):
            elevation[i, j] = (
                comb(source_order, i)
                * comb(degree_gap, j - i)
                / comb(target_order, j)
            )
    return np.ascontiguousarray(elevation)


def prolong_trace_coefficients(trace: np.ndarray, source_order: int, target_order: int) -> np.ndarray:
    """Degree-elevate global trace coefficients from ``source_order`` to ``target_order``.

    The operation is applied independently on each global mesh edge.  It is
    exact for the lower-order trace polynomial because both spaces use the same
    Bernstein edge basis.
    """
    trace = np.asarray(trace, dtype=np.float64)
    source_dof = int(source_order) + 1
    if trace.ndim != 1:
        raise ValueError("trace must be a one-dimensional global trace vector")
    if source_dof <= 0 or trace.size % source_dof != 0:
        raise ValueError("trace size is incompatible with source_order")
    elevation = bernstein_degree_elevation_matrix(source_order, target_order)
    low_coeffs = trace.reshape(trace.size // source_dof, source_dof)
    return np.ascontiguousarray((low_coeffs @ elevation).ravel())


def bootstrap_trace_initial_guess(
        source,
        reaction,
        boundary_condition: Callable,
        target_space: DGSpace,
        *,
        diffusion=1.0,
        bootstrap_order: int = 1,
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
        local_solver_backend: LocalSolverBackend = "numpy",
        boundary_penalty: float = 1e20,
        boundary_mode: Literal["penalty", "eliminate"] = "penalty",
        ilu_drop_tol: float = 1e-10,
        ilu_fill_factor: float = 35,
        ilu_failure: Literal["raise", "none"] = "raise",
        verbose: bool | int = False,
) -> np.ndarray:
    """Build a same-mesh coarse-order trace initial guess for ``target_space``.

    The coarse trace is solved in
    :class:`DGSpace(target_space.mesh, bootstrap_order)`, then elevated
    edge-by-edge to the target order.  The returned vector is a global trace
    coefficient vector of length ``target_space.mesh.num_edg * (target_order+1)``.
    Boundary coefficients are corrected by the target solve before use.
    """
    bootstrap_order = int(bootstrap_order)
    if bootstrap_order < 0:
        raise ValueError("bootstrap_order must be nonnegative")
    if bootstrap_order >= target_space.order:
        raise ValueError("bootstrap_order must be smaller than the target space order")

    bootstrap_space = DGSpace(
        target_space.mesh,
        bootstrap_order,
        basis_type=target_space.quad_data.basis_type,
        name=f"{target_space.name}_p{bootstrap_order}_bootstrap",
    )
    bootstrap_result = _solve_plain_diffusion_reaction_hdg(
        source,
        reaction,
        boundary_condition,
        bootstrap_space,
        diffusion=diffusion,
        stabilization=stabilization,
        solver=solver,
        preconditioner=preconditioner,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        maxiter=maxiter,
        scale_system=scale_system,
        petsc_preset=petsc_preset,
        petsc_levels=petsc_levels,
        petsc_options=petsc_options,
        petsc_divtol=petsc_divtol,
        petsc_monitor=petsc_monitor,
        local_solver_backend=local_solver_backend,
        boundary_penalty=boundary_penalty,
        boundary_mode=boundary_mode,
        ilu_drop_tol=ilu_drop_tol,
        ilu_fill_factor=ilu_fill_factor,
        ilu_failure=ilu_failure,
        initial_guess=None,
        verbose=verbose,
    )
    return prolong_trace_coefficients(
        bootstrap_result.trace,
        bootstrap_order,
        target_space.order,
    )


def _with_bootstrap_metadata(
        result: DiffusionReactionResult,
        *,
        initial_guess: np.ndarray | None,
        bootstrap_order: int | None,
        initial_guess_time: float,
) -> DiffusionReactionResult:
    """Return ``result`` with bootstrap timing/metadata attached."""
    if initial_guess_time == 0.0 and bootstrap_order is None:
        return result
    timings = replace(
        result.timings,
        initial_guess=initial_guess_time,
        total=result.timings.total + initial_guess_time,
    )
    result_values = {field.name: getattr(result, field.name) for field in fields(DiffusionReactionResult)}
    result_values["timings"] = timings
    result_values["initial_guess"] = initial_guess if initial_guess is not None else result.initial_guess
    return DiffusionReactionBootstrapResult(**result_values, bootstrap_order=bootstrap_order)


def solve_diffusion_reaction_hdg(
        source,
        reaction,
        boundary_condition: Callable,
        space: DGSpace,
        *,
        diffusion=1.0,
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
        initial_guess: np.ndarray | None = None,
        bootstrap_order: int | None = None,
        bootstrap_solver: str | None = None,
        bootstrap_preconditioner="ilu",
        bootstrap_maxiter: int | None = None,
        bootstrap_ilu_drop_tol: float = 1e-10,
        bootstrap_ilu_fill_factor: float = 35,
        bootstrap_ilu_failure: Literal["raise", "none"] = "raise",
        local_solver_backend: LocalSolverBackend = "numpy",
        boundary_penalty: float = 1e20,
        boundary_mode: Literal["penalty", "eliminate"] = "penalty",
        verbose: bool | int = True,
        return_: Iterable[ReturnKey] = ("result",),
):
    r"""Solve :math:`-\nabla\cdot(\kappa\nabla u) + r u=f` with an optional bootstrap trace guess.

    This wrapper keeps the normal solver in :mod:`hdgfem.solvers.diff_rea` untouched.  If
    ``bootstrap_order`` is provided, it first solves the same problem on the
    same mesh with a lower polynomial order, degree-elevates the trace, and uses
    the result as ``initial_guess`` for the target-order solve.
    """
    if initial_guess is not None and bootstrap_order is not None:
        raise ValueError("provide either initial_guess or bootstrap_order, not both")

    verbosity = _verbosity_level(verbose)
    initial_guess_time = 0.0
    bootstrap_guess = None
    if initial_guess is None and bootstrap_order is not None:
        effective_scale_system = False if solver is not None and str(solver).lower() == "petsc" else scale_system
        bootstrap_guess, initial_guess_time = _timed_call(
            f"building p={int(bootstrap_order)} bootstrap initial guess",
            verbosity,
            lambda: bootstrap_trace_initial_guess(
                source,
                reaction,
                boundary_condition,
                space,
                diffusion=diffusion,
                bootstrap_order=int(bootstrap_order),
                stabilization=stabilization,
                solver=solver if bootstrap_solver is None else bootstrap_solver,
                preconditioner=bootstrap_preconditioner,
                solver_rtol=solver_rtol,
                solver_atol=solver_atol,
                maxiter=bootstrap_maxiter,
                scale_system=effective_scale_system,
                petsc_preset=petsc_preset,
                petsc_levels=petsc_levels,
                petsc_options=petsc_options,
                petsc_divtol=petsc_divtol,
                petsc_monitor=petsc_monitor,
                local_solver_backend=local_solver_backend,
                boundary_penalty=boundary_penalty,
                ilu_drop_tol=bootstrap_ilu_drop_tol,
                ilu_fill_factor=bootstrap_ilu_fill_factor,
                ilu_failure=bootstrap_ilu_failure,
                verbose=max(0, verbosity - 1),
            ),
            multiline=verbosity >= 2,
        )
        initial_guess = bootstrap_guess

    want = tuple(return_)
    result = _solve_plain_diffusion_reaction_hdg(
        source,
        reaction,
        boundary_condition,
        space,
        diffusion=diffusion,
        stabilization=stabilization,
        solver=solver,
        preconditioner=preconditioner,
        solver_rtol=solver_rtol,
        solver_atol=solver_atol,
        maxiter=maxiter,
        scale_system=scale_system,
        petsc_preset=petsc_preset,
        petsc_levels=petsc_levels,
        petsc_options=petsc_options,
        petsc_divtol=petsc_divtol,
        petsc_monitor=petsc_monitor,
        ilu_drop_tol=ilu_drop_tol,
        ilu_fill_factor=ilu_fill_factor,
        ilu_failure=ilu_failure,
        initial_guess=initial_guess,
        local_solver_backend=local_solver_backend,
        boundary_penalty=boundary_penalty,
        boundary_mode=boundary_mode,
        verbose=verbose,
        return_=want,
    )

    if want != ("result",):
        if "result" not in want:
            return result
        output = list(result)
        result_index = want.index("result")
        output[result_index] = _with_bootstrap_metadata(
            output[result_index],
            initial_guess=output[result_index].initial_guess,
            bootstrap_order=bootstrap_order,
            initial_guess_time=initial_guess_time,
        )
        return tuple(output)

    return _with_bootstrap_metadata(
        result,
        initial_guess=result.initial_guess,
        bootstrap_order=bootstrap_order,
        initial_guess_time=initial_guess_time,
    )


def _parse_bootstrap_order(value: str | int | None) -> int | None:
    """Parse ``--bootstrap-order`` while accepting ``none`` to disable it."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "off", "false"}:
        return None
    return int(value)


def _as_optional_preconditioner(value: str | None):
    """Map CLI ``none`` spelling to the solver API's ``None`` preconditioner."""
    if value is None:
        return None
    return None if value == "none" else value


def _summarize_solve(result: DiffusionReactionResult, exact: Callable, *, args, mesh, space) -> float:
    """Print the same compact solve summary as the plain diffusion CLI."""
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
    items = [
        ("p", space.order, ",d"),
        ("#triangles", mesh.num_tri, ",d"),
        ("# edges", mesh.num_edg, ",d"),
        ("#global_dof", result.trace.size, ",d"),
        ("tau", args.tau, ".3e"),
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
        ("local backend", args.local_backend, "s"),
        ("boundary mode", result.boundary_mode, "s"),
    ]
    if str(args.solver).lower() == "petsc":
        items.extend(
            [
                ("PETSc preset", args.petsc_preset, "s"),
                ("PETSc levels", -1 if args.petsc_levels is None else args.petsc_levels, ",d"),
            ]
        )
    bootstrap_order = getattr(result, "bootstrap_order", None)
    if bootstrap_order is not None:
        items.extend(
            [
                ("bootstrap p", bootstrap_order, ",d"),
                ("init guess time(s)", result.timings.initial_guess, "1.1f"),
            ]
        )
    if global_solve is not None:
        free_trace_relative_residual = global_solve.diagnostic_relative_residual_norm
        if free_trace_relative_residual is None and result.boundary_mode == "eliminate":
            free_trace_relative_residual = global_solve.solver_relative_residual_norm
        items.extend(
            [
                ("iterations", -1 if global_solve.iteration_count is None else global_solve.iteration_count, ",d"),
                (
                    "solver rel res",
                    np.nan if global_solve.solver_relative_residual_norm is None else global_solve.solver_relative_residual_norm,
                    ".3e",
                ),
                ("free trace rel res", np.nan if free_trace_relative_residual is None else free_trace_relative_residual, ".3e"),
                (
                    "prec time(s)",
                    0.0 if global_solve.preconditioner_elapsed_seconds is None else global_solve.preconditioner_elapsed_seconds,
                    ".3f",
                ),
                ("Krylov time(s)", 0.0 if global_solve.solve_elapsed_seconds is None else global_solve.solve_elapsed_seconds, ".3f"),
            ]
        )
    pretty_print_ncol(items, ncols=3, title="Diffusion-Reaction Bootstrap Solve Summary")
    return l2_error


diff_rea_hdg_solve = solve_diffusion_reaction_hdg


def _main() -> None:
    """Run the bootstrap-enabled diffusion-reaction CLI."""
    from argparse import ArgumentParser

    from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_lshape_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh
    from hdgfem.io.plot import plot_solution_comparison
    from scripts.diffusion_reaction.diff_rea_cases import case_by_legacy_id

    parser = ArgumentParser(description="Run the bootstrap-enabled hdgfem diffusion-reaction HDG solver.")
    parser.add_argument("--order", "-p", type=int, default=2, help="uniform DG polynomial order")
    parser.add_argument("--test", type=int, default=0, choices=(0, 2, 3, 5, 6), help="manufactured legacy test id")
    parser.add_argument(
        "--domain",
        default="auto",
        choices=("auto", "rectangle", "unit-rectangle", "disc", "triangle", "lshape", "structured-rectangle"),
    )
    parser.add_argument("--mesh-size", "--lc", type=float, default=0.35, help="Gmsh target mesh size")
    parser.add_argument("--nx", type=int, default=8, help="structured rectangle cells in x")
    parser.add_argument("--ny", type=int, default=None, help="structured rectangle cells in y")
    parser.add_argument("--gmsh-verbosity", type=int, default=0, help="Gmsh verbosity level")
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--tau", type=float, default=1.0, help="constant HDG stabilization")
    parser.add_argument("--local-backend", default="numpy", choices=("numpy", "numba"), help="local solver backend")
    parser.add_argument("--solver", default="BICGSTAB", help="global trace solver; use direct for sparse direct")
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
        help="extra PETSc option without leading dash; repeatable, for example pc_gamg_threshold=0.02",
    )
    parser.add_argument(
        "--scale-system",
        dest="scale_system",
        action="store_true",
        default=True,
        help="use legacy left diagonal row scaling for iterative solves",
    )
    parser.add_argument(
        "--no-scale-system",
        dest="scale_system",
        action="store_false",
        help="disable left scaling; required for CG/MINRES symmetry",
    )
    parser.add_argument("--ilu-drop-tol", type=float, default=1e-10, help="ILU drop tolerance")
    parser.add_argument("--ilu-fill-factor", type=float, default=35.0, help="ILU fill factor")
    parser.add_argument(
        "--ilu-failure",
        default="none",
        choices=("raise", "none"),
        help="behavior if main ILU factorization fails",
    )
    parser.add_argument(
        "--bootstrap-order",
        type=str,
        default="1",
        help="coarse same-mesh order used to build a trace initial guess; use none to disable",
    )
    parser.add_argument("--bootstrap-solver", default=None, help="bootstrap trace solver; defaults to the main solver")
    parser.add_argument(
        "--bootstrap-preconditioner",
        default="ilu",
        choices=("ilu", "jacobi", "none"),
        help="bootstrap trace preconditioner",
    )
    parser.add_argument("--bootstrap-maxiter", type=int, default=None)
    parser.add_argument("--bootstrap-ilu-drop-tol", type=float, default=1e-10)
    parser.add_argument("--bootstrap-ilu-fill-factor", type=float, default=35.0)
    parser.add_argument(
        "--bootstrap-ilu-failure",
        default="raise",
        choices=("raise", "none"),
        help="behavior if bootstrap ILU factorization fails",
    )
    parser.add_argument(
        "--boundary-mode",
        default="penalty",
        choices=("penalty", "eliminate"),
        help="Dirichlet trace treatment: legacy penalty rows or reduced known-dof elimination",
    )
    parser.add_argument("--verbosity", "-v", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-resolution", type=int, default=20)
    parser.add_argument(
        "--exact-plot-resolution",
        type=int,
        default=None,
        help="exact-solution panel resolution; default uses an automatic dense reference sampling",
    )
    parser.add_argument("--hide-mesh", action="store_true")
    args = parser.parse_args()

    verbosity = 0 if args.quiet else max(0, int(args.verbosity))
    args.bootstrap_order = _parse_bootstrap_order(args.bootstrap_order)

    def build_mesh():
        domain = args.domain
        if domain == "auto":
            if args.test == 3:
                domain = "disc"
            elif args.test == 6:
                domain = "lshape"
            else:
                domain = "rectangle"
        if domain == "structured-rectangle":
            return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        if domain == "unit-rectangle":
            return gmsh_rectangle_mesh(args.mesh_size, xlim=(0.0, 1.0), ylim=(0.0, 1.0), verbosity=args.gmsh_verbosity)
        if domain == "rectangle":
            return gmsh_rectangle_mesh(args.mesh_size, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), verbosity=args.gmsh_verbosity)
        if domain == "disc":
            radius = 5.0 if args.test == 3 and args.domain == "auto" else 1.0
            return gmsh_disc_mesh(args.mesh_size, center=(0.0, 0.0), radius=radius, verbosity=args.gmsh_verbosity)
        if domain == "lshape":
            return gmsh_lshape_mesh(
                args.mesh_size,
                corner_mesh_size=args.mesh_size / 10.0 if args.domain == "auto" else None,
                corner_refine_radius=0.1 if args.domain == "auto" else 0.4,
                verbosity=args.gmsh_verbosity,
            )
        return gmsh_triangle_mesh(
            args.mesh_size,
            vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
            verbosity=args.gmsh_verbosity,
        )

    mesh, _ = _timed_call(f"generating {args.domain} mesh", verbosity, build_mesh)
    space = DGSpace(mesh, args.order, basis_type=args.basis)
    diffusion, reaction, source, exact = case_by_legacy_id(args.test)
    petsc_options = _parse_key_value_options(args.petsc_option)
    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=args.tau,
        solver=args.solver,
        preconditioner=_as_optional_preconditioner(args.preconditioner),
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
        bootstrap_order=args.bootstrap_order,
        bootstrap_solver=args.bootstrap_solver,
        bootstrap_preconditioner=_as_optional_preconditioner(args.bootstrap_preconditioner),
        bootstrap_maxiter=args.bootstrap_maxiter,
        bootstrap_ilu_drop_tol=args.bootstrap_ilu_drop_tol,
        bootstrap_ilu_fill_factor=args.bootstrap_ilu_fill_factor,
        bootstrap_ilu_failure=args.bootstrap_ilu_failure,
        local_solver_backend=args.local_backend,
        boundary_mode=args.boundary_mode,
        verbose=verbosity,
    )

    l2_error = _summarize_solve(result, exact, args=args, mesh=mesh, space=space)

    if args.plot:
        title = f"diff bootstrap test {args.test}, p={space.order}, elements={mesh.num_tri}, L2={l2_error:.2e}"
        plot_solution_comparison(
            result.field,
            exact,
            resolution=args.plot_resolution,
            exact_resolution="auto" if args.exact_plot_resolution is None else args.exact_plot_resolution,
            title=title,
            show_mesh=not args.hide_mesh,
        )


__all__ = [
    "bernstein_degree_elevation_matrix",
    "DiffusionReactionBootstrapResult",
    "bootstrap_trace_initial_guess",
    "diff_rea_hdg_solve",
    "impose_boundary_trace_on_guess",
    "prolong_trace_coefficients",
    "solve_diffusion_reaction_hdg",
]


if __name__ == "__main__":
    _main()
