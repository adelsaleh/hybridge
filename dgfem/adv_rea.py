"""Compact HDG solver for linear advection-reaction problems.

This module is the :mod:`dgfem` rewrite of the legacy
``adv_rea_vec_msh4.py`` solver.  The numerical structure is the same HDG
trace formulation, but the public API works with :class:`DGSpace`,
:class:`DGField`, and :class:`VectorDGField` objects instead of raw mesh and
quadrature tuples.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in {None, ""}:
    import runpy
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("dgfem.adv_rea", run_name="__main__")
    raise SystemExit

import time
from argparse import ArgumentParser
from dataclasses import dataclass
from math import pi
from typing import Callable, Iterable, Literal

import numpy as np

from . import hdg_assembly, hdg_mats
from .global_system import SolveResult
from .space import DGField, DGSpace


ReturnKey = Literal[
    "trace",
    "trace_coeffs",
    "matrix_rows",
    "matrix_cols",
    "matrix_data",
    "local_solver",
    "element_boundary_mats",
    "global_solve_result",
    "timings",
    "result",
]


@dataclass(frozen=True)
class AdvectionReactionTimings:
    """Wall-clock timings for the main HDG solve phases."""

    preparation: float
    assembly: float
    solve: float
    reconstruction: float
    total: float


@dataclass(frozen=True)
class AdvectionReactionResult:
    """Container returned by :func:`solve_advection_reaction_hdg`."""

    field: DGField
    trace: np.ndarray
    timings: AdvectionReactionTimings
    matrix_rows: np.ndarray | None = None
    matrix_cols: np.ndarray | None = None
    matrix_data: np.ndarray | None = None
    local_solver: np.ndarray | None = None
    element_boundary_mats: np.ndarray | None = None
    global_solve_result: SolveResult | None = None


def _format_seconds(seconds: float) -> str:
    """Format elapsed wall time for concise solver logging."""
    if seconds >= 100.0:
        return f"{seconds:.1f}s"
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds:.4f}s"


def _verbosity_level(verbose: bool | int) -> int:
    """Normalize bool/int verbosity flags to an integer level."""
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    return max(0, int(verbose))


def _timed_call(label: str, verbosity: bool | int, function, *, level: int = 1, multiline: bool = False):
    """Run ``function`` with legacy-style one-line timing output."""
    should_print = _verbosity_level(verbosity) >= level
    if should_print:
        indent = "  " * (level - 1)
        label = f"{indent}{label}"
        if multiline:
            print(f"{label} ...", flush=True)
        else:
            print(f"{label} ... ", end="", flush=True)
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    if should_print:
        if multiline:
            print(f"{label} ... done in {_format_seconds(elapsed)}")
        else:
            print(f"done in {_format_seconds(elapsed)}")
    return result, elapsed


def solve_advection_reaction_hdg(
        source,
        beta,
        reaction,
        boundary_condition: Callable,
        space: DGSpace,
        *,
        solver: str | None = "BICGSTAB",
        preconditioner="ilu",
        solver_rtol: float = 1e-13,
        solver_atol: float = 0.0,
        maxiter: int | None = None,
        project_reaction: bool = False,
        boundary_penalty: float = 1e20,
        verbose: bool | int = True,
        return_: Iterable[ReturnKey] = ("result",),
):
    r"""Solve :math:`\beta\cdot\nabla u + r u = f` with an HDG trace system.

    Parameters
    ----------
    source
        Callable, :class:`DGField`, source moment array, or source values on
        ``space`` volume quadrature points.
    beta
        Two-component :class:`VectorDGField`, coefficient array with shape
        ``(2, num_elements, el_dof)``, or tuple of two callables.
    reaction
        Scalar constant, callable, DG coefficient array, or reaction values on
        volume quadrature points.
    boundary_condition
        Dirichlet trace callable ``g(x, y)``.
    space
        Scalar solution DG space.
    solver
        Global trace solver name.  The default is ``"BICGSTAB"`` with ILU
        preconditioning.  Use ``"direct"`` or ``None`` for sparse direct solve.
    preconditioner
        Preconditioner passed to :func:`dgfem.global_system.solve_global_system`.
        The default ``"ilu"`` builds a SciPy ILU preconditioner.
    project_reaction
        If ``True`` and ``reaction`` is callable, first project it into
        ``space`` and assemble :math:`\int_K r_h\phi_i\phi_j` from cached
        reference triple products.  The default keeps callable reaction
        assembly exact on quadrature points.
    boundary_penalty
        Penalty used to impose boundary trace coefficients in the full trace
        system.
    verbose
        Verbosity level.  ``False`` disables logs, ``True``/``1`` prints one
        line per major solve phase, and ``2`` also prints assembly substeps.
    return_
        By default returns an :class:`AdvectionReactionResult`.  For legacy-like
        tuple output, request keys such as ``"trace"`` or ``"timings"``.
    """
    total_start = time.perf_counter()
    verbosity = _verbosity_level(verbose)
    if verbosity:
        print("\n----- DG FEM Advection-Reaction HDG Solve -----")

    def prepare_data():
        beta_field = hdg_assembly.as_vector_field(beta, space)
        beta_normal_flux = hdg_mats.advective_boundary_normal(beta_field, space)
        source_rhs = hdg_assembly.source_moments(source, space)
        if project_reaction and callable(reaction):
            reaction_data = space.project_callable(reaction, name="reaction_h")
        else:
            reaction_data = reaction
        return beta_field, beta_normal_flux, source_rhs, reaction_data

    (beta_h, beta_dot_normal, source_moments, reaction_h), preparation = _timed_call(
        "preparing projected data",
        verbosity,
        prepare_data,
    )

    def assemble_local_mats():
        local_blocks, _ = _timed_call(
            "assembling boundary mass matrices",
            verbosity,
            lambda: np.ascontiguousarray(hdg_mats.boundary_mass_from_normal_flux(space, beta_dot_normal)),
            level=2,
        )
        scratch_blocks = np.empty_like(local_blocks)
        _timed_call(
            "accumulating reaction mass matrices",
            verbosity,
            lambda: hdg_mats.add_reaction_mass(
                local_blocks,
                reaction_h,
                space,
                scratch=scratch_blocks,
            ),
            level=2,
        )
        _timed_call(
            "assembling advection matrices",
            verbosity,
            lambda: hdg_mats.add_advection_mats(
                local_blocks,
                space,
                beta_h,
                scale=-1.0,
            ),
            level=2,
        )
        return local_blocks

    local_mats, local_assembly = _timed_call(
        "assembling local element matrices",
        verbosity,
        assemble_local_mats,
        multiline=verbosity >= 2,
    )
    local_solver, local_inverse = _timed_call(
        "inverting local element matrices",
        verbosity,
        lambda: np.linalg.inv(local_mats),
    )
    element_boundary_mats, boundary_assembly = _timed_call(
        "assembling element boundary coupling",
        verbosity,
        lambda: hdg_mats.element_boundary_mats_from_normal_flux(space, beta_dot_normal),
    )

    def assemble_global_trace_system():
        trace_blocks, _ = _timed_call(
            "forming element trace Schur blocks",
            verbosity,
            lambda: hdg_assembly.element_to_trace_matrix(local_solver, element_boundary_mats, space),
            level=2,
        )
        (matrix_rows, matrix_cols), _ = _timed_call(
            "building global COO index arrays",
            verbosity,
            lambda: hdg_assembly.trace_matrix_indices(space),
            level=2,
        )
        matrix_data, _ = _timed_call(
            "assembling global COO data",
            verbosity,
            lambda: hdg_assembly.trace_matrix_data(trace_blocks, space, boundary_penalty),
            level=2,
        )
        (matrix_rhs, _), _ = _timed_call(
            "assembling global RHS",
            verbosity,
            lambda: hdg_assembly.global_rhs(source_moments, local_solver, boundary_condition, space, boundary_penalty),
            level=2,
        )
        return matrix_rows, matrix_cols, matrix_data, matrix_rhs

    (rows, cols, data, rhs), trace_assembly = _timed_call(
        "assembling global trace system",
        verbosity,
        assemble_global_trace_system,
        multiline=verbosity >= 2,
    )
    assembly = local_assembly + local_inverse + boundary_assembly + trace_assembly

    global_solve_result, solve_time = _timed_call(
        "solving global system",
        verbosity,
        lambda: hdg_assembly.solve_trace_system(
            rows,
            cols,
            data,
            rhs,
            solver=solver,
            preconditioner=preconditioner,
            rtol=solver_rtol,
            atol=solver_atol,
            maxiter=maxiter,
            verbose=False,
        ),
        multiline=verbosity >= 2,
    )
    trace = np.asarray(global_solve_result.x, dtype=np.float64)

    field, reconstruction = _timed_call(
        "reconstructing element field",
        verbosity,
        lambda: hdg_assembly.reconstruct_field(trace, source_moments, local_solver, element_boundary_mats, space),
    )

    timings = AdvectionReactionTimings(
        preparation=preparation,
        assembly=assembly,
        solve=solve_time,
        reconstruction=reconstruction,
        total=time.perf_counter() - total_start,
    )
    result = AdvectionReactionResult(
        field=field,
        trace=trace,
        timings=timings,
        matrix_rows=rows,
        matrix_cols=cols,
        matrix_data=data,
        local_solver=local_solver,
        element_boundary_mats=element_boundary_mats,
        global_solve_result=global_solve_result,
    )

    want = tuple(return_)
    if want == ("result",):
        return result
    output = []
    for key in want:
        if key == "result":
            output.append(result)
        elif key in {"trace", "trace_coeffs"}:
            output.append(trace)
        elif key == "matrix_rows":
            output.append(rows)
        elif key == "matrix_cols":
            output.append(cols)
        elif key == "matrix_data":
            output.append(data)
        elif key == "local_solver":
            output.append(local_solver)
        elif key == "element_boundary_mats":
            output.append(element_boundary_mats)
        elif key == "global_solve_result":
            output.append(global_solve_result)
        elif key == "timings":
            output.append(timings)
        else:
            raise ValueError(f"unknown return key {key!r}")
    return tuple(output)


adv_rea_hdg_solv = solve_advection_reaction_hdg


def test2(m: float = 5, n: float = 5, a: float = 2, b: float = 0):
    """Manufactured legacy advection-reaction test used by ``adv_rea_vec_msh4``."""

    def f2(t):
        return a * np.cos(m * pi * t) + b * np.sin(n * pi * t)

    return (
        lambda x, y: x + 0 * y,
        lambda x, y: -y + 0 * x,
        lambda x, y: y**2 + 0 * x,
        lambda x, y: y**2 + 0 * x,
        lambda x, y: f2(x * y) * np.exp(y**2 / 2.0) + 1.0,
    )


__all__ = [
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "adv_rea_hdg_solv",
    "solve_advection_reaction_hdg",
    "test2",
]


def _main() -> None:
    """Run the legacy manufactured advection-reaction test."""
    from .mesh import gmsh_disc_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh
    from .plot import plot_solution_comparison
    from .space import DGSpace
    from .output import pretty_print_ncol

    parser = ArgumentParser(description="Run the dgfem advection-reaction HDG test2 problem.")
    parser.add_argument("--order", "-p", type=int, default=2, help="uniform DG polynomial order")
    parser.add_argument(
        "--domain",
        default="rectangle",
        choices=("rectangle", "disc", "triangle", "structured-rectangle"),
        help="mesh domain; rectangle/disc/triangle use Gmsh",
    )
    parser.add_argument("--mesh-size", "--lc", type=float, default=0.35, help="Gmsh target mesh size")
    parser.add_argument("--nx", type=int, default=8, help="structured rectangle cells in x")
    parser.add_argument("--ny", type=int, default=None, help="structured rectangle cells in y; defaults to nx")
    parser.add_argument("--gmsh-verbosity", type=int, default=0, help="Gmsh verbosity level")
    parser.add_argument("--gmsh-algorithm", type=int, default=None, help="optional Gmsh 2D meshing algorithm")
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--solver", default="BICGSTAB", help="global trace solver; use 'direct' for sparse direct solve")
    parser.add_argument("--preconditioner", default="ilu", choices=("ilu", "none"), help="global trace preconditioner")
    parser.add_argument("--solver-rtol", type=float, default=1e-13, help="relative tolerance for iterative solves")
    parser.add_argument("--solver-atol", type=float, default=0.0, help="absolute tolerance for iterative solves")
    parser.add_argument("--maxiter", type=int, default=None, help="maximum Krylov iterations")
    parser.add_argument("--project-reaction", action="store_true", help="project callable reaction into Vh before assembly")
    parser.add_argument("--verbosity", "-v", type=int, default=1, help="logging verbosity: 0 quiet, 1 phases, 2 substeps")
    parser.add_argument("--quiet", action="store_true", help="suppress phase timing output")
    parser.add_argument("--plot", action="store_true", help="plot numerical, exact, and absolute-error fields")
    parser.add_argument("--plot-resolution", type=int, default=20, help="samples per reference axis for plotting")
    parser.add_argument("--hide-mesh", action="store_true", help="do not overlay the coarse mesh on plots")
    args = parser.parse_args()

    verbosity = 0 if args.quiet else max(0, int(args.verbosity))

    def build_mesh():
        if args.domain == "structured-rectangle":
            return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        if args.domain == "rectangle":
            return gmsh_rectangle_mesh(
                args.mesh_size,
                xlim=(-1.0, 1.0),
                ylim=(-1.0, 1.0),
                verbosity=args.gmsh_verbosity,
                algorithm=args.gmsh_algorithm,
            )
        if args.domain == "disc":
            return gmsh_disc_mesh(
                args.mesh_size,
                center=(0.0, 0.0),
                radius=1.0,
                verbosity=args.gmsh_verbosity,
                algorithm=args.gmsh_algorithm,
            )
        return gmsh_triangle_mesh(
            args.mesh_size,
            vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
        )

    mesh, _ = _timed_call(f"generating {args.domain} mesh", verbosity, build_mesh)
    space = DGSpace(mesh, args.order, basis_type=args.basis)
    beta_x, beta_y, reaction, source, exact = test2()
    result = solve_advection_reaction_hdg(
        source,
        (beta_x, beta_y),
        reaction,
        exact,
        space,
        solver=args.solver,
        preconditioner=None if args.preconditioner == "none" else args.preconditioner,
        solver_rtol=args.solver_rtol,
        solver_atol=args.solver_atol,
        maxiter=args.maxiter,
        project_reaction=args.project_reaction,
        verbose=verbosity,
    )
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
        ("ℓ_c (Gmsh)", args.mesh_size, ".3f"),
        ("h^p", mesh.h ** (space.order + 1), ".4e"),
        ("L₂ error", l2_error, ".4e"),
        ("L∞ error", linfty_error, ".4e"),
        ("avg error", avg_error, ".4e"),
        ("max_err at el", max_error_element, "d"),
        ("prep time(s)", result.timings.preparation, "1.1f"),
        ("setup time(s)", result.timings.assembly, "1.1f"),
        ("glb_solve time(s)", result.timings.solve, "1.1f"),
        ("recons time(s)", result.timings.reconstruction, "1.1f"),
        ("tot time(s)", result.timings.total, "1.1f"),
        ("solver", args.solver, "s"),
        ("reaction", "projected" if args.project_reaction else "exact", "s"),
    ]
    if global_solve is not None:
        items.extend(
            [
                ("iterations", -1 if global_solve.iteration_count is None else global_solve.iteration_count, ",d"),
                ("solver rel res", np.nan if global_solve.solver_relative_residual_norm is None else global_solve.solver_relative_residual_norm, ".3e"),
                ("physical rel res", np.nan if global_solve.physical_relative_residual_norm is None else global_solve.physical_relative_residual_norm, ".3e"),
                ("ILU time(s)", 0.0 if global_solve.preconditioner_elapsed_seconds is None else global_solve.preconditioner_elapsed_seconds, ".3f"),
                ("Krylov time(s)", 0.0 if global_solve.solve_elapsed_seconds is None else global_solve.solve_elapsed_seconds, ".3f"),
            ]
        )
    pretty_print_ncol(items, ncols=3, title="Solve Summary")

    if args.plot:
        title = f"test2, p={space.order}, elements={mesh.num_tri}, L2={l2_error:.2e}"
        plot_solution_comparison(
            result.field,
            exact,
            resolution=args.plot_resolution,
            title=title,
            show_mesh=not args.hide_mesh,
        )


if __name__ == "__main__":
    _main()
