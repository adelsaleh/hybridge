#!/usr/bin/env python3
"""Run manufactured advection-reaction presets."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.advection_reaction.presets import (
    AdvectionReactionRunPreset,
    DEFAULT_PRESET,
    PRESETS,
    preset_by_key,
    print_preset_details,
    print_presets,
)


def _runtime_config(config: AdvectionReactionRunPreset, args) -> AdvectionReactionRunPreset:
    updates = {}
    if args.volume_quadrature is not None:
        updates["volume_quadrature"] = args.volume_quadrature
    if args.order is not None:
        updates["order"] = args.order
    if args.mesh_size is not None:
        updates["mesh_size"] = args.mesh_size
    if args.boundary_mode is not None:
        updates["boundary_mode"] = args.boundary_mode
    if args.trace_basis is not None:
        updates["trace_basis"] = args.trace_basis
    if args.trace_ordering is not None:
        updates["trace_ordering"] = args.trace_ordering
    if args.ilu_permc_spec is not None:
        updates["ilu_permc_spec"] = args.ilu_permc_spec
    if args.ilu_drop_tol is not None:
        updates["ilu_drop_tol"] = args.ilu_drop_tol
    if args.ilu_fill_factor is not None:
        updates["ilu_fill_factor"] = args.ilu_fill_factor
    if args.maxiter is not None:
        updates["maxiter"] = args.maxiter
    if args.solver_rtol is not None:
        updates["solver_rtol"] = args.solver_rtol
    if args.scale_system is not None:
        updates["scale_system"] = {
            "auto": None,
            "on": True,
            "off": False,
        }[args.scale_system]
    if args.assembly_backend is not None:
        updates["assembly_backend"] = args.assembly_backend
    if args.petsc_levels is not None:
        updates["petsc_levels"] = args.petsc_levels
    if args.plot_matrix_pattern:
        updates["matrix_pattern_dir"] = str(args.matrix_pattern_dir)
    if args.matrix_pattern_only:
        updates["matrix_pattern_only"] = True
        updates["matrix_pattern_dir"] = str(args.matrix_pattern_dir)
    if args.matrix_pattern_prefix is not None:
        updates["matrix_pattern_prefix"] = args.matrix_pattern_prefix
    if args.matrix_pattern_max_points is not None:
        updates["matrix_pattern_max_points"] = args.matrix_pattern_max_points
    if args.matrix_pattern_dpi is not None:
        updates["matrix_pattern_dpi"] = args.matrix_pattern_dpi
    if args.verbosity is not None:
        updates["verbosity"] = args.verbosity
    if args.quiet:
        updates["verbosity"] = 0
    if args.plot:
        updates["plot"] = True
    if args.plot_resolution is not None:
        updates["plot_resolution"] = args.plot_resolution
    if args.exact_plot_resolution is not None:
        updates["exact_plot_resolution"] = args.exact_plot_resolution
    if args.hide_mesh:
        updates["hide_mesh"] = True
    return replace(config, **updates) if updates else config


def _build_mesh(config: AdvectionReactionRunPreset, case):
    from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh

    domain = case.default_domain if config.domain == "auto" else config.domain
    if domain == "structured-rectangle":
        return rectangle_mesh(config.nx, config.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    if domain == "rectangle":
        return gmsh_rectangle_mesh(
            config.mesh_size,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            verbosity=config.gmsh_verbosity,
            algorithm=config.gmsh_algorithm,
        )
    if domain == "disc":
        return gmsh_disc_mesh(
            config.mesh_size,
            center=(0.0, 0.0),
            radius=1.0,
            verbosity=config.gmsh_verbosity,
            algorithm=config.gmsh_algorithm,
        )
    return gmsh_triangle_mesh(
        config.mesh_size,
        vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
        verbosity=config.gmsh_verbosity,
        algorithm=config.gmsh_algorithm,
    )


def _timing_with_percent(seconds: float, total: float, *, precision: int = 1) -> str:
    """Format elapsed seconds with its percentage of total runtime."""
    percent = 0.0 if total <= 0.0 else 100.0 * float(seconds) / float(total)
    return f"{float(seconds):.{precision}f} ({percent:.1f}%)"


def _numba_thread_count() -> int | None:
    """Return the active Numba worker count when Numba is importable."""
    try:
        from numba import get_num_threads
    except Exception:
        return None
    return int(get_num_threads())


def _summarize_solve(result, exact, *, preset_key: str, case, mesh, space, config: AdvectionReactionRunPreset) -> float:
    import numpy as np

    from hdgfem.io.output import pretty_print_sections

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

    run_mesh_items = [
        ("preset", preset_key, "s"),
        ("case", case.key, "s"),
        ("p", space.order, ",d"),
        ("triangles", mesh.num_tri, ",d"),
        ("edges", mesh.num_edg, ",d"),
        ("trace dofs", result.trace.size, ",d"),
    ]
    error_items = [
        ("theoretical h^(p+1)", mesh.h ** (space.order + 1), ".4e"),
        ("L2 error", l2_error, ".4e"),
        ("Linf error", linfty_error, ".4e"),
        ("avg max error", avg_error, ".4e"),
        ("max-error element", max_error_element, "d"),
    ]
    solver_items = [
        ("solver", "none" if config.solver is None else config.solver, "s"),
        ("preconditioner", "petsc" if str(config.solver).lower() == "petsc" else config.preconditioner or "none", "s"),
    ]
    option_items = [
        ("assembly backend", result.assembly_backend, "s"),
        ("boundary mode", result.boundary_mode, "s"),
        ("trace basis", config.trace_basis, "s"),
        ("trace ordering", result.trace_ordering, "s"),
        (
            "linear scaling",
            "auto" if config.scale_system is None else ("on" if config.scale_system else "off"),
            "s",
        ),
    ]
    if result.assembly_backend == "numba":
        numba_threads = _numba_thread_count()
        if numba_threads is not None:
            option_items.append(("numba threads", numba_threads, ",d"))
    total_time = result.timings.total
    timing_items = [
        ("assembly (s)", _timing_with_percent(result.timings.assembly, total_time), "s"),
        ("global solve (s)", _timing_with_percent(result.timings.solve, total_time), "s"),
        ("reconstruct (s)", _timing_with_percent(result.timings.reconstruction, total_time), "s"),
        ("total (s)", result.timings.total, "1.1f"),
    ]
    if str(config.solver).lower() == "petsc":
        option_items.extend(
            [
                ("PETSc preset", config.petsc_preset, "s"),
                ("PETSc levels", -1 if config.petsc_levels is None else config.petsc_levels, ",d"),
            ]
        )
    else:
        option_items.extend(
            [
                ("ILU drop", -1.0 if config.ilu_drop_tol is None else config.ilu_drop_tol, ".1e"),
                ("ILU fill", -1.0 if config.ilu_fill_factor is None else config.ilu_fill_factor, ".1f"),
            ]
        )
    if global_solve is not None:
        free_trace_relative_residual = global_solve.diagnostic_relative_residual_norm
        if free_trace_relative_residual is None and result.boundary_mode in {"eliminate", "zero-flux"}:
            free_trace_relative_residual = global_solve.solver_relative_residual_norm
        solver_items.extend(
            [
                ("Krylov iterations", -1 if global_solve.iteration_count is None else global_solve.iteration_count,
                 ",d"),
                (
                    "solver rel res",
                    np.nan
                    if global_solve.solver_relative_residual_norm is None
                    else global_solve.solver_relative_residual_norm,
                    ".3e",
                ),
                (
                    "free-trace rel res",
                    np.nan if free_trace_relative_residual is None else free_trace_relative_residual,
                    ".3e",
                ),
            ]
        )
        timing_items.extend(
            [
                (
                    "precond build (s)",
                    _timing_with_percent(
                        0.0
                        if global_solve.preconditioner_elapsed_seconds is None
                        else global_solve.preconditioner_elapsed_seconds,
                        total_time,
                        precision=3,
                    ),
                    "s",
                ),
                (
                    "Krylov solve (s)",
                    _timing_with_percent(
                        0.0 if global_solve.solve_elapsed_seconds is None else global_solve.solve_elapsed_seconds,
                        total_time,
                        precision=3,
                    ),
                    "s",
                ),
            ]
        )
    pretty_print_sections(
        [
            ("Run / mesh", run_mesh_items),
            ("Options", option_items),
            ("Solver", solver_items),
            ("Errors", error_items),
            ("Timings", timing_items),
        ],
        title="Advection-Reaction Preset Solve Summary",
    )
    return l2_error


def _polynomial_plot_resolution(requested_resolution: int | None, order: int) -> int:
    """Choose a per-element plotting grid dense enough for degree-``order`` fields."""
    minimum = max(3, 2 * int(order) + 3)
    if requested_resolution is None:
        return max(20, minimum)
    return max(int(requested_resolution), minimum)


def _matplotlib_contour_levels(order: int) -> int:
    """Choose enough contour bands for coarse per-element degree-``order`` plots."""
    return min(256, max(128, 24 * (int(order) + 1)))


def _plot_solution_comparison_matplotlib(
        field,
        exact,
        *,
        resolution: int,
        exact_resolution: int | str | None,
        title: str,
        show_mesh: bool,
        show: bool = True,
):
    import numpy as np

    from hdgfem.io.plot import (
        plot_scalar_sample_panels_matplotlib,
        resolve_exact_plot_resolution,
        sample_callable_on_elements,
        sample_field_on_elements,
    )

    mesh = field.space.mesh
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

    exact_panel_resolution = resolve_exact_plot_resolution(
        exact_resolution,
        numerical_resolution=resolution,
        num_elements=mesh.num_tri,
    )
    exact_reference_points, _, exact_values = sample_callable_on_elements(
        mesh,
        exact,
        resolution=exact_panel_resolution,
    )
    return plot_scalar_sample_panels_matplotlib(
        mesh,
        (
            ("Numerical solution", reference_points, numerical_values),
            ("Exact solution", exact_reference_points, exact_values, {"show_mesh": False}),
            ("Absolute error", reference_points, absolute_error, {"cmap": "magma", "zero_min": True}),
        ),
        suptitle=title,
        show_mesh=show_mesh,
        cmap="jet",
        levels=_matplotlib_contour_levels(field.space.order),
        share_clim=False,
        show=show,
    )


def _plot_solution_comparison(
        field,
        exact,
        *,
        resolution: int,
        exact_resolution: int | str | None,
        title: str,
        show_mesh: bool,
        show: bool = True,
):
    if field.space.mesh.num_tri <= 130:
        return _plot_solution_comparison_matplotlib(
            field,
            exact,
            resolution=_polynomial_plot_resolution(resolution, field.space.order),
            exact_resolution=exact_resolution,
            title=title,
            show_mesh=show_mesh,
            show=show,
        )

    from hdgfem.io.plot import plot_solution_comparison

    return plot_solution_comparison(
        field,
        exact,
        resolution=resolution,
        exact_resolution=exact_resolution,
        title=title,
        show_mesh=show_mesh,
        show=show,
    )


def _main() -> None:
    parser = ArgumentParser(
        description="Run one manufactured advection-reaction preset.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Curated preset configuration lives in scripts/advection_reaction/presets.py.\n"
            "Use scripts/advection_reaction/sweep_ilu.py for ILU parameter grids.\n"
            "Add new manufactured cases in scripts/advection_reaction/cases.py."
        ),
    )
    parser.add_argument("preset", nargs="?", default=DEFAULT_PRESET, choices=tuple(sorted(PRESETS)))
    parser.add_argument("--list-presets", action="store_true", help="print available presets and where to edit them")
    parser.add_argument("--print-preset", action="store_true", help="print the selected preset fields and exit")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the selected preset without solving")
    parser.add_argument(
        "--volume-quadrature",
        choices=("auto", "symmetric", "duffy"),
        default=None,
        help="override the triangle volume quadrature family for this run only",
    )
    parser.add_argument("--order", "-p", type=int, default=None, help="override uniform DG polynomial order")
    parser.add_argument("--mesh-size", "--lc", type=float, default=None, help="override Gmsh target mesh size")
    parser.add_argument(
        "--boundary-mode",
        choices=("penalty", "eliminate", "zero-flux"),
        default=None,
        help="override Dirichlet trace treatment for this run only",
    )
    parser.add_argument(
        "--trace-basis",
        choices=("legacy-lagrange", "legendre-modal", "bernstein"),
        default=None,
        help="override trace basis for this run only; legendre-modal is supported by numpy, numba, cupy, and raw-cuda safe assembly",
    )
    parser.add_argument(
        "--trace-ordering",
        choices=("none", "upwind-scc"),
        default=None,
        help="override trace-DOF ordering for this run only",
    )
    parser.add_argument(
        "--ilu-permc-spec",
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        default=None,
        help="override SuperLU spilu column permutation for this run only",
    )
    parser.add_argument("--ilu-drop-tol", type=float, default=None, help="override SuperLU spilu drop tolerance")
    parser.add_argument("--ilu-fill-factor", type=float, default=None, help="override SuperLU spilu fill factor")
    parser.add_argument("--maxiter", type=int, default=None, help="override Krylov maximum iterations")
    parser.add_argument("--solver-rtol", type=float, default=None, help="override Krylov relative tolerance")
    parser.add_argument(
        "--scale-system",
        choices=("auto", "on", "off"),
        default=None,
        help="override left Jacobi row scaling policy: auto, on, or off",
    )
    parser.add_argument(
        "--assembly-backend",
        choices=("numpy", "numba", "cupy", "raw-cuda", "auto"),
        default=None,
        help="override assembly backend for this run only",
    )
    parser.add_argument(
        "--petsc-levels",
        type=int,
        default=None,
        help="override PETSc ILU/factor levels for PETSc presets",
    )
    parser.add_argument(
        "--plot-matrix-pattern",
        action="store_true",
        help="write sparse matrix pattern plots before and after upwind SCC ordering",
    )
    parser.add_argument(
        "--matrix-pattern-dir",
        type=Path,
        default=Path("run_outputs") / "matrix_patterns",
        help="output directory for matrix pattern plots; defaults outside the hdgfem package",
    )
    parser.add_argument(
        "--matrix-pattern-prefix",
        default=None,
        help="filename prefix for matrix pattern plots",
    )
    parser.add_argument(
        "--matrix-pattern-max-points",
        type=int,
        default=None,
        help="maximum plotted nonzeros per matrix-pattern figure",
    )
    parser.add_argument(
        "--matrix-pattern-dpi",
        type=int,
        default=None,
        help="DPI for matrix-pattern PNG files",
    )
    parser.add_argument(
        "--matrix-pattern-only",
        action="store_true",
        help="assemble, save matrix patterns, then stop before the global solve",
    )
    parser.add_argument("--verbosity", "-v", type=int, default=None,
                        help="override logging verbosity for this run only")
    parser.add_argument("--quiet", action="store_true", help="run with verbosity 0 for this run only")
    parser.add_argument("--plot", action="store_true", help="show numerical/exact/error plots for this run only")
    parser.add_argument("--plot-resolution", type=int, default=None, help="plot sampling resolution for this run only; coarse meshes use a polynomial-degree minimum")
    parser.add_argument(
        "--exact-plot-resolution",
        type=int,
        default=None,
        help="exact-solution panel resolution for this run only",
    )
    parser.add_argument("--hide-mesh", action="store_true", help="hide mesh overlay in plots for this run only")
    args = parser.parse_args()

    if args.list_presets:
        print_presets()
        return

    preset_key = args.preset
    config = _runtime_config(preset_by_key(preset_key), args)

    from scripts.advection_reaction.cases import CASE_BY_KEY, case_definition_by_key

    if config.case not in CASE_BY_KEY:
        parser.error(f"preset {preset_key!r} references unknown case {config.case!r}")

    if args.print_preset or args.dry_run:
        print_preset_details(preset_key, config)
        return

    from hdgfem.core.space import DGSpace, VectorDGField
    from hdgfem.solvers.advection_reaction import AdvectionReactionHDGOptions, AdvectionReactionHDGSolver, _timed_call

    case = case_definition_by_key(config.case)
    beta_x, beta_y, reaction, source, exact = case.build(**config.case_params)
    mesh, _ = _timed_call(
        f"generating {config.domain} mesh",
        config.verbosity,
        lambda: _build_mesh(config, case),
    )
    space = DGSpace(
        mesh,
        config.order,
        basis_type=config.basis,
        volume_quadrature=config.volume_quadrature,
        volume_quad_1d=config.volume_quad_1d,
        edge_quad_1d=config.edge_quad_1d,
    )
    projected_backend = config.assembly_backend in {"numba", "raw-cuda"}
    source_input = (
        space.project_callable(source, name="source_h")
        if config.project_source or projected_backend
        else source
    )
    reaction_input = (
        space.project_callable(reaction, name="reaction_h")
        if config.project_reaction or projected_backend
        else reaction
    )
    beta_input = (
        VectorDGField((beta_x, beta_y), space, name="beta_h")
        if config.project_beta or projected_backend
        else (beta_x, beta_y)
    )

    options = AdvectionReactionHDGOptions(
        solver=config.solver,
        preconditioner=config.preconditioner,
        solver_rtol=config.solver_rtol,
        solver_atol=config.solver_atol,
        maxiter=config.maxiter,
        petsc_preset=config.petsc_preset,
        petsc_levels=config.petsc_levels,
        petsc_options=dict(config.petsc_options),
        petsc_divtol=config.petsc_divtol,
        petsc_monitor=config.petsc_monitor,
        ilu_drop_tol=config.ilu_drop_tol,
        ilu_fill_factor=config.ilu_fill_factor,
        ilu_failure=config.ilu_failure,
        scale_system=config.scale_system,
        boundary_mode=config.boundary_mode,
        trace_ordering=config.trace_ordering,
        trace_ordering_flux_tolerance=config.trace_ordering_flux_tolerance,
        ilu_permc_spec=config.ilu_permc_spec,
        matrix_pattern_dir=config.matrix_pattern_dir,
        matrix_pattern_prefix=(
            f"{preset_key}_p{space.order}_ne{mesh.num_tri}"
            if config.matrix_pattern_prefix == "adv_rea_trace_matrix"
            else config.matrix_pattern_prefix
        ),
        matrix_pattern_max_points=config.matrix_pattern_max_points,
        matrix_pattern_dpi=config.matrix_pattern_dpi,
        matrix_pattern_only=config.matrix_pattern_only,
        assembly_backend=config.assembly_backend,
        trace_basis=config.trace_basis,
        materialize_host_solution=config.materialize_host_solution,
        cache_local_solvers=config.cache_local_solvers,
        verbose=config.verbosity,
    )
    solver = AdvectionReactionHDGSolver(
        space,
        source=source_input,
        beta=beta_input,
        reaction=reaction_input,
        boundary_condition=exact,
        options=options,
    )
    result = solver.solve()
    if config.matrix_pattern_only:
        if result.matrix_pattern_plots is not None:
            print(f"matrix pattern before: {result.matrix_pattern_plots.before_path}")
            print(f"matrix pattern after : {result.matrix_pattern_plots.after_path}")
        return
    l2_error = _summarize_solve(
        result,
        exact,
        preset_key=preset_key,
        case=case,
        mesh=mesh,
        space=space,
        config=config,
    )

    if config.plot:
        title = f"{preset_key}, {case.key}, p={space.order}, elements={mesh.num_tri}, L2={l2_error:.2e}"
        _plot_solution_comparison(
            result.field,
            exact,
            resolution=config.plot_resolution,
            exact_resolution="auto" if config.exact_plot_resolution is None else config.exact_plot_resolution,
            title=title,
            show_mesh=not config.hide_mesh,
        )


if __name__ == "__main__":
    _main()
