#!/usr/bin/env python3
"""Run manufactured advection-reaction presets."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class AdvectionReactionRunPreset:
    """Complete default configuration for one manufactured advection run."""

    case: str
    description: str
    case_params: dict[str, Any] = field(default_factory=dict)
    domain: str = "auto"
    mesh_size: float = 0.03
    nx: int = 8
    ny: int | None = None
    gmsh_verbosity: int = 0
    gmsh_algorithm: int | None = None
    basis: str = "dub_orth"
    order: int = 4
    solver: str | None = "BICGSTAB"
    preconditioner: str | None = "ilu"
    solver_rtol: float = 1.0e-13
    solver_atol: float = 0.0
    maxiter: int | None = None
    petsc_preset: str = "gmres_ilu"
    petsc_levels: int | None = None
    petsc_options: dict[str, str] = field(default_factory=dict)
    petsc_divtol: float = 1.0e4
    petsc_monitor: bool = False
    ilu_drop_tol: float | None = None
    ilu_fill_factor: float | None = None
    ilu_failure: str = "raise"
    boundary_mode: str = "eliminate"
    trace_ordering: str = "upwind-scc"
    trace_ordering_flux_tolerance: float = 0.0
    ilu_permc_spec: str | None = None
    matrix_pattern_dir: str | None = None
    matrix_pattern_prefix: str = "adv_rea_trace_matrix"
    matrix_pattern_max_points: int = 2_000_000
    matrix_pattern_dpi: int = 250
    matrix_pattern_only: bool = False
    assembly_backend: str = "numba"
    project_source: bool = True
    project_beta: bool = True
    project_reaction: bool = True
    cache_local_solvers: bool = False
    verbosity: int = 1
    plot: bool = False
    plot_resolution: int = 20
    exact_plot_resolution: int | str | None = None
    hide_mesh: bool = False


def _test2_solver_preset(
        *,
        description: str,
        solver: str | None,
        mesh_size: float= 0.01,
        order: int = 6,
        preconditioner: str | None,
        petsc_preset: str = "gmres_ilu",
        ilu_drop_tol: float | None = None,
        ilu_fill_factor: float | None = None,
        maxiter: int | None = None,
) -> AdvectionReactionRunPreset:
    return AdvectionReactionRunPreset(
        case="test2",
        description=description,
        solver=solver,
        preconditioner=preconditioner,
        petsc_preset=petsc_preset,
        ilu_drop_tol=ilu_drop_tol,
        ilu_fill_factor=ilu_fill_factor,
        maxiter=maxiter,
        mesh_size=mesh_size,
        order=order,
        boundary_mode="eliminate",
        trace_ordering="upwind-scc",
        assembly_backend="numba",
        project_source=True,
        project_beta=True,
        project_reaction=True,
        verbosity=2,
    )


PRESETS: dict[str, AdvectionReactionRunPreset] = {
    "test2_scipy_ilu_upwind": _test2_solver_preset(
        description="test2 with SciPy BICGSTAB, upwind ordering, and high-fill ILU.",
        solver="BICGSTAB",
        preconditioner="ilu",
        ilu_drop_tol=1.0e-10,
        ilu_fill_factor=35.0,
        maxiter=2000,
    ),
    "test2_scipy_direct": _test2_solver_preset(
        description="test2 with SciPy sparse direct solve.",
        solver="direct",
        preconditioner=None,
    ),
    "test2_petsc_bicgstab_ilu": _test2_solver_preset(
        description="test2 with PETSc BiCGStab and ILU.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="bicgstab_ilu",
        maxiter=2000,
    ),
    "test2_petsc_gmres_ilu": _test2_solver_preset(
        description="test2 with PETSc GMRES and ILU.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="gmres_ilu",
        maxiter=2000,
    ),
    "test2_petsc_lu": _test2_solver_preset(
        description="test2 with PETSc direct LU.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="lu",
    ),
    "test2_petsc_mumps_lu": _test2_solver_preset(
        description="test2 with PETSc direct LU using MUMPS.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="mumps_lu",
    ),
}

DEFAULT_PRESET = "test2_scipy_ilu_upwind"


def preset_by_key(key: str) -> AdvectionReactionRunPreset:
    """Return a run preset by name."""
    try:
        return PRESETS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown advection-reaction preset {key!r}; valid presets are {valid}") from exc


def _print_presets() -> None:
    script_path = Path(__file__).resolve()
    print(f"Preset definitions: {script_path}")
    print("Edit the PRESETS dictionary in this file to change or add runs.")
    print("Manufactured advection cases are registered in scripts/adv_rea_cases.py.\n")

    width = max(len(key) for key in PRESETS)
    for key in sorted(PRESETS):
        preset = PRESETS[key]
        solver = "petsc" if str(preset.solver).lower() == "petsc" else str(preset.solver)
        print(
            f"{key:<{width}}  "
            f"case={preset.case:<8} "
            f"p={preset.order:<2d} "
            f"lc={preset.mesh_size:<6.3f} "
            f"backend={preset.assembly_backend:<5} "
            f"solver={solver:<8} "
            f"{preset.description}"
        )


def _print_preset_details(preset_key: str, config: AdvectionReactionRunPreset) -> None:
    print(f"Preset: {preset_key}")
    print(f"Defined in: {Path(__file__).resolve()}")
    for key, value in asdict(config).items():
        print(f"{key}: {value!r}")


def _runtime_config(config: AdvectionReactionRunPreset, args) -> AdvectionReactionRunPreset:
    updates = {}
    if args.order is not None:
        updates["order"] = args.order
    if args.mesh_size is not None:
        updates["mesh_size"] = args.mesh_size
    if args.boundary_mode is not None:
        updates["boundary_mode"] = args.boundary_mode
    if args.trace_ordering is not None:
        updates["trace_ordering"] = args.trace_ordering
    if args.ilu_permc_spec is not None:
        updates["ilu_permc_spec"] = args.ilu_permc_spec
    if args.assembly_backend is not None:
        updates["assembly_backend"] = args.assembly_backend
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


def _summarize_solve(result, exact, *, preset_key: str, case, mesh, space, config: AdvectionReactionRunPreset) -> float:
    import numpy as np

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
        ("preset", preset_key, "s"),
        ("case", case.key, "s"),
        ("p", space.order, ",d"),
        ("#triangles", mesh.num_tri, ",d"),
        ("# edges", mesh.num_edg, ",d"),
        ("#global_dof", result.trace.size, ",d"),
        ("h^p", mesh.h ** (space.order + 1), ".4e"),
        ("L2 error", l2_error, ".4e"),
        ("Linf error", linfty_error, ".4e"),
        ("avg error", avg_error, ".4e"),
        ("max_err at el", max_error_element, "d"),
        ("setup time(s)", result.timings.assembly, "1.1f"),
        ("glb_solve time(s)", result.timings.solve, "1.1f"),
        ("recons time(s)", result.timings.reconstruction, "1.1f"),
        ("tot time(s)", result.timings.total, "1.1f"),
        ("solver", "none" if config.solver is None else config.solver, "s"),
        ("preconditioner", "petsc" if str(config.solver).lower() == "petsc" else config.preconditioner or "none", "s"),
        ("assembly backend", result.assembly_backend, "s"),
        ("boundary mode", result.boundary_mode, "s"),
        ("trace ordering", result.trace_ordering, "s"),
    ]
    if str(config.solver).lower() == "petsc":
        items.extend(
            [
                ("PETSc preset", config.petsc_preset, "s"),
                ("PETSc levels", -1 if config.petsc_levels is None else config.petsc_levels, ",d"),
            ]
        )
    else:
        items.extend(
            [
                ("ILU drop", -1.0 if config.ilu_drop_tol is None else config.ilu_drop_tol, ".1e"),
                ("ILU fill", -1.0 if config.ilu_fill_factor is None else config.ilu_fill_factor, ".1f"),
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
                    np.nan
                    if global_solve.solver_relative_residual_norm is None
                    else global_solve.solver_relative_residual_norm,
                    ".3e",
                ),
                (
                    "free trace rel res",
                    np.nan if free_trace_relative_residual is None else free_trace_relative_residual,
                    ".3e",
                ),
                (
                    "prec time(s)",
                    0.0
                    if global_solve.preconditioner_elapsed_seconds is None
                    else global_solve.preconditioner_elapsed_seconds,
                    ".3f",
                ),
                (
                    "Krylov time(s)",
                    0.0 if global_solve.solve_elapsed_seconds is None else global_solve.solve_elapsed_seconds,
                    ".3f",
                ),
            ]
        )
    pretty_print_ncol(items, ncols=3, title="Advection-Reaction Preset Solve Summary")
    return l2_error


def _main() -> None:
    parser = ArgumentParser(
        description="Run one manufactured advection-reaction preset.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Preset configuration lives in this file, scripts/run_adv_rea_cases.py.\n"
            "Edit PRESETS to change numerical parameters or add a new run.\n"
            "Add new manufactured cases in scripts/adv_rea_cases.py."
        ),
    )
    parser.add_argument("preset", nargs="?", default=DEFAULT_PRESET, choices=tuple(sorted(PRESETS)))
    parser.add_argument("--list-presets", action="store_true", help="print available presets and where to edit them")
    parser.add_argument("--print-preset", action="store_true", help="print the selected preset fields and exit")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the selected preset without solving")
    parser.add_argument("--order", "-p", type=int, default=None, help="override uniform DG polynomial order")
    parser.add_argument("--mesh-size", "--lc", type=float, default=None, help="override Gmsh target mesh size")
    parser.add_argument(
        "--boundary-mode",
        choices=("penalty", "eliminate"),
        default=None,
        help="override Dirichlet trace treatment for this run only",
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
    parser.add_argument(
        "--assembly-backend",
        choices=("numpy", "numba", "auto"),
        default=None,
        help="override assembly backend for this run only",
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
    parser.add_argument("--plot-resolution", type=int, default=None, help="plot sampling resolution for this run only")
    parser.add_argument(
        "--exact-plot-resolution",
        type=int,
        default=None,
        help="exact-solution panel resolution for this run only",
    )
    parser.add_argument("--hide-mesh", action="store_true", help="hide mesh overlay in plots for this run only")
    args = parser.parse_args()

    if args.list_presets:
        _print_presets()
        return

    preset_key = args.preset
    config = _runtime_config(preset_by_key(preset_key), args)

    from scripts.adv_rea_cases import CASE_BY_KEY, case_definition_by_key

    if config.case not in CASE_BY_KEY:
        parser.error(f"preset {preset_key!r} references unknown case {config.case!r}")

    if args.print_preset or args.dry_run:
        _print_preset_details(preset_key, config)
        return

    from hdgfem.core.space import DGField, DGSpace, VectorDGField
    from hdgfem.io.plot import plot_solution_comparison
    from hdgfem.solvers.adv_rea import AdvectionReactionHDGOptions, AdvectionReactionHDGSolver, _timed_call

    case = case_definition_by_key(config.case)
    beta_x, beta_y, reaction, source, exact = case.build(**config.case_params)
    mesh, _ = _timed_call(
        f"generating {config.domain} mesh",
        config.verbosity,
        lambda: _build_mesh(config, case),
    )
    space = DGSpace(mesh, config.order, basis_type=config.basis)
    source_input = DGField(source, space, name="source_h") if config.project_source else source
    reaction_input = DGField(reaction, space, name="reaction_h") if config.project_reaction else reaction
    beta_input = VectorDGField((beta_x, beta_y), space, name="beta_h") if config.project_beta else (beta_x, beta_y)

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
        plot_solution_comparison(
            result.field,
            exact,
            resolution=config.plot_resolution,
            exact_resolution="auto" if config.exact_plot_resolution is None else config.exact_plot_resolution,
            title=title,
            show_mesh=not config.hide_mesh,
        )


if __name__ == "__main__":
    _main()
