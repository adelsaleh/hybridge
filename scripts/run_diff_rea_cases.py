#!/usr/bin/env python3
"""Run manufactured diffusion-reaction presets.

Preset definitions live in this file, in the ``PRESETS`` dictionary below.

To create a new manufactured test:
1. Add a factory in ``scripts/diff_rea_cases.py`` that returns
   ``(diffusion, reaction, source, exact)``.
2. Register it in ``CASE_DEFINITIONS`` in that same file.
3. Add one or more ``DiffusionReactionRunPreset`` entries in ``PRESETS`` below.

To change mesh size, polynomial order, stabilization, quadrature, solver, PETSc
settings, plotting, or case parameters such as tensor-sine ``m``/``n``, edit
the corresponding preset here.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class DiffusionReactionRunPreset:
    """Complete default configuration for one manufactured-case run."""

    case: str
    description: str
    case_params: dict[str, Any] = field(default_factory=dict)
    domain: str = "auto"
    mesh_size: float = 0.35
    nx: int = 8
    ny: int | None = None
    gmsh_verbosity: int = 0
    basis: str = "dub_orth"
    order: int = 4
    volume_quad_1d: int | None = None
    edge_quad_1d: int | None = None
    tau: float = 1.0
    local_backend: str = "numpy"
    assembly_backend: str = "numpy"
    solver: str | None = "BICGSTAB"
    preconditioner: str | None = "ilu"
    solver_rtol: float = 1.0e-13
    solver_atol: float = 0.0
    maxiter: int | None = None
    scale_system: bool = True
    petsc_preset: str = "cg_gamg"
    petsc_levels: int | None = None
    petsc_options: dict[str, str] = field(default_factory=dict)
    petsc_divtol: float = 1.0e4
    petsc_monitor: bool = False
    ilu_drop_tol: float = 1.0e-10
    ilu_fill_factor: float = 35.0
    ilu_failure: str = "none"
    boundary_mode: str = "penalty"
    verbosity: int = 1
    plot: bool = False
    plot_resolution: int = 20
    exact_plot_resolution: int | str | None = None
    hide_mesh: bool = False


# Edit this dictionary to change existing runs or add new preset names.
# The ``case`` value must match a key from scripts/diff_rea_cases.py.
PRESETS: dict[str, DiffusionReactionRunPreset] = {
    "quadratic_poisson": DiffusionReactionRunPreset(
        case="quadratic-poisson",
        description="Default quadratic pure-Poisson smoke run.",
    ),
    "exponential_bubble": DiffusionReactionRunPreset(
        case="exponential-bubble",
        description="Legacy exponential bubble pure-Poisson case.",
    ),
    "trigonometric_poisson": DiffusionReactionRunPreset(
        case="trigonometric-poisson",
        description="Legacy trigonometric pure-Poisson case on its default disk domain.",
        solver="petsc",
        petsc_preset="cg_gamg",
        boundary_mode="eliminate",
        assembly_backend="numba",
        petsc_levels=10,
        order=6,
        verbosity=2,
        mesh_size=0.05,
    ),
    "quadratic_variable_reaction": DiffusionReactionRunPreset(
        case="quadratic-variable-reaction",
        description="Quadratic exact solution with smooth variable reaction.",
    ),
    "lshape_singular": DiffusionReactionRunPreset(
        case="lshape-singular",
        description="Legacy reentrant-corner singular harmonic case.",
    ),
    "tensor_sine_quick": DiffusionReactionRunPreset(
        case="tensor-sine",
        description="Small tensor-sine projected-Numba smoke run.",
        case_params={"m": 1, "n": 1},
        domain="structured-rectangle",
        nx=8,
        ny=8,
        order=2,
        tau=4.0,
        assembly_backend="numba",
        boundary_mode="eliminate",
        volume_quad_1d=4,
        edge_quad_1d=3,
    ),
    "tensor_sine_gamg": DiffusionReactionRunPreset(
        case="tensor-sine",
        description="Large p=6 tensor-sine run using projected tensor Numba assembly and PETSc GAMG.",
        case_params={"m": 1, "n": 1},
        domain="structured-rectangle",
        nx=200,
        ny=200,
        order=6,
        tau=4.0,
        assembly_backend="numba",
        solver="petsc",
        preconditioner=None,
        petsc_preset="cg_gamg",
        boundary_mode="eliminate",
        volume_quad_1d=7,
        edge_quad_1d=7,
    ),
}

DEFAULT_PRESET = "quadratic_poisson"


def preset_by_key(key: str) -> DiffusionReactionRunPreset:
    """Return a run preset by name."""
    try:
        return PRESETS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown diffusion-reaction preset {key!r}; valid presets are {valid}") from exc


def _print_presets() -> None:
    """Print available presets and the file location to edit them."""
    script_path = Path(__file__).resolve()
    print(f"Preset definitions: {script_path}")
    print("Edit the PRESETS dictionary in this file to change or add runs.")
    print("Manufactured PDE cases are registered in scripts/diff_rea_cases.py.\n")

    width = max(len(key) for key in PRESETS)
    for key in sorted(PRESETS):
        preset = PRESETS[key]
        solver = "petsc" if str(preset.solver).lower() == "petsc" else str(preset.solver)
        print(
            f"{key:<{width}}  "
            f"case={preset.case:<28} "
            f"p={preset.order:<2d} "
            f"domain={preset.domain:<20} "
            f"backend={preset.assembly_backend:<5} "
            f"solver={solver:<8} "
            f"{preset.description}"
        )


def _print_preset_details(preset_key: str, config: DiffusionReactionRunPreset) -> None:
    """Print every field in one preset for inspection."""
    print(f"Preset: {preset_key}")
    print(f"Defined in: {Path(__file__).resolve()}")
    for key, value in asdict(config).items():
        print(f"{key}: {value!r}")


def _runtime_config(config: DiffusionReactionRunPreset, args) -> DiffusionReactionRunPreset:
    """Apply CLI presentation/diagnostic choices without changing numerical inputs."""
    updates = {}
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


def _build_mesh(config: DiffusionReactionRunPreset, case):
    from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_lshape_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh

    domain = config.domain
    if domain == "auto":
        domain = case.default_domain
    if domain == "structured-rectangle":
        return rectangle_mesh(config.nx, config.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    if domain == "unit-rectangle":
        return gmsh_rectangle_mesh(
            config.mesh_size,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            verbosity=config.gmsh_verbosity,
        )
    if domain == "rectangle":
        return gmsh_rectangle_mesh(
            config.mesh_size,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            verbosity=config.gmsh_verbosity,
        )
    if domain == "disc":
        radius = 5.0 if case.key == "trigonometric-poisson" and config.domain == "auto" else 1.0
        return gmsh_disc_mesh(
            config.mesh_size,
            center=(0.0, 0.0),
            radius=radius,
            verbosity=config.gmsh_verbosity,
        )
    if domain == "lshape":
        use_auto_lshape_refinement = case.key == "lshape-singular" and config.domain == "auto"
        return gmsh_lshape_mesh(
            config.mesh_size,
            corner_mesh_size=config.mesh_size / 10.0 if use_auto_lshape_refinement else None,
            corner_refine_radius=0.1 if use_auto_lshape_refinement else 0.4,
            verbosity=config.gmsh_verbosity,
        )
    return gmsh_triangle_mesh(
        config.mesh_size,
        vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
        verbosity=config.gmsh_verbosity,
    )


def _summarize_solve(
        result,
        exact,
        *,
        diffusion,
        preset_key: str,
        case,
        mesh,
        space,
        config: DiffusionReactionRunPreset,
) -> float:
    import numpy as np

    from hdgfem.io.output import pretty_print_ncol
    from hdgfem.solvers.diff_rea import _diffusion_is_identity

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
        ("tau", config.tau, ".3e"),
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
        ("solver", "none" if config.solver is None else config.solver, "s"),
        ("preconditioner", "petsc" if str(config.solver).lower() == "petsc" else config.preconditioner or "none", "s"),
        ("scaling", "left" if result.scale_system else "none", "s"),
        ("assembly backend", result.assembly_backend, "s"),
        ("local backend", "fused" if result.assembly_backend == "numba" and result.local_solver is None else config.local_backend, "s"),
        ("boundary mode", result.boundary_mode, "s"),
    ]
    if str(config.solver).lower() == "petsc":
        items.extend(
            [
                ("PETSc preset", config.petsc_preset, "s"),
                ("PETSc levels", -1 if config.petsc_levels is None else config.petsc_levels, ",d"),
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
    pretty_print_ncol(items, ncols=3, title="Diffusion-Reaction Preset Solve Summary")
    return l2_error


def _main() -> None:
    parser = ArgumentParser(
        description="Run one manufactured diffusion-reaction preset.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Preset configuration lives in this file, scripts/run_diff_rea_cases.py.\n"
            "Edit PRESETS to change numerical parameters or add a new run.\n"
            "Add new manufactured PDE cases in scripts/diff_rea_cases.py.\n"
            "CLI flags are limited to plotting, verbosity, and preset inspection."
        ),
    )
    parser.add_argument("preset", nargs="?", default=DEFAULT_PRESET, choices=tuple(sorted(PRESETS)))
    parser.add_argument("--list-presets", action="store_true", help="print available presets and where to edit them")
    parser.add_argument("--print-preset", action="store_true", help="print the selected preset fields and exit")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the selected preset without solving")
    parser.add_argument("--verbosity", "-v", type=int, default=None, help="override logging verbosity for this run only")
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

    from scripts.diff_rea_cases import CASE_BY_KEY, case_definition_by_key

    if config.case not in CASE_BY_KEY:
        parser.error(f"preset {preset_key!r} references unknown case {config.case!r}")

    if args.print_preset or args.dry_run:
        _print_preset_details(preset_key, config)
        return

    from hdgfem.core.space import DGSpace
    from hdgfem.solvers.diff_rea import DiffusionReactionHDGOptions, DiffusionReactionHDGSolver, _timed_call

    case = case_definition_by_key(config.case)
    diffusion, reaction, source, exact = case.build(**config.case_params)
    mesh, _ = _timed_call(
        f"generating {config.domain} mesh",
        config.verbosity,
        lambda: _build_mesh(config, case),
    )
    space = DGSpace(
        mesh,
        config.order,
        basis_type=config.basis,
        volume_quad_1d=config.volume_quad_1d,
        edge_quad_1d=config.edge_quad_1d,
    )
    options = DiffusionReactionHDGOptions(
        diffusion=diffusion,
        stabilization=config.tau,
        solver=config.solver,
        preconditioner=config.preconditioner,
        solver_rtol=config.solver_rtol,
        solver_atol=config.solver_atol,
        maxiter=config.maxiter,
        scale_system=config.scale_system,
        petsc_preset=config.petsc_preset,
        petsc_levels=config.petsc_levels,
        petsc_options=dict(config.petsc_options),
        petsc_divtol=config.petsc_divtol,
        petsc_monitor=config.petsc_monitor,
        ilu_drop_tol=config.ilu_drop_tol,
        ilu_fill_factor=config.ilu_fill_factor,
        ilu_failure=config.ilu_failure,
        local_solver_backend=config.local_backend,
        assembly_backend=config.assembly_backend,
        boundary_mode=config.boundary_mode,
        verbose=config.verbosity,
    )
    solver = DiffusionReactionHDGSolver(
        space,
        source=source,
        reaction=reaction,
        boundary_condition=exact,
        options=options,
    )
    result = solver.solve()
    l2_error = _summarize_solve(
        result,
        exact,
        diffusion=diffusion,
        preset_key=preset_key,
        case=case,
        mesh=mesh,
        space=space,
        config=config,
    )

    if config.plot:
        from hdgfem.io.plot import plot_solution_comparison

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
