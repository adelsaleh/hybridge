#!/usr/bin/env python3
"""Run manufactured diffusion-reaction presets.

Preset definitions live in this file, in the ``PRESETS`` dictionary below.

To create a new manufactured test:
1. Add a factory in ``scripts/diff_rea_cases.py`` that returns
   manufactured diffusion, reaction, source, exact solution, and exact flux data.
2. Register it in ``CASE_DEFINITIONS`` in that same file.
3. Add one or more ``DiffusionReactionRunPreset`` entries in ``PRESETS`` below.

To change mesh size, polynomial order, stabilization, quadrature, solver, PETSc
settings, HDG post-processing, plotting, or case parameters such as tensor-sine
``m``/``n``, edit the corresponding preset here.
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
    hdg_postprocess: str = "both"
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
    "trigonometric_poisson_cg_gamg": DiffusionReactionRunPreset(
        case="trigonometric-poisson",
        description="Legacy trigonometric pure-Poisson case on its default disk domain.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="cg_gamg",
        boundary_mode="eliminate",
        assembly_backend="numba",
        petsc_levels=10,
        order=6,
        verbosity=2,
        mesh_size=0.1,
    ),
    "trigonometric_poisson_cg_hypre": DiffusionReactionRunPreset(
        case="trigonometric-poisson",
        description="Trigonometric pure-Poisson PETSc CG with Hypre BoomerAMG.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="cg_hypre",
        boundary_mode="eliminate",
        assembly_backend="numba",
        order=6,
        verbosity=2,
        mesh_size=0.1,
    ),
    "trigonometric_poisson_cg_icc": DiffusionReactionRunPreset(
        case="trigonometric-poisson",
        description="Trigonometric pure-Poisson PETSc CG with ICC.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="cg_icc",
        boundary_mode="eliminate",
        assembly_backend="numba",
        order=6,
        verbosity=2,
        mesh_size=0.1,
    ),
    "trigonometric_poisson_cg_ilu": DiffusionReactionRunPreset(
        case="trigonometric-poisson",
        description="Trigonometric pure-Poisson PETSc CG with ILU.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="cg_ilu",
        boundary_mode="eliminate",
        assembly_backend="numba",
        order=6,
        verbosity=2,
        mesh_size=0.1,
    ),
    "trigonometric_poisson_lu": DiffusionReactionRunPreset(
        case="trigonometric-poisson",
        description="Trigonometric pure-Poisson PETSc direct LU baseline.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="lu",
        boundary_mode="eliminate",
        assembly_backend="numba",
        order=6,
        verbosity=2,
        mesh_size=0.1,
    ),
    "trigonometric_poisson_mumps_lu": DiffusionReactionRunPreset(
        case="trigonometric-poisson",
        description="Trigonometric pure-Poisson PETSc direct LU with MUMPS.",
        solver="petsc",
        preconditioner=None,
        petsc_preset="mumps_lu",
        boundary_mode="eliminate",
        assembly_backend="numba",
        order=6,
        verbosity=2,
        mesh_size=0.1,
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
        nx=50,
        ny=50,
        order=6,
        tau=4.0,
        assembly_backend="numba",
        solver="petsc",
        preconditioner=None,
        petsc_preset="cg_hypre",
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
    from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_lshape_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, \
        rectangle_mesh

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


def _timing_with_percent(seconds: float, total: float, *, precision: int = 1) -> str:
    """Format elapsed seconds with its percentage of total runtime."""
    percent = 0.0 if total <= 0.0 else 100.0 * float(seconds) / float(total)
    return f"{float(seconds):.{precision}f} ({percent:.1f}%)"


def _summarize_solve(
        result,
        exact,
        exact_flux,
        *,
        diffusion,
        preset_key: str,
        case,
        mesh,
        space,
        config: DiffusionReactionRunPreset,
) -> float:
    import numpy as np
    from hdgfem.io.output import pretty_print_sections
    from hdgfem.solvers.diff_rea import _diffusion_is_identity

    l2_error = result.field.l2_error(exact)
    flux_l2_error = _vector_l2_error(result.flux, exact_flux)
    post_primal_l2_error = (
        np.nan
        if result.postprocessed_field is None
        else result.postprocessed_field.l2_error(exact)
    )
    post_flux_l2_error = (
        np.nan
        if result.postprocessed_flux is None
        else _vector_l2_error(result.postprocessed_flux, exact_flux)
    )
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
    option_items = [
        ("assembly backend", result.assembly_backend, "s"),
        ("local backend",
         "fused" if result.assembly_backend == "numba" and result.local_solver is None else config.local_backend, "s"),
        ("boundary mode", result.boundary_mode, "s"),
        ("postprocess", config.hdg_postprocess, "s"),
        ("diffusion", "identity" if _diffusion_is_identity(diffusion) else "tensor", "s"),
        ("tau", config.tau, ".3e"),
    ]
    solver_items = [
        ("solver", "none" if config.solver is None else config.solver, "s"),
        ("preconditioner", "petsc" if str(config.solver).lower() == "petsc" else config.preconditioner or "none", "s"),
        ("scaling", "left" if result.scale_system else "none", "s"),
    ]
    error_items = [
        ("theoretical h^(p+1)", mesh.h ** (space.order + 1), ".4e"),
        ("primal L2 error", l2_error, ".4e"),
        ("flux L2 error", flux_l2_error, ".4e"),
        ("post primal L2 error", post_primal_l2_error, ".4e"),
        ("post flux L2 error", post_flux_l2_error, ".4e"),
        ("Linf error", linfty_error, ".4e"),
        ("avg max error", avg_error, ".4e"),
        ("max-error element", max_error_element, "d"),
    ]
    total_time = result.timings.total
    timing_items = [
        ("assembly (s)", _timing_with_percent(result.timings.assembly, total_time), "s"),
        ("global solve (s)", _timing_with_percent(result.timings.solve, total_time), "s"),
        ("reconstruct (s)", _timing_with_percent(result.timings.reconstruction, total_time), "s"),
        ("postprocess (s)", _timing_with_percent(result.timings.postprocessing, total_time), "s"),
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
        if free_trace_relative_residual is None and result.boundary_mode == "eliminate":
            free_trace_relative_residual = global_solve.solver_relative_residual_norm
        solver_items.extend(
            [
                ("Krylov iterations", -1 if global_solve.iteration_count is None else global_solve.iteration_count, ",d"),
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
        title="Diffusion-Reaction Preset Solve Summary",
    )
    return l2_error


def _dense_exact_plot_resolution(
        exact_resolution: int | str | None,
        *,
        numerical_resolution: int,
        num_elements: int,
) -> int | str | None:
    """Choose a dense exact-panel resolution while bounding memory use.

    Exact callables are cheap to sample and should look smooth in comparison
    plots, but Matplotlib contouring becomes expensive for very dense triangular
    refinements.  Small meshes use a higher default than DG panels, capped by a
    total point budget.
    """
    if exact_resolution is not None:
        return exact_resolution
    if int(num_elements) <= 100:
        target = max(4 * int(numerical_resolution), 80)
    else:
        target = max(2 * int(numerical_resolution), int(numerical_resolution) + 20)
    max_total_points = 15_000_000
    while target > 2 and num_elements * target * (target + 1) // 2 > max_total_points:
        target -= 1
    return target


def _polynomial_plot_resolution(requested_resolution: int | None, order: int) -> int:
    """Choose a per-element plotting grid dense enough for degree-``order`` fields."""
    minimum = max(3, 2 * int(order) + 3)
    if requested_resolution is None:
        return max(20, minimum)
    return max(int(requested_resolution), minimum)


def _exact_centered_clim(
        exact_values,
        *comparison_values,
        relative_padding: float = 0.04,
        max_relative_expansion: float = 0.15,
) -> tuple[float, float]:
    """Return exact-dominated color limits with capped numerical expansion.

    The exact solution determines the dominant color scale.  Numerical and
    postprocessed values may expand the limits to avoid clipping moderate
    overshoot, but only up to ``max_relative_expansion`` of the exact range.
    """
    import numpy as np

    exact_min = float(np.nanmin(exact_values))
    exact_max = float(np.nanmax(exact_values))
    if not np.isfinite(exact_min) or not np.isfinite(exact_max):
        return 0.0, 1.0
    if exact_min == exact_max:
        exact_max = exact_min + 1.0

    span = exact_max - exact_min
    padding = relative_padding * span
    lower = exact_min - padding
    upper = exact_max + padding
    lower_cap = exact_min - max_relative_expansion * span
    upper_cap = exact_max + max_relative_expansion * span
    for values in comparison_values:
        values_min = float(np.nanmin(values))
        values_max = float(np.nanmax(values))
        if np.isfinite(values_min):
            lower = max(lower_cap, min(lower, values_min - padding))
        if np.isfinite(values_max):
            upper = min(upper_cap, max(upper, values_max + padding))
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def _normalize_quadrature_values(values, target_shape: tuple[int, int]):
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.shape == target_shape:
        return array
    if array.ndim == 0:
        return np.full(target_shape, float(array), dtype=np.float64)
    try:
        return np.asarray(np.broadcast_to(array, target_shape), dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"values must broadcast to {target_shape}; got {array.shape}") from exc


def _exact_flux_values(exact_flux, points):
    import numpy as np

    raw_values = exact_flux(points[:, :, 0], points[:, :, 1])
    target_shape = points.shape[:2]
    if isinstance(raw_values, tuple | list):
        if len(raw_values) != 2:
            raise ValueError(f"exact flux must have two components; got {len(raw_values)}")
        qx, qy = raw_values
    else:
        raw_array = np.asarray(raw_values, dtype=np.float64)
        if raw_array.shape[:1] != (2,):
            raise ValueError("exact flux must return a pair of components or an array with leading dimension 2")
        qx, qy = raw_array[0], raw_array[1]
    return np.stack(
        (
            _normalize_quadrature_values(qx, target_shape),
            _normalize_quadrature_values(qy, target_shape),
        ),
        axis=0,
    )


def _vector_l2_error(vector_field, exact_flux) -> float:
    import numpy as np

    if vector_field.dim != 2:
        raise ValueError(f"expected a two-component flux field; got {vector_field.dim}")
    space = vector_field.components[0].space
    points = space.mapped_quads()
    numerical_values = vector_field.values()
    exact_values = _exact_flux_values(exact_flux, points)
    diff = numerical_values - exact_values
    return float(
        np.sqrt(
            np.einsum(
                "K,dKq,q->",
                space.mesh.aff_jacs,
                diff * diff,
                space.quad_data.Krf_w,
                optimize=True,
            )
        )
    )


def _plot_primal_postprocess_comparison(
        result,
        exact,
        *,
        resolution: int,
        exact_resolution: int | str | None,
        suptitle: str,
        hdg_title: str,
        post_title: str,
        show_mesh: bool,
):
    """Plot HDG, postprocessed primal, and exact scalar panels.

    Meshes with at most 100 triangles use the Matplotlib discontinuous contour
    helper for high-detail per-element inspection.  Larger meshes use the
    PyVista refined-mesh path, which is more responsive for larger point sets.
    """
    from hdgfem.io.plot import (
        _require_pyvista,
        _resolve_exact_plot_resolution,
        add_field_to_plotter,
        add_samples_to_plotter,
        sample_callable_on_elements,
        sample_field_on_elements,
    )

    if result.field.space.mesh.num_tri <= 100:
        return _plot_primal_postprocess_comparison_matplotlib(
            result,
            exact,
            resolution=resolution,
            exact_resolution=exact_resolution,
            suptitle=suptitle,
            hdg_title=hdg_title,
            post_title=post_title,
            show_mesh=show_mesh,
        )

    pv = _require_pyvista()
    reference_points, _, primal_values = sample_field_on_elements(
        result.field,
        resolution=resolution,
    )
    postprocessed_field = result.postprocessed_field if result.postprocessed_field is not None else result.field
    postprocessed_values = postprocessed_field.values_at_ref(reference_points)
    exact_panel_resolution = _resolve_exact_plot_resolution(
        exact_resolution,
        numerical_resolution=resolution,
        num_elements=result.field.space.mesh.num_tri,
    )
    exact_reference_points, _, exact_values = sample_callable_on_elements(
        result.field.space.mesh,
        exact,
        resolution=exact_panel_resolution,
    )

    shared_clim = _exact_centered_clim(exact_values, primal_values, postprocessed_values)

    plotter = pv.Plotter(shape=(1, 3), window_size=[1800, 650])
    scalar_bar_args = {
        "vertical": False,
        "width": 0.55,
        "height": 0.08,
        "position_x": 0.225,
        "position_y": 0.02,
    }
    panels = (
        (hdg_title, "primal", reference_points, primal_values, result.field),
        (post_title, "post_primal", reference_points, postprocessed_values, postprocessed_field),
        ("Exact solution", "exact", exact_reference_points, exact_values, None),
    )
    for column, (panel_title, scalar_name, panel_reference_points, values, field) in enumerate(panels):
        if field is None:
            add_samples_to_plotter(
                plotter,
                result.field.space.mesh,
                panel_reference_points,
                values,
                scalar_name=scalar_name,
                title=None,
                subplot=(0, column),
                show_mesh=show_mesh,
                cmap="viridis",
                clim=shared_clim,
                scalar_bar_args=scalar_bar_args,
            )
        else:
            add_field_to_plotter(
                plotter,
                field,
                reference_points=panel_reference_points,
                values=values,
                scalar_name=scalar_name,
                title=None,
                subplot=(0, column),
                show_mesh=show_mesh,
                cmap="viridis",
                clim=shared_clim,
                scalar_bar_args=scalar_bar_args,
            )
        plotter.add_text(panel_title, position="upper_left", font_size=10, shadow=False)
    if suptitle:
        plotter.subplot(0, 1)
        plotter.add_title(suptitle, font_size=14, shadow=False)
    plotter.link_views()
    plotter.show()
    return plotter


def _plot_primal_postprocess_comparison_matplotlib(
        result,
        exact,
        *,
        resolution: int,
        exact_resolution: int | str | None,
        suptitle: str,
        hdg_title: str,
        post_title: str,
        show_mesh: bool,
):
    from hdgfem.io.plot import (
        _resolve_exact_plot_resolution,
        plot_scalar_sample_panels_matplotlib,
        sample_callable_on_elements,
        sample_field_on_elements,
    )

    mesh = result.field.space.mesh
    reference_points, _, primal_values = sample_field_on_elements(
        result.field,
        resolution=resolution,
    )
    postprocessed_field = result.postprocessed_field if result.postprocessed_field is not None else result.field
    postprocessed_values = postprocessed_field.values_at_ref(reference_points)
    exact_panel_resolution = _resolve_exact_plot_resolution(
        exact_resolution,
        numerical_resolution=resolution,
        num_elements=mesh.num_tri,
    )
    exact_reference_points, _, exact_values = sample_callable_on_elements(
        mesh,
        exact,
        resolution=exact_panel_resolution,
    )
    shared_clim = _exact_centered_clim(exact_values, primal_values, postprocessed_values)
    return plot_scalar_sample_panels_matplotlib(
        mesh,
        (
            (hdg_title, reference_points, primal_values),
            (post_title, reference_points, postprocessed_values),
            ("Exact solution", exact_reference_points, exact_values),
        ),
        suptitle=suptitle,
        show_mesh=show_mesh,
        cmap="jet",
        levels=128,
        clim=shared_clim,
        share_clim=True,
    )


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
    parser.add_argument("--verbosity", "-v", type=int, default=None,
                        help="override logging verbosity for this run only")
    parser.add_argument("--quiet", action="store_true", help="run with verbosity 0 for this run only")
    parser.add_argument("--plot", action="store_true", help="show HDG/postprocessed/exact plots for this run only")
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
    problem = case.build(**config.case_params)
    diffusion, reaction, source, exact = problem
    exact_flux = problem.exact_flux
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
        hdg_postprocess=config.hdg_postprocess,
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
        exact_flux,
        diffusion=diffusion,
        preset_key=preset_key,
        case=case,
        mesh=mesh,
        space=space,
        config=config,
    )

    if config.plot:
        plot_resolution = _polynomial_plot_resolution(config.plot_resolution, space.order + 1)
        post_primal_l2_error = (
            None if result.postprocessed_field is None else result.postprocessed_field.l2_error(exact)
        )
        suptitle = f"{case.name} - {preset_key}"
        hdg_title = f"HDG solution\np={space.order}, elements={mesh.num_tri:,}, L2={l2_error:.2e}"
        post_title = (
            "Postprocessed primal"
            if post_primal_l2_error is None
            else f"Postprocessed primal\nL2={post_primal_l2_error:.2e}"
        )
        _plot_primal_postprocess_comparison(
            result,
            exact,
            resolution=plot_resolution,
            exact_resolution=_dense_exact_plot_resolution(
                config.exact_plot_resolution,
                numerical_resolution=plot_resolution,
                num_elements=mesh.num_tri,
            ),
            suptitle=suptitle,
            hdg_title=hdg_title,
            post_title=post_title,
            show_mesh=not config.hide_mesh,
        )

if __name__ == "__main__":
    _main()
