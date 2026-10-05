"""Guiding-center runner helpers."""

from __future__ import annotations
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable
from hybridge.runtime.precision import (
    PRECISION,
    AMGX_MODE,
    KERNEL_AUDIT,
    PIPELINE_AUDIT,
    audit_arrays,
)
from hybridge.core.field_ops import project_callable_to_trace, solution_trace
from hybridge.core.transfer import project_same_mesh_field
from hybridge.diagnostics.guiding_center import transport_velocity_diagnostics
from hybridge.diagnostics.solver import result_transfer_time, solver_result_metrics
from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset
from hybridge.io.records import DiagnosticsRecorder, _json_safe
from scripts.guiding_center.runtime.configuration import (
    _detail_verbosity,
    _is_amgx_solver,
    _make_poisson_options,
    _make_transport_options,
    _phase_verbosity,
    _poisson_postprocess_overrides,
    _transport_trace_basis,
    _validate_config,
)
from scripts.guiding_center.runtime.diagnostics import _compute_diagnostics, _electric_flux
from scripts.guiding_center.runtime.labels import run_label
from scripts.guiding_center.runtime.steppers import make_stepper
from scripts.guiding_center.runtime.models import GuidingCenterRunResult, GuidingCenterStepSnapshot
from scripts.guiding_center.runtime.plotting import _make_plotter, _plot_output_settings
from scripts.guiding_center.runtime.reporting import (
    _print_linear_step_summary,
    _print_run_summary,
    _print_step_summary,
)


def _build_mesh(config: GuidingCenterRunPreset, case):
    from hybridge.core.mesh import (
        gmsh_disc_mesh, gmsh_polygon_mesh, gmsh_rectangle_mesh, gmsh_smooth_star_mesh,
        gmsh_triangle_mesh, rectangle_mesh,
    )

    domain = case.default_domain if config.domain == "auto" else config.domain
    if domain == "structured-rectangle":
        return rectangle_mesh(config.nx, config.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    log_mesh_cache = _phase_verbosity(config) >= 1
    if domain == "iter":
        from hybridge.core.geometry import iter_geometry_path
        from hybridge.core.mesh import gmsh_geo_mesh

        return gmsh_geo_mesh(
            config.mesh_size, path=iter_geometry_path(),
            verbosity=config.gmsh_verbosity, algorithm=config.gmsh_algorithm,
            log_cache=log_mesh_cache,
        )
    if domain in {"horseshoe", "pacman"}:
        from hybridge.core.geometry import shaped_domain

        geometry = shaped_domain(domain, **case.parameters.get("geometry", {}))
        return gmsh_polygon_mesh(
            config.mesh_size, vertices=geometry.vertices,
            verbosity=config.gmsh_verbosity, algorithm=config.gmsh_algorithm,
            log_cache=log_mesh_cache,
        )
    if domain == "rectangle":
        return gmsh_rectangle_mesh(
            config.mesh_size,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            verbosity=config.gmsh_verbosity,
            algorithm=config.gmsh_algorithm,
            log_cache=log_mesh_cache,
        )
    if domain == "disc":
        return gmsh_disc_mesh(
            config.mesh_size,
            center=(0.0, 0.0),
            radius=1.0,
            verbosity=config.gmsh_verbosity,
            algorithm=config.gmsh_algorithm,
            log_cache=log_mesh_cache,
        )
    if domain == "smooth-star":
        return gmsh_smooth_star_mesh(
            config.mesh_size,
            **case.parameters.get("geometry", {}),
            verbosity=config.gmsh_verbosity,
            algorithm=config.gmsh_algorithm,
            log_cache=log_mesh_cache,
        )
    if domain == "triangle":
        return gmsh_triangle_mesh(
            config.mesh_size,
            vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
            verbosity=config.gmsh_verbosity,
            algorithm=config.gmsh_algorithm,
            log_cache=log_mesh_cache,
        )
    raise ValueError(f"unknown domain {domain!r}")


def _initial_projection_backend(config: GuidingCenterRunPreset) -> str:
    """Match initial field projection to the configured assembly backends."""
    device_backends = {"cupy", "raw-cuda"}
    return "cupy" if (
        config.poisson_assembly_backend in device_backends
        or config.transport_assembly_backend in device_backends
    ) else "numpy"


def _project_initial_field(config: GuidingCenterRunPreset, space, function, *, name: str, timings=None):
    """Project on the selected backend, keeping device coefficients resident."""
    from hybridge.core.projection import project_callable

    return project_callable(function, space,
        backend="device" if _initial_projection_backend(config) == "cupy" else "host",
        volume_quad_1d=config.initial_projection_quad_1d, name=name, synchronize=True, timings=timings)



def _report_projection_timings(config, details, total, *, label, prefix):
    """Save detailed attribution and print only the main projection costs."""
    if details is None:
        return
    accounted = sum(value for key, value in details.items() if key.endswith("_time"))
    details["other_python_time"] = max(0.0, total-accounted)
    if "field_evaluation_time" in details:
        setup = sum(details.get(key, 0.0) for key in (
            "reference_setup_time", "backend_import_time",
            "device_initialization_and_pending_work_time",
            "device_mesh_and_reference_setup_time", "projection_operator_setup_time"))
        sampling = details.get("coordinate_mapping_time", 0.0) + details["field_evaluation_time"]
        print(f"[gc:init] projection: total={total:.3f}s | setup={setup:.3f}s | "
              f"sampling={sampling:.3f}s | "
              f"coefficients={details.get('coefficient_projection_time', 0.0):.4f}s",
              flush=True)
    path = Path(config.diagnostics_dir) / f"{prefix}_{label}_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "timing_mode": "synchronized wall time; instrumentation perturbs elapsed time",
        "total_wall_time": total,
        "phases": details,
    }, indent=2) + "\n")


def _solve_transport_stage(
        solver, *, initial_guess, beta, step, time_value, stage, beta_scale,
        failure_path: Path, diagnostics_enabled: bool = True,
):
    """Preserve diagnostics of the actual failed stage, then re-raise its error."""
    from hybridge.linalg.results import LinearSolveConvergenceError

    try:
        return solver.solve(initial_guess=initial_guess)
    except LinearSolveConvergenceError as error:
        if not diagnostics_enabled:
            raise
        report = {
            "step": int(step), "time": float(time_value), "stage": stage,
            "beta_scale": float(beta_scale), "error": str(error),
            "matrix_diagnostics": {
                key: _json_safe(value) for key, value in getattr(error, "matrix_diagnostics", {}).items()
            },
            "attempts": [
                {key: _json_safe(value) for key, value in entry.items()}
                for entry in getattr(error, "amgx_attempts", getattr(error.result, "amgx_attempts", ()))
            ],
        }
        try:
            report["beta_diagnostics"] = {
                key: _json_safe(value) for key, value in transport_velocity_diagnostics(beta).items()
            }
        except Exception as diagnostic_error:
            report["diagnostics_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        save_snapshot = getattr(error, "save_transport_snapshot", None)
        if save_snapshot is not None:
            try:
                snapshot_path = failure_path.with_name(f"{failure_path.stem}_system.npz")
                report.update(save_snapshot(snapshot_path))
                print(f"[gc] failed transport system: {snapshot_path}", flush=True)
            except Exception as diagnostic_error:
                report["snapshot_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        # A diagnostic or filesystem failure must not replace the solve error.
        try:
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            print(f"[gc] failed {stage} diagnostics: {failure_path}", flush=True)
        except Exception as diagnostic_error:
            error.add_note(f"Could not save transport diagnostics: {diagnostic_error}")
        error.transport_diagnostics = report
        raise


def run_guiding_center_case(
        config: GuidingCenterRunPreset,
        *,
        preset_key: str = "custom",
        step_observer: Callable[[GuidingCenterStepSnapshot], None] | None = None,
        terminal_log_path: str | Path | None = None,
) -> GuidingCenterRunResult:
    """Run a fixed-mesh guiding-center case with the selected time scheme."""
    if (config.plot_diagnostics or config.save_diagnostics) and config.case == "diocotron_k":
        config = replace(config, diocotron_diagnostics=True)
    _validate_config(config)
    model_label = run_label(config)
    if config.verbosity >= 1:
        print(f"[gc] {model_label} | p={config.order} | dt={config.dt:g} "
              f"| T={config.dt * config.num_steps:g}", flush=True)

    from hybridge.core.space import DGSpace
    from hybridge.solvers.advection_reaction import AdvectionReactionHDGSolver
    from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGSolver
    from hybridge.runtime.logging import timed_call
    from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key

    case_definition = case_definition_by_key(config.case)
    case = case_definition.build(**config.case_params)
    mesh, _ = timed_call(
        f"generating {case.default_domain if config.domain == 'auto' else config.domain} mesh",
        _phase_verbosity(config),
        lambda: _build_mesh(config, case),
    )
    if mesh.num_tri < config.minimum_triangles:
        raise RuntimeError(
            f"mesh has {mesh.num_tri:,} triangles, below the configured minimum of "
            f"{config.minimum_triangles:,}; reduce mesh_size"
        )
    post_mesh_start = time.perf_counter()
    space, _ = timed_call(
        "[gc:init] building DG space and quadrature data",
        _detail_verbosity(config),
        lambda: DGSpace(
            mesh,
            config.order,
            basis_type=config.basis,
            volume_quadrature=config.volume_quadrature,
            volume_quad_1d=config.volume_quad_1d,
            edge_quad_1d=config.edge_quad_1d,
        ),
    )
    poisson_space = space
    if config.poisson_order_offset:
        poisson_space = DGSpace(
            mesh, config.order + config.poisson_order_offset,
            basis_type=config.basis, volume_quadrature=config.volume_quadrature,
            volume_quad_1d=config.volume_quad_1d, edge_quad_1d=config.edge_quad_1d,
            name="poisson_h",
        )
    if config.diocotron_diagnostics and (case.key != "diocotron_k" or
            (config.domain != "auto" and config.domain != "disc")):
        raise ValueError("diocotron modal diagnostics require diocotron_k on the disk")
    projection_backend = _initial_projection_backend(config)
    projection_label = projection_backend
    if case.parameters.get("initial_profile") == "fft_gaussian":
        nx, ny = case.parameters["fft_grid_shape"]
        projection_label += f", FFT grid {nx}x{ny}"
    elif case.key == "positive_turbulence":
        projection_label += ", direct Gaussian blobs"
    initial_projection_timings = {} if _detail_verbosity(config) else None
    equilibrium_projection_timings = {} if _detail_verbosity(config) else None
    rho_field, initial_density_projection_time = timed_call(
        f"[gc:init] projecting initial density ({projection_label})",
        _detail_verbosity(config),
        lambda: _project_initial_field(config, space, case.initial_density_at(), name="rho_h",
                                       timings=initial_projection_timings),
    )
    _report_projection_timings(config, initial_projection_timings, initial_density_projection_time,
                               label="initial_density_projection", prefix=config.diagnostics_prefix or preset_key)
    positivity = None
    initial_positivity = {}
    if config.positivity_diagnostics:
        from hybridge.diagnostics.guiding_center import ScalarPositivityDiagnostics
        positivity, _ = timed_call("[gc:init] caching positivity diagnostics", _detail_verbosity(config),
            lambda: ScalarPositivityDiagnostics(space,
                backend="device" if projection_backend == "cupy" else "host",
                tolerance=config.positivity_tolerance))
        initial_positivity, _ = timed_call("[gc:init] checking projected density positivity", _detail_verbosity(config),
                                          lambda: positivity.measure(rho_field))
        # Keep the initial projection check even when a later Poisson/trace solve fails.
        projection_path = Path(config.diagnostics_dir) / f"{config.diagnostics_prefix or preset_key}_initial_positivity.json"
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        projection_path.write_text(json.dumps(initial_positivity, indent=2, allow_nan=False) + "\n")
        if _phase_verbosity(config):
            print(f"[gc:init] projected positivity={initial_positivity['positivity_status']} "
                  f"min={initial_positivity['rho_min_checked']:.6e} "
                  f"negative_mass={initial_positivity['rho_negative_mass_quadrature']:.6e}", flush=True)
    solver_data_start = time.perf_counter()
    if _detail_verbosity(config):
        print("[gc:init] preparing solver fields and options ... ", end="", flush=True)
    zero_reaction = poisson_space.zeros(name="zero_reaction_h")
    one_reaction = space.constant(1.0, name="one_reaction_h")
    poisson_options = _make_poisson_options(config)
    transport_boundary_mode = (
        case.density_transport_boundary_mode
        if config.transport_boundary_mode == "auto"
        else config.transport_boundary_mode
    )
    if case.key == "rho_helm_wave" and transport_boundary_mode != "eliminate":
        raise ValueError("rho_helm_wave requires boundary_mode='eliminate' with exact density data")
    transport_options = _make_transport_options(config, transport_boundary_mode)
    if _detail_verbosity(config):
        print(
            f"done in {time.perf_counter() - solver_data_start:.5f}s",
            flush=True,
        )

    equilibrium_potential = None
    equilibrium_density = None
    equilibrium_potential_l2 = None
    equilibrium_density_l2 = None
    poisson_solver = None
    poisson_initial_guess = None
    first_poisson_result = None
    first_poisson_wall_time = 0.0
    equilibrium_density_projection_time = 0.0
    if case.equilibrium_density is not None:
        equilibrium_density, equilibrium_density_projection_time = timed_call(
            f"[gc:init] projecting equilibrium density ({projection_backend})",
            _detail_verbosity(config),
            lambda: _project_initial_field(config, space, case.equilibrium_density, name="rho_eq_h",
                                           timings=equilibrium_projection_timings),
        )
        _report_projection_timings(config, equilibrium_projection_timings, equilibrium_density_projection_time,
                                   label="equilibrium_density_projection", prefix=config.diagnostics_prefix or preset_key)
        equilibrium_solver, _ = timed_call(
            "[gc:init] constructing equilibrium Poisson solver",
            _detail_verbosity(config),
            lambda: DiffusionReactionHDGSolver(
                poisson_space,
                source=project_same_mesh_field(equilibrium_density, poisson_space),
                reaction=zero_reaction,
                boundary_condition=case.potential_boundary_at(0.0),
                options=poisson_options,
            ),
        )
        if _detail_verbosity(config):
            print(
                "[gc:init] mesh-to-first-Poisson setup ... "
                f"done in {time.perf_counter() - post_mesh_start:.5f}s",
                flush=True,
            )
        (equilibrium_result, first_poisson_wall_time) = timed_call(
            "[gc:init] solving first diffusion/Poisson system (equilibrium)",
            _phase_verbosity(config),
            equilibrium_solver.solve,
        )
        first_poisson_result = equilibrium_result
        equilibrium_potential = equilibrium_result.field
        equilibrium_potential_l2 = None
        # Retain the Poisson operator while its coefficients and tau are fixed.
        # Retain the equilibrium solver unconditionally so the perturbed initial
        # state and every accepted step reuse its trace operator, local factors,
        # global factorization/preconditioner, and AMGX hierarchy.
        poisson_solver = equilibrium_solver
        poisson_initial_guess = solution_trace(equilibrium_result, poisson_space, reduced=False)
        # Keep the configured preconditioner policy for every scheme: a tau
        # change invalidates factors/hierarchies through with_options().
        poisson_solver.set_source(project_same_mesh_field(rho_field, poisson_space))
        poisson_solver.set_boundary_condition(case.potential_boundary_at(0.0))

    if poisson_solver is None:
        poisson_solver, _ = timed_call(
            "[gc:init] constructing initial Poisson solver",
            _detail_verbosity(config),
            lambda: DiffusionReactionHDGSolver(
                poisson_space,
                source=project_same_mesh_field(rho_field, poisson_space),
                reaction=zero_reaction,
                boundary_condition=case.potential_boundary_at(0.0),
                options=poisson_options,
            ),
        )
        if _detail_verbosity(config):
            print(
                "[gc:init] mesh-to-first-Poisson setup ... "
                f"done in {time.perf_counter() - post_mesh_start:.5f}s",
                flush=True,
            )
    initial_solve_label = (
        "[gc:init] solving first diffusion/Poisson system"
        if first_poisson_result is None
        else "[gc:init] solving initial-state diffusion/Poisson system (reused operator)"
    )
    (poisson_result, initial_poisson_wall_time) = timed_call(
        initial_solve_label,
        _phase_verbosity(config),
        lambda: poisson_solver.solve(initial_guess=poisson_initial_guess),
    )
    if first_poisson_result is None:
        first_poisson_result = poisson_result
        first_poisson_wall_time = initial_poisson_wall_time
    initialization_poisson_wall_time = initial_poisson_wall_time
    if first_poisson_result is not poisson_result:
        initialization_poisson_wall_time += first_poisson_wall_time
    transport_solver = AdvectionReactionHDGSolver(space, options=transport_options)
    transport_preconditioner_reused = False

    prefer_device_trace = config.transport_assembly_backend == "raw-cuda" and _is_amgx_solver(config.transport_solver)
    density_trace = project_callable_to_trace(
        space,
        case.initial_density_at(),
        trace_basis=_transport_trace_basis(config),
        reduced=True,
        backend="device" if prefer_device_trace else "host",
    )
    potential_trace = solution_trace(poisson_result, poisson_space, reduced=False)
    recovery_log_started = False

    def record_poisson_recovery(event):
        nonlocal recovery_log_started, transport_preconditioner_reused
        if event.get("status") == "retry":
            # Discard a user-requested first-solve preconditioner as well as
            # the cached transport operator; both belong to the old drift.
            transport_solver.with_options(preconditioner=transport_options.preconditioner)
            transport_preconditioner_reused = False
        # Keep events even if the attempt never completes.
        stem = config.diagnostics_prefix or preset_key
        path = Path(config.diagnostics_dir) / f"{stem}_poisson_tau_recovery.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if recovery_log_started else "w") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        recovery_log_started = True

    stage_stepper = make_stepper(
        config, case, space, rho_field, poisson_result, density_trace, potential_trace,
        transport_boundary_mode=transport_boundary_mode, poisson_solver=poisson_solver,
        positivity=positivity, recovery_record=record_poisson_recovery,
    )
    if hasattr(stage_stepper, "initial_poisson_result"):
        poisson_result = stage_stepper.initial_poisson_result
        initialization_poisson_wall_time += stage_stepper.initial_poisson_retry_wall_time
    density_trace, potential_trace = stage_stepper.density_trace, stage_stepper.potential_trace

    modal_diagnostics = None
    if config.diocotron_diagnostics:
        from scripts.guiding_center.diagnostics.diocotron_diagnostics import DiocotronModeDiagnostics
        modal_diagnostics, _ = timed_call("[gc:init] caching polar Fourier diagnostics", _detail_verbosity(config),
            lambda: DiocotronModeDiagnostics(poisson_space, equilibrium_potential,
                mode=int(case.parameters["k"]), inner=float(case.parameters["s_minus"]),
                outer=float(case.parameters["s_plus"]), radial_points=config.diocotron_radial_points,
                angular_points=config.diocotron_angular_points,
                backend="device" if projection_backend == "cupy" else "host"))

    def benchmark_metrics():
        metrics = {"poisson_tau": float(poisson_solver.options.stabilization)}
        if modal_diagnostics is not None:
            values, elapsed = timed_call("[gc] polar Fourier diagnostics", _detail_verbosity(config),
                                         lambda: modal_diagnostics.measure(poisson_result.field))
            metrics.update(values, diocotron_modal_time=elapsed)
        if positivity is not None and config.time_scheme != "imex-ark3":
            metrics.update(positivity.measure(rho_field))
        return metrics

    baseline_mass = None
    baseline_q_l2 = None
    baseline_enstrophy = None
    output_stem = config.diagnostics_prefix or preset_key
    headless_plot, effective_plot_off_screen, effective_screenshot_dir = _plot_output_settings(config, output_stem)
    if headless_plot and _phase_verbosity(config):
        capture = "" if effective_screenshot_dir is None else f" and saving frames to {effective_screenshot_dir}"
        print(f"[gc:init] no display; using {config.plot_backend} off-screen rendering{capture}", flush=True)
    diagnostics_enabled = config.diagnostics_every > 0
    collect_metrics = diagnostics_enabled or config.record_timings or config.verbosity >= 1
    recorder = DiagnosticsRecorder(config.diagnostics_dir, output_stem, enabled=diagnostics_enabled)
    timing_recorder = DiagnosticsRecorder(config.diagnostics_dir, f"{output_stem}_timings",
                                          enabled=config.record_timings)
    plotter = None
    try:
        if collect_metrics:
            initial_extra = solver_result_metrics("poisson", poisson_result)
            initial_extra.update(initial_positivity)
            if initial_projection_timings is not None:
                initial_extra["initial_density_projection_profile"] = initial_projection_timings
            if equilibrium_projection_timings:
                initial_extra["equilibrium_density_projection_profile"] = equilibrium_projection_timings
            initial_extra.update(benchmark_metrics())
            initial_extra["run_configuration"] = dict(case=case.key, case_parameters=case.parameters,
                run_label=model_label,
                preset=preset_key, mesh_size=config.mesh_size, triangles=mesh.num_tri, order=config.order,
                dt=config.dt, time_scheme=config.time_scheme, poisson_tau_initial=config.poisson_tau,
                initial_projection_quad_1d=config.initial_projection_quad_1d,
                volume_quadrature=config.volume_quadrature, volume_quad_1d=config.volume_quad_1d,
                edge_quad_1d=config.edge_quad_1d)
            if config.diocotron_diagnostics:
                from scripts.guiding_center.diagnostics.diocotron_reference import annulus_spectrum
                initial_extra["diocotron_sharp_annulus_reference"] = annulus_spectrum(
                    [int(case.parameters["k"])], inner=float(case.parameters["s_minus"]),
                    outer=float(case.parameters["s_plus"]), density=float(case.parameters["rho_bar"]))[0]
            initial_extra.update(solver_result_metrics("first_poisson", first_poisson_result))
            if config.time_scheme in {"h1-bdf3", "h2-bdf3", "imex-ark3"}:
                initial_extra.update(stage_stepper.initial_recovery_metrics)
                initial_extra.update(time_scheme_setup_time=stage_stepper.setup_time,
                                     hybrid_bdf3_setup_time=(stage_stepper.setup_time if config.time_scheme != "imex-ark3" else 0.),
                                     explicit_residual_setup_time=(stage_stepper.setup_time
                                                                  if stage_stepper.initial_residual_count else 0.0),
                                     explicit_residual_count=stage_stepper.initial_residual_count,
                                     explicit_residual_backend=stage_stepper.residual.backend)
            initial_extra.update(
                {
                    "phase": "initial",
                    "time_scheme": config.time_scheme,
                    "initial_projection_backend": projection_backend,
                    "initial_density_projection_time": initial_density_projection_time,
                    "equilibrium_density_projection_time": equilibrium_density_projection_time,
                    "beta_build_time": 0.0,
                    "poisson_time": initialization_poisson_wall_time,
                    "poisson_time_total": initialization_poisson_wall_time,
                    "first_poisson_wall_time": first_poisson_wall_time,
                    "initial_poisson_wall_time": initial_poisson_wall_time,
                    "initial_poisson_retry_wall_time": getattr(stage_stepper, "initial_poisson_retry_wall_time", 0.),
                    "equilibrium_poisson_wall_time": (
                        first_poisson_wall_time
                        if equilibrium_density is not None
                        else 0.0
                    ),
                    "poisson_flux_postprocessed": poisson_result.postprocessed_flux is not None,
                    "transport_time": 0.0,
                    "transport_time_total": 0.0,
                    "plot_time": 0.0,
                    "plot_backend": config.plot_backend,
                    "plot_headless": headless_plot,
                    "plot_off_screen_effective": effective_plot_off_screen,
                    "plot_screenshot_dir_effective": effective_screenshot_dir,
                }
            )
            timing_recorder.record({"step": 0, "time": 0.0, **initial_extra})
        row = {}
        if diagnostics_enabled:
            row = _compute_diagnostics(
                case=case,
                rho_field=rho_field,
                poisson_result=poisson_result,
                transport_electric_field=config.transport_electric_field,
                step=0,
                time_value=0.0,
                baseline_mass=baseline_mass,
                baseline_q_l2=baseline_q_l2,
                equilibrium_potential=equilibrium_potential,
                equilibrium_density=equilibrium_density,
                equilibrium_potential_l2=equilibrium_potential_l2,
                equilibrium_density_l2=equilibrium_density_l2,
                extra=initial_extra,
            )
            baseline_mass = float(row["mass"])
            baseline_q_l2 = float(row["q_l2"])
            baseline_enstrophy = float(row["enstrophy"])
            equilibrium_potential_l2 = row.get("diocotron_phi_eq_reference_l2")
            equilibrium_density_l2 = row.get("diocotron_rho_eq_reference_l2")
        if config.plot_every > 0:
            plot_start = time.perf_counter()
            plotter = _make_plotter(
                config,
                rho_field,
                poisson_result.field,
                title=model_label,
                off_screen=effective_plot_off_screen,
                screenshot_dir=effective_screenshot_dir,
                screenshot_prefix=config.diagnostics_prefix or preset_key,
                density_is_vorticity=case.density_is_vorticity,
            )
            plotter.update(rho_field, poisson_result.field, step=0, time_value=0.0)
            row["plot_time"] = time.perf_counter() - plot_start
        if diagnostics_enabled:
            recorder.record(row)
            _print_step_summary(config, row)

        current_time = 0.0
        for step in range(1, config.num_steps + 1):
            linear_step_start = time.perf_counter()
            next_time = current_time + config.dt
            endpoint_density_boundary = (
                None if transport_boundary_mode == "zero-flux" else case.density_boundary_at(next_time)
            )
            endpoint_poisson_boundary = case.potential_boundary_at(next_time)

            def solve_stage_transport(source, beta, guess, scale, *, stage_time=None,
                                      stage=None, reuse_operator=False, boundary_condition=...):
                """Apply case boundary/reuse policy around a guiding-center transport stage."""
                nonlocal transport_preconditioner_reused
                stage_time = next_time if stage_time is None else stage_time
                stage_boundary = boundary_condition
                if stage_boundary is Ellipsis:
                    stage_boundary = (None if transport_boundary_mode == "zero-flux" else
                                      case.density_boundary_at(stage_time))
                if reuse_operator:
                    transport_solver.set_source(source, boundary_condition=stage_boundary)
                else:
                    transport_solver.set_problem(source, beta, one_reaction, stage_boundary)
                result = _solve_transport_stage(
                    transport_solver, initial_guess=guess, beta=beta, step=step,
                    time_value=stage_time, stage=stage or config.time_scheme, beta_scale=scale,
                    diagnostics_enabled=diagnostics_enabled,
                    failure_path=Path(config.diagnostics_dir) / f"{output_stem}_transport_failure.json",
                )
                if config.transport_reuse_first_preconditioner and not transport_preconditioner_reused:
                    global_result = result.global_solve_result
                    preconditioner = None if global_result is None else global_result.preconditioner
                    if preconditioner is None:
                        raise RuntimeError("transport reuse requested, but no preconditioner was returned")
                    transport_solver.options = transport_solver.options.with_overrides(preconditioner=preconditioner)
                    transport_preconditioner_reused = True
                return result

            advanced = stage_stepper.advance(
                poisson_solver, solve_stage_transport,
                endpoint_postprocess=_poisson_postprocess_overrides(config, step),
            )
            rho_field, density_trace = advanced.density, advanced.density_trace
            poisson_result, potential_trace = advanced.poisson_result, advanced.potential_trace
            transport_result = advanced.transport_result
            transport_stage_results, poisson_stage_results = advanced.transport_results, advanced.poisson_results
            transport_step_wall_time = advanced.transport_wall_time
            poisson_step_wall_time = advanced.poisson_wall_time
            beta_build_time = advanced.beta_build_time
            potential_trace_time = advanced.potential_trace_time
            transport_time = sum(result.timings.total for result in transport_stage_results)
            poisson_time = sum(result.timings.total for result in poisson_stage_results)
            stage_extra = advanced.metrics
            if collect_metrics:
                for prefix, results in (("poisson", poisson_stage_results), ("transport", transport_stage_results)):
                    for index, result in enumerate(results):
                        stage_extra.update(solver_result_metrics(f"stage{index+1}_{prefix}", result))
            linear_step_end = time.perf_counter()
            post_poisson_start = advanced.post_poisson_start or linear_step_end

            if step_observer is not None:
                is_pc = config.time_scheme == "predictor-corrector"
                step_observer(GuidingCenterStepSnapshot(
                    step=step, time=next_time, space=space,
                    transport_source=advanced.transport_source,
                    transport_beta=advanced.transport_beta,
                    transport_reaction=one_reaction,
                    transport_boundary=advanced.transport_boundary if is_pc else endpoint_density_boundary,
                    transport_initial_guess=advanced.transport_initial_guess,
                    transport_result=transport_result, accepted_density=rho_field,
                    poisson_boundary=endpoint_poisson_boundary,
                    poisson_initial_guess=advanced.poisson_initial_guess,
                    poisson_result=poisson_result,
                    accepted_density_trace_reduced=density_trace,
                    accepted_density_boundary=endpoint_density_boundary,
                ))

            linear_step_wall_time = linear_step_end - linear_step_start
            timing_row = {}
            if collect_metrics:
                stage_extra.setdefault("poisson_predictor_order", 0)
                timing_row = solver_result_metrics("poisson", poisson_result)
                if transport_result is not None:
                    timing_row.update(solver_result_metrics("transport", transport_result))
                timing_row.update(stage_extra)
                for prefix, stage_results in (
                    ("transport", transport_stage_results),
                    ("poisson", poisson_stage_results),
                ):
                    timing_row[f"{prefix}_stage_count"] = len(stage_results)
                    timing_row[f"{prefix}_step_time_assembly"] = sum(
                        result.timings.assembly for result in stage_results
                    )
                    timing_row[f"{prefix}_step_time_solve"] = sum(
                        result.timings.solve for result in stage_results
                    )
                    timing_row[f"{prefix}_step_time_reconstruction"] = sum(
                        result.timings.reconstruction for result in stage_results
                    )
                    timing_row[f"{prefix}_step_iterations"] = sum(
                        int(result.global_solve_result.iteration_count or 0)
                        for result in stage_results
                        if result.global_solve_result is not None
                    )
                timing_row.update(
                    {
                        "step": step,
                        "time": next_time,
                        "phase": "step",
                        "time_scheme": config.time_scheme,
                        "beta_build_time": beta_build_time,
                        "linear_step_wall_time": linear_step_wall_time,
                        "transport_step_wall_time": transport_step_wall_time,
                        "poisson_step_wall_time": poisson_step_wall_time,
                        "poisson_time": poisson_time,
                        "poisson_flux_postprocessed": poisson_result.postprocessed_flux is not None,
                        "poisson_flux_postprocess_time": poisson_result.timings.postprocessing,
                        "transport_time": transport_time,
                        "potential_trace_update_time": potential_trace_time,
                    }
                )
                timing_recorder.record(timing_row)
                _print_linear_step_summary(config, timing_row)

            current_time = next_time
            should_plot = config.plot_every > 0 and step % config.plot_every == 0
            should_record = diagnostics_enabled and (
                step % config.diagnostics_every == 0 or step == config.num_steps)
            plot_elapsed = 0.0
            if should_plot:
                if _detail_verbosity(config):
                    print(f"[gc] updating {config.plot_backend} plot ... ", end="", flush=True)
                plot_start = time.perf_counter()
                if plotter is None:
                    plotter = _make_plotter(
                        config,
                        rho_field,
                        poisson_result.field,
                        title=model_label,
                        off_screen=effective_plot_off_screen,
                        screenshot_dir=effective_screenshot_dir,
                        screenshot_prefix=config.diagnostics_prefix or preset_key,
                        density_is_vorticity=case.density_is_vorticity,
                    )
                plotter.update(rho_field, poisson_result.field, step=step, time_value=current_time)
                plot_elapsed = time.perf_counter() - plot_start
                if _detail_verbosity(config):
                    print(f"done in {plot_elapsed:.5f}s", flush=True)

            if should_record:
                extra = solver_result_metrics("poisson", poisson_result)
                if transport_result is not None:
                    extra.update(solver_result_metrics("transport", transport_result))
                extra.update(stage_extra)
                extra.update(timing_row)
                extra.update(
                    {
                        "phase": "step",
                        "time_scheme": config.time_scheme,
                        "beta_build_time": beta_build_time,
                        "poisson_time": poisson_time,
                        "transport_time": transport_time,
                        "host_device_transfer_time": sum(
                            result_transfer_time(result)
                            for result in (*transport_stage_results, *poisson_stage_results)
                        ),
                        "plot_time": plot_elapsed,
                        "potential_trace_update_time": potential_trace_time,
                    }
                )
                extra.update(benchmark_metrics())
                row, diagnostics_wall_time = timed_call(
                    "[gc] computing accepted-step diagnostics",
                    _detail_verbosity(config),
                    lambda: _compute_diagnostics(
                        case=case,
                        rho_field=rho_field,
                        poisson_result=poisson_result,
                        transport_electric_field=config.transport_electric_field,
                        step=step,
                        time_value=current_time,
                        baseline_mass=baseline_mass,
                        baseline_q_l2=baseline_q_l2,
                        baseline_enstrophy=baseline_enstrophy,
                        equilibrium_potential=equilibrium_potential,
                        equilibrium_density=equilibrium_density,
                        equilibrium_potential_l2=equilibrium_potential_l2,
                        equilibrium_density_l2=equilibrium_density_l2,
                        extra=extra,
                    ),
                )
                row["diagnostics_wall_time"] = diagnostics_wall_time
                row["post_poisson_application_time"] = (
                    time.perf_counter() - post_poisson_start
                    if post_poisson_start is not None
                    else potential_trace_time + plot_elapsed + diagnostics_wall_time
                )
                timed_call(
                    "[gc] writing diagnostics JSONL",
                    _detail_verbosity(config),
                    lambda: recorder.record(row),
                )
                _print_step_summary(config, row)
    finally:
        import sys
        active_failure = sys.exc_info()[0] is not None
        try:
            # Persist the CSVs before anything that can wait for a GUI.
            try:
                recorder.close()
            finally:
                timing_recorder.close()
            if active_failure:
                if recorder.enabled:
                    print(f"[gc] partial diagnostics: {recorder.jsonl_path}", flush=True)
                if timing_recorder.enabled:
                    print(f"[gc] partial timings: {timing_recorder.jsonl_path}", flush=True)
            else:
                config = replace(config, poisson_tau=float(poisson_solver.options.stabilization))
                result = GuidingCenterRunResult(
                    config=config,
                    preset_key=preset_key,
                    case_key=case.key,
                    mesh=mesh,
                    space=space,
                    final_density=rho_field,
                    final_potential=poisson_result.field,
                    final_flux=_electric_flux(poisson_result),
                    final_density_trace_reduced=density_trace,
                    final_potential_trace_reduced=solution_trace(poisson_result, poisson_space, reduced=True),
                    diagnostics=recorder.rows,
                    csv_path=recorder.csv_path,
                    jsonl_path=recorder.jsonl_path,
                    timings_csv_path=timing_recorder.csv_path,
                    timings_jsonl_path=timing_recorder.jsonl_path,
                    terminal_log_path=(
                        None if terminal_log_path is None else Path(terminal_log_path)
                    ),
                )
                audit_arrays("guiding-center-final-state", result)
                if PRECISION == "float32" and diagnostics_enabled:
                    precision_report = {
                        "precision": PRECISION, "amgx_mode": AMGX_MODE,
                        "transport_amgx_config": transport_solver.options.amgx_config,
                        "transport_amgx_config_path": config.transport_amgx_config_path,
                        "kernels": KERNEL_AUDIT, "pipeline": PIPELINE_AUDIT,
                        "tolerances": {
                            "poisson_rtol": config.poisson_solver_rtol,
                            "poisson_atol": config.poisson_solver_atol,
                            "transport_rtol": config.transport_solver_rtol,
                            "transport_atol": config.transport_solver_atol,
                            "transport_amgx_tolerance": config.transport_amgx_tolerance,
                        },
                    }
                    report_path = Path(config.diagnostics_dir) / f"{output_stem}_precision.json"
                    report_path.write_text(json.dumps(precision_report, indent=2))
                _print_run_summary(result)
                # Flush the summary before GUI backends can block or abort.
                sys.stdout.flush()
        finally:
            if plotter is not None:
                try:
                    plotter.close()
                except Exception as error:
                    print(f"[gc] plot cleanup failed: {error}; "
                          "recorded numerical results are retained", flush=True)
            if config.plot_diagnostics or config.save_diagnostics:
                from scripts.guiding_center.runtime.diagnostic_plots import render_guiding_center_diagnostic_plots

                try:
                    render_guiding_center_diagnostic_plots(
                        config, recorder.rows, timing_recorder.rows, prefix=output_stem)
                except Exception as error:
                    # Presentation failures must not replace a numerical
                    # exception or erase successful numerical completion.
                    print(f"[gc] diagnostic plotting failed: {error}; "
                          f"recorded data: {recorder.jsonl_path}", flush=True)

    return result
