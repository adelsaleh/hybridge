"""Guiding-center configuration helpers."""

from __future__ import annotations
import ast
import copy
import math
from dataclasses import replace
from pathlib import Path
from typing import Any
from scripts.guiding_center.time_schemes import STEPPERS
from hdgfem.runtime.precision import PRECISION
from hdgfem.io.config import load_amgx_config, with_amgx_residual_history
from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset


_AMGX_DEFAULT_RTOL = 1.0e-11
_AMGX_DEFAULT_ATOL = 1.0e-12


def _load_amgx_config(
        path: str | None,
        *,
        tolerance: float | None = None,
        absolute_tolerance: float | None = None,
) -> dict | None:
    if path is None:
        return None
    config, _ = load_amgx_config(path)
    solver = config.setdefault("solver", {})
    convergence = str(solver.get("convergence", "")).upper()
    if convergence == "ABSOLUTE" and absolute_tolerance is not None:
        solver["tolerance"] = float(absolute_tolerance)
    elif tolerance is not None:
        solver["tolerance"] = float(tolerance)
    return config


def _verbosity_level(config: GuidingCenterRunPreset) -> int:
    return max(0, min(3, int(config.verbosity)))


def _solver_verbosity(config: GuidingCenterRunPreset) -> int:
    """Map runner verbosity to compact solver logging levels."""
    level = _verbosity_level(config)
    return 3 if level >= 3 else max(0, level - 1)


def _phase_verbosity(config: GuidingCenterRunPreset) -> int:
    return 1 if _verbosity_level(config) >= 2 else 0


def _detail_verbosity(config: GuidingCenterRunPreset) -> int:
    return 1 if _verbosity_level(config) >= 3 else 0


def _parse_case_param(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"case parameter {raw!r} must use key=value syntax")
    key, value_text = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("case parameter key cannot be empty")
    try:
        value = ast.literal_eval(value_text)
    except (ValueError, SyntaxError):
        lowered = value_text.strip().lower()
        if lowered == "true":
            value = True
        elif lowered == "false":
            value = False
        elif lowered == "none":
            value = None
        else:
            value = value_text
    return key, value


def _backend_profile_updates(profile: str) -> dict[str, Any]:
    if profile == "host":
        return {
            "poisson_assembly_backend": "numba",
            "poisson_solver": "BICGSTAB",
            "poisson_preconditioner": "ilu",
            "poisson_scale_system": True,
            "poisson_retry_policy": "none",
            "poisson_fb_hp_mg_preconditioner_policy": "standard",
            "transport_assembly_backend": "numba",
            "transport_solver": "BICGSTAB",
            "transport_preconditioner": "ilu",
            "transport_scale_system": True,
        }
    if profile == "hybrid":
        return {
            "poisson_assembly_backend": "numba",
            "poisson_solver": "amgx",
            "poisson_preconditioner": None,
            "poisson_solver_rtol": _AMGX_DEFAULT_RTOL,
            "poisson_solver_atol": _AMGX_DEFAULT_ATOL,
            "poisson_scale_system": False,
            "poisson_retry_policy": "none",
            "poisson_fb_hp_mg_preconditioner_policy": "standard",
            "transport_assembly_backend": "numba",
            "transport_solver": "amgx",
            "transport_preconditioner": None,
            "transport_solver_rtol": _AMGX_DEFAULT_RTOL,
            "transport_solver_atol": _AMGX_DEFAULT_ATOL,
            "transport_scale_system": True,
        }
    if profile == "device":
        return {
            "poisson_assembly_backend": "raw-cuda",
            "poisson_solver": "fb-hp-mg-pcg",
            "poisson_preconditioner": None,
            "poisson_solver_rtol": _AMGX_DEFAULT_RTOL,
            "poisson_solver_atol": _AMGX_DEFAULT_ATOL,
            "poisson_scale_system": False,
            "poisson_trace_basis": "legendre-modal",
            "poisson_raw_matrix_format": "bsr",
            "poisson_cache_local_factors": "schur-cholesky",
            "transport_assembly_backend": "raw-cuda",
            "transport_solver": "amgx",
            "transport_preconditioner": None,
            "transport_solver_rtol": _AMGX_DEFAULT_RTOL,
            "transport_solver_atol": _AMGX_DEFAULT_ATOL,
            "transport_scale_system": True,
            "transport_trace_basis": "legacy-lagrange",
            "transport_raw_local_assembly": "fused",
            "transport_raw_matrix_format": "bsr",
            "transport_initial_guess": "initial-density-trace",
            "transport_materialize_host_solution": False,
        }
    raise ValueError(f"unknown backend profile {profile!r}")


def _scale_choice(value: str | None, *, poisson: bool) -> bool | None:
    if value is None:
        return None
    if value == "on":
        return True
    if value == "off":
        return False
    if value == "auto":
        return True if poisson else None
    raise ValueError(f"unknown scale choice {value!r}")


def _materialize_choice(value: str | None) -> bool | None:
    if value is None:
        return None
    if value == "on":
        return True
    if value == "off":
        return False
    if value == "auto":
        return None
    raise ValueError(f"unknown materialize choice {value!r}")


def _is_amgx_solver(solver: str | None) -> bool:
    if solver is None:
        return False
    return str(solver).lower() in {"amgx", "pyamgx"}


def _poisson_trace_basis(config: GuidingCenterRunPreset) -> str:
    """Return the Poisson trace basis with legacy shared-setting fallback."""
    return config.poisson_trace_basis or config.trace_basis


def _transport_trace_basis(config: GuidingCenterRunPreset) -> str:
    """Return the transport trace basis with legacy shared-setting fallback."""
    return config.transport_trace_basis or config.trace_basis


def _poisson_postprocess_overrides(
        config: GuidingCenterRunPreset, step: int,
) -> dict[str, Any]:
    """Return per-call accepted-state flux postprocessing controls."""
    cadence = int(config.poisson_flux_postprocess_every)
    if cadence <= 0 or int(step) <= 0 or int(step) % cadence:
        return {}
    return {
        "postprocess_overrides": {
            "hdg_postprocess": "flux",
            "flux_postprocess_space": config.poisson_flux_postprocess_space,
            "postprocessing_backend": config.poisson_postprocessing_backend,
        }
    }


def _with_fp32_transport_solver(config: GuidingCenterRunPreset) -> GuidingCenterRunPreset:
    """Use FGMRES for the stock FP32 transport profile that BiCGSTAB destabilizes."""
    if PRECISION != "float32" or not _is_amgx_solver(config.transport_solver):
        return config
    config_dir = Path(__file__).resolve().parents[3] / "configs" / "amgx"
    stock_bicgstab = config_dir / "adv_rea_gpu4_hdg_bicgstab_scaled_none.json"
    selected_path = config.transport_amgx_config_path
    if selected_path is None or Path(selected_path).resolve() != stock_bicgstab:
        return config
    return replace(
        config,
        transport_amgx_config_path=str(config_dir / "adv_rea_gpu4_hdg_fgmres_scaled_none.json"),
    )


def _runtime_config(config: GuidingCenterRunPreset, args) -> GuidingCenterRunPreset:
    if PRECISION == "float32":
        config = replace(
            config,
            poisson_solver_rtol=2.0e-3, poisson_solver_atol=0.0,
            transport_solver_rtol=5.0e-3, transport_solver_atol=0.0,
            transport_amgx_tolerance=5.0e-3,
            poisson_maxiter=500, transport_maxiter=300,
        )
    updates: dict[str, Any] = {}
    if args.backend_profile is not None:
        updates.update(_backend_profile_updates(args.backend_profile))
    direct_updates = {
        "plot_diagnostics": getattr(args, "plot_diagnostics", None),
        "save_diagnostics": getattr(args, "save_diagnostics", None),
        "case": args.case,
        "domain": args.domain,
        "mesh_size": args.mesh_size,
        "minimum_triangles": args.minimum_triangles,
        "nx": args.nx,
        "ny": args.ny,
        "gmsh_verbosity": args.gmsh_verbosity,
        "gmsh_algorithm": args.gmsh_algorithm,
        "basis": args.basis,
        "trace_basis": args.trace_basis,
        "poisson_trace_basis": args.poisson_trace_basis,
        "transport_trace_basis": args.transport_trace_basis,
        "order": args.order,
        "volume_quadrature": args.volume_quadrature,
        "initial_projection_quad_1d": getattr(args,"initial_projection_quad_1d",None),
        "volume_quad_1d": args.volume_quad_1d,
        "edge_quad_1d": args.edge_quad_1d,
        "dt": args.dt,
        "num_steps": args.num_steps,
        "time_scheme": args.time_scheme,
        "h1_startup": getattr(args, "h1_startup", None),
        "h2_startup": getattr(args, "h2_startup", None),
        "poisson_tau": args.poisson_tau,
        "poisson_tau_retry_factor": getattr(args, "poisson_tau_retry_factor", None),
        "poisson_tau_max_retries": getattr(args, "poisson_tau_max_retries", None),
        "poisson_assembly_backend": args.poisson_assembly_backend,
        "poisson_local_backend": args.poisson_local_backend,
        "poisson_solver": args.poisson_solver,
        "poisson_preconditioner": args.poisson_preconditioner,
        "poisson_solver_rtol": args.poisson_solver_rtol,
        "poisson_solver_atol": args.poisson_solver_atol,
        "poisson_maxiter": args.poisson_maxiter,
        "poisson_petsc_preset": args.poisson_petsc_preset,
        "poisson_petsc_levels": args.poisson_petsc_levels,
        "poisson_cupyx_solver": args.poisson_cupyx_solver,
        "poisson_amgx_config_path": None if args.poisson_amgx_config is None else str(args.poisson_amgx_config),
        "poisson_retry_policy": args.poisson_retry_policy,
        "poisson_retry_amgx_config_path": (
            None
            if args.poisson_retry_amgx_config is None
            else str(args.poisson_retry_amgx_config)
        ),
        "poisson_fb_hp_mg_preconditioner_policy": (
            args.poisson_fb_hp_mg_preconditioner_policy
        ),
        "poisson_ilu_drop_tol": args.poisson_ilu_drop_tol,
        "poisson_ilu_fill_factor": args.poisson_ilu_fill_factor,
        "poisson_ilu_permc_spec": args.poisson_ilu_permc_spec,
        "poisson_raw_matrix_format": args.poisson_raw_matrix_format,
        "poisson_raw_block_size": args.poisson_raw_block_size,
        "poisson_cache_local_factors": args.poisson_cache_local_factors,
        "poisson_order_offset": args.poisson_order_offset,
        "transport_electric_field": args.transport_electric_field,
        "poisson_hdg_postprocess": args.poisson_hdg_postprocess,
        "poisson_flux_postprocess_every": args.poisson_flux_postprocess_every,
        "poisson_flux_postprocess_space": args.poisson_flux_postprocess_space,
        "poisson_postprocessing_backend": args.poisson_postprocessing_backend,
        "transport_assembly_backend": args.transport_assembly_backend,
        "transport_solver": args.transport_solver,
        "transport_preconditioner": args.transport_preconditioner,
        "transport_solver_rtol": args.transport_solver_rtol,
        "transport_solver_atol": args.transport_solver_atol,
        "transport_maxiter": args.transport_maxiter,
        "transport_petsc_preset": args.transport_petsc_preset,
        "transport_petsc_levels": args.transport_petsc_levels,
        "transport_cupyx_solver": args.transport_cupyx_solver,
        "transport_amgx_config_path": None if args.transport_amgx_config is None else str(args.transport_amgx_config),
        "transport_amgx_tolerance": args.transport_amgx_tolerance,
        "transport_ilu_drop_tol": args.transport_ilu_drop_tol,
        "transport_ilu_fill_factor": args.transport_ilu_fill_factor,
        "transport_boundary_mode": args.transport_boundary_mode,
        "transport_trace_ordering": args.transport_trace_ordering,
        "transport_trace_ordering_flux_tolerance": args.transport_trace_ordering_flux_tolerance,
        "transport_ilu_permc_spec": args.transport_ilu_permc_spec,
        "transport_raw_local_assembly": args.transport_raw_local_assembly,
        "transport_raw_lu_mode": args.transport_raw_lu_mode,
        "transport_raw_block_size": args.transport_raw_block_size,
        "transport_raw_matrix_format": args.transport_raw_matrix_format,
        "transport_reuse_first_preconditioner": args.transport_reuse_first_preconditioner,
        "transport_initial_guess": args.transport_initial_guess,
        "transport_retry_policy": args.transport_retry_policy,
        "transport_direct_fallback": args.transport_direct_fallback,
        "transport_retry_amgx_config_path": None if args.transport_retry_amgx_config is None else str(args.transport_retry_amgx_config),
        "verbosity": args.verbosity,
        "plot_every": args.plot_every,
        "plot_backend": args.plot_backend,
        "plot_width": args.plot_width,
        "plot_height": args.plot_height,
        "plot_max_fps": args.plot_max_fps,
        "save_movie": args.save_movie,
        "movie_path": None if args.movie_path is None else str(args.movie_path),
        "movie_fps": args.movie_fps,
        "plot_resolution": args.plot_resolution,
        "plot_potential": True if args.plot_both else None,
        "screenshot_dir": None if args.screenshot_dir is None else str(args.screenshot_dir),
        "diagnostics_dir": None if args.diagnostics_dir is None else str(args.diagnostics_dir),
        "diagnostics_prefix": args.diagnostics_prefix,
        "positivity_diagnostics": getattr(args, "positivity_diagnostics", None),
        "positivity_tolerance": getattr(args, "positivity_tolerance", None),
        "diocotron_diagnostics": getattr(args, "diocotron_diagnostics", None),
        "diocotron_radial_points": getattr(args, "diocotron_radial_points", None),
        "diocotron_angular_points": getattr(args, "diocotron_angular_points", None),
        "diagnostics_every": args.diagnostics_every,
        "record_timings": getattr(args, "record_timings", None),
        "poisson_true_residual_every": getattr(args, "poisson_true_residual_every", None),
        "poisson_residual_history": getattr(args, "poisson_residual_history", None),
        "amgx_residual_history": getattr(args, "amgx_residual_history", None),
    }
    for key, value in direct_updates.items():
        if value is not None:
            updates[key] = value
    if args.case is not None:
        updates.setdefault("case_params", {})
    case_params = dict(updates.get("case_params", config.case_params))
    for raw in args.case_param or ():
        key, value = _parse_case_param(raw)
        case_params[key] = value
    if args.case_param or args.case is not None:
        updates["case_params"] = case_params
    poisson_scale = _scale_choice(args.poisson_scale_system, poisson=True)
    if poisson_scale is not None:
        updates["poisson_scale_system"] = poisson_scale
    if args.transport_scale_system is not None:
        updates["transport_scale_system"] = _scale_choice(args.transport_scale_system, poisson=False)
    materialize = _materialize_choice(args.transport_materialize_host_solution)
    if args.transport_materialize_host_solution is not None:
        updates["transport_materialize_host_solution"] = materialize
    if args.transport_materialize_host_system:
        updates["transport_materialize_host_system"] = True
    if args.no_transport_materialize_host_system:
        updates["transport_materialize_host_system"] = False
    if args.plot:
        updates["plot_every"] = 1 if args.plot_every is None else int(args.plot_every)
    if args.plot_off_screen:
        updates["plot_off_screen"] = True
    if args.no_plot_mesh:
        updates["plot_show_mesh"] = False
    if args.quiet:
        updates["verbosity"] = 0
    final_time = getattr(args, "final_time", None)
    if final_time is not None:
        if args.num_steps is not None:
            raise ValueError("--final-time and --num-steps cannot be used together")
        dt = updates.get("dt", config.dt)
        if not math.isfinite(final_time) or final_time < 0:
            raise ValueError("--final-time must be finite and nonnegative")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("--dt must be finite and positive")
        ratio = final_time / dt
        if not math.isfinite(ratio):
            raise ValueError("--final-time / --dt must be finite")
        steps = int(round(ratio))
        if (final_time > 0 and steps == 0) or not math.isclose(
                steps * dt, final_time, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError(
                f"--final-time={final_time:g} must be an integer multiple of --dt={dt:g}; "
                "choose a compatible --dt (fixed-step schemes keep dt unchanged)")
        updates["num_steps"] = steps
    transport_stabilization = getattr(args, "transport_advection_stabilization", None)
    if transport_stabilization is not None:
        updates["transport_advection_stabilization"] = (
            None if transport_stabilization == "upwind" else transport_stabilization
        )
    upwind_scale = getattr(args, "transport_upwind_factor", None)
    if upwind_scale is not None:
        from hdgfem.hdg.stabilization import ScaledUpwind
        updates["transport_advection_stabilization"] = ScaledUpwind(upwind_scale)
    runtime = replace(config, **updates) if updates else config
    if args.transport_amgx_config is None:
        runtime = _with_fp32_transport_solver(runtime)
    if (
        args.backend_profile == "device"
        and not 4 <= int(runtime.order) <= 6
        and str(runtime.poisson_solver).replace("_", "-").lower() == "fb-hp-mg-pcg"
    ):
        runtime = replace(runtime, poisson_solver="amgx")
    if PRECISION == "float64" and _is_amgx_solver(runtime.poisson_solver) and args.poisson_solver_rtol is None and runtime.poisson_solver_rtol < _AMGX_DEFAULT_RTOL:
        runtime = replace(runtime, poisson_solver_rtol=_AMGX_DEFAULT_RTOL)
    if PRECISION == "float64" and _is_amgx_solver(runtime.poisson_solver) and args.poisson_solver_atol is None and runtime.poisson_solver_atol < _AMGX_DEFAULT_ATOL:
        runtime = replace(runtime, poisson_solver_atol=_AMGX_DEFAULT_ATOL)
    if PRECISION == "float64" and _is_amgx_solver(runtime.transport_solver) and args.transport_solver_rtol is None and runtime.transport_solver_rtol < _AMGX_DEFAULT_RTOL:
        runtime = replace(runtime, transport_solver_rtol=_AMGX_DEFAULT_RTOL)
    if PRECISION == "float64" and _is_amgx_solver(runtime.transport_solver) and args.transport_solver_atol is None and runtime.transport_solver_atol < _AMGX_DEFAULT_ATOL:
        runtime = replace(runtime, transport_solver_atol=_AMGX_DEFAULT_ATOL)
    if (runtime.plot_diagnostics or runtime.save_diagnostics) and runtime.case == "diocotron_k":
        if getattr(args, "diocotron_diagnostics", None) is False:
            raise ValueError("diagnostic figures need diocotron diagnostics for diocotron_k")
        runtime = replace(runtime, diocotron_diagnostics=True)
    return runtime



def _hybrid_startup_method(config: GuidingCenterRunPreset) -> str:
    """Select the startup configuration of the requested hybrid."""
    return config.h1_startup if config.time_scheme == "h1-bdf3" else config.h2_startup


def _validate_config(config: GuidingCenterRunPreset) -> None:
    if config.poisson_order_offset not in {-1, 0}:
        raise ValueError("poisson_order_offset must be -1 or 0")
    if config.order + config.poisson_order_offset < 0:
        raise ValueError("Poisson degree must be nonnegative; reduced-order Poisson requires density order >= 1")
    if config.transport_electric_field not in {"raw", "postprocessed"}:
        raise ValueError("transport_electric_field must be raw or postprocessed")
    if config.poisson_order_offset or config.transport_electric_field == "postprocessed":
        if config.time_scheme != "si-bdf2":
            raise ValueError("reduced-order Poisson and recovered transport drift currently require si-bdf2")
    if config.transport_electric_field == "postprocessed":
        if config.poisson_order_offset != -1:
            raise ValueError("recovered BDF2 drift requires poisson_order_offset=-1 to fit density DG(p)")
        if config.poisson_hdg_postprocess not in {"flux", "both"}:
            raise ValueError("recovered transport drift requires Poisson flux postprocessing on every solve")
        if config.poisson_flux_postprocess_every:
            raise ValueError("recovered transport drift uses continuous postprocessing; set poisson_flux_postprocess_every=0")
    if not math.isfinite(config.positivity_tolerance) or config.positivity_tolerance < 0:
        raise ValueError("positivity_tolerance must be finite and nonnegative")
    if config.diocotron_radial_points < 2:
        raise ValueError("diocotron_radial_points must be at least 2")
    if config.diocotron_angular_points is not None and config.diocotron_angular_points < 2:
        raise ValueError("diocotron_angular_points must be at least 2")
    if not math.isfinite(config.movie_fps) or config.movie_fps <= 0:
        raise ValueError("movie_fps must be finite and positive")
    if config.save_movie:
        if config.plot_backend != "holoviz" or config.plot_every <= 0:
            raise ValueError("movie recording requires Holoviz with plot_every > 0")
        if config.movie_path is None or not str(config.movie_path).lower().endswith(".mp4"):
            raise ValueError("movie_path must end in .mp4")
    if config.plot_backend not in {"pyvista", "holoviz"}:
        raise ValueError("plot_backend must be 'pyvista' or 'holoviz'")
    if config.plot_every < 0:
        raise ValueError("plot_every must be nonnegative")
    if config.plot_width < 2 or config.plot_height < 2:
        raise ValueError("plot width and height must be at least 2")
    if not math.isfinite(config.plot_max_fps) or config.plot_max_fps <= 0:
        raise ValueError("plot_max_fps must be finite and positive")
    if config.num_steps < 0:
        raise ValueError("num_steps must be nonnegative")
    if not math.isfinite(config.dt) or config.dt <= 0.0:
        raise ValueError("dt must be positive")
    if config.minimum_triangles < 0:
        raise ValueError("minimum_triangles must be nonnegative")
    if not 0 <= int(config.verbosity) <= 3:
        raise ValueError("verbosity must be one of 0, 1, 2, or 3")
    if config.time_scheme not in STEPPERS:
        raise ValueError(f"time_scheme must be one of {tuple(STEPPERS)}")
    if config.time_scheme in {"h1-bdf3", "h2-bdf3", "imex-ark3"}:
        if config.time_scheme != "imex-ark3" and _hybrid_startup_method(config) not in {"si-euler-extrap3", "ssprk3"}:
            raise ValueError("hybrid startup must be si-euler-extrap3 or ssprk3")
        if config.transport_advection_stabilization is not None:
            raise ValueError(f"{config.time_scheme} requires the standard upwind stabilization")
        if config.transport_boundary_mode not in {"auto", "zero-flux", "eliminate"}:
            raise ValueError(f"{config.time_scheme} requires zero-flux or eliminated transport boundaries")
    from scripts.guiding_center.poisson.poisson_recovery import PoissonTauRecovery
    PoissonTauRecovery(factor=config.poisson_tau_retry_factor, max_retries=config.poisson_tau_max_retries)
    if config.poisson_tau_max_retries or config.time_scheme == "imex-ark3":
        if not math.isfinite(config.poisson_tau) or config.poisson_tau <= 0:
            raise ValueError("Poisson tau recovery requires finite positive poisson_tau")
    if config.time_scheme == "imex-ark3":
        if config.transport_assembly_backend not in {"numpy", "raw-cuda"}:
            raise ValueError("IMEX-ARK3 operator reuse currently requires numpy or raw-cuda transport")
        if config.transport_assembly_backend == "raw-cuda" and config.transport_raw_local_assembly != "fused":
            raise ValueError("IMEX-ARK3 raw-cuda transport requires fused local assembly")
        if config.transport_trace_ordering != "none" or config.transport_reuse_first_preconditioner:
            raise ValueError("IMEX-ARK3 manages per-step operator reuse; use no trace ordering or first-preconditioner override")
    if config.poisson_true_residual_every < 0:
        raise ValueError("poisson_true_residual_every must be nonnegative")
    if config.diagnostics_every < 0:
        raise ValueError("diagnostics_every must be nonnegative (0 disables field diagnostics)")
    if config.diagnostics_every == 0 and (
        config.plot_diagnostics or config.save_diagnostics or config.diocotron_diagnostics
        or config.positivity_diagnostics
    ):
        raise ValueError("diagnostics_every=0 requires diagnostic plots, positivity and modal diagnostics disabled")
    if config.poisson_flux_postprocess_every < 0:
        raise ValueError("poisson_flux_postprocess_every must be nonnegative")
    if config.transport_retry_policy not in {"none", "amgx-robust"}:
        raise ValueError("transport_retry_policy must be 'none' or 'amgx-robust'")
    if config.poisson_retry_policy not in {"none", "amgx-robust"}:
        raise ValueError("poisson_retry_policy must be 'none' or 'amgx-robust'")
    if config.poisson_fb_hp_mg_preconditioner_policy not in {"standard", "fast", "robust"}:
        raise ValueError(
            "poisson_fb_hp_mg_preconditioner_policy must be 'standard', 'fast' or 'robust'"
        )
    poisson_is_native = (
        str(config.poisson_solver).replace("_", "-").lower() == "fb-hp-mg-pcg"
    )
    if (
        config.poisson_fb_hp_mg_preconditioner_policy != "standard"
        and not poisson_is_native
    ):
        raise ValueError(
            f"{config.poisson_fb_hp_mg_preconditioner_policy} FB-HP-MG conditioning "
            "requires fb-hp-mg-pcg Poisson"
        )
    if config.poisson_retry_policy == "amgx-robust" and (
        config.poisson_assembly_backend != "raw-cuda"
        or not (_is_amgx_solver(config.poisson_solver) or poisson_is_native)
    ):
        raise ValueError(
            "robust Poisson retries require a raw-CUDA AMGX or FB-HP-MG solve"
        )
    if config.transport_direct_fallback not in {"none", "cusolver-qr"}:
        raise ValueError("transport_direct_fallback must be 'none' or 'cusolver-qr'")
    if config.transport_direct_fallback != "none" and (
        config.transport_retry_policy != "amgx-robust"
        or not _is_amgx_solver(config.transport_solver)
        or config.transport_assembly_backend not in {"cupy", "raw-cuda"}
    ):
        raise ValueError("device direct fallback requires amgx-robust retries and a CuPy/raw-CUDA AMGX transport solve")
    if (
        config.transport_amgx_tolerance is not None
        and (
            not math.isfinite(float(config.transport_amgx_tolerance))
            or float(config.transport_amgx_tolerance) < 0.0
        )
    ):
        raise ValueError("transport_amgx_tolerance must be finite and nonnegative")
    if config.transport_reuse_first_preconditioner and config.transport_trace_ordering != "none":
        raise ValueError(
            "transport_reuse_first_preconditioner requires trace_ordering='none' so the "
            "cached preconditioner remains in the same trace coordinate system"
        )
    if config.case == "rho_helm_wave" and config.transport_boundary_mode == "zero-flux":
        raise ValueError("rho_helm_wave requires eliminated exact density boundary data; zero-flux is invalid")
    if config.poisson_assembly_backend == "cupy":
        if not _is_amgx_solver(config.poisson_solver):
            raise ValueError("poisson_assembly_backend='cupy' requires poisson_solver='amgx'")
        if config.poisson_cache_local_factors not in {"none", "schur-cholesky"}:
            raise ValueError("CuPy guiding-center Poisson local-factor caching requires 'schur-cholesky'")
        if config.poisson_hdg_postprocess != "none":
            raise ValueError("CuPy guiding-center Poisson currently requires poisson_hdg_postprocess='none'")
    if config.poisson_assembly_backend == "raw-cuda":
        native = poisson_is_native
        if not (_is_amgx_solver(config.poisson_solver) or native):
            raise ValueError(
                "poisson_assembly_backend='raw-cuda' requires poisson_solver='amgx' "
                "or 'fb-hp-mg-pcg'"
            )
        matrix_format = str(config.poisson_raw_matrix_format).lower()
        if matrix_format not in {"auto", "csr", "bsr"}:
            raise ValueError(
                "raw-CUDA guiding-center Poisson requires raw matrix format auto, csr, or bsr"
            )
        if native and matrix_format not in {"auto", "bsr"}:
            raise ValueError("FB-HP-MG Poisson requires raw matrix format auto or bsr")
        if native and _poisson_trace_basis(config) != "legendre-modal":
            raise ValueError("FB-HP-MG Poisson requires poisson_trace_basis='legendre-modal'")
        if config.poisson_hdg_postprocess not in {"none", "flux"}:
            raise ValueError("raw-CUDA guiding-center Poisson supports none or flux postprocessing")


def _make_poisson_options(config: GuidingCenterRunPreset):
    from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGOptions

    config_dir = Path(__file__).resolve().parents[3] / "configs" / "amgx"
    primary_path = config.poisson_amgx_config_path
    if primary_path is None and config.poisson_retry_policy == "amgx-robust":
        primary_path = str(
            config_dir / "diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_robust_abs.json"
        )
    primary_config = _load_amgx_config(
        primary_path,
        tolerance=config.poisson_solver_rtol,
        absolute_tolerance=config.poisson_solver_atol,
    )
    retry_attempts = None
    if config.poisson_retry_policy == "amgx-robust":
        pure_csr_config = _load_amgx_config(
            str(config_dir / "diff_rea_gpu4_hdg_pcgf_classical_gs_robust_abs.json"),
            tolerance=config.poisson_solver_rtol,
            absolute_tolerance=config.poisson_solver_atol,
        )
        terminal_path = config.poisson_retry_amgx_config_path or str(
            config_dir / "diff_rea_gpu4_hdg_fgmres_dilu_robust_abs.json"
        )
        terminal_config = _load_amgx_config(
            terminal_path,
            tolerance=config.poisson_solver_rtol,
            absolute_tolerance=config.poisson_solver_atol,
        )

        def outer_solver(amgx_config: dict | None) -> str:
            return str((amgx_config or {}).get("solver", {}).get("solver", "")).upper()

        if outer_solver(primary_config) != "PCGF" or outer_solver(pure_csr_config) != "PCGF":
            raise ValueError("robust Poisson primary and scalar-CSR retries must use PCGF")
        terminal_solver = (terminal_config or {}).get("solver", {})
        terminal_preconditioner = terminal_solver.get("preconditioner", {})
        if (
            outer_solver(terminal_config) != "FGMRES"
            or str(terminal_preconditioner.get("solver", "")).upper()
            != "MULTICOLOR_DILU"
        ):
            raise ValueError(
                "the terminal Poisson retry must use FGMRES with MULTICOLOR_DILU"
            )
        retry_attempts = (
            {
                "label": "hybrid-pcgf-zero",
                "config": primary_config,
                "reuse_primary_solver": True,
                "use_initial_guess": False,
                "scale_system": config.poisson_scale_system,
            },
            {
                "label": "pure-csr-pcgf-zero",
                "config": pure_csr_config,
                "scalarize_bsr": True,
                "reuse_preconditioner": True,
                "solver_cache_key": "poisson-pure-csr-pcgf",
                "use_initial_guess": False,
                "scale_system": False,
            },
            {
                "label": "pure-csr-pcgf-correction-1",
                "config": pure_csr_config,
                "scalarize_bsr": True,
                "reuse_preconditioner": True,
                "solver_cache_key": "poisson-pure-csr-pcgf",
                "use_initial_guess": False,
                "use_best_solution": True,
                "residual_correction": True,
                "scale_system": False,
            },
            {
                "label": "pure-csr-pcgf-correction-2",
                "config": pure_csr_config,
                "scalarize_bsr": True,
                "reuse_preconditioner": True,
                "solver_cache_key": "poisson-pure-csr-pcgf",
                "use_initial_guess": False,
                "use_best_solution": True,
                "residual_correction": True,
                "scale_system": False,
            },
            {
                "label": "pure-csr-fgmres-dilu-last",
                "config": terminal_config,
                "scalarize_bsr": True,
                "reuse_preconditioner": True,
                "solver_cache_key": "poisson-pure-csr-fgmres-dilu",
                "use_initial_guess": False,
                "use_best_solution": True,
                "scale_system": False,
            },
        )

    primary_config = with_amgx_residual_history(primary_config, config.amgx_residual_history)
    if retry_attempts is not None:
        retry_attempts = tuple(
            {**attempt, "config": with_amgx_residual_history(attempt["config"], config.amgx_residual_history)}
            if "config" in attempt else attempt for attempt in retry_attempts
        )

    return DiffusionReactionHDGOptions(
        diffusion=1.0,
        stabilization=config.poisson_tau,
        solver=config.poisson_solver,
        preconditioner=config.poisson_preconditioner,
        solver_rtol=config.poisson_solver_rtol,
        solver_atol=config.poisson_solver_atol,
        maxiter=config.poisson_maxiter,
        scale_system=bool(config.poisson_scale_system),
        petsc_preset=config.poisson_petsc_preset,
        petsc_levels=config.poisson_petsc_levels,
        petsc_options=dict(config.poisson_petsc_options),
        petsc_divtol=config.poisson_petsc_divtol,
        petsc_monitor=config.poisson_petsc_monitor,
        cupyx_solver=config.poisson_cupyx_solver,
        amgx_config=primary_config,
        amgx_retry_attempts=retry_attempts,
        fb_hp_mg_true_residual_every=config.poisson_true_residual_every,
        fb_hp_mg_residual_history=config.poisson_residual_history,
        fb_hp_mg_preconditioner_policy=(
            config.poisson_fb_hp_mg_preconditioner_policy
        ),
        ilu_drop_tol=config.poisson_ilu_drop_tol,
        ilu_fill_factor=config.poisson_ilu_fill_factor,
        ilu_failure=config.poisson_ilu_failure,
        ilu_permc_spec=config.poisson_ilu_permc_spec,
        local_solver_backend=config.poisson_local_backend,
        assembly_backend=config.poisson_assembly_backend,
        trace_basis=_poisson_trace_basis(config),
        raw_matrix_format=config.poisson_raw_matrix_format,
        raw_block_size=config.poisson_raw_block_size,
        cache_local_factors=config.poisson_cache_local_factors,
        boundary_mode="eliminate",
        hdg_postprocess=config.poisson_hdg_postprocess,
        flux_postprocess_space=config.poisson_flux_postprocess_space,
        postprocessing_backend=config.poisson_postprocessing_backend,
        verbose=_solver_verbosity(config),
    )


def _transport_amgx_divergence_config(config: dict | None, *, tolerance: float) -> dict:
    """Enable native divergence exits for transport, preserving explicit overrides."""
    if config is None:
        from hdgfem.backends.cupy import default_pyamgx_config
        config = default_pyamgx_config(tolerance=tolerance, maxiter=None)
    config = copy.deepcopy(config)
    solver = config.setdefault("solver", {})
    solver.setdefault("rel_div_tolerance", 1.0e3)
    solver.setdefault("divergence_patience", 5)
    solver.setdefault("divergence_grace_iters", 10)
    solver.setdefault("print_solve_stats_interval", 10)
    return config


def _make_transport_options(config: GuidingCenterRunPreset, boundary_mode: str):
    from hdgfem.solvers.advection_reaction import AdvectionReactionHDGOptions

    retry_attempts = None
    primary_amgx_tolerance = (
        config.transport_solver_rtol
        if config.transport_amgx_tolerance is None
        else float(config.transport_amgx_tolerance)
    )
    if config.transport_retry_policy == "amgx-robust":
        fallback_path = config.transport_retry_amgx_config_path
        if fallback_path is None:
            fallback_path = str(
                Path(__file__).resolve().parents[3]
                / "configs"
                / "amgx"
                / "adv_rea_gpu4_hdg_fgmres_dilu_abs.json"
            )
        fallback_config = _load_amgx_config(
            fallback_path,
            tolerance=config.transport_solver_rtol,
            absolute_tolerance=config.transport_solver_atol,
        )
        amgx_config_dir = Path(__file__).resolve().parents[3] / "configs" / "amgx"
        # AMGX's plain BICGSTAB ignores preconditioners; PBICGSTAB applies them.
        # These one-sweep Jacobi retries keep native BSR and rebuild their cheap
        # diagonal data for the current matrix. Apply the caller's scaling to
        # these retries too; acceptance still checks the physical residual.
        bsr_retries = tuple(
            {
                "label": label,
                "config": _load_amgx_config(
                    str(amgx_config_dir / filename),
                    tolerance=config.transport_solver_rtol,
                ),
                "scalarize_bsr": False,
                "reuse_preconditioner": False,
                "use_initial_guess": False,
                "scale_system": config.transport_scale_system,
            }
            for label, filename in (
                ("pbicgstab-l1-zero", "adv_rea_gpu4_hdg_pbicgstab_l1_bsr.json"),
                (
                    "pbicgstab-block-jacobi-zero",
                    "adv_rea_gpu4_hdg_pbicgstab_block_jacobi_bsr.json",
                ),
            )
        )
        retry_attempts = (
            *bsr_retries,
            {
                "label": "robust-zero-scaled",
                "config": fallback_config,
                "scalarize_bsr": True,
                "reuse_preconditioner": True,
                "solver_cache_key": "transport-fgmres-dilu",
                "use_initial_guess": False,
                "scale_system": config.transport_scale_system,
            },
            {
                "label": "robust-correction-1",
                "config": fallback_config,
                "scalarize_bsr": True,
                "reuse_preconditioner": True,
                "solver_cache_key": "transport-fgmres-dilu",
                "use_initial_guess": False,
                "use_best_solution": True,
                "residual_correction": True,
                "scale_system": config.transport_scale_system,
            },
            {
                "label": "robust-correction-2",
                "config": fallback_config,
                "scalarize_bsr": True,
                "reuse_preconditioner": True,
                "solver_cache_key": "transport-fgmres-dilu",
                "use_initial_guess": False,
                "use_best_solution": True,
                "residual_correction": True,
                "scale_system": config.transport_scale_system,
            },
        )

    if config.transport_direct_fallback == "cusolver-qr":
        retry_attempts = (*tuple(retry_attempts or ()), {
            "label": "direct-qr", "backend": "cusolver-qr",
            "use_initial_guess": False, "scale_system": False,
        })

    if config.time_scheme in {"h1-bdf3", "h2-bdf3", "imex-ark3"} and retry_attempts:
        # Correction solves use zero correction around the best endpoint
        # iterate. Other iterative attempts receive an endpoint guess.
        retry_attempts = tuple(
            attempt if attempt.get("backend") == "cusolver-qr" or attempt.get("residual_correction")
            else {**attempt, "label": attempt["label"].replace("-zero", "-warm"),
                  "use_initial_guess": True, "use_best_solution": True}
            for attempt in retry_attempts
        )

    primary_config = _load_amgx_config(
        config.transport_amgx_config_path, tolerance=primary_amgx_tolerance
    )
    if config.transport_solver == "amgx" or config.transport_cupyx_solver == "pyamgx":
        primary_config = _transport_amgx_divergence_config(
            primary_config, tolerance=primary_amgx_tolerance
        )
        if retry_attempts is not None:
            retry_attempts = tuple(
                dict(attempt, config=_transport_amgx_divergence_config(
                    attempt["config"], tolerance=config.transport_solver_rtol
                )) if attempt.get("backend", "amgx") == "amgx" else attempt
                for attempt in retry_attempts
            )

    primary_config = with_amgx_residual_history(primary_config, config.amgx_residual_history)
    if retry_attempts is not None:
        retry_attempts = tuple(
            {**attempt, "config": with_amgx_residual_history(attempt["config"], config.amgx_residual_history)}
            if "config" in attempt else attempt for attempt in retry_attempts
        )

    return AdvectionReactionHDGOptions(
        solver=config.transport_solver,
        preconditioner=config.transport_preconditioner,
        solver_rtol=config.transport_solver_rtol,
        solver_atol=config.transport_solver_atol,
        maxiter=config.transport_maxiter,
        petsc_preset=config.transport_petsc_preset,
        petsc_levels=config.transport_petsc_levels,
        petsc_options=dict(config.transport_petsc_options),
        petsc_divtol=config.transport_petsc_divtol,
        petsc_monitor=config.transport_petsc_monitor,
        cupyx_solver=config.transport_cupyx_solver,
        amgx_config=primary_config,
        amgx_retry_attempts=retry_attempts,
        ilu_drop_tol=config.transport_ilu_drop_tol,
        ilu_fill_factor=config.transport_ilu_fill_factor,
        ilu_failure=config.transport_ilu_failure,
        scale_system=config.transport_scale_system,
        boundary_mode=boundary_mode,
        trace_ordering=config.transport_trace_ordering,
        trace_ordering_flux_tolerance=config.transport_trace_ordering_flux_tolerance,
        ilu_permc_spec=config.transport_ilu_permc_spec,
        assembly_backend=config.transport_assembly_backend,
        trace_basis=_transport_trace_basis(config),
        raw_local_assembly=config.transport_raw_local_assembly,
        raw_lu_mode=config.transport_raw_lu_mode,
        raw_block_size=config.transport_raw_block_size,
        raw_matrix_format=config.transport_raw_matrix_format,
        materialize_host_system=config.transport_materialize_host_system,
        materialize_host_solution=config.transport_materialize_host_solution,
        advection_stabilization=config.transport_advection_stabilization,
        cache_local_solvers=config.transport_cache_local_solvers,
        cache_operator=config.time_scheme == "imex-ark3",
        verbose=_solver_verbosity(config),
    )
