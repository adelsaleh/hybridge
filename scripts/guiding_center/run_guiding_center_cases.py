#!/usr/bin/env python3
"""Run fixed-mesh guiding-center cases."""

from __future__ import annotations

import ast
import csv
import json
import math
import sys
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.guiding_center.guiding_center_cases import CASE_DEFINITIONS
from scripts.guiding_center.guiding_center_presets import (
    DEFAULT_PRESET,
    PRESETS,
    GuidingCenterRunPreset,
    preset_by_key,
    print_preset_details,
    print_presets,
)


@dataclass(frozen=True)
class GuidingCenterRunResult:
    """Artifacts returned by :func:`run_guiding_center_case`."""

    config: GuidingCenterRunPreset
    preset_key: str
    case_key: str
    mesh: Any
    space: Any
    final_density: Any
    final_potential: Any
    final_flux: Any
    diagnostics: list[dict[str, Any]]
    csv_path: Path
    jsonl_path: Path


class DiagnosticsRecorder:
    """Collect per-step diagnostics and write JSONL incrementally plus CSV at close."""

    def __init__(self, directory: str | Path, prefix: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = str(prefix).strip() or "guiding_center"
        self.csv_path = self.directory / f"{stem}.csv"
        self.jsonl_path = self.directory / f"{stem}.jsonl"
        self.rows: list[dict[str, Any]] = []
        self._jsonl = self.jsonl_path.open("w", encoding="utf-8")

    def record(self, row: dict[str, Any]) -> None:
        clean = {key: _json_safe(value) for key, value in row.items()}
        self.rows.append(clean)
        self._jsonl.write(json.dumps(clean, sort_keys=True) + "\n")
        self._jsonl.flush()

    def close(self) -> None:
        self._jsonl.close()
        fieldnames: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


class GuidingCenterPyVistaPanels:
    """PyVista plot whose scalar values are updated in place."""

    def __init__(
            self,
            density_field,
            potential_field,
            *,
            resolution: int,
            title: str,
            show_mesh: bool,
            off_screen: bool,
            screenshot_dir: str | None,
            screenshot_prefix: str,
            include_potential: bool = False,
    ) -> None:
        from hdgfem.io.plot import (
            _mesh_overlay_line_width,
            _require_pyvista,
            _safe_clim,
            coarse_mesh_polydata,
            reference_plot_points,
            refined_field_polydata,
        )

        self.pv = _require_pyvista()
        self._safe_clim = _safe_clim
        self.reference_points = reference_plot_points(max(2, int(resolution)))
        self.rho_name = "density"
        self.phi_name = "potential"
        self.include_potential = bool(include_potential)
        self.rho_mesh = refined_field_polydata(
            density_field,
            reference_points=self.reference_points,
            scalar_name=self.rho_name,
        )
        self.phi_mesh = None
        if self.include_potential:
            self.phi_mesh = refined_field_polydata(
                potential_field,
                reference_points=self.reference_points,
                scalar_name=self.phi_name,
            )
        shape = (1, 2) if self.include_potential else (1, 1)
        window_size = [1500, 650] if self.include_potential else [820, 720]
        self.plotter = self.pv.Plotter(shape=shape, window_size=window_size, off_screen=off_screen)
        self.off_screen = bool(off_screen)
        self.screenshot_dir = None if screenshot_dir is None else Path(screenshot_dir)
        if self.screenshot_dir is not None:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_prefix = screenshot_prefix
        scalar_bar_args = {
            "vertical": False,
            "width": 0.55,
            "height": 0.08,
            "position_x": 0.225,
            "position_y": 0.02,
        }
        self.plotter.subplot(0, 0)
        self.rho_actor = self.plotter.add_mesh(
            self.rho_mesh,
            scalars=self.rho_name,
            cmap="viridis",
            clim=self._safe_clim(self.rho_mesh.point_data[self.rho_name]),
            scalar_bar_args=scalar_bar_args,
        )
        if show_mesh:
            self.plotter.add_mesh(
                coarse_mesh_polydata(density_field.space.mesh),
                style="wireframe",
                color="black",
                line_width=_mesh_overlay_line_width(density_field.space.mesh),
                opacity=0.45,
            )
        self.plotter.add_text("Density", position="upper_left", font_size=10, shadow=False)
        self.plotter.enable_parallel_projection()
        self.plotter.view_xy()

        self.phi_actor = None
        if self.include_potential:
            self.plotter.subplot(0, 1)
            self.phi_actor = self.plotter.add_mesh(
                self.phi_mesh,
                scalars=self.phi_name,
                cmap="viridis",
                clim=self._safe_clim(self.phi_mesh.point_data[self.phi_name]),
                scalar_bar_args=scalar_bar_args,
            )
            if show_mesh:
                self.plotter.add_mesh(
                    coarse_mesh_polydata(potential_field.space.mesh),
                    style="wireframe",
                    color="black",
                    line_width=_mesh_overlay_line_width(potential_field.space.mesh),
                    opacity=0.45,
                )
            self.plotter.add_text("Potential", position="upper_left", font_size=10, shadow=False)
            self.plotter.enable_parallel_projection()
            self.plotter.view_xy()
            self.plotter.link_views()
        self.plotter.add_title(title, font_size=10)
        self._shown = False

    def _update_mesh_values(self, mesh, scalar_name: str, field) -> np.ndarray:
        values = np.asarray(field.values_at_ref(self.reference_points), dtype=np.float64).reshape(-1)
        scalars = mesh.point_data[scalar_name]
        scalars[:] = values
        mesh.Modified()
        return values

    def _update_actor_clim(self, actor, values: np.ndarray) -> None:
        try:
            actor.mapper.scalar_range = self._safe_clim(values)
        except Exception:
            pass

    def update(self, density_field, potential_field, *, step: int, time_value: float) -> None:
        """Update active scalar arrays in place and refresh the render window."""
        rho_values = self._update_mesh_values(self.rho_mesh, self.rho_name, density_field)
        self._update_actor_clim(self.rho_actor, rho_values)
        if self.include_potential:
            phi_values = self._update_mesh_values(self.phi_mesh, self.phi_name, potential_field)
            self._update_actor_clim(self.phi_actor, phi_values)
        if not self._shown:
            self.plotter.show(auto_close=False, interactive_update=True)
            self._shown = True
        else:
            self.plotter.update()
        if self.screenshot_dir is not None:
            path = self.screenshot_dir / f"{self.screenshot_prefix}_step{int(step):05d}_t{float(time_value):.6f}.png"
            self.plotter.screenshot(str(path))

def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def _load_amgx_config(
        path: str | None,
        *,
        tolerance: float | None = None,
        absolute_tolerance: float | None = None,
) -> dict | None:
    if path is None:
        return None
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
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
    return max(0, _verbosity_level(config) - 1)


def _phase_verbosity(config: GuidingCenterRunPreset) -> int:
    return 1 if _verbosity_level(config) >= 2 else 0


def _format_metric(value: Any, fmt: str = ".3e") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return format(number, fmt)


def _first_metric(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _print_step_summary(config: GuidingCenterRunPreset, row: dict[str, Any]) -> None:
    if _verbosity_level(config) < 1:
        return
    step = int(row.get("step", 0))
    phase = str(row.get("phase", "step"))
    step_label = "initial" if phase == "initial" else f"{step:05d}/{int(config.num_steps):05d}"
    pieces = [
        f"[gc] step={step_label}",
        f"t={_format_metric(row.get('time'), '.6f')}",
        f"mass_rel={_format_metric(row.get('mass_relative_drift'))}",
        f"q_rel={_format_metric(row.get('q_l2_relative_drift'))}",
        (
            "rho=["
            f"{_format_metric(row.get('rho_min'))},"
            f"{_format_metric(row.get('rho_max'))}]"
        ),
        (
            "phi=["
            f"{_format_metric(row.get('phi_min'))},"
            f"{_format_metric(row.get('phi_max'))}]"
        ),
    ]
    beta_time = row.get("beta_build_time")
    if beta_time is not None and phase != "initial":
        pieces.append(f"beta={_format_metric(beta_time, '.3f')}s")
    poisson_time = _first_metric(row, "poisson_time_total", "poisson_time")
    if poisson_time is not None:
        pieces.append(f"poisson={_format_metric(poisson_time, '.3f')}s")
    transport_time = _first_metric(row, "transport_time_total", "transport_time")
    if transport_time is not None and phase != "initial":
        pieces.append(f"transport={_format_metric(transport_time, '.3f')}s")
    plot_time = row.get("plot_time")
    if plot_time:
        pieces.append(f"plot={_format_metric(plot_time, '.3f')}s")
    poisson_rel = row.get("poisson_solver_rel_residual")
    if poisson_rel is not None:
        pieces.append(f"p_rel={_format_metric(poisson_rel)}")
    transport_rel = row.get("transport_solver_rel_residual")
    if transport_rel is not None:
        pieces.append(f"t_rel={_format_metric(transport_rel)}")
    if row.get("rho_l2_error") is not None:
        pieces.append(f"rho_l2={_format_metric(row.get('rho_l2_error'))}")
    if row.get("phi_l2_error") is not None:
        pieces.append(f"phi_l2={_format_metric(row.get('phi_l2_error'))}")
    if row.get("diocotron_phi_eq_relative_l2") is not None:
        pieces.append(f"phi_eq_rel={_format_metric(row.get('diocotron_phi_eq_relative_l2'))}")
    print(" ".join(pieces), flush=True)


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
            "transport_assembly_backend": "numba",
            "transport_solver": "amgx",
            "transport_preconditioner": None,
            "transport_solver_rtol": _AMGX_DEFAULT_RTOL,
            "transport_solver_atol": _AMGX_DEFAULT_ATOL,
            "transport_scale_system": True,
        }
    if profile == "device":
        return {
            "poisson_assembly_backend": "numba",
            "poisson_solver": "amgx",
            "poisson_preconditioner": None,
            "poisson_solver_rtol": _AMGX_DEFAULT_RTOL,
            "poisson_solver_atol": _AMGX_DEFAULT_ATOL,
            "poisson_scale_system": False,
            "transport_assembly_backend": "raw-cuda",
            "transport_solver": "amgx",
            "transport_preconditioner": None,
            "transport_solver_rtol": _AMGX_DEFAULT_RTOL,
            "transport_solver_atol": _AMGX_DEFAULT_ATOL,
            "transport_scale_system": True,
            "transport_raw_local_assembly": "fused",
            "transport_raw_matrix_format": "csr",
            "transport_materialize_host_solution": True,
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


_AMGX_DEFAULT_RTOL = 1.0e-11
_AMGX_DEFAULT_ATOL = 1.0e-12


def _is_amgx_solver(solver: str | None) -> bool:
    if solver is None:
        return False
    return str(solver).lower() in {"amgx", "pyamgx"}


def _runtime_config(config: GuidingCenterRunPreset, args) -> GuidingCenterRunPreset:
    updates: dict[str, Any] = {}
    if args.backend_profile is not None:
        updates.update(_backend_profile_updates(args.backend_profile))
    direct_updates = {
        "case": args.case,
        "domain": args.domain,
        "mesh_size": args.mesh_size,
        "nx": args.nx,
        "ny": args.ny,
        "gmsh_verbosity": args.gmsh_verbosity,
        "gmsh_algorithm": args.gmsh_algorithm,
        "basis": args.basis,
        "trace_basis": args.trace_basis,
        "order": args.order,
        "volume_quadrature": args.volume_quadrature,
        "volume_quad_1d": args.volume_quad_1d,
        "edge_quad_1d": args.edge_quad_1d,
        "dt": args.dt,
        "num_steps": args.num_steps,
        "poisson_tau": args.poisson_tau,
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
        "poisson_ilu_drop_tol": args.poisson_ilu_drop_tol,
        "poisson_ilu_fill_factor": args.poisson_ilu_fill_factor,
        "poisson_raw_matrix_format": args.poisson_raw_matrix_format,
        "poisson_raw_block_size": args.poisson_raw_block_size,
        "poisson_hdg_postprocess": args.poisson_hdg_postprocess,
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
        "verbosity": args.verbosity,
        "plot_every": args.plot_every,
        "plot_resolution": args.plot_resolution,
        "plot_potential": True if args.plot_both else None,
        "screenshot_dir": None if args.screenshot_dir is None else str(args.screenshot_dir),
        "diagnostics_dir": None if args.diagnostics_dir is None else str(args.diagnostics_dir),
        "diagnostics_prefix": args.diagnostics_prefix,
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
    runtime = replace(config, **updates) if updates else config
    if _is_amgx_solver(runtime.poisson_solver) and args.poisson_solver_rtol is None and runtime.poisson_solver_rtol < _AMGX_DEFAULT_RTOL:
        runtime = replace(runtime, poisson_solver_rtol=_AMGX_DEFAULT_RTOL)
    if _is_amgx_solver(runtime.poisson_solver) and args.poisson_solver_atol is None and runtime.poisson_solver_atol < _AMGX_DEFAULT_ATOL:
        runtime = replace(runtime, poisson_solver_atol=_AMGX_DEFAULT_ATOL)
    if _is_amgx_solver(runtime.transport_solver) and args.transport_solver_rtol is None and runtime.transport_solver_rtol < _AMGX_DEFAULT_RTOL:
        runtime = replace(runtime, transport_solver_rtol=_AMGX_DEFAULT_RTOL)
    if _is_amgx_solver(runtime.transport_solver) and args.transport_solver_atol is None and runtime.transport_solver_atol < _AMGX_DEFAULT_ATOL:
        runtime = replace(runtime, transport_solver_atol=_AMGX_DEFAULT_ATOL)
    return runtime


def _build_mesh(config: GuidingCenterRunPreset, case):
    from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh

    domain = case.default_domain if config.domain == "auto" else config.domain
    if domain == "structured-rectangle":
        return rectangle_mesh(config.nx, config.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    log_mesh_cache = _phase_verbosity(config) >= 1
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
    if domain == "triangle":
        return gmsh_triangle_mesh(
            config.mesh_size,
            vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
            verbosity=config.gmsh_verbosity,
            algorithm=config.gmsh_algorithm,
            log_cache=log_mesh_cache,
        )
    raise ValueError(f"unknown domain {domain!r}")


def _validate_config(config: GuidingCenterRunPreset) -> None:
    if config.num_steps < 0:
        raise ValueError("num_steps must be nonnegative")
    if config.dt <= 0.0:
        raise ValueError("dt must be positive")
    if not 0 <= int(config.verbosity) <= 3:
        raise ValueError("verbosity must be one of 0, 1, 2, or 3")
    if config.poisson_assembly_backend == "cupy":
        raise NotImplementedError(
            "Guiding-center Poisson solves do not use assembly_backend='cupy'; "
            "use 'numba' for host assembly + GPU solve or 'raw-cuda' for direct device CSR AMGX."
        )
    if config.poisson_assembly_backend == "raw-cuda":
        if not _is_amgx_solver(config.poisson_solver):
            raise ValueError("poisson_assembly_backend='raw-cuda' requires poisson_solver='amgx'")
        if str(config.poisson_raw_matrix_format).lower() != "csr":
            raise ValueError("poisson_assembly_backend='raw-cuda' requires poisson_raw_matrix_format='csr'")
        if config.poisson_hdg_postprocess != "none":
            raise ValueError("raw-CUDA guiding-center Poisson currently requires poisson_hdg_postprocess='none'")


def _zero_field(space, *, name: str):
    return space.zeros(name=name)


def _ensure_field(result, label: str, space, *, name: str):
    field = getattr(result, "field", None)
    if field is not None:
        return field
    field_device = getattr(result, "field_device", None)
    if field_device is None:
        raise RuntimeError(
            f"{label} result did not materialize a host or device DGField. "
            "Enable materialize_host_solution or use a raw-CUDA path that returns field_device."
        )
    from hdgfem.backends.cupy import field_from_cupy_coefficients

    device_id = int(getattr(getattr(field_device, "device", None), "id", 0))
    return field_from_cupy_coefficients(space, field_device, device=device_id, name=name)


def _build_beta_from_flux(flux, dt: float, space):
    from hdgfem.core.space import VectorDGField

    qx, qy = flux.components
    device_ids = set(getattr(qx, "_device_coeffs", {}) or {}).intersection(set(getattr(qy, "_device_coeffs", {}) or {}))
    if device_ids:
        from hdgfem.backends.cupy import field_from_cupy_coefficients, require_cupy

        cp = require_cupy()
        device_id = min(device_ids)
        qx_cp = qx._device_coefficients_for(device_id)
        qy_cp = qy._device_coefficients_for(device_id)
        beta_x = field_from_cupy_coefficients(
            space,
            cp.ascontiguousarray(-float(dt) * qy_cp),
            device=device_id,
            name="beta_x_h",
        )
        beta_y = field_from_cupy_coefficients(
            space,
            cp.ascontiguousarray(float(dt) * qx_cp),
            device=device_id,
            name="beta_y_h",
        )
        return VectorDGField((beta_x, beta_y), name="beta_h")

    beta_x = space.field(-float(dt) * qy.coeffs, name="beta_x_h")
    beta_y = space.field(float(dt) * qx.coeffs, name="beta_y_h")
    return VectorDGField((beta_x, beta_y), name="beta_h")


def _integral(field) -> float:
    values = field.values()
    space = field.space
    return float(np.einsum("K,Kq,q->", space.mesh.aff_jacs, values, space.quad_data.Krf_w, optimize=True))


def _field_min_max(field) -> tuple[float, float]:
    values = np.asarray(field.values(), dtype=np.float64)
    return float(np.min(values)), float(np.max(values))


def _field_l2_norm_from_values(space, values: np.ndarray) -> float:
    return float(np.sqrt(np.einsum("K,Kq,q->", space.mesh.aff_jacs, values * values, space.quad_data.Krf_w, optimize=True)))


def _field_l2_difference(field, reference_field) -> float:
    return _field_l2_norm_from_values(field.space, field.values() - reference_field.values())


def _field_linf_difference(field, reference_field) -> float:
    return float(np.max(np.abs(field.values() - reference_field.values())))


def _field_exact_errors(field, exact) -> tuple[float | None, float | None]:
    if exact is None:
        return None, None
    points = field.space.mapped_quads()
    exact_values = np.asarray(exact(points[:, :, 0], points[:, :, 1]), dtype=np.float64)
    diff = field.values() - exact_values
    return _field_l2_norm_from_values(field.space, diff), float(np.max(np.abs(diff)))


def _vector_l2_norm(vector_field) -> float:
    values = np.asarray(vector_field.values(), dtype=np.float64)
    space = vector_field.components[0].space
    return float(np.sqrt(np.einsum("K,dKq,q->", space.mesh.aff_jacs, values * values, space.quad_data.Krf_w, optimize=True)))


def _relative_drift(value: float, baseline: float) -> float:
    scale = max(abs(float(baseline)), 1.0e-300)
    return (float(value) - float(baseline)) / scale


def _sum_detail_timings(result) -> float:
    timings = getattr(result, "timings", None)
    details = getattr(timings, "details", None) or {}
    total = 0.0
    for key, value in details.items():
        key_text = str(key)
        if "host" in key_text or "to_device" in key_text or "materialization" in key_text:
            total += float(value)
    return total


def _solver_metrics(prefix: str, result) -> dict[str, Any]:
    timings = result.timings
    global_solve = result.global_solve_result
    row: dict[str, Any] = {
        f"{prefix}_assembly_backend": result.assembly_backend,
        f"{prefix}_boundary_mode": result.boundary_mode,
        f"{prefix}_time_total": timings.total,
        f"{prefix}_time_assembly": timings.assembly,
        f"{prefix}_time_solve": timings.solve,
        f"{prefix}_time_reconstruction": timings.reconstruction,
        f"{prefix}_host_device_transfer_time": _sum_detail_timings(result),
    }
    if hasattr(timings, "postprocessing"):
        row[f"{prefix}_time_postprocessing"] = timings.postprocessing
    for key, value in (getattr(timings, "details", None) or {}).items():
        if isinstance(value, (int, float)):
            safe_key = "".join(ch if ch.isalnum() else "_" for ch in str(key)).strip("_")
            row[f"{prefix}_detail_{safe_key}"] = float(value)
    if global_solve is not None:
        row.update(
            {
                f"{prefix}_solver_iterations": -1 if global_solve.iteration_count is None else global_solve.iteration_count,
                f"{prefix}_solver_residual": global_solve.solver_residual_norm,
                f"{prefix}_solver_rhs_norm": global_solve.solver_rhs_norm,
                f"{prefix}_solver_residual_target": global_solve.solver_residual_target,
                f"{prefix}_solver_rel_residual": global_solve.solver_relative_residual_norm,
                f"{prefix}_physical_residual": global_solve.physical_residual_norm,
                f"{prefix}_physical_rhs_norm": global_solve.physical_rhs_norm,
                f"{prefix}_physical_residual_target": global_solve.physical_residual_target,
                f"{prefix}_physical_rel_residual": global_solve.physical_relative_residual_norm,
                f"{prefix}_diagnostic_residual": global_solve.diagnostic_residual_norm,
                f"{prefix}_diagnostic_residual_target": global_solve.diagnostic_residual_target,
                f"{prefix}_diagnostic_rel_residual": global_solve.diagnostic_relative_residual_norm,
                f"{prefix}_preconditioner_time": global_solve.preconditioner_elapsed_seconds,
                f"{prefix}_krylov_time": global_solve.solve_elapsed_seconds,
            }
        )
    return row


def _compute_diagnostics(
        *,
        case,
        rho_field,
        poisson_result,
        step: int,
        time_value: float,
        baseline_mass: float,
        baseline_q_l2: float,
        equilibrium_potential=None,
        equilibrium_density=None,
        extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    phi_field = poisson_result.field
    q_l2 = _vector_l2_norm(poisson_result.flux)
    mass = _integral(rho_field)
    rho_min, rho_max = _field_min_max(rho_field)
    phi_min, phi_max = _field_min_max(phi_field)
    rho_l2_error, rho_linf_error = _field_exact_errors(rho_field, case.exact_density_at(time_value))
    phi_l2_error, phi_linf_error = _field_exact_errors(phi_field, case.exact_potential_at(time_value))
    row: dict[str, Any] = {
        "step": int(step),
        "time": float(time_value),
        "mass": mass,
        "mass_drift": mass - baseline_mass,
        "mass_relative_drift": _relative_drift(mass, baseline_mass),
        "q_l2": q_l2,
        "q_l2_drift": q_l2 - baseline_q_l2,
        "q_l2_relative_drift": _relative_drift(q_l2, baseline_q_l2),
        "energy_from_q_l2": 0.5 * q_l2 * q_l2,
        "rho_min": rho_min,
        "rho_max": rho_max,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "rho_l2_error": rho_l2_error,
        "rho_linf_error": rho_linf_error,
        "phi_l2_error": phi_l2_error,
        "phi_linf_error": phi_linf_error,
    }
    if equilibrium_potential is not None:
        phi_eq_l2 = _field_l2_difference(phi_field, equilibrium_potential)
        eq_norm = max(equilibrium_potential.l2_norm(), 1.0e-300)
        row["diocotron_phi_eq_l2"] = phi_eq_l2
        row["diocotron_phi_eq_relative_l2"] = phi_eq_l2 / eq_norm
        row["diocotron_phi_eq_linf"] = _field_linf_difference(phi_field, equilibrium_potential)
    if equilibrium_density is not None:
        rho_eq_l2 = _field_l2_difference(rho_field, equilibrium_density)
        eq_norm = max(equilibrium_density.l2_norm(), 1.0e-300)
        row["diocotron_rho_eq_l2"] = rho_eq_l2
        row["diocotron_rho_eq_relative_l2"] = rho_eq_l2 / eq_norm
    if extra:
        row.update(extra)
    row["diagnostics_time"] = time.perf_counter() - start
    return row


def _make_poisson_options(config: GuidingCenterRunPreset):
    from hdgfem.solvers.diff_rea import DiffusionReactionHDGOptions

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
        amgx_config=_load_amgx_config(
            config.poisson_amgx_config_path,
            tolerance=config.poisson_solver_rtol,
            absolute_tolerance=config.poisson_solver_atol,
        ),
        ilu_drop_tol=config.poisson_ilu_drop_tol,
        ilu_fill_factor=config.poisson_ilu_fill_factor,
        ilu_failure=config.poisson_ilu_failure,
        local_solver_backend=config.poisson_local_backend,
        assembly_backend=config.poisson_assembly_backend,
        trace_basis=config.trace_basis,
        raw_matrix_format=config.poisson_raw_matrix_format,
        raw_block_size=config.poisson_raw_block_size,
        boundary_mode="eliminate",
        hdg_postprocess=config.poisson_hdg_postprocess,
        verbose=_solver_verbosity(config),
    )


def _make_transport_options(config: GuidingCenterRunPreset, boundary_mode: str):
    from hdgfem.solvers.adv_rea import AdvectionReactionHDGOptions

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
        amgx_config=_load_amgx_config(config.transport_amgx_config_path, tolerance=config.transport_solver_rtol),
        ilu_drop_tol=config.transport_ilu_drop_tol,
        ilu_fill_factor=config.transport_ilu_fill_factor,
        ilu_failure=config.transport_ilu_failure,
        scale_system=config.transport_scale_system,
        boundary_mode=boundary_mode,
        trace_ordering=config.transport_trace_ordering,
        trace_ordering_flux_tolerance=config.transport_trace_ordering_flux_tolerance,
        ilu_permc_spec=config.transport_ilu_permc_spec,
        assembly_backend=config.transport_assembly_backend,
        trace_basis=config.trace_basis,
        raw_local_assembly=config.transport_raw_local_assembly,
        raw_lu_mode=config.transport_raw_lu_mode,
        raw_block_size=config.transport_raw_block_size,
        raw_matrix_format=config.transport_raw_matrix_format,
        materialize_host_system=config.transport_materialize_host_system,
        materialize_host_solution=config.transport_materialize_host_solution,
        advection_stabilization=config.transport_advection_stabilization,
        cache_local_solvers=config.transport_cache_local_solvers,
        verbose=_solver_verbosity(config),
    )


def _print_run_summary(result: GuidingCenterRunResult) -> None:
    if _verbosity_level(result.config) < 1:
        return
    from hdgfem.io.output import pretty_print_sections

    final = result.diagnostics[-1]
    pretty_print_sections(
        [
            (
                "Run / mesh",
                [
                    ("preset", result.preset_key, "s"),
                    ("case", result.case_key, "s"),
                    ("p", result.space.order, ",d"),
                    ("triangles", result.mesh.num_tri, ",d"),
                    ("steps", result.config.num_steps, ",d"),
                    ("dt", result.config.dt, ".4e"),
                ],
            ),
            (
                "Final diagnostics",
                [
                    ("time", final["time"], ".4e"),
                    ("mass drift", final["mass_relative_drift"], ".4e"),
                    ("q L2 drift", final["q_l2_relative_drift"], ".4e"),
                    ("rho L2 error", np.nan if final.get("rho_l2_error") is None else final["rho_l2_error"], ".4e"),
                    ("phi L2 error", np.nan if final.get("phi_l2_error") is None else final["phi_l2_error"], ".4e"),
                ],
            ),
            (
                "Outputs",
                [
                    ("CSV", str(result.csv_path), "s"),
                    ("JSONL", str(result.jsonl_path), "s"),
                ],
            ),
        ],
        title="Guiding-Center Run Summary",
    )


def run_guiding_center_case(config: GuidingCenterRunPreset, *, preset_key: str = "custom") -> GuidingCenterRunResult:
    """Run a fixed-mesh first-order semi-implicit guiding-center case."""
    _validate_config(config)
    from hdgfem.core.space import DGSpace
    from hdgfem.solvers.adv_rea import AdvectionReactionHDGSolver
    from hdgfem.solvers.diff_rea import DiffusionReactionHDGSolver, _timed_call
    from scripts.guiding_center.guiding_center_cases import case_definition_by_key

    case_definition = case_definition_by_key(config.case)
    case = case_definition.build(**config.case_params)
    mesh, _ = _timed_call(
        f"generating {case.default_domain if config.domain == 'auto' else config.domain} mesh",
        _phase_verbosity(config),
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
    rho_field = space.project_callable(case.initial_density_at(), name="rho_h")
    zero_reaction = _zero_field(space, name="zero_reaction_h")
    one_reaction = space.constant(1.0, name="one_reaction_h")
    poisson_options = _make_poisson_options(config)
    transport_boundary_mode = case.density_transport_boundary_mode if config.transport_boundary_mode == "auto" else config.transport_boundary_mode
    transport_options = _make_transport_options(config, transport_boundary_mode)

    equilibrium_potential = None
    equilibrium_density = None
    if case.equilibrium_density is not None:
        equilibrium_density = space.project_callable(case.equilibrium_density, name="rho_eq_h")
        equilibrium_solver = DiffusionReactionHDGSolver(
            space,
            source=equilibrium_density,
            reaction=zero_reaction,
            boundary_condition=case.potential_boundary_at(0.0),
            options=poisson_options,
        )
        equilibrium_result = equilibrium_solver.solve()
        equilibrium_potential = equilibrium_result.field
        equilibrium_solver.clear_cache()

    poisson_solver = DiffusionReactionHDGSolver(
        space,
        source=rho_field,
        reaction=zero_reaction,
        boundary_condition=case.potential_boundary_at(0.0),
        options=poisson_options,
    )
    poisson_result = poisson_solver.solve()
    transport_solver = AdvectionReactionHDGSolver(space, options=transport_options)

    baseline_mass = _integral(rho_field)
    baseline_q_l2 = _vector_l2_norm(poisson_result.flux)
    recorder = DiagnosticsRecorder(config.diagnostics_dir, config.diagnostics_prefix or preset_key)
    plotter = None
    try:
        initial_extra = _solver_metrics("poisson", poisson_result)
        initial_extra.update(
            {
                "phase": "initial",
                "beta_build_time": 0.0,
                "transport_time_total": 0.0,
                "transport_time_assembly": 0.0,
                "transport_time_solve": 0.0,
                "transport_time_reconstruction": 0.0,
                "transport_host_device_transfer_time": 0.0,
                "plot_time": 0.0,
            }
        )
        row = _compute_diagnostics(
            case=case,
            rho_field=rho_field,
            poisson_result=poisson_result,
            step=0,
            time_value=0.0,
            baseline_mass=baseline_mass,
            baseline_q_l2=baseline_q_l2,
            equilibrium_potential=equilibrium_potential,
            equilibrium_density=equilibrium_density,
            extra=initial_extra,
        )
        if config.plot_every > 0:
            plot_start = time.perf_counter()
            plotter = GuidingCenterPyVistaPanels(
                rho_field,
                poisson_result.field,
                resolution=config.plot_resolution,
                title=f"{preset_key}: {case.key}",
                show_mesh=config.plot_show_mesh,
                off_screen=config.plot_off_screen,
                screenshot_dir=config.screenshot_dir,
                screenshot_prefix=config.diagnostics_prefix or preset_key,
                include_potential=config.plot_potential,
            )
            plotter.update(rho_field, poisson_result.field, step=0, time_value=0.0)
            row["plot_time"] = time.perf_counter() - plot_start
        recorder.record(row)
        _print_step_summary(config, row)

        current_time = 0.0
        for step in range(1, config.num_steps + 1):
            beta_start = time.perf_counter()
            beta_h = _build_beta_from_flux(poisson_result.flux, config.dt, space)
            beta_build_time = time.perf_counter() - beta_start
            next_time = current_time + config.dt
            density_boundary = case.density_boundary_at(next_time) if transport_boundary_mode != "zero-flux" else None
            transport_solver.set_problem(rho_field, beta_h, one_reaction, density_boundary)
            transport_result = transport_solver.solve()
            rho_field = _ensure_field(transport_result, "transport", space, name="rho_h")
            poisson_solver.set_source(rho_field)
            poisson_solver.set_boundary_condition(case.potential_boundary_at(next_time))
            poisson_result = poisson_solver.solve()
            current_time = next_time

            extra = _solver_metrics("poisson", poisson_result)
            extra.update(_solver_metrics("transport", transport_result))
            extra.update(
                {
                    "phase": "step",
                    "beta_build_time": beta_build_time,
                    "poisson_time": poisson_result.timings.total,
                    "transport_time": transport_result.timings.total,
                    "host_device_transfer_time": _sum_detail_timings(poisson_result) + _sum_detail_timings(transport_result),
                    "plot_time": 0.0,
                }
            )
            row = _compute_diagnostics(
                case=case,
                rho_field=rho_field,
                poisson_result=poisson_result,
                step=step,
                time_value=current_time,
                baseline_mass=baseline_mass,
                baseline_q_l2=baseline_q_l2,
                equilibrium_potential=equilibrium_potential,
                equilibrium_density=equilibrium_density,
                extra=extra,
            )
            if config.plot_every > 0 and step % config.plot_every == 0:
                plot_start = time.perf_counter()
                if plotter is None:
                    plotter = GuidingCenterPyVistaPanels(
                        rho_field,
                        poisson_result.field,
                        resolution=config.plot_resolution,
                        title=f"{preset_key}: {case.key}",
                        show_mesh=config.plot_show_mesh,
                        off_screen=config.plot_off_screen,
                        screenshot_dir=config.screenshot_dir,
                        screenshot_prefix=config.diagnostics_prefix or preset_key,
                        include_potential=config.plot_potential,
                    )
                plotter.update(rho_field, poisson_result.field, step=step, time_value=current_time)
                row["plot_time"] = time.perf_counter() - plot_start
            recorder.record(row)
            _print_step_summary(config, row)
    finally:
        recorder.close()

    result = GuidingCenterRunResult(
        config=config,
        preset_key=preset_key,
        case_key=case.key,
        mesh=mesh,
        space=space,
        final_density=rho_field,
        final_potential=poisson_result.field,
        final_flux=poisson_result.flux,
        diagnostics=recorder.rows,
        csv_path=recorder.csv_path,
        jsonl_path=recorder.jsonl_path,
    )
    _print_run_summary(result)
    return result


def _add_solver_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--poisson-assembly-backend", choices=("numpy", "numba", "cupy", "raw-cuda", "auto"), default=None)
    parser.add_argument("--poisson-local-backend", choices=("numpy", "numba"), default=None)
    parser.add_argument("--poisson-solver", default=None)
    parser.add_argument("--poisson-preconditioner", default=None)
    parser.add_argument("--poisson-solver-rtol", type=float, default=None)
    parser.add_argument("--poisson-solver-atol", type=float, default=None)
    parser.add_argument("--poisson-maxiter", type=int, default=None)
    parser.add_argument("--poisson-scale-system", choices=("auto", "on", "off"), default=None)
    parser.add_argument("--poisson-petsc-preset", default=None)
    parser.add_argument("--poisson-petsc-levels", type=int, default=None)
    parser.add_argument("--poisson-cupyx-solver", default=None)
    parser.add_argument("--poisson-amgx-config", type=Path, default=None)
    parser.add_argument("--poisson-ilu-drop-tol", type=float, default=None)
    parser.add_argument("--poisson-ilu-fill-factor", type=float, default=None)
    parser.add_argument("--poisson-raw-matrix-format", choices=("coo", "csr"), default=None)
    parser.add_argument("--poisson-raw-block-size", type=int, default=None)
    parser.add_argument("--poisson-hdg-postprocess", choices=("none", "primal", "flux", "both"), default=None)

    parser.add_argument("--transport-assembly-backend", choices=("numpy", "numba", "cupy", "raw-cuda", "auto"), default=None)
    parser.add_argument("--transport-solver", default=None)
    parser.add_argument("--transport-preconditioner", default=None)
    parser.add_argument("--transport-solver-rtol", type=float, default=None)
    parser.add_argument("--transport-solver-atol", type=float, default=None)
    parser.add_argument("--transport-maxiter", type=int, default=None)
    parser.add_argument("--transport-scale-system", choices=("auto", "on", "off"), default=None)
    parser.add_argument("--transport-petsc-preset", default=None)
    parser.add_argument("--transport-petsc-levels", type=int, default=None)
    parser.add_argument("--transport-cupyx-solver", default=None)
    parser.add_argument("--transport-amgx-config", type=Path, default=None)
    parser.add_argument("--transport-ilu-drop-tol", type=float, default=None)
    parser.add_argument("--transport-ilu-fill-factor", type=float, default=None)
    parser.add_argument("--transport-boundary-mode", choices=("auto", "eliminate", "zero-flux", "penalty"), default=None)
    parser.add_argument("--transport-trace-ordering", choices=("none", "upwind-scc"), default=None)
    parser.add_argument("--transport-trace-ordering-flux-tolerance", type=float, default=None)
    parser.add_argument("--transport-ilu-permc-spec", choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"), default=None)
    parser.add_argument("--transport-raw-local-assembly", choices=("precomputed", "fused"), default=None)
    parser.add_argument("--transport-raw-lu-mode", choices=("safe", "coop"), default=None)
    parser.add_argument("--transport-raw-block-size", type=int, default=None)
    parser.add_argument("--transport-raw-matrix-format", choices=("auto", "coo", "csr"), default=None)
    parser.add_argument("--transport-materialize-host-system", action="store_true")
    parser.add_argument("--no-transport-materialize-host-system", action="store_true")
    parser.add_argument("--transport-materialize-host-solution", choices=("auto", "on", "off"), default=None)


def _main() -> None:
    parser = ArgumentParser(
        description="Run a fixed-mesh first-order semi-implicit guiding-center case.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Curated preset configuration lives in scripts/guiding_center/guiding_center_presets.py.\n"
            "Cases are registered in scripts/guiding_center/guiding_center_cases.py."
        ),
    )
    parser.add_argument("preset_name", nargs="?", default=None, choices=tuple(sorted(PRESETS)))
    parser.add_argument("--preset", dest="preset", choices=tuple(sorted(PRESETS)), default=None)
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--print-preset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend-profile", choices=("host", "device", "hybrid"), default=None)
    parser.add_argument("--case", choices=tuple(sorted(CASE_DEFINITIONS)), default=None)
    parser.add_argument("--case-param", action="append", default=None, help="override case parameter with key=value syntax")
    parser.add_argument("--domain", choices=("auto", "structured-rectangle", "rectangle", "disc", "triangle"), default=None)
    parser.add_argument("--mesh-size", "--lc", type=float, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--gmsh-verbosity", type=int, default=None)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--basis", choices=("dub_orth", "hier_C0", "bernstein"), default=None)
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default=None)
    parser.add_argument("--order", "-p", type=int, default=None)
    parser.add_argument("--volume-quadrature", choices=("auto", "symmetric", "duffy"), default=None)
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--poisson-tau", type=float, default=None)
    _add_solver_arguments(parser)
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="logging level: 0 quiet, 1 per-step summaries, 2 solver phase logs, 3 detailed backend/AMGX diagnostics",
    )
    parser.add_argument("--quiet", action="store_true", help="same as --verbosity 0")
    parser.add_argument("--plot", action="store_true", help="enable PyVista plotting every frame unless --plot-every is set")
    parser.add_argument("--plot-every", type=int, default=None, help="update PyVista panels every N steps; 0 disables plotting")
    parser.add_argument("--plot-resolution", type=int, default=None)
    parser.add_argument("--plot-both", action="store_true", help="plot density and potential; default plotting shows density only")
    parser.add_argument("--plot-off-screen", action="store_true")
    parser.add_argument("--no-plot-mesh", action="store_true")
    parser.add_argument("--screenshot-dir", type=Path, default=None)
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--diagnostics-prefix", default=None)
    args = parser.parse_args()

    if args.list_presets:
        print_presets()
        return

    preset_key = args.preset or args.preset_name or DEFAULT_PRESET
    config = _runtime_config(preset_by_key(preset_key), args)
    if args.print_preset or args.dry_run:
        print_preset_details(preset_key, config)
        return
    run_guiding_center_case(config, preset_key=preset_key)


if __name__ == "__main__":
    _main()
