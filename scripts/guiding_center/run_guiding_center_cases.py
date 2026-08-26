#!/usr/bin/env python3
"""Run fixed-mesh guiding-center cases."""

from __future__ import annotations

import ast
import csv
import json
import shlex
import math
import os
import sys
import threading
import time
import traceback
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem.core.field_ops import (
    field_linear_combination,
    perpendicular_vector_field,
    project_callable_to_trace,
    solution_field,
    solution_trace,
    trace_linear_combination,
    vector_field_linear_combination,
)
from hdgfem.diagnostics import (
    evaluate_scalar_error,
    guiding_center_field_diagnostics,
    relative_drift,
    result_transfer_time,
    solver_result_metrics,
)
from hdgfem.io.config import load_amgx_config
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
    timings_csv_path: Path
    timings_jsonl_path: Path
    terminal_log_path: Path | None = None


@dataclass(frozen=True)
class GuidingCenterStepSnapshot:
    """Accepted SI-Euler step state exposed to solver benchmark observers."""

    step: int
    time: float
    space: Any
    transport_source: Any
    transport_beta: Any
    transport_reaction: Any
    transport_boundary: Any
    transport_initial_guess: Any
    transport_result: Any
    accepted_density: Any
    poisson_boundary: Any
    poisson_initial_guess: Any
    poisson_result: Any


class GuidingCenterArgumentParser(ArgumentParser):
    """Argument parser that supports shell-like ``@file`` response files."""

    def convert_arg_line_to_args(self, arg_line: str):
        stripped = arg_line.strip()
        if not stripped or stripped.startswith("#"):
            return []
        return shlex.split(stripped, comments=True)


def _write_all_fd(fd: int, data: bytes) -> None:
    """Write a complete byte buffer to a file descriptor."""
    remaining = memoryview(data)
    while remaining:
        try:
            written = os.write(fd, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("file-descriptor write made no progress")
        remaining = remaining[written:]


def _flush_terminal_streams() -> None:
    """Flush Python and C stdio before redirecting or restoring descriptors."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        fflush = libc.fflush
        fflush.argtypes = [ctypes.c_void_p]
        fflush.restype = ctypes.c_int
        fflush(None)
    except (AttributeError, OSError):
        pass


class _TerminalLogTee:
    """Mirror process stdout/stderr to their original descriptors and one log."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._log_fd: int | None = None
        self._saved_fds: dict[int, int] = {}
        self._threads: list[threading.Thread] = []
        self._log_lock = threading.Lock()
        self._log_error: OSError | None = None

    def __enter__(self):
        _flush_terminal_streams()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644,
        )
        try:
            for target_fd in (1, 2):
                saved_fd = os.dup(target_fd)
                read_fd, write_fd = os.pipe()
                self._saved_fds[target_fd] = saved_fd
                thread = threading.Thread(
                    target=self._pump,
                    args=(read_fd, saved_fd),
                    name=f"guiding-center-log-fd{target_fd}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
                try:
                    os.dup2(write_fd, target_fd)
                finally:
                    os.close(write_fd)
        except BaseException:
            self._restore_descriptors()
            self._join_and_close()
            raise
        return self

    def _pump(self, read_fd: int, mirror_fd: int) -> None:
        try:
            while True:
                try:
                    chunk = os.read(read_fd, 64 * 1024)
                except InterruptedError:
                    continue
                if not chunk:
                    break
                with self._log_lock:
                    if self._log_error is None and self._log_fd is not None:
                        try:
                            _write_all_fd(self._log_fd, chunk)
                        except OSError as error:
                            self._log_error = error
                try:
                    _write_all_fd(mirror_fd, chunk)
                except OSError:
                    # A detached terminal must not stop draining the pipe or
                    # deadlock a long-running native solver.
                    pass
        finally:
            os.close(read_fd)

    def _restore_descriptors(self) -> None:
        for target_fd, saved_fd in self._saved_fds.items():
            try:
                os.dup2(saved_fd, target_fd)
            except OSError:
                pass

    def _join_and_close(self) -> None:
        for thread in self._threads:
            thread.join()
        for saved_fd in self._saved_fds.values():
            try:
                os.close(saved_fd)
            except OSError:
                pass
        if self._log_fd is not None:
            try:
                os.close(self._log_fd)
            finally:
                self._log_fd = None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        _flush_terminal_streams()
        self._restore_descriptors()
        self._join_and_close()
        if self._log_error is not None and exc_type is None:
            raise RuntimeError(f"failed to write terminal log {self.path}") from self._log_error
        return False


def _terminal_log_path(
        config: GuidingCenterRunPreset,
        preset_key: str,
) -> Path:
    """Return the terminal-log path matching the diagnostics output stem."""
    output_stem = str(config.diagnostics_prefix or preset_key).strip() or "guiding_center"
    return Path(config.diagnostics_dir) / f"{output_stem}.log"


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
        base_window_size = [1500, 650] if self.include_potential else [820, 720]
        window_size = [int(round(2.5 * extent)) for extent in base_window_size]
        self.plotter = self.pv.Plotter(shape=shape, window_size=window_size, off_screen=off_screen)
        self.off_screen = bool(off_screen)
        render_window = getattr(self.plotter, "render_window", None)
        render_window_name = (
            ""
            if render_window is None or not hasattr(render_window, "GetClassName")
            else str(render_window.GetClassName())
        )
        self._render_only = self.off_screen or any(
            marker in render_window_name for marker in ("EGL", "OSOpenGL", "Offscreen")
        )
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
            self.plotter.show(
                auto_close=False,
                interactive_update=not self._render_only,
            )
            self._shown = True
        elif self._render_only:
            # EGL/OSMesa windows have no X event queue. Calling Plotter.update()
            # would invoke an X interactor and raise MismatchedInteractorError.
            self.plotter.render()
        else:
            self.plotter.update()
        if self.screenshot_dir is not None:
            path = self.screenshot_dir / f"{self.screenshot_prefix}_step{int(step):05d}_t{float(time_value):.6f}.png"
            self.plotter.screenshot(str(path))

    def close(self) -> None:
        """Release the VTK render window and interactor resources."""
        self.plotter.close()


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


def _print_linear_step_summary(config: GuidingCenterRunPreset, row: dict[str, Any]) -> None:
    """Print one balanced, machine-readable summary for the accepted linear stages."""
    if _verbosity_level(config) < 3 or str(row.get("phase", "step")) != "step":
        return

    step = int(row.get("step", 0))
    step_label = f"{step:05d}/{int(config.num_steps):05d}"
    coupled_wall = _format_metric(row.get("linear_step_wall_time"), ".4f")
    beta_wall = _format_metric(row.get("beta_build_time"), ".4f")
    trace_wall = _format_metric(row.get("potential_trace_update_time"), ".4f")
    lines = [
        "",
        (
            f"[gc:linear] step {step_label} | t={_format_metric(row.get('time'), '.6f')} "
            f"| coupled wall={coupled_wall}s | beta={beta_wall}s | trace={trace_wall}s"
        ),
    ]

    for prefix, label in (("transport", "transport"), ("poisson", "poisson")):
        wall = _format_metric(row.get(f"{prefix}_step_wall_time"), ".4f")
        hdg_total = _format_metric(row.get(f"{prefix}_time"), ".4f")
        assembly = row.get(f"{prefix}_step_time_assembly")
        phase_label = "asm"
        if prefix == "poisson" and float(row.get("poisson_time_rhs_assembly", 0.0) or 0.0) > 0.0:
            phase_label = "rhs"
        solve = _format_metric(row.get(f"{prefix}_step_time_solve"), ".4f")
        reconstruction = _format_metric(row.get(f"{prefix}_step_time_reconstruction"), ".4f")
        iterations = _first_metric(
            row,
            f"{prefix}_step_iterations",
            f"{prefix}_solver_iterations",
        )
        iteration_text = "n/a" if iterations is None or int(iterations) < 0 else str(int(iterations))
        relative = _first_metric(
            row,
            f"{prefix}_physical_rel_residual",
            f"{prefix}_solver_rel_residual",
        )
        parts = [
            f"  {label:<9} HDG={hdg_total}s",
            f"stage wall={wall}s",
            f"{phase_label}={_format_metric(assembly, '.4f')}s",
            f"solve={solve}s",
            f"recon={reconstruction}s",
            f"it={iteration_text}",
            f"true_rel={_format_metric(relative)}",
        ]
        stage_count = int(row.get(f"{prefix}_stage_count", 1) or 1)
        if stage_count > 1:
            parts.append(f"stages={stage_count}")
        attempts = row.get(f"{prefix}_amgx_attempt_count")
        if attempts is not None and int(attempts) > 1:
            parts.append(f"attempts={int(attempts)}")
        if prefix == "poisson":
            reuse = []
            if bool(row.get("poisson_detail_raw_assembly_operator_reused", 0.0)):
                reuse.append("operator")
            hierarchy_reused = bool(
                row.get("poisson_detail_solve_fb_hp_mg_hierarchy_reused", 0.0)
            ) or bool(row.get("poisson_detail_solve_amgx_hierarchy_reused", 0.0))
            if hierarchy_reused:
                reuse.append("hierarchy")
            if reuse:
                parts.append("reuse=" + "+".join(reuse))
        lines.append(" | ".join(parts))

    print("\n".join(lines), flush=True)


def _print_step_summary(config: GuidingCenterRunPreset, row: dict[str, Any]) -> None:
    if _verbosity_level(config) < 1:
        return
    step = int(row.get("step", 0))
    phase = str(row.get("phase", "step"))
    step_label = "initial" if phase == "initial" else f"{step:05d}/{int(config.num_steps):05d}"
    if _verbosity_level(config) >= 2:
        _print_diagnostics_block(config, row, step_label=step_label, phase=phase)
        return

    pieces = [
        f"[gc] step={step_label}",
        f"t={_format_metric(row.get('time'), '.6f')}",
        f"mass_rel_drift={_format_metric(row.get('mass_relative_drift'))}",
        f"energy_rel_drift={_format_metric(row.get('energy_relative_drift'))}",
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
        pieces.append(f"instability_l2={_format_metric(row.get('diocotron_phi_eq_l2'))}")
    print(" ".join(pieces), flush=True)


def _print_diagnostics_block(
        config: GuidingCenterRunPreset,
        row: dict[str, Any],
        *,
        step_label: str,
        phase: str,
) -> None:
    """Print accepted-state physics separately from backend solver logs."""
    lines = [
        "",
        "=" * 78,
        (
            "GUIDING-CENTER ACCEPTED-STATE DIAGNOSTICS"
            f" | step {step_label} | t={_format_metric(row.get('time'), '.6f')}"
        ),
        "-" * 78,
        "Conservation",
        f"  mass                                  {_format_metric(row.get('mass'), '.10e')}",
        f"  relative mass drift                   {_format_metric(row.get('mass_relative_drift'))}",
        f"  electrostatic energy (1/2 ||q||²)     {_format_metric(row.get('energy_from_q_l2'), '.10e')}",
        f"  relative energy drift                 {_format_metric(row.get('energy_relative_drift'))}",
        f"  electric-field norm ||q|| L2          {_format_metric(row.get('q_l2'), '.10e')}",
    ]
    if row.get("q_l2_postprocessed") is not None:
        lines.append(
            "  postprocessed RT_p field norm L2      "
            f"{_format_metric(row.get('q_l2_postprocessed'), '.10e')}"
        )

    if row.get("diocotron_phi_eq_l2") is not None:
        lines.extend(
            [
                "",
                "Instability relative to equilibrium",
                f"  potential amplitude ||phi-phi_eq|| L2 {_format_metric(row.get('diocotron_phi_eq_l2'))}",
                f"  relative potential amplitude          {_format_metric(row.get('diocotron_phi_eq_relative_l2'))}",
                f"  potential difference Linf             {_format_metric(row.get('diocotron_phi_eq_linf'))}",
            ]
        )
    if row.get("diocotron_rho_eq_l2") is not None:
        lines.extend(
            [
                f"  density amplitude ||rho-rho_eq|| L2   {_format_metric(row.get('diocotron_rho_eq_l2'))}",
                f"  relative density amplitude            {_format_metric(row.get('diocotron_rho_eq_relative_l2'))}",
            ]
        )
    if row.get("diocotron_mode_1k_amplitude") is not None:
        mode = _format_metric(row.get("diocotron_mode_base"), ".0f")
        lines.extend(
            [
                (
                    f"  {f'normalized mode k={mode} amplitude':<38}"
                    f"{_format_metric(row.get('diocotron_mode_1k_amplitude'))}"
                ),
                f"  normalized mode 2k amplitude           {_format_metric(row.get('diocotron_mode_2k_amplitude'))}",
                f"  normalized mode 3k amplitude           {_format_metric(row.get('diocotron_mode_3k_amplitude'))}",
                f"  harmonic ratio (2k/k)                  {_format_metric(row.get('diocotron_harmonic_ratio'))}",
            ]
        )

    lines.extend(
        [
            "",
            "Field ranges",
            (
                "  density rho                           "
                f"[{_format_metric(row.get('rho_min'))}, {_format_metric(row.get('rho_max'))}]"
            ),
            (
                "  potential phi                        "
                f"[{_format_metric(row.get('phi_min'))}, {_format_metric(row.get('phi_max'))}]"
            ),
        ]
    )
    if row.get("rho_l2_error") is not None or row.get("phi_l2_error") is not None:
        lines.extend(
            [
                "",
                "Manufactured-solution errors",
                f"  density L2 / Linf                    {_format_metric(row.get('rho_l2_error'))} / {_format_metric(row.get('rho_linf_error'))}",
                f"  potential L2 / Linf                  {_format_metric(row.get('phi_l2_error'))} / {_format_metric(row.get('phi_linf_error'))}",
            ]
        )

    lines.extend(["", "Linear-solver checks"])
    if row.get("poisson_solver_rel_residual") is not None:
        lines.append(
            "  Poisson independently checked residual "
            f"{_format_metric(row.get('poisson_solver_rel_residual'))}"
        )
    if phase != "initial" and row.get("transport_solver_rel_residual") is not None:
        lines.append(
            "  transport independently checked residual "
            f"{_format_metric(row.get('transport_solver_rel_residual'))}"
        )

    lines.extend(["", "Phase timings"])
    timing_rows = [
        ("complete coupled linear step", row.get("linear_step_wall_time") if phase != "initial" else None),
        ("complete transport stage wall", row.get("transport_step_wall_time") if phase != "initial" else None),
        ("complete Poisson stage wall", row.get("poisson_step_wall_time") if phase != "initial" else None),
        ("beta construction", row.get("beta_build_time") if phase != "initial" else None),
        ("transport HDG solve", _first_metric(row, "transport_time_total", "transport_time") if phase != "initial" else None),
        ("Poisson HDG solve", _first_metric(row, "poisson_time_total", "poisson_time")),
    ]
    if phase == "initial":
        timing_rows.extend([
            ("first Poisson wall", row.get("first_poisson_wall_time")),
            ("first Poisson operator assembly", row.get("first_poisson_time_operator_assembly")),
            ("first Poisson native hierarchy", row.get("first_poisson_detail_solve_fb_hp_mg_setup_outer")),
            ("first Poisson Krylov", row.get("first_poisson_krylov_time")),
            ("first Poisson reconstruction", row.get("first_poisson_time_reconstruction")),
            ("reused initial-state Poisson wall", row.get("initial_poisson_wall_time")),
        ])
    elif row.get("poisson_time_rhs_assembly"):
        timing_rows.append(("Poisson cached RHS-only assembly", row.get("poisson_time_rhs_assembly")))
    timing_rows.extend([
        ("accepted potential trace", row.get("potential_trace_update_time")),
        ("plot update", row.get("plot_time") if row.get("plot_time") else None),
        ("accepted-state diagnostics", row.get("diagnostics_time")),
        ("post-Poisson application work", row.get("post_poisson_application_time")),
    ])
    for label, value in timing_rows:
        if value is not None:
            lines.append(f"  {label:<38} {_format_metric(value, '.5f')} s")
    if _verbosity_level(config) >= 3:
        lines.extend(
            [
                "  diagnostics: core                     "
                f"{_format_metric(row.get('diagnostics_core_time'), '.5f')} s",
                "  diagnostics: equilibrium potential    "
                f"{_format_metric(row.get('diagnostics_equilibrium_potential_time'), '.5f')} s",
                "  diagnostics: equilibrium density      "
                f"{_format_metric(row.get('diagnostics_equilibrium_density_time'), '.5f')} s",
                "  diagnostics: azimuthal modes           "
                f"{_format_metric(row.get('diagnostics_azimuthal_mode_time'), '.5f')} s",
            ]
        )
    lines.extend(["=" * 78, ""])
    print("\n".join(lines), flush=True)


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


_AMGX_DEFAULT_RTOL = 1.0e-11
_AMGX_DEFAULT_ATOL = 1.0e-12


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


def _electric_flux(poisson_result):
    """Return the accepted higher-order electric field when available."""
    return poisson_result.postprocessed_flux or poisson_result.flux


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


def _runtime_config(config: GuidingCenterRunPreset, args) -> GuidingCenterRunPreset:
    updates: dict[str, Any] = {}
    if args.backend_profile is not None:
        updates.update(_backend_profile_updates(args.backend_profile))
    direct_updates = {
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
        "volume_quad_1d": args.volume_quad_1d,
        "edge_quad_1d": args.edge_quad_1d,
        "dt": args.dt,
        "num_steps": args.num_steps,
        "time_scheme": args.time_scheme,
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
        "poisson_ilu_permc_spec": args.poisson_ilu_permc_spec,
        "poisson_raw_matrix_format": args.poisson_raw_matrix_format,
        "poisson_raw_block_size": args.poisson_raw_block_size,
        "poisson_cache_local_factors": args.poisson_cache_local_factors,
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
        "transport_retry_amgx_config_path": None if args.transport_retry_amgx_config is None else str(args.transport_retry_amgx_config),
        "verbosity": args.verbosity,
        "plot_every": args.plot_every,
        "plot_resolution": args.plot_resolution,
        "plot_potential": True if args.plot_both else None,
        "screenshot_dir": None if args.screenshot_dir is None else str(args.screenshot_dir),
        "diagnostics_dir": None if args.diagnostics_dir is None else str(args.diagnostics_dir),
        "diagnostics_prefix": args.diagnostics_prefix,
        "diagnostics_every": args.diagnostics_every,
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
    if (
        args.backend_profile == "device"
        and not 4 <= int(runtime.order) <= 6
        and str(runtime.poisson_solver).replace("_", "-").lower() == "fb-hp-mg-pcg"
    ):
        runtime = replace(runtime, poisson_solver="amgx")
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
    if config.minimum_triangles < 0:
        raise ValueError("minimum_triangles must be nonnegative")
    if not 0 <= int(config.verbosity) <= 3:
        raise ValueError("verbosity must be one of 0, 1, 2, or 3")
    if config.time_scheme not in {"si-euler", "predictor-corrector"}:
        raise ValueError("time_scheme must be 'si-euler' or 'predictor-corrector'")
    if config.diagnostics_every < 1:
        raise ValueError("diagnostics_every must be positive")
    if config.poisson_flux_postprocess_every < 0:
        raise ValueError("poisson_flux_postprocess_every must be nonnegative")
    if config.transport_retry_policy not in {"none", "amgx-robust"}:
        raise ValueError("transport_retry_policy must be 'none' or 'amgx-robust'")
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
        native = str(config.poisson_solver).replace("_", "-").lower() == "fb-hp-mg-pcg"
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
        if config.poisson_hdg_postprocess != "none":
            raise ValueError(
                "set poisson_flux_postprocess_every for periodic raw-CUDA flux postprocessing"
            )


def _fixed_operator_trace_predictor(current, previous=None, older=None):
    """Predict the next trace from up to three accepted fixed-operator solves."""
    if older is not None:
        return (
            trace_linear_combination(
                [(3.0, current), (-3.0, previous), (1.0, older)]
            ),
            2,
        )
    if previous is not None:
        return (
            trace_linear_combination([(2.0, current), (-1.0, previous)]),
            1,
        )
    return current, 0


def _average_boundary_data(left, right):
    """Average two time-level boundary callables for the midpoint solve."""
    if left is None or right is None:
        return None
    return lambda x, y: 0.5 * (left(x, y) + right(x, y))


def _build_beta_from_flux_pair(left_flux, right_flux, dt: float, space):
    """Build ``(dt/2) * v_mid = (dt/4) * (q_left + q_right)^perp``."""
    midpoint_flux = vector_field_linear_combination(
        space,
        [(0.5, left_flux), (0.5, right_flux)],
        name="q_mid_h",
    )
    return perpendicular_vector_field(midpoint_flux, 0.5 * float(dt), space, name="beta_h")


def _compute_diagnostics(
        *,
        case,
        rho_field,
        poisson_result,
        step: int,
        time_value: float,
        baseline_mass: float | None,
        baseline_q_l2: float | None,
        equilibrium_potential=None,
        equilibrium_density=None,
        equilibrium_potential_l2: float | None = None,
        equilibrium_density_l2: float | None = None,
        extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    core_start = time.perf_counter()
    phi_field = poisson_result.field
    field_metrics = guiding_center_field_diagnostics(
        rho_field,
        phi_field,
        poisson_result.flux,
        postprocessed_flux=poisson_result.postprocessed_flux,
        equilibrium_potential=equilibrium_potential,
        equilibrium_density=equilibrium_density,
        mode=int(case.parameters.get("k", 0)),
        backend="auto",
    )
    standard_q_l2 = float(field_metrics["q_l2_standard"])
    postprocessed_q_l2 = field_metrics.get("q_l2_postprocessed")
    if postprocessed_q_l2 is not None:
        postprocessed_q_l2 = float(postprocessed_q_l2)
    q_l2 = standard_q_l2
    mass = float(field_metrics["mass"])
    effective_baseline_mass = mass if baseline_mass is None else float(baseline_mass)
    effective_baseline_q_l2 = q_l2 if baseline_q_l2 is None else float(baseline_q_l2)
    exact_density = case.exact_density_at(time_value)
    exact_potential = case.exact_potential_at(time_value)
    rho_report = (
        None
        if exact_density is None
        else evaluate_scalar_error(rho_field, exact_density, backend="auto")
    )
    phi_report = (
        None
        if exact_potential is None
        else evaluate_scalar_error(phi_field, exact_potential, backend="auto")
    )
    rho_l2_error = None if rho_report is None else rho_report.metrics.l2
    rho_linf_error = None if rho_report is None else rho_report.metrics.linf
    phi_l2_error = None if phi_report is None else phi_report.metrics.l2
    phi_linf_error = None if phi_report is None else phi_report.metrics.linf
    core_time = time.perf_counter() - core_start
    diagnostic_backend = str(field_metrics["diagnostics_backend"])
    row: dict[str, Any] = {
        "step": int(step),
        "time": float(time_value),
        "mass": mass,
        "mass_drift": mass - effective_baseline_mass,
        "mass_relative_drift": relative_drift(mass, effective_baseline_mass),
        "q_l2": q_l2,
        "q_l2_standard": standard_q_l2,
        "q_l2_postprocessed": postprocessed_q_l2,
        "electric_flux_postprocessed": poisson_result.postprocessed_flux is not None,
        "q_l2_drift": q_l2 - effective_baseline_q_l2,
        "q_l2_relative_drift": relative_drift(q_l2, effective_baseline_q_l2),
        "energy_from_q_l2": 0.5 * q_l2 * q_l2,
        "energy_drift": 0.5 * (q_l2 * q_l2 - effective_baseline_q_l2 * effective_baseline_q_l2),
        "energy_relative_drift": relative_drift(
            0.5 * q_l2 * q_l2,
            0.5 * effective_baseline_q_l2 * effective_baseline_q_l2,
        ),
        "rho_min": float(field_metrics["rho_min"]),
        "rho_max": float(field_metrics["rho_max"]),
        "phi_min": float(field_metrics["phi_min"]),
        "phi_max": float(field_metrics["phi_max"]),
        "rho_l2_error": rho_l2_error,
        "rho_linf_error": rho_linf_error,
        "phi_l2_error": phi_l2_error,
        "phi_linf_error": phi_linf_error,
        "diagnostics_backend": diagnostic_backend,
        "diagnostics_core_time": core_time,
        "diagnostics_device_reduction_time": core_time if diagnostic_backend == "cuda" else 0.0,
        "diagnostics_equilibrium_potential_time": 0.0,
        "diagnostics_equilibrium_density_time": 0.0,
        "diagnostics_azimuthal_mode_time": 0.0,
    }
    if equilibrium_potential is not None:
        phi_eq_l2 = float(field_metrics["diocotron_phi_eq_l2"])
        eq_norm = max(
            float(field_metrics["diocotron_phi_eq_reference_l2"])
            if equilibrium_potential_l2 is None
            else float(equilibrium_potential_l2),
            1.0e-300,
        )
        row["diocotron_phi_eq_l2"] = phi_eq_l2
        row["diocotron_phi_eq_relative_l2"] = phi_eq_l2 / eq_norm
        row["diocotron_phi_eq_linf"] = float(field_metrics["diocotron_phi_eq_linf"])
        row["diocotron_phi_eq_reference_l2"] = eq_norm
    if equilibrium_density is not None:
        rho_eq_l2 = float(field_metrics["diocotron_rho_eq_l2"])
        eq_norm = max(
            float(field_metrics["diocotron_rho_eq_reference_l2"])
            if equilibrium_density_l2 is None
            else float(equilibrium_density_l2),
            1.0e-300,
        )
        row["diocotron_rho_eq_l2"] = rho_eq_l2
        row["diocotron_rho_eq_relative_l2"] = rho_eq_l2 / eq_norm
        row["diocotron_rho_eq_reference_l2"] = eq_norm
        for key in (
            "diocotron_mode_base",
            "diocotron_mode_1k_amplitude",
            "diocotron_mode_2k_amplitude",
            "diocotron_mode_3k_amplitude",
            "diocotron_harmonic_ratio",
        ):
            if key in field_metrics:
                row[key] = float(field_metrics[key])
    if extra:
        row.update(extra)
    row["diagnostics_time"] = time.perf_counter() - start
    return row


def _make_poisson_options(config: GuidingCenterRunPreset):
    from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGOptions

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
                Path(__file__).resolve().parents[2]
                / "configs"
                / "amgx"
                / "adv_rea_gpu4_hdg_fgmres_dilu_abs.json"
            )
        fallback_config = _load_amgx_config(
            fallback_path,
            tolerance=config.transport_solver_rtol,
            absolute_tolerance=config.transport_solver_atol,
        )
        primary_config = _load_amgx_config(
            config.transport_amgx_config_path,
            tolerance=primary_amgx_tolerance,
        )
        retry_attempts = (
            {
                "label": "primary-zero",
                "config": primary_config,
                "use_initial_guess": False,
                "scale_system": config.transport_scale_system,
            },
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
        amgx_config=_load_amgx_config(
            config.transport_amgx_config_path, tolerance=primary_amgx_tolerance
        ),
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
        verbose=_solver_verbosity(config),
    )


def _print_run_summary(result: GuidingCenterRunResult) -> None:
    if _verbosity_level(result.config) < 1:
        return
    from hdgfem.io.output import pretty_print_sections

    final = result.diagnostics[-1]
    run_rows = [
        ("preset", result.preset_key, "s"),
        ("case", result.case_key, "s"),
        ("DG order", result.space.order, ",d"),
        ("triangles", result.mesh.num_tri, ",d"),
        ("steps", result.config.num_steps, ",d"),
        ("dt", result.config.dt, ".4e"),
    ]
    radial_power = result.config.case_params.get("p")
    if radial_power is not None:
        run_rows.insert(3, ("radial p", radial_power, ".4g"))
    final_rows = [
        ("time", final["time"], ".4e"),
        ("relative mass drift", final["mass_relative_drift"], ".4e"),
        ("electrostatic energy", final["energy_from_q_l2"], ".4e"),
        ("relative energy drift", final["energy_relative_drift"], ".4e"),
    ]
    if final.get("diocotron_phi_eq_l2") is not None:
        final_rows.extend(
            [
                ("||phi - phi_eq|| L2", final["diocotron_phi_eq_l2"], ".4e"),
                ("relative equilibrium departure", final["diocotron_phi_eq_relative_l2"], ".4e"),
            ]
        )
    if final.get("diocotron_mode_1k_amplitude") is not None:
        final_rows.extend(
            [
                ("normalized k-mode amplitude", final["diocotron_mode_1k_amplitude"], ".4e"),
                ("2k/k harmonic ratio", final["diocotron_harmonic_ratio"], ".4e"),
            ]
        )
    if final.get("rho_l2_error") is not None:
        final_rows.append(("rho L2 error", final["rho_l2_error"], ".4e"))
    if final.get("phi_l2_error") is not None:
        final_rows.append(("phi L2 error", final["phi_l2_error"], ".4e"))
    output_rows = [
        ("Diagnostics CSV", str(result.csv_path), "s"),
        ("Diagnostics JSONL", str(result.jsonl_path), "s"),
        ("Every-step timings CSV", str(result.timings_csv_path), "s"),
        ("Every-step timings JSONL", str(result.timings_jsonl_path), "s"),
    ]
    if result.terminal_log_path is not None:
        output_rows.append(("Terminal log", str(result.terminal_log_path), "s"))
    pretty_print_sections(
        [
            (
                "Run / mesh",
                run_rows,
            ),
            (
                "Final diagnostics",
                final_rows,
            ),
            (
                "Outputs",
                output_rows,
            ),
        ],
        title="Guiding-Center Run Summary",
    )



def run_guiding_center_case(
        config: GuidingCenterRunPreset,
        *,
        preset_key: str = "custom",
        step_observer: Callable[[GuidingCenterStepSnapshot], None] | None = None,
        terminal_log_path: str | Path | None = None,
) -> GuidingCenterRunResult:
    """Run a fixed-mesh guiding-center case with the selected time scheme."""
    _validate_config(config)
    if step_observer is not None and config.time_scheme != "si-euler":
        raise ValueError("step_observer currently supports only the accepted SI-Euler stage")

    from hdgfem.core.space import DGSpace
    from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver
    from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGSolver
    from hdgfem.io.output import timed_call
    from scripts.guiding_center.guiding_center_cases import case_definition_by_key

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
    rho_field, _ = timed_call(
        "[gc:init] projecting initial density",
        _detail_verbosity(config),
        lambda: space.project_callable(case.initial_density_at(), name="rho_h"),
    )
    solver_data_start = time.perf_counter()
    if _detail_verbosity(config):
        print("[gc:init] preparing solver fields and options ... ", end="", flush=True)
    zero_reaction = space.zeros(name="zero_reaction_h")
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
    if case.equilibrium_density is not None:
        equilibrium_density, _ = timed_call(
            "[gc:init] projecting equilibrium density",
            _detail_verbosity(config),
            lambda: space.project_callable(case.equilibrium_density, name="rho_eq_h"),
        )
        equilibrium_solver, _ = timed_call(
            "[gc:init] constructing equilibrium Poisson solver",
            _detail_verbosity(config),
            lambda: DiffusionReactionHDGSolver(
                space,
                source=equilibrium_density,
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
        # The guiding-center Poisson operator is fixed for the complete run.
        # Retain the equilibrium solver unconditionally so the perturbed initial
        # state and every accepted step reuse its trace operator, local factors,
        # global factorization/preconditioner, and AMGX hierarchy.
        poisson_solver = equilibrium_solver
        poisson_initial_guess = solution_trace(equilibrium_result, space, reduced=False)
        if config.poisson_preconditioner is not None:
            global_result = equilibrium_result.global_solve_result
            reusable_preconditioner = None if global_result is None else global_result.preconditioner
            if reusable_preconditioner is None:
                raise RuntimeError(
                    "fixed guiding-center Poisson reuse requires a reusable preconditioner, "
                    "but the equilibrium solve did not produce one"
                )
            poisson_solver.options = poisson_solver.options.with_overrides(
                preconditioner=reusable_preconditioner
            )
        poisson_solver.set_source(rho_field)
        poisson_solver.set_boundary_condition(case.potential_boundary_at(0.0))

    if poisson_solver is None:
        poisson_solver, _ = timed_call(
            "[gc:init] constructing initial Poisson solver",
            _detail_verbosity(config),
            lambda: DiffusionReactionHDGSolver(
                space,
                source=rho_field,
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
    potential_trace = solution_trace(poisson_result, space, reduced=False)
    previous_potential_trace = None
    older_potential_trace = None

    baseline_mass = None
    baseline_q_l2 = None
    output_stem = config.diagnostics_prefix or preset_key
    headless_plot = config.plot_every > 0 and not bool(os.environ.get("DISPLAY"))
    effective_plot_off_screen = bool(config.plot_off_screen or headless_plot)
    effective_screenshot_dir = config.screenshot_dir
    if headless_plot and effective_screenshot_dir is None:
        effective_screenshot_dir = str(
            Path(config.diagnostics_dir) / f"{output_stem}_frames"
        )
    if headless_plot and _phase_verbosity(config):
        print(
            "[gc:init] DISPLAY is unavailable; using PyVista EGL/off-screen "
            f"rendering and saving frames to {effective_screenshot_dir}",
            flush=True,
        )
    recorder = DiagnosticsRecorder(config.diagnostics_dir, output_stem)
    timing_recorder = DiagnosticsRecorder(config.diagnostics_dir, f"{output_stem}_timings")
    plotter = None
    try:
        initial_extra = solver_result_metrics("poisson", poisson_result)
        initial_extra.update(solver_result_metrics("first_poisson", first_poisson_result))
        initial_extra.update(
            {
                "phase": "initial",
                "time_scheme": config.time_scheme,
                "beta_build_time": 0.0,
                "poisson_time": initialization_poisson_wall_time,
                "poisson_time_total": initialization_poisson_wall_time,
                "first_poisson_wall_time": first_poisson_wall_time,
                "initial_poisson_wall_time": initial_poisson_wall_time,
                "equilibrium_poisson_wall_time": (
                    first_poisson_wall_time
                    if first_poisson_result is not poisson_result
                    else 0.0
                ),
                "poisson_flux_postprocessed": False,
                "transport_time": 0.0,
                "transport_time_total": 0.0,
                "plot_time": 0.0,
                "plot_headless": headless_plot,
                "plot_off_screen_effective": effective_plot_off_screen,
                "plot_screenshot_dir_effective": effective_screenshot_dir,
            }
        )
        timing_recorder.record({"step": 0, "time": 0.0, **initial_extra})
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
            equilibrium_potential_l2=equilibrium_potential_l2,
            equilibrium_density_l2=equilibrium_density_l2,
            extra=initial_extra,
        )
        baseline_mass = float(row["mass"])
        baseline_q_l2 = float(row["q_l2"])
        equilibrium_potential_l2 = row.get("diocotron_phi_eq_reference_l2")
        equilibrium_density_l2 = row.get("diocotron_rho_eq_reference_l2")
        if config.plot_every > 0:
            plot_start = time.perf_counter()
            plotter = GuidingCenterPyVistaPanels(
                rho_field,
                poisson_result.field,
                resolution=config.plot_resolution,
                title=f"{preset_key}: {case.key}",
                show_mesh=config.plot_show_mesh,
                off_screen=effective_plot_off_screen,
                screenshot_dir=effective_screenshot_dir,
                screenshot_prefix=config.diagnostics_prefix or preset_key,
                include_potential=config.plot_potential,
            )
            plotter.update(rho_field, poisson_result.field, step=0, time_value=0.0)
            row["plot_time"] = time.perf_counter() - plot_start
        recorder.record(row)
        _print_step_summary(config, row)

        current_time = 0.0
        for step in range(1, config.num_steps + 1):
            linear_step_start = time.perf_counter()
            linear_step_end = linear_step_start
            transport_step_wall_time = 0.0
            poisson_step_wall_time = 0.0
            transport_stage_results = []
            poisson_stage_results = []
            next_time = current_time + config.dt
            endpoint_density_boundary = (
                None if transport_boundary_mode == "zero-flux" else case.density_boundary_at(next_time)
            )
            endpoint_poisson_boundary = case.potential_boundary_at(next_time)
            step_transport_source = rho_field
            step_transport_initial_guess = density_trace
            accepted_potential_trace = potential_trace
            step_poisson_initial_guess, poisson_predictor_order = _fixed_operator_trace_predictor(
                potential_trace,
                previous_potential_trace,
                older_potential_trace,
            )
            potential_trace_time = 0.0
            post_poisson_start = None

            beta_start = time.perf_counter()
            predictor_beta = perpendicular_vector_field(
                poisson_result.flux, config.dt, space
            )
            beta_build_time = time.perf_counter() - beta_start

            transport_stage_start = time.perf_counter()
            transport_solver.set_problem(
                rho_field,
                predictor_beta,
                one_reaction,
                endpoint_density_boundary,
            )
            predictor_transport_result = transport_solver.solve(initial_guess=density_trace)
            if config.transport_reuse_first_preconditioner and not transport_preconditioner_reused:
                global_result = predictor_transport_result.global_solve_result
                reusable_preconditioner = None if global_result is None else global_result.preconditioner
                if reusable_preconditioner is None:
                    raise RuntimeError(
                        "transport_reuse_first_preconditioner requested reuse, but the first "
                        "transport solve did not produce a preconditioner"
                    )
                transport_solver.options = transport_solver.options.with_overrides(
                    preconditioner=reusable_preconditioner
                )
                transport_preconditioner_reused = True
            predictor_density = solution_field(
                predictor_transport_result, space, name="rho_predictor_h",
            )
            predictor_density_trace = solution_trace(
                predictor_transport_result,
                space,
                reduced=True,
            )
            transport_step_wall_time += time.perf_counter() - transport_stage_start
            transport_stage_results.append(predictor_transport_result)

            if config.time_scheme == "si-euler":
                rho_field = predictor_density
                density_trace = predictor_density_trace
                transport_result = predictor_transport_result
                poisson_stage_start = time.perf_counter()
                poisson_solver.set_source(rho_field)
                poisson_solver.set_boundary_condition(case.potential_boundary_at(next_time))
                poisson_result = poisson_solver.solve(
                    initial_guess=step_poisson_initial_guess,
                    **_poisson_postprocess_overrides(config, step),
                )
                post_poisson_start = time.perf_counter()
                potential_trace, potential_trace_time = timed_call(
                    "[gc] updating accepted potential trace",
                    _detail_verbosity(config),
                    lambda: solution_trace(poisson_result, space, reduced=False),
                )
                poisson_step_wall_time += time.perf_counter() - poisson_stage_start
                poisson_stage_results.append(poisson_result)
                linear_step_end = time.perf_counter()
                stage_extra: dict[str, Any] = {}
                poisson_time = poisson_result.timings.total
                transport_time = transport_result.timings.total
                if step_observer is not None:
                    step_observer(
                        GuidingCenterStepSnapshot(
                            step=step,
                            time=next_time,
                            space=space,
                            transport_source=step_transport_source,
                            transport_beta=predictor_beta,
                            transport_reaction=one_reaction,
                            transport_boundary=endpoint_density_boundary,
                            transport_initial_guess=step_transport_initial_guess,
                            transport_result=transport_result,
                            accepted_density=rho_field,
                            poisson_boundary=endpoint_poisson_boundary,
                            poisson_initial_guess=step_poisson_initial_guess,
                            poisson_result=poisson_result,
                        )
                    )
            else:
                poisson_stage_start = time.perf_counter()
                poisson_solver.set_source(predictor_density)
                poisson_solver.set_boundary_condition(case.potential_boundary_at(next_time))
                predictor_poisson_result = poisson_solver.solve(initial_guess=step_poisson_initial_guess)
                predictor_potential_trace = solution_trace(
                    predictor_poisson_result,
                    space,
                    reduced=False,
                )
                poisson_step_wall_time += time.perf_counter() - poisson_stage_start
                poisson_stage_results.append(predictor_poisson_result)

                midpoint_beta_start = time.perf_counter()
                midpoint_beta = _build_beta_from_flux_pair(
                    poisson_result.flux,
                    predictor_poisson_result.flux,
                    config.dt,
                    space,
                )
                beta_build_time += time.perf_counter() - midpoint_beta_start
                midpoint_boundary = (
                    None
                    if transport_boundary_mode == "zero-flux"
                    else _average_boundary_data(
                        case.density_boundary_at(current_time),
                        endpoint_density_boundary,
                    )
                )
                midpoint_trace_guess = trace_linear_combination(
                    [(0.5, density_trace), (0.5, predictor_density_trace)]
                )
                transport_stage_start = time.perf_counter()
                transport_solver.set_problem(
                    rho_field,
                    midpoint_beta,
                    one_reaction,
                    midpoint_boundary,
                )
                transport_result = transport_solver.solve(initial_guess=midpoint_trace_guess)
                midpoint_density = solution_field(
                    transport_result, space, name="rho_midpoint_h",
                )
                midpoint_density_trace = solution_trace(
                    transport_result,
                    space,
                    reduced=True,
                )
                rho_field = field_linear_combination(
                    space,
                    [(2.0, midpoint_density), (-1.0, rho_field)],
                    name="rho_h",
                )
                density_trace = trace_linear_combination(
                    [(2.0, midpoint_density_trace), (-1.0, density_trace)]
                )
                transport_step_wall_time += time.perf_counter() - transport_stage_start
                transport_stage_results.append(transport_result)

                poisson_stage_start = time.perf_counter()
                poisson_solver.set_source(rho_field)
                poisson_solver.set_boundary_condition(case.potential_boundary_at(next_time))
                poisson_result = poisson_solver.solve(
                    initial_guess=predictor_potential_trace,
                    **_poisson_postprocess_overrides(config, step),
                )
                post_poisson_start = time.perf_counter()
                potential_trace, potential_trace_time = timed_call(
                    "[gc] updating accepted potential trace",
                    _detail_verbosity(config),
                    lambda: solution_trace(poisson_result, space, reduced=False),
                )
                poisson_step_wall_time += time.perf_counter() - poisson_stage_start
                poisson_stage_results.append(poisson_result)
                linear_step_end = time.perf_counter()
                poisson_time = predictor_poisson_result.timings.total + poisson_result.timings.total
                transport_time = predictor_transport_result.timings.total + transport_result.timings.total
                stage_extra = solver_result_metrics("predictor_transport", predictor_transport_result)
                stage_extra.update(solver_result_metrics("predictor_poisson", predictor_poisson_result))
                stage_extra.update(solver_result_metrics("corrector_transport", transport_result))
                stage_extra.update(solver_result_metrics("final_poisson", poisson_result))

            linear_step_wall_time = linear_step_end - linear_step_start
            stage_extra["poisson_predictor_order"] = poisson_predictor_order
            timing_row = solver_result_metrics("poisson", poisson_result)
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
            older_potential_trace = previous_potential_trace
            previous_potential_trace = accepted_potential_trace

            current_time = next_time
            should_plot = config.plot_every > 0 and step % config.plot_every == 0
            should_record = step % config.diagnostics_every == 0 or step == config.num_steps
            plot_elapsed = 0.0
            if should_plot:
                if _detail_verbosity(config):
                    print("[gc] updating PyVista plot ... ", end="", flush=True)
                plot_start = time.perf_counter()
                if plotter is None:
                    plotter = GuidingCenterPyVistaPanels(
                        rho_field,
                        poisson_result.field,
                        resolution=config.plot_resolution,
                        title=f"{preset_key}: {case.key}",
                        show_mesh=config.plot_show_mesh,
                        off_screen=effective_plot_off_screen,
                        screenshot_dir=effective_screenshot_dir,
                        screenshot_prefix=config.diagnostics_prefix or preset_key,
                        include_potential=config.plot_potential,
                    )
                plotter.update(rho_field, poisson_result.field, step=step, time_value=current_time)
                plot_elapsed = time.perf_counter() - plot_start
                if _detail_verbosity(config):
                    print(f"done in {plot_elapsed:.5f}s", flush=True)

            if should_record:
                extra = solver_result_metrics("poisson", poisson_result)
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
                        "host_device_transfer_time": (
                            result_transfer_time(poisson_result)
                            + result_transfer_time(transport_result)
                            + sum(
                                float(value)
                                for key, value in stage_extra.items()
                                if key.endswith("host_device_transfer_time")
                            )
                        ),
                        "plot_time": plot_elapsed,
                        "potential_trace_update_time": potential_trace_time,
                    }
                )
                row, diagnostics_wall_time = timed_call(
                    "[gc] computing accepted-step diagnostics",
                    _detail_verbosity(config),
                    lambda: _compute_diagnostics(
                        case=case,
                        rho_field=rho_field,
                        poisson_result=poisson_result,
                        step=step,
                        time_value=current_time,
                        baseline_mass=baseline_mass,
                        baseline_q_l2=baseline_q_l2,
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
        try:
            if plotter is not None:
                plotter.close()
        finally:
            try:
                recorder.close()
            finally:
                timing_recorder.close()

    result = GuidingCenterRunResult(
        config=config,
        preset_key=preset_key,
        case_key=case.key,
        mesh=mesh,
        space=space,
        final_density=rho_field,
        final_potential=poisson_result.field,
        final_flux=_electric_flux(poisson_result),
        diagnostics=recorder.rows,
        csv_path=recorder.csv_path,
        jsonl_path=recorder.jsonl_path,
        timings_csv_path=timing_recorder.csv_path,
        timings_jsonl_path=timing_recorder.jsonl_path,
        terminal_log_path=(
            None if terminal_log_path is None else Path(terminal_log_path)
        ),
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
    parser.add_argument(
        "--poisson-ilu-permc-spec",
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        default=None,
    )
    parser.add_argument(
        "--poisson-raw-matrix-format", choices=("auto", "coo", "csr", "bsr"), default=None
    )
    parser.add_argument("--poisson-raw-block-size", choices=("auto", "1", "32", "64", "128"), default=None)
    parser.add_argument(
        "--poisson-cache-local-factors",
        choices=("none", "schur-lu", "schur-cholesky"),
        default=None,
    )
    parser.add_argument("--poisson-hdg-postprocess", choices=("none", "primal", "flux", "both"), default=None)
    parser.add_argument("--poisson-flux-postprocess-every", type=int, default=None)
    parser.add_argument(
        "--poisson-flux-postprocess-space",
        choices=("l2_closest", "RT_projection"),
        default=None,
    )
    parser.add_argument(
        "--poisson-postprocessing-backend", choices=("auto", "numba", "cupy", "raw-cuda"), default=None
    )

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
    parser.add_argument(
        "--transport-amgx-tolerance",
        type=float,
        default=None,
        help=(
            "primary AMGX stopping tolerance; physical acceptance still uses "
            "--transport-solver-rtol/atol"
        ),
    )
    parser.add_argument("--transport-ilu-drop-tol", type=float, default=None)
    parser.add_argument("--transport-ilu-fill-factor", type=float, default=None)
    parser.add_argument("--transport-boundary-mode", choices=("auto", "eliminate", "zero-flux", "penalty"), default=None)
    parser.add_argument("--transport-trace-ordering", choices=("none", "upwind-scc"), default=None)
    parser.add_argument("--transport-trace-ordering-flux-tolerance", type=float, default=None)
    parser.add_argument("--transport-ilu-permc-spec", choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"), default=None)
    parser.add_argument("--transport-raw-local-assembly", choices=("precomputed", "fused", "split3"), default=None)
    parser.add_argument("--transport-raw-lu-mode", choices=("safe", "coop"), default=None)
    parser.add_argument("--transport-raw-block-size", choices=("auto", "1", "32", "64", "128"), default=None)
    parser.add_argument("--transport-raw-matrix-format", choices=("auto", "coo", "csr", "bsr"), default=None)
    parser.add_argument("--transport-initial-guess", choices=("solver-default", "initial-density-trace"), default=None)
    parser.add_argument(
        "--transport-reuse-first-preconditioner",
        action="store_true",
        default=None,
        help="reuse the first transport preconditioner on later matrices; requires --transport-trace-ordering none",
    )
    parser.add_argument("--transport-retry-policy", choices=("none", "amgx-robust"), default=None)
    parser.add_argument("--transport-retry-amgx-config", type=Path, default=None)
    parser.add_argument("--transport-materialize-host-system", action="store_true")
    parser.add_argument("--no-transport-materialize-host-system", action="store_true")
    parser.add_argument("--transport-materialize-host-solution", choices=("auto", "on", "off"), default=None)


def _run_cli_case_with_terminal_log(
        config: GuidingCenterRunPreset,
        *,
        preset_key: str,
) -> GuidingCenterRunResult:
    """Run one CLI case while capturing Python and native terminal output."""
    log_path = _terminal_log_path(config, preset_key)
    result: GuidingCenterRunResult | None = None
    failure: BaseException | None = None
    with _TerminalLogTee(log_path):
        if _verbosity_level(config) >= 1:
            print(f"[gc] full terminal log: {log_path}", flush=True)
        try:
            result = run_guiding_center_case(
                config,
                preset_key=preset_key,
                terminal_log_path=log_path,
            )
        except BaseException as error:
            traceback.print_exc()
            failure = error
    if failure is not None:
        if isinstance(failure, SystemExit):
            raise failure
        if isinstance(failure, KeyboardInterrupt):
            raise SystemExit(130) from None
        raise SystemExit(1) from None
    if result is None:
        raise RuntimeError("guiding-center run completed without a result")
    return result


def _main() -> None:
    parser = GuidingCenterArgumentParser(
        description="Run fixed-mesh semi-implicit Euler or predictor-corrector guiding-center cases.",
        formatter_class=RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
        epilog=(
            "Curated preset configuration lives in scripts/guiding_center/guiding_center_presets.py.\n"
            "Cases are registered in scripts/guiding_center/guiding_center_cases.py.\n"
            "Long commands can be stored in response files and passed as @path/to/args."
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
    parser.add_argument("--minimum-triangles", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--gmsh-verbosity", type=int, default=None)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--basis", choices=("dub_orth", "hier_C0", "bernstein"), default=None)
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default=None)
    parser.add_argument("--poisson-trace-basis", choices=("legacy-lagrange", "legendre-modal"), default=None)
    parser.add_argument("--transport-trace-basis", choices=("legacy-lagrange", "legendre-modal"), default=None)
    parser.add_argument("--order", "-p", type=int, default=None)
    parser.add_argument("--volume-quadrature", choices=("auto", "symmetric", "duffy"), default=None)
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--time-scheme", choices=("si-euler", "predictor-corrector"), default=None)
    parser.add_argument("--poisson-tau", type=float, default=None)
    _add_solver_arguments(parser)
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        choices=(0, 1, 2, 3),
        default=None,
        help="logging level: 0 quiet, 1 per-step summaries, 2 solver phase logs, 3 detailed backend timings plus compact native/AMGX iteration tables",
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
    parser.add_argument("--diagnostics-every", type=int, default=None, help="materialize and record diagnostics every N accepted steps")
    args = parser.parse_args()

    if args.list_presets:
        print_presets()
        return

    preset_key = args.preset or args.preset_name or DEFAULT_PRESET
    config = _runtime_config(preset_by_key(preset_key), args)
    if args.print_preset or args.dry_run:
        print_preset_details(preset_key, config)
        return
    _run_cli_case_with_terminal_log(config, preset_key=preset_key)


if __name__ == "__main__":
    _main()
