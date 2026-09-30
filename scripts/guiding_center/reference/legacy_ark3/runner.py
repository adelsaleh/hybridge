#!/usr/bin/env python3
"""Run fixed-mesh guiding-center cases."""

from __future__ import annotations

import ast
import copy
import csv
import json
import shlex
import math
import os
import sys
import subprocess
import time
import traceback
from argparse import ArgumentParser, BooleanOptionalAction, RawDescriptionHelpFormatter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

if __name__ == "__main__":
    from scripts.guiding_center.runtime.precision_runtime import configure_precision_cli
    configure_precision_cli()

import numpy as np
from hdgfem.runtime.precision import (
    REAL_DTYPE,
    PRECISION,
    AMGX_MODE,
    KERNEL_AUDIT,
    PIPELINE_AUDIT,
    audit_arrays,
)

from hdgfem.core.field_ops import (
    perpendicular_vector_field,
    project_callable_to_trace,
    solution_field,
    solution_trace,
    trace_linear_combination,
)
from hdgfem.diagnostics.errors import evaluate_scalar_error
from hdgfem.diagnostics.guiding_center import (
    guiding_center_field_diagnostics,
    transport_velocity_diagnostics,
)
from hdgfem.diagnostics.solver import (
    relative_drift,
    result_transfer_time,
    solver_result_metrics,
)
from hdgfem.linalg.amgx.config import load_amgx_config
from scripts.guiding_center.cases.guiding_center_cases import CASE_DEFINITIONS
from scripts.guiding_center.reference.legacy_ark3.presets import (
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
    # Interior-edge coefficients in the configured trace bases, retaining
    # host/device residency. For PC this is the accepted extrapolated trace.
    final_density_trace_reduced: Any = None
    final_potential_trace_reduced: Any = None


@dataclass(frozen=True)
class GuidingCenterStepSnapshot:
    """Accepted endpoint plus the last transport solve exposed to observers."""

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
    accepted_density_trace_reduced: Any = None
    accepted_density_boundary: Any = None


class GuidingCenterArgumentParser(ArgumentParser):
    """Argument parser that supports shell-like ``@file`` response files."""

    def convert_arg_line_to_args(self, arg_line: str):
        stripped = arg_line.strip()
        if not stripped or stripped.startswith("#"):
            return []
        return shlex.split(stripped, comments=True)


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
    """Mirror stdout/stderr using a process independent of the solver's GIL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._log_fd: int | None = None
        self._saved_fds: dict[int, int] = {}
        self._pump_process: subprocess.Popen | None = None

    def __enter__(self):
        _flush_terminal_streams()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fd = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644,
        )
        pipe_fds = []
        write_ends = {}
        pump_args = [str(self._log_fd)]
        inherited = [self._log_fd]
        try:
            for target_fd in (1, 2):
                saved_fd = os.dup(target_fd)
                self._saved_fds[target_fd] = saved_fd
                read_fd, write_fd = os.pipe()
                pipe_fds.extend((read_fd, write_fd))
                write_ends[target_fd] = write_fd
                inherited.extend((read_fd, saved_fd))
                pump_args.extend((str(read_fd), str(saved_fd)))
            # A Python thread cannot drain native output while PyAMGX holds
            # the GIL. Exec a tiny independent process before redirecting FDs.
            self._pump_process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve().parents[2] / "runtime" / "_terminal_log_pump.py"), *pump_args],
                pass_fds=tuple(inherited),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._saved_fds[2],
                start_new_session=True,
            )
            for target_fd, write_fd in write_ends.items():
                os.dup2(write_fd, target_fd)
        except BaseException:
            self._restore_descriptors()
            # Close every writer before waiting for the pump to observe EOF.
            for fd in pipe_fds:
                os.close(fd)
            self._wait_and_close()
            raise
        for fd in pipe_fds:
            os.close(fd)
        return self

    def _restore_descriptors(self) -> None:
        for target_fd, saved_fd in self._saved_fds.items():
            try:
                os.dup2(saved_fd, target_fd)
            except OSError:
                pass

    def _wait_and_close(self) -> int:
        try:
            return 0 if self._pump_process is None else self._pump_process.wait()
        finally:
            for saved_fd in self._saved_fds.values():
                try:
                    os.close(saved_fd)
                except OSError:
                    pass
            self._saved_fds.clear()
            if self._log_fd is not None:
                os.close(self._log_fd)
                self._log_fd = None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        _flush_terminal_streams()
        self._restore_descriptors()
        returncode = self._wait_and_close()
        if returncode and exc_type is None:
            raise RuntimeError(f"terminal log pump failed for {self.path} (exit {returncode})")
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
    """Case labels and color policies for the shared live DG viewer."""

    def __init__(
        self, density_field, potential_field, *, resolution, title, show_mesh,
        off_screen, screenshot_dir, screenshot_prefix, include_potential=False,
        density_is_vorticity=False,
    ):
        from hdgfem.io import PyVistaFieldPanels

        self.include_potential = bool(include_potential)
        density_options = {"scalar_name": "density"}
        if density_is_vorticity:
            density_options.update(
                scalar_name="vorticity", cmap="RdBu_r", symmetric_clim=True,
                fixed_clim=True, robust_percentile=100.0,
            )
        panels = [("Vorticity" if density_is_vorticity else "Density", density_field, density_options)]
        if self.include_potential:
            panels.append(("Potential", potential_field, {"scalar_name": "potential"}))
        base_size = (1500, 650) if self.include_potential else (820, 720)
        self.viewer = PyVistaFieldPanels(
            panels, resolution=max(2, int(resolution)), title=title, show_mesh=show_mesh,
            off_screen=off_screen, window_size=tuple(int(round(2.5 * n)) for n in base_size),
            screenshot_dir=screenshot_dir, screenshot_prefix=screenshot_prefix,
        )

    def update(self, density_field, potential_field, *, step: int, time_value: float) -> None:
        fields = [density_field] + ([potential_field] if self.include_potential else [])
        self.viewer.update(fields, step=step, time_value=time_value)

    def close(self) -> None:
        self.viewer.close()



def _make_plotter(
        config, density_field, potential_field, *, title, off_screen,
        screenshot_dir, screenshot_prefix, density_is_vorticity: bool,
):
    """Construct only the selected optional visualization backend."""
    options = dict(
        title=title, show_mesh=config.plot_show_mesh, off_screen=off_screen,
        screenshot_dir=screenshot_dir, screenshot_prefix=screenshot_prefix,
        include_potential=config.plot_potential,
        density_is_vorticity=density_is_vorticity,
    )
    if config.plot_backend == "holoviz":
        from hdgfem.io.holoviz import GuidingCenterHolovizPanels
        return GuidingCenterHolovizPanels(
            density_field, potential_field, width=config.plot_width, height=config.plot_height,
            max_fps=config.plot_max_fps, **options,
        )
    return GuidingCenterPyVistaPanels(
        density_field, potential_field, resolution=config.plot_resolution, **options,
    )


def _plot_output_settings(config, output_stem):
    """Preserve PyVista's legacy headless saves; Holoviz saves only explicitly."""
    display = bool(os.environ.get("DISPLAY")) or (
        config.plot_backend == "holoviz" and bool(os.environ.get("WAYLAND_DISPLAY"))
    )
    headless = config.plot_every > 0 and not display
    directory = config.screenshot_dir
    if headless and directory is None and config.plot_backend == "pyvista":
        directory = str(Path(config.diagnostics_dir) / f"{output_stem}_frames")
    return headless, bool(config.plot_off_screen or headless), directory


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
        stage_count = int(row.get(f"{prefix}_stage_count", 1))
        if stage_count != 1:
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

    scheme_key = str(row.get("time_scheme")).replace("-", "_")
    if row.get("time_scheme") in {"h1-bdf3", "h2-bdf3"}:
        mode = (str(row.get(f"{scheme_key}_startup_method")) + " startup") if row.get(f"{scheme_key}_startup") else "BDF3"
        lines.append(
            f"  {row['time_scheme']:<9} mode={mode} | residuals={row.get('explicit_residual_count', 0)}"
            f" | residual={_format_metric(row.get('explicit_residual_time'), '.4f')}s"
            f" | predictor={_format_metric(row.get('density_predictor_time'), '.4f')}s"
            f" | trace projection={_format_metric(row.get('transport_trace_projection_time'), '.4f')}s"
        )

    if row.get("time_scheme") == "imex-ark3":
        lines.append(
            "  imex-ark3 embedded relative error="
            f"{_format_metric(row.get('imex_ark3_embedded_error_relative'))}"
            f" | operator assemblies/reuses={row.get('imex_ark3_operator_assemblies', 0)}/{row.get('imex_ark3_operator_reuses', 0)}"
            f" | residual={_format_metric(row.get('explicit_residual_time'), '.4f')}s"
            f" | Poisson tau={_format_metric(row.get('poisson_tau'))}"
            f" | tau retries={row.get('poisson_tau_retry_count', 0)}"
        )

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
        (f"circulation_drift={_format_metric(row['circulation_drift'])}"
         if "circulation_drift" in row else
         f"mass_rel_drift={_format_metric(row.get('mass_relative_drift'))}"),
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
    poisson_time = _first_metric(row, "poisson_time", "poisson_time_total")
    if poisson_time is not None:
        pieces.append(f"poisson={_format_metric(poisson_time, '.3f')}s")
    transport_time = _first_metric(row, "transport_time", "transport_time_total")
    if transport_time is not None and phase != "initial":
        pieces.append(f"transport={_format_metric(transport_time, '.3f')}s")
    scheme_key = str(row.get("time_scheme")).replace("-", "_")
    if row.get("time_scheme") in {"h1-bdf3", "h2-bdf3"} and phase != "initial":
        if row.get(f"{scheme_key}_startup"):
            pieces.append(f"startup={row.get(f'{scheme_key}_startup_method')}")
        pieces.append(f"rhs={_format_metric(row.get('explicit_residual_time'), '.3f')}s")
    if row.get("time_scheme") == "imex-ark3" and phase != "initial":
        pieces.append(f"embedded_rel={_format_metric(row.get('imex_ark3_embedded_error_relative'))}")
        pieces.append(f"poisson_tau={_format_metric(row.get('poisson_tau'))}")
        pieces.append(f"tau_retries={row.get('poisson_tau_retry_count', 0)}")
        pieces.append(f"rhs={_format_metric(row.get('explicit_residual_time'), '.3f')}s")
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
    lines.extend([
        f"  enstrophy (1/2 ||rho||²)              {_format_metric(row.get('enstrophy'), '.10e')}",
        f"  relative enstrophy drift              {_format_metric(row.get('enstrophy_relative_drift'))}",
    ])
    if row.get("rho_min_checked") is not None:
        lines.extend(["", "Positivity (no limiter)",
            f"  initial/endpoint status               {row.get('positivity_status')}",
            f"  checked minimum / cell-average min    {_format_metric(row.get('rho_min_checked'))} / {_format_metric(row.get('rho_cell_average_min'))}",
            f"  negative mass (volume quadrature)     {_format_metric(row.get('rho_negative_mass_quadrature'))}",
            f"  all-stage minimum this step           {_format_metric(row.get('positivity_stage_min'))}"])
    if row.get("diocotron_phi_mode_target_l2") is not None:
        lines.extend(["", "Polar Fourier potential diagnostics",
            f"  target mode amplitude L2              {_format_metric(row.get('diocotron_phi_mode_target_l2'))}",
            f"  axisymmetric departure L2             {_format_metric(row.get('diocotron_phi_axisymmetric_l2'))}",
            f"  potential harmonic ratio (2k/k)       {_format_metric(row.get('diocotron_phi_harmonic_ratio'))}"])
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

    scheme_key = str(row.get("time_scheme")).replace("-", "_")
    if row.get("time_scheme") in {"h1-bdf3", "h2-bdf3"} and phase != "initial":
        mode = (str(row.get(f"{scheme_key}_startup_method")) + " startup") if row.get(f"{scheme_key}_startup") else "BDF3"
        lines.extend([
            "", f"{row['time_scheme'].upper()} stages ({mode})",
            f"  transport / Poisson / residual       {row.get('transport_stage_count', 0)} / "
            f"{row.get('poisson_stage_count', 0)} / {row.get('explicit_residual_count', 0)}",
        ])
    if row.get("time_scheme") == "imex-ark3" and phase != "initial":
        lines.extend([
            "", "IMEX-ARK3 stages (ARK3(2)4L[2]SA)",
            f"  transport / Poisson / residual       {row.get('transport_stage_count', 0)} / "
            f"{row.get('poisson_stage_count', 0)} / {row.get('explicit_residual_count', 0)}",
            f"  embedded second-order difference    L2={_format_metric(row.get('imex_ark3_embedded_error_l2'))} "
            f"relative={_format_metric(row.get('imex_ark3_embedded_error_relative'))}",
            f"  transport assemblies / reuses       {row.get('imex_ark3_operator_assemblies', 0)} / "
            f"{row.get('imex_ark3_operator_reuses', 0)}",
            f"  Poisson tau / recovery retries      {_format_metric(row.get('poisson_tau'))} / "
            f"{row.get('poisson_tau_retry_count', 0)}",
            f"  rejected transport / Poisson stages {row.get('transport_rejected_stage_count', 0)} / "
            f"{row.get('poisson_rejected_stage_count', 0)}",
        ])
    lines.extend(["", "Phase timings"])
    timing_rows = [
        ("complete coupled linear step", row.get("linear_step_wall_time") if phase != "initial" else None),
        ("complete transport stage wall", row.get("transport_step_wall_time") if phase != "initial" else None),
        ("complete Poisson stage wall", row.get("poisson_step_wall_time") if phase != "initial" else None),
        ("beta construction", row.get("beta_build_time") if phase != "initial" else None),
        ("transport HDG solve", _first_metric(row, "transport_time", "transport_time_total") if phase != "initial" else None),
        ("Poisson HDG solve", _first_metric(row, "poisson_time", "poisson_time_total")),
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
        ("time scheme state and cache priming", row.get("time_scheme_setup_time", row.get("hybrid_bdf3_setup_time"))),
        ("explicit HDG residuals", row.get("explicit_residual_time")),
        ("density prediction / extrapolation", row.get("density_predictor_time")),
        ("transport guess trace projection", row.get("transport_trace_projection_time")),
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


def _with_fp32_transport_solver(config: GuidingCenterRunPreset) -> GuidingCenterRunPreset:
    """Use FGMRES for the stock FP32 transport profile that BiCGSTAB destabilizes."""
    if PRECISION != "float32" or not _is_amgx_solver(config.transport_solver):
        return config
    config_dir = Path(__file__).resolve().parents[4] / "configs" / "amgx"
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
        "transport_direct_fallback": args.transport_direct_fallback,
        "transport_retry_amgx_config_path": None if args.transport_retry_amgx_config is None else str(args.transport_retry_amgx_config),
        "verbosity": args.verbosity,
        "plot_every": args.plot_every,
        "plot_backend": args.plot_backend,
        "plot_width": args.plot_width,
        "plot_height": args.plot_height,
        "plot_max_fps": args.plot_max_fps,
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
    return runtime


def _build_mesh(config: GuidingCenterRunPreset, case):
    from hdgfem.core.mesh import (
        gmsh_disc_mesh, gmsh_rectangle_mesh, gmsh_smooth_star_mesh,
        gmsh_triangle_mesh, rectangle_mesh,
    )

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


def _hybrid_startup_method(config: GuidingCenterRunPreset) -> str:
    """Select the startup configuration of the requested hybrid."""
    return config.h1_startup if config.time_scheme == "h1-bdf3" else config.h2_startup


def _validate_config(config: GuidingCenterRunPreset) -> None:
    if not math.isfinite(config.positivity_tolerance) or config.positivity_tolerance < 0:
        raise ValueError("positivity_tolerance must be finite and nonnegative")
    if config.diocotron_radial_points < 2:
        raise ValueError("diocotron_radial_points must be at least 2")
    if config.diocotron_angular_points is not None and config.diocotron_angular_points < 2:
        raise ValueError("diocotron_angular_points must be at least 2")
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
    if config.time_scheme not in {"si-euler", "predictor-corrector", "si-bdf2", "h1-bdf3", "h2-bdf3", "imex-ark3"}:
        raise ValueError("time_scheme must be 'si-euler', 'predictor-corrector', 'si-bdf2', 'h1-bdf3', 'h2-bdf3', or 'imex-ark3'")
    if config.time_scheme in {"h1-bdf3", "h2-bdf3", "imex-ark3"}:
        if config.time_scheme != "imex-ark3" and _hybrid_startup_method(config) not in {"si-euler-extrap3", "ssprk3"}:
            raise ValueError("hybrid startup must be si-euler-extrap3 or ssprk3")
        if config.transport_advection_stabilization is not None:
            raise ValueError(f"{config.time_scheme} requires the standard upwind stabilization")
        if config.transport_boundary_mode not in {"auto", "zero-flux", "eliminate"}:
            raise ValueError(f"{config.time_scheme} requires zero-flux or eliminated transport boundaries")
    if config.time_scheme == "imex-ark3":
        from scripts.guiding_center.reference.legacy_ark3.poisson_recovery import PoissonTauRecovery
        PoissonTauRecovery(factor=config.poisson_tau_retry_factor, max_retries=config.poisson_tau_max_retries)
        if not math.isfinite(config.poisson_tau) or config.poisson_tau <= 0:
            raise ValueError("IMEX-ARK3 requires finite positive poisson_tau")
        if config.transport_assembly_backend not in {"numpy", "raw-cuda"}:
            raise ValueError("IMEX-ARK3 operator reuse currently requires numpy or raw-cuda transport")
        if config.transport_assembly_backend == "raw-cuda" and config.transport_raw_local_assembly != "fused":
            raise ValueError("IMEX-ARK3 raw-cuda transport requires fused local assembly")
        if config.transport_trace_ordering != "none" or config.transport_reuse_first_preconditioner:
            raise ValueError("IMEX-ARK3 manages per-step operator reuse; use no trace ordering or first-preconditioner override")
    if config.diagnostics_every < 1:
        raise ValueError("diagnostics_every must be positive")
    if config.poisson_flux_postprocess_every < 0:
        raise ValueError("poisson_flux_postprocess_every must be nonnegative")
    if config.transport_retry_policy not in {"none", "amgx-robust"}:
        raise ValueError("transport_retry_policy must be 'none' or 'amgx-robust'")
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
    midpoint_flux = 0.5 * (left_flux + right_flux)
    midpoint_flux.name = "q_mid_h"
    return perpendicular_vector_field(midpoint_flux, 0.5 * float(dt), space, name="beta_h")


def _bdf2_transport_data(space, density, flux, dt, *, previous_density=None, previous_flux=None):
    """Return source and beta for constant-step BDF2, with SI-Euler startup.

    (I + 2*dt/3 A(2*v_n - v_previous)) rho_next
        = (4*rho_n - rho_previous)/3.

    Both history fields must come from the same previous accepted endpoint.
    All combinations preserve device residency and leave their inputs intact.
    """
    if (previous_density is None) != (previous_flux is None):
        raise ValueError("BDF2 requires both previous density and previous flux, or neither")
    if previous_density is None:
        return density, perpendicular_vector_field(flux, dt, space), float(dt)
    source = (4.0 * density - previous_density) / 3.0
    source.name = "rho_bdf2_source_h"
    extrapolated_flux = 2.0 * flux - previous_flux
    extrapolated_flux.name = "q_bdf2_extrapolated_h"
    beta_scale = 2.0 * float(dt) / 3.0
    return source, perpendicular_vector_field(extrapolated_flux, beta_scale, space), beta_scale


def _compute_diagnostics(
        *,
        case,
        rho_field,
        poisson_result,
        step: int,
        time_value: float,
        baseline_mass: float | None,
        baseline_q_l2: float | None,
        baseline_enstrophy: float | None = None,
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
    # The stage builder uses the standard HDG flux, even when a postprocessed
    # flux is present. Measure the same velocity, without timestep scaling.
    velocity_metrics = transport_velocity_diagnostics(
        perpendicular_vector_field(poisson_result.flux, 1.0, rho_field.space),
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
        "mass_relative_drift": None if case.density_is_vorticity else relative_drift(mass, effective_baseline_mass),
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
    row.update(velocity_metrics)
    enstrophy = 0.5 * float(field_metrics["rho_l2_squared"])
    z0 = enstrophy if baseline_enstrophy is None else float(baseline_enstrophy)
    row.update(enstrophy=enstrophy, enstrophy_drift=enstrophy-z0,
               enstrophy_relative_drift=relative_drift(enstrophy, z0))
    if case.density_is_vorticity:
        row["circulation"] = mass
        row["circulation_drift"] = mass - effective_baseline_mass
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


def _solve_transport_stage(
        solver, *, initial_guess, beta, step, time_value, stage, beta_scale,
        failure_path: Path,
):
    """Preserve diagnostics of the actual failed stage, then re-raise its error."""
    from hdgfem.linalg.results import LinearSolveConvergenceError

    try:
        return solver.solve(initial_guess=initial_guess)
    except LinearSolveConvergenceError as error:
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


def _transport_amgx_divergence_config(config: dict | None, *, tolerance: float) -> dict:
    """Enable native divergence exits for transport, preserving explicit overrides."""
    if config is None:
        from hdgfem.linalg.amgx.host import default_pyamgx_config
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
                Path(__file__).resolve().parents[4]
                / "configs"
                / "amgx"
                / "adv_rea_gpu4_hdg_fgmres_dilu_abs.json"
            )
        fallback_config = _load_amgx_config(
            fallback_path,
            tolerance=config.transport_solver_rtol,
            absolute_tolerance=config.transport_solver_atol,
        )
        amgx_config_dir = Path(__file__).resolve().parents[4] / "configs" / "amgx"
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


def _print_run_summary(result: GuidingCenterRunResult) -> None:
    if _verbosity_level(result.config) < 1:
        return
    from hdgfem.io.output import pretty_print_sections

    final = result.diagnostics[-1]
    run_rows = [
        ("preset", result.preset_key, "s"),
        ("case", result.case_key, "s"),
        ("time scheme", result.config.time_scheme, "s"),
        ("precision", PRECISION, "s"),
        ("DG order", result.space.order, ",d"),
        ("triangles", result.mesh.num_tri, ",d"),
        ("steps", result.config.num_steps, ",d"),
        ("dt", result.config.dt, ".4e"),
    ]
    if result.config.time_scheme in {"h1-bdf3", "h2-bdf3"}:
        run_rows.append(("startup", _hybrid_startup_method(result.config), "s"))
    radial_power = result.config.case_params.get("p")
    if radial_power is not None:
        run_rows.insert(3, ("radial p", radial_power, ".4g"))
    final_rows = [
        ("time", final["time"], ".4e"),
        (("circulation drift", final["circulation_drift"], ".4e") if "circulation_drift" in final else
         ("relative mass drift", final["mass_relative_drift"], ".4e")),
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



def _initial_projection_backend(config: GuidingCenterRunPreset) -> str:
    """Match initial field projection to the configured assembly backends."""
    device_backends = {"cupy", "raw-cuda"}
    return "cupy" if (
        config.poisson_assembly_backend in device_backends
        or config.transport_assembly_backend in device_backends
    ) else "numpy"


def _project_initial_field(config: GuidingCenterRunPreset, space, function, *, name: str):
    """Project on the selected backend, keeping device coefficients resident."""
    from hdgfem.core.projection import project_callable

    return project_callable(function, space,
        backend="device" if _initial_projection_backend(config) == "cupy" else "host",
        volume_quad_1d=config.initial_projection_quad_1d, name=name, synchronize=True)


def run_guiding_center_case(
        config: GuidingCenterRunPreset,
        *,
        preset_key: str = "custom",
        step_observer: Callable[[GuidingCenterStepSnapshot], None] | None = None,
        terminal_log_path: str | Path | None = None,
) -> GuidingCenterRunResult:
    """Run a fixed-mesh guiding-center case with the selected time scheme."""
    _validate_config(config)

    from hdgfem.core.space import DGSpace
    from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver
    from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGSolver
    from hdgfem.runtime.logging import timed_call
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
    if config.diocotron_diagnostics and (case.key != "diocotron_k" or
            (config.domain != "auto" and config.domain != "disc")):
        raise ValueError("diocotron modal diagnostics require diocotron_k on the disk")
    projection_backend = _initial_projection_backend(config)
    rho_field, initial_density_projection_time = timed_call(
        f"[gc:init] projecting initial density ({projection_backend})",
        _detail_verbosity(config),
        lambda: _project_initial_field(config, space, case.initial_density_at(), name="rho_h"),
    )
    positivity = None
    initial_positivity = {}
    if config.positivity_diagnostics:
        from hdgfem.diagnostics.guiding_center import ScalarPositivityDiagnostics
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
    equilibrium_density_projection_time = 0.0
    if case.equilibrium_density is not None:
        equilibrium_density, equilibrium_density_projection_time = timed_call(
            f"[gc:init] projecting equilibrium density ({projection_backend})",
            _detail_verbosity(config),
            lambda: _project_initial_field(config, space, case.equilibrium_density, name="rho_eq_h"),
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
        # Retain the Poisson operator while its coefficients and tau are fixed.
        # Retain the equilibrium solver unconditionally so the perturbed initial
        # state and every accepted step reuse its trace operator, local factors,
        # global factorization/preconditioner, and AMGX hierarchy.
        poisson_solver = equilibrium_solver
        poisson_initial_guess = solution_trace(equilibrium_result, space, reduced=False)
        # ARK can change tau on rank loss. Keep its configured preconditioner
        # policy so a rebuild cannot retain this equilibrium matrix's factors.
        if config.poisson_preconditioner is not None and config.time_scheme != "imex-ark3":
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
    previous_density = None
    previous_flux = None

    stage_stepper = None
    if config.time_scheme in {"h1-bdf3", "h2-bdf3", "imex-ark3"}:
        from hdgfem.transport.residual import (
                    HDGTraceWorkspace,
                    UpwindHDGTransportResidual,
                )
        from scripts.guiding_center.time_schemes.h1_bdf3 import H1BDF3Stepper
        from scripts.guiding_center.time_schemes.h2_bdf3 import H2BDF3Stepper
        from scripts.guiding_center.reference.legacy_ark3.imex_ark3 import IMEXARK3Stepper

        recovery_log_started = False

        def record_poisson_recovery(event):
            nonlocal recovery_log_started
            # Record events immediately, even if the attempt never completes.
            stem = config.diagnostics_prefix or preset_key
            path = Path(config.diagnostics_dir) / f"{stem}_poisson_tau_recovery.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a" if recovery_log_started else "w") as stream:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            recovery_log_started = True

        def make_stage_stepper():
            startup_method = _hybrid_startup_method(config)
            needs_residual = config.time_scheme in {"h1-bdf3", "imex-ark3"} or startup_method == "ssprk3"
            workspace_type = UpwindHDGTransportResidual if needs_residual else HDGTraceWorkspace
            workspace = workspace_type(
                space, trace_basis=_transport_trace_basis(config),
                backend="device" if config.transport_assembly_backend in {"cupy", "raw-cuda"} else "host",
                **({"boundary_mode": transport_boundary_mode} if needs_residual else {}),
            )
            stepper_type = {"h1-bdf3": H1BDF3Stepper, "h2-bdf3": H2BDF3Stepper, "imex-ark3": IMEXARK3Stepper}[config.time_scheme]
            return stepper_type(
                space, config.dt, rho_field, poisson_result, workspace,
                density_boundary=(lambda t: None) if transport_boundary_mode == "zero-flux" else case.density_boundary_at,
                potential_boundary=case.potential_boundary_at,
                phase_verbosity=_phase_verbosity(config), detail_verbosity=_detail_verbosity(config),
                **(dict(poisson_solver=poisson_solver,
                        poisson_tau_retry_factor=config.poisson_tau_retry_factor,
                        poisson_tau_max_retries=config.poisson_tau_max_retries,
                        recovery_verbosity=_verbosity_level(config), recovery_record=record_poisson_recovery,
                        density_diagnostics=None if positivity is None else positivity.measure)
                   if config.time_scheme == "imex-ark3" else {"startup_method": startup_method}),
            )

        stage_stepper, hybrid_setup_time = timed_call(
            f"[gc:init] priming {config.time_scheme.upper()} state and trace caches",
            _detail_verbosity(config), make_stage_stepper,
        )
        stage_stepper.setup_time = hybrid_setup_time
        if config.time_scheme == "imex-ark3":
            poisson_result = stage_stepper.initial_poisson_result
            initialization_poisson_wall_time += stage_stepper.initial_poisson_retry_wall_time
        density_trace = stage_stepper.density_trace
        potential_trace = stage_stepper.potential_trace

    modal_diagnostics = None
    if config.diocotron_diagnostics:
        from scripts.guiding_center.diagnostics.diocotron_diagnostics import DiocotronModeDiagnostics
        modal_diagnostics, _ = timed_call("[gc:init] caching polar Fourier diagnostics", _detail_verbosity(config),
            lambda: DiocotronModeDiagnostics(space, equilibrium_potential,
                mode=int(case.parameters["k"]), inner=float(case.parameters["s_minus"]),
                outer=float(case.parameters["s_plus"]), radial_points=config.diocotron_radial_points,
                angular_points=config.diocotron_angular_points,
                backend="device" if projection_backend == "cupy" else "host"))

    def benchmark_metrics():
        metrics = {}
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
    recorder = DiagnosticsRecorder(config.diagnostics_dir, output_stem)
    timing_recorder = DiagnosticsRecorder(config.diagnostics_dir, f"{output_stem}_timings")
    plotter = None
    try:
        initial_extra = solver_result_metrics("poisson", poisson_result)
        initial_extra.update(initial_positivity)
        initial_extra.update(benchmark_metrics())
        initial_extra["run_configuration"] = dict(case=case.key, case_parameters=case.parameters,
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
        if stage_stepper is not None:
            if config.time_scheme == "imex-ark3":
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
                "initial_poisson_retry_wall_time": (stage_stepper.initial_poisson_retry_wall_time
                    if config.time_scheme == "imex-ark3" else 0.),
                "equilibrium_poisson_wall_time": (
                    first_poisson_wall_time
                    if equilibrium_density is not None
                    else 0.0
                ),
                "poisson_flux_postprocessed": False,
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
        baseline_enstrophy = float(row["enstrophy"])
        equilibrium_potential_l2 = row.get("diocotron_phi_eq_reference_l2")
        equilibrium_density_l2 = row.get("diocotron_rho_eq_reference_l2")
        if config.plot_every > 0:
            plot_start = time.perf_counter()
            plotter = _make_plotter(
                config,
                rho_field,
                poisson_result.field,
                title=f"{preset_key}: {case.key}",
                off_screen=effective_plot_off_screen,
                screenshot_dir=effective_screenshot_dir,
                screenshot_prefix=config.diagnostics_prefix or preset_key,
                density_is_vorticity=case.density_is_vorticity,
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
            if stage_stepper is None:
                step_poisson_initial_guess, poisson_predictor_order = _fixed_operator_trace_predictor(
                    potential_trace, previous_potential_trace, older_potential_trace,
                )
            else:
                step_poisson_initial_guess, poisson_predictor_order = potential_trace, 0
            potential_trace_time = 0.0
            post_poisson_start = None

            if stage_stepper is not None:
                def solve_stage_transport(source, beta, guess, scale, *, stage_time=None, stage=None, reuse_operator=False):
                    nonlocal transport_preconditioner_reused
                    stage_time = next_time if stage_time is None else stage_time
                    stage_boundary = (None if transport_boundary_mode == "zero-flux" else
                                      case.density_boundary_at(stage_time))
                    if reuse_operator:
                        transport_solver.set_source(source, boundary_condition=stage_boundary)
                    else:
                        transport_solver.set_problem(source, beta, one_reaction, stage_boundary)
                    result = _solve_transport_stage(
                        transport_solver, initial_guess=guess, beta=beta, step=step,
                        time_value=stage_time, stage=stage or config.time_scheme, beta_scale=scale,
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

                hybrid = stage_stepper.advance(poisson_solver, solve_stage_transport,
                                       endpoint_postprocess=_poisson_postprocess_overrides(config, step))
                rho_field, density_trace = hybrid.density, hybrid.density_trace
                poisson_result, potential_trace = hybrid.poisson_result, hybrid.potential_trace
                transport_result = hybrid.transport_result
                step_transport_source, predictor_beta = hybrid.transport_source, hybrid.transport_beta
                step_transport_initial_guess = hybrid.transport_initial_guess
                step_poisson_initial_guess = hybrid.poisson_initial_guess
                transport_stage_results, poisson_stage_results = hybrid.transport_results, hybrid.poisson_results
                transport_step_wall_time, poisson_step_wall_time = hybrid.transport_wall_time, hybrid.poisson_wall_time
                beta_build_time = hybrid.beta_build_time
                transport_time = sum(result.timings.total for result in transport_stage_results)
                poisson_time = sum(result.timings.total for result in poisson_stage_results)
                stage_extra = hybrid.metrics
                for prefix, results in (("poisson", poisson_stage_results), ("transport", transport_stage_results)):
                    for index, result in enumerate(results):
                        stage_extra.update(solver_result_metrics(f"stage{index+1}_{prefix}", result))
                linear_step_end = time.perf_counter()
                post_poisson_start = linear_step_end
            else:
                beta_start = time.perf_counter()
                transport_beta_scale = config.dt
                transport_stage = "predictor"
                bdf2_startup = config.time_scheme == "si-bdf2" and previous_density is None
                if config.time_scheme == "si-bdf2":
                    accepted_density = rho_field
                    accepted_flux = poisson_result.flux
                    step_transport_source, predictor_beta, transport_beta_scale = _bdf2_transport_data(
                        space, rho_field, poisson_result.flux, config.dt,
                        previous_density=previous_density, previous_flux=previous_flux,
                    )
                    transport_stage = "bdf2-startup" if bdf2_startup else "bdf2"
                else:
                    predictor_beta = perpendicular_vector_field(
                        poisson_result.flux, config.dt, space
                    )
                beta_build_time = time.perf_counter() - beta_start

                transport_stage_start = time.perf_counter()
                transport_solver.set_problem(
                    step_transport_source,
                    predictor_beta,
                    one_reaction,
                    endpoint_density_boundary,
                )
                predictor_transport_result = _solve_transport_stage(
                    transport_solver, initial_guess=density_trace, beta=predictor_beta,
                    step=step, time_value=next_time, stage=transport_stage, beta_scale=transport_beta_scale,
                    failure_path=Path(config.diagnostics_dir) / f"{output_stem}_transport_failure.json",
                )
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

                if config.time_scheme in {"si-euler", "si-bdf2"}:
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
                    if config.time_scheme == "si-bdf2":
                        # Commit history only after transport AND endpoint Poisson succeed.
                        previous_density = accepted_density
                        previous_flux = accepted_flux
                        stage_extra.update(
                            bdf2_startup=bdf2_startup,
                            transport_time_order=1 if bdf2_startup else 2,
                            transport_beta_scale=transport_beta_scale,
                        )
                    poisson_time = poisson_result.timings.total
                    transport_time = transport_result.timings.total
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
                    transport_result = _solve_transport_stage(
                        transport_solver, initial_guess=midpoint_trace_guess, beta=midpoint_beta,
                        step=step, time_value=current_time + 0.5 * config.dt, stage="corrector", beta_scale=0.5 * config.dt,
                        failure_path=Path(config.diagnostics_dir) / f"{output_stem}_transport_failure.json",
                    )
                    midpoint_density = solution_field(
                        transport_result, space, name="rho_midpoint_h",
                    )
                    midpoint_density_trace = solution_trace(
                        transport_result,
                        space,
                        reduced=True,
                    )
                    rho_field = 2.0 * midpoint_density - rho_field
                    rho_field.name = "rho_h"
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

            if step_observer is not None:
                is_pc = config.time_scheme == "predictor-corrector"
                step_observer(GuidingCenterStepSnapshot(
                    step=step, time=next_time, space=space,
                    transport_source=step_transport_source,
                    transport_beta=midpoint_beta if is_pc else predictor_beta,
                    transport_reaction=one_reaction,
                    transport_boundary=midpoint_boundary if is_pc else endpoint_density_boundary,
                    transport_initial_guess=midpoint_trace_guess if is_pc else step_transport_initial_guess,
                    transport_result=transport_result, accepted_density=rho_field,
                    poisson_boundary=endpoint_poisson_boundary,
                    poisson_initial_guess=predictor_potential_trace if is_pc else step_poisson_initial_guess,
                    poisson_result=poisson_result,
                    accepted_density_trace_reduced=density_trace,
                    accepted_density_boundary=endpoint_density_boundary,
                ))

            linear_step_wall_time = linear_step_end - linear_step_start
            stage_extra["poisson_predictor_order"] = poisson_predictor_order
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
            older_potential_trace = previous_potential_trace
            previous_potential_trace = accepted_potential_trace

            current_time = next_time
            should_plot = config.plot_every > 0 and step % config.plot_every == 0
            should_record = step % config.diagnostics_every == 0 or step == config.num_steps
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
                        title=f"{preset_key}: {case.key}",
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
        try:
            if plotter is not None:
                plotter.close()
        finally:
            try:
                recorder.close()
            finally:
                timing_recorder.close()

    if config.time_scheme == "imex-ark3":
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
        final_potential_trace_reduced=solution_trace(poisson_result, space, reduced=True),
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
    if PRECISION == "float32":
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
    parser.add_argument(
        "--transport-direct-fallback", choices=("none", "cusolver-qr"), default=None,
        help="optional device sparse QR after all amgx-robust attempts fail",
    )
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
        description="Run fixed-mesh guiding-center cases with SI Euler, predictor-corrector, SI BDF2, H1-BDF3, or H2-BDF3.",
        formatter_class=RawDescriptionHelpFormatter,
        fromfile_prefix_chars="@",
        epilog=(
            "Curated preset configuration lives in scripts/guiding_center/cases/guiding_center_presets.py.\n"
            "Cases are registered in scripts/guiding_center/cases/guiding_center_cases.py.\n"
            "Long commands can be stored in response files and passed as @path/to/args."
        ),
    )
    parser.add_argument("--precision", choices=("float32", "float64"), default=PRECISION, help="floating precision for the full numerical pipeline; FP32 defaults: FGMRES transport, Poisson rtol=2e-3, transport rtol=5e-3, AMGX tolerance=5e-3, atol=0; calibrated on a p=6 mesh with about 12k triangles")
    parser.add_argument("preset_name", nargs="?", default=None, choices=tuple(sorted(PRESETS)))
    parser.add_argument("--preset", dest="preset", choices=tuple(sorted(PRESETS)), default=None)
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--print-preset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend-profile", choices=("host", "device", "hybrid"), default=None)
    parser.add_argument("--case", choices=tuple(sorted(CASE_DEFINITIONS)), default=None)
    parser.add_argument("--case-param", action="append", default=None, help="override case parameter with key=value syntax")
    parser.add_argument("--domain", choices=("auto", "structured-rectangle", "rectangle", "disc", "triangle", "smooth-star"), default=None)
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
    parser.add_argument("--time-scheme", choices=("si-euler", "predictor-corrector", "si-bdf2", "h1-bdf3", "h2-bdf3", "imex-ark3"), default=None)
    parser.add_argument("--h2-startup", choices=("si-euler-extrap3", "ssprk3"), default=None,
                        help="H2 third-order initializer: SI-Euler extrapolation (default), or explicit SSPRK3")
    parser.add_argument("--h1-startup", choices=("si-euler-extrap3", "ssprk3"), default=None,
                        help="H1 third-order initializer: SI-Euler extrapolation (default), or explicit SSPRK3")
    parser.add_argument("--poisson-tau", type=float, default=None)
    parser.add_argument("--poisson-tau-retry-factor", type=float, default=None,
                        help="IMEX-ARK3: multiply Poisson tau on confirmed trace rank loss (default 2)")
    parser.add_argument("--poisson-tau-max-retries", type=int, default=None,
                        help="IMEX-ARK3: maximum tau increases per unaccepted step (default 4; 0 disables)")
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
    parser.add_argument("--plot", action="store_true", help="enable plotting every frame unless --plot-every is set")
    parser.add_argument("--plot-every", type=int, default=None, help="offer a plot update every N steps; 0 disables plotting")
    parser.add_argument("--plot-backend", choices=("pyvista", "holoviz"), default=None)
    parser.add_argument("--plot-width", type=int, default=None, help="Holoviz pixels per panel horizontally (default 1024)")
    parser.add_argument("--plot-height", type=int, default=None, help="Holoviz pixels per panel vertically (default 1024)")
    parser.add_argument("--plot-max-fps", type=float, default=None, help="Holoviz live preview rate cap (default 10); explicit screenshots retain every requested frame")
    parser.add_argument("--plot-resolution", type=int, default=None)
    parser.add_argument("--plot-both", action="store_true", help="plot density and potential; default plotting shows density only")
    parser.add_argument("--plot-off-screen", action="store_true")
    parser.add_argument("--no-plot-mesh", action="store_true")
    parser.add_argument("--screenshot-dir", type=Path, default=None)
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--diagnostics-prefix", default=None)
    parser.add_argument("--initial-projection-quad-1d", type=int,
                        help="Richer initial/equilibrium projection quadrature; evolution quadrature stays fixed.")
    parser.add_argument("--positivity-diagnostics", action=BooleanOptionalAction, default=None,
                        help="Measure polynomial bounds, negative mass, and every ARK stage; no limiter.")
    parser.add_argument("--positivity-tolerance", type=float)
    parser.add_argument("--diocotron-diagnostics", action=BooleanOptionalAction, default=None,
                        help="Cache polar Fourier potential diagnostics for a disk diocotron_k run.")
    parser.add_argument("--diocotron-radial-points", type=int,
                        help="Gauss points per radial segment (four segments by default).")
    parser.add_argument("--diocotron-angular-points", type=int,
                        help="Polar FFT points; must exceed six times the selected mode.")
    parser.add_argument("--diagnostics-every", type=int, default=None, help="materialize and record diagnostics every N accepted steps")
    args = parser.parse_args()

    if args.list_presets:
        print_presets()
        return

    preset_key = args.preset or args.preset_name or DEFAULT_PRESET
    config = _runtime_config(preset_by_key(preset_key), args)
    if config.time_scheme != "imex-ark3":
        parser.error("this archived comparison runner supports only --time-scheme imex-ark3")
    if args.print_preset or args.dry_run:
        print_preset_details(preset_key, config)
        return
    _run_cli_case_with_terminal_log(config, preset_key=preset_key)


if __name__ == "__main__":
    _main()
