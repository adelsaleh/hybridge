"""Guiding-center models helpers."""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset


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
    csv_path: Path | None
    jsonl_path: Path | None
    timings_csv_path: Path | None
    timings_jsonl_path: Path | None
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

