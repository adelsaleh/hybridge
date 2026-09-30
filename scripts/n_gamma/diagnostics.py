"""Errors, sampled extrema and study records for the n-Gamma D-BDF2 model.

Errors use the norm of the chosen geometry on the actual polygonal mesh
domain: the plain ``L2`` norm for ``"cartesian"`` and the weighted
``||e||_{L2_R} = (int |e|^2 R dR dZ)^{1/2}`` for ``"axisymmetric"``. Minima are *sampled* on the volume and
face quadrature points and are reported as samples, not certified bounds.
Per-step rows go through :class:`hdgfem.io.records.DiagnosticsRecorder`
(JSONL while running, CSV at the end); convergence summaries are JSON and
Markdown tables with error ratios and observed orders.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from hdgfem import DGField, evaluate_scalar_error, field_values_at_ref
from hdgfem.io.records import DiagnosticsRecorder, _json_safe


def radius_weight(R, Z):
    """Axisymmetric measure weight ``R`` (NumPy or CuPy)."""
    return R


def error_norm(field: DGField, exact: Callable, *, geometry: str, volume_degree: int | None = None,
               volume_quad_1d: int | None = None, backend: str = "auto") -> float:
    """Return ``||field - exact||`` in the geometry's norm; ``exact`` takes mesh coordinates at a fixed time."""
    from .coefficients import check_geometry

    weight = radius_weight if check_geometry(geometry) == "axisymmetric" else None
    report = evaluate_scalar_error(field, exact, weight=weight, volume_degree=volume_degree,
                                   volume_quad_1d=volume_quad_1d, backend=backend)
    return float(report.metrics.l2)


def error_norms(fields: dict[str, DGField], exact: dict[str, Callable], *, geometry: str,
                **options) -> dict[str, float]:
    """Return ``{"<name>_l2": ...}`` (Cartesian) or ``{"<name>_l2R": ...}`` (axisymmetric) per field."""
    suffix = "l2R" if geometry == "axisymmetric" else "l2"
    return {f"{name}_{suffix}": error_norm(field, exact[name], geometry=geometry, **options)
            for name, field in fields.items()}


def sampled_minimum(field: DGField, space, trace_space, *, device: bool = False) -> float:
    """Return the minimum of ``field`` sampled on volume and element-side face points."""
    from hdgfem.core.quadrature import _reference_edge_points_from_1d

    face = np.asarray(_reference_edge_points_from_1d(trace_space.quads)).reshape(-1, 2)
    volume = field_values_at_ref(field, space.quad_data.Krf_quads, device=device)
    faces = field_values_at_ref(field, face, device=device)
    return float(min(volume.min(), faces.min()))


def flatten(record: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dataclasses/dicts into ``a.b`` keys for CSV rows."""
    if is_dataclass(record) and not isinstance(record, type):
        record = asdict(record)
    if not isinstance(record, dict):
        return {prefix.rstrip("."): record}
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if is_dataclass(value) or isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def convergence_rows(parameter_name: str, parameters: Sequence[float],
                     errors: dict[str, Sequence[float]]) -> list[dict[str, Any]]:
    """Return rows with each error, its ratio to the previous row and the observed order.

    The observed order is ``log(e_{i-1}/e_i)/log(p_{i-1}/p_i)`` for a
    refinement parameter ``p`` (mesh size or timestep); missing or nonpositive
    errors (for example a rejected run) give ``None``.
    """
    rows = []
    for index, parameter in enumerate(parameters):
        row: dict[str, Any] = {parameter_name: float(parameter)}
        for name, values in errors.items():
            error = values[index]
            row[name] = None if error is None else float(error)
            ratio = order = None
            if index > 0:
                previous = values[index - 1]
                if error and previous and error > 0 and previous > 0:
                    ratio = previous/error
                    order = math.log(ratio)/math.log(parameters[index - 1]/parameter)
            row[f"{name}_ratio"] = ratio
            row[f"{name}_order"] = order
        rows.append(row)
    return rows


def markdown_table(rows: Iterable[dict[str, Any]], columns: Sequence[str] | None = None) -> str:
    """Render rows as a Markdown table; floats use 4 significant digits, missing values ``-``."""
    rows = list(rows)
    if columns is None:
        columns = []
        for row in rows:
            columns.extend(key for key in row if key not in columns)

    def cell(value):
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "-"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|"*len(columns)]
    lines += ["| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join(lines)


class StudyRecorder:
    """Per-step JSONL/CSV rows plus a JSON/Markdown convergence summary for one study."""

    def __init__(self, directory: str | Path, prefix: str, *, enabled: bool = True):
        self.directory = Path(directory)
        self.prefix = prefix
        self.steps = DiagnosticsRecorder(self.directory, f"{prefix}_steps", enabled=enabled)
        self.enabled = bool(enabled)

    def record_step(self, record: Any, **context) -> None:
        """Record one accepted or rejected step, flattened, with run context columns."""
        self.steps.record({**context, **flatten(record)})

    def write_summary(self, summary: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
        """Close the step stream and write ``<prefix>_summary.json`` and ``.md``."""
        self.steps.close()
        if not self.enabled:
            return
        payload = {"summary": summary, "tables": tables}
        (self.directory / f"{self.prefix}_summary.json").write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
        blocks = [f"# {self.prefix}", ""]
        blocks += [f"- {key}: {value}" for key, value in summary.items()]
        for name, rows in tables.items():
            blocks += ["", f"## {name}", "", markdown_table(rows)]
        (self.directory / f"{self.prefix}_summary.md").write_text("\n".join(blocks) + "\n")


__all__ = [
    "StudyRecorder",
    "convergence_rows",
    "flatten",
    "markdown_table",
    "error_norm",
    "error_norms",
    "radius_weight",
    "sampled_minimum",
]
