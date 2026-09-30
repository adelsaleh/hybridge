#!/usr/bin/env python3
"""Render unconditional PNG diagnostics for an adaptive threshold search.

The adaptive search can legitimately finish without a geometrically eligible
threshold pair.  Its figures must therefore depend only on the completed
search archive, not on a reduced-optimizer handoff.  This module validates
that every solved row has a complete candidate-equilibrium PNG, renders the
threshold-space diagnostics, and builds both an all-candidate contact archive
and a compact representative storyboard.

No finite-element package is imported here.  Figure generation is a serial,
post-processing operation over rank-zero-gathered PNGs and CSV/JSON records.
All generated visual assets are PNG files; mesh lines are absent because the
source candidate renderer suppresses them before this module is invoked.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from projects.diocotron.paths import resolve_archive_path


class CandidateArchiveError(RuntimeError):
    """Raised when the one-PNG-per-solved-candidate contract is violated."""


@dataclass(frozen=True)
class AdaptiveFigureBundle:
    """Paths and counts produced by :func:`render_adaptive_search_figures`."""

    search_summary: Path
    curve_alignment: Path
    newton_work: Path
    candidate_storyboard: Path
    candidate_contacts: tuple[Path, ...]
    candidate_count: int
    pruned_count: int
    has_eligible_winner: bool

    def output_paths(self) -> tuple[Path, ...]:
        return (
            self.search_summary,
            self.curve_alignment,
            self.newton_work,
            self.candidate_storyboard,
            *self.candidate_contacts,
        )


def _truth(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "nan", "none"}
    return bool(value)


def _number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _value(row: dict[str, Any], names: Iterable[str], default: float = math.nan) -> float:
    normalized = {
        str(key).replace("_", "").lower(): value for key, value in row.items()
    }
    for name in names:
        key = str(name).replace("_", "").lower()
        if key in normalized:
            return _number(normalized[key], default)
    return default


def _bool_value(row: dict[str, Any], names: Iterable[str]) -> bool:
    normalized = {
        str(key).replace("_", "").lower(): value for key, value in row.items()
    }
    for name in names:
        key = str(name).replace("_", "").lower()
        if key in normalized:
            return _truth(normalized[key])
    return False


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise CandidateArchiveError(f"adaptive search table is empty: {path}")
    return rows


def locate_search_directory(workflow_dir: Path) -> Path:
    """Locate ``grid.csv`` for either a complete workflow or search directory."""

    workflow_dir = resolve_archive_path(workflow_dir).resolve()
    for candidate in (workflow_dir / "grid_search", workflow_dir):
        if (candidate / "grid.csv").is_file():
            return candidate
    raise FileNotFoundError(f"no adaptive grid.csv below {workflow_dir}")


def _candidate_identity(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(_number(row.get("stage"), -1)),
        int(_number(row.get("candidate"), -1)),
        str(row.get("uid", "")),
    )


def validate_candidate_png_archive(
    rows: Sequence[dict[str, Any]], search_dir: Path
) -> list[tuple[dict[str, Any], Path]]:
    """Require one unique, valid PNG beneath ``search_dir`` for every row.

    ``grid.csv`` contains solved candidates only.  Points rejected by the
    dominance rule before a nonlinear solve live in ``pruned.csv`` and do not
    need an equilibrium image.
    """

    from PIL import Image

    search_dir = resolve_archive_path(search_dir).resolve()
    result: list[tuple[dict[str, Any], Path]] = []
    identities: set[tuple[int, int, str]] = set()
    paths: set[Path] = set()
    for row in rows:
        identity = _candidate_identity(row)
        if identity in identities:
            raise CandidateArchiveError(f"duplicate solved candidate identity: {identity}")
        identities.add(identity)
        raw = str(row.get("candidatePng", "")).strip()
        if not raw:
            raise CandidateArchiveError(
                f"solved candidate {identity} has no candidatePng entry"
            )
        candidate_path = resolve_archive_path(raw)
        if not candidate_path.is_absolute():
            candidate_path = search_dir / candidate_path
        candidate_path = candidate_path.resolve()
        try:
            candidate_path.relative_to(search_dir)
        except ValueError as exc:
            raise CandidateArchiveError(
                f"candidate PNG escapes the search archive: {candidate_path}"
            ) from exc
        if candidate_path.suffix.lower() != ".png":
            raise CandidateArchiveError(f"candidate image is not PNG: {candidate_path}")
        if candidate_path in paths:
            raise CandidateArchiveError(f"candidate PNG is reused by two rows: {candidate_path}")
        paths.add(candidate_path)
        if not candidate_path.is_file() or candidate_path.stat().st_size == 0:
            raise CandidateArchiveError(f"candidate PNG is missing or empty: {candidate_path}")
        try:
            with Image.open(candidate_path) as image:
                if image.format != "PNG":
                    raise CandidateArchiveError(
                        f"candidate file does not contain PNG data: {candidate_path}"
                    )
                image.verify()
        except CandidateArchiveError:
            raise
        except Exception as exc:
            raise CandidateArchiveError(
                f"candidate PNG cannot be decoded: {candidate_path}: {exc}"
            ) from exc
        result.append((row, candidate_path))
    if len(result) != len(rows):
        raise CandidateArchiveError(
            f"validated {len(result)} candidate PNGs for {len(rows)} solved rows"
        )
    return sorted(result, key=lambda item: _candidate_identity(item[0]))


def _diagnostic_key(row: dict[str, Any]) -> tuple[Any, ...]:
    eligible = _bool_value(row, ("selectionEligible",)) and _bool_value(
        row, ("geometryEligible",)
    )
    pde = _bool_value(row, ("converged",)) and _bool_value(
        row, ("boundSatisfied",)
    )
    fit = _value(row, ("pairFitScore",), -math.inf)
    containment = _value(row, ("pairContainmentScore",), -math.inf)
    discrepancy = _value(row, ("leakageRel",), math.inf) + _value(
        row, ("missingRel",), math.inf
    )
    residual = _value(row, ("residual",), math.inf)
    candidate = int(_number(row.get("candidate"), 10**9))
    if eligible:
        return (0, discrepancy, -containment, -fit, residual, candidate)
    if pde:
        return (1, -fit, -containment, discrepancy, residual, candidate)
    return (2, residual, discrepancy, candidate)


def _row_for_document_entry(
    rows: Sequence[dict[str, Any]], entry: Any
) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    candidate = int(_number(entry.get("candidate"), -1))
    uid = str(entry.get("uid", ""))
    for row in rows:
        if uid and str(row.get("uid", "")) == uid:
            return row
        if candidate >= 0 and int(_number(row.get("candidate"), -2)) == candidate:
            return row
    return None


def _winner_record(
    rows: Sequence[dict[str, Any]], search_dir: Path
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    document: dict[str, Any] = {}
    winner_path = Path(search_dir) / "winner.json"
    if winner_path.is_file():
        document = json.loads(winner_path.read_text(encoding="utf-8"))
    winner = _row_for_document_entry(rows, document.get("winner"))
    if winner is not None:
        return winner, True, document
    diagnostic = _row_for_document_entry(rows, document.get("diagnosticBest"))
    return diagnostic or min(rows, key=_diagnostic_key), False, document


def _uses_corroborated_thickness(document: dict[str, Any]) -> bool:
    """Detect the v2 thickness rule recorded in the search manifest."""

    search = document.get("search")
    pruning = search.get("pruning") if isinstance(search, dict) else None
    decision = pruning.get("thicknessDecision", "") if isinstance(pruning, dict) else ""
    normalized = str(decision).replace("_", "").replace(" ", "").upper()
    return "ROBUSTTOOTHICKAND" in normalized


def _columns(rows: Sequence[dict[str, Any]], names: Iterable[str]) -> np.ndarray:
    return np.asarray([_value(row, names) for row in rows], dtype=np.float64)


def _triangle_axes(axis: Any) -> None:
    from matplotlib.patches import Polygon

    axis.add_patch(
        Polygon(
            ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            closed=True,
            facecolor="#f6f6f6",
            edgecolor="none",
            zorder=-5,
        )
    )
    axis.plot((0.0, 1.0), (0.0, 1.0), color="black", linewidth=0.9)
    axis.set(
        xlabel=r"$c_{1,\phi}/T_{\max}$",
        ylabel=r"$c_{2,\phi}/T_{\max}$",
        xlim=(-0.025, 1.025),
        ylim=(-0.025, 1.025),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.20)


def _mark_selected(axis: Any, row: dict[str, Any], *, winner: bool) -> None:
    axis.scatter(
        [_value(row, ("c1Hat", "geometryC1Normalized"))],
        [_value(row, ("c2Hat", "geometryC2Normalized"))],
        marker="*",
        s=210,
        facecolor="#ffcc00" if winner else "white",
        edgecolor="black",
        linewidth=1.0,
        label="eligible winner" if winner else "best observed; no handoff",
        zorder=12,
    )


def _metric_scatter(
    fig: Any,
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    logarithmic: bool = False,
    sizes: np.ndarray | float = 46.0,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    plotted = values[finite]
    label = colorbar_label
    if logarithmic:
        plotted = np.log10(np.maximum(plotted, 1.0e-18))
        label = rf"$\log_{{10}}$({colorbar_label})"
    if np.isscalar(sizes):
        point_sizes: Any = sizes
    else:
        point_sizes = np.asarray(sizes)[finite]
    if np.any(finite):
        artist = axis.scatter(
            x[finite],
            y[finite],
            c=plotted,
            cmap=cmap,
            s=point_sizes,
            edgecolors="black",
            linewidths=0.28,
            zorder=3,
        )
        fig.colorbar(artist, ax=axis, label=label, shrink=0.82)
    else:
        axis.text(
            0.5,
            0.5,
            "metric unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_title(title)


def _pruned_points(search_dir: Path) -> list[dict[str, str]]:
    path = Path(search_dir) / "pruned.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _render_search_summary(
    rows: Sequence[dict[str, Any]],
    pruned: Sequence[dict[str, Any]],
    selected: dict[str, Any],
    has_winner: bool,
    path: Path,
    *,
    corroborated_thickness: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = _columns(rows, ("c1Hat", "geometryC1Normalized"))
    y = _columns(rows, ("c2Hat", "geometryC2Normalized"))
    generation = _columns(rows, ("stage", "generation"))
    sizes = 40.0 + 8.0 * np.maximum(generation, 0.0)
    converged = np.asarray([_bool_value(row, ("converged",)) for row in rows])
    failure = np.asarray(
        [
            any(
                token in str(row.get("newtonStatus", "")).upper()
                for token in ("FAIL", "ERROR", "DIVERG")
            )
            or not math.isfinite(_value(row, ("residual",)))
            for row in rows
        ]
    )
    partial = ~converged & ~failure
    has_handoff_geometry = any("handoffGeometryEligible" in row for row in rows)
    practical = np.asarray(
        [
            _bool_value(
                row,
                ("handoffGeometryEligible",)
                if has_handoff_geometry
                else ("geometryEligible", "selectionEligible"),
            )
            for row in rows
        ]
    )
    strict = np.asarray(
        [_bool_value(row, ("strictGeometryEligible",)) for row in rows]
    )
    thick = np.asarray([_bool_value(row, ("tooThick",)) for row in rows])

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 10.1), constrained_layout=True)
    status_axis = axes[0, 0]
    for mask, marker, color, label in (
        (converged, "o", "#0072b2", "Newton converged"),
        (partial, "^", "#e69f00", "partial / capped"),
        (failure, "x", "#d55e00", "failed"),
    ):
        marker_style = (
            {"edgecolors": "black", "linewidths": 0.35}
            if marker != "x"
            else {"linewidths": 1.1}
        )
        status_axis.scatter(
            x[mask], y[mask], marker=marker, s=sizes[mask], color=color, label=label,
            **marker_style,
        )
    status_axis.scatter(
        x[practical], y[practical], s=sizes[practical] + 65.0, facecolors="none",
        edgecolors="#00a65a", linewidths=1.4, label="geometry gate passed",
    )
    status_axis.scatter(
        x[strict], y[strict], s=sizes[strict] + 100.0, facecolors="none",
        edgecolors="#7a0177", linewidths=1.3, label="strictly certified",
    )
    if pruned:
        px = _columns(pruned, ("c1Hat",))
        py = _columns(pruned, ("c2Hat",))
        status_axis.scatter(
            px, py, marker="x", s=26, color="#777777", linewidths=0.8,
            label="pruned before solve",
        )
    status_axis.set_title("Solved, partial, failed, and pruned points")
    _mark_selected(status_axis, selected, winner=has_winner)
    status_axis.legend(loc="upper left", fontsize=7.4, frameon=True)

    _metric_scatter(
        fig, axes[0, 1], x, y, _columns(rows, ("pairFitScore",)),
        title="Two-curve fit", colorbar_label="pair fit score", sizes=sizes,
    )
    axes[0, 1].scatter(
        x[thick], y[thick], marker="x", s=64, color="#d62728", linewidths=1.1,
        label="too thick",
    )
    _mark_selected(axes[0, 1], selected, winner=has_winner)
    axes[0, 1].legend(loc="upper left", fontsize=7.5)

    span = _columns(rows, ("robustTauSpan",))
    span_limit = _columns(rows, ("robustSpanLimit",))
    span_ratio = np.divide(
        span,
        span_limit,
        out=np.full_like(span, np.nan),
        where=np.isfinite(span_limit) & (span_limit > 0.0),
    )
    robust_warning = np.asarray(
        [
            _bool_value(row, ("robustSpanWarning", "robustTooThick"))
            for row in rows
        ]
    )
    thickness_title = (
        "v2 corroborated thickness"
        if corroborated_thickness
        else "v1 robust-span classification"
    )
    _metric_scatter(
        fig, axes[0, 2], x, y, span_ratio,
        title=thickness_title, colorbar_label=r"robust span / limit",
        cmap="coolwarm", sizes=sizes,
    )
    if corroborated_thickness:
        warning_only = robust_warning & ~thick
        axes[0, 2].scatter(
            x[warning_only], y[warning_only], marker="s", s=58,
            facecolors="none", edgecolors="#377eb8", linewidths=1.0,
            label="robust-span warning only",
        )
    axes[0, 2].scatter(
        x[thick], y[thick], marker="x", s=68, color="black", linewidths=1.0,
        label=(
            "corroborated too-thick"
            if corroborated_thickness
            else "v1 robust-only witness"
        ),
    )
    if pruned:
        axes[0, 2].scatter(
            _columns(pruned, ("c1Hat",)), _columns(pruned, ("c2Hat",)),
            marker="1", s=45, color="#777777", linewidths=0.8,
            label=(
                "v2 dominance-pruned"
                if corroborated_thickness
                else "v1 dominated point"
            ),
        )
    _mark_selected(axes[0, 2], selected, winner=has_winner)
    axes[0, 2].legend(loc="upper left", fontsize=7.5)

    r1 = _columns(rows, ("r1", "lowerPositionResidual"))
    r2 = _columns(rows, ("r2", "upperPositionResidual"))
    t1 = _columns(rows, ("lowerPositionTolerance",))
    t2 = _columns(rows, ("upperPositionTolerance",))
    normalized_residual = np.hypot(
        np.divide(r1, t1, out=np.full_like(r1, np.nan), where=t1 > 0.0),
        np.divide(r2, t2, out=np.full_like(r2, np.nan), where=t2 > 0.0),
    )
    _metric_scatter(
        fig, axes[1, 0], x, y, normalized_residual,
        title="Level-curve root residual", colorbar_label=r"$\|(r_1/\delta_1,r_2/\delta_2)\|_2$",
        logarithmic=True, cmap="magma_r", sizes=sizes,
    )
    _mark_selected(axes[1, 0], selected, winner=has_winner)

    _metric_scatter(
        fig, axes[1, 1], x, y,
        _columns(rows, ("hardJaccard", "activeJaccard")),
        title="Crisp band overlap", colorbar_label="Jaccard index", cmap="viridis", sizes=sizes,
    )
    _mark_selected(axes[1, 1], selected, winner=has_winner)

    _metric_scatter(
        fig, axes[1, 2], x, y, _columns(rows, ("newtonIterations",)),
        title="Nonlinear work", colorbar_label="Newton iterations", cmap="plasma", sizes=sizes,
    )
    _mark_selected(axes[1, 2], selected, winner=has_winner)

    for axis in axes.flat:
        _triangle_axes(axis)
    status = "eligible handoff" if has_winner else "no eligible handoff; star is diagnostic best"
    fig.suptitle(
        f"Adaptive full-triangle threshold search: {len(rows)} solved, "
        f"{len(pruned)} pruned; {status}", fontsize=15,
    )
    fig.savefig(path, dpi=270, facecolor="white", format="png")
    plt.close(fig)


def _render_curve_alignment(
    rows: Sequence[dict[str, Any]],
    selected: dict[str, Any],
    has_winner: bool,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    generation = _columns(rows, ("stage", "generation"))
    r1 = _columns(rows, ("r1", "lowerPositionResidual"))
    r2 = _columns(rows, ("r2", "upperPositionResidual"))
    t1 = _columns(rows, ("lowerPositionTolerance",))
    t2 = _columns(rows, ("upperPositionTolerance",))
    rn1 = np.divide(r1, t1, out=np.full_like(r1, np.nan), where=t1 > 0.0)
    rn2 = np.divide(r2, t2, out=np.full_like(r2, np.nan), where=t2 > 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), constrained_layout=True)

    finite = np.isfinite(rn1) & np.isfinite(rn2) & np.isfinite(generation)
    if np.any(finite):
        artist = axes[0].scatter(
            rn1[finite], rn2[finite], c=generation[finite], cmap="viridis",
            s=48, edgecolors="black", linewidths=0.3,
        )
        fig.colorbar(artist, ax=axes[0], label="adaptive generation")
    axes[0].add_patch(
        Rectangle((-1.0, -1.0), 2.0, 2.0, facecolor="#00a65a", alpha=0.10,
                  edgecolor="#00a65a", linestyle="--", linewidth=1.0)
    )
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].axvline(0.0, color="black", linewidth=0.7)
    axes[0].set_xscale("symlog", linthresh=1.0)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set(
        xlabel=r"lower-curve residual $r_1/\delta_1$",
        ylabel=r"upper-curve residual $r_2/\delta_2$",
        title="Two-dimensional level-curve root",
    )

    lower_fit = _columns(rows, ("lowerBoundaryFit", "lowerContainment"))
    upper_fit = _columns(rows, ("upperBoundaryFit", "upperContainment"))
    finite = np.isfinite(lower_fit) & np.isfinite(upper_fit) & np.isfinite(generation)
    if np.any(finite):
        artist = axes[1].scatter(
            lower_fit[finite], upper_fit[finite], c=generation[finite], cmap="viridis",
            s=48, edgecolors="black", linewidths=0.3,
        )
        fig.colorbar(artist, ax=axes[1], label="adaptive generation")
    axes[1].plot((0.0, 1.0), (0.0, 1.0), "--", color="#777777", linewidth=0.8)
    axes[1].set(
        xlabel="lower target-curve fit",
        ylabel="upper target-curve fit",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        title="Balanced boundary containment",
    )

    precision = _columns(rows, ("hardPrecision",))
    recall = _columns(rows, ("hardRecall",))
    jaccard = _columns(rows, ("hardJaccard", "activeJaccard"))
    finite = np.isfinite(precision) & np.isfinite(recall) & np.isfinite(jaccard)
    if np.any(finite):
        artist = axes[2].scatter(
            precision[finite], recall[finite], c=jaccard[finite], cmap="viridis",
            s=52, edgecolors="black", linewidths=0.3,
        )
        fig.colorbar(artist, ax=axes[2], label="hard-band Jaccard")
    axes[2].scatter([1.0], [1.0], marker="*", s=180, color="#00a65a", label="ideal")
    axes[2].set(
        xlabel="hard-band precision",
        ylabel="hard-band recall",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        title="Leakage--missing-area tradeoff",
    )
    axes[2].legend(loc="lower left")
    for axis in axes:
        axis.grid(alpha=0.22, which="both")
    fig.suptitle(
        "Candidate level-curve alignment and crisp-set overlap"
        + ("" if has_winner else " (no eligible handoff)"),
        fontsize=14,
    )
    fig.savefig(path, dpi=280, facecolor="white", format="png")
    plt.close(fig)


def _render_newton_work(
    rows: Sequence[dict[str, Any]],
    selected: dict[str, Any],
    has_winner: bool,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = _columns(rows, ("c1Hat", "geometryC1Normalized"))
    y = _columns(rows, ("c2Hat", "geometryC2Normalized"))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), constrained_layout=True)
    specs = (
        ("residual", "Terminal Newton residual", r"$\|R\|_2$", "magma_r", True),
        ("newtonIterations", "Newton iterations", "iterations", "plasma", False),
        ("wallTime", "Candidate wall time", "seconds", "cividis", True),
    )
    for axis, (key, title, label, cmap, logarithmic) in zip(axes, specs):
        _metric_scatter(
            fig, axis, x, y, _columns(rows, (key,)), title=title,
            colorbar_label=label, cmap=cmap, logarithmic=logarithmic,
        )
        _mark_selected(axis, selected, winner=has_winner)
        _triangle_axes(axis)
    fig.suptitle("Adaptive candidate nonlinear cost and qualification", fontsize=14)
    fig.savefig(path, dpi=280, facecolor="white", format="png")
    plt.close(fig)


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _candidate_label(row: dict[str, Any], role: str | None = None) -> str:
    prefix = f"{role}: " if role else ""
    return (
        f"{prefix}g={int(_number(row.get('stage'), -1))}, "
        f"id={int(_number(row.get('candidate'), -1))}; "
        f"c1/Tmax={_value(row, ('c1Hat',)):.4f}, "
        f"c2/Tmax={_value(row, ('c2Hat',)):.4f}\n"
        f"R={_value(row, ('residual',)):.2e}, "
        f"fit={_value(row, ('pairFitScore',)):.3f}, "
        f"J={_value(row, ('hardJaccard', 'activeJaccard')):.3f}"
    )


def _contact_sheets(
    archive: Sequence[tuple[dict[str, Any], Path]],
    figure_dir: Path,
    prefix: str,
    per_sheet: int,
) -> tuple[Path, ...]:
    from PIL import Image, ImageDraw

    if per_sheet not in {1, 2, 4}:
        raise ValueError("contacts-per-sheet must be one, two, or four")
    columns = 1 if per_sheet == 1 else 2
    rows_per_page = int(math.ceil(per_sheet / columns))
    cell_width = 1400
    image_height = 760
    header_height = 92
    cell_height = header_height + image_height
    font = _font(25)
    paths: list[Path] = []
    for page, offset in enumerate(range(0, len(archive), per_sheet), start=1):
        page_items = archive[offset : offset + per_sheet]
        sheet = Image.new(
            "RGB", (columns * cell_width, rows_per_page * cell_height), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for index, (row, source) in enumerate(page_items):
            image = Image.open(source).convert("RGB")
            image.thumbnail((cell_width, image_height), Image.Resampling.LANCZOS)
            column = index % columns
            row_index = index // columns
            x0 = column * cell_width
            y0 = row_index * cell_height
            draw.text((x0 + 18, y0 + 16), _candidate_label(row), fill="black", font=font)
            x = x0 + (cell_width - image.width) // 2
            y = y0 + header_height + (image_height - image.height) // 2
            sheet.paste(image, (x, y))
        path = figure_dir / f"{prefix}_candidate_contact_{page:03d}.png"
        sheet.save(path, format="PNG", compress_level=6)
        paths.append(path)
    return tuple(paths)


def representative_candidate_rows(
    rows: Sequence[dict[str, Any]], selected: dict[str, Any], has_winner: bool
) -> list[tuple[str, dict[str, Any]]]:
    """Choose a deterministic qualitative progression without requiring a winner."""

    result: list[tuple[str, dict[str, Any]]] = []
    used: set[tuple[int, int, str]] = set()

    def add(role: str, row: dict[str, Any]) -> None:
        identity = _candidate_identity(row)
        if identity not in used:
            result.append((role, row))
            used.add(identity)

    first_generation = min(int(_number(row.get("stage"), 0)) for row in rows)
    last_generation = max(int(_number(row.get("stage"), 0)) for row in rows)
    add(
        "coarse incumbent",
        min(
            (row for row in rows if int(_number(row.get("stage"), 0)) == first_generation),
            key=_diagnostic_key,
        ),
    )

    def normalized_curve_error(row: dict[str, Any], lower: bool) -> float:
        residual = abs(_value(row, ("r1",) if lower else ("r2",), math.inf))
        tolerance = _value(
            row,
            ("lowerPositionTolerance",) if lower else ("upperPositionTolerance",),
            math.nan,
        )
        return residual / tolerance if math.isfinite(tolerance) and tolerance > 0.0 else residual

    add("best lower curve", min(rows, key=lambda row: normalized_curve_error(row, True)))
    add("best upper curve", min(rows, key=lambda row: normalized_curve_error(row, False)))
    add("best paired fit", max(rows, key=lambda row: _value(row, ("pairFitScore",), -math.inf)))
    add(
        "best hard overlap",
        max(rows, key=lambda row: _value(row, ("hardJaccard", "activeJaccard"), -math.inf)),
    )
    add(
        "last-generation incumbent",
        min(
            (row for row in rows if int(_number(row.get("stage"), 0)) == last_generation),
            key=_diagnostic_key,
        ),
    )
    selected_identity = _candidate_identity(selected)
    selected_role = "eligible handoff" if has_winner else "best observed; no handoff"
    if selected_identity in used:
        result = [
            ((f"{role}; {selected_role}" if _candidate_identity(row) == selected_identity else role), row)
            for role, row in result
        ]
    else:
        result.append((selected_role, selected))
    if len(result) > 6:
        selected_item = next(item for item in result if _candidate_identity(item[1]) == selected_identity)
        result = result[:5]
        if all(_candidate_identity(item[1]) != selected_identity for item in result):
            result.append(selected_item)
    return result


def _candidate_storyboard(
    archive: Sequence[tuple[dict[str, Any], Path]],
    selected: dict[str, Any],
    has_winner: bool,
    path: Path,
) -> None:
    from PIL import Image, ImageDraw

    by_identity = {_candidate_identity(row): source for row, source in archive}
    representatives = representative_candidate_rows(
        [row for row, _ in archive], selected, has_winner
    )
    first_source = by_identity[_candidate_identity(representatives[0][1])]
    first_image = Image.open(first_source).convert("RGB")
    midpoint = first_image.height // 2
    design = first_image.crop((0, 0, first_image.width, midpoint))
    canvas_width = 2800
    design_height = round(design.height * canvas_width / design.width)
    design = design.resize((canvas_width, design_height), Image.Resampling.LANCZOS)
    design_header = 82
    gutter = 22
    state_header = 92
    state_width = canvas_width // 2
    state_images: list[tuple[str, Any]] = []
    for role, row in representatives:
        image = Image.open(by_identity[_candidate_identity(row)]).convert("RGB")
        state = image.crop((0, image.height // 2, round(0.75 * image.width), image.height))
        state_height = round(state.height * state_width / state.width)
        state = state.resize((state_width, state_height), Image.Resampling.LANCZOS)
        state_images.append((_candidate_label(row, role), state))
    rows_count = int(math.ceil(len(state_images) / 2))
    state_height = max(image.height for _, image in state_images)
    canvas_height = (
        design_header + design_height + gutter
        + rows_count * (state_header + state_height)
        + max(0, rows_count - 1) * gutter
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (20, 20),
        "Target design fields (shown once); orange target curves, blue equilibrium curves",
        fill="black", font=_font(31),
    )
    canvas.paste(design, (0, design_header))
    states_top = design_header + design_height + gutter
    label_font = _font(24)
    for index, (label, image) in enumerate(state_images):
        column = index % 2
        row_index = index // 2
        x0 = column * state_width
        y0 = states_top + row_index * (state_header + state_height + gutter)
        draw.text((x0 + 16, y0 + 15), label, fill="black", font=label_font)
        y_image = y0 + state_header + (state_height - image.height) // 2
        canvas.paste(image, (x0, y_image))
    canvas.save(path, format="PNG", compress_level=6)


def render_adaptive_search_figures(
    workflow_dir: Path,
    figure_dir: Path,
    prefix: str = "adaptive_threshold",
    *,
    contacts_per_sheet: int = 4,
) -> AdaptiveFigureBundle:
    """Validate a search archive and render all unconditional PNG assets."""

    search_dir = locate_search_directory(Path(workflow_dir))
    rows = _read_csv(search_dir / "grid.csv")
    archive = validate_candidate_png_archive(rows, search_dir)
    pruned = _pruned_points(search_dir)
    selected, has_winner, document = _winner_record(rows, search_dir)
    figure_dir = Path(figure_dir).resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not prefix or any(character in prefix for character in ("/", "\\")):
        raise ValueError("figure prefix must be a nonempty filename stem")
    search_summary = figure_dir / f"{prefix}_search_summary.png"
    curve_alignment = figure_dir / f"{prefix}_curve_alignment.png"
    newton_work = figure_dir / f"{prefix}_newton_work.png"
    candidate_storyboard = figure_dir / f"{prefix}_candidate_storyboard.png"
    _render_search_summary(
        rows,
        pruned,
        selected,
        has_winner,
        search_summary,
        corroborated_thickness=_uses_corroborated_thickness(document),
    )
    _render_curve_alignment(rows, selected, has_winner, curve_alignment)
    _render_newton_work(rows, selected, has_winner, newton_work)
    contacts = _contact_sheets(
        archive, figure_dir, prefix, contacts_per_sheet
    )
    _candidate_storyboard(archive, selected, has_winner, candidate_storyboard)
    bundle = AdaptiveFigureBundle(
        search_summary=search_summary,
        curve_alignment=curve_alignment,
        newton_work=newton_work,
        candidate_storyboard=candidate_storyboard,
        candidate_contacts=contacts,
        candidate_count=len(rows),
        pruned_count=len(pruned),
        has_eligible_winner=has_winner,
    )
    for output in bundle.output_paths():
        if output.suffix.lower() != ".png" or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"adaptive figure generator did not create PNG: {output}")
    return bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--figure-prefix", default="adaptive_threshold")
    parser.add_argument("--contacts-per-sheet", type=int, choices=(1, 2, 4), default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = render_adaptive_search_figures(
        args.workflow_dir,
        args.figure_dir,
        args.figure_prefix,
        contacts_per_sheet=args.contacts_per_sheet,
    )
    for path in bundle.output_paths():
        print(f"FIGURE {path}")
    print(
        f"CANDIDATE_ARCHIVE solved={bundle.candidate_count} "
        f"contacts={len(bundle.candidate_contacts)} "
        f"eligibleWinner={int(bundle.has_eligible_winner)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
