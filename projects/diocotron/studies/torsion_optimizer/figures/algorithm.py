"""Focused homotopy, inexact-Newton, and threshold-plateau figures."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from projects.diocotron.paths import resolve_archive_path


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _log(row: dict[str, Any], name: str) -> list[dict[str, str]]:
    run_dir = row.get("run_dir")
    if not run_dir:
        return []
    path = resolve_archive_path(str(run_dir)) / "logs" / name
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig, output: Path, filename: str) -> list[Path]:
    import matplotlib.pyplot as plt

    path = output / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return [path]


def _homotopy_figure(output: Path, rows: Sequence[dict[str, Any]], stage: str):
    import matplotlib.pyplot as plt

    planned = [row for row in rows if row.get("kind") == "homotopy_robustness"]
    terminal = [row for row in planned if row.get("state") in {"completed", "failed"}
                and _number(row.get("homotopy_lambda_max_logged")) is not None]
    if not terminal:
        return None, []
    terminal.sort(key=lambda row: (
        str(row.get("geometry")), str(row.get("difficulty")),
        str(row.get("homotopy_scheme")),
    ))
    scheme_names = {
        "production": "prod.",
        "conservative": "cons.",
        "conservative_no_predictor": "cons., no pred.",
    }
    labels = [
        f"{str(row.get('geometry')).replace('smooth_star', 'star')}\n"
        f"{str(row.get('difficulty', ''))[:4]}, "
        f"{scheme_names.get(str(row.get('homotopy_scheme')), str(row.get('homotopy_scheme')))}"
        for row in terminal
    ]
    x = np.arange(len(terminal))
    lambdas = np.asarray([float(row["homotopy_lambda_max_logged"]) for row in terminal])
    colors = np.where(lambdas >= 1.0 - 1.0e-12, "#31a354", "#de2d26")
    work = np.asarray([_number(row.get("homotopy_logged_newton_iterations")) or 0.0
                       for row in terminal])
    timing = np.asarray([_number(row.get("homotopy_logged_seconds")) or 0.0
                         for row in terminal])
    rejected = np.asarray([_number(row.get("homotopy_logged_rejected")) or 0.0
                           for row in terminal])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    axes[0].bar(x, lambdas, color=colors)
    axes[0].axhline(1.0, color="black", linewidth=.7, linestyle="--")
    axes[0].set(ylabel=r"terminal $\lambda$", ylim=(0.0, 1.06))
    axes[1].bar(x, work, color="#3182bd", label="Newton iterations")
    axes[1].plot(x, rejected, "o-", color="#e6550d", label="rejected stages")
    axes[1].set_ylabel("continuation work")
    axes[1].legend(fontsize=7)
    axes[2].bar(x, timing, color="#756bb1")
    axes[2].set_ylabel("homotopy time [s]")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=42, ha="right", fontsize=7)
        ax.grid(axis="y", linewidth=.3, alpha=.4)
    status = "actual" if len(terminal) == len(planned) else "partial"
    entry = {
        "status": status,
        "case_ids": [str(row["id"]) for row in terminal],
        "stage": stage,
    }
    return entry, _save(fig, output, "homotopy_robustness.png")


def _inner_newton_figure(output: Path, rows: Sequence[dict[str, Any]], stage: str):
    import matplotlib.pyplot as plt

    planned = [row for row in rows if row.get("kind") == "inner_newton_accuracy"]
    terminal = [row for row in planned if row.get("state") in {"completed", "failed"}]
    if not terminal:
        return None, []
    references: dict[tuple[str, str], dict[str, Any]] = {}
    for row in terminal:
        if row.get("inner_accuracy") == "adaptive" and all(
                _number(row.get(key)) is not None
                for key in ("bestC1Phi", "bestC2Phi", "bestLeakageRel", "bestMissingRel")):
            references[(str(row.get("geometry")), str(row.get("difficulty")))] = row
    usable: list[tuple[dict[str, Any], float, float]] = []
    for row in terminal:
        reference = references.get((str(row.get("geometry")), str(row.get("difficulty"))))
        if reference is None or any(_number(row.get(key)) is None for key in
                                    ("bestC1Phi", "bestC2Phi", "bestLeakageRel", "bestMissingRel")):
            continue
        reference_width = max(
            abs(float(reference["bestC2Phi"]) - float(reference["bestC1Phi"])), 1.0e-30
        )
        threshold_error = math.hypot(
            float(row["bestC1Phi"]) - float(reference["bestC1Phi"]),
            float(row["bestC2Phi"]) - float(reference["bestC2Phi"]),
        ) / reference_width
        discrepancy = float(row["bestLeakageRel"]) + float(row["bestMissingRel"])
        reference_discrepancy = (
            float(reference["bestLeakageRel"]) + float(reference["bestMissingRel"])
        )
        discrepancy_error = abs(discrepancy - reference_discrepancy) / max(
            abs(reference_discrepancy), 1.0e-30
        )
        usable.append((row, threshold_error, discrepancy_error))
    if not usable:
        return None, []
    order = {"adaptive": 0, "fixed_1e-7": 1, "fixed_1e-5": 2, "fixed_1e-3": 3}
    usable.sort(key=lambda item: (
        str(item[0].get("geometry")), str(item[0].get("difficulty")),
        order.get(str(item[0].get("inner_accuracy")), 99),
    ))
    labels = [
        f"{str(row.get('geometry')).replace('smooth_star', 'star')}\n"
        f"{str(row.get('difficulty', ''))[:4]}, {str(row.get('inner_accuracy')).replace('fixed_', '')}"
        for row, _, _ in usable
    ]
    x = np.arange(len(usable))
    runtime = [_number(row.get("timeTotal")) or np.nan for row, _, _ in usable]
    threshold_error = [max(error, 1.0e-14) for _, error, _ in usable]
    discrepancy_error = [max(error, 1.0e-14) for _, _, error in usable]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    axes[0].bar(x, runtime, color="#3182bd")
    axes[0].set_ylabel("total time [s]")
    axes[1].bar(x, threshold_error, color="#31a354")
    axes[1].set(ylabel="threshold-pair error / reference width", yscale="log")
    axes[2].bar(x, discrepancy_error, color="#e6550d")
    axes[2].set(ylabel="relative discrepancy change", yscale="log")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=42, ha="right", fontsize=7)
        ax.grid(axis="y", linewidth=.3, alpha=.4)
    status = "actual" if len(terminal) == len(planned) else "partial"
    entry = {
        "status": status,
        "case_ids": [str(row["id"]) for row, _, _ in usable],
        "stage": stage,
    }
    return entry, _save(fig, output, "inner_newton_accuracy.png")


def _plateau_figure(output: Path, rows: Sequence[dict[str, Any]], stage: str):
    import matplotlib.pyplot as plt

    sources: dict[tuple[str, float, float], tuple[dict[str, Any], list[dict[str, str]]]] = {}
    for row in rows:
        if row.get("state") not in {"completed", "failed"}:
            continue
        records = _log(row, "optimization.csv")
        if not records:
            continue
        key = (str(row.get("geometry")), float(row.get("alpha_t1", -1)),
               float(row.get("alpha_t2", -1)))
        previous = sources.get(key)
        preference = 0 if row.get("kind") == "trajectory" else 1
        previous_preference = 0 if previous and previous[0].get("kind") == "trajectory" else 1
        if previous is None or preference < previous_preference or len(records) > len(previous[1]):
            sources[key] = (row, records)
    selected = sorted(sources.values(), key=lambda item: (
        str(item[0].get("geometry")), float(item[0].get("alpha_t1", 0))))[:4]
    if not selected:
        return None, []
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    for ax, (row, records) in zip(axes.flat, selected, strict=False):
        k = np.asarray([int(float(record["k"])) for record in records])
        c1 = np.asarray([float(record["c1Phi"]) for record in records])
        c2 = np.asarray([float(record["c2Phi"]) for record in records])
        width = max(abs(c2[-1] - c1[-1]), 1.0e-30)
        threshold_distance = np.hypot(c1 - c1[-1], c2 - c2[-1]) / width
        discrepancy = np.asarray([
            float(record["leakageRel"]) + float(record["missingRel"]) for record in records
        ])
        gradient = np.asarray([
            _number(record.get("projectedGradNorm")) or np.nan for record in records
        ])
        ax.semilogy(k, np.maximum(threshold_distance, 1.0e-14), label="distance to terminal pair")
        ax.semilogy(k, np.maximum(discrepancy, 1.0e-14), label="leakage + missing")
        ax.semilogy(k, np.maximum(gradient, 1.0e-14), label="projected gradient")
        onset = _number(row.get("plateau_onset_iteration"))
        if onset is not None:
            ax.axvline(onset, color="#de2d26", linestyle="--", linewidth=.9,
                       label=f"plateau ({row.get('plateau_type', '')})")
        ax.set(
            title=(f"{str(row.get('geometry')).replace('_', ' ')} "
                   f"[{float(row.get('alpha_t1', 0)):.2f}, {float(row.get('alpha_t2', 0)):.2f}]"),
            xlabel="outer iteration",
        )
        ax.legend(fontsize=7)
        ax.grid(axis="y", linewidth=.3, alpha=.4)
    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")
    entry = {
        "status": "actual" if len(selected) == 4 else "partial",
        "case_ids": [str(row["id"]) for row, _ in selected],
        "stage": stage,
    }
    return entry, _save(fig, output, "threshold_optimization_plateaus.png")


def create_algorithm_figures(
        bundle: Path,
        rows: Sequence[dict[str, Any]],
        stage: str,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Render controlled algorithmic studies and return registry updates."""
    output = bundle / "figures"
    output.mkdir(parents=True, exist_ok=True)
    updates: dict[str, dict[str, Any]] = {}
    written: list[Path] = []
    for identifier, renderer in (
        ("homotopy_robustness", _homotopy_figure),
        ("inner_newton", _inner_newton_figure),
        ("threshold_plateau", _plateau_figure),
    ):
        entry, assets = renderer(output, rows, stage)
        if entry is not None:
            updates[identifier] = entry
            written.extend(assets)
    return updates, written
