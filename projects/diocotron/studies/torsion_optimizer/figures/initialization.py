"""Initialization-policy comparison for full, fast, and fit-window starts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def create_initialization_figure(
    bundle: Path,
    rows: Sequence[dict[str, Any]],
    stage: str,
) -> tuple[dict[str, Any] | None, list[Path]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    planned = [row for row in rows if row.get("kind") == "initialization_policy_v3"]
    terminal = [row for row in planned if row.get("state") in {"completed", "failed"}]
    if not terminal:
        return None, []
    terminal.sort(key=lambda row: (
        str(row.get("geometry")), str(row.get("difficulty")),
        str(row.get("algorithm_variant")),
    ))
    labels = [
        f"{str(row['geometry']).replace('smooth_star', 'star')}\n"
        f"{str(row.get('difficulty', ''))[:4]}, "
        f"{str(row.get('algorithm_variant', '')).replace('_hminus1', '').replace('_', ' ')}"
        for row in terminal
    ]
    x = np.arange(len(terminal))
    lam = np.asarray([_number(row.get("homotopy_lambda_max_logged")) or 0.0 for row in terminal])
    hminus = np.asarray([_number(row.get("phase_hminus1_search_seconds")) or 0.0 for row in terminal])
    fit = np.asarray([_number(row.get("phase_window_fit_seconds")) or 0.0 for row in terminal])
    transfer = np.asarray([_number(row.get("phase_mesh_transfer_seconds")) or 0.0 for row in terminal])
    homotopy = np.asarray([_number(row.get("phase_homotopy_seconds")) or 0.0 for row in terminal])
    newton = np.asarray([_number(row.get("homotopy_logged_newton_iterations")) or 0.0 for row in terminal])

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), constrained_layout=True)
    colors = np.where(lam >= 1.0 - 1.0e-12, "#31a354", "#de2d26")
    axes[0].bar(x, lam, color=colors)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=.7)
    axes[0].set(ylabel=r"terminal homotopy $\lambda$", ylim=(0.0, 1.06))
    bottom = np.zeros(len(terminal))
    for values, name in ((hminus, r"$H^{-1}$ search"), (fit, "window fit"),
                         (transfer, "mesh transfer"), (homotopy, "homotopy")):
        axes[1].bar(x, values, bottom=bottom, label=name)
        bottom += values
    axes[1].set_ylabel("initialization time [s]")
    axes[1].legend(fontsize=7)
    axes[2].bar(x, newton, color="#3182bd")
    axes[2].set_ylabel("homotopy Newton iterations")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=48, ha="right", fontsize=6)
        ax.grid(axis="y", linewidth=.3, alpha=.35)

    output = bundle / "figures"
    output.mkdir(parents=True, exist_ok=True)
    png = output / "initialization_homotopy_summary.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    entry = {
        "status": "actual" if len(terminal) == len(planned) else "partial",
        "case_ids": [str(row["id"]) for row in terminal],
        "stage": stage,
        "caption": (
            "Automatic initialization comparison: original and value-only fast H-minus-one "
            "searches, same-run window-fit fallback, continuation success, and nonlinear work."
        ),
    }
    return entry, [png]
