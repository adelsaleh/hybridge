"""Detailed inexact-Newton replay figure for non-final threshold pairs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from projects.diocotron.paths import resolve_archive_path


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _records(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    case_ids: list[str] = []
    for row in rows:
        run_dir = row.get("run_dir")
        if not run_dir:
            continue
        path = resolve_archive_path(str(run_dir)) / "logs" / "inexact_newton.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                records.append({**record, "case_id": str(row["id"])})
        case_ids.append(str(row["id"]))
    return records, sorted(set(case_ids))


def create_inexact_newton_figure(
    bundle: Path,
    rows: Sequence[dict[str, Any]],
    stage: str,
) -> tuple[dict[str, Any] | None, list[Path]]:
    """Plot state, sensitivity, reduced-gradient, step, decision, and cost errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records, case_ids = _records(rows)
    usable = [
        record for record in records
        if _number(record.get("requested_tolerance")) is not None
        and _number(record.get("reduced_gradient_relative_error")) is not None
    ]
    if not usable:
        return None, []
    policies = sorted({str(record.get("policy", "unknown")) for record in usable})
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), constrained_layout=True)
    for policy_index, policy in enumerate(policies):
        subset = [record for record in usable if str(record.get("policy")) == policy]
        tolerance_groups: dict[float, list[dict[str, Any]]] = {}
        for record in subset:
            tolerance_groups.setdefault(float(record["requested_tolerance"]), []).append(record)
        tolerances = sorted(tolerance_groups)
        color = colors(policy_index % 10)

        def maximum(field: str) -> np.ndarray:
            return np.asarray([
                max(
                    (_number(item.get(field)) or 0.0 for item in tolerance_groups[tolerance]),
                    default=np.nan,
                )
                for tolerance in tolerances
            ])

        state = np.maximum(maximum("state_l2_relative_error"),
                           maximum("state_h1_relative_error"))
        sensitivity = np.maximum.reduce((
            maximum("sensitivity1_l2_relative_error"),
            maximum("sensitivity1_h1_relative_error"),
            maximum("sensitivity2_l2_relative_error"),
            maximum("sensitivity2_h1_relative_error"),
        ))
        gradient = maximum("reduced_gradient_relative_error")
        gradient_angle = maximum("reduced_gradient_angle_degrees")
        step = maximum("threshold_step_relative_error")
        step_angle = maximum("threshold_step_angle_degrees")
        elapsed = np.asarray([
            sum(_number(item.get("elapsed")) or 0.0 for item in tolerance_groups[tolerance])
            for tolerance in tolerances
        ])
        disagreement = np.asarray([
            1.0 - sum(
                str(item.get("acceptance_decision_agrees", "false")).lower()
                in {"1", "true", "yes"}
                for item in tolerance_groups[tolerance]
            ) / max(len(tolerance_groups[tolerance]), 1)
            for tolerance in tolerances
        ])
        for ax, values, label in (
            (axes[0, 0], state, policy),
            (axes[0, 1], sensitivity, policy),
            (axes[0, 2], gradient, policy),
            (axes[1, 0], step, policy),
            (axes[1, 1], disagreement, policy),
            (axes[1, 2], elapsed, policy),
        ):
            ax.loglog(tolerances, np.maximum(values, 1.0e-15), "o-",
                      color=color, label=label)
        axes[0, 2].plot(tolerances, np.maximum(gradient_angle / 5.0, 1.0e-15), "--",
                        color=color, alpha=.6)
        axes[1, 0].plot(tolerances, np.maximum(step_angle / 5.0, 1.0e-15), "--",
                        color=color, alpha=.6)

    axes[0, 0].set(title="state error", ylabel=r"max relative $L^2/H^1$")
    axes[0, 1].set(title="sensitivity error", ylabel=r"max relative $L^2/H^1$")
    axes[0, 2].set(title="reduced-gradient error", ylabel="relative error; dashed = angle / 5 deg")
    axes[1, 0].set(title="threshold-step error", ylabel="relative error; dashed = angle / 5 deg")
    axes[1, 1].set(title="accept/reject disagreement", ylabel="fraction")
    axes[1, 2].set(title="replay work", ylabel="summed time [s]")
    for ax in axes.flat:
        ax.set_xlabel("intermediate Newton residual target")
        ax.invert_xaxis()
        ax.grid(True, which="both", linewidth=.3, alpha=.35)
        if ax.lines:
            ax.legend(fontsize=6)
    axes[0, 2].axhline(.10, color="black", linestyle=":", linewidth=.8)
    axes[1, 0].axhline(.10, color="black", linestyle=":", linewidth=.8)
    axes[1, 1].axhline(.05, color="black", linestyle=":", linewidth=.8)

    output = bundle / "figures"
    output.mkdir(parents=True, exist_ok=True)
    png = output / "inner_newton_accuracy.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    snapshots = {str(record.get("snapshot_id")) for record in usable}
    tolerances = {float(record["requested_tolerance"]) for record in usable}
    status = "actual" if len(snapshots) >= 4 and len(tolerances) >= 5 else "partial"
    entry = {
        "status": status,
        "case_ids": case_ids,
        "stage": stage,
        "caption": (
            "Inexact intermediate-state errors relative to tight MUMPS references: state and "
            "sensitivity fields, reduced gradient, threshold step, acceptance decisions, and work."
        ),
    }
    return entry, [png]
