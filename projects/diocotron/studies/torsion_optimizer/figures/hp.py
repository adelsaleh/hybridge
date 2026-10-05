"""Unambiguous mesh/order and matched-cost figures for the numerical tests."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SUCCESS = {"strict_convergence", "certified_subband_convergence"}
FIXED_P_ORDERS = (2, 4, 6)
FIXED_P_CAPTION = (
    "Fixed-p enrichment on the same linear triangulation: P4/P6 threshold "
    "agreement is shown explicitly. Nonmonotone external-reference L2/H1 "
    "errors include optimizer branch/stopping differences and do not mean "
    "higher p is intrinsically worse. Runtime and global finite-element DOFs "
    "show the corresponding cost growth."
)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _save(fig, output: Path, stem: str) -> list[Path]:
    import matplotlib.pyplot as plt

    png = output / f"{stem}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [png]


def _star_easy_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("kind") == "mesh_order"
        and row.get("geometry") == "smooth_star"
        and math.isclose(float(row.get("alpha_t1", -1.0)), 0.60)
        and math.isclose(float(row.get("alpha_t2", -1.0)), 0.70)
        and row.get("classification") in SUCCESS
        and _number(row.get("normalized_h")) is not None
    ]


def _reference(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    explicit = [row for row in rows if row.get("is_numerical_reference")]
    if explicit:
        return explicit[0]
    usable = [row for row in rows if _number(row.get("bestC1Phi")) is not None
              and _number(row.get("bestC2Phi")) is not None]
    return min(usable, key=lambda row: float(row["normalized_h"])) if usable else None


def create_hp_figures(
    bundle: Path,
    rows: Sequence[dict[str, Any]],
    stage: str,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Render h-convergence separately from matched-DOF polynomial comparisons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _star_easy_rows(rows)
    if len(data) < 2:
        return {}, []
    output = bundle / "figures"
    output.mkdir(parents=True, exist_ok=True)
    reference = _reference(data)
    reference_width = None
    if reference is not None:
        reference_width = max(
            float(reference["bestC2Phi"]) - float(reference["bestC1Phi"]),
            1.0e-30,
        )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), constrained_layout=True)
    colors = {2: "#1f77b4", 4: "#2ca02c", 6: "#d62728"}
    for order in (2, 4, 6):
        subset = sorted(
            (row for row in data if int(row["order"]) == order),
            key=lambda row: float(row["normalized_h"]),
        )
        if not subset:
            continue
        h = np.asarray([float(row["normalized_h"]) for row in subset])
        if reference is not None and reference_width is not None:
            threshold_error = np.asarray([
                math.hypot(
                    float(row["bestC1Phi"]) - float(reference["bestC1Phi"]),
                    float(row["bestC2Phi"]) - float(reference["bestC2Phi"]),
                ) / reference_width
                for row in subset
            ])
            mask = threshold_error > 0.0
            if np.any(mask):
                axes[0, 0].loglog(h[mask], threshold_error[mask], "o-",
                                  color=colors[order], label=f"P{order}")
        for ax, key in ((axes[0, 1], "relative_l2"), (axes[1, 0], "relative_h1")):
            points = [
                (float(row["normalized_h"]), float(row[key]))
                for row in subset
                if _number(row.get(key)) is not None and float(row[key]) > 0.0
            ]
            if points:
                ax.loglog(*zip(*points, strict=True), "o-", color=colors[order], label=f"P{order}")
        axes[1, 1].loglog(
            [float(row.get("ndof", row["dof_target"])) for row in subset],
            h,
            "o-",
            color=colors[order],
            label=f"P{order}",
        )

    axes[0, 0].set(title="threshold-pair error", ylabel=r"$\|c-c_{\rm ref}\|/\Delta c_{\rm ref}$")
    axes[0, 1].set(title="field error", ylabel=r"relative $L^2$ error")
    axes[1, 0].set(title="gradient error", ylabel=r"relative $H^1$ error")
    axes[1, 1].set(
        title="matched DOFs imply different meshes",
        xlabel="global finite-element DOFs",
        ylabel=r"$h_{\max}/\operatorname{diam}(\Omega)$",
    )
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel(r"$h_{\max}/\operatorname{diam}(\Omega)$")
    for ax in axes.flat:
        if ax.lines:
            ax.legend(fontsize=8)
        ax.grid(True, which="both", linewidth=.3, alpha=.35)
    fig.suptitle(
        "Smooth-star convergence: lines compare h-refinement within each p;\n"
        "the lower-right panel exposes the coarser linear geometry at higher p",
        fontsize=11,
    )
    assets = _save(fig, output, "hp_convergence_summary")

    cost = [row for row in data if _number(row.get("timeTotal")) is not None]
    if cost:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
        for order in (2, 4, 6):
            subset = sorted(
                (row for row in cost if int(row["order"]) == order),
                key=lambda row: float(row["timeTotal"]),
            )
            for ax, key in zip(axes, ("relative_l2", "relative_h1"), strict=True):
                points = [
                    (float(row["timeTotal"]), float(row[key]), int(row.get("ndof", row["dof_target"])))
                    for row in subset
                    if _number(row.get(key)) is not None and float(row[key]) > 0.0
                ]
                if points:
                    x, y, dofs = zip(*points, strict=True)
                    ax.loglog(x, y, "o-", color=colors[order], label=f"P{order}")
                    for px, py, ndof in points:
                        ax.annotate(f"{ndof / 1000:.0f}k", (px, py), xytext=(3, 3),
                                    textcoords="offset points", fontsize=6)
        axes[0].set(xlabel="total runtime [s]", ylabel=r"relative $L^2$ error")
        axes[1].set(xlabel="total runtime [s]", ylabel=r"relative $H^1$ error")
        for ax in axes:
            if ax.lines:
                ax.legend()
            ax.grid(True, which="both", linewidth=.3, alpha=.35)
        assets.extend(_save(fig, output, "matched_cost_accuracy"))

    case_ids = [str(row["id"]) for row in data]
    status = "actual" if all(any(int(row["order"]) == p for row in data) for p in (2, 4, 6)) else "partial"
    updates = {
        "hp_convergence": {
            "status": status,
            "case_ids": case_ids,
            "stage": stage,
            "caption": (
                "Smooth-star h-convergence within P2, P4, and P6, with the matched-DOF "
                "mesh sizes shown explicitly. Higher p is not intrinsically less accurate; "
                "at fixed global DOFs it currently uses a much coarser linear boundary mesh."
            ),
        },
        "matched_cost": {
            "status": status if cost else "partial",
            "case_ids": [str(row["id"]) for row in cost],
            "stage": stage,
            "caption": "Relative field errors against runtime; labels give global DOFs.",
        },
    }
    return updates, assets
