"""Adaptive follow-up scheduling for the torsion numerical-test manifest."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


SUCCESS = {"strict_convergence", "certified_subband_convergence"}


def _value(row: dict[str, Any], name: str, default: float = math.inf) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return default


def _band(row: dict[str, Any]) -> tuple[float, float]:
    return round(float(row["alpha_t1"]), 12), round(float(row["alpha_t2"]), 12)


def fine_confirmation_parameters(
    rows: Iterable[dict[str, Any]],
    *,
    fine_targets: dict[str, int],
) -> list[dict[str, Any]]:
    """Choose fixed anchors, worst success, and both sides of a status boundary."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("kind") == "robustness" and row.get("state") in {"completed", "failed"}:
            grouped[str(row["geometry"])].append(row)

    output: list[dict[str, Any]] = []
    fixed = ((0.60, 0.70), (0.45, 0.50), (0.05, 0.25), (0.75, 0.95))
    for geometry, geometry_rows in grouped.items():
        chosen: dict[tuple[float, float], set[str]] = {band: {"fixed_anchor"} for band in fixed}
        successes = [row for row in geometry_rows if row.get("classification") in SUCCESS]
        failures = [row for row in geometry_rows if row.get("classification") not in SUCCESS]
        if successes:
            worst = max(
                successes,
                key=lambda row: _value(row, "bestLeakageRel", 0.0) + _value(row, "bestMissingRel", 0.0),
            )
            chosen.setdefault(_band(worst), set()).add("worst_success")
        if successes and failures:
            success, failure = min(
                ((success, failure) for success in successes for failure in failures),
                key=lambda pair: math.dist(_band(pair[0]), _band(pair[1])),
            )
            chosen.setdefault(_band(success), set()).add("boundary_success_side")
            chosen.setdefault(_band(failure), set()).add("boundary_failure_side")
        for (alpha_t1, alpha_t2), reasons in sorted(chosen.items()):
            output.append({
                "kind": "fine_confirmation",
                "geometry": geometry,
                "order": 4,
                "dof_target": int(fine_targets[geometry]),
                "alpha_t1": alpha_t1,
                "alpha_t2": alpha_t2,
                "adaptive_reasons": sorted(reasons),
            })
    return output


def additional_refinement_parameters(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Schedule one finer level when the last two changes do not decrease."""
    grouped: dict[tuple[str, int, float, float], list[dict[str, Any]]] = defaultdict(list)
    available_targets: dict[tuple[str, int, float, float], set[int]] = defaultdict(set)
    for row in rows:
        if row.get("kind") not in {"mesh_order", "mesh_order_difficult", "mesh_order_refinement"}:
            continue
        key = (str(row["geometry"]), int(row["order"]), float(row["alpha_t1"]), float(row["alpha_t2"]))
        available_targets[key].add(int(row.get("dof_target", row.get("ndof", 0))))
        if row.get("state") != "completed" or row.get("classification") not in SUCCESS:
            continue
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (geometry, order, alpha_t1, alpha_t2), group in grouped.items():
        ordered = sorted(group, key=lambda row: int(row.get("ndof", row.get("dof_target", 0))))
        if len(ordered) < 2:
            continue
        changes = [
            _value(row, "two_grid_change")
            for row in ordered[-2:]
            if _value(row, "two_grid_change") < math.inf
        ]
        unresolved = len(changes) < 2 or changes[-1] >= changes[-2] or changes[-1] > 0.02
        finest_completed_target = int(ordered[-1].get("dof_target", ordered[-1]["ndof"]))
        finer_level_already_scheduled = any(
            target > finest_completed_target for target in available_targets[
                (geometry, order, alpha_t1, alpha_t2)
            ]
        )
        if unresolved and not finer_level_already_scheduled:
            output.append({
                "kind": "mesh_order_refinement",
                "geometry": geometry,
                "order": order,
                "dof_target": int(math.ceil(1.6 * finest_completed_target / 10_000) * 10_000),
                "alpha_t1": alpha_t1,
                "alpha_t2": alpha_t2,
                "adaptive_reasons": ["unresolved_two_grid_change"],
            })
    return output
