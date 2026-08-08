#!/usr/bin/env python3
"""Consolidate validation, timing, tuning, and profiling campaign outputs.

Only configurations that passed every recorded numerical gate are eligible for
performance recommendations.  The report keeps solve-only, setup-plus-solve,
retuned cold-start, and end-to-end metrics separate.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


TIME_METRICS = (
    ("median_solve_ms", "solve_only"),
    ("median_hot_time_to_solution_ms", "setup_plus_solve"),
    ("representative_cold_time_to_solution_ms", "retuned_cold_start"),
    ("median_end_to_end_ms", "end_to_end"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Defaults to RESULTS_DIR/final_analysis.",
    )
    parser.add_argument("--require-full", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload if isinstance(payload, dict) else None


def _float(row: dict[str, Any], field: str) -> float:
    try:
        value = float(row.get(field, "nan"))
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _integer(row: dict[str, Any], field: str) -> int | None:
    value = _float(row, field)
    return None if not math.isfinite(value) else int(value)


def _bool(row: dict[str, Any], field: str) -> bool:
    return str(row.get(field, "")).strip().lower() in {"1", "true", "yes"}


def _group(
    rows: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        grouped.setdefault(key, []).append(row)
    return grouped


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fields:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _eligible(rows: Iterable[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _bool(row, "all_passed") and math.isfinite(_float(row, metric))
    ]


def _winners(
    rows: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
    metric: str,
    winner_kind: str,
) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for key, group in sorted(_group(rows, group_fields).items()):
        candidates = _eligible(group, metric)
        if not candidates:
            winners.append(
                {
                    **dict(zip(group_fields, key, strict=True)),
                    "winner_kind": winner_kind,
                    "metric": metric,
                    "configuration": "",
                    "value_ms": math.nan,
                    "status": "no-passing-configuration",
                }
            )
            continue
        winner = min(candidates, key=lambda row: _float(row, metric))
        winners.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "winner_kind": winner_kind,
                "metric": metric,
                "configuration": winner.get("configuration", ""),
                "preconditioner": winner.get("preconditioner", ""),
                "polynomial_degree": winner.get("polynomial_degree", ""),
                "requested_operator": winner.get("operator", ""),
                "resolved_operators": winner.get("resolved_operators", ""),
                "block_jacobi_applications": winner.get(
                    "resolved_block_jacobi_applications",
                    winner.get("block_jacobi_application", ""),
                ),
                "asm_applications": winner.get(
                    "resolved_asm_applications", winner.get("asm_application", "")
                ),
                "iterations": _float(winner, "median_iterations"),
                "value_ms": _float(winner, metric),
                "worst_solver_relative_residual": _float(
                    winner, "worst_solver_relative_residual"
                ),
                "worst_physical_relative_residual": _float(
                    winner, "worst_physical_relative_residual"
                ),
                "status": "passing",
            }
        )
    return winners


def _robust_rankings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = sorted({(row.get("case", ""), row.get("order", "")) for row in rows})
    best_by_pair: dict[tuple[str, str], float] = {}
    for pair, group in _group(rows, ("case", "order")).items():
        candidates = _eligible(group, "median_solve_ms")
        if candidates:
            best_by_pair[pair] = min(_float(row, "median_solve_ms") for row in candidates)

    rankings: list[dict[str, Any]] = []
    for (configuration,), group in _group(rows, ("configuration",)).items():
        by_pair = {(row.get("case", ""), row.get("order", "")): row for row in group}
        passing = [row for row in group if _bool(row, "all_passed")]
        solve_values = [
            _float(row, "median_solve_ms")
            for row in passing
            if math.isfinite(_float(row, "median_solve_ms"))
        ]
        slowdowns = [
            _float(row, "median_solve_ms") / best_by_pair[pair]
            for pair, row in by_pair.items()
            if pair in best_by_pair
            and _bool(row, "all_passed")
            and math.isfinite(_float(row, "median_solve_ms"))
        ]
        complete = len(by_pair) == len(pairs)
        all_passed = complete and len(passing) == len(pairs)
        geometric = (
            math.exp(statistics.mean(math.log(value) for value in solve_values))
            if solve_values and all(value > 0.0 for value in solve_values)
            else math.nan
        )
        first = group[0]
        rankings.append(
            {
                "configuration": configuration,
                "preconditioner": first.get("preconditioner", ""),
                "polynomial_degree": first.get("polynomial_degree", ""),
                "coverage": len(by_pair),
                "expected_case_orders": len(pairs),
                "coverage_fraction": 0.0 if not pairs else len(by_pair) / len(pairs),
                "passing_case_orders": len(passing),
                "all_case_orders_passed": all_passed,
                "geometric_mean_solve_ms": geometric,
                "median_solve_ms_across_case_orders": (
                    statistics.median(solve_values) if solve_values else math.nan
                ),
                "worst_slowdown_to_case_best": max(slowdowns, default=math.nan),
                "worst_solver_relative_residual": max(
                    (
                        value
                        for value in (
                            _float(row, "worst_solver_relative_residual")
                            for row in group
                        )
                        if math.isfinite(value)
                    ),
                    default=math.nan,
                ),
                "worst_physical_relative_residual": max(
                    (
                        value
                        for value in (
                            _float(row, "worst_physical_relative_residual")
                            for row in group
                        )
                        if math.isfinite(value)
                    ),
                    default=math.nan,
                ),
            }
        )
    rankings.sort(
        key=lambda row: (
            not bool(row["all_case_orders_passed"]),
            -int(row["passing_case_orders"]),
            (
                _float(row, "geometric_mean_solve_ms")
                if math.isfinite(_float(row, "geometric_mean_solve_ms"))
                else math.inf
            ),
        )
    )
    for index, row in enumerate(rankings, start=1):
        row["robust_rank"] = index if row["all_case_orders_passed"] else ""
    return rankings


def _profile_bottlenecks(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    result: list[dict[str, Any]] = []
    for profile in payload.get("detailed_gmres_profiles", []):
        operation_profile = profile.get("operation_profile", {})
        operations = operation_profile.get("operations", [])
        total = sum(float(item.get("gpu_time_ms", 0.0)) for item in operations)
        ordered = sorted(
            operations,
            key=lambda item: float(item.get("gpu_time_ms", 0.0)),
            reverse=True,
        )
        for rank, operation in enumerate(ordered[:5], start=1):
            gpu_ms = float(operation.get("gpu_time_ms", 0.0))
            result.append(
                {
                    "case": profile.get("case", ""),
                    "order": profile.get("polynomial_order", ""),
                    "configuration": profile.get("configuration", ""),
                    "preconditioner": profile.get("preconditioner", ""),
                    "polynomial_degree": profile.get("polynomial_degree", ""),
                    "operation_rank": rank,
                    "operation": operation.get("category", ""),
                    "count": operation.get("count", 0),
                    "gpu_time_ms": gpu_ms,
                    "gpu_time_fraction": 0.0 if total <= 0.0 else gpu_ms / total,
                }
            )
    return result


def _discretization_error_trends(
    validation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Report exact-solution errors without turning p-trends into solver gates."""

    trends: list[dict[str, Any]] = []
    for (case,), case_rows in sorted(_group(validation, ("case",)).items()):
        previous_primal = math.nan
        previous_flux = math.nan
        by_order = _group(case_rows, ("order",))
        for (order,), group in sorted(
            by_order.items(),
            key=lambda item: int(float(item[0][0])),
        ):
            passing = [row for row in group if _bool(row, "all_passed")]
            source = passing[0] if passing else group[0]
            primal = _float(source, "reference_primal_l2_error")
            flux = _float(source, "reference_flux_l2_error")
            if not math.isfinite(primal):
                primal = _float(source, "median_primal_l2_error")
            if not math.isfinite(flux):
                flux = _float(source, "median_flux_l2_error")
            trends.append(
                {
                    "case": case,
                    "order": order,
                    "primal_l2_error": primal,
                    "flux_l2_error": flux,
                    "primal_reduction_from_previous_order": (
                        previous_primal / primal
                        if math.isfinite(previous_primal) and primal > 0.0
                        else math.nan
                    ),
                    "flux_reduction_from_previous_order": (
                        previous_flux / flux
                        if math.isfinite(previous_flux) and flux > 0.0
                        else math.nan
                    ),
                    "has_passing_solver_configuration": bool(passing),
                }
            )
            previous_primal = primal
            previous_flux = flux
    return trends


def _format_ms(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.3f}"


def _markdown_report(
    *,
    coverage: dict[str, Any],
    best: list[dict[str, Any]],
    robust: list[dict[str, Any]],
    kernel: list[dict[str, Any]],
    polynomial: list[dict[str, Any]],
    bottlenecks: list[dict[str, Any]],
    discretization: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    lines = [
        "# Diffusion-reaction GPU campaign analysis",
        "",
        "## Numerical validation",
        "",
        f"- Cases: {coverage['cases']}.",
        f"- Orders: {coverage['orders']}.",
        f"- Validated case/order/configuration rows: {coverage['summary_rows']}.",
        f"- Fully passing rows: {coverage['passing_summary_rows']}.",
        f"- Case/order pairs without any passing configuration: {coverage['uncovered_case_orders']}.",
        "",
        "A configuration is ranked only if every measured repetition passed the true-residual, physical-residual, finiteness, and independent-trace checks.",
        "",
        "## Best validated configuration by case and order",
        "",
        "| Case | p | Configuration | Solve ms | Iterations |",
        "|---|---:|---|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row.get('case', '')} | {row.get('order', '')} | "
            f"{row.get('configuration', '') or 'none passed'} | "
            f"{_format_ms(row.get('value_ms'))} | {_format_ms(row.get('iterations'))} |"
        )

    lines.extend(["", "## Robust configuration across all cases/orders", ""])
    robust_passing = [row for row in robust if row["all_case_orders_passed"]]
    if robust_passing:
        lines.extend(
            [
                "| Rank | Configuration | Geometric mean solve ms | Worst slowdown |",
                "|---:|---|---:|---:|",
            ]
        )
        for row in robust_passing[:10]:
            lines.append(
                f"| {row['robust_rank']} | {row['configuration']} | "
                f"{_format_ms(row['geometric_mean_solve_ms'])} | "
                f"{_format_ms(row['worst_slowdown_to_case_best'])}x |"
            )
    else:
        lines.append("No single configuration passed every case/order pair.")

    lines.extend(
        [
            "",
            "## Performance tuning",
            "",
            f"- Kernel winners recorded: {len(kernel)}.",
            f"- Polynomial-degree winners recorded: {len(polynomial)}.",
            f"- Detailed GMRES bottleneck rows recorded: {len(bottlenecks)}.",
            f"- Exact-solution discretization error rows recorded: {len(discretization)}.",
            "",
            "Use solve-only winners for repeated right-hand sides, setup-plus-solve winners for one solve on a cached architecture, and retuned-cold-start winners for the first run on a new GPU/problem shape.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    results = args.results_dir
    prefix = args.output_prefix or (results / "final_analysis")

    validation_path = results / "all_cases_p4_p6_validation_summary.csv"
    kernel_path = results / "hard_kernel_sweep_summary.csv"
    polynomial_path = results / "hard_polynomial_degree_sweep_summary.csv"
    profile_path = results / "component_profile_p4_p6.json"

    validation = _read_csv(validation_path)
    kernel_rows = _read_csv(kernel_path)
    polynomial_rows = _read_csv(polynomial_path)
    profile_payload = _read_json(profile_path)
    if not validation:
        raise SystemExit(f"missing or empty validation summary: {validation_path}")

    pairs = sorted({(row.get("case", ""), row.get("order", "")) for row in validation})
    uncovered = sum(
        not any(_bool(row, "all_passed") for row in group)
        for group in _group(validation, ("case", "order")).values()
    )
    coverage = {
        "cases": len({row.get("case", "") for row in validation}),
        "orders": len({row.get("order", "") for row in validation}),
        "case_orders": len(pairs),
        "summary_rows": len(validation),
        "passing_summary_rows": sum(_bool(row, "all_passed") for row in validation),
        "uncovered_case_orders": uncovered,
    }

    best: list[dict[str, Any]] = []
    for metric, label in TIME_METRICS:
        best.extend(
            _winners(
                validation,
                group_fields=("case", "order"),
                metric=metric,
                winner_kind=label,
            )
        )
    best_solve = [row for row in best if row["winner_kind"] == "solve_only"]
    robust = _robust_rankings(validation)
    kernel_winners = _winners(
        kernel_rows,
        group_fields=("case", "order", "preconditioner"),
        metric="median_solve_ms",
        winner_kind="kernel_solve_only",
    ) if kernel_rows else []
    polynomial_winners = _winners(
        polynomial_rows,
        group_fields=("case", "order", "preconditioner"),
        metric="median_hot_time_to_solution_ms",
        winner_kind="polynomial_setup_plus_solve",
    ) if polynomial_rows else []
    bottlenecks = _profile_bottlenecks(profile_payload)
    discretization = _discretization_error_trends(validation)

    warnings: list[str] = []
    if uncovered:
        warnings.append(f"{uncovered} case/order pairs have no passing configuration.")
    for path in (kernel_path, polynomial_path, profile_path):
        if not path.exists():
            warnings.append(f"Optional full-campaign output is missing: {path.name}.")
    if args.require_full and warnings:
        missing_full = [warning for warning in warnings if "output is missing" in warning]
        if missing_full:
            raise SystemExit("; ".join(missing_full))

    outputs = {
        "best_configurations_csv": prefix.with_name(prefix.name + "_best_configurations.csv"),
        "robust_rankings_csv": prefix.with_name(prefix.name + "_robust_rankings.csv"),
        "kernel_winners_csv": prefix.with_name(prefix.name + "_kernel_winners.csv"),
        "polynomial_winners_csv": prefix.with_name(prefix.name + "_polynomial_winners.csv"),
        "profile_bottlenecks_csv": prefix.with_name(prefix.name + "_profile_bottlenecks.csv"),
        "discretization_errors_csv": prefix.with_name(
            prefix.name + "_discretization_errors.csv"
        ),
        "json": prefix.with_suffix(".json"),
        "markdown": prefix.with_suffix(".md"),
    }
    _write_csv(outputs["best_configurations_csv"], best)
    _write_csv(outputs["robust_rankings_csv"], robust)
    _write_csv(outputs["kernel_winners_csv"], kernel_winners)
    _write_csv(outputs["polynomial_winners_csv"], polynomial_winners)
    _write_csv(outputs["profile_bottlenecks_csv"], bottlenecks)
    _write_csv(outputs["discretization_errors_csv"], discretization)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": coverage,
        "best_by_case_order_and_metric": best,
        "robust_rankings": robust,
        "kernel_winners": kernel_winners,
        "polynomial_degree_winners": polynomial_winners,
        "profile_bottlenecks": bottlenecks,
        "discretization_error_trends": discretization,
        "warnings": warnings,
        "sources": {
            "validation": str(validation_path),
            "kernel_sweep": str(kernel_path),
            "polynomial_sweep": str(polynomial_path),
            "component_profile": str(profile_path),
        },
    }
    outputs["json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    outputs["markdown"].write_text(
        _markdown_report(
            coverage=coverage,
            best=best_solve,
            robust=robust,
            kernel=kernel_winners,
            polynomial=polynomial_winners,
            bottlenecks=bottlenecks,
            discretization=discretization,
            warnings=warnings,
        ),
        encoding="utf-8",
    )

    print(f"Analysis JSON : {outputs['json']}")
    print(f"Analysis report: {outputs['markdown']}")
    print(
        f"Validation coverage: {coverage['cases']} cases, {coverage['orders']} orders, "
        f"{coverage['uncovered_case_orders']} uncovered case/order pairs"
    )
    return 1 if uncovered else 0


if __name__ == "__main__":
    raise SystemExit(main())
