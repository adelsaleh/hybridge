#!/usr/bin/env python3
"""Repeat the advection-reaction preset runner and summarize printed timings.

This benchmark intentionally drives ``scripts/advection_reaction/run_cases.py`` as a
subprocess.  It is meant to answer "what does the preset runner itself report?"
rather than timing a lower-level assembly function.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ADV_REA_SCRIPT = REPO_ROOT / "scripts" / "advection_reaction" / "run_cases.py"

SUMMARY_LABELS = (
    "prep time(s)",
    "setup time(s)",
    "glb_solve time(s)",
    "recons time(s)",
    "tot time(s)",
    "ordering time(s)",
    "ILU time(s)",
    "perm time(s)",
    "Krylov time(s)",
    "iterations",
    "solver rel res",
    "free trace rel res",
    "assembly",
    "boundary mode",
    "trace ordering",
)

PRINTED_PHASE_RE = re.compile(
    r"^\s*(?P<label>.+?)\s+\.\.\.\s+done in\s+"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>ms|s)\s*$"
)
NUMBA_TIMING_RE = re.compile(r"numba trace assembly timings:\s*(?P<body>.+)")


@dataclass(frozen=True)
class RunResult:
    backend: str
    repeat: int
    returncode: int
    elapsed_seconds: float
    log_path: str
    metrics: dict[str, Any]


def _metric_key(label: str) -> str:
    key = label.strip().lower()
    key = key.replace("time(s)", "time_s")
    key = key.replace("(s)", "_s")
    key = key.replace("/", "_")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def _parse_scalar(raw: str) -> float | int | str:
    value = raw.strip().replace(",", "")
    try:
        number = float(value)
    except ValueError:
        return raw.strip()
    if number.is_integer() and re.fullmatch(r"[-+]?\d+(?:\.0+)?", value):
        return int(number)
    return number


def _seconds(value: str, unit: str) -> float:
    seconds = float(value)
    if unit == "ms":
        return seconds / 1000.0
    return seconds


def parse_adv_rea_output(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for label in SUMMARY_LABELS:
        match = re.search(rf"{re.escape(label)}\s*:\s*([^\s]+)", output)
        if match:
            metrics[_metric_key(label)] = _parse_scalar(match.group(1))

    phase_counts: dict[str, int] = defaultdict(int)
    for line in output.splitlines():
        phase_match = PRINTED_PHASE_RE.match(line)
        if phase_match:
            key = _metric_key(phase_match.group("label")) + "_s"
            phase_counts[key] += 1
            value = _seconds(phase_match.group("value"), phase_match.group("unit"))
            metrics[key] = value
            if phase_counts[key] > 1:
                metrics[f"{key}_{phase_counts[key]}"] = value
            continue

        numba_match = NUMBA_TIMING_RE.search(line)
        if numba_match:
            for part in numba_match.group("body").split(","):
                name, _, raw_value = part.strip().partition("=")
                if not name or not raw_value:
                    continue
                value_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ms|s)", raw_value.strip())
                if value_match:
                    metrics[f"numba_trace_{_metric_key(name)}_s"] = _seconds(
                        value_match.group(1),
                        value_match.group(2),
                    )

    return metrics


def _base_adv_rea_args(args: argparse.Namespace, backend: str) -> list[str]:
    command = [
        sys.executable,
        str(ADV_REA_SCRIPT),
        "test2_scipy_ilu_upwind",
        "-p",
        str(args.order),
        "--lc",
        str(args.lc),
        "--boundary-mode",
        args.boundary_mode,
        "--trace-ordering",
        args.trace_ordering,
        "--ilu-permc-spec",
        args.ilu_permc_spec,
        "--verbosity",
        str(args.verbosity),
        "--assembly-backend",
        backend,
    ]
    if args.plot:
        command.append("--plot")
    return command


def _run_once(args: argparse.Namespace, backend: str, repeat: int, log_dir: Path) -> RunResult:
    command = _base_adv_rea_args(args, backend)
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start
    output = completed.stdout
    log_path = log_dir / f"adv_rea_{backend}_repeat{repeat:02d}.log"
    log_path.write_text(output, encoding="utf-8")
    metrics = parse_adv_rea_output(output)
    metrics["subprocess_elapsed_s"] = elapsed
    return RunResult(
        backend=backend,
        repeat=repeat,
        returncode=completed.returncode,
        elapsed_seconds=elapsed,
        log_path=str(log_path),
        metrics=metrics,
    )


def _numeric_values(results: list[RunResult], key: str) -> list[float]:
    values = []
    for result in results:
        value = result.metrics.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _format_float(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _print_metric_table(results: list[RunResult]) -> None:
    by_backend: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        by_backend[result.backend].append(result)

    metric_order = [
        "tot_time_s",
        "setup_time_s",
        "glb_solve_time_s",
        "recons_time_s",
        "ordering_time_s",
        "ilu_time_s",
        "krylov_time_s",
        "perm_time_s",
        "assembling_local_element_matrices_s",
        "assembling_element_boundary_coupling_s",
        "inverting_local_element_matrices_s",
        "assembling_global_trace_system_s",
        "eliminating_boundary_trace_dofs_s",
        "assembling_reduced_projected_trace_system_numba_s",
        "numba_trace_kernel_s",
        "subprocess_elapsed_s",
    ]

    print()
    print("Summary from scripts/advection_reaction/run_cases.py printed timings")
    print("backend  metric                                           n  median     min        max        mean")
    print("-------  -----------------------------------------------  -  ---------  ---------  ---------  ---------")
    for backend in sorted(by_backend):
        backend_results = by_backend[backend]
        for key in metric_order:
            values = _numeric_values(backend_results, key)
            if not values:
                continue
            print(
                f"{backend:<7}  {key:<47}  {len(values):>1}  "
                f"{_format_float(statistics.median(values)):>9}  "
                f"{_format_float(min(values)):>9}  "
                f"{_format_float(max(values)):>9}  "
                f"{_format_float(statistics.fmean(values)):>9}"
            )

    if {"numpy", "numba"}.issubset(by_backend):
        print()
        print("Backend ratios from medians")
        for key in ("tot_time_s", "setup_time_s", "glb_solve_time_s", "subprocess_elapsed_s"):
            numpy_values = _numeric_values(by_backend["numpy"], key)
            numba_values = _numeric_values(by_backend["numba"], key)
            if not numpy_values or not numba_values:
                continue
            numpy_median = statistics.median(numpy_values)
            numba_median = statistics.median(numba_values)
            if numba_median == 0:
                continue
            winner = "numba" if numba_median < numpy_median else "numpy"
            ratio = max(numpy_median, numba_median) / min(numpy_median, numba_median)
            print(f"- {key}: {winner} faster by {ratio:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the organized advection-reaction preset runner repeatedly and parse its printed timings."
    )
    parser.add_argument("-p", "--order", type=int, default=6)
    parser.add_argument("--lc", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=("numpy", "numba", "both"), default="both")
    parser.add_argument("--boundary-mode", choices=("penalty", "eliminate"), default="eliminate")
    parser.add_argument("--trace-ordering", choices=("none", "upwind-scc"), default="upwind-scc")
    parser.add_argument("--ilu-permc-spec", default="NATURAL")
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="include runner --plot; useful for exact manual parity but can open/block a GUI window",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="directory for raw solver logs; defaults to run_logs/adv_rea_cli_timings/<timestamp>",
    )
    args = parser.parse_args()

    backends = ["numpy", "numba"] if args.backend == "both" else [args.backend]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_dir or (REPO_ROOT / "run_logs" / "adv_rea_cli_timings" / timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running scripts/advection_reaction/run_cases.py repeats={args.repeats} "
        f"order={args.order} lc={args.lc} "
        f"boundary_mode={args.boundary_mode} trace_ordering={args.trace_ordering} plot={args.plot}"
    )
    print(f"Raw logs: {log_dir}")

    results: list[RunResult] = []
    for repeat in range(1, args.repeats + 1):
        for backend in backends:
            print(f"[{backend} repeat {repeat}/{args.repeats}] running...", flush=True)
            result = _run_once(args, backend, repeat, log_dir)
            results.append(result)
            status = "ok" if result.returncode == 0 else f"exit={result.returncode}"
            total = result.metrics.get("tot_time_s", "?")
            setup = result.metrics.get("setup_time_s", "?")
            print(f"[{backend} repeat {repeat}/{args.repeats}] {status}; setup={setup}s total={total}s")
            if result.returncode != 0:
                print(f"  see log: {result.log_path}")

    summary_path = log_dir / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    _print_metric_table(results)
    print()
    print(f"Wrote parsed summary: {summary_path}")


if __name__ == "__main__":
    main()
