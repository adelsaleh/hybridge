#!/usr/bin/env python3
"""Inspect AMGX hierarchy statistics for GPU diffusion-reaction runs.

This diagnostic intentionally lives outside the solver stack.  It enables AMGX
`print_grid_stats` on a temporary copy of a project config, runs the GPU
Diffusion-Reaction runner, parses the AMGX hierarchy block, and stores both a
compact CSV/JSONL summary and the raw runner output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "amgx"
RUN_LOG_DIR = ROOT / "run_logs"
TMP_CONFIG_DIR = Path("/tmp/hybridge/amgx_sweeps")
RUNNER_MODULE = "scripts.gpu.run_diffusion_reaction_cuda"
DEFAULT_CONFIG = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json"

PYAMGX_SETUP_RE = re.compile(r"PyAMGX setup/upload:\s*([0-9.eE+-]+)s")
PYAMGX_SOLVE_RE = re.compile(
    r"PyAMGX solve:\s*([0-9.eE+-]+)s,\s*iterations=([0-9,]+|unknown),\s*"
    r"solver_rel_res=([0-9.eE+-]+|nan|inf|-inf),\s*physical_rel_res=([0-9.eE+-]+|nan|inf|-inf)"
)
DIRECT_CSR_RE = re.compile(r"direct CSR view setup:\s*([0-9.eE+-]+)s,\s*nnz=([0-9,]+)")
TRIANGLES_RE = re.compile(r"triangles[:=]\s*([0-9,]+)")
GLOBAL_DOF_RE = re.compile(r"global dof:\s*([0-9,]+)")
GRID_LEVELS_RE = re.compile(r"Number of Levels:\s*([0-9,]+)")
LEVEL_RE = re.compile(
    r"^\s*(\d+)\(D\)\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+"
    r"([0-9.eE+-]+|nan|inf|-inf)\s+([0-9.eE+-]+|nan|inf|-inf)\s*$"
)
GRID_COMPLEXITY_RE = re.compile(r"Grid Complexity:\s*([0-9.eE+-]+|nan|inf|-inf)")
OPERATOR_COMPLEXITY_RE = re.compile(r"Operator Complexity:\s*([0-9.eE+-]+|nan|inf|-inf)")
GRID_MEMORY_RE = re.compile(r"Total Memory Usage:\s*([0-9.eE+-]+|nan|inf|-inf)\s*GB")
TOTAL_ITERATIONS_RE = re.compile(r"Total Iterations:\s*([0-9,]+)")
AVG_RATE_RE = re.compile(r"(?:Geometric mean res/previous|Average residual ratio|Avg Convergence Rate):\s*([0-9.eE+-]+|nan|inf|-inf)")
FINAL_RESIDUAL_RE = re.compile(r"Final (?:monitored r|R)esidual:\s*([0-9.eE+-]+|nan|inf|-inf)")
TOTAL_REDUCTION_RE = re.compile(r"(?:Final residual/initial|Residual/initial|Total Reduction in Residual):\s*([0-9.eE+-]+|nan|inf|-inf)")
MAX_MEMORY_RE = re.compile(r"Maximum Memory Usage:\s*([0-9.eE+-]+|nan|inf|-inf)\s*GB")
AMGX_MEMORY_RE = re.compile(
    r"AMGX memory \(process, GiB\): current used=([0-9.eE+-]+) held=([0-9.eE+-]+); "
    r"sampled solve peak used=([0-9.eE+-]+) held=([0-9.eE+-]+)"
)
AMGX_TOTAL_TIME_RE = re.compile(r"^Total Time:\s*([0-9.eE+-]+|nan|inf|-inf)\s*$", re.MULTILINE)
AMGX_SETUP_TIME_RE = re.compile(r"^\s+setup:\s*([0-9.eE+-]+|nan|inf|-inf)\s*s\s*$", re.MULTILINE)
AMGX_SOLVE_TIME_RE = re.compile(r"^\s+solve:\s*([0-9.eE+-]+|nan|inf|-inf)\s*s\s*$", re.MULTILINE)
AMGX_ERROR_RE = re.compile(r"(?:pyamgx\.AMGXError:|Caught amgx exception:|what\(\):)\s*(.+)")


@dataclass(frozen=True)
class RunCase:
    mesh_size: float
    trace_basis: str
    name: str


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_int(raw: str | None) -> int | None:
    if raw is None or raw == "unknown":
        return None
    try:
        return int(raw.replace(",", ""))
    except ValueError:
        return None


def _csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def make_grid_stats_config(args: argparse.Namespace) -> Path:
    config = json.loads(args.amgx_config.read_text())
    solver = config.setdefault("solver", {})
    solver["solver"] = args.amgx_solver
    solver["tolerance"] = float(args.amgx_tolerance)
    solver["max_iters"] = int(args.amgx_maxiter)
    solver["monitor_residual"] = 1
    solver["print_solve_stats"] = 1
    solver["store_res_history"] = 1
    solver["obtain_timings"] = 1
    solver.setdefault("norm", "L2")
    preconditioner = solver.setdefault("preconditioner", {})
    preconditioner["print_grid_stats"] = 1
    args.generated_config_dir.mkdir(parents=True, exist_ok=True)
    stem = args.amgx_config.stem + "_grid_stats"
    path = args.generated_config_dir / f"{stem}.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n")
    return path


def runner_command(case: RunCase, config_path: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--case",
        args.case,
        "--mesh-type",
        args.mesh_type,
        "-ms",
        f"{case.mesh_size:g}",
        "-o",
        str(args.order),
        "--basis",
        args.basis,
        "--trace-basis",
        case.trace_basis,
        "--volume-quadrature",
        args.volume_quadrature,
        "--assembly-backend",
        args.assembly_backend,
        "--raw-matrix-format",
        args.raw_matrix_format,
        "--raw-block-size",
        str(args.raw_block_size),
        "--amgx-solver",
        args.amgx_solver,
        "--amgx-config",
        str(config_path),
        "--amgx-tolerance",
        str(args.amgx_tolerance),
        "--amgx-maxiter",
        str(args.amgx_maxiter),
        "-v",
        str(args.verbosity),
    ]


def env_with_amgx(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    paths = [
        "/tmp/AMGX-build-cuda13.0.1",
        "/tmp/AMGX-install-cuda13.0.1/lib",
        "/tmp/cuda-13.0.1/targets/x86_64-linux/lib",
    ]
    if existing := env.get("LD_LIBRARY_PATH"):
        paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(paths)
    env["HYBRIDGE_CUDA_AMGX_MONITOR"] = "1"
    return env


def parse_failure(stdout: str) -> str | None:
    if "out of memory" in stdout.lower():
        return "AMGX setup out of memory"
    if "CUDA kernel launch error" in stdout:
        return "AMGX setup CUDA kernel launch error"
    if match := AMGX_ERROR_RE.search(stdout):
        return match.group(1).strip()
    if "Traceback" in stdout:
        return "Python traceback"
    return None


def parse_output(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if match := PYAMGX_SETUP_RE.search(stdout):
        parsed["pyamgx_setup_seconds"] = _parse_float(match.group(1))
    if match := PYAMGX_SOLVE_RE.search(stdout):
        parsed["pyamgx_solve_seconds"] = _parse_float(match.group(1))
        parsed["pyamgx_iterations"] = _parse_int(match.group(2))
        parsed["solver_rel_residual"] = _parse_float(match.group(3))
        parsed["physical_rel_residual"] = _parse_float(match.group(4))
    if match := DIRECT_CSR_RE.search(stdout):
        parsed["csr_seconds"] = _parse_float(match.group(1))
        parsed["matrix_nnz"] = _parse_int(match.group(2))
    triangles = TRIANGLES_RE.findall(stdout)
    if triangles:
        parsed["triangles"] = _parse_int(triangles[-1])
    if match := GLOBAL_DOF_RE.search(stdout):
        parsed["global_dof"] = _parse_int(match.group(1))
    if match := GRID_LEVELS_RE.search(stdout):
        parsed["amgx_levels"] = _parse_int(match.group(1))
    levels = []
    for line in stdout.splitlines():
        if match := LEVEL_RE.match(line):
            levels.append(
                {
                    "level": _parse_int(match.group(1)),
                    "rows": _parse_int(match.group(2)),
                    "nnz": _parse_int(match.group(3)),
                    "parts": _parse_int(match.group(4)),
                    "sparsity": _parse_float(match.group(5)),
                    "memory_gb": _parse_float(match.group(6)),
                }
            )
    if levels:
        parsed["levels"] = levels
        parsed["finest_rows"] = levels[0].get("rows")
        parsed["finest_nnz"] = levels[0].get("nnz")
        parsed["coarsest_rows"] = levels[-1].get("rows")
        parsed["coarsest_nnz"] = levels[-1].get("nnz")
        if len(levels) > 1 and levels[0].get("rows"):
            parsed["first_coarsening_ratio"] = levels[1].get("rows") / levels[0].get("rows")
        if levels[0].get("rows"):
            parsed["coarsest_ratio"] = levels[-1].get("rows") / levels[0].get("rows")
    if match := AMGX_MEMORY_RE.search(stdout):
        parsed["amgx_current_used_gib"] = _parse_float(match.group(1))
        parsed["amgx_current_held_gib"] = _parse_float(match.group(2))
        parsed["amgx_peak_used_gib"] = _parse_float(match.group(3))
        parsed["amgx_peak_held_gib"] = _parse_float(match.group(4))
        # Preserve the historical aggregate key for existing sweep consumers.
        parsed["amgx_max_memory_gb"] = parsed["amgx_peak_held_gib"]
    for key, regex in (
        ("grid_complexity", GRID_COMPLEXITY_RE),
        ("operator_complexity", OPERATOR_COMPLEXITY_RE),
        ("grid_memory_gb", GRID_MEMORY_RE),
        ("amgx_total_iterations", TOTAL_ITERATIONS_RE),
        ("amgx_average_residual_ratio", AVG_RATE_RE),
        ("amgx_final_residual", FINAL_RESIDUAL_RE),
        ("amgx_residual_over_initial", TOTAL_REDUCTION_RE),
        ("amgx_max_memory_gb", MAX_MEMORY_RE),
        ("amgx_total_time_seconds", AMGX_TOTAL_TIME_RE),
        ("amgx_reported_setup_seconds", AMGX_SETUP_TIME_RE),
        ("amgx_reported_solve_seconds", AMGX_SOLVE_TIME_RE),
    ):
        if match := regex.search(stdout):
            if key in {"amgx_total_iterations"}:
                parsed[key] = _parse_int(match.group(1))
            else:
                parsed[key] = _parse_float(match.group(1))
    return parsed


def raw_output_path(case: RunCase, stamp: str, args: argparse.Namespace) -> Path:
    mesh_token = f"ms{case.mesh_size:g}".replace(".", "p")
    return args.log_dir / f"diff_rea_amgx_hierarchy_o{args.order}_{mesh_token}_{case.name}_{stamp}.out"


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mesh_token = "_".join(f"ms{size:g}".replace(".", "p") for size in _csv_floats(args.mesh_sizes))
    stem = f"diff_rea_amgx_hierarchy_o{args.order}_{mesh_token}_{stamp}"
    return args.log_dir / f"{stem}.csv", args.log_dir / f"{stem}.jsonl", stamp


def run_case(case: RunCase, config_path: Path, stamp: str, args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    cmd = runner_command(case, config_path, args)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        stdout = proc.stdout
        status = "ok" if proc.returncode == 0 else "failed"
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = exc.stdout or ""
        status = "timeout"
        returncode = -1
    raw_path = raw_output_path(case, stamp, args)
    raw_path.write_text(stdout)
    row: dict[str, Any] = {
        "status": status,
        "returncode": returncode,
        "wall_seconds": elapsed,
        "name": case.name,
        "trace_basis": case.trace_basis,
        "mesh_size": case.mesh_size,
        "order": args.order,
        "basis": args.basis,
        "config_path": str(config_path),
        "raw_output_path": str(raw_path),
        "command": " ".join(cmd),
    }
    row.update(parse_output(stdout))
    if status != "ok":
        row["failure_reason"] = parse_failure(stdout)
    return row


def write_results(rows: list[dict[str, Any]], csv_path: Path, jsonl_path: Path) -> None:
    preferred = [
        "status",
        "name",
        "trace_basis",
        "mesh_size",
        "triangles",
        "global_dof",
        "matrix_nnz",
        "amgx_levels",
        "finest_rows",
        "finest_nnz",
        "coarsest_rows",
        "coarsest_nnz",
        "first_coarsening_ratio",
        "coarsest_ratio",
        "grid_complexity",
        "operator_complexity",
        "grid_memory_gb",
        "pyamgx_setup_seconds",
        "pyamgx_solve_seconds",
        "pyamgx_iterations",
        "physical_rel_residual",
        "amgx_average_residual_ratio",
        "amgx_residual_over_initial",
        "amgx_current_used_gib",
        "amgx_current_held_gib",
        "amgx_peak_used_gib",
        "amgx_peak_held_gib",
        "amgx_max_memory_gb",
        "failure_reason",
        "raw_output_path",
        "command",
    ]
    fields = list(preferred)
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen and key != "levels":
                fields.append(key)
                seen.add(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    try:
        return format(float(value), spec)
    except Exception:
        return str(value)


def print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['status']:<8} {row['name']:<14} ms={row['mesh_size']:<6g} "
        f"levels={str(row.get('amgx_levels')):>4} coarse={str(row.get('coarsest_rows')):>8} "
        f"opC={_fmt(row.get('operator_complexity')):>7} iter={str(row.get('pyamgx_iterations')):>5} "
        f"setup={_fmt(row.get('pyamgx_setup_seconds')):>7}s solve={_fmt(row.get('pyamgx_solve_seconds')):>7}s "
        f"phys={_fmt(row.get('physical_rel_residual'), '.2e'):>9} {row.get('failure_reason') or ''}",
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="trigonometric-poisson")
    parser.add_argument("--mesh-type", default="disc", choices=("auto", "disc", "rectangle", "unit-rectangle", "triangle", "lshape", "structured-rectangle"))
    parser.add_argument("--mesh-sizes", default="0.18,0.04", help="comma-separated mesh sizes")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--trace-bases", default="legacy-lagrange,legendre-modal", help="comma-separated trace bases")
    parser.add_argument("--volume-quadrature", default="symmetric", choices=("auto", "symmetric", "duffy"))
    parser.add_argument("--assembly-backend", default="raw-cuda", choices=("cupy", "raw-cuda"))
    parser.add_argument("--raw-matrix-format", default="csr", choices=("coo", "csr"))
    parser.add_argument("--raw-block-size", type=int, default=128, choices=(1, 32, 64, 128))
    parser.add_argument("--amgx-solver", default="PCGF")
    parser.add_argument("--amgx-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--amgx-maxiter", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--generated-config-dir", type=Path, default=TMP_CONFIG_DIR)
    parser.add_argument("--log-dir", type=Path, default=RUN_LOG_DIR)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mesh_sizes = _csv_floats(args.mesh_sizes)
    trace_bases = _csv_strings(args.trace_bases)
    config_path = make_grid_stats_config(args)
    csv_path, jsonl_path, stamp = output_paths(args)
    env = env_with_amgx(args)
    cases = [RunCase(mesh_size=mesh_size, trace_basis=trace, name=trace.replace("-", "_")) for mesh_size in mesh_sizes for trace in trace_bases]
    print(
        f"Running {len(cases)} AMGX hierarchy checks: p={args.order}, mesh_sizes={args.mesh_sizes}, "
        f"basis={args.basis}, config={config_path}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"\n[{idx}/{len(cases)}] {case.trace_basis} ms={case.mesh_size:g}", flush=True)
        row = run_case(case, config_path, stamp, args, env)
        rows.append(row)
        print_row(row)
        write_results(rows, csv_path, jsonl_path)
    print(f"\nCSV:   {csv_path}", flush=True)
    print(f"JSONL: {jsonl_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
