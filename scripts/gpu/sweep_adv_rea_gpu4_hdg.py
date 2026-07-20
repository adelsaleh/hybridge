#!/usr/bin/env python3
"""Sweep standalone GPU4-style HDG advection-reaction quadrature/basis choices.

The driver intentionally runs scripts/gpu/run_adv_rea_gpu4_hdg.py in subprocesses so
AMGX, CuPy memory pools, and per-run timers are reset between cases. Launch this
with the project venv, for example:

    LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib \
      .venv/bin/python scripts/sweep_adv_rea_gpu4_hdg.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "gpu" / "run_adv_rea_gpu4_hdg.py"
DEFAULT_LOG_DIR = ROOT / "run_logs"
AMGX_LIBRARY_PATHS = ("/tmp/AMGX-build", "/tmp/AMGX-install/lib")

KEY_VALUE_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$")
FLOAT_KEYS = {
    "h",
    "h^(p+1)",
    "L2 error",
    "Linf error",
    "avg max error",
    "mesh generation (s)",
    "space/reference setup (s)",
    "GPU mirror/setup (s)",
    "advection projection (s)",
    "assembly total (s)",
    "index arrays",
    "local matrices",
    "local solve",
    "trace blocks/data",
    "RHS/source",
    "boundary elimination",
    "trace ordering (s)",
    "global solve total (s)",
    "permutation",
    "CSR+scaling",
    "AMGX setup",
    "AMGX solve",
    "reconstruct total (s)",
    "plot/error eval (s)",
    "total measured (s)",
    "this assembly / baseline",
    "this AMGX / baseline",
}
INT_KEYS = {
    "order",
    "triangles",
    "edges",
    "interior edges",
    "global dof",
    "max-error element",
    "AMGX iterations",
}


@dataclass(frozen=True)
class SweepCase:
    order: int
    mesh_size: float
    basis: str
    trace_basis: str
    quad_label: str
    quad_value: int | None


def parse_range_spec(spec: str, *, cast=int) -> list:
    values = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in {2, 3}:
                raise ValueError(f"invalid range spec {part!r}")
            start = cast(pieces[0])
            stop = cast(pieces[1])
            step = cast(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            if cast is not int:
                raise ValueError("range syntax is only supported for integer specs")
            values.extend(range(start, stop + (1 if step > 0 else -1), step))
        else:
            values.append(cast(part))
    return values


def parse_mesh_sizes(spec: str) -> list[float]:
    return parse_range_spec(spec, cast=float)


def quad_from_label(label: str, order: int) -> int | None:
    normalized = label.strip().lower()
    if normalized in {"default", "none", "hdgfem"}:
        return None
    if normalized == "p":
        value = order
    elif normalized == "p+1":
        value = order + 1
    elif normalized == "p+2":
        value = order + 2
    elif normalized == "2p-2":
        value = 2 * order - 2
    elif normalized == "2p-1":
        value = 2 * order - 1
    elif normalized == "2p":
        value = 2 * order
    elif normalized == "2p+1":
        value = 2 * order + 1
    elif normalized == "2p+2":
        value = 2 * order + 2
    elif normalized == "2p+3":
        value = 2 * order + 3
    elif normalized == "2p+4":
        value = 2 * order + 4
    else:
        value = int(normalized)
    return max(2, int(value))


def unique_preserve_order(values: Iterable) -> list:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def build_cases(args: argparse.Namespace) -> list[SweepCase]:
    cases: list[SweepCase] = []
    for order in parse_range_spec(args.orders, cast=int):
        quad_pairs = []
        for label in args.quad_rules:
            value = quad_from_label(label, order)
            resolved_label = "default" if value is None else f"q{value}({label})"
            quad_pairs.append((resolved_label, value))
        quad_pairs = unique_preserve_order(quad_pairs)
        for mesh_size in parse_mesh_sizes(args.mesh_sizes):
            for basis in args.bases:
                for trace_basis in args.trace_bases:
                    for quad_label, quad_value in quad_pairs:
                        cases.append(SweepCase(order, mesh_size, basis, trace_basis, quad_label, quad_value))
    return cases


def parse_value(label: str, value: str):
    stripped = value.strip().replace(",", "")
    if label in FLOAT_KEYS:
        try:
            return float(stripped)
        except ValueError:
            return value.strip()
    if label in INT_KEYS:
        try:
            return int(stripped)
        except ValueError:
            return value.strip()
    return value.strip()


def parse_runner_output(stdout: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in stdout.splitlines():
        match = KEY_VALUE_RE.match(line)
        if not match:
            continue
        label = match.group(1).strip()
        value = match.group(2).strip()
        parsed[label] = parse_value(label, value)
    residual = re.search(r"scaled_rel_res=([0-9.eE+-]+)", stdout)
    if residual:
        parsed["scaled_rel_res"] = float(residual.group(1))
    return parsed


def run_case(case: SweepCase, args: argparse.Namespace, env: dict[str, str]) -> dict[str, object]:
    error_quad = args.error_volume_quad_1d
    if error_quad is None:
        assembly_quad = case.quad_value if case.quad_value is not None else max(2 * case.order + 2, 2)
        error_quad = max(assembly_quad + args.error_quad_margin, 2 * case.order + args.error_quad_extra, 8)
    cmd = [
        sys.executable,
        str(args.runner),
        "--case",
        args.case,
        "-o",
        str(case.order),
        "-ms",
        f"{case.mesh_size:g}",
        "--basis",
        case.basis,
        "--trace-basis",
        case.trace_basis,
        "--error-volume-quad-1d",
        str(error_quad),
        "--plot-resolution",
        str(args.plot_resolution),
        "--amgx-solver",
        args.amgx_solver,
        "--amgx-tolerance",
        str(args.amgx_tolerance),
        "--amgx-maxiter",
        str(args.amgx_maxiter),
    ]
    if case.quad_value is not None:
        cmd.extend(["--volume-quad-1d", str(case.quad_value)])
    if args.mesh_type:
        cmd.extend(["--mesh-type", args.mesh_type])
    if args.trace_ordering != "none":
        cmd.extend(["--trace-ordering", args.trace_ordering])
    started = time.perf_counter()
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
    parsed = parse_runner_output(proc.stdout)
    row: dict[str, object] = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_wall_s": elapsed,
        "order": case.order,
        "mesh_size_requested": case.mesh_size,
        "basis_requested": case.basis,
        "trace_basis_requested": case.trace_basis,
        "quad_label": case.quad_label,
        "quad_value_requested": case.quad_value if case.quad_value is not None else "default",
        "error_quad_requested": error_quad,
        "command": " ".join(cmd),
    }
    row.update(parsed)
    if args.keep_output or proc.returncode != 0:
        row["output"] = proc.stdout
    else:
        row["output_tail"] = "\n".join(proc.stdout.splitlines()[-12:])
    return row


def summarize(rows: list[dict[str, object]]) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok" and isinstance(r.get("L2 error"), float)]
    if not ok_rows:
        print("No successful rows with parsed L2 error.")
        return
    print("\nBest by order, mesh, element basis, trace basis")
    print("order mesh     basis      trace            quad        L2          Linf        asm(s)  amgx(s) total(s)")
    grouped: dict[tuple, list[dict[str, object]]] = {}
    for row in ok_rows:
        key = (
            row.get("order"),
            row.get("mesh_size_requested"),
            row.get("basis_requested"),
            row.get("trace_basis_requested"),
        )
        grouped.setdefault(key, []).append(row)
    for key in sorted(grouped):
        best = min(grouped[key], key=lambda r: float(r["L2 error"]))
        print(
            f"{int(best['order']):>5} {float(best['mesh_size_requested']):<8.4g} "
            f"{str(best['basis_requested']):<10} {str(best['trace_basis_requested']):<16} "
            f"{str(best['quad_label']):<11} {float(best['L2 error']):.3e} "
            f"{float(best.get('Linf error', float('nan'))):.3e} "
            f"{float(best.get('assembly total (s)', float('nan'))):>6.3f} "
            f"{float(best.get('AMGX solve', float('nan'))):>7.3f} "
            f"{float(best.get('total measured (s)', float('nan'))):>7.3f}"
        )

    print("\nOverall best per order and mesh")
    print("order mesh     basis      trace            quad        L2          Linf        asm(s)  amgx(s) total(s)")
    grouped2: dict[tuple, list[dict[str, object]]] = {}
    for row in ok_rows:
        grouped2.setdefault((row.get("order"), row.get("mesh_size_requested")), []).append(row)
    for key in sorted(grouped2):
        best = min(grouped2[key], key=lambda r: float(r["L2 error"]))
        print(
            f"{int(best['order']):>5} {float(best['mesh_size_requested']):<8.4g} "
            f"{str(best['basis_requested']):<10} {str(best['trace_basis_requested']):<16} "
            f"{str(best['quad_label']):<11} {float(best['L2 error']):.3e} "
            f"{float(best.get('Linf error', float('nan'))):.3e} "
            f"{float(best.get('assembly total (s)', float('nan'))):>6.3f} "
            f"{float(best.get('AMGX solve', float('nan'))):>7.3f} "
            f"{float(best.get('total measured (s)', float('nan'))):>7.3f}"
        )


def write_results(rows: list[dict[str, object]], args: argparse.Namespace) -> tuple[Path, Path]:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.log_dir / f"adv_rea_gpu4_hdg_sweep_{stamp}.json"
    csv_path = args.log_dir / f"adv_rea_gpu4_hdg_sweep_{stamp}.csv"
    payload = {"args": vars(args) | {"runner": str(args.runner), "log_dir": str(args.log_dir)}, "rows": rows}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    fieldnames = sorted({key for row in rows for key in row.keys() if key not in {"output", "output_tail"}})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("LD_LIBRARY_PATH", "")
    pieces = list(AMGX_LIBRARY_PATHS)
    if current:
        pieces.append(current)
    env["LD_LIBRARY_PATH"] = ":".join(unique_preserve_order(pieces))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--case", default="test2_legacy_gpu3")
    parser.add_argument("--orders", default="1:10", help="comma/range spec, e.g. 1:10 or 4,6,8")
    parser.add_argument("--mesh-sizes", default="0.08", help="comma-separated mesh sizes")
    parser.add_argument("--bases", nargs="+", default=["dub_orth", "hier_C0", "bernstein"])
    parser.add_argument("--trace-bases", nargs="+", default=["legacy-lagrange", "legendre-modal"])
    parser.add_argument(
        "--quad-rules",
        nargs="+",
        default=["2p-1", "2p", "2p+1", "default", "2p+4"],
        help="quadrature candidates; supports default, p+2, 2p-1, 2p, 2p+1, 2p+2, 2p+4, or integers",
    )
    parser.add_argument("--mesh-type", default="rectangle", choices=("rectangle", "structured-rectangle"))
    parser.add_argument("--plot-resolution", type=int, default=10)
    parser.add_argument("--error-volume-quad-1d", type=int, default=None)
    parser.add_argument("--error-quad-margin", type=int, default=4)
    parser.add_argument("--error-quad-extra", type=int, default=8)
    parser.add_argument("--trace-ordering", default="none", choices=("none", "upwind-scc"))
    parser.add_argument("--amgx-solver", default="BICGSTAB")
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--amgx-maxiter", type=int, default=1500)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quick", action="store_true", help="smaller matrix of candidates for a first pass")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.quick:
        args.bases = ["dub_orth", "hier_C0"]
        args.trace_bases = ["legacy-lagrange", "legendre-modal"]
        args.quad_rules = ["2p", "default", "2p+4"]
        args.mesh_sizes = "0.10"
    args.runner = args.runner.resolve()
    if not args.runner.exists():
        raise SystemExit(f"runner not found: {args.runner}")
    cases = build_cases(args)
    if args.max_runs is not None:
        cases = cases[: args.max_runs]
    print(f"Prepared {len(cases)} sweep runs")
    for index, case in enumerate(cases, start=1):
        quad = "default" if case.quad_value is None else str(case.quad_value)
        print(
            f"[{index:03d}/{len(cases):03d}] p={case.order} ms={case.mesh_size:g} "
            f"basis={case.basis} trace={case.trace_basis} q={quad} ({case.quad_label})"
        )
    if args.dry_run:
        return 0

    env = build_env()
    rows: list[dict[str, object]] = []
    sweep_start = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        quad = "default" if case.quad_value is None else str(case.quad_value)
        print(
            f"\n[{index:03d}/{len(cases):03d}] running p={case.order} ms={case.mesh_size:g} "
            f"basis={case.basis} trace={case.trace_basis} q={quad}",
            flush=True,
        )
        try:
            row = run_case(case, args, env)
        except subprocess.TimeoutExpired as exc:
            row = {
                "status": "timeout",
                "returncode": None,
                "elapsed_wall_s": args.timeout,
                "order": case.order,
                "mesh_size_requested": case.mesh_size,
                "basis_requested": case.basis,
                "trace_basis_requested": case.trace_basis,
                "quad_label": case.quad_label,
                "quad_value_requested": case.quad_value if case.quad_value is not None else "default",
                "output": exc.stdout or "",
            }
        rows.append(row)
        if row.get("status") == "ok":
            print(
                "  ok: "
                f"L2={float(row.get('L2 error', float('nan'))):.3e}, "
                f"Linf={float(row.get('Linf error', float('nan'))):.3e}, "
                f"asm={float(row.get('assembly total (s)', float('nan'))):.3f}s, "
                f"amgx={float(row.get('AMGX solve', float('nan'))):.3f}s, "
                f"total={float(row.get('total measured (s)', float('nan'))):.3f}s",
                flush=True,
            )
        else:
            print(f"  {row.get('status')}: returncode={row.get('returncode')}", flush=True)
            if args.fail_fast:
                break
    total = time.perf_counter() - sweep_start
    print(f"\nSweep wall time: {total:.2f}s")
    summarize(rows)
    json_path, csv_path = write_results(rows, args)
    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote CSV : {csv_path}")
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
