#!/usr/bin/env python3
"""Sweep standalone GPU4-style HDG quadrature/basis choices.

The driver runs a selected GPU4 runner in subprocesses so AMGX, CuPy memory
pools, and per-run timers are reset between cases. It defaults to the
advection-reaction runner, but ``--runner scripts/gpu/run_diff_rea_gpu4_hdg.py``
now selects diffusion-reaction compatible arguments. Launch this with the
project venv, for example:

    LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib \
      .venv/bin/python -m scripts.gpu.sweep_adv_rea_gpu4_hdg --quick
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = ROOT / "scripts" / "gpu" / "run_adv_rea_gpu4_hdg.py"
DEFAULT_LOG_DIR = ROOT / "run_logs"
AMGX_LIBRARY_PATHS = ("/tmp/AMGX-build", "/tmp/AMGX-install/lib")
DEFAULT_CASES = {"adv-rea": "test2_legacy_gpu3", "diff-rea": "trigonometric-poisson"}
ADV_MESH_TYPES = {"rectangle", "structured-rectangle"}

KEY_VALUE_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
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
    "element dof",
    "edge dof",
    "max-error element",
    "AMGX iterations",
    "iterations",
    "maxiter",
}

FLOAT_KEYS.update({
    "mesh setup",
    "space setup",
    "problem setup",
    "solver call",
    "package solve total",
    "solver internal total",
    "setup subtotal",
    "assembly",
    "assembly total",
    "global solve",
    "global solve phase",
    "global solve total",
    "CSR/view",
    "AMGX setup",
    "AMGX setup (s)",
    "AMGX solve",
    "AMGX iterate (s)",
    "AMGX subtotal",
    "AMGX call total (s)",
    "AMGX overhead (s)",
    "reconstruct",
    "reconstruct total",
    "local recon solve",
    "plot/error",
    "error eval",
    "total measured",
    "raw map/setup",
    "raw zero",
    "raw kernel",
    "raw total",
    "local LHS",
    "CuPy local LHS",
    "CuPy local solve",
    "row scaling (s)",
    "AMGX CSR build (s)",
    "finite check (s)",
    "scaled residual check (s)",
    "physical residual check (s)",
    "solve validation total (s)",
    "global solve overhead (s)",
    "solve unaccounted (s)",
    "backend assembly total (s)",
    "raw kernel (s)",
    "raw CSR kernel (s)",
    "scaled residual",
    "scaled rel residual",
    "L2",
    "Linf",
    "avg max",
    "tol",
    "check rtol",
})

INT_KEYS.update({
    "matrix nnz",
    "max element",
    "volume quad pts",
    "volume quad 1d",
    "edge quad 1d",
    "raw block",
    "verbosity",
})


@dataclass(frozen=True)
class SweepCase:
    order: int
    mesh_size: float
    basis: str
    trace_basis: str
    quad_label: str
    quad_value: int | None
    amgx_config: Path | None


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


def resolve_amgx_configs(values: list[str] | None) -> list[Path | None]:
    if not values:
        return [None]
    configs: list[Path | None] = []
    for raw in values:
        label = str(raw).strip()
        if label.lower() in {"default", "none", "embedded"}:
            configs.append(None)
            continue
        path = Path(label).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        configs.append(path.resolve())
    return unique_preserve_order(configs)


def amgx_config_label(path: Path | None) -> str:
    return "default" if path is None else path.stem


def infer_runner_kind(runner: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    name = runner.name.lower()
    if "diff_rea" in name or "diffusion" in name:
        return "diff-rea"
    return "adv-rea"


def sweep_output_stem(runner_kind: str) -> str:
    if runner_kind == "diff-rea":
        return "diff_rea_gpu4_hdg_sweep"
    return "adv_rea_gpu4_hdg_sweep"


def effective_raw_matrix_format(args: argparse.Namespace) -> str:
    if args.runner_kind == "diff-rea" and args.raw_matrix_format == "auto":
        return "csr" if args.assembly_backend == "raw-cuda" else "coo"
    return args.raw_matrix_format


def validate_runner_options(args: argparse.Namespace) -> None:
    if args.runner_kind == "adv-rea":
        if args.mesh_type is not None and args.mesh_type not in ADV_MESH_TYPES:
            allowed = ", ".join(sorted(ADV_MESH_TYPES))
            raise SystemExit(f"advection runner supports --mesh-type in {{{allowed}}}; got {args.mesh_type!r}")
        return
    if args.trace_ordering != "none":
        raise SystemExit("diffusion runner does not support --trace-ordering")
    if args.raw_local_assembly != "precomputed":
        raise SystemExit("diffusion runner does not support --raw-local-assembly")
    if args.raw_lu_mode != "safe":
        raise SystemExit("diffusion runner does not support --raw-lu-mode")


def build_cases(args: argparse.Namespace) -> list[SweepCase]:
    cases: list[SweepCase] = []
    amgx_configs = resolve_amgx_configs(args.amgx_configs)
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
                        for amgx_config in amgx_configs:
                            cases.append(SweepCase(order, mesh_size, basis, trace_basis, quad_label, quad_value, amgx_config))
    return cases


def _first_number(value: str) -> str | None:
    match = NUMBER_RE.search(value.replace(",", ""))
    return None if match is None else match.group(0)


def parse_value(label: str, value: str):
    if label in FLOAT_KEYS:
        number = _first_number(value)
        if number is None:
            return value.strip()
        try:
            return float(number)
        except ValueError:
            return value.strip()
    if label in INT_KEYS:
        number = _first_number(value)
        if number is None:
            return value.strip()
        try:
            return int(float(number))
        except ValueError:
            return value.strip()
    return value.strip()


def _parse_timing_table_cells(parsed: dict[str, object], cells: list[str]) -> bool:
    if len(cells) != 3:
        return False
    label, seconds, percent = cells
    if label in {"Timing", "Seconds"} or not (percent.endswith("%") or percent == "n/a"):
        return False
    number = _first_number(seconds)
    if number is None:
        return False
    parsed[label] = parse_value(label, seconds)
    return True


def parse_runner_output(stdout: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in stdout.splitlines():
        cells = [cell.strip() for cell in re.split(r"\s{2,}", line.strip()) if cell.strip()]
        if _parse_timing_table_cells(parsed, cells):
            continue
        matched_cell = False
        for cell in cells if cells else [line.strip()]:
            match = KEY_VALUE_RE.match(cell)
            if not match:
                continue
            label = match.group(1).strip()
            value = match.group(2).strip()
            parsed[label] = parse_value(label, value)
            matched_cell = True
        if matched_cell:
            continue
    residual = re.search(r"scaled_rel_res=([0-9.eE+-]+)", stdout)
    if residual:
        parsed["scaled_rel_res"] = float(residual.group(1))
    return parsed


def run_case(case: SweepCase, args: argparse.Namespace, env: dict[str, str]) -> dict[str, object]:
    error_quad = args.error_volume_quad_1d
    if error_quad is None:
        assembly_quad = case.quad_value if case.quad_value is not None else max(2 * case.order + 2, 2)
        error_quad = max(assembly_quad + args.error_quad_margin, 2 * case.order + args.error_quad_extra, 8)
    raw_matrix_format = effective_raw_matrix_format(args)
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
        "--assembly-backend",
        args.assembly_backend,
        "--raw-matrix-format",
        raw_matrix_format,
        "--raw-block-size",
        str(args.raw_block_size),
        "--verbosity",
        str(args.verbosity),
    ]
    if args.runner_kind == "adv-rea":
        if args.check_rtol is not None:
            cmd.extend(["--check-rtol", str(args.check_rtol)])
        cmd.extend(["--raw-local-assembly", args.raw_local_assembly])
        cmd.extend(["--raw-lu-mode", args.raw_lu_mode])
        if args.trace_ordering != "none":
            cmd.extend(["--trace-ordering", args.trace_ordering])
    if args.amgx_tolerance is not None:
        cmd.extend(["--amgx-tolerance", str(args.amgx_tolerance)])
    if args.amgx_maxiter is not None:
        cmd.extend(["--amgx-maxiter", str(args.amgx_maxiter)])
    if args.amgx_solver is not None:
        cmd.extend(["--amgx-solver", args.amgx_solver])
    if case.amgx_config is not None:
        cmd.extend(["--amgx-config", str(case.amgx_config)])
    if case.quad_value is not None:
        cmd.extend(["--volume-quad-1d", str(case.quad_value)])
    if args.mesh_type:
        cmd.extend(["--mesh-type", args.mesh_type])
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
        "runner_kind": args.runner_kind,
        "assembly_backend_requested": args.assembly_backend,
        "raw_matrix_format_requested": args.raw_matrix_format,
        "raw_matrix_format_effective": raw_matrix_format,
        "amgx_config_requested": amgx_config_label(case.amgx_config),
        "amgx_config_path": "" if case.amgx_config is None else str(case.amgx_config),
        "command": " ".join(cmd),
    }
    row.update(parsed)
    if args.keep_output or proc.returncode != 0:
        row["output"] = proc.stdout
    else:
        row["output_tail"] = "\n".join(proc.stdout.splitlines()[-12:])
    return row


def row_float(row: dict[str, object], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float("nan")


def row_int(row: dict[str, object], *keys: str) -> int:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return -1


def summarize_metric(rows: list[dict[str, object]], metric: str, title: str) -> None:
    grouped: dict[tuple, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                row.get("order"),
                row.get("mesh_size_requested"),
                row.get("basis_requested"),
                row.get("trace_basis_requested"),
                row.get("quad_label"),
            ),
            [],
        ).append(row)
    print(f"\n{title}")
    print("order mesh     config                         basis      trace            quad        iter  amgx(s) solve(s) total(s) L2")
    for key in sorted(grouped):
        candidates = [row for row in grouped[key] if row_float(row, metric) == row_float(row, metric)]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row_float(row, metric))
        print(
            f"{int(best['order']):>5} {float(best['mesh_size_requested']):<8.4g} "
            f"{str(best.get('amgx_config_requested', 'default')):<30.30} "
            f"{str(best['basis_requested']):<10} {str(best['trace_basis_requested']):<16} "
            f"{str(best['quad_label']):<11} {row_int(best, 'iterations', 'AMGX iterations'):>5} "
            f"{row_float(best, 'AMGX iterate (s)', 'AMGX solve'):>7.3f} "
            f"{row_float(best, 'global solve phase', 'global solve total (s)', 'global solve'):>8.3f} "
            f"{row_float(best, 'total measured', 'total measured (s)'):>8.3f} "
            f"{row_float(best, 'L2 error', 'L2'):.3e}"
        )


def summarize(rows: list[dict[str, object]]) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok" and row_float(r, "L2 error", "L2") == row_float(r, "L2 error", "L2")]
    if not ok_rows:
        print("No successful rows with parsed L2 error.")
        return
    summarize_metric(ok_rows, "global solve phase", "Fastest by global solve phase")
    summarize_metric(ok_rows, "total measured", "Fastest by total measured time")


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, list):
            out[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            out[key] = value
    return out


def write_results(rows: list[dict[str, object]], args: argparse.Namespace) -> tuple[Path, Path]:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = sweep_output_stem(args.runner_kind)
    json_path = args.log_dir / f"{stem}_{stamp}.json"
    csv_path = args.log_dir / f"{stem}_{stamp}.csv"
    payload = {"args": _jsonable_args(args) | {"runner": str(args.runner), "log_dir": str(args.log_dir)}, "rows": rows}
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
    parser.add_argument("--runner-kind", choices=("auto", "adv-rea", "diff-rea"), default="auto")
    parser.add_argument("--case", default=None, help="case key; defaults depend on --runner-kind")
    parser.add_argument("--orders", default="1:10", help="comma/range spec, e.g. 1:10 or 4,6,8")
    parser.add_argument("--mesh-sizes", default="0.08", help="comma-separated mesh sizes")
    parser.add_argument("--bases", nargs="+", default=["dub_orth", "hier_C0", "bernstein"])
    parser.add_argument("--trace-bases", nargs="+", default=None, help="trace bases to sweep; defaults depend on --runner-kind")
    parser.add_argument(
        "--quad-rules",
        nargs="+",
        default=["2p-1", "2p", "2p+1", "default", "2p+4"],
        help="quadrature candidates; supports default, p+2, 2p-1, 2p, 2p+1, 2p+2, 2p+4, or integers",
    )
    parser.add_argument("--mesh-type", default=None, choices=("auto", "disc", "rectangle", "unit-rectangle", "triangle", "lshape", "structured-rectangle"))
    parser.add_argument("--plot-resolution", type=int, default=10)
    parser.add_argument("--error-volume-quad-1d", type=int, default=None)
    parser.add_argument("--error-quad-margin", type=int, default=4)
    parser.add_argument("--error-quad-extra", type=int, default=8)
    parser.add_argument("--trace-ordering", default="none", choices=("none", "upwind-scc"))
    parser.add_argument("--amgx-solver", default=None, help="override the solver named in each AMGX config")
    parser.add_argument("--amgx-configs", nargs="+", default=None, help="AMGX config JSONs to sweep; use 'default' for the runner default")
    parser.add_argument("--amgx-tolerance", type=float, default=None, help="override runner AMGX tolerance")
    parser.add_argument("--check-rtol", type=float, default=None, help="advection runner residual check tolerance")
    parser.add_argument("--amgx-maxiter", type=int, default=None, help="override runner AMGX max iterations")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="raw-cuda")
    parser.add_argument("--raw-local-assembly", choices=("precomputed", "fused"), default="precomputed")
    parser.add_argument("--raw-lu-mode", choices=("safe", "coop"), default="safe")
    parser.add_argument("--raw-matrix-format", choices=("auto", "coo", "csr"), default="auto")
    parser.add_argument("--raw-block-size", type=int, choices=(1, 32, 64, 128), default=32)
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=0)
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
    args.runner = args.runner.resolve()
    if not args.runner.exists():
        raise SystemExit(f"runner not found: {args.runner}")
    args.runner_kind = infer_runner_kind(args.runner, args.runner_kind)
    if args.case is None:
        args.case = DEFAULT_CASES[args.runner_kind]
    if args.trace_bases is None:
        args.trace_bases = ["legacy-lagrange", "legendre-modal"] if args.runner_kind == "adv-rea" else ["legacy-lagrange"]
    if args.quick:
        args.bases = ["dub_orth", "hier_C0"]
        args.trace_bases = ["legacy-lagrange", "legendre-modal"] if args.runner_kind == "adv-rea" else ["legacy-lagrange"]
        args.quad_rules = ["2p", "default", "2p+4"]
        args.mesh_sizes = "0.10"
    validate_runner_options(args)
    cases = build_cases(args)
    if args.max_runs is not None:
        cases = cases[: args.max_runs]
    print(f"Prepared {len(cases)} sweep runs")
    for index, case in enumerate(cases, start=1):
        quad = "default" if case.quad_value is None else str(case.quad_value)
        print(
            f"[{index:03d}/{len(cases):03d}] p={case.order} ms={case.mesh_size:g} "
            f"basis={case.basis} trace={case.trace_basis} q={quad} ({case.quad_label}) "
            f"amgx={amgx_config_label(case.amgx_config)}"
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
            f"basis={case.basis} trace={case.trace_basis} q={quad} "
            f"amgx={amgx_config_label(case.amgx_config)}",
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
                "runner_kind": args.runner_kind,
                "assembly_backend_requested": args.assembly_backend,
                "raw_matrix_format_requested": args.raw_matrix_format,
                "raw_matrix_format_effective": effective_raw_matrix_format(args),
                "amgx_config_requested": amgx_config_label(case.amgx_config),
                "amgx_config_path": "" if case.amgx_config is None else str(case.amgx_config),
                "output": exc.stdout or "",
            }
        rows.append(row)
        if row.get("status") == "ok":
            print(
                "  ok: "
                f"L2={row_float(row, 'L2 error', 'L2'):.3e}, "
                f"Linf={row_float(row, 'Linf error', 'Linf'):.3e}, "
                f"asm={row_float(row, 'assembly total', 'assembly total (s)', 'assembly'):.3f}s, "
                f"amgx={row_float(row, 'AMGX iterate (s)', 'AMGX solve'):.3f}s, "
                f"solve={row_float(row, 'global solve phase', 'global solve total (s)', 'global solve'):.3f}s, "
                f"total={row_float(row, 'total measured', 'total measured (s)'):.3f}s",
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
