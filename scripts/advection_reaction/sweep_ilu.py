#!/usr/bin/env python3
"""Sweep SciPy ILU parameters for manufactured advection-reaction presets."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.advection_reaction.presets import DEFAULT_PRESET, PRESETS

_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PATTERNS = {
    "l2_error": re.compile(r"L2 error:\s*(%s)" % _FLOAT_RE),
    "linf_error": re.compile(r"Linf error:\s*(%s)" % _FLOAT_RE),
    "iterations": re.compile(r"Krylov iterations:\s*([0-9,]+)"),
    "solver_rel_res": re.compile(r"solver rel res:\s*(%s)" % _FLOAT_RE),
    "free_trace_rel_res": re.compile(r"free-trace rel res:\s*(%s)" % _FLOAT_RE),
    "total_seconds": re.compile(r"total \(s\):\s*(%s)" % _FLOAT_RE),
    "precond_seconds": re.compile(r"precond build \(s\):\s*(%s)" % _FLOAT_RE),
    "krylov_seconds": re.compile(r"Krylov solve \(s\):\s*(%s)" % _FLOAT_RE),
}


def _csv_floats(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_summary(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, pattern in _PATTERNS.items():
        match = pattern.search(output)
        if match:
            values[key] = match.group(1).replace(",", "")
    return values


def _build_command(args, *, fill: float, drop_tol: float, ordering: str, scaling: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.advection_reaction.run_cases",
        args.preset,
        "--ilu-fill-factor",
        str(fill),
        "--ilu-drop-tol",
        str(drop_tol),
        "--trace-ordering",
        ordering,
        "--scale-system",
        scaling,
        "--maxiter",
        str(args.maxiter),
        "--solver-rtol",
        str(args.solver_rtol),
        "--ilu-permc-spec",
        args.ilu_permc_spec,
    ]
    if args.mesh_size is not None:
        cmd.extend(["--mesh-size", str(args.mesh_size)])
    if args.order is not None:
        cmd.extend(["--order", str(args.order)])
    if args.assembly_backend is not None:
        cmd.extend(["--assembly-backend", args.assembly_backend])
    if args.volume_quadrature is not None:
        cmd.extend(["--volume-quadrature", args.volume_quadrature])
    if args.quiet:
        cmd.append("--quiet")
    return cmd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", default=DEFAULT_PRESET, choices=tuple(sorted(PRESETS)))
    parser.add_argument("--fills", default="35,20,10,5,3,2,1.75", help="comma-separated ILU fill factors")
    parser.add_argument("--drop-tols", default="1e-10,1e-8,1e-6", help="comma-separated ILU drop tolerances")
    parser.add_argument("--orderings", default="upwind-scc,none", help="comma-separated trace orderings")
    parser.add_argument("--scalings", default="off,on", help="comma-separated scale-system modes: auto,on,off")
    parser.add_argument("--mesh-size", type=float, default=None, help="override runner mesh size")
    parser.add_argument("--order", type=int, default=None, help="override DG polynomial order")
    parser.add_argument("--assembly-backend", choices=("numpy", "numba", "cupy", "raw-cuda", "auto"), default=None)
    parser.add_argument("--volume-quadrature", choices=("auto", "symmetric", "duffy"), default=None)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--solver-rtol", type=float, default=1.0e-13)
    parser.add_argument("--ilu-permc-spec", choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"), default="COLAMD")
    parser.add_argument("--timeout", type=float, default=900.0, help="per-run timeout in seconds")
    parser.add_argument("--output-csv", type=Path, default=None, help="optional CSV output path")
    parser.add_argument("--dry-run", action="store_true", help="print generated commands without running them")
    parser.add_argument("--quiet", action="store_true", help="pass --quiet to the runner")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    fills = _csv_floats(args.fills)
    drop_tols = _csv_floats(args.drop_tols)
    orderings = _csv_strings(args.orderings)
    scalings = _csv_strings(args.scalings)
    invalid_scalings = sorted(set(scalings) - {"auto", "on", "off"})
    if invalid_scalings:
        raise SystemExit(f"invalid scale-system modes: {', '.join(invalid_scalings)}")

    rows: list[dict[str, str]] = []
    for fill in fills:
        for drop_tol in drop_tols:
            for ordering in orderings:
                for scaling in scalings:
                    cmd = _build_command(args, fill=fill, drop_tol=drop_tol, ordering=ordering, scaling=scaling)
                    label = f"fill={fill:g} drop={drop_tol:g} ordering={ordering} scaling={scaling}"
                    print(label, flush=True)
                    print("  " + " ".join(cmd), flush=True)
                    row = {
                        "fill_factor": f"{fill:g}",
                        "drop_tol": f"{drop_tol:g}",
                        "trace_ordering": ordering,
                        "scale_system": scaling,
                        "returncode": "",
                    }
                    if args.dry_run:
                        row["returncode"] = "dry-run"
                        rows.append(row)
                        continue
                    try:
                        proc = subprocess.run(
                            cmd,
                            cwd=Path.cwd(),
                            env=os.environ.copy(),
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            timeout=float(args.timeout),
                        )
                    except subprocess.TimeoutExpired as exc:
                        row["returncode"] = "timeout"
                        row["error_tail"] = "timeout"
                        print("  timeout", flush=True)
                    else:
                        row["returncode"] = str(proc.returncode)
                        row.update(_parse_summary(proc.stdout))
                        if proc.returncode != 0:
                            row["error_tail"] = " | ".join(proc.stdout.splitlines()[-6:])
                        print(
                            "  rc={returncode} iter={iterations} l2={l2_error} rel={solver_rel_res} total={total_seconds}".format(
                                returncode=row.get("returncode", ""),
                                iterations=row.get("iterations", ""),
                                l2_error=row.get("l2_error", ""),
                                solver_rel_res=row.get("solver_rel_res", ""),
                                total_seconds=row.get("total_seconds", ""),
                            ),
                            flush=True,
                        )
                    rows.append(row)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "fill_factor",
            "drop_tol",
            "trace_ordering",
            "scale_system",
            "returncode",
            "iterations",
            "solver_rel_res",
            "free_trace_rel_res",
            "l2_error",
            "linf_error",
            "precond_seconds",
            "krylov_seconds",
            "total_seconds",
            "error_tail",
        ]
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
