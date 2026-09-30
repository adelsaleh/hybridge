#!/usr/bin/env python3
"""Capture unprofiled timings and Nsight reports for raw assembly variants.

Run as an ordinary user with GPU counter access enabled. This script never
changes driver permissions, builds AMGX, runs a global solve, or time-steps.
"""
from __future__ import annotations

import argparse
import itertools
import json
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "scripts/gpu/benchmark_fused_raw_assembly_kernels.py"


def configurations():
    """Enumerate implemented emission/cache combinations without silent fallback."""
    for fmt, policy in itertools.product(("coo", "csr", "bsr"), ("none", "schur-lu")):
        yield "poisson", fmt, policy, "assembly", "fused"
        if fmt != "coo":
            yield "poisson", fmt, policy, "rhs", "fused"
            yield "poisson", fmt, policy, "reconstruction", "fused"
    for fmt in ("coo", "csr", "bsr"):
        yield "transport", fmt, "none", "assembly", "fused"
    yield "transport", "bsr", "none", "assembly", "split3"
    for fmt, phase in itertools.product(("csr", "bsr"), ("cache", "rhs", "reconstruction")):
        yield "poisson", fmt, "schur-cholesky", phase, "fused"
    for fmt in ("coo", "csr", "bsr"):
        yield "adr", fmt, "none", "assembly", "fused"


def main():
    """Save each exact command, output, and exit status alongside profiler artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ncu", default="/usr/local/cuda-13.0/bin/ncu")
    parser.add_argument("--mode", choices=("baseline", "profile", "both"), default="both")
    parser.add_argument("--cases", nargs="+", choices=("poisson", "transport", "adr"), default=["poisson", "transport"])
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 6])
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 128])
    parser.add_argument("--bases", nargs="+", choices=("legacy-lagrange", "legendre-modal"), default=["legacy-lagrange", "legendre-modal"])
    parser.add_argument("--blocks", nargs="+", type=int, choices=(32, 64, 128), default=[128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--coefficient-cases", nargs="+", choices=("constant", "variable"), default=["constant"])
    parser.add_argument("--phases", nargs="+", choices=("assembly", "cache", "rhs", "reconstruction"), default=["assembly", "cache", "rhs", "reconstruction"])
    parser.add_argument("--formats", nargs="+", choices=("coo", "csr", "bsr"), default=["coo", "csr", "bsr"])
    parser.add_argument("--cache-policies", nargs="+", choices=("none", "schur-lu", "schur-cholesky"), default=["none", "schur-lu", "schur-cholesky"])
    args = parser.parse_args()
    max_order = 9 if set(args.cases) == {"transport"} else 6
    if any(order < 1 or order > max_order for order in args.orders) or any(nx < 1 for nx in args.sizes) or args.repeats < 1:
        parser.error(f"require orders in 1..{max_order}, positive sizes, and repeats >= 1; use --cases transport for p=7..9")
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [BENCHMARK, Path(__file__), *sorted((ROOT / "hdgfem/backends").glob("*diffusion*cuda.py")),
                    ROOT / "hdgfem/mixed/cupy.py", ROOT / "hdgfem/transport/raw_cuda.py",
                    ROOT / "hdgfem/transport/tsle_bsr.py",
                    ROOT / "hdgfem/mixed/raw_cuda/tensor.py",
                    ROOT / "hdgfem/hdg/cuda/raw_source.py"]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    manifest = {"scope": "Raw Poisson none/Schur-LU assembly, RHS and reconstruction; separate CuPy compact Schur-Cholesky construction/reuse; fused transport COO/CSR/BSR; transport TSLE BSR; cooperative tensor-ready ADR COO/CSR/BSR",
                "gaps": ["hardware-counter access required for shared-memory/stall attribution", "representative capture is not exhaustive numerical qualification"],
                "git_head": head.stdout.strip(), "source_sha256": source_hashes, "runs": []}
    manifest_path = args.output / f"manifest-{args.mode}.json"
    if manifest_path.exists():
        parser.error(f"{manifest_path} already exists; choose a new output directory to preserve its evidence")
    for order, nx, basis, block, coefficient_case, config in itertools.product(args.orders, args.sizes, args.bases, args.blocks, args.coefficient_cases, configurations()):
        case, fmt, policy, phase, local = config
        if case not in args.cases or phase not in args.phases or fmt not in args.formats or policy not in args.cache_policies:
            continue
        name = f"{case}-{fmt}-{policy}-{phase}-{local}-p{order}-n{nx}-{basis}-b{block}"
        if coefficient_case != "constant":
            name += f"-{coefficient_case}"
        common = [sys.executable, str(BENCHMARK), "--case", case, "--matrix-format", fmt,
                  "--cache-policy", policy, "--phase", phase, "--local-assembly", local,
                  "--order", str(order), "--nx", str(nx), "--trace-basis", basis,
                  "--block-size", str(block), "--warmup", "2", "--coefficient-case", coefficient_case]
        commands = []
        if args.mode in ("baseline", "both"):
            commands.append(("baseline", common + ["--repeats", str(args.repeats)]))
        if args.mode in ("profile", "both"):
            commands.append(("profile", [args.ncu, "--target-processes", "all", "--nvtx",
                "--nvtx-include", "hdgfem_raw_assembly/",
                "--section", "SpeedOfLight", "--section", "ComputeWorkloadAnalysis",
                "--section", "MemoryWorkloadAnalysis_Tables", "--section", "LaunchStats",
                "--section", "Occupancy", "--section", "SchedulerStats",
                "--section", "WarpStateStats", "--section", "SourceCounters",
                "--section", "InstructionStats", "--force-overwrite",
                "--export", str((args.output / name).resolve())] + common + ["--repeats", "1"]))
        for mode, command in commands:
            if (args.output / f"{name}.{mode}.stdout").exists():
                parser.error(f"capture for {name} ({mode}) already exists; choose a new output directory")
            print(f"{mode}: {name}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            (args.output / f"{name}.{mode}.stdout").write_text(completed.stdout)
            (args.output / f"{name}.{mode}.stderr").write_text(completed.stderr)
            manifest["runs"].append({"name": name, "mode": mode, "command": command, "returncode": completed.returncode})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            if completed.returncode:
                print(completed.stderr + completed.stdout, file=sys.stderr)
                return completed.returncode
            if mode == "profile":
                report = args.output / f"{name}.ncu-rep"
                exported = subprocess.run([args.ncu, "--import", str(report), "--page", "raw", "--csv"], text=True, capture_output=True)
                (args.output / f"{name}.csv").write_text(exported.stdout)
                (args.output / f"{name}.export.stderr").write_text(exported.stderr)
                manifest["runs"][-1]["export_returncode"] = exported.returncode
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                if exported.returncode:
                    return exported.returncode
    if not manifest["runs"]:
        parser.error("the selected filters contain no implemented configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
