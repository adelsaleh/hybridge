#!/usr/bin/env python3
"""Profile one Python command under Nsight Systems and keep the GPU kernel summary.

    nsys_kernel_summary.py --output-dir DIR -- SCRIPT ARGS...

Runs ``nsys profile --trace=cuda`` on ``python -u -B SCRIPT ARGS`` (this interpreter), then writes
``DIR/kernels.csv`` (``nsys stats --report cuda_gpu_kern_sum``) and prints the kernels by total time,
so library kernel names (for example cuBLAS/cuTENSOR FP64 tensor-core ``dmma`` kernels) are on record.
The .nsys-rep and .sqlite stay in DIR. ``HYBRIDGE_NSYS`` selects the nsys binary.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_NSYS = "/opt/nvidia/nsight-systems/2026.1.3/bin/nsys"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing SCRIPT ARGS after --")
    nsys = os.environ.get("HYBRIDGE_NSYS", DEFAULT_NSYS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "profile"
    profiled = subprocess.run([nsys, "profile", "--trace=cuda", "--force-overwrite=true", "-o", str(report),
                               sys.executable, "-u", "-B", *command])
    print(f"profiled command exited {profiled.returncode}", flush=True)
    stats = subprocess.run([nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--force-export=true",
                            f"{report}.nsys-rep"], capture_output=True, text=True)
    text = stats.stdout[stats.stdout.find("Time (%)"):] if "Time (%)" in stats.stdout else stats.stdout
    (args.output_dir / "kernels.csv").write_text(text)
    rows = list(csv.DictReader(io.StringIO(text)))
    for row in rows[:args.top]:
        print(f"{float(row.get('Time (%)', 0)):6.2f}%  {float(row.get('Total Time (ns)', 0)) / 1e6:10.3f} ms  "
              f"x{row.get('Instances', '?'):>6}  {row.get('Name', '?')[:200]}", flush=True)
    if stats.returncode:
        print(stats.stderr[-2000:], file=sys.stderr)
    return profiled.returncode or stats.returncode


if __name__ == "__main__":
    raise SystemExit(main())
