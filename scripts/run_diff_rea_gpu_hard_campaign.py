#!/usr/bin/env python3
"""Orchestrate hard-case validation, tuning, and profiling on one CUDA GPU.

Campaign levels
---------------
``smoke``
    Two difficult cases at p=4, all six preconditioner families, two degrees.
``validation``
    Every manufactured case at p=4,5,6 and tight 1e-12 tolerance.
``full``
    Validation plus explicit matvec/ASM/BJ kernel sweeps, polynomial-degree
    sweeps with repeated timings, the component-level CUDA profiler, and a
    consolidated winner/bottleneck analysis.

Optional Nsight Systems and Nsight Compute captures target one p=6 anisotropic
ASM-polynomial solve after ordinary warmup.  They are opt-in because profiling
can substantially extend allocation time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Iterable


@dataclass(frozen=True)
class Command:
    name: str
    argv: tuple[str, ...]
    expected_outputs: tuple[str, ...] = ()

    @property
    def shell(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class CommandResult:
    name: str
    argv: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    log_file: str
    expected_outputs: tuple[str, ...]
    outputs_present: tuple[bool, ...]


def _validation_outputs(prefix: Path) -> tuple[str, ...]:
    return (
        str(prefix.with_name(prefix.name + "_raw.csv")),
        str(prefix.with_name(prefix.name + "_summary.csv")),
        str(prefix.with_suffix(".json")),
    )


def _validator(python: str, prefix: Path, *arguments: str) -> Command:
    return Command(
        prefix.name,
        (
            python,
            "scripts/validate_diff_rea_gpu_hard_cases.py",
            *arguments,
            "--output-prefix",
            str(prefix),
        ),
        _validation_outputs(prefix),
    )


def _analysis_command(
    python: str,
    output_dir: Path,
    *,
    require_full: bool,
) -> Command:
    prefix = output_dir / "final_analysis"
    arguments = [
        python,
        "scripts/analyze_diff_rea_gpu_hard_results.py",
        "--results-dir",
        str(output_dir),
        "--output-prefix",
        str(prefix),
    ]
    if require_full:
        arguments.append("--require-full")
    return Command(
        "final_analysis",
        tuple(arguments),
        (
            str(prefix.with_suffix(".json")),
            str(prefix.with_suffix(".md")),
            str(prefix.with_name(prefix.name + "_best_configurations.csv")),
            str(prefix.with_name(prefix.name + "_robust_rankings.csv")),
            str(prefix.with_name(prefix.name + "_kernel_winners.csv")),
            str(prefix.with_name(prefix.name + "_polynomial_winners.csv")),
            str(prefix.with_name(prefix.name + "_profile_bottlenecks.csv")),
            str(prefix.with_name(prefix.name + "_discretization_errors.csv")),
        ),
    )


def build_commands(
    level: str,
    *,
    output_dir: Path,
    python: str,
    skip_tests: bool,
    with_nsys: bool,
    with_ncu: bool,
) -> list[Command]:
    if level not in {"smoke", "validation", "full"}:
        raise ValueError("level must be smoke, validation, or full")
    commands: list[Command] = []
    if not skip_tests:
        commands.append(
            Command(
                "gpu_regression_tests",
                (
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_diff_rea_gpu_integration.py",
                    "tests/test_cupy_solver.py",
                    "tests/test_cupy_autotune.py",
                    "tests/test_cupy_autotune_cache.py",
                    "tests/test_cupy_face_dense.py",
                    "tests/test_cupy_raw_preconditioner.py",
                    "tests/test_cupy_fused_additive_schwarz.py",
                    "tests/test_cupy_polynomial.py",
                ),
            )
        )

    cache = output_dir / "autotune_cache.json"
    smoke_prefix = output_dir / "smoke_hard_cases"
    commands.append(
        _validator(
            python,
            smoke_prefix,
            "--cases",
            "trigonometric-poisson",
            "tensor-sine",
            "--orders",
            "4",
            "--structured-nx",
            "8",
            "--mesh-size",
            "1.0",
            "--preconditioners",
            "none",
            "polynomial",
            "bj",
            "bj-polynomial",
            "asm",
            "asm-polynomial",
            "--polynomial-degrees",
            "8",
            "18",
            "--rtol",
            "1e-11",
            "--restart",
            "75",
            "--max-iterations",
            "3000",
            "--autotune-cache",
            str(cache),
            "--autotune-warmup",
            "3",
            "--autotune-repeats",
            "10",
            "--reference-solver",
            "direct",
            "--strict",
        )
    )
    if level == "smoke":
        return commands

    validation_prefix = output_dir / "all_cases_p4_p6_validation"
    commands.append(
        _validator(
            python,
            validation_prefix,
            "--cases",
            "all",
            "--orders",
            "4",
            "5",
            "6",
            "--structured-nx",
            "24",
            "--mesh-size",
            "0.4",
            "--preconditioners",
            "none",
            "polynomial",
            "bj",
            "bj-polynomial",
            "asm",
            "asm-polynomial",
            "--polynomial-degrees",
            "8",
            "12",
            "18",
            "24",
            "--rtol",
            "1e-12",
            "--restart",
            "100",
            "--max-iterations",
            "5000",
            "--autotune-cache",
            str(cache),
            "--autotune-warmup",
            "10",
            "--autotune-repeats",
            "50",
            "--reference-solver",
            "direct",
            "--strict",
        )
    )
    if level == "validation":
        commands.append(
            _analysis_command(python, output_dir, require_full=False)
        )
        return commands

    kernel_prefix = output_dir / "hard_kernel_sweep"
    commands.append(
        _validator(
            python,
            kernel_prefix,
            "--cases",
            "trigonometric-poisson",
            "tensor-sine",
            "--orders",
            "4",
            "6",
            "--structured-nx",
            "48",
            "--mesh-size",
            "0.25",
            "--preconditioners",
            "bj",
            "asm",
            "--operators",
            "raw",
            "raw_fused",
            "matmul",
            "--bj-applications",
            "raw",
            "matmul",
            "--asm-applications",
            "raw",
            "fused",
            "matmul",
            "--repeats",
            "3",
            "--warmup-solves",
            "1",
            "--no-autotune",
            "--no-measure-cold-autotune",
            "--reference-solver",
            "first-passing",
            "--rtol",
            "1e-12",
            "--restart",
            "100",
            "--max-iterations",
            "5000",
            "--strict",
        )
    )

    polynomial_prefix = output_dir / "hard_polynomial_degree_sweep"
    commands.append(
        _validator(
            python,
            polynomial_prefix,
            "--cases",
            "trigonometric-poisson",
            "tensor-sine",
            "--orders",
            "4",
            "5",
            "6",
            "--structured-nx",
            "48",
            "--mesh-size",
            "0.25",
            "--preconditioners",
            "polynomial",
            "bj-polynomial",
            "asm-polynomial",
            "--polynomial-degrees",
            "4",
            "8",
            "12",
            "18",
            "24",
            "32",
            "--operators",
            "auto",
            "--asm-applications",
            "auto",
            "--bj-applications",
            "auto",
            "--repeats",
            "3",
            "--warmup-solves",
            "1",
            "--autotune-cache",
            str(cache),
            "--autotune-warmup",
            "20",
            "--autotune-repeats",
            "100",
            "--reference-solver",
            "first-passing",
            "--rtol",
            "1e-12",
            "--restart",
            "100",
            "--max-iterations",
            "5000",
            "--strict",
        )
    )

    component_prefix = output_dir / "component_profile_p4_p6"
    commands.append(
        Command(
            "component_profile_p4_p6",
            (
                python,
                "scripts/profile_face_dense_gpu.py",
                "--cases",
                "trigonometric-poisson",
                "tensor-sine",
                "--orders",
                "4",
                "5",
                "6",
                "--meshes",
                "32",
                "--gmsh-mesh-size",
                "0.4",
                "--boundary-mode",
                "eliminate",
                "--matvec-implementations",
                "raw_fused",
                "raw",
                "matmul",
                "--gmres-matvec-implementation",
                "raw_fused",
                "--preconditioners",
                "none",
                "polynomial",
                "block_jacobi",
                "block_jacobi_polynomial",
                "asm",
                "asm_polynomial",
                "--polynomial-degrees",
                "18",
                "--preconditioner-application",
                "raw",
                "--asm-application",
                "fused",
                "--local-solver",
                "cublas_inverse",
                "--orthogonalization",
                "cgs",
                "--restart",
                "100",
                "--max-iterations",
                "5000",
                "--rtol",
                "1e-12",
                "--warmup",
                "20",
                "--repeats",
                "100",
                "--gmres-repeats",
                "5",
                "--output-prefix",
                str(component_prefix),
            ),
            (str(component_prefix.with_suffix(".csv")), str(component_prefix.with_suffix(".json"))),
        )
    )

    profile_target = (
        python,
        "scripts/validate_diff_rea_gpu_hard_cases.py",
        "--cases",
        "tensor-sine",
        "--orders",
        "6",
        "--structured-nx",
        "48",
        "--preconditioners",
        "asm-polynomial",
        "--polynomial-degrees",
        "18",
        "--operators",
        "raw_fused",
        "--asm-applications",
        "fused",
        "--orthogonalizations",
        "cgs",
        "--warmup-solves",
        "1",
        "--repeats",
        "1",
        "--no-autotune",
        "--no-measure-cold-autotune",
        "--reference-solver",
        "first-passing",
        "--rtol",
        "1e-12",
        "--restart",
        "100",
        "--max-iterations",
        "5000",
        "--output-prefix",
        str(output_dir / "nsight_target"),
    )
    if with_nsys:
        commands.append(
            Command(
                "nsight_systems_tensor_p6",
                (
                    "nsys",
                    "profile",
                    "--trace=cuda,nvtx,osrt,cublas",
                    "--sample=none",
                    "--force-overwrite=true",
                    "--output",
                    str(output_dir / "nsys_tensor_p6_asm_poly"),
                    *profile_target,
                ),
                (str(output_dir / "nsys_tensor_p6_asm_poly.nsys-rep"),),
            )
        )
    if with_ncu:
        commands.append(
            Command(
                "nsight_compute_tensor_p6",
                (
                    "ncu",
                    "--set",
                    "full",
                    "--target-processes",
                    "all",
                    "--force-overwrite",
                    "--export",
                    str(output_dir / "ncu_tensor_p6_asm_poly"),
                    *profile_target,
                ),
                (str(output_dir / "ncu_tensor_p6_asm_poly.ncu-rep"),),
            )
        )
    commands.append(_analysis_command(python, output_dir, require_full=True))
    return commands


def _capture(argv: Iterable[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            tuple(argv),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {"returncode": int(completed.returncode), "output": completed.stdout}
    except FileNotFoundError as error:
        return {"returncode": 127, "output": str(error)}


def collect_environment(python: str) -> dict[str, object]:
    commands = {
        "nvidia_smi": ("nvidia-smi",),
        "gpu_query": (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv",
        ),
        "python": (python, "--version"),
        "cupy": (
            python,
            "-c",
            "import cupy as cp; cp.show_config(); print(cp.cuda.runtime.getDeviceProperties(0)['name'])",
        ),
        "nsys": ("nsys", "--version"),
        "ncu": ("ncu", "--version"),
    }
    return {name: _capture(argv) for name, argv in commands.items()}


def run_commands(
    commands: list[Command],
    *,
    output_dir: Path,
    dry_run: bool,
    continue_on_error: bool,
) -> list[CommandResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    results: list[CommandResult] = []
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {command.name}\n  {command.shell}", flush=True)
        log_path = log_dir / f"{index:02d}_{command.name}.log"
        if dry_run:
            log_path.write_text(command.shell + "\n", encoding="utf-8")
            returncode = 0
            elapsed = 0.0
        else:
            environment = os.environ.copy()
            root = str(Path.cwd())
            existing = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = root if not existing else root + os.pathsep + existing
            start = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command.argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=environment,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                returncode = int(process.wait())
            elapsed = time.perf_counter() - start
        present = tuple(Path(path).exists() for path in command.expected_outputs)
        result = CommandResult(
            name=command.name,
            argv=command.argv,
            returncode=returncode,
            elapsed_seconds=elapsed,
            log_file=str(log_path),
            expected_outputs=command.expected_outputs,
            outputs_present=present,
        )
        results.append(result)
        if returncode != 0 and not continue_on_error:
            break
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("smoke", "validation", "full"), default="validation")
    parser.add_argument("--output", type=Path, default=Path("results/hard_gpu_campaign"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--with-nsys", action="store_true")
    parser.add_argument("--with-ncu", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.with_nsys and shutil.which("nsys") is None and not args.dry_run:
        raise SystemExit("--with-nsys requested but nsys is not on PATH")
    if args.with_ncu and shutil.which("ncu") is None and not args.dry_run:
        raise SystemExit("--with-ncu requested but ncu is not on PATH")
    commands = build_commands(
        args.campaign,
        output_dir=args.output,
        python=args.python,
        skip_tests=args.skip_tests,
        with_nsys=args.with_nsys,
        with_ncu=args.with_ncu,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(args.python) if not args.dry_run else {}
    results = run_commands(
        commands,
        output_dir=args.output,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": args.campaign,
        "dry_run": args.dry_run,
        "environment": environment,
        "commands": [asdict(command) for command in commands],
        "results": [asdict(result) for result in results],
    }
    summary = args.output / "campaign_summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = [result for result in results if result.returncode != 0]
    missing = [
        path
        for result in results
        for path, present in zip(result.expected_outputs, result.outputs_present, strict=True)
        if not present and not args.dry_run
    ]
    print(f"\nCampaign summary: {summary}")
    print(f"Commands completed: {len(results)}/{len(commands)}; failures={len(failures)}; missing_outputs={len(missing)}")
    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
