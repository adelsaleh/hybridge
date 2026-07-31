"""Run a reproducible T600/P100/V100 validation and profiling campaign.

The campaign is intentionally orchestration-only: every numerical experiment
remains available as a standalone script.  This wrapper records the software
and GPU environment, executes a selected command matrix, writes one log per
command, and produces a machine-readable summary.  ``--dry-run`` is useful for
checking a mesocentre allocation before consuming GPU time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable, Literal

CampaignName = Literal["smoke", "medium", "full"]


@dataclass(frozen=True)
class CampaignCommand:
    name: str
    argv: tuple[str, ...]
    expected_outputs: tuple[str, ...] = ()

    def shell(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class CampaignCommandResult:
    name: str
    argv: tuple[str, ...]
    returncode: int
    elapsed_seconds: float
    log_file: str
    expected_outputs: tuple[str, ...]
    outputs_present: tuple[bool, ...]


def _python_command(python: str, script: str, *arguments: str) -> tuple[str, ...]:
    return (python, script, *arguments)


def build_campaign_commands(
    campaign: CampaignName,
    *,
    output_dir: str | os.PathLike[str],
    python_executable: str = sys.executable,
    skip_tests: bool = False,
) -> list[CampaignCommand]:
    """Return the deterministic command matrix for one campaign level."""

    if campaign not in {"smoke", "medium", "full"}:
        raise ValueError("campaign must be smoke, medium, or full")
    output = Path(output_dir)
    commands: list[CampaignCommand] = []

    if not skip_tests:
        commands.append(
            CampaignCommand(
                "cuda_tests",
                (
                    python_executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_diff_rea_gpu_integration.py",
                    "tests/test_cupy_solver.py",
                    "tests/test_cupy_autotune.py",
                    "tests/test_cupy_autotune_cache.py",
                    "tests/test_cupy_fused_additive_schwarz.py",
                    "tests/test_cupy_polynomial.py",
                ),
            )
        )

    smoke_prefix = output / "smoke"
    commands.extend(
        [
            CampaignCommand(
                "autotune_p1",
                _python_command(
                    python_executable,
                    "scripts/autotune_face_dense_gpu.py",
                    "--mesh",
                    "32",
                    "--order",
                    "1",
                    "--boundary-mode",
                    "eliminate",
                    "--warmup",
                    "10",
                    "--repeats",
                    "30",
                    "--cache-file",
                    str(output / "autotune_cache.json"),
                    "--output",
                    str(smoke_prefix) + "_autotune_p1.json",
                ),
                (str(smoke_prefix) + "_autotune_p1.json",),
            ),
            CampaignCommand(
                "production_p1",
                _python_command(
                    python_executable,
                    "scripts/validate_face_dense_gpu_production_solver.py",
                    "--mesh",
                    "32",
                    "--order",
                    "1",
                    "--operator",
                    "raw",
                    "--preconditioner",
                    "asm_poly",
                    "--asm-application",
                    "fused",
                    "--polynomial-degree",
                    "12",
                    "--restart",
                    "75",
                    "--rtol",
                    "1e-8",
                ),
            ),
        ]
    )
    if campaign == "smoke":
        return commands

    medium_p1 = output / "medium_64_p1"
    medium_p4 = output / "medium_64_p4"
    commands.extend(
        [
            CampaignCommand(
                "numerical_matrix_float64",
                _python_command(
                    python_executable,
                    "scripts/validate_face_dense_gpu_numerics.py",
                    "--meshes",
                    "2",
                    "4",
                    "8",
                    "--orders",
                    "1",
                    "2",
                    "3",
                    "--boundary-modes",
                    "eliminate",
                    "penalty",
                    "--dtypes",
                    "float64",
                    "--methods",
                    "none",
                    "block_jacobi",
                    "asm",
                    "asm_poly",
                    "--polynomial-degree",
                    "8",
                    "--restart",
                    "50",
                    "--max-iterations",
                    "2000",
                    "--rtol",
                    "1e-8",
                    "--strict",
                    "--output-prefix",
                    str(output / "numerics_float64"),
                ),
                (
                    str(output / "numerics_float64.csv"),
                    str(output / "numerics_float64.json"),
                ),
            ),
            CampaignCommand(
                "autotune_p4",
                _python_command(
                    python_executable,
                    "scripts/autotune_face_dense_gpu.py",
                    "--mesh",
                    "64",
                    "--order",
                    "4",
                    "--boundary-mode",
                    "eliminate",
                    "--warmup",
                    "20",
                    "--repeats",
                    "100",
                    "--cache-file",
                    str(output / "autotune_cache.json"),
                    "--output",
                    str(output / "autotune_64_p4.json"),
                ),
                (str(output / "autotune_64_p4.json"),),
            ),
            CampaignCommand(
                "polynomial_p1",
                _python_command(
                    python_executable,
                    "scripts/validate_face_dense_gpu_polynomial.py",
                    "--mesh",
                    "64",
                    "--order",
                    "1",
                    "--degrees",
                    "8",
                    "12",
                    "18",
                    "20",
                    "--methods",
                    "asm",
                    "asm_poly",
                    "--setup-mode",
                    "shared",
                    "--rhs-counts",
                    "1",
                    "5",
                    "20",
                    "--restart",
                    "75",
                    "--max-iterations",
                    "2000",
                    "--outer-orthogonalization",
                    "cgs",
                    "--setup-orthogonalization",
                    "cgs2",
                    "--output-prefix",
                    str(medium_p1),
                ),
                (str(medium_p1) + ".csv", str(medium_p1) + ".json"),
            ),
            CampaignCommand(
                "polynomial_p4",
                _python_command(
                    python_executable,
                    "scripts/validate_face_dense_gpu_polynomial.py",
                    "--mesh",
                    "64",
                    "--order",
                    "4",
                    "--degrees",
                    "8",
                    "12",
                    "18",
                    "20",
                    "--methods",
                    "asm",
                    "asm_poly",
                    "--setup-mode",
                    "shared",
                    "--rhs-counts",
                    "1",
                    "5",
                    "20",
                    "--restart",
                    "100",
                    "--max-iterations",
                    "2000",
                    "--outer-orthogonalization",
                    "cgs",
                    "--setup-orthogonalization",
                    "cgs2",
                    "--output-prefix",
                    str(medium_p4),
                ),
                (str(medium_p4) + ".csv", str(medium_p4) + ".json"),
            ),
        ]
    )
    if campaign == "medium":
        return commands

    profile_prefix = output / "full_profile"
    commands.extend(
        [
            CampaignCommand(
                "local_assembly_p1",
                _python_command(
                    python_executable,
                    "scripts/validate_face_dense_gpu_local_assembly.py",
                    "--mesh",
                    "64",
                    "--order",
                    "1",
                    "--dtype",
                    "float64",
                    "--inverse-backend",
                    "cublas_inverse",
                    "--warmup",
                    "3",
                    "--repeats",
                    "30",
                ),
            ),
            CampaignCommand(
                "local_assembly_p4",
                _python_command(
                    python_executable,
                    "scripts/validate_face_dense_gpu_local_assembly.py",
                    "--mesh",
                    "64",
                    "--order",
                    "4",
                    "--dtype",
                    "float64",
                    "--inverse-backend",
                    "cublas_inverse",
                    "--warmup",
                    "5",
                    "--repeats",
                    "30",
                ),
            ),
            CampaignCommand(
                "scaling_profile",
                _python_command(
                    python_executable,
                    "scripts/profile_face_dense_gpu.py",
                    "--orders",
                    "1",
                    "2",
                    "3",
                    "4",
                    "--meshes",
                    "32",
                    "64",
                    "96",
                    "128",
                    "--boundary-mode",
                    "eliminate",
                    "--matvec-implementations",
                    "raw_fused",
                    "raw",
                    "--gmres-matvec-implementation",
                    "raw",
                    "--preconditioners",
                    "asm",
                    "--asm-application",
                    "fused",
                    "--local-solver",
                    "cublas_inverse",
                    "--orthogonalization",
                    "cgs",
                    "--restart",
                    "75",
                    "--max-iterations",
                    "2000",
                    "--rtol",
                    "1e-8",
                    "--output-prefix",
                    str(profile_prefix),
                ),
                (str(profile_prefix) + ".csv", str(profile_prefix) + ".json"),
            ),
        ]
    )
    return commands


def _run_capture(argv: Iterable[str], *, cwd: Path | None = None) -> dict[str, object]:
    try:
        completed = subprocess.run(
            tuple(argv),
            cwd=None if cwd is None else str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {"returncode": completed.returncode, "output": completed.stdout}
    except FileNotFoundError as error:
        return {"returncode": 127, "output": str(error)}


def collect_environment(python_executable: str = sys.executable) -> dict[str, object]:
    """Collect environment details without failing when optional tools are absent."""

    commands = {
        "nvidia_smi": ("nvidia-smi",),
        "nvidia_smi_query": (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv",
        ),
        "python": (python_executable, "--version"),
        "cupy": (
            python_executable,
            "-c",
            "import cupy as cp; cp.show_config(); "
            "print('device=', cp.cuda.runtime.getDeviceProperties(0)['name'])",
        ),
    }
    return {name: _run_capture(argv) for name, argv in commands.items()}


def run_campaign(
    commands: Iterable[CampaignCommand],
    *,
    output_dir: Path,
    continue_on_error: bool,
    dry_run: bool,
) -> list[CampaignCommandResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    results: list[CampaignCommandResult] = []

    for index, command in enumerate(commands, start=1):
        print(f"[{index}] {command.name}: {command.shell()}", flush=True)
        log_path = log_dir / f"{index:02d}_{command.name}.log"
        if dry_run:
            log_path.write_text(command.shell() + "\n", encoding="utf-8")
            returncode = 0
            elapsed = 0.0
        else:
            start = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as stream:
                environment = os.environ.copy()
                project_root = str(Path.cwd())
                existing_pythonpath = environment.get("PYTHONPATH")
                environment["PYTHONPATH"] = (
                    project_root
                    if not existing_pythonpath
                    else project_root + os.pathsep + existing_pythonpath
                )
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
        result = CampaignCommandResult(
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("smoke", "medium", "full"), default="smoke")
    parser.add_argument("--output", required=True, help="campaign output directory")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(args.python)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )
    commands = build_campaign_commands(
        args.campaign,
        output_dir=output_dir,
        python_executable=args.python,
        skip_tests=args.skip_tests,
    )
    results = run_campaign(
        commands,
        output_dir=output_dir,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": args.campaign,
        "dry_run": bool(args.dry_run),
        "commands": [asdict(item) for item in results],
    }
    (output_dir / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    failures = [item for item in results if item.returncode != 0]
    if failures:
        print(f"Campaign completed with {len(failures)} failure(s).", flush=True)
        return 1
    print(f"Campaign completed successfully: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
