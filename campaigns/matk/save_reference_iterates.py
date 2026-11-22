"""Reproduce and archive the two best matK iterates, sequentially on one GPU.

First validate checkpoint warm starts on small known systems. The full matK
references start from zero; no direct-solver solution seeds those runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from benchmark_amgx_strong import equilibrate, smoke_system
from common import load_iterate, residual, save_iterate, system_fingerprint, write_report


def run_candidate(output, name, options, *, verbosity):
    command = [sys.executable, "-u", str(Path(__file__).with_name("benchmark_amgx_strong.py")),
               "--output", str(output), "--candidates", name, "--repeats", "1",
               "--residual-corrections", "0", "--save-solution", "--verbosity", str(verbosity), *options]
    print(f"[matK] Checkpoint run: {name}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"Benchmark exited with {completed.returncode}: {name}")
    report = json.loads(output.read_text())
    if len(report["results"]) != 1:
        raise RuntimeError(f"Expected exactly one candidate result: {output}")
    result = report["results"][0]
    if result["status"] not in ("passed", "numerical_failure") or "saved_solution" not in result:
        raise RuntimeError(f"No completed finite iterate: {result['status']} ({output})")
    saved = result["saved_solution"]
    x, metadata = load_iterate(saved["path"], result["system_sha256"], result["matrix"]["shape"][0])
    if metadata["relative_residual"] != saved["relative_residual"]:
        raise RuntimeError("Checkpoint residual disagrees with its report")
    return result, x


def check_warm_starts(output, verbosity):
    matrix, rhs, _ = smoke_system()
    # Direct solve is ONLY a small synthetic-test oracle, never a matK seed.
    exact = np.linalg.solve(matrix.toarray(), rhs)
    seed = output / "synthetic_half_solution.npz"
    save_iterate(seed, 0.5 * exact, dict(system_sha256=system_fingerprint(matrix, rhs),
                                      synthetic_smoke_only=True, **residual(matrix, rhs, 0.5 * exact, 1e-8)))
    cases = (("gmres_natural_ilu0_csr_r6000", "column-max"),
             ("fgmres_natural_ilu0_b20", "ruiz"))
    for number, (name, scaling) in enumerate(cases):
        result, x = run_candidate(output / f"warm_start_{number}.json", name,
            ["--smoke-test", "--initial-guess", str(seed), "--scaling", scaling,
             "--right-scale-multiplier", "1000", "--max-iters", "40", "--timeout", "180"],
            verbosity=verbosity)
        np.testing.assert_allclose(result["initial_guess"]["relative_residual"], 0.5, rtol=1e-12)
        if result["status"] != "passed" or not residual(matrix, rhs, x, 1e-8)["passed"]:
            raise RuntimeError(f"Warm-start recovery failed: {name}")
        np.testing.assert_allclose(x, exact, rtol=1e-7, atol=1e-10)
        first_pass = result["samples"][0]["passes"][0]
        _, dr, _ = equilibrate(matrix, 5, scaling, 1000)
        expected_initial_residual = np.linalg.norm(dr * (rhs - matrix @ (0.5 * exact)))
        # Convergence alone would also pass if AMGX silently ignored x0.
        np.testing.assert_allclose(first_pass["native_residual_history"][0],
                                   expected_initial_residual, rtol=1e-10)
        if first_pass["iterations"] < 1:
            raise RuntimeError("Warm-start smoke did not exercise an iteration")
    print("[matK] PASS: CSR and BSR warm starts, scaling and checkpoint reload", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=Path(__file__).with_name("matK.h5"))
    parser.add_argument("--verbosity", type=int, choices=range(4), default=3)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Existing archives must survive accidental reuse of an output directory.
    if (output / "index.json").exists():
        parser.error("Use a new output directory; index.json already exists")
    check_warm_starts(output, args.verbosity)
    if args.smoke_only:
        return 0
    manifest = dict(status="running", warm_start_smoke="passed", references=[])
    write_report(output / "index.json", manifest)
    cases = (
        ("bicgstab_unpreconditioned_reference", "bicgstab_unpreconditioned", "none", "1000", "1e-8", "300"),
        ("gmres_natural_ilu0_csr_r6000", "gmres_natural_ilu0", "column-max", "6000", "1e-10", "3600"),
    )
    for name, short_name, scaling, iterations, tolerance, timeout in cases:
        result, _ = run_candidate(output / f"{short_name}.json", name,
            ["--matrix", str(args.matrix.resolve()), "--scaling", scaling,
             "--max-iters", iterations, "--inner-rtol", tolerance, "--timeout", timeout],
            verbosity=args.verbosity)
        saved = result["saved_solution"]
        destination = output / f"{short_name}.solution.npz"
        temporary = destination.with_suffix(".npz.tmp")
        shutil.copyfile(saved["path"], temporary)
        temporary.replace(destination)
        manifest["references"].append(dict(candidate=name, path=str(destination),
            report=result["report"], relative_residual=saved["relative_residual"],
            system_sha256=result["system_sha256"], vector_sha256=saved["vector_sha256"],
            converged=saved["passed"]))
        write_report(output / "index.json", manifest)
        print(f"[matK] Archived {destination} | original relative residual {saved['relative_residual']:.12e}", flush=True)
    manifest["status"] = "saved"
    write_report(output / "index.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
