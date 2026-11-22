#!/usr/bin/env python3
"""Qualify PARDISO permutation export and native AMGX zero/warm solves."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verbosity", type=int, choices=range(4), default=3)
    parser.add_argument("--unpreconditioned", action="store_true",
                        help="Check zero-start BiCGSTAB/GMRES with none, column-max and Ruiz scaling")
    parser.add_argument("--smoke-max-iters", type=int, default=1000)
    parser.add_argument("--scalings", nargs="+", choices=("none", "diagonal", "row-max", "column-max", "ruiz"),
                        default=("none", "diagonal", "row-max", "column-max", "ruiz"))
    args = parser.parse_args()
    import numpy as np
    from benchmark_amgx_strong import candidates, equilibrate, smoke_system
    from common import residual, save_iterate, system_fingerprint, write_report
    from pardiso_ordering import compute_permutation, reorder_csr, save_permutation
    from save_reference_iterates import run_candidate

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix, rhs, _ = smoke_system()
    fingerprint = system_fingerprint(matrix, rhs)
    p, metadata = compute_permutation(matrix, threads=1, verbosity=args.verbosity)
    metadata["system_sha256"] = fingerprint
    ordering_path = output / "synthetic.permutation.npz"
    save_permutation(ordering_path, p, metadata)
    exact = np.linalg.solve(matrix.toarray(), rhs)  # Tiny synthetic oracle only.
    seed = output / "half_solution.npz"
    if not args.unpreconditioned:
        save_iterate(seed, 0.5 * exact, dict(system_sha256=fingerprint, synthetic_smoke_only=True))
    report = dict(status="running", synthetic=True, permutation=str(ordering_path), results=[])
    write_report(output / "qualification.json", report)
    cases = (("gmres_legacy_ilu0_csr_ordering", "column-max"),
             ("pbicgstab_legacy_ilu0_csr_ordering", "column-max"),
             ("gmres_natural_ilu0_csr", "ruiz"))
    if args.unpreconditioned:
        cases = [(candidate, scaling) for scaling in args.scalings
                 for candidate in ("bicgstab_unpreconditioned_csr", "gmres_unpreconditioned_csr")]
    for number, (candidate, scaling) in enumerate(cases):
        for warm in ((False,) if args.unpreconditioned else (False, True)):
            name = f"case{number}_{'warm' if warm else 'zero'}"
            options = ["--smoke-test", "--ordering", "pardiso", "--permutation", str(ordering_path),
                       "--scaling", scaling, "--max-iters", str(args.smoke_max_iters) if args.unpreconditioned else "100",
                       "--inner-rtol", "1e-12", "--timeout", "120"]
            if warm:
                options += ["--initial-guess", str(seed)]
            result, x = run_candidate(output / f"{name}.json", candidate, options, verbosity=args.verbosity)
            checked = residual(matrix, rhs, x, 1e-8)
            converged = result["status"] == "passed" and checked["passed"]
            if converged:
                np.testing.assert_allclose(x, exact, rtol=1e-7, atol=1e-8)
            spec = candidates()[candidate]
            _, dr, _ = equilibrate(reorder_csr(matrix, p), 5, scaling,
                                   spec.get("right_scale_multiplier", 1.))
            x0 = 0.5 * exact if warm else np.zeros_like(rhs)
            expected = np.linalg.norm(dr * (rhs - matrix @ x0)[p])
            first = result["samples"][0]["passes"][0]
            np.testing.assert_allclose(first["native_residual_history"][0], expected, rtol=1e-9)
            assert result["ordering"]["matvec_mapping_relative_error"] <= 1e-12
            assert result["saved_solution"]["system_sha256"] == fingerprint
            report["results"].append(dict(candidate=candidate, warm=warm, scaling=scaling,
                converged=converged,
                native_initial_residual=first["native_residual_history"][0],
                original_relative_residual=checked["relative_residual"], report=result["report"]))
            write_report(output / "qualification.json", report)
            label = "PASS" if converged else "NUMERICAL NONCONVERGENCE"
            print(f"[matK ordering smoke] {name}: {label}, original residual {checked['relative_residual']:.3e}", flush=True)
    report["status"] = "passed" if all(row["converged"] for row in report["results"]) else "numerical_smoke_failed"
    write_report(output / "qualification.json", report)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
