"""Qualify the scalar IDRMSYNC probe and disabled-history reporting on GPU."""
import argparse
from pathlib import Path
import re

import numpy as np

from benchmark_amgx_strong import smoke_system
from common import residual, save_iterate, system_fingerprint
from save_reference_iterates import run_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subspace-dim", type=int, choices=(4, 16, 32, 64), default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix, rhs, _ = smoke_system()
    exact = np.linalg.solve(matrix.toarray(), rhs)  # Tiny synthetic oracle only.
    seed = output / "half_solution.npz"
    save_iterate(seed, 0.5 * exact, dict(system_sha256=system_fingerprint(matrix, rhs),
                                      synthetic_smoke_only=True))
    # Native IDRMSYNC increments its counter inside each s+1 cycle. Align the
    # cap and keep dense history disabled, as in the scalar probe configs.
    cycle = args.subspace_dim + 1
    iteration_cap = ((1000 + cycle - 1) // cycle) * cycle
    candidate = f"idrmsync{args.subspace_dim}_unpreconditioned_absolute_csr"
    for name, fraction in (("zero", 1.), ("warm", 0.5)):
        options = ["--smoke-test", "--scaling", "none", "--max-iters", str(iteration_cap),
                   "--inner-rtol", "1e-10", "--timeout", "120"]
        if name == "warm":
            options += ["--initial-guess", str(seed)]
        result, x = run_candidate(output / f"{name}.json",
                                 candidate, options, verbosity=3)
        assert result["status"] == "passed" and residual(matrix, rhs, x, 1e-8)["passed"]
        np.testing.assert_allclose(x, exact, rtol=1e-7, atol=1e-8)
        sample = result["samples"][0]["passes"][0]
        assert sample["history_kind"] == "not_recorded" and sample["native_residual_history"] == []
        log = Path(result["report"]).with_suffix(".log").read_text()
        initial = re.findall(r"^\s*Ini\s+\S+\s+\S+\s+(\S+)", log, flags=re.MULTILINE)
        assert len(initial) == 1, initial
        # The printed native initial norm establishes that x0 was used even
        # though this solver's broken dense-history facility is disabled.
        np.testing.assert_allclose(float(initial[0]), fraction * np.linalg.norm(rhs), rtol=1e-6)
        print(f"[IDRMSYNC smoke] s={args.subspace_dim} {name}: passed, original relative residual {sample['relative_residual']:.12e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
