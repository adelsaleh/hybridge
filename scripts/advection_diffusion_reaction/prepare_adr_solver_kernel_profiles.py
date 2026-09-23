#!/usr/bin/env python3
"""Prepare short solver-only replays of one archived, real HDG ADR system.

The generated specs preserve the original operator, RHS, solver configuration,
and input hashes. They deliberately stop after a few Krylov iterations: a
nonconverged status is expected and must not be reported as a solver failure or
used for end-to-end performance rankings. Run the prepared workers separately
under Nsight; this preparation step performs no CUDA work or compilation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "run_outputs/solver_studies/adr_closed_loop_stress_strong_numba_h150k"
CASE = "stress_main_orthogonal_t150000_p6"
CANDIDATES = (
    "asm_pp", "bj_pp", "native_hp_standard", "native_hp_robust",
    "amgx_block_jacobi_fgmres", "amgx_block_jacobi_pbicgstab",
    "amgx_dilu_fgmres", "amgx_dilu_pbicgstab",
    "amgx_multicolor_dilu_fgmres", "amgx_multicolor_dilu_pbicgstab",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4,
                        help="Short diagnostic Krylov cap; default: 4")
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES,
                        default=CANDIDATES)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 20:
        parser.error("--iterations must be between 1 and 20")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for candidate in dict.fromkeys(args.candidates):
        source = SOURCE / "specs" / f"{CASE}_{candidate}.json"
        spec = json.loads(source.read_text())
        cache = Path(spec["cache"])
        if not cache.is_dir() or not Path(spec["mesh_path"]).is_file():
            raise FileNotFoundError(f"Missing archived inputs for {candidate}")
        spec.update(maxiter=args.iterations, restart=args.iterations,
                    warmup=0, repeats=1, solves_per_setup=1,
                    record_residual_history=False, profile_nvtx=True,
                    profile_residual_only=True,
                    result=str(output / f"{candidate}.result.json"))
        if isinstance(spec.get("configuration"), dict):
            spec["configuration"]["restart"] = args.iterations
        if isinstance(spec.get("amgx_config"), dict):
            solver = spec["amgx_config"]["solver"]
            if solver.get("solver") in ("FGMRES", "GMRES"):
                solver["gmres_n_restart"] = args.iterations
        path = output / f"{candidate}.spec.json"
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        print(f"{candidate}: {path}")
    print(f"Archived system: {CASE}; 1,547,455 trace DOFs; p=6")
    print("Short runs intentionally stop before convergence; use archived full solves for rankings.")


if __name__ == "__main__":
    main()
