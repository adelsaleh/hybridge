#!/usr/bin/env python3
"""Compare robust native GMRES finalists with robust AMGX BSR presets.

The named million-DOF problems are VV (variable velocity), MT (transport),
TR (trigonometric robustness), and AR (anisotropic robustness).  Native
BJ+PP and ASM+PP finalists are selected from a completed full ADR campaign;
the exact assembly caches from that campaign are reused by every solver.
"""
from __future__ import annotations

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))
_campaign_sys.path.insert(0, str(_campaign_workspace_path('hdgfem-gmres')))


import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

from scripts.adr_performance_common import atomic_json, read_json, trace_dofs
from campaigns.legacy_adr_performance.run_adv_diff_rea_performance import source_hash


ROOT = _campaign_workspace_path('hdgfem-gmres')
NAMED_CASES = {
    "VV": "variable_velocity",
    "MT": "advection_dominated",
    "TR": "trigonometric",
    "AR": "anisotropic",
}
MESHES = {3: 289, 4: 259}
NATIVE_FAMILIES = {
    "pp": "poly",
    "bj_pp": "block_jacobi_poly",
    "asm_pp": "asm_poly",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observations(row):
    """Return verification-bearing attempts from either worker format."""
    values = []
    for sample in row.get("samples", []) + row.get("warmups", []):
        if "true_relative_residual" in sample:
            values.append(sample)
        values.extend(s for s in sample.get("solves", []) if "true_relative_residual" in s)
    return values


def residual(row):
    values = [s["true_relative_residual"] for s in observations(row)]
    return min((float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)),
               default=math.inf)


def elapsed(row):
    if isinstance(row.get("setup_solve_median_ms"), (int, float)):
        return float(row["setup_solve_median_ms"])
    values = [s.get("setup_solve_ms", math.inf) for s in observations(row)]
    return min(values, default=math.inf)


def target_time_proxy(row, internal_rtol):
    """Estimate time to an earlier tolerance from saved restart residuals."""
    best = math.inf
    restart = int(row.get("configuration", {}).get("restart", 75))
    for sample in observations(row):
        history = sample.get("residual_history", [])
        relative = sample.get("true_relative_residual")
        absolute = sample.get("true_residual_norm")
        iterations = sample.get("iterations")
        solve_ms = sample.get("solve_ms")
        if not history or not all(isinstance(v, (int, float)) for v in
                                  (relative, absolute, iterations, solve_ms)) or relative <= 0 or iterations <= 0:
            continue
        rhs_norm = absolute / relative
        crossing = next((i for i, value in enumerate(history)
                         if isinstance(value, (int, float)) and value <= internal_rtol * rhs_norm), None)
        if crossing is not None:
            estimated_iterations = min(iterations, max(1, crossing) * restart)
            best = min(best, solve_ms * estimated_iterations / iterations)
    return best


def source_rows(source, case, n, p):
    rows = []
    for path in sorted((source / "jobs").glob(f"measure_{case}_n{n}_p{p}_*.json")):
        try:
            row = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        row["_source_job"] = str(path.resolve())
        rows.append(row)
    return rows


def select_native(source, case, n, p, expected, internal_rtol):
    rows = source_rows(source, case, n, p)
    if len(rows) != expected or any(r.get("status") == "running" for r in rows):
        raise RuntimeError(
            f"source campaign is incomplete for {case} p={p}: "
            f"found {len(rows)}/{expected} terminal large-candidate records"
        )
    selected = []
    for label, family in NATIVE_FAMILIES.items():
        family_rows = [r for r in rows if r.get("configuration", {}).get("preconditioner") == family]
        if not family_rows:
            raise RuntimeError(f"source campaign has no {family} candidates for {case} p={p}")
        passed = [r for r in family_rows if r.get("status") == "passed"]
        if passed:
            winner = min(passed, key=elapsed)
            rule = "fastest_fully_validated"
        else:
            usable = [r for r in family_rows if math.isfinite(residual(r))]
            if not usable:
                raise RuntimeError(f"source campaign has no verified residual for {family}, {case} p={p}")
            target = [r for r in usable if math.isfinite(target_time_proxy(r, internal_rtol))]
            if target:
                winner = min(target, key=lambda r: (target_time_proxy(r, internal_rtol), residual(r)))
                rule = "earliest_internal_target_proxy"
            else:
                winner = min(usable, key=lambda r: (residual(r), elapsed(r)))
                rule = "smallest_verified_residual_fallback"
        selected.append({
            "family": label,
            "candidate": f"{label}_{winner['configuration']['operator']}_d"
                         f"{winner['configuration']['polynomial_degree']}",
            "configuration": winner["configuration"],
            "selection": {
                "rule": rule,
                "source_status": winner.get("status"),
                "source_job": winner["_source_job"],
                "source_job_sha256": sha256(winner["_source_job"]),
                "source_setup_solve_ms": elapsed(winner),
                "source_true_relative_residual": residual(winner),
                "source_internal_target_time_proxy_ms": target_time_proxy(winner, internal_rtol),
            },
        })
    return selected


def amgx_candidates(config_root, case):
    """Choose BSR presets supported by the S3/OC regime results.

    S3 and OC favored block-graph AMG with block Jacobi for high/anisotropic
    diffusion, and block-graph AMG plus direct DILU for transport.  VV is the
    new crossover, so it retains both AMG smoothers and direct DILU.
    """
    direct = {
        "dilu": read_json(config_root / "adv_rea_gpu4_hdg_pbicgstab_dilu_bsr_p1_p3.json"),
        "block_jacobi": read_json(config_root / "adv_rea_gpu4_hdg_pbicgstab_block_jacobi_bsr.json"),
    }
    dense = read_json(config_root / "diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_block_graph_dense_bsr.json")
    pool = []
    for outer in ("FGMRES", "PBICGSTAB"):
        for name, base in direct.items():
            config = deepcopy(base)
            config["solver"].update(solver=outer, bsr_spmv_backend="cusparse_generic", use_scalar_norm=1)
            pool.append({"family": "amgx", "candidate": f"amgx_bsr_{outer.lower()}_{name}",
                         "matrix_format": "bsr", "amgx_config": config})
        for smoother in ("MULTICOLOR_DILU", "BLOCK_JACOBI"):
            config = deepcopy(dense)
            solver = config["solver"]
            solver.update(solver=outer, bsr_spmv_backend="cusparse_generic", use_scalar_norm=1)
            pre = solver["preconditioner"]
            pre.update(classical_bsr_hierarchy="block_graph_dense", presweeps=1, postsweeps=1)
            pre["smoother"] = {"solver": smoother, "max_iters": 1, "relaxation_factor": 1.0}
            name = smoother.lower()
            pool.append({"family": "amgx", "candidate": f"amgx_bsr_{outer.lower()}_amg_{name}",
                         "matrix_format": "bsr", "amgx_config": config})
    keep = {
        "variable_velocity": ("_dilu", "_amg_multicolor_dilu", "_amg_block_jacobi"),
        "advection_dominated": ("_dilu", "_amg_multicolor_dilu"),
        "trigonometric": ("_amg_multicolor_dilu", "_amg_block_jacobi"),
        "anisotropic": ("_amg_multicolor_dilu", "_amg_block_jacobi"),
    }[case]
    # Direct block Jacobi was explored previously but was not a regime winner.
    return [candidate for candidate in pool
            if any(candidate["candidate"].endswith(suffix) for suffix in keep)]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="completed full ADR performance campaign")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config-root", type=Path, default=_campaign_workspace_path('.') / "hdgfem" / "configs" / "amgx")
    p.add_argument("--cases", nargs="+", choices=NAMED_CASES, default=list(NAMED_CASES))
    p.add_argument("--orders", nargs="+", type=int, choices=MESHES, default=sorted(MESHES))
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--solves-per-setup", type=int, default=2)
    p.add_argument("--rtol", type=float, default=1e-10,
                   help="independently verified physical residual tolerance")
    p.add_argument("--internal-rtol", type=float, default=1e-11,
                   help="common native/AMGX stopping tolerance")
    p.add_argument("--maxiter", type=int, default=1000)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="write and print the immutable plan only")
    return p


def validate(a, p):
    if min(a.warmup, a.repeats, a.solves_per_setup) < 1 or a.maxiter < 1 or a.timeout <= 0:
        p.error("warmup, repeats, solves-per-setup, maxiter and timeout must be positive")
    if not 0 < a.rtol < 1 or not 0 < a.internal_rtol < 1:
        p.error("rtol and internal-rtol must lie in (0,1)")
    for path in (a.source / "manifest.json", a.source / "jobs"):
        if not path.exists():
            p.error(f"missing source campaign artifact: {path}")


def main():
    p = parser()
    a = p.parse_args()
    a.source, a.output, a.config_root = a.source.resolve(), a.output.resolve(), a.config_root.resolve()
    validate(a, p)
    source_manifest = read_json(a.source / "manifest.json")
    expected = len(source_manifest.get("paths", []))
    if not expected:
        p.error("source manifest has no implementation paths")

    problems = []
    for short in a.cases:
        case = NAMED_CASES[short]
        for order in sorted(set(a.orders)):
            n = MESHES[order]
            assembly = a.source / "jobs" / f"assemble_{case}_n{n}_p{order}.json"
            spec = a.source / "specs" / f"assemble_{case}_n{n}_p{order}.json"
            if not assembly.exists() or read_json(assembly).get("status") != "passed" or not spec.exists():
                raise RuntimeError(f"missing passed source assembly for {short}, p={order}")
            cache = Path(read_json(spec)["cache"]).resolve()
            if not (cache / "metadata.json").exists():
                raise RuntimeError(f"missing source cache for {short}, p={order}: {cache}")
            native = select_native(a.source, case, n, order, expected, a.internal_rtol)
            amgx = amgx_candidates(a.config_root, case)
            hp = ([{"family": "native_hp", "candidate": "native_hp_bsr", "policy": "standard"}]
                  if case in ("variable_velocity", "trigonometric") else [])
            problems.append({"name": short, "case": case, "n": n, "p": order,
                             "trace_dofs": trace_dofs(n, order), "cache": str(cache),
                             "source_assembly": str(assembly.resolve()),
                             "source_assembly_sha256": sha256(assembly),
                             "candidates": native + hp + amgx})

    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                 if k not in ("resume", "dry_run")}
    manifest = {
        "arguments": arguments,
        "source_campaign_manifest": str((a.source / "manifest.json").resolve()),
        "source_campaign_manifest_sha256": sha256(a.source / "manifest.json"),
        "solver_source_sha256": source_hash(),
        "worker_sha256": {
            "comparison": sha256(_campaign_source_path(Path('hdgfem-gmres/scripts') / 'adr_solver_comparison_worker.py')),
            "native_hp": sha256(_campaign_source_path(Path('hdgfem-gmres/scripts') / 'adr_native_hp_worker.py')),
        },
        "preset_basis": {
            "S3": "block-graph AMG/Jacobi led high and anisotropic diffusion; block-graph AMG/DILU was the robust second finalist; AMG/DILU and direct DILU covered transport; native hp-BSR covered high diffusion",
            "OC": "same transferred class presets; direct DILU failed in oscillatory low diffusion and AMG/Jacobi failed in directional anisotropy",
            "VV": "crossover probe retaining both S3 AMG smoothers and direct DILU",
        },
        "problems": problems,
        "policy": (
            "Each named problem reuses one immutable full-campaign FP64 matrix and RHS. "
            "Native finalists are the fastest fully validated PP, BJ+PP and ASM+PP records, or the "
            "smallest verified-residual fallback when a family has no passing record. AMGX uses "
            "natural (p+1)-square BSR blocks only. Every timed solve starts from zero; setup and "
            "hot solve times are separated; synchronized timings exclude independent verification."
        ),
    }
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "specs", "logs"):
        (a.output / name).mkdir(exist_ok=True)
    manifest_path = a.output / "manifest.json"
    if manifest_path.exists():
        if not a.resume or read_json(manifest_path) != manifest:
            p.error("existing or changed campaign; use --resume with identical inputs or a new output")
    atomic_json(manifest_path, manifest)
    count = sum(len(problem["candidates"]) for problem in problems)
    print(f"Plan: {len(problems)} named systems, {count} solver jobs; BSR-only AMGX", flush=True)
    for problem in problems:
        native = problem["candidates"][:len(NATIVE_FAMILIES)]
        choices = ", ".join(f"{c['family']}:{c['selection']['rule']}" for c in native)
        print(f"{problem['name']} p={problem['p']} n={problem['n']} "
              f"dofs={problem['trace_dofs']}: {choices}", flush=True)
    if a.dry_run:
        return

    common = {"rtol": a.rtol, "internal_rtol": a.internal_rtol, "maxiter": a.maxiter, "device": 0,
              "engine": "gpu", "warmup": a.warmup, "repeats": a.repeats,
              "solves_per_setup": a.solves_per_setup, "reference_max_dofs": 20000,
              "memory_fraction": 0.8, "host_limit_gib": None}
    records = []
    for problem in problems:
        active = list(problem["candidates"])
        random.Random(a.seed + problem["p"] + sum(map(ord, problem["name"]))).shuffle(active)
        for candidate in active:
            key = f"{problem['name'].lower()}_p{problem['p']}_{candidate['candidate']}"
            result_path = a.output / "jobs" / f"{key}.json"
            job = {**common, **candidate, "case": problem["case"], "n": problem["n"],
                   "p": problem["p"], "cache": problem["cache"], "result": str(result_path)}
            spec_path = a.output / "specs" / f"{key}.json"
            if spec_path.exists() and read_json(spec_path) != job:
                raise RuntimeError(f"job specification changed: {key}")
            atomic_json(spec_path, job)
            if a.resume and result_path.exists() and read_json(result_path).get("status") != "running":
                result = read_json(result_path)
            else:
                with (a.output / "logs" / f"{key}.log").open("w") as log:
                    try:
                        module = ("scripts.adr_native_hp_worker" if candidate["family"] == "native_hp"
                                  else "scripts.adr_solver_comparison_worker")
                        done = subprocess.run([sys.executable, "-m", module,
                                               "--spec", str(spec_path)], cwd=ROOT, stdout=log,
                                              stderr=subprocess.STDOUT, timeout=a.timeout)
                        result = read_json(result_path) if result_path.exists() else {
                            "status": "process_error", "returncode": done.returncode}
                    except subprocess.TimeoutExpired:
                        result = {"status": "timeout", "timeout_seconds": a.timeout}
                        atomic_json(result_path, result)
            row = {**result, "name": problem["name"], "case": problem["case"], "n": problem["n"],
                   "p": problem["p"], "trace_dofs": problem["trace_dofs"],
                   "family": candidate["family"], "candidate": candidate["candidate"]}
            records.append(row)
            atomic_json(a.output / "summary.json", records)
            print(key, row.get("status"), row.get("fresh_setup_solve_median_ms"), flush=True)

    failures = [{"name": r["name"], "p": r["p"], "candidate": r["candidate"],
                 "status": r.get("status")} for r in records if r.get("status") != "passed"]
    completion = {"status": "completed", "attempted": len(records),
                  "passed": sum(r.get("status") == "passed" for r in records),
                  "failures": failures,
                  "note": "completed means every scheduled candidate was attempted; numerical failures are retained."}
    atomic_json(a.output / "completion.json", completion)
    print(completion, flush=True)


if __name__ == "__main__":
    main()
