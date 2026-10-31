#!/usr/bin/env python3
"""Run S3/OC-style h/p scaling with finalists from named ADR comparisons."""
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

from scripts.adr_performance_common import atomic_json, estimates, read_json, trace_dofs, write_csv
from campaigns.legacy_adr_performance.run_adv_diff_rea_performance import source_hash


ROOT = _campaign_workspace_path('hdgfem-gmres')
NAMED_CASES = {
    "VV": "variable_velocity",
    "MT": "advection_dominated",
    "TR": "trigonometric",
    "AR": "anisotropic",
}
# Exact S3/OC ladder: h scaling at p=6 and p scaling at n=64.
POINTS = ((32, 6), (64, 1), (64, 2), (64, 3), (64, 4), (64, 6), (128, 6), (223, 6))
NATIVE_FAMILIES = ("pp", "bj_pp", "asm_pp")
DEFAULT_CACHE_ROOTS = (
    ROOT / "run_logs" / "adr_scaling_20260917" / "cache",
    ROOT / "run_logs" / "adr_solver_screen_bsr_p6_20260917" / "cache",
    ROOT / "run_logs" / "adr_solver_large_p6_20260917" / "cache",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_index(manifest):
    result = {}
    for problem in manifest["problems"]:
        for candidate in problem["candidates"]:
            result[(problem["name"], problem["p"], candidate["candidate"])] = candidate
    return result


def load_comparisons(paths):
    manifests, rows, definitions = [], [], {}
    protocol = None
    for root in paths:
        root = root.resolve()
        completion = read_json(root / "completion.json")
        if completion.get("status") != "completed":
            raise RuntimeError(f"comparison is not complete: {root}")
        manifest = read_json(root / "manifest.json")
        current = tuple(manifest["arguments"][key] for key in
                        ("rtol", "internal_rtol", "maxiter", "warmup", "repeats", "solves_per_setup"))
        if protocol is not None and current != protocol:
            raise RuntimeError("comparison artifacts use different numerical/timing protocols")
        protocol = current
        manifests.append((root, manifest))
        definitions.update(candidate_index(manifest))
        for row in read_json(root / "summary.json"):
            rows.append({**row, "_comparison": str(root)})
    return manifests, rows, definitions, protocol


def freeze_finalists(case, rows, definitions):
    """Freeze p=4 native controls and the best passing AMGX result per outer solver."""
    p4 = [row for row in rows if row.get("name") == case and row.get("p") == 4]
    if not p4:
        raise RuntimeError(f"no p=4 comparison records for {case}")
    passed = [row for row in p4 if row.get("status") == "passed"
              and isinstance(row.get("fresh_setup_solve_median_ms"), (int, float))]
    selected = []
    for family in NATIVE_FAMILIES:
        options = [row for row in passed if row.get("family") == family]
        if not options:
            raise RuntimeError(f"no passing p=4 {family} finalist for {case}")
        selected.append(min(options, key=lambda row: row["fresh_setup_solve_median_ms"]))

    amgx = [row for row in passed if row.get("family") == "amgx"]
    for outer in ("FGMRES", "PBICGSTAB"):
        options = []
        for row in amgx:
            definition = definitions[(case, 4, row["candidate"])]
            if definition["amgx_config"]["solver"]["solver"] == outer:
                options.append(row)
        if not options:
            raise RuntimeError(f"no passing p=4 AMGX/{outer} finalist for {case}")
        selected.append(min(options, key=lambda row: row["fresh_setup_solve_median_ms"]))

    # Retain direct PBICGSTAB/DILU on transport as the no-coarse-grid baseline.
    if case == "MT":
        direct = [row for row in amgx if row["candidate"] == "amgx_bsr_pbicgstab_dilu"]
        if direct and direct[0] not in selected:
            selected.append(direct[0])
    hp = [row for row in passed if row.get("family") == "native_hp"]
    if hp:
        selected.append(min(hp, key=lambda row: row["fresh_setup_solve_median_ms"]))

    result = []
    for row in selected:
        definition = deepcopy(definitions[(case, 4, row["candidate"])])
        definition["selection"] = {
            "comparison": row["_comparison"],
            "p4_fresh_setup_solve_median_ms": row["fresh_setup_solve_median_ms"],
            "p4_hot_solve_median_ms": row.get("hot_solve_median_ms"),
            "rule": "p4 fastest passing per native family or AMGX outer solver",
        }
        result.append(definition)
    return result


def valid_cache(path):
    metadata = path / "metadata.json"
    if not metadata.exists():
        return False
    try:
        record = read_json(metadata)
    except (OSError, json.JSONDecodeError):
        return False
    required = ("system_blocks.npy", "system_rhs.npy", "system_neighbors.npy", "local_inverse.npy",
                "boundary.npy", "source_rhs.npy", "loc2glob.npy", "element_blocks.npy")
    return record.get("status") == "passed" and all((path / name).exists() for name in required)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", nargs="+", type=Path, required=True,
                   help="one or more completed named comparison directories")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", choices=NAMED_CASES,
                   help="default: every named case with a p=4 comparison")
    p.add_argument("--cache-root", action="append", type=Path, default=[],
                   help="additional immutable assembly-cache root")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--assembly-timeout", type=float, default=1800)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    p = parser()
    a = p.parse_args()
    if a.timeout <= 0 or a.assembly_timeout <= 0:
        p.error("timeouts must be positive")
    a.output = a.output.resolve()
    comparisons = [path.resolve() for path in a.comparison]
    manifests, rows, definitions, protocol = load_comparisons(comparisons)
    available = sorted({row["name"] for row in rows if row.get("p") == 4})
    cases = a.cases or available
    missing = sorted(set(cases) - set(available))
    if missing:
        p.error("missing completed p=4 comparison for: " + ", ".join(missing))
    finalists = {case: freeze_finalists(case, rows, definitions) for case in cases}

    cache_roots = [path.resolve() for path in a.cache_root]
    cache_roots += [path.resolve() for path in DEFAULT_CACHE_ROOTS]
    # A comparison's source campaign may contain reusable caches too.
    for _, manifest in manifests:
        cache_roots.append(Path(manifest["arguments"]["source"]).resolve() / "cache")
    cache_roots = list(dict.fromkeys(cache_roots))

    rtol, internal_rtol, maxiter, warmup, repeats, solves_per_setup = protocol
    problems = []
    for case in cases:
        long_name = NAMED_CASES[case]
        for n, degree in POINTS:
            key = f"{long_name}_n{n}_p{degree}"
            shared = next((root / key for root in cache_roots if valid_cache(root / key)), None)
            cache = shared or (a.output / "cache" / key)
            problems.append({"name": case, "case": long_name, "n": n, "p": degree,
                             "triangles": 2 * n * n, "trace_dofs": trace_dofs(n, degree),
                             "cache": str(cache), "cache_origin": "reused" if shared else "new",
                             "estimate": estimates(n, degree), "candidates": finalists[case]})

    arguments = {"comparison": [str(path) for path in comparisons], "output": str(a.output),
                 "cases": cases, "cache_root": [str(path) for path in cache_roots],
                 "timeout": a.timeout, "assembly_timeout": a.assembly_timeout, "seed": a.seed}
    manifest = {
        "arguments": arguments,
        "points": [{"n": n, "p": degree} for n, degree in POINTS],
        "protocol": {"rtol": rtol, "internal_rtol": internal_rtol, "maxiter": maxiter,
                     "warmup": warmup, "repeats": repeats, "solves_per_setup": solves_per_setup},
        "comparison_sha256": {str(root): sha256(root / "manifest.json") for root, _ in manifests},
        "source_sha256": source_hash(),
        "worker_sha256": {
            "comparison": sha256(_campaign_source_path(Path('hdgfem-gmres/scripts') / 'adr_solver_comparison_worker.py')),
            "native_hp": sha256(_campaign_source_path(Path('hdgfem-gmres/scripts') / 'adr_native_hp_worker.py')),
            "assembly": sha256(_campaign_source_path(Path('hdgfem-gmres/scripts') / 'adr_performance_worker.py')),
        },
        "finalists": finalists,
        "problems": problems,
        "policy": (
            "Exact S3/OC h/p ladder. Each case freezes its passing p=4 named-comparison finalists "
            "across every point: PP, BJ+PP, ASM+PP, fastest AMGX BSR FGMRES and PBICGSTAB, native "
            "hp-BSR where qualified, plus direct PBICGSTAB/DILU for MT. Common FP64 matrices, zero "
            "initial guesses, one warmup and three measured fresh setups with two solves per setup."
        ),
    }
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "specs", "logs", "cache"):
        (a.output / name).mkdir(exist_ok=True)
    manifest_path = a.output / "manifest.json"
    if manifest_path.exists():
        if not a.resume or read_json(manifest_path) != manifest:
            p.error("existing or changed scaling campaign; use --resume with identical inputs or a new output")
    atomic_json(manifest_path, manifest)

    solver_jobs = sum(len(problem["candidates"]) for problem in problems)
    new_assemblies = sum(problem["cache_origin"] == "new" for problem in problems)
    print(f"Plan: {len(problems)} systems, {solver_jobs} solver jobs, "
          f"{new_assemblies} new assemblies", flush=True)
    for case in cases:
        names = ", ".join(candidate["candidate"] for candidate in finalists[case])
        print(f"{case}: {names}", flush=True)
    if a.dry_run:
        return

    common = {"rtol": rtol, "internal_rtol": internal_rtol, "maxiter": maxiter,
              "warmup": warmup, "repeats": repeats, "solves_per_setup": solves_per_setup,
              "device": 0, "engine": "gpu", "assembly_backend": "cupy", "assembly_warmup": 1,
              "assembly_repeats": 3, "reference_max_dofs": 100000,
              "memory_fraction": 0.8, "host_limit_gib": None}
    records, assemblies = [], []

    def run(spec, module, key, timeout):
        result_path = a.output / "jobs" / f"{key}.json"
        spec = {**spec, "result": str(result_path)}
        spec_path = a.output / "specs" / f"{key}.json"
        if spec_path.exists() and read_json(spec_path) != spec:
            raise RuntimeError(f"job specification changed: {key}")
        atomic_json(spec_path, spec)
        if a.resume and result_path.exists() and read_json(result_path).get("status") != "running":
            return read_json(result_path)
        with (a.output / "logs" / f"{key}.log").open("w") as log:
            try:
                done = subprocess.run([sys.executable, "-m", module, "--spec", str(spec_path)],
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                result = read_json(result_path) if result_path.exists() else {
                    "status": "process_error", "returncode": done.returncode}
                if result.get("status") == "running":
                    result.update(status="process_error", returncode=done.returncode)
            except subprocess.TimeoutExpired:
                result = {"status": "timeout", "timeout_seconds": timeout}
            atomic_json(result_path, result)
            return result

    for problem in problems:
        if problem["cache_origin"] == "new":
            assembled = run({**common, "task": "assemble", "case": problem["case"],
                             "n": problem["n"], "p": problem["p"], "cache": problem["cache"]},
                            "scripts.adr_performance_worker",
                            f"assemble_{problem['case']}_n{problem['n']}_p{problem['p']}",
                            a.assembly_timeout)
        else:
            assembled = read_json(Path(problem["cache"]) / "metadata.json")
        assemblies.append({"name": problem["name"], "case": problem["case"], "n": problem["n"],
                           "p": problem["p"], "cache": problem["cache"],
                           "origin": problem["cache_origin"], **assembled})
        atomic_json(a.output / "assemblies.json", assemblies)
        print("assembly", problem["name"], problem["n"], problem["p"],
              assembled.get("status"), problem["cache_origin"], flush=True)
        if assembled.get("status") != "passed":
            raise RuntimeError(f"assembly failed for {problem['name']} n={problem['n']} p={problem['p']}")

        active = deepcopy(problem["candidates"])
        random.Random(a.seed + problem["n"] + problem["p"] + sum(map(ord, problem["name"]))).shuffle(active)
        for candidate in active:
            key = f"{problem['name'].lower()}_n{problem['n']}_p{problem['p']}_{candidate['candidate']}"
            module = ("scripts.adr_native_hp_worker" if candidate["family"] == "native_hp"
                      else "scripts.adr_solver_comparison_worker")
            record = run({**common, **candidate, "case": problem["case"], "n": problem["n"],
                          "p": problem["p"], "cache": problem["cache"]}, module, key, a.timeout)
            row = {**record, "name": problem["name"], "case": problem["case"],
                   "n": problem["n"], "p": problem["p"], "triangles": problem["triangles"],
                   "trace_dofs": problem["trace_dofs"], "family": candidate["family"],
                   "candidate": candidate["candidate"]}
            records.append(row)
            atomic_json(a.output / "summary.json", records)
            write_csv(a.output / "scaling.csv", records)
            print(key, row.get("status"), row.get("fresh_setup_solve_median_ms"), flush=True)

    failures = [{"name": row["name"], "n": row["n"], "p": row["p"],
                 "candidate": row["candidate"], "status": row.get("status")}
                for row in records if row.get("status") != "passed"]
    completion = {"status": "completed", "systems": len(problems), "attempted": len(records),
                  "passed": sum(row.get("status") == "passed" for row in records),
                  "failures": failures,
                  "note": "completed means every frozen curve was attempted at every scaling point."}
    atomic_json(a.output / "completion.json", completion)
    print(completion, flush=True)


if __name__ == "__main__":
    main()
