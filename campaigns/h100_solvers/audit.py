"""Audit portable ADR replay coverage and cost before choosing the Q2 campaign.

Read-only, standard-library-only planning. Estimates are conservative and must
be calibrated on the target device; they are not measured capacity limits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from campaigns.adr_unified.adr_unified_plan import build_plan, memory_estimate, problem_class, read_inventory
from campaigns.common.production_placement import resources

ROOT = Path(__file__).resolve().parents[2]


def audit(inventory_path, *, l5_triangles=400000):
    plan = build_plan(read_inventory(inventory_path), l5_triangles=l5_triangles)
    classes = defaultdict(list)
    for system in plan["systems"]:
        classes[problem_class(system)].append(system)
    rows = []
    for key, systems in sorted(classes.items()):
        candidates = {choice["effective_solver_sha256"]: choice
                      for system in systems for choice in system["candidates"]}
        estimates = [memory_estimate(system, choice) for system in systems for choice in system["candidates"]]
        rows.append(dict(geometry=key[0], case=key[1], stress_parameters_sha256=key[2],
            systems=len(systems), degrees=sorted({s["problem"]["p"] for s in systems}),
            iterative_trials=sum(len(s["candidates"]) for s in systems),
            candidate_families=dict(Counter(c["solver"]["family"] for c in candidates.values())),
            source_system_ids=[s["id"] for s in systems],
            planned_device_gib_max=max(max(e["solver_device_bytes"], e["assembly_device_bytes"]) for e in estimates) / 2**30,
            planned_host_gib_max=max(max(e["solver_host_bytes"], e["assembly_host_bytes"]) for e in estimates) / 2**30,
            planned_cache_gib=sum(memory_estimate(s, s["candidates"][0])["cache_disk_bytes"] for s in systems) / 2**30))
    jobs = plan["coverage"]["solver_jobs"]
    return dict(schema_version=1, launch_ready=False, inventory_sha256=plan["inventory_sha256"],
        l5_triangles=l5_triangles, coverage=plan["coverage"],
        dossier=dict(approximate_trials=250, gpu_hours=12, assumed_minutes_per_trial=3),
        full_replay_hours_at_dossier_trial_rate=jobs * 3 / 60,
        full_replay_duration_is_measured=False,
        duration_note="This is iterative-trial count x an assumed 3 minutes; it is not a measured host/device breakdown.",
        production_campaigns=dict(
            device=dict(resources("device"), iterative_trials=jobs, pardiso_trials=0,
                        cpu_assembly_fallback_allowed=False),
            host=dict(resources("host"), input_preparation_jobs=len(plan["systems"]),
                      pardiso_trials=len(plan["systems"]), on_gpu_nodes_allowed=False),
            smoke_exception="Existing smoke jobs retain in-allocation PARDISO checks."),
        estimated_cache_gib=sum(r["planned_cache_gib"] for r in rows),
        estimates_are_measured=False,
        gaps=["This inventory contains ADR iterative candidates only; Poisson, AR and matched PARDISO trials must be added.",
              "Archived stress cases cover annular p6 systems; square stress and p2/p4 coverage need explicit definitions.",
              "Four-size production ladders and memory limits require target-device calibration.",
              "The full replay exceeds the dossier trial count; select and document coverage before submission.",
              "Persist identical matrix/RHS hashes for every iterative and direct comparison; retain failures."],
        classes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "run_configs/adr_unified_l5/inventory.json")
    parser.add_argument("--l5-triangles", type=int, default=400000)
    args = parser.parse_args()
    print(json.dumps(audit(args.inventory, l5_triangles=args.l5_triangles), indent=2))


if __name__ == "__main__":
    main()
