"""Separate host/device production registries with shared-system dependencies.

This is inspection only. Worker dispatch, reduced trial selection and measured
memory preflight remain required; no submission or numerical work occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from campaigns.adr_unified.adr_unified_plan import build_plan, fingerprint, read_inventory
from campaigns.common.production_placement import resources

ROOT = Path(__file__).resolve().parents[2]


def build_campaigns(inventory_path, *, l5_triangles=400000):
    archive = build_plan(read_inventory(inventory_path), l5_triangles=l5_triangles)
    prepare, device, direct = [], [], []
    for system in sorted(archive["systems"], key=lambda s: s["id"]):
        key = system["id"]
        inputs = f"shared/inputs/{key}/manifest.json"
        saved = f"shared/systems/{key}/manifest.json"
        identity = dict(manifest=saved, required_hashes=["matrix_rhs_sha256", "topology_sha256"],
                        source_system_id=key)
        prepare_id, assembly_id = f"host-input-{key}", f"device-assemble-{key}"
        prepare.append(dict(id=prepare_id, stage="input_preparation", campaign="host", needs=[],
                            system=system, produces=inputs, resources=resources("host")))
        # The assembly here is the GPU path, with no CPU/Numba fallback. It
        # publishes immutable arrays once for both iterative and direct solvers.
        device.append(dict(id=assembly_id, stage="device_assembly", campaign="device", needs=[prepare_id],
                           inputs=inputs, produces=saved, system_id=key, assembly_backend="cupy",
                           host_fallback_allowed=False, resources=resources("device")))
        for choice in system["candidates"]:
            if "pardiso" in json.dumps(choice).lower():
                raise ValueError("CPU direct solver cannot appear in the device campaign")
            device.append(dict(id=f"device-solve-{key}-{choice['id']}", stage="iterative_solve", campaign="device",
                needs=[assembly_id], system_identity=identity, solver=choice,
                produces=f"device/results/{key}/{choice['id']}.json", resources=resources("device")))
        direct.append(dict(id=f"host-pardiso-{key}", stage="pardiso", campaign="host", needs=[assembly_id],
            system_identity=identity, produces=f"host/results/{key}/pardiso.json",
            protocol=archive["protocol"], resources=resources("host")))
    tasks = prepare + device + direct
    ids = {task["id"] for task in tasks}
    if len(ids) != len(tasks) or any(dependency not in ids for task in tasks for dependency in task["needs"]):
        raise ValueError("Invalid production task dependencies")
    for task in tasks:
        task["specification_sha256"] = fingerprint(task)
    return dict(schema_version=1, launch_ready=False, inventory_sha256=archive["inventory_sha256"],
        coverage=archive["coverage"],
        separation=dict(host_partition="genoa", host_on_gpu_nodes_allowed=False,
                        device_partition="hopper", inline_pardiso_on_device_allowed=False,
                        smoke_exception="Existing smoke jobs may run PARDISO in their GPU allocation.",
                        dependency_policy="Per-system CPU inputs -> GPU assembly -> independent GPU iterative and CPU PARDISO trials"),
        timing=dict(measured=False, device_iterative_trials=archive["coverage"]["solver_jobs"],
                    host_pardiso_trials=len(direct),
                    note="947 x 3 minutes is an assumed device-trial scenario; no host/device duration split has been measured."),
        stages=dict(Counter(task["stage"] for task in tasks)),
        required_before_launch=["passing smoke campaigns", "selected production trial matrix",
                                "measured memory limits", "worker dispatch and resumable stage runners",
                                "stage runners must call production_placement.require_placement before importing workers"],
        tasks=tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "run_configs/adr_unified_l5/inventory.json")
    parser.add_argument("--l5-triangles", type=int, default=400000)
    parser.add_argument("--campaign", choices=("host", "device", "all"), default="all")
    parser.add_argument("--scope", choices=("expanded", "archive"), default="expanded")
    args = parser.parse_args()
    if args.scope == "expanded":
        from campaigns.h100_solvers.expanded import design
        report = design(args.inventory)
        report["selection"] = args.campaign
        report["allowed_stages"] = {
            "host": ["input_preparation", "host_inverse_preparation", "pardiso"],
            "device": ["device_assembly", "iterative_solve", "reconstruction", "postprocessing"],
        }
        if args.campaign != "all":
            report["allowed_stages"] = {args.campaign: report["allowed_stages"][args.campaign]}
            report["selected_resources"] = resources(args.campaign)
        print(json.dumps(report, indent=2))
        return
    report = build_campaigns(args.inventory, l5_triangles=args.l5_triangles)
    if args.campaign != "all":
        report["tasks"] = [task for task in report["tasks"] if task["campaign"] == args.campaign]
        report["selection"] = args.campaign
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
