#!/usr/bin/env python3
"""Adapt one stress specification to the established, isolated campaign workers.

The branch owns assembly/ASM/BJ/AMGX; its native adapter selects master before
importing numerical modules. Only a pure NumPy case factory crosses this boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

# Direct-file entry point intentionally avoids importing master's scripts package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"cases"))
from closed_loop_stress_cases import StressParameters, make_case


def operator_hash(cache):
    import numpy as np

    digest = hashlib.sha256()
    for name in ("system_blocks", "system_rhs"):
        value = np.load(Path(cache)/(name+".npy"), mmap_mode="r", allow_pickle=False)
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def dispatch(spec, common):
    kind = spec["worker_kind"]
    if kind in ("native", "native_profile"):
        # No branch hdgfem import may precede this adapter.
        from scripts.adr_native_hp_worker import measure
        result = measure(spec, master_root=Path(spec["master_root"]))
        if kind == "native_profile" and result.get("status") == "passed" and not result.get("profile_validation", {}).get("passed"):
            result["status"] = "numerical_failure"
        return result
    if kind == "amgx_profile":
        from scripts.profile_adr_amgx_preconditioner import main
        sys.argv = [sys.argv[0], "--spec", spec["spec_path"], "--output", spec["result"]]
        main()
        return common.read_json(spec["result"])
    if kind == "compare":
        from scripts.adr_solver_comparison_worker import measure
        return measure(spec)
    from scripts.adr_performance_worker import assemble, profile
    if kind == "assemble":
        result = assemble(spec)
        result["operator_sha256"] = operator_hash(spec["cache"])
        return result
    if kind == "profile":
        # The profiler passes rtol directly to the solver; preserve the stricter
        # internal target, with a fresh independent physical residual afterwards.
        return profile(dict(spec, rtol=spec["internal_rtol"]))
    raise ValueError(f"Unknown worker kind: {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    spec["spec_path"] = str(args.spec.resolve())
    sys.path.insert(0, spec["branch_root"])
    os.environ["HDGFEM_PRECISION"] = "float64"
    from scripts.adr_performance_common import atomic_json, read_json
    from scripts import adr_performance_common as common

    try:
        mesh = Path(spec["mesh_path"])
        if hashlib.sha256(mesh.read_bytes()).hexdigest() != spec["mesh_sha256"]:
            raise ValueError("Mesh changed since preparation")
        if spec.get("expected_operator_sha256") and operator_hash(spec["cache"]) != spec["expected_operator_sha256"]:
            raise ValueError("Cached operator/RHS changed since assembly")
        from scripts.adv_diff_rea_cases import register_case
        parameters = StressParameters(**spec["stress_parameters"])
        register_case(spec["case"], lambda: make_case(parameters, spec["velocity_normalization"]))
        result = dispatch(spec, common)
    except (Exception, SystemExit) as exc:
        result = read_json(spec["result"]) if Path(spec["result"]).exists() else {}
        if result.get("status") in (None, "passed", "running"):
            result.update(status="error", error=str(exc), traceback=traceback.format_exc())
    result.update(stress_parameters=spec["stress_parameters"],
                  velocity_normalization=spec["velocity_normalization"], mesh_sha256=spec["mesh_sha256"])
    atomic_json(spec["result"], result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
