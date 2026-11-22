#!/usr/bin/env python3
"""Run committed native ILU qualification stages on one allocated GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.lifecycle import CampaignStop, stop_on_signals, terminate_group
from campaigns.common.protocol import host_provenance

STAGES = ("smoke", "numerical", "racecheck", "synccheck", "reuse-racecheck")
SOURCE_FILES = (
    "campaigns/matk/qualify_ilu_synchronization.py",
    "campaigns/matk/ilu_qualification.sbatch",
    "campaigns/matk/check_ilu_synchronization.py",
    "campaigns/matk/check_ilu_capacity.py",
    "campaigns/matk/check_natural_iluk.py",
    "campaigns/matk/common.py",
    "campaigns/matk/benchmark_amgx_strong.py",
    "campaigns/common/lifecycle.py", "campaigns/common/protocol.py",
    "campaigns/common/hopper_env.sh", "scripts/gpu/run_cuda13.sh",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(args):
    out = args.output.resolve()
    library = args.build.resolve() / "libamgxsh.so"
    sanitizer = Path(os.environ.get("HYBRIDGE_CUDA13_ROOT", "/usr/local/cuda-13.1")) / "bin/compute-sanitizer"
    steps = []
    def add(name, command, tool=None, timeout=600, filters=()):
        report = out / (name + ".json")
        command = [sys.executable, "-B", "-u", *command, "--output", str(report)]
        if tool:
            prefix = [str(sanitizer), "--tool", tool, "--error-exitcode", "99"]
            if tool == "racecheck":
                prefix += ["--racecheck-report", "analysis"]
            for pattern in filters:
                prefix += ["--kernel-name", "kns=" + pattern]
            command = prefix + command
        steps.append(dict(name=name, command=command, tool=tool, timeout_seconds=timeout,
                          report=str(report), log=str(out / (name + ".log"))))
    def oracle(case=None, mode=None, layout=None, single=False, diagnostics=1):
        cmd = ["campaigns/matk/check_ilu_synchronization.py", "--library", str(library), "--verbosity", "3"]
        for option, value in (("--case", case), ("--mode", mode), ("--layout", layout)):
            if value:
                cmd += [option, value]
        if single:
            cmd += ["--diagnostics", str(diagnostics), "--single-setup"]
        return cmd
    def capacity(case=None, mode=None, layout=None):
        cmd = ["campaigns/matk/check_ilu_capacity.py", "--capacity", "1024",
               "--expected-library", str(library), "--verbosity", "3"]
        for option, value in (("--case", case), ("--mode", mode), ("--layout", layout)):
            if value:
                cmd += [option, value]
        return cmd
    for stage in args.stages:
        if stage == "smoke":
            for case in ("coupled", "external", "inverse", "nonfinite"):
                add("smoke-" + case, oracle(case, "dDDI", "ROW_MAJOR", True))
        elif stage == "numerical":
            add("numerical-capacity", capacity(), timeout=1200)
            add("numerical-oracles", oracle(), timeout=1800)
        elif stage in ("racecheck", "synccheck"):
            for mode in ("dDDI", "dFFI"):
                for layout in ("ROW_MAJOR", "COL_MAJOR"):
                    suffix = mode + "-" + layout
                    add(stage + "-bipartite129-" + suffix,
                        capacity("bipartite129", mode, layout), stage, 900)
                    if stage == "synccheck":
                        add(stage + "-star1024-" + suffix,
                            capacity("star1024", mode, layout), stage, 600)
                    cases = ("coupled", "external", "inverse") if stage == "racecheck" else (
                        "coupled", "external", "inverse", "nonfinite", "reuse")
                    for case in cases:
                        add(stage + "-" + case + "-" + suffix,
                            oracle(case, mode, layout, True), stage, 900)
                    # The default nullable-counter path has different generated
                    # control flow, so it needs sanitizer evidence of its own.
                    off_cases = ("coupled", "inverse") if stage == "racecheck" else (
                        "coupled", "inverse", "reuse")
                    for case in off_cases:
                        add(stage + "-" + case + "-diagnostics-off-" + suffix,
                            oracle(case, mode, layout, True, 0), stage, 900)
        else:
            for mode in ("dDDI", "dFFI"):
                for layout in ("ROW_MAJOR", "COL_MAJOR"):
                    for diagnostics in (0, 1):
                        add(stage + "-" + mode + "-" + layout + "-diagnostics-" + str(diagnostics),
                            oracle("reuse", mode, layout, True, diagnostics), "racecheck", 1800,
                            ("LU_forward_4x4_kernel_warp", "LU_backward_4x4_kernel_warp"))
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--build", type=Path, default=REPO.parent / "builds/AMGX/qualify-ilu")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get(
        "HYBRIDGE_CAMPAIGN_OUT", REPO / "run_outputs/matk/ilu-qualification")))
    args = parser.parse_args()
    steps = plan(args)
    if args.plan:
        print(json.dumps(dict(gpu_count=1, build=str(args.build), steps=steps), indent=2))
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {name: sha(REPO / name) for name in SOURCE_FILES}
    library = args.build.resolve() / "libamgxsh.so"
    identity_inputs = dict(sources=sources, library=str(library), library_sha256=sha(library), steps=steps)
    identity = hashlib.sha256(json.dumps(identity_inputs, sort_keys=True).encode()).hexdigest()
    marker = out / "source-identity.txt"
    if marker.exists() and marker.read_text().strip() != identity:
        raise SystemExit("Qualification source, library, or plan changed; use a new output directory.")
    marker.write_text(identity + "\n")
    (out / "sources.json").write_text(json.dumps(identity_inputs, indent=2) + "\n")
    device = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                            check=True, text=True, capture_output=True).stdout.strip()
    cuda = subprocess.run([str(Path(os.environ["HYBRIDGE_CUDA13_ROOT"]) / "bin/nvcc"), "--version"],
                          check=True, text=True, capture_output=True).stdout.strip()
    host_provenance(out, device=device, cuda=cuda, qualification=identity_inputs,
                    build_provenance={name: str(args.build.resolve() / name)
                                      for name in ("plan.json", "provenance.jsonl", "COMPLETE")})
    completion = dict(status="running", source_identity=identity, requested_steps=len(steps), results=[])
    def save():
        (out / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    save()
    try:
        with stop_on_signals():
            for step in steps:
                done = out / (step["name"] + ".done")
                if done.exists():
                    completion["results"].append(json.loads(done.read_text()))
                    continue
                assert sources == {name: sha(REPO / name) for name in SOURCE_FILES}, "Qualification sources changed"
                assert identity_inputs["library_sha256"] == sha(library), "Qualification library changed"
                print("[ILU qualification] Starting " + step["name"], flush=True)
                started = time.monotonic()
                with Path(step["log"]).open("w") as stream:
                    child = subprocess.Popen(step["command"], cwd=REPO, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        returncode = child.wait(timeout=step["timeout_seconds"])
                    finally:
                        terminate_group(child)
                log = Path(step["log"]).read_text()
                observed = json.loads(Path(step["report"]).read_text()) if Path(step["report"]).exists() else {}
                passed = returncode == 0 and observed.get("status") == "passed"
                if step["tool"] == "racecheck":
                    summaries = re.findall(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", log)
                    passed = passed and bool(summaries) and all(all(int(x) == 0 for x in row) for row in summaries)
                elif step["tool"] == "synccheck":
                    summaries = re.findall(r"ERROR SUMMARY: (\d+) errors", log)
                    passed = passed and bool(summaries) and all(int(x) == 0 for x in summaries)
                result = dict(name=step["name"], status="passed" if passed else "failed",
                              returncode=returncode, seconds=time.monotonic()-started,
                              report=step["report"], log=step["log"])
                completion["results"].append(result)
                save()
                if not passed:
                    raise RuntimeError(f"{step['name']} failed; inspect {step['log']}")
                done.write_text(json.dumps(result, indent=2) + "\n")
                print("[ILU qualification] Passed " + step["name"], flush=True)
        completion["status"] = "passed"
    except CampaignStop as error:
        completion.update(status="interrupted", reason=error.signal_name)
        raise
    except Exception as error:
        completion.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    try:
        main()
    except CampaignStop:
        raise SystemExit(130)
