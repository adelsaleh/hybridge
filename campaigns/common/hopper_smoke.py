"""Small Q1/Q2 campaigns. Plan on a login node; run only inside a GPU allocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from campaigns.common.timing_summary import format_rows, summarize
from campaigns.common.lifecycle import (
    CampaignStop, describe_release, log, release_device, stop_on_signals, stop_signal_name, tail, terminate_group,
)

ROOT = Path(__file__).resolve().parents[2]


def tasks(suite, output):
    """Each task runs in a fresh process; all optional Q1 libraries are required."""
    python = [sys.executable, "-u"]
    if suite == "assembly":
        tensor = "scripts/gpu/benchmark_advection_tsle_tensor.py"
        configurations = ",".join(f"{precision}/{contraction}/default/{solver}"
                                  for precision in ("float64", "float32")
                                  for contraction in ("cublas", "cutensor")
                                  for solver in ("coop", "cublas", "magma"))
        return [
            ("tensor-parity", python + ["-m", "pytest", "-q", "tests/test_advection_tsle_tensor.py",
                                       "tests/test_advection_tsle_bsr.py"], {}, 900),
            ("transport-fp64", python + [tensor, "--mesh-size", "0.1", "--orders", "4,9", "--repeats", "2",
                "--native", "fused,split3", "--configs", configurations, "--output-json", str(output / "transport.json")],
                {"HYBRIDGE_PRECISION": "float64"}, 900),
            ("transport-fp32", python + [tensor, "--mesh-size", "0.1", "--orders", "4,9", "--repeats", "2",
                "--native", "fused,split3", "--configs", "none", "--output-json", str(output / "transport-fp32.json")],
                {"HYBRIDGE_PRECISION": "float32"}, 900),
            ("poisson-assembly", python + ["scripts/gpu/benchmark_poisson_assembly.py", "--nx", "8", "--ny", "8",
                "--orders", "2,6,9", "--repeats", "2", "--output", str(output / "poisson.json")], {}, 900),
            ("adr-tensor", python + ["scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_raw_cuda.py",
                "--nx", "8", "--orders", "2", "6", "--warmup", "1", "--repeats", "2",
                "--block-sizes", "128", "--reconstruction", "--output", str(output / "adr.jsonl")], {}, 900),
            ("nsight", python + ["scripts/gpu/capture_raw_assembly_profiles.py", "--mode", "both",
                "--cases", "poisson", "--orders", "2", "--sizes", "8", "--bases", "legendre-modal",
                "--formats", "bsr", "--cache-policies", "none", "--phases", "assembly", "--repeats", "2",
                "--output", str(output / "nsight")], {}, 600),
        ]
    result = [("transport-solver", python + ["-m", "pytest", "-q",
               "tests/test_cupy_backend.py::test_advection_reaction_raw_cuda_amgx_solver_smoke"], {}, 300)]
    # Well-conditioned smoke cases test all implementations, with LU on exactly
    # the same cached operator. Hard stress failures belong in the later study.
    # The annulus runs at ~100k triangles (97,002; neck screen passes) so the iterative
    # solvers are in their converging regime; weak ones may still stop as not_converged.
    # Per-job and task timeouts allow ~1 min per 2,000-iteration solve at that size.
    for geometry, orders, triangles, job_timeout, task_timeout in (
            ("square", ["2", "6"], "2000", "120", 1200), ("annulus", ["2"], "100000", "900", 3600)):
        result.append((f"adr-{geometry}", python + ["campaigns/adr_stress/run_closed_loop_stress.py",
            "--geometry", geometry, "--levels", "entry", "--variants", "orthogonal", "--epsilon", "0.1",
            "--speed", "1", "--triangles", triangles, "--max-triangles", "110000", "--orders", *orders,
            "--assembly-backend", "cupy",
            "--numba-threads", os.environ.get("SLURM_CPUS_PER_TASK", "16"), "--warmup", "1", "--repeats", "2",
            "--pardiso-all", "--pardiso-threads", os.environ.get("SLURM_CPUS_PER_TASK", "16"),
            "--timeout", job_timeout, "--skip-profiles", "--execute", "--output", str(output / f"adr-{geometry}")],
            {}, task_timeout))
    # Exercise the reusable Poisson solvers on their bounded, matched-system
    # regressions. The diocotron ladder and its saved meshes belong to the
    # research repository's gc_h200 campaign.
    parity = "tests/test_diffusion_reaction_assembly_parity.py::"
    result.append(("poisson-solvers", python + ["-m", "pytest", "-q",
        parity + "test_diffusion_raw_cuda_compressed_amgx_full_solve_stays_device_resident",
        parity + "test_fb_hp_mg_reuses_all_fixed_poisson_state_and_supports_periodic_rt_flux[6]",
        parity + "test_fb_hp_mg_runtime_gate_falls_back_once_and_reuses_hybrid_amgx"], {}, 1200))
    return result


def log_timings(name, command):
    """Print the timings a finished task wrote, so the Slurm log shows them as the smoke runs."""
    for flag in ("--output-json", "--output"):
        if flag in command and command.index(flag) + 1 < len(command):
            artifact = Path(command[command.index(flag) + 1])
            if artifact.is_file() and artifact.suffix in (".json", ".jsonl"):
                try:
                    lines = format_rows(summarize(artifact))
                except (OSError, ValueError) as error:
                    lines = [f"  summary unavailable ({type(error).__name__})"]
                for line in lines:
                    print(f"timings {name}{line}", flush=True)
            elif (artifact / "completion.json").is_file():
                completion = json.loads((artifact / "completion.json").read_text())
                print(f"timings {name}: {completion.get('passed')}/{completion.get('attempted')} solver jobs passed, "
                      f"{len(completion.get('not_converged', []))} not converged; per-job times in {artifact / 'events.jsonl'}",
                      flush=True)
            return


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status"))
    parser.add_argument("suite", choices=("assembly", "solvers"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT / f"run_outputs/campaigns/h100_{args.suite}/smoke").resolve()
    if args.action == "status":
        for path in sorted(output.glob("attempt_*/completion.json")):
            print(path, path.read_text())
        return 0
    if args.action == "plan":
        for name, command, env, timeout in tasks(args.suite, output):
            print(f"{name} (timeout {timeout}s): {shlex.join([f'{k}={v}' for k,v in env.items()] + command)}")
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run requires a Slurm GPU allocation; use plan on the login node")
    # Every attempt is separate so interrupted/failed runs and Nsight captures
    # remain intact. No done marker is reused for different sources.
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = 1
        while (output / f"attempt_{attempt}").exists():
            attempt += 1
        work = output / f"attempt_{attempt}"
        work.mkdir()
        from scripts.reports.run_provenance import source_snapshot, run_provenance
        sources = [p.relative_to(ROOT) for folder in ("hybridge", "scripts", "campaigns", "vendor", "tests", "run_configs")
                   for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".sbatch", ".json", ".jsonl", ".npz")]
        snapshot = source_snapshot(ROOT, sources, work / "source.tar.gz")
        atomic_json(work / "provenance.json", dict(run_provenance(ROOT), source=snapshot,
                    slurm_job=os.environ["SLURM_JOB_ID"], cpu_affinity=sorted(os.sched_getaffinity(0))))
        rows, stop_reason = [], None
        commands = [("stack", [sys.executable, "-m", "campaigns.common.hopper_stack",
                              "--output", str(work / "stack.json")], {}, 120)] + tasks(args.suite, work)
        log(f"{args.suite} smoke attempt {attempt}, job {os.environ['SLURM_JOB_ID']}: {len(commands)} tasks; records {work}")
        baseline = release_device()
        log(f"before first task: {describe_release(baseline)}")
        with stop_on_signals():
            try:
                if baseline.get("released") is False:
                    stop_reason = "device_not_released"
                for name, command, overrides, timeout in commands if stop_reason is None else ():
                    log(f"Starting {name} (timeout {timeout}s); log: {work / (name + '.log')}")
                    started = time.monotonic()
                    row = dict(task=name, command=command, environment=overrides, status="running")
                    atomic_json(work / f"{name}.result.json", row)
                    rows.append(row)
                    process = None
                    try:
                        with (work / f"{name}.log").open("w") as task_log:
                            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "HYBRIDGE_PRECISION": "float64", **overrides},
                                                       stdout=task_log, stderr=subprocess.STDOUT, start_new_session=True)
                            while True:
                                try:
                                    code = process.wait(timeout=min(60, max(timeout - (time.monotonic() - started), 0.1)))
                                    break
                                except subprocess.TimeoutExpired:
                                    if time.monotonic() - started >= timeout:
                                        raise
                                    log(f"{name}: running {time.monotonic() - started:.0f}s of {timeout}s")
                        row.update(returncode=code, status="passed" if code == 0 else "failed")
                        if stop_signal_name(code):
                            row.update(status="interrupted", stop_signal=stop_signal_name(code))
                    except subprocess.TimeoutExpired:
                        row.update(status="failed", error="timeout")
                    except CampaignStop as error:
                        row.update(status="interrupted", stop_signal=error.signal_name)
                        raise
                    finally:
                        if process:
                            row["worker_exit"] = terminate_group(process)
                        row["wall_seconds"] = time.monotonic() - started
                        row["device"] = release_device()
                        if name == "nsight" and row["status"] == "failed" and "ERR_NVGPUCTRPERM" in (work / "nsight.log").read_text():
                            row.update(status="blocked", error="Cluster denies GPU performance counters; timing smoke remains eligible")
                        atomic_json(work / f"{name}.result.json", row)
                        log(f"{name}: {row['status']} in {row['wall_seconds']:.1f}s, {describe_release(row['device'])}")
                        if row["status"] == "failed":
                            print(tail(work / f"{name}.log"), flush=True)
                        elif row["status"] == "passed":
                            log_timings(name, command)
                    if name == "stack" and row["status"] != "passed":
                        stop_reason = "stack_failed"
                        break
                    if row["device"].get("released") is False:
                        stop_reason = "device_not_released"
                        break
            except CampaignStop as error:
                stop_reason = error.signal_name
            finally:
                from scripts.reports.run_provenance import changed_snapshot_files
                changed = changed_snapshot_files(ROOT, snapshot)
                passed = (stop_reason is None and len(rows) == len(commands) and not changed
                          and all(r["status"] == "passed" or (r["task"] == "nsight" and r["status"] == "blocked") for r in rows))
                status = ("passed" if passed else "interrupted" if stop_reason and stop_reason.startswith("SIG") else "failed")
                completion = dict(status=status, stop_reason=stop_reason, tasks=rows, source_changes=changed,
                                  remaining=[name for name, *_ in commands[len(rows):]])
                atomic_json(work / "completion.json", completion)
                log(f"{args.suite} smoke {status}" + (f" ({stop_reason})" if stop_reason else "")
                    + f"; {len(rows)}/{len(commands)} tasks run; completion {work / 'completion.json'}")
        return 0 if status == "passed" else 3 if status == "interrupted" else 1

if __name__ == "__main__":
    sys.exit(main())
