"""Stress-study progress and saved-result inspection, without solver imports."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache, partial
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
from time import monotonic


@lru_cache(maxsize=1)
def _recorder():
    # Loading this I/O module directly keeps hdgfem's numerical __init__ out of
    # the planning/status process, as with the campaign's pure common helpers.
    path = Path(__file__).resolve().parents[3]/"hdgfem/io/records.py"
    spec = importlib.util.spec_from_file_location("_stress_event_records", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.append_jsonl_record


def event(output, name, *, message=None, **fields):
    """Persist a timestamped parent-process event and optionally print its label."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = dict(timestamp_utc=timestamp, event=name, pid=os.getpid(), **fields)
    _recorder()(Path(output)/"events.jsonl", row)
    if message is not None:
        print(f"[{timestamp}] {message}", flush=True)
    return row


@contextmanager
def timed_phase(output, name, **fields):
    """Record elapsed preparation time, including errors and interruptions."""
    start = monotonic()
    label = " ".join(f"{key}={value}" for key, value in fields.items())
    event(output, name+"_started", message=f"{name} started {label}", **fields)
    details = {}
    try:
        yield details
    except BaseException as exc:
        wall = monotonic()-start
        event(output, name+"_failed", message=f"{name} failed {label} wall={wall:.3f}s: {exc}",
              wall_seconds=wall, error=str(exc), **fields)
        raise
    else:
        wall = monotonic()-start
        event(output, name+"_finished", message=f"{name} finished {label} wall={wall:.3f}s "+json.dumps(details),
              wall_seconds=wall, **fields, **details)


def read_json(path):
    """Read an atomically updated artifact, exposing corrupt records as errors."""
    try:
        result = json.loads(Path(path).read_text())
        if not isinstance(result, dict):
            raise ValueError("expected a JSON object")
        return result
    except (OSError, ValueError) as exc:
        return dict(status="artifact_error", error=f"{path}: {exc}")


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _format(value, precision=".3e"):
    return format(value, precision) if _finite(value) else "unavailable"


def result_details(result, spec):
    """Summarize observations without treating warmups as measured repetitions."""
    observations = []
    for phase in ("warmups", "samples"):
        for setup_index, sample in enumerate(result.get(phase, []), 1):
            for solve_index, solve in enumerate(sample.get("solves", []), 1):
                observations.append((phase, setup_index, solve_index, sample, solve))
    details = dict(status=result.get("status", "unknown"), reasons=[],
                   warmup_setups=len(result.get("warmups", [])),
                   measured_setups=len(result.get("samples", [])),
                   requested_measured_setups=spec.get("repeats"),
                   completed_solves=len(observations), physical_rtol=spec.get("rtol"),
                   internal_rtol=spec.get("internal_rtol"), maxiter=spec.get("maxiter"),
                   returncode=result.get("returncode"))
    if observations:
        phase, setup_index, solve_index, sample, solve = observations[-1]
        details.update(last_phase=phase, setup_index=setup_index, solve_index=solve_index,
                       iterations=solve.get("iterations"), solver_status=solve.get("status"),
                       true_relative_residual=solve.get("true_relative_residual"),
                       l2_error=solve.get("l2_error"), trace_reference_error=solve.get("trace_reference_error"),
                       setup_ms=sample.get("setup_ms"), solve_ms=solve.get("solve_ms"),
                       setup_stages_ms=sample.get("setup_stages", {}),
                       fresh_setup_solve_ms=sample.get("fresh_setup_solve_ms"))
        measured = result.get("samples", [])
        if measured:
            setups = [s["setup_ms"] for s in measured if _finite(s.get("setup_ms"))]
            details["setup_median_ms"] = statistics.median(setups) if setups else None
            reused = [s["solve_ms"] for row in measured for s in row.get("solves", [])[1:] if _finite(s.get("solve_ms"))]
            details["reused_solve_mean_ms"] = statistics.mean(reused) if reused else None
        history = solve.get("residual_history", [])
        finite_history = [value for value in history if _finite(value)]
        if history:
            first, last = history[0], history[-1]
            details["history"] = dict(kind=solve.get("residual_history_kind", "reported residuals"),
                                      count=len(history), initial=first,
                                      best=min(finite_history) if finite_history else None, final=last,
                                      nonfinite_count=len(history)-len(finite_history), tail=history[-4:],
                                      reduction=first/last if _finite(first) and _finite(last) and last > 0 else None)
            if len(finite_history) != len(history):
                details["reasons"].append("nonfinite or unavailable residual-history values")
        cap, iterations = spec.get("maxiter"), solve.get("iterations")
        if solve.get("passed") is False and _finite(cap) and _finite(iterations) and iterations >= cap:
            details["reasons"].append("iteration cap reached")
        residual, target = solve.get("true_relative_residual"), spec.get("rtol")
        if _finite(residual) and _finite(target) and target > 0:
            details["residual_over_target"] = residual/target
            if residual > target:
                details["reasons"].append("physical residual above target")
        elif "true_relative_residual" in solve and not _finite(residual):
            details["reasons"].append("nonfinite or unavailable physical residual")
        bound, error = spec.get("l2_bound"), solve.get("l2_error")
        if _finite(bound) and _finite(error) and error > bound:
            details["reasons"].append("manufactured L2 bound exceeded")
        if _finite(solve.get("trace_reference_error")) and solve["trace_reference_error"] > max(1e-8, 100*(spec.get("rtol") or 0)):
            details["reasons"].append("CPU reference disagreement")
        if solve.get("passed") is False and not details["reasons"]:
            details["reasons"].append("solver convergence or validation check failed")
    elif isinstance(result.get("validation"), dict):
        for key in ("true_relative_residual", "l2_error", "trace_reference_error"):
            if key in result["validation"]:
                details[key] = result["validation"][key]
    if result.get("error"):
        details["error"] = result["error"]
        details["reasons"].append("profiling warmup did not converge" if "profiling warmup did not converge" in str(result["error"]) else str(result["error"]))
    if result.get("status") in ("timeout", "process_error", "assembly_failed", "artifact_error"):
        details["reasons"].append(result["status"])
    if result.get("kind") == "assembly":
        timings = result.get("assembly_samples_ms", [])
        details["assembly_stages_ms"] = timings[-1] if timings else {}
        details["space_setup_ms"] = result.get("space_setup_ms")
        details["assembly_median_ms"] = result.get("assembly_median_ms")
        validation = result.get("validation", {})
        details["cpu_reference_checked"] = "reference_residual_norm" in validation
        details["cpu_gpu_assembly_checked"] = "cpu_gpu_blocks_relative_error" in validation
        details["numpy_numba_assembly_checked"] = "numpy_numba_blocks_relative_error" in validation
        for key in ("numba_warmup_ms", "numba_threads", "numba_threading_layer", "numba_fastmath"):
            if key in result:
                details[key] = result[key]
        details["reference_max_dofs"] = spec.get("reference_max_dofs")
        details["validation"] = validation
    return details


def artifact_details(result_path, spec, result=None):
    """Include AMGX profile sidecars and actual matrix dimensions when available."""
    result_path = Path(result_path)
    result = result if result is not None else read_json(result_path)
    sidecar = result_path.with_suffix(".instrumented.json")
    if spec.get("worker_kind") == "amgx_profile" and sidecar.exists():
        # Keep the profiling adapter's final status/error while exposing solves.
        result = dict(read_json(sidecar), **result)
    details = result_details(result, spec)
    if result.get("kind") == "assembly" and spec.get("cache"):
        rhs = Path(spec["cache"])/"system_rhs.npy"
        if rhs.exists():
            import numpy as np
            try:
                array = np.load(rhs, mmap_mode="r", allow_pickle=False)
                details["trace_dofs"] = int(array.size)
                details["face_rows"] = int(array.shape[0])
            except (OSError, ValueError) as exc:
                details["reasons"].append(f"cannot inspect cached RHS: {exc}")
    return details


def print_details(key, details, *, wall_seconds=None, log_path=None, reused=False, stream=None):
    """Render a compact, explicit failure and timing report for one job."""
    emit = partial(print, file=stream, flush=True)
    wall = f" wall={wall_seconds:.3f}s" if _finite(wall_seconds) else ""
    code = f" exit={details['returncode']}" if details.get("returncode") is not None else ""
    emit(f"{'reused' if reused else 'result'} {key}: {details['status']}{wall}{code}")
    if "iterations" in details:
        emit(f"  {details['last_phase']} setup={details['setup_index']} solve={details['solve_index']} "
              f"iterations={details['iterations']}/{details.get('maxiter')} solver={details.get('solver_status') or 'not reported'} "
              f"measured_setups={details['measured_setups']}/{details.get('requested_measured_setups')}")
    if "true_relative_residual" in details:
        emit(f"  physical_relres={_format(details.get('true_relative_residual'))} "
              f"target={_format(details.get('physical_rtol'))} internal={_format(details.get('internal_rtol'))} "
              f"residual/target={_format(details.get('residual_over_target'))} "
              f"L2={_format(details.get('l2_error'))} reference_error={_format(details.get('trace_reference_error'))}")
    if "solve_ms" in details:
        emit(f"  last_setup_ms={_format(details.get('setup_ms'), '.3f')} "
              f"last_solve_ms={_format(details.get('solve_ms'), '.3f')} "
              f"fresh_setup_solve_ms={_format(details.get('fresh_setup_solve_ms'), '.3f')} "
              f"measured_setup_median_ms={_format(details.get('setup_median_ms'), '.3f')} "
              f"measured_reused_mean_ms={_format(details.get('reused_solve_mean_ms'), '.3f')}")
    if "assembly_median_ms" in details:
        emit(f"  space_setup_ms={_format(details.get('space_setup_ms'), '.3f')} "
             f"assembly_median_ms={_format(details.get('assembly_median_ms'), '.3f')}")
    if "numba_warmup_ms" in details:
        emit(f"  Numba threads={details.get('numba_threads')} scheduler={details.get('numba_threading_layer')} "
             f"fastmath={details.get('numba_fastmath')} excluded_warmup_ms={_format(details['numba_warmup_ms'], '.3f')} "
             f"NumPy/Numba_assembly={'checked' if details.get('numpy_numba_assembly_checked') else 'NOT CHECKED'}")
    for label, field in (("assembly", "assembly_stages_ms"), ("setup", "setup_stages_ms")):
        if details.get(field):
            emit(f"  {label} stages (ms): "+" ".join(f"{name}={_format(value, '.3f')}" for name, value in details[field].items()))
    if "cpu_reference_checked" in details:
        emit(f"  trace_dofs={details.get('trace_dofs', 'unknown')} "
              f"CPU_reference={'checked' if details['cpu_reference_checked'] else 'NOT CHECKED'} "
              f"CPU/GPU_assembly={'checked' if details['cpu_gpu_assembly_checked'] else 'NOT CHECKED'} "
              f"reference_limit={details.get('reference_max_dofs')}")
    if details.get("history"):
        history = details["history"]
        emit(f"  history ({history['kind']}): n={history['count']} "
              f"initial={_format(history['initial'])} best={_format(history['best'])} final={_format(history['final'])} "
              f"reduction={_format(history['reduction'])} nonfinite={history['nonfinite_count']} "
              f"tail="+",".join(_format(v) for v in history["tail"]))
    if details.get("reasons"):
        emit("  reason: "+"; ".join(dict.fromkeys(details["reasons"])))
    if log_path is not None:
        emit(f"  log: {log_path}")


def campaign_status(output):
    """Inspect existing artifacts, including a live campaign, without writing."""
    output = Path(output)
    if not (output/"manifest.json").is_file():
        raise ValueError(f"No campaign manifest: {output}")
    counts = Counter()
    for specpath in sorted((output/"specs").glob("*.json")):
        spec = read_json(specpath)
        path = Path(spec.get("result", output/"jobs"/specpath.name))
        if not path.exists():
            print(f"pending {specpath.stem}", flush=True)
            counts["pending"] += 1
            continue
        result = read_json(path)
        details = artifact_details(path, spec, result)
        counts[("profile:" if "profile" in spec.get("worker_kind", "") else "")+details["status"]] += 1
        print_details(specpath.stem, details, wall_seconds=result.get("runner_wall_seconds"),
                      log_path=output/"logs"/(specpath.stem+".log"))
    # Direct diagnostics are separate from the iterative-worker specifications.
    for path in sorted((output/"jobs").glob("pardiso_*.json")):
        result = read_json(path)
        directory = Path(result["diagnostic_output"]) if result.get("diagnostic_output") else None
        if result.get("status") == "running" and directory is not None:
            result = dict(result, **read_json(directory/"result.json"))
        status = result.get("status", "unknown")
        counts["pardiso:"+status] += 1
        print(f"{path.stem}: {status} stage={result.get('stage', 'not reported')} "
              f"face_relres={result.get('face_relative_residual')} "
              f"probe_solution_error={result.get('probe_relative_solution_error')} "
              f"log={directory/'worker.log' if directory else 'unavailable'}", flush=True)
    print("Saved job states: "+json.dumps(dict(counts), sort_keys=True), flush=True)
    completion = output/"completion.json"
    if completion.exists():
        print("Campaign completion: "+json.dumps(read_json(completion), sort_keys=True), flush=True)
    else:
        print("No completion record; running artifacts alone do not prove a process is still alive.", flush=True)
    return 0
