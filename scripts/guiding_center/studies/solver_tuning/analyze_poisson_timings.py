"""Summarize existing guiding-center Poisson logs without importing HYBRIDGE.

This is offline analysis: no CUDA initialization, compilation, or time stepping.
Endpoint metrics and all-stage metrics are deliberately kept separate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


METRICS = (
    "poisson_stage_count", "poisson_step_iterations",
    "poisson_step_time_assembly", "poisson_step_time_solve",
    "poisson_step_time_reconstruction", "poisson_step_wall_time",
    "linear_step_wall_time", "poisson_time", "poisson_time_rhs_assembly",
    "poisson_time_solve", "poisson_time_reconstruction", "poisson_solver_iterations",
    "poisson_detail_raw_assembly_cached_rhs_source_moments",
    "poisson_detail_raw_assembly_cached_rhs_fused_solve_flux_scatter",
    "poisson_detail_raw_assembly_cached_rhs_total",
    "poisson_detail_raw_assembly_solver_headline_unaccounted",
)
FLAGS = (
    "poisson_detail_raw_assembly_operator_reused",
    "poisson_detail_raw_assembly_rhs_only",
    "poisson_detail_solve_fb_hp_mg_hierarchy_reused",
    "poisson_detail_solve_fb_hp_mg_fallback",
    "poisson_detail_solve_amgx_hierarchy_reused",
    "poisson_detail_cupy_reconstruction_compact",
)
COARSE_PAIR = re.compile(
    r"Coarse AMGX: applications=(\d+) elapsed=([\d.eE+-]+)s; hierarchy=reused\s+"
    r"FB-HP-MG-PCG: hierarchy=(\w+) setup=([\d.eE+-]+)s "
    r"solve=([\d.eE+-]+)s iterations=(\d+)"
)


def provenance(path, data):
    return {"path": str(path.resolve()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def describe(values):
    values = sorted(v for v in values if numeric(v))
    if not values:
        return None
    return {"count": len(values), "min": values[0], "median": statistics.median(values),
            "mean": statistics.mean(values), "p95": values[math.ceil(.95*len(values))-1],
            "max": values[-1], "sum": math.fsum(values)}


def summarize(rows):
    stats = {key: describe([r.get(key) for r in rows]) for key in METRICS}
    shares = {}
    for numerator, denominator in (
        ("poisson_step_time_assembly", "poisson_step_wall_time"),
        ("poisson_step_time_solve", "poisson_step_wall_time"),
        ("poisson_step_time_reconstruction", "poisson_step_wall_time"),
        ("poisson_step_wall_time", "linear_step_wall_time"),
    ):
        pairs = [(r[numerator], r[denominator]) for r in rows
                 if numeric(r.get(numerator)) and numeric(r.get(denominator))]
        total = math.fsum(b for _, b in pairs)
        shares[numerator + "/" + denominator] = (
            math.fsum(a for a, _ in pairs)/total if total else None)
    stage_walls = {}
    for row in rows:
        labels, walls = row.get("poisson_stage_labels", []), row.get("poisson_stage_wall_times", [])
        if len(labels) != len(walls):
            raise ValueError(f"stage label/wall mismatch at step {row.get('step')}")
        for label, wall in zip(labels, walls):
            stage_walls.setdefault(label, []).append(wall)
    return {
        "rows": len(rows), "metrics": stats, "shares_of_summed_wall": shares,
        "stage_counts": dict(Counter(str(r.get("poisson_stage_count")) for r in rows)),
        "endpoint_flags": {k: dict(Counter(str(r.get(k)) for r in rows)) for k in FLAGS},
        "stage_wall_seconds": {k: describe(v) for k, v in stage_walls.items()},
    }


def analyze(path, first_step):
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    steps = [r for r in rows if r.get("phase") == "step"]
    if not steps:
        raise ValueError(f"no accepted-step records: {path}")
    if len({r["step"] for r in steps}) != len(steps):
        raise ValueError(f"duplicate step numbers: {path}")
    selected = [r for r in steps if r["step"] >= first_step]
    ordinary = [r for r in selected if not r.get("poisson_tau_retry_count", 0)]
    recovery = [r for r in selected if r.get("poisson_tau_retry_count", 0)]
    by_tau = {}
    for row in ordinary:
        by_tau.setdefault(str(row.get("poisson_tau", "unrecorded")), []).append(row)
    initial = next((r for r in rows if r.get("phase") == "initial"), {})
    result = {
        "source": provenance(path, data),
        "schemes": sorted({r.get("time_scheme", "unrecorded") for r in steps}),
        "recorded_steps": len(steps), "last_step": steps[-1]["step"],
        "last_time": steps[-1]["time"],
        "dt_from_records": describe([b["time"]-a["time"] for a, b in zip(rows, rows[1:])
                                     if b.get("step", -1) == a.get("step", -1)+1]),
        "first_poisson": {k: v for k, v in initial.items()
                          if k.startswith("first_poisson_time_") or k in (
                              "first_poisson_detail_solve_fb_hp_mg_setup_outer",
                              "first_poisson_detail_cupy_local_factors_bytes",
                              "first_poisson_detail_cupy_local_factors_trace_response_bytes")},
        "all_selected": summarize(selected), "ordinary": summarize(ordinary),
        "ordinary_by_tau": {k: summarize(v) for k, v in by_tau.items()},
        "startup": [{k: r.get(k) for k in ("step", "poisson_stage_count",
                     "poisson_stage_labels", "poisson_step_wall_time")} for r in steps
                    if r["step"] < first_step],
        "recovery": [{k: r.get(k) for k in ("step", "poisson_tau", "poisson_stage_count",
                      "poisson_step_wall_time", "poisson_tau_recovery_wall_time",
                      "poisson_stage_labels")} for r in recovery],
    }
    log = path.with_name(path.name.removesuffix("_timings.jsonl") + ".log")
    if log.exists():
        log_data = log.read_bytes()
        text = log_data.decode(errors="replace")
        matches = COARSE_PAIR.findall(text)
        reused = [(int(n), float(c), float(t)) for n, c, h, _, t, _ in matches
                  if h == "reused" and int(n) > 0]
        total = math.fsum(t for _, _, t in reused)
        result["console"] = {
            "source": provenance(log, log_data),
            "scope": "all matched reused nonzero-iteration solves in the console, including startup/retries",
            "triangles": sorted({int(n) for n in re.findall(r"triangles=(\d+)", text)}),
            "matched_solves": len(matches), "reused_solves": len(reused),
            "coarse_seconds_per_application": describe([c/n for n, c, _ in reused]),
            "coarse_fraction_of_summed_solve": math.fsum(c for _, c, _ in reused)/total if total else None,
            "hierarchy_builds": sum(h == "created" for _, _, h, _, _, _ in matches),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timings", nargs="+", type=Path, help="existing *_timings.jsonl files")
    parser.add_argument("--first-step", type=int, default=3, help="exclude steps 0–2 by default")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.first_step < 1:
        parser.error("--first-step must be positive")
    runs = [analyze(p, args.first_step) for p in args.timings]
    output = {"first_step": args.first_step,
              "units": "seconds unless metric names specify counts or bytes",
              "notes": ["Endpoint details do not represent all stages.",
                        "Console coarse cost includes adapter calls, copies, and synchronization.",
                        "Input hashes identify saved evidence, not the code version that generated it.",
                        "Ordinary excludes tau-recovery rows; all_selected retains them."],
              "analyzer": provenance(Path(__file__), Path(__file__).read_bytes()), "runs": runs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    fields = ["file", "scheme", "steps", "dt", "poisson_calls", "assembly_ms", "solve_ms",
              "reconstruction_ms", "poisson_wall_ms", "linear_step_ms", "iterations",
              "solve_share", "coarse_solve_share"]
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            stats = run["ordinary"]["metrics"]
            row = {"file": run["source"]["path"], "scheme": "/".join(run["schemes"]),
                   "steps": run["ordinary"]["rows"],
                   "dt": (run["dt_from_records"] or {}).get("median"),
                   "solve_share": run["ordinary"]["shares_of_summed_wall"][
                       "poisson_step_time_solve/poisson_step_wall_time"],
                   "coarse_solve_share": run.get("console", {}).get("coarse_fraction_of_summed_solve")}
            for label, key, scale in (
                ("poisson_calls", "poisson_stage_count", 1),
                ("assembly_ms", "poisson_step_time_assembly", 1000),
                ("solve_ms", "poisson_step_time_solve", 1000),
                ("reconstruction_ms", "poisson_step_time_reconstruction", 1000),
                ("poisson_wall_ms", "poisson_step_wall_time", 1000),
                ("linear_step_ms", "linear_step_wall_time", 1000),
                ("iterations", "poisson_step_iterations", 1),
            ):
                row[label] = scale*stats[key]["median"] if stats[key] else None
            writer.writerow(row)
            print(f"{row['scheme']}: {row['steps']} ordinary steps; "
                  f"Poisson wall median={row['poisson_wall_ms']} ms")


if __name__ == "__main__":
    main()
