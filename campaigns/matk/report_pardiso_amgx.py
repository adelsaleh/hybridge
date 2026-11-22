#!/usr/bin/env python3
"""Report measured PARDISO analysis and AMGX timings without adding nested timers."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
MARKER = re.compile(r"\[PARDISO native\s*\|\s*phase\s+11\]\s*(BEGIN|END)")
TIMING = re.compile(rf"^\s*(Time spent.*?|Total time spent)\s*:\s*({NUMBER})\s*s\s*$", re.I)


def numeric(value):
    return value if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) else None


def first_number(*values):
    return next((number for value in values if (number := numeric(value)) is not None), None)


def parse_phase11(text):
    """Preserve complete labels; phase totals and component sums remain separate."""
    blocks, current = [], None
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = MARKER.search(line)
        if marker:
            if marker.group(1) == "BEGIN":
                if current is not None:
                    blocks.append(current)
                current = dict(begin_line=line_number, end_line=None, complete=False, timings=[])
            elif current is not None:
                current.update(end_line=line_number, complete=True)
                blocks.append(current)
                current = None
            continue
        match = TIMING.match(line) if current is not None else None
        if match:
            label, seconds = match.group(1).strip(), float(match.group(2))
            current["timings"].append(dict(label=label, seconds=seconds, line=line_number,
                                           is_total=label.lower() == "total time spent"))
    if current is not None:
        blocks.append(current)
    for block in blocks:
        components = [row["seconds"] for row in block["timings"] if not row["is_total"]]
        totals = [row["seconds"] for row in block["timings"] if row["is_total"]]
        block["component_sum_seconds"] = sum(components) if components else None
        block["reported_total_seconds"] = totals[-1] if totals else None
    return blocks


def read_json(path):
    return json.loads(path.read_text())


def resolve_worker_path(raw, source):
    if not raw:
        return None
    path = Path(raw)
    for candidate in (path, source.parent / path, source.parent / path.name):
        if candidate.exists():
            return candidate.resolve()
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def worker_snapshots(paths):
    """Read sweep outcomes plus available worker snapshots of unfinished candidates."""
    found = {}
    for path in paths:
        data = read_json(path)
        if not isinstance(data.get("results"), list):
            entries = [(data, path)]
        else:
            entries = []
            recorded = set()
            for outcome in data["results"]:
                worker_path = resolve_worker_path(outcome.get("report"), path)
                worker = read_json(worker_path) if worker_path and worker_path.exists() else {}
                # Parent outcomes (timeout, killed worker, etc.) override stale
                # worker stages; missing detail can still come from the worker.
                entries.append((dict(worker, **outcome), worker_path))
                recorded.add(outcome.get("candidate"))
            directory = path.parent / (path.stem + "_trials")
            for name in data.get("planned_candidates", []):
                worker_path = directory / f"{name}.json"
                if name not in recorded and worker_path.exists():
                    worker = read_json(worker_path)
                    if data.get("status") == "interrupted":
                        worker.update(last_stage=worker.get("status"), status="interrupted",
                                      signal=data.get("signal"))
                    entries.append((worker, worker_path.resolve()))
        for worker, worker_path in entries:
            key = (str(worker_path), worker.get("candidate"))
            if key not in found:
                found[key] = (worker, worker_path, [])
            found[key][2].append(str(path))
    return list(found.values())


def original_gate(record, tolerance):
    relative = numeric(record.get("relative_residual"))
    finite = record.get("finite")
    if relative is None or tolerance is None or not isinstance(finite, bool):
        return None
    return bool(finite and relative <= tolerance)


def summarize_worker(worker, worker_path, sources, pardiso_fingerprint):
    parameters = worker.get("parameters") or {}
    ordering = worker.get("ordering") or {}
    mode = ordering.get("method") or parameters.get("ordering")
    if mode is None:
        mode = "unknown"
    if worker.get("initial_guess") or parameters.get("initial_guess"):
        initial = "nonzero/checkpoint"
    elif "initial_guess" in parameters:
        initial = "zero"
    else:
        initial = "unknown"
    tolerance = numeric(worker.get("rtol"))
    warnings = []
    if initial != "zero":
        warnings.append("ZERO-START COMPARISON: checkpoint/nonzero or unverified initial guess; do not silently mix with zero-start trials.")
    fingerprint = worker.get("system_sha256")
    if fingerprint and pardiso_fingerprint and fingerprint != pardiso_fingerprint:
        warnings.append("Original system fingerprint differs from the PARDISO export report.")
    samples = []
    for sample_number, sample in enumerate(worker.get("samples", []), 1):
        passes = []
        for index, item in enumerate(sample.get("passes", []), 1):
            gate = original_gate(item, tolerance)
            if gate is not None and isinstance(item.get("passed"), bool) and gate != item["passed"]:
                warnings.append(f"Sample {sample_number} pass {index}: recorded passed flag disagrees with finite/original-residual target.")
            passes.append(dict(pass_number=item.get("pass_number", index),
                solve_seconds=numeric(item.get("solve_seconds")),
                rhs_reset_seconds=numeric(item.get("rhs_reset_seconds")),
                solution_recovery_seconds=numeric(item.get("solution_recovery_seconds")),
                original_residual_check_seconds=numeric(item.get("original_residual_check_seconds")),
                iterations=item.get("iterations"), native_status=item.get("solver_status", item.get("amgx_status")),
                original_relative_residual=numeric(item.get("relative_residual")),
                finite=item.get("finite"), reported_passed=item.get("passed"),
                original_residual_gate=gate))
        samples.append(dict(sample_number=sample_number, solve_seconds=numeric(sample.get("solve_seconds")),
            validation_seconds=numeric(sample.get("validation_seconds")), passes=passes,
            original_relative_residual=numeric(sample.get("relative_residual")),
            finite=sample.get("finite"), reported_passed=sample.get("passed"),
            original_residual_gate=original_gate(sample, tolerance)))
    fields = ("fingerprint_seconds", "preparation_seconds", "scaling_seconds", "block_preparation_seconds",
              "backend_initialization_seconds", "upload_seconds", "setup_seconds", "checkpoint_seconds",
              "cleanup_seconds", "worker_wall_seconds", "trial_wall_seconds")
    timings = {name: numeric(worker.get(name)) for name in fields}
    timings["matrix_load_seconds"] = numeric(worker.get("matrix", {}).get("load_seconds"))
    timings["ordering_load_seconds"] = numeric(ordering.get("load_seconds"))
    timings["ordering_application_seconds"] = numeric(ordering.get("application_seconds"))
    timings["ordering_validation_seconds"] = numeric(ordering.get("validation_seconds"))
    timings["legacy_ordering_application_validation_seconds"] = numeric(ordering.get("application_validation_seconds"))
    evidence = [row["original_residual_gate"] for row in samples]
    aggregate_gate = (False if any(value is False for value in evidence) else
                      True if evidence and all(value is True for value in evidence) else None)
    return dict(candidate=worker.get("candidate"), source_reports=sources,
        worker_report=str(worker_path) if worker_path else None, status=worker.get("status"),
        error=worker.get("error"),
        last_stage=worker.get("last_stage"), job_id=worker.get("machine", {}).get("slurm_job_id"),
        ordering_mode=mode, ordering=ordering, initial_guess_mode=initial,
        initial_guess=worker.get("initial_guess"), zero_start_protocol=(initial == "zero"),
        smoke_test=worker.get("smoke_test"), original_relative_target=tolerance,
        original_system_sha256=fingerprint,
        all_observed_samples_pass_original_target=aggregate_gate,
        amgx_library=worker.get("amgx_library"), gpu=worker.get("gpu"),
        configuration_sha256=worker.get("configuration_sha256"),
        specification=worker.get("specification"),
        configuration=(worker.get("specification") or {}).get("config"),
        equilibration=worker.get("equilibration"), parameters=parameters,
        timings=timings, samples=samples, warnings=warnings)


def shown(value, precision=6):
    number = numeric(value)
    return "--" if number is None else f"{number:.{precision}f}"


def gate_label(value):
    return "PASS" if value is True else "FAIL" if value is False else "--"


def render(report):
    lines = ["PARDISO analysis followed by native AMGX", "=" * 43,
             f"Snapshot: {report['generated_at']}", "Missing measurements: --. Runtime state is a report snapshot, not a scheduler query.", ""]

    def fields(rows, indent="  "):
        width = max(len(label) for label, _ in rows)
        lines.extend(f"{indent}{label:<{width}} : {value}" for label, value in rows)

    def table(headers, rows):
        if not rows:
            lines.append("  --")
            return
        rows = [[str(value) for value in row] for row in rows]
        widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
        for row in [headers, ["-" * width for width in widths], *rows]:
            lines.append("  " + "  ".join(f"{item:<{width}}" if i in (0, len(row) - 1) else f"{item:>{width}}"
                                           for i, (item, width) in enumerate(zip(row, widths))))

    p = report["pardiso"]
    lines.append("PARDISO phase 11: analysis/reordering (no numerical LU factorization)")
    fields([("Report", p["source_report"]), ("Native log", p["source_log"]),
            ("Status / job", f"{p['status']} / {p['job_id']}"),
            ("Effective threads", p["threads_effective"] if p["threads_effective"] is not None else "--"),
            ("Matrix load [s]", shown(p["matrix_load_seconds"])),
            ("Original-system fingerprint [s]", shown(p["fingerprint_seconds"])),
            ("Wrapper analysis [s]", shown(p["analysis_seconds"]))])
    lines.append("  Wrapper analysis includes wrapper index conversion and the native phase-11 call.")
    for index, block in enumerate(p["native_phase11_blocks"], 1):
        lines.append(f"\n  Native phase-11 block {index}: {'complete' if block['complete'] else 'INCOMPLETE'}")
        rows = [(item["label"], shown(item["seconds"])) for item in block["timings"]]
        if rows:
            fields(rows, indent="    ")
        fields([("Native component sum [s]", shown(block["component_sum_seconds"])),
                ("Native reported total [s]", shown(block["reported_total_seconds"]))], indent="    ")
    if not p["native_phase11_blocks"]:
        lines.append("  Native phase-11 timing block: --")
    for heading, values in (("IPARM before (one-based)", p["iparm_before"]), ("IPARM after (one-based)", p["iparm_after"])):
        lines.append(f"\n  {heading}")
        if isinstance(values, dict):
            pairs = [f"{key:>2}={value}" for key, value in sorted(values.items(), key=lambda row: int(row[0]))]
            widths = [max((len(pair) for pair in pairs[column::8]), default=0) for column in range(8)]
            lines.extend("    " + "  ".join(f"{pair:<{widths[column]}}" for column, pair in enumerate(pairs[i:i + 8]))
                         for i in range(0, len(pairs), 8))
        else:
            lines.append("    --")
    for trial in report["amgx_trials"]:
        lines.extend(["", "AMGX: " + str(trial["candidate"]), "-" * (6 + len(str(trial["candidate"])))])
        solver = (trial["configuration"] or {}).get("solver", {})
        child = solver.get("preconditioner", "--")
        child_name = child.get("solver", "--") if isinstance(child, dict) else child
        child_detail = (f"{child_name}; backend={child.get('block_ilu_backend', '--')}; fill={child.get('ilu_sparsity_level', '--')}"
                        if isinstance(child, dict) else str(child_name))
        fields([("Snapshot status / job", f"{trial['status']} / {trial['job_id']}"),
                ("Ordering", trial["ordering_mode"]), ("Initial guess", trial["initial_guess_mode"]),
                ("Outer solver / max iterations", f"{solver.get('solver', '--')} / {solver.get('max_iters', '--')}"),
                ("Restart", solver.get("gmres_n_restart", "--")),
                ("Native convergence / tolerance", f"{solver.get('convergence', '--')} / {solver.get('tolerance', '--')}"),
                ("Preconditioner", child_detail),
                ("Host scaling / right multiplier", f"{(trial['equilibration'] or {}).get('method', '--')} / {(trial['equilibration'] or {}).get('right_multiplier', '--')}"),
                ("Synthetic smoke", str(trial["smoke_test"])),
                ("Original relative target", "--" if trial["original_relative_target"] is None else f"{trial['original_relative_target']:.3e}"),
                ("Observed sample residual gates", gate_label(trial["all_observed_samples_pass_original_target"])),
                ("Worker report", trial["worker_report"] or "--"),
                ("AMGX library", (trial["amgx_library"] or {}).get("path", "--")),
                ("AMGX library SHA256", (trial["amgx_library"] or {}).get("sha256", "--")),
                ("Configuration SHA256", trial["configuration_sha256"] or "--")])
        for warning in trial["warnings"]:
            lines.append("  WARNING: " + warning)
        if trial.get("error"):
            fields([("Failure detail", trial["error"])])
        if not trial["samples"]:
            lines.append("  No completed solve sample: final original residual and pure solve time are unavailable.")
        timing = trial["timings"]
        labels = [("matrix_load_seconds", "Matrix load [s]"), ("fingerprint_seconds", "Original fingerprint [s]"),
                  ("preparation_seconds", "Preparation [s]"), ("ordering_load_seconds", "  ordering artifact load [s]"),
                  ("ordering_application_seconds", "  ordering application [s]"),
                  ("ordering_validation_seconds", "  ordering validation [s]"),
                  ("legacy_ordering_application_validation_seconds", "  legacy combined ordering/check [s]"),
                  ("scaling_seconds", "  scaling [s]"), ("block_preparation_seconds", "  block preparation [s]"),
                  ("backend_initialization_seconds", "AMGX backend initialization [s]"),
                  ("upload_seconds", "GPU upload [s]"), ("setup_seconds", "AMGX setup [s]"),
                  ("checkpoint_seconds", "Checkpoint writes [s]"),
                  ("cleanup_seconds", "AMGX cleanup [s]"),
                  ("worker_wall_seconds", "Worker wall incl. cleanup [s]"),
                  ("trial_wall_seconds", "Parent-observed trial wall [s]")]
        lines.append("")
        fields([(label, shown(timing[key])) for key, label in labels])
        lines.append("  Indented preparation subphases are nested; do not add them to Preparation.")
        for sample in trial["samples"]:
            lines.append(f"\n  Sample {sample['sample_number']}")
            fields([("Combined GPU solves [s]", shown(sample["solve_seconds"])),
                    ("Combined recovery + validation [s]", shown(sample["validation_seconds"])),
                    ("Original relative residual", "--" if sample["original_relative_residual"] is None else f"{sample['original_relative_residual']:.12e}"),
                    ("Original residual gate", gate_label(sample["original_residual_gate"]))], indent="    ")
            rows = [[item["pass_number"], shown(item["solve_seconds"]), shown(item["rhs_reset_seconds"]),
                     shown(item["solution_recovery_seconds"]), shown(item["original_residual_check_seconds"]),
                     item["iterations"] if item["iterations"] is not None else "--",
                     "--" if item["original_relative_residual"] is None else f"{item['original_relative_residual']:.12e}",
                     gate_label(item["original_residual_gate"]), item["native_status"] if item["native_status"] is not None else "--"] for item in sample["passes"]]
            table(["Pass", "Solve[s]", "RHS reset[s]", "Recovery[s]", "Orig.check[s]", "Iters", "Original rel.residual", "Gate", "Native status"], rows)
            lines.append("    Recovery/original-check detail is nested inside combined validation; pass solve times are inside combined solves.")
        lines.append("  Full configuration and source report paths are preserved in the adjacent JSON.")
    if not report["amgx_trials"]:
        lines.extend(["", "AMGX worker results: -- (no worker snapshot available yet)."])
    lines.extend(["", "Timing interpretation", "---------------------",
        "Native PARDISO component sum, native reported total and wrapper analysis are alternative views of overlapping work.",
        "They are never added together. No exact wrapper overhead is inferred by subtracting rounded/native wall timers.",
        "Preparation detail, sample solve totals and validation detail are nested; no double-counted end-to-end total is invented.",
        "Worker wall includes cleanup through its final report; parent trial wall additionally covers startup/wait/report collection.",
        "A native success/status or a smoke pass does not establish full-matrix convergence. Original residual evidence is shown separately."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pardiso-report", type=Path, required=True)
    parser.add_argument("--pardiso-log", type=Path, required=True)
    parser.add_argument("--amgx-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json; the aligned text report uses the same stem and .txt")
    source = read_json(args.pardiso_report)
    permutation = source.get("permutation", {})
    pardiso = dict(source_report=str(args.pardiso_report.resolve()), source_log=str(args.pardiso_log.resolve()),
        status=source.get("status"), job_id=source.get("machine", {}).get("slurm_job_id"),
        matrix_load_seconds=numeric(source.get("matrix", {}).get("load_seconds")),
        fingerprint_seconds=numeric(source.get("fingerprint_seconds")),
        analysis_seconds=first_number(source.get("analysis_seconds"), permutation.get("analysis_seconds")),
        threads_effective=permutation.get("threads_effective"),
        iparm_before=permutation.get("iparm_before"), iparm_after=permutation.get("iparm_after"),
        permutation=permutation, original_system_sha256=source.get("system_sha256"),
        native_phase11_blocks=parse_phase11(args.pardiso_log.read_text(errors="replace")))
    trials = [summarize_worker(worker, path, sources, pardiso["original_system_sha256"])
              for worker, path, sources in worker_snapshots([path.resolve() for path in args.amgx_reports])]
    protected = {path.resolve() for path in (args.pardiso_report, args.pardiso_log, *args.amgx_reports)}
    protected.update(Path(trial["worker_report"]).resolve() for trial in trials if trial["worker_report"])
    if args.output.resolve() in protected or args.output.with_suffix(".txt").resolve() in protected:
        parser.error("Output artifacts must not overwrite an input benchmark report, worker report, or native log")
    report = dict(format="matk_pardiso_amgx_timing_report", format_version=1,
                  generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                  pardiso=pardiso, amgx_trials=trials,
                  notes=["Times are measured snapshots; missing values are null/--.",
                         "Nested preparation/validation/solve timers are not added twice.",
                         "No exact unmeasured overhead or end-to-end total is inferred.",
                         "Full-matrix acceptance requires independently checked original residual and finite solution; status alone is insufficient."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in ((args.output, json.dumps(report, indent=2, allow_nan=False) + "\n"),
                          (args.output.with_suffix(".txt"), render(report))):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    print(args.output.resolve())
    print(args.output.with_suffix(".txt").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
