"""Selection and report of the D8-S smoke campaign (fastest correct device preset per degree); no solves.

Reads the launcher records (``runs.jsonl``: status, peak device memory, contamination, host load) and
the diocotron driver's summary (``runs/d8s_summary.json``: synchronized step times, their breakdown,
the cold first Poisson solve, solver health and conservation checks) of a study directory written by
``run_outputs/diocotron_order_smoke_20261016/launcher.py``.

Metrics per run:
- ``step``: median synchronized step time (``step_advance_wall_time``, steps 3..N);
- ``hard``: the step time with the warm Poisson Krylov time replaced by that of the cold first solve
  (zero initial guess); reported only. Ranking on it picked classical AMG at p = 7 and 9 (2026-10-16):
  a better cold solve bought with 0.5 s more per warm step, while every campaign step has the
  extrapolated potential as its initial guess;
- ``nonlinear_step`` (stage H): median step time over the steps with t >= ``NONLINEAR_FROM`` of the
  1000-step runs seeded at epsilon = 0.1 (saturation at t of about 15), where that initial guess is
  poorest; also the Poisson Krylov time and the iteration counts over the same steps.
Poisson candidates are ranked by ``nonlinear_step`` where stage H has run for the degree, otherwise by
``step`` (ties within 2 % by ``hard``); transport candidates by ``step``.

A run counts only if it finished, saw no foreign GPU process, used one transport attempt per stage and
no fb-hp-mg fallback or tau retry, reached its solver targets, conserved mass to 1e-11 and kept the SI-BDF2
L2 identity to 1e-12.

``select_presets(study)`` writes ``presets_selected.json``; ``smoke_report(study)`` writes
``smoke_runs.csv``, ``smoke_report.md`` and ``smoke_step_times.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TIE = 0.02
NONLINEAR_FROM = 15.0  # stage H: medians over the steps with t >= this (seed 0.1 saturates at t of about 15)
NONLINEAR_KEYS = {"step_advance_wall_time": "nonlinear_step", "poisson_krylov_time": "nonlinear_poisson_krylov",
                  "poisson_step_iterations": "nonlinear_poisson_iterations",
                  "transport_step_iterations": "nonlinear_transport_iterations"}


def _records(study: Path) -> dict:
    """Last attempt of every run id."""
    path = study / "runs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    last = {}
    for row in rows:
        last[row["run_id"]] = row
    return last


def _summary_rows(study: Path) -> dict:
    path = study / "runs" / "d8s_summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {row["prefix"]: row for row in data.get("rows", [])}


def _number(value):
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def runs(study) -> list[dict]:
    """Launcher records merged with the driver's summary row and the derived metrics."""
    study = Path(study)
    summary = _summary_rows(study)
    out = []
    for run_id, record in _records(study).items():
        row = next((value for prefix, value in summary.items() if prefix.endswith(f"_{run_id}_seed")), {})
        merged = dict(record, **{f"s_{key}": value for key, value in row.items()})
        step = _number(row.get("median_step_advance_wall_time"))
        warm = _number(row.get("median_poisson_krylov_time"))
        cold = _number(row.get("first_poisson_krylov_time"))
        if warm is None:
            warm = _number(row.get("median_poisson_step_time_solve"))
        if cold is None:
            cold = _number(row.get("first_poisson_time_solve"))
        merged["step"] = step
        merged["hard"] = None if None in (step, warm, cold) else step - warm + cold
        merged.update(_gpu_activity(study, record))
        if record.get("stage") == "H":
            merged.update(_nonlinear(study, record))
        merged["problems"] = _problems(record, row)
        merged["valid"] = not merged["problems"]
        out.append(merged)
    return out


def _nonlinear(study: Path, record: dict) -> dict:
    """Medians of the per-step records with t >= NONLINEAR_FROM of a stage-H run (the saturated phase)."""
    path = next((study / "runs").glob(f"*_{record['run_id']}_seed.jsonl"), None)
    if path is None:
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    late = [row for row in rows if _number(row.get("time")) is not None and row["time"] >= NONLINEAR_FROM]
    out = {"nonlinear_steps": len(late)}
    for key, name in NONLINEAR_KEYS.items():
        values = [row[key] for row in late if _number(row.get(key)) is not None]
        out[name] = float(np.median(values)) if values else None
    return out


def _gpu_activity(study: Path, record: dict) -> dict:
    """Median power and largest temperature while the GPU computes (utilization above 50 %), from the run's
    nvidia-smi samples. The SM clock that nvidia-smi reports lags on this card (510 MHz at 300 W), so the
    power stands for the device state instead."""
    path = study / "runs" / "gpu" / f"{record['run_id']}_attempt{record.get('attempt', 1)}.csv"
    if not path.exists():
        return {}
    power, temperature = [], []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                if float(row["utilization_gpu"]) > 50:
                    power.append(float(row["power"]))
                    temperature.append(float(row["temperature"]))
            except (KeyError, ValueError):
                continue
    if not power:
        return {}
    return dict(power_w_active=float(np.median(power)), temperature_c_active_max=max(temperature),
                active_seconds=2.0 * len(power))


def _problems(record: dict, row: dict) -> list[str]:
    problems = []
    if record.get("status") != "ok":
        problems.append(str(record.get("status")))
    if not row:
        return problems + ["no summary row"]
    if row.get("median_step_advance_wall_time") is None or not row.get("step_timing_synchronized"):
        problems.append("no synchronized step time")
    checks = (("max_transport_attempts", 1, "transport retries"), ("poisson_fallbacks", 0, "fb-hp-mg fallback"),
              ("poisson_tau_retries", 0, "tau retries"), ("max_poisson_residual_ratio", 1.0 + 1e-9, "Poisson target"),
              ("max_transport_residual_ratio", 1.0 + 1e-9, "transport target"),
              ("max_mass_relative_drift", 1e-11, "mass drift"), ("max_l2_budget_relative_defect", 1e-12, "L2 identity"))
    for key, limit, label in checks:
        value = row.get(key)
        if isinstance(value, (int, float)) and value > limit:
            problems.append(f"{label} {value:.3g}")
    return problems


def _is_base(run, p, level=None) -> bool:
    return (run["p"] == p and run["family"] == "m128" and run["velocity"] == "raw" and not run["offset"]
            and not run.get("vq") and (level is None or run["level"] == level) and run["stage"] in "ABC")


def _best(candidates, key, tie_key):
    candidates = [run for run in candidates if run["valid"] and run.get(key) is not None]
    if not candidates:
        return None, []
    ranked = sorted(candidates, key=lambda run: run[key])
    best = ranked[0]
    tied = [run for run in ranked if run[key] <= best[key] * (1 + TIE) and run.get(tie_key) is not None]
    if len(tied) > 1:
        best = min(tied, key=lambda run: run[tie_key])
    return best, ranked


def select_presets(study) -> dict:
    """Fastest valid Poisson and transport candidate per degree; writes presets_selected.json."""
    study = Path(study)
    all_runs = runs(study)
    selected = {}
    for p in range(1, 10):
        levels = [level for level in ("L3", "L2") if any(_is_base(run, p, level) and run["valid"] for run in all_runs)]
        if not levels:
            continue
        level = levels[0]
        base = [run for run in all_runs if _is_base(run, p, level)]
        # Poisson: stages A and B share the default transport of the stage A runs.
        poisson_runs = [run for run in base if run["stage"] in "AB"]
        winner, ranked = _best(poisson_runs, "step", "hard")
        if winner is None:
            continue
        choice = dict(poisson=winner["poisson"], transport=winner["transport"], level=level,
                      poisson_run=winner["run_id"], hard=winner["hard"], step=winner["step"],
                      poisson_ranking=[(run["poisson"], round(run["step"], 5), round(run["hard"], 5)) for run in ranked])
        # Stage H (the saturated phase, large seed, L2) overrides the Poisson choice where it has run.
        nonlinear = [run for run in all_runs if run["stage"] == "H" and run["p"] == p and run["family"] == "m128"]
        best, ranked_nonlinear = _best(nonlinear, "nonlinear_step", "step")
        if best is not None:
            base_run = next((run for run in ranked if run["poisson"] == best["poisson"]), winner)
            choice.update(poisson=best["poisson"], poisson_run=base_run["run_id"], hard=base_run["hard"],
                          step=base_run["step"], nonlinear_run=best["run_id"], nonlinear_step=best["nonlinear_step"],
                          nonlinear_ranking=[(run["poisson"], round(run["nonlinear_step"], 5),
                                              run.get("nonlinear_poisson_iterations")) for run in ranked_nonlinear])
        # Transport: the stage C runs made with the chosen Poisson preset (after the stage H override; the
        # transport solve does not depend on it), else those made with the stage A/B winner.
        transport_runs = [run for run in base if run["poisson"] == choice["poisson"]]
        if not any(run["stage"] == "C" for run in transport_runs):
            transport_runs = [run for run in base if run["poisson"] == winner["poisson"]]
        best, ranked = _best(transport_runs, "step", "hard")
        if best is not None:
            choice.update(transport=best["transport"], run=best["run_id"],
                          transport_ranking=[(run["transport"], round(run["step"], 5)) for run in ranked])
            if best["poisson"] == choice["poisson"]:
                choice.update(step=best["step"], hard=best["hard"])
        selected[str(p)] = choice
    (study / "presets_selected.json").write_text(json.dumps(selected, indent=2) + "\n")
    return selected


def _snapshot_density(study: Path, run: dict):
    path = next((study / "runs").glob(f"*_{run['run_id']}_seed_snapshots.npz"), None)
    if path is None:
        return None
    with np.load(path) as data:
        return np.asarray(data["density"][-1])


def _consistency(study: Path, all_runs: list[dict]) -> list[dict]:
    """Relative coefficient difference of the final densities of valid runs that solve the same discrete
    problem (same mesh, degree, drift and volume rule); they may differ only by the solver tolerances."""
    groups = {}
    for run in all_runs:
        if run["valid"]:
            key = (run["family"], run["level"], run["p"], run["velocity"], run["offset"], run.get("vq"))
            groups.setdefault(key, []).append(run)
    out = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        reference = _snapshot_density(study, members[0])
        if reference is None:
            continue
        for run in members[1:]:
            density = _snapshot_density(study, run)
            if density is None or density.shape != reference.shape:
                continue
            out.append(dict(group=key, reference=members[0]["run_id"], run=run["run_id"],
                            relative_difference=float(np.linalg.norm(density - reference) / np.linalg.norm(reference))))
    return out


def _quadrature(study: Path, all_runs: list[dict]) -> list[dict]:
    """Stage E: the collapsed rule with vq^2 points (exact for the advection term when vq = (3p+2)//2)
    against the default rule, same preset and mesh."""
    out = []
    for run in all_runs:
        if run["stage"] != "E" or not run["valid"]:
            continue
        twin = next((other for other in all_runs if _is_base(other, run["p"], run["level"]) and other["valid"]
                     and other["poisson"] == run["poisson"] and other["transport"] == run["transport"]), None)
        if twin is None:
            continue
        a, b = _snapshot_density(study, run), _snapshot_density(study, twin)
        difference = (float(np.linalg.norm(a - b) / np.linalg.norm(b))
                      if a is not None and b is not None and a.shape == b.shape else None)
        out.append(dict(p=run["p"], points=run["vq"] ** 2, step=run["step"], default_step=twin["step"],
                        relative_difference=difference))
    return out


def _fmt(value, digits=3):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def smoke_report(study) -> Path:
    """Write smoke_runs.csv, smoke_report.md and the step-time figure."""
    study = Path(study)
    all_runs = sorted(runs(study), key=lambda run: (run["stage"], run["family"], run["level"], run["p"], run["run_id"]))
    selected = select_presets(study)
    keys = list(dict.fromkeys(key for run in all_runs for key in run))
    with (study / "smoke_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for run in all_runs:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
                             for key, value in run.items()})
    ms = lambda value: None if value is None else 1e3 * value  # noqa: E731
    lines = ["# D8-S smoke campaign", "",
             "Synchronized step time = median of steps 3..N; 'hard' replaces the warm Poisson Krylov time by the "
             "cold first solve's (reported, not ranked on). Times in ms, memory in MiB (nvidia-smi, per process).", ""]
    for stage in "ABCDEFGH":
        members = [run for run in all_runs if run["stage"] == stage]
        if not members:
            continue
        lines += [f"## Stage {stage}", "",
                  "| run | triangles | step | hard | Poisson RHS / solve / recon / post | transport asm / solve "
                  "| it P / T (cold P) | peak | W / °C (busy) | checks |", "|---|---|---|---|---|---|---|---|---|---|"]
        for run in members:
            parts = "/".join(_fmt(ms(run.get(f"s_median_{key}")), 3) for key in (
                "poisson_time_rhs_assembly", "poisson_step_time_solve", "poisson_step_time_reconstruction",
                "poisson_flux_postprocess_time"))
            transport = "/".join(_fmt(ms(run.get(f"s_median_{key}")), 3) for key in (
                "transport_step_time_assembly", "transport_step_time_solve"))
            iterations = (f"{_fmt(run.get('s_median_poisson_step_iterations'))} / "
                          f"{_fmt(run.get('s_median_transport_step_iterations'))} "
                          f"({_fmt(run.get('s_first_poisson_solver_iterations'))})")
            lines.append(f"| {run['run_id']} | {run.get('triangles')} | {_fmt(ms(run['step']), 4)} | "
                         f"{_fmt(ms(run['hard']), 4)} | {parts} | {transport} | {iterations} | "
                         f"{_fmt(run.get('peak_process_mib'))} | {_fmt(run.get('power_w_active'), 3)} / "
                         f"{_fmt(run.get('temperature_c_active_max'))} | {'ok' if run['valid'] else '; '.join(run['problems'])} |")
        lines.append("")
    lines += ["## Selected presets", "", "| p | level | Poisson | transport | step | hard | saturated-phase step (L2) |",
              "|---|---|---|---|---|---|---|"]
    for p, choice in sorted(selected.items(), key=lambda item: int(item[0])):
        lines.append(f"| {p} | {choice['level']} | {choice['poisson']} | {choice['transport']} | "
                     f"{_fmt(ms(choice['step']), 4)} | {_fmt(ms(choice['hard']), 4)} | "
                     f"{_fmt(ms(choice.get('nonlinear_step')), 4)} |")
    saturated = [run for run in all_runs if run["stage"] == "H"]
    if saturated:
        lines += ["", f"## Saturated phase (stage H: seed 0.1, L2, medians over t >= {NONLINEAR_FROM:g})", "",
                  "| run | step | Poisson Krylov | it P / T | steps | checks |", "|---|---|---|---|---|---|"]
        for run in saturated:
            lines.append(f"| {run['run_id']} | {_fmt(ms(run.get('nonlinear_step')), 4)} | "
                         f"{_fmt(ms(run.get('nonlinear_poisson_krylov')), 3)} | "
                         f"{_fmt(run.get('nonlinear_poisson_iterations'))} / {_fmt(run.get('nonlinear_transport_iterations'))} | "
                         f"{_fmt(run.get('nonlinear_steps'))} | {'ok' if run['valid'] else '; '.join(run['problems'])} |")
    lines += ["", "## Drift variants (selected presets, synchronized step time, ms)", "",
              "| p | raw q=p | raw q=p+1 | V2 | V2 postprocess |", "|---|---|---|---|---|"]
    for p in range(1, 10):
        def pick(velocity, offset):
            return next((run for run in all_runs if run["p"] == p and run["family"] == "m128" and run["valid"]
                         and run["level"] == (selected.get(str(p)) or {}).get("level") and run["velocity"] == velocity
                         and run["offset"] == offset and not run.get("vq") and run["stage"] in "ABCD"
                         and (velocity != "raw" or offset or run["run_id"] == (selected.get(str(p)) or {}).get("run"))), None)
        raw, raised, v2 = pick("raw", 0), pick("raw", 1), pick("compatible", 0)
        lines.append(f"| {p} | {_fmt(ms(raw and raw['step']), 4)} | {_fmt(ms(raised and raised['step']), 4)} | "
                     f"{_fmt(ms(v2 and v2['step']), 4)} | "
                     f"{_fmt(ms(v2 and v2.get('s_median_poisson_flux_postprocess_time')), 3)} |")
    consistency = _consistency(study, all_runs)
    worst = max((item["relative_difference"] for item in consistency), default=None)
    lines += ["", "## Consistency of runs solving the same discrete problem", "",
              f"{len(consistency)} pairs; largest relative difference of the final density coefficients: {_fmt(worst)}.", ""]
    for item in sorted(consistency, key=lambda item: -item["relative_difference"])[:10]:
        lines.append(f"- {item['run']} vs {item['reference']}: {_fmt(item['relative_difference'])}")
    quadrature = _quadrature(study, all_runs)
    if quadrature:
        lines += ["", "## Volume quadrature (stage E)", ""]
        for item in quadrature:
            lines.append(f"- p = {item['p']}: {item['points']} points, step {_fmt(ms(item['step']), 4)} ms against "
                         f"{_fmt(ms(item['default_step']), 4)} ms; density difference {_fmt(item['relative_difference'])}")
    contaminated = [row for row in _all_attempts(study) if row["status"] == "contaminated"]
    lines += ["", "## Tooling", "",
              f"- contaminated attempts (requeued): {[(row['run_id'], row['intruders']) for row in contaminated]}",
              f"- runs not valid: {[(run['run_id'], run['problems']) for run in all_runs if not run['valid']]}"]
    path = study / "smoke_report.md"
    path.write_text("\n".join(lines) + "\n")
    _figure(study, all_runs, selected)
    return path


def _all_attempts(study: Path) -> list[dict]:
    path = study / "runs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _figure(study: Path, all_runs: list[dict], selected: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hybridge.io.figures import publication_style, save_publication_figure

    with publication_style(font_size=8.0):
        figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
        for velocity, offset, label, marker in (("raw", 0, "raw, $q=p$", "o"), ("raw", 1, "raw, $q=p+1$", "s"),
                                                ("compatible", 0, "V2", "^")):
            points = []
            for p in range(1, 10):
                choice = selected.get(str(p)) or {}
                run = next((run for run in all_runs if run["p"] == p and run["family"] == "m128" and run["valid"]
                            and run["level"] == choice.get("level") and run["velocity"] == velocity
                            and run["offset"] == offset and not run.get("vq")
                            and (velocity != "raw" or offset or run["run_id"] == choice.get("run"))), None)
                if run is not None:
                    points.append((p, 1e3 * run["step"]))
            if points:
                left.plot(*zip(*points), marker=marker, label=label)
        left.set(xlabel="$p$", ylabel="step time [ms] (synchronized)", yscale="log")
        left.legend(frameon=False)
        valid = [run for run in all_runs if run["valid"] and run.get("peak_process_mib") and run["family"] == "m128"
                 and run["level"] == "L3"]
        right.scatter([run["p"] for run in valid], [run["peak_process_mib"] / 1024 for run in valid], s=12)
        right.set(xlabel="$p$", ylabel="peak device memory [GiB]")
        for ax in (left, right):
            ax.grid(True, alpha=.4)
        save_publication_figure(figure, study / "smoke_step_times", formats=("pdf", "png"), dpi=200)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("select", "report"))
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(select_presets(args.study), indent=2))
    else:
        print(smoke_report(args.study))


if __name__ == "__main__":
    main()
