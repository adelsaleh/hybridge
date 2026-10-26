#!/usr/bin/env python
"""Compare a few diocotron runs of one campaign study: cost per phase, reference-free accuracy, cross-mesh
differences and figures. No solves.

Written for the D8 intermediate campaign (phase I of the D8 launcher: p = 5 against p = 9 at L3, each
without and with the KKT positivity projection, SI-BDF2 to T = 70); it reads only the study's outputs
(``runs.jsonl``, the driver's ``runs/<prefix>_summary.json``, the per-step ``*_<run_id>_seed_timings.jsonl``,
the diagnostics ``*_<run_id>_seed.jsonl`` and the snapshots ``*_<run_id>_seed_snapshots.npz``) and
compares every pair of the named runs:

* runs on different meshes (different degree or level): the relative L2 difference of the saved fields at
  every snapshot time by :func:`diocotron_order_errors.cross_mesh_errors`, both ways (each run once as the
  reference, whose volume rule integrates; ``reference_quad_1d = p_ref + 1`` keeps it exact for the squared
  reference field), cached in ``<output>/errors.jsonl`` so that a rerun skips finished pairs;
* runs on the same mesh (e.g. KKT on against off): the relative coefficient L2 difference at every common
  snapshot time.

Per run: the wall time to the final time (the launcher record's ``wall_s`` with the setup, and the sum of the
synchronized step times without it), medians of the step, Poisson (Krylov, postprocess), transport
(assembly, solve), enstrophy-budget and positivity-projection times and of the iteration counts in the
linear phase (t < ``--linear-until``), the nonlinear phase (t >= ``--nonlinear-from``) and overall; the
growth rate of the seeded mode over ``--window`` against ``--reference-rate``; the extreme values of
the conservation drifts, the enstrophy, the minimum density, the minimum cell average, the negative mass
and the KKT projection report (flagged elements, mass returned, relative correction).

Outputs in ``--output-dir``: ``report.md``, ``cost.csv``, ``accuracy.csv``, ``cross.csv``,
``same_mesh.csv``, ``errors.jsonl`` and the figures ``step_time``, ``time_series``, ``cross_difference``
and ``density_sector`` (the m = 128 sector of the no-KKT runs at ``--figure-times`` with the mesh drawn,
signed density colours), each as PDF and PNG.

Example::

    python -m scripts.guiding_center.studies.diocotron.analyze_diocotron_pairs STUDY \\
        --runs m128L3_p5_dt0p0125,m128L3_p5_dt0p0125_kkt,m128L3_p9_dt0p0125,m128L3_p9_dt0p0125_kkt \\
        --times 30,40,45,50,55,60,70 --output-dir STUDY/analysis_I
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HYBRIDGE = Path(__file__).resolve().parents[4]  # sys.path guard for direct starts (runs before paths.py is importable)
if str(HYBRIDGE) not in sys.path:
    sys.path.insert(0, str(HYBRIDGE))

from scripts.guiding_center.studies.diocotron.diocotron_order_errors import cross_mesh_errors  # noqa: E402

TIME_KEYS = (("step_advance_wall_time", "step"), ("poisson_time_total", "poisson"),
             ("poisson_krylov_time", "poisson_krylov"), ("poisson_time_postprocessing", "postprocess"),
             ("transport_time_assembly", "transport_assembly"), ("transport_time_solve", "transport_solve"),
             ("enstrophy_budget_time", "enstrophy"), ("positivity_projection_time", "positivity"))
DIAGNOSTICS_KEYS = ("diagnostics_wall_time", "diagnostics_time")  # in the diagnostics rows, not the step rows
COUNT_KEYS = (("poisson_solver_iterations", "poisson_its"), ("transport_solver_iterations", "transport_its"),
              ("positivity_flagged", "flagged"))
SERIES_KEYS = ("enstrophy", "rho_min", "rho_cell_average_min", "mass_relative_drift", "energy_relative_drift",
               "rho_negative_mass_quadrature", "enstrophy_relative_drift", "l2_budget_relative_defect")
CROSS_FIELDS = ("density", "potential", "flux")
DEFAULT_TIMES = (30., 40., 45., 50., 55., 60., 70.)
REFERENCE_RATE = 0.18509326  # m = 128 Gaussian ring, the D8 reference growth rate
WINDOW = (20., 45.)


# readers --------------------------------------------------------------------------------------------------
def _jsonl(path: Path) -> list[dict]:
    rows = []
    if path is None or not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) else None


def _get(row: dict, name: str):
    """``row[name]``, else the value of the first key ending with ``_<name>`` (stage-prefixed records)."""
    if name in row:
        return _number(row[name])
    for key, value in row.items():
        if key.endswith("_" + name):
            return _number(value)
    return None


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def _mode_key(rows: list[dict]) -> str | None:
    keys = set()
    for row in rows[:3]:
        keys.update(k for k in row if re.fullmatch(r"diocotron_phi_mode_\d+_l2", k))
    if not keys:
        return None
    if "diocotron_phi_mode_128_l2" in keys:
        return "diocotron_phi_mode_128_l2"
    base = _number(rows[0].get("diocotron_mode_base")) if rows else None
    if base is not None and f"diocotron_phi_mode_{int(base)}_l2" in keys:
        return f"diocotron_phi_mode_{int(base)}_l2"
    return sorted(keys, key=lambda k: int(k.split("_")[3]))[-1]


def load_run(study: Path, run_id: str, *, summary_name: str) -> dict:
    """Everything the comparison needs of one run; missing pieces are None/empty, never an error."""
    runs = study / "runs"
    records = [row for row in _jsonl(study / "runs.jsonl") if row.get("run_id") == run_id]
    record = records[-1] if records else {}
    summary = {}
    path = runs / summary_name
    if path.exists():
        for row in json.loads(path.read_text()).get("rows", []):
            if f"_{run_id}_seed" in str(row.get("prefix", "")):
                summary = row
    timings = _jsonl(next(runs.glob(f"*_{run_id}_seed_timings.jsonl"), None))
    diagnostics = _jsonl(next(runs.glob(f"*_{run_id}_seed.jsonl"), None))
    snapshots = next(runs.glob(f"*_{run_id}_seed_snapshots.npz"), None)
    p = record.get("p") or summary.get("order")
    triangles = record.get("triangles") or summary.get("triangles")
    if snapshots is not None and (p is None or triangles is None):
        with np.load(snapshots) as npz:
            p = p or int(npz["order"])
            triangles = triangles or int(len(npz["triangles"]))
    steps = [row for row in timings if isinstance(row.get("step"), (int, float)) and _number(row.get("time")) is not None]
    return dict(run_id=run_id, record=record, summary=summary, timings=steps, diagnostics=diagnostics,
                snapshots=snapshots, p=None if p is None else int(p),
                triangles=None if triangles is None else int(triangles),
                kkt=bool(record.get("kkt")) or run_id.endswith("_kkt"),
                label=(f"p={p}" if p is not None else run_id) + (" KKT" if run_id.endswith("_kkt") or record.get("kkt") else ""))


# cost -----------------------------------------------------------------------------------------------------
def cost_rows(run: dict, *, linear_until: float, nonlinear_from: float, skip: int = 2) -> list[dict]:
    """One row per phase (linear, nonlinear, all): medians of the per-step times and counts."""
    steps = [row for row in run["timings"] if row["step"] > skip] or run["timings"]
    phases = (("linear", lambda t: t < linear_until), ("nonlinear", lambda t: t >= nonlinear_from), ("all", lambda t: True))
    out = []
    for phase, keep in phases:
        rows = [row for row in steps if keep(row["time"])]
        item = dict(run_id=run["run_id"], label=run["label"], phase=phase, steps=len(rows))
        for key, name in TIME_KEYS:
            item[name + "_ms"] = None if not rows else _median(_get(row, key) for row in rows)
            if item[name + "_ms"] is not None:
                item[name + "_ms"] *= 1000.0
        for key, name in COUNT_KEYS:
            item[name] = None if not rows else _median(_get(row, key) for row in rows)
        diagnostics = [row for row in run["diagnostics"] if _number(row.get("time")) is not None and keep(row["time"])]
        values = [next((_number(row.get(key)) for key in DIAGNOSTICS_KEYS if _number(row.get(key)) is not None), None)
                  for row in diagnostics]
        item["diagnostics_ms"] = _median(values)
        if item["diagnostics_ms"] is not None:
            item["diagnostics_ms"] *= 1000.0
        item["diagnostics_rows"] = len(diagnostics)
        out.append(item)
    return out


def wall_summary(run: dict) -> dict:
    """Wall time of the whole run (record), of the stepping alone (sum of the synchronized step times), the
    final time reached and the peak device memory."""
    steps = run["timings"]
    advance = [_get(row, "step_advance_wall_time") for row in steps]
    advance = [v for v in advance if v is not None]
    final = max((row["time"] for row in steps), default=None)
    projection = [v for v in (_get(row, "positivity_projection_time") for row in steps) if v is not None]
    return dict(run_id=run["run_id"], label=run["label"], p=run["p"], triangles=run["triangles"], kkt=run["kkt"],
                status=run["record"].get("status"), steps=len(steps), final_time=final,
                wall_s=_number(run["record"].get("wall_s")) or _number(run["summary"].get("wall_time_s")),
                stepping_s=float(np.sum(advance)) if advance else None,
                positivity_s=float(np.sum(projection)) if projection else None,
                peak_process_mib=_number(run["record"].get("peak_process_mib")),
                peak_total_mib=_number(run["record"].get("peak_total_mib")))


def cumulative_wall(run: dict, times) -> dict:
    """Stepping wall time (sum of the step times) reached at every requested time."""
    steps = run["timings"]
    if not steps:
        return {}
    t = np.array([row["time"] for row in steps])
    w = np.cumsum([(_get(row, "step_advance_wall_time") or 0.0) for row in steps])
    return {float(wanted): float(w[t <= wanted + 1e-9][-1]) if np.any(t <= wanted + 1e-9) else None for wanted in times}


# accuracy without a reference -----------------------------------------------------------------------------
def _fit_growth(times, amplitude, window, reference):
    try:
        from scripts.guiding_center.diagnostics.diocotron_reference import fit_growth
        fit = fit_growth(np.asarray(times), np.asarray(amplitude), window=tuple(window), reference=reference)
        return dict(gamma=_number(fit.get("growth_rate")), relative_error=_number(fit.get("relative_error")),
                    r_squared=_number(fit.get("r_squared")), status=fit.get("status", "ok"))
    except Exception:  # noqa: BLE001 - the fallback below is the same least-squares fit
        pass
    t, a = np.asarray(times, dtype=float), np.asarray(amplitude, dtype=float)
    keep = (t >= window[0]) & (t <= window[1]) & (a > 0)
    if keep.sum() < 4:
        return dict(gamma=None, relative_error=None, r_squared=None, status="too_few_samples")
    slope, intercept = np.polyfit(t[keep], np.log(a[keep]), 1)
    fitted = slope * t[keep] + intercept
    residual = np.log(a[keep]) - fitted
    total = np.log(a[keep]) - np.log(a[keep]).mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total) if float(total @ total) > 0 else None
    return dict(gamma=float(slope), relative_error=None if reference is None else float(slope / reference - 1.0),
                r_squared=r2, status="ok")


def accuracy_row(run: dict, *, window, reference_rate, times) -> dict:
    rows = run["diagnostics"]
    key = _mode_key(rows)
    series = {name: [] for name in SERIES_KEYS}
    t, amplitude = [], []
    for row in rows:
        time = _number(row.get("time"))
        if time is None:
            continue
        for name in SERIES_KEYS:
            value = _number(row.get(name))
            if value is not None:
                series[name].append((time, value))
        if key is not None and _number(row.get(key)) is not None:
            t.append(time)
            amplitude.append(row[key])
    fit = _fit_growth(t, amplitude, window, reference_rate) if t else dict(gamma=None, relative_error=None,
                                                                            r_squared=None, status="no_mode_series")
    summary = run["summary"]
    out = dict(run_id=run["run_id"], label=run["label"], mode_key=key, fit_window=list(window),
               gamma=fit["gamma"], gamma_relative_error=fit["relative_error"], gamma_r_squared=fit["r_squared"],
               fit_status=fit["status"], summary_gamma=_number(summary.get("fitted_gamma")),
               summary_relative_error=_number(summary.get("relative_error")))

    def extreme(name, function):
        values = [v for _, v in series[name]]
        return function(values) if values else None

    out.update(max_abs_mass_relative_drift=extreme("mass_relative_drift", lambda v: float(np.max(np.abs(v)))),
               max_abs_energy_relative_drift=extreme("energy_relative_drift", lambda v: float(np.max(np.abs(v)))),
               max_abs_l2_budget_relative_defect=extreme("l2_budget_relative_defect", lambda v: float(np.max(np.abs(v)))),
               min_rho_min=extreme("rho_min", min), min_rho_cell_average=extreme("rho_cell_average_min", min),
               max_negative_mass=extreme("rho_negative_mass_quadrature", max),
               enstrophy_initial=series["enstrophy"][0][1] if series["enstrophy"] else None,
               enstrophy_final=series["enstrophy"][-1][1] if series["enstrophy"] else None)
    if series["enstrophy"]:
        e = np.array(series["enstrophy"])
        for wanted in times:
            index = int(np.argmin(np.abs(e[:, 0] - wanted)))
            out[f"enstrophy_t{wanted:g}"] = float(e[index, 1]) if abs(e[index, 0] - wanted) < 0.5 else None
        out["enstrophy_relative_change"] = float(e[-1, 1] / e[0, 1] - 1.0) if e[0, 1] else None
    if key is not None and t:
        a = np.array([t, amplitude]).T
        for wanted in times:
            index = int(np.argmin(np.abs(a[:, 0] - wanted)))
            out[f"amplitude_t{wanted:g}"] = float(a[index, 1]) if abs(a[index, 0] - wanted) < 0.5 else None
    return out


def positivity_row(run: dict) -> dict:
    steps = run["timings"]
    flagged = [v for v in (_get(row, "positivity_flagged") for row in steps) if v is not None]
    out = dict(run_id=run["run_id"], label=run["label"], kkt=run["kkt"], projection_steps=len(flagged),
               steps_with_flagged=int(sum(1 for v in flagged if v > 0)) if flagged else None,
               max_flagged=int(max(flagged)) if flagged else None,
               first_flagged_time=next((row["time"] for row in steps if (_get(row, "positivity_flagged") or 0) > 0), None))
    for name in ("positivity_mass_returned", "positivity_correction_relative", "positivity_fallback",
                 "positivity_negative_mean_projected", "positivity_negative_mean_zeroed"):
        values = [v for v in (_get(row, name) for row in steps) if v is not None]
        out["max_" + name.replace("positivity_", "")] = max(values) if values else None
    values = [v for v in (_get(row, "positivity_projection_time") for row in steps) if v is not None]
    out["projection_time_total_s"] = float(np.sum(values)) if values else None
    out["projection_time_median_ms"] = 1000.0 * float(np.median(values)) if values else None
    return out


# cross comparisons ---------------------------------------------------------------------------------------
def _saved_times(path: Path) -> list[float]:
    with np.load(path) as npz:
        return [float(t) for t in npz["times"]]


def _common_times(a: Path, b: Path, times, tolerance=1e-6) -> list[float]:
    ta, tb = _saved_times(a), _saved_times(b)
    return [float(t) for t in times if any(abs(t - s) <= tolerance for s in ta) and any(abs(t - s) <= tolerance for s in tb)]


def _same_mesh(a: dict, b: dict) -> bool:
    if a["p"] != b["p"] or a["triangles"] != b["triangles"] or a["snapshots"] is None or b["snapshots"] is None:
        return False
    with np.load(a["snapshots"]) as x, np.load(b["snapshots"]) as y:
        return x["triangles"].shape == y["triangles"].shape and np.array_equal(x["triangles"], y["triangles"]) \
            and np.allclose(x["node_coords"], y["node_coords"])


def same_mesh_rows(a: dict, b: dict, times) -> list[dict]:
    """Relative coefficient L2 difference of every saved field at the common times (same discrete space)."""
    out = []
    common = _common_times(a["snapshots"], b["snapshots"], times)
    with np.load(a["snapshots"]) as x, np.load(b["snapshots"]) as y:
        tx, ty = [float(t) for t in x["times"]], [float(t) for t in y["times"]]
        for wanted in common:
            i = int(np.argmin(np.abs(np.array(tx) - wanted)))
            j = int(np.argmin(np.abs(np.array(ty) - wanted)))
            for field in CROSS_FIELDS:
                if field not in x.files or field not in y.files:
                    continue
                u, v = np.asarray(x[field][i], dtype=float), np.asarray(y[field][j], dtype=float)
                if u.shape != v.shape:
                    continue
                norm = float(np.linalg.norm(v))
                out.append(dict(run=a["run_id"], other=b["run_id"], time=wanted, field=field,
                                absolute=float(np.linalg.norm(u - v)), relative=float(np.linalg.norm(u - v) / norm) if norm else None))
    return out


def cross_rows(run: dict, reference: dict, times, cache: Path, *, chunk_elements: int) -> list[dict]:
    """Cross-mesh errors of ``run`` against ``reference`` at ``times`` (cached in ``cache``)."""
    done = {(row["run"], row["reference"], round(row["time"], 6), row["field"]): row for row in _jsonl(cache)}
    wanted = [t for t in _common_times(run["snapshots"], reference["snapshots"], times)
              if (run["run_id"], reference["run_id"], round(t, 6), "density") not in done]
    rows = [row for key, row in done.items() if key[0] == run["run_id"] and key[1] == reference["run_id"]]
    if wanted:
        print(f"cross-mesh: {run['run_id']} against {reference['run_id']} at t = {wanted}", flush=True)
        new = cross_mesh_errors(run["snapshots"], reference["snapshots"], wanted, fields=CROSS_FIELDS,
                                chunk_elements=chunk_elements, reference_quad_1d=int(reference["p"]) + 1)
        with cache.open("a") as stream:
            for row in new:
                row.update(run=run["run_id"], reference=reference["run_id"], reference_quad_1d=int(reference["p"]) + 1)
                stream.write(json.dumps(row) + "\n")
                rows.append(row)
    return sorted(rows, key=lambda row: (row["time"], row["field"]))


# figures --------------------------------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from hybridge.io.figures import publication_style
    except ImportError:  # pragma: no cover
        from contextlib import nullcontext as publication_style
    return plt, publication_style


def _save(figure, stem: Path):
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=200)
    return stem


COLORS = {False: {5: "#2a78d6", 9: "#eb6834"}, True: {5: "#7fb0e8", 9: "#f3a98a"}}


def _color(run: dict):
    palette = COLORS[run["kkt"]]
    return palette.get(run["p"], "#555555")


def step_time_figure(runs: list[dict], stem: Path, *, linear_until, nonlinear_from):
    plt, style = _style()
    with style():
        figure, axes = plt.subplots(3, 1, figsize=(5.2, 6.0), sharex=True, constrained_layout=True)
        for run in runs:
            steps = run["timings"][2:]
            if not steps:
                continue
            t = np.array([row["time"] for row in steps])
            for ax, key in zip(axes, ("step_advance_wall_time", "poisson_solver_iterations", "positivity_projection_time")):
                values = np.array([(_get(row, key) if _get(row, key) is not None else np.nan) for row in steps], dtype=float)
                if np.all(np.isnan(values)):
                    continue
                ax.plot(t, values, lw=.6, color=_color(run), label=run["label"],
                        ls="--" if run["kkt"] else "-")
        axes[0].set_ylabel("step wall time (s)")
        axes[1].set_ylabel("Poisson iterations")
        axes[2].set_ylabel("KKT projection (s)")
        axes[2].set_xlabel("$t$")
        for ax in axes:
            ax.axvspan(0, linear_until, color="#eeeeee", lw=0)
            ax.axvline(nonlinear_from, color="#bbbbbb", lw=.5)
        if not axes[2].lines:
            axes[2].set_ylabel("KKT projection: none")
        axes[0].legend(fontsize=7, ncol=2)
        _save(figure, stem)
        plt.close(figure)


def time_series_figure(runs: list[dict], stem: Path, *, reference_rate, window):
    plt, style = _style()
    panels = (("amplitude", "mode amplitude", True), ("enstrophy", "enstrophy", False),
              ("rho_min", "min density", False), ("rho_cell_average_min", "min cell average", False),
              ("mass_relative_drift", "relative mass drift", False), ("energy_relative_drift", "relative energy drift", False))
    with style():
        figure, axes = plt.subplots(3, 2, figsize=(7.0, 7.0), sharex=True, constrained_layout=True)
        for run in runs:
            rows = run["diagnostics"]
            key = _mode_key(rows)
            t = np.array([_number(row.get("time")) or np.nan for row in rows])
            for ax, (name, label, log) in zip(axes.flat, panels):
                source = key if name == "amplitude" else name
                if source is None:
                    continue
                values = np.array([(_number(row.get(source)) if _number(row.get(source)) is not None else np.nan) for row in rows])
                if np.all(np.isnan(values)):
                    continue
                ax.plot(t, values, lw=.7, color=_color(run), label=run["label"], ls="--" if run["kkt"] else "-")
                ax.set_ylabel(label)
                if log:
                    ax.set_yscale("log")
        if reference_rate:
            ax = axes.flat[0]
            t0, t1 = window
            rows = runs[0]["diagnostics"]
            key = _mode_key(rows)
            if key is not None:
                start = next((_number(row.get(key)) for row in rows if (_number(row.get("time")) or -1) >= t0), None)
                if start:
                    ax.plot([t0, t1], [start, start * np.exp(reference_rate * (t1 - t0))], color="k", lw=.8, ls=":",
                            label=f"reference rate {reference_rate:.5f}")
        axes.flat[0].legend(fontsize=6)
        for ax in axes[-1]:
            ax.set_xlabel("$t$")
        _save(figure, stem)
        plt.close(figure)


def cross_figure(cross: list[dict], same: list[dict], walls: dict, stem: Path, labels: dict | None = None):
    plt, style = _style()
    labels = labels or {}
    name = lambda run_id: labels.get(run_id, run_id)
    with style():
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
        pairs = sorted({(row["run"], row["reference"]) for row in cross})
        for run, reference in pairs:
            rows = [row for row in cross if row["run"] == run and row["reference"] == reference and row["field"] == "density"
                    and row.get("relative") is not None]
            if rows:
                axes[0].plot([row["time"] for row in rows], [row["relative"] for row in rows], marker="o", ms=3, lw=.8,
                             label=f"{name(run)} against {name(reference)} ({name(reference)} rule)")
        for run, other in sorted({(row["run"], row["other"]) for row in same}):
            rows = [row for row in same if row["run"] == run and row["other"] == other and row["field"] == "density"
                    and row.get("relative") is not None]
            if rows:
                axes[0].plot([row["time"] for row in rows], [row["relative"] for row in rows], marker="s", ms=3, lw=.8, ls="--",
                             label=f"{name(run)} against {name(other)} (same mesh)")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("$t$")
        axes[0].set_ylabel("relative $L^2$ density difference")
        if axes[0].lines:
            axes[0].legend(fontsize=6)
        for label, series in walls.items():
            times = sorted(series)
            axes[1].plot(times, [series[t] / 60.0 for t in times], marker="o", ms=3, lw=.8, label=label)
        axes[1].set_xlabel("$t$")
        axes[1].set_ylabel("stepping wall time (min)")
        axes[1].legend(fontsize=6)
        _save(figure, stem)
        plt.close(figure)


def density_sector_figure(runs: list[dict], stem: Path, *, times, pixels=1200, sector_pixels=1400, supersample=2):
    """Density in the m = 128 sector for the given runs (rows) at ``times`` (columns), mesh drawn, signed colours."""
    from types import SimpleNamespace

    from scripts.guiding_center.studies.diocotron.plot_diocotron_highmode import (
        MESH_STYLE, _draw_mesh, _save_raster_figure, _sector, _snapshot_rasters)
    from hybridge.io.figures import signed_density_colors

    plt, style = _style()
    sector = _sector(128)
    rows = []
    for run in runs:
        if run["snapshots"] is None:
            continue
        with np.load(run["snapshots"]) as npz:
            saved = [round(float(t), 6) for t in npz["times"]]
            chosen = [t for t in times if round(t, 6) in saved]
            if not chosen:
                continue
            (_, _), (bounds, images) = _snapshot_rasters(npz, pixels, sector=sector,
                                                         indices=[saved.index(round(t, 6)) for t in chosen],
                                                         sector_pixels=sector_pixels, supersample=supersample)
            mesh = SimpleNamespace(node_coords=np.asarray(npz["node_coords"]), triangles=np.asarray(npz["triangles"]))
        rows.append((run, chosen, bounds, images, mesh))
    if not rows:
        return None
    columns = max(len(row[1]) for row in rows)
    shown = [image for row in rows for image in row[3]]
    cmap, norm = signed_density_colors(min(float(image.min()) for image in shown), 1.0)
    height = (bounds[3] - bounds[2]) / (bounds[1] - bounds[0])
    with style():
        figure, axes = plt.subplots(len(rows), columns, figsize=(2.3 * columns, 2.3 * height * len(rows) + .6),
                                    constrained_layout=True, squeeze=False)
        overlays = []
        for r, (run, chosen, bounds, images, mesh) in enumerate(rows):
            for c, t in enumerate(chosen):
                ax = axes[r, c]
                artist = ax.imshow(images[c], extent=bounds, origin="upper", cmap=cmap, norm=norm, interpolation="none")
                overlays.append((ax, mesh, bounds))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"$t={t:g}$", fontsize=8)
            axes[r, 0].set_ylabel(f"{run['label']}, {run['triangles']} triangles", fontsize=7)
            for c in range(len(chosen), columns):
                axes[r, c].axis("off")
        figure.colorbar(artist, ax=axes, shrink=.5, aspect=35, location="bottom", pad=.01, label=r"density $\rho_h$")
        figure.draw_without_rendering()
        for ax, mesh, bounds in overlays:
            points_per_unit = ax.bbox.width * 72 / figure.dpi / (bounds[1] - bounds[0])
            _draw_mesh(ax, mesh, bounds, points_per_unit, **MESH_STYLE)
            ax.set(xlim=bounds[:2], ylim=bounds[2:])
        out = _save_raster_figure(figure, str(stem), dpi=400)
        plt.close(figure)
        return out


# report ---------------------------------------------------------------------------------------------------
def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1000:
            return f"{value:.0f}"
        if abs(value) < 1e-2:
            return f"{value:.{digits - 1}e}"
        return f"{value:.{digits}g}"
    return str(value)


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(title for _, title in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def report(runs, walls, costs, accuracy, positivity, cross, same, figures, output: Path, *, study, times, linear_until,
           nonlinear_from, window, reference_rate) -> Path:
    lines = ["# Run comparison", "", f"Study: `{study}`; runs: " + ", ".join(f"`{r['run_id']}`" for r in runs) + ".",
             f"Linear phase t < {linear_until:g}, nonlinear phase t >= {nonlinear_from:g}; growth fit over "
             f"t in [{window[0]:g}, {window[1]:g}] against the reference rate {reference_rate}.", ""]
    lines += ["## Wall time and memory", "",
              _table(walls, [("label", "run"), ("p", "p"), ("triangles", "triangles"), ("kkt", "KKT"), ("status", "status"),
                             ("steps", "steps"), ("final_time", "final t"), ("wall_s", "wall (s)"),
                             ("stepping_s", "stepping (s)"), ("positivity_s", "KKT projection (s)"),
                             ("peak_process_mib", "peak process (MiB)")]), ""]
    for phase in ("linear", "nonlinear", "all"):
        rows = [row for row in costs if row["phase"] == phase]
        lines += [f"## Cost per step, {phase} phase (medians, ms)", "",
                  _table(rows, [("label", "run"), ("steps", "steps"), ("step_ms", "step"), ("poisson_ms", "Poisson"),
                                ("poisson_krylov_ms", "Poisson Krylov"), ("postprocess_ms", "postprocess"),
                                ("transport_assembly_ms", "transport assembly"), ("transport_solve_ms", "transport solve"),
                                ("positivity_ms", "KKT projection"), ("enstrophy_ms", "enstrophy budget"),
                                ("diagnostics_ms", "diagnostics (per row)"), ("poisson_its", "Poisson it."),
                                ("transport_its", "transport it."), ("flagged", "flagged elements")]), ""]
    lines += ["## Growth rate and conservation", "",
              _table(accuracy, [("label", "run"), ("gamma", "fitted rate"), ("gamma_relative_error", "relative error"),
                                ("gamma_r_squared", "R²"), ("summary_gamma", "driver's fit"),
                                ("max_abs_mass_relative_drift", "max |mass drift|"),
                                ("max_abs_energy_relative_drift", "max |energy drift|"),
                                ("max_abs_l2_budget_relative_defect", "max |L² identity defect|"),
                                ("enstrophy_initial", "enstrophy t=0"), ("enstrophy_final", "enstrophy final"),
                                ("enstrophy_relative_change", "enstrophy change"), ("min_rho_min", "min density"),
                                ("min_rho_cell_average", "min cell average"), ("max_negative_mass", "max negative mass")]), ""]
    amplitude_columns = [("label", "run")] + [(f"amplitude_t{t:g}", f"t={t:g}") for t in times]
    lines += ["## Mode amplitude at the snapshot times", "", _table(accuracy, amplitude_columns), ""]
    enstrophy_columns = [("label", "run")] + [(f"enstrophy_t{t:g}", f"t={t:g}") for t in times]
    lines += ["## Enstrophy at the snapshot times", "", _table(accuracy, enstrophy_columns), ""]
    lines += ["## KKT positivity projection", "",
              _table(positivity, [("label", "run"), ("kkt", "KKT"), ("projection_steps", "steps with a report"),
                                  ("steps_with_flagged", "steps with flagged elements"), ("first_flagged_time", "first flagged t"),
                                  ("max_flagged", "max flagged"), ("max_fallback", "max Zhang-Shu fallbacks"),
                                  ("max_negative_mean_projected", "max negative means"), ("max_mass_returned", "max mass returned"),
                                  ("max_correction_relative", "max relative correction"),
                                  ("projection_time_total_s", "projection total (s)"), ("projection_time_median_ms", "median (ms)")]), ""]
    if cross:
        lines += ["## Cross-mesh differences (relative L², the reference's volume rule integrates)", "",
                  _table(cross, [("run", "run"), ("reference", "reference"), ("time", "t"), ("field", "field"),
                                 ("relative", "relative"), ("absolute", "absolute"), ("reference_norm", "reference norm"),
                                 ("points_outside", "points outside")]), ""]
    if same:
        lines += ["## Same-mesh differences (relative coefficient L²)", "",
                  _table(same, [("run", "run"), ("other", "other"), ("time", "t"), ("field", "field"),
                                ("relative", "relative"), ("absolute", "absolute")]), ""]
    if figures:
        names = []
        for item in figures:
            for f in (item if isinstance(item, (tuple, list)) else [item]):
                if f:
                    names.append(Path(f).name)
        lines += ["## Figures", ""] + [f"- `{name}`" for name in names] + [""]
    path = output / "report.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("study", type=Path)
    parser.add_argument("--runs", required=True, help="comma-separated run ids (launcher run_id)")
    parser.add_argument("--times", type=lambda s: [float(v) for v in s.split(",")], default=list(DEFAULT_TIMES))
    parser.add_argument("--output-dir", type=Path, default=None, help="default <study>/analysis_pairs")
    parser.add_argument("--summary", default="d8_summary.json", help="driver summary file name under <study>/runs")
    parser.add_argument("--linear-until", type=float, default=40.0)
    parser.add_argument("--nonlinear-from", type=float, default=50.0)
    parser.add_argument("--window", type=lambda s: tuple(float(v) for v in s.split(",")), default=WINDOW)
    parser.add_argument("--reference-rate", type=float, default=REFERENCE_RATE)
    parser.add_argument("--figure-times", type=lambda s: [float(v) for v in s.split(",")], default=[50., 60., 70.])
    parser.add_argument("--chunk-elements", type=int, default=20000)
    parser.add_argument("--skip-cross", action="store_true", help="no cross-mesh errors (minutes per time level)")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-density-figure", action="store_true", help="no sector rasters (the slowest figure)")
    args = parser.parse_args()

    output = args.output_dir or args.study / "analysis_pairs"
    output.mkdir(parents=True, exist_ok=True)
    runs = [load_run(args.study, run_id.strip(), summary_name=args.summary) for run_id in args.runs.split(",") if run_id.strip()]
    labels = [run["label"] for run in runs]
    for run in runs:
        if labels.count(run["label"]) > 1:
            run["label"] = run["run_id"]  # two runs of the same degree and KKT setting: name them by their ids
    for run in runs:
        print(f"{run['run_id']}: p={run['p']} triangles={run['triangles']} status={run['record'].get('status')} "
              f"steps={len(run['timings'])} diagnostics={len(run['diagnostics'])} "
              f"snapshots={'yes' if run['snapshots'] else 'no'}", flush=True)

    walls = [wall_summary(run) for run in runs]
    costs = [row for run in runs for row in cost_rows(run, linear_until=args.linear_until, nonlinear_from=args.nonlinear_from)]
    accuracy = [accuracy_row(run, window=args.window, reference_rate=args.reference_rate, times=args.times) for run in runs]
    positivity = [positivity_row(run) for run in runs]
    cross, same = [], []
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            if a["snapshots"] is None or b["snapshots"] is None:
                continue
            if _same_mesh(a, b):
                same += same_mesh_rows(a, b, args.times)
            elif not args.skip_cross:
                cross += cross_rows(a, b, args.times, output / "errors.jsonl", chunk_elements=args.chunk_elements)
                cross += cross_rows(b, a, args.times, output / "errors.jsonl", chunk_elements=args.chunk_elements)
    _csv(walls, output / "wall.csv"); _csv(costs, output / "cost.csv"); _csv(accuracy, output / "accuracy.csv")
    _csv(positivity, output / "positivity.csv"); _csv(cross, output / "cross.csv"); _csv(same, output / "same_mesh.csv")
    figures = []
    if not args.skip_figures:
        step_time_figure(runs, output / "step_time", linear_until=args.linear_until, nonlinear_from=args.nonlinear_from)
        figures.append(output / "step_time.pdf")
        time_series_figure(runs, output / "time_series", reference_rate=args.reference_rate, window=args.window)
        figures.append(output / "time_series.pdf")
        cross_figure(cross, same, {run["label"]: {t: w for t, w in cumulative_wall(run, args.times).items() if w is not None}
                                   for run in runs}, output / "cross_difference", {run["run_id"]: run["label"] for run in runs})
        figures.append(output / "cross_difference.pdf")
        if not args.skip_density_figure:
            chosen = [run for run in runs if not run["kkt"]] or runs
            figures.append(density_sector_figure(chosen, output / "density_sector", times=args.figure_times))
    path = report(runs, walls, costs, accuracy, positivity, cross, same, figures, output, study=args.study, times=args.times,
                  linear_until=args.linear_until, nonlinear_from=args.nonlinear_from, window=args.window,
                  reference_rate=args.reference_rate)
    print(f"report: {path}")


if __name__ == "__main__":
    main()
