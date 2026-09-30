"""Case-specific diagnostic choices for the shared Matplotlib output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
import re

import numpy as np

from scripts.guiding_center.runtime.labels import run_label

from hdgfem.io.time_series import (
    DiagnosticPanel, TimeSeries, numeric_time_series, plot_diagnostic_panels, plot_mode_history,
    show_diagnostic_figures, diagnostic_pages,
)


_MODE_KEY = re.compile(r"diocotron_phi_mode_(\d+)_l2")
_LABELS = {
    "poisson_tau": r"Poisson stabilization $\tau_{\mathrm{P}}$",
    "mass": r"Mass $M(t)$",
    "circulation": r"Circulation $\Gamma(t)$",
    "enstrophy": r"Enstrophy $Z(t)$",
    "energy_from_q_l2": r"Electric energy $\mathcal{E}(t)$",
    "energy_relative_drift": "Electric energy: signed relative drift",
    "mass_relative_drift": "Mass: signed relative drift",
    "enstrophy_relative_drift": "Enstrophy: signed relative drift",
    "mass_drift": "Mass: absolute drift",
    "circulation_drift": "Circulation: absolute drift",
    "energy_drift": "Electric energy: absolute drift",
    "enstrophy_drift": "Enstrophy: absolute drift",
    "q_l2": r"Electric field: $L^2$ norm",
    "rho_l2_error": r"Density: $L^2$ error",
    "phi_l2_error": r"Potential: $L^2$ error",
    "rho_min": "Density: minimum", "rho_min_checked": "Density: sampled minimum",
    "rho_max": "Density: maximum",
    "phi_min": "Potential: minimum", "phi_max": "Potential: maximum",
    "diocotron_phi_eq_l2": "Diocotron instability: potential departure",
    "diocotron_phi_eq_relative_l2": "Potential departure: relative equilibrium error",
}


_AXIS_LABELS = {
    "mass": r"$M(t)=\int_\Omega \rho_h(t)\,\mathrm{d}x$",
    "circulation": r"$\Gamma(t)=\int_\Omega \rho_h(t)\,\mathrm{d}x$",
    "energy_from_q_l2": r"$\mathcal{E}(t)=\frac{1}{2}\|\mathbf{E}_h(t)\|_{L^2(\Omega)}^2$",
    "enstrophy": r"$Z(t)=\frac{1}{2}\|\rho_h(t)\|_{L^2(\Omega)}^2$",
    "energy_relative_drift": r"$\frac{\mathcal{E}(t)-\mathcal{E}(0)}{|\mathcal{E}(0)|}$",
    "mass_relative_drift": r"$\frac{M(t)-M(0)}{|M(0)|}$",
    "enstrophy_relative_drift": r"$\frac{Z(t)-Z(0)}{|Z(0)|}$",
    "mass_drift": r"$M(t)-M(0)$", "circulation_drift": r"$\Gamma(t)-\Gamma(0)$",
    "energy_drift": r"$\mathcal{E}(t)-\mathcal{E}(0)$", "enstrophy_drift": r"$Z(t)-Z(0)$",
    "q_l2": r"$\|\mathbf{E}_h(t)\|_{L^2(\Omega)}$",
    "rho_l2_error": r"$\|e_\rho(t)\|_{L^2(\Omega)}$",
    "phi_l2_error": r"$\|e_\phi(t)\|_{L^2(\Omega)}$",
    "poisson_tau": r"$\tau_{\mathrm{P}}(t)$",
    "rho_min": r"$\min_\Omega \rho_h(t)$", "rho_max": r"$\max_\Omega \rho_h(t)$",
    "rho_min_checked": r"$\min_{x\in X_h}\rho_h(x,t)$",
    "phi_min": r"$\min_\Omega \phi_h(t)$", "phi_max": r"$\max_\Omega \phi_h(t)$",
    "diocotron_phi_eq_l2": r"$\|\phi_h(t)-\phi_{\mathrm{eq},h}\|_{L^2(\Omega)}$",
    "diocotron_phi_eq_relative_l2": r"$\frac{\|\phi_h(t)-\phi_{\mathrm{eq},h}\|_{L^2(\Omega)}}{\|\phi_{\mathrm{eq},h}\|_{L^2(\Omega)}}$",
}


def _panel_group(key):
    if key.startswith("relative_") and key.endswith("_conservation"):
        return "conservation"
    if key in {"mass", "circulation", "energy_from_q_l2", "enstrophy"}:
        return "conservation"
    if "drift" in key:
        return "conservation_drifts"
    if key.startswith("diocotron_instability") or "diocotron_phi_eq" in key:
        return "diocotron_instability"
    if key.startswith("diocotron"):
        return "diocotron_modal_diagnostics"
    if "tau" in key:
        return "poisson_stabilization"
    if "positivity" in key or "negative" in key or "bernstein" in key or "checked" in key or "cell_average" in key:
        return "density_positivity"
    if key.startswith(("rho_", "phi_", "q_", "velocity_", "beta_")):
        return "fields_and_velocity"
    for solver in ("poisson", "transport"):
        if key.startswith(solver):
            if "residual" in key or "rhs_norm" in key:
                return f"{solver}_solver_residuals"
            if "bytes" in key or "nnz" in key or "permutation_size" in key:
                return f"{solver}_memory_and_sparsity"
            if "iteration" in key or "count" in key or "order" in key:
                return f"{solver}_iterations_and_counts"
            if "assembly" in key or "reconstruction" in key or "factor" in key:
                return f"{solver}_assembly_and_reconstruction"
            return f"{solver}_solver_timings"
    if "diagnostic" in key:
        return "diagnostic_overhead"
    if key.startswith(("first_", "initial", "equilibrium")):
        return "initialization"
    return "time_integration"


def poisson_tau_caption(rows, timing_rows=()):
    values = [float(row["poisson_tau"]) for row in (*rows, *timing_rows)
              if row.get("poisson_tau") is not None and np.isfinite(row["poisson_tau"])]
    if values:
        lower, upper = min(values), max(values)
        if lower == upper:
            return rf"$\tau_{{\mathrm{{Poisson}}}}={lower:g}$"
        return rf"$\tau_{{\mathrm{{Poisson}}}}(t)\in[{lower:g},\,{upper:g}]$"
    initial = rows[0].get("run_configuration", {}).get("poisson_tau_initial") if rows else None
    if initial is not None and np.isfinite(initial):
        return rf"$\tau_{{\mathrm{{Poisson}}}}(0)={initial:g}$ (later values unrecorded)"
    return r"$\tau_{\mathrm{Poisson}}$: unrecorded"


def diagnostic_panels(rows, timing_rows=()):
    """Include every recorded numeric history, with case-appropriate axes."""
    exclude = ("run_configuration", "diocotron_sharp_annulus_reference")
    histories = numeric_time_series(rows, exclude=exclude)
    timings = numeric_time_series(timing_rows, exclude=exclude)
    panels = []
    # Give the user's instability norm a dedicated, prominently placed panel.
    if "diocotron_phi_eq_l2" in histories:
        series = histories.pop("diocotron_phi_eq_l2")
        panels.append(DiagnosticPanel(
            "Diocotron instability", _AXIS_LABELS[series.label], (series,), "log", "log", "diocotron_instability"))
        panels.append(DiagnosticPanel(
            "Potential perturbation: exponential-growth view", _AXIS_LABELS[series.label],
            (series,), "linear", "log", "diocotron_instability_semilog"))
    # Reuse signed drift measurements; magnitudes make conservation errors
    # explicit while the original signed histories remain available below.
    for drift_key, total_key, symbol, title, filename in (
        ("energy_relative_drift", "energy_from_q_l2", r"\mathcal{E}", "Electric energy: relative conservation error", "relative_electric_energy_conservation"),
        ("mass_relative_drift", "mass", "M", "Mass: relative conservation error", "relative_mass_conservation"),
    ):
        if total_key == "mass":
            # Use the actual initial signed mass, including Euler cases whose
            # legacy mass_relative_drift records are intentionally null.
            baseline = rows[0].get("mass") if rows and rows[0].get("time") == 0 else None
            defined = baseline is not None and np.isfinite(baseline) and baseline != 0
            times = np.asarray([row.get("time", np.nan) for row in rows], dtype=float)
            values = np.asarray([
                (row["mass"] - baseline) / baseline
                if defined and row.get("mass") is not None else np.nan
                for row in rows
            ], dtype=float)
            mass_title = "Mass: signed relative error"
            if not defined:
                mass_title += (r" (undefined: $M(0)=0$)" if baseline == 0
                               else " (initial mass unavailable)")
            panels.append(DiagnosticPanel(
                mass_title, r"$\frac{M(t)-M(0)}{M(0)}$",
                (TimeSeries("Relative mass error", times, values),),
                "linear", "symlog", filename))
            continue
        measured = histories.get(drift_key)
        # Backfill old records only when the field was absent, never when it
        # was deliberately null (e.g. zero-circulation signed Euler gas).
        if measured is None and rows and not any(drift_key in row for row in rows):
            baseline = rows[0].get(total_key)
            if rows[0].get("time") == 0 and baseline is not None and np.isfinite(baseline) and baseline != 0:
                from hdgfem.diagnostics.solver import relative_drift

                times = np.asarray([row.get("time", np.nan) for row in rows])
                values = np.asarray([
                    relative_drift(row[total_key], baseline)
                    if row.get(total_key) is not None else np.nan for row in rows
                ])
                if np.count_nonzero(np.isfinite(times) & np.isfinite(values)) >= 2:
                    measured = TimeSeries(drift_key, times, values)
        if measured is not None:
            error = TimeSeries(title, measured.times, np.abs(measured.values))
            positive = np.any(np.isfinite(error.values) & (error.values > 0))
            panels.append(DiagnosticPanel(
                title, rf"$\frac{{|{symbol}(t)-{symbol}(0)|}}{{|{symbol}(0)|}}$", (error,),
                "log" if positive else "linear", "log" if positive else "linear", filename))
    # Timings have the denser every-step cadence. Prefer them for duplicated
    # solver metrics while keeping endpoint physics at its recorded cadence.
    for key, series in timings.items():
        if key not in histories or any(word in key for word in ("poisson", "transport", "step_wall", "solve", "assembly")):
            histories[key] = series
    # Preserve endpoint-only tau samples (especially t=0) alongside denser timings.
    tau_samples = {}
    for row in (*rows, *timing_rows):
        time, value = row.get("time"), row.get("poisson_tau")
        if time is not None and value is not None and np.isfinite(time) and np.isfinite(value):
            tau_samples[float(time)] = float(value)
    if len(tau_samples) >= 2:
        times = np.asarray(sorted(tau_samples))
        histories["poisson_tau"] = TimeSeries(
            "poisson_tau", times, np.asarray([tau_samples[t] for t in times]))
    # Compare repeated stages in the same panel, retaining every curve.
    grouped = {}
    for key, series in histories.items():
        if _MODE_KEY.fullmatch(key):
            continue  # Every mode amplitude appears in the spectrum.
        stage = re.match(r"stage(\d+)_(.*)", key)
        metric = stage.group(2) if stage else key
        labels = [f"Stage {stage.group(1)}"] if stage else []
        if re.search(r"\.\d+(?:\.|$)", metric):
            labels.extend(f"Entry {int(number)+1}" for number in re.findall(r"\.(\d+)(?:\.|$)", metric))
            metric = re.sub(r"\.\d+(?=\.|$)", "", metric)
        grouped.setdefault(metric, []).append(TimeSeries(
            ", ".join(labels) or "Accepted", series.times, series.values))
    priority = ("mass", "circulation", "energy_from_q_l2", "enstrophy")
    keys = [key for key in priority if key in grouped] + sorted(set(grouped)-set(priority))
    for key in keys:
        series_group = tuple(grouped[key])
        finite = np.concatenate([series.values[np.isfinite(series.values)] for series in series_group])
        xscale = yscale = "linear"
        if key in priority:
            pass  # Conserved quantities use linear axes, including energy_from_q_l2.
        elif "drift" in key or "phase" in key:
            yscale = "linear" if "phase" in key else "symlog"
        elif any(word in key for word in ("error", "residual", "_l2", "_linf", "amplitude", "harmonic_ratio")):
            if np.any(finite > 0) and np.all(finite >= 0):
                xscale = yscale = "log"
            elif np.any(finite < 0):
                yscale = "symlog"
        label = _LABELS.get(key, key.replace("_", " ").replace(".", " / "))
        if key.endswith("_time") or "_wall_time" in key:
            ylabel = r"Wall time $[\mathrm{s}]$"
        elif "iteration" in key or key.endswith("_count"):
            ylabel = r"Count $N$"
        else:
            ylabel = "Value"
        if "residual" in key:
            ylabel = r"$r_{\mathrm{rel}}$" if "rel" in key.replace("residual", "") else r"$\|r\|_2$"
        elif "bytes" in key:
            ylabel = r"Memory $[\mathrm{bytes}]$"
        elif "error" in key:
            ylabel = r"Error $\varepsilon$"
        ylabel = _AXIS_LABELS.get(key, ylabel)
        panels.append(DiagnosticPanel(label, ylabel, series_group, xscale, yscale, key,
                                      drawstyle="steps-post" if key == "poisson_tau" else "default"))
    return [replace(panel, group=_panel_group(panel.filename or panel.title)) for panel in panels]



def interactive_diagnostic_panels(panels):
    """Select three compact overview pages; modal activity gets a fourth."""
    available = {panel.filename: panel for panel in panels}
    selected = []

    def choose(*keys):
        return next((available[key] for key in keys if key in available), None)

    def page(group, members):
        selected.extend(replace(panel, group=group) for panel in members if panel is not None)

    def conservation(*keys):
        panel = choose(*keys)
        if panel is None:
            return None
        symbol = {"relative_mass_conservation": "M",
                  "relative_electric_energy_conservation": r"\mathcal{E}",
                  "enstrophy_relative_drift": "Z", "enstrophy_drift": "Z"}[panel.filename]
        numerator = rf"|{symbol}(t)-{symbol}(0)|"
        ylabel = (rf"$\frac{{{numerator}}}{{|{symbol}(0)|}}$"
                  if "relative" in panel.filename else rf"${numerator}$")
        # Display-only copies: neither recorded rows nor saved panels lose signs.
        curves = tuple(replace(curve, values=np.abs(curve.values)) for curve in panel.series)
        positive = any(np.any(np.isfinite(curve.values) & (curve.values > 0)) for curve in curves)
        return replace(
            panel, series=curves, ylabel=ylabel,
            title=panel.title.replace("signed relative error", "relative conservation error")
                             .replace("signed relative drift", "relative conservation error")
                             .replace("absolute drift", "drift magnitude"),
            yscale="log" if positive else "linear",
        )

    def comparison(title, ylabel, choices, *, logarithmic=False):
        curves = []
        for label, keys in choices:
            panel = choose(*keys)
            if panel is not None:
                curves.extend(TimeSeries(
                    f"{label} ({curve.label})" if len(panel.series) > 1 else label,
                    curve.times, curve.values) for curve in panel.series)
        if not curves:
            return None
        return DiagnosticPanel(title, ylabel, tuple(curves),
                               "log" if logarithmic else "linear",
                               "log" if logarithmic else "linear")

    # Lead with errors against the initial invariants. Actual mass and energy
    # remain in the full saved collection, not the compact interactive overview.
    # Undefined relative mass errors are labelled, never replaced by absolute drift.
    page("physics_conservation", [
        conservation("relative_mass_conservation"),
        conservation("relative_electric_energy_conservation"),
        conservation("enstrophy_relative_drift", "enstrophy_drift"),
    ])
    # Keep the requested instability norm on log-log axes, plus its
    # exponential-growth view. Omit absent quantities rather than filling
    # the screen with unrelated implementation counters.
    dynamics = [
        choose("diocotron_instability"),
        choose("diocotron_instability_semilog"),
        choose("diocotron_phi_eq_relative_l2", "rho_l2_error"),
        choose("rho_min", "rho_min_checked"),
        choose("rho_max"),
        choose("phi_l2_error", "q_l2"),
    ]
    page("field_evolution_and_instability", dynamics)
    page("solvers_and_essential_timings", [
        comparison("Physical relative solver residuals", r"$\|b-Ax\|_2/\|b\|_2$", [
            ("Poisson", ("poisson_physical_rel_residual", "poisson_solver_rel_residual")),
            ("Transport", ("transport_physical_rel_residual", "transport_solver_rel_residual")),
        ], logarithmic=True),
        comparison("Solver iterations", r"Iterations $N$", [
            ("Poisson", ("poisson_solver_iterations",)),
            ("Transport", ("transport_solver_iterations",)),
        ]),
        comparison("Step and solver wall times", r"Wall time $[\mathrm{s}]$", [
            ("Step", ("linear_step_wall_time", "step_wall_time")),
            ("Poisson", ("poisson_step_wall_time", "poisson_time")),
            ("Transport", ("transport_step_wall_time", "transport_time")),
        ]),
        comparison("Assembly and solve costs", r"Wall time $[\mathrm{s}]$", [
            ("Poisson assembly", ("poisson_step_time_assembly", "poisson_time_assembly")),
            ("Poisson solve", ("poisson_step_time_solve", "poisson_time_solve")),
            ("Transport assembly", ("transport_step_time_assembly", "transport_time_assembly")),
            ("Transport solve", ("transport_step_time_solve", "transport_time_solve")),
        ]),
        choose("poisson_tau"),
        choose("imex_ark3_embedded_error_relative", "diagnostics_wall_time", "diagnostics_time"),
    ])
    return selected


def render_guiding_center_diagnostic_plots(config, rows, timing_rows=(), *, prefix=None):
    """Display and/or save recorded figures according to independent CLI flags."""
    if not (config.plot_diagnostics or config.save_diagnostics) or not (rows or timing_rows):
        return []
    if config.save_diagnostics and config.plot_diagnostics:
        # A Qt platform-plugin abort cannot be caught by Python. Save with
        # standalone Agg canvases first, before constructing any GUI figures.
        paths = render_guiding_center_diagnostic_plots(
            replace(config, plot_diagnostics=False), rows, timing_rows, prefix=prefix)
        render_guiding_center_diagnostic_plots(
            replace(config, save_diagnostics=False), rows, timing_rows, prefix=prefix)
        return paths
    metadata = rows[0].get("run_configuration", {}) if rows else {}
    model_label = metadata.get("run_label") or run_label(config)
    stem = prefix or config.diagnostics_prefix
    directory = Path(config.diagnostics_dir)/f"{stem}_diagnostic_plots" if config.save_diagnostics else None
    panels = diagnostic_panels(rows, timing_rows)
    if config.plot_diagnostics:
        panels = interactive_diagnostic_panels(panels)
    tau_caption = poisson_tau_caption(rows, timing_rows)
    paths = plot_diagnostic_panels(panels, directory,
                                  title=f"{model_label}\n{tau_caption}", display=config.plot_diagnostics)
    descriptions = {filename: [panel.title for panel in page]
                    for filename, page in diagnostic_pages(panels)}
    mode_keys = sorted(
        (int(match.group(1)), key)
        for key in {key for row in rows for key in row}
        if (match := _MODE_KEY.fullmatch(key))
    )
    if mode_keys and len(rows) >= 2:
        times = np.asarray([row["time"] for row in rows])
        values = np.asarray([[row.get(key, np.nan) for _, key in mode_keys] for row in rows], dtype=float)
        parameters = rows[0].get("run_configuration", {}).get("case_parameters", {})
        paths.extend(plot_mode_history(
            times, [mode for mode, _ in mode_keys], values, directory,
            title=f"{model_label}\nActive angular modes · {tau_caption}", target_mode=parameters.get("k"), display=config.plot_diagnostics,
        ))
    if mode_keys and len(rows) >= 2 and config.save_diagnostics:
        descriptions["active_modes.png"] = ["Angular-mode spectrum, dominant modes, amplitudes, and active count"]
    if paths:
        manifest = {
            "run_label": model_label,
            "figure_descriptions": descriptions,
            "poisson_tau": tau_caption,
            "conservation_errors": "Signed relative mass error (M(t)-M(0))/M(0); electric-energy error is an absolute relative drift. Zero initial mass makes its relative error undefined.",
            "figures": [file.name for file in paths],
            "instability": "diocotron_phi_eq_l2 = ||phi - phi_eq||_L2",
            "active_mode_threshold": "amplitude >= 0.001 * strongest resolved mode",
            "log_axes": "Only finite, strictly positive time/value samples are displayed.",
            "histories": "All numeric endpoint and timing histories; full modal amplitudes share a heatmap.",
        }
        (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(f"[gc] diagnostic figures saved: {directory}", flush=True)
    if config.plot_diagnostics and (panels or mode_keys):
        print("[gc] opening diagnostic windows; close them to return to the terminal.", flush=True)
        show_diagnostic_figures()
    return paths


def main():
    """Render saved logs without running a simulation."""
    from argparse import ArgumentParser
    from dataclasses import replace
    from scripts.guiding_center.cases.guiding_center_presets import DEFAULT_PRESET, PRESETS

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-diagnostics", action="store_true", help="display the figures (default when not saving)")
    parser.add_argument("--save-diagnostics", action="store_true", help="write PNG/PDF figures")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.diagnostics.read_text().splitlines() if line.strip()]
    timing_path = args.diagnostics.with_name(args.diagnostics.stem+"_timings.jsonl")
    timings = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()] if timing_path.exists() else []
    metadata = rows[0].get("run_configuration", {}) if rows else {}
    saved_preset = PRESETS.get(metadata.get("preset"), PRESETS[DEFAULT_PRESET])
    config = replace(saved_preset,
                     plot_diagnostics=args.plot_diagnostics or not args.save_diagnostics,
                     save_diagnostics=args.save_diagnostics,
                     diagnostics_dir=str(args.output_dir or args.diagnostics.parent),
                     diagnostics_prefix=args.diagnostics.stem,
                     time_scheme=metadata.get("time_scheme", saved_preset.time_scheme))
    for file in render_guiding_center_diagnostic_plots(config, rows, timings):
        print(file)


if __name__ == "__main__":
    main()
