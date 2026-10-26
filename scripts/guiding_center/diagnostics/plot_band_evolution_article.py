"""Article figures and table of E7 (Section 7.2) from the band-evolution runs, finished or still running.

Finished runs are read from ``<prefix>_snapshots.npz``; running ones from the spilled states
``<prefix>_states/state_*.npz`` with the mesh rebuilt from the archived checkpoint named in
``<prefix>_config.json`` (no solve). Writes into ``--article``:

- ``figures/e7_preservation.{pdf,png}``: control drift of rho_h and phi_h from the projected
  equilibrium, angular departure max_{m>=1} A_m(rho_h) of controls and seeds, log10 A_m(t) of one run;
- ``figures/e7_<geometry>_<kind>.{pdf,png}``: density at the first time, then (phi_h - phi_eq)/max at
  later times (``--deviation``), or density only (``--density``);
- ``tables/e7_preservation.tex``: one row per control (transfer, drift, onset, conservation, final time);
- ``tables/e7_rates.tex``: growth rates. Seeded runs: the plateau rate of the seeded band mode (``seeded_growth``)
  and its fit window when the window covers at least one e-folding, otherwise the largest
  local rate while the seeded amplitude is below twice its initial value (marked *); controls: the fastest
  band mode of the noise-seeded growth (``modal_growth_rates``, window from t = 3 to the first time a mode
  reaches 1e-2). Each against the straight vortex-strip rate of that mode. ``--extra-seed`` adds seeded runs
  of other studies (e.g. an exploration at lower degree);
- ``figures/e7_activity.{pdf,png}`` (``--activity``): active band modes (amplitude above 1 % of the largest)
  and coherent vortices (cores of {rho_h > 1/2} eroded by 0.4 band thicknesses, in regions holding at least
  2 % of the initial band area; counted on the saved states) against time, one curve per run;
- ``figures/e7_vortex_tracks.{pdf,png}`` (``--tracks <analysis>/e7_summary.json``): band curvature against the
  normalized arclength s of the middle contour, and the coherent vortices of each perturbed run on the (s, t) plane;
- ``tables/e7_conservation.tex``: largest relative drifts of mass and energy, relative change of enstrophy,
  largest L2 identity defect and minimum density over each run;
- ``tables/e7_search.tex`` (``--search``): the perturbation search, one row per perturbed run (seed shape, mode,
  edge displacement, degree, final time) with its coherent-vortex counts N: largest N, the longest stretch of
  saved states with N equal to the seeded mode, N at the final time. Counts are cached in
  ``<prefix>_coherent.json`` next to each run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.guiding_center.diagnostics.analyze_band_evolution import (
    COLORS, GEOMETRIES, NAMES, angular_departure, band_parameters, boundary_segments, find_run, first_crossing,
    modal_growth_rates, plt, save, seeded_growth, series, spectra, style, value_at, vortex_strip_rate)


# ------------------------------------------------------------------------------------------- states
class RunStates:
    """Saved states of one run: ``times``, ``state(i) -> (density, potential)``, ``space``, ``phi_eq``."""

    def __init__(self, runs: Path, stem: str):
        from hybridge.core.mesh import DGMesh
        from hybridge.core.space import DGSpace
        snapshots = runs / f"{stem}_snapshots.npz"
        if snapshots.exists():
            self._npz = np.load(snapshots)
            self.times = np.asarray(self._npz["times"], float)
            self.phi_eq = np.asarray(self._npz["equilibrium_potential"])
            mesh = DGMesh(self._npz["node_coords"], self._npz["triangles"])
            order, basis = int(self._npz["order"]), str(self._npz["basis"])
            self._files = None
        else:
            spill = runs / f"{stem}_states"
            self._files = sorted(spill.glob("state_*.npz"))
            if not self._files:
                raise FileNotFoundError(f"no snapshots or spilled states for {stem}")
            self.times = np.array([float(np.load(f)["time"]) for f in self._files])
            self.phi_eq = np.load(spill / "equilibrium_potential.npy")
            config = json.loads((runs / f"{stem}_config.json").read_text())
            mesh = self._checkpoint_mesh(config["case_params"])
            order, basis = int(config["order"]), str(config["basis"])
        self.space = DGSpace(mesh, order, basis_type=basis)
        self.mesh = mesh

    @staticmethod
    def _checkpoint_mesh(params):
        from scripts.guiding_center.cases.imported_equilibrium import WindowEquilibrium, archive_checkpoint
        if params.get("remesh_size"):
            raise ValueError("spilled states of remeshed runs need the finished snapshot file")
        checkpoint, entry = archive_checkpoint(params["archive"], params["equilibrium"])
        return WindowEquilibrium(checkpoint, entry=entry, import_diagnostics="none").to_mesh()

    def state(self, index):
        if self._files is None:  # each npz access decompresses the whole array: read it once
            if not hasattr(self, "_fields"):
                self._fields = {key: self._npz[key] for key in ("density", "potential")}
            return self._fields["density"][index], self._fields["potential"][index]
        z = np.load(self._files[index])
        return z["density"], z["potential"]


class Raster:
    def __init__(self, states: RunStates, width: int):
        from hybridge.io.raster import RasterGeometry
        lo, hi = states.mesh.node_coords.min(axis=0), states.mesh.node_coords.max(axis=0)
        self.width, self.height = width, max(16, int(round(width * (hi[1] - lo[1]) / (hi[0] - lo[0]))))
        geometry = RasterGeometry.from_mesh(states.mesh, self.width, self.height)
        self.matrix = geometry.sampling_matrix(states.space, max_bytes=2 * 1024**3)
        self.inside = geometry.element_ids >= 0
        self.extent = tuple(float(b) for b in geometry.bounds)
        self.walls = boundary_segments(states.mesh)

    def __call__(self, coeffs):
        values = np.full(self.width * self.height, np.nan)
        values[self.inside] = (self.matrix @ np.asarray(coeffs).ravel())[self.inside]
        return values.reshape(self.height, self.width)

    def frame(self, ax):
        ax.plot(self.walls[:, :, 0].T, self.walls[:, :, 1].T, color="#222222", lw=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)


# ------------------------------------------------------------------------------------------ figures
TEXT_WIDTH = 7.2  # inches: figures are drawn at the article's text width, fonts at print size


def article_style():
    style()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "lines.linewidth": 1.1})


def article_preservation_figure(data, out, spectrum):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.3), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=(1, 1, 1.25)))
    for g, pair in data.items():
        control, seed = pair.get("control"), pair.get("seed")
        if control is not None:
            for key, ls, label in (("diocotron_rho_eq_relative_l2", "-", r"$\rho_h$"),
                                   ("diocotron_phi_eq_relative_l2", "--", r"$\phi_h$")):
                t, v = series(control["rows"], key)
                axes[0].semilogy(t[v > 0], v[v > 0], ls, color=COLORS[g], label=f"{NAMES[g]}, {label}")
        for kind, run, ls in (("control", control, "-"), ("seed", seed, "--")):
            if run is not None:
                t, v = angular_departure(run["rows"])
                axes[1].semilogy(t[v > 0], v[v > 0], ls, color=COLORS[g], label=f"{NAMES[g]} {kind}")
    axes[0].set_title("(a) control: distance to equilibrium")
    axes[1].set_title(r"(b) angular departure $\max_{m\geq1}A_m(\rho_h)$")
    axes[1].set_ylim(bottom=1e-10)  # A_m(0) of a control is exactly zero
    for ax in axes[:2]:
        ax.set_xlabel("t")
    # one legend for the domains; the caption gives the line styles (a: rho solid, phi dashed; b: control solid,
    # perturbed dashed)
    handles = [plt.matplotlib.lines.Line2D([], [], color=COLORS[g], lw=1.2) for g in data]
    fig.legend(handles, [NAMES[g] for g in data], loc="outside lower center", ncol=len(data))
    g, kind = spectrum
    run = (data.get(g) or {}).get(kind)
    ax = axes[2]
    if run is not None:
        t, A = spectra(run["rows"])
        image = ax.imshow(np.log10(np.maximum(A[:, 1:], 1e-14)).T, origin="lower", aspect="auto", cmap="Blues",
                          vmin=-9, vmax=0, extent=(t[0], t[-1], 0.5, A.shape[1] - 0.5), interpolation="nearest")
        fig.colorbar(image, ax=ax, label=r"$\log_{10}A_m$")
        ax.set_title(f"(c) {NAMES[g]} {kind}: band modes")
        ax.set_xlabel("t")
        ax.set_ylabel("band mode m")
        ax.grid(False)
    save(fig, out / "e7_preservation")


def snapshot_figure(states: RunStates, panels, out, stem, *, width=900, columns=4, title=None):
    """``panels``: (field, time) pairs, field 'rho' (signed density colours, shared scale) or 'phi'
    ((phi_h - phi_eq)/max per panel, with the band edges rho_h = 1/2 drawn over it). Times not yet saved
    are dropped."""
    from hybridge.io.figures import signed_density_colors
    raster = Raster(states, width)
    chosen = []
    for field, time in panels:
        if time <= states.times[-1] + 1e-9:
            chosen.append((field, int(np.argmin(np.abs(states.times - time)))))
    n = len(chosen)
    rows = -(-n // columns)
    panel = (TEXT_WIDTH - 0.9) / columns
    fig, grid = plt.subplots(rows, columns, figsize=(TEXT_WIDTH, rows * panel * raster.height / raster.width + 0.35),
                             constrained_layout=True, squeeze=False)
    for ax in grid.ravel()[n:]:
        ax.set_visible(False)
    axes = grid.ravel()
    densities = {i: raster(states.state(index)[0]) for i, (field, index) in enumerate(chosen) if field == "rho"}
    images = {}
    if densities:
        cmap, norm = signed_density_colors(min(np.nanmin(image) for image in densities.values()), 1.0)
    for i, (field, index) in enumerate(chosen):
        ax = axes[i]
        if field == "rho":
            images["rho"] = ax.imshow(densities[i], extent=raster.extent, origin="upper", cmap=cmap, norm=norm,
                                      interpolation="none")
            ax.set_title(f"t = {states.times[index]:g}")
        else:
            density, potential = states.state(index)
            delta = raster(potential - states.phi_eq)
            peak = np.nanmax(np.abs(delta))
            images["phi"] = ax.imshow(delta / peak, extent=raster.extent, origin="upper", cmap="RdBu_r", vmin=-1,
                                      vmax=1, interpolation="none")
            # band edges at this time (rho_h = 1/2), so the departure can be read against the band
            ax.contour(np.nan_to_num(raster(density)), levels=[0.5], extent=raster.extent, origin="upper",
                       colors="#222222", linewidths=0.35)
            ax.set_title(f"t = {states.times[index]:g}")
            ax.text(0.5, 0.0, f"max {peak:.0e}", transform=ax.transAxes, ha="center", va="top", fontsize=6)
        raster.frame(ax)
    if "rho" in images:
        fig.colorbar(images["rho"], ax=[axes[i] for i in densities], shrink=0.8, label=r"$\rho_h$", location="left")
    if "phi" in images:
        fig.colorbar(images["phi"], ax=[axes[i] for i in range(n) if i not in densities], shrink=0.8,
                     label=r"$(\phi_h-\phi_h^{\rm eq})/\max|\cdot|$")
    if title:
        fig.suptitle(title)
    save(fig, out / stem)


def coherent_counts(states: RunStates, thickness, width=900):
    """Coherent vortices at every saved state (finished or running run), ``analyze_band_evolution.coherent``."""
    from scripts.guiding_center.diagnostics.analyze_band_evolution import COHERENT_WIDTH, coherent, vortex_census
    raster = Raster(states, width)
    counts, band_area = [], None
    for index in range(len(states.times)):
        census = vortex_census([raster(states.state(index)[0])], raster.extent,
                               core_radius=COHERENT_WIDTH * thickness)[0]
        band_area = sum(census["areas"]) if band_area is None else band_area
        counts.append(len(coherent(census, band_area, thickness)))
    return states.times, np.asarray(counts)


def cached_coherent_counts(runs: Path, stem: str, thickness, width=900):
    """``coherent_counts`` of a run, cached in ``<stem>_coherent.json`` and recomputed once more states are saved."""
    from scripts.guiding_center.diagnostics.analyze_band_evolution import COHERENT_FRACTION, COHERENT_WIDTH
    path = runs / f"{stem}_coherent.json"
    states = RunStates(runs, stem)
    criteria = dict(width=width, fraction=COHERENT_FRACTION, core_width=COHERENT_WIDTH)
    if path.exists():
        cached = json.loads(path.read_text())
        if len(cached["times"]) == len(states.times) and all(cached.get(k) == v for k, v in criteria.items()):
            return np.asarray(cached["times"]), np.asarray(cached["counts"])
    times, counts = coherent_counts(states, thickness, width)
    path.write_text(json.dumps(dict(times=times.tolist(), counts=counts.tolist(), **criteria)) + "\n")
    return times, counts


ACTIVITY_COLORS = ("#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # neutral, then slots 1-5


def activity_figure(entries, out):
    """``entries``: (label, run, runs dir) per curve, in legend order."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8), constrained_layout=True)
    for (label, run, runs), color in zip(entries, ACTIVITY_COLORS):
        t, active = series(run["rows"], "band_rho_active_modes")
        axes[0].plot(t, active, color=color, lw=0.9, label=label)
        times, counts = cached_coherent_counts(runs, run["jsonl"].stem, band_parameters(run)["thickness"])
        axes[1].step(times, counts, where="post", color=color, lw=1.1, label=label)
    axes[0].set_title("(a) active band modes")
    axes[1].set_title("(b) coherent vortices")
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_yticks([0, 1, 5, 10, 15, 20, 40])
    axes[1].get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    for ax in axes:
        ax.set_xlabel("t")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    save(fig, out / "e7_activity")


def tracks_figure(summary_path: Path, out):
    """Curvature of the middle contour (top) and coherent vortex positions s(t) of the perturbed runs."""
    from hybridge.cases.band_displacement import BandCoordinates
    from scripts.guiding_center.diagnostics.analyze_band_evolution import contour_curvature
    results = json.loads(Path(summary_path).read_text())
    present = [g for g in GEOMETRIES if (results.get(g) or {}).get("seed_census")]
    fig, axes = plt.subplots(2, len(present), figsize=(TEXT_WIDTH, 3.7), constrained_layout=True, squeeze=False,
                             sharex=True, gridspec_kw=dict(height_ratios=(1, 2.4)))
    for col, g in enumerate(present):
        r = results[g]
        coords = BandCoordinates.from_polyline(np.load(Path(r["stage"]) / "contours.npz")["middle"])
        s_curve, curvature = contour_curvature(coords, smoothing=0.25 * r["thickness"])
        axes[0, col].plot(s_curve, curvature, color="#52514e", lw=0.8)
        axes[0, col].axhline(0, color="#c3c2b7", lw=0.5)
        axes[0, col].set_title(NAMES[g])
        census = r["seed_census"]
        for t, centres in zip(census["times"], census["centroids"]):
            if len(centres) < 2:  # the unbroken band
                continue
            _, _, s_band, _, _ = coords.evaluate(np.asarray(centres))
            axes[1, col].plot(s_band, np.full(len(s_band), t), "o", ms=1.3, color=COLORS[g], markeredgewidth=0)
        axes[1, col].set_xlim(0, 1)
        axes[1, col].set_xlabel("s")
    axes[0, 0].set_ylabel("curvature")
    axes[1, 0].set_ylabel("t")
    save(fig, out / "e7_vortex_tracks")


def conservation_table(data, path: Path):
    from scripts.guiding_center.diagnostics.analyze_band_evolution import summary_conservation
    lines = ["% Generated by scripts/guiding_center/diagnostics/plot_band_evolution_article.py",
             "\\begin{tabular}{llrrrrr}\\toprule",
             "domain & run & mass & energy & enstrophy & $L^2$ identity & $\\min\\rho_h$\\\\\\midrule"]
    for g, pair in data.items():
        for kind in ("control", "seed"):
            run = pair.get(kind)
            if run is None or len(run["rows"]) < 10:
                continue
            c = summary_conservation(run)
            t, enstrophy = series(run["rows"], "enstrophy_relative_drift")
            lines.append(" & ".join([NAMES[g], "control" if kind == "control" else "perturbed",
                                     tex_number(c["max_abs_mass_relative_drift"]),
                                     tex_number(c["max_abs_energy_relative_drift"]),
                                     f"${100 * enstrophy[-1]:+.0f}\\%$" if len(enstrophy) else "--",
                                     tex_number(c["max_abs_l2_budget_relative_defect"]),
                                     f"${c['rho_min']:.2f}$"]) + "\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


# -------------------------------------------------------------------------------------------- table
def tex_number(value, digits=1):
    if value is None:
        return "--"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def preservation_table(data, path: Path):
    lines = ["% Generated by scripts/guiding_center/diagnostics/plot_band_evolution_article.py",
             "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lrrrrrrrrrrr}\\toprule",
             "domain & triangles & $\\ell$ & $e_\\phi^{\\rm transfer}$ & $\\rho$ drift & \\multicolumn{2}{c}{$\\phi$ drift} & "
             "\\multicolumn{2}{c}{$\\max_{m\\geq1}A_m$} & $t(10^{-3})$ & mass & $t_{\\rm end}$\\\\",
             " & & & & $t=1$ & $t=1$ & $t=50$ & $t=1$ & $t=50$ & & & \\\\\\midrule"]
    for g, pair in data.items():
        control, seed = pair.get("control"), pair.get("seed")
        if control is None:
            continue
        rows = control["rows"]
        mode = band_parameters(seed).get("mode") if seed else None
        transfer = rows[0].get("equilibrium_poisson_cg_relative_l2")
        _, rho = series(rows, "diocotron_rho_eq_relative_l2")
        t, phi = series(rows, "diocotron_phi_eq_relative_l2")
        ta, ang = angular_departure(rows)
        _, mass = series(rows, "mass_relative_drift")
        triangles = ((control.get("seed_data") or {}).get("case_parameters") or {}).get("source_cells")
        lines.append(" & ".join([
            NAMES[g], f"{triangles:,}".replace(",", "{,}") if triangles else "--", str(mode) if mode else "--",
            tex_number(transfer), tex_number(value_at(t, rho, 1.0)), tex_number(value_at(t, phi, 1.0)),
            tex_number(value_at(t, phi, 50.0)), tex_number(value_at(ta, ang, 1.0)), tex_number(value_at(ta, ang, 50.0)),
            "--" if first_crossing(ta, ang, 1e-3) is None else f"{first_crossing(ta, ang, 1e-3):.0f}",
            tex_number(float(np.max(np.abs(mass))) if len(mass) else None), f"{t[-1]:g}"]) + "\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def seed_rate_row(run, label, *, order):
    params = band_parameters(run)
    mode, length, thickness = params.get("mode"), params.get("band_length"), params.get("thickness")
    t, A = spectra(run["rows"])
    if mode is None or len(t) < 10:  # a run that has just started
        return None
    fit = seeded_growth(t, A, mode) or {}
    if fit.get("growth_rate") is not None and _is_plateau(t, A[:, mode], fit.get("window"), fit["growth_rate"]):
        rate, window, note = fit["growth_rate"], fit["window"], ""
    else:  # no exponential phase: largest local rate while the seeded amplitude is below twice its start
        local = np.gradient(np.log(A[:, mode]), t)
        doubled = first_crossing(t, A[:, mode], 2.0 * float(A[0, mode]))
        before = t <= (t[-1] if doubled is None else doubled)
        k = int(np.argmax(np.where(before, local, -np.inf)))
        rate, window, note = float(local[k]), None, "$^\\ast$"
    return [label, f"{order}", f"$\\ell={mode}$", f"{rate:.3f}{note}",
            "--" if window is None else f"{window[0]:.1f}--{window[1]:.1f}",
            f"{float(vortex_strip_rate(mode, length, thickness)):.3f}"]


def _is_plateau(t, amplitude, window, rate):
    """A fitted rate is reported only when its window covers at least one e-folding of growth."""
    return bool(window) and rate * (window[1] - window[0]) >= 1.0


def control_rate_row(run, label, *, order):
    params = band_parameters(run)
    t, A = spectra(run["rows"])
    if len(t) < 10:  # a run that has just started
        return None
    rates, window = modal_growth_rates(t, A)
    if not rates:
        return None
    mode = max(rates, key=lambda m: rates[m]["growth_rate"])
    return [label, f"{order}", f"$m={mode}$ (fastest)", f"{rates[mode]['growth_rate']:.3f}",
            f"{window[0]:.1f}--{window[1]:.1f}",
            f"{float(vortex_strip_rate(mode, params['band_length'], params['thickness'])):.3f}"]


def rates_table(data, extra, path: Path, *, order):
    lines = ["% Generated by scripts/guiding_center/diagnostics/plot_band_evolution_article.py",
             "\\begin{tabular}{llllll}\\toprule",
             "domain, run & $p$ & mode & $\\gamma_h$ & fit window & vortex strip\\\\\\midrule"]
    for g, pair in data.items():
        for kind in ("seed", "control"):
            run = pair.get(kind)
            if run is None:
                continue
            label = f"{NAMES[g]}, {'perturbed' if kind == 'seed' else 'control'}"
            row = (seed_rate_row if kind == "seed" else control_rate_row)(run, label, order=order)
            if row:
                lines.append(" & ".join(row) + "\\\\")
    for label, run, run_order in extra:
        row = seed_rate_row(run, label, order=run_order)
        if row:
            lines.append(" & ".join(row) + "\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def longest_interval(times, flags):
    """First and last time of the longest stretch of consecutive saved states where ``flags`` holds."""
    best, start = None, None
    for i, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or times[i - 1] - times[start] > times[best[1]] - times[best[0]]:
                best = (start, i - 1)
            start = None
    return None if best is None else (float(times[best[0]]), float(times[best[1]]))


def search_table(entries, path: Path, *, width=900):
    """Perturbation search: one row per perturbed run (seed shape, mode, edge displacement, degree, final time)
    with its coherent-vortex counts N at the saved states: largest N, the longest stretch with N equal to the
    seeded mode, and N at the final time. ``entries``: (domain label, crest label, run, runs dir)."""
    lines = ["% Generated by scripts/guiding_center/diagnostics/plot_band_evolution_article.py",
             "\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lllrrrrrlr}\\toprule",
             "domain & crests at & shape & $\\ell$ & edges/$d$ & $p$ & $T$ & $\\max N$ & $N=\\ell$ for $t\\in$ & $N(T)$"
             "\\\\\\midrule"]
    for label, crests, run, runs in entries:
        params = band_parameters(run)
        sites = params.get("sites")
        mode = len(sites) if sites else params.get("mode")
        shape = "bulges" if sites else ("varicose" if "varicose" in params.get("normal_profile", "") else "sinuous")
        stem = run["jsonl"].with_suffix("")
        order = json.loads(Path(f"{stem}_config.json").read_text())["order"]
        times, counts = cached_coherent_counts(runs, run["jsonl"].stem, params["thickness"], width)
        span = longest_interval(times, counts == mode)
        if span is None:
            stretch = "--"
        elif span[0] == span[1]:
            stretch = f"{span[0]:.0f} only"
        else:
            stretch = f"{span[0]:.0f}--{span[1]:.0f}"
        lines.append(" & ".join([label, crests, shape, str(mode), f"{params['edge_fraction']:g}", str(order),
                                 f"{times[-1]:.0f}", str(int(counts.max())), stretch, str(int(counts[-1]))])
                     + "\\\\")
    lines.append("\\bottomrule\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--article", type=Path, required=True)
    parser.add_argument("--prefix", default="e7")
    parser.add_argument("--geometries", default=",".join(GEOMETRIES))
    parser.add_argument("--spectrum", default="star:control", help="geometry:kind of the heat-map panel")
    parser.add_argument("--panels", default="",
                        help="'g:kind:rho@t,phi@t,...[:stem];...' snapshot figure (default stem e7_<g>_<kind>)")
    parser.add_argument("--extra-seed", action="append", default=[],
                        help="'label|runs dir|prefix|geometry|order': a seeded run of another study, rates table row")
    parser.add_argument("--order", type=int, default=6, help="degree of the --runs runs (rates table)")
    parser.add_argument("--activity", default="",
                        help="'label|runs dir|prefix|geometry|kind;...': curves of the activity figure")
    parser.add_argument("--tracks", type=Path, default=None, help="analysis e7_summary.json: vortex track figure")
    parser.add_argument("--search", default="",
                        help="'domain label|crest label|runs dir|prefix|geometry;...': perturbation search table")
    parser.add_argument("--skip-summary", action="store_true",
                        help="only the --panels figures (no preservation figure or tables)")
    parser.add_argument("--width", type=int, default=900)
    return parser.parse_args(argv)


def panel_requests(text):
    for item in filter(None, text.split(";")):
        g, kind, panels, *stem = item.split(":")
        yield (g, kind, [(field, float(time)) for field, time in (p.split("@") for p in panels.split(","))],
               stem[0] if stem else f"e7_{g}_{kind}")


def main(argv=None):
    args = parse_args(argv)
    article_style()
    figures, tables = args.article / "figures", args.article / "tables"
    data = {}
    for g in args.geometries.split(","):
        pair = {k: find_run(args.runs, args.prefix, g, k) for k in ("control", "seed")}
        if any(pair.values()):
            data[g] = pair
    if not args.skip_summary:
        extra = []
        for item in args.extra_seed:
            label, runs, prefix, g, order = item.split("|")
            run = find_run(Path(runs), prefix, g, "seed")
            if run is not None:
                extra.append((label, run, int(order)))
        article_preservation_figure(data, figures, tuple(args.spectrum.split(":")))
        preservation_table(data, tables / "e7_preservation.tex")
        rates_table(data, extra, tables / "e7_rates.tex", order=args.order)
        conservation_table(data, tables / "e7_conservation.tex")
    entries = []
    for item in filter(None, args.activity.split(";")):
        label, runs, prefix, g, kind = item.split("|")
        run = find_run(Path(runs), prefix, g, kind)
        if run is not None:
            entries.append((label, run, Path(runs)))
    if entries:
        activity_figure(entries, figures)
    if args.tracks is not None:
        tracks_figure(args.tracks, figures)
    search = []
    for item in filter(None, args.search.split(";")):
        label, crests, runs, prefix, g = item.split("|")
        run = find_run(Path(runs), prefix, g, "seed")
        if run is not None:
            search.append((label, crests, run, Path(runs)))
    if search:
        search_table(search, tables / "e7_search.tex", width=args.width)
    for g, kind, panels, stem in panel_requests(args.panels):
        run = (data.get(g) or {}).get(kind)
        if run is not None:
            states = RunStates(args.runs, run["jsonl"].stem)
            snapshot_figure(states, panels, figures, stem, width=args.width)
    print(f"wrote the figures in {figures} and the tables in {tables}")


if __name__ == "__main__":
    main()
