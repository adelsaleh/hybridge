"""Article figures for the central-injection diocotron runs (explicit density source); no time integration.

Reads the summaries of ``run_diocotron_benchmark.py --benchmark disk-injection``
(smooth m = 9 ring, central Gaussian source of charge Q):

``<stem>_early`` (``--early``, injection before the linear phase)
    amplitude of the mode 9 for every Q with e^{gamma(Q) t}, and the plateau
    rate against Q with the Nystrom rates of the ring rotated by a central charge
    (modes 9 to 12).
``<stem>_late`` (``--late``, injection after the roll-up, with saved snapshots)
    density on the disk at the saved times, one row per Q labelled with
    mu = Q/Gamma (Gamma = ring charge / 9), and the symmetry-breaking ratio of the
    potential spectrum (modes not multiple of 9 over multiples of 9) against
    time with the point-vortex growth rates of the ring of nine vortices.
``--table``
    booktabs rows: Q, mu, measured and reference rates (early), point-vortex
    growth rate and outcome (late).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hybridge.io.figures import draw_signed_density, publication_style, save_publication_figure  # noqa: E402
from scripts.guiding_center.diagnostics.plot_diocotron_highmode import _local_rates, _records, _series, load_rows  # noqa: E402

INK, MUTED = "#0b0b0b", "#52514e"
MODE = 9


def injected_charge(row) -> float:
    """Total injected charge of a run (0 without injection)."""
    record = _records(row["jsonl"])[0]["run_configuration"]["case_parameters"].get("injection") or {}
    return float(record.get("total_charge", 0.0))


def ring_charge(row) -> float:
    """Charge of the ring at t = 0 (the first recorded mass)."""
    return float(_records(row["jsonl"])[0]["mass"])


def _colors(count):
    import matplotlib as mpl

    return mpl.colormaps["viridis"](np.linspace(0.0, 0.85, max(count, 2)))


MARKERS = {("si-bdf2", 0.05): dict(marker="o", color="#2a78d6", markerfacecolor="#2a78d6"),
           ("si-bdf2", 0.025): dict(marker="o", color="#2a78d6", markerfacecolor="none"),
           ("imex-ark3", 0.05): dict(marker="D", color="#eb6834", markerfacecolor="#eb6834")}
SCHEME_LABELS = {"si-bdf2": "SI-BDF2", "imex-ark3": "IMEX-ARK3"}


LATE_WINDOW = (60.0, 100.0)


def measured_rate(row, *, r_squared=0.99999):
    """(rate, kind) of the mode 9: modal plateau when clean; else a late window [60, 100]
    (slowly growing modes, whose seed transient decays only by t ~ 60); else None."""
    if row.get("linear_window") and (row.get("linear_r_squared") or 0.) > r_squared:
        return row["linear_gamma"], "plateau"
    from scripts.guiding_center.diagnostics.diocotron_reference import fit_growth

    records = _records(row["jsonl"])
    times = _series(records, "time")
    if times[-1] >= LATE_WINDOW[1]:
        late = fit_growth(times, _series(records, f"diocotron_phi_mode_{MODE}_l2"), window=LATE_WINDOW,
                          reference=row["reference_gamma"] or None)
        if (late.get("r_squared") or 0.) > r_squared and (late.get("growth_rate") or 0.) > 0:
            return late["growth_rate"], "late"
    return None, None


def early_figure(rows, stem, *, curve_charges=np.linspace(0.0, 0.3, 31), curve_points=48):
    from scripts.guiding_center.diagnostics.diocotron_reference import RING_EDGES, smooth_annulus_spectrum

    rows = sorted(rows, key=injected_charge)
    primary = [row for row in rows if (row["scheme"], row["dt"]) == ("si-bdf2", 0.05)] or rows
    parameters = _records(rows[0]["jsonl"])[0]["run_configuration"]["case_parameters"]
    with publication_style():
        figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
        for color, row in zip(_colors(len(primary)), primary):
            records = _records(row["jsonl"])
            t, amplitude = _series(records, "time"), _series(records, f"diocotron_phi_mode_{MODE}_l2")
            left.semilogy(t, amplitude, color=color, lw=1.2, label=f"$Q={injected_charge(row):g}$")
            window, gamma = row.get("linear_window"), row["reference_gamma"]
            if measured_rate(row)[1] == "plateau" and gamma > 0:
                middle = .5 * (window[0] + window[1])
                level = np.interp(middle, t, np.log(amplitude))
                x = np.linspace(window[0] - 8, window[1] + 8, 50)
                left.semilogy(x, np.exp(level + gamma * (x - middle)), color=INK, lw=.8, linestyle=(0, (4, 2)))
        left.set(xlabel="$t$", ylabel=rf"$\|\hat\phi_{MODE}\|$", xlim=(0, max(_series(_records(r['jsonl']), 'time')[-1]
                                                                             for r in primary)))
        left.legend(frameon=False, fontsize=6.5, ncol=2)
        left.grid(True, which="major", alpha=.5)
        styles = {9: dict(color=INK, lw=1.3), 10: dict(color=MUTED, lw=.8, linestyle=(0, (4, 2))),
                  11: dict(color=MUTED, lw=.8, linestyle=(0, (1, 1.5))), 12: dict(color=MUTED, lw=.8, linestyle="-.")}
        # One Nystrom resolution for the curves (about 1e-4 relative); the table has the converged rates.
        curves = np.array([[entry["growth_rate"] for entry in smooth_annulus_spectrum(
            list(styles), parameters=parameters, radial_points=curve_points, ring_edges=RING_EDGES,
            central_charge=float(q))] for q in curve_charges])
        for column, (mode, style) in enumerate(styles.items()):
            right.plot(curve_charges, curves[:, column], label=f"$m={mode}$, eigenvalue problem", **style)
        for (scheme, dt), style in MARKERS.items():
            points = [(injected_charge(row), *measured_rate(row)) for row in rows
                      if (row["scheme"], row["dt"]) == (scheme, dt)]
            for kind, fill in (("plateau", style["markerfacecolor"]), ("late", style["markerfacecolor"])):
                chosen = [(q, g) for q, g, k in points if k == kind]
                if chosen:
                    q, g = np.array(chosen).T
                    right.plot(q, g, linestyle="none", markersize=5, **dict(style, markerfacecolor=fill),
                               label=None if kind == "late" and any(k == "plateau" for _, _, k in points)
                               else f"{SCHEME_LABELS[scheme]}, $\\Delta t={dt:g}$")
        right.set(xlabel="injected central charge $Q$", ylabel=r"growth rate $\gamma$", ylim=(0, None))
        right.legend(frameon=False, fontsize=6, loc="lower left")
        right.grid(True, which="major", alpha=.5)
        return save_publication_figure(figure, stem)


def _symmetry_breaking(records, mode=MODE, top=63):
    """sqrt(sum_{k not multiple of m} |phi_k|^2 / sum_{k multiple of m} |phi_k|^2) per record."""
    keys = [k for k in range(1, top + 1) if f"diocotron_phi_mode_{k}_l2" in records[0]]
    on = np.array([[r.get(f"diocotron_phi_mode_{k}_l2") or 0.0 for k in keys if k % mode == 0] for r in records])
    off = np.array([[r.get(f"diocotron_phi_mode_{k}_l2") or 0.0 for k in keys if k % mode] for r in records])
    return np.sqrt((off ** 2).sum(axis=1) / np.maximum((on ** 2).sum(axis=1), 1e-300))


def late_figure(rows, stem, *, times=(70., 100., 200., 300., 400.), pixels=500, radius=.475):
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from hybridge.io.raster import RasterGeometry
    from scripts.guiding_center.diagnostics.diocotron_reference import point_vortex_ring_growth

    rows = sorted(rows, key=injected_charge)
    circulation = ring_charge(rows[0]) / MODE
    panels = []
    with publication_style():
        figure = plt.figure(figsize=(7.2, 1.42 * len(rows) + 2.3), layout="constrained")
        grid = figure.add_gridspec(len(rows) + 1, len(times), height_ratios=[1.0] * len(rows) + [1.45])
        for i, row in enumerate(rows):
            npz_path = Path(row["jsonl"]).with_name(f"{row['prefix']}_snapshots.npz")
            charge = injected_charge(row)
            mu = charge / circulation
            if not npz_path.exists():
                continue
            npz = np.load(npz_path)
            saved = [round(float(t), 6) for t in npz["times"]]
            density = npz["density"]
            mesh = DGMesh(npz["node_coords"], npz["triangles"])
            space = DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]))
            geometry = RasterGeometry.from_mesh(mesh, pixels, pixels)
            matrix = geometry.sampling_matrix(space)
            mask = (geometry.element_ids < 0).reshape(geometry.height, geometry.width)
            for j, t in enumerate(times):
                ax = figure.add_subplot(grid[i, j])
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if round(t, 6) in saved:
                    values = (matrix @ np.asarray(density[saved.index(round(t, 6))]).ravel()).reshape(
                        geometry.height, geometry.width)
                    panels.append((ax, np.ma.array(values, mask=mask), geometry.bounds))
                if i == 0:
                    ax.set_title(f"$t={t:g}$", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"$Q={charge:g}$\n$\\mu={mu:.1f}$", fontsize=7.5)
        # The central vortex exceeds the ring's peak: saturated above 1.
        artist = draw_signed_density(panels, 1.0, over="#fde725")
        if artist is not None:
            figure.colorbar(artist, ax=figure.axes, location="right", shrink=.6, extend="max",
                            label=r"density $\rho_h$")
        ax = figure.add_subplot(grid[len(rows), :])
        for color, row in zip(_colors(len(rows)), rows):
            records = _records(row["jsonl"])
            t, ratio = _series(records, "time"), _symmetry_breaking(records)
            mu = injected_charge(row) / circulation
            ax.semilogy(t, ratio, color=color, lw=1.1, label=f"$\\mu={mu:.1f}$")
            rate = point_vortex_ring_growth(MODE, mu, radius=radius, circulation=circulation)
            if rate > 0:
                anchor = 100.
                level = np.interp(anchor, t, np.log(ratio))
                x = np.linspace(anchor, min(t[-1], anchor + 8 / rate), 40)
                ax.semilogy(x, np.exp(level + rate * (x - anchor)), color=color, lw=.8, linestyle=(0, (4, 2)))
        ax.axvspan(70, 75, color="#eceae4", lw=0, zorder=0)
        ax.set(xlabel="$t$", ylabel="modes $k\\notin 9\\mathbb{N}$ / $k\\in9\\mathbb{N}$")
        ax.legend(frameon=False, fontsize=6.5, ncol=len(rows), loc="lower right")
        ax.grid(True, which="major", alpha=.5)
        return save_publication_figure(figure, stem, dpi=300)


def latex_table(early, late) -> str:
    from scripts.guiding_center.diagnostics.diocotron_reference import point_vortex_ring_growth, point_vortex_ring_window

    lines = [r"\begin{tabular}{rrrrrl}\toprule",
             r"$Q$ & $\mu=Q/\Gamma$ & $\gamma_h$ & $\gamma(Q)$ & $\gamma_h/\gamma-1$ & \\\midrule",
             r"\multicolumn{6}{l}{\emph{injection on $[0,4]$, before the linear phase}}\\"]
    for row in sorted(early, key=lambda r: (injected_charge(r), r["scheme"], -r["dt"])):
        charge = injected_charge(row)
        circulation = ring_charge(row) / MODE
        fitted, kind = measured_rate(row)
        error = None if fitted is None or not row["reference_gamma"] else fitted / row["reference_gamma"] - 1
        lines.append(f"{charge:g} & -- & "  # no vortices yet: mu only means something after the roll-up
                     + ("--" if fitted is None else f"{fitted:.5f}") + f" & {row['reference_gamma']:.5f} & "
                     + ("--" if error is None else f"${100 * error:+.2f}\\%$")
                     + f" & {SCHEME_LABELS[row['scheme']]}, $\\Delta t={row['dt']:g}$"
                     + ("" if kind != "late" else ", window $[60,100]$") + r"\\")
    low, high = point_vortex_ring_window(MODE)
    lines.append(r"\midrule\multicolumn{6}{l}{\emph{injection on $[70,75]$: break-up rate, measured and for point vortices}}\\")
    for row in sorted(late, key=injected_charge):
        charge = injected_charge(row)
        circulation = ring_charge(row) / MODE
        mu = charge / circulation
        rate = point_vortex_ring_growth(MODE, mu, radius=.475, circulation=circulation)
        records = _records(row["jsonl"])
        times, ratio = _series(records, "time"), _symmetry_breaking(records)
        late_part = times >= times[-1] - 100.  # break-up rate over the last 100 time units
        measured = float(np.polyfit(times[late_part], np.log(ratio[late_part]), 1)[0])
        lines.append(f"{charge:g} & {mu:.1f} & {measured:+.3f} & "
                     + ("0 (stable)" if rate < 1e-6 else f"{rate:.3f}") + r" & & \\")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines) + "\n"


def movie_density_grid(movies, stem, *, times=(4., 30., 50., 60., 80., 100., 150.), interval=.5, label=None):
    """Density panels of campaign movies at several times: one row per movie (no new data).

    Frame k of a movie is at t = interval*k (frame 0 at the first step); the density
    panel is located by rebuilding the movie layout (``_movie_panel_boxes``). Colours
    are the movie's (values above the ring's peak, the injected central vortex,
    saturated).
    """
    import imageio_ffmpeg

    from scripts.guiding_center.diagnostics.plot_diocotron_highmode import _movie_panel_boxes

    rows = []
    for movie in movies:
        wanted = {int(round(t / interval)): t for t in times}
        reader = imageio_ffmpeg.read_frames(str(movie))
        meta = next(reader)
        width, height = meta["size"]
        frames = {}
        for index, frame in enumerate(reader):
            if index in wanted:
                frames[index] = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
            if len(frames) == len(wanted):
                break
        reader.close()
        top, bottom, left, right = _movie_panel_boxes(size=(width, height),
                                                      title="caption with the movie's length " + "x" * 90)[0]
        inset = 8
        rows.append((movie, {wanted[k]: f[top + inset:bottom - inset, left + inset:right - inset]
                             for k, f in frames.items()}))
    with publication_style():
        figure, axes = plt.subplots(len(rows), len(times), figsize=(1.05 * len(times) + .5, 1.08 * len(rows) + .3),
                                    layout="constrained", squeeze=False)
        for i, (movie, frames) in enumerate(rows):
            for j, t in enumerate(times):
                ax = axes[i, j]
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if t in frames:
                    ax.imshow(frames[t], interpolation="antialiased")
                if i == 0:
                    ax.set_title(f"$t={t:g}$", fontsize=8)
                if j == 0:
                    ax.set_ylabel(label(movie) if label else Path(movie).stem[:20], fontsize=7.5)
        return save_publication_figure(figure, stem, dpi=250)


def _movie_charge(movie) -> str:
    """'Q=0.15' from a campaign movie name (..._inject_q0p15_t0_...)."""
    for part in Path(movie).stem.split("_"):
        if part.startswith("q") and part[1:2].isdigit():
            return f"$Q={part[1:].replace('p', '.')}$"
    return Path(movie).stem[:20]


def positivity_comparison_figure(runs, stem, *, times=(75., 80., 100., 150., 200., 300., 400.), pixels=420):
    """Same run with and without the KKT positivity projection.

    ``runs`` is a list of (label, summary row). Top: density from the saved
    snapshots (one row per run). Bottom: sampled minimum of rho_h ((2p+2)
    lattice), relative energy drift, symmetry-breaking ratio, and the fraction of
    elements the projection corrects with its relative correction.
    """

    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from hybridge.io.raster import RasterGeometry

    colors = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a"]
    panels = []
    with publication_style():
        figure = plt.figure(figsize=(7.2, 1.25 * len(runs) + 2.4), layout="constrained")
        grid = figure.add_gridspec(len(runs) + 1, len(times), height_ratios=[1.0] * len(runs) + [1.6])
        geometry = matrix = mask = None
        for i, (label, row) in enumerate(runs):
            npz = np.load(Path(row["jsonl"]).with_name(f"{row['prefix']}_snapshots.npz"))
            saved = [round(float(t), 6) for t in npz["times"]]
            if geometry is None:
                mesh = DGMesh(npz["node_coords"], npz["triangles"])
                space = DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]))
                geometry = RasterGeometry.from_mesh(mesh, pixels, pixels)
                matrix = geometry.sampling_matrix(space)
                mask = (geometry.element_ids < 0).reshape(geometry.height, geometry.width)
            density = npz["density"]
            for j, t in enumerate(times):
                ax = figure.add_subplot(grid[i, j])
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if round(t, 6) in saved:
                    values = (matrix @ np.asarray(density[saved.index(round(t, 6))]).ravel()).reshape(
                        geometry.height, geometry.width)
                    panels.append((ax, np.ma.array(values, mask=mask), geometry.bounds))
                if i == 0:
                    ax.set_title(f"$t={t:g}$", fontsize=8)
                if j == 0:
                    ax.set_ylabel(label, fontsize=7.5)
        draw_signed_density(panels, 1.0, over="#fde725")
        bottom = grid[len(runs), :].subgridspec(1, 4, wspace=.35)
        axes = [figure.add_subplot(bottom[0, k]) for k in range(4)]
        for color, (label, row) in zip(colors, runs):
            records = _records(row["jsonl"])
            t = _series(records, "time")
            minimum = _series(records, "rho_min_checked")
            axes[0].plot(t, minimum, color=color, lw=1., label=label)
            axes[1].semilogy(t[1:], np.maximum(np.abs(_series(records, "energy_relative_drift"))[1:], 1e-16),
                             color=color, lw=1.)
            axes[2].semilogy(t, _symmetry_breaking(records), color=color, lw=1.)
            timings = _records(Path(row["jsonl"]).with_name(f"{row['prefix']}_timings.jsonl"))
            flagged = np.array([r.get("positivity_flagged", np.nan) for r in timings], dtype=float)
            if np.isfinite(flagged).any():
                tt = np.array([r.get("time", np.nan) for r in timings], dtype=float)
                elements = int(row.get("triangles") or 1)
                axes[3].plot(tt, flagged / elements, color=color, lw=1.)
        axes[0].set(xlabel="$t$", ylabel=r"$\min\rho_h$ (lattice)")
        axes[1].set(xlabel="$t$", ylabel=r"$|\delta W|$")
        axes[2].set(xlabel="$t$", ylabel="symmetry breaking")
        axes[3].set(xlabel="$t$", ylabel="corrected elements (fraction)")
        for ax in axes:
            ax.grid(True, which="major", alpha=.5)
        axes[0].legend(frameon=False, fontsize=6)
        return save_publication_figure(figure, stem, dpi=250)



def rollup_edges(npz_path, *, threshold=3e-2, radii=(.30, .65), angles=1440, radial_points=141, final_time=80.):
    """Lab-frame edges of the rolled-up arc of a ring disturbed at one angle, from saved snapshots.

    At each saved time the charge centroid R(theta) = int r^2 rho dr / int r rho dr over
    ``radii`` is compared with that of the first saved step; the disturbed arc is the
    complement of the largest arc where |R - R_0| <= ``threshold``. Returns times,
    trailing and leading angles (radians, unwrapped from the source at theta = 0) and
    the deviation profiles.
    """
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace

    npz = np.load(npz_path)
    space = DGSpace(DGMesh(npz["node_coords"], npz["triangles"]), int(npz["order"]), basis_type=str(npz["basis"]))
    theta = np.arange(angles) * 2 * np.pi / angles
    r = np.linspace(*radii, radial_points)
    x = (r[None, :] * np.cos(theta)[:, None]).ravel()
    y = (r[None, :] * np.sin(theta)[:, None]).ravel()

    def centroid(k):
        rho = np.nan_to_num(space.field(npz["density"][k], name="rho").evaluate(x, y).reshape(angles, radial_points))
        weight = np.clip(rho, 0., None)
        return np.trapezoid(weight * r ** 2, r, axis=1) / np.trapezoid(weight * r, r, axis=1)

    reference = centroid(0)
    times, trailing, leading, profiles = [], [], [], []
    for k, t in enumerate(npz["times"]):
        if k == 0 or t > final_time:
            continue
        deviation = np.abs(centroid(k) - reference)
        profiles.append(deviation)
        disturbed = np.flatnonzero(deviation > threshold)
        if disturbed.size == 0:
            continue
        gaps = np.diff(np.r_[disturbed, disturbed[0] + angles])
        g = int(np.argmax(gaps))
        lead, trail = theta[disturbed[g]], theta[disturbed[(g + 1) % disturbed.size]]
        times.append(float(t))
        trailing.append(trail if trail < np.pi else trail - 2 * np.pi)  # just behind the source
        leading.append(lead)
    return dict(theta=theta, times=np.array(times), trailing=np.array(trailing), leading=np.array(leading),
                profile_times=np.array([t for t in npz["times"][1:] if t <= final_time]), profiles=np.array(profiles))



def _ring_rotation(parameters, radius=.475):
    """E x B angular velocity Omega = M(r)/r^2 of the unperturbed ring at ``radius``."""
    from scipy.integrate import quad

    from scripts.guiding_center.cases.guiding_center_cases import diocotron_k

    case = diocotron_k(**parameters)
    return quad(lambda s: s * float(case.equilibrium_density(s, 0)), 0, radius, points=[.4, .45])[0] / radius ** 2

def front_figure(edges, speeds, stem, *, rotation=None):
    """Deviation of the ring's charge centroid (left) and edges of the rolled-up arc against pulled-front speeds."""
    with publication_style():
        figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
        colors = _colors(len(edges["profile_times"]))
        for color, t, profile in zip(colors, edges["profile_times"], edges["profiles"]):
            left.semilogy(np.degrees(edges["theta"]), profile, color=color, lw=.8, label=f"$t={t:g}$")
        left.axhline(edges.get("threshold", 3e-2), color=MUTED, lw=.7, ls=(0, (3, 2)))
        left.set(xlabel=r"$\theta$ (degrees, lab frame)", ylabel=r"$|R(\theta,t)-R(\theta,0)|$", ylim=(1e-5, .5),
                 xlim=(0, 360), xticks=range(0, 361, 90))
        left.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
        t = edges["times"]
        for name, marker in (("leading", "o"), ("trailing", "s")):
            angles = np.degrees(edges[name])
            speed = np.degrees(speeds[name]["speed"])
            offset = np.mean(angles - speed * t)
            fit = np.polyfit(t, angles, 1)[0]
            right.plot(t, angles, marker, ms=4, mfc="none", color=INK, label=f"{name} edge: {np.radians(fit):.3f}")
            line = np.linspace(t[0], t[-1], 2)
            right.plot(line, offset + speed * line, color=INK, lw=.9,
                       label=f"pulled front: {speeds[name]['speed']:.3f}")
        if rotation is not None:
            line = np.linspace(t[0], t[-1], 2)
            right.plot(line, np.degrees(edges["leading"][0]) + np.degrees(rotation) * (line - t[0]), color=MUTED,
                       lw=.8, ls=(0, (3, 2)), label=f"ring rotation: {rotation:.3f}")
        right.set(xlabel="$t$", ylabel=r"edge angle (degrees)")
        right.legend(frameon=False, fontsize=6.5, title="rad per time", title_fontsize=6.5)
        right.grid(True, which="major", alpha=.5)
        return save_publication_figure(figure, stem, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--early", type=Path, nargs="*", default=(),
                        help="summaries of the early-injection campaign and its time-step checks")
    parser.add_argument("--late", type=Path, help="summary of the late-injection campaign (with snapshots)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="diocotron_injection")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--movies", type=Path, nargs="*", default=(),
                        help="campaign movies for a density frame grid (one row per movie)")
    parser.add_argument("--movie-times", type=lambda text: tuple(float(v) for v in text.split(",")),
                        default=(4., 30., 50., 60., 80., 100., 150.))
    parser.add_argument("--movie-stem", default="movie_frames")
    parser.add_argument("--front", type=Path, help="snapshots of a fixed-injector seed run (rolled-up arc edges)")
    parser.add_argument("--front-threshold", type=float, default=3e-2)
    args = parser.parse_args()
    # Runs with the KKT positivity projection belong to the positivity comparison, not to these figures.
    early = [row for row in (load_rows(list(args.early)) if args.early else []) if "_kkt" not in row["prefix"]]
    late = [row for row in (load_rows([args.late]) if args.late else []) if "_kkt" not in row["prefix"]]
    if early:
        print(early_figure(early, args.output_dir / f"{args.stem}_early"))
    if late:
        print(late_figure(late, args.output_dir / f"{args.stem}_late"))
    if args.movies:
        movies = sorted(args.movies, key=lambda m: float(_movie_charge(m)[3:-1]) if _movie_charge(m).startswith("$Q") else 0.)
        print(movie_density_grid(movies, args.output_dir / f"{args.stem}_{args.movie_stem}", times=args.movie_times,
                                 label=_movie_charge))
    if args.front is not None:
        from scripts.guiding_center.diagnostics.diocotron_reference import RingDispersion, pulled_front_speeds

        parameters = dict(k=MODE, s_minus=.45, s_plus=.5, p=4.0, truncate=False, epsilon=0.)
        speeds = pulled_front_speeds(RingDispersion(parameters, radial_points=16))
        edges = rollup_edges(args.front, threshold=args.front_threshold)
        edges["threshold"] = args.front_threshold
        for name in ("leading", "trailing"):
            fit = np.polyfit(edges["times"], edges[name], 1)[0]
            print(f"{name} edge: measured {fit:.4f} rad/time, pulled front {speeds[name]['speed']:.4f}")
        print(front_figure(edges, speeds, args.output_dir / f"{args.stem}_front", rotation=_ring_rotation(parameters)))
    if args.table is not None and (early or late):
        args.table.write_text("% Generated by scripts/guiding_center/diagnostics/plot_diocotron_injection.py\n"
                              + latex_table(early, late))
        print(args.table)


if __name__ == "__main__":
    main()
