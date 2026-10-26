"""E7 analysis: preservation, band-mode growth and vortex census of control/seed pairs (no solve).

Reads, per geometry, ``<prefix>_<geometry>_*_{control,seed}.jsonl`` (per-step diagnostics with the
band-mode spectra ``band_{rho,phi}_mode_amplitudes``), ``*_summary.json``, ``*_seed.json`` and
``*_snapshots.npz`` from ``--runs`` and writes into ``--output``:

- ``e7_summary.json`` / ``e7_summary.md``: transfer metrics, control drift at fixed times, conservation,
  seeded-mode growth rate and window, per-mode growth rates of the control (noise-seeded linear phase),
  active-mode counts, vortex counts;
- figures (PDF + PNG): preservation and conservation, spectra heat maps, mode growth and active modes,
  measured growth rate against mode number with the vortex-strip estimate, snapshot grids of the
  seeded runs and of the controls, vortex positions along the band against curvature and wall clearance.

Growth rates are amplitude rates of A_m = 2|int (rho_h - rho_eq) g e^{-i m theta}| / int rho_eq g.
The vortex-strip estimate is Rayleigh's sigma(k) = (omega/2) sqrt(e^{-2kd} - (1 - kd)^2) for a strip
of vorticity omega = 1 (the band density) and thickness d, with k = m 2 pi / L; it ignores the wall,
the curvature and the finite smoothing, so the comparison measures their combined effect.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GEOMETRIES = ("star", "uglypacman", "iter", "horseshoe")
NAMES = {"star": "Star", "uglypacman": "Ugly Pac-Man", "iter": "ITER", "horseshoe": "Horseshoe"}
# Reference categorical palette, slots 1-4 in fixed order (dataviz skill, validated light-mode steps).
COLORS = {"star": "#2a78d6", "uglypacman": "#eb6834", "iter": "#1baf7a", "horseshoe": "#eda100"}
SEED, CONTROL = "#2a78d6", "#52514e"
INK, MUTED = "#0b0b0b", "#52514e"
DRIFT_TIMES = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0)
COHERENT_FRACTION = 0.02  # a coherent vortex holds at least this fraction of the initial band area (rho_h > 0.5)
COHERENT_WIDTH = 0.4  # ... and contains a disk of radius COHERENT_WIDTH * band thickness


# ---------------------------------------------------------------------------------------------- data
def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row.get("time"), (int, float)):
                    rows.append(row)
    return rows


def series(rows, key):
    pairs = [(r["time"], r[key]) for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    if not pairs:
        return np.array([]), np.array([])
    t, v = map(np.asarray, zip(*pairs))
    return t, v


def spectra(rows, field="rho"):
    key = f"band_{field}_mode_amplitudes"
    good = [r for r in rows if isinstance(r.get(key), list)]
    return np.array([r["time"] for r in good]), np.array([r[key] for r in good], dtype=float)


def find_run(runs: Path, prefix: str, geometry: str, kind: str):
    matches = sorted(runs.glob(f"{prefix}_{geometry}_*_{kind}.jsonl"))
    if not matches:
        return None
    stem = matches[0].with_suffix("")
    files = {name: Path(f"{stem}{suffix}") for name, suffix in (
        ("jsonl", ".jsonl"), ("summary", "_summary.json"), ("seed", "_seed.json"), ("snapshots", "_snapshots.npz"))}
    files["rows"] = read_rows(files["jsonl"])
    files["summary_data"] = json.loads(files["summary"].read_text()) if files["summary"].exists() else None
    files["seed_data"] = json.loads(files["seed"].read_text()) if files["seed"].exists() else None
    return files


def band_parameters(run):
    seed = run.get("seed_data") or {}
    return (seed.get("case_parameters") or {}).get("band") or {}


# ------------------------------------------------------------------------------------------ analysis
def value_at(t, v, time):
    if not len(t) or time < t[0] - 1e-9 or time > t[-1] + 1e-9:
        return None
    return float(np.interp(time, t, v))


def angular_departure(rows, field="rho"):
    """max over m >= 1 of A_m(t): the part of the drift that depends on the band angle. The angle average
    (edge relaxation of the projected profile) is left out, so this is what the instability grows from."""
    t, A = spectra(rows, field)
    return (t, A[:, 1:].max(axis=1)) if len(t) else (t, np.array([]))


def first_crossing(t, v, level):
    above = np.flatnonzero(v >= level)
    return None if not len(above) else float(t[above[0]])


def fit_log(t, a, start, end):
    mask = (t >= start) & (t <= end) & (a > 0) & np.isfinite(a)
    if mask.sum() < 8:
        return None
    x, y = t[mask], np.log(a[mask])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = ((y - y.mean()) ** 2).sum()
    return dict(growth_rate=float(slope), log_intercept=float(intercept), window=[float(x[0]), float(x[-1])],
                samples=int(mask.sum()), r_squared=float(1 - (residual @ residual) / total) if total else None)


def seeded_growth(t, A, mode, *, start=2.0, lower=3.0, upper=0.1, local=4.0, tolerance=0.1):
    """Growth of the seeded mode on its exponential phase.

    Candidate interval: after ``start``, from where A_mode exceeds ``lower`` times its initial value (or
    from ``start`` if it grows at once) to where it reaches ``upper`` times its maximum. Within it the
    window is the longest run of local rates (log-slope over ``local`` time units) within ``tolerance``
    (relative) of their median.
    """
    a = A[:, mode]
    if len(t) < 16 or not np.all(np.isfinite(a)):
        return None
    peak = float(np.max(a))
    end_index = np.flatnonzero((a >= upper * peak) & (t > start))
    if not len(end_index):
        return None
    stop = end_index[0]
    begin = np.flatnonzero((t >= start) & (np.arange(len(t)) < stop))
    if len(begin) < 16:
        return None
    first = begin[0]
    grown = np.flatnonzero((a[first:stop] >= lower * a[first]))
    candidate = slice(first, stop + 1)
    tt, aa = t[candidate], a[candidate]
    spacing = float(np.median(np.diff(tt)))
    width = max(2, int(round(local / spacing)))
    if len(tt) <= width + 8:
        return fit_log(tt, aa, tt[0], tt[-1])
    rates = (np.log(aa[width:]) - np.log(aa[:-width])) / (tt[width:] - tt[:-width])
    level = np.median(rates)
    good = np.abs(rates - level) <= tolerance * abs(level)
    best, run_start = (0, 0), None
    for index, flag in enumerate(np.append(good, False)):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            if index - run_start > best[1] - best[0]:
                best = (run_start, index)
            run_start = None
    if best[1] - best[0] < 4:
        return fit_log(tt, aa, tt[0], tt[-1])
    out = fit_log(tt, aa, tt[best[0]], tt[min(best[1] - 1 + width, len(tt) - 1)])
    if out is not None:
        out["grew_from_start"] = bool(len(grown))
    return out


def modal_growth_rates(t, A, *, start=3.0, saturation=1e-2, max_mode=None):
    """Per-mode amplitude growth rates over the common linear window of a noise-seeded run.

    Window: [start, t_sat], t_sat the first time the largest mode (m >= 1) exceeds ``saturation``.
    """
    top = A[:, 1:].max(axis=1)
    t_sat = first_crossing(t, top, saturation) or float(t[-1])
    end = max(t_sat, start + 5.0)
    out = {}
    for m in range(1, (max_mode or A.shape[1] - 1) + 1):
        fit = fit_log(t, A[:, m], start, end)
        if fit is not None:
            out[m] = fit
    return out, [float(start), float(end)]


def vortex_strip_rate(m, length, thickness, vorticity=1.0):
    kd = 2 * np.pi * np.asarray(m, float) / length * thickness
    value = np.exp(-2 * kd) - (1 - kd) ** 2
    return 0.5 * vorticity * np.sqrt(np.maximum(value, 0.0))


def rasterize(npz, indices, width):
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from hybridge.io.raster import RasterGeometry
    mesh = DGMesh(npz["node_coords"], npz["triangles"])
    space = DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]))
    lo, hi = mesh.node_coords.min(axis=0), mesh.node_coords.max(axis=0)
    height = max(16, int(round(width * (hi[1] - lo[1]) / (hi[0] - lo[0]))))
    geometry = RasterGeometry.from_mesh(mesh, width, height)
    matrix = geometry.sampling_matrix(space, max_bytes=2 * 1024**3)
    inside = geometry.element_ids >= 0
    images = []
    for index in indices:
        values = np.full(width * height, np.nan)
        values[inside] = (matrix @ np.asarray(npz["density"][index]).ravel())[inside]
        images.append(values.reshape(height, width))
    extent = tuple(float(b) for b in geometry.bounds)  # (x0, x1, y0, y1) of the pixel grid
    return images, extent, mesh


def boundary_segments(mesh):
    tri = mesh.triangles
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return mesh.node_coords[unique[counts == 1]]


def vortex_census(images, extent, *, threshold=0.5, minimum_pixels=12, core_radius=None):
    """Connected components of {rho_h > threshold} per image: count, areas, centroids and inscribed radii
    (largest distance from a component point to the background: filaments are thin, vortices are not).
    With ``core_radius``, also the cores: components of the points at least ``core_radius`` from the
    background (erosion), each with its area, centroid and the area of the component holding it; vortices
    still joined by filaments have separate cores."""
    from scipy import ndimage
    out = []
    for image in images:
        height, width = image.shape
        mask = np.nan_to_num(image, nan=0.0) > threshold
        labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
        sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
        keep = np.flatnonzero(sizes >= minimum_pixels) + 1
        centres = ndimage.center_of_mass(mask, labels, keep) if len(keep) else []
        dx, dy = (extent[1] - extent[0]) / width, (extent[3] - extent[2]) / height
        inscribed = (ndimage.maximum(ndimage.distance_transform_edt(mask, sampling=(dy, dx)), labels, keep)
                     if len(keep) else [])
        xy = [(extent[0] + (c + 0.5) * dx, extent[3] - (r + 0.5) * dy) for r, c in centres]
        entry = dict(count=int(len(keep)), areas=[float(sizes[k - 1] * dx * dy) for k in keep], centroids=xy,
                     inscribed=[float(r) for r in np.atleast_1d(inscribed)])
        if core_radius is not None:
            core = ndimage.distance_transform_edt(mask, sampling=(dy, dx)) >= core_radius
            core_labels, core_count = ndimage.label(core, structure=np.ones((3, 3)))
            index = np.arange(1, core_count + 1)
            core_sizes = ndimage.sum(core, core_labels, index) if core_count else []
            parents = ndimage.maximum(labels, core_labels, index) if core_count else []
            core_centres = ndimage.center_of_mass(core, core_labels, index) if core_count else []
            entry["cores"] = [dict(area=float(n * dx * dy), parent_area=float(sizes[int(k) - 1] * dx * dy),
                                   centroid=(extent[0] + (c + 0.5) * dx, extent[3] - (r + 0.5) * dy))
                              for n, k, (r, c) in zip(core_sizes, parents, core_centres)]
        out.append(entry)
    return out


def coherent(census, band_area, thickness):
    """Coherent vortices of one census entry (computed with ``core_radius = COHERENT_WIDTH * thickness``): cores
    whose component holds at least COHERENT_FRACTION of the initial band area and which are at least as large as
    a disk of radius COHERENT_WIDTH * thickness / 2 (no specks where a filament thickens). Returns their centroids."""
    minimum = np.pi * (0.5 * COHERENT_WIDTH * thickness) ** 2
    return [c["centroid"] for c in census["cores"]
            if c["parent_area"] >= COHERENT_FRACTION * band_area and c["area"] >= minimum]


# ------------------------------------------------------------------------------------------- figures
def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
                         "legend.frameon": False})


def save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def preservation_figure(data, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), constrained_layout=True)
    for g, d in data.items():
        for kind, ls in (("control", "-"), ("seed", "--")):
            run = d.get(kind)
            if run is None:
                continue
            for ax, key in zip(axes[:2], ("diocotron_rho_eq_relative_l2", "diocotron_phi_eq_relative_l2")):
                t, v = series(run["rows"], key)
                ax.semilogy(t[v > 0], v[v > 0], ls, color=COLORS[g], lw=1.5 if kind == "control" else 1.0,
                            label=f"{NAMES[g]} {kind}")
            t, v = angular_departure(run["rows"])
            axes[2].semilogy(t[v > 0], v[v > 0], ls, color=COLORS[g], lw=1.5 if kind == "control" else 1.0)
            t, v = series(run["rows"], "mass_relative_drift")
            axes[3].semilogy(t, np.abs(v) + 1e-18, ls, color=COLORS[g], lw=1.0)
    axes[0].set_title(r"$\|\rho_h-\rho_h^{\rm eq}\|/\|\rho_h^{\rm eq}\|$", fontsize=9)
    axes[1].set_title(r"$\|\phi_h-\phi_h^{\rm eq}\|/\|\phi_h^{\rm eq}\|$", fontsize=9)
    axes[2].set_title(r"angular departure $\max_{m\geq1} A_m(\rho_h)$", fontsize=9)
    axes[3].set_title("relative mass drift", fontsize=9)
    axes[2].set_ylim(bottom=1e-10)  # A_m(0) of a control is exactly zero
    for ax in axes:
        ax.set_xlabel("t")
    axes[0].legend(fontsize=7, ncol=2)
    save(fig, out / "e7_preservation")


def spectrum_figure(g, d, out, mode):
    runs = [(k, d[k]) for k in ("control", "seed") if d.get(k) is not None]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.6), constrained_layout=True, squeeze=False)
    image = None
    for ax, (kind, run) in zip(axes[0], runs):
        t, A = spectra(run["rows"])
        if not len(t):
            continue
        logA = np.log10(np.maximum(A[:, 1:], 1e-14)).T
        image = ax.imshow(logA, origin="lower", aspect="auto", cmap="Blues", vmin=-9, vmax=0,
                          extent=(t[0], t[-1], 0.5, A.shape[1] - 0.5), interpolation="nearest")
        if mode:
            for m, lab in ((mode, r"$\ell$"), (2 * mode, r"$2\ell$")):
                if m < A.shape[1]:
                    ax.axhline(m, color="#e34948", lw=0.6, ls=":")
                    ax.text(t[-1], m, " " + lab, va="center", fontsize=7, color=INK)
        ax.set_title(f"{NAMES[g]} {kind}: " + r"$\log_{10}A_m(t)$", fontsize=9)
        ax.set_xlabel("t")
        ax.set_ylabel("band mode m")
        ax.grid(False)
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), shrink=0.9, label=r"$\log_{10}A_m$")
    save(fig, out / f"e7_{g}_spectrum")


def modes_figure(data, results, out):
    present = [g for g in GEOMETRIES if g in data]
    fig, axes = plt.subplots(3, len(present), figsize=(3.4 * len(present), 8.2), constrained_layout=True, squeeze=False)
    for col, g in enumerate(present):
        d, r = data[g], results[g]
        mode = r.get("mode")
        ax, ax_count, ax_rate = axes[0, col], axes[1, col], axes[2, col]
        for kind, color in (("seed", SEED), ("control", CONTROL)):
            run = d.get(kind)
            if run is None:
                continue
            t, A = spectra(run["rows"])
            if not len(t):
                continue
            if mode:
                ax.semilogy(t, A[:, mode], color=color, label=f"{kind}: m = {mode}")
            others = A[:, 1:].copy()
            if mode:
                others[:, mode - 1] = 0
            ax.semilogy(t, others.max(axis=1), color=color, lw=0.8, ls=":", label=f"{kind}: largest other m")
            ta, na = series(run["rows"], "band_rho_active_modes")
            ax_count.plot(ta, na, color=color, lw=1.0, label=f"{kind}: active modes")
            tp, pn = series(run["rows"], "band_rho_participation")
            ax_count.plot(tp, pn, color=color, lw=1.0, ls="--", label=f"{kind}: participation")
        fit = r.get("seed_fit")
        if fit and mode:
            tt = np.linspace(*fit["window"], 20)
            ax.semilogy(tt, np.exp(fit["log_intercept"] + fit["growth_rate"] * tt), color="#e34948", lw=0.8,
                        label=rf"fit $\gamma$ = {fit['growth_rate']:.4f}")
        ax.set_title(NAMES[g], fontsize=9)
        ax.set_xlabel("t")
        ax_count.set_xlabel("t")
        ax.legend(fontsize=6)
        ax_count.legend(fontsize=6)
        rates = r.get("control_mode_rates") or {}
        if rates:
            m = np.array(sorted(int(k) for k in rates))
            gamma = np.array([rates[str(k)]["growth_rate"] if str(k) in rates else rates[k]["growth_rate"] for k in m])
            ax_rate.plot(m, gamma, "o", ms=3, color=CONTROL, label="control (noise-seeded)")
            mm = np.linspace(1, m.max(), 400)
            ax_rate.plot(mm, vortex_strip_rate(mm, r["band_length"], r["thickness"]), color=MUTED, lw=0.8, ls="--",
                         label="straight vortex strip")
        if fit and mode:
            ax_rate.plot([mode], [fit["growth_rate"]], "s", ms=5, color=SEED, label="seed fit")
        ax_rate.axhline(0, color=MUTED, lw=0.5)
        ax_rate.set_xlabel("band mode m")
        if ax_rate.get_legend_handles_labels()[0]:
            ax_rate.legend(fontsize=6)
    axes[0, 0].set_ylabel(r"$A_m$")
    axes[1, 0].set_ylabel("active modes, participation")
    axes[2, 0].set_ylabel(r"growth rate $\gamma_m$")
    save(fig, out / "e7_modes")


def snapshots_figure(data, out, *, kind="seed", times=(0.05, 10, 20, 30, 40, 60, 100, 200), width=700, stem=None):
    from hybridge.io.figures import signed_density_colors
    present = [g for g in GEOMETRIES if g in data and data[g].get(kind) is not None
               and data[g][kind]["snapshots"].exists()]
    if not present:
        return {}
    fig, axes = plt.subplots(len(present), len(times), figsize=(1.9 * len(times), 2.0 * len(present) + 0.4),
                             constrained_layout=True, squeeze=False)
    census = {}
    raster_cache = {}
    for row, g in enumerate(present):
        npz = np.load(data[g][kind]["snapshots"])
        saved = npz["times"]
        indices = [int(np.argmin(np.abs(saved - t))) for t in times]
        images, extent, mesh = rasterize(npz, indices, width)
        raster_cache[g] = (images, extent)
        low = min(np.nanmin(i) for i in images)
        cmap, norm = signed_density_colors(low, 1.0)
        walls = boundary_segments(mesh)
        for col, (index, image) in enumerate(zip(indices, images)):
            ax = axes[row, col]
            ax.imshow(image, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="none")
            ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color="#222222", lw=0.4)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"t = {saved[index]:g}", fontsize=8)
        axes[row, 0].set_ylabel(NAMES[g], fontsize=9)
        census[g] = dict(times=[float(saved[i]) for i in indices], **{
            "counts": [c["count"] for c in vortex_census(images, extent)]})
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes[:, -1].tolist(), shrink=0.8, label=r"$\rho_h$")
    save(fig, out / (stem or f"e7_{kind}_snapshots"))
    return census


def full_census(run, width=900):
    """Coherent vortex counts and centroids (``coherent``) at every saved state of one run."""
    if run is None or not run["snapshots"].exists():
        return None
    thickness = band_parameters(run).get("thickness") or 0.0
    npz = np.load(run["snapshots"])
    images, extent, mesh = rasterize(npz, range(len(npz["times"])), width)
    census = vortex_census(images, extent, core_radius=COHERENT_WIDTH * thickness)
    band_area = sum(census[0]["areas"])
    centroids = [coherent(c, band_area, thickness) for c in census]
    return dict(times=npz["times"].tolist(), raw_counts=[c["count"] for c in census],
                counts=[len(k) for k in centroids], coherent_fraction=COHERENT_FRACTION,
                coherent_width=COHERENT_WIDTH, band_area=band_area, centroids=centroids,
                areas=[c["areas"] for c in census])


def contour_curvature(coords, *, smoothing, samples=4096):
    """Signed curvature of the closed middle contour against normalized arclength, from a uniform
    resampling smoothed over ``smoothing`` (length): raw vertices can nearly coincide."""
    from scipy.ndimage import gaussian_filter1d
    closed = np.vstack([coords.vertices, coords.vertices[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    s = np.linspace(0.0, arc[-1], samples, endpoint=False)
    step = arc[-1] / samples
    xy = [gaussian_filter1d(np.interp(s, arc, closed[:, k]), max(smoothing / step, 1.0), mode="wrap") for k in (0, 1)]
    d1 = [np.gradient(np.concatenate([c[-2:], c, c[:2]]), step)[2:-2] for c in xy]
    d2 = [np.gradient(np.concatenate([c[-2:], c, c[:2]]), step)[2:-2] for c in d1]
    curvature = (d1[0] * d2[1] - d1[1] * d2[0]) / (d1[0] ** 2 + d1[1] ** 2) ** 1.5
    return s / arc[-1], curvature


def census_figure(data, results, out):
    from hybridge.cases.band_displacement import BandCoordinates
    present = [g for g in GEOMETRIES if results.get(g, {}).get("seed_census")]
    if not present:
        return
    fig, axes = plt.subplots(2, len(present), figsize=(3.4 * len(present), 6.0), constrained_layout=True,
                             squeeze=False, gridspec_kw=dict(height_ratios=(1, 2.2)))
    for col, g in enumerate(present):
        r = results[g]
        stage = Path(r["stage"])
        contours = np.load(stage / "contours.npz")
        coords = BandCoordinates.from_polyline(contours["middle"])
        s_curve, curvature = contour_curvature(coords, smoothing=0.25 * (r.get("thickness") or 0.01))
        npz = np.load(data[g]["seed"]["snapshots"])
        walls = boundary_segments(type("M", (), dict(triangles=npz["triangles"], node_coords=npz["node_coords"]))())
        from scipy.spatial import cKDTree
        wall_points = np.concatenate([walls[:, 0] + f * (walls[:, 1] - walls[:, 0]) for f in np.linspace(0, 1, 5)])
        clearance = cKDTree(wall_points).query(coords.vertices)[0]
        ax0 = axes[0, col]
        ax0.plot(s_curve, curvature, color=MUTED, lw=0.8, label="curvature")
        ax0.set_ylabel("curvature", fontsize=8)
        ax0.set_title(NAMES[g], fontsize=9)
        ax0.set_xlim(0, 1)
        ax1 = axes[1, col]
        census = r["seed_census"]
        for t, centres in zip(census["times"], census["centroids"]):
            if len(centres) < 2:  # the unbroken band is one component, not a vortex
                continue
            _, _, s, _, _ = coords.evaluate(np.asarray(centres))
            ax1.plot(s, np.full(len(s), t), "o", ms=2, color=SEED)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("normalized band arclength s")
        ax1.set_ylabel("t")
        # wall clearance as a shaded strip under the vortex track (darker = closer to the wall)
        extent = (0, 1, census["times"][0] - 6, census["times"][0] - 2)
        strip = (clearance - clearance.min()) / max(np.ptp(clearance), 1e-12)
        ax1.imshow(strip[None, :], extent=extent, aspect="auto", cmap="Greys_r", vmin=0, vmax=1)
        ax1.set_ylim(extent[2], census["times"][-1] + 2)
        ax1.text(0.01, extent[3] + 0.5, "wall clearance (dark: close)", fontsize=6, color=MUTED, va="bottom")
    save(fig, out / "e7_vortex_tracks")


# ------------------------------------------------------------------------------------------- driver
def analyse(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    style()
    data, results = {}, {}
    for g in args.geometries:
        pair = {k: find_run(args.runs, args.prefix, g, k) for k in ("control", "seed")}
        if all(v is None for v in pair.values()):
            continue
        data[g] = pair
        r = results[g] = {}
        seed_band = band_parameters(pair["seed"]) if pair["seed"] else {}
        band = seed_band or (band_parameters(pair["control"]) if pair["control"] else {})
        r.update(mode=seed_band.get("mode"), band_length=band.get("band_length"), thickness=band.get("thickness"),
                 circulation_period=band.get("circulation_period"), stage=band.get("stage"),
                 edge_displacement=seed_band.get("edge_displacement"))
        control, seed = pair["control"], pair["seed"]
        if control is not None:
            rows = control["rows"]
            r["transfer"] = {k: v for k, v in rows[0].items() if k.startswith("equilibrium_") and isinstance(v, (int, float))}
            t, v = series(rows, "diocotron_rho_eq_relative_l2")
            r["control_rho_drift"] = {f"{tt:g}": value_at(t, v, tt) for tt in DRIFT_TIMES}
            r["control_rho_drift_crossing"] = {f"{lvl:g}": first_crossing(t, v, lvl) for lvl in (1e-3, 1e-2, 1e-1)}
            t, v = series(rows, "diocotron_phi_eq_relative_l2")
            r["control_phi_drift"] = {f"{tt:g}": value_at(t, v, tt) for tt in DRIFT_TIMES}
            t, v = angular_departure(rows)
            r["control_angular_departure"] = {f"{tt:g}": value_at(t, v, tt) for tt in DRIFT_TIMES}
            r["control_angular_crossing"] = {f"{lvl:g}": first_crossing(t, v, lvl) for lvl in (1e-4, 1e-3, 1e-2, 1e-1)}
            ts, A = spectra(rows)
            if len(ts):
                rates, window = modal_growth_rates(ts, A)
                r["control_mode_rates"] = {str(m): f for m, f in rates.items()}
                r["control_rate_window"] = window
                positive = {m: f["growth_rate"] for m, f in rates.items() if f["growth_rate"] > 0}
                if positive:
                    best = max(positive, key=positive.get)
                    r["control_fastest_mode"] = dict(mode=best, growth_rate=positive[best])
                    r["control_unstable_band"] = [min(positive), max(positive)]
                    r["control_unstable_modes"] = len(positive)
                r["control_initial_noise"] = dict(max=float(A[0, 1:].max()), median=float(np.median(A[0, 1:])))
                r["control_dominant_modes_final"] = rows[-1].get("band_rho_dominant_modes")
            r["control_conservation"] = summary_conservation(control)
        if seed is not None:
            rows = seed["rows"]
            ts, A = spectra(rows)
            if len(ts) and r["mode"]:
                r["seed_initial_amplitude"] = float(A[0, r["mode"]])
                r["seed_initial_harmonic_ratio"] = float(A[0, 2 * r["mode"]] / A[0, r["mode"]]) if 2 * r["mode"] < A.shape[1] else None
                r["seed_fit"] = seeded_growth(ts, A, r["mode"])
                if r["seed_fit"] and r.get("band_length"):
                    r["seed_vortex_strip_rate"] = float(vortex_strip_rate(r["mode"], r["band_length"], r["thickness"]))
                if control is not None:
                    tc, Ac = spectra(control["rows"])
                    common = min(len(tc), len(ts))
                    ratio = Ac[:common, r["mode"]] / np.maximum(A[:common, r["mode"]], 1e-300)
                    r["control_to_seed_ratio_max_before_saturation"] = float(np.max(ratio[: max(1, common // 4)]))
                saturation = first_crossing(ts, A[:, r["mode"]], 0.5 * float(A[:, r["mode"]].max()))
                r["seed_saturation_time"] = saturation
            for key in ("band_rho_active_modes", "band_rho_participation", "band_rho_dominant_mode"):
                t, v = series(rows, key)
                r[f"seed_{key}"] = {f"{tt:g}": value_at(t, v, tt) for tt in DRIFT_TIMES}
            r["seed_conservation"] = summary_conservation(seed)
            if not args.skip_census:
                r["seed_census"] = full_census(seed)
                r["control_census"] = full_census(control) if control else None
        spectrum_figure(g, pair, out, r.get("mode"))
    if not data:
        raise SystemExit(f"no runs found in {args.runs}")
    preservation_figure(data, out)
    modes_figure(data, results, out)
    snapshots_figure(data, out, kind="seed")
    snapshots_figure(data, out, kind="control", times=(0.05, 20, 40, 60, 100, 200), stem="e7_control_snapshots")
    if not args.skip_census:
        census_figure(data, results, out)
    (out / "e7_summary.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (out / "e7_summary.md").write_text(markdown(results))
    print((out / "e7_summary.md").read_text())


def summary_conservation(run):
    rows = run["rows"]
    out = {}
    for key in ("mass_relative_drift", "energy_relative_drift", "enstrophy_relative_drift", "l2_budget_relative_defect",
                "poisson_physical_rel_residual"):
        t, v = series(rows, key)
        out[f"max_abs_{key}"] = float(np.max(np.abs(v))) if len(v) else None
    t, v = series(rows, "rho_min")
    out["rho_min"] = float(v.min()) if len(v) else None
    t, v = series(rows, "rho_max")
    out["rho_max"] = float(v.max()) if len(v) else None
    out["final_time"] = float(rows[-1]["time"]) if rows else None
    summary = run.get("summary_data") or {}
    out["wall_time_s"] = summary.get("wall_time_s")
    return out


def fmt(value, spec=".2e"):
    return "–" if value is None else format(value, spec)


def markdown(results):
    lines = ["# E7 band evolution: summary", ""]
    lines += ["## Preservation (controls)", "",
              "ρ drift at t = 1 is the edge relaxation of the projected profile (angle independent); the angular "
              "departure max_{m≥1} A_m is what the instability grows from.", "",
              "| Domain | transfer: HDG/CG potential | quadrature sensitivity | ρ drift t=1 | φ drift t=1 / 10 / 50 | "
              "angular departure t=1 / 10 / 50 | reaches 1e-3 / 1e-2 / 1e-1 at t | φ drift t=200 |",
              "|---|---|---|---|---|---|---|---|"]
    for g, r in results.items():
        tr, d = r.get("transfer", {}), r.get("control_rho_drift", {})
        phi, ang = r.get("control_phi_drift") or {}, r.get("control_angular_departure") or {}
        cross = r.get("control_angular_crossing") or {}
        lines.append(f"| {NAMES[g]} | {fmt(tr.get('equilibrium_poisson_cg_relative_l2'))} | "
                     f"{fmt(tr.get('equilibrium_projection_quadrature_relative_l2'))} | {fmt(d.get('1'))} | "
                     f"{fmt(phi.get('1'))} / {fmt(phi.get('10'))} / {fmt(phi.get('50'))} | "
                     f"{fmt(ang.get('1'))} / {fmt(ang.get('10'))} / {fmt(ang.get('50'))} | "
                     f"{fmt(cross.get('0.001'), '.1f')} / {fmt(cross.get('0.01'), '.1f')} / {fmt(cross.get('0.1'), '.1f')} | "
                     f"{fmt(phi.get('200'))} |")
    lines += ["", "## Band modes", "",
              "| Domain | ℓ | seed A_ℓ(0) | γ_ℓ seed (window) | vortex-strip γ_ℓ | control fastest m (γ) | unstable m (count) | active modes t=40 / 100 / 200 (seed) | vortices t=0 → T (seed) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for g, r in results.items():
        fit = r.get("seed_fit") or {}
        fast = r.get("control_fastest_mode") or {}
        active = r.get("seed_band_rho_active_modes", {})
        census = r.get("seed_census") or {}
        counts = census.get("counts") or []
        band = r.get("control_unstable_band")
        lines.append(f"| {NAMES[g]} | {r.get('mode')} | {fmt(r.get('seed_initial_amplitude'))} | "
                     f"{fmt(fit.get('growth_rate'), '.4f')} ({fmt((fit.get('window') or [None])[0], '.1f')}–{fmt((fit.get('window') or [None, None])[1], '.1f')}) | "
                     f"{fmt(r.get('seed_vortex_strip_rate'), '.4f')} | {fast.get('mode', '–')} ({fmt(fast.get('growth_rate'), '.4f')}) | "
                     f"{'–' if not band else f'{band[0]}–{band[1]}'} ({r.get('control_unstable_modes', '–')}) | "
                     f"{fmt(active.get('40'), '.0f')} / {fmt(active.get('100'), '.0f')} / {fmt(active.get('200'), '.0f')} | "
                     f"{counts[0] if counts else '–'} → {counts[-1] if counts else '–'} (max {max(counts) if counts else '–'}) |")
    lines += ["", "## Conservation", "", "| Domain | run | max mass drift | energy drift | enstrophy drift | max L² identity defect | ρ min | wall time (h) |",
              "|---|---|---|---|---|---|---|---|"]
    for g, r in results.items():
        for kind in ("control", "seed"):
            c = r.get(f"{kind}_conservation")
            if c:
                lines.append(f"| {NAMES[g]} | {kind} | {fmt(c['max_abs_mass_relative_drift'])} | {fmt(c['max_abs_energy_relative_drift'])} | "
                             f"{fmt(c['max_abs_enstrophy_relative_drift'])} | {fmt(c['max_abs_l2_budget_relative_defect'])} | "
                             f"{fmt(c['rho_min'], '.3f')} | {fmt(None if c['wall_time_s'] is None else c['wall_time_s'] / 3600, '.2f')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="e7")
    parser.add_argument("--geometries", default=",".join(GEOMETRIES))
    parser.add_argument("--skip-census", action="store_true")
    args = parser.parse_args(argv)
    args.geometries = [g for g in args.geometries.split(",") if g]
    analyse(args)


if __name__ == "__main__":
    main()
