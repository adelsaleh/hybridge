"""Article figures and table for the high-mode diocotron rings (m = 64, 128); no time integration.

Reads the summaries of ``run_diocotron_benchmark.py --benchmark disk-highmode``:

``<stem>_growth``
    one row per mode: amplitude of the seeded mode on the finest mesh with the
    matched unperturbed control; local growth rate against time with the
    modal-plateau window; plateau rate error against the number of triangles.
``<stem>_spectrum``
    azimuthal spectrum of phi_h - phi_eq at several times (finest mesh,
    IMEX-ARK3), with the unperturbed control at the final time.
``<stem>_cost``
    wall time per step and throughput (element unknowns per second) against the
    number of element unknowns, for both schemes.
``<stem>_rollup`` / ``<stem>_conservation`` (with ``--long``)
    density of the long runs at the saved times (full disk and a magnified
    sector; ``--rollup-mesh`` draws the triangles over the sector) and running
    maxima of the mass, W and enstrophy drifts.
``--table``
    booktabs rows: mode, band mesh, triangles, element and trace unknowns,
    scheme, drift, ms/step, rate error, control noise.

The field rasters of ``--snapshot-summary`` and ``--long`` average each pixel over
``--supersample`` x ``--supersample`` samples, so that a ring a few pixels wide is
not aliased into dots; their PDFs embed the rasters without resampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.guiding_center.diagnostics.run_records import records as _records, series as _series  # noqa: E402
from hybridge.io.figures import (  # noqa: E402
    add_matplotlib_mesh, publication_style, save_publication_figure, signed_density_colors)
from scripts.guiding_center.studies.diocotron.analyze_diocotron_campaign import analyse_row  # noqa: E402

SCHEME_COLORS = {"si-bdf2": "#2a78d6", "imex-ark3": "#eb6834"}
SCHEME_LABELS = {"si-bdf2": "SI-BDF2", "imex-ark3": "IMEX-ARK3"}
MODE_MARKERS = {64: "o", 128: "s"}
INK, MUTED = "#0b0b0b", "#52514e"
# Roll-up sector mesh overlay (_draw_mesh): edge widths ~`fraction` of the triangle size on the page, <= `linewidth` pt.
MESH_STYLE = dict(color="#b8b8b8", alpha=.3, linewidth=.12, fraction=.1)


def load_rows(summaries, *, velocity=None):
    rows = []
    for path in summaries:
        for row in json.loads(Path(path).read_text())["rows"]:
            if row["epsilon"] == 0.0 or (velocity is not None and row["velocity"] != velocity):
                continue
            rows.append(analyse_row(row, threshold=5e-3))
    return rows


def _error(row):
    value = row.get("linear_relative_error")
    return row["relative_error"] if value is None else value


def _control_path(row):
    return Path(row["jsonl"]).with_name(row["prefix"][: -len("seed")] + "control.jsonl")


def _local_rates(records, mode, width=4.0):
    t = _series(records, "time")
    log_amplitude = np.log(_series(records, f"diocotron_phi_mode_{mode}_l2"))
    step = int(round(width / np.median(np.diff(t))))
    return .5 * (t[step:] + t[:-step]), (log_amplitude[step:] - log_amplitude[:-step]) / (t[step:] - t[:-step])


def growth_figure(rows, stem):
    modes = sorted({row["mode"] for row in rows})
    with publication_style():
        figure, axes = plt.subplots(len(modes), 3, figsize=(7.2, 2.45 * len(modes) + .35), layout="constrained",
                                    squeeze=False)
        for (left, middle, right), mode in zip(axes, modes):
            mine = [row for row in rows if row["mode"] == mode]
            finest = max(row["triangles"] for row in mine)
            chosen = sorted((row for row in mine if row["triangles"] == finest), key=lambda r: r["scheme"])
            reference = chosen[0]["reference_gamma"]
            window = next((row["linear_window"] for row in chosen if row.get("linear_window")), None)
            for ax in (left, middle):
                if window:
                    ax.axvspan(*window, color="#eceae4", lw=0, zorder=0)
            for row in chosen:
                records = _records(row["jsonl"])
                color = SCHEME_COLORS[row["scheme"]]
                left.semilogy(_series(records, "time"), _series(records, f"diocotron_phi_mode_{mode}_l2"),
                              color=color, lw=1.2, label=SCHEME_LABELS[row["scheme"]])
                times, rates = _local_rates(records, mode)
                middle.plot(times, 100 * (rates / reference - 1), color=color, lw=1.1)
            anchor = next((row for row in chosen if row["scheme"] == "imex-ark3"), chosen[0])
            control = _control_path(anchor)
            if control.exists():
                records = _records(control)
                left.semilogy(_series(records, "time"), _series(records, f"diocotron_phi_mode_{mode}_l2"),
                              color=MUTED, lw=.8, linestyle=":", label="unperturbed control")
            if window:
                # Slope of the eigenvalue problem, anchored at the middle of the modal plateau.
                records = _records(anchor["jsonl"])
                t = _series(records, "time")
                log_amplitude = np.log(_series(records, f"diocotron_phi_mode_{mode}_l2"))
                middle_time = .5 * (window[0] + window[1])
                level = np.interp(middle_time, t, log_amplitude)
                x = np.linspace(0., min(70., window[1] + 25.), 200)
                left.semilogy(x, np.exp(level + reference * (x - middle_time)), color=INK, lw=.9,
                              linestyle=(0, (4, 2)), label="eigenvalue problem, $e^{\\gamma t}$")
            left.set(xlabel="$t$", ylabel=f"$\\|\\hat\\phi_{{{mode}}}\\|$ (mode ${mode}$)", xlim=(0, 70))
            left.grid(True, which="major", alpha=.5)
            middle.axhline(0, color=INK, lw=.8)
            middle.set(xlabel="$t$", ylabel=r"local rate $\gamma_{\rm loc}/\gamma-1$ [%]", xlim=(0, 50),
                       ylim=(-1.0, 1.0))
            middle.grid(True, which="major", alpha=.5)
            for scheme in ("si-bdf2", "imex-ark3"):
                series = sorted((row for row in mine if row["scheme"] == scheme and row["velocity"] == "raw"),
                                key=lambda r: r["triangles"])
                if series:
                    right.semilogx([row["triangles"] for row in series], [100 * _error(row) for row in series],
                                   color=SCHEME_COLORS[scheme], marker=MODE_MARKERS.get(mode, "o"), lw=1.2,
                                   markersize=4.5)
            right.axhline(0, color=INK, lw=.8)
            counts = sorted({row["triangles"] for row in mine})
            right.set_xticks(counts, [f"{count / 1e3:.0f}k" for count in counts])
            right.minorticks_off()
            right.set(xlabel="triangles", ylabel=r"$\gamma_h/\gamma-1$ [%] (plateau)")
            right.annotate(f"$m={mode}$, $\\gamma={reference:.7f}$", (.96, .5), xycoords="axes fraction",
                           fontsize=7, color=INK, ha="right")
            right.grid(True, which="major", alpha=.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
        return save_publication_figure(figure, stem)


def spectrum_figure(rows, stem, *, times=(0., 10., 20., 30., 40., 50., 60., 70.)):
    import matplotlib as mpl

    modes = sorted({row["mode"] for row in rows})
    colors = mpl.colormaps["Blues"](np.linspace(.35, 1., len(times)))
    with publication_style():
        figure, axes = plt.subplots(1, len(modes), figsize=(3.6 * len(modes) + 1.0, 2.7), layout="constrained",
                                    squeeze=False)
        for ax, mode in zip(axes[0], modes):
            mine = [row for row in rows if row["mode"] == mode and row["scheme"] == "imex-ark3"
                    and row["velocity"] == "raw"]
            row = max(mine, key=lambda r: r["triangles"])
            records = _records(row["jsonl"])
            keys = sorted((int(k[19:-3]) for k in records[1] if k.startswith("diocotron_phi_mode_")
                           and k.endswith("_l2") and k[19:-3].isdigit()))
            k = np.array(keys)
            t = _series(records, "time")
            for color, when in zip(colors, times):
                index = int(np.argmin(np.abs(t - when)))
                values = np.array([records[index][f"diocotron_phi_mode_{key}_l2"] for key in keys])
                ax.semilogy(k, np.maximum(values, 1e-30), color=color, lw=.9, label=f"$t={t[index]:g}$")
            control = _control_path(row)
            if control.exists():
                last = _records(control)[-1]
                values = np.array([last[f"diocotron_phi_mode_{key}_l2"] for key in keys])
                ax.semilogy(k, np.maximum(values, 1e-30), color=MUTED, lw=.8, linestyle=":",
                            label=f"control, $t={last['time']:g}$")
            for harmonic in (1, 2, 3):
                if harmonic * mode <= k[-1]:
                    ax.axvline(harmonic * mode, color=INK, lw=.6, linestyle=(0, (1, 2)))
            ax.set(xlabel="azimuthal mode $k$", ylabel=r"$\|\hat\phi_k\|$", xlim=(1, k[-1]),
                   title=f"$m={mode}$, {int(row['triangles']):,} triangles")
            ax.set_ylim(bottom=max(1e-20, np.nanmin(ax.get_ylim())))
            ax.grid(True, which="major", alpha=.5)
        handles, labels = axes[0, -1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside right center", frameon=False, fontsize=6.5)
        return save_publication_figure(figure, stem)


def cost_figure(rows, stem):
    rows = [row for row in rows if row["velocity"] == "raw"]  # the compatible drift adds <5% (table)
    with publication_style():
        figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.85), layout="constrained")
        for scheme in ("si-bdf2", "imex-ark3"):
            for mode in sorted({row["mode"] for row in rows}):
                points = sorted({(row["element_unknowns"], row["median_linear_step_wall_time"]) for row in rows
                                 if row["scheme"] == scheme and row["mode"] == mode
                                 and row.get("median_linear_step_wall_time")})
                if not points:
                    continue
                unknowns, step = np.array(points, dtype=float).T
                style = dict(color=SCHEME_COLORS[scheme], marker=MODE_MARKERS.get(mode, "o"), lw=1.2,
                             markersize=4.5, linestyle="-" if mode == 64 else (0, (4, 2)),
                             label=f"{SCHEME_LABELS[scheme]}, $m={mode}$")
                left.loglog(unknowns, step, **style)
                right.semilogx(unknowns, unknowns / step / 1e6, **style)
        guide = np.array(sorted({row["element_unknowns"] for row in rows}), dtype=float)
        if guide.size:
            first = min((row["element_unknowns"], row["median_linear_step_wall_time"]) for row in rows
                        if row["scheme"] == "si-bdf2" and row.get("median_linear_step_wall_time"))
            left.loglog(guide, first[1] * guide / first[0], color=MUTED, lw=.8, linestyle=(0, (4, 2)),
                        label="linear scaling")
        left.set(xlabel="element unknowns", ylabel="wall time per step [s]")
        handles, labels = left.get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=7)
        left.grid(True, which="major", alpha=.5)
        right.set(xlabel="element unknowns", ylabel="throughput [$10^6$ element unknowns/s]")
        right.set_ylim(bottom=0)
        right.grid(True, which="major", alpha=.5)
        return save_publication_figure(figure, stem)


def _mesh_bounds(mesh, width, height):
    """Raster bounds of ``RasterGeometry.from_mesh``: the mesh box enlarged by 4% to the raster aspect ratio."""
    from hybridge.io.raster import raster_bounds

    return raster_bounds(mesh, width, height)


def _sector_raster(sector, width):
    """Bounds (x0, x1, y0, y1) and (width, height) of a sector raster ``width`` pixels wide, square pixels."""
    (x0, x1), (y0, y1) = sector
    return (x0, x1, y0, y1), (width, max(2, int(round(width * (y1 - y0) / (x1 - x0)))))


def _area_rasters(space, bounds, width, height, fields, *, supersample=1, max_bytes=2 * 1024**3):
    """Rasters of DG fields, each pixel the mean of ``supersample**2`` samples on a regular sub-grid.

    Rows run top to bottom (``imshow(..., origin="upper")``), with the pixel centres of
    ``RasterGeometry.from_mesh`` for the same bounds; ``supersample=1`` samples the pixel
    centres only. Samples outside the mesh are ignored, and a pixel is masked only when all
    its samples are. ``fields`` are callables of ``sample``, which evaluates DG coefficients
    at the current samples; a nonlinear field such as ``|q|^2`` is formed before averaging.
    One sampling matrix ((p+1)(p+2)/2 nonzeros per sample) is held at a time: one per
    sub-pixel offset (:func:`hybridge.io.raster.subpixel_geometries`), applied to every field
    and released before the next offset. For a linear field sampled at many times,
    :func:`hybridge.io.raster.area_sampling_matrix` gives the same pixels as one sparse map.
    """
    from hybridge.io.raster import subpixel_geometries

    sums = [np.zeros(width * height) for _ in fields]
    counts = np.zeros(width * height)
    for geometry in subpixel_geometries(space.mesh, bounds, width, height, supersample=supersample):
        inside = geometry.element_ids >= 0
        matrix = geometry.sampling_matrix(space, max_bytes=max_bytes)

        def sample(coefficients):
            return matrix @ np.asarray(coefficients).ravel()

        for total, field in zip(sums, fields):
            total[inside] += np.asarray(field(sample))[inside]
        counts += inside
        del geometry, matrix, sample
    mask = (counts == 0).reshape(height, width)
    return [np.ma.array((total / np.maximum(counts, 1)).reshape(height, width), mask=mask) for total in sums]


def _save_raster_figure(figure, stem, *, dpi=600):
    """Save a figure of field rasters: PDF with the computed pixels, SVG and PNG anti-aliased.

    With ``interpolation="none"`` the PDF embeds every raster as computed and the
    viewer scales it. The PNG (Agg) backend would take nearest-neighbour pixels
    instead, and SVG renderers show such images pixelated, so those two formats are
    resampled at ``dpi`` with Matplotlib's anti-aliasing filter.
    """
    from matplotlib.image import AxesImage

    images = figure.findobj(AxesImage)
    for image in images:
        image.set_interpolation("none")
    paths = save_publication_figure(figure, stem, formats=("pdf",), dpi=dpi)
    for image in images:
        image.set_interpolation("antialiased")
    return paths + save_publication_figure(figure, stem, formats=("svg", "png"), dpi=dpi)


def _snapshot_rasters(npz, pixels, *, sector, indices=None, sector_pixels=None, supersample=1):
    """Full-disk and magnified-sector rasters of the saved densities ``indices`` (default: all).

    The full disk is ``pixels`` square with the bounds of ``RasterGeometry.from_mesh``,
    the sector ``sector_pixels`` (default ``pixels``) wide; each pixel is the mean of
    ``supersample**2`` samples (:func:`_area_rasters`).
    """
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace

    mesh = DGMesh(npz["node_coords"], npz["triangles"])
    space = DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]))
    densities = npz["density"]  # decompress once
    fields = [lambda sample, c=densities[i]: sample(c)
              for i in (range(len(densities)) if indices is None else indices)]
    full = _mesh_bounds(mesh, pixels, pixels)
    zoom, (width, height) = _sector_raster(sector, sector_pixels or pixels)
    return [(full, _area_rasters(space, full, pixels, pixels, fields, supersample=supersample)),
            (zoom, _area_rasters(space, zoom, width, height, fields, supersample=supersample))]


def _draw_mesh(ax, mesh, bounds, points_per_unit, *, color, alpha, linewidth, fraction):
    """Triangle edges over a field panel, line widths proportional to the triangle size on the page.

    One width cannot serve a graded mesh: lines wide enough to show around coarse triangles
    at print size cover the smallest triangles at any magnification. Here each triangle's
    edges are ``linewidth / 2**k`` points wide (``k >= 0``), between one and two times
    ``fraction`` of its mean edge length on the page (``points_per_unit`` points per unit
    length), capped at ``linewidth``; the smallest triangles then stay open when the PDF
    is magnified. Each octave class is one vector path
    (:func:`hybridge.io.figures.add_matplotlib_mesh` with ``bounds``); edges between two
    classes are drawn in both.
    """
    from types import SimpleNamespace

    nodes, triangles = np.asarray(mesh.node_coords), np.asarray(mesh.triangles)
    corners = nodes[triangles]
    size = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).mean(axis=1) * points_per_unit
    classes = np.floor(np.log2(linewidth / np.minimum(linewidth, fraction * size)))
    for level in np.unique(classes):
        add_matplotlib_mesh(ax, SimpleNamespace(node_coords=nodes, triangles=triangles[classes == level]),
                            color=color, alpha=alpha, linewidth=linewidth / 2 ** level, bounds=bounds)


def rollup_figure(snapshot_files, stem, *, times=None, pixels=1800, sector_pixels=1600, supersample=1,
                  sector=((-.2, .2), (.62, .97)), dpi=600, mesh_overlay=False):
    """Density of the long runs at the saved times: full disk and one sector at the top of the ring.

    The sector is the same for every mode, wide enough to follow the vortices
    after they merge and leave the ring radius. The full disk is ``pixels`` square,
    the sector ``sector_pixels`` wide; ``supersample`` > 1 averages each pixel over a
    sub-grid (:func:`_area_rasters`), so that a ring a few pixels wide stays continuous
    on the full disk instead of breaking into dots. ``mesh_overlay`` draws the edges of
    the run's triangles over the sector panels as vector lines (:func:`_draw_mesh`,
    ``MESH_STYLE``), so the triangle size can be compared with the vortices; the
    finest triangles resolve when the PDF is magnified.
    """
    from types import SimpleNamespace

    from matplotlib.patches import Rectangle

    rows = []
    overlays = []
    (x0, x1), (y0, y1) = sector
    for mode, path in snapshot_files:
        npz = np.load(path)
        saved = [round(float(t), 6) for t in npz["times"]]
        chosen = [t for t in (times or saved) if round(t, 6) in saved]
        (full_bounds, full_images), (zoom_bounds, zoom_images) = _snapshot_rasters(
            npz, pixels, sector=sector, indices=[saved.index(round(t, 6)) for t in chosen],
            sector_pixels=sector_pixels, supersample=supersample)
        mesh = SimpleNamespace(node_coords=npz["node_coords"], triangles=npz["triangles"]) if mesh_overlay else None
        rows.append((mode, chosen, full_bounds, full_images, zoom_bounds, zoom_images, mesh))
    columns = max(len(row[1]) for row in rows)
    shown = [image for row in rows for images in (row[3], row[5]) for image in images]
    density_map, density_norm = signed_density_colors(min(float(image.min()) for image in shown), 1.0)
    with publication_style():
        figure, axes = plt.subplots(2 * len(rows), columns, figsize=(1.2 * columns, 2.35 * len(rows) + .5),
                                    constrained_layout=True, squeeze=False)
        for r, (mode, chosen, full_bounds, full_images, zoom_bounds, zoom_images, mesh) in enumerate(rows):
            for c, t in enumerate(chosen):
                for k, (bounds, images) in enumerate(((full_bounds, full_images), (zoom_bounds, zoom_images))):
                    ax = axes[2 * r + k, c]
                    artist = ax.imshow(images[c], extent=bounds, origin="upper", cmap=density_map,
                                       norm=density_norm, interpolation="none")
                    if k == 1 and mesh is not None:
                        overlays.append((ax, mesh, bounds))
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(k == 1)
                        spine.set_color(MUTED)
                    if k == 0:
                        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=.6))
                        if r == 0:
                            ax.set_title(f"$t={t:g}$", fontsize=8)
                if c == 0:
                    axes[2 * r, 0].set_ylabel(f"$m={mode}$")
                    axes[2 * r + 1, 0].set_ylabel("sector")
            for c in range(len(chosen), columns):
                axes[2 * r, c].axis("off"); axes[2 * r + 1, c].axis("off")
        figure.colorbar(artist, ax=axes, shrink=.35, aspect=35, location="bottom", pad=.01,
                        label=r"density $\rho_h$")
        if overlays:
            figure.draw_without_rendering()  # final panel sizes: line widths follow the triangles on the page
            for ax, mesh, bounds in overlays:
                points_per_unit = ax.bbox.width * 72 / figure.dpi / (bounds[1] - bounds[0])
                _draw_mesh(ax, mesh, bounds, points_per_unit, **MESH_STYLE)
                ax.set(xlim=bounds[:2], ylim=bounds[2:])  # triangles crossing the sector edge are clipped there
        return _save_raster_figure(figure, stem, dpi=dpi)


def _sector(mode, *, radius=.8, wavelengths=3.):
    """Window ((x0, x1), (y0, y1)) at the top of the ring, ``wavelengths`` azimuthal wavelengths wide."""
    half = .5 * wavelengths * 2 * np.pi * radius / mode
    return (-half, half), (radius - .54 * half, radius + .54 * half)


def highmode_snapshot_figure(row, npz_path, stem, *, tau, times=None, pixels=1800, sector_pixels=1600, zoom=None,
                             supersample=1, dpi=600):
    """Density (full disk and magnified), potential perturbation and |E|^2 at several times.

    As the Zoni--Guclu snapshot figure, with a magnified sector at the top of the ring
    (the m vortices are too small on the full disk; by default three azimuthal
    wavelengths wide) and phi_h - phi_h^eq instead of phi_h, whose equilibrium part
    hides the perturbation. phi_h^eq is read from the snapshot file when the run
    saved it; otherwise it, and t = 0 when requested but not saved, are rebuilt
    offline: conforming projection plus one host HDG Poisson solve each. The full
    disk is ``pixels`` square, the sector ``sector_pixels`` wide; ``supersample`` > 1
    averages each pixel of every panel over a sub-grid (:func:`_area_rasters`).
    """
    import matplotlib as mpl
    from matplotlib.patches import Rectangle

    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from scripts.guiding_center.studies.diocotron.plot_diocotron_zg import _initial_state

    npz = np.load(npz_path)
    saved = [round(float(t), 6) for t in npz["times"]]
    if times is None:  # the first saved step when the run stored it (t = dt), else t = 0
        times = (saved[0] if saved[0] < 1. else 0., 50., 60., 70.)
    density_all, potential_all, flux_all = npz["density"], npz["potential"], npz["flux"]  # decompress once
    states = {round(t, 6): (density_all[saved.index(round(t, 6))], potential_all[saved.index(round(t, 6))],
                            flux_all[saved.index(round(t, 6))]) for t in times if round(t, 6) in saved}
    if "equilibrium_potential" in npz.files:
        space = DGSpace(DGMesh(npz["node_coords"], npz["triangles"]), int(npz["order"]),
                        basis_type=str(npz["basis"]))
        potential_eq = npz["equilibrium_potential"]
    else:
        space, _, potential_eq, _ = _initial_state(npz, row, tau, equilibrium=True)
    if 0.0 in times and 0.0 not in states:
        states[0.0] = _initial_state(npz, row, tau)[1:]
    missing = [t for t in times if round(t, 6) not in states]
    if missing:
        print(f"{npz_path}: no saved state at t = {missing} (saved: {saved})")
    times = [t for t in times if round(t, 6) in states]
    if zoom is None:
        zoom = _sector(int(row["label"].split("_")[0][1:]))
    (x0, x1), (y0, y1) = zoom
    full_bounds = _mesh_bounds(space.mesh, pixels, pixels)
    sector_bounds, (width, height) = _sector_raster(zoom, sector_pixels)
    chosen = [states[round(t, 6)] for t in times]

    def sampler(bounds, width, height, fields):
        return _area_rasters(space, bounds, width, height, fields, supersample=supersample)

    full_densities = sampler(full_bounds, pixels, pixels,
                             [lambda sample, c=density: sample(c) for density, _, _ in chosen])
    sector_fields = []
    for density, potential, flux in chosen:  # |q_h|^2 is averaged after squaring
        sector_fields += [lambda sample, c=density: sample(c),
                          lambda sample, c=np.asarray(potential) - potential_eq: sample(c),
                          lambda sample, q=flux: sample(q[0]) ** 2 + sample(q[1]) ** 2]
    sector_images = sampler(sector_bounds, width, height, sector_fields)
    rows = []
    for i, t in enumerate(times):
        density_sector, perturbation, energy = sector_images[3 * i:3 * i + 3]
        scale = float(np.ma.max(np.ma.abs(perturbation)))
        rows.append((t, full_densities[i], density_sector, perturbation / scale, scale, energy))
    energy_max = max(float(row_[5].max()) for row_ in rows)
    density_map, density_norm = signed_density_colors(
        min(min(float(row_[1].min()), float(row_[2].min())) for row_ in rows), 1.0)
    titles = ("density $\\rho_h$", "density (sector)",
              "$(\\phi_h-\\phi_h^{\\rm eq})/\\max|\\phi_h-\\phi_h^{\\rm eq}|$", "$|E_h|^2=|q_h|^2$ (sector)")
    with publication_style():
        figure, axes = plt.subplots(len(times), 4, figsize=(7.2, 1.3 * len(times) + .6), layout="constrained",
                                    gridspec_kw=dict(width_ratios=(1., 2., 2., 2.)), squeeze=False)
        for i, (t, density_full, density_sector, perturbation, scale, energy) in enumerate(rows):
            panels = ((density_full, full_bounds, density_map, density_norm),
                      (density_sector, sector_bounds, density_map, density_norm),
                      (perturbation, sector_bounds, "RdBu_r", mpl.colors.Normalize(-1., 1.)),
                      (energy, sector_bounds, "magma", mpl.colors.Normalize(0., energy_max)))
            for j, (image, bounds, cmap, norm) in enumerate(panels):
                ax = axes[i, j]
                artist = ax.imshow(image, extent=bounds, origin="upper", cmap=cmap, norm=norm,
                                   interpolation="none")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(j > 0)
                    spine.set_color(MUTED)
                if i == 0:
                    ax.set_title(titles[j], fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"$t={t:g}$")
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=.6))
                if j == 2:
                    ax.annotate(f"max {scale:.1e}", (.98, .04), xycoords="axes fraction", ha="right", fontsize=6,
                                color=INK)
                if i == len(times) - 1:
                    figure.colorbar(artist, ax=axes[:, j], location="bottom", shrink=.8, pad=.01, aspect=25)
        return _save_raster_figure(figure, stem, dpi=dpi)


def _movie_panel_boxes(raster=1024, size=(1700, 820), bounds=(-1.04, 1.04, -1.04, 1.04), title="caption"):
    """Pixel boxes (top, bottom, left, right) of the two panels in a campaign movie frame.

    Rebuilds the movie figure of ``run_diocotron_benchmark._MovieObserver`` (same panel
    layout, size and fonts) and reads the drawn image extents.
    """
    import matplotlib as mpl

    from hybridge.io.matplotlib import MatplotlibRasterPanels

    blank = np.ma.array(np.zeros((raster, raster)))
    density_map = mpl.colormaps["viridis"].with_extremes(under="magenta")
    panels = MatplotlibRasterPanels(
        [("Density", blank, dict(cmap=density_map, clim=(-1e-3, 1.02))),
         ("perturbation", blank, dict(cmap="RdBu_r", clim=(-1., 1.)))],
        bounds, title=title, size=size, ticks=False)  # as RasterMovieRecorder
    try:
        height = panels.figure.bbox.height
        boxes = []
        for image in panels.images:
            extent = image.get_window_extent()
            boxes.append((int(round(height - extent.y1)), int(round(height - extent.y0)),
                          int(round(extent.x0)), int(round(extent.x1))))
        return boxes
    finally:
        panels.close()


def movie_frames_figure(movie, stem, *, times=(0., 50., 60., 70.), interval=.5, first_time=.05,
                        bounds=(-1.04, 1.04, -1.04, 1.04), sector=((-.16, .16), (.715, .885))):
    """Density, a magnified density sector and the normalized potential perturbation from a movie.

    Frame k is at t = interval*k (k >= 1) and frame 0 at the first step. Colors and
    limits are those of the movie: density with undershoots below -1e-3 in magenta,
    (phi_h - phi_h^eq)/max|phi_h - phi_h^eq| on [-1, 1]. The sector is magnified from
    the movie pixels (no new data), at the top of the ring.
    """
    import imageio_ffmpeg
    import matplotlib as mpl
    from matplotlib.patches import Rectangle

    wanted = {(0 if t <= first_time else int(round(t / interval))): t for t in times}
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
    boxes = _movie_panel_boxes(size=(width, height), title="caption with the movie's length " + "x" * 90)
    inset = 8  # skip the axes frame drawn around each movie panel (the disk keeps a 2% margin)

    def crop(index, box):
        top, bottom, left, right = box
        return frames[index][top + inset:bottom - inset, left + inset:right - inset]

    def sector_crop(image):
        rows, columns = image.shape[:2]
        (x0, x1), (y0, y1) = sector
        xmin, xmax, ymin, ymax = bounds
        c0, c1 = (int(round((x - xmin) / (xmax - xmin) * columns)) for x in (x0, x1))
        r0, r1 = (int(round((ymax - y) / (ymax - ymin) * rows)) for y in (y1, y0))
        return image[r0:r1, c0:c1]

    density_map = mpl.colormaps["viridis"].with_extremes(under="magenta")
    (x0, x1), (y0, y1) = sector
    with publication_style():
        figure, axes = plt.subplots(3, len(times), figsize=(1.75 * len(times) + .3, 4.8), layout="constrained",
                                    squeeze=False, gridspec_kw=dict(height_ratios=(1., (y1 - y0) / (x1 - x0), 1.)))
        for column, index in enumerate(sorted(wanted)):
            density, perturbation = (crop(index, box) for box in boxes)
            panels = (density, sector_crop(density), perturbation)
            for row, image in enumerate(panels):
                ax = axes[row, column]
                ax.imshow(image, interpolation="nearest" if row == 1 else "antialiased")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(row == 1)
                    spine.set_color(MUTED)
            rows_, columns_ = density.shape[:2]
            xmin, xmax, ymin, ymax = bounds
            axes[0, column].add_patch(Rectangle(((x0 - xmin) / (xmax - xmin) * columns_,
                                                 (ymax - y1) / (ymax - ymin) * rows_),
                                                (x1 - x0) / (xmax - xmin) * columns_,
                                                (y1 - y0) / (ymax - ymin) * rows_,
                                                fill=False, ec="white", lw=.6))
            axes[0, column].set_title(f"$t={wanted[index]:g}$", fontsize=8)
        axes[0, 0].set_ylabel(r"density $\rho_h$", fontsize=8)
        axes[1, 0].set_ylabel("sector", fontsize=8)
        axes[2, 0].set_ylabel(r"$(\phi_h-\phi_h^{\rm eq})/\max|\cdot|$", fontsize=8)
        for rows, (cmap, clim, extend) in (((0, 1), (density_map, (-1e-3, 1.02), "min")),
                                           ((2,), ("RdBu_r", (-1., 1.), "neither"))):
            mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*clim), cmap=cmap)
            figure.colorbar(mappable, ax=[axes[r, c] for r in rows for c in range(len(times))], location="right",
                            shrink=.85, pad=.01, extend=extend)
        return save_publication_figure(figure, stem, dpi=300)


def conservation_figure(long_rows, stem):
    keys = (("mass_relative_drift", r"$\max_{s\leq t}\delta M$"),
            ("energy_relative_drift", r"$\max_{s\leq t}\delta W$"),
            ("enstrophy_relative_drift", r"$\max_{s\leq t}\delta\mathcal{E}$"))
    with publication_style():
        figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True, sharex=True)
        for row in long_rows:
            records = _records(row["jsonl"])
            t = _series(records, "time")
            for ax, (key, label) in zip(axes, keys):
                drift = np.maximum.accumulate(np.nan_to_num(np.abs(_series(records, key))))
                raw = row["velocity"] == "raw"  # wide and pale under the compatible curve, which overlays it
                ax.semilogy(t[1:], np.maximum(drift[1:], 1e-16), lw=2.4 if raw else .9, alpha=.55 if raw else 1.,
                            color=SCHEME_COLORS[row["scheme"]] if raw else INK,
                            linestyle=(0, (1, 0)) if row["mode"] == 64 else (0, (4, 2)),
                            label=f"$m={row['mode']}$, {row['velocity']}")
                ax.set(xlabel="$t$", ylabel=label)
                ax.grid(True, which="major", alpha=.5)
        axes[-1].legend(frameon=False, fontsize=6.5)
        return save_publication_figure(figure, stem)


def _scientific(value) -> str:
    """LaTeX a x 10^b with one significant digit."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def latex_table(rows) -> str:
    lines = [r"\setlength{\tabcolsep}{4.5pt}\begin{tabular}{rrrrrllrrr}\toprule",
             r"$m$ & $h_{\rm band}$ & triangles & DG unknowns & trace unknowns & scheme & drift & ms/step & "
             r"$\gamma_h/\gamma-1$ & control\\\midrule"]
    for row in sorted(rows, key=lambda r: (r["mode"], -r["band_mesh_size"], r["scheme"], r["velocity"])):
        step = row.get("median_linear_step_wall_time")
        control = row.get("control_fraction")
        lines.append(
            f"{row['mode']} & {row['band_mesh_size']:g} & {int(row['triangles']):,} & "
            f"{row['element_unknowns'] / 1e6:.2f}M & {row['trace_unknowns'] / 1e6:.2f}M & "
            f"{SCHEME_LABELS[row['scheme']]} & {row['velocity']} & "
            + ("--" if step is None else f"{1e3 * step:.0f}") + f" & ${100 * _error(row):+.3f}\\%$ & "
            + ("--" if control is None else _scientific(control)) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines).replace(",", "{,}") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summaries", nargs="+", type=Path, help="hm64/hm128 campaign summaries")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="diocotron_highmode")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--long", nargs="*", type=Path, default=(), help="long-run summaries (T=400)")
    parser.add_argument("--rollup-times", type=lambda text: tuple(float(v) for v in text.split(",")), default=None)
    parser.add_argument("--rollup-modes", type=lambda text: tuple(int(v) for v in text.split(",")), default=None,
                        help="modes shown in the roll-up figure (default: every long run)")
    parser.add_argument("--rollup-mesh", action="store_true",
                        help="draw the mesh triangles over the roll-up sector panels (vector lines)")
    parser.add_argument("--snapshot-summary", type=Path, help="summary of a seeded run with --snapshot-times")
    parser.add_argument("--snapshot-times", type=lambda text: tuple(float(v) for v in text.split(",")),
                        default=None, help="default: first saved step (or 0), 50, 60, 70")
    parser.add_argument("--supersample", type=int, default=1,
                        help="snapshot and roll-up rasters: each pixel is the mean of N x N samples on a sub-grid "
                             "(default 1: pixel centres only; the article figures use 3)")
    parser.add_argument("--poisson-tau", type=float, default=128000.0)
    parser.add_argument("--movie-frames", nargs="*", type=Path, default=(),
                        help="campaign movies to crop into a density/perturbation frame figure")
    parser.add_argument("--frame-times", type=lambda text: tuple(float(v) for v in text.split(",")),
                        default=(0., 50., 60., 70.))
    args = parser.parse_args()
    if args.supersample < 1:
        parser.error("--supersample must be a positive integer")
    rows = load_rows(args.summaries)
    raw = [row for row in rows if row["velocity"] == "raw"]
    print(growth_figure(raw, args.output_dir / f"{args.stem}_growth"))
    print(spectrum_figure(raw, args.output_dir / f"{args.stem}_spectrum"))
    print(cost_figure(rows, args.output_dir / f"{args.stem}_cost"))
    if args.table is not None:
        args.table.write_text("% Generated by scripts/guiding_center/studies/diocotron/plot_diocotron_highmode.py\n"
                              + latex_table(rows))
        print(args.table)
    for movie in args.movie_frames:
        mode = next((part[1:] for part in movie.stem.split("_") if part.startswith("m") and part[1:].isdigit()), "")
        print(movie_frames_figure(movie, args.output_dir / f"{args.stem}_frames_m{mode}", times=args.frame_times))
    if args.snapshot_summary is not None:
        for row in json.loads(args.snapshot_summary.read_text())["rows"]:
            npz = Path(row["jsonl"]).with_name(f"{row['prefix']}_snapshots.npz")
            # The compatible drift gives the same fields to ~1e-6 (table): one figure per mode.
            if row["epsilon"] != 0.0 and row["velocity"] == "raw" and npz.exists():
                stem = args.output_dir / f"{args.stem}_snapshots_{row['label'].split('_')[0]}"
                print(highmode_snapshot_figure(row, npz, stem, tau=args.poisson_tau, times=args.snapshot_times,
                                               supersample=args.supersample))
    if args.long:
        long_rows = load_rows(args.long)
        print(conservation_figure(long_rows, args.output_dir / f"{args.stem}_conservation"))
        snapshots = []
        for row in sorted(long_rows, key=lambda r: r["mode"]):
            path = Path(row["jsonl"]).with_name(f"{row['prefix']}_snapshots.npz")
            if row["velocity"] == "raw" and path.exists() and (args.rollup_modes is None
                                                                 or row["mode"] in args.rollup_modes):
                snapshots.append((row["mode"], path))
        if snapshots:
            print(rollup_figure(snapshots, args.output_dir / f"{args.stem}_rollup", times=args.rollup_times,
                                supersample=args.supersample, mesh_overlay=args.rollup_mesh))


if __name__ == "__main__":
    main()
