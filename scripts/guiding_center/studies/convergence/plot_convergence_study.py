#!/usr/bin/env python3
"""Figures of the manufactured guiding-center convergence study (article section 5).

* ``spatial_convergence.pdf``: 2x2 panels against h = sqrt(|Omega|/N_K), one curve per p:
  rho_h and phi_h in L2(Omega); the traces rhohat_h and phihat_h on the interior skeleton
  (rhohat: |u.n|-weighted solid, unweighted dashed; phihat: vs phi solid, vs P_F phi dashed);
  reference slopes p + 1/2 (density) and p + 1 (potential); ``--raised poisson_degree.csv`` adds the
  p_phi = p_rho + 1 density dashed (density slopes p_rho + 1; labels p_rho, p_phi), ``--lattice
  gmsh_lattice.json`` draws the meshes with a vertical lattice edge family with open markers;
  ``--compatible CSV...`` (per-run CSVs of the compatible drift with p_phi = p_rho, ``*_dthalf`` skipped) adds
  its density errors as crosses, with a line-style legend in the density panel;
* ``work_precision.pdf``: rho and phi L2 errors against the steady wall time per step (``--work-total``: per
  run, steps x median step time) and against the number of element unknowns, one curve per p;
  ``--compatible`` with ``--compatible-cost ark3_step_times.json`` (timing_device/ark3_timing_summary.py) adds
  the compatible drift dashed with crosses in the density panels, at the time of the raw p_phi = p_rho run times the measured ratio of
  their IMEX-ARK3 step times (meshes without a measured ratio are left out);
* ``temporal_convergence.pdf``: density (left) and potential (right) against dt, one colour per
  scheme, element unknowns solid and traces dashed (top axis: Courant number; ladders shorter
  than two decades are labelled at their time steps); errors against the exact solution, with the
  ``--temporal-reference`` run's own error against it (the spatial error of that mesh) as a grey horizontal
  line (``--temporal-against-reference``: the differences from that run instead).

* ``<tag>wave_<stem>.pdf`` (``--wave-figure NPZ``): exact density at t=0 and T, numerical
  rho_h(T) (with the mesh) and the error, from ``snapshot_manufactured_solution.py``;
* ``<tag>alignment_mesh.pdf`` (``--alignment-mesh``): element-level zoom of the structured
  aligned / misaligned meshes, edges coloured by the exact u.n, points where u.n = 0, integral
  lines of the drift and the density in grey (no simulation);
* ``vortex_setup.pdf`` (``--vortex-setup``): exact density of ``orbiting_vortex`` at
  t = 0, T/2, T on its polygon, with the orbit, the support circle |x - c(t)| = R, the
  re-entrant corner it covers, and a coarse mesh (h0 = 0.1) in the first panel.

h is (|Omega|/N_K)^(1/2) with the recorded domain area (4 for the square). ``--tag`` prefixes
the output names (e.g. ``vortex_``).

Input: the CSV files written by ``run_guiding_center_temporal_convergence.py``.
"""

from __future__ import annotations

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

SCHEMES = {"si-euler": ("SI-Euler", "o", 1), "si-bdf2": ("SI-BDF2", "s", 2), "imex-ark3": ("IMEX-ARK3", "^", 3)}


def _read(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _by_order(rows):
    groups = {}
    for row in rows:
        groups.setdefault(int(row["order"]), []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: int(r["triangles"]))
    return dict(sorted(groups.items()))


def _style():
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 7,
                         "lines.markersize": 4})


PANELS = (
    # (row, col, solid key, dashed key or None, y label, slope offset)
    (0, 0, "rho_l2_error", None, r"$\|\rho_h-\rho\|_{L^2(\Omega)}$", 0.5),
    (0, 1, "rho_trace_flux_l2_error", "rho_trace_l2_error",
     r"$\||\mathbf{u}\cdot\mathbf{n}|^{1/2}(\widehat\rho_h-\rho)\|_h$ (solid), "
     r"$\|\widehat\rho_h-\rho\|_h$ (dashed)", 0.5),
    (1, 0, "phi_l2_error", None, r"$\|\phi_h-\phi\|_{L^2(\Omega)}$", 1.0),
    (1, 1, "phi_trace_l2_error", "phi_trace_projection_error",
     r"$\|\widehat\phi_h-\phi\|_h$ (solid), $\|\widehat\phi_h-\Pi_\partial\phi\|_h$ (dashed)", 1.0),
)


TEMPORAL_PANELS = (
    # (element key, trace key, legend title, element label, trace label); with a reference run the
    # keys become *_ref_l2_error and the element key names the reference's own error (grey floor).
    ("rho_l2_error", "rho_trace_flux_l2_error", "density", r"$\rho_h$", r"$\widehat\rho_h$"),
    ("phi_l2_error", "phi_trace_l2_error", "potential", r"$\phi_h$", r"$\widehat\phi_h$"),
)


def _h(row):
    return np.sqrt(float(row.get("domain_area") or 4.0) / int(row["triangles"]))


def _values(group, key):
    if not key or not all(r.get(key) for r in group):
        return None
    return np.array([float(r[key]) for r in group])


def _hollow(axis, h, values, hollow, color, **kwargs) -> None:
    """Redraw the markers selected by ``hollow`` (bool array or None) open."""
    if hollow is not None and hollow.any():
        axis.loglog(h[hollow], values[hollow], "o", color=color, mfc="white", zorder=3, **kwargs)


def _compatible_rows(paths) -> dict:
    """{(p, N_K): row} of the compatible-drift runs (p_phi = p_rho, article dt; ``*_dthalf`` skipped)."""
    rows = _read([path for path in paths if not Path(path).stem.endswith("_dthalf")])
    return {(int(float(r["order"])), int(float(r["triangles"]))): r for r in rows
            if int(float(r["poisson_order"])) == int(float(r["order"]))}


def _style_legend(axis, entries, **kwargs) -> None:
    """Second legend of line styles (label, line style, marker), kept beside the existing one."""
    from matplotlib.lines import Line2D

    first = axis.get_legend()
    handles = [Line2D([], [], color="0.3", ls=ls, marker=marker, lw=0.9, ms=4) for _, ls, marker in entries]
    axis.legend(handles, [label for label, _, _ in entries], fontsize=6.5, **kwargs)
    if first is not None:
        axis.add_artist(first)


def spatial_figure(rows, output: Path, best: list | None = None, regularity: float | None = None,
                   raised: dict | None = None, lattice: dict | None = None, compatible: dict | None = None) -> None:
    """``best``: rows {order, triangles, rho_best_l2, phi_best_l2} of L2 best approximations at the final
    time, drawn dashed in the element panels. ``raised``: {(p, N_K): rho L2 error} with a Poisson solve of
    degree p+1, dashed in the density panel, whose dotted slopes are then p+1. ``lattice``: {N_K: angle
    of the nearest lattice edge family}; meshes with angle 30 (one vertical family) get open markers.
    ``compatible``: {(p, N_K): row} of the compatible drift with p_phi = p_rho, crosses in the density panel."""
    groups = _by_order(rows)
    area = float(rows[0].get("domain_area") or 4.0)
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(groups)))
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.8), constrained_layout=True, sharex=True)
    degree = r"p_\rho" if raised else "p"  # with a raised Poisson degree, p is the density degree

    def vertical(triangles):
        return None if lattice is None else np.array([lattice.get(int(n), 0.0) > 15.0 for n in triangles])

    for (order, group), color in zip(groups.items(), colors):
        h = np.array([_h(r) for r in group])
        hollow = vertical(r["triangles"] for r in group)
        for row, col, solid, dashed, _, offset in PANELS:
            axis = axes[row, col]
            values = _values(group, solid)
            if values is None:
                continue
            axis.loglog(h, values, "o-", color=color, label=f"${degree}={order}$")
            _hollow(axis, h, values, hollow, color)
            slope = order + offset
            if regularity is not None:  # density in H^s, potential in H^(s+2)
                slope = min(slope, regularity + (2.0 if offset == 1.0 else 0.0))
            anchor_h, anchor = h, values
            if raised and solid == "rho_l2_error":
                points = sorted((n, e) for (p, n), e in raised.items() if p == order)
                anchor_h = np.array([np.sqrt(area / n) for n, _ in points])
                anchor = np.array([e for _, e in points])
                axis.loglog(anchor_h, anchor, "o--", color=color, lw=0.9)
                _hollow(axis, anchor_h, anchor, vertical(n for n, _ in points), color)
                slope = order + 1.0
                if compatible:
                    marks = sorted((n, float(r["rho_l2_error"])) for (p, n), r in compatible.items() if p == order)
                    axis.loglog([np.sqrt(area / n) for n, _ in marks], [e for _, e in marks], "x", color=color,
                                ms=6, mew=1.1, zorder=4)
            axis.loglog(h, anchor[-1] * (h / anchor_h[-1]) ** slope, ":", color=color, lw=0.8)
            extra = _values(group, dashed)
            if extra is not None:
                axis.loglog(h, extra, "--", color=color, lw=0.8, alpha=0.7)
        for row, key in ((0, "rho_best_l2"), (1, "phi_best_l2")):
            points = sorted((int(b["triangles"]), float(b[key])) for b in best or () if int(b["order"]) == order)
            if points:
                axes[row, 0].loglog([np.sqrt(area / k) for k, _ in points], [e for _, e in points], "--",
                                    color=color, lw=0.8, alpha=0.7)
    for row, col, solid, _, label, offset in PANELS:
        if raised and solid == "rho_l2_error":
            label, offset = (label if compatible else
                             label + r", $p_\phi=p_\rho$ (solid), $p_\phi=p_\rho+1$ (dashed)"), 1.0
        axes[row, col].set_ylabel(label, fontsize=7 if col or raised else 9)
        title = f"$h^{{{degree}+{'1/2' if offset == 0.5 else '1'}}}$"
        if regularity is not None:
            title = rf"$h^{{\min(p+{'1/2' if offset == 0.5 else '1'},{regularity + (2 if offset == 1.0 else 0):g})}}$"
        axes[row, col].set_title("dotted: " + title, fontsize=8)
    for axis in axes[1, :]:
        axis.set_xlabel(r"$h=(|\Omega|/N_K)^{1/2}$")
    axes[0, 0].legend(ncol=2)
    if raised and compatible:
        _style_legend(axes[0, 0], [(r"raw, $p_\phi=p_\rho$", "-", "o"), (r"raw, $p_\phi=p_\rho+1$", "--", "o"),
                                   (r"compatible, $p_\phi=p_\rho$", "none", "x")], loc="lower right")
    figure.savefig(output)
    plt.close(figure)


def work_precision_figure(rows, output: Path, total: bool = False, compatible: dict | None = None,
                          cost: dict | None = None) -> None:
    """Left: wall time per step (ms) or, with ``total``, per run (s, median step x steps)."""
    groups = _by_order(rows)
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(groups)))
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
    for (order, group), color in zip(groups.items(), colors):
        step = np.array([float(r["steady_time_s"]) if total else 1e3 * float(r["step_time_median_s"])
                         for r in group])
        unknowns = np.array([int(r["element_unknowns"]) for r in group])
        for row_index, key in enumerate(("rho_l2_error", "phi_l2_error")):
            error = np.array([float(r[key]) for r in group])
            axes[row_index, 0].loglog(step, error, "o-", color=color, label=f"$p={order}$")
            axes[row_index, 1].loglog(unknowns, error, "o-", color=color, label=f"$p={order}$")
        if compatible and cost:
            # Compatible drift at the raw run's time times the measured step-time ratio (same mesh, same dt).
            points = [(float(r["steady_time_s"] if total else 1e3 * float(r["step_time_median_s"]))
                       * cost[f"{order},{r['triangles']}"]["compatible/q=p"], compatible[(order, int(r["triangles"]))],
                       int(r["element_unknowns"])) for r in group
                      if (order, int(r["triangles"])) in compatible
                      and "compatible/q=p" in cost.get(f"{order},{r['triangles']}", {})]
            # Density panels only: the potential errors of the two drifts agree within a few percent.
            for row_index, key in enumerate(("rho_l2_error",)):
                if points:
                    error = [float(c[key]) for _, c, _ in points]
                    axes[row_index, 0].loglog([t for t, _, _ in points], error, "x--", color=color, lw=0.9, ms=5)
                    axes[row_index, 1].loglog([n for _, _, n in points], error, "x--", color=color, lw=0.9, ms=5)
    for row_index, label in enumerate((r"$\|\rho_h-\rho\|_{L^2}$", r"$\|\phi_h-\phi\|_{L^2}$")):
        axes[row_index, 0].set_ylabel(label)
    for axis in axes[1, :]:
        axis.set_xlabel("")
    axes[1, 0].set_xlabel("GPU time per run: steps × median step time [s]" if total else "GPU wall time per IMEX-ARK3 step [ms]")
    axes[1, 1].set_xlabel("element unknowns of the density")
    axes[0, 1].legend(ncol=2)
    if compatible and cost:
        _style_legend(axes[0, 0], [(r"raw drift", "-", "o"), (r"compatible drift", "--", "x")], loc="lower left")
    figure.savefig(output)
    plt.close(figure)


def _short_ladder(courant: dict) -> bool:
    return len(courant) > 1 and max(courant) / min(courant) < 100.0


def temporal_figure(rows, failures, output: Path, reference: dict | None = None,
                    regularity: float | None = None, against_reference: bool = False) -> None:
    """Density (left) and potential (right) against dt, one colour per scheme: element unknowns
    solid, their traces dashed (rhohat in the |u.n|-weighted skeleton norm, phihat in ||.||_h).

    Errors against the exact solution; ``reference`` (the ``--reference-dt`` run's row) adds its own error
    against the exact solution as a grey floor, the spatial error of the mesh. ``against_reference``
    plots the differences from that run instead.

    ``regularity`` s: density slopes min(q, q s/(q+1)) (Brenner-Thomee order for data in H^s),
    potential slopes q; otherwise slopes q. Anchored at the smallest dt."""
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True, sharex=True)
    courant = {}
    key = ((lambda k: k.replace("_l2_error", "_ref_l2_error")) if reference is not None and against_reference
           else (lambda k: k))
    for scheme, (label, marker, order) in SCHEMES.items():
        group = sorted((r for r in rows if r["scheme"] == scheme), key=lambda r: float(r["dt"]))
        if not group:
            continue
        dt = np.array([float(r["dt"]) for r in group])
        courant.update({float(r["dt"]): float(r["courant"]) for r in group})
        for index, (axis, (element, trace, *_)) in enumerate(zip(axes, TEMPORAL_PANELS)):
            values = _values(group, key(element))
            if values is None:
                continue
            line, = axis.loglog(dt, values, marker + "-", label=label)
            slope = order if regularity is None or index else min(order, order * regularity / (order + 1))
            axis.loglog(dt, values[0] * (dt / dt[0]) ** slope, "k:", lw=0.7)
            extra = _values(group, key(trace))
            if extra is not None:
                axis.loglog(dt, extra, "--", color=line.get_color(), lw=0.8, alpha=0.8)
    for axis, (element, _, title, element_label, trace_label) in zip(axes, TEMPORAL_PANELS):
        if reference is not None and reference.get(element):
            axis.axhline(float(reference[element]), color="0.5", lw=0.8, ls="-.")
        for _, dt_failed in failures:
            axis.axvline(dt_failed, color="0.6", lw=0.6, ls="--")
        styles = [Line2D([], [], color="k", lw=1.0, label=element_label),
                  Line2D([], [], color="k", lw=0.8, ls="--", label=trace_label)]
        axis.add_artist(axis.legend(handles=styles, title=title, loc="lower right", title_fontsize=7))
        axis.xaxis.set_minor_formatter(NullFormatter())
        if courant:
            ratio = np.mean([c / d for d, c in courant.items()])
            top = axis.secondary_xaxis("top", functions=(lambda d, r=ratio: d * r, lambda c, r=ratio: c / r))
            top.set_xlabel("Courant number $C$", fontsize=8)
            top.xaxis.set_minor_formatter(NullFormatter())
            if _short_ladder(courant):
                steps = sorted(courant)
                top.set_xticks([ratio * d for d in steps], [f"{ratio * d:.2g}" for d in steps])
    xlabel = r"$\Delta t$"
    if _short_ladder(courant):
        # Less than two decades: matplotlib would label the minor ticks, which collide; label the
        # time steps of the ladder instead, scaled by a power of ten (top axis: their Courant numbers).
        steps = sorted(courant)
        exponent = int(np.floor(np.log10(steps[-1])))
        axes[0].set_xticks(steps, [f"{d / 10.0 ** exponent:.2g}" for d in steps])
        xlabel = rf"$10^{{{-exponent}}}\,\Delta t$"
    for axis in axes:
        axis.set_xlabel(xlabel)
    axes[0].set_ylabel("difference from the reference solution" if reference is not None and against_reference
                       else "error")
    axes[0].legend(loc="upper left")
    figure.savefig(output)
    plt.close(figure)


def vortex_setup_figure(output: Path, final_time: float = 0.5) -> None:
    """Exact density of the orbiting vortex at t = 0, T/2, T (no simulation)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from matplotlib.path import Path as PolygonPath

    from hybridge.core.mesh import gmsh_polygon_mesh
    from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key

    case = case_definition_by_key("orbiting_vortex").build()
    parameters = case.parameters
    vertices = np.asarray(parameters["geometry"]["vertices"])
    outline = np.vstack((vertices, vertices[:1]))
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 801), np.linspace(-1.0, 1.0, 801))
    inside = PolygonPath(vertices).contains_points(np.column_stack((x.ravel(), y.ravel())), radius=1e-9)
    mesh = gmsh_polygon_mesh(0.1, vertices=parameters["geometry"]["vertices"], log_cache=False)
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    omega, orbit, radius = 0.5 * parameters["rho0"], parameters["d"], parameters["R"]
    angle = np.linspace(0.0, 2.0 * np.pi, 400)
    for axis, t in zip(axes, (0.0, 0.5 * final_time, final_time)):
        values = np.where(inside.reshape(x.shape), case.exact_density(x, y, t), np.nan)
        image = axis.pcolormesh(x, y, values, shading="auto", cmap="viridis",
                                vmin=parameters["rho0"], vmax=parameters["rho0"] + parameters["A"], rasterized=True)
        if t == 0.0:
            axis.triplot(mesh.node_coords[:, 0], mesh.node_coords[:, 1], mesh.triangles,
                         color="w", lw=0.15, alpha=0.6)
        centre = orbit * np.array((np.cos(parameters["theta0"] + omega * t), np.sin(parameters["theta0"] + omega * t)))
        axis.plot(orbit * np.cos(angle), orbit * np.sin(angle), "w:", lw=0.6)
        axis.plot(centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle), "w--", lw=0.6)
        axis.plot(*outline.T, "k-", lw=0.8)
        axis.plot(0.6, 0.6, "r+", ms=6)
        axis.set_title(f"$t={t:g}$", fontsize=8)
        axis.set_aspect("equal")
        axis.set_xticks([-1, 0, 1])
        axis.set_yticks([-1, 0, 1])
        axis.grid(False)
    figure.colorbar(image, ax=axes, shrink=0.8, label=r"$\rho$")
    figure.savefig(output, dpi=300)
    plt.close(figure)


def wave_figure(snapshot: Path, output: Path, mesh_overlay_max: int = 4000) -> None:
    """Exact density at t=0 and T, numerical rho_h(T) and its error, from a snapshot NPZ.

    ``snapshot_manufactured_solution.py`` writes the NPZ (exact DG raster evaluation).
    The mesh is drawn on the rho_h panel when it has at most ``mesh_overlay_max`` triangles,
    otherwise a zoomed inset shows rho_h and the mesh on a 0.2 x 0.2 window.
    """
    data = np.load(snapshot, allow_pickle=False)
    x0, x1, y0, y1 = (float(v) for v in data["bounds"])
    extent = (x0, x1, y0, y1)
    exact0, exact, numerical = data["rho_exact_initial"], data["rho_exact_final"], data["rho_numerical_final"]
    error = numerical - exact
    limit = float(np.nanmax(np.abs(exact0)))
    # Error colour scale: 99.9th percentile of |error| (extended colourbar), so that a few
    # boundary pixels do not wash out the interior pattern; the maximum goes in the caption.
    elimit = float(np.nanpercentile(np.abs(error), 99.9))
    nodes, cells = data["nodes"], data["cells"]
    t = float(data["time"])
    figure, axes = plt.subplots(2, 2, figsize=(6.4, 5.6), constrained_layout=True)
    panels = ((axes[0, 0], exact0, r"exact $\rho$, $t=0$", "RdBu_r", limit),
              (axes[0, 1], exact, rf"exact $\rho$, $t={t:g}$", "RdBu_r", limit),
              (axes[1, 0], numerical, rf"$\rho_h$, $t={t:g}$", "RdBu_r", limit),
              (axes[1, 1], error, rf"$\rho_h-\rho$, $t={t:g}$", "PuOr_r", elimit))
    for axis, image, title, cmap, vmax in panels:
        handle = axis.imshow(image, extent=extent, origin="upper", cmap=cmap, vmin=-vmax, vmax=vmax,
                             interpolation="nearest", rasterized=True)
        axis.set_title(title, fontsize=8)
        axis.set_aspect("equal")
        axis.set_xticks([-1, 0, 1])
        axis.set_yticks([-1, 0, 1])
        axis.grid(False)
        figure.colorbar(handle, ax=axis, shrink=0.8, extend="both" if cmap == "PuOr_r" else "neither")
    if len(cells) <= mesh_overlay_max:
        axes[1, 0].triplot(nodes[:, 0], nodes[:, 1], cells, color="k", lw=0.12, alpha=0.5)
    else:
        inset = axes[1, 0].inset_axes([0.58, 0.58, 0.4, 0.4])
        inset.imshow(numerical, extent=extent, origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit,
                     interpolation="nearest", rasterized=True)
        inset.triplot(nodes[:, 0], nodes[:, 1], cells, color="k", lw=0.2, alpha=0.6)
        inset.set_xlim(-0.1, 0.1)
        inset.set_ylim(-0.1, 0.1)
        inset.set_xticks([])
        inset.set_yticks([])
    figure.savefig(output, dpi=300)
    plt.close(figure)


def alignment_mesh_figure(output: Path, wave_number: float = 8.0 * np.pi, cells: int = 50,
                          window: tuple[float, float] = (-0.12, 0.12)) -> None:
    """Element-level zoom of the structured mesh (``cells`` x ``cells`` squares of (-1,1)^2 split
    along their (-1,1) diagonal, as ``--mesh-family structured``) for k_x = k_y (aligned) and
    k_y = -k_x (misaligned), at t=0: every edge coloured by the normal drift u.n along it (n fixed
    per edge family, n_x + n_y > 0) with its zeros marked, integral lines of the frozen drift
    u = -grad^perp phi, and the exact density in grey. No simulation."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    from hybridge.core.mesh import rectangle_mesh
    from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key

    mesh = rectangle_mesh(cells, cells, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    nodes, triangles = mesh.node_coords, mesh.triangles
    edges = np.unique(np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    lo, hi = window[0] - 1.0e-9, window[1] + 1.0e-9
    inside = np.all((nodes[edges] >= lo) & (nodes[edges] <= hi), axis=(1, 2))
    start, stop = nodes[edges[inside, 0]], nodes[edges[inside, 1]]
    tangent = stop - start
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0])) / np.hypot(*tangent.T)[:, None]
    normal *= np.where(normal.sum(axis=1) > 0.0, 1.0, -1.0)[:, None]
    pieces = 24
    fractions = np.linspace(0.0, 1.0, pieces + 1)
    ends = start[:, None, :] + fractions[None, :, None] * tangent[:, None, :]
    segments = np.stack((ends[:, :-1], ends[:, 1:]), axis=2).reshape(-1, 2, 2)
    middles = segments.mean(axis=1)
    segment_normals = np.repeat(normal, pieces, axis=0)

    amplitude = 1.0 / (np.sqrt(2.0) * wave_number)
    grid = np.linspace(*window, 401)
    x, y = np.meshgrid(grid, grid)
    norm = Normalize(-1.85, 1.85)
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), constrained_layout=True)
    for axis, sign, title in zip(axes, (1.0, -1.0), (r"aligned: $k_y=k_x$", r"misaligned: $k_y=-k_x$")):
        case = case_definition_by_key("rho_helm_wave").build(U=1.0, kx=wave_number, ky=sign * wave_number,
                                                             A=amplitude)

        def drift(px, py, case=case):
            qx, qy = case.exact_flux(px, py, 0.0)
            return -qy, qx

        limit = amplitude * 2.0 * wave_number ** 2
        axis.pcolormesh(x, y, case.exact_density(x, y, 0.0), shading="auto", cmap="Greys",
                        vmin=-1.2 * limit, vmax=4.0 * limit, rasterized=True)
        ux, uy = drift(middles[:, 0], middles[:, 1])
        edges_drawn = LineCollection(segments, array=ux * segment_normals[:, 0] + uy * segment_normals[:, 1],
                                     cmap="PuOr_r", norm=norm, linewidths=2.4, capstyle="butt")
        axis.add_collection(edges_drawn)
        # Points where the drift is tangent to an edge (u.n changes sign along it).
        ux, uy = drift(ends[..., 0], ends[..., 1])
        flux = ux * normal[:, None, 0] + uy * normal[:, None, 1]
        edge, piece = np.nonzero(flux[:, :-1] * flux[:, 1:] < 0.0)
        weight = flux[edge, piece] / (flux[edge, piece] - flux[edge, piece + 1])
        zeros = ends[edge, piece] + weight[:, None] * (ends[edge, piece + 1] - ends[edge, piece])
        axis.plot(zeros[:, 0], zeros[:, 1], "o", ms=3.2, mfc="w", mec="k", mew=0.7, zorder=4)
        vx, vy = drift(x, y)
        axis.streamplot(grid, grid, vx, vy, density=0.55, color="k", linewidth=0.5, arrowsize=0.6, zorder=3)
        axis.set_xlim(*window)
        axis.set_ylim(*window)
        axis.set_title(title, fontsize=8)
        axis.set_aspect("equal")
        axis.set_xticks([window[0], 0.0, window[1]])
        axis.set_yticks([window[0], 0.0, window[1]])
        axis.grid(False)
    figure.colorbar(edges_drawn, ax=axes, shrink=0.8, label=r"$\mathbf{u}\cdot\mathbf{n}$ on the edges")
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spatial", nargs="*", default=[])
    parser.add_argument("--temporal", nargs="*", default=[])
    parser.add_argument("--temporal-failures", nargs="*", default=[])
    parser.add_argument("--temporal-reference", type=Path, default=None,
                        help="<prefix>_reference.json of a --reference-dt temporal study: its error against the "
                             "exact solution is drawn as the spatial-error floor.")
    parser.add_argument("--temporal-against-reference", action="store_true",
                        help="Plot the differences from the reference run instead of the errors against the exact "
                             "solution.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="", help="Prefix of the output file names.")
    parser.add_argument("--vortex-setup", action="store_true", help="Exact orbiting-vortex snapshots.")
    parser.add_argument("--wave-figure", nargs="*", default=[], type=Path,
                        help="Snapshot NPZ(s) from snapshot_manufactured_solution.py -> <tag>wave_<stem>.pdf.")
    parser.add_argument("--alignment-mesh", action="store_true",
                        help="Aligned/misaligned structured mesh over the exact wave -> <tag>alignment_mesh.pdf.")
    parser.add_argument("--best-approximation", type=Path, default=None,
                        help="JSON rows {order, triangles, rho_best_l2, phi_best_l2}: dashed in the element panels.")
    parser.add_argument("--regularity", type=float, default=None,
                        help="Sobolev index s of the density: temporal density slopes min(q, qs/(q+1)).")
    parser.add_argument("--work-total", action="store_true",
                        help="Work-precision against the time per run (number of steps times the median step time, setup "
                             "excluded) instead of per step.")
    parser.add_argument("--raised", type=Path, default=None,
                        help="poisson_degree.csv: gmsh p_phi=p_rho+1 density errors, dashed in the spatial density panel.")
    parser.add_argument("--lattice", type=Path, default=None,
                        help="gmsh_lattice.json: open markers on meshes with a vertical lattice edge family.")
    parser.add_argument("--compatible", nargs="*", default=[],
                        help="Per-run CSVs of the compatible drift with p_phi=p_rho (*_dthalf skipped): crosses in the "
                             "spatial density panel; with --compatible-cost also in the work-precision figure.")
    parser.add_argument("--compatible-cost", type=Path, default=None,
                        help="ark3_step_times.json (timing_device/ark3_timing_summary.py): measured IMEX-ARK3 step-time "
                             "ratio compatible/raw per (p, N_K) for the work-precision figure.")
    args = parser.parse_args()
    _style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for snapshot in args.wave_figure:
        wave_figure(snapshot, args.output_dir / f"{args.tag}wave_{snapshot.stem}.pdf")
    if args.alignment_mesh:
        alignment_mesh_figure(args.output_dir / f"{args.tag}alignment_mesh.pdf")
    if args.vortex_setup:
        vortex_setup_figure(args.output_dir / "vortex_setup.pdf")
    if args.spatial:
        rows = _read(args.spatial)
        import json
        best = json.loads(args.best_approximation.read_text()) if args.best_approximation else None
        raised = lattice = None
        if args.raised or args.lattice:
            if __package__ in {None, ""}:
                sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from scripts.guiding_center.studies.convergence.make_convergence_tables import lattice_angles, raised_density_errors
            raised = raised_density_errors(_read([args.raised])) if args.raised else None
            lattice = lattice_angles(args.lattice) if args.lattice else None
        compatible = _compatible_rows(args.compatible) if args.compatible else None
        cost = json.loads(args.compatible_cost.read_text()) if args.compatible_cost else None
        spatial_figure(rows, args.output_dir / f"{args.tag}spatial_convergence.pdf", best, args.regularity,
                       raised, lattice, compatible)
        work_precision_figure(rows, args.output_dir / f"{args.tag}work_precision.pdf", args.work_total,
                              compatible, cost)
    if args.temporal:
        import json
        failures = [(f["scheme"], float(f["dt"])) for path in args.temporal_failures
                    for f in json.loads(Path(path).read_text())]
        reference = json.loads(args.temporal_reference.read_text()) if args.temporal_reference else None
        temporal_figure(_read(args.temporal), failures, args.output_dir / f"{args.tag}temporal_convergence.pdf",
                        reference, args.regularity, args.temporal_against_reference)
    print(f"figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
