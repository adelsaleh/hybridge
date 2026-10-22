"""Article figures of the D7 collisional vane-magnetron discharge (Section 11); no time integration.

Reads a ``run_magnetron.py`` record (JSONL + snapshot NPZ) of the collisional parallel-vane case:

``<stem>_overview``
    density rho_h at the snapshot times with thin equipotentials of phi_h (the E x B drift follows
    them; ``--no-equipotentials`` drops them), and charge M(t) and anode current I(t) below, with
    the emission pulses shaded.
``<stem>_zoom`` (``--zoom``)
    close-ups with the mesh drawn: the cathode layer one time unit after a pulse switches off
    (``--cathode-time``), and the slot entrance where the most charge enters at ``--tip-time``.
``<stem>_supg`` (``--supg-dir``)
    HDG and CG-SUPG (DOLFINx, ``projects/diocotron/comparisons/discharge_supg.py``) densities at
    the SUPG run's final time on the same mesh, the negative part of the SUPG density, and the
    minimum density and anode current of both runs in time.

    python -m scripts.guiding_center.diagnostics.plot_discharge --run RUN.jsonl --stem OUT [--zoom] \\
        [--supg-dir DIR --supg-run HDG_ON_THE_SUPG_MESH.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hybridge.io.figures import (add_matplotlib_mesh, draw_signed_density, publication_style,  # noqa: E402
                                 save_publication_figure)
from scripts.guiding_center.diagnostics.magnetron_convergence import _snapshot_path  # noqa: E402
from scripts.guiding_center.diagnostics.plot_diocotron_highmode import _records, _series  # noqa: E402

INK, MUTED, PULSE = "#0b0b0b", "#52514e", "#e8e3d3"


def _degree(basis_size: int) -> int:
    """Degree q of a 2D DG space with (q+1)(q+2)/2 basis functions."""
    return int(round((math.sqrt(8 * basis_size + 1) - 3) / 2))


class _Snapshots:
    """Snapshot fields of one run, evaluated on raster grids without a sampling matrix."""

    def __init__(self, jsonl: Path):
        from hybridge.core.mesh import DGMesh
        from hybridge.core.space import DGSpace

        self.npz = np.load(_snapshot_path(jsonl))
        self.mesh = DGMesh(self.npz["node_coords"], self.npz["triangles"])
        basis = str(self.npz["basis"])
        self.density_space = DGSpace(self.mesh, int(self.npz["order"]), basis_type=basis)
        self.potential_space = DGSpace(self.mesh, _degree(self.npz["potential"].shape[-1]), basis_type=basis)

    def frame(self, time: float) -> int:
        k = int(np.argmin(np.abs(self.npz["times"] - time)))
        if abs(self.npz["times"][k] - time) > 1e-6:
            raise ValueError(f"no snapshot at t = {time:g} (have {list(self.npz['times'])})")
        return k

    def grid(self, pixels: int, window=None):
        from hybridge.io.raster import RasterGeometry

        if window is None:
            return RasterGeometry.from_mesh(self.mesh, pixels, pixels)
        x0, x1, y0, y1 = window
        height = max(2, int(round(pixels * (y1 - y0) / (x1 - x0))))
        x = x0 + (np.arange(pixels) + 0.5) * (x1 - x0) / pixels
        y = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height
        xx, yy = np.meshgrid(x, y)
        return RasterGeometry.from_points(self.mesh, np.column_stack((xx.ravel(), yy.ravel())),
                                          width=pixels, height=height, bounds=window)

    def density(self, geometry, time):
        return geometry.evaluate(self.density_space, self.npz["density"][self.frame(time)])

    def potential(self, geometry, time):
        return geometry.evaluate(self.potential_space, self.npz["potential"][self.frame(time)])


def _blank(ax):
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _pulse(parameters):
    from scripts.guiding_center.cases.guiding_center_cases import pulse_train

    if parameters.get("emission_period") is None:
        return None
    train = pulse_train(parameters["emission_period"], parameters.get("emission_duty", 0.5),
                        parameters.get("emission_ramp"))
    phase = float(parameters.get("emission_phase", 0.0))
    return lambda t: train(t + phase)


def _histories(jsonl: Path):
    records = _records(jsonl)
    configuration = next(row["run_configuration"] for row in records if row.get("run_configuration"))
    rows = [row for row in records if row.get("step", 0) > 0 and "mass" in row]
    return configuration, {key: _series(rows, column) for key, column in
                           (("t", "time"), ("charge", "mass"), ("anode", "wall_outflux_outer"),
                            ("rho_min", "rho_min"))}


def _shade_pulses(ax, pulse, t_end):
    if pulse is None:
        return
    t = np.linspace(0.0, t_end, 4001)
    on = np.array([pulse(value) for value in t]) > 0.5
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = ([0.0] if on[0] else []) + [t[i + 1] for i in edges if not on[i]]
    stops = [t[i + 1] for i in edges if on[i]] + ([t_end] if on[-1] else [])
    for start, stop in zip(starts, stops):
        ax.axvspan(start, stop, color=PULSE, lw=0, zorder=0)


def overview_figure(jsonl: Path, stem: Path, *, times, pixels=900, equipotentials=12):
    snapshots = _Snapshots(jsonl)
    configuration, history = _histories(jsonl)
    pulse = _pulse(configuration["case_parameters"])
    geometry = snapshots.grid(pixels)
    x, y = geometry.pixel_centres()
    with publication_style():
        figure = plt.figure(figsize=(1.6 * len(times) + 0.9, 3.9), layout="constrained")
        top, bottom = figure.subfigures(2, 1, height_ratios=(1.0, 0.85))
        density_axes = top.subplots(1, len(times))
        panels = []
        for ax, time in zip(density_axes, times):
            _blank(ax)
            panels.append((ax, np.ma.masked_invalid(snapshots.density(geometry, time)), geometry.bounds))
            ax.set_title(rf"$t={time:g}$")
        artist = draw_signed_density(panels, maximum=1.0, over="#fde725")
        if equipotentials:
            for ax, time in zip(density_axes, times):
                potential = snapshots.potential(geometry, time)
                ax.contour(x, y, np.ma.masked_invalid(potential), levels=int(equipotentials), colors="white",
                           linewidths=0.3, alpha=0.5, linestyles="solid")
                ax.set_xlim(geometry.bounds[0], geometry.bounds[1]), ax.set_ylim(geometry.bounds[2], geometry.bounds[3])
        top.colorbar(artist, ax=density_axes, shrink=0.85, label=r"$\rho_h$")
        charge_ax, current_ax = bottom.subplots(1, 2)
        t_end = float(history["t"][-1])
        for ax, key, label in ((charge_ax, "charge", r"charge $M(t)$"), (current_ax, "anode", r"anode current $I(t)$")):
            _shade_pulses(ax, pulse, t_end)
            ax.plot(history["t"], history[key], color=INK, lw=0.9)
            ax.set(xlabel=r"$t$", ylabel=label, xlim=(0.0, t_end))
            for time in times:
                ax.axvline(time, color=MUTED, lw=0.5, ls=":")
        save_publication_figure(figure, stem.with_name(stem.name + "_overview"))
        plt.close(figure)


def _cathode_window(radius, half_angle=0.08, depth=0.09):
    return (radius - 0.01, radius + depth, -radius * half_angle, radius * half_angle)


def _busiest_slot(snapshots, time, parameters, pixels=500):
    """Window around the slot entrance (r = anode, mid-slot angle) holding the most charge at ``time``."""
    geometry = snapshots.grid(pixels)
    density = snapshots.density(geometry, time)
    x, y = geometry.pixel_centres()
    geometry_record = parameters["geometry"]
    count, anode = int(geometry_record["vanes"]), float(geometry_record["anode_radius"])
    best, best_angle = -1.0, 0.0
    for k in range(count):
        angle = (k + 0.5) * 2.0 * math.pi / count
        near = np.hypot(x - anode * math.cos(angle), y - anode * math.sin(angle)) < 0.12
        charge = float(np.nansum(np.where(near, density, 0.0)))
        if charge > best:
            best, best_angle = charge, angle
    cx, cy = anode * math.cos(best_angle), anode * math.sin(best_angle)
    return (cx - 0.2, cx + 0.2, cy - 0.2, cy + 0.2)


def zoom_figure(jsonl: Path, stem: Path, *, cathode_time, tip_time, pixels=700):
    snapshots = _Snapshots(jsonl)
    configuration, _ = _histories(jsonl)
    parameters = configuration["case_parameters"]
    cathode = float(parameters["geometry"]["inner_radius"])
    windows = [(_cathode_window(cathode), cathode_time, rf"cathode layer, $t={cathode_time:g}$"),
               (_busiest_slot(snapshots, tip_time, parameters), tip_time, rf"slot entrance, $t={tip_time:g}$")]
    with publication_style():
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), layout="constrained")
        panels = []
        for ax, (window, time, title) in zip(axes, windows):
            geometry = snapshots.grid(pixels, window)
            panels.append((ax, np.ma.masked_invalid(snapshots.density(geometry, time)), window))
            ax.set_title(title)
        artist = draw_signed_density(panels, maximum=1.0, over="#fde725")
        for ax, (window, _, _) in zip(axes, windows):
            add_matplotlib_mesh(ax, snapshots.mesh, color="white", linewidth=0.25, alpha=0.6, bounds=window)
            ax.set_xlim(window[0], window[1]), ax.set_ylim(window[2], window[3])
            ax.set_aspect("equal"), ax.tick_params(labelsize=7)
        figure.colorbar(artist, ax=axes, shrink=0.85, label=r"$\rho_h$")
        save_publication_figure(figure, stem.with_name(stem.name + "_zoom"))
        plt.close(figure)


def _supg_histories(directory: Path):
    with (directory / "discharge_supg.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[column]) for row in rows]) for key, column in
            (("t", "time"), ("anode", "outflux_anode"), ("rho_min", "rho_min"))}


def supg_figure(hdg_jsonl: Path, supg_dir: Path, stem: Path, *, pixels=900):
    from scipy.interpolate import LinearNDInterpolator

    snapshots = _Snapshots(hdg_jsonl)
    supg_history = _supg_histories(supg_dir)
    time = float(supg_history["t"][-1])
    geometry = snapshots.grid(pixels)
    hdg = snapshots.density(geometry, time)
    supg_nodes = np.load(supg_dir / "final_density.npz")
    x, y = geometry.pixel_centres()
    inside = np.isfinite(hdg)
    supg = np.full(hdg.shape, np.nan)
    supg[inside] = LinearNDInterpolator(supg_nodes["coordinates"], supg_nodes["density"])(x[inside], y[inside])
    negative = np.where(supg < 0.0, np.log10(np.maximum(-supg, 1e-12)), np.nan)
    _, hdg_history = _histories(hdg_jsonl)
    with publication_style():
        figure = plt.figure(figsize=(7.4, 5.0), layout="constrained")
        top, bottom = figure.subfigures(2, 1, height_ratios=(1.0, 0.8))
        axes = top.subplots(1, 3)
        for ax, title in zip(axes, (r"HDG $\rho_h$", r"CG-SUPG $\rho_h$", r"CG-SUPG: $\log_{10}(-\rho_h)$ where $\rho_h<0$")):
            _blank(ax), ax.set_title(title, fontsize=8)
        artist = draw_signed_density([(axes[0], np.ma.masked_invalid(hdg), geometry.bounds),
                                      (axes[1], np.ma.masked_invalid(supg), geometry.bounds)], maximum=1.0,
                                     over="#fde725")
        top.colorbar(artist, ax=list(axes[:2]), shrink=0.8, label=rf"$\rho_h$, $t={time:g}$")
        axes[2].imshow(np.ma.masked_where(~inside, np.ones_like(hdg)), extent=geometry.bounds, origin="upper",
                       cmap="Greys", vmin=0.0, vmax=6.0, interpolation="nearest")  # the domain, light grey
        image = axes[2].imshow(np.ma.masked_invalid(negative), extent=geometry.bounds, origin="upper",
                               cmap="magma", vmin=-6, vmax=-2, interpolation="nearest")
        top.colorbar(image, ax=axes[2], shrink=0.8, label=r"$\log_{10}(-\rho_h)$")
        minimum_ax, current_ax = bottom.subplots(1, 2)
        for history, label, style in ((hdg_history, "HDG", "-"), (supg_history, "CG-SUPG", "--")):
            minimum_ax.semilogy(history["t"], np.maximum(-history["rho_min"], 1e-18), style, color=INK, lw=0.9,
                                label=label)
            current_ax.plot(history["t"], history["anode"], style, color=INK, lw=0.8, label=label)
        minimum_ax.set(xlabel=r"$t$", ylabel=r"$\max(-\min\rho_h,\,10^{-18})$", xlim=(0.0, time))
        current_ax.set(xlabel=r"$t$", ylabel=r"anode current", xlim=(0.0, time))
        minimum_ax.legend(frameon=False), current_ax.legend(frameon=False)
        save_publication_figure(figure, stem.with_name(stem.name + "_supg"))
        plt.close(figure)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="run JSONL with snapshots (overview and zoom)")
    parser.add_argument("--stem", type=Path, required=True)
    parser.add_argument("--times", type=_floats, default=(10., 20., 30., 40.))
    parser.add_argument("--pixels", type=int, default=900)
    parser.add_argument("--no-equipotentials", action="store_true")
    parser.add_argument("--zoom", action="store_true")
    parser.add_argument("--cathode-time", type=float, default=10.0)
    parser.add_argument("--tip-time", type=float, default=30.0)
    parser.add_argument("--supg-dir", type=Path, default=None)
    parser.add_argument("--supg-run", type=Path, default=None, help="HDG run on the SUPG mesh (default: --run)")
    args = parser.parse_args()
    overview_figure(args.run, args.stem, times=args.times, pixels=args.pixels,
                    equipotentials=0 if args.no_equipotentials else 12)
    if args.zoom:
        zoom_figure(args.run, args.stem, cathode_time=args.cathode_time, tip_time=args.tip_time)
    if args.supg_dir is not None:
        supg_figure(args.supg_run or args.run, args.supg_dir, args.stem, pixels=args.pixels)


if __name__ == "__main__":
    main()
