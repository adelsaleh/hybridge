"""Article figures for the guiding-center magnetron runs; no time integration.

Reads the run records of ``run_magnetron.py``:

``<stem>_current`` (``--runs``)
    anode current I(t) per run; the late-time mean current against the wave
    speed W with the synchronism window of :mod:`magnetron_reference` (shaded:
    a contour of the wave frame joins the hub top to the anode) and the
    sharp-edge Kelvin-wave speed; the relative defect of the discrete charge
    balance M^{n+1} - H^n - s(int S_h + int alpha_h rho - I).
``<stem>_frames`` (``--snapshots``)
    density snapshots, one row per run, from the saved NPZ fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hybridge.io.figures import publication_style, save_publication_figure  # noqa: E402
from scripts.guiding_center.diagnostics.run_records import (  # noqa: E402
    image as _image, raster as _raster, records as _records, series as _series)

INK, MUTED = "#0b0b0b", "#52514e"


def _parameters(records) -> dict:
    for row in records:
        configuration = row.get("run_configuration")
        if configuration:
            return dict(configuration["case_parameters"])
    raise ValueError("run records carry no run_configuration")


def _label(parameters) -> str:
    if not parameters.get("wave_amplitude"):
        return "no wave"
    return rf"$W={parameters['wave_speed']:g}$"


def current_figure(paths, stem, *, late_fraction=1.0 / 3.0):
    from scripts.guiding_center.diagnostics.magnetron_reference import HubEquilibrium

    runs = []
    for path in paths:
        records = [row for row in _records(path) if "mass" in row]
        runs.append((_parameters(records), records))
    runs.sort(key=lambda run: run[0].get("wave_speed", 0.0) if run[0].get("wave_amplitude") else -1.0)
    forced = [run for run in runs if run[0].get("wave_amplitude")]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, max(len(forced), 1)))
    with publication_style():
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), layout="constrained")
        speeds, means, spreads = [], [], []
        index = 0
        for parameters, records in runs:
            t, current = _series(records, "time"), _series(records, "wall_outflux_outer")
            if parameters.get("wave_amplitude"):
                color = colors[index]
                index += 1
                # A standing (pi-mode) drive collects in pulses: average over one RF period.
                period = 2.0 * np.pi / (parameters["wave_mode"] * parameters["wave_speed"])
                width = max(1, int(round(period / float(np.median(np.diff(t))))))
                averaged = np.convolve(np.nan_to_num(current), np.ones(width) / width, mode="same")
                late = t >= t[-1] * (1.0 - late_fraction)
                speeds.append(parameters["wave_speed"])
                means.append(np.nanmean(current[late]))
                spreads.append(np.nanstd(averaged[late]))
                if parameters.get("anode_geometry") == "vanes":
                    axes[0].plot(t, current, color=color, lw=.3, alpha=.25)
                    current = averaged
            else:
                color = MUTED
            axes[0].plot(t, current, color=color, lw=.9, label=_label(parameters))
            defect = np.abs(_series(records, "charge_balance_relative_defect"))
            axes[2].semilogy(t, np.where(defect > 0, defect, np.nan), color=color, lw=.6)
        axes[0].set(xlabel=r"$t$", ylabel=r"anode current $I$ (RF-period mean for vanes)")
        axes[0].legend(frameon=False, fontsize=6, ncol=2)
        if forced:
            hub = HubEquilibrium(forced[0][0])
            if forced[0][0].get("anode_geometry", "circle") == "circle":  # the window assumes a circular anode
                window = hub.synchronism_window(forced[0][0]["wave_amplitude"])
                if window is not None:
                    axes[1].axvspan(*window, color="#d9e6f2", lw=0, label="channel opens")
            axes[1].axvline(hub.edge_wave_speed(), color=MUTED, ls="--", lw=.8, label="hub-edge wave")
            axes[1].errorbar(speeds, means, yerr=spreads, fmt="o", ms=4, color=INK, capsize=2, lw=.8,
                             label=f"late mean (last {late_fraction:.0%})")
            axes[1].legend(frameon=False, fontsize=6, loc="upper left")
            axes[1].margins(y=.25)
        axes[1].set(xlabel=r"wave speed $W$", ylabel=r"$\langle I\rangle$")
        axes[2].set(xlabel=r"$t$", ylabel="charge balance defect (rel.)")
        save_publication_figure(figure, stem)
        plt.close(figure)


def frames_figure(npz_paths, stem, *, times, labels=None, pixels=420, vmax=1.0):
    snapshots = [np.load(path) for path in npz_paths]
    geometry, matrix, mask = _raster(snapshots[0], pixels)
    labels = labels or [Path(path).stem for path in npz_paths]
    with publication_style():
        figure, axes = plt.subplots(len(snapshots), len(times), figsize=(1.25 * len(times) + .9,
                                                                         1.25 * len(snapshots) + .3),
                                    layout="constrained", squeeze=False)
        artist = None
        for row, (npz, label) in enumerate(zip(snapshots, labels)):
            for column, time in enumerate(times):
                k = int(np.argmin(np.abs(npz["times"] - time)))
                values = (matrix @ npz["density"][k].ravel()).reshape(mask.shape)
                artist = _image(axes[row, column], values, geometry, mask, vmax=vmax)
                if row == 0:
                    axes[row, column].set_title(rf"$t={npz['times'][k]:.0f}$")
            axes[row, 0].set_ylabel(label, fontsize=7)
        figure.colorbar(artist, ax=axes, shrink=.6, extend="both", label=r"$\rho_h$")
        save_publication_figure(figure, stem)
        plt.close(figure)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, nargs="*", default=())
    parser.add_argument("--snapshots", type=Path, nargs="*", default=())
    parser.add_argument("--labels", default=None, help="comma-separated row labels of --snapshots")
    parser.add_argument("--times", type=_floats, default=(20., 60., 100., 200., 300.))
    parser.add_argument("--stem", type=Path, required=True)
    args = parser.parse_args()
    if args.runs:
        current_figure(args.runs, Path(f"{args.stem}_current"))
    if args.snapshots:
        labels = None if args.labels is None else args.labels.split(",")
        frames_figure(args.snapshots, Path(f"{args.stem}_frames"), times=args.times, labels=labels)


if __name__ == "__main__":
    main()
