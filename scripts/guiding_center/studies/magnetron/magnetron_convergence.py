"""Convergence of the vane-magnetron runs across mesh and time-step ladders; no time integration.

Reads ``run_magnetron.py`` records (JSONL + snapshot NPZ) of runs that differ only in
the mesh (``--band-mesh-size``/``--corner-mesh-size``) or only in the time step.
Every snapshot density is sampled on one common pixel grid (the meshes share the
domain's bounding box; pixels outside any mesh are dropped), so runs on different
meshes are compared without interpolating between meshes.

``<stem>_<label>_fields``: density at the requested times, one row per run.
``<stem>_<label>_errors``: L2 distance of each run to the finest (pixel quadrature)
    and between successive runs, against the refinement parameter, at each time;
    the RF-period-averaged anode current and the charge M(t) of every run.
``<stem>_<label>.json``: the numbers, with observed orders between successive levels.

    python -m scripts.guiding_center.studies.magnetron.magnetron_convergence \\
        --runs A.jsonl B.jsonl C.jsonl --parameter 0.015,0.0106,0.0075 --label space_kkt --stem OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hybridge.io.figures import publication_style, save_publication_figure  # noqa: E402
from scripts.guiding_center.diagnostics.run_records import (  # noqa: E402
    image as _image, records as _records, series as _series)


def _snapshot_path(jsonl: Path) -> Path:
    return jsonl.with_name(jsonl.name.replace(".jsonl", "_snapshots.npz"))


def _sampler(npz, pixels: int, bounds):
    """Sampling matrix of one run's mesh onto the common pixel grid, and its coverage mask."""
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from hybridge.io.raster import RasterGeometry

    mesh = DGMesh(npz["node_coords"], npz["triangles"])
    space = DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]))
    geometry = RasterGeometry.from_mesh(mesh, pixels, pixels)
    if not np.allclose(geometry.bounds, bounds, atol=1e-9):
        raise ValueError("runs must share the domain bounding box")
    outside = (geometry.element_ids < 0).reshape(geometry.height, geometry.width)
    return geometry, geometry.sampling_matrix(space), outside


def _rf_average(t, current, parameters):
    period = 2.0 * np.pi / (parameters["wave_mode"] * parameters["wave_speed"])
    width = max(1, int(round(period / float(np.median(np.diff(t))))))
    averaged = np.convolve(np.nan_to_num(current), np.ones(width) / width, mode="same")
    averaged[:width // 2] = averaged[len(averaged) - width // 2:] = np.nan  # incomplete windows
    return averaged


def analyse(runs, parameter, label, stem, *, times=(20., 40., 60., 100., 150.), pixels=800):
    runs = [Path(path) for path in runs]
    records = [[row for row in _records(path) if "mass" in row] for path in runs]
    configuration = next(row["run_configuration"] for row in _records(runs[0]) if row.get("run_configuration"))
    parameters = configuration["case_parameters"]
    snapshots = [np.load(_snapshot_path(path)) for path in runs]
    bounds = None
    samples, masks, geometry = [], [], None
    for npz in snapshots:
        if bounds is None:
            from hybridge.core.mesh import DGMesh
            from hybridge.io.raster import RasterGeometry
            geometry = RasterGeometry.from_mesh(DGMesh(npz["node_coords"], npz["triangles"]), pixels, pixels)
            bounds = geometry.bounds
        _, matrix, outside = _sampler(npz, pixels, bounds)
        fields = {}
        for time in times:
            k = int(np.argmin(np.abs(npz["times"] - time)))
            if abs(npz["times"][k] - time) < 1e-6 + 0.5 * float(np.min(np.diff(npz["times"])) if len(npz["times"]) > 1 else 1):
                fields[time] = (matrix @ npz["density"][k].ravel()).reshape(outside.shape)
        samples.append(fields)
        masks.append(outside)
    common = ~np.logical_or.reduce(masks)
    x0, x1, y0, y1 = bounds
    area = (x1 - x0) * (y1 - y0) / common.size

    def distance(a, b):
        return float(np.sqrt(np.sum((a - b)[common] ** 2) * area))

    result = dict(label=label, parameter=list(parameter), runs=[str(path) for path in runs], times=list(times),
                  to_finest={}, successive={}, orders={}, norms={})
    for time in times:
        if not all(time in fields for fields in samples):
            continue
        finest = samples[-1][time]
        result["norms"][time] = distance(finest, 0.0 * finest)
        result["to_finest"][time] = [distance(fields[time], finest) for fields in samples[:-1]]
        steps = [distance(samples[i][time], samples[i + 1][time]) for i in range(len(samples) - 1)]
        result["successive"][time] = steps
        ratios = [parameter[i] / parameter[i + 1] for i in range(len(parameter) - 1)]
        result["orders"][time] = [float(np.log(steps[i] / steps[i + 1]) / np.log(ratios[i + 1]))
                                  if steps[i + 1] > 0 else None for i in range(len(steps) - 1)]
    # integral quantities
    result["current"] = []
    result["charge"] = []
    for path, rows in zip(runs, records):
        rows = [row for row in rows if row.get("step", 0) > 0]
        t = _series(rows, "time")
        averaged = _rf_average(t, _series(rows, "wall_outflux_outer"), parameters)
        late = (t >= 100.0) & (t <= t[-1] - 2.0)
        result["current"].append(dict(late_mean=float(np.nanmean(averaged[late])) if late.any() else None))
        result["charge"].append(dict(final=float(_series(rows, "mass")[-1]),
                                     rho_min=float(np.nanmin(_series(rows, "rho_min"))),
                                     balance=float(np.nanmax(np.abs(_series(rows, "charge_balance_relative_defect"))))))
    Path(f"{stem}_{label}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")

    with publication_style():
        figure, axes = plt.subplots(len(runs), len(times), figsize=(1.3 * len(times) + .8, 1.3 * len(runs) + .3),
                                    layout="constrained", squeeze=False)
        artist = None
        for row, (fields, value) in enumerate(zip(samples, parameter)):
            for column, time in enumerate(times):
                axis = axes[row, column]
                if time in fields:
                    artist = _image(axis, np.where(common, fields[time], np.nan), geometry, ~common)
                else:
                    axis.axis("off")
                if row == 0:
                    axis.set_title(rf"$t={time:g}$")
            axes[row, 0].set_ylabel(f"{value:g}", fontsize=7)
        figure.colorbar(artist, ax=axes, shrink=.6, extend="both", label=r"$\rho_h$")
        save_publication_figure(figure, Path(f"{stem}_{label}_fields"))
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), layout="constrained")
        colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(times)))
        for color, time in zip(colors, times):
            if time not in result["to_finest"]:
                continue
            norm = max(result["norms"][time], 1e-300)
            axes[0].loglog(parameter[:-1], np.array(result["to_finest"][time]) / norm, "o-", color=color, ms=3,
                           lw=.9, label=rf"$t={time:g}$")
            axes[0].loglog(parameter[1:], np.array(result["successive"][time]) / norm, "s--", color=color, ms=2.5,
                           lw=.6)
        axes[0].set(xlabel="refinement parameter", ylabel=r"$\|\rho-\rho_{\rm finest}\|/\|\rho_{\rm finest}\|$")
        axes[0].set_xticks(list(parameter), [f"{value:g}" for value in parameter], minor=False)
        axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[0].legend(frameon=False, fontsize=6)
        lines = plt.cm.magma(np.linspace(0.1, 0.8, len(runs)))
        for color, rows, value in zip(lines, records, parameter):
            rows = [row for row in rows if row.get("step", 0) > 0]
            t = _series(rows, "time")
            axes[1].plot(t, _rf_average(t, _series(rows, "wall_outflux_outer"), parameters), color=color, lw=.9,
                         label=f"{value:g}")
            axes[2].plot(t, _series(rows, "mass"), color=color, lw=.9)
        axes[1].set(xlabel=r"$t$", ylabel=r"RF-averaged anode current")
        axes[1].legend(frameon=False, fontsize=6)
        axes[2].set(xlabel=r"$t$", ylabel=r"charge $M(t)$")
        save_publication_figure(figure, Path(f"{stem}_{label}_errors"))
        plt.close(figure)
    return result


def side_by_side(left: Path, right: Path, output: Path) -> None:
    """Stack the density halves of two run movies (e.g. with and without KKT) into one MP4."""
    import subprocess

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filters = ("[0:v]crop=iw/2:ih:0:0[a];[1:v]crop=iw/2:ih:0:0[b];[a][b]hstack=inputs=2,"
               "pad=ceil(iw/2)*2:ceil(ih/2)*2[v]")
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(left), "-i", str(right),
                    "-filter_complex", filters, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(output)], check=True)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="coarsest to finest")
    parser.add_argument("--parameter", type=_floats, required=True, help="mesh size or time step per run")
    parser.add_argument("--label", required=True)
    parser.add_argument("--stem", type=Path, required=True)
    parser.add_argument("--times", type=_floats, default=(20., 40., 60., 100., 150.))
    parser.add_argument("--pixels", type=int, default=800)
    args = parser.parse_args()
    if len(args.runs) != len(args.parameter):
        parser.error("one parameter value per run")
    result = analyse(args.runs, args.parameter, args.label, args.stem, times=args.times, pixels=args.pixels)
    for time, orders in result["orders"].items():
        print(f"t={time:g}: to finest {['%.2e' % e for e in result['to_finest'][time]]}, "
              f"successive {['%.2e' % e for e in result['successive'][time]]}, orders {orders}")
    print("late RF-averaged current:", [c["late_mean"] for c in result["current"]])


if __name__ == "__main__":
    main()
