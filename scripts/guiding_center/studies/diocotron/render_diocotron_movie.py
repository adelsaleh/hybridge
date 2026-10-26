"""Supplementary movies of diocotron runs, rendered from their saved snapshots; no time integration.

Draws the density rho_h of one run, or of several runs side by side, from the ``*_snapshots.npz``
files of ``run_diocotron_benchmark.py --snapshot-times`` and encodes an H.264 MP4 (yuv420p):

* the full mesh or a window ``--sector x0,x1,y0,y1``, with square pixels;
* each pixel is the mean of ``--supersample`` x ``--supersample`` samples of the DG density, as in the
  article figures (``plot_diocotron_highmode.py --supersample``), through one sparse map per mesh
  (:func:`hybridge.io.raster.area_sampling_matrix`) applied to every saved state;
* one colour scale for the whole movie (and for the movies named by ``--share-scale``):
  :func:`hybridge.io.figures.signed_density_colors` from the most negative pixel of all frames up to
  the profile maximum ``--density-max``; the colour bar shows the negative range;
* the time of each saved state and a timeline of the saved states.

A snapshot file holds the states saved at ``--snapshot-times``, not every step. Each state is one
image, shown for ``--duration``/(number of states) seconds at ``--fps`` frames per second (the
frames repeat). ``--initial-state`` prepends t = 0 when the file starts later: the initial density
projected as in the run (host projection, no solve; ``plot_diocotron_zg._initial_density``).
Densities are read one state at a time (:func:`hybridge.io.records.iter_npz_slices`), never the
potentials or fluxes. Host only: the GPU is not used.

The rasters of a movie are cached in ``<work-dir>/<movie stem>/`` (float32, NaN outside the mesh), so
that ``--sample-only`` can precede the rendering, movies with ``--share-scale`` find each other's
pixels, and a rerun only redraws. A finished movie is skipped unless ``--force``; it is written to a
``.partial.mp4`` file first and renamed when complete. ``--preview i,j`` writes the frames of saved
states i, j (indices into the states, negative from the end) as PNG files in the work directory
instead of a movie.

Example (Zoni--Guclu ring, t = 0 rebuilt, 8 states of 4 s):
    PYTHONPATH=. .venv/bin/python -m scripts.guiding_center.studies.diocotron.render_diocotron_movie \\
        run_outputs/article/diocotron/d3_..._raw_seed_snapshots.npz --initial-state --duration 32 \\
        --title "Zoni-Guclu truncated ring, \\$m=9\\$" --output movies/zg_m9.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hybridge.io.figures import publication_style, signed_density_colors  # noqa: E402

INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#d3d1cb"
DPI = 100  # frame pixels per inch: one point is 100/72 pixels
# Frame layout in pixels: title rows, panel-label row, timeline row, gaps and colour-bar column.
TOP, LABEL_ROW, BOTTOM = 118, 50, 112
GAP, BAR_GAP, BAR, BAR_TEXT, MARGIN = 36, 34, 26, 132, 30


def _log(message: str) -> None:
    print(f"[movie] {datetime.now():%H:%M:%S} {message}", flush=True)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value)


def _size(text: str) -> tuple[int, int]:
    width, height = (int(value) for value in text.lower().split("x"))
    if width < 320 or height < 240 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError("frame size must be even and at least 320x240, e.g. 1440x1080")
    return width, height


def _plain(text: str) -> str:
    """Mathtext label as plain text (MP4 title tag)."""
    for tex, plain in ((r"\leq", "≤"), (r"\geq", "≥"), (r"\Delta", "Δ"), (r"\rho", "ρ"), (r"\phi", "φ"),
                       (r"\,", " "), ("$", "")):
        text = text.replace(tex, plain)
    return " ".join(text.split())


def _space(npz):
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace

    return DGSpace(DGMesh(npz["node_coords"], npz["triangles"]), int(npz["order"]), basis_type=str(npz["basis"]))


def _run_configuration(source: Path):
    """``run_configuration`` of the first JSONL record of the run that saved ``source`` (first line only)."""
    jsonl = source.with_name(source.name[:-len("_snapshots.npz")] + ".jsonl")
    if not source.name.endswith("_snapshots.npz") or not jsonl.exists():
        raise FileNotFoundError(f"no run record next to {source} (needed by --initial-state)")
    with jsonl.open() as stream:
        return json.loads(stream.readline())["run_configuration"]


def _mesh_key(npz) -> str:
    digest = hashlib.sha1()
    for key in ("node_coords", "triangles", "order", "basis"):
        digest.update(np.ascontiguousarray(npz[key]).tobytes())
    return digest.hexdigest()


def frame_layout(size, panels: int, aspect: float, *, labelled: bool):
    """Pixel boxes ``(left, top, width, height)`` (from the top-left corner) of the panels and colour bar.

    The panels are as large as the frame allows below the title rows (and the panel labels) and above
    the timeline, side by side, with the colour bar to their right; ``raster`` is the panel size in
    pixels, so the rasters are drawn one pixel per frame pixel.
    """
    width, height = size
    label = LABEL_ROW if labelled else 0
    room_h = height - TOP - label - BOTTOM
    room_w = width - 2 * MARGIN - BAR_GAP - BAR - BAR_TEXT - (panels - 1) * GAP
    panel_h = int(min(room_h, room_w / (panels * aspect)))
    panel_w = int(round(panel_h * aspect))
    content = panels * panel_w + (panels - 1) * GAP + BAR_GAP + BAR + BAR_TEXT
    left = (width - content) // 2
    top = TOP + label + (room_h - panel_h) // 2
    right = left + panels * panel_w + (panels - 1) * GAP
    return SimpleNamespace(panels=[(left + k * (panel_w + GAP), top, panel_w, panel_h) for k in range(panels)],
                           colorbar=(right + BAR_GAP, top, BAR, panel_h), left=left, right=right,
                           raster=(panel_w, panel_h))


def view_bounds(mesh_nodes, sector, raster):
    """Bounds of the panels with square pixels: the mesh box (as ``RasterGeometry.from_mesh``) or the sector
    widened or narrowed in x about its centre to the raster aspect ratio."""
    width, height = raster
    if sector is None:
        from hybridge.io.raster import raster_bounds

        return raster_bounds(SimpleNamespace(node_coords=mesh_nodes), width, height)
    x0, x1, y0, y1 = sector
    centre, half = .5 * (x0 + x1), .5 * (y1 - y0) * width / height
    return centre - half, centre + half, y0, y1


def view_aspect(mesh_nodes, sector) -> float:
    if sector is not None:
        x0, x1, y0, y1 = sector
        return (x1 - x0) / (y1 - y0)
    span = np.ptp(np.asarray(mesh_nodes), axis=0)
    return float(span[0] / span[1])


def _signature(sources, bounds, raster, supersample, initial_state):
    return dict(files=[dict(path=str(source.resolve()), bytes=source.stat().st_size,
                            mtime=round(source.stat().st_mtime, 3)) for source in sources],
                bounds=[float(value) for value in bounds], raster=[int(value) for value in raster],
                supersample=int(supersample), initial_state=bool(initial_state))


def sample_rasters(sources, bounds, raster, *, supersample, initial_state, work, only=None):
    """Area-averaged density rasters of every source: ``(times, [array (states, height, width)], complete)``.

    The rasters are written to ``work`` (``rasters_<k>.npy`` and ``rasters.json``) and reused while
    the sources, bounds, raster size and options are unchanged. Without such a cache, ``only``
    (state indices, for previews) samples just those states into memory (the others are NaN and
    ``complete`` is False) and leaves the cache alone.
    """
    width, height = raster
    meta_path = work / "rasters.json"
    signature = _signature(sources, bounds, raster, supersample, initial_state)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("complete") and meta.get("signature") == signature:
            _log(f"rasters cached in {work}")
            return (meta["times"], [np.load(work / f"rasters_{k}.npy", mmap_mode="r") for k in range(len(sources))],
                    True)
    from hybridge.io.raster import area_sampling_matrix
    from hybridge.io.records import iter_npz_slices

    if only is None:
        work.mkdir(parents=True, exist_ok=True)
        meta_path.unlink(missing_ok=True)
    operator = mask = previous = None
    common_times, outputs = None, []
    for k, source in enumerate(sources):
        npz = np.load(source)
        saved = [round(float(t), 6) for t in npz["times"]]
        prepend = bool(initial_state) and saved[0] > 0.0
        times = ([0.0] if prepend else []) + saved
        if common_times is None:
            common_times = times
        elif not np.allclose(times, common_times, rtol=0, atol=1e-6):
            raise ValueError(f"{source.name}: saved times {times} differ from those of {sources[0].name}")
        key = _mesh_key(npz)
        if key != previous:
            start = time.perf_counter()
            space = _space(npz)
            operator, mask = area_sampling_matrix(space, bounds, width, height, supersample=supersample)
            previous = key
            _log(f"{source.name}: {space.mesh.triangles.shape[0]:,} triangles, p={space.order}; "
                 f"{width}x{height} pixel map, {supersample}x{supersample} samples per pixel, "
                 f"{operator.nnz / 1e6:.1f}M nonzeros ({time.perf_counter() - start:.1f} s)")
        wanted = range(len(times)) if only is None else sorted({index % len(times) for index in only})
        if only is None:
            output = np.lib.format.open_memmap(work / f"rasters_{k}.npy", mode="w+", dtype=np.float32,
                                               shape=(len(times), height, width))
        else:
            output = np.full((len(times), height, width), np.nan, dtype=np.float32)

        def store(index, coefficients):
            image = (operator @ np.asarray(coefficients).ravel()).reshape(height, width).astype(np.float32)
            image[mask] = np.nan
            output[index] = image
            _log(f"{source.name}: state {index + 1}/{len(times)} t={times[index]:g} "
                 f"min={np.nanmin(image):+.4f} max={np.nanmax(image):.4f}")

        if prepend and 0 in wanted:
            from scripts.guiding_center.studies.diocotron.plot_diocotron_zg import _initial_density

            start = time.perf_counter()
            _, density = _initial_density(npz, _run_configuration(source))
            _log(f"{source.name}: t=0 density projected on the host ({time.perf_counter() - start:.1f} s)")
            store(0, density.coeffs)
        last = max(wanted)
        for index, coefficients in enumerate(iter_npz_slices(source, "density"), start=int(prepend)):
            if index > last:
                break
            if index in wanted:
                store(index, coefficients)
        if only is None:
            output.flush()
        outputs.append(output)
    if only is None:
        minima = [[float(np.nanmin(image)) for image in output] for output in outputs]
        maxima = [[float(np.nanmax(image)) for image in output] for output in outputs]
        meta_path.write_text(json.dumps(dict(
            signature=signature, times=common_times, sources=[str(source) for source in sources],
            minimum=min(min(row) for row in minima), maximum=max(max(row) for row in maxima),
            state_minima=minima, state_maxima=maxima, complete=True), indent=1) + "\n")
        _log(f"rasters written to {work}")
    return common_times, outputs, only is None


def _repeats(states: int, duration: float, fps: float) -> list[int]:
    """Frames per state: ``duration * fps`` frames in all, as evenly as whole frames allow."""
    total = int(round(duration * fps))
    if total < states:
        raise ValueError(f"{states} states need at least {states} frames: raise --duration or --fps")
    return [int(round((i + 1) * total / states)) - int(round(i * total / states)) for i in range(states)]


class FrameRenderer:
    """One persistent figure of the panels, colour bar, time label and timeline; draws one state at a time."""

    def __init__(self, size, layout, bounds, times, *, cmap, norm, extend, labels, title, subtitle, mark=None,
                 note=None, sector_frame=False):
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import MaxNLocator

        width, height = self.size = size
        self.times = [float(t) for t in times]
        self.figure = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)

        def box(left, top, w, h):
            return [left / width, 1 - (top + h) / height, w / width, h / height]

        self.images = []
        for k, rect in enumerate(layout.panels):
            ax = self.figure.add_axes(box(*rect))
            blank = np.full(layout.raster[::-1], np.nan)
            self.images.append(ax.imshow(blank, extent=bounds, origin="upper", cmap=cmap, norm=norm,
                                         interpolation="nearest", resample=False, aspect="auto"))
            ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(sector_frame)
                spine.set_color(MUTED)
                spine.set_linewidth(.8)
            if labels:
                ax.set_title(labels[k], fontsize=17, pad=12, color=INK)
            if mark is not None:
                x0, x1, y0, y1 = mark
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.1))
        cax = self.figure.add_axes(box(*layout.colorbar))
        bar = self.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend=extend)
        span = norm.vmax - norm.vmin
        step = .2 if span >= .6 else .1 if span >= .3 else .05
        ticks = [step * k for k in range(int(np.ceil(norm.vmin / step - 1e-9)), int(np.floor(norm.vmax / step + 1e-9)) + 1)]
        pixels_per_unit = layout.colorbar[3] / span
        if norm.vmin < 0 and min(abs(norm.vmin - tick) for tick in ticks) * pixels_per_unit >= 24:
            ticks.insert(0, norm.vmin)  # the bottom of the scale, the most negative pixel, unless its label would touch 0
        bar.set_ticks(ticks, labels=[(f"{tick:.1f}" if abs(tick / step - round(tick / step)) < 1e-9 else f"{tick:.2f}")
                                     .replace("-", "−") for tick in ticks])
        bar.outline.set_edgecolor(MUTED)
        bar.outline.set_linewidth(.8)
        cax.tick_params(labelsize=13, color=MUTED, length=4)
        cax.set_ylabel(r"density $\rho_h$", fontsize=16, labelpad=10)
        self.figure.text(.5, 1 - 42 / height, title, ha="center", va="center", fontsize=21, color=INK)
        if subtitle:
            self.figure.text(.5, 1 - 84 / height, subtitle, ha="center", va="center", fontsize=14.5, color=MUTED)
        # Time label and timeline of the saved states, in the bottom row under the panels.
        row = height - BOTTOM / 2 - 6
        self.label = self.figure.text(layout.left / width, 1 - row / height, "", ha="left", va="center",
                                      fontsize=22, color=INK)
        start = layout.left + 190
        axis = self.figure.add_axes(box(start, row - 15, layout.right - start, 30))
        end = max(self.times)
        axis.set_xlim(0., end)
        axis.set_ylim(-1., 1.)
        axis.plot([0., end], [0., 0.], color=FAINT, lw=3, solid_capstyle="butt", clip_on=False, zorder=1)
        self.progress, = axis.plot([0., 0.], [0., 0.], color=MUTED, lw=3, solid_capstyle="butt", clip_on=False,
                                   zorder=2)
        if len(self.times) <= 60:  # one mark per saved state
            axis.plot(self.times, np.zeros(len(self.times)), linestyle="none", marker="|", markersize=15,
                      markeredgewidth=1.6, color=MUTED, clip_on=False, zorder=3)
        self.marker, = axis.plot([0.], [0.], "o", markersize=11, color=INK, clip_on=False, zorder=4)
        axis.set_yticks([])
        axis.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=(1, 2, 2.5, 5, 10), integer=True))
        axis.tick_params(axis="x", length=0, pad=12, labelsize=12, labelcolor=MUTED)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.patch.set_alpha(0)
        if note:  # above the right end of the timeline
            self.figure.text(layout.right / width, 1 - (row - 33) / height, note, ha="right", va="center",
                             fontsize=12, color=MUTED)

    def draw(self, rasters, index):
        """Draw saved state ``index`` (one raster per panel) and return the frame as uint8 RGBA."""
        t = self.times[index]
        for image, raster in zip(self.images, rasters):
            image.set_data(np.ma.masked_invalid(np.asarray(raster[index], dtype=float)))
        self.label.set_text(f"$t = {t:g}$")
        self.progress.set_xdata([0., t])
        self.marker.set_xdata([t])
        self.figure.canvas.draw()
        frame = np.asarray(self.figure.canvas.buffer_rgba())
        if frame.shape[:2] != self.size[::-1]:
            raise RuntimeError(f"frame is {frame.shape[1]}x{frame.shape[0]}, expected {self.size[0]}x{self.size[1]}")
        return frame

    def close(self):
        plt.close(self.figure)


def _scale(work_root, stems, own_minimum, own_maximum):
    """Colour-scale minimum and maximum over this movie and the cached rasters of ``stems``."""
    minimum, maximum = own_minimum, own_maximum
    for stem in stems:
        meta = work_root / stem / "rasters.json"
        if not meta.exists():
            raise FileNotFoundError(f"--share-scale {stem}: no cached rasters ({meta}); run it with --sample-only first")
        record = json.loads(meta.read_text())
        minimum, maximum = min(minimum, record["minimum"]), max(maximum, record["maximum"])
    return minimum, maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshots", nargs="+", type=Path, help="*_snapshots.npz files, one panel each")
    parser.add_argument("--output", type=Path, required=True, help="MP4 path")
    parser.add_argument("--labels", nargs="*", default=(), help="panel titles, one per snapshot file")
    parser.add_argument("--title", default="")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--sector", type=_floats, default=None, help="window x0,x1,y0,y1 (default: the whole mesh)")
    parser.add_argument("--mark-sector", type=_floats, default=None,
                        help="outline the window x0,x1,y0,y1 on every panel (e.g. the sector of another movie)")
    parser.add_argument("--size", type=_size, default=(1440, 1080), help="frame size WxH in pixels")
    parser.add_argument("--supersample", type=int, default=3,
                        help="each pixel is the mean of N x N samples (default 3, as the article figures)")
    parser.add_argument("--initial-state", action="store_true",
                        help="prepend t=0 (the projected initial density) when the file starts later")
    parser.add_argument("--density-max", type=float, default=1.0, help="top of the colour scale (profile maximum)")
    parser.add_argument("--share-scale", nargs="*", type=Path, default=(),
                        help="other movies (same --work-dir) whose pixels also set the colour-scale minimum")
    parser.add_argument("--duration", type=float, default=40.0, help="movie length in seconds")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--crf", type=int, default=18, help="libx264 quality (lower: better, larger)")
    parser.add_argument("--preset", default="slow", help="libx264 preset")
    parser.add_argument("--threads", type=int, default=4, help="libx264 threads")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="raster cache and previews (default: <output dir>/render_work); <stem>/ per movie")
    parser.add_argument("--sample-only", action="store_true", help="cache the rasters and stop")
    parser.add_argument("--preview", type=lambda text: tuple(int(v) for v in text.split(",") if v), default=None,
                        help="write the frames of these saved states (indices) as PNG files, no movie")
    parser.add_argument("--force", action="store_true", help="render even if the movie is complete")
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.snapshots):
        parser.error("--labels needs one label per snapshot file")
    for name in ("sector", "mark_sector"):
        value = getattr(args, name)
        if value is not None and (len(value) != 4 or value[0] >= value[1] or value[2] >= value[3]):
            parser.error(f"--{name.replace('_', '-')} needs x0,x1,y0,y1 with x0<x1 and y0<y1")
    if args.supersample < 1:
        parser.error("--supersample must be a positive integer")
    output = args.output.resolve()
    stem = output.stem
    work_root = (args.work_dir or output.parent / "render_work").resolve()
    work = work_root / stem
    record_path = work / "movie.json"
    if args.preview is None and not args.sample_only and output.exists() and not args.force and record_path.exists():
        if json.loads(record_path.read_text()).get("complete"):
            _log(f"{output.name} is complete; skipped (--force to render again)")
            return

    first = np.load(args.snapshots[0])
    nodes = first["node_coords"]
    layout = frame_layout(args.size, len(args.snapshots), view_aspect(nodes, args.sector),
                          labelled=bool(args.labels))
    bounds = view_bounds(nodes, args.sector, layout.raster)
    _log(f"{output.name}: {args.size[0]}x{args.size[1]} frames, panels {layout.raster[0]}x{layout.raster[1]} px, "
         f"view x=[{bounds[0]:.4f}, {bounds[1]:.4f}] y=[{bounds[2]:.4f}, {bounds[3]:.4f}]")
    times, rasters, complete = sample_rasters(args.snapshots, bounds, layout.raster, supersample=args.supersample,
                                              initial_state=args.initial_state, work=work, only=args.preview)
    if args.sample_only:
        return
    shown = range(len(times)) if args.preview is None else sorted({i % len(times) for i in args.preview})
    scaled = range(len(times)) if complete else shown
    own_min = min(float(np.nanmin(raster[i])) for raster in rasters for i in scaled)
    own_max = max(float(np.nanmax(raster[i])) for raster in rasters for i in scaled)
    stems = [Path(other).stem for other in args.share_scale if Path(other).stem != stem]
    minimum, maximum = _scale(work_root, stems, own_min, own_max)
    cmap, norm = signed_density_colors(minimum, args.density_max)
    extend = "max" if maximum > args.density_max * (1 + 1e-3) else "neither"
    _log(f"colour scale [{norm.vmin:+.4f}, {norm.vmax:g}] (pixel maximum {maximum:.4f}"
         + (", saturated above the scale)" if extend == "max" else ")")
         + ("" if complete else "; preview scale: from the previewed states only"))
    rebuilt = bool(args.initial_state) and times[0] == 0.0 and round(float(first["times"][0]), 6) > 0.0
    note = (f"{'initial state and ' if rebuilt else ''}{len(times) - rebuilt} saved states, "
            f"{args.duration / len(times):g} s each") if len(times) <= 60 else None
    with publication_style(font_size=15):
        renderer = FrameRenderer(args.size, layout, bounds, times, cmap=cmap, norm=norm, extend=extend,
                                 labels=list(args.labels), title=args.title, subtitle=args.subtitle,
                                 mark=args.mark_sector, note=note, sector_frame=args.sector is not None)
        try:
            if args.preview is not None:
                from PIL import Image

                directory = work / "preview"
                directory.mkdir(parents=True, exist_ok=True)
                for index in shown:
                    path = directory / f"{stem}_state{index:02d}_t{times[index]:g}.png"
                    Image.fromarray(renderer.draw(rasters, index)[..., :3].copy()).save(path)
                    _log(f"preview {path}")
                return
            from hybridge.io.movie import MovieWriter

            repeats = _repeats(len(times), args.duration, args.fps)
            partial = output.with_name(f"{stem}.partial.mp4")
            metadata = dict(title=_plain(args.title) or stem,
                            comment=f"density rho_h from {', '.join(p.name for p in args.snapshots)}; "
                                    f"rendered by HYBRIDGE scripts/guiding_center/diagnostics/"
                                    f"render_diocotron_movie.py")
            writer = MovieWriter(partial, fps=args.fps, crf=args.crf, preset=args.preset, threads=args.threads,
                                 metadata=metadata)
            start = time.perf_counter()
            try:
                for index, count in enumerate(repeats):
                    frame = renderer.draw(rasters, index)
                    for _ in range(count):
                        writer.append(frame)
                    _log(f"{output.name}: state {index + 1}/{len(times)} t={times[index]:g} "
                         f"({count} frames, {time.perf_counter() - start:.0f} s)")
            finally:
                writer.close()
        finally:
            renderer.close()
    os.replace(partial, output)
    frames = sum(repeats)
    record = dict(output=str(output), complete=True, sources=[str(p) for p in args.snapshots],
                  labels=list(args.labels), times=times, frames=frames, fps=args.fps,
                  duration_seconds=frames / args.fps, seconds_per_state=args.duration / len(times),
                  size_bytes=output.stat().st_size, frame_size=list(args.size), raster=list(layout.raster),
                  bounds=list(bounds), supersample=args.supersample, colour_scale=[norm.vmin, norm.vmax],
                  pixel_maximum=maximum, shared_with=stems, crf=args.crf, preset=args.preset,
                  command="PYTHONPATH=. .venv/bin/python -m scripts.guiding_center.studies.diocotron.render_diocotron_movie "
                          + " ".join(shlex.quote(arg) for arg in sys.argv[1:]),
                  rendered=datetime.now().isoformat(timespec="seconds"))
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    _log(f"{output} written: {frames} frames, {frames / args.fps:.1f} s, {output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
