"""Aligned, paginated Matplotlib figures for recorded diagnostic time series."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import textwrap
import re
import unicodedata

import numpy as np


@dataclass(frozen=True)
class TimeSeries:
    label: str
    times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class DiagnosticPanel:
    title: str
    ylabel: str
    series: tuple[TimeSeries, ...]
    xscale: str = "linear"
    yscale: str = "linear"
    filename: str | None = None
    group: str | None = None
    drawstyle: str = "default"


def numeric_time_series(rows, *, exclude=()):
    """Flatten numeric histories, preserving missing samples as NaNs.

    Nested dictionaries and per-stage lists are retained. Strings, booleans,
    excluded top-level metadata, and values with fewer than two finite samples
    do not define numeric histories.
    """
    def flatten(value, prefix=""):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if not prefix and key in {*exclude, "time", "step"}:
                continue
            if isinstance(item, dict):
                yield from flatten(item, name)
            elif isinstance(item, (tuple, list)):
                yield from flatten({str(i): entry for i, entry in enumerate(item)}, name)
            elif isinstance(item, (float, int, np.number)) and not isinstance(item, (bool, np.bool_)):
                yield name, float(item)

    times = np.asarray([row.get("time", np.nan) for row in rows], dtype=float)
    flattened = [dict(flatten(row)) for row in rows]
    names = sorted({key for row in flattened for key in row})
    result = {}
    for name in names:
        values = np.asarray([row.get(name, np.nan) for row in flattened])
        if np.count_nonzero(np.isfinite(times) & np.isfinite(values)) >= 2:
            result[name] = TimeSeries(name, times, values)
    return result


def _figure(rows=3, columns=2, *, display=False):
    if display:
        from matplotlib import pyplot as plt

        return plt.subplots(rows, columns, squeeze=False, figsize=(6.2*columns, 3.25*rows),
                            layout="constrained", facecolor="white")

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(6.2*columns, 3.25*rows), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, columns, squeeze=False)


def _style_axis(axis, title, ylabel):
    axis.set_title(title if "$" in title else textwrap.fill(title, 62), fontsize=10.5, fontweight="bold", loc="left")
    axis.set_xlabel(r"Time $t$", fontsize=10)
    axis.set_ylabel(ylabel, fontsize=10)
    axis.grid(True, which="major", alpha=0.22, linewidth=0.65)
    axis.grid(True, which="minor", alpha=0.08, linewidth=0.4)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)
    axis.set_axisbelow(True)


def draw_time_series_panel(axis, panel):
    """Draw a panel without inventing positive replacements for zeros/signs."""
    all_nonzero = np.concatenate([
        np.abs(np.asarray(series.values, dtype=float)).ravel() for series in panel.series
    ])
    nonzero = all_nonzero[np.isfinite(all_nonzero) & (all_nonzero > 0)]
    linthresh = max(float(nonzero.max())*1e-6, float(nonzero.min())) if len(nonzero) else 1e-12
    axis.set_xscale(panel.xscale)
    axis.set_yscale(panel.yscale, **({"linthresh": linthresh} if panel.yscale == "symlog" else {}))
    shown = False
    for series in panel.series:
        times, values = np.broadcast_arrays(np.asarray(series.times, dtype=float),
                                            np.asarray(series.values, dtype=float))
        valid = np.isfinite(times) & np.isfinite(values)
        if panel.xscale == "log":
            valid &= times > 0
        if panel.yscale == "log":
            valid &= values > 0
        if valid.any():
            # NaNs preserve gaps at missing and nonpositive log samples.
            axis.plot(np.where(valid, times, np.nan), np.where(valid, values, np.nan),
                      linewidth=1.65, marker="o" if len(times) <= 12 else None,
                      markersize=3, label=series.label, drawstyle=panel.drawstyle)
            shown = True
    if not shown:
        axis.text(.5, .5, "No finite samples on these axes", ha="center", va="center",
                  transform=axis.transAxes, fontsize=10)
        if panel.xscale == "log":
            axis.set_xlim(1, 10)
        if panel.yscale == "log":
            axis.set_ylim(1, 10)
    if shown and len(panel.series) > 1:
        axis.legend(fontsize=8, frameon=False)
    _style_axis(axis, panel.title, panel.ylabel)
    if panel.yscale == "linear":
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=True)


def diagnostic_pages(panels):
    """Group related panels and assign safe descriptive PNG names."""
    groups = {}
    for index, panel in enumerate(panels):
        key = ("group", panel.group) if panel.group else ("single", index)
        groups.setdefault(key, []).append(panel)
    pages, used_names = [], set()
    for (kind, key), members in groups.items():
        raw_name = key if kind == "group" else members[0].filename or members[0].title
        raw_name = unicodedata.normalize("NFKD", raw_name).encode("ascii", "ignore").decode()
        stem = re.sub(r"[^a-z0-9]+", "_", raw_name.lower()).strip("_")[:150] or "diagnostic"
        for start in range(0, len(members), 6):
            name = stem if len(members) <= 6 else f"{stem}_{start//6+1:02d}"
            candidate, index = name, 2
            while candidate in used_names:
                candidate = f"{name}_{index}"
                index += 1
            used_names.add(candidate)
            pages.append((f"{candidate}.png", members[start:start+6]))
    return pages


def plot_diagnostic_panels(panels, output_dir=None, *, title="Diagnostics", prefix="diagnostics", display=False):
    """Prepare interactive pages and/or save named diagnostic PNGs and a paginated PDF.

    Interactive figures remain open for a final show_diagnostic_figures call.
    Saving alone uses Agg and does not open windows or change the GUI backend.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    panels = list(panels)
    if not panels or (output_dir is None and not display):
        return []
    paths = []
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths.append(output_dir/f"{prefix}.pdf")
    with PdfPages(paths[0]) if paths else nullcontext() as pdf:
        for filename, page in diagnostic_pages(panels):
            figure, axes = _figure(rows=(len(page)+1)//2, columns=2 if len(page)>1 else 1, display=display)
            topic = (page[0].group or page[0].title).replace("_", " ").title()
            figure.suptitle(f"{title}\n{topic}", fontsize=12, fontweight="bold")
            for axis, panel in zip(axes.flat, page):
                draw_time_series_panel(axis, panel)
            for axis in list(axes.flat)[len(page):]:
                axis.set_visible(False)
            figure.supxlabel("Log axes omit nonpositive samples; signed drifts retain their sign.",
                                fontsize=8, color="0.4")
            figure.align_ylabels()
            if output_dir is not None:
                pdf.savefig(figure)
                png = output_dir/filename
                figure.savefig(png, dpi=170)
                paths.append(png)
            if not display:
                figure.clear()
    return paths


def plot_mode_history(times, modes, amplitudes, output_dir=None, *, title="Mode activity",
                      relative_threshold=1e-3, target_mode=None, prefix="active_modes", display=False):
    """Prepare and/or save modal figures; output_dir=None creates no files."""
    if output_dir is None and not display:
        return []
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator
    from hdgfem.diagnostics import modal_activity

    times, modes, amplitudes = np.asarray(times), np.asarray(modes), np.asarray(amplitudes)
    activity = modal_activity(amplitudes, modes, relative_threshold=relative_threshold)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = _figure(rows=2, display=display)
    figure.suptitle(title, fontsize=15, fontweight="bold")
    relative = activity["relative_amplitudes"]
    masked = np.ma.masked_where(~np.isfinite(relative) | (relative <= 0), relative)
    heatmap = axes[0, 0].pcolormesh(times, modes, masked.T, shading="nearest",
                                  norm=LogNorm(vmin=1e-6, vmax=1), cmap="magma", rasterized=True)
    figure.colorbar(heatmap, ax=axes[0, 0], label=r"$A_m(t)/\max_j A_j(t)$", extend="min")
    _style_axis(axes[0, 0], "Resolved angular modes", r"Mode $m$")
    for rank in range(activity["dominant_modes"].shape[1]):
        axes[0, 1].plot(times, activity["dominant_modes"][:, rank], ".",
                       markersize=3.5, label=f"Rank {rank+1}")
    axes[0, 1].legend(fontsize=8, frameon=False)
    axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axis(axes[0, 1], "Strongest active modes at each time", r"Mode $m$")
    peaks = np.max(np.where(np.isfinite(relative), relative, 0), axis=0)
    shown = list(np.argsort(-peaks, kind="stable")[:6])
    if target_mode in modes and int(np.flatnonzero(modes == target_mode)[0]) not in shown:
        shown[-1] = int(np.flatnonzero(modes == target_mode)[0])
    series = tuple(TimeSeries(f"m={modes[index]}", times, amplitudes[:, index]) for index in shown)
    draw_time_series_panel(axes[1, 0], DiagnosticPanel(
        "Leading potential-mode amplitudes", r"$A_m(t)=\|\widehat{\phi-\phi_{\rm eq}}_m\|_{L^2}$", series, "log", "log"))
    axes[1, 1].plot(times, activity["active_counts"], color="tab:green", linewidth=1.7)
    axes[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axis(axes[1, 1], f"Active modes (amplitude ≥ {relative_threshold:g} × strongest)", "Count")
    figure.align_ylabels()
    paths = [] if output_dir is None else [output_dir/f"{prefix}.png", output_dir/f"{prefix}.pdf"]
    for file in paths:
        figure.savefig(file, dpi=170)
    if not display:
        figure.clear()
    return paths


def show_diagnostic_figures():
    """Enter the selected Matplotlib GUI event loop after preparing all figures."""
    from matplotlib import pyplot as plt

    plt.show(block=True)


__all__ = ["TimeSeries", "DiagnosticPanel", "numeric_time_series",
           "draw_time_series_panel", "diagnostic_pages", "plot_diagnostic_panels", "plot_mode_history", "show_diagnostic_figures"]
