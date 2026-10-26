"""Tables for the article's diocotron campaigns (no simulation).

Reads the ``<prefix>_summary.json`` files written by
``scripts/guiding_center/studies/diocotron/run_diocotron_benchmark.py`` and the run
JSONL records they point to. Each seeded run gets two growth-rate fits of the
target-mode potential amplitude:

``declared``
    the window declared in advance (the semi-Lagrangian windows of Madaule et
    al. (2013) on the annulus), as recorded by the campaign;
``linear``
    the modal plateau (:func:`linear_window`): it ends before the second-harmonic
    ratio |phi_{2m}|/|phi_m| exceeds ``--harmonic-threshold`` (default 5e-3,
    quadratic nonlinearity) and starts once the local growth rate is constant
    to 0.5% (non-modal transients of the seed excluded).

Writes ``<output>.csv`` (all runs) and, with ``--latex``, a booktabs table of the
annulus rows: linear theory, semi-Lagrangian (half the tabulated value) and the
HDG rates per scheme and drift on the finest band mesh.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.guiding_center.diagnostics.diocotron_reference import fit_growth


def _series(rows, key):
    return np.array([np.nan if row.get(key) is None else float(row[key]) for row in rows], dtype=float)


def linear_window(times, amplitude, ratio, *, threshold=5e-3, local=4.0, tolerance=1e-3, minimum_samples=8):
    """Modal plateau of the growth: the fit window, or None.

    End: within the longest interval on which the second-harmonic ratio
    |phi_{2m}|/|phi_m| stays at or below ``threshold`` (quadratic nonlinearity),
    its last time. Start: local rates (log-amplitude slopes over ``local`` time
    units) are compared with their median over the second half of that interval
    (the late linear phase); the window starts where the trailing run of local
    rates within ``tolerance`` (relative) of that level begins, so the decaying
    non-modal transients and beatings of the seed are excluded.
    """
    good = np.isfinite(ratio) & (ratio <= threshold) & np.isfinite(amplitude) & (amplitude > 0)
    best, start = None, None
    for index, flag in enumerate(np.append(good, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if best is None or best[1] - best[0] < minimum_samples:
        return None
    first, last = best[0], best[1] - 1
    log_amplitude = np.log(np.where(amplitude > 0, amplitude, np.nan))
    spacing = np.median(np.diff(times))
    width = max(2, int(round(local / spacing)))
    if last - first < 2 * width + minimum_samples:
        return float(times[first]), float(times[last])
    slopes = (log_amplitude[first + width:last + 1] - log_amplitude[first:last + 1 - width]) / (
        times[first + width:last + 1] - times[first:last + 1 - width])
    level = np.median(slopes[len(slopes) // 2:])
    inside = np.abs(slopes - level) <= tolerance * abs(level)
    begin = len(slopes)
    while begin > 0 and inside[begin - 1]:
        begin -= 1
    if begin == len(slopes) or last - (first + begin) < minimum_samples:
        return None
    return float(times[first + begin]), float(times[last])


def analyse_row(row, *, threshold):
    """Return the declared and linear-window fits of one seeded run."""
    records = [json.loads(line) for line in Path(row["jsonl"]).read_text().splitlines() if line.strip()]
    mode = int(records[0]["run_configuration"]["case_parameters"]["k"])
    times = _series(records, "time")
    amplitude = _series(records, f"diocotron_phi_mode_{mode}_l2")
    ratio = _series(records, "diocotron_phi_harmonic_ratio")
    reference = float(row["reference_gamma"])
    declared = fit_growth(times, amplitude, window=tuple(row["window"]), reference=reference)
    window = linear_window(times, amplitude, ratio, threshold=threshold)
    linear = (fit_growth(times, amplitude, window=window, reference=reference)
              if window is not None else dict(status="no_linear_window"))
    halves = []
    if window is not None:
        middle = .5 * (window[0] + window[1])
        halves = [fit_growth(times, amplitude, window=part, reference=reference, minimum_samples=4)
                  for part in ((window[0], middle), (middle, window[1]))]
    out = dict(row)
    out.update(mode=mode, declared_gamma=declared.get("growth_rate"),
               declared_relative_error=declared.get("relative_error"),
               declared_r_squared=declared.get("r_squared"),
               linear_window=None if window is None else list(window),
               linear_gamma=linear.get("growth_rate"), linear_relative_error=linear.get("relative_error"),
               linear_r_squared=linear.get("r_squared"), harmonic_threshold=threshold,
               linear_half_relative_errors=[half.get("relative_error") for half in halves])
    return out


def _percent(value):
    return "--" if value is None else f"${100 * value:+.3f}\\%$"


def annulus_table(rows, *, fit="linear") -> str:
    """Booktabs table body: one line per annulus row, finest band mesh, both schemes and drifts."""
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    columns = [("si-bdf2", "raw"), ("imex-ark3", "raw"), ("si-bdf2", "compatible"), ("imex-ark3", "compatible")]
    lines = [r"\begin{tabular}{ccccccccc}\toprule",
             r"$\ell$ & $(r^-,r^+)$ & theory & semi-Lagrangian & \multicolumn{2}{c}{HDG raw drift} & "
             r"\multicolumn{2}{c}{HDG compatible drift} & window\\",
             r" & & $\gamma_\ell$ & \cite{Madaule2013diocotron} & SI-BDF2 & ARK3 & SI-BDF2 & ARK3 & \\\midrule"]
    for label in sorted(by_label, key=lambda text: int(text.split("_")[0][1:])):
        group = by_label[label]
        finest = min(row["band_mesh_size"] for row in group)
        group = [row for row in group if row["band_mesh_size"] == finest]
        first = group[0]
        inner, outer = first["label"].split("_r")[1].split("_")
        cells, fallback = [], False
        for scheme, velocity in columns:
            match = [row for row in group if row["scheme"] == scheme and row["velocity"] == velocity]
            if not match:
                cells.append("--")
                continue
            value = match[0][f"{fit}_relative_error"]
            if value is None and fit == "linear":
                # No sub-threshold harmonic interval (e.g. l = 2, whose harmonic m = 4 is itself
                # a faster unstable mode): fall back to the declared window.
                value, fallback = match[0]["declared_relative_error"], True
            cells.append(_percent(value))
        window = first.get("linear_window") if fit == "linear" else first.get("window")
        if fit == "linear" and window is None:
            window = first.get("window")
        window_text = "--" if window is None else f"[{window[0]:.0f},{window[1]:.0f}]" + ("$^\\dagger$" if fallback else "")
        semi = first.get("semi_lagrangian_gamma")
        semi_text = "--" if semi is None else f"{semi:.5f} ({_percent(first.get('semi_lagrangian_relative_error'))})"
        lines.append(f"{first['mode']} & $({inner},{outer})$ & {first['reference_gamma']:.6f} & {semi_text} & "
                     + " & ".join(cells) + f" & {window_text}\\\\")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines) + "\n"


def highmode_table(rows, *, fit="linear") -> str:
    """Booktabs table body for the high-mode rings: resolution, cost and accuracy per run."""
    lines = [r"\begin{tabular}{rrrrllrrr}\toprule",
             r"$m$ & $h_{\rm band}$ & triangles & unknowns & scheme & drift & step [s] & "
             r"$\gamma_h/\gamma-1$ & control\\\midrule"]
    order = sorted(rows, key=lambda row: (row["mode"], -row["band_mesh_size"], row["scheme"], row["velocity"]))
    for row in order:
        error = row.get(f"{fit}_relative_error")
        if error is None:
            error = row.get("declared_relative_error")
        unknowns = row.get("element_unknowns")
        step = row.get("median_linear_step_wall_time")
        control = row.get("control_fraction")
        lines.append(
            f"{row['mode']} & {row['band_mesh_size']:g} & {int(row['triangles']):,} & "
            + ("--" if unknowns is None else f"{int(unknowns) / 1e6:.2f}M") + " & "
            + f"{row['scheme']} & {row['velocity']} & " + ("--" if step is None else f"{float(step):.3f}")
            + f" & {_percent(error)} & " + ("--" if control is None else f"{100 * float(control):.1f}\\%")
            + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines).replace(",", "{,}") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="output stem (CSV)")
    parser.add_argument("--harmonic-threshold", type=float, default=5e-3)
    parser.add_argument("--latex", type=Path, help="write the annulus table (linear-window fits)")
    parser.add_argument("--latex-fit", choices=("linear", "declared"), default="linear")
    parser.add_argument("--latex-highmode", type=Path, help="write the high-mode resolution/cost/accuracy table")
    args = parser.parse_args()
    rows = []
    for path in args.summaries:
        rows.extend(analyse_row(row, threshold=args.harmonic_threshold)
                    for row in json.loads(path.read_text())["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        flat = [{key: (json.dumps(value) if isinstance(value, (list, dict)) else value)
                 for key, value in row.items()} for row in rows]
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    for row in rows:
        print(f"{row['label']:<14} {row['scheme']:<9} {row['velocity']:<10} hb={row['band_mesh_size']:<7g} "
              f"declared {row['declared_gamma']:.6f} ({_percent(row['declared_relative_error'])}) "
              f"linear {row['linear_window']} "
              + (f"{row['linear_gamma']:.6f} ({_percent(row['linear_relative_error'])})"
                 if row["linear_gamma"] is not None else "--"))
    if args.latex_highmode is not None:
        args.latex_highmode.write_text("% Generated by scripts/guiding_center/studies/diocotron/analyze_diocotron_campaign.py\n"
                                       + highmode_table(rows, fit=args.latex_fit))
    if args.latex is not None:
        args.latex.write_text("% Generated by scripts/guiding_center/studies/diocotron/analyze_diocotron_campaign.py\n"
                              + annulus_table(rows, fit=args.latex_fit))


if __name__ == "__main__":
    main()
