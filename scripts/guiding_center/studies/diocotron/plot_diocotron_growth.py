"""Growth of the diocotron instability for every disk case on shared axes; no time integration.

Reads the campaign summaries of ``run_diocotron_benchmark.py`` (smooth m = 9 ring,
Zoni--Guclu truncated ring, Gaussian rings m = 64 and 128) and draws one figure:

(a) amplitude of the seeded mode on the finest mesh (IMEX-ARK3, raw drift), the
    matched unperturbed control (dotted) and e^{gamma t} of the eigenvalue problem
    anchored in the middle of the modal plateau (dashed);
(b) local growth rate relative to gamma (slope of the log amplitude over 4 time
    units), the modal plateau drawn thicker;
(c) relative error of the plateau rate against the element unknowns per field,
    both schemes (filled: IMEX-ARK3, open: SI-BDF2).

Colours identify the case, from a validated categorical palette (all pairs pass the
normal-vision floor; the colour-vision band is relieved by a marker per case and
direct labels), so the scheme colours of the other figures are not reused.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from hybridge.io.figures import publication_style, save_publication_figure  # noqa: E402
from scripts.guiding_center.studies.diocotron.plot_diocotron_highmode import (  # noqa: E402
    INK, MUTED, _control_path, _error, _local_rates, _records, _series, load_rows)

# label prefix of the summary rows -> (legend text, short direct label, colour, marker, label offset in points)
CASES = {
    "m9_smooth": ("smooth ring, $m=9$", "$m=9$", "#008300", "o", 6),
    "m9_zg": ("Zoni–Güçlü ring, $m=9$", None, "#e34948", "D", 0),
    "m64_gaussian": ("Gaussian ring, $m=64$", "$m=64$", "#eda100", "^", 0),
    "m128_gaussian": ("Gaussian ring, $m=128$", "$m=128$", "#4a3aa7", "s", 0),
}


def _case(row):
    return next((key for key in CASES if row["label"].startswith(key)), None)


def growth_figure(rows, stem, *, dt=0.05, final_time=70.0):
    rows = [row for row in rows if row["velocity"] == "raw" and round(float(row["dt"]), 6) == dt
            and _case(row) is not None]
    cases = [key for key in CASES if any(_case(row) == key for row in rows)]
    with publication_style(font_size=8.0):
        figure, (amplitude, local, error) = plt.subplots(1, 3, figsize=(7.2, 2.75), layout="constrained",
                                                         gridspec_kw=dict(width_ratios=(1.15, 1.0, 1.0)))
        for key in cases:
            legend, short, color, marker, offset = CASES[key]
            mine = [row for row in rows if _case(row) == key]
            mode = int(mine[0]["mode"])
            finest = max(row["triangles"] for row in mine)
            anchor = next(row for row in mine if row["triangles"] == finest and row["scheme"] == "imex-ark3")
            reference = float(anchor["reference_gamma"])
            records = _records(anchor["jsonl"])
            t, a = _series(records, "time"), _series(records, f"diocotron_phi_mode_{mode}_l2")
            keep = t <= final_time + 1e-9
            amplitude.semilogy(t[keep], a[keep], color=color, lw=1.3)
            if short:  # the two m = 9 rings nearly coincide: one label, the legend tells them apart
                amplitude.annotate(short, (t[keep][-1], a[keep][-1]), xytext=(3, offset), textcoords="offset points",
                                   color=INK, fontsize=6.5, va="center", ha="left")
            control = _control_path(anchor)
            if control.exists():
                crecords = _records(control)
                tc = _series(crecords, "time")
                amplitude.semilogy(tc[tc <= final_time + 1e-9],
                                   _series(crecords, f"diocotron_phi_mode_{mode}_l2")[tc <= final_time + 1e-9],
                                   color=color, lw=.9, linestyle=":", alpha=.85)
            window = anchor.get("linear_window")
            times, rates = _local_rates(records, mode)
            relative = 100 * (rates / reference - 1)
            local.plot(times, relative, color=color, lw=.9)
            if window:
                middle = .5 * (window[0] + window[1])
                level = np.interp(middle, t, np.log(a))
                x = np.linspace(0.0, min(final_time, window[1] + 25.0), 200)
                amplitude.semilogy(x, np.exp(level + reference * (x - middle)), color=INK, lw=.7,
                                   linestyle=(0, (4, 2)))
                on = (times >= window[0]) & (times <= window[1])
                local.plot(times[on], relative[on], color=color, lw=2.4, solid_capstyle="butt")
            for scheme, filled in (("imex-ark3", True), ("si-bdf2", False)):
                series = sorted((row for row in mine if row["scheme"] == scheme), key=lambda r: r["element_unknowns"])
                if series:
                    error.semilogx([row["element_unknowns"] for row in series], [100 * _error(row) for row in series],
                                   color=color, lw=1.0, marker=marker, markersize=5.0,
                                   markerfacecolor=color if filled else "white", markeredgecolor=color,
                                   markeredgewidth=1.0)
        amplitude.set(xlabel="$t$", ylabel=r"$\|\hat\phi_m\|$ (seeded mode $m$)", xlim=(0, final_time + 12),
                      ylim=(1e-19, None))
        amplitude.grid(True, which="major", alpha=.45)
        local.axhline(0.0, color=INK, lw=.7)
        local.set(xlabel="$t$", ylabel=r"local rate $\gamma_{\rm loc}/\gamma-1$ [%]", xlim=(0, 50), ylim=(-1.0, 1.0))
        local.grid(True, which="major", alpha=.45)
        error.axhline(0.0, color=INK, lw=.7)
        error.set(xlabel="element unknowns per field", ylabel=r"$\gamma_h/\gamma-1$ [%] (plateau)")
        error.grid(True, which="major", alpha=.45)
        for ax, tag in zip((amplitude, local, error), "abc"):
            ax.set_title(f"({tag})", loc="left", fontsize=8)
        case_handles = [Line2D([], [], color=CASES[key][2], marker=CASES[key][3], lw=1.3, markersize=4.5,
                               label=CASES[key][0]) for key in cases]
        style_handles = [
            Line2D([], [], color=MUTED, lw=1.3, label="seeded run (finest mesh)"),
            Line2D([], [], color=MUTED, lw=.9, linestyle=":", label="unperturbed control"),
            Line2D([], [], color=INK, lw=.7, linestyle=(0, (4, 2)), label=r"$e^{\gamma t}$, eigenvalue problem"),
            Line2D([], [], color=MUTED, lw=2.4, label="modal plateau"),
            Line2D([], [], color=MUTED, lw=0, marker="o", markersize=4.5, label="IMEX-ARK3"),
            Line2D([], [], color=MUTED, lw=0, marker="o", markersize=4.5, markerfacecolor="white", label="SI-BDF2"),
        ]
        figure.legend(handles=case_handles, loc="outside upper center", ncol=len(case_handles), frameon=False,
                      fontsize=7)
        figure.legend(handles=style_handles, loc="outside lower center", ncol=len(style_handles), frameon=False,
                      fontsize=6.5)
        return save_publication_figure(figure, stem, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summaries", nargs="+", type=Path,
                        help="campaign summaries: smooth m=9 (d1), Zoni-Guclu (d3), high modes (hm64, hm128)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="diocotron_growth")
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()
    print(growth_figure(load_rows(args.summaries), args.output_dir / args.stem, dt=args.dt))


if __name__ == "__main__":
    main()
