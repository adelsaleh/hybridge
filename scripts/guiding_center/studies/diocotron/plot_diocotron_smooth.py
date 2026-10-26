"""Article figures and table of the smooth ring on the disk (D1 ladders, D2, D4); no time integration.

Reads the summaries written by ``run_diocotron_benchmark.py --benchmark disk-smooth``;
growth rates are the modal-plateau fits of
:func:`scripts.guiding_center.studies.diocotron.analyze_diocotron_campaign.analyse_row`.

``--ladder-table``
    booktabs table of the plateau rate error of the m = 9 ring (raw drift):
    (a) polynomial degree against band mesh for both schemes; (b) time step,
    Poisson stabilization tau and sign of the seed on the 36,334-triangle mesh. A
    seeded run that stopped in the nonlinear phase (recorded as a failure in the
    summary) is fitted on the linear phase it recorded and marked.
``<stem>_dispersion``
    growth rate against the mode number: the Nystrom dispersion relation of the
    ring continued to real m (:class:`RingDispersion`) with the measured rates,
    and the relative errors against the converged references, with the fits on
    the two halves of each window as bars.
``<stem>_snapshots`` and ``<stem>_conservation`` (``--nonlinear``)
    density of the long run at four times on one signed colour scale (no
    undershoot colour), and the histories of the relative drifts of mass and W,
    the decrease of ||rho_h||^2, the L1 excess ||rho_h||_1/M - 1 (twice the
    negative mass), min rho_h, the dissipation D_h and defect G_h of every SI-BDF2
    transport solve, and the relative defect of the exact L2 identity of SI-BDF2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hybridge.io.figures import draw_signed_density, publication_style, save_publication_figure  # noqa: E402
from scripts.guiding_center.studies.diocotron.analyze_diocotron_campaign import analyse_row  # noqa: E402
from scripts.guiding_center.studies.diocotron.plot_diocotron_highmode import (  # noqa: E402
    INK, MUTED, SCHEME_COLORS, SCHEME_LABELS, _records, _series)

SCHEMES = ("si-bdf2", "imex-ark3")
BANDS = (0.016, 0.008, 0.004)
REFERENCE_TAU = 128000.0


def _first_record(path) -> dict:
    with Path(path).open() as stream:
        return json.loads(stream.readline())


def load_rows(summaries) -> list[dict]:
    """Seeded and control rows of the summaries, plus seeded runs that stopped early.

    A failed seeded run whose record exists is added as a copy of a completed row
    of the same profile, with its own prefix, record, scheme, step and degree, and
    ``partial_end`` = its last recorded time.
    """
    rows = []
    for summary in summaries:
        data = json.loads(Path(summary).read_text())
        rows.extend(data["rows"])
        for failure in data.get("failures", []):
            prefix = failure.get("prefix", "")
            if not prefix.endswith("_seed") or not data["rows"]:
                continue
            path = Path(data["rows"][0]["jsonl"]).with_name(f"{prefix}.jsonl")
            if not path.exists():
                continue
            configuration = _first_record(path)["run_configuration"]
            label = next((row for row in data["rows"] if row["epsilon"] != 0.0), data["rows"][0])
            records = _records(path)
            rows.append(dict(label, prefix=prefix, jsonl=str(path), dt=float(configuration["dt"]),
                             scheme=configuration["time_scheme"], order=int(configuration["order"]),
                             partial_end=float(records[-1]["time"])))
    for row in rows:
        row["tau"] = float(_first_record(row["jsonl"])["run_configuration"]["poisson_tau_initial"])
    return rows


def plateau_error(row) -> dict:
    """Relative rate error on the modal plateau, with the two half-window errors.

    When the standard criterion (second harmonic below 0.5% of the target mode)
    leaves no plateau, it is retried with 1%: for slowly growing modes the seed's
    transient decays late and the harmonic 2m may itself be unstable. ``threshold``
    records the one used.
    """
    for threshold in (5e-3, 1e-2):
        fit = analyse_row(row, threshold=threshold)
        if fit["linear_window"] is not None:
            return dict(error=fit["linear_relative_error"], halves=fit["linear_half_relative_errors"],
                        window=fit["linear_window"], threshold=threshold)
    return dict(error=None, halves=[], window=None, threshold=None)


def _percent(value, digits=3, mark="") -> str:
    return "--" if value is None else f"${100 * value:+.{digits}f}\\%{mark}$"


def ladder_table(rows) -> str:
    """Booktabs bodies (a) degree x mesh and (b) one-parameter ladders, raw drift, mode 9."""
    seeded = [row for row in rows if row["epsilon"] != 0.0 and row["velocity"] == "raw"
              and row["label"].startswith("m9_")]
    errors = {}
    for row in seeded:
        key = (int(row["order"]), float(row["band_mesh_size"]), row["scheme"], round(float(row["dt"]), 6),
               row["tau"], float(np.sign(row["epsilon"])))
        errors[key] = (plateau_error(row)["error"], row.get("partial_end"))

    def cell(order=6, band=0.008, scheme="si-bdf2", dt=0.05, tau=REFERENCE_TAU, sign=1.0):
        error, partial_end = errors.get((order, band, scheme, dt, tau, sign), (None, None))
        return _percent(error, mark="^{*}" if partial_end is not None else "")

    triangles = {float(row["band_mesh_size"]): int(row["triangles"]) for row in seeded}
    lines = [r"\begin{tabular}{crrrrrr}\toprule",
             r"& \multicolumn{3}{c}{SI-BDF2} & \multicolumn{3}{c}{IMEX-ARK3}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             "$p$ & " + " & ".join(f"{triangles.get(band, 0):,}".replace(",", "{,}") for band in BANDS * 2)
             + r"\\\midrule"]
    for order in (4, 5, 6):
        lines.append(f"{order} & " + " & ".join(cell(order=order, band=band, scheme=scheme)
                                                for scheme in SCHEMES for band in BANDS) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    lines += [r"\par\medskip", r"\begin{tabular}{llrr}\toprule",
              r"parameter & value & SI-BDF2 & IMEX-ARK3\\\midrule"]
    for k, dt in enumerate((0.8, 0.4, 0.2, 0.1, 0.05)):
        lines.append(("$\\dt$" if k == 0 else "") + f" & ${dt:g}$ & "
                     + " & ".join(cell(scheme=scheme, dt=dt) for scheme in SCHEMES) + r"\\")
    lines.append(r"\midrule")
    for k, tau in enumerate((128.0, 1280.0, 12800.0, REFERENCE_TAU)):
        mantissa, exponent = f"{tau:.2e}".split("e")
        lines.append(("$\\tau$" if k == 0 else "") + f" & ${mantissa}\\times10^{{{int(exponent)}}}$ & "
                     + " & ".join(cell(scheme=scheme, tau=tau) for scheme in SCHEMES) + r"\\")
    lines.append(r"\midrule")
    for k, sign in enumerate((-1.0, 1.0)):
        lines.append(("$\\varepsilon$" if k == 0 else "") + f" & ${'-' if sign < 0 else '+'}10^{{-4}}$ & "
                     + " & ".join(cell(scheme=scheme, sign=sign) for scheme in SCHEMES) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines) + "\n"


def dispersion_figure(rows, stem, *, scheme="imex-ark3", band=0.008, curve=np.linspace(2.0, 15.8, 70),
                      radial_points=48):
    """Growth rate against the mode number: dispersion relation, measured rates and their errors."""
    from scripts.guiding_center.diagnostics.diocotron_reference import RingDispersion

    measured = {}
    for row in rows:
        if (row["epsilon"] > 0.0 and row["velocity"] == "raw" and row["scheme"] == scheme
                and float(row["band_mesh_size"]) == band and int(row["order"]) == 6
                and round(float(row["dt"]), 6) == 0.05 and row["tau"] == REFERENCE_TAU):
            mode = int(_first_record(row["jsonl"])["run_configuration"]["case_parameters"]["k"])
            fit = plateau_error(row)
            if fit["error"] is not None:
                measured[mode] = (row, fit)
    if not measured:
        raise ValueError("no seeded smooth-ring runs for the dispersion figure")
    parameters = _first_record(next(iter(measured.values()))[0]["jsonl"])["run_configuration"]["case_parameters"]
    dispersion = RingDispersion(parameters, radial_points=radial_points)
    rates = np.array([dispersion(m).imag for m in curve])
    modes = sorted(measured)
    reference = np.array([float(measured[m][0]["reference_gamma"]) for m in modes])
    gamma = np.array([reference[i] * (1.0 + measured[m][1]["error"]) for i, m in enumerate(modes)])
    errors = 100 * np.array([measured[m][1]["error"] for m in modes])
    halves = [100 * np.array(measured[m][1]["halves"], dtype=float) for m in modes]
    relaxed = [measured[m][1]["threshold"] != 5e-3 for m in modes]
    color = SCHEME_COLORS[scheme]
    with publication_style():
        figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
        left.plot(curve, rates, color=INK, lw=1.0, label="dispersion relation (Nyström)")
        left.plot(modes, gamma, "o", ms=5, color=color, mec="white", mew=.8,
                  label=f"HDG, {SCHEME_LABELS[scheme]}")
        left.set(xlabel="azimuthal mode $m$", ylabel=r"growth rate $\gamma$", xlim=(1.5, 16.2), ylim=(0, None))
        left.legend(frameon=False, loc="lower center")
        left.grid(True, alpha=.5)
        right.axhline(0.0, color=MUTED, lw=.6)
        for m, value, half, flag in zip(modes, errors, halves, relaxed):
            if len(half):
                right.plot([m, m], [half.min(), half.max()], color=color, lw=1.2, alpha=.45)
            right.plot(m, value, "o", ms=5, color=color, mfc="white" if flag else color, mec=color, mew=1.0)
        right.set(xlabel="azimuthal mode $m$", ylabel=r"$\gamma_h/\gamma-1$ (%)", xlim=(1.5, 16.2))
        right.grid(True, alpha=.5)
        return save_publication_figure(figure, stem, dpi=300)


def snapshot_figure(npz_path, stem, *, times=(40., 50., 60., 150.), pixels=900, half_width=0.62):
    """Density at four saved times (2 x 2), cropped to the ring, on one signed colour scale."""
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from hybridge.io.raster import RasterGeometry

    npz = np.load(npz_path)
    saved = [round(float(t), 6) for t in npz["times"]]
    mesh = DGMesh(npz["node_coords"], npz["triangles"])
    space = DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]))
    centres = -half_width + (np.arange(pixels) + .5) * 2 * half_width / pixels
    xx, yy = np.meshgrid(centres, centres[::-1])
    bounds = (-half_width, half_width, -half_width, half_width)
    geometry = RasterGeometry.from_points(mesh, np.column_stack((xx.ravel(), yy.ravel())), width=pixels,
                                          height=pixels, bounds=bounds)
    matrix = geometry.sampling_matrix(space)
    mask = (geometry.element_ids < 0).reshape(pixels, pixels)
    density = npz["density"]
    with publication_style():
        figure, axes = plt.subplots(2, 2, figsize=(5.0, 5.35), layout="constrained")
        panels = []
        for ax, t in zip(axes.ravel(), times):
            values = (matrix @ np.asarray(density[saved.index(round(t, 6))]).ravel()).reshape(pixels, pixels)
            panels.append((ax, np.ma.array(values, mask=mask), bounds))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            ax.set_title(f"$t={t:g}$")
        artist = draw_signed_density(panels, 1.0)
        figure.colorbar(artist, ax=axes, location="bottom", shrink=.7, aspect=35, pad=.02,
                        label=r"density $\rho_h$")
        return save_publication_figure(figure, stem, dpi=300)


def _running_max(values):
    return np.fmax.accumulate(np.where(np.isfinite(values), values, 0.0))


def conservation_figure(rows, stem):
    """Histories of the D4 runs (both schemes) and the SI-BDF2 enstrophy terms."""
    runs = sorted(((row["scheme"], float(row["dt"]), _records(row["jsonl"])) for row in rows),
                  key=lambda item: item[0])
    with publication_style(font_size=8.0):
        figure, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), layout="constrained")
        (mass, energy, l2, l1), (minimum, terms, identity, legend_ax) = axes
        for scheme, dt, records in runs:
            color, label = SCHEME_COLORS[scheme], SCHEME_LABELS[scheme]
            t = _series(records, "time")
            mass.semilogy(t[1:], np.maximum(_running_max(np.abs(_series(records, "mass_relative_drift")))[1:],
                                            1e-17), color=color, lw=1.0, label=label)
            energy.semilogy(t[1:], np.maximum(_running_max(np.abs(_series(records, "energy_relative_drift")))[1:],
                                              1e-17), color=color, lw=1.0)
            l2.semilogy(t[1:], np.maximum(-_series(records, "enstrophy_relative_drift")[1:], 1e-17), color=color,
                        lw=1.0)
            minimum.plot(t, _series(records, "rho_min"), color=color, lw=.9)
            l1_norm = _series(records, "rho_l1")
            if np.isfinite(l1_norm).any():
                excess = l1_norm / _series(records, "mass") - 1.0
                l1.semilogy(t, np.maximum(excess, 1e-17), color=color, lw=1.0)
            dissipation = _series(records, "enstrophy_dissipation")
            if np.isfinite(dissipation).any():
                norm0 = 2.0 * float(records[0]["enstrophy"])  # ||rho_h^0||^2
                defect = _series(records, "enstrophy_defect")
                terms.semilogy(t, 4 * dt * dissipation / norm0, color=INK, lw=.9,
                               label=r"$4\Delta t\,\mathcal{D}_h$")
                terms.semilogy(t, np.where(defect > 0, 2 * dt * defect / norm0, np.nan), color=color, lw=.0,
                               marker=".", ms=1.6, label=r"$2\Delta t\,\mathcal{G}_h>0$")
                terms.semilogy(t, np.where(defect < 0, -2 * dt * defect / norm0, np.nan), color=MUTED, lw=.0,
                               marker=".", ms=1.6, label=r"$-2\Delta t\,\mathcal{G}_h>0$")
                identity.semilogy(t[1:], np.maximum(_running_max(np.abs(
                    _series(records, "l2_budget_relative_defect")))[1:], 1e-19), color=color, lw=1.0)
        mass.set(title="mass", ylabel="running max. rel. drift")
        energy.set(title=r"$W=\int|E_h|^2$")
        l2.set(title=r"decrease of $\|\rho_h\|^2$")
        l1.set(title=r"$\|\rho_h\|_{L^1}/M-1$")
        minimum.set(title=r"$\min\rho_h$ (quadrature)", xlabel="$t$")
        terms.set(title=r"SI-BDF2 terms $/\|\rho_h^0\|^2$", xlabel="$t$")
        identity.set(title="SI-BDF2 identity defect", xlabel="$t$", ylim=(1e-17, 1e-14))
        terms.legend(frameon=False, fontsize=6, loc="lower right", markerscale=4)
        for ax in (mass, energy, l2, l1, minimum, terms, identity):
            ax.grid(True, which="major", alpha=.5)
        handles, labels = mass.get_legend_handles_labels()
        legend_ax.axis("off")
        legend_ax.legend(handles, labels, frameon=False, loc="center")
        return save_publication_figure(figure, stem, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summaries", nargs="+", type=Path, help="disk-smooth campaign summaries (D1, D2)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="diocotron_smooth")
    parser.add_argument("--ladder-table", type=Path, help="write the D1 ladder table")
    parser.add_argument("--dispersion", action="store_true", help="write the D2 dispersion figure")
    parser.add_argument("--nonlinear", type=Path, help="summary of the long runs (D4)")
    parser.add_argument("--snapshot-times", type=lambda text: tuple(float(v) for v in text.split(",")),
                        default=(40., 50., 60., 150.))
    parser.add_argument("--snapshot-scheme", default="imex-ark3")
    args = parser.parse_args()
    rows = load_rows(args.summaries)
    if args.ladder_table is not None:
        args.ladder_table.write_text("% Generated by scripts/guiding_center/studies/diocotron/plot_diocotron_smooth.py\n"
                                     + ladder_table(rows))
        print(args.ladder_table)
    if args.dispersion:
        print(dispersion_figure(rows, args.output_dir / f"{args.stem}_dispersion"))
    if args.nonlinear is not None:
        long_rows = [row for row in load_rows([args.nonlinear]) if row["epsilon"] != 0.0]
        print(conservation_figure(long_rows, args.output_dir / f"{args.stem}_conservation"))
        for row in long_rows:
            npz = Path(row["jsonl"]).with_name(f"{row['prefix']}_snapshots.npz")
            if row["scheme"] == args.snapshot_scheme and npz.exists():
                print(snapshot_figure(npz, args.output_dir / f"{args.stem}_snapshots", times=args.snapshot_times))


if __name__ == "__main__":
    main()
