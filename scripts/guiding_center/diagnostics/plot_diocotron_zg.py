"""Article figures and table for the Zoni--Guclu diocotron reproduction (D3); no time integration.

Reads the summary of ``run_diocotron_benchmark.py --benchmark disk-zg`` and the run records:

``<stem>_growth``
    left: ||phi_h - phi_eq|| (the quantity fitted by Zoni--Guclu) and the mode-9
    amplitude against time on the finest band mesh, with the exact linear rate of
    the truncated profile and the sharp-band rate of their eq. (34); right: relative
    error of the fitted rate against the band mesh size.
``<stem>_conservation``
    relative drifts of the mass and of W = int |E|^2 on [0, 70], with the maxima
    reported by Zoni--Guclu (their Fig. 8).
``<stem>_snapshots``
    density, potential and |E|^2 = |q_h|^2 at t = 0, 50, 70 (their Fig. 7 shows
    t = 0 and 50). t = 0 is rebuilt offline: the same conforming projection of the
    initial density and one host HDG Poisson solve (Numba + PyPardiso).
``--table``
    booktabs rows: band mesh, triangles, unknowns, step time, fitted rates and
    their errors, maximum drifts, minimum density.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.guiding_center.diagnostics.run_records import records as _records, series as _series  # noqa: E402
from hybridge.io.figures import publication_style, save_publication_figure, signed_density_colors  # noqa: E402
from scripts.guiding_center.diagnostics.analyze_diocotron_campaign import analyse_row  # noqa: E402
from scripts.guiding_center.diagnostics.diocotron_reference import fit_growth  # noqa: E402

# Categorical slots (validated reference palette): scheme identity; dt by line style.
SCHEME_COLORS = {"si-bdf2": "#2a78d6", "imex-ark3": "#eb6834"}
SCHEME_LABELS = {"si-bdf2": "SI-BDF2", "imex-ark3": "IMEX-ARK3"}
INK, MUTED = "#0b0b0b", "#52514e"
DT_STYLES = {0.1: ("--", "^"), 0.05: ("-", "o")}
ZG_MAX_MASS_DRIFT, ZG_MAX_ENERGY_DRIFT = 5.8e-4, 1.8e-3
ZG_WINDOW = (20.0, 50.0)


def _seeded(rows, velocity="raw"):
    return [row for row in rows if row["velocity"] == velocity and row["epsilon"] != 0.0]


def _style_for(row):
    linestyle, marker = DT_STYLES.get(round(float(row["dt"]), 6), ("-.", "s"))
    return dict(color=SCHEME_COLORS.get(row["scheme"], MUTED), linestyle=linestyle, marker=marker)


def _label(row):
    return f"{SCHEME_LABELS.get(row['scheme'], row['scheme'])}, $\\Delta t={row['dt']:g}$"


def _local_rates(records, mode=9, width=4.0):
    """Centered log-amplitude slopes of the target mode over ``width`` time units."""
    t = _series(records, "time")
    log_amplitude = np.log(_series(records, f"diocotron_phi_mode_{mode}_l2"))
    step = int(round(width / np.median(np.diff(t))))
    rates = (log_amplitude[step:] - log_amplitude[:-step]) / (t[step:] - t[:-step])
    return .5 * (t[step:] + t[:-step]), rates


def growth_figure(rows, controls, stem):
    finest = min(row["band_mesh_size"] for row in rows)
    chosen = [row for row in rows if row["band_mesh_size"] == finest]
    reference = chosen[0]["reference_gamma"]
    sharp = chosen[0]["sharp_reference_gamma"]
    with publication_style():
        figure, (left, middle, right) = plt.subplots(1, 3, figsize=(7.2, 2.85), layout="constrained")
        for ax in (left, middle):
            ax.axvspan(*ZG_WINDOW, color="#eceae4", lw=0, zorder=0)
        for row in chosen:
            records = _records(row["jsonl"])
            style = _style_for(row)
            left.semilogy(_series(records, "time"), _series(records, "diocotron_phi_eq_l2"), color=style["color"],
                          linestyle=style["linestyle"], lw=1.2, label=_label(row))
            times, rates = _local_rates(records)
            middle.plot(times, 100 * (rates / reference - 1), color=style["color"], linestyle=style["linestyle"],
                        lw=1.1, label=_label(row))
        control = controls.get(finest)
        if control is not None:
            records = _records(control)
            left.semilogy(_series(records, "time"), _series(records, "diocotron_phi_eq_l2"), color=MUTED,
                          lw=.8, linestyle=":", label="unperturbed control")
        anchor = next(row for row in chosen if row["scheme"] == "imex-ark3")
        records = _records(anchor["jsonl"])
        t, a = _series(records, "time"), _series(records, "diocotron_phi_eq_l2")
        window = anchor.get("linear_window") or ZG_WINDOW
        # Slope of the eigenvalue problem, anchored at the middle of the modal plateau.
        centre = .5 * (window[0] + window[1])
        level = np.interp(centre, t, np.log(a))
        x = np.linspace(0., 60., 200)
        left.semilogy(x, np.exp(level + reference * (x - centre)), color=INK, lw=.9, linestyle=(0, (4, 2)),
                      label=f"eigenvalue problem, $\\gamma={reference:.5f}$")
        left.set(xlabel="$t$", ylabel=r"$\|\phi_h-\phi_h^{\rm eq}\|_{L^2}$", xlim=(0, 70))
        handles, labels = left.get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, fontsize=6.5)
        left.grid(True, which="major", alpha=.5)
        middle.axhline(0, color=INK, lw=.8)
        middle.axhline(100 * (sharp / reference - 1), color=INK, lw=.8, linestyle=(0, (1, 2)))
        middle.annotate("eq. (34)", (59, 100 * (sharp / reference - 1)), xytext=(0, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=7, color=INK)
        middle.set(xlabel="$t$", ylabel=r"local rate $\gamma_{\rm loc}(t)/\gamma-1$ [%]", xlim=(0, 60),
                   ylim=(-1.5, 1.0))
        middle.grid(True, which="major", alpha=.5)
        for key in sorted({(row["scheme"], row["dt"]) for row in rows}):
            series = sorted((row for row in rows if (row["scheme"], row["dt"]) == key),
                            key=lambda row: row["band_mesh_size"])
            sizes = [row["band_mesh_size"] for row in series]
            style = _style_for(series[0])
            plateau = [100 * (row["linear_relative_error"] if row.get("linear_relative_error") is not None
                              else np.nan) for row in series]
            declared = [100 * row["declared_relative_error"] for row in series]
            right.semilogx(sizes, plateau, **style, lw=1.2, markersize=4, label=_label(series[0]))
            right.semilogx(sizes, declared, color=style["color"], linestyle=style["linestyle"], lw=.7,
                           marker=style["marker"], markersize=4, markerfacecolor="white")
        right.axhline(0, color=INK, lw=.8)
        right.axhline(100 * (sharp / reference - 1), color=INK, lw=.8, linestyle=(0, (1, 2)))
        sizes = sorted({row["band_mesh_size"] for row in rows})
        right.set_xticks(sizes, [f"{size:g}" for size in sizes])
        right.minorticks_off()
        right.invert_xaxis()
        right.set(xlabel="band mesh size $h_{\\rm band}$", ylabel=r"$\gamma_h/\gamma-1$ [%]")
        # Scheme and time step are identified by the shared legend of the other panels.
        right.annotate("filled: modal plateau\nopen: window [20, 50]", (.04, .44), xycoords="axes fraction",
                       fontsize=6.5, color=MUTED, va="center")
        right.annotate("eq. (34)", (sizes[0], 100 * (sharp / reference - 1)), xytext=(0, 3),
                       textcoords="offset points", ha="right", va="bottom", fontsize=7, color=INK)
        right.grid(True, which="major", alpha=.5)
        return save_publication_figure(figure, stem)


def conservation_figure(rows, stem):
    finest = min(row["band_mesh_size"] for row in rows)
    chosen = [row for row in rows if row["band_mesh_size"] == finest]
    with publication_style():
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True, sharex=True)
        for row in chosen:
            records = _records(row["jsonl"])
            t = _series(records, "time")
            style = _style_for(row)
            for ax, key in zip(axes, ("mass_relative_drift", "energy_relative_drift")):
                # Running maximum: the quantity Zoni--Guclu report, and readable at roundoff level.
                drift = np.maximum.accumulate(np.nan_to_num(np.abs(_series(records, key))))
                ax.semilogy(t[1:], np.maximum(drift[1:], 1e-16), color=style["color"],
                            linestyle=style["linestyle"], lw=1.1, label=_label(row))
        for ax, value, name in zip(axes, (ZG_MAX_MASS_DRIFT, ZG_MAX_ENERGY_DRIFT),
                                   (r"$\max_{s\leq t}\,\delta M(s)$",
                                    r"$\max_{s\leq t}\,\delta W(s),\ W=\int|E|^2$")):
            ax.axhline(value, color=INK, lw=.8, linestyle=(0, (4, 2)))
            ax.annotate("Zoni–Güçlü, max on [0, 70]", (69, value), xytext=(0, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=7, color=INK)
            ax.set(xlabel="$t$", ylabel=name, xlim=(0, 70))
            ax.set_ylim(bottom=1e-16)
            ax.grid(True, which="major", alpha=.5)
        axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
        return save_publication_figure(figure, stem)


def _initial_density(npz, record, *, equilibrium=False):
    """DG space of a snapshot file and the initial (or equilibrium) density projected as in the run (host).

    ``record`` is the ``run_configuration`` of the run's first JSONL record. No solve: the
    projection of the case density, with the exact elementwise indicator on conforming bands.
    """
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
    from scripts.guiding_center.runtime.runner import _project_case_density

    mesh = DGMesh(npz["node_coords"], npz["triangles"])
    space = DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]))
    parameters = {key: value for key, value in record["case_parameters"].items()
                  if key not in {"conforming", "geometry", "s_bar", "s_d"}}
    parameters["modes"] = tuple(parameters.get("modes") or (parameters["k"],))
    case = case_definition_by_key("diocotron_annulus").build(**parameters)
    config = SimpleNamespace(poisson_assembly_backend="numba", transport_assembly_backend="numba",
                             initial_projection_quad_1d=record.get("initial_projection_quad_1d"))
    return space, _project_case_density(config, space, case, equilibrium=equilibrium, name="rho0")


def _initial_state(npz, row, tau, *, equilibrium=False):
    """Projected initial (or equilibrium) density and its HDG potential and flux, as in the run (host solve)."""
    from hybridge.linalg.pardiso_runtime import pardiso_thread_limit
    from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGOptions, DiffusionReactionHDGSolver

    space, density = _initial_density(npz, _records(row["jsonl"])[0]["run_configuration"],
                                      equilibrium=equilibrium)
    # Direct solve: a relative residual of ~1e-12 is roundoff for this system.
    options = DiffusionReactionHDGOptions(stabilization=float(tau), solver="pypardiso", preconditioner=None,
                                          solver_rtol=1e-10, assembly_backend="numba", boundary_mode="eliminate",
                                          scale_system=False, verbose=0)
    with pardiso_thread_limit(16):
        result = DiffusionReactionHDGSolver(space, source=density, reaction=space.zeros(),
                                            boundary_condition=0.0, options=options).solve()
    flux = np.stack([np.asarray(component.coeffs) for component in result.flux.components])
    return space, np.asarray(density.coeffs), np.asarray(result.field.coeffs), flux


def snapshot_figure(row, npz_path, stem, *, tau, times=(0.0, 50.0, 70.0), pixels=700):
    from hybridge.io.raster import RasterGeometry

    npz = np.load(npz_path)
    space, density0, potential0, flux0 = _initial_state(npz, row, tau)
    geometry = RasterGeometry.from_mesh(space.mesh, pixels, pixels)
    matrix = geometry.sampling_matrix(space)
    mask = (geometry.element_ids < 0).reshape(geometry.height, geometry.width)

    def raster(coefficients):
        values = matrix @ np.asarray(coefficients).ravel()
        return np.ma.array(values.reshape(geometry.height, geometry.width), mask=mask)

    states = {0.0: (density0, potential0, flux0)}
    for index, t in enumerate(npz["times"]):
        states[round(float(t), 6)] = (npz["density"][index], npz["potential"][index], npz["flux"][index])
    import matplotlib as mpl

    images = {}
    for t in times:
        density, potential, flux = states[round(t, 6)]
        images[t] = (raster(density), raster(potential), raster(flux[0]) ** 2 + raster(flux[1]) ** 2)
    density_map, density_norm = signed_density_colors(min(float(images[t][0].min()) for t in times), 1.0)
    columns = [("density $\\rho_h$", density_map), ("potential $\\phi_h$", "cividis"),
               ("$|E_h|^2=|q_h|^2$", "magma")]
    limits = [density_norm, mpl.colors.Normalize(0.0, max(float(images[t][1].max()) for t in times)),
              mpl.colors.Normalize(0.0, max(float(images[t][2].max()) for t in times))]
    xmin, xmax, ymin, ymax = geometry.bounds
    with publication_style():
        figure, axes = plt.subplots(len(times), 3, figsize=(6.6, 2.15 * len(times)), constrained_layout=True)
        for i, t in enumerate(times):
            for j, ((title, cmap), norm) in enumerate(zip(columns, limits)):
                ax = axes[i, j]
                artist = ax.imshow(images[t][j], extent=(xmin, xmax, ymin, ymax), origin="upper", cmap=cmap,
                                   norm=norm, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if i == 0:
                    ax.set_title(title)
                if j == 0:
                    ax.set_ylabel(f"$t={t:g}$")
                if i == len(times) - 1:
                    figure.colorbar(artist, ax=axes[:, j], shrink=.6, location="bottom", pad=.02)
        return save_publication_figure(figure, stem, dpi=300)


def _minimum_density(row, t_max):
    records = [r for r in _records(row["jsonl"]) if r["time"] <= t_max + 1e-9]
    values = [r.get("rho_min_checked", r.get("rho_min")) for r in records]
    values = [float(v) for v in values if v is not None]
    return min(values) if values else None


def _percent(value, digits) -> str:
    """Signed percentage in math mode (a true minus sign)."""
    return "--" if value is None else f"${100 * value:+.{digits}f}\\%$"


def _power(value) -> str:
    """$a\\times10^{b}$ with two significant digits."""
    if value is None:
        return "--"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def latex_table(rows) -> str:
    """Booktabs table body (raw drift) with the Zoni--Guclu reference line.

    Columns: band mesh, element unknowns per field (Zoni--Guclu: spline coefficients),
    scheme, dt, median wall time per step, plateau and [20, 50] potential-norm rate
    errors, maximum relative mass and W drifts on [0, 70], and the sampled minimum
    density (quadrature points and an order-(2p+2) lattice per element) for t <= 50 / 70.
    """
    lines = [r"\setlength{\tabcolsep}{4.5pt}\begin{tabular}{rrlrrrrrrc}\toprule",
             r" & & & & & \multicolumn{2}{c}{$\gamma_h/\gamma-1$} & & & $\min\rho_h$\\\cmidrule(lr){6-7}",
             r"$h_{\rm band}$ & unknowns & scheme & $\Delta t$ & ms/step & plateau & "
             r"$\|\phi_h-\phi_h^{\rm eq}\|$ & $\max\delta M$ & $\max\delta W$ & "
             r"$t\le50$ / $70$\\\midrule"]
    for row in sorted(rows, key=lambda r: (-r["band_mesh_size"], r["scheme"], -r["dt"])):
        error = row.get("linear_relative_error")
        if error is None:
            error = row["relative_error"]
        step = row.get("median_linear_step_wall_time")
        invariants = row.get("invariants") or {}
        minima = [_minimum_density(row, 50.), _minimum_density(row, 70.)]
        lines.append(
            f"{row['band_mesh_size']:g} & {row['element_unknowns'] / 1e6:.2f}M & "
            f"{SCHEME_LABELS[row['scheme']]} & {row['dt']:g} & "
            + ("--" if step is None else f"{1e3 * step:.0f}") + " & "
            + _percent(error, 3) + " & " + _percent(row.get("potential_norm_relative_error"), 2) + " & "
            + _power(invariants.get("mass")) + " & " + _power(invariants.get("energy_from_q_l2")) + " & "
            + " / ".join("--" if value is None else f"${value:.2f}$" for value in minima) + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\cite{ZONI2019108889} & 0.03M & SL, explicit & 0.1 & -- & -- & -- & "
                 r"$5.8\times10^{-4}$ & $1.8\times10^{-3}$ & --\\")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines).replace(",", "{,}") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summary", type=Path, help="d3 campaign summary JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="diocotron_zg")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--poisson-tau", type=float, default=32000.0)
    parser.add_argument("--snapshot-run", help="prefix of the run for the snapshot figure "
                                                "(default: finest mesh, IMEX-ARK3, smallest dt)")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    rows = [analyse_row(row, threshold=5e-3) for row in _seeded(summary["rows"])]
    directory = args.summary.parent
    controls = {}
    for path in sorted(directory.glob(args.summary.name.replace("_summary.json", "_*_control.jsonl"))):
        for row in rows:
            if path.name.startswith(row["prefix"].rsplit("_", 2)[0]):
                controls.setdefault(row["band_mesh_size"], path)
    print(growth_figure(rows, controls, args.output_dir / f"{args.stem}_growth"))
    print(conservation_figure(rows, args.output_dir / f"{args.stem}_conservation"))
    finest = min(row["band_mesh_size"] for row in rows)
    candidates = [row for row in rows if row["band_mesh_size"] == finest and row["scheme"] == "imex-ark3"]
    target = (next(row for row in rows if row["prefix"] == args.snapshot_run) if args.snapshot_run
              else min(candidates, key=lambda row: row["dt"]))
    npz = directory / f"{target['prefix']}_snapshots.npz"
    if npz.exists():
        print(snapshot_figure(target, npz, args.output_dir / f"{args.stem}_snapshots", tau=args.poisson_tau))
    if args.table is not None:
        args.table.write_text("% Generated by scripts/guiding_center/diagnostics/plot_diocotron_zg.py\n"
                              + latex_table(rows))
        print(args.table)


if __name__ == "__main__":
    main()
