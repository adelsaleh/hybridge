"""Article figures for the ionization diocotron runs (S = nu(x) rho); no time integration.

Reads the run records of ``run_diocotron_benchmark.py --benchmark disk-ionization``:

``<stem>_checks`` (``--column``, ``--map``, ``--reference``)
    left, the gas column without a seed: relative L2 distance to the exact
    density rho0(r) exp(nu(r) t), one curve per time step; right, the uniform
    gas: amplitude of the mode 9 of the forced run, of the unforced reference
    run and of the exact map exp(nu t) A_ref((exp(nu t) - 1)/nu).
``<stem>_map_frames`` (``--map-snapshots``, ``--reference-snapshots``)
    density of the unforced run at s and exp(-nu t) rho of the forced run at
    t(s) = log(1 + nu s)/nu, from the saved snapshots.
``<stem>_puff`` (``--puff``)
    density snapshots of the gas-puff runs with the cloud's 1/e contour, the
    ionization rate int nu rho (a synthetic gas-puff-imaging signal) and the
    mass growth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hybridge.io.figures import draw_signed_density, publication_style, save_publication_figure  # noqa: E402
from scripts.guiding_center.diagnostics.run_records import (  # noqa: E402
    image as _image, raster as _raster, records as _records, series as _series)

INK, MUTED = "#0b0b0b", "#52514e"
SERIES = ("#2a5b8c", "#c4622d", "#3f8f5a", "#8a5aa8")


def _configuration(records) -> dict:
    """Case parameters of a run (the initial record)."""
    for record in records:
        configuration = record.get("run_configuration")
        if configuration:
            return configuration
    raise ValueError("run without a recorded configuration")


def _ionization(configuration) -> dict:
    """The run's gas record: rate, center, width (None: uniform), start, ramp; None without gas."""
    return configuration["case_parameters"].get("ionization")


def _case(configuration):
    """Rebuild the run's case from its recorded parameters (profile, seed and gas)."""
    from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key

    recorded = configuration["case_parameters"]
    keys = ("k", "phase_scale", "epsilon", "s_minus", "s_plus", "rho_bar", "edge_width", "s_bar", "s_d", "p",
            "theta_shift", "truncate", "inner_wall", "wall")
    gas = _ionization(configuration)
    return case_definition_by_key("diocotron_annulus").build(
        **{key: recorded[key] for key in keys if key in recorded}, modes=tuple(recorded["modes"]),
        ionization_rate=gas["rate"], ionization_center=tuple(gas["center"]), ionization_width=gas["width"],
        ionization_start=gas["start"], ionization_ramp=gas["ramp"])


def _exact_norm(case, times):
    """||rho0 exp(int nu)||_L2 of an axisymmetric gas, by radial quadrature."""
    from scipy.integrate import quad

    def norm(t):
        integrand = lambda r: 2 * np.pi * r * float(case.exact_density(np.array([r]), np.array([0.0]), t)[0]) ** 2
        return np.sqrt(quad(integrand, 0.0, 1.0, points=[.35, .45, .475, .5, .6], limit=400)[0])

    return np.array([norm(t) for t in times])


def checks_figure(column_paths, map_path, reference_path, stem, *, mode=9):
    """Exact gas-column distance per time step (left) and the uniform-gas map (right)."""
    with publication_style():
        figure, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
        runs = sorted((_records(path) for path in column_paths), key=lambda records: -_configuration(records)["dt"])
        for color, records in zip(SERIES, runs):
            configuration = _configuration(records)
            records = [r for r in records if r.get("rho_l2_error") is not None]
            t, error = _series(records, "time"), _series(records, "rho_l2_error")
            sample = np.linspace(0.0, t[-1], 41)
            relative = error / np.interp(t, sample, _exact_norm(_case(configuration), sample))
            left.semilogy(t, relative, color=color, lw=1.1, label=f"$\\Delta t={configuration['dt']:g}$")
        left.set(xlabel="$t$", ylabel=r"$\|\rho_h-\rho_0\,e^{\nu(r)t}\|/\|\rho_0\,e^{\nu(r)t}\|$")
        left.legend(frameon=False, fontsize=7)
        left.grid(True, which="major", alpha=.5)
        if map_path is not None and reference_path is not None:
            forced, reference = _records(map_path), _records(reference_path)
            nu = float(_ionization(_configuration(forced))["rate"])
            key = f"diocotron_phi_mode_{mode}_l2"
            t, amplitude = _series(forced, "time"), _series(forced, key)
            t_ref, amplitude_ref = _series(reference, "time"), _series(reference, key)
            right.semilogy(t_ref, amplitude_ref, color=MUTED, lw=1.0, label="no gas")
            right.semilogy(t, amplitude, color=SERIES[0], lw=1.1, label=f"uniform gas, $\\nu={nu:g}$")
            # markers only where the reference run reaches s(t)
            marks = np.linspace(2.0, min(t[-1], np.log1p(nu * t_ref[-1]) / nu), 28)
            mapped = np.exp(nu * marks) * np.interp(np.expm1(nu * marks) / nu, t_ref, amplitude_ref)
            right.semilogy(marks, mapped, "o", ms=3.2, mfc="none", color=SERIES[1],
                           label=r"$e^{\nu t}A_{\mathrm{no\,gas}}((e^{\nu t}-1)/\nu)$")
            right.set(xlabel="$t$", ylabel=f"$\\|\\hat\\varphi_{{{mode}}}\\|$", xlim=(0, t[-1]))
            right.legend(frameon=False, fontsize=7, loc="lower right")
            right.grid(True, which="major", alpha=.5)
        return save_publication_figure(figure, stem, dpi=300)


def map_frames_figure(map_npz, reference_npz, stem, *, nu, s_values=(70., 100., 150., 200., 250.), pixels=420):
    """Unforced density at s (top) and exp(-nu t) rho_h at t(s) of the uniform-gas run (bottom)."""
    forced, reference = np.load(map_npz), np.load(reference_npz)
    geometry, matrix, mask = _raster(reference, pixels)
    with publication_style():
        figure, axes = plt.subplots(2, len(s_values), figsize=(1.32 * len(s_values) + .9, 3.0),
                                    layout="constrained", squeeze=False)
        panels = []
        for j, s in enumerate(s_values):
            k_ref = int(np.argmin(np.abs(reference["times"] - s)))
            t_target = np.log1p(nu * s) / nu
            k = int(np.argmin(np.abs(forced["times"] - t_target)))
            t = float(forced["times"][k])
            ref = (matrix @ np.asarray(reference["density"][k_ref]).ravel()).reshape(geometry.height, geometry.width)
            values = np.exp(-nu * t) * (matrix @ np.asarray(forced["density"][k]).ravel()).reshape(
                geometry.height, geometry.width)
            panels += [_image(axes[0, j], ref, geometry, mask), _image(axes[1, j], values, geometry, mask)]
            axes[0, j].set_title(f"$s={float(reference['times'][k_ref]):g}$", fontsize=8)
            axes[1, j].set_title(f"$t={t:.2f}$", fontsize=8)
        axes[0, 0].set_ylabel("no gas, $\\rho_h(s)$", fontsize=7.5)
        axes[1, 0].set_ylabel(f"$\\nu={nu:g}$, $e^{{-\\nu t}}\\rho_h(t)$", fontsize=7.5)
        artist = draw_signed_density(panels, 1.0, over="#fde725")
        figure.colorbar(artist, ax=axes, location="right", shrink=.75, extend="max")
        return save_publication_figure(figure, stem, dpi=300)



def _label(parameters) -> str:
    return "no gas" if parameters is None else f"$\\nu_0={parameters['rate']:g}$"

def puff_figure(paths, stem, *, times=(70., 100., 150., 200., 300., 400.), pixels=420, labels=None, radius=.475,
                rate_window=(230., 330.)):
    """Density snapshots with the gas cloud's 1/e contour (one row per run); ionization rate, mass and
    symmetry breaking of the crystal of nine vortices, with the point-vortex (Thomson) rate drawn on the
    runs without gas over ``rate_window``."""
    from matplotlib.patches import Circle

    from scripts.guiding_center.diagnostics.diocotron_reference import point_vortex_ring_growth
    from scripts.guiding_center.studies.diocotron.plot_diocotron_injection import MODE, _symmetry_breaking

    runs = []
    for path in paths:
        records = [r for r in _records(path) if "mass" in r]
        runs.append((Path(path), records, _ionization(_configuration(_records(path)))))
    with publication_style():
        figure = plt.figure(figsize=(7.2, 1.25 * len(runs) + 2.3), layout="constrained")
        outer = figure.add_gridspec(2, 1, height_ratios=[1.25 * len(runs), 2.0])
        images = outer[0].subgridspec(len(runs), len(times))
        panels = []
        for i, (path, records, parameters) in enumerate(runs):
            npz_path = path.with_name(path.stem + "_snapshots.npz")
            if npz_path.exists():
                npz = np.load(npz_path)
                geometry, matrix, mask = _raster(npz, pixels)
                saved = [round(float(t), 6) for t in npz["times"]]
            for j, t in enumerate(times):
                ax = figure.add_subplot(images[i, j])
                if npz_path.exists() and round(t, 6) in saved:
                    values = (matrix @ np.asarray(npz["density"][saved.index(round(t, 6))]).ravel()).reshape(
                        geometry.height, geometry.width)
                    panels.append(_image(ax, values, geometry, mask))
                    if parameters is not None and parameters["width"] is not None and t >= parameters["start"]:
                        center, width = parameters["center"], parameters["width"]
                        ax.add_patch(Circle(tuple(center), width, fill=False, ec="white", lw=.7, ls=(0, (2, 1.5))))
                else:
                    ax.set_axis_off()
                if i == 0:
                    ax.set_title(f"$t={t:g}$", fontsize=8)
                if j == 0:
                    ax.set_ylabel(labels[i] if labels else _label(parameters), fontsize=7.5)
        artist = draw_signed_density(panels, 1.5, over="#fde725")
        if artist is not None:
            figure.colorbar(artist, ax=figure.axes, location="right", shrink=.5, extend="max", label=r"$\rho_h$")
        signal, mass, breaking = (figure.add_subplot(cell) for cell in outer[1].subgridspec(1, 3))
        for color, (path, records, parameters), i in zip(SERIES, runs, range(len(runs))):
            t, m = _series(records, "time"), _series(records, "mass")
            label = labels[i] if labels else _label(parameters)
            if parameters is not None:
                production = _series(records, "reaction_production")
                signal.plot(t, production / (float(parameters["rate"]) * m[0]), color=color, lw=.7, label=label)
            mass.plot(t, m / m[0], color=color, lw=1.1, label=label)
            breaking.semilogy(t, _symmetry_breaking(records), color=color, lw=1.0, label=label)
        # Ring of nine point vortices, no centre: Thomson-unstable at the rate of point_vortex_ring_growth.
        circulation = _series(runs[0][1], "mass")[0] / MODE
        rate = point_vortex_ring_growth(MODE, 0.0, radius=radius, circulation=circulation)
        for color, (path, records, parameters) in zip(SERIES, runs):
            if parameters is not None:  # the gas forces the symmetry breaking; compare the free crystal only
                continue
            t, ratio = _series(records, "time"), _symmetry_breaking(records)
            anchor = rate_window[0]
            line = np.linspace(anchor, rate_window[1], 2)
            level = np.interp(anchor, t, np.log(ratio))
            breaking.semilogy(line, np.exp(level + rate * (line - anchor)), color=INK, lw=.8, ls=(0, (4, 2)))
        breaking.plot([], [], color=INK, lw=.8, ls=(0, (4, 2)), label=f"point vortices, {rate:.3f}")
        signal.set(xlabel="$t$", ylabel=r"$\int\nu\rho_h\,/\,(\nu_0 M_0)$")
        mass.set(xlabel="$t$", ylabel=r"$M(t)/M_0$")
        breaking.set(xlabel="$t$", ylabel=r"modes $k\notin9\mathbb{N}$ / $k\in9\mathbb{N}$")
        for ax in (signal, mass, breaking):
            ax.grid(True, which="major", alpha=.5)
        mass.legend(frameon=False, fontsize=6.5)
        breaking.legend(frameon=False, fontsize=6.5, loc="lower right")
        return save_publication_figure(figure, stem, dpi=300)


def _movie_label(movie) -> str:
    """'nu0=0.05, t>70' from a campaign movie name (..._ionize_nu0p05_w0p1_x0p475_y0_t70_...)."""
    parts = Path(movie).stem.split("_")
    if "inject" in parts:  # an explicit injector (S independent of rho), for comparison
        charge = next(q[1:].replace("p", ".") for q in parts if q.startswith("q") and q[1:2].isdigit())
        return f"injector $Q={charge}$"
    rate = next((q[2:].replace("p", ".") for q in parts if q.startswith("nu") and q[2:3].isdigit()), "?")
    start = next((q[1:].replace("p", ".") for q in parts if q.startswith("t") and q[1:2].isdigit()), None)
    label = f"$\\nu_0={rate}$" + (f", $t>{start}$" if start else "")
    return label + (", $\\epsilon=10^{-12}$" if Path(movie).stem.startswith("ion_seed") else "")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="diocotron_ionization")
    parser.add_argument("--column", type=Path, nargs="*", default=(), help="gas-column runs (one per time step)")
    parser.add_argument("--map", type=Path, help="uniform-gas run")
    parser.add_argument("--reference", type=Path, help="unforced run with the same mesh and seed")
    parser.add_argument("--map-snapshots", type=Path)
    parser.add_argument("--reference-snapshots", type=Path)
    parser.add_argument("--puff", type=Path, nargs="*", default=(), help="gas-puff runs (one row each)")
    parser.add_argument("--puff-labels", type=lambda text: text.split(";"), default=None)
    parser.add_argument("--puff-times", type=lambda text: tuple(float(v) for v in text.split(",")),
                        default=(70., 100., 150., 200., 300., 400.))
    parser.add_argument("--puff-stem", default="puff")
    parser.add_argument("--movies", type=Path, nargs="*", default=(), help="campaign movies for a density frame grid")
    parser.add_argument("--movie-times", type=lambda text: tuple(float(v) for v in text.split(",")),
                        default=(70., 100., 150., 200., 250., 300., 400.))
    parser.add_argument("--movie-stem", default="puff_frames")
    args = parser.parse_args()
    if args.column or args.map:
        print(checks_figure(args.column, args.map, args.reference, args.output_dir / f"{args.stem}_checks"))
    if args.map_snapshots and args.reference_snapshots:
        nu = float(_ionization(_configuration(_records(args.map)))["rate"])
        print(map_frames_figure(args.map_snapshots, args.reference_snapshots,
                                args.output_dir / f"{args.stem}_map_frames", nu=nu))
    if args.movies:
        from scripts.guiding_center.studies.diocotron.plot_diocotron_injection import movie_density_grid

        print(movie_density_grid(args.movies, args.output_dir / f"{args.stem}_{args.movie_stem}",
                                 times=args.movie_times, label=_movie_label))
    if args.puff:
        print(puff_figure(args.puff, args.output_dir / f"{args.stem}_{args.puff_stem}", times=args.puff_times,
                          labels=args.puff_labels))


if __name__ == "__main__":
    main()
