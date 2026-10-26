"""Magnetron (crossed-field diode) campaigns in the guiding-center limit.

SIM (time integration, operated by the user); ``--dry-run`` prints the run
configurations, building only the meshes (for their triangle counts).

Case :func:`scripts.guiding_center.cases.guiding_center_cases.magnetron`:
electrons between a cathode (phi = 0) and an anode carrying a DC voltage and a
rotating wave of N periods, phi = -V + V1 s(t) cos(N(theta - W t)); space-
charge-limited emission S + alpha rho = kappa chi(r)(rho_c - rho) keeps a hub on
the cathode. The anode wave makes the wall non-equipotential, so the drift
crosses it: electrons are collected where u.n > 0 (the anode current) and the
inflow elsewhere carries no charge. One run per (N, V1, W) of the CLI lists;
``--wave-amplitudes 0`` is the unforced control (the hub is linearly stable).

Every run uses the fastest recorded device configuration
(:mod:`scripts.guiding_center.runtime.fast_profiles`). The Poisson solves keep
zero Dirichlet data: the wall potential enters as discrete harmonic fields
(:mod:`scripts.guiding_center.runtime.electrodes`). Defaults: the compatible
drift (its wall normal is the tangential derivative of the wall data, so no
charge crosses the equipotential cathode), SI-BDF2, a fixed Poisson tau. Each
run records the wall outflux split between cathode and anode, the source and
reaction terms and the defect of the discrete charge balance
(``charge_balance_defect``); a summary JSON/CSV is written after every run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from scripts.guiding_center.studies.diocotron.run_diocotron_benchmark import (
    BASE_PRESET,
    VELOCITIES,
    _num_steps,
    _write_summary,
)
from scripts.guiding_center.runtime.device import release_device_memory
from scripts.guiding_center.runtime.observers import Observers, SnapshotSaver


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value)


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(",") if value)


def _case_params(args, mode: int, amplitude: float, speed: float) -> dict:
    if args.anode_geometry == "vanes" and not mode:  # the vane count is 2N even without a drive
        mode = args.wave_modes[0]
    return dict(cathode=args.cathode, anode=1.0, voltage=args.voltage, hub_top=args.hub_top,
                hub_density=args.hub_density, hub_edge=args.hub_edge, emission_rate=args.emission_rate,
                emission_top=args.emission_top, initial_hub=not args.empty_start, wave_mode=mode, wave_amplitude=amplitude, wave_speed=speed,
                wave_start=args.wave_start, wave_ramp=args.wave_ramp,
                band_mesh_size=args.band_mesh_size, refine_band=args.refine_band,
                anode_geometry=args.anode_geometry, back_radius=args.back_radius, vane_fraction=args.vane_fraction,
                corner_mesh_size=args.corner_mesh_size, corner_grading=args.corner_grading,
                vane_width=args.vane_width, tip_radius=args.tip_radius, emission_profile=args.emission_profile,
                cathode_mesh_size=args.cathode_mesh_size, cathode_grading=args.cathode_grading,
                curvature_points=args.curvature_points, anode_mesh_size=args.anode_mesh_size,
                anode_grading=args.anode_grading,
                hall_parameter=args.hall_parameter,
                electron_temperature=args.electron_temperature, emission_period=args.emission_period,
                emission_duty=args.emission_duty, emission_ramp=args.emission_ramp,
                emission_phase=args.emission_phase)


def _device_overrides(order: int, triangles: int, poisson_order: int | None = None) -> dict:
    """Fastest recorded device profile; outside p = 4..6 a fallback.

    ``poisson_order`` (default ``order``) is the Poisson degree q, raised above p with
    ``--poisson-order-offset``. fb-hp-mg has a recorded path for q = 4..6 only; any other q uses
    raw-CUDA CSR + AMGX with the nodal trace (the raised-degree study's profile). It is untuned for
    this case: time it before any cost claim. Transport keeps the fastest profile of the nearest
    recorded degree (face BSR; DILU has block size 8, so p <= 7); below p = 4 it is used by the
    collisional (ADR) runs only, whose density solver sets its own AMGX configuration.
    """
    from scripts.guiding_center.runtime.fast_profiles import fastest_device_overrides

    p = int(order)
    q = p if poisson_order is None else int(poisson_order)
    if p > 7:
        raise ValueError("the magnetron device path supports p <= 7 (AMGX DILU lacks transport block size 9)")
    if 4 <= p <= 6 and q == p:
        return fastest_device_overrides(p, zero_dirichlet=True, triangles=triangles)
    overrides = {key: value for key, value in
                 fastest_device_overrides(min(max(p, 4), 6), zero_dirichlet=True, triangles=triangles).items()
                 if not key.startswith("poisson_")}
    if 4 <= q <= 6:
        overrides.update({key: value for key, value in
                          fastest_device_overrides(q, zero_dirichlet=True, triangles=triangles).items()
                          if key.startswith("poisson_")})
    else:
        overrides.update(poisson_assembly_backend="raw-cuda", poisson_solver="amgx", poisson_raw_matrix_format="csr",
                         poisson_trace_basis="legacy-lagrange", poisson_cache_local_factors="schur-lu",
                         poisson_fb_hp_mg_preconditioner_policy="standard")
    return overrides


def _d7_tag(args) -> str:
    """Prefix tag of the D7 options (empty without them, so D6 names are unchanged)."""
    tag = "" if args.vane_width is None else f"_vw{args.vane_width:g}"
    tag += "" if args.tip_radius is None else f"_tr{args.tip_radius:g}"
    tag += "" if args.cathode_mesh_size is None else f"_cm{args.cathode_mesh_size:g}g{args.cathode_grading:g}"
    tag += "" if not args.curvature_points else f"_cv{args.curvature_points}"
    tag += "" if args.anode_mesh_size is None else f"_am{args.anode_mesh_size:g}g{args.anode_grading:g}"
    tag += "" if args.emission_profile == "cos2" else f"_em{args.emission_profile}"
    if args.hall_parameter is not None:
        tag += f"_hall{args.hall_parameter:g}_te{args.electron_temperature:g}"
        tag += "_pardiso" if args.density_backend == "host" else ""
    if args.emission_period is not None:
        tag += f"_pulse{args.emission_period:g}d{args.emission_duty:g}"
        tag += "" if args.emission_phase == 0.0 else f"ph{args.emission_phase:g}"
    return tag


def _mesh(case_params: dict, mesh_size: float):
    from hybridge.core.mesh import gmsh_annulus_mesh, gmsh_vane_anode_mesh
    from scripts.guiding_center.cases.guiding_center_cases import magnetron

    geometry = magnetron(**case_params).parameters["geometry"]
    if "vanes" in geometry:
        return gmsh_vane_anode_mesh(
            float(mesh_size), cathode_radius=geometry["inner_radius"], anode_radius=geometry["anode_radius"],
            back_radius=geometry["back_radius"], vanes=geometry["vanes"], vane_fraction=geometry["vane_fraction"],
            band=geometry["band"], band_mesh_size=geometry["band_mesh_size"], band_grading=geometry["band_grading"],
            corner_mesh_size=geometry["corner_mesh_size"], corner_grading=geometry["corner_grading"],
            vane_width=geometry.get("vane_width"), tip_radius=geometry.get("tip_radius"),
            cathode_mesh_size=geometry.get("cathode_mesh_size"),
            cathode_grading=float(geometry.get("cathode_grading", 0.3)),
            curvature_points=geometry.get("curvature_points"), anode_mesh_size=geometry.get("anode_mesh_size"),
            anode_grading=float(geometry.get("anode_grading", 0.3)), log_cache=False)
    return gmsh_annulus_mesh(
        float(mesh_size), inner_radius=geometry["inner_radius"], outer_radius=geometry["outer_radius"],
        band=geometry["band"], band_mesh_size=geometry["band_mesh_size"], band_grading=geometry["band_grading"],
        log_cache=False)


def build_configs(args) -> list[dict]:
    """One record (config + metadata) per run."""
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.fast_profiles import fastest_device_overrides

    base = preset_by_key(BASE_PRESET)
    records = []
    waves = [(0, 0.0, 0.0)] if args.controls else []
    waves += [(n, a, w) for n in args.wave_modes for a in args.wave_amplitudes for w in args.wave_speeds
              if a != 0.0]
    if 0.0 in args.wave_amplitudes and not args.controls:
        waves.insert(0, (0, 0.0, 0.0))
    for mode, amplitude, speed in waves:
        params = _case_params(args, mode, amplitude, speed)
        triangles = int(_mesh(params, args.mesh_size).num_tri)
        label = (f"N{mode}_a{amplitude:g}_w{speed:g}" if amplitude else "nowave")
        for order in args.orders:
            for scheme in args.schemes:
                for dt in args.dts:
                    for velocity in args.velocities:
                        field, potential = VELOCITIES[velocity]
                        kkt = ("" if args.density_positivity != "kkt" else
                               "_kkt" + ("dense" if args.density_positivity_points.endswith("dense") else ""))
                        layer = "" if args.emission_top is None else f"_e{args.emission_top:g}"
                        edge = "" if args.hub_edge == 0.01 else f"_d{args.hub_edge:g}"
                        band = "" if args.band_mesh_size is None else f"_hb{args.band_mesh_size:g}"
                        if args.anode_geometry == "vanes":
                            band += f"_vanes{2 * mode or 2 * args.wave_modes[0]}_rb{args.back_radius:g}"
                        band += _d7_tag(args)
                        offset = int(args.poisson_order_offset)
                        degree = "" if offset == 0 else f"q{order + offset}"
                        prefix = (f"{args.prefix}_{label}_rc{args.cathode:g}_rh{args.hub_top:g}{edge}_V{args.voltage:g}_"
                                  f"k{args.emission_rate:g}{layer}_p{order}{degree}_h{args.mesh_size:g}{band}_"
                                  f"{scheme.replace('-', '_')}_dt{dt:g}_{velocity}{kkt}").replace(".", "p")
                        config = replace(
                            base,
                            # Poisson data stay zero: the electrodes are superposed harmonic fields.
                            **_device_overrides(order, triangles, order + offset),
                            poisson_order_offset=offset,
                            description=f"article magnetron {label}",
                            case="magnetron", case_params=params,
                            domain="vane-anode" if args.anode_geometry == "vanes" else "annulus",
                            mesh_size=float(args.mesh_size), minimum_triangles=0, order=int(order),
                            dt=float(dt), num_steps=_num_steps(args.final_time, dt), time_scheme=scheme,
                            poisson_tau=float(args.poisson_tau), poisson_tau_max_retries=0,
                            transport_electric_field=field, compatible_potential=potential,
                            transport_advection_stabilization="conflict-averaged-upwind",
                            transport_boundary_mode="eliminate",  # open anode: zero inflow data
                            diocotron_diagnostics=False, positivity_diagnostics=True,
                            density_positivity=args.density_positivity,
                            density_positivity_points=args.density_positivity_points,
                            diagnostics_every=1, record_timings=True, save_diagnostics=True,
                            plot_every=0, plot_diagnostics=False, save_movie=False, screenshot_dir=None,
                            verbosity=int(args.verbosity), diagnostics_dir=str(args.output_dir),
                            diagnostics_prefix=prefix,
                        )
                        if args.poisson_atol is not None:
                            config = replace(config, poisson_solver_atol=float(args.poisson_atol))
                        if args.density_rtol is not None:
                            config = replace(config, transport_solver_rtol=float(args.density_rtol))
                        if args.density_amgx_tolerance is not None:
                            config = replace(config, transport_amgx_tolerance=float(args.density_amgx_tolerance))
                        if args.hall_parameter is not None and args.density_backend == "host":
                            # Collisional density by Numba ADR assembly and PyPardiso on all threads.
                            config = replace(config, transport_assembly_backend="numba",
                                             transport_solver="pypardiso")
                        records.append(dict(config=config, prefix=prefix, label=label, order=order, scheme=scheme,
                                            dt=dt, velocity=velocity, triangles=triangles, wave_mode=mode,
                                            wave_amplitude=amplitude, wave_speed=speed))
    return records


class _MagnetronMovie:
    """Density and the non-axisymmetric potential (phi - <phi>_theta)/max|.|, streamed to an MP4."""

    def __init__(self, path: Path, *, interval: float, dt: float, density_scale: float, label: str,
                 raster: int, fps: float):
        self.path, self.interval, self.half_step = path, float(interval), .5 * float(dt)
        self.density_scale, self.label, self.raster, self.fps = float(density_scale), label, int(raster), float(fps)
        self.next_time, self.recorder, self.bins = 0.0, None, None

    def _build(self, snapshot):
        import matplotlib as mpl

        from hybridge.io.recording import RasterMovieRecorder

        density_map = mpl.colormaps["viridis"].with_extremes(under="magenta")
        density_map.colorbar_extend = "min"
        self.recorder = RasterMovieRecorder(
            snapshot.space, self.path,
            panels=[(r"Electron density $\rho_h$", dict(cmap=density_map, clim=(-1.0e-3 * self.density_scale,
                                                                                   1.02 * self.density_scale))),
                    (r"$(\phi_h-\langle\phi_h\rangle_\theta)/\max|\cdot|$", dict(cmap="RdBu_r", clim=(-1.0, 1.0)))],
            width=self.raster, height=self.raster, size=(1700, 820), fps=self.fps, title=self.label,
            boundary_mesh=snapshot.space.mesh)
        self.x, self.y = self.recorder.geometry.pixel_centres()
        self.bins = self.recorder.shape[1] // 2

    def _non_axisymmetric(self, raster):
        from hybridge.io.raster import remove_azimuthal_mean

        mask = np.ma.getmaskarray(raster)
        return np.ma.array(remove_azimuthal_mean(np.ma.getdata(raster), self.x, self.y, bins=self.bins, mask=mask),
                           mask=mask)

    def __call__(self, snapshot) -> None:
        if snapshot.time + self.half_step < self.next_time:
            return
        while self.next_time <= snapshot.time + self.half_step:
            self.next_time += self.interval
        if self.recorder is None:
            self._build(snapshot)
        density = self.recorder.sample(snapshot.accepted_density)
        perturbation = self._non_axisymmetric(self.recorder.sample(snapshot.poisson_result.field))
        amplitude = float(np.ma.max(np.ma.abs(perturbation))) if perturbation.count() else 0.0
        scaled = perturbation / amplitude if amplitude > 0 else perturbation
        self.recorder.frame((density, scaled), caption=(
            f"{self.label}   t = {snapshot.time:7.2f}   max|phi - <phi>| = {amplitude:.2e}"))

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()


def _summary_row(record, jsonl: Path, wall_time) -> dict:
    """Anode current, charge, balance and solver statistics of one run."""
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("step", 0) > 0 and "mass" in row]
    t = np.array([row["time"] for row in rows])
    late = t >= t[-1] * 2.0 / 3.0

    def series(key):
        return np.array([np.nan if row.get(key) is None else float(row[key]) for row in rows])

    anode, cathode, mass = series("wall_outflux_outer"), series("wall_outflux_inner"), series("mass")
    defect = series("charge_balance_relative_defect")
    transport, poisson = series("transport_step_iterations"), series("poisson_step_iterations")
    step_times = [row["linear_step_wall_time"] for row in rows[2:] if row.get("linear_step_wall_time") is not None]
    return dict(
        prefix=record["prefix"], label=record["label"], wave_mode=record["wave_mode"],
        wave_amplitude=record["wave_amplitude"], wave_speed=record["wave_speed"], order=record["order"],
        scheme=record["scheme"], dt=record["dt"], velocity=record["velocity"], triangles=record["triangles"],
        final_time=float(t[-1]), anode_current_late_mean=float(np.nanmean(anode[late])),
        anode_current_late_std=float(np.nanstd(anode[late])), anode_current_max=float(np.nanmax(anode)),
        cathode_flux_max_abs=float(np.nanmax(np.abs(cathode))), charge_final=float(mass[-1]),
        charge_late_mean=float(np.nanmean(mass[late])),
        charge_balance_relative_defect_max=float(np.nanmax(np.abs(defect))),
        rho_min=float(np.nanmin(series("rho_min"))),
        transport_iterations_mean=float(np.nanmean(transport)), transport_iterations_max=float(np.nanmax(transport)),
        poisson_iterations_mean=float(np.nanmean(poisson)), poisson_iterations_max=float(np.nanmax(poisson)),
        median_linear_step_wall_time=statistics.median(step_times) if step_times else None,
        wall_time_s=wall_time, jsonl=str(jsonl),
    )


def run_campaign(args) -> None:
    from scripts.guiding_center.runtime.runner import run_guiding_center_case

    records = build_configs(args)
    if args.dry_run:
        for record in records:
            config = asdict(record["config"])
            print(json.dumps(dict(prefix=record["prefix"], triangles=record["triangles"], **{key: config[key] for key in (
                "case_params", "mesh_size", "order", "dt", "num_steps", "time_scheme", "poisson_tau",
                "poisson_solver", "poisson_fb_hp_mg_preconditioner_policy", "transport_raw_matrix_format",
                "transport_electric_field")}), default=str))
        print(f"{len(records)} runs")
        return
    output_dir = Path(args.output_dir)
    rows, failures = [], []
    for record in records:
        jsonl = output_dir / f"{record['prefix']}.jsonl"
        started = time.perf_counter()
        saver = SnapshotSaver(args.snapshot_times, record["dt"]) if args.snapshot_times else None
        movie = None
        if args.movie_interval > 0:
            positivity = "KKT positivity" if args.density_positivity == "kkt" else "no positivity limiter"
            label = (f"{record['label']}  p={record['order']}  {record['triangles']} triangles  {record['scheme']}  "
                     f"dt={record['dt']:g}  {positivity}")
            movie = _MagnetronMovie(output_dir / "movies" / f"{record['prefix']}.mp4", interval=args.movie_interval,
                                    dt=record["dt"], density_scale=args.hub_density, label=label,
                                    raster=args.movie_raster, fps=args.movie_fps)
        try:
            try:
                run_guiding_center_case(record["config"], preset_key=record["prefix"],
                                        step_observer=Observers(saver, movie))
            finally:
                if movie is not None:
                    movie.close()
                release_device_memory()
            if saver is not None:
                saver.save(output_dir / f"{record['prefix']}_snapshots.npz")
        except Exception as error:  # a failed run is a result
            failures.append(dict(prefix=record["prefix"], error=f"{type(error).__name__}: {error}"))
            print(f"FAILED {record['prefix']}: {type(error).__name__}: {error}", flush=True)
            _write_summary(rows, failures, output_dir, args.prefix)
            continue
        row = _summary_row(record, jsonl, time.perf_counter() - started)
        rows.append(row)
        print(f"{record['prefix']}: anode current {row['anode_current_late_mean']:.4e} "
              f"charge {row['charge_final']:.4f} balance {row['charge_balance_relative_defect_max']:.1e}", flush=True)
        _write_summary(rows, failures, output_dir, args.prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cathode", type=float, default=0.5)
    parser.add_argument("--hub-top", type=float, default=0.65)
    parser.add_argument("--hub-density", type=float, default=1.0)
    parser.add_argument("--hub-edge", type=float, default=0.01)
    parser.add_argument("--voltage", type=float, default=0.1)
    parser.add_argument("--emission-rate", type=float, default=1.0)
    parser.add_argument("--emission-top", type=float, default=None,
                        help="confine the emission to a cathode layer below this radius (default: the hub top)")
    parser.add_argument("--empty-start", action="store_true", help="start without the hub (emission fills it)")
    parser.add_argument("--density-positivity", choices=("none", "kkt"), default="none",
                        help="KKT positivity projection of every transported density (mass conserving)")
    parser.add_argument("--density-positivity-points", choices=("quadrature+lattice", "quadrature+lattice+dense"),
                        default="quadrature+lattice+dense")
    parser.add_argument("--anode-geometry", choices=("circle", "vanes"), default="circle",
                        help="circular anode with a rotating wave, or a vane block driven in the pi mode")
    parser.add_argument("--back-radius", type=float, default=1.35, help="vanes: radius of the slot back walls")
    parser.add_argument("--vane-fraction", type=float, default=0.5, help="vanes: fraction of the pitch per vane")
    parser.add_argument("--corner-mesh-size", type=float, default=None, help="vanes: mesh size at the vane tips")
    parser.add_argument("--corner-grading", type=float, default=0.3)
    parser.add_argument("--vane-width", type=float, default=None,
                        help="vanes: parallel-sided vanes of this width (D7) instead of radial ones")
    parser.add_argument("--tip-radius", type=float, default=None,
                        help="vanes with --vane-width: round the vane-tip corners by fillets of this radius")
    parser.add_argument("--poisson-atol", type=float, default=None,
                        help="absolute residual target of the Poisson solve (default: the preset's, 1e-10)")
    parser.add_argument("--density-rtol", type=float, default=None,
                        help="relative tolerance of the AMGX density solve (default: the preset's, 1e-11)")
    parser.add_argument("--density-amgx-tolerance", type=float, default=None,
                        help="AMGX stopping tolerance of the density solve, relative to the initial residual; it also "
                             "sets the solver-system residual check (default: the config file's 1e-11)")
    parser.add_argument("--cathode-mesh-size", type=float, default=None,
                        help="vanes: mesh size at the cathode (grades towards the cathode layer)")
    parser.add_argument("--cathode-grading", type=float, default=0.3)
    parser.add_argument("--anode-mesh-size", type=float, default=None,
                        help="vanes: mesh size on the anode walls (outflow layers where the charge is absorbed)")
    parser.add_argument("--anode-grading", type=float, default=0.3)
    parser.add_argument("--curvature-points", type=int, default=None,
                        help="vanes: bound the size on the curved walls by 2 pi R / points (affine chord error)")
    parser.add_argument("--emission-profile", choices=("cos2", "smooth"), default="cos2",
                        help="taper of the emission layer (with --emission-top): C^1 cos^2 or C^infinity")
    parser.add_argument("--hall-parameter", type=float, default=None,
                        help="collisional electrons (D7): Hall parameter h; drift u_E + q/h, diffusion theta/h")
    parser.add_argument("--electron-temperature", type=float, default=0.0,
                        help="collisional electrons: theta (electron temperature over the potential unit)")
    parser.add_argument("--density-backend", choices=("device", "host"), default="device",
                        help="collisional ADR density solve: raw-CUDA + AMGX (device) or Numba + PyPardiso (host)")
    parser.add_argument("--emission-period", type=float, default=None, help="pulsed emission: pulse period")
    parser.add_argument("--emission-duty", type=float, default=0.5, help="pulsed emission: on fraction")
    parser.add_argument("--emission-ramp", type=float, default=None, help="pulsed emission: sin^2 ramp length")
    parser.add_argument("--emission-phase", type=float, default=0.0,
                        help="pulsed emission: evaluate the train at t + phase (half a pulse: start with emission on)")
    parser.add_argument("--wave-modes", type=_ints, default=(6,))
    parser.add_argument("--wave-amplitudes", type=_floats, default=(0.025,), help="V1 per run (0 = no wave)")
    parser.add_argument("--wave-speeds", type=_floats, default=(0.29,), help="wave angular velocity W per run")
    parser.add_argument("--wave-start", type=float, default=0.0)
    parser.add_argument("--wave-ramp", type=float, default=20.0)
    parser.add_argument("--controls", action="store_true", help="also run without the anode wave")
    parser.add_argument("--orders", type=_ints, default=(6,))
    parser.add_argument("--mesh-size", type=float, default=0.025)
    parser.add_argument("--band-mesh-size", type=float, default=None)
    parser.add_argument("--refine-band", type=_floats, default=None)
    parser.add_argument("--dts", type=_floats, default=(0.1,))
    parser.add_argument("--schemes", type=lambda text: tuple(text.split(",")), default=("si-bdf2",))
    parser.add_argument("--velocities", type=lambda text: tuple(text.split(",")), default=("compatible",))
    parser.add_argument("--final-time", type=float, default=300.0)
    parser.add_argument("--poisson-tau", type=float, default=32000.0)
    parser.add_argument("--poisson-order-offset", type=int, default=0,
                        help="Poisson degree q = p + offset; offset > 0 needs --velocities raw (tag _p<p>q<q>)")
    parser.add_argument("--movie-interval", type=float, default=1.0, help="MP4 frame interval (0 disables)")
    parser.add_argument("--movie-raster", type=int, default=1024)
    parser.add_argument("--movie-fps", type=float, default=12.0)
    parser.add_argument("--snapshot-times", type=_floats, default=())
    parser.add_argument("--output-dir", default="run_outputs/article/magnetron")
    parser.add_argument("--prefix", default="mag")
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    unknown = set(args.velocities) - set(VELOCITIES)
    if unknown:
        parser.error(f"unknown velocities {sorted(unknown)}; choose from {sorted(VELOCITIES)}")
    run_campaign(args)


if __name__ == "__main__":
    main()
