"""E7 driver: unperturbed control and band-mode seeded evolution of imported equilibrium bands.

For each geometry the equilibrium (CG potential checkpoint and its ``contours.npz``) comes from a stage
directory ``<stage-root>/<geometry>/<stage-name>`` and the authenticated archive entry named in its
``completed.json``. Two runs share every numerical choice: ``control`` starts from the projected
equilibrium W(phi*), ``seed`` from the band-mode displacement of ``scripts/guiding_center/cases/
band_seed.py``. Both record the band-mode spectrum of rho_h and phi_h every step (JSONL keys
``band_*``), the conservation and enstrophy-budget diagnostics, and rho_h, phi_h at the snapshot
times (``<prefix>_snapshots.npz``). The control also records the transfer diagnostics
(``equilibrium_*``, including the quadrature-enrichment check); the seeded run skips them because its
mesh and equilibrium are the control's.

``--equilibrium-transfer consistent`` starts both runs from the root of rho = Pi_h W(phi_c(rho)) instead of the
projection (Newton-GMRES, ``runtime/equilibrium_transfer.py``); the seeded run keeps its displacement on top.

Defaults (PLAN.md E7): p = 6, compatible drift (Oswald average of the p+1 postprocessed potential),
tau = 1 without retries, SI-BDF2, dt = 0.05, T = 200, fastest recorded device configuration,
the embedded equilibrium mesh. ``--seed-check`` only builds and checks the seeds (host, no solve).

Example (20-step smoke of the star pair):
  .venv/bin/python -m scripts.guiding_center.benchmarks.run_band_evolution --geometries star \
      --kinds control,seed --steps 20 --output-dir run_outputs/e7_smoke --prefix e7smoke
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HYBRIDGE = Path(__file__).resolve().parents[3]
ARTICLE = Path("/home/adelsaleh/src/gc-hdg-article")
STAGE_ROOT = ARTICLE / "run_outputs" / "equilibrium_dt080_eps0005_20261016" / "stages"
STAGE_NAME = "eps0p005_q64"
ARCHIVE = ARTICLE / "equilibria"
GEOMETRIES = ("star", "uglypacman", "iter", "horseshoe")
# Most unstable vortex-sheet mode k d = 0.8 for the band length L and thickness d: l = 0.8 L / (2 pi d).
MODES = {"star": 16, "uglypacman": 16, "iter": 18, "horseshoe": 20}


def parse_mapping(text, cast):
    out = {}
    for item in filter(None, (part.strip() for part in (text or "").split(";"))):
        key, value = item.split("=", 1)
        out[key.strip()] = cast(value)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--stage-name", default=STAGE_NAME)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--geometries", default=",".join(GEOMETRIES))
    parser.add_argument("--kinds", default="control,seed")
    parser.add_argument("--modes", default=";".join(f"{k}={v}" for k, v in MODES.items()),
                        help="seed mode per geometry, 'star=16;iter=18'")
    parser.add_argument("--wall-features", default="",
                        help="crest point per geometry 'star=x,y;...' (default: minimum wall clearance)")
    parser.add_argument("--sites", default="",
                        help="seed by outward bulges at the band points nearest these points instead of a mode, "
                             "'star=x1,y1,x2,y2,...;...'")
    parser.add_argument("--site-width", type=float, default=0.35,
                        help="angular width (C^2 bump support, radians of the flow angle) of each bulge")
    parser.add_argument("--varicose", action="store_true",
                        help="edges move in opposite normal directions (band thickens at the crests) instead of together")
    parser.add_argument("--edge-fraction", type=float, default=0.05,
                        help="largest normal displacement of the band edges / band thickness")
    parser.add_argument("--packet-kd", type=float, default=None,
                        help="localized packet wavelength 2*pi*d/kd instead of a global mode")
    parser.add_argument("--packet-wavelengths", type=float, default=3.,
                        help="full packet support in carrier wavelengths")
    parser.add_argument("--band-half-width-factor", type=float, default=2.)
    parser.add_argument("--wall-margin-factor", type=float, default=0.5,
                        help="seed generator: zero within this distance of the wall (units of d), one beyond twice it")
    parser.add_argument("--grid-fraction", type=float, default=0.05,
                        help="seed generator spline-grid spacing in units of d (finer than the wall gap)")
    parser.add_argument("--linear-startup", choices=("euler", "richardson"), default="euler",
                        help="kind 'linear': first step SI-Euler (as the nonlinear runs) or its Richardson extrapolation")
    parser.add_argument("--linear-scale", type=float, default=None,
                        help="kind 'linear': q_0 = -scale u_seed . grad rho* (default 1/edge-fraction: per unit "
                             "edge-displacement fraction)")
    parser.add_argument("--profile-interval", type=float, default=0.,
                        help="transverse density/potential probe interval; 0 disables")
    parser.add_argument("--profile-angles", type=int, default=1024)
    parser.add_argument("--profile-transverse", type=int, default=32)
    parser.add_argument("--profile-half-width-factor", type=float, default=None,
                        help="normal half-width of the probe strip in units of d (default min(1, band half-width "
                             "factor)); a near-wall band needs it below its smallest outer-wall distance")
    parser.add_argument("--max-mode", type=int, default=80)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--final-time", type=float, default=200.0)
    parser.add_argument("--steps", type=int, default=None, help="override the step count (smoke runs)")
    parser.add_argument("--time-scheme", default="si-bdf2")
    parser.add_argument("--volume-quad-1d", type=int, default=None, help="default (3p+2)//2")
    parser.add_argument("--initial-projection-quad-1d", type=int, default=24)
    parser.add_argument("--poisson-solver-atol", type=float, default=5e-14)
    parser.add_argument("--equilibrium-transfer", choices=("projection", "consistent"), default="projection",
                        help="initial state: rho_h = Pi_h W(phi_CG) (projection) or the root of rho = Pi_h W(phi_c(rho)) "
                             "with phi_c the compatible potential of rho (consistent, runtime/equilibrium_transfer.py)")
    # the pMG-AMG Poisson solve stalls near 1.4e-11 of |b| on these CG6 meshes (star, ITER): 5e-11 leaves a margin
    parser.add_argument("--poisson-solver-rtol", type=float, default=5e-11)
    parser.add_argument("--snapshot-times", default=None,
                        help="comma-separated times (default: every 2.5 to t = 60, then every 5)")
    parser.add_argument("--snapshot-fields", default="density,potential")
    parser.add_argument("--diagnostics-every", type=int, default=1)
    parser.add_argument("--movie-interval", type=float, default=0.5,
                        help="time between MP4 frames of rho_h and phi_h - phi_eq (0: no movie)")
    parser.add_argument("--movie-raster", type=int, default=1024)
    parser.add_argument("--backend", choices=("device", "host"), default="device",
                        help="device: fastest recorded GPU configuration; host: Numba + PyPardiso (pipeline checks)")
    parser.add_argument("--remesh-size", type=float, default=None,
                        help="uniform Gmsh remesh of the wall polygon instead of the embedded mesh (pipeline checks)")
    parser.add_argument("--mesh-npz", type=Path, default=None,
                        help="HDG mesh (node_coords, triangles) of the same polygon instead of the embedded source mesh")
    parser.add_argument("--host-threads", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="e7")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the run configurations, run nothing")
    parser.add_argument("--seed-check", action="store_true", help="build and check the seeds only (host)")
    parser.add_argument("--verbosity", type=int, default=1)
    args = parser.parse_args(argv)
    args.geometries = [g.strip() for g in args.geometries.split(",") if g.strip()]
    args.kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    if set(args.kinds) - {"control", "seed", "linear"}:
        parser.error("kinds are control, seed and linear (the seed's linearized response)")
    args.modes = dict(MODES, **parse_mapping(args.modes, int))
    args.wall_features = parse_mapping(args.wall_features, lambda v: tuple(float(c) for c in v.split(",")))
    args.sites = parse_mapping(args.sites, lambda v: [float(c) for c in v.split(",")])
    if any(len(v) % 2 or not v for v in args.sites.values()):
        parser.error("--sites needs x,y pairs")
    if args.packet_kd is not None and (args.varicose or args.sites):
        parser.error("--packet-kd cannot be combined with --varicose or --sites")
    if args.packet_kd is not None and args.packet_kd <= 0:
        parser.error("--packet-kd must be positive")
    if args.profile_interval < 0 or args.band_half_width_factor <= 0:
        parser.error("profile interval must be nonnegative and band half-width positive")
    args.num_steps = args.steps if args.steps is not None else int(round(args.final_time / args.dt))
    if args.steps is None and abs(args.num_steps * args.dt - args.final_time) > 1e-9 * args.final_time:
        parser.error("final time must be a multiple of dt")
    args.volume_quad_1d = args.volume_quad_1d or (3 * args.order + 2) // 2
    return args


def stage_dir(args, geometry) -> Path:
    return args.stage_root / geometry / args.stage_name


def archive_id(args, geometry) -> str:
    return json.loads((stage_dir(args, geometry) / "completed.json").read_text())["archive_id"]


def prefix_of(args, geometry, kind) -> str:
    return (f"{args.prefix}_{geometry}_p{args.order}_{args.time_scheme.replace('-', '_')}_dt{args.dt:g}_"
            f"{kind}").replace(".", "p")


def case_params(args, geometry, kind) -> dict:
    seeded = kind in ("seed", "linear")
    feature = args.wall_features.get(geometry, "min-clearance")
    return dict(archive=str(args.archive.resolve()), equilibrium=archive_id(args, geometry),
                band_stage=str(stage_dir(args, geometry).resolve()),
                seed_mode=args.modes[geometry] if seeded and geometry not in args.sites
                and args.packet_kd is None else None,
                seed_sites=args.sites[geometry] if seeded and geometry in args.sites else None,
                seed_site_width=args.site_width, seed_varicose=args.varicose, seed_edge_fraction=args.edge_fraction,
                seed_packet_kd=args.packet_kd if seeded else None,
                seed_packet_wavelengths=args.packet_wavelengths, band_half_width_factor=args.band_half_width_factor,
                band_coordinate_spacing_factor=.025 if args.packet_kd is not None else None,
                seed_wall_feature=feature if isinstance(feature, str) else list(feature),
                band_max_mode=args.max_mode, import_diagnostics="none" if seeded else "full",
                remesh_size=args.remesh_size, mesh_npz=None if args.mesh_npz is None else str(args.mesh_npz.resolve()),
                seed_wall_margin_factor=args.wall_margin_factor,
                seed_grid_fraction=args.grid_fraction, linear_scale=args.linear_scale)


def build_case(args, geometry, kind):
    from scripts.guiding_center.cases.imported_equilibrium import imported_equilibrium
    params = case_params(args, geometry, kind)
    if not isinstance(params["seed_wall_feature"], str):
        params["seed_wall_feature"] = tuple(params["seed_wall_feature"])
    return imported_equilibrium(**params)


def build_config(args, geometry, kind, triangles):
    from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset
    from scripts.guiding_center.runtime.fast_profiles import fastest_device_overrides
    prefix = prefix_of(args, geometry, kind)
    if args.backend == "device":
        backend = fastest_device_overrides(args.order, zero_dirichlet=True, triangles=triangles)
    else:
        backend = dict(trace_basis="legendre-modal", poisson_assembly_backend="numba", poisson_local_backend="numba",
                       poisson_solver="pypardiso", transport_assembly_backend="numba", transport_solver="pypardiso",
                       poisson_scale_system=False, transport_scale_system=False)
    return GuidingCenterRunPreset(
        case="imported_equilibrium", description=f"E7 {geometry} {kind}",
        case_params=case_params(args, geometry, kind), domain="auto", mesh_size=args.remesh_size or 0.0,
        **backend,
        order=args.order, poisson_order_offset=0, transport_electric_field="compatible",
        compatible_potential="postprocessed", equilibrium_transfer=args.equilibrium_transfer,
        dt=args.dt, num_steps=args.num_steps, time_scheme=args.time_scheme,
        poisson_tau=1.0, poisson_tau_max_retries=0, poisson_solver_rtol=args.poisson_solver_rtol,
        poisson_solver_atol=args.poisson_solver_atol, transport_solver_rtol=1e-11, transport_solver_atol=1e-12,
        volume_quad_1d=args.volume_quad_1d, initial_projection_quad_1d=args.initial_projection_quad_1d,
        transport_advection_stabilization="conflict-averaged-upwind", positivity_diagnostics=kind != "linear",
        linearized=kind == "linear", linearized_startup=args.linear_startup,
        enstrophy_budget=args.time_scheme in ("si-euler", "si-bdf2") and kind != "linear",
        diagnostics_every=args.diagnostics_every,
        record_timings=True, save_diagnostics=False, plot_every=0, plot_diagnostics=False, save_movie=False,
        verbosity=args.verbosity, diagnostics_dir=str(args.output_dir), diagnostics_prefix=prefix)


def completed(args, geometry, kind) -> bool:
    prefix = prefix_of(args, geometry, kind)
    path = args.output_dir / f"{prefix}.jsonl"
    if not path.exists() or not (args.output_dir / f"{prefix}_summary.json").exists():
        return False
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return bool(lines) and json.loads(lines[-1]).get("step") == args.num_steps


def snapshot_times(args):
    from scripts.guiding_center.benchmarks.run_observers import snapshot_schedule
    if args.snapshot_times:
        return tuple(float(t) for t in args.snapshot_times.split(","))
    return snapshot_schedule(args.num_steps * args.dt)


def seed_figure(case, path, check):
    """Stream function, original and displaced band edges, and a zoom at the crest."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.guiding_center.cases.band_seed import load_stage
    from scripts.guiding_center.cases.imported_equilibrium import boundary_polygon
    band = case.imported_equilibrium.band
    stage = load_stage(band.parameters["stage"])
    wall = boundary_polygon(case.imported_equilibrium.checkpoint)
    disp = band.displacement
    crest = np.asarray(band.parameters["crest_point"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    for ax, zoom in zip(axes, (False, True)):
        ax.fill(*np.vstack([wall, wall[:1]]).T, color="#f2f2f2", lw=0)
        ax.plot(*np.vstack([wall, wall[:1]]).T, color="#222222", lw=0.6)
        X, Y = np.meshgrid(disp.x, disp.y, indexing="ij")
        limit = float(np.abs(disp.chi).max())
        ax.pcolormesh(X, Y, disp.chi, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto", rasterized=True)
        for name in ("inner", "outer"):
            line = stage[name]
            moved = disp.flow(line, time=1.0, steps=band.steps)
            ax.plot(*line.T, color="#555555", lw=0.6)
            ax.plot(*moved.T, color="#111111", lw=1.0)
        ax.plot(*crest, "o", ms=5, mfc="none", mec="#d62728")
        ax.set_aspect("equal")
        if zoom:
            half = 6 * band.half_width
            ax.set_xlim(crest[0] - half, crest[0] + half)
            ax.set_ylim(crest[1] - half, crest[1] + half)
    p = band.parameters
    fig.suptitle(f"{p['archive_id'].split('_')[0]}: mode {p['mode']}, edge displacement {p['edge_displacement']:.2e} "
                 f"({p['edge_fraction']:g} d), measured/target {check['edge_displacement_ratio']:.3f}; "
                 "stream function, band edges before (grey) and after (black)", fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_one(args, geometry, kind) -> dict:
    from scripts.guiding_center.benchmarks.run_observers import (
        BandMovieObserver, BandProfileObserver, Observers, SpillingSnapshotSaver, release_device_memory)
    from scripts.guiding_center.runtime.runner import run_guiding_center_case
    from scripts.reports.run_provenance import run_provenance
    prefix = prefix_of(args, geometry, kind)
    started = time.perf_counter()
    case = build_case(args, geometry, kind)
    mesh = case.imported_equilibrium.to_mesh()
    config = build_config(args, geometry, kind, mesh.num_tri)
    out = args.output_dir
    (out / f"{prefix}_config.json").write_text(json.dumps(asdict(config), indent=2, default=str) + "\n")
    band = case.imported_equilibrium.band
    seed_info = dict(case_parameters=case.parameters)
    if kind in ("seed", "linear"):
        seed_info["check"] = band.check(band.parameters["stage"])
    (out / f"{prefix}_seed.json").write_text(json.dumps(seed_info, indent=2, default=str) + "\n")
    spill = out / f"{prefix}_states"
    shutil.rmtree(spill, ignore_errors=True)
    fields = tuple(f.strip() for f in args.snapshot_fields.split(",") if f.strip())
    saver = SpillingSnapshotSaver(snapshot_times(args), args.dt, spill, fields=fields)
    profiles = None
    if args.profile_interval > 0:
        from hybridge.diagnostics.band_profiles import BandProbeGrid
        grid = BandProbeGrid.from_coordinates(
            band.coordinates, band.thickness, angles=args.profile_angles,
            transverse=args.profile_transverse,
            half_width_factor=(min(1., args.band_half_width_factor) if args.profile_half_width_factor is None
                               else args.profile_half_width_factor))
        profiles = BandProfileObserver(
            out / f"{prefix}_profiles", grid, interval=args.profile_interval, dt=args.dt)
    provenance = run_provenance(HYBRIDGE)
    movie = None
    if args.movie_interval > 0:
        mode = (", unperturbed control" if kind == "control" else ", linearized response" if kind == "linear" else
                ", localized packet" if args.packet_kd is not None else
                f", {len(args.sites[geometry]) // 2} site bulges" if geometry in args.sites else
                f", mode {args.modes[geometry]} seed")
        movie = BandMovieObserver(out / "movies" / f"{prefix}.mp4", interval=args.movie_interval, dt=args.dt,
                                  label=f"{geometry}{mode}  p={args.order}  {mesh.num_tri} triangles  "
                                        f"{args.time_scheme}  dt={args.dt:g}  compatible drift",
                                  raster=args.movie_raster)
    try:
        from contextlib import nullcontext
        threads = nullcontext()
        if args.backend == "host":
            import numba
            from hybridge.linalg.pardiso_runtime import pardiso_thread_limit
            from hybridge.runtime.threads import set_host_threads
            numba.set_num_threads(min(args.host_threads, numba.config.NUMBA_NUM_THREADS))
            set_host_threads(args.host_threads)
            threads = pardiso_thread_limit(args.host_threads)
        with threads:
            try:
                result = run_guiding_center_case(config, preset_key=prefix, step_observer=Observers(saver, movie, profiles),
                                                 terminal_log_path=out / f"{prefix}.log", case=case)
            finally:
                if movie is not None:
                    movie.close()
        snapshots = saver.save(out / f"{prefix}_snapshots.npz")
        profile_path = None if profiles is None else profiles.save(out / f"{prefix}_profiles.npz")
        rows = result.diagnostics
        first, last = rows[0], rows[-1]
        def series_max(key):
            values = [abs(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
            return max(values) if values else None
        summary = dict(prefix=prefix, geometry=geometry, kind=kind, triangles=int(mesh.num_tri),
                       element_unknowns=int(result.space.shape[0] * result.space.shape[1]),
                       order=args.order, dt=args.dt, steps=args.num_steps, final_time=args.num_steps * args.dt,
                       wall_time_s=time.perf_counter() - started, snapshots=None if snapshots is None else snapshots.name,
                       profiles=None if profile_path is None else profile_path.name,
                       transfer={k: v for k, v in first.items() if k.startswith("equilibrium_") and k != "equilibrium_import"},
                       initial={k: first.get(k) for k in ("diocotron_rho_eq_relative_l2", "diocotron_phi_eq_relative_l2",
                                                         "band_rho_dominant_mode", "band_rho_active_modes", "mass", "rho_min", "rho_max",
                                                         "linear_density_l2", "linear_potential_l2")},
                       final={k: last.get(k) for k in ("time", "mass_relative_drift", "energy_relative_drift",
                                                       "enstrophy_relative_drift", "rho_min", "rho_max",
                                                       "diocotron_rho_eq_relative_l2", "diocotron_phi_eq_relative_l2",
                                                       "band_rho_dominant_mode", "band_rho_active_modes",
                                                       "band_rho_participation", "linear_density_l2",
                                                       "linear_potential_l2")},
                       max_abs={k: series_max(k) for k in ("mass_relative_drift", "energy_relative_drift",
                                                          "l2_budget_relative_defect", "poisson_physical_rel_residual")},
                       seed=case.parameters.get("band"), provenance=provenance)
        (out / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        return summary
    finally:
        release_device_memory()


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for geometry in args.geometries:
            for kind in args.kinds:
                print(prefix_of(args, geometry, kind), json.dumps(case_params(args, geometry, kind)))
        return 0
    if args.seed_check:
        report = {}
        for geometry in args.geometries:
            case = build_case(args, geometry, "seed")
            band = case.imported_equilibrium.band
            check = band.check(band.parameters["stage"])
            report[geometry] = dict(parameters=band.parameters, check=check)
            seed_figure(case, args.output_dir / f"{args.prefix}_{geometry}_seed.png", check)
            print(geometry, json.dumps(check), flush=True)
        (args.output_dir / f"{args.prefix}_seed_check.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        return 0
    failures = 0
    for geometry in args.geometries:
        for kind in args.kinds:
            if args.skip_existing and completed(args, geometry, kind):
                print(f"skip completed run {prefix_of(args, geometry, kind)}", flush=True)
                continue
            try:
                summary = run_one(args, geometry, kind)
                print("SUMMARY " + json.dumps({k: summary[k] for k in ("prefix", "wall_time_s", "final")}, default=str),
                      flush=True)
            except Exception as error:
                failures += 1
                import traceback
                traceback.print_exc()
                print(f"FAILED {prefix_of(args, geometry, kind)}: {type(error).__name__}: {error}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
