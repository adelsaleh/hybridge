"""Diocotron benchmark campaigns for the guiding-center article.

SIM (time integration, operated by the user); ``--dry-run`` prints the run
configurations without building meshes or solving.

Benchmarks (all with phi = 0 on every wall and zero-flux transport):

``annulus-sl``
    Semi-Lagrangian comparison of Madaule et al. (2013) / Crouseilles et al.
    (2014), "Dirichlet for all modes": inner wall r = 0.1, outer wall r = 1
    (their r_min = 1, r_max = 10, scaled), rho = 1 on [r-, r+] with
    epsilon*cos(l*theta). Rows l = 2, 3, 4 on (0.4, 0.5), l = 5 on (0.7, 0.8),
    l = 7 on (0.6, 0.7). The band is meshed conformingly (no Gibbs effect at
    t = 0). Reference: :func:`interface_spectrum` (reproduces their analytic
    column); their tables list Im(omega/omega_D) = 2*gamma.
``disk-smooth``
    Disk m = 9, rho0 = exp(-((r-0.475)/0.025)^4) (no cutoff); reference: the
    Richardson-extrapolated Nystrom radial eigenvalue (gamma = 0.17966095).
    ``--modes`` seeds the same ring at other modes (dispersion gamma(m)).
``disk-zg``
    Zoni--Guclu (2019) section 6.3 profile (super-Gaussian p = 50 truncated at
    r = 0.45, 0.50), meshed conformingly; reference: the truncated-profile
    Nystrom rate 0.18024882 (:func:`converged_truncated_reference`), 0.34%
    above the sharp band of their eq. (34), 0.1796309594.
``disk-highmode``
    High azimuthal modes m = 64, 128 (``--modes``): Gaussian rings
    exp(-((r-0.8)/d)^2) with the radii of :func:`candidate_annulus`
    (m(1-(r-/r+)^2) = 1.59), epsilon = 1e-4; the mesh is graded around
    0.8 +- 2.5 ring widths, so the ring is resolved by many cells at a moderate
    total cost. References: Richardson-extrapolated Nystrom rates
    (:func:`converged_smooth_reference`: 0.18400849 for m = 64, 0.18509326
    for m = 128).
``disk-injection``
    The smooth m = 9 ring of ``disk-smooth`` with an explicit central density
    source (independent of rho): a normalized Gaussian of width
    ``--injection-width`` injecting ``--injection-charges`` (one run each, 0 =
    none) with a sin^2 window over ``--injection-start`` +
    [0, ``--injection-duration``]. Injected before the fit window, the central
    charge rotates the ring and lowers its rate (reference:
    :func:`converged_smooth_reference` with ``central_charge``); injected after
    the roll-up, it can stabilize the ring of nine vortices
    (:func:`point_vortex_ring_window`).
``disk-ionization``
    The same ring with electron-impact ionization of a prescribed neutral gas,
    S = nu(x) rho (one run per ``--ionization-rates`` peak rate): a Gaussian
    cloud of width ``--ionization-width`` at ``--ionization-center`` (uniform
    gas without a width), switched on at ``--ionization-start`` over
    ``--ionization-ramp``. A uniform gas maps every solution exactly onto the
    unforced one, rho(x, t) = exp(nu t) r(x, (exp(nu t) - 1)/nu); without a
    seed (``--epsilon 0``) an axisymmetric gas gives the exact density
    rho0(r) exp(int_0^t nu), recorded as ``rho_l2_error``. Every run records
    int nu_h rho_h (``reaction_production``), the reaction term of the
    discrete mass balance.

Every run uses the fastest recorded device configuration
(:mod:`scripts.guiding_center.runtime.fast_profiles`), a fixed Poisson tau
(no tau recovery: a failure is recorded as a result) and conflict-averaged
upwind transport. ``--velocities`` selects the raw HDG drift and/or the
compatible (Oswald-averaged potential) drift. A summary JSON/CSV is written
after every completed run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import numpy as np
from dataclasses import asdict, replace
from pathlib import Path

from scripts.guiding_center.diagnostics.diocotron_reference import (
    SEMI_LAGRANGIAN_DIRICHLET_ANNULUS,
    SEMI_LAGRANGIAN_INNER_WALL,
    candidate_annulus,
)
from scripts.guiding_center.runtime.device import release_device_memory
from scripts.guiding_center.runtime.observers import MovieObserver, Observers, SnapshotSaver

BASE_PRESET = "diocotron_gaussian_m64_si_bdf2_p6_h0068_dt05_t400"
BENCHMARKS = ("annulus-sl", "disk-smooth", "disk-zg", "disk-highmode", "disk-injection", "disk-ionization")
VELOCITIES = {"raw": ("raw", "auto"), "compatible": ("compatible", "auto")}


def _benchmark_cases(benchmark: str, *, modes, epsilon: float | None, injection: dict | None = None,
                     ionization: dict | None = None):
    """Yield (label, case_params, fit_window, extra) for one benchmark."""
    if benchmark == "annulus-sl":
        rows = [row for row in SEMI_LAGRANGIAN_DIRICHLET_ANNULUS if not modes or row["mode"] in modes]
        for row in rows:
            params = dict(k=row["mode"], s_minus=row["inner"], s_plus=row["outer"],
                          inner_wall=SEMI_LAGRANGIAN_INNER_WALL, wall=1.0,
                          epsilon=1.0e-6 if epsilon is None else epsilon)
            extra = dict(semi_lagrangian_twice_gamma=row["semi_lagrangian"],
                         analytic_twice_gamma=row["analytic"])
            yield f"l{row['mode']}_r{row['inner']:g}_{row['outer']:g}", params, tuple(row["fit_window"]), extra
        return
    common = dict(k=9, s_minus=.45, s_plus=.50, inner_wall=0.0, wall=1.0,
                  epsilon=1.0e-4 if epsilon is None else epsilon)
    if benchmark == "disk-smooth":
        # D2: the same ring seeded at other modes (``--modes``); D1 is m = 9.
        for mode in (modes or (9,)):
            yield (f"m{int(mode)}_smooth_p4", dict(common, k=int(mode), p=4.0, truncate=False, refine_band=(.35, .60)),
                   (20.0, 50.0), {})
        return
    if benchmark == "disk-injection":
        injection = dict(injection or {})
        for charge in injection.get("charges", (0.1,)):
            params = dict(common, p=4.0, truncate=False, refine_band=(.35, .60))
            center = injection.get("center")
            if charge:
                params.update(injection_charges=(float(charge),), injection_width=injection.get("width", .06),
                              injection_start=injection.get("start", 0.), injection_duration=injection.get("duration", 4.),
                              injection_centers=None if center is None else (tuple(center),))
            label = (f"m9_inject_q{charge:g}_t{injection.get('start', 0.):g}"
                     + ("" if center is None else f"_x{center[0]:g}_y{center[1]:g}"))
            yield label, params, (20.0, 50.0), {}
        return
    if benchmark == "disk-ionization":
        ionization = dict(ionization or {})
        width, center = ionization.get("width"), tuple(ionization.get("center") or (0.0, 0.0))
        start, ramp = ionization.get("start", 0.0), ionization.get("ramp", 0.0)
        for rate in ionization.get("rates", (0.01,)):
            params = dict(common, p=4.0, truncate=False, refine_band=(.35, .60), ionization_rate=float(rate),
                          ionization_width=width, ionization_center=center, ionization_start=start,
                          ionization_ramp=ramp)
            label = (f"m9_ionize_nu{rate:g}" + ("" if width is None else f"_w{width:g}_x{center[0]:g}_y{center[1]:g}")
                     + (f"_t{start:g}" if start or ramp else ""))
            yield label, params, (20.0, 50.0), {}
        return
    if benchmark == "disk-zg":
        yield "m9_zg_p50", dict(common, p=50.0, truncate=True), (20.0, 50.0), {}
        return
    if benchmark == "disk-highmode":
        for mode in (modes or (64, 128)):
            ring = candidate_annulus(int(mode))
            width = ring["outer"] - ring["inner"]
            params = dict(k=int(mode), s_minus=ring["inner"], s_plus=ring["outer"], inner_wall=0.0, wall=1.0,
                          epsilon=1.0e-4 if epsilon is None else epsilon, p=2.0, truncate=False,
                          refine_band=(ring["center"] - 2.5 * width, ring["center"] + 2.5 * width),
                          band_grading=0.5)
            yield f"m{int(mode)}_gaussian", params, (20.0, 45.0), dict(diocotron_radial_points=64)
        return
    raise ValueError(f"unknown benchmark {benchmark!r}; choose one of {BENCHMARKS}")


def _num_steps(final_time: float, dt: float) -> int:
    steps = int(round(final_time / dt))
    if steps < 1 or not math.isclose(steps * dt, final_time, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"final_time={final_time} must be an integer multiple of dt={dt}")
    return steps


def build_configs(args) -> list[dict]:
    """Return one record (config + metadata) per run of the campaign."""
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.fast_profiles import (
        default_transport_candidate, device_candidate_overrides, fastest_device_overrides)

    base = preset_by_key(BASE_PRESET)
    output_dir = Path(args.output_dir)
    records = []
    injection = dict(charges=args.injection_charges, start=args.injection_start,
                     duration=args.injection_duration, width=args.injection_width, center=args.injection_center)
    ionization = dict(rates=args.ionization_rates, width=args.ionization_width, center=args.ionization_center,
                      start=args.ionization_start, ramp=args.ionization_ramp)
    for label, case_params, window, extra in _benchmark_cases(
            args.benchmark, modes=args.modes, epsilon=args.epsilon, injection=injection, ionization=ionization):
        settings = {key: extra.pop(key) for key in ("diocotron_radial_points",) if key in extra}
        for order in args.orders:
            offset = int(args.poisson_order_offset)
            if args.poisson_candidate is None:
                if args.transport_candidate is not None or offset:
                    raise ValueError("--transport-candidate and --poisson-order-offset need --poisson-candidate")
            for band_size in args.band_mesh_sizes:
                triangles = _triangle_count(case_params, band_size, args.mesh_size)
                device = (fastest_device_overrides(order, zero_dirichlet=True, triangles=triangles)
                          if args.poisson_candidate is None else device_candidate_overrides(
                              args.poisson_candidate, args.transport_candidate or default_transport_candidate(order),
                              order=order, poisson_order=order + offset))
                for scheme in args.schemes:
                    for dt in args.dts:
                        for velocity in args.velocities:
                            field, potential = VELOCITIES[velocity]
                            seeds = [case_params["epsilon"]] + ([0.0] if args.controls and not args.linearized else [])
                            for seed in seeds:
                                params = dict(case_params, epsilon=seed, band_mesh_size=band_size)
                                tag = "control" if seed == 0.0 else "linear" if args.linearized else "seed"
                                run_tag = f"_{args.run_tag}" if args.run_tag else ""
                                kkt = ("" if args.density_positivity != "kkt" else
                                       "_kkt" + ("dense" if args.density_positivity_points.endswith("dense") else ""))
                                prefix = (f"{args.prefix}_{args.benchmark.replace('-', '_')}_{label}_p{order}_"
                                          f"hb{band_size:g}_h{args.mesh_size:g}_{scheme.replace('-', '_')}_"
                                          f"dt{dt:g}_{velocity}{kkt}{run_tag}_{tag}").replace(".", "p")
                                config = replace(
                                    base,
                                    **device,
                                    **settings,
                                    description=f"article diocotron {args.benchmark} {label}",
                                    case="diocotron_annulus",
                                    case_params=params,
                                    domain="annulus",
                                    mesh_size=float(args.mesh_size),
                                    minimum_triangles=0,
                                    order=int(order),
                                    dt=float(dt),
                                    num_steps=_num_steps(args.final_time, dt),
                                    time_scheme=scheme,
                                    poisson_tau=float(args.poisson_tau),
                                    poisson_tau_max_retries=0,
                                    transport_electric_field=field,
                                    compatible_potential=(potential if args.compatible_potential is None
                                                          else args.compatible_potential),
                                    poisson_order_offset=offset,
                                    volume_quad_1d=args.volume_quad_1d,
                                    synchronize_step_timing=bool(args.synchronize_step_timing),
                                    **{key: float(value) for key, value in (
                                        ("poisson_solver_rtol", args.poisson_solver_rtol),
                                        ("poisson_solver_atol", args.poisson_solver_atol)) if value is not None},
                                    transport_advection_stabilization="conflict-averaged-upwind",
                                    diocotron_diagnostics=True,
                                    positivity_diagnostics=not args.linearized,
                                    linearized=bool(args.linearized),
                                    enstrophy_budget=bool(args.enstrophy_budget and scheme in ("si-euler", "si-bdf2")
                                                          and not args.linearized),
                                    density_positivity=args.density_positivity,
                                    density_positivity_points=args.density_positivity_points,
                                    diagnostics_every=int(args.diagnostics_every),
                                    record_timings=True,
                                    # Conservation, positivity, modal and solver figures per run
                                    # (JSONL/CSV records need only diagnostics_every > 0).
                                    save_diagnostics=True,
                                    plot_every=0,
                                    plot_diagnostics=False,
                                    save_movie=False,
                                    screenshot_dir=None,
                                    verbosity=int(args.verbosity),
                                    diagnostics_dir=str(output_dir),
                                    diagnostics_prefix=prefix,
                                )
                                if args.fit_window is not None:
                                    window = tuple(args.fit_window)
                                records.append(dict(config=config, prefix=prefix, label=label, order=order,
                                                    band_mesh_size=band_size, scheme=scheme, dt=dt,
                                                    triangles=triangles,
                                                    velocity=velocity, epsilon=seed, window=window,
                                                    poisson_candidate=args.poisson_candidate,
                                                    transport_candidate=(
                                                        None if args.poisson_candidate is None else
                                                        args.transport_candidate or default_transport_candidate(order)),
                                                    **extra))
    return records


def _run_mesh(case_params: dict, band_size: float, mesh_size: float):
    """The run's mesh (built once; the runner then hits the gmsh mesh cache)."""
    from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
    from hybridge.core.mesh import gmsh_annulus_mesh

    case = case_definition_by_key("diocotron_annulus").build(**dict(case_params, band_mesh_size=band_size))
    geometry = case.parameters["geometry"]
    return gmsh_annulus_mesh(
        float(mesh_size), inner_radius=geometry["inner_radius"], outer_radius=geometry["outer_radius"],
        interface_radii=tuple(geometry["interface_radii"]), band=geometry["band"],
        band_mesh_size=geometry["band_mesh_size"], band_grading=geometry["band_grading"], log_cache=False)


def _triangle_count(case_params: dict, band_size: float, mesh_size: float) -> int:
    """Triangles of the run's mesh."""
    return int(_run_mesh(case_params, band_size, mesh_size).num_tri)


def _completed_run(output_dir: Path, record):
    """Stand-in result for a run whose diagnostics already reach the final step, else None."""
    from types import SimpleNamespace

    path = output_dir / f"{record['prefix']}.jsonl"
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines or json.loads(lines[-1]).get("step") != record["config"].num_steps:
        return None
    timings = output_dir / f"{record['prefix']}_timings.jsonl"
    mesh = _run_mesh(record["config"].case_params, record["band_mesh_size"], record["config"].mesh_size)
    return SimpleNamespace(jsonl_path=path, timings_jsonl_path=timings if timings.exists() else None,
                           space=SimpleNamespace(mesh=mesh))


# Per-step records summarized by their median over the accepted steps 3..N.
_STEP_TIMING_KEYS = (
    "linear_step_wall_time", "poisson_step_wall_time", "transport_step_wall_time", "step_advance_wall_time",
    "enstrophy_budget_time", "beta_build_time", "potential_trace_update_time",
    "poisson_time_rhs_assembly", "poisson_step_time_solve", "poisson_krylov_time",
    "poisson_step_time_reconstruction", "poisson_flux_postprocess_time", "poisson_step_iterations",
    "transport_step_time_assembly", "transport_step_time_solve", "transport_step_time_reconstruction",
    "transport_step_iterations", "poisson_detail_postprocess_primal_operator",
    "poisson_detail_postprocess_primal_cache_bytes")
# The cold first Poisson solve (zero initial guess) and other one-off costs of the initial row.
_INITIAL_TIMING_KEYS = (
    "first_poisson_time_operator_assembly", "first_poisson_time_rhs_assembly", "first_poisson_time_solve",
    "first_poisson_krylov_time", "first_poisson_solver_iterations", "first_poisson_time_reconstruction",
    "first_poisson_time_postprocessing", "first_poisson_wall_time", "equilibrium_poisson_wall_time",
    "initial_density_projection_time")


def _step_timings(result) -> dict:
    """Median per-step wall times and iterations (accepted steps 3..N), the interquartile range of the
    step totals, and the one-off costs of the initial row (cold first Poisson solve), from the timing records."""
    import statistics

    path = getattr(result, "timings_jsonl_path", None)
    if path is None or not Path(path).exists():
        return {}
    rows_all = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    initial = next((row for row in rows_all if row.get("phase") == "initial"), {})
    rows = [row for row in rows_all if row.get("phase") == "step"][2:]
    timings = dict(timed_steps=len(rows), step_timing_synchronized=bool(rows and rows[0].get("step_timing_synchronized")))
    for key in _STEP_TIMING_KEYS:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            timings[f"median_{key}"] = statistics.median(values)
        if key in ("linear_step_wall_time", "step_advance_wall_time") and len(values) >= 4:
            quartiles = statistics.quantiles(values, n=4)
            timings[f"iqr_{key}"] = quartiles[2] - quartiles[0]
    timings.update({key: initial[key] for key in _INITIAL_TIMING_KEYS if key in initial})
    every = [row for row in rows_all if row.get("phase") == "step"]

    def ratio(row, solver):
        residual, target = row.get(f"{solver}_solver_residual"), row.get(f"{solver}_solver_residual_target")
        return (float(residual) / float(target) if isinstance(residual, (int, float))
                and isinstance(target, (int, float)) and target > 0 else None)

    # The device p+1 postprocess builds its cached operator once (V2): the largest build time over all records.
    builds = [float(row["poisson_detail_postprocess_primal_build_seconds"]) for row in rows_all
              if isinstance(row.get("poisson_detail_postprocess_primal_build_seconds"), (int, float))]
    if builds:
        timings["postprocess_primal_build_seconds"] = max(builds)
    # Solver health over every step: attempts beyond the first, fb-hp-mg fallbacks, tau retries,
    # and the final residual against its target (<= 1 when converged).
    timings.update(
        max_transport_attempts=max((int(row.get("transport_amgx_attempt_count") or row.get("transport_attempt_count")
                                        or 1) for row in every), default=None),
        poisson_fallbacks=sum(bool(row.get("poisson_detail_solve_fb_hp_mg_fallback")) for row in every),
        poisson_tau_retries=sum(int(row.get("poisson_tau_retry_count") or 0) for row in every),
        max_poisson_residual_ratio=max((value for value in (ratio(row, "poisson") for row in every)
                                        if value is not None), default=None),
        max_transport_residual_ratio=max((value for value in (ratio(row, "transport") for row in every)
                                          if value is not None), default=None))
    return timings


def _run_checks(run) -> dict:
    """Largest relative mass drift and L2-identity defect over the run, and the smallest sampled density."""
    rows = run["rows"]

    def extreme(key, reduce=max, magnitude=True):
        values = [abs(row[key]) if magnitude else row[key] for row in rows if isinstance(row.get(key), (int, float))]
        return reduce(values) if values else None

    diagnostics = sorted(row["diagnostics_time"] for row in rows[1:] if isinstance(row.get("diagnostics_time"), (int, float)))
    return dict(median_diagnostics_time=diagnostics[len(diagnostics) // 2] if diagnostics else None,
                max_mass_relative_drift=extreme("mass_relative_drift"),
                max_l2_budget_relative_defect=extreme("l2_budget_relative_defect"),
                max_energy_relative_drift=extreme("energy_relative_drift"),
                min_rho=extreme("rho_min", min, magnitude=False), final_time=float(run["times"][-1]))


def _summary_row(record, result, wall_time, control_path=None) -> dict:
    """Fit growth and audit invariants for one completed seeded run.

    Runs too short for the fit window (timing smoke runs) keep every other field and record the
    analysis error instead of the growth fields.
    """
    from scripts.guiding_center.studies.diocotron.analyze_diocotron import analyze, read_run

    run = read_run(result.jsonl_path)
    control = read_run(control_path) if control_path is not None else None
    config = record["config"]
    row = dict(
        prefix=record["prefix"], label=record["label"], order=record["order"],
        band_mesh_size=record["band_mesh_size"], mesh_size=config.mesh_size,
        triangles=run["config"].get("triangles"), scheme=record["scheme"], dt=record["dt"],
        velocity=record["velocity"], epsilon=record["epsilon"], window=list(record["window"]),
        poisson_candidate=record.get("poisson_candidate"), transport_candidate=record.get("transport_candidate"),
        poisson_order_offset=config.poisson_order_offset, compatible_potential=config.compatible_potential,
        volume_quad_1d=config.volume_quad_1d, num_steps=config.num_steps,
        wall_time_s=wall_time, jsonl=str(result.jsonl_path),
        element_unknowns=int(result.space.mesh.num_tri) * (record["order"] + 1) * (record["order"] + 2) // 2,
        trace_unknowns=int(result.space.mesh.num_edg) * (record["order"] + 1),
        **_run_checks(run), **_step_timings(result),
    )
    try:
        summary = analyze(run, control=control, window=record["window"])
    except Exception as error:
        row["analysis_error"] = f"{type(error).__name__}: {error}"
        return row
    growth = summary["growth"]
    target = growth["target_mode"]
    gamma = growth["reference"]["growth_rate"]
    row.update(
        reference_type=growth["reference"].get("reference_type"), reference_gamma=gamma,
        sharp_reference_gamma=growth["sharp_reference"]["growth_rate"],
        fitted_gamma=target.get("growth_rate"), relative_error=target.get("relative_error"),
        potential_norm_gamma=growth["paper_potential_norm"].get("growth_rate"),
        potential_norm_relative_error=growth["paper_potential_norm"].get("relative_error"),
        r_squared=target.get("r_squared"),
        subwindow_gammas=[fit.get("growth_rate") for fit in growth["subwindows"]],
        control_fraction=(growth.get("control") or {}).get("max_control_fraction"),
        positivity=summary["positivity"]["status"],
        density_minimum=summary["positivity"].get("minimum"),
        invariants={key: value.get("max_relative_drift") for key, value in summary["invariants"].items()},
    )
    if "semi_lagrangian_twice_gamma" in record:
        semi = record["semi_lagrangian_twice_gamma"] / 2
        row.update(semi_lagrangian_gamma=semi,
                   semi_lagrangian_relative_error=(semi - gamma) / gamma if gamma else None,
                   published_analytic_gamma=record["analytic_twice_gamma"] / 2)
    return row


def _write_summary(rows, failures, output_dir: Path, prefix: str) -> None:
    """Write <prefix>_summary.json/csv, merging with an existing summary by run prefix.

    Several invocations may share a campaign prefix (e.g. a later variant of one
    run): their rows accumulate, and a rerun of the same run replaces its row.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_summary.json"
    if path.exists():
        previous = json.loads(path.read_text())
        current = {row["prefix"] for row in rows} | {failure.get("prefix") for failure in failures}
        rows = [row for row in previous.get("rows", []) if row["prefix"] not in current] + list(rows)
        failures = [failure for failure in previous.get("failures", [])
                    if failure.get("prefix") not in current] + list(failures)
    path.write_text(json.dumps(dict(rows=rows, failures=failures), indent=2, default=str) + "\n")
    if rows:
        flat = [{key: (json.dumps(value) if isinstance(value, (list, dict)) else value)
                 for key, value in row.items()} for row in rows]
        fields = list(dict.fromkeys(key for row in flat for key in row))
        with (output_dir / f"{prefix}_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)



def run_campaign(args) -> None:
    from scripts.guiding_center.runtime.runner import run_guiding_center_case

    records = build_configs(args)
    if args.dry_run:
        for record in records:
            config = asdict(record["config"])
            print(json.dumps(dict(prefix=record["prefix"], window=record["window"], triangles=record["triangles"],
                                  **{key: config[key] for key in (
                                      "case", "case_params", "mesh_size", "order", "dt", "num_steps",
                                      "diocotron_radial_points",
                                      "time_scheme", "poisson_tau", "poisson_solver",
                                      "poisson_fb_hp_mg_preconditioner_policy", "transport_raw_matrix_format",
                                      "transport_electric_field")}), default=str))
        print(f"{len(records)} runs")
        return
    output_dir = Path(args.output_dir)
    rows, failures, controls = [], [], {}
    # Controls first, so every seeded run can be compared with its matched control.
    for record in sorted(records, key=lambda item: item["epsilon"] != 0.0):
        started = time.perf_counter()
        kind = "control" if record["epsilon"] == 0.0 else "seed"
        if args.only is not None and kind != args.only:
            # Not run here; a completed control still serves as the seed's reference.
            completed = _completed_run(output_dir, record)
            if completed is not None and kind == "control":
                controls[record["prefix"].rsplit("_", 1)[0]] = completed.jsonl_path
            continue
        completed = _completed_run(output_dir, record) if args.skip_existing else None
        if completed is not None:
            print(f"skip completed run {record['prefix']}", flush=True)
            key = record["prefix"].rsplit("_", 1)[0]
            if record["epsilon"] == 0.0:
                controls[key] = completed.jsonl_path
            else:
                try:
                    rows.append(_summary_row(record, completed, None, control_path=controls.get(key)))
                except Exception as error:
                    failures.append(dict(prefix=record["prefix"], stage="analysis",
                                         error=f"{type(error).__name__}: {error}"))
            _write_summary(rows, failures, output_dir, args.prefix)
            continue
        saver = (SnapshotSaver(args.snapshot_times, record["dt"], density_only=args.snapshot_density_only)
                 if args.snapshot_times and record["epsilon"] != 0.0 else None)
        movie = None
        if args.movie_interval > 0:
            # Each movie panel is ~800 px wide; 1024 px rasters keep the p = 6 sampling map
            # within the 512 MiB limit of RasterGeometry.sampling_matrix.
            raster = args.movie_raster or 1024
            label = (f"{record['label']}  p={record['order']}  h_band={record['band_mesh_size']:g}  "
                     f"{record['triangles']} triangles  "
                     f"{record['scheme']}  dt={record['dt']:g}  {record['velocity']} drift  "
                     f"eps={record['epsilon']:g}")
            movie = MovieObserver(output_dir / "movies" / f"{record['prefix']}.mp4",
                                   interval=args.movie_interval, dt=record["dt"], density_scale=1.0,
                                   label=label, raster=raster, fps=args.movie_fps)
        try:
            try:
                result = run_guiding_center_case(record["config"], preset_key=record["prefix"],
                                                 step_observer=Observers(saver, movie))
            finally:
                if movie is not None:
                    movie.close()
                release_device_memory()
            if saver is not None:
                saver.save(output_dir / f"{record['prefix']}_snapshots.npz")
        except Exception as error:  # a failed run at a fixed tau is a result
            failures.append(dict(prefix=record["prefix"], error=f"{type(error).__name__}: {error}"))
            print(f"FAILED {record['prefix']}: {type(error).__name__}: {error}", flush=True)
            _write_summary(rows, failures, output_dir, args.prefix)
            continue
        wall = time.perf_counter() - started
        key = record["prefix"].rsplit("_", 1)[0]
        if record["epsilon"] == 0.0:
            controls[key] = result.jsonl_path
        else:
            try:
                row = _summary_row(record, result, wall, control_path=controls.get(key))
            except Exception as error:  # the run's records are complete; keep the campaign going
                failures.append(dict(prefix=record["prefix"], stage="analysis",
                                     error=f"{type(error).__name__}: {error}"))
                print(f"ANALYSIS FAILED {record['prefix']}: {type(error).__name__}: {error}", flush=True)
            else:
                rows.append(row)
                if "analysis_error" in row:
                    print(f"{record['prefix']}: no growth fit ({row['analysis_error']}); step "
                          f"{row.get('median_step_advance_wall_time', row.get('median_linear_step_wall_time'))} s",
                          flush=True)
                else:
                    print(f"{record['prefix']}: gamma={row['fitted_gamma']} reference={row['reference_gamma']:.8f} "
                          f"relative={row['relative_error']} R2={row['r_squared']}", flush=True)
        _write_summary(rows, failures, output_dir, args.prefix)
        # Drop the finished run's device fields before the next (possibly larger) run.
        result = None
        release_device_memory()


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value)


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(",") if value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--modes", type=_ints, default=(),
                        help="annulus-sl rows or disk-highmode modes to run (default: all / 64,128)")
    parser.add_argument("--orders", type=_ints, default=(6,))
    parser.add_argument("--mesh-size", type=float, default=0.04, help="mesh size far from the band")
    parser.add_argument("--band-mesh-sizes", type=_floats, default=(0.01,), help="mesh sizes on the band (ladder)")
    parser.add_argument("--dts", type=_floats, default=(0.1,))
    parser.add_argument("--schemes", type=lambda text: tuple(text.split(",")), default=("si-bdf2",))
    parser.add_argument("--velocities", type=lambda text: tuple(text.split(",")), default=("raw",))
    parser.add_argument("--epsilon", type=float, default=None,
                        help="seed amplitude (default 1e-6 annulus, 1e-4 disk)")
    parser.add_argument("--final-time", type=float, default=None, help="default 80 (annulus) or 64 (disk)")
    parser.add_argument("--density-positivity", choices=("none", "kkt"), default="none",
                        help="KKT positivity projection of every transported density (SI-Euler/SI-BDF2)")
    parser.add_argument("--density-positivity-points", choices=("quadrature+lattice", "quadrature+lattice+dense"),
                        default="quadrature+lattice",
                        help="KKT enforcement points; 'dense' (181 at p=6) also covers the (2p+2) diagnostic lattice")
    parser.add_argument("--injection-charges", type=_floats, default=(0.1,),
                        help="disk-injection: central charge per run (0 = no injection)")
    parser.add_argument("--injection-start", type=float, default=0.0, help="disk-injection: start of the window")
    parser.add_argument("--injection-duration", type=float, default=4.0, help="disk-injection: window length")
    parser.add_argument("--injection-width", type=float, default=0.06, help="disk-injection: Gaussian width")
    parser.add_argument("--injection-center", type=_floats, default=None,
                        help="disk-injection: x,y of a fixed (lab-frame) injector (default: the centre)")
    parser.add_argument("--ionization-rates", type=_floats, default=(0.01,),
                        help="disk-ionization: peak ionization rate nu per run")
    parser.add_argument("--ionization-width", type=float, default=None,
                        help="disk-ionization: Gaussian width of the gas cloud (default: uniform gas)")
    parser.add_argument("--ionization-center", type=_floats, default=None,
                        help="disk-ionization: x,y of the gas cloud (default: the centre)")
    parser.add_argument("--ionization-start", type=float, default=0.0, help="disk-ionization: switch-on time")
    parser.add_argument("--ionization-ramp", type=float, default=0.0,
                        help="disk-ionization: sin^2 switch-on duration (0 = at once)")
    parser.add_argument("--poisson-tau", type=float, default=32000.0)
    parser.add_argument("--controls", action="store_true", help="also run matched epsilon=0 controls")
    parser.add_argument("--linearized", action="store_true",
                        help="evolve the equations linearized about the ring (runtime/linearized.py): the "
                             "response to q_0 = epsilon*rho*(r) cos(m theta), exactly linear in epsilon")
    parser.add_argument("--diagnostics-every", type=int, default=1)
    parser.add_argument("--movie-interval", type=float, default=0.5,
                        help="MP4 frame interval in time units for every run (0 disables movies)")
    parser.add_argument("--movie-raster", type=int, default=None,
                        help="raster pixels per side (default 1024)")
    parser.add_argument("--movie-fps", type=float, default=10.0)
    parser.add_argument("--snapshot-times", type=_floats, default=(),
                        help="save rho_h and phi_h coefficients of seeded runs at these times (NPZ)")
    parser.add_argument("--snapshot-density-only", action="store_true",
                        help="store only rho_h in the snapshots (no phi_h, q_h, phi_h^eq)")
    parser.add_argument("--output-dir", default="run_outputs/article/diocotron")
    parser.add_argument("--prefix", default="art")
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--enstrophy-budget", action="store_true",
                        help="record D_h, G_h and the L2 identity defect (SI-Euler/SI-BDF2 runs)")
    parser.add_argument("--only", choices=("seed", "control"), default=None,
                        help="run only the seeded runs or only the controls (a completed control is still used)")
    parser.add_argument("--fit-window", type=_floats, default=None,
                        help="declared fit window a,b of the summary (default per benchmark)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="reuse runs whose diagnostics already reach the final step (no rerun)")
    parser.add_argument("--poisson-candidate", default=None,
                        help="named Poisson candidate of fast_profiles.POISSON_CANDIDATES with '+' modifiers "
                             "(any p = 1..9) instead of the fastest recorded profile (p = 4..6)")
    parser.add_argument("--transport-candidate", default=None,
                        help="named transport candidate ('fused'|'split3' with '+csr', '+bsr', '+dilu'); "
                             "default fast_profiles.default_transport_candidate(p); needs --poisson-candidate")
    parser.add_argument("--poisson-order-offset", type=int, default=0,
                        help="Poisson degree q = p + offset (raw drift; needs --poisson-candidate)")
    parser.add_argument("--compatible-potential", choices=("auto", "postprocessed", "primal"), default=None,
                        help="potential averaged by the compatible drift (default auto = postprocessed, V2)")
    parser.add_argument("--volume-quad-1d", type=int, default=None,
                        help="collapsed volume rule with n^2 points instead of the default rule (exact to 2n-1)")
    parser.add_argument("--poisson-solver-rtol", type=float, default=None,
                        help="Poisson stopping rule, relative to the right-hand side (default of the preset, 1e-11)")
    parser.add_argument("--poisson-solver-atol", type=float, default=None,
                        help="Poisson stopping rule, absolute residual (default of the preset, 1e-10; the reduced "
                             "right-hand side of the diocotron systems has norm 4-8e-5, so 1e-10 is only about "
                             "1e-6 relative; D8 uses 1e-14)")
    parser.add_argument("--synchronize-step-timing", action="store_true",
                        help="synchronize the device at the end of every step (performance runs)")
    parser.add_argument("--run-tag", default=None, help="tag inserted in every run prefix before seed/control")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.final_time is None:
        args.final_time = 80.0 if args.benchmark == "annulus-sl" else 64.0
    unknown = set(args.velocities) - set(VELOCITIES)
    if unknown:
        parser.error(f"unknown velocities {sorted(unknown)}; choose from {sorted(VELOCITIES)}")
    run_campaign(args)


if __name__ == "__main__":
    main()
