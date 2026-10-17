#!/usr/bin/env python3
"""Pure Poisson comparison of ``fb-hp-mg-pcg`` with the AMGX candidates at p = 5..9.

Performance diagnostic (Poisson solves only, no time integration): on the
diocotron campaign's own Poisson systems (``disk-highmode --modes 128``,
annulus band meshes of ``mesh_levels_smoke.jsonl``, tau = 128000, phi = 0 on
both walls, source = the density) it measures, per (p, level, candidate):

* the cold solve: operator assembly, solver setup (native hierarchy or AMGX
  hierarchy) and the first Krylov solve from a zero guess, its iterations and
  final true residual, then a second zero-guess solve with the setup reused
  (``cold_repeat``: Krylov cost alone);
* ``--warm-solves`` solves of slightly different right-hand sides (equilibrium
  density times ``1 + delta * cos(m theta + phase)``, a new low mode and
  phase per solve) from the previous solution, as in the linear phase of the
  time loop (1-2 iterations for every solver): median iterations and times;
* ``warm_saturated`` (levels with a stage H snapshot, default L2): the same
  from the saturated density of the smoke campaign's stage H run of the same
  p and mesh (``*_H_<family><level>_p<p>_*_seed_snapshots.npz``, t = 25,
  checked against the built mesh, degree and basis): sources
  ``rho_eq + s_k (rho_sat - rho_eq)`` with ``s_k = 0.80, 0.82, ..., 0.98``,
  i.e. O(1 %) changes between consecutive solves as in the saturated phase,
  where the Chebyshev AMGX configs needed 29-41 iterations per step;
* the peak device memory of the candidate (in-process ``memGetInfo`` sampler);
* for ``fb-hp-mg-pcg`` the per-level setup diagnostics (lambda estimates,
  kernels) and the cold residual history; at ``-v 3`` the solver prints its
  iteration table (``verbose=3``) and AMGX prints its residual rows.

The problem, trace basis, stabilization, solver options and stopping rule are
those of the campaign: the candidate overrides come from
``scripts.guiding_center.runtime.fast_profiles.device_candidate_overrides``,
the options from ``scripts.guiding_center.runtime.configuration._make_poisson_options``
(rtol 1e-11, atol 1e-10; the ABSOLUTE AMGX configs stop at the absolute 1e-10,
the native solver at ``max(atol, rtol*|b|)``), the mesh and density from the
guiding-center runner helpers. Solutions of the candidates of one system are
compared (``max_rel_diff_vs_ref``: max |phi - phi_ref| / max |phi_ref| against
the first candidate's cold and last warm solutions).

Outputs (``--output-dir``): ``results.jsonl`` (one row per (p, level,
candidate), appended as soon as it is measured; finished rows are skipped on a
restart), ``summary.md`` (regenerated from the rows), ``<run_id>.log`` (the
terminal output, line-buffered, native AMGX output included). ``--dry-run``
prints the plan without touching the device. Run it under the GPU campaign
lock (``scripts/guiding_center/benchmarks/campaign_protocol.py``), one
process per (p, level), e.g. from a study launcher::

    .venv/bin/python -u -m scripts.diffusion_reaction.benchmark_poisson_pmg_degrees \\
        --degrees 9 --levels L3 --output-dir <study>/runs -v 3
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESH_LEVELS = Path(
    "/home/adelsaleh/src/gc-hdg-article/run_outputs/diocotron_order_smoke_20261016/mesh_levels_smoke.jsonl")
DEFAULT_CANDIDATES = ("fbmg-standard", "amgx-cheb-csr", "amgx-classical-abs-csr")
DEFAULT_SATURATED_DIR = DEFAULT_MESH_LEVELS.parent / "runs"
SATURATED_BLENDS = tuple(0.80 + 0.02 * k for k in range(10))
NATIVE_SOLVER = "fb-hp-mg-pcg"


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def _fmt(value, spec=".0f", scale=1.0):
    """``value * scale`` formatted with ``spec`` ('-' for None); ',d' formats integers."""
    if value is None:
        return "-"
    try:
        if spec.endswith("d"):
            return format(int(value), spec)
        return format(float(value) * scale, spec)
    except (TypeError, ValueError):
        return str(value)


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.replace(";", ",").split(",") if item.strip())


def _names(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.replace(";", ",").split(",") if item.strip())


def read_mesh_levels(path: Path, family: str) -> dict[tuple[str, int], dict]:
    """``{(level, p): row}`` of one mesh family of ``mesh_levels_smoke.jsonl``."""
    levels = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("family") == family:
            levels[(str(row["level"]), int(row["p"]))] = row
    return levels


def run_id_of(level: str, p: int, candidate: str, tag: str) -> str:
    return f"{level}_p{p}_{candidate}" + (f"_{tag}" if tag else "")


def finished_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    rows[row["run_id"]] = row
    return rows


# --------------------------------------------------------------------------------------------------
# problem construction (the campaign's helpers)


def campaign_config(args, *, p: int, hb: float, h: float, candidate: str):
    """The diocotron runner's preset for one candidate: ``build_configs`` without the time loop."""
    from scripts.guiding_center.benchmarks.run_diocotron_benchmark import BASE_PRESET, _benchmark_cases
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.fast_profiles import default_transport_candidate, device_candidate_overrides

    cases = list(_benchmark_cases(args.benchmark, modes=(int(args.modes),), epsilon=None))
    if len(cases) != 1:
        raise ValueError(f"expected one case for {args.benchmark} mode {args.modes}, got {len(cases)}")
    label, case_params, _window, _extra = cases[0]
    device = device_candidate_overrides(candidate, default_transport_candidate(p), order=p)
    config = replace(
        preset_by_key(BASE_PRESET), **device,
        description=f"Poisson pMG degree study {args.benchmark} {label}",
        case="diocotron_annulus", case_params=dict(case_params, band_mesh_size=float(hb)), domain="annulus",
        mesh_size=float(h), minimum_triangles=0, order=int(p), poisson_tau=float(args.tau),
        poisson_tau_max_retries=0, poisson_order_offset=0, verbosity=int(args.verbosity),
        record_timings=True, diagnostics_every=0, plot_every=0, plot_diagnostics=False, save_diagnostics=False,
        positivity_diagnostics=False, diocotron_diagnostics=False, enstrophy_budget=False,
        save_movie=False, screenshot_dir=None, diagnostics_dir=str(args.output_dir),
        diagnostics_prefix=f"pmg_{label}_p{p}")
    return config, label


def build_system(args, *, p: int, level_row: dict, candidate: str):
    """Mesh, DG space, case, equilibrium density of the campaign's (p, level) Poisson problem."""
    from hybridge.core.space import DGSpace
    from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
    from scripts.guiding_center.runtime.configuration import _validate_config
    from scripts.guiding_center.runtime.runner import _build_mesh, _project_case_density

    config, label = campaign_config(args, p=p, hb=level_row["hb"], h=level_row["h"], candidate=candidate)
    _validate_config(config)
    case = case_definition_by_key(config.case).build(**config.case_params)
    started = time.perf_counter()
    mesh = _build_mesh(config, case)
    log(f"mesh: {mesh.num_tri:,} triangles, {mesh.num_edg:,} edges in {time.perf_counter() - started:.2f} s "
        f"(level row: {level_row.get('triangles'):,} triangles)")
    if int(level_row.get("triangles", mesh.num_tri)) != int(mesh.num_tri):
        log("WARNING: the mesh differs from the campaign's level row (triangle count)")
    started = time.perf_counter()
    space = DGSpace(mesh, config.order, basis_type=config.basis, volume_quadrature=config.volume_quadrature,
                    volume_quad_1d=config.volume_quad_1d, edge_quad_1d=config.edge_quad_1d)
    log(f"space: p={config.order}, {space.el_dof * mesh.num_tri:,} element unknowns, "
        f"{(config.order + 1) * len(mesh.int_edges_inds):,} interior trace unknowns "
        f"in {time.perf_counter() - started:.2f} s")
    started = time.perf_counter()
    rho_eq = _project_case_density(config, space, case, equilibrium=True, name="rho_eq_h")
    log(f"equilibrium density projected on the device in {time.perf_counter() - started:.2f} s")
    return config, label, case, mesh, space, rho_eq


def saturated_density(args, *, p: int, level: str, mesh, space) -> dict | None:
    """The saturated (t = 25) density of the stage H smoke run of this p and level, or None.

    The snapshot must have been written on exactly the mesh, degree and basis built here."""
    if level not in args.saturated_levels or not args.saturated_dir:
        return None
    pattern = f"*_H_{args.family}{level}_p{p}_*_seed_snapshots.npz"
    files = sorted(Path(args.saturated_dir).glob(pattern))
    if not files:
        log(f"no stage H snapshot {pattern} in {args.saturated_dir}; no warm_saturated block")
        return None
    path = files[0]
    data = np.load(path)
    problems = []
    if int(data["order"]) != int(p):
        problems.append(f"order {int(data['order'])} != {p}")
    if str(data["basis"]) != str(space.reference.basis_type):
        problems.append(f"basis {data['basis']} != {space.reference.basis_type}")
    if data["triangles"].shape != mesh.triangles.shape or not np.array_equal(data["triangles"], mesh.triangles):
        problems.append("triangle table differs")
    if data["node_coords"].shape != mesh.node_coords.shape or not np.allclose(
            data["node_coords"], mesh.node_coords, rtol=0.0, atol=1e-12):
        problems.append("node coordinates differ")
    density = data["density"]
    if density.shape[1:] != (mesh.num_tri, space.el_dof):
        problems.append(f"density shape {density.shape} != (n, {mesh.num_tri}, {space.el_dof})")
    if problems:
        log(f"stage H snapshot {path.name} does not match the built system ({'; '.join(problems)}); "
            "no warm_saturated block")
        return None
    index = int(np.argmax(data["times"]))
    log(f"saturated density: {path.name}, t = {float(data['times'][index]):g}, "
        f"min {float(density[index].min()):.3f} max {float(density[index].max()):.3f}")
    return dict(path=str(path), time=float(data["times"][index]), coeffs=np.ascontiguousarray(density[index]))


def perturbed_source(config, space, case, index: int, delta: float):
    """Equilibrium density times ``1 + delta cos(m theta + phase)``; a new (m, phase) per solve."""
    from hybridge.runtime.optional import array_module
    from scripts.guiding_center.runtime.runner import _project_initial_field

    equilibrium = case.equilibrium_density
    mode = 2 + (index % 5)
    phase = 0.37 * (index + 1)

    def source(x, y):
        xp = array_module(x, y)
        theta = xp.arctan2(y, x)
        return equilibrium(x, y) * (1.0 + delta * xp.cos(mode * theta + phase))

    return _project_initial_field(config, space, source, name=f"rho_warm_{index}")


# --------------------------------------------------------------------------------------------------
# measurement


class DeviceMemorySampler(threading.Thread):
    """Peak device memory in use (``memGetInfo``: total - free, the whole device) while running."""

    def __init__(self, cp, interval: float = 0.05):
        super().__init__(daemon=True)
        self.cp, self.interval = cp, float(interval)
        self.stop_event = threading.Event()
        self.peak_bytes = 0
        self.samples = 0

    @staticmethod
    def used_bytes(cp) -> int:
        free, total = cp.cuda.runtime.memGetInfo()
        return int(total) - int(free)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.peak_bytes = max(self.peak_bytes, self.used_bytes(self.cp))
                self.samples += 1
            except Exception:
                pass
            self.stop_event.wait(self.interval)

    def finish(self) -> int:
        self.stop_event.set()
        self.join(timeout=10)
        try:
            self.peak_bytes = max(self.peak_bytes, self.used_bytes(self.cp))
        except Exception:
            pass
        return self.peak_bytes


def _solve_record(result, *, wall: float) -> dict:
    """Iterations, residuals and the phase timings of one ``DiffusionReactionHDGSolver.solve``."""
    timings = result.timings
    details = dict(timings.details or {})
    solve = result.global_solve_result
    native = details.get("solve.fb_hp_mg.fallback", 0.0) == 0.0 and "solve.fb_hp_mg.krylov" in details
    krylov = details.get("solve.fb_hp_mg.krylov") if native else details.get("solve.amgx.solve")
    setup = details.get("solve.fb_hp_mg.setup_outer", 0.0) if native else details.get("solve.amgx.setup", 0.0)
    record = dict(
        wall_s=float(wall), total_s=float(timings.total), solve_s=float(timings.solve),
        trace_assembly_s=float(timings.trace_assembly), reconstruction_s=float(timings.reconstruction),
        krylov_s=None if krylov is None else float(krylov), setup_s=float(setup or 0.0),
        iterations=None if solve is None else solve.iteration_count,
        residual_norm=None if solve is None else solve.residual_norm,
        physical_residual_norm=None if solve is None else solve.physical_residual_norm,
        residual_target=None if solve is None else (solve.physical_residual_target or solve.residual_target),
        rhs_norm=None if solve is None else solve.rhs_norm,
        initial_residual_norm=None if solve is None else solve.initial_residual_norm,
        backend=None if solve is None else solve.backend,
        native=bool(native), fallback=bool(details.get("solve.fb_hp_mg.fallback", 0.0)),
        operator_reused=bool(details.get("raw.assembly.operator_reused", 0.0)),
        hierarchy_reused=bool(details.get("solve.fb_hp_mg.hierarchy_reused", 0.0)
                              or details.get("solve.amgx.hierarchy_reused", 0.0)),
    )
    for key in ("solve.fb_hp_mg.setup", "solve.fb_hp_mg.best_iteration", "solve.fb_hp_mg.best_residual",
                "solve.fb_hp_mg.terminal_residual", "solve.fb_hp_mg.residual_restarts",
                "solve.fb_hp_mg.true_residual_checks", "solve.fb_hp_mg.workspace_bytes",
                "solve.fb_hp_mg.symmetry_defect", "solve.fb_hp_mg.positive_curvature",
                "solve.amgx.setup", "solve.amgx.solver_setup", "solve.amgx.matrix_upload", "solve.amgx.solve",
                "solve.amgx.total", "raw.assembly.operator", "raw.assembly.total"):
        if key in details:
            record[key] = float(details[key])
    return record


def _format_solve(label: str, record: dict) -> str:
    its = record.get("iterations")
    residual = record.get("physical_residual_norm") or record.get("residual_norm")
    return (f"{label}: {its} it, krylov {_fmt(record.get('krylov_s'), '.4f')} s, "
            f"setup {record.get('setup_s', 0.0):.3f} s, solve {record['solve_s']:.4f} s, "
            f"total {record['total_s']:.4f} s, true residual {_fmt(residual, '.3e')}"
            f" (target {_fmt(record.get('residual_target'), '.3e')})"
            + (" FALLBACK" if record.get("fallback") else ""))


def _median(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(values) if values else None


def native_diagnostics(solver) -> dict | None:
    """Per-level setup diagnostics and coarse-cycle counters of the native solver, if it was used."""
    native = getattr(solver, "_raw_cuda_fb_hp_mg_solver", None)
    if native is None:
        return None
    preconditioner = native.preconditioner
    coarse = preconditioner.coarse_solver
    levels = [dict(degree=int(item.degree), block_size=int(item.block_size),
                   lambda_max=None if item.lambda_max is None else float(item.lambda_max),
                   lambda_low=None if item.lambda_low is None else float(item.lambda_low),
                   spmv_backend=item.spmv_backend, spmv_fallback_reason=item.spmv_fallback_reason,
                   smoother_backend=item.smoother_backend)
              for item in preconditioner.diagnostics]
    return dict(
        policy=native.preconditioner_policy, schedule=[int(d) for d in preconditioner.degree_schedule],
        chebyshev_order=int(preconditioner.chebyshev_order), presweeps=int(preconditioner.presweeps),
        postsweeps=int(preconditioner.postsweeps), levels=levels,
        setup_seconds=float(native.setup_seconds), workspace_bytes=int(native.workspace_bytes),
        symmetry_defect=float(native.symmetry_defect), positive_curvature=float(native.positive_curvature),
        coarse_apply_count=int(getattr(coarse, "apply_count", 0)),
        coarse_apply_seconds=float(getattr(coarse, "apply_seconds", 0.0)),
        solve_count=int(native.solve_count),
    )


def _release_device_memory(cp) -> None:
    import gc

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def measure_candidate(args, *, p: int, level: str, level_row: dict, candidate: str, system, reference: dict | None,
                      saturated: dict | None = None):
    """All measurements of one candidate on one system; returns the JSONL row."""
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space, field_from_cupy_coefficients
    from hybridge.core.field_ops import solution_trace
    from hybridge.runtime.optional import require_cupy
    from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGSolver
    from scripts.guiding_center.runtime.configuration import _make_poisson_options

    cp = require_cupy()
    config, label, case, mesh, space, rho_eq = system
    config, _ = campaign_config(args, p=p, hb=level_row["hb"], h=level_row["h"], candidate=candidate)
    options = _make_poisson_options(config)
    amgx = options.amgx_config
    stopping = dict(rtol=float(options.solver_rtol), atol=float(options.solver_atol),
                    maxiter=options.maxiter, trace_basis=options.trace_basis,
                    raw_matrix_format=options.raw_matrix_format, cache_local_factors=options.cache_local_factors,
                    scale_system=options.scale_system, stabilization=float(config.poisson_tau),
                    fb_hp_mg_policy=options.fb_hp_mg_preconditioner_policy,
                    amgx_config=None if amgx is None else Path(config.poisson_amgx_config_path or "").name,
                    amgx_convergence=None if amgx is None else amgx["solver"].get("convergence"),
                    amgx_tolerance=None if amgx is None else amgx["solver"].get("tolerance"),
                    amgx_max_iters=None if amgx is None else amgx["solver"].get("max_iters"),
                    retry_policy=config.poisson_retry_policy)
    log(f"candidate {candidate}: solver={options.solver} " + " ".join(f"{k}={v}" for k, v in stopping.items()))
    row = dict(run_id=run_id_of(level, p, candidate, args.run_tag), p=int(p), level=level, candidate=candidate,
               solver=options.solver, tau=float(config.poisson_tau), benchmark=args.benchmark, modes=int(args.modes),
               label=label, triangles=int(mesh.num_tri), edges=int(mesh.num_edg),
               interior_edges=int(len(mesh.int_edges_inds)),
               element_unknowns=int(space.el_dof * mesh.num_tri),
               trace_unknowns=int((p + 1) * len(mesh.int_edges_inds)), hb=level_row["hb"], h=level_row["h"],
               warm_solves=int(args.warm_solves), perturbation=float(args.perturbation), stopping=stopping,
               started=datetime.now().isoformat(timespec="seconds"), status="running")
    _release_device_memory(cp)
    cp.cuda.get_current_stream().synchronize()
    baseline = DeviceMemorySampler.used_bytes(cp)
    row["device_used_before_mib"] = baseline / 2**20
    sampler = DeviceMemorySampler(cp)
    sampler.start()
    solver = None
    try:
        started = time.perf_counter()
        solver = DiffusionReactionHDGSolver(space, source=rho_eq, reaction=space.zeros(name="zero_reaction_h"),
                                            boundary_condition=case.potential_boundary_at(0.0), options=options)
        row["solver_construction_s"] = time.perf_counter() - started

        # 1. cold: assembly + setup + first Krylov solve from zero
        log(f"{candidate}: cold solve (assembly, setup, zero guess)")
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        result = solver.solve()
        cp.cuda.get_current_stream().synchronize()
        cold = _solve_record(result, wall=time.perf_counter() - started)
        row["cold"] = cold
        log(_format_solve(f"{candidate} cold", cold))
        if cold.get("fallback"):
            row["fallback_reason"] = getattr(solver, "_raw_cuda_fb_hp_mg_failure_reason", None)
            log(f"{candidate}: native solver fell back to AMGX: {row['fallback_reason']}")
        history = getattr(result.global_solve_result, "residual_history", None)
        if history:
            row["cold_residual_history"] = [float(v) for v in history]
        row["native"] = native_diagnostics(solver)
        if row["native"] is not None:
            for item in row["native"]["levels"]:
                log(f"  level p={item['degree']} b={item['block_size']} lambda_max={item['lambda_max']} "
                    f"lambda_low={item['lambda_low']} spmv={item['spmv_backend']} smoother={item['smoother_backend']}")
        cold_solution = np.array(result.field.coeffs, dtype=np.float64, copy=True)

        # 2. cold_repeat: zero guess again, operator and hierarchy reused (pure Krylov cost)
        reduced = solution_trace(result, space, reduced=False)
        zero_guess = cp.zeros_like(reduced)
        log(f"{candidate}: cold_repeat solve (zero guess, setup reused)")
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        result = solver.solve(initial_guess=zero_guess)
        cp.cuda.get_current_stream().synchronize()
        cold_repeat = _solve_record(result, wall=time.perf_counter() - started)
        row["cold_repeat"] = cold_repeat
        log(_format_solve(f"{candidate} cold_repeat", cold_repeat))
        del zero_guess

        # 3. warm sequence: perturbed sources, previous solution as the guess
        warm = []
        guess = solution_trace(result, space, reduced=False)
        for index in range(int(args.warm_solves)):
            source = perturbed_source(config, space, case, index, float(args.perturbation))
            solver.set_source(source)
            cp.cuda.get_current_stream().synchronize()
            started = time.perf_counter()
            result = solver.solve(initial_guess=guess)
            cp.cuda.get_current_stream().synchronize()
            record = _solve_record(result, wall=time.perf_counter() - started)
            warm.append(record)
            guess = solution_trace(result, space, reduced=False).copy()
            log(_format_solve(f"{candidate} warm {index + 1}/{args.warm_solves}", record))
        row["warm"] = warm
        row["warm_median"] = dict(
            iterations=_median(r["iterations"] for r in warm), krylov_s=_median(r["krylov_s"] for r in warm),
            solve_s=_median(r["solve_s"] for r in warm), total_s=_median(r["total_s"] for r in warm),
            wall_s=_median(r["wall_s"] for r in warm),
            max_iterations=max((int(r["iterations"] or 0) for r in warm), default=None),
            fallbacks=sum(bool(r.get("fallback")) for r in warm),
            max_physical_residual=max((float(r.get("physical_residual_norm") or r.get("residual_norm") or 0.0)
                                       for r in warm), default=None))
        warm_solution = np.array(result.field.coeffs, dtype=np.float64, copy=True)

        # 3b. saturated-phase sequence: rho_eq + s_k (rho_sat - rho_eq), O(1 %) changes between solves
        saturated_solution = None
        if saturated is not None:
            cspace = as_cupy_space(space)
            eq_c = cp.asarray(as_cupy_coefficients(rho_eq, cspace), dtype=np.float64)
            delta_c = cp.asarray(saturated["coeffs"], dtype=np.float64) - eq_c
            eq_norm = float(cp.linalg.norm(eq_c))
            step_change = float(0.02 * cp.linalg.norm(delta_c)) / (eq_norm or 1.0)
            row["saturated"] = dict(snapshot=saturated["path"], time=saturated["time"], blends=list(SATURATED_BLENDS),
                                    relative_change_per_solve=step_change,
                                    relative_difference=float(cp.linalg.norm(delta_c)) / (eq_norm or 1.0))
            log(f"{candidate}: warm_saturated: |rho_sat - rho_eq| / |rho_eq| = "
                f"{row['saturated']['relative_difference']:.3e}, change per solve {step_change:.3e}")
            sat = []
            for index, blend in enumerate(SATURATED_BLENDS):
                source = field_from_cupy_coefficients(space, eq_c + float(blend) * delta_c, device=cspace.device_id,
                                                      name=f"rho_sat_{index}")
                solver.set_source(source)
                cp.cuda.get_current_stream().synchronize()
                started = time.perf_counter()
                result = solver.solve(initial_guess=guess)
                cp.cuda.get_current_stream().synchronize()
                record = _solve_record(result, wall=time.perf_counter() - started)
                record["blend"] = float(blend)
                sat.append(record)
                guess = solution_trace(result, space, reduced=False).copy()
                log(_format_solve(f"{candidate} warm_saturated {index + 1}/{len(SATURATED_BLENDS)} s={blend:.2f}",
                                  record))
            row["warm_saturated"] = sat
            row["warm_saturated_median"] = dict(
                iterations=_median(r["iterations"] for r in sat), krylov_s=_median(r["krylov_s"] for r in sat),
                solve_s=_median(r["solve_s"] for r in sat), total_s=_median(r["total_s"] for r in sat),
                wall_s=_median(r["wall_s"] for r in sat),
                max_iterations=max((int(r["iterations"] or 0) for r in sat), default=None),
                fallbacks=sum(bool(r.get("fallback")) for r in sat),
                max_physical_residual=max((float(r.get("physical_residual_norm") or r.get("residual_norm") or 0.0)
                                           for r in sat), default=None))
            saturated_solution = np.array(result.field.coeffs, dtype=np.float64, copy=True)
            del eq_c, delta_c, source
        counters = native_diagnostics(solver)
        if row["native"] is not None and counters is not None:
            row["native"].update({k: v for k, v in counters.items()
                                  if k in ("coarse_apply_count", "coarse_apply_seconds", "solve_count")})

        # 4. agreement with the first candidate of this system
        scale = float(np.max(np.abs(cold_solution))) or 1.0
        row["solution_max_abs"] = scale
        if reference is None:
            row["reference_candidate"] = candidate
            row["max_rel_diff_vs_ref"] = 0.0
            row["max_rel_diff_vs_ref_warm"] = 0.0
            if saturated_solution is not None:
                row["max_rel_diff_vs_ref_saturated"] = 0.0
            reference = dict(candidate=candidate, cold=cold_solution, warm=warm_solution, saturated=saturated_solution)
        else:
            row["reference_candidate"] = reference["candidate"]
            ref_scale = float(np.max(np.abs(reference["cold"]))) or 1.0
            row["max_rel_diff_vs_ref"] = float(np.max(np.abs(cold_solution - reference["cold"])) / ref_scale)
            ref_scale = float(np.max(np.abs(reference["warm"]))) or 1.0
            row["max_rel_diff_vs_ref_warm"] = float(np.max(np.abs(warm_solution - reference["warm"])) / ref_scale)
            if saturated_solution is not None and reference.get("saturated") is not None:
                ref_scale = float(np.max(np.abs(reference["saturated"]))) or 1.0
                row["max_rel_diff_vs_ref_saturated"] = float(
                    np.max(np.abs(saturated_solution - reference["saturated"])) / ref_scale)
        log(f"{candidate}: max rel diff vs {row['reference_candidate']}: cold {row['max_rel_diff_vs_ref']:.3e}, "
            f"warm {row['max_rel_diff_vs_ref_warm']:.3e}"
            + (f", saturated {row['max_rel_diff_vs_ref_saturated']:.3e}" if "max_rel_diff_vs_ref_saturated" in row
               else ""))
        row["status"] = "ok"
    except Exception as error:
        row["status"] = "failed"
        row["error"] = f"{type(error).__name__}: {error}"
        row["traceback"] = traceback.format_exc()
        log(f"{candidate} FAILED: {row['error']}")
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        peak = sampler.finish()
        row["peak_device_used_mib"] = peak / 2**20
        row["peak_candidate_mib"] = max(0.0, peak - baseline) / 2**20
        row["memory_samples"] = sampler.samples
        if solver is not None:
            try:
                solver.close()
            except Exception as error:  # the measurements stand
                log(f"{candidate}: close failed: {error}")
        _release_device_memory(cp)
        row["ended"] = datetime.now().isoformat(timespec="seconds")
    log(f"{candidate}: peak device memory {row['peak_device_used_mib']:.0f} MiB "
        f"(+{row['peak_candidate_mib']:.0f} MiB over the baseline {row['device_used_before_mib']:.0f} MiB)")
    return row, reference


# --------------------------------------------------------------------------------------------------
# outputs


def write_summary(rows: list[dict], path: Path) -> None:
    """Markdown table of every row of ``results.jsonl`` (ok and failed)."""
    lines = ["# Poisson p-multigrid degree study: fb-hp-mg-pcg vs AMGX candidates", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')}. Times in ms unless noted; "
             "cold = assembly + solver setup + first zero-guess Krylov solve (setup = native hierarchy or AMGX "
             "hierarchy), cold_repeat = zero-guess solve with the setup reused, warm = median over the perturbed "
             "right-hand sides from the previous solution (linear phase); sat = median over the saturated-phase "
             "sequence rho_eq + s_k (rho_sat - rho_eq), s_k = 0.80..0.98, rho_sat = the stage H smoke density at t = 25 "
             "of the same p and mesh (levels with a snapshot only). Peak = device memory in use (whole device) during the "
             "candidate; +cand = above the level before its solver existed. Diff = max |phi - phi_ref| / max |phi_ref| "
             "against the first candidate's cold solution (warm: last warm solution).", "",
             "| p | level | tau | triangles | trace unk. | candidate | cold it | cold setup | cold Krylov | cold solve | "
             "cold total | cold2 it | cold2 Krylov | warm it (max) | warm Krylov | warm solve | warm total | "
             "sat it (max) | sat Krylov | sat solve | sat total | "
             "peak MiB | +cand MiB | diff cold | diff warm | diff sat | residual cold / warm max / sat max | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
             "---|---|---|"]
    for row in sorted(rows, key=lambda r: (r.get("level", ""), r.get("p", 0), r.get("tau", 0), r.get("candidate", ""))):
        cold, cold2, warm = row.get("cold") or {}, row.get("cold_repeat") or {}, row.get("warm_median") or {}
        sat = row.get("warm_saturated_median") or {}
        status = row.get("status", "?")
        if row.get("fallback_reason"):
            status += " (native fallback)"
        if row.get("run_tag"):
            status += f" [{row['run_tag']}]"
        lines.append(
            f"| {row.get('p')} | {row.get('level')} | {_fmt(row.get('tau'), 'g')} | {_fmt(row.get('triangles'), ',d')} | "
            f"{_fmt(row.get('trace_unknowns'), ',d')} | {row.get('candidate')} | {_fmt(cold.get('iterations'), 'g')} | "
            f"{_fmt(cold.get('setup_s'), '.0f', 1e3)} | {_fmt(cold.get('krylov_s'), '.0f', 1e3)} | "
            f"{_fmt(cold.get('solve_s'), '.0f', 1e3)} | {_fmt(cold.get('total_s'), '.0f', 1e3)} | "
            f"{_fmt(cold2.get('iterations'), 'g')} | {_fmt(cold2.get('krylov_s'), '.0f', 1e3)} | "
            f"{_fmt(warm.get('iterations'), 'g')} ({_fmt(warm.get('max_iterations'), 'g')}) | "
            f"{_fmt(warm.get('krylov_s'), '.1f', 1e3)} | {_fmt(warm.get('solve_s'), '.1f', 1e3)} | "
            f"{_fmt(warm.get('total_s'), '.1f', 1e3)} | "
            f"{_fmt(sat.get('iterations'), 'g')} ({_fmt(sat.get('max_iterations'), 'g')}) | "
            f"{_fmt(sat.get('krylov_s'), '.1f', 1e3)} | {_fmt(sat.get('solve_s'), '.1f', 1e3)} | "
            f"{_fmt(sat.get('total_s'), '.1f', 1e3)} | {_fmt(row.get('peak_device_used_mib'))} | "
            f"{_fmt(row.get('peak_candidate_mib'))} | {_fmt(row.get('max_rel_diff_vs_ref'), '.2e')} | "
            f"{_fmt(row.get('max_rel_diff_vs_ref_warm'), '.2e')} | "
            f"{_fmt(row.get('max_rel_diff_vs_ref_saturated'), '.2e')} | "
            f"{_fmt(cold.get('physical_residual_norm') or cold.get('residual_norm'), '.2e')} / "
            f"{_fmt(warm.get('max_physical_residual'), '.2e')} / {_fmt(sat.get('max_physical_residual'), '.2e')} | "
            f"{status} |")
    native_rows = [row for row in rows if row.get("native")]
    if native_rows:
        lines += ["", "## fb-hp-mg-pcg hierarchy diagnostics", "",
                  "| p | level | tau | policy | schedule | Cheb | levels (p: b, lambda_max, spmv/smoother) | "
                  "setup s | workspace GiB | symmetry defect | coarse applications (s) | restarts cold | status |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in sorted(native_rows, key=lambda r: (r.get("level", ""), r.get("p", 0), r.get("tau", 0))):
            native = row["native"]
            levels = "; ".join(
                f"{item['degree']}: {item['block_size']}, {_fmt(item['lambda_max'], '.3g')}, "
                f"{item['spmv_backend']}/{item['smoother_backend']}" for item in native["levels"])
            cold = row.get("cold") or {}
            lines.append(
                f"| {row['p']} | {row['level']} | {_fmt(row.get('tau'), 'g')} | {native['policy']} | "
                f"{'->'.join(str(d) for d in native['schedule'])} | {native['chebyshev_order']} "
                f"({native['presweeps']}+{native['postsweeps']}) | {levels} | {native['setup_seconds']:.3f} | "
                f"{native['workspace_bytes'] / 2**30:.3f} | {native['symmetry_defect']:.2e} | "
                f"{native['coarse_apply_count']} ({native['coarse_apply_seconds']:.3f}) | "
                f"{_fmt(cold.get('solve.fb_hp_mg.residual_restarts'), 'g')} | {row.get('status')} |")
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        lines += ["", "## Failures", ""]
        for row in failed:
            lines.append(f"- {row.get('run_id')}: {row.get('error')}")
    path.write_text("\n".join(lines) + "\n")


def append_row(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, default=str) + "\n")
        stream.flush()


def all_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    latest = {}
    for row in rows:  # the last record of a run_id wins (restarted rows)
        latest[row["run_id"]] = row
    return list(latest.values())


# --------------------------------------------------------------------------------------------------


def plan(args, levels) -> list[tuple[str, int, dict]]:
    out = []
    for level in args.levels:
        for p in args.degrees:
            row = levels.get((level, p))
            if row is None:
                log(f"no mesh level row for {level} p={p}; skipped")
                continue
            out.append((level, p, row))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degrees", type=_ints, default=(5, 6, 7, 8, 9), help="comma-separated p (default 5..9)")
    parser.add_argument("--levels", type=_names, default=("L2", "L3"), help="mesh levels of the family (default L2,L3)")
    parser.add_argument("--candidates", type=_names, default=DEFAULT_CANDIDATES,
                        help="Poisson candidates of fast_profiles.POISSON_CANDIDATES, in run order; the first one "
                             "is the reference of the solution comparison")
    parser.add_argument("--warm-solves", type=int, default=10)
    parser.add_argument("--perturbation", type=float, default=1.0e-3, help="delta of the warm right-hand sides")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-levels", type=Path, default=DEFAULT_MESH_LEVELS)
    parser.add_argument("--family", default="m128", help="mesh family of the levels file")
    parser.add_argument("--benchmark", default="disk-highmode")
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--tau", type=float, default=128000.0, help="Poisson stabilization (campaign: 128000)")
    parser.add_argument("--saturated-dir", type=Path, default=DEFAULT_SATURATED_DIR,
                        help="directory of the stage H *_seed_snapshots.npz files (saturated densities); '' disables")
    parser.add_argument("--saturated-levels", type=_names, default=("L2",),
                        help="levels that get the warm_saturated block (default L2; L3 has no snapshot)")
    parser.add_argument("--run-tag", default="", help="suffix of the run ids (e.g. tau1 for the sanity check)")
    parser.add_argument("--redo", action="store_true", help="measure rows that results.jsonl already holds")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no device work")
    parser.add_argument("-v", "--verbosity", type=int, choices=(0, 1, 2, 3), default=3,
                        help="0 quiet, 1 per-solve lines, 2 solver phase logs, 3 detailed timings and the "
                             "native/AMGX iteration tables (solver verbose=3); default 3")
    args = parser.parse_args(argv)
    args.output_dir = Path(args.output_dir)
    if str(args.saturated_dir) in {"", "."} or str(args.saturated_dir).lower() == "none":
        args.saturated_dir = None
    levels = read_mesh_levels(args.mesh_levels, args.family)
    systems = plan(args, levels)
    if args.dry_run:
        for level, p, row in systems:
            for candidate in args.candidates:
                print(f"{run_id_of(level, p, candidate, args.run_tag)}: {row['triangles']:,} triangles, hb={row['hb']}, "
                      f"h={row['h']}, tau={args.tau:g}, warm solves {args.warm_solves}")
        return 0

    sys.path.insert(0, str(ROOT))
    from scripts.guiding_center.runtime.terminal_log import _TerminalLogTee

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.md"
    tag = f"_{args.run_tag}" if args.run_tag else ""
    log_path = args.output_dir / ("_".join(f"{level}_p{p}" for level, p, _ in systems) + f"{tag}.log")
    done = {} if args.redo else finished_rows(results_path)
    failures = 0
    with _TerminalLogTee(log_path):
        log(f"benchmark_poisson_pmg_degrees: {' '.join(sys.argv[1:])}")
        log(f"log {log_path}; results {results_path}")
        for level, p, level_row in systems:
            pending = [c for c in args.candidates if run_id_of(level, p, c, args.run_tag) not in done]
            if not pending:
                log(f"{level} p={p}: every candidate measured; skipped")
                continue
            log(f"=== {level} p={p}: {level_row['triangles']:,} triangles, hb={level_row['hb']}, h={level_row['h']}, "
                f"tau={args.tau:g}; candidates {pending}")
            try:
                system = build_system(args, p=p, level_row=level_row, candidate=pending[0])
            except Exception as error:
                failures += 1
                log(f"{level} p={p} FAILED to build the system: {type(error).__name__}: {error}")
                traceback.print_exc()
                for candidate in pending:
                    append_row(results_path, dict(run_id=run_id_of(level, p, candidate, args.run_tag), p=p, level=level,
                                                  candidate=candidate, tau=float(args.tau), run_tag=args.run_tag,
                                                  status="failed", error=f"system: {type(error).__name__}: {error}"))
                write_summary(all_rows(results_path), summary_path)
                continue
            reference = None
            saturated = saturated_density(args, p=p, level=level, mesh=system[3], space=system[4])
            for candidate in pending:
                log(f"--- {level} p={p} {candidate}")
                row, reference = measure_candidate(args, p=p, level=level, level_row=level_row, candidate=candidate,
                                                   system=system, reference=reference, saturated=saturated)
                row["run_tag"] = args.run_tag
                append_row(results_path, row)
                write_summary(all_rows(results_path), summary_path)
                failures += row["status"] != "ok"
            del system, reference, saturated
        log(f"done; {failures} failed candidate measurements" if failures else "done; every measurement ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
