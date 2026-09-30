#!/usr/bin/env python3
"""Manufactured-solution studies of the decoupled n-Gamma D-BDF2 scheme (TODO L98 F1).

Runs the plan's studies (docs/development/plans/n_gamma_d_bdf2.md) for one or
more cases in an explicit geometry (``--geometry`` has no default):

* stationary cases: spatial refinement (baseline ``h``; stress ``h`` with the
  paired outer/hole polygonizations) at constant ``dt``, then a temporal
  contamination check that halves ``dt`` on the finest mesh;
* transient cases: temporal refinement on a fixed mesh, then a spatial
  contamination check that reruns the finest ``dt`` on ``h/2``; if either
  field's error changes by more than 10%% the fixed mesh is refined (at most
  twice more) and the ``dt`` sequence is repeated;
* ``--study startup``: the Euler startup alone, as one-step errors and full
  runs started from a single exact level, kept separate from exact-history runs.

Main runs start from independently projected exact fields at ``t=0`` and
``t=dt``, so the first computed endpoint is ``2 dt``. Every run uses a
constant step; a rejected step (failed solve or nonfinite state) ends that
run, which is recorded and flagged, and the next run starts fresh. Errors use
the geometry's norm (plain L2 or R-weighted) with a ``p+7`` Duffy error rule.

Named presets (``--list-presets``, ``--print-preset``, ``--preset KEY``) set
the defaults of every option, and options on the command line override them;
see ``scripts/n_gamma/presets.py``. Host runs use all available Numba threads
by default and a PARDISO thread count matched to the reduced system size;
OpenBLAS is limited to one thread (set before NumPy loads), because its
spinning workers otherwise compete with the Numba and MKL pools. The actual
thread counts and the CPU/wall ratio of every run are recorded.

``--plot-every N`` shows exact, numerical and error panels of ``n`` and
``Gamma`` every ``N`` steps of every run (``--plot-backend auto`` uses Holoviz
on the device path and PyVista on the host path; headless PyVista saves
frames, Holoviz saves frames or a movie on request). ``-v/--verbosity``
selects 0 (quiet), 1 (runs, progress and tables), 2 (every step: solve
iterations, reused caches, clamps, timings) or 3 (also the ADR solver's own
stage output).

This script runs time integration: execute it only when authorized.
``--dry-run`` prints the planned runs without meshing or solving. Example::

    .venv/bin/python -m scripts.n_gamma.run_d_bdf2 --preset mms_xy_p6_device --dry-run
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import sys
import time

# Must precede the first NumPy import to take effect: idle OpenBLAS workers spin and, measured on
# 2026-09-29 (24 cores, p=6), doubled host step times by competing with the Numba and MKL pools.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdgfem.runtime.threads import host_threads, set_host_threads
from scripts.n_gamma.cases import CASE_NAMES, get_case  # noqa: E402
from scripts.n_gamma.cases.forcing import C_S, D, MU  # noqa: E402
from scripts.n_gamma.cases.geometry import BASELINE_SIZES, STRESS_POLYGONIZATIONS  # noqa: E402
from scripts.n_gamma.diagnostics import StudyRecorder, convergence_rows, error_norms, markdown_table  # noqa: E402
from scripts.n_gamma.presets import BLOCK_AMG_BSR, L1_BSR, preset_by_key, print_presets  # noqa: E402

# Measured 2026-09-29 on the p=4 finest stress meshes (both geometries, dt=0.02 and 0.0025): PBICGSTAB +
# L1 Jacobi on face BSR took ~0.08-0.11 s per warm solve against ~0.51 s for FGMRES + block AMG, which
# stays the per-solve fallback (see scripts/n_gamma/README.md).
DEFAULT_AMGX_CONFIG = L1_BSR
DEFAULT_AMGX_FALLBACK = BLOCK_AMG_BSR
# PARDISO factor+solve scaled little beyond 8 threads at 10k trace unknowns and beyond 16 at 38k
# (2026-09-29, p=6); the rule below picks 8 up to 15,000 unknowns and 16 above, capped by the CPU count.
PARDISO_SMALL_SYSTEM = 15_000
CONTAMINATION_TOLERANCE = 0.10
EXTRA_REFINEMENTS = 2


@dataclass(frozen=True)
class RunSpec:
    """One constant-step run of one case."""

    case: str
    geometry: str
    label: str
    order: int
    h: float
    outer_vertices: int
    hole_vertices: int
    dt: float
    final_time: float
    startup: str = "exact"      # "exact" two-level history or single-level "euler"
    max_steps: int | None = None

    @property
    def start_time(self) -> float:
        """Time of the last initial level: ``dt`` (exact history) or ``0`` (Euler startup)."""
        return self.dt if self.startup == "exact" else 0.

    @property
    def steps(self) -> int:
        """Constant-step count to the final time (capped by ``max_steps``)."""
        count = (self.final_time - self.start_time)/self.dt
        if abs(count - round(count)) > 1e-9 or round(count) < 1:
            raise ValueError(f"final time {self.final_time} is not a positive multiple of dt={self.dt} "
                             f"after t={self.start_time}")
        count = int(round(count))
        return count if self.max_steps is None else min(count, self.max_steps)

    @property
    def end_time(self) -> float:
        """Time reached after all steps."""
        return self.start_time + self.steps*self.dt


def _refined(spec: RunSpec, label: str) -> RunSpec:
    """Return ``spec`` on the next mesh level: half ``h`` and, for the stress domain, doubled polygonization."""
    factor = 2 if spec.hole_vertices else 1
    return replace(spec, label=label, h=spec.h/2, outer_vertices=factor*spec.outer_vertices,
                   hole_vertices=factor*spec.hole_vertices)


def plan_runs(case_name: str, args) -> dict[str, list[RunSpec]]:
    """Return the planned runs of a case, grouped by study, without meshing anything."""
    case = get_case(case_name, geometry=args.geometry)
    stress = case.domain == "H"
    base = dict(case=case_name, geometry=args.geometry, order=args.order, final_time=args.final_time,
                max_steps=args.max_steps)
    polygonizations = list(zip(args.outer_vertices, args.hole_vertices)) if stress else [(4, 0)]*len(args.mesh_sizes)
    if stress and len(polygonizations) != len(args.mesh_sizes):
        raise ValueError("stress meshes need one outer/hole polygonization per mesh size")
    if case.stationary and args.study != "startup":
        spatial = [RunSpec(label=f"space_h{h:g}", h=h, outer_vertices=o, hole_vertices=v, dt=args.dt, **base)
                   for h, (o, v) in zip(args.mesh_sizes, polygonizations)]
        check = replace(spatial[-1], label=f"{spatial[-1].label}_dt_half", dt=args.dt/2)
        return {"spatial": spatial, "temporal_check": [check]}
    h, (o, v) = args.fixed_mesh_size, (args.fixed_outer_vertices, args.fixed_hole_vertices) if stress else (4, 0)
    startup = "euler" if args.study == "startup" else "exact"
    temporal = [RunSpec(label=f"time_dt{dt:g}", h=h, outer_vertices=o, hole_vertices=v, dt=dt, startup=startup,
                        **base) for dt in args.timesteps]
    if args.study == "startup":
        one_step = [replace(spec, label=f"one_step_dt{spec.dt:g}", max_steps=1) for spec in temporal]
        return {"startup_one_step": one_step, "startup_full": temporal}
    if not getattr(args, "spatial_check", True):
        return {"temporal": temporal}
    return {"temporal": temporal, "spatial_check": [_refined(temporal[-1], f"{temporal[-1].label}_h_half")]}


def _amgx_config(name):
    """Load an AMGX JSON config (repository-relative or absolute); ``none`` keeps the built-in DILU."""
    if name is None or str(name).lower() == "none":
        return None
    from hdgfem.io.config import load_amgx_config
    path = Path(name)
    config, _ = load_amgx_config(path if path.is_absolute() else ROOT/path)
    return config


def _fallback_options(args) -> dict | None:
    """Per-solve fallback for a failed raw-CUDA AMGX solve (``None`` disables it)."""
    if args.backend != "raw-cuda":
        return None
    config = _amgx_config(args.amgx_fallback_config)
    return None if config is None else dict(amgx_config=config)


def available_cpus() -> int:
    """CPUs visible to this process (affinity-aware)."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def pardiso_threads_for(setting, unknowns: int) -> int:
    """Resolve ``--pardiso-threads`` (``auto``, ``all`` or a count) for a reduced system size."""
    available = available_cpus()
    if setting == "all":
        return available
    if setting == "auto":
        return min(available, 8 if unknowns <= PARDISO_SMALL_SYSTEM else 16)
    return max(1, min(int(setting), available))


def configure_numba_threads(setting) -> int:
    """Set and verify the Numba thread count (``all`` or a count); return the active count."""
    import numba
    requested = available_cpus() if setting == "all" else max(1, int(setting))
    numba.set_num_threads(min(requested, numba.config.NUMBA_NUM_THREADS))
    return int(numba.get_num_threads())


def log(args, level: int, message: str) -> None:
    """Print ``message`` when the verbosity is at least ``level``."""
    if getattr(args, "verbosity", 1) >= level:
        print(message, flush=True)


def _cache_flags(summary) -> str:
    """Compact reused-cache marks of one solve: S static, A PARDISO analysis, P AMGX setup, R reconstruction."""
    return "".join(mark for mark, used in (("S", summary.static_reused), ("A", summary.analysis_reused),
                                           ("P", summary.setup_reused), ("R", summary.reconstruction_reused)) if used) or "-"


def _step_line(spec, diagnostics) -> str:
    """One verbosity-2 line per accepted step."""
    density, momentum = diagnostics.density_solve, diagnostics.momentum_solve
    return (f"  [{spec.label}] step {diagnostics.step:5d} t={diagnostics.time:.5f} {diagnostics.stage:5s} "
            f"iters n/G {density.iterations}/{momentum.iterations} reused {_cache_flags(density)}/"
            f"{_cache_flags(momentum)} fallback {int(density.fallback)}/{int(momentum.fallback)} "
            f"clamps {diagnostics.floor.volume_clamps + diagnostics.floor.face_clamps} "
            f"min n {diagnostics.min_density_sampled:.4f} {diagnostics.total_seconds*1e3:.1f} ms")


def _make_plotter(spec, args, space, case):
    """Exact/numerical/error panels for one run, or ``None`` when plotting is off."""
    if args.plot_every <= 0:
        return None
    from scripts.n_gamma.plotting import NGammaPanels, output_settings, resolve_backend
    backend = resolve_backend(args.plot_backend, "raw-cuda" if args.backend == "raw-cuda" else "numba")
    run_dir = Path(args.output_dir)/spec.geometry/spec.case
    off_screen, frames = output_settings(backend, args.plot_every, off_screen=args.plot_off_screen,
                                         screenshot_dir=None if args.plot_dir is None else
                                         Path(args.plot_dir)/spec.case/spec.label,
                                         default_dir=run_dir/"frames"/spec.label)
    movie = str(run_dir/f"{spec.label}.mp4") if args.plot_movie and backend == "holoviz" else None
    # PyVista repeats the title in every (narrow) panel, so it gets only the run label.
    title = spec.label if backend == "pyvista" else f"{spec.case} ({spec.geometry}) {spec.label}"
    return NGammaPanels(backend, space, case, title=title,
                        off_screen=off_screen, screenshot_dir=frames, screenshot_prefix=f"{spec.case}_{spec.label}",
                        time_step=spec.dt, total_steps=spec.steps, show_mesh=args.plot_show_mesh,
                        resolution=args.plot_resolution, width=args.plot_width, height=args.plot_height,
                        max_fps=args.plot_max_fps, movie_path=movie)


def _backend_options(args, pardiso_threads=None) -> dict:
    """ADR solver options shared by both solves of every step."""
    options = dict(trace_basis=args.trace_basis, hdg_postprocess="none",
                   verbose=1 if getattr(args, "verbosity", 1) >= 3 else False)
    if args.backend == "raw-cuda":
        options.update(assembly_backend="raw-cuda", solver="amgx", solver_rtol=args.solver_rtol,
                       materialize_host_solution=False, raw_matrix_format=args.raw_matrix_format,
                       amgx_config=_amgx_config(args.amgx_config), amgx_reuse=args.amgx_reuse,
                       amgx_refresh_interval=args.amgx_refresh_interval,
                       amgx_refresh_iteration_growth=args.amgx_refresh_growth)
    else:
        pardiso = str(args.host_solver).lower() == "pypardiso"
        options.update(assembly_backend="numba", solver=args.host_solver, solver_rtol=args.solver_rtol,
                       numba_reuse_local_columns=args.numba_reuse_local_columns,
                       pardiso_reuse_analysis=bool(args.pardiso_reuse_analysis and pardiso),
                       pardiso_threads=pardiso_threads if pardiso else None)
    return options


def _project(space, function, projection_space):
    """L2-project with the higher-order projection rule, returned as a field of ``space``."""
    return space.field(projection_space.project_callable(function).coeffs)


def _compiled_coefficients(args, space, case):
    """Numba-compiled coefficients for the host path (``--compiled-coefficients``), else ``None``."""
    if args.backend != "numba" or not args.compiled_coefficients:
        return None
    from scripts.n_gamma.compiled import CompiledCoefficients
    return CompiledCoefficients(space, case, density_floor=args.density_floor, sound_speed=C_S)


def run_single(spec: RunSpec, args, recorder: StudyRecorder | None) -> dict:
    """Mesh, initialize, step and measure one run; never raises on a rejected step."""
    from hdgfem import DGSpace
    from scripts.n_gamma.stepper import NGammaBDF2Stepper, StepRejected

    started = time.perf_counter()
    case = get_case(spec.case, geometry=spec.geometry)
    mesh_options = dict(outer_vertices=spec.outer_vertices, hole_vertices=spec.hole_vertices) if case.domain == "H" else {}
    record = case.build_mesh(spec.h, cache_dir=args.mesh_cache_dir, **mesh_options)
    quadrature = dict(volume_degree=args.volume_degree) if args.volume_quad_1d is None else dict(
        volume_quad_1d=args.volume_quad_1d)
    space = DGSpace(record.mesh, spec.order, basis_type=args.basis, **quadrature)
    unknowns = int(record.mesh.num_edg) * (spec.order + 1)  # upper bound of the reduced trace size
    pardiso_threads = pardiso_threads_for(args.pardiso_threads, unknowns) if args.backend == "numba" else None
    projection_space = DGSpace(record.mesh, spec.order, basis_type=args.basis,
                               volume_quad_1d=spec.order + args.error_quad_offset)
    exact = lambda field, t: (lambda a, b: field(a, b, t))
    levels = [0.] if spec.startup == "euler" else [0., spec.dt]
    fields = [(_project(space, exact(case.density, t), projection_space),
               _project(space, exact(case.momentum, t), projection_space)) for t in levels]
    history = {} if len(fields) == 1 else dict(previous_density=fields[0][0], previous_momentum=fields[0][1])
    stepper = NGammaBDF2Stepper(
        space, *fields[-1], dt=spec.dt, time=levels[-1], geometry=spec.geometry, b_poloidal=case.b_poloidal,
        diffusion=D, viscosity=MU, sound_speed=C_S, density_floor=args.density_floor,
        source_density=case.density_source_at, source_momentum=case.momentum_source_at,
        boundary_density=case.density_boundary_at, boundary_momentum=case.momentum_boundary_at,
        options=_backend_options(args, pardiso_threads), fallback_options=_fallback_options(args),
        compiled=_compiled_coefficients(args, space, case), **history)
    rejected = None
    min_density = math.inf
    clamps = 0
    result = None
    log(args, 1, f"[{spec.case}/{spec.label}] {record.mesh.num_tri} triangles (h={record.metadata['actual_h']:.4g}), "
                 f"p={spec.order}, dt={spec.dt:g}, {spec.steps} steps from t={stepper.time:g}"
                 + (f", PARDISO {pardiso_threads} threads" if pardiso_threads else ""))
    plotter = _make_plotter(spec, args, space, case)
    plot_seconds = 0.
    progress = max(1, spec.steps // 10)
    interrupted = False
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    try:
        if plotter is not None:
            plot_start = time.perf_counter()
            plotter.update(*stepper.current, step=0, time_value=stepper.time)
            plot_seconds += time.perf_counter() - plot_start
        for index in range(spec.steps):
            final = index == spec.steps - 1
            try:
                result = stepper.advance(postprocess=args.final_postprocess if final else None)
            except StepRejected as error:
                rejected = str(error)
                log(args, 1, f"[{spec.case}/{spec.label}] step {index + 1} rejected: {error}")
                if recorder is not None and error.diagnostics is not None:
                    recorder.record_step(error.diagnostics, run=spec.label, dt=spec.dt, h=spec.h)
                break
            diagnostics = result.diagnostics
            min_density = min(min_density, diagnostics.min_density_sampled)
            clamps += diagnostics.floor.volume_clamps + diagnostics.floor.face_clamps
            if recorder is not None:
                recorder.record_step(diagnostics, run=spec.label, dt=spec.dt, h=spec.h)
            if args.verbosity >= 2:
                log(args, 2, _step_line(spec, diagnostics))
            elif (index + 1) % progress == 0 or final:
                log(args, 1, f"  [{spec.label}] step {index + 1}/{spec.steps} t={diagnostics.time:.5f} "
                             f"({diagnostics.total_seconds*1e3:.0f} ms/step)")
            if plotter is not None and ((index + 1) % args.plot_every == 0 or final):
                plot_start = time.perf_counter()
                plotter.update(*stepper.current, step=index + 1, time_value=stepper.time)
                plot_seconds += time.perf_counter() - plot_start
    except BaseException:
        interrupted = True
        raise
    finally:
        stepper.close()
        if plotter is not None:
            plotter.close(suppress_errors=interrupted)
    stepping_wall = time.perf_counter() - wall_start
    stepping_cpu = time.process_time() - cpu_start
    density, momentum = stepper.current
    errors = {}
    if rejected is None:
        exact_fields = {"n": exact(case.density, stepper.time), "Gamma": exact(case.momentum, stepper.time)}
        rule = dict(geometry=spec.geometry, volume_quad_1d=spec.order + args.error_quad_offset)
        errors = error_norms({"n": density, "Gamma": momentum}, exact_fields, **rule)
        post = {name: getattr(getattr(result, solve), "postprocessed_field", None) if result is not None else None
                for name, solve in (("n_post", "density"), ("Gamma_post", "momentum"))}
        if all(field is not None for field in post.values()):
            # Post-processed fields are degree p+1; the error rule gains one point accordingly.
            errors.update(error_norms(post, {"n_post": exact_fields["n"], "Gamma_post": exact_fields["Gamma"]},
                                      geometry=spec.geometry, volume_quad_1d=spec.order + 1 + args.error_quad_offset))
    if errors:
        log(args, 1, f"[{spec.case}/{spec.label}] t={stepper.time:.5f} "
                     + ", ".join(f"{key} {value:.3e}" for key, value in errors.items())
                     + f"; {(stepping_wall - plot_seconds)/max(stepper.step_count, 1)*1e3:.1f} ms/step")
    return dict(spec=asdict(spec), mesh=record.metadata, steps_taken=stepper.step_count, time=stepper.time,
                errors=errors, rejected=rejected, min_density_sampled=min_density if min_density < math.inf else None,
                floor_clamps=clamps, seconds=time.perf_counter() - started,
                threads=dict(pardiso=pardiso_threads, numba=_numba_threads(), host_numpy=host_threads(),
                             available=available_cpus(), openblas=os.environ.get("OPENBLAS_NUM_THREADS")),
                stepping=dict(wall_seconds=stepping_wall, cpu_seconds=stepping_cpu,
                              cpu_wall_ratio=stepping_cpu/max(stepping_wall, 1e-12),
                              plot_seconds=plot_seconds,
                              seconds_per_step=(stepping_wall - plot_seconds)/max(stepper.step_count, 1)))


def _numba_threads():
    """Active Numba thread count (``None`` without Numba)."""
    try:
        import numba
        return int(numba.get_num_threads())
    except Exception:
        return None


def _error_keys(geometry: str) -> tuple[str, str]:
    """Error-record keys of the geometry's norm."""
    suffix = "l2R" if geometry == "axisymmetric" else "l2"
    return f"n_{suffix}", f"Gamma_{suffix}"


def _table(results, parameter):
    """Convergence rows; spatial orders use the measured largest edge ``actual_h``, not the Gmsh target.

    Post-processed errors (``n_post_*``, ``Gamma_post_*``) get their own columns when present.
    """
    keys = _error_keys(results[0]["spec"]["geometry"])
    post = tuple(key.replace("_", "_post_", 1) for key in keys)
    if any(post[0] in r["errors"] for r in results):
        keys = keys + post
    values = [r["mesh"]["actual_h"] if parameter == "actual_h" else r["spec"][parameter] for r in results]
    rows = convergence_rows(parameter, values, {key: [r["errors"].get(key) for r in results] for key in keys})
    for row, result in zip(rows, results):
        row.update(h=result["spec"]["h"], dt=result["spec"]["dt"], actual_h=result["mesh"]["actual_h"],
                   elements=result["mesh"]["elements"],
                   steps=result["steps_taken"], rejected=bool(result["rejected"]),
                   min_n_sampled=result["min_density_sampled"], clamps=result["floor_clamps"],
                   s_per_step=result["stepping"]["seconds_per_step"],
                   cpu_wall=result["stepping"]["cpu_wall_ratio"])
    return rows


def _relative_change(reference, check, keys):
    """Relative error change between a run and its contamination check (``None`` if unavailable)."""
    changes = {}
    for key in keys:
        a, b = reference["errors"].get(key), check["errors"].get(key)
        changes[key] = None if not a or b is None else abs(b - a)/a
    return changes


def run_case(case_name: str, args) -> dict:
    """Run every planned study of one case and write its records."""
    runs = plan_runs(case_name, args)
    directory = Path(args.output_dir)/args.geometry/case_name
    recorder = StudyRecorder(directory, f"{case_name}_{args.study}")
    log(args, 1, f"== {args.geometry} {case_name}: {args.study} study, "
                 + ", ".join(f"{study} {len(specs)} run(s)" for study, specs in runs.items()) + f" -> {directory}")
    keys = _error_keys(args.geometry)
    summary = dict(case=case_name, geometry=args.geometry, study=args.study, order=args.order,
                   backend=args.backend, basis=args.basis, trace_basis=args.trace_basis,
                   amgx_config=args.amgx_config if args.backend == "raw-cuda" else None,
                   amgx_fallback_config=args.amgx_fallback_config if args.backend == "raw-cuda" else None,
                   raw_matrix_format=args.raw_matrix_format if args.backend == "raw-cuda" else None,
                   amgx_reuse=args.amgx_reuse if args.backend == "raw-cuda" else None,
                   solver_rtol=args.solver_rtol,
                   volume_degree=args.volume_degree, volume_quad_1d=args.volume_quad_1d,
                   error_quad_1d=args.order + args.error_quad_offset, density_floor=args.density_floor,
                   preset=args.preset, final_postprocess=args.final_postprocess,
                   host_caches=None if args.backend != "numba" else dict(
                       compiled_coefficients=args.compiled_coefficients,
                       pardiso_reuse_analysis=args.pardiso_reuse_analysis,
                       numba_reuse_local_columns=args.numba_reuse_local_columns),
                   threads=dict(numba=_numba_threads(), host_numpy=host_threads(),
                                pardiso_setting=args.pardiso_threads, available=available_cpus(),
                                openblas=os.environ.get("OPENBLAS_NUM_THREADS")))
    tables, results = {}, {}
    if "spatial" in runs:
        results["spatial"] = [run_single(spec, args, recorder) for spec in runs["spatial"]]
        results["temporal_check"] = [run_single(spec, args, recorder) for spec in runs["temporal_check"]]
        tables["spatial"] = _table(results["spatial"], "actual_h")
        summary["temporal_contamination"] = _relative_change(results["spatial"][-1], results["temporal_check"][0], keys)
    elif "temporal" in runs:
        temporal, checks = runs["temporal"], runs.get("spatial_check")
        if not checks:      # --no-spatial-check
            results["temporal"] = [run_single(spec, args, recorder) for spec in temporal]
            summary["spatially_resolved"] = None
        else:
            check_spec, resolved = checks[0], False
            for attempt in range(EXTRA_REFINEMENTS + 1):
                results["temporal"] = [run_single(spec, args, recorder) for spec in temporal]
                check = run_single(check_spec, args, recorder)
                change = _relative_change(results["temporal"][-1], check, keys)
                summary.setdefault("spatial_contamination_attempts", []).append(
                    dict(h=temporal[0].h, check_h=check_spec.h, relative_change=change))
                if all(value is not None and value <= CONTAMINATION_TOLERANCE for value in change.values()):
                    resolved = True
                    break
                if attempt < EXTRA_REFINEMENTS:
                    temporal = [_refined(spec, f"{spec.label}_ref{attempt + 1}") for spec in temporal]
                    check_spec = _refined(check_spec, f"{check_spec.label}_ref{attempt + 1}")
            summary["spatially_resolved"] = resolved
        tables["temporal"] = _table(results["temporal"], "dt")
    else:
        results["startup_one_step"] = [run_single(spec, args, recorder) for spec in runs["startup_one_step"]]
        results["startup_full"] = [run_single(spec, args, recorder) for spec in runs["startup_full"]]
        tables["startup_one_step"] = _table(results["startup_one_step"], "dt")
        tables["startup_full"] = _table(results["startup_full"], "dt")
    summary["rejected_runs"] = [r["spec"]["label"] for group in results.values() for r in group if r["rejected"]]
    recorder.write_summary(summary, tables)
    (directory/f"{case_name}_{args.study}_runs.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    return dict(summary=summary, tables=tables)


def build_parser():
    """Command-line options; ``--geometry`` must be given here or by the preset."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", default=None, help="named defaults (see --list-presets); options override it")
    parser.add_argument("--list-presets", action="store_true", help="print the presets and exit")
    parser.add_argument("--print-preset", action="store_true", help="print the resolved options and exit")
    parser.add_argument("--geometry", default=None, choices=("cartesian", "axisymmetric"))
    parser.add_argument("--case", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES))
    parser.add_argument("--study", choices=("main", "startup"), default="main",
                        help="main exact-history studies, or the separate Euler-startup study")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--final-time", type=float, default=1.)
    parser.add_argument("--dt", type=float, default=.005, help="stationary-study timestep")
    parser.add_argument("--timesteps", type=float, nargs="+", default=[.02, .01, .005, .0025])
    parser.add_argument("--mesh-sizes", type=float, nargs="+", default=list(BASELINE_SIZES))
    parser.add_argument("--outer-vertices", type=int, nargs="+", default=[o for o, _ in STRESS_POLYGONIZATIONS])
    parser.add_argument("--hole-vertices", type=int, nargs="+", default=[v for _, v in STRESS_POLYGONIZATIONS])
    parser.add_argument("--fixed-mesh-size", type=float, default=.05, help="transient-study mesh size")
    parser.add_argument("--fixed-outer-vertices", type=int, default=STRESS_POLYGONIZATIONS[-1][0])
    parser.add_argument("--fixed-hole-vertices", type=int, default=STRESS_POLYGONIZATIONS[-1][1])
    parser.add_argument("--max-steps", type=int, default=None, help="truncate every run (smoke tests only)")
    parser.add_argument("--spatial-check", action=argparse.BooleanOptionalAction, default=True,
                        help="run the transient study's h/2 spatial-contamination check (4x the elements, "
                             "refined up to twice more if it fails); --no-spatial-check skips it for large meshes")
    parser.add_argument("--backend", choices=("raw-cuda", "numba"), default="raw-cuda")
    parser.add_argument("--host-solver", default="pypardiso")
    parser.add_argument("--pardiso-threads", default="auto",
                        help="PARDISO threads: 'auto' (8 up to 15k unknowns, else 16), 'all', or a count")
    parser.add_argument("--numba-threads", default="all", help="Numba threads: 'all' or a count")
    parser.add_argument("--pardiso-reuse-analysis", action=argparse.BooleanOptionalAction, default=True,
                        help="reuse the PARDISO reordering/analysis across steps (host)")
    parser.add_argument("--compiled-coefficients", action=argparse.BooleanOptionalAction, default=True,
                        help="host: Numba-compiled (cfunc) coefficients and forcing instead of NumPy evaluation")
    parser.add_argument("--numba-reuse-local-columns", action=argparse.BooleanOptionalAction, default=True,
                        help="reconstruct from the assembly's local columns (host)")
    parser.add_argument("--final-postprocess", choices=("none", "primal", "flux", "both"), default="none",
                        help="HDG post-processing on the final step of every run")
    parser.add_argument("--amgx-config", default=DEFAULT_AMGX_CONFIG,
                        help="raw-CUDA AMGX JSON (default: PBICGSTAB + L1 Jacobi, face BSR); 'none' = built-in DILU")
    parser.add_argument("--amgx-fallback-config", default=DEFAULT_AMGX_FALLBACK,
                        help="config retried once after a failed solve (default: FGMRES + block AMG); 'none' disables")
    parser.add_argument("--raw-matrix-format", choices=("csr", "bsr"), default="bsr")
    parser.add_argument("--amgx-reuse", choices=("none", "solver", "preconditioner"), default="preconditioner",
                        help="keep AMGX objects across steps ('solver') and also the previous setup "
                             "('preconditioner', refreshed periodically and after failures)")
    parser.add_argument("--amgx-refresh-interval", type=int, default=20,
                        help="fresh AMGX setup at least every this many solves (preconditioner reuse)")
    parser.add_argument("--amgx-refresh-growth", type=float, default=2.0,
                        help="fresh setup when iterations exceed this factor times the post-refresh count")
    parser.add_argument("--solver-rtol", type=float, default=1e-11)
    parser.add_argument("--basis", default="dub_orth")
    parser.add_argument("--trace-basis", default="legacy-lagrange", choices=("legacy-lagrange", "legendre-modal"))
    parser.add_argument("--volume-degree", type=int, default=14,
                        help="volume rule exactness (14 = 42-point Dunavant); ignored with --volume-quad-1d")
    parser.add_argument("--volume-quad-1d", type=int, default=None, help="Duffy volume rule instead, e.g. p+5")
    parser.add_argument("--error-quad-offset", type=int, default=7, help="error/projection Duffy points = p+offset")
    parser.add_argument("--density-floor", type=float, default=1e-8)
    parser.add_argument("--output-dir", default="run_outputs/n_gamma")
    parser.add_argument("--mesh-cache-dir", default=None, help="Gmsh mesh cache (default: the package cache)")
    parser.add_argument("--plot-every", type=int, default=0, help="plot every N steps of every run (0: off)")
    parser.add_argument("--plot-backend", choices=("auto", "pyvista", "holoviz"), default="auto",
                        help="auto: Holoviz on the device path, PyVista on the host path")
    parser.add_argument("--plot-off-screen", action="store_true", help="render without a window")
    parser.add_argument("--plot-dir", default=None, help="save frames under DIR/<case>/<run>")
    parser.add_argument("--plot-movie", action="store_true", help="Holoviz: write <output>/<geometry>/<case>/<run>.mp4")
    parser.add_argument("--plot-show-mesh", action="store_true")
    parser.add_argument("--plot-resolution", type=int, default=6, help="PyVista sub-samples per element edge")
    parser.add_argument("--plot-width", type=int, default=1500)
    parser.add_argument("--plot-height", type=int, default=900)
    parser.add_argument("--plot-max-fps", type=float, default=10., help="Holoviz live-preview frame cap")
    parser.add_argument("-v", "--verbosity", type=int, choices=(0, 1, 2, 3), default=1,
                        help="0 quiet, 1 runs/progress/tables, 2 every step, 3 plus ADR solver stages")
    parser.add_argument("--quiet", action="store_true", help="same as --verbosity 0")
    parser.add_argument("--dry-run", action="store_true", help="print the planned runs and exit")
    return parser


def parse_arguments(argv=None):
    """Parse options; a ``--preset`` supplies defaults that explicit options override."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.preset is not None:
        try:
            preset = preset_by_key(args.preset)
        except ValueError as error:
            parser.error(str(error))
        parser.set_defaults(**dict(preset.options))
        args = parser.parse_args(argv)
    if not args.list_presets and args.geometry is None:
        parser.error("--geometry is required (cartesian or axisymmetric) unless a preset sets it")
    if args.quiet:
        args.verbosity = 0
    if args.plot_every < 0:
        parser.error("--plot-every must be nonnegative")
    return args


def main(argv=None) -> int:
    """Print planned runs (``--dry-run``) or run the studies; Ctrl-C exits with status 130."""
    args = parse_arguments(argv)
    try:
        return _main(args)
    except KeyboardInterrupt:
        print("\n[n-gamma] interrupted; completed runs are recorded in their output directories", flush=True)
        return 130


def _main(args) -> int:
    """Body of :func:`main` after argument parsing."""
    if args.list_presets:
        print_presets()
        return 0
    if args.print_preset:
        for key, value in sorted(vars(args).items()):
            print(f"{key:28s} {value}")
        return 0
    if args.dry_run:
        for case_name in args.case:
            for study, specs in plan_runs(case_name, args).items():
                rows = [dict(study=study, label=s.label, h=s.h, vertices=f"{s.outer_vertices}/{s.hole_vertices}",
                             dt=s.dt, steps=s.steps, startup=s.startup) for s in specs]
                print(f"\n{args.geometry} {case_name} {study}\n" + markdown_table(rows))
        return 0
    if args.backend == "numba":
        # Element-chunked NumPy (coefficient sampling, dense face tables) shares the Numba thread count.
        set_host_threads(configure_numba_threads(args.numba_threads))
    for case_name in args.case:
        outcome = run_case(case_name, args)
        for name, rows in outcome["tables"].items():
            log(args, 1, f"\n{args.geometry} {case_name} {name}\n" + markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
