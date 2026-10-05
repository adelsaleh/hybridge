"""User-facing fixed-width disk/ellipse/mesh workflow; see the equiband guide."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import traceback


def parser():
    result = argparse.ArgumentParser(description="Fixed threshold-width equilibrium bands; distance uses zeta_T=s/L.",
                                     allow_abbrev=False)
    result.add_argument("--config", required=True, help="version-2 TOML or JSON configuration")
    result.add_argument("--output", required=True, help="run directory; if it exists, warn and ask whether to start fresh or resume")
    result.add_argument("--threshold-width-delta", type=float,
                        help="override the fixed threshold-space width from the configuration")
    result.add_argument("--target-distance", type=float,
                        help="override the normalized torsion-flow target distance (strictly between 0 and 1)")
    result.add_argument(
        "--mesh-file",
        help="load this exact external .msh and leave canonical generated/cache mode",
    )
    result.add_argument("--mesh-size", type=float,
                        help="override the generated-mesh cache key; for explicit .msh import this is metadata only")
    result.add_argument("--geometry-degree", type=int,
                        help="override generated coordinate-map degree (1, 2 or 3)")
    result.add_argument("--mesh-cache-directory",
                        help="override the generated .msh cache directory")
    result.add_argument("--rebuild-mesh-cache", action="store_true",
                        help="regenerate the selected canonical cache key atomically")
    result.add_argument("--allow-mesh-size-mismatch", action="store_true",
                        help="deliberately import a canonical .msh whose sidecar size differs from mesh_size")
    result.add_argument("--torsion-degree", type=int, help="override the torsion finite-element degree")
    result.add_argument("--recovered-gradient-degree", type=int,
                        help="override the continuous recovered-gradient degree")
    result.add_argument("--quadrature-degree", type=int,
                        help="override the nonlinear-form quadrature degree")
    result.add_argument("--number-of-rays", type=int, help="override the number of boundary-arclength ray labels")
    result.add_argument("--samples-per-ray", type=int, help="override the maximum integration chord density")
    result.add_argument("--ray-tolerance", type=float, help="override the dimensionless local ray-integration tolerance")
    result.add_argument("--threads", type=int, help="override Numba worker threads per MPI rank")
    result.add_argument("--maximum-iterations", type=int,
                        help="override the SNES and reduced-correction iteration limit")
    reuse = result.add_mutually_exclusive_group()
    reuse.add_argument("--restart", action="store_true", help="resume committed states with identical configuration and MPI partition; skip the prompt")
    reuse.add_argument("--overwrite-output", action="store_true", help="archive an existing directory to a unique sibling backup, then run fresh without prompting")
    result.add_argument("--save-terminal-log", action=argparse.BooleanOptionalAction, default=False,
                        help="tee Python and native stdout/stderr into OUTPUT/logs/SESSION/terminal_rankNNNN.log")
    result.add_argument("--seed", choices=["radial", "homotopy"], default="radial")
    result.add_argument("--m-start", type=float, help="seed midpoint; radial default comes from an analytical annulus guess")
    result.add_argument("--m-stop", type=float, help="required endpoint for midpoint scans; not an arclength target-search bound")
    result.add_argument("--continuation", choices=("auto", "midpoint", "pseudo-arclength"), default="auto",
                        help="auto: target-oriented arclength search, or midpoint mode with --scan-only")
    result.add_argument("--arc-direction", choices=("target", "increasing-m", "decreasing-m"), default="target",
                        help="initial orientation only; m may turn around at a fold (default: toward target)")
    result.add_argument("--arc-step", type=float, default=.01, help="initial dimensionless H1/midpoint arc step")
    result.add_argument("--arc-min-step", type=float, default=1e-7)
    result.add_argument("--arc-max-step", type=float, default=.025)
    result.add_argument("--arc-max-steps", type=int, default=200)
    result.add_argument("--arc-max-length", type=float, default=2., help="maximum dimensionless arc length per invocation")
    result.add_argument("--scan-only", action="store_true", help="map the chart without solving the configured distance target")
    result.add_argument("--atlas-only", action="store_true",
                        help="validate and write the mesh/torsion/ray atlas, then stop before any equilibrium solve")
    result.add_argument("--stability", action="store_true", help="label accepted states with the lowest energy-Hessian eigenvalue (SLEPc)")
    result.add_argument("--write-vtk", action="store_true", help="write final target/nearest fields for ParaView or PyVista")
    result.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=2,
                        help="0: results/errors; 1: accepted states; 2: full solver/branch diagnostics (default)")
    result.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                        help="interactive PyVista T/phi/rho panels (default); --no-plot for batch runs")
    result.add_argument("--plot-mode", choices=("blocking", "nonblocking"), default="nonblocking",
                        help="live updates (default), or Enter-to-continue after every displayed state")
    result.add_argument("--plot-final", action=argparse.BooleanOptionalAction, default=True,
                        help="pause for inspection of the final target/nearest state (default)")
    result.add_argument("--plot-off-screen", action="store_true", help="render saved frames without opening a window")
    result.add_argument("--save-frames", action="store_true", help="save PNGs under OUTPUT/frames, including on restart")
    result.add_argument("--plot-every", type=int, default=1, help="display every N accepted states; final states always displayed")
    result.add_argument("--plot-min-interval", type=float, default=.5,
                        help="minimum seconds between live nonblocking updates; 0 disables throttling (frames/final updates unaffected)")
    result.add_argument("--plot-refinement", type=int, default=4, help="plot subintervals per affine-cell edge, 1..32 (display only)")
    result.add_argument("--plot-mesh-edges", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--plot-window-width", type=int, default=1800)
    result.add_argument("--plot-window-height", type=int, default=700)
    return result


def apply_config_overrides(config, args):
    """Return one fully revalidated immutable configuration from CLI flags."""
    from dataclasses import replace
    if args.mesh_file is not None:
        # An explicit file override intentionally leaves generated/cache mode.
        # Clear canonical parameters so the resulting msh configuration is
        # unambiguous and independently validated against any sidecar.
        config = replace(config, geometry="msh", mesh_file=args.mesh_file,
                         geometry_parameters=None)
    if args.threshold_width_delta is not None:
        # BandConfig.__post_init__ recomputes epsilon from the fixed relative
        # ratio when that smoothing mode is selected.
        config = replace(config, band=replace(
            config.band, threshold_width_delta=args.threshold_width_delta))
    overrides = {
        "target_distance": args.target_distance,
        "mesh_file": args.mesh_file,
        "mesh_size": args.mesh_size,
        "geometry_degree": args.geometry_degree,
        "mesh_cache_directory": args.mesh_cache_directory,
        "torsion_degree": args.torsion_degree,
        "recovered_gradient_degree": args.recovered_gradient_degree,
        "quadrature_degree": args.quadrature_degree,
        "number_of_rays": args.number_of_rays,
        "samples_per_ray": args.samples_per_ray,
        "ray_tolerance": args.ray_tolerance,
        "threads": args.threads,
        "maximum_iterations": args.maximum_iterations,
    }
    return replace(config, **{key: value for key, value in overrides.items()
                              if value is not None})


def imported_mesh_provenance_messages(config):
    """Return deterministic log records for an imported mesh and its sidecar.

    ``mesh_size`` cannot refine a pre-existing ``.msh`` file.  The canonical
    geometry generator records its requested size in ``FILE.msh.json``; when
    available, compare that immutable provenance with the runtime metadata and
    report rather than silently suggesting that a smaller mesh was used.  The
    CLI rejects a detected mismatch by default; a deliberate import requires
    ``--allow-mesh-size-mismatch``.  The sidecar is optional so third-party
    tagged meshes remain supported.
    """
    if config.geometry != "msh":
        return []
    from projects.diocotron.paths import resolve_archive_path
    mesh_path = resolve_archive_path(config.mesh_file)
    sidecar = Path(str(mesh_path) + ".json")
    if not sidecar.is_file():
        return [(1, f"MESH_PROVENANCE_UNAVAILABLE mesh_file={mesh_path} "
                    f"sidecar={sidecar} config_mesh_size={config.mesh_size:.10g}")]
    try:
        provenance = json.loads(sidecar.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [(0, f"MESH_PROVENANCE_INVALID mesh_file={mesh_path} sidecar={sidecar} "
                    f"error={type(error).__name__}")]
    requested = provenance.get("requested_size")
    messages = [(2, f"MESH_PROVENANCE mesh_file={mesh_path} sidecar={sidecar} "
                    f"requested_size={requested!r} cells={provenance.get('cells')!r} "
                    f"mesh_sha256={provenance.get('mesh_sha256')!r}")]
    try:
        requested_value = float(requested)
    except (TypeError, ValueError):
        messages.append((0, f"MESH_PROVENANCE_INVALID requested_size={requested!r} "
                            f"sidecar={sidecar}"))
        return messages
    if (not math.isfinite(requested_value)
            or not math.isclose(requested_value, config.mesh_size,
                                rel_tol=1e-12, abs_tol=1e-15)):
        messages.append((0, f"MESH_SIZE_MISMATCH config_mesh_size={config.mesh_size:.10g} "
                            f"sidecar_requested_size={requested_value:.10g} "
                            f"mesh_file={mesh_path} imported_mesh_unchanged=1 "
                            "action=regenerate_or_select_matching_mesh"))
    return messages


def main(argv=None, *, _process_entry=False):
    """Start transcript capture before importing NumPy, MPI or solver modules."""
    from .terminal_logging import run_logged
    return run_logged(_main, list(sys.argv[1:] if argv is None else argv), process_entry=_process_entry)


def _main(argv, session):
    arguments = parser()
    args = arguments.parse_args(argv)
    if (args.m_stop is not None and not math.isfinite(args.m_stop)) or (args.m_start is not None and not math.isfinite(args.m_start)):
        arguments.error("--m-start and --m-stop must be finite")
    mode = ("midpoint" if args.scan_only else "pseudo-arclength") if args.continuation == "auto" else args.continuation
    if not args.atlas_only and mode == "midpoint" and args.m_stop is None:
        arguments.error("midpoint continuation requires --m-stop")
    if (not args.atlas_only and mode == "pseudo-arclength" and args.scan_only
            and args.arc_direction == "target"):
        arguments.error("arclength --scan-only requires --arc-direction increasing-m or decreasing-m")
    if args.plot_every < 1 or not 1 <= args.plot_refinement <= 32:
        arguments.error("--plot-every must be positive and --plot-refinement must be between 1 and 32")
    if not math.isfinite(args.plot_min_interval) or args.plot_min_interval < 0:
        arguments.error("--plot-min-interval must be finite and nonnegative")
    if min(args.plot_window_width, args.plot_window_height) < 200:
        arguments.error("plot window dimensions must be at least 200 pixels")
    if args.plot_off_screen and not args.save_frames:
        arguments.error("--plot-off-screen requires --save-frames; use --no-plot to disable rendering")
    from .config import SolverConfig
    session.phase = "configuration"
    try:
        config = SolverConfig.load(args.config)
        config = apply_config_overrides(config, args)
        if args.rebuild_mesh_cache and config.geometry == "msh":
            raise ValueError("--rebuild-mesh-cache applies to generated geometries, not explicit .msh import")
        provenance_messages = imported_mesh_provenance_messages(config)
        mismatch = next((message for _, message in provenance_messages
                         if message.startswith("MESH_SIZE_MISMATCH ")), None)
        if mismatch is not None and not args.allow_mesh_size_mismatch:
            raise ValueError(
                mismatch + "; refusing an unintended resolution mismatch before FE setup. "
                "Select/generate the matching mesh, or pass --allow-mesh-size-mismatch deliberately."
            )
        from .pseudo_arclength import ArcControls
        arc_controls = ArcControls(args.arc_step, args.arc_min_step, args.arc_max_step,
                                   args.arc_max_steps, args.arc_max_length)
    except (ValueError, TypeError, OSError) as error:
        arguments.error(str(error))
    session.phase = "MPI initialization"
    try:
        from mpi4py import MPI
    except ImportError:
        print("equiband requires the FEniCSx/MPI environment; see projects/diocotron/docs/equiband.md")
        return 2
    from .reporting import ProgressReporter
    from .run_diagnostics import (RunTelemetry, compact_json,
                                  mpi_stack_preflight, target_certificate,
                                  timing_log_lines)
    from .run_directory import OutputCancelled, prepare_run_directory
    started = session.started
    comm = MPI.COMM_WORLD
    session.set_mpi_identity(comm)
    report = ProgressReporter(comm, args.verbosity, started=started)
    telemetry = RunTelemetry(comm, started)
    telemetry_finalized = False
    plotter = solver = None
    try:
        session.phase = "output selection"
        phase_started = time.perf_counter()
        directory = prepare_run_directory(args.output, comm, restart=args.restart, overwrite=args.overwrite_output,
                                          report=lambda message: report(message, level=0))
        telemetry.record("output_selection", time.perf_counter()-phase_started)
        args.output, args.restart = str(directory.path), directory.restart
        session.record.update(config=asdict(config), config_hash=config.signature, options=vars(args).copy())
        session.bind(directory, comm)
        report(f"RUN verbosity={args.verbosity} plot={int(args.plot)} plot_mode={args.plot_mode} "
               f"off_screen={int(args.plot_off_screen)} save_frames={int(args.save_frames)} "
               f"terminal_log={int(args.save_terminal_log)} ranks={comm.size}", level=0)
        report(f"RUN_START utc={session.record['started_utc']} output={directory.path}", level=0)
        report("COMMAND " + session.record["command"], level=2)
        report(f"CONFIG source={args.config} hash={config.signature}\n" + json.dumps(asdict(config), indent=2), level=2)
        for level, message in provenance_messages:
            report(message, level=level)
        report(f"PHYSICS threshold_width_delta={config.band.threshold_width_delta:.10g} "
               f"epsilon={config.band.epsilon:.10g} epsilon_over_delta={config.band.epsilon_over_delta:.10g} "
               f"target_distance={config.target_distance:.10g}", level=0)
        from .nonlinearities import Window
        report(f"SMOOTHING mode={config.band.smoothing_mode} relative_epsilon={config.band.relative_epsilon} "
               f"resolved_epsilon={config.band.epsilon:.10g} window_peak={Window(config.band).peak:.10g} "
               "epsilon_fixed_during_midpoint_search=1", level=0)
        report("DISTANCE zeta_T=s/L; D_T=sum(fixed boundary-arclength weights * zeta_middle); "
               "middle level phi=m; spatial thickness is diagnostic only", level=0)
        continuation_label = "not_applicable" if args.atlas_only else mode
        report(f"CONTROLS seed={args.seed} m_start={args.m_start} m_stop={args.m_stop} "
               f"continuation={continuation_label} scan_only={int(args.scan_only)} "
               f"atlas_only={int(args.atlas_only)} restart={int(args.restart)} "
               f"PDE_tolerance={config.pde_tolerance:g} distance_tolerance={config.distance_tolerance:g}")
        report("THREADS " + json.dumps(session.record["threads"], sort_keys=True), level=2)
        if mode == "pseudo-arclength" and not args.atlas_only:
            report("ARC_CONTROLS " + json.dumps(asdict(arc_controls), sort_keys=True), level=0)
            if args.m_stop is not None:
                report("ARC_PARAMETERIZATION m is free to turn at folds; --m-stop is not used in this mode. "
                       "Use --continuation midpoint for an explicit m-endpoint scan.", level=0)
        session.phase = "solver imports"
        phase_started = time.perf_counter()
        import numpy as np
        from .equilibrium import EquilibriumSolver, SolveFailure
        from .continuation import BranchController, MidpointTargetSolver
        from .output import RunStore, json_safe, software_versions
        # Reassert signal handlers after native imports while retaining the
        # same transcript; PETSc can install handlers during initialization.
        if session.capture is not None:
            session.capture.retarget(session.capture.path, rank=comm.rank)
        versions = software_versions()
        telemetry.record("solver_imports", time.perf_counter()-phase_started)
        report("SOFTWARE " + json.dumps(versions, sort_keys=True), level=2)
        # Scanning the Gmsh shared library for its native MPI guard is a
        # filesystem provenance check; rank zero performs it once.
        stack_preflight = comm.bcast(
            mpi_stack_preflight(versions["mpi_library"])
            if comm.rank == 0 else None,
            root=0,
        )
        session.record["mpi_stack_preflight"] = stack_preflight
        report("MPI_STACK_PREFLIGHT " + compact_json(stack_preflight),
               level=0 if stack_preflight["status"] != "OK" else 2)
        session.phase = "plot preflight"
        phase_started = time.perf_counter()
        if (args.plot or args.save_frames) and not args.atlas_only:
            from .plotting import EquibandPlotter, preflight_plotting
            preflight_plotting(args, comm)
        telemetry.record("plot_preflight", time.perf_counter()-phase_started)
        session.phase = "mesh, torsion and rays"
        phase_started = time.perf_counter()
        solver = EquilibriumSolver(config, comm, report=report,
                                   rebuild_mesh_cache=args.rebuild_mesh_cache)
        telemetry.record("mesh_torsion_ray_atlas", time.perf_counter()-phase_started)
        if session.capture is not None:
            session.capture.retarget(session.capture.path, rank=comm.rank)
        report(f"torsion maximum={solver.potential_scale:.9g}; {config.number_of_rays} valid rays; "
               f"delta={config.band.threshold_width_delta:g}; epsilon/delta={config.band.epsilon_over_delta:g}")
        session.phase = "checkpoint initialization"
        phase_started = time.perf_counter()
        store = RunStore(args.output, solver, restart=args.restart, prepared=directory)
        telemetry.record("checkpoint_initialization", time.perf_counter()-phase_started)
        report("PETSC_OPTIONS " + json.dumps(solver.petsc_options, sort_keys=True), level=2)
        if args.atlas_only:
            session.phase = "atlas summary commit"
            atlas = solver.atlas
            store.write_summary([], search={
                "atlas_only": True,
                "number_of_rays": config.number_of_rays,
                "resolved_torsion_fraction": atlas.resolved_torsion_fraction,
                "flow_resolution": atlas.flow_resolution,
                "unresolved_neighbor_pairs": atlas.unresolved_neighbor_pairs,
            })
            session.outcome = "ATLAS_VALIDATED"
            session.record.update(
                committed_checkpoints=store.committed_count,
                resolved_torsion_fraction=atlas.resolved_torsion_fraction,
                flow_resolution=atlas.flow_resolution,
                unresolved_neighbor_pairs=atlas.unresolved_neighbor_pairs)
            report(f"RESULT status=ATLAS_VALIDATED rays={config.number_of_rays} "
                   f"resolved_for_T_over_Tmax_below={atlas.resolved_torsion_fraction:.6f} "
                   f"elapsed={time.perf_counter()-started:.3f}s", level=0)
            timing = telemetry.summarize(output_directory=store.path)
            telemetry_finalized = True
            session.record["timing"] = timing
            for line in timing_log_lines(timing):
                report(line, level=0)
            session.phase = "complete"
            return 0
        if args.plot or args.save_frames:
            plotter = EquibandPlotter(solver, args, store.path, report)
            report.pump = plotter.pump
        def accepted(point, stage="ACCEPTED"):
            if args.stability:
                point.state = solver.classify_stability(point.state)
            store.write_point(point)
            session.record.update(committed_checkpoints=store.committed_count,
                                  last_accepted_m=point.state.m, last_accepted_distance=point.metrics.distance,
                                  last_accepted_state=point.state.state_id)
            report(f"CHECKPOINT committed={store.committed_count} state_id={point.state.state_id} "
                   f"chart={point.segment_id} stage={stage}", level=2)
            report(f"accepted m={point.state.m:.10g} D={point.metrics.distance:.10g} "
                   f"PDE={point.state.residual_norm:.2e} {point.state.stability}")
            report(f"BAND_METRICS source_mass={point.state.source_mass:.8g} energy={point.state.energy:.8g} "
                   f"distance_error={point.metrics.distance_error:.6e} "
                   f"transversality={point.metrics.min_transversality:.6e} "
                   f"inner_threshold_margin={point.metrics.inner_threshold_margin:.6e} "
                   f"flow_core_torsion_margin={point.metrics.flow_core_torsion_margin:.6e} "
                   f"spatial_thickness_diagnostic={point.metrics.mean_physical_thickness:.6e}", level=2)
            if plotter is not None:
                plotter.emit(point, stage=stage)
        controller = BranchController(solver, on_accept=accepted, report=report)
        session.phase = "restart audit" if args.restart else "branch seed"
        phase_started = time.perf_counter()
        previous = store.load_points() if args.restart else []
        if previous:
            # A parallel direct solve need only reproduce the rebuilt atlas to
            # numerical tolerance, not coefficient-byte identity.  Refresh
            # every stored geometric observable against that new atlas before
            # it participates in branch selection or target bracketing.  A
            # checkpoint that no longer passes the current guards is rejected
            # explicitly rather than silently retaining stale distances.
            distance_drifts = []
            for point in previous:
                refreshed = solver.evaluate(point.state)
                if not refreshed.admissible:
                    raise SolveFailure(
                        "RESTART_STATE_FAILED_AUDIT",
                        f"state={point.state.state_id}, reason={refreshed.reason}")
                distance_drifts.append(abs(refreshed.distance-point.metrics.distance))
                point.metrics = refreshed
            report(f"RESTART_GEOMETRY_AUDIT states={len(previous)} "
                   f"maximum_distance_drift={max(distance_drifts, default=0.):.6e} "
                   "atlas_rebuilt=1", level=0)
            # A target search can append states behind the scan endpoint.
            # Resume from the closest stored state on the most recent chart,
            # not blindly from the last scalar-root trial in the ledger.
            chart = previous[-1].segment_id
            candidates = [p for p in previous if p.segment_id == chart]
            if mode == "midpoint":
                if any(p.parameterization == "arclength" for p in candidates):
                    raise ValueError("RESTART_METHOD_MISMATCH: do not reinterpret an arclength branch as a single-valued m chart")
                seed_point = min(candidates, key=lambda p: abs(p.state.m-args.m_stop))
            elif args.scan_only:
                seed_point = max(candidates, key=lambda p: p.arc_length)
            else:
                # A target-oriented restart chooses its closest stored state,
                # not an irrelevant m endpoint on the opposite side of a fold.
                seed_point = min(candidates, key=lambda p: abs(p.metrics.distance-config.target_distance))
            # An independent check catches field/metadata corruption or an
            # incompatible external solver change before any warm start.
            solver.restore(seed_point.state)
            if solver.residual_norm() > config.pde_tolerance or not solver.evaluate(seed_point.state).admissible:
                raise SolveFailure("RESTART_STATE_FAILED_AUDIT")
            report(f"RESTART points={len(previous)} m={seed_point.state.m:.10g} audit=OK")
            if plotter is not None:
                plotter.emit(seed_point, stage="RESTART", force=True)
        else:
            if args.restart:
                raise SolveFailure("NO_COMMITTED_RESTART_STATE")
            if args.seed == "radial":
                if config.geometry != "disk":
                    raise ValueError("radial seed requires a disk; use --seed homotopy for other domains")
                from dolfinx import fem
                from .radial import solve_radial
                radial = solve_radial(config.band, config.radius, m=args.m_start)
                guess = fem.Function(solver.V)
                guess.interpolate(lambda x: radial.values(np.hypot(x[0], x[1])))
                state = solver.solve(radial.m, initial_values=guess.x.array[:solver.owned])
                report(f"independent radial reference: D={radial.distance:.10g}")
            else:
                if args.m_start is None:
                    raise ValueError("homotopy initialization requires --m-start")
                state = solver.homotopy_seed(args.m_start)
            seed_point = controller.seed(state)
            accepted(seed_point, stage="SEED")
        telemetry.record("branch_initialization", time.perf_counter()-phase_started)
        results = []
        report(f"SCAN_START mode={mode} seed_m={seed_point.state.m:.10g} seed_D={seed_point.metrics.distance:.10g} "
               f"m_stop={args.m_stop} target={config.target_distance:.10g}", level=0)
        phase_started = time.perf_counter()
        if mode == "midpoint":
            direction = "increasing" if args.m_stop > seed_point.state.m else (
                "decreasing" if args.m_stop < seed_point.state.m else "stationary")
            report(f"MIDPOINT_DIRECTION {direction}", level=0)
            session.phase = "midpoint scan"
            scan = controller.scan(seed_point, [args.m_stop])
            points = previous+scan.points[1:] if previous else scan.points
            events = list(scan.events)
            reached_stop = abs(scan.points[-1].state.m-args.m_stop) < 32*np.finfo(float).eps*solver.potential_scale
            scan_reason = scan.events[-1]["reason"] if scan.events and not reached_stop else "none"
            if not args.scan_only:
                session.phase = "target correction"
                target_solver = MidpointTargetSolver(controller)
                results = target_solver.solve(config.target_distance, points)
                events += target_solver.events
        else:
            from .pseudo_arclength import PseudoArclengthController
            arc = PseudoArclengthController(solver, arc_controls, on_accept=accepted, report=report)
            parent = next((p for p in previous if p.state.state_id == seed_point.state.parent_id), None)
            session.phase = "pseudo-arclength continuation"
            scan = arc.trace(seed_point, target=None if args.scan_only else config.target_distance,
                             direction=-1 if args.arc_direction == "decreasing-m" else 1, parent=parent,
                             orient_to_target=args.arc_direction == "target")
            points = previous+scan.points[1:] if previous else scan.points
            events, scan_reason = list(scan.events), scan.stop_reason
            reached_stop = scan_reason in {"ARC_LENGTH_LIMIT", "ARC_STEP_LIMIT"} if args.scan_only else scan_reason == "TARGET_REACHED"
            if not args.scan_only:
                results = [arc.target_result(scan, config.target_distance)]
            session.record.update(arclength_folds_crossed=scan.folds, arc_controls=asdict(arc_controls))
        telemetry.record("continuation_and_target", time.perf_counter()-phase_started)
        session.record["continuation_mode"] = mode
        # A restart combines the previously committed chart with only the
        # points/events produced by this invocation.  Keep those scopes
        # explicit in both the terminal transcript and machine-readable
        # summary; otherwise a full-chart distance range beside a per-run arc
        # span is easy to misread as one internally inconsistent statistic.
        chart_distances = [p.metrics.distance for p in points]
        invocation_distances = [p.metrics.distance for p in scan.points]
        invocation_point_count = len(scan.points)
        accepted_new_points = invocation_point_count - (1 if previous else 0)
        chart_point_count = len(points)
        arc_span = (max(p.arc_length for p in scan.points)-min(p.arc_length for p in scan.points)
                    if mode == "pseudo-arclength" else None)
        chart_arc_span = (max(p.arc_length for p in points)-min(p.arc_length for p in points)
                          if mode == "pseudo-arclength" else None)
        rejected_trials = sum(not event.get("accepted", False) for event in events)
        rejected_trial_seconds = sum(float(event.get("elapsed_seconds", 0.)) for event in events
                                     if not event.get("accepted", False))
        report(f"SCAN_END mode={mode} endpoint_reached={reached_stop} last_m={scan.points[-1].state.m:.10g} "
               f"stop_reason={scan_reason} chart_D_min={min(chart_distances):.10g} "
               f"chart_D_max={max(chart_distances):.10g} "
               f"invocation_D_min={min(invocation_distances):.10g} "
               f"invocation_D_max={max(invocation_distances):.10g} "
               f"chart_points_total={chart_point_count} invocation_points={invocation_point_count} "
               f"accepted_new_points={accepted_new_points} "
               f"rejected_trials_this_invocation={rejected_trials} "
               f"rejected_trial_seconds={rejected_trial_seconds:.3f}" +
               (f" arc_span_this_invocation={arc_span:.6e} "
                f"chart_arc_span_total={chart_arc_span:.6e} folds_this_invocation={scan.folds}"
                if arc_span is not None else ""), level=0)
        if mode == "pseudo-arclength" and scan_reason == "ARC_STEP_LIMIT":
            report(f"ARC_BUDGET_EXHAUSTED maximum_steps={arc_controls.maximum_steps} "
                   f"accepted_predictor_steps={max(0, invocation_point_count-1)} "
                   f"accepted_new_points={accepted_new_points} chart_points_total={chart_point_count} "
                   f"rejected_trials_this_invocation={rejected_trials} "
                   f"rejected_trial_seconds={rejected_trial_seconds:.3f} "
                   f"arc_span_this_invocation={arc_span:.6e} "
                   f"chart_arc_span_total={chart_arc_span:.6e} "
                   f"maximum_arc_length_per_invocation={arc_controls.maximum_length:.6e} "
                   f"length_budget_fraction={arc_span/arc_controls.maximum_length:.6e}; "
                   "increase --arc-max-steps to extend this same chart", level=0)
        visualization_states = []
        certificates = []
        if not args.scan_only:
            for result in results:
                if result.point:
                    # Acceptance callbacks have already written substeps; this
                    # also handles an existing endpoint that is the nearest.
                    store.write_point(result.point)
                    if args.write_vtk:
                        visualization_states.append(result.point.state)
                    certificate = target_certificate(
                        solver, result.point, config.target_distance,
                        result_status=result.status)
                    certificates.append(certificate)
                    report("TARGET_CERTIFICATE " + compact_json(certificate), level=0)
                report("TARGET_RESULT " + json.dumps(json_safe({"status": result.status,
                                                  "distance_error": result.distance_error,
                                                  "m": result.point.state.m if result.point else None,
                                                  "distance": result.point.metrics.distance if result.point else None}),
                                                     sort_keys=True), level=0)
        session.phase = "summary commit"
        phase_started = time.perf_counter()
        store.write_summary(results, events, search={
            "continuation_mode": mode, "scan_only": args.scan_only, "stop_reason": scan_reason,
            "initial_state": seed_point.state.state_id, "initial_m": seed_point.state.m,
            "target_distance": config.target_distance, "m_stop": args.m_stop,
            "arc_direction": args.arc_direction if mode == "pseudo-arclength" else None,
            "arc_controls": asdict(arc_controls) if mode == "pseudo-arclength" else None,
            "folds_crossed": scan.folds,
            "chart_points_total": chart_point_count,
            "invocation_points": invocation_point_count,
            "accepted_new_points": accepted_new_points,
            "rejected_trials_this_invocation": rejected_trials,
            "rejected_trial_seconds_this_invocation": rejected_trial_seconds,
            "observed_arc_span_this_invocation": arc_span,
            "observed_arc_span_total": chart_arc_span,
            # Retain the first reporting revision's keys for readers of
            # already-produced schema-version-2 summaries.
            "accepted_scan_points": len(scan.points), "rejected_trials": rejected_trials,
            "observed_arc_span": arc_span,
        }, certificates=certificates)
        telemetry.record("summary_commit", time.perf_counter()-phase_started)
        # VTK/PVD output is optional.  Commit the authoritative numerical
        # summary first, then report a visualization failure without changing
        # a successful solve into a numerical failure or losing restart data.
        phase_started = time.perf_counter()
        for state in visualization_states:
            try:
                store.write_visualization(state)
            except (ValueError, RuntimeError, OSError) as error:
                warning = (f"state={state.state_id}, {type(error).__name__}: {error}")
                session.record.setdefault("optional_output_warnings", []).append(warning)
                report(f"OPTIONAL_OUTPUT_WARNING kind=VTK {warning}", level=0)
        telemetry.record("optional_visualization_output", time.perf_counter()-phase_started)
        session.outcome = ("SCAN_COMPLETE" if reached_stop else "SCAN_STOPPED") if args.scan_only else (
            "TARGET_REACHED" if any(result.exact_target_reached for result in results) else "TARGET_NOT_ATTAINED")
        session.record.update(committed_checkpoints=store.committed_count, scan_endpoint_reached=bool(reached_stop),
                              scan_stop_reason=scan_reason,
                              rejection_counts=dict(Counter(event["reason"] for event in events if not event.get("accepted", False))),
                              rejected_trial_seconds=rejected_trial_seconds,
                              targets=json_safe([{"status": r.status, "distance_error": r.distance_error,
                                                  "explored_m_interval": r.explored_m_interval,
                                                  "explored_branch_ids": r.explored_branch_ids} for r in results]))
        report(f"RESULT status={session.outcome} elapsed={time.perf_counter()-started:.3f}s "
               f"checkpoints={store.committed_count} scan_endpoint_reached={reached_stop}", level=0)
        if events:
            report("REJECTION_SUMMARY " + json.dumps(session.record["rejection_counts"], sort_keys=True), level=0)
        if not reached_stop:
            report(f"CHART_STOPPED reason={scan_reason}; this is not evidence of global nonexistence", level=0)
        if not args.scan_only and not any(r.exact_target_reached for r in results):
            report("TARGET_SCOPE no target found on the explored charts; width and smoothing were not changed. "
                   "Inspect the scan direction, explored intervals and nearest distance before extending the search.", level=0)
        # All numerical output is committed before an interactive final pause.
        # Present the actual final target/nearest state, not the last trial.
        if plotter is not None:
            session.phase = "final plot inspection (numerical outputs committed)"
            phase_started = time.perf_counter()
            if args.scan_only:
                plotter.emit(scan.points[-1], stage="SCAN_END", force=True, pause=args.plot_final)
            else:
                for result in results:
                    if result.point is not None:
                        plotter.emit(result.point, stage=result.status, force=True, pause=args.plot_final)
            telemetry.record("final_plot_inspection", time.perf_counter()-phase_started)
        # Stop GUI pumping before the final accounting so the timing records
        # themselves cannot trigger an unrelated render/event failure.
        report.pump = None
        timing = telemetry.summarize(plotter=plotter, output_directory=store.path)
        telemetry_finalized = True
        session.record["timing"] = timing
        for line in timing_log_lines(timing):
            report(line, level=0)
        session.phase = "complete"
        if args.scan_only:
            return 0 if reached_stop else 2
        return 0 if any(result.exact_target_reached for result in results) else 2
    except OutputCancelled as error:
        session.outcome = "CANCELLED"
        report(f"RUN_CANCELLED {error}", level=0)
        return 2
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        session.outcome = "FAILED"
        session.record["error"] = f"{type(error).__name__}: {error}"
        report(f"RUN_FAILED phase={session.phase} error={type(error).__name__}: {error}", level=0)
        if args.verbosity == 2:
            report(traceback.format_exc().rstrip(), level=2)
        return 2
    finally:
        report.pump = None
        if not telemetry_finalized:
            # Error paths deliberately avoid a new MPI collective: a peer may
            # already be unwinding.  Rank zero still records local wall/memory
            # evidence, while RUN_END and per-rank status files carry outcome.
            partial = telemetry.local_snapshot(plotter)
            session.record["partial_timing"] = partial
            if comm.rank == 0:
                report("RUN_ABORT_TIMING " + compact_json(partial), level=0)
        if plotter is not None:
            plotter.close()
        if solver is not None:
            solver.close()
