"""Record and measure the README's GPU example, with bounded execution time.

The mesh, initial profile and solvers come from
``scripts/reports/gpu_showcase_setup.py`` and match
``examples/gpu_vortex_gas.py``; recording and validation controls stay out of
that short example. ``--movie`` writes an MP4 of two Matplotlib panels and
``--gif-mb`` optionally adds a byte-capped GIF. Every run records its code,
environment and GPU-stack provenance. Run as
``python -m scripts.reports.record_gpu_showcase``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
import hashlib
import json
import math
from pathlib import Path
import time
import threading

import numpy as np

import hybridge as hdg
from hybridge.diagnostics import transport_velocity_diagnostics
from hybridge.io import HolovizScalarPanels
from scripts.reports.gpu_showcase_setup import (
    COUNTS, ORDER, SEED, SIGMAS, poisson_solver, resolve_tau, showcase_mesh,
    showcase_profile, showcase_space, transport_solver)
from scripts.reports.run_provenance import run_provenance

ROOT = Path(__file__).resolve().parents[2]
# Velocity compatibility measures logged with every conservation row.
VELOCITY_ROW_KEYS = ("velocity_divergence_l2", "velocity_normal_jump_l2",
                     "velocity_double_outflow_measure_fraction",
                     "velocity_boundary_normal_relative_l2")


def positive_transport_options():
    """Reuse the guiding-center robust retries and corrected face upwinding."""
    from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset
    from scripts.guiding_center.runtime.configuration import _make_transport_options

    config_dir = Path(__file__).resolve().parents[2] / "configs" / "amgx"
    preset = GuidingCenterRunPreset(
        case="readme-positive", description="Positive-density recording recovery",
        time_scheme="si-bdf2", transport_solver="amgx", transport_preconditioner=None,
        transport_solver_rtol=1.e-9, transport_solver_atol=1.e-10,
        transport_scale_system=True, transport_retry_policy="amgx-robust",
        transport_amgx_config_path=str(config_dir / "adv_rea_gpu4_hdg_bicgstab_scaled_none.json"),
        transport_advection_stabilization="conflict-averaged-upwind")
    options = _make_transport_options(preset, "zero-flux")
    return dict(amgx_config=options.amgx_config,
                amgx_retry_attempts=options.amgx_retry_attempts,
                advection_stabilization=options.advection_stabilization)


def solve_positive_transport(transport, space, rhs, beta, reaction, trace, step, metadata):
    """Recover a rejected GPU solve on the host before committing either history.

    The host retry reassembles the same corrected raw-CUDA operator as COO,
    downloads the reduced system, and solves with nonsymmetric PyPardiso.
    The public solver uploads the accepted trace for GPU field reconstruction.
    """
    from hybridge.linalg.results import LinearSolveConvergenceError
    from hybridge.linalg.pardiso_runtime import pardiso_thread_limit

    try:
        return transport.solve(initial_guess=trace)
    except LinearSolveConvergenceError as error:
        event = dict(step=step, backend="pypardiso", reason=str(error),
                     gpu_attempts=list(getattr(error, "amgx_attempts", ())),
                     status="started")
        metadata.setdefault("transport_host_recoveries", []).append(event)
        print(json.dumps(dict(transport_recovery=event)), flush=True)
        # Observe actual process CPU use as well as the verified MKL limit.
        stop = threading.Event()
        samples = []

        def monitor():
            """Measure aggregate process CPU seconds per wall second."""
            wall, cpu = time.perf_counter(), time.process_time()
            while not stop.wait(.05):
                next_wall, next_cpu = time.perf_counter(), time.process_time()
                samples.append((next_cpu-cpu)/(next_wall-wall))
                wall, cpu = next_wall, next_cpu

        started = time.perf_counter()
        observer = threading.Thread(target=monitor, daemon=True)
        with pardiso_thread_limit(16) as actual_threads:
            event["mkl_threads"] = actual_threads
            observer.start()
            try:
                with closing(hdg.AdvectionReactionHDGSolver(
                    space, solver="pypardiso", preconditioner=None,
                    solver_rtol=transport.options.solver_rtol,
                    solver_atol=transport.options.solver_atol,
                    assembly_backend="raw-cuda", raw_local_assembly="fused",
                    raw_matrix_format="coo", boundary_mode="zero-flux",
                    trace_basis=transport.options.trace_basis,
                    advection_stabilization="conflict-averaged-upwind",
                    materialize_host_solution=False, scale_system=False,
                    verbose=False)) as host:
                    host.set_problem(rhs, beta, reaction, None)
                    result = host.solve(initial_guess=None)
                    if result.field_device is None or result.trace_device is None:
                        raise RuntimeError("Host recovery did not return device field and trace")
                    event.update(status="accepted",
                                 physical_residual=result.global_solve_result.physical_residual_norm,
                                 physical_target=result.global_solve_result.physical_residual_target)
                    return result
            except BaseException as host_error:
                event.update(status="failed", error=repr(host_error))
                raise
            finally:
                stop.set()
                observer.join()
                event.update(wall_seconds=time.perf_counter()-started,
                             observed_peak_cpu_cores=max(samples, default=0.))
                print(json.dumps(dict(transport_recovery=event)), flush=True)



def display_path(path):
    """Repository-relative path for metadata, without machine-specific prefixes."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def file_sha256(path):
    """Digest of a published media file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_restart(path, space, rho, potential, velocity, trace, step, args, tau, color_limits=None):
    """Atomically preserve only the full-precision final endpoint, not its history."""
    import cupy as cp
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space
    cspace = as_cupy_space(space)
    def coefficients(field):
        return cp.asnumpy(as_cupy_coefficients(field, cspace))
    arrays = dict(version=1, step=step, dt=args.dt, h=args.h, order=ORDER,
                  strength_mode=args.strength_mode, background=args.background,
                  amplitude=args.amplitude, cutoff=args.cutoff, poisson_tau=tau,
                  node_coords=space.mesh.node_coords, triangles=space.mesh.triangles,
                  rho=coefficients(rho), phi=coefficients(potential.field),
                  poisson_trace=cp.asnumpy(cp.asarray(hdg.solution_trace(potential, space))))
    for index, component in enumerate(velocity.components):
        arrays[f"velocity_{index}"] = coefficients(component)
    if trace is not None:
        arrays["transport_trace"] = cp.asnumpy(cp.asarray(trace))
    if color_limits is not None:
        arrays["color_limits"] = np.asarray(color_limits, dtype=float)
    path = Path(path)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def record(args):
    """Run one measured case, retaining compact diagnostics and optional frames."""
    import cupy as cp

    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / args.name
    if prefix.with_suffix(".jsonl").exists():
        raise FileExistsError(f"choose a fresh run name; {prefix}.jsonl already exists")
    metadata = dict(vars(args), output=display_path(output), seed=SEED, counts=list(COUNTS),
                    sigmas=list(SIGMAS), order=ORDER,
                    gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                    provenance=run_provenance(ROOT), status="initializing")
    if args.resume:
        metadata["resume"] = display_path(args.resume)

    def save():
        """Persist enough metadata to reproduce completed or interrupted work."""
        metadata["wall_seconds"] = time.perf_counter() - started
        prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    mesh = showcase_mesh(args.h)
    space = showcase_space(mesh)
    metadata.update(triangles=mesh.num_tri, trace_dofs=int(len(mesh.int_edges_inds) * (ORDER + 1)),
                    scalar_dofs=int(mesh.num_tri * space.el_dof), mesh_seconds=time.perf_counter()-started)
    profile_path = (Path(args.profile) if args.profile else output /
                    f"initial_{args.strength_mode}_cutoff{args.cutoff:g}_h{args.h:g}.npz")
    initial = showcase_profile(profile_path, mesh, strength_mode=args.strength_mode,
                               amplitude=args.amplitude, cutoff=args.cutoff)
    metadata["profile"] = display_path(profile_path)
    tau = resolve_tau(args.poisson_tau, space)
    metadata["poisson_tau_value"] = tau
    def initial_density(x, y):
        """Add a uniform physical background without clipping the evolution."""
        return initial(x, y) + args.background

    restart = inherited_limits = None
    start_step = 0
    if args.resume:
        restart = np.load(args.resume, allow_pickle=False)
        if int(restart["version"]) != 1:
            raise ValueError("unsupported restart version")
        # Checkpoints written before the cutoff and tau options used these values.
        defaults = dict(cutoff=8., poisson_tau=1000.)
        requested = dict(dt=args.dt, h=args.h, strength_mode=args.strength_mode,
                         background=args.background, amplitude=args.amplitude,
                         cutoff=args.cutoff, poisson_tau=tau)
        for key, value in requested.items():
            saved = restart[key].item() if key in restart else defaults[key]
            if saved != value:
                raise ValueError(f"restart {key} differs from requested run")
        if not (np.array_equal(restart["node_coords"], mesh.node_coords)
                and np.array_equal(restart["triangles"], mesh.triangles)):
            raise ValueError("restart mesh differs; refusing to remap coefficients")
        from hybridge.core.device import field_from_cupy_coefficients
        rho = field_from_cupy_coefficients(space, cp.asarray(restart["rho"], blocking=True))
        start_step = int(restart["step"])
        metadata["resumed_from_step"] = start_step
        # A continuation draws with its predecessor's color scales, so joined
        # movies keep one scale; older checkpoints keep them in their metadata.
        if "color_limits" in restart:
            inherited_limits = restart["color_limits"].tolist()
        else:
            predecessor = Path(args.resume).with_name(Path(args.resume).name.removesuffix(".restart.npz") + ".json")
            inherited_limits = (json.loads(predecessor.read_text()).get("color_limits")
                                if predecessor.exists() else None)
    else:
        rho = hdg.project_callable(initial_density, space, backend="device")
    final_step = start_step + args.steps
    print(f"mesh={mesh.num_tri} projection ready at {time.perf_counter()-started:.2f}s", flush=True)
    projection_error = hdg.evaluate_scalar_error(rho, initial_density, volume_quad_1d=9,
                                                backend="device").metrics.l2
    poisson = poisson_solver(space, rho, stabilization=tau)
    transport_recovery_options = positive_transport_options() if args.strength_mode == "positive" else {}
    metadata["transport_recovery_policy"] = (
        "guiding-center-amgx-robust-then-host-pypardiso" if transport_recovery_options else "none")
    metadata["advection_stabilization"] = transport_recovery_options.get("advection_stabilization", "conflict-averaged-upwind")
    transport = transport_solver(space, **transport_recovery_options)
    previous_rho = previous_velocity = trace = None
    one = space.constant(1.)  # Reaction of the scaled BDF2 transport step.
    rows = []
    positivity = None
    if args.strength_mode == "positive":
        from hybridge.diagnostics.guiding_center import ScalarPositivityDiagnostics
        positivity = ScalarPositivityDiagnostics(space, backend="device")
    primary_error = None
    potential = velocity = None
    transport_iterations = None
    accepted_step = start_step
    def checkpoint():
        if potential is not None and velocity is not None:
            path = prefix.with_suffix(".restart.npz")
            save_restart(path, space, rho, potential, velocity, trace, accepted_step, args, tau,
                         metadata.get("color_limits"))
            metadata["restart_checkpoint"] = display_path(path)
            metadata["checkpoint_step"] = accepted_step
    try:
        potential = poisson.solve(initial_guess=(
            cp.asarray(restart["poisson_trace"], blocking=True) if restart is not None else None))
        velocity = hdg.perpendicular_vector_field(potential.flux)
        if restart is not None:
            from hybridge.core.device import field_from_cupy_coefficients
            from hybridge.core.space import VectorDGField
            def restored(key):
                return field_from_cupy_coefficients(space, cp.asarray(restart[key], blocking=True), name=key)
            velocity = VectorDGField(tuple(restored(f"velocity_{i}") for i in range(2)))
            # Endpoint-only restart: use one Euler step to rebuild BDF2 history.
            metadata["restart_startup"] = "Euler, then BDF2"
            if "transport_trace" in restart:
                trace = cp.asarray(restart["transport_trace"], blocking=True)
            restart.close()
        metadata["initial_velocity"] = transport_velocity_diagnostics(velocity, backend="device")
        metadata["initial_step_speed_over_min_edge"] = (
            args.dt * metadata["initial_velocity"]["velocity_max_speed_over_min_edge"])
        print(f"initial Poisson ready at {time.perf_counter()-started:.2f}s", flush=True)

        def measure(step):
            """Download scalar conservation and velocity measures, leaving fields resident."""
            values = hdg.guiding_center_field_diagnostics(rho, potential.field,
                       potential.flux, backend="device")
            velocity_values = transport_velocity_diagnostics(velocity, backend="device")
            row = dict(step=step, time=step*args.dt, circulation=values["mass"],
                       enstrophy=.5*values["rho_l2_squared"],
                       energy=.5*values["q_l2_standard"]**2,
                       minimum=values["rho_min"], maximum=values["rho_max"],
                       potential_minimum=values["phi_min"], potential_maximum=values["phi_max"],
                       poisson_iterations=potential.global_solve_result.iteration_count,
                       transport_iterations=transport_iterations,
                       elapsed=time.perf_counter()-started)
            row.update({key: velocity_values[key] for key in VELOCITY_ROW_KEYS})
            if positivity is not None:
                row.update(positivity.measure(rho))
            if not all(math.isfinite(row[key]) for key in (
                "circulation", "enstrophy", "energy", "minimum", "maximum",
                "potential_minimum", "potential_maximum")):
                raise FloatingPointError("nonfinite field diagnostics; refusing to render the state")
            if rows:
                row.update(enstrophy_loss=1-row["enstrophy"]/rows[0]["enstrophy"],
                           energy_drift=row["energy"]/rows[0]["energy"]-1)
            rows.append(row)
            with prefix.with_suffix(".jsonl").open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            return row

        initial_row = measure(start_step)
        metadata["projection_relative_l2"] = projection_error / np.sqrt(2*initial_row["enstrophy"])
        metadata["initialization_seconds"] = time.perf_counter()-started
        metadata["color_limit"] = limit = (float(inherited_limits[0][1]) if inherited_limits else
                                           float(np.ceil(max(abs(initial_row["minimum"]),
                                                             abs(initial_row["maximum"])))))
        metadata["status"] = "running"
        save()
        with ExitStack() as stack:
            viewer = None
            gif = panels = sampler = mp4 = None
            frames = 0
            if args.movie or args.gif_mb is not None:
                from PIL import Image
                from hybridge.io import GifWriter, MatplotlibRasterPanels
                from hybridge.io.raster import DeviceRasterSampler, RasterGeometry

                geometry = RasterGeometry.from_mesh(mesh, args.raster_size, args.raster_size)
                sampler = DeviceRasterSampler(space, geometry, device_id=cp.cuda.runtime.getDevice())
                mask = geometry.element_ids.reshape(geometry.height, geometry.width) < 0

                def rasters():
                    """Download only two discontinuous DG rasters for CPU drawing."""
                    images = []
                    for field in (rho, potential.field):
                        image = cp.asnumpy(sampler.sample(field)).reshape(geometry.height, geometry.width)
                        images.append(np.ma.array(image, mask=mask))
                    return tuple(images)

                def save_rasters(images, step):
                    """Archive display samples so colors can be changed without a GPU rerun."""
                    if not args.save_rasters:
                        return
                    directory = output / f"{args.name}_rasters"
                    directory.mkdir(exist_ok=True)
                    np.savez_compressed(directory / f"frame_{step:07d}.npz",
                        rho=np.asarray(images[0].filled(np.nan), dtype=np.float32),
                        phi=np.asarray(images[1].filled(np.nan), dtype=np.float32),
                        bounds=geometry.bounds, step=step, time=step*args.dt)

                initial_images = rasters()
                signed = args.strength_mode == "balanced"
                phi_limit = (float(inherited_limits[1][1]) if inherited_limits else
                             (1.5 if signed else 1.15) * max(abs(initial_row["potential_minimum"]),
                                       abs(initial_row["potential_maximum"]), 1.e-12))
                if signed:
                    rho_options = dict(cmap="RdBu_r", clim=(-limit, limit))
                    phi_options = dict(cmap="RdBu_r", clim=(-phi_limit, phi_limit))
                else:
                    import matplotlib as mpl

                    # Sequential density from zero. Undershoots below a small
                    # fraction of the scale get a distinct color, not the
                    # bottom of the map, so positivity loss stays visible.
                    density_map = mpl.colormaps["viridis"].with_extremes(under=args.negative_color)
                    density_map.colorbar_extend = "min"
                    rho_options = dict(cmap=density_map, clim=(-args.negative_threshold*limit, limit))
                    phi_options = dict(cmap="cividis", clim=(0., phi_limit))
                metadata["color_limits"] = [list(rho_options["clim"]), list(phi_options["clim"])]
                if inherited_limits and not np.allclose(metadata["color_limits"], inherited_limits):
                    raise ValueError("inherited color limits do not match this strength mode")
                metadata["color_limits_inherited"] = inherited_limits is not None
                label = "Two-species guiding-center plasma" if signed else "Single-species guiding-center plasma"

                def caption(step):
                    return f"{label} | p = {ORDER} | t = {step*args.dt:.2f}"

                panels = stack.enter_context(MatplotlibRasterPanels(
                    [(r"Charge density $\rho$", initial_images[0], rho_options),
                     (r"Potential $\phi$", initial_images[1], phi_options)],
                    geometry.bounds, title=caption(start_step), size=(args.width, args.height),
                    boundary_mesh=mesh, background=args.plot_background,
                    font_size=args.font_size, ticks=False))
                if args.gif_mb is not None:
                    gif = stack.enter_context(GifWriter(prefix.with_suffix(".gif"), fps=args.fps,
                                                        max_bytes=round(args.gif_mb*1_000_000)))
                if args.movie:
                    from hybridge.io.movie import MovieWriter

                    mp4 = MovieWriter(prefix.with_suffix(".mp4"), fps=args.fps)
                    stack.callback(mp4.close)
                frame = panels.capture()
                if gif is not None and not gif.append(frame):
                    raise ValueError("GIF budget cannot hold its initial frame")
                if mp4 is not None:
                    mp4.append(frame)
                frames, last_frame = 1, frame
                metadata.update(last_rendered_step=start_step, last_rendered_time=start_step*args.dt,
                                rendered_frames=frames, last_rendered=initial_row)
                if gif is not None:
                    metadata.update(gif_bytes=gif.bytes_written, gif_duration_quantization_ms=10)
                Image.fromarray(frame).save(prefix.with_suffix(".png"))
                save_rasters(initial_images, start_step)
            elif args.holoviz_movie:
                viewer = stack.enter_context(HolovizScalarPanels(
                    (space,), ("Charge density",), width=640, height=640,
                    title="HDG guiding-center plasma", cmap="RdBu_r", show_mesh=False,
                    off_screen=True, movie_path=prefix.with_suffix(".mp4"), movie_fps=12.,
                    screenshot_dir=output / f"{args.name}_frames"))
                viewer.update_fields((rho,), limits=(-limit, limit), captions=("t = 0.00",))
                metadata.update(last_rendered_step=0, last_rendered_time=0.)
            metadata["completed_steps"] = start_step
            for step in range(start_step+1, final_step+1):
                if time.perf_counter()-started >= args.seconds:
                    metadata["status"] = "time_budget"
                    break
                cp.cuda.runtime.deviceSynchronize()
                rhs, beta, _ = hdg.bdf2_transport_data(rho, velocity, args.dt,
                    previous_field=previous_rho, previous_velocity=previous_velocity)
                cp.cuda.runtime.deviceSynchronize()
                if args.strength_mode == "positive":
                    transport.set_problem(rhs, beta, one, None)
                    result = solve_positive_transport(
                        transport, space, rhs, beta, one, trace, step, metadata)
                else:
                    # The example's call; only a resumed run seeds its first guess.
                    guess = dict(initial_guess=trace) if trace is not None and step == start_step+1 else {}
                    result = transport.solve(source=rhs, beta=beta, reaction=one, **guess)
                transport_iterations = result.global_solve_result.iteration_count
                next_rho = result.field
                next_potential = poisson.set_source(next_rho).solve()
                previous_rho, previous_velocity, rho = rho, velocity, next_rho
                potential = next_potential
                velocity = hdg.perpendicular_vector_field(potential.flux)
                trace = hdg.solution_trace(result, space)
                accepted_step = step
                metadata["completed_steps"] = step
                if step % args.every == 0 or step == final_step:
                    row = measure(step)
                    print(json.dumps(row), flush=True)
                    if abs(row["enstrophy_loss"]) > args.max_loss:
                        metadata["status"] = "enstrophy_limit"
                        break
                    if viewer:
                        viewer.update_fields((rho,), step=step, time_value=step*args.dt,
                            limits=(-limit, limit), captions=(f"t = {step*args.dt:.2f}",))
                        metadata.update(last_rendered_step=step, last_rendered_time=step*args.dt)
                    if panels is not None:
                        images = rasters()
                        panels.update(images, caption=caption(step))
                        frame = panels.capture()
                        if gif is not None and not gif.append(frame):
                            metadata["status"] = "gif_budget"
                            break
                        if mp4 is not None:
                            mp4.append(frame)
                        frames, last_frame = frames + 1, frame
                        metadata.update(last_rendered_step=step, last_rendered_time=step*args.dt,
                                        rendered_frames=frames, last_rendered=row)
                        if gif is not None:
                            metadata["gif_bytes"] = gif.bytes_written
                        save_rasters(images, step)
                        if frames % 25 == 0:
                            Image.fromarray(frame).save(prefix.with_suffix(".png"))
                    save()
            else:
                metadata["status"] = "completed"
            if rows[-1]["step"] != metadata["completed_steps"]:
                measure(metadata["completed_steps"])
            metadata["final"] = rows[-1]
            metadata["final_velocity"] = transport_velocity_diagnostics(velocity, backend="device")
            metadata["device_used_gib_at_finish"] = (
                cp.cuda.runtime.memGetInfo()[1]-cp.cuda.runtime.memGetInfo()[0])/2**30
            if viewer:
                metadata["rendered_frames"] = viewer.frames_rendered
            if panels is not None:
                playback = frames / args.fps
                metadata.update(playback_seconds=playback, simulation_time_per_playback_second=(
                    (metadata["last_rendered_time"]-start_step*args.dt) / playback))
                if gif is not None:
                    # The rejected endpoint is measured, but the PNG and GIF
                    # retain the last frame that fit the requested budget.
                    metadata.update(gif_bytes=gif.bytes_written, rendered_frames=gif.frames_written,
                                    gif_playback_seconds=gif.playback_ms/1000)
                Image.fromarray(last_frame).save(prefix.with_suffix(".png"))
        # Encoders are closed here, so the media files are complete.
        for suffix, written in ((".mp4", args.movie), (".gif", args.gif_mb is not None)):
            path = prefix.with_suffix(suffix)
            if written and path.exists():
                metadata[f"{suffix[1:]}_bytes"] = path.stat().st_size
                metadata[f"{suffix[1:]}_sha256"] = file_sha256(path)
    except BaseException as error:
        primary_error = error
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        cleanup_error = None
        for label, finish in (("final checkpoint", checkpoint), ("metadata save", save), ("transport close", transport.close),
                              ("Poisson close", poisson.close)):
            try:
                finish()
            except BaseException as error:
                active_error = primary_error if primary_error is not None else cleanup_error
                if active_error is None:
                    cleanup_error = error
                else:
                    add_note = getattr(active_error, "add_note", None)
                    if callable(add_note):
                        add_note(f"Showcase {label} also failed: {error!r}")
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    """Parse recording controls without changing the introductory example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", type=float, default=.006)
    parser.add_argument("--dt", type=float, default=.0125)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=180.)
    parser.add_argument("--max-loss", type=float, default=.05)
    parser.add_argument("--poisson-tau", default="1000",
                        help="Poisson stabilization: a positive number or 'global' (kappa/L)")
    parser.add_argument("--cutoff", type=float, default=8.,
                        help="blob cutoff in core widths; also the sampled wall clearance")
    parser.add_argument("--movie", action="store_true",
                        help="write an MP4 of the density and potential Matplotlib panels")
    parser.add_argument("--gif-mb", type=float,
                        help="also write a GIF and stop before it exceeds this many megabytes")
    parser.add_argument("--holoviz-movie", action="store_true",
                        help="record one off-screen Holoviz density panel instead of Matplotlib")
    parser.add_argument("--fps", type=float, default=12.)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--font-size", type=float, default=18.)
    parser.add_argument("--raster-size", type=int, default=720)
    parser.add_argument("--save-rasters", action="store_true",
                        help="archive CPU display samples for recoloring without rerunning the solver")
    parser.add_argument("--checkpoint", action="store_true", default=True,
                        help="compatibility flag: full final checkpoints are always saved")
    parser.add_argument("--resume", help="continue from a full .restart.npz checkpoint")
    parser.add_argument("--plot-background", default="#dfe3e8",
                        help="Matplotlib color of the coordinate boxes around the domain")
    parser.add_argument("--negative-color", default="#ff2d55",
                        help="positive case: color of density below the negative threshold")
    parser.add_argument("--negative-threshold", type=float, default=.01,
                        help="positive case: flag density below -threshold * color limit")
    parser.add_argument("--profile")
    parser.add_argument("--strength-mode", choices=("balanced", "positive"), default="balanced")
    parser.add_argument("--background", type=float, default=0.)
    parser.add_argument("--amplitude", type=float, default=4.)
    parser.add_argument("--output", default="outputs/readme_showcase")
    parser.add_argument("--name", default="calibration")
    args = parser.parse_args()
    from matplotlib.colors import is_color_like

    if (any(not math.isfinite(value) or value <= 0 for value in (args.h, args.dt, args.seconds))
            or args.steps < 1 or args.every < 1):
        parser.error("h, dt, steps, every and seconds must be positive")
    if not math.isfinite(args.max_loss) or not 0 < args.max_loss < 1:
        parser.error("max-loss must lie strictly between zero and one")
    if not math.isfinite(args.background) or args.background < 0:
        parser.error("background must be finite and nonnegative")
    if not math.isfinite(args.amplitude) or args.amplitude <= 0:
        parser.error("amplitude must be finite and positive")
    if not math.isfinite(args.cutoff) or args.cutoff <= 0:
        parser.error("cutoff must be finite and positive")
    if args.poisson_tau != "global":
        try:
            if not math.isfinite(float(args.poisson_tau)) or float(args.poisson_tau) <= 0:
                raise ValueError
        except ValueError:
            parser.error("poisson-tau must be a positive number or 'global'")
    if args.gif_mb is not None and (not math.isfinite(args.gif_mb) or args.gif_mb <= 0):
        parser.error("gif-mb must be finite and positive")
    if args.holoviz_movie and (args.movie or args.gif_mb is not None):
        parser.error("holoviz-movie excludes the Matplotlib --movie and --gif-mb outputs")
    if not math.isfinite(args.fps) or not 0 < args.fps <= 100:
        parser.error("fps must lie in (0, 100]")
    if min(args.width, args.height, args.raster_size) < 2:
        parser.error("image dimensions must be at least two pixels")
    if not math.isfinite(args.font_size) or args.font_size <= 0:
        parser.error("font-size must be finite and positive")
    if not (is_color_like(args.plot_background) and is_color_like(args.negative_color)):
        parser.error("plot-background and negative-color must be Matplotlib colors")
    if not math.isfinite(args.negative_threshold) or not 0 <= args.negative_threshold < 1:
        parser.error("negative-threshold must lie in [0, 1)")
    record(args)


if __name__ == "__main__":
    main()
