"""Record and measure the README's GPU example, with bounded execution time.

The application remains a composition of the public Poisson and transport
solvers. Recording and validation controls deliberately live outside the
short introductory example. Run as ``python -m scripts.reports.record_gpu_showcase``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
import json
import math
from pathlib import Path
import time
import threading

import numpy as np

import hdgfem as hdg
from hdgfem.cases.profiles import GaussianBlobField
from hdgfem.diagnostics import transport_velocity_diagnostics
from hdgfem.io import HolovizScalarPanels


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
    from hdgfem.linalg.results import LinearSolveConvergenceError
    from hdgfem.linalg.pardiso_runtime import pardiso_thread_limit

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


def save_restart(path, space, rho, potential, velocity, trace, step, args):
    """Atomically preserve only the full-precision final endpoint, not its history."""
    import cupy as cp
    from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
    cspace = as_cupy_space(space)
    def coefficients(field):
        return cp.asnumpy(as_cupy_coefficients(field, cspace))
    arrays = dict(version=1, step=step, dt=args.dt, h=args.h, order=6,
                  strength_mode=args.strength_mode, background=args.background,
                  amplitude=args.amplitude, node_coords=space.mesh.node_coords,
                  triangles=space.mesh.triangles, rho=coefficients(rho),
                  phi=coefficients(potential.field),
                  poisson_trace=cp.asnumpy(cp.asarray(hdg.solution_trace(potential, space))))
    for index, component in enumerate(velocity.components):
        arrays[f"velocity_{index}"] = coefficients(component)
    if trace is not None:
        arrays["transport_trace"] = cp.asnumpy(cp.asarray(trace))
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
    metadata = dict(vars(args), seed=17, counts=[512, 256, 128, 64],
                    sigmas=[.008, .016, .024, .032], order=6,
                    gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                    status="initializing")

    def save():
        """Persist enough metadata to reproduce completed or interrupted work."""
        metadata["wall_seconds"] = time.perf_counter() - started
        prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    mesh = hdg.gmsh_smooth_star_mesh(
        args.h, radius=1., amplitude=.35, mode=5, hole_radius=.3,
        boundary_points=500, num_threads=16, log_cache=False,
    )
    space = hdg.DGSpace(mesh, 6, basis_type="dub_orth")
    metadata.update(triangles=mesh.num_tri, trace_dofs=int(len(mesh.int_edges_inds) * 7),
                    scalar_dofs=int(mesh.num_tri * 28), mesh_seconds=time.perf_counter()-started)
    profile_path = (Path(args.profile) if args.profile else output /
                    ("initial_profile.npz" if args.strength_mode == "balanced"
                     else "initial_positive_profile.npz"))
    if profile_path.exists():
        with np.load(profile_path) as data:
            saved_amplitude = float(data["amplitude"]) if "amplitude" in data else 4.
            initial = GaussianBlobField(data["centers"], data["sigmas"],
                                        data["strengths"] * args.amplitude / saved_amplitude)
    else:
        initial = hdg.sample_gaussian_blob_field(hdg.MeshDomain(mesh),
                    (512, 256, 128, 64), (.008, .016, .024, .032), seed=17,
                    strength_mode=args.strength_mode, amplitude=args.amplitude)
        np.savez(profile_path, centers=initial.centers, sigmas=initial.sigmas,
                 strengths=initial.strengths, amplitude=args.amplitude)
    metadata["profile"] = str(profile_path)
    def initial_density(x, y):
        """Add a uniform physical background without clipping the evolution."""
        return initial(x, y) + args.background

    restart = None
    start_step = 0
    if args.resume:
        restart = np.load(args.resume, allow_pickle=False)
        if int(restart["version"]) != 1:
            raise ValueError("unsupported restart version")
        for key in ("dt", "h", "strength_mode", "background", "amplitude"):
            if restart[key].item() != getattr(args, key):
                raise ValueError(f"restart {key} differs from requested run")
        if not (np.array_equal(restart["node_coords"], mesh.node_coords)
                and np.array_equal(restart["triangles"], mesh.triangles)):
            raise ValueError("restart mesh differs; refusing to remap coefficients")
        from hdgfem.core.device import field_from_cupy_coefficients
        rho = field_from_cupy_coefficients(space, cp.asarray(restart["rho"], blocking=True))
        start_step = int(restart["step"])
        metadata["resumed_from_step"] = start_step
    else:
        rho = hdg.project_callable(initial_density, space, backend="device")
    print(f"mesh={mesh.num_tri} projection ready at {time.perf_counter()-started:.2f}s", flush=True)
    projection_error = hdg.evaluate_scalar_error(rho, initial_density, volume_quad_1d=9,
                                                backend="device").metrics.l2
    common = dict(assembly_backend="raw-cuda", raw_matrix_format="bsr",
                  solver_rtol=1.e-9, solver_atol=1.e-10, verbose=False)
    poisson = hdg.DiffusionReactionHDGSolver(
        space, source=rho, reaction=space.zeros(), boundary_condition=0.,
        solver="fb-hp-mg-pcg", trace_basis="legendre-modal", scale_system=False,
        stabilization=1000., boundary_mode="eliminate",
        cache_local_factors="schur-cholesky", **common)
    transport_recovery_options = positive_transport_options() if args.strength_mode == "positive" else {}
    metadata["transport_recovery_policy"] = (
        "guiding-center-amgx-robust-then-host-pypardiso" if transport_recovery_options else "none")
    metadata["advection_stabilization"] = transport_recovery_options.get("advection_stabilization", "conflict-averaged-upwind")
    transport = hdg.AdvectionReactionHDGSolver(
        space, solver="amgx", boundary_mode="zero-flux", raw_local_assembly="fused",
        materialize_host_solution=False, scale_system=True, **common, **transport_recovery_options)
    previous_rho = previous_velocity = trace = None
    rows = []
    positivity = None
    if args.strength_mode == "positive":
        from hdgfem.diagnostics.guiding_center import ScalarPositivityDiagnostics
        positivity = ScalarPositivityDiagnostics(space, backend="device")
    primary_error = None
    potential = velocity = None
    accepted_step = start_step
    def checkpoint():
        if potential is not None and velocity is not None:
            path = prefix.with_suffix(".restart.npz")
            save_restart(path, space, rho, potential, velocity, trace, accepted_step, args)
            metadata["restart_checkpoint"] = str(path)
            metadata["checkpoint_step"] = accepted_step
    try:
        potential = poisson.solve(initial_guess=(
            cp.asarray(restart["poisson_trace"], blocking=True) if restart is not None else None))
        velocity = hdg.perpendicular_vector_field(potential.flux, 1., space)
        if restart is not None:
            from hdgfem.core.device import field_from_cupy_coefficients
            from hdgfem.core.space import VectorDGField
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
            """Download scalar conservation measures only, leaving fields resident."""
            values = hdg.guiding_center_field_diagnostics(rho, potential.field,
                       potential.flux, backend="device")
            row = dict(step=step, time=step*args.dt, circulation=values["mass"],
                       enstrophy=.5*values["rho_l2_squared"],
                       energy=.5*values["q_l2_standard"]**2,
                       minimum=values["rho_min"], maximum=values["rho_max"],
                       potential_minimum=values["phi_min"], potential_maximum=values["phi_max"],
                       elapsed=time.perf_counter()-started)
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
        metadata["color_limit"] = limit = float(np.ceil(max(abs(initial_row["minimum"]),
                                                          abs(initial_row["maximum"]))))
        metadata["status"] = "running"
        save()
        with ExitStack() as stack:
            viewer = None
            gif = panels = sampler = mp4 = None
            if args.gif_mb is not None:
                from hdgfem.io import GifWriter, MatplotlibRasterPanels
                from hdgfem.io.raster import DeviceRasterSampler, RasterGeometry

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
                phi_limit = (1.5 if signed else 1.15) * max(abs(initial_row["potential_minimum"]),
                                       abs(initial_row["potential_maximum"]), 1.e-12)
                # Positive initial data can develop negative undershoots: retain
                # a visible negative range rather than clipping those samples.
                rho_limits = (-limit, limit) if signed else (-.25*limit, limit)
                phi_limits = (-phi_limit, phi_limit) if signed else (0., phi_limit)
                metadata["color_limits"] = [list(rho_limits), list(phi_limits)]
                label = "Signed vorticity" if signed else "Initially positive density"
                panel_specs = [
                    (r"Vorticity $\omega$" if signed else r"Density $\rho$", initial_images[0],
                     dict(cmap="RdBu_r" if signed else "viridis", clim=rho_limits)),
                    (r"Potential $\phi$", initial_images[1],
                     dict(cmap="RdBu_r" if signed else "cividis", clim=phi_limits))]
                panels = stack.enter_context(MatplotlibRasterPanels(
                    panel_specs, geometry.bounds, title=f"{label} | p = 6 | t = {start_step*args.dt:.4f}",
                    size=(args.width, args.height), boundary_mesh=mesh,
                    background=args.plot_background))
                gif = stack.enter_context(GifWriter(prefix.with_suffix(".gif"), fps=args.fps,
                                                    max_bytes=round(args.gif_mb*1_000_000)))
                if args.movie:
                    from hdgfem.io.movie import MovieWriter

                    mp4 = MovieWriter(prefix.with_suffix(".mp4"), fps=args.fps)
                    stack.callback(mp4.close)
                frame = panels.capture()
                if not gif.append(frame):
                    raise ValueError("GIF budget cannot hold its initial frame")
                last_frame = frame
                if mp4 is not None:
                    mp4.append(frame)
                metadata.update(last_rendered_step=start_step, last_rendered_time=start_step*args.dt,
                                gif_bytes=gif.bytes_written, rendered_frames=1,
                                gif_duration_quantization_ms=10,
                                last_rendered=initial_row)
                from PIL import Image

                Image.fromarray(frame).save(prefix.with_suffix(".png"))
                save_rasters(initial_images, start_step)
            elif args.movie:
                viewer = stack.enter_context(HolovizScalarPanels(
                    (space,), ("Vorticity",), width=640, height=640,
                    title="HDG vortex gas", cmap="RdBu_r", show_mesh=False,
                    off_screen=True, movie_path=prefix.with_suffix(".mp4"), movie_fps=12.,
                    screenshot_dir=output / f"{args.name}_frames"))
                viewer.update_fields((rho,), limits=(-limit, limit), captions=("t = 0.00",))
                metadata.update(last_rendered_step=0, last_rendered_time=0.)
            metadata["completed_steps"] = start_step
            for step in range(start_step+1, start_step+args.steps+1):
                if time.perf_counter()-started >= args.seconds:
                    metadata["status"] = "time_budget"
                    break
                cp.cuda.runtime.deviceSynchronize()
                rhs, beta, _ = hdg.bdf2_transport_data(rho, velocity, args.dt,
                    previous_field=previous_rho, previous_velocity=previous_velocity)
                cp.cuda.runtime.deviceSynchronize()
                reaction = space.constant(1.)
                transport.set_problem(rhs, beta, reaction, None)
                if args.strength_mode == "positive":
                    result = solve_positive_transport(
                        transport, space, rhs, beta, reaction, trace, step, metadata)
                else:
                    result = transport.solve(initial_guess=trace)
                next_rho = hdg.solution_field(result, space).copy()
                poisson.set_source(next_rho)
                next_potential = poisson.solve(initial_guess=hdg.solution_trace(potential, space))
                previous_rho, previous_velocity, rho = rho, velocity, next_rho
                potential = next_potential
                velocity = hdg.perpendicular_vector_field(potential.flux, 1., space)
                trace = hdg.solution_trace(result, space).copy()
                accepted_step = step
                metadata["completed_steps"] = step
                if step % args.every == 0 or step == args.steps:
                    row = measure(step)
                    print(json.dumps(row), flush=True)
                    if abs(row["enstrophy_loss"]) > args.max_loss:
                        metadata["status"] = "enstrophy_limit"
                        break
                    if viewer:
                        viewer.update_fields((rho,), step=step, time_value=step*args.dt,
                            limits=(-limit, limit), captions=(f"t = {step*args.dt:.2f}",))
                        metadata.update(last_rendered_step=step, last_rendered_time=step*args.dt)
                    if gif is not None:
                        images = rasters()
                        panels.update(images, caption=f"{label} | p = 6 | t = {step*args.dt:.4f}")
                        frame = panels.capture()
                        if not gif.append(frame):
                            metadata["status"] = "gif_budget"
                            break
                        last_frame = frame
                        if mp4 is not None:
                            mp4.append(frame)
                        metadata.update(last_rendered_step=step, last_rendered_time=step*args.dt,
                                        gif_bytes=gif.bytes_written, rendered_frames=gif.frames_written,
                                        last_rendered=row)
                        save_rasters(images, step)
                        if gif.frames_written % 25 == 0:
                            Image.fromarray(frame).save(prefix.with_suffix(".png"))
                    save()
            else:
                metadata["status"] = "completed"
            if rows[-1]["step"] != metadata["completed_steps"]:
                measure(metadata["completed_steps"])
            metadata["final"] = rows[-1]
            metadata["device_used_gib_at_finish"] = (
                cp.cuda.runtime.memGetInfo()[1]-cp.cuda.runtime.memGetInfo()[0])/2**30
            if viewer:
                metadata["rendered_frames"] = viewer.frames_rendered
            if gif is not None:
                metadata.update(gif_bytes=gif.bytes_written, rendered_frames=gif.frames_written,
                                gif_playback_seconds=gif.playback_ms/1000,
                                simulation_time_per_playback_second=(
                                    (metadata["last_rendered_time"]-start_step*args.dt) / (gif.playback_ms/1000)))
                # The rejected endpoint is measured, but the PNG and GIF retain
                # the last frame that fit the requested file budget.
                Image.fromarray(last_frame).save(prefix.with_suffix(".png"))
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
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--gif-mb", type=float,
                        help="record density and potential with Matplotlib until this GIF byte budget")
    parser.add_argument("--fps", type=float, default=12.)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--raster-size", type=int, default=720)
    parser.add_argument("--save-rasters", action="store_true",
                        help="archive CPU display samples for recoloring without rerunning the solver")
    parser.add_argument("--checkpoint", action="store_true", default=True,
                        help="compatibility flag: full final checkpoints are always saved")
    parser.add_argument("--resume", help="continue from a full .restart.npz checkpoint")
    parser.add_argument("--plot-background", choices=("white", "black"), default="black")
    parser.add_argument("--profile")
    parser.add_argument("--strength-mode", choices=("balanced", "positive"), default="balanced")
    parser.add_argument("--background", type=float, default=0.)
    parser.add_argument("--amplitude", type=float, default=4.)
    parser.add_argument("--output", default="outputs/readme_showcase")
    parser.add_argument("--name", default="calibration")
    args = parser.parse_args()
    if (any(not math.isfinite(value) or value <= 0 for value in (args.h, args.dt, args.seconds))
            or args.steps < 1 or args.every < 1):
        parser.error("h, dt, steps, every and seconds must be positive")
    if not math.isfinite(args.max_loss) or not 0 < args.max_loss < 1:
        parser.error("max-loss must lie strictly between zero and one")
    if not math.isfinite(args.background) or args.background < 0:
        parser.error("background must be finite and nonnegative")
    if not math.isfinite(args.amplitude) or args.amplitude <= 0:
        parser.error("amplitude must be finite and positive")
    if args.gif_mb is not None and (not math.isfinite(args.gif_mb) or args.gif_mb <= 0):
        parser.error("gif-mb must be finite and positive")
    if not math.isfinite(args.fps) or not 0 < args.fps <= 100:
        parser.error("fps must lie in (0, 100]")
    if min(args.width, args.height, args.raster_size) < 2:
        parser.error("image dimensions must be at least two pixels")
    record(args)


if __name__ == "__main__":
    main()
