"""Probe the Poisson stabilization tau behind the GPU showcase velocity.

For each tau, solve the showcase Poisson problem on the device and record the
solver work, the face diagnostics of u = (-q_y, q_x), and the velocity and
potential errors against a higher-degree reference on the same mesh (host
Numba assembly and 16-thread PyPardiso; the raw-CUDA path stops at p = 6).
The shared mesh isolates the p = 6 discretization error from wall geometry.
Errors use shared physical points, split into a wall band and the interior.
No time integration is performed. Run as
``python -m scripts.reports.probe_poisson_tau``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np

import hdgfem as hdg
from hdgfem.diagnostics import transport_velocity_diagnostics
from hdgfem.io.raster import DeviceRasterSampler, RasterGeometry
from scripts.reports.gpu_showcase_setup import (
    TOLERANCES, poisson_solver, resolve_tau, showcase_mesh, showcase_profile, showcase_space)


class PointSampler:
    """Evaluate fields of one space at fixed physical points on the device."""

    def __init__(self, space, points):
        geometry = RasterGeometry.from_points(space.mesh, points, width=len(points), height=1)
        self.valid = geometry.element_ids >= 0
        self.sampler = DeviceRasterSampler(space, geometry, device_id=0)

    def __call__(self, field):
        return self.sampler.sample(field)


def host_reference(mesh, profile, points, *, order, tau):
    """Solve on the host at a higher degree and sample at the shared points.

    PyPardiso runs under a verified 16-thread MKL limit, and the process CPU
    use is measured around the solve.
    """
    from hdgfem.linalg.pardiso_runtime import pardiso_thread_limit

    space = hdg.DGSpace(mesh, order, basis_type="dub_orth")
    # The blob sum is evaluated on the device; the host solver downloads the
    # projected coefficients once.
    rho = hdg.project_callable(profile, space, backend="device")
    stop, cores = threading.Event(), []

    def monitor():
        """Aggregate process CPU seconds per wall second."""
        wall, cpu = time.perf_counter(), time.process_time()
        while not stop.wait(.1):
            next_wall, next_cpu = time.perf_counter(), time.process_time()
            cores.append((next_cpu - cpu) / (next_wall - wall))
            wall, cpu = next_wall, next_cpu

    started = time.perf_counter()
    with pardiso_thread_limit(16) as threads:
        observer = threading.Thread(target=monitor, daemon=True)
        observer.start()
        try:
            solver = hdg.DiffusionReactionHDGSolver(
                space, source=rho, reaction=space.zeros(), boundary_condition=0.,
                stabilization=tau, assembly_backend="numba", boundary_mode="eliminate",
                solver="pypardiso-spd", trace_basis="legendre-modal", verbose=False,
                **TOLERANCES)
            try:
                result = solver.solve()
            finally:
                solver.close()
        finally:
            stop.set()
            observer.join()
    velocity = hdg.perpendicular_vector_field(result.flux, 1., space)
    geometry = RasterGeometry.from_points(mesh, points, width=len(points), height=1)
    matrix = geometry.sampling_matrix(space, max_bytes=4 * 1024**3)
    samples = dict(ux=matrix @ np.ravel(velocity.components[0].coeffs),
                   uy=matrix @ np.ravel(velocity.components[1].coeffs),
                   phi=matrix @ np.ravel(result.field.coeffs))
    info = dict(order=order, tau=tau, seconds=time.perf_counter() - started,
                mkl_threads=threads, observed_peak_cpu_cores=max(cores, default=0.),
                trace_assembly_seconds=result.timings.trace_assembly,
                solve_seconds=result.timings.solve,
                physical_residual=getattr(result.global_solve_result, "physical_residual_norm", None))
    return info, samples, geometry.element_ids >= 0


def solve_case(space, rho, tau, sampler):
    """Two solves (assembly + cached operator), velocity diagnostics, samples."""
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    solver = poisson_solver(space, rho, stabilization=tau)
    try:
        first = solver.solve()
        cp.cuda.runtime.deviceSynchronize()
        first_seconds = time.perf_counter() - started
        # A zero guess times a full solve with the cached operator; the
        # default would warm-start from the converged first solution.
        zero = cp.zeros_like(cp.asarray(hdg.solution_trace(first, space)))
        started = time.perf_counter()
        result = solver.solve(initial_guess=zero)
        cp.cuda.runtime.deviceSynchronize()
        repeat_seconds = time.perf_counter() - started
        velocity = hdg.perpendicular_vector_field(result.flux, 1., space)
        row = dict(tau=tau, first_solve_seconds=first_seconds, repeat_solve_seconds=repeat_seconds,
                   first_iterations=first.global_solve_result.iteration_count,
                   repeat_iterations=result.global_solve_result.iteration_count,
                   repeat_solve_stage_seconds=result.timings.solve)
        row.update(transport_velocity_diagnostics(velocity, backend="device"))
        fields = hdg.guiding_center_field_diagnostics(rho, result.field, result.flux, backend="device")
        row.update(energy=.5*fields["q_l2_standard"]**2, potential_minimum=fields["phi_min"],
                   potential_maximum=fields["phi_max"])
        samples = dict(ux=sampler(velocity.components[0]), uy=sampler(velocity.components[1]),
                       phi=sampler(result.field))
    finally:
        solver.close()
    return row, samples


def compare(samples, reference, masks):
    """Relative L2 (area-uniform points) and max differences on each mask."""
    import cupy as cp

    errors = {}
    for label, mask in masks.items():
        du = (samples["ux"] - reference["ux"])**2 + (samples["uy"] - reference["uy"])**2
        uu = reference["ux"]**2 + reference["uy"]**2
        dphi = (samples["phi"] - reference["phi"])**2
        errors[f"velocity_relative_l2_{label}"] = float(cp.sqrt(cp.sum(du[mask]) / cp.sum(uu[mask])))
        errors[f"velocity_relative_max_{label}"] = float(cp.sqrt(cp.max(du[mask]) / cp.max(uu[mask])))
        errors[f"potential_relative_l2_{label}"] = float(
            cp.sqrt(cp.sum(dphi[mask]) / cp.sum(reference["phi"][mask]**2)))
    return errors


def probe(args):
    """Run every tau against one higher-degree reference and save a JSON table."""
    import cupy as cp

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{args.name}.json"
    if target.exists():
        raise FileExistsError(f"choose a fresh probe name; {target} already exists")
    mesh = showcase_mesh(args.h)
    space = showcase_space(mesh)
    profile = showcase_profile(args.profile, mesh, strength_mode=args.strength_mode,
                               amplitude=args.amplitude, cutoff=args.cutoff)
    rho = hdg.project_callable(profile, space, backend="device")

    lo, hi = np.asarray(mesh.node_coords).min(axis=0), np.asarray(mesh.node_coords).max(axis=0)
    x = np.linspace(lo[0], hi[0], args.points)
    y = np.linspace(lo[1], hi[1], args.points)
    points = np.column_stack([axis.ravel() for axis in np.meshgrid(x, y)])
    sampler = PointSampler(space, points)
    common = sampler.valid
    distance = np.full(len(points), np.inf)
    distance[common] = hdg.MeshDomain(mesh).boundary_distance(points[common])
    masks = {"all": cp.asarray(common), "wall": cp.asarray(common & (distance < args.wall_band)),
             "interior": cp.asarray(common & (distance >= args.wall_band))}

    metadata = dict(vars(args), triangles=mesh.num_tri,
                    common_points=int(common.sum()), wall_points=int((common & (distance < args.wall_band)).sum()),
                    global_length_tau=resolve_tau("global", space), profile_cutoff=profile.cutoff,
                    gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), rows=[])
    reference_tau = resolve_tau(args.reference_tau, space)
    info, samples, valid = host_reference(mesh, profile, points, order=args.reference_order,
                                          tau=reference_tau)
    if not np.array_equal(valid, common):
        raise RuntimeError("reference and probe spaces disagree on point ownership")
    reference = {key: cp.asarray(value) for key, value in samples.items()}
    metadata["reference"] = info
    print(f"reference p={args.reference_order} tau={reference_tau:.4g}: {info['seconds']:.1f}s, "
          f"MKL threads {info['mkl_threads']}, peak CPU cores {info['observed_peak_cpu_cores']:.1f}",
          flush=True)
    for value in args.taus:
        tau = resolve_tau(value, space)
        row, samples = solve_case(space, rho, tau, sampler)
        row.update(compare(samples, reference, masks), tau_label=value, tau_h=tau*args.h)
        metadata["rows"].append(row)
        target.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"tau={tau:9.4g} it={row['repeat_iterations']:4} solve={row['repeat_solve_seconds']:.3f}s "
              f"|u-u_ref|/|u_ref| all={row['velocity_relative_l2_all']:.3e} "
              f"wall={row['velocity_relative_l2_wall']:.3e} "
              f"interior={row['velocity_relative_l2_interior']:.3e} "
              f"div_l2={row['velocity_divergence_l2']:.3e} jump_l2={row['velocity_normal_jump_l2']:.3e} "
              f"double_outflow={row['velocity_double_outflow_measure_fraction']:.3e}", flush=True)
        del samples
        cp.get_default_memory_pool().free_all_blocks()
    target.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", type=float, default=.005)
    parser.add_argument("--reference-order", type=int, default=8)
    parser.add_argument("--reference-tau", default="global")
    parser.add_argument("--profile", required=True,
                        help="saved profile; sampled on the --h mesh and saved there if missing")
    parser.add_argument("--strength-mode", choices=("balanced", "positive"), default="balanced")
    parser.add_argument("--amplitude", type=float, default=4.)
    parser.add_argument("--cutoff", type=float, default=8.)
    parser.add_argument("--taus", nargs="+", default=["1000", "100", "10", "global", "1", "0.1"])
    parser.add_argument("--points", type=int, default=1000, help="shared grid points per axis")
    parser.add_argument("--wall-band", type=float, default=.05)
    parser.add_argument("--output", default="outputs/readme_showcase")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.reference_order <= 6:
        parser.error("reference-order must exceed the showcase degree 6")
    probe(args)


if __name__ == "__main__":
    main()
