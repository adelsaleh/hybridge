"""BDF2 turbulence on a star with an island, with live panels or an MP4.

The two initial conditions use the geometry, Gaussian blobs, and time steps
of the README videos. Historical recordings and their full research workflow
are preserved separately; this example demonstrates the numerical API.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--case', choices=('vortex-gas', 'positive-turbulence'), default='vortex-gas')
    command.add_argument('--mesh-size', type=float, default=.005)
    command.add_argument('--dt', type=float, help='Default: .003125 (vortex gas), .000390625 (positive).')
    command.add_argument('--steps', type=int, help='Default: 5080 (vortex gas), 16384 (positive).')
    command.add_argument('--every', type=int, default=16, help='Render every N time steps.')
    command.add_argument('--movie', type=Path, help='Write an MP4 and initial PNG instead of live panels.')
    command.add_argument('--fps', type=float, default=48.)
    return command


def run(args) -> None:
    """Couple reusable Poisson and transport solves with constant-step BDF2."""
    import hybridge as hdg
    from hybridge.io import HolovizScalarPanels
    from hybridge.io.recording import RasterMovieRecorder
    from hybridge.transport.positivity import DensityPositivityProjector

    positive = args.case == 'positive-turbulence'
    dt = args.dt if args.dt is not None else (.000390625 if positive else .003125)
    steps = args.steps if args.steps is not None else (16384 if positive else 5080)
    mesh = hdg.gmsh_smooth_star_mesh(
        args.mesh_size, radius=1., amplitude=.35, mode=5, hole_radius=.3,
        boundary_points=500, num_threads=16)
    space = hdg.DGSpace(mesh, 6, basis_type='dub_orth')
    initial = hdg.sample_gaussian_blob_field(
        hdg.MeshDomain(mesh), counts=(512, 256, 128, 64),
        sigmas=(.008, .016, .024, .032), cutoff=5., seed=17,
        amplitude=4., strength_mode='positive' if positive else 'balanced')
    rho = hdg.project_callable(initial, space, backend='device')
    projector = DensityPositivityProjector(
        space, points='quadrature+lattice+dense', backend='device') if positive else None
    if projector is not None:
        rho, _ = projector.project(rho)
    gpu = dict(assembly_backend='raw-cuda', solver_rtol=1e-9, solver_atol=1e-10, verbose=False)
    one = space.constant(1.)
    limits = ((0., 22.), (0., .17561634457388905)) if positive else ((-18., 18.), (-.07, .07))
    cmap = 'viridis' if positive else 'RdBu_r'
    previous_rho = previous_velocity = None
    with ExitStack() as stack:
        poisson = stack.enter_context(hdg.DiffusionReactionHDGSolver(
            space, source=rho, boundary_condition=0., solver='fb-hp-mg-pcg',
            stabilization='global_length', cache_local_factors='schur-cholesky', **gpu))
        transport = stack.enter_context(hdg.AdvectionReactionHDGSolver(
            space, boundary_mode='zero-flux', solver='amgx',
            advection_stabilization='conflict-averaged-upwind', **gpu))
        if args.movie:
            args.movie.parent.mkdir(parents=True, exist_ok=True)
            plot = stack.enter_context(RasterMovieRecorder(
                space, args.movie, panels=[('Charge density', dict(cmap=cmap)),
                                           ('Potential', dict(cmap=cmap))],
                width=800, height=800, size=(1600, 800), fps=args.fps, boundary_mesh=mesh))
        else:
            plot = stack.enter_context(HolovizScalarPanels(
                (space, space), ('Charge density', 'Potential'), cmap=cmap,
                width=640, height=640, show_mesh=False))

        def render(step, potential):
            if args.movie:
                plot.frame((plot.sample(rho), plot.sample(potential.field)), limits=limits,
                           caption=f't = {step * dt:.6g}')
            else:
                plot.update_fields((rho, potential.field), limits=limits, step=step, time_value=step*dt)

        potential = poisson.solve()
        velocity = hdg.perpendicular_vector_field(potential.flux)
        render(0, potential)
        for step in range(1, steps + 1):
            if previous_rho is None:  # The one Euler startup step needed by BDF2.
                source, beta = rho, dt * velocity
            else:
                source = (4*rho - previous_rho)/3
                beta = (2*dt/3) * (2*velocity - previous_velocity)
            next_rho = transport.solve(source=source, beta=beta, reaction=one).field
            if projector is not None:
                next_rho, _ = projector.project(next_rho)
            potential = poisson.set_source(next_rho).solve()
            previous_rho, previous_velocity, rho = rho, velocity, next_rho
            velocity = hdg.perpendicular_vector_field(potential.flux)
            if step % args.every == 0 or step == steps:
                render(step, potential)
                print(f'step {step}/{steps}, t={step*dt:.6g}', flush=True)


def main() -> None:
    command = parser()
    args = command.parse_args()
    for name in ('mesh_size', 'dt', 'fps'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            command.error(f'{name.replace("_", "-")} must be finite and positive')
    if args.every < 1 or (args.steps is not None and args.steps < 1):
        command.error('steps and every must be positive integers')
    if args.movie and args.movie.suffix.lower() != '.mp4':
        command.error('movie must have an .mp4 extension')
    run(args)


if __name__ == '__main__':
    main()
