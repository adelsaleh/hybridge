"""GPU vortex gas: mesh, reusable HDG solves, and two live scalar panels.

Run from a configured checkout. Recording and measured validation are documented
in docs/getting_started/gpu_showcase.md.
"""


def main():
    """Compose Poisson and transport solvers with constant-step BDF2."""
    # README example begins
    from contextlib import closing
    import hdgfem as hdg
    from hdgfem.io import HolovizScalarPanels

    # Mesh and degree-6 approximation.
    mesh = hdg.gmsh_smooth_star_mesh(
        0.005, radius=1., amplitude=.35, mode=5,              # Decrease h to refine.
        hole_radius=.3, boundary_points=500, num_threads=16)  # Circular island.
    space = hdg.DGSpace(mesh, 6, basis_type="dub_orth")

    # Multiscale vortices: the counts and widths control the initial structure.
    # For positive density, set strength_mode="positive" and dt=0.0015625.
    # Unlimited high-order HDG can produce negative undershoots.
    initial = hdg.sample_gaussian_blob_field(
        hdg.MeshDomain(mesh),                               # Respect both walls.
        (512, 256, 128, 64), (.008, .016, .024, .032),        # Counts and widths.
        seed=17, strength_mode="balanced")                  # Reproducible profile.
    rho = hdg.project_callable(initial, space, backend="device")

    # Reuse the Poisson operator; rebuild transport as the velocity changes.
    gpu = dict(assembly_backend="raw-cuda", raw_matrix_format="bsr",
               solver_rtol=1e-9, solver_atol=1e-10, verbose=False)
    poisson = hdg.DiffusionReactionHDGSolver(
        space, source=rho, reaction=space.zeros(),
        boundary_condition=0.,                             # Zero wall potential.
        solver="fb-hp-mg-pcg", trace_basis="legendre-modal", scale_system=False,
        stabilization=1000., boundary_mode="eliminate",
        cache_local_factors="schur-cholesky", **gpu)
    transport = hdg.AdvectionReactionHDGSolver(
        space, solver="amgx", boundary_mode="zero-flux",     # Impermeable walls.
        raw_local_assembly="fused", materialize_host_solution=False,
        scale_system=True, **gpu)

    dt, steps = 0.00625, 1200                               # Final time: steps*dt.
    previous_rho = previous_velocity = trace = None         # Euler startup, then BDF2.

    # Fixed scales for the live GPU view; recordings use Matplotlib colorbars.
    # For positive density, use limits=((-5.75, 23.), (0., .25)).
    limits = ((-18., 18.), (-.066, .066))                    # Density, potential.
    with closing(poisson), closing(transport), HolovizScalarPanels(
        (space, space), ("Vorticity / density", "Potential"), cmap="RdBu_r",
        width=640, height=640, show_mesh=False) as plot:
        potential = poisson.solve()                        # Recover phi from rho.
        velocity = hdg.perpendicular_vector_field(          # u=(-q_y,q_x), q=-grad(phi).
            potential.flux, 1., space)
        plot.update_fields((rho, potential.field), limits=limits)

        for step in range(1, steps + 1):
            source, beta, _ = hdg.bdf2_transport_data(       # History and velocity extrapolation.
                rho, velocity, dt,
                previous_field=previous_rho, previous_velocity=previous_velocity)
            result = transport.solve(
                source=source, beta=beta, reaction=space.constant(1.),
                initial_guess=trace)                       # Reuse the transport trace.

            next_rho = hdg.solution_field(result, space).copy()  # Own reused-buffer data.
            potential = poisson.set_source(next_rho).solve(
                initial_guess=hdg.solution_trace(potential, space))

            # Commit history only after both solves succeed.
            previous_rho, previous_velocity, rho = rho, velocity, next_rho
            velocity = hdg.perpendicular_vector_field(potential.flux, 1., space)
            trace = hdg.solution_trace(result, space).copy()
            if step % 2 == 0:                               # Draw every other step.
                plot.update_fields((rho, potential.field), step=step,
                                   time_value=step * dt, limits=limits)
    # README example ends


if __name__ == "__main__":
    main()
