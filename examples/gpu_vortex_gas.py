"""GPU vortex gas: a guiding-center plasma evolved by two reusable HDG solves.

Signed charge between grounded walls drifts like a two-dimensional vortex
gas. Run from a configured checkout; docs/getting_started/gpu_showcase.md
records the published run and its checks.
"""


def main():
    """Couple the Poisson and transport solvers with constant-step BDF2."""
    # README example begins
    import hybridge as hdg
    from hybridge.io import HolovizScalarPanels

    # A five-lobed star around a circular island, with degree-6 elements.
    mesh = hdg.gmsh_smooth_star_mesh(
        0.005, radius=1., amplitude=.35, mode=5,               # Decrease h to refine.
        hole_radius=.3, boundary_points=500, num_threads=16)
    space = hdg.DGSpace(mesh, 6, basis_type="dub_orth")

    # 960 Gaussian charges of both signs, starting as close as five widths to a wall.
    initial = hdg.sample_gaussian_blob_field(
        hdg.MeshDomain(mesh), counts=(512, 256, 128, 64),
        sigmas=(.008, .016, .024, .032), cutoff=5., seed=17, strength_mode="balanced")
    rho = hdg.project_callable(initial, space, backend="device")

    gpu = dict(assembly_backend="raw-cuda", solver_rtol=1e-9, solver_atol=1e-10,
               verbose=False)
    # Poisson: -Δφ = ρ with φ = 0 on both walls. Transport: no flux through them.
    poisson = hdg.DiffusionReactionHDGSolver(
        space, source=rho, boundary_condition=0., solver="fb-hp-mg-pcg",
        cache_local_factors="schur-cholesky", **gpu)
    transport = hdg.AdvectionReactionHDGSolver(
        space, boundary_mode="zero-flux", solver="amgx", **gpu)
    one = space.constant(1.)                                   # BDF2 as ρ + ∇·(βρ) = s.

    dt, steps = 0.003125, 5080                                 # Final time 15.875.
    previous_rho = previous_velocity = None                    # Euler startup, then BDF2.
    limits = ((-18., 18.), (-.07, .07))                        # Fixed color scales.
    with poisson, transport, HolovizScalarPanels(
            (space, space), ("Charge density", "Potential"), cmap="RdBu_r",
            width=640, height=640, show_mesh=False) as plot:
        potential = poisson.solve()
        velocity = hdg.perpendicular_vector_field(potential.flux)  # u = (-q_y, q_x).
        plot.update_fields((rho, potential.field), limits=limits)

        for step in range(1, steps + 1):
            source, beta, _ = hdg.bdf2_transport_data(
                rho, velocity, dt,
                previous_field=previous_rho, previous_velocity=previous_velocity)
            next_rho = transport.solve(source=source, beta=beta, reaction=one).field
            potential = poisson.set_source(next_rho).solve()       # Both solves warm-start.

            # Commit history only after both solves succeed.
            previous_rho, previous_velocity, rho = rho, velocity, next_rho
            velocity = hdg.perpendicular_vector_field(potential.flux)
            if step % 4 == 0:                                      # Draw every fourth step.
                plot.update_fields((rho, potential.field), step=step,
                                   time_value=step * dt, limits=limits)
    # README example ends


if __name__ == "__main__":
    main()
