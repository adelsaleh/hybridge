"""Public-library fixtures for assembly and local-solve benchmarks.

The mesh and stabilization preserve the historical transport benchmark inputs.
These workers assemble operators only; application time stepping, AMGX retry
policies and recording workflows are owned by guiding-center-cg-hdg.
"""
import hybridge as hdg


def transport_mesh(mesh_size):
    """Five-lobed star with a circular island used by the assembly baselines."""
    return hdg.gmsh_smooth_star_mesh(
        mesh_size, radius=1., amplitude=.35, mode=5, hole_radius=.3,
        boundary_points=500, num_threads=16, log_cache=False)


def transport_options():
    """Return the numerical-flux policy shared by the measured assembly paths."""
    return dict(advection_stabilization="conflict-averaged-upwind")


def transport_solver(space, **options):
    """Construct the public device solver used to prepare assembly inputs."""
    return hdg.AdvectionReactionHDGSolver(
        space, boundary_mode="zero-flux", solver="amgx", assembly_backend="raw-cuda",
        solver_rtol=1.e-9, solver_atol=1.e-10, verbose=False, **options)
