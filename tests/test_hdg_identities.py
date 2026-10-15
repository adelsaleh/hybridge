"""Exact SI-Euler L2 budget of one upwind HDG transport solve (D_h, G_h); single linear solves."""
from dataclasses import replace

import numpy as np
import pytest


@pytest.mark.parametrize("tau", [32000.0, 320.0])
def test_si_euler_l2_budget_closes_to_round_off(tau):
    pytest.importorskip("gmsh")
    from hybridge.core.field_ops import (field_linear_combination, perpendicular_vector_field, solution_field,
                                         solution_trace)
    from hybridge.core.mesh import gmsh_disc_mesh
    from hybridge.core.space import DGSpace
    from hybridge.diagnostics.hdg_identities import field_l2_squared, transport_enstrophy_terms
    from hybridge.linalg.pardiso_runtime import pardiso_thread_limit
    from hybridge.solvers.advection_reaction import AdvectionReactionHDGSolver
    from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGSolver
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.configuration import _make_poisson_options, _make_transport_options
    from scripts.guiding_center.runtime.fast_profiles import fastest_host_overrides

    order = 4
    mesh = gmsh_disc_mesh(0.12, log_cache=False)
    space = DGSpace(mesh, order)
    config = replace(preset_by_key("diocotron_gaussian_m64_si_bdf2_p6_h0068_dt05_t400"),
                     **fastest_host_overrides(order), order=order, poisson_tau=tau,
                     transport_advection_stabilization="conflict-averaged-upwind")
    rho0 = space.project_callable(
        lambda x, y: np.exp(-((np.hypot(x, y) - 0.475) / 0.08) ** 2) * (1 + 0.3 * np.cos(3 * np.arctan2(y, x))))
    dt = 0.2
    with pardiso_thread_limit(16):
        poisson = DiffusionReactionHDGSolver(space, source=rho0, reaction=space.zeros(), boundary_condition=0.0,
                                             options=_make_poisson_options(config)).solve()
        transport = AdvectionReactionHDGSolver(space, options=_make_transport_options(config, "zero-flux"))
        transport.set_problem(rho0, perpendicular_vector_field(poisson.flux, dt, space), space.constant(1.0), None)
        result = transport.solve(initial_guess=None)
    rho1 = solution_field(result, space)
    terms = transport_enstrophy_terms(perpendicular_vector_field(poisson.flux, 1.0, space), rho1,
                                      solution_trace(result, space, reduced=True), trace_basis="legacy-lagrange")
    assert terms["dissipation"] >= 0.0
    difference = field_linear_combination(space, [(1.0, rho1), (-1.0, rho0)])
    left = (field_l2_squared(rho1) + field_l2_squared(difference) + 2 * dt * terms["dissipation"]
            + dt * terms["defect"])
    assert left == pytest.approx(field_l2_squared(rho0), rel=1e-13)
