"""Guiding-center magnetron: case data, electrode superposition, wall outflux and charge balance.

Single Poisson and transport solves on a coarse annulus (host Numba + PyPardiso,
optionally the raw-CUDA device path); no time integration.
"""
import math

import numpy as np
import pytest

from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key, magnetron


def _cuda_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    return cp


def test_magnetron_wall_potential_emission_and_registry():
    case = case_definition_by_key("magnetron").build(wave_amplitude=0.03, wave_ramp=0.0, wave_speed=0.29)
    assert case.key == "magnetron" and case.default_domain == "annulus"
    assert case.density_transport_boundary_mode == "eliminate" and case.wall_flux_diagnostics
    t, theta = 3.7, np.linspace(0.0, 2.0 * np.pi, 13)
    anode = case.wall_potential_at(t)(np.cos(theta), np.sin(theta))
    assert np.allclose(anode, -0.1 + 0.03 * np.cos(6 * (theta - 0.29 * t)), atol=1e-14)
    assert np.allclose(case.wall_potential_at(t)(0.5 * np.cos(theta), 0.5 * np.sin(theta)), 0.0)
    boundary = case.potential_boundary_at(t)
    assert boundary._hybridge_time == t and boundary._hybridge_constant_value == 0.0
    # Emission kappa chi (rho_c - rho): source and reaction cancel where rho = rho_c (= 1).
    x, y = np.array([0.55, 0.65, 0.9]), np.zeros(3)
    hub = case.initial_density(x, y)
    assert hub[0] == pytest.approx(1.0, abs=1e-6) and hub[1] == pytest.approx(0.5) and hub[2] < 1e-6
    assert np.allclose(case.density_source(x, y, t) + case.density_reaction_profile(x, y), 0.0, atol=1e-15)
    unforced = magnetron(wave_amplitude=0.0)
    assert len(unforced.wall_potentials) == 1  # DC voltage only
    layer = magnetron(emission_top=0.55)  # emission confined to a cathode layer, hub unchanged
    assert layer.density_source(np.array([0.62]), np.array([0.0]), 0.0)[0] < 1e-2
    assert layer.initial_density(np.array([0.62]), np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-2)
    with pytest.raises(ValueError):
        magnetron(emission_top=0.7)


def test_magnetron_hub_reference():
    from scripts.guiding_center.diagnostics.magnetron_reference import HubEquilibrium

    hub = HubEquilibrium(magnetron().parameters)
    # The hub attached to the cathode has a monotone profile: no unstable free mode.
    assert max(hub.growth_rates(range(1, 9)).values()) < 1e-10
    # Charge and drift: Q(r_a) = 2 pi int rho r dr, and phi(r_a) = -V.
    expected = 2 * math.pi * 0.5 * (0.65 ** 2 - 0.5 ** 2)
    assert hub.Q[-1] == pytest.approx(expected, rel=2e-3)
    assert hub.phi[-1] == pytest.approx(-0.1, abs=1e-6)
    window = hub.synchronism_window(0.025)
    assert window is not None and window[0] < hub.edge_wave_speed() < window[1]
    assert hub.synchronism_window(0.005) is None


def _setup(backend):
    from dataclasses import replace

    pytest.importorskip("gmsh")
    from hybridge.core.mesh import gmsh_annulus_mesh
    from hybridge.core.space import DGSpace
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.fast_profiles import fastest_device_overrides, fastest_host_overrides

    case = magnetron(wave_amplitude=0.03, wave_ramp=0.0)
    mesh = gmsh_annulus_mesh(0.1, inner_radius=0.5, outer_radius=1.0, log_cache=False)
    order = 4
    space = DGSpace(mesh, order)
    overrides = (fastest_host_overrides(order) if backend == "host"
                 else fastest_device_overrides(order, zero_dirichlet=True, triangles=mesh.num_tri))
    config = replace(preset_by_key("diocotron_gaussian_m64_si_bdf2_p6_h0068_dt05_t400"), **overrides,
                     order=order, poisson_tau=32000.0, transport_advection_stabilization="conflict-averaged-upwind")
    return case, space, config


def _electrode_solver(case, space, options, rho):
    from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGSolver
    from scripts.guiding_center.runtime.electrodes import ElectrodePoissonSolver

    zero = space.zeros(name="zero")
    inner = DiffusionReactionHDGSolver(space, source=rho, reaction=zero,
                                       boundary_condition=case.potential_boundary_at(0.0), options=options)
    return ElectrodePoissonSolver(inner, case, lambda boundary, opts: DiffusionReactionHDGSolver(
        space, source=space.zeros(name="electrode_source"), reaction=zero, boundary_condition=boundary,
        options=opts))


@pytest.mark.parametrize("backend", ["host", "device"])
def test_electrode_superposition_matches_direct_nonzero_dirichlet_solve(backend):
    if backend == "device":
        _cuda_or_skip()
    from hybridge.linalg.pardiso_runtime import pardiso_thread_limit
    from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGSolver
    from scripts.guiding_center.runtime.configuration import _make_poisson_options

    case, space, config = _setup(backend)
    options = _make_poisson_options(config)
    rho = space.project_callable(case.initial_density, name="rho")
    t = 3.7
    with pardiso_thread_limit(16):
        direct = DiffusionReactionHDGSolver(space, source=rho, reaction=space.zeros(name="zero"),
                                            boundary_condition=case.wall_potential_at(t), options=options).solve()
        solver = _electrode_solver(case, space, options, rho)
        solver.solve()
        solver.set_boundary_condition(case.potential_boundary_at(t))
        superposed = solver.solve()
    tolerance = 1e-12 if backend == "host" else 1e-8
    assert np.abs(superposed.field.coeffs - direct.field.coeffs).max() < tolerance
    for a, b in zip(superposed.flux.components, direct.flux.components):
        assert np.abs(a.coeffs - b.coeffs).max() < 100 * tolerance
    assert solver.boundary_condition(np.array([1.0]), np.array([0.0]))[0] == pytest.approx(
        -0.1 + 0.03 * math.cos(6 * (0.0 - 0.29 * t)) * 1.0, abs=1e-14)


@pytest.mark.parametrize("backend", ["host", "device"])
@pytest.mark.parametrize("drift", ["raw", "compatible"])
def test_one_transport_solve_satisfies_the_open_wall_charge_balance(backend, drift):
    if backend == "device":
        _cuda_or_skip()
    from hybridge.core.field_ops import perpendicular_vector_field, solution_field
    from hybridge.linalg.pardiso_runtime import pardiso_thread_limit
    from hybridge.solvers.advection_reaction import AdvectionReactionHDGSolver
    from scripts.guiding_center.runtime.compatible_drift import CompatibleFluxPoissonSolver
    from scripts.guiding_center.runtime.configuration import _make_poisson_options, _make_transport_options
    from scripts.guiding_center.runtime.diagnostics import WallChargeBalance
    from scripts.guiding_center.runtime.forcing import DensityForcing

    case, space, config = _setup(backend)
    rho = space.project_callable(case.initial_density, name="rho")
    t, dt = 3.7, 0.1
    with pardiso_thread_limit(16):
        solver = _electrode_solver(case, space, _make_poisson_options(config), rho)
        if drift == "compatible":
            solver = CompatibleFluxPoissonSolver(solver, space, potential="primal")
        solver.set_boundary_condition(case.potential_boundary_at(t))
        beta = perpendicular_vector_field(solver.solve().flux, dt, space)
        forcing = DensityForcing(space, case)
        source = forcing.stage_source(rho, dt, t + dt)
        transport = AdvectionReactionHDGSolver(space, options=_make_transport_options(config, "eliminate"))
        transport.set_problem(source, beta, forcing.reaction(dt, space.constant(1.0), t + dt, like=source),
                              case.density_boundary_at(t + dt))
        new = solution_field(transport.solve(initial_guess=None), space)
    balance = WallChargeBalance(case, "si-euler", "legacy-lagrange")
    flux = balance.flux(beta, new, dt)
    rows = [dict(mass=rho.integral()),
            dict(mass=new.integral(), transport_beta_scale=dt, wall_outflux=flux["wall_outflux"],
                 source_production=forcing.source_production(t + dt), reaction_production=forcing.production(new, t + dt))]
    balance.balance(0, rows[0])
    defect = balance.balance(1, rows[1])["charge_balance_defect"]
    assert abs(defect) < 5e-14
    assert flux["wall_outflux"] == pytest.approx(flux["wall_outflux_inner"] + flux["wall_outflux_outer"], abs=1e-15)
    if drift == "compatible":  # the Oswald potential takes the wall data: no flux through the cathode
        assert abs(flux["wall_outflux_inner"]) < 1e-13


def test_boundary_upwind_outflux_device_matches_host():
    cp = _cuda_or_skip()
    pytest.importorskip("gmsh")
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space
    from hybridge.core.mesh import gmsh_annulus_mesh
    from hybridge.core.space import DGSpace, VectorDGField
    from hybridge.diagnostics.guiding_center import boundary_upwind_outflux

    mesh = gmsh_annulus_mesh(0.1, inner_radius=0.5, outer_radius=1.0, log_cache=False)
    space = DGSpace(mesh, 3)
    velocity = VectorDGField((space.project_callable(lambda x, y: x - 0.3 * y),
                              space.project_callable(lambda x, y: 0.2 + x * y)))
    rho = space.project_callable(lambda x, y: 1.0 + x * x)
    groups = {"inner": lambda x, y: x * x + y * y < 0.5625, "outer": lambda x, y: x * x + y * y >= 0.5625}
    host = boundary_upwind_outflux(velocity, rho, groups=groups, backend="host")
    cspace = as_cupy_space(space, device=0)
    for field in (*velocity.components, rho):
        as_cupy_coefficients(field, cspace)
    device = boundary_upwind_outflux(velocity, rho, groups=groups, backend="device")
    assert set(host) == set(device)
    for key in host:
        assert device[key] == pytest.approx(host[key], rel=1e-12, abs=1e-14)
    assert host["outflux"] == pytest.approx(host["outflux_inner"] + host["outflux_outer"], rel=1e-12)
    del cp


def test_vane_anode_pi_mode_wall_potential_and_mesh():
    pytest.importorskip("gmsh")
    from hybridge.core.mesh import gmsh_vane_anode_mesh, vane_anode_corners

    case = magnetron(anode_geometry="vanes", wave_amplitude=0.05, wave_ramp=0.0, wave_speed=0.29)
    assert case.default_domain == "vane-anode" and case.parameters["geometry"]["vanes"] == 12
    t = 1.3
    wall = case.wall_potential_at(t)
    drive = 0.05 * math.cos(6 * 0.29 * t)
    theta = np.radians([0.0, 7.5, 15.0, 22.5, 30.0, 180.0])
    for radius in (1.0, 1.35):  # vane tips/sides and slot back walls
        values = wall(radius * np.cos(theta), radius * np.sin(theta))
        assert np.allclose(values, -0.1 + drive * np.array([1.0, 1.0, 0.0, -1.0, -1.0, 1.0]), atol=1e-12)
    assert np.allclose(wall(0.5 * np.cos(theta), 0.5 * np.sin(theta)), 0.0)
    corners = vane_anode_corners(anode_radius=1.0, vanes=12, vane_fraction=0.5)
    assert corners.shape == (24, 2) and np.allclose(np.hypot(*corners.T), 1.0)
    mesh = gmsh_vane_anode_mesh(0.1, corner_mesh_size=0.03, log_cache=False)
    radius = np.hypot(mesh.node_coords[:, 0], mesh.node_coords[:, 1])
    assert radius.min() == pytest.approx(0.5, abs=1e-9) and radius.max() == pytest.approx(1.35, abs=1e-9)
    # every corner is a mesh vertex
    distance = np.hypot(mesh.node_coords[:, None, 0] - corners[None, :, 0], mesh.node_coords[:, None, 1] - corners[None, :, 1])
    assert distance.min(axis=0).max() < 1e-9


def test_wall_charge_balance_uses_the_bdf3_history():
    from scripts.guiding_center.runtime.diagnostics import WallChargeBalance

    balance = WallChargeBalance(magnetron(), "si-bdf3", "legacy-lagrange")
    dt, masses = 0.1, [1.0, 1.1, 1.25, 1.3, 1.42]
    production = 0.7  # int S + int alpha rho - I, chosen per step below
    assert balance.balance(0, dict(mass=masses[0])) == {}
    # Richardson SI-Euler start-up: three solves, no single-solve identity.
    assert balance.balance(1, dict(mass=masses[1], bdf3_startup_method="si-euler-extrap2")) == {}
    rows = []
    for step, order, scale in ((2, 2, 2 * dt / 3), (3, 3, 6 * dt / 11), (4, 3, 6 * dt / 11)):
        weights = {2: (4 / 3, -1 / 3), 3: (18 / 11, -9 / 11, 2 / 11)}[order]
        history = sum(w * masses[step - 1 - k] for k, w in enumerate(weights))
        source = (masses[step] - history) / scale - production
        row = dict(mass=masses[step], transport_beta_scale=scale, source_production=source,
                   reaction_production=production, wall_outflux=0.0,
                   bdf3_startup_method="si-bdf2" if order == 2 else None)
        rows.append(balance.balance(step, row))
    assert all(abs(r["charge_balance_defect"]) < 1e-14 for r in rows)
