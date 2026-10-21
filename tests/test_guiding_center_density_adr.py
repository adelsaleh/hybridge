"""D7 pieces (collisional parallel-vane magnetron): mesh, case options and the ADR density adapter.

Host only: a coarse Gmsh mesh, small Numba ADR solves with a sparse direct solver.
"""
import math
from collections import Counter

import numpy as np
import pytest

from hybridge import (AdvectionDiffusionReactionHDGOptions, AdvectionDiffusionReactionHDGSolver, DGSpace,
                      rectangle_mesh)
from hybridge.core.mesh import gmsh_vane_anode_mesh, vane_anode_corners, vane_tip_fillets
from scripts.guiding_center.cases.guiding_center_cases import magnetron, pulse_train
from scripts.guiding_center.runtime.density_adr import (ADRDensitySolver, collisional_parameters,
                                                        make_adr_density_solver)

PITCH = 2.0 * math.pi / 12


def test_vane_corners_radial_unchanged_parallel_at_half_width():
    radial = vane_anode_corners(anode_radius=1.0, vanes=12, vane_fraction=0.5)
    angles = np.array([PITCH * k + side * math.pi * 0.5 / 12 for k in range(12) for side in (-1, 1)])
    np.testing.assert_allclose(radial, np.stack((np.cos(angles), np.sin(angles)), axis=1))
    width = 0.2
    corners = vane_anode_corners(anode_radius=1.0, vanes=12, vane_fraction=0.5, vane_width=width)
    radius = np.hypot(*corners.T)
    axis = np.repeat(PITCH * np.arange(12), 2)
    np.testing.assert_allclose(radius, 1.0)
    np.testing.assert_allclose(np.abs(radius * np.sin(np.arctan2(corners[:, 1], corners[:, 0]) - axis)), width / 2,
                               atol=1e-12)


def test_parallel_vane_mesh_has_parallel_sides():
    width = 0.26
    mesh = gmsh_vane_anode_mesh(0.08, vane_width=width, cache=False, log_cache=False)
    points, triangles = np.asarray(mesh.node_coords), np.asarray(mesh.triangles)[:, :3]
    count = Counter(tuple(sorted((int(t[i]), int(t[(i + 1) % 3])))) for t in triangles for i in range(3))
    boundary = np.array([edge for edge, n in count.items() if n == 1])
    middle = 0.5 * (points[boundary[:, 0]] + points[boundary[:, 1]])
    radius = np.hypot(*middle.T)
    side = (radius > 1.0 + 1e-3) & (radius < 1.35 - 1e-3)
    theta = np.arctan2(middle[side, 1], middle[side, 0])
    axis = np.rint(theta / PITCH) * PITCH
    assert side.sum() >= 48
    np.testing.assert_allclose(np.abs(radius[side] * np.sin(theta - axis)), width / 2, atol=1e-9)
    with pytest.raises(ValueError):
        gmsh_vane_anode_mesh(0.08, vane_width=0.6, cache=False, log_cache=False)  # neighbouring tips would touch


def test_rounded_vane_tips_follow_the_fillet_circles():
    width, radius = 0.26, 0.04
    centres = vane_tip_fillets(anode_radius=1.0, vanes=12, vane_width=width, tip_radius=radius)
    sharp = vane_anode_corners(anode_radius=1.0, vanes=12, vane_fraction=0.5, vane_width=width)
    np.testing.assert_allclose(np.hypot(*centres.T), 1.0 + radius)  # tangent to the tip arc
    axis = np.repeat(PITCH * np.arange(12), 2)
    offset = np.hypot(*centres.T) * np.sin(np.arctan2(centres[:, 1], centres[:, 0]) - axis)
    np.testing.assert_allclose(np.abs(offset), width / 2 - radius, atol=1e-12)  # tangent to the side
    assert np.all(np.sign(offset) == np.sign(np.sin(np.arctan2(sharp[:, 1], sharp[:, 0]) - axis)))  # corner order
    mesh = gmsh_vane_anode_mesh(0.06, vane_width=width, tip_radius=radius, corner_mesh_size=0.01,
                                cache=False, log_cache=False)
    distance = np.linalg.norm(np.asarray(mesh.node_coords)[:, None, :2] - centres[None], axis=2)
    assert distance.min() > radius - 1e-9  # no node inside a fillet disk
    assert np.all((np.abs(distance - radius) < 1e-9).sum(axis=0) >= 5)  # every fillet arc is resolved
    with pytest.raises(ValueError):
        gmsh_vane_anode_mesh(0.06, tip_radius=radius, cache=False, log_cache=False)  # radial vanes
    with pytest.raises(ValueError):
        vane_tip_fillets(anode_radius=1.0, vanes=12, vane_width=width, tip_radius=width / 2)


def test_cathode_grading_refines_the_cathode_layer():
    def cathode_edges(mesh):
        nodes, triangles = np.asarray(mesh.node_coords)[:, :2], np.asarray(mesh.triangles)[:, :3]
        near = np.hypot(*nodes[triangles].mean(axis=1).T) < 0.51
        return np.linalg.norm(nodes[triangles[near, 0]] - nodes[triangles[near, 1]], axis=1).mean()

    plain = gmsh_vane_anode_mesh(0.06, vane_width=0.26, cache=False, log_cache=False)
    graded = gmsh_vane_anode_mesh(0.06, vane_width=0.26, cathode_mesh_size=0.006, cache=False, log_cache=False)
    assert graded.num_tri > plain.num_tri
    assert cathode_edges(graded) < 0.5 * cathode_edges(plain)
    case = magnetron(anode_geometry="vanes", wave_amplitude=0.05, vane_width=0.26, cathode_mesh_size=0.006)
    assert case.parameters["geometry"]["cathode_mesh_size"] == 0.006
    assert "cathode_mesh_size" not in magnetron(anode_geometry="vanes", wave_amplitude=0.05).parameters["geometry"]


def test_emission_phase_shifts_the_pulse_and_the_cathode_data():
    common = dict(anode_geometry="vanes", wave_amplitude=0.05, vane_width=0.26, emission_top=0.6, hub_edge=0.025,
                  hall_parameter=30.0, electron_temperature=0.01, emission_period=20.0)
    plain, shifted = magnetron(**common), magnetron(**common, emission_phase=5.0)
    assert "emission_phase" not in plain.parameters and shifted.parameters["emission_phase"] == 5.0
    for t in (0.0, 3.0, 4.5, 7.0, 15.5):
        assert shifted.density_reaction_rate(t) == pytest.approx(plain.density_reaction_rate(t + 5.0))
    cathode = (np.array([0.5]), np.array([0.0]))
    assert plain.density_boundary(*cathode, 0.0)[0] == 0.0  # the plain train starts with the emission off
    assert shifted.density_boundary(*cathode, 0.0)[0] == pytest.approx(1.0)  # matches the initial hub
    with pytest.raises(ValueError):
        magnetron(anode_geometry="vanes", wave_amplitude=0.05, emission_phase=5.0)  # no pulse train


def test_smooth_emission_profile_and_tip_radius_options():
    common = dict(anode_geometry="vanes", wave_amplitude=0.05, emission_top=0.6, hub_edge=0.025, vane_width=0.26)
    cos2, smooth = magnetron(**common), magnetron(**common, emission_profile="smooth", tip_radius=0.04)
    assert "emission_profile" not in cos2.parameters and "tip_radius" not in cos2.parameters["geometry"]
    assert smooth.parameters["emission_profile"] == "smooth" and smooth.parameters["geometry"]["tip_radius"] == 0.04
    r = np.linspace(0.5, 0.65, 3001)
    a, b = cos2.density_source_profile(r, 0 * r), smooth.density_source_profile(r, 0 * r)
    np.testing.assert_allclose(b[r <= 0.55], 1.0)
    assert np.all(b[r >= 0.6] == 0.0) and np.all(np.diff(b) <= 1e-15)
    second = lambda f: np.diff(f, 2) / (r[1] - r[0]) ** 2  # noqa: E731
    i = np.searchsorted(r, 0.55)
    assert abs(second(a)[i + 2] - second(a)[i - 3]) > 100.0  # the cos^2 taper is only C^1
    assert abs(second(b)[i + 2] - second(b)[i - 3]) < 1e-6
    with pytest.raises(ValueError):
        magnetron(anode_geometry="vanes", wave_amplitude=0.05, tip_radius=0.04)  # radial vanes
    with pytest.raises(ValueError):
        magnetron(anode_geometry="vanes", wave_amplitude=0.05, emission_profile="smooth")  # no emission layer
    with pytest.raises(ValueError):
        magnetron(**common, emission_profile="gaussian")


def test_magnetron_defaults_unchanged():
    case = magnetron(anode_geometry="vanes", wave_amplitude=0.05)
    for key in ("collisional", "vane_width", "emission_period"):
        assert key not in case.parameters
    assert "vane_width" not in case.parameters["geometry"]
    assert case.density_reaction_rate is None
    assert collisional_parameters(case) is None


def test_pulse_train():
    rate = pulse_train(20.0, 0.5)  # 10 on, sin^2 ramps of 1
    assert [rate(t) for t in (0.0, 1.0, 5.0, 9.0, 10.0, 15.0, 20.0)] == pytest.approx([0, 1, 1, 1, 0, 0, 0])
    assert rate(0.5) == pytest.approx(0.5) and rate(9.5) == pytest.approx(0.5) and rate(40.5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pulse_train(10.0, 0.1, ramp=1.0)  # two ramps longer than the pulse


def test_collisional_parallel_vane_case():
    width, hall, theta = 0.26, 30.0, 0.01
    case = magnetron(anode_geometry="vanes", wave_amplitude=0.05, emission_top=0.6, vane_width=width,
                     hall_parameter=hall, electron_temperature=theta, emission_period=20.0)
    assert collisional_parameters(case) == {"hall_parameter": hall, "diffusion": pytest.approx(theta / hall)}
    assert case.parameters["geometry"]["vane_width"] == width
    assert case.density_reaction_rate(5.0) == 1.0 and case.density_reaction_rate(15.0) == 0.0
    pattern = case.wall_potentials[1][0]  # pi mode: +1 on vane 0 (tip, side), 0 mid back wall, -1 on vane 1
    x = np.array([1.0, 1.2, 1.35 * math.cos(PITCH / 2), 1.1 * math.cos(PITCH), 0.5])
    y = np.array([0.0, 0.13, 1.35 * math.sin(PITCH / 2), 1.1 * math.sin(PITCH), 0.0])
    np.testing.assert_allclose(pattern(x, y), [1.0, 1.0, 0.0, -1.0, 0.0], atol=1e-12)
    with pytest.raises(ValueError):
        magnetron(vane_width=0.2)  # circular anode
    with pytest.raises(ValueError):
        magnetron(anode_geometry="vanes", wave_amplitude=0.05, hall_parameter=30.0)  # no temperature


class _Config:
    def __init__(self, scheme="si-bdf2", backend="numba", solver="pypardiso"):
        self.time_scheme, self.transport_assembly_backend, self.transport_solver = scheme, backend, solver


def _collisional_case():
    return magnetron(anode_geometry="vanes", wave_amplitude=0.05, hall_parameter=20.0, electron_temperature=0.02)


def test_make_adr_density_solver_selection():
    space = DGSpace(rectangle_mesh(2, 2, xlim=(0.0, 1.0), ylim=(0.0, 1.0)), 1)
    assert make_adr_density_solver(space, magnetron(), _Config()) is None
    solver = make_adr_density_solver(space, _collisional_case(), _Config())
    assert solver.options.solver == "pypardiso" and solver.options.assembly_backend == "numba"
    assert solver.options.pardiso_reuse_analysis and solver.options.pardiso_threads == "all"
    assert solver.options.diffusion == pytest.approx(0.02 / 20.0)
    with pytest.raises(ValueError):
        make_adr_density_solver(space, _collisional_case(), _Config(scheme="imex-ark3"))
    with pytest.raises(ValueError):
        make_adr_density_solver(space, _collisional_case(), _Config(backend="raw-cuda", solver="pypardiso"))


def test_adapter_drift_adds_the_collisional_drift():
    space = DGSpace(rectangle_mesh(3, 3, xlim=(0.0, 1.0), ylim=(0.0, 1.0)), 2)
    s, h, (qx, qy) = 0.3, 4.0, (0.7, -0.4)
    beta = (space * space).field((space.constant(-s * qy), space.constant(s * qx)))  # s q^perp
    solver = ADRDensitySolver(space, hall_parameter=h, diffusion=1e-3)
    solver.set_stage_scale(s)
    drift = solver._drift(beta)  # q^perp + q/h
    np.testing.assert_allclose(drift.components[0].values(), -qy + qx / h, atol=1e-12)
    np.testing.assert_allclose(drift.components[1].values(), qx + qy / h, atol=1e-12)


def test_adapter_stage_equals_the_transport_scaled_adr_solve():
    """Dividing a stage by s leaves the HDG solution unchanged (the stabilization scales with beta and D)."""
    space = DGSpace(rectangle_mesh(4, 4, xlim=(0.0, 1.0), ylim=(0.0, 1.0)), 3)
    s, h, diffusion, alpha, (qx, qy) = 0.05, 10.0, 2e-3, 0.4, (0.8, 0.3)
    common = dict(hdg_postprocess="none", verbose=0, assembly_backend="numba", solver="direct")
    source = space.project_callable(lambda x, y: 1.0 + x * y)
    reaction = space.constant(1.0 - s * alpha)

    def boundary(x, y):
        return 0.0 * x

    adapter = ADRDensitySolver(space, hall_parameter=h, diffusion=diffusion,
                               options=AdvectionDiffusionReactionHDGOptions(**common))
    adapter.set_stage_scale(s)
    adapter.set_problem(source, (space * space).field((space.constant(-s * qy), space.constant(s * qx))),
                        reaction, boundary)
    got = adapter.solve().field
    u = (-qy + qx / h, qx + qy / h)
    direct = AdvectionDiffusionReactionHDGSolver(
        space, options=AdvectionDiffusionReactionHDGOptions(diffusion=s * diffusion, **common))
    direct.set_problem(source, (space * space).field((space.constant(s * u[0]), space.constant(s * u[1]))),
                       reaction, boundary)
    want = direct.solve().field
    np.testing.assert_allclose(got.values(), want.values(), rtol=1e-9, atol=1e-11)


def test_boundary_outflux_closes_the_stage_charge_balance():
    """Testing a stage with 1: int((1 - s alpha)/s) rho - int(f)/s + outflux = 0 to round-off."""
    space = DGSpace(rectangle_mesh(4, 4, xlim=(0.0, 1.0), ylim=(0.0, 1.0)), 3)
    s, h, diffusion, alpha, (qx, qy) = 0.05, 10.0, 2e-3, 0.4, (0.8, 0.3)
    source = space.project_callable(lambda x, y: 1.0 + x * y)
    reaction = space.constant(1.0 - s * alpha)
    adapter = ADRDensitySolver(space, hall_parameter=h, diffusion=diffusion,
                               options=AdvectionDiffusionReactionHDGOptions(
                                   hdg_postprocess="none", verbose=0, assembly_backend="numba", solver="direct"))
    adapter.set_stage_scale(s)
    adapter.set_problem(source, (space * space).field((space.constant(-s * qy), space.constant(s * qx))),
                        reaction, lambda x, y: 0.0 * x)
    rho = adapter.solve().field
    flux = adapter.boundary_outflux(groups={"left": lambda x, y: x < 1e-12})
    volume = ((1.0 - s * alpha) / s) * rho.integral() - source.integral() / s
    assert abs(volume + flux["outflux"]) <= 1e-10 * max(abs(volume), 1.0)
    assert flux["outflux"] == pytest.approx(flux["outflux_advective"] + flux["outflux_diffusive"])
    assert flux["outflux_diffusive"] != 0.0 and "outflux_left" in flux
