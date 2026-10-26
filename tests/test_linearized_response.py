"""Linearized guiding-center pieces: coupling projection, equilibrium gradients, linear seeds, step-0 observers.

Host only, no time integration; the fixtures are synthetic (not PDE equilibria).
"""
import numpy as np
import pytest

from hybridge.cases.band_displacement import BandCoordinates, HamiltonianDisplacement
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from projects.diocotron.comparisons.hdg_projection import load_dolfinx_checkpoint
from scripts.guiding_center.cases.band_seed import BandSeed
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.runtime.linearized import (
    EquilibriumCoupling, equilibrium_gradient, linear_initial_density)
from tests.test_imported_equilibrium import cubic_window_checkpoint


def test_coupling_is_the_projection_of_minus_u_dot_grad_rho():
    space = DGSpace(rectangle_mesh(3, 4, xlim=(0, 1), ylim=(0, 2)), 4)
    qx = lambda x, y: x**2*y - 0.3*y**3 + 1
    qy = lambda x, y: -x*y**2 + 0.5*x**3
    gx = lambda x, y: np.sin(3*x)*np.cos(2*y)
    gy = lambda x, y: np.exp(x - y)
    flux = space.vector_field([space.project_callable(qx), space.project_callable(qy)], name="q")
    source = EquilibriumCoupling(space, lambda x, y: (gx(x, y), gy(x, y))).source(flux)
    # u = (-q_y, q_x), so -u . g = q_y g_x - q_x g_y
    expected = space.project_callable(lambda x, y: qy(x, y)*gx(x, y) - qx(x, y)*gy(x, y))
    np.testing.assert_allclose(source.coeffs, expected.coeffs, atol=1e-12*np.abs(expected.coeffs).max())


def test_coupling_rejects_missing_or_nonfinite_gradients():
    space = DGSpace(rectangle_mesh(2, 2, xlim=(0, 1), ylim=(0, 1)), 2)
    with pytest.raises(ValueError, match="grad rho"):
        EquilibriumCoupling(space, None)
    with pytest.raises(ValueError, match="finite"):
        EquilibriumCoupling(space, lambda x, y: (np.nan*x, 0*y))


def test_extrapolated_coupling_uses_the_bdf2_weights():
    space = DGSpace(rectangle_mesh(2, 2, xlim=(0, 1), ylim=(0, 1)), 2)
    coupling = EquilibriumCoupling(space, lambda x, y: (0*x, 0*y))
    a, b = space.project_callable(lambda x, y: x + y), space.project_callable(lambda x, y: x*y)
    assert coupling.extrapolated(a, None) is a
    np.testing.assert_allclose(coupling.extrapolated(a, b).coeffs, 2*a.coeffs - b.coeffs, atol=1e-14)


def test_disk_gradient_comes_from_the_radial_derivative_and_vanishes_at_the_center():
    case = case_definition_by_key("diocotron_annulus").build(k=9, s_minus=.45, s_plus=.5, inner_wall=0.0,
                                                             wall=1.0, epsilon=1.0, p=4.0, truncate=False)
    gradient = equilibrium_gradient(case)
    x, y = np.array([0.3, 0.0, 0.47, -0.2]), np.array([0.4, 0.0, 0.1, -0.44])
    gx, gy = gradient(x, y)
    h = 1e-6
    fx = (case.equilibrium_density(x + h, y) - case.equilibrium_density(x - h, y))/(2*h)
    fy = (case.equilibrium_density(x, y + h) - case.equilibrium_density(x, y - h))/(2*h)
    scale = max(np.abs(fx).max(), np.abs(fy).max())
    np.testing.assert_allclose(gx, fx, atol=1e-7*scale)
    np.testing.assert_allclose(gy, fy, atol=1e-7*scale)
    assert gx[1] == 0.0 and gy[1] == 0.0
    # the multiplicative seed is exactly linear: q_0 = rho* cos(9 theta)
    q0 = linear_initial_density(case)
    theta = np.arctan2(y, x)
    np.testing.assert_allclose(q0(x, y), case.equilibrium_density(x, y)*np.cos(9*theta), atol=1e-14)


def test_checkpoint_gradient_is_the_exact_polynomial_gradient(tmp_path):
    path, cubic = cubic_window_checkpoint(tmp_path)
    source = load_dolfinx_checkpoint(path)
    points = np.random.default_rng(3).uniform(0.02, 0.98, (257, 2))
    gx, gy = source.evaluate_gradients("phi", points, chunk_size=50)
    h = 1e-6
    fx = (cubic(points[:, 0] + h, points[:, 1]) - cubic(points[:, 0] - h, points[:, 1]))/(2*h)
    fy = (cubic(points[:, 0], points[:, 1] + h) - cubic(points[:, 0], points[:, 1] - h))/(2*h)
    np.testing.assert_allclose(gx, fx, atol=1e-8)
    np.testing.assert_allclose(gy, fy, atol=1e-8)
    # boundary round-off goes through the triangulation locator, as for values
    edge = np.array([[-np.finfo(float).eps, 0.3]])
    np.testing.assert_allclose(np.ravel(source.evaluate_gradients("phi", edge)), [
        (cubic(h, 0.3) - cubic(-h, 0.3))/(2*h), (cubic(0, 0.3 + h) - cubic(0, 0.3 - h))/(2*h)], atol=1e-8)


def test_window_density_gradient_is_window_slope_times_potential_gradient(tmp_path):
    from scripts.guiding_center.cases.imported_equilibrium import imported_equilibrium
    path, cubic = cubic_window_checkpoint(tmp_path)
    profile = imported_equilibrium(checkpoint=path).imported_equilibrium
    x, y = np.random.default_rng(5).uniform(0.05, 0.95, (2, 64))
    gx, gy = profile.density_gradient(x, y)
    h = 1e-6
    fx = (profile.density(x + h, y) - profile.density(x - h, y))/(2*h)
    fy = (profile.density(x, y + h) - profile.density(x, y - h))/(2*h)
    np.testing.assert_allclose(gx, fx, atol=1e-6*max(1.0, np.abs(fx).max()))
    np.testing.assert_allclose(gy, fy, atol=1e-6*max(1.0, np.abs(fy).max()))


class _Profile:
    """Stand-in for WindowEquilibrium: grad rho* = (1, 2) everywhere, no hint cells."""

    def row_cells(self, shape):
        return None

    def density_gradient(self, x, y, cells=None):
        return np.ones_like(x), 2.0*np.ones_like(y)


def test_linear_perturbation_is_minus_seed_velocity_dot_gradient():
    t = np.linspace(0, 2*np.pi, 64, endpoint=False)
    coordinates = BandCoordinates.from_polyline(np.column_stack([np.cos(t), np.sin(t)]))
    x = y = np.linspace(-2, 2, 81)
    X, Y = np.meshgrid(x, y, indexing="ij")
    displacement = HamiltonianDisplacement(x, y, 1e-3*np.sin(2*X)*np.cos(Y))
    seed = BandSeed(coordinates=coordinates, half_width=0.1, thickness=0.1, max_mode=8, parameters={},
                    displacement=displacement)
    q0 = seed.linear_perturbation(_Profile(), scale=50.0)
    points = np.random.default_rng(7).uniform(-1.5, 1.5, (40, 2))
    velocity = displacement.velocity(points)
    np.testing.assert_allclose(q0(points[:, 0], points[:, 1]),
                               -50.0*(velocity[:, 0] + 2.0*velocity[:, 1]), atol=1e-15)
    with pytest.raises(ValueError, match="displacement"):
        BandSeed(coordinates=coordinates, half_width=0.1, thickness=0.1, max_mode=8,
                 parameters={}).linear_perturbation(_Profile())


def test_initial_state_is_forwarded_only_to_observers_that_record_it(tmp_path):
    from scripts.guiding_center.runtime.observers import Observers, SpillingSnapshotSaver

    calls = []
    saver = SpillingSnapshotSaver((0.5,), 0.1, tmp_path/"spill")
    space = DGSpace(rectangle_mesh(1, 1, xlim=(0, 1), ylim=(0, 1)), 1)
    density = space.project_callable(lambda x, y: x)

    class Result:
        field = space.project_callable(lambda x, y: y)
        flux = None

    class Snapshot:
        step, time, accepted_density, poisson_result, equilibrium_potential = 0, 0.0, density, Result(), None

    Snapshot.space = space
    Observers(saver, lambda snapshot: calls.append(snapshot.step)).observe_initial(Snapshot())
    assert calls == [] and [p.name for p in saver.files] == ["state_0000000.npz"]
    stored = np.load(saver.files[0])
    assert float(stored["time"]) == 0.0
    np.testing.assert_allclose(stored["density"], density.coeffs)


def test_linearized_preset_validation():
    from dataclasses import replace

    from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset
    from scripts.guiding_center.runtime.configuration import _validate_config

    base = GuidingCenterRunPreset(case="diocotron_annulus", description="linear", linearized=True,
                                  time_scheme="si-bdf2")
    _validate_config(base)
    for change, message in (({"time_scheme": "si-euler"}, "si-bdf2"), ({"density_positivity": "kkt"}, "positivity"),
                            ({"enstrophy_budget": True}, "enstrophy")):
        with pytest.raises(ValueError, match=message):
            _validate_config(replace(base, **change))


def test_imported_equilibrium_on_an_external_mesh(tmp_path):
    from scripts.guiding_center.cases.imported_equilibrium import imported_equilibrium
    from tests.test_imported_equilibrium import window_checkpoint
    path = window_checkpoint(tmp_path)
    mesh = rectangle_mesh(5, 4, xlim=(0, 1), ylim=(0, 1))
    np.savez(tmp_path/"mesh.npz", node_coords=mesh.node_coords, triangles=mesh.triangles)
    case = imported_equilibrium(checkpoint=path, mesh_npz=tmp_path/"mesh.npz")
    profile = case.imported_equilibrium
    assert profile.to_mesh().num_tri == mesh.num_tri
    assert profile.row_cells((mesh.num_tri, 3)) is None
    assert profile.provenance["mesh_npz_sha256"] is not None
    x, y = np.array([0.2, 0.7]), np.array([0.4, 0.9])
    np.testing.assert_allclose(profile.density(x, y), case.equilibrium_density(x, y))
    with pytest.raises(ValueError, match="remesh_size or mesh_npz"):
        imported_equilibrium(checkpoint=path, mesh_npz=tmp_path/"mesh.npz", remesh_size=0.1)
    np.savez(tmp_path/"outside.npz", node_coords=mesh.node_coords + 2.0, triangles=mesh.triangles)
    with pytest.raises(ValueError, match="bounding box"):
        imported_equilibrium(checkpoint=path, mesh_npz=tmp_path/"outside.npz").imported_equilibrium.to_mesh()
