"""Checks of a displacement-sensitive band diagnostic; no time integration."""
import numpy as np

from hybridge.cases.band_displacement import BandCoordinates
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.projection import project_callable
from hybridge.core.space import DGSpace
from hybridge.diagnostics.band_profiles import BandProbeGrid, BandProfileSampler


def grid():
    theta = np.arange(2048) * 2*np.pi/2048
    coordinates = BandCoordinates.from_polyline(
        .5 * np.column_stack([np.cos(theta), np.sin(theta)]))
    return BandProbeGrid.from_coordinates(coordinates, .1, angles=128, transverse=16)


def test_opposite_edge_signals_do_not_cancel():
    probes = grid()
    # Linear sinuous response of an even density: odd in the normal coordinate.
    values = (probes.eta[:, None]/.1) * np.cos(7*probes.theta)
    signed_integral = np.einsum("r,rt->t", probes.weights, values)
    np.testing.assert_allclose(signed_integral, 0., atol=1e-15)
    amplitude = probes.amplitudes(values, max_mode=20)
    np.testing.assert_allclose(amplitude[7], 1/np.sqrt(3), rtol=1e-13)
    assert np.delete(amplitude, 7).max() < 1e-13


def test_phase_and_transverse_structure_are_retained():
    probes = grid()
    normal = 1 + probes.eta[:, None]
    values = normal * np.cos(3*probes.theta + .4)
    coefficient = probes.coefficients(values, max_mode=10)
    np.testing.assert_allclose(coefficient[:, 3], normal[:, 0]*np.exp(.4j)/2,
                               rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(probes.rms(values),
                               probes.amplitudes(values, max_mode=10)[3]/np.sqrt(2),
                               rtol=1e-13)


def test_sampler_recovers_polynomial_field_and_reuses_location():
    space = DGSpace(rectangle_mesh(8, 8, xlim=(-1, 1), ylim=(-1, 1)), 2)
    probes = grid()
    sampler = BandProfileSampler(space, probes)
    function = lambda x, y: x*x + 3*y - 2*x*y
    field = project_callable(function, space)
    reference = function(probes.points[..., 0], probes.points[..., 1])
    np.testing.assert_allclose(sampler(field), reference, atol=2e-13, rtol=0)
    np.testing.assert_allclose(sampler(field), reference, atol=2e-13, rtol=0)


def test_profile_observer_keeps_signed_fields_and_separate_degrees(tmp_path):
    from types import SimpleNamespace
    from scripts.guiding_center.runtime.observers import BandProfileObserver
    mesh = rectangle_mesh(6, 6, xlim=(-1, 1), ylim=(-1, 1))
    density_space, potential_space = DGSpace(mesh, 2), DGSpace(mesh, 3)
    density = project_callable(lambda x, y: x-y, density_space)
    potential = project_callable(lambda x, y: x*x*y, potential_space)
    probes = grid()
    observer = BandProfileObserver(tmp_path / "frames", probes, interval=.25, dt=.05)
    for step in (1, 2, 5, 10):
        observer(SimpleNamespace(
            time=step*.05, step=step, accepted_density=density,
            poisson_result=SimpleNamespace(field=potential),
            equilibrium_potential=potential))
    path = observer.save(tmp_path / "profiles.npz")
    with np.load(path) as saved:
        np.testing.assert_allclose(saved["time"], [.05, .25, .5])
        x, y = probes.points[..., 0], probes.points[..., 1]
        np.testing.assert_allclose(saved["density"][0], x-y, atol=1e-13)
        np.testing.assert_allclose(saved["potential"][0], x*x*y, atol=1e-13)
        assert saved["density"].min() < 0
