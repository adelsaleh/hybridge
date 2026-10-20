"""Band-mode spectrum along a closed band (synthetic ring on a square mesh)."""
import numpy as np
import pytest

from hybridge.cases.band_displacement import BandCoordinates, c2_bump
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.projection import project_callable
from hybridge.core.space import DGSpace
from hybridge.diagnostics.band_modes import BandModeSpectrum, participation_number

R, W = 0.6, 0.15


def setup():
    space = DGSpace(rectangle_mesh(30, 30, xlim=(-1, 1), ylim=(-1, 1)), 3, volume_quad_1d=8)
    t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    coords = BandCoordinates.from_polyline(np.column_stack([R * np.cos(t), R * np.sin(t)]))

    def theta_and_weight(x, y):
        theta, eta, *_ = coords.evaluate(np.column_stack([x, y]), max_distance=W)
        ok = np.isfinite(theta)
        return np.where(ok, theta, 0.0), np.where(ok, c2_bump(np.where(ok, eta, 0.0) / W), 0.0)

    ring = lambda x, y: np.exp(-((np.hypot(x, y) - R) / 0.05) ** 2)
    return space, theta_and_weight, ring


def test_spectrum_recovers_seeded_harmonics():
    space, tw, ring = setup()
    eps = 1e-3
    pert = lambda x, y: ring(x, y) * (1 + eps * np.cos(7 * np.arctan2(y, x)) + 0.5 * eps * np.sin(3 * np.arctan2(y, x)))
    spectrum = BandModeSpectrum(space, tw, max_mode=20, density_reference=project_callable(ring, space))
    out = spectrum.measure(project_callable(pert, space))
    a = np.array(out["band_rho_mode_amplitudes"])
    assert abs(a[7] - eps) < 1e-3 * eps and abs(a[3] - 0.5 * eps) < 1e-3 * eps
    assert np.delete(a[1:], [2, 6]).max() < 1e-3 * eps and abs(a[0]) < 1e-10
    assert out["band_rho_dominant_modes"][:2] == [7, 3] and out["band_rho_active_modes"] == 2
    assert abs(out["band_rho_participation"] - (1.25 ** 2) / (1 + 0.0625)) < 1e-3


def test_participation_number():
    np.testing.assert_allclose(participation_number([[0.0, 1.0, 0.0]]), [1.0])
    np.testing.assert_allclose(participation_number([[5.0, 1.0, 1.0, 1.0, 1.0]]), [4.0])


def test_host_device_parity():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    space, tw, ring = setup()
    pert = lambda x, y: ring(x, y) * (1 + 1e-3 * np.cos(5 * np.arctan2(y, x)))
    reference, field = project_callable(ring, space), project_callable(pert, space)
    host = BandModeSpectrum(space, tw, max_mode=10, density_reference=reference).measure(field)
    device = BandModeSpectrum(space, tw, max_mode=10, density_reference=reference, backend="device").measure(field)
    np.testing.assert_allclose(device["band_rho_mode_amplitudes"], host["band_rho_mode_amplitudes"], atol=1e-14)
