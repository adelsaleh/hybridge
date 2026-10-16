"""Reduced-operation Oswald flux (hybridge.core.compatible_flux) against oswald_average(...).flux_field(...).

Polynomial tables and random potentials only; no PDE solve.
"""
import numpy as np
import pytest

from hybridge.core.compatible_flux import compatible_flux_field
from hybridge.core.conforming import oswald_average
from hybridge.core.mesh import DGMesh, rectangle_mesh
from hybridge.core.space import DGSpace


def _mesh():
    """Distorted rectangle mesh: both edge orientations, boundary and interior nodes."""
    mesh = rectangle_mesh(4, 3, xlim=(-1.0, 1.0), ylim=(-1.0, 0.5))
    nodes = np.array(mesh.node_coords, dtype=np.float64)
    interior = (np.abs(nodes[:, 0]) < 1.0 - 1e-12) & (nodes[:, 1] > -1.0 + 1e-12) & (nodes[:, 1] < 0.5 - 1e-12)
    nodes[interior] += 0.05 * np.sin(3.0 * nodes[interior][:, ::-1])
    return DGMesh.from_arrays(nodes, np.asarray(mesh.triangles, dtype=np.int64))


def _boundary(x, y):
    return np.cos(2.0 * x) + 0.3 * y


def _case(k: int, boundary):
    mesh = _mesh()
    post = DGSpace(mesh, k, basis_type="dub_orth")
    target = DGSpace(mesh, max(k - 1, 0), basis_type="dub_orth")
    coefficients = np.random.default_rng(k).standard_normal(post.shape)
    reference = oswald_average(post.field(coefficients), boundary_values=boundary).flux_field(target)
    return post, target, coefficients, reference


@pytest.mark.parametrize("boundary", (0.0, _boundary))
@pytest.mark.parametrize("k", (1, 2, 4, 7, 10))
def test_host_compatible_flux_matches_oswald_flux(k, boundary):
    post, target, coefficients, reference = _case(k, boundary)
    fast = compatible_flux_field(post.field(coefficients), target, boundary_values=boundary)
    scale = max(np.abs(component.coeffs).max() for component in reference.components)
    for got, expected in zip(fast.components, reference.components):
        np.testing.assert_allclose(got.coeffs, expected.coeffs, rtol=0.0, atol=1e-13 * scale)


@pytest.mark.parametrize("k", (2, 5, 8, 10))
def test_device_compatible_flux_matches_host(k):
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    from hybridge.core.device import field_from_cupy_coefficients

    post, target, coefficients, reference = _case(k, _boundary)
    fast = compatible_flux_field(field_from_cupy_coefficients(post, cp.asarray(coefficients)), target,
                                 boundary_values=_boundary)
    assert not fast.components[0].coefficients_materialized
    scale = max(np.abs(component.coeffs).max() for component in reference.components)
    for got, expected in zip(fast.components, reference.components):
        np.testing.assert_allclose(got.coeffs, expected.coeffs, rtol=0.0, atol=1e-12 * scale)
