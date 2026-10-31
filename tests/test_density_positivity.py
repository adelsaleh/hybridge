"""KKT density positivity projection on small affine meshes.

The experiment-specific stepper checks live with the research application.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.runtime.precision import REAL_DTYPE
from hybridge.transport.positivity import DensityPositivityProjector, positivity_points






def touching_zero(x, y):
    """Nonnegative, non-polynomial density that vanishes along curves."""
    return np.sin(3.0 * (x * x + y * y) + x) ** 2


def partly_negative(x, y):
    """Mostly nonnegative density with shallow negative pockets (negative element means)."""
    return touching_zero(x, y) - 0.02 * np.cos(7.0 * x)


def element_masses(space, coeffs, projector):
    """Element integrals of a coefficient array."""
    return space.mesh.aff_jacs * (np.asarray(coeffs) @ projector.tables.integrals)


@pytest.mark.parametrize("basis", ["dub_orth", "bernstein"])
def test_projection_is_nonnegative_at_points_and_conserves_mass(basis):
    space = DGSpace(rectangle_mesh(10, 10), 4, basis_type=basis)
    field = space.project_callable(partly_negative)
    projector = DensityPositivityProjector(space)
    projected, report = projector.project(field)
    V = positivity_points(space)
    scale = np.abs(field.coeffs @ V.T).max()
    assert (field.coeffs @ V.T).min() < -1e-3 * scale
    assert (projected.coeffs @ V.T).min() >= -1e-14 * scale
    before, after = element_masses(space, field.coeffs, projector), element_masses(space, projected.coeffs, projector)
    assert abs(after.sum() - before.sum()) <= 1e-13 * abs(before).sum()
    assert report["positivity_negative_mean_projected"] + report["positivity_negative_mean_zeroed"] > 0
    assert report["positivity_mass_returned"] > 0
    assert report["positivity_flagged"] == np.count_nonzero((field.coeffs @ V.T).min(axis=1) < 0)


def test_nonnegative_means_are_kept_and_the_projection_is_optimal_and_idempotent():
    pytest.importorskip("scipy")
    from scipy.optimize import nnls

    space = DGSpace(rectangle_mesh(8, 8), 3, basis_type="dub_orth")
    field = space.project_callable(touching_zero)
    projector = DensityPositivityProjector(space)
    projected, report = projector.project(field)
    assert report["positivity_mass_returned"] == 0.0 and report["positivity_flagged"] > 0
    np.testing.assert_allclose(element_masses(space, projected.coeffs, projector),
                               element_masses(space, field.coeffs, projector), rtol=0, atol=1e-15)
    # Each flagged element matches the exact least-distance solution (Lawson-Hanson NNLS)
    # and is never farther from the input than Zhang-Shu scaling toward its mean.
    t = projector.tables
    y, x = np.asarray(field.coeffs), np.asarray(projected.coeffs)
    flagged = np.flatnonzero((y @ t.V.T).min(axis=1) < 0)
    for K in flagged:
        b = y[K] @ t.V.T
        E = np.vstack([t.W_mean.T, -b[None, :]])
        target = np.zeros(E.shape[0]); target[-1] = 1.0
        u, _ = nnls(E, target)
        residual = E @ u - target
        reference = -residual[:-1] / residual[-1]
        distance = (x[K] - y[K]) @ t.mass @ (x[K] - y[K])
        assert distance == pytest.approx(reference @ reference, rel=1e-8, abs=1e-14)
        mean = (y[K] @ t.integrals) / t.area
        theta = mean / (mean - b.min())
        scaled = mean * t.constant + theta * (y[K] - mean * t.constant)
        assert distance <= (scaled - y[K]) @ t.mass @ (scaled - y[K]) * (1 + 1e-12)
    again, second = projector.project(projected)
    assert second["positivity_flagged"] == 0
    np.testing.assert_array_equal(again.coeffs, projected.coeffs)


def test_point_constraints_keep_the_spatial_order():
    errors = []
    for nx in (8, 16, 32):
        space = DGSpace(rectangle_mesh(nx, nx), 3, basis_type="dub_orth", volume_quad_1d=10)
        field = space.project_callable(touching_zero)
        projected, _ = DensityPositivityProjector(space).project(field)
        points, weights = space.mapped_quads(), space.quad_data.Krf_w
        exact = touching_zero(points[..., 0], points[..., 1])
        error = lambda c: np.sqrt((space.mesh.aff_jacs[:, None] * weights
                                   * (np.asarray(c) @ space.quad_data.bas_of_quads - exact) ** 2).sum())
        errors.append((error(field.coeffs), error(projected.coeffs)))
    plain, kkt = np.array(errors).T
    assert np.log2(kkt[-2] / kkt[-1]) > 3.8
    assert kkt[-1] <= 1.05 * plain[-1]


def test_device_projection_matches_host():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space, field_from_cupy_coefficients

    space = DGSpace(rectangle_mesh(10, 10), 4, basis_type="dub_orth")
    field = space.project_callable(partly_negative)
    host, host_report = DensityPositivityProjector(space, backend="host").project(field)
    cspace = as_cupy_space(space)
    device_field = field_from_cupy_coefficients(space, cp.asarray(field.coeffs), device=cspace.device_id)
    device, device_report = DensityPositivityProjector(space, backend="device").project(device_field)
    assert not device.coefficients_materialized
    scale = np.abs(field.coeffs).max()
    np.testing.assert_allclose(cp.asnumpy(as_cupy_coefficients(device, cspace)), host.coeffs,
                               rtol=0, atol=1e-12 * scale)
    assert device_report["positivity_flagged"] == host_report["positivity_flagged"]


def test_projector_rejects_foreign_fields_and_nonpositive_mass():
    space = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
    projector = DensityPositivityProjector(space)
    other = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
    with pytest.raises(ValueError, match="own DGSpace"):
        projector.project(other.constant(1.0))
    with pytest.raises(ValueError, match="nonpositive total mass"):
        projector.project(space.constant(-1.0))
    with pytest.raises(ValueError, match="points"):
        DensityPositivityProjector(space, points="bernstein")






















@pytest.mark.parametrize("order", [8, 9])
def test_device_projection_matches_host_above_32_active_constraints(order):
    """At p >= 8 (45 and 55 modes) the active set of an element can exceed 32 constraints: the device kernel
    strides every lane loop over the whole set (it once indexed the set by lane and returned feasible but
    suboptimal projections), and its warps per block follow the device's opt-in shared memory limit."""
    pytest.importorskip("cupy")
    mesh = rectangle_mesh(2, 2)
    space = DGSpace(mesh, order)
    host = DensityPositivityProjector(space, points="quadrature+lattice", backend="host")
    try:
        device = DensityPositivityProjector(space, points="quadrature+lattice", backend="device")
    except Exception as error:  # noqa: BLE001 - no usable device in this environment
        pytest.skip(f"device projector unavailable: {error}")
    assert 1 <= device._device["warps"] <= 4
    rng = np.random.default_rng(order)
    coeffs = rng.normal(size=(mesh.num_tri, host.tables.modes)) * 0.04
    coeffs[:, 0] += 1.0
    coeffs[: mesh.num_tri // 2] *= 0.1
    from hybridge.core.space import DGField
    projected_host, report_host = host.project(DGField(space, coeffs.copy()), name="rho")
    projected_device, report_device = device.project(DGField(space, coeffs.copy()), name="rho")
    assert report_host["positivity_flagged"] == report_device["positivity_flagged"] > 0
    assert report_host["positivity_fallback"] == report_device["positivity_fallback"]
    mass = host.tables.mass
    difference = np.asarray(projected_host.coeffs) - np.asarray(projected_device.coeffs)
    distance = np.sqrt(np.einsum("ei,ij,ej->e", difference, mass, difference))
    assert distance.max() < 1e-12
    values = np.asarray(projected_device.coeffs) @ host.tables.V.T
    assert values.min() > -1e-12
