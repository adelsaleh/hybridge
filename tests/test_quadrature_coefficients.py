"""Quadrature-point coefficients: layouts, single-valued normal flux, arithmetic and moments.

``QuadratureVectorField`` and ``QuadratureScalar`` (``hybridge.core.quadrature_coefficients``) carry
exact PDE coefficients at the points where the transport assembly integrates them. Device
copies run only with ``HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.element_coefficients import ElementCoefficient
from hybridge.core.mesh import DGMesh
from hybridge.core.projection import scalar_moments_from_values
from hybridge.core.quadrature_coefficients import QuadratureVectorField, QuadratureScalar, is_quadrature_coefficient
from hybridge.core.space import VectorDGField
from hybridge.core.time_integration import bdf2_transport_data, bdf3_transport_data

CUDA = os.environ.get("HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS") == "1"


def _cuda_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - no usable CUDA runtime
        return False


cuda_only = pytest.mark.skipif(not (CUDA and _cuda_available()),
                               reason="set HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1 on a CUDA machine")


def law(x, y):
    """A smooth advection that is tangent to no face of the test meshes."""
    return np.sin(2.0 * x + y) + 1.7, np.cos(x - 2.0 * y) * x + 0.4


LAW_PAIR = (lambda x, y: law(x, y)[0], lambda x, y: law(x, y)[1])


def _distorted_mesh() -> DGMesh:
    """A 4x3 rectangle mesh with perturbed interior nodes."""
    base = rectangle_mesh(4, 3)
    nodes = np.array(base.node_coords, dtype=float, copy=True)
    interior = (np.abs(nodes[:, 0]) < 1 - 1e-12) & (np.abs(nodes[:, 1]) < 1 - 1e-12)
    rng = np.random.default_rng(7)
    nodes[interior] += 0.08 * rng.uniform(-1.0, 1.0, size=(int(interior.sum()), 2))
    return DGMesh.from_arrays(nodes, np.array(base.triangles))


def _spaces(order=3, mesh=None, trace_basis="legacy-lagrange"):
    mesh = rectangle_mesh(4, 3) if mesh is None else mesh
    space = DGSpace(mesh, order, basis_type="dub_orth")
    return space, space.trace_space(trace_basis)


def _interior_pairs(mesh):
    slots = np.asarray(mesh.edge_side_indices)
    interior = slots[:, 1] >= 0
    return slots[interior, 0], slots[interior, 1]


@pytest.mark.parametrize("make_mesh", [lambda: rectangle_mesh(4, 3), _distorted_mesh], ids=["regular", "distorted"])
@pytest.mark.parametrize("order", [1, 3])
def test_callable_drift_layout_and_bitwise_antisymmetry(make_mesh, order):
    space, trace = _spaces(order, make_mesh())
    advection = QuadratureVectorField.from_callable(law, space, trace_space=trace)
    nqv = space.quad_data.Krf_w.shape[0]
    nqf = np.asarray(trace.quads).size
    assert advection.volume.shape == (space.mesh.num_tri, nqv, 2)
    assert advection.normal_flux.shape == (space.mesh.num_tri, 3, nqf)
    assert advection.single_valued and is_quadrature_coefficient(advection)
    first, second = _interior_pairs(space.mesh)
    flux = advection.normal_flux.reshape(-1, nqf)
    assert np.array_equal(flux[second], -flux[first][:, ::-1])
    assert advection.antisymmetry_defect() == 0.0
    # Volume values are the law at the mapped volume points.
    points = space.mesh.map_reference_points(space.quad_data.Krf_quads)
    bx, by = law(points[..., 0], points[..., 1])
    np.testing.assert_allclose(advection.volume[..., 0], bx, rtol=0, atol=1e-14)
    np.testing.assert_allclose(advection.volume[..., 1], by, rtol=0, atol=1e-14)


def test_face_points_coincide_and_boundary_sides_use_their_own_normal():
    space, trace = _spaces(2, _distorted_mesh())
    from hybridge.core.quadrature_coefficients import _side_points

    points = _side_points(space, trace, np)
    first, second = _interior_pairs(space.mesh)
    h = float(np.max(np.linalg.norm(np.diff(space.mesh.node_coords[space.mesh.triangles], axis=1), axis=-1)))
    assert np.max(np.abs(points[second] - points[first][:, ::-1])) < 1e-14 * max(h, 1.0)
    advection = QuadratureVectorField.from_callable(law, space, trace_space=trace)
    slots = np.asarray(space.mesh.edge_side_indices)
    boundary = slots[slots[:, 1] < 0, 0]
    normals = space.mesh.normals.reshape(-1, 2)[boundary]
    bx, by = law(points[boundary][..., 0], points[boundary][..., 1])
    expected = bx * normals[:, 0, None] + by * normals[:, 1, None]
    flux = advection.normal_flux.reshape(-1, points.shape[1])[boundary]
    np.testing.assert_allclose(flux, expected, rtol=0, atol=1e-14)


def test_pair_of_laws_matches_vector_law():
    space, trace = _spaces(2)
    one = QuadratureVectorField.from_callable(law, space, trace_space=trace)
    two = QuadratureVectorField.from_callable((lambda x, y: law(x, y)[0], lambda x, y: law(x, y)[1]),
                                        space, trace_space=trace)
    assert np.array_equal(one.volume, two.volume) and np.array_equal(one.normal_flux, two.normal_flux)


def test_linear_combinations_keep_bitwise_antisymmetry_and_bdf_algebra():
    space, trace = _spaces(2, _distorted_mesh())
    a = QuadratureVectorField.from_callable(law, space, trace_space=trace)
    b = QuadratureVectorField.from_callable(lambda x, y: (x * y - 0.3, np.exp(0.2 * x) + y), space, trace_space=trace)
    c = QuadratureVectorField.from_callable(lambda x, y: (1.0 + 0.0 * x, 0.5 - y), space, trace_space=trace)
    combo = QuadratureVectorField.linear_combination([(3, a), (-3, b), (1, c)])
    assert combo.single_valued and combo.antisymmetry_defect() == 0.0
    for value in (0.37 * a, a / 3.0, -a, a + b, a - b):
        assert value.antisymmetry_defect() == 0.0
    density = space.project_callable(lambda x, y: np.cos(x) * y)
    _, beta2, scale2 = bdf2_transport_data(density, a, 0.1, previous_field=density, previous_velocity=b)
    expected2 = (2 * 0.1 / 3) * QuadratureVectorField.linear_combination([(2, a), (-1, b)])
    assert scale2 == pytest.approx(2 * 0.1 / 3)
    assert np.array_equal(beta2.volume, expected2.volume)
    assert np.array_equal(beta2.normal_flux, expected2.normal_flux)
    _, beta3, scale3 = bdf3_transport_data(density, a, 0.1, previous_field=density, previous_velocity=b,
                                           older_field=density, older_velocity=c)
    expected3 = (6 * 0.1 / 11) * combo
    assert scale3 == pytest.approx(6 * 0.1 / 11)
    assert np.array_equal(beta3.volume, expected3.volume)
    assert np.array_equal(beta3.normal_flux, expected3.normal_flux)
    _, euler, _ = bdf2_transport_data(density, a, 0.1)
    assert np.array_equal(euler.volume, 0.1 * a.volume)
    dg = VectorDGField(LAW_PAIR, space)
    with pytest.raises(TypeError):
        bdf2_transport_data(density, a, 0.1, previous_field=density, previous_velocity=dg)


def test_sampled_drifts_follow_their_sources():
    space, trace = _spaces(2)
    high = DGSpace(space.mesh, 4, basis_type="dub_orth")
    field = VectorDGField(LAW_PAIR, high)
    sampled = QuadratureVectorField.from_vector_field(field, space, trace_space=trace)
    assert not sampled.single_valued
    expected_x = field.components[0].values_at_ref(space.quad_data.Krf_quads)
    np.testing.assert_allclose(sampled.volume[..., 0], expected_x, rtol=0, atol=1e-13)

    def element_law(points, *, xp, t=None):
        """Element-local constant advection (1, 2) on every element."""
        values = xp.zeros((space.mesh.num_tri, points.shape[0], 2))
        values[..., 0] = 1.0
        values[..., 1] = 2.0
        return values

    coefficient = ElementCoefficient(element_law, space.mesh, 2, "constant")
    element = QuadratureVectorField.from_element_coefficient(coefficient, space, trace_space=trace)
    normals = space.mesh.normals
    expected = (normals[..., 0] + 2.0 * normals[..., 1])[..., None]
    np.testing.assert_allclose(element.normal_flux, np.broadcast_to(expected, element.normal_flux.shape),
                               rtol=0, atol=1e-14)


def test_scalar_moments_and_projection_round_trip():
    space, _ = _spaces(3, _distorted_mesh())
    scalar = QuadratureScalar.from_callable(lambda x, y: np.exp(x) * y + 0.5, space, name="source")
    expected = scalar_moments_from_values(space, scalar.volume)
    np.testing.assert_allclose(scalar.moments(), expected, rtol=0, atol=1e-13 * np.max(np.abs(expected)))
    field = space.project_callable(lambda x, y: np.sin(x + 2 * y))
    sampled = QuadratureScalar.from_field(field, space)
    round_trip = sampled.project()
    np.testing.assert_allclose(round_trip.coeffs, field.coeffs, rtol=0, atol=1e-13)
    combo = QuadratureScalar.linear_combination([(2.0, scalar), (-1.0, sampled)])
    np.testing.assert_allclose(combo.volume, 2.0 * scalar.volume - sampled.volume, rtol=0, atol=0)
    constant = QuadratureScalar.from_callable(lambda x, y: 2.0, space)
    assert np.all(constant.volume == 2.0)


def test_layout_checks_reject_other_spaces():
    space, trace = _spaces(2)
    advection = QuadratureVectorField.from_callable(law, space, trace_space=trace)
    other_mesh, _ = _spaces(2, rectangle_mesh(3, 3))
    with pytest.raises(ValueError, match="different mesh"):
        advection.check(other_mesh)
    other_rule = DGSpace(space.mesh, 2, basis_type="dub_orth", volume_degree=9)
    with pytest.raises(ValueError, match="volume quadrature"):
        advection.check(other_rule)
    with pytest.raises(ValueError, match="face quadrature"):
        advection.check(space, DGSpace(space.mesh, 3, basis_type="dub_orth").trace_space("legacy-lagrange"))
    with pytest.raises(ValueError, match="shape"):
        QuadratureVectorField(space, trace, advection.volume[:, :-1], advection.normal_flux)
    with pytest.raises(ValueError, match="shape"):
        QuadratureScalar(space, np.zeros((3, 3)))
    bad = advection.copy()
    bad.volume[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bad.check(space)
    with pytest.raises(TypeError):
        advection + QuadratureScalar.from_callable(lambda x, y: x, space)


@cuda_only
def test_device_drift_matches_host_and_stays_resident():
    import cupy as cp

    space, trace = _spaces(3, _distorted_mesh())
    host = QuadratureVectorField.from_callable(law, space, trace_space=trace)

    def device_law(x, y):
        """The same advection law on CuPy arrays."""
        return cp.sin(2.0 * x + y) + 1.7, cp.cos(x - 2.0 * y) * x + 0.4

    device = QuadratureVectorField.from_callable(device_law, space, trace_space=trace, device=True)
    assert device.on_device and device.single_valued
    np.testing.assert_allclose(cp.asnumpy(device.volume), host.volume, rtol=0, atol=1e-13)
    np.testing.assert_allclose(cp.asnumpy(device.normal_flux), host.normal_flux, rtol=0, atol=1e-13)
    assert device.antisymmetry_defect() == 0.0
    scaled = 0.5 * device - device
    assert scaled.on_device and scaled.antisymmetry_defect() == 0.0
    scalar = QuadratureScalar.from_callable(lambda x, y: cp.exp(x) * y, space, device=True)
    moments = scalar.moments()
    assert hasattr(moments, "__cuda_array_interface__")
    np.testing.assert_allclose(cp.asnumpy(moments), scalar.to_host().moments(), rtol=0, atol=1e-13)
    field = scalar.project()
    assert not field.coefficients_materialized
