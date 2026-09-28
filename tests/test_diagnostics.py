from __future__ import annotations

import numpy as np

from hdgfem import DGSpace, evaluate_scalar_error, evaluate_vector_error, rectangle_mesh
from hdgfem.core.field_ops import (
    coefficient_field,
    field_linear_combination,
    perpendicular_vector_field,
    project_callable_to_trace,
    trace_linear_combination,
    vector_field_linear_combination,
)


def _space(order: int = 2) -> DGSpace:
    return DGSpace(rectangle_mesh(1, 1), order, basis_type="dub_orth", volume_quad_1d=6)


def test_scalar_error_report_reuses_field_quadrature_and_returns_plot_samples() -> None:
    space = _space()
    exact = lambda x, y: 1.0 + 2.0 * x - 0.5 * y
    field = space.project_callable(exact)

    report = evaluate_scalar_error(field, exact, sample_resolution=7, include_samples=True)

    assert report.metrics.l2 < 1.0e-12
    assert report.metrics.linf < 1.0e-12
    assert report.samples is not None
    assert report.samples.reference_points.ndim == 2
    assert report.samples.numerical_values.shape == report.samples.exact_values.shape
    np.testing.assert_allclose(report.samples.absolute_error, 0.0, atol=1.0e-12)


def test_vector_error_report_uses_euclidean_sampled_maximum() -> None:
    """Report vector L2, Euclidean sampled Linf, and component maxima."""
    space = _space(1)
    vector = (space * space).field(
        (space.constant(3.0), space.constant(-4.0)),
        name="constant_vector",
    )
    report = evaluate_vector_error(
        vector,
        lambda x, y: (0.0 * x, 0.0 * y),
        volume_quad_1d=7,
        sample_resolution=8,
        include_samples=True,
    )
    domain_measure = float(
        np.sum(space.mesh.aff_jacs) * np.sum(space.quad_data.Krf_w)
    )
    np.testing.assert_allclose(
        report.metrics.l2,
        5.0 * np.sqrt(domain_measure),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(report.metrics.linf, 5.0, rtol=0.0, atol=3.0e-15)
    np.testing.assert_allclose(
        report.metrics.component_linf,
        (3.0, 4.0),
        rtol=0.0,
        atol=3.0e-15,
    )
    assert report.samples is not None
    np.testing.assert_allclose(report.samples.error_magnitude, 5.0)


def test_vector_error_report_is_exact_for_linear_vector() -> None:
    """Resolve exactly represented vector formulas on the independent sample grid."""
    space = _space(1)
    exact = lambda x, y: (1.0 + x - 0.5 * y, -0.25 + 2.0 * y)
    vector = (space * space).field(
        (
            space.project_callable(lambda x, y: exact(x, y)[0]).coeffs,
            space.project_callable(lambda x, y: exact(x, y)[1]).coeffs,
        )
    )
    report = evaluate_vector_error(vector, exact, sample_resolution=9)
    assert report.metrics.l2 < 1.0e-12
    assert report.metrics.linf < 1.0e-12


def test_field_and_vector_operations_use_one_shared_api() -> None:
    space = _space(1)
    one = coefficient_field(space, 1.0, name="one")
    x = coefficient_field(space, lambda x, y: x, name="x")
    combined = field_linear_combination(space, [(2.0, one), (-0.5, x)])
    expected = space.project_callable(lambda x, y: 2.0 - 0.5 * x)
    np.testing.assert_allclose(combined.coeffs, expected.coeffs, rtol=1.0e-12, atol=1.0e-12)

    vector = (space * space).field((one.coeffs, x.coeffs), name="q")
    doubled = vector_field_linear_combination(space, [(2.0, vector)])
    rotated = perpendicular_vector_field(vector, 3.0, space)
    np.testing.assert_allclose(doubled.components[0].coeffs, 2.0 * one.coeffs)
    np.testing.assert_allclose(doubled.components[1].coeffs, 2.0 * x.coeffs)
    np.testing.assert_allclose(rotated.components[0].coeffs, -3.0 * x.coeffs)
    np.testing.assert_allclose(rotated.components[1].coeffs, 3.0 * one.coeffs)
    np.testing.assert_allclose(vector.l2_norm() ** 2, one.l2_norm() ** 2 + x.l2_norm() ** 2)


def test_trace_projection_and_linear_combination_are_basis_aware() -> None:
    space = _space(2)
    trace_space = space.trace_space("legendre-modal")
    trace = project_callable_to_trace(
        space, lambda x, y: 2.5 + 0.0 * x * y, trace_basis="legendre-modal", reduced=True,
    )
    coefficients = trace.reshape((-1, trace_space.edg_dof))
    values = coefficients @ trace_space.bas1d_of_ref_edg_qds
    np.testing.assert_allclose(values, 2.5, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(trace_linear_combination([(0.25, trace), (0.75, trace)]), trace)


def test_space_l2_diff_accepts_every_field_callable_pairing() -> None:
    mesh = rectangle_mesh(2, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=7)
    other_space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8)
    left_exact = lambda x, y: 1.0 + x - 2.0 * y
    right_exact = lambda x, y: -0.5 + 0.25 * x + y
    left = space.project_callable(left_exact)
    right = other_space.project_callable(right_exact)
    expected = space.l2_diff(left_exact, right_exact)

    np.testing.assert_allclose(space.l2_diff(left, right), expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(space.l2_diff(left, right_exact), expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(space.l2_diff(left_exact, right), expected, rtol=1.0e-12, atol=1.0e-12)
    assert space.l2_diff(left, left_exact) < 1.0e-12
    assert space.linf_diff(left, left_exact) < 1.0e-12


def test_scalar_field_integral_and_min_max_are_field_operations() -> None:
    space = _space(1)
    field = space.constant(2.5)
    domain_measure = float(np.sum(space.mesh.aff_jacs) * np.sum(space.quad_data.Krf_w))

    np.testing.assert_allclose(field.integral(), 2.5 * domain_measure, rtol=1.0e-13, atol=1.0e-13)
    assert field.min_max() == (2.5, 2.5)


def test_diagnostic_packing_rejects_unreduced_arrays_before_download() -> None:
    from types import SimpleNamespace
    import pytest
    from hdgfem.diagnostics import _diagnostic_scalars

    calls = []
    namespace = SimpleNamespace(asarray=np.asarray, stack=np.stack,
                                asnumpy=lambda array: calls.append(array.copy()) or array)
    with pytest.raises(ValueError, match="scalar reductions"):
        _diagnostic_scalars({"mass": 2.0, "unreduced": np.ones((4, 8))}, namespace)
    assert calls == []
    assert _diagnostic_scalars({"mass": 2.0, "energy": 3.0}, namespace) == {"mass": 2.0, "energy": 3.0}
    assert len(calls) == 1 and calls[0].shape == (2,)


def test_azimuthal_host_override_preserves_explicit_backend_choice() -> None:
    import pytest
    from hdgfem.diagnostics import azimuthal_mode_diagnostics

    space = _space(2)
    equilibrium = space.project_callable(lambda x, y: 1.0 + 0.1*x)
    density = space.project_callable(lambda x, y: 1.0 + 0.1*x + 0.02*(x*x-y*y))
    expected = azimuthal_mode_diagnostics(density, equilibrium, 2)
    # Existing host coefficients remain authoritative for an explicit host call.
    density._device_coeffs = {0: object()}
    assert azimuthal_mode_diagnostics(density, equilibrium, 2, backend="host") == expected
    assert azimuthal_mode_diagnostics(density, equilibrium, 0) == {}
    with pytest.raises(ValueError, match="backend"):
        azimuthal_mode_diagnostics(density, equilibrium, 2, backend="invalid")


def test_solution_trace_prefers_device_and_reduces_host_trace() -> None:
    from types import SimpleNamespace

    from hdgfem.core.field_ops import solution_trace

    space = _space(1)
    edge_dofs = space.quad_data.edg_dof
    full = np.arange(space.mesh.num_edg * edge_dofs, dtype=np.float64)
    device_sentinel = object()
    result = SimpleNamespace(trace=full, trace_reduced_device=device_sentinel)

    assert solution_trace(result, space) is device_sentinel
    expected = full.reshape(space.mesh.num_edg, edge_dofs)[space.mesh.int_edges_inds].ravel()
    np.testing.assert_array_equal(
        solution_trace(result, space, reduced=True, prefer_device=False), expected
    )
