"""Reusable error diagnostics for scalar and vector DG fields."""

from __future__ import annotations

from hybridge.runtime.precision import REAL_DTYPE

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from hybridge.core.quadrature import ReferenceElementData
from hybridge.core.space import DGField, VectorDGField


@dataclass(frozen=True)
class ScalarErrorMetrics:
    """Scalar DG error norms and per-element maximum-error information."""

    l2: float
    linf: float
    mean_element_linf: float
    max_element: int


@dataclass(frozen=True)
class ScalarHDGErrorMetrics:
    """Scalar exact-error norms with the HDG gradient and face terms separate."""

    l2: float
    gradient_l2: float
    trace_mismatch: float
    backend: str = "host"

    @property
    def hdg_h1_seminorm(self) -> float:
        """Return sqrt(gradient error squared + weighted face mismatch)."""
        return float(np.hypot(self.gradient_l2, self.trace_mismatch))

    @property
    def hdg_h1(self) -> float:
        """Return the full HDG H1 error, including the volume L2 error."""
        return float(np.hypot(self.l2, self.hdg_h1_seminorm))


@dataclass(frozen=True)
class ScalarComparisonSamples:
    """Host samples used to compare a numerical and exact scalar field."""

    reference_points: np.ndarray
    numerical_values: np.ndarray
    exact_values: np.ndarray

    @property
    def absolute_error(self) -> np.ndarray:
        """Return the pointwise absolute error on the shared sample grid."""
        return np.abs(self.numerical_values - self.exact_values)


@dataclass(frozen=True)
class ScalarErrorReport:
    """Scalar metrics with optional samples suitable for plotting."""

    metrics: ScalarErrorMetrics
    samples: ScalarComparisonSamples | None = None


@dataclass(frozen=True)
class VectorErrorMetrics:
    """Vector DG L2 and sampled maximum-error information."""

    l2: float
    linf: float
    component_linf: tuple[float, ...]
    mean_element_linf: float
    max_element: int


@dataclass(frozen=True)
class VectorComparisonSamples:
    """Host samples used to compare numerical and exact vector fields."""

    reference_points: np.ndarray
    numerical_values: np.ndarray
    exact_values: np.ndarray

    @property
    def error_magnitude(self) -> np.ndarray:
        """Return Euclidean pointwise error magnitudes with shape (K,q)."""
        difference = self.numerical_values - self.exact_values
        return np.sqrt(np.sum(difference * difference, axis=0))


@dataclass(frozen=True)
class VectorErrorReport:
    """Vector metrics with optional samples suitable for plotting."""

    metrics: VectorErrorMetrics
    samples: VectorComparisonSamples | None = None


def _error_quadrature(field: DGField, volume_quad_1d: int | None, volume_degree: int | None = None):
    """Return reference points, weights, and basis values for error integration.

    ``volume_quad_1d`` selects a collapsed Gauss rule and ``volume_degree`` a
    rule of that polynomial exactness (see ``DGSpace``); by default the
    field's own volume quadrature is used.
    """
    space = field.space
    if volume_quad_1d is None and volume_degree is None:
        return space.quad_data.Krf_quads, space.quad_data.Krf_w, space.quad_data.bas_of_quads
    reference = ReferenceElementData.triangle(
        space.order,
        basis_type=space.quad_data.basis_type,
        volume_quad_1d=None if volume_quad_1d is None else int(volume_quad_1d),
        edge_quad_1d=space.quad_data.edge_quad_1d,
        volume_degree=volume_degree,
    )
    return reference.Krf_quads, reference.Krf_w, reference.bas_of_quads


def _weight_values(weight, x, y, xp):
    """Evaluate an optional spatial weight ``w(x, y)`` on mapped points (ones when absent)."""
    if weight is None:
        return None
    values = xp.asarray(weight(x, y), dtype=REAL_DTYPE)
    values = xp.broadcast_to(values, x.shape)
    if not bool(xp.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("error weight must be finite and nonnegative")
    return values


def _sample_reference_points(resolution: int) -> np.ndarray:
    """Return the standard triangular plotting sample grid."""
    from hybridge.core.quadrature import reference_plot_points

    return reference_plot_points(int(resolution))


def _evaluate_host(field, exact, *, volume_quad_1d, sample_resolution, include_samples,
                   weight=None, volume_degree=None):
    """Evaluate scalar metrics and optional samples with NumPy."""
    space = field.space
    error_points, weights, basis = _error_quadrature(field, volume_quad_1d, volume_degree)
    mapped = space.mesh.map_reference_points(error_points)
    exact_values = np.asarray(exact(mapped[:, :, 0], mapped[:, :, 1]), dtype=REAL_DTYPE)
    numerical_values = field.coeffs @ basis
    diff = numerical_values - exact_values
    squared = diff * diff
    spatial = _weight_values(weight, mapped[:, :, 0], mapped[:, :, 1], np)
    if spatial is not None:
        squared = squared * spatial
    l2 = float(np.sqrt(np.einsum("K,Kq,q->", space.mesh.aff_jacs, squared, weights, optimize=True)))
    if sample_resolution is None:
        reference_points, sampled_numerical, sampled_exact = error_points, numerical_values, exact_values
    else:
        reference_points = _sample_reference_points(sample_resolution)
        mapped = space.mesh.map_reference_points(reference_points)
        sampled_exact = np.asarray(exact(mapped[:, :, 0], mapped[:, :, 1]), dtype=REAL_DTYPE)
        sampled_numerical = field.coeffs @ space.basis_at(reference_points).T
    element_maximum = np.max(np.abs(sampled_numerical - sampled_exact), axis=1)
    metrics = ScalarErrorMetrics(
        l2=l2,
        linf=float(np.max(element_maximum)),
        mean_element_linf=float(np.mean(element_maximum)),
        max_element=int(np.argmax(element_maximum)),
    )
    samples = None if not include_samples else ScalarComparisonSamples(
        np.ascontiguousarray(reference_points, dtype=REAL_DTYPE),
        np.ascontiguousarray(sampled_numerical, dtype=REAL_DTYPE),
        np.ascontiguousarray(sampled_exact, dtype=REAL_DTYPE),
    )
    return ScalarErrorReport(metrics, samples)


def _evaluate_device(field, exact, *, volume_quad_1d, sample_resolution, include_samples,
                     weight=None, volume_degree=None):
    """Evaluate scalar metrics on the resident CUDA device with CuPy."""
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space
    from hybridge.runtime.optional import require_cupy

    cp = require_cupy()
    space = field.space
    cspace = as_cupy_space(space)
    coefficients = as_cupy_coefficients(field, cspace)
    if volume_quad_1d is None and volume_degree is None:
        error_points = cspace.quad_data.Krf_quads
        weights = cspace.quad_data.Krf_w
        basis = cspace.quad_data.bas_of_quads
    else:
        host_points, host_weights, host_basis = _error_quadrature(field, volume_quad_1d, volume_degree)
        error_points = cp.asarray(host_points, dtype=REAL_DTYPE)
        weights = cp.asarray(host_weights, dtype=REAL_DTYPE)
        basis = cp.asarray(host_basis, dtype=REAL_DTYPE)
    mapped = cp.einsum("Krc,qc->Krq", cspace.mesh.aff_mats, error_points) + cspace.mesh.aff_vecs[:, :, None]
    exact_values = cp.asarray(exact(mapped[:, 0, :], mapped[:, 1, :]), dtype=REAL_DTYPE)
    numerical_values = coefficients @ basis
    diff = numerical_values - exact_values
    squared = diff * diff
    spatial = _weight_values(weight, mapped[:, 0, :], mapped[:, 1, :], cp)
    if spatial is not None:
        squared = squared * spatial
    l2 = cp.sqrt(cp.einsum("K,Kq,q->", cspace.mesh.aff_jacs, squared, weights, optimize=True))
    if sample_resolution is None:
        reference_points, sampled_numerical, sampled_exact = error_points, numerical_values, exact_values
    else:
        host_reference_points = _sample_reference_points(sample_resolution)
        reference_points = cp.asarray(host_reference_points, dtype=REAL_DTYPE)
        sample_basis = cp.asarray(space.basis_at(host_reference_points), dtype=REAL_DTYPE)
        mapped = cp.einsum("Krc,qc->Krq", cspace.mesh.aff_mats, reference_points) + cspace.mesh.aff_vecs[:, :, None]
        sampled_exact = cp.asarray(exact(mapped[:, 0, :], mapped[:, 1, :]), dtype=REAL_DTYPE)
        sampled_numerical = coefficients @ sample_basis.T
    element_maximum = cp.max(cp.abs(sampled_numerical - sampled_exact), axis=1)
    # One compact transfer after the reductions; retain full samples only
    # when explicitly requested by the caller.
    packed = cp.asnumpy(cp.stack((l2, cp.max(element_maximum), cp.mean(element_maximum),
                                 cp.argmax(element_maximum).astype(cp.float64))))
    metrics = ScalarErrorMetrics(
        l2=float(packed[0]),
        linf=float(packed[1]),
        mean_element_linf=float(packed[2]),
        max_element=int(packed[3]),
    )
    samples = None if not include_samples else ScalarComparisonSamples(
        np.ascontiguousarray(cp.asnumpy(reference_points), dtype=REAL_DTYPE),
        np.ascontiguousarray(cp.asnumpy(sampled_numerical), dtype=REAL_DTYPE),
        np.ascontiguousarray(cp.asnumpy(sampled_exact), dtype=REAL_DTYPE),
    )
    return ScalarErrorReport(metrics, samples)


def _exact_vector_values(
        exact: Callable,
        x: np.ndarray,
        y: np.ndarray,
        *,
        dim: int,
) -> np.ndarray:
    """Evaluate and broadcast an exact vector formula to shape (d,K,q)."""
    target = x.shape
    raw = exact(x, y)
    if isinstance(raw, (tuple, list)):
        if len(raw) != dim:
            raise ValueError(f"exact vector must have {dim} components; got {len(raw)}")
        components = raw
    else:
        array = np.asarray(raw, dtype=REAL_DTYPE)
        if array.shape[:1] != (dim,):
            raise ValueError(
                f"exact vector must return {dim} components or an array "
                f"with leading dimension {dim}"
            )
        components = array
    normalized = []
    for component in components:
        values = np.asarray(component, dtype=REAL_DTYPE)
        if values.ndim == 0:
            values = np.full(target, float(values), dtype=REAL_DTYPE)
        else:
            try:
                values = np.broadcast_to(values, target)
            except ValueError as exc:
                raise ValueError(
                    f"exact vector component must broadcast to {target}; got {values.shape}"
                ) from exc
        normalized.append(values)
    return np.ascontiguousarray(np.stack(normalized, axis=0), dtype=REAL_DTYPE)


def evaluate_vector_error(
        field: VectorDGField,
        exact: Callable,
        *,
        volume_quad_1d: int | None = None,
        sample_resolution: int | None = None,
        include_samples: bool = False,
        weight: Callable | None = None,
        volume_degree: int | None = None,
) -> VectorErrorReport:
    """Evaluate vector L2 error and a sampled Euclidean maximum on the host.

    ``weight(x, y) >= 0`` weights only the L2 integral, for example ``R`` for
    the axisymmetric norm ``(int |e|^2 R dR dZ)^(1/2)``; sampled maxima stay
    unweighted.
    """
    if not isinstance(field, VectorDGField):
        raise TypeError("evaluate_vector_error expects a VectorDGField")
    space = field.components[0].space
    for component in field.components[1:]:
        space.assert_same_mesh(component.space)
        if component.space is not space:
            raise ValueError("vector components must share one DGSpace object")

    error_points, weights, basis = _error_quadrature(field.components[0], volume_quad_1d, volume_degree)
    mapped = space.mesh.map_reference_points(error_points)
    spatial = _weight_values(weight, mapped[:, :, 0], mapped[:, :, 1], np)
    exact_values = _exact_vector_values(
        exact,
        mapped[:, :, 0],
        mapped[:, :, 1],
        dim=field.dim,
    )
    coefficients = field.as_component_first()
    numerical_values = np.einsum("dKi,iq->dKq", coefficients, basis, optimize=True)
    difference = numerical_values - exact_values
    squared = difference * difference
    if spatial is not None:
        squared = squared * spatial[None]
    l2 = float(np.sqrt(np.einsum("K,dKq,q->", space.mesh.aff_jacs, squared, weights, optimize=True)))

    if sample_resolution is None:
        reference_points = error_points
        sampled_numerical = numerical_values
        sampled_exact = exact_values
    else:
        reference_points = _sample_reference_points(sample_resolution)
        mapped = space.mesh.map_reference_points(reference_points)
        sampled_exact = _exact_vector_values(
            exact,
            mapped[:, :, 0],
            mapped[:, :, 1],
            dim=field.dim,
        )
        sampled_numerical = np.einsum(
            "dKi,qi->dKq",
            coefficients,
            space.basis_at(reference_points),
            optimize=True,
        )
    sampled_difference = sampled_numerical - sampled_exact
    sampled_magnitude = np.sqrt(np.sum(sampled_difference * sampled_difference, axis=0))
    element_maximum = np.max(sampled_magnitude, axis=1)
    component_linf = tuple(
        float(value)
        for value in np.max(np.abs(sampled_difference), axis=(1, 2))
    )
    metrics = VectorErrorMetrics(
        l2=l2,
        linf=float(np.max(element_maximum)),
        component_linf=component_linf,
        mean_element_linf=float(np.mean(element_maximum)),
        max_element=int(np.argmax(element_maximum)),
    )
    samples = None if not include_samples else VectorComparisonSamples(
        np.ascontiguousarray(reference_points, dtype=REAL_DTYPE),
        np.ascontiguousarray(sampled_numerical, dtype=REAL_DTYPE),
        np.ascontiguousarray(sampled_exact, dtype=REAL_DTYPE),
    )
    return VectorErrorReport(metrics, samples)


def evaluate_scalar_error(
        field: DGField,
        exact: Callable,
        *,
        volume_quad_1d: int | None = None,
        sample_resolution: int | None = None,
        backend: Literal["auto", "host", "device"] = "auto",
        include_samples: bool = False,
        weight: Callable | None = None,
        volume_degree: int | None = None,
) -> ScalarErrorReport:
    """Evaluate scalar errors on the host or the field's resident GPU.

    ``weight(x, y) >= 0`` weights only the L2 integral, for example ``R`` for
    the axisymmetric norm ``(int |e|^2 R dR dZ)^(1/2)``; on the device it
    must accept CuPy arrays. Sampled maxima stay unweighted. ``volume_degree``
    selects an error rule of that polynomial exactness (exclusive with
    ``volume_quad_1d``).
    """
    if not isinstance(field, DGField):
        raise TypeError("evaluate_scalar_error expects a DGField")
    normalized = str(backend).lower()
    if normalized not in {"auto", "host", "device"}:
        raise ValueError("backend must be 'auto', 'host', or 'device'")
    use_device = normalized == "device" or (
        normalized == "auto" and field.device_coefficients_materialized()
    )
    evaluator = _evaluate_device if use_device else _evaluate_host
    return evaluator(
        field,
        exact,
        volume_quad_1d=volume_quad_1d,
        sample_resolution=sample_resolution,
        include_samples=include_samples,
        weight=weight,
        volume_degree=volume_degree,
    )


def evaluate_hdg_scalar_error(
        field: DGField,
        trace: np.ndarray,
        exact: Callable,
        exact_gradient: Callable,
        *,
        trace_basis: str = "legacy-lagrange",
        include_boundary: bool = True,
        chunk_size: int = 16384,
        backend: Literal["auto", "host", "device"] = "auto",
) -> ScalarHDGErrorMetrics:
    r"""Measure exact scalar error in the fixed-p, 1/h_K HDG H1 norm.

    Volume errors are integrated against the analytic value and physical
    gradient, not a projection of the exact field. For a smooth exact field
    with its own restriction as exact trace, (u_h-u)-(uhat_h-u)=u_h-uhat_h
    on each face. Thus the face term is exactly the numerical mismatch J.
    The supplied trace must be full, globally oriented, and correspond to
    the same time as the field. Boundary faces may be excluded where the
    discretization has no numerical trace. All reductions use float64 on
    the selected backend and the space's quadrature. Device-backed fields
    stay resident; only reduced scalar results are copied to the host.
    """
    from hybridge.hdg.gram import ScalarHDGGram

    space = field.space
    if backend not in {"auto", "host", "device"}:
        raise ValueError("backend must be 'auto', 'host', or 'device'")
    use_device = backend == "device" or (backend == "auto" and field.device_coefficients_materialized())
    gram = ScalarHDGGram(space, trace_basis=trace_basis, include_boundary=include_boundary,
                         chunk_size=chunk_size, backend="device" if use_device else "host")
    xp = gram.xp
    if use_device:
        from hybridge.core.device import as_cupy_coefficients

        q, mesh = gram.cspace.quad_data, gram.cspace.mesh
        coefficients = as_cupy_coefficients(field, gram.cspace, copy=False)
    else:
        q, mesh, coefficients = space.quad_data, space.mesh, field.coeffs
    phi = xp.asarray(q.phi, dtype=xp.float64)
    grad = xp.asarray(q.gphi, dtype=xp.float64)
    points = xp.asarray(q.Krf_quads, dtype=xp.float64)
    weights = xp.asarray(q.Krf_w, dtype=xp.float64)
    l2_squared = xp.asarray(0.0, dtype=xp.float64)
    gradient_squared = xp.asarray(0.0, dtype=xp.float64)
    for start in range(0, mesh.num_tri, chunk_size):
        selection = slice(start, start + chunk_size)
        c = xp.asarray(coefficients[selection], dtype=xp.float64)
        mapped = xp.einsum("krc,qc->kqr", mesh.aff_mats[selection], points)
        mapped += mesh.aff_vecs[selection, None, :]
        x, y = mapped[:, :, 0], mapped[:, :, 1]
        difference = c @ phi.T - xp.asarray(exact(x, y), dtype=xp.float64)
        numerical_gradient = xp.einsum("ki,qid->kqd", c, grad, optimize=True)
        numerical_gradient = xp.einsum("kqd,kdc->kqc", numerical_gradient,
                                       mesh.inv_aff_mats[selection], optimize=True)
        exact_components = exact_gradient(x, y)
        if len(exact_components) != 2:
            raise ValueError("exact_gradient must return two physical components")
        for axis, component in enumerate(exact_components):
            numerical_gradient[:, :, axis] -= xp.asarray(component, dtype=xp.float64)
        l2_squared += xp.einsum("k,kq,q->", mesh.aff_jacs[selection],
                               difference*difference, weights, optimize=True)
        gradient_squared += xp.einsum("k,kqc,q->", mesh.aff_jacs[selection],
                                     numerical_gradient*numerical_gradient, weights, optimize=True)
    mismatch_squared = gram.trace_mismatch_squared(coefficients, trace)
    values = xp.sqrt(xp.stack((l2_squared, gradient_squared)))
    values = values.get() if use_device else values
    return ScalarHDGErrorMetrics(float(values[0]), float(values[1]), float(np.sqrt(mismatch_squared)),
                                 backend=gram.backend)


__all__ = [
    "ScalarComparisonSamples",
    "ScalarErrorMetrics",
    "ScalarHDGErrorMetrics",
    "ScalarErrorReport",
    "VectorComparisonSamples",
    "VectorErrorMetrics",
    "VectorErrorReport",
    "evaluate_scalar_error",
    "evaluate_hdg_scalar_error",
    "evaluate_vector_error",
]
