"""Reusable error diagnostics for scalar and vector DG fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .core.quadrature import ReferenceElementData
from .core.space import DGField, VectorDGField


@dataclass(frozen=True)
class ScalarErrorMetrics:
    """Scalar DG error norms and per-element maximum-error information."""

    l2: float
    linf: float
    mean_element_linf: float
    max_element: int


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


def relative_drift(value: float, baseline: float) -> float:
    """Return ``(value - baseline) / abs(baseline)`` with safe zero scaling."""
    scale = max(abs(float(baseline)), np.finfo(np.float64).tiny)
    return (float(value) - float(baseline)) / scale


def result_transfer_time(result) -> float:
    """Sum timing details associated with host/device data movement."""
    details = getattr(getattr(result, "timings", None), "details", None) or {}
    return sum(
        float(value)
        for key, value in details.items()
        if "host" in str(key) or "to_device" in str(key) or "materialization" in str(key)
    )


def solver_result_metrics(prefix: str, result) -> dict[str, object]:
    """Flatten common HDG result, timing, linear-solve, and ordering metrics."""
    timings = result.timings
    global_solve = result.global_solve_result
    row: dict[str, object] = {
        f"{prefix}_assembly_backend": result.assembly_backend,
        f"{prefix}_boundary_mode": result.boundary_mode,
        f"{prefix}_time_total": timings.total,
        f"{prefix}_time_assembly": timings.assembly,
        f"{prefix}_time_solve": timings.solve,
        f"{prefix}_time_reconstruction": timings.reconstruction,
        f"{prefix}_host_device_transfer_time": result_transfer_time(result),
    }
    for attribute in ("trace_ordering", "postprocessing"):
        if hasattr(timings, attribute):
            row[f"{prefix}_time_{attribute}"] = getattr(timings, attribute)
    for key, value in (getattr(timings, "details", None) or {}).items():
        if isinstance(value, (int, float)):
            safe_key = "".join(ch if ch.isalnum() else "_" for ch in str(key)).strip("_")
            row[f"{prefix}_detail_{safe_key}"] = float(value)
    if global_solve is not None:
        attempts = getattr(global_solve, "amgx_attempts", None)
        if attempts is not None:
            row[f"{prefix}_amgx_attempt_count"] = int(
                getattr(global_solve, "amgx_attempt_count", len(attempts))
            )
            row[f"{prefix}_amgx_attempts"] = list(attempts)
        attributes = {
            "solver_residual": "solver_residual_norm",
            "solver_rhs_norm": "solver_rhs_norm",
            "solver_residual_target": "solver_residual_target",
            "solver_rel_residual": "solver_relative_residual_norm",
            "physical_residual": "physical_residual_norm",
            "physical_rhs_norm": "physical_rhs_norm",
            "physical_residual_target": "physical_residual_target",
            "physical_rel_residual": "physical_relative_residual_norm",
            "diagnostic_residual": "diagnostic_residual_norm",
            "diagnostic_residual_target": "diagnostic_residual_target",
            "diagnostic_rel_residual": "diagnostic_relative_residual_norm",
            "preconditioner_time": "preconditioner_elapsed_seconds",
            "krylov_time": "solve_elapsed_seconds",
            "matrix_csr_time": "matrix_assembly_elapsed_seconds",
            "solver_global_time": "global_elapsed_seconds",
            "scale_time": "scale_elapsed_seconds",
            "initial_residual_time": "initial_residual_elapsed_seconds",
            "callback_time": "callback_elapsed_seconds",
            "final_residual_time": "residual_diagnostics_elapsed_seconds",
            "preconditioner_apply_count": "preconditioner_apply_count",
            "permutation_time": "permutation_elapsed_seconds",
            "permutation_size": "permutation_size",
            "ilu_permc_spec": "ilu_permc_spec",
            "preconditioner_apply_time": "preconditioner_apply_seconds",
            "preconditioner_factor_nnz": "preconditioner_factor_nnz",
        }
        row[f"{prefix}_solver_iterations"] = (
            -1 if global_solve.iteration_count is None else global_solve.iteration_count
        )
        row.update({f"{prefix}_{key}": getattr(global_solve, attribute) for key, attribute in attributes.items()})
    ordering = getattr(result, "ordering_result", None)
    if ordering is not None:
        diagnostics = ordering.diagnostics
        widths = diagnostics.level_widths
        ordering_values = {
            "nodes": diagnostics.num_nodes,
            "directed_edges": diagnostics.num_directed_edges,
            "components": diagnostics.num_components,
            "largest_scc": diagnostics.largest_component_size,
            "cyclic_components": diagnostics.cyclic_components,
            "cyclic_nodes": diagnostics.cyclic_nodes,
            "levels": widths.num_levels,
            "max_level_width": widths.max_width,
            "median_level_width": widths.median_width,
            "mean_level_width": widths.mean_width,
            "top10_width_fraction": widths.top10_width_fraction,
        }
        row.update({f"{prefix}_ordering_{key}": value for key, value in ordering_values.items()})
    return row


def azimuthal_mode_diagnostics(field: DGField, equilibrium: DGField, mode: int) -> dict[str, float]:
    """Return normalized base, second, and third azimuthal harmonic amplitudes."""
    if int(mode) <= 0:
        return {}
    if hasattr(field.space, "assert_same_mesh"):
        field.space.assert_same_mesh(equilibrium.space)
    elif field.space is not equilibrium.space:
        raise ValueError("field and equilibrium must share the same diagnostic space")
    space = field.space
    points = space.mapped_quads()
    theta = np.arctan2(points[:, :, 1], points[:, :, 0])
    perturbation = np.asarray(field.values() - equilibrium.values(), dtype=np.float64)
    weights = space.mesh.aff_jacs[:, None] * space.quad_data.Krf_w[None, :]
    normalization = max(abs(float(np.sum(equilibrium.values() * weights))), np.finfo(np.float64).tiny)
    amplitudes = []
    for harmonic in (1, 2, 3):
        angle = float(harmonic * int(mode)) * theta
        cosine = float(np.sum(perturbation * np.cos(angle) * weights))
        sine = float(np.sum(perturbation * np.sin(angle) * weights))
        amplitudes.append(2.0 * float(np.hypot(cosine, sine)) / normalization)
    return {
        "diocotron_mode_base": float(mode),
        "diocotron_mode_1k_amplitude": amplitudes[0],
        "diocotron_mode_2k_amplitude": amplitudes[1],
        "diocotron_mode_3k_amplitude": amplitudes[2],
        "diocotron_harmonic_ratio": amplitudes[1] / max(amplitudes[0], np.finfo(np.float64).tiny),
    }


def _error_quadrature(field: DGField, volume_quad_1d: int | None):
    """Return reference points, weights, and basis values for error integration."""
    space = field.space
    if volume_quad_1d is None:
        return space.quad_data.Krf_quads, space.quad_data.Krf_w, space.quad_data.bas_of_quads
    reference = ReferenceElementData.triangle(
        space.order,
        basis_type=space.quad_data.basis_type,
        volume_quad_1d=int(volume_quad_1d),
        edge_quad_1d=space.quad_data.edge_quad_1d,
    )
    return reference.Krf_quads, reference.Krf_w, reference.bas_of_quads


def _sample_reference_points(resolution: int) -> np.ndarray:
    """Return the standard triangular plotting sample grid."""
    from .io.plot import reference_plot_points

    return reference_plot_points(int(resolution))


def _evaluate_host(field, exact, *, volume_quad_1d, sample_resolution, include_samples):
    """Evaluate scalar metrics and optional samples with NumPy."""
    space = field.space
    error_points, weights, basis = _error_quadrature(field, volume_quad_1d)
    mapped = space.mesh.map_reference_points(error_points)
    exact_values = np.asarray(exact(mapped[:, :, 0], mapped[:, :, 1]), dtype=np.float64)
    numerical_values = field.coeffs @ basis
    diff = numerical_values - exact_values
    l2 = float(np.sqrt(np.einsum("K,Kq,q->", space.mesh.aff_jacs, diff * diff, weights, optimize=True)))
    if sample_resolution is None:
        reference_points, sampled_numerical, sampled_exact = error_points, numerical_values, exact_values
    else:
        reference_points = _sample_reference_points(sample_resolution)
        mapped = space.mesh.map_reference_points(reference_points)
        sampled_exact = np.asarray(exact(mapped[:, :, 0], mapped[:, :, 1]), dtype=np.float64)
        sampled_numerical = field.coeffs @ space.basis_at(reference_points).T
    element_maximum = np.max(np.abs(sampled_numerical - sampled_exact), axis=1)
    metrics = ScalarErrorMetrics(
        l2=l2,
        linf=float(np.max(element_maximum)),
        mean_element_linf=float(np.mean(element_maximum)),
        max_element=int(np.argmax(element_maximum)),
    )
    samples = None if not include_samples else ScalarComparisonSamples(
        np.ascontiguousarray(reference_points, dtype=np.float64),
        np.ascontiguousarray(sampled_numerical, dtype=np.float64),
        np.ascontiguousarray(sampled_exact, dtype=np.float64),
    )
    return ScalarErrorReport(metrics, samples)


def _evaluate_device(field, exact, *, volume_quad_1d, sample_resolution, include_samples):
    """Evaluate scalar metrics on the resident CUDA device with CuPy."""
    from .backends.cupy import as_cupy_coefficients, as_cupy_space, require_cupy

    cp = require_cupy()
    space = field.space
    cspace = as_cupy_space(space)
    coefficients = as_cupy_coefficients(field, cspace)
    if volume_quad_1d is None:
        error_points = cspace.quad_data.Krf_quads
        weights = cspace.quad_data.Krf_w
        basis = cspace.quad_data.bas_of_quads
    else:
        host_points, host_weights, host_basis = _error_quadrature(field, volume_quad_1d)
        error_points = cp.asarray(host_points, dtype=cp.float64)
        weights = cp.asarray(host_weights, dtype=cp.float64)
        basis = cp.asarray(host_basis, dtype=cp.float64)
    mapped = cp.einsum("Krc,qc->Krq", cspace.mesh.aff_mats, error_points) + cspace.mesh.aff_vecs[:, :, None]
    exact_values = cp.asarray(exact(mapped[:, 0, :], mapped[:, 1, :]), dtype=cp.float64)
    numerical_values = coefficients @ basis
    diff = numerical_values - exact_values
    l2 = cp.sqrt(cp.einsum("K,Kq,q->", cspace.mesh.aff_jacs, diff * diff, weights, optimize=True))
    if sample_resolution is None:
        reference_points, sampled_numerical, sampled_exact = error_points, numerical_values, exact_values
    else:
        host_reference_points = _sample_reference_points(sample_resolution)
        reference_points = cp.asarray(host_reference_points, dtype=cp.float64)
        sample_basis = cp.asarray(space.basis_at(host_reference_points), dtype=cp.float64)
        mapped = cp.einsum("Krc,qc->Krq", cspace.mesh.aff_mats, reference_points) + cspace.mesh.aff_vecs[:, :, None]
        sampled_exact = cp.asarray(exact(mapped[:, 0, :], mapped[:, 1, :]), dtype=cp.float64)
        sampled_numerical = coefficients @ sample_basis.T
    element_maximum = cp.max(cp.abs(sampled_numerical - sampled_exact), axis=1)
    cp.cuda.get_current_stream().synchronize()
    metrics = ScalarErrorMetrics(
        l2=float(l2.get()),
        linf=float(cp.max(element_maximum).get()),
        mean_element_linf=float(cp.mean(element_maximum).get()),
        max_element=int(cp.argmax(element_maximum).get()),
    )
    samples = None if not include_samples else ScalarComparisonSamples(
        np.ascontiguousarray(cp.asnumpy(reference_points), dtype=np.float64),
        np.ascontiguousarray(cp.asnumpy(sampled_numerical), dtype=np.float64),
        np.ascontiguousarray(cp.asnumpy(sampled_exact), dtype=np.float64),
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
        array = np.asarray(raw, dtype=np.float64)
        if array.shape[:1] != (dim,):
            raise ValueError(
                f"exact vector must return {dim} components or an array "
                f"with leading dimension {dim}"
            )
        components = array
    normalized = []
    for component in components:
        values = np.asarray(component, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(target, float(values), dtype=np.float64)
        else:
            try:
                values = np.broadcast_to(values, target)
            except ValueError as exc:
                raise ValueError(
                    f"exact vector component must broadcast to {target}; got {values.shape}"
                ) from exc
        normalized.append(values)
    return np.ascontiguousarray(np.stack(normalized, axis=0), dtype=np.float64)


def evaluate_vector_error(
        field: VectorDGField,
        exact: Callable,
        *,
        volume_quad_1d: int | None = None,
        sample_resolution: int | None = None,
        include_samples: bool = False,
) -> VectorErrorReport:
    """Evaluate vector L2 error and a sampled Euclidean maximum on the host."""
    if not isinstance(field, VectorDGField):
        raise TypeError("evaluate_vector_error expects a VectorDGField")
    space = field.components[0].space
    for component in field.components[1:]:
        space.assert_same_mesh(component.space)
        if component.space is not space:
            raise ValueError("vector components must share one DGSpace object")

    error_points, weights, basis = _error_quadrature(field.components[0], volume_quad_1d)
    mapped = space.mesh.map_reference_points(error_points)
    exact_values = _exact_vector_values(
        exact,
        mapped[:, :, 0],
        mapped[:, :, 1],
        dim=field.dim,
    )
    coefficients = field.as_component_first()
    numerical_values = np.einsum("dKi,iq->dKq", coefficients, basis, optimize=True)
    difference = numerical_values - exact_values
    l2 = float(
        np.sqrt(
            np.einsum(
                "K,dKq,q->",
                space.mesh.aff_jacs,
                difference * difference,
                weights,
                optimize=True,
            )
        )
    )

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
        np.ascontiguousarray(reference_points, dtype=np.float64),
        np.ascontiguousarray(sampled_numerical, dtype=np.float64),
        np.ascontiguousarray(sampled_exact, dtype=np.float64),
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
) -> ScalarErrorReport:
    """Evaluate scalar errors on the host or the field's resident GPU."""
    if not isinstance(field, DGField):
        raise TypeError("evaluate_scalar_error expects a DGField")
    normalized = str(backend).lower()
    if normalized not in {"auto", "host", "device"}:
        raise ValueError("backend must be 'auto', 'host', or 'device'")
    use_device = normalized == "device" or (
        normalized == "auto" and field.device_coefficients_materialized() and not field.coefficients_materialized
    )
    evaluator = _evaluate_device if use_device else _evaluate_host
    return evaluator(
        field,
        exact,
        volume_quad_1d=volume_quad_1d,
        sample_resolution=sample_resolution,
        include_samples=include_samples,
    )


__all__ = [
    "ScalarComparisonSamples",
    "ScalarErrorMetrics",
    "ScalarErrorReport",
    "VectorComparisonSamples",
    "VectorErrorMetrics",
    "VectorErrorReport",
    "azimuthal_mode_diagnostics",
    "evaluate_scalar_error",
    "evaluate_vector_error",
    "relative_drift",
    "result_transfer_time",
    "solver_result_metrics",
]
