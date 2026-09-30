"""Reusable error diagnostics for scalar and vector DG fields."""

from __future__ import annotations

from hdgfem.precision import audit_arrays, REAL_DTYPE

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .core.quadrature import ReferenceElementData
from .core.space import DGField, VectorDGField


def modal_activity(amplitudes, modes, *, relative_threshold=1e-3, top=3):
    """Rank active angular modes independently at each recorded time.

    Activity means positive finite amplitude at least relative_threshold
    times the current maximum. Rankings describe amplitude, without treating
    a large amplitude as an estimate of an exponential growth exponent.
    Missing and all-zero spectra produce no active modes.
    """
    values = np.atleast_2d(np.asarray(amplitudes, dtype=float))
    modes = np.asarray(modes)
    if values.ndim != 2 or modes.ndim != 1 or values.shape[1] != len(modes) or not len(modes):
        raise ValueError("amplitudes must have shape (samples, number of modes)")
    if not np.all(np.isfinite(modes)) or np.any(modes < 1) or np.any(modes != modes.astype(int)) or len(np.unique(modes)) != len(modes):
        raise ValueError("modes must be distinct positive integers")
    if not np.isfinite(relative_threshold) or not 0 <= relative_threshold <= 1 or int(top) != top or top < 1:
        raise ValueError("relative_threshold must be in [0, 1] and top a positive integer")
    clean = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    maxima = clean.max(axis=1, keepdims=True)
    relative = np.divide(clean, maxima, out=np.zeros_like(clean), where=maxima > 0)
    active = (clean > 0) & (relative >= relative_threshold)
    ranking = np.argsort(-clean, axis=1, kind="stable")[:, :min(int(top), len(modes))]
    valid = np.take_along_axis(active, ranking, axis=1)
    return {
        "dominant_modes": np.where(valid, modes[ranking], np.nan),
        "dominant_amplitudes": np.where(valid, np.take_along_axis(clean, ranking, axis=1), np.nan),
        "active_counts": active.sum(axis=1),
        "relative_amplitudes": np.where(np.isfinite(values), relative, np.nan),
    }


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


def relative_drift(value: float, baseline: float) -> float:
    """Return ``(value - baseline) / abs(baseline)`` with safe zero scaling."""
    scale = max(abs(float(baseline)), np.finfo(REAL_DTYPE).tiny)
    return (float(value) - float(baseline)) / scale


def result_transfer_time(result) -> float:
    """Sum timing details associated with host/device data movement."""
    details = getattr(getattr(result, "timings", None), "details", None) or {}
    return sum(
        float(value)
        for key, value in details.items()
        if "host" in str(key) or "to_device" in str(key) or "materialization" in str(key)
    )



def solver_diagnostics_snapshot(result):
    """Keep common HDG solve metrics without owning fields, matrices or factors.

    Rejected time stages still contribute iterations and timings. Their device
    systems can be released once the caller has copied any warm-start traces.
    The original result and its cached preconditioner are never mutated.
    """
    from copy import copy
    from types import SimpleNamespace

    global_solve = getattr(result, "global_solve_result", None)
    if global_solve is not None:
        global_solve = copy(global_solve)
        global_solve.x = global_solve.x_device = global_solve.preconditioner = None
    return SimpleNamespace(
        timings=result.timings, global_solve_result=global_solve,
        assembly_backend=result.assembly_backend, boundary_mode=result.boundary_mode,
        ordering_result=getattr(result, "ordering_result", None),
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
    timing_details = getattr(timings, "details", None) or {}
    rhs_only = bool(timing_details.get("raw.assembly.rhs_only", 0.0))
    operator_reused = bool(timing_details.get("raw.assembly.operator_reused", 0.0))
    row[f"{prefix}_time_rhs_assembly"] = timings.assembly if rhs_only else 0.0
    row[f"{prefix}_time_operator_assembly"] = (
        0.0 if operator_reused else timings.assembly
    )
    for key, value in timing_details.items():
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
        for key, attribute in {
            "retry_seed_label": "amgx_retry_seed_label",
            "retry_seed_physical_residual": "amgx_retry_seed_physical_residual",
            "retry_seed_physical_rhs_norm": "amgx_retry_seed_physical_rhs_norm",
            "retry_seed_physical_rel_residual": (
                "amgx_retry_seed_physical_relative_residual"
            ),
            "retry_seed_physical_target": "amgx_retry_seed_physical_target",
        }.items():
            if hasattr(global_solve, attribute):
                row[f"{prefix}_{key}"] = getattr(global_solve, attribute)
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


def _diagnostic_scalars(pending, xp) -> dict[str, float]:
    """Download one compact vector after verifying every metric is scalar."""
    if not pending:
        return {}
    scalars = [xp.asarray(value) for value in pending.values()]
    if any(value.ndim != 0 for value in scalars):
        raise ValueError("diagnostic outputs must be scalar reductions")
    packed = xp.stack(scalars)
    values = packed if xp is np else xp.asnumpy(packed)
    return {key: float(value) for key, value in zip(pending, values, strict=True)}


def _azimuthal_reductions(theta, perturbation, weights, equilibrium_integral, mode, xp):
    """Reduce three density harmonics in the selected array namespace."""
    normalization = xp.maximum(xp.abs(equilibrium_integral), xp.finfo(REAL_DTYPE).tiny)
    amplitudes = []
    for harmonic in (1, 2, 3):
        angle = float(harmonic * int(mode)) * theta
        cosine = xp.sum(perturbation * xp.cos(angle) * weights)
        sine = xp.sum(perturbation * xp.sin(angle) * weights)
        amplitudes.append(2.0 * xp.hypot(cosine, sine) / normalization)
    return {
        "diocotron_mode_base": xp.asarray(float(mode), dtype=REAL_DTYPE),
        "diocotron_mode_1k_amplitude": amplitudes[0],
        "diocotron_mode_2k_amplitude": amplitudes[1],
        "diocotron_mode_3k_amplitude": amplitudes[2],
        "diocotron_harmonic_ratio": amplitudes[1] / xp.maximum(amplitudes[0], xp.finfo(REAL_DTYPE).tiny),
    }


def _device_azimuthal_reductions(cspace, difference, equilibrium_integral, mode, cp):
    """Form density-mode moments using resident geometry and DG coefficients."""
    reference_points = cspace.quad_data.Krf_quads
    x = (
        cspace.mesh.aff_mats[:, 0, 0, None] * reference_points[None, :, 0]
        + cspace.mesh.aff_mats[:, 0, 1, None] * reference_points[None, :, 1]
        + cspace.mesh.aff_vecs[:, 0, None]
    )
    y = (
        cspace.mesh.aff_mats[:, 1, 0, None] * reference_points[None, :, 0]
        + cspace.mesh.aff_mats[:, 1, 1, None] * reference_points[None, :, 1]
        + cspace.mesh.aff_vecs[:, 1, None]
    )
    theta = cp.arctan2(y, x)
    del x, y
    perturbation = difference @ cspace.quad_data.bas_of_quads
    weights = cspace.mesh.aff_jacs[:, None] * cspace.quad_data.Krf_w[None, :]
    return _azimuthal_reductions(theta, perturbation, weights, equilibrium_integral, mode, cp)


def azimuthal_mode_diagnostics(
        field: DGField,
        equilibrium: DGField,
        mode: int,
        *,
        backend: Literal["auto", "host", "device"] = "auto",
) -> dict[str, float]:
    """Return density harmonics, downloading only five scalars on the device path.

    ``auto`` uses device reductions when either field has resident coefficients.
    Device equilibrium and density must use the same DGSpace. ``host`` permits
    explicit host materialization, matching the other diagnostic helpers.
    """
    normalized = str(backend).lower()
    if normalized not in {"auto", "host", "device"}:
        raise ValueError("backend must be 'auto', 'host', or 'device'")
    if int(mode) <= 0:
        return {}
    if hasattr(field.space, "assert_same_mesh"):
        field.space.assert_same_mesh(equilibrium.space)
    elif field.space is not equilibrium.space:
        raise ValueError("field and equilibrium must share the same diagnostic space")
    space = field.space
    use_device = normalized == "device" or (
        normalized == "auto"
        and any(isinstance(value, DGField) and value.device_coefficients_materialized()
                for value in (field, equilibrium))
    )
    if use_device:
        from .backends.cupy import as_cupy_coefficients, as_cupy_space, require_cupy

        if equilibrium.space is not space:
            raise ValueError("device equilibrium density must use the scalar DGSpace")
        cp = require_cupy()
        cspace = as_cupy_space(space)
        coefficients = as_cupy_coefficients(field, cspace)
        equilibrium_coefficients = as_cupy_coefficients(equilibrium, cspace)
        moments = cp.sum(cspace.quad_data.weighted_phi, axis=0)
        integral = cp.sum(cspace.mesh.aff_jacs * (equilibrium_coefficients @ moments))
        pending = _device_azimuthal_reductions(
            cspace, coefficients - equilibrium_coefficients, integral, mode, cp
        )
        return _diagnostic_scalars(pending, cp)
    points = space.mapped_quads()
    theta = np.arctan2(points[:, :, 1], points[:, :, 0])
    perturbation = np.asarray(field.values() - equilibrium.values(), dtype=REAL_DTYPE)
    weights = space.mesh.aff_jacs[:, None] * space.quad_data.Krf_w[None, :]
    integral = np.sum(equilibrium.values() * weights)
    return _diagnostic_scalars(
        _azimuthal_reductions(theta, perturbation, weights, integral, mode, np), np
    )



class ScalarPositivityDiagnostics:
    """Cached host/device polynomial bounds and sampled negative-density metrics.

    Bernstein coefficients bound the polynomial on each whole affine triangle.
    A negative lower bound alone is inconclusive; a negative sampled value is
    a witness. Floating-point bounds are interpreted with the stated tolerance.
    Negative mass/L2 use volume quadrature and are not exact negative-part integrals.
    No limiter or density modification is performed.
    """

    def __init__(self, space, *, backend="host", tolerance=1.e-12, chunk_size=8192):
        from .assembly.advection_residual import HDGTraceWorkspace
        from .core.basis import evaluate_bernstein_basis

        if not np.isfinite(tolerance) or tolerance < 0 or chunk_size < 1:
            raise ValueError("nonnegative finite tolerance and positive chunk_size required")
        self.space, self.tolerance, self.chunk_size = space, float(tolerance), int(chunk_size)
        self.workspace = HDGTraceWorkspace(space, backend=backend)
        self.xp = xp = self.workspace.xp
        order = space.order
        def lattice(n):
            return np.array([(-1+2*i/n, -1+2*j/n)
                             for i in range(n+1) for j in range(n+1-i)], dtype=REAL_DTYPE)
        nodes = lattice(max(order, 1))
        if order == 0:
            nodes = np.array([[-1/3, -1/3]], dtype=REAL_DTYPE)
        # One small reference-space conversion, reused for every cell and step.
        transform = np.linalg.solve(evaluate_bernstein_basis(order, nodes), space.reference.basis_at(nodes)).T
        sample_points = lattice(max(2*order+2, 2))
        with self.workspace._device_context():
            self.bernstein_transform = xp.asarray(transform)
            self.sample_basis = xp.asarray(space.reference.basis_at(sample_points).T)
            self.volume_basis = xp.asarray(space.quad_data.bas_of_quads)
            self.weights = xp.asarray(space.quad_data.Krf_w)
            self.weight_sum = float(space.quad_data.Krf_w.sum())
            self.jacobians = xp.asarray(space.mesh.aff_jacs)

    def measure(self, field):
        """Return small scalar diagnostics while keeping device coefficients resident."""
        if field.space is not self.space:
            raise ValueError("positivity diagnostics require their original DGSpace")
        with self.workspace._device_context():
            xp = self.xp
            if self.workspace.cspace is None:
                coefficients = field.coeffs
            else:
                from .backends.cupy import as_cupy_coefficients
                coefficients = as_cupy_coefficients(field, self.workspace.cspace)
            low, high = xp.asarray(np.inf), xp.asarray(-np.inf)
            lower, upper, mean_low = xp.asarray(np.inf), xp.asarray(-np.inf), xp.asarray(np.inf)
            negative_mass, negative_l2, negative_cells = xp.asarray(0.), xp.asarray(0.), xp.asarray(0.)
            for start in range(0, self.space.mesh.num_tri, self.chunk_size):
                stop = start+self.chunk_size
                c = coefficients[start:stop]
                volume = c @ self.volume_basis
                samples = c @ self.sample_basis
                cell_min = xp.minimum(volume.min(axis=1), samples.min(axis=1))
                low = xp.minimum(low, cell_min.min())
                high = xp.maximum(high, xp.maximum(volume.max(), samples.max()))
                negative_cells += xp.count_nonzero(cell_min < -self.tolerance)
                b = c @ self.bernstein_transform
                lower, upper = xp.minimum(lower, b.min()), xp.maximum(upper, b.max())
                averages = volume @ self.weights / self.weight_sum
                mean_low = xp.minimum(mean_low, averages.min())
                negative = xp.maximum(-volume, 0.)
                weight = self.jacobians[start:stop, None]*self.weights[None, :]
                negative_mass += xp.sum(negative*weight)
                negative_l2 += xp.sum(negative*negative*weight)
            packed = xp.stack([low, high, lower, upper, mean_low, negative_mass,
                               xp.sqrt(negative_l2), negative_cells])
            values = xp.asnumpy(packed) if self.workspace.cspace is not None else packed
        keys = ("rho_min_checked", "rho_max_checked", "rho_bernstein_lower_bound",
                "rho_bernstein_upper_bound", "rho_cell_average_min",
                "rho_negative_mass_quadrature", "rho_negative_l2_quadrature", "rho_negative_cells_sampled")
        result = {key: float(value) for key,value in zip(keys,values)}
        result["positivity_status"] = ("nonfinite" if not np.isfinite(values).all() else
            "violated" if values[0] < -self.tolerance else
            "bound_satisfied" if values[2] >= -self.tolerance else "inconclusive")
        result["positivity_tolerance"] = self.tolerance
        result["positivity_backend"] = self.workspace.backend
        return result


def guiding_center_field_diagnostics(
        density: DGField,
        potential: DGField,
        flux: VectorDGField,
        *,
        postprocessed_flux: VectorDGField | None = None,
        equilibrium_potential: DGField | None = None,
        equilibrium_density: DGField | None = None,
        mode: int = 0,
        backend: Literal["auto", "host", "device"] = "auto",
) -> dict[str, float | str]:
    """Reduce guiding-center field diagnostics on host or resident CUDA data.

    The device path works from DG coefficients, evaluates only the two fields
    needed for sampled extrema, computes all norms and integrals in coefficient
    space, and downloads one compact scalar vector.  It therefore avoids
    materializing full field/quadrature tables on the host.
    """
    if not isinstance(density, DGField) or not isinstance(potential, DGField):
        raise TypeError("density and potential must be DGField instances")
    if not isinstance(flux, VectorDGField):
        raise TypeError("flux must be a VectorDGField")
    normalized = str(backend).lower()
    if normalized not in {"auto", "host", "device"}:
        raise ValueError("backend must be 'auto', 'host', or 'device'")
    fields = [density, potential, *flux.components]
    if postprocessed_flux is not None:
        fields.extend(postprocessed_flux.components)
    if equilibrium_potential is not None:
        fields.append(equilibrium_potential)
    if equilibrium_density is not None:
        fields.append(equilibrium_density)
    use_device = normalized == "device" or (
        normalized == "auto"
        and any(field.device_coefficients_materialized() for field in fields)
    )
    if not use_device:
        standard_q_l2 = flux.l2_norm()
        rho_min, rho_max = density.min_max()
        phi_min, phi_max = potential.min_max()
        result: dict[str, float | str] = {
            "mass": density.integral(),
            "rho_min": rho_min,
            "rho_max": rho_max,
            "phi_min": phi_min,
            "phi_max": phi_max,
            "q_l2_standard": standard_q_l2,
            "rho_l2_squared": density.l2_norm()**2,
            "diagnostics_backend": "host",
        }
        if postprocessed_flux is not None:
            result["q_l2_postprocessed"] = postprocessed_flux.l2_norm()
        if equilibrium_potential is not None:
            result.update({
                "diocotron_phi_eq_l2": potential.space.l2_diff(
                    potential, equilibrium_potential
                ),
                "diocotron_phi_eq_linf": potential.space.linf_diff(
                    potential, equilibrium_potential
                ),
                "diocotron_phi_eq_reference_l2": equilibrium_potential.l2_norm(),
            })
        if equilibrium_density is not None:
            result.update({
                "diocotron_rho_eq_l2": density.space.l2_diff(
                    density, equilibrium_density
                ),
                "diocotron_rho_eq_reference_l2": equilibrium_density.l2_norm(),
            })
            result.update(azimuthal_mode_diagnostics(density, equilibrium_density, mode, backend="host"))
        return result

    from .backends.cupy import as_cupy_coefficients, as_cupy_space, require_cupy

    cp = require_cupy()
    base_space = density.space
    base_space.assert_same_mesh(potential.space)
    cspace = as_cupy_space(base_space)
    potential_cspace = as_cupy_space(potential.space, device=cspace.device_id)
    jacobians = cspace.mesh.aff_jacs
    reference_moments = cp.sum(cspace.quad_data.weighted_phi, axis=0)
    density_coeffs = as_cupy_coefficients(density, cspace)
    potential_coeffs = as_cupy_coefficients(potential, potential_cspace)

    pending: dict[str, object] = {}

    def l2_squared(coefficients, local_cspace):
        """Reduce the physical squared L2 norm of resident coefficients."""
        weighted = coefficients @ local_cspace.quad_data.MKrf
        return cp.sum(
            local_cspace.mesh.aff_jacs
            * cp.sum(coefficients * weighted, axis=1)
        )

    density_integral = cp.sum(
        jacobians * (density_coeffs @ reference_moments)
    )
    density_values = density_coeffs @ cspace.quad_data.bas_of_quads
    potential_values = potential_coeffs @ potential_cspace.quad_data.bas_of_quads
    pending["mass"] = density_integral
    pending["rho_l2_squared"] = cp.maximum(l2_squared(density_coeffs, cspace), 0.0)
    pending["rho_min"] = cp.min(density_values)
    pending["rho_max"] = cp.max(density_values)
    pending["phi_min"] = cp.min(potential_values)
    pending["phi_max"] = cp.max(potential_values)
    del density_values, potential_values

    flux_l2_squared = cp.asarray(0.0, dtype=REAL_DTYPE)
    for component in flux.components:
        base_space.assert_same_mesh(component.space)
        component_cspace = as_cupy_space(component.space, device=cspace.device_id)
        flux_l2_squared = flux_l2_squared + l2_squared(
            as_cupy_coefficients(component, component_cspace), component_cspace
        )
    pending["q_l2_standard"] = cp.sqrt(cp.maximum(flux_l2_squared, 0.0))

    if postprocessed_flux is not None:
        post_l2_squared = cp.asarray(0.0, dtype=REAL_DTYPE)
        for component in postprocessed_flux.components:
            base_space.assert_same_mesh(component.space)
            component_cspace = as_cupy_space(component.space, device=cspace.device_id)
            post_l2_squared = post_l2_squared + l2_squared(
                as_cupy_coefficients(component, component_cspace), component_cspace
            )
        pending["q_l2_postprocessed"] = cp.sqrt(cp.maximum(post_l2_squared, 0.0))

    if equilibrium_potential is not None:
        if equilibrium_potential.space is not potential.space:
            raise ValueError("device equilibrium potential must use the potential DGSpace")
        equilibrium_phi_coeffs = as_cupy_coefficients(equilibrium_potential, potential_cspace)
        phi_difference = potential_coeffs - equilibrium_phi_coeffs
        pending["diocotron_phi_eq_l2"] = cp.sqrt(
            cp.maximum(l2_squared(phi_difference, potential_cspace), 0.0)
        )
        phi_difference_values = phi_difference @ potential_cspace.quad_data.bas_of_quads
        pending["diocotron_phi_eq_linf"] = cp.max(cp.abs(phi_difference_values))
        pending["diocotron_phi_eq_reference_l2"] = cp.sqrt(
            cp.maximum(l2_squared(equilibrium_phi_coeffs, potential_cspace), 0.0)
        )
        del phi_difference_values

    if equilibrium_density is not None:
        if equilibrium_density.space is not base_space:
            raise ValueError("device equilibrium density must use the scalar DGSpace")
        equilibrium_rho_coeffs = as_cupy_coefficients(equilibrium_density, cspace)
        rho_difference = density_coeffs - equilibrium_rho_coeffs
        pending["diocotron_rho_eq_l2"] = cp.sqrt(
            cp.maximum(l2_squared(rho_difference, cspace), 0.0)
        )
        equilibrium_integral = cp.sum(
            jacobians * (equilibrium_rho_coeffs @ reference_moments)
        )
        pending["diocotron_rho_eq_reference_l2"] = cp.sqrt(
            cp.maximum(l2_squared(equilibrium_rho_coeffs, cspace), 0.0)
        )
        if int(mode) > 0:
            pending.update(_device_azimuthal_reductions(
                cspace, rho_difference, equilibrium_integral, mode, cp
            ))

    audit_arrays('diagnostic-reductions', pending, cspace)
    result = _diagnostic_scalars(pending, cp)
    result["diagnostics_backend"] = "cuda"
    return result

def transport_velocity_diagnostics(
        velocity: VectorDGField,
        *,
        backend: Literal["auto", "host", "device"] = "auto",
) -> dict[str, float | str]:
    """Measure compatibility of a 2D DG transport velocity on the mesh faces.

    Boundary normal flux and interior jumps use the actual polygonal mesh
    normals. Jumps sum the two outward normal traces at aligned quadrature
    points. Divergence is the physical, elementwise polynomial derivative.
    Maxima are sampled, not rigorous bounds. If passed a stage coefficient
    beta=c*v, every absolute norm is scaled by abs(c).

    Device fields stay resident; only the final scalar reductions are copied
    to the host. These diagnostics do not alter the velocity or its fluxes.
    """
    if not isinstance(velocity, VectorDGField) or len(velocity.components) != 2:
        raise TypeError("velocity must be a two-component VectorDGField")
    space = velocity.components[0].space
    if any(component.space is not space for component in velocity.components):
        raise ValueError("velocity components must share one scalar DGSpace")
    normalized = str(backend).lower()
    if normalized not in {"auto", "host", "device"}:
        raise ValueError("backend must be 'auto', 'host', or 'device'")
    use_device = normalized == "device" or (
        normalized == "auto"
        and any(component.device_coefficients_materialized() for component in velocity.components)
    )
    if use_device:
        from .backends.cupy import as_cupy_coefficients, as_cupy_space, require_cupy

        xp = require_cupy()
        cspace = as_cupy_space(space)
        mesh, quad = cspace.mesh, cspace.quad_data
        coefficients = [as_cupy_coefficients(component, cspace) for component in velocity.components]
    else:
        xp = np
        mesh, quad = space.mesh, space.quad_data
        coefficients = [component.coeffs for component in velocity.components]

    trace = space.trace_space("legendre-modal")
    t = trace.quads
    ones = np.ones_like(t)
    points = np.stack((np.stack((t, -ones), axis=1),
                       np.stack((-t, t), axis=1),
                       np.stack((-ones, -t), axis=1)))
    face_basis = xp.asarray(space.basis_at(points.reshape(-1, 2)).reshape(3, t.size, -1))
    face_values = [xp.einsum("ki,fqi->kfq", coeff, face_basis) for coeff in coefficients]
    normal = face_values[0] * mesh.normals[:, :, 0, None] + face_values[1] * mesh.normals[:, :, 1, None]
    face_speed_squared = face_values[0]**2 + face_values[1]**2
    weights = mesh.jacs_el_fc[:, :, None] * xp.asarray(trace.weights)
    boundary_normal = xp.where(mesh.interior_face_mask[:, :, None], 0.0, normal)
    boundary_speed_squared = xp.where(mesh.interior_face_mask[:, :, None], 0.0, face_speed_squared)
    boundary_l2 = xp.sqrt(xp.sum(weights * boundary_normal**2))
    boundary_speed_l2 = xp.sqrt(xp.sum(weights * boundary_speed_squared))

    # Local face orientations differ on the two sides of an interior edge.
    aligned = xp.where(mesh.orientations[:, :, None], normal, normal[:, :, ::-1])
    jumps = xp.zeros((mesh.num_edg, t.size), dtype=coefficients[0].dtype)
    xp.add.at(jumps, mesh.loc2glob_edge.reshape(-1), aligned.reshape(-1, t.size))
    jumps[mesh.bnd_edges_inds] = 0.0
    jump_weights = mesh.edge_jacs[:, None] * xp.asarray(trace.weights)

    divergence = xp.zeros((mesh.num_tri, quad.Krf_w.size), dtype=coefficients[0].dtype)
    speed_squared = xp.zeros_like(divergence)
    for axis, coeff in enumerate(coefficients):
        reference_gradient = xp.einsum("ki,qid->kqd", coeff, quad.gphi)
        divergence += xp.einsum("kd,kqd->kq", mesh.inv_aff_mats_t[:, axis, :], reference_gradient)
        speed_squared += (coeff @ quad.bas_of_quads)**2
    volume_weights = mesh.aff_jacs[:, None] * quad.Krf_w
    # Dimensionless for beta=dt*v; a diagnostic, not a timestep stability bound.
    cell_speed = xp.sqrt(xp.maximum(xp.max(speed_squared, axis=1), xp.max(face_speed_squared, axis=(1, 2))))
    min_edge_length = 2.0 * xp.min(mesh.jacs_el_fc, axis=1)
    pending = {
        "velocity_boundary_normal_l2": boundary_l2,
        "velocity_boundary_normal_linf": xp.max(xp.abs(boundary_normal)),
        "velocity_boundary_speed_l2": boundary_speed_l2,
        "velocity_boundary_normal_relative_l2": boundary_l2 / xp.maximum(boundary_speed_l2, xp.finfo(coefficients[0].dtype).tiny),
        "velocity_normal_jump_l2": xp.sqrt(xp.sum(jump_weights * jumps**2)),
        "velocity_normal_jump_linf": xp.max(xp.abs(jumps)),
        "velocity_divergence_l2": xp.sqrt(xp.sum(volume_weights * divergence**2)),
        "velocity_divergence_linf": xp.max(xp.abs(divergence)),
        "velocity_speed_linf": xp.max(cell_speed),
        "velocity_max_speed_over_min_edge": xp.max(cell_speed / min_edge_length),
    }
    result = _diagnostic_scalars(pending, xp)
    result["velocity_diagnostics_backend"] = "cuda" if use_device else "host"
    return result


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
    from .io.plot import reference_plot_points

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
    from .backends.cupy import as_cupy_coefficients, as_cupy_space, require_cupy

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
    from .assembly.hdg_gram import ScalarHDGGram

    space = field.space
    if backend not in {"auto", "host", "device"}:
        raise ValueError("backend must be 'auto', 'host', or 'device'")
    use_device = backend == "device" or (backend == "auto" and field.device_coefficients_materialized())
    gram = ScalarHDGGram(space, trace_basis=trace_basis, include_boundary=include_boundary,
                         chunk_size=chunk_size, backend="device" if use_device else "host")
    xp = gram.xp
    if use_device:
        from .backends.cupy import as_cupy_coefficients

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
    "modal_activity",
    "ScalarComparisonSamples",
    "ScalarErrorMetrics",
    "ScalarPositivityDiagnostics",
    "ScalarHDGErrorMetrics",
    "ScalarErrorReport",
    "VectorComparisonSamples",
    "VectorErrorMetrics",
    "VectorErrorReport",
    "azimuthal_mode_diagnostics",
    "guiding_center_field_diagnostics",
    "evaluate_scalar_error",
    "evaluate_hdg_scalar_error",
    "evaluate_vector_error",
    "relative_drift",
    "result_transfer_time",
    "solver_result_metrics",
    "solver_diagnostics_snapshot",
]
