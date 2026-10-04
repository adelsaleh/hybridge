"""hdgfem.diagnostics.guiding_center."""

from __future__ import annotations

import numpy as np
from hdgfem.core.space import DGField, VectorDGField
from typing import Literal
from hdgfem.runtime.precision import REAL_DTYPE, audit_arrays


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
        from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
        from hdgfem.runtime.optional import require_cupy

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
        from hdgfem.transport.residual import HDGTraceWorkspace
        from hdgfem.core.basis import evaluate_bernstein_basis

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
                from hdgfem.core.device import as_cupy_coefficients
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

    from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
    from hdgfem.runtime.optional import require_cupy

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
    Double-outflow (both outward traces nonnegative, positive sum) and
    double-inflow fractions classify aligned interior nodes exactly as
    conflict-averaged upwinding does, by face measure and by face count.
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
        from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
        from hdgfem.runtime.optional import require_cupy

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

    # Upwind classification of the two outward traces at aligned interior
    # nodes. Double outflow is what conflict-averaged upwinding repairs.
    slots = mesh.edge_side_indices[mesh.int_edges_inds]
    sides = normal.reshape(-1, t.size)
    side_orientations = mesh.orientations.reshape(-1)
    left = xp.where(side_orientations[slots[:, 0], None], sides[slots[:, 0]], sides[slots[:, 0], ::-1])
    right = xp.where(side_orientations[slots[:, 1], None], sides[slots[:, 1]], sides[slots[:, 1], ::-1])
    double_outflow = (left >= 0) & (right >= 0) & ((left + right) > 0)
    double_inflow = (left < 0) & (right < 0)
    interior_weights = jump_weights[mesh.int_edges_inds]
    interior_measure = xp.sum(interior_weights)

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
        "velocity_double_outflow_measure_fraction":
            xp.sum(interior_weights * double_outflow) / interior_measure,
        "velocity_double_outflow_face_fraction":
            xp.mean(xp.any(double_outflow, axis=1).astype(coefficients[0].dtype)),
        "velocity_double_inflow_measure_fraction":
            xp.sum(interior_weights * double_inflow) / interior_measure,
        "velocity_divergence_l2": xp.sqrt(xp.sum(volume_weights * divergence**2)),
        "velocity_divergence_linf": xp.max(xp.abs(divergence)),
        "velocity_speed_linf": xp.max(cell_speed),
        "velocity_max_speed_over_min_edge": xp.max(cell_speed / min_edge_length),
    }
    result = _diagnostic_scalars(pending, xp)
    result["velocity_diagnostics_backend"] = "cuda" if use_device else "host"
    return result
