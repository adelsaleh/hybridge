"""Residency-preserving field and trace operations for solver orchestration."""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE

import numpy as np

from hdgfem.core.space import DGField, DGSpace, VectorDGField


def coefficient_field(space: DGSpace, value, *, name: str = "coefficient_h") -> DGField:
    """Represent a scalar, callable, array, or existing field in ``space``."""
    if isinstance(value, DGField):
        value.space.assert_same_mesh(space)
        if value.space is not space:
            raise ValueError("an existing coefficient field must live in the requested DGSpace object")
        return value
    if callable(value):
        return space.project_callable(value, name=name)
    if np.isscalar(value):
        return space.constant(float(value), name=name)
    return space.field(value, name=name)


def solution_field(result, space: DGSpace, *, name: str = "u_h") -> DGField:
    """Return a host or lazily device-backed field from a solver result."""
    field = getattr(result, "field", None)
    if field is not None:
        return field
    coefficients = getattr(result, "field_device", None)
    if coefficients is None:
        raise RuntimeError("solver result contains neither a host field nor device field coefficients")
    from hdgfem.core.device import field_from_cupy_coefficients

    device_id = int(getattr(getattr(coefficients, "device", None), "id", 0))
    return field_from_cupy_coefficients(space, coefficients, device=device_id, name=name)


def solution_trace(result, space: DGSpace, *, reduced: bool = False, prefer_device: bool = True):
    """Return a solver result trace in a reusable host/device representation.

    A device result normally stores only the reduced interior trace. When
    ``prefer_device`` is true that array is returned without materialization,
    irrespective of ``reduced``. Otherwise the full trace may be restricted
    to interior edges. A full device trace stays on device unless
    ``prefer_device=False`` explicitly requests a download.
    """
    device_trace = getattr(result, "trace_reduced_device", None)
    if prefer_device and device_trace is not None:
        return device_trace
    trace = getattr(result, "trace", None)
    if trace is None:
        return device_trace
    if hasattr(trace, "__cuda_array_interface__"):
        from hdgfem.core.device import as_cupy_space
        from hdgfem.runtime.optional import require_cupy
        cp = require_cupy()
        trace_array = cp.asarray(trace, dtype=REAL_DTYPE)
        if reduced:
            indices = as_cupy_space(space).mesh.int_edges_inds
            trace_array = trace_array.reshape(space.mesh.num_edg, -1)[indices]
        trace_array = cp.ascontiguousarray(trace_array.reshape(-1))
        return trace_array if prefer_device else cp.asnumpy(trace_array)
    trace_array = np.asarray(trace, dtype=REAL_DTYPE)
    if reduced:
        trace_array = trace_array.reshape(space.mesh.num_edg, -1)[space.mesh.int_edges_inds]
    return np.ascontiguousarray(trace_array.reshape(-1))


def field_linear_combination(
        space: DGSpace,
        terms,
        *,
        name: str = "u_h",
) -> DGField:
    """Combine nested-basis fields in ``space`` without host staging."""
    normalized = [(float(weight), field) for weight, field in terms]
    if not normalized:
        raise ValueError("field linear combination needs at least one term")
    for _, field in normalized:
        if not isinstance(field, DGField):
            raise TypeError("field linear combination terms must contain DGField objects")
        space.assert_basis_compatible(field.space)
        if field.space.order > space.order:
            raise ValueError(
                "field linear-combination space must have at least the maximum "
                "polynomial order of its terms"
            )

    active = [(weight, field) for weight, field in normalized if weight != 0.0]
    if not active:
        return space.zeros(name=name)

    constants = [field.constant_value for _, field in active]
    if all(value is not None for value in constants):
        value = sum(
            weight * float(constant)
            for (weight, _), constant in zip(active, constants)
        )
        return space.constant(value, name=name)

    nonconstant_device_sets = [
        set(field._device_coeffs or {})
        for _, field in active
        if field.constant_value is None
    ]
    common_devices = (
        set.intersection(*nonconstant_device_sets)
        if nonconstant_device_sets and all(nonconstant_device_sets)
        else set()
    )
    if common_devices:
        from hdgfem.core.device import as_cupy_space, field_from_cupy_coefficients
        from hdgfem.runtime.optional import require_cupy

        cp = require_cupy()
        device_id = min(common_devices)
        with cp.cuda.Device(device_id):
            cspace = as_cupy_space(space, device=device_id)
            coefficients = cp.zeros(space.shape, dtype=REAL_DTYPE)
            for weight, field in active:
                constant = field.constant_value
                if constant is None:
                    field_coefficients = field._device_coefficients_for(device_id)
                    if field.space.order < space.order:
                        field_coefficients = (
                            field_coefficients
                            @ cspace.degree_elevation_matrix_from(field.space)
                        )
                else:
                    reference = cp.asarray(
                        space._constant_reference_coeffs(constant), dtype=REAL_DTYPE,
                    )
                    field_coefficients = cp.broadcast_to(reference[None, :], space.shape)
                coefficients += weight * field_coefficients
            return field_from_cupy_coefficients(
                space,
                cp.ascontiguousarray(coefficients),
                device=device_id,
                name=name,
            )

    coefficients = np.zeros(space.shape, dtype=REAL_DTYPE)
    for weight, field in active:
        constant = field.constant_value
        if constant is None:
            field_coefficients = field.coeffs
            if field.space.order < space.order:
                field_coefficients = (
                    field_coefficients
                    @ space.degree_elevation_matrix_from(field.space)
                )
        else:
            field_coefficients = np.broadcast_to(
                space._constant_reference_coeffs(constant)[None, :], space.shape,
            )
        coefficients += weight * field_coefficients
    return space.field(np.ascontiguousarray(coefficients), name=name)


def vector_field_linear_combination(
        space: DGSpace,
        terms,
        *,
        name: str = "u_h",
) -> VectorDGField:
    """Combine vector DG fields componentwise while preserving residency."""
    normalized = [(float(weight), field) for weight, field in terms]
    if not normalized:
        raise ValueError("vector linear combination needs at least one term")
    dimension = normalized[0][1].dim
    if any(field.dim != dimension for _, field in normalized):
        raise ValueError("all vector fields must have the same dimension")
    components = tuple(
        field_linear_combination(
            space,
            [(weight, field.components[index]) for weight, field in normalized],
            name=f"{name}_{index}",
        )
        for index in range(dimension)
    )
    return VectorDGField(components, name=name)


def perpendicular_vector_field(
        flux: VectorDGField,
        scale: float = 1.0,
        space: DGSpace | None = None,
        *,
        name: str = "beta_h",
) -> VectorDGField:
    """Return ``scale * (-q_y, q_x)`` while preserving device residency.

    ``space`` defaults to the flux components' space.
    """
    if flux.dim != 2:
        raise ValueError(f"perpendicular_vector_field expects two components; got {flux.dim}")
    qx, qy = flux.components
    if space is None:
        space = qx.space
    return VectorDGField(
        (
            field_linear_combination(space, [(-float(scale), qy)], name=f"{name}_x"),
            field_linear_combination(space, [(float(scale), qx)], name=f"{name}_y"),
        ),
        name=name,
    )


def trace_linear_combination(terms):
    """Return a contiguous host/device linear combination of trace arrays."""
    normalized = list(terms)
    present = [(float(weight), trace) for weight, trace in normalized if trace is not None]
    if len(present) != len(normalized) or not present:
        return None
    first = present[0][1]
    if type(first).__module__.split(".", 1)[0] == "cupy" or hasattr(first, "__cuda_array_interface__"):
        from hdgfem.runtime.optional import require_cupy

        cp = require_cupy()
        return cp.ascontiguousarray(sum(weight * cp.asarray(trace) for weight, trace in present))
    return np.ascontiguousarray(sum(weight * np.asarray(trace) for weight, trace in present))


def _edge_points(mesh, points_1d, xp):
    """Map reference edge coordinates to every physical mesh edge."""
    edge_vertices = mesh.node_coords[mesh.edges]
    return 0.5 * (
        (1.0 - points_1d)[None, :, None] * edge_vertices[:, 0:1, :]
        + (1.0 + points_1d)[None, :, None] * edge_vertices[:, 1:2, :]
    )


def project_callable_to_trace(
        space: DGSpace,
        function,
        *,
        trace_basis: str = "legacy-lagrange",
        reduced: bool = True,
        backend: str = "auto",
):
    """L2-project a callable onto the mesh skeleton on host or device."""
    normalized_backend = str(backend).lower()
    if normalized_backend not in {"auto", "host", "device"}:
        raise ValueError("backend must be 'auto', 'host', or 'device'")
    use_device = normalized_backend == "device"
    if use_device:
        from hdgfem.core.device import as_cupy_trace_space
        from hdgfem.core.device import as_cupy_space
        from hdgfem.runtime.optional import require_cupy

        xp = require_cupy()
        cspace = as_cupy_space(space)
        mesh = cspace.mesh
        trace = as_cupy_trace_space(space.trace_space(trace_basis), device=cspace.device_id)
    else:
        xp = np
        mesh = space.mesh
        trace = space.trace_space(trace_basis)
    points = _edge_points(mesh, trace.quads, xp)
    values = xp.asarray(function(points[:, :, 0], points[:, :, 1]), dtype=REAL_DTYPE)
    target = (int(mesh.num_edg), int(trace.quads.size))
    if values.ndim == 0:
        values = xp.full(target, float(values), dtype=REAL_DTYPE)
    elif tuple(values.shape) == (target[1],):
        values = xp.broadcast_to(values[None, :], target)
    if tuple(values.shape) != target:
        raise ValueError(f"trace callable values must broadcast to {target}; got {values.shape}")
    rhs = (values * trace.weights[None, :]) @ trace.bas1d_of_ref_edg_qds.T
    coefficients = xp.linalg.solve(trace.M_rf_fc, rhs.T).T
    if reduced:
        coefficients = coefficients[mesh.int_edges_inds]
    return xp.ascontiguousarray(coefficients.ravel())


def project_field_to_trace(
        field: DGField,
        *,
        trace_basis: str = "legacy-lagrange",
        reduced: bool = True,
        backend: str = "auto",
):
    """Project element sides into the requested trace basis, reusing its tables.

    Cache backend data, face mass inverse and incidence counts on the fixed
    space. Use the package's six oriented coupling tables; no retabulation of
    the volume basis or per-call factorization is necessary.
    """
    if backend not in {"auto", "host", "device"}:
        raise ValueError("backend must be 'auto', 'host', or 'device'")
    use_device = backend == "device" or (backend == "auto" and
        field.device_coefficients_materialized() and not field.coefficients_materialized)
    xp, mesh, trace, inverse, counts = _trace_projection_data(field.space, trace_basis, use_device)
    if use_device:
        from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
        coefficients = as_cupy_coefficients(field, as_cupy_space(field.space))
    else:
        coefficients = field.coeffs
    tables = trace.face_trace_test_element_trial_oriented
    rhs = xp.einsum("ki,fai->kfa", coefficients, tables[:3])
    reverse = xp.einsum("ki,fai->kfa", coefficients, tables[3:])
    xp.copyto(rhs, reverse, where=(~mesh.orientations)[:, :, None])
    full = xp.zeros((mesh.num_edg, trace.edg_dof), dtype=REAL_DTYPE)
    xp.add.at(full, mesh.loc2glob_edge.ravel(), rhs.reshape(-1, trace.edg_dof))
    full = (full / counts[:, None]) @ inverse.T
    # Preserve the established host helper's zero boundary convention.
    full[mesh.bnd_edges_inds] = 0
    if reduced:
        full = full[mesh.int_edges_inds]
    return xp.ascontiguousarray(full.ravel())


def _trace_projection_data(space, trace_basis, use_device):
    """Fixed-space projection data shared by boundary and density traces."""
    trace_host = space.trace_space(trace_basis)
    if use_device:
        from hdgfem.core.device import as_cupy_trace_space
        from hdgfem.core.device import as_cupy_space
        from hdgfem.runtime.optional import require_cupy
        xp = require_cupy()
        cspace = as_cupy_space(space)
        mesh = cspace.mesh
        trace = as_cupy_trace_space(trace_host, device=cspace.device_id)
        device = cspace.device_id
    else:
        xp, mesh, trace, device = np, space.mesh, trace_host, None
    cache = getattr(space, "_trace_projection_cache", None)
    if cache is None:
        cache = {}
        setattr(space, "_trace_projection_cache", cache)
    key = (trace_host.kind, device)
    if key not in cache:
        inverse = trace.mass_inverse
        counts = xp.asarray(np.bincount(space.mesh.loc2glob_edge.ravel(), minlength=mesh.num_edg), dtype=REAL_DTYPE)
        cache[key] = (xp, mesh, trace, inverse, counts)
    return cache[key]


def _host_reference_points(reference_points) -> np.ndarray:
    """Return reference points as a host ``(n, 2)`` table (they are small tables)."""
    if hasattr(reference_points, "__cuda_array_interface__"):
        from hdgfem.runtime.optional import asnumpy
        reference_points = asnumpy(reference_points)
    return np.ascontiguousarray(np.asarray(reference_points, dtype=REAL_DTYPE).reshape(-1, 2))


def field_values_at_ref(field: DGField, reference_points, *, device: bool = False):
    """Evaluate ``field`` at the same reference points on every element, shape ``(K, n)``.

    Host evaluation is :meth:`DGField.values_at_ref`. ``device=True`` contracts
    resident (or uploaded and cached) device coefficients with the reference
    basis table and returns CuPy values; constant fields fill exactly.
    """
    points = _host_reference_points(reference_points)
    if not device:
        return field.values_at_ref(points)
    from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
    from hdgfem.runtime.optional import require_cupy
    cp = require_cupy()
    if field.constant_value is not None:
        return cp.full((field.space.mesh.num_tri, points.shape[0]), float(field.constant_value), dtype=REAL_DTYPE)
    table = cp.asarray(field.space.basis_at(points).T, dtype=REAL_DTYPE)
    return cp.ascontiguousarray(as_cupy_coefficients(field, as_cupy_space(field.space)) @ table)


def field_gradient_at_ref(field: DGField, reference_points, *, device: bool = False):
    """Return the physical elementwise gradient ``(d/dx, d/dy)`` at reference points.

    Each component has shape ``(K, n)``. Host evaluation is
    :meth:`DGField.grad_at_ref`; ``device=True`` applies the same reference
    gradients and inverse-transpose affine maps to device coefficients.
    """
    points = _host_reference_points(reference_points)
    if not device:
        return field.grad_at_ref(points)
    from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
    from hdgfem.runtime.optional import require_cupy
    cp = require_cupy()
    if field.constant_value is not None:
        zeros = cp.zeros((field.space.mesh.num_tri, points.shape[0]), dtype=REAL_DTYPE)
        return zeros, zeros.copy()
    cspace = as_cupy_space(field.space)
    grad_basis = cp.asarray(field.space.gradient_basis_at(points), dtype=REAL_DTYPE)
    reference = cp.einsum("Ki,qid->Kqd", as_cupy_coefficients(field, cspace), grad_basis)
    physical = cp.einsum("Krd,Kqd->Kqr", cspace.mesh.inv_aff_mats_t, reference)
    return cp.ascontiguousarray(physical[..., 0]), cp.ascontiguousarray(physical[..., 1])


__all__ = [
    "coefficient_field",
    "field_gradient_at_ref",
    "field_linear_combination",
    "field_values_at_ref",
    "perpendicular_vector_field",
    "project_callable_to_trace",
    "project_field_to_trace",
    "solution_field",
    "solution_trace",
    "trace_linear_combination",
    "vector_field_linear_combination",
]


