"""Residency-preserving field and trace operations for solver orchestration."""

from __future__ import annotations

import numpy as np

from .space import DGField, DGSpace, VectorDGField


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
    from ..backends.cupy import field_from_cupy_coefficients

    device_id = int(getattr(getattr(coefficients, "device", None), "id", 0))
    return field_from_cupy_coefficients(space, coefficients, device=device_id, name=name)


def solution_trace(result, space: DGSpace, *, reduced: bool = False, prefer_device: bool = True):
    """Return a solver result trace in a reusable host/device representation.

    A device result normally stores only the reduced interior trace. When
    ``prefer_device`` is true that array is returned without materialization,
    irrespective of ``reduced``. Otherwise a host full trace is used and may
    be restricted to interior edges.
    """
    device_trace = getattr(result, "trace_reduced_device", None)
    if prefer_device and device_trace is not None:
        return device_trace
    trace = getattr(result, "trace", None)
    if trace is None:
        return device_trace
    trace_array = np.asarray(trace, dtype=np.float64)
    if reduced:
        trace_array = trace_array.reshape(space.mesh.num_edg, -1)[space.mesh.int_edges_inds]
    return np.ascontiguousarray(trace_array.reshape(-1))


def field_linear_combination(
        space: DGSpace,
        terms,
        *,
        name: str = "u_h",
) -> DGField:
    """Combine scalar fields without staging common device coefficients through host memory."""
    normalized = [(float(weight), field) for weight, field in terms]
    if not normalized:
        raise ValueError("field linear combination needs at least one term")
    device_sets = [set(getattr(field, "_device_coeffs", {}) or {}) for _, field in normalized]
    common_devices = set.intersection(*device_sets) if all(device_sets) else set()
    if common_devices:
        from ..backends.cupy import field_from_cupy_coefficients, require_cupy

        cp = require_cupy()
        device_id = min(common_devices)
        coefficients = sum(weight * field._device_coefficients_for(device_id) for weight, field in normalized)
        return field_from_cupy_coefficients(
            space,
            cp.ascontiguousarray(coefficients),
            device=device_id,
            name=name,
        )
    coefficients = sum(weight * field.coeffs for weight, field in normalized)
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
        scale: float,
        space: DGSpace,
        *,
        name: str = "beta_h",
) -> VectorDGField:
    """Return ``scale * (-q_y, q_x)`` while preserving device residency."""
    if flux.dim != 2:
        raise ValueError(f"perpendicular_vector_field expects two components; got {flux.dim}")
    qx, qy = flux.components
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
        from ..backends.cupy import require_cupy

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
        from ..backends.advection_cuda import as_cupy_trace_space
        from ..backends.cupy import as_cupy_space, require_cupy

        xp = require_cupy()
        cspace = as_cupy_space(space)
        mesh = cspace.mesh
        trace = as_cupy_trace_space(space.trace_space(trace_basis), device=cspace.device_id)
    else:
        xp = np
        mesh = space.mesh
        trace = space.trace_space(trace_basis)
    points = _edge_points(mesh, trace.quads, xp)
    values = xp.asarray(function(points[:, :, 0], points[:, :, 1]), dtype=xp.float64)
    target = (int(mesh.num_edg), int(trace.quads.size))
    if values.ndim == 0:
        values = xp.full(target, float(values), dtype=xp.float64)
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
    """Project element-side values to a shared skeleton trace."""
    normalized_backend = str(backend).lower()
    use_device = normalized_backend == "device" or (
        normalized_backend == "auto"
        and field.device_coefficients_materialized()
        and not field.coefficients_materialized
    )
    if not use_device:
        from ..assembly.hdg import trace_from_field_faces

        full = trace_from_field_faces(field)
        if not reduced:
            return full
        return np.ascontiguousarray(
            full.reshape(field.space.layout.trace_shape)[field.space.mesh.int_edges_inds].ravel()
        )

    from ..backends.advection_cuda import as_cupy_trace_space
    from ..backends.cupy import as_cupy_coefficients, as_cupy_space, require_cupy

    cp = require_cupy()
    cspace = as_cupy_space(field.space)
    trace = as_cupy_trace_space(field.space.trace_space(trace_basis), device=cspace.device_id)
    coefficients = as_cupy_coefficients(field, cspace)
    host_trace = field.space.trace_space(trace_basis)
    t = host_trace.quads
    face_points = np.stack((
        np.stack((t, -np.ones_like(t)), axis=1),
        np.stack((-t, t), axis=1),
        np.stack((-np.ones_like(t), -t), axis=1),
    ), axis=1)
    basis = cp.asarray(
        field.space.basis_at(face_points.reshape(-1, 2)).reshape(
            t.size, 3, field.space.el_dof,
        ).transpose(1, 2, 0),
        dtype=cp.float64,
    )
    values = cp.einsum("ki,fiq->kfq", coefficients, basis)
    rhs = cp.einsum("kfq,aq,q->kfa", values, trace.bas1d_of_ref_edg_qds, trace.weights)
    local = cp.linalg.solve(trace.M_rf_fc, rhs.reshape(-1, trace.edg_dof).T).T
    local = local.reshape(cspace.mesh.num_tri, 3, trace.edg_dof)
    negative = ~cspace.mesh.orientations
    if cspace.mesh.num_negative_orientations:
        if str(trace_basis).replace("_", "-").lower() == "legendre-modal":
            signs = cp.where(cp.arange(trace.edg_dof) % 2 == 0, 1.0, -1.0)
            local[negative] *= signs[None, :]
        else:
            local[negative] = local[negative][:, ::-1]
    full = cp.zeros((cspace.mesh.num_edg, trace.edg_dof), dtype=cp.float64)
    counts = cp.zeros(cspace.mesh.num_edg, dtype=cp.float64)
    edge_ids = cspace.mesh.loc2glob_edge.reshape(-1)
    cp.add.at(full, edge_ids, local.reshape(-1, trace.edg_dof))
    cp.add.at(counts, edge_ids, cp.ones(edge_ids.size, dtype=cp.float64))
    full /= cp.maximum(counts[:, None], 1.0)
    if reduced:
        full = full[cspace.mesh.int_edges_inds]
    return cp.ascontiguousarray(full.ravel())


__all__ = [
    "coefficient_field",
    "field_linear_combination",
    "perpendicular_vector_field",
    "project_callable_to_trace",
    "project_field_to_trace",
    "solution_field",
    "solution_trace",
    "trace_linear_combination",
    "vector_field_linear_combination",
]
