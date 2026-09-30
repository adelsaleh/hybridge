"""Reusable HDG static-condensation and trace-assembly helpers.

This module contains the mesh/trace glue that is independent of the concrete
PDE local operator.  Local element matrices still come from problem-specific
code, usually via :mod:`hdgfem.assembly.matrices_numpy`; once those matrices and element RHS
moments are available, the functions here assemble the global trace system and
recover element coefficients.
"""

from __future__ import annotations

from hdgfem.precision import REAL_DTYPE

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from hdgfem.assembly import matrices_numpy as hdg_mats
from hdgfem.core.space import DGField, DGSpace, DGTraceSpace, VectorDGField


@dataclass(frozen=True)
class TraceSystem:
    """COO representation of an assembled HDG trace system."""

    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    rhs: np.ndarray
    boundary_trace: np.ndarray | None = None


def as_vector_field(beta, space: DGSpace) -> VectorDGField:
    """Normalize two-component advection data to a :class:`VectorDGField`.

    Accepted inputs are already-built vector fields, coefficient arrays accepted
    by ``space * space``, or a tuple of two callables that are projected into
    ``space``.
    """
    if isinstance(beta, VectorDGField):
        if beta.dim != 2:
            raise ValueError("advection field must have two components")
        beta.components[0].space.assert_same_mesh(space)
        beta.components[1].space.assert_same_mesh(space)
        return beta

    vector_space = space * space
    if isinstance(beta, np.ndarray):
        return vector_space.field(beta, name="beta_h")
    if isinstance(beta, (tuple, list)) and len(beta) == 2 and all(isinstance(component, DGField) for component in beta):
        return vector_space.field(beta, name="beta_h")
    if (
        isinstance(beta, tuple)
        and len(beta) == 2
        and not any(isinstance(component, DGField) for component in beta)
        and all(callable(component) for component in beta)
    ):
        return vector_space.field(
            (
                space.project_callable(beta[0], name="beta_x").coeffs,
                space.project_callable(beta[1], name="beta_y").coeffs,
            ),
            name="beta_h",
        )
    raise TypeError("beta must be a VectorDGField, coefficient array, or tuple of two callables")


def reaction_mass(reaction, space: DGSpace) -> np.ndarray:
    """Assemble reaction mass matrices, with an exact constant fast path."""
    if np.isscalar(reaction):
        return float(reaction) * space.mesh.aff_jacs[:, None, None] * space.quad_data.MKrf[None, :, :]
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(space)
        constant_value = reaction.constant_value
        if constant_value is not None:
            if constant_value == 0.0:
                return np.zeros((space.mesh.num_tri, space.el_dof, space.el_dof), dtype=REAL_DTYPE)
            return constant_value * space.mesh.aff_jacs[:, None, None] * space.quad_data.MKrf[None, :, :]
        return hdg_mats.mass_from_field(space, reaction)
    if callable(reaction):
        return space.weighted_mass(reaction)

    values = np.asarray(reaction, dtype=REAL_DTYPE)
    if values.shape == (space.mesh.num_tri, space.quad_data.Krf_w.shape[0]):
        out = np.empty((space.mesh.num_tri, space.el_dof, space.el_dof), dtype=REAL_DTYPE)
        return hdg_mats.set_weighted_mass_from_values(out, values, space)
    if values.shape == (space.mesh.num_tri, space.el_dof):
        return hdg_mats.mass_from_field(space, space.field(values, name="reaction"))
    raise TypeError("reaction must be a scalar, callable, quadrature values, or DG coefficient array")


def source_moments(source, space: DGSpace) -> np.ndarray:
    r"""Compute element moments :math:`\int_K f\phi_i\,dx`."""
    if isinstance(source, DGField):
        source.space.assert_same_mesh(space)
        constant_value = source.constant_value
        if constant_value is not None:
            rhs = space._constant_reference_moments(constant_value)
            return np.ascontiguousarray(space.mesh.aff_jacs[:, None] * rhs[None, :], dtype=REAL_DTYPE)
        if source.space is space:
            rhs = source.coeffs @ space.quad_data.MKrf
            rhs *= space.mesh.aff_jacs[:, None]
            return np.ascontiguousarray(rhs, dtype=REAL_DTYPE)
        values = source.values_at_ref(space.quad_data.Krf_quads)
    elif callable(source):
        points = space.mapped_quads()
        values = source(points[:, :, 0], points[:, :, 1])
    else:
        array = np.asarray(source, dtype=REAL_DTYPE)
        if array.shape == space.shape:
            return np.ascontiguousarray(array)
        if array.shape == (space.mesh.num_tri, space.quad_data.Krf_w.shape[0]):
            values = array
        else:
            raise TypeError("source must be a DGField, callable, source moments, or quadrature values")

    return source_moments_from_values(values, space)


def source_moments_from_values(values, space: DGSpace) -> np.ndarray:
    r"""Compute :math:`\int_K f\phi_i\,dx` from ``f`` on volume quadrature, shape ``(K, nq)``.

    Unlike :func:`source_moments`, an array whose shape also matches the
    coefficient layout is never mistaken for precomputed moments.
    """
    values = np.asarray(values, dtype=REAL_DTYPE)
    if values.ndim == 0:
        values = np.full((space.mesh.num_tri, space.quad_data.Krf_w.shape[0]), float(values))
    elif values.shape == (space.quad_data.Krf_w.shape[0],):
        values = np.broadcast_to(values[None, :], (space.mesh.num_tri, values.size))
    if values.shape != (space.mesh.num_tri, space.quad_data.Krf_w.shape[0]):
        raise ValueError(
            "source values must have shape "
            f"({space.mesh.num_tri}, {space.quad_data.Krf_w.shape[0]}); got {values.shape}"
        )

    rhs = values @ space.quad_data.weighted_phi
    rhs *= space.mesh.aff_jacs[:, None]
    return np.ascontiguousarray(rhs, dtype=REAL_DTYPE)


def block_source_moments(
        source,
        space: DGSpace,
        *,
        num_blocks: int,
        source_block: int = 0,
) -> np.ndarray:
    """Embed scalar source moments into a block local-system RHS.

    For example, mixed diffusion-reaction uses local unknowns
    ``[u_h, q_{x,h}, q_{y,h}]`` and therefore needs source moments in the
    first block and zeros in the two flux blocks.
    """
    num_blocks = int(num_blocks)
    source_block = int(source_block)
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if source_block < 0 or source_block >= num_blocks:
        raise ValueError("source_block must satisfy 0 <= source_block < num_blocks")

    array = np.asarray(source, dtype=REAL_DTYPE) if not callable(source) and not isinstance(source, DGField) else None
    if array is not None and array.shape == (space.mesh.num_tri, num_blocks * space.el_dof):
        return np.ascontiguousarray(array)

    scalar_moments = source_moments(source, space)
    result = np.zeros((space.mesh.num_tri, num_blocks * space.el_dof), dtype=REAL_DTYPE)
    start = source_block * space.el_dof
    result[:, start:start + space.el_dof] = scalar_moments
    return result


def normalize_boundary_condition(boundary_condition, *, require_none: bool = False):
    """Return a callable boundary condition from a callable or real constant.

    When ``require_none`` is true, ``None`` is the only accepted value.
    Discrete volume and trace fields are deliberately not boundary-condition
    inputs. Their projection/interpolation semantics will be designed as a
    separate API rather than inferred here.
    """
    if require_none and boundary_condition is not None:
        raise ValueError("boundary_condition must be None when boundary_mode='zero-flux'")
    if boundary_condition is None:
        if require_none:
            return None
        raise ValueError("boundary_condition may not be None")
    if isinstance(boundary_condition, DGField):
        raise TypeError(
            "boundary_condition must be a callable or real scalar constant; "
            "DGField and HDGTraceField boundary data are not supported"
        )
    if callable(boundary_condition):
        return boundary_condition
    if isinstance(boundary_condition, Real):
        value = float(boundary_condition)

        def constant_boundary_condition(_x, _y):
            """Return the normalized constant boundary value."""
            return value

        constant_boundary_condition._hdgfem_constant_value = value
        return constant_boundary_condition
    raise TypeError(
        "boundary_condition must be a callable or real scalar constant; "
        "DGField and HDGTraceField boundary data are not supported"
    )


def boundary_trace_coefficients(
        boundary_condition,
        space: DGSpace,
        *,
        trace_basis: str = "legacy-lagrange",
        trace_space: DGTraceSpace | None = None,
        backend: str = "host",
        boundary_only: bool = False,
):
    """Return prescribed coefficients using the existing nodal/modal convention.

    Host behavior is unchanged by default. The device path caches sample
    points and mass inverses and evaluates compatible callables on device.
    boundary_only returns just prescribed edge rows, avoiding a full trace.
    """
    if backend not in {"host", "device"}:
        raise ValueError("boundary trace backend must be 'host' or 'device'")
    trace_ref = space.trace_space(trace_basis) if trace_space is None else trace_space
    boundary_condition = normalize_boundary_condition(boundary_condition)
    if backend == "host":
        values = trace_ref.boundary_coefficients(boundary_condition)
        return values[space.mesh.bnd_edges_inds] if boundary_only else values
    from hdgfem.backends.cupy import as_cupy_space, require_cupy
    from hdgfem.backends.advection_cuda import as_cupy_trace_space
    xp = require_cupy()
    cspace = as_cupy_space(space)
    mesh = cspace.mesh
    trace = as_cupy_trace_space(trace_ref, device=cspace.device_id)
    points = getattr(trace, "_boundary_sample_points", None)
    if points is None:
        vertices = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]]
        t = trace.interpolation_nodes if trace.nodal else trace.quads
        points = .5*((1-t)[None, :, None]*vertices[:, :1] + (1+t)[None, :, None]*vertices[:, 1:])
        object.__setattr__(trace, "_boundary_sample_points", points)
    try:
        sampled = boundary_condition(points[:, :, 0], points[:, :, 1])
    except TypeError:
        # Existing NumPy-only callables remain supported; guiding-center
        # callables handle device arrays and avoid this transfer fallback.
        values = xp.asarray(trace_ref.boundary_coefficients(boundary_condition)[space.mesh.bnd_edges_inds])
    else:
        values = xp.broadcast_to(xp.asarray(sampled, dtype=REAL_DTYPE), points.shape[:2])
        if not trace.nodal:
            values = ((values*trace.weights) @ trace.bas1d_of_ref_edg_qds.T) @ trace.mass_inverse
    if boundary_only:
        return xp.ascontiguousarray(values)
    full = xp.zeros((mesh.num_edg, trace.edg_dof), dtype=REAL_DTYPE)
    full[mesh.bnd_edges_inds] = values
    return full


def free_trace_dofs(
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Return a boolean mask selecting non-boundary global trace dofs."""
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof if trace_space is None else trace_space.edg_dof
    mask = np.ones(mesh.num_edg * edg_dof, dtype=bool)
    boundary_dofs = mesh.bnd_edges_inds[:, None] * edg_dof + np.arange(edg_dof)[None, :]
    mask[boundary_dofs.ravel()] = False
    return mask


InteriorMassMode = Literal["edge", "face"]


def _validate_interior_mass_mode(interior_mass_mode: str) -> InteriorMassMode:
    """Validate the trace-mass contribution convention."""
    if interior_mass_mode not in {"edge", "face"}:
        raise ValueError("interior_mass_mode must be 'edge' or 'face'")
    return interior_mass_mode


def trace_matrix_indices(
        space: DGSpace,
        *,
        interior_mass_mode: InteriorMassMode = "edge",
        trace_space: DGTraceSpace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return COO row/column indices for the full trace system.

    ``interior_mass_mode="edge"`` contributes one trace mass block per
    interior edge.  ``"face"`` contributes one block per element-side incidence
    on an interior edge, which is needed for HDG operators with element-local
    stabilization such as diffusion.
    """
    interior_mass_mode = _validate_interior_mass_mode(interior_mass_mode)
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof if trace_space is None else trace_space.edg_dof
    n_interior_flux = mesh.int_edges_inds.size * 2 * edg_dof * 3 * edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    if interior_mass_mode == "edge":
        n_interior_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    else:
        n_interior_mass = valid_elements.size * edg_dof * edg_dof
    n_boundary = mesh.bnd_edges_inds.size * edg_dof
    rows = np.empty(n_interior_flux + n_interior_mass + n_boundary, dtype=np.int64)
    cols = np.empty_like(rows)

    i_grid, j_grid = np.meshgrid(np.arange(edg_dof), np.arange(edg_dof), indexing="ij")
    row_edges = mesh.loc2glob_edge[valid_elements, valid_faces]
    col_edges = mesh.loc2glob_edge[valid_elements]
    rows[:n_interior_flux] = np.broadcast_to(
        row_edges[:, None, None, None] * edg_dof + i_grid[None, None, :, :],
        (valid_elements.size, 3, edg_dof, edg_dof),
    ).ravel()
    cols[:n_interior_flux] = (col_edges[:, :, None, None] * edg_dof + j_grid[None, None, :, :]).ravel()

    l0 = np.broadcast_to(np.arange(edg_dof)[:, None], (edg_dof, edg_dof)).ravel()
    l1 = np.broadcast_to(np.arange(edg_dof)[None, :], (edg_dof, edg_dof)).ravel()
    offset = n_interior_flux
    if interior_mass_mode == "edge":
        rows[offset:offset + n_interior_mass] = (mesh.int_edges_inds[:, None] * edg_dof + l0).ravel()
        cols[offset:offset + n_interior_mass] = (mesh.int_edges_inds[:, None] * edg_dof + l1).ravel()
    else:
        mass_edges = mesh.loc2glob_edge[valid_elements, valid_faces]
        rows[offset:offset + n_interior_mass] = (
            mass_edges[:, None, None] * edg_dof + i_grid[None, :, :]
        ).ravel()
        cols[offset:offset + n_interior_mass] = (
            mass_edges[:, None, None] * edg_dof + j_grid[None, :, :]
        ).ravel()

    offset += n_interior_mass
    boundary_dofs = mesh.bnd_edges_inds[:, None] * edg_dof + np.arange(edg_dof)[None, :]
    rows[offset:] = boundary_dofs.ravel()
    cols[offset:] = boundary_dofs.ravel()
    return rows, cols


def element_to_trace_matrix_from_lift(
        trace_lift: np.ndarray,
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Assemble oriented element-to-trace Schur complement blocks from lifts."""
    mesh = space.mesh
    q = space.quad_data
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    edg_dof = trace_ref.edg_dof
    trace_lift = np.asarray(trace_lift, dtype=REAL_DTYPE)
    expected_shape = (mesh.num_tri, 3, edg_dof, local_solver.shape[-1])
    if trace_lift.shape != expected_shape:
        raise ValueError(f"trace_lift must have shape {expected_shape}; got {trace_lift.shape}")
    schur = trace_lift @ (local_solver @ element_boundary_mats)[:, None, :, :]
    schur = schur.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    negative_elements, negative_faces = np.nonzero(~mesh.orientations)
    if negative_elements.size:
        columns = schur[negative_elements, :, :, negative_faces, :]
        if trace_ref.kind == "legendre-modal":
            signs = np.where(np.arange(edg_dof) % 2 == 0, 1.0, -1.0)
            columns = columns * signs[None, None, None, :]
        else:
            columns = columns[..., ::-1]
        schur[negative_elements, :, :, negative_faces, :] = columns
    return np.ascontiguousarray(schur.swapaxes(2, 3))


def element_to_trace_matrix(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Assemble oriented element-to-trace Schur complement blocks."""
    mesh = space.mesh
    q = space.quad_data
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    trace_lift = (
        mesh.jacs_el_fc[..., None, None]
        / 2.0
        * trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    )
    return element_to_trace_matrix_from_lift(
        trace_lift,
        local_solver,
        element_boundary_mats,
        space,
        trace_space=trace_ref,
    )


def trace_matrix_data(
        trace_blocks: np.ndarray,
        space: DGSpace,
        boundary_penalty: float,
        *,
        interior_mass_mode: InteriorMassMode = "edge",
        interior_mass_blocks: np.ndarray | None = None,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Return COO data values matching :func:`trace_matrix_indices`."""
    interior_mass_mode = _validate_interior_mass_mode(interior_mass_mode)
    mesh = space.mesh
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    edg_dof = trace_ref.edg_dof
    n_interior_flux = mesh.int_edges_inds.size * 2 * edg_dof * 3 * edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    if interior_mass_mode == "edge":
        n_interior_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    else:
        n_interior_mass = valid_elements.size * edg_dof * edg_dof
    n_boundary = mesh.bnd_edges_inds.size * edg_dof
    data = np.empty(n_interior_flux + n_interior_mass + n_boundary, dtype=REAL_DTYPE)

    data[:n_interior_flux] = -trace_blocks[valid_elements, valid_faces].ravel()

    offset = n_interior_flux
    if interior_mass_blocks is None:
        if interior_mass_mode == "face":
            raise ValueError("interior_mass_blocks is required for interior_mass_mode='face'")
        edge_jacs = mesh.edge_jacs[mesh.int_edges_inds]
        data[offset:offset + n_interior_mass] = (edge_jacs[:, None, None] * trace_ref.M_rf_fc[None, :, :]).ravel()
    else:
        blocks = np.asarray(interior_mass_blocks, dtype=REAL_DTYPE)
        if interior_mass_mode == "edge":
            expected_shape = (mesh.int_edges_inds.size, edg_dof, edg_dof)
        else:
            expected_shape = (valid_elements.size, edg_dof, edg_dof)
        if blocks.shape != expected_shape:
            raise ValueError(f"interior_mass_blocks must have shape {expected_shape}; got {blocks.shape}")
        data[offset:offset + n_interior_mass] = blocks.ravel()

    offset += n_interior_mass
    data[offset:] = boundary_penalty
    return data


def trace_rhs_from_lift(
        trace_lift: np.ndarray,
        source_rhs: np.ndarray,
        local_solver: np.ndarray,
        boundary_condition: Callable,
        space: DGSpace,
        boundary_penalty: float,
        *,
        trace_space: DGTraceSpace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble trace RHS from an element-to-trace lift tensor."""
    mesh = space.mesh
    q = space.quad_data
    edg_dof = q.edg_dof if trace_space is None else trace_space.edg_dof
    trace_lift = np.asarray(trace_lift, dtype=REAL_DTYPE)
    expected_shape = (mesh.num_tri, 3, edg_dof, local_solver.shape[-1])
    if trace_lift.shape != expected_shape:
        raise ValueError(f"trace_lift must have shape {expected_shape}; got {trace_lift.shape}")
    face_rhs = (trace_lift @ (local_solver @ source_rhs[..., None])[:, None, :, :]).squeeze(-1)

    rhs = np.zeros((mesh.num_edg, edg_dof), dtype=REAL_DTYPE)
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    if valid_elements.size:
        np.add.at(rhs, mesh.loc2glob_edge[valid_elements, valid_faces], face_rhs[valid_elements, valid_faces])

    boundary_trace = boundary_trace_coefficients(boundary_condition, space, trace_space=trace_space)
    rhs[mesh.bnd_edges_inds] = boundary_penalty * boundary_trace[mesh.bnd_edges_inds]
    return rhs.ravel(), boundary_trace


def global_rhs(
        source_rhs: np.ndarray,
        local_solver: np.ndarray,
        boundary_condition: Callable,
        space: DGSpace,
        boundary_penalty: float,
        *,
        trace_space: DGTraceSpace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the full trace RHS and known boundary trace coefficients."""
    mesh = space.mesh
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    trace_lift = (
        mesh.jacs_el_fc[..., None, None]
        / 2.0
        * trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    )
    return trace_rhs_from_lift(
        trace_lift,
        source_rhs,
        local_solver,
        boundary_condition,
        space,
        boundary_penalty,
        trace_space=trace_ref,
    )


def assemble_trace_system(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        source_rhs: np.ndarray,
        boundary_condition: Callable,
        space: DGSpace,
        *,
        boundary_penalty: float = 1e20,
        trace_space: DGTraceSpace | None = None,
) -> TraceSystem:
    """Assemble the global HDG trace system from condensed element data."""
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    trace_blocks = element_to_trace_matrix(local_solver, element_boundary_mats, space, trace_space=trace_ref)
    rows, cols = trace_matrix_indices(space, trace_space=trace_ref)
    data = trace_matrix_data(trace_blocks, space, boundary_penalty, trace_space=trace_ref)
    rhs, boundary_trace = global_rhs(
        source_rhs,
        local_solver,
        boundary_condition,
        space,
        boundary_penalty,
        trace_space=trace_ref,
    )
    return TraceSystem(rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace)


def element_traces(
        trace: np.ndarray,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Return oriented element-local trace coefficients."""
    if trace_space is not None:
        return trace_space.element_coefficients(trace)
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof
    trace = np.asarray(trace, dtype=REAL_DTYPE)
    if trace.shape != (mesh.num_edg * edg_dof,):
        raise ValueError(f"trace must have shape ({mesh.num_edg * edg_dof},); got {trace.shape}")
    traces = trace.reshape(mesh.num_edg, edg_dof)[mesh.loc2glob_edge].copy()
    traces[~mesh.orientations] = traces[~mesh.orientations][:, ::-1]
    return np.ascontiguousarray(traces.reshape(mesh.num_tri, 3 * edg_dof))


def trace_from_field_faces(field: DGField) -> np.ndarray:
    """Project interior element face values to global trace coefficients.

    Boundary trace coefficients are left at zero.  On each interior edge the
    returned trace polynomial is the mass projection of the two neighboring
    element-side traces into the shared edge basis.
    """
    space = field.space
    mesh = space.mesh
    q = space.quad_data
    rhs = np.zeros((mesh.num_edg, q.edg_dof), dtype=REAL_DTYPE)
    mass = np.zeros((mesh.num_edg, q.edg_dof, q.edg_dof), dtype=REAL_DTYPE)
    oriented = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    face_rhs = mesh.jacs_el_fc[:, :, None] * np.einsum("Kfai,Ki->Kfa", oriented, field.coeffs, optimize=True)
    face_mass = mesh.jacs_el_fc[:, :, None, None] * q.M_rf_fc[None, None, :, :]
    for face in range(3):
        edges = mesh.loc2glob_edge[:, face]
        interior = np.isin(edges, mesh.int_edges_inds)
        np.add.at(rhs, edges[interior], face_rhs[interior, face])
        np.add.at(mass, edges[interior], face_mass[interior, face])

    trace = np.zeros((mesh.num_edg, q.edg_dof), dtype=REAL_DTYPE)
    for edge in mesh.int_edges_inds:
        trace[edge] = np.linalg.solve(mass[edge], rhs[edge])
    return trace.reshape(-1)


def h1_flux_jump_norm(field: DGField, flux_coeffs: np.ndarray, trace: np.ndarray) -> tuple[float, float, float]:
    r"""Return ``(||q|| + ||u-\widehat u||, ||q||, ||u-\widehat u||)``.

    The first component matches the HDG-style primal diagnostic used by the
    strategy scripts: an elementwise flux :math:`L^2` norm plus the trace jump
    :math:`L^2(\partial K)` norm.
    """
    space = field.space
    mesh = space.mesh
    q = space.quad_data
    flux_coeffs = np.asarray(flux_coeffs, dtype=REAL_DTYPE)
    expected = (2, mesh.num_tri, q.el_dof)
    if flux_coeffs.shape != expected:
        raise ValueError(f"flux_coeffs must have shape {expected}; got {flux_coeffs.shape}")
    qx_values = flux_coeffs[0] @ q.bas_of_quads
    qy_values = flux_coeffs[1] @ q.bas_of_quads
    flux_l2 = float(np.sqrt(np.einsum(
        "K,Kq,q->",
        mesh.aff_jacs,
        qx_values * qx_values + qy_values * qy_values,
        q.Krf_w,
        optimize=True,
    )))

    local_trace = element_traces(trace, space).reshape(mesh.num_tri, 3, q.edg_dof)
    u_face = np.einsum("Ki,fiq->Kfq", field.coeffs, q.bas_of_bd_quads, optimize=True)
    trace_face = np.einsum("Kfa,aq->Kfq", local_trace, q.bas1d_of_ref_edg_qds, optimize=True)
    jump_l2 = float(np.sqrt(np.einsum(
        "Kf,Kfq,q->",
        mesh.jacs_el_fc,
        (u_face - trace_face) * (u_face - trace_face),
        q.weights_JGL,
        optimize=True,
    )))
    return flux_l2 + jump_l2, flux_l2, jump_l2


def mixed_u_block_rhs_from_residual(residual: np.ndarray, space: DGSpace, *, num_blocks: int = 3) -> np.ndarray:
    """Extract ``-R_u`` from an element-block residual.

    Mixed HDG diffusion operators usually store local unknowns as blocks
    ``[u_h, q_{x,h}, q_{y,h}]``.  This helper turns the first block of a flat
    residual vector into the scalar source moments used by the next linearized
    solve.
    """
    num_blocks = int(num_blocks)
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    local_size = space.mesh.num_tri * num_blocks * space.el_dof
    local = np.asarray(residual[:local_size], dtype=REAL_DTYPE).reshape(space.mesh.num_tri, num_blocks * space.el_dof)
    return np.ascontiguousarray(-local[:, :space.el_dof])


def reconstruct_local_unknowns(
        trace: np.ndarray,
        source_rhs: np.ndarray,
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
):
    """Recover raw element-local unknown coefficients from a solved trace."""
    traces = element_traces(trace, space, trace_space=trace_space)
    unknowns = local_solver @ (source_rhs[..., None] + element_boundary_mats @ traces[..., None])
    return np.ascontiguousarray(unknowns.squeeze(-1))


def reconstruct_field(
        trace: np.ndarray,
        source_rhs: np.ndarray,
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        space: DGSpace,
        *,
        name: str = "u_h",
        trace_space: DGTraceSpace | None = None,
) -> DGField:
    """Recover element coefficients from the solved trace vector."""
    coeffs = reconstruct_local_unknowns(
        trace,
        source_rhs,
        local_solver,
        element_boundary_mats,
        space,
        trace_space=trace_space,
    )
    return space.field(coeffs, name=name)


__all__ = [
    "TraceSystem",
    "as_vector_field",
    "assemble_trace_system",
    "boundary_trace_coefficients",
    "block_source_moments",
    "element_to_trace_matrix_from_lift",
    "element_traces",
    "element_to_trace_matrix",
    "normalize_boundary_condition",
    "free_trace_dofs",
    "global_rhs",
    "h1_flux_jump_norm",
    "mixed_u_block_rhs_from_residual",
    "trace_rhs_from_lift",
    "trace_from_field_faces",
    "reaction_mass",
    "reconstruct_local_unknowns",
    "reconstruct_field",
    "source_moments",
    "trace_matrix_data",
    "trace_matrix_indices",
]
