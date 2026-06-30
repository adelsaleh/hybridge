"""DGSpace-aware Numba assembly adapter for advection-reaction HDG.

This module is the boundary between the public :mod:`dgfem` abstractions and
pure ndarray Numba kernels.  It intentionally keeps Python callables and
``DGField`` objects out of the kernels.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..assembly import hdg as hdg_assembly
from ..assembly import matrices_numpy as hdg_mats
from ..kernels import NUMBA_AVAILABLE
from ..kernels.adv_rea import assemble_local_mats_and_boundary_kernel
from ..linalg.system import KnownDofReduction
from ..kernels.adv_rea_fused import (
    assemble_projected_trace_system_eliminated_kernel,
    assemble_projected_trace_system_kernel,
    reconstruct_projected_field_kernel,
)
from ..kernels.diff_rea_fused import (
    assemble_diffusion_trace_rhs_eliminated_kernel,
    assemble_diffusion_trace_system_eliminated_kernel,
    assemble_projected_diffusion_trace_rhs_eliminated_kernel,
    assemble_projected_diffusion_trace_system_eliminated_kernel,
    reconstruct_diffusion_local_unknowns_kernel,
    reconstruct_projected_diffusion_local_unknowns_kernel,
)
from ..core.space import DGField, DGSpace, VectorDGField


@dataclass(frozen=True)
class NumbaAdvectionLocalAssembly:
    """Local data assembled by the Numba advection-reaction backend."""

    local_mats: np.ndarray
    element_boundary_mats: np.ndarray
    beta_dot_normal: np.ndarray
    timings: dict[str, float]


@dataclass(frozen=True)
class NumbaProjectedTraceAssembly:
    """Global trace system assembled by the fused projected-coefficient backend."""

    trace_system: hdg_assembly.TraceSystem
    beta_dot_normal: np.ndarray
    timings: dict[str, float]
    reduction: KnownDofReduction | None = None


@dataclass(frozen=True)
class NumbaDiffusionTraceAssembly:
    """Reduced diffusion-reaction trace system assembled by Numba."""

    trace_system: hdg_assembly.TraceSystem
    timings: dict[str, float]
    reduction: KnownDofReduction


def _normalize_values(values, num_elements: int, num_points: int, label: str) -> np.ndarray:
    """Normalize scalar/quadrature values to ``(num_elements, num_points)``."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (num_elements, num_points):
        return np.ascontiguousarray(values)
    if values.shape == (num_points,):
        return np.ascontiguousarray(np.broadcast_to(values[None, :], (num_elements, num_points)))
    if values.ndim == 0:
        return np.full((num_elements, num_points), float(values), dtype=np.float64)
    raise ValueError(
        f"{label} must be a scalar, have shape ({num_points},), or have shape "
        f"({num_elements}, {num_points}); got {values.shape}"
    )


def beta_values_on_volume(
        beta_field: VectorDGField | None,
        beta_callables: tuple[Callable, Callable] | None,
        space: DGSpace,
) -> np.ndarray:
    """Return advection values on volume quadrature points."""
    num_elements = space.mesh.num_tri
    num_points = space.quad_data.Krf_w.size
    values = np.empty((num_elements, num_points, 2), dtype=np.float64)
    if beta_field is not None:
        if beta_field.dim != 2:
            raise ValueError("beta_field must have two components")
        beta_field.components[0].space.assert_same_mesh(space)
        beta_field.components[1].space.assert_same_mesh(space)
        values[..., 0] = beta_field.components[0].values_at_ref(space.quad_data.Krf_quads)
        values[..., 1] = beta_field.components[1].values_at_ref(space.quad_data.Krf_quads)
        return np.ascontiguousarray(values)

    if beta_callables is None:
        raise ValueError("either beta_field or beta_callables must be provided")
    points = space.mapped_quads()
    values[..., 0] = _normalize_values(
        beta_callables[0](points[:, :, 0], points[:, :, 1]),
        num_elements,
        num_points,
        "beta[0]",
    )
    values[..., 1] = _normalize_values(
        beta_callables[1](points[:, :, 0], points[:, :, 1]),
        num_elements,
        num_points,
        "beta[1]",
    )
    return np.ascontiguousarray(values)


def reaction_values_on_volume(reaction, space: DGSpace) -> np.ndarray:
    """Return reaction values on solution-space volume quadrature points."""
    num_elements = space.mesh.num_tri
    num_points = space.quad_data.Krf_w.size
    if np.isscalar(reaction):
        return np.full((num_elements, num_points), float(reaction), dtype=np.float64)
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(space)
        return np.ascontiguousarray(reaction.values_at_ref(space.quad_data.Krf_quads), dtype=np.float64)
    if callable(reaction):
        points = space.mapped_quads()
        return _normalize_values(
            reaction(points[:, :, 0], points[:, :, 1]),
            num_elements,
            num_points,
            "reaction",
        )

    values = np.asarray(reaction, dtype=np.float64)
    if values.shape == (num_elements, num_points):
        return np.ascontiguousarray(values)
    if values.shape == (num_points,):
        return np.ascontiguousarray(np.broadcast_to(values[None, :], (num_elements, num_points)))
    if values.shape == space.shape:
        return np.ascontiguousarray(space.field(values, name="reaction").values_at_ref(space.quad_data.Krf_quads))
    raise TypeError(
        "reaction must be a scalar, callable, DGField, quadrature values, or DG coefficients"
    )


def assemble_local_advection_reaction_numba(
        space: DGSpace,
        *,
        beta_field: VectorDGField | None,
        beta_callables: tuple[Callable, Callable] | None,
        beta_dot_normal: np.ndarray | None,
        reaction,
) -> NumbaAdvectionLocalAssembly:
    """Assemble local HDG advection-reaction data with Numba.

    This first backend is deliberately conservative: it returns the same dense
    local matrices and element-boundary coupling tensors as the NumPy assembly
    path.  It does not cache local solvers; callers decide whether the computed
    inverse/local solver should be retained after reconstruction.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}

    start = time.perf_counter()
    beta_volume = beta_values_on_volume(beta_field, beta_callables, space)
    if beta_dot_normal is None:
        if beta_field is None:
            raise ValueError("beta_dot_normal is required when beta is provided as callables")
        beta_dot_normal = hdg_mats.advective_boundary_normal(beta_field, space)
    beta_dot_normal = np.ascontiguousarray(beta_dot_normal, dtype=np.float64)
    reaction_values = reaction_values_on_volume(reaction, space)
    timings["coefficient_values"] = time.perf_counter() - start

    local_mats = np.empty((space.mesh.num_tri, space.el_dof, space.el_dof), dtype=np.float64)
    element_boundary_mats = np.empty(
        (space.mesh.num_tri, space.el_dof, 3 * space.quad_data.edg_dof),
        dtype=np.float64,
    )

    start = time.perf_counter()
    assemble_local_mats_and_boundary_kernel(
        local_mats,
        element_boundary_mats,
        np.ascontiguousarray(space.mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(space.mesh.inv_aff_mats_t, dtype=np.float64),
        np.ascontiguousarray(space.mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.phi, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.gphi, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.Krf_w, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.weighted_bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.weighted_bas1d_of_ref_edg_qds, dtype=np.float64),
        beta_volume,
        beta_dot_normal,
        reaction_values,
    )
    timings["kernel"] = time.perf_counter() - start
    timings["total"] = timings["coefficient_values"] + timings["kernel"]
    return NumbaAdvectionLocalAssembly(
        local_mats=local_mats,
        element_boundary_mats=element_boundary_mats,
        beta_dot_normal=beta_dot_normal,
        timings=timings,
    )


def _same_space_field_coefficients(field, space: DGSpace, label: str) -> np.ndarray:
    """Return contiguous same-space DG coefficients for a scalar projected input."""
    if isinstance(field, DGField):
        field.space.assert_same_mesh(space)
        if field.space is not space:
            raise ValueError(f"{label} must live in the same DGSpace object for the fused Numba backend")
        return np.ascontiguousarray(field.coeffs, dtype=np.float64)
    if callable(field):
        raise TypeError(
            f"{label} is callable; assembly_backend='numba' currently requires explicit projection "
            f"with DGField({label}, space) before calling the solver"
        )

    values = np.asarray(field, dtype=np.float64)
    if values.shape == space.shape:
        return np.ascontiguousarray(values, dtype=np.float64)
    raise TypeError(
        f"{label} must be a DGField or coefficient array with shape {space.shape} "
        "for assembly_backend='numba'. Project callables before calling the solver."
    )


def _same_space_vector_coefficients(beta_field, space: DGSpace) -> np.ndarray:
    """Return contiguous ``(2, nK, nel)`` coefficients for a projected beta."""
    if not isinstance(beta_field, VectorDGField):
        raise TypeError(
            "assembly_backend='numba' currently requires projected beta as a "
            "two-component VectorDGField. Use VectorDGField((beta_x, beta_y), space)."
        )
    if beta_field.dim != 2:
        raise ValueError("projected beta must have exactly two components")
    for component in beta_field.components:
        component.space.assert_same_mesh(space)
        if component.space is not space:
            raise ValueError("projected beta components must live in the same DGSpace object")
    return np.ascontiguousarray(beta_field.as_component_first(), dtype=np.float64)


def _reaction_coefficients(reaction, space: DGSpace) -> tuple[np.ndarray, float, bool]:
    """Normalize reaction data for projected Numba kernels."""
    if np.isscalar(reaction):
        return np.zeros((1, 1), dtype=np.float64), float(reaction), True
    return _same_space_field_coefficients(reaction, space, "reaction"), 0.0, False


def _interior_side_index(space: DGSpace) -> np.ndarray:
    """Map ``(element, local_face)`` to the interior-side COO block id."""
    index = np.full((space.mesh.num_tri, 3), -1, dtype=np.int64)
    index[space.mesh.interior_elements, space.mesh.interior_faces] = np.arange(
        space.mesh.interior_elements.size,
        dtype=np.int64,
    )
    return np.ascontiguousarray(index)


def _reference_advection_tensor(space: DGSpace) -> np.ndarray:
    r"""Return :math:`\int_{\hat K}\phi_k\phi_j\nabla_{\hat x_D}\phi_i`."""
    q = space.quad_data
    return np.ascontiguousarray(
        np.einsum(
            "q,qk,qj,qiD->Dkij",
            q.Krf_w,
            q.phi,
            q.phi,
            q.gphi,
            optimize=True,
        ),
        dtype=np.float64,
    )


def _reference_diffusion_derivative_matrices(space: DGSpace) -> tuple[np.ndarray, np.ndarray]:
    """Return reference derivative matrices in diffusion local-solver layout."""
    q = space.quad_data
    d0 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[0], optimize=True)
    d1 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[1], optimize=True)
    return np.ascontiguousarray(d0.T), np.ascontiguousarray(d1.T)


def _edge_order_to_solve_map(
        num_edges: int,
        active_edges: np.ndarray,
        edge_order: np.ndarray | None,
) -> np.ndarray:
    """Map global mesh edges to contiguous solve-edge positions."""
    active_edges = np.asarray(active_edges, dtype=np.int64)
    if edge_order is None:
        ordered_edges = active_edges
    else:
        ordered_edges = np.asarray(edge_order, dtype=np.int64)
        if ordered_edges.ndim != 1:
            raise ValueError("edge_order must be one-dimensional")
        if ordered_edges.shape != active_edges.shape:
            raise ValueError(
                f"edge_order must have shape {active_edges.shape} for this solve system; "
                f"got {ordered_edges.shape}"
            )

    if ordered_edges.size and (ordered_edges.min() < 0 or ordered_edges.max() >= num_edges):
        raise ValueError("edge_order contains edge ids outside the mesh")

    active_mask = np.zeros(num_edges, dtype=bool)
    active_mask[active_edges] = True
    seen = np.zeros(num_edges, dtype=bool)
    seen[ordered_edges] = True
    if not np.array_equal(active_mask, seen):
        raise ValueError("edge_order must contain every active solve edge exactly once")

    edge_to_solve_edge = np.full(num_edges, -1, dtype=np.int64)
    edge_to_solve_edge[ordered_edges] = np.arange(ordered_edges.size, dtype=np.int64)
    return np.ascontiguousarray(edge_to_solve_edge)


def _full_edge_order_map(space: DGSpace, edge_order: np.ndarray | None) -> np.ndarray:
    """Return the full-system global-edge to solve-edge map."""
    return _edge_order_to_solve_map(
        space.mesh.num_edg,
        np.arange(space.mesh.num_edg, dtype=np.int64),
        edge_order,
    )


def _boundary_reduction_maps(
        space: DGSpace,
        boundary_trace: np.ndarray,
        edge_order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, KnownDofReduction]:
    """Return edge and dof maps for direct boundary elimination."""
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof

    edge_is_free = np.ones(mesh.num_edg, dtype=bool)
    edge_is_free[mesh.bnd_edges_inds] = False
    free_edges = np.flatnonzero(edge_is_free).astype(np.int64)

    edge_to_solve_edge = _edge_order_to_solve_map(mesh.num_edg, free_edges, edge_order)

    system_size = mesh.num_edg * edg_dof
    free_mask = np.repeat(edge_is_free, edg_dof)
    known_mask = ~free_mask
    old_to_new = np.full(system_size, -1, dtype=np.int64)
    old_to_new[free_mask] = np.arange(np.count_nonzero(free_mask), dtype=np.int64)

    reduction = KnownDofReduction(
        rows=np.empty(0, dtype=np.int64),
        cols=np.empty(0, dtype=np.int64),
        data=np.empty(0, dtype=np.float64),
        rhs=np.empty(np.count_nonzero(free_mask), dtype=np.float64),
        free_mask=np.ascontiguousarray(free_mask),
        known_mask=np.ascontiguousarray(known_mask),
        known_values=np.ascontiguousarray(boundary_trace.ravel(), dtype=np.float64),
        old_to_new=np.ascontiguousarray(old_to_new),
    )
    return edge_to_solve_edge, np.ascontiguousarray(free_edges), reduction


def _reduction_with_system(
        reduction: KnownDofReduction,
        rows: np.ndarray,
        cols: np.ndarray,
        data: np.ndarray,
        rhs: np.ndarray,
) -> KnownDofReduction:
    """Attach assembled reduced COO arrays to a precomputed reduction map."""
    return KnownDofReduction(
        rows=rows,
        cols=cols,
        data=data,
        rhs=rhs,
        free_mask=reduction.free_mask,
        known_mask=reduction.known_mask,
        known_values=reduction.known_values,
        old_to_new=reduction.old_to_new,
    )


def assemble_projected_trace_system_numba(
        source,
        beta_field: VectorDGField,
        reaction,
        boundary_condition: Callable,
        space: DGSpace,
        *,
        boundary_penalty: float = 1e20,
        edge_order: np.ndarray | None = None,
        beta_dot_normal: np.ndarray | None = None,
) -> NumbaProjectedTraceAssembly:
    """Assemble the full HDG trace system with fused Numba kernels.

    This backend is the fast projected-coefficient path.  The PDE coefficients
    must already be represented in the solution space:

    ``source``
        :class:`DGField` or coefficient array in ``space``.
    ``beta_field``
        two-component :class:`VectorDGField` with both components in ``space``.
    ``reaction``
        scalar constant, :class:`DGField`, or coefficient array in ``space``.

    Python callables are intentionally rejected here.  Project them explicitly
    with :class:`DGField` or :class:`VectorDGField` before calling the solver.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    source_coeffs = _same_space_field_coefficients(source, space, "source")
    beta_coeffs = _same_space_vector_coefficients(beta_field, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    interior_side_index = _interior_side_index(space)
    timings["coefficient_validation"] = time.perf_counter() - start

    start = time.perf_counter()
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space)
    if beta_dot_normal is None:
        beta_dot_normal = hdg_mats.advective_boundary_normal(beta_field, space)
    else:
        beta_dot_normal = np.ascontiguousarray(beta_dot_normal, dtype=np.float64)
    timings["boundary_trace_and_flux"] = time.perf_counter() - start

    mesh = space.mesh
    q = space.quad_data
    edg_dof = q.edg_dof
    edge_to_solve_edge = _full_edge_order_map(space, edge_order)
    n_interior_flux = mesh.int_edges_inds.size * 2 * edg_dof * 3 * edg_dof
    n_interior_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    n_boundary = mesh.bnd_edges_inds.size * edg_dof
    nnz = n_interior_flux + n_interior_mass + n_boundary

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty((mesh.num_tri * 3 + mesh.bnd_edges_inds.size) * edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_projected_trace_system_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        interior_side_index,
        edge_to_solve_edge,
        np.ascontiguousarray(mesh.int_edges_inds, dtype=np.int64),
        np.ascontiguousarray(mesh.bnd_edges_inds, dtype=np.int64),
        np.ascontiguousarray(mesh.edge_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.inv_aff_mats_t, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        _reference_advection_tensor(space),
        np.ascontiguousarray(q.bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(q.weights_JGL, dtype=np.float64),
        np.ascontiguousarray(q.bas1d_of_ref_edg_qds, dtype=np.float64),
        np.ascontiguousarray(q.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(q.face_trace_test_element_trial_oriented, dtype=np.float64),
        source_coeffs,
        beta_coeffs,
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        float(boundary_penalty),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(mesh.num_edg * edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    return NumbaProjectedTraceAssembly(
        trace_system=hdg_assembly.TraceSystem(
            rows=rows,
            cols=cols,
            data=data,
            rhs=rhs,
            boundary_trace=boundary_trace,
        ),
        beta_dot_normal=beta_dot_normal,
        timings=timings,
    )


def assemble_projected_trace_system_eliminated_numba(
        source,
        beta_field: VectorDGField,
        reaction,
        boundary_condition: Callable,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
        beta_dot_normal: np.ndarray | None = None,
) -> NumbaProjectedTraceAssembly:
    """Assemble the reduced trace system with boundary dofs eliminated in Numba."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    source_coeffs = _same_space_field_coefficients(source, space, "source")
    beta_coeffs = _same_space_vector_coefficients(beta_field, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    interior_side_index = _interior_side_index(space)
    timings["coefficient_validation"] = time.perf_counter() - start

    start = time.perf_counter()
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space)
    if beta_dot_normal is None:
        beta_dot_normal = hdg_mats.advective_boundary_normal(beta_field, space)
    else:
        beta_dot_normal = np.ascontiguousarray(beta_dot_normal, dtype=np.float64)
    timings["boundary_trace_and_flux"] = time.perf_counter() - start

    mesh = space.mesh
    q = space.quad_data
    edg_dof = q.edg_dof

    start = time.perf_counter()
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(space, boundary_trace, edge_order)
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[mesh.interior_elements], axis=1).astype(np.int64)
    side_flux_offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    side_flux_offsets[0] = 0
    np.cumsum(side_col_counts * edg_dof * edg_dof, out=side_flux_offsets[1:])
    n_flux = int(side_flux_offsets[-1])
    n_mass = free_edges.size * edg_dof * edg_dof
    nnz = n_flux + n_mass
    timings["reduction_map"] = time.perf_counter() - start

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty(mesh.num_tri * 3 * edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_projected_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        interior_side_index,
        edge_to_solve_edge,
        np.ascontiguousarray(free_edges, dtype=np.int64),
        np.ascontiguousarray(side_flux_offsets, dtype=np.int64),
        np.ascontiguousarray(mesh.edge_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.inv_aff_mats_t, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        _reference_advection_tensor(space),
        np.ascontiguousarray(q.bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(q.weights_JGL, dtype=np.float64),
        np.ascontiguousarray(q.bas1d_of_ref_edg_qds, dtype=np.float64),
        np.ascontiguousarray(q.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(q.face_trace_test_element_trial_oriented, dtype=np.float64),
        source_coeffs,
        beta_coeffs,
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())

    reduction = _reduction_with_system(reduction_template, rows, cols, data, rhs)
    return NumbaProjectedTraceAssembly(
        trace_system=hdg_assembly.TraceSystem(
            rows=rows,
            cols=cols,
            data=data,
            rhs=rhs,
            boundary_trace=boundary_trace,
        ),
        beta_dot_normal=beta_dot_normal,
        timings=timings,
        reduction=reduction,
    )


def _normalize_diffusion_stabilization(stabilization, space: DGSpace) -> np.ndarray:
    """Return diffusion stabilization values with shape ``(num_elements, 3)``."""
    mesh = space.mesh
    if np.isscalar(stabilization):
        return np.full((mesh.num_tri, 3), float(stabilization), dtype=np.float64)
    tau = np.asarray(stabilization, dtype=np.float64)
    if tau.shape == (mesh.num_tri,):
        return np.ascontiguousarray(np.broadcast_to(tau[:, None], (mesh.num_tri, 3)))
    if tau.shape != (mesh.num_tri, 3):
        raise ValueError(f"stabilization must be scalar or have shape ({mesh.num_tri}, 3); got {tau.shape}")
    return np.ascontiguousarray(tau)


def assemble_projected_diffusion_trace_system_eliminated_numba(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
) -> NumbaDiffusionTraceAssembly:
    """Assemble the reduced projected diffusion-reaction trace system in Numba."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    source_coeffs = _same_space_field_coefficients(source, space, "source")
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    interior_side_index = _interior_side_index(space)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    timings["coefficient_validation"] = time.perf_counter() - start

    start = time.perf_counter()
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space)
    mesh = space.mesh
    q = space.quad_data
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(space, boundary_trace, edge_order)
    valid_elements = np.ascontiguousarray(mesh.interior_elements, dtype=np.int64)
    valid_faces = np.ascontiguousarray(mesh.interior_faces, dtype=np.int64)
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[valid_elements], axis=1).astype(np.int64)
    side_flux_offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    side_flux_offsets[0] = 0
    np.cumsum(side_col_counts * q.edg_dof * q.edg_dof, out=side_flux_offsets[1:])
    n_flux = int(side_flux_offsets[-1])
    n_mass = valid_elements.size * q.edg_dof * q.edg_dof
    nnz = n_flux + n_mass
    timings["reduction_map"] = time.perf_counter() - start

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty(mesh.num_tri * 3 * q.edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_projected_diffusion_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        interior_side_index,
        np.ascontiguousarray(edge_to_solve_edge, dtype=np.int64),
        valid_elements,
        valid_faces,
        np.ascontiguousarray(side_flux_offsets, dtype=np.int64),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.MKrf_inv, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_trace_trial, dtype=np.float64),
        np.ascontiguousarray(q.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(q.face_trace_test_element_trial_oriented, dtype=np.float64),
        d0_reference,
        d1_reference,
        source_coeffs,
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * q.edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())

    reduction = _reduction_with_system(reduction_template, rows, cols, data, rhs)
    return NumbaDiffusionTraceAssembly(
        trace_system=hdg_assembly.TraceSystem(
            rows=rows,
            cols=cols,
            data=data,
            rhs=rhs,
            boundary_trace=boundary_trace,
        ),
        timings=timings,
        reduction=reduction,
    )


def assemble_projected_diffusion_trace_rhs_eliminated_numba(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, KnownDofReduction, dict[str, float]]:
    """Assemble only the reduced RHS for a cached projected diffusion matrix."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    source_coeffs = _same_space_field_coefficients(source, space, "source")
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    interior_side_index = _interior_side_index(space)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space)
    mesh = space.mesh
    q = space.quad_data
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(space, boundary_trace, edge_order)
    timings["preparation"] = time.perf_counter() - start

    rhs_indices = np.empty(mesh.num_tri * 3 * q.edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_projected_diffusion_trace_rhs_eliminated_kernel(
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        interior_side_index,
        np.ascontiguousarray(edge_to_solve_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.MKrf_inv, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_trace_trial, dtype=np.float64),
        np.ascontiguousarray(q.face_trace_test_element_trial_oriented, dtype=np.float64),
        d0_reference,
        d1_reference,
        source_coeffs,
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * q.edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    reduction = _reduction_with_system(
        reduction_template,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        rhs,
    )
    return rhs, boundary_trace, reduction, timings


def assemble_diffusion_trace_system_eliminated_numba(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        source_rhs: np.ndarray,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
) -> NumbaDiffusionTraceAssembly:
    """Assemble a boundary-eliminated diffusion-reaction trace system.

    This adapter mirrors the strong-boundary path used by the advection
    backend: only non-boundary trace dofs appear in the emitted COO matrix, and
    contributions from prescribed boundary trace columns are accumulated
    directly into the reduced RHS.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    mesh = space.mesh
    q = space.quad_data
    local_solver = np.ascontiguousarray(local_solver, dtype=np.float64)
    element_boundary_mats = np.ascontiguousarray(element_boundary_mats, dtype=np.float64)
    source_rhs = np.ascontiguousarray(source_rhs, dtype=np.float64)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    expected_solver = (mesh.num_tri, 3 * q.el_dof, 3 * q.el_dof)
    expected_boundary = (mesh.num_tri, 3 * q.el_dof, 3 * q.edg_dof)
    expected_rhs = (mesh.num_tri, 3 * q.el_dof)
    if local_solver.shape != expected_solver:
        raise ValueError(f"local_solver must have shape {expected_solver}; got {local_solver.shape}")
    if element_boundary_mats.shape != expected_boundary:
        raise ValueError(f"element_boundary_mats must have shape {expected_boundary}; got {element_boundary_mats.shape}")
    if source_rhs.shape != expected_rhs:
        raise ValueError(f"source_rhs must have shape {expected_rhs}; got {source_rhs.shape}")
    timings["input_validation"] = time.perf_counter() - start

    start = time.perf_counter()
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space)
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(space, boundary_trace, edge_order)
    valid_elements = np.ascontiguousarray(mesh.interior_elements, dtype=np.int64)
    valid_faces = np.ascontiguousarray(mesh.interior_faces, dtype=np.int64)
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[valid_elements], axis=1).astype(np.int64)
    side_flux_offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    side_flux_offsets[0] = 0
    np.cumsum(side_col_counts * q.edg_dof * q.edg_dof, out=side_flux_offsets[1:])
    n_flux = int(side_flux_offsets[-1])
    n_mass = valid_elements.size * q.edg_dof * q.edg_dof
    nnz = n_flux + n_mass
    timings["reduction_map"] = time.perf_counter() - start

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty(mesh.num_tri * 3 * q.edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_diffusion_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        _interior_side_index(space),
        np.ascontiguousarray(edge_to_solve_edge, dtype=np.int64),
        valid_elements,
        valid_faces,
        side_flux_offsets,
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(q.face_trace_test_element_trial_oriented, dtype=np.float64),
        local_solver,
        element_boundary_mats,
        source_rhs,
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * q.edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())

    reduction = _reduction_with_system(reduction_template, rows, cols, data, rhs)
    return NumbaDiffusionTraceAssembly(
        trace_system=hdg_assembly.TraceSystem(
            rows=rows,
            cols=cols,
            data=data,
            rhs=rhs,
            boundary_trace=boundary_trace,
        ),
        timings=timings,
        reduction=reduction,
    )


def assemble_diffusion_trace_rhs_eliminated_numba(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        source_rhs: np.ndarray,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, KnownDofReduction, dict[str, float]]:
    """Assemble only the reduced RHS for a cached diffusion trace operator."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    mesh = space.mesh
    q = space.quad_data
    local_solver = np.ascontiguousarray(local_solver, dtype=np.float64)
    element_boundary_mats = np.ascontiguousarray(element_boundary_mats, dtype=np.float64)
    source_rhs = np.ascontiguousarray(source_rhs, dtype=np.float64)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space)
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(space, boundary_trace, edge_order)
    timings["preparation"] = time.perf_counter() - start

    rhs_indices = np.empty(mesh.num_tri * 3 * q.edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_diffusion_trace_rhs_eliminated_kernel(
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        _interior_side_index(space),
        np.ascontiguousarray(edge_to_solve_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.face_trace_test_element_trial_oriented, dtype=np.float64),
        local_solver,
        element_boundary_mats,
        source_rhs,
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * q.edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    reduction = _reduction_with_system(
        reduction_template,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        rhs,
    )
    return rhs, boundary_trace, reduction, timings


def reconstruct_diffusion_local_unknowns_numba(
        trace: np.ndarray,
        source_rhs: np.ndarray,
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        space: DGSpace,
) -> np.ndarray:
    """Recover mixed diffusion local unknowns with the Numba backend."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    mesh = space.mesh
    q = space.quad_data
    trace = np.ascontiguousarray(np.asarray(trace, dtype=np.float64))
    expected_trace_shape = (mesh.num_edg * q.edg_dof,)
    if trace.shape != expected_trace_shape:
        raise ValueError(f"trace must have shape {expected_trace_shape}; got {trace.shape}")

    local_solver = np.ascontiguousarray(local_solver, dtype=np.float64)
    element_boundary_mats = np.ascontiguousarray(element_boundary_mats, dtype=np.float64)
    source_rhs = np.ascontiguousarray(source_rhs, dtype=np.float64)
    local_unknowns = np.empty((mesh.num_tri, 3 * q.el_dof), dtype=np.float64)
    reconstruct_diffusion_local_unknowns_kernel(
        local_unknowns,
        trace,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        local_solver,
        element_boundary_mats,
        source_rhs,
    )
    return local_unknowns


def reconstruct_projected_diffusion_local_unknowns_numba(
        trace: np.ndarray,
        source,
        reaction,
        stabilization,
        space: DGSpace,
) -> np.ndarray:
    """Recover mixed diffusion local unknowns with fully fused projected kernels."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    source_coeffs = _same_space_field_coefficients(source, space, "source")
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    mesh = space.mesh
    q = space.quad_data
    trace = np.ascontiguousarray(np.asarray(trace, dtype=np.float64))
    expected_trace_shape = (mesh.num_edg * q.edg_dof,)
    if trace.shape != expected_trace_shape:
        raise ValueError(f"trace must have shape {expected_trace_shape}; got {trace.shape}")

    local_unknowns = np.empty((mesh.num_tri, 3 * q.el_dof), dtype=np.float64)
    reconstruct_projected_diffusion_local_unknowns_kernel(
        local_unknowns,
        trace,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.MKrf_inv, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_trace_trial, dtype=np.float64),
        d0_reference,
        d1_reference,
        source_coeffs,
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
    )
    return local_unknowns


def reconstruct_projected_field_numba(
        trace: np.ndarray,
        source,
        beta_field: VectorDGField,
        reaction,
        space: DGSpace,
        *,
        name: str = "u_h",
) -> DGField:
    """Recover element coefficients with the projected fused Numba backend."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    source_coeffs = _same_space_field_coefficients(source, space, "source")
    beta_coeffs = _same_space_vector_coefficients(beta_field, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    trace = np.ascontiguousarray(np.asarray(trace, dtype=np.float64))
    expected_trace_shape = (space.mesh.num_edg * space.quad_data.edg_dof,)
    if trace.shape != expected_trace_shape:
        raise ValueError(f"trace must have shape {expected_trace_shape}; got {trace.shape}")

    coeffs = np.empty(space.shape, dtype=np.float64)
    mesh = space.mesh
    q = space.quad_data
    reconstruct_projected_field_kernel(
        coeffs,
        trace,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.inv_aff_mats_t, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        _reference_advection_tensor(space),
        np.ascontiguousarray(q.bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(q.weights_JGL, dtype=np.float64),
        np.ascontiguousarray(q.bas1d_of_ref_edg_qds, dtype=np.float64),
        source_coeffs,
        beta_coeffs,
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
    )
    return space.field(coeffs, name=name)


__all__ = [
    "NumbaAdvectionLocalAssembly",
    "NumbaDiffusionTraceAssembly",
    "NumbaProjectedTraceAssembly",
    "assemble_diffusion_trace_rhs_eliminated_numba",
    "assemble_diffusion_trace_system_eliminated_numba",
    "assemble_projected_diffusion_trace_rhs_eliminated_numba",
    "assemble_projected_diffusion_trace_system_eliminated_numba",
    "assemble_local_advection_reaction_numba",
    "assemble_projected_trace_system_eliminated_numba",
    "assemble_projected_trace_system_numba",
    "beta_values_on_volume",
    "reconstruct_diffusion_local_unknowns_numba",
    "reconstruct_projected_diffusion_local_unknowns_numba",
    "reconstruct_projected_field_numba",
    "reaction_values_on_volume",
]
