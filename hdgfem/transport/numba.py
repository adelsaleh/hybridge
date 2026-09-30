"""hdgfem.transport.numba."""

from __future__ import annotations

import hdgfem.hdg.coefficients as hdg_coefficients
import hdgfem.hdg.stabilization as hdg_stabilization
import numpy as np
import time
from collections.abc import Callable
from hdgfem.core.space import DGField, DGSpace, DGTraceSpace, VectorDGField
from hdgfem.linalg.reduction import KnownDofReduction
from hdgfem.runtime.optional import NUMBA_AVAILABLE
from hdgfem.hdg.trace_maps import (
    _boundary_reduction_maps,
    _edge_order_to_solve_map,
    _interior_side_index,
    _reduction_with_system,
    _trace_orientation_mode,
    _trace_ref,
)
from hdgfem.hdg.coefficients import (
    _reaction_coefficients,
    _same_space_field_coefficients,
    _same_space_vector_coefficients,
    _source_coefficients,
    beta_values_on_volume,
    reaction_values_on_volume,
)
from hdgfem.transport.numba_kernels import (
    assemble_face_trace_weights_kernel,
    assemble_projected_trace_system_eliminated_kernel,
    assemble_projected_trace_system_kernel,
    reconstruct_projected_field_kernel,
)
from hdgfem.transport.numba_local_kernels import assemble_local_mats_and_boundary_kernel
from dataclasses import dataclass
from hdgfem.hdg import condensation as hdg_assembly


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
    block_rows: np.ndarray | None = None
    block_cols: np.ndarray | None = None
    block_data: np.ndarray | None = None


def assemble_local_advection_reaction_numba(
        space: DGSpace,
        *,
        beta_field: VectorDGField | None,
        beta_callables: tuple[Callable, Callable] | None,
        beta_dot_normal: np.ndarray | None,
        reaction,
        advection_stabilization=None,
        zero_boundary_flux: bool = False,
        trace_space: DGTraceSpace | None = None,
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
    trace_ref = _trace_ref(space, trace_space)
    _trace_orientation_mode(trace_ref)
    beta_volume = beta_values_on_volume(beta_field, beta_callables, space)
    if beta_dot_normal is None:
        if beta_field is None:
            raise ValueError("beta_dot_normal is required when beta is provided as callables")
        beta_dot_normal = hdg_coefficients.advective_boundary_normal(beta_field, space, trace_space=trace_ref)
    beta_dot_normal = np.ascontiguousarray(beta_dot_normal, dtype=np.float64)
    if zero_boundary_flux:
        beta_dot_normal = beta_dot_normal.copy()
    tau_face = hdg_stabilization.advection_trace_stabilization_values(
        space,
        beta_dot_normal,
        advection_stabilization,
        trace_space=trace_ref,
    )
    tau_face = np.ascontiguousarray(tau_face, dtype=np.float64)
    if zero_boundary_flux:
        tau_face = tau_face.copy()
        _zero_boundary_face_values(space, beta_dot_normal, tau_face)
    reaction_values = reaction_values_on_volume(reaction, space)
    timings["coefficient_values"] = time.perf_counter() - start

    local_mats = np.empty((space.mesh.num_tri, space.el_dof, space.el_dof), dtype=np.float64)
    element_boundary_mats = np.empty(
        (space.mesh.num_tri, space.el_dof, 3 * trace_ref.edg_dof),
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
        np.ascontiguousarray(trace_ref.bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(trace_ref.weighted_bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(trace_ref.weighted_bas1d_of_ref_edg_qds, dtype=np.float64),
        beta_volume,
        beta_dot_normal,
        tau_face,
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


def _advection_stabilization_coefficients(stabilization, space: DGSpace) -> tuple[int, float, np.ndarray]:
    """Return a compact Numba descriptor for advection stabilization.

    The fused kernels evaluate ``tau`` on face quadrature.  ``kind=0`` selects
    the built-in upwind value ``abs(beta_h.n)``, ``kind=1`` uses a scalar, and
    ``kind=2`` evaluates a same-space DG coefficient field, and ``kind=3``
    selects ``tau_scalar*abs(beta_h.n)`` for scaled upwind stabilization.
    ``kind=4`` builds conflict-averaged weights inside the element loop. Callable
    stabilizations must be projected before using the fused backend.
    """
    from hdgfem.hdg.stabilization import upwind_factor, is_conflict_averaged_upwind
    if is_conflict_averaged_upwind(stabilization):
        return 4, 1.0, np.zeros((1, 1), dtype=np.float64)
    factor = upwind_factor(stabilization)
    if stabilization is not None and factor is not None:
        return 3, factor, np.zeros((1, 1), dtype=np.float64)
    if stabilization is None:
        return 0, 0.0, np.zeros((1, 1), dtype=np.float64)
    if np.isscalar(stabilization):
        return 1, float(stabilization), np.zeros((1, 1), dtype=np.float64)
    if isinstance(stabilization, DGField):
        return 2, 0.0, _same_space_field_coefficients(
            stabilization,
            space,
            "advection_stabilization",
        )
    if callable(stabilization):
        raise TypeError(
            "assembly_backend='numba' requires advection_stabilization to be "
            "None, a scalar, or a DGField. Project callable stabilizations "
            "before calling the solver."
        )
    return 2, 0.0, _same_space_field_coefficients(
        stabilization,
        space,
        "advection_stabilization",
    )


def _advection_trace_weight_tables(
        space: DGSpace,
        beta_coeffs: np.ndarray,
        tau_kind: int,
        tau_scalar: float,
        tau_coeffs: np.ndarray,
        *,
        trace_space: DGTraceSpace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Allocate side weights and precompute policies without neighbor coupling.

    Kind 4 deliberately returns uninitialized storage: each owning element
    fills its weights during assembly/reconstruction from immutable beta data.
    """
    mesh = space.mesh
    trace_ref = _trace_ref(space, trace_space)
    tau_face_values = np.empty((mesh.num_tri, 3, trace_ref.weights.size), dtype=np.float64)
    gamma_face_values = np.empty_like(tau_face_values)
    if tau_kind == 4:
        return tau_face_values, gamma_face_values
    assemble_face_trace_weights_kernel(
        tau_face_values,
        gamma_face_values,
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        np.ascontiguousarray(trace_ref.bas_of_bd_quads, dtype=np.float64),
        beta_coeffs,
        int(tau_kind),
        float(tau_scalar),
        np.ascontiguousarray(tau_coeffs, dtype=np.float64),
    )
    return tau_face_values, gamma_face_values


def _zero_boundary_face_values(space: DGSpace, *tables: np.ndarray) -> None:
    """Set exterior local-face quadrature values to zero in-place."""
    mesh = space.mesh
    edge_is_boundary = np.zeros(mesh.num_edg, dtype=bool)
    edge_is_boundary[mesh.bnd_edges_inds] = True
    boundary_faces = edge_is_boundary[mesh.loc2glob_edge]
    for table in tables:
        table[boundary_faces] = 0.0


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


def _full_edge_order_map(space: DGSpace, edge_order: np.ndarray | None) -> np.ndarray:
    """Return the full-system global-edge to solve-edge map."""
    return _edge_order_to_solve_map(
        space.mesh.num_edg,
        np.arange(space.mesh.num_edg, dtype=np.int64),
        edge_order,
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
        advection_stabilization=None,
        trace_space: DGTraceSpace | None = None,
) -> NumbaProjectedTraceAssembly:
    """Assemble the full HDG trace system with fused Numba kernels.

    This backend is the fast projected-coefficient path.  The PDE coefficients
    must already be represented in the solution space:

    ``source``
        :class:`DGField` in ``space``.
    ``beta_field``
        two-component :class:`VectorDGField` with both components in ``space``.
    ``reaction``
        :class:`DGField` in ``space``; use ``space.zeros`` or ``space.constant``
        for exact zero/constant coefficients.

    Python callables, scalars, and loose coefficient arrays are intentionally
    rejected here.  Convert them through the owning :class:`DGSpace` before
    calling the solver.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    source_coeffs, source_kind = _source_coefficients(source, space)
    beta_coeffs = _same_space_vector_coefficients(beta_field, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    tau_kind, tau_scalar, tau_coeffs = _advection_stabilization_coefficients(
        advection_stabilization,
        space,
    )
    interior_side_index = _interior_side_index(space.mesh)
    timings["coefficient_validation"] = time.perf_counter() - start

    start = time.perf_counter()
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_ref)
    if beta_dot_normal is None:
        beta_dot_normal = hdg_coefficients.advective_boundary_normal(beta_field, space, trace_space=trace_ref)
    else:
        beta_dot_normal = np.ascontiguousarray(beta_dot_normal, dtype=np.float64)
    timings["boundary_trace_and_flux"] = time.perf_counter() - start

    mesh = space.mesh
    q = space.quad_data
    edg_dof = trace_ref.edg_dof
    edge_to_solve_edge = _full_edge_order_map(space, edge_order)

    start = time.perf_counter()
    tau_face_values, gamma_face_values = _advection_trace_weight_tables(
        space,
        beta_coeffs,
        tau_kind,
        tau_scalar,
        tau_coeffs,
        trace_space=trace_ref,
    )
    timings["trace_weights"] = time.perf_counter() - start

    n_interior_flux = mesh.int_edges_inds.size * 2 * edg_dof * 3 * edg_dof
    n_interior_mass = mesh.interior_elements.size * edg_dof * edg_dof
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
        np.ascontiguousarray(trace_ref.bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(trace_ref.weights, dtype=np.float64),
        np.ascontiguousarray(trace_ref.bas1d_of_ref_edg_qds, dtype=np.float64),
        np.ascontiguousarray(trace_ref.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(trace_ref.face_trace_test_element_trial_oriented, dtype=np.float64),
        int(trace_orientation_mode),
        source_coeffs,
        int(source_kind),
        beta_coeffs,
        tau_face_values,
        gamma_face_values,
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        float(boundary_penalty),
        tau_kind == 4,
        mesh.edge_side_indices,
        False,
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
        boundary_condition: Callable | None,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
        beta_dot_normal: np.ndarray | None = None,
        advection_stabilization=None,
        return_block_coo: bool = False,
        zero_boundary_flux: bool = False,
        trace_space: DGTraceSpace | None = None,
) -> NumbaProjectedTraceAssembly:
    """Assemble the reduced trace system with boundary dofs eliminated in Numba."""
    if zero_boundary_flux and boundary_condition is not None:
        raise ValueError("boundary_condition must be None when boundary_mode='zero-flux'")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    source_coeffs, source_kind = _source_coefficients(source, space)
    beta_coeffs = _same_space_vector_coefficients(beta_field, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    tau_kind, tau_scalar, tau_coeffs = _advection_stabilization_coefficients(
        advection_stabilization,
        space,
    )
    interior_side_index = _interior_side_index(space.mesh)
    timings["coefficient_validation"] = time.perf_counter() - start

    start = time.perf_counter()
    if zero_boundary_flux:
        boundary_trace = np.zeros((space.mesh.num_edg, trace_ref.edg_dof), dtype=np.float64)
    else:
        if boundary_condition is None:
            raise ValueError("boundary_condition is required unless zero_boundary_flux=True")
        boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_ref)
    if beta_dot_normal is None:
        beta_dot_normal = hdg_coefficients.advective_boundary_normal(beta_field, space, trace_space=trace_ref)
    else:
        beta_dot_normal = np.ascontiguousarray(beta_dot_normal, dtype=np.float64)
    timings["boundary_trace_and_flux"] = time.perf_counter() - start

    mesh = space.mesh
    q = space.quad_data
    edg_dof = trace_ref.edg_dof

    start = time.perf_counter()
    tau_face_values, gamma_face_values = _advection_trace_weight_tables(
        space,
        beta_coeffs,
        tau_kind,
        tau_scalar,
        tau_coeffs,
        trace_space=trace_ref,
    )
    timings["trace_weights"] = time.perf_counter() - start
    if zero_boundary_flux and tau_kind != 4:
        start = time.perf_counter()
        _zero_boundary_face_values(space, tau_face_values, gamma_face_values)
        timings["boundary_flux_zeroing"] = time.perf_counter() - start

    start = time.perf_counter()
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(
        space,
        boundary_trace,
        edge_order,
        trace_space=trace_ref,
    )
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[mesh.interior_elements], axis=1).astype(np.int64)
    side_flux_offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    side_flux_offsets[0] = 0
    np.cumsum(side_col_counts * edg_dof * edg_dof, out=side_flux_offsets[1:])
    n_flux = int(side_flux_offsets[-1])
    n_mass = mesh.interior_elements.size * edg_dof * edg_dof
    nnz = n_flux + n_mass
    side_block_offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    side_block_offsets[0] = 0
    np.cumsum(side_col_counts, out=side_block_offsets[1:])
    n_flux_blocks = int(side_block_offsets[-1])
    n_mass_blocks = int(mesh.interior_elements.size)
    block_nnz = n_flux_blocks + n_mass_blocks if return_block_coo else 0
    timings["reduction_map"] = time.perf_counter() - start

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty(mesh.num_tri * 3 * edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)
    block_rows = np.empty(block_nnz, dtype=np.int64)
    block_cols = np.empty_like(block_rows)
    block_data = np.empty((block_nnz, edg_dof, edg_dof), dtype=np.float64)

    start = time.perf_counter()
    assemble_projected_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        block_rows,
        block_cols,
        block_data,
        np.ascontiguousarray(side_block_offsets, dtype=np.int64),
        bool(return_block_coo),
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
        np.ascontiguousarray(trace_ref.bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(trace_ref.weights, dtype=np.float64),
        np.ascontiguousarray(trace_ref.bas1d_of_ref_edg_qds, dtype=np.float64),
        np.ascontiguousarray(trace_ref.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(trace_ref.face_trace_test_element_trial_oriented, dtype=np.float64),
        int(trace_orientation_mode),
        source_coeffs,
        int(source_kind),
        beta_coeffs,
        tau_face_values,
        gamma_face_values,
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        tau_kind == 4,
        mesh.edge_side_indices,
        zero_boundary_flux,
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    if return_block_coo:
        timings["block_coo_entries"] = float(block_nnz)

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
        block_rows=block_rows if return_block_coo else None,
        block_cols=block_cols if return_block_coo else None,
        block_data=block_data if return_block_coo else None,
    )


def assemble_projected_trace_system_zero_flux_numba(
        source,
        beta_field: VectorDGField,
        reaction,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
        beta_dot_normal: np.ndarray | None = None,
        advection_stabilization=None,
        return_block_coo: bool = False,
        trace_space: DGTraceSpace | None = None,
) -> NumbaProjectedTraceAssembly:
    """Assemble a reduced trace system with zero numerical flux on the boundary."""
    return assemble_projected_trace_system_eliminated_numba(
        source,
        beta_field,
        reaction,
        None,
        space,
        edge_order=edge_order,
        beta_dot_normal=beta_dot_normal,
        advection_stabilization=advection_stabilization,
        return_block_coo=return_block_coo,
        zero_boundary_flux=True,
        trace_space=trace_space,
    )


def reconstruct_projected_field_numba(
        trace: np.ndarray,
        source,
        beta_field: VectorDGField,
        reaction,
        space: DGSpace,
        *,
        advection_stabilization=None,
        zero_boundary_flux: bool = False,
        name: str = "u_h",
        trace_space: DGTraceSpace | None = None,
) -> DGField:
    """Recover element coefficients with the projected fused Numba backend."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    source_coeffs, source_kind = _source_coefficients(source, space)
    beta_coeffs = _same_space_vector_coefficients(beta_field, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    tau_kind, tau_scalar, tau_coeffs = _advection_stabilization_coefficients(
        advection_stabilization,
        space,
    )
    trace = np.ascontiguousarray(np.asarray(trace, dtype=np.float64))
    expected_trace_shape = (space.mesh.num_edg * trace_ref.edg_dof,)
    if trace.shape != expected_trace_shape:
        raise ValueError(f"trace must have shape {expected_trace_shape}; got {trace.shape}")

    coeffs = np.empty(space.shape, dtype=np.float64)
    mesh = space.mesh
    q = space.quad_data
    tau_face_values, gamma_face_values = _advection_trace_weight_tables(
        space,
        beta_coeffs,
        tau_kind,
        tau_scalar,
        tau_coeffs,
        trace_space=trace_ref,
    )
    if zero_boundary_flux and tau_kind != 4:
        _zero_boundary_face_values(space, tau_face_values, gamma_face_values)
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
        np.ascontiguousarray(trace_ref.bas_of_bd_quads, dtype=np.float64),
        np.ascontiguousarray(trace_ref.weights, dtype=np.float64),
        np.ascontiguousarray(trace_ref.bas1d_of_ref_edg_qds, dtype=np.float64),
        int(trace_orientation_mode),
        source_coeffs,
        int(source_kind),
        beta_coeffs,
        tau_face_values,
        gamma_face_values,
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        tau_kind == 4,
        mesh.edge_side_indices,
        zero_boundary_flux,
    )
    return space.field(coeffs, name=name)
