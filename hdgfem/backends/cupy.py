"""CuPy adapters for HDG assembly and PyAMGX solves.

The public solvers keep using :class:`DGSpace`, :class:`DGField`, and NumPy
result arrays. This module owns persistent device mirrors of package mesh and
reference-element data so CuPy assembly does not repeatedly copy static tables
from host RAM to VRAM.
"""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from hdgfem.hdg import condensation as hdg_assembly
import hdgfem.hdg.stabilization as hdg_stabilization
import hdgfem.hdg.coefficients as hdg_coefficients
from hdgfem.core.space import DGField, DGSpace, DGTraceSpace, VectorDGField
from hdgfem.linalg.reduction import KnownDofReduction

try:  # pragma: no cover - depends on optional runtime dependency.
    import cupy as cp
except ImportError as error:  # pragma: no cover
    cp = None
    _CUPY_IMPORT_ERROR = error
else:  # pragma: no cover
    _CUPY_IMPORT_ERROR = None

try:  # pragma: no cover - depends on optional runtime dependency.
    import cupyx.scipy.sparse as cupyx_sparse
except ImportError as error:  # pragma: no cover
    cupyx_sparse = None
    _CUPYX_SPARSE_IMPORT_ERROR = error
else:  # pragma: no cover
    _CUPYX_SPARSE_IMPORT_ERROR = None

try:  # pragma: no cover - depends on optional runtime dependency.
    import cupyx.scipy.sparse.linalg as cupyx_sparse_linalg
except ImportError as error:  # pragma: no cover
    cupyx_sparse_linalg = None
    _CUPYX_SPARSE_LINALG_IMPORT_ERROR = error
else:  # pragma: no cover
    _CUPYX_SPARSE_LINALG_IMPORT_ERROR = None

try:  # pragma: no cover - depends on optional runtime dependency.
    import pyamgx
except ImportError as error:  # pragma: no cover
    pyamgx = None
    _PYAMGX_IMPORT_ERROR = error
else:  # pragma: no cover
    _PYAMGX_IMPORT_ERROR = None
from hdgfem.runtime.optional import (
    require_cupy,
)
from hdgfem.core.device import (
    CupyDGSpace,
    CupyTraceReferenceData,
    as_cupy_coefficients,
    as_cupy_space,
    as_cupy_trace_reference,
    as_cupy_vector_coefficients,
)


@dataclass(frozen=True)
class CupyAdvectionTraceAssembly:
    """Advection-reaction trace data assembled with CuPy."""

    trace_system: hdg_assembly.TraceSystem | None
    beta_dot_normal: np.ndarray
    local_solver: Any | None
    element_boundary_mats: Any | None
    timings: dict[str, float]
    reduction: KnownDofReduction | None = None
    rows_device: Any | None = None
    cols_device: Any | None = None
    data_device: Any | None = None
    rhs_device: Any | None = None
    boundary_trace: np.ndarray | None = None
    local_solver_device: Any | None = None
    element_boundary_mats_device: Any | None = None


def expand_boundary_trace_cupy(
        reduced_solution,
        boundary_trace,
        space: DGSpace | CupyDGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
):
    """Insert a reduced interior-edge trace into its full device trace table."""
    cupy = require_cupy()
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_reference(trace_space, cspace)
    reduced_cp = cupy.asarray(reduced_solution, dtype=REAL_DTYPE)
    expected = int(cspace.mesh.int_edges_inds.size * trace_ref.edg_dof)
    if reduced_cp.shape != (expected,):
        raise ValueError(f"reduced_solution must have shape ({expected},); got {reduced_cp.shape}")
    full = cupy.asarray(boundary_trace, dtype=REAL_DTYPE).copy()
    full[cspace.mesh.int_edges_inds] = reduced_cp.reshape((-1, trace_ref.edg_dof))
    return cupy.ascontiguousarray(full.ravel())


def element_traces_cupy(
        trace,
        space: DGSpace | CupyDGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
):
    """Gather and orient global trace coefficients entirely on-device."""
    cupy = require_cupy()
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_reference(trace_space, cspace)
    expected_size = int(cspace.mesh.num_edg * trace_ref.edg_dof)
    trace_cp = cupy.asarray(trace, dtype=REAL_DTYPE)
    if trace_cp.shape != (expected_size,):
        raise ValueError(f"trace must have shape ({expected_size},); got {trace_cp.shape}")

    traces = trace_cp.reshape((cspace.mesh.num_edg, trace_ref.edg_dof))[
        cspace.mesh.loc2glob_edge
    ].copy()
    if cspace.mesh.num_negative_orientations:
        elements = cspace.mesh.negative_orientation_elements
        faces = cspace.mesh.negative_orientation_faces
        negative = traces[elements, faces]
        if trace_ref.kind == "legendre-modal":
            signs = cupy.where(
                cupy.arange(trace_ref.edg_dof, dtype=cupy.int64) % 2 == 0,
                REAL_DTYPE(1.0),
                REAL_DTYPE(-1.0),
            )
            traces[elements, faces] = negative * signs[None, :]
        else:
            traces[elements, faces] = negative[:, ::-1]
    return cupy.ascontiguousarray(
        traces.reshape((cspace.mesh.num_tri, 3 * trace_ref.edg_dof))
    )


def _normalize_values(values, num_elements: int, num_points: int, label: str) -> np.ndarray:
    """Normalize scalar/quadrature values to ``(num_elements, num_points)``."""
    values = np.asarray(values, dtype=REAL_DTYPE)
    if values.shape == (num_elements, num_points):
        return np.ascontiguousarray(values)
    if values.shape == (num_points,):
        return np.ascontiguousarray(np.broadcast_to(values[None, :], (num_elements, num_points)))
    if values.ndim == 0:
        return np.full((num_elements, num_points), float(values), dtype=REAL_DTYPE)
    raise ValueError(
        f"{label} must be a scalar, have shape ({num_points},), or have shape "
        f"({num_elements}, {num_points}); got {values.shape}"
    )


def _beta_values_on_volume(
        beta_field: VectorDGField | None,
        beta_callables: tuple[Callable, Callable] | None,
        cspace: CupyDGSpace,
):
    """Return advection values on volume quadrature points as a CuPy array."""
    cupy = require_cupy()
    space = cspace.host
    num_elements = space.mesh.num_tri
    num_points = space.quad_data.Krf_w.size
    values = np.empty((num_elements, num_points, 2), dtype=REAL_DTYPE)
    if beta_field is not None:
        if beta_field.dim != 2:
            raise ValueError("beta_field must have two components")
        beta_field.components[0].space.assert_same_mesh(space)
        beta_field.components[1].space.assert_same_mesh(space)
        coeffs = as_cupy_vector_coefficients(beta_field, cspace)
        return cupy.ascontiguousarray(cupy.einsum("dKi,iq->Kqd", coeffs, cspace.quad_data.bas_of_quads))

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
    return cupy.asarray(np.ascontiguousarray(values))


def _reaction_values_on_volume(reaction, cspace: CupyDGSpace) -> np.ndarray:
    """Return reaction values on volume quadrature points as a host array."""
    space = cspace.host
    num_elements = space.mesh.num_tri
    num_points = space.quad_data.Krf_w.size
    if np.isscalar(reaction):
        return np.full((num_elements, num_points), float(reaction), dtype=REAL_DTYPE)
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(space)
        return np.ascontiguousarray(reaction.values_at_ref(space.quad_data.Krf_quads), dtype=REAL_DTYPE)
    if callable(reaction):
        points = space.mapped_quads()
        return _normalize_values(
            reaction(points[:, :, 0], points[:, :, 1]),
            num_elements,
            num_points,
            "reaction",
        )

    values = np.asarray(reaction, dtype=REAL_DTYPE)
    if values.shape == (num_elements, num_points):
        return np.ascontiguousarray(values)
    if values.shape == (num_points,):
        return np.ascontiguousarray(np.broadcast_to(values[None, :], (num_elements, num_points)))
    if values.shape == space.shape:
        return np.ascontiguousarray(space.field(values, name="reaction").values_at_ref(space.quad_data.Krf_quads))
    raise TypeError("reaction must be a scalar, callable, DGField, quadrature values, or DG coefficients")


def _reaction_mass_cupy(reaction, cspace: CupyDGSpace):
    """Assemble reaction mass matrices on the GPU."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    if np.isscalar(reaction):
        return float(reaction) * mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(cspace.host)
        constant_value = reaction.constant_value
        if constant_value is not None:
            if constant_value == 0.0:
                return 0.0
            return constant_value * mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
        coeffs = as_cupy_coefficients(reaction, cspace)
        values = coeffs @ q.bas_of_quads
    elif callable(reaction):
        mapped_quads = cupy.einsum(
            "Krc,qc->Krq",
            mesh.aff_mats,
            q.Krf_quads,
        ) + mesh.aff_vecs[:, :, None]
        try:
            values = cupy.asarray(reaction(mapped_quads[:, 0, :], mapped_quads[:, 1, :]), dtype=REAL_DTYPE)
        except Exception:
            values = cupy.asarray(_reaction_values_on_volume(reaction, cspace))
    else:
        values = cupy.asarray(_reaction_values_on_volume(reaction, cspace))
    scaled_values = values * mesh.aff_jacs[:, None]
    flat = scaled_values @ q.weighted_phi_phi_flat
    return flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof)


def _advection_mats_cupy(
        cspace: CupyDGSpace,
        *,
        beta_field: VectorDGField | None,
        beta_callables: tuple[Callable, Callable] | None,
):
    """Assemble local advection matrices on the GPU."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    beta_values = _beta_values_on_volume(beta_field, beta_callables, cspace)
    scaled_inv_t = mesh.aff_jacs[:, None, None] * mesh.inv_aff_mats_t
    return cupy.einsum(
        "Kqd,KdD,jq,Diq,q->Kij",
        beta_values,
        scaled_inv_t,
        q.bas_of_quads,
        q.dbas_of_quads,
        q.Krf_w,
        optimize=True,
    )


def _advection_trace_weights_cupy(
        stabilization,
        cspace: CupyDGSpace,
        beta_dot_normal,
        trace_ref: CupyTraceReferenceData,
):
    """Return device tau and gamma tables on element-side face quadrature."""
    cupy = require_cupy()
    from hdgfem.hdg.stabilization import (
            upwind_factor,
            effective_advection_normal_flux,
            is_conflict_averaged_upwind,
        )
    if is_conflict_averaged_upwind(stabilization):
        beta_dot_normal = effective_advection_normal_flux(beta_dot_normal, cspace.mesh, stabilization, xp=cupy)
    factor = upwind_factor(stabilization)
    if factor is not None:
        tau_face = factor * cupy.abs(beta_dot_normal)
    elif np.isscalar(stabilization):
        tau_face = cupy.full(beta_dot_normal.shape, float(stabilization), dtype=REAL_DTYPE)
    elif isinstance(stabilization, DGField):
        stabilization.space.assert_same_mesh(cspace.host)
        constant_value = stabilization.constant_value
        if constant_value is not None:
            tau_face = cupy.full(beta_dot_normal.shape, constant_value, dtype=REAL_DTYPE)
        else:
            field_cspace = as_cupy_space(stabilization.space, device=cspace.device_id)
            coefficients = as_cupy_coefficients(stabilization, field_cspace)
            if stabilization.space is cspace.host:
                face_basis = trace_ref.bas_of_bd_quads
            else:
                face_basis = cupy.asarray(
                    hdg_coefficients.dg_field_basis_on_trace_faces(stabilization.space, trace_ref.host),
                    dtype=REAL_DTYPE,
                )
            tau_face = cupy.einsum("Ki,fiq->Kfq", coefficients, face_basis, optimize=True)
    else:
        tau_host = hdg_stabilization.advection_trace_stabilization_values(
            cspace.host,
            cupy.asnumpy(beta_dot_normal),
            stabilization,
            trace_space=trace_ref.host,
        )
        tau_face = cupy.asarray(tau_host, dtype=REAL_DTYPE)
    tau_face = cupy.ascontiguousarray(tau_face)
    return tau_face, cupy.ascontiguousarray(tau_face - beta_dot_normal)


def _boundary_mass_from_trace_stabilization_cupy(cspace: CupyDGSpace, tau_face, trace_ref: CupyTraceReferenceData):
    """Assemble tau-weighted boundary mass matrices on the GPU."""
    cupy = require_cupy()
    mesh = cspace.mesh
    return cupy.einsum(
        "Kf,Kfq,fiq,fjq->Kij",
        mesh.jacs_el_fc,
        tau_face,
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas_of_bd_quads,
        optimize=True,
    )


def _element_boundary_mats_from_trace_weight_cupy(cspace: CupyDGSpace, gamma_face, trace_ref: CupyTraceReferenceData):
    """Assemble gamma-weighted element-to-trace coupling matrices on the GPU."""
    cupy = require_cupy()
    mesh = cspace.mesh
    result = cupy.einsum(
        "Kf,Kfq,fiq,jq->Kifj",
        mesh.jacs_el_fc,
        gamma_face,
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas1d_of_ref_edg_qds,
        optimize=True,
    )
    return result.reshape(mesh.num_tri, cspace.el_dof, 3 * trace_ref.edg_dof)


def reconstruct_advection_reaction_field_cupy(
        trace,
        source_rhs,
        beta_field: VectorDGField | None,
        beta_callables: tuple[Callable, Callable] | None,
        beta_dot_normal,
        reaction,
        space: DGSpace | CupyDGSpace,
        *,
        advection_stabilization=None,
        trace_space: DGTraceSpace | None = None,
        local_solver_device=None,
        element_boundary_mats_device=None,
):
    """Reconstruct element coefficients on-device, rebuilding uncached operators.

    The condensed local inverse and element-boundary matrices are intentionally
    not retained by default. If explicitly cached, they are reused directly;
    otherwise reconstruction performs one fresh batched local solve.
    """
    cupy = require_cupy()
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_reference(trace_space, cspace)
    has_local_solver = local_solver_device is not None
    has_boundary_mats = element_boundary_mats_device is not None
    if has_local_solver != has_boundary_mats:
        raise ValueError(
            "cached local solver and element-boundary matrices must be supplied together"
        )
    if has_local_solver:
        traces = element_traces_cupy(trace, cspace, trace_space=trace_ref.host)
        source_cp = cupy.asarray(source_rhs, dtype=REAL_DTYPE)
        boundary_cp = cupy.asarray(element_boundary_mats_device, dtype=REAL_DTYPE)
        rhs = source_cp[..., None] + boundary_cp @ traces[..., None]
        solver_cp = cupy.asarray(local_solver_device, dtype=REAL_DTYPE)
        return cupy.ascontiguousarray((solver_cp @ rhs).squeeze(-1))

    beta_normal_cp = cupy.asarray(beta_dot_normal, dtype=REAL_DTYPE)
    tau_face, gamma_face = _advection_trace_weights_cupy(
        advection_stabilization,
        cspace,
        beta_normal_cp,
        trace_ref,
    )
    local_mats = _boundary_mass_from_trace_stabilization_cupy(cspace, tau_face, trace_ref)
    local_mats += _reaction_mass_cupy(reaction, cspace)
    local_mats -= _advection_mats_cupy(
        cspace,
        beta_field=beta_field,
        beta_callables=beta_callables,
    )
    element_boundary = _element_boundary_mats_from_trace_weight_cupy(
        cspace,
        gamma_face,
        trace_ref,
    )
    traces = element_traces_cupy(trace, cspace, trace_space=trace_ref.host)
    source_cp = cupy.asarray(source_rhs, dtype=REAL_DTYPE)
    rhs = source_cp[..., None] + element_boundary @ traces[..., None]
    return cupy.ascontiguousarray(cupy.linalg.solve(local_mats, rhs).squeeze(-1))


def _oriented_trace_basis_cupy(cspace: CupyDGSpace, trace_ref: CupyTraceReferenceData):
    """Return trace basis values in global edge orientation on every side."""
    from hdgfem.core.device import as_cupy_trace_space
    from hdgfem.backends.advection_cuda import oriented_trace_basis_cupy
    trace = as_cupy_trace_space(trace_ref.host, device=cspace.device_id)
    return oriented_trace_basis_cupy(cspace, trace)


def _advection_trace_lift_cupy(cspace: CupyDGSpace, tau_face, trace_ref: CupyTraceReferenceData):
    """Return the tau-weighted row lift used by advection trace equations."""
    cupy = require_cupy()
    mesh = cspace.mesh
    oriented_trace = _oriented_trace_basis_cupy(cspace, trace_ref)
    return cupy.ascontiguousarray(
        cupy.einsum(
            "Kf,Kfq,Kfaq,fiq,q->Kfai",
            mesh.jacs_el_fc,
            tau_face,
            oriented_trace,
            trace_ref.bas_of_bd_quads,
            trace_ref.weights,
            optimize=True,
        )
    )


def _element_to_trace_matrix_cupy(
        local_solver,
        element_boundary_mats,
        trace_lift,
        cspace: CupyDGSpace,
        trace_ref: CupyTraceReferenceData,
):
    """Assemble oriented element Schur complement blocks on the GPU."""
    cupy = require_cupy()
    mesh = cspace.mesh
    schur = trace_lift @ (local_solver @ element_boundary_mats)[:, None, :, :]
    edg_dof = trace_ref.edg_dof
    schur = schur.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    if mesh.num_negative_orientations:
        neg = schur[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :]
        if trace_ref.kind == "legendre-modal":
            signs = cupy.where(cupy.arange(edg_dof, dtype=cupy.int64) % 2 == 0, REAL_DTYPE(1.0), REAL_DTYPE(-1.0))
            schur[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :] = neg * signs
        else:
            schur[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :] = neg[..., ::-1]
    return cupy.ascontiguousarray(schur.swapaxes(2, 3))


def _advection_interior_trace_mass_blocks_cupy(cspace: CupyDGSpace, gamma_face, trace_ref: CupyTraceReferenceData, *, inactive_tau=None):
    """Return side-wise interior trace masses for explicit gamma weights."""
    cupy = require_cupy()
    mesh = cspace.mesh
    oriented_trace = _oriented_trace_basis_cupy(cspace, trace_ref)
    side_blocks = cupy.einsum(
        "Kf,Kfq,Kfaq,Kfbq,q->Kfab",
        mesh.jacs_el_fc,
        gamma_face,
        oriented_trace,
        oriented_trace,
        trace_ref.weights,
        optimize=True,
    )
    blocks = cupy.ascontiguousarray(side_blocks[mesh.interior_elements, mesh.interior_faces])
    if inactive_tau is not None:
        from hdgfem.hdg.stabilization import gauge_inactive_advection_trace_blocks
        gauge_inactive_advection_trace_blocks(blocks, inactive_tau, mesh, xp=cupy)
    return blocks


def _trace_matrix_data_cupy(
        trace_blocks,
        cspace: CupyDGSpace,
        trace_ref: CupyTraceReferenceData,
        gamma_face,
        boundary_penalty: float,
        *, inactive_tau=None,
):
    """Return COO data values matching :func:`hdg_assembly.trace_matrix_indices`."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = trace_ref.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_interior_flux = mesh.int_edges_inds.size * 2 * edg_dof * 3 * edg_dof
    n_interior_mass = valid_elements.size * edg_dof * edg_dof
    n_boundary = mesh.bnd_edges_inds.size * edg_dof
    data = cupy.empty(n_interior_flux + n_interior_mass + n_boundary, dtype=REAL_DTYPE)

    data[:n_interior_flux] = -trace_blocks[valid_elements, valid_faces].ravel()

    offset = n_interior_flux
    data[offset:offset + n_interior_mass] = _advection_interior_trace_mass_blocks_cupy(
        cspace,
        gamma_face,
        trace_ref,
        inactive_tau=inactive_tau,
    ).ravel()

    offset += n_interior_mass
    data[offset:] = float(boundary_penalty)
    return data


def _reduced_trace_matrix_indices_cupy(cspace: CupyDGSpace, trace_ref: CupyTraceReferenceData):
    """Return full-edge-numbered COO indices before boundary block elimination."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = trace_ref.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_interior_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_interior_mass = valid_elements.size * edg_dof * edg_dof
    rows = cupy.empty(n_interior_flux + n_interior_mass, dtype=cupy.int64)
    cols = cupy.empty_like(rows)

    i_grid, j_grid = cupy.meshgrid(cupy.arange(edg_dof), cupy.arange(edg_dof), indexing="ij")
    row_edges = mesh.loc2glob_edge[valid_elements, valid_faces]
    col_edges = mesh.loc2glob_edge[valid_elements]
    rows[:n_interior_flux] = cupy.broadcast_to(
        row_edges[:, None, None, None] * edg_dof + i_grid[None, None, :, :],
        (valid_elements.size, 3, edg_dof, edg_dof),
    ).ravel()
    cols[:n_interior_flux] = (col_edges[:, :, None, None] * edg_dof + j_grid[None, None, :, :]).ravel()

    l0 = cupy.broadcast_to(cupy.arange(edg_dof)[:, None], (edg_dof, edg_dof)).ravel()
    l1 = cupy.broadcast_to(cupy.arange(edg_dof)[None, :], (edg_dof, edg_dof)).ravel()
    offset = n_interior_flux
    mass_edges = mesh.loc2glob_edge[valid_elements, valid_faces]
    rows[offset:] = (mass_edges[:, None] * edg_dof + l0[None, :]).ravel()
    cols[offset:] = (mass_edges[:, None] * edg_dof + l1[None, :]).ravel()
    return rows, cols


def _reduced_trace_matrix_data_cupy(
        trace_blocks,
        cspace: CupyDGSpace,
        trace_ref: CupyTraceReferenceData,
        gamma_face,
        *, inactive_tau=None,
):
    """Return COO data values before boundary block elimination."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = trace_ref.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_interior_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_interior_mass = valid_elements.size * edg_dof * edg_dof
    data = cupy.empty(n_interior_flux + n_interior_mass, dtype=REAL_DTYPE)

    data[:n_interior_flux] = -trace_blocks[valid_elements, valid_faces].ravel()
    data[n_interior_flux:] = _advection_interior_trace_mass_blocks_cupy(
        cspace,
        gamma_face,
        trace_ref,
        inactive_tau=inactive_tau,
    ).ravel()
    return data


def _global_rhs_without_boundary_penalty_cupy(
        source_rhs,
        local_solver,
        trace_lift,
        cspace: CupyDGSpace,
        trace_ref: CupyTraceReferenceData,
):
    """Assemble the full trace RHS without prescribed-boundary penalty rows."""
    cupy = require_cupy()
    mesh = cspace.mesh
    source_rhs_cp = cupy.asarray(np.ascontiguousarray(source_rhs, dtype=REAL_DTYPE))
    face_rhs = (trace_lift @ (local_solver @ source_rhs_cp[..., None])[:, None, :, :]).squeeze(-1)
    rhs = cupy.zeros((mesh.num_edg, trace_ref.edg_dof), dtype=REAL_DTYPE)
    if mesh.interior_elements.size:
        cupy.add.at(
            rhs,
            mesh.loc2glob_edge[mesh.interior_elements, mesh.interior_faces],
            face_rhs[mesh.interior_elements, mesh.interior_faces],
        )
    return rhs.ravel()


def _eliminate_boundary_trace_dofs_cupy(
        row,
        col,
        data,
        rhs,
        boundary_trace,
        cspace: CupyDGSpace,
        trace_ref: CupyTraceReferenceData,
        *,
        transfer_host: bool = True,
):
    """Eliminate boundary trace columns on-device using the legacy block layout."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = trace_ref.edg_dof
    row_r = row.reshape((row.size // edg_dof, edg_dof))
    col_r = col.reshape((col.size // edg_dof, edg_dof))
    data_r = data.reshape((data.size // edg_dof, edg_dof))

    boundary_starts = mesh.bnd_edges_inds * edg_dof
    col_is_boundary = cupy.isin(col_r[:, 0], boundary_starts)
    keep_blocks = cupy.where(~col_is_boundary)[0]
    remove_blocks = cupy.where(col_is_boundary)[0]
    keep_count = int(keep_blocks.size)

    free_mask_cp = cupy.ones(rhs.size, dtype=cupy.bool_)
    boundary_dofs = (mesh.bnd_edges_inds[:, None] * edg_dof + cupy.arange(edg_dof)[None, :]).ravel()
    free_mask_cp[boundary_dofs] = False
    old_to_new_cp = cupy.full(rhs.size, -1, dtype=cupy.int64)
    old_to_new_cp[free_mask_cp] = cupy.arange(int(free_mask_cp.sum()), dtype=cupy.int64)

    reduced_rows = cupy.empty(keep_count * edg_dof, dtype=cupy.int64)
    reduced_cols = cupy.empty_like(reduced_rows)
    reduced_data = cupy.empty(keep_count * edg_dof, dtype=REAL_DTYPE)
    reduced_rows.reshape((keep_count, edg_dof))[:] = old_to_new_cp[row_r[keep_blocks]]
    reduced_cols.reshape((keep_count, edg_dof))[:] = old_to_new_cp[col_r[keep_blocks]]
    reduced_data.reshape((keep_count, edg_dof))[:] = data_r[keep_blocks]

    boundary_trace_cp = cupy.asarray(np.ascontiguousarray(boundary_trace, dtype=REAL_DTYPE))
    if remove_blocks.size:
        row_ids = row_r[remove_blocks, 0]
        col_edges = col_r[remove_blocks, 0] // edg_dof
        cupy.add.at(
            rhs,
            row_ids,
            cupy.sum(-data_r[remove_blocks] * boundary_trace_cp[col_edges], axis=1),
        )
    reduced_rhs = rhs.reshape((mesh.num_edg, edg_dof))[mesh.int_edges_inds].ravel()

    reduction = None
    if transfer_host:
        free_mask = cupy.asnumpy(free_mask_cp)
        old_to_new = cupy.asnumpy(old_to_new_cp)
        known_mask = ~free_mask
        known_values = boundary_trace.ravel()
        reduction = KnownDofReduction(
            rows=np.ascontiguousarray(cupy.asnumpy(reduced_rows)),
            cols=np.ascontiguousarray(cupy.asnumpy(reduced_cols)),
            data=np.ascontiguousarray(cupy.asnumpy(reduced_data)),
            rhs=np.ascontiguousarray(cupy.asnumpy(reduced_rhs)),
            free_mask=np.ascontiguousarray(free_mask),
            known_mask=np.ascontiguousarray(known_mask),
            known_values=np.ascontiguousarray(known_values),
            old_to_new=np.ascontiguousarray(old_to_new),
        )
    return (
        reduction,
        reduced_rows,
        reduced_cols,
        reduced_data,
        reduced_rhs,
    )


def _global_rhs_cupy(
        source_rhs,
        local_solver,
        trace_lift,
        boundary_condition: Callable,
        cspace: CupyDGSpace,
        trace_ref: CupyTraceReferenceData,
        boundary_penalty: float,
):
    """Assemble the trace RHS on the GPU and boundary trace on the host."""
    cupy = require_cupy()
    mesh = cspace.mesh
    host_space = cspace.host
    source_rhs_cp = cupy.asarray(np.ascontiguousarray(source_rhs, dtype=REAL_DTYPE))
    face_rhs = (trace_lift @ (local_solver @ source_rhs_cp[..., None])[:, None, :, :]).squeeze(-1)
    rhs = cupy.zeros((mesh.num_edg, trace_ref.edg_dof), dtype=REAL_DTYPE)
    if mesh.interior_elements.size:
        cupy.add.at(
            rhs,
            mesh.loc2glob_edge[mesh.interior_elements, mesh.interior_faces],
            face_rhs[mesh.interior_elements, mesh.interior_faces],
        )

    boundary_trace = hdg_assembly.boundary_trace_coefficients(
        boundary_condition,
        host_space,
        trace_space=trace_ref.host,
    )
    if mesh.bnd_edges_inds.size:
        rhs[mesh.bnd_edges_inds] = float(boundary_penalty) * cupy.asarray(boundary_trace[host_space.mesh.bnd_edges_inds])
    return rhs.ravel(), boundary_trace


def assemble_advection_reaction_trace_system_cupy(
        source_rhs,
        beta_field: VectorDGField | None,
        beta_callables: tuple[Callable, Callable] | None,
        beta_dot_normal,
        reaction,
        boundary_condition: Callable,
        space: DGSpace | CupyDGSpace,
        *,
        boundary_penalty: float = 1e20,
        transfer_local_solver: bool = False,
        transfer_trace_system: bool = False,
        advection_stabilization=None,
        trace_space: DGTraceSpace | None = None,
) -> CupyAdvectionTraceAssembly:
    """Assemble the full advection-reaction HDG trace system with CuPy.

    COO values and RHS remain device-resident unless ``transfer_trace_system`` is set.
    """
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)
    host_space = cspace.host
    trace_ref = as_cupy_trace_reference(trace_space, cspace)

    start = time.perf_counter()
    beta_dot_normal_cp = cupy.asarray(np.ascontiguousarray(beta_dot_normal, dtype=REAL_DTYPE))
    tau_face, gamma_face = _advection_trace_weights_cupy(
        advection_stabilization,
        cspace,
        beta_dot_normal_cp,
        trace_ref,
    )
    local_mats = _boundary_mass_from_trace_stabilization_cupy(cspace, tau_face, trace_ref)
    local_mats += _reaction_mass_cupy(reaction, cspace)
    local_mats -= _advection_mats_cupy(cspace, beta_field=beta_field, beta_callables=beta_callables)
    element_boundary_mats = _element_boundary_mats_from_trace_weight_cupy(cspace, gamma_face, trace_ref)
    timings["local_assembly"] = time.perf_counter() - start

    start = time.perf_counter()
    local_solver = cupy.linalg.inv(local_mats)
    timings["local_inverse"] = time.perf_counter() - start

    start = time.perf_counter()
    trace_lift = _advection_trace_lift_cupy(cspace, tau_face, trace_ref)
    trace_blocks = _element_to_trace_matrix_cupy(local_solver, element_boundary_mats, trace_lift, cspace, trace_ref)
    rows, cols = hdg_assembly.trace_matrix_indices(
        host_space,
        interior_mass_mode="face",
        trace_space=trace_ref.host,
    )
    rows_device = cupy.asarray(rows, dtype=cupy.int64)
    cols_device = cupy.asarray(cols, dtype=cupy.int64)
    from hdgfem.hdg.stabilization import is_conflict_averaged_upwind
    data = _trace_matrix_data_cupy(trace_blocks, cspace, trace_ref, gamma_face, boundary_penalty,
                                   inactive_tau=tau_face if is_conflict_averaged_upwind(advection_stabilization) else None)
    rhs, boundary_trace = _global_rhs_cupy(
        source_rhs,
        local_solver,
        trace_lift,
        boundary_condition,
        cspace,
        trace_ref,
        boundary_penalty,
    )
    cupy.cuda.get_current_stream().synchronize()
    timings["trace_assembly"] = time.perf_counter() - start

    start = time.perf_counter()
    trace_system = None
    if transfer_trace_system:
        trace_system = hdg_assembly.TraceSystem(
            rows=np.ascontiguousarray(rows),
            cols=np.ascontiguousarray(cols),
            data=cupy.asnumpy(data),
            rhs=cupy.asnumpy(rhs),
            boundary_trace=boundary_trace,
        )
    if transfer_local_solver:
        local_solver_result = np.ascontiguousarray(cupy.asnumpy(local_solver))
        element_boundary_mats_result = np.ascontiguousarray(cupy.asnumpy(element_boundary_mats))
    else:
        local_solver_result = None
        element_boundary_mats_result = None
    beta_dot_normal_host = np.ascontiguousarray(beta_dot_normal, dtype=REAL_DTYPE)
    timings["host_transfer"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())

    return CupyAdvectionTraceAssembly(
        trace_system=trace_system,
        beta_dot_normal=beta_dot_normal_host,
        local_solver=local_solver_result,
        element_boundary_mats=element_boundary_mats_result,
        timings=timings,
        rows_device=rows_device,
        cols_device=cols_device,
        data_device=data,
        rhs_device=rhs,
        boundary_trace=boundary_trace,
        local_solver_device=local_solver if transfer_local_solver else None,
        element_boundary_mats_device=element_boundary_mats if transfer_local_solver else None,
    )


def assemble_advection_reaction_trace_system_eliminated_cupy(
        source_rhs,
        beta_field: VectorDGField | None,
        beta_callables: tuple[Callable, Callable] | None,
        beta_dot_normal,
        reaction,
        boundary_condition: Callable,
        space: DGSpace | CupyDGSpace,
        *,
        transfer_local_solver: bool = False,
        transfer_trace_system: bool = False,
        advection_stabilization=None,
        trace_space: DGTraceSpace | None = None,
) -> CupyAdvectionTraceAssembly:
    """Assemble the reduced advection-reaction HDG trace system with CuPy.

    Reduced COO values and RHS stay on-device unless explicitly transferred.
    """
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_reference(trace_space, cspace)

    start = time.perf_counter()
    beta_dot_normal_cp = cupy.asarray(np.ascontiguousarray(beta_dot_normal, dtype=REAL_DTYPE))
    tau_face, gamma_face = _advection_trace_weights_cupy(
        advection_stabilization,
        cspace,
        beta_dot_normal_cp,
        trace_ref,
    )
    local_mats = _boundary_mass_from_trace_stabilization_cupy(cspace, tau_face, trace_ref)
    local_mats += _reaction_mass_cupy(reaction, cspace)
    local_mats -= _advection_mats_cupy(cspace, beta_field=beta_field, beta_callables=beta_callables)
    element_boundary_mats = _element_boundary_mats_from_trace_weight_cupy(cspace, gamma_face, trace_ref)
    cupy.cuda.get_current_stream().synchronize()
    timings["local_assembly"] = time.perf_counter() - start

    start = time.perf_counter()
    local_solver = cupy.linalg.inv(local_mats)
    cupy.cuda.get_current_stream().synchronize()
    timings["local_inverse"] = time.perf_counter() - start

    start = time.perf_counter()
    trace_lift = _advection_trace_lift_cupy(cspace, tau_face, trace_ref)
    trace_blocks = _element_to_trace_matrix_cupy(local_solver, element_boundary_mats, trace_lift, cspace, trace_ref)
    rows_cp, cols_cp = _reduced_trace_matrix_indices_cupy(cspace, trace_ref)
    from hdgfem.hdg.stabilization import is_conflict_averaged_upwind
    data_cp = _reduced_trace_matrix_data_cupy(trace_blocks, cspace, trace_ref, gamma_face,
                                            inactive_tau=tau_face if is_conflict_averaged_upwind(advection_stabilization) else None)
    rhs_cp = _global_rhs_without_boundary_penalty_cupy(source_rhs, local_solver, trace_lift, cspace, trace_ref)
    boundary_trace = hdg_assembly.boundary_trace_coefficients(
        boundary_condition,
        cspace.host,
        trace_space=trace_ref.host,
    )
    cupy.cuda.get_current_stream().synchronize()
    timings["trace_assembly"] = time.perf_counter() - start

    start = time.perf_counter()
    (
        reduction,
        reduced_rows_device,
        reduced_cols_device,
        reduced_data_device,
        reduced_rhs_device,
    ) = _eliminate_boundary_trace_dofs_cupy(
        rows_cp,
        cols_cp,
        data_cp,
        rhs_cp,
        boundary_trace,
        cspace,
        trace_ref,
        transfer_host=transfer_trace_system,
    )
    cupy.cuda.get_current_stream().synchronize()
    timings["boundary_elimination"] = time.perf_counter() - start

    start = time.perf_counter()
    beta_dot_normal_host = np.ascontiguousarray(beta_dot_normal, dtype=REAL_DTYPE)
    if transfer_local_solver:
        local_solver_result = np.ascontiguousarray(cupy.asnumpy(local_solver))
        element_boundary_mats_result = np.ascontiguousarray(cupy.asnumpy(element_boundary_mats))
    else:
        local_solver_result = None
        element_boundary_mats_result = None
    timings["host_transfer"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())

    trace_system = None
    if transfer_trace_system:
        trace_system = hdg_assembly.TraceSystem(
            rows=reduction.rows,
            cols=reduction.cols,
            data=reduction.data,
            rhs=reduction.rhs,
            boundary_trace=boundary_trace,
        )
    return CupyAdvectionTraceAssembly(
        trace_system=trace_system,
        beta_dot_normal=beta_dot_normal_host,
        local_solver=local_solver_result,
        element_boundary_mats=element_boundary_mats_result,
        timings=timings,
        reduction=reduction,
        rows_device=reduced_rows_device,
        cols_device=reduced_cols_device,
        data_device=reduced_data_device,
        rhs_device=reduced_rhs_device,
        boundary_trace=boundary_trace,
        local_solver_device=local_solver if transfer_local_solver else None,
        element_boundary_mats_device=element_boundary_mats if transfer_local_solver else None,
    )


__all__ = [
    "CupyAdvectionTraceAssembly",
    "assemble_advection_reaction_trace_system_cupy",
    "assemble_advection_reaction_trace_system_eliminated_cupy",
    "element_traces_cupy",
    "expand_boundary_trace_cupy",
    "reconstruct_advection_reaction_field_cupy",
]
