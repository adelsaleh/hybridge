"""CUDA-resident CuPy/raw-CUDA advection-reaction HDG assembly helpers.

This module owns the device execution formalism for the fast advection-reaction
path. Public solvers still accept host :class:`DGSpace` and field objects; this
backend mirrors immutable space/trace data to CUDA and returns device arrays to
callers that can keep the global solve on device, or host arrays when requested.
"""

from __future__ import annotations

from hdgfem.runtime.precision import (
    REAL_DTYPE,
)

import time
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np

from hdgfem.core.space import DGField, DGSpace, DGTraceSpace
from hdgfem.core.device import CupyDGSpace, as_cupy_coefficients, as_cupy_space
from hdgfem.runtime.optional import require_cupy
from hdgfem.hdg.cuda.launch import RawCudaBlockSize
from hdgfem.transport.raw_cuda import (
    assemble_projected_advection_trace_system_eliminated_raw_cuda,
    assemble_projected_advection_trace_system_eliminated_raw_cuda_fused,
    reconstruct_projected_advection_field_raw_cuda,
    reconstruct_projected_advection_field_raw_cuda_fused,
    reconstruct_projected_advection_field_from_response_raw_cuda,
)
from hdgfem.transport.tsle_bsr import (
    RawAdvectionTsleWorkspace,
    assemble_projected_advection_trace_system_eliminated_tsle_bsr,
)
from hdgfem.core.device import CupyDGTraceSpace, as_cupy_trace_space
from hdgfem.hdg.condensation_device import require_finite_device_values, CudaAdvectionAssembly, reconstruct_trace_cupy
from hdgfem.hdg.coefficients_device import source_moments_cupy
from hdgfem.runtime.logging import sync_elapsed
from hdgfem.core.device import mapped_quads_cupy


RawLocalAssembly = Literal["precomputed", "fused", "split3"]
RawLuMode = Literal["safe", "coop"]
CudaAdvectionAssemblyBackend = Literal["cupy", "raw-cuda"]


def project_callable_cupy(func: Callable, cspace: CupyDGSpace, timings: dict[str, float] | str | None = None, key: str | None = None):
    """Project an analytic coefficient into the device DG space."""
    if isinstance(timings, str):
        timings = TIMINGS
    cp = require_cupy()
    start = time.perf_counter()
    q = cspace.quad_data
    points = mapped_quads_cupy(cspace)
    values = cp.asarray(func(points[:, 0, :], points[:, 1, :]), dtype=REAL_DTYPE)
    coeffs = values @ q.projection_operator.T
    if timings is not None and key is not None:
        timings[key] = timings.get(key, 0.0) + sync_elapsed(start)
    return cp.ascontiguousarray(coeffs)


def source_coefficients_cupy(source, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Return device DG coefficients for the source term."""
    cp = require_cupy()
    if isinstance(source, DGField):
        source.space.assert_same_mesh(cspace.host)
        return as_cupy_coefficients(source, cspace)
    return project_callable_cupy(source, cspace, timings, "projection.source")


def reaction_coefficients_cupy(reaction, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Normalize reaction data into device coefficients or a scalar value."""
    cp = require_cupy()
    if isinstance(reaction, (int, float, np.integer, np.floating)):
        return cp.empty(1, dtype=REAL_DTYPE), float(reaction), True
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(cspace.host)
        return as_cupy_coefficients(reaction, cspace), 0.0, False
    coeffs = project_callable_cupy(reaction, cspace, timings, "projection.reaction")
    return coeffs, 0.0, False


def _require_raw_dg_field(value, cspace: CupyDGSpace, label: str) -> DGField:
    """Require a DGField bound to the exact space used by raw CUDA assembly."""
    if isinstance(value, DGField):
        value.space.assert_same_mesh(cspace.host)
        if value.space is not cspace.host:
            raise ValueError(f"{label} must live in the same DGSpace object for assembly_backend='raw-cuda'")
        return value
    if callable(value):
        raise TypeError(
            f"assembly_backend='raw-cuda' requires {label} to be a DGField; "
            "project callables first with space.project_callable(...)."
        )
    if isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"assembly_backend='raw-cuda' requires {label} to be a DGField; "
            "use space.zeros(...) or space.constant(...) for constants."
        )
    raise TypeError(
        f"assembly_backend='raw-cuda' requires {label} to be a DGField; "
        "wrap coefficient arrays with space.field(...)."
    )


def _reference_advection_tensor_host(cspace: CupyDGSpace) -> np.ndarray:
    """Return the cached dense host reference tensor used by all CUDA variants."""
    q = cspace.host.quad_data
    cached = getattr(q, "_hdgfem_reference_advection_tensor", None)
    if cached is None:
        cached = np.ascontiguousarray(
            np.einsum(
                "q,qk,qj,qiD->Dkij",
                q.Krf_w,
                q.phi,
                q.phi,
                q.gphi,
                optimize=True,
            ),
            dtype=REAL_DTYPE,
        )
        object.__setattr__(q, "_hdgfem_reference_advection_tensor", cached)
    return cached


def reference_advection_tensor_cupy(cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Build or reuse the dense reference advection contraction tensor on device."""
    cp = require_cupy()
    start = time.perf_counter()
    q = cspace.host.quad_data
    cache = getattr(q, "_hdgfem_reference_advection_tensor_cupy", None)
    if cache is None:
        cache = {}
        object.__setattr__(q, "_hdgfem_reference_advection_tensor_cupy", cache)
    if cspace.device_id not in cache:
        with cp.cuda.Device(cspace.device_id):
            cache[cspace.device_id] = cp.asarray(_reference_advection_tensor_host(cspace))
    result = cache[cspace.device_id]
    if timings is not None:
        timings["reference_advection_tensor"] = timings.get("reference_advection_tensor", 0.0) + sync_elapsed(start)
    return result


def reference_advection_sparse_cupy(
        cspace: CupyDGSpace,
        timings: dict[str, float] | None = None,
):
    """Return a CSR-by-matrix-entry Dubiner advection contraction."""
    cp = require_cupy()
    start = time.perf_counter()
    q = cspace.host.quad_data
    sparse_candidate = str(q.basis_type) == "dub_orth"
    cached = getattr(q, "_hdgfem_sparse_advection_tensor", None)
    if sparse_candidate and cached is None:
        dense = _reference_advection_tensor_host(cspace)
        nel = int(q.el_dof)
        by_entry = np.ascontiguousarray(dense.transpose(0, 2, 3, 1))
        scale = max(float(np.max(np.abs(by_entry))), 1.0)
        zero_tolerance = 128.0 * np.finfo(REAL_DTYPE).eps * scale
        mask = np.any(np.abs(by_entry) > zero_tolerance, axis=0)
        counts = np.count_nonzero(mask, axis=1).reshape(-1)
        offsets = np.empty(nel * nel + 1, dtype=np.int32)
        offsets[0] = 0
        cumulative = np.cumsum(counts, dtype=np.int64)
        if cumulative.size and int(cumulative[-1]) > np.iinfo(np.int32).max:
            raise OverflowError("sparse advection contraction exceeds int32 indexing")
        offsets[1:] = cumulative
        modes = np.broadcast_to(
            np.arange(nel, dtype=np.int32), (nel, nel, nel)
        )[mask]
        values0 = np.ascontiguousarray(by_entry[0][mask], dtype=REAL_DTYPE)
        values1 = np.ascontiguousarray(by_entry[1][mask], dtype=REAL_DTYPE)
        cached = (offsets, np.ascontiguousarray(modes), values0, values1)
        object.__setattr__(q, "_hdgfem_sparse_advection_tensor", cached)
    # Indirect sparse loads only beat the dense compile-time contraction when
    # enough k entries vanish.  The p=6 Dubiner tensor retains about 57% of its
    # entries and was 18 ms slower on the 157k-element production mesh, so keep
    # that case on the dense path.  Retain sparse support for genuinely sparse
    # lower-order tensors rather than imposing the regression globally.
    sparse_density = (
        float(cached[2].size) / float(int(q.el_dof) ** 3)
        if sparse_candidate and cached is not None else 1.0
    )
    enabled = bool(sparse_candidate and sparse_density <= 0.40)
    device_cache = getattr(q, "_hdgfem_sparse_advection_tensor_cupy", None)
    if device_cache is None:
        device_cache = {}
        object.__setattr__(q, "_hdgfem_sparse_advection_tensor_cupy", device_cache)
    key = int(cspace.device_id)
    if enabled and key not in device_cache:
        device_cache[key] = tuple(cp.asarray(array) for array in cached)
    if enabled:
        offsets, modes, values0, values1 = device_cache[key]
    else:
        offsets = cp.zeros(1, dtype=cp.int32)
        modes = cp.zeros(1, dtype=cp.int32)
        values0 = cp.zeros(1, dtype=REAL_DTYPE)
        values1 = cp.zeros(1, dtype=REAL_DTYPE)
    mass = np.asarray(q.MKrf, dtype=REAL_DTYPE)
    diagonal = np.diag(np.diag(mass))
    mass_is_diagonal = bool(
        np.max(np.abs(mass - diagonal))
        <= 128.0 * np.finfo(REAL_DTYPE).eps * max(float(np.max(np.abs(mass))), 1.0)
    )
    if timings is not None:
        timings["reference_advection_sparse"] = (
            timings.get("reference_advection_sparse", 0.0) + sync_elapsed(start)
        )
        timings["reference_advection_sparse.enabled"] = float(enabled)
        timings["reference_advection_sparse.density"] = float(sparse_density)
        if sparse_candidate and cached is not None:
            timings["reference_advection_sparse.nnz"] = float(cached[2].size)
    return offsets, modes, values0, values1, enabled, mass_is_diagonal


def beta_dot_normal_from_coeffs(beta_coeffs, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace):
    """Evaluate the element-side normal advection flux from DG coefficients."""
    cp = require_cupy()
    return cp.einsum("dKi,Kfd,fiq->Kfq", beta_coeffs, cspace.mesh.normals, trace_ref.bas_of_bd_quads, optimize=True)


def reaction_mass_cupy(reaction, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Assemble element reaction mass matrices on the device."""
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    if isinstance(reaction, (int, float, np.integer, np.floating)):
        result = float(reaction) * mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
    elif isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(cspace.host)
        constant_value = reaction.constant_value
        if constant_value is not None:
            if constant_value == 0.0:
                result = cp.zeros((mesh.num_tri, cspace.el_dof, cspace.el_dof), dtype=REAL_DTYPE)
            else:
                result = constant_value * mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
        else:
            coeffs = as_cupy_coefficients(reaction, cspace)
            flat = coeffs @ q.weighted_triple_phi_flat
            result = mesh.aff_jacs[:, None, None] * flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof)
    else:
        points = mapped_quads_cupy(cspace)
        values = cp.asarray(reaction(points[:, 0, :], points[:, 1, :]), dtype=REAL_DTYPE)
        scaled = values * mesh.aff_jacs[:, None]
        flat = scaled @ q.weighted_phi_phi_flat
        result = flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof)
    if timings is not None:
        timings["local.reaction_mass"] = timings.get("local.reaction_mass", 0.0) + sync_elapsed(start)
    return result


def boundary_mass_cupy(beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None, *, upwind_scale: float = 1.0):
    """Assemble element upwind boundary mass matrices on the device."""
    cp = require_cupy()
    start = time.perf_counter()
    result = cp.einsum(
        "Kf,Kfq,fiq,fjq->Kij",
        cspace.mesh.jacs_el_fc,
        upwind_scale * cp.abs(beta_dot_normal),
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas_of_bd_quads,
        optimize=True,
    )
    if timings is not None:
        timings["local.boundary_mass"] = timings.get("local.boundary_mass", 0.0) + sync_elapsed(start)
    return result


def advection_mats_cupy(beta_coeffs, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Assemble element advection matrices on the device."""
    cp = require_cupy()
    start = time.perf_counter()
    q = cspace.quad_data
    beta_values = cp.einsum("dKi,iq->Kqd", beta_coeffs, q.bas_of_quads, optimize=True)
    scaled_inv_t = cspace.mesh.aff_jacs[:, None, None] * cspace.mesh.inv_aff_mats_t
    result = cp.einsum(
        "Kqd,KdD,jq,Diq,q->Kij",
        beta_values,
        scaled_inv_t,
        q.bas_of_quads,
        q.dbas_of_quads,
        q.Krf_w,
        optimize=True,
    )
    if timings is not None:
        timings["local.advection"] = timings.get("local.advection", 0.0) + sync_elapsed(start)
    return result


def element_boundary_mats_cupy(beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None, *, upwind_scale: float = 1.0):
    """Assemble element-to-trace boundary coupling matrices on the device."""
    cp = require_cupy()
    start = time.perf_counter()
    flux_weight = upwind_scale * cp.abs(beta_dot_normal) - beta_dot_normal
    result = cp.einsum(
        "Kf,Kfq,fiq,jq->Kifj",
        cspace.mesh.jacs_el_fc,
        flux_weight,
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas1d_of_ref_edg_qds,
        optimize=True,
    ).reshape(cspace.mesh.num_tri, cspace.el_dof, 3 * cspace.edg_dof)
    if timings is not None:
        timings["local.element_boundary"] = timings.get("local.element_boundary", 0.0) + sync_elapsed(start)
    return cp.ascontiguousarray(result)


def local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None, *, upwind_scale: float = 1.0):
    """Assemble complete element-local advection-reaction matrices on the device."""
    return (
        reaction_mass_cupy(reaction, cspace, timings)
        + boundary_mass_cupy(beta_dot_normal, cspace, trace_ref, timings, upwind_scale=upwind_scale)
        - advection_mats_cupy(beta_coeffs, cspace, timings)
    )


def solve_local_mats(local_mats, rhs, timings: dict[str, float] | None = None, key: str = "local.solve"):
    """Solve a batch of dense element-local systems on the device."""
    cp = require_cupy()
    start = time.perf_counter()
    result = cp.linalg.solve(local_mats, rhs)
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + sync_elapsed(start)
    return cp.ascontiguousarray(result)


def setup_reduced_indices(cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Build COO row and column indices for the reduced trace system."""
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = valid_elements.size * edg_dof * edg_dof
    rows = cp.empty(n_flux + n_mass, dtype=cp.int64)
    cols = cp.empty_like(rows)
    i_grid, j_grid = cp.meshgrid(cp.arange(edg_dof, dtype=cp.int64), cp.arange(edg_dof, dtype=cp.int64), indexing="ij")
    row_edges = mesh.loc2glob_edge[valid_elements, valid_faces]
    col_edges = mesh.loc2glob_edge[valid_elements]
    rows[:n_flux] = cp.broadcast_to(
        row_edges[:, None, None, None] * edg_dof + i_grid[None, None, :, :],
        (valid_elements.size, 3, edg_dof, edg_dof),
    ).ravel()
    cols[:n_flux] = (col_edges[:, :, None, None] * edg_dof + j_grid[None, None, :, :]).ravel()
    local = cp.arange(edg_dof, dtype=cp.int64)
    l0 = cp.broadcast_to(local[:, None], (edg_dof, edg_dof)).ravel()
    l1 = cp.broadcast_to(local[None, :], (edg_dof, edg_dof)).ravel()
    mass_edges = mesh.loc2glob_edge[valid_elements, valid_faces]
    rows[n_flux:] = (mass_edges[:, None] * edg_dof + l0[None, :]).ravel()
    cols[n_flux:] = (mass_edges[:, None] * edg_dof + l1[None, :]).ravel()
    if timings is not None:
        timings["indices"] = timings.get("indices", 0.0) + sync_elapsed(start)
    return rows, cols


def trace_lift_cupy(beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None, *, upwind_scale: float = 1.0):
    """Assemble oriented local trace-lift matrices on the device."""
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    oriented_trace = oriented_trace_basis_cupy(cspace, trace_ref)
    result = cp.ascontiguousarray(
        cp.einsum(
            "Kf,Kfq,Kfaq,fiq,q->Kfai",
            mesh.jacs_el_fc,
            upwind_scale * cp.abs(beta_dot_normal),
            oriented_trace,
            trace_ref.bas_of_bd_quads,
            trace_ref.weights,
            optimize=True,
        )
    )
    if timings is not None:
        timings["trace_lift"] = timings.get("trace_lift", 0.0) + sync_elapsed(start)
    return result


def trace_blocks_cupy(solved_el_bd_mats, trace_lift, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None):
    """Form oriented element Schur-complement trace blocks on the device."""
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    blocks = trace_lift @ solved_el_bd_mats[:, None, :, :]
    blocks = blocks.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    if mesh.num_negative_orientations:
        neg = blocks[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :]
        if trace_ref.kind == "legendre-modal":
            signs = cp.where(cp.arange(edg_dof, dtype=cp.int64) % 2 == 0, REAL_DTYPE(1.0), REAL_DTYPE(-1.0))
            blocks[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :] = neg * signs
        else:
            blocks[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :] = neg[..., ::-1]
    result = cp.ascontiguousarray(blocks.swapaxes(2, 3))
    if timings is not None:
        timings["trace_blocks"] = timings.get("trace_blocks", 0.0) + sync_elapsed(start)
    return result


def oriented_trace_basis_cupy(cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace):
    """Return device trace basis values in global edge orientation."""
    cp = require_cupy()
    return cp.ascontiguousarray(trace_ref.oriented_basis_table[(~cspace.mesh.orientations).astype(cp.int32)])


def interior_trace_mass_blocks_cupy(beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None, *, upwind_scale: float = 1.0, gauge_inactive: bool = False):
    """Assemble upwind trace mass blocks for interior sides on the device."""
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    gamma_face = upwind_scale * cp.abs(beta_dot_normal) - beta_dot_normal
    oriented_trace = oriented_trace_basis_cupy(cspace, trace_ref)
    side_blocks = cp.einsum(
        "Kf,Kfq,Kfaq,Kfbq,q->Kfab",
        mesh.jacs_el_fc,
        gamma_face,
        oriented_trace,
        oriented_trace,
        trace_ref.weights,
        optimize=True,
    )
    result = cp.ascontiguousarray(side_blocks[mesh.interior_elements, mesh.interior_faces])
    if gauge_inactive:
        from hdgfem.hdg.stabilization import gauge_inactive_advection_trace_blocks
        gauge_inactive_advection_trace_blocks(result, cp.abs(beta_dot_normal), mesh, xp=cp)
    if timings is not None:
        timings["interior_mass"] = timings.get("interior_mass", 0.0) + sync_elapsed(start)
    return result


def trace_data_cupy(trace_blocks, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, beta_dot_normal, timings: dict[str, float] | None = None, *, upwind_scale: float = 1.0, gauge_inactive: bool = False):
    """Pack reduced trace matrix values in device COO ordering."""
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = valid_elements.size * edg_dof * edg_dof
    data = cp.empty(n_flux + n_mass, dtype=REAL_DTYPE)
    data[:n_flux] = -trace_blocks[valid_elements, valid_faces].ravel()
    data[n_flux:] = interior_trace_mass_blocks_cupy(beta_dot_normal, cspace, trace_ref, timings, upwind_scale=upwind_scale, gauge_inactive=gauge_inactive).ravel()
    if timings is not None:
        timings["data"] = timings.get("data", 0.0) + sync_elapsed(start)
    return data


def face_rhs_cupy(solved_src, trace_lift, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Form per-face trace RHS contributions on the device."""
    cp = require_cupy()
    start = time.perf_counter()
    result = (trace_lift @ solved_src[:, None, :, :]).squeeze(-1)
    if timings is not None:
        timings["rhs_faces"] = timings.get("rhs_faces", 0.0) + sync_elapsed(start)
    return result


def boundary_trace_values_cupy(boundary_condition: Callable, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace):
    """Evaluate prescribed boundary trace coefficients on the device."""
    from hdgfem.hdg.condensation import boundary_trace_coefficients
    with require_cupy().cuda.Device(cspace.device_id):
        return boundary_trace_coefficients(boundary_condition, cspace.host, trace_space=trace_ref.host,
                                           backend="device", boundary_only=True)


def build_dof_maps(cspace: CupyDGSpace):
    """Build device maps for boundary and reduced trace degrees of freedom."""
    cp = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    edge_to_boundary_slot = cp.full(mesh.num_edg, -1, dtype=cp.int64)
    edge_to_boundary_slot[mesh.bnd_edges_inds] = cp.arange(mesh.bnd_edges_inds.size, dtype=cp.int64)
    full_to_reduced = cp.full(mesh.num_edg * edg_dof, -1, dtype=cp.int64)
    local = cp.arange(edg_dof, dtype=cp.int64)
    full_to_reduced[(mesh.int_edges_inds[:, None] * edg_dof + local[None, :]).ravel()] = (
        cp.arange(mesh.int_edges_inds.size, dtype=cp.int64)[:, None] * edg_dof + local[None, :]
    ).ravel()
    return edge_to_boundary_slot, full_to_reduced


def eliminate_boundary_cupy(rows, cols, data, rhs, boundary_trace, maps, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Eliminate prescribed boundary columns from a device COO trace system."""
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    edge_to_boundary_slot, full_to_reduced = maps
    row_r = rows.reshape((rows.size // edg_dof, edg_dof))
    col_r = cols.reshape((cols.size // edg_dof, edg_dof))
    data_r = data.reshape((data.size // edg_dof, edg_dof))
    col_edges = col_r[:, 0] // edg_dof
    boundary_slots = edge_to_boundary_slot[col_edges]
    keep = cp.where(boundary_slots < 0)[0]
    remove = cp.where(boundary_slots >= 0)[0]
    keep_count = int(keep.size)
    reduced_rows = cp.empty(keep_count * edg_dof, dtype=cp.int64)
    reduced_cols = cp.empty_like(reduced_rows)
    reduced_data = cp.empty(keep_count * edg_dof, dtype=REAL_DTYPE)
    reduced_rows.reshape((keep_count, edg_dof))[:] = full_to_reduced[row_r[keep]]
    reduced_cols.reshape((keep_count, edg_dof))[:] = full_to_reduced[col_r[keep]]
    reduced_data.reshape((keep_count, edg_dof))[:] = data_r[keep]
    if remove.size:
        row_ids = row_r[remove, 0]
        slots = boundary_slots[remove]
        cp.add.at(rhs, row_ids, cp.sum(-data_r[remove] * boundary_trace[slots], axis=1))
    reduced_rhs = rhs.reshape((mesh.num_edg, edg_dof))[mesh.int_edges_inds].ravel()
    if timings is not None:
        timings["boundary_elimination"] = timings.get("boundary_elimination", 0.0) + sync_elapsed(start)
    return reduced_rows, reduced_cols, reduced_data, reduced_rhs


def assemble_reduced_system_cuda(
    source,
    reaction,
    boundary_condition: Callable | None,
    beta_coeffs,
    cspace: DGSpace | CupyDGSpace,
    trace_space: DGTraceSpace | CupyDGTraceSpace,
    *,
    backend: CudaAdvectionAssemblyBackend = "raw-cuda",
    beta_dot_normal=None,
    raw_block_size: RawCudaBlockSize = "auto",
    raw_local_assembly: RawLocalAssembly = "precomputed",
    raw_lu_mode: RawLuMode = "safe",
    raw_matrix_format: str = "coo",
    zero_boundary_flux: bool = False,
    raw_response_workspace=None,
    raw_tsle_workspace: RawAdvectionTsleWorkspace | None = None,
    raw_cache_local_response: bool = True,
    raw_factor_workspace=None,
    advection_stabilization=None,
) -> CudaAdvectionAssembly:
    """Assemble the boundary-eliminated advection trace system on device."""
    if zero_boundary_flux and boundary_condition is not None:
        raise ValueError("boundary_condition must be None when boundary_mode='zero-flux'")
    from hdgfem.hdg.stabilization import (
            upwind_factor,
            effective_advection_normal_flux,
            is_conflict_averaged_upwind,
        )
    factor = upwind_factor(advection_stabilization)
    if factor is None:
        raise NotImplementedError("This CUDA wrapper requires an upwind stabilization policy")
    cp = require_cupy()
    cspace = as_cupy_space(cspace)
    trace_ref = trace_space if isinstance(trace_space, CupyDGTraceSpace) else as_cupy_trace_space(trace_space, device=cspace.device_id)
    timings: dict[str, float] = {}
    start_total = time.perf_counter()
    if backend == "raw-cuda":
        source = _require_raw_dg_field(source, cspace, "source")
        reaction = _require_raw_dg_field(reaction, cspace, "reaction")
        if raw_local_assembly not in {"precomputed", "fused", "split3"}:
            raise ValueError("raw_local_assembly must be 'precomputed', 'fused', or 'split3'")
        if raw_lu_mode not in {"safe", "coop"}:
            raise ValueError("raw_lu_mode must be 'safe' or 'coop'")
        zero_boundary_flux = bool(zero_boundary_flux)
        eliminated_local_assembly = raw_local_assembly in {"fused", "split3"}
        if zero_boundary_flux and not eliminated_local_assembly:
            raise NotImplementedError(
                "raw-CUDA zero-flux assembly requires raw_local_assembly='fused' or 'split3'"
            )
        if raw_lu_mode != "safe" and not eliminated_local_assembly:
            raise ValueError(
                "raw_lu_mode='coop' is supported only with raw_local_assembly='fused' or 'split3'"
            )
        if raw_local_assembly == "split3" and raw_lu_mode != "coop":
            raise ValueError("raw_local_assembly='split3' requires raw_lu_mode='coop'")
        if raw_local_assembly == "split3":
            normalized_matrix_format = str(raw_matrix_format).lower()
            if normalized_matrix_format == "auto":
                raw_matrix_format = "bsr"
            elif normalized_matrix_format != "bsr":
                raise ValueError(
                    "raw_local_assembly='split3' requires raw_matrix_format='bsr'"
                )
        if eliminated_local_assembly:
            source_coeffs = source_coefficients_cupy(source, cspace, timings)
            reaction_coeffs, reaction_scalar, reaction_is_scalar = reaction_coefficients_cupy(reaction, cspace, timings)
            advection_tensor = reference_advection_tensor_cupy(cspace, timings)
            (
                advection_sparse_offsets,
                advection_sparse_modes,
                advection_sparse_values0,
                advection_sparse_values1,
                use_sparse_advection,
                mass_is_diagonal,
            ) = reference_advection_sparse_cupy(cspace, timings)
            if zero_boundary_flux:
                boundary_trace = cp.zeros((cspace.mesh.bnd_edges_inds.size, cspace.edg_dof), dtype=REAL_DTYPE)
            else:
                if boundary_condition is None:
                    raise ValueError("boundary_condition is required unless zero_boundary_flux=True")
                boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
            common_raw_options = dict(
                source_coeffs=source_coeffs,
                beta_coeffs=beta_coeffs,
                reaction_coeffs=reaction_coeffs,
                reaction_scalar=reaction_scalar,
                reaction_is_scalar=reaction_is_scalar,
                boundary_trace=boundary_trace,
                cspace=cspace,
                trace_ref=trace_ref,
                advection_tensor=advection_tensor,
                advection_sparse_offsets=advection_sparse_offsets,
                advection_sparse_modes=advection_sparse_modes,
                advection_sparse_values0=advection_sparse_values0,
                advection_sparse_values1=advection_sparse_values1,
                use_sparse_advection=use_sparse_advection,
                mass_is_diagonal=mass_is_diagonal,
                block_size=raw_block_size,
                matrix_format=raw_matrix_format,
                zero_boundary_flux=zero_boundary_flux,
                cache_local_response=raw_cache_local_response,
                advection_stabilization=advection_stabilization,
            )
            if raw_local_assembly == "split3":
                raw = assemble_projected_advection_trace_system_eliminated_tsle_bsr(
                    **common_raw_options,
                    workspace=raw_tsle_workspace,
                )
            else:
                raw = assemble_projected_advection_trace_system_eliminated_raw_cuda_fused(
                    **common_raw_options,
                    lu_mode=raw_lu_mode,
                    local_response=raw_response_workspace,
                    factor_workspace=raw_factor_workspace,
                )
        else:
            if zero_boundary_flux:
                raise NotImplementedError("raw-CUDA zero-flux assembly currently requires raw_local_assembly='fused'")
            if beta_dot_normal is None:
                beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
            beta_dot_normal = effective_advection_normal_flux(beta_dot_normal, cspace.mesh, advection_stabilization, xp=cp)
            local_mats = local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref, timings, upwind_scale=factor)
            element_boundary = element_boundary_mats_cupy(beta_dot_normal, cspace, trace_ref, timings, upwind_scale=factor)
            source_rhs = source_moments_cupy(source, cspace, timings)
            boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
            side_mass_blocks = interior_trace_mass_blocks_cupy(beta_dot_normal, cspace, trace_ref, timings, upwind_scale=factor,
                gauge_inactive=is_conflict_averaged_upwind(advection_stabilization))
            trace_lift = trace_lift_cupy(beta_dot_normal, cspace, trace_ref, timings, upwind_scale=factor)
            raw = assemble_projected_advection_trace_system_eliminated_raw_cuda(
                local_mats=local_mats,
                element_boundary=element_boundary,
                source_rhs=source_rhs,
                boundary_trace=boundary_trace,
                side_mass_blocks=side_mass_blocks,
                trace_lift=trace_lift,
                cspace=cspace,
                trace_ref=trace_ref,
                block_size=raw_block_size,
            )
        from dataclasses import replace
        raw = replace(raw, advection_stabilization=advection_stabilization)
        for key, value in raw.timings.items():
            timings[key if str(key).startswith("raw.") else f"raw.{key}"] = value
        timings["total"] = sync_elapsed(start_total)
        return CudaAdvectionAssembly(
            rows=raw.rows,
            cols=raw.cols,
            data=raw.data,
            rhs=raw.rhs,
            local_mats=raw.local_mats,
            element_boundary=raw.element_boundary,
            source_rhs=raw.source_rhs,
            boundary_trace=raw.boundary_trace,
            beta_dot_normal=beta_dot_normal,
            cspace=cspace,
            trace_ref=trace_ref,
            raw=raw,
            indptr=raw.indptr,
            indices=raw.indices,
            matrix_format=raw.matrix_format,
            timings=timings,
        )

    if zero_boundary_flux:
        raise NotImplementedError("CuPy zero-flux advection assembly is not implemented yet; use backend='raw-cuda' with raw_local_assembly='fused' or backend='numba'")
    if beta_dot_normal is None:
        beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
    beta_dot_normal = effective_advection_normal_flux(beta_dot_normal, cspace.mesh, advection_stabilization, xp=cp)
    maps = build_dof_maps(cspace)
    rows, cols = setup_reduced_indices(cspace, timings)
    local_mats = local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref, timings, upwind_scale=factor)
    element_boundary = element_boundary_mats_cupy(beta_dot_normal, cspace, trace_ref, timings, upwind_scale=factor)
    source_rhs = source_moments_cupy(source, cspace, timings)
    local_rhs = cp.concatenate((element_boundary, source_rhs[..., None]), axis=2)
    solved = solve_local_mats(local_mats, local_rhs, timings, "local.solve.assembly")
    solved_el_bd = solved[:, :, : 3 * cspace.edg_dof]
    solved_src = solved[:, :, 3 * cspace.edg_dof :]
    trace_lift = trace_lift_cupy(beta_dot_normal, cspace, trace_ref, timings, upwind_scale=factor)
    blocks = trace_blocks_cupy(solved_el_bd, trace_lift, cspace, trace_ref, timings)
    data = trace_data_cupy(blocks, cspace, trace_ref, beta_dot_normal, timings, upwind_scale=factor,
                           gauge_inactive=is_conflict_averaged_upwind(advection_stabilization))
    rhs_full = cp.zeros(cspace.mesh.num_edg * cspace.edg_dof, dtype=REAL_DTYPE)
    rhs_full_r = rhs_full.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    faces = face_rhs_cupy(solved_src, trace_lift, cspace, timings)
    cp.add.at(
        rhs_full_r,
        cspace.mesh.loc2glob_edge[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
        faces[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
    )
    boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
    rows, cols, data, rhs = eliminate_boundary_cupy(rows, cols, data, rhs_full, boundary_trace, maps, cspace, timings)
    timings["total"] = sync_elapsed(start_total)
    return CudaAdvectionAssembly(
        rows=rows,
        cols=cols,
        data=data,
        rhs=rhs,
        local_mats=local_mats,
        element_boundary=element_boundary,
        source_rhs=source_rhs,
        boundary_trace=boundary_trace,
        beta_dot_normal=beta_dot_normal,
        cspace=cspace,
        trace_ref=trace_ref,
        raw=None,
        timings=timings,
    )


def update_reduced_system_rhs_cuda(assembly, source, boundary_condition, factors):
    """Condense a new source/boundary RHS with an unchanged raw transport operator.

    Reuses source moments, trace lifting, orientation, scatter, and response
    reconstruction formalism from the regular assembly path.
    """
    from hdgfem.transport.raw_cuda import solve_cached_advection_source_raw
    cp = require_cupy()
    start = time.perf_counter()
    cspace, trace_ref, raw = assembly.cspace, assembly.trace_ref, assembly.raw
    if raw is None or raw.local_response is None or factors.signature is None:
        raise RuntimeError("source-only transport update requires cached raw local factors")
    timings = {}
    moments = source_moments_cupy(source, cspace, timings)
    solved = solve_cached_advection_source_raw(moments, raw.local_response, factors, cspace, trace_ref)
    if raw.zero_boundary_flux:
        if boundary_condition is not None:
            raise ValueError("boundary_condition must be None for zero-flux transport")
        boundary = assembly.boundary_trace
    else:
        boundary = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
        boundary_full = reconstruct_trace_cupy(cp.zeros_like(assembly.rhs), boundary, cspace)
        # This applies the retained trace response to prescribed boundary data
        # and includes the newly solved source column, with correct orientation.
        solved, _ = reconstruct_advection_field_cuda(boundary_full, source, None, None, assembly)
    faces = face_rhs_cupy(solved[..., None], factors.trace_lift, cspace, timings)
    rhs_full = cp.zeros((cspace.mesh.num_edg, cspace.edg_dof), dtype=REAL_DTYPE)
    elements, sides = cspace.mesh.interior_elements, cspace.mesh.interior_faces
    cp.add.at(rhs_full, cspace.mesh.loc2glob_edge[elements, sides], faces[elements, sides])
    rhs = cp.ascontiguousarray(rhs_full[cspace.mesh.int_edges_inds].ravel())
    raw = replace(raw, rhs=rhs, boundary_trace=boundary,
                  source_coeffs=source_coefficients_cupy(source, cspace), timings={})
    timings["operator.reused"] = 1.0
    timings["local.factors.reused"] = 1.0
    timings["total"] = sync_elapsed(start)
    return replace(assembly, rhs=rhs, boundary_trace=boundary, raw=raw, timings=timings)


def reconstruct_advection_field_cuda(trace, source, reaction, beta_coeffs, assembly: CudaAdvectionAssembly):
    """Recover device element coefficients from the solved trace field."""
    cp = require_cupy()
    cspace = assembly.cspace
    trace_ref = assembly.trace_ref
    raw = assembly.raw
    if raw is not None:
        if raw.local_response is not None:
            block_size = int(raw.timings.get("raw.block_size", 32.0))
            uh, kernel_elapsed = reconstruct_projected_advection_field_from_response_raw_cuda(
                trace=trace,
                local_response=raw.local_response,
                cspace=cspace,
                trace_ref=trace_ref,
                block_size=block_size,
            )
        elif raw.local_mats is None:
            block_size = int(raw.timings.get("raw.block_size", 32.0))
            uh, kernel_elapsed = reconstruct_projected_advection_field_raw_cuda_fused(
                trace=trace,
                source_coeffs=raw.source_coeffs,
                beta_coeffs=raw.beta_coeffs,
                reaction_coeffs=raw.reaction_coeffs,
                reaction_scalar=raw.reaction_scalar,
                reaction_is_scalar=raw.reaction_is_scalar,
                cspace=cspace,
                trace_ref=trace_ref,
                advection_tensor=raw.advection_tensor,
                advection_sparse_offsets=raw.advection_sparse_offsets,
                advection_sparse_modes=raw.advection_sparse_modes,
                advection_sparse_values0=raw.advection_sparse_values0,
                advection_sparse_values1=raw.advection_sparse_values1,
                use_sparse_advection=raw.use_sparse_advection,
                mass_is_diagonal=raw.mass_is_diagonal,
                block_size=block_size,
                lu_mode=raw.lu_mode,
                zero_boundary_flux=raw.zero_boundary_flux,
                advection_stabilization=raw.advection_stabilization,
            )
        else:
            uh, kernel_elapsed = reconstruct_projected_advection_field_raw_cuda(
                trace=trace,
                local_mats=raw.local_mats,
                element_boundary=raw.element_boundary,
                source_rhs=raw.source_rhs,
                cspace=cspace,
                trace_ref=trace_ref,
            )
        return require_finite_device_values(uh, "raw-CUDA advection reconstruction"), kernel_elapsed

    beta_dot_normal = assembly.beta_dot_normal
    if beta_dot_normal is None:
        beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
    trace_r = trace.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    element_traces = trace_r[cspace.mesh.loc2glob_edge, :]
    if cspace.mesh.num_negative_orientations:
        neg = element_traces[cspace.mesh.negative_orientation_elements, cspace.mesh.negative_orientation_faces]
        if trace_ref.kind == "legendre-modal":
            signs = cp.where(cp.arange(cspace.edg_dof, dtype=cp.int64) % 2 == 0, REAL_DTYPE(1.0), REAL_DTYPE(-1.0))
            element_traces[cspace.mesh.negative_orientation_elements, cspace.mesh.negative_orientation_faces] = neg * signs
        else:
            element_traces[cspace.mesh.negative_orientation_elements, cspace.mesh.negative_orientation_faces] = neg[:, ::-1]
    element_traces = element_traces.reshape((cspace.mesh.num_tri, 3 * cspace.edg_dof))
    local_mats = assembly.local_mats
    element_boundary = assembly.element_boundary
    source_rhs = source_moments_cupy(source, cspace)
    rhs = source_rhs[..., None] + element_boundary @ element_traces[..., None]
    start = time.perf_counter()
    uh = cp.linalg.solve(local_mats, rhs).squeeze(-1)
    elapsed = sync_elapsed(start)
    return require_finite_device_values(cp.ascontiguousarray(uh), "CuPy advection reconstruction"), elapsed


TIMINGS: dict[str, float] = {}


def build_trace_reference(cspace: DGSpace | CupyDGSpace, kind: str) -> CupyDGTraceSpace:
    """Compatibility helper returning a CUDA trace-space mirror by basis kind."""
    cspace = as_cupy_space(cspace)
    return as_cupy_trace_space(cspace.host.trace_space(kind), device=cspace.device_id)


def assemble_reduced_system(
    source,
    reaction,
    exact,
    beta_coeffs,
    beta_dot_normal,
    maps,
    cspace,
    trace_ref,
    backend: str = "cupy",
    raw_block_size: RawCudaBlockSize = "auto",
    raw_local_assembly: str = "precomputed",
    raw_lu_mode: str = "safe",
    raw_matrix_format: str = "coo",
    zero_boundary_flux: bool = False,
):
    """Compatibility wrapper matching the former GPU runner helper signature."""
    assembly = assemble_reduced_system_cuda(
        source,
        reaction,
        exact,
        beta_coeffs,
        cspace,
        trace_ref,
        backend=backend,
        beta_dot_normal=beta_dot_normal,
        raw_block_size=raw_block_size,
        raw_local_assembly=raw_local_assembly,
        raw_lu_mode=raw_lu_mode,
        raw_matrix_format=raw_matrix_format,
        zero_boundary_flux=zero_boundary_flux,
    )
    TIMINGS.update(assembly.timings)
    return (
        assembly.rows,
        assembly.cols,
        assembly.data,
        assembly.rhs,
        assembly.local_mats,
        assembly.element_boundary,
        assembly.source_rhs,
        assembly.boundary_trace,
        assembly.raw,
    )


__all__ = [
    "TIMINGS",
    "assemble_reduced_system",
    "build_trace_reference",
    "assemble_reduced_system_cuda",
    "beta_dot_normal_from_coeffs",
    "build_dof_maps",
    "project_callable_cupy",
    "reconstruct_advection_field_cuda",
]
