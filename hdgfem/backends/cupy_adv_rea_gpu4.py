"""GPU4-style CuPy/Raw CUDA advection-reaction HDG assembly helpers.

This module owns the device execution formalism for the fast advection-reaction
path. Public solvers still accept host :class:`DGSpace` and field objects; this
backend mirrors immutable space/trace data to CUDA and returns device arrays to
callers that can keep the global solve on device, or host arrays when requested.
"""

from __future__ import annotations

import copy
import ctypes
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..core.space import DGField, DGSpace, DGTraceSpace
from ..linalg.system import KnownDofReduction, SolveResult
from .cupy import CupyDGSpace, as_cupy_coefficients, as_cupy_space, require_cupy, require_cupyx_sparse, require_pyamgx
from .cupy_adv_rea_raw import (
    RawAdvectionAssemblyResult,
    assemble_projected_advection_trace_system_eliminated_raw_cuda,
    assemble_projected_advection_trace_system_eliminated_raw_cuda_fused,
    reconstruct_projected_advection_field_raw_cuda,
    reconstruct_projected_advection_field_raw_cuda_fused,
)


def _flush_c_stdio() -> None:
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass

RawLocalAssembly = Literal["precomputed", "fused"]
RawLuMode = Literal["safe", "coop"]
Gpu4AssemblyBackend = Literal["cupy", "raw-cuda"]


@dataclass(frozen=True)
class CupyDGTraceSpace:
    """CUDA mirror of host :class:`DGTraceSpace` data."""

    host: DGTraceSpace
    device_id: int
    kind: str
    nodal: bool
    interpolation_nodes: Any
    quads: Any
    weights: Any
    bas_of_bd_quads: Any
    bas1d_of_ref_edg_qds: Any
    weighted_bas_of_bd_quads: Any
    weighted_bas1d_of_ref_edg_qds: Any
    face_trace_test_element_trial_oriented: Any
    M_rf_fc: Any

    @classmethod
    def from_host(cls, trace_space: DGTraceSpace, *, device_id: int) -> "CupyDGTraceSpace":
        cp = require_cupy()
        return cls(
            host=trace_space,
            device_id=int(device_id),
            kind=trace_space.kind,
            nodal=bool(trace_space.nodal),
            interpolation_nodes=cp.asarray(trace_space.interpolation_nodes, dtype=cp.float64),
            quads=cp.asarray(trace_space.quads, dtype=cp.float64),
            weights=cp.asarray(trace_space.weights, dtype=cp.float64),
            bas_of_bd_quads=cp.asarray(trace_space.bas_of_bd_quads, dtype=cp.float64),
            bas1d_of_ref_edg_qds=cp.asarray(trace_space.bas1d_of_ref_edg_qds, dtype=cp.float64),
            weighted_bas_of_bd_quads=cp.asarray(trace_space.weighted_bas_of_bd_quads, dtype=cp.float64),
            weighted_bas1d_of_ref_edg_qds=cp.asarray(trace_space.weighted_bas1d_of_ref_edg_qds, dtype=cp.float64),
            face_trace_test_element_trial_oriented=cp.asarray(
                trace_space.face_trace_test_element_trial_oriented,
                dtype=cp.float64,
            ),
            M_rf_fc=cp.asarray(trace_space.M_rf_fc, dtype=cp.float64),
        )

    @property
    def edg_dof(self) -> int:
        return self.host.edg_dof


def as_cupy_trace_space(trace_space: DGTraceSpace, *, device: int | None = None) -> CupyDGTraceSpace:
    """Return a cached CUDA mirror of ``trace_space``."""
    cp = require_cupy()
    device_id = int(cp.cuda.runtime.getDevice()) if device is None else int(device)
    cache = getattr(trace_space, "_hdgfem_cupy_trace_space_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(trace_space, "_hdgfem_cupy_trace_space_cache", cache)
    if device_id not in cache:
        with cp.cuda.Device(device_id):
            cache[device_id] = CupyDGTraceSpace.from_host(trace_space, device_id=device_id)
    return cache[device_id]


@dataclass(frozen=True)
class Gpu4AdvectionAssembly:
    """Reduced trace system assembled by the GPU4 path.

    Rows, columns, data, RHS, local tensors, and boundary trace are CUDA arrays.
    Use :meth:`to_host_reduction` when passing the system to host-only solver
    APIs.
    """

    rows: Any | None
    cols: Any | None
    data: Any
    rhs: Any
    local_mats: Any | None
    element_boundary: Any | None
    source_rhs: Any | None
    boundary_trace: Any
    beta_dot_normal: Any | None
    cspace: CupyDGSpace
    trace_ref: CupyDGTraceSpace
    raw: RawAdvectionAssemblyResult | None = None
    indptr: Any | None = None
    indices: Any | None = None
    matrix_format: str = "coo"
    timings: dict[str, float] = field(default_factory=dict)

    def to_host_reduction(self) -> KnownDofReduction:
        """Transfer the reduced system metadata to a host KnownDofReduction."""
        if self.matrix_format != "coo":
            raise RuntimeError("host KnownDofReduction materialization is currently supported only for COO raw systems")
        cp = require_cupy()
        space = self.cspace.host
        edg_dof = self.cspace.edg_dof
        full_size = space.mesh.num_edg * edg_dof
        free_mask = np.zeros(full_size, dtype=bool)
        local = np.arange(edg_dof, dtype=np.int64)
        free_mask[(space.mesh.int_edges_inds[:, None] * edg_dof + local[None, :]).ravel()] = True
        known_mask = ~free_mask
        known_values = np.zeros(full_size, dtype=np.float64)
        boundary_host = np.ascontiguousarray(cp.asnumpy(self.boundary_trace), dtype=np.float64)
        known_values[(space.mesh.bnd_edges_inds[:, None] * edg_dof + local[None, :]).ravel()] = boundary_host.ravel()
        old_to_new = np.full(full_size, -1, dtype=np.int64)
        old_to_new[free_mask] = np.arange(np.count_nonzero(free_mask), dtype=np.int64)
        return KnownDofReduction(
            rows=np.ascontiguousarray(cp.asnumpy(self.rows), dtype=np.int64),
            cols=np.ascontiguousarray(cp.asnumpy(self.cols), dtype=np.int64),
            data=np.ascontiguousarray(cp.asnumpy(self.data), dtype=np.float64),
            rhs=np.ascontiguousarray(cp.asnumpy(self.rhs), dtype=np.float64),
            free_mask=np.ascontiguousarray(free_mask),
            known_mask=np.ascontiguousarray(known_mask),
            known_values=np.ascontiguousarray(known_values),
            old_to_new=np.ascontiguousarray(old_to_new),
        )


def sync_elapsed(start: float) -> float:
    cp = require_cupy()
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter() - start


def mapped_quads_cupy(cspace: CupyDGSpace):
    return cspace.mapped_quads


def project_callable_cupy(func: Callable, cspace: CupyDGSpace, timings: dict[str, float] | str | None = None, key: str | None = None):
    if isinstance(timings, str):
        timings = TIMINGS
    cp = require_cupy()
    start = time.perf_counter()
    q = cspace.quad_data
    points = mapped_quads_cupy(cspace)
    values = cp.asarray(func(points[:, 0, :], points[:, 1, :]), dtype=cp.float64)
    coeffs = values @ q.projection_operator.T
    if timings is not None and key is not None:
        timings[key] = timings.get(key, 0.0) + sync_elapsed(start)
    return cp.ascontiguousarray(coeffs)


def source_coefficients_cupy(source, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    if isinstance(source, DGField):
        source.space.assert_same_mesh(cspace.host)
        return as_cupy_coefficients(source, cspace)
    return project_callable_cupy(source, cspace, timings, "projection.source")


def reaction_coefficients_cupy(reaction, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    if isinstance(reaction, (int, float, np.integer, np.floating)):
        return cp.empty(1, dtype=cp.float64), float(reaction), True
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(cspace.host)
        return as_cupy_coefficients(reaction, cspace), 0.0, False
    coeffs = project_callable_cupy(reaction, cspace, timings, "projection.reaction")
    return coeffs, 0.0, False


def _require_raw_dg_field(value, cspace: CupyDGSpace, label: str) -> DGField:
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


def reference_advection_tensor_cupy(cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    start = time.perf_counter()
    q = cspace.host.quad_data
    tensor = np.einsum("q,qk,qj,qiD->Dkij", q.Krf_w, q.phi, q.phi, q.gphi, optimize=True)
    result = cp.asarray(np.ascontiguousarray(tensor, dtype=np.float64))
    if timings is not None:
        timings["reference_advection_tensor"] = timings.get("reference_advection_tensor", 0.0) + sync_elapsed(start)
    return result


def source_moments_cupy(source, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    if isinstance(source, DGField):
        source.space.assert_same_mesh(cspace.host)
        constant_value = source.constant_value
        if constant_value is not None:
            ref_moments = cp.asarray(cspace.host._constant_reference_moments(constant_value))
            rhs = mesh.aff_jacs[:, None] * ref_moments[None, :]
        else:
            coeffs = as_cupy_coefficients(source, cspace)
            rhs = mesh.aff_jacs[:, None] * (coeffs @ q.MKrf)
    else:
        points = mapped_quads_cupy(cspace)
        values = cp.asarray(source(points[:, 0, :], points[:, 1, :]), dtype=cp.float64)
        rhs = mesh.aff_jacs[:, None] * cp.einsum("Kq,iq,q->Ki", values, q.bas_of_quads, q.Krf_w)
    if timings is not None:
        timings["source_moments"] = timings.get("source_moments", 0.0) + sync_elapsed(start)
    return cp.ascontiguousarray(rhs)


def beta_dot_normal_from_coeffs(beta_coeffs, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace):
    cp = require_cupy()
    return cp.einsum("dKi,Kfd,fiq->Kfq", beta_coeffs, cspace.mesh.normals, trace_ref.bas_of_bd_quads, optimize=True)


def reaction_mass_cupy(reaction, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
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
                result = cp.zeros((mesh.num_tri, cspace.el_dof, cspace.el_dof), dtype=cp.float64)
            else:
                result = constant_value * mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
        else:
            coeffs = as_cupy_coefficients(reaction, cspace)
            flat = coeffs @ q.weighted_triple_phi_flat
            result = mesh.aff_jacs[:, None, None] * flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof)
    else:
        points = mapped_quads_cupy(cspace)
        values = cp.asarray(reaction(points[:, 0, :], points[:, 1, :]), dtype=cp.float64)
        scaled = values * mesh.aff_jacs[:, None]
        flat = scaled @ q.weighted_phi_phi_flat
        result = flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof)
    if timings is not None:
        timings["local.reaction_mass"] = timings.get("local.reaction_mass", 0.0) + sync_elapsed(start)
    return result


def boundary_mass_cupy(beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    start = time.perf_counter()
    result = cp.einsum(
        "Kf,Kfq,fiq,fjq->Kij",
        cspace.mesh.jacs_el_fc,
        cp.abs(beta_dot_normal),
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas_of_bd_quads,
        optimize=True,
    )
    if timings is not None:
        timings["local.boundary_mass"] = timings.get("local.boundary_mass", 0.0) + sync_elapsed(start)
    return result


def advection_mats_cupy(beta_coeffs, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
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


def element_boundary_mats_cupy(beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    start = time.perf_counter()
    flux_weight = cp.abs(beta_dot_normal) - beta_dot_normal
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


def local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None):
    return (
        reaction_mass_cupy(reaction, cspace, timings)
        + boundary_mass_cupy(beta_dot_normal, cspace, trace_ref, timings)
        - advection_mats_cupy(beta_coeffs, cspace, timings)
    )


def solve_local_mats(local_mats, rhs, timings: dict[str, float] | None = None, key: str = "local.solve"):
    cp = require_cupy()
    start = time.perf_counter()
    result = cp.linalg.solve(local_mats, rhs)
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + sync_elapsed(start)
    return cp.ascontiguousarray(result)


def setup_reduced_indices(cspace: CupyDGSpace, timings: dict[str, float] | None = None):
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


def trace_lift_cupy(beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    oriented_trace = oriented_trace_basis_cupy(cspace, trace_ref)
    result = cp.ascontiguousarray(
        cp.einsum(
            "Kf,Kfq,Kfaq,fiq,q->Kfai",
            mesh.jacs_el_fc,
            cp.abs(beta_dot_normal),
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
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    blocks = trace_lift @ solved_el_bd_mats[:, None, :, :]
    blocks = blocks.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    if mesh.num_negative_orientations:
        neg = blocks[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :]
        if trace_ref.kind == "legendre-modal":
            signs = cp.where(cp.arange(edg_dof, dtype=cp.int64) % 2 == 0, 1.0, -1.0)
            blocks[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :] = neg * signs
        else:
            blocks[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :] = neg[..., ::-1]
    result = cp.ascontiguousarray(blocks.swapaxes(2, 3))
    if timings is not None:
        timings["trace_blocks"] = timings.get("trace_blocks", 0.0) + sync_elapsed(start)
    return result


def oriented_trace_basis_cupy(cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace):
    cp = require_cupy()
    mesh = cspace.mesh
    basis = cp.broadcast_to(
        trace_ref.bas1d_of_ref_edg_qds[None, None, :, :],
        (mesh.num_tri, 3, cspace.edg_dof, trace_ref.weights.size),
    ).copy()
    if mesh.num_negative_orientations:
        neg = basis[mesh.negative_orientation_elements, mesh.negative_orientation_faces]
        if trace_ref.kind == "legendre-modal":
            signs = cp.where(cp.arange(cspace.edg_dof, dtype=cp.int64) % 2 == 0, 1.0, -1.0)
            basis[mesh.negative_orientation_elements, mesh.negative_orientation_faces] = neg * signs[:, None]
        else:
            basis[mesh.negative_orientation_elements, mesh.negative_orientation_faces] = neg[:, ::-1, :]
    return cp.ascontiguousarray(basis)


def interior_trace_mass_blocks_cupy(beta_dot_normal, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    gamma_face = cp.abs(beta_dot_normal) - beta_dot_normal
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
    if timings is not None:
        timings["interior_mass"] = timings.get("interior_mass", 0.0) + sync_elapsed(start)
    return result


def trace_data_cupy(trace_blocks, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace, beta_dot_normal, timings: dict[str, float] | None = None):
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = valid_elements.size * edg_dof * edg_dof
    data = cp.empty(n_flux + n_mass, dtype=cp.float64)
    data[:n_flux] = -trace_blocks[valid_elements, valid_faces].ravel()
    data[n_flux:] = interior_trace_mass_blocks_cupy(beta_dot_normal, cspace, trace_ref, timings).ravel()
    if timings is not None:
        timings["data"] = timings.get("data", 0.0) + sync_elapsed(start)
    return data


def face_rhs_cupy(solved_src, trace_lift, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    start = time.perf_counter()
    result = (trace_lift @ solved_src[:, None, :, :]).squeeze(-1)
    if timings is not None:
        timings["rhs_faces"] = timings.get("rhs_faces", 0.0) + sync_elapsed(start)
    return result


def boundary_trace_values_cupy(boundary_condition: Callable, cspace: CupyDGSpace, trace_ref: CupyDGTraceSpace):
    cp = require_cupy()
    return cp.asarray(trace_ref.host.boundary_coefficients(boundary_condition)[cspace.host.mesh.bnd_edges_inds], dtype=cp.float64)


def build_dof_maps(cspace: CupyDGSpace):
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
    reduced_data = cp.empty(keep_count * edg_dof, dtype=cp.float64)
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


def assemble_reduced_system_gpu4(
    source,
    reaction,
    boundary_condition: Callable,
    beta_coeffs,
    cspace: DGSpace | CupyDGSpace,
    trace_space: DGTraceSpace | CupyDGTraceSpace,
    *,
    backend: Gpu4AssemblyBackend = "raw-cuda",
    beta_dot_normal=None,
    raw_block_size: int = 32,
    raw_local_assembly: RawLocalAssembly = "precomputed",
    raw_lu_mode: RawLuMode = "safe",
    raw_matrix_format: str = "coo",
) -> Gpu4AdvectionAssembly:
    """Assemble the boundary-eliminated GPU4 trace system on device."""
    cp = require_cupy()
    cspace = as_cupy_space(cspace)
    trace_ref = trace_space if isinstance(trace_space, CupyDGTraceSpace) else as_cupy_trace_space(trace_space, device=cspace.device_id)
    timings: dict[str, float] = {}
    start_total = time.perf_counter()
    if backend == "raw-cuda":
        source = _require_raw_dg_field(source, cspace, "source")
        reaction = _require_raw_dg_field(reaction, cspace, "reaction")
        if raw_local_assembly not in {"precomputed", "fused"}:
            raise ValueError("raw_local_assembly must be 'precomputed' or 'fused'")
        if raw_lu_mode not in {"safe", "coop"}:
            raise ValueError("raw_lu_mode must be 'safe' or 'coop'")
        if raw_lu_mode != "safe" and raw_local_assembly != "fused":
            raise ValueError("raw_lu_mode='coop' is only supported with raw_local_assembly='fused'")
        if raw_local_assembly == "fused":
            source_coeffs = source_coefficients_cupy(source, cspace, timings)
            reaction_coeffs, reaction_scalar, reaction_is_scalar = reaction_coefficients_cupy(reaction, cspace, timings)
            advection_tensor = reference_advection_tensor_cupy(cspace, timings)
            boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
            raw = assemble_projected_advection_trace_system_eliminated_raw_cuda_fused(
                source_coeffs=source_coeffs,
                beta_coeffs=beta_coeffs,
                reaction_coeffs=reaction_coeffs,
                reaction_scalar=reaction_scalar,
                reaction_is_scalar=reaction_is_scalar,
                boundary_trace=boundary_trace,
                cspace=cspace,
                trace_ref=trace_ref,
                advection_tensor=advection_tensor,
                block_size=raw_block_size,
                lu_mode=raw_lu_mode,
                matrix_format=raw_matrix_format,
            )
        else:
            if beta_dot_normal is None:
                beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
            local_mats = local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref, timings)
            element_boundary = element_boundary_mats_cupy(beta_dot_normal, cspace, trace_ref, timings)
            source_rhs = source_moments_cupy(source, cspace, timings)
            boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
            side_mass_blocks = interior_trace_mass_blocks_cupy(beta_dot_normal, cspace, trace_ref, timings)
            trace_lift = trace_lift_cupy(beta_dot_normal, cspace, trace_ref, timings)
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
        for key, value in raw.timings.items():
            timings[key if str(key).startswith("raw.") else f"raw.{key}"] = value
        timings["total"] = sync_elapsed(start_total)
        return Gpu4AdvectionAssembly(
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

    if beta_dot_normal is None:
        beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
    maps = build_dof_maps(cspace)
    rows, cols = setup_reduced_indices(cspace, timings)
    local_mats = local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref, timings)
    element_boundary = element_boundary_mats_cupy(beta_dot_normal, cspace, trace_ref, timings)
    source_rhs = source_moments_cupy(source, cspace, timings)
    local_rhs = cp.concatenate((element_boundary, source_rhs[..., None]), axis=2)
    solved = solve_local_mats(local_mats, local_rhs, timings, "local.solve.assembly")
    solved_el_bd = solved[:, :, : 3 * cspace.edg_dof]
    solved_src = solved[:, :, 3 * cspace.edg_dof :]
    trace_lift = trace_lift_cupy(beta_dot_normal, cspace, trace_ref, timings)
    blocks = trace_blocks_cupy(solved_el_bd, trace_lift, cspace, trace_ref, timings)
    data = trace_data_cupy(blocks, cspace, trace_ref, beta_dot_normal, timings)
    rhs_full = cp.zeros(cspace.mesh.num_edg * cspace.edg_dof, dtype=cp.float64)
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
    return Gpu4AdvectionAssembly(
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


def reconstruct_trace_cupy(trace_reduced, boundary_trace, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    cp = require_cupy()
    start = time.perf_counter()
    trace = cp.empty(cspace.mesh.num_edg * cspace.edg_dof, dtype=cp.float64)
    trace_r = trace.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    trace_r[cspace.mesh.int_edges_inds] = trace_reduced.reshape((cspace.mesh.int_edges_inds.size, cspace.edg_dof))
    trace_r[cspace.mesh.bnd_edges_inds] = boundary_trace
    if timings is not None:
        timings["reconstruct.trace"] = timings.get("reconstruct.trace", 0.0) + sync_elapsed(start)
    return trace


def reconstruct_field_gpu4(trace, source, reaction, beta_coeffs, assembly: Gpu4AdvectionAssembly):
    cp = require_cupy()
    cspace = assembly.cspace
    trace_ref = assembly.trace_ref
    raw = assembly.raw
    if raw is not None:
        if raw.local_mats is None:
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
                block_size=block_size,
                lu_mode=raw.lu_mode,
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
        return uh, kernel_elapsed

    beta_dot_normal = assembly.beta_dot_normal
    if beta_dot_normal is None:
        beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
    trace_r = trace.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    element_traces = trace_r[cspace.mesh.loc2glob_edge, :]
    if cspace.mesh.num_negative_orientations:
        neg = element_traces[cspace.mesh.negative_orientation_elements, cspace.mesh.negative_orientation_faces]
        if trace_ref.kind == "legendre-modal":
            signs = cp.where(cp.arange(cspace.edg_dof, dtype=cp.int64) % 2 == 0, 1.0, -1.0)
            element_traces[cspace.mesh.negative_orientation_elements, cspace.mesh.negative_orientation_faces] = neg * signs
        else:
            element_traces[cspace.mesh.negative_orientation_elements, cspace.mesh.negative_orientation_faces] = neg[:, ::-1]
    element_traces = element_traces.reshape((cspace.mesh.num_tri, 3 * cspace.edg_dof))
    local_mats = local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref)
    element_boundary = element_boundary_mats_cupy(beta_dot_normal, cspace, trace_ref)
    source_rhs = source_moments_cupy(source, cspace)
    rhs = source_rhs[..., None] + element_boundary @ element_traces[..., None]
    start = time.perf_counter()
    uh = cp.linalg.solve(local_mats, rhs).squeeze(-1)
    return cp.ascontiguousarray(uh), sync_elapsed(start)


def _residual_stats_cp(residual, rhs, *, rtol: float, atol: float):
    cp = require_cupy()
    residual_norm = float(cp.linalg.norm(residual).get())
    rhs_norm = float(cp.linalg.norm(rhs).get())
    relative = residual_norm / rhs_norm if rhs_norm > 0.0 else residual_norm
    target = max(float(rtol) * rhs_norm, float(atol))
    return residual_norm, rhs_norm, relative, target


@dataclass(frozen=True)
class _DeviceCsrMatrixView:
    """Device-owned scalar CSR arrays accepted by PyAMGX Matrix.upload."""

    data: Any
    indices: Any
    indptr: Any
    shape: tuple[int, int]


def _as_cupyx_csr_matrix(matrix, sparse, cp):
    if isinstance(matrix, _DeviceCsrMatrixView):
        return sparse.csr_matrix(
            (matrix.data, matrix.indices, matrix.indptr),
            shape=matrix.shape,
            dtype=cp.float64,
        )
    return matrix


_CSR_ROW_SCALE_SOURCE = r"""
extern "C" __global__ void diagonal_scale_csr_rows(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        double* __restrict__ data,
        double* __restrict__ rhs,
        double* __restrict__ row_diagonal,
        const long long nrows)
{
    const long long row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int start = indptr[row];
    const int end = indptr[row + 1];
    __shared__ double diagonal;
    if (threadIdx.x == 0) {
        double value = 0.0;
        for (int p = start; p < end; ++p) {
            if (indices[p] == (int)row) {
                value += data[p];
            }
        }
        if (value == 0.0) {
            value = 1.0;
        }
        diagonal = value;
        row_diagonal[row] = value;
        rhs[row] /= value;
    }
    __syncthreads();
    const double inverse = 1.0 / diagonal;
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        data[p] *= inverse;
    }
}
"""
_CSR_ROW_SCALE_KERNELS: dict[int, Any] = {}


def _diagonal_scale_csr_rows_in_place(matrix, rhs):
    cp = require_cupy()
    device_id = int(cp.cuda.runtime.getDevice())
    kernel = _CSR_ROW_SCALE_KERNELS.get(device_id)
    if kernel is None:
        kernel = cp.RawKernel(_CSR_ROW_SCALE_SOURCE, "diagonal_scale_csr_rows")
        _CSR_ROW_SCALE_KERNELS[device_id] = kernel
    nrows = int(rhs.size)
    diagonal = cp.empty(nrows, dtype=cp.float64)
    if nrows:
        kernel(
            (nrows,),
            (128,),
            (matrix.indptr, matrix.indices, matrix.data, rhs, diagonal, np.int64(nrows)),
        )
    return diagonal


def _pyamgx_solve_csr_device(matrix, rhs, *, config=None, tolerance: float = 1e-13, maxiter: int | None = None, verbose: bool | int = 0):
    """Solve a device CSR system with PyAMGX without staging through host CSR."""
    cp = require_cupy()
    pyamgx = require_pyamgx()
    from .cupy import default_pyamgx_config

    if config is None:
        amgx_config = default_pyamgx_config(tolerance=tolerance, maxiter=maxiter, verbose=verbose)
    else:
        amgx_config = copy.deepcopy(config)
        if maxiter is not None:
            amgx_config.setdefault("solver", {})["max_iters"] = int(maxiter)

    verbose_level = 1 if isinstance(verbose, bool) and verbose else (0 if not verbose else int(verbose))
    if verbose_level >= 2:
        solver_config = amgx_config.setdefault("solver", {})
        solver_config["print_solve_stats"] = 1
        solver_config["obtain_timings"] = 1

    pyamgx.initialize()
    cfg = rsrc = mat = vec_b = vec_x = solver = None
    x = cp.zeros_like(rhs)
    info = {"amgx_status": "unknown", "amgx_iterations": None}
    try:
        cfg = pyamgx.Config().create_from_dict(amgx_config)
        rsrc = pyamgx.Resources().create_simple(cfg)
        mat = pyamgx.Matrix().create(rsrc, mode="dDDI")
        vec_b = pyamgx.Vector().create(rsrc, mode="dDDI")
        vec_x = pyamgx.Vector().create(rsrc, mode="dDDI")
        setup_start = time.perf_counter()
        mat.upload(matrix.indptr, matrix.indices, matrix.data, shape=matrix.shape)
        vec_b.upload_raw(rhs.data.ptr, rhs.size)
        vec_x.upload_raw(x.data.ptr, x.size)
        solver = pyamgx.Solver().create(rsrc, cfg)
        solver.setup(mat)
        cp.cuda.get_current_stream().synchronize()
        setup_elapsed = time.perf_counter() - setup_start
        solve_start = time.perf_counter()
        solver.solve(vec_b, vec_x)
        vec_x.download_raw(x.data.ptr)
        cp.cuda.get_current_stream().synchronize()
        solve_elapsed = time.perf_counter() - solve_start
        try:
            info["amgx_status"] = str(solver.status)
        except Exception:
            pass
        try:
            info["amgx_iterations"] = int(solver.iterations_number)
        except Exception:
            pass
    finally:
        for obj in (solver, mat, vec_x, vec_b, rsrc, cfg):
            if obj is not None:
                try:
                    obj.destroy()
                except AttributeError:
                    pass
        pyamgx.finalize()
        _flush_c_stdio()
    info["amgx_setup_elapsed_seconds"] = setup_elapsed
    info["amgx_solve_elapsed_seconds"] = solve_elapsed
    return x, info


def solve_reduced_system_amgx_device(
    assembly: Gpu4AdvectionAssembly,
    *,
    config=None,
    tolerance: float = 1e-13,
    check_rtol: float | None = None,
    atol: float = 0.0,
    maxiter: int | None = None,
    scale_system: bool = True,
    raise_on_nonconvergence: bool = True,
    materialize_host_solution: bool = True,
    verbose: bool | int = 0,
):
    """Solve a GPU4 reduced trace system with device CSR and PyAMGX."""
    cp = require_cupy()
    sparse = require_cupyx_sparse()
    total_start = time.perf_counter()
    system_size = int(assembly.rhs.size)

    matrix_start = time.perf_counter()
    if getattr(assembly, "matrix_format", "coo") == "csr":
        if assembly.indptr is None or assembly.indices is None:
            raise RuntimeError("CSR assembly is missing indptr/indices")
        matrix = _DeviceCsrMatrixView(
            data=assembly.data,
            indices=assembly.indices.astype(cp.int32, copy=False),
            indptr=assembly.indptr.astype(cp.int32, copy=False),
            shape=(system_size, system_size),
        )
    else:
        matrix = sparse.coo_matrix(
            (assembly.data, (assembly.rows.astype(cp.int32), assembly.cols.astype(cp.int32))),
            shape=(system_size, system_size),
            dtype=cp.float64,
        ).tocsr()
        matrix.sum_duplicates()
        if matrix.indices.dtype != cp.int32 or matrix.indptr.dtype != cp.int32:
            matrix = sparse.csr_matrix(
                (matrix.data, matrix.indices.astype(cp.int32, copy=False), matrix.indptr.astype(cp.int32, copy=False)),
                shape=matrix.shape,
                dtype=cp.float64,
            )
    cp.cuda.get_current_stream().synchronize()
    matrix_elapsed = time.perf_counter() - matrix_start

    physical_rhs = assembly.rhs
    row_diagonal = None
    scale_start = time.perf_counter()
    if scale_system:
        solve_matrix = matrix
        solve_rhs = physical_rhs.copy()
        row_diagonal = _diagonal_scale_csr_rows_in_place(solve_matrix, solve_rhs)
    else:
        solve_matrix = matrix
        solve_rhs = physical_rhs
    cp.cuda.get_current_stream().synchronize()
    scale_elapsed = time.perf_counter() - scale_start

    solve_tolerance = float(tolerance)
    result_check_rtol = solve_tolerance if check_rtol is None else float(check_rtol)
    amgx_call_start = time.perf_counter()
    x_cp, amgx_info = _pyamgx_solve_csr_device(
        solve_matrix,
        solve_rhs,
        config=config,
        tolerance=solve_tolerance,
        maxiter=maxiter,
        verbose=verbose,
    )
    amgx_call_elapsed = time.perf_counter() - amgx_call_start

    finite_start = time.perf_counter()
    solution_is_finite = bool(cp.all(cp.isfinite(x_cp)).get())
    finite_elapsed = time.perf_counter() - finite_start

    solver_residual_start = time.perf_counter()
    residual_matrix = _as_cupyx_csr_matrix(solve_matrix, sparse, cp)
    solver_residual = residual_matrix @ x_cp - solve_rhs
    solver_residual_norm, solver_rhs_norm, solver_relative, solver_target = _residual_stats_cp(
        solver_residual,
        solve_rhs,
        rtol=result_check_rtol,
        atol=atol,
    )
    solver_residual_elapsed = time.perf_counter() - solver_residual_start

    physical_residual_start = time.perf_counter()
    if row_diagonal is not None:
        physical_residual = row_diagonal * solver_residual
        physical_residual_rhs = row_diagonal * solve_rhs
    else:
        physical_residual = residual_matrix @ x_cp - physical_rhs
        physical_residual_rhs = physical_rhs
    physical_residual_norm, physical_rhs_norm, physical_relative, physical_target = _residual_stats_cp(
        physical_residual,
        physical_residual_rhs,
        rtol=result_check_rtol,
        atol=atol,
    )
    physical_residual_elapsed = time.perf_counter() - physical_residual_start
    validation_elapsed = finite_elapsed + solver_residual_elapsed + physical_residual_elapsed
    info = 0 if solution_is_finite and solver_residual_norm <= solver_target else 1
    total_elapsed = time.perf_counter() - total_start
    if info != 0 and raise_on_nonconvergence:
        if not solution_is_finite:
            raise RuntimeError("PyAMGX solver returned non-finite solution values")
        raise RuntimeError(
            "PyAMGX solver did not satisfy the requested residual target. "
            f"residual={solver_residual_norm:.3e}, target={solver_target:.3e}, "
            f"relative_residual={solver_relative:.3e}"
        )
    verbose_level = 1 if isinstance(verbose, bool) and verbose else (0 if not verbose else int(verbose))
    if verbose_level >= 2:
        print("  PyAMGX device solve timings:", flush=True)
        print(f"    csr build: {matrix_elapsed:.5f}s", flush=True)
        print(f"    row scaling: {scale_elapsed:.5f}s", flush=True)
        print(f"    setup: {amgx_info['amgx_setup_elapsed_seconds']:.5f}s", flush=True)
        print(f"    iterate: {amgx_info['amgx_solve_elapsed_seconds']:.5f}s", flush=True)
        print(f"    amgx call total: {amgx_call_elapsed:.5f}s", flush=True)
        print(f"    validation: {validation_elapsed:.5f}s", flush=True)
        print(f"    solver relative residual: {solver_relative:.3e}", flush=True)
    elif verbose_level:
        print(
            f"  PyAMGX: csr={matrix_elapsed:.5f}s scale={scale_elapsed:.5f}s "
            f"setup={amgx_info['amgx_setup_elapsed_seconds']:.5f}s "
            f"solve={amgx_info['amgx_solve_elapsed_seconds']:.5f}s rel={solver_relative:.3e}",
            flush=True,
        )
    result = SolveResult(
        x=cp.asnumpy(x_cp) if materialize_host_solution else None,
        residual_norm=solver_residual_norm,
        info=info,
        preconditioner=None,
        total_elapsed_seconds=total_elapsed,
        scale_elapsed_seconds=scale_elapsed,
        preconditioner_elapsed_seconds=matrix_elapsed + amgx_info["amgx_setup_elapsed_seconds"],
        solve_elapsed_seconds=amgx_info["amgx_solve_elapsed_seconds"],
        iteration_count=amgx_info.get("amgx_iterations"),
        rhs_norm=solver_rhs_norm,
        relative_residual_norm=solver_relative,
        residual_target=solver_target,
        solver_residual_norm=solver_residual_norm,
        solver_rhs_norm=solver_rhs_norm,
        solver_relative_residual_norm=solver_relative,
        solver_residual_target=solver_target,
        physical_residual_norm=physical_residual_norm,
        physical_rhs_norm=physical_rhs_norm,
        physical_relative_residual_norm=physical_relative,
        physical_residual_target=physical_target,
        rtol=result_check_rtol,
        atol=atol,
    )
    result.cupyx_solver = "pyamgx-device"
    result.amgx_csr_elapsed_seconds = matrix_elapsed
    result.amgx_setup_elapsed_seconds = amgx_info["amgx_setup_elapsed_seconds"]
    result.amgx_solve_elapsed_seconds = amgx_info["amgx_solve_elapsed_seconds"]
    result.amgx_call_elapsed_seconds = amgx_call_elapsed
    result.amgx_overhead_elapsed_seconds = max(
        0.0,
        amgx_call_elapsed - amgx_info["amgx_setup_elapsed_seconds"] - amgx_info["amgx_solve_elapsed_seconds"],
    )
    result.solve_finite_check_elapsed_seconds = finite_elapsed
    result.solve_solver_residual_elapsed_seconds = solver_residual_elapsed
    result.solve_physical_residual_elapsed_seconds = physical_residual_elapsed
    result.solve_validation_elapsed_seconds = validation_elapsed
    result.solve_accounted_elapsed_seconds = matrix_elapsed + scale_elapsed + amgx_call_elapsed + validation_elapsed
    result.solve_unaccounted_elapsed_seconds = max(0.0, total_elapsed - result.solve_accounted_elapsed_seconds)
    result.solve_global_overhead_elapsed_seconds = result.solve_unaccounted_elapsed_seconds
    return result, x_cp


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
    raw_block_size: int = 1,
    raw_local_assembly: str = "precomputed",
    raw_lu_mode: str = "safe",
    raw_matrix_format: str = "coo",
):
    """Compatibility wrapper matching the former GPU runner helper signature."""
    assembly = assemble_reduced_system_gpu4(
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
    "CupyDGTraceSpace",
    "Gpu4AdvectionAssembly",
    "as_cupy_trace_space",
    "assemble_reduced_system_gpu4",
    "beta_dot_normal_from_coeffs",
    "build_dof_maps",
    "project_callable_cupy",
    "reconstruct_field_gpu4",
    "reconstruct_trace_cupy",
    "solve_reduced_system_amgx_device",
]
