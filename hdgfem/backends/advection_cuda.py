"""CUDA-resident CuPy/raw-CUDA advection-reaction HDG assembly helpers.

This module owns the device execution formalism for the fast advection-reaction
path. Public solvers still accept host :class:`DGSpace` and field objects; this
backend mirrors immutable space/trace data to CUDA and returns device arrays to
callers that can keep the global solve on device, or host arrays when requested.
"""

from __future__ import annotations

from hdgfem.runtime.precision import (
    audit_arrays,
    REAL_DTYPE,
    AMGX_MODE,
    real_raw_kernel,
)

import copy
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from hdgfem.core.space import DGField, DGSpace, DGTraceSpace
from hdgfem.io.config import format_amgx_configuration
from hdgfem.linalg.system import (
    KnownDofReduction,
    LinearSolveCapacityError,
    LinearSolveConvergenceError,
    SolveResult,
    finalize_solve_result,
)
from hdgfem.core.device import CupyDGSpace, as_cupy_coefficients, as_cupy_space
from hdgfem.backends.cupy import (
    initialize_pyamgx_once,
    symmetric_scale_cupy_csr_in_place,
    diagonal_scale_cupy_csr_rows_in_place as _diagonal_scale_csr_rows_in_place,
)
from hdgfem.runtime.optional import require_cupy, require_cupyx_sparse, require_pyamgx
from hdgfem.backends.raw_cuda import RawCudaBlockSize
from hdgfem.backends.amgx_errors import (
    as_amgx_capacity_error as _as_amgx_capacity_error,
    destroy_amgx_objects,
)
from hdgfem.backends.advection_raw_cuda import (
    RawAdvectionAssemblyResult,
    assemble_projected_advection_trace_system_eliminated_raw_cuda,
    assemble_projected_advection_trace_system_eliminated_raw_cuda_fused,
    reconstruct_projected_advection_field_raw_cuda,
    reconstruct_projected_advection_field_raw_cuda_fused,
    reconstruct_projected_advection_field_from_response_raw_cuda,
)
from hdgfem.backends.advection_tsle_bsr import (
    RawAdvectionTsleWorkspace,
    assemble_projected_advection_trace_system_eliminated_tsle_bsr,
)
from hdgfem.core.device import CupyDGTraceSpace, as_cupy_trace_space


RawLocalAssembly = Literal["precomputed", "fused", "split3"]
RawLuMode = Literal["safe", "coop"]
CudaAdvectionAssemblyBackend = Literal["cupy", "raw-cuda"]


@dataclass(frozen=True)
class CudaAdvectionAssembly:
    """Reduced trace system assembled by the CUDA path.

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
        known_values = np.zeros(full_size, dtype=REAL_DTYPE)
        boundary_host = np.ascontiguousarray(cp.asnumpy(self.boundary_trace), dtype=REAL_DTYPE)
        known_values[(space.mesh.bnd_edges_inds[:, None] * edg_dof + local[None, :]).ravel()] = boundary_host.ravel()
        old_to_new = np.full(full_size, -1, dtype=np.int64)
        old_to_new[free_mask] = np.arange(np.count_nonzero(free_mask), dtype=np.int64)
        return KnownDofReduction(
            rows=np.ascontiguousarray(cp.asnumpy(self.rows), dtype=np.int64),
            cols=np.ascontiguousarray(cp.asnumpy(self.cols), dtype=np.int64),
            data=np.ascontiguousarray(cp.asnumpy(self.data), dtype=REAL_DTYPE),
            rhs=np.ascontiguousarray(cp.asnumpy(self.rhs), dtype=REAL_DTYPE),
            free_mask=np.ascontiguousarray(free_mask),
            known_mask=np.ascontiguousarray(known_mask),
            known_values=np.ascontiguousarray(known_values),
            old_to_new=np.ascontiguousarray(old_to_new),
        )


def sync_elapsed(start: float) -> float:
    """Synchronize the active CUDA stream and return elapsed wall time."""
    cp = require_cupy()
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter() - start


def mapped_quads_cupy(cspace: CupyDGSpace):
    """Return physical volume quadrature points resident on the device."""
    return cspace.mapped_quads


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


def source_moments_cupy(source, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Assemble element source moments on the device.

    ``source`` is a DG field, a CuPy-compatible callable, or a device array of
    element moments ``(K, el_dof)`` or volume-quadrature values ``(K, nq)``;
    as in ``hdg.source_moments``, the moment shape wins when both match.
    """
    cp = require_cupy()
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    if isinstance(source, cp.ndarray):
        num_elements, el_dof, num_points = cspace.host.mesh.num_tri, cspace.host.el_dof, q.Krf_w.size
        values = source.astype(REAL_DTYPE, copy=False)
        if values.shape == (num_elements, el_dof):
            rhs = values
        elif values.shape == (num_elements, num_points):
            rhs = source_moments_from_values_cupy(values, cspace)
        else:
            raise ValueError(f"device source must have shape ({num_elements}, {el_dof}) moments or "
                             f"({num_elements}, {num_points}) quadrature values; got {values.shape}")
    elif isinstance(source, DGField):
        source.space.assert_same_mesh(cspace.host)
        constant_value = source.constant_value
        if constant_value is not None:
            ref_moments = cp.asarray(cspace.host._constant_reference_moments(constant_value))
            rhs = mesh.aff_jacs[:, None] * ref_moments[None, :]
        elif source.space is cspace.host:
            coeffs = as_cupy_coefficients(source, cspace)
            rhs = mesh.aff_jacs[:, None] * (coeffs @ q.MKrf)
        else:
            # A field from another DG space on this mesh is sampled on this
            # space's volume quadrature, as in hdg.source_moments.
            rhs = source_moments_from_values_cupy(_field_on_volume_quadrature_cupy(source, cspace), cspace)
    else:
        points = mapped_quads_cupy(cspace)
        values = cp.asarray(source(points[:, 0, :], points[:, 1, :]), dtype=REAL_DTYPE)
        rhs = source_moments_from_values_cupy(values, cspace)
    if timings is not None:
        timings["source_moments"] = timings.get("source_moments", 0.0) + sync_elapsed(start)
    return cp.ascontiguousarray(rhs)


def _field_on_volume_quadrature_cupy(field: DGField, cspace: CupyDGSpace):
    """Device values ``(K, nq)`` of a same-mesh DG field on ``cspace`` volume quadrature."""
    cp = require_cupy()
    source_space = field.space
    coeffs = as_cupy_coefficients(field, as_cupy_space(source_space, device=cspace.device_id))
    basis = cp.asarray(source_space.basis_at(cspace.host.quad_data.Krf_quads), dtype=REAL_DTYPE)
    return coeffs @ basis.T


def source_moments_from_values_cupy(values, cspace: CupyDGSpace):
    """Element moments from device values on volume quadrature, shape ``(K, nq)``."""
    cp = require_cupy()
    q = cspace.quad_data
    return cspace.mesh.aff_jacs[:, None] * cp.einsum("Kq,iq,q->Ki", values, q.bas_of_quads, q.Krf_w)


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
        from hdgfem.solvers.stabilization import gauge_inactive_advection_trace_blocks
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
    from hdgfem.assembly.hdg import boundary_trace_coefficients
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
    from hdgfem.solvers.stabilization import upwind_factor, effective_advection_normal_flux, is_conflict_averaged_upwind
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
    from hdgfem.backends.advection_raw_cuda import solve_cached_advection_source_raw
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


def reconstruct_trace_cupy(trace_reduced, boundary_trace, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Expand reduced trace values into the full device trace vector."""
    cp = require_cupy()
    start = time.perf_counter()
    trace = cp.empty(cspace.mesh.num_edg * cspace.edg_dof, dtype=REAL_DTYPE)
    trace_r = trace.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    trace_r[cspace.mesh.int_edges_inds] = trace_reduced.reshape((cspace.mesh.int_edges_inds.size, cspace.edg_dof))
    trace_r[cspace.mesh.bnd_edges_inds] = boundary_trace
    if timings is not None:
        timings["reconstruct.trace"] = timings.get("reconstruct.trace", 0.0) + sync_elapsed(start)
    return trace


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
        return uh, kernel_elapsed

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
    return cp.ascontiguousarray(uh), sync_elapsed(start)


def _residual_stats_cp(residual, rhs, *, rtol: float, atol: float):
    """Compute device residual norms and the requested convergence target."""
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


@dataclass(frozen=True)
class _DeviceBsrMatrixView:
    """Device-owned face-BSR arrays accepted directly by PyAMGX."""

    data: Any
    indices: Any
    indptr: Any
    shape: tuple[int, int]
    block_size: int


_DEVICE_BSR_MATVEC_SOURCE = r"""
extern "C" __global__ void device_bsr_matvec(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data,
        const double* __restrict__ x,
        double* __restrict__ y,
        const int num_block_rows,
        const int block_size)
{
    const int scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_rows = num_block_rows * block_size;
    if (scalar_row >= num_rows) {
        return;
    }
    const int block_row = scalar_row / block_size;
    const int row_dof = scalar_row - block_row * block_size;
    double value = 0.0;
    for (int block = indptr[block_row]; block < indptr[block_row + 1]; ++block) {
        const int column_base = indices[block] * block_size;
        const long long data_base = (
            (long long)block * block_size + row_dof
        ) * block_size;
        for (int col_dof = 0; col_dof < block_size; ++col_dof) {
            value += data[data_base + col_dof] * x[column_base + col_dof];
        }
    }
    y[scalar_row] = value;
}
"""
_DEVICE_BSR_MATVEC_KERNELS: dict[int, Any] = {}


def _as_cupyx_csr_matrix(matrix, sparse, cp):
    """Expose a scalar compressed device view as a Cupyx CSR matrix."""
    if isinstance(matrix, _DeviceCsrMatrixView):
        return sparse.csr_matrix(
            (matrix.data, matrix.indices, matrix.indptr),
            shape=matrix.shape,
            dtype=REAL_DTYPE,
        )
    return matrix


_DEVICE_BSR_TO_SCALAR_CSR_SOURCE = r"""
extern "C" __global__ void device_bsr_to_scalar_csr(
        const int* __restrict__ block_indptr,
        const int* __restrict__ block_indices,
        const double* __restrict__ block_data,
        int* __restrict__ scalar_indptr,
        int* __restrict__ scalar_indices,
        double* __restrict__ scalar_data,
        const int num_block_rows,
        const int block_size)
{
    const int scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_scalar_rows = num_block_rows * block_size;
    if (scalar_row > num_scalar_rows) {
        return;
    }
    if (scalar_row == num_scalar_rows) {
        scalar_indptr[scalar_row] =
            block_indptr[num_block_rows] * block_size * block_size;
        return;
    }
    const int block_row = scalar_row / block_size;
    const int row_dof = scalar_row - block_row * block_size;
    const int block_begin = block_indptr[block_row];
    const int block_end = block_indptr[block_row + 1];
    const int blocks_in_row = block_end - block_begin;
    const int scalar_begin =
        block_begin * block_size * block_size
        + row_dof * blocks_in_row * block_size;
    scalar_indptr[scalar_row] = scalar_begin;
    int output = scalar_begin;
    for (int block = block_begin; block < block_end; ++block) {
        const int scalar_column = block_indices[block] * block_size;
        const long long data_begin =
            ((long long)block * block_size + row_dof) * block_size;
        for (int column_dof = 0; column_dof < block_size; ++column_dof) {
            scalar_indices[output] = scalar_column + column_dof;
            scalar_data[output] = block_data[data_begin + column_dof];
            ++output;
        }
    }
}
"""
_DEVICE_BSR_TO_SCALAR_CSR_KERNELS: dict[int, Any] = {}


def _scalarize_device_bsr_matrix(matrix: _DeviceBsrMatrixView, sparse, cp):
    """Expand a face-BSR view to scalar CUDA CSR without host staging."""
    block_size = int(matrix.block_size)
    num_block_rows = int(matrix.shape[0] // block_size)
    scalar_indptr = cp.empty(matrix.shape[0] + 1, dtype=cp.int32)
    scalar_indices = cp.empty(int(matrix.data.size), dtype=cp.int32)
    scalar_data = cp.empty(int(matrix.data.size), dtype=REAL_DTYPE)
    device_id = int(cp.cuda.runtime.getDevice())
    kernel = _DEVICE_BSR_TO_SCALAR_CSR_KERNELS.get(device_id)
    if kernel is None:
        kernel = real_raw_kernel(
            _DEVICE_BSR_TO_SCALAR_CSR_SOURCE, "device_bsr_to_scalar_csr"
        )
        _DEVICE_BSR_TO_SCALAR_CSR_KERNELS[device_id] = kernel
    threads = 256
    rows_with_terminal = int(matrix.shape[0]) + 1
    kernel(
        ((rows_with_terminal + threads - 1) // threads,),
        (threads,),
        (
            matrix.indptr,
            matrix.indices,
            matrix.data,
            scalar_indptr,
            scalar_indices,
            scalar_data,
            np.int32(num_block_rows),
            np.int32(block_size),
        ),
    )
    return sparse.csr_matrix(
        (scalar_data, scalar_indices, scalar_indptr),
        shape=matrix.shape,
        dtype=REAL_DTYPE,
    )


def _device_compressed_matvec(matrix, vector, sparse, cp):
    """Apply a device CSR or face-BSR matrix without host materialization."""
    if not isinstance(matrix, _DeviceBsrMatrixView):
        return _as_cupyx_csr_matrix(matrix, sparse, cp) @ vector
    output = cp.empty(matrix.shape[0], dtype=REAL_DTYPE)
    threads = 256
    blocks = (matrix.shape[0] + threads - 1) // threads
    device_id = int(cp.cuda.runtime.getDevice())
    kernel = _DEVICE_BSR_MATVEC_KERNELS.get(device_id)
    if kernel is None:
        kernel = real_raw_kernel(_DEVICE_BSR_MATVEC_SOURCE, "device_bsr_matvec")
        _DEVICE_BSR_MATVEC_KERNELS[device_id] = kernel
    kernel(
        (blocks,),
        (threads,),
        (
            matrix.indptr,
            matrix.indices,
            matrix.data,
            vector,
            output,
            np.int32(matrix.shape[0] // matrix.block_size),
            np.int32(matrix.block_size),
        ),
    )
    return output


def _assembly_device_csr_matrix(assembly: CudaAdvectionAssembly, cp, sparse):
    """Build a device CSR matrix view from assembled COO or CSR data."""
    system_size = int(assembly.rhs.size)
    matrix_format = getattr(assembly, "matrix_format", "coo")
    if matrix_format == "csr":
        if assembly.indptr is None or assembly.indices is None:
            raise RuntimeError("CSR assembly is missing indptr/indices")
        return _DeviceCsrMatrixView(
            data=assembly.data,
            indices=assembly.indices.astype(cp.int32, copy=False),
            indptr=assembly.indptr.astype(cp.int32, copy=False),
            shape=(system_size, system_size),
        )
    if matrix_format == "bsr":
        if assembly.indptr is None or assembly.indices is None:
            raise RuntimeError("BSR assembly is missing block indptr/indices")
        block_size = int(assembly.data.shape[-1])
        return _DeviceBsrMatrixView(
            data=assembly.data,
            indices=assembly.indices.astype(cp.int32, copy=False),
            indptr=assembly.indptr.astype(cp.int32, copy=False),
            shape=(system_size, system_size),
            block_size=block_size,
        )

    matrix = sparse.coo_matrix(
        (assembly.data, (assembly.rows.astype(cp.int32), assembly.cols.astype(cp.int32))),
        shape=(system_size, system_size),
        dtype=REAL_DTYPE,
    ).tocsr()
    matrix.sum_duplicates()
    if matrix.indices.dtype != cp.int32 or matrix.indptr.dtype != cp.int32:
        matrix = sparse.csr_matrix(
            (
                matrix.data,
                matrix.indices.astype(cp.int32, copy=False),
                matrix.indptr.astype(cp.int32, copy=False),
            ),
            shape=matrix.shape,
            dtype=REAL_DTYPE,
        )
    return matrix


_CSR_ROW_UNSCALE_SOURCE = r"""
extern "C" __global__ void restore_left_scaled_csr_rows(
        const int* __restrict__ indptr,
        double* __restrict__ data,
        const double* __restrict__ row_diagonal,
        const long long nrows)
{
    const long long row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int start = indptr[row];
    const int end = indptr[row + 1];
    const double row_scale = row_diagonal[row];
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        data[p] *= row_scale;
    }
}

extern "C" __global__ void restore_symmetric_scaled_csr_rows(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        double* __restrict__ data,
        const double* __restrict__ inverse_sqrt_diagonal,
        const long long nrows)
{
    const long long row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int start = indptr[row];
    const int end = indptr[row + 1];
    const double row_scale = inverse_sqrt_diagonal[row];
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        data[p] /= row_scale * inverse_sqrt_diagonal[indices[p]];
    }
}
"""
_CSR_ROW_UNSCALE_KERNELS: dict[int, tuple[Any, Any]] = {}


def _restore_scaled_csr_rows_in_place(
    matrix,
    *,
    row_diagonal=None,
    inverse_sqrt_diagonal=None,
):
    """Restore CSR values after left or symmetric device scaling."""
    if row_diagonal is None and inverse_sqrt_diagonal is None:
        return
    if row_diagonal is not None and inverse_sqrt_diagonal is not None:
        raise ValueError("exactly one CSR scaling vector may be restored")
    cp = require_cupy()
    device_id = int(cp.cuda.runtime.getDevice())
    kernels = _CSR_ROW_UNSCALE_KERNELS.get(device_id)
    if kernels is None:
        left_kernel = real_raw_kernel(_CSR_ROW_UNSCALE_SOURCE, "restore_left_scaled_csr_rows")
        symmetric_kernel = real_raw_kernel(
            _CSR_ROW_UNSCALE_SOURCE,
            "restore_symmetric_scaled_csr_rows",
        )
        kernels = (left_kernel, symmetric_kernel)
        _CSR_ROW_UNSCALE_KERNELS[device_id] = kernels
    nrows = int(matrix.shape[0])
    if not nrows:
        return
    left_kernel, symmetric_kernel = kernels
    if row_diagonal is not None:
        left_kernel(
            (nrows,),
            (128,),
            (matrix.indptr, matrix.data, row_diagonal, np.int64(nrows)),
        )
    else:
        symmetric_kernel(
            (nrows,),
            (128,),
            (matrix.indptr, matrix.indices, matrix.data, inverse_sqrt_diagonal, np.int64(nrows)),
        )


_BSR_ROW_SCALE_SOURCE = r"""
extern "C" __global__ void diagonal_scale_bsr_rows(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        double* __restrict__ data,
        double* __restrict__ rhs,
        double* __restrict__ row_diagonal,
        const int num_block_rows,
        const int block_size)
{
    const int row = blockIdx.x;
    const int nrows = num_block_rows * block_size;
    if (row >= nrows) {
        return;
    }
    const int block_row = row / block_size;
    const int row_dof = row - block_row * block_size;
    const int start = indptr[block_row];
    const int end = indptr[block_row + 1];
    __shared__ double row_scale;
    if (threadIdx.x == 0) {
        double diagonal = 0.0;
        double row_max = 0.0;
        for (int block = start; block < end; ++block) {
            const long long base = ((long long)block * block_size + row_dof) * block_size;
            for (int col_dof = 0; col_dof < block_size; ++col_dof) {
                const double entry = data[base + col_dof];
                row_max = fmax(row_max, fabs(entry));
                if (indices[block] == block_row && col_dof == row_dof) {
                    diagonal += entry;
                }
            }
        }
        double value = diagonal;
        if (!isfinite(value) || fabs(value) <= 1.0e-10 * row_max) {
            value = row_max;
        }
        if (!isfinite(value) || value == 0.0) {
            value = 1.0;
        }
        row_scale = value;
        row_diagonal[row] = value;
        rhs[row] /= value;
    }
    __syncthreads();
    const int row_entries = (end - start) * block_size;
    const double inverse = 1.0 / row_scale;
    for (int entry = threadIdx.x; entry < row_entries; entry += blockDim.x) {
        const int block_offset = entry / block_size;
        const int col_dof = entry - block_offset * block_size;
        const long long offset = (
            ((long long)(start + block_offset) * block_size + row_dof) * block_size
            + col_dof
        );
        data[offset] *= inverse;
    }
}

extern "C" __global__ void restore_left_scaled_bsr_rows(
        const int* __restrict__ indptr,
        double* __restrict__ data,
        const double* __restrict__ row_diagonal,
        const int num_block_rows,
        const int block_size)
{
    const int row = blockIdx.x;
    const int nrows = num_block_rows * block_size;
    if (row >= nrows) {
        return;
    }
    const int block_row = row / block_size;
    const int row_dof = row - block_row * block_size;
    const int start = indptr[block_row];
    const int end = indptr[block_row + 1];
    const int row_entries = (end - start) * block_size;
    const double row_scale = row_diagonal[row];
    for (int entry = threadIdx.x; entry < row_entries; entry += blockDim.x) {
        const int block_offset = entry / block_size;
        const int col_dof = entry - block_offset * block_size;
        const long long offset = (
            ((long long)(start + block_offset) * block_size + row_dof) * block_size
            + col_dof
        );
        data[offset] *= row_scale;
    }
}
"""
_BSR_ROW_SCALE_KERNELS: dict[int, tuple[Any, Any]] = {}


def _bsr_row_scale_kernels():
    """Return cached scalar-row scale/restore kernels for face-BSR matrices."""
    cp = require_cupy()
    device_id = int(cp.cuda.runtime.getDevice())
    kernels = _BSR_ROW_SCALE_KERNELS.get(device_id)
    if kernels is None:
        kernels = (
            real_raw_kernel(_BSR_ROW_SCALE_SOURCE, "diagonal_scale_bsr_rows"),
            real_raw_kernel(_BSR_ROW_SCALE_SOURCE, "restore_left_scaled_bsr_rows"),
        )
        _BSR_ROW_SCALE_KERNELS[device_id] = kernels
    return kernels


def _diagonal_scale_bsr_rows_in_place(matrix: _DeviceBsrMatrixView, rhs):
    """Apply the scalar CSR left-scaling rule directly to face-BSR values."""
    cp = require_cupy()
    nrows = int(rhs.size)
    diagonal = cp.empty(nrows, dtype=REAL_DTYPE)
    if nrows:
        scale_kernel, _ = _bsr_row_scale_kernels()
        scale_kernel(
            (nrows,),
            (128,),
            (
                matrix.indptr,
                matrix.indices,
                matrix.data,
                rhs,
                diagonal,
                np.int32(nrows // matrix.block_size),
                np.int32(matrix.block_size),
            ),
        )
    return diagonal


def _restore_left_scaled_bsr_rows_in_place(
    matrix: _DeviceBsrMatrixView,
    row_diagonal,
) -> None:
    """Restore face-BSR values after scalar-row left scaling."""
    nrows = int(matrix.shape[0])
    if not nrows:
        return
    _, restore_kernel = _bsr_row_scale_kernels()
    restore_kernel(
        (nrows,),
        (128,),
        (
            matrix.indptr,
            matrix.data,
            row_diagonal,
            np.int32(nrows // matrix.block_size),
            np.int32(matrix.block_size),
        ),
    )


_AMGX_REUSABLE_SOLVERS = []


class _PyAMGXSharedResourceManager:
    """Own one process-wide AMGX Resources handle shared by live solvers."""

    def __init__(self):
        """Initialize the instance."""
        self.pyamgx = None
        self.resource_cfg = None
        self.rsrc = None
        self.refcount = 0

    def acquire(self, pyamgx, resource_config: dict):
        """Acquire the shared AMGX resource handle and increment its owner count."""
        initialize_pyamgx_once()
        if self.rsrc is None:
            self.pyamgx = pyamgx
            try:
                self.resource_cfg = pyamgx.Config()
                self.resource_cfg.create_from_dict(copy.deepcopy(resource_config))
                self.rsrc = pyamgx.Resources()
                self.rsrc.create_simple(self.resource_cfg)
            except Exception as exc:
                capacity_error = _as_amgx_capacity_error(
                    exc, phase="resource acquisition", cp=require_cupy(), pyamgx=pyamgx
                )
                self.release(suppress_errors=True)
                if capacity_error is not None and capacity_error is not exc:
                    raise capacity_error from exc
                raise
        self.refcount += 1
        return self.rsrc

    def release(self, *, suppress_errors: bool = False) -> None:
        """Release one owner and destroy shared AMGX resources when unused."""
        if self.refcount > 0:
            self.refcount -= 1
        if self.refcount != 0:
            return
        try:
            destroy_amgx_objects(
                (self.rsrc, self.resource_cfg), suppress_errors=suppress_errors
            )
        finally:
            self.rsrc = None
            self.resource_cfg = None
            self.pyamgx = None


_AMGX_SHARED_RESOURCES = _PyAMGXSharedResourceManager()


def _close_reusable_amgx_solvers() -> None:
    """Close every registered reusable AMGX solver and prune the registry."""
    live = []
    for solver in list(_AMGX_REUSABLE_SOLVERS):
        if solver.closed:
            continue
        solver.close()
        live.append(solver)
    _AMGX_REUSABLE_SOLVERS[:] = [solver for solver in live if not solver.closed]

def _amgx_config_for_solve(*, config=None, tolerance: float = 1e-13, maxiter: int | None = None, verbose: bool | int = 0, fixed_amg_cycles: int | None = None):
    """Build an AMGX solver configuration with normalized controls and diagnostics."""
    from hdgfem.backends.cupy import default_pyamgx_config

    if config is None:
        amgx_config = default_pyamgx_config(tolerance=tolerance, maxiter=maxiter, verbose=verbose)
    else:
        amgx_config = copy.deepcopy(config)
        solver_config = amgx_config.setdefault("solver", {})
        if "tolerance" not in solver_config:
            solver_config["tolerance"] = float(tolerance)
        if maxiter is not None:
            solver_config["max_iters"] = int(maxiter)

    verbose_level = 1 if isinstance(verbose, bool) and verbose else (0 if not verbose else int(verbose))
    solver_config = amgx_config.setdefault("solver", {})
    if fixed_amg_cycles is not None:
        if (isinstance(fixed_amg_cycles, bool) or not isinstance(fixed_amg_cycles, int)
                or fixed_amg_cycles <= 0):
            raise ValueError("fixed_amg_cycles must be a positive integer")
        if solver_config.get("solver") != "AMG":
            raise ValueError("fixed_amg_cycles requires an AMG solver")
        solver_config["max_iters"] = fixed_amg_cycles
        solver_config["monitor_residual"] = 0
        solver_config["store_res_history"] = 0
    else:
        solver_config["monitor_residual"] = 1
        solver_config.setdefault("store_res_history", 1)
    if verbose_level == 2 or verbose_level >= 4:
        solver_config["obtain_timings"] = 1
    if verbose_level >= 3:
        solver_config["print_solve_stats"] = 1
        # Honor an explicit cadence, including the transport guard's compact
        # default. The native loop always prints the last completed iteration.
        solver_config.setdefault("print_solve_stats_interval", 1)
    return amgx_config


def _amgx_relative_residual_check_rtol(config, fallback: float) -> float:
    """Return the configured relative solver-system validation tolerance."""
    solver = {} if config is None else config.get("solver", {})
    convergence = str(solver.get("convergence", "")).upper()
    if convergence.startswith("RELATIVE"):
        value = float(solver.get("tolerance", fallback))
        if np.isfinite(value) and value >= 0.0:
            return value
    return float(fallback)


def _validate_amgx_block_configuration(config, block_dim: int) -> None:
    """Reject known scalar-only solvers before invoking native block setup."""
    if block_dim <= 1:
        return

    def check(node, path: str) -> None:
        solver_name = node.get("solver") if isinstance(node, dict) else node
        if isinstance(solver_name, str) and solver_name.upper() == "CHEBYSHEV_POLY":
            raise ValueError(
                f"AMGX {path} uses scalar-only CHEBYSHEV_POLY with "
                f"block size {block_dim}; use CHEBYSHEV with a "
                "block-compatible preconditioner or scalarize the matrix"
            )
        if isinstance(node, dict):
            # These solvers act on the same matrix. A coarse_solver may act
            # on scalar-expanded levels, so its block size is not known here.
            for key in ("solver", "preconditioner", "smoother"):
                if key in node:
                    check(node[key], f"{path}.{key}")

    check(config, "config")


class PyAMGXCsrDeviceSolver:
    """Reusable PyAMGX CSR solver for a fixed device-resident matrix."""

    def __init__(
            self,
            *,
            config=None,
            tolerance: float = 1e-13,
            maxiter: int | None = None,
            verbose: bool | int = 0,
            reusable: bool = False,
            fixed_amg_cycles: int | None = None,
    ):
        """Initialize a solve, or an explicitly fixed number of AMG cycles.

        Fixed cycles disable tolerance-based early exit for preconditioner use.
        Ordinary solver calls retain convergence monitoring; history is optional.
        """
        self.cp = require_cupy()
        self.pyamgx = require_pyamgx()
        self.config_dict = _amgx_config_for_solve(config=config, tolerance=tolerance, maxiter=maxiter, verbose=verbose, fixed_amg_cycles=fixed_amg_cycles)
        self.verbose_level = (
            1
            if isinstance(verbose, bool) and verbose
            else (0 if not verbose else int(verbose))
        )
        self.cfg = self.rsrc = self.mat = self.vec_b = self.vec_x = self.solver = None
        self.shape = None
        self.size = None
        self.block_rows = None
        self.block_dim = 1
        self.last_matrix_upload_elapsed_seconds = 0.0
        self.last_solver_setup_elapsed_seconds = 0.0
        self.last_coefficients_replace_elapsed_seconds = 0.0
        self.setup_count = 0
        self.coefficients_replace_count = 0
        self.is_setup = False
        self.closed = False
        self.reusable = bool(reusable)
        self._shared_resources_acquired = False
        failure_phase = "resource acquisition"
        try:
            self.rsrc = _AMGX_SHARED_RESOURCES.acquire(self.pyamgx, self.config_dict)
            self._shared_resources_acquired = True
            failure_phase = "configuration creation"
            self.cfg = self.pyamgx.Config()
            self.cfg.create_from_dict(self.config_dict)
            failure_phase = "solver-object creation"
            self.mat = self.pyamgx.Matrix()
            self.mat.create(self.rsrc, mode=AMGX_MODE)
            self.vec_b = self.pyamgx.Vector()
            self.vec_b.create(self.rsrc, mode=AMGX_MODE)
            self.vec_x = self.pyamgx.Vector()
            self.vec_x.create(self.rsrc, mode=AMGX_MODE)
            self.solver = self.pyamgx.Solver()
            self.solver.create(self.rsrc, self.cfg, mode=AMGX_MODE)
            if self.reusable:
                _AMGX_REUSABLE_SOLVERS.append(self)
        except Exception as exc:
            capacity_error = _as_amgx_capacity_error(
                exc, phase=failure_phase, cp=self.cp, pyamgx=self.pyamgx
            )
            self.close(suppress_errors=True)
            if capacity_error is not None and capacity_error is not exc:
                raise capacity_error from exc
            raise

    def setup(self, matrix) -> float:
        """Upload and set up a fixed device CSR or face-BSR matrix in AMGX."""
        if self.closed:
            raise RuntimeError("cannot set up a closed PyAMGXCsrDeviceSolver")
        setup_start = time.perf_counter()
        self.last_matrix_upload_elapsed_seconds = 0.0
        self.last_solver_setup_elapsed_seconds = 0.0
        failure_phase = "matrix upload"
        try:
            block_dim = int(getattr(matrix, "block_size", 1))
            if block_dim < 1 or matrix.shape[0] % block_dim or matrix.shape[1] % block_dim:
                raise ValueError(
                    f"matrix shape {matrix.shape} is incompatible with block size {block_dim}"
                )
            _validate_amgx_block_configuration(self.config_dict, block_dim)
            block_shape = (
                int(matrix.shape[0] // block_dim),
                int(matrix.shape[1] // block_dim),
            )
            matrix_upload_start = time.perf_counter()
            audit_arrays('amgx-matrix-upload', matrix)
            self.mat.upload(
                matrix.indptr,
                matrix.indices,
                matrix.data,
                block_dims=[block_dim, block_dim],
                shape=block_shape,
            )
            self.cp.cuda.get_current_stream().synchronize()
            self.last_matrix_upload_elapsed_seconds = time.perf_counter() - matrix_upload_start
            failure_phase = "solver setup"
            solver_setup_start = time.perf_counter()
            self.solver.setup(self.mat)
            failure_phase = "setup synchronization"
            self.cp.cuda.get_current_stream().synchronize()
            self.last_solver_setup_elapsed_seconds = time.perf_counter() - solver_setup_start
            self.shape = tuple(matrix.shape)
            self.size = int(matrix.shape[0])
            self.block_rows = int(block_shape[0])
            self.block_dim = block_dim
            self.is_setup = True
            self.setup_count += 1
        except Exception as exc:
            capacity_error = _as_amgx_capacity_error(
                exc, phase=failure_phase, cp=self.cp, pyamgx=self.pyamgx
            )
            self.close(suppress_errors=True)
            if capacity_error is not None and capacity_error is not exc:
                raise capacity_error from exc
            raise
        return time.perf_counter() - setup_start

    def replace_coefficients(self, matrix) -> float:
        """Replace fixed-pattern coefficients while retaining solver setup state."""
        if self.closed or not self.is_setup:
            raise RuntimeError(
                "PyAMGXCsrDeviceSolver must be set up before coefficient replacement"
            )
        block_dim = int(getattr(matrix, "block_size", 1))
        if tuple(matrix.shape) != self.shape or block_dim != self.block_dim:
            raise ValueError(
                "replacement matrix shape/block size does not match the cached AMGX matrix"
            )
        expected_values = int(self.mat.get_nnz()) * block_dim * block_dim
        if int(matrix.data.size) != expected_values:
            raise ValueError(
                "replacement matrix nonzero count does not match the cached AMGX pattern"
            )
        started = time.perf_counter()
        try:
            self.mat.replace_coefficients(matrix.data)
            self.cp.cuda.get_current_stream().synchronize()
        except Exception as exc:
            capacity_error = _as_amgx_capacity_error(
                exc, phase="coefficient replacement", cp=self.cp, pyamgx=self.pyamgx
            )
            self.close(suppress_errors=True)
            if capacity_error is not None and capacity_error is not exc:
                raise capacity_error from exc
            raise
        elapsed = time.perf_counter() - started
        self.last_coefficients_replace_elapsed_seconds = elapsed
        self.last_matrix_upload_elapsed_seconds = elapsed
        self.last_solver_setup_elapsed_seconds = 0.0
        self.coefficients_replace_count += 1
        return elapsed

    def solve(self, rhs, *, initial_guess=None):
        """Solve the configured AMGX system for one device RHS."""
        if self.closed or not self.is_setup:
            raise RuntimeError("PyAMGXCsrDeviceSolver must be set up before solve()")
        if tuple(rhs.shape) != (self.size,):
            raise ValueError(f"rhs must have shape ({self.size},); got {rhs.shape}")
        info = {"amgx_status": "unknown", "amgx_iterations": None, "residual_history": ()}
        if self.verbose_level == 2 or self.verbose_level >= 4:
            print(format_amgx_configuration(self.config_dict), flush=True)
        solve_start = time.perf_counter()
        failure_phase = "solution allocation"
        try:
            if initial_guess is None:
                x = self.cp.zeros_like(rhs)
                zero_initial_guess = True
            else:
                x = self.cp.asarray(initial_guess, dtype=REAL_DTYPE).copy()
                if tuple(x.shape) != tuple(rhs.shape):
                    raise ValueError(f"initial_guess must have shape {rhs.shape}; got {x.shape}")
                zero_initial_guess = False
            failure_phase = "vector upload"
            audit_arrays('amgx-vector-upload', rhs, x)
            self.vec_b.upload_raw(rhs.data.ptr, self.block_rows, self.block_dim)
            self.vec_x.upload_raw(x.data.ptr, self.block_rows, self.block_dim)
            failure_phase = "solver iteration"
            self.solver.solve(self.vec_b, self.vec_x, zero_initial_guess=zero_initial_guess)
            failure_phase = "solution download"
            self.vec_x.download_raw(x.data.ptr)
            self.cp.cuda.get_current_stream().synchronize()
        except Exception as exc:
            capacity_error = _as_amgx_capacity_error(
                exc, phase=failure_phase, cp=self.cp, pyamgx=self.pyamgx
            )
            if (
                failure_phase == "solution allocation"
                and capacity_error is None
                and isinstance(exc, (TypeError, ValueError))
            ):
                # Malformed guesses do not invalidate a healthy hierarchy.
                raise
            self.close(suppress_errors=True)
            if capacity_error is not None and capacity_error is not exc:
                raise capacity_error from exc
            raise
        solve_elapsed = time.perf_counter() - solve_start
        try:
            info["amgx_status"] = str(self.solver.status)
        except Exception:
            pass
        try:
            info["amgx_iterations"] = int(self.solver.iterations_number)
        except Exception:
            pass
        store_residual_history = bool(
            self.config_dict.get("solver", {}).get("store_res_history", 0)
        )
        if store_residual_history and info["amgx_iterations"] is not None:
            history = []
            first = max(0, info["amgx_iterations"] - 63)
            for iteration in range(first, info["amgx_iterations"] + 1):
                try:
                    history.append(float(self.solver.get_residual(iteration)))
                except Exception:
                    history = []
                    break
            info["residual_history"] = tuple(history)
        info["amgx_setup_elapsed_seconds"] = 0.0
        info["amgx_solve_elapsed_seconds"] = solve_elapsed
        return x, info

    def close(self, *, suppress_errors: bool = False) -> None:
        """Destroy owned AMGX objects and release the shared resource handle."""
        if self.closed:
            return
        first_error = None
        try:
            destroy_amgx_objects((self.solver, self.vec_x, self.vec_b, self.mat, self.cfg))
        except Exception as exc:
            first_error = exc
        if self._shared_resources_acquired:
            try:
                _AMGX_SHARED_RESOURCES.release()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            self._shared_resources_acquired = False
        self.solver = self.mat = self.vec_x = self.vec_b = self.rsrc = self.cfg = None
        self._hdgfem_fixed_operator = None
        self.is_setup = False
        self.closed = True
        if first_error is not None and not suppress_errors:
            raise first_error

    def __del__(self):
        """Best-effort cleanup for an unclosed AMGX solver instance."""
        try:
            self.close(suppress_errors=True)
        except Exception:
            pass


def _pyamgx_solve_csr_device(
        matrix,
        rhs,
        *,
        initial_guess=None,
        config=None,
        tolerance: float = 1e-13,
        maxiter: int | None = None,
        verbose: bool | int = 0,
):
    """Solve a device CSR system with PyAMGX without staging through host CSR."""
    solver = PyAMGXCsrDeviceSolver(config=config, tolerance=tolerance, maxiter=maxiter, verbose=verbose)
    try:
        setup_elapsed = solver.setup(matrix)
        x, info = solver.solve(rhs, initial_guess=initial_guess)
    finally:
        solver.close()
    info["amgx_setup_elapsed_seconds"] = setup_elapsed
    info["amgx_matrix_upload_elapsed_seconds"] = solver.last_matrix_upload_elapsed_seconds
    info["amgx_solver_setup_elapsed_seconds"] = solver.last_solver_setup_elapsed_seconds
    return x, info


def _normalize_device_scale_mode(value) -> str:
    """Normalize public boolean/string scaling controls for device solves."""
    if isinstance(value, (bool, np.bool_)):
        return "left" if bool(value) else "none"
    normalized = str(value).replace("_", "-").lower()
    aliases = {"on": "left", "off": "none", "true": "left", "false": "none"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"none", "left", "symmetric"}:
        raise ValueError("scale_system must be bool or one of 'none', 'left', 'symmetric'")
    return normalized


def _solve_reduced_system_amgx_device_once(
    assembly: CudaAdvectionAssembly,
    *,
    config=None,
    tolerance: float = 1e-13,
    check_rtol: float | None = None,
    solver_check_rtol: float | None = None,
    atol: float = 0.0,
    maxiter: int | None = None,
    initial_guess=None,
    reusable_solver: PyAMGXCsrDeviceSolver | None = None,
    scale_system: bool | str = True,
    scalarize_bsr: bool = False,
    replace_reusable_coefficients: bool = False,
    raise_on_nonconvergence: bool = True,
    materialize_host_solution: bool = True,
    verbose: bool | int = 0,
):
    """Solve a CUDA-resident reduced trace system with device CSR and PyAMGX."""
    cp = require_cupy()
    sparse = require_cupyx_sparse()
    total_start = time.perf_counter()
    system_size = int(assembly.rhs.size)
    solve_tolerance = float(tolerance)
    result_check_rtol = solve_tolerance if check_rtol is None else float(check_rtol)
    solver_result_check_rtol = (
        result_check_rtol
        if solver_check_rtol is None
        else float(solver_check_rtol)
    )
    if not np.isfinite(solve_tolerance) or solve_tolerance < 0.0:
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance}")
    if not np.isfinite(result_check_rtol) or result_check_rtol < 0.0:
        raise ValueError(f"check_rtol must be finite and non-negative, got {check_rtol}")
    if not np.isfinite(solver_result_check_rtol) or solver_result_check_rtol < 0.0:
        raise ValueError(
            "solver_check_rtol must be finite and non-negative, got "
            f"{solver_check_rtol}"
        )
    if not np.isfinite(atol) or float(atol) < 0.0:
        raise ValueError(f"atol must be finite and non-negative, got {atol}")
    if maxiter is not None and int(maxiter) <= 0:
        raise ValueError(f"maxiter must be positive when provided, got {maxiter}")
    if not bool(cp.all(cp.isfinite(assembly.data)).get()):
        raise ValueError("device matrix contains non-finite values")
    if not bool(cp.all(cp.isfinite(assembly.rhs)).get()):
        raise ValueError("device rhs contains non-finite values")
    if initial_guess is not None and not bool(cp.all(cp.isfinite(cp.asarray(initial_guess))).get()):
        raise ValueError("initial_guess contains non-finite values")

    matrix_start = time.perf_counter()
    matrix_format = getattr(assembly, "matrix_format", "coo")
    if matrix_format == "csr":
        if assembly.indptr is None or assembly.indices is None:
            raise RuntimeError("CSR assembly is missing indptr/indices")
        matrix = _DeviceCsrMatrixView(
            data=assembly.data,
            indices=assembly.indices.astype(cp.int32, copy=False),
            indptr=assembly.indptr.astype(cp.int32, copy=False),
            shape=(system_size, system_size),
        )
    elif matrix_format == "bsr":
        if assembly.indptr is None or assembly.indices is None:
            raise RuntimeError("BSR assembly is missing block indptr/indices")
        if assembly.data.ndim != 3 or assembly.data.shape[1] != assembly.data.shape[2]:
            raise RuntimeError(
                f"BSR data must have shape (nnzb, block_size, block_size); got {assembly.data.shape}"
            )
        block_dim = int(assembly.data.shape[1])
        if system_size % block_dim:
            raise RuntimeError(
                f"system size {system_size} is not divisible by BSR block size {block_dim}"
            )
        matrix = _DeviceBsrMatrixView(
            data=assembly.data,
            indices=assembly.indices.astype(cp.int32, copy=False),
            indptr=assembly.indptr.astype(cp.int32, copy=False),
            shape=(system_size, system_size),
            block_size=block_dim,
        )
    else:
        matrix = sparse.coo_matrix(
            (assembly.data, (assembly.rows.astype(cp.int32), assembly.cols.astype(cp.int32))),
            shape=(system_size, system_size),
            dtype=REAL_DTYPE,
        ).tocsr()
        matrix.sum_duplicates()
        if matrix.indices.dtype != cp.int32 or matrix.indptr.dtype != cp.int32:
            matrix = sparse.csr_matrix(
                (
                matrix.data,
                matrix.indices.astype(cp.int32, copy=False),
                matrix.indptr.astype(cp.int32, copy=False),
            ),
                shape=matrix.shape,
                dtype=REAL_DTYPE,
            )
    bsr_scalarized = bool(scalarize_bsr and isinstance(matrix, _DeviceBsrMatrixView))
    if bsr_scalarized:
        matrix = _scalarize_device_bsr_matrix(matrix, sparse, cp)
    cp.cuda.get_current_stream().synchronize()
    matrix_elapsed = time.perf_counter() - matrix_start

    physical_rhs = assembly.rhs
    scale_mode = _normalize_device_scale_mode(scale_system)
    if isinstance(matrix, _DeviceBsrMatrixView) and scale_mode == "symmetric":
        raise ValueError(
            "device BSR solves do not support symmetric scaling; "
            "use scale_system='left' or False"
        )
    row_diagonal = None
    inverse_sqrt_diagonal = None
    scale_start = time.perf_counter()
    if scale_mode == "left":
        solve_matrix = matrix
        solve_rhs = physical_rhs.copy()
        if isinstance(solve_matrix, _DeviceBsrMatrixView):
            row_diagonal = _diagonal_scale_bsr_rows_in_place(solve_matrix, solve_rhs)
        else:
            row_diagonal = _diagonal_scale_csr_rows_in_place(solve_matrix, solve_rhs)
        solve_initial_guess = initial_guess
    elif scale_mode == "symmetric":
        solve_matrix = matrix
        solve_rhs = physical_rhs.copy()
        inverse_sqrt_diagonal = symmetric_scale_cupy_csr_in_place(solve_matrix, solve_rhs)
        solve_initial_guess = (
            None if initial_guess is None else cp.asarray(initial_guess) / inverse_sqrt_diagonal
        )
    else:
        solve_matrix = matrix
        solve_rhs = physical_rhs
        solve_initial_guess = initial_guess
    cp.cuda.get_current_stream().synchronize()
    scale_elapsed = time.perf_counter() - scale_start
    matrix_is_scaled = row_diagonal is not None or inverse_sqrt_diagonal is not None

    def restore_scaled_matrix() -> float:
        """Restore the physical CSR coefficients after an in-place scaled solve."""
        nonlocal matrix_is_scaled
        if not matrix_is_scaled:
            return 0.0
        restore_start = time.perf_counter()
        if isinstance(solve_matrix, _DeviceBsrMatrixView):
            _restore_left_scaled_bsr_rows_in_place(solve_matrix, row_diagonal)
        else:
            _restore_scaled_csr_rows_in_place(
                solve_matrix,
                row_diagonal=row_diagonal,
                inverse_sqrt_diagonal=inverse_sqrt_diagonal,
            )
        cp.cuda.get_current_stream().synchronize()
        matrix_is_scaled = False
        return time.perf_counter() - restore_start

    try:
        amgx_call_start = time.perf_counter()
        if reusable_solver is None:
            x_cp, amgx_info = _pyamgx_solve_csr_device(
                solve_matrix,
                solve_rhs,
                initial_guess=solve_initial_guess,
                config=config,
                tolerance=solve_tolerance,
                maxiter=maxiter,
                verbose=verbose,
            )
        else:
            setup_elapsed = 0.0
            matrix_upload_elapsed = 0.0
            solver_setup_elapsed = 0.0
            preconditioner_reused = bool(reusable_solver.is_setup)
            if not reusable_solver.is_setup:
                setup_elapsed = reusable_solver.setup(solve_matrix)
                matrix_upload_elapsed = reusable_solver.last_matrix_upload_elapsed_seconds
                solver_setup_elapsed = reusable_solver.last_solver_setup_elapsed_seconds
            elif replace_reusable_coefficients:
                matrix_upload_elapsed = reusable_solver.replace_coefficients(solve_matrix)
                preconditioner_reused = True
            x_cp, amgx_info = reusable_solver.solve(solve_rhs, initial_guess=solve_initial_guess)
            amgx_info["amgx_setup_elapsed_seconds"] = setup_elapsed
            amgx_info["amgx_matrix_upload_elapsed_seconds"] = matrix_upload_elapsed
            amgx_info["amgx_solver_setup_elapsed_seconds"] = solver_setup_elapsed
            amgx_info["amgx_preconditioner_reused"] = preconditioner_reused
        audit_arrays('amgx-solution', x_cp)
        solver_x_cp = x_cp
        if inverse_sqrt_diagonal is not None:
            x_cp = inverse_sqrt_diagonal * solver_x_cp
        amgx_call_elapsed = time.perf_counter() - amgx_call_start
    except BaseException as exc:
        try:
            restore_scaled_matrix()
        except Exception as restore_exc:
            if not isinstance(exc, LinearSolveCapacityError):
                raise
            # A depleted or failed CUDA runtime can also reject restoration.
            # Preserve the terminal native phase and its pre-cleanup counters.
            exc.matrix_restore_error = f"{type(restore_exc).__name__}: {restore_exc}"
        raise

    try:
        finite_start = time.perf_counter()
        solution_is_finite = bool(cp.all(cp.isfinite(x_cp)).get())
        finite_elapsed = time.perf_counter() - finite_start

        solver_residual_start = time.perf_counter()
        solver_residual = (
            _device_compressed_matvec(solve_matrix, solver_x_cp, sparse, cp)
            - solve_rhs
        )
        solver_residual_norm, solver_rhs_norm, solver_relative, solver_target = _residual_stats_cp(
            solver_residual,
            solve_rhs,
            rtol=solver_result_check_rtol,
            atol=atol,
        )
        solver_residual_elapsed = time.perf_counter() - solver_residual_start

        # Check b-A*x after restoring the physical coefficients. Undoing the
        # residual scaling algebraically can hide cancellation/roundoff errors.
        unscale_elapsed = restore_scaled_matrix()
        physical_residual_start = time.perf_counter()
        if scale_mode == "none":
            # The solver and physical systems are identical. Reuse the SpMV
            # and norms, but retain the independent physical acceptance target.
            physical_residual_norm = solver_residual_norm
            physical_rhs_norm = solver_rhs_norm
            physical_relative = solver_relative
            physical_target = max(float(result_check_rtol) * physical_rhs_norm, float(atol))
        else:
            physical_residual = _device_compressed_matvec(matrix, x_cp, sparse, cp) - physical_rhs
            physical_residual_norm, physical_rhs_norm, physical_relative, physical_target = _residual_stats_cp(
                physical_residual,
                physical_rhs,
                rtol=result_check_rtol,
                atol=atol,
            )
        physical_residual_elapsed = time.perf_counter() - physical_residual_start
        validation_elapsed = finite_elapsed + solver_residual_elapsed + physical_residual_elapsed
    except BaseException:
        restore_scaled_matrix()
        raise
    native_status = str(amgx_info.get("amgx_status", "unknown"))
    normalized_status = native_status.lower().replace("-", "_").replace(" ", "_")
    backend_success = normalized_status == "unknown" or not any(
        marker in normalized_status
        for marker in ("fail", "diverg", "not_converged", "notconverged")
    )
    total_elapsed = time.perf_counter() - total_start
    verbose_level = 1 if isinstance(verbose, bool) and verbose else (0 if not verbose else int(verbose))
    if verbose_level == 2 or verbose_level >= 4:
        print("  PyAMGX device solve timings:", flush=True)
        print(f"    matrix view: {matrix_elapsed:.5f}s", flush=True)
        print(f"    row scaling: {scale_elapsed:.5f}s", flush=True)
        print(f"    matrix unscale: {unscale_elapsed:.5f}s", flush=True)
        print(f"    setup: {amgx_info['amgx_setup_elapsed_seconds']:.5f}s", flush=True)
        print(
            f"      matrix upload: {amgx_info['amgx_matrix_upload_elapsed_seconds']:.5f}s",
            flush=True,
        )
        print(f"      solver setup: {amgx_info['amgx_solver_setup_elapsed_seconds']:.5f}s", flush=True)
        print(f"    iterate: {amgx_info['amgx_solve_elapsed_seconds']:.5f}s", flush=True)
        print(f"    amgx call total: {amgx_call_elapsed:.5f}s", flush=True)
        print(f"    validation: {validation_elapsed:.5f}s", flush=True)
        print(f"    solver relative residual: {solver_relative:.3e}", flush=True)
    elif verbose_level:
        print(
            f"  PyAMGX: matrix={matrix_elapsed:.5f}s scale={scale_elapsed:.5f}s "
            f"setup={amgx_info['amgx_setup_elapsed_seconds']:.5f}s "
            f"solve={amgx_info['amgx_solve_elapsed_seconds']:.5f}s rel={solver_relative:.3e}",
            flush=True,
        )
    result = SolveResult(
        x=cp.asnumpy(x_cp) if materialize_host_solution else None,
        residual_norm=solver_residual_norm,
        info=0 if backend_success else 1,
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
    result.device_scale_mode = scale_mode
    result.amgx_csr_elapsed_seconds = matrix_elapsed
    result.amgx_bsr_scalarized = bsr_scalarized
    result.amgx_preconditioner_reused = bool(
        amgx_info.get("amgx_preconditioner_reused", False)
    )
    result.amgx_matrix_unscale_elapsed_seconds = unscale_elapsed
    result.amgx_setup_elapsed_seconds = amgx_info["amgx_setup_elapsed_seconds"]
    result.amgx_matrix_upload_elapsed_seconds = amgx_info["amgx_matrix_upload_elapsed_seconds"]
    result.amgx_solver_setup_elapsed_seconds = amgx_info["amgx_solver_setup_elapsed_seconds"]
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
    result.solve_accounted_elapsed_seconds = (
        matrix_elapsed + scale_elapsed + amgx_call_elapsed + validation_elapsed + unscale_elapsed
    )
    result.solve_unaccounted_elapsed_seconds = max(0.0, total_elapsed - result.solve_accounted_elapsed_seconds)
    result.solve_global_overhead_elapsed_seconds = result.solve_unaccounted_elapsed_seconds
    result = finalize_solve_result(
        result,
        backend="pyamgx-device",
        backend_info=native_status,
        backend_success=backend_success,
        solution_is_finite=solution_is_finite,
        residual_history=amgx_info.get("residual_history"),
        raise_on_nonconvergence=raise_on_nonconvergence,
    )
    return result, x_cp


def _solve_reduced_system_cusolver_qr_device_once(
    assembly: CudaAdvectionAssembly,
    *,
    tolerance: float = 1e-13,
    check_rtol: float | None = None,
    atol: float = 0.0,
    materialize_host_solution: bool = True,
):
    """Last-resort device sparse QR with an explicit physical residual check.

    Uses CuPy's cuSOLVER sparse QR binding, not a host sparse solve. BSR is
    expanded on device. The direct solver receives its own canonical CSR copy
    and the original, unscaled system; it never reuses a failed iterate.
    """
    from cupyx.cusolver import csrlsvqr

    cp, sparse = require_cupy(), require_cupyx_sparse()
    started = time.perf_counter()
    rtol = float(tolerance if check_rtol is None else check_rtol)
    if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
        raise ValueError("direct residual tolerances must be finite and nonnegative")
    if not bool(cp.all(cp.isfinite(assembly.data)).get()) or not bool(cp.all(cp.isfinite(assembly.rhs)).get()):
        raise ValueError("device direct system contains non-finite values")
    matrix = _assembly_device_csr_matrix(assembly, cp, sparse)
    scalarized = isinstance(matrix, _DeviceBsrMatrixView)
    if max(int(matrix.shape[0]), int(matrix.data.size)) > np.iinfo(np.int32).max:
        raise ValueError("cuSOLVER sparse QR requires matrix size and scalar nnz to fit int32")
    csr = (_scalarize_device_bsr_matrix(matrix, sparse, cp) if scalarized
           else _as_cupyx_csr_matrix(matrix, sparse, cp).copy())
    csr.sum_duplicates()
    csr.sort_indices()
    rhs = cp.ascontiguousarray(assembly.rhs).copy()
    cp.cuda.get_current_stream().synchronize()
    setup_elapsed = time.perf_counter() - started
    solve_start = time.perf_counter()
    # tol is a pivot-singularity threshold, not the residual tolerance. Reject
    # a singularity warning as well as explicit backend failures.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="(?i).*singular.*")
        solution = csrlsvqr(csr, rhs, tol=0.0, reorder=1)
    cp.cuda.get_current_stream().synchronize()
    solve_elapsed = time.perf_counter() - solve_start
    residual = _device_compressed_matvec(matrix, solution, sparse, cp) - assembly.rhs
    norm, rhs_norm, relative, target = _residual_stats_cp(residual, assembly.rhs, rtol=rtol, atol=atol)
    finite = bool(cp.all(cp.isfinite(solution)).get())
    result = SolveResult(
        x=cp.asnumpy(solution) if materialize_host_solution else None,
        residual_norm=norm, info=0, preconditioner=None,
        total_elapsed_seconds=time.perf_counter() - started,
        preconditioner_elapsed_seconds=setup_elapsed, solve_elapsed_seconds=solve_elapsed,
        rhs_norm=rhs_norm, relative_residual_norm=relative, residual_target=target,
        solver_residual_norm=norm, solver_rhs_norm=rhs_norm,
        solver_relative_residual_norm=relative, solver_residual_target=target,
        physical_residual_norm=norm, physical_rhs_norm=rhs_norm,
        physical_relative_residual_norm=relative, physical_residual_target=target,
        rtol=rtol, atol=atol,
    )
    result.device_scale_mode = "none"
    result.amgx_bsr_scalarized = scalarized
    result.cupyx_solver = "cusolver-qr-device"
    return finalize_solve_result(
        result, backend="cusolver-qr-device", backend_info=0,
        backend_success=True, solution_is_finite=finite, raise_on_nonconvergence=False,
    ), solution


def _device_transport_matrix_diagnostics(assembly) -> dict[str, float | int]:
    """Cheap row-scale diagnostics on failure; these are not condition estimates."""
    cp, sparse = require_cupy(), require_cupyx_sparse()
    matrix = _assembly_device_csr_matrix(assembly, cp, sparse)
    if isinstance(matrix, (_DeviceBsrMatrixView, _DeviceCsrMatrixView)):
        absolute_matrix = replace(matrix, data=cp.abs(matrix.data))
    else:
        absolute_matrix = matrix.copy()
        absolute_matrix.data = cp.abs(absolute_matrix.data)
    row_l1 = _device_compressed_matvec(
        absolute_matrix, cp.ones(matrix.shape[1], dtype=REAL_DTYPE), sparse, cp,
    )
    zero_rows = cp.count_nonzero(row_l1 == 0)
    values = cp.asnumpy(cp.stack((cp.min(row_l1), cp.max(row_l1), zero_rows)))
    return {"matrix_size": int(matrix.shape[0]), "matrix_scalar_stored_entries": int(matrix.data.size),
            "matrix_row_l1_min": float(values[0]), "matrix_row_l1_max": float(values[1]),
            "matrix_zero_rows": int(values[2])}


def solve_reduced_system_amgx_device(
    assembly: CudaAdvectionAssembly,
    *,
    config=None,
    retry_attempts=None,
    tolerance: float = 1e-13,
    check_rtol: float | None = None,
    atol: float = 0.0,
    maxiter: int | None = None,
    initial_guess=None,
    reusable_solver: PyAMGXCsrDeviceSolver | None = None,
    retry_solver_cache: dict[Any, PyAMGXCsrDeviceSolver] | None = None,
    retry_seed_solution=None,
    retry_seed_label: str | None = None,
    cache_fixed_operator: bool = False,
    reuse_primary_preconditioner: bool = False,
    scale_system: bool | str = True,
    raise_on_nonconvergence: bool = True,
    materialize_host_solution: bool = True,
    verbose: bool | int = 0,
):
    """Solve one assembled device system, optionally retrying without reassembly.

    ``retry_seed_solution`` initializes the wrapper's single best-candidate
    slot from a failed upstream solver.  Its physical residual is evaluated
    here, and no vector history is retained.

    With ``reusable_solver`` and ``reuse_primary_preconditioner=True``, an
    already set-up solver receives the new matrix through
    ``replace_coefficients`` and keeps its previous setup (a possibly stale
    preconditioner); callers force a fresh setup by clearing ``is_setup``.
    """
    def fixed_solver(key, solver_config):
        """Keep native allocations; refresh numeric setup only for a new matrix."""
        if retry_solver_cache is None:
            raise ValueError("fixed-operator AMGX reuse requires an owning solver cache")
        key = ("fixed-operator", key)
        active = retry_solver_cache.get(key)
        if active is None or active.closed:
            active = PyAMGXCsrDeviceSolver(config=solver_config, tolerance=tolerance,
                                           maxiter=maxiter, verbose=verbose, reusable=True)
            retry_solver_cache[key] = active
        if getattr(active, "_hdgfem_fixed_operator", None) is not assembly.data:
            active.is_setup = False
            active._hdgfem_fixed_operator = assembly.data
        return active

    if cache_fixed_operator:
        if reusable_solver is not None:
            raise ValueError("supply either reusable_solver or cache_fixed_operator")
        reusable_solver = fixed_solver("primary", config)
    configured_retries = tuple(retry_attempts or ())
    if retry_seed_solution is not None and not raise_on_nonconvergence:
        raise ValueError("retry_seed_solution requires raise_on_nonconvergence=True")
    max_attempts = 8
    if 1 + len(configured_retries) > max_attempts:
        raise ValueError(f"AMGX solve supports at most {max_attempts} bounded attempts")
    attempts = [
        {
            "label": "primary-stage-guess" if initial_guess is not None else "primary-zero",
            "backend": "amgx",
            "config": config,
            "initial_guess": initial_guess,
            "scale_system": _normalize_device_scale_mode(scale_system),
            "reusable_solver": reusable_solver,
            "use_best_solution": False,
            "residual_correction": False,
            "scalarize_bsr": False,
            "reuse_preconditioner": False,
            "solver_cache_key": None,
        }
    ]
    for index, retry in enumerate(configured_retries, start=1):
        backend = str(retry.get("backend", "amgx"))
        if backend not in {"amgx", "cusolver-qr"}:
            raise ValueError(f"unsupported device retry backend {backend!r}")
        if backend == "cusolver-qr" and (
            index != len(configured_retries)
            or retry.get("residual_correction", False)
            or retry.get("reuse_preconditioner", False)
            or _normalize_device_scale_mode(retry.get("scale_system", False)) != "none"
        ):
            raise ValueError("cusolver-qr must be the final, unscaled direct retry without preconditioner reuse or correction")
        use_initial_guess = bool(retry.get("use_initial_guess", True))
        reuse_primary_solver = bool(retry.get("reuse_primary_solver", False))
        if reuse_primary_solver and (
            backend != "amgx"
            or retry.get("scalarize_bsr", False)
            or retry.get("reuse_preconditioner", False)
            or retry.get("residual_correction", False)
        ):
            raise ValueError(
                "reuse_primary_solver requires a nonscalarized AMGX retry "
                "without correction or separate preconditioner reuse"
            )
        retry_config = retry.get("config", config)
        retry_scale_mode = _normalize_device_scale_mode(
            retry.get("scale_system", False if backend == "cusolver-qr" else scale_system)
        )
        if reuse_primary_solver and (
            retry_config != config
            or retry_scale_mode != _normalize_device_scale_mode(scale_system)
        ):
            raise ValueError(
                "reuse_primary_solver requires the primary configuration and scaling"
            )
        attempts.append(
            {
                "label": str(retry.get("label", f"retry-{index}")),
                "backend": backend,
                "config": retry_config,
                "initial_guess": initial_guess if use_initial_guess else None,
                "scale_system": retry_scale_mode,
                "reusable_solver": reusable_solver if reuse_primary_solver else None,
                "reuse_primary_solver": reuse_primary_solver,
                "use_best_solution": bool(retry.get("use_best_solution", False)),
                "residual_correction": bool(retry.get("residual_correction", False)),
                "scalarize_bsr": bool(retry.get("scalarize_bsr", False)),
                "reuse_preconditioner": bool(
                    retry.get("reuse_preconditioner", False)
                ),
                "solver_cache_key": retry.get("solver_cache_key"),
            }
        )

    if len(attempts) == 1 and retry_seed_solution is None:
        return _solve_reduced_system_amgx_device_once(
            assembly,
            config=config,
            tolerance=tolerance,
            check_rtol=check_rtol,
            solver_check_rtol=_amgx_relative_residual_check_rtol(
                config, tolerance if check_rtol is None else check_rtol
            ),
            atol=atol,
            maxiter=maxiter,
            initial_guess=initial_guess,
            reusable_solver=reusable_solver,
            scale_system=scale_system,
            replace_reusable_coefficients=bool(
                reuse_primary_preconditioner and reusable_solver is not None
                and not reusable_solver.closed and reusable_solver.is_setup),
            raise_on_nonconvergence=raise_on_nonconvergence,
            materialize_host_solution=materialize_host_solution,
            verbose=verbose,
        )

    cp = require_cupy()
    retry_wrapper_start = time.perf_counter()
    attempt_log = []
    last_result = None
    last_error = None
    best_result = None
    best_solution = None
    last_solution = None
    best_score = float("inf")
    seed_metrics = None
    verbose_level = 1 if isinstance(verbose, bool) and verbose else (0 if not verbose else int(verbose))
    if retry_seed_solution is not None:
        sparse = require_cupyx_sparse()
        seed = cp.asarray(retry_seed_solution, dtype=REAL_DTYPE)
        if seed.ndim != 1 or int(seed.size) != int(assembly.rhs.size):
            raise ValueError(
                "retry_seed_solution must have the reduced-system shape "
                f"({int(assembly.rhs.size)},)"
            )
        if bool(cp.all(cp.isfinite(seed)).get()):
            seed_matrix = _assembly_device_csr_matrix(assembly, cp, sparse)
            seed_residual = (
                _device_compressed_matvec(seed_matrix, seed, sparse, cp)
                - assembly.rhs
            )
            seed_check_rtol = (
                float(tolerance) if check_rtol is None else float(check_rtol)
            )
            residual_norm, rhs_norm, relative_residual, residual_target = (
                _residual_stats_cp(
                    seed_residual, assembly.rhs,
                    rtol=seed_check_rtol, atol=atol,
                )
            )
            seed_score = residual_norm / max(residual_target, 1.0e-300)
            if all(np.isfinite(value) for value in (
                residual_norm, rhs_norm, relative_residual, residual_target, seed_score,
            )):
                best_solution = seed.copy()
                best_score = seed_score
                seed_metrics = {
                    "label": retry_seed_label or "upstream-best",
                    "physical_residual": residual_norm,
                    "physical_rhs_norm": rhs_norm,
                    "physical_relative_residual": relative_residual,
                    "physical_target": residual_target,
                }

    def attach_seed_metrics(result) -> None:
        if result is None or seed_metrics is None:
            return
        result.amgx_retry_seed_label = seed_metrics["label"]
        result.amgx_retry_seed_physical_residual = seed_metrics["physical_residual"]
        result.amgx_retry_seed_physical_rhs_norm = seed_metrics["physical_rhs_norm"]
        result.amgx_retry_seed_physical_relative_residual = seed_metrics[
            "physical_relative_residual"
        ]
        result.amgx_retry_seed_physical_target = seed_metrics["physical_target"]

    try:
        for index, attempt in enumerate(attempts, start=1):
            try:
                attempt_initial_guess = (
                    best_solution
                    if attempt["use_best_solution"] and best_solution is not None
                    else attempt["initial_guess"]
                )
                solve_assembly = assembly
                base_solution = None
                residual_matrix = None
                if attempt["residual_correction"] and best_solution is not None:
                    sparse = require_cupyx_sparse()
                    residual_matrix = _assembly_device_csr_matrix(assembly, cp, sparse)
                    correction_rhs = assembly.rhs - _device_compressed_matvec(
                        residual_matrix, best_solution, sparse, cp
                    )
                    solve_assembly = replace(assembly, rhs=correction_rhs)
                    base_solution = best_solution
                    attempt_initial_guess = None

                attempt_solver = attempt["reusable_solver"]
                if (
                    attempt.get("reuse_primary_solver", False)
                    and attempt_solver is not None
                    and attempt_solver.closed
                ):
                    # Setup/iteration exceptions close the primary's AMGX
                    # objects. Retry with a fresh ephemeral solver in that case.
                    attempt_solver = None
                replace_coefficients = False
                if (index == 1 and reuse_primary_preconditioner and attempt_solver is not None
                        and not attempt_solver.closed):
                    replace_coefficients = bool(attempt_solver.is_setup)
                if cache_fixed_operator and attempt["backend"] == "amgx":
                    attempt_solver = fixed_solver("primary" if index == 1 else index-1, attempt["config"])
                elif (
                    attempt["reuse_preconditioner"]
                    and retry_solver_cache is not None
                ):
                    cache_key = attempt["solver_cache_key"]
                    if cache_key is None:
                        cache_key = attempt["label"]
                    attempt_solver = retry_solver_cache.get(cache_key)
                    if attempt_solver is None or attempt_solver.closed:
                        attempt_solver = PyAMGXCsrDeviceSolver(
                            config=attempt["config"],
                            tolerance=tolerance,
                            maxiter=maxiter,
                            verbose=verbose,
                            reusable=True,
                        )
                        retry_solver_cache[cache_key] = attempt_solver
                    replace_coefficients = bool(attempt_solver.is_setup)

                if attempt["backend"] == "cusolver-qr":
                    result, solution = _solve_reduced_system_cusolver_qr_device_once(
                        assembly, tolerance=tolerance, check_rtol=check_rtol, atol=atol,
                        materialize_host_solution=materialize_host_solution,
                    )
                else:
                    result, solution = _solve_reduced_system_amgx_device_once(
                        solve_assembly,
                        config=attempt["config"],
                        tolerance=tolerance,
                        check_rtol=check_rtol,
                        solver_check_rtol=_amgx_relative_residual_check_rtol(
                            attempt["config"],
                            tolerance if check_rtol is None else check_rtol,
                        ),
                        atol=atol,
                        maxiter=maxiter,
                        initial_guess=attempt_initial_guess,
                        reusable_solver=attempt_solver,
                        scale_system=attempt["scale_system"],
                        scalarize_bsr=attempt["scalarize_bsr"],
                        replace_reusable_coefficients=replace_coefficients,
                        raise_on_nonconvergence=False,
                        materialize_host_solution=materialize_host_solution,
                        verbose=verbose,
                    )
                if base_solution is not None:
                    result.amgx_correction_info = int(result.info)
                    result.amgx_correction_relative_residual_norm = result.solver_relative_residual_norm
                    solution = base_solution + solution
                    combined_residual = (
                        _device_compressed_matvec(residual_matrix, solution, sparse, cp)
                        - assembly.rhs
                    )
                    result_check_rtol = float(tolerance) if check_rtol is None else float(check_rtol)
                    residual_norm, rhs_norm, relative_residual, residual_target = _residual_stats_cp(
                        combined_residual,
                        assembly.rhs,
                        rtol=result_check_rtol,
                        atol=atol,
                    )
                    solution_is_finite = bool(cp.all(cp.isfinite(solution)).get())
                    result.x = cp.asnumpy(solution) if materialize_host_solution else None
                    result.residual_norm = residual_norm
                    result.rhs_norm = rhs_norm
                    result.relative_residual_norm = relative_residual
                    result.residual_target = residual_target
                    result.solver_residual_norm = residual_norm
                    result.solver_rhs_norm = rhs_norm
                    result.solver_relative_residual_norm = relative_residual
                    result.solver_residual_target = residual_target
                    result.physical_residual_norm = residual_norm
                    result.physical_rhs_norm = rhs_norm
                    result.physical_relative_residual_norm = relative_residual
                    result.physical_residual_target = residual_target
                    result.amgx_residual_correction = True
                    result = finalize_solve_result(
                        result,
                        backend=result.backend or "pyamgx-device",
                        backend_info=result.backend_info,
                        backend_success=result.failure_reason not in {"backend-nonconvergence", "backend-divergence"},
                        solution_is_finite=solution_is_finite,
                        residual_history=result.residual_history,
                        raise_on_nonconvergence=False,
                    )

                finite = bool(cp.all(cp.isfinite(solution)).get())
                success = finite and result.converged
                score = float(result.physical_residual_norm) / max(
                    float(result.physical_residual_target), 1.0e-300
                )
                if finite and np.isfinite(score) and score < best_score:
                    best_result = result
                    best_solution = solution.copy()
                    best_score = score
                attempt_log.append(
                    {
                        "attempt": index,
                        "label": attempt["label"],
                        "success": success,
                        "finite": finite,
                        "scale_system": attempt["scale_system"],
                        "used_best_solution": bool(
                            attempt["use_best_solution"] and attempt_initial_guess is not None
                        ),
                        "residual_correction": base_solution is not None,
                        "scalarized_bsr": bool(
                            getattr(result, "amgx_bsr_scalarized", False)
                        ),
                        "preconditioner_reused": bool(
                            getattr(result, "amgx_preconditioner_reused", False)
                        ),
                        "primary_solver_reused": bool(
                            attempt.get("reuse_primary_solver", False)
                            and attempt_solver is not None
                        ),
                        "status": result.status,
                        "failure_reason": result.failure_reason,
                        "backend_info": result.backend_info,
                        "iterations": result.iteration_count,
                        "relative_residual": result.solver_relative_residual_norm,
                        "residual": result.solver_residual_norm,
                        "target": result.solver_residual_target,
                        "backend": result.backend,
                        "physical_relative_residual": result.physical_relative_residual_norm,
                        "physical_residual": result.physical_residual_norm,
                        "physical_target": result.physical_residual_target,
                    }
                )
                last_result = result
                last_solution = solution
                if verbose_level and len(attempts) > 1:
                    print(
                        f"  AMGX attempt {index}/{len(attempts)} {attempt['label']}: "
                        f"{'accepted' if success else 'rejected'} "
                        f"rel={result.solver_relative_residual_norm:.3e} "
                        f"true_rel={result.physical_relative_residual_norm:.3e} "
                        f"status={result.status} iterations={result.iteration_count}",
                        flush=True,
                    )
                if success:
                    result.amgx_attempts = tuple(attempt_log)
                    result.amgx_attempt_count = index
                    attach_seed_metrics(result)
                    return result, solution
            except Exception as exc:
                capacity_error = _as_amgx_capacity_error(
                    exc, phase="AMGX call", cp=cp, pyamgx=None
                )
                entry = {
                    "attempt": index,
                    "label": attempt["label"],
                    "success": False,
                    "finite": False,
                    "scale_system": attempt["scale_system"],
                    "residual_correction": attempt["residual_correction"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if capacity_error is not None:
                    entry["terminal_capacity_failure"] = True
                    entry["phase"] = capacity_error.phase
                    attempt_log.append(entry)
                    capacity_error.amgx_attempts = tuple(attempt_log)
                    capacity_error.amgx_attempt_count = index
                    if verbose_level:
                        print(
                            f"  AMGX attempt {index}/{len(attempts)} {attempt['label']}: "
                            f"terminal capacity failure ({capacity_error})",
                            flush=True,
                        )
                    raise capacity_error
                last_error = exc
                attempt_log.append(entry)
                if verbose_level:
                    print(
                        f"  AMGX attempt {index}/{len(attempts)} {attempt['label']}: "
                        f"failed ({type(exc).__name__}: {exc})",
                        flush=True,
                    )
    finally:
        retry_wrapper_elapsed = time.perf_counter() - retry_wrapper_start
        timed_result = last_result if last_result is not None else best_result
        if timed_result is not None:
            attach_seed_metrics(timed_result)
            timed_result.amgx_retry_matrix_backup_elapsed_seconds = 0.0
            timed_result.amgx_retry_matrix_restore_elapsed_seconds = 0.0
            timed_result.amgx_retry_matrix_restore_count = 0
            timed_result.amgx_retry_matrix_backup_bytes = 0
            timed_result.amgx_retry_wrapper_elapsed_seconds = retry_wrapper_elapsed
            attempt_elapsed = float(getattr(timed_result, "total_elapsed_seconds", 0.0) or 0.0)
            timed_result.amgx_retry_outer_overhead_elapsed_seconds = max(
                0.0, retry_wrapper_elapsed - attempt_elapsed
            )
        if verbose_level == 2 or verbose_level >= 4:
            print("  AMGX retry-wrapper timings:", flush=True)
            print("    matrix backup to host: disabled", flush=True)
            print(f"    wrapper total: {retry_wrapper_elapsed:.5f}s", flush=True)

    if not raise_on_nonconvergence and (best_result is not None or last_result is not None):
        selected_result = best_result or last_result
        selected_solution = best_solution if best_result is not None else last_solution
        selected_result.amgx_attempts = tuple(attempt_log)
        selected_result.amgx_attempt_count = len(attempt_log)
        return selected_result, selected_solution
    failed_result = best_result or last_result
    if failed_result is not None:
        failed_result.amgx_attempts = tuple(attempt_log)
        failed_result.amgx_attempt_count = len(attempt_log)
        attach_seed_metrics(failed_result)
    details = "; ".join(
        f"{entry['label']}: {entry.get('error') or entry.get('failure_reason') or 'residual target not met'}"
        for entry in attempt_log
    )
    message = f"pyamgx-device solve exhausted {len(attempts)} bounded attempts: {details}"
    error = LinearSolveConvergenceError(message, result=failed_result)
    error.amgx_attempts = tuple(attempt_log)
    if seed_metrics is not None:
        error.amgx_retry_seed = dict(seed_metrics)
    # Preserve the failed system for the application error handler. Successful
    # solves incur no host transfer or snapshot allocation.
    from hdgfem.linalg.transport_diagnostics import save_transport_failure_snapshot

    def save_failure_snapshot(path):
        return save_transport_failure_snapshot(
            path, assembly, initial_guess=initial_guess, best_solution=best_solution,
        )

    error.save_transport_snapshot = save_failure_snapshot
    try:
        error.matrix_diagnostics = _device_transport_matrix_diagnostics(assembly)
    except Exception as diagnostic_error:
        error.matrix_diagnostics = {"error": f"{type(diagnostic_error).__name__}: {diagnostic_error}"}
    if last_error is not None:
        raise error from last_error
    raise error


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
    "CudaAdvectionAssembly",
    "assemble_reduced_system_cuda",
    "beta_dot_normal_from_coeffs",
    "build_dof_maps",
    "project_callable_cupy",
    "reconstruct_advection_field_cuda",
    "reconstruct_trace_cupy",
    "PyAMGXCsrDeviceSolver",
    "solve_reduced_system_amgx_device",
]
