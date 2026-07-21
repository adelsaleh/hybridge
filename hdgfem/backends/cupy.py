"""CuPy adapters for HDG assembly and PyAMGX solves.

The public solvers keep using :class:`DGSpace`, :class:`DGField`, and NumPy
result arrays. This module owns persistent device mirrors of package mesh and
reference-element data so CuPy assembly does not repeatedly copy static tables
from host RAM to VRAM.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse

from ..assembly import hdg as hdg_assembly
from ..core.space import DGField, DGSpace, VectorDGField
from ..linalg.system import KnownDofReduction

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


@dataclass(frozen=True)
class CupyReferenceElementData:
    """Device mirror of immutable reference-element tables."""

    host: Any
    device_id: int
    order: int
    basis_type: str
    el_dof: int
    edg_dof: int
    Krf_quads: Any
    Krf_w: Any
    bas_of_quads: Any
    dbas_of_quads: Any
    quads_JGL: Any
    weights_JGL: Any
    rf_edg_lag_nodes: Any
    pts_fc: Any
    bas_of_bd_quads: Any
    bas1d_of_ref_edg_qds: Any
    weighted_bas_of_bd_quads: Any
    weighted_bas1d_of_ref_edg_qds: Any
    face_trace_test_element_trial_oriented: Any
    M_rf_fc: Any
    M_rf_fc_f: Any
    MKrf: Any
    MKrf_inv: Any
    phi: Any
    gphi: Any
    weighted_phi: Any
    weighted_phi_phi_flat: Any
    weighted_triple_phi_flat: Any
    projection_operator: Any

    @classmethod
    def from_host(cls, quad_data, *, device_id: int) -> "CupyReferenceElementData":
        return cls(
            host=quad_data,
            device_id=int(device_id),
            order=int(quad_data.order),
            basis_type=str(quad_data.basis_type),
            el_dof=int(quad_data.el_dof),
            edg_dof=int(quad_data.edg_dof),
            Krf_quads=_cp_array(quad_data.Krf_quads),
            Krf_w=_cp_array(quad_data.Krf_w),
            bas_of_quads=_cp_array(quad_data.bas_of_quads),
            dbas_of_quads=_cp_array(quad_data.dbas_of_quads),
            quads_JGL=_cp_array(quad_data.quads_JGL),
            weights_JGL=_cp_array(quad_data.weights_JGL),
            rf_edg_lag_nodes=_cp_array(quad_data.rf_edg_lag_nodes),
            pts_fc=_cp_array(quad_data.pts_fc),
            bas_of_bd_quads=_cp_array(quad_data.bas_of_bd_quads),
            bas1d_of_ref_edg_qds=_cp_array(quad_data.bas1d_of_ref_edg_qds),
            weighted_bas_of_bd_quads=_cp_array(quad_data.weighted_bas_of_bd_quads),
            weighted_bas1d_of_ref_edg_qds=_cp_array(quad_data.weighted_bas1d_of_ref_edg_qds),
            face_trace_test_element_trial_oriented=_cp_array(quad_data.face_trace_test_element_trial_oriented),
            M_rf_fc=_cp_array(quad_data.M_rf_fc),
            M_rf_fc_f=_cp_array(quad_data.M_rf_fc_f),
            MKrf=_cp_array(quad_data.MKrf),
            MKrf_inv=_cp_array(quad_data.MKrf_inv),
            phi=_cp_array(quad_data.phi),
            gphi=_cp_array(quad_data.gphi),
            weighted_phi=_cp_array(quad_data.weighted_phi),
            weighted_phi_phi_flat=_cp_array(quad_data.weighted_phi_phi_flat),
            weighted_triple_phi_flat=_cp_array(quad_data.weighted_triple_phi_flat),
            projection_operator=_cp_array(quad_data.MKrf_inv @ quad_data.weighted_phi.T),
        )


@dataclass(frozen=True)
class CupyDGMesh:
    """Device mirror of immutable mesh topology and geometry arrays."""

    host: Any
    device_id: int
    num_tri: int
    num_edg: int
    node_coords: Any
    triangles: Any
    edges: Any
    loc2glob_edge: Any
    loc2oriented_face_coupling: Any
    orientations: Any
    interior_face_mask: Any
    interior_elements: Any
    interior_faces: Any
    int_edges_inds: Any
    bnd_edges_inds: Any
    edge_jacs: Any
    aff_mats: Any
    aff_vecs: Any
    aff_jacs: Any
    inv_aff_mats: Any
    inv_aff_mats_t: Any
    normals: Any
    jacs_el_fc: Any
    negative_orientation_elements: Any
    negative_orientation_faces: Any
    num_negative_orientations: int

    @classmethod
    def from_host(cls, mesh, *, device_id: int) -> "CupyDGMesh":
        cupy = require_cupy()
        negative_elements, negative_faces = np.nonzero(~mesh.orientations)
        return cls(
            host=mesh,
            device_id=int(device_id),
            num_tri=int(mesh.num_tri),
            num_edg=int(mesh.num_edg),
            node_coords=_cp_array(mesh.node_coords),
            triangles=_cp_array(mesh.triangles, dtype=cupy.int64),
            edges=_cp_array(mesh.edges, dtype=cupy.int64),
            loc2glob_edge=_cp_array(mesh.loc2glob_edge, dtype=cupy.int64),
            loc2oriented_face_coupling=_cp_array(mesh.loc2oriented_face_coupling, dtype=cupy.int64),
            orientations=_cp_array(mesh.orientations, dtype=cupy.bool_),
            interior_face_mask=_cp_array(mesh.interior_face_mask, dtype=cupy.bool_),
            interior_elements=_cp_array(mesh.interior_elements, dtype=cupy.int64),
            interior_faces=_cp_array(mesh.interior_faces, dtype=cupy.int64),
            int_edges_inds=_cp_array(mesh.int_edges_inds, dtype=cupy.int64),
            bnd_edges_inds=_cp_array(mesh.bnd_edges_inds, dtype=cupy.int64),
            edge_jacs=_cp_array(mesh.edge_jacs),
            aff_mats=_cp_array(mesh.aff_mats),
            aff_vecs=_cp_array(mesh.aff_vecs),
            aff_jacs=_cp_array(mesh.aff_jacs),
            inv_aff_mats=_cp_array(mesh.inv_aff_mats),
            inv_aff_mats_t=_cp_array(mesh.inv_aff_mats_t),
            normals=_cp_array(mesh.normals),
            jacs_el_fc=_cp_array(mesh.jacs_el_fc),
            negative_orientation_elements=_cp_array(negative_elements, dtype=cupy.int64),
            negative_orientation_faces=_cp_array(negative_faces, dtype=cupy.int64),
            num_negative_orientations=int(negative_elements.size),
        )


@dataclass(frozen=True)
class CupyDGSpace:
    """Device mirror of a :class:`DGSpace` for CuPy backend kernels."""

    host: DGSpace
    device_id: int
    mesh: CupyDGMesh
    quad_data: CupyReferenceElementData
    name: str
    order: int
    el_dof: int
    edg_dof: int
    _mapped_quads_cache: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def mapped_quads(self):
        """Mapped volume quadrature coordinates on the CUDA device, shape ``(K, 2, q)``."""
        cached = self._mapped_quads_cache
        if cached is None:
            cupy = require_cupy()
            cached = cupy.einsum("Krc,qc->Krq", self.mesh.aff_mats, self.quad_data.Krf_quads) + self.mesh.aff_vecs[:, :, None]
            object.__setattr__(self, "_mapped_quads_cache", cached)
        return cached

    @classmethod
    def from_host(cls, space: DGSpace, *, device_id: int) -> "CupyDGSpace":
        return cls(
            host=space,
            device_id=int(device_id),
            mesh=CupyDGMesh.from_host(space.mesh, device_id=device_id),
            quad_data=CupyReferenceElementData.from_host(space.quad_data, device_id=device_id),
            name=space.name,
            order=int(space.order),
            el_dof=int(space.el_dof),
            edg_dof=int(space.quad_data.edg_dof),
        )


@dataclass(frozen=True)
class CupyAdvectionTraceAssembly:
    """Advection-reaction trace data assembled with CuPy."""

    trace_system: hdg_assembly.TraceSystem
    beta_dot_normal: np.ndarray
    local_solver: Any | None
    element_boundary_mats: Any | None
    timings: dict[str, float]
    reduction: KnownDofReduction | None = None


def require_cupy():
    """Return the CuPy module or raise a clear dependency error."""
    if cp is None:
        raise RuntimeError("CuPy is not installed in this environment") from _CUPY_IMPORT_ERROR
    return cp


def require_cupyx_sparse():
    """Return ``cupyx.scipy.sparse`` or raise a clear dependency error."""
    if cupyx_sparse is None:
        raise RuntimeError("cupyx.scipy.sparse is not available in this environment") from _CUPYX_SPARSE_IMPORT_ERROR
    return cupyx_sparse


def require_cupyx_sparse_linalg():
    """Return ``cupyx.scipy.sparse.linalg`` or raise a clear dependency error."""
    if cupyx_sparse_linalg is None:
        raise RuntimeError(
            "cupyx.scipy.sparse.linalg is not available in this environment"
        ) from _CUPYX_SPARSE_LINALG_IMPORT_ERROR
    return cupyx_sparse_linalg


def require_pyamgx():
    """Return the PyAMGX module or raise a clear dependency error."""
    if pyamgx is None:
        raise RuntimeError("PyAMGX solve requested, but pyamgx is not importable") from _PYAMGX_IMPORT_ERROR
    return pyamgx


def asnumpy(array) -> np.ndarray:
    """Return ``array`` as a NumPy array without importing CuPy at call sites."""
    cupy = require_cupy()
    return cupy.asnumpy(array)


def _cp_array(value, *, dtype=None):
    """Create a contiguous CuPy array from package-owned NumPy data."""
    cupy = require_cupy()
    if dtype is None:
        dtype = cupy.float64
    return cupy.asarray(np.ascontiguousarray(value), dtype=dtype)


def _current_device_id() -> int:
    """Return the active CUDA device id."""
    cupy = require_cupy()
    return int(cupy.cuda.runtime.getDevice())


def as_cupy_space(space: DGSpace | CupyDGSpace, *, device: int | None = None) -> CupyDGSpace:
    """Return a cached device mirror of ``space``.

    Static mesh geometry and reference-element tensors are copied to the active
    CUDA device once per ``DGSpace`` and device id. Dynamic problem data such as
    source values or callable coefficient samples remains per-solve data.
    """
    cupy = require_cupy()
    if isinstance(space, CupyDGSpace):
        if device is None or int(device) == space.device_id:
            return space
        space = space.host

    device_id = _current_device_id() if device is None else int(device)
    cache = getattr(space, "_hdgfem_cupy_space_cache", None)
    if cache is None:
        cache = {}
        setattr(space, "_hdgfem_cupy_space_cache", cache)
    if device_id not in cache:
        with cupy.cuda.Device(device_id):
            cache[device_id] = CupyDGSpace.from_host(space, device_id=device_id)
    return cache[device_id]


def clear_cupy_space_cache(space: DGSpace) -> None:
    """Drop cached CuPy mirrors attached to ``space``."""
    cache = getattr(space, "_hdgfem_cupy_space_cache", None)
    if cache is not None:
        cache.clear()


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
    values = np.empty((num_elements, num_points, 2), dtype=np.float64)
    if beta_field is not None:
        if beta_field.dim != 2:
            raise ValueError("beta_field must have two components")
        beta_field.components[0].space.assert_same_mesh(space)
        beta_field.components[1].space.assert_same_mesh(space)
        coeffs = cupy.asarray(np.ascontiguousarray(beta_field.as_component_first(), dtype=np.float64))
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
        coeffs = cupy.asarray(np.ascontiguousarray(reaction.coeffs, dtype=np.float64))
        values = coeffs @ q.bas_of_quads
    elif callable(reaction):
        mapped_quads = cupy.einsum(
            "Krc,qc->Krq",
            mesh.aff_mats,
            q.Krf_quads,
        ) + mesh.aff_vecs[:, :, None]
        try:
            values = cupy.asarray(reaction(mapped_quads[:, 0, :], mapped_quads[:, 1, :]), dtype=cupy.float64)
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


def _boundary_mass_from_normal_flux_cupy(cspace: CupyDGSpace, beta_dot_normal):
    """Assemble upwind boundary mass matrices on the GPU."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    return cupy.einsum(
        "Kf,Kfq,fiq,fjq->Kij",
        mesh.jacs_el_fc,
        cupy.abs(beta_dot_normal),
        q.bas_of_bd_quads,
        q.weighted_bas_of_bd_quads,
        optimize=True,
    )


def _element_boundary_mats_from_normal_flux_cupy(cspace: CupyDGSpace, beta_dot_normal):
    """Assemble element-to-trace upwind coupling matrices on the GPU."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    flux_weight = cupy.abs(beta_dot_normal) - beta_dot_normal
    result = cupy.einsum(
        "Kf,Kfq,fiq,jq->Kifj",
        mesh.jacs_el_fc,
        flux_weight,
        q.bas_of_bd_quads,
        q.weighted_bas1d_of_ref_edg_qds,
        optimize=True,
    )
    return result.reshape(mesh.num_tri, cspace.el_dof, 3 * cspace.edg_dof)


def _element_to_trace_matrix_cupy(local_solver, element_boundary_mats, cspace: CupyDGSpace):
    """Assemble oriented element Schur complement blocks on the GPU."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    trace_lift = (
        mesh.jacs_el_fc[..., None, None]
        / 2.0
        * q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    )
    schur = trace_lift @ (local_solver @ element_boundary_mats)[:, None, :, :]
    edg_dof = cspace.edg_dof
    schur = schur.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    if mesh.num_negative_orientations:
        schur[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :] = (
            schur[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, ::-1]
        )
    return cupy.ascontiguousarray(schur.swapaxes(2, 3))


def _trace_matrix_data_cupy(trace_blocks, cspace: CupyDGSpace, boundary_penalty: float):
    """Return COO data values matching :func:`hdg_assembly.trace_matrix_indices`."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    edg_dof = cspace.edg_dof
    n_interior_flux = mesh.int_edges_inds.size * 2 * edg_dof * 3 * edg_dof
    n_interior_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    n_boundary = mesh.bnd_edges_inds.size * edg_dof
    data = cupy.empty(n_interior_flux + n_interior_mass + n_boundary, dtype=cupy.float64)

    data[:n_interior_flux] = -trace_blocks[mesh.interior_elements, mesh.interior_faces].ravel()

    offset = n_interior_flux
    edge_jacs = mesh.edge_jacs[mesh.int_edges_inds]
    data[offset:offset + n_interior_mass] = (edge_jacs[:, None, None] * q.M_rf_fc[None]).ravel()

    offset += n_interior_mass
    data[offset:] = float(boundary_penalty)
    return data


def _reduced_trace_matrix_indices_cupy(cspace: CupyDGSpace):
    """Return full-edge-numbered COO indices before boundary block elimination."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_interior_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_interior_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
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
    rows[offset:] = (mesh.int_edges_inds[:, None] * edg_dof + l0).ravel()
    cols[offset:] = (mesh.int_edges_inds[:, None] * edg_dof + l1).ravel()
    return rows, cols


def _reduced_trace_matrix_data_cupy(trace_blocks, cspace: CupyDGSpace):
    """Return COO data values before boundary block elimination."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_interior_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_interior_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    data = cupy.empty(n_interior_flux + n_interior_mass, dtype=cupy.float64)

    data[:n_interior_flux] = -trace_blocks[valid_elements, valid_faces].ravel()
    edge_jacs = mesh.edge_jacs[mesh.int_edges_inds]
    data[n_interior_flux:] = (edge_jacs[:, None, None] * q.M_rf_fc[None]).ravel()
    return data


def _global_rhs_without_boundary_penalty_cupy(source_rhs, local_solver, cspace: CupyDGSpace):
    """Assemble the full trace RHS without prescribed-boundary penalty rows."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    trace_lift = (
        mesh.jacs_el_fc[..., None, None]
        / 2.0
        * q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    )
    source_rhs_cp = cupy.asarray(np.ascontiguousarray(source_rhs, dtype=np.float64))
    face_rhs = (trace_lift @ (local_solver @ source_rhs_cp[..., None])[:, None, :, :]).squeeze(-1)
    rhs = cupy.zeros((mesh.num_edg, cspace.edg_dof), dtype=cupy.float64)
    if mesh.interior_elements.size:
        cupy.add.at(
            rhs,
            mesh.loc2glob_edge[mesh.interior_elements, mesh.interior_faces],
            face_rhs[mesh.interior_elements, mesh.interior_faces],
        )
    return rhs.ravel()


def _eliminate_boundary_trace_dofs_cupy(row, col, data, rhs, boundary_trace, cspace: CupyDGSpace) -> KnownDofReduction:
    """Eliminate boundary trace columns on-device using the legacy block layout."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
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
    reduced_data = cupy.empty(keep_count * edg_dof, dtype=cupy.float64)
    reduced_rows.reshape((keep_count, edg_dof))[:] = old_to_new_cp[row_r[keep_blocks]]
    reduced_cols.reshape((keep_count, edg_dof))[:] = old_to_new_cp[col_r[keep_blocks]]
    reduced_data.reshape((keep_count, edg_dof))[:] = data_r[keep_blocks]

    boundary_trace_cp = cupy.asarray(np.ascontiguousarray(boundary_trace, dtype=np.float64))
    if remove_blocks.size:
        row_ids = row_r[remove_blocks, 0]
        col_edges = col_r[remove_blocks, 0] // edg_dof
        cupy.add.at(
            rhs,
            row_ids,
            cupy.sum(-data_r[remove_blocks] * boundary_trace_cp[col_edges], axis=1),
        )
    reduced_rhs = rhs.reshape((mesh.num_edg, edg_dof))[mesh.int_edges_inds].ravel()

    free_mask = cupy.asnumpy(free_mask_cp)
    old_to_new = cupy.asnumpy(old_to_new_cp)
    known_mask = ~free_mask
    known_values = boundary_trace.ravel()
    return KnownDofReduction(
        rows=np.ascontiguousarray(cupy.asnumpy(reduced_rows)),
        cols=np.ascontiguousarray(cupy.asnumpy(reduced_cols)),
        data=np.ascontiguousarray(cupy.asnumpy(reduced_data)),
        rhs=np.ascontiguousarray(cupy.asnumpy(reduced_rhs)),
        free_mask=np.ascontiguousarray(free_mask),
        known_mask=np.ascontiguousarray(known_mask),
        known_values=np.ascontiguousarray(known_values),
        old_to_new=np.ascontiguousarray(old_to_new),
    )


def _global_rhs_cupy(source_rhs, local_solver, boundary_condition: Callable, cspace: CupyDGSpace, boundary_penalty: float):
    """Assemble the trace RHS on the GPU and boundary trace on the host."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    host_space = cspace.host
    trace_lift = (
        mesh.jacs_el_fc[..., None, None]
        / 2.0
        * q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    )
    source_rhs_cp = cupy.asarray(np.ascontiguousarray(source_rhs, dtype=np.float64))
    face_rhs = (trace_lift @ (local_solver @ source_rhs_cp[..., None])[:, None, :, :]).squeeze(-1)
    rhs = cupy.zeros((mesh.num_edg, cspace.edg_dof), dtype=cupy.float64)
    if mesh.interior_elements.size:
        cupy.add.at(
            rhs,
            mesh.loc2glob_edge[mesh.interior_elements, mesh.interior_faces],
            face_rhs[mesh.interior_elements, mesh.interior_faces],
        )

    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, host_space)
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
) -> CupyAdvectionTraceAssembly:
    """Assemble the full advection-reaction HDG trace system with CuPy."""
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)
    host_space = cspace.host

    start = time.perf_counter()
    beta_dot_normal_cp = cupy.asarray(np.ascontiguousarray(beta_dot_normal, dtype=np.float64))
    local_mats = _boundary_mass_from_normal_flux_cupy(cspace, beta_dot_normal_cp)
    local_mats += _reaction_mass_cupy(reaction, cspace)
    local_mats -= _advection_mats_cupy(cspace, beta_field=beta_field, beta_callables=beta_callables)
    element_boundary_mats = _element_boundary_mats_from_normal_flux_cupy(cspace, beta_dot_normal_cp)
    timings["local_assembly"] = time.perf_counter() - start

    start = time.perf_counter()
    local_solver = cupy.linalg.inv(local_mats)
    timings["local_inverse"] = time.perf_counter() - start

    start = time.perf_counter()
    trace_blocks = _element_to_trace_matrix_cupy(local_solver, element_boundary_mats, cspace)
    rows, cols = hdg_assembly.trace_matrix_indices(host_space)
    data = _trace_matrix_data_cupy(trace_blocks, cspace, boundary_penalty)
    rhs, boundary_trace = _global_rhs_cupy(source_rhs, local_solver, boundary_condition, cspace, boundary_penalty)
    cupy.cuda.get_current_stream().synchronize()
    timings["trace_assembly"] = time.perf_counter() - start

    start = time.perf_counter()
    trace_system = hdg_assembly.TraceSystem(
        rows=np.ascontiguousarray(rows),
        cols=np.ascontiguousarray(cols),
        data=cupy.asnumpy(data),
        rhs=cupy.asnumpy(rhs),
        boundary_trace=boundary_trace,
    )
    local_solver_host = cupy.asnumpy(local_solver)
    element_boundary_mats_host = cupy.asnumpy(element_boundary_mats)
    beta_dot_normal_host = cupy.asnumpy(beta_dot_normal_cp)
    timings["host_transfer"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())

    return CupyAdvectionTraceAssembly(
        trace_system=trace_system,
        beta_dot_normal=beta_dot_normal_host,
        local_solver=np.ascontiguousarray(local_solver_host),
        element_boundary_mats=np.ascontiguousarray(element_boundary_mats_host),
        timings=timings,
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
) -> CupyAdvectionTraceAssembly:
    """Assemble the reduced advection-reaction HDG trace system with CuPy."""
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)

    start = time.perf_counter()
    beta_dot_normal_cp = cupy.asarray(np.ascontiguousarray(beta_dot_normal, dtype=np.float64))
    local_mats = _boundary_mass_from_normal_flux_cupy(cspace, beta_dot_normal_cp)
    local_mats += _reaction_mass_cupy(reaction, cspace)
    local_mats -= _advection_mats_cupy(cspace, beta_field=beta_field, beta_callables=beta_callables)
    element_boundary_mats = _element_boundary_mats_from_normal_flux_cupy(cspace, beta_dot_normal_cp)
    cupy.cuda.get_current_stream().synchronize()
    timings["local_assembly"] = time.perf_counter() - start

    start = time.perf_counter()
    local_solver = cupy.linalg.inv(local_mats)
    cupy.cuda.get_current_stream().synchronize()
    timings["local_inverse"] = time.perf_counter() - start

    start = time.perf_counter()
    trace_blocks = _element_to_trace_matrix_cupy(local_solver, element_boundary_mats, cspace)
    rows_cp, cols_cp = _reduced_trace_matrix_indices_cupy(cspace)
    data_cp = _reduced_trace_matrix_data_cupy(trace_blocks, cspace)
    rhs_cp = _global_rhs_without_boundary_penalty_cupy(source_rhs, local_solver, cspace)
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, cspace.host)
    cupy.cuda.get_current_stream().synchronize()
    timings["trace_assembly"] = time.perf_counter() - start

    start = time.perf_counter()
    reduction = _eliminate_boundary_trace_dofs_cupy(rows_cp, cols_cp, data_cp, rhs_cp, boundary_trace, cspace)
    cupy.cuda.get_current_stream().synchronize()
    timings["boundary_elimination"] = time.perf_counter() - start

    start = time.perf_counter()
    beta_dot_normal_host = cupy.asnumpy(beta_dot_normal_cp)
    if transfer_local_solver:
        local_solver_result = np.ascontiguousarray(cupy.asnumpy(local_solver))
        element_boundary_mats_result = np.ascontiguousarray(cupy.asnumpy(element_boundary_mats))
    else:
        local_solver_result = local_solver
        element_boundary_mats_result = element_boundary_mats
    timings["host_transfer"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())

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
    )


def scipy_csr_to_cupy(matrix: scipy.sparse.spmatrix | scipy.sparse.sparray, *, dtype=None):
    """Convert a SciPy sparse matrix to a CuPy CSR matrix."""
    cupy = require_cupy()
    sparse = require_cupyx_sparse()
    if dtype is None:
        dtype = cupy.float64
    csr = matrix.tocsr()
    csr.sum_duplicates()
    return sparse.csr_matrix(
        (
            cupy.asarray(csr.data, dtype=dtype),
            cupy.asarray(csr.indices, dtype=cupy.int32),
            cupy.asarray(csr.indptr, dtype=cupy.int32),
        ),
        shape=csr.shape,
    )


def scipy_coo_to_cupy_csr(row_indices, col_indices, matrix_values, shape, *, dtype=None):
    """Copy host COO triplets to the GPU and construct CSR on the device."""
    cupy = require_cupy()
    sparse = require_cupyx_sparse()
    if dtype is None:
        dtype = cupy.float64
    coo = sparse.coo_matrix(
        (
            cupy.asarray(matrix_values, dtype=dtype),
            (
                cupy.asarray(row_indices, dtype=cupy.int32),
                cupy.asarray(col_indices, dtype=cupy.int32),
            ),
        ),
        shape=shape,
    )
    csr = coo.tocsr()
    csr.sum_duplicates()
    cupy.cuda.get_current_stream().synchronize()
    return csr


def build_cupyx_ilu_preconditioner(
        matrix,
        *,
        drop_tol: float,
        fill_factor: float,
        permc_spec: str | None,
):
    """Build a Cupyx ILU preconditioner as a GPU LinearOperator."""
    linalg = require_cupyx_sparse_linalg()
    kwargs = {
        "drop_tol": float(drop_tol),
        "fill_factor": float(fill_factor),
    }
    if permc_spec is not None:
        kwargs["permc_spec"] = permc_spec
    ilu = linalg.spilu(matrix, **kwargs)
    return linalg.LinearOperator(matrix.shape, matvec=ilu.solve, dtype=matrix.dtype)


def solve_cupyx_csr(
        matrix,
        rhs,
        *,
        solver: str = "cg",
        preconditioner=None,
        initial_guess=None,
        rtol: float = 1e-13,
        atol: float = 0.0,
        maxiter: int | None = None,
        restart: int | None = None,
):
    """Solve a CuPy CSR system with ``cupyx.scipy.sparse.linalg``.

    Parameters are intentionally close to :func:`scipy.sparse.linalg` Krylov
    solvers.  ``matrix`` is expected to already be a CuPy sparse matrix; callers
    that assemble on the host should convert once with :func:`scipy_csr_to_cupy`
    and cache that device matrix when the operator is reused.

    CuPy has used both SciPy's modern ``rtol``/``atol`` convention and older
    ``tol``-only signatures across releases.  The wrapper first tries
    ``rtol``/``atol`` and falls back to ``tol`` so this optional backend remains
    version tolerant.
    """
    cupy = require_cupy()
    linalg = require_cupyx_sparse_linalg()
    normalized = str(solver).lower().replace("-", "_")
    aliases = {
        "bicgstab": "bicgstab",
        "bicg_stab": "bicgstab",
        "bcgs": "bicgstab",
        "cg": "cg",
        "cgs": "cgs",
        "gmres": "gmres",
    }
    solver_name = aliases.get(normalized, normalized)
    if solver_name not in {"cg", "bicgstab", "cgs", "gmres"}:
        raise ValueError("cupyx solver must be one of 'cg', 'bicgstab', 'cgs', or 'gmres'")
    solver_fn = getattr(linalg, solver_name, None)
    if solver_fn is None:
        raise RuntimeError(f"cupyx.scipy.sparse.linalg.{solver_name} is not available")

    rhs_cp = cupy.asarray(rhs, dtype=matrix.dtype)
    kwargs: dict[str, Any] = {}
    if initial_guess is not None:
        x0 = cupy.asarray(initial_guess, dtype=matrix.dtype)
        if x0.shape != rhs_cp.shape:
            raise ValueError(f"initial_guess must have shape {rhs_cp.shape}; got {x0.shape}")
        kwargs["x0"] = x0
    if maxiter is not None:
        kwargs["maxiter"] = int(maxiter)
    if restart is not None and solver_name == "gmres":
        kwargs["restart"] = int(restart)
    if preconditioner is not None:
        kwargs["M"] = preconditioner

    try:
        solution, info = solver_fn(matrix, rhs_cp, rtol=rtol, atol=atol, **kwargs)
    except TypeError:
        solution, info = solver_fn(matrix, rhs_cp, tol=rtol, **kwargs)
    cupy.cuda.get_current_stream().synchronize()
    return solution, int(info)


def default_pyamgx_config(*, tolerance: float, maxiter: int | None, verbose: bool | int = 0) -> dict[str, Any]:
    """Return the default AMGX BICGSTAB+AMG configuration."""
    monitor = int(bool(verbose) and int(verbose) >= 3)
    return {
        "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "solver": "BICGSTAB",
            "monitor_residual": monitor,
            "convergence": "RELATIVE_INI_CORE",
            "tolerance": float(tolerance),
            "max_iters": int(maxiter) if maxiter is not None else 1500,
            "obtain_timings": int(bool(verbose) and int(verbose) >= 2),
            "preconditioner": {
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "selector": "PMIS",
                "cycle": "V",
                "strength_threshold": 0.5,
                "coarse_solver": "DENSE_LU_SOLVER",
                "presweeps": 2,
                "postsweeps": 2,
                "max_levels": 50,
            },
        },
    }


def solve_pyamgx_csr(
        matrix,
        rhs,
        *,
        initial_guess=None,
        config: Mapping[str, Any] | None = None,
        tolerance: float = 1e-13,
        maxiter: int | None = None,
        verbose: bool | int = 0,
):
    """Solve a CuPy CSR system with PyAMGX and return a CuPy solution."""
    cupy = require_cupy()
    amgx = require_pyamgx()
    rhs_cp = cupy.asarray(rhs, dtype=cupy.float64)
    if initial_guess is None:
        x_cp = cupy.zeros_like(rhs_cp)
    else:
        x_cp = cupy.asarray(initial_guess, dtype=cupy.float64).copy()
        if x_cp.shape != rhs_cp.shape:
            raise ValueError(f"initial_guess must have shape {rhs_cp.shape}; got {x_cp.shape}")

    amgx_config = default_pyamgx_config(tolerance=tolerance, maxiter=maxiter, verbose=verbose)
    if config is not None:
        amgx_config = dict(config)

    amgx.initialize()
    cfg = rsrc = mat = vec_b = vec_x = solver = None
    try:
        cfg = amgx.Config().create_from_dict(amgx_config)
        rsrc = amgx.Resources().create_simple(cfg)
        mat = amgx.Matrix().create(rsrc, mode="dDDI")
        vec_b = amgx.Vector().create(rsrc, mode="dDDI")
        vec_x = amgx.Vector().create(rsrc, mode="dDDI")
        mat.upload_CSR(matrix)
        vec_b.upload_raw(rhs_cp.data.ptr, rhs_cp.size)
        vec_x.upload_raw(x_cp.data.ptr, x_cp.size)
        solver = amgx.Solver().create(rsrc, cfg)
        solver.setup(mat)
        solver.solve(vec_b, vec_x)
        vec_x.download_raw(x_cp.data.ptr)
        cupy.cuda.get_current_stream().synchronize()
    finally:
        for obj in (solver, mat, vec_x, vec_b, rsrc, cfg):
            if obj is not None:
                try:
                    obj.destroy()
                except AttributeError:
                    pass
        amgx.finalize()
    return x_cp


__all__ = [
    "CupyAdvectionTraceAssembly",
    "CupyDGMesh",
    "CupyDGSpace",
    "CupyReferenceElementData",
    "assemble_advection_reaction_trace_system_cupy",
    "assemble_advection_reaction_trace_system_eliminated_cupy",
    "as_cupy_space",
    "asnumpy",
    "clear_cupy_space_cache",
    "default_pyamgx_config",
    "require_cupy",
    "require_cupyx_sparse",
    "require_cupyx_sparse_linalg",
    "require_pyamgx",
    "build_cupyx_ilu_preconditioner",
    "scipy_coo_to_cupy_csr",
    "scipy_csr_to_cupy",
    "solve_cupyx_csr",
    "solve_pyamgx_csr",
]
