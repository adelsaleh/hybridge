"""CuPy adapters for HDG assembly and PyAMGX solves.

The public solvers keep using :class:`DGSpace`, :class:`DGField`, and NumPy
result arrays. This module owns persistent device mirrors of package mesh and
reference-element data so CuPy assembly does not repeatedly copy static tables
from host RAM to VRAM.
"""

from __future__ import annotations

from hdgfem.precision import REAL_DTYPE, AMGX_MODE, real_raw_kernel

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from ..assembly import hdg as hdg_assembly
from ..assembly import matrices_numpy as hdg_mats
from ..core.space import DGField, DGSpace, DGTraceSpace, VectorDGField
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

_PYAMGX_RUNTIME_INITIALIZED = False


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
        """Convert host-side data to device representation."""
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
    edge_side_indices: Any
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
        """Convert host-side data to device representation."""
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
            edge_side_indices=_cp_array(mesh.edge_side_indices, dtype=cupy.int64),
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
    _degree_elevation_cache: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def mapped_quads(self):
        """Mapped volume quadrature coordinates on the CUDA device, shape ``(K, 2, q)``."""
        cached = self._mapped_quads_cache
        if cached is None:
            cupy = require_cupy()
            cached = cupy.einsum("Krc,qc->Krq", self.mesh.aff_mats, self.quad_data.Krf_quads) + self.mesh.aff_vecs[:, :, None]
            object.__setattr__(self, "_mapped_quads_cache", cached)
        return cached

    def degree_elevation_matrix_from(self, source: DGSpace):
        """Return a cached device degree-elevation matrix from ``source``."""
        cupy = require_cupy()
        host_matrix = self.host.degree_elevation_matrix_from(source)
        cache = self._degree_elevation_cache
        if cache is None:
            cache = {}
            object.__setattr__(self, "_degree_elevation_cache", cache)
        matrix = cache.get(source.order)
        if matrix is None:
            matrix = cupy.ascontiguousarray(cupy.asarray(host_matrix, dtype=REAL_DTYPE))
            cache[source.order] = matrix
        return matrix

    def field(
            self,
            coeffs,
            *,
            copy: bool = False,
            name: str = "u",
            _coefficient_kind: str = "table",
            _constant_value: float | None = None,
    ) -> DGField:
        """Create a same-space DGField backed by a CuPy coefficient table."""
        return field_from_cupy_coefficients(
            self,
            coeffs,
            copy=copy,
            name=name,
            _coefficient_kind=_coefficient_kind,
            _constant_value=_constant_value,
        )

    def project_callable(self, func: Callable, *, parameters=None, name: str = "Pi_h f",
                         quadrature_integration: DGSpace | None = None, chunk_size: int = 8192,
                         timings: dict[str, float] | None = None) -> DGField:
        """Project on device, optionally accumulating synchronized phase times.

        The caller must drain earlier stream work before collecting timings.
        Without timings, the normal asynchronous batching is preserved.
        """
        from contextlib import nullcontext
        from ..io.output import timed_section

        cupy = require_cupy()
        sync = cupy.cuda.get_current_stream().synchronize if timings is not None else None

        def section(key):
            return (nullcontext() if timings is None else
                    timed_section(None, 2, key, timings=timings, synchronize=sync))

        if quadrature_integration is not None and quadrature_integration is not self.host:
            self.host.assert_same_mesh(quadrature_integration)
            if chunk_size < 1:
                raise ValueError("projection chunk_size must be positive")
            with section("projection_operator_setup_time"):
                cache = getattr(self, "_callable_projection_cache", None)
                if cache is None:
                    cache = {}
                    object.__setattr__(self, "_callable_projection_cache", cache)
                key = id(quadrature_integration.reference)
                if timings is not None:
                    timings["projection_operator_cache_hit"] = float(key in cache)
                if key not in cache:
                    rule = quadrature_integration.quad_data
                    points = cupy.asarray(rule.Krf_quads)
                    weighted = self.host.reference.basis_at(rule.Krf_quads) * rule.Krf_w[:, None]
                    operator = self.quad_data.MKrf_inv @ cupy.asarray(weighted).T
                    cache[key] = (quadrature_integration.reference, points, cupy.ascontiguousarray(operator.T))
                _, reference, operator = cache[key]
                coeffs = cupy.empty(self.host.shape, dtype=REAL_DTYPE)
            if timings is not None:
                timings["quadrature_points_per_element"] = len(reference)
                timings["sample_count"] = self.mesh.num_tri * len(reference)
                timings["batch_count"] = (self.mesh.num_tri + chunk_size - 1)//chunk_size
            for start in range(0, self.mesh.num_tri, chunk_size):
                stop = min(start+chunk_size, self.mesh.num_tri)
                with section("coordinate_mapping_time"):
                    points = (cupy.einsum("Krc,qc->Krq", self.mesh.aff_mats[start:stop], reference)
                              + self.mesh.aff_vecs[start:stop, :, None])
                with section("field_evaluation_time"):
                    raw = (func(points[:, 0], points[:, 1]) if parameters is None else
                           func(points[:, 0], points[:, 1], parameters))
                    values = _normalize_cupy_values(raw, stop-start, len(reference), "callable")
                with section("coefficient_projection_time"):
                    coeffs[start:stop] = values @ operator
            with section("field_wrap_time"):
                return self.field(coeffs, name=name, _coefficient_kind="projected")
        q = self.quad_data
        if timings is not None:
            timings["quadrature_points_per_element"] = len(q.Krf_w)
            timings["sample_count"] = self.mesh.num_tri * len(q.Krf_w)
            timings["batch_count"] = 1
        with section("coordinate_mapping_time"):
            points = self.mapped_quads
        with section("field_evaluation_time"):
            if parameters is None:
                raw = func(points[:, 0, :], points[:, 1, :])
            else:
                raw = func(points[:, 0, :], points[:, 1, :], parameters)
            values = _normalize_cupy_values(raw, self.mesh.num_tri, q.Krf_w.shape[0], "callable")
        with section("coefficient_projection_time"):
            coeffs = cupy.ascontiguousarray(values @ q.projection_operator.T)
        with section("field_wrap_time"):
            return self.field(coeffs, name=name, _coefficient_kind="projected")

    @classmethod
    def from_host(cls, space: DGSpace, *, device_id: int) -> "CupyDGSpace":
        """Convert host-side data to device representation."""
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
class CupyTraceReferenceData:
    """Device mirror of a host :class:`DGTraceSpace`."""

    host: DGTraceSpace
    device_id: int
    kind: str
    nodal: bool
    weights: Any
    bas_of_bd_quads: Any
    bas1d_of_ref_edg_qds: Any
    weighted_bas_of_bd_quads: Any
    weighted_bas1d_of_ref_edg_qds: Any
    face_trace_test_element_trial_oriented: Any
    M_rf_fc: Any

    @classmethod
    def from_host(cls, trace_space: DGTraceSpace, *, device_id: int) -> "CupyTraceReferenceData":
        """Convert host-side data to device representation."""
        return cls(
            host=trace_space,
            device_id=int(device_id),
            kind=trace_space.kind,
            nodal=bool(trace_space.nodal),
            weights=_cp_array(trace_space.weights),
            bas_of_bd_quads=_cp_array(trace_space.bas_of_bd_quads),
            bas1d_of_ref_edg_qds=_cp_array(trace_space.bas1d_of_ref_edg_qds),
            weighted_bas_of_bd_quads=_cp_array(trace_space.weighted_bas_of_bd_quads),
            weighted_bas1d_of_ref_edg_qds=_cp_array(trace_space.weighted_bas1d_of_ref_edg_qds),
            face_trace_test_element_trial_oriented=_cp_array(trace_space.face_trace_test_element_trial_oriented),
            M_rf_fc=_cp_array(trace_space.M_rf_fc),
        )

    @property
    def edg_dof(self) -> int:
        """Return the number of trace degrees of freedom per edge."""
        return self.host.edg_dof


def as_cupy_trace_reference(
        trace_space: DGTraceSpace | None,
        cspace: CupyDGSpace,
) -> CupyTraceReferenceData:
    """Return a cached device mirror for the requested trace space."""
    trace_ref = cspace.host.trace_space("legacy-lagrange") if trace_space is None else trace_space
    if trace_ref.kind == "legacy-lagrange" and trace_ref.nodal:
        pass
    elif trace_ref.kind == "legendre-modal" and not trace_ref.nodal:
        pass
    else:
        raise NotImplementedError(
            "assembly_backend='cupy' currently supports trace_basis='legacy-lagrange' "
            "and trace_basis='legendre-modal' for advection"
        )
    cache = getattr(trace_ref, "_hdgfem_cupy_trace_reference_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(trace_ref, "_hdgfem_cupy_trace_reference_cache", cache)
    if cspace.device_id not in cache:
        cache[cspace.device_id] = CupyTraceReferenceData.from_host(trace_ref, device_id=cspace.device_id)
    return cache[cspace.device_id]


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


def expand_known_dofs_cupy(reduced_solution, reduction: KnownDofReduction):
    """Expand a reduced trace vector on-device without a host round trip."""
    cupy = require_cupy()
    reduced_cp = cupy.asarray(reduced_solution, dtype=REAL_DTYPE)
    expected_size = int(np.count_nonzero(reduction.free_mask))
    if reduced_cp.shape != (expected_size,):
        raise ValueError(f"reduced_solution must have shape ({expected_size},); got {reduced_cp.shape}")
    full = cupy.asarray(reduction.known_values, dtype=REAL_DTYPE).copy()
    full[cupy.asarray(reduction.free_mask)] = reduced_cp
    return cupy.ascontiguousarray(full)


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


def require_cupy():
    """Return the CuPy module or raise a clear dependency error."""
    if cp is None:
        raise RuntimeError("CuPy is not installed in this environment") from _CUPY_IMPORT_ERROR
    return cp


def require_cupy_device():
    """Return CuPy after verifying that a CUDA device is available."""
    cupy = require_cupy()
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as error:  # pragma: no cover - CUDA-runtime dependent.
        raise RuntimeError(
            "CuPy is installed, but the CUDA runtime or driver is unavailable"
        ) from error
    if device_count < 1:  # pragma: no cover - hardware dependent.
        raise RuntimeError("CuPy is installed, but no CUDA device is available")
    return cupy


def device_arrays_overlap(first: Any, second: Any) -> bool:
    """Return whether two contiguous device arrays overlap in memory."""
    if int(first.device.id) != int(second.device.id):
        return False
    first_bytes = int(first.nbytes)
    second_bytes = int(second.nbytes)
    if first_bytes == 0 or second_bytes == 0:
        return False
    first_begin = int(first.data.ptr)
    second_begin = int(second.data.ptr)
    return (
        first_begin < second_begin + second_bytes
        and second_begin < first_begin + first_bytes
    )


def solve_batched_vectors(array_module: Any, matrices: Any, vectors: Any) -> Any:
    """Solve batched square systems with one vector RHS per matrix."""
    return array_module.linalg.solve(matrices, vectors[..., None])[..., 0]


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


def initialize_pyamgx_once():
    """Initialize AMGX once per process.

    PyAMGX/AMGX 2.5 does not tolerate repeated ``initialize()`` calls in one
    Python process because plugin readers are registered globally.
    Register a flushed output callback on first initialization so native
    progress remains visible when stdout is redirected to a terminal log.
    """
    global _PYAMGX_RUNTIME_INITIALIZED
    amgx = require_pyamgx()
    if not _PYAMGX_RUNTIME_INITIALIZED:
        amgx.initialize()
        _PYAMGX_RUNTIME_INITIALIZED = True
        # AMGX's default printf callback becomes block-buffered under terminal
        # log pipes. Its public callback emits each native iteration row now,
        # rather than leaving it in C stdout until a failed solve is unwound.
        from ..io.terminal import flush_native_stdio, write_native_solver_output

        flush_native_stdio()
        amgx.register_print_callback(write_native_solver_output)
    return amgx


def asnumpy(array) -> np.ndarray:
    """Return ``array`` as a NumPy array without importing CuPy at call sites."""
    cupy = require_cupy()
    return cupy.asnumpy(array)


def _cp_array(value, *, dtype=None):
    """Create a contiguous CuPy array from package-owned NumPy data."""
    cupy = require_cupy()
    if dtype is None:
        dtype = REAL_DTYPE
    return cupy.asarray(np.ascontiguousarray(value), dtype=dtype)


def _normalize_cupy_values(values, num_elements: int, num_points: int, label: str):
    """Normalize scalar/quadrature values to a contiguous device array."""
    cupy = require_cupy()
    values = cupy.asarray(values, dtype=REAL_DTYPE)
    if values.shape == (num_elements, num_points):
        return cupy.ascontiguousarray(values)
    if values.shape == (num_points,):
        return cupy.ascontiguousarray(cupy.broadcast_to(values[None, :], (num_elements, num_points)))
    if values.ndim == 0:
        return cupy.full((num_elements, num_points), float(values), dtype=REAL_DTYPE)
    raise ValueError(
        f"{label} must return a scalar, shape ({num_points},), or shape "
        f"({num_elements}, {num_points}); got {values.shape}"
    )


def _normalize_cupy_coefficients(coeffs, cspace: CupyDGSpace, *, copy: bool = False, label: str = "coeffs"):
    """Normalize a coefficient table to contiguous storage at the selected precision on ``cspace``."""
    cupy = require_cupy()
    array = cupy.asarray(coeffs, dtype=REAL_DTYPE)
    if array.shape != cspace.host.shape:
        raise ValueError(f"{label} must have shape {cspace.host.shape}; got {array.shape}")
    if copy:
        array = array.copy()
    if not array.flags.c_contiguous:
        array = cupy.ascontiguousarray(array)
    return array


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


def field_from_cupy_coefficients(
        space: DGSpace | CupyDGSpace,
        coeffs,
        *,
        device: int | None = None,
        copy: bool = False,
        name: str = "u",
        _coefficient_kind: str = "table",
        _constant_value: float | None = None,
) -> DGField:
    """Create a DGField whose coefficient table is cached on a CUDA device."""
    cupy = require_cupy()
    cspace = as_cupy_space(space, device=device)
    with cupy.cuda.Device(cspace.device_id):
        coeffs_cp = _normalize_cupy_coefficients(coeffs, cspace, copy=copy)
        return DGField.from_device_coefficients(
            cspace.host,
            coeffs_cp,
            device_id=cspace.device_id,
            name=name,
            _coefficient_kind=_coefficient_kind,
            _constant_value=_constant_value,
        )


def as_cupy_coefficients(field: DGField, cspace: CupyDGSpace, *, copy: bool = False):
    """Return ``field`` coefficients on ``cspace`` without unnecessary host copies."""
    if not isinstance(field, DGField):
        raise TypeError("as_cupy_coefficients expects a DGField")
    field.space.assert_same_mesh(cspace.host)
    cupy = require_cupy()
    with cupy.cuda.Device(cspace.device_id):
        cached = field._device_coefficients_for(cspace.device_id)
        if cached is not None:
            result = _normalize_cupy_coefficients(cached, cspace, copy=copy)
            if result is not cached:
                field._store_device_coefficients(result, device_id=cspace.device_id)
            return result

        constant_value = field.constant_value
        if constant_value is not None:
            reference = cupy.asarray(cspace.host._constant_reference_coeffs(constant_value), dtype=REAL_DTYPE)
            result = cupy.broadcast_to(reference[None, :], cspace.host.shape).copy()
        else:
            result = cupy.asarray(np.ascontiguousarray(field.coeffs, dtype=REAL_DTYPE), dtype=REAL_DTYPE)
            if not result.flags.c_contiguous:
                result = cupy.ascontiguousarray(result)
        field._store_device_coefficients(result, device_id=cspace.device_id)
        return result.copy() if copy else result


def as_cupy_vector_coefficients(field: VectorDGField, cspace: CupyDGSpace, *, copy: bool = False):
    """Return component-first vector DG coefficients on ``cspace``."""
    if not isinstance(field, VectorDGField):
        raise TypeError("as_cupy_vector_coefficients expects a VectorDGField")
    cupy = require_cupy()
    components = [as_cupy_coefficients(component, cspace) for component in field.components]
    result = cupy.ascontiguousarray(cupy.stack(components, axis=0))
    return result.copy() if copy else result


def clear_cupy_space_cache(space: DGSpace) -> None:
    """Drop cached CuPy mirrors attached to ``space``."""
    cache = getattr(space, "_hdgfem_cupy_space_cache", None)
    if cache is not None:
        cache.clear()


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
    from ..solvers.stabilization import upwind_factor, effective_advection_normal_flux, is_conflict_averaged_upwind
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
                    hdg_mats.dg_field_basis_on_trace_faces(stabilization.space, trace_ref.host),
                    dtype=REAL_DTYPE,
                )
            tau_face = cupy.einsum("Ki,fiq->Kfq", coefficients, face_basis, optimize=True)
    else:
        tau_host = hdg_mats.advection_trace_stabilization_values(
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
    from .advection_cuda import as_cupy_trace_space, oriented_trace_basis_cupy
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
        from ..solvers.stabilization import gauge_inactive_advection_trace_blocks
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
    from ..solvers.stabilization import is_conflict_averaged_upwind
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
    from ..solvers.stabilization import is_conflict_averaged_upwind
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


def scipy_csr_to_cupy(matrix: scipy.sparse.spmatrix | scipy.sparse.sparray, *, dtype=None):
    """Convert a SciPy sparse matrix to a CuPy CSR matrix."""
    cupy = require_cupy()
    sparse = require_cupyx_sparse()
    if dtype is None:
        dtype = REAL_DTYPE
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
        dtype = REAL_DTYPE
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
    """Build a CuPy ILU preconditioner directly on the device."""
    linalg = require_cupyx_sparse_linalg()
    cupy = require_cupy()
    kwargs = {
        "drop_tol": float(drop_tol),
        "fill_factor": float(fill_factor),
    }
    if permc_spec is not None:
        kwargs["permc_spec"] = permc_spec
    ilu = linalg.spilu(matrix, **kwargs)
    state = {"count": 0, "seconds": 0.0}

    def matvec(vec):
        """Apply a matrix-vector product."""
        start = time.perf_counter()
        out = ilu.solve(vec)
        cupy.cuda.get_current_stream().synchronize()
        state["count"] += 1
        state["seconds"] += time.perf_counter() - start
        operator.apply_count = state["count"]
        operator.apply_seconds = state["seconds"]
        return out

    operator = linalg.LinearOperator(matrix.shape, matvec=matvec, dtype=matrix.dtype)
    operator.apply_count = 0
    operator.apply_seconds = 0.0
    return operator


def build_cupyx_exported_host_ilu_preconditioner(
        matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
        *,
        drop_tol: float,
        fill_factor: float,
        permc_spec: str | None,
        dtype=None,
):
    """Build SciPy ILU on host, then apply its factors on the GPU.

    SciPy SuperLU stores factors satisfying ``Pr A Pc = L U``.  The returned
    CuPy ``LinearOperator`` applies ``M^{-1}`` as two device sparse triangular
    solves with the exported ``L`` and ``U`` factors.  Host memory is touched
    only while constructing and transferring the factorization.
    """
    cupy = require_cupy()
    linalg = require_cupyx_sparse_linalg()
    if dtype is None:
        dtype = REAL_DTYPE

    matrix_csc = matrix.tocsc()
    matrix_csc.eliminate_zeros()
    kwargs = {
        "drop_tol": float(drop_tol),
        "fill_factor": float(fill_factor),
    }
    if permc_spec is not None:
        kwargs["permc_spec"] = permc_spec
    ilu = scipy.sparse.linalg.spilu(matrix_csc, **kwargs)

    lower = scipy_csr_to_cupy(ilu.L.tocsr(), dtype=dtype)
    upper = scipy_csr_to_cupy(ilu.U.tocsr(), dtype=dtype)
    inv_perm_r = cupy.asarray(np.argsort(np.asarray(ilu.perm_r, dtype=np.int64)), dtype=cupy.int64)
    perm_c = cupy.asarray(np.asarray(ilu.perm_c, dtype=np.int64), dtype=cupy.int64)
    cupy.cuda.get_current_stream().synchronize()

    def matvec(vec):
        """Apply a matrix-vector product."""
        rhs_perm = vec[inv_perm_r]
        y = linalg.spsolve_triangular(lower, rhs_perm, lower=True, unit_diagonal=True)
        z = linalg.spsolve_triangular(upper, y, lower=False)
        return z[perm_c]

    operator = linalg.LinearOperator(matrix.shape, matvec=matvec, dtype=dtype)
    operator.host_ilu_nnz = int(ilu.L.nnz + ilu.U.nnz)
    operator.host_ilu_fill_ratio = float(operator.host_ilu_nnz) / max(int(matrix_csc.nnz), 1)
    return operator


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

    class _IterationCounter:
        """Callback class that counts iterative solver callback invocations."""

        def __init__(self):
            """Initialize the instance."""
            self.count = 0

        def __call__(self, *_args, **_kwargs):
            """Execute the configured call behavior."""
            self.count += 1

    counter = _IterationCounter()
    kwargs["callback"] = counter

    try:
        solution, info = solver_fn(matrix, rhs_cp, rtol=rtol, atol=atol, **kwargs)
    except TypeError:
        solution, info = solver_fn(matrix, rhs_cp, tol=rtol, **kwargs)
    cupy.cuda.get_current_stream().synchronize()
    return solution, int(info), counter.count

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
        return_info: bool = False,
):
    """Solve a CuPy CSR system with PyAMGX and optionally return native diagnostics."""
    cupy = require_cupy()
    amgx = initialize_pyamgx_once()
    rhs_cp = cupy.asarray(rhs, dtype=REAL_DTYPE)
    if initial_guess is None:
        x_cp = cupy.zeros_like(rhs_cp)
    else:
        x_cp = cupy.asarray(initial_guess, dtype=REAL_DTYPE).copy()
        if x_cp.shape != rhs_cp.shape:
            raise ValueError(f"initial_guess must have shape {rhs_cp.shape}; got {x_cp.shape}")

    amgx_config = default_pyamgx_config(tolerance=tolerance, maxiter=maxiter, verbose=verbose)
    if config is not None:
        amgx_config = dict(config)

    info = {"amgx_status": "unknown", "amgx_iterations": None, "residual_history": ()}
    cfg = rsrc = mat = vec_b = vec_x = solver = None
    try:
        cfg = amgx.Config().create_from_dict(amgx_config)
        rsrc = amgx.Resources().create_simple(cfg)
        mat = amgx.Matrix().create(rsrc, mode=AMGX_MODE)
        vec_b = amgx.Vector().create(rsrc, mode=AMGX_MODE)
        vec_x = amgx.Vector().create(rsrc, mode=AMGX_MODE)
        mat.upload_CSR(matrix)
        vec_b.upload_raw(rhs_cp.data.ptr, rhs_cp.size)
        vec_x.upload_raw(x_cp.data.ptr, x_cp.size)
        solver = amgx.Solver().create(rsrc, cfg, mode=AMGX_MODE)
        solver.setup(mat)
        solver.solve(vec_b, vec_x)
        vec_x.download_raw(x_cp.data.ptr)
        cupy.cuda.get_current_stream().synchronize()

        try:
            info["amgx_status"] = str(solver.status)
        except Exception:
            pass
        try:
            info["amgx_iterations"] = int(solver.iterations_number)
        except Exception:
            pass
        store_residual_history = bool(
            amgx_config.get("solver", {}).get("store_res_history", 0)
        )
        if store_residual_history and info["amgx_iterations"] is not None:
            history = []
            first = max(0, info["amgx_iterations"] - 63)
            for iteration in range(first, info["amgx_iterations"] + 1):
                try:
                    history.append(float(solver.get_residual(iteration)))
                except Exception:
                    history = []
                    break
            info["residual_history"] = tuple(history)
    finally:
        for obj in (solver, mat, vec_x, vec_b, rsrc, cfg):
            if obj is not None:
                try:
                    obj.destroy()
                except AttributeError:
                    pass
    return (x_cp, info) if return_info else x_cp


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


def diagonal_scale_cupy_csr_rows_in_place(matrix, rhs):
    """Apply left Jacobi row scaling to a CuPy CSR matrix and RHS in place."""
    cupy = require_cupy()
    device_id = int(cupy.cuda.runtime.getDevice())
    kernel = _CSR_ROW_SCALE_KERNELS.get(device_id)
    if kernel is None:
        kernel = real_raw_kernel(_CSR_ROW_SCALE_SOURCE, "diagonal_scale_csr_rows")
        _CSR_ROW_SCALE_KERNELS[device_id] = kernel
    nrows = int(rhs.size)
    diagonal = cupy.empty(nrows, dtype=REAL_DTYPE)
    if nrows:
        kernel(
            (nrows,),
            (128,),
            (
                matrix.indptr,
                matrix.indices,
                matrix.data,
                rhs,
                diagonal,
                np.int64(nrows),
            ),
        )
    return diagonal


_CSR_SYMMETRIC_DIAGONAL_SOURCE = r"""
extern "C" __global__ void csr_inverse_sqrt_diagonal(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data,
        double* __restrict__ inverse_sqrt_diagonal,
        const long long nrows)
{
    const long long row = blockIdx.x;
    if (row >= nrows || threadIdx.x != 0) {
        return;
    }
    const int start = indptr[row];
    const int end = indptr[row + 1];
    double value = 0.0;
    for (int p = start; p < end; ++p) {
        if (indices[p] == (int)row) {
            value += data[p];
        }
    }
    if (value < 0.0) {
        value = -value;
    }
    if (value == 0.0) {
        value = 1.0;
    }
    inverse_sqrt_diagonal[row] = 1.0 / sqrt(value);
}

extern "C" __global__ void symmetric_scale_csr_rows(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        double* __restrict__ data,
        double* __restrict__ rhs,
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
    if (threadIdx.x == 0) {
        rhs[row] *= row_scale;
    }
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        data[p] *= row_scale * inverse_sqrt_diagonal[indices[p]];
    }
}
"""
_CSR_SYMMETRIC_SCALE_KERNELS: dict[int, tuple[Any, Any]] = {}


def symmetric_scale_cupy_csr_in_place(matrix, rhs):
    """Apply symmetric Jacobi scaling ``D^-1/2 A D^-1/2`` to CSR/RHS in place.

    The returned vector is ``D^-1/2``. After solving the scaled system for
    ``y``, recover the physical unknown by multiplying ``x = D^-1/2 y``.
    """
    cupy = require_cupy()
    device_id = int(cupy.cuda.runtime.getDevice())
    kernels = _CSR_SYMMETRIC_SCALE_KERNELS.get(device_id)
    if kernels is None:
        diag_kernel = real_raw_kernel(_CSR_SYMMETRIC_DIAGONAL_SOURCE, "csr_inverse_sqrt_diagonal")
        scale_kernel = real_raw_kernel(_CSR_SYMMETRIC_DIAGONAL_SOURCE, "symmetric_scale_csr_rows")
        kernels = (diag_kernel, scale_kernel)
        _CSR_SYMMETRIC_SCALE_KERNELS[device_id] = kernels
    diag_kernel, scale_kernel = kernels
    nrows = int(rhs.size)
    inverse_sqrt_diagonal = cupy.empty(nrows, dtype=REAL_DTYPE)
    if nrows:
        diag_kernel(
            (nrows,),
            (128,),
            (matrix.indptr, matrix.indices, matrix.data, inverse_sqrt_diagonal, np.int64(nrows)),
        )
        scale_kernel(
            (nrows,),
            (128,),
            (matrix.indptr, matrix.indices, matrix.data, rhs, inverse_sqrt_diagonal, np.int64(nrows)),
        )
    return inverse_sqrt_diagonal


__all__ = [
    "CupyAdvectionTraceAssembly",
    "CupyDGMesh",
    "CupyDGSpace",
    "CupyReferenceElementData",
    "assemble_advection_reaction_trace_system_cupy",
    "assemble_advection_reaction_trace_system_eliminated_cupy",
    "element_traces_cupy",
    "expand_boundary_trace_cupy",
    "expand_known_dofs_cupy",
    "reconstruct_advection_reaction_field_cupy",
    "as_cupy_space",
    "as_cupy_coefficients",
    "as_cupy_vector_coefficients",
    "asnumpy",
    "clear_cupy_space_cache",
    "field_from_cupy_coefficients",
    "initialize_pyamgx_once",
    "default_pyamgx_config",
    "diagonal_scale_cupy_csr_rows_in_place",
    "device_arrays_overlap",
    "require_cupy",
    "require_cupy_device",
    "require_cupyx_sparse",
    "require_cupyx_sparse_linalg",
    "require_pyamgx",
    "build_cupyx_ilu_preconditioner",
    "build_cupyx_exported_host_ilu_preconditioner",
    "scipy_coo_to_cupy_csr",
    "scipy_csr_to_cupy",
    "solve_batched_vectors",
    "solve_cupyx_csr",
    "solve_pyamgx_csr",
    "symmetric_scale_cupy_csr_in_place",
]
