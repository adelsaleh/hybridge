"""hdgfem.core.device."""

from __future__ import annotations

import numpy as np
from typing import Any
from collections.abc import Callable
from hdgfem.core.space import DGField, DGSpace, DGTraceSpace, VectorDGField
from hdgfem.runtime.precision import REAL_DTYPE
from dataclasses import dataclass, field
from hdgfem.runtime.optional import require_cupy


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
        from hdgfem.runtime.logging import timed_section

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
        """Convert host-side data to device representation."""
        cp = require_cupy()
        return cls(
            host=trace_space,
            device_id=int(device_id),
            kind=trace_space.kind,
            nodal=bool(trace_space.nodal),
            interpolation_nodes=cp.asarray(trace_space.interpolation_nodes, dtype=REAL_DTYPE),
            quads=cp.asarray(trace_space.quads, dtype=REAL_DTYPE),
            weights=cp.asarray(trace_space.weights, dtype=REAL_DTYPE),
            bas_of_bd_quads=cp.asarray(trace_space.bas_of_bd_quads, dtype=REAL_DTYPE),
            bas1d_of_ref_edg_qds=cp.asarray(trace_space.bas1d_of_ref_edg_qds, dtype=REAL_DTYPE),
            weighted_bas_of_bd_quads=cp.asarray(trace_space.weighted_bas_of_bd_quads, dtype=REAL_DTYPE),
            weighted_bas1d_of_ref_edg_qds=cp.asarray(trace_space.weighted_bas1d_of_ref_edg_qds, dtype=REAL_DTYPE),
            face_trace_test_element_trial_oriented=cp.asarray(
                trace_space.face_trace_test_element_trial_oriented,
                dtype=REAL_DTYPE,
            ),
            M_rf_fc=cp.asarray(trace_space.M_rf_fc, dtype=REAL_DTYPE),
        )

    @property
    def oriented_basis_table(self):
        """Cached compact orientation tables owned by the host trace formalism."""
        cached = getattr(self, "_oriented_basis_table", None)
        if cached is None:
            cp = require_cupy()
            with cp.cuda.Device(self.device_id):
                cached = cp.asarray(self.host.oriented_basis_table)
            object.__setattr__(self, "_oriented_basis_table", cached)
        return cached

    @property
    def mass_inverse(self):
        """Cached device mirror of the trace mass inverse."""
        inverse = getattr(self, "_mass_inverse", None)
        if inverse is None:
            cp = require_cupy()
            with cp.cuda.Device(self.device_id):
                inverse = cp.asarray(self.host.mass_inverse)
            object.__setattr__(self, "_mass_inverse", inverse)
        return inverse

    @property
    def edg_dof(self) -> int:
        """Return the number of trace degrees of freedom per edge."""
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


def mapped_quads_cupy(cspace: CupyDGSpace):
    """Return physical volume quadrature points resident on the device."""
    return cspace.mapped_quads
