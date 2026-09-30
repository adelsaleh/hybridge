"""Device (CuPy) sampling of scalar coefficients on volume and element-side quadrature.

Device twins of the host samplers
``solvers.diffusion_reaction._component_quadrature_values`` (volume points,
shape ``(K, nq)``) and ``assembly.matrices_numpy._face_quadrature_values_from_scalar_input``
(element-side trace points, shape ``(K, 3, nfq)``), with the same accepted forms
and layouts:

* scalars fill exactly; DG fields on any space of the same mesh are contracted
  with that space's reference tables (constant fields fill exactly);
* callables receive CuPy coordinates (face laws also receive the device
  ``element``/``local_face``/``normal`` context of ``_evaluate_face_callable``);
  a callable that cannot handle CuPy input raises ``TypeError``, which callers
  may catch to fall back to the host sampler;
* device arrays already on the target quadrature pass through; other
  precomputed arrays are normalized by the host sampler and uploaded.

Results use the package precision (``REAL_DTYPE``) and stay on the device.
"""
from __future__ import annotations

import numpy as np

from hdgfem.runtime.precision import REAL_DTYPE

from hdgfem.core.space import DGField, DGSpace, DGTraceSpace
from hdgfem.core.device import (
    _normalize_cupy_values,
    as_cupy_coefficients,
    as_cupy_space,
)
from hdgfem.runtime.optional import require_cupy

import time
from hdgfem.core.device import CupyDGSpace
from hdgfem.core.device import mapped_quads_cupy
from hdgfem.runtime.logging import sync_elapsed



def mapped_face_points_cupy(space: DGSpace, trace_space: DGTraceSpace):
    """Physical trace points, shape ``(K, 3*nfq, 2)``, flat index ``q*3 + face`` (host layout)."""
    from hdgfem.core.quadrature import _reference_edge_points_from_1d

    cp = require_cupy()
    mesh = as_cupy_space(space).mesh
    reference = cp.asarray(_reference_edge_points_from_1d(trace_space.quads).reshape(-1, 2), dtype=REAL_DTYPE)
    return cp.einsum("Kdc,pc->Kpd", mesh.aff_mats, reference) + mesh.aff_vecs[:, None, :]


def field_on_volume_cupy(field: DGField, space: DGSpace):
    """Evaluate ``field`` on ``space`` volume quadrature points, shape ``(K, nq)``."""
    cp = require_cupy()
    field.space.assert_same_mesh(space)
    shape = (space.mesh.num_tri, space.quad_data.Krf_w.size)
    if field.constant_value is not None:
        return cp.full(shape, float(field.constant_value), dtype=REAL_DTYPE)
    if field.space is space:
        table = as_cupy_space(space).quad_data.bas_of_quads
    else:
        table = cp.asarray(field.space.basis_at(space.quad_data.Krf_quads).T, dtype=REAL_DTYPE)
    return cp.ascontiguousarray(as_cupy_coefficients(field, as_cupy_space(field.space)) @ table)


def field_on_faces_cupy(field: DGField, space: DGSpace, trace_space: DGTraceSpace):
    """Evaluate ``field`` on the element-side trace quadrature, shape ``(K, 3, nfq)``."""
    from hdgfem.hdg.coefficients import _basis_on_test_faces

    cp = require_cupy()
    field.space.assert_same_mesh(space)
    nfq = trace_space.weights.size
    shape = (space.mesh.num_tri, 3, nfq)
    if field.constant_value is not None:
        return cp.full(shape, float(field.constant_value), dtype=REAL_DTYPE)
    table = np.asarray(_basis_on_test_faces(field.space, space, trace_space=trace_space))  # (3, nb, nfq)
    # One (nb x 3*nfq) GEMM for all faces instead of a per-face contraction.
    table = cp.asarray(table.transpose(1, 0, 2).reshape(table.shape[1], 3 * nfq), dtype=REAL_DTYPE)
    coefficients = as_cupy_coefficients(field, as_cupy_space(field.space))
    return cp.ascontiguousarray((coefficients @ table).reshape(shape))


def volume_samples_cupy(value, space: DGSpace, *, label: str):
    """Device twin of ``_component_quadrature_values``: samples of shape ``(K, nq)``."""
    from hdgfem.hdg.coefficients import _component_quadrature_values

    cp = require_cupy()
    num_elements, num_points = space.mesh.num_tri, space.quad_data.Krf_w.size
    if np.isscalar(value):
        return cp.full((num_elements, num_points), float(value), dtype=REAL_DTYPE)
    if isinstance(value, DGField):
        return field_on_volume_cupy(value, space)
    if isinstance(value, cp.ndarray):
        if value.shape == (num_elements, num_points):
            return cp.ascontiguousarray(value, dtype=REAL_DTYPE)
        raise ValueError(f"{label} device volume samples must have shape ({num_elements}, {num_points})")
    if callable(value):
        points = as_cupy_space(space).mapped_quads  # (K, 2, nq)
        return _normalize_cupy_values(value(points[:, 0], points[:, 1]), num_elements, num_points, label)
    return cp.asarray(_component_quadrature_values(value, space, label=label), dtype=REAL_DTYPE)


def face_samples_cupy(value, space: DGSpace, *, label: str, trace_space: DGTraceSpace, t=None):
    """Device twin of ``_face_quadrature_values_from_scalar_input``: shape ``(K, 3, nfq)``."""
    from hdgfem.hdg.coefficients import (
            _evaluate_face_callable,
            _face_quadrature_values_from_scalar_input,
        )

    cp = require_cupy()
    num_elements, nfq = space.mesh.num_tri, trace_space.weights.size
    if np.isscalar(value):
        return cp.full((num_elements, 3, nfq), float(value), dtype=REAL_DTYPE)
    if isinstance(value, DGField):
        return field_on_faces_cupy(value, space, trace_space)
    if isinstance(value, cp.ndarray):
        if value.shape == (num_elements, 3):
            return cp.broadcast_to(value[..., None], (num_elements, 3, nfq))
        if value.shape == (num_elements, 3, nfq):
            return value
        raise ValueError(f"{label} device face samples must have shape (K, 3) or (K, 3, nfq)")
    if callable(value):
        points = mapped_face_points_cupy(space, trace_space)
        raw = _evaluate_face_callable(value, points, nfq, normals=as_cupy_space(space).mesh.normals, t=t)
        flat = _normalize_cupy_values(raw, num_elements, 3 * nfq, label)
        return cp.ascontiguousarray(flat.reshape(num_elements, nfq, 3).transpose(0, 2, 1))
    return cp.asarray(_face_quadrature_values_from_scalar_input(
        value, space, label, trace_space=trace_space, t=t), dtype=REAL_DTYPE)


__all__ = [
    "face_samples_cupy",
    "field_on_faces_cupy",
    "field_on_volume_cupy",
    "mapped_face_points_cupy",
    "volume_samples_cupy",
]


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
