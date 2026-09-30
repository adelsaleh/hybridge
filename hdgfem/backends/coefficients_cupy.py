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

from hdgfem.precision import REAL_DTYPE

from hdgfem.core.space import DGField, DGSpace, DGTraceSpace
from hdgfem.backends.cupy import _normalize_cupy_values, as_cupy_coefficients, as_cupy_space, require_cupy


def mapped_face_points_cupy(space: DGSpace, trace_space: DGTraceSpace):
    """Physical trace points, shape ``(K, 3*nfq, 2)``, flat index ``q*3 + face`` (host layout)."""
    from hdgfem.assembly.matrices_numpy import _reference_edge_points_from_1d

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
    from hdgfem.assembly.matrices_numpy import _basis_on_test_faces

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
    from hdgfem.solvers.diffusion_reaction import _component_quadrature_values

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
    from hdgfem.assembly.matrices_numpy import _evaluate_face_callable, _face_quadrature_values_from_scalar_input

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
