"""Small reference maps for conservative flux recovery on affine triangles.

The raw P_k flux already meets all interior moment constraints. Only the
stabilization jump needs lifting. RT uses a reference Piola lift; L2-closest
adds the metric-minimizing correction in the constraint nullspace (dimension
k), avoiding a per-element system with O(k**2) constraints.
"""
from dataclasses import dataclass
import numpy as np

from hdgfem.runtime.precision import REAL_DTYPE


@dataclass
class FluxRecoveryReference:
    post_space: object
    embedding: np.ndarray
    face_moments: np.ndarray
    trace_moments: np.ndarray
    lift: np.ndarray
    nullspace: np.ndarray
    gram: np.ndarray
    cross: np.ndarray


def build_flux_recovery_reference(space, trace_space, *, l2_closest=False):
    """Build geometry-independent maps; no per-element factors or JIT calls."""
    from hdgfem.mixed.postprocess.flux import (
            _new_hdg_postprocess_cache,
            _edge_legendre_basis,
            _trace_basis_at,
        )
    cache = _new_hdg_postprocess_cache(space, trace_space)
    post = cache.post_space
    q = post.quad_data
    n, b, e = post.el_dof, space.el_dof, space.order + 1
    # Build the tiny reference algebra in double precision, then cast once.
    phi = np.asarray(q.phi, dtype=np.float64)
    weights = np.asarray(q.Krf_w, dtype=np.float64)
    mass = phi.T @ (weights[:, None] * phi)
    project = np.linalg.solve(mass, phi.T * weights)
    embedding = project @ space.basis_at(q.Krf_quads)
    face_basis = post.basis_at(q.pts_fc.reshape(-1, 2)).reshape(-1, 3, n).transpose(1, 2, 0)
    base_face = space.basis_at(q.pts_fc.reshape(-1, 2)).reshape(-1, 3, b).transpose(1, 2, 0)
    tests = _edge_legendre_basis(space.order, q.quads_JGL)
    face_moments = np.einsum('q,fiq,aq->fai', q.weights_JGL, base_face, tests)
    trace_moments = np.einsum('q,iq,aq->ai', q.weights_JGL, _trace_basis_at(trace_space, q.quads_JGL), tests)
    # n_ref ds_ref = J.T n_phys ds_phys / det(J), in the package face order.
    mesh = space.mesh
    normal_measure = np.einsum('ij,fi,f->fj', mesh.aff_mats[0], mesh.normals[0], mesh.jacs_el_fc[0]) / mesh.aff_jacs[0]
    face_post = np.einsum('q,fiq,aq->fai', q.weights_JGL, face_basis, tests)
    face_constraint = np.concatenate([face_post * normal_measure[:, c, None, None] for c in range(2)], axis=2).reshape(3*e, 2*n)
    low = np.asarray(cache.interior_low_to_post, dtype=np.float64)
    interior = np.zeros((2*len(low), 2*n))
    interior[:len(low), :n] = low
    interior[len(low):, n:] = low
    constraints = np.vstack((face_constraint, interior))
    span = np.zeros((2*n, 2*b+e))
    span[:n, :b] = embedding
    span[n:, b:2*b] = embedding
    r, s = np.asarray(q.Krf_quads, dtype=np.float64).T
    for a in range(e):
        homogeneous = r**a * s**(space.order-a)
        span[:n, 2*b+a] = project @ (r*homogeneous)
        span[n:, 2*b+a] = project @ (s*homogeneous)
    targets = np.zeros((len(constraints), 3*e))
    targets[:3*e] = np.eye(3*e)
    lift = span @ np.linalg.solve(constraints @ span, targets)
    nullspace = np.empty((2*n, 0))
    gram = np.empty((3, 0, 0))
    cross = np.empty((3, 0, 3*e))
    if l2_closest:
        full_face = np.asarray(q.face_element_test_trace_trial, dtype=np.float64)
        full_constraint = np.concatenate([full_face * normal_measure[:, c, None, None] for c in range(2)], axis=1).transpose(0, 2, 1).reshape(3*(e+1), 2*n)
        full_constraint = np.vstack((full_constraint, interior))
        scaled = full_constraint / np.linalg.norm(full_constraint, axis=1)[:, None]
        _, singular, vh = np.linalg.svd(scaled, full_matrices=True)
        if singular[-1] < 1.e-12 * singular[0]:
            raise ArithmeticError('Rank-deficient flux recovery constraints')
        nullspace = vh[len(full_constraint):].T.copy()
        x, y = nullspace[:n], nullspace[n:]
        lx, ly = lift[:n], lift[n:]
        gram = np.stack((x.T@mass@x, x.T@mass@y+y.T@mass@x, y.T@mass@y))
        cross = np.stack((x.T@mass@lx, x.T@mass@ly+y.T@mass@lx, y.T@mass@ly))
    cast = lambda value: np.ascontiguousarray(value, dtype=REAL_DTYPE)
    return FluxRecoveryReference(post, *(cast(v) for v in (
        embedding, face_moments, trace_moments, lift, nullspace, gram, cross)))
