"""Device-resident CuPy assembly of the statically condensed ADR HDG system.

The host-prep CuPy route of :func:`hdgfem.solvers.adv_diff_rea.assemble_advection_diffusion_reaction`
samples coefficients and builds every local matrix with NumPy, builds face
topology and the global contribution map with Python loops over elements and
faces, and eliminates Dirichlet faces with NumPy. This module performs all of
that work on the GPU:

* coefficient sampling at mapped volume, face and boundary quadrature points
  (callables are evaluated with CuPy arrays; a callable that rejects them is
  evaluated on the host for that chunk, and the fallback is recorded);
* local matrices ``A, B, C, D`` and the source moments as batched GEMMs against
  mesh-independent reference tables, plus elementwise kernels for the
  orientation-dependent and geometric terms;
* in-place batched LU inversion (the same cuBLAS ``getrf``/``getri`` calls
  ``cupy.linalg.inv`` makes, without its extra copy of the batch);
* the existing condensation expressions;
* face topology and the deterministic two-contribution gather map from a
  device sort of the element-to-face table, and the existing gather kernel;
* the interior right-hand side, Dirichlet trace projection and elimination.

Element-local work runs in chunks so the transient device footprint is bounded
(``element_chunk``); only per-element condensed blocks, local right-hand sides
and face-sized arrays persist across chunks. Results are returned as host
arrays with the same layout and conventions as the host-prep route.

Every function takes an array module ``xp``. With ``xp=numpy`` the same code
runs on the CPU, which the tests use to check the device algorithm (topology,
orientation flips, chunking, elimination) without a GPU.
"""
from __future__ import annotations

import os
from time import perf_counter

import numpy as np

from ..assembly.face_dense import FaceDenseSystem

# Local faces of a triangle other than r, in increasing order (the order in
# which build_face_topology lists an element's other faces in a face row).
_OTHER_FACES = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
_DEFAULT_WORKSPACE_BYTES = int(float(os.environ.get("HDGFEM_CUPY_ADR_WORKSPACE_GIB", "2"))*2**30)


def _is_device(xp):
    return xp is not np


def _sync(xp):
    if _is_device(xp):
        xp.cuda.get_current_stream().synchronize()


def _to_host(xp, array, out=None):
    if not _is_device(xp):
        if out is None:
            return np.ascontiguousarray(array)
        out[...] = array
        return out
    if out is None:
        return xp.asnumpy(array)
    xp.ascontiguousarray(array).get(out=out)
    return out


class _Timer:
    def __init__(self, xp, timings):
        self.xp, self.timings = xp, timings

    def add(self, key, start):
        _sync(self.xp)
        self.timings[key] = self.timings.get(key, 0.) + perf_counter()-start
        return perf_counter()


class _Prefault:
    """Write the pages of fresh host outputs in worker threads, chunk by chunk.

    A device-to-host copy into never-touched pageable memory runs at about half
    the speed of one into resident pages, because the kernel faults and zeroes
    each page during the copy. Touching the pages of chunk k+1.. in threads
    while the GPU computes overlaps that cost with device work.
    """

    def __init__(self, arrays, bounds, threads):
        self.futures = []
        self.pool = None
        if threads <= 0:
            return
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(threads)
        for s, e in bounds:
            jobs = []
            for array in arrays:
                for part in np.array_split(np.arange(s, e), threads):
                    if part.size:
                        jobs.append(self.pool.submit(array[part[0]:part[-1]+1].fill, 0.))
            self.futures.append(jobs)

    def wait(self, index):
        if self.pool is not None:
            for future in self.futures[index]:
                future.result()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


class _Sampler:
    """Evaluate user coefficient data at device points, with a host fallback."""

    def __init__(self, xp, fallbacks):
        self.xp, self.fallbacks = xp, fallbacks

    def callable_values(self, function, x, y, label):
        xp = self.xp
        if _is_device(xp):
            try:
                value = function(x, y)
                if isinstance(value, xp.ndarray) or np.ndim(value) == 0:
                    return self.normalize(value, x.shape, label)
                if isinstance(value, np.ndarray):
                    return self.normalize(xp.asarray(value), x.shape, label)
            except Exception:  # Callables written for NumPy only; evaluate those on the host.
                pass
            if label not in self.fallbacks:
                self.fallbacks.append(label)
            value = function(xp.asnumpy(x), xp.asnumpy(y))
            return self.normalize(xp.asarray(np.asarray(value, dtype=np.float64)), x.shape, label)
        return self.normalize(function(x, y), x.shape, label)

    def normalize(self, values, shape, label):
        """Scalar, one value per reference point, or one per (element, point)."""
        xp = self.xp
        if np.ndim(values) == 0:
            return xp.full(shape, float(values), dtype=np.float64)
        values = xp.asarray(values, dtype=np.float64)
        if values.shape == shape:
            return values
        if values.shape == shape[1:]:
            return xp.broadcast_to(values[None], shape)
        raise ValueError(f"{label} must return a scalar, shape {shape[1:]}, or shape {shape}; got {values.shape}")


class _Reference:
    """Mesh-independent reference tables, uploaded once per assembly."""

    def __init__(self, xp, space):
        from ..solvers.diff_rea import _reference_derivative_matrices
        q = space.quad_data
        P, F = space.el_dof, q.edg_dof
        Q, Qf = q.Krf_w.size, q.weights_JGL.size
        self.P, self.F, self.Q, self.Qf = P, F, Q, Qf
        phi_f, psi = q.bas_of_bd_quads, q.bas1d_of_ref_edg_qds  # (3,P,Qf), (F,Qf)
        d0, d1 = _reference_derivative_matrices(space)
        # Advection: X(K,q,D) @ Z(q,D,ij), Z = dphi[D,i,q] phi[j,q] w[q] (the host einsum's last pairwise step).
        z = np.einsum('Diq,jq,q->qDij', q.dbas_of_quads, q.bas_of_quads, q.Krf_w).reshape(2*Q, P*P)
        host = dict(
            d0=d0, d1=d1,
            face_mass=q.face_element_test_element_trial.reshape(3, P*P),
            face_trace=np.ascontiguousarray(q.face_element_test_trace_trial.swapaxes(0, 1)),  # (P,3,F)
            trace_lift=q.face_trace_test_element_trial_oriented,  # (6,F,P)
            advection=z,
            upwind_mass=np.einsum('fiq,fjq->fqij', phi_f, phi_f).reshape(3*Qf, P*P),
            upwind_trace=np.einsum('fiq,jq->fqij', phi_f, psi).reshape(3, Qf, P*F),
            upwind_lift=np.einsum('iq,fjq->fqij', psi, phi_f).reshape(3, Qf, F*P),
            trace_mass=np.einsum('iq,jq->qij', psi, psi).reshape(Qf, F*F),
            weighted_phi_phi=q.weighted_phi_phi_flat,  # (Q,P*P)
            weighted_phi=q.weighted_phi,  # (Q,P)
            mass=q.MKrf,
            volume_points=q.Krf_quads,  # (Q,2)
            face_points=q.pts_fc.reshape(-1, 2),  # (Qf*3,2), flat index q*3+f
            weights_face=q.weights_JGL,
            boundary_basis=np.ascontiguousarray(q.bas1d_of_ref_edg_qds.T),  # (Qf,F)
            boundary_mass_inverse=np.linalg.inv(q.M_rf_fc),
        )
        for key, value in host.items():
            setattr(self, key, xp.asarray(np.ascontiguousarray(value, dtype=np.float64)))
        self.boundary_points_1d = np.asarray(q.quads_JGL, dtype=np.float64)


class _Mesh:
    """Device copies of the per-element geometry the assembly reads."""

    def __init__(self, xp, mesh):
        get = lambda name, dtype=np.float64: xp.asarray(np.ascontiguousarray(getattr(mesh, name), dtype=dtype))
        self.aff_mats, self.aff_vecs, self.aff_jacs = get('aff_mats'), get('aff_vecs'), get('aff_jacs')
        self.inv_aff_mats_t, self.normals, self.jacs = get('inv_aff_mats_t'), get('normals'), get('jacs_el_fc')
        self.orientations = get('orientations', bool)
        self.coupling = get('loc2oriented_face_coupling', np.int64)
        self.loc2glob = get('loc2glob_edge', np.int64)
        self.free_faces = get('int_edges_inds', np.int64)
        self.num_elements = int(mesh.num_tri)


def _map_points(xp, mesh, s, e, ref):
    """Physical coordinates (x, y), each (m, len(ref)), of reference points in elements s:e."""
    a, v = mesh.aff_mats[s:e], mesh.aff_vecs[s:e]
    xi, eta = ref[:, 0], ref[:, 1]
    x = a[:, 0, 0, None]*xi + a[:, 0, 1, None]*eta + v[:, 0, None]
    y = a[:, 1, 0, None]*xi + a[:, 1, 1, None]*eta + v[:, 1, None]
    return x, y


def build_device_face_topology(xp, loc2glob):
    """Face rows and the deterministic gather map from a device sort of ``loc2glob``.

    Returns ``(neighbors (NF,5) int64, element_ids, row_local_faces,
    column_local_faces (NF,5,2) int32, incidence (NF,) int64, first, second)``,
    where ``first``/``second`` are the flat (element*3+local face) positions of
    each face's side-0/side-1 incidence (``second == first`` on boundary faces),
    equal to
    :func:`build_face_topology` ``.neighbors`` and
    :func:`prepare_face_assembly_contributions` with the interior-face row mask.
    Side 0 of a face is its lower-numbered element, as in the host loops.
    """
    ne = loc2glob.shape[0]
    if loc2glob.ndim != 2 or loc2glob.shape[1] != 3 or ne == 0:
        raise ValueError('device face topology requires a non-empty (NE, 3) triangle face table')
    flat = loc2glob.reshape(-1)
    lo, hi = (int(v) for v in _to_host(xp, xp.stack((flat.min(), flat.max()))))
    if lo < 0:
        raise ValueError('loc2glob_face cannot contain negative face ids')
    nf = hi+1
    count = xp.bincount(flat, minlength=nf)
    cmin, cmax = (int(v) for v in _to_host(xp, xp.stack((count.min(), count.max()))))
    if cmin < 1:
        raise ValueError('unused global face ids detected')
    if cmax > 2:
        raise ValueError('a global face belongs to more than two elements; the fixed manifold-face layout is not applicable')
    slots = xp.arange(3*ne, dtype=np.int64)
    order = xp.argsort(flat*(3*ne)+slots)  # unique keys: by face, then element, as the host loops
    start = xp.cumsum(count)-count
    interior = count == 2
    first = order[start]
    second = xp.where(interior, order[xp.minimum(start+1, 3*ne-1)], first)
    e0, l0, e1, l1 = first//3, first % 3, second//3, second % 3
    others = xp.asarray(_OTHER_FACES)
    o0, o1 = others[l0], others[l1]  # (NF,2)
    faces = xp.arange(nf, dtype=np.int64)
    neighbors = xp.full((nf, 5), -1, dtype=np.int64)
    neighbors[:, 0] = faces
    neighbors[:, 1:3] = loc2glob[e0[:, None], o0]
    neighbors[:, 3:5] = xp.where(interior[:, None], loc2glob[e1[:, None], o1], -1)
    # On a conforming mesh the four off-diagonal faces of an interior row are distinct.
    row = neighbors[interior]
    duplicates = xp.zeros(1, dtype=bool)
    for i in range(5):
        for j in range(i+1, 5):
            duplicates = duplicates | xp.any(row[:, i] == row[:, j])
    if bool(_to_host(xp, duplicates)[0]):
        raise ValueError('two elements share more than one face; the fixed five-slot face layout is not applicable')
    element_ids = xp.full((nf, 5, 2), -1, dtype=np.int32)
    row_local = xp.full((nf, 5, 2), -1, dtype=np.int32)
    column_local = xp.full((nf, 5, 2), -1, dtype=np.int32)
    i = xp.nonzero(interior)[0]
    element_ids[i, 0, 0], element_ids[i, 0, 1] = e0[i], e1[i]
    row_local[i, 0, 0], row_local[i, 0, 1] = l0[i], l1[i]
    column_local[i, 0, 0], column_local[i, 0, 1] = l0[i], l1[i]
    element_ids[i, 1:3, 0], element_ids[i, 3:5, 0] = e0[i, None], e1[i, None]
    row_local[i, 1:3, 0], row_local[i, 3:5, 0] = l0[i, None], l1[i, None]
    column_local[i, 1:3, 0], column_local[i, 3:5, 0] = o0[i], o1[i]
    return neighbors, element_ids, row_local, column_local, count, first, second


def _gather_global_blocks(xp, element_blocks, element_ids, row_local, column_local, block_size):
    nf = element_ids.shape[0]
    if _is_device(xp):
        from .cupy_assembly import CuPyGlobalFaceAssembler
        assembler = CuPyGlobalFaceAssembler(element_ids=element_ids, row_local_faces=row_local,
            column_local_faces=column_local, num_elements=element_blocks.shape[0], num_local_faces=3,
            block_size=block_size, dtype=np.float64, device_id=int(xp.cuda.Device().id))
        return assembler.assemble(element_blocks)
    out = np.zeros((nf, 5, block_size, block_size))
    for c in range(2):  # Same order as the kernel: 0 + first + second.
        valid = element_ids[..., c] >= 0
        e, r, k = element_ids[..., c][valid], row_local[..., c][valid], column_local[..., c][valid]
        out[valid] += element_blocks[e, r, k]
    return out


def _batched_inverse_inplace(xp, a):
    """Invert a C-contiguous float64 batch, overwriting ``a`` with its LU factors."""
    if not _is_device(xp):
        return np.linalg.inv(a)
    import cupy
    from cupy.cuda import cublas, device
    from cupy.linalg import _util
    batch, n = a.shape[0], a.shape[1]
    if batch == 0:
        return xp.empty_like(a)
    handle = device.get_cublas_handle()
    step = n*n*8
    a_ptrs = cupy.arange(a.data.ptr, a.data.ptr+step*batch, step, dtype=cupy.uintp)
    pivots = cupy.empty((batch, n), dtype=cupy.int32)
    info = cupy.empty((batch,), dtype=cupy.int32)
    cublas.dgetrfBatched(handle, n, a_ptrs.data.ptr, n, pivots.data.ptr, info.data.ptr, batch)
    _util._check_cublas_info_array_if_synchronization_allowed(cublas.dgetrfBatched, info)
    c = cupy.empty_like(a)
    c_ptrs = cupy.arange(c.data.ptr, c.data.ptr+step*batch, step, dtype=cupy.uintp)
    cublas.dgetriBatched(handle, n, a_ptrs.data.ptr, n, pivots.data.ptr, c_ptrs.data.ptr, n, info.data.ptr, batch)
    _util._check_cublas_info_array_if_synchronization_allowed(cublas.dgetriBatched, info)
    return c


def element_bytes(P, F, Q, Qf):
    """Transient device bytes per element of one chunk (local matrices, inverse, temporaries)."""
    return 8*(2*9*P*P + 2*P*P + 5*9*P*F + 9*F*F + 14*Q + 12*3*Qf + 4*P) + 4*3*P


def default_element_chunk(xp, ne, P, F, Q, Qf, workspace_bytes=None):
    per = element_bytes(P, F, Q, Qf)
    if workspace_bytes is None:
        workspace_bytes = _DEFAULT_WORKSPACE_BYTES
        if _is_device(xp):
            free = xp.cuda.runtime.memGetInfo()[0]+xp.get_default_memory_pool().free_bytes()
            workspace_bytes = min(workspace_bytes, int(0.5*free))
    return int(max(1, min(ne, workspace_bytes//per)))


def _diffusion_values(xp, diffusion, space, sample, x, y, s, e):
    """k00, k01, k10, k11 on volume points of elements s:e (shape (m,Q) or broadcastable scalars)."""
    from ..solvers.diff_rea import _component_quadrature_values
    from ..core.space import DGField
    m, Q = x.shape
    full = lambda value: xp.full((m, Q), float(value), dtype=np.float64)
    if np.isscalar(diffusion):
        return full(diffusion), full(0.), full(0.), full(diffusion)
    try:
        constant = np.asarray(diffusion, dtype=np.float64)
    except (TypeError, ValueError):
        constant = None
    if constant is not None and constant.shape == (2, 2):
        return full(constant[0, 0]), full(constant[0, 1]), full(constant[1, 0]), full(constant[1, 1])
    if isinstance(diffusion, (tuple, list)):
        if len(diffusion) == 3:
            parts = (diffusion[0], diffusion[1], diffusion[1], diffusion[2])
        elif len(diffusion) == 4:
            parts = tuple(diffusion)
        elif len(diffusion) == 2 and all(isinstance(r, (tuple, list)) and len(r) == 2 for r in diffusion):
            parts = (diffusion[0][0], diffusion[0][1], diffusion[1][0], diffusion[1][1])
        else:
            raise ValueError("diffusion must be scalar, 2x2 constant, (k00,k01,k11), or (k00,k01,k10,k11)")
        labels = ('diffusion[0,0]', 'diffusion[0,1]', 'diffusion[1,0]', 'diffusion[1,1]')
        out = []
        for part, label in zip(parts, labels):
            if np.isscalar(part):
                out.append(full(part))
            elif callable(part) and not isinstance(part, DGField):
                out.append(sample.callable_values(part, x, y, label))
            else:  # DGField, DG coefficients or quadrature values: host evaluation of this chunk.
                out.append(xp.asarray(_component_quadrature_values(part, space, label=label)[s:e]))
        return tuple(out)
    raise TypeError("diffusion must be scalar, a constant 2x2 array, or component callables/fields")


def _check_diffusion(xp, k00, k01, k10, k11):
    det = k00*k11-k01*k10
    finite = xp.all(xp.isfinite(k00)) & xp.all(xp.isfinite(k01)) & xp.all(xp.isfinite(k10)) & xp.all(xp.isfinite(k11))
    symmetric = xp.all(xp.abs(k01-k10) <= 1e-14+1e-12*xp.abs(k10))
    flags = _to_host(xp, xp.stack((finite, symmetric, xp.all(k00 > 0), xp.all(det > 0))))
    if not flags.all():
        raise ValueError("diffusion must be symmetric positive definite at quadrature points")
    return det


def _reaction_mass(xp, reaction, space, sample, x, y, mesh, ref, s, e):
    from ..core.space import DGField
    from ..assembly import hdg
    jac = mesh.aff_jacs[s:e]
    if np.isscalar(reaction):
        return float(reaction)*jac[:, None, None]*ref.mass[None]
    if callable(reaction) and not isinstance(reaction, DGField):
        values = sample.callable_values(reaction, x, y, 'reaction')
        return xp.matmul(values*jac[:, None], ref.weighted_phi_phi).reshape(e-s, ref.P, ref.P)
    full = hdg.reaction_mass(reaction, space)  # DGField or arrays: host evaluation, then this chunk.
    return xp.asarray(np.ascontiguousarray(full[s:e]))


def _source_moments(xp, source, space, sample, x, y, mesh, ref, s, e):
    from ..core.space import DGField
    from ..assembly import hdg
    m, P = e-s, ref.P
    rhs = xp.zeros((m, 3*P), dtype=np.float64)
    if callable(source) and not isinstance(source, DGField):
        values = sample.callable_values(source, x, y, 'source')
        moments = values @ ref.weighted_phi
        moments *= mesh.aff_jacs[s:e, None]
        rhs[:, :P] = moments
        return rhs
    full = hdg.block_source_moments(source, space, num_blocks=3, source_block=0)
    return xp.asarray(np.ascontiguousarray(full[s:e]))


def _local_chunk(xp, coefficients, space, mesh, ref, tau_full, sample, s, e, timer):
    """A, B, C, D and source moments for elements s:e, as in prepare_adr_local_matrices."""
    source, beta, reaction, diffusion = coefficients
    m, P, F, Q, Qf = e-s, ref.P, ref.F, ref.Q, ref.Qf
    t = perf_counter()
    tau = tau_full[s:e]
    jac_f = mesh.jacs[s:e]
    normals = mesh.normals[s:e]
    x, y = _map_points(xp, mesh, s, e, ref.volume_points)
    k00, k01, k10, k11 = _diffusion_values(xp, diffusion, space, sample, x, y, s, e)
    det = _check_diffusion(xp, k00, k01, k10, k11)
    fx, fy = _map_points(xp, mesh, s, e, ref.face_points)
    bx = sample.callable_values(beta[0], fx, fy, 'beta[0]').reshape(m, Qf, 3)
    by = sample.callable_values(beta[1], fx, fy, 'beta[1]').reshape(m, Qf, 3)
    bn = bx.transpose(0, 2, 1)*normals[:, :, 0, None] + by.transpose(0, 2, 1)*normals[:, :, 1, None]
    del bx, by, fx, fy
    if not bool(_to_host(xp, xp.all(xp.isfinite(bn)))):
        raise ValueError('nonfinite velocity on faces')
    up = xp.maximum(bn, 0.)
    beta_volume = xp.stack((sample.callable_values(beta[0], x, y, 'beta[0]'),
                            sample.callable_values(beta[1], x, y, 'beta[1]')), axis=-1)  # (m,Q,2)
    t = timer.add('device_coefficient_sampling', t)

    a = xp.empty((m, 3*P, 3*P), dtype=np.float64)
    v = a.reshape(m, 3, P, 3, P)
    weights = jac_f[..., None]*ref.weights_face  # (m,3,Qf)
    # A00 = (reaction mass + sum_f tau|f| M_f) - advection + upwind face mass.
    block = _reaction_mass(xp, reaction, space, sample, x, y, mesh, ref, s, e).reshape(m, P*P)
    block = block + (tau*jac_f) @ ref.face_mass
    scaled = mesh.aff_jacs[s:e, None, None]*mesh.inv_aff_mats_t[s:e]  # (m,2,2)
    flow = beta_volume[..., 0, None]*scaled[:, None, 0, :] + beta_volume[..., 1, None]*scaled[:, None, 1, :]
    del beta_volume
    block -= flow.reshape(m, 2*Q) @ ref.advection
    del flow
    block += (weights*up).reshape(m, 3*Qf) @ ref.upwind_mass
    v[:, 0, :, 0, :] = block.reshape(m, P, P)
    am = mesh.aff_mats[s:e]
    d0 = am[:, 1, 1, None, None]*ref.d0[None] - am[:, 1, 0, None, None]*ref.d1[None]
    d1 = -am[:, 0, 1, None, None]*ref.d0[None] + am[:, 0, 0, None, None]*ref.d1[None]
    v[:, 0, :, 1, :] = ((jac_f*normals[..., 0]) @ ref.face_mass).reshape(m, P, P) - d0
    v[:, 0, :, 2, :] = ((jac_f*normals[..., 1]) @ ref.face_mass).reshape(m, P, P) - d1
    v[:, 1, :, 0, :], v[:, 2, :, 0, :] = d0, d1
    del d0, d1
    # Inverse-diffusion mass blocks G_ab = int (kappa^-1)_ab phi_i phi_j.
    jac = mesh.aff_jacs[s:e, None]
    for (r, c), values in (((1, 1), k11/det), ((1, 2), -k01/det), ((2, 1), -k10/det), ((2, 2), k00/det)):
        values *= jac
        v[:, r, :, c, :] = -(values @ ref.weighted_phi_phi).reshape(m, P, P)
    del k00, k01, k10, k11, det, values, block
    t = timer.add('device_local_matrix_a', t)

    # B: diffusion trace coupling plus the upwind (up - beta.n) face term on the u rows.
    b = xp.empty((m, 3*P, 3*F), dtype=np.float64)
    br = b.reshape(m, 3, P, 3, F)
    br[:, 0] = (tau*jac_f)[:, None, :, None]*ref.face_trace[None]
    br[:, 1] = (normals[..., 0][:, None, :, None]*jac_f[:, None, :, None])*ref.face_trace[None]
    br[:, 2] = (normals[..., 1][:, None, :, None]*jac_f[:, None, :, None])*ref.face_trace[None]
    upwind = xp.matmul((weights*(up-bn)).transpose(1, 0, 2), ref.upwind_trace)  # (3,m,P*F)
    br[:, 0] += upwind.reshape(3, m, P, F).transpose(1, 2, 0, 3)
    del upwind
    # C: trace lift (global orientation) plus the upwind term on the u columns.
    restriction = ref.trace_lift[mesh.coupling[s:e]]  # (m,3,F,P)
    restriction *= jac_f[..., None, None]
    c = xp.empty((m, 3, F, 3*P), dtype=np.float64)
    c[..., :P] = tau[..., None, None]*restriction
    c[..., P:2*P] = normals[..., 0, None, None]*restriction
    c[..., 2*P:] = normals[..., 1, None, None]*restriction
    del restriction
    extra = xp.matmul((weights*up).transpose(1, 0, 2), ref.upwind_lift).reshape(3, m, F, P).transpose(1, 0, 2, 3)
    neg = ~mesh.orientations[s:e]
    extra = xp.where(neg[..., None, None], extra[:, :, ::-1, :], extra)
    c[..., :P] += extra
    del extra
    d = ((weights*(tau[..., None]+up-bn)).reshape(m*3, Qf) @ ref.trace_mass).reshape(m, 3, F, F)
    d = xp.where(neg[..., None, None], d[:, :, ::-1, ::-1], d)
    rhs = _source_moments(xp, source, space, sample, x, y, mesh, ref, s, e)
    del x, y, weights, up, bn
    finite = xp.stack([xp.all(xp.isfinite(z)) for z in (a, b, c, d, rhs)])
    if not bool(_to_host(xp, xp.all(finite))):
        raise ValueError('nonfinite local coefficient matrices')
    timer.add('device_local_matrices_bcd', t)
    return a, b, xp.ascontiguousarray(c), xp.ascontiguousarray(d), rhs


def _boundary_trace(xp, boundary_condition, space, ref, sample, nf):
    """Dirichlet data projected on boundary faces, as hdg.boundary_trace_coefficients."""
    mesh = space.mesh
    trace = xp.zeros((nf, ref.F), dtype=np.float64)
    if mesh.bnd_edges_inds.size == 0:
        return trace
    ends = xp.asarray(mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]])  # (Nb,2,2), boundary only
    t = xp.asarray(ref.boundary_points_1d)
    points = 0.5*((1.-t)[None, :, None]*ends[:, 0:1, :] + (1.+t)[None, :, None]*ends[:, 1:2, :])
    values = sample.callable_values(boundary_condition, points[:, :, 0], points[:, :, 1], 'boundary_condition')
    trace[xp.asarray(mesh.bnd_edges_inds)] = ((values*ref.weights_face[None, :]) @ ref.boundary_basis) @ ref.boundary_mass_inverse
    return trace


def _assemble_chunk(xp, coefficients, space, mesh, ref, tau, sample, s, e, timer, prefault, index,
                    inv_host, b_host, rhs_host, a_host, blocks, local_rhs, chunk_hook=None):
    """Local matrices, inverse, host copies (or the device hook) and condensation of elements s:e."""
    m, F = e-s, ref.F
    a, b, c, d, rhs = _local_chunk(xp, coefficients, space, mesh, ref, tau, sample, s, e, timer)
    t = perf_counter()
    if a_host is not None:
        prefault.wait(index)
        _to_host(xp, a, a_host[s:e])
        t = timer.add('device_to_host_local', t)
    inv = _batched_inverse_inplace(xp, a)
    del a
    t = timer.add('local_inverse', t)
    if chunk_hook is not None:
        # Device outputs: the caller derives what it keeps from the local inverse, the
        # boundary matrices and the source moments of this chunk, which are then freed.
        chunk_hook(s, e, inv, b, rhs)
        _sync(xp)
        t = timer.add('device_chunk_hook', t)
    if inv_host is not None:
        prefault.wait(index)
        t = timer.add('host_prefault_wait', t)
        _to_host(xp, inv, inv_host[s:e])
        _to_host(xp, b, b_host[s:e])
        _to_host(xp, rhs, rhs_host[s:e])
        t = timer.add('device_to_host_local', t)
    # Condensation: the expressions of the host-prep route.
    schur = (c @ (inv @ b)[:, None]).reshape(m, 3, F, 3, F)
    ei, fi = xp.nonzero(~mesh.orientations[s:e])
    schur[ei, :, :, fi, :] = schur[ei, :, :, fi, ::-1]
    out = -schur.swapaxes(2, 3)
    diag = xp.arange(3)
    out[:, diag, diag] += d
    blocks[s:e] = out
    local_rhs[s:e] = (c @ (inv @ rhs[..., None])[:, None]).squeeze(-1)
    timer.add('condensation', t)


def assemble_adr_device(source, beta, reaction, boundary_condition, space, *, diffusion=1., stabilization=1.,
                        element_chunk=None, workspace_bytes=None, return_local_matrix=False, prefault_threads=None,
                        timings=None, xp=None, device_outputs=False, chunk_hook=None):
    """Assemble the ADR trace system with all mesh-sized work on the device.

    ``prefault_threads`` host threads (default ``HDGFEM_CUPY_ADR_PREFAULT_THREADS``
    or 8) write the pages of the returned host arrays ahead of each chunk's
    device-to-host copy (0 disables).

    Returns a dict with ``system`` (host :class:`FaceDenseSystem`),
    ``element_blocks``, ``local_solver``, ``boundary``, ``source_rhs`` and
    ``local_matrix`` (``None`` unless ``return_local_matrix``), all host arrays
    laid out as in the host-prep route, and ``info`` (chunking, and the labels
    of coefficient callables that had to be evaluated on the host).

    ``device_outputs=True`` copies nothing back: the local inverse, boundary
    matrices and source moments of each element chunk are handed to
    ``chunk_hook(s, e, inverse, boundary, source_rhs)`` (device arrays, freed
    after the call) and the eliminated trace system and element blocks come
    back as device arrays under ``device`` (``blocks``, ``neighbors``, ``rhs``,
    ``global_faces``, ``global_to_local``, ``boundary_trace``, ``element_blocks``,
    ``loc2glob``, ``orientations``, ``aff_mats``, ``aff_vecs``, ``aff_jacs``);
    ``system`` and the host local arrays are ``None``.
    """
    from ..solvers.diff_rea import _normalize_tau
    from ..solvers.adv_diff_rea import _velocity
    if xp is None:
        from .cupy import require_cupy_device
        xp = require_cupy_device()
    if prefault_threads is None:
        import os
        prefault_threads = int(os.environ.get('HDGFEM_CUPY_ADR_PREFAULT_THREADS', 8))
    timings = {} if timings is None else timings
    timer = _Timer(xp, timings)
    fallbacks = []
    sample = _Sampler(xp, fallbacks)
    _sync(xp)
    t = perf_counter()
    ref = _Reference(xp, space)
    t = timer.add('reference_tables', t)
    mesh = _Mesh(xp, space.mesh)
    if np.isscalar(stabilization):
        if not np.isfinite(stabilization) or stabilization <= 0:
            raise ValueError('diffusive stabilization must be finite and strictly positive')
        tau = xp.full((mesh.num_elements, 3), float(stabilization), dtype=np.float64)
    else:
        tau_host = _normalize_tau(stabilization, space)
        if not np.all(np.isfinite(tau_host)) or np.any(tau_host <= 0):
            raise ValueError('diffusive stabilization must be finite and strictly positive')
        tau = xp.asarray(tau_host)
    beta = _velocity(beta)
    t = timer.add('host_to_device_mesh', t)

    neighbors, element_ids, row_local, column_local, incidence, first, second = build_device_face_topology(
        xp, mesh.loc2glob)
    nf = int(neighbors.shape[0])
    t = timer.add('device_topology', t)

    ne, P, F = mesh.num_elements, ref.P, ref.F
    chunk = element_chunk or default_element_chunk(xp, ne, P, F, ref.Q, ref.Qf, workspace_bytes)
    chunk = int(max(1, min(ne, chunk)))
    if device_outputs:
        inv_host = b_host = rhs_host = a_host = None
        prefault_threads = 0
    else:
        inv_host = np.empty((ne, 3*P, 3*P))
        b_host = np.empty((ne, 3*P, 3*F))
        rhs_host = np.empty((ne, 3*P))
        a_host = np.empty((ne, 3*P, 3*P)) if return_local_matrix else None
    blocks = xp.empty((ne, 3, 3, F, F), dtype=np.float64)
    local_rhs = xp.empty((ne, 3, F), dtype=np.float64)
    coefficients = (source, beta, reaction, diffusion)
    bounds = [(s, min(ne, s+chunk)) for s in range(0, ne, chunk)]
    outputs = [x for x in (inv_host, b_host, rhs_host, a_host) if x is not None]
    prefault = _Prefault(outputs, bounds, prefault_threads if _is_device(xp) else 0)
    t = timer.add('host_output_allocation', t)
    try:
        for index, (s, e) in enumerate(bounds):
            _assemble_chunk(xp, coefficients, space, mesh, ref, tau, sample, s, e, timer, prefault, index,
                            inv_host, b_host, rhs_host, a_host, blocks, local_rhs, chunk_hook)
    finally:
        prefault.close()


    t = perf_counter()
    global_blocks = _gather_global_blocks(xp, blocks, element_ids, row_local, column_local, F)
    del element_ids, row_local, column_local
    t = timer.add('global_blocks', t)
    # Interior RHS: 0 + first + second contribution per interior face, in element order.
    interior_rhs = xp.zeros((nf, F), dtype=np.float64)
    flat_rhs = local_rhs.reshape(3*ne, F)
    interior = xp.nonzero(incidence == 2)[0]
    interior_rhs[interior] = flat_rhs[first[interior]] + flat_rhs[second[interior]]
    del flat_rhs, first, second
    trace = _boundary_trace(xp, boundary_condition, space, ref, sample, nf)
    t = timer.add('device_rhs_and_boundary', t)
    # Dirichlet elimination, slot by slot as eliminate_dirichlet_faces.
    free = mesh.free_faces
    num_free = int(free.shape[0])
    global_to_free = xp.full(nf, -1, dtype=np.int64)
    global_to_free[free] = xp.arange(num_free, dtype=np.int64)
    row_neighbors = neighbors[free]
    row_blocks = global_blocks[free]
    del global_blocks
    reduced_rhs = interior_rhs[free]
    reduced_neighbors = xp.full((num_free, 5), -1, dtype=np.int64)
    for slot in range(5):
        column = row_neighbors[:, slot]
        valid = column >= 0
        safe = xp.where(valid, column, 0)
        local = xp.where(valid, global_to_free[safe], -1)
        free_column = valid & (local >= 0)
        known = valid & (local < 0)
        reduced_neighbors[:, slot] = xp.where(free_column, local, -1)
        moved = (row_blocks[:, slot] @ trace[safe][..., None])[..., 0]
        reduced_rhs = reduced_rhs - xp.where(known[:, None], moved, 0.)
        row_blocks[:, slot] = xp.where(free_column[:, None, None], row_blocks[:, slot], 0.)
    t = timer.add('device_dirichlet_elimination', t)
    info = dict(route='device', element_chunk=chunk, chunks=len(bounds), prefault_threads=prefault_threads,
                chunk_workspace_estimate_bytes=chunk*element_bytes(P, F, ref.Q, ref.Qf),
                host_callable_fallbacks=list(fallbacks))
    if device_outputs:
        del local_rhs, interior_rhs
        device = dict(blocks=row_blocks, neighbors=reduced_neighbors, rhs=reduced_rhs, global_faces=free,
                      global_to_local=global_to_free, boundary_trace=trace, element_blocks=blocks,
                      loc2glob=mesh.loc2glob, orientations=mesh.orientations,
                      aff_mats=mesh.aff_mats, aff_vecs=mesh.aff_vecs, aff_jacs=mesh.aff_jacs)
        _sync(xp)
        timer.add('device_outputs', t)
        info['device_outputs'] = True
        return dict(system=None, element_blocks=None, local_solver=None, boundary=None, source_rhs=None,
                    local_matrix=None, info=info, device=device)
    system = FaceDenseSystem(
        blocks=_to_host(xp, row_blocks), neighbors=_to_host(xp, reduced_neighbors),
        rhs=_to_host(xp, reduced_rhs), global_faces=np.ascontiguousarray(space.mesh.int_edges_inds, dtype=np.int64),
        global_to_local=_to_host(xp, global_to_free), boundary_trace=_to_host(xp, trace), mode='eliminate')
    element_blocks = _to_host(xp, blocks)
    del blocks, local_rhs, row_blocks
    timer.add('device_to_host', t)
    return dict(system=system, element_blocks=element_blocks, local_solver=inv_host, boundary=b_host,
                source_rhs=rhs_host, local_matrix=a_host, info=info)
