"""Solution checks of the ADR campaign workers on the GPU, with the vendored pass criteria.

The vendored ``verify`` (scripts/adr_performance_worker.py) rebuilds the DG space on the host
and checks every solve with NumPy: the face-dense residual, the local reconstruction from the
stored local inverses and the physical L2 error. At 520k triangles, p=9, that is about 83 s
per solve on the host. Here the same quantities are computed on the device:

- residual ``||A x - b|| / ||b||`` with the solver's operator, or a batched face-dense matvec
  over the device blocks;
- the primal unknowns ``u = L_u (s + B t)`` as ``w + M t``, with ``w = L_u s`` and
  ``M = L_u B`` (``L_u`` = the u rows of the local inverse, ``B`` = the element boundary
  matrices, ``s`` = the source moments, ``t`` = oriented element traces). The reassociation
  changes the round-off only;
- the L2 error against the case's exact solution on the mapped volume quadrature points.

``DeviceVerifier.from_device`` takes ``w``, ``M`` and the geometry straight from a device
assembly (``device_system.assemble_on_device``), with no host round trip.
``DeviceVerifier.from_cache`` builds them from a cache written by the host-output route (the
v2 campaign), with ``verify_geometry.npz`` saved next to it by ``save_geometry``.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

GEOMETRY = 'verify_geometry.npz'


def save_geometry(cache, space):
    """Store what the device checks need from the DG space (called with the assembly's space)."""
    mesh, quad = space.mesh, space.quad_data
    np.savez(Path(cache)/GEOMETRY, orientations=np.ascontiguousarray(mesh.orientations),
             aff_mats=mesh.aff_mats, aff_vecs=mesh.aff_vecs, aff_jacs=mesh.aff_jacs,
             bas_of_quads=quad.bas_of_quads, Krf_w=quad.Krf_w, Krf_quads=quad.Krf_quads,
             el_dof=np.int64(space.el_dof), edg_dof=np.int64(quad.edg_dof), num_edg=np.int64(mesh.num_edg))


class FaceDenseMatvec:
    """y = A x for the face-dense trace system on the device (rows chunked)."""

    def __init__(self, cp, system, blocks=None, chunk_rows=1 << 18):
        self.cp = cp
        self.blocks = cp.asarray(np.asarray(system.blocks)) if blocks is None else blocks
        neighbors = cp.asarray(np.asarray(system.neighbors)) if blocks is None else system.neighbors
        self.valid = neighbors >= 0
        self.gather = cp.where(self.valid, neighbors, 0)
        self.rows, self.slots, self.b = self.blocks.shape[0], self.blocks.shape[1], self.blocks.shape[2]
        self.chunk = chunk_rows

    def __call__(self, x):
        cp = self.cp
        x2 = x.reshape(-1, self.b)
        out = cp.empty((self.rows, self.b), dtype=cp.float64)
        for s in range(0, self.rows, self.chunk):
            e = min(s+self.chunk, self.rows)
            xs = x2[self.gather[s:e]] * self.valid[s:e, :, None]
            out[s:e] = cp.matmul(self.blocks[s:e], xs[..., None])[..., 0].sum(axis=1)
        return out.reshape(-1)


class DeviceVerifier:
    def __init__(self, cp, spec, *, rhs, boundary_trace, global_faces, loc2glob, orientations, aff_mats, aff_vecs,
                 aff_jacs, bas, weights, ref_points, w, M, exact, reference=None, matvec=None, chunk=None,
                 M_host=None, info=None, log=print):
        """All arrays are CuPy arrays on the current device (``M_host``: pinned host M when M is streamed)."""
        self.cp, self.spec, self.log = cp, spec, log
        self.rhs = rhs.reshape(-1)
        self.rhs_norm = float(cp.linalg.norm(self.rhs))
        self.boundary_trace, self.global_faces = boundary_trace, global_faces
        self.loc2glob, self.orient = loc2glob, orientations
        self.aff_mats, self.aff_vecs, self.aff_jacs = aff_mats, aff_vecs, aff_jacs
        self.bas, self.weights, self.ref = bas, weights, ref_points      # (P, Q), (Q,), (Q, 2)
        self.w, self.M, self.M_host = w, M, M_host
        self.resident = M_host is None
        self.ne, self.el_dof = int(w.shape[0]), int(w.shape[1])
        self.edg_dof = int(boundary_trace.shape[1])
        self.num_edg = int(boundary_trace.shape[0])
        self.chunk = chunk or max(1024, int(2**30 // (8*max(self.el_dof*3*self.edg_dof, self.el_dof*int(weights.size)))))
        self.own_matvec = matvec
        self.exact = exact
        self.exact_on_device = True
        try:
            probe = exact(cp.asarray([0.31, 0.47]), cp.asarray([0.29, -0.53]))
            if not isinstance(probe, cp.ndarray):
                raise TypeError('exact returned a host array')
        except TypeError:
            self.exact_on_device = False
        self.exact_host = None
        if not self.exact_on_device:
            # The case's exact solution rejects CuPy: tabulate it once per job (a one-way copy of
            # the quadrature points to the host). No campaign case needs this.
            log('[verify] WARNING exact solution evaluated on the host (tabulated once)')
            self.exact_host = np.empty((self.ne, int(weights.size)))
            for s in range(0, self.ne, self.chunk):
                e = min(s+self.chunk, self.ne)
                points = cp.asnumpy(self._points(s, e))
                self.exact_host[s:e] = exact(points[..., 0], points[..., 1])
        self.reference = reference
        self.info = dict(device_verification=True, verifier_M_resident=self.resident, verifier_M_bytes=int(M.nbytes),
                         verifier_element_chunk=self.chunk, exact_on_device=self.exact_on_device, **(info or {}))

    @classmethod
    def from_device(cls, cp, spec, assembly, space, exact, *, log=print):
        """From ``device_system.DeviceAssembly`` and the host DG space's reference quadrature tables."""
        quad = space.quad_data
        started = perf_counter()
        reference = Path(spec['cache'])/'reference_trace.npy'
        verifier = cls(cp, spec, rhs=assembly.system.rhs, boundary_trace=assembly.system.boundary_trace,
                       global_faces=assembly.system.global_faces, loc2glob=assembly.loc2glob,
                       orientations=assembly.orientations, aff_mats=assembly.aff_mats, aff_vecs=assembly.aff_vecs,
                       aff_jacs=assembly.aff_jacs, bas=cp.asarray(quad.bas_of_quads), weights=cp.asarray(quad.Krf_w),
                       ref_points=cp.asarray(quad.Krf_quads), w=assembly.w, M=assembly.M, exact=exact,
                       reference=cp.asarray(np.load(reference)) if reference.exists() else None,
                       matvec=FaceDenseMatvec(cp, assembly.system, blocks=assembly.system.blocks), log=log)
        cp.cuda.Device().synchronize()
        verifier.info['verifier_setup_seconds'] = perf_counter()-started
        log(f'[verify] device verifier from the device assembly in {verifier.info["verifier_setup_seconds"]:.3f} s: '
            f'{verifier.ne:,} elements, M {assembly.M.nbytes/2**30:.2f} GiB on device')
        return verifier

    @classmethod
    def from_cache(cls, cp, spec, system, get, exact, *, budget_bytes, log=print):
        """From a host-output cache: w and M are formed here from the stored local inverses (one upload)."""
        started = perf_counter()
        cache = Path(spec['cache'])
        geometry = np.load(cache/GEOMETRY)
        el_dof, edg_dof = int(geometry['el_dof']), int(geometry['edg_dof'])
        loc2glob = np.asarray(get('loc2glob'))
        ne = loc2glob.shape[0]
        local = get('local_inverse')
        width = local.shape[-1]
        chunk = max(1024, int(2**30 // (8*max(el_dof*width, el_dof*len(geometry['Krf_w'])))))
        faces3 = 3*edg_dof
        m_bytes = ne*el_dof*faces3*8
        resident = m_bytes <= budget_bytes
        w = cp.empty((ne, el_dof), dtype=cp.float64)
        M_host = None
        if resident:
            M = cp.empty((ne, el_dof, faces3), dtype=cp.float64)
        else:
            M = cp.cuda.alloc_pinned_memory(m_bytes)
            M_host = np.frombuffer(M, dtype=np.float64, count=ne*el_dof*faces3).reshape(ne, el_dof, faces3)
        boundary, source = get('boundary'), get('source_rhs')
        for s in range(0, ne, chunk):
            e = min(s+chunk, ne)
            lu = cp.asarray(np.ascontiguousarray(local[s:e, :el_dof, :]))
            m = cp.matmul(lu, cp.asarray(np.ascontiguousarray(boundary[s:e])))
            w[s:e] = cp.matmul(lu, cp.asarray(np.ascontiguousarray(source[s:e]))[..., None])[..., 0]
            if resident:
                M[s:e] = m
            else:
                m.get(out=M_host[s:e])
        reference = cache/'reference_trace.npy'
        verifier = cls(cp, spec, rhs=cp.asarray(np.asarray(system.rhs).reshape(-1)),
                       boundary_trace=cp.asarray(np.asarray(system.boundary_trace)),
                       global_faces=cp.asarray(np.asarray(system.global_faces)), loc2glob=cp.asarray(loc2glob),
                       orientations=cp.asarray(geometry['orientations']), aff_mats=cp.asarray(geometry['aff_mats']),
                       aff_vecs=cp.asarray(geometry['aff_vecs']), aff_jacs=cp.asarray(geometry['aff_jacs']),
                       bas=cp.asarray(geometry['bas_of_quads']), weights=cp.asarray(geometry['Krf_w']),
                       ref_points=cp.asarray(geometry['Krf_quads']), w=w, M=M, exact=exact,
                       reference=cp.asarray(np.load(reference)) if reference.exists() else None,
                       matvec=None, chunk=chunk, M_host=M_host, log=log)
        verifier.system = system
        cp.cuda.Device().synchronize()
        verifier.info['verifier_setup_seconds'] = perf_counter()-started
        log(f'[verify] device verifier ready in {verifier.info["verifier_setup_seconds"]:.2f} s: {ne:,} elements, '
            f'M {m_bytes/2**30:.2f} GiB {"on device" if resident else "streamed from pinned host"}, '
            f'exact solution on {"device" if verifier.exact_on_device else "host (tabulated once)"}')
        return verifier

    def _points(self, s, e):
        cp = self.cp
        return cp.einsum('krc,qc->kqr', self.aff_mats[s:e], self.ref) + self.aff_vecs[s:e, None, :]

    def _matvec(self, matvec):
        if matvec is not None:
            return matvec
        if self.own_matvec is None:
            self.own_matvec = FaceDenseMatvec(self.cp, self.system)
        return self.own_matvec

    def residual(self, x, matvec=None):
        cp = self.cp
        r = self._matvec(matvec)(x).reshape(-1)-self.rhs
        rn = float(cp.linalg.norm(r))
        return rn, rn/self.rhs_norm if self.rhs_norm else rn

    def relative_residual(self, x, b, matvec=None):
        cp = self.cp
        norm = float(cp.linalg.norm(b))
        r = float(cp.linalg.norm(self._matvec(matvec)(x).reshape(-1)-b))
        return r/norm if norm else r

    def l2_error(self, x):
        cp = self.cp
        b = self.edg_dof
        full = self.boundary_trace.copy()
        full[self.global_faces] = x.reshape(-1, b)
        full = full.reshape(self.num_edg, b)
        total = cp.zeros((), dtype=cp.float64)
        for s in range(0, self.ne, self.chunk):
            e = min(s+self.chunk, self.ne)
            t = full[self.loc2glob[s:e]]                                   # (m, 3, b)
            t = cp.where(self.orient[s:e, :, None], t, t[..., ::-1]).reshape(e-s, 3*b)
            m = self.M[s:e] if self.resident else cp.asarray(self.M_host[s:e])
            u = self.w[s:e] + cp.matmul(m, t[..., None])[..., 0]          # (m, el_dof)
            values = u @ self.bas                                           # (m, Q)
            if self.exact_on_device:
                points = self._points(s, e)
                exact = self.exact(points[..., 0], points[..., 1])
            else:
                exact = cp.asarray(self.exact_host[s:e])
            diff = values-exact
            total += cp.sum(self.aff_jacs[s:e, None]*diff*diff*self.weights[None, :])
        return float(cp.sqrt(total)), full.reshape(-1)

    def verify(self, x, *, solver_converged=True, matvec=None, case_l2_bound=True):
        """The vendored ``verify`` result dict, computed on the device.

        ``case_l2_bound=False`` drops the vendored 2e-8 L2 rule for quadratic/variable_velocity,
        which native_hp_worker never applied.
        """
        cp = self.cp
        cp.cuda.Device().synchronize()
        started = perf_counter()
        rn, rel = self.residual(x, matvec)
        l2, trace = self.l2_error(x)
        spec = self.spec
        passed = solver_converged and np.isfinite(rn) and rn <= spec['rtol']*self.rhs_norm and np.isfinite(l2)
        if case_l2_bound and spec['case'] in ('quadratic', 'variable_velocity') and spec['p'] >= 2:
            passed = passed and l2 < 2e-8
        trace_error = None
        if self.reference is not None:
            trace_error = float(cp.linalg.norm(trace-self.reference)/max(float(cp.linalg.norm(self.reference)), 1e-300))
            passed = passed and trace_error <= max(1e-8, 100*spec['rtol'])
        bound = spec.get('l2_bound')
        if bound is not None:
            passed = passed and l2 <= bound
        cp.cuda.Device().synchronize()
        return dict(passed=bool(passed), true_residual_norm=rn, true_relative_residual=rel, l2_error=l2,
                    trace_reference_error=trace_error, l2_bound=bound,
                    validation_reconstruction_ms=1000*(perf_counter()-started), validation_device=True)
