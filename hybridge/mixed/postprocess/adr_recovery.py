"""Explicit, reusable ADR recovery operators for bounded performance studies.

The coupled Neumann equations are identical to ``primal_raw_cuda``. Tensor
assembly contracts quadrature against cached reference products using GEMM;
it does not change precision or replace ADR recovery with diffusion recovery.
Only geometry/reference data are cached. Velocity, stabilization, diffusion,
flux and the primal mean are refreshed on every assembly.
"""
from __future__ import annotations

import numpy as np


def primal_rows(order):
    """Dimension of the p+1 mixed Neumann/trace/mean system."""
    d = (order + 2) * (order + 3) // 2
    return 3*d + 3*(order + 2) + 1


def recovery_working_bytes(order, elements):
    """Conservative peak estimate including references, copies and solve scratch.

    The campaign enforces a separate free-memory reserve and observes actual
    memory. This deliberately bounds a *chunk*, not an entire mesh.
    """
    d = (order + 2) * (order + 3) // 2
    m = 3*(order + 2) + order*(order + 1)
    return int(elements * 8 * (6*primal_rows(order)**2 + 8*m*d + 8*m*m + 16384))


class TensorPrimalAssembly:
    """Reference-product GEMM assembly; NumPy is an independent host backend."""

    def __init__(self, space, cache, *, xp=np):
        """Cache the geometry and reference products of one space; no PDE coefficient is kept."""
        self.xp, self.space, self.cache = xp, space, cache
        q = cache.post_space.quad_data
        self.d, self.e = cache.post_space.el_dof, q.edg_dof
        self.rows = 3*self.d + 3*self.e + 1
        convert = lambda a: xp.asarray(np.ascontiguousarray(a), dtype=xp.float64)
        self.jac = convert(space.mesh.aff_jacs)
        self.invmap = convert(space.mesh.inv_aff_mats_t)
        self.face_jac = convert(space.mesh.jacs_el_fc)
        self.normals = convert(space.mesh.normals)
        self.phi = convert(q.phi)
        self.mean = convert(q.Krf_w @ q.phi)
        self.mean_base = convert(cache.mean_base)
        self.mass_products = convert((q.Krf_w[:, None, None] * q.phi[:, :, None]
                                      * q.phi[:, None, :]).reshape(len(q.Krf_w), -1))
        self.grad_products = tuple(convert((q.Krf_w[:, None, None] * q.gphi[:, :, a, None]
                                           * q.phi[:, None, :]).reshape(len(q.Krf_w), -1))
                                   for a in range(2))
        self.grad_mass = tuple(convert(np.einsum('q,qi,qj->ij', q.Krf_w,
                                                 q.gphi[:, :, a], q.phi)) for a in range(2))
        self.grad_weighted = tuple(convert(q.Krf_w[:, None] * q.gphi[:, :, a]) for a in range(2))
        self.faces = []
        for f in range(3):
            phi, psi, w = q.bas_of_bd_quads[f], q.bas1d_of_ref_edg_qds, q.weights_JGL
            self.faces.append(tuple(convert(a) for a in (
                phi, psi, phi*w, psi*w,
                np.einsum('q,iq,jq->ij', w, phi, phi),
                np.einsum('q,iq,aq->ia', w, phi, psi),
                np.einsum('q,iq,jq->qij', w, phi, phi).reshape(len(w), -1),
                np.einsum('q,iq,aq->qia', w, phi, psi).reshape(len(w), -1),
                np.einsum('q,aq,bq->qab', w, psi, psi).reshape(len(w), -1))))

    def assemble(self, local_unknowns, flux, samples, inverse):
        """Return row-major matrices/RHS, retaining all input arrays unchanged.

        ``flux`` contains p+1 vector coefficients, shape (elements, 2, d).
        This low-level assembly accepts the existing prepared sample contract.
        """
        xp, d, e = self.xp, self.d, self.e
        n = self.space.mesh.num_tri
        beta, beta_face, tau = samples
        matrix = xp.zeros((n, self.rows, self.rows), dtype=xp.float64)
        rhs = xp.zeros((n, self.rows), dtype=xp.float64)
        jac, im = self.jac, self.invmap
        for a in range(2):
            gradient = jac[:, None, None] * (
                im[:, a, 0, None, None]*self.grad_mass[0] + im[:, a, 1, None, None]*self.grad_mass[1])
            matrix[:, a*d:(a+1)*d, :d] = -gradient
            matrix[:, 2*d:3*d, (a+1)*d:(a+2)*d] = -gradient
            for b in range(2):
                matrix[:, a*d:(a+1)*d, (b+1)*d:(b+2)*d] = jac[:, None, None] * (
                    inverse[..., 2*a+b] @ self.mass_products).reshape(n, d, d)
        beta_r = beta[..., 0]*im[:, 0, 0, None] + beta[..., 1]*im[:, 1, 0, None]
        beta_s = beta[..., 0]*im[:, 0, 1, None] + beta[..., 1]*im[:, 1, 1, None]
        matrix[:, 2*d:3*d, :d] = -jac[:, None, None] * (
            beta_r @ self.grad_products[0] + beta_s @ self.grad_products[1]).reshape(n, d, d)
        values = flux @ self.phi.T
        vr = values[:, 0]*im[:, 0, 0, None] + values[:, 1]*im[:, 1, 0, None]
        vs = values[:, 0]*im[:, 0, 1, None] + values[:, 1]*im[:, 1, 1, None]
        rhs[:, 2*d:3*d] = -jac[:, None] * (vr @ self.grad_weighted[0] + vs @ self.grad_weighted[1])
        matrix[:, 2*d:3*d, -1] = jac[:, None]*self.mean
        matrix[:, -1, :d] = jac[:, None]*self.mean
        rhs[:, -1] = jac*(local_unknowns[:, :self.space.el_dof] @ self.mean_base)
        for f, (phi, psi, wphi, wpsi, mm, cross, mprod, cprod, tprod) in enumerate(self.faces):
            js = self.face_jac[:, f, None, None]
            normal = self.normals[:, f]
            bn = xp.sum(beta_face[:, f]*normal[:, None, :], axis=2)
            bnt = bn-tau[:, f]
            tr = slice(3*d+f*e, 3*d+(f+1)*e)
            matrix[:, 2*d:3*d, :d] += js*(tau[:, f] @ mprod).reshape(n, d, d)
            matrix[:, 2*d:3*d, tr] = js*(bnt @ cprod).reshape(n, d, e)
            matrix[:, tr, :d] = js*(tau[:, f] @ cprod).reshape(n, d, e).transpose(0, 2, 1)
            matrix[:, tr, tr] = js*(bnt @ tprod).reshape(n, e, e)
            for a in range(2):
                scale = js*normal[:, a, None, None]
                matrix[:, a*d:(a+1)*d, tr] = scale*cross
                matrix[:, 2*d:3*d, (a+1)*d:(a+2)*d] += scale*mm
                matrix[:, tr, (a+1)*d:(a+2)*d] = scale*cross.T
            fn = (flux[:, 0]*normal[:, 0, None] + flux[:, 1]*normal[:, 1, None]) @ phi
            weighted = self.face_jac[:, f, None]*fn
            rhs[:, 2*d:3*d] += weighted @ wphi.T
            rhs[:, tr] = weighted @ wpsi.T
        return matrix, rhs


class CachedTotalFlux:
    """L2-closest correction map for fixed affine geometry on one CUDA device.

    Row equilibration reduces mesh-size scaling of the moment constraints.
    Setup solves the same constraint Schur system for all correction columns;
    each new field needs only matrix-vector products. No PDE coefficient is
    cached, and setup is always reported separately by the benchmark.
    """

    def __init__(self, space, cache):
        """Factor the fixed-geometry correction map once on the current device, and check it."""
        from hybridge.runtime.optional import require_cupy_device
        from hybridge.linalg.gpu.cublas_batched import lu_solve_batched_cublas
        cp = require_cupy_device()
        self.cp, self.space, self.cache = cp, space, cache
        self.device = cp.cuda.runtime.getDevice()
        q, n, d = cache.post_space.quad_data, space.mesh.num_tri, cache.post_space.el_dof
        self.jac = cp.asarray(space.mesh.aff_jacs)
        self.face_jac = cp.asarray(space.mesh.jacs_el_fc)
        normals = cp.asarray(space.mesh.normals)
        self.base_project = cp.asarray((q.Krf_w[:, None]*space.basis_at(q.Krf_quads)) @ space.quad_data.MKrf_inv)
        self.embed = cp.asarray(cache.base_to_post_mass.T @ q.MKrf_inv)
        self.low_base = cp.asarray(cache.interior_low_to_base)
        self.face_test = cp.asarray(q.bas1d_of_ref_edg_qds*q.weights_JGL)
        faces = cp.einsum('Kf,Kfc,fia->Kfaci', self.face_jac, normals,
                          cp.asarray(q.face_element_test_trace_trial)).reshape(n, -1, 2*d)
        low = cp.asarray(cache.interior_low_to_post)
        interior = cp.zeros((n, 2*len(low), 2*d))
        interior[:, :len(low), :d] = self.jac[:, None, None]*low
        interior[:, len(low):, d:] = self.jac[:, None, None]*low
        constraints = cp.concatenate((faces, interior), axis=1)
        self.scale = cp.sqrt(cp.sum(constraints*constraints, axis=2))
        self.constraints = constraints/self.scale[..., None]
        mass = cp.asarray(q.MKrf_inv)
        lift = cp.concatenate((mass @ self.constraints[:, :, :d].transpose(0, 2, 1),
                               mass @ self.constraints[:, :, d:].transpose(0, 2, 1)), axis=1)
        schur = self.constraints @ lift
        m = schur.shape[1]
        inverse = cp.broadcast_to(cp.eye(m), schur.shape).copy()
        factors = cp.ascontiguousarray(schur.transpose(0, 2, 1))
        lu_solve_batched_cublas(factors, inverse, check_info=True)
        inverse = inverse.transpose(0, 2, 1)
        self.correction = lift @ inverse
        error = cp.max(cp.abs(self.constraints @ self.correction-cp.eye(m))).item()
        if not np.isfinite(error) or error > 2e-8:
            raise ArithmeticError(f'Cached flux constraint-map defect {error:g}')
        self.constraint_map_defect = float(error)
        self.bytes = sum(v.nbytes for v in vars(self).values() if isinstance(v, cp.ndarray))

    def __call__(self, total_values, numerical, space, cache, *, phase=None):
        """Recover one total flux by matrix-vector products; the spaces and device must be the cached ones."""
        from hybridge.core.device import field_from_cupy_coefficients
        from hybridge.core.space import VectorDGField
        cp = self.cp
        if space is not self.space or cache is not self.cache or cp.cuda.runtime.getDevice() != self.device:
            raise ValueError('Cached flux recovery belongs to a different space/cache/device')
        n, d = space.mesh.num_tri, cache.post_space.el_dof
        step = phase or (lambda name, fn: fn())
        base = step('flux.base_projection', lambda: total_values @ self.base_project)
        initial = (base @ self.embed).transpose(1, 0, 2).reshape(n, 2*d)
        target_face = (numerical @ self.face_test.T * self.face_jac[..., None]).reshape(n, -1)
        target_interior = (base @ self.low_base.T).transpose(1, 0, 2).reshape(n, -1)*self.jac[:, None]
        target = cp.concatenate((target_face, target_interior), axis=1)/self.scale
        gap = target-(self.constraints @ initial[..., None])[..., 0]
        result = step('flux.cached_map_apply', lambda: (initial+(self.correction @ gap[..., None])[..., 0]).reshape(n, 2, d))
        return VectorDGField(tuple(field_from_cupy_coefficients(cache.post_space, result[:, a],
                               name='total_flux_h_star') for a in range(2)))


class ADRRecoveryWorkspace:
    """Opt-in recovery of an existing ADR mixed solution with phase callbacks.

    Reuse this object for successive fields on the same immutable affine mesh
    and spaces. To benchmark rebuilding local data, construct a new object on
    each invocation. Upstream ADR assembly/reconstruction is not repeated here:
    its already reconstructed local unknowns and trace are the inputs.
    """

    def __init__(self, space, trace_space, *, assembly='tensor', solver='cublas',
                 threads=128, cached_flux=True, reuse_samples=True, reference_cache=None, phase=None):
        """Select the assembly and solver and build their geometry tables; ``phase`` times each stage."""
        from hybridge.runtime.optional import require_cupy_device
        from hybridge.mixed.postprocess.flux import _new_hdg_postprocess_cache
        if assembly not in ('raw', 'compact', 'tensor'):
            raise ValueError(f'Unknown ADR recovery assembly {assembly}')
        self.cp = require_cupy_device()
        self.space, self.trace_space = space, trace_space
        self.assembly, self.solver, self.threads = assembly, solver, threads
        self.reuse_samples = reuse_samples
        self.phase = phase or (lambda name, function: function())
        if reference_cache is None:
            self.cache = self.phase('setup.reference', lambda: _new_hdg_postprocess_cache(space, trace_space))
        else:
            from dataclasses import replace
            if reference_cache.base_space is not space or reference_cache.trace_space is not trace_space:
                raise ValueError('Reference cache must use the workspace spaces')
            # Clone only reference tables; never inherit geometry factors or
            # dynamically attached device geometry/assembly caches.
            self.cache = self.phase('setup.reference_reuse', lambda: replace(reference_cache,
                flux_ainv_constraint_t=None, flux_schur_lu=None, flux_schur_pivots=None,
                primal_lu=None, primal_pivots=None, raw_flux_cache=None))
        self.tensor = self.phase('setup.tensor_tables', lambda: TensorPrimalAssembly(space, self.cache, xp=self.cp)) \
            if assembly == 'tensor' else None
        self.flux = self.phase('setup.flux_map', lambda: CachedTotalFlux(space, self.cache)) if cached_flux else None

    def recover(self, local_unknowns, trace, beta, prepared, stabilization, diffusion):
        """Recover the p+1 primal field from one reconstructed solve, refreshing all PDE coefficients."""
        from hybridge.mixed.postprocess.total_flux import _adr_postprocess_samples, _postprocess_total_flux
        from hybridge.mixed.coefficients import sample_diffusion_tensor, inverse_diffusion_values
        from hybridge.mixed.postprocess.primal_raw_cuda import primal_system_raw_cuda
        from hybridge.core.device import as_cupy_coefficients, as_cupy_space, field_from_cupy_coefficients
        from hybridge.linalg.gpu.cublas_batched import BatchedLUWorkspace
        from hybridge.linalg.gpu.tuned_batched_solve import solve_selected
        cp, phase, space = self.cp, self.phase, self.space
        samples = phase('sampling.advection_stabilization', lambda: _adr_postprocess_samples(
            beta, prepared, space, self.cache.post_space, stabilization, xp=cp))
        # The legacy function uses this cache; restore its prior owner afterward.
        previous = getattr(space, '_adr_recovery_cache', None)
        space._adr_recovery_cache = self.cache
        from hybridge.mixed.postprocess.flux_cupy import postprocess_total_flux_l2_cupy
        recover_flux = self.flux or postprocess_total_flux_l2_cupy
        try:
            flux = phase('flux.inputs_and_recovery', lambda: _postprocess_total_flux(
                local_unknowns, trace, beta, prepared, space, self.trace_space, stabilization,
                'l2_closest', 'cupy', samples=samples,
                recovery=lambda *args: recover_flux(*args, phase=phase)))
        finally:
            space._adr_recovery_cache = previous
        if not self.reuse_samples:
            samples = phase('sampling.advection_stabilization_repeat', lambda: _adr_postprocess_samples(
                beta, prepared, space, self.cache.post_space, stabilization, xp=cp))
        inverse = phase('sampling.inverse_diffusion', lambda: inverse_diffusion_values(
            sample_diffusion_tensor(diffusion, self.cache.post_space, device=True)))
        if self.tensor is None:
            matrix, rhs = phase('primal.assembly', lambda: primal_system_raw_cuda(
                local_unknowns, flux, space, self.cache, samples, inverse,
                threads=self.threads, compact=self.assembly == 'compact'))
        else:
            values = phase('primal.flux_layout', lambda: cp.stack([
                as_cupy_coefficients(f, as_cupy_space(f.space)) for f in flux.components], axis=1))
            matrix, rhs = phase('primal.assembly', lambda: self.tensor.assemble(local_unknowns, values, samples, inverse))
        a, b = phase('primal.solver_layout', lambda: (
            cp.ascontiguousarray(matrix.transpose(0, 2, 1)), rhs[:, None, :].copy()))
        info = cp.empty(space.mesh.num_tri, dtype=cp.int32)
        phase('primal.factor_and_solve', lambda: solve_selected(
            self.solver, a, b, info=info, workspace=BatchedLUWorkspace()))
        def accept():
            """Reject a failed or nonfinite dense solve, then extract the primal block."""
            if bool(cp.any(info).item()) or not bool(cp.all(cp.isfinite(b)).item()):
                raise ArithmeticError('ADR recovery dense solve failed')
            return field_from_cupy_coefficients(self.cache.post_space,
                cp.ascontiguousarray(b[:, 0, :self.cache.post_space.el_dof]), name='u_h_star')
        result = phase('primal.check_and_extract', accept)
        return result
