r"""Conservative scalar HDG: div(q + beta*u) + reaction*u = source

q = -diffusion grad(u), prescribed Dirichlet trace on the whole boundary

Numerical normal flux: q.n + beta.n*uhat + tau*(u-uhat),
    tau = stabilization + max(beta.n, 0) at face quadrature points

CPU quadrature supports constant or callable velocity components. CUDA mode
accelerates local inversion, condensation and global block assembly; coefficient
integration, boundary elimination and reconstruction remain on the CPU.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, gmres, LinearOperator
from ..assembly import hdg
from ..assembly.face_dense import (build_face_topology, assemble_global_face_blocks,
    eliminate_dirichlet_faces, expand_eliminated_solution, face_dense_matvec)
from .diff_rea import (_normalize_tau, _local_solver_pre_mats,
    diffusion_inverse_mass_blocks, _diffusion_components, diffusion_element_boundary_mats,
    diffusion_trace_lift, split_diffusion_unknowns)
from .adv_rea import _callable_advection_mats, _callable_beta_normal_flux


@dataclass
class ADRAssembly:
    system: object
    element_blocks: np.ndarray
    local_solver: np.ndarray
    boundary: np.ndarray
    source_rhs: np.ndarray
    local_matrix: np.ndarray
    timings: dict


@dataclass
class ADRResult:
    field: object
    flux: object
    trace: np.ndarray
    assembly: ADRAssembly
    relative_residual: float
    residual_norm: float
    iterations: int
    timings: dict


def _velocity(beta):
    if len(beta) != 2:
        raise ValueError('beta must have two scalar or callable components')
    return tuple(c if callable(c) else (lambda x, y, c=float(c): c) for c in beta)


def prepare_adr_local_matrices(source, beta, reaction, space, *, diffusion=1., stabilization=1.):
    """Return A, B, C, D, f with A*x = f+B*uhat and flux = C*x-D*uhat.

    A:(NE,3P,3P), B:(NE,3P,3F), C:(NE,3,F,3P), D:(NE,3,F,F).
    B uses local trace orientation; C and D use global trace orientation.
    """
    mesh, q = space.mesh, space.quad_data
    tau = _normalize_tau(stabilization, space)
    if not np.all(np.isfinite(tau)) or np.any(tau <= 0):
        raise ValueError('diffusive stabilization must be finite and strictly positive')
    k00,k01,k10,k11 = _diffusion_components(diffusion, space)
    if (not all(np.all(np.isfinite(k)) for k in (k00,k01,k10,k11))
            or not np.allclose(k01,k10,rtol=1e-12,atol=1e-14)
            or np.any(k00 <= 0) or np.any(k00*k11-k01*k10 <= 0)):
        raise ValueError("diffusion must be symmetric positive definite at quadrature points")
    beta = _velocity(beta)
    bn = _callable_beta_normal_flux(beta, space)
    if not np.all(np.isfinite(bn)):
        raise ValueError('nonfinite velocity on faces')
    up = np.maximum(bn, 0.)
    d0, d1, mt, n0, n1, _ = _local_solver_pre_mats(reaction, tau, space)
    g00, g01, g10, g11 = diffusion_inverse_mass_blocks(diffusion, space)
    ne, p, f = mesh.num_tri, space.el_dof, q.edg_dof
    a = np.zeros((ne, 3*p, 3*p))
    v = a.reshape(ne, 3, p, 3, p)
    weights = mesh.jacs_el_fc[..., None] * q.weights_JGL
    phi, psi = q.bas_of_bd_quads, q.bas1d_of_ref_edg_qds
    v[:, 0, :, 0, :] = mt - _callable_advection_mats(space, beta) + np.einsum(
        'efq,fiq,fjq->eij', weights*up, phi, phi, optimize=True)
    v[:, 0, :, 1, :], v[:, 0, :, 2, :] = n0-d0, n1-d1
    v[:, 1, :, 0, :], v[:, 2, :, 0, :] = d0, d1
    v[:, 1, :, 1, :], v[:, 1, :, 2, :] = -g00, -g01
    v[:, 2, :, 1, :], v[:, 2, :, 2, :] = -g10, -g11
    b = diffusion_element_boundary_mats(tau, space)
    b[:, :p, :] += np.einsum('efq,fiq,jq->eifj', weights*(up-bn), phi, psi,
                             optimize=True).reshape(ne, p, 3*f)
    c = diffusion_trace_lift(tau, space)
    extra = np.einsum('efq,iq,fjq->efij', weights*up, psi, phi, optimize=True)
    neg = ~mesh.orientations
    extra[neg] = extra[neg][:, ::-1, :]
    c[..., :p] += extra
    d = np.einsum('efq,iq,jq->efij', weights*(tau[..., None]+up-bn), psi, psi,
                  optimize=True)
    d[neg] = d[neg][:, ::-1, ::-1]
    rhs = hdg.block_source_moments(source, space, num_blocks=3, source_block=0)
    if not all(np.all(np.isfinite(x)) for x in (a,b,c,d,rhs)):
        raise ValueError('nonfinite local coefficient matrices')
    return a, b, c, d, rhs


def assemble_advection_diffusion_reaction(source, beta, reaction, boundary_condition, space,
                                         *, diffusion=1., stabilization=1., backend='numpy'):
    if backend not in ('numpy', 'cupy'):
        raise ValueError("backend must be 'numpy' or 'cupy'")
    start = perf_counter()
    a,b,c,d,rhs = prepare_adr_local_matrices(source,beta,reaction,space,
                                          diffusion=diffusion,stabilization=stabilization)
    topology = build_face_topology(space.mesh.loc2glob_edge)
    timings = {'host_preparation': perf_counter()-start}
    xp = np
    if backend == 'cupy':
        from ..backends.cupy import require_cupy_device
        xp = require_cupy_device()
    def sync():
        if backend == 'cupy':
            xp.cuda.get_current_stream().synchronize()
    sync(); t = perf_counter()
    aa,bb,cc,dd,ff = [xp.asarray(x) for x in (a,b,c,d,rhs)]
    sync(); timings['host_to_device'] = perf_counter()-t if backend == 'cupy' else 0.
    t = perf_counter()
    inv = xp.linalg.inv(aa)
    sync(); timings['local_inverse'] = perf_counter()-t
    t = perf_counter()
    ne, f = space.mesh.num_tri, space.quad_data.edg_dof
    schur = (cc @ (inv @ bb)[:,None]).reshape(ne,3,f,3,f)
    ei,fi = np.nonzero(~space.mesh.orientations)
    if backend == 'cupy':
        ei,fi = xp.asarray(ei),xp.asarray(fi)
    schur[ei,:,:,fi,:] = schur[ei,:,:,fi,::-1]
    blocks = -schur.swapaxes(2,3).copy()
    diag = xp.arange(3)
    blocks[:,diag,diag] += dd
    blocks = xp.ascontiguousarray(blocks)
    local_rhs = (cc @ (inv @ ff[...,None])[:,None]).squeeze(-1)
    sync(); timings['condensation'] = perf_counter()-t
    t = perf_counter()
    if backend == 'cupy':
        from ..backends.cupy_assembly import CuPyGlobalFaceAssembler
        assembler = CuPyGlobalFaceAssembler.from_topology(space.mesh.loc2glob_edge,topology,
            block_size=f, active_row_faces=space.mesh.interior_face_mask)
        global_blocks = assembler.assemble(blocks)
    else:
        global_blocks = assemble_global_face_blocks(blocks,space.mesh.loc2glob_edge,topology,
            active_row_faces=space.mesh.interior_face_mask)
    sync(); timings['global_blocks'] = perf_counter()-t
    t = perf_counter()
    if backend == 'cupy':
        inv,blocks,global_blocks,local_rhs = map(xp.asnumpy,(inv,blocks,global_blocks,local_rhs))
    timings['device_to_host'] = perf_counter()-t if backend == 'cupy' else 0.
    t = perf_counter()
    interior_rhs = np.zeros((space.mesh.num_edg,f))
    ie, jf = space.mesh.interior_elements, space.mesh.interior_faces
    np.add.at(interior_rhs,space.mesh.loc2glob_edge[ie,jf],local_rhs[ie,jf])
    bc = hdg.boundary_trace_coefficients(boundary_condition,space)
    system = eliminate_dirichlet_faces(global_blocks,topology,interior_rhs,bc,
                                        space.mesh.int_edges_inds)
    timings['rhs_and_boundary'] = perf_counter()-t
    timings['assembly_total'] = perf_counter()-start
    return ADRAssembly(system,blocks,inv,b,rhs,a,timings)


def _csr(system):
    """Sparse conversion with O(NF*slots*F²) storage, never a dense global matrix."""
    nf, slots, f, _ = system.blocks.shape
    rows = np.broadcast_to(np.arange(nf)[:,None,None,None]*f + np.arange(f)[None,None,:,None],
                           (nf,slots,f,f))
    cols = np.broadcast_to(system.neighbors[:,:,None,None]*f + np.arange(f)[None,None,None,:],
                           (nf,slots,f,f))
    mask = np.broadcast_to(system.neighbors[:,:,None,None]>=0,(nf,slots,f,f))
    return coo_matrix((system.blocks[mask],(rows[mask],cols[mask])),shape=(nf*f,nf*f)).tocsr()


def solve_advection_diffusion_reaction(source,beta,reaction,boundary_condition,space,*,
        diffusion=1.,stabilization=1.,assembly_backend='numpy',solver='direct',
        rtol=1e-11,atol=0.,restart=75,maxiter=3000,gpu_options=None):
    if solver not in ('direct','gmres','gpu'):
        raise ValueError("solver must be direct, gmres or gpu")
    if rtol <= 0 or atol < 0 or restart < 1 or maxiter < 1:
        raise ValueError('invalid solver tolerance or iteration limit')
    start = perf_counter()
    assembly = assemble_advection_diffusion_reaction(source,beta,reaction,boundary_condition,space,
        diffusion=diffusion,stabilization=stabilization,backend=assembly_backend)
    system = assembly.system
    rhs = system.rhs.reshape(-1)
    timings = dict(assembly.timings)
    iterations = 0
    t = perf_counter()
    if rhs.size == 0:
        x = rhs.copy()
        timings['solver_setup'] = 0.
        timings['solve'] = 0.
    elif solver == 'gpu':
        from .diff_rea_gpu import build_diffusion_reaction_gpu_solver, DiffusionReactionGPUOptions
        from ..backends.cupy import require_cupy_device
        cp = require_cupy_device()
        opts = DiffusionReactionGPUOptions.normalize(gpu_options)
        if opts.dtype != 'float64':
            raise ValueError('ADR validation currently requires float64')
        with cp.cuda.Device(opts.device_id):
            gs,_,_,_,_ = build_diffusion_reaction_gpu_solver(system,assembly.element_blocks,
                space.mesh.loc2glob_edge,polynomial_order=int(space.order),options=opts,
                rtol=rtol,atol=atol,max_iterations=maxiter)
            bdev = cp.asarray(rhs)
            cp.cuda.get_current_stream().synchronize()
            timings['solver_setup'] = perf_counter()-t
            t = perf_counter()
            result = gs.solve(bdev)
            cp.cuda.get_current_stream().synchronize()
            timings['solve'] = perf_counter()-t
            t = perf_counter()
            x = cp.asnumpy(result.solution)
            timings['solution_to_host'] = perf_counter()-t
            iterations = int(result.iterations)
    else:
        if solver == 'direct':
            operator = _csr(system)
        else:
            operator = LinearOperator((rhs.size,rhs.size),dtype=np.float64,
                matvec=lambda x: face_dense_matvec(system.blocks,system.neighbors,x).reshape(-1))
        timings['solver_setup'] = perf_counter()-t
        t = perf_counter()
        if solver == 'direct':
            x = spsolve(operator,rhs)
        else:
            def callback(_):
                nonlocal iterations
                iterations += 1
            x,info = gmres(operator,rhs,rtol=rtol,atol=atol,restart=restart,maxiter=maxiter,
                           callback=callback,callback_type='legacy')
            if info != 0:
                raise RuntimeError(f'CPU GMRES failed: info={info}')
        timings['solve'] = perf_counter()-t
    t = perf_counter()
    residual = face_dense_matvec(system.blocks,system.neighbors,x).reshape(-1)-rhs
    rn, bn = float(np.linalg.norm(residual)),float(np.linalg.norm(rhs))
    if not np.isfinite(rn) or rn > max(atol,rtol*bn):
        raise RuntimeError(f'True residual {rn:.3e} exceeds requested tolerance {max(atol,rtol*bn):.3e}')
    trace = expand_eliminated_solution(x,system)
    unknowns = hdg.reconstruct_local_unknowns(trace,assembly.source_rhs,assembly.local_solver,
                                             assembly.boundary,space)
    field,flux = split_diffusion_unknowns(unknowns,space)
    timings['residual_and_reconstruction'] = perf_counter()-t
    timings['total'] = perf_counter()-start
    return ADRResult(field,flux,trace,assembly,rn/bn if bn else rn,rn,iterations,timings)
