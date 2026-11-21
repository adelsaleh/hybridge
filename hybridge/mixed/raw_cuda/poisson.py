"""Device-adaptive pure Poisson assembly, RHS refresh, and reconstruction.

Only fixed reference tensors and requested factors/responses persist. Each
element block owns one scalar factor plus a bounded batch of local columns.
The global sparse format changes only direct emission, not the local algebra.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
from typing import Any

import numpy as np

from hybridge.hdg.cuda.pattern import build_reduced_csr_pattern_raw
from hybridge.runtime.cuda_tuning import (
    KernelConfiguration, device_signature, event_seconds, select_configuration,
)
from hybridge.hdg.trace_maps import _edge_to_solve_edge, _interior_side_index, _side_flux_offsets
from hybridge.runtime.optional import require_cupy
from hybridge.runtime.precision import REAL_DTYPE, REAL_ITEMSIZE, real_raw_kernel
from .poisson_source import SOURCE


_KERNELS = {}
SOURCE_VERSION = hashlib.sha256(SOURCE.encode()).hexdigest()


@dataclass
class PoissonContext:
    cspace: Any
    trace_ref: Any
    reference: dict
    tau: float
    policy: str
    factors: Any
    pivots: Any
    response: Any
    source_solution: Any
    block_size: Any = 'auto'
    local_factor_key: Any = None
    reports: dict = field(default_factory=dict)
    boundary_is_zero: bool = False
    # Launch selections made in this process, and the status buffer every launch rewrites.
    resolved: dict = field(default_factory=dict)
    failures: Any = None

    @property
    def local_factor_bytes(self):
        """Count retained per-element factor and response arrays exactly."""
        if self.policy == 'none':
            return 0
        arrays = (self.factors, self.pivots)
        if _keeps_responses(self.policy):
            arrays += (self.response, self.source_solution)
        return sum(a.nbytes for a in arrays)


def _keeps_responses(policy):
    """Reconstruction applies the trace responses assembly solved for, without a local solve."""
    return policy in ('schur-cholesky', 'schur-lu+responses')


def _keeps_lu(policy):
    return policy in ('schur-lu', 'schur-lu+responses')


def validate_block_size(requested, order):
    """Preserve automatic selection and validate deterministic overrides."""
    if requested is None or isinstance(requested, str) and requested.lower() == 'auto':
        return 'auto'
    if isinstance(requested, bool) or not isinstance(requested, (str, int, np.integer)):
        raise ValueError('Poisson raw_block_size must be auto or 1,32,64,128,256')
    value = int(requested)
    if value not in (1, 32, 64, 128, 256):
        raise ValueError('Poisson raw_block_size must be auto or 1,32,64,128,256')
    if value == 1 and order > 6:
        raise ValueError('p=7..9 Poisson requires a cooperative block size')
    return value


def shared_bytes(nel, ntr, config, *, factor=True):
    """Size the selected factor/coupling/column layout and status storage."""
    return ((int(factor)+2*config.retained_coupling)*nel*nel
            +(3*nel+3*ntr)*config.columns)*REAL_ITEMSIZE + (nel+1)*4


def _warp_solve(context, phase):
    """Phases that apply stored LU factors solve on warps and read the factors in place."""
    return phase > 0 and _keeps_lu(context.policy) and not (phase == 2 and _keeps_responses(context.policy))


def _workspace_bytes(context, phase, config):
    """Omit shared factor storage for response-based reconstruction and in-place LU factors."""
    nel = int(context.cspace.el_dof)
    warp = _warp_solve(context, phase)
    return shared_bytes(nel, int(context.cspace.edg_dof), config,
                        factor=not warp and not (phase == 2 and _keeps_responses(context.policy))) + (
        nel*REAL_ITEMSIZE + nel*4 if warp else 0)  # Pivot reciprocals and the folded row permutation.


def _failure_buffer(context, ne):
    """One per-element status buffer per context; every launch writes all of it."""
    cp = require_cupy()
    if context.failures is None or int(context.failures.size) != ne:
        context.failures = cp.empty(ne, dtype=cp.int32)
    return context.failures


def _check_failures(context, phase, ne, failures):
    """Read the status back only when the launch factorized: stored factors cannot fail."""
    if not ne or (phase > 0 and context.policy != 'none'):
        return
    cp = require_cupy()
    if bool(cp.any(failures).item()):
        element = int(cp.flatnonzero(failures)[0].item())
        raise np.linalg.LinAlgError(f'Poisson local factorization failed in element {element}, '
                                    f'status={int(failures[element].item())}')


def _reference(cspace, trace_ref, d0, d1, face_mass):
    """Cache reference-only Schur contractions and oriented trace lifts."""
    cp = require_cupy()
    cache = cspace._poisson_reference_cache
    if cache is None:
        cache = {}
        object.__setattr__(cspace, '_poisson_reference_cache', cache)
    key = trace_ref.kind
    if key not in cache:
        d = cp.stack((d0, d1))
        md = cp.ascontiguousarray(cspace.quad_data.MKrf_inv[None] @ d)
        lifts = cp.ascontiguousarray(trace_ref.face_trace_test_element_trial_oriented)
        cache[key] = dict(
            dd=cp.ascontiguousarray(cp.einsum('aik,bkj->abij', d, md)).reshape(-1),
            fd=cp.ascontiguousarray(cp.einsum('fik,akj->faij', face_mass, md)).reshape(-1),
            d=cp.ascontiguousarray(d), md=md, fm=cp.ascontiguousarray(face_mass), lifts=lifts,
            mlifts=cp.ascontiguousarray(cp.einsum('ij,ftj->fti', cspace.quad_data.MKrf_inv, lifts)),
            edge_mass=cp.ascontiguousarray(trace_ref.M_rf_fc),
        )
    return cache[key]


def _context(cspace, trace_ref, d0, d1, face_mass, tau, policy, block_size, local_factor_key):
    """Allocate only the factor/response arrays requested by the cache policy."""
    cp = require_cupy()
    if policy not in ('none', 'schur-lu', 'schur-lu+responses', 'schur-cholesky'):
        raise ValueError('unsupported Poisson local-factor policy')
    if policy == 'schur-cholesky' and tau <= 0:
        raise ValueError('Schur-Cholesky caching requires strictly positive stabilization')
    ne, n, nt = int(cspace.mesh.num_tri), int(cspace.el_dof), int(cspace.edg_dof)
    return PoissonContext(cspace, trace_ref, _reference(cspace, trace_ref, d0, d1, face_mass),
        float(tau), policy,
        cp.empty((ne,n,n) if policy != 'none' else 1, dtype=REAL_DTYPE),
        cp.empty((ne,n) if policy != 'none' else 1, dtype=cp.int32),
        cp.empty((ne,n,3*nt) if _keeps_responses(policy) else 1, dtype=REAL_DTYPE),
        cp.empty((ne,n) if _keeps_responses(policy) else 1, dtype=REAL_DTYPE),
        block_size, local_factor_key)


def _validate_context(context, cspace, trace_ref, tau, local_factor_key):
    """Reject factor reuse across different operator owners or parameters."""
    if context.cspace is not cspace or context.trace_ref.kind != trace_ref.kind or context.tau != float(tau):
        raise ValueError('Poisson cached local factors belong to a different space, trace basis, or stabilization')
    if local_factor_key is not None and context.local_factor_key != local_factor_key:
        raise ValueError('Poisson cached local factor key does not match the current operator')


def _compile(context, phase, fmt, config):
    """Compile a precision-specialized phase and opt into its shared budget."""
    cp = require_cupy()
    n, nt = int(context.cspace.el_dof), int(context.cspace.edg_dof)
    key = (cp.cuda.runtime.getDevice(), n, nt, phase, fmt, context.policy,
           config.columns, config.retained_coupling, np.dtype(REAL_DTYPE).str)
    if key not in _KERNELS:
        warp = _warp_solve(context, phase)
        constants = dict(NEL=n, NTR=nt, PHASE=phase, FORMAT={'coo':0,'csr':1,'bsr':2}[fmt],
                         FACTOR_STORAGE=0 if warp or (phase == 2 and _keeps_responses(context.policy)) else n*n,
                         WARP_SOLVE=int(warp),
                         BATCH=config.columns, RETAIN=int(config.retained_coupling),
                         CHOLESKY=int(context.policy=='schur-cholesky'),
                         RESPONSE=int(_keeps_responses(context.policy)),
                         USE_FACTORS=int(phase>0 and context.policy!='none'),
                         WRITE_FACTORS=int(phase==0 and context.policy!='none'),
                         SYMMETRY_RTOL=max(5.e-11, 32*np.finfo(REAL_DTYPE).eps))
        source = ''.join(f'#define {name} {value}\n' for name,value in constants.items())+SOURCE
        kernel = real_raw_kernel(source, 'poisson_local', options=('--std=c++11',))
        kernel.compile()
        required = _workspace_bytes(context, phase, config)
        if required > 48*1024:
            kernel.max_dynamic_shared_size_bytes = required
        _KERNELS[key] = kernel
    return _KERNELS[key]


FP64_RICH_PRIORS = 'fp64-rich-1'


def _fp64_rich_priors_apply(phase, device, explicit):
    """Return whether automatic assembly tuning ranks the FP64-rich priors first."""
    ratio = device.get('fp32_fp64_ratio')
    return phase == 0 and explicit is None and ratio is not None and ratio <= 4


def _fp64_rich_assembly_priors(n, width):
    """Assembly candidates for GPUs with full-rate FP64 (FP32:FP64 <= 4).

    On an H200 NVL (100,352 triangles, thorough tuning, 2026-10-28) every
    order's winner is among these four: all 3*NTR+1 columns in one batch, 32
    threads up to NEL=15, 64 at NEL=21, 128 up to NEL=45 and 256 at NEL=55,
    retained coupling up to NEL=28, and 256 threads x 16 retained columns for
    the p>=7 Schur-Cholesky kernels. The quick tuner times only the first four
    legal candidates; with the generic ranking it ran 1.1-4x slower.
    """
    threads = 32 if n <= 15 else 64 if n <= 21 else 128 if n <= 45 else 256
    retain = n <= 28
    return [KernelConfiguration(threads, width, retain), KernelConfiguration(threads, width, not retain),
            KernelConfiguration(min(2*threads, 256), width, retain),
            KernelConfiguration(256, min(16, width), True)]


def configurations(context, phase, device, explicit=None):
    """Architecture priors only rank candidates; timings determine the winner."""
    n, nt = int(context.cspace.el_dof), int(context.cspace.edg_dof)
    width = 3*nt+1 if phase == 0 else 1
    block = explicit or (32 if n<=6 else 64)
    first = KernelConfiguration(block, min(4,width) if explicit else 1, False)
    compute_limited = (device.get('fp32_fp64_ratio') or 1) >= 16
    retain = compute_limited and not (phase>0 and _keeps_responses(context.policy))
    ranked = [first, KernelConfiguration(128,min(4,width),retain),
              KernelConfiguration(128,min(8,width),False),
              KernelConfiguration(256,min(8,width),retain)]
    if _fp64_rich_priors_apply(phase, device, explicit):
        ranked = _fp64_rich_assembly_priors(n, width) + ranked
    for threads in (32,64,128,256):
        for batch in sorted({1,2,4,8,16,width}):
            if batch<=width:
                for coupling in ((False,) if phase>0 and _keeps_responses(context.policy) else (False,True)):
                    ranked.append(KernelConfiguration(threads,batch,coupling))
    return list(dict.fromkeys(ranked))


def sample_elements(mesh, limit=2048):
    """Uniformly sample both boundary and interior elements, without RNG state."""
    ne = int(mesh.num_tri)
    if ne<=limit:
        return np.arange(ne,dtype=np.int64)
    boundary = np.any(np.isin(mesh.loc2glob_edge, mesh.bnd_edges_inds),axis=1)
    groups = [np.flatnonzero(boundary),np.flatnonzero(~boundary)]
    chosen=[]
    for group in groups:
        take=min(len(group),limit//2)
        if take:
            chosen.append(group[np.linspace(0,len(group)-1,take,dtype=int)])
    selected=np.concatenate(chosen)
    remaining=limit-len(selected)
    if remaining:
        others=np.setdiff1d(np.arange(ne),selected,assume_unique=True)
        selected=np.concatenate((selected,others[np.linspace(0,len(others)-1,remaining,dtype=int)]))
    return selected.astype(np.int64)


def _launch(context, *, phase, fmt, source_rhs, trace, pattern=None, maps=None,
            rows=None, cols=None, data=None, rhs=None, local=None,
            refresh=False, thorough=False):
    """Tune on isolated samples, launch production work, and check failures."""
    cp = require_cupy()
    preparation_start = time.perf_counter()
    space, mesh = context.cspace, context.cspace.mesh
    n, nt, ne = int(space.el_dof), int(space.edg_dof), int(mesh.num_tri)
    reference=context.reference
    i64=cp.empty(1,dtype=cp.int64)
    i32=cp.empty(1,dtype=cp.int32)
    real=cp.empty(1,dtype=REAL_DTYPE)
    if pattern is not None:
        sides, solve, indptr, positions, offsets = (pattern.interior_side_index,
            pattern.edge_to_solve_edge,pattern.indptr,pattern.side_csr_block_pos,i64)
    elif maps is not None:
        sides,solve,offsets=maps
        indptr,positions=i32,i32
    else:
        sides,solve,offsets=i64,i64,i64
        indptr,positions=i32,i32
    def optional(array, fallback):
        """Supply valid dummy pointers for compile-time unused outputs."""
        return fallback if array is None else array
    def args(elements,sample,count,validation,failures):
        """Pack the common kernel ABI for isolated or production launches."""
        return (elements,np.int32(sample),np.int64(count),
            *(reference[k] for k in ('dd','fd','d','md','fm','lifts','mlifts','edge_mass')),
            mesh.aff_mats,mesh.aff_jacs,mesh.jacs_el_fc,mesh.normals,
            mesh.loc2glob_edge,mesh.loc2oriented_face_coupling,sides,solve,indptr,positions,offsets,
            source_rhs,np.int32(source_rhs.shape[-1]),trace,REAL_DTYPE(context.tau),
            context.factors,context.pivots,context.response,context.source_solution,
            optional(rows,i64),optional(cols,i64),optional(data,real),optional(rhs,real),
            optional(local,real),validation,failures)
    requested=validate_block_size(context.block_size,space.order)
    explicit=None if requested=='auto' else requested
    resolve=not refresh and not thorough
    selection=(phase,fmt,requested)
    if resolve and selection in context.resolved:
        # The selection holds for this context, phase and format for the rest of the process:
        # skip the device query, candidate ranking and record lookup, which cost more than
        # the launched kernel on small meshes (campaigns/poisson_element_cost, 2026-11-02).
        config,kernel,report,shared,static=context.resolved[selection]
        if phase!=2:
            context.boundary_is_zero=not bool(cp.any(trace!=0).item())
        failures=_failure_buffer(context,ne)
        preparation_seconds=time.perf_counter()-preparation_start
        launch_start=time.perf_counter()
        elapsed=event_seconds(cp,lambda:kernel((ne,),(config.threads,),args(i64,0,ne,real,failures),
                                            shared_mem=shared))
        _check_failures(context,phase,ne,failures)
        context.reports[('assembly','rhs','reconstruction')[phase]]=report
        return dict(static,**{'raw.kernel.device':elapsed,'raw.kernel.wall':time.perf_counter()-launch_start,
            'raw.kernel.jit':0.,'raw.kernel.prepare':preparation_seconds,
            'raw.poisson.autotune.wall':0.,'raw.poisson.autotune.reused':1.})
    device=device_signature(cp)
    candidates=configurations(context,phase,device,explicit)
    # Tuning scratch is allocated lazily: cache hits and explicit launches do not
    # allocate a sample or touch production factor/RHS buffers.
    scratch={}
    compilation_seconds = 0.
    def compile_candidate(config):
        """Compile a candidate while recording JIT/lookup wall time."""
        nonlocal compilation_seconds
        begin = time.perf_counter()
        kernel = _compile(context, phase, fmt, config)
        compilation_seconds += time.perf_counter() - begin
        return kernel
    def trial(config,kernel):
        """Run representative elements without changing production buffers."""
        if not scratch:
            ids=cp.asarray(sample_elements(space.host.mesh))
            columns=3*nt+1 if phase==0 else 1
            scratch.update(ids=ids,output=cp.empty((ids.size,columns,3*n+3*nt),dtype=REAL_DTYPE),
                           failures=cp.empty(ids.size,dtype=cp.int32),seen=set())
        if config not in scratch['seen']:
            scratch['output'].fill(cp.nan)
            scratch['seen'].add(config)
        kernel((scratch['ids'].size,),(config.threads,),
            args(scratch['ids'],1,scratch['ids'].size,scratch['output'],scratch['failures']),
            shared_mem=_workspace_bytes(context,phase,config))
        return scratch['output'], scratch['failures']
    identity=dict(family='poisson',source_version=SOURCE_VERSION,phase=phase,order=int(space.order),
        element_basis=space.host.reference.basis_type,trace_basis=context.trace_ref.kind,
        precision=np.dtype(REAL_DTYPE).str,format=fmt,policy=context.policy,
        volume_points=int(space.quad_data.Krf_w.size),
        boundary_nonzero=bool(cp.any(trace!=0).item()),
        workload_bucket=max(0,(ne-1).bit_length()))
    if _fp64_rich_priors_apply(phase, device, explicit):
        # Records tuned under the earlier ranking stay valid for it, not for these priors.
        identity['candidate_priors'] = FP64_RICH_PRIORS
    if phase != 2:
        context.boundary_is_zero = not identity['boundary_nonzero']
    preparation_seconds = time.perf_counter() - preparation_start
    config,kernel,report=select_configuration(cupy=cp,identity=identity,candidates=candidates,
        shared_bytes=lambda c: _workspace_bytes(context,phase,c),compile_kernel=compile_candidate,
        trial=trial,explicit=explicit,refresh=refresh,thorough=thorough,
        baseline=KernelConfiguration(32 if n<=6 else 64, 1, False),
        validate=lambda output: not bool(cp.any(output[1]).item()))
    scratch.clear()
    failures=_failure_buffer(context,ne)
    launch_start = time.perf_counter()
    elapsed=event_seconds(cp,lambda:kernel((ne,),(config.threads,),args(i64,0,ne,real,failures),
                                        shared_mem=_workspace_bytes(context,phase,config)))
    _check_failures(context,phase,ne,failures)
    label=('assembly','rhs','reconstruction')[phase]
    context.reports[label]=report
    timings={'raw.kernel.device':elapsed,'raw.kernel.wall':time.perf_counter()-launch_start,
        'raw.kernel.jit':compilation_seconds, 'raw.kernel.prepare':preparation_seconds,
        'raw.block_size':float(config.threads),'raw.poisson.columns':float(config.columns),
        'raw.poisson.retained_coupling':float(config.retained_coupling),
        'raw.poisson.shared_bytes':float(_workspace_bytes(context,phase,config)),
        'raw.poisson.autotune.reused':float(report['reused']),
        'raw.poisson.autotune.wall':float(report['wall_seconds']),
        'raw.local_factors.bytes':float(context.local_factor_bytes),
        'raw.local_factors.created':float(phase==0 and context.policy!='none'),
        'raw.local_factors.reused':float(phase>0 and context.policy!='none')}
    for key,value in kernel.attributes.items():
        timings['raw.poisson.resource.'+key]=float(value)
    if resolve:
        context.resolved[selection]=(config,kernel,report,_workspace_bytes(context,phase,config),dict(timings))
    return timings


def _full_trace(boundary_trace,cspace):
    """Expand compact boundary values with zero interior trace entries."""
    cp=require_cupy()
    result=cp.zeros((cspace.mesh.num_edg,cspace.edg_dof),dtype=REAL_DTYPE)
    result[cspace.mesh.bnd_edges_inds]=boundary_trace
    return result.reshape(-1)


def assemble(*, source_rhs,boundary_trace,cspace,trace_ref,d0_reference,d1_reference,
             face_element_mass,tau,matrix_format='coo',block_size='auto',
             cache_local_factors=False,local_factor_kind='schur-lu',local_factor_key=None,
             refresh=False,thorough=False):
    """Emit a reduced Poisson matrix and RHS directly in the requested format."""
    from .identity import RawDiffusionAssemblyResult
    cp=require_cupy()
    start=time.perf_counter()
    fmt=str(matrix_format).lower()
    if fmt not in ('coo','csr','bsr'):
        raise ValueError('matrix_format must be coo, csr, or bsr')
    context=_context(cspace,trace_ref,d0_reference,d1_reference,face_element_mass,tau,
        local_factor_kind if cache_local_factors else 'none',block_size,local_factor_key)
    timings={}
    pattern=None
    maps=None
    rows=cols=indptr=indices=None
    ntr=int(cspace.edg_dof)
    mesh=cspace.host.mesh
    if fmt=='coo':
        solve=_edge_to_solve_edge(mesh)
        offsets=_side_flux_offsets(mesh,solve,ntr)
        maps=(cp.asarray(_interior_side_index(mesh)),cp.asarray(solve),cp.asarray(offsets))
        rows=cp.empty(int(offsets[-1]),dtype=cp.int64)
        cols=cp.empty_like(rows)
        data=cp.empty(rows.size,dtype=REAL_DTYPE)
    else:
        pattern=build_reduced_csr_pattern_raw(cspace,timings,matrix_format=fmt)
        indptr,indices=pattern.indptr,pattern.indices
        shape=(pattern.num_blocks,ntr,ntr) if fmt=='bsr' else indices.size
        data=cp.zeros(shape,dtype=REAL_DTYPE)
    rhs=cp.zeros(mesh.int_edges_inds.size*ntr,dtype=REAL_DTYPE)
    timings.update(_launch(context,phase=0,fmt=fmt,source_rhs=source_rhs,
        trace=_full_trace(boundary_trace,cspace),pattern=pattern,maps=maps,
        rows=rows,cols=cols,data=data,rhs=rhs,refresh=refresh,thorough=thorough))
    timings['raw.kernel']=timings['raw.kernel.device']
    timings['raw.total']=timings['raw.wall_total']=time.perf_counter()-start
    timings['raw.unaccounted']=max(0.,timings['raw.total']-sum(timings[k] for k in
        ('raw.kernel.wall','raw.kernel.prepare','raw.poisson.autotune.wall')))
    return RawDiffusionAssemblyResult(rows=rows,cols=cols,data=data,rhs=rhs,source_rhs=source_rhs,
        boundary_trace=boundary_trace,d0_reference=d0_reference,d1_reference=d1_reference,
        face_element_mass=face_element_mass,timings=timings,indptr=indptr,indices=indices,
        matrix_format=fmt,csr_pattern=pattern,
        schur_lu=context.factors if _keeps_lu(context.policy) else None,
        schur_pivots=context.pivots if _keeps_lu(context.policy) else None,
        local_factor_key=local_factor_key,
        local_factor_bytes=context.local_factor_bytes if _keeps_lu(context.policy) else 0,
        local_factor_kind=context.policy,poisson_context=context)


def refresh_rhs(*,source_rhs,boundary_trace,cspace,trace_ref,d0_reference,d1_reference,
                face_element_mass,tau,csr_pattern,block_size='auto',cached_factors=None,
                local_factor_key=None,refresh=False,thorough=False):
    """Refresh source and boundary contributions while reusing compressed topology."""
    from .identity import RawDiffusionAssemblyResult
    if csr_pattern is None or getattr(csr_pattern, 'matrix_format', None) not in ('csr', 'bsr'):
        raise ValueError('Poisson RHS refresh requires a cached CSR or BSR pattern')
    cp=require_cupy()
    context=getattr(cached_factors,'poisson_context',None)
    if context is None:
        context=_context(cspace,trace_ref,d0_reference,d1_reference,face_element_mass,tau,
                         'none',block_size,local_factor_key)
    else:
        _validate_context(context, cspace, trace_ref, tau, local_factor_key)
    context.block_size=block_size
    rhs=cp.zeros(cspace.mesh.int_edges_inds.size*cspace.edg_dof,dtype=REAL_DTYPE)
    timings=_launch(context,phase=1,fmt=csr_pattern.matrix_format,source_rhs=source_rhs,
                    trace=_full_trace(boundary_trace,cspace),pattern=csr_pattern,rhs=rhs,
                    refresh=refresh,thorough=thorough)
    timings['raw.cached_rhs_kernel']=timings['raw.kernel.device']
    return RawDiffusionAssemblyResult(rows=None,cols=None,data=None,rhs=rhs,source_rhs=source_rhs,
        boundary_trace=boundary_trace,d0_reference=d0_reference,d1_reference=d1_reference,
        face_element_mass=face_element_mass,timings=timings,indptr=csr_pattern.indptr,
        indices=csr_pattern.indices,matrix_format=csr_pattern.matrix_format,csr_pattern=csr_pattern,
        schur_lu=context.factors if _keeps_lu(context.policy) else None,
        schur_pivots=context.pivots if _keeps_lu(context.policy) else None,
        local_factor_key=local_factor_key,
        local_factor_bytes=context.local_factor_bytes if _keeps_lu(context.policy) else 0,
        local_factor_kind=context.policy,poisson_context=context)


def reconstruct(*,trace,source_rhs,cspace,trace_ref,d0_reference,d1_reference,
                face_element_mass,tau,block_size='auto',return_local_unknowns=False,
                cached_factors=None,local_factor_key=None,context=None,refresh=False,thorough=False):
    """Recover the scalar field and optionally both physical flux components."""
    cp=require_cupy()
    if context is None:
        context=getattr(cached_factors,'poisson_context',None)
    if context is None:
        context=_context(cspace,trace_ref,d0_reference,d1_reference,face_element_mass,tau,
                         'none',block_size,local_factor_key)
    else:
        _validate_context(context, cspace, trace_ref, tau, local_factor_key)
    context.block_size=block_size
    local=cp.empty((cspace.mesh.num_tri,3*cspace.el_dof),dtype=REAL_DTYPE)
    timings=_launch(context,phase=2,fmt='bsr',source_rhs=source_rhs,
                    trace=cp.ascontiguousarray(trace).reshape(-1),local=local,
                    refresh=refresh,thorough=thorough)
    uh=cp.ascontiguousarray(local[:,:cspace.el_dof])
    elapsed=timings['raw.kernel.device']
    return (uh,local,elapsed) if return_local_unknowns else (uh,elapsed)
