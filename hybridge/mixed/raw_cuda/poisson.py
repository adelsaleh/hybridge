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
import warnings

import numpy as np

from hybridge.hdg.cuda.pattern import build_reduced_csr_pattern_raw
from hybridge.runtime.cuda_tuning import (
    KernelConfiguration, device_signature, event_seconds, select_configuration,
)
from hybridge.hdg.trace_maps import _edge_to_solve_edge, _interior_side_index, _side_flux_offsets
from hybridge.runtime.env import getenv
from hybridge.runtime.optional import require_cupy
from hybridge.runtime.precision import REAL_DTYPE, REAL_ITEMSIZE, real_raw_kernel
from .poisson_source import SOURCE


_KERNELS = {}
SOURCE_VERSION = hashlib.sha256(SOURCE.encode()).hexdigest()


# --- Experimental switches: default off, NOT production ---------------------------------------
# A switch is an environment variable that builds a variant of the kernels below. The variants
# come from the Poisson element-cost study of 2026-11-02 and are kept so that they can be
# measured again. Each one is marked with its result in that study:
#   "FAILED"        it gave no gain or was slower;
#   "NOT PROMOTED"  it worked, but was not made the default code, for the reason given;
#   "Diagnostic"    it measures or checks something and was never meant to be faster.
# The parts of the study that succeeded are the default code now and have no switch; _PROMOTED
# lists their old switch names. With no variable set, the kernels and their tuning records are
# the default ones. Measurements: docs/lessons/poisson_local_solves.md, cited below by section.
# Table of every item: campaigns/poisson_element_cost/README.md.
#
# HYBRIDGE_POISSON_LOCAL_FLAGS=INVERSE   Works, NOT PROMOTED (section 4).
#   Replaces the stored LU factors by explicit local inverses after assembly, so a cached local
#   solve is one matrix-vector product. It was 0.2 to 1.2% of a step faster than the warp solve,
#   for 33 to 69 ms of setup per operator, an inverse residual of 1e-13 and one article row
#   above the parity floor (2.3e-10 against 6.2e-11).
#
# HYBRIDGE_POISSON_KERNEL_FLAGS (comma list of NAME or NAME=VALUE; RHS and reconstruction
# kernels only, never assembly):
#   CONTRACT_Q, CONTRACT_COUPLING, LOAD_WIDE=N   FAILED: no gain at p=9 (section 3).
#     Reference-table contractions of the flux and coupling products, and N overlapped loads in
#     the factor copy to shared memory (a copy only the schur-cholesky kernels still make).
#   CHOLESKY_WARP   Works, NOT PROMOTED. Warp column solve with the factor read in place for
#     the schur-cholesky policy (RHS 8.7 to 2.4 ms at p=9). That policy is not at parity with
#     schur-lu (section 4), so its kernels were left as they are. Needs 32 threads or more.
#   NO_RESPONSE     Diagnostic. Builds the kernels of a response policy without the stored
#     responses, to compare the response sum with a full local solve.
#   PHASE_CLOCK, STOP=k   Diagnostics for the kernel replay of campaigns/poisson_element_cost
#     only: a per-stage clock64 and a kernel truncated after stage k. Both write into the
#     validation argument, which solves do not allocate, so _launch refuses them.
#
# HYBRIDGE_POISSON_KERNEL_OPTIONS   Extra NVRTC options for every Poisson kernel. FAILED as
#   tried: --ptxas-options=-O1 was slower, --maxrregcount=80 a small gain that the warp solve
#   superseded (section 3).
#
# HYBRIDGE_POISSON_HOST_FLAGS=boundary   Works, NOT PROMOTED: small gain, 0.38 to 0.30 ms per refresh.
#   mixed/cupy.py keeps the boundary quadrature points instead of rebuilding them per solve.
_PROMOTED = {
    'RESPONSES': 'cache_local_factors="schur-lu+responses" (2026-11-02)',
    'WARP_SOLVE': 'the warp LU solve of cached factors (2026-11-02)',
    'DIRECT_FACTORS': 'the in-place read of cached LU factors (2026-11-02)',
    'SKIP_INTERIOR': 'the interior skip of the RHS phase (2026-11-02)',
    'resolved': 'the launch resolved once per context (2026-11-02)',
    'nocheck': 'the status read back only after a factorization (2026-11-02)',
    'signature': 'the device query made once per process (2026-11-02)',
}
_MOVED = {('POISSON_KERNEL_FLAGS', 'INVERSE'): 'HYBRIDGE_POISSON_LOCAL_FLAGS=INVERSE'}
_KERNEL_SWITCHES = frozenset({'CONTRACT_Q', 'CONTRACT_COUPLING', 'LOAD_WIDE', 'CHOLESKY_WARP',
                              'NO_RESPONSE', 'PHASE_CLOCK', 'STOP'})
_REPLAY_ONLY = ('PHASE_CLOCK', 'STOP')


def _experimental_names(variable, known):
    """Return the switches listed in the environment variable ``HYBRIDGE_<variable>``.

    The value is a comma-separated list of ``NAME`` or ``NAME=VALUE`` items, and the items whose
    name is in ``known`` are returned in the order given. A name in ``_PROMOTED`` is left out
    with a warning, because that code is the default now. A name in ``_MOVED`` or any other
    name raises ``ValueError``.
    """
    names = []
    for item in filter(None, (part.strip() for part in (getenv(variable, '') or '').split(','))):
        name = item.partition('=')[0]
        if name in _PROMOTED:
            warnings.warn(f'HYBRIDGE_{variable}: {name} is production now, as {_PROMOTED[name]}; '
                          'the switch has no effect', stacklevel=2)
        elif (variable, name) in _MOVED:
            raise ValueError(f'HYBRIDGE_{variable}: {name} moved; set {_MOVED[(variable, name)]}')
        elif name not in known:
            raise ValueError(f'HYBRIDGE_{variable}: unknown experimental switch {name!r}; '
                             f'known: {", ".join(sorted(known))}')
        else:
            names.append(item)
    return names


def _kernel_flags(names):
    """Return ``{name: integer value}`` for items ``NAME`` (value 1) or ``NAME=VALUE``.

    The values become the kernel macros ``EXP_<name>``. An item with value 0 is left out, so
    that a switch set to 0 is the same as a switch that is not set.
    """
    flags = {}
    for item in names:
        name, _, value = item.partition('=')
        flags[name] = int(value) if value else 1
    return {name: value for name, value in flags.items() if value}


LOCAL_FLAGS = frozenset(_experimental_names('POISSON_LOCAL_FLAGS', {'INVERSE'}))
KERNEL_FLAGS = _kernel_flags(_experimental_names('POISSON_KERNEL_FLAGS', _KERNEL_SWITCHES))
COMPILE_OPTIONS = tuple((getenv('POISSON_KERNEL_OPTIONS', '') or '').split())
HOST_FLAGS = frozenset(_experimental_names('POISSON_HOST_FLAGS', {'boundary'}))


def _phase_flags(phase):
    """Return the kernel switches of one phase (0: assembly, 1: right-hand side, 2: reconstruction).

    The switches apply to phases 1 and 2. Assembly always gets an empty dictionary.
    """
    return KERNEL_FLAGS if phase > 0 else {}


def _experiment(phase):
    """Return what distinguishes the kernel of one phase from the default kernel.

    The result is ``()`` when no kernel switch and no extra compiler option is set. Otherwise it
    is the pair (sorted switch items, compiler options). It is part of the keys under which
    kernels are compiled and launch sizes are stored, so a variant never takes the place of the
    default kernel.
    """
    flags = _phase_flags(phase)
    return (tuple(sorted(flags.items())), COMPILE_OPTIONS) if flags or COMPILE_OPTIONS else ()


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
    """The RHS phase solves with stored pivoted LU factors."""
    return policy in ('schur-lu', 'schur-lu+responses')


def _inverse(context):
    """Whether the stored LU factors of this context are replaced by explicit local inverses.

    Experimental, not promoted: true only with ``HYBRIDGE_POISSON_LOCAL_FLAGS=INVERSE`` and a
    cache option that stores LU factors.
    """
    return _keeps_lu(context.policy) and 'INVERSE' in LOCAL_FLAGS


def _applies_responses(context, phase):
    """Whether the kernel of this phase uses the stored trace responses.

    It does whenever the cache option stores them, except under the diagnostic switch
    ``NO_RESPONSE``, which makes phases 1 and 2 solve the local system again instead.
    """
    return _keeps_responses(context.policy) and not _phase_flags(phase).get('NO_RESPONSE', 0)


def _cholesky_warp(context, phase):
    """Whether this phase solves with the stored Cholesky factor on one warp, reading it in place.

    Experimental, not promoted: true only with the switch ``CHOLESKY_WARP``, the cache option
    ``schur-cholesky``, and a phase that solves locally: phase 1, and phase 2 when it does not
    use the stored responses.
    """
    return (phase > 0 and context.policy == 'schur-cholesky' and bool(_phase_flags(phase).get('CHOLESKY_WARP', 0))
            and not (phase == 2 and _applies_responses(context, phase)))


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
    return phase > 0 and _keeps_lu(context.policy) and not (phase == 2 and _applies_responses(context, phase))


def _workspace_bytes(context, phase, config):
    """Omit shared factor storage for response-based reconstruction and in-place LU factors."""
    nel = int(context.cspace.el_dof)
    warp = _warp_solve(context, phase)
    in_place = warp or _cholesky_warp(context, phase)
    return shared_bytes(nel, int(context.cspace.edg_dof), config,
                        factor=not in_place and not (phase == 2 and _applies_responses(context, phase))) + (
        nel*REAL_ITEMSIZE + nel*4 if warp else 0) + (  # Pivot reciprocals and the folded row permutation.
        # Experimental layouts: Cholesky pivot reciprocals, and the five contracted coupling tables.
        (nel*REAL_ITEMSIZE if in_place and not warp else 0)
        + (5*nel*REAL_ITEMSIZE if _phase_flags(phase).get('CONTRACT_COUPLING', 0) else 0))


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
    key = (cp.cuda.runtime.getDevice(), n, nt, phase, fmt, context.policy, _inverse(context),
           config.columns, config.retained_coupling, np.dtype(REAL_DTYPE).str, _experiment(phase))
    if key not in _KERNELS:
        warp = _warp_solve(context, phase)
        flags = _phase_flags(phase)
        cholesky_warp = _cholesky_warp(context, phase)
        if cholesky_warp and config.threads < 32:
            raise ValueError('the experimental CHOLESKY_WARP solve needs a block of at least 32 threads')
        responses = _applies_responses(context, phase)
        constants = dict(NEL=n, NTR=nt, PHASE=phase, FORMAT={'coo':0,'csr':1,'bsr':2}[fmt],
                         FACTOR_STORAGE=0 if warp or cholesky_warp or (phase == 2 and responses) else n*n,
                         WARP_SOLVE=int(warp), INVERSE=int(warp and _inverse(context)),
                         BATCH=config.columns, RETAIN=int(config.retained_coupling),
                         CHOLESKY=int(context.policy=='schur-cholesky'),
                         RESPONSE=int(responses),
                         USE_FACTORS=int(phase>0 and context.policy!='none'),
                         WRITE_FACTORS=int(phase==0 and context.policy!='none'),
                         SYMMETRY_RTOL=max(5.e-11, 32*np.finfo(REAL_DTYPE).eps))
        # Experimental switches, all 0 in production (see the top of this module).
        constants.update(EXP_CONTRACT_Q=flags.get('CONTRACT_Q', 0),
                         EXP_CONTRACT_COUPLING=flags.get('CONTRACT_COUPLING', 0),
                         EXP_LOAD_WIDE=flags.get('LOAD_WIDE', 0), EXP_CHOLESKY_WARP=int(cholesky_warp),
                         EXP_PHASE_CLOCK=flags.get('PHASE_CLOCK', 0), EXP_STOP=flags.get('STOP', 0))
        source = ''.join(f'#define {name} {value}\n' for name,value in constants.items())+SOURCE
        kernel = real_raw_kernel(source, 'poisson_local', options=('--std=c++11',)+COMPILE_OPTIONS)
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
    retain = compute_limited and not (phase>0 and _applies_responses(context, phase))
    ranked = [first, KernelConfiguration(128,min(4,width),retain),
              KernelConfiguration(128,min(8,width),False),
              KernelConfiguration(256,min(8,width),retain)]
    if _fp64_rich_priors_apply(phase, device, explicit):
        ranked = _fp64_rich_assembly_priors(n, width) + ranked
    for threads in (32,64,128,256):
        for batch in sorted({1,2,4,8,16,width}):
            if batch<=width:
                for coupling in ((False,) if phase>0 and _applies_responses(context, phase) else (False,True)):
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
    if any(_phase_flags(phase).get(name, 0) for name in _REPLAY_ONLY):
        raise ValueError('the experimental PHASE_CLOCK and STOP kernels write into the validation argument, '
                         'which a solve does not allocate; launch them from the kernel replay of '
                         'campaigns/poisson_element_cost/harness.py')
    resolve=not refresh and not thorough
    selection=(phase,fmt,requested,_experiment(phase))
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
    if LOCAL_FLAGS and _keeps_lu(context.policy):
        identity['local_flags']=sorted(LOCAL_FLAGS)  # Other kernels: never reuse a record of the plain ones.
    if _experiment(phase):
        identity['experiment']=[sorted(_phase_flags(phase).items()),list(COMPILE_OPTIONS)]
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


def local_inverses(context):
    """Return the inverse of every local matrix and the largest inverse residual.

    Experimental, not promoted. ``context.factors`` holds, per element, the factors of
    ``P A = L U`` with the row interchanges in ``context.pivots``. Each ``A`` is rebuilt from
    them and inverted with batched cuBLAS. The first result has the shape of
    ``context.factors`` and replaces it in ``assemble``, so that array no longer holds LU
    factors afterwards. The second result is the largest infinity norm of ``A A^-1 - I``.
    """
    from hybridge.linalg.gpu.cublas_batched import invert_batched_cublas
    cp=require_cupy()
    n=int(context.cspace.el_dof)
    factors=context.factors
    matrix=(cp.tril(factors,-1)+cp.eye(n,dtype=REAL_DTYPE))@cp.triu(factors)
    elements=cp.arange(factors.shape[0])
    # PA=LU: undo the row interchanges, last first, across all elements at once.
    for k in range(n-1,-1,-1):
        p=context.pivots[:,k]
        kept=matrix[elements,k].copy()
        matrix[elements,k]=matrix[elements,p]
        matrix[elements,p]=kept
    result=invert_batched_cublas(cp.ascontiguousarray(matrix),label='Poisson local operators')
    return cp.ascontiguousarray(result.inverse_matrices),result.maximum_inverse_residual


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
    if _inverse(context):
        inverse_start=time.perf_counter()
        context.factors,inverse_residual=local_inverses(context)
        timings['raw.local_factors.inverse_seconds']=time.perf_counter()-inverse_start
        timings['raw.local_factors.inverse_residual']=float(inverse_residual)
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
