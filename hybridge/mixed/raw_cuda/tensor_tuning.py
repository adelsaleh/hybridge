"""Bounded ADR execution calibration and representation-safe mass-cache reuse.

The sample kernels use the production algebra and actual mesh/coefficient
tables. They write local condensed operators and solution responses into
private slots, so calibration cannot modify a production matrix or cache.
Dense same-precision mixed equations provide an independent backward-error
gate; sampled reconstruction must also equal the assembled local response.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
import uuid
import weakref

import numpy as np

from hybridge.runtime.cuda_tuning import (
    CandidateUnavailable, KernelConfiguration, UnrepresentativeInput, device_signature,
    fingerprint, kernel_attributes, legal_configuration, resolved_selection, select_variant,
)
from hybridge.runtime.precision import REAL_DTYPE, REAL_ITEMSIZE, real_raw_module


SAMPLE_ELEMENTS = 256
SAMPLE_BYTES = 64 * 1024 * 1024
_IDENTITY_TOKENS = {}
_SAMPLE_SOURCE = r'''
extern "C" __global__ void sample_adr_element_indices(
        const long long* kinds, long long elements, const long long* present,
        int classes, int quota, long long* output) {
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;
    if(slot>=classes*quota) return;
    const int kind=present[slot/quota], bin=slot%quota;
    const long long first=elements*bin/quota, last=elements*(bin+1)/quota;
    long long selected=-1;
    for(long long e=first;e<last;++e) if(kinds[e]==kind) {selected=e;break;}
    output[slot]=selected;
}
'''


class ADRMassFactors(tuple):
    """A two-array cache whose kind-6 storage is explicitly LU or inverse.

    Tuple unpacking, indexing and the existing ``mass_factors=`` argument are
    preserved. The signature describes the space/storage contract; callers
    remain responsible for reusing only an unchanged diffusion coefficient.
    """

    def __new__(cls, mass, pivots, *, mass_inverse, signature):
        """Attach representation metadata without changing the two-array API."""
        result = super().__new__(cls, (mass, pivots))
        result.mass_inverse = bool(mass_inverse)
        result.signature = signature
        return result


def array_digest(*arrays):
    """Hash small host reference tables with shape and dtype boundaries."""
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _identity_token(value):
    """Bind a retained cache to a live object without hashing mesh-sized arrays.

    Weak references prevent stale entries and UUID tokens prevent recycled
    Python IDs from making a different space or mesh appear compatible.
    This follows the identity ownership of the existing device-space caches.
    """
    key = id(value)
    entry = _IDENTITY_TOKENS.get(key)
    if entry is not None and entry[0]() is value:
        return entry[1]
    token = uuid.uuid4().hex

    def forget(reference):
        """Drop this identity only if its original weak reference still owns it."""
        current = _IDENTITY_TOKENS.get(key)
        if current is not None and current[0] is reference:
            del _IDENTITY_TOKENS[key]

    _IDENTITY_TOKENS[key] = (weakref.ref(value, forget), token)
    return token


def mass_cache_signature(space, trace_space, *, mass_stride, max_kind):
    """Identify mass layout, precision, reference data and owning mesh/space.

    Equal-size spaces on differently scaled/reordered meshes are incompatible.
    Geometry is owned by the frozen DGMesh and its cached device mirror, so
    cache reuse follows object ownership rather than rehashing every element.
    """
    return fingerprint(dict(elements=space.mesh.num_tri, nel=space.el_dof,
        space=_identity_token(space), mesh=_identity_token(space.mesh),
        reference_owner=_identity_token(space.reference),
        stride=mass_stride, maximum_kind=max_kind, dtype=np.dtype(REAL_DTYPE).str,
        reference=array_digest(space.quad_data.bas_of_quads, space.quad_data.Krf_w),
        face_points=int(trace_space.weights.size)))


def reusable_mass_cache(cache, *, elements, nel, mass_stride, max_kind, order,
                        signature, requested_inverse=None):
    """Return a proven representation, or ``None`` when the cache must rebuild.

    An untagged historical pair is unambiguously factorized below p=7 or
    without kind 6, so those existing calls still reuse it. High-order general
    tensors may contain either representation and are rebuilt once safely.
    An explicit inverse/factor override also rebuilds incompatible caches.
    """
    if not mass_stride or cache is None or len(cache) != 2:
        return None
    mass, pivots = cache
    if mass.shape != (elements, mass_stride) or mass.dtype != np.dtype(REAL_DTYPE):
        return None
    if max_kind == 6 and (pivots.shape != (elements, 2 * nel) or pivots.dtype != np.dtype(np.int32)):
        return None
    if isinstance(cache, ADRMassFactors):
        if cache.signature != signature:
            return None
        inverse = cache.mass_inverse
    elif order < 7 or max_kind < 6:
        inverse = False
    else:
        return None
    return inverse if requested_inverse is None or inverse == requested_inverse else None


@dataclass(frozen=True)
class ADRConfiguration:
    """Equivalent element algorithm, shared layout and CUDA launch shape."""

    threads: int
    workspace: object
    maxrregcount: int | None = None

    @property
    def name(self):
        """Return a stable name used by the persistent shared selector."""
        work = self.workspace
        return (f't{self.threads}-b{work.batch}-c{int(work.compact)}'
                f'-e{int(work.external_mass)}-i{int(work.mass_inverse)}'
                f'-r{self.maxrregcount or 0}')


def candidate_configurations(*, nel, kinds, nq, nfq, order, block_size,
                             workspace, shared_limit, device, placement='auto',
                             inverse_mode='auto', retained_inverse=None):
    """Order legal layouts by historical priors without declaring a winner.

    The conservative factorized layout stays first. Inverse and Hopper wide
    launches are early alternatives, followed by placement, batch and thread
    changes. Overrides constrain the roster; retained factors lock its
    representation until the caller deliberately rebuilds them.
    """
    from hybridge.mixed.raw_cuda import tensor as raw

    maximum = int(np.max(kinds)) if np.size(kinds) else 0
    minimum_inverse_bytes = raw.tensor_shared_bytes(nel, maximum, nq, nfq, 1, compact=True)[1] + REAL_ITEMSIZE * 2 * nel
    inverse_allowed = (order >= 7 and maximum == 6 and placement != 'external'
                       and minimum_inverse_bytes <= shared_limit)
    required = retained_inverse
    if required is None and inverse_mode != 'auto' and inverse_allowed:
        required = inverse_mode == 'on'
    entries = {}

    def add(threads, compact, external, inverse, batch=None, cap=None):
        """Keep a requested configuration only if all workspace limits fit."""
        if required is not None and inverse != required:
            return
        if inverse and (not inverse_allowed or external):
            return
        if order >= 7 and placement != 'auto' and external != (placement == 'external'):
            return
        if order <= 6 and external:
            return
        widths = (batch,) if batch is not None else range(raw.TENSOR_MAX_BATCH_COLUMNS, 0, -1)
        for width in widths:
            storage, size = raw.tensor_shared_bytes(nel, maximum, nq, nfq, width,
                                                    compact=compact, external_mass=external)
            if inverse:
                size += REAL_ITEMSIZE * 2 * nel * width
            plan = raw.TensorExecutionWorkspace(width, storage, size, compact, external, inverse)
            if not legal_configuration(KernelConfiguration(threads), size,
                                       dict(device, sharedMemPerBlockOptin=shared_limit)):
                continue
            config = ADRConfiguration(threads, plan, cap)
            entries.setdefault(config.name, config)
            return

    baseline_inverse = bool(required)
    base_external = (placement == 'external') if order >= 7 and placement != 'auto' else workspace.external_mass
    add(block_size, workspace.compact, base_external, baseline_inverse,
        batch=workspace.batch)
    if not entries:
        add(block_size, workspace.compact, base_external, baseline_inverse)
    if not entries:
        raise raw.TensorWorkspaceError('ADR retained mass representation does not fit the selected device workspace')
    preferred_inverse = inverse_allowed and required is not False
    hopper = int(device.get('major') or 0) == 9
    if hopper and order >= 9 and maximum >= 3:
        add(512, False, False, preferred_inverse, cap=128)
    add(block_size, workspace.compact, workspace.external_mass, preferred_inverse)
    if order >= 7:
        add(256, True, False, preferred_inverse)
        add(128, True, True, False)
        add(128, False, False, preferred_inverse)
        add(128, True, False, False)
        if not hopper and order >= 9 and maximum >= 3:
            add(512, False, False, preferred_inverse, cap=128)
    else:
        add(block_size, True, False, False)
        add(128, workspace.compact, False, False)
    add(block_size, workspace.compact, workspace.external_mass, preferred_inverse,
        batch=max(1, workspace.batch // 2))
    add(64, workspace.compact, workspace.external_mass, preferred_inverse)
    add(256, workspace.compact, workspace.external_mass, preferred_inverse)
    return entries


def sample_element_indices(cupy, kinds, *, elements, nel, ntr, present_kinds=None):
    """Select bounded, spread-out actual elements covering every diffusion kind."""
    # Includes the dense residual oracle, response copies, factor caches and
    # intermediate contractions. The estimate deliberately exceeds live sizes.
    per_element = REAL_ITEMSIZE * (32 * nel * nel + 24 * nel * (3 * ntr + 1)
                                   + 8 * (3 * ntr) * (3 * ntr + 1))
    limit = min(elements, SAMPLE_ELEMENTS, max(7, SAMPLE_BYTES // max(1, per_element)))
    if present_kinds is None:
        present_kinds = [kind for kind in range(7) if bool(cupy.any(kinds == kind))]
    quota = max(1, limit // max(1, len(present_kinds)))
    if isinstance(kinds, np.ndarray):
        selected = []
        for kind in present_kinds:
            for bin_index in range(quota):
                first, last = elements*bin_index//quota, elements*(bin_index+1)//quota
                matches = np.flatnonzero(kinds[first:last] == kind)
                if matches.size:
                    selected.append(first + matches[0])
        return np.unique(selected).astype(np.int64)
    present = cupy.asarray(present_kinds, dtype=cupy.int64)
    selected = cupy.empty(present.size * quota, dtype=cupy.int64)
    kernel = cupy.RawKernel(_SAMPLE_SOURCE, 'sample_adr_element_indices')
    kernel(((selected.size+127)//128,), (128,),
        (kinds, np.int64(elements), present, np.int32(present.size), np.int32(quota), selected))
    return cupy.ascontiguousarray(cupy.unique(selected[selected >= 0]))


def _coefficient_class(cupy, array):
    """Describe bounded samples while downloading only one scalar bit mask."""
    value = array.reshape(-1)
    positions = np.linspace(0, max(0, value.size - 1), min(64, value.size), dtype=np.int64)
    xp = np if isinstance(value, np.ndarray) else cupy
    sample = value[xp.asarray(positions)]
    flags = 2 if not sample.size else int(
        xp.any(sample).astype(xp.int32)
        + 2 * xp.all(sample == sample[0]).astype(xp.int32)
        + 4 * xp.any(sample > 0).astype(xp.int32)
        + 8 * xp.any(sample < 0).astype(xp.int32))
    return dict(shape=list(array.shape[1:]), zero=not bool(flags & 1),
                constant=bool(flags & 2), positive=bool(flags & 4), negative=bool(flags & 8))


def _mixed_oracle(cupy, *, space, trace_space, prepared, tensor, indices):
    """Build independent same-precision mixed equations on sampled elements."""
    cp, q, mesh = cupy, space.quad_data, space.mesh
    if not isinstance(indices, np.ndarray):
        from hybridge.core.device import as_cupy_space
        mesh = as_cupy_space(space).mesh

    def samples(value):
        """Gather bounded rows using indices in their existing memory space."""
        if isinstance(value, np.ndarray) and isinstance(indices, np.ndarray):
            return cp.asarray(value[indices], dtype=REAL_DTYPE)
        return cp.asarray(value, dtype=REAL_DTYPE)[indices]

    ne, n, nt = indices.size, space.el_dof, trace_space.edg_dof
    basis, weights = cp.asarray(q.bas_of_quads), cp.asarray(q.Krf_w)
    jac = samples(mesh.aff_jacs)
    gradients = cp.einsum('Kab,biq->Kaiq', samples(mesh.inv_aff_mats_t),
                          cp.asarray(q.dbas_of_quads))
    derivative = cp.einsum('Kaiq,jq,q,K->Kaij', gradients, basis, weights, jac)
    if tensor.inverse_values.size:
        inverse = samples(tensor.inverse_values)
    else:
        from hybridge.mixed.coefficients import inverse_diffusion_values
        constant = inverse_diffusion_values(cp.asarray(tensor.constants, dtype=REAL_DTYPE))[0]
        inverse = cp.broadcast_to(constant, (ne, weights.size, 4))
    mass = cp.einsum('Kqc,iq,jq,q,K->Kcij', inverse, basis, basis, weights, jac)
    reaction, beta = samples(prepared.reaction_values), samples(prepared.beta_values)
    scalar = cp.einsum('Kq,iq,jq,q,K->Kij', reaction, basis, basis, weights, jac)
    scalar -= cp.einsum('Kqa,Kaiq,jq,q,K->Kij', beta, gradients, basis, weights, jac)
    phi, fw = cp.asarray(trace_space.bas_of_bd_quads), cp.asarray(trace_space.weights)
    orientations = (~mesh.orientations[indices]).astype(cp.int64)
    psi = cp.asarray(trace_space.oriented_basis_table)[orientations]
    fj, normals = samples(mesh.jacs_el_fc), samples(mesh.normals)
    tau, gamma = samples(prepared.tau_total), samples(prepared.gamma)
    normal = cp.einsum('q,fiq,fjq,Kfa,Kf->Kaij', fw, phi, phi, normals, fj)
    scalar += cp.einsum('q,fiq,fjq,Kfq,Kf->Kij', fw, phi, phi, tau, fj)
    matrix = cp.zeros((ne, 3*n, 3*n), dtype=REAL_DTYPE)
    blocks = matrix.reshape(ne, 3, n, 3, n)
    blocks[:, 0, :, 0] = scalar
    for dim in range(2):
        blocks[:, 0, :, dim+1] = normal[:, dim] - derivative[:, dim]
        blocks[:, dim+1, :, 0] = derivative[:, dim]
        for other in range(2):
            blocks[:, dim+1, :, other+1] = -mass[:, 2*dim+other]
    coupling = cp.einsum('q,fiq,Kfdq,Kf->Kifd', fw, phi, psi, fj)
    gamma_coupling = cp.einsum('q,fiq,Kfdq,Kfq,Kf->Kifd', fw, phi, psi, gamma, fj)
    rhs = cp.zeros((ne, 3*n, 3*nt+1), dtype=REAL_DTYPE)
    rhs[:, :, :-1] = cp.concatenate((gamma_coupling,
        coupling*normals[:, None, :, 0, None], coupling*normals[:, None, :, 1, None]), axis=1).reshape(ne, 3*n, 3*nt)
    rhs[:, :n, -1] = samples(prepared.source_rhs)
    return matrix, rhs


def select_tensor_execution(*, cupy, space, trace_space, prepared, tensor, args,
                            matrix_format, workspace, block_size, shared_limit,
                            cached_factors, mass_stride, reuse_mass, retained_inverse=None):
    """Select the ADR execution once per space, trace space, diffusion classes and cache policy.

    Rebuilding the identity cost 4.6 ms per assembly at p=6 and repeated device read-backs
    (campaigns/adr_selection_cost, 2026-11-02). The key omits the sampled coefficient classes on
    purpose: they do not change the kernel source, the candidates or the production status checks,
    only the sample on which the first measurement ran, so a later flip keeps the selection.
    A conservative fallback chosen because the sampled input itself fails (a singular local
    system) is used for this call only; the next call with this key measures again.
    """
    from hybridge.mixed.raw_cuda import tensor as raw

    key = (cupy.cuda.runtime.getDevice(), _identity_token(space), _identity_token(trace_space),
           tuple(bool(count) for count in tensor.counts.values()), matrix_format, workspace,
           block_size, shared_limit, cached_factors, mass_stride, reuse_mass, retained_inverse,
           raw.tensor_mass_placement(), raw.tensor_mass_inverse_mode())
    selection, reused = resolved_selection(key, lambda: _measure_tensor_execution(
        cupy=cupy, space=space, trace_space=trace_space, prepared=prepared, tensor=tensor, args=args,
        matrix_format=matrix_format, workspace=workspace, block_size=block_size,
        shared_limit=shared_limit, cached_factors=cached_factors, mass_stride=mass_stride,
        reuse_mass=reuse_mass, retained_inverse=retained_inverse),
        keep=lambda selection: selection[3].get('source') != 'unrepresentative-input')
    if not reused:
        return selection
    config, module, kernel, record = selection
    return config, module, kernel, dict(record, reused=True, source='resolved', wall_seconds=0.,
                                        sample_elements=0)


def _measure_tensor_execution(*, cupy, space, trace_space, prepared, tensor, args,
                              matrix_format, workspace, block_size, shared_limit,
                              cached_factors, mass_stride, reuse_mass, retained_inverse=None):
    """Compile, validate and time equivalent ADR algorithms on isolated samples."""
    from hybridge.hdg.trace_maps import _trace_orientation_mode
    from hybridge.mixed.raw_cuda import tensor as raw

    cp = cupy
    device = device_signature(cp)
    counts = tensor.counts
    # The host count keys follow DIFFUSION_KINDS; no full device classification
    # array is transferred merely to construct a roster or cache key.
    kinds = np.array([kind for kind, count in enumerate(counts.values()) if count], dtype=np.int64)
    n, nt, nq, nfq = space.el_dof, trace_space.edg_dof, space.quad_data.Krf_w.size, trace_space.weights.size
    configs = candidate_configurations(nel=n, kinds=kinds, nq=nq, nfq=nfq,
        order=space.order, block_size=block_size, workspace=workspace,
        shared_limit=shared_limit, device=device, placement=raw.tensor_mass_placement(),
        inverse_mode=raw.tensor_mass_inverse_mode(), retained_inverse=retained_inverse)
    common = dict(nel=n, ntr=nt, nq=nq, nfq=nfq,
        orientation=_trace_orientation_mode(trace_space), matrix_format=matrix_format,
        cached_factors=cached_factors, mass_stride=mass_stride, reuse_mass=reuse_mass,
        cooperative_only=space.order > 6, tuning=True)
    sources = {}
    for name, config in configs.items():
        work = config.workspace
        sources[name] = raw.kernel_source(**common, batch=work.batch, storage=work.storage,
            compact=work.compact, external_mass=work.external_mass, mass_inverse=work.mass_inverse)
    identity = dict(operation='adr-tensor-execution-v1', precision=np.dtype(REAL_DTYPE).str,
        source=fingerprint(dict(kernels=sources, adapter=Path(__file__).read_text())),
        order=space.order, nel=n, ntr=nt, nq=nq, nfq=nfq,
        basis=space.reference.basis_type, trace_basis=str(trace_space.kind),
        quadrature=array_digest(space.quad_data.bas_of_quads, space.quad_data.dbas_of_quads,
                               space.quad_data.Krf_w, trace_space.quads, trace_space.weights,
                               trace_space.bas_of_bd_quads, trace_space.bas1d_of_ref_edg_qds),
        matrix_format=matrix_format, cache_policy=(cached_factors, mass_stride > 0),
        reuse_mass=reuse_mass, mass_representation=retained_inverse,
        coefficient_classes={name: _coefficient_class(cp, getattr(prepared, name))
            for name in ('reaction_values', 'beta_values', 'tau_total', 'gamma', 'source_rhs')},
        diffusion_classes={name: int(value > 0) for name, value in counts.items()},
        size_bucket=1 << max(0, (max(1, space.mesh.num_tri)-1).bit_length()),
        sample_limit=(SAMPLE_ELEMENTS, SAMPLE_BYTES), shared_limit=shared_limit)
    trial = {}

    def prepare(name):
        """Compile and screen production and sample kernels without allocating trials."""
        config = configs[name]
        try:
            module = real_raw_module(code=sources[name], options=raw._compile_options(config.maxrregcount))
            kernels = [module.get_function(entry) for entry in
                ('assemble_adr_tensor', 'sample_adr_tensor', 'sample_reconstruct_adr_tensor')]
            attributes = [kernel_attributes(cp, kernel) for kernel in kernels]
        except (cp.cuda.compiler.CompileException, ValueError) as error:
            raise CandidateUnavailable(str(error)) from error
        for kernel, attributes_one in zip(kernels, attributes):
            if not legal_configuration(KernelConfiguration(config.threads), config.workspace.shared_bytes,
                                       device, attributes_one):
                raise CandidateUnavailable('ADR kernel exceeds compiled thread/register/shared-memory limits')
            if config.workspace.shared_bytes > raw.TENSOR_SHARED_MEMORY_LIMIT:
                kernel.max_dynamic_shared_size_bytes = config.workspace.shared_bytes
        return dict(config=config, module=module, kernels=kernels, attributes=attributes)

    def initialize_trial():
        """Allocate bounded scratch and independent equations only on a cache miss."""
        indices = sample_element_indices(cp, args[-9], elements=space.mesh.num_tri, nel=n, ntr=nt,
                                          present_kinds=kinds.tolist())
        count = indices.size
        trial.update(indices=indices, count=count,
            matrix=cp.empty((count, 3*nt, 3*nt+1), dtype=REAL_DTYPE),
            states=cp.empty((count, 3*n, 3*nt+1), dtype=REAL_DTYPE),
            unknowns=cp.empty((count, 3*n), dtype=REAL_DTYPE),
            schur=cp.empty((count, n, n) if cached_factors else 1, dtype=REAL_DTYPE),
            # Zero, not empty: the sampled reconstruction reads these pivots as shared-memory
            # row indices (lu_solve_setup) even for an element whose assembly factorization
            # failed and never wrote them; uninitialized values index out of bounds.
            pivots=cp.zeros((count, n) if cached_factors else 1, dtype=cp.int32),
            mass=cp.empty((count, mass_stride) if mass_stride else 1, dtype=REAL_DTYPE),
            mass_pivots=cp.zeros((count, 2*n) if mass_stride else 1, dtype=cp.int32),
            statuses=cp.zeros(count, dtype=cp.int32), reconstruction_statuses=cp.zeros(count, dtype=cp.int32))
        trial['trace'] = cp.sin(cp.arange(count * 3*nt, dtype=REAL_DTYPE).reshape(count, 3*nt) + REAL_DTYPE(.37))
        if reuse_mass:
            trial['saved_mass'] = args[-3][indices].copy()
            if args[-2].size > 1:
                trial['saved_pivots'] = args[-2][indices].copy()
        # Reuse the production uploads even for host-prepared coefficients.
        # Copying those complete tables again would defeat the sample bound;
        # downloading sample indices would violate the device-resident API.
        sampled_prepared = replace(prepared, **dict(zip(
            ('reaction_values', 'beta_values', 'tau_total', 'gamma', 'source_rhs'),
            args[-15:-10])))
        sampled_tensor = replace(tensor, constants=args[-8], inverse_values=args[-7])
        trial['oracle'] = _mixed_oracle(cp, space=space, trace_space=trace_space,
            prepared=sampled_prepared, tensor=sampled_tensor, indices=indices)
        maximum_storage = max(config.workspace.storage for config in configs.values())
        trial['scratch'] = cp.empty((count, maximum_storage), dtype=REAL_DTYPE)

    def before(name, handle):
        """Restore private statuses and retained input factors outside the event interval."""
        if not trial:
            initialize_trial()
        trial['statuses'].fill(0)
        trial['reconstruction_statuses'].fill(0)
        if reuse_mass:
            cp.copyto(trial['mass'], trial['saved_mass'])
            if 'saved_pivots' in trial:
                cp.copyto(trial['mass_pivots'], trial['saved_pivots'])

    def run(name, handle):
        """Run assembly and cache-consistent reconstruction using only private outputs."""
        config, kernels = handle['config'], handle['kernels']
        private = (*args[:-5], trial['schur'], trial['pivots'], trial['mass'], trial['mass_pivots'])
        suffix = (trial['indices'], trial['matrix'], trial['states'], trial['trace'])
        launch = dict(elements=trial['count'], block_size=config.threads, workspace=config.workspace,
                      scratch=trial['scratch'], chunk=trial['count'])
        raw._launch_tensor_kernel(kernels[1], (*private, trial['statuses']), **launch, suffix=suffix)
        raw._launch_tensor_kernel(kernels[2], (*private, trial['reconstruction_statuses']), **launch,
                                  suffix=(*suffix, args[-10], trial['unknowns']))
        return (trial['matrix'], trial['states'], trial['unknowns'])

    def validate(output):
        """Check factor statuses, independent mixed residuals and reconstruction consistency.

        A failed assembly factorization means the sampled local systems are singular or
        nonfinite; the production assembly reports that per element, so it is raised as an
        input failure rather than a candidate failure.
        """
        failed = int(cp.count_nonzero(trial['statuses']))
        if failed:
            raise UnrepresentativeInput(f'{failed} sampled element(s) failed local factorization')
        if bool(cp.any(trial['reconstruction_statuses'])):
            return False
        _, states, unknowns = output
        matrix, rhs = trial['oracle']
        residual = matrix @ states - rhs
        scale = (cp.max(cp.sum(cp.abs(matrix), axis=2), axis=1)
                 * cp.max(cp.abs(states), axis=(1, 2)) + cp.max(cp.abs(rhs), axis=(1, 2)))
        error = cp.max(cp.abs(residual), axis=(1, 2)) / cp.maximum(scale, REAL_DTYPE(1.e-30))
        expected = states[:, :, -1] + cp.einsum('Kij,Kj->Ki', states[:, :, :-1], trial['trace'])
        relative = cp.max(cp.abs(unknowns-expected)) / cp.maximum(cp.max(cp.abs(expected)), REAL_DTYPE(1.))
        tolerance = max(2.e-10, 1024 * np.finfo(REAL_DTYPE).eps)
        return bool(cp.all(cp.isfinite(error))) and float(cp.max(error)) <= tolerance and float(relative) <= tolerance

    def describe(name, handle):
        """Persist each candidate's exact launch, layout and compiled resource usage."""
        return dict(configuration=asdict(handle['config']), kernels=handle['attributes'])

    name, handle, record = select_variant(cupy=cp, identity=identity, candidates=list(configs),
        prepare=prepare, before=before, run=run, validate=validate, describe=describe, rtol=2.e-10)
    record['sample_elements'] = int(trial.get('count', 0))
    return handle['config'], handle['module'], handle['kernels'][0], record
