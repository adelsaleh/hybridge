"""One assembled ADR trace system kept on the GPU, and the solver inputs built from it there.

The vendored device route (``assemble_adr_device(device_outputs=True)``) leaves the eliminated
face-dense system, the element blocks and the mesh geometry on the device. Its chunk hook
derives the verification data ``w = L_u s`` and ``M = L_u B`` (``L_u`` = the u rows of the local
inverse, ``B`` = the element boundary matrices, ``s`` = the source moments) from each chunk's
local inverse before that inverse is freed; nothing of the local problem is copied back.

From the device system this module builds, without host round trips:

- the face-dense operator (``matrix_batches[f, i, s*b+j]`` permutation of the blocks);
- block-Jacobi (diagonal blocks inverted with cuBLAS);
- element additive Schwarz (the host ``build_face_additive_schwarz_local_matrices`` and
  ``build_face_additive_schwarz_incidence_slots``, vectorized on the device);
- the hp-MG hierarchy input (``BernsteinHpSymmetricPartPreconditioner.from_face_dense``) and
  the AMGX BSR (``device_measure.amgx_bsr_on_device``) take the device blocks directly.

``export_for_pardiso`` writes the trace system to the cache (``system_*.npy``) in one direction,
device to host to disk, for the host PyPardiso reference; the GPU side never reads it back.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


@dataclass
class DeviceFaceDenseSystem:
    """The eliminated trace system with CuPy arrays, and the host ``FaceDenseSystem`` properties."""

    blocks: object            # (NF, S, b, b)
    neighbors: object         # (NF, S) int64, -1 = empty slot
    rhs: object               # (NF, b)
    global_faces: object      # (NF,) int64: free face -> mesh edge
    global_to_local: object   # (num_edges,) int64: mesh edge -> free face or -1
    boundary_trace: object    # (num_edges, b)
    mode: str = 'eliminate'

    @property
    def num_rows(self):
        return int(self.blocks.shape[0])

    @property
    def num_slots(self):
        return int(self.blocks.shape[1])

    @property
    def block_size(self):
        return int(self.blocks.shape[2])

    @property
    def num_dofs(self):
        return self.num_rows*self.block_size


@dataclass
class DeviceAssembly:
    system: DeviceFaceDenseSystem
    element_blocks: object    # (NE, 3, 3, F, F)
    loc2glob: object          # (NE, 3) int64
    orientations: object      # (NE, 3) bool
    aff_mats: object
    aff_vecs: object
    aff_jacs: object
    w: object                 # (NE, P)
    M: object                 # (NE, P, 3F)
    timings: dict
    info: dict


@dataclass
class SolveContext:
    """What a solver candidate needs from a device assembly (``element_blocks`` may be released)."""

    cp: object
    system: DeviceFaceDenseSystem
    element_blocks: object
    loc2glob: object
    verifier: object


def assemble_on_device(cp, space, case, *, stabilization=1., element_chunk=None, log=print, xp=None):
    """Assemble with the vendored device route and keep everything on the GPU.

    ``case`` is the vendored case dict (source, beta, reaction, boundary_condition, diffusion).
    ``xp`` is the route's array module (``cp``; NumPy in the CPU tests).
    """
    from hdgfem.backends.cupy_adr_assembly import assemble_adr_device
    P, F = space.el_dof, space.quad_data.edg_dof
    ne = space.mesh.num_tri
    w = cp.empty((ne, P), dtype=cp.float64)
    M = cp.empty((ne, P, 3*F), dtype=cp.float64)

    def keep_verification_data(s, e, inverse, boundary, source_rhs):
        lu = inverse[:, :P, :]
        w[s:e] = cp.matmul(lu, source_rhs[..., None])[..., 0]
        M[s:e] = cp.matmul(lu, boundary)

    timings = {}
    cp.cuda.Device().synchronize()
    start = perf_counter()
    out = assemble_adr_device(case['source'], case['beta'], case['reaction'], case['boundary_condition'], space,
                              diffusion=case['diffusion'], stabilization=stabilization, element_chunk=element_chunk,
                              timings=timings, xp=cp if xp is None else xp, device_outputs=True,
                              chunk_hook=keep_verification_data)
    cp.cuda.Device().synchronize()
    timings['assembly_total'] = perf_counter()-start
    d = out['device']
    system = DeviceFaceDenseSystem(blocks=d['blocks'], neighbors=d['neighbors'], rhs=d['rhs'],
                                   global_faces=d['global_faces'], global_to_local=d['global_to_local'],
                                   boundary_trace=d['boundary_trace'])
    info = out['info']
    if info.get('host_callable_fallbacks'):
        log(f"[assembly] WARNING host coefficient sampling: {', '.join(info['host_callable_fallbacks'])}")
    log(f"[assembly] {ne:,} elements, {system.num_dofs:,} trace DOFs in {timings['assembly_total']:.3f} s "
        f"({info['chunks']} chunks of {info['element_chunk']:,}); verification data w {w.nbytes/2**20:.0f} MiB, "
        f"M {M.nbytes/2**30:.2f} GiB kept on the device")
    return DeviceAssembly(system=system, element_blocks=d['element_blocks'], loc2glob=d['loc2glob'],
                          orientations=d['orientations'], aff_mats=d['aff_mats'], aff_vecs=d['aff_vecs'],
                          aff_jacs=d['aff_jacs'], w=w, M=M, timings=timings, info=info)


def retained_bytes(triangles, p):
    """Device bytes the system worker keeps across candidates (element blocks freed after ASM)."""
    P, F = (p+1)*(p+2)//2, p+1
    faces = int(1.5*triangles+4*triangles**0.5)
    blocks = faces*5*F*F*8
    element_blocks = triangles*9*F*F*8
    w_and_M = triangles*P*(1+3*F)*8
    return dict(system_blocks=blocks, element_blocks=element_blocks, verification=w_and_M,
                during_asm=blocks+element_blocks+w_and_M, after_asm=blocks+w_and_M)


def _write_npy_chunks(cp, path, array, chunk_rows, digest=None):
    """Write a device array as ``.npy`` through pageable host buffers, hashing the raw bytes.

    Returns the seconds spent in device-to-host copies, in ``write`` calls and in ``close``.
    The scratch file system stalls some writes or closes for a fixed ~307 s (tier 1 of v3,
    whatever the size, memory map or not), which is why the export runs off the GPU's path.
    """
    from numpy.lib.format import dtype_to_descr, write_array_header_1_0
    dtype = np.dtype(array.dtype.name)
    copy = write = 0.
    stream = open(path, 'wb')
    try:
        write_array_header_1_0(stream, dict(descr=dtype_to_descr(dtype), fortran_order=False, shape=tuple(int(n) for n in array.shape)))
        for s in range(0, int(array.shape[0]), chunk_rows):
            t = perf_counter()
            chunk = cp.ascontiguousarray(array[s:s+chunk_rows])
            host = chunk.get() if hasattr(chunk, 'get') else np.ascontiguousarray(chunk)
            raw = memoryview(host).cast('B')
            copy += perf_counter()-t
            t = perf_counter()
            stream.write(raw)
            write += perf_counter()-t
            if digest is not None:
                digest.update(raw)
    finally:
        t = perf_counter()
        stream.close()
        close = perf_counter()-t
    return dict(copy_seconds=copy, write_seconds=write, close_seconds=close)


def export_for_pardiso(cp, cache, system, *, chunk_rows=1 << 16, log=print):
    """Write ``system_*.npy`` (device to host to disk, streamed) for the host PyPardiso reference.

    Returns ``(cache_files, operator_sha256, info)``; the hash is ``operator_hash``'s (SHA-256 of
    the raw bytes of ``system_blocks`` then ``system_rhs``) and ``info`` holds the per-file
    timings. Safe to run in a thread beside device work: the device arrays are only read.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    arrays = dict(blocks=system.blocks, neighbors=system.neighbors, rhs=system.rhs, global_faces=system.global_faces,
                  global_to_local=system.global_to_local, boundary_trace=system.boundary_trace)
    digest = hashlib.sha256()
    timings = {}
    for name in ('blocks', 'rhs', 'neighbors', 'global_faces', 'global_to_local', 'boundary_trace'):
        t = perf_counter()
        timings[name] = _write_npy_chunks(cp, cache/f'system_{name}.npy', arrays[name], chunk_rows,
                                          digest if name in ('blocks', 'rhs') else None)
        timings[name]['seconds'] = perf_counter()-t
        if timings[name]['seconds'] > 5:
            log(f"[export] WARNING system_{name}.npy took {timings[name]['seconds']:.1f} s "
                f"(copy {timings[name]['copy_seconds']:.2f}, write {timings[name]['write_seconds']:.2f}, "
                f"close {timings[name]['close_seconds']:.2f})")
    operator_sha256 = digest.hexdigest()
    files = {p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(cache.glob('*.npy'))}
    bytes_written = sum(f['bytes'] for f in files.values())
    elapsed = perf_counter()-start
    (cache/'metadata.json').write_text(json.dumps(dict(
        status='passed', kind='assembly', route='device', export='system_* only (host PyPardiso reference); '
        'local inverses, boundary matrices and element blocks stay on the device',
        operator_sha256=operator_sha256, arrays={k: v['bytes'] for k, v in files.items()},
        export_seconds=elapsed, file_timings=timings), indent=2)+'\n')
    log(f'[export] {bytes_written/2**30:.2f} GiB of system_* written for PyPardiso in {elapsed:.2f} s')
    return files, operator_sha256, dict(seconds=elapsed, bytes=bytes_written, file_timings=timings)


def operator_batches_on_device(cp, system):
    """The host ``prepare_face_dense_batch_layout`` (``matrix_batches[f, i, s*b+j]``, int32 neighbors) on the device."""
    blocks = system.blocks
    nf, slots, b, _ = blocks.shape
    if not bool(cp.isfinite(blocks).all()):
        raise ValueError('blocks contain non-finite values')
    neighbors = system.neighbors
    if bool((neighbors < -1).any()) or bool((neighbors >= nf).any()):
        raise ValueError('neighbors may contain only valid row ids or -1')
    matrix_batches = cp.ascontiguousarray(blocks.transpose(0, 2, 1, 3).reshape(nf, b, slots*b))
    return matrix_batches, cp.ascontiguousarray(neighbors.astype(cp.int32))


def operator_from_device(cp, system, *, implementation, device_id=None):
    """``CuPyFaceDenseOperator`` from device blocks, with the layout permutation on the device."""
    from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
    matrix_batches, neighbors = operator_batches_on_device(cp, system)
    device = int(cp.cuda.Device().id) if device_id is None else int(device_id)
    return CuPyFaceDenseOperator(matrix_batches=matrix_batches, neighbors=neighbors,
                                 implementation=implementation, device_id=device)


def block_jacobi_from_device(cp, system, *, local_solver='cublas_inverse', application='raw', device_id=None,
                             inverse_residual_tolerance=None):
    from hybridge.linalg.gpu.cublas_batched import invert_batched_cublas
    from hybridge.linalg.gpu.preconditioners import (
        CuPyFaceBlockJacobiPreconditioner, _check_gpu_inverse_tolerance, _gpu_batched_inverse,
    )
    if local_solver not in ('cublas_inverse', 'gpu_inverse'):
        raise ValueError(f'device block-Jacobi supports cublas_inverse or gpu_inverse, not {local_solver}')
    device = int(cp.cuda.Device().id) if device_id is None else int(device_id)
    local_blocks = cp.ascontiguousarray(system.blocks[:, 0])
    factorization_info = inversion_info = None
    if local_solver == 'cublas_inverse':
        result = invert_batched_cublas(local_blocks, label='block-Jacobi diagonal blocks')
        inverse_blocks, residuals = result.inverse_matrices, result.inverse_residuals
        factorization_info, inversion_info = result.factorization_info, result.inversion_info
    else:
        inverse_blocks, residuals = _gpu_batched_inverse(cp, local_blocks, label='block-Jacobi diagonal blocks')
    _check_gpu_inverse_tolerance(cp, residuals, inverse_residual_tolerance, label='block-Jacobi')
    return CuPyFaceBlockJacobiPreconditioner(
        inverse_blocks=inverse_blocks, local_blocks=local_blocks, device_id=device, local_solver=local_solver,
        inverse_residuals=residuals, factorization_info=factorization_info, inversion_info=inversion_info,
        application=application)


def asm_local_matrices_on_device(cp, system, element_blocks, loc2glob):
    """The host ``build_face_additive_schwarz_local_matrices`` for eliminated systems, vectorized."""
    ne, nlf, _, F, _ = element_blocks.shape
    faces = system.global_to_local[loc2glob]                       # (NE, 3) free face or -1
    active = faces >= 0
    local = element_blocks.copy()
    # Rows and columns of eliminated faces are zero, then their diagonal is the identity.
    keep = active[:, :, None] & active[:, None, :]                  # (NE, 3, 3)
    local *= keep[..., None, None]
    safe = cp.where(active, faces, 0)
    diagonal = system.blocks[safe, 0]                               # (NE, 3, F, F)
    identity = cp.eye(F, dtype=local.dtype)
    idx = cp.arange(nlf)
    local[:, idx, idx] = cp.where(active[..., None, None], diagonal, identity)
    local_matrices = cp.ascontiguousarray(local.transpose(0, 1, 3, 2, 4).reshape(ne, nlf*F, nlf*F))
    del local
    return local_matrices, faces


def asm_incidence_slots_on_device(cp, faces, num_system_faces):
    """The host ``build_face_additive_schwarz_incidence_slots`` (int32 ``(NF, 2)``) on the device."""
    ne, nlf = faces.shape
    flat = faces.reshape(-1)
    slot = cp.arange(flat.size, dtype=cp.int64)
    active = flat >= 0
    system_face = flat[active]
    slot = slot[active]
    order = cp.argsort(system_face, kind='stable')
    system_face, slot = system_face[order], slot[order]
    counts = cp.bincount(system_face, minlength=num_system_faces)
    if bool((counts > 2).any()):
        raise ValueError('fused ASM requires at most two incident elements per system face')
    if bool((counts == 0).any()):
        raise ValueError('every system face must occur in element_system_faces')
    offsets = cp.concatenate((cp.zeros(1, dtype=cp.int64), cp.cumsum(counts)[:-1]))
    slots = cp.full((num_system_faces, 2), -1, dtype=cp.int64)
    slots[:, 0] = slot[offsets]
    second = counts == 2
    slots[second, 1] = slot[offsets[second]+1]
    return cp.ascontiguousarray(slots.astype(cp.int32))


def asm_from_device(cp, system, element_blocks, loc2glob, *, local_solver='cublas_inverse', application='fused',
                    device_id=None, inverse_residual_tolerance=None):
    from hybridge.linalg.gpu.cublas_batched import invert_batched_cublas
    from hybridge.linalg.gpu.preconditioners import (
        _FUSED_APPLICATIONS, CuPyFaceAdditiveSchwarzPreconditioner, _check_gpu_inverse_tolerance, _gpu_batched_inverse,
    )
    if local_solver not in ('cublas_inverse', 'gpu_inverse'):
        raise ValueError(f'device ASM supports cublas_inverse or gpu_inverse, not {local_solver}')
    device = int(cp.cuda.Device().id) if device_id is None else int(device_id)
    local_matrices, faces = asm_local_matrices_on_device(cp, system, element_blocks, loc2glob)
    if system.num_rows > np.iinfo(np.int32).max:
        raise OverflowError('the CUDA ASM kernels use int32 face indices')
    faces32 = cp.ascontiguousarray(faces.astype(cp.int32))
    slots = asm_incidence_slots_on_device(cp, faces, system.num_rows) if application in _FUSED_APPLICATIONS else None
    factorization_info = inversion_info = None
    if local_solver == 'cublas_inverse':
        result = invert_batched_cublas(local_matrices, label='additive-Schwarz local matrices')
        inverse_matrices, residuals = result.inverse_matrices, result.inverse_residuals
        factorization_info, inversion_info = result.factorization_info, result.inversion_info
    else:
        inverse_matrices, residuals = _gpu_batched_inverse(cp, local_matrices, label='additive-Schwarz local matrices')
    _check_gpu_inverse_tolerance(cp, residuals, inverse_residual_tolerance, label='additive-Schwarz')
    return CuPyFaceAdditiveSchwarzPreconditioner(
        inverse_matrices=inverse_matrices, local_matrices=local_matrices, element_system_faces=faces32,
        block_size=int(element_blocks.shape[-1]), num_system_faces=system.num_rows, device_id=device,
        local_solver=local_solver, inverse_residuals=residuals, factorization_info=factorization_info,
        inversion_info=inversion_info, application=application, face_element_slots=slots)
