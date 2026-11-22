"""Bounded, same-precision calibration for repeated solver applications.

Only implementation choices live here. Solver/preconditioner parameters and
the vectors used by a real solve never enter or change during calibration.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from hybridge.runtime.cuda_tuning import (
    CandidateUnavailable, KernelConfiguration, device_signature, fingerprint,
    kernel_attributes, legal_configuration, select_variant,
)
from hybridge.runtime.env import getenv


MAX_SAMPLE_ROWS = 65536
MAX_SAMPLE_MATRIX_BYTES = 128 * 1024**2
SAMPLING_VERSION = 1


@lru_cache(maxsize=None)
def module_source_fingerprint(path: str) -> str:
    """Invalidate selections when their adapter or compiled-kernel source changes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_group_indices(groups: int, bytes_per_group: int) -> np.ndarray:
    """Sample actual groups across a batch, bounding both rows and matrix bytes."""
    count = min(int(groups), MAX_SAMPLE_ROWS,
                max(1, MAX_SAMPLE_MATRIX_BYTES // max(1, int(bytes_per_group))))
    return np.linspace(0, int(groups) - 1, count, dtype=np.int64)


def trial_vector(cp, group_ids, block: int, dtype):
    """A deterministic nonzero vector indexed by original global coordinates."""
    ids = group_ids.astype(cp.int64, copy=False)[:, None] * int(block)
    ids = ids + cp.arange(int(block), dtype=cp.int64)[None, :]
    # Small bounded integer arithmetic gives the same values in each precision.
    return cp.ascontiguousarray(((ids % 97) - 48).astype(dtype) / 49
                                + ((ids % 13) - 6).astype(dtype) / 17).ravel()


def screen_kernel(cp, kernel, threads: int, device: dict) -> dict:
    """Compile and reject impossible launches before any trial reaches CUDA."""
    try:
        attributes = kernel_attributes(cp, kernel)
    except Exception as exc:
        raise CandidateUnavailable(f"compilation: {exc}") from exc
    if not legal_configuration(KernelConfiguration(threads=threads), 0, device, attributes):
        raise CandidateUnavailable("compiled kernel resources")
    return attributes


def sample_bsr(cp, indptr, indices, data, diagonal_inverse=None):
    """Keep sampled BSR rows intact and compress their gathered column IDs.

    Sampled row IDs lead the compressed vector, so a smoother's own-row
    correction and its arbitrary neighbours still refer to the same values.
    No global coefficient array or Krylov vector is modified or duplicated.
    """
    block = int(data.shape[1])
    row_nnz = indptr[1:] - indptr[:-1]
    largest_row = max(1, int(cp.max(row_nnz).item()))
    chosen = cp.asarray(sample_group_indices(int(row_nnz.size),
                                             largest_row * block**2 * data.dtype.itemsize))
    counts = row_nnz[chosen]
    pointers = cp.concatenate((cp.zeros(1, dtype=cp.int32), cp.cumsum(counts, dtype=cp.int32)))
    offsets = (cp.arange(int(pointers[-1].item()), dtype=cp.int64)
               + cp.repeat(indptr[chosen] - pointers[:-1], counts))
    columns = indices[offsets]
    unique = cp.unique(cp.concatenate((chosen, columns)))
    chosen_positions = cp.searchsorted(unique, chosen)
    remaining = cp.ones(int(unique.size), dtype=cp.bool_)
    remaining[chosen_positions] = False
    vector_ids = cp.concatenate((chosen, unique[remaining]))
    remap = cp.empty(int(unique.size), dtype=cp.int32)
    remap[cp.searchsorted(unique, vector_ids)] = cp.arange(int(unique.size), dtype=cp.int32)
    sample = dict(
        indptr=cp.ascontiguousarray(pointers, dtype=cp.int32),
        indices=cp.ascontiguousarray(remap[cp.searchsorted(unique, columns)], dtype=cp.int32),
        data=cp.ascontiguousarray(data[offsets]), rows=int(chosen.size), block=block,
        x=trial_vector(cp, vector_ids, block, data.dtype),
        rhs=trial_vector(cp, chosen + 101, block, data.dtype),
        out=cp.empty(int(chosen.size) * block, dtype=data.dtype),
    )
    if diagonal_inverse is not None:
        sample["inverse"] = cp.ascontiguousarray(diagonal_inverse[chosen])
        sample["zero_out"] = cp.empty_like(sample["out"])
    return sample


def select_gather_gemv(cp, matrices, index, *, operation: str, block: int,
                       baseline: str, baseline_source: str, compile_baseline,
                       run_baseline, identity: bool = False, skip_absent: bool = True,
                       reuse_regime: str = "iterative-fixed-matrix"):
    """Compare legacy, interleaved and grouped applications on private samples.

    The callbacks launch the existing legacy kernel, so the reference retains
    its original arithmetic and performance. Layout conversion is setup work;
    measured trials are repeated applications to an already prepared layout.
    A cache hit compiles/screens only its winner and allocates no trial arrays.
    """
    from hybridge.linalg.gpu.face_dense import (
        GroupedGatherGemv, InterleavedGatherGemv, face_dense_threads_per_block,
        _GROUPED_GEMV_SOURCE, _INTERLEAVED_GEMV_SOURCE,
    )
    from hybridge.linalg.gpu import face_dense

    groups, rows, columns = map(int, matrices.shape)
    slots = columns // int(block)
    threads = face_dense_threads_per_block()
    variants = [(baseline, threads), ("interleaved", threads), ("coalesced", threads)]
    if not getenv("FACE_DENSE_THREADS"):
        variants += [(baseline, 128), ("interleaved", 128), ("coalesced", 128),
                     ("interleaved", 512)]
    variants = dict((f"{kind}:{size}", (kind, size)) for kind, size in variants)
    device = device_signature(cp)
    attributes = {}
    sample = {}

    def prepare(name):
        """Compile and screen one implementation without allocating sample data."""
        kind, size = variants[name]
        if kind == baseline:
            kernel = compile_baseline()
            raw = kernel
        elif kind == "interleaved":
            kernel = InterleavedGatherGemv(cp, matrices.dtype, rows=rows, slots=slots,
                                           block=block, identity=identity,
                                           skip_absent=skip_absent)
            raw = kernel.kernel
        else:
            kernel = GroupedGatherGemv(cp, matrices.dtype, rows=rows, slots=slots,
                                       block=block, identity=identity)
            raw = kernel.kernel
        attributes[name] = screen_kernel(cp, raw, size, device)
        if kind == "interleaved":
            free, _ = cp.cuda.runtime.memGetInfo()
            if int(matrices.nbytes) > int(free) * .8:
                raise CandidateUnavailable("insufficient space for persistent interleaved layout")
        return kernel

    def inputs():
        """Build private actual-coefficient samples only on calibration misses."""
        if sample:
            return sample
        chosen = cp.asarray(sample_group_indices(groups, rows * columns * matrices.dtype.itemsize))
        sample["matrices"] = cp.ascontiguousarray(matrices[chosen])
        sample["groups"] = int(chosen.size)
        if identity:
            sample["index"] = None
            ids = chosen
        else:
            neighbors = index[chosen]
            ids, compressed = cp.unique(neighbors, return_inverse=True)
            sample["index"] = cp.ascontiguousarray(
                cp.where(neighbors >= 0, compressed.reshape(neighbors.shape), -1), dtype=cp.int32)
        sample["x"] = trial_vector(cp, ids, block, matrices.dtype)
        sample["out"] = cp.empty((int(chosen.size), rows), dtype=matrices.dtype)
        return sample

    def trial_prepare(name, kernel):
        """Prepare sample layouts outside the measured repeated application."""
        data = inputs()
        if variants[name][0] == "interleaved" and "interleaved" not in data:
            data["interleaved"] = kernel.interleave(cp, data["matrices"])

    def run(name, kernel):
        """Apply one candidate to the same isolated vectors and coefficients."""
        data = sample
        kind, size = variants[name]
        if kind == baseline:
            run_baseline(kernel, data["matrices"], data["index"], data["x"],
                         data["out"], data["groups"], size)
        elif kind == "interleaved":
            kernel(data["interleaved"], data["index"], data["x"], data["out"],
                   num_groups=data["groups"], threads=size)
        else:
            kernel(data["matrices"], data["index"], data["x"], data["out"],
                   num_groups=data["groups"], threads=size)
        return data["out"]

    name, _, report = select_variant(
        cupy=cp, identity=dict(operation=operation, shape=list(matrices.shape),
                              dtype=str(matrices.dtype), layout="group-row-column",
                              block=int(block), identity=identity, skip_absent=skip_absent,
                              reuse_regime=reuse_regime, sampling=SAMPLING_VERSION,
                              sample_rows=MAX_SAMPLE_ROWS, sample_matrix_bytes=MAX_SAMPLE_MATRIX_BYTES,
                              group_lanes=getenv("FACE_DENSE_GROUP_LANES"),
                              source=fingerprint([baseline_source, _GROUPED_GEMV_SOURCE,
                                                  _INTERLEAVED_GEMV_SOURCE,
                                                  module_source_fingerprint(__file__),
                                                  module_source_fingerprint(face_dense.__file__)])),
        candidates=list(variants), prepare=prepare, trial_prepare=trial_prepare, run=run,
        describe=lambda name, _: dict(attributes=attributes[name],
                                      threads=variants[name][1],
                                      layout_setup_in_timing=False),
    )
    kind, size = variants[name]
    return kind, size, report
