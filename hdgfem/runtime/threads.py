"""Thread-parallel NumPy over element chunks.

NumPy releases the GIL inside its ufunc, einsum, matmul and copy loops, so
element-local NumPy work split into contiguous element chunks runs on several
cores from a plain thread pool. OpenBLAS threads do not help here: the hot
loops are elementwise ufuncs and stacks of tiny matrix products (see
``OPENBLAS_NUM_THREADS=1`` in the n-Gamma runner).

The pool is shared and sized by :func:`set_host_threads` (default: the CPUs
visible to the process, or ``HDGFEM_HOST_THREADS``). Work below
``min_chunk`` elements per thread runs serially in the calling thread, so
small meshes pay no pool overhead. Chunk results are independent per element,
so they match a serial evaluation to round-off.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import threading
from typing import Callable

import numpy as np

MIN_CHUNK = 256
POINT_CHUNK = 32768  # points per elementwise task (measured best for a ~300-ufunc MMS source, 24 threads)
_PREFIX = "hdgfem-host"
_LOCK = threading.Lock()
_LOCAL = threading.local()
_POOL: ThreadPoolExecutor | None = None
_POOL_SIZE = 0
_THREADS: int | None = None


def _available_cpus() -> int:
    """CPUs available to this process, from the affinity mask when supported."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def host_threads() -> int:
    """Threads used by :func:`for_element_chunks` and :func:`elementwise`."""
    if _THREADS is None:
        setting = os.environ.get("HDGFEM_HOST_THREADS")
        return max(1, int(setting)) if setting else _available_cpus()
    return _THREADS


def set_host_threads(count: int | None) -> int:
    """Set the host NumPy thread count (``None`` restores the default); return the active count."""
    global _THREADS
    _THREADS = None if count is None else max(1, int(count))
    return host_threads()


def _pool(threads: int) -> ThreadPoolExecutor:
    """Return the shared thread pool, recreating it when the thread count changes."""
    global _POOL, _POOL_SIZE
    with _LOCK:
        if _POOL is None or _POOL_SIZE != threads:
            if _POOL is not None:
                _POOL.shutdown(wait=False)
            _POOL, _POOL_SIZE = ThreadPoolExecutor(threads, thread_name_prefix=_PREFIX), threads
        return _POOL


def element_chunks(count: int, *, min_chunk: int = MIN_CHUNK, max_chunk: int | None = None,
                   threads: int | None = None) -> list[tuple[int, int]]:
    """Contiguous ``(start, stop)`` ranges covering ``count`` elements.

    One range per worker, each at least ``min_chunk`` long; ``max_chunk``
    splits further (more ranges than workers) to keep temporaries small.
    """
    count = int(count)
    workers = host_threads() if threads is None else max(1, int(threads))
    chunks = max(1, min(workers, count // max(1, int(min_chunk))))
    if max_chunk is not None and chunks > 1:
        chunks = max(chunks, -(-count // max(1, int(max_chunk))))
    bounds = np.linspace(0, count, chunks + 1).astype(np.int64)
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def for_element_chunks(function: Callable[[int, int], None], count: int, *, min_chunk: int = MIN_CHUNK,
                       threads: int | None = None) -> None:
    """Call ``function(start, stop)`` on disjoint element chunks, in parallel when worthwhile.

    ``function`` must only write to its own element range. The first chunk
    exception is re-raised after every chunk has finished. Calls nested inside
    a chunk run serially.
    """
    _run_chunks(function, element_chunks(count, min_chunk=min_chunk, threads=threads), threads)


def elementwise(function: Callable[..., np.ndarray], *arrays, min_chunk: int = POINT_CHUNK // 2,
                max_chunk: int = POINT_CHUNK, threads: int | None = None) -> np.ndarray:
    """Evaluate a pointwise NumPy ``function(*arrays)`` in chunks of the leading axis.

    ``arrays`` broadcast to a common shape; ``function`` must act pointwise
    (trailing output axes, such as vector components, are allowed).
    ``min_chunk`` and ``max_chunk`` count points per task; 0-d inputs are
    evaluated directly. Every NumPy call dispatches under the GIL, so tasks
    much smaller than ``POINT_CHUNK`` serialize on dispatch while much larger
    ones pay for large temporaries.
    """
    arrays = np.broadcast_arrays(*(np.asarray(a) for a in arrays))
    if not arrays or arrays[0].ndim == 0:
        return function(*arrays)
    rows = arrays[0].shape[0]
    per_row = max(1, int(np.prod(arrays[0].shape[1:], dtype=np.int64)))
    chunks = element_chunks(rows, min_chunk=max(1, min_chunk // per_row), max_chunk=max(1, max_chunk // per_row),
                            threads=threads)
    if len(chunks) == 1:
        return function(*arrays)
    probe = np.asarray(function(*(a[:1] for a in arrays)))  # output trailing shape and dtype
    out = np.empty((rows,) + probe.shape[1:], dtype=probe.dtype)

    def evaluate(start, stop):
        """Evaluate ``function`` on rows ``start:stop`` into ``out``."""
        out[start:stop] = function(*(a[start:stop] for a in arrays))

    _run_chunks(evaluate, chunks, threads)
    return out


def _run_chunks(function, chunks, threads: int | None = None) -> None:
    """Run ``function(start, stop)`` over ``chunks`` on a pool of ``threads`` workers."""
    if not chunks:
        return
    if len(chunks) == 1 or getattr(_LOCAL, "active", False):
        # Nested use inside a chunk runs serially: the pool is already busy, and a
        # worker waiting on the pool could deadlock.
        for chunk in chunks:
            function(*chunk)
        return

    def run(start, stop):
        """Run one chunk with the worker flag set, so nested parallel helpers run serially."""
        _LOCAL.active = True
        try:
            function(start, stop)
        finally:
            _LOCAL.active = False

    pool = _pool(host_threads() if threads is None else max(1, int(threads)))
    futures = [pool.submit(run, a, b) for a, b in chunks]
    error = None
    for future in futures:
        try:
            future.result()
        except BaseException as exception:  # noqa: BLE001 - re-raised after every chunk finished
            error = error or exception
    if error is not None:
        raise error


def parallel_copy(array: np.ndarray, *, min_chunk: int = MIN_CHUNK) -> np.ndarray:
    """C-contiguous copy of ``array``, copied in leading-axis chunks."""
    array = np.asarray(array)
    out = np.empty(array.shape, dtype=array.dtype)
    if array.ndim == 0:
        out[...] = array
        return out

    def copy(start, stop):
        """Copy rows ``start:stop`` into ``out``."""
        out[start:stop] = array[start:stop]

    for_element_chunks(copy, array.shape[0], min_chunk=min_chunk)
    return out


__all__ = ["MIN_CHUNK", "POINT_CHUNK", "element_chunks", "elementwise", "for_element_chunks", "host_threads",
           "parallel_copy", "set_host_threads"]
