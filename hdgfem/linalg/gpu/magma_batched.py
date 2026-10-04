"""Optional MAGMA batched LU solves for small dense matrices.

MAGMA is not a Python dependency. This module binds the few batched routines
it needs through ``ctypes`` and shares the layout and workspace contract of
:func:`hdgfem.linalg.gpu.cublas_batched.lu_solve_batched_cublas`, so the two
libraries can be swapped and compared on identical buffers.

The shared library is located through ``HDGFEM_MAGMA_LIBRARY`` (the path of
``libmagma.so``), then ``HDGFEM_MAGMA_ROOT`` (a MAGMA build or install tree
containing ``lib/libmagma.so``), then the dynamic-linker search path. The
binding assumes the default LP64 build (32-bit ``magma_int_t``).

MAGMA queues are created on the current CuPy device and stream with their own
cuBLAS/cuSPARSE handles, so CuPy's cuBLAS handle and its stream binding are
never modified. Queues are cached per ``(device, stream)`` for the process.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Any

from hdgfem.linalg.gpu.cublas_batched import (
    BatchedLUWorkspace,
    _copy_and_check_info,
    validate_batched_lu_operands,
)
from hdgfem.runtime.optional import require_cupy_device

# magma_trans_t values from magma_types.h.
_MAGMA_NO_TRANS = 111
_MAGMA_TRANS = 112

_LIBRARY: ctypes.CDLL | None = None
_QUEUES: dict[tuple[int, int], ctypes.c_void_p] = {}


def _library_candidates() -> list[str]:
    """Return MAGMA library paths in lookup order."""
    candidates = []
    explicit = os.environ.get("HDGFEM_MAGMA_LIBRARY")
    if explicit:
        candidates.append(explicit)
    root = os.environ.get("HDGFEM_MAGMA_ROOT")
    if root:
        candidates.append(str(Path(root) / "lib" / "libmagma.so"))
    found = ctypes.util.find_library("magma")
    if found:
        candidates.append(found)
    return candidates


def _declare(library: ctypes.CDLL) -> None:
    """Declare the argument and return types of every bound routine."""
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    library.magma_init.argtypes = []
    library.magma_init.restype = integer
    library.magma_queue_create_from_cuda_internal.argtypes = [
        integer, pointer, pointer, pointer, ctypes.POINTER(pointer),
        ctypes.c_char_p, ctypes.c_char_p, integer,
    ]
    library.magma_queue_create_from_cuda_internal.restype = None
    for prefix in ("s", "d"):
        getrf = getattr(library, f"magma_{prefix}getrf_batched")
        getrf.argtypes = [integer, integer, pointer, integer, pointer, pointer, integer, pointer]
        getrf.restype = integer
        getrs = getattr(library, f"magma_{prefix}getrs_batched")
        getrs.argtypes = [integer, integer, integer, pointer, integer, pointer, pointer, integer, integer, pointer]
        getrs.restype = integer
        gesv = getattr(library, f"magma_{prefix}gesv_batched")
        gesv.argtypes = [integer, integer, pointer, integer, pointer, pointer, integer, pointer, integer, pointer]
        gesv.restype = integer


def load_magma() -> ctypes.CDLL:
    """Load and initialize MAGMA once per process.

    Raises
    ------
    RuntimeError
        If no candidate library loads or ``magma_init`` fails.
    """
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    errors = []
    for candidate in _library_candidates():
        try:
            library = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        _declare(library)
        status = library.magma_init()
        if status != 0:
            raise RuntimeError(f"magma_init failed with status {status} ({candidate})")
        _LIBRARY = library
        return library
    detail = "; ".join(errors) if errors else "no candidate found"
    raise RuntimeError(
        "MAGMA is unavailable; set HDGFEM_MAGMA_LIBRARY or HDGFEM_MAGMA_ROOT "
        f"({detail})"
    )


def magma_available() -> bool:
    """Return whether the MAGMA library can be loaded."""
    try:
        load_magma()
    except RuntimeError:
        return False
    return True


def _queue(cp: Any, library: ctypes.CDLL) -> ctypes.c_void_p:
    """Return a cached MAGMA queue for the current CuPy device and stream."""
    device = int(cp.cuda.runtime.getDevice())
    stream = int(cp.cuda.get_current_stream().ptr)
    key = (device, stream)
    queue = _QUEUES.get(key)
    if queue is None:
        queue = ctypes.c_void_p()
        library.magma_queue_create_from_cuda_internal(
            device, ctypes.c_void_p(stream), None, None, ctypes.byref(queue),
            b"hdgfem", b"magma_batched.py", 0,
        )
        if not queue.value:
            raise RuntimeError("MAGMA queue creation failed")
        _QUEUES[key] = queue
    return queue


def lu_solve_batched_magma(
    matrices: Any,
    rhs: Any,
    *,
    trans: bool = False,
    fused: bool = False,
    workspace: BatchedLUWorkspace | None = None,
    check_info: bool = True,
    label: str = "batched matrices",
) -> BatchedLUWorkspace:
    """Solve ``A_b X_b = B_b`` in place with MAGMA batched LU.

    The layout, ``trans`` meaning, in-place results, and ``check_info``
    behavior match :func:`~hdgfem.linalg.gpu.cublas_batched.lu_solve_batched_cublas`.
    ``fused=True`` calls ``magma_<t>gesv_batched`` (factor and solve in one
    call, no transpose option) instead of ``getrf`` followed by ``getrs``.
    """
    cp = require_cupy_device()
    library = load_magma()
    batch, size, nrhs = validate_batched_lu_operands(cp, matrices, rhs)
    if fused and trans:
        raise ValueError("MAGMA gesv_batched has no transpose option; use fused=False")
    workspace = BatchedLUWorkspace() if workspace is None else workspace
    workspace.ensure(cp, matrices, rhs)
    queue = _queue(cp, library)
    prefix = "s" if matrices.dtype == cp.float32 else "d"
    a_array = ctypes.c_void_p(int(workspace.matrix_pointers.data.ptr))
    b_array = ctypes.c_void_p(int(workspace.rhs_pointers.data.ptr))
    pivots = ctypes.c_void_p(int(workspace.pivot_pointers.data.ptr))
    info = ctypes.c_void_p(int(workspace.info.data.ptr))
    if fused:
        status = getattr(library, f"magma_{prefix}gesv_batched")(
            size, nrhs, a_array, size, pivots, b_array, size, info, batch, queue)
        stages = (("gesv_batched", status),)
    else:
        status = getattr(library, f"magma_{prefix}getrf_batched")(
            size, size, a_array, size, pivots, info, batch, queue)
        solve_status = getattr(library, f"magma_{prefix}getrs_batched")(
            _MAGMA_TRANS if trans else _MAGMA_NO_TRANS, size, nrhs, a_array, size,
            pivots, b_array, size, batch, queue)
        stages = (("getrf_batched", status), ("getrs_batched", solve_status))
    for stage, value in stages:
        if value:
            raise RuntimeError(f"{label} MAGMA {stage} returned status {int(value)}")
    if check_info:
        _copy_and_check_info(cp, workspace.info, stage="MAGMA LU", label=label)
    return workspace


__all__ = [
    "load_magma",
    "lu_solve_batched_magma",
    "magma_available",
]
