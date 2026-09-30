"""hdgfem.runtime.optional."""

from __future__ import annotations

import numpy as np
from typing import Any

try:  # pragma: no cover - depends on optional runtime dependency.
    import cupyx.scipy.sparse as cupyx_sparse
except ImportError as error:  # pragma: no cover
    cupyx_sparse = None
    _CUPYX_SPARSE_IMPORT_ERROR = error
else:  # pragma: no cover
    _CUPYX_SPARSE_IMPORT_ERROR = None

try:  # pragma: no cover - depends on optional runtime dependency.
    import cupyx.scipy.sparse.linalg as cupyx_sparse_linalg
except ImportError as error:  # pragma: no cover
    cupyx_sparse_linalg = None
    _CUPYX_SPARSE_LINALG_IMPORT_ERROR = error
else:  # pragma: no cover
    _CUPYX_SPARSE_LINALG_IMPORT_ERROR = None

try:  # pragma: no cover - depends on optional runtime dependency.
    import cupy as cp
except ImportError as error:  # pragma: no cover
    cp = None
    _CUPY_IMPORT_ERROR = error
else:  # pragma: no cover
    _CUPY_IMPORT_ERROR = None

try:  # pragma: no cover - depends on optional runtime dependency.
    import pyamgx
except ImportError as error:  # pragma: no cover
    pyamgx = None
    _PYAMGX_IMPORT_ERROR = error
else:  # pragma: no cover
    _PYAMGX_IMPORT_ERROR = None

try:  # pragma: no cover - availability depends on the runtime environment.
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None



def require_cupy():
    """Return the CuPy module or raise a clear dependency error."""
    if cp is None:
        raise RuntimeError("CuPy is not installed in this environment") from _CUPY_IMPORT_ERROR
    return cp


def require_cupy_device():
    """Return CuPy after verifying that a CUDA device is available."""
    cupy = require_cupy()
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as error:  # pragma: no cover - CUDA-runtime dependent.
        raise RuntimeError(
            "CuPy is installed, but the CUDA runtime or driver is unavailable"
        ) from error
    if device_count < 1:  # pragma: no cover - hardware dependent.
        raise RuntimeError("CuPy is installed, but no CUDA device is available")
    return cupy


def device_arrays_overlap(first: Any, second: Any) -> bool:
    """Return whether two contiguous device arrays overlap in memory."""
    if int(first.device.id) != int(second.device.id):
        return False
    first_bytes = int(first.nbytes)
    second_bytes = int(second.nbytes)
    if first_bytes == 0 or second_bytes == 0:
        return False
    first_begin = int(first.data.ptr)
    second_begin = int(second.data.ptr)
    return (
        first_begin < second_begin + second_bytes
        and second_begin < first_begin + first_bytes
    )


def require_cupyx_sparse():
    """Return ``cupyx.scipy.sparse`` or raise a clear dependency error."""
    if cupyx_sparse is None:
        raise RuntimeError("cupyx.scipy.sparse is not available in this environment") from _CUPYX_SPARSE_IMPORT_ERROR
    return cupyx_sparse


def require_cupyx_sparse_linalg():
    """Return ``cupyx.scipy.sparse.linalg`` or raise a clear dependency error."""
    if cupyx_sparse_linalg is None:
        raise RuntimeError(
            "cupyx.scipy.sparse.linalg is not available in this environment"
        ) from _CUPYX_SPARSE_LINALG_IMPORT_ERROR
    return cupyx_sparse_linalg


def require_pyamgx():
    """Return the PyAMGX module or raise a clear dependency error."""
    if pyamgx is None:
        raise RuntimeError("PyAMGX solve requested, but pyamgx is not importable") from _PYAMGX_IMPORT_ERROR
    return pyamgx


def array_module(*values):
    """Return CuPy if any value is a device array, else NumPy (CuPy is not imported for host data)."""
    if any(hasattr(value, "__cuda_array_interface__") for value in values):
        return require_cupy()
    return np


def asnumpy(array) -> np.ndarray:
    """Return ``array`` as a NumPy array without importing CuPy at call sites."""
    cupy = require_cupy()
    return cupy.asnumpy(array)


NUMBA_AVAILABLE = nb is not None


prange = nb.prange if nb is not None else range


def njit(*args, **kwargs):
    """Return ``numba.njit`` when available, otherwise a no-op decorator."""
    if nb is None:
        def decorator(function):
            """Return the decorated function unchanged when Numba is unavailable."""
            return function

        return decorator
    return nb.njit(*args, **kwargs)
