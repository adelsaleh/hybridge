"""Optional CuPy backend and CUDA runtime validation helpers."""

from __future__ import annotations

from typing import Any


try:  # pragma: no cover - depends on optional runtime dependency.
    import cupy as cp
except ImportError as error:  # pragma: no cover
    cp = None
    _IMPORT_ERROR = error
else:  # pragma: no cover
    _IMPORT_ERROR = None


def require_cupy():
    """Return the CuPy module or raise a clear dependency error."""
    if cp is None:
        raise RuntimeError("CuPy is not installed in this environment") from _IMPORT_ERROR
    return cp


def require_cupy_device():
    """Return CuPy after verifying that at least one CUDA device is usable."""

    module = require_cupy()
    try:
        device_count = int(module.cuda.runtime.getDeviceCount())
    except Exception as error:  # pragma: no cover - CUDA-runtime dependent.
        raise RuntimeError(
            "CuPy is installed, but the CUDA runtime or driver is unavailable"
        ) from error

    if device_count < 1:  # pragma: no cover - depends on runtime hardware.
        raise RuntimeError("CuPy is installed, but no CUDA device is available")
    return module


def solve_batched_vectors(array_module: Any, matrices: Any, vectors: Any) -> Any:
    """Solve batched square systems with one vector right-hand side each.

    NumPy 2.0, and CuPy 14 following it, interpret every right-hand side with
    more than one dimension as a stack of matrices.  Older releases accepted
    ``(..., M)`` as a stack of vectors.  Expressing each vector as an explicit
    one-column matrix works with both conventions.
    """

    return array_module.linalg.solve(matrices, vectors[..., None])[..., 0]


__all__ = ["require_cupy", "require_cupy_device", "solve_batched_vectors"]
