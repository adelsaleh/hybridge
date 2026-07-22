"""CuPy backend placeholder.

The project has older GPU experiments outside the canonical :mod:`hdgfem`
package.  A supported CuPy backend should live here once those routines are
ported to the package mesh/space data model and covered by tests.
"""
from __future__ import annotations

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

__all__ = ["require_cupy", "require_cupy_device"]
