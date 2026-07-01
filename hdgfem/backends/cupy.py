"""CuPy backend placeholder.

The project has older GPU experiments outside the canonical :mod:`hdgfem`
package.  A supported CuPy backend should live here once those routines are
ported to the package mesh/space data model and covered by tests.
"""

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


__all__ = ["require_cupy"]
