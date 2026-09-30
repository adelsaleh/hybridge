"""Backend adapters and the public early-alpha capability contract.

Only the capability records and lookup helpers are public from this package.
Equation-specific CuPy/raw-CUDA modules are implementation details used by the
solver layer; their historical filenames do not define API or residency. See
``docs/backends/README.md`` for the role map and migration policy, and query
:func:`get_backend_capability` before relying on a backend combination.

Optional runtime modules remain lazy: importing :mod:`hdgfem.backends` does not
import CuPy, AMGX, PETSc, PARDISO, Gmsh, or DOLFINx.
"""

from hdgfem.backends.capabilities import (
    BACKEND_CAPABILITIES,
    BackendCapability,
    get_backend_capability,
)
from hdgfem.runtime.errors import UnsupportedBackendConfigurationError


__all__ = [
    "BACKEND_CAPABILITIES",
    "BackendCapability",
    "UnsupportedBackendConfigurationError",
    "cupy",
    "get_backend_capability",
    "numba",
    "numpy",
    "raw_cuda",
]
