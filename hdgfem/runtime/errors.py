"""hdgfem.runtime.errors."""

from __future__ import annotations


class UnsupportedBackendConfigurationError(NotImplementedError):
    """A valid option combination is outside the supported backend matrix."""
