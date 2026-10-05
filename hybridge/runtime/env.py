"""Environment-variable configuration with the pre-rename fallback.

HYBRIDGE reads its settings from ``HYBRIDGE_<NAME>`` variables. The package was
called ``hdgfem`` before release ``0.1.0a2``; for one release the old
``HDGFEM_<NAME>`` spelling is still accepted and emits a
:class:`DeprecationWarning`. The new name wins when both are set.
"""

from __future__ import annotations

import os
import warnings

PREFIX = "HYBRIDGE_"
LEGACY_PREFIX = "HDGFEM_"


def getenv(name: str, default: str | None = None) -> str | None:
    """Return ``HYBRIDGE_<name>``, else the deprecated ``HDGFEM_<name>``, else ``default``.

    ``name`` is given without a prefix, e.g. ``getenv("PRECISION", "float64")``.
    """
    value = os.environ.get(PREFIX + name)
    if value is not None:
        return value
    legacy = os.environ.get(LEGACY_PREFIX + name)
    if legacy is not None:
        warnings.warn(
            f"{LEGACY_PREFIX}{name} is deprecated; set {PREFIX}{name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    return default


__all__ = ["LEGACY_PREFIX", "PREFIX", "getenv"]
