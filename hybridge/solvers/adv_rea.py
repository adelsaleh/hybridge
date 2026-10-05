"""Compatibility alias for :mod:`hybridge.solvers.advection_reaction`.

New code must use the full-name module. This shim remains for the documented
alpha compatibility period and delegates private implementation helpers too.
"""

from hybridge.solvers import advection_reaction as _implementation
from hybridge.solvers.advection_reaction import *  # noqa: F401,F403


__all__ = _implementation.__all__


def __getattr__(name: str):
    """Forward attribute lookup to the wrapped object."""
    return getattr(_implementation, name)
