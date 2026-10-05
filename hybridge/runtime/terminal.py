"""Flush and forward native solver output through the active Python streams."""

from __future__ import annotations

import ctypes
import sys


def flush_native_stdio() -> None:
    """Flush pending C output before changing descriptors or output callbacks."""
    try:
        libc = ctypes.CDLL(None)
        fflush = libc.fflush
        fflush.argtypes = [ctypes.c_void_p]
        fflush.restype = ctypes.c_int
        fflush(None)
    except (AttributeError, OSError):
        pass


def flush_terminal_streams() -> None:
    """Flush Python and C output before redirecting or restoring descriptors."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    flush_native_stdio()


def write_native_solver_output(message: str) -> None:
    """Forward one native print callback immediately, preserving its newlines.

    Native libraries commonly use buffered C stdout when a tee redirects it to
    a pipe. Register this writer with their supported print callback instead:
    each progress message reaches the terminal/log while the solve is running.
    No Python background thread is needed, even when the solve holds the GIL.
    """
    try:
        sys.stdout.write(message)
        sys.stdout.flush()
    except (AttributeError, OSError, ValueError):
        # A closed/replaced output stream must not raise through a native
        # callback declared noexcept (including during interpreter shutdown).
        pass
