"""Scoped, verified MKL thread settings for the existing PyPardiso backend."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os


@contextmanager
def pardiso_thread_limit(threads=16):
    """Set and verify the loaded backend's thread limit, restoring it on exit.

    ``threads='all'`` uses every affinity-visible CPU. Small hosts use all
    available CPUs when fewer than the requested number are available. The
    existing solver lock protects this process-wide MKL setting. This reports
    the actual backend limit; CPU activity must be measured around the work.
    """
    from .system import _import_pypardiso, _PYPARDISO_LOCK

    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    requested = available if threads == "all" else min(int(threads), available)
    if requested <= 0:
        raise ValueError("Pardiso thread count must be positive")
    with _PYPARDISO_LOCK:
        library = _import_pypardiso().ps.libmkl
        getter, setter = library.MKL_Get_Max_Threads, library.MKL_Set_Num_Threads
        get_dynamic, set_dynamic = library.MKL_Get_Dynamic, library.MKL_Set_Dynamic
        for function in (getter, get_dynamic):
            function.argtypes, function.restype = [], ctypes.c_int
        for function in (setter, set_dynamic):
            function.argtypes, function.restype = [ctypes.c_int], None
        previous, dynamic = int(getter()), int(get_dynamic())
        try:
            set_dynamic(0)
            setter(requested)
            actual = int(getter())
            if actual != requested:
                raise RuntimeError(f"MKL thread limit {actual} != requested {requested}")
            yield actual
        finally:
            setter(previous)
            set_dynamic(dynamic)
