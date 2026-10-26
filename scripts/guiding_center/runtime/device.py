"""Device memory housekeeping between runs of one process."""
from __future__ import annotations


def release_device_memory() -> None:
    """Drop finished runs' device objects between runs of one process.

    Garbage-collect (AMGX solvers free their device memory when collected) and return CuPy's
    cached blocks, so later (larger) runs start from a clean pool; without this, long in-process
    ladders end in AMGX setup out-of-memory.
    """
    import gc

    gc.collect()
    try:
        from hybridge.runtime.optional import require_cupy

        cupy = require_cupy()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
