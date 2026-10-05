"""hybridge.runtime.logging."""

from __future__ import annotations

import time
from contextlib import contextmanager

from hybridge.runtime.optional import require_cupy



def logv(config, level: int, message: str) -> None:
    """Print ``message`` when ``config.verbosity`` is at least ``level``.

    This small helper is intended for scripts and examples that expose an
    argparse-style ``verbosity`` attribute but do not need a full logging setup.
    """
    if int(getattr(config, "verbosity", 1)) >= int(level):
        print(message, flush=True)


def format_elapsed_percent(seconds: float, total: float, *, precision: int = 1) -> str:
    """Format elapsed seconds followed by its percentage of ``total``."""
    percent = 0.0 if float(total) <= 0.0 else 100.0 * float(seconds) / float(total)
    return f"{float(seconds):.{int(precision)}f} ({percent:.1f}%)"


def timed_call(label: str, verbosity: bool | int, function):
    """Call ``function``, optionally printing a concise elapsed-time line."""
    level = 1 if isinstance(verbosity, bool) and verbosity else int(verbosity or 0)
    if level:
        print(f"{label} ... ", end="", flush=True)
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    if level:
        print(f"done in {elapsed:.5f}s", flush=True)
    return result, elapsed


@contextmanager
def timed_section(config, level: int, label: str, *, timings=None, synchronize=None, **fields):
    """Emit ``LABEL_START`` and ``LABEL_DONE time=...`` messages around a block.

    Parameters in ``fields`` are printed on the ``START`` line.  The messages
    are suppressed unless ``config.verbosity >= level``.
    An optional timings mapping accumulates seconds by label across repeated
    sections. A synchronize callback completes queued work before stopping
    the timer. Drain earlier work separately before the first GPU section;
    this helper deliberately does not hide the cost of a pre-section wait.
    """
    verbose = int(getattr(config, "verbosity", 1)) >= int(level)
    if verbose:
        extras = " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"{label}_START{(' ' + extras) if extras else ''}", flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        if synchronize is not None:
            synchronize()
        elapsed = time.perf_counter() - start
        if timings is not None:
            timings[label] = timings.get(label, 0.0) + elapsed
        if verbose:
            print(f"{label}_DONE time={elapsed:.3f}", flush=True)


def sync_elapsed(start: float) -> float:
    """Synchronize the active CUDA stream and return elapsed wall time."""
    cp = require_cupy()
    cp.cuda.get_current_stream().synchronize()
    return time.perf_counter() - start


def _format_seconds(seconds: float) -> str:
    """Format elapsed wall time for concise solver logging."""
    if seconds >= 100.0:
        return f"{seconds:.1f}s"
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds:.4f}s"


def _verbosity_level(verbose: bool | int) -> int:
    """Normalize bool/int verbosity flags to an integer level."""
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    return max(0, int(verbose))


def _detailed_logging(verbose: bool | int) -> bool:
    """Return whether verbose backend micro-timings should be printed."""
    level = _verbosity_level(verbose)
    return level == 2 or level >= 4


def _timed_call(label: str, verbosity: bool | int, function, *, level: int = 1, multiline: bool = False):
    """Run ``function`` and optionally print one-line timing output."""
    should_print = (_verbosity_level(verbosity) >= level) if level <= 1 else _detailed_logging(verbosity)
    if should_print:
        indent = "  " * (level - 1)
        label = f"{indent}{label}"
        if multiline:
            print(f"{label} ...", flush=True)
        else:
            print(f"{label} ... ", end="", flush=True)
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    if should_print:
        if multiline:
            print(f"{indent}done in {_format_seconds(elapsed)}", flush=True)
        else:
            print(f"done in {_format_seconds(elapsed)}", flush=True)
    return result, elapsed
