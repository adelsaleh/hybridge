"""hybridge.runtime.logging."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from hybridge.runtime.optional import require_cupy

# Timed rows end their dots here, so the seconds of all phases form one column.
LOG_TIME_COLUMN = 72
_line_lock = threading.RLock()
_row_open = False
_deferred_lines: list[str] = []
_log_clock_origin: float | None = None


def start_log_clock(origin: float | None = None) -> None:
    """Show the time since ``origin`` (default: now) on later timed rows; only the first call counts.

    A jump in this column larger than the row's own seconds is work that no row covers.
    """
    global _log_clock_origin
    if _log_clock_origin is None:
        _log_clock_origin = time.perf_counter() if origin is None else float(origin)


def log_line(message: str) -> None:
    """Print whole lines from any thread, holding them back while a timed row waits for its seconds."""
    with _line_lock:
        if _row_open:
            _deferred_lines.append(message)
        else:
            print(message, flush=True)


def _row_label(label: str) -> str:
    """Return ``label`` and a space, padded with dots to column ``LOG_TIME_COLUMN``, then another space."""
    return f"{label} ".ljust(LOG_TIME_COLUMN, ".") + " "


def _row_time(elapsed: float, note: str | None = None) -> str:
    """Return ``elapsed`` in seconds, then the time since the log clock origin when it is set, then ``note`` in parentheses when given."""
    text = f"{elapsed:9.4f} s"
    if _log_clock_origin is not None:
        text += f" | t={time.perf_counter() - _log_clock_origin:8.2f} s"
    return text if note is None else f"{text} ({note})"


def log_row(label: str, elapsed: float) -> None:
    """Print one finished timed row in the common layout."""
    log_line(_row_label(label) + _row_time(elapsed))


def _begin_row(label: str, multiline: bool) -> None:
    """Show what is running now; ``multiline`` is for work that prints lines of its own."""
    global _row_open
    with _line_lock:
        if multiline:
            print(f"{label} ...", flush=True)
        else:
            print(_row_label(label), end="", flush=True)
            _row_open = True


def _end_row(label: str, elapsed: float, multiline: bool, note: str | None = None) -> None:
    """Print the seconds of row ``label`` (after the label again when ``multiline``); a single-line row is then closed and the lines held back while it was open are printed."""
    global _row_open
    with _line_lock:
        print((_row_label(label) if multiline else "") + _row_time(elapsed, note), flush=True)
        if not multiline:
            _row_open = False
            while _deferred_lines:
                print(_deferred_lines.pop(0), flush=True)


def _run_row(label: str, show: bool, function, multiline: bool):
    """Call ``function`` under one timed row, which also closes if the call fails or is interrupted."""
    if show:
        _begin_row(label, multiline)
    start = time.perf_counter()
    try:
        result = function()
    except BaseException as error:
        if show:
            note = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            _end_row(label, time.perf_counter() - start, multiline, note)
        raise
    elapsed = time.perf_counter() - start
    if show:
        _end_row(label, elapsed, multiline)
    return result, elapsed



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


def timed_call(label: str, verbosity: bool | int, function, *, multiline: bool = False):
    """Call ``function``, optionally printing one aligned timed row.

    ``multiline`` is for a ``function`` that prints lines of its own: the label goes on a line
    of its own first, and the timed row follows them.
    """
    level = 1 if isinstance(verbosity, bool) and verbosity else int(verbosity or 0)
    return _run_row(label, bool(level), function, multiline)


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
    return _run_row("  " * (level - 1) + label, should_print, function, multiline)
