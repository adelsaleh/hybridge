"""Mirror process stdout/stderr to a durable terminal transcript.

The optimizer uses native libraries (PETSc, MUMPS, VTK) that can write
directly to file descriptors 1 and 2.  Replacing only ``sys.stdout`` and
``sys.stderr`` would miss those messages, so this module tees the underlying
file descriptors instead.
"""

from __future__ import annotations

import atexit
import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
from types import FrameType
from typing import Callable


@dataclass
class _Redirect:
    """Bookkeeping for one redirected standard stream."""

    target_fd: int
    saved_fd: int
    read_fd: int
    thread: threading.Thread


class TerminalLogCapture:
    """Tee stdout and stderr to one append-only log while preserving display.

    Each MPI rank may open the same path: ``O_APPEND`` keeps every individual
    write intact while the ranks continue forwarding their normal output to
    the launcher.  The class itself deliberately has no MPI dependency.
    """

    _CHUNK_SIZE = 64 * 1024

    def __init__(
            self,
            path: Path,
            *,
            rank: int | None = None,
            remove_on_close: bool = False,
            close_on_signal: bool = True,
            mirror_to_terminal: bool = True,
    ) -> None:
        self.path = Path(path)
        self.rank = rank
        self._remove_on_close = bool(remove_on_close)
        # A scoped CLI may need to write an exit summary before draining the
        # pipes. Existing script callers retain immediate signal-close behavior.
        self._close_on_signal = bool(close_on_signal)
        # MPI workers still need complete native transcripts, but forwarding
        # eight identical shutdown/bootstrap records to the launcher's stdout
        # makes the useful rank-zero narrative hard to read.  Mirroring is a
        # presentation switch only: both descriptor pumps always write every
        # byte to this rank's durable log.
        self._mirror_to_terminal = bool(mirror_to_terminal)
        self._log_fd: int | None = None
        self._redirects: list[_Redirect] = []
        self._previous_signal_handlers: dict[
            int, int | Callable[[int, FrameType | None], object] | None
        ] = {}
        self._log_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._started = False
        self._closing = False
        self._closed = False

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        """Write a complete byte string, retrying interrupted/partial writes."""
        view = memoryview(data)
        while view:
            try:
                written = os.write(fd, view)
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError("file-descriptor write made no progress")
            view = view[written:]

    @classmethod
    def _best_effort_write(cls, fd: int, data: bytes) -> None:
        try:
            cls._write_all(fd, data)
        except OSError:
            # Losing the terminal must not prevent the durable log write, and
            # losing the log must not deadlock the process during shutdown.
            pass

    def _pump(self, read_fd: int, display_fd: int) -> None:
        """Copy one pipe to the log and its original terminal descriptor."""
        try:
            while True:
                try:
                    chunk = os.read(read_fd, self._CHUNK_SIZE)
                except InterruptedError:
                    continue
                if not chunk:
                    break
                with self._log_lock:
                    log_fd = self._log_fd
                    if log_fd is not None:
                        self._best_effort_write(log_fd, chunk)
                if self._mirror_to_terminal:
                    self._best_effort_write(display_fd, chunk)
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass

    @staticmethod
    def _supported_shutdown_signals() -> tuple[int, ...]:
        names = ("SIGINT", "SIGTERM", "SIGHUP")
        return tuple(
            int(getattr(signal, name))
            for name in names
            if hasattr(signal, name)
        )

    def _interruption_marker(self, signum: int) -> bytes:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        rank_field = f" rank={self.rank}" if self.rank is not None else ""
        return (
            f"\nTERMINAL_LOG_INTERRUPTED signal={signal_name}"
            f"{rank_field} pid={os.getpid()}\n"
        ).encode("utf-8", errors="replace")

    def _handle_shutdown_signal(self, signum: int, frame: FrameType | None) -> None:
        """Record catchable interruptions and preserve prior signal semantics."""
        # Descriptor 2 points at our pipe, so the marker is ordered with that
        # rank's preceding stderr and is mirrored to both destinations.
        self._best_effort_write(2, self._interruption_marker(signum))
        previous = self._previous_signal_handlers.get(signum, signal.SIG_DFL)
        if callable(previous):
            # Python's default SIGINT handler raises KeyboardInterrupt.  Keep
            # capture active so the resulting traceback is logged as well.
            previous(signum, frame)
            return
        if previous == signal.SIG_IGN:
            return

        # SIGTERM/SIGHUP have no Python exception by default.  Drain the pipes
        # explicitly before performing a normal interpreter exit, which also
        # permits other atexit handlers to save partial run artifacts.
        if self._close_on_signal:
            self.close()
        raise SystemExit(128 + int(signum))

    @staticmethod
    def _flush_streams() -> None:
        """Flush Python and C stdio while the descriptors still point at the tee.

        Native ``printf``/``puts`` can remain buffered when stdout is a pipe;
        flushing only Python would lose those bytes from the transcript.
        Direct descriptor writers such as ``os.write`` need no extra flush.
        """
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        try:
            fflush = ctypes.CDLL(None).fflush
            fflush.argtypes = [ctypes.c_void_p]
            fflush.restype = ctypes.c_int
            fflush(None)
        except (AttributeError, OSError):
            # Platforms without an exported libc fflush retain Python/fd teeing.
            pass

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in self._supported_shutdown_signals():
            previous = signal.getsignal(signum)
            if previous == signal.SIG_IGN:
                continue
            if previous != self._handle_shutdown_signal:
                self._previous_signal_handlers.setdefault(signum, previous)
            signal.signal(signum, self._handle_shutdown_signal)

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, previous in self._previous_signal_handlers.items():
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError):
                pass
        self._previous_signal_handlers.clear()

    def start(self) -> None:
        """Begin capturing both standard streams."""
        with self._state_lock:
            if self._started and not self._closed:
                return
            if self._closed:
                raise RuntimeError("a closed terminal capture cannot be restarted")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fd = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )
            try:
                for target_fd, stream_name in ((1, "stdout"), (2, "stderr")):
                    saved_fd = os.dup(target_fd)
                    read_fd, write_fd = os.pipe()
                    try:
                        os.dup2(write_fd, target_fd)
                    finally:
                        os.close(write_fd)
                    thread = threading.Thread(
                        target=self._pump,
                        args=(read_fd, saved_fd),
                        name=f"terminal-log-{stream_name}",
                        daemon=True,
                    )
                    self._redirects.append(
                        _Redirect(target_fd, saved_fd, read_fd, thread)
                    )
                    thread.start()
                self._install_signal_handlers()
                self._started = True
                atexit.register(self.close)
            except BaseException:
                self._started = True
                self.close()
                raise

    def retarget(self, path: Path, *, rank: int | None = None) -> None:
        """Move a live bootstrap transcript into its final run-log path.

        Output already drained by the pump threads is copied first.  Output
        still queued in either descriptor pipe is written to the new file
        afterward, so the transition does not lose bytes.
        """
        destination = Path(path)
        with self._state_lock:
            if not self._started or self._closed or self._closing:
                raise RuntimeError("terminal capture must be active before retargeting")
            source = self.path
            if destination == source:
                self.rank = rank
                self._remove_on_close = False
                self._install_signal_handlers()
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            new_log_fd = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )

        self._flush_streams()

        old_log_fd: int | None = None
        try:
            with self._log_lock:
                early_output = source.read_bytes()
                if early_output:
                    self._write_all(new_log_fd, early_output)
                old_log_fd = self._log_fd
                self._log_fd = new_log_fd
                self.path = destination
                self.rank = rank
                self._remove_on_close = False
        except BaseException:
            try:
                os.close(new_log_fd)
            except OSError:
                pass
            raise

        if old_log_fd is not None:
            try:
                os.close(old_log_fd)
            except OSError:
                pass
        try:
            source.unlink()
        except FileNotFoundError:
            pass
        # PETSc/MPI initialization may have installed native handlers after a
        # bootstrap capture started.  Reassert the interruption-aware Python
        # handlers once all solver imports have completed.
        self._install_signal_handlers()

    def set_terminal_mirroring(self, enabled: bool) -> None:
        """Enable/disable live display without changing durable capture.

        This is intentionally safe to call after MPI identifies the rank.  A
        launcher rank hint normally suppresses worker bootstrap chatter even
        earlier; this method is the authoritative correction once ``comm`` is
        available.
        """
        with self._state_lock:
            self._mirror_to_terminal = bool(enabled)

    def close(self) -> None:
        """Flush, restore the terminal descriptors, and durably close the log."""
        with self._state_lock:
            if self._closed or self._closing or not self._started:
                return
            self._closing = True

        self._flush_streams()

        self._restore_signal_handlers()
        for redirect in self._redirects:
            try:
                os.dup2(redirect.saved_fd, redirect.target_fd)
            except OSError:
                pass

        current_thread = threading.current_thread()
        for redirect in self._redirects:
            if redirect.thread is not current_thread:
                redirect.thread.join()

        for redirect in self._redirects:
            try:
                os.close(redirect.saved_fd)
            except OSError:
                pass
        self._redirects.clear()

        with self._log_lock:
            if self._log_fd is not None:
                try:
                    os.fsync(self._log_fd)
                except OSError:
                    pass
                try:
                    os.close(self._log_fd)
                except OSError:
                    pass
                self._log_fd = None

        if self._remove_on_close:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

        with self._state_lock:
            self._closed = True
            self._closing = False

    def __enter__(self) -> "TerminalLogCapture":
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


_BOOTSTRAP_CAPTURE: TerminalLogCapture | None = None
_BOOTSTRAP_LOCK = threading.Lock()


def terminal_log_requested(argv: list[str] | tuple[str, ...]) -> bool:
    """Apply argparse-style last-option-wins handling to the boolean flag."""
    requested = False
    for argument in argv:
        if argument == "--save-terminal-log":
            requested = True
        elif argument == "--no-save-terminal-log":
            requested = False
    return requested


def start_bootstrap_terminal_log_capture(
        argv: list[str] | tuple[str, ...],
) -> TerminalLogCapture | None:
    """Start a temporary transcript before heavyweight solver imports."""
    global _BOOTSTRAP_CAPTURE
    if not terminal_log_requested(argv):
        return None
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAP_CAPTURE is not None:
            return _BOOTSTRAP_CAPTURE
        file_fd, file_name = tempfile.mkstemp(
            prefix=f"hdgfem-terminal-{os.getpid()}-",
            suffix=".log",
        )
        os.close(file_fd)
        capture = TerminalLogCapture(Path(file_name), remove_on_close=True)
        try:
            capture.start()
        except BaseException:
            try:
                Path(file_name).unlink()
            except FileNotFoundError:
                pass
            raise
        _BOOTSTRAP_CAPTURE = capture
        return capture


def get_bootstrap_terminal_log_capture() -> TerminalLogCapture | None:
    """Return the early capture, if the current CLI requested one."""
    return _BOOTSTRAP_CAPTURE
