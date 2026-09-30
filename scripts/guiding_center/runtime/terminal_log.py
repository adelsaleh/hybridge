"""Guiding-center terminal log helpers."""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from hdgfem.runtime.terminal import flush_terminal_streams as _flush_terminal_streams
from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset


class _TerminalLogTee:
    """Mirror stdout/stderr using a process independent of the solver's GIL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._log_fd: int | None = None
        self._saved_fds: dict[int, int] = {}
        self._pump_process: subprocess.Popen | None = None

    def __enter__(self):
        _flush_terminal_streams()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fd = os.open(
            self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644,
        )
        pipe_fds = []
        write_ends = {}
        pump_args = [str(self._log_fd)]
        inherited = [self._log_fd]
        try:
            for target_fd in (1, 2):
                saved_fd = os.dup(target_fd)
                self._saved_fds[target_fd] = saved_fd
                read_fd, write_fd = os.pipe()
                pipe_fds.extend((read_fd, write_fd))
                write_ends[target_fd] = write_fd
                inherited.extend((read_fd, saved_fd))
                pump_args.extend((str(read_fd), str(saved_fd)))
            # A Python thread cannot drain native output while PyAMGX holds
            # the GIL. Exec a tiny independent process before redirecting FDs.
            self._pump_process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("_terminal_log_pump.py")), *pump_args],
                pass_fds=tuple(inherited),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._saved_fds[2],
                start_new_session=True,
            )
            for target_fd, write_fd in write_ends.items():
                os.dup2(write_fd, target_fd)
        except BaseException:
            self._restore_descriptors()
            # Close every writer before waiting for the pump to observe EOF.
            for fd in pipe_fds:
                os.close(fd)
            self._wait_and_close()
            raise
        for fd in pipe_fds:
            os.close(fd)
        return self

    def _restore_descriptors(self) -> None:
        for target_fd, saved_fd in self._saved_fds.items():
            try:
                os.dup2(saved_fd, target_fd)
            except OSError:
                pass

    def _wait_and_close(self) -> int:
        try:
            return 0 if self._pump_process is None else self._pump_process.wait()
        finally:
            for saved_fd in self._saved_fds.values():
                try:
                    os.close(saved_fd)
                except OSError:
                    pass
            self._saved_fds.clear()
            if self._log_fd is not None:
                os.close(self._log_fd)
                self._log_fd = None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        _flush_terminal_streams()
        self._restore_descriptors()
        returncode = self._wait_and_close()
        if returncode and exc_type is None:
            raise RuntimeError(f"terminal log pump failed for {self.path} (exit {returncode})")
        return False


def _terminal_log_path(
        config: GuidingCenterRunPreset,
        preset_key: str,
) -> Path:
    """Return the terminal-log path matching the diagnostics output stem."""
    output_stem = str(config.diagnostics_prefix or preset_key).strip() or "guiding_center"
    return Path(config.diagnostics_dir) / f"{output_stem}.log"
