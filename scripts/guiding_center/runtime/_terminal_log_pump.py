"""Drain guiding-center native stdout/stderr independently of the solver GIL.

Private subprocess entry point. Arguments are inherited descriptor numbers:
log_fd stdout_read_fd stdout_mirror_fd stderr_read_fd stderr_mirror_fd.
Only the standard library is loaded; this process never initializes CUDA.
"""

from __future__ import annotations

import os
import selectors
import sys


def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        try:
            written = os.write(fd, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("log write made no progress")
        remaining = remaining[written:]


def _pump(log_fd: int, streams: list[tuple[int, int]]) -> int:
    log_error = None
    failed_mirrors = set()
    with selectors.DefaultSelector() as selector:
        for read_fd, mirror_fd in streams:
            selector.register(read_fd, selectors.EVENT_READ, mirror_fd)
        while selector.get_map():
            for key, _ in selector.select():
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    selector.unregister(key.fd)
                    os.close(key.fd)
                    continue
                if log_error is None:
                    try:
                        _write_all(log_fd, chunk)
                    except OSError as error:
                        # Keep draining even when logging fails, so the solver
                        # can return and the parent can report the failure.
                        log_error = error
                if key.data not in failed_mirrors:
                    try:
                        _write_all(key.data, chunk)
                    except OSError:
                        # A closed terminal must not prevent capture to disk.
                        failed_mirrors.add(key.data)
    if log_error is not None:
        print(f"[gc] terminal log write failed: {log_error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    log_fd, stdout_read, stdout_mirror, stderr_read, stderr_mirror = map(int, sys.argv[1:])
    sys.exit(_pump(log_fd, [(stdout_read, stdout_mirror), (stderr_read, stderr_mirror)]))
