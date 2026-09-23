from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from pathlib import Path


def test_native_output_larger_than_pipe_capacity_does_not_deadlock_while_holding_gil(tmp_path: Path):
    log_path = tmp_path / "native_burst.log"
    code = r"""
import ctypes
import fcntl
import os
import sys
from scripts.guiding_center.runtime.terminal_log import _TerminalLogTee
# PyDLL deliberately retains the GIL, as the PyAMGX native solve does.
libc = ctypes.PyDLL(None, use_errno=True)
libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.write.restype = ctypes.c_ssize_t
with _TerminalLogTee(sys.argv[1]):
    capacity = fcntl.fcntl(1, fcntl.F_GETPIPE_SZ)
    size = max(1024*1024, 4*capacity)
    for fd, byte in ((1, b'S'), (2, b'E')):
        payload = byte*size + b'\n'
        buffer = ctypes.create_string_buffer(payload)
        written = libc.write(fd, buffer, len(payload))
        assert written == len(payload), (written, len(payload))
print('completed', flush=True)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(log_path)],
        capture_output=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stderr[-1000:]
    assert completed.stdout.endswith(b"completed\n")
    assert completed.stdout.count(b'S') >= 1024*1024
    assert completed.stderr.count(b'E') >= 1024*1024
    captured = log_path.read_bytes()
    assert captured.count(b'S') == completed.stdout.count(b'S')
    assert captured.count(b'E') == completed.stderr.count(b'E')
    assert b'completed' not in captured


def test_terminal_log_restores_streams_and_reaps_pump_after_exception(tmp_path: Path):
    log_path = tmp_path / "exception.log"
    code = r"""
import os
import sys
from scripts.guiding_center.runtime.terminal_log import _TerminalLogTee
tee = _TerminalLogTee(sys.argv[1])
try:
    with tee:
        print('captured stdout', flush=True)
        print('captured stderr', file=sys.stderr, flush=True)
        raise ValueError('expected')
except ValueError:
    pass
assert tee._pump_process.poll() == 0
try:
    os.waitpid(tee._pump_process.pid, os.WNOHANG)
except ChildProcessError:
    pass
else:
    raise AssertionError('log pump was not reaped')
print('restored stdout', flush=True)
print('restored stderr', file=sys.stderr, flush=True)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(log_path)],
        capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'captured stdout' in completed.stdout and 'restored stdout' in completed.stdout
    assert 'captured stderr' in completed.stderr and 'restored stderr' in completed.stderr
    log = log_path.read_text()
    assert 'captured stdout' in log and 'captured stderr' in log
    assert 'restored' not in log


def test_amgx_callback_progress_is_visible_before_native_call_returns(tmp_path: Path):
    log_path = tmp_path / "live_native_progress.log"
    code = r"""
import ctypes
import sys
from types import SimpleNamespace
from hdgfem.backends import cupy as backend
from scripts.guiding_center.runtime.terminal_log import _TerminalLogTee

# No GPU initialization: exercise the shared registration path with a mock
# AMGX runtime, then block in libc while retaining the GIL as PyAMGX does.
libc = ctypes.PyDLL(None)
libc.printf.argtypes = [ctypes.c_char_p]
libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.read.restype = ctypes.c_ssize_t
calls = []
def initialize():
    calls.append('initialize')
    libc.printf(b'native initialization banner\n')
def register(callback):
    calls.append('register')
    amgx.callback = callback
amgx = SimpleNamespace(initialize=initialize, register_print_callback=register)
backend.require_pyamgx = lambda: amgx
backend._PYAMGX_RUNTIME_INITIALIZED = False

with _TerminalLogTee(sys.argv[1]):
    assert backend.initialize_pyamgx_once() is amgx
    assert backend.initialize_pyamgx_once() is amgx
    assert calls == ['initialize', 'register']
    # Deliberately omit the newline: each native message must be forwarded
    # immediately, without relying on Python or C line buffering.
    amgx.callback('native iteration 1: ')
    release = ctypes.create_string_buffer(1)
    assert libc.read(0, release, 1) == 1
    amgx.callback('completed\n')
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(log_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    output = bytearray()
    marker = b"native iteration 1: "
    try:
        deadline = time.monotonic() + 15.0
        while marker not in output:
            remaining = deadline - time.monotonic()
            assert remaining > 0.0, "native progress was buffered until completion"
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            assert ready, "native progress was buffered until completion"
            chunk = os.read(process.stdout.fileno(), 4096)
            assert chunk, "child exited before reporting native progress"
            output.extend(chunk)
        # The child cannot finish until we release its blocking native read.
        assert process.poll() is None
        assert bytes(output) == b"native initialization banner\n" + marker
        assert log_path.read_bytes() == bytes(output)
    finally:
        tail, errors = process.communicate(input=b"x", timeout=10)
    assert process.returncode == 0, errors.decode(errors="replace")
    assert bytes(output) + tail == b"native initialization banner\nnative iteration 1: completed\n"
    assert log_path.read_bytes() == bytes(output) + tail
