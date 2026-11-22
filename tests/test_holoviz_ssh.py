"""X11 wire framing checks; no GPU, renderer, or display required."""
import io
import socket
import struct
import subprocess
import threading
from types import SimpleNamespace

import pytest

from hybridge.io.holoviz_ssh import _check_display, _request, _x_setup_accepted


class FragmentedSocket:
    def __init__(self, data):
        self.stream = io.BytesIO(data)

    def recv(self, size):
        return self.stream.read(min(size, 3))


@pytest.mark.parametrize("order", ["<", ">"])
@pytest.mark.parametrize("big", [False, True])
@pytest.mark.parametrize("name", [b"NV-GLX", b"DRI3", b"NV-CONTROL"])
def test_masks_only_nv_glx_preserving_wire_framing(order, big, name):
    payload = struct.pack(order + "HH", len(name), 0) + name
    payload += b"\0" * (-len(payload) % 4)
    size = len(payload) + (8 if big else 4)
    wire = struct.pack(order + "BBH", 98, 0, 0 if big else size // 4)
    if big:
        wire += struct.pack(order + "I", size // 4)
    wire += payload
    result = _request(FragmentedSocket(wire), order)
    assert result == (wire.replace(b"NV-GLX", b"ZZ-GLX") if name == b"NV-GLX" else wire)


def test_truncated_request_stops():
    with pytest.raises(EOFError):
        _request(FragmentedSocket(b"\x62\0\x04\0"), "<")


def test_invalid_big_request_length_stops():
    with pytest.raises(ValueError):
        _request(FragmentedSocket(struct.pack("<BBHI", 98, 0, 0, 1)), "<")


def _fake_x_server(reply):
    """Serve one X connection setup on a loopback port; return the port and the received request."""
    server = socket.create_server(("127.0.0.1", 0))
    received = {}

    def serve():
        connection, _ = server.accept()
        with connection, server:
            received["request"] = connection.recv(4096)
            connection.sendall(bytes([reply]))

    threading.Thread(target=serve, daemon=True).start()
    return server.getsockname()[1], received


def test_setup_request_carries_the_cookie_and_reads_the_status():
    cookie = bytes(range(16))
    port, received = _fake_x_server(1)
    assert _x_setup_accepted(port, cookie)
    request = received["request"]
    assert request[:1] == b"l" and struct.unpack_from("<HHHH", request, 2) == (11, 0, 18, 16)
    assert request[12:30] == b"MIT-MAGIC-COOKIE-1" and request[32:48] == cookie and len(request) == 48
    port, _ = _fake_x_server(0)
    assert not _x_setup_accepted(port, cookie)


def test_display_check_without_xdpyinfo_uses_the_setup_request(monkeypatch):
    port, _ = _fake_x_server(1)

    def run(command, **kwargs):
        if command[0] == "xdpyinfo":
            raise FileNotFoundError
        return SimpleNamespace(stdout=f"host/unix:7  MIT-MAGIC-COOKIE-1  {'ab' * 16}\n")

    monkeypatch.setattr(subprocess, "run", run)
    _check_display(dict(DISPLAY=f"localhost:{port - 6000}"))
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: (_ for _ in ()).throw(FileNotFoundError)
                        if command[0] == "xdpyinfo" else SimpleNamespace(stdout=""))
    with pytest.raises(RuntimeError, match="Cannot open DISPLAY"):
        _check_display(dict(DISPLAY="localhost:10.0"))
