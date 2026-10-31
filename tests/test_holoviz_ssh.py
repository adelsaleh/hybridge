"""X11 wire framing checks; no GPU, renderer, or display required."""
import io
import struct

import pytest

from hybridge.io.holoviz_ssh import _request


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
