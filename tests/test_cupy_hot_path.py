from __future__ import annotations

from dataclasses import dataclass

from hdgfem.backends.cupy import device_arrays_overlap


@dataclass(frozen=True)
class _Pointer:
    ptr: int


@dataclass(frozen=True)
class _Device:
    id: int


@dataclass(frozen=True)
class _FakeDeviceArray:
    pointer: int
    nbytes: int
    device_id: int = 0

    @property
    def data(self) -> _Pointer:
        return _Pointer(self.pointer)

    @property
    def device(self) -> _Device:
        return _Device(self.device_id)


def test_device_arrays_overlap_detects_aliases_without_device_work() -> None:
    base = _FakeDeviceArray(1000, 800)
    same = _FakeDeviceArray(1000, 800)
    interior = _FakeDeviceArray(1200, 100)
    touching_end = _FakeDeviceArray(1800, 200)
    separate = _FakeDeviceArray(2000, 200)

    assert device_arrays_overlap(base, same)
    assert device_arrays_overlap(base, interior)
    assert not device_arrays_overlap(base, touching_end)
    assert not device_arrays_overlap(base, separate)


def test_device_arrays_overlap_handles_devices_and_empty_arrays() -> None:
    first = _FakeDeviceArray(1000, 128, device_id=0)
    other_device = _FakeDeviceArray(1000, 128, device_id=1)
    empty = _FakeDeviceArray(1000, 0, device_id=0)

    assert not device_arrays_overlap(first, other_device)
    assert not device_arrays_overlap(first, empty)
