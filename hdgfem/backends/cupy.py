"""Optional CuPy backend and CUDA runtime validation helpers."""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any


_CUDA_DLL_HANDLES: list[Any] = []


def _configure_windows_cuda_wheels() -> None:
    """Expose NVIDIA CUDA component-wheel DLLs to CuPy on Windows.

    CuPy 13 can use the CUDA 11 component wheels without a system-wide CUDA
    Toolkit, but Windows does not automatically search their package-local
    ``bin`` directories. Keep the ``add_dll_directory`` handles alive for the
    process lifetime and also update ``PATH`` for CuPy's ``ctypes`` loaders.
    """

    if os.name != "nt":
        return

    nvidia_spec = find_spec("nvidia")
    if nvidia_spec is None or nvidia_spec.submodule_search_locations is None:
        return

    path_entries = {
        os.path.normcase(os.path.normpath(entry))
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry
    }
    for location in nvidia_spec.submodule_search_locations:
        nvidia_root = Path(location)
        nvrtc_root = nvidia_root / "cuda_nvrtc"
        if nvrtc_root.is_dir():
            os.environ.setdefault("CUDA_PATH", str(nvrtc_root))

        for component in ("cublas", "cuda_nvrtc"):
            bin_directory = nvidia_root / component / "bin"
            if not bin_directory.is_dir():
                continue
            bin_path = str(bin_directory)
            normalized_bin_path = os.path.normcase(os.path.normpath(bin_path))
            if normalized_bin_path not in path_entries:
                os.environ["PATH"] = (
                    bin_path + os.pathsep + os.environ.get("PATH", "")
                )
                path_entries.add(normalized_bin_path)
            if hasattr(os, "add_dll_directory"):
                try:
                    _CUDA_DLL_HANDLES.append(os.add_dll_directory(bin_path))
                except OSError:
                    # CuPy's loader can still use the PATH entry above.
                    pass


_configure_windows_cuda_wheels()

try:  # pragma: no cover - depends on optional runtime dependency.
    import cupy as cp
except ImportError as error:  # pragma: no cover
    cp = None
    _IMPORT_ERROR = error
else:  # pragma: no cover
    _IMPORT_ERROR = None


def require_cupy():
    """Return the CuPy module or raise a clear dependency error."""
    if cp is None:
        raise RuntimeError("CuPy is not installed in this environment") from _IMPORT_ERROR
    return cp


def require_cupy_device():
    """Return CuPy after verifying that at least one CUDA device is usable."""

    module = require_cupy()
    try:
        device_count = int(module.cuda.runtime.getDeviceCount())
    except Exception as error:  # pragma: no cover - CUDA-runtime dependent.
        raise RuntimeError(
            "CuPy is installed, but the CUDA runtime or driver is unavailable"
        ) from error

    if device_count < 1:  # pragma: no cover - depends on runtime hardware.
        raise RuntimeError("CuPy is installed, but no CUDA device is available")
    return module


__all__ = ["require_cupy", "require_cupy_device"]
