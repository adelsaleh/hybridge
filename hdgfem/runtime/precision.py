"""Process-wide floating-point precision for the numerical pipeline.

Set HDGFEM_PRECISION before importing hdgfem. Precision is immutable within a
process so cached reference tables, device mirrors and compiled kernels cannot
silently mix formats. Integer topology and indices are unaffected.
"""

from __future__ import annotations

import os
import re

import numpy as np

PRECISION = os.environ.get("HDGFEM_PRECISION", "float64")
if PRECISION not in {"float32", "float64"}:
    raise ValueError("HDGFEM_PRECISION must be float32 or float64")
REAL_DTYPE = np.float32 if PRECISION == "float32" else np.float64
REAL_ITEMSIZE = np.dtype(REAL_DTYPE).itemsize
AMGX_MODE = "dFFI" if PRECISION == "float32" else "dDDI"


def specialize_real_source(source: str, dtype) -> str:
    """Specialize a double-precision CUDA template to ``dtype``.

    ``float64`` returns the source unchanged. ``float32`` replaces ``double``,
    switches the listed math calls to their ``f`` forms, and suffixes floating
    literals so that C++ promotion cannot move FP32 arithmetic back to FP64.
    Apply it after the assembly/layout substitutions. Unlike
    :func:`cuda_source`, the target does not depend on ``HDGFEM_PRECISION``,
    which lets mixed-precision paths compile FP32 kernels in an FP64 process.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return source
    if dtype != np.float32:
        raise ValueError(f"unsupported CUDA real type {dtype}")
    source = re.sub(r"\bdouble\b", "float", source)
    source = re.sub(
        r"\b(fabs|sqrt|rsqrt|fmax|fmin|pow|exp|log|sin|cos|hypot|copysign)\s*\(",
        lambda match: match.group(1) + "f(", source,
    )
    return re.sub(
        r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])",
        r"\1f", source,
    )


def cuda_source(source: str) -> str:
    """Specialize real-valued CUDA templates, including constants and math.

Specialization happens after the existing assembly/layout substitutions. Float
suffixes prevent C++ literals from promoting FP32 arithmetic back to FP64.
"""
    return specialize_real_source(source, REAL_DTYPE)


def real_raw_kernel(source: str, name: str, **kwargs):
    """Compile a CUDA kernel using the selected numerical precision."""
    import cupy

    kernel = cupy.RawKernel(cuda_source(source), name, **kwargs)
    return _CheckedKernel(kernel, name) if PRECISION == "float32" else kernel


def real_raw_module(*, code: str, **kwargs):
    """Compile a CUDA module using the selected numerical precision."""
    import cupy

    module = cupy.RawModule(code=cuda_source(code), **kwargs)
    return _CheckedModule(module) if PRECISION == "float32" else module


KERNEL_AUDIT: dict[str, dict] = {}


def check_real_arrays(stage: str, values) -> None:
    """Reject floating arrays or NumPy scalars that violate selected precision."""
    for index, value in enumerate(values):
        if PRECISION == "float32" and isinstance(value, float):
            raise TypeError(f"{stage}: argument {index} is a Python float; use an explicit float32 scalar")
        dtype = getattr(value, "dtype", None)
        if dtype is not None and np.dtype(dtype).kind == "f" and dtype != np.dtype(REAL_DTYPE):
            raise TypeError(f"{stage}: argument {index} has {dtype}; expected {PRECISION}")


class _CheckedKernel:
    """Forward a compiled kernel while auditing every floating argument."""

    def __init__(self, kernel, name: str):
        """Store the compiled kernel and its descriptive audit key."""
        object.__setattr__(self, "_kernel", kernel)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, name):
        """Forward CUDA attributes and compilation controls."""
        return getattr(self._kernel, name)

    def __setattr__(self, name, value):
        """Forward CUDA resource attributes such as dynamic shared memory."""
        setattr(self._kernel, name, value)

    def __call__(self, grid, block, args, **kwargs):
        """Check kernel arguments before forwarding the launch unchanged."""
        check_real_arrays(self._name, args)
        record = KERNEL_AUDIT.setdefault(self._name, {"calls": 0, "precision": PRECISION})
        record["calls"] += 1
        return self._kernel(grid, block, args, **kwargs)


class _CheckedModule:
    """Forward module compilation and wrap each retrieved CUDA function."""

    def __init__(self, module):
        """Store the compiled module."""
        self._module = module

    def __getattr__(self, name):
        """Forward attributes and compilation controls."""
        return getattr(self._module, name)

    def get_function(self, name):
        """Return an argument-audited function from this module."""
        return _CheckedKernel(self._module.get_function(name), name)


PIPELINE_AUDIT: dict[str, dict] = {}


def audit_arrays(stage: str, *objects) -> None:
    """Audit owned numerical arrays, including mirrors, factors and workspaces.

    Python control scalars and third-party implementation objects are excluded;
    all NumPy/CuPy floating arrays reachable through hdgfem objects are checked.
    """
    if os.environ.get('HDGFEM_PRECISION_AUDIT', '1' if PRECISION == 'float32' else '0') != '1':
        return
    visited = set()
    counts = {'host_arrays': 0, 'device_arrays': 0, 'host_bytes': 0, 'device_bytes': 0}

    def visit(value, path: str) -> None:
        """Walk owned containers without following functions or library internals."""
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, np.ndarray) or hasattr(value, '__cuda_array_interface__'):
            if np.dtype(value.dtype).kind == 'f':
                if value.dtype != np.dtype(REAL_DTYPE):
                    raise TypeError(f'{stage}: {path} has {value.dtype}; expected {PRECISION}')
                location = 'device' if hasattr(value, '__cuda_array_interface__') else 'host'
                counts[f'{location}_arrays'] += 1
                counts[f'{location}_bytes'] += int(value.nbytes)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f'{path}.{key}')
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f'{path}[{index}]')
        elif type(value).__module__.startswith(('hdgfem.', 'scripts.guiding_center.')) and hasattr(value, '__dict__'):
            for key, item in vars(value).items():
                visit(item, f'{path}.{key}')

    for index, value in enumerate(objects):
        visit(value, str(index))
    previous = PIPELINE_AUDIT.get(stage, {})
    PIPELINE_AUDIT[stage] = dict(precision=PRECISION, calls=previous.get('calls', 0)+1, **counts)
