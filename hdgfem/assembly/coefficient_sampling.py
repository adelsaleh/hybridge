"""Memory-bounded, GPU-first sampling of bundled analytic coefficients.

This array-only module deliberately has no imports from other HDGFEM modules:
isolated campaign adapters can share it without mixing solver worktrees.
See ``docs/reference/coefficient_sampling.md`` for the callable contract.
Importing this module neither initializes CUDA nor compiles a kernel.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np


def _point_loop(function, parameters, x, y, out):
    for element in prange(x.shape[0]):
        for q in range(x.shape[1]):
            values = function(x[element, q], y[element, q], parameters)
            for component in range(out.shape[0]):
                out[component, element, q] = values[component]


# Replaced by Numba's intrinsic only when the CPU backend is requested.
prange = range


class CoefficientSampler:
    """Evaluate all components together, with bounded device temporaries.

    ``function(x, y, parameters)`` returns a fixed-length tuple, accepting
    NumPy/CuPy arrays and (for Numba) scalar arguments. Dependencies of the
    scalar function must be Numba-jitable. No object-mode fallback is used.

    Results are host FP64 arrays with shape ``(components, *x.shape)``. This
    explicit residency contract supports host assembly without retaining a
    full-mesh device mirror. Only device-unavailable/OOM failures select CPU;
    coefficient/programming errors are never silently retried on another backend.
    """

    def __init__(self, *, backend="auto", device=0, chunk_points=131072,
                 memory_fraction=0.5, reserve_bytes=512*1024**2,
                 scratch_arrays=256):
        if backend not in {"auto", "cupy", "numba", "numpy"}:
            raise ValueError("sampling backend must be auto, cupy, numba or numpy")
        if int(chunk_points) != chunk_points or chunk_points < 1:
            raise ValueError("chunk_points must be a positive integer")
        if not 0 < memory_fraction <= 1 or not np.isfinite(memory_fraction):
            raise ValueError("memory_fraction must be in (0, 1]")
        if reserve_bytes < 0 or scratch_arrays < 1:
            raise ValueError("reserve_bytes must be nonnegative and scratch_arrays positive")
        self.backend, self.device = backend, int(device)
        self.chunk_points, self.memory_fraction = int(chunk_points), float(memory_fraction)
        self.reserve_bytes, self.scratch_arrays = int(reserve_bytes), int(scratch_arrays)
        self._gpu_checked, self._cp, self._pool = False, None, None
        self._cpu_functions, self._cpu_loop = {}, None
        self.stats = dict(requested_backend=backend, gpu_batches=0, cpu_batches=0,
                          oom_retries=0, fallback_reason=None, warmup_ms=0.0,
                          evaluation_ms=0.0, host_to_device_ms=0.0,
                          device_to_host_ms=0.0, sampled_points=0)

    def _gpu(self):
        if self._gpu_checked:
            return self._cp
        self._gpu_checked = True
        try:
            import cupy as cp
        except ImportError as exc:
            if self.backend == "cupy":
                raise RuntimeError("CuPy sampling requested but CuPy is unavailable") from exc
            self.stats["fallback_reason"] = "CuPy unavailable"
            return None
        try:
            with cp.cuda.Device(self.device):
                cp.cuda.runtime.memGetInfo()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            # Invalid device IDs and unexpected CUDA failures must not be hidden.
            if self.backend == "cupy" or exc.status not in (35, 100):
                raise
            self.stats["fallback_reason"] = f"CUDA unavailable: {exc}"
            return None
        self._cp, self._pool = cp, cp.cuda.MemoryPool()
        return cp

    def _cpu(self, function, parameters, x, y, out):
        if self.backend == "numpy":
            start = perf_counter()
            values = function(x, y, parameters)
            if len(values) != out.shape[0]:
                raise ValueError("coefficient component count mismatch")
            for i, value in enumerate(values):
                out[i] = value
            self.stats["evaluation_ms"] += 1000*(perf_counter()-start)
        else:
            import numba
            if numba.config.DISABLE_JIT:
                raise RuntimeError("Numba sampling requires JIT; use backend='numpy' for non-compiling checks")
            if function not in self._cpu_functions:
                global prange
                prange = numba.prange
                if self._cpu_loop is None:
                    self._cpu_loop = numba.njit(parallel=True, cache=True, fastmath=False)(_point_loop)
                compiled = numba.njit(cache=True, fastmath=False)(function)
                self._cpu_functions[function] = compiled
            compiled = self._cpu_functions[function]
            signature = tuple(numba.typeof(value) for value in (compiled, parameters, x, y, out))
            if signature not in self._cpu_loop.signatures:
                start = perf_counter()
                self._cpu_loop.compile(signature)
                self.stats["warmup_ms"] += 1000*(perf_counter()-start)
            start = perf_counter()
            self._cpu_loop(compiled, parameters, x, y, out)
            self.stats["numba_threads"] = numba.get_num_threads()
            self.stats["numba_threading_layer"] = numba.threading_layer()
            self.stats["evaluation_ms"] += 1000*(perf_counter()-start)
        self.stats["cpu_batches"] += 1

    def _gpu_batch(self, cp, function, parameters, x, y, out):
        # Failed-batch arrays die when this frame unwinds, before OOM retry.
        sync = cp.cuda.get_current_stream().synchronize
        start = perf_counter()
        dx, dy = cp.asarray(x), cp.asarray(y)
        sync()
        self.stats["host_to_device_ms"] += 1000*(perf_counter()-start)
        start = perf_counter()
        values = function(dx, dy, parameters)
        if len(values) != out.shape[0]:
            raise ValueError("coefficient component count mismatch")
        values = tuple(cp.broadcast_to(cp.asarray(v, dtype=cp.float64), x.shape) for v in values)
        sync()
        self.stats["evaluation_ms"] += 1000*(perf_counter()-start)
        start = perf_counter()
        for i, value in enumerate(values):
            out[i] = cp.asnumpy(value)
        self.stats["device_to_host_ms"] += 1000*(perf_counter()-start)
        self.stats["gpu_batches"] += 1

    def sample(self, function, x, y, parameters=(), *, components):
        """Sample broadcastable host coordinates without projection/interpolation.

        For rank-two inputs, batches and ``prange`` preserve the element axis;
        all quadrature points of an element are evaluated by the same CPU worker.
        Device OOM halves the batch, then falls back to Numba at one element
        in ``auto`` mode. Forced ``cupy`` mode reports an unrecoverable OOM.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        if int(components) != components or components < 1:
            raise ValueError("components must be a positive integer")
        shape = x.shape
        if x.ndim != 2:
            x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        out = np.empty((int(components), *x.shape), dtype=np.float64)
        if not x.size:
            return out.reshape((int(components), *shape))
        cp = self._gpu() if self.backend in {"auto", "cupy"} else None
        rows = max(1, self.chunk_points//x.shape[1])
        first = 0
        while first < len(x):
            count = min(rows, len(x)-first)
            if cp is not None:
                with cp.cuda.Device(self.device), cp.cuda.using_allocator(self._pool.malloc):
                    free, _ = cp.cuda.runtime.memGetInfo()
                    budget = max(0, (free+self._pool.free_bytes()-self.reserve_bytes)*self.memory_fraction)
                    affordable = int(budget//(8*x.shape[1]*(self.scratch_arrays+components+2)))
                    if affordable < 1:
                        if self.backend == "cupy":
                            raise MemoryError("Insufficient GPU sampling budget for one element")
                        self.stats["fallback_reason"] = "GPU sampling memory reserve"
                        cp = None
                        self._pool.free_all_blocks()
                        continue
                    count = min(count, affordable)
                    try:
                        self._gpu_batch(cp, function, parameters, x[first:first+count],
                                        y[first:first+count], out[:, first:first+count])
                    except cp.cuda.memory.OutOfMemoryError:
                        self.stats["oom_retries"] += 1
                        self._pool.free_all_blocks()
                        if count > 1:
                            rows = max(1, count//2)
                            continue
                        if self.backend == "cupy":
                            raise
                        self.stats["fallback_reason"] = "GPU OOM at minimum batch"
                        cp = None
                        rows = max(1, self.chunk_points//x.shape[1])
                        continue
            else:
                self._cpu(function, parameters, np.ascontiguousarray(x[first:first+count]),
                          np.ascontiguousarray(y[first:first+count]), out[:, first:first+count])
            if not np.all(np.isfinite(out[:, first:first+count])):
                raise ValueError("Nonfinite sampled coefficient")
            self.stats["sampled_points"] += count*x.shape[1]
            first += count
        if self._pool is not None:
            self._pool.free_all_blocks()
        return out.reshape((int(components), *shape))
