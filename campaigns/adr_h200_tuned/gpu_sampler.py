"""Sample this job's GPU utilization through NVML (ctypes, no pynvml) in a background thread.

``GpuSampler(pci_bus_id)`` records, every ``period`` seconds, the NVML utilization rates of
the device (SM busy % and memory-controller busy %) and its used memory. ``summary()`` gives
the timeline and the statistics a GPU-usage audit looks at: mean and median SM busy, the 10th
percentile, and the share of samples below 10%. NVML rates are whole-device and
sampling-window based, so they include any other process on the same GPU (none in the lanes).
"""
from __future__ import annotations

import ctypes
import statistics
import threading
from time import perf_counter


class _Utilization(ctypes.Structure):
    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]


class GpuSampler:
    def __init__(self, pci_bus_id, period=1.0):
        self.period, self.samples, self.error = period, [], None
        self._stop = threading.Event()
        self._thread = None
        try:
            self.nvml = ctypes.CDLL('libnvidia-ml.so.1')
            if self.nvml.nvmlInit_v2() != 0:
                raise RuntimeError('nvmlInit_v2 failed')
            self.handle = ctypes.c_void_p()
            bus = pci_bus_id.encode()
            if self.nvml.nvmlDeviceGetHandleByPciBusId_v2(ctypes.c_char_p(bus), ctypes.byref(self.handle)) != 0:
                # NVML spells the PCI domain with 8 digits, CUDA with 4.
                bus = ('0000'+pci_bus_id).encode()
                if self.nvml.nvmlDeviceGetHandleByPciBusId_v2(ctypes.c_char_p(bus), ctypes.byref(self.handle)) != 0:
                    raise RuntimeError(f'no NVML handle for {pci_bus_id}')
        except (OSError, RuntimeError) as error:
            self.nvml, self.error = None, str(error)

    def _sample(self):
        utilization, memory = _Utilization(), _Memory()
        if self.nvml.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(utilization)) != 0:
            return None
        self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory))
        return utilization.gpu, utilization.memory, memory.used

    def _run(self):
        while not self._stop.wait(self.period):
            value = self._sample()
            if value is not None:
                self.samples.append((round(perf_counter()-self.started, 2), *value))

    def start(self):
        self.started = perf_counter()
        if self.nvml is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def now(self):
        """Seconds since ``start()``, the clock of the timeline (for per-candidate windows)."""
        return perf_counter()-self.started

    def summary(self, since=None, until=None, timeline=True):
        """Statistics over the samples in ``[since, until]`` seconds of the timeline (all by default)."""
        if self.nvml is None:
            return dict(available=False, error=self.error)
        rows = [s for s in self.samples if (since is None or s[0] >= since) and (until is None or s[0] <= until)]
        if not rows:
            return dict(available=True, samples=0)
        busy = [s[1] for s in rows]
        ordered = sorted(busy)
        out = dict(available=True, period_seconds=self.period, samples=len(busy),
                   sm_busy_mean=statistics.fmean(busy), sm_busy_median=statistics.median(busy),
                   sm_busy_p10=ordered[len(ordered)//10], idle_share_below_10=sum(b < 10 for b in busy)/len(busy),
                   memory_controller_busy_mean=statistics.fmean(s[2] for s in rows),
                   peak_used_bytes=max(s[3] for s in rows),
                   note='NVML whole-device utilization rates (sampled), including any other process on this GPU.')
        if since is not None or until is not None:
            out['window_seconds'] = [since, until]
        if timeline:
            out['timeline'] = dict(columns=['seconds', 'sm_busy', 'memory_busy', 'used_bytes'], rows=rows)
        return out
