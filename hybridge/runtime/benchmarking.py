"""Repeated measurements and cooperative device reservation for benchmarks."""
from contextlib import contextmanager
import gc
import os
from pathlib import Path
import statistics
import subprocess
import time


def measure(function, repeats, minimum_seconds):
    """Warm a phase, then collect fixed-count samples with adaptive batching."""
    function()
    start = time.perf_counter()
    function()
    elapsed = time.perf_counter() - start
    batch = max(1, min(100, int(minimum_seconds / max(elapsed, 1e-9))))
    walls, cpus = [], []
    for _ in range(repeats):
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        for _ in range(batch):
            function()
        walls.append((time.perf_counter() - wall_start) / batch)
        cpus.append((time.process_time() - cpu_start) / batch)
    return dict(median_seconds=statistics.median(walls), min_seconds=min(walls),
                max_seconds=max(walls), cpu_wall_ratio=sum(cpus) / sum(walls),
                samples_seconds=walls, batch=batch)


def measure_synchronized(function, *, synchronize, warmups=1, repeats=3, on_sample=None):
    """Time complete calls, releasing each preceding result outside the timer.

    This avoids retaining two large device results during an allocation-heavy
    measurement. Compilation and tuning belong in the unrecorded warmup calls.
    ``on_sample(index, elapsed)`` receives negative indices for warmup calls.
    """
    if warmups < 1 or repeats < 1:
        raise ValueError("at least one warmup and one measurement are required")
    result = None
    samples = []
    for index in range(-warmups, repeats):
        result = None
        gc.collect()
        synchronize()
        start = time.perf_counter()
        result = function()
        synchronize()
        elapsed = time.perf_counter() - start
        if index >= 0:
            samples.append(elapsed)
        if on_sample is not None:
            on_sample(index, elapsed)
    return result, dict(median_seconds=statistics.median(samples), samples_seconds=samples)


def gpu_compute_processes():
    """List NVIDIA compute processes without initializing a CUDA context.

    Probe failures propagate: inability to inspect the device is not idleness.
    The reservation covers all visible NVIDIA devices, matching campaign locks.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    )
    return [line.strip() for line in result.stdout.splitlines()
            if line.strip() and int(line.split(",", 1)[0]) != os.getpid()]


@contextmanager
def reserve_gpu(lock_path, *, poll_seconds=5.0, timeout_seconds=None, notify=None):
    """Hold a shared campaign flock, then wait for other CUDA compute jobs.

    Call before any CUDA initialization. Other cooperating processes must use
    the same lock file. An uncoordinated job can still start after the check.
    The timeout covers both waiting for the lock and waiting for idle devices.
    """
    import fcntl

    if poll_seconds <= 0 or (timeout_seconds is not None and timeout_seconds < 0):
        raise ValueError("poll interval must be positive and timeout nonnegative")
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    next_notice = started

    def wait(message):
        """Report bounded waiting and enforce the shared reservation deadline."""
        nonlocal next_notice
        now = time.monotonic()
        remaining = None if timeout_seconds is None else timeout_seconds - (now - started)
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f"GPU reservation timed out: {message}")
        if notify is not None and now >= next_notice:
            notify(message)
            next_notice = now + 10.0
        time.sleep(poll_seconds if remaining is None else min(poll_seconds, remaining))

    with lock_path.open("a") as lock:
        try:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    wait(f"Waiting for GPU reservation: {lock_path}")
            while active := gpu_compute_processes():
                wait("Waiting for CUDA compute processes: " + "; ".join(active))
            if notify is not None:
                notify(f"GPU reserved: {lock_path}; no other compute processes")
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
