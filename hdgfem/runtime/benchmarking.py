"""Warmed, repeated wall/CPU measurements for bounded kernel benchmarks."""
import statistics
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
