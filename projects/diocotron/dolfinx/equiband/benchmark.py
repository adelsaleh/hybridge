"""Warm-kernel crossing benchmark; deliberately excludes JIT and PDE timings.

Run ``python -m projects.diocotron.dolfinx.equiband.benchmark --threads 1 4``. These timings are
not an end-to-end speed claim; use the CLI elapsed time for complete solves.
"""
if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "projects.diocotron.dolfinx.equiband"

import argparse
import json
import time
import numpy as np
from numba import set_num_threads
from .crossings import _crossings_numba, _crossings_numpy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=512)
    parser.add_argument("--segments", type=int, default=400)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)
    if min(args.rays, args.segments, args.repeats, *args.threads) <= 0:
        parser.error("sizes, repeats and thread counts must be positive")
    s = np.tile(np.arange(args.segments)/args.segments, args.rays)
    h = 1/args.segments
    q = np.column_stack((.2*(1-s*s), -.4*s*h, np.full(len(s), -.2*h*h)))
    data = (q, np.arange(args.rays+1)*args.segments, s, np.full(len(s), h), np.array([.13, .12, .11]), 1e-12)
    reference = _crossings_numpy(*data)
    results = []
    for name, threads, kernel in [("numpy", 1, _crossings_numpy)]+[("numba", n, _crossings_numba) for n in args.threads]:
        set_num_threads(threads)
        measured = kernel(*data)  # cold compilation/warm-up is not timed
        for actual, expected in zip(measured, reference):
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        timings = []
        for _ in range(args.repeats):
            begin = time.perf_counter()
            kernel(*data)
            timings.append(time.perf_counter()-begin)
        results.append({"backend": name, "threads": threads, "median_seconds": float(np.median(timings)),
                        "p95_seconds": float(np.quantile(timings, .95))})
    print(json.dumps({"rays": args.rays, "segments_per_ray": args.segments, "warm_crossing_timings": results}, indent=2))


if __name__ == "__main__":
    main()
