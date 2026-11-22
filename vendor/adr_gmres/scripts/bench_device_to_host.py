"""Device-to-host copy strategies for chunked assembly outputs (diagnostic).

Copies ``--total-gib`` of float64 from the device to a fresh host array in
``--chunk-gib`` pieces, as the device ADR route returns its local inverses:

* pageable: ``ndarray.get(out=host_slice)`` into a fresh ``np.empty`` array;
* pretouched: the same into a host array whose pages were already written;
* pinned: async copy into one of two reusable pinned buffers on a side stream,
  then a thread pool copies the pinned buffer into the fresh pageable array
  while the next chunk transfers. Pinned allocation is timed separately.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-gib', type=float, default=7.)
    parser.add_argument('--chunk-gib', type=float, default=1.4)
    parser.add_argument('--threads', type=int, nargs='+', default=[4, 8, 16])
    args = parser.parse_args()
    import cupy as cp
    import cupyx
    chunk = int(args.chunk_gib*2**30)//8
    count = int(args.total_gib*2**30)//8//chunk
    total = chunk*count
    device = cp.random.random(chunk)
    cp.cuda.Device().synchronize()
    results = {}

    def timed(name, fn):
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        results[name] = time.perf_counter()-t
        print(f'{name:28s} {1000*results[name]:9.1f} ms  {total*8/2**30/results[name]:6.1f} GiB/s', flush=True)

    def pageable(out):
        for i in range(count):
            device.get(out=out[i*chunk:(i+1)*chunk])

    timed('pageable_fresh', lambda: pageable(np.empty(total)))
    touched = np.ones(total)
    timed('pageable_pretouched', lambda: pageable(touched))
    del touched
    t = time.perf_counter()
    pinned = [cupyx.empty_pinned(chunk) for _ in range(2)]
    results['pinned_alloc_2_buffers'] = time.perf_counter()-t
    print(f"pinned alloc {2*chunk*8/2**30:.2f} GiB: {1000*results['pinned_alloc_2_buffers']:.1f} ms", flush=True)
    stream = cp.cuda.Stream(non_blocking=True)
    for threads in args.threads:
        pool = ThreadPoolExecutor(threads)

        def staged(out):
            pending = []
            for i in range(count):
                buf = pinned[i % 2]
                if len(pending) >= 2:
                    for f in pending.pop(0):
                        f.result()
                done = cp.cuda.Event()
                cp.cuda.Device().synchronize()  # chunk i computed on the main stream
                with stream:
                    device.get(out=buf, stream=stream, blocking=False)
                    done.record(stream)
                done.synchronize()
                base = i*chunk
                parts = np.array_split(np.arange(chunk), threads)
                pending.append([pool.submit(np.copyto, out[base+p[0]:base+p[-1]+1], buf[p[0]:p[-1]+1]) for p in parts])
            for group in pending:
                for f in group:
                    f.result()

        timed(f'pinned_staged_{threads}_threads', lambda: staged(np.empty(total)))
        pool.shutdown()
    print(json.dumps({k: 1000*v for k, v in results.items()}))


if __name__ == '__main__':
    main()
