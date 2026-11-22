"""Phase timing and device-memory profile of the CuPy ADR assembly routes.

Each (case, n, p, route) point runs in a fresh process: one first call (CUDA
context already created; includes CuPy kernel compilation or cache loading and
cuBLAS handle creation) followed by warm repeated calls on the same DGSpace.
Every phase timer in ``ADRAssembly.timings`` is bracketed by stream
synchronizations, so host and device phases do not overlap. Device memory is
reported twice per call: the CuPy memory-pool high-water mark (the pool is
emptied before each call and keeps freed blocks, so ``total_bytes`` after the
call is the most it held at once) and a 2 ms ``memGetInfo`` sample of the whole
device, which also counts cuBLAS workspaces outside the pool.

    python vendor/adr_gmres/scripts/profile_cupy_adr_assembly.py --n 128 --degrees 1 4 6 9 \
        --routes host device --output DIR
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

BRANCH = Path(__file__).resolve().parents[1]


def _measure(case, n, p, route, repeats, chunk, cprofile_path):
    sys.path.insert(0, str(BRANCH))
    os.chdir(BRANCH)
    import numpy as np
    import cupy as cp
    from scripts.adv_diff_rea_cases import get_case
    from scripts.adr_performance_worker import get_space
    from hdgfem.solvers.adv_diff_rea import assemble_advection_diffusion_reaction
    cp.cuda.Device(0).use()
    cp.zeros(1).sum()
    cp.cuda.Device(0).synchronize()
    pool = cp.get_default_memory_pool()
    lowest = [cp.cuda.runtime.memGetInfo()[0]]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.002):
            lowest[0] = min(lowest[0], cp.cuda.runtime.memGetInfo()[0])

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    started = time.perf_counter()
    space = get_space(dict(n=n, p=p))
    space_ms = 1000*(time.perf_counter()-started)
    kw, _ = get_case(case)
    calls = []
    try:
        for i in range(1+repeats):
            gc.collect()
            pool.free_all_blocks()
            cp.cuda.Device(0).synchronize()
            before = cp.cuda.runtime.memGetInfo()[0]
            lowest[0] = before
            profiler = None
            if cprofile_path and i == repeats:
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
            started = time.perf_counter()
            a = assemble_advection_diffusion_reaction(space=space, backend='cupy', cupy_route=route,
                                                      element_chunk=chunk, **kw)
            cp.cuda.Device(0).synchronize()
            wall = 1000*(time.perf_counter()-started)
            if profiler is not None:
                profiler.disable()
                import pstats
                with open(cprofile_path, 'w') as stream:
                    stats = pstats.Stats(profiler, stream=stream)
                    stats.sort_stats('tottime').print_stats(35)
                    stats.sort_stats('cumulative').print_stats(45)
            time.sleep(0.01)
            calls.append(dict(kind='first' if i == 0 else 'warm', wall_ms=wall, info=getattr(a, 'info', {}),
                              timings_ms={k: 1000*v for k, v in a.timings.items()},
                              pool_high_water_bytes=int(pool.total_bytes()),
                              sampled_device_peak_bytes=int(before-lowest[0])))
            dofs = int(a.system.num_dofs)
            del a
    finally:
        stop.set()
        watcher.join()
    return dict(case=case, n=n, p=p, triangles=2*n*n, route=route, element_chunk=chunk, trace_dofs=dofs,
                space_setup_ms=space_ms, calls=calls, gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                cupy=cp.__version__, numpy=np.__version__, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('HYBRIDGE_STUDY_OUT', '.')))
    parser.add_argument('--cases', nargs='+', default=['trigonometric'])
    parser.add_argument('--n', type=int, nargs='+', default=[128])
    parser.add_argument('--degrees', type=int, nargs='+', default=[1, 4, 6, 9])
    parser.add_argument('--routes', nargs='+', default=['host', 'device'])
    parser.add_argument('--chunks', type=int, nargs='+', default=[0], help='element chunk; 0 means automatic')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--cprofile-degrees', type=int, nargs='*', default=[], help='cProfile the last call (host route)')
    parser.add_argument('--prefault-threads', type=int, nargs='+', default=[8],
                        help='device route: HDGFEM_CUPY_ADR_PREFAULT_THREADS values to measure')
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--point', nargs=6, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.point:
        case, n, p, route, chunk, cprof = args.point
        print('RESULT '+json.dumps(_measure(case, int(n), int(p), route, args.repeats,
                                           int(chunk) or None, cprof if cprof != '-' else None)), flush=True)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'points.jsonl'
    failures = 0
    for case in args.cases:
        for n in args.n:
            for p in args.degrees:
                for route in args.routes:
                    for chunk, threads in [(c, t) for c in (args.chunks if route == 'device' else [0])
                                           for t in (args.prefault_threads if route == 'device' else [0])]:
                        cprof = (str(args.output/f'cprofile_{case}_n{n}_p{p}_{route}.txt')
                                 if p in args.cprofile_degrees else '-')
                        command = [sys.executable, '-u', '-B', __file__, '--repeats', str(args.repeats),
                                   '--point', case, str(n), str(p), route, str(chunk), cprof]
                        started = time.perf_counter()
                        try:
                            env = dict(os.environ, HDGFEM_CUPY_ADR_PREFAULT_THREADS=str(threads))
                            done = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, env=env)
                            lines = [l for l in done.stdout.splitlines() if l.startswith('RESULT ')]
                            if done.returncode == 0 and lines:
                                row = dict(json.loads(lines[-1][7:]), status='ok', prefault_threads=threads)
                            else:
                                tail = (done.stderr or done.stdout)[-3000:]
                                row = dict(case=case, n=n, p=p, route=route, element_chunk=chunk or None,
                                           status='out_of_memory' if 'OutOfMemory' in tail else 'error', error=tail)
                        except subprocess.TimeoutExpired:
                            row = dict(case=case, n=n, p=p, route=route, element_chunk=chunk or None, status='timeout')
                        row['process_seconds'] = time.perf_counter()-started
                        failures += row['status'] != 'ok'
                        with path.open('a') as stream:
                            stream.write(json.dumps(row)+'\n')
                        if row['status'] == 'ok':
                            warm = [c for c in row['calls'] if c['kind'] == 'warm']
                            med = sorted(c['wall_ms'] for c in warm)[len(warm)//2] if warm else float('nan')
                            print(f"{case} n={n} p={p} {route} chunk={chunk or 'auto'} prefault={threads}: first {row['calls'][0]['wall_ms']:.1f} ms, "
                                  f"warm {med:.1f} ms, pool peak {max(c['pool_high_water_bytes'] for c in row['calls'])/2**30:.2f} GiB, "
                                  f"sampled peak {max(c['sampled_device_peak_bytes'] for c in row['calls'])/2**30:.2f} GiB", flush=True)
                            for key, value in sorted(warm[-1]['timings_ms'].items() if warm else []):
                                print(f'    {key:34s} {value:10.2f}', flush=True)
                        else:
                            print(f"{case} n={n} p={p} {route}: {row['status']}\n{row.get('error', '')[-1500:]}", flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
