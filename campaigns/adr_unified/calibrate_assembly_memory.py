"""Measure GPU (CuPy) ADR assembly memory and time on this device for p=1-9.

The runner's planning model (memory_estimate) was written for the 48 GB RTX PRO 5000 and was
never compared with measured peaks. This measures the vendored CuPy assembly that the
unified campaign uses, one fresh process per (n, p) point so peaks are not inherited, and
fits device and host (peak RSS) bytes = slope * triangles + intercept per degree. The runner applies the fits
with --memory-calibration instead of the planning formula. A degree's size ladder stops at
its first out-of-memory point, which is recorded as a result.

    run_cuda13.sh python campaigns/adr_unified/calibrate_assembly_memory.py --output DIR
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
BRANCH = ROOT/'vendor/adr_gmres'
SIZES = (64, 128, 223, 316, 447)


def _plan_module():
    # Pure planning code, loaded by path so the vendored ``scripts`` package keeps its name.
    spec = importlib.util.spec_from_file_location('_adr_unified_plan', ROOT/'campaigns/adr_unified/adr_unified_plan.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_point(case, n, p):
    """Assemble twice on the GPU (the first includes kernel compilation) and sample device memory."""
    sys.path.insert(0, str(BRANCH))
    os.chdir(BRANCH)
    import cupy as cp
    from scripts.adv_diff_rea_cases import get_case
    from scripts.adr_performance_worker import get_space
    from hdgfem.solvers.adv_diff_rea import assemble_advection_diffusion_reaction
    cp.cuda.Device(0).use()
    cp.zeros(1).sum()
    baseline, total = cp.cuda.runtime.memGetInfo()
    lowest = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.005):
            lowest[0] = min(lowest[0], cp.cuda.runtime.memGetInfo()[0])

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    walls = []
    try:
        space = get_space(dict(n=n, p=p))
        kw, _ = get_case(case)
        for _ in range(2):
            started = time.perf_counter()
            result = assemble_advection_diffusion_reaction(space=space, backend='cupy', **kw)
            cp.cuda.Device(0).synchronize()
            walls.append(time.perf_counter()-started)
            dofs = int(result.system.num_dofs)
            del result
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        stop.set()
        watcher.join()
    # One fresh process per point, so the high-water RSS is this point's host peak (interpreter,
    # CuPy and the host outputs the worker contract keeps: local inverses, coupling, blocks).
    return dict(trace_dofs=dofs, peak_device_bytes=int(baseline-lowest[0]), device_total_bytes=int(total),
                peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                first_wall_seconds=walls[0], warm_wall_seconds=walls[1])


def _line(ok, field):
    """Line through the two largest points; a zero-intercept ratio if that line is unusable."""
    top = ok[-1]
    slope, intercept = top[field]/top['triangles'], 0
    if len(ok) >= 2:
        low = ok[-2]
        line = (top[field]-low[field])/(top['triangles']-low['triangles'])
        if line > 0 and top[field] >= line*top['triangles']:
            slope, intercept = line, top[field]-line*top['triangles']
    return slope, intercept


def fit(rows):
    """Per degree: device bytes and, when recorded, host bytes as slope * triangles + intercept."""
    model = {}
    for p in sorted({row['p'] for row in rows}):
        ok = sorted((row for row in rows if row['p'] == p and row['status'] == 'ok'), key=lambda row: row['triangles'])
        if not ok:
            continue
        slope, intercept = _line(ok, 'peak_device_bytes')
        model[str(p)] = dict(bytes_per_triangle=slope, intercept_bytes=intercept,
                             largest_measured_triangles=ok[-1]['triangles'], points=len(ok))
        hosted = [row for row in ok if row.get('peak_host_bytes')]
        if hosted:
            slope, intercept = _line(hosted, 'peak_host_bytes')
            model[str(p)].update(host_bytes_per_triangle=slope, host_intercept_bytes=intercept)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', default='trigonometric')
    parser.add_argument('--degrees', type=int, nargs='+', default=list(range(1, 10)))
    parser.add_argument('--sizes', type=int, nargs='+', default=list(SIZES), help='square mesh n (2n^2 triangles)')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--point', nargs=2, type=int, metavar=('N', 'P'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.point:
        print(json.dumps(measure_point(args.case, *args.point)), flush=True)
        return 0
    if args.output is None:
        parser.error('--output is required')
    plan = _plan_module()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'points.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(row['case'], row['n'], row['p']) for row in rows}
    for p in args.degrees:
        for n in args.sizes:
            if any(row['p'] == p and row['status'] == 'out_of_memory' and row['n'] <= n for row in rows):
                break
            if (args.case, n, p) in done:
                continue
            system = dict(triangles=2*n*n, problem=dict(p=p), trace_dofs=None)
            planning = plan.memory_estimate(system, dict(solver=dict(family='pp')))['assembly_device_bytes']
            row = dict(case=args.case, n=n, p=p, triangles=2*n*n, planning_assembly_device_bytes=planning)
            started = time.perf_counter()
            try:
                process = subprocess.run([sys.executable, '-u', '-B', __file__, '--case', args.case, '--point', str(n), str(p)],
                                         capture_output=True, text=True, timeout=args.timeout)
                text = process.stdout + process.stderr
                if process.returncode == 0:
                    row.update(json.loads(process.stdout.strip().splitlines()[-1]), status='ok')
                    row['measured_over_planning'] = row['peak_device_bytes']/planning
                else:
                    row.update(status='out_of_memory' if 'out of memory' in text.lower() or 'OutOfMemory' in text else 'error',
                               returncode=process.returncode, error=text.strip().splitlines()[-1][:400] if text.strip() else '')
            except subprocess.TimeoutExpired:
                row.update(status='timeout')
            row['wall_seconds'] = time.perf_counter()-started
            rows.append(row)
            with path.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            gib = row.get('peak_device_bytes', 0)/2**30
            print(f'[{time.strftime("%H:%M:%S")}] n={n} p={p} triangles={2*n*n}: {row["status"]} '
                  f'peak={gib:.2f} GiB planning={planning/2**30:.2f} GiB warm={row.get("warm_wall_seconds", 0):.2f}s', flush=True)
    calibration = dict(device=os.environ.get('CUDA_VISIBLE_DEVICES'), case=args.case, model=fit(rows),
                       note='Device and host bytes ~ slope * triangles + intercept for vendored CuPy assembly; measured, not planned.')
    (args.output/'calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
    for p, entry in calibration['model'].items():
        print(f'p={p}: {entry["bytes_per_triangle"]/1024:.1f} KiB/triangle (+{entry["intercept_bytes"]/2**30:.2f} GiB), '
              f'measured up to {entry["largest_measured_triangles"]} triangles', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
