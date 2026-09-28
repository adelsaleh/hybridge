#!/usr/bin/env python3
"""Diagnose one cached ADR system using HDGFEM's existing PyPardiso backend.

No assembly, JIT, GPU, or campaign writes. Planning is the default; --execute
runs each thread-count/repetition in a fresh, monitored CPU process.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.advection_diffusion_reaction.campaigns.logging import event


def write_json(path, value):
    # Reuse campaign persistence without importing the branch's numerical code.
    from scripts.advection_diffusion_reaction.campaigns.stress.run_closed_loop_stress import load_common
    common = load_common(ROOT/'vendor/adr_gmres')
    common.atomic_json(path, value)


def physical_threads():
    allowed = os.sched_getaffinity(0)
    try:
        return len({tuple((Path(f'/sys/devices/system/cpu/cpu{i}/topology')/name).read_text().strip()
                          for name in ('physical_package_id', 'core_id')) for i in allowed})
    except OSError:
        return len(allowed)


def memory_kib(path, key):
    for line in Path(path).read_text().splitlines():
        if line.startswith(key+':'):
            return int(line.split()[1])
    return 0


def inspect_cache(spec_path, max_dofs):
    import numpy as np
    spec = json.loads(Path(spec_path).read_text())
    cache = Path(spec['cache'])
    load = lambda name: np.load(cache/(name+'.npy'), mmap_mode='r', allow_pickle=False)
    blocks, neighbors, rhs = (load('system_'+name) for name in ('blocks', 'neighbors', 'rhs'))
    if blocks.ndim != 4 or blocks.shape[2] != blocks.shape[3] or neighbors.shape != blocks.shape[:2]:
        raise ValueError('Invalid cached face-block layout')
    dofs = blocks.shape[0]*blocks.shape[2]
    if not 0 < dofs <= max_dofs or rhs.size != dofs:
        raise ValueError(f'Cached system has {dofs} DOFs; --max-dofs={max_dofs}')
    return spec, blocks, neighbors, rhs


def solve_cached(args, threads, output):
    import numpy as np
    import pypardiso
    from hdgfem.assembly.face_dense import face_dense_matvec
    from hdgfem.linalg.bsr import face_dense_to_bsr
    from hdgfem.linalg.system import clear_pypardiso_cache, solve_pypardiso_system

    report = dict(status='running', threads=threads, timings_ms={})
    destination = output/'result.json'

    def stage(name, **fields):
        report.update(fields)
        report['stage'] = name
        write_json(destination, report)
        event(output, name, message=f'{name}: threads={threads}', **fields)

    try:
        stage('load_cache')
        spec, blocks, neighbors, rhs = inspect_cache(args.spec, args.max_dofs)
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        digest = hashlib.sha256()
        for value in (blocks, rhs):
            digest.update(memoryview(value).cast('B'))
        operator_hash = digest.hexdigest()
        expected = spec.get('expected_operator_sha256')
        if not expected and Path(spec['result']).exists():
            expected = json.loads(Path(spec['result']).read_text()).get('operator_sha256')
        if expected and operator_hash != expected:
            raise ValueError('Cached operator/RHS differs from recorded campaign hash')
        stage('convert_to_csr', case=spec['case'], dofs=rhs.size, operator_sha256=operator_hash,
              campaign_hash_checked=bool(expected), spec=str(args.spec.resolve()),
              spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
              neighbors_sha256=hashlib.sha256(memoryview(neighbors).cast('B')).hexdigest())
        t = time.perf_counter()
        if np.iscomplexobj(blocks):
            raise ValueError('PARDISO diagnostic requires a real matrix')
        matrix = face_dense_to_bsr(blocks, neighbors).tocsr().astype(np.float64, copy=False)
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
            raise ValueError('Nonfinite matrix or RHS')
        if np.any(np.diff(matrix.indptr) == 0):
            raise ValueError('Singular matrix: empty scalar row')
        if max(matrix.nnz, rhs.size) >= np.iinfo(np.int32).max:
            raise ValueError('PyPardiso requires 32-bit sparse indices')
        report['timings_ms']['csr_conversion'] = 1000*(time.perf_counter()-t)
        getter = pypardiso.ps.libmkl.MKL_Get_Max_Threads
        getter.argtypes, getter.restype = [], ctypes.c_int
        actual = int(getter())
        if actual != threads:
            raise RuntimeError(f'MKL thread limit {actual} != requested {threads}')
        clear_pypardiso_cache()
        stage('factorization_started', nnz=matrix.nnz, mkl_max_threads=actual,
              matrix_type='real_nonsymmetric', pypardiso_version=importlib.metadata.version('pypardiso'),
              mkl_version=importlib.metadata.version('mkl'),
              thread_environment={k: os.environ.get(k) for k in
                                  ('MKL_NUM_THREADS', 'MKL_DYNAMIC', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
        # Pre-factor the very same process-wide instance used by the existing
        # solve_pypardiso_system wrapper, solely to separate setup/solve timing.
        t = time.perf_counter()
        pypardiso.ps.factorize(matrix)
        report['timings_ms']['analysis_factorization'] = 1000*(time.perf_counter()-t)
        stage('physical_solve_started', perturbed_pivots=int(pypardiso.ps.get_iparm(14)),
              pardiso_memory_kib={str(i): int(pypardiso.ps.get_iparm(i)) for i in (15, 16, 17)})
        result = solve_pypardiso_system(matrix, rhs, rtol=args.rtol, matrix_type='nonsymmetric')
        report['timings_ms']['solve'] = 1000*result.solve_elapsed_seconds
        report['iterative_refinement_steps'] = int(pypardiso.ps.get_iparm(7))
        report['physical_relative_residual'] = result.physical_relative_residual_norm
        # Check against original face storage as well as converted CSR storage.
        block_residual = face_dense_matvec(blocks, neighbors, result.x).reshape(-1)-rhs
        bn = float(np.linalg.norm(rhs))
        rn = float(np.linalg.norm(block_residual))
        report['face_relative_residual'] = rn/bn if bn else (0.0 if rn == 0 else float('inf'))
        denominator = float(np.max(np.asarray(abs(matrix).sum(axis=1))))*float(np.max(np.abs(result.x)))
        denominator += float(np.max(np.abs(rhs)))
        report['backward_error_inf'] = float(np.max(np.abs(block_residual)))/denominator if denominator else 0.0
        stage('planted_solution_check_started')
        known = np.random.default_rng(210921).standard_normal(rhs.size)
        probe_rhs = face_dense_matvec(blocks, neighbors, known).reshape(-1)
        probe = solve_pypardiso_system(matrix, probe_rhs, rtol=args.rtol, matrix_type='nonsymmetric')
        report['timings_ms']['probe_reused_solve'] = 1000*probe.solve_elapsed_seconds
        report['probe_relative_residual'] = probe.physical_relative_residual_norm
        report['probe_relative_solution_error'] = float(np.linalg.norm(probe.x-known)/np.linalg.norm(known))
        report['timings_ms']['setup_solve'] = report['timings_ms']['analysis_factorization']+report['timings_ms']['solve']
        passed = (result.converged and probe.converged and report['face_relative_residual'] <= args.rtol
                  and report['probe_relative_solution_error'] <= 1e-6)
        report['status'] = 'passed' if passed else 'numerical_failure'
        if passed:
            np.save(output/'eliminated_solution.npy', result.x, allow_pickle=False)
        stage('finished')
    except Exception as exc:
        report.update(status='error', error=str(exc), traceback=traceback.format_exc())
        write_json(destination, report)
        raise
    finally:
        clear_pypardiso_cache()
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


def monitor(command, env, output, args):
    start = last = time.monotonic()
    failure = None
    peak_rss = peak_hwm = 0.0
    minimum_available = float('inf')
    live_output = getattr(args, 'live_output', False)
    heartbeat_seconds = getattr(args, 'heartbeat_seconds', 30.)
    with (output/'worker.log').open('x') as log, (output/'worker.log').open(errors='replace') as reader:
        def forward_output():
            if live_output:
                text = reader.read()
                if text:
                    print(text, end='', flush=True)

        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                forward_output()
                now = time.monotonic()
                try:
                    rss = memory_kib(f'/proc/{process.pid}/status', 'VmRSS')/1024**2
                    free = memory_kib('/proc/meminfo', 'MemAvailable')/1024**2
                    hwm = memory_kib(f'/proc/{process.pid}/status', 'VmHWM')/1024**2
                    peak_rss, peak_hwm = max(peak_rss, rss), max(peak_hwm, hwm)
                    minimum_available = min(minimum_available, free)
                except FileNotFoundError:
                    continue
                if now-start > args.timeout:
                    failure = 'timeout'
                elif rss > args.max_rss_gib or free < args.reserve_gib:
                    failure = 'memory_budget_exceeded'
                if failure:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                if now-last >= heartbeat_seconds:
                    event(output, 'heartbeat', message=f"{getattr(args, 'monitor_label', 'PARDISO')} wall={now-start:.1f}s RSS={rss:.2f} GiB",
                          wall_seconds=now-start, rss_gib=rss, available_gib=free)
                    last = now
                time.sleep(getattr(args, 'poll_seconds', 0.5))
        except KeyboardInterrupt as error:
            failure = 'interrupted'
            event(output, 'interrupted', message=f'{output.name}: monitor interrupted',
                  error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            forward_output()
            if failure == 'interrupted':
                path = output/'result.json'
                result = json.loads(path.read_text()) if path.exists() else {}
                result.update(status='interrupted', returncode=process.returncode,
                              worker_wall_seconds=time.monotonic()-start)
                write_json(path, result)
    result_path = output/'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if failure or process.returncode != 0 and result.get('status') in (None, 'running', 'passed'):
        result.update(status=failure or 'error', error=failure or 'worker failed; inspect worker.log')
    result.update(returncode=process.returncode, worker_wall_seconds=time.monotonic()-start,
                  monitored_peak_rss_gib=peak_rss, monitored_peak_hwm_gib=peak_hwm,
                  minimum_available_gib=minimum_available if math.isfinite(minimum_available) else None,
                  memory_poll_seconds=getattr(args, 'poll_seconds', 0.5))
    write_json(result_path, result)
    details = ''
    if 'face_relative_residual' in result:
        details += f" relres={result['face_relative_residual']}"
    if 'setup_solve' in result.get('timings_ms', {}):
        details += f" setup_solve_ms={result['timings_ms']['setup_solve']}"
    print(f"{output.name}: {result['status']}{details} log={output/'worker.log'}", flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True, type=Path, help='Existing assembly or comparison specification')
    p.add_argument('--output', required=True, type=Path, help='New diagnostic directory, outside campaign cache')
    p.add_argument('--threads', nargs='+', type=int, default=None, help='Default: affinity-visible physical cores')
    p.add_argument('--repeats', type=int, default=1, help='Fresh factorizations per thread count')
    p.add_argument('--rtol', type=float, default=1e-10)
    p.add_argument('--max-dofs', type=int, default=600000, help='Safety gate for the original 50k-triangle case')
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--max-rss-gib', type=float, default=32)
    p.add_argument('--reserve-gib', type=float, default=8)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.threads = args.threads or [physical_threads()]
    if any(t < 1 or t > len(os.sched_getaffinity(0)) for t in args.threads) or len(set(args.threads)) != len(args.threads):
        raise ValueError('Thread counts must be distinct, positive, and within process affinity')
    if args.repeats < 1 or args.max_dofs < 1 or any(not math.isfinite(v) or v <= 0 for v in
                                                (args.rtol, args.timeout, args.max_rss_gib, args.reserve_gib)):
        raise ValueError('Counts, tolerances, and resource limits must be positive and finite')
    if args.worker:
        return solve_cached(args, args.threads[0], args.output)
    spec, blocks, _, _ = inspect_cache(args.spec, args.max_dofs)
    plan = dict(spec=str(args.spec.resolve()), cache=spec['cache'], dofs=blocks.shape[0]*blocks.shape[2],
                threads=args.threads, repeats=args.repeats, max_rss_gib=args.max_rss_gib,
                rtol=args.rtol, note='No reassembly; best measured thread count is not a universal optimum.')
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    if args.output.exists():
        raise FileExistsError('Use a new diagnostic output directory')
    if args.output.resolve().is_relative_to(Path(spec['cache']).resolve()):
        raise ValueError('Diagnostic output must not be inside the assembly cache')
    args.output.mkdir(parents=True)
    write_json(args.output/'plan.json', plan)
    rows = []
    for threads in args.threads:
        for repeat in range(args.repeats):
            output = args.output/f'threads_{threads}_repeat_{repeat+1}'
            output.mkdir()
            env = dict(os.environ, MKL_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                       MKL_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', NUMBA_DISABLE_JIT='1',
                       HDGFEM_PRECISION='float64', PYTHONDONTWRITEBYTECODE='1')
            command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--worker',
                       '--spec', str(args.spec.resolve()), '--output', str(output.resolve()),
                       '--threads', str(threads), '--rtol', str(args.rtol), '--max-dofs', str(args.max_dofs)]
            rows.append(monitor(command, env, output, args))
    medians = {t: statistics.median(r['timings_ms']['setup_solve'] for r in rows if r.get('threads') == t)
               for t in args.threads if sum(r.get('threads') == t and r.get('status') == 'passed'
                                           for r in rows) == args.repeats}
    summary = dict(results=rows, setup_solve_median_ms=medians,
                   best_measured_threads=min(medians, key=medians.get) if medians else None,
                   note='Run on an idle machine; one RHS/probe cannot prove nonsingularity or PDE accuracy.')
    write_json(args.output/'summary.json', summary)
    return 0 if all(r.get('status') == 'passed' for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
