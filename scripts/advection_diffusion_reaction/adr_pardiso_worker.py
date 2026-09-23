"""One isolated CPU-only repeated LU benchmark; called by the campaign driver.

Reuses HDGFEM's face conversion, residual operator and PyPardiso solve wrapper.
No mesh/space reconstruction, assembly, JIT, or GPU import is needed explicitly.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.metadata
import os
from pathlib import Path
import resource
import time
import traceback

from scripts.advection_diffusion_reaction.adr_pardiso_inventory import digest, read, timing_metrics
from scripts.advection_diffusion_reaction.check_cached_adr_pardiso import memory_kib, write_json
from scripts.advection_diffusion_reaction.closed_loop_stress_logging import event


def process_memory():
    """Linux RSS and cumulative high-water mark, sampled outside solve timers."""
    return dict(rss_kib=memory_kib('/proc/self/status', 'VmRSS'),
                peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                available_kib=memory_kib('/proc/meminfo', 'MemAvailable'))


def validate_cache(system):
    import numpy as np
    cache = Path(system['cache'])
    for name, expected in system['cache_files'].items():
        st = (cache / name).stat()
        if dict(size=st.st_size, mtime_ns=st.st_mtime_ns) != expected:
            raise ValueError(f'Cache changed since planning: {cache/name}')
    if digest(cache/'system_neighbors.npy') != system['neighbors_file_sha256']:
        raise ValueError('Cached topology hash mismatch')
    blocks, neighbors, rhs = [np.load(cache/f'system_{name}.npy', mmap_mode='r', allow_pickle=False)
                              for name in ('blocks', 'neighbors', 'rhs')]
    if blocks.dtype != np.float64 or rhs.dtype != np.float64:
        raise ValueError('Benchmark requires original FP64 matrix and RHS')
    h = hashlib.sha256()
    for array in (blocks, rhs):
        h.update(memoryview(array).cast('B'))
    if h.hexdigest() != system['system_id']:
        raise ValueError('Matrix/RHS hash differs from archived iterative system')
    if rhs.size != system['trace_dofs']:
        raise ValueError('Trace DOF count changed')
    return blocks, neighbors, np.asarray(rhs).reshape(-1)


def run(spec_path, output):
    import numpy as np
    import pypardiso
    from hdgfem.assembly.face_dense import face_dense_matvec
    from hdgfem.linalg.bsr import face_dense_to_bsr
    from hdgfem.linalg.pardiso_diagnostics import pardiso_factor_statistics
    from hdgfem.linalg.system import clear_pypardiso_cache, solve_pypardiso_system

    spec = read(spec_path)
    system, protocol = spec['system'], spec['protocol']
    output = Path(output)
    report = dict(status='running', system_id=system['system_id'], threads=spec['threads'],
                  case=system['case'], trace_dofs=system['trace_dofs'], rtol=protocol['rtol'],
                  warmups=[], samples=[], requested_repeats=protocol['repeats'],
                  solves_per_setup=2, memory_snapshots=[], timings_ms={},
                  measurement_phase=spec.get('measurement_phase', 'measurement'),
                  selected_for=spec.get('selected_for', ['fresh', 'reused']),
                  selection_fingerprint=spec.get('selection_fingerprint'))

    def stage(name, **fields):
        memory = process_memory()
        report['stage'] = name
        report['memory_snapshots'].append(dict(stage=name, **memory, **fields))
        report['peak_process_rss_kib'] = memory['peak_rss_kib']
        write_json(output/'result.json', report)
        event(output, name, message=f"{name} threads={spec['threads']} RSS={memory['rss_kib']/1024**2:.3f} GiB", **memory, **fields)

    def checked_solve(matrix, rhs, blocks, neighbors):
        result = solve_pypardiso_system(matrix, rhs, rtol=protocol['rtol'], matrix_type='nonsymmetric')
        # This also catches accidental re-factorization in a reused solve.
        if pypardiso.ps.phase != 33:
            raise RuntimeError('PARDISO did not reuse the prefactored matrix in phase 33')
        residual = face_dense_matvec(blocks, neighbors, result.x).reshape(-1)-rhs
        rn, bn = float(np.linalg.norm(residual)), float(np.linalg.norm(rhs))
        rel = rn/bn if bn else (0.0 if rn == 0 else float('inf'))
        finite = bool(np.all(np.isfinite(result.x)))
        return result.x, dict(solve_ms=1000*result.solve_elapsed_seconds,
                              true_relative_residual=rel,
                              csr_relative_residual=result.physical_relative_residual_norm,
                              passed=bool(result.converged and finite and rel <= protocol['rtol']),
                              pardiso_phase=33, iterative_refinement_steps=int(pypardiso.ps.get_iparm(7)))

    try:
        stage('load_and_hash')
        blocks, neighbors, rhs = validate_cache(system)
        if rhs.size > protocol['max_dofs']:
            raise ValueError('DOF safety limit exceeded')
        getter = pypardiso.ps.libmkl.MKL_Get_Max_Threads
        getter.argtypes, getter.restype = [], ctypes.c_int
        actual = int(getter())
        if actual != spec['threads']:
            raise RuntimeError(f"MKL threads {actual} != requested {spec['threads']}")
        report.update(operator_sha256=system['system_id'], campaign_hash_checked=True,
                      mkl_max_threads=actual, matrix_type='real_nonsymmetric',
                      versions={k: importlib.metadata.version(k) for k in ('pypardiso', 'mkl', 'numpy', 'scipy')},
                      thread_environment={k: os.environ.get(k) for k in
                                          ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_DYNAMIC',
                                           'OPENBLAS_NUM_THREADS', 'MKL_DYNAMIC', 'NUMBA_DISABLE_JIT')})
        stage('cache_verified')
        matrix = solution = None
        for trial in range(protocol['warmup'] + protocol['repeats']):
            phase = 'warmups' if trial < protocol['warmup'] else 'samples'
            number = trial + 1 if phase == 'warmups' else trial - protocol['warmup'] + 1
            clear_pypardiso_cache()
            matrix = solution = None
            gc.collect()
            stage('conversion_started', trial=number, group=phase)
            start = time.perf_counter()
            matrix = face_dense_to_bsr(blocks, neighbors).tocsr()
            matrix.eliminate_zeros()
            matrix.sort_indices()
            conversion_ms = 1000*(time.perf_counter()-start)
            if matrix.dtype != np.float64 or not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
                raise ValueError('Invalid/nonfinite FP64 system')
            if np.any(np.diff(matrix.indptr) == 0):
                raise ValueError('Singular matrix: empty scalar row')
            if max(matrix.nnz, rhs.size) >= np.iinfo(np.int32).max:
                raise ValueError('PyPardiso requires 32-bit sparse indices')
            report['nnz'] = int(matrix.nnz)
            report['csr_storage_bytes'] = sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr))
            stage('factorization_started', trial=number, group=phase)
            pypardiso.ps.set_iparm(18, -1)  # request fresh factor-nnz reporting
            pypardiso.ps.set_iparm(60, 0)   # in-core only; no implicit disk spill
            start = time.perf_counter()
            pypardiso.ps.factorize(matrix)
            factor_ms = 1000*(time.perf_counter()-start)
            stats = pardiso_factor_statistics(pypardiso.ps, matrix_nnz=matrix.nnz)
            if not stats['in_core']:
                raise RuntimeError('Unexpected out-of-core PARDISO mode')
            sample = dict(setup_ms=conversion_ms+factor_ms, csr_conversion_ms=conversion_ms,
                          analysis_factorization_ms=factor_ms, factor_statistics=stats, solves=[])
            report[phase].append(sample)
            stage('factorization_finished', trial=number, group=phase, factor_statistics=stats)
            for index in range(2):
                stage('solve_started', trial=number, group=phase, solve=index+1)
                solution, observed = checked_solve(matrix, rhs, blocks, neighbors)
                sample['solves'].append(observed)
                stage('solve_validated', trial=number, group=phase, solve=index+1,
                      relative_residual=observed['true_relative_residual'])
                if not observed['passed']:
                    report['status'] = 'numerical_failure'
                    stage('finished')
                    return 1
            sample['fresh_setup_solve_ms'] = sample['setup_ms'] + sample['solves'][0]['solve_ms']
            sample['factor_solve_ms'] = factor_ms + sample['solves'][0]['solve_ms']
        stage('planted_solution_check')
        known = np.random.default_rng(210921).standard_normal(rhs.size)
        probe_rhs = face_dense_matvec(blocks, neighbors, known).reshape(-1)
        probe, observed = checked_solve(matrix, probe_rhs, blocks, neighbors)
        error = float(np.linalg.norm(probe-known) / np.linalg.norm(known))
        report['probe'] = dict(**observed, relative_solution_error=error, excluded_from_timings=True)
        # Do not present this as a proof of nonsingularity or a PDE error check.
        report['status'] = 'passed' if observed['passed'] and np.isfinite(error) and error <= 1e-6 else 'numerical_failure'
        if report['status'] == 'passed':
            report['metrics'] = timing_metrics(report)
            report['timings_ms']['setup_solve'] = report['metrics']['fresh_median_ms']
            report['face_relative_residual'] = report['metrics']['worst_relative_residual']
            if report['measurement_phase'] != 'tuning':
                np.save(output/'eliminated_solution.npy', solution, allow_pickle=False)
        stage('finished')
        return 0 if report['status'] == 'passed' else 1
    except Exception as exc:
        report.update(status='error', error=str(exc), traceback=traceback.format_exc())
        stage('failed')
        return 1
    finally:
        try:
            clear_pypardiso_cache()
        finally:
            stage('factors_released')
