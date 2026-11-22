"""ASM+PP, BJ+PP and AMGX timed on one cached ADR system, with every tensor operation on the GPU.

Same protocol and record fields as the vendored ``adr_solver_comparison_worker.measure`` (one
warm-up and the repeats; per setup two zero-guess solves of the same RHS, then the warm series
over the RHS perturbations; ``samples``/``warmups``, medians), so records stay comparable with
21958. What moved from the host to the device:

- solution checks (residual, local reconstruction, L2 error): ``device_verify.DeviceVerifier``;
- the warm series: perturbed RHS, guesses and residuals stay on the device;
- AMGX: the BSR matrix (valid blocks, rows sorted by column) is built with CuPy and handed to
  AMGX through hybridge's ``PyAMGXCsrDeviceSolver``, the one path for CuPy arrays into pyamgx
  (attached zero-copy with ``amgx_borrow_bsr`` and a pyamgx that has ``attach_BSR``, otherwise
  uploaded device-to-device, with the reason in ``borrow_note``); the RHS and solution move by
  device pointer inside the wrapper;
- the operator hash is computed once per job by the worker, not again here.

ASM+PP and BJ+PP use the hybridge-stack builder of ``tuned_face_dense`` (H200 kernels).

Every job records ``phases``: wall seconds of each phase (cache, uploads, verifier, per setup
the setup stages, solves, checks and warm series, final save), synchronized with the device,
and prints them as it goes.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
from pathlib import Path
import resource
from time import perf_counter

import numpy as np


class Phases:
    """Wall seconds per named phase; device-synchronized at both ends."""

    def __init__(self, cp, log):
        self.cp, self.log, self.totals, self.order = cp, log, {}, []

    @contextmanager
    def __call__(self, name, *, quiet=False):
        self.cp.cuda.Device().synchronize()
        started = perf_counter()
        try:
            yield
        finally:
            self.cp.cuda.Device().synchronize()
            elapsed = perf_counter()-started
            if name not in self.totals:
                self.order.append(name)
            self.totals[name] = self.totals.get(name, 0.)+elapsed
            if not quiet:
                self.log(f'[phase] {name}: {elapsed:.3f} s')

    def summary(self):
        return {name: self.totals[name] for name in self.order}


def amgx_bsr_on_device(cp, blocks, neighbors):
    """The vendored host BSR (valid blocks, indices sorted per row) built on the device."""
    rows, slots = neighbors.shape
    valid = neighbors >= 0
    key = cp.where(valid, neighbors, cp.iinfo(cp.int64).max)
    order = cp.argsort(key, axis=1)
    sorted_neighbors = cp.take_along_axis(neighbors, order, axis=1)
    sorted_valid = cp.take_along_axis(valid, order, axis=1)
    sorted_blocks = blocks[cp.arange(rows)[:, None], order]
    counts = sorted_valid.sum(axis=1)
    indptr = cp.concatenate((cp.zeros(1, dtype=cp.int64), cp.cumsum(counts))).astype(cp.int32)
    indices = cp.ascontiguousarray(sorted_neighbors[sorted_valid].astype(cp.int32))
    data = cp.ascontiguousarray(sorted_blocks[sorted_valid])
    return indptr, indices, data


def measure(spec, log=print, context=None):
    """``context`` (``device_system.SolveContext``): the system is already on the device, with its verifier."""
    import cupy as cp
    from scripts.adr_performance_common import atomic_json
    from scripts.adr_performance_worker import load_cache, metadata
    from scripts.adv_diff_rea_cases import get_case
    from scripts.warm_series import direction
    from campaigns.adr_h200_tuned.device_verify import DeviceVerifier, FaceDenseMatvec

    phases = Phases(cp, log)
    if context is None:
        with phases('cache_open'):
            system, get = load_cache(spec['cache'])
            rhs_host = np.array(system.rhs.reshape(-1), copy=True)
    else:
        system, get = context.system, None
    with phases('cuda_context'):
        cp.zeros(1).sum().item()
    tuned = bool(spec.get('tuned_face_dense'))
    native = spec['family'] != 'amgx'
    if not native:
        # One hand-off path for CuPy arrays into AMGX: hybridge's wrapper attaches (CSR or BSR)
        # or uploads through pyamgx. Nothing here calls pyamgx directly.
        from hybridge.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver, amgx_device_memory_stats
        from hybridge.linalg.gpu.sparse import _DeviceBsrMatrixView
    result = {'status': 'running', 'family': spec['family'], 'candidate': spec['candidate'],
              'case': spec['case'], 'n': spec['n'], 'p': spec['p'], 'rtol': spec['rtol'],
              'internal_rtol': spec.get('internal_rtol', spec['rtol']),
              'environment': metadata({**spec, 'engine': 'gpu'}), 'samples': [], 'warmups': [],
              'requested_repeats': spec['repeats'], 'solves_per_setup': spec['solves_per_setup'],
              'matrix_cache': str(spec['cache']), 'matrix_format': spec.get('matrix_format', 'face_dense'),
              'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'measure': 'adr_h200_tuned.device_measure', 'operator_sha256': spec.get('verified_operator_sha256'),
              'configuration': spec.get('configuration', spec.get('amgx_config'))}
    _, exact = get_case(spec['case'])
    free, total = cp.cuda.runtime.memGetInfo()
    if context is None:
        with phases('verifier_setup'):
            verifier = DeviceVerifier.from_cache(cp, spec, system, get, exact, budget_bytes=int(0.2*total), log=log)
        with phases('device_upload'):
            rhs = cp.asarray(rhs_host)
            if not native:
                # One device copy of the blocks: AMGX's BSR is built from it and the checks reuse it.
                blocks = cp.asarray(np.asarray(system.blocks))
                neighbors = cp.asarray(np.asarray(system.neighbors))
                verifier.own_matvec = FaceDenseMatvec(cp, system, blocks=blocks)
    else:
        verifier = context.verifier
        rhs = cp.ascontiguousarray(system.rhs.reshape(-1))
        blocks, neighbors = system.blocks, system.neighbors
    result.update(verifier.info)
    warm_unit = None
    if spec.get('warm_deltas'):
        with phases('warm_rhs_direction'):
            # As scripts/warm_series.steps: the unit direction scaled by ||b|| (host RNG, uploaded once).
            warm_unit = cp.asarray(direction(int(rhs.size), spec.get('warm_seed', 261027))*verifier.rhs_norm)
    if context is not None:
        from campaigns.adr_h200_tuned.tuned_face_dense import build_gpu_device
        build = lambda s: build_gpu_device(s, context)
    elif tuned:
        from campaigns.adr_h200_tuned.tuned_face_dense import build_gpu
        build = lambda s: build_gpu(s, system, get('element_blocks'), get('loc2glob'))
    else:
        from scripts.adr_performance_worker import build_gpu
        build = lambda s: build_gpu(s, system, get('element_blocks'), get('loc2glob'))
    block_size = system.block_size if spec.get('matrix_format') == 'bsr' else 1
    x = None

    def save():
        result['phases'] = phases.summary()
        atomic_json(spec['result'], result)

    try:
        for repetition in range(spec['warmup']+spec['repeats']):
            label = f"{'warmup' if repetition < spec['warmup'] else 'sample'} {repetition}"
            solver = op = base = pre = amgx = matrix = None
            matvec = None
            try:
                cp.cuda.Device().synchronize()
                start = perf_counter()
                if native:
                    solver, op, base, pre, rhs_dev, stages = build({**spec, 'rtol': spec.get('internal_rtol', spec['rtol'])})
                    matvec = op.matvec if hasattr(op, 'matvec') else None
                else:
                    config = deepcopy(spec['amgx_config'])
                    cfgsolver = config['solver']
                    cfgsolver.update(convergence='ABSOLUTE', tolerance=float(spec.get('internal_rtol', spec.get('amgx_internal_rtol', spec['rtol']))*verifier.rhs_norm),
                                     max_iters=spec['maxiter'], monitor_residual=1, norm='L2', use_scalar_norm=1,
                                     print_solve_stats=0, obtain_timings=0)
                    if cfgsolver['solver'] in ('FGMRES', 'GMRES'):
                        cfgsolver['gmres_n_restart'] = spec.get('restart', 75)
                    cfgsolver['store_res_history'] = 1 if spec.get('record_residual_history') else 0
                    result['effective_amgx_config'] = config
                    result['amgx_internal_rtol'] = spec.get('internal_rtol', spec.get('amgx_internal_rtol', spec['rtol']))
                    stamp = perf_counter()
                    if block_size > 1:
                        indptr, indices, data = amgx_bsr_on_device(cp, blocks, neighbors)
                    else:
                        raise ValueError('device AMGX hand-off implemented for BSR matrices only')
                    cp.cuda.Device().synchronize()
                    conversion = perf_counter()-stamp
                    matrix = _DeviceBsrMatrixView(data=data, indices=indices, indptr=indptr,
                                                  shape=(system.num_rows*block_size,)*2, block_size=block_size)
                    # borrow_csr: AMGX aliases the device BSR (zero copy) when pyamgx has attach_BSR
                    # and the block rows are canonical; otherwise the wrapper uploads device-to-device
                    # and records why in borrow_note.
                    amgx = PyAMGXCsrDeviceSolver(config=config, tolerance=cfgsolver['tolerance'], maxiter=spec['maxiter'],
                                                 borrow_csr=bool(spec.get('amgx_borrow_bsr')), residual_history_limit=None)
                    amgx.setup(matrix)
                    cp.cuda.Device().synchronize()
                    stages = {'csr_conversion_ms': 1000*conversion,
                              'matrix_upload_ms': 1000*amgx.last_matrix_upload_elapsed_seconds,
                              'hierarchy_setup_ms': 1000*amgx.last_solver_setup_elapsed_seconds, 'bsr_built_on_device': True,
                              'bsr_borrowed': bool(amgx.csr_borrowed), 'borrow_note': amgx.borrow_note}
                    result['effective_amgx_config'] = amgx.config_dict
                    if not amgx.csr_borrowed:
                        matrix = None  # AMGX holds its own copy; release ours
                        del indptr, indices, data
                cp.cuda.Device().synchronize()
                setup = 1000*(perf_counter()-start)
                phases.totals[f'setup'] = phases.totals.get('setup', 0.)+setup/1000
                if 'setup' not in phases.order:
                    phases.order.append('setup')
                log(f'[{label}] setup {setup:.1f} ms ' + ' '.join(f'{k}={v:.1f}' for k, v in stages.items() if isinstance(v, float)))
                solves = []
                for solve_index in range(spec['solves_per_setup']):
                    if native:
                        before = (pre.matvec_count, pre.base_preconditioner_count)
                    cp.cuda.Device().synchronize(); stamp = perf_counter()
                    if native:
                        answer = solver.solve(rhs_dev, x0=None, raise_on_failure=False)
                    else:
                        x, info = amgx.solve(rhs)
                    cp.cuda.Device().synchronize(); elapsed = 1000*(perf_counter()-stamp)
                    phases.totals['solve'] = phases.totals.get('solve', 0.)+elapsed/1000
                    if 'solve' not in phases.order:
                        phases.order.append('solve')
                    if native:
                        x = answer.solution
                        converged = answer.converged
                        base_applications = int(pre.base_preconditioner_count-before[1])
                        counts = {'iterations': int(answer.iterations), 'outer_matvec_count': int(answer.matvec_count),
                                  'polynomial_applications': int(answer.preconditioner_count),
                                  'polynomial_inner_matvecs': int(pre.matvec_count-before[0]),
                                  'base_preconditioner_applications': base_applications,
                                  'total_solve_matvecs': int(answer.matvec_count+pre.matvec_count-before[0]),
                                  'status': str(answer.status)}
                        family_counter = {'asm_pp': 'asm_applications', 'bj_pp': 'block_jacobi_applications',
                                          'pp': 'unbased_polynomial_applications'}.get(spec['family'])
                        if family_counter is not None:
                            counts[family_counter] = base_applications
                    else:
                        converged = info['amgx_status'] == 'success'
                        counts = {'iterations': int(info['amgx_iterations']), 'status': info['amgx_status'],
                                  'outer_matvec_count': None, 'preconditioner_applications': None,
                                  'counter_note': 'AMGX public bindings do not expose exact application counts; '
                                                  'solve_ms includes the RHS/solution hand-off by device pointer.'}
                    with phases('verify', quiet=True):
                        checked = verifier.verify(x, solver_converged=converged, matvec=matvec)
                    if spec.get('record_residual_history'):
                        if native:
                            counts['residual_history'] = answer.true_residual_history.tolist()
                            counts['estimated_preconditioned_residual_history'] = answer.estimated_preconditioned_residual_history.tolist()
                            counts['residual_history_kind'] = 'absolute true residuals at restart boundaries'
                        else:
                            counts['residual_history'] = [float(v) for v in info['residual_history']]
                            counts['residual_history_kind'] = 'AMGX reported absolute L2 residuals; final residual checked independently'
                    solves.append({'solve_ms': elapsed, **counts, **checked})
                    log(f"[{label}] solve {solve_index+1}: {elapsed:.1f} ms, {counts['iterations']} it, "
                        f"relres {checked['true_relative_residual']:.3e}, L2 {checked['l2_error']:.3e}, "
                        f"check {checked['validation_reconstruction_ms']:.1f} ms, {'passed' if checked['passed'] else 'FAILED'}")
                    if not checked['passed']:
                        break
                warm, warm_error = [], None
                try:
                    if spec.get('warm_deltas') and len(solves) == spec['solves_per_setup'] and all(s['passed'] for s in solves):
                        with phases('warm_series', quiet=True):
                            for delta in spec['warm_deltas']:
                                current = rhs.copy()
                                guess = x.copy()
                                for step in range(1, spec.get('warm_steps', 4)+1):
                                    current = current+delta*warm_unit
                                    initial = verifier.relative_residual(guess, current, matvec)
                                    if native:
                                        cp.cuda.Device().synchronize(); stamp = perf_counter()
                                        answer = solver.solve(current, x0=guess, raise_on_failure=False)
                                        cp.cuda.Device().synchronize(); elapsed = 1000*(perf_counter()-stamp)
                                        new, iterations, converged = answer.solution, int(answer.iterations), bool(answer.converged)
                                    else:
                                        cp.cuda.Device().synchronize(); stamp = perf_counter()
                                        new, winfo = amgx.solve(current, initial_guess=guess)
                                        cp.cuda.Device().synchronize(); elapsed = 1000*(perf_counter()-stamp)
                                        iterations, converged = int(winfo['amgx_iterations']), winfo['amgx_status'] == 'success'
                                    rel = verifier.relative_residual(new, current, matvec)
                                    warm.append({'delta': delta, 'step': step, 'iterations': iterations, 'solve_ms': elapsed,
                                                 'initial_relative_residual': initial, 'true_relative_residual': rel,
                                                 'passed': bool(converged and np.isfinite(rel) and rel <= spec['rtol'])})
                                    log(f'[{label}] warm delta={delta:g} step {step}: {elapsed:.1f} ms, {iterations} it, relres {rel:.3e}')
                                    if bool(cp.all(cp.isfinite(new))):
                                        guess = new.copy()
                except Exception as error:
                    warm_error = f'{type(error).__name__}: {error}'
                    log(f'[{label}] warm series error: {warm_error}')
                sample = {'setup_ms': setup, 'setup_stages': stages, 'solves': solves, 'warm': warm, 'warm_error': warm_error,
                          'fresh_setup_solve_ms': setup+solves[0]['solve_ms'],
                          'amortized_ms': (setup+sum(s['solve_ms'] for s in solves))/len(solves),
                          'passed': len(solves) == spec['solves_per_setup'] and all(s['passed'] for s in solves),
                          'cupy_pool_used_bytes': int(cp.get_default_memory_pool().used_bytes()),
                          'cupy_pool_reserved_bytes': int(cp.get_default_memory_pool().total_bytes())}
                if not native:
                    sample['amgx_memory'] = amgx_device_memory_stats()
                (result['warmups'] if repetition < spec['warmup'] else result['samples']).append(sample)
                save()
                if not sample['passed']:
                    result['status'] = 'numerical_failure'
                    return result
            finally:
                if amgx is not None:
                    amgx.close(suppress_errors=True)
                solver = op = base = pre = amgx = matrix = None
                if 'answer' in locals():
                    del answer
                gc.collect()
        with phases('save_solution'):
            np.save(Path(spec['result']).with_suffix('.solution.npy'), cp.asnumpy(x), allow_pickle=False)
        result['status'] = 'passed'
        for key in ('setup_ms', 'fresh_setup_solve_ms', 'amortized_ms'):
            result[key.replace('_ms', '_median_ms')] = float(np.median([r[key] for r in result['samples']]))
        result['hot_solve_median_ms'] = float(np.median([s['solve_ms'] for r in result['samples'] for s in r['solves'][1:]]))
        result['worst_true_relative_residual'] = max(s['true_relative_residual'] for r in result['samples'] for s in r['solves'])
        result['l2_error'] = max(s['l2_error'] for r in result['samples'] for s in r['solves'])
        result['peak_process_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['memory_note'] = ('CuPy counters are post-solve pool snapshots; AMGX peaks are allocator-managed process peaks. '
                                 'Whole-device peak comparison requires separate profiling.')
        save()
        return result
    finally:
        result['phases'] = phases.summary()
        log('[phases] ' + ', '.join(f'{k} {v:.2f} s' for k, v in result['phases'].items()))
