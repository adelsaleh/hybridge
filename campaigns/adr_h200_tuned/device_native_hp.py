"""Native pMG-AMG (hp-MG on the symmetric part + GMRES) with every tensor operation on the GPU.

The protocol and record fields of campaigns/adr_unified/native_hp_worker.py, but:

- the face-BSR symmetric part, its row sort, the Bernstein-to-modal products and the diagonal
  search are built on the device (``BernsteinHpSymmetricPartPreconditioner.from_face_dense``),
  inside the timed setup as before, instead of SciPy and a Python loop over faces;
- solution checks and the warm series run on the device (``device_verify``); no DG space is
  rebuilt on the host;
- every phase is timed (``phases``) and logged.
"""
from __future__ import annotations

from dataclasses import fields
import gc
import hashlib
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
import traceback


def measure(spec, master_root, log=print, context=None):
    """``context`` (``device_system.SolveContext``): the system is already on the device, with its verifier."""
    from scripts.adv_diff_rea_cases import get_case
    from scripts.warm_series import direction
    _, exact = get_case(spec['case'])
    master_root = Path(master_root).resolve()
    if str(master_root) not in sys.path:
        sys.path.append(str(master_root))
    import numpy as np
    import cupy as cp
    from hybridge.linalg.face_dense import FaceDenseSystem
    from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
    from hybridge.linalg.gpu.production_gmres import CuPyProductionGMRESSolver, CuPyProductionGMRESOptions
    from hybridge.linalg.multigrid.krylov import BernsteinHpSymmetricPartPreconditioner
    from hybridge.linalg.multigrid.policy import face_hp_mg_preconditioner_parameters
    from campaigns.adr_h200_tuned.device_measure import Phases
    from campaigns.adr_h200_tuned.device_system import operator_from_device
    from campaigns.adr_h200_tuned.device_verify import DeviceVerifier
    cp.cuda.Device(spec.get('device', 0)).use()
    phases = Phases(cp, log)
    cache = Path(spec['cache'])
    if context is None:
        with phases('cache_open'):
            get = lambda name: np.load(cache/(name+'.npy'), mmap_mode='r', allow_pickle=False)
            system = FaceDenseSystem(**{f.name: ('eliminate' if f.name == 'mode' else get('system_'+f.name))
                                        for f in fields(FaceDenseSystem)})
            rhs_host = np.array(system.rhs.ravel(), copy=True)
            triangles = int(get('loc2glob').shape[0])
    else:
        system, triangles = context.system, int(context.loc2glob.shape[0])
    with phases('cuda_context'):
        cp.zeros(1).sum().item()
    output = {'status': 'running', 'family': 'native_hp', 'candidate': spec.get('candidate', 'native_hp_standard'),
              'case': spec['case'], 'n': spec['n'], 'p': spec['p'], 'rtol': spec['rtol'],
              'internal_rtol': spec.get('internal_rtol', spec['rtol']), 'samples': [], 'warmups': [],
              'note': 'Master native hp-BSR hierarchy on symmetric part, Bernstein coordinate transforms, GMRES on '
                      'unchanged full ADR operator; not native PCGF on ADR. Device-built hierarchy input and device checks.',
              'measure': 'adr_h200_tuned.device_native_hp', 'operator_sha256': spec.get('verified_operator_sha256')}
    output.update(policy=spec.get('policy', 'standard'),
                  configuration=face_hp_mg_preconditioner_parameters(spec.get('policy', 'standard'), overrides=spec.get('native_tuning')),
                  restart=spec.get('restart', 75), maxiter=spec['maxiter'],
                  hp_mg_kernels=spec.get('hp_mg_kernels') or os.environ.get('HYBRIDGE_HP_MG_KERNELS', 'legacy'),
                  operator_implementation=spec.get('operator_implementation', 'raw_fused'),
                  estimate_target=spec.get('estimate_target', 'cycle_relative'),
                  requested_repeats=spec['repeats'], requested_warmups=spec['warmup'],
                  solves_per_setup=spec['solves_per_setup'], trace_dofs=system.num_dofs, triangles=triangles)
    import platform
    name = cp.cuda.runtime.getDeviceProperties(spec.get('device', 0))['name']
    output['environment'] = dict(python=platform.python_version(), numpy=np.__version__, cupy=cp.__version__,
                                 cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
                                 gpu=name.decode() if isinstance(name, bytes) else name)
    output['worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output['adapter_sha256'] = hashlib.sha256((master_root/'hybridge/linalg/multigrid/krylov.py').read_bytes()).hexdigest()
    output['matrix_cache'], output['matrix_format'] = str(cache), 'bsr'
    total = cp.cuda.runtime.memGetInfo()[1]
    if context is None:
        with phases('verifier_setup'):
            verifier = DeviceVerifier.from_cache(cp, spec, system, get, exact, budget_bytes=int(0.2*total), log=log)
        with phases('device_upload'):
            # Host->device copies of the cached system for the hierarchy input; the timed setup
            # below builds from these, as the host constructor built from the host cache.
            blocks = cp.asarray(np.asarray(system.blocks))
            neighbors = cp.asarray(np.asarray(system.neighbors))
            rhs = cp.asarray(rhs_host)
        operator = lambda: CuPyFaceDenseOperator.from_system(
            system, implementation=spec.get('operator_implementation', 'raw_fused'), dtype=np.float64,
            device_id=spec.get('device', 0))
    else:
        verifier = context.verifier
        blocks, neighbors = system.blocks, system.neighbors
        rhs = cp.ascontiguousarray(system.rhs.reshape(-1))
        operator = lambda: operator_from_device(cp, system, implementation=spec.get('operator_implementation', 'raw_fused'),
                                                device_id=spec.get('device', 0))
    output.update(verifier.info)
    warm_unit = None
    if spec.get('warm_deltas'):
        with phases('warm_rhs_direction'):
            warm_unit = cp.asarray(direction(int(rhs.size), spec.get('warm_seed', 261027))*verifier.rhs_norm)

    def save():
        from scripts.adr_performance_common import atomic_json
        output['phases'] = phases.summary()
        atomic_json(spec['result'], output)

    sync = cp.cuda.Device().synchronize
    phase = 'setup'
    x = None
    try:
        for repetition in range(spec['warmup']+spec['repeats']):
            label = f"{'warmup' if repetition < spec['warmup'] else 'sample'} {repetition}"
            pre = solver = op = None
            try:
                phase = 'setup'
                stages = {}
                sync(); start = perf_counter()
                pre = BernsteinHpSymmetricPartPreconditioner.from_face_dense(
                    blocks, neighbors, degree=spec['p'], policy=spec.get('policy', 'standard'),
                    tuning=spec.get('native_tuning'), kernels=spec.get('hp_mg_kernels'))
                sync(); stages['hierarchy_setup_ms'] = 1000*(perf_counter()-start)
                stamp = perf_counter()
                op = operator()
                solver = CuPyProductionGMRESSolver(op, preconditioner=pre, options=CuPyProductionGMRESOptions(
                    restart=spec.get('restart', 75), max_iterations=spec['maxiter'], rtol=spec.get('internal_rtol', spec['rtol']),
                    atol=0., orthogonalization='cgs2', raise_on_failure=False, cgs2_fallback_threshold=None,
                    estimate_target=spec.get('estimate_target', 'cycle_relative')))
                sync(); stages['operator_and_gmres_setup_ms'] = 1000*(perf_counter()-stamp)
                setup = 1000*(perf_counter()-start)
                phases.totals['setup'] = phases.totals.get('setup', 0.)+setup/1000
                if 'setup' not in phases.order:
                    phases.order.append('setup')
                log(f'[{label}] setup {setup:.1f} ms ' + ' '.join(f'{k}={v:.1f}' for k, v in stages.items()))
                solves = []
                phase = 'solve'
                for solve_index in range(spec['solves_per_setup']):
                    sync(); start = perf_counter()
                    r = solver.solve(rhs, x0=None, raise_on_failure=False)
                    sync(); elapsed = 1000*(perf_counter()-start)
                    phases.totals['solve'] = phases.totals.get('solve', 0.)+elapsed/1000
                    if 'solve' not in phases.order:
                        phases.order.append('solve')
                    x = r.solution
                    with phases('verify', quiet=True):
                        checked = verifier.verify(x, solver_converged=r.converged, matvec=op.matvec, case_l2_bound=False)
                    if spec.get('record_residual_history'):
                        checked.update(residual_history=r.true_residual_history.tolist(),
                                       estimated_preconditioned_residual_history=r.estimated_preconditioned_residual_history.tolist(),
                                       residual_history_kind='absolute true residuals at restart boundaries')
                    solves.append(dict(solve_ms=elapsed, iterations=r.iterations, matvec_count=r.matvec_count,
                                       preconditioner_applications=r.preconditioner_count, **checked))
                    log(f"[{label}] solve {solve_index+1}: {elapsed:.1f} ms, {r.iterations} it, relres {checked['true_relative_residual']:.3e}, "
                        f"L2 {checked['l2_error']:.3e}, check {checked['validation_reconstruction_ms']:.1f} ms, "
                        f"{'passed' if checked['passed'] else 'FAILED'}")
                    if not checked['passed']:
                        break
                warm, warm_error = [], None
                try:
                    if spec.get('warm_deltas') and len(solves) == spec['solves_per_setup'] and all(s['passed'] for s in solves):
                        with phases('warm_series', quiet=True):
                            for delta in spec['warm_deltas']:
                                current, guess = rhs.copy(), x.copy()
                                for step in range(1, spec.get('warm_steps', 4)+1):
                                    current = current+delta*warm_unit
                                    initial = verifier.relative_residual(guess, current, op.matvec)
                                    sync(); start = perf_counter()
                                    r = solver.solve(current, x0=guess, raise_on_failure=False)
                                    sync(); elapsed = 1000*(perf_counter()-start)
                                    rel = verifier.relative_residual(r.solution, current, op.matvec)
                                    warm.append(dict(delta=delta, step=step, iterations=int(r.iterations), solve_ms=elapsed,
                                                     initial_relative_residual=initial, true_relative_residual=rel,
                                                     passed=bool(r.converged and np.isfinite(rel) and rel <= spec['rtol'])))
                                    log(f'[{label}] warm delta={delta:g} step {step}: {elapsed:.1f} ms, {int(r.iterations)} it, relres {rel:.3e}')
                                    if bool(cp.all(cp.isfinite(r.solution))):
                                        guess = r.solution.copy()
                except Exception as error:
                    warm_error = f'{type(error).__name__}: {error}'
                    log(f'[{label}] warm series error: {warm_error}')
                row = dict(setup_ms=setup, setup_stages=stages, solves=solves, warm=warm, warm_error=warm_error,
                           fresh_setup_solve_ms=setup+solves[0]['solve_ms'],
                           amortized_ms=(setup+sum(s['solve_ms'] for s in solves))/len(solves))
                if 'pyamgx' in sys.modules:
                    row['amgx_memory'] = sys.modules['pyamgx'].get_device_memory_stats()
                (output['warmups'] if repetition < spec['warmup'] else output['samples']).append(row)
                save()
                if not all(s['passed'] for s in solves):
                    output['status'] = 'numerical_failure'
                    save()
                    return output
            finally:
                if pre is not None:
                    pre.close()
                solver = op = pre = None
                if 'r' in locals():
                    del r
                gc.collect()
        for key in ('setup_ms', 'fresh_setup_solve_ms', 'amortized_ms'):
            output[key.replace('_ms', '_median_ms')] = float(np.median([r[key] for r in output['samples']]))
        hot = [s['solve_ms'] for r in output['samples'] for s in r['solves'][1:]]
        output.update(hot_solve_median_ms=float(np.median(hot)), hot_solve_mean_ms=float(np.mean(hot)),
                      hot_solve_min_ms=float(min(hot)), hot_solve_max_ms=float(max(hot)))
        output['worst_true_relative_residual'] = max(s['true_relative_residual'] for r in output['samples']+output['warmups']
                                                     for s in r['solves'])
        with phases('save_solution'):
            np.save(Path(spec['result']).with_suffix('.solution.npy'), cp.asnumpy(x), allow_pickle=False)
        output['peak_process_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        output['status'] = 'passed'
    except Exception as exc:
        output.update(status='setup_error' if phase == 'setup' else 'solve_error', error=str(exc), traceback=traceback.format_exc())
    output['phases'] = phases.summary()
    log('[phases] ' + ', '.join(f'{k} {v:.2f} s' for k, v in output['phases'].items()))
    save()
    return output
