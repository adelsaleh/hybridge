#!/usr/bin/env python3
"""Plan, preflight or explicitly run the portable ADR L1–L5 iterative campaign."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaigns.adr_unified.adr_unified_plan import (
    AMGX_BACKENDS, build_plan, fingerprint, memory_estimate, read_inventory,
)
from campaigns.adr_unified.adr_unified_worker import load_file
from campaigns.common.reporting import convergence_status, print_details, result_details
from campaigns.common.lifecycle import (
    REQUEUE_SIGNALS, CampaignStop, describe_release, release_device, stop_on_signals, stop_signal_name, tail,
    terminate_group,
)

GMRES_ROOT = ROOT/'vendor/adr_gmres'
WORKER = Path(__file__).with_name('adr_unified_worker.py')
RETRYABLE = {'running', 'interrupted', 'process_error'}


class DeviceNotReleased(RuntimeError):
    """A finished worker left a process on the GPU; later timings would be contaminated."""


def event(message):
    print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)


def mask_for_device(ordinal, inherited):
    if ordinal < 0:
        raise ValueError('CUDA ordinal must be nonnegative')
    if inherited is None:
        return str(ordinal)
    tokens = inherited.split(',')
    if ordinal < 0 or ordinal >= len(tokens) or not tokens[ordinal].strip():
        raise ValueError('Selected ordinal is outside inherited CUDA_VISIBLE_DEVICES')
    return tokens[ordinal].strip()


def cache_files(cache):
    return {p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(Path(cache).glob('*.npy'))}


CHECKOUT_KEYS = ('branch_root', 'inventory_directory')


def job_identity(specification):
    """Hash of a job spec without its checkout paths, so a frozen copy resumes another copy's
    output; the content checks stay (mesh_sha256, cache files, worker_sha256 in the records)."""
    return fingerprint({k: v for k, v in specification.items() if k not in CHECKOUT_KEYS})


def specification_matches(prior, specification, directory):
    """Whether a finished record belongs to this job.

    Accepts the current identity, the identity that still covered the checkout paths (records
    before 2026-10-28), and a record whose own attempt spec equals this job up to the checkout
    paths (built by another checkout, as the prefetched meshes were)."""
    stored = prior.get('specification_sha256')
    if stored in (job_identity(specification), fingerprint(specification)):
        return True
    attempts = sorted(directory.glob('attempt_*/spec.json'))
    if not attempts:
        return False
    try:
        own = json.loads(attempts[-1].read_text())
    except (OSError, ValueError):
        return False
    own.pop('result', None)
    return stored == fingerprint(own) and job_identity(own) == job_identity(specification)


def run_job(key, specification, args, common, environment, *, retry=False, command=None):
    """Serial bounded subprocess with durable attempt logs and exact-spec resume."""
    directory = args.output/'jobs'/key
    directory.mkdir(parents=True, exist_ok=True)
    identity = job_identity(specification)
    final = directory/'result.json'
    if final.exists():
        prior = common.read_json(final)
        if not specification_matches(prior, specification, directory):
            raise ValueError(f'Changed specification for {key}')
        if args.resume and prior.get('status') not in RETRYABLE and not retry:
            if specification['stage'] == 'assemble' and prior.get('status') == 'passed':
                if prior.get('cache_files') != cache_files(specification['cache']):
                    raise ValueError(f'Assembly cache changed or is incomplete: {key}')
            if specification['stage'] == 'mesh' and prior.get('status') == 'passed':
                mesh_path = Path(prior['mesh_path'])
                if not mesh_path.is_file() or hashlib.sha256(mesh_path.read_bytes()).hexdigest() != prior['mesh_sha256']:
                    raise ValueError(f'Prepared mesh changed or is missing: {key}')
            event(f'reuse {key}: {prior["status"]}')
            return prior
        if not args.resume and not retry:
            raise ValueError(f'Existing job {key}; use --resume')
    attempt = 1
    while (directory/f'attempt_{attempt}').exists():
        attempt += 1
    work = directory/f'attempt_{attempt}'
    work.mkdir()
    spec = dict(specification, result=str(work/'result.json'))
    common.atomic_json(work/'spec.json', spec)
    common.atomic_json(final, dict(status='running', specification_sha256=identity, attempt=attempt))
    argv = command or [sys.executable, '-B', '-u', str(WORKER), '--spec', str(work/'spec.json')]
    started = last = time.monotonic()
    process = None
    status = None
    observed_rss_kib = 0
    stopped = None
    event(f'start {key} attempt={attempt} timeout={args.timeout:g}s log={work/"worker.log"}')
    try:
        with (work/'worker.log').open('w') as stream:
            process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                try:
                    for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                        if line.startswith(('VmRSS:', 'VmHWM:')):
                            observed_rss_kib = max(observed_rss_kib, int(line.split()[1]))
                except (OSError, ValueError):
                    pass
                elapsed = time.monotonic()-started
                if elapsed > args.timeout:
                    terminate_group(process)
                    status = 'timeout'
                    break
                if time.monotonic()-last >= args.heartbeat:
                    progress = ''
                    try:
                        current = common.read_json(work/'result.json')
                        trials = current.get('samples') or current.get('warmups') or []
                        if trials and trials[-1].get('solves'):
                            solve = trials[-1]['solves'][-1]
                            progress = f" iterations={solve.get('iterations')} relres={solve.get('true_relative_residual')}"
                    except (OSError, ValueError):
                        pass
                    event(f'running {key} wall={elapsed:.1f}s{progress}')
                    last = time.monotonic()
                time.sleep(.25)
    except KeyboardInterrupt as error:
        stopped = getattr(error, 'signal_name', 'SIGINT')
        status = 'interrupted'
    except OSError as exc:
        status = 'process_error'
        event(f'{key}: {exc}')
    finally:
        # Also sweeps grandchildren of workers that exited on their own.
        worker_exit = terminate_group(process) if process else None
    device = release_device()
    try:
        result = common.read_json(work/'result.json')
    except (OSError, ValueError):
        result = dict(status='process_error', error='No valid worker result')
    returncode = process.returncode if process else None
    if not status and stop_signal_name(returncode):
        status, result['stop_signal'] = 'interrupted', stop_signal_name(returncode)
    if status:
        result['status'] = status
    elif result.get('status') == 'running' or (returncode and result.get('status') == 'passed'):
        result['status'] = 'process_error'
    # A stalled iterative solve is a measured outcome, not a broken job.
    result['status'] = convergence_status(result, specification)
    result.update(specification_sha256=identity, attempt=attempt, returncode=returncode,
                  runner_wall_seconds=time.monotonic()-started, log=str(work/'worker.log'),
                  runner_observed_peak_rss_kib=observed_rss_kib, worker_exit=worker_exit, device=device)
    if stopped:
        result['stop_signal'] = stopped
    if specification['stage'] == 'assemble' and result.get('status') == 'passed':
        result['cache_files'] = cache_files(specification['cache'])
    common.atomic_json(final, result)
    event(f'finish {key}: {result.get("status")} wall={result["runner_wall_seconds"]:.3f}s; {describe_release(device)}')
    # Iterations, residuals and setup/solve timings per job, so a live campaign can be evaluated as it runs.
    print_details(key, result_details(result, specification), wall_seconds=result['runner_wall_seconds'],
                  log_path=work/'worker.log')
    # Timing diagnostics of device campaigns: where the wall time went, and how busy the GPU was.
    phases = {**result.get('worker_phases', {}), **result.get('phases', {})}
    if phases:
        timed = sum(result.get('phases', {}).values()) or sum(result.get('worker_phases', {}).values())
        print(f"  phases ({timed:.1f} of {result.get('worker_wall_seconds', 0):.1f} s worker wall): "
              + ', '.join(f'{k} {v:.2f}' for k, v in phases.items()), flush=True)
    usage = result.get('gpu_utilization') or {}
    if usage.get('samples'):
        print(f"  gpu: SM busy mean {usage['sm_busy_mean']:.0f}% median {usage['sm_busy_median']:.0f}% "
              f"p10 {usage['sm_busy_p10']}%, below 10% for {100*usage['idle_share_below_10']:.0f}% of "
              f"{usage['samples']} samples, peak {usage['peak_used_bytes']/2**30:.1f} GiB", flush=True)
    if result.get('status') not in ('passed', 'interrupted'):
        print(tail(work/'worker.log'), flush=True)
    if stopped:
        raise CampaignStop(stopped)
    if device.get('released') is False:
        raise DeviceNotReleased(f'{key}: {describe_release(device)}')
    return result


def source_hashes(branch):
    sources = {}
    for label, root in (('master', ROOT), ('branch', branch)):
        # The vendored snapshot keeps the pre-rename package name.
        package = 'hybridge' if label == 'master' else 'hdgfem'
        paths = list((root/package).rglob('*.py'))
        paths += list((root/'scripts').glob('*adr*.py'))
        paths += list((root/'scripts').glob('*adv_diff_rea*.py'))
        paths += list((root/'scripts/advection_diffusion_reaction').rglob('*.py'))
        for path in sorted(set(paths)):
            sources[f'{label}/{path.relative_to(root)}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    adapter = ROOT/'campaigns/adr_unified/native_hp_worker.py'
    sources['master/campaigns/adr_unified/native_hp_worker.py'] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    return sources


def estimate_memory(system, choice, calibration):
    """Planning estimate, with GPU assembly bytes replaced by this device's measured fit when given."""
    estimate = memory_estimate(system, choice)
    model = (calibration or {}).get('model', {}).get(str(system['problem']['p']))
    if model:
        triangles = system.get('triangles') or math.ceil(1.03*system['target_triangles'])
        measured = model['bytes_per_triangle']*triangles + model['intercept_bytes']
        # 15% margin over calibrate_assembly_memory.py; solver bounds keep the planning model.
        estimate.update(planning_assembly_device_bytes=estimate['assembly_device_bytes'],
                        assembly_device_bytes=math.ceil(1.15*measured), assembly_device_source='calibrated')
        if 'host_bytes_per_triangle' in model:
            # Measured host peak replaces the 6x planning model (1.7x high on L4 in 21958). Solves
            # memory-map the cache, so the local inverses are clean, reclaimable page cache there.
            host = model['host_bytes_per_triangle']*triangles + model['host_intercept_bytes']
            p = system['problem']['p']
            local = (p+1)*(p+2)//2
            estimate.update(planning_assembly_host_bytes=estimate['assembly_host_bytes'],
                            assembly_host_bytes=math.ceil(1.15*host), assembly_host_source='calibrated',
                            planning_solver_host_bytes=estimate['solver_host_bytes'],
                            solver_host_bytes=max(0, estimate['solver_host_bytes']-triangles*(3*local)**2*8),
                            solver_host_source='planning without the memory-mapped local inverses')
    return estimate


def execute(plan, args, common):
    if plan['amgx_backend'] != args.amgx_backend:
        raise ValueError('Plan and requested AMGX backend differ')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HYBRIDGE_PRECISION='float64', HDGFEM_PRECISION='float64',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           NUMBA_NUM_THREADS=str(args.numba_threads), NUMBA_THREADING_LAYER='omp')
        if os.environ.get('NUMBA_DISABLE_JIT') == '1':
            raise ValueError('Execution requires JIT enabled; plan/tests do not')
        probe = dict(stage='preflight', branch_root=str(args.branch_root), amgx_backend=args.amgx_backend,
                     fp64_overrides=dict(args.fp64_overrides),
                     amgx_configs=list({fingerprint(c['solver']['amgx_config']): c['solver']['amgx_config']
                                        for s in plan['systems'] for c in s['candidates']
                                        if 'amgx_config' in c['solver']}.values()))
        checked = run_job('preflight', probe, args, common, environment, retry=True)
        if checked['status'] != 'passed':
            raise ValueError(f'Preflight failed: {checked.get("error", checked["status"])}; see {checked["log"]}')
        if args.preflight:
            return 0
        selected = checked['selected']
        environment['CUDA_VISIBLE_DEVICES'] = mask_for_device(selected['device'], os.environ.get('CUDA_VISIBLE_DEVICES'))
        event(f'selected {selected["name"]}, visible ordinal {selected["device"]}, '
              f'FP64 peak estimate {selected["selection_fp64_flops"]/1e12:.3f} TFLOP/s; '
              f'AMGX BSR backend={args.amgx_backend}')
        manifest = dict(plan=plan, source_sha256=source_hashes(args.branch_root),
                        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                  if k not in ('execute', 'preflight', 'resume', 'heartbeat')},
                        hardware=dict(name=selected['name'], total_bytes=selected['total_bytes'],
                                      cuda_runtime=checked['cuda_runtime'], cupy=checked['cupy'],
                                      library_sha256=checked['library_sha256']))
        path = args.output/'manifest.json'
        if path.exists():
            if not args.resume or common.read_json(path) != json.loads(json.dumps(manifest)):
                raise ValueError('Resume requires identical plan, settings, source and GPU model/memory/runtime')
        common.atomic_json(path, manifest)
        records = []
        for system in sorted(plan['systems'], key=lambda s: (s.get('triangles') or s['target_triangles'], s['id'])):
            system = deepcopy(system)
            shared = dict(system['problem'], **{k: v for k, v in plan['protocol'].items()
                                               if k not in ('initial_guess', 'precision')},
                          branch_root=str(args.branch_root), device=0, engine='gpu',
                          amgx_backend=args.amgx_backend,
                          cache=str(args.output/'cache'/system['id']),
                          assembly_backend=args.assembly_backend, assembly_warmup=0, assembly_repeats=1,
                          memory_fraction=args.memory_fraction, reserve_gpu_gib=args.reserve_gpu_gib,
                          host_limit_gib=args.host_limit_gib, reference_max_dofs=0,
                          record_residual_history=True, mesh_path=None)
            if getattr(args, 'warm_deltas', None):
                # Warm-started series per setup (time stepping): previous solution as the guess.
                shared.update(warm_deltas=args.warm_deltas, warm_steps=args.warm_steps)
            mesh_result = dict(status='passed')
            if system['mesh']:
                definition = system['mesh']
                # The mesh result includes degree-specific trace DOFs. Do not
                # reuse a differently fingerprinted preparation job at another p.
                mesh_id = system['id']
                mesh_result = run_job('mesh_'+mesh_id, dict(stage='mesh', system=system,
                    branch_root=str(args.branch_root), inventory_directory=str(args.inventory.parent),
                    mesh_directory=str(args.output/'meshes'/mesh_id)), args, common, environment)
                if mesh_result['status'] == 'passed':
                    shared.update({k: mesh_result[k] for k in ('mesh_path', 'mesh_sha256', 'n')})
                    system.update(triangles=mesh_result['triangles'], trace_dofs=mesh_result['trace_dofs'])
            shared['memory_estimate'] = estimate_memory(system, system['candidates'][0], getattr(args, 'calibration', None))
            assembled = mesh_result
            if mesh_result['status'] == 'passed':
                assembled = run_job('assemble_'+system['id'], dict(shared, stage='assemble'), args, common, environment)
            choices = deepcopy(system['candidates'])
            random.Random(system['id']).shuffle(choices)
            for choice in choices:
                key = system['id']+'_'+choice['id']
                if assembled['status'] != 'passed':
                    result = dict(status='preparation_failed', preparation_status=assembled['status'])
                else:
                    solver = dict(choice['solver'], candidate=choice['id'])
                    spec = dict(shared, **solver, stage='solve',
                                expected_operator_sha256=assembled['operator_sha256'])
                    if os.environ.get('HYBRIDGE_ADR_CACHE_IDENTITY') == 'metadata':
                        # Lets workers check the cache by array size and mtime instead of re-hashing it.
                        spec['expected_cache_files'] = assembled.get('cache_files')
                    spec['memory_estimate'] = estimate_memory(system, choice, getattr(args, 'calibration', None))
                    spec['assembly_backend'] = assembled.get('assembly_backend', shared['assembly_backend'])
                    result = run_job(key, spec, args, common, environment)
                records.append(dict(result, system_id=system['id'], case=shared['case'], p=shared['p'],
                                    level=system['level'], triangles=system.get('triangles'), trace_dofs=system.get('trace_dofs'),
                                    solver=choice, origins=system['origins'], amgx_backend=args.amgx_backend))
                common.atomic_json(args.output/'summary.json', records)
                event(f'finished={len(records)}/{plan["coverage"]["solver_jobs"]} '
                      f'statuses={dict(Counter(r["status"] for r in records))}')
        completion = dict(status='completed', scheduled=plan['coverage']['solver_jobs'], attempted=len(records),
                          statuses=dict(Counter(r['status'] for r in records)))
        common.atomic_json(args.output/'completion.json', completion)
        # Non-convergence is expected for weak preconditioners; it is reported, never a failure.
        return int(any(r['status'] not in ('passed', 'not_converged') for r in records))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, default=ROOT/'run_configs/adr_unified_l5/inventory.json')
    p.add_argument('--branch-root', type=Path, default=GMRES_ROOT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--l5-triangles', type=int, default=400000)
    p.add_argument('--maxiter', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--assembly-backend', choices=('auto', 'cupy', 'numba'), default='auto')
    # Element-local assembly/reconstruction is embarrassingly parallel: use every allocated core.
    p.add_argument('--numba-threads', type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument('--memory-fraction', type=float, default=.8)
    p.add_argument('--reserve-gpu-gib', type=float, default=1.)
    p.add_argument('--host-limit-gib', type=float)
    p.add_argument('--amgx-backend', choices=AMGX_BACKENDS, default='cusparse_generic',
                   help='Explicit AMGX BSR implementation; use legacy with a CUDA-12 V100 build')
    p.add_argument('--fp64-override', dest='fp64_overrides', action='append', default=[], metavar='ORDINAL=TFLOPS')
    p.add_argument('--timeout', type=float, default=7200.)
    p.add_argument('--heartbeat', type=float, default=30.)
    p.add_argument('--memory-calibration', type=Path,
                   help='calibration.json from calibrate_assembly_memory.py on this GPU model')
    p.add_argument('--warm-deltas', type=float, nargs='+',
                   help='Relative RHS change per step of warm-started series on each reused setup')
    p.add_argument('--warm-steps', type=int, default=4, help='Warm-started solves per series')
    p.add_argument('--hp-grid', action='store_true',
                   help='Add p=1-9 on every L1-L5 mesh of the representative classes')
    p.add_argument('--lu-levels', action='store_true',
                   help='Add L6/L7 (~8M/~20M trace DOFs) at p=2,6,9 for the representative classes')
    p.add_argument('--resume', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    return p


def main():
    args = parser().parse_args()
    for key in ('inventory', 'output', 'branch_root'):
        setattr(args, key, getattr(args, key).resolve())
    # The measured model is recorded in the manifest, so a resume must use the same calibration.
    args.calibration = json.loads(args.memory_calibration.read_text()) if args.memory_calibration else None
    if not 0 < args.memory_fraction < 1 or not 1 <= args.numba_threads <= len(os.sched_getaffinity(0)):
        raise ValueError('Invalid memory fraction or Numba thread count outside CPU affinity')
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.heartbeat, args.reserve_gpu_gib)):
        raise ValueError('Timeout, heartbeat and GPU reserve must be positive and finite')
    if args.host_limit_gib is not None and (not math.isfinite(args.host_limit_gib) or args.host_limit_gib <= 0):
        raise ValueError('Host limit must be finite and positive')
    args.fp64_overrides = [(int(k), float(v)*1e12) for k, v in (x.split('=') for x in args.fp64_overrides)]
    inventory = read_inventory(args.inventory)
    plan = build_plan(inventory, l5_triangles=args.l5_triangles, maxiter=args.maxiter,
                      repeats=args.repeats, warmup=args.warmup, amgx_backend=args.amgx_backend,
                      hp_grid=args.hp_grid, lu_levels=args.lu_levels)
    print(f'AMGX BSR backend: {plan["amgx_backend"]}')
    print(json.dumps(plan['coverage'], indent=2))
    if not args.execute and not args.preflight:
        # A login-node plan needs neither CUDA nor the second worktree.
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output/'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Existing different plan; choose a fresh output directory')
        path.write_text(json.dumps(plan, indent=2)+'\n')
        return 0
    common = load_file('_adr_unified_runner_common', args.branch_root/'scripts/adr_performance_common.py')
    with stop_on_signals():
        try:
            return execute(plan, args, common)
        except (CampaignStop, DeviceNotReleased) as error:
            reason = error.signal_name if isinstance(error, CampaignStop) else 'device_not_released'
            records = common.read_json(args.output/'summary.json') if (args.output/'summary.json').exists() else []
            common.atomic_json(args.output/'completion.json', dict(
                status='interrupted' if isinstance(error, CampaignStop) else 'device_not_released', stop_reason=reason,
                error=str(error), scheduled=plan['coverage']['solver_jobs'], attempted=len(records),
                statuses=dict(Counter(r['status'] for r in records))))
            event(f'stopped ({reason}) after {len(records)}/{plan["coverage"]["solver_jobs"]} solver jobs; '
                  f'job results are saved under {args.output/"jobs"}; rerun with --resume to continue')
            return 2 if reason in REQUEUE_SIGNALS else 3


if __name__ == '__main__':
    raise SystemExit(main())
