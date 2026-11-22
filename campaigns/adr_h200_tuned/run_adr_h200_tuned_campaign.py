#!/usr/bin/env python3
"""Plan, preflight or run the H200 tuned ADR campaign (``adr_h200_tuned``).

The unified runner's bounded subprocesses, resume, memory guard and summary.json, with the
ladder of ``adr_h200_tuned_plan`` and one worker process per system: it assembles on the GPU
with the vendored device route, exports the trace system once for the host PyPardiso
reference, and runs every solver candidate in place from the device arrays (no device -> host
-> device round trip; ``system_worker``). The raw-CUDA assembly timing row of each system is
its own job, as it needs no cache.

Fixed environment: the device route with a 4 GiB chunk workspace and 8 prefault threads
(cupy_adr_device records), the calibrated guard of adr_unified_worker as the only memory
guard, device checks and NVML sampling. The calibration must carry host fits
(calibrate_assembly_memory.py of this branch).

    run_adr_h200_tuned_campaign.py --output DIR                 # plan only (login node)
    run_cuda13.sh python run_adr_h200_tuned_campaign.py --output DIR --memory-calibration CAL --execute --resume
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaigns.adr_h200_tuned.adr_h200_tuned_plan import CLASSES, DEGREES, build_plan
from campaigns.adr_unified import run_adr_unified_campaign as runner
from campaigns.adr_unified.adr_unified_plan import fingerprint, read_inventory
from campaigns.adr_unified.adr_unified_worker import load_file
from campaigns.common.lifecycle import stop_on_signals
from campaigns.common.reporting import print_details, result_details

ENVIRONMENT = dict(HYBRIDGE_ADR_SINGLE_GUARD='1', HDGFEM_CUPY_ADR_ROUTE='device',
                   HDGFEM_CUPY_ADR_WORKSPACE_GIB='4', HDGFEM_CUPY_ADR_PREFAULT_THREADS='8',
                   HYBRIDGE_ADR_DEVICE_CHECKS='1', HYBRIDGE_ADR_GPU_SAMPLING='1', HYBRIDGE_ADR_CACHE_IDENTITY='metadata')
RETRYABLE = runner.RETRYABLE
FINAL_CANDIDATE_STATUSES = ('passed', 'not_converged')


def parser():
    p = runner.parser()
    p.description = __doc__
    p.add_argument('--degrees', type=int, nargs='+', default=list(DEGREES))
    p.add_argument('--levels', nargs='+', help='Keep only these levels (L1-L5, L4b, L6-L9, Lmax); ids are unchanged')
    p.add_argument('--largest', action='store_true',
                   help='Add the Lmax level: per degree, the largest trace-DOF target whose system job fits one H200')
    p.add_argument('--classes', nargs='+', metavar='GEOMETRY:CASE',
                   help='Keep only these classes, e.g. square:trigonometric')
    p.add_argument('--mesh-only', action='store_true',
                   help='Run only the mesh jobs into --output (CPU node, ahead of the GPU lane) and exit')
    p.add_argument('--retry-mesh-errors', action='store_true',
                   help='Rerun mesh jobs that ended in error (passed meshes are reused)')
    p.set_defaults(assembly_backend='cupy', warm_deltas=[1e-4, 1e-3, 1e-2], memory_fraction=.9, reserve_gpu_gib=8.)
    return p


def system_budget(args, candidates):
    """The runner arguments with the system job's wall budget: assembly plus one ``--timeout`` per candidate."""
    return types.SimpleNamespace(**{**vars(args), 'timeout': args.timeout*(1+candidates)})


def pending_candidates(output, system_id, candidates):
    """Candidates without a final result (missing, or retryable) in ``jobs/<system>_<candidate>/result.json``."""
    pending = []
    for choice in candidates:
        final = output/'jobs'/f"{system_id}_{choice['id']}"/'result.json'
        try:
            status = json.loads(final.read_text()).get('status')
        except (OSError, ValueError):
            status = None
        if status is None or status in RETRYABLE:
            pending.append(choice['id'])
    return pending


def mesh_job(system, args, common, environment):
    """The system's mesh job (archived mesh check or gmsh generation), as the campaign loop runs it.

    ``--retry-mesh-errors`` reruns a mesh job that ended in ``error`` (a size search that a later
    revision fixed, for example); a passed mesh is always reused.
    """
    key = 'mesh_'+system['id']
    prior = args.output/'jobs'/key/'result.json'
    retry = bool(getattr(args, 'retry_mesh_errors', False)) and prior.exists() and common.read_json(prior).get('status') == 'error'
    return runner.run_job(key, dict(stage='mesh', system=system, branch_root=str(args.branch_root),
                                    inventory_directory=str(args.inventory.parent),
                                    mesh_directory=str(args.output/'meshes'/system['id'])),
                          args, common, environment, retry=retry)


def prepare_meshes(plan, args, common):
    """``--mesh-only``: run every mesh job of the plan (host gmsh work) and nothing else.

    Meant for a CPU node ahead of the GPU lane: the campaign's ``execute`` reuses a finished
    mesh job, so the H200 never waits for gmsh. No campaign lock, no preflight, no manifest;
    a concurrent campaign generating the same mesh produces the same deterministic file.
    """
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', NUMBA_NUM_THREADS=str(args.numba_threads))
    statuses = Counter()
    for system in sorted(plan['systems'], key=lambda s: (s.get('triangles') or s['target_triangles'], s['id'])):
        if not system['mesh']:
            continue
        result = mesh_job(deepcopy(system), args, common, environment)
        statuses[result['status']] += 1
        runner.event(f"mesh {system['id']} ({system['level']} {system['problem']['case']} p={system['problem']['p']}): "
                     f"{result['status']} triangles={result.get('triangles')} in {result.get('runner_wall_seconds', 0):.1f} s")
    runner.event(f'meshes done: {dict(statuses)}')
    return int(any(status != 'passed' for status in statuses))


def execute(plan, args, common):
    if plan['amgx_backend'] != args.amgx_backend:
        raise ValueError('Plan and requested AMGX backend differ')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HYBRIDGE_PRECISION='float64', HDGFEM_PRECISION='float64',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           NUMBA_NUM_THREADS=str(args.numba_threads), NUMBA_THREADING_LAYER='omp')
        probe = dict(stage='preflight', branch_root=str(args.branch_root), amgx_backend=args.amgx_backend,
                     fp64_overrides=dict(args.fp64_overrides),
                     amgx_configs=list({fingerprint(c['solver']['amgx_config']): c['solver']['amgx_config']
                                        for s in plan['systems'] for c in s['candidates']
                                        if 'amgx_config' in c['solver']}.values()))
        checked = runner.run_job('preflight', probe, args, common, environment, retry=True)
        if checked['status'] != 'passed':
            raise ValueError(f'Preflight failed: {checked.get("error", checked["status"])}; see {checked["log"]}')
        if args.preflight:
            return 0
        selected = checked['selected']
        environment['CUDA_VISIBLE_DEVICES'] = runner.mask_for_device(selected['device'], os.environ.get('CUDA_VISIBLE_DEVICES'))
        runner.event(f'selected {selected["name"]}, visible ordinal {selected["device"]}, '
                     f'FP64 peak estimate {selected["selection_fp64_flops"]/1e12:.3f} TFLOP/s; AMGX BSR backend={args.amgx_backend}')
        manifest = dict(plan=plan, source_sha256=runner.source_hashes(args.branch_root),
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
        calibration = getattr(args, 'calibration', None)
        for system in sorted(plan['systems'], key=lambda s: (s.get('triangles') or s['target_triangles'], s['id'])):
            system = deepcopy(system)
            shared = dict(system['problem'], **{k: v for k, v in plan['protocol'].items()
                                               if k not in ('initial_guess', 'precision')},
                          branch_root=str(args.branch_root), device=0, engine='gpu',
                          amgx_backend=args.amgx_backend, cache=str(args.output/'cache'/system['id']),
                          assembly_backend=args.assembly_backend, assembly_warmup=0, assembly_repeats=1,
                          memory_fraction=args.memory_fraction, reserve_gpu_gib=args.reserve_gpu_gib,
                          host_limit_gib=args.host_limit_gib, reference_max_dofs=0,
                          record_residual_history=True, mesh_path=None)
            if getattr(args, 'warm_deltas', None):
                shared.update(warm_deltas=args.warm_deltas, warm_steps=args.warm_steps)
            mesh_result = dict(status='passed')
            if system['mesh']:
                mesh_result = mesh_job(system, args, common, environment)
                if mesh_result['status'] == 'passed':
                    shared.update({k: mesh_result[k] for k in ('mesh_path', 'mesh_sha256', 'n')})
                    system.update(triangles=mesh_result['triangles'], trace_dofs=mesh_result['trace_dofs'])
            solvers = [c for c in system['candidates'] if c['solver']['family'] != 'raw_assembly']
            raw = [c for c in system['candidates'] if c['solver']['family'] == 'raw_assembly']
            system_result = dict(status='preparation_failed', preparation_status=mesh_result['status'])
            if mesh_result['status'] == 'passed':
                candidates = []
                for choice in solvers:
                    candidate = dict(choice['solver'], candidate=choice['id'],
                                     memory_estimate=runner.estimate_memory(system, choice, calibration))
                    candidates.append(candidate)
                spec = dict(shared, stage='system', system_id=system['id'], triangles=system['triangles'],
                            candidates=candidates, candidate_timeout=args.timeout, output=str(args.output),
                            memory_estimate=runner.estimate_memory(system, solvers[0], calibration))
                key = 'system_'+system['id']
                pending = pending_candidates(args.output, system['id'], solvers)
                prior_path = args.output/'jobs'/key/'result.json'
                prior_status = common.read_json(prior_path).get('status') if prior_path.exists() else None
                budget = system_budget(args, len(candidates))
                if prior_status == 'passed' and not pending:
                    system_result = runner.run_job(key, spec, budget, common, environment)
                else:
                    system_result = runner.run_job(key, spec, budget, common, environment,
                                                   retry=prior_status is not None and prior_status not in RETRYABLE)
                # The assembly record the host PyPardiso reference reads (jobs/assemble_<id>/result.json).
                assembly = system_result.get('assembly') or {}
                assemble_record = dict(assembly, status=assembly.get('status', system_result['status']),
                                       error=system_result.get('error'), system_job=key,
                                       specification_sha256=system_result.get('specification_sha256'))
                assemble_dir = args.output/'jobs'/('assemble_'+system['id'])
                assemble_dir.mkdir(parents=True, exist_ok=True)
                common.atomic_json(assemble_dir/'result.json', assemble_record)
            for choice in raw:
                solver = dict(choice['solver'], candidate=choice['id'])
                rspec = dict(shared, **solver, stage='solve')
                rspec['memory_estimate'] = runner.estimate_memory(system, choice, calibration)
                if mesh_result['status'] != 'passed':
                    result = dict(status='preparation_failed', preparation_status=mesh_result['status'])
                else:
                    result = runner.run_job(system['id']+'_'+choice['id'], rspec, args, common, environment)
                records.append(_record(result, system, choice, shared, args))
                common.atomic_json(args.output/'summary.json', records)
            for choice in solvers:
                key = system['id']+'_'+choice['id']
                final = args.output/'jobs'/key/'result.json'
                if system_result['status'] in ('preparation_failed',) or not final.exists():
                    result = dict(status='preparation_failed', preparation_status=system_result['status'],
                                  error=system_result.get('error'))
                else:
                    result = common.read_json(final)
                    if result.get('status') in RETRYABLE:
                        result = dict(result, status='process_error', error='system worker ended before this candidate')
                    print_details(key, result_details(result, dict(shared, **choice['solver'])),
                                  wall_seconds=result.get('worker_wall_seconds', 0), log_path=Path(result.get('log') or ''))
                    _print_diagnostics(result)
                records.append(_record(result, system, choice, shared, args))
                common.atomic_json(args.output/'summary.json', records)
            runner.event(f'finished={len(records)}/{plan["coverage"]["solver_jobs"]} '
                         f'statuses={dict(Counter(r["status"] for r in records))}')
        completion = dict(status='completed', scheduled=plan['coverage']['solver_jobs'], attempted=len(records),
                          statuses=dict(Counter(r['status'] for r in records)))
        common.atomic_json(args.output/'completion.json', completion)
        return int(any(r['status'] not in FINAL_CANDIDATE_STATUSES for r in records))


def _record(result, system, choice, shared, args):
    return dict(result, system_id=system['id'], case=shared['case'], p=shared['p'], level=system['level'],
                triangles=system.get('triangles'), trace_dofs=system.get('trace_dofs'), solver=choice,
                origins=system['origins'], amgx_backend=args.amgx_backend)


def _print_diagnostics(result):
    phases = {**result.get('worker_phases', {}), **result.get('phases', {})}
    if phases:
        print(f"  phases ({sum(phases.values()):.1f} of {result.get('worker_wall_seconds', 0):.1f} s): "
              + ', '.join(f'{k} {v:.2f}' for k, v in phases.items()), flush=True)
    usage = result.get('gpu_utilization') or {}
    if usage.get('samples'):
        print(f"  gpu: SM busy mean {usage['sm_busy_mean']:.0f}% median {usage['sm_busy_median']:.0f}% "
              f"p10 {usage['sm_busy_p10']}%, below 10% for {100*usage['idle_share_below_10']:.0f}% of "
              f"{usage['samples']} samples, peak {usage['peak_used_bytes']/2**30:.1f} GiB", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ('inventory', 'output', 'branch_root'):
        setattr(args, key, getattr(args, key).resolve())
    if args.assembly_backend != 'cupy':
        raise ValueError('Device campaign: GPU (CuPy device route) assembly only')
    args.calibration = json.loads(args.memory_calibration.read_text()) if args.memory_calibration else None
    if args.mesh_only and (args.execute or args.preflight):
        raise ValueError('--mesh-only runs alone (no --execute/--preflight)')
    if args.execute or args.preflight:
        model = (args.calibration or {}).get('model', {})
        if not model or any('host_bytes_per_triangle' not in entry for entry in model.values()):
            raise ValueError('Execution needs a calibration with host fits (calibrate_assembly_memory.py of this branch)')
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.heartbeat, args.reserve_gpu_gib)):
        raise ValueError('Timeout, heartbeat and GPU reserve must be positive and finite')
    args.fp64_overrides = [(int(k), float(v)*1e12) for k, v in (x.split('=') for x in args.fp64_overrides)]
    classes = tuple(tuple(c.split(':', 1)) for c in args.classes) if args.classes else CLASSES
    plan = build_plan(read_inventory(args.inventory), degrees=tuple(args.degrees), maxiter=args.maxiter,
                      repeats=args.repeats, warmup=args.warmup, amgx_backend=args.amgx_backend,
                      classes=classes, levels=args.levels, largest=args.largest)
    plan['environment'] = ENVIRONMENT
    plan['execution'] = 'one worker per system; candidates in process from device arrays'
    print(json.dumps(plan['coverage'], indent=2))
    if args.mesh_only:
        common = load_file('_adr_h200_tuned_mesh_common', args.branch_root/'scripts/adr_performance_common.py')
        args.resume = True
        with stop_on_signals():
            return prepare_meshes(plan, args, common)
    if not args.execute and not args.preflight:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output/'plan.json'
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(plan)):
            raise ValueError('Existing different plan; choose a fresh output directory')
        path.write_text(json.dumps(plan, indent=2)+'\n')
        return 0
    os.environ.update(ENVIRONMENT)
    common = load_file('_adr_h200_tuned_runner_common', args.branch_root/'scripts/adr_performance_common.py')
    with stop_on_signals():
        try:
            return execute(plan, args, common)
        except (runner.CampaignStop, runner.DeviceNotReleased) as error:
            reason = error.signal_name if isinstance(error, runner.CampaignStop) else 'device_not_released'
            runner.event(f'stopped ({reason}); rerun with --resume to continue')
            return 2 if reason in runner.REQUEUE_SIGNALS else 3


if __name__ == '__main__':
    raise SystemExit(main())
