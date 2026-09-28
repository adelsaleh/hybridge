#!/usr/bin/env python3
"""Retest every archived ADR report system with repeated CPU PyPardiso solves.

Default invocation is a read-only inventory. --execute writes only a separate
output directory and starts sequential, memory/timeout-guarded CPU workers.
It never rebuilds, reassembles, reruns iterative solvers, or edits the reports.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.advection_diffusion_reaction.campaigns.pardiso import adr_pardiso_inventory as archive
from scripts.advection_diffusion_reaction.campaigns.pardiso import adr_pardiso_tuning as tuning
from scripts.advection_diffusion_reaction.diagnostics.check_cached_adr_pardiso import (
    memory_kib, monitor, physical_threads, write_json,
)
from scripts.advection_diffusion_reaction.campaigns.logging import event

DEFAULT_OUTPUT = archive.STUDY / 'pardiso_all_2026_09_22'
SOURCE_FILES = (
    Path(__file__), Path(archive.__file__), Path(tuning.__file__),
    ROOT/'scripts/advection_diffusion_reaction/campaigns/pardiso/adr_pardiso_worker.py',
    ROOT/'scripts/advection_diffusion_reaction/diagnostics/check_cached_adr_pardiso.py',
    ROOT/'hdgfem/linalg/pardiso_diagnostics.py', ROOT/'hdgfem/linalg/system.py',
    ROOT/'hdgfem/linalg/bsr.py', ROOT/'hdgfem/assembly/face_dense.py', ROOT/'hdgfem/precision.py',
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--native-campaign', type=Path, default=archive.NATIVE,
                   help='Completed 101-system coverage registry (not a solver choice)')
    p.add_argument('--stress-campaign', type=Path, default=archive.STRESS,
                   help='Completed annular stress archive; both mesh sizes are included')
    p.add_argument('--threads', nargs='+', type=int, default=None,
                   help='MKL counts; with autotuning, default a ladder from 1 to physical cores')
    p.add_argument('--autotune-threads', action='store_true',
                   help='Per-system pilot sweep, then independent fresh/reused winner confirmations')
    p.add_argument('--tuning-repeats', type=int, default=3, help='Measured pilot setups per count')
    p.add_argument('--tuning-seed', type=int, default=220922, help='Reproducible pilot-order shuffle')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--rtol', type=float, default=1e-10)
    p.add_argument('--max-dofs', type=int, default=2000000)
    p.add_argument('--max-rss-gib', type=float, default=64)
    p.add_argument('--reserve-gib', type=float, default=16)
    p.add_argument('--timeout', type=float, default=7200, help='Seconds per system/thread worker')
    p.add_argument('--poll-seconds', type=float, default=0.2)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-failed', action='store_true', help='With --resume; retain earlier attempts')
    p.add_argument('--status', action='store_true', help='Read saved summary without launching work')
    p.add_argument('--worker-spec', type=Path, help=argparse.SUPPRESS)
    return p


def validate(args):
    args.threads = args.threads or (tuning.default_candidates(physical_threads()) if args.autotune_threads else [physical_threads()])
    if len(set(args.threads)) != len(args.threads) or any(t < 1 or t > len(os.sched_getaffinity(0)) for t in args.threads):
        raise ValueError('Thread counts must be distinct, positive, and within process affinity')
    if args.autotune_threads and (args.tuning_repeats < 2 or args.repeats < 3):
        raise ValueError('Thread tuning requires >=2 pilot setups and >=3 independent confirmation setups')
    if args.repeats < 1 or args.warmup < 1 or args.max_dofs < 1:
        raise ValueError('At least one warmup/setup and positive DOF limit required')
    if any(not math.isfinite(x) or x <= 0 for x in (args.rtol, args.max_rss_gib, args.reserve_gib, args.timeout, args.poll_seconds)):
        raise ValueError('Tolerances, memory limits and timeouts must be finite and positive')
    if args.rtol > 1e-10:
        raise ValueError('Do not relax the report acceptance tolerance of 1e-10')
    if args.retry_failed and not args.resume:
        raise ValueError('--retry-failed requires --resume')


def make_plan(args):
    systems, artifacts = archive.inventory(args.native_campaign, args.stress_campaign)
    too_large = [s['key'] for s in systems if s['trace_dofs'] > args.max_dofs]
    if too_large:
        raise ValueError(f'--max-dofs excludes report systems: {too_large}; increase explicitly')
    output = args.output.resolve()
    protected = {Path(s['cache']).resolve() for s in systems}
    protected.update((ROOT/'docs', args.native_campaign.resolve(), args.stress_campaign.resolve()))
    protected.update(Path(o['campaign_root']).resolve() for s in systems for o in s['origins'] if o.get('campaign_root'))
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must be separate from reports, source campaigns, and caches')
    protocol = {k: getattr(args, k) for k in ('repeats', 'warmup', 'rtol', 'max_dofs', 'max_rss_gib', 'reserve_gib', 'timeout', 'poll_seconds')}
    plan = dict(version=2, systems=systems, coverage=archive.coverage(systems),
                source_artifacts=artifacts, numerical_sources={str(p.resolve()): archive.digest(p) for p in SOURCE_FILES},
                threads=args.threads, protocol=protocol, output=str(output),
                thread_tuning=dict(enabled=args.autotune_threads, repeats=args.tuning_repeats,
                                   seed=args.tuning_seed,
                                   selection='separate fresh/reused pilot winners; confirmation only for comparisons'),
                scheduled=len(systems)*(len(args.threads) + min(2, len(args.threads)) if args.autotune_threads else len(args.threads)),
                scheduled_is_upper_bound=args.autotune_threads,
                scheduled_tuning=len(systems)*len(args.threads) if args.autotune_threads else 0,
                timing=dict(fresh='median(CSR conversion + analysis/factorization + first solve)',
                            reused='mean(second solve with the same factors and physical RHS)',
                            validation='hashes, residuals, planted probe, snapshots and file I/O outside timers',
                            comparisons='archived iterative timings; not concurrently remeasured'),
                machine=dict(platform=platform.platform(), affinity=sorted(os.sched_getaffinity(0)),
                             physical_cores=physical_threads(), total_ram_kib=memory_kib('/proc/meminfo', 'MemTotal')))
    plan['fingerprint'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def worker_environment(threads):
    env = dict(os.environ, MKL_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               OMP_THREAD_LIMIT=str(threads), OMP_DYNAMIC='FALSE',
               MKL_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', NUMBA_DISABLE_JIT='1',
               HDGFEM_PRECISION='float64', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
    # An inherited per-domain override can defeat the candidate count.
    env.pop('MKL_DOMAIN_NUM_THREADS', None)
    return env


def csv_table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def summarize(plan, records, output, *, finished=False, selections=()):
    lookup = {s['system_id']: s for s in plan['systems']}
    timings, paired = [], []
    comparable = {}
    for record in records:
        system = lookup[record['system_id']]
        metrics = archive.timing_metrics(record)
        if metrics and record.get('measurement_phase') != 'tuning':
            comparable.setdefault(record['system_id'], set()).update(record.get('selected_for', ['fresh', 'reused']))
        samples = record.get('samples', [])
        def measured_median(key):
            values = [s[key] for s in samples if key in s]
            return statistics.median(values) if metrics and values else None
        stats = [s['factor_statistics'] for s in record.get('warmups', []) + record.get('samples', []) if 'factor_statistics' in s]
        internal = [v['estimated_solver_peak_kib'] for v in stats if v.get('estimated_solver_peak_kib') is not None]
        fill = [v['fill_ratio'] for v in stats if v.get('fill_ratio') is not None]
        peak = max(record.get('peak_process_rss_kib', 0)/1024**2,
                   record.get('monitored_peak_hwm_gib', 0), record.get('monitored_peak_rss_gib', 0))
        timings.append(dict(system_id=system['system_id'], key=system['key'], case=system['case'],
                            geometry=system['geometry'], p=system['p'], triangles=system['triangles'],
                            trace_dofs=system['trace_dofs'], origins=system['origins'],
                            threads=record['threads'], status=record['status'],
                            measurement_phase=record.get('measurement_phase', 'measurement'),
                            selected_for=record.get('selected_for', ['fresh', 'reused']),
                            fresh_median_ms=metrics['fresh_median_ms'] if metrics else None,
                            reused_mean_ms=metrics['reused_mean_ms'] if metrics else None,
                            setup_median_ms=metrics['setup_median_ms'] if metrics else None,
                            fresh_min_ms=metrics['fresh_min_ms'] if metrics else None,
                            fresh_max_ms=metrics['fresh_max_ms'] if metrics else None,
                            reused_min_ms=metrics['reused_min_ms'] if metrics else None,
                            reused_max_ms=metrics['reused_max_ms'] if metrics else None,
                            csr_conversion_median_ms=measured_median('csr_conversion_ms'),
                            analysis_factorization_median_ms=measured_median('analysis_factorization_ms'),
                            factor_solve_median_ms=measured_median('factor_solve_ms'),
                            face_storage_bytes=system.get('face_storage_bytes'),
                            csr_storage_bytes=record.get('csr_storage_bytes'),
                            peak_process_rss_gib=peak,
                            estimated_lu_solver_peak_gib=max(internal)/1024**2 if internal else None,
                            fill_ratio=max(fill) if fill else None,
                            diagnostic_output=record['diagnostic_output']))
        paired.extend(archive.comparisons(system, record))
    counts = dict(Counter(r['status'] for r in records))
    summary = dict(status='completed' if finished else 'running',
                   scheduled=len(records) if finished and plan['thread_tuning']['enabled'] else plan['scheduled'],
                   scheduled_upper_bound=plan['scheduled'],
                   scheduled_is_upper_bound=plan['scheduled_is_upper_bound'] and not finished,
                   phase_counts={phase:dict(Counter(r['status'] for r in records if r.get('measurement_phase', 'measurement') == phase))
                                 for phase in ('tuning', 'confirmation', 'measurement')},
                   systems_with_confirmed_baselines=sum(v == {'fresh', 'reused'} for v in comparable.values()),
                   thread_tuning=plan['thread_tuning'],
                   attempted=len(records), counts=counts, passed=counts.get('passed', 0),
                   coverage=plan['coverage'], records=timings,
                   note='Speedup = archived iterative time / new direct time; >1 favors PyPardiso. '
                        'No time is ranked for a failed/incomplete solve. CPU RAM is not GPU VRAM.')
    write_json(output/'summary.json', summary)
    write_json(output/'comparisons.json', paired)
    if plan['thread_tuning']['enabled']:
        write_json(output/'thread_selection.json', list(selections))
        csv_table(output/'tuning.csv', [r for r in timings if r['measurement_phase'] == 'tuning'])
        csv_table(output/'confirmed_timings.csv', [r for r in timings if r['measurement_phase'] == 'confirmation'])
    csv_table(output/'timings.csv', timings)
    csv_table(output/'comparisons.csv', paired)
    return summary


def execute(plan, args):
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError('Use a new output directory or --resume with identical settings')
    if args.resume:
        previous = archive.read(output/'manifest.json')
        if previous['fingerprint'] != plan['fingerprint']:
            raise ValueError('Source artifacts/code/protocol/cache changed; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume:
            write_json(output/'manifest.json', plan)
            snapshot = output/'source_snapshot'
            snapshot.mkdir()
            for p in SOURCE_FILES:
                # Archival copy only, never executed in place of the live source.
                (snapshot/p.name).write_bytes(p.read_bytes())
        records, selections = [], []

        def run_job(system, threads, phase='measurement', selection=None):
            suffix = '' if phase == 'measurement' else f"_{phase}"
            if selection is not None:
                suffix += '_' + selection['fingerprint'][:12]
            job = output/'jobs'/f"{system['key']}{suffix}_threads{threads}"
            job.mkdir(parents=True, exist_ok=True)
            saved = job/'result.json'
            old = archive.read(saved) if saved.exists() else None
            if old and old.get('status') not in ('running', 'interrupted') and not (args.retry_failed and old.get('status') != 'passed'):
                records.append(old)
                return old
            attempt_number = 1
            while (job/f'attempt_{attempt_number}').exists():
                attempt_number += 1
            attempt = job/f'attempt_{attempt_number}'
            attempt.mkdir()
            protocol = dict(plan['protocol'])
            if phase == 'tuning':
                protocol['repeats'] = plan['thread_tuning']['repeats']
            selected_for = tuning.confirmation_objectives(selection, threads) if selection else ([] if phase == 'tuning' else ['fresh', 'reused'])
            tags = dict(measurement_phase=phase, selected_for=selected_for,
                        selection_fingerprint=selection['fingerprint'] if selection else None)
            spec = dict(system=system, threads=threads, protocol=protocol, **tags)
            spec_path = attempt/'spec.json'
            write_json(spec_path, spec)
            record = dict(system_id=system['system_id'], threads=threads,
                          diagnostic_output=str(attempt), status='running', **tags)
            write_json(saved, record)
            summarize(plan, records, output, selections=selections)
            command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                       '--worker-spec', str(spec_path), '--output', str(attempt)]
            event(output, 'start', message=f"{len(records)+1}/{plan['scheduled']} {job.name} log={attempt/'worker.log'}")
            available = memory_kib('/proc/meminfo', 'MemAvailable')/1024**2
            try:
                if available <= args.reserve_gib:
                    result = dict(status='memory_budget_exceeded', error='Available memory below reserve before launch')
                else:
                    result = monitor(command, worker_environment(threads), attempt, args)
            except KeyboardInterrupt:
                record['status'] = 'interrupted'
                write_json(saved, record)
                raise
            record.update(result)
            write_json(saved, record)
            records.append(record)
            summary = summarize(plan, records, output, selections=selections)
            event(output, 'progress', message=f"finished={len(records)}/{plan['scheduled']} statuses={summary['counts']}")
            return record

        try:
            for system in plan['systems']:
                if not plan['thread_tuning']['enabled']:
                    for threads in plan['threads']:
                        run_job(system, threads)
                    continue
                order = tuning.candidate_order(system['system_id'], plan['threads'], plan['thread_tuning']['seed'])
                pilots = [run_job(system, threads, 'tuning') for threads in order]
                selection = tuning.select_threads(system['system_id'], pilots)
                selections.append(selection)
                summarize(plan, records, output, selections=selections)
                event(output, 'thread_selection',
                      message=f"{system['key']}: selected threads {selection['selected_threads']}; starting independent confirmation",
                      selection=selection)
                winners = sorted({t for t in selection['selected_threads'].values() if t is not None})
                for threads in winners:
                    run_job(system, threads, 'confirmation', selection)
        except KeyboardInterrupt:
            summarize(plan, records, output, selections=selections)
            print('Interrupted; completed jobs retained. Resume with the same command plus --resume.', flush=True)
            return 130
        summary = summarize(plan, records, output, finished=True, selections=selections)
        print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))
        complete = summary['systems_with_confirmed_baselines'] == len(plan['systems'])
        return 0 if complete and summary['passed'] == len(records) else 1


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_spec:
        from scripts.advection_diffusion_reaction.campaigns.pardiso.adr_pardiso_worker import run
        return run(args.worker_spec, args.output)
    if args.status:
        data = archive.read(args.output/'summary.json')
        print(json.dumps({k: v for k, v in data.items() if k != 'records'}, indent=2))
        for path in sorted((args.output/'jobs').glob('*/result.json')):
            row = archive.read(path)
            if row.get('status') == 'running':
                attempt = Path(row['diagnostic_output'])
                live = archive.read(attempt/'result.json') if (attempt/'result.json').exists() else {}
                print(f"{path.parent.name}: {live.get('stage', 'starting')} log={attempt/'worker.log'}")
        return 0
    validate(args)
    plan = make_plan(args)
    print(json.dumps(dict(**plan['coverage'], threads=plan['threads'], scheduled=plan['scheduled'],
                          protocol=plan['protocol'], thread_tuning=plan['thread_tuning'],
                          scheduled_is_upper_bound=plan['scheduled_is_upper_bound'],
                          scheduled_tuning=plan['scheduled_tuning'], output=plan['output'],
                          note='Run on an idle machine after the existing campaign finishes. No GPU wrapper required.'), indent=2), flush=True)
    return execute(plan, args) if args.execute else 0


if __name__ == '__main__':
    raise SystemExit(main())
