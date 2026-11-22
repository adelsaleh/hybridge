#!/usr/bin/env python3
"""Host (CPU) PyPardiso reference for the unified ADR campaign's GPU-assembled systems.

Placement: a whole CPU node (genoa) with no GPU; the device campaign never runs PARDISO.
Each run reads the device campaign's manifest and assembly records, checks every operator
cache against the size/mtime and matrix/RHS hash its GPU assembly recorded, and runs the
isolated worker of campaigns/adr_pardiso on every system assembled so far, largest first.

Second version (the first version's workers spent ~150 s of ~151 s starting up):
- Workers start in seconds: PYPARDISO_MKL_RT skips pypardiso's recursive venv glob for
  mkl_rt, bytecode comes precompiled from PYTHONPYCACHEPREFIX, and the worker imports only
  the hybridge.linalg modules it uses (no mesh/Numba/CuPy package __init__s).
- Thread tuning only where it matters. Below --tune-min-dofs every system gets one
  measurement at --small-threads (the first version found fresh LU flat within ~10% from 8 to 96 threads
  up to 24k DOFs, reused solves best at 8-32). Above it, a few representatives per level
  (its largest, and its smallest when the level spans more than --representative-span) are
  tuned and confirmed; every other large system is measured at the thread counts selected
  for the nearest representative of its level.
- Every system gets a memory and time estimate from its cache shapes before it runs; one
  whose estimated peak exceeds the RSS limit is recorded as memory_excluded, never run.
- Systems finished by an earlier campaign (--previous) are cited, not recomputed.

Instead of the device campaign, --systems takes a list of exported caches (e.g. the H200 tuned
campaign's systems without LU), verified against each cache's metadata.json. --borrow-selections
reuses an earlier output's tuned thread selections by nearest size, and --fixed-threads measures
systems from --fixed-threads-from-dofs up at fixed counts without tuning.

Index-width, memory and time failures are recorded results, never skipped silently.

Rerunning resumes: finished jobs are kept. Exit 0 when the device campaign is complete and
every system is handled; 4 when caught up while the device campaign is still assembling
(the Slurm wrapper then resubmits itself later instead of idling on a node); 2/3 after a
requeue warning / deliberate stop.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import glob
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaigns.adr_pardiso import adr_pardiso_tuning as tuning
from campaigns.adr_pardiso.adr_pardiso_inventory import digest, read, timing_metrics
from campaigns.adr_pardiso.run_adr_pardiso_campaign import worker_environment as pardiso_environment
from campaigns.common.lifecycle import REQUEUE_SIGNALS, CampaignStop, log, stop_on_signals
from campaigns.common.production_placement import require_placement
from scripts.advection_diffusion_reaction.diagnostics.check_cached_adr_pardiso import (
    memory_kib, monitor, physical_threads, write_json,
)

WORKER = ROOT/'campaigns/adr_pardiso/run_adr_pardiso_campaign.py'
CAUGHT_UP = 4
FINAL = {'passed', 'numerical_failure', 'error', 'timeout', 'memory_budget_exceeded', 'cache_changed',
         'memory_excluded', 'index_width_excluded', 'residual_floor'}
# PyPardiso calls MKL's 32-bit pardiso interface: the matrix's CSR indices (n, nnz) are int32. The
# factors are not bounded by it: the H200 LU follow-up factorized and solved 8M-DOF systems whose factor count
# overflowed iparm(18) (reported as unavailable), all passing the residual check.
INT32_MAX = 2**31 - 1

# Cost model fitted on the first version's 367 passed runs (16-32 threads, <= 24k DOFs, p = 1..7).
# Applied far beyond its fit, so it orders and budgets the work; the worker monitor's RSS
# guard, not this estimate, is what protects the node.
FILL = (1.957, 0.341)            # factor nnz / matrix nnz = a + b ln(faces)
SYMBOLIC_BYTES_PER_FACTOR_NNZ = 17.5  # PARDISO's reported peak (iparm 15/16) per factor entry
MATRIX_BYTES_PER_NNZ = 28.0      # mmapped blocks + BSR/CSR conversion transients
FACTOR_MS = 7.4e-7               # one analysis+factorization ~ FACTOR_MS * faces**1.5 * b**3
CONVERSION_MS_PER_NNZ = 1e-4
HASH_BYTES_PER_SECOND = 1e9
WORKER_STARTUP_SECONDS = 5.0


def device_systems(device):
    """Systems whose GPU assembly passed, and whether more can still come."""
    if not (device/'manifest.json').exists():
        return [], False, 0  # the device campaign has not started its execution yet
    plan = read(device/'manifest.json')['plan']
    ready = []
    for system in plan['systems']:
        path = device/'jobs'/f"assemble_{system['id']}"/'result.json'
        record = read(path) if path.exists() else {}
        if record.get('status') == 'passed' and record.get('operator_sha256'):
            ready.append((system, record))
    completion = device/'completion.json'
    finished = completion.exists() and read(completion).get('status') in ('completed', 'device_not_released')
    return ready, finished, len(plan['systems'])


def estimate(faces, block, nnz):
    """Planning estimate of one system's LU memory and per-factorization time."""
    factor_nnz = (FILL[0] + FILL[1]*math.log(faces))*nnz
    return dict(factor_nnz=int(factor_nnz),
                peak_rss_gib=0.5 + (MATRIX_BYTES_PER_NNZ*nnz + SYMBOLIC_BYTES_PER_FACTOR_NNZ*factor_nnz)/1024**3,
                factorization_s=FACTOR_MS*faces**1.5*block**3/1000,
                conversion_s=CONVERSION_MS_PER_NNZ*nnz/1000,
                hash_s=8*nnz/HASH_BYTES_PER_SECOND)


def run_seconds(spec, factorizations):
    cost = spec['estimate']
    return WORKER_STARTUP_SECONDS + cost['hash_s'] + factorizations*(cost['factorization_s'] + cost['conversion_s'])


def worker_system(system, record, device):
    """The adr_pardiso worker's system record, verified against the GPU assembly's cache record."""
    import numpy as np
    cache = device/'cache'/system['id']
    files = {}
    for name, saved in record['cache_files'].items():
        stat = (cache/name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (saved['bytes'], saved['mtime_ns']):
            raise ValueError(f'{cache/name} changed since GPU assembly')
        files[name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    rhs = np.load(cache/'system_rhs.npy', mmap_mode='r', allow_pickle=False)
    blocks = np.load(cache/'system_blocks.npy', mmap_mode='r', allow_pickle=False)
    neighbors = np.load(cache/'system_neighbors.npy', mmap_mode='r', allow_pickle=False)
    block = int(blocks.shape[-1])
    nnz = int(np.count_nonzero(np.asarray(neighbors) >= 0))*block*block
    return dict(key=system['id'], cache=str(cache), cache_files=files, system_id=record['operator_sha256'],
                neighbors_file_sha256=digest(cache/'system_neighbors.npy'), trace_dofs=int(rhs.size),
                case=system['problem']['case'], geometry=system['geometry'], p=system['problem']['p'],
                level=system['level'], triangles=system.get('triangles'),
                suites=sorted({o['suite'] for o in system['origins']}),
                block_size=block, nnz=nnz, estimate=estimate(int(rhs.shape[0]), block, nnz))


def listed_systems(path):
    """Worker system records from a system list (CSV of exported host-PARDISO caches).

    Columns: level, geometry, case, p, triangles, trace_dofs, system_id, key, cache,
    operator_sha256. Each cache holds the system_*.npy arrays and a metadata.json whose
    operator_sha256 and array sizes must match; the worker re-hashes matrix and RHS before it runs.
    """
    import numpy as np
    specs, skipped = [], []
    for row in csv.DictReader(open(path)):
        cache = Path(row['cache'])
        try:
            meta = read(cache/'metadata.json')
            if meta.get('operator_sha256') != row['operator_sha256']:
                raise ValueError(f"{cache/'metadata.json'} operator_sha256 differs from {path}")
            files = {}
            for name, size in meta['arrays'].items():
                stat = (cache/name).stat()
                if stat.st_size != size:
                    raise ValueError(f'{cache/name} has {stat.st_size} bytes, the export recorded {size}')
                files[name] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            rhs = np.load(cache/'system_rhs.npy', mmap_mode='r', allow_pickle=False)
            blocks = np.load(cache/'system_blocks.npy', mmap_mode='r', allow_pickle=False)
            neighbors = np.load(cache/'system_neighbors.npy', mmap_mode='r', allow_pickle=False)
            if rhs.size != int(row['trace_dofs']):
                raise ValueError(f"{cache} has {rhs.size} trace DOFs, the list says {row['trace_dofs']}")
        except (OSError, ValueError, KeyError) as error:
            skipped.append((row['key'], str(error)))
            continue
        block = int(blocks.shape[-1])
        nnz = int(np.count_nonzero(np.asarray(neighbors) >= 0))*block*block
        specs.append(dict(key=row['key'], cache=str(cache), cache_files=files, system_id=row['operator_sha256'],
                          neighbors_file_sha256=digest(cache/'system_neighbors.npy'), trace_dofs=int(rhs.size),
                          case=row['case'], geometry=row['geometry'], p=int(row['p']), level=row['level'],
                          triangles=int(row['triangles']), suites=[], origins=[], list_system_id=row['system_id'],
                          block_size=block, nnz=nnz, estimate=estimate(int(rhs.shape[0]), block, nnz)))
    return specs, skipped


def borrowed_selections(output):
    """Thread selections tuned and confirmed by an earlier PARDISO output, for nearest-size reuse."""
    path = Path(output)/'thread_selection.json'
    return [dict(selection, key=key) for key, selection in (read(path) if path.exists() else {}).items()
            if selection.get('status') == 'selected' and selection.get('source') is None]


def mkl_runtime():
    """The venv's mkl_rt, found without pypardiso's recursive glob over the whole venv."""
    found = sorted(glob.glob(f'{sys.prefix}/lib/libmkl_rt.so*'), key=len)
    if not found:
        raise SystemExit(f'No libmkl_rt.so* in {sys.prefix}/lib; set PYPARDISO_MKL_RT')
    return found[0]


def worker_environment(threads):
    """MKL gets the measured LU thread count; pypardiso is told where mkl_rt is."""
    env = pardiso_environment(threads)
    env['PYPARDISO_MKL_RT'] = os.environ.get('PYPARDISO_MKL_RT') or mkl_runtime()
    return env


def failure_kind(record):
    text = f"{record.get('error', '')} {record.get('traceback', '')}".lower()
    if record.get('status') == 'index_width_excluded':
        return 'index_width'
    if record.get('status') == 'residual_floor':
        return 'residual_floor'
    if record.get('status') in ('memory_budget_exceeded', 'memory_excluded') or 'memory' in text:
        return 'memory'
    if '32-bit' in text or 'int32' in text or 'index' in text and 'pardiso' in text:
        return 'index_width'
    if record.get('status') == 'timeout':
        return 'time'
    return None if record.get('status') == 'passed' else 'other'


def representatives(specs, threshold, span):
    """Per level, the largest system at or above THRESHOLD DOFs, plus its smallest if the level spans more than SPAN."""
    levels = {}
    for spec in specs:
        if spec['trace_dofs'] >= threshold:
            levels.setdefault(spec['level'], []).append(spec)
    chosen = {}
    for level, members in levels.items():
        members.sort(key=lambda s: (s['trace_dofs'], s['key']))
        picks = [members[-1]]
        if members[-1]['trace_dofs'] > span*members[0]['trace_dofs']:
            picks.append(members[0])
        chosen[level] = picks
    return chosen


class Campaign:
    def __init__(self, args, specs):
        self.args, self.output = args, args.output
        self.records, self.selections, self.previous = {}, {}, {}
        if (self.output/'thread_selection.json').exists():
            self.selections = read(self.output/'thread_selection.json')
        if args.previous:
            self.previous = self.previous_results(args.previous)
        self.borrowed = borrowed_selections(args.borrow_selections) if args.borrow_selections else []
        self.representatives = representatives(specs, args.tune_min_dofs, args.representative_span)
        self.representative_keys = {s['key'] for picks in self.representatives.values() for s in picks}

    @staticmethod
    def previous_results(previous):
        """Systems an earlier campaign tuned and confirmed: their passed confirmation records, by key."""
        path = previous/'thread_selection.json'
        found = {}
        for key, selection in (read(path) if path.exists() else {}).items():
            if selection.get('status') != 'selected':
                continue
            records = []
            for threads in sorted({t for t in selection['selected_threads'].values() if t is not None}):
                result = previous/'jobs'/f"{key}_confirmation_{selection['fingerprint'][:12]}_threads{threads}"/'attempt_1'/'result.json'
                job = result.parent.parent/'result.json'
                source = job if job.exists() else result
                record = read(source) if source.exists() else {}
                if record.get('status') != 'passed':
                    break
                records.append((threads, source, record, selection))
            else:
                found[key] = records
        return found

    def run_job(self, system, threads, phase, selection=None, thread_source=None):
        suffix = '' if phase == 'measurement' else f'_{phase}'
        if selection is not None:
            suffix += '_' + selection['fingerprint'][:12]
        job = self.output/'jobs'/f"{system['key']}{suffix}_threads{threads}"
        job.mkdir(parents=True, exist_ok=True)
        saved = job/'result.json'
        old = read(saved) if saved.exists() else None
        if old and old.get('status') in FINAL and not (self.args.retry_failed and old['status'] != 'passed'):
            self.records[job.name] = old
            return old
        attempt = 1
        while (job/f'attempt_{attempt}').exists():
            attempt += 1
        attempt = job/f'attempt_{attempt}'
        attempt.mkdir()
        protocol = dict(repeats=self.args.tuning_repeats if phase == 'tuning' else self.args.repeats,
                        warmup=self.args.warmup, rtol=self.args.rtol, floor_rtol=self.args.floor_rtol, max_dofs=self.args.max_dofs,
                        max_rss_gib=self.args.max_rss_gib, reserve_gib=self.args.reserve_gib,
                        timeout=self.args.timeout, poll_seconds=self.args.poll_seconds)
        if selection:
            selected_for = tuning.confirmation_objectives(selection, threads)
        else:
            selected_for = [] if phase == 'tuning' else (thread_source or {}).get('objectives', ['fresh', 'reused'])
        factorizations = protocol['warmup'] + protocol['repeats']
        spec = dict(system=system, threads=threads, protocol=protocol, measurement_phase=phase,
                    selected_for=selected_for, selection_fingerprint=selection['fingerprint'] if selection else None,
                    thread_source=thread_source, estimated_run_seconds=run_seconds(system, factorizations))
        write_json(attempt/'spec.json', spec)
        record = dict(system_id=system['system_id'], key=system['key'], threads=threads, measurement_phase=phase,
                      selected_for=selected_for, thread_source=thread_source, estimate=system['estimate'],
                      diagnostic_output=str(attempt), status='running', started=time.time())
        write_json(saved, record)
        cost = system['estimate']
        log(f"start {job.name}: {system['trace_dofs']:,} DOFs nnz {system['nnz']:,} p={system['p']} {system['level']} "
            f"{system['geometry']}/{system['case']} threads={threads}; estimate {cost['peak_rss_gib']:.1f} GiB, "
            f"{cost['factorization_s']:.1f} s/factorization, {spec['estimated_run_seconds']:.0f} s run; "
            f"log={attempt/'worker.log'}")
        command = [sys.executable, '-u', '-B', str(WORKER), '--worker-spec', str(attempt/'spec.json'),
                   '--output', str(attempt)]
        available = memory_kib('/proc/meminfo', 'MemAvailable')/1024**2
        try:
            if available <= self.args.reserve_gib:
                result = dict(status='memory_budget_exceeded', error='Available memory below reserve before launch')
            else:
                result = monitor(command, worker_environment(threads), attempt, self.args)
        except KeyboardInterrupt as error:
            record.update(status='interrupted', stop_signal=getattr(error, 'signal_name', 'SIGINT'))
            write_json(saved, record)
            raise
        record.update(result, failure_kind=failure_kind(result))
        write_json(saved, record)
        self.records[job.name] = record
        metrics = timing_metrics(record)
        startup = record.get('startup', {})
        first = record.get('first_call', {})
        log(f"finish {job.name}: {record['status']}"
            + (f" fresh={metrics['fresh_median_ms']:.1f} ms (setup {metrics['setup_median_ms']:.1f}) "
               f"reused={metrics['reused_mean_ms']:.2f} ms" if metrics else '')
            + (f" first-call fresh={first['fresh_setup_solve_ms']:.1f} ms" if first else '')
            + (f"; startup {startup['total_seconds']:.1f} s (numpy/scipy {startup['numpy_scipy_seconds']:.1f}, "
               f"pypardiso {startup['pypardiso_seconds']:.1f}, hybridge {startup['hybridge_seconds']:.1f})" if startup else '')
            + f"; wall {record.get('worker_wall_seconds', 0):.1f} s"
            + (f" ({record['failure_kind']}: {str(record.get('error', ''))[:160]})" if record['failure_kind'] else '')
            + f", peak RSS {record.get('monitored_peak_hwm_gib', 0):.2f} GiB (estimate {cost['peak_rss_gib']:.2f})")
        self.summarize()
        return record

    def tune(self, system):
        """Thread sweep and confirmation of one representative; returns its selection."""
        if system['key'] in self.selections and self.selections[system['key']].get('source') is None:
            selection = self.selections[system['key']]
            for threads in sorted({t for t in selection['selected_threads'].values() if t is not None}):
                self.run_job(system, threads, 'confirmation', selection)
            return selection
        order = tuning.candidate_order(system['system_id'], self.args.tune_threads, self.args.tuning_seed)
        pilots = [self.run_job(system, threads, 'tuning') for threads in order]
        selection = tuning.select_threads(system['system_id'], pilots)
        selection.update(p=system['p'], trace_dofs=system['trace_dofs'], level=system['level'])
        self.selections[system['key']] = selection
        write_json(self.output/'thread_selection.json', self.selections)
        log(f"{system['key']}: thread selection {selection['selected_threads']} "
            f"({system['level']} representative, {system['trace_dofs']:,} DOFs)")
        for threads in sorted({t for t in selection['selected_threads'].values() if t is not None}):
            self.run_job(system, threads, 'confirmation', selection)
        return selection

    def representative_threads(self, system, specs_by_key):
        """Thread counts (and objectives) of the nearest tuned representative of SYSTEM's level."""
        picks = [s for s in self.representatives.get(system['level'], []) if s['trace_dofs'] <= self.args.tune_max_dofs]
        if not picks:  # every representative of this level is too large to tune: borrow the largest tuned one
            tuned = [s for s in self.selections.values() if s.get('source') is None and s.get('status') == 'selected']
            if not tuned:
                return {self.args.fallback_threads: ['fresh', 'reused']}, dict(kind='fallback')
            best = max(tuned, key=lambda s: s['trace_dofs'])
            source = dict(kind='largest_tuned', system_id=best['system_id'], trace_dofs=best['trace_dofs'])
            selection = best
        else:
            nearest = min(picks, key=lambda s: (abs(math.log(s['trace_dofs']/system['trace_dofs'])), s['key']))
            selection = self.tune(specs_by_key[nearest['key']])
            source = dict(kind='representative', key=nearest['key'], trace_dofs=nearest['trace_dofs'])
        if selection.get('status') != 'selected':
            return {self.args.fallback_threads: ['fresh', 'reused']}, dict(source, kind='fallback')
        counts = {}
        for objective, threads in selection['selected_threads'].items():
            if threads is not None:
                counts.setdefault(threads, []).append(objective)
        return counts, source

    def index_width(self, system):
        """Why SYSTEM cannot go through the 32-bit pardiso interface, or None."""
        if system['nnz'] > INT32_MAX or system['trace_dofs'] > INT32_MAX:
            return f"{system['nnz']:,} matrix entries exceed the 32-bit CSR indices of the pardiso interface"
        if self.args.max_factor_nnz and system['estimate']['factor_nnz'] > self.args.max_factor_nnz:
            return (f"Estimated {system['estimate']['factor_nnz']:,} factor entries exceed --max-factor-nnz "
                    f"{self.args.max_factor_nnz:,}")
        return None

    def fixed_large(self, system):
        """Systems from --fixed-threads-from-dofs up are measured at each --fixed-threads count, untuned."""
        return bool(self.args.fixed_threads) and system['trace_dofs'] >= self.args.fixed_threads_from_dofs

    def borrowed_threads(self, system):
        """Thread counts of the nearest-size selection borrowed from --borrow-selections."""
        nearest = min(self.borrowed, key=lambda s: (abs(math.log(s['trace_dofs']/system['trace_dofs'])), s['key']))
        counts = {}
        for objective, threads in nearest['selected_threads'].items():
            if threads is not None:
                counts.setdefault(threads, []).append(objective)
        return counts, dict(kind='borrowed', key=nearest['key'], trace_dofs=nearest['trace_dofs'],
                            source=str(self.args.borrow_selections))

    def cite(self, system):
        for threads, source, record, selection in self.previous[system['key']]:
            self.records[f"{system['key']}_previous_threads{threads}"] = dict(
                record, measurement_phase='previous', reused_from=str(source), estimate=system['estimate'])
            self.selections.setdefault(system['key'], dict(selection, source=str(self.args.previous)))

    def system(self, system, specs_by_key):
        if system['key'] in self.previous:
            self.cite(system)
            log(f"{system['key']}: cited from {self.args.previous} (tuned and confirmed there)")
            return
        reason = self.index_width(system)
        if reason:
            job = self.output/'jobs'/f"{system['key']}_index_width"
            job.mkdir(parents=True, exist_ok=True)
            record = dict(status='index_width_excluded', failure_kind='index_width', key=system['key'],
                          system_id=system['system_id'], trace_dofs=system['trace_dofs'], estimate=system['estimate'],
                          error=reason)
            write_json(job/'result.json', record)
            self.records[job.name] = record
            log(f"skip {system['key']}: {record['error']}")
            return
        if system['estimate']['peak_rss_gib'] > self.args.max_rss_gib:
            job = self.output/'jobs'/f"{system['key']}_memory"
            job.mkdir(parents=True, exist_ok=True)
            record = dict(status='memory_excluded', key=system['key'], system_id=system['system_id'],
                          trace_dofs=system['trace_dofs'], estimate=system['estimate'],
                          error=f"Estimated peak {system['estimate']['peak_rss_gib']:.0f} GiB exceeds the "
                                f"{self.args.max_rss_gib:.0f} GiB RSS limit")
            write_json(job/'result.json', record)
            self.records[job.name] = record
            log(f"skip {system['key']}: {record['error']}")
            return
        if self.fixed_large(system):
            for threads in self.args.fixed_threads:
                self.run_job(system, threads, 'measurement', thread_source=dict(kind='fixed_large', objectives=['fresh', 'reused']))
            return
        if system['trace_dofs'] < self.args.tune_min_dofs:
            self.run_job(system, self.args.small_threads, 'measurement',
                         thread_source=dict(kind='fixed_small', objectives=['fresh', 'reused']))
            return
        if self.borrowed:
            counts, source = self.borrowed_threads(system)
            for threads, objectives in sorted(counts.items()):
                self.run_job(system, threads, 'measurement', thread_source=dict(source, objectives=objectives))
            return
        if system['key'] in self.representative_keys and system['trace_dofs'] <= self.args.tune_max_dofs:
            self.tune(system)
            return
        counts, source = self.representative_threads(system, specs_by_key)
        for threads, objectives in sorted(counts.items()):
            self.run_job(system, threads, 'measurement', thread_source=dict(source, objectives=objectives))

    def summarize(self):
        rows = []
        for name, record in sorted(self.records.items()):
            metrics = timing_metrics(record) if record.get('status') in ('passed', 'residual_floor') else None
            stats = [s.get('factor_statistics', {}) for s in record.get('warmups', []) + record.get('samples', [])]
            cost = record.get('estimate') or {}
            rows.append(dict(job=name, key=record.get('key'), system_id=record.get('system_id'), threads=record.get('threads'),
                             phase=record.get('measurement_phase'), status=record.get('status'),
                             thread_source=(record.get('thread_source') or {}).get('kind'),
                             failure_kind=record.get('failure_kind'), trace_dofs=record.get('trace_dofs'), nnz=record.get('nnz'),
                             fresh_median_ms=metrics and metrics['fresh_median_ms'],
                             setup_median_ms=metrics and metrics['setup_median_ms'],
                             reused_mean_ms=metrics and metrics['reused_mean_ms'],
                             first_call_fresh_ms=(record.get('first_call') or {}).get('fresh_setup_solve_ms'),
                             factor_nnz=max((s.get('factor_nnz') or 0 for s in stats), default=None),
                             estimated_factor_nnz=cost.get('factor_nnz'),
                             peak_hwm_gib=record.get('monitored_peak_hwm_gib'),
                             estimated_peak_gib=cost.get('peak_rss_gib'),
                             startup_s=(record.get('startup') or {}).get('total_seconds'),
                             worker_wall_s=record.get('worker_wall_seconds'),
                             reused_from=record.get('reused_from'), error=str(record.get('error', ''))[:300],
                             worst_true_relative_residual=record.get('worst_true_relative_residual')))
        write_json(self.output/'summary.json', dict(counts=dict(Counter(r['status'] for r in rows)),
                                                    failure_kinds=dict(Counter(r['failure_kind'] for r in rows if r['failure_kind'])),
                                                    rows=rows, updated=time.time()))
        if rows:
            with (self.output/'timings.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def plan(self, specs):
        """Write and log the processing order with each system's role and estimates."""
        tune_runs = len(self.args.tune_threads)
        rows, total = [], 0.0
        for index, spec in enumerate(specs, 1):
            if spec['key'] in self.previous:
                role, seconds = 'cited', 0.0
            elif self.index_width(spec):
                role, seconds = 'index_width_excluded', 0.0
            elif spec['estimate']['peak_rss_gib'] > self.args.max_rss_gib:
                role, seconds = 'memory_excluded', 0.0
            elif self.fixed_large(spec):
                role = 'fixed_large'
                seconds = len(self.args.fixed_threads)*run_seconds(spec, self.args.warmup + self.args.repeats)
            elif spec['trace_dofs'] < self.args.tune_min_dofs:
                role = 'fixed_small'
                seconds = run_seconds(spec, self.args.warmup + self.args.repeats)
            elif self.borrowed:
                role = 'borrowed'
                seconds = len(self.borrowed_threads(spec)[0])*run_seconds(spec, self.args.warmup + self.args.repeats)
            elif spec['key'] in self.representative_keys and spec['trace_dofs'] <= self.args.tune_max_dofs:
                role = 'representative'
                seconds = (tune_runs*run_seconds(spec, self.args.warmup + self.args.tuning_repeats)
                           + 1.5*run_seconds(spec, self.args.warmup + self.args.repeats))
            else:
                role = 'measured'
                seconds = 1.5*run_seconds(spec, self.args.warmup + self.args.repeats)
            total += seconds
            rows.append(dict(order=index, key=spec['key'], level=spec['level'], p=spec['p'], case=spec['case'],
                             geometry=spec['geometry'], trace_dofs=spec['trace_dofs'], nnz=spec['nnz'], role=role,
                             estimated_factor_nnz=spec['estimate']['factor_nnz'],
                             estimated_peak_gib=round(spec['estimate']['peak_rss_gib'], 2),
                             estimated_factorization_s=round(spec['estimate']['factorization_s'], 2),
                             estimated_seconds=round(seconds, 1)))
        write_json(self.output/'plan.json', dict(created=time.time(), estimated_hours=total/3600, rows=rows,
                                                 representatives={level: [s['key'] for s in picks]
                                                                  for level, picks in sorted(self.representatives.items())}))
        roles = Counter(r['role'] for r in rows)
        peak = max((r['estimated_peak_gib'] for r in rows if r['role'] not in ('index_width_excluded', 'memory_excluded')),
                   default=0)
        log(f"plan: {len(rows)} systems {'smallest' if self.args.smallest_first else 'largest'} first ({dict(roles)}); "
            f"estimated {total/3600:.1f} h; largest {max(r['trace_dofs'] for r in rows):,} DOFs, "
            f"peak estimate of the runs up to {peak:.1f} GiB" if rows else 'plan: nothing to do')
        for level, picks in sorted(self.representatives.items()):
            log(f"representatives {level}: " + ', '.join(f"{s['key']} ({s['trace_dofs']:,} DOFs p={s['p']})" for s in picks))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--device-campaign', type=Path, help='Unified ADR device campaign output')
    source.add_argument('--systems', type=Path, help='System list CSV of exported caches (level, geometry, case, p, '
                                                     'triangles, trace_dofs, system_id, key, cache, operator_sha256)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous', type=Path, help='Earlier PARDISO output whose confirmed systems are cited, not rerun')
    p.add_argument('--tune-threads', nargs='+', type=int, help='MKL thread ladder for representatives '
                                                              '(default 16, 32, 64, 96, 128, physical cores)')
    p.add_argument('--small-threads', type=int, default=32, help='Fixed MKL threads below --tune-min-dofs')
    p.add_argument('--borrow-selections', type=Path, help='Earlier PARDISO output whose tuned thread selections are '
                                                          'reused by nearest size instead of tuning representatives')
    p.add_argument('--fixed-threads', nargs='+', type=int, help='Untuned thread counts from --fixed-threads-from-dofs up')
    p.add_argument('--fixed-threads-from-dofs', type=int, default=2_500_000)
    p.add_argument('--fallback-threads', type=int, default=64, help='When no representative selection exists')
    p.add_argument('--tune-min-dofs', type=int, default=100_000)
    p.add_argument('--tune-max-dofs', type=int, default=4_000_000, help='Larger representatives are not swept')
    p.add_argument('--representative-span', type=float, default=2.0)
    p.add_argument('--only', nargs='+', help='Restrict to these system ids (smoke tests)')
    p.add_argument('--tuning-repeats', type=int, default=2)
    p.add_argument('--tuning-seed', type=int, default=261027)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--rtol', type=float, default=1e-10)
    p.add_argument('--floor-rtol', type=float, default=1e-8,
                   help='Runs whose residuals all stay below it but miss --rtol are kept with status residual_floor')
    p.add_argument('--max-dofs', type=int, default=10**9, help='Only a sanity bound; LU limits are measured')
    p.add_argument('--max-rss-gib', type=float, help='Default: node memory minus the reserve')
    p.add_argument('--reserve-gib', type=float, default=32)
    p.add_argument('--max-factor-nnz', type=int,
                   help='Optional cap on estimated factor entries (index_width_excluded above it); systems whose '
                        'matrix entries exceed 2^31-1 are always excluded')
    p.add_argument('--smallest-first', action='store_true', help='Process systems in increasing size (default: largest first)')
    p.add_argument('--timeout', type=float, default=6*3600, help='Seconds per worker')
    p.add_argument('--poll-seconds', type=float, default=0.5)
    p.add_argument('--heartbeat-seconds', type=float, default=60)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--execute', action='store_true', help='Without it, only write and log the plan')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.output = args.output.resolve()
    args.device_campaign = args.device_campaign and args.device_campaign.resolve()
    args.previous = args.previous and args.previous.resolve()
    physical = physical_threads()
    args.tune_threads = sorted(set(args.tune_threads or [t for t in (16, 32, 64, 96, 128) if t < physical] + [physical]))
    affinity = len(os.sched_getaffinity(0))
    if args.execute and any(t < 1 or t > affinity for t in (*args.tune_threads, *(args.fixed_threads or ()),
                                                             args.small_threads, args.fallback_threads)):
        raise SystemExit('Thread counts must lie within the CPU affinity')
    total_gib = memory_kib('/proc/meminfo', 'MemTotal')/1024**2
    args.max_rss_gib = args.max_rss_gib or total_gib-args.reserve_gib
    args.monitor_label = 'PARDISO'
    if args.systems:
        listed, skipped = listed_systems(args.systems)
        ready, finished, planned = listed, True, len(listed)+len(skipped)
        if args.only:
            ready = [s for s in ready if s['key'] in args.only]
    else:
        ready, finished, planned = device_systems(args.device_campaign)
        if args.only:
            ready = [(s, r) for s, r in ready if s['id'] in args.only]
    log(f'{len(ready)}/{planned} {"listed" if args.systems else "device"} systems ready; '
        f'{"list " + str(args.systems) if args.systems else "device campaign " + ("complete" if finished else "still running")}; '
        f'fixed threads {args.fixed_threads} from {args.fixed_threads_from_dofs:,} DOFs; borrowed selections {args.borrow_selections}; '
        f'tune threads {args.tune_threads} from {args.tune_min_dofs:,} DOFs, {args.small_threads} threads below; '
        f'RSS limit {args.max_rss_gib:.0f} GiB; previous {args.previous}; '
        f'PYPARDISO_MKL_RT={os.environ.get("PYPARDISO_MKL_RT") or mkl_runtime()} '
        f'PYTHONPYCACHEPREFIX={os.environ.get("PYTHONPYCACHEPREFIX")}')
    if args.execute:
        require_placement('host')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = list(ready) if args.systems else []
        for key, error in (skipped if args.systems else []):
            log(f"skip {key}: {error}")
            path = args.output/'jobs'/f"{key}_cache"/'result.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, dict(status='cache_changed', error=error, key=key))
        for system, record in ([] if args.systems else ready):
            try:
                specs.append(worker_system(system, record, args.device_campaign))
            except (OSError, ValueError) as error:
                log(f"skip {system['id']}: {error}")
                path = args.output/'jobs'/f"{system['id']}_cache"/'result.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, dict(status='cache_changed', error=str(error), key=system['id']))
        specs.sort(key=lambda spec: ((1 if args.smallest_first else -1)*spec['trace_dofs'], spec['key']))
        campaign = Campaign(args, specs)
        campaign.plan(specs)
        if not args.execute:
            return 0
        specs_by_key = {spec['key']: spec for spec in specs}
        with stop_on_signals():
            try:
                for index, spec in enumerate(specs, 1):
                    log(f'system {index}/{len(specs)}: {spec["key"]} ({spec["trace_dofs"]:,} DOFs, p={spec["p"]}, {spec["level"]})')
                    campaign.system(spec, specs_by_key)
            except CampaignStop as stop:
                campaign.summarize()
                log(f'stopped by {stop.signal_name}; finished jobs are kept, rerun to resume')
                return 2 if stop.signal_name in REQUEUE_SIGNALS else 3
        campaign.summarize()
    if args.systems:
        log(f'done: {len(ready)} listed systems handled')
        return 0
    log(f'caught up on {len(ready)} systems; ' + ('device campaign complete' if finished
                                                   else 'device campaign still running, will resume later'))
    return CAUGHT_UP if not finished else 0


if __name__ == '__main__':
    raise SystemExit(main())
