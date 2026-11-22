#!/usr/bin/env python3
"""CPU (genoa) ADR thread scaling on the device ADR campaign's own systems.

Every point is one system of the unified device campaign (same plan id, case, stress
parameters and velocity normalization, mesh file or square size, degree and quadrature) at
one thread count, measured by an isolated worker (``point.py``) running hybridge's host ADR
pipeline: Numba assembly of the global trace system, PyPardiso, Numba reconstruction, Numba
total-flux and primal postprocessing.

Tiers:
- ``compile``: per degree, the smallest square system with an empty Numba cache
  (JIT compile cost) and then with the shared cache (cache-load cost), at 32 threads;
- ``scaling``: full degree range of the representative and failed-stress classes, threads from
  a per-system lower bound (estimated single-thread time / bound <= --budget-seconds) up to the
  whole node, ascending;
- ``coverage``: every system the device campaign assembled, plus the failed-stress classes'
  systems whose meshes exist, at the whole node, largest first.

Every worker runs pinned to cores 0..t-1 with NUMBA/MKL/OpenMP/NumPy-OpenBLAS at t threads (scipy's OpenBLAS,
called inside Numba kernels, at one). Rerunning
resumes (finished runs are kept); a Slurm time-limit warning saves and exits 2 for a requeue.
Failures are recorded outcomes, never crashes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaigns.common.lifecycle import REQUEUE_SIGNALS, CampaignStop, log, stop_on_signals
from campaigns.common.protocol import host_provenance
from campaigns.common.production_placement import require_placement
from scripts.advection_diffusion_reaction.diagnostics.check_cached_adr_pardiso import (
    memory_kib, monitor, physical_threads, write_json,
)

HERE = Path(__file__).resolve().parent
WORKER = HERE/'point.py'
KIND = 'hybridge'  # recorded per run
FINAL = {'passed', 'error', 'timeout', 'memory_budget_exceeded'}
# Representative classes of the device campaign's h-p grid, and the stress classes whose
# iterative solves failed there (no candidate converged on stress_main_trap / stress_main_cross).
SCALING_CLASSES = (('square', 'trigonometric', 32768), ('square', 'trigonometric', 99458),
                   ('annulus', 'stress_main_trap', 98699), ('annulus', 'stress_main_orthogonal', 98699),
                   ('annulus', 'stress_main_cross', 98699), ('annulus', 'cellular7_weak', 33174),
                   ('annulus', 'cellular7_weak', 99984))
FAILED_CLASSES = {'stress_main_trap', 'stress_main_cross', 'stress_main_orthogonal', 'cellular7_weak'}
SPEC_KEYS = ('case', 'n', 'p', 'volume_quad_1d', 'edge_quad_1d', 'stress_parameters', 'velocity_normalization',
             'normalization_record', 'mesh_path', 'mesh_sha256')


def read(path):
    return json.loads(Path(path).read_text())


def latest_attempt(job):
    attempts = sorted(job.glob('attempt_*'), key=lambda p: int(p.name.split('_')[1]))
    return attempts[-1] if attempts else None


def device_systems(device):
    """Plan systems with the device campaign's exact assembly spec (or one built like it) and record."""
    plan = read(device/'manifest.json')['plan']
    systems = []
    for system in plan['systems']:
        job = device/'jobs'/f"assemble_{system['id']}"
        attempt = latest_attempt(job) if job.exists() else None
        record = read(job/'result.json') if (job/'result.json').exists() else {}
        if attempt and (attempt/'spec.json').exists():
            source = read(attempt/'spec.json')
            spec = {k: source[k] for k in SPEC_KEYS if k in source}
        else:
            spec = dict(system['problem'], mesh_path=None)
            if system['mesh']:
                mesh = device/'jobs'/f"mesh_{system['id']}"/'result.json'
                result = read(mesh) if mesh.exists() else {}
                if result.get('status') != 'passed':
                    spec = None
                else:
                    spec.update({k: result[k] for k in ('mesh_path', 'mesh_sha256', 'n')})
        triangles = system.get('triangles') or (system['mesh'] or {}).get('target_triangles')
        systems.append(dict(id=system['id'], level=system['level'], geometry=system['geometry'],
                            case=system['problem']['case'], p=system['problem']['p'], triangles=triangles,
                            trace_dofs=system.get('trace_dofs') or record.get('estimates', {}).get('trace_dofs'),
                            spec=spec, device_status=record.get('status', 'not_yet')))
    return systems


class Estimate:
    """Single-thread seconds of one solve of a system, for the scaling tier's lower thread bound.

    seconds_1 = triangles * (QUADRATURE * points * el_dof + DENSE * el_dof**3), el_dof = (p+1)(p+2)/2
    and points = volume_quad_1d**2 (default 2p+2): coefficient sampling and quadrature dominate the
    stress cases (20-point rules at every p), the dense local solves dominate high p. Both
    constants are fitted on the genoa smoke run (--work QUADRATURE DENSE).
    """

    def __init__(self, work):
        self.quadrature, self.dense = work

    def __call__(self, system):
        p = system['p']
        el_dof = (p+1)*(p+2)/2
        points = (system['spec'].get('volume_quad_1d') or 2*p+2)**2
        return system['triangles']*(self.quadrature*points*el_dof + self.dense*el_dof**3)


def lower_bound(system, ladder, estimate, budget):
    seconds = estimate(system)
    for threads in ladder:
        if seconds/threads <= budget:
            return threads
    return ladder[-1]


def small_spec(spec, directory, triangles=256):
    """The same case, degree and quadrature on a tiny mesh: compiles the kernels a system needs, untimed.

    Squares get n=4; annuli the first TRIANGLES of the system's own mesh (points inside its
    domain, so the stress coefficients stay defined), saved with their own hash.
    """
    import hashlib
    import numpy as np
    small = dict(spec, mesh_path=None, mesh_sha256=None)
    if not spec.get('mesh_path'):
        return dict(small, n=4)
    with np.load(spec['mesh_path'], allow_pickle=False) as data:
        cells = np.asarray(data['triangles'][:triangles])
        nodes, inverse = np.unique(cells, return_inverse=True)
        coords = np.asarray(data['node_coords'])[nodes]
    path = directory/'submesh.npz'
    np.savez(path, node_coords=coords, triangles=inverse.reshape(cells.shape).astype(cells.dtype))
    return dict(small, mesh_path=str(path), mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                n=max(1, round(math.sqrt(triangles/2))))


def worker_environment(threads, cache):
    # Numba parallelises across elements, MKL threads PARDISO and NumPy's OpenBLAS keeps t threads for
    # array-level BLAS. scipy's OpenBLAS, which Numba's np.linalg/@ call inside prange kernels, is
    # limited to one thread by the worker (point.configure_blas): a multithreaded OpenBLAS started
    # from Numba workers segfaults.
    env = dict(os.environ, NUMBA_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(min(threads, 64)),  # NumPy's OpenBLAS build caps at 64 (measured)
               MKL_DYNAMIC='FALSE', OMP_DYNAMIC='FALSE', NUMBA_CACHE_DIR=str(cache),
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
    env.setdefault('PYPARDISO_MKL_RT', str(Path(sys.prefix)/'lib/libmkl_rt.so.3'))
    env.pop('NUMBA_DISABLE_JIT', None)
    env.pop('MKL_DOMAIN_NUM_THREADS', None)
    return env


def cache_files(cache):
    """Numba cache index and object files under ``cache``, relative to it."""
    cache = Path(cache)
    return {str(path.relative_to(cache)) for path in cache.rglob('*') if path.suffix in ('.nbi', '.nbc')} \
        if cache.is_dir() else set()


class Campaign:
    def __init__(self, args):
        self.args, self.output = args, args.output
        self.records = {}

    def run(self, system, kind, threads, tier, cache, label=None, small=False):
        name = label or f"{system['id']}_{kind}_t{threads}"
        job = self.output/'runs'/name
        job.mkdir(parents=True, exist_ok=True)
        saved = job/'result.json'
        old = read(saved) if saved.exists() else None
        if old and old.get('status') in FINAL and not (self.args.retry_failed and old['status'] != 'passed'):
            self.records[name] = old
            return old
        attempt = 1
        while (job/f'attempt_{attempt}').exists():
            attempt += 1
        attempt = job/f'attempt_{attempt}'
        attempt.mkdir()
        spec = dict(system['spec'], rtol=self.args.rtol, threads=threads, pardiso_threads=threads, repeats=self.args.repeats,
                    verbosity=self.args.verbosity)
        if small:  # untimed kernel compilation into the shared cache
            # Accept any residual: the tiny meshes can be ill-conditioned enough that a direct solve's
            # rounding floor misses 1e-10 (coverage tier: 120 dofs, kappa ~4e7, 3e-10), and the solver
            # raises instead of reconstructing, which skips compiling the remaining kernels.
            spec = dict(small_spec(spec, attempt), repeats=0, rtol=1.0)
        write_json(attempt/'spec.json', spec)
        record = dict(name=name, system=system['id'], level=system['level'], geometry=system['geometry'],
                      case=system['case'], p=system['p'], triangles=system['triangles'], trace_dofs=system['trace_dofs'],
                      kind=kind, threads=threads, tier=tier, numba_cache=str(cache), device_status=system['device_status'],
                      diagnostic_output=str(attempt), status='running',
                      started=time.time())
        write_json(saved, record)
        log(f"start {name} [{tier}]: {system['case']} {system['geometry']} {system['triangles']:,} triangles p={system['p']} "
            f"{kind} threads={threads}; log={attempt/'worker.log'}")
        command = ['taskset', '-c', f'0-{threads-1}', sys.executable, '-u', '-B', str(WORKER),
                   '--spec', str(attempt/'spec.json'), '--output', str(attempt)]
        cached = cache_files(cache)
        if memory_kib('/proc/meminfo', 'MemAvailable')/1024**2 <= self.args.reserve_gib:
            result = dict(status='memory_budget_exceeded', error='Available memory below reserve before launch')
        else:
            result = monitor(command, worker_environment(threads, cache), attempt, self.args)
        record.update(result)
        # Kernels compiled by this run: none expected after the compile and populate runs.
        compiled = sorted(cache_files(cache)-cached)
        record.update(numba_cache_new_files=len(compiled), numba_cache_new_sample=compiled[:20])
        if compiled and (tier in ('scaling', 'coverage') or name.startswith('cached')):
            log(f"WARNING {name}: compiled {len(compiled)} Numba cache file(s) during a timed run: {compiled[:5]}")
        write_json(saved, record)
        self.records[name] = record
        log(f"finish {name}: {record['status']} wall {record.get('worker_wall_seconds', 0):.1f} s, "
            f"startup {record.get('startup', {}).get('total_seconds', float('nan')):.1f} s, {headline(record)}"
            f"peak RSS {record.get('monitored_peak_hwm_gib', 0):.1f} GiB"
            + (f" ({str(record.get('error', ''))[:200]})" if record['status'] != 'passed' else ''))
        self.summarize()
        return record

    def summarize(self):
        rows = [row(record) for _, record in sorted(self.records.items())]
        add_efficiency(rows)
        write_json(self.output/'summary.json', dict(counts=dict(Counter(r['status'] for r in rows)), rows=rows,
                                                    updated=time.time()))
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with (self.output/'timings.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def headline(record):
    """One-line warm medians of the stages, for the log."""
    r = row(record)
    if record.get('status') != 'passed':
        return ''
    return (f"pipeline cold {r['cold_s']:.3f} warm {r['warm_s']:.3f} s (assembly {r['warm_assembly_s']:.3f} = "
            f"preparation {r['warm_preparation_s']:.3f} + kernels {r['warm_assembly_s']-r['warm_preparation_s']:.3f}, "
            f"solve {r['warm_solve_s']:.3f}, reconstruction {r['warm_reconstruction_s']:.3f}, postprocessing "
            f"{r['warm_postprocessing_s']:.3f} = flux {r['warm_post_flux_s']:.3f} + primal {r['warm_post_primal_s']:.3f}), ")


def row(record):
    out = {k: record.get(k) for k in ('name', 'tier', 'system', 'level', 'geometry', 'case', 'p', 'triangles', 'trace_dofs',
                                      'kind', 'threads', 'status', 'device_status', 'coefficients', 'numba_cache_new_files')}
    out.update(startup_s=(record.get('startup') or {}).get('total_seconds'), worker_wall_s=record.get('worker_wall_seconds'),
               peak_hwm_gib=record.get('monitored_peak_hwm_gib'), error=str(record.get('error', ''))[:200] or None,
               space_s=(record.get('setup') or {}).get('space_seconds'))
    if record.get('status') != 'passed':
        return out
    solves = record['solves']
    warm = [s for s in solves if s['kind'] == 'warm'] or solves
    stage = lambda s, key: s['timings_seconds'][key]
    step = lambda s, word: sum(x['seconds'] for x in s['steps'] if x['kind'] == 'substep' and word in x['label'])
    out.update(cold_s=solves[0]['wall_seconds'], warm_s=median(s['wall_seconds'] for s in warm),
               warm_cpu_wall=median(s['cpu_wall_ratio'] for s in warm),
               **{f'cold_{k}_s': stage(solves[0], k) for k in ('assembly', 'solve', 'reconstruction', 'postprocessing')},
               **{f'warm_{k}_s': median(stage(s, k) for s in warm)
                  for k in ('preparation', 'local_solver', 'trace_assembly', 'assembly', 'solve', 'reconstruction',
                            'postprocessing')},
               warm_post_flux_s=median(step(s, 'recovering total flux') for s in warm),
               warm_post_primal_s=median(step(s, 'recovering primal field') for s in warm),
               **(record.get('trace_system') or {}), **{f'error_{k}': v for k, v in (record.get('errors') or {}).items()})
    for key in warm[0]['details']:
        out[f'warm_detail_{key}_s'] = median(s['details'].get(key) for s in warm)
    return out


def add_efficiency(rows):
    """Speedup and parallel efficiency against each (system, kind)'s lowest passed thread count."""
    groups = defaultdict(list)
    for r in rows:
        if r['status'] == 'passed' and r['tier'] == 'scaling':
            groups[(r['system'], r['kind'])].append(r)
    for members in groups.values():
        base = min(members, key=lambda r: r['threads'])
        for r in members:
            for key in [k for k in r if k.startswith('warm_') and k.endswith('_s') or k in (
                    'pardiso_factorization_s', 'pardiso_solve_s', 'reconstruction_s', 'pipeline_warm_s')]:
                if isinstance(r.get(key), (int, float)) and isinstance(base.get(key), (int, float)) and r[key] > 0:
                    speedup = base[key]/r[key]
                    r[f'speedup_{key}'] = speedup
                    r[f'efficiency_{key}'] = speedup*base['threads']/r['threads']
            r['base_threads'] = base['threads']


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device-campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, default=os.environ.get('HYBRIDGE_CAMPAIGN_OUT'),
                   help='Output directory (default: $HYBRIDGE_CAMPAIGN_OUT)')
    p.add_argument('--tiers', nargs='+', default=['compile', 'scaling'], choices=('compile', 'scaling', 'coverage'))
    p.add_argument('--threads', nargs='+', type=int, help='Ladder (default 1, 2, 4, 8, 16, 32, 64, 96, 128, all cores)')
    p.add_argument('--coverage-threads', type=int, help='Default: all physical cores')
    p.add_argument('--compile-threads', type=int, default=32)
    p.add_argument('--budget-seconds', type=float, default=180., help='Estimated single-solve seconds at the lower bound')
    p.add_argument('--work', type=float, nargs=2, default=(1.56e-7, 6.3e-8),
                   help='Single-core seconds per triangle*point*el_dof and per triangle*el_dof**3 (fitted on the 16- and 192-thread runs of a genoa smoke)')
    p.add_argument('--repeats', type=int, default=2, help='Warm solves after the cold one')
    p.add_argument('--rtol', type=float, default=1e-10, help="Linear-solve acceptance, as the device campaign's protocol")
    p.add_argument('--verbosity', type=int, default=2)
    p.add_argument('--only', nargs='+', help='Restrict to these system ids (smoke tests)')
    p.add_argument('--max-rss-gib', type=float)
    p.add_argument('--reserve-gib', type=float, default=32)
    p.add_argument('--timeout', type=float, default=3*3600, help='Seconds per worker')
    p.add_argument('--poll-seconds', type=float, default=0.5)
    p.add_argument('--heartbeat-seconds', type=float, default=60)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--local', action='store_true', help='Allow running outside a genoa allocation (tests)')
    p.add_argument('--plan', action='store_true', help='Only write and log the plan (the default without --execute)')
    p.add_argument('--execute', action='store_true', help='Run the plan')
    return p


def plan(args, systems):
    physical = physical_threads()
    ladder = sorted(set(args.threads or [t for t in (1, 2, 4, 8, 16, 32, 64, 96, 128) if t < physical] + [physical]))
    estimate = Estimate(args.work)
    runs = []
    ready = [s for s in systems if s['spec'] is not None and (not args.only or s['id'] in args.only)]
    if 'compile' in args.tiers:
        smallest = {}
        for s in ready:
            if s['geometry'] == 'square' and s['case'] == 'trigonometric':
                if s['p'] not in smallest or s['triangles'] < smallest[s['p']]['triangles']:
                    smallest[s['p']] = s
        for p in sorted(smallest):
            s = smallest[p]
            # Empty cache: JIT compile. Then fill the shared cache, and time a pure cache load.
            for cache, prefix in (('empty', 'compile'), ('shared', 'populate'), ('shared', 'cached')):
                runs.append(dict(system=s, kind=KIND, threads=args.compile_threads, tier='compile', cache=cache,
                                 label=f"{prefix}_{s['id']}_{KIND}_t{args.compile_threads}"))
    timed = []
    if 'scaling' in args.tiers:
        chosen = [s for s in ready if (s['geometry'], s['case'], s['triangles']) in SCALING_CLASSES]
        for s in sorted(chosen, key=lambda s: (estimate(s), s['id'])):
            start = lower_bound(s, ladder, estimate, args.budget_seconds)
            s['lower_bound_threads'] = start
            for threads in [t for t in ladder if t >= start]:
                timed.append(dict(system=s, kind=KIND, threads=threads, tier='scaling', cache='shared'))
    if 'coverage' in args.tiers:
        threads = args.coverage_threads or physical
        chosen = [s for s in ready if s['device_status'] == 'passed' or s['case'] in FAILED_CLASSES]
        for s in sorted(chosen, key=lambda s: (-(s['triangles'] or 0)*(s['p']+1)**2, s['id'])):
            timed.append(dict(system=s, kind=KIND, threads=threads, tier='coverage', cache='shared'))
    # Compile every kernel the timed runs need before any of them: one untimed tiny-mesh run per
    # (geometry, case, degree) into the shared Numba cache, so a timed cold solve only
    # loads cached code. Stress, cellular and tensor cases take other kernel paths than the squares.
    seen = set()
    for r in timed:
        s = r['system']
        key = (s['geometry'], s['case'], s['p'], r['kind'])
        if key not in seen:
            seen.add(key)
            runs.append(dict(system=s, kind=r['kind'], threads=args.compile_threads, tier='populate', cache='shared',
                             small=True, label=f"populate_{s['id']}_{r['kind']}"))
    return ladder, estimate, runs + timed


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output is None:
        raise SystemExit('Give --output or set HYBRIDGE_CAMPAIGN_OUT')
    args.execute = args.execute and not args.plan
    args.device_campaign, args.output = args.device_campaign.resolve(), Path(args.output).resolve()
    total_gib = memory_kib('/proc/meminfo', 'MemTotal')/1024**2
    args.max_rss_gib = args.max_rss_gib or total_gib-args.reserve_gib
    args.monitor_label = 'CPU-ADR'
    systems = device_systems(args.device_campaign)
    ladder, estimate, runs = plan(args, systems)
    affinity = len(os.sched_getaffinity(0))
    if args.execute and any(r['threads'] > affinity for r in runs):
        raise SystemExit(f'Thread counts must lie within the CPU affinity ({affinity})')
    if args.execute and not args.local:
        require_placement('host')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tiers = Counter(r['tier'] for r in runs)
        unique = {(r['system']['id'], r['tier']) for r in runs}
        log(f"{len([s for s in systems if s['spec']])}/{len(systems)} device systems have a spec; ladder {ladder} "
            f"runs {dict(tiers)} over {len(unique)} system-tiers; repeats 1 cold + {args.repeats} warm; "
            f"RSS limit {args.max_rss_gib:.0f} GiB; PYTHONPYCACHEPREFIX={os.environ.get('PYTHONPYCACHEPREFIX')}")
        bounds = {}
        for r in runs:
            if r['tier'] == 'scaling':
                s = r['system']
                bounds[s['id']] = (s['case'], s['geometry'], s['triangles'], s['p'], s['lower_bound_threads'], estimate(s))
        for key, (case, geometry, triangles, p, start, seconds) in sorted(bounds.items(), key=lambda kv: kv[1][5]):
            log(f"scaling {key}: {case} {geometry} {triangles:,} triangles p={p}: estimated {seconds:.0f} s on one core, "
                f"threads from {start}")
        write_json(args.output/'plan.json', dict(ladder=ladder, work=args.work, budget_seconds=args.budget_seconds,
                                                 runs=[dict(system=r['system']['id'], kind=r['kind'], threads=r['threads'],
                                                            tier=r['tier'], cache=r['cache'], label=r.get('label'))
                                                       for r in runs], created=time.time()))
        if not args.execute:
            return 0
        host_provenance(args.output, campaign='cpu_adr_scaling', tiers=args.tiers, ladder=ladder,
                        device_campaign=str(args.device_campaign), runs=len(runs))
        campaign = Campaign(args)
        shared = args.output/'numba_cache'
        with stop_on_signals():
            try:
                for index, r in enumerate(runs, 1):
                    cache = shared if r['cache'] == 'shared' else args.output/'numba_cache_empty'/r['label']
                    if r['cache'] == 'empty' and not (args.output/'runs'/r['label']/'result.json').exists():
                        cache.mkdir(parents=True, exist_ok=False) if not cache.exists() else None
                    log(f"run {index}/{len(runs)}")
                    campaign.run(r['system'], r['kind'], r['threads'], r['tier'], cache, r.get('label'), r.get('small', False))
            except CampaignStop as stop:
                campaign.summarize()
                log(f'stopped by {stop.signal_name}; finished runs are kept, rerun to resume')
                return 2 if stop.signal_name in REQUEUE_SIGNALS else 3
        campaign.summarize()
        counts = dict(Counter(r['status'] for r in map(row, campaign.records.values())))
        write_json(args.output/'completion.json', dict(status='completed', finished=time.time(), counts=counts,
                                                       tiers=args.tiers, runs=len(runs)))
    log(f"done: {counts}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
