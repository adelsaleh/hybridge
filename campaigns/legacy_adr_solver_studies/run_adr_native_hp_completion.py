#!/usr/bin/env python3
"""Complete native hp-BSR coverage of the archived ADR manuscripts.

Nothing runs on import. --dry-run reads metadata only and creates no files.
A normal invocation runs only native solver jobs on existing immutable caches;
it never assembles a PDE, builds dependencies, or regenerates manuscripts.
"""
from __future__ import annotations

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))
_campaign_sys.path.insert(0, str(_campaign_workspace_path('hdgfem-gmres')))


import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import math
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time

from scripts.adr_performance_common import atomic_json, read_json

BRANCH = _campaign_workspace_path('hdgfem-gmres')
MASTER = _campaign_workspace_path('.') / 'hdgfem'
REPORT = MASTER / 'docs/research/solver_studies/adr_scaling_2026_09_17'
STUDY = MASTER / 'run_outputs/solver_studies/adr_scaling_2026_09_17'
DEFAULT_OUTPUT = STUDY / 'native_hp_completion_2026_09_21'
PROBE_NAMES = (
    'native_hp_robust',
    'square_directional_target32768_p6_native_hp_robust',
    'square_directional_target99458_p6_native_hp_robust',
    'annulus_directional_target99458_p6_native_hp_robust',
)
SPEC_KEYS = ('case', 'n', 'p', 'cache', 'device', 'rtol', 'internal_rtol',
             'maxiter', 'warmup', 'repeats', 'solves_per_setup', 'mesh_path',
             'volume_quad_1d', 'edge_quad_1d')
TERMINAL = {'passed', 'numerical_failure', 'setup_error', 'solve_error',
            'process_error', 'timeout', 'validation_failure'}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def metrics(row):
    """Rank only complete, independently accepted uninstrumented measurements."""
    if row.get('status') != 'passed' or row.get('profile_note'):
        return None
    samples = row.get('samples', [])
    if not samples or len(samples) != row.get('requested_repeats', len(samples)):
        return None
    if any(len(s.get('solves', [])) != 2 for s in samples):
        return None
    solves = [s for t in samples + row.get('warmups', []) for s in t['solves']]
    if not solves or any(not s.get('passed') or
                         not math.isfinite(s.get('true_relative_residual', math.inf)) or
                         s['true_relative_residual'] > row.get('rtol', 1e-10)
                         for s in solves):
        return None
    hot = [t['solves'][1] for t in samples]
    times = [s['solve_ms'] for s in hot]
    if any(not math.isfinite(t) or t <= 0 for t in times):
        return None
    fresh = [t['setup_ms'] + t['solves'][0]['solve_ms'] for t in samples]
    setup = [t['setup_ms'] for t in samples]
    if any(not math.isfinite(t) or t < 0 for t in fresh + setup):
        return None
    return dict(hot_mean_ms=statistics.mean(times), hot_min_ms=min(times),
                hot_max_ms=max(times), fresh_median_ms=statistics.median(fresh),
                setup_median_ms=statistics.median(setup),
                iterations=statistics.mean(s['iterations'] for s in hot),
                residual=max(s['true_relative_residual'] for s in solves))


def best(rows, metric):
    eligible = [(metrics(r), r) for r in rows]
    eligible = [(m, r) for m, r in eligible if m is not None]
    return min(eligible, key=lambda item: item[0][metric])[1] if eligible else None


def suites():
    """Read source archives, keeping report membership separate from system identity."""
    smooth = REPORT / 'scaling.data.json'
    smooth_data = read_json(smooth)
    yield 'smooth', 'smooth', smooth, smooth_data['rows'], Path(smooth_data['local_campaign']).resolve()
    osc = REPORT / 'oscillatory/scaling.data.json'
    for campaign in read_json(osc)['campaigns']:
        root = STUDY / 'oscillatory/scaling' / campaign['name']
        yield 'oscillatory', campaign['name'], osc, campaign['rows'], root.resolve()
    initial = REPORT / 'oscillatory/initial/results.data.json'
    for campaign in read_json(initial)['campaigns']:
        root = Path(campaign['manifest']['arguments']['output'])
        if not root.is_absolute():
            root = BRANCH / root
        yield 'initial', campaign['name'], initial, campaign['rows'], root.resolve()
    for label, folder in (
        ('focused_scaling', 'adr_named_hp_scaling_vv_mt_tr_20260919'),
        ('focused_endpoints', 'adr_named_comparison_vv_mt_tr_20260919'),
    ):
        source = BRANCH / 'run_logs' / folder / 'summary.json'
        yield label, folder, source, read_json(source), source.parent.resolve()


def inventory():
    systems = {}
    sources = {}
    legacy_counts = {}
    with (REPORT / 'oscillatory/initial/comparison.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            legacy_counts[(row['campaign'], row['case'], int(row['p']))] = (
                int(row['triangles']), int(row['trace_dofs']))
    for suite, campaign, source, rows, campaign_root in suites():
        sources[str(source)] = digest(source)
        peers = defaultdict(list)
        for row in rows:
            peers[(row['case'], row['n'], row['p'])].append(row)
        for (case, n, p), group in peers.items():
            hashes = {r['operator_sha256'] for r in group if r.get('operator_sha256')}
            if len(hashes) != 1:
                raise ValueError(f'Inconsistent matrix/RHS identities: {suite} {case} n{n} p{p}')
            identity = hashes.pop()
            representative = next(r for r in group if r.get('matrix_cache'))
            geometry = representative.get('geometry') or ('annulus' if '_star_' in campaign else 'square')
            counts = legacy_counts.get((campaign, case, p))
            triangles = counts[0] if counts else int(representative.get('triangles') or (2*n*n if geometry == 'square' else 0))
            if triangles <= 0:
                raise ValueError('Missing actual annular triangle count')
            dofs = counts[1] if counts else int(representative['trace_dofs'])
            origin = dict(suite=suite, campaign=campaign, source=str(source),
                          campaign_root=str(campaign_root), case=case,
                          n=n, nominal_n=representative.get('nominal_n', n), p=p,
                          geometry=geometry, triangles=triangles, trace_dofs=dofs)
            if identity not in systems:
                systems[identity] = dict(system_id=identity, case=case, n=n, p=p,
                    geometry=geometry, triangles=triangles, trace_dofs=dofs,
                    cache=str(Path(representative['matrix_cache']).resolve()),
                    origins=[], existing=[])
            entry = systems[identity]
            if (entry['case'], entry['p'], entry['trace_dofs'], entry['geometry']) != (case, p, dofs, geometry):
                raise ValueError('Conflicting metadata for a shared matrix/RHS')
            entry['origins'].append(origin)
            for row in group:
                if row.get('family') == 'native_hp' and row.get('status') == 'passed':
                    record = dict(row, operator_sha256=identity, policy=row.get('policy', 'standard'))
                    if metrics(record) is None:
                        raise ValueError('An archived native result is not fully measured')
                    entry['existing'].append(dict(policy=record['policy'], source=str(source),
                                                  origin=origin, record=record))
    for name in PROBE_NAMES:
        path = STUDY / 'oscillatory/strong_probe' / (name + '.json')
        specpath = path.with_suffix('.spec.json')
        row, spec = read_json(path), read_json(specpath)
        sources[str(path)] = digest(path)
        sources[str(specpath)] = digest(specpath)
        entry = systems[row['operator_sha256']]
        if metrics(row) is None or spec['policy'] != 'robust':
            raise ValueError(f'Invalid archived directional result: {path}')
        entry['existing'].append(dict(policy='robust', source=str(path), spec=str(specpath),
                                      record=dict(row, policy='robust')))
    # These guards stop accidental scope drift instead of silently scheduling new studies.
    if len(systems) != 101 or sum(not s['existing'] for s in systems.values()) != 61:
        raise ValueError('The source inventory changed; review the expected 101 systems / 61 native gaps')
    return systems, sources


def comparison_spec(spec):
    """Recognize measured comparison jobs, never assembly or tuning inputs."""
    protocol = dict(rtol=1e-10, internal_rtol=1e-11, maxiter=1000,
                    warmup=1, solves_per_setup=2)
    return bool(spec.get('candidate') and spec.get('family') and
                spec.get('task') in (None, 'measure') and not spec.get('profile') and
                spec.get('cache') and spec.get('case') and spec.get('result') and
                all(spec.get(key) == value for key, value in protocol.items()) and
                spec.get('repeats') in (2, 3))


def specification(entry, indices):
    cache = Path(entry['cache']).resolve()
    # Comparisons often borrow another campaign's assembly cache. The cache
    # owner is not the source of the report's solve protocol or reference solve.
    roots = [Path(origin['campaign_root']).resolve() for origin in entry['origins']]
    roots.append(cache.parent.parent)
    roots = list(dict.fromkeys(roots))
    choices = []
    for root in roots:
        if root not in indices:
            by_cache = defaultdict(list)
            for path in sorted((root / 'specs').glob('*.json')):
                spec = read_json(path)
                if comparison_spec(spec):
                    by_cache[str(Path(spec['cache']).resolve())].append((path, spec))
            indices[root] = by_cache
        choices = [(path, spec) for path, spec in indices[root].get(str(cache), [])
                   if (spec.get('case'), spec.get('n'), spec.get('p')) ==
                      (entry['case'], entry['n'], entry['p'])]
        if choices:
            break
    choices.sort(key=lambda item: (not str(item[1].get('candidate', '')).startswith('asm'), str(item[0])))
    if not choices:
        searched = ', '.join(str(root / 'specs') for root in roots)
        raise ValueError(f'No comparison solver specification with the required protocol for {cache}; searched {searched}')
    path, original = choices[0]
    spec = {k: original[k] for k in SPEC_KEYS if k in original}
    spec['cache'] = str(cache)
    spec.setdefault('device', 0)
    spec.setdefault('mesh_path', None)
    for key, value in dict(rtol=1e-10, internal_rtol=1e-11, maxiter=1000,
                           warmup=1, solves_per_setup=2).items():
        if spec.get(key) != value:
            raise ValueError(f'Unexpected {key} in {path}')
    if spec.get('repeats') not in (2, 3):
        raise ValueError(f'Unexpected repetition count in {path}')
    required = ('system_blocks.npy', 'system_neighbors.npy', 'system_rhs.npy',
                'local_inverse.npy', 'source_rhs.npy', 'boundary.npy')
    for name in required:
        if not (cache / name).is_file():
            raise FileNotFoundError(cache / name)
    if entry['geometry'] == 'annulus' and not spec['mesh_path']:
        raise ValueError(f'Missing archived annular mesh: {path}')
    if spec['mesh_path']:
        spec['mesh_path'] = str(Path(spec['mesh_path']).resolve(strict=True))
    spec.update(family='native_hp', expected_operator_sha256=entry['system_id'])
    for _, candidate in choices:
        solution = Path(candidate['result']).with_suffix('.solution.npy')
        if str(candidate.get('candidate', '')).startswith('asm') and solution.is_file():
            spec['peer_solution'] = str(solution.resolve())
            break
    return spec, path


def prepare(output, timeout, profiles):
    systems, sources = inventory()
    indices, jobs = {}, []
    for identity, entry in sorted(systems.items()):
        if entry['existing'] and entry['case'] != 'cellular7_directional':
            continue
        common, original = specification(entry, indices)
        sources[str(original)] = digest(original)
        if common['mesh_path']:
            sources[common['mesh_path']] = digest(common['mesh_path'])
        entry['common_spec'] = common
        present = {r['policy'] for r in entry['existing']}
        for policy in ('standard', 'robust'):
            if policy in present:
                continue
            key = f"{entry['geometry']}_{entry['case']}_p{entry['p']}_{identity[:12]}_{policy}"
            spec = dict(common, policy=policy, candidate='native_hp_' + policy,
                        result=str(output / 'jobs' / (key + '.json')))
            jobs.append(dict(key=key, system_id=identity, spec=spec))
    if len(jobs) != 126:
        raise ValueError(f'Expected 126 new timing jobs; found {len(jobs)}')
    # Finish smaller systems early and deterministically alternate policy order.
    random.Random(1729).shuffle(jobs)
    jobs.sort(key=lambda job: systems[job['system_id']]['trace_dofs'])
    source_files = [Path(__file__), _campaign_source_path(Path('hdgfem-gmres/scripts') / 'adr_native_hp_worker.py'),
                    _campaign_source_path(Path('hdgfem-gmres/scripts') / 'adv_diff_rea_cases.py'),
                    _campaign_source_path(Path('hdgfem-gmres/scripts') / 'oscillatory_adr_cases.py'),
                    _campaign_source_path(Path('hdgfem-gmres/scripts') / 'adr_performance_common.py')]
    source_files += [MASTER / p for p in (
        'hdgfem/linalg/face_hp_krylov.py', 'hdgfem/linalg/face_hp_multigrid.py',
        'hdgfem/backends/cupy_solver.py', 'hdgfem/backends/cupy_face_dense.py')]
    manifest = dict(version=1, systems=list(systems.values()), jobs=jobs,
                    source_artifacts=sources,
                    numerical_sources={str(p): digest(p) for p in source_files},
                    timeout=timeout, profiles=profiles,
                    timing='Fresh: setup plus first zero-guess solve; reused: mean second solve. Runtime/JIT warmed; assembly excluded.')
    return manifest


def load_result(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def run_job(job, output, timeout, resume, directory='jobs'):
    key, spec = job['key'], job['spec']
    destination = Path(spec['result'])
    specpath = output / ('profile_specs' if directory == 'profiles' else 'specs') / (key + '.json')
    if specpath.exists() and read_json(specpath) != spec:
        raise ValueError(f'Changed job specification: {key}')
    atomic_json(specpath, spec)
    previous = load_result(destination)
    if resume and previous.get('status') in TERMINAL:
        return previous, True
    atomic_json(destination, dict(status='running', system_id=job['system_id']))
    logpath = output / 'logs' / (key + '.log')
    with logpath.open('a') as log:
        process = subprocess.Popen(
            [sys.executable, '-m', 'scripts.adr_native_hp_worker', '--spec', str(specpath),
             '--master-root', str(MASTER)], cwd=BRANCH, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        interrupted = False
        try:
            code = process.wait(timeout=timeout)
            record = load_result(destination)
            if record.get('status') == 'running' or not record:
                record.update(status='process_error', returncode=code)
            elif code != 0 and record.get('status') == 'passed':
                record.update(status='process_error', returncode=code)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            try:
                signal_number = signal.SIGTERM
                import os
                os.killpg(process.pid, signal_number)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            record = load_result(destination)
            record.update(status='interrupted' if interrupted else 'timeout', timeout_seconds=timeout)
        record.update(system_id=job['system_id'], policy=spec['policy'], case=spec['case'],
                      n=spec['n'], p=spec['p'], family='native_hp', candidate=spec['candidate'],
                      requested_repeats=spec['repeats'], requested_warmups=spec['warmup'],
                      solves_per_setup=spec['solves_per_setup'], result_path=str(destination))
        if record.get('status') == 'passed' and spec.get('profile') and not record.get('profile_validation', {}).get('passed'):
            record.update(status='validation_failure', validation_errors=['instrumented solve failed acceptance'])
        if record.get('status') == 'passed' and not spec.get('profile'):
            issues = []
            if metrics(record) is None:
                issues.append('incomplete or invalid measured samples')
            if record.get('operator_sha256') != job['system_id']:
                issues.append('operator/RHS identity mismatch')
            if issues:
                record.update(status='validation_failure', validation_errors=issues)
        atomic_json(destination, record)
        if interrupted:
            raise KeyboardInterrupt
    return record, False


def save_progress(manifest, output, records, status):
    systems = {s['system_id']: s for s in manifest['systems']}
    rows = []
    for record in records:
        entry = systems[record['system_id']]
        rows.append(dict(record, geometry=entry['geometry'], triangles=entry['triangles'],
                         trace_dofs=entry['trace_dofs'], origins=entry['origins']))
    atomic_json(output / 'summary.json', rows)
    fields = ('system_id', 'geometry', 'case', 'n', 'p', 'triangles', 'trace_dofs',
              'policy', 'status', 'setup_median_ms', 'fresh_median_ms',
              'hot_mean_ms', 'hot_min_ms', 'hot_max_ms', 'iterations', 'residual')
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k) for k in fields}
            flat.update(metrics(row) or {})
            writer.writerow(flat)
    all_rows = []
    coverage = []
    for entry in systems.values():
        existing = [dict(x['record'], system_id=entry['system_id'], policy=x['policy'],
                         source_artifact=x['source'], origins=entry['origins']) for x in entry['existing']]
        added = [r for r in rows if r['system_id'] == entry['system_id']]
        candidates = existing + added
        all_rows.extend(candidates)
        winners = {metric: best(candidates, metric) for metric in ('fresh_median_ms', 'hot_mean_ms')}
        coverage.append(dict(system_id=entry['system_id'], case=entry['case'],
            geometry=entry['geometry'], p=entry['p'], trace_dofs=entry['trace_dofs'],
            origins=entry['origins'], attempted=bool(candidates),
            passed=any(metrics(r) is not None for r in candidates),
            policies={r.get('policy', 'standard'): r['status'] for r in candidates},
            winners={metric: ({'policy': row['policy'], **metrics(row)} if row else None)
                     for metric, row in winners.items()}))
    atomic_json(output / 'all_native_results.json', all_rows)
    atomic_json(output / 'coverage.json', coverage)
    counts = Counter(r['status'] for r in rows)
    completion = dict(status=status, scheduled=len(manifest['jobs']), attempted=len(rows),
                      outcomes=dict(counts), systems=len(systems),
                      native_attempted_systems=sum(c['attempted'] for c in coverage),
                      native_passed_systems=sum(c['passed'] for c in coverage),
                      note='Completed means attempted, not necessarily converged. Numerical failures, setup errors and timeouts remain separate.')
    atomic_json(output / 'completion.json', completion)
    return rows, completion


def profile_targets(manifest):
    for entry in manifest['systems']:
        if 'common_spec' not in entry:
            continue
        regular = any(o['suite'] in ('smooth', 'oscillatory', 'focused_scaling') and
                      o['nominal_n'] == 223 and entry['p'] == 6 for o in entry['origins'])
        endpoint = any(o['suite'] == 'focused_endpoints' for o in entry['origins'])
        stress = entry['case'] == 'cellular7_anisotropic' and entry['triangles'] == 22825
        if regular or endpoint or stress:
            yield entry


def run_profiles(manifest, output, rows, resume):
    results = []
    for entry in profile_targets(manifest):
        candidates = [r for r in rows if r['system_id'] == entry['system_id']]
        candidates += [dict(e['record'], policy=e['policy']) for e in entry['existing']]
        winner = best(candidates, 'hot_mean_ms')
        if winner is None:
            results.append(dict(system_id=entry['system_id'], status='skipped', reason='No passing native policy'))
            atomic_json(output / 'profiles.json', results)
            continue
        # The directional endpoint profiles already use the robust policy.
        reuse_name = 'native_hp_robust_square_profile.json' if entry['geometry'] == 'square' else 'native_hp_robust_profile.json'
        reuse = STUDY / 'oscillatory/asm_optimality_audit' / reuse_name
        old = load_result(reuse)
        if (winner['policy'] == 'robust' and old.get('operator_sha256') == entry['system_id'] and
                old.get('profile_validation', {}).get('passed') and old.get('internal_rtol') == 1e-11):
            results.append(dict(system_id=entry['system_id'], policy='robust', status='passed',
                                source=str(reuse), reused=True, record=old))
        else:
            key = f"profile_{entry['geometry']}_{entry['case']}_p{entry['p']}_{entry['system_id'][:12]}_{winner['policy']}"
            spec = dict(entry['common_spec'], policy=winner['policy'], candidate='native_hp_' + winner['policy'],
                        warmup=1, repeats=1, profile=True,
                        result=str(output / 'profiles' / (key + '.json')))
            record, reused = run_job(dict(key=key, system_id=entry['system_id'], spec=spec),
                                      output, manifest['timeout'], resume, 'profiles')
            results.append(dict(system_id=entry['system_id'], policy=winner['policy'],
                                status=record['status'], record=record, reused=reused))
        atomic_json(output / 'profiles.json', results)
        print('Profile', entry['geometry'], entry['case'], entry['p'], results[-1]['status'], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--timeout', type=float, default=900.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='read-only plan; no GPU imports or writes')
    parser.add_argument('--skip-profiles', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    output = args.output.resolve()
    manifest = prepare(output, args.timeout, not args.skip_profiles)
    print('Plan: 101 distinct systems; 61 previously untested; 126 new native timing jobs.', flush=True)
    print('Policies: standard and robust. Fresh and reused times are recorded separately.', flush=True)
    print('Output:', output, flush=True)
    if args.dry_run:
        for job in manifest['jobs']:
            print(job['key'])
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another completion driver already owns this output directory')
        destination = output / 'manifest.json'
        if destination.exists():
            if not args.resume or read_json(destination) != manifest:
                parser.error('Existing or changed campaign: resume requires identical inputs and numerical sources')
        else:
            atomic_json(destination, manifest)
        for directory in ('jobs', 'specs', 'logs', 'profiles', 'profile_specs', 'source_snapshot'):
            (output / directory).mkdir(exist_ok=True)
        for filename in manifest['numerical_sources']:
            path = Path(filename)
            prefix = 'master' if path.is_relative_to(MASTER) else 'branch'
            relative = path.relative_to(MASTER if prefix == 'master' else BRANCH)
            target = output / 'source_snapshot' / prefix / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
        records = []
        started = time.monotonic()
        executed = 0
        save_progress(manifest, output, records, 'running')
        try:
            for index, job in enumerate(manifest['jobs'], 1):
                record, reused = run_job(job, output, args.timeout, args.resume)
                records.append(record)
                executed += not reused
                rows, completion = save_progress(manifest, output, records, 'running')
                measured = metrics(record)
                detail = (f"fresh={measured['fresh_median_ms']:.3f} ms reused={measured['hot_mean_ms']:.3f} ms "
                          f"iterations={measured['iterations']:g}" if measured else record.get('error', ''))
                print(f"[{index}/126] {job['key']} {record['status']} {detail}" + (' (reused)' if reused else ''), flush=True)
                if index % 5 == 0:
                    mean_seconds = (time.monotonic() - started) / max(executed, 1)
                    remaining = sum(not (args.resume and load_result(j['spec']['result']).get('status') in TERMINAL)
                                    for j in manifest['jobs'][index:])
                    print(f"Recap: {completion['outcomes']}; native passes on {completion['native_passed_systems']}/101 systems; "
                          f"remaining timing estimate {remaining*mean_seconds/60:.1f} min (profiles excluded).", flush=True)
            rows, completion = save_progress(manifest, output, records, 'profiling' if manifest['profiles'] else 'completed')
            if manifest['profiles']:
                profiles = run_profiles(manifest, output, rows, args.resume)
                completion['profiles'] = dict(Counter(r['status'] for r in profiles))
            validation = dict(
                status='passed' if len(records) == 126 and completion['native_attempted_systems'] == 101 and
                    not any(r['status'] == 'validation_failure' for r in records) else 'failed',
                scheduled=126, attempted=len(records),
                all_systems_attempted=completion['native_attempted_systems'] == 101,
                native_passed_systems=completion['native_passed_systems'],
                numerical_failures=sum(r['status'] == 'numerical_failure' for r in records),
                execution_errors=[dict(system_id=r['system_id'], policy=r['policy'], status=r['status'])
                    for r in records if r['status'] not in ('passed', 'numerical_failure')],
                worst_physical_relative_residual=max((metrics(r)['residual'] for r in records if metrics(r)), default=None),
                worst_peer_trace_relative_difference=max((solve['peer_trace_relative_difference']
                    for r in records if metrics(r) for sample in r['samples'] for solve in sample['solves']
                    if solve.get('peer_trace_relative_difference') is not None), default=None),
                note='Worker physical residuals and manufactured-solution errors are evaluated outside timing. Peer differences are diagnostics, not a replacement for the physical residual.')
            if validation['execution_errors']:
                validation['status'] = 'needs_review'
            atomic_json(output / 'validation.json', validation)
            completion['status'] = 'completed'
            completion['validation'] = validation['status']
            atomic_json(output / 'completion.json', completion)
            print('Completed:', completion, flush=True)
        except KeyboardInterrupt:
            save_progress(manifest, output, records, 'interrupted')
            print('Interrupted safely. Repeat the same command with --resume.', flush=True)
            raise SystemExit(130)


if __name__ == '__main__':
    main()
