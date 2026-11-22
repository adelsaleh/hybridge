#!/usr/bin/env python3
"""Per-job solver performance table for the unified H200 campaign (read-only).

Writes jobs.csv (one row per finished job) and prints aggregate tables:
  * status by level and family;
  * per family x level: median setup / cold / repeated / amortized, iterations, ms per iteration,
    ns per iteration per trace DOF, warm ms/its per delta;
  * best candidate per system (amortized, among converged);
  * H200 vs Blackwell (scaling.csv) on matched case/p/triangles/candidate alias;
  * non-convergence by case x family.
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
import os
from pathlib import Path

ROOT = Path(os.environ.get('HYBRIDGE_SCRATCH', 'run_outputs'))/'adr_unified_l5'
BLACKWELL = Path(__file__).resolve().parents[2]/'docs/research/solver_studies/adr_scaling_2026_09_17/scaling.csv'
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
DELTAS = (0.0001, 0.001, 0.01)


def med(values):
    values = [v for v in values if v is not None]
    return st.median(values) if values else None


def candidates():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    table = {}
    for system in manifest['plan']['systems']:
        for c in system.get('candidates', []):
            table[c['id']] = c
    return table


def job_row(record, cands):
    key = Path(record['log']).parent.parent.name if record.get('log') else None
    cand = cands.get(record.get('candidate'), {})
    row = dict(key=key, level=record.get('level'), triangles=record.get('triangles'), trace_dofs=record.get('trace_dofs'),
               case=record.get('case'), p=record.get('p'), family=record.get('family') or cand.get('solver', {}).get('family'),
               candidate=record.get('candidate'), aliases='|'.join(cand.get('aliases', [])), status=record['status'],
               setup_ms=record.get('setup_median_ms'), cold_ms=record.get('fresh_setup_solve_median_ms'),
               repeated_ms=record.get('hot_solve_median_ms'), amortized_ms=record.get('amortized_median_ms'),
               residual=record.get('worst_true_relative_residual'), l2=record.get('l2_error'),
               wall_s=record.get('runner_wall_seconds'))
    try:
        result = json.loads((ROOT / 'jobs' / key / 'result.json').read_text())
    except (OSError, ValueError, TypeError):
        result = {}
    samples = result.get('samples') or []
    solves = [s for smp in samples for s in smp.get('solves', [])]
    row['iterations'] = med([s.get('iterations') for s in solves])
    row['first_solve_ms'] = med([smp['solves'][0]['solve_ms'] for smp in samples if smp.get('solves')])
    if row['iterations'] and row['repeated_ms']:
        row['ms_per_iter'] = row['repeated_ms'] / row['iterations']
        if row['trace_dofs']:
            row['ns_per_iter_dof'] = row['ms_per_iter'] * 1e6 / row['trace_dofs']
    for d in DELTAS:
        warm = [w for smp in samples for w in (smp.get('warm') or []) if abs(w.get('delta', -1) - d) < d * 1e-6]
        row[f'warm_{d:g}_ms'] = med([w.get('solve_ms') for w in warm])
        row[f'warm_{d:g}_its'] = med([w.get('iterations') for w in warm])
    return row


def fmt(v, digits=1):
    if v is None:
        return '–'
    if isinstance(v, float):
        return f'{v:.{digits}f}' if abs(v) < 1e5 else f'{v:.3g}'
    return str(v)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cands = candidates()
    records = json.loads((ROOT / 'summary.json').read_text())
    rows = [job_row(r, cands) for r in records]
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (OUT / 'jobs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f'## Status by triangles x family ({len(rows)} jobs)')
    status = defaultdict(lambda: defaultdict(int))
    for r in rows:
        status[(r['triangles'], r['family'])][r['status']] += 1
    for k in sorted(status, key=lambda k: (k[0] or 0, k[1] or '')):
        print(k, dict(status[k]))

    print('\n## Converged jobs: medians per triangles x family (ms)')
    print('tri | family | n | setup | cold | repeated | amortized | its | ms/it | ns/it/dof | warm 1e-4 | warm 1e-3 | warm 1e-2')
    groups = defaultdict(list)
    for r in rows:
        if r['status'] == 'passed':
            groups[(r['triangles'], r['family'])].append(r)
    for k in sorted(groups, key=lambda k: (k[0] or 0, k[1] or '')):
        g = groups[k]
        m = lambda f: med([r.get(f) for r in g])
        warm = ' | '.join(f"{fmt(m(f'warm_{d:g}_ms'))}/{fmt(m(f'warm_{d:g}_its'), 0)}" for d in DELTAS)
        print(f"{k[0]} | {k[1]} | {len(g)} | {fmt(m('setup_ms'))} | {fmt(m('cold_ms'))} | {fmt(m('repeated_ms'))} | "
              f"{fmt(m('amortized_ms'))} | {fmt(m('iterations'), 0)} | {fmt(m('ms_per_iter'), 3)} | {fmt(m('ns_per_iter_dof'), 3)} | {warm}")

    print('\n## Best converged candidate per system (amortized), wins by family')
    best = {}
    for r in rows:
        if r['status'] == 'passed' and r['amortized_ms']:
            sid = (r['triangles'], r['case'], r['p'])
            if sid not in best or r['amortized_ms'] < best[sid]['amortized_ms']:
                best[sid] = r
    wins = defaultdict(lambda: defaultdict(int))
    for (tri, _, _), r in best.items():
        wins[tri][r['family']] += 1
    for tri in sorted(wins, key=lambda t: t or 0):
        print(tri, dict(wins[tri]))

    print('\n## H200 vs Blackwell (scaling.csv), matched case/p/triangles/alias, passed on both')
    bw = {}
    with BLACKWELL.open() as f:
        for b in csv.DictReader(f):
            if b['status'] == 'passed':
                bw[(b['case'], int(b['p']), int(b['triangles']), b['candidate'])] = b
    print('case | p | tri | candidate | its H200/BW | setup H200/BW | repeated H200/BW | speed-up repeated | speed-up amortized')
    for r in rows:
        if r['status'] != 'passed':
            continue
        for alias in r['aliases'].split('|'):
            b = bw.get((r['case'], r['p'], r['triangles'], alias))
            if b and r['repeated_ms']:
                print(f"{r['case']} | {r['p']} | {r['triangles']} | {alias} | {fmt(r['iterations'], 0)}/{b['iterations']} | "
                      f"{fmt(r['setup_ms'])}/{float(b['setup']):.1f} | {fmt(r['repeated_ms'])}/{float(b['mean']):.1f} | "
                      f"{float(b['mean']) / r['repeated_ms']:.2f}x | {float(b['amortized']) / r['amortized_ms']:.2f}x")

    print('\n## Not converged by case x family')
    nc = defaultdict(lambda: defaultdict(int))
    for r in rows:
        nc[r['case']][r['family']] += r['status'] == 'not_converged'
    for case in sorted(nc):
        hits = {f: n for f, n in nc[case].items() if n}
        if hits:
            print(case, hits)


if __name__ == '__main__':
    main()
