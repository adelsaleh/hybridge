#!/usr/bin/env python3
"""Summarize consecutive-block ILU trials, ordering, and detailed timings."""
import argparse
import json
from pathlib import Path
import re
from common import write_report


def collect(root, job, prefix='ordered-block-ilu'):
    directory = root / f'{prefix}-{job}'
    aggregate = directory / 'full.json'
    if not aggregate.exists():
        smoke = directory / 'smoke.json'
        if smoke.exists():
            parent = json.loads(smoke.read_text())
            return dict(job=job, status=parent.get('status', 'smoke_pending'), stage='smoke', report=str(smoke))
        return dict(job=job, status='no_full_report')
    parent = json.loads(aggregate.read_text())
    if parent.get('results'):
        d = parent['results'][0]
        path = Path(d['report'])
    else:
        paths = list((directory / 'full_trials').glob('*.json'))
        if not paths:
            return dict(job=job, status=parent['status'])
        path = paths[0]
        d = json.loads(path.read_text())
    samples = d.get('samples', [])
    sample = samples[0] if samples else {}
    step = sample.get('passes', [{}])[0]
    log = path.with_suffix('.log').read_text() if path.with_suffix('.log').exists() else ''
    native_errors = re.findall(r'Caught amgx exception: ([^\n]+)', log)
    peaks = re.findall(r'sampled solve peak used=([\d.]+) held=([\d.]+)', log)
    symbolic = re.findall(r'Natural ILU\((\d+)\): input blocks=(\d+), factor blocks=(\d+), fill ratio=([\d.]+), factor values=([\d.]+) GiB, host symbolic=([\d.]+) s', log)
    timing = {key: d.get(key) for key in ('fingerprint_seconds', 'scaling_seconds',
        'block_preparation_seconds', 'preparation_seconds', 'backend_initialization_seconds',
        'upload_seconds', 'setup_seconds', 'cleanup_seconds', 'worker_wall_seconds')}
    timing['matrix_load_seconds'] = d.get('matrix', {}).get('load_seconds')
    timing['solve_seconds'] = sample.get('solve_seconds')
    ordering = d.get('ordering', {})
    timing['saved_pardiso_analysis_seconds'] = ordering.get('metadata', {}).get('analysis_seconds')
    timing['permutation_application_seconds'] = ordering.get('application_seconds')
    timing['permutation_validation_seconds'] = ordering.get('validation_seconds')
    for key in ('rhs_reset_seconds', 'solution_recovery_seconds', 'original_residual_check_seconds'):
        timing[key] = step.get(key)
    cfg = d['specification']['config']['solver']
    fill = cfg.get('preconditioner', {}).get('ilu_sparsity_level', 0)
    factor = None
    if symbolic:
        k,original,expanded,ratio,gib,seconds = symbolic[-1]
        factor = dict(fill_level=int(k), original_blocks=int(original), factor_blocks=int(expanded),
                      fill_ratio=float(ratio), factor_values_gib=float(gib), host_symbolic_seconds=float(seconds))
        timing['host_symbolic_seconds'] = float(seconds)
    return dict(job=job, status=d['status'], report=str(path), fill_level=fill, factor=factor,
                outer_solver='P'+cfg['solver'], native_tolerance=cfg.get('tolerance'),
                ordering=ordering.get('method', d['parameters'].get('ordering', 'none')),
                error=d.get('error') or (native_errors[-1] if native_errors else None),
                block_size=d['specification']['block_size'], scaling=d.get('equilibration', {}).get('method', d['parameters']['scaling']),
                original_relative_residual=sample.get('relative_residual'), passed=sample.get('passed'),
                iterations=step.get('iterations'), restart=cfg['gmres_n_restart'],
                native_final_residual=(step.get('native_residual_history') or [None])[-1],
                amgx_sampled_peak_used_gib=max(float(a) for a,b in peaks) if peaks else None,
                amgx_sampled_peak_held_gib=max(float(b) for a,b in peaks) if peaks else None,
                timing=timing, block_storage=d.get('block_storage'),
                permutation_sha256=ordering.get('metadata', {}).get('permutation_sha256'),
                library=d.get('amgx_library'), configuration_sha256=d.get('configuration_sha256'))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--jobs', nargs='+', type=int, required=True)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2]/'run_outputs/matk')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix', default='ordered-block-ilu', help='Run directory prefix; natural-iluk for higher-fill trials')
    args = p.parse_args()
    rows = [collect(args.root, j, args.prefix) for j in args.jobs]
    note = ('All residuals are independently evaluated in original coordinates. PARDISO analysis is a reused one-time cost; '
            'permutation application/validation, scaling and block conversion are nested in preparation and must not be added twice. '
            'Host symbolic time is included in setup. '
            'AMGX memory peaks are sampled process counters, not whole-device or guaranteed transient peaks. '
            'Concurrent search timings are not isolated performance benchmarks.')
    write_report(args.output, dict(note=note, results=rows))
    lines = ['Consecutive-block ILU(k) results', note, '',
             'Job    Order    Block ILU Scaling    Solver   R     Tol       Iters    Original rel.L2   Setup[s]   Solve[s]   AMGX peak GiB  Status']
    def fmt(x, form):
        return format(x, form) if isinstance(x, (int, float)) else '-'
    for r in rows:
        t = r.get('timing', {})
        lines.append(f"{r['job']:<7}{r.get('ordering','-'):<9}{r.get('block_size','-'):<6}{r.get('fill_level','-'):<4}{r.get('scaling','-'):<11}"
                     f"{r.get('outer_solver','-'):<9}{r.get('restart','-'):<6}{fmt(r.get('native_tolerance'),'.0e'):<10}"
                     f"{str(r.get('iterations') or '-'):>5}  {fmt(r.get('original_relative_residual'),'.8e'):>17}"
                     f"  {fmt(t.get('setup_seconds'),'.3f'):>9}  {fmt(t.get('solve_seconds'),'.3f'):>9}"
                     f"  {fmt(r.get('amgx_sampled_peak_used_gib'),'.3f'):>13}  {r['status']}")
    for r in rows:
        lines.extend(['', f"Job {r['job']} detailed timings (seconds):"])
        lines.extend(f'  {key:<40} {fmt(value,".6f")}' for key,value in r.get('timing', {}).items())
        if r.get('error'):
            lines.append(f"  Error: {r['error']}")
    text = '\n'.join(lines)+'\n'
    args.output.with_suffix('.txt').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
