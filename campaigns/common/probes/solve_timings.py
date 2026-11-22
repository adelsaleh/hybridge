"""Tabulate solve timings from closed-loop stress campaign output directories.

usage: python solve_timings.py <stress output dir> [...]
Run from the hybridge checkout so campaigns.common.reporting imports.
"""
import glob
import json
import os
import statistics
import sys

from campaigns.common.reporting import convergence_status


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def fmt(value, spec='{:.1f}'):
    return spec.format(value) if isinstance(value, (int, float)) else '-'


def report(directory):
    meshes = {}
    for path in glob.glob(f'{directory}/meshes/**/mesh.json', recursive=True):
        meshes[path] = json.load(open(path)).get('triangles')
    print(f'\n## {directory}  (mesh triangles: {sorted(set(meshes.values()))})')
    pardiso = {}
    for path in glob.glob(f'{directory}/jobs/pardiso_*.json'):
        row = json.load(open(path))
        pardiso[int(path.rsplit('_p', 1)[1][:-5])] = row
    rows = []
    for path in sorted(glob.glob(f'{directory}/jobs/stress*.json')):
        if path.endswith('.solution.npy'):
            continue
        result = json.load(open(path))
        spec = json.load(open(path.replace('/jobs/', '/specs/')))
        samples = result.get('samples') or result.get('warmups') or []
        solves = [solve for sample in samples for solve in sample.get('solves', [])]
        last = solves[-1] if solves else {}
        rows.append(dict(p=spec.get('p'), candidate=spec.get('candidate') or os.path.basename(path)[:-5],
                         status=convergence_status(result, spec), iterations=last.get('iterations'),
                         relres=last.get('true_relative_residual'),
                         setup=median(s.get('setup_ms') for s in samples),
                         first=median(s['solves'][0].get('solve_ms') for s in samples if s.get('solves')),
                         reused=median(s['solves'][1].get('solve_ms') for s in samples if len(s.get('solves', [])) > 1),
                         fresh=median(s.get('fresh_setup_solve_ms') for s in samples)))
    for p in sorted({row['p'] for row in rows}):
        lu = pardiso.get(p, {})
        timings = lu.get('timings_ms', {})
        print(f'\np={p}  trace DOFs={lu.get("dofs", "-")}  nnz={lu.get("nnz", "-")}  '
              f'PyPardiso: factor {fmt(timings.get("analysis_factorization"))} ms, '
              f'solve {fmt(timings.get("solve"))} ms, reused solve {fmt(timings.get("probe_reused_solve"))} ms, '
              f'factor+solve {fmt(timings.get("setup_solve"))} ms ({lu.get("status", "not run")})')
        print(f'{"candidate":34} {"status":14} {"iters":>6} {"relres":>8} {"setup":>8} {"1st solve":>9} '
              f'{"fresh":>9} {"reused":>8} {"vs LU fresh":>11} {"vs LU reused":>12}')
        for row in sorted((r for r in rows if r['p'] == p), key=lambda r: (r['status'] != 'passed', r['fresh'] or 1e99)):
            fresh_ratio = row['fresh'] / timings['setup_solve'] if row['fresh'] and timings.get('setup_solve') else None
            reused_lu = timings.get('probe_reused_solve')
            reused_ratio = row['reused'] / reused_lu if row['reused'] and reused_lu else None
            print(f'{row["candidate"]:34} {row["status"]:14} {str(row["iterations"]):>6} {fmt(row["relres"], "{:.1e}"):>8} '
                  f'{fmt(row["setup"]):>8} {fmt(row["first"]):>9} {fmt(row["fresh"]):>9} {fmt(row["reused"]):>8} '
                  f'{fmt(fresh_ratio, "{:.2f}x"):>11} {fmt(reused_ratio, "{:.2f}x"):>12}')


if __name__ == '__main__':
    for directory in sys.argv[1:]:
        report(directory.rstrip('/'))
