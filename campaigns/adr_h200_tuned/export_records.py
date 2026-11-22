"""Compact, tracked records of the H200 tuned ADR campaign (read-only on the scratch outputs).

The raw per-job results (about 40 MB, residual histories included) stay on scratch; this writes
one row per solver job and one per system to ``records/``, with the medians the reports use:
setup, cold (setup + first solve), repeated (second solve on the same setup), amortized
(setup + both solves, per solve), iterations of the first solve, and the warm series per delta
(median solve time and iterations), and for every job its first solve's accuracy and the point
where that solve reached its lowest residual (``first_floor_*``, :func:`floor_point`). The same rows are derived from the stock-kernel H200 run
(unified campaign 21958) for the systems and base candidates this campaign shares with it, and for
every record of that run (``stock_unified_all.csv``, with its own plan's systems). Each
system row names its exported host system (``cache``, the PyPardiso reference input) and that
export's operator hash.

usage: python -m campaigns.adr_h200_tuned.export_records [--output-root DIR] [--unified DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDS = Path(__file__).resolve().parents[2]/'docs/campaigns/adr_h200_tuned/records'
OUTPUTS = ('50-l1-l5', '51-l6-l7', '70-l8-l9', '82-lmax-p2-p5', '83-lmax-p3-p7', '84-lmax-p4-p8-p9',
           '85-l4-p3-p7', '86-l4-p4-p8', '87-l4-p5')
DELTAS = (1e-4, 1e-3, 1e-2)
FIELDS = ('run', 'level', 'geometry', 'case', 'p', 'triangles', 'trace_dofs', 'system_id', 'family', 'candidate',
          'base_candidate', 'aliases', 'status', 'iterations', 'setup_ms', 'cold_ms', 'repeated_ms', 'amortized_ms',
          *(f'warm_{d:g}_{k}' for d in DELTAS for k in ('ms', 'its')),
          'l2_error', 'true_relative_residual', 'first_l2_error', 'first_true_relative_residual', 'first_solver_status',
          'first_iterations', 'first_floor_iterations', 'first_floor_bound', 'first_floor_relative_residual',
          'first_floor_solve_ms', 'first_floor_fresh_ms', 'pp_degree', 'policy', 'bsr_borrowed', 'gpu_sm_busy_mean',
          'samples')


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _cycle_lengths(estimated, restart, jump=1.5):
    """Iterations per restart cycle from the GMRES estimated-residual history ``[beta, e_1..e_k]`` per
    cycle: a cycle starts at a jump of the estimate or after ``restart`` iterations."""
    lengths, k = [], -1
    for i, value in enumerate(estimated):
        if k < 0 or k >= restart or value > jump*estimated[i-1]:
            if k >= 0:
                lengths.append(k)
            k = 0
        else:
            k += 1
    return lengths+[k]


def floor_point(record, solve, setup_ms, factor=2.):
    """Where a recorded solve first came within ``factor`` of its lowest true residual.

    Native GMRES records the true residual at every restart-cycle end; the iterations up to that
    cycle come from the cycle lengths, exact when they reproduce the recorded cycle and iteration
    counts, otherwise bounded above (``bound``: every later cycle takes at least one iteration and no
    cycle more than the restart length). AMGX stops by itself, so its whole solve counts. Times scale
    the measured solve by the iteration fraction. ``None`` without a history.
    """
    history, iterations = solve.get('residual_history'), solve.get('iterations')
    if not history or len(history) < 2 or not iterations or solve.get('solve_ms') is None:
        return None
    if solve.get('estimated_preconditioned_residual_history') is None:
        reached, bound, index = iterations, False, len(history)-1
    else:
        lowest = min(history[1:])
        index = next(i for i in range(1, len(history)) if history[i] <= factor*lowest)
        restart = (record.get('configuration') or {}).get('restart') or record.get('restart')
        lengths = _cycle_lengths(solve['estimated_preconditioned_residual_history'], restart)
        bound = not (len(lengths) == len(history)-1 and abs(sum(lengths)-iterations) <= 1)
        reached = min(iterations, restart*index, iterations-(len(history)-1-index)) if bound \
            else min(iterations, sum(lengths[:index]))
    solve_ms = solve['solve_ms']*reached/iterations
    return dict(first_floor_iterations=reached, first_floor_bound=bound,
                first_floor_relative_residual=solve.get('true_relative_residual') if index == len(history)-1
                else history[index]/history[0],
                first_floor_solve_ms=solve_ms, first_floor_fresh_ms=setup_ms+solve_ms if setup_ms is not None else None)


def solver_row(record, system, candidate, run):
    """One solver job as medians over its measured samples (warm-ups excluded)."""
    samples = record.get('samples') or []
    solves = [s.get('solves') or [] for s in samples]
    warm = {d: ([], []) for d in DELTAS}
    for sample in samples:
        for step in sample.get('warm') or []:
            if step.get('delta') in warm:
                warm[step['delta']][0].append(step.get('solve_ms'))
                warm[step['delta']][1].append(step.get('iterations'))
    stages = [s.get('setup_stages') or {} for s in samples]
    # The first recorded solve (warm-up included) also exists for jobs that did not pass, so the
    # accuracy actually reached is kept for every outcome.
    attempts = [x for g in (record.get('warmups') or []) + samples for x in (g.get('solves') or [])]
    first = attempts[0] if attempts else {}
    first_group = next((g for g in (record.get('warmups') or []) + samples if g.get('solves')), {})
    row = dict(first_l2_error=first.get('l2_error'), first_true_relative_residual=first.get('true_relative_residual'),
               first_solver_status=first.get('status'), first_iterations=first.get('iterations'),run=run, level=system.get('level'), geometry=system['geometry'], case=system['problem']['case'],
               p=system['problem']['p'], triangles=system.get('triangles') or record.get('triangles'),
               trace_dofs=system.get('trace_dofs') or record.get('trace_dofs'), system_id=system['id'],
               family=candidate['solver']['family'], candidate=candidate['id'],
               base_candidate=candidate.get('base_candidate', candidate['id']),
               aliases='|'.join(candidate.get('aliases', [])), status=record.get('status'),
               pp_degree=(candidate['solver'].get('configuration') or {}).get('polynomial_degree'),
               policy=candidate['solver'].get('policy'),
               iterations=_median([s[0].get('iterations') for s in solves if s]),
               setup_ms=_median([s.get('setup_ms') for s in samples]),
               cold_ms=_median([s.get('fresh_setup_solve_ms') for s in samples]),
               repeated_ms=_median([x.get('solve_ms') for s in solves for x in s[1:]]),
               amortized_ms=_median([s.get('amortized_ms') for s in samples]),
               l2_error=record.get('l2_error'), true_relative_residual=record.get('worst_true_relative_residual'),
               bsr_borrowed=_median([1 if st.get('bsr_borrowed') else 0 for st in stages if 'bsr_borrowed' in st]),
               gpu_sm_busy_mean=(record.get('gpu_utilization') or {}).get('sm_busy_mean'),
               samples=len(samples))
    # Jobs that did not pass have only their first solve: where it reached its lowest residual.
    if first:
        row.update(floor_point(record, first, first_group.get('setup_ms')) or {})
    for d, (ms, its) in warm.items():
        row[f'warm_{d:g}_ms'], row[f'warm_{d:g}_its'] = _median(ms), _median(its)
    return row


def exported_cache(root: Path, run: str, system_id: str):
    """The exported host system of ``system_id`` (its own run first, then any run that exported it)."""
    runs = [run] + sorted(d.name for d in root.iterdir() if d.is_dir() and d.name != run)
    for name in runs:
        cache = root/name/'cache'/system_id
        if (cache/'system_blocks.npy').exists() and (cache/'metadata.json').exists():
            return cache, json.loads((cache/'metadata.json').read_text()).get('operator_sha256')
    return None, None


def tuned_rows(root: Path, finished=True):
    """Rows of the finished tiers (``finished``) or of the tiers still running (preliminary)."""
    rows, systems_out = [], []
    for run in OUTPUTS:
        out = root/run
        # A tier still running would put partial levels into the reports; it is exported apart.
        if not (out/'plan.json').exists() or (out/'completion.json').exists() != finished:
            continue
        plan = json.loads((out/'plan.json').read_text())
        for system in plan['systems']:
            result = out/'jobs'/f"system_{system['id']}"/'result.json'
            mesh = out/'jobs'/f"mesh_{system['id']}"/'result.json'
            if mesh.exists() and not (system.get('triangles') and system.get('trace_dofs')):
                # Generated meshes: the plan leaves size open; the mesh job records it.
                m = json.loads(mesh.read_text())
                system = dict(system, triangles=system.get('triangles') or m.get('triangles'),
                              trace_dofs=system.get('trace_dofs') or m.get('trace_dofs'))
            if result.exists():
                r = json.loads(result.read_text())
                a = r.get('assembly') or {}
                cache, operator = exported_cache(root, run, system['id'])
                systems_out.append(dict(run=run, level=system['level'], geometry=system['geometry'],
                                        case=system['problem']['case'], p=system['problem']['p'],
                                        triangles=a.get('triangles') or system.get('triangles'),
                                        trace_dofs=a.get('trace_dofs') or system.get('trace_dofs'), system_id=system['id'],
                                        status=r.get('status'), error=(r.get('error') or '')[:160],
                                        device_assembly_ms=a.get('assembly_median_ms'),
                                        export_join_wait_s=(r.get('phases') or {}).get('export_join_wait'),
                                        cache=str(cache or ''), operator_sha256=operator or ''))
            for candidate in system['candidates']:
                path = out/'jobs'/f"{system['id']}_{candidate['id']}"/'result.json'
                if not path.exists():
                    continue
                record = json.loads(path.read_text())
                if candidate['solver']['family'] == 'raw_assembly':
                    systems_out.append(dict(run=run, level=system['level'], geometry=system['geometry'],
                                            case=system['problem']['case'], p=system['problem']['p'],
                                            triangles=record.get('triangles') or system.get('triangles'),
                                            trace_dofs=record.get('trace_dofs') or system.get('trace_dofs'),
                                            system_id=system['id'], status=record.get('status'),
                                            error=(record.get('error') or '')[:160], raw_cuda_assembly_ms=record.get('assembly_median_ms'),
                                            raw_cuda_peak_bytes=record.get('sampled_device_peak_bytes')))
                    continue
                rows.append(solver_row(record, system, candidate, run))
    return rows, systems_out


def stock_rows(unified: Path, shared):
    """Rows of the stock-kernel H200 run for the (system, base candidate) pairs shared with this campaign."""
    rows = []
    for record in json.loads((unified/'summary.json').read_text()):
        key = (record.get('system_id'), record.get('candidate'))
        if key not in shared:
            continue
        system, candidate = shared[key]
        stock = dict(candidate, id=record['candidate'], base_candidate=record['candidate'])
        rows.append(solver_row(record, system, stock, 'unified_21958'))
    return rows


def stock_all_rows(unified: Path):
    """Every solver record of the stock-kernel H200 run, with the systems and candidates of its own plan."""
    plan = json.loads((unified/'manifest.json').read_text())['plan']
    systems = {s['id']: s for s in plan['systems']}
    rows = []
    for record in json.loads((unified/'summary.json').read_text()):
        system = systems.get(record.get('system_id'))
        candidate = next((c for c in (system or {}).get('candidates', []) if c['id'] == record.get('candidate')), None)
        if system and candidate:
            rows.append(solver_row(record, system, candidate, 'unified_21958'))
    return rows


def write(path: Path, rows, fields=None):
    fields = fields or sorted({k for r in rows for k in r}, key=lambda k: (k not in FIELDS, FIELDS.index(k) if k in FIELDS else 0, k))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Since 2026-11-08 only outputs 82-87 remain under the default root; the other runs' records are in
    # run_outputs/campaigns/adr_h200_tuned (no caches, so the cache/operator columns would come out empty),
    # and adr_unified_l5's json/logs (for --unified) are in run_outputs/campaigns/adr_scratch_essentials_20261108.tar.gz.
    parser.add_argument('--output-root', type=Path, default=Path(os.environ.get('HYBRIDGE_SCRATCH', 'run_outputs'))/'adr_h200_tuned')
    parser.add_argument('--unified', type=Path, default=Path(os.environ.get('HYBRIDGE_SCRATCH', 'run_outputs'))/'adr_unified_l5')
    args = parser.parse_args(argv)
    RECORDS.mkdir(exist_ok=True)
    rows, systems = tuned_rows(args.output_root)
    unfinished, _ = tuned_rows(args.output_root, finished=False)
    shared = {}
    for run in OUTPUTS:
        plan_path = args.output_root/run/'plan.json'
        if plan_path.exists():
            for system in json.loads(plan_path.read_text())['systems']:
                for candidate in system['candidates']:
                    if candidate.get('base_candidate'):
                        shared[(system['id'], candidate['base_candidate'])] = (system, candidate)
    stock = stock_rows(args.unified, shared) if (args.unified/'summary.json').exists() else []
    stock_all = stock_all_rows(args.unified) if (args.unified/'manifest.json').exists() else []
    hashes = {'solver_jobs.csv': write(RECORDS/'solver_jobs.csv', rows, list(FIELDS)),
              'systems.csv': write(RECORDS/'systems.csv', systems),
              'stock_unified_rows.csv': write(RECORDS/'stock_unified_rows.csv', stock, list(FIELDS)),
              'stock_unified_all.csv': write(RECORDS/'stock_unified_all.csv', stock_all, list(FIELDS)),
              'solver_jobs_unfinished.csv': write(RECORDS/'solver_jobs_unfinished.csv', unfinished, list(FIELDS))}
    manifest = dict(sources=dict(tuned=str(args.output_root), outputs=list(OUTPUTS), stock=str(args.unified)),
                    counts=dict(solver_jobs=len(rows), systems=len(systems), stock_rows=len(stock),
                                stock_all_rows=len(stock_all), unfinished_solver_jobs=len(unfinished)), sha256=hashes,
                    note='Derived medians; raw per-job results stay on scratch (residual histories, samples).')
    (RECORDS/'export_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    main()
