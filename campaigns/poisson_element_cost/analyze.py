"""CPU-only summaries of the investigation's outputs. Prints Markdown, optionally writes compact CSVs.

``arms STAGE_DIR``      real-run arms: stage medians, pairing within each repetition, field parity
``spans STAGE_DIR``     per-operation attribution of RHS and reconstruction, per solve
``probe PROBE_JSON``    kernel variants: event time, resources, block cycles, parity
Reuses ``stage_values`` and ``field_difference`` of the research article campaign.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from guiding_center_cg_hdg.campaigns.gc_article_kernel_tuning.analyze import field_difference, stage_values

CLOCK_ORDER = ('entry', 'factor load', 'pivot load/setup', 'column build (trace lifts)', 'coupling',
               'triangular solve', 'q and outputs', 'flux', 'emit')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def arm_runs(stage):
    runs = defaultdict(lambda: defaultdict(dict))
    for marker in sorted(Path(stage).glob('*/*-rep*/DONE.json')):
        path = marker.parent
        arm, _, rep = path.name.rpartition('-rep')
        spec = json.loads((path/'spec.json').read_text())
        runs[path.parent.name][arm][int(rep)] = SimpleNamespace(path=path, spec=spec)
    return runs


def selected(path):
    result = {}
    for record in json.loads((path/'selected_kernels.json').read_text()):
        workload = record.get('workload') or {}
        if workload.get('family') == 'poisson':
            result[workload['phase']] = (record.get('candidates') or [{}])[0] and next(
                (c.get('configuration') for c in record.get('candidates', []) if c.get('variant') == record.get('variant')), None)
    return result


ACCURACY = ('rho_l2_error', 'phi_l2_error', 'rho_gradient_l2_error', 'phi_gradient_l2_error', 'phi_trace_l2_error')


def retained_rows(run):
    with (run.path/'measurement_timings.jsonl').open() as stream:
        return [row for row in map(json.loads, filter(str.strip, stream))
                if row.get('phase') == 'step' and row['step'] > run.spec['discard_steps']]


def stage_detail(run):
    """Per-solve medians of the last-stage timers, and stored bytes."""
    rows = retained_rows(run)
    def median(key):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        return statistics.median(values) if values else None
    return dict(rhs_ms=1e3*median('poisson_time_rhs_assembly'), reconstruction_ms=1e3*median('poisson_time_reconstruction'),
                rhs_device_ms=1e3*median('poisson_detail_raw_assembly_raw_kernel_device'),
                local_factor_gb=(median('poisson_detail_raw_assembly_raw_local_factors_bytes') or 0)/1e9,
                postprocess_ms=1e3*(median('poisson_time_postprocessing') or 0),
                primal_gb=(median('poisson_detail_postprocess_primal_cache_bytes') or 0)/1e9,
                primal_operator=median('poisson_detail_postprocess_primal_operator'),
                rhs_threads=median('poisson_detail_raw_assembly_raw_block_size'))


def accuracy_change(run, other):
    """Largest relative change of the manufactured-solution errors against the same repetition of the reference."""
    a, b = (json.loads((r.path/'measurement.json').read_text()).get('accuracy') or {} for r in (run, other))
    return max((abs(a[k]-b[k])/abs(b[k]) for k in ACCURACY if b.get(k)), default=None)


def arms(stage, reference='fixed', write=None):
    table = []
    for row, row_arms in arm_runs(stage).items():
        print(f'\n### {row}\n')
        print('| arm | step ms | RHS ms/solve | recon ms/solve | RHS device ms | stored GB | RHS threads | Poisson its/step | '
              'solve ms/step | step saved vs fixed (paired mean, min..max) | field L2 vs fixed | accuracy change |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        base = {rep: stage_values(run) for rep, run in row_arms[reference].items()}
        reps_reference = row_arms[reference]
        floor = []
        keys = sorted(reps_reference)
        for i, left in enumerate(keys):
            for right in keys[i+1:]:
                if (reps_reference[left].path/'final_fields.npz').exists():
                    floor.append(max(field_difference(reps_reference[left].path, reps_reference[right].path).values()))
        print(f'Reference against itself across repetitions: field L2 up to {max(floor) if floor else float("nan"):.2e}.\n')
        for arm, reps in row_arms.items():
            values = {rep: stage_values(run) for rep, run in reps.items()}
            details = [stage_detail(run) for run in reps.values()]
            mean = lambda key: statistics.fmean(v[key] for v in values.values())
            detail = lambda key: statistics.fmean(d[key] for d in details)
            deltas = [base[rep]['step']-values[rep]['step'] for rep in values if rep in base]
            shares = [(base[rep]['step']-values[rep]['step'])/base[rep]['step'] for rep in values if rep in base]
            parity, accuracy = [], []
            for rep, run in reps.items():
                other = reps_reference.get(rep)
                if other is not None and (run.path/'final_fields.npz').exists() and (other.path/'final_fields.npz').exists():
                    parity.append(max(field_difference(run.path, other.path).values()))
                    accuracy.append(accuracy_change(run, other))
            worst = max(parity) if parity else None
            drift = max(accuracy) if accuracy else None
            print(f"| {arm} | {1e3*mean('step'):.2f} | {detail('rhs_ms'):.3f} | {detail('reconstruction_ms'):.3f} | {detail('rhs_device_ms'):.3f} | "
                  f"{detail('local_factor_gb'):.2f} | {detail('rhs_threads'):.0f} | {mean('poisson_iterations'):.1f} | {1e3*mean('poisson_solve'):.2f} | "
                  f"{1e3*statistics.fmean(deltas):.2f} ms = {100*statistics.fmean(shares):.1f}% ({1e3*min(deltas):.2f}..{1e3*max(deltas):.2f}) | "
                  f"{worst if worst is None else f'{worst:.2e}'} | {drift if drift is None else f'{drift:.1e}'} |")
            table.append(dict(row=row, arm=arm, reps=len(reps), step_ms=1e3*mean('step'), rhs_ms_per_solve=detail('rhs_ms'),
                              reconstruction_ms_per_solve=detail('reconstruction_ms'), rhs_device_ms=detail('rhs_device_ms'),
                              local_factor_gb=detail('local_factor_gb'), rhs_threads=detail('rhs_threads'),
                              postprocess_ms_per_solve=detail('postprocess_ms'), primal_gb=detail('primal_gb'),
                              primal_operator=details[0]['primal_operator'],
                              poisson_assembly_ms_per_step=1e3*mean('poisson_assembly'),
                              poisson_reconstruction_ms_per_step=1e3*mean('poisson_reconstruction'),
                              poisson_solve_ms_per_step=1e3*mean('poisson_solve'), transport_ms_per_step=1e3*mean('transport'),
                              poisson_iterations=mean('poisson_iterations'), step_saved_ms=1e3*statistics.fmean(deltas),
                              step_saved_percent=100*statistics.fmean(shares), step_saved_min_ms=1e3*min(deltas),
                              step_saved_max_ms=1e3*max(deltas), field_l2_vs_fixed=worst, accuracy_relative_change=drift,
                              reference_self_field_l2=max(floor) if floor else None,
                              selected=json.dumps(selected(next(iter(reps.values())).path), sort_keys=True)))
    if write:
        write_csv(write, table)
    return table


def spans(stage, write=None):
    table = []
    for path in sorted(Path(stage).glob('spans-*/spans.json')):
        data = json.loads(path.read_text())
        for root in ('RHS', 'RECONSTRUCT'):
            solves = data[root]['count']
            for key, value in data.items():
                if key == root or key.startswith(root+'/'):
                    table.append(dict(run=path.parent.name, stage=root, path=key.split('/', 1)[1] if '/' in key else '(total)',
                                      calls_per_solve=value['count']/solves, median_ms=value['median_ms'],
                                      trimmed_ms_per_solve=value['trimmed_mean_ms']*value['count']/solves))
    if write:
        write_csv(write, table)
    print(f'{len(table)} span rows')
    return table


def probe(path, write=None, phase=None):
    rows = json.loads(Path(path).read_text())
    print('| variant | phase | threads | regs | shared KiB | blocks/SM | occupancy | device ms | block cycles | parity |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    flat = []
    for row in rows:
        if phase and row['phase'] != phase:
            continue
        attributes = row.get('attributes') or {}
        cycles = row.get('clock_cycles_per_block_total')
        print(f"| {row['variant']} | {row['phase']} | {row['threads']} | {attributes.get('num_regs')} | "
              f"{(row.get('shared_bytes') or 0)/1024:.1f} | {row.get('blocks_per_sm')} | {row.get('occupancy') or 0:.3f} | "
              f"{row.get('device_ms_median', float('nan')):.3f} | {'' if cycles is None else f'{cycles:.0f}'} | {row.get('parity_relative_max')} |")
        flat.append(dict({k: row.get(k) for k in ('variant', 'phase', 'threads', 'tuned_threads', 'shared_bytes', 'blocks_per_sm', 'occupancy',
                                                  'device_ms_median', 'device_ms_min', 'device_ms_max', 'parity_relative_max',
                                                  'against_last_production_output', 'clock_cycles_per_block_total', 'status')},
                         num_regs=attributes.get('num_regs'), **{f'cycles {k}': (row.get('clock_cycles_per_block') or {}).get(k) for k in CLOCK_ORDER}))
    if write:
        write_csv(write, flat)
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=('arms', 'spans', 'probe'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--phase')
    parser.add_argument('--reference', default='fixed', help='arm the arms table pairs against')
    args = parser.parse_args()
    if args.kind == 'arms':
        arms(args.path, reference=args.reference, write=args.csv)
    elif args.kind == 'spans':
        spans(args.path, write=args.csv)
    else:
        probe(args.path, write=args.csv, phase=args.phase)


if __name__ == '__main__':
    main()
