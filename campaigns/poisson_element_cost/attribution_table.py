"""Group the span records into the per-solve table of FINDINGS (CPU only)."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

W = 'assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy'
R = 'assemble_projected_diffusion_trace_rhs_eliminated_raw_cuda'
RR = 'reconstruct_projected_diffusion_field_raw_cuda'


def per_solve(data, root):
    solves = data[root]['count']
    return {key: value['trimmed_mean_ms']*value['count']/solves for key, value in data.items()
            if key == root or key.startswith(root+'/')}


def launch_rows(values, launch):
    get = lambda suffix: values.get(f'{launch}/{suffix}', 0.)
    signature = get('device_signature')+get('select_configuration/device_signature')+get('select_configuration/select_variant/device_signature')
    selection = get('select_configuration')-get('select_configuration/device_signature')-get('select_configuration/select_variant/device_signature')
    known = (get('device_signature')+get('select_configuration')+get('configurations')+get('any')+get('event_seconds')
             +get('kernel_attributes')+get('empty')+get('validate_block_size'))
    return [('device_signature (3 calls, getDeviceProperties)', signature),
            ('tuner lookup without the device query (fingerprints, asdict, legality, record)', selection),
            ('candidate list (configurations)', get('configurations')),
            ('cp.any: boundary-zero test and failure read-back', get('any')),
            ('kernel launch, event and device wait (event_seconds)', get('event_seconds')),
            ('rest of _launch (identity dict, comparisons, closures, small allocations, attributes)', values.get(launch, 0.)-known
             + get('kernel_attributes')+get('empty')+get('validate_block_size'))]


def rhs_table(values):
    top = 'RHS/'+W
    raw = f'{top}/{R}/refresh_rhs'
    launch = f'{raw}/_launch'
    rows = launch_rows(values, launch)
    rows += [('_full_trace', values.get(f'{raw}/_full_trace', 0.)),
             ('source_moments_cupy', values.get(f'{top}/source_moments_cupy', 0.)),
             ('boundary_trace_values_cupy', values.get(f'{top}/boundary_trace_values_cupy', 0.))]
    total = values['RHS']
    rows.append(('everything else (rhs zeros, wrappers, validation, logging)', total-sum(v for _, v in rows)))
    rows.append(('RHS total', total))
    return rows


def reconstruction_table(values):
    top = f'RECONSTRUCT/{RR}/reconstruct'
    launch = f'{top}/_launch'
    rows = launch_rows(values, launch)
    rows += [('reconstruct_trace_cupy', values.get('RECONSTRUCT/reconstruct_trace_cupy', 0.)),
             ('require_finite_device_values', values.get('RECONSTRUCT/require_finite_device_values', 0.)),
             ('field_from_cupy_coefficients (3)', values.get('RECONSTRUCT/field_from_cupy_coefficients', 0.)),
             ('slice copies (4 ascontiguousarray)', values.get('RECONSTRUCT/ascontiguousarray', 0.)+values.get(f'{top}/ascontiguousarray', 0.))]
    total = values['RECONSTRUCT']
    rows.append(('everything else (wrappers, local array, logging)', total-sum(v for _, v in rows)))
    rows.append(('reconstruction total', total))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=Path)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    out = []
    rows = sorted({Path(p).parent.name.rsplit('-rep', 1)[0] for p in glob.glob(str(args.stage/'spans-*-rep*/spans.json'))})
    for row in rows:
        reps = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(args.stage/f'{row}-rep*/spans.json')))]
        for stage, table, root in (('RHS', rhs_table, 'RHS'), ('reconstruction', reconstruction_table, 'RECONSTRUCT')):
            columns = [table(per_solve(rep, root)) for rep in reps]
            print(f'\n{row} {stage}')
            for index, (label, _) in enumerate(columns[0]):
                values = [c[index][1] for c in columns]
                print(f'  {label:<84} {statistics.fmean(values):7.3f} ms  ({" ".join(f"{v:.3f}" for v in values)})')
                out.append(dict(row=row.removeprefix('spans-'), stage=stage, operation=label, ms_per_solve=statistics.fmean(values),
                                repetitions=' '.join(f'{v:.4f}' for v in values)))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(out[0]))
            writer.writeheader()
            writer.writerows(out)


if __name__ == '__main__':
    main()
