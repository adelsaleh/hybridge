"""host_pardiso.py --systems: list loading, verification and plan roles (no PARDISO runs)."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from campaigns.adr_unified import host_pardiso


def _export(root, key, faces=8, block=3, operator='abc'):
    cache = root/f'L1_p2_{key}'
    cache.mkdir()
    rng = np.random.default_rng(0)
    arrays = dict(system_blocks=rng.random((faces, 5, block, block)), system_rhs=rng.random((faces, block)),
                  system_neighbors=np.tile(np.arange(5), (faces, 1)), system_global_faces=np.arange(faces),
                  system_global_to_local=np.arange(faces), system_boundary_trace=np.zeros((faces, block)))
    for name, array in arrays.items():
        np.save(cache/f'{name}.npy', array)
    meta = dict(operator_sha256=operator, arrays={f'{n}.npy': (cache/f'{n}.npy').stat().st_size for n in arrays})
    (cache/'metadata.json').write_text(json.dumps(meta))
    return dict(level='L1', geometry='square', case='trigonometric', p=2, triangles=faces, trace_dofs=faces*block,
                system_id=f'L1_p2_{key}', key=key, cache=str(cache), operator_sha256=operator)


def _write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_listed_systems_verify_metadata(tmp_path):
    good = _export(tmp_path, 'a'*16)
    bad = _export(tmp_path, 'b'*16)
    bad['operator_sha256'] = 'other'
    _write(tmp_path/'list.csv', [good, bad])
    specs, skipped = host_pardiso.listed_systems(tmp_path/'list.csv')
    assert [s['key'] for s in specs] == ['a'*16] and specs[0]['system_id'] == 'abc'
    assert specs[0]['trace_dofs'] == 24 and specs[0]['block_size'] == 3 and specs[0]['nnz'] == 8*5*9
    assert skipped[0][0] == 'b'*16 and 'operator_sha256' in skipped[0][1]


def test_plan_roles_index_width_fixed_and_borrowed(tmp_path):
    rows = [_export(tmp_path, k) for k in ('c'*16, 'd'*16)]
    _write(tmp_path/'list.csv', rows)
    selections = {'L2_p9_' + 'e'*16: dict(status='selected', trace_dofs=20, selected_threads=dict(fresh=96, reused=32))}
    (tmp_path/'v2').mkdir()
    (tmp_path/'v2'/'thread_selection.json').write_text(json.dumps(selections))
    out = tmp_path/'out'
    assert host_pardiso.main(['--systems', str(tmp_path/'list.csv'), '--output', str(out), '--borrow-selections',
                              str(tmp_path/'v2'), '--tune-min-dofs', '1', '--smallest-first']) == 0
    roles = {r['key']: r['role'] for r in json.loads((out/'plan.json').read_text())['rows']}
    assert set(roles.values()) == {'borrowed'}
    host_pardiso.main(['--systems', str(tmp_path/'list.csv'), '--output', str(out), '--max-factor-nnz', '10'])
    assert {r['role'] for r in json.loads((out/'plan.json').read_text())['rows']} == {'index_width_excluded'}
    host_pardiso.main(['--systems', str(tmp_path/'list.csv'), '--output', str(out), '--fixed-threads', '2',
                       '--fixed-threads-from-dofs', '1'])
    assert {r['role'] for r in json.loads((out/'plan.json').read_text())['rows']} == {'fixed_large'}


def test_failure_kind_index_width():
    assert host_pardiso.failure_kind(dict(status='index_width_excluded')) == 'index_width'
    with pytest.raises(SystemExit):
        host_pardiso.parser().parse_args(['--output', 'x'])  # a source is required


def test_timing_metrics_keep_residual_floor_runs():
    from campaigns.adr_pardiso.adr_pardiso_inventory import timing_metrics
    solve = lambda rel: dict(solve_ms=5.0, true_relative_residual=rel, passed=rel <= 1e-10)
    trial = lambda rel: dict(setup_ms=100.0, solves=[solve(rel), solve(rel)])
    row = dict(status='residual_floor', rtol=1e-10, floor_rtol=1e-8, requested_repeats=2,
               warmups=[trial(1.2e-10)], samples=[trial(1.2e-10), trial(1.1e-10)])
    metrics = timing_metrics(row)
    assert metrics['fresh_median_ms'] == 105.0 and metrics['worst_relative_residual'] == 1.2e-10
    assert timing_metrics(dict(row, status='numerical_failure')) is None
    assert timing_metrics(dict(row, floor_rtol=1e-11)) is None
    assert host_pardiso.failure_kind(dict(status='residual_floor')) == 'residual_floor'
