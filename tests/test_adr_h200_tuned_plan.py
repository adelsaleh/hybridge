"""The H200 tuned ADR campaign plan: ids shared with the unified campaign, curated tuned candidates."""
from __future__ import annotations

from pathlib import Path

import pytest

from campaigns.adr_h200_tuned import adr_h200_tuned_plan as tuned
from campaigns.adr_unified import calibrate_assembly_memory as calibration
from campaigns.adr_unified.adr_unified_plan import build_plan as build_unified_plan, problem_class, read_inventory
from campaigns.adr_unified.run_adr_unified_campaign import estimate_memory
from hybridge.runtime import devices

INVENTORY = Path(__file__).resolve().parents[1]/'run_configs/adr_unified_l5/inventory.json'


@pytest.fixture(scope='module')
def plans():
    inventory = read_inventory(INVENTORY)
    unified = build_unified_plan(inventory, l5_triangles=400000, hp_grid=True, lu_levels=True)
    return tuned.build_plan(inventory), unified


def test_shared_systems_keep_unified_ids(plans):
    plan, unified = plans
    unified_ids = {s['id'] for s in unified['systems']}
    shared = [s for s in plan['systems'] if s['shared_with_unified']]
    assert shared and all(s['id'] in unified_ids for s in shared)
    # Every L1-L7 system of the six unified representative classes in the ladder is shared.
    representative = {k for k in tuned.CLASSES if k != ('square', 'anisotropic')}
    for system in plan['systems']:
        if problem_class(system)[:2] in representative and system['level'] not in ('L8', 'L9'):
            assert system['shared_with_unified'], (system['level'], system['problem']['case'], system['problem']['p'])


def test_ladder_levels_degrees_and_classes(plans):
    plan, _ = plans
    assert {s['problem']['p'] for s in plan['systems']} == set(tuned.DEGREES)
    assert {problem_class(s)[:2] for s in plan['systems']} == set(tuned.CLASSES)
    assert {'L1', 'L5', 'L6', 'L7', 'L8', 'L9'} <= {s['level'] for s in plan['systems']}
    cases = {s['problem']['case'] for s in plan['systems']}
    assert not cases & {'stress_main_trap', 'stress_main_cross', 'cellular7_directional'}


def test_candidates_are_curated_and_tuned(plans):
    plan, unified = plans
    for system in plan['systems']:
        key = problem_class(system)[:2]
        families = [c['solver']['family'] for c in system['candidates']]
        assert families.count('raw_assembly') == 1
        bases = [c['base_candidate'] for c in system['candidates'] if c['solver']['family'] != 'raw_assembly']
        assert bases == list(tuned.CLASS_CANDIDATES[key])
        if key in (('annulus', 'cellular7_weak'), ('annulus', 'stress_main_orthogonal')):
            assert 'native_hp' not in families
        if key == ('annulus', 'stress_main_orthogonal'):
            assert 'amgx' not in families
        for choice in system['candidates']:
            solver = choice['solver']
            family = solver['family']
            if family == 'asm_pp':
                assert solver['configuration']['operator'] == 'interleaved'
                assert solver['configuration']['application'] == 'interleaved_f32'
                assert solver['estimate_target'] == 'true_target' and solver['tuned_face_dense']
            elif family == 'bj_pp':
                assert solver['configuration']['application'] == 'interleaved'
                assert solver['estimate_target'] == 'true_target' and solver['tuned_face_dense']
            elif family == 'native_hp':
                assert solver['hp_mg_kernels'] == 'graph' and solver['operator_implementation'] == 'interleaved'
            elif family == 'amgx':
                assert 'amgx_config' in solver and not solver['tuned_face_dense']
            elif family == 'raw_assembly':
                p = system['problem']['p']
                assert solver['block_size'] == (32 if p <= 4 else 'auto')
            assert solver['maxiter'] == 2000
    unified_by_id = {c['id'] for s in unified['systems'] for c in s['candidates']}
    assert all(c['base_candidate'] in unified_by_id for s in plan['systems'] for c in s['candidates']
               if 'base_candidate' in c)


def test_tuned_ids_differ_from_unified_ids(plans):
    plan, _ = plans
    for choice in plan['systems'][0]['candidates']:
        if 'base_candidate' in choice:
            assert choice['id'] != choice['base_candidate']


def test_system_budget_and_pending_candidates(tmp_path):
    import json
    from types import SimpleNamespace
    from campaigns.adr_h200_tuned.run_adr_h200_tuned_campaign import pending_candidates, system_budget
    args = SimpleNamespace(timeout=100., output=tmp_path, resume=True, heartbeat=30.)
    budget = system_budget(args, 4)
    assert budget.timeout == 500. and budget.output == tmp_path and budget.resume is True
    candidates = [dict(id='a'), dict(id='b'), dict(id='c')]
    (tmp_path/'jobs'/'sys_a').mkdir(parents=True)
    (tmp_path/'jobs'/'sys_a'/'result.json').write_text(json.dumps(dict(status='passed')))
    (tmp_path/'jobs'/'sys_b').mkdir(parents=True)
    (tmp_path/'jobs'/'sys_b'/'result.json').write_text(json.dumps(dict(status='process_error')))
    assert pending_candidates(tmp_path, 'sys', candidates) == ['b', 'c']


def test_reclaimable_page_cache_counts_as_available(tmp_path):
    v2 = tmp_path/'v2.stat'
    v2.write_text('anon 100\ninactive_file 4000\nactive_file 1000\nfile_dirty 300\nfile_writeback 200\n')
    assert devices._reclaimable_file_bytes(v2) == 4500
    v1 = tmp_path/'v1.stat'
    v1.write_text('cache 9\ninactive_file 1\ntotal_inactive_file 700\ntotal_active_file 300\ntotal_dirty 100\n')
    assert devices._reclaimable_file_bytes(v1) == 900
    assert devices._reclaimable_file_bytes(tmp_path/'missing') == 0


def test_calibration_fits_host_and_runner_uses_it():
    rows = [dict(p=6, status='ok', triangles=t, peak_device_bytes=1000*t+10, peak_host_bytes=5000*t+7)
            for t in (100, 200, 400)]
    model = calibration.fit(rows)['6']
    assert model['host_bytes_per_triangle'] == pytest.approx(5000)
    assert model['host_intercept_bytes'] == pytest.approx(7)
    system = dict(triangles=1000, problem=dict(p=6), trace_dofs=None)
    choice = dict(solver=dict(family='asm_pp', configuration=dict(polynomial_degree=24)))
    planned = estimate_memory(system, choice, None)
    measured = estimate_memory(system, choice, dict(model={'6': model}))
    assert measured['assembly_host_bytes'] == pytest.approx(1.15*(5000*1000+7), abs=1)
    assert measured['assembly_host_source'] == 'calibrated'
    assert measured['solver_host_bytes'] < planned['solver_host_bytes']
    # Old calibrations without host fits leave the host planning model in place.
    legacy = {k: v for k, v in model.items() if not k.startswith('host_')}
    assert estimate_memory(system, choice, dict(model={'6': legacy}))['assembly_host_bytes'] == planned['assembly_host_bytes']


def test_outcome_gate_passes_non_convergence_and_stops_on_crashes(tmp_path):
    import json
    from campaigns.adr_h200_tuned.outcome_gate import check
    good, crashed, unfinished = tmp_path/'good', tmp_path/'crashed', tmp_path/'unfinished'
    for out, statuses in ((good, ('passed', 'not_converged', 'numerical_failure')), (crashed, ('passed', 'process_error')),
                          (unfinished, ('passed',))):
        out.mkdir()
        (out/'summary.json').write_text(json.dumps([dict(status=s, system_id='s', solver=dict(id='c')) for s in statuses]))
        if out != unfinished:
            (out/'completion.json').write_text('{}')
    assert check([good]) == ({str(good): {'passed': 1, 'not_converged': 1, 'numerical_failure': 1}}, [])
    assert len(check([good, crashed])[1]) == 1 and 'process_error' in check([crashed])[1][0]
    assert check([unfinished])[1] == [f'{unfinished}: no completion.json']


def test_retry_mesh_errors_reruns_only_failed_meshes(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from campaigns.adr_h200_tuned import run_adr_h200_tuned_campaign as campaign
    calls = {}
    monkeypatch.setattr(campaign.runner, 'run_job', lambda key, spec, args, common, env, retry=False: calls.update({key: retry}))
    common = SimpleNamespace(read_json=lambda path: json.loads(Path(path).read_text()))
    for sid, status in (('a', 'error'), ('b', 'passed')):
        (tmp_path/'jobs'/f'mesh_{sid}').mkdir(parents=True)
        (tmp_path/'jobs'/f'mesh_{sid}'/'result.json').write_text(json.dumps(dict(status=status)))
    for flag in (False, True):
        args = SimpleNamespace(output=tmp_path, branch_root=tmp_path, inventory=tmp_path/'inventory.json', retry_mesh_errors=flag)
        for sid in ('a', 'b', 'c'):
            campaign.mesh_job(dict(id=sid), args, common, {})
        assert calls == {'mesh_a': flag, 'mesh_b': False, 'mesh_c': False}


def _solvers(system):
    return [c for c in system['candidates'] if c['solver']['family'] != 'raw_assembly']


def test_system_device_need_matches_the_worker_exclusion(plans):
    plan, _ = plans
    # 51-l6-l7 recorded "Device estimate 167.74 GiB exceeds budget 125.36 GiB" for the L7 p=9 squares.
    system = next(s for s in plan['systems'] if s['level'] == 'L7' and s['problem']['p'] == 9
                  and s['problem']['case'] == 'trigonometric')
    assert tuned.system_device_need(system, _solvers(system))/2**30 == pytest.approx(167.74, abs=.01)


def test_largest_level_is_the_largest_fitting_size_per_degree(plans):
    from campaigns.adr_unified.adr_unified_plan import _sized_for_dofs
    default, unified = plans
    assert 'largest' not in default and tuned.LARGEST_LEVEL not in {s['level'] for s in default['systems']}
    plan = tuned.build_plan(read_inventory(INVENTORY), degrees=(2, 9), levels=[tuned.LARGEST_LEVEL], largest=True)
    sizes = {int(p): d for p, d in plan['largest']['trace_dofs_by_degree'].items()}
    assert sizes[2] > 20_000_000 > sizes[9] > 8_000_000
    assert {s['level'] for s in plan['systems']} == {tuned.LARGEST_LEVEL}
    assert {problem_class(s)[:2] for s in plan['systems']} == set(tuned.CLASSES)
    l5 = {key: next(base for level, base, _ in tuned._mesh_bases(tuned._class_systems(unified, key)) if level == 'L5')
          for key in tuned.CLASSES}
    for p, dofs in sizes.items():
        systems = {problem_class(s)[:2]: s for s in plan['systems'] if s['problem']['p'] == p}
        assert max(tuned.system_device_need(s, _solvers(s)) for s in systems.values()) <= tuned.LARGEST_BUDGET_BYTES
        # One step larger, some class no longer fits.
        assert any(tuned.system_device_need(_sized_for_dofs(l5[key], p, tuned.LARGEST_LEVEL, dofs+tuned.LARGEST_STEP,
                                                            'h200_tuned'), _solvers(s)) > tuned.LARGEST_BUDGET_BYTES
                   for key, s in systems.items())
    # A class subset keeps the sizes and the ids.
    subset = tuned.build_plan(read_inventory(INVENTORY), degrees=(9,), levels=[tuned.LARGEST_LEVEL], largest=True,
                              classes=(('square', 'trigonometric'),))
    assert [s['id'] for s in subset['systems']] == [s['id'] for s in plan['systems']
                                                    if s['problem']['p'] == 9 and s['problem']['case'] == 'trigonometric']
