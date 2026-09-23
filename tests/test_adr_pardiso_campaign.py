"""Read-only inventory and tiny CPU matrix tests; never a PDE run or JIT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hdgfem.linalg.pardiso_diagnostics import pardiso_factor_statistics
from scripts.advection_diffusion_reaction import adr_pardiso_inventory as inventory
from scripts.advection_diffusion_reaction import adr_pardiso_worker as worker
from scripts.advection_diffusion_reaction import run_adr_pardiso_campaign as campaign


def passing_row():
    return dict(status='passed', candidate='asm_test', family='asm_pp', rtol=1e-10,
                requested_repeats=2, warmups=[dict(setup_ms=999, solves=[
                    dict(solve_ms=999, passed=True, true_relative_residual=1e-12)]*2)],
                samples=[dict(setup_ms=setup, solves=[
                    dict(solve_ms=first, passed=True, true_relative_residual=1e-12),
                    dict(solve_ms=second, passed=True, true_relative_residual=1e-12)])
                    for setup, first, second in [(10, 20, 4), (20, 30, 8)]])


def test_metrics_pair_setup_first_and_exclude_warmups():
    metrics = inventory.timing_metrics(passing_row())
    assert metrics['fresh_median_ms'] == 40
    assert metrics['setup_median_ms'] == 15
    assert metrics['reused_mean_ms'] == 6


@pytest.mark.parametrize('change', ['status', 'short', 'residual', 'nan', 'failed_solve', 'profile'])
def test_no_rank_for_bad_or_incomplete_measurements(change):
    row = passing_row()
    if change == 'status': row['status'] = 'numerical_failure'
    if change == 'short': row['samples'].pop()
    if change == 'residual': row['warmups'][0]['solves'][0]['true_relative_residual'] = 1e-5
    if change == 'nan': row['samples'][0]['setup_ms'] = float('nan')
    if change == 'failed_solve': row['samples'][0]['solves'][0]['passed'] = False
    if change == 'profile': row['profile_note'] = 'instrumented'
    assert inventory.timing_metrics(row) is None


def test_comparison_direction_and_failure_retention():
    system = dict(system_id='abc', case='test', geometry='square', p=6, triangles=2, trace_dofs=4,
                  comparators=[dict(candidate='asm', family='ASM+PP', status='passed', metrics=inventory.timing_metrics(passing_row())),
                               dict(candidate='amgx', family='AMGX', status='numerical_failure', metrics=None)])
    direct = passing_row()
    direct['threads'] = 24
    for s in direct['samples']:
        s['setup_ms'] /= 2
        for v in s['solves']: v['solve_ms'] /= 2
    compared = inventory.comparisons(system, direct)
    assert compared[0]['speedup_fresh'] == 2
    assert compared[0]['speedup_reused'] == 2
    assert compared[1]['speedup_reused'] is None
    direct['status'] = 'error'
    assert all(r['speedup_fresh'] is None for r in inventory.comparisons(system, direct))


def test_pardiso_statistics_one_based_not_process_rss():
    class Solver:
        factorized_A = 'hashed'
        def get_iparm(self, i):
            return {7: 2, 14: 0, 15: 100, 16: 80, 17: 200, 18: 500, 60: 0}[i]
    stats = pardiso_factor_statistics(Solver(), matrix_nnz=50)
    assert stats['estimated_solver_peak_kib'] == 280
    assert stats['fill_ratio'] == 10
    assert stats['wrapper_uses_matrix_hash']
    assert stats['wrapper_matrix_copy_bytes'] == 0


@pytest.fixture
def tiny_spec(tmp_path):
    cache = tmp_path/'cache'
    cache.mkdir()
    blocks = np.array([[[[4.]], [[1.]]], [[[3.]], [[.5]]]])
    neighbors = np.array([[0, 1], [1, 0]], dtype=np.int64)
    rhs = np.array([6., 6.5])
    for name, array in [('blocks', blocks), ('neighbors', neighbors), ('rhs', rhs)]:
        np.save(cache/f'system_{name}.npy', array, allow_pickle=False)
    h = hashlib.sha256()
    for array in (blocks, rhs): h.update(memoryview(array).cast('B'))
    identity = h.hexdigest()
    system = dict(system_id=identity, key='tiny', case='tiny', cache=str(cache),
                  trace_dofs=2, triangles=1, geometry='square', p=1, origins=[], comparators=[],
                  neighbors_file_sha256=inventory.digest(cache/'system_neighbors.npy'),
                  cache_files={p.name:dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in cache.glob('*.npy')})
    args = campaign.parser().parse_args(['--threads', '1', '--repeats', '2', '--max-dofs', '100'])
    protocol = {k: getattr(args, k) for k in ('repeats','warmup','max_dofs','rtol')}
    spec = tmp_path/'spec.json'
    spec.write_text(json.dumps(dict(system=system, threads=1, protocol=protocol)))
    return spec, system


def test_cache_validation_is_readonly(tiny_spec):
    _, system = tiny_spec
    blocks, neighbors, rhs = worker.validate_cache(system)
    assert not blocks.flags.writeable
    assert not neighbors.flags.writeable
    np.testing.assert_array_equal(rhs, [6, 6.5])


def test_cache_hash_mismatch_rejected(tiny_spec):
    _, system = tiny_spec
    system['system_id'] = 'wrong'
    with pytest.raises(ValueError, match='hash differs'):
        worker.validate_cache(system)


def test_changed_cache_rejected(tiny_spec):
    _, system = tiny_spec
    system['cache_files']['system_blocks.npy']['size'] += 8
    with pytest.raises(ValueError, match='changed since planning'):
        worker.validate_cache(system)


def test_tiny_worker_repeats_fresh_lu_and_reuses_factors(tiny_spec, tmp_path, monkeypatch):
    pardiso = pytest.importorskip('pypardiso')
    import hdgfem.linalg.system as backend
    spec, system = tiny_spec
    data = inventory.read(spec)
    data['threads'] = int(pardiso.ps.libmkl.MKL_Get_Max_Threads())
    spec.write_text(json.dumps(data))
    output = tmp_path/'worker'
    output.mkdir()
    calls = []
    original = backend.solve_pypardiso_system
    def solve(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(backend, 'solve_pypardiso_system', solve)
    assert worker.run(spec, output) == 0
    result = inventory.read(output/'result.json')
    assert len(result['warmups']) == 1 and len(result['samples']) == 2
    assert len(calls) == 7  # three setups x two physical RHS solves + one probe
    assert all(v['pardiso_phase'] == 33 for s in result['samples'] for v in s['solves'])
    assert all(s['factor_statistics']['in_core'] for s in result['samples'])
    assert result['probe']['relative_solution_error'] < 1e-12
    assert result['peak_process_rss_kib'] > 0
    np.testing.assert_allclose(np.load(output/'eliminated_solution.npy'), [1, 2])
    assert not (Path(system['cache'])/'reference_trace.npy').exists()


def test_all_archive_systems_levels_degrees_and_fine_stress_present():
    if not (inventory.NATIVE/'manifest.json').exists():
        pytest.skip('local archived campaigns unavailable')
    systems, _ = inventory.inventory()
    coverage = inventory.coverage(systems)
    assert coverage['unique_systems'] == 107
    assert coverage['mesh_levels'] == ['L1','L2','L3','L4']
    assert coverage['degrees'] == [1,2,3,4,6]
    assert coverage['systems_by_suite']['stress'] == 6
    assert coverage['max_trace_dofs'] == 1547455
    assert {o['target_triangles'] for s in systems for o in s['origins'] if o['suite']=='stress'} == {100000,150000}
    assert all(s['comparators'] for s in systems)
    ladders = {}
    for s in systems:
        for o in s['origins']:
            if 'mesh_level' in o:
                ladders.setdefault((o['suite'],o['case'],o['geometry']),set()).add(o['mesh_level'])
    assert all(v == {'L1','L2','L3','L4'} for v in ladders.values())


def test_environment_is_cpu_only_and_disables_jit():
    env = campaign.worker_environment(24)
    assert env['MKL_NUM_THREADS'] == env['OMP_NUM_THREADS'] == '24'
    assert env['NUMBA_DISABLE_JIT'] == '1'
    assert env['HDGFEM_PRECISION'] == 'float64'
    assert env['OPENBLAS_NUM_THREADS'] == '1'


@pytest.mark.parametrize('options', [['--warmup','0'], ['--rtol','1e-8'], ['--reserve-gib','nan'],
                                     ['--threads','0'], ['--threads','1','1'], ['--retry-failed']])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        campaign.validate(campaign.parser().parse_args(options))


def test_plan_readonly_and_separate_output(tiny_spec, tmp_path, monkeypatch):
    _, system = tiny_spec
    monkeypatch.setattr(inventory, 'inventory', lambda *args: ([system], {}))
    output = tmp_path/'new'
    assert campaign.main(['--output',str(output),'--threads','1']) == 0
    assert not output.exists()
    with pytest.raises(ValueError, match='separate'):
        campaign.main(['--output',system['cache'],'--threads','1'])
    with pytest.raises(ValueError, match='max-dofs'):
        campaign.main(['--output',str(output),'--threads','1','--max-dofs','1'])


def test_mocked_execution_resume_and_changed_protocol(tiny_spec, tmp_path, monkeypatch):
    _, system = tiny_spec
    monkeypatch.setattr(inventory, 'inventory', lambda *args: ([system], {}))
    calls = []
    def monitor(command, env, output, args):
        calls.append((command, env))
        row = passing_row()
        row['threads'] = 1
        return row
    monkeypatch.setattr(campaign, 'monitor', monitor)
    output = tmp_path/'campaign'
    cmd = ['--output',str(output),'--threads','1','--execute']
    assert campaign.main(cmd) == 0
    assert len(calls) == 1
    assert campaign.main(cmd+['--resume']) == 0
    assert len(calls) == 1
    with pytest.raises(ValueError, match='changed'):
        campaign.main(cmd+['--resume','--repeats','4'])
    with pytest.raises(FileExistsError):
        campaign.main(cmd)
    summary = inventory.read(output/'summary.json')
    assert summary['status'] == 'completed' and summary['passed'] == 1


def test_mocked_failures_are_retained_and_retry_preserves_attempt(tiny_spec, tmp_path, monkeypatch):
    _, system = tiny_spec
    monkeypatch.setattr(inventory, 'inventory', lambda *args: ([system], {}))
    calls = []
    def monitor(*args):
        calls.append(1)
        return dict(status='memory_budget_exceeded', monitored_peak_rss_gib=65)
    monkeypatch.setattr(campaign, 'monitor', monitor)
    output = tmp_path/'campaign'
    cmd = ['--output',str(output),'--threads','1','--execute']
    assert campaign.main(cmd) == 1
    assert campaign.main(cmd+['--resume']) == 1
    assert len(calls) == 1
    assert campaign.main(cmd+['--resume','--retry-failed']) == 1
    assert len(calls) == 2
    assert len(list((output/'jobs').glob('*/attempt_*'))) == 2
    row = inventory.read(output/'summary.json')['records'][0]
    assert row['fresh_median_ms'] is None and row['peak_process_rss_gib'] == 65



def test_tiny_isolated_worker_entrypoint_and_monitor(tiny_spec, tmp_path):
    pytest.importorskip('pypardiso')
    import sys
    spec, _ = tiny_spec
    output = tmp_path/'isolated'
    output.mkdir()
    args = campaign.parser().parse_args(['--threads','1','--max-rss-gib','4','--reserve-gib','0.001'])
    result = campaign.monitor([sys.executable, '-u', '-B', str(Path(campaign.__file__).resolve()),
                               '--worker-spec', str(spec), '--output', str(output)],
                              campaign.worker_environment(1), output, args)
    assert result['status'] == 'passed'
    assert result['mkl_max_threads'] == 1
    assert result['monitored_peak_hwm_gib'] > 0
    assert result['minimum_available_gib'] > 0
    assert result['memory_poll_seconds'] == 0.2


@pytest.mark.parametrize('failure', ['timeout', 'memory_budget_exceeded'])
def test_monitor_stops_worker_and_preserves_memory(failure, tmp_path, monkeypatch):
    from scripts.advection_diffusion_reaction import check_cached_adr_pardiso as diagnostic
    class Process:
        pid = 123
        returncode = None
        terminated = False
        def poll(self): return self.returncode
        def terminate(self): self.terminated = True
        def kill(self): self.terminated = True
        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode
    process = Process()
    monkeypatch.setattr(diagnostic.subprocess, 'Popen', lambda *a, **kw: process)
    times = iter([0., 2., 4.])
    monkeypatch.setattr(diagnostic.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(diagnostic, 'memory_kib', lambda path, key: 100*1024**2 if key=='MemAvailable' else 2*1024**2)
    args = campaign.parser().parse_args(['--timeout','1' if failure=='timeout' else '10','--max-rss-gib','1'])
    result = diagnostic.monitor(['mock'], {}, tmp_path, args)
    assert process.terminated
    assert result['status'] == failure
    assert result['monitored_peak_rss_gib'] == 2
    assert result['monitored_peak_hwm_gib'] == 2
    assert result['returncode'] == -15


# Thread tuning uses pilots only; confirmation data must not select a winner.
def thread_row(threads, fresh, reused, *, phase='tuning', system_id='abc'):
    row = passing_row()
    row.update(threads=threads, measurement_phase=phase, system_id=system_id,
               diagnostic_output=f'/diagnostics/{system_id}/{threads}/{phase}')
    for sample in row['samples']:
        sample['setup_ms'] = fresh - 1
        sample['solves'][0]['solve_ms'] = 1
        sample['solves'][1]['solve_ms'] = reused
    return row


def test_thread_grid_includes_serial_and_caps_default_at_physical_cores():
    from scripts.advection_diffusion_reaction import adr_pardiso_tuning as tuning
    assert tuning.default_candidates(24) == [1,2,4,6,8,12,16,24]
    assert tuning.default_candidates(3) == [1,2,3]
    assert tuning.default_candidates(1) == [1]
    with pytest.raises(ValueError):
        tuning.default_candidates(0)


def test_thread_order_is_reproducible_and_not_always_increasing():
    from scripts.advection_diffusion_reaction import adr_pardiso_tuning as tuning
    expected = list(range(1,25))
    order = tuning.candidate_order('abc', expected, 77)
    assert order == tuning.candidate_order('abc', list(reversed(expected)), 77)
    assert sorted(order) == expected and order != expected
    assert order != tuning.candidate_order('def', expected, 77)


def test_fresh_and_reused_threads_selected_separately_with_serial_winner():
    from scripts.advection_diffusion_reaction import adr_pardiso_tuning as tuning
    rows = [thread_row(1, 5, 4), thread_row(24, 20, 1)]
    selected = tuning.select_threads('abc', rows)
    assert selected['selected_threads'] == {'fresh':1, 'reused':24}
    assert tuning.confirmation_objectives(selected, 1) == ['fresh']
    assert tuning.confirmation_objectives(selected, 24) == ['reused']


def test_thread_selection_rejects_failure_and_incomplete_pilots():
    from scripts.advection_diffusion_reaction import adr_pardiso_tuning as tuning
    rows = [thread_row(1, 5, 4), thread_row(2, 2, 1), thread_row(4, 3, 2)]
    rows[1]['status'] = 'numerical_failure'
    rows[2]['samples'].pop()
    selected = tuning.select_threads('abc', rows)
    assert selected['selected_threads'] == {'fresh':1, 'reused':1}
    rows[0]['status'] = 'memory_budget_exceeded'
    selected = tuning.select_threads('abc', rows)
    assert selected['status'] == 'no_passing_candidate'
    assert selected['selected_threads'] == {'fresh':None, 'reused':None}


def test_thread_selection_exact_ties_prefer_fewer_threads():
    from scripts.advection_diffusion_reaction import adr_pardiso_tuning as tuning
    assert tuning.select_threads('abc', [thread_row(24, 5, 4), thread_row(1, 5, 4)])['selected_threads'] == {'fresh':1,'reused':1}


@pytest.mark.parametrize('bad', ['confirmation', 'foreign_system', 'duplicate'])
def test_selection_does_not_use_confirmation_or_other_system(bad):
    from scripts.advection_diffusion_reaction import adr_pardiso_tuning as tuning
    rows = [thread_row(1, 5, 4)]
    if bad == 'confirmation': rows[0]['measurement_phase'] = 'confirmation'
    if bad == 'foreign_system': rows[0]['system_id'] = 'different'
    if bad == 'duplicate': rows.append(thread_row(1, 3, 1))
    with pytest.raises(ValueError):
        tuning.select_threads('abc', rows)


def test_thread_comparisons_exclude_pilots_and_nonselected_objective():
    system = dict(system_id='abc', case='test', geometry='square', p=6, triangles=2, trace_dofs=4,
                  comparators=[dict(candidate='asm', family='ASM+PP', status='passed',
                                    metrics=inventory.timing_metrics(passing_row()))])
    row = thread_row(1, 5, 4)
    assert inventory.comparisons(system, row) == []
    row.update(measurement_phase='confirmation', selected_for=['fresh'])
    result = inventory.comparisons(system, row)[0]
    assert result['speedup_fresh'] == 8
    assert result['speedup_reused'] is None
    assert result['selected_for'] == ['fresh']


@pytest.mark.parametrize('options', [['--tuning-repeats','1'], ['--repeats','2']])
def test_tuning_requires_repeated_pilots_and_confirmation(options):
    with pytest.raises(ValueError, match='pilot'):
        campaign.validate(campaign.parser().parse_args(['--autotune-threads','--threads','1']+options))


def test_tuned_campaign_confirms_winners_independently_and_resumes(tiny_spec, tmp_path, monkeypatch):
    spec_path, system = tiny_spec
    # No parallel work or MKL calls; permit two candidates on restricted CI.
    monkeypatch.setattr(campaign.os, 'sched_getaffinity', lambda _: {0,1})
    monkeypatch.setattr(inventory, 'inventory', lambda *args: ([system], {}))
    system['comparators'] = [dict(candidate='asm', family='ASM+PP', status='passed',
                                  metrics=inventory.timing_metrics(passing_row()))]
    calls = []
    def monitor(command, env, output, args):
        spec = inventory.read(output/'spec.json')
        phase, threads = spec['measurement_phase'], spec['threads']
        calls.append((phase, threads, spec['protocol']['repeats']))
        # Reverse the confirmation ranking deliberately. It must not change the
        # selected count or cherry-pick the other confirmation's faster metric.
        times = {('tuning',1):(5,4), ('tuning',2):(20,1),
                 ('confirmation',1):(50,40), ('confirmation',2):(2,0.5)}
        fresh, reused = times[(phase,threads)]
        row = thread_row(threads, fresh, reused, phase=phase, system_id=system['system_id'])
        row['diagnostic_output'] = str(output)
        row['requested_repeats'] = spec['protocol']['repeats']
        row['samples'] = [row['samples'][0]] * row['requested_repeats']
        return row
    monkeypatch.setattr(campaign, 'monitor', monitor)
    output = tmp_path/'tuned'
    cmd = ['--output',str(output),'--autotune-threads','--threads','1','2',
           '--tuning-repeats','3','--repeats','5','--execute']
    assert campaign.main(cmd) == 0
    assert sorted(calls) == [('confirmation',1,5),('confirmation',2,5),('tuning',1,3),('tuning',2,3)]
    selected = inventory.read(output/'thread_selection.json')[0]
    assert selected['selected_threads'] == {'fresh':1,'reused':2}
    compared = inventory.read(output/'comparisons.json')
    assert len(compared) == 2 and all(r['measurement_phase']=='confirmation' for r in compared)
    fresh_row = next(r for r in compared if r['selected_for'] == ['fresh'])
    assert fresh_row['threads'] == 1 and fresh_row['direct_metrics']['fresh_median_ms'] == 50
    assert fresh_row['speedup_reused'] is None
    summary = inventory.read(output/'summary.json')
    assert summary['systems_with_confirmed_baselines'] == 1
    assert summary['scheduled'] == 4
    assert (output/'tuning.csv').exists() and (output/'confirmed_timings.csv').exists()
    assert campaign.main(cmd+['--resume']) == 0
    assert len(calls) == 4
    with pytest.raises(ValueError, match='changed'):
        campaign.main(cmd+['--resume','--tuning-repeats','4'])


def test_retry_pilot_cannot_reuse_confirmation_from_old_selection(tiny_spec, tmp_path, monkeypatch):
    _, system = tiny_spec
    monkeypatch.setattr(campaign.os, 'sched_getaffinity', lambda _: {0,1})
    monkeypatch.setattr(inventory, 'inventory', lambda *args: ([system], {}))
    calls = []
    failing = True
    def monitor(command, env, output, args):
        spec = inventory.read(output/'spec.json')
        phase, threads = spec['measurement_phase'], spec['threads']
        calls.append((phase, threads))
        if failing and phase=='tuning' and threads==1:
            return dict(status='timeout')
        row = thread_row(threads, 5*threads, 4*threads, phase=phase, system_id=system['system_id'])
        row['diagnostic_output'] = str(output)
        return row
    monkeypatch.setattr(campaign, 'monitor', monitor)
    output = tmp_path/'retry'
    cmd = ['--output',str(output),'--autotune-threads','--threads','1','2','--execute']
    assert campaign.main(cmd) == 1  # pilot failure preserved, not silently hidden
    first = inventory.read(output/'thread_selection.json')[0]
    assert first['selected_threads'] == {'fresh':2,'reused':2}
    failing = False
    assert campaign.main(cmd+['--resume','--retry-failed']) == 0
    second = inventory.read(output/'thread_selection.json')[0]
    assert second['selected_threads'] == {'fresh':1,'reused':1}
    assert first['fingerprint'] != second['fingerprint']
    assert ('confirmation',1) in calls and ('confirmation',2) in calls
    assert len(list((output/'jobs').glob('*confirmation*/result.json'))) == 2


def test_tuning_environment_clears_overrides(monkeypatch):
    monkeypatch.setenv('MKL_DOMAIN_NUM_THREADS', 'MKL_PARDISO=24')
    monkeypatch.setenv('OMP_THREAD_LIMIT', '1')
    env = campaign.worker_environment(2)
    assert 'MKL_DOMAIN_NUM_THREADS' not in env
    assert env['OMP_THREAD_LIMIT'] == '2' and env['OMP_DYNAMIC'] == 'FALSE'
