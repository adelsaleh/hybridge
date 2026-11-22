"""H200 node LU comparison of scripts/reports/make_adr_h200_comparison.py on synthetic rows."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/reports'))

import make_adr_h200_comparison as report  # noqa: E402


def _job(system, dofs, family, status, cold, repeated, l2=1e-6, residual=1e-12):
    # Same columns as campaigns/adr_h200_tuned/export_records.py: ``l2_error`` is the worst L2 error of
    # the measured solves and is empty in the CSV for a job that did not pass.
    return dict(system_id=system, level='L2', geometry='square', case='trigonometric', p=6, triangles=dofs//10,
                trace_dofs=dofs, family=family, status=status, cold_ms=cold, repeated_ms=repeated,
                l2_error=str(l2) if status == 'passed' else '',
                first_l2_error=str(l2), first_true_relative_residual=str(residual))


def _lu(fresh, fresh_threads, reused, reused_threads):
    # Same keys as node_lu(): a passed factorization has no residual recorded.
    return dict(fresh=fresh, fresh_threads=fresh_threads, reused=reused, reused_threads=reused_threads,
                factor_nnz=1., peak_gib=.1, status='passed', residual=None)


def test_lu_rows_ratios_outcomes_and_crossover(monkeypatch):
    # FP64 residual floor of 1e-10 for every size, in place of the fit to the measured floors.
    monkeypatch.setattr(report, 'fp64_floor_model', lambda path=None: {('square', 'trigonometric'): lambda dofs: 1e-10})
    tuned = [_job('L1_p6_aaaaaaaaaaaaaaaa', 1000, 'asm_pp', 'passed', 20., 10.),
             _job('L1_p6_aaaaaaaaaaaaaaaa', 1000, 'native_hp', 'passed', 30., 8.),
             _job('bbbbbbbbbbbbbbbb', 4000, 'asm_pp', 'passed', 40., 30.),
             _job('cccccccccccccccc', 16000, 'asm_pp', 'passed', 60., 20.),
             # accurate but stopped by the residual target: no iterative time, not an LU win to count
             _job('dddddddddddddddd', 64000, 'native_hp', 'not_converged', None, None, residual=5e-10),
             _job('dddddddddddddddd', 64000, 'asm_pp', 'not_converged', None, None, residual=5e-10)]
    lu = {'aaaaaaaaaaaaaaaa': _lu(10., 16, 2., 16),
          'bbbbbbbbbbbbbbbb': _lu(80., 32, 20., 32),
          'cccccccccccccccc': _lu(300., 96, 40., 96)}
    rows = {r['system_id']: r for r in report.lu_rows(tuned, lu)}
    a = rows['L1_p6_aaaaaaaaaaaaaaaa']  # prefixed id matches the bare LU key
    assert (a['iterative_fresh_ms'], a['iterative_fresh_family']) == (20., 'asm_pp')
    assert (a['iterative_repeated_ms'], a['iterative_repeated_family']) == (8., 'native_hp')
    assert a['iterative_over_lu_fresh'] == 2. and a['iterative_over_lu_repeated'] == 4.
    d = rows['dddddddddddddddd']
    assert d['lu_status'] == 'not_measured' and d['iterative_outcome'] == 'fp64_floor'
    assert d['iterative_over_lu_repeated'] is None
    (cross,) = report.lu_crossover(list(rows.values()))
    assert cross['systems'] == 3 and (cross['dofs_min'], cross['dofs_max']) == (1000, 16000)
    assert cross['fresh_iterative_from_dofs'] == 4000      # 2.0, 0.5, 0.2
    assert cross['repeated_iterative_from_dofs'] == 16000  # 4.0, 1.5, 0.5
    assert abs(cross['repeated_ratio_largest'] - .5) < 1e-12
    assert cross['repeated_no_iterative_pass'] == 0
    assert 'trig.' in report.lu_table(list(rows.values())) and r'\bottomrule' in report.lu_crossover_table([cross])
