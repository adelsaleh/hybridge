"""Read the completed native ADR campaign for report rendering only.

No solver imports, matrix assembly, or numerical execution. Original campaigns
remain distinct; repeated measurements of an existing policy are not pooled or
selected by their timing. Prefer the record from the report's original campaign.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'run_outputs/solver_studies/adr_scaling_2026_09_17/native_hp_completion_2026_09_21'


def read(name):
    return json.loads((CAMPAIGN / name).read_text())


@lru_cache(None)
def records():
    check = read('validation.json')
    assert check['status'] == 'passed' and check['all_systems_attempted']
    result = read('all_native_results.json')
    assert len({r['system_id'] for r in result}) == 101
    for row in result:
        assert row['operator_sha256'] == row['system_id']
        if row['status'] == 'passed':
            assert row['samples'] and all(len(s['solves']) == 2 for s in row['samples'])
            assert all(s['passed'] and s['true_relative_residual'] <= 1e-10
                       for trial in row['samples'] + row['warmups'] for s in trial['solves'])
    return result


def metric(row, quantity='hot'):
    if row['status'] != 'passed':
        return None
    hot = [s['solves'][1] for s in row['samples']]
    return {'hot': statistics.mean(s['solve_ms'] for s in hot),
            'fresh': row['fresh_setup_solve_median_ms'],
            'setup': row['setup_median_ms'],
            'iterations': statistics.mean(s['iterations'] for s in hot)}[quantity]


def rows_for_suite(suite, campaign=None):
    """One saved record per system/policy, with the destination suite's mesh IDs."""
    candidates = {}
    for row in records():
        for origin in row['origins']:
            if origin['suite'] != suite or (campaign and origin['campaign'] != campaign):
                continue
            key = (row['system_id'], row['policy'], origin['campaign'])
            priority = 0 if row.get('source_artifact') == origin['source'] else 1
            if key in candidates and candidates[key][0] <= priority:
                continue
            enriched = deepcopy(row)
            enriched.update({k: origin[k] for k in
                             ('case', 'n', 'nominal_n', 'p', 'geometry', 'triangles', 'trace_dofs')})
            enriched['h'] = 2 / origin['nominal_n']
            enriched['origin'] = 'native_completion'
            enriched['candidate'] = 'native_hp' if row['policy'] == 'standard' else 'native_hp_robust'
            candidates[key] = (priority, enriched)
    return [v[1] for v in candidates.values()]


def overlay(rows, suite, campaign=None):
    """Keep every nonnative result and add available native policies, including NC."""
    return [r for r in rows if 'native_hp' not in r.get('candidate', '')] + rows_for_suite(suite, campaign)


def best_native(rows, quantity='hot'):
    passed = [r for r in rows if 'native_hp' in r.get('candidate', '') and metric(r, quantity) is not None]
    return min(passed, key=lambda r: metric(r, quantity)) if passed else None
