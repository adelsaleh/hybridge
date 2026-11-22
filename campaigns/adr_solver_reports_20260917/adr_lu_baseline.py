"""Match confirmed CPU LU measurements to ADR report systems (no solver execution).

Use the pilot-selected thread count for each objective, then report its independent
confirmation. Never reselect threads from confirmation timings. Origins in the
campaign inventory preserve the identity of overlapping report suites.
"""

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))

from functools import lru_cache
from pathlib import Path
import csv
import json

ROOT = _campaign_workspace_path('hybridge')
# The confirmed timings are tracked with the PARDISO campaign records; the untracked run-output
# copy is the original location and still wins where it exists.
_RUN_OUTPUT = ROOT / 'run_outputs/solver_studies/adr_scaling_2026_09_17/pardiso_tuned_2026_09_22'
CAMPAIGN = _RUN_OUTPUT if (_RUN_OUTPUT / 'confirmed_timings.csv').exists() else (
    ROOT / 'docs/campaigns/adr_pardiso/records/pardiso_tuned_2026_09_22')
COLOR = '#30343b'
LABEL = r'P-LU$_n$ ($n$ threads at points)'
CASE_NAMES = {'Transport dominated': 'advection_dominated',
              'Variable velocity': 'variable_velocity', 'High diffusion': 'trigonometric'}
SUITES = {'focused_million_dof': 'focused_endpoints'}


@lru_cache(None)
def records():
    with (CAMPAIGN / 'confirmed_timings.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        assert row['status'] == 'passed' and row['measurement_phase'] == 'confirmation'
        row['origins'] = json.loads(row['origins'])
        row['selected_for'] = json.loads(row['selected_for'])
    return rows


def lookup(row, suite, metric='hot', campaign=None):
    """Return one independently confirmed timing, failing on ambiguous identity."""
    objective = 'reused' if metric in ('hot', 'mean', 'reused') else 'fresh'
    suite = SUITES.get(suite, suite)
    case = CASE_NAMES.get(row['case'], row['case'])
    geometry = row.get('geometry', 'square')
    matches = {}
    for direct in records():
        if objective not in direct['selected_for']:
            continue
        for origin in direct['origins']:
            if (origin['suite'], origin['case'], origin['geometry'], int(origin['p'])) != (suite, case, geometry, int(row['p'])):
                continue
            if campaign and origin['campaign'] != campaign:
                continue
            dofs = row.get('trace_dofs', row.get('dofs'))
            if dofs is not None:
                if int(origin['trace_dofs']) != int(dofs):
                    continue
            elif row.get('triangles') is not None:
                if int(origin['triangles']) != int(row['triangles']):
                    continue
            elif int(origin.get('nominal_n', origin.get('n', -1))) != int(row.get('nominal_n', row.get('n', -2))):
                continue
            matches[direct['system_id']] = direct
    if len(matches) != 1:
        raise ValueError(f'Expected one LU match, found {len(matches)}: {suite}, {case}, {geometry}, p={row["p"]}, {row.get("n")}, {row.get("triangles")}')
    direct = next(iter(matches.values()))
    return dict(system_id=direct['system_id'], threads=int(direct['threads']),
                fresh=float(direct['fresh_median_ms']), hot=float(direct['reused_mean_ms']),
                setup=float(direct['setup_median_ms']),
                minimum=float(direct['reused_min_ms']), maximum=float(direct['reused_max_ms']),
                peak_rss=float(direct['peak_process_rss_gib']))


def alias(threads, tex=False):
    return (r'$\mathrm{P\!-\!LU}_{' if tex else r'P-LU$_{') + str(threads) + '}$'


def cell(row, suite, metric, scale=1):
    direct = lookup(row, suite, metric)
    return alias(direct['threads'], tex=True) + f" {direct[metric]/scale:.3f}"


def handle():
    from matplotlib.lines import Line2D
    return Line2D([], [], color=COLOR, marker='v', linestyle=':', label=LABEL)


def curve(ax, rows, suite, metric, xkey, campaign=None, lookup_fn=None):
    """Add a neutral LU timing curve, with selected thread count at every point.

    ``lookup_fn(row, suite, metric, campaign)`` replaces the confirmed-timing lookup for another
    LU campaign (the H200 node's); it returns ``None`` for a system without a measurement, which
    is then left out of the curve.
    """
    if metric == 'iterations':
        return []
    metric = 'hot' if metric == 'mean' else metric
    chosen = {}
    for row in rows:
        direct = (lookup_fn or lookup)(row, suite, metric, campaign)
        if direct is None:
            continue
        x = xkey(row) if callable(xkey) else row[xkey]
        chosen[direct['system_id']] = (x, direct)
    points = sorted(chosen.values(), key=lambda item: item[0])
    ax.plot([x for x, d in points], [d[metric] for x, d in points],
            color=COLOR, marker='v', linestyle=':', linewidth=1.1, markersize=4,
            label=LABEL, zorder=5)
    ax.margins(y=.18)
    for x, direct in points:
        ax.annotate(str(direct['threads']), (x, direct[metric]), xytext=(0, -10),
                    textcoords='offset points', ha='center', fontsize=5.8, color=COLOR,
                    bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=.3))
    return points
