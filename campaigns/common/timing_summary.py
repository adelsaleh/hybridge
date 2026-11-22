"""Compact timing rows from heterogeneous benchmark artifacts, for live campaign logs.

Benchmarks write different layouts (``results``/``cases``/``runs`` lists, JSONL
rows, per-sample dictionaries in seconds). This reduces any of them to rows of
identifying labels and median milliseconds without understanding each format;
the artifacts themselves remain the authoritative record.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

ROW_LISTS = ('results', 'cases', 'runs', 'rows')
LABEL_KEYS = ('case', 'mode', 'equation', 'order', 'p', 'config', 'precision', 'contraction', 'local_solver',
              'local_assembly', 'basis', 'trace_basis', 'matrix_format', 'format', 'coefficient', 'block_size',
              'cache_policy', 'cache_local_factors', 'phase', 'flux', 'elements', 'nx')
# Most informative first; anything else that is a time fills the remaining slots.
PRIORITY = ('device', 'kernel', 'total', 'solve', 'build', 'assembly', 'setup', 'reconstruction', 'wall')


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _rows(document):
    for key in ROW_LISTS:
        value = document.get(key)
        if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
            return value
    return [document]


def _label(row):
    parts = []
    nested = row.get('configuration') if isinstance(row.get('configuration'), dict) else {}
    for key in LABEL_KEYS:
        value = row.get(key, nested.get(key))
        if isinstance(value, (str, int)) and not isinstance(value, bool) and value != '':
            parts.append(f'{key}={value}')
    return ' '.join(parts) or 'row'


def _metrics(row, prefix='', depth=0):
    """Median milliseconds for every timing leaf; seconds are converted."""
    found = {}
    for key, value in row.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict) and depth < 2:
            found.update(_metrics(value, name + '.', depth + 1))
        elif _finite(value):
            if key.endswith('_ms') or key == 'ms':
                found[name] = value
            elif key in ('median_seconds', 'seconds') or key.endswith('_seconds') and 'samples' not in key:
                found[name.replace('_seconds', '').replace('seconds', '') + '_ms'] = 1000 * value
            elif key.endswith('_median'):
                # benchmark_advection_tsle_bsr.py reports bare medians in seconds.
                found[name + '_ms'] = 1000 * value
        elif isinstance(value, list) and value and key.endswith(('samples_ms', '_ms')) and all(_finite(v) for v in value):
            found[name.replace('_samples', '').replace('samples_', '') + '(med)'] = statistics.median(value)
        elif key == 'samples' and isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            # Per-sample dictionaries in seconds (raw CUDA ADR); keep the headline stages.
            for stage in sorted({k for sample in value for k in sample}):
                numbers = [sample[stage] for sample in value if _finite(sample.get(stage))]
                if numbers and any(word in stage for word in ('device', 'total', 'wall', 'kernel')):
                    found[f'{name}.{stage}(med)_ms'] = 1000 * statistics.median(numbers)
    return found


def _rank(name):
    lowered = name.lower()
    return next((index for index, word in enumerate(PRIORITY) if word in lowered), len(PRIORITY))


def summarize(path, *, max_rows=12, max_metrics=6):
    """Return [(label, {metric: ms})] for a JSON or JSONL artifact; [] if it has no timings."""
    path = Path(path)
    text = path.read_text()
    documents = ([json.loads(line) for line in text.splitlines() if line.strip()]
                 if path.suffix == '.jsonl' else [json.loads(text)])
    rows = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        for row in _rows(document):
            metrics = _metrics(row)
            # A benchmark's own scalar median makes the median of its sample list redundant.
            metrics = {name: value for name, value in metrics.items()
                       if not (name.endswith('(med)') and name[:-5] in metrics)}
            if metrics:
                chosen = sorted(metrics, key=lambda name: (_rank(name), name))[:max_metrics]
                rows.append((_label(row), {name: metrics[name] for name in chosen}))
    return rows[:max_rows]


def format_rows(rows):
    return [f'  {label}: ' + ' '.join(f'{name}={value:.3f}' for name, value in metrics.items())
            for label, metrics in rows]
