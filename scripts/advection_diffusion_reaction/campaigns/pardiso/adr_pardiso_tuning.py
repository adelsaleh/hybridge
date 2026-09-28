"""Campaign-only thread selection from validated pilot timings.

Fresh and reused-factor objectives are selected independently. Selection never
looks at confirmation measurements; failed or incomplete pilots cannot win.
"""
from __future__ import annotations

import hashlib
import json
import random

from scripts.advection_diffusion_reaction.campaigns.pardiso.adr_pardiso_inventory import timing_metrics

OBJECTIVES = {'fresh': 'fresh_median_ms', 'reused': 'reused_mean_ms'}


def default_candidates(physical_cores):
    """A one-thread-to-physical-core ladder, with no assumed small-system cutoff."""
    if physical_cores < 1:
        raise ValueError('physical_cores must be positive')
    return sorted({t for t in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, physical_cores)
                   if t <= physical_cores})


def candidate_order(system_id, candidates, seed):
    """Reproducibly shuffle each system's pilot order to vary ordering bias."""
    result = sorted(candidates)
    random.Random(f'{seed}:{system_id}').shuffle(result)
    return result


def select_threads(system_id, records):
    """Freeze separate winners using pilots only; break exact ties by fewer cores.

    The fingerprint includes pilot results and their attempt paths. Retrying a
    pilot creates a new selection identity and cannot silently reuse an old
    confirmation. These are best-tested settings, not universal optima.
    """
    if any(r.get('measurement_phase') != 'tuning' or r['system_id'] != system_id for r in records):
        raise ValueError('Thread selection requires pilots from exactly one system')
    candidates = sorted((dict(threads=r['threads'], status=r['status'],
                              metrics=timing_metrics(r), diagnostic_output=r['diagnostic_output'])
                         for r in records), key=lambda r: r['threads'])
    if len({r['threads'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate thread candidates')
    eligible = [r for r in candidates if r['metrics'] is not None]
    selected = {}
    for name, key in OBJECTIVES.items():
        selected[name] = min(eligible, key=lambda r: (r['metrics'][key], r['threads']))['threads'] if eligible else None
    result = dict(system_id=system_id, status='selected' if eligible else 'no_passing_candidate',
                  selected_threads=selected, candidates=candidates,
                  rule='minimum validated pilot metric; fewer threads on exact ties; independent confirmation')
    result['fingerprint'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def confirmation_objectives(selection, threads):
    return [name for name, chosen in selection['selected_threads'].items() if chosen == threads]
