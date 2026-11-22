"""Stage the compact text records of every finished stage, then collect them with the shared helper.

Per task: DONE/FAILED markers, measurement.json, harness.json, spans.json,
microbenchmarks.json, kernel_probe.json, variants.json, spec.json; per stage: provenance.jsonl,
COMPLETE markers, the source snapshot. Bulk per-step timing files stay in run_outputs.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.records import collect  # noqa: E402

KEEP = {'DONE.json', 'FAILED.json', 'measurement.json', 'harness.json', 'spans.json',
        'microbenchmarks.json', 'kernel_probe.json', 'variants.json', 'spec.json', 'mesh.json', 'calibration_events.json'}


def main():
    root = REPO/'run_outputs'/'poisson_element_cost'
    staging = root/'compact'
    if staging.exists():
        shutil.rmtree(staging)
    for stage in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ('logs', 'compact')):
        for path in stage.rglob('*'):
            if not path.is_file() or 'cuda-tuning' in path.parts or 'warmup' in path.parts:
                continue
            relative = path.relative_to(root)
            top_level = len(relative.parts) == 2
            if path.name in KEEP or (top_level and path.suffix in ('.jsonl', '.json')) or 'source' in relative.parts:
                target = staging/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    # The shared collector copies the data directory's top level plus explicit globs.
    print(collect('poisson_element_cost', staging, include=('*/*', '*/*/*', '*/*/*/*'),
                  status='investigation complete 2026-11-02; completion markers per stage'))


if __name__ == '__main__':
    main()
