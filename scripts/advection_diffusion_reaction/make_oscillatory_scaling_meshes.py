#!/usr/bin/env python3
"""Match the prior ADR triangle-count ladder with canonical annular meshes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

TARGETS = (2048, 8192, 32768, 99458)


def main():
    """Adapt mesh size through the existing case generator, preserving every trial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    (out/'trials').mkdir(parents=True, exist_ok=True)
    generator = Path(__file__).with_name('make_oscillatory_geometry.py')
    records = []
    for target in TARGETS:
        boundary_points = 220 if target == 2048 else 400
        trials = out/'trials'/f'outer{boundary_points}'
        trials.mkdir(parents=True, exist_ok=True)
        size = (9.3/target)**0.5
        candidates = []
        for prior in trials.glob('star_*.json'):
            meta=json.loads(prior.read_text())
            candidates.append((abs(meta['triangles']/target-1),prior,meta))
        for attempt in range(16):
            above=[r for r in candidates if r[2]['triangles']>target]
            below=[r for r in candidates if r[2]['triangles']<target]
            if above and below:
                fine=max(r[2]['mesh_size'] for r in above)
                coarse=min(r[2]['mesh_size'] for r in below)
                if fine<coarse:size=(fine+coarse)/2
            if candidates:
                best=min((r for r in candidates if r[2]['triangles']<=100000),key=lambda r:r[0],default=None)
                if best and best[0]<.015:break
            size = float(f'{size:.7g}')
            meta_path = trials/f'star_h{size:g}.json'
            if not meta_path.exists():
                with (trials/f'generate_h{size:g}.log').open('w') as log:
                    subprocess.run([sys.executable, str(generator), '--output', str(trials),
                                    '--sizes', str(size), '--boundary-points', str(boundary_points)], stdout=log, stderr=subprocess.STDOUT, check=True)
            meta = json.loads(meta_path.read_text())
            candidates.append((abs(meta['triangles']/target-1), meta_path, meta))
            print('target',target,'attempt',attempt,'size',size,'triangles',meta['triangles'],flush=True)
            if abs(meta['triangles']/target-1) < .015 and meta['triangles'] <= 100000:
                break
            ratio = (meta['triangles']/target)**0.5
            # Aim slightly below the upper authorization bound when nearly at 100k.
            if target == max(TARGETS) and meta['triangles'] > 100000:
                ratio = max(ratio, 1.007)
            size *= ratio
        allowed = [r for r in candidates if r[2]['triangles'] <= 100000]
        _, selected, meta = min(allowed, key=lambda row:row[0])
        assert abs(meta['triangles']/target-1) < .03, 'Triangle-count target not reached'
        mesh_path = out/f'star_target{target}.npz'
        shutil.copy2(selected.with_suffix('.npz'), mesh_path)
        meta.update(target_triangles=target, path=str(mesh_path),
                    sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
                    selection='Closest canonical Gmsh mesh within 3% of the prior square count, at most 100,000 triangles. The coarsest polygon has 220 outer segments because 400 segments force over 3,000 triangles; other levels retain 400.')
        (out/f'star_target{target}.json').write_text(json.dumps(meta,indent=2)+'\n')
        records.append(meta)
        (out/'meshes.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Completed',[(r['target_triangles'],r['triangles']) for r in records],flush=True)


if __name__=='__main__':
    main()
