#!/usr/bin/env python3
"""Summarize saved evidence; does not run a solver or alter an archive."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

ROOT = output_directory('equilibrium_dt080_20261016')
GEOMETRIES = ['star', 'uglypacman', 'iter', 'horseshoe']


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def atomic(path, text):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def main():
    cases = []
    for level in ['placement', 'resolved', 'ultra', 'cg4', 'cg6', 'cg6_refined']:
        for geometry in GEOMETRIES:
            directory = ROOT/level/geometry
            completion = read(directory/'completed.json')
            if completion is None:
                continue
            certificate = read(directory/(completion['name']+'.json'))['certificate']
            with np.load(completion['checkpoint'], allow_pickle=False) as checkpoint:
                metadata = json.loads(str(checkpoint['metadata'].item()))
            record = dict(level=level, geometry=geometry, **completion,
                triangles=metadata['num_cells'], dofs=metadata['num_dofs'], degree=metadata['source_order'],
                guards=certificate['guards'], distance_spread=certificate['distance'],
                transition_resolution=read(directory/'transition_resolution.json'),
                refinement=read(directory/'refinement.json'),
                distance_refinement=read(directory/'distance_refinement.json'))
            cases.append(record)
    report = dict(updated=datetime.now(timezone.utc).isoformat(), target_distance=.8,
        definition='zeta_T=s/L; fixed boundary-arclength trajectory weights',
        guiding_center_evolution_performed=False, cases=cases,
        driver_completion=read(ROOT/'high_order.completed.json'),
        saved_field_checks=read(ROOT/'high_order_checks.completed.json'),
        selected=read(ROOT/'selected.json'))
    atomic(ROOT/'summary.json', json.dumps(report, indent=2)+'\n')
    lines = ['# Saved equilibrium results', '', 'Updated '+report['updated']+'.', '',
        'Target: D_T = 0.8 in the established torsion-flow coordinate. '
        'Only completed, archived states appear below. Placement is intermediate; '
        'the finer levels and their comparisons determine spatial resolution. '
        'Degree 6 and the mesh-size request apply to DOLFINx only. HDG preservation '
        'tests are excluded from this campaign.', '',
        '| Geometry | Level | Triangles | FE DOFs | D_T | PDE residual | Mean width along flow |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for case in cases:
        lines.append(f"| {case['geometry']} | {case['level']} (CG{case['degree']}) | {case['triangles']:,} | "
            f"{case['dofs']:,} | {case['distance']:.9f} | {case['dual_residual']:.2e} | "
            f"{case['mean_thickness']:.7f} |")
    lines += ['', '## Refinement evidence', '',
        'Relative L2 differences use the finer field as denominator and the '
        'higher quadrature order saved in each comparison (CG4 against CG6 for the new family). Widths and epsilon '
        'are fixed between levels; midpoint is fitted again at each level.', '',
        '| Geometry | Potential | Density | Gradient | Density error estimate: quadrature change | Smallest transition length / crossing-cell diameter | Distance sampling check |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for case in cases:
        if case['level'] not in ['ultra', 'cg6', 'cg6_refined']:
            continue
        comparison = case['refinement']
        errors = (['pending']*3 if comparison is None else [
            f"{comparison['comparisons'][-1][name]['relative_l2_difference']:.3e}"
            for name in ['potential', 'density', 'gradient']])
        quadrature_change = 'pending'
        if comparison is not None and len(comparison['comparisons']) > 1:
            estimates = [item['density']['relative_l2_difference'] for item in comparison['comparisons']]
            quadrature_change = f'{abs(estimates[-1]-estimates[-2])/max(estimates[-1], 1e-300):.3e}'
        resolution = case['transition_resolution']
        ratio = 'unavailable' if resolution is None else f"{resolution['minimum_epsilon_over_slope_h']:.3f}"
        sampling = case['distance_refinement']
        status = 'pending' if sampling is None or 'passed' not in sampling else ('passed' if sampling['passed'] else 'failed')
        lines.append('| '+ ' | '.join([case['geometry']+' / '+case['level'], *errors, quadrature_change, ratio, status])+' |')
    lines += ['', '## Reusable archive identifiers', '']
    for case in cases:
        lines.append(f"- {case['geometry']} / {case['level']}: `{case['archive_id']}`")
    lines += ['', 'These are equilibrium inputs. No guiding-center preservation '
        'result has been computed for this family.', '']
    atomic(ROOT/'RESULTS.md', '\n'.join(lines))
    print(f"Summarized {len(cases)} accepted states", flush=True)


if __name__ == '__main__':
    main()
