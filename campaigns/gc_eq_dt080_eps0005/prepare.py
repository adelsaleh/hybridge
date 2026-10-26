#!/usr/bin/env python3
"""Prepare smaller-smoothing CG6 inputs from the accepted archived potentials."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
from copy import deepcopy
import json
from pathlib import Path
import sys

repo = EQUILIBRIUM_REPO
sys.path.insert(0, str(repo))
from projects.diocotron.dolfinx.equiband.config import SolverConfig

root = output_directory('equilibrium_dt080_eps0005_20261016')
previous = root.parent / 'equilibrium_dt080_20261016'
stages = [(.04, 32), (.02, 32), (.01, 48), (.005, 48), (.005, 64)]


def label(ratio, quadrature):
    return 'eps' + format(ratio, '.6g').replace('.', 'p') + '_q' + str(quadrature)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError('Existing input differs: ' + str(path))
    else:
        path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    seeds = {}
    for geometry in ['star', 'uglypacman', 'iter', 'horseshoe']:
        source = previous / 'cg6_refined' / geometry
        record = json.loads((source / 'completed.json').read_text())
        config = json.loads((source / 'config.json').read_text())
        assert config['degree'] == 6 and record['target_reached']
        assert Path(config['mesh_file']).exists()
        seeds[geometry] = {
            'archive_id': record['archive_id'], 'checkpoint': record['checkpoint'],
            'config': str(source / 'config.json'), 'relative_epsilon': .08,
        }
        for ratio, quadrature in stages:
            target = deepcopy(config)
            target['band'].update(relative_epsilon=ratio,
                                  epsilon=ratio * target['band']['threshold_width_delta'])
            # Keep the accepted torsion/recovery quadrature while sharpening
            # only the nonlinear equilibrium; this preserves its flow atlas.
            target['auxiliary_quadrature_degree'] = (
                config['auxiliary_quadrature_degree'] or config['quadrature_degree'])
            target['quadrature_degree'] = quadrature
            target['maximum_iterations'] = 40
            path = root / 'inputs' / geometry / (label(ratio, quadrature) + '.json')
            save(path, target)
            parsed = SolverConfig.load(path)
            assert parsed.band.epsilon_over_delta == ratio
    save(root / 'seeds.json', seeds)
    save(root / 'schedule.json', {
        'target_relative_epsilon': .005, 'target_distance': .8,
        'mpi_ranks': 8, 'numerical_threads_per_rank': 1,
        'stages': [{'relative_epsilon': ratio, 'quadrature_degree': q,
                    'label': label(ratio, q)} for ratio, q in stages],
        'guiding_center_evolution_performed': False,
    })
    print('Validated 20 CG6 inputs; four archived CG6 potentials will initialize the continuation.')


if __name__ == '__main__':
    main()
