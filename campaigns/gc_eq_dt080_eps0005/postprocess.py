#!/usr/bin/env python3
"""Render only accepted target-epsilon fields and summarize their diagnostics."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
import json
from pathlib import Path
import sys

sys.path.insert(0, str(EQUILIBRIUM_REPO))
from projects.diocotron.studies.equilibrium_catalog.render import plt, panel, plot_data

root = output_directory('equilibrium_dt080_eps0005_20261016')
selected = json.loads((root / 'selected.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout='constrained')
rows = ['# Smaller-smoothing CG6 equilibria', '',
        'Target epsilon/delta = 0.005, target D_T = 0.8. Each solve starts from an accepted archived potential.',
        'Only accepted target-epsilon equilibria appear below. No HDG evolution is performed.', '',
        '| Geometry | Triangles | Quadrature degree | D_T | PDE residual | Transition length / cell diameter |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
artist = None
for ax, (geometry, title) in zip(axes.ravel(), [('star', 'Star'), ('uglypacman', 'Ugly Pac-Man'),
                                               ('iter', 'ITER'), ('horseshoe', 'Horseshoe')]):
    if geometry not in selected:
        ax.set_axis_off()
        ax.set_title(title + ': pending')
        continue
    record = selected[geometry]
    case = root / 'accepted' / geometry
    config = json.loads((case / 'config.json').read_text())
    resolution = json.loads((case / 'transition_resolution.json').read_text())
    assert config['degree'] == 6 and config['band']['relative_epsilon'] == .005
    data = plot_data(record['checkpoint'])
    title += f": $D_T={record['distance']:.6f}$\nCG6, $\\varepsilon/\\delta=0.005$"
    artist = panel(ax, data, 'density', title)
    figure, individual_axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    for individual_ax, field in zip(individual_axes, ['density', 'potential']):
        im = panel(individual_ax, data, field, title + '\n' + field)
        figure.colorbar(im, ax=individual_ax, shrink=.78)
    figure.savefig(root / (geometry + '_density_potential.png'), dpi=220)
    plt.close(figure)
    rows.append(f"| {geometry} | {len(data[0].triangles):,} | {config['quadrature_degree']} | "
                f"{record['distance']:.9f} | {record['dual_residual']:.3e} | "
                f"{resolution['minimum_epsilon_over_slope_h']:.4f} |")
if artist is not None:
    fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=.7, label=r'$\rho=W_\varepsilon(\phi)$')
    fig.savefig(root / 'density_bands.png', dpi=220)
    fig.savefig(root / 'density_bands.pdf', dpi=300)
plt.close(fig)
rows += ['', '## Quadrature comparisons', '',
         'These compare two nonlinear assembly rules at fixed mesh, degree, smoothing and target placement.',
         'They do not establish spatial mesh convergence of the sharper transitions.', '',
         '| Geometry | Potential relative L2 difference | Density relative L2 difference | Gradient relative L2 difference |',
         '| --- | ---: | ---: | ---: |']
for geometry in selected:
    path = root / 'checks' / (geometry + '_quadrature.json')
    if path.exists():
        data = json.loads(path.read_text())['comparisons'][-1]
        rows.append('| ' + geometry + ' | ' + ' | '.join(
            f"{data[field]['relative_l2_difference']:.3e}" for field in ['potential', 'density', 'gradient']) + ' |')
    else:
        rows.append('| ' + geometry + ' | pending | pending | pending |')
rows += ['', '## Reusable archive entries', '']
for geometry, record in selected.items():
    rows.append('- ' + geometry + ': `' + record['archive_id'] + '`')
(root / 'RESULTS.md').write_text('\n'.join(rows) + '\n')
print(root / 'RESULTS.md')
