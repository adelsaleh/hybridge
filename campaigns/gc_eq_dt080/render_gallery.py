#!/usr/bin/env python3
"""Four-panel density figure from accepted polynomial checkpoints."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(EQUILIBRIUM_REPO))
from projects.diocotron.studies.equilibrium_catalog.render import plt, panel, plot_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', choices=['placement', 'resolved', 'ultra', 'cg4', 'cg6', 'cg6_refined', 'final'], default='ultra')
    args = parser.parse_args()
    root = output_directory('equilibrium_dt080_20261016')/args.level
    figure, axes = plt.subplots(2, 2, figsize=(11, 10), layout='constrained')
    artist = None
    for ax, (geometry, title) in zip(axes.ravel(), [
            ('star', 'Star'), ('uglypacman', 'Ugly Pac-Man'),
            ('iter', 'ITER'), ('horseshoe', 'Horseshoe')]):
        completed = root/geometry/'completed.json'
        if not completed.exists():
            ax.set_axis_off()
            ax.set_title(title+' — pending')
            continue
        record = json.loads(completed.read_text())
        data = plot_data(record['checkpoint'])
        triangles = len(data[0].triangles)
        degree = json.loads((root/geometry/'config.json').read_text())['degree']
        artist = panel(ax, data, 'density',
            f"{title}: $D_T={record['distance']:.6f}$\n{triangles:,} triangles, CG{degree} potential")
    if artist is None:
        raise ValueError('no accepted states at this level')
    figure.colorbar(artist, ax=axes.ravel().tolist(), shrink=.7,
                    label=r'$\rho=W_\varepsilon(\phi)$')
    destination = root/'density_bands.png'
    figure.savefig(destination, dpi=220)
    figure.savefig(destination.with_suffix('.pdf'), dpi=300)
    plt.close(figure)
    print(destination)


if __name__ == '__main__':
    main()
