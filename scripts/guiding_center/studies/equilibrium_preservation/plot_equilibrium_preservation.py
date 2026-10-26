#!/usr/bin/env python3
"""Render saved preservation checks; no PDE solves or time integration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from hybridge.core.mesh import DGMesh
from hybridge.core.space import DGSpace
from hybridge.io.figures import publication_style, save_publication_figure
from hybridge.io.raster import RasterGeometry
from scripts.guiding_center.cases.imported_equilibrium import WindowEquilibrium

LABELS = dict(star='Star', uglypacman='Ugly Pac-Man', iter='ITER', horseshoe='Horseshoe')


def display_name(name):
    return LABELS.get(name.split('_')[0], name.replace('_', ' '))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summaries', type=Path, nargs='+')
    p.add_argument('--figure', type=Path, required=True)
    p.add_argument('--table', type=Path, required=True)
    args = p.parse_args(argv)
    runs = [(r, path.parent) for path in args.summaries for r in json.loads(path.read_text())['runs']]
    names = list(dict.fromkeys(r['name'] for r, _ in runs))
    selected = [max([(r, root) for r, root in runs if r['name'] == name and r['scheme'] == 'si-bdf2'],
                    key=lambda pair: pair[0]['order']) for name in names]
    panels = []
    for run, root in selected:
        with np.load(root/run['fields'], allow_pickle=False) as fields:
            mesh = DGMesh.from_arrays(fields['nodes'], fields['triangles'])
            space = DGSpace(mesh, int(fields['order']), basis_type=str(fields['basis']))
            raster = RasterGeometry.from_mesh(mesh, 320, 300)
            sampling = raster.sampling_matrix(space)
            change = sampling @ (fields['rho_final']-fields['rho_initial']).ravel()
            mask = raster.element_ids < 0
            change = np.ma.array(change, mask=mask).reshape(raster.height, raster.width)
        source = WindowEquilibrium(run['metrics']['equilibrium_import']['checkpoint'])
        points = raster.valid_points
        source_values = np.zeros(len(mask))
        source_values[raster.valid_pixels] = source.density(points[:, 0], points[:, 1])
        source_values = np.ma.array(source_values, mask=mask).reshape(raster.height, raster.width)
        boundary = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]]
        panels.append((run, raster, source_values, change, boundary))
        del source
    limit = max(float(np.max(np.abs(panel[3]))) for panel in panels)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    with publication_style(font_size=10):
        fig, axes = plt.subplots(3, len(names), figsize=(12, 8), constrained_layout=True)
        for j, (run, raster, density, change, boundary) in enumerate(panels):
            for row, values, cmap, lo, hi in ((0, density, 'viridis', 0., 1.),
                                            (1, change, 'RdBu_r', -limit, limit)):
                ax = axes[row, j]
                artist = ax.imshow(values, extent=raster.bounds, origin='upper', cmap=cmap,
                                   vmin=lo, vmax=hi, interpolation='nearest')
                ax.add_collection(LineCollection(boundary, colors='black', linewidths=.45))
                ax.set_aspect('equal')
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(display_name(run['name']))
                if j == len(names)-1:
                    fig.colorbar(artist, ax=axes[row, :], shrink=.75, pad=.01)
            ax = axes[2, j]
            for data, root in runs:
                if data['name'] != run['name']:
                    continue
                rows = [json.loads(line) for line in (root/data['diagnostics']).read_text().splitlines()]
                label = f"p={data['order']}, " + ('Euler' if data['scheme'] == 'si-euler' else 'BDF2')
                ax.plot([r['time'] for r in rows], [r['diocotron_rho_eq_relative_l2'] for r in rows],
                        label=label, marker='o', markersize=3,
                        color='#2166ac' if data['scheme'] == 'si-bdf2' else '#b35806',
                        linestyle='-' if data['order'] == run['order'] else '--')
            ax.set_xlabel('time')
            ax.grid(True)
            if j == 0:
                ax.set_ylabel(r'$\|\rho_h(t)-\rho_h(0)\|/\|\rho_h(0)\|$')
                ax.legend(fontsize=8)
        axes[0, 0].set_ylabel('CG2 source density')
        axes[1, 0].set_ylabel(r'$\rho_h(T)-\rho_h(0)$'+'\nBDF2, finest tested space')
        save_publication_figure(fig, args.figure, formats=('pdf', 'png'))
        plt.close(fig)
    lines = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
             r'Domain & $p$ & $N_K$ & $e_\phi$ & $s_h$ & $r_h$ & $d_\rho^{\rm E}$ & $d_\rho^{\rm B2}$ & $d_\phi^{\rm B2}$ \\',
             r'\midrule']
    def tex(value):
        return r'\num{'+f'{value:.2e}'+'}'
    for name in names:
        for order in sorted({r['order'] for r, _ in runs if r['name'] == name}):
            pair = {r['scheme']: r for r, _ in runs if r['name'] == name and r['order'] == order}
            bdf, euler = pair['si-bdf2'], pair['si-euler']
            m = bdf['metrics']
            values = [m[k] for k in ('equilibrium_poisson_cg_relative_l2', 'equilibrium_semilinear_relative_l2',
                                     'equilibrium_steady_transport_relative_l2')]
            values += [euler['metrics']['diocotron_rho_eq_relative_l2'], m['diocotron_rho_eq_relative_l2'],
                       m['diocotron_phi_eq_relative_l2']]
            lines.append(' & '.join([display_name(name), str(order), str(bdf['triangles']),
                                     *map(tex, values)])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
