#!/usr/bin/env python3
"""Display or save compact diagnostics for Strategy A fast residual logs.

Reads:
  ../logs/*_newton.csv
  ../logs/*_adapt.csv

Main focus:
  - resEuclid and resHm1 histories
  - timing per Newton step: solveTime, metricTime, stepTime
  - mass/maxRho/relRhoDesign histories
  - adaptation sizes and times

Usage:
  python postprocess_strategyA_fastres.py --show
  python postprocess_strategyA_fastres.py --save --figdir ../out/figures
  python postprocess_strategyA_fastres.py --runTag strategyA_adaptive_torsion_newton_fastres
"""
from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', type=Path, default=Path('../logs'))
    p.add_argument('--out', type=Path, default=Path('../out'))
    p.add_argument('--runTag', default='*')
    p.add_argument('--show', action='store_true')
    p.add_argument('--save', action='store_true')
    p.add_argument('--figdir', type=Path, default=None)
    p.add_argument('--dpi', type=int, default=180)
    return p.parse_args()


def load_csvs(logs: Path, run_tag: str, suffix: str) -> pd.DataFrame:
    if run_tag == '*':
        files = sorted(logs.glob(f'*{suffix}.csv'))
    else:
        files = sorted(logs.glob(f'{run_tag}{suffix}.csv'))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source'] = f.name
            frames.append(df)
        except Exception as e:
            print(f'[warn] failed reading {f}: {e}')
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if c not in {'record', 'runTag', 'status', 'source'}:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def save_show(fig, path: Path | None, show: bool, dpi: int):
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'[saved] {path}')
    if show:
        plt.show()
    plt.close(fig)


def plot_residuals(newton: pd.DataFrame, figdir: Path | None, show: bool, dpi: int):
    gdf = newton[newton['record'].eq('NEWTON')].copy()
    if gdf.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (src, ieps), g in gdf.groupby(['source', 'ieps'], dropna=False):
        g = g.sort_values('k')
        label0 = f'{Path(str(src)).stem} eps={int(ieps) if pd.notna(ieps) else ieps}'
        ax.semilogy(g['k'], g['resHm1'], marker='o', markersize=3, linewidth=1.2, label=label0 + ' H-1')
        ax.semilogy(g['k'], g['resEuclid'], marker='x', markersize=3, linewidth=1.0, linestyle='--', label=label0 + ' E')
    ax.set_xlabel('Newton iteration k')
    ax.set_ylabel('residual')
    ax.set_title('Newton residual histories')
    ax.grid(True, which='both', linewidth=0.4)
    ax.legend(fontsize=7)
    save_show(fig, None if figdir is None else figdir / 'newton_residuals_resE_resHm1.png', show, dpi)


def plot_timing(newton: pd.DataFrame, figdir: Path | None, show: bool, dpi: int):
    gdf = newton[newton['record'].eq('NEWTON')].copy()
    if gdf.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (src, ieps), g in gdf.groupby(['source', 'ieps'], dropna=False):
        g = g.sort_values('k')
        tag = f'{Path(str(src)).stem} eps={int(ieps) if pd.notna(ieps) else ieps}'
        for col, style in [('solveTime', '-'), ('metricTime', '--'), ('stepTime', ':')]:
            if col in g:
                ax.plot(g['k'], g[col], marker='o', markersize=3, linewidth=1.2, linestyle=style, label=f'{tag} {col}')
    ax.set_xlabel('Newton iteration k')
    ax.set_ylabel('seconds')
    ax.set_title('Per-step timing')
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=7)
    save_show(fig, None if figdir is None else figdir / 'newton_timing.png', show, dpi)


def plot_state(newton: pd.DataFrame, figdir: Path | None, show: bool, dpi: int):
    gdf = newton[newton['record'].eq('NEWTON')].copy()
    if gdf.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (src, ieps), g in gdf.groupby(['source', 'ieps'], dropna=False):
        g = g.sort_values('k')
        tag = f'{Path(str(src)).stem} eps={int(ieps) if pd.notna(ieps) else ieps}'
        for col in ['massRho', 'maxRho', 'relRhoDesign']:
            if col in g:
                ax.plot(g['k'], g[col], marker='o', markersize=3, linewidth=1.2, label=f'{tag} {col}')
    ax.set_xlabel('Newton iteration k')
    ax.set_ylabel('value')
    ax.set_title('State diagnostics')
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=7)
    save_show(fig, None if figdir is None else figdir / 'newton_state_metrics.png', show, dpi)


def plot_adapt(adapt: pd.DataFrame, figdir: Path | None, show: bool, dpi: int):
    if adapt.empty:
        return
    gdf = adapt[adapt['record'].eq('ADAPT')].copy()
    if gdf.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for src, g in gdf.groupby('source', dropna=False):
        g = g.sort_values('ieps')
        tag = Path(str(src)).stem
        ax.plot(g['ieps'], g['ntOld'], marker='o', label=tag + ' ntOld')
        ax.plot(g['ieps'], g['ntNew'], marker='o', label=tag + ' ntNew')
    ax.set_xlabel('eps stage')
    ax.set_ylabel('number of triangles')
    ax.set_title('Mesh adaptation size')
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_show(fig, None if figdir is None else figdir / 'adapt_mesh_size.png', show, dpi)

    fig, ax = plt.subplots(figsize=(8, 5))
    for src, g in gdf.groupby('source', dropna=False):
        g = g.sort_values('ieps')
        tag = Path(str(src)).stem
        for col in ['adaptTime', 'massRelChange', 'maxRhoRelChange']:
            if col in g:
                ax.plot(g['ieps'], g[col], marker='o', label=tag + ' ' + col)
    ax.set_xlabel('eps stage')
    ax.set_ylabel('value')
    ax.set_title('Adaptation diagnostics')
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_show(fig, None if figdir is None else figdir / 'adapt_diagnostics.png', show, dpi)


def main():
    a = args()
    show = a.show or not a.save
    figdir = a.figdir or (a.out / 'figures')
    figdir = figdir if a.save else None

    newton = load_csvs(a.logs, a.runTag, '_newton')
    adapt = load_csvs(a.logs, a.runTag, '_adapt')

    if newton.empty:
        raise SystemExit(f'No Newton CSV logs found in {a.logs} for runTag={a.runTag}')

    newton_rows = newton[newton['record'].eq('NEWTON')]
    print('\nNewton log rows:', len(newton_rows))
    cols = ['source', 'ieps', 'k', 'resEuclid', 'resHm1', 'massRho', 'maxRho', 'relRhoDesign', 'solveTime', 'metricTime', 'stepTime', 'status']
    cols = [c for c in cols if c in newton_rows]
    if cols:
        summary = newton_rows.groupby(['source', 'ieps'], dropna=False).tail(1)[cols]
        print('\nFinal row per eps stage:')
        print(summary.to_string(index=False))

    if a.save:
        a.out.mkdir(parents=True, exist_ok=True)
        newton.to_csv(a.out / 'parsed_strategyA_fastres_newton.csv', index=False)
        if not adapt.empty:
            adapt.to_csv(a.out / 'parsed_strategyA_fastres_adapt.csv', index=False)

    plot_residuals(newton, figdir, show, a.dpi)
    plot_timing(newton, figdir, show, a.dpi)
    plot_state(newton, figdir, show, a.dpi)
    plot_adapt(adapt, figdir, show, a.dpi)


if __name__ == '__main__':
    main()
