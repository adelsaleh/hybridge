#!/usr/bin/env python3
"""Postprocess Strategy A pre-band/readaptive Euclidean-residual logs.

Reads:
  ../logs/*_newton.csv
  ../logs/*_adapt.csv

Displays or saves:
  - resEuclid histories
  - per-step timing
  - mass/maxRho/relRhoDesign histories
  - adaptation trigger metrics
  - adaptation search tries and accepted mesh sizes
  - adaptation preservation errors: mass/area/energy changes

Usage:
  python postprocess_strategyA_readapt_fastres.py --show
  python postprocess_strategyA_readapt_fastres.py --save --figdir ../out/figures
  python postprocess_strategyA_readapt_fastres.py --runTag strategyA_preband_readapt_euclid_fastres
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
        tag = f'{Path(str(src)).stem} eps={int(ieps) if pd.notna(ieps) else ieps}'
        ax.semilogy(g['k'], g['resEuclid'], marker='o', markersize=3, linewidth=1.2, label=tag)
    ax.set_xlabel('Newton iteration k')
    ax.set_ylabel('Euclidean weak residual resE')
    ax.set_title('Newton residual histories')
    ax.grid(True, which='both', linewidth=0.4)
    ax.legend(fontsize=7)
    save_show(fig, None if figdir is None else figdir / 'newton_residuals_resE.png', show, dpi)


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


def plot_adapt_size(adapt: pd.DataFrame, figdir: Path | None, show: bool, dpi: int):
    if adapt.empty:
        return
    gdf = adapt[adapt['record'].isin(['PREADAPT', 'ADAPT'])].copy()
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
    ax.set_title('Mesh size after preadapt/readapt')
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_show(fig, None if figdir is None else figdir / 'adapt_mesh_size.png', show, dpi)


def plot_adapt_trigger(adapt: pd.DataFrame, figdir: Path | None, show: bool, dpi: int):
    if adapt.empty:
        return
    gdf = adapt[adapt['record'].isin(['ADAPT', 'ADAPT_SKIP', 'ADAPT_TRY'])].copy()
    if gdf.empty:
        return

    # For trigger overview use final ADAPT and SKIP rows; tries duplicate trigger values.
    overview = gdf[gdf['record'].isin(['ADAPT', 'ADAPT_SKIP'])].copy()
    if overview.empty:
        overview = gdf[gdf['record'].eq('ADAPT_TRY')].groupby(['source', 'ieps'], as_index=False).tail(1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for src, g in overview.groupby('source', dropna=False):
        g = g.sort_values('ieps')
        tag = Path(str(src)).stem
        for col in ['activeDrift', 'relDesignJump']:
            if col in g:
                ax.plot(g['ieps'], g[col], marker='o', label=tag + ' ' + col)
    ax.set_xlabel('eps stage')
    ax.set_ylabel('trigger metric')
    ax.set_title('Readaptation trigger metrics')
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_show(fig, None if figdir is None else figdir / 'adapt_trigger_metrics.png', show, dpi)


def plot_adapt_preservation(adapt: pd.DataFrame, figdir: Path | None, show: bool, dpi: int):
    if adapt.empty:
        return
    gdf = adapt[adapt['record'].isin(['PREADAPT', 'ADAPT', 'ADAPT_TRY'])].copy()
    if gdf.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for src, g in gdf.groupby('source', dropna=False):
        g = g.sort_values(['ieps', 'adaptTry'])
        tag = Path(str(src)).stem
        for col in ['massRelChange', 'activeAreaRelChange', 'energyRelChange']:
            if col in g:
                ax.semilogy(g.index, g[col].clip(lower=1e-16), marker='o', linewidth=1.2, markersize=3, label=tag + ' ' + col)
    ax.set_xlabel('adaptation record index')
    ax.set_ylabel('relative change')
    ax.set_title('Preservation error across adaptation')
    ax.grid(True, which='both', linewidth=0.4)
    ax.legend(fontsize=8)
    save_show(fig, None if figdir is None else figdir / 'adapt_preservation_errors.png', show, dpi)


def plot_adapt_search(adapt: pd.DataFrame, figdir: Path | None, show: bool, dpi: int):
    if adapt.empty:
        return
    tries = adapt[adapt['record'].eq('ADAPT_TRY')].copy()
    if tries.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for (src, ieps), g in tries.groupby(['source', 'ieps'], dropna=False):
        g = g.sort_values('adaptTry')
        tag = f'{Path(str(src)).stem} eps={int(ieps) if pd.notna(ieps) else ieps}'
        ax.plot(g['adaptTry'], g['ntNew'], marker='o', label=tag + ' ntNew')
    ax.set_xlabel('adaptation try')
    ax.set_ylabel('triangles')
    ax.set_title('Coarsest-acceptable adaptation search')
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_show(fig, None if figdir is None else figdir / 'adapt_search_triangles.png', show, dpi)


def print_summaries(newton: pd.DataFrame, adapt: pd.DataFrame):
    newton_rows = newton[newton['record'].eq('NEWTON')]
    print('\nNewton log rows:', len(newton_rows))
    cols = ['source', 'ieps', 'k', 'resEuclid', 'massRho', 'maxRho', 'relRhoDesign', 'solveTime', 'metricTime', 'stepTime', 'status']
    cols = [c for c in cols if c in newton_rows]
    if cols and not newton_rows.empty:
        summary = newton_rows.groupby(['source', 'ieps'], dropna=False).tail(1)[cols]
        print('\nFinal Newton row per eps stage:')
        print(summary.to_string(index=False))

    if not adapt.empty:
        rows = adapt[adapt['record'].isin(['PREADAPT', 'ADAPT', 'ADAPT_SKIP'])]
        cols = [
            'source', 'record', 'ieps', 'ntOld', 'ntNew', 'adaptNeeded',
            'adaptAccepted', 'activeDrift', 'relDesignJump',
            'massRelChange', 'activeAreaRelChange', 'energyRelChange',
            'adaptTry', 'adaptSearchScale', 'adaptTime', 'status'
        ]
        cols = [c for c in cols if c in rows]
        if cols and not rows.empty:
            print('\nAdaptation summary:')
            print(rows[cols].to_string(index=False))


def main():
    a = args()
    show = a.show or not a.save
    figdir = a.figdir or (a.out / 'figures')
    figdir = figdir if a.save else None

    newton = load_csvs(a.logs, a.runTag, '_newton')
    adapt = load_csvs(a.logs, a.runTag, '_adapt')

    if newton.empty:
        raise SystemExit(f'No Newton CSV logs found in {a.logs} for runTag={a.runTag}')

    print_summaries(newton, adapt)

    if a.save:
        a.out.mkdir(parents=True, exist_ok=True)
        newton.to_csv(a.out / 'parsed_strategyA_readapt_fastres_newton.csv', index=False)
        if not adapt.empty:
            adapt.to_csv(a.out / 'parsed_strategyA_readapt_fastres_adapt.csv', index=False)

    plot_residuals(newton, figdir, show, a.dpi)
    plot_timing(newton, figdir, show, a.dpi)
    plot_state(newton, figdir, show, a.dpi)
    plot_adapt_size(adapt, figdir, show, a.dpi)
    plot_adapt_trigger(adapt, figdir, show, a.dpi)
    plot_adapt_preservation(adapt, figdir, show, a.dpi)
    plot_adapt_search(adapt, figdir, show, a.dpi)


if __name__ == '__main__':
    main()
