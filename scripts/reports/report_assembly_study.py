#!/usr/bin/env python3
"""Per-degree tables of the H200 assembly-kernel study.

``--q1``: what the Q1 campaign already measured on the H200, per polynomial degree, from
run_outputs/campaigns/h200_expanded_20261027/assembly: the advection kernels (fused and split3,
CSR/BSR, best block per p and the recommended block), the measured kernel resources, and the
Poisson assembly per policy and format (assembly, RHS refresh, reconstruction, selected launch
configuration, modeled occupancy), with the Blackwell references where the problems match.
Best configuration per cell only; no medians across block sizes.

``--figures``: graphs and tables of the assembly kernels for the H200 reports: H200 against the
RTX PRO 5000 Blackwell on matched meshes (AR, DR, ADR), each tuning step, launch shape, phases,
local solves and the tuned ADR campaign's ladder. No GPU work. Inputs are the tracked records under
docs/campaigns and docs/research/solver_studies, plus the Q1 timings above for ADR at p<=6. Writes
adr_scaling_2026_09_17/h200_tuned/assembly_*.{csv,tex} and run_outputs/.../figures/h200_assembly.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q1 = ROOT / 'run_outputs/campaigns/h200_expanded_20261027/assembly'
STUDIES = ROOT / 'docs/research/solver_studies'
RECOMMENDED = {p: 32 for p in range(1, 7)} | {7: 64, 8: 128, 9: 128}
CAMPAIGNS = ROOT / 'docs/campaigns'
REPORT = STUDIES / 'adr_scaling_2026_09_17/h200_tuned'
FIGURES = ROOT / 'run_outputs/solver_studies/adr_scaling_2026_09_17/figures/h200_assembly'
SLOTS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')
TUNED = dict(color=SLOTS[0], marker='o')
BEFORE = dict(color=SLOTS[1], marker='s')
BLACKWELL = dict(color=SLOTS[2], marker='D')
BLACKWELL_TENSOR = dict(color=SLOTS[3], marker='^')
MIDDLE = dict(color=SLOTS[3], marker='^')


def rows_of(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def fmt(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


def ar_controls(timings):
    """Kernel-control benchmark (rectangle nx=128, 32,768 triangles, eliminate, conflict-averaged): ms per block."""
    table = defaultdict(dict)
    for r in timings:
        m = re.match(r'ar-control-(fused|split3)-p(\d+)-n128-(legacy-lagrange|legendre-modal)-(coo|csr|bsr)-b(\d+)', r['task'])
        if not m:
            continue
        ms = r['rows'][0]['ms']
        value = ms.get('median_device_ms') or ms.get('device_ms(med)')
        if value:
            table[(int(m.group(2)), m.group(3), m.group(1), m.group(4))][int(m.group(5))] = value
    return table


def ar_showcase(timings):
    """Showcase star meshes (tangent walls, conflict-averaged, BSR): fused/split3 device ms per p and h."""
    table = defaultdict(list)
    for r in timings:
        m = re.match(r'ar-p(\d+)-h([\d.]+)-legacy-lagrange-native-float64', r['task'])
        if not m:
            continue
        for row in r['rows']:
            c = re.search(r'config=(\S+)', row['label']).group(1)
            if c in ('fused/float64', 'split3/float64') and 'device_ms' in row['ms']:
                table[(int(m.group(1)), m.group(2), c.split('/')[0])].append(row['ms']['device_ms'])
    return {k: statistics.median(v) for k, v in table.items()}


def blackwell_native():
    data = json.loads((STUDIES / 'advection_assembly_baseline_2026_10_03.data.json').read_text())
    return {(r['p'], r['path']): r['device_ms'] for r in data['native'] if 'p' in r and r.get('precision') == 'float64'}


def ar_attributes():
    table = {}
    for path in Q1.glob('tasks/ar-layout-p*-h0.0095-legacy-lagrange-b*/attempt_*/layouts/*.json'):
        m = re.search(r'ar-layout-p(\d+)-h0\.0095-legacy-lagrange-b(\d+)', str(path))
        kp = json.load(open(path))['results'][0]['kernel_profiles']
        table[(int(m.group(1)), int(m.group(2)))] = kp
    return table


def poisson_q1():
    table = {}
    for path in Q1.glob('tasks/dr-tune-p*-n128-*/attempt_*/poisson-kernels/*.json'):
        d = json.load(open(path))
        for c in d['cases']:
            tune = c['assembly_tuning']
            selected = tune['configuration']
            winner = next((x for x in tune['candidates'] if x['configuration'] == selected), None)
            table[(c['order'], c['trace_basis'], c['format'], c['policy'])] = dict(
                assembly=1000 * c['assembly_and_factor_construction']['kernel_median_seconds'],
                rhs=1000 * c['rhs_refresh']['kernel_median_seconds'] if c.get('rhs_refresh') else None,
                reconstruction=1000 * c['reconstruction']['kernel_median_seconds'] if c.get('reconstruction') else None,
                legacy=1000 * c['legacy_assembly']['kernel_median_seconds'] if c.get('legacy_assembly') else None,
                config=f"{selected['threads']}/{selected['columns']}/{int(selected['retained_coupling'])}",
                occupancy=winner['modeled_occupancy'] if winner else None,
                regs=winner['attributes']['num_regs'] if winner else None,
                candidates=len(tune['candidates']), triangles=c['triangles'])
    return table


def blackwell_poisson():
    """Oct-14 study (100,000 triangles, legendre-modal, block auto): kernel seconds per (p, policy, format)."""
    data = json.loads((STUDIES / 'poisson_raw_cupy_2026_10_14.data.json').read_text())
    cases = data['cases'] + data['matched_low_orders']['cases']
    return {(c['order'], r['policy'], r['format']): 1000 * r['assembly_kernel_seconds'] for c in cases for r in c['raw']}


def q1_report():
    timings = rows_of(Q1 / 'timings.jsonl')
    out = ['# H200 assembly performance per degree, from the Q1 campaign (preliminary)', '',
           f'Source: {Q1.relative_to(ROOT)} ({len(timings)} timing rows so far). Device times are CUDA-event medians of the',
           'kernel(s) only. Blackwell = RTX PRO 5000 (docs/research/solver_studies). Best block per cell; the recommended',
           'block (docs/backends/raw_cuda.md: 32 for p<=6, 64 at p=7, 128 at p=8-9) is shown next to it.', '']
    out += ['## AR assembly, showcase star mesh h=0.0095 (117,439 triangles), tangent walls, conflict-averaged upwind, BSR', '',
            '| p | fused ms (H200) | Blackwell fused | gain | split3 ms (H200) | Blackwell split3 | gain |', '|---|---:|---:|---:|---:|---:|---:|']
    show, bw = ar_showcase(timings), blackwell_native()
    for p in range(1, 10):
        f, s = show.get((p, '0.0095', 'fused')), show.get((p, '0.0095', 'split3'))
        bf, bs = bw.get((p, 'fused')), bw.get((p, 'split3'))
        out.append(f"| {p} | {fmt(f)} | {fmt(bf, 1)} | {fmt(bf/f, 1) if f and bf else '-'} | {fmt(s)} | {fmt(bs, 1)} | {fmt(bs/s, 1) if s and bs else '-'} |")
    out += ['', 'Same kernels at h=0.02 (40,715 triangles): ' + '; '.join(
        f"p{p} fused {fmt(show.get((p, '0.02', 'fused')))} / split3 {fmt(show.get((p, '0.02', 'split3')))} ms"
        for p in range(1, 10) if (p, '0.02', 'fused') in show or (p, '0.02', 'split3') in show), '']
    out += ['## AR kernel-control benchmark, rectangle 32,768 triangles, Dirichlet (eliminate), conflict-averaged upwind', '',
            'Cells: best block (ms) | ms at the recommended block.', '',
            '| p | basis | fused coo | fused csr | fused bsr | split3 bsr |', '|---|---|---|---|---|---|']
    controls = ar_controls(timings)
    for p in range(1, 10):
        for basis in ('legacy-lagrange', 'legendre-modal'):
            cells = []
            for path, f in (('fused', 'coo'), ('fused', 'csr'), ('fused', 'bsr'), ('split3', 'bsr')):
                d = controls.get((p, basis, path, f))
                if not d:
                    cells.append('-')
                    continue
                best = min(d, key=d.get)
                cells.append(f"b{best} {d[best]:.3f} \\| {fmt(d.get(RECOMMENDED[p]), 3)}")
            if any(c != '-' for c in cells):
                out.append(f"| {p} | {basis} | " + ' | '.join(cells) + ' |')
    out += ['', '## AR measured kernel resources (layouts benchmark, disk-tangent, plain upwind compile), h=0.0095', '',
            '| p | fused block | fused regs / local B / dyn shared KiB | split3 build / solve / scatter: block, regs, dyn KiB |', '|---|---|---|---|']
    for (p, b), kp in sorted(ar_attributes().items()):
        f, s = kp['fused'], kp['split3']
        out.append(f"| {p} | {b} | {f['registers']} / {f['local_size_bytes']} / {f['dynamic_shared_bytes']/1024:.1f} | "
                   + ' ; '.join(f"{s[k]['block_size']}, {s[k]['registers']}, {s[k]['dynamic_shared_bytes']/1024:.1f}" for k in ('build', 'solve', 'scatter')) + ' |')
    out += ['', 'Note: the production conflict-averaged fused kernel compiles to 254-255 registers at p=7-9 on sm_90',
            '(offline ptxas); these layout rows measured the plain-upwind compile. Step 0 of the study records the real ones.', '']
    out += ['## Poisson raw-CUDA assembly, rectangle 32,768 triangles (Q1 dr-tune, thorough on-device tuning)', '',
            'assembly / RHS refresh / reconstruction kernel ms; config = threads/columns/retain of the tuned winner; occ = its modeled occupancy.',
            'Blackwell (Oct-14, 100,000 triangles, legendre-modal) scaled to 32,768 triangles by element count, for CSR none / schur-cholesky.', '',
            '| p | basis | format | policy | assembly | RHS | reconstruction | config | occ | regs | Blackwell assembly (scaled) | gain |',
            '|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|']
    poisson, bwp = poisson_q1(), blackwell_poisson()
    for (p, basis, f, policy), v in sorted(poisson.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        ref = bwp.get((p, policy, f))
        scaled = ref * v['triangles'] / 100000 if ref and basis == 'legendre-modal' else None
        out.append(f"| {p} | {basis} | {f} | {policy} | {fmt(v['assembly'], 3)} | {fmt(v['rhs'], 3)} | {fmt(v['reconstruction'], 3)} | "
                   f"{v['config']} | {fmt(v['occupancy'], 2)} | {v['regs'] or '-'} | {fmt(scaled, 2)} | {fmt(scaled / v['assembly'], 1) if scaled else '-'} |")
    return '\n'.join(out) + '\n'


def ar_sweep(study, step, mesh='star:0.0095'):
    """AR assembly rows of one sweep: tangent walls, conflict-averaged upwind, BSR."""
    rows = rows_of(CAMPAIGNS / study / 'outputs' / step / 'sweep.jsonl')
    return [r for r in rows if r.get('status') == 'ok' and r.get('mesh') == mesh and r.get('boundary') == 'tangent'
            and r.get('variant') == 'conflict' and r.get('fmt') == 'bsr']


def fastest(rows, paths):
    """Fastest device time per degree over the given paths and every measured block size."""
    best = {}
    for r in rows:
        if r['path'] in paths:
            best[r['p']] = min(r['device_ms'], best.get(r['p'], float('inf')))
    return best


def dr_log(step, stem):
    """Poisson kernel ms per (p, policy) on 100,352 triangles, legendre-modal CSR: (assembly, reconstruction)."""
    with gzip.open(CAMPAIGNS / 'dr_review/outputs' / step / f'{stem}.log.gz', 'rt') as handle:
        text = handle.read()
    pattern = r'p=(\d+) triangles=100352 legendre-modal csr (\S+)\n\s+assembly ([\d.]+) ms; reconstruction ([\d.]+) ms'
    return {(int(p), policy): (float(a), float(r)) for p, policy, a, r in re.findall(pattern, text)}


def adr_kernel(step):
    """Raw ADR tensor kernel ms per (elements, coefficient, p, block), median of the timed samples."""
    table = {}
    for r in rows_of(CAMPAIGNS / 'asm_h200_tuning/outputs' / step / 'adr.jsonl'):
        if r.get('matrix_format') == 'bsr' and r.get('cache_local_factors') == 'none' and r.get('samples'):
            table[(r['elements'], r['coefficient'], r['order'], r['block_size'])] = (
                1000 * statistics.median(s['raw.kernel.device'] for s in r['samples']))
    return table


def adr_phases(step, order, block, coefficient='variable-full', elements=131072):
    """Phase seconds of the phase-clock build, grouped; medians of the samples."""
    row = next(r for r in rows_of(CAMPAIGNS / 'asm_h200_tuning/outputs' / step / 'adr.jsonl')
               if r.get('samples') and r['order'] == order and r['block_size'] == block
               and r['coefficient'] == coefficient and r['elements'] == elements)
    keys = [k for k in row['samples'][0] if k.startswith('raw.phase.')]
    phase = {k.removeprefix('raw.phase.'): statistics.median(s[k] for s in row['samples']) for k in keys}
    groups = dict(mass_factor=('mass_factor',), mass_solves=('schur_mass_solve', 'out_mass_solve'), local_matrix=('local_matrix',),
                  mass_assembly=('mass_assembly', 'mass_points'), coupling=('schur_coupling', 'out_coupling'))
    named = {g: 1000 * sum(phase[k] for k in members) for g, members in groups.items()}
    used = {k for members in groups.values() for k in members}
    named['rest'] = 1000 * sum(v for k, v in phase.items() if k not in used)
    return named


def adr_q1_kernels():
    """Q1 raw ADR kernel ms per (coefficient, p) on 32,768 triangles: best of blocks 32, 64, 128."""
    cells = defaultdict(dict)
    pattern = (r'mode=adr order=(\d+) basis=legacy-lagrange matrix_format=bsr coefficient=(\S+) block_size=(\d+) '
               r'cache_local_factors=none elements=32768')
    for r in rows_of(Q1 / 'timings.jsonl'):
        for row in r['rows'] if r.get('group') == 'adr' else ():
            m = re.match(pattern, row['label'])
            value = row['ms'].get('samples.raw.kernel.device(med)_ms')
            if m and value and int(m[3]) > 1:
                cells[(m[2], int(m[1]))][int(m[3])] = value
    return {k: min(v.values()) for k, v in cells.items()}


def adr_blackwell_kernels():
    """Blackwell raw ADR kernel ms per (coefficient, p) on 32,768 triangles: best of blocks 32, 64, 128."""
    cells = defaultdict(list)
    for r in rows_of(STUDIES / 'adr_assembly/raw_cuda_adr_tensor_2026_09_28.jsonl'):
        if r.get('elements') == 32768 and r.get('basis') == 'legacy-lagrange' and r.get('matrix_format') == 'bsr':
            cells[(r['coefficient'], r['order'])].append(1000 * r['median']['raw.kernel.device'])
    return {k: min(v) for k, v in cells.items()}


def cupy_route(step):
    """Warm wall ms of the vendored CuPy assembler per (route, p)."""
    table = {}
    for r in rows_of(CAMPAIGNS / 'cupy_adr_device/outputs' / step / 'points.jsonl'):
        if r.get('status') == 'ok':
            table[(r['route'], r['p'])] = statistics.median(c['wall_ms'] for c in r['calls'] if c['kind'] == 'warm')
    return table


def ladder_series():
    """Tuned campaign: median over the classes per (p, level) of both assembly routes, against trace DOFs."""
    cells = defaultdict(list)
    with (CAMPAIGNS / 'adr_h200_tuned/records/systems.csv').open() as handle:
        for r in csv.DictReader(handle):
            if r['status'] != 'passed' or r['level'] == 'L4b':
                continue
            for route, key in (('device', 'device_assembly_ms'), ('raw', 'raw_cuda_assembly_ms')):
                if r[key]:
                    cells[(route, int(r['p']), r['level'])].append((float(r['trace_dofs']), float(r[key])))
    series = defaultdict(list)
    for (route, p, _), values in cells.items():
        series[(route, p)].append((statistics.median(v[0] for v in values), statistics.median(v[1] for v in values)))
    return {k: sorted(v) for k, v in series.items()}


def assembly_data():
    """Every assembly series plotted or tabulated, from the records; values in ms."""
    baseline = json.loads((STUDIES / 'advection_assembly_baseline_2026_10_03.data.json').read_text())
    bw_native = {}
    for (p, _), value in blackwell_native().items():
        bw_native[p] = min(value, bw_native.get(p, float('inf')))
    bw_tensor = {r['p']: r['device_ms'] for r in baseline['tensor'] if r.get('config') == 'float64/cutensor/default/magma'}
    h200_native = fastest(ar_sweep('assembly_h200', '01-ar-base'), ('fused', 'split3'))
    h200_tensor = fastest(ar_sweep('ar_batched_lu', '12-auto-default-117k'), ('tensor',))
    bw_dr = blackwell_poisson()
    old, thorough, priors = (dr_log('02-auto-100k', 'dr_auto_100k'), dr_log('01-thorough-100k', 'dr_thorough_100k'),
                             dr_log('04-auto-priors-100k', 'dr_auto_priors_100k'))
    q1, bw_adr = adr_q1_kernels(), adr_blackwell_kernels()
    start, threads, noncompact, auto, final = (adr_kernel('30-adr-mass-auto'), adr_kernel('43-adr-threads'), adr_kernel('44-adr-noncompact'),
                                               adr_kernel('49-adr-auto-131k'), adr_kernel('54-adr-default-131k'))

    def before_adr(coefficient):
        cells = {p: v for (c, p), v in q1.items() if c == coefficient}
        for (elements, c, p, block), value in start.items():
            if elements == 32768 and c == coefficient and block == 128:
                cells[p] = value
        return cells

    def by_p(table, elements, coefficient, block):
        return {p: v for (e, c, p, b), v in table.items() if e == elements and c == coefficient and b == block}

    fused_runs = rows_of(CAMPAIGNS / 'ar_batched_lu/outputs/04-fused-variants-117k/solvers.jsonl')
    library_runs = rows_of(CAMPAIGNS / 'ar_batched_lu/outputs/02-solvers-117k/solvers.jsonl')

    def local_solve(rows, method):
        best = {}
        for r in rows:
            if r.get('kind') == 'solve' and r.get('status') == 'ok' and r['method'] == method:
                best[r['order']] = min(r['ms'], best.get(r['order'], float('inf')))
        return best

    def stages(study, step, path, mesh='star:0.0095', p=9):
        rows = [r for r in ar_sweep(study, step, mesh) if r['path'] == path and r['p'] == p and r.get('block') == 'auto']
        return rows[0]['stage_ms'], rows[0]['device_ms']

    volume = {name: {mesh: stages('asm_h200_tuning', f'37-volume-{name}', 'split3', mesh)[1] for mesh in ('star:0.0095', 'star:0.005')}
              for name in ('inline', 'dmma', 'cublas')}
    routes_32k, routes_200k = cupy_route('09-before-after-32k'), cupy_route('10-before-after-200k')
    return dict(
        ar=dict(blackwell=bw_native, blackwell_tensor=bw_tensor, before=h200_native, tuned=h200_tensor),
        dr=dict(blackwell={p: v for (p, policy, f), v in bw_dr.items() if policy == 'schur-lu' and f == 'csr'},
                before={p: v[0] for (p, policy), v in old.items() if policy == 'schur-lu'},
                thorough={p: v[0] for (p, policy), v in thorough.items() if policy == 'schur-lu'},
                tuned={p: v[0] for (p, policy), v in priors.items() if policy == 'schur-lu'},
                reconstruction={policy: {p: v[1] for (p, q), v in priors.items() if q == policy} for policy in ('none', 'schur-lu', 'schur-cholesky')}),
        adr=dict(blackwell={p: v for (c, p), v in bw_adr.items() if c == 'variable-full'},
                 before=before_adr('variable-full'), tuned=by_p(final, 32768, 'variable-full', 'auto'),
                 blackwell_constant={p: v for (c, p), v in bw_adr.items() if c == 'constant-full'},
                 before_constant=before_adr('constant-full')),
        ar_360k=dict(before=fastest(ar_sweep('ar_batched_lu', '08-e2e-360k-fused', 'star:0.005'), ('split3',)),
                     middle=fastest(ar_sweep('ar_batched_lu', '09-e2e-360k-magma', 'star:0.005'), ('tensor',)),
                     tuned=fastest(ar_sweep('ar_batched_lu', '08-e2e-360k-fused', 'star:0.005'), ('tensor',))),
        ar_2m=dict(before=fastest(ar_sweep('ar_batched_lu', '18-2m-p9-parity', 'star:0.00199'), ('fused',))[9],
                   tuned=fastest(ar_sweep('ar_batched_lu', '21-2m-auto', 'star:0.00199'), ('tensor',))[9]),
        adr_131k=dict(before=by_p(start, 131072, 'variable-full', 128), middle=by_p(auto, 131072, 'variable-full', 'auto'),
                      tuned=by_p(final, 131072, 'variable-full', 'auto'), constant=by_p(final, 131072, 'constant-full', 'auto')),
        launch=dict(p9_compact={**{b: v for (e, c, p, b), v in start.items() if (e, c, p) == (131072, 'variable-full', 9)},
                                **{b: v for (e, c, p, b), v in threads.items() if (e, c, p) == (131072, 'variable-full', 9) and b == 256}},
                    p9_noncompact={b: v for (e, c, p, b), v in noncompact.items() if (e, c, p) == (131072, 'variable-full', 9)},
                    p8_compact={**{b: v for (e, c, p, b), v in start.items() if (e, c, p) == (131072, 'variable-full', 8)},
                                **{b: v for (e, c, p, b), v in threads.items() if (e, c, p) == (131072, 'variable-full', 8) and b == 256}}),
        phases=dict(before=adr_phases('41-adr-phase-131k', 9, 128), middle=adr_phases('46-adr-phase-best', 9, 512),
                    tuned=adr_phases('52-mass-inverse-phase', 9, 'auto')),
        local_solve=dict(magma=local_solve(fused_runs, 'magma_getrf_getrs'), cublas=local_solve(library_runs, 'cublas'),
                         coop=local_solve(library_runs, 'coop'), fused=local_solve(fused_runs, 'fused')),
        ar_stages=dict(before=stages('ar_batched_lu', '06-e2e-117k-fused', 'split3')[0], middle=stages('ar_batched_lu', '07-e2e-117k-magma', 'tensor')[0],
                       tuned=stages('ar_batched_lu', '06-e2e-117k-fused', 'tensor')[0]),
        volume=volume,
        cupy=dict(host_32k={p: v for (r, p), v in routes_32k.items() if r == 'host'}, device_32k={p: v for (r, p), v in routes_32k.items() if r == 'device'},
                  host_200k={p: v for (r, p), v in routes_200k.items() if r == 'host'}, device_200k={p: v for (r, p), v in routes_200k.items() if r == 'device'}),
        ladder=ladder_series())


def _line(ax, cells, label, style, **extra):
    points = sorted(cells.items())
    return ax.plot(*zip(*points), label=label, **{**style, **extra})


def assembly_figures(data):
    """Write the assembly figures; returns their paths."""
    sys.path.insert(0, str(ROOT))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from hybridge.io.figures import publication_style, save_publication_figure

    plain = matplotlib.ticker.FuncFormatter(lambda value, _: f'{value:g}')
    # Narrow log axes label 2, 3 and 5 of each decade; the others would collide.
    sparse = matplotlib.ticker.FuncFormatter(lambda value, _: f'{value:g}' if f'{value:e}'[0] in '235' else '')

    def finish(ax, xlabel=None, ylabel=None, log=True):
        ax.grid(True, which='major', axis='y')
        if log:
            ax.set_yscale('log')
            low, high = ax.get_ylim()
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(sparse if high / low < 30 else matplotlib.ticker.NullFormatter())
        if xlabel:
            ax.set_xlabel(xlabel)
            if xlabel.startswith('degree'):
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if ylabel:
            ax.set_ylabel(ylabel)

    with publication_style(font_size=8.5):
        # H200 against the Blackwell on matched meshes.
        fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.5), sharex=True, sharey='row', height_ratios=(1.6, 1))
        panels = (('ar', 'AR, about 117k triangles'), ('dr', 'DR, about 100k triangles'), ('adr', 'ADR, variable tensor, 32,768 triangles'))
        for column, (key, title) in enumerate(panels):
            d, top, low = data[key], axes[0, column], axes[1, column]
            _line(top, d['blackwell'], 'Blackwell', BLACKWELL)
            if key == 'ar':
                _line(top, d['blackwell_tensor'], 'Blackwell, cuTENSOR + MAGMA', BLACKWELL_TENSOR)
            _line(top, d['before'], 'H200, before tuning', BEFORE)
            _line(top, d['tuned'], 'H200, tuned', TUNED)
            top.set_title(title, fontsize=8.5)
            finish(top, ylabel='assembly kernel (ms)' if column == 0 else None)
            best = {p: min(d['blackwell'].get(p, float('inf')), d.get('blackwell_tensor', {}).get(p, float('inf')))
                    for p in d['blackwell']}
            same = {p: d['blackwell'][p] / d['before'][p] for p in d['blackwell'] if p in d['before']}
            tuned = {p: best[p] / d['tuned'][p] for p in best if p in d['tuned']}
            _line(low, same, 'before tuning', BEFORE)
            if tuned:
                _line(low, tuned, 'tuned', TUNED)
            last = tuned or same
            p = max(last)
            low.annotate(f'{last[p]:.0f}x', (p, last[p]), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=7.5)
            low.axhline(1, color='#667085', lw=.7)
            low.set_ylim(0, 17)
            low.set_xticks(range(1, 10))
            finish(low, xlabel='degree $p$', ylabel='Blackwell / H200 time' if column == 0 else None, log=False)
        handles = [plt.Line2D([], [], label=label, **style) for label, style in (
            ('Blackwell', BLACKWELL), ('Blackwell, cuTENSOR + MAGMA', BLACKWELL_TENSOR), ('H200, before tuning', BEFORE), ('H200, tuned', TUNED))]
        fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.03))
        fig.tight_layout(rect=(0, 0, 1, .97))
        save_publication_figure(fig, FIGURES / 'assembly_machines')
        plt.close(fig)

        # Each tuning step.
        fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.0))
        ax = axes[0, 0]
        _line(ax, data['ar_360k']['before'], 'split3 (raw)', BEFORE)
        _line(ax, data['ar_360k']['middle'], 'tensor + MAGMA', MIDDLE)
        _line(ax, data['ar_360k']['tuned'], 'tensor + fused solve', TUNED)
        ax.set_title('AR, 360,379 triangles', fontsize=8.5)
        finish(ax, 'degree $p$', 'assembly (ms)')
        ax.legend(frameon=False)
        ax = axes[0, 1]
        _line(ax, data['dr']['before'], 'quick tuner, Blackwell priors', BEFORE)
        _line(ax, data['dr']['thorough'], 'thorough tuner', MIDDLE, markersize=7, markerfacecolor='none')
        _line(ax, data['dr']['tuned'], 'quick tuner, FP64-rich priors', TUNED)
        ax.set_title('DR, 100,352 triangles', fontsize=8.5)
        finish(ax, 'degree $p$', 'assembly (ms)')
        ax.legend(frameon=False)
        ax = axes[1, 0]
        _line(ax, data['adr_131k']['before'], '128 threads, mass LU', BEFORE)
        _line(ax, data['adr_131k']['middle'], 'wide launch, mass LU', MIDDLE)
        _line(ax, data['adr_131k']['tuned'], 'wide launch, explicit inverse', TUNED)
        ax.set_title('ADR, variable tensor, 131,072 triangles', fontsize=8.5)
        ax.set_xticks((7, 8, 9))
        finish(ax, 'degree $p$', 'assembly (ms)')
        ax.legend(frameon=False)
        ax = axes[1, 1]
        _line(ax, data['cupy']['host_32k'], 'host route, 32,768', BEFORE)
        _line(ax, data['cupy']['device_32k'], 'device route, 32,768', TUNED)
        _line(ax, data['cupy']['host_200k'], 'host route, 199,712', BEFORE, linestyle=':', markerfacecolor='white')
        _line(ax, data['cupy']['device_200k'], 'device route, 199,712', TUNED, linestyle=':', markerfacecolor='white')
        ax.set_title('CuPy ADR assembly (the matrices solved)', fontsize=8.5)
        finish(ax, 'degree $p$', 'warm assembly (ms)')
        ax.set_ylim(top=3e5)
        ax.legend(frameon=False, ncol=2, loc='upper left', columnspacing=1)
        fig.tight_layout()
        save_publication_figure(fig, FIGURES / 'assembly_tuning_steps')
        plt.close(fig)

        # Launch shape, phases, local solves, stages, volume operator, cache policies.
        fig, axes = plt.subplots(2, 3, figsize=(7.3, 5.0))
        ax = axes[0, 0]
        _line(ax, data['launch']['p9_compact'], '$p=9$, compact', BEFORE)
        _line(ax, data['launch']['p9_noncompact'], '$p=9$, non-compact', TUNED)
        _line(ax, data['launch']['p8_compact'], '$p=8$, compact', MIDDLE)
        ax.set_xscale('log', base=2)
        ax.set_xticks((32, 64, 128, 256, 512), labels=('32', '64', '128', '256', '512'))
        ax.minorticks_off()
        ax.set_title('ADR launch, 131,072 triangles', fontsize=8.5)
        finish(ax, 'threads per block', 'assembly (ms)')
        ax.set_ylim(top=5e4)
        ax.legend(frameon=False, loc='upper right')

        def stacked(ax, bars, parts, labels):
            left = [0.0] * len(bars)
            for slot, part in enumerate(parts):
                values = [b.get(part, 0.0) for _, b in bars]
                ax.barh(range(len(bars)), values, left=left, height=.62, color=SLOTS[slot], edgecolor='white', linewidth=.8, label=labels[part])
                left = [a + b for a, b in zip(left, values)]
            ax.set_yticks(range(len(bars)), labels=[name for name, _ in bars])
            ax.invert_yaxis()
            ax.grid(True, axis='x')
            ax.set_ylim(len(bars) - .5, -2.6)

        ax = axes[0, 1]
        phases = data['phases']
        stacked(ax, (('128, LU', phases['before']), ('512, LU', phases['middle']), ('512, inverse', phases['tuned'])),
                ('mass_factor', 'mass_solves', 'local_matrix', 'mass_assembly', 'coupling', 'rest'),
                dict(mass_factor='mass LU / inverse', mass_solves='mass solves', local_matrix='local matrix',
                     mass_assembly='mass assembly', coupling='coupling', rest='rest'))
        ax.set_title('ADR phases, $p=9$, 131,072', fontsize=8.5)
        ax.set_xlabel('ms')
        ax.legend(frameon=False, ncol=2, fontsize=6, loc='upper right', columnspacing=.7, handlelength=1)
        ax = axes[0, 2]
        solve = data['local_solve']
        _line(ax, solve['cublas'], 'cuBLAS', dict(color=SLOTS[2], marker='D'))
        _line(ax, solve['coop'], 'cooperative', MIDDLE)
        _line(ax, solve['magma'], 'MAGMA', BEFORE)
        _line(ax, solve['fused'], 'fused (tuned)', TUNED)
        ax.set_xticks((6, 7, 8, 9))
        ax.set_title('AR local solve, about 117k', fontsize=8.5)
        finish(ax, 'degree $p$', 'batched solve (ms)')
        ax.legend(frameon=False)
        ax = axes[1, 0]
        s = data['ar_stages']
        stacked(ax, (('split3', s['before']), ('tensor, MAGMA', s['middle']), ('tensor, fused', s['tuned'])),
                ('prepare', 'build', 'solve', 'condense', 'scatter'),
                dict(prepare='prepare', build='build', solve='solve', condense='condense', scatter='scatter'))
        ax.set_title('AR stages, $p=9$, about 117k', fontsize=8.5)
        ax.set_xlabel('ms')
        ax.legend(frameon=False, ncol=2, fontsize=6, loc='upper right', columnspacing=.7, handlelength=1)
        ax = axes[1, 1]
        names = (('inline', 'inline (default)', BEFORE), ('dmma', 'hand DMMA', MIDDLE), ('cublas', 'cuBLAS', TUNED))
        for slot, (key, label, style) in enumerate(names):
            values = [data['volume'][key][mesh] for mesh in ('star:0.0095', 'star:0.005')]
            ax.bar([i + (slot - 1) * .27 for i in range(2)], values, width=.25, color=style['color'], label=label)
        ax.set_xticks((0, 1), labels=('about 117k', '360,379'))
        ax.set_title('AR split3 volume, $p=9$ (opt-in)', fontsize=8.5)
        ax.grid(True, axis='y')
        ax.set_ylabel('assembly (ms)')
        ax.set_ylim(0, 430)
        ax.legend(frameon=False, loc='upper left')
        ax = axes[1, 2]
        recon = data['dr']['reconstruction']
        _line(ax, recon['none'], 'no cache', BEFORE)
        _line(ax, recon['schur-lu'], 'Schur LU', MIDDLE)
        _line(ax, recon['schur-cholesky'], 'Schur Cholesky', TUNED)
        ax.set_title('DR reconstruction, 100,352', fontsize=8.5)
        finish(ax, 'degree $p$', 'reconstruction (ms)')
        ax.set_ylim(top=800)
        ax.legend(frameon=False, loc='upper left')
        fig.tight_layout()
        save_publication_figure(fig, FIGURES / 'assembly_details')
        plt.close(fig)

        # The tuned ADR campaign's ladder.
        fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.6), sharey=True)
        for ax, p in zip(axes, (2, 6, 9)):
            for route, label, style in (('device', 'CuPy device route (matrices solved)', BEFORE), ('raw', 'raw tensor kernel (timing only)', TUNED)):
                points = data['ladder'].get((route, p))
                if points:
                    ax.plot(*zip(*points), label=label, **style)
            ax.set_xscale('log')
            ax.set_title(f'$p={p}$', fontsize=8.5)
            finish(ax, 'trace DOFs', 'assembly (ms)' if p == 2 else None)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.06))
        fig.tight_layout(rect=(0, 0, 1, .96))
        save_publication_figure(fig, FIGURES / 'assembly_ladder')
        plt.close(fig)

    FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CAMPAIGNS / 'adr_recovery/outputs/29580/adr_recovery_h200.pdf', FIGURES / 'adr_recovery_h200.pdf')
    return sorted(FIGURES.glob('*.pdf'))


def _ms(value):
    if value is None:
        return '--'
    if value >= 100:
        return f'{value:,.0f}'.replace(',', '\\,')
    return f'{value:.1f}' if value >= 10 else f'{value:.2f}'


def assembly_tables(data):
    """Write the matched and tuning-step tables and their CSV; returns the CSV rows."""
    rows, lines = [], [r'\begin{tabular}{lrrrrrr}', r'\toprule',
                       r'family and mesh & $p$ & Blackwell & H200 before & H200 tuned & BW / before & BW / tuned\\', r'\midrule']
    groups = (('ar', r'AR, about 117k triangles'), ('dr', r'DR, about 100k triangles'), ('adr', r'ADR variable tensor, 32,768 triangles'))
    for index, (key, title) in enumerate(groups):
        d = data[key]
        if index:
            lines.append(r'\addlinespace[2pt]')
        degrees = sorted(set(d['blackwell']) | set(d['tuned']))
        for number, p in enumerate(degrees):
            bw = min(d['blackwell'].get(p, float('inf')), d.get('blackwell_tensor', {}).get(p, float('inf')))
            bw = None if bw == float('inf') else bw
            before, tuned = d['before'].get(p), d['tuned'].get(p)
            native = d['blackwell'].get(p)
            same = native / before if native and before else None
            gain = bw / tuned if bw and tuned else None
            rows.append(dict(family=key, p=p, blackwell_best_ms=bw, blackwell_native_ms=native, h200_before_ms=before, h200_tuned_ms=tuned,
                             blackwell_over_before=same, blackwell_over_tuned=gain))
            lines.append(f"{title if number == 0 else ''} & {p} & {_ms(bw)} & {_ms(before)} & {_ms(tuned)} & "
                         f"{f'{same:.1f}' if same else '--'} & {f'{gain:.1f}' if gain else '--'}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    (REPORT / 'assembly_kernels_table.tex').write_text('\n'.join(lines) + '\n')
    with (REPORT / 'assembly_kernels.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    steps = [r'\begin{tabular}{llrrrrr}', r'\toprule', r'kernel and mesh & steps & $p$ & start & middle & tuned & gain\\', r'\midrule']

    def add(title, names, degrees, first, middle, last):
        for number, p in enumerate(degrees):
            a, b, c = first.get(p), (middle or {}).get(p), last.get(p)
            steps.append(f"{title if number == 0 else ''} & {names if number == 0 else ''} & {p} & {_ms(a)} & {_ms(b)} & {_ms(c)} & {a / c:.1f}\\\\")
        steps.append(r'\addlinespace[2pt]')

    r = data['ar_360k']
    add('AR, 360,379', 'split3; tensor + MAGMA; tensor + fused', (6, 7, 8, 9), r['before'], r['middle'], r['tuned'])
    add('AR, 1,995,756', 'raw fused; tensor + fused', (9,), {9: data['ar_2m']['before']}, None, {9: data['ar_2m']['tuned']})
    r = data['dr']
    add('DR, 100,352', 'Blackwell priors; thorough; FP64-rich priors', (1, 2, 4, 6, 7, 8, 9), r['before'], r['thorough'], r['tuned'])
    r = data['adr_131k']
    add('ADR variable tensor, 131,072', '128 threads; wide launch; explicit inverse', (7, 8, 9), r['before'], r['middle'], r['tuned'])
    r = data['cupy']
    add('CuPy ADR, 32,768', 'host route; device route', (1, 4, 7, 9), r['host_32k'], None, r['device_32k'])
    add('CuPy ADR, 199,712', 'host route; device route', (4, 7), r['host_200k'], None, r['device_200k'])
    steps[-1:] = [r'\bottomrule', r'\end{tabular}']
    (REPORT / 'assembly_steps_table.tex').write_text('\n'.join(steps) + '\n')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--q1', action='store_true')
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'run_outputs/campaigns/assembly_h200/00a-q1-baseline.md')
    args = parser.parse_args(argv)
    if args.figures:
        data = assembly_data()
        for row in assembly_tables(data):
            print(row)
        for key in ('ar_360k', 'ar_2m', 'adr_131k', 'volume', 'local_solve'):
            print(key, data[key])
        print('phases', {k: {a: round(b) for a, b in v.items()} for k, v in data['phases'].items()})
        print('figures', *assembly_figures(data), sep='\n  ')
    if args.q1:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        text = q1_report()
        args.output.write_text(text)
        print(text)
        print(f'written {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
