"""H200 (tuned kernels) versus RTX PRO 5000 Blackwell on the ADR study systems; no solver execution.

Inputs, all tracked:
- Blackwell: ``docs/research/solver_studies/adr_scaling_2026_09_17`` (smooth ``scaling.csv``,
  ``oscillatory/scaling.csv``, ``native_completion/results.csv``, focused ``comparison/results.csv``,
  family selections in ``synthesis/selected_results.csv``, PARDISO LU through ``adr_lu_baseline``);
- H200: ``docs/campaigns/adr_h200_tuned/records`` (``solver_jobs.csv`` from the tuned campaign,
  ``stock_unified_rows.csv`` from the stock-kernel unified run 21958 on the same H200);
- genoa LU: ``docs/campaigns/adr_pardiso/records/{unified_l5_pardiso_v2,h200_tuned_lu}`` (PyPardiso on a
  genoa CPU node, not the H200 node's host, on systems the device campaigns exported; a missing
  directory is skipped).
- genoa host pipeline (end-to-end table only): the ``cpu_adr_scaling`` campaign's records
  (``HYBRIDGE_CPU_ADR_RECORDS``, default the main checkout's ``run_outputs/solver_studies/cpu_adr_scaling``,
  untracked there); the derived ``end_to_end.csv`` is tracked and kept when they are absent.

Matching: an H200 job and a Blackwell measurement are the same configuration when geometry, case,
triangle count and degree agree and one of the H200 candidate's recorded aliases names the
Blackwell candidate (the unified plan derived its candidates and aliases from these studies).
Family comparisons use each machine's fastest passed member of a family on the same system.

Writes ``h200_tuned/{matched,family_best,extension,outcomes,node_lu,node_lu_crossover}.csv``, LaTeX tables and ``validation.json``
beside the study, and figures under ``run_outputs/.../figures/h200_tuned``.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adr_lu_baseline as lu  # noqa: E402
from campaigns.adr_solver_reports_20260917.adr_report_labels import PMG_TEX  # noqa: E402
from hybridge.io.figures import publication_style, save_publication_figure  # noqa: E402

STUDY = ROOT/'docs/research/solver_studies/adr_scaling_2026_09_17'
OUT = STUDY/'h200_tuned'
RECORDS = ROOT/'docs/campaigns/adr_h200_tuned/records'
PARDISO = tuple(ROOT/'docs/campaigns/adr_pardiso/records'/label for label in ('unified_l5_pardiso_v2', 'h200_tuned_lu'))
FIGURES = ROOT/'run_outputs/solver_studies/adr_scaling_2026_09_17/figures/h200_tuned'
CPU_PIPELINE = Path(os.environ.get('HYBRIDGE_CPU_ADR_RECORDS', ROOT/'run_outputs/solver_studies/cpu_adr_scaling'))
FAMILY_GROUP = {'asm_pp': 'asm', 'bj_pp': 'bj', 'amgx': 'amg', 'native_hp': 'native'}
GROUP_LABEL = {'asm': 'ASM+PP', 'bj': 'BJ+PP', 'amg': 'AMGX', 'native': r'$p$MG--AMG', 'lu': 'P-LU'}
FOCUSED_ALIAS = {'AMG/DILU--BiCGSTAB': 'amgx_bsr_pbicgstab_amg_multicolor_dilu', '$hp$-BSR--GMRES': 'native_hp_bsr'}
SUITE_ORDER = ('smooth', 'oscillatory', 'focused', 'native_completion')
LADDER_CASE = 'trigonometric'
COLORS = {'asm': '#1f6f8b', 'bj': '#99c24d', 'amg': '#c2185b', 'native': '#f29e4c', 'lu': '#30343b'}


def _f(value):
    try:
        return float(value) if value not in ('', None) else None
    except ValueError:
        return None


def _amortized(cold, repeated):
    """Two solves per setup, as in the Blackwell CSVs: (setup + first + second) / 2."""
    return (cold+repeated)/2 if cold is not None and repeated is not None else None


def blackwell():
    """Normalized Blackwell measurements keyed by (geometry, case, triangles, p, alias)."""
    rows = []
    for r in csv.DictReader(open(STUDY/'scaling.csv')):
        rows.append(dict(suite='smooth', geometry='square', case=r['case'], triangles=int(r['triangles']), p=int(r['p']),
                         dofs=int(r['trace_dofs']), alias=r['candidate'], status=r['status'], iterations=_f(r['iterations']),
                         setup=_f(r['setup']), cold=_f(r['fresh']), repeated=_f(r['mean']), amortized=_f(r['amortized'])))
    for r in csv.DictReader(open(STUDY/'oscillatory/scaling.csv')):
        rows.append(dict(suite='oscillatory', geometry=r['geometry'], case=r['case'], triangles=int(r['triangles']), p=int(r['p']),
                         dofs=int(r['trace_dofs']), alias=r['candidate'], status=r['status'], iterations=_f(r['iterations']),
                         setup=_f(r['setup']), cold=_f(r['fresh']), repeated=_f(r['mean']), amortized=_f(r['amortized'])))
    for r in csv.DictReader(open(STUDY/'native_completion/results.csv')):
        cold, rep = _f(r['fresh_median_ms']), _f(r['hot_mean_ms'])
        rows.append(dict(suite='native_completion', geometry=r['geometry'], case=r['case'], triangles=int(r['triangles']),
                         p=int(r['p']), dofs=int(r['trace_dofs']), alias='native_hp_'+r['policy'], status=r['status'],
                         iterations=_f(r['iterations']), setup=_f(r['setup_median_ms']), cold=cold, repeated=rep,
                         amortized=_amortized(cold, rep)))
    # Focused suite: names and PP degrees come from the synthesis selection of the same rows.
    selected = list(csv.DictReader(open(STUDY/'synthesis/selected_results.csv')))
    triangles = {(r['case'], int(r['dofs']), int(r['p'])): int(r['triangles']) for r in selected if r['triangles']}
    degree = {(r['case'], r['method']): r['pp_degree'] for r in selected if r['pp_degree']}
    for r in csv.DictReader(open(STUDY/'comparison/results.csv')):
        method, case = r['method'], lu.CASE_NAMES.get(r['problem'], r['problem'])
        if method in FOCUSED_ALIAS:
            alias = FOCUSED_ALIAS[method]
        elif method in ('ASM+PP--GMRES', 'BJ+PP--GMRES') and (r['problem'], method) in degree:
            alias = f"{'asm' if method.startswith('ASM') else 'bj'}_pp_raw_fused_d{degree[(r['problem'], method)]}"
        else:
            continue
        tri = triangles.get((r['problem'], int(r['trace_dofs']), int(r['p'])))
        if tri is None:
            continue
        cold, rep = _f(r['fresh_ms']), _f(r['hot_ms'])
        rows.append(dict(suite='focused', geometry='square', case=case, triangles=tri, p=int(r['p']), dofs=int(r['trace_dofs']),
                         alias=alias, status=r['status'], iterations=_f(r['iterations']), setup=_f(r['setup_ms']),
                         cold=cold, repeated=rep, amortized=_amortized(cold, rep)))
    index = {}
    for r in sorted(rows, key=lambda r: SUITE_ORDER.index(r['suite'])):
        index.setdefault((r['geometry'], r['case'], r['triangles'], r['p'], r['alias']), r)
    return index, selected


def h200(name):
    rows = list(csv.DictReader(open(RECORDS/name)))
    for r in rows:
        for k in ('p', 'triangles', 'trace_dofs'):
            r[k] = int(float(r[k])) if r[k] else None
        for k in ('iterations', 'setup_ms', 'cold_ms', 'repeated_ms', 'amortized_ms', 'warm_0.01_ms', 'warm_0.01_its',
                  'bsr_borrowed'):
            r[k] = _f(r.get(k))
    return rows


def ratio(a, b):
    return a/b if a and b else None


def matched_rows(bw, tuned, stock):
    stock_by = {(r['system_id'], r['base_candidate']): r for r in stock}
    out = []
    for t in tuned:
        for alias in t['aliases'].split('|'):
            b = bw.get((t['geometry'], t['case'], t['triangles'], t['p'], alias))
            if b is None:
                continue
            s = stock_by.get((t['system_id'], t['base_candidate']))
            row = dict(level=t['level'], geometry=t['geometry'], case=t['case'], p=t['p'], triangles=t['triangles'],
                       dofs=t['trace_dofs'], family=t['family'], group=FAMILY_GROUP[t['family']], alias=alias,
                       blackwell_suite=b['suite'], blackwell_status=b['status'], h200_status=t['status'],
                       stock_status=s['status'] if s else None,
                       it_blackwell=b['iterations'], it_stock=s['iterations'] if s else None, it_h200=t['iterations'],
                       setup_blackwell=b['setup'], setup_stock=s['setup_ms'] if s else None, setup_h200=t['setup_ms'],
                       cold_blackwell=b['cold'], cold_stock=s['cold_ms'] if s else None, cold_h200=t['cold_ms'],
                       repeated_blackwell=b['repeated'], repeated_stock=s['repeated_ms'] if s else None, repeated_h200=t['repeated_ms'],
                       amortized_blackwell=b['amortized'], amortized_stock=s['amortized_ms'] if s else None,
                       amortized_h200=t['amortized_ms'])
            both = b['status'] == 'passed' and t['status'] == 'passed'
            for m in ('setup', 'cold', 'repeated', 'amortized'):
                row[f'speedup_{m}'] = ratio(row[f'{m}_blackwell'], row[f'{m}_h200']) if both else None
                row[f'speedup_{m}_stock'] = (ratio(row[f'{m}_blackwell'], row[f'{m}_stock'])
                                             if b['status'] == 'passed' and s and s['status'] == 'passed' else None)
            if both and row['it_blackwell'] and row['it_h200']:
                row['speedup_per_iteration'] = ratio(row['repeated_blackwell']/row['it_blackwell'], row['repeated_h200']/row['it_h200'])
            out.append(row)
            break
    return out


def family_best(selected, tuned):
    """Fastest passed member per family on each machine, for systems both machines measured."""
    bw = defaultdict(dict)
    for r in selected:
        if r['status'] != 'passed' or not r['triangles']:
            continue
        key = (r['geometry'], lu.CASE_NAMES.get(r['case'], r['case']), int(r['triangles']), int(r['p']))
        cell = bw[key].setdefault((r['group'], r['metric']), (float('inf'), None))
        if float(r['time_ms']) < cell[0]:
            bw[key][(r['group'], r['metric'])] = (float(r['time_ms']), r['method'] or r['family'])
    hw = defaultdict(dict)
    for t in tuned:
        if t['status'] != 'passed':
            continue
        key = (t['geometry'], t['case'], t['triangles'], t['p'])
        group = FAMILY_GROUP[t['family']]
        for metric, field in (('hot', 'repeated_ms'), ('fresh', 'cold_ms')):
            value = t[field]
            if value and value < hw[key].get((group, metric), (float('inf'),))[0]:
                hw[key][(group, metric)] = (value, t['aliases'].split('|')[0])
    rows = []
    for key in sorted(set(bw) & set(hw), key=lambda k: (k[1], k[3], k[2])):
        row = dict(geometry=key[0], case=key[1], triangles=key[2], p=key[3])
        for metric in ('hot', 'fresh'):
            for group in ('asm', 'bj', 'amg', 'native', 'lu'):
                row[f'blackwell_{group}_{metric}'] = bw[key].get((group, metric), (None,))[0]
                if group != 'lu':
                    row[f'h200_{group}_{metric}'] = hw[key].get((group, metric), (None,))[0]
            bw_iter = {g: row[f'blackwell_{g}_{metric}'] for g in ('asm', 'bj', 'amg', 'native') if row[f'blackwell_{g}_{metric}']}
            h_iter = {g: row[f'h200_{g}_{metric}'] for g in ('asm', 'bj', 'amg', 'native') if row[f'h200_{g}_{metric}']}
            row[f'blackwell_winner_{metric}'] = min(bw_iter, key=bw_iter.get) if bw_iter else None
            row[f'h200_winner_{metric}'] = min(h_iter, key=h_iter.get) if h_iter else None
            row[f'blackwell_best_iterative_{metric}'] = min(bw_iter.values()) if bw_iter else None
            row[f'h200_best_iterative_{metric}'] = min(h_iter.values()) if h_iter else None
            lu_ms = row[f'blackwell_lu_{metric}']
            row[f'blackwell_iterative_over_lu_{metric}'] = ratio(row[f'blackwell_best_iterative_{metric}'], lu_ms)
            row[f'h200_iterative_over_lu_{metric}'] = ratio(row[f'h200_best_iterative_{metric}'], lu_ms)
        rows.append(row)
    return rows


def extension(tuned):
    """H200-only levels: per level, degree and family, passed counts and medians over classes."""
    groups = defaultdict(list)
    for t in tuned:
        groups[(t['level'], t['p'], t['family'])].append(t)
    rows = []
    for (level, p, family), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        ok = [t for t in group if t['status'] == 'passed']
        med = lambda k: statistics.median([t[k] for t in ok if t[k]]) if any(t[k] for t in ok) else None  # noqa: E731
        rows.append(dict(level=level, p=p, family=family, passed=len(ok), attempted=len(group),
                         dofs=statistics.median([t['trace_dofs'] for t in group if t['trace_dofs']]) if group else None,
                         iterations=med('iterations'), setup_ms=med('setup_ms'), cold_ms=med('cold_ms'),
                         repeated_ms=med('repeated_ms'), warm_1e2_ms=med('warm_0.01_ms'), amortized_ms=med('amortized_ms'),
                         statuses=json.dumps({s: sum(1 for t in group if t['status'] == s) for s in sorted({t['status'] for t in group})})))
    return rows


CAPACITY = ('memory_excluded', 'preparation_failed', 'out_of_memory', 'timeout', 'error')


def fp64_floor_model(path=None):
    """Per (geometry, case), a function DOFs -> FP64 residual floor fitted to the measured floors.

    From ``fp64_floor.csv`` (scripts/advection_diffusion_reaction/diagnostics/fp64_residual_floor.py):
    ``eps*|| |A||x| ||/||b||`` on small exported systems, all degrees together, fitted as
    ``c*dofs**alpha`` in log-log (alpha is about 1 for the classes with zero boundary data and
    O(1) diffusion, 0.3--0.7 for the others).
    """
    points = defaultdict(list)
    for r in csv.DictReader((path or OUT/'fp64_floor.csv').open()):
        points[(r['geometry'], r['case'])].append((math.log(float(r['trace_dofs'])), math.log(float(r['fp64_floor']))))
    model = {}
    for key, xy in points.items():
        mx, my = statistics.fmean(x for x, _ in xy), statistics.fmean(y for _, y in xy)
        sxx = sum((x-mx)**2 for x, _ in xy)
        alpha = sum((x-mx)*(y-my) for x, y in xy)/sxx if sxx else 1.
        model[key] = lambda dofs, a=alpha, mx=mx, my=my: math.exp(my+a*(math.log(dofs)-mx))
    return model


def outcome_kinds(tuned, floor_model=None, floor_tolerance=1.25, residual_limit=1e-9, floor_factor=10.):
    """Each job's outcome: passed, stopped by the FP64 residual floor, not passed, excluded.

    ``fp64_floor``: the first solve's L2 error is within ``floor_tolerance`` of the smallest L2 error
    any family reached on the same system with a true relative residual below ``residual_limit``
    (the iterate is as good as the discretization allows; where that error is itself round-off,
    1e-12 to 1e-11 at p >= 6, it varies by tens of percent between runs) or at most 1.05 times the
    L2 error of a solve that passed on the same system, and its true relative residual is within
    ``floor_factor`` of the system's FP64 round-off floor (:func:`fp64_floor_model`), so only the
    fixed residual targets (rtol 1e-10, internal 1e-11) were not met. Other iterates that did not
    pass are ``not_passed``. Returns ``[(job, kind)]``.
    """
    floor_model = fp64_floor_model() if floor_model is None else floor_model
    by_system = defaultdict(list)
    for t in tuned:
        by_system[t['system_id']].append(t)
    kinds = []
    for rows in by_system.values():
        good = [_f(t['first_l2_error']) for t in rows
                if _f(t['first_l2_error']) and _f(t['first_true_relative_residual']) is not None
                and _f(t['first_true_relative_residual']) < residual_limit]
        best = min(good) if good else None
        # An accepted solve's accuracy also qualifies: matching it is as good as a pass.
        accepted = [_f(t['l2_error']) or _f(t['first_l2_error']) for t in rows if t['status'] == 'passed']
        accurate = max(([floor_tolerance*best] if best else []) + [1.05*e for e in accepted if e], default=None)
        for t in rows:
            if t['status'] == 'passed':
                kind = 'passed'
            elif t['status'] in CAPACITY:
                kind = 'excluded'
            else:
                l2, rr = _f(t['first_l2_error']), _f(t['first_true_relative_residual'])
                fit = floor_model.get((t['geometry'], t['case']))
                floor = fit(float(t['trace_dofs'])) if fit and t['trace_dofs'] else 0.
                kind = ('fp64_floor' if accurate and l2 and rr is not None and rr < residual_limit
                        and l2 <= accurate and rr <= floor_factor*floor else 'not_passed')
            kinds.append((t, kind))
    return kinds


OUTCOME_KINDS = ('passed', 'fp64_floor', 'not_passed', 'excluded')


def outcomes(tuned):
    """Per level and family: counts of the :func:`outcome_kinds` outcomes."""
    counts = defaultdict(lambda: defaultdict(int))
    for t, kind in outcome_kinds(tuned):
        counts[(t['level'], t['family'])][kind] += 1
    order = ['L1', 'L2', 'L3', 'L4', 'L4b', 'L5', 'L6', 'L7', 'L8', 'L9', 'Lmax']
    return [dict(level=level, family=family, **{k: counts[(level, family)].get(k, 0) for k in OUTCOME_KINDS})
            for level in order for family in FAMILY_GROUP if (level, family) in counts]


def outcome_table(rows):
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'level & family & passed & FP64 floor$^*$ & not passed & excluded\\', r'\midrule']
    for r in rows:
        lines.append(f"{r['level']} & {GROUP_LABEL[FAMILY_GROUP[r['family']]]} & {r['passed']} & {r['fp64_floor']} & "
                     f"{r['not_passed']} & {r['excluded']}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def node_lu():
    """Fastest passed PyPardiso fresh and reused solve per system on a genoa CPU node.

    From the PARDISO campaigns on the device campaigns' exported systems (genoa, 2x EPYC 9655):
    fresh is CSR conversion + analysis + factorization + first solve, reused a solve with the
    existing factors. Each metric keeps its own fastest thread count. Keyed by the 16-hex system id.
    ``residual_floor`` runs (the factorization completed; the solve ended just above the 1e-10
    target at the FP64 round-off floor, L7) keep their timings, with ``status`` and ``residual``.
    """
    best = {}
    rows = [r for d in PARDISO if (d/'timings.csv').exists() for r in csv.DictReader(open(d/'timings.csv'))]
    for r in rows:
        if r['status'] not in ('passed', 'residual_floor'):
            continue
        # factor_nnz 0 is MKL's int32 count overflowing (a factor above 2^31 entries), not an empty factor.
        cell = best.setdefault(r['key'].split('_')[-1], dict(fresh=None, fresh_threads=None, reused=None, reused_threads=None,
                                              factor_nnz=_f(r['factor_nnz']) or None, peak_gib=_f(r['peak_hwm_gib']),
                                              status='passed', residual=None))
        if r['status'] == 'residual_floor':
            cell['status'] = 'residual_floor'
            cell['residual'] = max(cell['residual'] or 0, _f(r.get('worst_true_relative_residual')) or 0) or None
        for metric, field in (('fresh', 'fresh_median_ms'), ('reused', 'reused_mean_ms')):
            value = _f(r[field])
            if value and (cell[metric] is None or value < cell[metric]):
                cell[metric], cell[f'{metric}_threads'] = value, int(r['threads'])
    return best


def lu_rows(tuned, lu_best):
    """Per H200 system: best iterative fresh/repeated, its outcome, PyPardiso LU on the genoa node, ratios."""
    systems = defaultdict(list)
    for t, kind in outcome_kinds(tuned):
        systems[t['system_id']].append((t, kind))
    rows = []
    for jobs in systems.values():
        t0 = jobs[0][0]
        passed = [t for t, kind in jobs if kind == 'passed']
        kinds = {kind for _, kind in jobs}
        row = dict(level=t0['level'], geometry=t0['geometry'], case=t0['case'], p=t0['p'], triangles=t0['triangles'],
                   dofs=t0['trace_dofs'], system_id=t0['system_id'],
                   iterative_outcome=next(k for k in OUTCOME_KINDS if k in kinds))
        for metric, field in (('fresh', 'cold_ms'), ('repeated', 'repeated_ms')):
            pts = [(t[field], t['family']) for t in passed if t[field]]
            value, family = min(pts) if pts else (None, None)
            row[f'iterative_{metric}_ms'], row[f'iterative_{metric}_family'] = value, family
        lu = lu_best.get(t0['system_id'].split('_')[-1])
        row['lu_status'] = lu['status'] if lu else 'not_measured'
        row['lu_residual'] = lu and lu['residual']
        row.update(lu_fresh_ms=lu and lu['fresh'], lu_fresh_threads=lu and lu['fresh_threads'],
                   lu_reused_ms=lu and lu['reused'], lu_reused_threads=lu and lu['reused_threads'],
                   lu_factor_nnz=lu and lu['factor_nnz'], lu_peak_gib=lu and lu['peak_gib'])
        row['iterative_over_lu_fresh'] = ratio(row['iterative_fresh_ms'], row['lu_fresh_ms'])
        row['iterative_over_lu_repeated'] = ratio(row['iterative_repeated_ms'], row['lu_reused_ms'])
        rows.append(row)
    order = ['L1', 'L2', 'L3', 'L4', 'L4b', 'L5', 'L6', 'L7']
    return sorted(rows, key=lambda r: (r['case'], r['p'], r['dofs'] or 0, order.index(r['level']) if r['level'] in order else 99))


LADDER_CLASSES = {('square', 'trigonometric'), ('square', 'advection_dominated'), ('square', 'variable_velocity'),
                  ('square', 'anisotropic'), ('annulus', 'cellular7_weak'), ('annulus', 'stress_main_orthogonal')}


def stock_oscillatory(stock_all, lu_best):
    """Classes outside the tuned ladder: best passed stock-kernel H200 solve per system against genoa LU.

    Stock kernels (unified campaign 21958, the study's own configurations) on the systems the LU
    campaign measured; one row per system with both.
    """
    by_system = defaultdict(list)
    for t in stock_all:
        if (t['geometry'], t['case']) not in LADDER_CLASSES and t['status'] == 'passed':
            by_system[t['system_id']].append(t)
    rows = []
    for sid, jobs in by_system.items():
        cell = lu_best.get(sid.split('_')[-1])
        if not cell:
            continue
        t0 = jobs[0]
        row = dict(case=t0['case'], geometry=t0['geometry'], level=t0['level'], p=t0['p'], dofs=t0['trace_dofs'], system_id=sid,
                   lu_fresh_ms=cell['fresh'], lu_reused_ms=cell['reused'])
        for metric, field in (('fresh', 'cold_ms'), ('repeated', 'repeated_ms')):
            value, family = min(((t[field], t['family']) for t in jobs if t[field]), default=(None, None))
            row[f'stock_{metric}_ms'], row[f'stock_{metric}_family'] = value, family
        row['stock_over_lu_fresh'] = ratio(row['stock_fresh_ms'], row['lu_fresh_ms'])
        row['stock_over_lu_repeated'] = ratio(row['stock_repeated_ms'], row['lu_reused_ms'])
        rows.append(row)
    return sorted(rows, key=lambda r: (r['case'], r['geometry'], r['p'] or 0, r['dofs'] or 0))


def stock_oscillatory_table(rows):
    """Largest system per class and geometry (the oscillatory suite's counterpart of the largest table)."""
    largest = {}
    for r in rows:
        key = (r['case'], r['geometry'])
        if key not in largest or (r['dofs'] or 0) > (largest[key]['dofs'] or 0):
            largest[key] = r
    lines = [r'\begin{tabular}{llrlrrrrrr}', r'\toprule',
             r'class & geometry & $p$ & level & DOFs & fresh & P-LU fresh & repeated & P-LU reused & families\\', r'\midrule']
    for (case, geometry), r in sorted(largest.items()):
        families = '/'.join(GROUP_LABEL[FAMILY_GROUP[f]] for f in (r['stock_fresh_family'], r['stock_repeated_family']) if f)
        lines.append(f"{case.replace('_', r'\_')} & {geometry} & {r['p']} & {r['level']} & {num(r['dofs'], 0)} & "
                     f"{num(r['stock_fresh_ms'])} & {num(r['lu_fresh_ms'])} & {num(r['stock_repeated_ms'])} & "
                     f"{num(r['lu_reused_ms'])} & {families}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def end_to_end(tuned, lu_best, cpu_dir=CPU_PIPELINE):
    """Assembly plus one fresh solve on the systems both pipelines ran (None without the CPU records).

    genoa: the assembly of hybridge's host pipeline (``cpu_adr_scaling``, Numba; warm run at the
    fastest thread count) plus the LU campaign's fresh PyPardiso solve (analysis, factorization and
    first solve, each at its fastest thread count). The pipeline's own warm solve is not used:
    PyPardiso skips the factorization of a matrix identical to the last one it factored, so that
    solve reused the cold run's factors. H200: the system job's device assembly (one cold assembly
    in the solver process) plus the fastest passed fresh iterative solve.
    """
    paths = [cpu_dir/tier/'timings.csv' for tier in ('scaling', 'coverage')]
    if not all(path.exists() for path in paths):
        return None
    cpu = defaultdict(list)
    for path in paths:
        for r in csv.DictReader(open(path)):
            if r['status'] == 'passed' and r['tier'] in ('scaling', 'coverage') and _f(r['warm_assembly_s']) is not None:
                cpu[r['system']].append(r)
    passed = defaultdict(list)
    for t in tuned:
        if t['status'] == 'passed' and t['cold_ms']:
            passed[t['system_id']].append(t)
    rows = []
    for s in csv.DictReader(open(RECORDS/'systems.csv')):
        key = s['system_id'].split('_')[-1]
        runs, lu_cell, assembly = cpu.get(key), lu_best.get(key), _f(s.get('device_assembly_ms'))
        if not s.get('cache') or not runs or not lu_cell or not lu_cell['fresh'] or not passed.get(s['system_id']) \
                or assembly is None:
            continue
        host = min(runs, key=lambda r: _f(r['warm_assembly_s']))
        device = min(passed[s['system_id']], key=lambda t: t['cold_ms'])
        cpu_total = _f(host['warm_assembly_s'])*1e3+lu_cell['fresh']
        rows.append(dict(level=s['level'], case=s['case'], p=int(s['p']), dofs=_f(s['trace_dofs']), system_id=s['system_id'],
                         cpu_assembly_threads=int(host['threads']), cpu_assembly_ms=_f(host['warm_assembly_s'])*1e3,
                         lu_fresh_threads=lu_cell['fresh_threads'], lu_fresh_ms=lu_cell['fresh'], cpu_total_ms=cpu_total,
                         cpu_reconstruction_ms=(_f(host['warm_reconstruction_s']) or 0)*1e3,
                         h200_assembly_ms=assembly, h200_fresh_ms=device['cold_ms'], h200_family=device['family'],
                         h200_total_ms=assembly+device['cold_ms'], cpu_over_h200=cpu_total/(assembly+device['cold_ms'])))
    return sorted(rows, key=lambda r: (r['case'], r['p'], r['dofs']))


def end_to_end_table(rows):
    lines = [r'\begin{tabular}{lrlrrrrrrrr}', r'\toprule',
             r'case & $p$ & level & DOFs & \multicolumn{3}{c}{genoa} & \multicolumn{3}{c}{H200} & genoa/H200\\',
             r'\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
             r' & & & & assembly$_t$ & P-LU fresh$_t$ & sum & assembly & fresh & sum & \\', r'\midrule']
    for r in rows:
        lines.append(f"{CASE_TEX.get(r['case'], r['case'])} & {r['p']} & {r['level']} & {num(r['dofs'], 0)} & "
                     f"{num(r['cpu_assembly_ms'])}$_{{{r['cpu_assembly_threads']}}}$ & "
                     f"{num(r['lu_fresh_ms'])}$_{{{r['lu_fresh_threads']}}}$ & {num(r['cpu_total_ms'])} & "
                     f"{num(r['h200_assembly_ms'])} & {num(r['h200_fresh_ms'])} & {num(r['h200_total_ms'])} & "
                     f"{r['cpu_over_h200']:.2f}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def lu_followup(rows):
    """H200 systems without an LU measurement, with the exported host system the LU campaign reads."""
    exported = {s['system_id']: s for s in csv.DictReader(open(RECORDS/'systems.csv')) if s.get('cache')}
    return [dict(level=r['level'], geometry=r['geometry'], case=r['case'], p=r['p'], triangles=r['triangles'],
                 trace_dofs=r['dofs'], system_id=r['system_id'], key=r['system_id'].split('_')[-1],
                 cache=exported.get(r['system_id'], {}).get('cache', ''),
                 operator_sha256=exported.get(r['system_id'], {}).get('operator_sha256', ''))
            for r in rows if r['lu_status'] == 'not_measured']


def lu_crossover(rows):
    """Per case and degree: the smallest measured DOFs from which iterative stays faster than LU."""
    groups = defaultdict(list)
    for r in rows:
        if r['lu_status'] != 'not_measured':
            groups[(r['case'], r['p'])].append(r)
    out = []
    for (case, p), group in sorted(groups.items()):
        group.sort(key=lambda r: r['dofs'])
        row = dict(case=case, p=p, systems=len(group), dofs_min=group[0]['dofs'], dofs_max=group[-1]['dofs'])
        for metric in ('fresh', 'repeated'):
            ratios = [(r['dofs'], r[f'iterative_over_lu_{metric}'], r['iterative_outcome']) for r in group]
            measured = [x for x in ratios if x[1] is not None]
            # Smallest size such that iterative wins there and at every larger measured size.
            start = None
            for i, (dofs, value, _) in enumerate(measured):
                if all(v < 1 for _, v, _ in measured[i:]):
                    start = dofs
                    break
            row[f'{metric}_iterative_from_dofs'] = start
            row[f'{metric}_ratio_largest'] = measured[-1][1] if measured else None
            row[f'{metric}_no_iterative_pass'] = sum(1 for x in ratios if x[1] is None)
        out.append(row)
    return out


def lu_table(rows, levels=None):
    """Measured systems (``levels`` restricts them); a dagger marks an LU solve at the residual floor."""
    lines = [r'\begin{tabular}{llrrrrlrrrr}', r'\toprule',
             r'case & geom. & $p$ & level & DOFs & iter.\ fresh & iter.\ rep.\ (family) & LU fresh ($t$) & LU reused ($t$) & '
             r'fresh $\frac{\mathrm{iter}}{\mathrm{LU}}$ & rep.\ $\frac{\mathrm{iter}}{\mathrm{LU}}$\\', r'\midrule']
    for r in rows:
        if r['lu_status'] == 'not_measured' or (levels and r['level'] not in levels):
            continue
        floor = r'$^\dagger$' if r['lu_status'] == 'residual_floor' else ''
        family = GROUP_LABEL[FAMILY_GROUP[r['iterative_repeated_family']]] if r['iterative_repeated_family'] else (
            r'\emph{FP64}$^*$' if r['iterative_outcome'] == 'fp64_floor' else r'\emph{' + r['iterative_outcome'].replace('_', ' ') + '}')
        lines.append(f"{CASE_TEX.get(r['case'], r['case'].replace('_', ' '))} & {r['geometry'][:3]} & {r['p']} & {r['level']} & "
                     f"{num(r['dofs'], 0)} & {num(r['iterative_fresh_ms'])} & {num(r['iterative_repeated_ms'])} ({family}) & "
                     f"{num(r['lu_fresh_ms'])}{floor} ({r['lu_fresh_threads']}) & {num(r['lu_reused_ms'])} ({r['lu_reused_threads']}) & "
                     f"{num(r['iterative_over_lu_fresh'], 2)} & {num(r['iterative_over_lu_repeated'], 2)}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def lu_crossover_table(rows):
    lines = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'case & $p$ & systems & DOFs range & fresh: iter.\ from & fresh ratio, largest & rep.: iter.\ from & rep.\ ratio, largest\\',
             r'\midrule']
    for r in rows:
        lines.append(f"{CASE_TEX.get(r['case'], r['case'].replace('_', ' '))} & {r['p']} & {r['systems']} & "
                     f"{num(r['dofs_min'], 0)}--{num(r['dofs_max'], 0)} & {num(r['fresh_iterative_from_dofs'], 0)} & "
                     f"{num(r['fresh_ratio_largest'], 2)} & {num(r['repeated_iterative_from_dofs'], 0)} & "
                     f"{num(r['repeated_ratio_largest'], 2)}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def num(v, nd=1):
    if v is None:
        return '--'
    if abs(v) >= 1000:
        return f'{v:,.0f}'.replace(',', r'\,')
    return f'{v:.{nd}f}'


CASE_TEX = {'trigonometric': 'trig.', 'advection_dominated': 'transport', 'anisotropic': 'aniso.',
            'variable_velocity': 'var.\\ vel.', 'cellular7_weak': 'cell.\\ weak', 'stress_main_orthogonal': 'stress orth.'}


def candidate_label(alias):
    """A candidate's display name: pMG--AMG under its canonical name and policy, others verbatim."""
    if alias.startswith('native_hp_'):
        return f"{PMG_TEX} {alias.removeprefix('native_hp_')}"
    return f"\\texttt{{{alias.replace('_', chr(92)+'_')}}}"


def matched_table(rows):
    lines = [r'\begin{tabular}{llrrlrrrrrrr}', r'\toprule',
             r'case & geom. & $p$ & DOFs & candidate & it.\ BW/H200 & setup BW & setup H200 & rep.\ BW & rep.\ H200 & rep.\ $\times$ & amort.\ $\times$\\',
             r'\midrule']
    for r in sorted(rows, key=lambda r: (r['group'], r['case'], r['p'], r['dofs'])):
        if r['speedup_repeated'] is None:
            continue
        lines.append(f"{CASE_TEX.get(r['case'], r['case'])} & {r['geometry'][:3]} & {r['p']} & {num(r['dofs'], 0)} & "
                     f"{candidate_label(r['alias'])} & {num(r['it_blackwell'], 0)}/{num(r['it_h200'], 0)} & "
                     f"{num(r['setup_blackwell'])} & {num(r['setup_h200'])} & {num(r['repeated_blackwell'])} & {num(r['repeated_h200'])} & "
                     f"{num(r['speedup_repeated'], 2)} & {num(r['speedup_amortized'], 2)}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def summary_table(rows):
    """Per family: median and range of the matched speedups, stock and tuned."""
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'family & matched & rep.\ stock & rep.\ tuned & amort.\ stock & amort.\ tuned & per-iteration tuned\\', r'\midrule']
    stats = {}
    for group in ('asm', 'bj', 'amg', 'native'):
        g = [r for r in rows if r['group'] == group and r['speedup_repeated']]
        if not g:
            continue
        def band(key):
            values = sorted(r[key] for r in g if r.get(key))
            if not values:
                return '--', None
            return f"{statistics.median(values):.2f} ({values[0]:.2f}--{values[-1]:.2f})", statistics.median(values)
        cells = {k: band(k) for k in ('speedup_repeated_stock', 'speedup_repeated', 'speedup_amortized_stock',
                                      'speedup_amortized', 'speedup_per_iteration')}
        stats[group] = {k: v[1] for k, v in cells.items()} | dict(matched=len(g))
        lines.append(f"{GROUP_LABEL[group]} & {len(g)} & " + ' & '.join(cells[k][0] for k in (
            'speedup_repeated_stock', 'speedup_repeated', 'speedup_amortized_stock', 'speedup_amortized', 'speedup_per_iteration')) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n', stats


def family_table(rows):
    lines = [r'\begin{tabular}{llrrllrrr}', r'\toprule',
             r'case & geom. & $p$ & triangles & winner BW & winner H200 & best BW & best H200 & P-LU (BW host)\\', r'\midrule']
    for r in rows:
        dofs = None
        lines.append(f"{CASE_TEX.get(r['case'], r['case'])} & {r['geometry'][:3]} & {r['p']} & {num(r['triangles'], 0)} & "
                     f"{GROUP_LABEL.get(r['blackwell_winner_hot'], '--')} & {GROUP_LABEL.get(r['h200_winner_hot'], '--')} & "
                     f"{num(r['blackwell_best_iterative_hot'])} & {num(r['h200_best_iterative_hot'])} & {num(r['blackwell_lu_hot'])}\\\\")
        del dofs
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def extension_table(rows):
    lines = [r'\begin{tabular}{lrlrrrrrrr}', r'\toprule',
             r'level & $p$ & family & passed & it. & setup & cold & repeated & warm $10^{-2}$ & amortized\\', r'\midrule']
    for r in rows:
        if r['level'] not in ('L5', 'L6', 'L7'):
            continue
        lines.append(f"{r['level']} & {r['p']} & {GROUP_LABEL[FAMILY_GROUP[r['family']]]} & {r['passed']}/{r['attempted']} & "
                     f"{num(r['iterations'], 0)} & {num(r['setup_ms'])} & {num(r['cold_ms'])} & {num(r['repeated_ms'])} & "
                     f"{num(r['warm_1e2_ms'])} & {num(r['amortized_ms'])}\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def figures(matched, extension_rows, lu_best=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with publication_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
        for ax, key, title in zip(axes, ('speedup_repeated_stock', 'speedup_repeated', 'speedup_amortized'),
                                  ('repeated, stock kernels', 'repeated, tuned kernels', 'amortized, tuned kernels')):
            for group in ('asm', 'bj', 'amg', 'native'):
                pts = [(r['dofs'], r[key]) for r in matched if r['group'] == group and r.get(key)]
                if pts:
                    ax.scatter(*zip(*pts), s=14, color=COLORS[group], label=GROUP_LABEL[group], alpha=.85)
            ax.axhline(1, color='#98a2b3', lw=.7, ls='--')
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(title); ax.set_xlabel('trace DOFs'); ax.grid(True, which='major')
        axes[0].set_ylabel('Blackwell time / H200 time')
        axes[0].legend(frameon=False, loc='upper left')
        fig.tight_layout()
        save_publication_figure(fig, FIGURES/'speedup_matched'); plt.close(fig)

        # One class across the whole ladder, so every point is the same problem at a finer mesh;
        # missing points are runs that did not pass (recorded in extension.csv).
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
        floor = {id(t) for t, kind in outcome_kinds(extension_rows) if kind == 'fp64_floor'}
        for ax, p in zip(axes, (2, 6, 9)):
            for row, (family, group) in enumerate(FAMILY_GROUP.items()):
                pts = sorted((r['trace_dofs'], r['amortized_ms']) for r in extension_rows
                             if r['case'] == LADDER_CASE and r['p'] == p and r['family'] == family
                             and r['status'] == 'passed' and r['amortized_ms'] and r['trace_dofs'])
                if pts:
                    ax.plot(*zip(*pts), marker='o', color=COLORS[group], label=GROUP_LABEL[group])
                # NC*: stopped at the FP64 residual floor; hollow, at the time its first solve took to the
                # lowest residual, amortized like the others: (setup + two such solves) / 2.
                stars = sorted((float(r['trace_dofs']), (_f(r['first_floor_fresh_ms'])+_f(r['first_floor_solve_ms']))/2)
                               for r in extension_rows if id(r) in floor and r['case'] == LADDER_CASE
                               and r['p'] == p and r['family'] == family and _f(r['first_floor_fresh_ms']))
                if stars:
                    ax.plot(*zip(*(pts[-1:]+stars)), color=COLORS[group], linestyle=':', linewidth=1.1)
                    ax.plot(*zip(*stars), marker='o', markerfacecolor='white', color=COLORS[group], linestyle='none')
            # genoa LU on the same systems, amortized the same way: (fresh + reused) / 2, each
            # metric at its own fastest thread count.
            systems = {r['system_id']: r['trace_dofs'] for r in extension_rows
                       if r['case'] == LADDER_CASE and r['p'] == p and r['trace_dofs']}
            pts = sorted((dofs, _amortized(cell['fresh'], cell['reused'])) for sid, dofs in systems.items()
                         if (cell := (lu_best or {}).get(sid.split('_')[-1])) and cell['fresh'] and cell['reused'])
            if pts:
                ax.plot(*zip(*pts), marker='s', ls='--', color=COLORS['lu'], label='P-LU (genoa)')
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(f'$p={p}$'); ax.set_xlabel('trace DOFs'); ax.grid(True, which='major')
        axes[0].set_ylabel(f'amortized solve (ms), {LADDER_CASE}')
        handles, _ = axes[0].get_legend_handles_labels()
        if floor:
            handles.append(Line2D([], [], color='#555555', marker='o', markerfacecolor='white', linestyle=':',
                                  label='NC*: time to FP64 residual floor'))
        fig.tight_layout(rect=(0, .12, 1, 1))
        fig.legend(handles=handles, frameon=False, loc='lower center', ncol=len(handles), bbox_to_anchor=(.5, 0))
        save_publication_figure(fig, FIGURES/'h200_ladder_amortized'); plt.close(fig)


# --- Synthesis-style figures: H200 iterative families against PyPardiso LU on a genoa node -----
# Drawn with the Blackwell synthesis' own grid and envelope functions
# (make_adr_synthesis_section.comparison_overview / focused_figure), fed with H200 records.

LEVELS = ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7')
SYNTH_SUITE = 'h200'
SYNTH_HEAT = [(SYNTH_SUITE, 'trigonometric', 'square', 'Trigonometric'),
              (SYNTH_SUITE, 'advection_dominated', 'square', 'Smooth transport'),
              (SYNTH_SUITE, 'variable_velocity', 'square', 'Variable velocity'),
              (SYNTH_SUITE, 'anisotropic', 'square', 'Smooth anisotropic'),
              (SYNTH_SUITE, 'cellular7_weak', 'annulus', 'Oscillatory low diffusion'),
              (SYNTH_SUITE, 'stress_main_orthogonal', 'annulus', 'Closed-loop stress')]
SYNTH_CASE_LABELS = ['Trigonometric\nSquare', 'Smooth transport\nSquare', 'Variable velocity\nSquare',
                     'Smooth anisotropy\nSquare', 'Oscillatory low diffusion\nAnnulus', 'Closed-loop stress\nAnnulus']
SYNTH_FIGURES = FIGURES.parent/'h200_synthesis'
# Names the synthesis predicates recognize: asm*, BJ+PP*, amgx_*_amg*, native_hp_*.
SYNTH_METHOD = {'asm_pp': 'asm_pp', 'bj_pp': 'BJ+PP--GMRES', 'native_hp': 'native_hp_standard'}


LEVEL_GEOMETRIES = (('square', 'Square (trigonometric, transport, variable velocity, anisotropic)'),
                    ('cellular7_weak', 'Annulus, cellular (five-lobed)'),
                    ('stress_main_orthogonal', 'Annulus, closed-loop stress'))


def levels_table(tuned):
    """Triangles, unknown edges (trace DOFs / (p+1)) and trace DOFs per geometry, level and degree."""
    sizes = {}
    for t in tuned:
        key = 'square' if t['geometry'] == 'square' else t['case']
        if t['triangles'] and t['trace_dofs']:
            sizes[(key, t['level'], t['p'])] = (t['triangles'], t['trace_dofs'])
    lines = [r'\begin{tabular}{l' + 'rrr'*3 + '}', r'\toprule',
             r' & \multicolumn{3}{c}{$p=2$} & \multicolumn{3}{c}{$p=6$} & \multicolumn{3}{c}{$p=9$}\\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
             r'level & triangles & edges & DOFs & triangles & edges & DOFs & triangles & edges & DOFs\\']
    for key, title in LEVEL_GEOMETRIES:
        lines += [r'\midrule', rf'\multicolumn{{10}}{{l}}{{\emph{{{title}}}}}\\']
        for level in (*LEVELS[:4], 'L4b', *LEVELS[4:]):
            cells = []
            for p in (2, 6, 9):
                size = sizes.get((key, level, p))
                cells += [num(size[0], 0), num(size[1]/(p+1), 0), num(size[1], 0)] if size else ['--']*3
            if any(c != '--' for c in cells):
                lines.append(f"{level} & " + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def lmax_table(tuned):
    """Lmax rows: every class and degree with its mesh (triangles), trace DOFs and the four families."""
    records = synthesis_records(tuned, {}, extra_levels=('Lmax',))
    groups = defaultdict(list)
    for r in records:
        if r['level'] == 'Lmax':
            groups[(r['case'], r['p'])].append(r)
    lines = [r'\begin{tabular}{lrrrlllll}', r'\toprule',
             r'case & $p$ & triangles & DOFs & metric & ASM+PP & BJ+PP & AMGX & $p$MG--AMG \\', r'\midrule']
    for case in CASE_NAME:
        for p in sorted({p for c, p in groups if c == case}):
            rows = groups[(case, p)]
            if any(r['status'] == 'running' for r in rows):
                continue
            for metric, label in (('fresh', 'fresh'), ('hot', 'reused')):
                cells = ' & '.join(_cell(rows, f, metric) for f in ('asm', 'bj', 'amg', 'native'))
                head = (f"{CASE_NAME[case]} & {p} & {num(rows[0]['triangles'], 0)} & {num(rows[0]['dofs'], 0)}"
                        if metric == 'fresh' else ' & & & ')
                lines.append(f"{head} & {label} & {cells}\\\\")
            lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def synthesis_records(tuned, lu_best, extra_levels=('L4b',)):
    """H200 jobs and genoa LU rows in the record layout of the Blackwell synthesis."""
    rows, systems_seen = [], {}
    # A job stopped by the FP64 residual floor: NC* in the grids, its time to that floor in the graphs.
    floor = {id(t) for t, kind in outcome_kinds(tuned) if kind == 'fp64_floor'}
    for t in tuned:
        if t['level'] not in LEVELS and t['level'] not in extra_levels:
            continue
        # L4b (the Blackwell stress mesh) has no grid column; the level-sweep envelopes place it by size.
        n = LEVELS.index(t['level'])+1 if t['level'] in LEVELS else None
        method = SYNTH_METHOD.get(t['family']) or next(
            (a for a in t['aliases'].split('|') if a.startswith('amgx_') or a.startswith('amg_')), 'amgx_pbicgstab_amg')
        passed = t['status'] == 'passed'
        degree = _f(t.get('pp_degree'))
        base = dict(suite=SYNTH_SUITE, geometry=t['geometry'], case=t['case'], n=n, level=t['level'], p=t['p'],
                    dofs=t['trace_dofs'], triangles=t['triangles'], system_id=t['system_id'])
        rows.append(dict(base, method=method, status=t['status'], hot=t['repeated_ms'] if passed else None,
                         fresh=t['cold_ms'] if passed else None, setup=t['setup_ms'] if passed else None,
                         iterations=t['iterations'] if passed else None, fresh_iterations=t['iterations'] if passed else None,
                         pp_degree=int(degree) if degree else None, policy=t.get('policy') or None,
                         source_candidate=t['candidate'], fp64_floor=id(t) in floor,
                         # time of its first solve to its lowest residual (export_records.floor_point)
                         floor_fresh=_f(t.get('first_floor_fresh_ms')) if id(t) in floor else None,
                         floor_hot=_f(t.get('first_floor_solve_ms')) if id(t) in floor else None))
        systems_seen[t['system_id']] = base
    for sid, base in systems_seen.items():
        cell = lu_best.get(sid.split('_')[-1])
        if not cell:
            continue
        for metric, field in (('fresh', 'fresh'), ('hot', 'reused')):
            if cell[field] is None:
                continue
            rows.append(dict(base, method='pardiso_lu', status='passed', fresh=cell['fresh'] if metric == 'fresh' else None,
                             hot=cell['reused'] if metric == 'hot' else None, setup=None, iterations=None,
                             fresh_iterations=None, pp_degree=None, policy=None, source_candidate='pardiso_lu',
                             threads=cell[f'{field}_threads']))
    return rows


def synthesis_systems(p=None, level=None):
    """``systems`` for the H200 records: a level sweep at degree ``p`` (one column per level, empty
    levels kept as untested placeholders so the columns align) or a degree sweep at ``level``, a level
    name or a function of the degree (:func:`largest_level`)."""
    def select(records, suite, case, geometry='square', sweep=None):
        # Each H200 case lives on one geometry, so the case alone selects the systems.
        rows = [r for r in records if r['suite'] == suite and r['case'] == case]
        groups = defaultdict(list)
        if sweep == 'h':
            for r in rows:
                if r['p'] == p:
                    groups[r['n']].append(r)
            first = next(iter(rows), None)
            out = []
            for n in range(1, len(LEVELS)+1):
                if groups.get(n):
                    out.append(groups[n])
                else:  # keep the column: every family is reported as not tested
                    out.append([dict(suite=suite, case=case, geometry=geometry, n=n, p=p, level=LEVELS[n-1],
                                     dofs=None, triangles=None, method='not_tested', status='not_tested',
                                     hot=None, fresh=None)] if first else [])
            return [g for g in out if g]
        for r in rows:
            if r['level'] == (level(r['p']) if callable(level) else level):
                groups[r['p']].append(r)
        return [groups[k] for k in sorted(groups)]
    return select


def synthesis_envelope_systems(p=None, level=None):
    """Like ``synthesis_systems`` but without placeholders (envelope plots need sizes); a level sweep
    also places the case's L4b and Lmax systems by their size."""
    inner = synthesis_systems(p, level)

    def select(records, suite, case, geometry='square', sweep=None):
        groups = [g for g in inner(records, suite, case, geometry, sweep) if g[0].get('dofs')]
        for extra_level in ('L4b', 'Lmax'):
            extra = [r for r in records if sweep == 'h' and r['suite'] == suite and r['case'] == case
                     and r['p'] == p and r['level'] == extra_level]
            if extra and extra[0].get('dofs'):
                groups.append(extra)
        return sorted(groups, key=lambda g: g[0]['dofs'])
    return select


def largest_level(p):
    """The largest mesh one H200 takes at degree ``p``: Lmax, except p=6, whose L7 is that limit."""
    return 'L7' if p == 6 else 'Lmax'


def synthesis_lu_curve(lu_best):
    """The genoa LU timings through the study's LU curve helper."""
    def lookup_fn(row, suite, metric, campaign=None):
        cell = lu_best.get(row['system_id'].split('_')[-1]) if row.get('system_id') else None
        field = 'reused' if metric in ('hot', 'reused') else 'fresh'
        if not cell or cell[field] is None:
            return None
        return dict(system_id=row['system_id'], threads=cell[f'{field}_threads'], hot=cell['reused'], fresh=cell['fresh'])

    def draw(ax, groups, metric, xmode):
        lu.curve(ax, [g[0] for g in groups], SYNTH_SUITE, metric, 'p' if xmode == 'p' else 'dofs', lookup_fn=lookup_fn)
    return draw


def synthesis_triangles(records, p):
    """Triangles per level for the square classes and the annular cellular class at degree ``p``."""
    by_level = defaultdict(dict)
    for r in records:
        if r['p'] == p and r.get('triangles') and r['case'] in ('trigonometric', 'cellular7_weak'):
            by_level[r['geometry']][r['level']] = r['triangles']
    return '; '.join(f"{g} " + ' / '.join(f"{int(by_level[g][lv]):,}" if lv in by_level[g] else '--' for lv in LEVELS)
                     for g in ('square', 'annulus'))


CASE_NAME = {'trigonometric': 'Trigonometric', 'advection_dominated': 'Smooth transport',
             'variable_velocity': 'Variable velocity', 'anisotropic': 'Smooth anisotropic',
             'cellular7_weak': 'Oscillatory low diffusion', 'stress_main_orthogonal': 'Closed-loop stress'}


def _cell(rows, family, metric):
    ok = [r for r in rows if r['method'] != 'pardiso_lu' and FAMILY_OF_METHOD(r) == family
          and r['status'] == 'passed' and r[metric] is not None]
    if not ok:
        attempted = [r for r in rows if r['method'] != 'pardiso_lu' and FAMILY_OF_METHOD(r) == family]
        return ('NC$^*$' if any(r.get('fp64_floor') for r in attempted) else 'NC') if attempted else '--'
    best = min(ok, key=lambda r: r[metric])
    return f"{num(best[metric])} ({best['iterations']:.0f})" if best.get('iterations') else num(best[metric])


def FAMILY_OF_METHOD(r):
    m = r['method']
    return ('bj' if m.startswith('BJ+PP') else 'asm' if m.startswith('asm') else 'native' if m.startswith('native_hp')
            else 'amg')


def largest_table(records):
    """Largest level per case and degree where some iterative family passed (the million_table analog)."""
    groups = defaultdict(list)
    for r in records:
        groups[(r['case'], r['p'], r['level'])].append(r)
    lines = [r'\begin{tabular}{lrlrlllll}', r'\toprule',
             r'case & $p$ & level & DOFs & metric & ASM+PP & BJ+PP & AMGX & $p$MG--AMG \\', r'\midrule']
    for case in CASE_NAME:
        for p in (2, 6, 9):
            levels = [lv for lv in LEVELS if any(r['status'] == 'passed' and r['method'] != 'pardiso_lu'
                                                 for r in groups.get((case, p, lv), []))]
            if not levels:
                continue
            rows = groups[(case, p, levels[-1])]
            for metric, label in (('fresh', 'fresh'), ('hot', 'reused')):
                cells = ' & '.join(_cell(rows, f, metric) for f in ('asm', 'bj', 'amg', 'native'))
                head = (f"{CASE_NAME[case]} & {p} & {levels[-1]} & {num(rows[0]['dofs'], 0)}" if metric == 'fresh'
                        else ' & & & ')
                lines.append(f"{head} & {label} & {cells}\\\\")
            lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def stress_table(tuned, lu_best):
    """Closed-loop stress (orthogonal variant): block preconditioners per level and degree, with genoa LU."""
    lines = [r'\begin{tabular}{lrrlrrrrrr}', r'\toprule',
             r'level & $p$ & DOFs & family & outcome & it. & setup & fresh & reused & warm $10^{-2}$\\', r'\midrule']
    order = {lv: i for i, lv in enumerate(('L4', 'L4b', 'L5', 'L6', 'L7'))}
    rows = sorted((t for t in tuned if t['case'] == 'stress_main_orthogonal'),
                  key=lambda t: (t['p'], order.get(t['level'], 9), t['family']))
    for t in rows:
        lines.append(f"{t['level']} & {t['p']} & {num(t['trace_dofs'], 0)} & {GROUP_LABEL[FAMILY_GROUP[t['family']]]} & "
                     f"{t['status'].replace('_', ' ')} & {num(t['iterations'], 0)} & {num(t['setup_ms'])} & {num(t['cold_ms'])} & "
                     f"{num(t['repeated_ms'])} & {num(t['warm_0.01_ms'])}\\\\")
        cell = lu_best.get(t['system_id'].split('_')[-1])
        if cell and t['family'] == 'bj_pp':
            lines.append(f"{t['level']} & {t['p']} & {num(t['trace_dofs'], 0)} & P-LU$_{{{cell['fresh_threads']}/{cell['reused_threads']}}}$ & "
                         f"passed & -- & -- & {num(cell['fresh'])} & {num(cell['reused'])} & --\\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)+'\n'


def synthesis_figures(records, lu_best):
    import make_adr_synthesis_section as synthesis
    cells = []
    with publication_style(font_size=8.2):
        for p in (2, 6, 9):
            # Seven level columns: one portrait page per metric (the cells carry both metrics).
            for metric, label in (('fresh', 'fresh'), ('hot', 'reused')):
                drawn = synthesis.comparison_overview(
                    records, 'h', heat=SYNTH_HEAT, case_labels=SYNTH_CASE_LABELS, column_labels=list(LEVELS),
                    heading=rf'H200 · mesh refinement at $p_{{\mathrm{{FE}}}}={p}$ · columns are refinement levels',
                    mesh='', stem=f'h200_mesh_p{p}_{label}', metrics=(metric,),
                    figsize=(7.3, 9.3), systems_fn=synthesis_systems(p=p), figures=SYNTH_FIGURES)
            cells += drawn
        # Degree sweep at L4: the ladder's p = 2, 6, 9 and the L4 degree fill (p = 3, 4, 5, 7, 8), one
        # portrait page per metric; the columns are the degrees present at L4.
        l4_degrees = sorted({r['p'] for r in records if r['level'] == 'L4' and r['method'] != 'pardiso_lu'})
        for metric, label in (('fresh', 'fresh'), ('hot', 'reused')):
            cells += synthesis.comparison_overview(
                records, 'p', heat=SYNTH_HEAT, case_labels=SYNTH_CASE_LABELS, column_labels=[str(p) for p in l4_degrees],
                heading=r'H200 · degree sweep at L4 · columns are finite-element degree $p_{\mathrm{FE}}$',
                mesh='', stem=f'h200_degree_L4_{label}', metrics=(metric,), figsize=(7.3, 9.3),
                systems_fn=synthesis_systems(level='L4'), figures=SYNTH_FIGURES)
        smooth = ('advection_dominated', 'variable_velocity', 'trigonometric')
        titles = ('Smooth transport', 'Variable velocity', 'Trigonometric')
        curve = synthesis_lu_curve(lu_best)
        level_labels = lambda groups: [g[0]['level'] for g in groups]  # noqa: E731
        synthesis.focused_figure(records, 'h', 'h200_h_crossover', cases=smooth, titles=titles, suite=SYNTH_SUITE,
                                 systems_fn=synthesis_envelope_systems(p=6), lu_curve=curve, level_labels=level_labels,
                                 figures=SYNTH_FIGURES, title_pad=26, hspace=.5, mark_local_nc=True)
        hard = ('anisotropic', 'cellular7_weak', 'stress_main_orthogonal')
        for p in (6, 9):
            synthesis.focused_figure(records, 'h', f'h200_hard_cases_p{p}', cases=hard,
                                     titles=('Smooth anisotropic', 'Oscillatory low diffusion (annulus)', 'Closed-loop stress (annulus)'),
                                     suite=SYNTH_SUITE, systems_fn=synthesis_envelope_systems(p=p), lu_curve=curve,
                                     level_labels=level_labels, figures=SYNTH_FIGURES, title_pad=26, hspace=.6, mark_local_nc=True,
                                     tick_rotation=45)
        synthesis.focused_figure(records, 'p', 'h200_p_crossover', cases=smooth, titles=titles, suite=SYNTH_SUITE,
                                 systems_fn=synthesis_envelope_systems(level='L4'), lu_curve=curve, level_labels=level_labels,
                                 figures=SYNTH_FIGURES, degree_label=r'$p_{\mathrm{FE}}$ (L4, 99,458 triangles)',
                                 title_pad=26, hspace=.5, mark_local_nc=True)
        # Degree sweep on the largest mesh one H200 takes per degree (34 to 14 million trace DOFs).
        for metric, label in (('fresh', 'fresh'), ('hot', 'reused')):
            cells += synthesis.comparison_overview(
                records, 'p', heat=SYNTH_HEAT, case_labels=SYNTH_CASE_LABELS,
                column_labels=[str(p) for p in range(2, 10)],
                heading=r'H200 · degree sweep on the largest mesh per degree · columns are $p_{\mathrm{FE}}$',
                mesh='', stem=f'h200_degree_lmax_{label}', metrics=(metric,), figsize=(7.3, 9.3),
                systems_fn=synthesis_systems(level=largest_level), figures=SYNTH_FIGURES)
        largest_label = r'$p_{\mathrm{FE}}$ (largest mesh per degree)'
        synthesis.focused_figure(records, 'p', 'h200_p_lmax', cases=smooth, titles=titles, suite=SYNTH_SUITE,
                                 systems_fn=synthesis_envelope_systems(level=largest_level), lu_curve=curve,
                                 level_labels=level_labels, figures=SYNTH_FIGURES, degree_label=largest_label,
                                 title_pad=26, hspace=.5, mark_local_nc=True)
        synthesis.focused_figure(records, 'p', 'h200_p_lmax_hard', cases=hard,
                                 titles=('Smooth anisotropic', 'Oscillatory low diffusion (annulus)', 'Closed-loop stress (annulus)'),
                                 suite=SYNTH_SUITE, systems_fn=synthesis_envelope_systems(level=largest_level),
                                 lu_curve=curve, level_labels=level_labels, figures=SYNTH_FIGURES,
                                 degree_label=largest_label, title_pad=26, hspace=.5, mark_local_nc=True)
    return cells


def main():
    OUT.mkdir(exist_ok=True)
    bw, selected = blackwell()
    tuned, stock = h200('solver_jobs.csv'), h200('stock_unified_rows.csv')
    matched = matched_rows(bw, tuned, stock)
    best = family_best(selected, tuned)
    ext = extension(tuned)
    write_csv(OUT/'matched.csv', matched)
    write_csv(OUT/'family_best.csv', best)
    write_csv(OUT/'extension.csv', ext)
    (OUT/'matched_table.tex').write_text(matched_table(matched))
    summary, stats = summary_table(matched)
    (OUT/'summary_table.tex').write_text(summary)
    (OUT/'family_table.tex').write_text(family_table(best))
    (OUT/'extension_table.tex').write_text(extension_table(ext))
    outcome_rows = outcomes(tuned)
    write_csv(OUT/'outcomes.csv', outcome_rows)
    (OUT/'outcome_table.tex').write_text(outcome_table(outcome_rows))
    node = lu_rows(tuned, node_lu())
    crossover = lu_crossover(node)
    write_csv(OUT/'node_lu.csv', node)
    write_csv(OUT/'node_lu_crossover.csv', crossover)
    write_csv(OUT/'lu_followup_systems.csv', lu_followup(node))
    oscillatory = stock_oscillatory(h200('stock_unified_all.csv'), node_lu())
    write_csv(OUT/'stock_oscillatory.csv', oscillatory)
    (OUT/'stock_oscillatory_table.tex').write_text(stock_oscillatory_table(oscillatory))
    e2e = end_to_end(tuned, node_lu())
    if e2e is not None:
        write_csv(OUT/'end_to_end.csv', e2e)
        (OUT/'end_to_end_table.tex').write_text(end_to_end_table(e2e))
    (OUT/'node_lu_table.tex').write_text(lu_table(node, levels=('L1', 'L2', 'L3', 'L4')))
    (OUT/'node_lu_large_table.tex').write_text(lu_table(node, levels=('L4b', 'L5', 'L6', 'L7')))
    (OUT/'node_lu_crossover_table.tex').write_text(lu_crossover_table(crossover))
    lu_best = node_lu()
    figures(matched, tuned, lu_best)
    synth_records = synthesis_records(tuned, lu_best, extra_levels=('L4b', 'Lmax'))
    synthesis_cells = synthesis_figures(synth_records, lu_best)
    (OUT/'synthesis_largest_table.tex').write_text(largest_table(synth_records))
    (OUT/'levels_table.tex').write_text(levels_table(tuned))
    (OUT/'lmax_table.tex').write_text(lmax_table(tuned))
    (OUT/'stress_table.tex').write_text(stress_table(tuned, lu_best))
    write_csv(OUT/'synthesis_cells.csv', synthesis_cells)
    validation = dict(blackwell_measurements=len(bw), h200_jobs=len(tuned), stock_rows=len(stock), matched=len(matched),
                      matched_passed_both=sum(1 for r in matched if r['speedup_repeated']),
                      matched_with_stock=sum(1 for r in matched if r['speedup_repeated_stock']),
                      iteration_equal=sum(1 for r in matched if r['speedup_repeated'] and r['it_blackwell'] == r['it_h200']),
                      family_best_systems=len(best), stats=stats,
                      node_lu=dict(systems=len(node), lu_measured=sum(1 for r in node if r['lu_status'] != 'not_measured'),
                                   lu_residual_floor=sum(1 for r in node if r['lu_status'] == 'residual_floor'),
                                   lu_not_measured_by_level=dict(sorted(Counter(r['level'] for r in node if r['lu_status'] == 'not_measured').items())),
                                   iterative_faster_fresh=sum(1 for r in node if (r['iterative_over_lu_fresh'] or 9) < 1),
                                   iterative_faster_repeated=sum(1 for r in node if (r['iterative_over_lu_repeated'] or 9) < 1),
                                   lu_only_iterative_outcomes=dict(sorted(Counter(r['iterative_outcome'] for r in node
                                                                           if r['lu_status'] != 'not_measured' and r['iterative_repeated_ms'] is None).items()))),
                      status_pairs={f"{k[0]}->{k[1]}": v for k, v in sorted(
                          ((s, sum(1 for r in matched if (r['blackwell_status'], r['h200_status']) == s))
                           for s in {(r['blackwell_status'], r['h200_status']) for r in matched}), key=lambda kv: kv[0])})
    (OUT/'validation.json').write_text(json.dumps(validation, indent=2, default=str)+'\n')
    print(json.dumps({k: v for k, v in validation.items() if k != 'stats'}, default=str))
    print(json.dumps(stats, indent=1, default=lambda v: round(v, 3)))


if __name__ == '__main__':
    main()
