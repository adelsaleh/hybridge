#!/usr/bin/env python3
"""Create portable ADR scaling figures and a concise insertable LaTeX section."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
from hybridge.io.figures import publication_style, save_publication_figure
import adr_native_completion as native_completion
import adr_lu_baseline as lu
from adr_report_labels import PMG, PMG_TEX, polynomial_label, prose

CASES = ('trigonometric', 'advection_dominated', 'anisotropic')
TITLES = ('High diffusion', 'Transport dominated', 'Anisotropic diffusion')
SHORT = {'trigonometric': 'High', 'advection_dominated': 'Low', 'anisotropic': 'Aniso.'}
STYLE = {
    'asm_gmres': ('Block ASM–GMRES', '#0072B2', 'o', '-'),
    'amgx_fgmres_amg': ('AMG–FGMRES', '#D55E00', 's', '-'),
    'amgx_pbicgstab_amg': ('AMG–BiCGSTAB', '#009E73', 'D', '--'),
    'amgx_fgmres_dilu': ('DILU–FGMRES', '#6B7280', '^', '-.'),
    'amgx_pbicgstab_dilu': ('DILU–BiCGSTAB', '#E69F00', 'v', ':'),
    'native_hp': (PMG+' (S)', '#A65492', 'P', '-'),
    'native_hp_robust': (PMG+' (R)', '#785EF0', 'X', ':'),
}
TEX_METHOD = {
    'asm_gmres': r'ASM+PP / GMRES',
    'amgx_fgmres_amg': r'AMG / FGMRES',
    'amgx_pbicgstab_amg': r'AMG / BiCGSTAB',
    'amgx_fgmres_dilu': r'DILU / FGMRES',
    'amgx_pbicgstab_dilu': r'DILU / BiCGSTAB',
    'native_hp': PMG_TEX+' (standard)',
    'native_hp_robust': PMG_TEX+' (robust)',
}


def read(path):
    """Read a recorded JSON artifact."""
    return json.loads(Path(path).read_text())


def stats(row):
    """Derive hot-solve statistics only from fully passing candidates."""
    if row['status'] != 'passed':
        return None
    samples = row['samples']
    hot = [s for sample in samples for s in sample['solves'][1:]]
    values = [s['solve_ms'] for s in hot]
    return dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values),
        iterations=statistics.mean(s['iterations'] for s in hot),
        setup=row['setup_median_ms'], fresh=row['fresh_setup_solve_median_ms'],
        amortized=row['amortized_median_ms'], residual=max(s['true_relative_residual'] for sample in samples for s in sample['solves']),
        l2=max(s['l2_error'] for sample in samples for s in sample['solves']))


def legend(figure, *, y=0.01):
    """Draw a consistent method legend shared by every scaling panel."""
    handles = [Line2D([], [], label=label, color=color, marker=marker, linestyle=style)
               for label, color, marker, style in STYLE.values()]
    handles.append(lu.handle())
    figure.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
                  bbox_to_anchor=(0.52, y), columnspacing=1.2, handlelength=2.3)


def series(ax, rows, method, xkey, quantity):
    """Plot means and ranges, leaving failures as gaps and explicit crosses."""
    rows = sorted((r for r in rows if r['candidate'] == method), key=lambda r:r[xkey])
    if not rows:
        return
    label, color, marker, linestyle = STYLE[method]
    if method=='asm_gmres':
        ax.text(.025,.98,polynomial_label(rows[0]),transform=ax.transAxes,va='top',fontsize=7,color=color,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
    xs, ys, low, high = [], [], [], []
    for row in rows:
        value = stats(row)
        xs.append(row[xkey])
        if value is None:
            ys.append(np.nan); low.append(0); high.append(0)
            ax.plot(row[xkey], .975, marker='x', color=color, markersize=6,
                    transform=ax.get_xaxis_transform(), clip_on=False, linestyle='none')
            ax.annotate('NC', (row[xkey], .975), xycoords=ax.get_xaxis_transform(),
                        xytext=(-3,-10), textcoords='offset points', ha='right', fontsize=7, color=color)
        else:
            y = value[quantity]
            ys.append(y)
            low.append(y-value['minimum'] if quantity == 'mean' else 0)
            high.append(value['maximum']-y if quantity == 'mean' else 0)
    ax.errorbar(xs, ys, yerr=np.asarray([low, high]), label=label, color=color,
                marker=marker, linestyle=linestyle, capsize=2, elinewidth=.7,
                markeredgewidth=.65, markeredgecolor='white', zorder=4)


def mesh_figure(rows, out):
    """Compare solve time and iteration growth on the p=6 mesh ladder."""
    figure, axes = plt.subplots(2, 3, figsize=(6.65, 4.30), sharex='col', sharey='row')
    ticks = [2048, 8192, 32768, 99458]
    for column, (case, title) in enumerate(zip(CASES, TITLES)):
        subset = [r for r in rows if r['case'] == case and r['p'] == 6]
        for method in STYLE:
            series(axes[0, column], subset, method, 'triangles', 'mean')
            series(axes[1, column], subset, method, 'triangles', 'iterations')
        lu.curve(axes[0,column],subset,'smooth','hot','triangles')
        axes[0, column].set_title(title, pad=8)
        for ax in axes[:, column]:
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xticks(ticks, ['2k', '8k', '33k', '99k'])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis='both', which='major', alpha=.8)
            ax.set_xlim(1650, 125000)
        axes[1, column].set_xlabel(r'Triangles $N_T$ ($p=6$)')
    axes[0, 0].set_ylabel('Hot solve (ms)')
    axes[1, 0].set_ylabel('Krylov iterations')
    axes[0, 0].set_ylim(2, max(15000, max(stats(r)['mean'] for r in rows if r['p']==6 and stats(r))*1.7))
    axes[1, 0].set_ylim(5, 1500)
    figure.subplots_adjust(left=.09, right=.99, bottom=.23, top=.91, wspace=.12, hspace=.17)
    legend(figure)
    save_publication_figure(figure, out/'h_scaling')
    plt.close(figure)


def order_figure(rows, out):
    """Compare degree dependence on a fixed 8,192-triangle mesh."""
    figure, axes = plt.subplots(1, 3, figsize=(6.65, 2.60), sharey=True)
    for ax, case, title in zip(axes, CASES, TITLES):
        subset = [r for r in rows if r['case']==case and r['n']==64]
        for method in STYLE:
            series(ax, subset, method, 'p', 'mean')
        lu.curve(ax,subset,'smooth','hot','p')
        ax.set_title(title, pad=8)
        ax.set_yscale('log'); ax.set_xticks([1, 2, 3, 4, 6])
        ax.set_xlim(.8, 6.2); ax.grid(which='major', alpha=.8)
        ax.set_xlabel(r'Finite-element degree $p_{\mathrm{FE}}$')
    axes[0].set_ylabel('Hot solve (ms)')
    values = [stats(r)['mean'] for r in rows if r['n']==64 and stats(r)]
    values += [lu.lookup(r,'smooth')['hot'] for r in rows if r['n']==64]
    axes[0].set_ylim(min(values)*.65, max(values)*1.7)
    figure.subplots_adjust(left=.09, right=.99, bottom=.38, top=.85, wspace=.12)
    legend(figure)
    save_publication_figure(figure, out/'p_scaling')
    plt.close(figure)


def selected_large(rows):
    """Keep ASM, fastest block-AMG, native hp and transport direct-DILU."""
    selected=[]
    for case in CASES:
        group=[r for r in rows if r['case']==case and r['n']==223 and r['p']==6 and stats(r)]
        selected.extend(r for r in group if r['candidate']=='asm_gmres')
        amg=[r for r in group if r['candidate'] in ('amgx_fgmres_amg','amgx_pbicgstab_amg')]
        if amg:selected.append(min(amg,key=lambda r:stats(r)['mean']))
        selected.extend(r for r in group if r['candidate'].startswith('native_hp'))
        if case=='advection_dominated':selected.extend(r for r in group if r['candidate']=='amgx_pbicgstab_dilu')
    return selected


def cost_accuracy_figure(rows, assemblies, out):
    """Show setup amortization and the accuracy benefit of raising p."""
    figure, axes = plt.subplots(1, 2, figsize=(6.65, 3.7), gridspec_kw={'width_ratios':[1.35,1]})
    chosen=selected_large(rows)
    direct=[(case,lu.lookup(next(r for r in chosen if r['case']==case),'smooth','hot')) for case in CASES]
    y=np.arange(len(chosen)+len(direct))
    setup=np.asarray([stats(r)['setup']/1000 for r in chosen]+[d['setup']/1000 for _,d in direct])
    hot=np.asarray([stats(r)['mean']/1000 for r in chosen]+[d['hot']/1000 for _,d in direct])
    axes[0].barh(y,setup,color='#B8C2D1',height=.62,label='Setup')
    axes[0].barh(y,hot,left=setup,color='#244A73',height=.62,label='One hot solve')
    abbr={'asm_gmres':'ASM','amgx_fgmres_amg':'AMG(F)','amgx_pbicgstab_amg':'AMG(B)',
          'amgx_pbicgstab_dilu':'DILU(B)','native_hp':PMG+' (S)','native_hp_robust':PMG+' (R)'}
    labels=[SHORT[r['case']]+' · '+(polynomial_label(r) if r['candidate']=='asm_gmres' else abbr[r['candidate']]) for r in chosen]
    labels += [SHORT[case]+' · '+lu.alias(d['threads']) for case,d in direct]
    axes[0].set_yticks(y,labels); axes[0].invert_yaxis()
    for i,total in enumerate(setup+hot):
        axes[0].text(total+.08,i,f'{total:.2f}',va='center',fontsize=7.5)
    axes[0].set_xlim(0,float(max(setup+hot))*1.21)
    axes[0].set_xlabel('Setup + one hot solve (s)')
    axes[0].set_title('At 99,458 triangles, p=6',pad=8)
    axes[0].grid(axis='x',alpha=.7)
    axes[0].legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(-.32,-.16),fontsize=8)
    for case,color,marker in zip(CASES,('#244A73','#D55E00','#009E73'),('o','s','D')):
        a=sorted((v for v in assemblies if v['case']==case and v['n']==64),key=lambda v:v['p'])
        ps=[];errors=[]
        for v in a:
            e=v.get('validation',{}).get('reference_l2')
            if e is None:
                good=[stats(r)['l2'] for r in rows if r['case']==case and r['n']==64 and r['p']==v['p'] and stats(r)]
                e=statistics.median(good) if good else None
            if e is not None:ps.append(v['p']);errors.append(e)
        axes[1].semilogy(ps,errors,color=color,marker=marker,label=SHORT[case])
    axes[1].set_xlabel(r'Finite-element degree $p_{\mathrm{FE}}$')
    axes[1].set_ylabel(r'$\|u-u_h\|_{L^2(\Omega)}$')
    axes[1].set_xticks([1,2,3,4,6]); axes[1].set_xlim(.8,6.2)
    axes[1].set_title('Accuracy at 8,192 triangles',pad=8)
    axes[1].grid(which='major',alpha=.7)
    axes[1].legend(frameon=False,loc='upper right',fontsize=8)
    figure.subplots_adjust(left=.13,right=.985,top=.86,bottom=.22,wspace=.51)
    save_publication_figure(figure,out/'cost_accuracy')
    plt.close(figure)


def audit_campaign(rows, assemblies, campaign, out):
    """Cross-check common inputs, source hashes and independently saved solutions."""
    manifest = read(campaign / 'manifest.json')
    checks = []
    for assembly in assemblies:
        case, n, p = (assembly[key] for key in ('case', 'n', 'p'))
        group = [r for r in rows if (r['case'], r['n'], r['p']) == (case, n, p)]
        assert len(group) == len(manifest['candidates'][case])
        passed = [r for r in group if r['status'] == 'passed']
        digests = {r['operator_sha256'] for r in passed if 'operator_sha256' in r}
        assert len(digests) <= 1, 'Operators differ across solver families'
        reference_row = next((r for r in passed if r['candidate'] == 'asm_gmres'), None)
        reference = None
        if reference_row is not None:
            reference = np.load(campaign / 'jobs' / f'{case}_n{n}_p{p}_asm_gmres.solution.npy')
        deviations = []
        for row in passed:
            key = f"{case}_n{n}_p{p}_{row['candidate']}"
            spec = read(campaign / 'specs' / (key + '.json'))
            assert Path(spec['cache']).resolve() == Path(assembly['cache']).resolve()
            assert spec['internal_rtol'] == 1e-11 and spec['rtol'] == 1e-10
            worker = 'adr_native_hp_worker.py' if row['family'] == 'native_hp' else 'adr_solver_comparison_worker.py'
            assert row['worker_sha256'] == manifest['worker_sha256'][worker]
            solution = np.load(campaign / 'jobs' / (key + '.solution.npy'))
            assert solution.size == row['trace_dofs'] == (3*n*n - 2*n)*(p+1)
            if reference is not None:
                deviations.append(float(np.linalg.norm(solution-reference)/np.linalg.norm(reference)))
        assert not deviations or max(deviations) < 1e-8
        checks.append(dict(case=case, n=n, p=p, operator_sha256=next(iter(digests), None),
            matrix_input_paths_match=True,
            max_trace_relative_difference_to_asm=max(deviations, default=None)))
    solves = [s for r in rows if r['status'] == 'passed'
              for trial in r['warmups'] + r['samples'] for s in trial['solves']]
    references = [a for a in assemblies if 'reference_l2' in a.get('validation', {})]
    differences = [c['max_trace_relative_difference_to_asm'] for c in checks
                   if c['max_trace_relative_difference_to_asm'] is not None]
    result = dict(status='passed', configurations=len(rows), systems=len(assemblies), solves=len(solves),
        worst_physical_relative_residual=max(s['true_relative_residual'] for s in solves),
        worst_cross_solver_trace_relative_difference=max(differences, default=None),
        independent_cpu_reference_systems=len(references),
        worst_cpu_gpu_blocks_relative_error=max((a['validation']['cpu_gpu_blocks_relative_error'] for a in references), default=None),
        worst_cpu_gpu_rhs_relative_error=max((a['validation']['cpu_gpu_rhs_relative_error'] for a in references), default=None),
        matrix_and_worker_provenance_checks=checks)
    (out / 'validation.json').write_text(json.dumps(result, indent=2) + chr(10))
    return result


def write_data(rows, campaign, out):
    """Archive tabular points and the exact campaign manifest beside TeX."""
    flattened=[]
    for r in rows:
        values=stats(r)
        flattened.append({key:r[key] for key in ('case','n','p','triangles','trace_dofs','h','candidate','status')} | (values or {}))
    keys=['case','n','p','triangles','trace_dofs','h','candidate','status','mean','minimum','maximum','iterations','setup','fresh','amortized','residual','l2']
    with (out/'scaling.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(flattened)
    previous=read(ROOT/'docs/research/solver_studies/adr_solver_comparison_2026_09_17.data.json')
    provenance=previous['provenance']
    for binary, description in provenance['binaries'].items():
        digest=hashlib.sha256()
        with open(binary,'rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024**2),b''):digest.update(chunk)
        assert digest.hexdigest()==description['sha256'], 'Runtime binary changed: '+binary
    archive=dict(manifest=read(campaign/'manifest.json'),completion=read(campaign/'completion.json'),
        binary_and_repository_provenance=provenance,
        cited_prior_profiles={key:value for key,value in previous['profiles'].items() if key.endswith(('_components.json','_memory.json')) or key=='transport_tolerance_cycles.json'},
        rows=rows,assemblies=read(campaign/'assemblies.json'),validation=read(out/'validation.json'),local_campaign=str(campaign.resolve()),
        prior_study=str(ROOT/'docs/research/solver_studies/adr_solver_comparison_2026_09_17.data.json'))
    (out/'scaling.data.json').write_text(json.dumps(archive,indent=2)+'\n')
    return flattened



def results_prose(rows, completion, audit):
    """Derive compact statements from the measured scaling data."""
    def get(case, n, p, method):
        return next(r for r in rows if r['case']==case and r['n']==n and r['p']==p and r['candidate']==method)
    high_small=stats(get('trigonometric',32,6,'asm_gmres'))
    high_large=stats(get('trigonometric',223,6,'asm_gmres'))
    native_small=stats(get('trigonometric',32,6,'native_hp'))
    native_large=stats(get('trigonometric',223,6,'native_hp'))
    large_amg=[r for r in rows if r['case']=='trigonometric' and r['n']==223 and r['p']==6 and r['candidate'] in ('amgx_fgmres_amg','amgx_pbicgstab_amg') and stats(r)]
    best_amg=min(large_amg,key=lambda r:stats(r)['mean'])
    a=stats(best_amg)
    htext=(f"For high diffusion, ASM+PP grows from {high_small['iterations']:.0f} to {high_large['iterations']:.0f} iterations over the mesh ladder, "
        f"whereas native $hp$ grows from {native_small['iterations']:.0f} to {native_large['iterations']:.0f}. "
        f"At the largest size, native $hp$ takes {native_large['mean']/1000:.3f} s and the faster block-AMG variant {a['mean']/1000:.3f} s, "
        f"versus {high_large['mean']/1000:.3f} s for ASM+PP. "
        "The small-mesh ranking therefore does not predict the large-mesh winner.")
    direct_f=get('advection_dominated',223,6,'amgx_fgmres_dilu')
    direct_b=get('advection_dominated',223,6,'amgx_pbicgstab_dilu')
    if stats(direct_f) and stats(direct_b):
        f,b=stats(direct_f),stats(direct_b)
        dtext=(f"on the largest transport system, FGMRES/DILU takes {f['mean']/1000:.3f} s ({f['iterations']:.0f} iterations), "
            f"while BiCGSTAB/DILU takes {b['mean']/1000:.3f} s ({b['iterations']:.0f} iterations).")
    else:
        good=direct_b if stats(direct_b) else direct_f
        bad=direct_f if stats(direct_b) else direct_b
        if stats(good):
            v=stats(good)
            good_name='BiCGSTAB' if 'pbicgstab' in good['candidate'] else 'FGMRES'
            bad_name='BiCGSTAB' if 'pbicgstab' in bad['candidate'] else 'FGMRES'
            dtext=f"on the largest transport system, {good_name}/DILU passes in {v['mean']/1000:.3f} s ({v['iterations']:.0f} iterations), while {bad_name}/DILU fails the common contract."
        else:dtext="both direct-DILU outer methods fail the common contract on the largest transport system."
    p1=stats(get('trigonometric',64,1,'native_hp'))
    p6=stats(get('trigonometric',64,6,'native_hp'))
    ptext=(f"For high diffusion at fixed mesh, native $hp$ rises only from {p1['mean']:.1f} to {p6['mean']:.1f} ms as $p$ increases from 1 to 6 "
        f"({p1['iterations']:.0f} to {p6['iterations']:.0f} iterations). "
        "AMGX and ASM exhibit nonmonotonic degree dependence under the frozen configurations; no universal power law is inferred.")
    best={}
    for case in CASES:
        candidates=[r for r in rows if r['case']==case and r['n']==223 and r['p']==6 and stats(r)]
        best[case]=min(candidates,key=lambda r:stats(r)['mean'])
    ltext='At the largest size, the fastest tested hot-solve configurations are '+', '.join(
        TEX_METHOD[best[c]['candidate']]+f" for {title.lower()} ({stats(best[c])['mean']/1000:.3f} s)"
        for c,title in zip(CASES,TITLES))+'. Setup can change the preferred method for a single right-hand side; Table~'+chr(92)+'ref{tab:adr-largest} and Figure~'+chr(92)+'ref{fig:adr-cost-accuracy} keep it separate.'
    passed=[r for r in rows if stats(r) and r.get('origin') != 'native_completion']
    # Original campaign validation stays separate from the native completion audit.
    passed += [r for r in read(ROOT/'docs/research/solver_studies/adr_scaling_2026_09_17/scaling.data.json')['rows'] if r['candidate']=='native_hp']
    solves=[v for r in passed for sample in r['samples']+r['warmups'] for v in sample['solves']]
    worst=max(v['true_relative_residual'] for v in solves)
    exponent=int(np.floor(np.log10(worst)));mantissa=worst/10**exponent
    validation=(f"The original campaign attempts {completion['attempted']} solver/system combinations on 24 distinct systems; "
        f"{completion['passed']} pass and {len(completion['failures'])} fail. "
        f"The {len(solves)} warmup/measured solves belonging to passing configurations have a worst physical relative residual "
        f"of ${mantissa:.2f}\\times10^{{{exponent}}}$.")
    agreement = audit['worst_cross_solver_trace_relative_difference']
    power = int(np.floor(np.log10(agreement)))
    validation += (f" {audit['independent_cpu_reference_systems']} systems also pass independent CPU direct-solution checks. "
        f"Saved trace solutions agree across solvers within ${agreement/10**power:.2f}\\times10^{{{power}}}$ "
        "relative to the ASM+PP solution.")
    return dict(H_FINDING=htext,DILU_FINDING=dtext,P_FINDING=ptext,LARGE_FINDING=ltext,VALIDATION=validation)



def portable_bundle(out, figures):
    """Bundle the report fragment, standalone wrapper, figures and supporting data."""
    destination = figures.parent / 'adr_results_bundle.zip'
    portable_readme = (
        'ADR scaling results: insertable LaTeX and supporting measurements.\n\n'
        'Standalone: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex\n'
        'Typesetting was left to the user under the workspace no-compilation rule.\n'
        'The combined report includes the longer oscillatory subsection; actual pagination depends on the report layout.\n\n'
        'For a larger report load amsmath, graphicx, booktabs and hyperref. Set \\ADRResultsPath\n'
        'to this directory and \\ADRResultsFigurePath to its figures/ subdirectory,\n'
        'then \\input{\\ADRResultsPath/section.tex}.\n\n'
        'scaling.csv: mean/minimum/maximum are hot-solve times in ms; setup, fresh\n'
        'and amortized are median ms. The amortized field includes two solves per setup.\n'
        'iterations are mean hot iterations; residual and l2 are worst measured values.\n'
        'scaling.data.json retains samples, effective configurations, residual checks\n'
        'and original source/binary provenance; validation.json records the data audit.\n'
        'Benchmark/cache paths in JSON describe the original machine.\n'
        'comparison/ contains the focused AMGX--polynomial results and scaling data.\n'
        'oscillatory/ contains the three-class scaling subsection and its initial stress-study evidence.\n'
        'All 22,825-triangle measurements are preserved; they were not repeated.\n'
        'meshes/ and sources/ preserve the geometric inputs and numerical workers.\n'
    )
    section = (out / 'section.tex').read_text()
    section = section.replace(
        r'\providecommand{\ADRResultsPath}{docs/research/solver_studies/adr_scaling_2026_09_17}',
        r'\providecommand{\ADRResultsPath}{.}')
    section = '\n'.join(
        r'\providecommand{\ADRResultsFigurePath}{figures}'
        if line.startswith(r'\providecommand{\ADRResultsFigurePath}') else line
        for line in section.splitlines()) + '\n'
    preview = '\n'.join(
        r'\newcommand{\ADRResultsFigurePath}{figures}'
        if line.startswith(r'\newcommand{\ADRResultsFigurePath}') else line
        for line in (out / 'preview.tex').read_text().splitlines()) + '\n'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('README.txt', portable_readme)
        archive.writestr('section.tex', section)
        archive.writestr('preview.tex', preview)
        if (out/'main.tex').exists():
            main = '\n'.join(
                r'\newcommand{\ADRResultsFigurePath}{figures}'
                if line.startswith(r'\newcommand{\ADRResultsFigurePath}') else line
                for line in (out/'main.tex').read_text().splitlines()) + '\n'
            archive.writestr('main.tex', main)
        for name in ('largest_table.tex', 'scaling.csv', 'scaling.data.json', 'validation.json'):
            archive.write(out / name, name)
        for path in sorted((out/'closed_loop_stress').glob('*')):
            if path.suffix in ('.tex','.md'):
                archive.write(path,'closed_loop_stress/'+path.name)
        for path in sorted((out/'pardiso_lu').glob('*.tex')):
            archive.write(path,'pardiso_lu/'+path.name)
        for name in ('confirmed_timings.csv','comparisons.csv'):
            archive.write(lu.CAMPAIGN/name,'pardiso_lu/'+name)
        for path in sorted((figures/'closed_loop_stress').glob('*.json')):
            archive.write(path,'figures/closed_loop_stress/'+path.name)
        for path in sorted(figures.rglob('*')):
            if path.is_file() and path.suffix in ('.pdf', '.svg', '.png'):
                archive.write(path, 'figures/' + str(path.relative_to(figures)))
        for path in sorted((out/'native_completion').rglob('*')):
            if path.is_file() and path.suffix in ('.tex','.csv','.json'):
                archive.write(path,str(path.relative_to(out)))
        comparison=out/'comparison'
        if comparison.exists():
            for path in sorted(comparison.rglob('*')):
                if path.is_file() and path.suffix in ('.tex','.csv','.json'):
                    archive.write(path,str(path.relative_to(out)))
        # Keep the new subsection, complete data and initial stress evidence together.
        oscillatory=out/'oscillatory'
        if oscillatory.exists():
            for path in sorted(oscillatory.rglob('*')):
                if path.is_file() and path.suffix in ('.tex','.csv','.json'):
                    name=str(path.relative_to(out))
                    if path.suffix=='.tex' and path.parent.name=='initial':
                        content=path.read_text()
                        content=content.replace(
                            r'\providecommand{\ADRStressPath}{docs/research/solver_studies/adr_scaling_2026_09_17/oscillatory/initial}',
                            r'\providecommand{\ADRStressPath}{oscillatory/initial}')
                        content=content.replace(
                            r'\providecommand{\ADRStressFigurePath}{run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/initial/figures}',
                            r'\providecommand{\ADRStressFigurePath}{figures/oscillatory/initial}')
                        content=content.replace(
                            r'\newcommand{\ADRStressFigurePath}{../../../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/initial/figures}',
                            r'\newcommand{\ADRStressFigurePath}{../../figures/oscillatory/initial}')
                        archive.writestr(name,content)
                    else:archive.write(path,name)
            study=figures.parent/'oscillatory'
            for path in sorted((study/'initial/figures').glob('*')):
                if path.suffix in ('.pdf','.svg','.png'):archive.write(path,'figures/oscillatory/initial/'+path.name)
            for geometry in ('square','annulus'):
                snapshots=study/'scaling'/geometry/'source_snapshot'
                for path in sorted(snapshots.rglob('*.py')):
                    archive.write(path,'sources/oscillatory/'+geometry+'/'+str(path.relative_to(snapshots)))
            for subdir in ('diagnostics','smoother_probe'):
                for path in sorted((study/subdir).rglob('*')):
                    if path.is_file() and path.suffix in ('.py','.json','.log'):
                        archive.write(path,'evidence/oscillatory/'+str(path.relative_to(study)))
            for name in ('raw_migration.json',):
                archive.write(study/name,'evidence/oscillatory/'+name)
            geometry_root=ROOT.parent/'hdgfem-gmres/run_logs/adr_oscillatory_scaling_geometry_20260918'
            for path in sorted(geometry_root.glob('star_target*')):
                if path.suffix in ('.npz','.json'):archive.write(path,'meshes/oscillatory/'+path.name)
            for path in sorted((ROOT.parent/'hdgfem-gmres/run_logs/adr_oscillatory_geometry_20260918').glob('star_*')):
                if path.suffix in ('.npz','.json'):archive.write(path,'meshes/oscillatory/initial/'+path.name)
            for path in sorted((ROOT.parent/'hdgfem-gmres/run_logs/adr_oscillatory_screen_20260918/source_snapshot').glob('*.py')):
                archive.write(path,'sources/oscillatory/initial/'+path.name)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figures',type=Path,help='Figure output directory (default: run_outputs/solver_studies/OUTPUT_NAME/figures)')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    figures=(args.figures or ROOT/'run_outputs'/'solver_studies'/out.name/'figures').resolve()
    assert not figures.is_relative_to(ROOT/'docs'), 'Generated PDFs belong outside docs/'
    figures.mkdir(exist_ok=True,parents=True)
    completion=read(args.campaign/'completion.json')
    assert completion['attempted']==completion['scheduled'],'Scaling campaign is unfinished'
    rows=read(args.campaign/'summary.json');assemblies=read(args.campaign/'assemblies.json')
    assert len(rows)==completion['scheduled']
    for row in rows:
        if row['status']=='passed':
            assert row['rtol']==1e-10 and row['internal_rtol']==1e-11
            assert len(row['samples'])==3 and len(row['warmups'])==1
            assert all(len(v['solves'])==2 and all(s['passed'] and s['true_relative_residual']<=1e-10 for s in v['solves']) for v in row['samples']+row['warmups'])
    audit = audit_campaign(rows,assemblies,args.campaign,out)
    original_rows = rows
    write_data(original_rows,args.campaign,out)
    rows = native_completion.overlay(rows, 'smooth')
    with publication_style():
        mesh_figure(rows,figures)
        order_figure(rows,figures)
        cost_accuracy_figure(rows,assemblies,figures)
    chosen=selected_large(rows)
    table=[r'\begin{tabular}{llrrr}',r'\toprule',r'Case & Preconditioner / Krylov & Iter. & Hot (ms) & Setup (ms) \\',r'\midrule']
    for row in chosen:
        v=stats(row)
        table.append(f"{SHORT[row['case']]} & {polynomial_label(row)+' / GMRES' if row['candidate']=='asm_gmres' else TEX_METHOD[row['candidate']]} & {v['iterations']:.0f} & {v['mean']:.1f} & {v['setup']:.1f} \\\\")
    for case in CASES:
        direct=lu.lookup(next(r for r in chosen if r['case']==case),'smooth','hot')
        table.append(f"{SHORT[case]} & {lu.alias(direct['threads'],tex=True)} & -- & {direct['hot']:.1f} & {direct['setup']:.1f}"+r' \\')
    table += [r'\bottomrule',r'\end{tabular}']
    (out/'largest_table.tex').write_text('\n'.join(table)+'\n')
    replacements = results_prose(rows, completion, audit)
    replacements['FIGURE_PATH'] = str(figures.relative_to(ROOT) if figures.is_relative_to(ROOT) else figures)
    template = Path(__file__).with_name('adr_results_section.tex.in').read_text()
    for key, value in replacements.items():
        template = template.replace('@@' + key + '@@', value)
    if not completion['failures']:
        template=template.replace("A cross in a panel's top strip marks a failed run, not a timing\n value.  ", '')
    assert '@@' not in template
    template=prose(template)
    (out/'reference_section.tex').write_text(template)
    combined=template
    for subsection in ('comparison/subsections.tex', 'oscillatory/subsection.tex', 'native_completion/summary.tex'):
        if (out/subsection).exists():
            combined += '\n'+rf'\input{{\ADRResultsPath/{subsection}}}'+'\n'
    (out/'section.tex').write_text(combined)
    provenance={str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(figures.rglob('*')) if p.is_file()}
    (out/'figure_hashes.json').write_text(json.dumps(provenance,indent=2)+'\n')
    portable_bundle(out,figures)
    print(json.dumps({'completion':completion,'largest':[dict(case=r['case'],method=r['candidate'],**stats(r)) for r in chosen]},indent=2))


if __name__=='__main__':
    main()
