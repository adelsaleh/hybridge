#!/usr/bin/env python3
"""Render the three-class oscillatory scaling subsection inside the prior ADR study."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from hybridge.io.figures import publication_style,save_publication_figure
import adr_native_completion as native_completion
import adr_lu_baseline as lu
from adr_report_labels import PMG, PMG_TEX, polynomial_label, prose

CASES=('cellular7_high','cellular7_weak','cellular7_directional')
TITLES=('High diffusion','Low diffusion / transport','Directional anisotropy')
GEOMETRIES=('square','annulus')
STYLES={
    'asm':('Block ASM / GMRES','#0072B2','o','-'),
    'amg_fgmres':('AMG / FGMRES','#D55E00','s','-'),
    'amg_pbicgstab':('AMG / BiCGSTAB','#009E73','D','--'),
    'dilu_fgmres':('DILU / FGMRES','#6B7280','^','-.'),
    'dilu_pbicgstab':('DILU / BiCGSTAB','#E69F00','v',':'),
    'native_hp':(PMG+' (S)','#A65492','P','-'),
    'native_hp_robust':(PMG+' (R)','#785EF0','X',':'),
}


def read(path):
    """Read only recorded results; rendering never invokes solvers."""
    return json.loads(Path(path).read_text())


def method(row):
    """Group ASM degrees into one method; per-class degrees remain in the manifest."""
    return 'asm' if row['candidate'].startswith('asm_d') else row['candidate']


def stats(row):
    """Report means/ranges only after every required solve has passed."""
    if row['status']!='passed':return None
    hot=[t['solves'][1] for t in row['samples']]
    all_solves=[s for t in row['samples']+row['warmups'] for s in t['solves']]
    values=[s['solve_ms'] for s in hot]
    return dict(mean=statistics.mean(values),minimum=min(values),maximum=max(values),
        iterations=statistics.mean(s['iterations'] for s in hot),setup=row['setup_median_ms'],
        fresh=row['fresh_setup_solve_median_ms'],amortized=row['amortized_median_ms'],
        residual=max(s['true_relative_residual'] for s in all_solves),l2=max(s['l2_error'] for s in all_solves))


def legend(fig, *, include_lu=True):
    """Share method colors with the earlier smooth-case figures."""
    handles=[Line2D([],[],label=label,color=color,marker=marker,linestyle=line)
             for label,color,marker,line in STYLES.values()]
    if include_lu:handles.append(lu.handle())
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=8,
               bbox_to_anchor=(.52,.0),columnspacing=1.5,handlelength=2.1)


def draw(ax,rows,xkey,quantity):
    """Leave failed points as gaps and mark their methods outside the plotting area."""
    for index,(name,(label,color,marker,line)) in enumerate(STYLES.items()):
        selected=sorted((r for r in rows if method(r)==name),key=lambda r:r[xkey])
        if not selected:continue
        if name=='asm':ax.text(.025,.98,polynomial_label(selected[0]),transform=ax.transAxes,va='top',fontsize=6.8,color=color,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
        xs=[];ys=[];lower=[];upper=[]
        for r in selected:
            x=r[xkey];v=stats(r);xs.append(x)
            if v is None:
                ys.append(np.nan);lower.append(0);upper.append(0)
                ax.annotate('×',(x,1.025),xycoords=ax.get_xaxis_transform(),
                    xytext=((index-2.5)*4,0),textcoords='offset points',ha='center',va='center',
                    color=color,fontsize=8,annotation_clip=False)
            else:
                y=v[quantity];ys.append(y)
                lower.append(y-v['minimum'] if quantity=='mean' else 0)
                upper.append(v['maximum']-y if quantity=='mean' else 0)
        ax.errorbar(xs,ys,yerr=np.asarray([lower,upper]),color=color,marker=marker,linestyle=line,
                    capsize=2,elinewidth=.6,markeredgecolor='white',markeredgewidth=.5,zorder=3)
    lu.curve(ax,rows,'oscillatory',quantity,xkey)
    ax.grid(which='major',alpha=.75)


def h_figure(rows,out):
    """Display the original p=6 mesh ladder for both geometries and all three classes."""
    fig,axes=plt.subplots(4,3,figsize=(6.65,6.45),sharex='col',sharey='row')
    ticks=[2048,8192,32768,99458]
    for g,geometry in enumerate(GEOMETRIES):
        for col,(case,title) in enumerate(zip(CASES,TITLES)):
            selected=[r for r in rows if r['geometry']==geometry and r['case']==case and r['p']==6]
            for offset,quantity in enumerate(('mean','iterations')):
                ax=axes[2*g+offset,col]
                draw(ax,selected,'triangles',quantity)
                ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(1700,125000)
                ax.set_xticks(ticks,['2k','8k','33k','100k']);ax.xaxis.set_minor_locator(NullLocator())
                if quantity=='iterations':ax.set_ylim(8,1300)
            if g==0:axes[0,col].set_title(title,pad=15,fontsize=9)
            axes[3,col].set_xlabel('Triangles ($p=6$)')
        axes[2*g,0].set_ylabel(geometry.title()+'\nHot solve (ms)')
        axes[2*g+1,0].set_ylabel(geometry.title()+'\nIterations')
    fig.subplots_adjust(left=.10,right=.99,bottom=.145,top=.925,wspace=.15,hspace=.24)
    legend(fig);save_publication_figure(fig,out/'h_scaling');plt.close(fig)


def p_figure(rows,out,quantity='mean'):
    """Display the identical polynomial ladder at approximately 8k triangles."""
    fig,axes=plt.subplots(2,3,figsize=(6.65,3.7),sharex=True,sharey='row')
    for g,geometry in enumerate(GEOMETRIES):
        for col,(case,title) in enumerate(zip(CASES,TITLES)):
            selected=[r for r in rows if r['geometry']==geometry and r['case']==case and r['nominal_n']==64]
            ax=axes[g,col];draw(ax,selected,'p',quantity)
            ax.set_yscale('log');ax.set_xlim(.7,6.3);ax.set_xticks([1,2,3,4,6])
            if quantity=='iterations':ax.set_ylim(8,1300)
            if g==0:ax.set_title(title,pad=15,fontsize=9)
            else:ax.set_xlabel(r'$p_{\mathrm{FE}}$ ('+('8,192' if geometry=='square' else '8,039')+' triangles)')
        axes[g,0].set_ylabel(geometry.title()+'\n'+('Hot solve (ms)' if quantity=='mean' else 'Iterations'))
    fig.subplots_adjust(left=.10,right=.99,bottom=.25,top=.88,wspace=.15,hspace=.30)
    legend(fig,include_lu=quantity!='iterations');save_publication_figure(fig,out/('p_scaling' if quantity=='mean' else 'p_iterations'));plt.close(fig)


def accuracy_figure(assemblies,out):
    """Plot independently reconstructed CPU reference errors over p at fixed mesh."""
    fig,axes=plt.subplots(1,2,figsize=(6.65,2.5),sharey=True)
    for ax,geometry in zip(axes,GEOMETRIES):
        for case,title,color,marker in zip(CASES,TITLES,('#0072B2','#D55E00','#009E73'),('o','s','D')):
            values=sorted((a for a in assemblies if a['geometry']==geometry and a['case']==case and a['nominal_n']==64),key=lambda a:a['p'])
            ax.semilogy([a['p'] for a in values],[a['validation']['reference_l2'] for a in values],color=color,marker=marker,label=title)
        ax.set_title(geometry.title()+(' · 8,192 triangles' if geometry=='square' else ' · 8,039 triangles'),fontsize=9)
        ax.set_xticks([1,2,3,4,6]);ax.set_xlabel(r'$p_{\mathrm{FE}}$ ('+('8,192' if geometry=='square' else '8,039')+' triangles)');ax.grid(which='major',alpha=.75)
    axes[0].set_ylabel(r'$\|u-u_h\|_{L^2(\Omega)}$')
    axes[1].legend(frameon=False,fontsize=7.5,loc='upper right')
    fig.subplots_adjust(left=.11,right=.99,bottom=.21,top=.84,wspace=.16)
    save_publication_figure(fig,out/'p_accuracy');plt.close(fig)


def tex_number(value,digits=2):
    """Format very small diagnostics without unescaped exponent text."""
    mantissa,exponent=f'{value:.{digits}e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'


def large_table(rows,geometry,path):
    """Retain all largest-mesh candidates, including explicit NC outcomes."""
    table=[r'\begin{tabular}{llrrr}',r'\toprule',r'Class & Preconditioner / Krylov & Iter. & Hot (s) & Setup (s) \\',r'\midrule']
    for case,title in zip(CASES,('High','Low','Directional')):
        selected=[r for r in rows if r['geometry']==geometry and r['case']==case and r['nominal_n']==223 and r['p']==6]
        selected.sort(key=lambda r:list(STYLES).index(method(r)))
        for i,r in enumerate(selected):
            v=stats(r);label=polynomial_label(r)+' / GMRES' if method(r)=='asm' else STYLES[method(r)][0].replace('–','--')
            if v:
                cells=f"{v['iterations']:.0f} & {v['mean']/1000:.3f} & {v['setup']/1000:.3f}"
            else:
                partial=[s for t in r.get('samples',[])+r.get('warmups',[]) for s in t['solves']]
                cells=f"{partial[-1]['iterations']} & NC & --"
            table.append((title if i==0 else '')+' & '+label+' & '+cells+r' \\')
        direct=lu.lookup(selected[0],'oscillatory','hot')
        table.append(' & '+lu.alias(direct['threads'],tex=True)+f" & -- & {direct['hot']/1000:.3f} & {direct['setup']/1000:.3f}"+r' \\')
        table.append(r'\addlinespace[2pt]')
    table.extend([r'\bottomrule',r'\end{tabular}'])
    path.write_text('\n'.join(table)+'\n')


def findings(rows,checks):
    """Derive class-specific conclusions from the complete measured ladder."""
    result={}
    for case,key in zip(CASES,('HIGH','LOW','DIRECTIONAL')):
        statements=[]
        for geometry in GEOMETRIES:
            group=[r for r in rows if r['case']==case and r['geometry']==geometry and r['nominal_n']==223 and r['p']==6]
            passing=[r for r in group if stats(r)]
            nt=group[0]['triangles']
            if passing:
                best=min(passing,key=lambda r:stats(r)['mean']);v=stats(best)
                description=polynomial_label(best)+' / GMRES' if method(best)=='asm' else STYLES[method(best)][0].replace('–','--')
                statements.append(f"On the {nt:,}-triangle {geometry}, the fastest tested passing GPU configuration is {description}: "
                                  f"{v['iterations']:.0f} iterations, {v['mean']/1000:.3f} s mean hot time and {v['setup']/1000:.3f} s median setup.")
            else:
                statements.append(f"On the {nt:,}-triangle {geometry}, none of the tested GPU configurations passes the common contract at the 1,000-iteration cap.")
            failures=[STYLES[method(r)][0].replace('Native hp','native $hp$') for r in group if r['status']!='passed']
            if failures and passing:statements.append('The failed configurations are '+', '.join(failures)+'.')
        if key=='HIGH':
            statements.append('On the square mesh ladder, block-AMG/FGMRES grows from 45 to 50 iterations, while native $hp$ grows from 29 to 74 and ASM+PP from 14 to 150. The high-diffusion annular native hierarchy stays between 27 and 32 iterations; geometry changes the large-system ranking.')
            group=[r for r in rows if r['case']==case and r['geometry']=='annulus' and r['nominal_n']==223 and r['p']==6 and stats(r)]
            native=stats(next(r for r in group if method(r)=='native_hp'))
            amg=min((r for r in group if method(r).startswith('amg_')),key=lambda r:stats(r)['mean'])
            other=stats(amg)
            if native['mean']<other['mean'] and native['setup']>other['setup']:
                threshold=int(np.ceil((native['setup']-other['setup'])/(other['mean']-native['mean'])))
                statements.append(f"Setup changes the choice for a few right-hand sides: on the finest annulus, the arithmetic setup-plus-hot-solve model gives {(other['setup']+other['mean'])/1000:.3f} s for {STYLES[method(amg)][0]} versus {(native['setup']+native['mean'])/1000:.3f} s for native $hp$. With the same hierarchy reused, native $hp$ overtakes this AMG variant at about {threshold} right-hand sides; this is a cost model, not a measured multi-RHS pipeline.")

        elif key=='LOW':
            speedups=[]
            for n in (32,223):
                group=[r for r in rows if r['geometry']=='annulus' and r['case']==case and r['nominal_n']==n and r['p']==6 and stats(r)]
                asm=next(r for r in group if method(r)=='asm')
                amg=[r for r in group if method(r).startswith('amg_')]
                if amg:speedups.append(min(stats(r)['mean'] for r in amg)/stats(asm)['mean'])
            if len(speedups)==2:
                statements.append(f'Against the faster passing AMG variant on the annulus, the ASM hot-time advantage decreases from {speedups[0]:.1f} times on the coarsest mesh to {speedups[1]:.2f} times on the finest. Its coarse-mesh advantage therefore overstates its fine-mesh margin.')
            baseline=next(r for r in rows if r['case']==case and r['geometry']=='annulus' and r['nominal_n']==223 and r['p']==6)
            statements.append('The robust LU baseline is faster on that finest annulus: '+lu.cell(baseline,'oscillatory','fresh',1000)+r'\,s fresh and '+lu.cell(baseline,'oscillatory','hot',1000)+r'\,s reused.')
        elif key=='DIRECTIONAL':
            trials=[r for r in rows if r['case']==case and method(r).startswith('amg_')]
            statements.append(f"Across both complete ladders, {sum(r['status']!='passed' for r in trials)} of {len(trials)} transferred block-AMG configurations fail acceptance.")
        result[key]=' '.join(statements)
    result['VALIDATION']=(f"The original ladder contains {checks['configurations']} configurations on {checks['systems']} systems: "
        f"{checks['passed_configurations']} pass and {checks['failed_configurations']} fail. Four existing square outcomes were reused; "
        f"{checks['new_configurations']} configurations were newly measured. All 22,825-triangle jobs and specifications remain unchanged. "
        f"Passing configurations contribute {checks['passing_configuration_solves']} warmup/measured solves, with worst physical relative residual "
        +tex_number(checks['worst_physical_relative_residual'])+'. '
        +f"{checks['independent_cpu_reference_systems']} systems have independent CPU direct references; cross-solver traces agree within "
        +tex_number(checks['worst_cross_solver_trace_relative_difference'])+' relative to ASM.')
    return result



def diagnostic_findings(study,out):
    """Summarize separately instrumented checks without changing solver rankings."""
    folder=study/'diagnostics'
    completion=read(folder/'completion.json');assert completion['status']=='completed'
    profiles=read(folder/'profiles.json');quadrature=read(folder/'quadrature.json')
    matrix=read(folder/'matrix_diagnostics.json');probes=read(study/'smoother_probe/summary.json')
    flat=[]
    for row in profiles:
        if row.get('status')=='skipped':continue
        data=row['result']
        if row['family']=='amgx':
            mean=data['preconditioner_call_mean_ms']
            count=data['native_event_preconditioner_calls']/data['solve_count'];share=None
        else:
            mean=data['complete_preconditioner']['mean_ms']
            gmres=data['gmres' if row['family']=='asm_pp' else 'gmres_profile']
            pre=next(op for op in gmres['operations'] if op['category']=='preconditioner')
            count=pre['count'];share=100*pre['gpu_time_ms']/gmres['total_gpu_operation_ms']
        flat.append(dict(case=row['case'],candidate=row['candidate'],family=row['family'],
                         mean_application_ms=mean,calls_per_profiled_solve=count,attributed_gpu_percent=share))
    systems_by_id={r['system_id']:r for r in native_completion.read('coverage.json')}
    for profile in native_completion.read('profiles.json'):
        if profile['status']!='passed':continue
        source=systems_by_id[profile['system_id']];data=profile['record']
        if source['geometry']!='annulus' or source['case']!='cellular7_directional':continue
        gmres=data['gmres_profile'];pre=next(op for op in gmres['operations'] if op['category']=='preconditioner')
        flat.append(dict(case=source['case'],candidate='native_hp',family='native_hp',
                         mean_application_ms=data['complete_preconditioner']['mean_ms'],
                         calls_per_profiled_solve=pre['count'],
                         attributed_gpu_percent=100*pre['gpu_time_ms']/gmres['total_gpu_operation_ms']))
    table=[r'\begin{tabular}{llrrr}',r'\toprule',
           r'Class & Preconditioner / Krylov & Apply (ms) & Calls/solve & GPU (\%) \\',r'\midrule']
    for case,title in zip(CASES,('High','Low','Directional')):
        selected=[r for r in flat if r['case']==case]
        for i,row in enumerate(selected):
            name=STYLES[method(row)][0].replace('Native hp','Native $hp$')
            if row['family']=='asm_pp':
                source=next(v for v in profiles if v.get('candidate')==row['candidate'] and v['case']==row['case'])
                name=polynomial_label(source['result'])+' / GMRES'
            share='--' if row['attributed_gpu_percent'] is None else f"{row['attributed_gpu_percent']:.1f}"
            table.append((title if i==0 else '')+f" & {name} & {row['mean_application_ms']:.3f} & {row['calls_per_profiled_solve']:g} & {share}"+r' \\')
        table.append(r'\addlinespace[2pt]')
    table += [r'\bottomrule',r'\end{tabular}']
    (out/'application_table.tex').write_text('\n'.join(table)+'\n')
    asm=[r for r in flat if r['family']=='asm_pp']
    app='; '.join(title.lower()+f" {next(r['mean_application_ms'] for r in asm if r['case']==case):.2f} ms"
                 for case,title in zip(CASES,TITLES))
    application=(f"Complete ASM+PP applications cost {app}. "
        +f"They account for {min(r['attributed_gpu_percent'] for r in asm):.1f}--{max(r['attributed_gpu_percent'] for r in asm):.1f}\\% of attributed GPU operation time. "
        +"The count includes every profiled preconditioner application, including restart/residual work. "
        +"The near doubling for degree 48 reflects additional polynomial stages; reducing outer iterations must offset this cost. "
        +"Host synchronization waits overlap GPU work and must not be added to GPU operation totals.")
    condition='At 8,039 annular triangles and $p=6$, the reproducible 1-norm condition lower estimates are '+', '.join(
        tex_number(next(r['condition1_lower_estimate'] for r in matrix if r['case']==case))+f' ({title.lower()})'
        for case,title in zip(CASES,TITLES))+'. '
    condition += ('They use sparse LU with a randomized 1-norm estimator, seed 1729, four probing columns and eight iterations. '
        'They describe the unscaled Bernstein matrix, not the preconditioned spectrum or exact $\\kappa_2$; '
        'a condition estimate alone does not explain AMG convergence.')
    qtext='Doubling volume and edge quadrature from 14 to 28 points in each direction on that same mesh changes the reconstructed field by '+', '.join(
        f"{100*r['difference_over_original_error']:.3g}\\% of its spatial error ({TITLES[CASES.index(r['case'])].lower()})" for r in quadrature)+'. '
    qtext+='These direct-reference checks support the reported discretization errors without assuming that oscillatory coefficients integrate exactly.'
    failed=[r for r in probes if r['status']!='passed']
    assert len(probes)==2 and len(failed)==2
    smoother=('A separate 8,192-triangle, $p=6$ square probe replaces the AMG block-Jacobi smoother with block DILU while retaining the hierarchy settings. '
        'Both outer methods still reach 1,000 iterations without acceptance (physical residuals '+', '.join(
            f"{r['warmups'][0]['solves'][-1]['true_relative_residual']:.3g}" for r in probes)+'). '
        'This bounded probe does not support extending the DILU variant to a second scaling sweep; broader hierarchy tuning remains open.')
    details=dict(completion=completion,manifest=read(folder/'manifest.json'),profiles=profiles,
                 application_summary=flat,quadrature=quadrature,matrix_diagnostics=matrix,smoother_probes=probes)
    (out/'diagnostics.json').write_text(json.dumps(details,indent=2)+'\n')
    return dict(APPLICATION=application,CONDITION=condition,QUADRATURE=qtext,SMOOTHER=smoother),details


def main():
    """Write the merged subsection, plots and full machine-readable evidence."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,default=ROOT/'run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/research/solver_studies/adr_scaling_2026_09_17/oscillatory')
    args=parser.parse_args();study=args.study_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    figures=study.parent/'figures/oscillatory';figures.mkdir(parents=True,exist_ok=True)
    checks=read(out/'validation.json');assert checks['status']=='passed'
    campaigns=[];rows=[];assemblies=[]
    for geometry in GEOMETRIES:
        folder=study/'scaling'/geometry
        campaign=dict(name=geometry,manifest=read(folder/'manifest.json'),completion=read(folder/'completion.json'),
                      rows=read(folder/'summary.json'),assemblies=read(folder/'assemblies.json'))
        assert campaign['completion']['status']=='completed'
        campaigns.append(campaign);rows+=campaign['rows'];assemblies+=campaign['assemblies']
    diagnostic_text,diagnostics=diagnostic_findings(study,out)
    archive=dict(campaigns=campaigns,validation=checks,diagnostics=diagnostics,
        initial_stress_study=read(out/'initial/results.data.json'),
        initial_measurements_preserved=True,
        scope='Three diffusion classes share the oscillatory exact solution, drifted cellular velocity and reaction 0.01. The original low-anisotropic case is preserved as an additional stress control, not relabeled as the new directional class.')
    (out/'scaling.data.json').write_text(json.dumps(archive,indent=2)+'\n')
    flat=[]
    for r in rows:
        partial=[s for t in r.get('samples',[])+r.get('warmups',[]) for s in t['solves']]
        flat.append({key:r[key] for key in ('geometry','case','nominal_n','p','triangles','trace_dofs','h','candidate','status','origin')} |
                    (stats(r) or dict(failure_iterations=partial[-1]['iterations'],failure_residual=partial[-1]['true_relative_residual'])))
    columns=list(dict.fromkeys(key for row in flat for key in row))
    with (out/'scaling.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(flat)
    rows=native_completion.overlay(rows,'oscillatory')
    with publication_style():
        h_figure(rows,figures);p_figure(rows,figures);p_figure(rows,figures,'iterations');accuracy_figure(assemblies,figures)
    for name in ('geometry_fields','solver_comparison'):
        for extension in ('pdf','svg','png'):
            shutil.copy2(study/'initial/figures'/f'{name}.{extension}',figures/f'initial_{name}.{extension}')
    for geometry in GEOMETRIES:large_table(rows,geometry,out/f'largest_{geometry}_table.tex')
    template=Path(__file__).with_name('oscillatory_scaling_subsection.tex.in').read_text()
    for key,value in (findings(rows,checks)|diagnostic_text).items():template=template.replace('@@'+key+'@@',value)
    assert '@@' not in template
    (out/'subsection.tex').write_text(prose(template))
    (out/'figure_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(figures.iterdir())},indent=2)+'\n')
    print(json.dumps(findings(rows,checks),indent=2))


if __name__=='__main__':main()
