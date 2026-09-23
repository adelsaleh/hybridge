#!/usr/bin/env python3
"""Create the concise ASM/BJ+PP-centred ADR report from saved measurements."""
from __future__ import annotations
import csv, hashlib, json, statistics, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
from hdgfem.io.figures import publication_style, save_publication_figure
import adr_native_completion as native_completion
import adr_lu_baseline as lu
from adr_report_labels import PMG, PMG_TEX, polynomial_label
REPORT=ROOT/'docs/research/solver_studies/adr_scaling_2026_09_17'
SYNTH=REPORT/'synthesis'
FIGURES=ROOT/'run_outputs/solver_studies/adr_scaling_2026_09_17/figures/synthesis'
BUNDLE=ROOT/'run_outputs/solver_studies/adr_scaling_2026_09_17/adr_results_synthesis_bundle.zip'
TITLES={'Transport dominated':'Smooth transport','Variable velocity':'Variable velocity','High diffusion':'Smooth high diffusion'}
ASM_LABEL='ASM+PP'
BJ_LABEL='BJ+PP'

def num(v):
    try:return float(v)
    except (TypeError,ValueError):return None

def csvrows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def normalize():
    """Preserve configuration metadata directly from the saved measurements."""
    from make_adr_named_comparison_report import method as focused_method, LABEL
    out=[]
    def add(d,suite,geometry='square',case=None,n=None,method=None):
        n=d.get('nominal_n',d['n']) if n is None else n
        p=d['p'];passed=d['status']=='passed'
        hot=[sample['solves'][1] for sample in d['samples']] if passed else []
        first=[sample['solves'][0] for sample in d['samples']] if passed else []
        config=d.get('configuration',{})
        out.append(dict(suite=suite,geometry=geometry,case=case or d['case'],n=n,p=p,
                        dofs=d.get('trace_dofs',(3*d['n']**2-2*d['n'])*(p+1)),
                        triangles=d.get('triangles',2*d['n']**2),
                        method=method or d['candidate'],status=d['status'],
                        hot=statistics.mean(s['solve_ms'] for s in hot) if hot else None,
                        fresh=d.get('fresh_setup_solve_median_ms') if passed else None,
                        setup=d.get('setup_median_ms') if passed else None,
                        iterations=statistics.mean(s['iterations'] for s in hot) if hot else None,
                        fresh_iterations=statistics.mean(s['iterations'] for s in first) if first else None,
                        pp_degree=config.get('polynomial_degree'),policy=d.get('policy'),
                        configuration=config,source_candidate=d['candidate']))
    for d in json.loads((REPORT/'scaling.data.json').read_text())['rows']:
        if not d['candidate'].startswith('native_hp'):add(d,'smooth')
    for study,name in (('million_dof','adr_named_comparison_vv_mt_tr_20260919'),('scaling','adr_named_hp_scaling_vv_mt_tr_20260919')):
        for d in json.loads((ROOT.parent/'hdgfem-gmres/run_logs'/name/'summary.json').read_text()):
            if d['candidate'].startswith('native_hp'):continue
            case={'advection_dominated':'Transport dominated','trigonometric':'High diffusion','variable_velocity':'Variable velocity'}[d['case']]
            add(d,'focused_'+study,case=case,method=LABEL[focused_method(d)])
    for c in json.loads((REPORT/'oscillatory/scaling.data.json').read_text())['campaigns']:
        for d in c['rows']:
            if not d['candidate'].startswith('native_hp'):add(d,'oscillatory',d['geometry'])
    for source_suite,destination in (('smooth','smooth'),('oscillatory','oscillatory'),('focused_scaling','focused_scaling'),('focused_endpoints','focused_million_dof')):
        for d in native_completion.rows_for_suite(source_suite):
            case={'advection_dominated':'Transport dominated','variable_velocity':'Variable velocity','trigonometric':'High diffusion'}.get(d['case'],d['case']) if destination.startswith('focused') else d['case']
            add(d,destination,d['geometry'],case=case,method='native_hp_'+d['policy'])
            out[-1]['system_id']=d['system_id']
    audit=native_completion.CAMPAIGN.parent/'oscillatory/asm_optimality_audit'
    for geometry,name in (('square','square_asm_d20_raw_final.json'),('annulus','annulus_asm_d29_raw_final.json')):
        d=json.loads((audit/name).read_text())
        dofs=int(np.load(Path(d['matrix_cache'])/'system_rhs.npy',mmap_mode='r').size)
        source=next(r for r in out if r['suite']=='oscillatory' and r['geometry']==geometry and r['case']==d['case'] and r['p']==d['p'] and r['dofs']==dofs)
        add(d,'oscillatory',geometry,n=source['n'],method=f"asm_d{d['configuration']['polynomial_degree']}_tuned")
        out[-1].update(dofs=dofs,triangles=source['triangles'])
    for r in out:
        if is_local(r) or r['source_candidate'].startswith('pp_'):assert isinstance(r['pp_degree'],int)
    groups={}
    for r in out:groups.setdefault((r['suite'],r['case'],r['geometry'],r['n'],r['p'],r['dofs']),r)
    for r in groups.values():
        for metric in ('fresh','hot'):
            direct=lu.lookup(r,r['suite'],metric)
            out.append(dict(r,method='pardiso_lu',source_candidate='pardiso_lu',status='passed',
                            fresh=direct['fresh'] if metric=='fresh' else None,
                            hot=direct['hot'] if metric=='hot' else None,setup=direct['setup'],
                            iterations=None,fresh_iterations=None,pp_degree=None,policy=None,
                            threads=direct['threads'],system_id=direct['system_id']))
    return out

def is_local(r):return r['method'].startswith(('asm','ASM+PP','BJ+PP'))
def family(r):return 'BJ+PP' if r['method'].startswith('BJ+PP') else 'ASM+PP'
def is_asm(r):return is_local(r) and family(r)=='ASM+PP'
def is_bj(r):return is_local(r) and family(r)=='BJ+PP'
def is_amg(r):
    m=r['method']; return m.startswith(('AMG/','amg_')) or (m.startswith('amgx_') and '_amg' in m)
def outer(r):return 'BiCGSTAB' if 'bicgstab' in r['method'].lower() else 'FGMRES'
def is_native(r):return r['method'].startswith('native_hp')
def is_lu(r):return r['method']=='pardiso_lu'

def is_other(r):
    m=r['method'].lower(); return 'dilu--' in m or m.startswith('dilu_') or m.startswith('pp')
def best(rows,pred,metric):
    a=[r for r in rows if pred(r) and r['status']=='passed' and r[metric] is not None]
    return min(a,key=lambda r:r[metric]) if a else None

def iteration_count(row,metric):
    """Use the solve associated with the selected time, before rounding for display."""
    return row['fresh_iterations' if metric=='fresh' else 'iterations']

def systems(records,suite,case,geometry='square',sweep=None):
    rows=[r for r in records if r['suite']==suite and r['case']==case and r['geometry']==geometry]
    if sweep=='h':rows=[r for r in rows if r['p']==6]
    if sweep=='p':rows=[r for r in rows if r['n']==64]
    groups={}
    for r in rows:groups.setdefault((r['n'],r['p'],r['dofs']),[]).append(r)
    key=(lambda k:k[2]) if sweep=='h' else (lambda k:k[1])
    return [groups[k] for k in sorted(groups,key=key)]

HEAT=[
 ('oscillatory','cellular7_weak','square','Low diffusion · square'),
 ('oscillatory','cellular7_weak','annulus','Low diffusion · annulus'),
 ('oscillatory','cellular7_directional','square','Directional · square'),
 ('oscillatory','cellular7_directional','annulus','Directional · annulus'),
 ('oscillatory','cellular7_high','square','High diffusion · square'),
 ('oscillatory','cellular7_high','annulus','High diffusion · annulus'),
 ('smooth','anisotropic','square','Smooth anisotropic'),
 ('focused_scaling','Transport dominated','square','Smooth transport'),
 ('focused_scaling','Variable velocity','square','Variable velocity'),
 ('focused_scaling','High diffusion','square','Smooth high diffusion')]

OVERVIEW_FAMILIES=(('asm',ASM_LABEL,is_asm),('bj',BJ_LABEL,is_bj),('amg','AMG',is_amg),('native',PMG,is_native),('lu','P-LU$_n$',is_lu))


def overview_cells(records,sweep):
    cells=[]
    for index,(suite,case,geometry,title) in enumerate(HEAT):
        for column,group in enumerate(systems(records,suite,case,geometry,sweep)):
            for metric in ('fresh','hot'):
                passing=[best(group,pred,metric) for _,_,pred in OVERVIEW_FAMILIES]
                minimum=min(r[metric] for r in passing if r)
                for family_id,_,predicate in OVERVIEW_FAMILIES:
                    attempted=[r for r in group if predicate(r)];chosen=best(group,predicate,metric)
                    degree=chosen['pp_degree'] if chosen else None
                    if family_id in ('asm','bj'):
                        degrees=sorted({r['pp_degree'] for r in attempted})
                        config=f'PP({degree})' if chosen else ', '.join(f'PP({d})' for d in degrees)
                    elif chosen and family_id=='lu':config=lu.alias(chosen['threads'])
                    elif chosen and family_id=='native':config='S' if chosen['policy']=='standard' else 'R'
                    elif chosen:config='F' if outer(chosen)=='FGMRES' else 'B'
                    else:config=''
                    cells.append(dict(sweep=sweep,suite=suite,case=case,geometry=geometry,case_index=index,
                                      column=column,n=group[0]['n'],p=group[0]['p'],triangles=group[0]['triangles'],
                                      dofs=group[0]['dofs'],metric=metric,family=family_id,
                                      status='passed' if chosen else 'NC' if attempted else 'not_tested',
                                      time_ms=chosen[metric] if chosen else None,
                                      iterations=iteration_count(chosen,metric) if chosen else None,
                                      slowdown=chosen[metric]/minimum if chosen else None,
                                      winner=bool(chosen and chosen[metric]==minimum),configuration=config,
                                      pp_degree=degree,method=chosen['method'] if chosen else None,
                                      policy=chosen['policy'] if chosen else None))
    return cells


def comparison_overview(records,sweep):
    """Time, corresponding iteration count and outcome for every solver family."""
    cells=overview_cells(records,sweep)
    columns=4 if sweep=='h' else 5
    fig,axes=plt.subplots(1,2,figsize=(7.15,9.15),sharey=True)
    gap=.45;stride=len(OVERVIEW_FAMILIES)+gap;height=len(HEAT)*stride-gap
    cmap=plt.get_cmap('Blues');norm=Normalize(0,6)
    for ax,metric,title in zip(axes,('fresh','hot'),('Fresh · setup + first solve','Reused · second solve')):
        ax.set_xlim(-.5,columns-.5);ax.set_ylim(height-.5,-.5)
        ax.set_title(title,fontsize=8.3,pad=10)
        ax.set_xticks(range(columns),['1','2','3','4'] if sweep=='h' else ['1','2','3','4','6'])
        ax.xaxis.tick_top();ax.tick_params(axis='both',length=0)
        ax.set_yticks([i*stride+j for i in range(len(HEAT)) for j in range(len(OVERVIEW_FAMILIES))],
                      [label for _ in HEAT for _,label,_ in OVERVIEW_FAMILIES],fontsize=6.6)
        for spine in ax.spines.values():spine.set_visible(False)
        for cell in (c for c in cells if c['metric']==metric):
            x=cell['column'];y=cell['case_index']*stride+[v[0] for v in OVERVIEW_FAMILIES].index(cell['family'])
            status=cell['status'];passing=status=='passed'
            shade=min(6,np.log2(cell['slowdown'])) if passing else None
            face=cmap(norm(shade)) if passing else '#fbe9e7' if status=='NC' else '#f2f4f7'
            ax.add_patch(Rectangle((x-.49,y-.48),.98,.96,facecolor=face,
                                   edgecolor='#14532d' if cell['winner'] else 'white',
                                   linewidth=1.1 if cell['winner'] else .6))
            color='white' if passing and shade>3.5 else '#912018' if status=='NC' else '#17212e'
            value=(f"{cell['time_ms']:.0f}" if cell['time_ms']>=1000 else f"{cell['time_ms']:.1f}") if passing else 'NC' if status=='NC' else '—'
            if passing and cell['iterations'] is not None:value+=f" ({cell['iterations']:.0f})"
            offset=-.225 if cell['configuration'] else 0
            ax.text(x,y+offset,value,ha='center',va='center',fontsize=6.1,
                    fontweight='bold' if cell['winner'] else 'normal',color=color)
            if cell['configuration']:ax.text(x,y+.265,cell['configuration'],ha='center',va='center',fontsize=5.2,color=color)
    axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(left=.295,right=.993,top=.918,bottom=.098,wspace=.075)
    case_labels=['Oscillatory low diffusion\nSquare','Oscillatory low diffusion\nAnnulus',
                 'Oscillatory directional\nSquare','Oscillatory directional\nAnnulus',
                 'Oscillatory high diffusion\nSquare','Oscillatory high diffusion\nAnnulus',
                 'Smooth anisotropy\nSquare','Smooth transport\nSquare',
                 'Variable velocity\nSquare','Smooth high diffusion\nSquare']
    for i,label in enumerate(case_labels):
        y=axes[0].get_position().y1-(i*stride+2)/height*axes[0].get_position().height
        fig.text(.007,y,label,fontsize=7.0,va='center',linespacing=1.3)
    heading=(r'Mesh refinement at $p_{\mathrm{FE}}=6$ · columns are refinement levels' if sweep=='h'
             else r'Degree sweep · columns are finite-element degree $p_{\mathrm{FE}}$')
    fig.suptitle(heading,fontsize=9.3,y=.992)
    if sweep=='h':
        mesh='Triangles by level: square 2,048 / 8,192 / 32,768 / 99,458; annulus 2,043 / 8,039 / 33,174 / 99,984.'
    else:mesh='Fixed mesh: square 8,192 triangles; annulus 8,039 triangles.'
    fig.text(.5,.96,mesh,ha='center',fontsize=6.6)
    fig.text(.5,.072,'Cell: time in ms (k iterations); selected configuration below. Green border: fastest passing family.',ha='center',fontsize=6.5)
    fig.text(.5,.052,'PP(d): degree; F/B: Krylov; S/R: policy; P-LU: subscript = CPU threads, no Krylov count.',ha='center',fontsize=6.4)
    cbax=fig.add_axes([.32,.019,.55,.011]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal')
    cb.set_ticks([0,1,2,3,4,5,6],labels=['1×','2×','4×','8×','16×','32×','≥64×']);cb.ax.tick_params(labelsize=6,length=2)
    fig.text(.01,.019,'Time / fastest passing time',fontsize=6.5,va='center')
    save_publication_figure(fig,FIGURES/('solver_mesh_comparison' if sweep=='h' else 'solver_degree_comparison'));plt.close(fig)
    return cells


def plot_envelope(ax,groups,metric,show_bj=False,callouts=False,xmode='h'):
    x=[g[0]['p'] if xmode=='p' else g[0]['dofs'] for g in groups]
    lu.curve(ax,[g[0] for g in groups],groups[0][0]['suite'],metric,'p' if xmode=='p' else 'dofs')
    styles=[(ASM_LABEL,lambda r:family(r)=='ASM+PP','#0072B2','s','-')]
    if show_bj:styles.append((BJ_LABEL,lambda r:family(r)=='BJ+PP','#E69F00','^','--'))
    for label,pred,color,marker,line in styles:
        sel=[best(g,lambda r,p=pred:is_local(r) and p(r),metric) for g in groups]
        ax.plot(x,[r[metric] if r else np.nan for r in sel],color=color,marker=marker,linestyle=line,label=label,markeredgecolor='white',markeredgewidth=.5)
    for i,(label,pred,color,marker,line) in enumerate(styles):
        selected=[best(g,lambda r,p=pred:is_local(r) and p(r),metric) for g in groups]
        passing=[r for r in selected if r]
        if not passing:continue
        base=passing[0]['pp_degree']
        ax.text(.025,.975-i*.085,polynomial_label(passing[0]),transform=ax.transAxes,
                fontsize=6.3,color=color,va='top',bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
        for xx,r in zip(x,selected):
            if r and r['pp_degree']!=base:
                ax.annotate(polynomial_label(r),(xx,r[metric]),xytext=(-3,-12),textcoords='offset points',ha='right',fontsize=6.2,color=color)
    chosen_native=[best(g,is_native,metric) for g in groups]
    ax.plot(x,[r[metric] if r else np.nan for r in chosen_native],color='#7F3C8D',marker='P',linestyle='-.',label=PMG)
    for xx,r,g in zip(x,chosen_native,groups):
        if r is None and any(is_native(v) for v in g):
            ax.plot(xx,1.09,marker='x',color='#7F3C8D',transform=ax.get_xaxis_transform(),clip_on=False,linestyle='none',markersize=6)
    chosen=[best(g,is_amg,metric) for g in groups]
    ax.plot(x,[r[metric] if r else np.nan for r in chosen],color='#D55E00',label='AMG')
    for xx,r,g in zip(x,chosen,groups):
        if r:ax.plot(xx,r[metric],marker='D' if outer(r)=='BiCGSTAB' else 'o',color='#D55E00',markeredgecolor='white',markeredgewidth=.5)
        elif any(is_amg(v) for v in g):ax.plot(xx,1.035,marker='x',color='#D55E00',transform=ax.get_xaxis_transform(),clip_on=False,linestyle='none',markersize=6)
        if callouts:
            core=[v for v in g if (is_local(v) or is_amg(v) or is_native(v) or is_lu(v)) and v['status']=='passed' and v[metric] is not None];other=best(g,is_other,metric)
            if other and core and other[metric]<min(v[metric] for v in core):
                label='$hp$' if 'hp' in other['method'].lower() else ('DILU' if 'dilu' in other['method'].lower() else polynomial_label(other))
                ax.plot(xx,other[metric],marker='*',color='#555555',markersize=7,linestyle='none');ax.annotate(label,(xx,other[metric]),xytext=(2,3),textcoords='offset points',fontsize=6.3,color='#555555')
    ax.set_xscale('log');ax.set_yscale('log');ax.grid(which='major',alpha=.75)

def oscillatory_figure(records):
    cases=('cellular7_high','cellular7_weak','cellular7_directional');titles=('High diffusion','Low diffusion','Directional anisotropy')
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.1),sharey='row')
    for i,geom in enumerate(('square','annulus')):
        for j,(case,title) in enumerate(zip(cases,titles)):
            g=systems(records,'oscillatory',case,geom,'h');plot_envelope(axs[i,j],g,'hot',callouts=True);axs[i,j].set_title(title if i==0 else '',pad=22);axs[i,j].set_xticks([v[0]['dofs'] for v in g],['L1','L2','L3','L4']);axs[i,j].minorticks_off()
            if i==1:axs[i,j].set_xlabel('Refinement level')
        axs[i,0].set_ylabel(('Square\n' if i==0 else 'Annulus\n')+'Reused time (ms)')
    handles=[Line2D([],[],color='#0072B2',marker='s',label=ASM_LABEL),Line2D([],[],color='#D55E00',marker='o',label='AMG · FGMRES'),Line2D([],[],color='#D55E00',marker='D',label='AMG · BiCGSTAB'),Line2D([],[],color='#D55E00',marker='x',linestyle='none',label='AMG: NC'),Line2D([],[],color='#7F3C8D',marker='P',linestyle='-.',label=PMG),Line2D([],[],color='#7F3C8D',marker='x',linestyle='none',label=PMG+': NC')]
    handles.append(lu.handle())
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,-.01));fig.subplots_adjust(left=.105,right=.99,top=.88,bottom=.27,wspace=.11,hspace=.29);save_publication_figure(fig,FIGURES/'oscillatory_local_advantage');plt.close(fig)

def focused_figure(records,sweep,stem):
    cases=('Transport dominated','Variable velocity','High diffusion');fig,axs=plt.subplots(2,3,figsize=(7.1,4.15),sharey='row')
    for i,metric in enumerate(('fresh','hot')):
        for j,case in enumerate(cases):
            g=systems(records,'focused_scaling',case,'square',sweep);plot_envelope(axs[i,j],g,metric,show_bj=True,callouts=True,xmode=sweep);axs[i,j].set_title(case if i==0 else '',pad=7)
            if sweep=='h':axs[i,j].set_xticks([v[0]['dofs'] for v in g],['L1','L2','L3','L4']);axs[i,j].minorticks_off()
            else:axs[i,j].set_xscale('linear');axs[i,j].set_xticks([1,2,3,4,6])
            if i==1:axs[i,j].set_xlabel('Refinement level' if sweep=='h' else r'$p_{\mathrm{FE}}$ (8,192 triangles)')
        axs[i,0].set_ylabel('Fresh time (ms)' if metric=='fresh' else 'Reused time (ms)')
    handles=[Line2D([],[],color='#0072B2',marker='s',label=ASM_LABEL),Line2D([],[],color='#E69F00',marker='^',linestyle='--',label=BJ_LABEL),Line2D([],[],color='#D55E00',marker='o',label='AMG · FGMRES'),Line2D([],[],color='#D55E00',marker='D',label='AMG · BiCGSTAB'),Line2D([],[],color='#7F3C8D',marker='P',linestyle='-.',label=PMG),Line2D([],[],color='#555555',marker='*',linestyle='none',label='Additional fastest solver')]
    handles.append(lu.handle())
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,-.01));fig.subplots_adjust(left=.10,right=.99,top=.93,bottom=.27,wspace=.11,hspace=.17);save_publication_figure(fig,FIGURES/stem);plt.close(fig)

def fmt(v):return '--' if v is None else f'{v:.1f}'
ROW=r'\\'
def tables(records):
    def timecell(row,metric,scale=1):return fmt(row[metric]/scale) if row else 'NC'
    lines=[r'\begin{tabular}{llrrrrrrll}',r'\toprule',
           r'Geometry & Class & \multicolumn{2}{c}{ASM+PP} & \multicolumn{2}{c}{AMG} & \multicolumn{2}{c}{$p$MG--AMG} & \multicolumn{2}{c}{CPU LU} \\',
           r' & & Fresh & Reused & Fresh & Reused & Fresh & Reused & Fresh & Reused \\',r'\midrule']
    for geom in ('square','annulus'):
        for case,short in (('cellular7_weak','Low'),('cellular7_directional','Directional')):
            g=systems(records,'oscillatory',case,geom,'h')[-1]
            values=[timecell(best(g,pred,metric),metric) for pred in (is_local,is_amg,is_native) for metric in ('fresh','hot')]
            values += [lu.cell(g[0],'oscillatory',metric) for metric in ('fresh','hot')]
            lines.append(f'{geom.title()} & {short} & '+' & '.join(values)+' '+ROW)
        if geom=='square':lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}'];(SYNTH/'hard_cases_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'Geometry & Solver & $k$ & Apply & Setup & Fresh & Reused \\',
           r' & & & (ms) & \multicolumn{3}{c}{time (s)} \\',r'\midrule']
    for gi,geom in enumerate(('square','annulus')):
        g=systems(records,'oscillatory','cellular7_directional',geom,'h')[-1]
        asm=best(g,lambda r:is_local(r) and 'tuned' in r['method'],'hot')
        natives=[best(g,lambda r:r['method']=='native_hp_'+policy,'hot') for policy in ('standard','robust')]
        chosen=[asm]+[r for r in natives if r]
        df=lu.lookup(g[0],'oscillatory','fresh');dh=lu.lookup(g[0],'oscillatory','hot')
        minima={k:min([r[k] for r in chosen]+[(dh if k=='hot' else df)[k]]) for k in ('setup','fresh','hot')}
        for i,r in enumerate(chosen):
            label=polynomial_label(r) if i==0 else (PMG_TEX+' '+r['method'].replace('native_hp_',''))
            cells=[]
            for k in ('setup','fresh','hot'):
                value=f'{r[k]/1000:.3f}';cells.append(r'\textbf{'+value+'}' if r[k]==minima[k] else value)
            audit=native_completion.CAMPAIGN.parent/'oscillatory/asm_optimality_audit'
            if i==0:
                profile=json.loads((audit/('square_asm_d20_raw_final_profile.json' if geom=='square' else 'annulus_asm_d29_raw_final_profile.json')).read_text())
            elif r['method']=='native_hp_robust':
                profile=json.loads((audit/('native_hp_robust_square_profile.json' if geom=='square' else 'native_hp_robust_profile.json')).read_text())
            else:
                profile=next(p['record'] for p in native_completion.read('profiles.json') if p['system_id']==r['system_id'] and p['status']=='passed')
            application=profile['complete_preconditioner']['mean_ms']
            lines.append(f"{geom.title() if i==0 else ''} & {label} & {r['iterations']:.0f} & {application:.2f} & "+' & '.join(cells)+' '+ROW)
        cpu_cells=[]
        for k in ('setup','fresh','hot'):
            direct=dh if k=='hot' else df
            value=lu.alias(direct['threads'],tex=True)+f" {direct[k]/1000:.3f}"
            cpu_cells.append(r'\textbf{'+value+'}' if direct[k]==minima[k] else value)
        lines.append(' & CPU LU & -- & -- & '+' & '.join(cpu_cells)+' '+ROW)
        if gi==0:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}'];(SYNTH/'directional_tuned_table.tex').write_text('\n'.join(lines)+'\n')
    # The detailed report keeps its existing iteration-column heading.
    (REPORT/'native_completion/directional_tuned_table.tex').write_text(
        ('\n'.join(lines)+'\n').replace('Solver & $k$ &','Solver & Iter. &'))
    lines=[r'\begin{tabular}{llrrrrrrrrrr}',r'\toprule',r'Problem & $p_{\mathrm{FE}}$ & \multicolumn{2}{c}{ASM+PP} & \multicolumn{2}{c}{BJ+PP} & \multicolumn{2}{c}{AMG} & \multicolumn{2}{c}{$p$MG--AMG} & \multicolumn{2}{c}{CPU LU} \\',r' & & Fresh & Reused & Fresh & Reused & Fresh & Reused & Fresh & Reused & Fresh & Reused \\',r'\midrule']
    for ci,case in enumerate(('Transport dominated','Variable velocity','High diffusion')):
        for g in sorted(systems(records,'focused_million_dof',case),key=lambda v:v[0]['p']):
            predicates=(lambda r:is_local(r) and family(r)=='ASM+PP',lambda r:is_local(r) and family(r)=='BJ+PP',is_amg,is_native,is_lu)
            minima={metric:min(r[metric] for r in g if r['status']=='passed' and r[metric] is not None) for metric in ('fresh','hot')}
            cells=[]
            for pred in predicates:
                for metric in ('fresh','hot'):
                    r=best(g,pred,metric);value=timecell(r,metric)
                    if r:
                        config=(lu.alias(r['threads'],tex=True) if is_lu(r) else f"PP({r['pp_degree']})" if is_local(r) else
                                ('S' if r['policy']=='standard' else 'R') if is_native(r) else
                                'B' if outer(r)=='BiCGSTAB' else 'F')
                        detail=config if is_lu(r) else config+f", $k={iteration_count(r,metric):.0f}$"
                        value=r'\shortstack{'+value+r'\\ {\tiny '+detail+'}}'
                    elif r is None and pred in predicates[:2]:
                        degrees=sorted({v['pp_degree'] for v in g if pred(v)})
                        if degrees:value=r'\shortstack{'+value+r'\\ {\tiny '+', '.join(f'PP({d})' for d in degrees)+'}}'
                    cells.append(r'\textbf{'+value+'}' if r and r[metric]==minima[metric] else value)
            name=TITLES[case] if g[0]['p']==3 else ''
            lines.append(f"{name} & {g[0]['p']} & "+' & '.join(cells)+' '+ROW)
        if ci<2:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}'];(SYNTH/'million_table.tex').write_text('\n'.join(lines)+'\n')


def anisotropic_figure(records):
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.55))
    for ax,sweep in zip(axs,('h','p')):
        groups=systems(records,'smooth','anisotropic','square',sweep)
        plot_envelope(ax,groups,'hot',xmode=sweep)
        if sweep=='h':
            ax.set_xticks([g[0]['dofs'] for g in groups],['21k','85k','342k','1.04M']);ax.minorticks_off();ax.set_xlabel(r'Trace unknowns ($p_{\mathrm{FE}}=6$)')
        else:ax.set_xscale('linear');ax.set_xticks([1,2,3,4,6]);ax.set_xlabel(r'$p_{\mathrm{FE}}$ (8,192 triangles)')
        ax.set_title('Mesh refinement' if sweep=='h' else 'Degree sweep')
    axs[0].set_ylabel('Reused time (ms)')
    handles=[Line2D([],[],color='#0072B2',marker='s',label=ASM_LABEL),Line2D([],[],color='#D55E00',marker='o',label='AMG'),Line2D([],[],color='#7F3C8D',marker='P',linestyle='-.',label=PMG)]
    handles.append(lu.handle())
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(left=.1,right=.99,top=.85,bottom=.31,wspace=.25)
    save_publication_figure(fig,FIGURES/'smooth_anisotropic');plt.close(fig)


def select(records):
    fields=('suite','geometry','case','n','p','dofs','metric','group','family','method','status','time_ms','iterations','pp_degree','triangles','policy');out=[];groups={}
    for r in records:groups.setdefault((r['suite'],r['geometry'],r['case'],r['n'],r['p'],r['dofs']),[]).append(r)
    for key,g in groups.items():
        for metric in ('fresh','hot'):
            for label,pred in (('asm',is_asm),('bj',is_bj),('amg',is_amg),('native',is_native),('lu',is_lu)):
                chosen=best(g,pred,metric)
                vals=(*key,metric,label,(polynomial_label(chosen) if label in ('asm','bj') else (chosen['method'].replace('native_hp_','') if label=='native' else outer(chosen))) if chosen else '',chosen['method'] if chosen else '',chosen['status'] if chosen else 'NC',chosen[metric] if chosen else '',iteration_count(chosen,metric) if chosen else '',chosen['pp_degree'] if chosen else '',g[0]['triangles'],chosen['policy'] if chosen else '')
                if chosen or any(pred(r) for r in g):
                    record=dict(zip(fields,vals))
                    if label=='lu' and chosen:record['family']=lu.alias(chosen['threads'])
                    out.append(record)
    with (SYNTH/'selected_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    return out

SECTION=Path(__file__).with_name('adr_synthesis_section.tex.in').read_text()

MAIN=r'''% Block-preconditioner synthesis with confirmed multithreaded PyPardiso LU baseline.
\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[margin=18mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,microtype}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{7pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{0.94}
\renewcommand{\bottomfraction}{0.88}
\renewcommand{\textfraction}{0.05}
\newcommand{\ADRResultsPath}{.}
\newcommand{\ADRResultsFigurePath}{../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/figures}
\newcommand{\PLU}[1]{\mathrm{P\!-\!LU}_{#1}}
\begin{document}
\input{section_synthesis.tex}
\clearpage
\input{\ADRResultsPath/closed_loop_stress/section.tex}
\end{document}
'''

def validate(records,selected):
    counts={s:sum(r['suite']==s and not is_lu(r) for r in records) for s in ('smooth','focused_million_dof','focused_scaling','oscillatory')}
    direction=[r for r in records if r['suite']=='oscillatory' and r['case']=='cellular7_directional' and is_amg(r)]
    assert len(direction)==32 and all(r['status']!='passed' for r in direction)
    assert all(r['status']=='passed' for r in selected if r['time_ms']!='')
    native=[r for r in records if r['suite']=='oscillatory' and r['case']=='cellular7_directional' and is_native(r)]
    tuned=[r for r in records if r['suite']=='oscillatory' and 'tuned' in r['method']]
    assert len({(r['geometry'],r['n'],r['p']) for r in native if r['status']=='passed'})==16
    assert len(tuned)==2 and all(r['status']=='passed' for r in tuned)
    assert all(any(is_native(r) for r in g) for suite,case,geom,_ in HEAT for g in systems(records,suite,case,geom))
    data={'status':'passed','source_counts':counts,'selected_rows':len(selected),
          'lu_baseline':{'confirmed_systems':len({r['system_id'] for r in lu.records()}),
                         'overview_cells':sum(c['family']=='lu' for sweep in ('h','p') for c in overview_cells(records,sweep)),
                         'selection':'Independent confirmations at pilot-selected threads; fresh and reused selected separately.'},
          'directional_amg_attempted':32,'directional_amg_passed':0,
          'directional_native_hp_passed_systems':16,'directional_tuned_asm_pp_passed':len(tuned),
          'native_completion':native_completion.read('completion.json'),
          'selection':'Metric-specific passing minima; native policy curves cover every system. Historic native measurements retain their original suite, without choosing faster duplicate trials.'}
    (SYNTH/'validation.json').write_text(json.dumps(data,indent=2)+'\n')

def bundle():
    section=(REPORT/'section_synthesis.tex').read_text().replace(r'\providecommand{\ADRResultsPath}{docs/research/solver_studies/adr_scaling_2026_09_17}',r'\providecommand{\ADRResultsPath}{.}').replace(r'\providecommand{\ADRResultsFigurePath}{run_outputs/solver_studies/adr_scaling_2026_09_17/figures}',r'\providecommand{\ADRResultsFigurePath}{figures}')
    main=(REPORT/'main_synthesis.tex').read_text().replace('../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/figures','figures')
    with zipfile.ZipFile(BUNDLE,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('section_synthesis.tex',section);z.writestr('main_synthesis.tex',main)
        for p in sorted(SYNTH.iterdir()):
            if p.suffix in ('.tex','.csv','.json'):z.write(p,'synthesis/'+p.name)
        probe=ROOT/'run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/strong_probe/summary.json'
        z.write(probe,'synthesis/directional_strong_probe_summary.json')
        audit=ROOT/'run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/asm_optimality_audit/summary.json'
        z.write(audit,'synthesis/directional_asm_optimality_audit.json')
        for p in sorted((REPORT/'native_completion').glob('*')):
            if p.suffix in ('.json','.csv'):z.write(p,'native_completion/'+p.name)
        for p in sorted(FIGURES.iterdir()):
            if p.suffix in ('.pdf','.png','.svg'):z.write(p,'figures/synthesis/'+p.name)
        for p in sorted((REPORT/'closed_loop_stress').glob('*')):
            if p.suffix in ('.tex','.md'):z.write(p,'closed_loop_stress/'+p.name)
        for p in sorted((REPORT/'pardiso_lu').glob('*.tex')):
            z.write(p,'pardiso_lu/'+p.name)
        for name in ('confirmed_timings.csv','comparisons.csv'):
            z.write(lu.CAMPAIGN/name,'pardiso_lu/'+name)
        for p in sorted((FIGURES.parent/'closed_loop_stress').glob('*')):
            if p.suffix in ('.pdf','.png','.svg','.json'):z.write(p,'figures/closed_loop_stress/'+p.name)

def main():
    SYNTH.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True);records=normalize();selected=select(records);validate(records,selected);tables(records)
    with publication_style(font_size=8.2):
        cells=comparison_overview(records,'h')+comparison_overview(records,'p');oscillatory_figure(records);focused_figure(records,'h','focused_h_crossover');focused_figure(records,'p','focused_p_crossover');anisotropic_figure(records)
    with (SYNTH/'overview_cells.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=cells[0].keys());writer.writeheader();writer.writerows(cells)
    (REPORT/'section_synthesis.tex').write_text(SECTION);(REPORT/'main_synthesis.tex').write_text(MAIN)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FIGURES.iterdir()) if p.is_file()};(SYNTH/'figure_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n');bundle()
    print(json.dumps({'validation':json.loads((SYNTH/'validation.json').read_text()),'bundle':str(BUNDLE),'bundle_bytes':BUNDLE.stat().st_size},indent=2))
if __name__=='__main__':main()
