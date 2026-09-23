#!/usr/bin/env python3
"""Render the archived initial oscillatory stress study inside the combined ADR study."""
from __future__ import annotations
import argparse,csv,hashlib,json,runpy,statistics,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from hdgfem.io.figures import publication_style,save_publication_figure
import adr_native_completion as native_completion
import adr_lu_baseline as lu
from adr_report_labels import PMG, PMG_TEX, polynomial_label, prose

BRANCH=ROOT.parent/'hdgfem-gmres'
NAMES=['adr_oscillatory_screen_20260918','adr_oscillatory_square_p6_20260918',
       'adr_oscillatory_star_h0.04_p6_20260918','adr_oscillatory_star_h0.02_p6_20260918']
CASES=['advection_dominated','cellular7_weak','cellular7_anisotropic']
METHODS=['asm_d24','amg_fgmres','amg_pbicgstab','dilu_pbicgstab','native_hp_best']
LABELS=['ASM+PP(24) / GMRES','AMG / FGMRES','AMG / BiCGSTAB','DILU / BiCGSTAB',PMG]


def read(path):
    """Read recorded measurements without running numerical solvers."""
    return json.loads(Path(path).read_text())


def stats(row):
    """Summarize measured second solves; never rank failed or partial runs."""
    if row['status']!='passed':return None
    hot=[trial['solves'][1] for trial in row['samples']]
    return dict(mean_ms=statistics.mean(s['solve_ms'] for s in hot),
        min_ms=min(s['solve_ms'] for s in hot),max_ms=max(s['solve_ms'] for s in hot),
        iterations=statistics.mean(s['iterations'] for s in hot),setup_ms=row['setup_median_ms'],
        fresh_ms=row['fresh_setup_solve_median_ms'],amortized_ms=row['amortized_median_ms'],
        residual=max(s['true_relative_residual'] for t in row['samples']+row['warmups'] for s in t['solves']),
        l2=max(s['l2_error'] for t in row['samples'] for s in t['solves']))


def fields_figure(figures):
    """Show the actual mesh and the analytic fields on the selected annular domain."""
    factory=runpy.run_path(str(BRANCH/'scripts/oscillatory_adr_cases.py'))['make_case']
    kw,exact=factory('cellular7_anisotropic')
    mesh=np.load(BRANCH/'run_logs/adr_oscillatory_geometry_20260918/star_h0.04.npz')
    nodes,tris=mesh['node_coords'],mesh['triangles']
    axis=np.linspace(-1.4,1.4,501);x,y=np.meshgrid(axis,axis)
    theta=np.arctan2(y,x);radius=np.hypot(x,y)
    outside=(radius<.58)|(radius>1+.35*np.cos(5*theta))
    u=np.ma.masked_where(outside,exact(x,y))
    bx=np.ma.masked_where(outside,kw['beta'][0](x,y));by=np.ma.masked_where(outside,kw['beta'][1](x,y))
    fig,axs=plt.subplots(1,3,figsize=(6.65,2.28))
    axs[0].triplot(nodes[:,0],nodes[:,1],tris,color='#9CA3AF',lw=.10)
    im=axs[1].pcolormesh(x,y,u,cmap='RdBu_r',vmin=-1.35,vmax=1.35,rasterized=True,shading='auto')
    fig.colorbar(im,ax=axs[1],shrink=.72,pad=.035,fraction=.05,ticks=[-1,0,1])
    speed=np.ma.sqrt(bx*bx+by*by)
    im=axs[2].pcolormesh(x,y,speed,cmap='viridis',rasterized=True,shading='auto')
    axs[2].streamplot(axis,axis,bx,by,density=2.4,color='white',linewidth=.25,arrowsize=.35)
    fig.colorbar(im,ax=axs[2],shrink=.72,pad=.035,fraction=.05,ticks=[0,2,4])
    t=np.linspace(0,2*np.pi,1001);r=1+.35*np.cos(5*t)
    for ax,title in zip(axs,['Nonconvex annulus',r'Exact solution $u_\star$',r'Oscillatory velocity $|\beta|$']):
        ax.plot(r*np.cos(t),r*np.sin(t),color='#344054',lw=.55)
        ax.plot(.58*np.cos(t),.58*np.sin(t),color='#344054',lw=.55)
        ax.set_aspect('equal');ax.set_xlim(-1.4,1.4);ax.set_ylim(-1.4,1.4)
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_title(title,pad=7)
        ax.set_xlabel('$x$')
    axs[0].set_ylabel('$y$')
    fig.subplots_adjust(left=.05,right=.99,bottom=.18,top=.88,wspace=.35)
    save_publication_figure(fig,figures/'geometry_fields');plt.close(fig)


def comparison_figure(campaigns,figures):
    """Compare frozen solvers with visibly unranked nonconvergence cells."""
    values=np.full((6,9),np.nan);entries={}
    for g,campaign in enumerate(campaigns[1:]):
        for j,case in enumerate(CASES):
            for i,method in enumerate(METHODS):
                r=next(r for r in campaign['rows'] if r['case']==case and r['candidate']==method)
                v=stats(r);entries[i,3*g+j]=v
                if v is not None:values[i,3*g+j]=v['mean_ms']
            origin=next(o for d in lu.records() for o in d['origins'] if o['suite']=='initial' and o['campaign']==campaign['name'] and o['case']==case)
            direct=lu.lookup(origin,'initial','hot',campaign['name'])
            entries[5,3*g+j]=dict(mean_ms=direct['hot'],threads=direct['threads'])
            values[5,3*g+j]=direct['hot']
    fig,ax=plt.subplots(figsize=(6.65,3.25))
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#E5E7EB')
    im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,norm=LogNorm(10,10000))
    for (i,j),v in entries.items():
        message='NC' if v is None else (lu.alias(v['threads'])+'\n'+f"{v['mean_ms']:.0f} ms" if 'threads' in v else f"{v['iterations']:.0f} it.\n{v['mean_ms']:.0f} ms")
        ax.text(j,i,message,ha='center',va='center',fontsize=7.3,color='white' if v and v['mean_ms']>450 else '#101828')
    ax.set_yticks(range(6),LABELS+['P-LU$_n$'],fontsize=8)
    ax.set_xticks(range(9),['Base','Low iso.','Aniso.']*3,fontsize=7.5)
    ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True)
    ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='both',length=0)
    for x,label in [(1,'Square · 8,192 triangles'),(4,'Star · 6,114 triangles'),(7,'Star · 22,825 triangles')]:
        ax.text(x,1.045,label,transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
    for x in (2.5,5.5):ax.axvline(x,color='white',lw=5)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.10,pad=.25,aspect=45,ticks=[10,100,1000,10000])
    cb.set_label('Mean hot solve time (ms); NC = failed acceptance contract',fontsize=8)
    cb.set_ticklabels(['10','100','1,000','10,000'])
    fig.subplots_adjust(left=.19,right=.985,top=.85,bottom=.10)
    save_publication_figure(fig,figures/'solver_comparison');plt.close(fig)



def tex_scientific(value):
    """Format small errors as legible scientific notation in a TeX table."""
    mantissa, exponent = f'{value:.2e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'


def portable_bundle(out, figures):
    """Package the report, supporting measurements and exact geometric inputs."""
    destination = figures.parent / 'adr_oscillatory_bundle.zip'
    section = (out/'section.tex').read_text().replace(
        r'\providecommand{\ADRStressPath}{docs/research/solver_studies/adr_scaling_2026_09_17/oscillatory/initial}',
        r'\providecommand{\ADRStressPath}{.}').replace(
        r'\providecommand{\ADRStressFigurePath}{run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/initial/figures}',
        r'\providecommand{\ADRStressFigurePath}{figures}')
    preview = (out/'preview.tex').read_text().replace(
        r'\newcommand{\ADRStressFigurePath}{../../../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/initial/figures}',
        r'\newcommand{\ADRStressFigurePath}{figures}')
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('section.tex', section)
        archive.writestr('preview.tex', preview)
        archive.writestr('README.txt',
            'Oscillatory ADR on a five-lobed annulus.\n\n'
            'Compile preview.tex with: latexmk -pdf -interaction=nonstopmode -halt-on-error preview.tex\n'
            'Typesetting was left to the user under the workspace no-compilation rule; pagination is unverified.\n'
            'For insertion load amsmath, graphicx, booktabs and hyperref. Set \\ADRStressPath\n'
            'to the unpacked directory and \\ADRStressFigurePath to its figures/ subdirectory.\n\n'
            'comparison.csv uses milliseconds; hot times are mean/min/max of second zero-guess solves.\n'
            'Setup, fresh and amortized times are medians. Failed runs have no ranked timing.\n'
            'results.data.json contains all samples, configs, binary hashes and validation.\n'
            'source_snapshot/ and meshes/ preserve the exact worker inputs. Re-executing these\n'
            'workers requires the matching HDGFEM checkouts and prebuilt AMGX/pyamgx; the\n'
            'JSON records original machine paths, not relocatable cache locations.\n'
            'source/ contains the additional checks and geometry generator.\n')
        for name in ('accuracy_table.tex','native_completion_table.tex','native_results.json','comparison.csv','results.data.json','verification.json','figure_hashes.json'):
            archive.write(out/name,name)
        for path in sorted(figures.iterdir()):
            if path.suffix in ('.pdf','.svg','.png'):archive.write(path,'figures/'+path.name)
        for name in NAMES:
            for path in sorted((BRANCH/'run_logs'/name/'source_snapshot').iterdir()):
                archive.write(path,'source_snapshot/'+name+'/'+path.name)
        for path in sorted((BRANCH/'run_logs/adr_oscillatory_geometry_20260918').glob('star_*')):
            if path.suffix in ('.npz','.json'):archive.write(path,'meshes/'+path.name)
        for name in ('audit_oscillatory_adr_study.py','check_oscillatory_adr_quadrature.py',
                     'diagnose_oscillatory_adr_matrix.py'):
            archive.write(BRANCH/'scripts'/name,'source/'+name)
        archive.write(ROOT/'scripts/advection_diffusion_reaction/make_oscillatory_geometry.py',
                      'source/make_oscillatory_geometry.py')
    return destination


def main():
    """Archive measured evidence and fill the compact report template."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/research/solver_studies/adr_scaling_2026_09_17/oscillatory/initial')
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    figures=ROOT/'run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/initial/figures';figures.mkdir(parents=True,exist_ok=True)
    campaigns=[];flat=[]
    for name in NAMES:
        folder=BRANCH/'run_logs'/name
        manifest=read(folder/'manifest.json');completion=read(folder/'completion.json')
        assert completion['attempted']==completion['scheduled']
        rows=read(folder/'summary.json')
        for r in rows:
            v=stats(r)
            if v:
                assert len(r['samples'])==manifest['arguments']['repeats'] and v['residual']<=1e-10
                assert all(s['passed'] for trial in r['samples']+r['warmups'] for s in trial['solves'])
            mesh=manifest.get('mesh')
            free_dofs=int(np.load(Path(r['matrix_cache'])/'system_rhs.npy',mmap_mode='r').size)
            failures=[s for t in r.get('warmups',[])+r.get('samples',[]) for s in t['solves'] if not s['passed']]
            flat.append(dict(campaign=name,case=r['case'],p=r['p'],
                triangles=mesh['triangles'] if mesh else 2*r['n']**2,trace_dofs=free_dofs,
                candidate=r['candidate'],status=r['status'],**(v or {}),
                failure_iterations=failures[-1]['iterations'] if failures else None,
                failure_residual=failures[-1]['true_relative_residual'] if failures else None))
        campaigns.append(dict(name=name,manifest=manifest,completion=completion,rows=rows,assemblies=read(folder/'assemblies.json'),
            matrix_diagnostics=read(folder/'matrix_diagnostics.json') if (folder/'matrix_diagnostics.json').exists() else None))
    checks=read(BRANCH/'run_logs/adr_oscillatory_verification_20260918/verification.json')
    assert checks['status']=='passed'
    (out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    archive=dict(campaigns=campaigns,verification=checks,
        geometry=[read(p) for p in sorted((BRANCH/'run_logs/adr_oscillatory_geometry_20260918').glob('star_*.json'))],
        sources=['https://arxiv.org/html/1401.6666v1','https://arxiv.org/abs/1710.09331'],
        provenance_note='Manufactured extension of the cited flow/domain families. Source snapshots, immutable meshes, caches and job logs remain in the recorded local campaign directories.')
    (out/'results.data.json').write_text(json.dumps(archive,indent=2)+'\n')
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with (out/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    display=[]
    for campaign in campaigns:
        augmented=native_completion.overlay(campaign['rows'],'initial',campaign['name'])
        native_groups={}
        for row in augmented:
            if 'native_hp' in row['candidate']:native_groups.setdefault(row['case'],[]).append(row)
        selected=[]
        for group in native_groups.values():
            chosen=native_completion.best_native(group) or group[0]
            selected.append(dict(chosen,candidate='native_hp_best'))
        display.append(dict(campaign,rows=campaign['rows']+selected))
    with publication_style():
        if not (figures/'geometry_fields.pdf').exists():fields_figure(figures)
        comparison_figure(display,figures)
    # Only ASM is used for the fine-mesh error row; its physical residual is recorded separately.
    error_rows=[]
    for case,label in [('cellular7_weak','Low isotropic'),('cellular7_anisotropic','Rotated anisotropic')]:
        vals=[stats(next(r for r in c['rows'] if r['case']==case and r['candidate']=='asm_d24')) for c in campaigns[2:]]
        error_rows.append(f"{label} & {tex_scientific(vals[0]['l2'])} & {tex_scientific(vals[1]['l2'])} & {vals[0]['l2']/vals[1]['l2']:.0f}"+r' \\')
    (out/'accuracy_table.tex').write_text('\n'.join([r'\begin{tabular}{lrrr}',r'\toprule',r'Diffusion & Coarse $L^2$ error & Fine $L^2$ error & Reduction \\',r'\midrule',*error_rows,r'\bottomrule',r'\end{tabular}'])+'\n')
    total=sum(c['completion']['attempted'] for c in campaigns);passed=sum(c['completion']['passed'] for c in campaigns)
    coarse=campaigns[2];fine=campaigns[3]
    g=lambda c,case:stats(next(r for r in c['rows'] if r['case']==case and r['candidate']=='asm_d24'))
    b,s=g(coarse,'advection_dominated'),g(coarse,'cellular7_anisotropic')
    f=g(fine,'cellular7_anisotropic')
    findings=(f"On the 6,114-triangle annulus, ASM+PP rises from {b['iterations']:.0f} iterations ({b['mean_ms']:.1f} ms) "
              f"for the smooth reference to {s['iterations']:.0f} iterations ({s['mean_ms']:.1f} ms) for the oscillatory anisotropic case. "
              f"On 22,825 triangles the latter takes {f['iterations']:.0f} iterations and {f['mean_ms']:.1f} ms.")
    template=Path(__file__).with_name('oscillatory_adr_section.tex.in').read_text()
    template=template.replace('@@FINDINGS@@',findings).replace('@@COUNTS@@',f'{passed} of {total} tested configurations pass; the remaining {total-passed} fail the common contract and are retained as NC.')
    assert '@@' not in template
    (out/'section.tex').write_text(prose(template))
    (out/'figure_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(figures.iterdir())},indent=2)+'\n')
    bundle=portable_bundle(out,figures)
    print(findings);print('Configurations',total,'passed',passed);print('Report',out);print('Bundle',bundle)


if __name__=='__main__':main()
