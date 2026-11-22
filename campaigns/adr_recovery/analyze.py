"""Authenticate completed recovery runs and export compact tables and a figure.

Run outputs are read-only inputs. Derived evidence is written to the supplied
directory, normally docs/campaigns/adr_recovery/outputs/JOBID.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from campaigns.adr_recovery.run import DEGREES, REPO, qualification_passed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output):
    output=Path(output)
    complete=json.loads((output/'COMPLETE.json').read_text())
    manifest=json.loads((output/'manifest.json').read_text())
    expected=hashlib.sha256(json.dumps(dict(configuration=manifest['configuration'],sources=manifest['sources']),
                                     sort_keys=True).encode()).hexdigest()
    assert expected==manifest['sha256']==complete['source_identity']
    assert complete['degrees']==list(DEGREES)
    assert sha(output/'REPORT.md')==complete['report_sha256']
    assert sha(output/'summary.json')==complete['summary_sha256']
    passed,gates=qualification_passed(output/'qualification.xml')
    assert passed
    source_matches={name:sha(REPO/name)==value for name,value in manifest['sources']['files'].items()}
    assert all(source_matches.values()), 'Current numerical sources differ from this run'
    records=[]
    for degree in DEGREES:
        path=output/f'p{degree}'/'result.json'
        marker=json.loads((path.parent/'COMPLETE.json').read_text())
        assert marker['source_identity']==expected and marker['result_sha256']==sha(path)
        result=json.loads(path.read_text())
        assert result['status']=='complete' and result['order']==degree
        assert len(result['assembly'])==11 and len(result['pipelines'])==12
        assert any(r['variant']=='magma' and r['status']=='passed' for r in result['solvers'])
        backward=max(r['backward_error'] for r in result['solvers'] if r['status']=='passed')
        assert np.isfinite(backward) and backward<=5e-11
        row=dict(degree=degree,elements=result['elements'],chunk_cells=result['chunk_cells'],
                 result_sha256=sha(path),solver_backward_error_pilot=backward,
                 maximum_field_difference=0.,maximum_cell_mean_defect=0.,
                 peak_chunk_reserved_bytes=max(c['device_pool_reserved'] for c in result['chunks']))
        for variant in ('baseline','optimized'):
            for policy in ('reuse','rebuild'):
                chunks=[c for c in result['chunks'] if c['variant']==variant and c['policy']==policy]
                end=0
                for c in chunks:
                    assert c['first']==end
                    end+=c['cells']
                    assert len(c['samples_seconds'])==manifest['configuration']['repeats']
                    assert all(np.isfinite(x) and x>0 for x in c['samples_seconds'])
                    assert c['mean_relative_defect']<=2e-9 and c['coefficient_relative_error']<=5e-8
                    row['maximum_field_difference']=max(row['maximum_field_difference'],c['coefficient_relative_error'])
                    row['maximum_cell_mean_defect']=max(row['maximum_cell_mean_defect'],c['mean_relative_defect'])
                assert end==result['elements']
                sums=np.sum([c['samples_seconds'] for c in chunks],axis=0)
                recorded=result['full_mesh_totals'][f'{variant}/{policy}']
                np.testing.assert_allclose(sums,recorded['samples_seconds'],rtol=1e-13,atol=1e-13)
                value=float(np.median(sums))
                assert np.isclose(value,recorded['median_seconds'],rtol=1e-13,atol=1e-13)
                row[f'{variant}_{policy}_seconds']=value
        for policy in ('reuse','rebuild'):
            row[f'{policy}_speedup']=row[f'baseline_{policy}_seconds']/row[f'optimized_{policy}_seconds']
            row[f'{policy}_configuration']=result['winners'][policy]['config']
        row['full_mesh_retained_flux_cache_bytes']=sum(c['cache_bytes'] for c in result['chunks']
            if c['variant']=='optimized' and c['policy']=='reuse')
        memory=[]
        path=output/f'p{degree}'/'worker.gpu.csv'
        for sample in csv.DictReader(path.open()):
            try: memory.append(float(sample['memory_used_mib']))
            except (ValueError,KeyError): pass
        row['peak_sampled_device_mib']=max(memory) if memory else None
        records.append(row)
    return dict(output=str(output.resolve()),source_identity=expected,verified_source_files=len(source_matches),
        qualification=gates,cases=records,
        maximum_field_difference=max(r['maximum_field_difference'] for r in records),
        maximum_cell_mean_defect=max(r['maximum_cell_mean_defect'] for r in records),
        maximum_solver_backward_error_pilot=max(r['solver_backward_error_pilot'] for r in records),
        cache_scope='Reuse retains each chunk cache across repeated calls; all chunks are not simultaneously resident. '
            'Full-mesh retained flux-cache bytes is a lower bound excluding other simulation data and workspaces.',
        timing_scope='Sums of actual synchronized chunk samples; input-fixture construction excluded; '
            'per-element cache setup excluded for reuse and included for rebuild; reference basis retained in both.',
        hashes={name:sha(output/name) for name in ('manifest.json','COMPLETE.json','REPORT.md','summary.json',
                                                  'qualification.xml','packages.txt','runtime.json')})


def export(output,destination):
    data=audit(output)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    (destination/'audit.json').write_text(json.dumps(data,indent=2)+'\n')
    names=['degree','elements','chunk_cells','baseline_reuse_seconds','optimized_reuse_seconds','reuse_speedup',
           'baseline_rebuild_seconds','optimized_rebuild_seconds','rebuild_speedup',
           'full_mesh_retained_flux_cache_bytes','peak_sampled_device_mib','maximum_field_difference','maximum_cell_mean_defect']
    with (destination/'timings.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=names,extrasaction='ignore');writer.writeheader();writer.writerows(data['cases'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.2),layout='constrained')
    p=np.array([r['degree'] for r in data['cases']])
    for variant,color,label in [('baseline','#666666','Raw reference'),('optimized','#087f8c','Optimized')]:
        for policy,style in [('reuse','-'),('rebuild','--')]:
            seconds=np.array([r[f'{variant}_{policy}_seconds']/r['elements']*100000 for r in data['cases']])
            axes[0].plot(p,seconds,style,marker='o',color=color,label=f'{label}, {policy}')
    axes[0].set(yscale='log',xlabel='Polynomial degree p',ylabel='Seconds per 100,000 triangles',xticks=p,
                title='Recovery and projection cost')
    axes[0].grid(alpha=.2,which='both');axes[0].legend(fontsize=8)
    for offset,policy,color in [(-.18,'reuse','#087f8c'),(.18,'rebuild','#e29b35')]:
        values=[r[f'{policy}_speedup'] for r in data['cases']]
        bars=axes[1].bar(p+offset,values,width=.34,label=policy.capitalize(),color=color)
        axes[1].bar_label(bars,fmt='%.1f×',fontsize=8,padding=3)
    axes[1].axhline(1,color='#555555',linewidth=.8)
    axes[1].set(xlabel='Polynomial degree p',ylabel='Raw reference / optimized',xticks=p,title='Measured speedup')
    axes[1].set_ylim(0,max(r['reuse_speedup'] for r in data['cases'])*1.16);axes[1].legend()
    fig.suptitle('ADR recovery · FP64 · NVIDIA H200 NVL')
    fig.supxlabel('524,288 triangles per degree, processed in bounded chunks. Reuse assumes the active chunk cache is retained.',fontsize=8)
    for suffix in ('png','svg','pdf'):
        fig.savefig(destination/f'adr_recovery_h200.{suffix}',dpi=180)
    plt.close(fig)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    result=export(args.output,args.destination)
    print(json.dumps(dict(degrees=len(result['cases']),sources=result['verified_source_files'],
        max_field_difference=result['maximum_field_difference'],max_mean_defect=result['maximum_cell_mean_defect']),indent=2))
