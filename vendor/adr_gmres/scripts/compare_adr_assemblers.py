#!/usr/bin/env python3
"""Compare cross-checkout ADR assemblies with explicit trace-basis parity.

Run workers serially to avoid contaminating GPU timing. Sparse systems are
compared in Bernstein coordinates; conversion timing is reported separately
from each implementation's native assembly and canonical host CSR timings.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',default=['quadratic','variable_velocity','trigonometric','advection_dominated','anisotropic'])
    parser.add_argument('--meshes',nargs='+',type=int,default=[4])
    parser.add_argument('--degrees',nargs='+',type=int,default=[2])
    parser.add_argument('--backends',nargs='+',default=['branch-numpy','branch-cupy','master-numpy','master-numba','master-raw'])
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--canonical-bernstein',action='store_true')
    a=parser.parse_args()
    import numpy as np
    from scipy import sparse
    if a.output.exists() and any(a.output.iterdir()):
        parser.error('Use a new empty output directory to preserve existing evidence')
    a.output.mkdir(parents=True,exist_ok=True)
    branch_root=Path(__file__).resolve().parents[1]
    records=[]
    for case in a.cases:
        for n in a.meshes:
            for p in a.degrees:
                reference=None
                for backend in a.backends:
                    out=a.output/f'{case}_n{n}_p{p}_{backend}'
                    out.mkdir()
                    root=branch_root if backend.startswith('branch') else a.master_root
                    command=[sys.executable,str(Path(__file__).with_name('compare_adr_assembly_worker.py')),
                        '--root',str(root),'--backend',backend,'--case',case,'--mesh',str(n),'--degree',str(p),
                        '--warmup',str(a.warmup),'--repeats',str(a.repeats),'--threads',str(a.threads),'--output',str(out)]
                    if a.canonical_bernstein: command.append('--canonical-bernstein')
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
                    with (out/'worker.log').open('w') as log:
                        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                    if result.returncode:
                        raise RuntimeError(f'{backend} failed; inspect {out}/worker.log')
                    r=json.loads((out/'result.json').read_text())
                    if r['status']=='unsupported':
                        records.append(r)
                        print(case,n,p,backend,'unsupported',flush=True)
                        continue
                    matrix=sparse.load_npz(out/'matrix.npz'); rhs=np.load(out/'rhs.npy')
                    geometry=dict(np.load(out/'geometry.npz'))
                    if reference is None:
                        if backend!='branch-numpy':
                            raise ValueError('branch-numpy must be first to define the reference')
                        reference=(matrix,rhs,geometry)
                    ref,ref_rhs,ref_geometry=reference
                    for key in ('faces','normals','orientations','free_faces','volume_points','volume_weights','volume_basis','face_weights'):
                        np.testing.assert_allclose(geometry[key],ref_geometry[key],atol=1e-13,rtol=1e-13,err_msg=key)
                    start=perf_counter()
                    transform=np.linalg.lstsq(geometry['face_basis'].T,ref_geometry['face_basis'].T,rcond=None)[0]
                    np.testing.assert_allclose(geometry['face_basis'].T@transform,ref_geometry['face_basis'].T,atol=1e-12)
                    if backend.startswith('master') and not a.canonical_bernstein:
                        global_transform=sparse.kron(sparse.eye(matrix.shape[0]//(p+1)),transform,format='csr')
                        matrix=(global_transform.T@matrix@global_transform).tocsr()
                        rhs=global_transform.T@rhs
                    conversion=perf_counter()-start
                    matrix_error=float(sparse.linalg.norm(matrix-ref)/sparse.linalg.norm(ref))
                    rhs_error=float(np.linalg.norm(rhs-ref_rhs)/max(np.linalg.norm(ref_rhs),1e-300))
                    comparison={'matrix_relative_error':matrix_error,'rhs_relative_error':rhs_error,
                        'comparison_basis_conversion_seconds':conversion}
                    if n<=8:
                        x=sparse.linalg.spsolve(matrix,rhs); y=sparse.linalg.spsolve(ref,ref_rhs)
                        comparison['trace_solution_relative_error']=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-300))
                        comparison['reference_system_relative_residual']=float(np.linalg.norm(ref@x-ref_rhs)/np.linalg.norm(ref_rhs))
                    r['parity']=comparison
                    r['status']='passed' if max(matrix_error,rhs_error)<1e-9 and comparison.get('trace_solution_relative_error',0)<1e-8 else 'parity_failed'
                    records.append(r)
                    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
                    (a.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
                    print(case,n,p,backend,r['status'],f"{1000*r['medians']['total_seconds']:.3f} ms",f"matrix error {matrix_error:.2e}",flush=True)
    (a.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    completion={'status':'completed' if all(r['status'] in ('passed','unsupported') for r in records) else 'failed',
        'passed':sum(r['status']=='passed' for r in records),'unsupported':sum(r['status']=='unsupported' for r in records),
        'scope':'native assembly with matched Duffy volume quadrature, explicit upwind and diffusion stabilization; basis-transformed parity; no solver ranking'}
    (a.output/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(completion,flush=True)
    if completion['status']!='completed':raise SystemExit(1)


if __name__=='__main__':
    main()
