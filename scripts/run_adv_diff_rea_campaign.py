"""Validate, time and profile complete conservative ADR HDG solves.

Run from the repository root with PYTHONPATH=.; all times are seconds.
"""
import argparse
import cProfile
import csv
import json
import platform
import pstats
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.adv_diff_rea import solve_advection_diffusion_reaction
from scripts.adv_diff_rea_cases import CASES,get_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--meshes',nargs='+',type=int,default=[4,8,16])
    p.add_argument('--orders',nargs='+',type=int,default=[1,2])
    p.add_argument('--assembly-backend',choices=['numpy','cupy'],default='numpy')
    p.add_argument('--solver',choices=['direct','gmres','gpu'],default='direct')
    p.add_argument('--preconditioner',choices=['none','poly','block_jacobi','block_jacobi_poly','asm','asm_poly'],default='asm_poly')
    p.add_argument('--polynomial-degree',type=int,default=18)
    p.add_argument('--restart',type=int,default=75)
    p.add_argument('--maxiter',type=int,default=3000)
    p.add_argument('--rtol',type=float,default=1e-11)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,default=Path('results/adv_diff_rea'))
    p.add_argument('--profile',action='store_true',help='Extra untimed solve per configuration, CPU cProfile')
    p.add_argument('--validate-gpu',action='store_true',help='Compare CUDA assembly and solution to CPU direct reference')
    args=p.parse_args()
    if min(args.meshes)<1 or min(args.orders)<1 or args.repeats<1 or args.warmup<0:
        p.error('positive meshes, orders, repeats and nonnegative warmup required')
    if args.validate_gpu and args.assembly_backend!='cupy' and args.solver!='gpu':
        p.error('--validate-gpu requires CUDA assembly or solver')
    args.output.mkdir(parents=True,exist_ok=True)
    meta={'python':platform.python_version(),'platform':platform.platform(),
          'numpy':np.__version__,'scipy':scipy.__version__,
          'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    if args.assembly_backend=='cupy' or args.solver=='gpu':
        from hdgfem.backends.cupy import require_cupy_device
        cp=require_cupy_device()
        name=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
        meta.update(cupy=cp.__version__,gpu=name.decode() if isinstance(name,bytes) else name,
                    cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
    (args.output/'environment.json').write_text(json.dumps(meta,indent=2))
    rows=[]
    failed=False
    previous={}
    # Incremental JSONL retains completed runs if a later configuration fails.
    with (args.output/'runs.jsonl').open('w') as log:
        for case in args.cases:
            kw,exact=get_case(case)
            for order in args.orders:
                for mesh in sorted(set(args.meshes)):
                    tag=f'{case}_n{mesh}_p{order}'
                    row={'case':case,'mesh':mesh,'order':order,'status':'failed'}
                    try:
                        t=perf_counter()
                        space=DGSpace(rectangle_mesh(mesh,mesh),order,basis_type='dub_orth')
                        row['space_setup']=perf_counter()-t
                        options=dict(preconditioner=args.preconditioner,polynomial_degree=args.polynomial_degree,
                                     restart=args.restart,orthogonalization='cgs2',autotune=False,
                                     operator='raw_fused',asm_application='fused',block_jacobi_application='raw')
                        def run():
                            return solve_advection_diffusion_reaction(space=space,**kw,
                                assembly_backend=args.assembly_backend,solver=args.solver,rtol=args.rtol,
                                restart=args.restart,maxiter=args.maxiter,gpu_options=options)
                        samples=[]
                        for i in range(args.warmup+args.repeats):
                            r=run()
                            if i>=args.warmup:
                                sample={'configuration':tag,'repeat':i-args.warmup,
                                    'timings':r.timings,'iterations':r.iterations,'residual':r.relative_residual}
                                log.write(json.dumps(sample)+'\n'); log.flush()
                                samples.append(r.timings)
                        error=r.field.l2_error(exact)
                        row.update(l2_error=error,true_relative_residual=r.relative_residual,
                                   iterations=r.iterations,trace_dofs=r.assembly.system.rhs.size)
                        if not np.isfinite(error):
                            raise RuntimeError('nonfinite L2 error')
                        if case in ('quadratic','variable_velocity') and order>=2 and error>2e-8:
                            raise RuntimeError(f'polynomial exactness failed: {error}')
                        if args.validate_gpu:
                            ref=solve_advection_diffusion_reaction(space=space,**kw,rtol=args.rtol)
                            for label,a,b in [('blocks',r.assembly.system.blocks,ref.assembly.system.blocks),
                                             ('rhs',r.assembly.system.rhs,ref.assembly.system.rhs),
                                             ('trace',r.trace,ref.trace)]:
                                rel=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
                                row['gpu_cpu_'+label+'_error']=rel
                                if rel>max(1e-9,100*args.rtol):
                                    raise RuntimeError(f'GPU/CPU mismatch: {label}={rel}')
                        key=(case,order)
                        if key in previous and error > 1e-10:
                            oldmesh,olderror=previous[key]
                            row['observed_l2_rate']=float(np.log(olderror/error)/np.log(mesh/oldmesh)) if error>0 and olderror>0 else None
                            # Polynomial roundoff is not a mesh-convergence failure.
                            if case not in ('quadratic','variable_velocity') and error>=olderror:
                                raise RuntimeError('L2 error did not decrease on mesh refinement')
                        previous[key]=(mesh,error)
                        for phase in samples[0]:
                            values=[s[phase] for s in samples]
                            row[phase+'_median']=float(np.median(values))
                            row[phase+'_min']=float(np.min(values))
                            row[phase+'_max']=float(np.max(values))
                        combined=[s['solver_setup']+s['solve'] for s in samples]
                        row['setup_plus_solve_median']=float(np.median(combined))
                        if args.profile:
                            profiler=cProfile.Profile()
                            profiler.runcall(run)
                            profiler.dump_stats(str(args.output/(tag+'.prof')))
                            with (args.output/(tag+'_profile.txt')).open('w') as f:
                                pstats.Stats(profiler,stream=f).sort_stats('cumulative').print_stats(50)
                        row['status']='passed'
                    except Exception as exc:
                        failed=True
                        row['error']=f'{type(exc).__name__}: {exc}'
                    rows.append(row)
                    (args.output/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
                    print(json.dumps(row),flush=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
