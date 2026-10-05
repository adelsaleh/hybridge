"""Parameter-only p=0 AMGX study using the production native Poisson solver.

Each candidate replaces only the scalar coarse solver of a shared native shadow
solver. The production reference trajectory, fine BSR, p-smoother, tolerances,
and normal AMGX wrapper diagnostics are unchanged. No new compilation allowed.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
import traceback

from hybridge.runtime.optional import require_cupy
from hybridge.hdg.gram import field_l2_norm
from hybridge.diagnostics.solver import solver_result_metrics
from hybridge.linalg.multigrid.face_hp import AmgxScalarVcycle
from hybridge.linalg.multigrid.policy import scalar_p0_amgx_config
from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGSolver
from scripts.guiding_center.poisson.benchmark_poisson_backends import PRESET, digest
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
from scripts.guiding_center.time_schemes.stage_support import _fixed_operator_trace_predictor
from scripts.guiding_center.runtime.configuration import _make_poisson_options
from scripts.guiding_center.runtime.runner import run_guiding_center_case


# Preserve the measured pre-tuning baseline when production defaults change.
BASELINE_PARAMETERS = {"strength_threshold": 0.25,
                       "dense_lu_num_rows": 2048, "dense_lu_max_rows": 4096}


class Study:
    def __init__(self, config, candidates, output, profile):
        self.config, self.candidates, self.output, self.profile = config, candidates, output, profile
        self.cp = require_cupy()
        self.shadow = None
        self.coarse, self.diagnostics, self.history = {}, {}, []
        self.stream = (output/'samples.jsonl').open('w')
        self.failures = []

    def initialize(self, snapshot):
        cfg = replace(self.config, poisson_solver_rtol=0., poisson_solver_atol=1e-12)
        self.shadow = DiffusionReactionHDGSolver(snapshot.space,
            source=snapshot.accepted_density, reaction=snapshot.space.constant(0.),
            boundary_condition=snapshot.poisson_boundary, options=_make_poisson_options(cfg))
        self.shadow.solve()
        self.native = self.shadow._raw_cuda_fb_hp_mg_solver
        pp = self.native.preconditioner
        original = pp.coarse_solver
        metadata = dict(config=asdict(cfg), triangles=int(snapshot.space.mesh.num_tri),
            free_faces=int(len(snapshot.space.mesh.int_edges_inds)),
            mesh_nodes_sha256=digest(snapshot.space.mesh.node_coords),
            mesh_triangles_sha256=digest(snapshot.space.mesh.triangles),
            candidates=self.candidates, scope='Only p=0 AMGX parameters vary; shared native fine operator and p-smoother; rotating matched-state shadow solves.',
            new_compilation_allowed=False, profile=self.profile)
        (self.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        for name, overrides in self.candidates.items():
            conf = copy.deepcopy(scalar_p0_amgx_config())
            conf['solver'].update(BASELINE_PARAMETERS)
            conf['solver'].update(overrides)
            conf['solver']['print_grid_stats'] = 1
            # Preserve the production fixed-cycle/symmetric smoother contract.
            assert conf['solver']['solver']=='AMG' and conf['solver']['max_iters']==1
            assert conf['solver']['presweeps']==conf['solver']['postsweeps']==1
            assert conf['solver']['smoother']=={'solver':'JACOBI_L1','max_iters':1}
            assert conf['solver']['error_scaling']==0
            print(f'[p0] SETUP {name} {json.dumps(overrides)}',flush=True)
            candidate = None
            try:
                candidate = AmgxScalarVcycle(pp.levels[-1].operator, config=conf)
                pp.coarse_solver = candidate
                symmetry = pp.symmetry_defect()
                curvature = pp.positive_action_sample()
                if not math.isfinite(symmetry) or symmetry>1e-10 or not math.isfinite(curvature) or curvature<=0:
                    raise RuntimeError(f'symmetry={symmetry}, curvature={curvature}')
                self.coarse[name] = candidate
                self.diagnostics[name] = dict(symmetry_defect=symmetry, positive_curvature=curvature,
                    setup_seconds=candidate.setup_seconds, effective_config=candidate.solver.config_dict)
                print(f'[p0] GATE {name}: symmetry={symmetry:.3e}, curvature={curvature:.3e}, setup={candidate.setup_seconds:.4f}s',flush=True)
            except Exception as exc:
                if candidate is not None:
                    candidate.close()
                self.failures.append(dict(variant=name,phase='setup',error=str(exc),traceback=traceback.format_exc()))
                print(f'[p0] REJECT {name}: {exc}',flush=True)
            finally:
                pp.coarse_solver = original
        original.close()
        if 'baseline' not in self.coarse:
            raise RuntimeError('baseline setup failed')
        pp.coarse_solver = self.coarse['baseline']
        (self.output/'setup.json').write_text(json.dumps(dict(accepted=self.diagnostics,rejected=self.failures),indent=2)+'\n')

    def __call__(self, snapshot):
        if self.shadow is None:
            self.initialize(snapshot)
        cp = self.cp
        if self.history:
            guess, _ = _fixed_operator_trace_predictor(*self.history)
            guess = cp.ascontiguousarray(guess).copy()
        else:
            guess = cp.zeros_like(snapshot.poisson_result.trace_reduced_device)
        names=list(self.coarse)
        offset=(snapshot.step-1)%len(names)
        names=names[offset:]+names[:offset]
        output=[]
        for slot,name in enumerate(names):
            coarse=self.coarse[name]
            self.native.preconditioner.coarse_solver=coarse
            self.native.symmetry_defect=self.diagnostics[name]['symmetry_defect']
            self.native.positive_curvature=self.diagnostics[name]['positive_curvature']
            before_time,before_count=coarse.apply_seconds,coarse.apply_count
            cp.cuda.get_current_stream().synchronize()
            started=time.perf_counter()
            self.shadow.set_source(snapshot.accepted_density)
            self.shadow.set_boundary_condition(snapshot.poisson_boundary)
            try:
                result=self.shadow.solve(initial_guess=guess)
                cp.cuda.get_current_stream().synchronize()
                wall=time.perf_counter()-started
                if result.global_solve_result.backend!='fb-hp-mg-pcg' or not result.global_solve_result.converged:
                    raise RuntimeError('native failed or fell back')
                row=dict(order=snapshot.space.order,step=snapshot.step,variant=name,execution_slot=slot,
                    wall_seconds=wall,coarse_wall_seconds=coarse.apply_seconds-before_time,
                    coarse_applications=coarse.apply_count-before_count,
                    **solver_result_metrics('poisson',result))
                if snapshot.step in {1,8,self.config.num_steps}:
                    for label,field,reference in (
                        ('potential',result.field,snapshot.poisson_result.field),
                        ('qx',result.flux.components[0],snapshot.poisson_result.flux.components[0]),
                        ('qy',result.flux.components[1],snapshot.poisson_result.flux.components[1])):
                        difference=field-reference
                        error=field_l2_norm(difference)/max(field_l2_norm(reference),1e-300)
                        row[label+'_relative_l2']=error
                        if not math.isfinite(error) or error>1e-8:
                            raise RuntimeError(f'{label} parity={error}')
                self.stream.write(json.dumps(row,sort_keys=True)+'\n');self.stream.flush()
                output.append(f'{name}={wall*1000:.2f}ms/{row["poisson_solver_iterations"]}it')
            except Exception as exc:
                self.failures.append(dict(variant=name,phase='solve',step=snapshot.step,error=str(exc),traceback=traceback.format_exc()))
                (self.output/'candidate_failures.json').write_text(json.dumps(self.failures,indent=2)+'\n')
                if name=='baseline':raise
                print(f'[p0] REJECT {name} step={snapshot.step}: {exc}',flush=True)
                coarse.close();del self.coarse[name]
                if self.shadow._raw_cuda_fb_hp_mg_solver is not self.native:
                    raise RuntimeError('native solver was replaced after failure')
        self.native.preconditioner.coarse_solver=self.coarse['baseline']
        self.history.insert(0,snapshot.poisson_result.trace_reduced_device.copy())
        self.history=self.history[:3]
        if snapshot.step==1 or snapshot.step%4==0 or snapshot.step==self.config.num_steps:
            print(f'[p0] p={snapshot.space.order} step={snapshot.step} '+ ' '.join(output),flush=True)
        if self.profile and snapshot.step==self.config.num_steps:
            rhs=self.native.preconditioner._workspaces[-1].coarse_rhs.copy()
            rhs/=cp.linalg.norm(rhs)
            for coarse in self.coarse.values():coarse(rhs)
            cp.cuda.get_current_stream().synchronize()
            cp.cuda.profiler.start()
            try:
                for repeat in range(8):
                    for name,coarse in self.coarse.items():
                        cp.cuda.nvtx.RangePush(f'p0_{name}_cycle_{repeat}')
                        try:coarse(rhs)
                        finally:cp.cuda.nvtx.RangePop()
            finally:
                cp.cuda.profiler.stop()

    def close(self):
        self.stream.close()
        if self.shadow is not None:
            native=getattr(self.shadow,'_raw_cuda_fb_hp_mg_solver',None)
            if native is not None:
                native.preconditioner.coarse_solver=None
                native.close()
        for coarse in self.coarse.values():coarse.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order',type=int,choices=(4,5,6),default=6)
    parser.add_argument('--num-steps',type=int,default=16)
    parser.add_argument('--configs',type=Path)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.num_steps<=100:parser.error('num-steps must be 1..100')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'samples.jsonl').exists():parser.error('choose a new output directory')
    candidates=json.loads(args.configs.read_text()) if args.configs else {'baseline':{}}
    if 'baseline' not in candidates:parser.error('configs must include baseline')
    cfg=replace(preset_by_key(PRESET),order=args.order,mesh_size=.0068,minimum_triangles=150000,
        num_steps=args.num_steps,dt=.01,poisson_tau=1.,poisson_maxiter=500,
        plot_every=0,diagnostics_every=4,verbosity=0,
        diagnostics_dir=str(args.output_dir/'trajectory'),diagnostics_prefix=f'p0_p{args.order}')
    study=None;started=time.perf_counter()
    try:
        with kernel_cache_only(True):
            study=Study(cfg,candidates,args.output_dir,args.profile)
            result=run_guiding_center_case(cfg,preset_key=PRESET,step_observer=study)
            completion=dict(status='complete',steps=args.num_steps,order=args.order,
                triangles=int(result.mesh.num_tri),elapsed_seconds=time.perf_counter()-started,
                accepted=list(study.coarse),rejected=study.failures)
            (args.output_dir/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
            print(json.dumps(completion),flush=True)
    except Exception as exc:
        (args.output_dir/'failure.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),indent=2)+'\n')
        raise
    finally:
        if study is not None:study.close()


if __name__=='__main__':
    main()
