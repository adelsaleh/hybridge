#!/usr/bin/env python3
"""Check tolerance-triggered restart behavior with one fixed ASM polynomial."""
from __future__ import annotations

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))
_campaign_sys.path.insert(0, str(_campaign_workspace_path('hdgfem-gmres')))

import argparse
from dataclasses import asdict
from pathlib import Path
import cupy as cp
from scripts.adr_performance_common import atomic_json,read_json
from scripts.adr_performance_worker import build_gpu,load_cache,get_space,verify
from hdgfem.backends.cupy_solver import CuPyProductionGMRESSolver,CuPyProductionGMRESOptions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    a=p.parse_args();spec=read_json(a.spec)
    system,get=load_cache(spec['cache']);space=get_space(spec)
    initial,op,base,pre,rhs,_=build_gpu(spec,system,get('element_blocks'),get('loc2glob'))
    del initial
    records=[]
    for tolerance in (1e-11,5e-11):
        solver=CuPyProductionGMRESSolver(op,preconditioner=pre,options=CuPyProductionGMRESOptions(
            restart=75,max_iterations=1000,rtol=tolerance,atol=0.,orthogonalization='cgs2',
            raise_on_failure=False,cgs2_fallback_threshold=None))
        result=solver.solve(rhs,x0=None,raise_on_failure=False)
        checked=verify(spec,space,system,get,cp.asnumpy(result.solution),solver_converged=result.converged)
        records.append(dict(internal_rtol=tolerance,iterations=result.iterations,restart_cycles=result.restart_cycles,
            cycle_records=[asdict(c) for c in result.cycle_records],
            true_relative_history=(result.true_residual_history/float(cp.linalg.norm(rhs).item())).tolist(),**checked))
        del solver,result
    report=dict(status='passed' if all(r['passed'] for r in records) else 'numerical_failure',records=records,
        note='Same operator, RHS and preconditioner object; zero guess. No diagnostic timing is ranked.')
    atomic_json(spec['result'],report);print(report,flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
