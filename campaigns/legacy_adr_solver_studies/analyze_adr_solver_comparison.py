#!/usr/bin/env python3
"""Archive the recorded ADR study and compute operator/solution diagnostics."""
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
import json
from pathlib import Path
import numpy as np
from scripts.adr_performance_common import atomic_json,read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();root=Path('run_logs')
    phases={'screen':'adr_solver_screen_bsr_p6_20260917','strict_large':'adr_solver_large_p6_20260917',
        'strict_tuning':'adr_solver_large_tuning_20260917','final':'adr_solver_final_20260917'}
    data={'campaigns':{},'native':{},'profiles':{},'operators':{},'solution_agreement':{},'assembly':{}}
    for name,directory in phases.items():
        p=root/directory
        data['campaigns'][name]={k:read_json(p/(k+'.json')) for k in ('manifest','completion','summary')}
        data['campaigns'][name]['local_artifact_directory']=str(p.resolve())
    for name,directory in [('screen','adr_native_hp_20260917'),('strict_large','adr_native_hp_large_20260917'),('final','adr_native_hp_final_20260917')]:
        p=root/directory
        data['native'][name]={f.stem:read_json(f) for f in p.glob('*.json') if not f.stem.endswith('_spec')}
    profiles=root/'adr_solver_profiles_20260917'
    for f in sorted(profiles.rglob('*.json')):
        if f.name.endswith(('.spec.json','.instrumented.json')) or 'assembly_profile_cache' in str(f):continue
        data['profiles'][str(f.relative_to(profiles))]=read_json(f)
    initial=root/phases['strict_large'];final=root/phases['final']
    for case in ('trigonometric','advection_dominated','anisotropic'):
        cache=initial/'cache'/f'{case}_n223_p6'
        blocks=np.load(cache/'system_blocks.npy',mmap_mode='r');neighbors=np.load(cache/'system_neighbors.npy',mmap_mode='r')
        rows=np.arange(len(neighbors));norm2=0.;skew2=0.
        for slot in range(neighbors.shape[1]):
            mask=neighbors[:,slot]>=0;r=rows[mask];j=neighbors[mask,slot]
            matches=neighbors[j]==r[:,None]
            if not np.all(matches.sum(axis=1)==1):raise RuntimeError('Unexpected nonsymmetric adjacency')
            k=matches.argmax(axis=1)
            own=blocks[r,slot];transpose=blocks[j,k].transpose(0,2,1)
            norm2+=float(np.sum(own*own));difference=own-transpose
            skew2+=float(np.sum(difference*difference))
        rhs=np.load(cache/'system_rhs.npy',mmap_mode='r')
        data['operators'][case]={'triangles':2*223**2,'free_faces':len(neighbors),'trace_dofs':rhs.size,
            'block_size':7,'relative_frobenius_nonsymmetry':float(np.sqrt(skew2/norm2)),
            'rhs_norm':float(np.linalg.norm(rhs)),'grid_edge_h':2/223,'triangle_diameter_h':2*np.sqrt(2)/223}
        solutions=sorted((final/'jobs').glob(case+'_*.solution.npy'))
        if case=='trigonometric':solutions.append(root/'adr_native_hp_final_20260917/standard.solution.npy')
        reference=next(f for f in solutions if 'asm_' in f.name);ref=np.load(reference)
        data['solution_agreement'][case]={'reference':reference.name,'relative_trace_coefficient_differences':{
            f.name:float(np.linalg.norm(np.load(f)-ref)/np.linalg.norm(ref)) for f in solutions},
            'note':'Cross-solver agreement, not an independent large sparse-direct reference. Small-mesh CPU assembly/direct-reference checks are in the screen artifacts.'}
        data['assembly'][case]={'large':read_json(initial/'jobs'/('assemble_'+case+'.json')),
            'screen':read_json(root/phases['screen']/'jobs'/('assemble_'+case+'.json'))}
    data['final_audit']=read_json(final/'audit.json')
    data['provenance']=read_json(initial/'provenance.json')
    data['residual_rounding_diagnostic']=read_json(initial/'residual_rounding_diagnostic.json')
    data['limitations']=['Manufactured smooth stationary ADR only; no transient or sharp-layer tests.',
        'Strict 1e-11 trials retain original failure status. Final contract is physical1e-10/internal5e-11 for all solvers.',
        'No exact AMGX internal multilevel SpMV count exposed by installed bindings. Native timer diagnostic records immediate preconditioner calls.',
        'Whole-device memory is a sampled peak/lower bound and includes other processes. RSS includes cache and validation.',
        'Prebuilt AMGX/binding source trees are dirty; exact runtime binaries identified by SHA256, not a claim of clean source reproduction.']
    atomic_json(a.output,data)
    print('Saved',a.output,'bytes',a.output.stat().st_size)
    print(json.dumps({'operators':data['operators'],'agreement':data['solution_agreement']},indent=2))


if __name__=='__main__':main()
