#!/usr/bin/env python3
"""Queue a bounded smoother diagnostic and the annular sweep after the square sweep."""
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
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
from scripts.adr_performance_common import read_json,atomic_json


def main():
    """Serialize GPU use and retain terminal diagnostics before running the next ladder."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--mesh-manifest',type=Path,required=True)
    args=parser.parse_args();study=args.study_root.resolve();square=study/'scaling/square'
    completion=square/'completion.json'
    active=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=path.read_bytes().decode().split('\0')
            if 'campaigns.legacy_adr_solver_studies.run_oscillatory_adr_scaling' in argv and str(square) in argv:
                active.append(int(path.parent.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    if not completion.exists():assert len(active)==1,('Expected one live square runner',active)
    print('Waiting for verified square runner',active,flush=True)
    while not completion.exists():
        if not Path('/proc') .joinpath(str(active[0])).exists():
            raise RuntimeError('Square runner stopped without a completion record')
        time.sleep(5)
    finished=read_json(completion)
    assert finished['status']=='completed' and finished['attempted']==finished['scheduled']==96
    probe=study/'smoother_probe';probe.mkdir(exist_ok=True)
    baseline=read_json(square/'specs/square_cellular7_directional_target8192_p6_amg_fgmres.json')
    records=[]
    for outer in ('FGMRES','PBICGSTAB'):
        spec=deepcopy(baseline);name='amg_dilu_'+outer.lower()
        spec.update(candidate=name,warmup=1,repeats=1,result=str(probe/(name+'.json')))
        spec['amgx_config']['solver']['solver']=outer
        spec['amgx_config']['solver']['preconditioner']['smoother']['solver']='MULTICOLOR_DILU'
        path=probe/(name+'.spec.json');atomic_json(path,spec)
        with (probe/(name+'.log')).open('w') as log:
            process=subprocess.run([sys.executable,'-m','scripts.adr_solver_comparison_worker','--spec',str(path)],
                stdout=log,stderr=subprocess.STDOUT,timeout=180)
        record=read_json(spec['result']);records.append(record)
        print('Directional smoother probe',name,record['status'],'returncode',process.returncode,flush=True)
    atomic_json(probe/'summary.json',records)
    print('Starting annular h/p sweep',flush=True)
    process=subprocess.run([sys.executable,'-m','campaigns.legacy_adr_solver_studies.run_oscillatory_adr_scaling',
        '--output',str(study/'scaling/annulus'),'--geometries','annulus',
        '--mesh-manifest',str(args.mesh_manifest.resolve())],check=False)
    if process.returncode:raise SystemExit(process.returncode)
    print('Annular h/p sweep complete',flush=True)


if __name__=='__main__':main()
