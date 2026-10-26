"""Trajectory-tolerance check using saved torsion fields; no PDE solve."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.paths import ARTICLE, ARTICLE_OUTPUTS, EQUILIBRIUM_REPO, FENICS_PYTHON, output_directory, record_file
CODE = Path(__file__).resolve().parent
from pathlib import Path
from dataclasses import replace
import json, sys, time
import numpy as np
from numba import set_num_threads
from projects.diocotron.dolfinx.equiband.config import SolverConfig
from projects.diocotron.dolfinx.equiband.geometry import PackedMesh, build_atlas

branch=Path(sys.argv[1]);destination=Path(sys.argv[2])
config=SolverConfig.load(branch.parent/'config.json')
with np.load(branch/'audit_mesh.npz',allow_pickle=False) as data:
    mesh=PackedMesh.from_geometry(data['geometry_coefficients'],int(data['geometry_degree']))
    torsion=data['torsion_coefficients'].copy();gradient=data['gradient_coefficients'].copy()
set_num_threads(4)
results=[]
for tolerance in [1e-13,1e-14]:
    started=time.monotonic()
    try:
        atlas=build_atlas(mesh,torsion,gradient,replace(config,ray_tolerance=tolerance,max_ray_steps=32768))
        result=dict(tolerance=tolerance,rays=config.number_of_rays,
            resolved_torsion_fraction=atlas.resolved_torsion_fraction,
            flow_resolution=atlas.flow_resolution,maximum_endpoint_error=float(atlas.endpoint_error.max()),
            minimum_length=float(atlas.total_lengths.min()),maximum_length=float(atlas.total_lengths.max()))
    except Exception as error:result=dict(tolerance=tolerance,error=repr(error))
    result['seconds']=time.monotonic()-started;results.append(result)
    destination.write_text(json.dumps(results,indent=2)+'\n');print(result,flush=True)
