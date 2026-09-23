#!/usr/bin/env python3
"""Export a narrow-necked five-lobed annulus for the oscillatory ADR study."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hdgfem.core.mesh import gmsh_smooth_star_mesh,mesh_edge_min_max


def main():
    """Reuse the canonical Gmsh generator and preserve exact mesh arrays/metadata."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sizes',nargs='+',type=float,default=[.04,.02])
    parser.add_argument('--boundary-points',type=int,default=400)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for size in a.sizes:
        mesh=gmsh_smooth_star_mesh(size,radius=1.,amplitude=.35,mode=5,hole_radius=.58,
            boundary_points=a.boundary_points,cache_dir=a.output/'mesh_cache',num_threads=1,log_cache=False)
        path=a.output/f'star_h{size:g}.npz'
        np.savez(path,node_coords=mesh.node_coords,triangles=mesh.triangles)
        boundary=mesh.edges[mesh.bnd_edges_inds]
        neighbors={}
        for first,second in boundary:
            neighbors.setdefault(int(first),[]).append(int(second));neighbors.setdefault(int(second),[]).append(int(first))
        assert all(len(v)==2 for v in neighbors.values())
        remaining=set(neighbors);components=0
        while remaining:
            components+=1;stack=[remaining.pop()]
            while stack:
                for v in neighbors[stack.pop()]:
                    if v in remaining:remaining.remove(v);stack.append(v)
        assert components==2
        data=dict(geometry='five-lobed annulus',outer_radius='1 + 0.35*cos(5*theta)',hole_radius=.58,
            nominal_minimum_neck_width=.07,outer_boundary_points=a.boundary_points,mesh_size=size,
            triangles=mesh.num_tri,vertices=len(mesh.node_coords),interior_faces=len(mesh.int_edges_inds),
            boundary_components=components,edge_lengths=mesh_edge_min_max(mesh),
            area=float(np.sum(mesh.aff_jacs)*2),mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            note='Affine triangles; sampled polygonal outer wall and polygonal approximation to circular hole. Exact Dirichlet trace is imposed on every discrete boundary edge.')
        path.with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')
        print(path,data,flush=True)


if __name__=='__main__':main()
