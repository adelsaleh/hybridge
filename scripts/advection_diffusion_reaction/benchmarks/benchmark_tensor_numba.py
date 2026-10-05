"""Compare exact ADR tensor specializations with a forced coupled LU path.

Only prepared local assembly/reconstruction is timed. No global sparse solve,
time stepping, AMGX setup or compilation is performed. Numba JIT is warmed.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import platform
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import numba
from scipy.sparse import coo_matrix

from hybridge import DGSpace, rectangle_mesh
from hybridge.mixed.adr_preparation import prepare_adr_data
from hybridge.mixed.coefficients import (
    prepare_diffusion,
    sample_diffusion_tensor,
    inverse_diffusion_values,
)
from hybridge.mixed.adr_numba import (
    assemble_projected_adr_trace_system_eliminated_numba as assemble,
    reconstruct_projected_adr_local_unknowns_numba as reconstruct,
)
from hybridge.runtime.benchmarking import measure
from hybridge.io.records import append_jsonl_record


# Keep the established benchmark import name.
from scripts.advection_diffusion_reaction.cases.tensor_cases import diffusion_cases as cases


def main():
    """Record prepared-path speedups with compilation excluded and parity gates."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subdivisions',type=int,default=16)
    parser.add_argument('--orders',type=int,nargs='+',default=[2,4,6])
    parser.add_argument('--threads',type=int,nargs='+',default=[1,16])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if numba.config.DISABLE_JIT:
        parser.error('Numba JIT must be enabled')
    for order in args.orders:
        space=DGSpace(rectangle_mesh(args.subdivisions,args.subdivisions),order,basis_type='dub_orth')
        source=space.constant(1.)
        reaction=space.constant(.4)
        beta=(space*space).field((space.constant(.7),space.constant(-.2)))
        trace_space=space.trace_space('legendre-modal')
        trace=np.linspace(-.1,.2,space.mesh.num_edg*trace_space.edg_dof)
        boundary=lambda x,y: .2+x-.3*y
        for name,diffusion in cases().items():
            prepared=prepare_adr_data(source,reaction,beta,space,diffusion=diffusion,trace_space=trace_space)
            lowered=prepare_diffusion(diffusion,space)
            general=replace(lowered,kinds=np.full_like(lowered.kinds,6),
                            inverse_values=inverse_diffusion_values(sample_diffusion_tensor(diffusion,space)))
            for threads in args.threads:
                numba.set_num_threads(threads)
                row=dict(case=name,order=order,elements=space.mesh.num_tri,threads=numba.get_num_threads(),
                         trace_basis=trace_space.kind,volume_basis='dub_orth',diffusion_structure=lowered.counts,
                         numba=numba.__version__,numpy=np.__version__,python=platform.python_version(),
                         threading_layer=numba.threading_layer(),solver='none',phases={})
                results={}
                for label,tensor in [('specialized',lowered),('general',general)]:
                    kwargs=dict(trace_space=trace_space,diffusion=diffusion,diffusion_data=tensor)
                    assembly=lambda: assemble(prepared,boundary,space,**kwargs)
                    recovery=lambda: reconstruct(trace,prepared,space,**kwargs)
                    row['phases'][label+'_assembly']=measure(assembly,args.repeats,.03)
                    row['phases'][label+'_reconstruction']=measure(recovery,args.repeats,.03)
                    result=assembly().trace_system
                    matrix=coo_matrix((result.data,(result.rows,result.cols)),shape=(result.rhs.size,result.rhs.size)).tocsr()
                    results[label]=(matrix,result.rhs,recovery())
                for actual,reference in zip(results['specialized'],results['general']):
                    if hasattr(actual,'indptr'):
                        np.testing.assert_array_equal(actual.indptr,reference.indptr)
                        np.testing.assert_array_equal(actual.indices,reference.indices)
                        actual,reference=actual.data,reference.data
                    np.testing.assert_allclose(actual,reference,rtol=3e-9,atol=3e-10)
                row['parity']='passed'
                for phase in ['assembly','reconstruction']:
                    row[phase+'_speedup']=row['phases']['general_'+phase]['median_seconds']/row['phases']['specialized_'+phase]['median_seconds']
                append_jsonl_record(args.output,row)
                print(f'p={order} {name} threads={threads}: assembly={row["assembly_speedup"]:.2f}x reconstruction={row["reconstruction_speedup"]:.2f}x',flush=True)


if __name__=='__main__':
    main()
