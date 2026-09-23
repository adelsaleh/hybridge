#!/usr/bin/env python3
"""Check that a resolved oscillatory ADR result is stable under more quadrature."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scripts.adr_performance_common import read_json,atomic_json
from scripts.adr_performance_worker import assemble,get_space,load_cache,relative
from scripts.adv_diff_rea_cases import get_case
from hdgfem.assembly.hdg import reconstruct_local_unknowns


def coefficients(spec):
    """Reconstruct an independently solved trace in its recorded local basis."""
    space=get_space(spec);_,get=load_cache(spec['cache'])
    trace=np.load(Path(spec['cache'])/'reference_trace.npy')
    unknown=reconstruct_local_unknowns(trace,get('source_rhs'),get('local_inverse'),get('boundary'),space)
    return unknown.reshape(space.mesh.num_tri,3,space.el_dof)[:,0]


def main():
    """Reassemble/direct-solve with doubled volume and edge quadrature, then compare fields."""
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--campaign',type=Path)
    source.add_argument('--spec',type=Path,help='Exact assembly specification for a scaling point')
    parser.add_argument('--case',default='cellular7_anisotropic')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    path=a.spec or next((a.campaign/'specs').glob('assemble_'+a.case+'_*.json'))
    original=read_json(path);case=original['case'];doubled=dict(original)
    q0=original.get('volume_quad_1d') or 2*original['p']+2
    qe=original.get('edge_quad_1d') or 2*original['p']+2
    doubled.update(volume_quad_1d=2*q0,edge_quad_1d=2*qe,cache=str(out/'cache'),result=str(out/'assembly.json'))
    atomic_json(out/'spec.json',doubled)
    assembled=assemble(doubled);atomic_json(out/'assembly.json',assembled)
    base,_=load_cache(original['cache']);high,_=load_cache(doubled['cache'])
    low_coeff=coefficients(original);high_coeff=coefficients(doubled);space=get_space(doubled)
    exact=get_case(case)[1]
    difference=space.field(high_coeff-low_coeff).l2_norm()
    low_error=space.field(low_coeff).l2_error(exact);high_error=space.field(high_coeff).l2_error(exact)
    result=dict(case=case,original_spec=str(path.resolve()),volume_points_1d=[q0,2*q0],edge_points_1d=[qe,2*qe],
        matrix_relative_change=relative(high.blocks,base.blocks),rhs_relative_change=relative(high.rhs,base.rhs),
        primal_l2_difference=difference,original_l2_error_high_quadrature=low_error,refined_l2_error=high_error,
        difference_over_original_error=difference/low_error,
        status='passed' if difference<.05*low_error else 'quadrature_sensitive',
        criterion='Primal-field change below 5% of the original discretization error; both errors evaluated at the higher volume quadrature. This does not assert matrix coefficients agree to roundoff.')
    atomic_json(out/'comparison.json',result);print(result,flush=True)


if __name__=='__main__':main()
