#!/usr/bin/env python3
"""Measure trace-matrix conditioning and coupling on cached oscillatory ADR cases."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,onenormest,splu,norm
from scripts.adr_performance_common import atomic_json,read_json
from scripts.adr_performance_worker import load_cache
from hdgfem.solvers.adv_diff_rea import _csr


def main():
    """Record reproducible 1-norm condition lower estimates, asymmetry and nonnormality."""
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--campaign',type=Path)
    source.add_argument('--specs',type=Path,nargs='+',help='Assembly specifications for selected scaling systems')
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    if a.specs:
        if a.output is None:parser.error('--specs requires --output')
        inputs=[read_json(path) for path in a.specs]
    else:
        args=read_json(a.campaign/'manifest.json')['arguments']
        inputs=[dict(case=case,n=args['n'],p=args['p'],cache=str(a.campaign/'cache'/f"{case}_n{args['n']}_p{args['p']}")) for case in args['cases']]
    output=a.output or a.campaign/'matrix_diagnostics.json'
    results=[];baseline=None
    for spec in inputs:
        case=spec['case'];cache=Path(spec['cache'])
        system,_=load_cache(cache);matrix=_csr(system).tocsc();matrix.sum_duplicates();matrix.sort_indices()
        lu=splu(matrix)
        inverse=LinearOperator(matrix.shape,dtype=float,matvec=lu.solve,matmat=lu.solve,
            rmatvec=lambda x:lu.solve(x,trans='T'),rmatmat=lambda x:lu.solve(x,trans='T'))
        np.random.seed(1729)
        one=float(norm(matrix,1));inverse_estimate=float(onenormest(inverse,t=4,itmax=8))
        frob=float(norm(matrix,'fro'))
        asym=float(norm(matrix-matrix.T,'fro')/frob)
        commutator=matrix.T@matrix-matrix@matrix.T
        departure=float(norm(commutator,'fro')/frob**2)
        blocks=np.asarray(system.blocks)
        norms=np.linalg.norm(blocks,axis=(-2,-1))
        diagonal=system.neighbors==np.arange(system.num_rows)[:,None]
        ratio=np.sum(np.where(diagonal,0,norms),axis=1)/np.sum(np.where(diagonal,norms,0),axis=1)
        row=dict(case=case,n=spec['n'],p=spec['p'],cache=str(cache),trace_dofs=system.num_dofs,
            matrix_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),
            norm1=one,inverse_norm1_lower_estimate=inverse_estimate,condition1_lower_estimate=one*inverse_estimate,
            relative_asymmetry_frobenius=asym,normalized_commutator_frobenius=departure,
            block_offdiag_to_diag_median=float(np.median(ratio)),block_offdiag_to_diag_max=float(max(ratio)),
            note='Condition estimate uses sparse LU and onenormest(t=4,itmax=8,seed=1729); this is a lower estimate, not exact kappa_2. Coupling metrics are in unchanged Bernstein trace coordinates.')
        if case=='advection_dominated':baseline=blocks.copy()
        if case=='oscillatory_rhs' and baseline is not None:
            row['same_matrix_as_smooth_reference']=bool(np.array_equal(blocks,baseline))
            assert row['same_matrix_as_smooth_reference']
        results.append(row);atomic_json(output,results)
        print(case,'cond1_est',row['condition1_lower_estimate'],'asymmetry',asym,'nonnormality',departure,flush=True)


if __name__=='__main__':main()
