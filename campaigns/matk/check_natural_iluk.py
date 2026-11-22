#!/usr/bin/env python3
"""Independent block ILU(k) application and operator-integrity GPU oracle."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import bsr_matrix
from common import write_report


def levels_reference(pattern, fill):
    n = len(pattern)
    levels = np.where(pattern, 0, 1_000_000)
    # Dense pivot-major recurrence, independent from the C++ sparse row heap.
    for j in range(n):
        for i in range(j+1, n):
            for k in range(j+1, n):
                candidate = int(levels[i,j])+int(levels[j,k])+1
                if candidate <= fill:
                    levels[i,k] = min(levels[i,k], candidate)
    return levels


def reference_factors(matrix, fill):
    b = matrix.blocksize[0]
    n = matrix.shape[0]//b
    graph = np.zeros((n,n), dtype=bool)
    for i in range(n):
        graph[i,matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]] = True
    levels = levels_reference(graph, fill)
    factors = [{j:np.zeros((b,b)) for j in range(n) if levels[i,j] <= fill} for i in range(n)]
    for i in range(n):
        for p in range(matrix.indptr[i],matrix.indptr[i+1]):
            factors[i][int(matrix.indices[p])] = matrix.data[p].astype(np.float64).copy()
        for j in sorted(k for k in factors[i] if k<i):
            factors[i][j] = np.linalg.solve(factors[j][j].T, factors[i][j].T).T
            for k in factors[j]:
                if k>j and k in factors[i]:
                    factors[i][k] -= factors[i][j] @ factors[j][k]
    return factors, int(np.count_nonzero(levels<=fill))


def apply_reference(factors, rhs):
    n = len(factors); b = factors[0][0].shape[0]
    y = rhs.astype(np.float64).reshape(n,b).copy()
    for i in range(n):
        for j in factors[i]:
            if j<i: y[i] -= factors[i][j] @ y[j]
    for i in range(n-1,-1,-1):
        for j in factors[i]:
            if j>i: y[i] -= factors[i][j] @ y[j]
        y[i] = np.linalg.solve(factors[i][i], y[i])
    return y.ravel()


def fixture(block, dtype, permuted, fill_chain=False):
    rng = np.random.default_rng(1974)
    n=32 if fill_chain else 14
    if fill_chain:
        # Each successive lower pivot creates a new entry at the next fill
        # level in the last column. Levels 5/10/20 have distinct patterns and
        # non-negligible numerical values (roughly powers of 0.8).
        mask = np.eye(n,dtype=bool) | np.eye(n,k=-1,dtype=bool)
        mask[0,-1] = True
    else:
        mask = rng.random((n,n))<.20
    np.fill_diagonal(mask,True)
    rows=[0]; columns=[]; values=[]
    for i in range(n):
        for j in range(n):
            if mask[i,j]:
                data = rng.normal(0,.005 if fill_chain else .04,(block,block))
                if fill_chain:
                    data += (2. if i==j else (-1.6 if j==i-1 else .4))*np.eye(block)
                elif i==j:
                    data += (5+block)*np.eye(block)
                columns.append(j); values.append(data)
        rows.append(len(columns))
    a=bsr_matrix((np.array(values,dtype=dtype),np.array(columns,np.int32),np.array(rows,np.int32)),shape=(n*block,n*block))
    if permuted:
        order=rng.permutation(n)
        p=(order[:,None]*block+np.arange(block)[None,:]).ravel()
        a=a.tocsr()[p,:][:,p].tobsr(blocksize=(block,block))
        a.sort_indices()
    return a


class API:
    def __init__(self, path, verbosity):
        self.lib=ct.CDLL(str(path.resolve()), mode=ct.RTLD_GLOBAL)
        P,I,S=ct.c_void_p,ct.c_int,ct.c_char_p
        PP=ct.POINTER(P)
        signatures={'initialize':[],'finalize':[], 'config_create':[PP,S],
          'resources_create_simple':[PP,P], 'matrix_create':[PP,P,I], 'vector_create':[PP,P,I],
          'solver_create':[PP,P,I,P], 'matrix_upload_all':[P,I,I,I,I,P,P,P,P],
          'matrix_download_all':[P,P,P,P,PP], 'matrix_replace_coefficients':[P,I,I,P,P],
          'vector_upload':[P,I,I,P], 'vector_download':[P,P], 'solver_setup':[P,P],
          'solver_resetup':[P,P], 'solver_solve':[P,P,P], 'solver_solve_with_0_initial_guess':[P,P,P]}
        for obj in ('config','resources','matrix','vector','solver'):
            signatures[obj+'_destroy']=[P]
        for name,types in signatures.items():
            fun=getattr(self.lib,'AMGX_'+name); fun.argtypes=types; fun.restype=I
        self.callback_type=ct.CFUNCTYPE(None,ct.c_char_p,ct.c_int)
        self.messages=[]
        def output(message,length):
            text=message[:length].decode(errors='replace'); self.messages.append(text)
            if verbosity>=3: print(text,end='',flush=True)
        self.callback=self.callback_type(output)
        self.lib.AMGX_register_print_callback.argtypes=[self.callback_type]
        self.lib.AMGX_register_print_callback(self.callback)
        self.call('initialize')

    def call(self,name,*args):
        status=getattr(self.lib,'AMGX_'+name)(*args)
        if status: raise RuntimeError(f'AMGX_{name} returned {status}')


def gpu_case(api,block,fill,mode,permuted,fill_chain=False):
    # Enum values are from this tree's include/amgx_config.h, not a guessed formula.
    dtype,enum,tol=(np.float64,8193,2e-10) if mode=='dDDI' else (np.float32,8465,5e-5)
    a=fixture(block,dtype,permuted,fill_chain); n=a.shape[0]//block
    rng=np.random.default_rng(91); weight=.73
    cfg=dict(config_version=2,exception_handling=0,determinism_flag=1,block_format='ROW_MAJOR',
      solver=dict(solver='MULTICOLOR_ILU',scope='oracle',max_iters=1,monitor_residual=0,
        store_res_history=0,print_solve_stats=0,verbosity_level=1,ilu_sparsity_level=fill,
        block_ilu_backend='cusparse_legacy',reorder_cols_by_color=0,insert_diag_while_reordering=0,
        bsr_spmv_backend='cusparse_generic',relaxation_factor=weight))
    handles=[]
    def make(kind,*args):
        h=ct.c_void_p(); api.call(kind+'_create',ct.byref(h),*args); handles.append((kind,h)); return h
    def ptr(x): return ct.c_void_p(x.ctypes.data)
    errors=[]; preserved=[]
    try:
        config=make('config',json.dumps(cfg).encode())
        resources=ct.c_void_p()
        api.call('resources_create_simple',ct.byref(resources),config); handles.append(('resources',resources))
        matrix=make('matrix',resources,enum); rhs=make('vector',resources,enum); x=make('vector',resources,enum)
        solver=make('solver',resources,enum,config)
        api.call('matrix_upload_all',matrix,n,len(a.indices),block,block,ptr(a.indptr),ptr(a.indices),ptr(a.data),None)
        for changed in (False,True):
            if changed:
                a.data *= np.linspace(.9,1.1,len(a.data),dtype=dtype)[:,None,None]
                api.call('matrix_replace_coefficients',matrix,n,len(a.indices),ptr(a.data),None)
                api.call('solver_resetup',solver,matrix)
            else:
                api.call('solver_setup',solver,matrix)
            factors,count=reference_factors(a,fill)
            rr=np.empty_like(a.indptr); cc=np.empty_like(a.indices); vv=np.empty_like(a.data); diagonal=ct.c_void_p()
            api.call('matrix_download_all',matrix,ptr(rr),ptr(cc),ptr(vv),ct.byref(diagonal))
            preserved.append(np.array_equal(rr,a.indptr) and np.array_equal(cc,a.indices) and np.array_equal(vv,a.data))
            for zero in (True,False):
                b=rng.normal(size=a.shape[0]).astype(dtype)
                initial=rng.normal(size=a.shape[0]).astype(dtype)
                api.call('vector_upload',rhs,n,block,ptr(b)); api.call('vector_upload',x,n,block,ptr(initial))
                expected=weight*apply_reference(factors,b) if zero else initial+weight*apply_reference(factors,b-a@initial)
                api.call('solver_solve_with_0_initial_guess' if zero else 'solver_solve',solver,rhs,x)
                actual=np.empty_like(b); api.call('vector_download',x,ptr(actual))
                error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
                errors.append(error)
        if not all(preserved) or max(errors)>tol or not np.isfinite(errors).all():
            raise AssertionError(f'operator preserved={preserved}, application errors={errors}, tolerance={tol}')
        return dict(block=block,fill=fill,mode=mode,permuted=permuted,fill_chain=fill_chain,
                    block_rows=n,original_blocks=len(a.indices),
                    factor_blocks=count,application_errors=errors,operator_unchanged=all(preserved),status='passed')
    finally:
        for kind,h in reversed(handles): api.call(kind+'_destroy',h)


def rejection_case(api, case, fill):
    """Exercise backend diagnostics through the public C API in a fresh process."""
    a = fixture(2, np.float64, False)
    if case in ('unsorted', 'duplicate'):
        i = next(i for i in range(len(a.indptr)-1) if a.indptr[i+1]-a.indptr[i] >= 2)
        lo = a.indptr[i]
        if case == 'unsorted':
            a.indices[lo:lo+2] = a.indices[lo:lo+2][::-1]
            a.data[lo:lo+2] = a.data[lo:lo+2][::-1]
        else:
            a.indices[lo+1] = a.indices[lo]
    elif case == 'missing-diagonal':
        where = int(np.flatnonzero(a.indices[:a.indptr[1]] == 0)[0])
        a.indices = np.delete(a.indices, where)
        a.data = np.delete(a.data, where, axis=0)
        a.indptr[1:] -= 1
    elif case == 'negative-level':
        fill = -1
    cfg = dict(config_version=2, exception_handling=0, block_format='ROW_MAJOR',
        solver=dict(solver='MULTICOLOR_ILU', scope='rejection', max_iters=1,
            ilu_sparsity_level=fill, block_ilu_backend='cusparse_legacy',
            reorder_cols_by_color=0, insert_diag_while_reordering=0))
    handles = []
    def make(kind, *args):
        h = ct.c_void_p()
        api.call(kind+'_create', ct.byref(h), *args)
        handles.append((kind, h))
        return h
    def ptr(x): return ct.c_void_p(x.ctypes.data)
    try:
        config = make('config', json.dumps(cfg).encode())
        resources = ct.c_void_p()
        api.call('resources_create_simple', ct.byref(resources), config)
        handles.append(('resources', resources))
        solver = ct.c_void_p()
        operation = 'solver_create'
        status = api.lib.AMGX_solver_create(ct.byref(solver), resources, 8193, config)
        if status == 0:
            handles.append(('solver', solver))
            if case == 'negative-level':
                raise AssertionError('Negative ILU level was accepted')
            matrix = make('matrix', resources, 8193)
            api.call('matrix_upload_all', matrix, a.shape[0]//2, len(a.indices), 2, 2,
                     ptr(a.indptr), ptr(a.indices), ptr(a.data), None)
            operation = 'solver_setup'
            status = api.lib.AMGX_solver_setup(solver, matrix)
        elif case != 'negative-level':
            raise AssertionError(f'Valid config rejected before input upload: {status}')
        # AMGX_RC_BAD_PARAMETERS / AMGX_ERR_BAD_PARAMETERS = 1.
        if status != 1:
            raise AssertionError(f'{operation}: expected BAD_PARAMETERS (1), got {status}')
        return dict(case=case, fill=fill, operation=operation, error_code=status,
                    native_messages=''.join(api.messages), status='passed')
    finally:
        for kind, h in reversed(handles):
            api.call(kind+'_destroy', h)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verbosity',type=int,choices=range(4),default=2)
    p.add_argument('--quick',action='store_true',help='One b8/ILU2 FP64 case for compute-sanitizer')
    p.add_argument('--zero-only',action='store_true',help='ILU0 regression on the unchanged baseline')
    p.add_argument('--reject-case',choices=('unsorted','duplicate','missing-diagonal','negative-level'))
    p.add_argument('--fill-level',type=int,default=2,help='Fill level for input-rejection tests only')
    p.add_argument('--fill-levels',type=int,nargs='+',help='Override numerical-oracle fill levels')
    p.add_argument('--blocks',type=int,nargs='+',help='Override numerical-oracle block sizes')
    p.add_argument('--fill-chain',action='store_true',help='32-block fixture with distinct levels up to 30')
    args=p.parse_args()
    api=API(args.library,args.verbosity)
    report=dict(library=str(args.library.resolve()),library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),results=[],status='running')
    try:
        if args.reject_case:
            report['results'].append(rejection_case(api,args.reject_case,args.fill_level))
            report['status']='passed'
            return
        for block in (args.blocks or ((8,) if args.quick else (1,2,4,8,16))):
            for fill in (args.fill_levels or ((0,) if args.zero_only else ((2,) if args.quick else (0,1,2,3)))):
                for mode in (('dDDI',) if args.quick else ('dDDI','dFFI')):
                    for permuted in ((True,) if args.quick else (False,True)):
                        row=gpu_case(api,block,fill,mode,permuted,args.fill_chain); report['results'].append(row)
                        write_report(args.output,report)
                        if args.verbosity>=1: print(f"[ILU(k) oracle] b{block} k{fill} {mode} permuted={permuted} PASS error={max(row['application_errors']):.3e}",flush=True)
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        api.call('finalize'); write_report(args.output,report)
        if args.verbosity>=1:
            print(f"[ILU(k) oracle] {report['status']}: {len(report['results'])} cases; {args.output}",flush=True)


if __name__=='__main__': main()
