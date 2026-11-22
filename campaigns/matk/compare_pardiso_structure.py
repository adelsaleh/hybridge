#!/usr/bin/env python3
"""Audit the full CSR permutation and compare exact coupling-distance statistics."""
import argparse
import json
import os
from pathlib import Path
import time

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matk-matplotlib')

import numpy as np
from common import load_system, system_fingerprint
from pardiso_ordering import load_permutation, reorder_csr


def log(args, message):
    if args.verbosity:
        print('[matK ordering audit] ' + message, flush=True)


def equal_chunks(left, right):
    return left.shape == right.shape and all(np.array_equal(left[k:k+4000000], right[k:k+4000000])
                                             for k in range(0, left.size, 4000000))


def distance_summary(hist):
    total = float(np.sum(hist))
    cumulative = np.cumsum(hist, dtype=np.float64)
    return dict(total=total,
                mean=float(np.dot(np.arange(hist.size, dtype=np.float64), hist) / total) if total else None,
                quantiles={str(q): int(np.searchsorted(cumulative, q*total)) for q in (.5, .9, .99)} if total else {},
                within={str(w): float(cumulative[min(w,hist.size-1)] / total) if total else None
                        for w in (4, 20, 100, 1000, 10000, 100000)})


def statistics(matrix, name, args):
    n = matrix.shape[0]
    hist = {key: np.zeros(n, dtype=np.float64) for key in
            ('offdiagonal_count', 'absolute_offdiagonal_mass', 'row_normalized_mass', 'strong_offdiagonal_count')}
    density = np.zeros((args.bins, args.bins), dtype=np.int64)
    strong_density = np.zeros_like(density)
    numerical_degrees = np.zeros(n, dtype=np.int32)
    row_spans = np.zeros(n, dtype=np.int32)
    block_mass = {str(b): 0. for b in (4,20,100)}
    stored_zeros = 0
    lower_bandwidth = upper_bandwidth = 0
    lower_profile = upper_profile = 0
    started = time.perf_counter()
    for first in range(0, n, args.chunk_rows):
        last = min(n, first + args.chunk_rows)
        begin, end = matrix.indptr[first], matrix.indptr[last]
        lengths = np.diff(matrix.indptr[first:last+1])
        offsets = matrix.indptr[first:last] - begin
        rows = np.repeat(np.arange(first,last,dtype=np.int32), lengths)
        cols = matrix.indices[begin:end]
        values = np.abs(matrix.data[begin:end])
        nz = values != 0
        numerical_degrees[first:last] = np.add.reduceat(nz.astype(np.int32), offsets)
        stored_zeros += int(np.count_nonzero(~nz))
        minimum = np.minimum.reduceat(np.where(nz, cols, n), offsets)
        maximum = np.maximum.reduceat(np.where(nz, cols, -1), offsets)
        active = numerical_degrees[first:last] > 0
        row_spans[first:last] = np.where(active, maximum-minimum, 0)
        row_numbers = np.arange(first,last)
        lower_profile += int(np.maximum(0, row_numbers[active]-minimum[active]).sum(dtype=np.int64))
        upper_profile += int(np.maximum(0, maximum[active]-row_numbers[active]).sum(dtype=np.int64))
        delta = cols - rows
        if np.any(nz):
            lower_bandwidth = max(lower_bandwidth, int(np.max(-delta[nz])))
            upper_bandwidth = max(upper_bandwidth, int(np.max(delta[nz])))
        distance = np.abs(delta)
        off = nz & (distance > 0)
        values[~off] = 0.
        row_sum = np.add.reduceat(values, offsets)
        row_max = np.maximum.reduceat(values, offsets)
        denominators = np.repeat(row_sum, lengths)
        normalized = np.divide(values, denominators, out=np.zeros_like(values), where=denominators>0)
        strong = off & (values >= .1*np.repeat(row_max, lengths))
        hist['offdiagonal_count'] += np.bincount(distance[off], minlength=n)
        hist['absolute_offdiagonal_mass'] += np.bincount(distance, weights=values, minlength=n)
        hist['row_normalized_mass'] += np.bincount(distance, weights=normalized, minlength=n)
        hist['strong_offdiagonal_count'] += np.bincount(distance[strong], minlength=n)
        cells = (rows.astype(np.int64)*args.bins//n)*args.bins + cols.astype(np.int64)*args.bins//n
        density += np.bincount(cells[nz], minlength=args.bins**2).reshape(args.bins,args.bins)
        strong_density += np.bincount(cells[strong], minlength=args.bins**2).reshape(args.bins,args.bins)
        for size in block_mass:
            local = rows//int(size) == cols//int(size)
            block_mass[size] += float(normalized[local].sum())
        if first == 0 or last == n or (first//args.chunk_rows) % 16 == 0:
            log(args, f'{name}: processed {last:,}/{n:,} rows ({time.perf_counter()-started:.1f} s)')
    summaries = {key: distance_summary(value) for key,value in hist.items()}
    normalization = summaries['row_normalized_mass']['total']
    result = dict(stored_nnz=int(matrix.nnz), numerical_nnz=int(numerical_degrees.sum(dtype=np.int64)),
                  stored_zeros=stored_zeros, lower_bandwidth=lower_bandwidth,
                  upper_bandwidth=upper_bandwidth, bandwidth=max(lower_bandwidth,upper_bandwidth),
                  lower_profile=lower_profile, upper_profile=upper_profile,
                  row_span_quantiles={str(q):float(np.quantile(row_spans,q)) for q in (.5,.9,.99)},
                  numerical_row_degree_quantiles={str(q):float(np.quantile(numerical_degrees,q)) for q in (0,.5,.9,1)},
                  mean_adjacent_row_degree_change=float(np.abs(np.diff(numerical_degrees)).mean()),
                  within_contiguous_block_row_normalized_mass={k:v/normalization for k,v in block_mass.items()},
                  distances=summaries, statistics_seconds=time.perf_counter()-started)
    return result, hist, density, strong_density, numerical_degrees


def plots(output, results, grids, histories, n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(2,2,figsize=(12,11),constrained_layout=True)
    for row, label in enumerate(('All numerical nonzeros', 'Strong off-diagonal couplings')):
        maximum = max(int(grids[name][row].max()) for name in grids)
        norm = LogNorm(vmin=1,vmax=max(2,maximum))
        for col,name in enumerate(('original','pardiso')):
            ax = axes[row,col]
            grid = grids[name][row]
            image = ax.imshow(np.ma.masked_equal(grid,0), origin='upper', cmap='magma', norm=norm,
                              extent=(0,n,n,0), interpolation='nearest')
            ax.set(title=f'{name.capitalize()}: {label}',xlabel='Column index',ylabel='Row index')
        fig.colorbar(image,ax=list(axes[row]),shrink=.85,label='Entries per pixel (exact bin counts)')
    fig.suptitle('matK: same coefficients, different row/column positions\nStrong = at least 10% of the largest off-diagonal magnitude in its row')
    fig.savefig(output/'structure.png',dpi=200)
    fig.savefig(output/'structure.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    points=np.unique(np.r_[1,np.geomspace(1,n-1,1200).astype(np.int64)])
    for ax,key,title in zip(axes,('offdiagonal_count','row_normalized_mass','strong_offdiagonal_count'),
                            ('All off-diagonal entries','Off-diagonal magnitude; each row weighted equally','Strong off-diagonal entries')):
        for name in ('original','pardiso'):
            hist=histories[name][key]
            cumulative=np.cumsum(hist)/np.sum(hist)
            ax.semilogx(points,100*cumulative[points],label=name.capitalize(),linewidth=2)
        ax.set(title=title,xlabel='Distance from diagonal |i−j|',ylabel='Cumulative fraction (%)',ylim=(0,101))
        ax.grid(alpha=.25);ax.legend()
    fig.savefig(output/'coupling_distances.png',dpi=200)
    fig.savefig(output/'coupling_distances.pdf')
    plt.close(fig)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--matrix',type=Path,default=Path(__file__).with_name('matK.h5'))
    cli.add_argument('--permutation',type=Path,required=True)
    cli.add_argument('--output-dir',type=Path,required=True)
    cli.add_argument('--verbosity',type=int,choices=range(4),default=2)
    cli.add_argument('--bins',type=int,default=768)
    cli.add_argument('--chunk-rows',type=int,default=16384)
    args=cli.parse_args()
    if args.bins<2 or args.chunk_rows<1: cli.error('Invalid bin/chunk count')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    original,rhs,info=load_system(args.matrix,'mat','auto',min(args.verbosity,1))
    fingerprint=system_fingerprint(original,rhs)
    p,metadata=load_permutation(args.permutation,fingerprint,original.shape[0])
    log(args,'Applying the exact benchmark reorder_csr implementation')
    changed=reorder_csr(original,p)
    log(args,'Building independent SciPy A[p,:][:,p] reference for every coefficient')
    reference=original[p,:][:,p].tocsr()
    reference.sort_indices()
    audit={key:equal_chunks(getattr(changed,key),getattr(reference,key)) for key in ('indptr','indices','data')}
    del reference
    audit['diagonal_exactly_permuted']=np.array_equal(changed.diagonal(),original.diagonal()[p])
    audit['matrix_changed']=not (equal_chunks(original.indptr,changed.indptr) and
                                equal_chunks(original.indices,changed.indices) and equal_chunks(original.data,changed.data))
    x=np.random.default_rng(20261030).standard_normal(original.shape[0])
    ax=(original@x)[p]
    audit['matvec_relative_error']=float(np.linalg.norm(changed@x[p]-ax)/np.linalg.norm(ax))
    recovered=np.empty_like(x);recovered[p]=x[p]
    audit['vector_coordinate_roundtrip_exact']=bool(np.array_equal(recovered,x))
    audit['rhs_norm_relative_error']=float(abs(np.linalg.norm(rhs[p])-np.linalg.norm(rhs))/np.linalg.norm(rhs))
    audit={key:bool(value) if isinstance(value,np.bool_) else value for key,value in audit.items()}
    assert all(audit[key] for key in ('indptr','indices','data','diagonal_exactly_permuted','matrix_changed','vector_coordinate_roundtrip_exact'))
    assert audit['matvec_relative_error']<1e-12
    log(args,'Full CSR comparison passed: row pointers, column indices and FP64 coefficients match exactly')
    results={};histories={};grids={};degrees={}
    for name,matrix in (('original',original),('pardiso',changed)):
        results[name],histories[name],density,strong,degrees[name]=statistics(matrix,name,args)
        grids[name]=(density,strong)
    audit['row_degree_distribution_preserved']=bool(np.array_equal(degrees['pardiso'],degrees['original'][p]))
    report=dict(status='passed',matrix=info,system_sha256=fingerprint,permutation=metadata,
                audit=audit,statistics=results,job_id=os.environ.get('SLURM_JOB_ID'),
                definitions={'strong':'|a_ij| >= 0.1 * max_{k != i}|a_ik|; i != j; numerical nonzeros only',
                             'row_normalized_mass':'Sum |a_ij|/sum_{k != i}|a_ik|; diagonal excluded; each nonempty off-diagonal row has unit weight',
                             'distance':'absolute scalar row-column index difference; all entries counted, no sampling',
                             'diagonal_note':'A simultaneous row/column permutation preserves the diagonal values and 2-norm condition number; no matching transformation exported'})
    (args.output_dir/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    arrays={f'{name}_{key}':values for name,h in histories.items() for key,values in h.items()}
    arrays.update({f'{name}_density_{i}':grid for name,group in grids.items() for i,grid in enumerate(group)})
    np.savez_compressed(args.output_dir/'histograms.npz',**arrays)
    plots(args.output_dir,results,grids,histories,original.shape[0])
    lines=['matK PARDISO ordering audit','All statistics use every numerical nonzero; diagonal excluded from coupling distances.',json.dumps(audit,indent=2),'',f"{'Metric':<52} {'Original':>16} {'PARDISO':>16}"]
    for label,path in [('Bandwidth',('bandwidth',)),('Median row span',('row_span_quantiles','0.5')),
                       ('Median off-diagonal distance',('distances','offdiagonal_count','quantiles','0.5')),
                       ('Median row-normalized coupling distance',('distances','row_normalized_mass','quantiles','0.5')),
                       ('Median strong-coupling distance',('distances','strong_offdiagonal_count','quantiles','0.5'))]:
        vals=[]
        for name in ('original','pardiso'):
            v=results[name]
            for key in path:v=v[key]
            vals.append(float(v))
        lines.append(f'{label:<52} {vals[0]:16,.3f} {vals[1]:16,.3f}')
    for width in (20,100,1000,10000):
        vals=[100*results[name]['distances']['row_normalized_mass']['within'][str(width)] for name in ('original','pardiso')]
        lines.append(f'{"Row-normalized mass within ±"+str(width)+" (%)":<52} {vals[0]:16.6f} {vals[1]:16.6f}')
    text='\n'.join(lines)+'\n';(args.output_dir/'comparison.txt').write_text(text);print(text,flush=True)


if __name__=='__main__':
    main()
