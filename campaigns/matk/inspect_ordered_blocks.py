#!/usr/bin/env python3
"""Count consecutive BSR blocks exactly before allocating dense block storage."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.sparse._sparsetools import csr_count_blocks
from common import load_system, system_fingerprint, write_report, provenance
from pardiso_ordering import load_permutation, reorder_csr
from benchmark_pardiso import print_fields, print_table


def block_profile(matrix, block_size, restart=1000, numerical_nnz=None):
    """Exact stored-pattern count; matches SciPy CSR->BSR including explicit zeros.

    Memory estimates exclude opaque cuSPARSE workspace and allocator overhead.
    No dense block value array is allocated by this function.
    """
    n, m = matrix.shape
    if n != m or block_size < 1 or n % block_size:
        raise ValueError('Square dimensions must be divisible by block size')
    begin = time.perf_counter()
    blocks = int(csr_count_blocks(n, m, block_size, block_size, matrix.indptr, matrix.indices))
    slots = blocks * block_size**2
    values_bytes = slots * matrix.dtype.itemsize
    bsr_bytes = values_bytes + 4 * (blocks + n // block_size + 1)
    vector_bytes = n * matrix.dtype.itemsize
    # FGMRES stores both V and preconditioned Z basis vectors.
    basis_bytes = (2 * restart + 1) * vector_bytes
    lower = 2 * bsr_bytes + basis_bytes
    planning = 4 * bsr_bytes + basis_bytes + 32 * vector_bytes + 8 * (restart + 1)**2
    return dict(block_size=block_size, block_rows=n // block_size, stored_blocks=blocks,
                scalar_slots=slots, original_stored_nnz=int(matrix.nnz),
                stored_occupancy=float(matrix.nnz / slots) if slots else 1.,
                numerical_occupancy=(float(numerical_nnz / slots) if slots else 1.) if numerical_nnz is not None else None,
                padding_entries=slots-int(matrix.nnz), storage_expansion=float(slots / max(matrix.nnz, 1)),
                values_bytes=values_bytes, bsr_bytes=bsr_bytes, fgmres_basis_bytes=basis_bytes,
                device_known_minimum_bytes=lower, device_planning_bytes=planning,
                int32_value_count_safe=slots <= np.iinfo(np.int32).max,
                count_seconds=time.perf_counter()-begin)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--matrix', type=Path, default=Path(__file__).with_name('matK.h5'))
    p.add_argument('--permutation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--block-sizes', nargs='+', type=int, default=[2, 4, 8, 16])
    p.add_argument('--restart', type=int, default=1000)
    p.add_argument('--verbosity', type=int, choices=range(4), default=2)
    args = p.parse_args()
    matrix, rhs, info = load_system(args.matrix, verbosity=min(args.verbosity, 1))
    report = dict(status='counting', matrix=info, machine=provenance(), blocks=[],
                  restart=args.restart, memory_note='Known minimum: A + LU + V/Z bases. Planning: four BSR copies + bases + vectors. Opaque cuSPARSE workspace and allocator overhead are unknown.')
    report['system_sha256'] = system_fingerprint(matrix, rhs)
    perm, metadata = load_permutation(args.permutation, report['system_sha256'], matrix.shape[0])
    report['permutation'] = metadata
    begin = time.perf_counter()
    ordered = reorder_csr(matrix, perm)
    report['permutation_application_seconds'] = time.perf_counter()-begin
    del matrix
    numerical_nnz = sum(int(np.count_nonzero(ordered.data[k:k+4_000_000])) for k in range(0, ordered.nnz, 4_000_000))
    report['numerical_nnz'] = numerical_nnz
    write_report(args.output, report)
    for size in args.block_sizes:
        print_fields(args.verbosity, 1, 'Counting PARDISO consecutive blocks', [('Block size', size)])
        row = block_profile(ordered, size, args.restart, numerical_nnz)
        report['blocks'].append(row)
        write_report(args.output, report)
        print_fields(args.verbosity, 2, 'Exact BSR storage', [(key, value) for key,value in row.items()])
    report['status'] = 'completed'
    write_report(args.output, report)
    print_table(args.verbosity, 0, 'PARDISO consecutive-block preflight',
                ['Block', 'Blocks', 'Occupancy', 'BSR GiB', 'GPU min GiB', 'GPU plan GiB', 'int32 safe'],
                [[r['block_size'], f"{r['stored_blocks']:,}", f"{r['stored_occupancy']:.2%}",
                  f"{r['bsr_bytes']/2**30:.2f}", f"{r['device_known_minimum_bytes']/2**30:.2f}",
                  f"{r['device_planning_bytes']/2**30:.2f}", r['int32_value_count_safe']] for r in report['blocks']])
    print_fields(args.verbosity, 0, 'Preflight report', [('JSON', args.output)])


if __name__ == '__main__':
    main()
