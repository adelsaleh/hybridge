#!/usr/bin/env python3
"""Show and verify a real PARDISO ordering of a reproducible 15x15 matrix."""
import argparse
import json
import os
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matk-matplotlib')

import numpy as np
from scipy.sparse import csr_matrix

from pardiso_ordering import compute_permutation, reorder_csr, save_permutation, load_permutation
from common import system_fingerprint


def terminal_matrix(matrix):
    lines = ['     ' + ''.join(f'{i+1:>7}' for i in range(matrix.shape[1]))]
    for i, row in enumerate(matrix):
        values = ['0' if value == 0 else f'{value:.0e}'.replace('e+0', 'e+').replace('e-0', 'e-')
                  for value in row]
        lines.append(f'{i+1:>3}  ' + ''.join(f'{value:>7}' for value in values))
    return '\n'.join(lines)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output-dir', type=Path, required=True)
    cli.add_argument('--seed', type=int, default=20261030)
    cli.add_argument('--verbosity', type=int, choices=range(4), default=2)
    args = cli.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    # A non-self-inverse permutation makes a direction error observable.
    for seed in range(args.seed, args.seed + 20):
        rng = np.random.default_rng(seed)
        mask = rng.random((15, 15)) < 0.16
        mask[np.arange(15), np.arange(15)] = True
        mask[np.arange(15), (np.arange(15)+1) % 15] = True
        dense = np.where(mask, rng.choice([-1., 1.], (15, 15)) *
                         10.**rng.uniform(-6, 6, (15, 15)), 0.)
        matrix = csr_matrix(dense)
        permutation, metadata = compute_permutation(matrix, threads=1, verbosity=3 if args.verbosity==3 else 0)
        inverse = np.argsort(permutation)
        trials.append(dict(seed=seed, self_inverse=bool(np.array_equal(permutation, inverse))))
        if not trials[-1]['self_inverse']:
            break
    else:
        raise RuntimeError('No direction-discriminating example found in the stated seed range')
    reordered = reorder_csr(matrix, permutation)
    reference = dense[np.ix_(permutation, permutation)]
    # Construct the permutation matrix independently and multiply dense arrays.
    p_matrix = np.eye(15)[permutation]
    checks = dict(csr_matches_numpy_exactly=bool(np.array_equal(reordered.toarray(), reference)),
                  matches_P_A_PT_exactly=bool(np.array_equal(reference, p_matrix @ dense @ p_matrix.T)),
                  diagonal_is_only_permuted=bool(np.array_equal(np.diag(reference), np.diag(dense)[permutation])),
                  inverse_direction_is_different=bool(not np.array_equal(reference, dense[np.ix_(inverse, inverse)])))
    x = rng.standard_normal(15)
    b = dense @ x
    checks['matvec_relative_error'] = float(np.linalg.norm(reference @ x[permutation] - b[permutation]) / np.linalg.norm(b))
    recovered = np.empty_like(x)
    recovered[permutation] = x[permutation]
    checks['solution_coordinate_roundtrip_exact'] = bool(np.array_equal(recovered, x))
    assert all(checks[key] for key in checks if key != 'matvec_relative_error')
    assert checks['matvec_relative_error'] < 1e-12
    metadata['system_sha256'] = system_fingerprint(matrix, b)
    archive = args.output_dir / 'example.permutation.npz'
    if archive.exists():
        existing,_ = load_permutation(archive,metadata['system_sha256'],15)
        if not np.array_equal(existing,permutation):
            raise FileExistsError('Output directory contains a different permutation; choose a new directory')
    else:
        save_permutation(archive, permutation, metadata)
    nonzero = np.transpose(np.nonzero(dense))
    strongest = sorted(nonzero, key=lambda ij: abs(dense[tuple(ij)]), reverse=True)[:5]
    moves = [dict(old_row=int(i+1), old_column=int(j+1), new_row=int(inverse[i]+1),
                  new_column=int(inverse[j]+1), value=float(dense[i, j])) for i, j in strongest]
    result = dict(seed=seed, trials=trials, size=15, nnz=matrix.nnz,
                  permutation_new_to_old_one_based=(permutation+1).tolist(),
                  permutation_old_to_new_one_based=(inverse+1).tolist(),
                  original=dense.tolist(), reordered=reference.tolist(),
                  checks=checks, strongest_entry_moves=moves, pardiso=metadata,
                  absolute_nonzero_range=[float(np.min(np.abs(dense[dense != 0]))),
                                          float(np.max(np.abs(dense)))])
    (args.output_dir / 'example.json').write_text(json.dumps(result, indent=2)+'\n')
    text = (f'Seed {seed}; actual PARDISO phase 11, mtype 11; 1-based labels.\n'
            'Display rounds to one significant digit; all checks use full FP64 values.\n'
            'New row/column -> old row/column: '+str((permutation+1).tolist())+'\n\n'
            'Original A\n'+terminal_matrix(dense)+'\n\n'
            'Reordered C = A[p,p]\n'+terminal_matrix(reference)+'\n\n'
            'Checks\n'+json.dumps(checks, indent=2)+'\n')
    (args.output_dir / 'example.txt').write_text(text)
    print(text if args.verbosity else f"15x15 ordering audit passed; matvec error {checks['matvec_relative_error']:.3e}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    norm = SymLogNorm(linthresh=1e-6, vmin=-1e6, vmax=1e6)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for ax, a, title in zip(axes, (dense, reference), ('Original', 'PARDISO reordered')):
        im = ax.imshow(a, cmap='coolwarm', norm=norm)
        ax.set(title=title, xlabel='Column (1-based)', ylabel='Row (1-based)')
        ax.set_xticks(range(15), range(1,16), fontsize=8)
        ax.set_yticks(range(15), range(1,16), fontsize=8)
        for i, j in zip(*np.nonzero(a)):
            label = f'{a[i,j]:.0e}'.replace('e+0','e+').replace('e-0','e-')
            ax.text(j,i,label,ha='center',va='center',fontsize=6,
                    color='white' if abs(a[i,j]) > 1e2 else 'black')
    fig.colorbar(im, ax=axes, shrink=.7, label='Signed coefficient (symmetric logarithmic scale)')
    fig.suptitle(f'Real PARDISO export; seed {seed}; 15 × 15; {matrix.nnz} nonzeros')
    fig.savefig(args.output_dir / 'example.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
