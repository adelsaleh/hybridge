#!/usr/bin/env python3
"""FP64 round-off floor of the relative residual of cached ADR trace systems.

A computed true residual ``b - A x`` carries rounding of about ``eps * || |A| |x| ||``, so a fixed
relative target ``||b - A x|| / ||b||`` cannot be met below

    floor = eps * || |A| |x| ||_2 / ||b||_2.

For each selected exported system (``systems.csv`` of the H200 tuned campaign) this solves with
HYBRIDGE's PyPardiso backend, then records the floor and the direct solve's own residual. CPU
only; no assembly, JIT or GPU. Keep the systems small (about 0.5M DOFs) when run on a login node.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybridge.linalg.bsr import face_dense_to_bsr
from hybridge.linalg.direct import clear_pypardiso_cache, solve_pypardiso_system

SYSTEMS = ROOT/'docs/campaigns/adr_h200_tuned/records/systems.csv'
OUTPUT = ROOT/'docs/research/solver_studies/adr_scaling_2026_09_17/h200_tuned/fp64_floor.csv'
# (geometry, case, p, level): the square classes at L1--L3, the annulus classes at their smallest meshes.
DEFAULT = ([('square', c, p, l) for c in ('trigonometric', 'advection_dominated', 'variable_velocity', 'anisotropic')
            for p in (2, 6, 9) for l in ('L1', 'L2', 'L3') if not (p == 9 and l == 'L3')]
           + [('annulus', 'cellular7_weak', p, l) for p in (2, 9) for l in ('L1', 'L2', 'L3') if not (p == 9 and l == 'L3')]
           + [('annulus', 'stress_main_orthogonal', 2, l) for l in ('L4', 'L4b')])
FIELDS = ('geometry', 'case', 'p', 'level', 'trace_dofs', 'system_id', 'rhs_norm', 'solution_norm',
          'fp64_floor', 'floor_per_dof', 'direct_relative_residual')


def floor_row(system):
    """Solve one cached system and return its floor record."""
    cache = Path(system['cache'])
    blocks, neighbors, rhs = (np.load(cache/f'system_{name}.npy') for name in ('blocks', 'neighbors', 'rhs'))
    b = rhs.reshape(-1)
    matrix = face_dense_to_bsr(blocks, neighbors).tocsr()
    x = solve_pypardiso_system(matrix, b, matrix_type='nonsymmetric').x
    clear_pypardiso_cache()
    b_norm = float(np.linalg.norm(b))
    floor = float(np.finfo(np.float64).eps*np.linalg.norm(abs(matrix)@abs(x))/b_norm)
    return dict(geometry=system['geometry'], case=system['case'], p=system['p'], level=system['level'],
                trace_dofs=b.size, system_id=system['system_id'], rhs_norm=b_norm,
                solution_norm=float(np.linalg.norm(x)), fp64_floor=floor, floor_per_dof=floor/b.size,
                direct_relative_residual=float(np.linalg.norm(b-matrix@x)/b_norm))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--systems', type=Path, default=SYSTEMS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    wanted = {(g, c, str(p), l) for g, c, p, l in DEFAULT}
    chosen = {}
    for system in csv.DictReader(args.systems.open()):
        key = (system['geometry'], system['case'], system['p'], system['level'])
        if key in wanted and system['cache'] and key not in chosen:
            chosen[key] = system
    rows = []
    for key in sorted(chosen, key=lambda k: (k[0], k[1], int(k[2]), k[3])):
        rows.append(floor_row(chosen[key]))
        r = rows[-1]
        print(f"{r['case']:24s} p={r['p']} {r['level']:4s} dofs={r['trace_dofs']:8d} floor={r['fp64_floor']:.2e} "
              f"direct={r['direct_relative_residual']:.2e}", flush=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {args.output} ({len(rows)} systems)')


if __name__ == '__main__':
    main()
