# Diocotron Run Policy

## MUMPS ranks for the torsion reduced optimizer

When running
`projects/diocotron/dolfinx/torsion/optimization/homotopy.py` with
`--linear-solver mumps` on the 20-physical-core Intel Xeon w5-3535X
workstation, use this preliminary MPI-rank map:

| Mesh size | order 2 | order 3 | order 4 | order 5 | order 6 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 1 | 1 | 1 | 1 | 1 |
| 0.15 | 1 | 1 | 1 | 2 | 4 |
| 0.075 | 2 | 4 | 4 | 4 | 4 |
| 0.05 | 4 | 4 | 8 | 8 | 8 |
| 0.03 | 4 | 8 | 8 | 12 | 16 |

The 0.30, 0.15, and 0.075 rows come from shortened measured runs. The 0.05
and 0.03 rows are extrapolations; validate them before long production runs.
When the exact pair is absent, use the global finite-element DOF count:

- Below 20,000 DOFs: 1 rank.
- 20,000--40,000 DOFs: 2 ranks, or 4 for order 5--6.
- 40,000--120,000 DOFs: 4 ranks.
- 120,000--250,000 DOFs: 8 ranks.
- Above 250,000 DOFs: start with 8 ranks and compare 12 and 16.

Use physical cores and one numerical-library thread per MPI rank, for example:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpirun --bind-to core --map-by core -n RANKS ...
```

Prefer the smaller rank count when adjacent choices are within about 10%.
Rank-zero initialization, scalar reductions, and MUMPS communication impose a
serial/communication floor. In particular, use 4 rather than 8 ranks for the
measured `(mesh_size=0.075, order=5 or 6)` cases unless a representative long
run shows otherwise. Re-benchmark on machines with a different CPU, NUMA
topology, MPI implementation, or MUMPS build.
