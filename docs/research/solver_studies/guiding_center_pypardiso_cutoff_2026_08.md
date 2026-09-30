# Guiding-Center PyPardiso Cutoff Study, 2026-08-23

This report records the host-solver measurements used to add the
`pypardiso-cutoff` transport policy to
`scripts/guiding_center/run_guiding_center_cases.py`. The result is
machine-specific guidance, not a universal sparse-solver crossover.

## Environment and method

- CPU: Intel Xeon w5-3535X, 20 physical cores, one NUMA node.
- Runtime: Conda environment `fenicsx-dgfem`, Python 3.13.14, NumPy 2.4.6,
  SciPy 1.18.0, Numba 0.66.0, and PyPardiso 0.4.7.
- Threads: `MKL_NUM_THREADS=20`, `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `NUMBA_NUM_THREADS=20`.
- Case: Gaussian-annulus diocotron, `k=3`, order 6, and `dt=0.1`.
- Matched transport comparison: nonsymmetric PyPardiso versus BiCGSTAB with
  medium COLAMD ILU (`drop_tol=1e-5`, `fill_factor=5`) and reuse of the first
  preconditioner. Both sides used the same Numba assembly and reusable
  `pypardiso-spd` Poisson solve.
- The 30k, 50k, and 100k rows average three accepted steps. The 202k row is a
  bounded one-step crossover probe.

## Results

| Triangles | Reduced transport DOFs | PyPardiso total / step | Iterative total / step | Result |
| ---: | ---: | ---: | ---: | --- |
| 37,233 | 389,375 | 4.886 s | 5.269 s | PyPardiso 7.3% faster |
| 50,676 | 530,264 | 6.538 s | 7.466 s | PyPardiso 12.4% faster |
| 113,894 | 1,193,136 | 15.290 s | 17.885 s | PyPardiso 14.5% faster |
| 202,256 | 2,120,020 | 27.724 s | 26.372 s | iterative 4.9% faster |

The corresponding global transport-solve portions were 1.702/2.141 s,
2.332/3.259 s, 5.430/8.111 s, and 10.273/9.643 s for
PyPardiso/iterative, respectively. Accepted mass, energy, and instability
metrics agreed to the precision printed by the runner; independently checked
transport residuals were between roughly `2e-16` and `9e-14` in the
three-step comparisons.

The untouched 30k high-fill upwind-SCC preset was also attempted, but it did
not complete three steps within five minutes and was manually capped. It is
not included in the ranking. The completed medium-ILU pairs are the cleaner
one-variable solver comparison.

## Selected policy

The configured ceiling is **1,200,000 reduced transport trace DOFs**. It
includes the largest three-step level where PyPardiso won and switches before
the measured 2.12-million-DOF crossover region. Above the ceiling, the new
preset falls back to the existing medium COLAMD-ILU BiCGSTAB path. The policy
uses reduced trace DOFs instead of triangle count so polynomial order and
boundary reduction are represented directly.

The fixed Poisson operator always uses `pypardiso-spd`. The backend now calls
PyPardiso's public `factorize()` and `solve()` phases explicitly, keeps the SPD
factorization separate from nonsymmetric transport factors, and records
`factorization_time` plus `factorization_reused`. In the post-change 30k auto
run, all recorded Poisson solves reported reused factors with zero new
factorization time; the native warm solve took about 0.119 s per right-hand
side. The first transport solve reported a 0.878 s factorization and a 0.218 s
native solve.

## Usage

```bash
env MKL_NUM_THREADS=20 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=20 conda run -n fenicsx-dgfem python \
  scripts/guiding_center/run_guiding_center_cases.py \
  --preset diocotron_gaussian_annulus_k3_p6_30k_numba_pypardiso_auto
```

Override the measured ceiling with
`--transport-pypardiso-max-trace-dofs`; select a fixed configured solver with
`--transport-solver-policy fixed`.

## Limitations

The 2.12-million-DOF point has only one accepted step. Reused ILU quality can
degrade as transport coefficients evolve, as it did at the 30k--100k levels,
so representative later-time checks remain appropriate for very large runs.
Re-benchmark the cutoff after changing CPU, MKL/PyPardiso build, thread count,
trace basis, ordering, polynomial order, or transport physics.
