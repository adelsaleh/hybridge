# Optimized guiding-center validation of the retained P4 equilibrium (2026-08-20)

## Conclusion

The retained optimizer state is a numerical guiding-center equilibrium. On the
86,112-cell, 690,677-dof P4 mesh, low-SUPG transport changes the density by only
`1.886379e-4` in relative L2 through `T=0.5`. With both SUPG and CIP disabled,
the change remains only `2.667529e-4`. Halving the timestep does not reduce
the short-time drift, which identifies the remaining defect as a
spatial/interface-discrete mismatch rather than time-integration error.

Checkpoint:

`projects/diocotron/runs/scratch/torsion_reduced_optimization_homotopy/optimized_h002_p4_cached_certstop_validation_20260820/out/equilibrium.npz`

## Handoff checks

The checkpoint has a certified semilinear residual of `1.008511e-14`. After
the guiding-center code independently solves Poisson from the saved density,
the new potential differs from the saved potential by

- relative L2: `9.741558e-8`;
- relative H1: `6.799822e-7`.

The initial relative advective defect is `5.846e-3`, a small finite-element
interface defect.

## Stationarity results

All runs use 20 physical-core MPI ranks, P4, quadrature degree 16, zero CIP,
BiCGStab with block-Jacobi ILU(0), CG/BoomerAMG Poisson, and MUMPS fallback.

| SUPG scale | dt | final time | relative density change | mass drift | energy drift | advective defect |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.0250 | 0.1 | `1.186168e-4` | `-1.00e-13` | `-1.82e-12` | `7.038e-3` |
| 0.1 | 0.0125 | 0.1 | `1.292903e-4` | `-1.87e-14` | `-3.76e-12` | `8.112e-3` |
| 0.1 | 0.0250 | 0.5 | `1.886379e-4` | `-8.41e-13` | `7.18e-12` | `8.440e-3` |
| 0.0 | 0.0250 | 0.5 | `2.667529e-4` | `-9.93e-12` | `-6.56e-12` | `1.526e-2` |

The zero-stabilization control shows that SUPG is not manufacturing the
stationarity result. Low SUPG reduces grid-scale defect and undershoot, but the
equilibrium is already stationary without it.

Persistent runs:

- `projects/diocotron/runs/guiding_center/dolfinx_torsion_supg/new_equilibrium_supg01_dt0025_T05_optimized_20260820`
- `projects/diocotron/runs/guiding_center/dolfinx_torsion_supg/new_equilibrium_supg0_dt0025_T05_optimized_20260820`
- `projects/diocotron/runs/guiding_center/dolfinx_torsion_supg/new_equilibrium_supg01_dt00125_T01_optimized_20260820`

## Solver and runtime optimization

The transport solve, not Poisson or diagnostics, was the original bottleneck.
A representative MUMPS step spent about `3.75 s` in transport factorization,
`0.23 s` in matrix assembly, `0.06 s` in cached Poisson, and `0.09 s` in
diagnostics.

Measured solver progression:

| configuration | median seconds/step | relative speedup |
|---|---:|---:|
| MUMPS, 16 ranks | 4.151251 | 1.00x |
| BiCGStab/block-Jacobi ILU(1), 16 ranks | 0.955828 | 4.34x |
| BiCGStab/block-Jacobi ILU(1), 20 ranks | 0.795967 | 5.22x |
| BiCGStab/block-Jacobi ILU(0), 20 ranks | 0.489500 | 8.48x |
| final code, fixed storage and sampled diagnostics | 0.403696 | 10.28x |

The ILU(1) MPI scaling study gave `1.524867`, `1.080105`, `0.955828`, and
`0.795967 s/step` at 8, 12, 16, and 20 ranks respectively. Scaling never
reversed before the 20-physical-core limit, so 20 ranks is selected for this
P4 problem.

The optimized code now:

- makes iterative-first/MUMPS-fallback the default `auto` policy;
- retains `--solver-preset legacy` for direct-first behavior;
- defaults transport to warm-started BiCGStab/block-Jacobi ILU(0);
- defaults to the validated low-SUPG scale `0.1` with optional CIP disabled;
  both remain explicit CLI overrides;
- reassembles the changing transport operator into fixed PETSc matrix/vector
  storage and reuses symbolic ordering/fill;
- permanently switches to the next fallback after a solver kind fails;
- supports `--diagnostics-every N`, always evaluating the final step.

MUMPS and optimized iterative transport produce the same `T=0.1` relative
density change to eight significant digits. The iterative linear-solver drift
in mass and energy stays between approximately `1e-14` and `1e-11`.

## Recommended command

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 20 \
  python \
  projects/diocotron/dolfinx/guiding_center/supg.py \
  --equilibrium projects/diocotron/runs/scratch/torsion_reduced_optimization_homotopy/optimized_h002_p4_cached_certstop_validation_20260820/out/equilibrium.npz \
  --order 4 --quad-degree 16 --dt 0.025 --num-steps 20 \
  --diagnostics-every 4 --supg-scale 0.1 --flux-stabilization 0 \
  --no-plot --no-hold-final --tau-report-every 0
```

