# High-resolution equilibrium handoff and stationarity — 2026-08-20

## Conclusion

The `h=0.02`, P4 semilinear state is a strong numerical equilibrium for the
DOLFINx guiding-center discretization.  Its relative density change is
`1.103797e-4` at `T=0.1` and `1.515568e-4` at `T=0.5`; mass and energy remain
constant to roundoff.  Halving the timestep does not reduce the drift, so the
small remaining defect is spatial/interface-discrete rather than temporal.

## Equilibrium solve

- Mesh: 86,112 triangles, smooth-star boundary with 500 samples.
- Space: continuous Lagrange P4, 690,677 global dofs, quadrature degree 16.
- Initialization study: 32-point coarse H-minus-one grid followed by four
  17-by-17 refinements, totaling 1,650 objective evaluations.
- Selected refined-grid seed: `c1=4.239558e-2`, `c2=4.771926e-2`.
- Homotopy: 8 accepted stages, 27 Newton corrections, no rejected stages.
- Reduced optimization: 80 iterations after calibrating the initial trust
  radius to avoid oversized threshold trials.
- Final thresholds: `c1=4.1518319704e-2`, `c2=4.7838503207e-2`,
  `eps=5.0561468021e-4`.
- Final Newton residual: `1.000455e-14` against a requested `1e-12` tolerance.
- Final status: `CONVERGED_CERTIFIED_SUBBAND`.
- Certified area fraction: `0.515073`; certified leakage: exactly zero.
- Soft missing area stalls because the torsion-designed target and final
  semilinear equilibrium band do not have exactly equal areas; it is not used
  as a stationarity failure criterion.
- Total production solve time: 933.405 seconds on 20 MPI ranks.

Checkpoint:

`projects/diocotron/runs/scratch/torsion_reduced_optimization_homotopy/equilibrium_h002_p4_converged_gridseed_20260820/out/equilibrium.npz`

All checkpoint coordinates, density values, and potential values are finite.
The checkpoint contains 690,677 nodal values for each scalar field.

## Solver and MPI calibration

The optimizer used one numerical-library thread per MPI rank and physical-core
binding.  Identical representative MUMPS workloads gave:

| MPI ranks | total seconds |
|---:|---:|
| 4 | 188.722 |
| 8 | 155.521 |
| 12 | 143.807 |
| 16 | 140.918 |
| 20 | 138.045 |

CG/GAMG accelerated the coercive stiffness and homotopy systems but failed on
the first reduced trial Jacobian with PETSc reason `-8`.  GMRES/GAMG failed on
the same trial system with reason `-3`.  Robust full-path optimization
therefore used MUMPS.

Guiding-center MUMPS timing used six steps at `dt=0.025`, with the first two
excluded:

| MPI ranks | median seconds/step |
|---:|---:|
| 4 | 5.158459 |
| 8 | 4.304405 |
| 12 | 4.193004 |
| 16 | 4.148645 |
| 20 | 4.185092 |

Sixteen ranks were selected.  Numerical diagnostics agreed across rank counts
to printed precision.

## Checkpoint handoff

The guiding-center runner loaded the exact P4 mesh and nodal density, then
re-solved Poisson for the transport potential.  Relative differences from the
saved optimized potential were:

- relative L2: `1.606786e-7`;
- relative H1: `7.278514e-7`.

The initial relative advective defect was `6.201e-3`.  After one transport
step it fell to `4.088e-3` and remained near `4.5e-3`; this does not translate
into appreciable density evolution.

## Stationarity results

All cases used SUPG, zero optional CIP/flux stabilization, direct MUMPS solves,
no plotting, and 16 MPI ranks.

| dt | final time | rho relative L2 change | mass drift | energy drift | rho Linf change |
|---:|---:|---:|---:|---:|---:|
| 0.0250 | 0.1 | 1.103797e-4 | -2.282145e-15 | 1.252039e-13 | 1.438872e-3 |
| 0.0125 | 0.1 | 1.210842e-4 | -5.325004e-15 | 8.848607e-14 | 1.713061e-3 |
| 0.0250 | 0.5 | 1.515568e-4 | 1.103037e-14 | 2.362656e-13 | 2.002513e-3 |

The drift grows sublinearly and is already flattening by `T=0.5`.  Relative to
the earlier `h=0.05`, P2 study, the new density drift is about 64 times smaller
at `T=0.1` and 121 times smaller at `T=0.5`.

Persistent transport outputs:

- `projects/diocotron/runs/guiding_center/dolfinx_torsion_supg/equilibrium_h002_p4_dt00125_T01_20260820`
- `projects/diocotron/runs/guiding_center/dolfinx_torsion_supg/equilibrium_h002_p4_dt0025_T05_20260820`

## Reproduction commands

The final transport validation was run as:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to core --map-by core -n 16 \
  python \
  projects/diocotron/dolfinx/guiding_center/supg.py \
  --equilibrium projects/diocotron/runs/scratch/torsion_reduced_optimization_homotopy/equilibrium_h002_p4_converged_gridseed_20260820/out/equilibrium.npz \
  --order 4 --dt 0.025 --num-steps 20 \
  --flux-stabilization 0 \
  --poisson-linear-solver mumps --transport-linear-solver mumps \
  --no-plot --no-hold-final --tau-report-every 0
```
