# High-resolution equilibrium SUPG sweep — 2026-08-20

This companion check uses the `h=0.02`, P4, 690,677-dof equilibrium documented
in `equilibrium_stationarity_report_20260820_h002_p4.md`.  Each run uses 16 MPI
ranks, `dt=0.025`, final time `T=0.5`, direct MUMPS solves, and zero optional
CIP/flux stabilization.  Only `supg_scale` changes.

| SUPG scale | rho relative L2 change | advective defect relative | rho minimum | mass drift | energy drift |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 2.9567e-4 | 1.710e-2 | -1.613e-3 | 2.624e-14 | 3.332e-13 |
| 0.1 | 2.091e-4 | 9.265e-3 | -1.072e-3 | 1.788e-14 | 2.562e-13 |
| 1.0 | 1.515568e-4 | 4.465e-3 | -6.563e-5 | 1.103e-14 | 2.363e-13 |

The equilibrium remains stationary without SUPG: relative density change is
below `3e-4` through `T=0.5`, and mass and energy remain at roundoff.  Reducing
SUPG monotonically increases advective defect, undershoot, and density drift.
Thus standard SUPG damps high-frequency discrete transport error; it does not
manufacture the observed equilibrium.

Persistent outputs:

- `projects/diocotron/runs/guiding_center/dolfinx_torsion_supg/equilibrium_h002_p4_supg0_dt0025_T05_20260820`
- `projects/diocotron/runs/guiding_center/dolfinx_torsion_supg/equilibrium_h002_p4_supg01_dt0025_T05_20260820`
- `projects/diocotron/runs/guiding_center/dolfinx_torsion_supg/equilibrium_h002_p4_dt0025_T05_20260820`
