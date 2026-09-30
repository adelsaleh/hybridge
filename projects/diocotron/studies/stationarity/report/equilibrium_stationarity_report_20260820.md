# Equilibrium handoff and stationarity check — 2026-08-20

## Equilibrium

- Optimizer: `torsion_reduced_optimization_homotopy.py`
- Mesh size: `0.05`
- Lagrange order: `2`
- Cells / dofs: `14,702 / 29,685`
- Restored optimized thresholds: `c1=4.102593e-2`, `c2=4.717566e-2`
- Final nonlinear residual: `6.252885e-16`
- Final status: `CONVERGED_CERTIFIED_SUBBAND`
- Checkpoint: `projects/diocotron/runs/scratch/torsion_reduced_optimization_homotopy/equilibrium_h005_p2_optimized_20260820_20260820-180220-890586/out/equilibrium.npz`

The checkpoint contains nodal coordinates, potential, density, mesh path,
finite-element order, thresholds, and residual metadata. It was written on two
MPI ranks and loaded successfully on 1, 2, 4, and 8 ranks. Re-solving Poisson
from the nodal density changes the saved optimized potential by `3.390187e-4`
in relative L2 and `9.673087e-4` in relative H1.

## Spatial refinement check

This preliminary resolution comparison uses the same homotopy-selected
thresholds on both meshes, `dt=0.05`, SUPG scale 1, and zero optional flux/CIP
stabilization.

| P2 mesh | cells | Poisson handoff rel. L2 | Poisson handoff rel. H1 | rho rel. L2 change at t=0.1 |
|---|---:|---:|---:|---:|
| coarse | 3,828 | 2.414978e-3 | 6.221914e-3 | 1.556e-2 |
| refined | 14,702 | 1.682497e-4 | 7.177023e-4 | 7.116e-3 |

The refined mesh reduces the density drift by a factor of 2.19 and the Poisson
handoff discrepancy by factors of 14.4 (L2) and 8.67 (H1).

## Stationarity controls on the refined mesh

| Method | dt | time | rho rel. L2 change | relative mass drift | relative energy drift |
|---|---:|---:|---:|---:|---:|
| default SUPG + CIP=1e-3 | 0.05 | 0.1 | 1.006e-2 | 5.395e-15 | -3.224e-7 |
| SUPG, CIP=0 | 0.05 | 0.1 | 7.649e-3 | 9.055e-15 | 2.608e-8 |
| SUPG, CIP=0, half dt | 0.025 | 0.1 | 7.773277e-3 | 8.669779e-15 | 1.408537e-8 |
| SUPG, CIP=0, longer run | 0.05 | 0.5 | 1.835e-2 | -1.406e-14 | 8.415e-8 |

The band is a continuum equilibrium and is approximately stationary under the
finite-element transport. It is not exactly stationary as a nodal P2 field:
interpolating the sharp nonlinear relation `rho=W(phi)` leaves a small discrete
advective defect. The weak timestep dependence and clear spatial convergence
confirm that this is primarily an interface-resolution effect. The optional
CIP term adds avoidable diffusion for this test, so `--flux-stabilization 0` is
the preferred equilibrium-validation setting.

## MPI timing

Ten steps, first two excluded, same 29,685-dof problem and direct MUMPS solves.
The benchmarks were run on an Intel Core i5-10210U laptop CPU at 1.60 GHz
(one socket, four physical cores, eight hardware threads), with 15 GiB of
OS-visible RAM, under 64-bit Linux 6.8.0-136-generic. Timings are therefore
host-specific and are not intended as cross-machine performance comparisons.

| MPI ranks | average seconds / step | speed relative to 1 rank |
|---:|---:|---:|
| 1 | 0.452347 | 1.00x |
| 2 | 0.358172 | 1.26x |
| 4 | 0.868438 | 0.52x |
| 8 hardware threads | 2.128665 | 0.21x |

Two MPI ranks are best for this mesh on this host. Final numerical diagnostics
agree across all rank counts to roundoff.

## Recommended run

```bash
mpiexec -n 2 env \
  XDG_CACHE_HOME=/tmp/hdgfem_fenics_cache \
  MPLCONFIGDIR=/tmp/hdgfem_mpl_cache \
  /home/as305/miniforge3/envs/fenicsx-dgfem/bin/python \
  projects/diocotron/dolfinx/guiding_center/supg.py \
  --equilibrium projects/diocotron/runs/scratch/torsion_reduced_optimization_homotopy/equilibrium_h005_p2_optimized_20260820_20260820-180220-890586/out/equilibrium.npz \
  --order 2 \
  --dt 0.05 \
  --num-steps 10 \
  --flux-stabilization 0 \
  --no-plot
```
