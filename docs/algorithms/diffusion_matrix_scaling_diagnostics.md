# Diffusion-Reaction GPU Matrix Scaling Diagnostics

Date: 2026-07-26

This note records the first device-side diagonal scaling diagnostics for `scripts/gpu/run_diffusion_reaction_cuda.py`.  The goal was to test whether symmetric Jacobi scaling fixes the slow `PCGF + Chebyshev/L1` AMGX convergence observed for `dub_orth + legendre-modal` diffusion traces, without changing the preconditioner configuration.

## Diagnostic Tool

The standalone diagnostic script is:

```bash
.venv/bin/python -m scripts.gpu.diagnose_diffusion_matrix_scaling
```

It reuses the GPU diffusion runner assembly path, builds a device CSR matrix, optionally applies device-side scaling to a copy, and reports matrix diagnostics outside the solver stack.  The reported quantities include symmetry defect, diagonal ranges, row/column norm percentiles, optional small-mesh condition estimates, and optional AMGX iteration counts.

The symmetric scaling path applies

```text
A_s = D^{-1/2} A D^{-1/2},    b_s = D^{-1/2} b,
```

on device and recovers the physical trace unknown with `x = D^{-1/2} y` after AMGX solves the scaled system.

## Heavy Raw-CUDA CSR Case

Common parameters:

```bash
LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib .venv/bin/python -m scripts.gpu.diagnose_diffusion_matrix_scaling   --case trigonometric-poisson --mesh-type disc -ms 0.04 -o 6   --basis dub_orth --volume-quadrature symmetric   --assembly-backend raw-cuda --raw-matrix-format csr --raw-block-size 128   --amgx-solver PCGF   --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json   --amgx-tolerance 1e-12
```

The matrix size was `1,192,968` reduced trace DOFs with `41,676,852` CSR nonzeros.

| trace basis | scaling | symmetry defect | row p95 | col p95 | PCGF iterations | AMGX solve |
|---|---:|---:|---:|---:|---:|---:|
| `legacy-lagrange` | none | `8.034e-15` | `4.765e+00` | `4.765e+00` | 32 | `0.453s` |
| `legacy-lagrange` | symmetric | `9.474e-15` | `1.321e+00` | `1.321e+00` | 242 | `3.341s` |
| `legendre-modal` | none | `7.028e-15` | `2.266e+01` | `2.266e+01` | not rerun in this sample | not rerun |
| `legendre-modal` | symmetric | `1.241e-14` | `1.569e+00` | `1.569e+00` | 2000 | `24.847s` |

The symmetric modal scaling kernel itself was cheap on the heavy CSR matrix, around `0.020s`; the failure is not scaling overhead.

## Small-Mesh Condition Checks

On a p2, 64-triangle disc diagnostic with `condition_max_dof=5000`, both nodal and modal matrices were symmetric to roundoff.  Symmetric scaling normalized row/column norms and, for nodal traces, reduced the estimated condition number from `1.825e+02` to `1.384e+02`.  For the same small modal case, the condition estimate changed from `1.554e+02` to `1.751e+02`.

These small-mesh estimates are useful sanity checks, but they do not predict the heavy modal AMGX behavior by themselves.

## Conclusions

- The assembled diffusion trace matrices are symmetric to roundoff for both `legacy-lagrange` and `legendre-modal` in the tested raw-CUDA CSR cases.
- Symmetric scaling is implemented correctly enough to normalize the diagonal to one and strongly reduce row/column norm spread.
- Symmetric scaling is not a robust fix for the modal PCGF/Chebyshev-L1 convergence problem.
- With the current Chebyshev/L1 AMGX config, symmetric scaling actually worsens the heavy nodal solve: 32 PCGF iterations became 242.
- The modal issue is therefore more likely an AMG/coarsening/preconditioner sensitivity than a raw-CUDA orientation or symmetry bug.
- Keep `legacy-lagrange + PCGF + Chebyshev/L1` as the recommended fast diffusion path.
- Keep `legendre-modal + BICGSTAB + classical AMG` as the practical modal diffusion path until modal AMG preconditioning is improved.
- Do not promote `--scale-system symmetric` as a default or modal recommendation for this AMGX configuration.
