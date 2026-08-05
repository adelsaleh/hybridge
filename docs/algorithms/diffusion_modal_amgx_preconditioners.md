# Diffusion-Reaction Modal AMGX Preconditioner Sweep

Date: 2026-07-26

This note records the follow-up sweep after symmetric diagonal scaling did not fix the slow `PCGF + Chebyshev/L1` AMGX behavior for `dub_orth + legendre-modal` diffusion trace systems.

A focused hierarchy-stat audit of the aggressive Chebyshev/L1 config is recorded separately in `docs/algorithms/diffusion_amgx_hierarchy_audit.md`.

The sweep tool is:

```bash
LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib .venv/bin/python -m scripts.gpu.sweep_diffusion_amgx_preconditioners
```

It keeps the HDG discretization fixed by default, generates temporary AMGX configs under `/tmp/hdgfem/amgx_sweeps`, runs `scripts.gpu.run_diffusion_reaction_cuda`, parses setup/solve/iteration/residual fields, and writes CSV/JSONL logs under `run_logs/`.

## What Was Tested

Common discretization for the main checks:

- `case=trigonometric-poisson`
- `mesh_type=disc`
- `order=6`
- `basis=dub_orth`
- `volume_quadrature=symmetric`
- `assembly_backend=raw-cuda`
- `raw_matrix_format=csr`
- `raw_block_size=128`
- `amgx_tolerance=1e-12`

The main comparison was between the fast nodal baseline and modal variants of PCGF/Chebyshev-L1 AMG:

- Nodal baseline: `legacy-lagrange + PCGF + diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json`.
- Modal aggressive control: `legendre-modal + PCGF + diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json`.
- Modal non-aggressive Chebyshev/L1 variants generated from the same base config with `aggressive_levels=0`.
- Modal `BICGSTAB + classical AMG` as the practical fallback control.

## Coarse Order Sweep

Run log: `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p18_20260726_205823.csv`.

Matrix size: 5,699 triangles, 59,227 trace DOFs.

| variant | trace | iterations | setup | solve | physical residual |
|---|---|---:|---:|---:|---:|
| nodal Cheb/L1 aggressive | legacy-lagrange | 31 | 0.268s | 0.082s | 5.271e-13 |
| modal noagg Cheb order 3 | legendre-modal | 257 | 0.224s | 0.641s | 9.483e-13 |
| modal noagg Cheb order 4 | legendre-modal | 219 | 0.232s | 0.597s | 9.303e-13 |
| modal noagg Cheb order 6 | legendre-modal | 181 | 0.228s | 0.590s | 8.953e-13 |
| modal noagg Cheb order 8 | legendre-modal | 160 | 0.229s | 0.614s | 7.068e-13 |
| modal noagg Cheb order 10 | legendre-modal | 134 | 0.225s | 0.597s | 8.510e-13 |

The coarse screen shows a real iteration-count reduction as the Chebyshev order increases.  The solve-time minimum is flatter: order 6 and order 10 were effectively tied in this sample.

A broader coarse screen (`run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p18_20260726_205533.csv`) found that most other knobs were not useful:

- `aggressive_levels=0` alone converged in 349 iterations.
- Strength threshold changes and interpolation cap changes did not improve solve time.
- `interp_max_elements=16` triggered an AMGX/CUDA illegal memory access in this modal case.
- Reducing the sweep counts stalled at the 1000-iteration cap with residuals around `1e-4`.

## Medium Mesh Check

Run log: `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p08_20260726_205902.csv`.

Matrix size: 28,569 triangles, 298,599 trace DOFs.

| variant | trace | iterations | setup | solve | physical residual |
|---|---|---:|---:|---:|---:|
| nodal Cheb/L1 aggressive | legacy-lagrange | 32 | 0.541s | 0.174s | 6.075e-13 |
| modal Cheb/L1 aggressive | legendre-modal | 754 | 0.409s | 3.565s | 9.714e-13 |
| modal BICGSTAB classical | legendre-modal | 2000 | 0.097s | 1.368s | 9.117e-14 |
| modal noagg Cheb order 2 | legendre-modal | 845 | 0.276s | 4.421s | 9.532e-13 |
| modal noagg Cheb order 6 | legendre-modal | 369 | 0.282s | 3.818s | 9.680e-13 |
| modal noagg Cheb order 10 | legendre-modal | 283 | 0.282s | 4.386s | 9.562e-13 |

The higher Chebyshev orders still reduce modal PCGF iterations on the medium mesh, but they do not improve wall solve time relative to the aggressive modal control.  The cost per iteration is higher.

## Diagonal Scaling Checks

Coarse scaled run log: `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p18_20260726_210956.csv`.

Matrix size: 5,699 triangles, 59,227 trace DOFs.

| variant | scaling | status | iterations | setup | solve | scale kernel | physical residual |
|---|---|---|---:|---:|---:|---:|---:|
| nodal Cheb/L1 aggressive | off | ok | 31 | 0.259s | 0.073s | 0.000s | 5.271e-13 |
| nodal Cheb/L1 aggressive | symmetric | ok | 69 | 0.276s | 0.161s | 0.011s | 9.295e-13 |
| modal Cheb/L1 aggressive | off | failed | - | - | - | 0.000s | AMGX setup CUDA kernel launch error |
| modal Cheb/L1 aggressive | symmetric | failed | - | - | - | 0.009s | AMGX setup CUDA kernel launch error |
| modal BICGSTAB classical | off | ok | 1000 | 0.119s | 0.258s | 0.000s | 4.205e-14 |
| modal BICGSTAB classical | symmetric | ok | 1000 | 0.094s | 0.247s | 0.008s | 4.585e-14 |
| modal noagg Cheb order 6 | off | ok | 181 | 0.226s | 0.620s | 0.000s | 8.953e-13 |
| modal noagg Cheb order 6 | symmetric | ok | 181 | 0.313s | 0.673s | 0.009s | 8.325e-13 |
| modal noagg Cheb order 10 | off | ok | 134 | 0.226s | 0.594s | 0.000s | 8.510e-13 |
| modal noagg Cheb order 10 | symmetric | ok | 158 | 0.307s | 0.776s | 0.009s | 8.879e-13 |

On the coarse case, symmetric scaling does not improve the new PCGF variants.  Order 6 keeps the same iteration count and gets slower; order 10 needs more iterations and also gets slower.  It also does not avoid the modal aggressive setup failure in this sample.

Heavy scaled run log: `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p04_20260726_211052.csv`.

Matrix size: 113,878 triangles, 1,192,968 trace DOFs.

| variant | scaling | iterations | setup | solve | scale kernel | physical residual |
|---|---|---:|---:|---:|---:|---:|
| nodal Cheb/L1 aggressive | symmetric | 242 | 0.455s | 3.338s | 0.019s | 8.493e-13 |
| modal Cheb/L1 aggressive | symmetric | 2000 | 0.240s | 24.920s | 0.019s | 1.394e-10 |
| modal BICGSTAB classical | symmetric | 2000 | 0.101s | 5.013s | 0.020s | 2.061e-06 |
| modal noagg Cheb order 6 | symmetric | 802 | 0.591s | 32.065s | 0.018s | 9.543e-13 |
| modal noagg Cheb order 10 | symmetric | 564 | 0.599s | 34.956s | 0.029s | 9.880e-13 |

Compared with the unscaled heavy data below, symmetric scaling is again worse for the practical solve time.  The scaling kernel itself is cheap, around 0.02 s, so the loss is in the AMGX hierarchy/preconditioner behavior.

A direct regression check was added in `tests/test_cupy_scaling.py`.  It compares the device CSR row-scaling and symmetric-scaling kernels against explicit NumPy/scipy reference matrices and verifies the symmetric physical-residual recovery formula.

### AMGX Internal Scaling Check

The local AMGX source registers a top-level solver parameter:

```text
scaling = NONE | BINORMALIZATION | DIAGONAL_SYMMETRIC
```

with default `NONE` in `/home/adelMounzer.saleh/src/AMGX/src/core.cu`.  The current project diffusion configs do not set `solver.scaling`, and the generated Chebyshev/L1 sweep configs do not set it either, so AMGX was not already applying hidden matrix diagonal scaling in the previous runs.

The `error_scaling=3` entry in the Chebyshev/L1 AMG preconditioner is different: AMGX documents it as scaling the coarse-grid correction vector to minimize the error in the A-norm, not as scaling the uploaded matrix/RHS.

A temporary config with AMGX internal `solver.scaling=DIAGONAL_SYMMETRIC` was checked on the p6/ms0.18 modal no-aggressive Cheb-order-6 case.  It matched the external `--scale-system symmetric` result: 181 iterations and about 0.66 s solve.  This supports the interpretation that our external scaling is mathematically equivalent to AMGX's diagonal-symmetric scaling when enabled.

The p6/ms0.18 modal aggressive config with AMGX internal `DIAGONAL_SYMMETRIC` failed during AMGX setup with an out-of-memory path in `DenseLUSolver::solver_setup`.  Treat AMGX's built-in scaling path as diagnostic only for now; the AMGX source also comments that the internal scaling implementation repeatedly scales/unscales during setup and solve and is slow.

## Heavy 114k-Triangle Check

Run log: `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p04_20260726_210015.csv`.

Matrix size: 113,878 triangles, 1,192,968 trace DOFs.

| variant | trace | iterations | setup | solve | physical residual |
|---|---|---:|---:|---:|---:|
| nodal Cheb/L1 aggressive | legacy-lagrange | 32 | 0.449s | 0.452s | 7.503e-13 |
| modal Cheb/L1 aggressive | legendre-modal | 1923 | 0.231s | 23.978s | 9.514e-13 |
| modal BICGSTAB classical | legendre-modal | 2000 | 0.095s | 5.034s | 1.448e-09 |
| modal noagg Cheb order 6 | legendre-modal | 779 | 0.440s | 28.869s | 9.689e-13 |
| modal noagg Cheb order 10 | legendre-modal | 539 | 0.437s | 31.012s | 9.571e-13 |

On the heavy case, increasing Chebyshev order cuts modal PCGF iterations substantially, but the solve is slower than the aggressive modal control because the preconditioner application is much more expensive.  No tested non-aggressive Chebyshev/L1 variant is a practical replacement for the current modal fallback.


## Non-Chebyshev Candidate Sweep

Run logs:

- Broad coarse screen: `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p18_20260726_215932.csv`.
- Focused fine screen: `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p04_20260726_220323.csv`.

The broad coarse pass tested generated non-Chebyshev configs with both `--scale-system off` and `--scale-system symmetric`: classical AMG with plain `JACOBI_L1`, classical AMG with `BLOCK_JACOBI`, classical `MULTIPASS` interpolation with symmetric multicolor-GS, aggregation AMG with `BLOCK_JACOBI`, aggregation AMG with `MULTICOLOR_DILU` or multicolor-GS under BICGSTAB/FGMRES, classical ILU0 W-cycle variants, and direct `MULTICOLOR_DILU` Krylov baselines.

Coarse p6/ms0.18 modal survivors:

| variant | scaling | solver | iterations | setup | solve | physical residual |
|---|---|---|---:|---:|---:|---:|
| BICGSTAB aggregation GS SIZE_2 | off | BICGSTAB | 619 | 0.092s | 0.212s | 1.99e-13 |
| BICGSTAB aggregation DILU SIZE_2 | off | BICGSTAB | 611 | 0.093s | 0.214s | 5.40e-13 |
| BICGSTAB direct multicolor DILU | off | BICGSTAB | 562 | 0.092s | 0.221s | 9.63e-13 |
| BICGSTAB classical ILU0 W-cycle | off | BICGSTAB | 669 | 0.095s | 0.227s | 6.73e-13 |
| PCGF classical L1 PMIS | off | PCGF | 605 | 0.190s | 0.795s | 9.81e-13 |
| PCGF classical MULTIPASS/GS | off | PCGF | 276 | 0.194s | 1.341s | 9.35e-13 |
| PCGF classical MULTIPASS/GS | symmetric | PCGF | 260 | 0.277s | 1.379s | 8.35e-13 |

The BICGSTAB non-Chebyshev candidates were cheap and accurate on the coarse mesh.  PCGF classical MULTIPASS/GS reduced the iteration count the most among non-Chebyshev candidates, but the cost per iteration was already high on the coarse mesh.  PCGF + `BLOCK_JACOBI`, PCGF + aggregation `BLOCK_JACOBI`, FGMRES + aggregation DILU, and FGMRES/direct DILU either capped with poor residuals or produced `nan` physical residuals.  FGMRES/classical ILU0 and IDR/IDRMSYNC direct DILU failed in AMGX setup.

Fine p6/ms0.04 focused screen:

| variant | scaling | solver | iterations | setup | solve | physical residual |
|---|---|---|---:|---:|---:|---:|
| modal Cheb/L1 aggressive control | off | PCGF | 1923 | 0.234s | 23.956s | 9.51e-13 |
| modal BICGSTAB classical control | off | BICGSTAB | 2000 | 0.098s | 5.199s | 9.79e-11 |
| BICGSTAB direct multicolor DILU | off | BICGSTAB | 2000 | 0.088s | 5.373s | 6.70e-11 |
| BICGSTAB aggregation GS SIZE_2 | off | BICGSTAB | 2000 | 0.090s | 5.365s | 2.21e-09 |
| BICGSTAB aggregation DILU SIZE_2 | off | BICGSTAB | 2000 | 0.089s | 5.375s | 1.59e-08 |
| PCGF classical L1 PMIS | off | PCGF | 2000 | 0.334s | 13.361s | 9.09e-10 |
| PCGF classical MULTIPASS/GS | off | PCGF | 1149 | 0.328s | 37.646s | 9.66e-13 |
| FGMRES aggregation GS SIZE_2 | off | FGMRES | 2000 | 0.212s | 133.389s | 3.23e-10 |

On the fine 114k-triangle case, none of the non-Chebyshev candidates beats the existing modal fallback.  Direct `BICGSTAB + MULTICOLOR_DILU` is the closest non-AMG baseline by wall time and residual, but it is still slightly slower than `BICGSTAB + classical AMG` in this sample.  PCGF classical MULTIPASS/GS reaches strict residual with fewer iterations than Cheb/L1 PCGF, but the solve is slower because the preconditioner application is much more expensive.

Symmetric diagonal scaling was again not useful.  It either worsened physical residuals at the iteration cap or slightly reduced PCGF MULTIPASS/GS iterations while increasing wall time.  Keep symmetric scaling diagnostic-only for these configs.

## Interpretation

The matrix-scaling diagnostics already showed that the raw-CUDA CSR matrices are symmetric to roundoff for both nodal and modal traces.  The new sweep reinforces that this is a preconditioner/hierarchy problem rather than an obvious assembly-orientation bug.

The most likely explanation for symmetric scaling failing is that diagonal scaling fixes row/column magnitudes but does not fix the AMG coarse-space representation of high-order modal edge modes.  It also changes the strength graph seen by AMGX.  The nodal aggressive config appears tuned to the unscaled nodal operator; after symmetric scaling, nodal PCGF iterations grew from 32 to 242 on the heavy p6 case.

The most likely explanation for the Chebyshev-order result is that stronger smoothing reduces the number of PCGF iterations but makes each V-cycle much more expensive.  That helps on iteration count but not on wall time at the 114k-triangle scale.

## Current Recommendation

- Keep `legacy-lagrange + PCGF + Chebyshev/L1 aggressive AMG` as the fast diffusion path.
- Do not promote symmetric diagonal scaling for the current diffusion AMGX configs, including the generated non-aggressive Chebyshev/L1 modal variants.
- Do not promote the generated non-aggressive Chebyshev/L1 modal configs yet; they are useful diagnostics, not production configs.
- Do not promote the generated non-Chebyshev candidates yet.  The closest practical fine-mesh candidate was direct `BICGSTAB + MULTICOLOR_DILU`, but it did not beat the existing `BICGSTAB + classical AMG` modal fallback.
- Keep `legendre-modal + BICGSTAB + classical AMG` as the practical modal fallback when solve time is the priority, but note that the strict p6/ms0.04 run above stopped at `max_iters=2000` with physical residual `1.448e-09`.
- If strict `1e-12` residual is required for modal traces today, modal PCGF/Cheb-L1 aggressive can reach it but is very slow on the heavy case.

## Non-Chebyshev AMGX Candidates From Source Audit

The local AMGX source and stock configs under `/home/adelMounzer.saleh/src/AMGX/src/configs` expose several families that are worth testing before adding permanent project configs:

- `PCGF + classical AMG + BLOCK_JACOBI`, using V/W/F cycles from `PCGF_CLASSICAL_{V,W,F}_JACOBI.json`.  This is SPD-compatible and cheap, but likely weaker than the current GS/Cheb paths.
- `PCGF + classical AMG + MULTIPASS` interpolation with PMIS/HMIS and no aggressive coarsening.  This directly targets the suspected modal coarse-space representation problem.
- `PCGF + aggregation AMG + BLOCK_JACOBI`, sweeping selectors `SIZE_2`, `SIZE_4`, `SIZE_8`, and `MULTI_PAIRWISE`.  Aggregation may be less sensitive to the modal trace graph than classical aggressive coarsening.
- `BICGSTAB` or `FGMRES + aggregation AMG + MULTICOLOR_DILU/MULTICOLOR_GS`.  These are not SPD/PCGF paths, but they are promising practical modal fallbacks because DILU/GS are stronger smoothers and aggregation changes the hierarchy.
- `BICGSTAB`, `FGMRES`, `IDR`, or `IDRMSYNC + MULTICOLOR_DILU` without AMG.  This is a useful sanity baseline for whether the problem is the coarse hierarchy rather than local smoothing.
- AMGX internal `DIAGONAL_SYMMETRIC` only as a diagnostic comparison to our external scaling.  Avoid `BINORMALIZATION` for now because the device implementation in the local source calls `exit(0)` after scaling; `NBINORMALIZATION` can be tested only with nonsymmetric outer solvers because it applies left/right scaling and may break the SPD assumptions behind PCGF.

For non-CG-family outer solvers, run the sweep with AMGX residual monitoring enabled (`--monitor` in the sweep driver, or `-v 2` in the runner).  Otherwise AMGX can run to `max_iters`, making iteration counts misleading.

## Next Checks

- Inspect AMGX hierarchy statistics for nodal versus modal aggressive Cheb/L1 runs with AMGX monitoring enabled: level sizes, coarse operator density, and setup failures.
- Test whether a basis-aware edge block scaling or mass-normalized modal trace basis improves AMG coarsening more than scalar diagonal scaling.
- If modal BICGSTAB remains the practical fallback, compare `BICGSTAB + classical AMG` and direct `BICGSTAB + MULTICOLOR_DILU` at practical tolerances such as `1e-9` and `1e-10` across more mesh sizes.
- Recheck modal `BICGSTAB + classical AMG` at practical tolerances such as `1e-9` and `1e-10`, because it is much faster when the application error tolerance does not require a `1e-12` physical residual.
