# Raw CUDA tensor ADR: assembly and reconstruction

Scope confirmed on 2026-09-28: finish assembly and reconstruction; exclude new
postprocessing work. Qualification is FP64, stationary, and bounded. No AMGX
builds or time integration are included. The existing AMGX runtime is sufficient
for the native CSR/BSR checks.

## Implementation requirements

Reuse `PreparedDiffusion` and its exact seven classifications. Constant tensors
apply the reference mass inverse; variable isotropic/diagonal tensors factor one
or two scalar masses by Cholesky; coupled symmetric/general tensors use coupled
Cholesky/pivoted LU. Classification never rounds small couplings to zero.

Each cooperative block owns one element. It builds the inverse-diffusion mass
factor and scalar Schur complement, then condenses trace/source columns in
batches of at most eight. Face contractions use quadrature samples and reference
tables inside the kernel. Reconstruction uses the same factorization and mass
algebra with source plus full-trace right-hand sides, retaining coefficient data
on device. The subsequent factor-cache implementation offers `none`, `schur-lu`
and `schur-lu+mass` (the default); static diffusion mass factors can also be reused
across assemblies while source, reaction, advection and boundary data change.

Diffusion's cooperative LU, column solves and orientation source were extracted
mechanically. Its generated cooperative source remains byte-identical; launch
policy, arithmetic order and workspaces remain unchanged. ADR shares the reduced
face graph. Direct CSR and `(p+1)` by `(p+1)` BSR emission require neither COO
conversion nor BSR scalarization. COO remains available for diagnostics.

At p=6, constant/isotropic/diagonal/coupled tensor workspaces use respectively
28,112 / 30,800 / 37,072 / 48,944 bytes. Coupled tensors use seven columns;
other classes use eight. Thus every supported p=0--6 case remains under 48 KiB.
The constant specialization does not reserve coupled tensor storage.

`prepare_adr_data(..., dense_local_matrices=False)` skips dense element operators.
The NumPy/Numba preparation defaults are preserved. The assembly-only operator
accepts `matrix_format="coo"|"csr"|"bsr"` and
`block_size="auto"|1|32|64|128`. Automatic cooperative tiers are 32/64/128 for
p<=2/4/6. Dense-prepared scalar CSR at block size 1 retains the old serial
assembly diagnostic; lightweight/tensor inputs use the shared kernel at size 1.

Shared diffusion stabilization sampling accepts positive scalars, per-element
or incidence arrays, same-mesh DG fields, geometry callables, legacy
`tau(x,y,K,e)`, and `tau(x,y,*,element,local_face,normal,t=None)`. It samples each
incidence independently and keeps `tau_adv` separate. Spatial laws are retained
for later quadrature; incompatible quadrature-only tables are rejected.
Stabilization policy values are preserved. Nonfinite coefficients, invalid
ellipticity, local factorization failures and unsupported resources raise before
returning a usable system or reconstruction.

The complete raw-CUDA solver now supports tensors with `hdg_postprocess="none"`,
`raw_matrix_format="csr"|"bsr"` (and explicit COO), and `raw_block_size`.
The original assembly/reconstruction qualification excluded tensor recovery.
The subsequent [tensor postprocessing qualification](../../backends/adr_device_postprocessing.md#tensor-qualification-2026-09-29)
enables primal and both total-flux recoveries, with Numba/CuPy parity and
bounded manufactured convergence.

## Qualification and evidence

The [dated report](../../research/solver_studies/raw_cuda_adr_tensor_2026_09_28.md)
records commands, results, environment and performance limitations. The original
12-check / 18-configuration baseline remains at
`/tmp/hybridge_adr_assembly_baseline_hiqvlnkv/baseline.json`; it is preliminary
measurement, not evidence of a regression.

Acceptance covers matrix/RHS and reconstruction against NumPy and Numba for all
seven classes, p=0--6, both trace bases and all sparse formats. Fixtures include
distortion, nonzero boundaries, varying face tau, jumps, mixed classifications,
cross-space coefficients, invalid inputs, factor failures and workspace bounds.
Small stationary manufactured tensor problems qualify device-resident
reconstruction and native CSR/BSR solves without postprocessing.

Assembly benchmarks use p=2,4,6 on 512 and 32,768 elements, both bases, all
formats and cooperative sizes, with scalar, constant coupled and variable
coupled coefficients. Timings separate preparation, diffusion classification,
upload, graph, JIT, kernel, conversion and total. Diffusion comparisons alternate
the preserved before/after implementations with repeated matched samples; a
repeatable slowdown above 5% requires investigation before acceptance.

## Deferred work

Tensor total-flux recovery (`l2_closest` and `RT_projection`) and tensor primal
postprocessing are now qualified separately, as linked above. The
[n-Gamma plan](n_gamma_d_bdf2.md) retains its remaining interface and model work.
The [main TODO](../../../TODO.md) owns feature status.

## Higher-order extension, 2026-10-14

The p=7--9 extension keeps the seven exact diffusion classifications, both
production trace bases, direct COO/CSR/BSR output, and all three cache policies.
This stage covers assembly and reconstruction; it does not extend the separate
postprocessing qualification or run time integration.

The scalar Schur solves distribute more than 32 rows across each warp. Coupled
mass solves distribute up to 110 rows across four registers per lane, including
LU pivot swaps between row groups. Through p=6 the original shared-memory
layout and 48 KiB resource ceiling remain unchanged. At p=7--9 the kernel
recomputes the two normal-minus-derivative coupling entries instead of retaining
two dense matrices, then selects the widest batch (up to eight columns) fitting
the selected device's opt-in shared-memory limit.

If a variable mass factor still does not fit, the kernel uses the requested
mass cache directly. With `none` or `schur-lu`, it processes consecutive element
chunks using at most 64 MiB of temporary mass storage. Reconstruction repeats
the same policy; no temporary scratch is retained by the operator. Quadrature
scratch remains bounded, and oversized rules still raise `TensorWorkspaceError`
with the degree, classification, point counts and fitting resource limit.
Assembly timings expose the selected shared bytes/limit, compact coupling,
external mass, temporary bytes and launch count. Automatic ADR launches use
32/64/128 threads through p=2/4/9, separately from the legacy DR launch policy.

Qualification is implemented in `tests/test_adr_tensor_raw_cuda.py`,
`tests/test_adr_high_order_raw_cuda.py`, and
`tests/test_adr_tensor_solver_cuda.py`. Besides the existing arbitrary-order
NumPy/Numba paths, the new tests independently assemble full mixed local
systems with CuPy quadrature contractions and dense solves. ADR has no general
CuPy assembly backend, so this is explicitly a test oracle. Tests cover all
classes, formats and caches; distorted meshes, nonzero boundary data and varying
face stabilization; 48 KiB fallback, scratch chunks, mixed classifications,
cross-space/device coefficients, changed inputs with static mass reuse, empty
reduced systems, and pivot exchanges spanning all four row groups. Native
CSR/BSR manufactured stationary checks use the existing AMGX library.
The zero-advection identity-diffusion subset additionally compares directly with
the existing arbitrary-order CuPy diffusion/reaction assembly and dense and
Schur-Cholesky reconstruction paths, for both zero and nonzero reaction.
Degree-nine Bernstein and hierarchical element bases receive normwise tensor
parity checks alongside the Dubiner-orthogonal sweeps.

The current AMGX fork's `MULTICOLOR_DILU` implementation supports 8- and
10-component blocks but has no 9-component specialization. Therefore p=8 native
BSR solves need an explicit compatible `amgx_config`, such as FGMRES with
`JACOBI_L1` and `jacobi_l1_scalar_rows_for_blocks=1` (keeping BSR storage).
The p=8 BSR qualification selects this configuration. CSR and COO assembly,
BSR assembly, and local reconstruction have no such DILU dependency; extending
AMGX DILU or changing the solver default is outside this kernel stage.

FP64 qualification completed on 2026-10-15: all **508 ADR checks passed**
(294 NumPy/Numba parity, 190 CuPy/cache/resource and 24 native CSR/BSR solve
cases), with no failures, errors or skips. The 306 host checks and p=6/p=9 smoke
gate passed. Eight selected cases passed under each of Compute Sanitizer
memcheck and racecheck, reporting zero errors and zero hazards. The
[dated qualification report](../../research/solver_studies/raw_cuda_adr_high_order_2026_10_15.md)
records coverage, tolerances, commands, environment, source/library digests and
limitations. Host, smoke and sanitizer selections overlap the 508 ADR cases.

The detached qualification records its frozen worktree archive, manifest,
per-stage logs/JUnit XML and terminal `completion.json` under
`run_outputs/qualification/adr_p7_p9_2026_10_15/`. After interruption to release
the GPU, the concurrent run retained 227 passed parity nodes and ran only the
remaining 67. The final audit verified exact coverage of all 508 selected nodes,
all frozen file hashes, unchanged executable ADR task sources and the unchanged
loaded AMGX library. Concurrent elapsed times are not benchmark measurements.
