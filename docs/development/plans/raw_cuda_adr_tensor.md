# Raw CUDA tensor ADR: assembly and reconstruction

Scope confirmed on 2026-09-28: finish assembly and reconstruction; exclude new
postprocessing work. Qualification is FP64, stationary, and bounded. No AMGX
builds or time integration are included. The existing AMGX runtime is sufficient
for the native CSR/BSR checks.

## Implementation contract

Reuse `PreparedDiffusion` and its exact seven classifications. Constant tensors
apply the reference mass inverse; variable isotropic/diagonal tensors factor one
or two scalar masses by Cholesky; coupled symmetric/general tensors use coupled
Cholesky/pivoted LU. Classification never rounds small couplings to zero.

Each cooperative block owns one element. It builds the inverse-diffusion mass
factor and scalar Schur complement, then condenses trace/source columns in
batches of at most eight. Face contractions use quadrature samples and reference
tables inside the kernel. Reconstruction uses the same factorization and mass
algebra with source plus full-trace right-hand sides, retaining coefficient data
on device and rebuilding local factors. Local factor caching is a future option.

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
