# Higher-order raw CUDA ADR qualification, 2026-10-15

Raw CUDA ADR assembly and reconstruction passed FP64 qualification through
p=9, preserving p=0--6 behavior, seven diffusion classifications, both production
trace bases, direct COO/CSR/BSR assembly and all three factor-cache policies.
This extends the [September tensor implementation](raw_cuda_adr_tensor_2026_09_28.md).
The [implementation plan](../../development/plans/raw_cuda_adr_tensor.md#higher-order-extension-2026-10-14)
describes warp row groups, compact coupling storage and bounded mass scratch.

## Results and coverage

| Selection | Passed | Scope |
|---|---:|---|
| `test_adr_tensor_raw_cuda.py` | 294 | NumPy/Numba parity through p=9, launches, quadrature, validation and reuse |
| `test_adr_high_order_raw_cuda.py` | 190 | CuPy comparisons, caches, resource limits, chunking and pivot exchanges |
| Native stationary CSR/BSR selection | 24 | p=2,4,6,7,8,9, both trace bases, device reconstruction |
| Host selection | 306 | Launch policy, resource planning, package layering and capabilities |
| GPU smoke selection | 2 | p=6 and p=9 tensor checks |
| Compute Sanitizer memcheck | 8 | Zero errors |
| Compute Sanitizer racecheck | 8 | Zero hazards, errors or warnings |

The three ADR selections total **508 passed**, with no failures, errors or skips.
They include 87 host resource cases; the host, smoke and sanitizer selections
overlap these 508 cases and should not be added as distinct cases.

The principal NumPy/Numba sweep covers 160 coefficient/order/trace fixtures:
eight coefficient representations spanning all seven classifications, orders
0--9 and `legacy-lagrange`/`legendre-modal` traces. Each fixture checks three
assembled formats, giving 480 matrix/RHS comparisons and 160 CSR-path local
reconstruction comparisons under the default `schur-lu+mass` policy.

The high-order suite checks p=7,8,9 against independently assembled full mixed
systems using CuPy quadrature contractions and dense solves. ADR has no general
production CuPy assembly backend; this implementation is a test oracle. Its 48
coefficient/order/trace fixtures each exercise three formats and three caches
(`none`, `schur-lu`, `schur-lu+mass`), checking matrix, RHS and reconstructed
unknowns with `rtol=atol=5e-10`. The zero-advection, identity-diffusion subset
also compares directly against the existing arbitrary-order CuPy DR dense and
Schur-Cholesky assembly/reconstruction paths, with reaction zero and 0.4.

Additional checks cover distorted geometry, nonzero boundary values, spatial
incidence stabilization, degree-nine Bernstein/hierarchical element bases,
cross-space device coefficients, mixed classifications, empty reduced systems,
forced 48 KiB shared-memory limits, bounded external-mass chunks and static-mass
reuse after source/reaction/advection/boundary changes. A synthetic 110-row LU
test forces pivots across warp row groups at rows 32, 64 and 96. Explicit block
sizes 1/32/64/128 are exercised.

Each sanitizer selection repeats six p=9 external-mass cases (classes 4/5/6,
uncached and mass-cached), one 110-row pivot test and one p=7 general-tensor
CuPy-reference case that loops over formats and caches. Both tools used
`--error-exitcode 99 --target-processes all` and returned zero.

## Native BSR limitation

The existing AMGX `MULTICOLOR_DILU` preconditioner lacks a 9-component block
specialization. Therefore p=8 native BSR solves require an explicit compatible
`amgx_config`. The qualification uses
`configs/amgx/adv_rea_gpu4_hdg_pbicgstab_l1_bsr.json`, overriding the outer solver
to FGMRES (restart 100, maximum 300 iterations) while retaining `JACOBI_L1` and
`jacobi_l1_scalar_rows_for_blocks=1`. BSR storage remains intact; tests forbid
scalarization. CSR/COO assembly, BSR assembly and local reconstruction do not
depend on DILU. No AMGX source, build or default configuration was changed.

## Reproduction and provenance

From the repository root, with the CUDA runtime and existing pyamgx/AMGX stack:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_adr_tensor_raw_cuda.py \
  tests/test_adr_high_order_raw_cuda.py \
  tests/test_adr_tensor_solver_cuda.py::test_native_tensor_solve_and_device_reconstruction
```

The detached qualification retained 227 passed parity nodes after the user
requested release of the GPU reservation, then ran the remaining 67 and all
subsequent stages concurrently without a GPU lock. JUnit results and retained
node identities exactly cover the original 508-node collection. The terminal
marker records success at **2026-10-15 09:04:58 UTC**.

Local ignored evidence resides in
`run_outputs/qualification/adr_p7_p9_2026_10_15/`: frozen source archive,
file-hash manifest, launcher, exact commands, logs, JUnit XML, per-stage markers,
resume records and `completion_audit.json`. The source snapshot includes the
complete dirty worktree at capture time, based on commit
`80653eec626097544e894e9c1472ae70cfbae1f4`. The final audit verified every frozen
file hash, exact node coverage and executable AST identity of the current ADR
task Python files, ignoring comments/docstrings.

| Item | Recorded value |
|---|---|
| GPU | NVIDIA RTX PRO 5000 Blackwell, 48,935 MiB |
| Driver / nvcc | 580.126.09 / 13.0.88 |
| Python / NumPy / Numba | 3.12.3 / 2.5.3 / 0.67.0 |
| CuPy | 14.2.0 (`cupy-cuda13x`) |
| AMGX source revision | `c564f9d866eaff427bac17fddaaeeb5c3bbad2d0` |
| pyamgx source revision | `a44d224d8b92f94b8d004794805528dfe3e8b460` |
| Source archive SHA256 | `0923008c5b480402207bde925315183319a19bca9781f03a46af15aab934116e` |
| Loaded AMGX library SHA256 | `b3453ec2d341094c3ffcb7ce4929c8fb8b8fedfc04b62497614dc5e194f5a744` |

The loaded library was rehashed unchanged at completion. This is correctness
evidence from bounded stationary problems, not a high-order performance,
large-mesh scalability or convergence study. Concurrent elapsed times do not
support speedup claims. Higher-order FP32 and postprocessing qualification are
separate; the existing tensor recovery evidence remains through p=6. Historical
September timings are unchanged by this result.
