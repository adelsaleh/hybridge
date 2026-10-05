# Raw CUDA tensor ADR qualification — 2026-09-28

Scope: FP64 assembly and reconstruction, native CSR/BSR stationary solves, and
assembly timings. Tensor postprocessing is excluded and remains gated. No AMGX
build, large CPU factorization, simulation, or time integration was performed.
The existing AMGX 2.5.0 CUDA-13 runtime was used for small stationary solves.

The [implementation plan](../../development/plans/raw_cuda_adr_tensor.md) records
the contract; the [backend guide](../../backends/raw_cuda.md#tensor-adr-assembly-and-reconstruction)
describes supported APIs.

## Numerical evidence

- Matrix/RHS parity against NumPy and Numba for all seven exact tensor classes
  (eight coefficient representations), p=0--6, both trace bases, COO/direct
  CSR/direct face BSR: 336 assembled operators across 112 fixtures.
- Reconstruction of prescribed nonzero traces against both independent host
  paths in those 112 fixtures. CUDA results remain device arrays.
- Additional distorted mesh, discontinuity, mixed classification, cross-space
  DG coefficients, spatial face tau, launch-size, failure and resource tests.
- Native CSR/BSR stationary manufactured solves at p=2,4,6 for both bases, with
  device-resident trace/local fields and an explicit prohibition on BSR
  scalarization. Affine primal and analytic diffusive flux errors are checked.
- Degree-1/2 smooth manufactured refinement checks without postprocessing.
- Shared face sampling covers scalar, element/incidence tables, DG fields,
  geometry/context/legacy callables, incompatible meshes and quadrature tables,
  internal callable errors and invalid positive/finite values.

Primary test files are `tests/test_adr_tensor_raw_cuda.py`,
`tests/test_adr_tensor_solver_cuda.py`, and `tests/test_adr_face_stabilization.py`.
Existing diffusion, advection, ADR, capability and documentation tests were also
run. The final combined run completed **576 numerical/API checks**, with four
optional-dependency skips, before reaching the pre-existing documentation-tree
failure. The separate documentation audit passed four checks and reported five
existing failures: an extra `docs/diocotron` directory, generated hp-AMG files,
PDF artifacts, older broken artifact links, and missing docstrings in unrelated
functions. The new plan/index/TODO checks passed and none of the new links or
functions appeared in the failures. Existing scalar raw-CUDA solve/recovery
checks were also rerun explicitly: **4 passed**.

The tensor-specific suite comprises **131 assembly/reconstruction checks** and
**15 native stationary solve/gate checks**. The latter include two smooth
manufactured refinement tests. During qualification these exposed an existing
ADR default mismatch: AMGX was given an absolute stop while `solver_rtol` was
checked as relative. The ADR default now uses `RELATIVE_INI_CORE`; user-supplied
AMGX configurations remain unchanged.

One pre-existing regression assertion is excluded from the aggregate:
`test_raw_cuda_rejects_explicit_advection_stabilization_before_device_setup`
expects the old `advection_stabilization=None` error text. The current capability
message also lists already-supported stabilization policies. The same failure
was reproduced using the pre-edit source snapshot; this work does not change it.

## Timings

GPU: NVIDIA RTX PRO 5000 Blackwell. CuPy 14.2.0, CUDA runtime 13.2, FP64.
Each ADR configuration uses three warmups and seven repetitions. The sweep
covers 512 and 32,768 triangles; p=2,4,6; both trace bases; COO/CSR/BSR;
32/64/128 threads; scalar, constant-full and variable-full diffusion.

All **324 configurations** completed. Every repetition, phase timing and
workspace size is preserved in
[the ADR JSONL](raw_cuda_adr_tensor_2026_09_28.jsonl); the
[source manifest](raw_cuda_adr_tensor_2026_09_28_manifest.json) identifies the
implementation. The following are ranges of
median kernel milliseconds across the two bases and three formats, using the
automatic thread tier:

| Elements | p | Scalar | Constant full | Variable full |
|---:|---:|---:|---:|---:|
| 512 | 2 | 0.103–0.111 | 0.101–0.110 | 0.172–0.180 |
| 512 | 4 | 0.568–0.577 | 0.563–0.573 | 0.999–1.013 |
| 512 | 6 | 3.370–3.392 | 3.360–3.379 | 10.393–10.596 |
| 32,768 | 2 | 4.785–4.854 | 4.710–4.989 | 8.230–8.442 |
| 32,768 | 4 | 35.525–36.599 | 35.413–36.253 | 59.744–63.018 |
| 32,768 | 6 | 180.820–181.362 | 180.409–180.964 | 536.779–543.866 |

For 32,768 elements/p=6, total assembly including the separately measured ADR
preparation is 329.87–348.51 ms (scalar), 327.65–347.34 ms (constant full), and
878.93–915.12 ms (variable full). The preparation sample is taken once per
space/basis/coefficient setup; it is added to each warmed wrapper sample rather
than resampled on every repetition. Diffusion classification, upload, graph,
JIT and kernel work are measured inside each wrapper call. Conversion time is
zero on the cooperative direct paths. Warmed JIT timings reflect cache lookup,
not a cold compilation estimate.

These are assembly timings, not solve or simulation speedups. The workstation
also drives a display, and no hardware clock lock or profiler replay was used.
The automatic launch policy remains a conservative starting policy rather than
an assertion that it wins every configuration.

## Diffusion regression check

[Paired diffusion samples](raw_cuda_adr_diffusion_regression_2026_09_28.jsonl)
cover all **36** combinations of those mesh sizes, p=2,4,6, both bases and three
formats. The preserved before/after implementations alternate in AB/BA order,
with 14 samples per version per configuration after warmup.

The cooperative diffusion and orientation source strings are byte-identical
after mechanical extraction. Kernel after/before ratios range from 0.99361 to
1.00639; wrapper ratios range from 0.96117 to 1.02839. No configuration exceeds
the 5% slowdown threshold. The maximum measured increases are 0.64% and 2.84%.

The earlier 12-check / 18-configuration baseline at
`/tmp/hybridge_adr_assembly_baseline_hiqvlnkv/baseline.json` remains preliminary;
it is not used as a regression conclusion. The matched comparison used a copy
of the pre-edit working tree at `/tmp/adr_tensor_before`, preserving uncommitted
work. No working-tree checkout or reset was performed.

## Reproduction

Run from the repository root with the configured CUDA-capable `.venv`:

```sh
NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 .venv/bin/python -m pytest -q tests/test_adr_tensor_raw_cuda.py tests/test_adr_tensor_solver_cuda.py tests/test_adr_face_stabilization.py
NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 .venv/bin/python scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_raw_cuda.py --output /tmp/adr-tensor.jsonl
NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 .venv/bin/python scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_raw_cuda.py --diffusion-before /tmp/adr_tensor_before/hybridge/backends/diffusion_raw_cuda.py --output /tmp/diffusion-paired.jsonl
```

The benchmark appends JSONL records; use a fresh output path for a new campaign.
For another checkout, save its original diffusion module before editing and
supply that file to `--diffusion-before`. This qualification requires no AMGX
rebuild. Tensor recovery and transient models remain separate work.
