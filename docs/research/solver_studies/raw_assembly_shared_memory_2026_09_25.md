# Raw assembly: shared-memory and execution investigation

The investigation prioritizes shared-memory access, synchronization, local
factorization dependencies, FP64 execution, and occupancy. The prior p=6 fused
Blackwell profiles in TODO.md showed 73.1% (Poisson) / 78.1% (transport) peak
FP64-pipe throughput and only 0.36% / 0.42% peak DRAM throughput. Those bounded
results motivate this focus; they do not establish the bottleneck of cache reuse
or every launch size.

On 2026-09-25 the current user could not access hardware counters:
`ERR_NVGPUCTRPERM`. No measured shared-memory bank-conflict/stall claim is made.

## Capture

From the repository root, with GPU performance-counter access enabled:

```sh
.venv/bin/python scripts/gpu/capture_raw_assembly_profiles.py \
  --output /tmp/hdgfem-shared-profiles --cases poisson \
  --orders 6 --sizes 128 --bases legendre-modal --blocks 128
```

This runs twenty diffusion configurations: COO/CSR/BSR full assembly without
and with Schur-LU construction, CSR/BSR RHS and prescribed-trace reconstruction
without/with LU reuse, and the separate CuPy compact Schur-Cholesky cache
construction, RHS and reconstruction phases attached to CSR/BSR operators. Each configuration gets two warmups, five unprofiled repetitions,
and a separate one-repetition Nsight capture. All kernels inside the measured
wrapper range are retained, so coefficient preparation/scatter kernels must be
separated from the named local assembly kernel during analysis. No AMGX build,
global solve, or time integration is invoked.

Return `manifest-both.json` (or `manifest-baseline.json` / `manifest-profile.json`), the `.csv` files, and baseline `.stdout` files; retain
`.ncu-rep` files for instruction-level follow-up. Reports include shared-memory
workload tables, source counters, warp states, scheduler statistics, occupancy,
launch resources, instruction statistics, and compute utilization. SASS-level
counters are available in the report; CUDA source-line correlation is not yet
qualified for the runtime-generated kernels. Profiler timings include replay
and must never replace the unprofiled timing baseline.

Next sweep: use `--orders 2 4 6 --sizes 16 128`, both `--bases`, and
`--blocks 32 64 128`. Omit `--cases poisson` to include fused transport
COO/CSR/BSR and TSLE BSR. Use `--mode baseline` or `--mode profile` to capture
these separately. The script stops at the first error and preserves its log
and exact command in the manifest. The manifest records source hashes and Git
HEAD; each mode refuses to overwrite an existing manifest of the same mode.

For a smaller first capture focused on the launch-size observation below:

```sh
.venv/bin/python scripts/gpu/capture_raw_assembly_profiles.py \
  --output /tmp/hdgfem-shared-launch --cases poisson --orders 6 --sizes 128 \
  --bases legendre-modal --blocks 32 64 128 \
  --formats bsr --cache-policies none --phases assembly
```

`--cases adr` profiles the extracted assembly-only ADR helper, using the
implemented one-thread-per-element COO-to-CSR path. It does not claim direct
CSR/BSR emission or cooperative ADR. `--coefficient-cases constant variable`
adds a variable source for Poisson and variable source/reaction/advection for
transport and ADR. These switches do not imply support for variable diffusion.
The compact Cholesky RHS/reconstruction kernels have their own fixed launch
policy; `--blocks` controls the preceding raw assembly, not those kernels.

## Questions the counters must resolve

- Compare actual versus ideal shared wavefronts and load/store bank conflicts;
  64-bit accesses alone are not evidence of avoidable conflicts.
- Locate short-scoreboard and MIO-throttle stalls alongside shared throughput.
  Determine whether access layout, instruction issue, or dependencies limit work.
- Locate barrier and wait stalls around cooperative LU, triangular solves, and
  column batching. Source inspection shows repeated block barriers and sequential
  factor/solve dependencies; their time contribution is not yet measured.
- Compare FP64 utilization and math-pipe stalls with eligible warps, achieved
  occupancy, registers, dynamic shared bytes, and resident blocks. Low occupancy
  by itself is not proof of a performance bottleneck.
- Compare uncached full assembly, cache construction, and RHS factor reuse;
  retain DRAM/L2 traffic as control measurements, especially for factor reuse.
- Confirm any proposed layout, batching, or launch change with matched unprofiled
  timings and matrix/RHS/reconstruction/physical-residual parity before changing
  production defaults.

## Unprofiled baseline and validation

RTX PRO 5000 Blackwell, CuPy 14.2.0, CUDA runtime 13020, driver API 13000.
The checked-in JSON preserves all phase samples and configuration metadata.
512 structured triangles, p=6, modal traces, 128 threads, two warmups and five
repetitions; these small-mesh results are not a production speedup claim.

| Poisson phase | COO median ms | CSR median ms | BSR median ms |
|---|---:|---:|---:|
| Uncached full assembly | 0.937 | 0.931 | 0.934 |
| Full assembly + Schur-LU construction | 0.935 | 0.930 | 0.934 |
| RHS without factor reuse | unavailable | 0.492 | 0.493 |
| RHS with Schur-LU reuse | unavailable | 0.118 | 0.121 |

The nearly equal full-assembly format timings are consistent with local work
dominating this case, but do not identify its hardware bottleneck. Reuse removes
substantial local work in the RHS path. Transport medians were 0.651/0.668/0.660
ms for fused COO/CSR/BSR and 0.505 ms for TSLE BSR, at this same small mesh.

All 14 harness combinations completed at p=2 and p=6. Existing small-matrix
factor-cache validation passed 18 tests with no skips:

```sh
NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 .venv/bin/python -m pytest -q \
  tests/test_numba_diffusion_schur_cache_cuda.py
```

That suite checks host/raw CSR matrix, RHS, reconstruction and physical-residual
parity at orders 1, 3 and 6 with both production bases, plus separate CuPy
Cholesky factor-action parity. It does not qualify every profiled emission mode.

## Historical kernel resource evidence

The preserved first-pass `.ncu-rep` files were imported without GPU counter
access. [Extracted resource metadata](raw_assembly_first_pass_resources_2026_09_25.json)
records their hashes, values and units. Both reports use 128 threads and a
48,896-block grid: the fused kernels launch `max(num_tri, num_interior_edges)`,
consistent with 32,768 triangles on the documented 128-by-128 mesh.

| Historical p=6 BSR kernel | Registers/thread | Dynamic shared bytes | Limiting resource | Maximum resident blocks/SM |
|---|---:|---:|---|---:|
| Poisson cooperative | 71 (72 allocated) | 32,232 | Shared memory | 3 |
| Transport fused | 168 | 15,072 | Registers | 3 |

Both have a **theoretical** 25% occupancy limit at that launch shape. This is
not measured achieved occupancy. Poisson's register limit permits seven blocks
and transport's shared-memory limit permits six, which distinguishes the
binding resource in each report. These reports contain neither shared bank
conflicts/wavefronts nor warp-stall breakdowns. New counters remain necessary.

## Expanded measurements

[Expanded samples](raw_assembly_expanded_2026_09_25.json) contain 85 configurations:
25 p=6 / 512-triangle cases covering diffusion, transport and combined ADR, and
60 p=6 / 32,768-triangle diffusion cases spanning 32/64/128 requested threads.
Both sets use modal traces, two warmups and five repetitions, with full per-phase
samples, variance, host-wall measurements and wrapper CUDA-event intervals.
The desktop GPU was not isolated. These are launch comparisons on one device,
not evidence for changing defaults or predicting another GPU's performance.

Representative BSR medians, in milliseconds:

| Phase on 32,768 triangles | 32 threads | 64 threads | 128 threads | Measurement |
|---|---:|---:|---:|---|
| Uncached full assembly | 73.739 | 54.146 | 59.874 | Local kernel CUDA events |
| Full assembly + Schur-LU construction | 78.191 | 61.055 | 59.473 | Local kernel CUDA events |
| RHS without factor reuse | 37.307 | 31.834 | 30.576 | Local kernel CUDA events |
| RHS with Schur-LU reuse | 6.259 | 5.467 | 5.318 | Local kernel CUDA events |
| Reconstruction without factors | 33.025 | 28.078 | 26.645 | Whole-wrapper event interval |
| Reconstruction with Schur-LU | 9.123 | 7.476 | 7.706 | Whole-wrapper event interval |

A wrapper event interval includes GPU idle gaps caused by host work; it is not
summed kernel time and must not be compared directly with isolated kernel time.
Raw reconstruction currently exposes synchronized wall time, so its isolated
kernel duration is deliberately left null instead of relabeling wall time.

At requested block size 128, separate compact Schur-Cholesky construction took
70.718 ms of wrapper event time, RHS took 2.685 ms of local-kernel time, and
reconstruction took 0.695 ms of local-kernel time. Construction includes compact
trace-response preparation; its cost and memory differ from Schur-LU factors.
The ADR assembly kernel took 37.929 ms on the 512-triangle case, compared with
about 0.932 ms for Poisson BSR; these are different equations/implementations,
not a matched algorithmic speedup. ADR has serial local arrays, so its memory
hierarchy and instruction behavior require separate profiling.

The 32-to-64-thread full-assembly gain motivates investigating work mapping and
available parallelism. Its nonmonotonic 128-thread result also argues against
assuming higher occupancy alone will improve every phase. Counters must resolve
shared access conflicts, synchronization and arithmetic dependency contributions.

The extraction of `assemble_projected_adr_trace_operator_raw_cuda` reuses the
production kernel and preparation code. The existing ADR solve wrapper calls
this helper; numerical algebra and solver defaults are unchanged. It records
local-kernel CUDA events separately from COO-to-CSR conversion wall time.

Focused validation passed **28 tests, 18 deselected**, without skips:

```sh
NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 .venv/bin/python -m pytest -q \
  tests/test_raw_assembly_profiling_cuda.py \
  tests/test_numba_diffusion_schur_cache_cuda.py \
  tests/test_advection_diffusion_reaction.py \
  -k 'assembly_only or profiled_factor or numba_cached or raw_cuda_matches_numpy'
```

This includes nonzero-boundary ADR matrix/RHS and independent physical-residual
parity, a guard against invoking AMGX in the assembly-only helper, existing ADR
solver integration, and the actual profiled RHS/reconstruction callables across
none/LU/compact-Cholesky policies with CSR/BSR and both trace bases. Cache-only
measurements are checked to exclude stale raw-assembly timing metadata.

The final driver also completed all 25 variable-coefficient execution checks
at p=2 on eight triangles with nodal traces; the [smoke artifact](raw_assembly_variable_smoke_2026_09_25.json)
contains exact commands and source hashes. These checks establish execution,
not variable-coefficient accuracy or meaningful performance.

## High-order coverage and completion audit

The capture driver now accepts `--cases transport --orders 7 8 9`. Both trace
bases and fused COO/CSR/BSR plus TSLE BSR completed 24 additional warmed captures
on 512 triangles, with five repetitions after two warmups. The
[high-order artifact](raw_assembly_transport_high_order_2026_09_25.json) preserves
samples, exact commands and source hashes. At p=9 the BSR medians were
4.535/4.608 ms fused versus 3.374/3.403 ms TSLE (nodal/modal). These are bounded
small-mesh measurements, not hardware explanations or production recommendations.
Diffusion and ADR remain capped at p=6 in this profiling driver; selecting a
higher order requires an explicit transport-only run.

The additional focused validation passed 14 tests, 104 deselected:

```sh
NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 .venv/bin/python -m pytest -q \
  tests/test_raw_assembly_profiling_cuda.py \
  tests/test_advection_tsle_bsr.py \
  tests/test_diffusion_reaction_assembly_parity.py \
  -k 'profiled_diffusion_formats or bsr_matches_csr or split3_bsr_matches_fused_local_elimination or p9_split3_256'
```

The new harness-specific checks compare variable-source diffusion COO/CSR/BSR,
with and without factor-cache writes, against the NumPy physical system at
p=2,6 and both trace bases. Independent dense reference solves on four-element
meshes verify the original unscaled residual for each format/policy. Existing
checks cover fused/TSLE high-order local elimination, reconstruction responses,
and the explicit p=9 256-thread TSLE solve kernel versus fused 128-thread work.

| Requirement | Authoritative evidence | Status |
|---|---|---|
| Reproducible emission/cache/phase capture | Capture driver and saved command manifests | Implemented; selected configurations executed |
| Separate host/preparation/kernel/wrapper timings | JSON phase samples and explicit timing scopes | Bounded evidence; fused subphases still need instruction attribution |
| Historical resource limits and FP64 utilization | Hashed first-pass Nsight resource extract | Verified for two historical p=6 BSR kernels |
| Shared bank conflicts, warp stalls, achieved occupancy | No new report; fresh metric query returns `ERR_NVGPUCTRPERM` | Blocked by counter permissions |
| Broader order/format/cache parity | Focused test commands above plus earlier 28-test validation | Bounded verified coverage; not exhaustive |
| Profiler perturbation and counter comparisons across the matrix | User-run capture command is ready | Awaits hardware-counter reports |
| Hardware bottleneck attribution and any default promotion | Insufficient counters; defaults unchanged | Not complete |

Counter access has failed in three consecutive goal turns. The requested
`/tmp/hdgfem-shared-profiles` and `/tmp/hdgfem-shared-launch` captures have not
arrived; only the already-examined historical reports are present. The next
necessary evidence is the focused shared-memory capture above, followed by the
broader matrix as needed. Further unprofiled timing permutations cannot supply
the missing bank-conflict or stall measurements.

## Outstanding scope

The original TODO remains open. New hardware-counter measurements (including
shared-memory bank conflicts, warp stalls and achieved occupancy), profiler
perturbation measurement, wider counter coverage across orders/bases/coefficient cases, exhaustive
cross-format physical-residual qualification, and measured
instruction-level attribution remain outstanding. The new focused capture was
rechecked and still fails with `ERR_NVGPUCTRPERM`. Existing resource counters and
unprofiled timing differences do not resolve the missing stall measurements.
Raw-CUDA diffusion currently accepts only `local_factor_kind="schur-lu"`; the
separate CuPy Schur-Cholesky path is labeled accordingly. No numerical kernel or
production default was changed.
