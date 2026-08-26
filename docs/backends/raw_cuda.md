# Raw-CUDA Backend Notes

This document records raw-CUDA implementation ownership and launch policy.
Current support is defined only by
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md).

## Kernel Ownership

The fused raw-CUDA element kernels perform the expensive local work:

- construct equation-specific local operators from prepared coefficient data;
- factor and solve the local condensed systems;
- eliminate known trace columns;
- emit reduced COO entries, direct scalar CSR values, or direct dense face-BSR blocks;
- accumulate the reduced right-hand side on device;
- reconstruct local fields with the matching equation kernel.

Reference tensors, projected/source moments, compact boundary data, topology,
and reduction maps are prepared before the element launch. Reusable solver
objects cache matrix- and graph-dependent data when their invalidation contract
allows it.

## Launch Policy

Public solvers and runners use `raw_block_size="auto"`. The established fused
kernels resolve that option before kernel selection and cache-key construction;
kernels receive an integer block size. TSLE-BSR is the exception: its three
stages retain `"auto"` until the backend selects an independent launch size for
each stage.

| Equation | Polynomial order | Recommended block size |
|---|---:|---:|
| Advection-reaction | `p <= 6` | 32 |
| Advection-reaction | `p = 7` | 64 |
| Advection-reaction | `p = 8, 9` (experimental) | 128 |
| Diffusion/Poisson | `p <= 2` | 32 |
| Diffusion/Poisson | `p = 3, 4` | 64 |
| Diffusion/Poisson | `p = 5, 6` | 128 |

These are policy defaults covered by launch-resolution and parity tests.
The p=8--9 fused recommendation is additionally backed by the CUDA-13 SM75
block sweep below; it remains experimental rather than a cross-architecture
claim. Explicit `1`, `32`, `64`, and `128` overrides remain available for
benchmark reproduction.
Block size 1 is a serial correctness baseline, not a performance
recommendation. Orders outside the policy fail explicitly rather than silently
choosing a launch shape.

## Tri-Stage Local Elimination BSR (`split3`)

`raw_local_assembly="split3"` selects **TSLE-BSR**, implemented in
`hdgfem.backends.advection_tsle_bsr`. It preserves the fused advection HDG
discretization and separates only the execution schedule:

1. `advection_tsle_build` constructs each local operator `A_e`, unsolved trace
   and source columns `[B_e, f_e]`, and the face `tau/gamma` tables.
2. `advection_tsle_solve` performs the cooperative scaled-pivot LU and applies
   it to every trace/source column. The deprecated `safe` LU path is not used.
3. `advection_tsle_scatter_bsr` forms the trace lifts, condensed Schur blocks
   and RHS, eliminates boundary columns, and writes direct face BSR. Unique
   off-diagonal blocks use plain stores; the two element contributions to each
   diagonal face block use the fused Schur-plus-mass atomic update.

The operator, solved response, and face-flux arrays live in a persistent
`RawAdvectionTsleWorkspace`. A reusable solver preserves their identities while
the device, mesh size, polynomial dimensions, and quadrature shape are fixed;
coefficient updates overwrite values without reallocating the workspace. BSR
is the only supported output, `raw_lu_mode="coop"` is required, p=1--7 is the
qualified scope, and p=8--9 is available as an explicitly experimental path.

For `raw_block_size="auto"`, TSLE screens candidates 32, 64, 128, and 256 for
each stage independently on at most 32,768 elements, then retimes the best two
per stage on the full mesh before caching the winning triple by device and
discrete shape. Subsequent assemblies in the process skip tuning.
Level-3 solver logging prints the three stage times, selected blocks, device
total, workspace GiB, and cold/reused tune state in two lines. Levels 2 and 4
retain the full candidate/micro-timing dump.

On the Quadro RTX 6000 CUDA-13 qualification mesh (157,280 triangles,
exact matrix/RHS/local-response parity), CUDA-event kernel medians were seven
alternating hot samples for p=1--7 and five for experimental p=8--9:

| p | Legacy fused | Legacy TSLE | Gain | Modal fused | Modal TSLE | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.664 ms | 5.636 ms | 0.5% | 5.857 ms | 5.817 ms | 0.7% |
| 2 | 13.347 ms | 13.343 ms | 0.0% | 13.625 ms | 13.612 ms | 0.1% |
| 3 | 27.214 ms | 27.230 ms | -0.1% | 28.004 ms | 28.023 ms | -0.1% |
| 4 | 54.815 ms | 54.899 ms | -0.2% | 55.525 ms | 55.591 ms | -0.1% |
| 5 | 121.766 ms | 122.156 ms | -0.3% | 122.992 ms | 123.639 ms | -0.5% |
| 6 | 241.261 ms | 243.599 ms | -1.0% | 243.584 ms | 247.125 ms | -1.4% |
| 7 | 534.697 ms | 474.323 ms | **12.7%** | 552.716 ms | 476.477 ms | **16.0%** |
| 8 | 1261.866 ms | 857.723 ms | **47.1%** | 1269.227 ms | 864.241 ms | **46.9%** |
| 9 | 1918.382 ms | 1757.016 ms | **9.2%** | 1984.066 ms | 1769.417 ms | **12.1%** |

The gate therefore keeps `fused` as the production `p <= 6` choice. TSLE
remains explicit at p=7 despite its measured gain, and p=8--9 remains
experimental. The high-order fused kernel uses 128 threads and moves its Schur
lift rows into block-shared storage: compiler profiles report zero local-memory
bytes for fused and all TSLE stages. Fused dynamic shared memory is 40,948 B at
p=8 and 56,604 B at p=9, below the target GPU's 65,536 B opt-in limit.

TSLE's extra persistent workspace is material: 1.732 GiB at p=6, 2.679 GiB at
p=7, 3.969 GiB at p=8, and 5.676 GiB at p=9 on this mesh. Full-mesh finalist
autotuning cost about 25.7 s at p=8 and 53.6 s at p=9 in the modal runs. The
corresponding measured cold-cost break-even is roughly 62 and 253 repeated
assemblies, respectively; use the high-order path only when that cost and
workspace are acceptable.

Reproduce the assembly-only sweep with:

```bash
scripts/gpu/run_cuda13.sh .venv/bin/python \
  scripts/gpu/benchmark_advection_tsle_bsr.py \
  --case disk-tangent --mesh-size 0.0068 --orders 1,2,3,4,5,6,7 \
  --trace-bases legacy-lagrange --warmups 2 --repeats 7
scripts/gpu/run_cuda13.sh .venv/bin/python \
  scripts/gpu/benchmark_advection_tsle_bsr.py \
  --case disk-tangent --mesh-size 0.0068 --orders 8,9 \
  --trace-bases legacy-lagrange --warmups 2 --repeats 5
```

Use `--trace-bases legendre-modal` for the second production orientation and
`--case test2` for the legacy nonzero-boundary manufactured problem. Focused
parity, autotune reuse, workspace identity, tangent-assembler, and explicit
p=9/256-thread TSLE scratch tests live in `tests/test_advection_tsle_bsr.py`.
The latter is also clean under CUDA-13 memcheck and racecheck.

## Discontinuous Advection Audit

CuPy assembly stores `beta . n` per element side. Raw-CUDA precomputed,
fused COO, fused CSR, and fused face-BSR paths preserve those side values through local
coupling, stabilization, and trace-mass construction. Interior sides are
emitted independently and summed into the global trace row; no face averaging
is introduced.

Fused reconstruction is element-local and uses the element's own projected
advection coefficient. It does not reconstruct an averaged interior-face
coefficient. The stateful raw-CUDA path retains the device workspace for the
local response ``A_e^-1 [B_e, f_e]``; each coefficient update overwrites that
workspace during assembly, and a small raw kernel applies it to the converged
trace. On the 157,280-triangle p=6 guiding-center case this reduced warm
reconstruction from about 0.212 s to 0.0021 s without changing the approximately
0.239 s operator assembly.

Validation anchors include:

- `tests/test_advection_reaction_numba.py` for side-weighted trace mass;
- `tests/test_cupy_backend.py` for CuPy/raw-CUDA matrix and RHS parity,
  modal traces, and COO/CSR/BSR equivalence;
- `tests/test_advection_reaction_conservation.py` for global conservation and
  no-through-flow boundaries.

## Diffusion Setup Ownership

The raw diffusion kernel owns local Schur construction, local solves, boundary
elimination, reduced emission, and RHS accumulation. It does not materialize
the pure-CuPy path's large `local_lhs`, `solved_el_bd`, `solved_src`,
`trace_blocks`, or `faces` intermediates.

Prepared outside the hot kernel:

| Data | Preparation |
|---|---|
| Source moments | CuPy quadrature/projection or a reusable prepared table |
| Boundary trace | Compact device interpolation/projection and edge mapping |
| Reference derivatives and face tensors | Small order-dependent device tables |
| COO reduction maps | Host-origin maps copied to device |
| CSR pattern and block positions | Raw-CUDA pattern kernels and device prefix operations |
| Mesh/reference mirrors | One-time host-to-device construction for the CuPy space |

The remaining optimization direction is to cache all order-dependent reference
tables, keep compact boundary storage, generalize table-driven coefficients,
and eliminate COO-only host map construction where measurements justify it.

## Failure And Validation Policy

Raw-CUDA requests are preflighted before assembly. Unsupported combinations
must raise a capability error or take an explicitly documented compatibility
path; they must not silently change coefficient semantics, trace basis, matrix
format, or convergence requirements.

The required parity and smoke lanes are documented in
[`../development/alpha_test_matrix.md`](../development/alpha_test_matrix.md).
