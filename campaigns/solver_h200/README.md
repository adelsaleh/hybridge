# H200 solver-kernel follow-up campaign (2026-10-27)

Follow-up to the unified ADR campaign (`campaigns/adr_unified`). On matched systems the
H200 NVL solves the native candidates only 1.2–1.9× faster than the RTX PRO 5000 Blackwell.

## What the early data show

- **No throttling.** The SM clock is at its 1785 MHz maximum, power draws 100–230 W of 600 W, and no
  throttle flag is active.
- **The solve is memory-bound.** The face-dense operator and the ASM application stream about
  0.25 flop/byte. The H200's FP64 advantage cannot help there; its bandwidth advantage, about 3.6×,
  could.
- **The kernels reach about 30% of H200 bandwidth.** Example: ASM-PP(24) at 32k triangles, p=6. One
  operator + ASM application takes 153 µs on the H200 and 288 µs on the Blackwell. That is about
  1.4 TB/s, roughly 30% of the H200 peak, against about 55% on the Blackwell.
  - Both kernels use one thread per row. The operator row is 35 doubles and the ASM row 21, so the
    loads don't coalesce.
  - Launches use a fixed 256-thread block.
- **The batched-BLAS route is untested at scale.** Its `matmul` variants (cuBLAS strided-batched,
  as in Welter & Nguyen, arXiv:2512.13619) were only compared on a Quadro RTX 6000 (Aug 2026) and on
  2k-triangle Blackwell systems (Sept 2026). They lost both times. They were never measured on the
  H200 or on large systems.
- **Warm starts over-solve.** The native GMRES's inner trigger is `rtol·beta`, relative to each
  cycle's own starting residual (`hybridge/linalg/gpu/gmres.py:1344`). A warm-started solve
  therefore iterates to the round-off floor:
  - ASM: 57 iterations and a 3e-12 final residual at every δ;
  - AMGX on the same system: iterations fall with δ, 364 → 271.
- **AMGX DILU is latency-bound.** An iteration costs 1.18 ms on the H200 and 1.23 ms on the
  Blackwell, against a roughly 60 µs bandwidth floor.

## Results (2026-10-27, L3 = 32k triangles, p=6)

- **Launch shape (slv-01).** Threads per block change the face-dense kernels by a few percent at
  most. The row-per-thread kernels reach 30–38% of H200 bandwidth, and the cuBLAS batched `matmul`
  paths are slower at every p.
- **Coalesced kernels (slv-05, slv-06).** The gate passed 63 tests, and the 54 primitive rows agree
  with the reference to within 2.6e-16.
  - At p=6 the operator is 1.61× faster (60% of bandwidth) and ASM 1.39× faster (42%).
  - At p=4: operator 1.48×, ASM 1.32×.
  - Block-Jacobi is about 0.9×, and ASM at p=2 is 0.88×, so the variant is chosen per kernel and degree.
  - Full solves: repeated solves are 1.33–2.02× faster with identical iterations, and warm solves
    gain by the same factor; setup is unchanged.
- **True-target trigger (slv-04).** The gate passed 55 tests.
  - Solves that span more than one restart cycle get faster. asm_pp, trigonometric: repeated
    490 → 250 ms (166 → 84 iterations), warm at δ=1e-4 216 → 167 ms. pp, advection-dominated:
    307 → 218 ms.
  - Single-cycle solves are unchanged, and residuals stay within target.
  - One trade-off: pp's warm solves at δ=1e-3 and 1e-2 get slower (153 → 193 and 216 ms).
- **slv-10** runs both options together.
- **Profiles (slv-00).**
  - AMGX FGMRES+DILU: 58% of GPU time is about 3 µs cuBLAS dot, reduce and axpy launches
    (Krylov orthogonalization, one launch per basis vector). It is launch-latency-bound, not DILU-bound.
  - ASM+PP: 91% is the ASM local apply plus the operator.
- **AMGX DILU blocks 6 and 9 (slv-03, slv-03b).** The rebuilt library is bitwise identical to
  production on 16/16 shared cases, and the 8 aborted L1 jobs pass. slv-09 reruns the rest.
- **DILU L2 misses (slv-08).** The AMGX target is `internal_rtol·‖b‖`.
  - At `internal_rtol` 1e-12, all 6 variable_velocity jobs meet the 2e-8 L2 bound (2.1e-9 to
    6.8e-9), for 6–25% more iterations.
  - At 1e-13 the errors are 3e-11 to 6e-10.
- **native_hp reruns (slv-07).** 7 passed and 1 not_converged (a stall at maxiter), which is
  also what the campaign runner's rule gives.

## Kernel follow-up (2026-10-28, branch `h200-solver-kernels`, slv-11 to slv-24)

All new kernels are opt-in. Those marked bitwise give exactly the production result (same
multiply-add chain), so iteration counts are unchanged by construction.

**What the robust hp-MG profile showed (slv-11, 98,699 triangles, p=6).**
- `legendre_face_bsr_block_jacobi_step` takes 79.5% of GPU time (766k calls, 155 µs each).
- That kernel and its zero-start and raw-SpMV siblings give one 32-lane block to each b×b block
  row. Only b lanes work (7/32 at p=6), the loads are strided, and one-warp blocks cap sm_90 at 32
  resident warps per SM.
- The AMGX coarse W-cycle (classical, JACOBI_L1, DENSE_LU) is about 13%. About 32 dense-LU getrs
  run per cycle, with about 70 internal stream syncs and memcpys per outer iteration.

**Options.**
- **hp-MG kernel presets.** Select one with `kernels=` on `FaceBlockHpMgPcgSolver` and on
  `BernsteinHpSymmetricPartPreconditioner`, or with `HYBRIDGE_HP_MG_KERNELS`; the default is
  `legacy`.
  - `packed`: one thread per scalar row (`PackedFaceBsrKernels`, block-transposed copies). Bitwise.
    The PCG solver also fuses its x/r and direction updates (`_rn` intrinsics), also bitwise.
  - `packed-fused`: adds the residual and restriction in one kernel, with the prolongation folded
    into the first post-smoothing stage. Differs by round-off.
  - `graph`: `packed-fused` with each p-sweep replayed as one CUDA graph around the AMGX call.
- **Face-dense interleaved kernels.** Operator `implementation="interleaved"` and ASM/BJ
  `application="interleaved"` use `InterleavedGatherGemv`: one thread per row on an interleaved
  copy of the matrices. Bitwise.
- **FP32 inverse storage.** `application="interleaved_f32"` stores the ASM/BJ inverses in FP32
  and accumulates in FP64.
- **Gates.** `tests/test_hp_mg_packed_kernels.py` (58 passed, slv-12) and
  `tests/test_cupy_face_dense_interleaved.py` (with the coalesced, ASM and polynomial suites,
  76 passed, slv-13).

**Primitives (slv-14, 98,699 triangles, p=6, 256 threads).**

| Kernel | Production | Coalesced | Interleaved |
|---|---|---|---|
| Operator | 153 µs (42% of peak) | 92 µs | **84 µs (76%), 1.83×** |
| ASM | 249 µs (31%) | 159 µs | **113 µs (67%), 2.20×** |
| ASM, FP32 storage | — | — | 74 µs, 3.4× |
| BJ | 24.7 µs | 26.6 µs | 23.7 µs |

On L3 (slv-15), interleaved is the fastest variant at p=2, 4 and 6: operator 1.4–2.1×, ASM
1.3–2.0×.

**ASM+PP and BJ+PP, 98,699-triangle orthogonal stress system (slv-16 to slv-22).** Repeated and
amortized times in ms, with iterations; every solve passed.

| Variant | ASM+PP repeated | ASM+PP amortized | BJ+PP repeated | BJ+PP amortized |
|---|---|---|---|---|
| Baseline (campaign kernels) | 2339 (58 it) | 1589 | 3054 (165 it) | 2185 |
| Interleaved (bitwise) | 1186, 1.97× | 843 | 1900, 1.61× | 1364 |
| Interleaved, FP32 inverses | 955, 2.45× | 693 | 1780 | 1402 (δ=1e-3 warm: 109 → 146 it) |
| Interleaved + true_target | 690 (33 it), 3.39× | 632 | 1088 (94 it), 2.81× | 1074 |
| Interleaved, FP32 inverses + true_target | **557 (33 it), 4.20×** | **524, 3.03×** | **984 (90 it), 3.10×** | **1007, 2.17×** |

Setup is unchanged (ASM about 1.08 s, BJ about 0.21 s). Against the Blackwell (5.20 and 8.30 s
repeated), the H200 goes from 2.2–2.7× to 4.4× with interleaved kernels, to about 7.5× with
true_target, and to 8.4–9.3× with FP32 inverses and true_target. The worst true residual over
all solves is 9.5e-12, against an rtol of 1e-10.

**Native hp-MG (GMRES), same system (slv-20).** Repeated solves; all passed with identical
iterations.

| Job | legacy | packed | packed-fused | graph |
|---|---|---|---|---|
| robust (900 it) | 26.65 s | 13.63 s, bitwise | 13.73 s | **13.33 s, 2.00×** |
| standard, direct-to-zero (1350 it) | 24.19 s | 16.96 s, bitwise | 16.90 s | **16.79 s, 1.44×** |

The robust job is now 14.8 ms per iteration, against 49.8 ms on the Blackwell.

**Robust job, warm series (slv-23).** Setup is 716–718 ms for every preset. Warm δ = 1e-4 / 1e-3 /
1e-2 takes 750 / 900 / 900 iterations for every preset, the same as the cold solve.

| Preset | Warm δ=1e-4 / 1e-3 / 1e-2 |
|---|---|
| legacy | 22.06 / 26.48 / 26.47 s |
| packed | 11.51 / 13.81 / 13.82 s |
| graph | 11.24 / 13.49 / 13.49 s |

**Robust job with `estimate_target='true_target'` (slv-25, graph preset).** The worker now accepts
`spec['estimate_target']`; `bench_hp_mg_kernels.py --estimate-target` sets it.

| Job | Cold / repeated | Warm δ=1e-4 / 1e-3 / 1e-2 |
|---|---|---|
| robust | 851 it, repeated **12.60 s (2.10× legacy)** | 647 / 756 / 831 it (9.58 / 11.21 / 12.30 s) |
| standard | 1300 it, repeated **15.92 s (1.52×)** | 933 / 1095 / 1250 it |

All passed.

**Profile with the graph preset (slv-24).**
- The packed step takes 54 µs against 155 µs and is now 59% of GPU time.
- The AMGX coarse path is now about 29%:
  - `csrmv` 14.3% (2.87M calls at 3.5 µs);
  - DENSE_LU getrs (laswp + 2×trsv) about 11.6% (174k calls, about 47 µs each);
  - JACOBI_L1 3.4%.
- About 80 host syncs and memcpys per iteration remain. They come from the AMGX/pyamgx coarse call:
  the fork's getrs path has none.

**Guiding-centre Poisson PCG, fbmg-standard, ladder G1 (about 4M trace unknowns, slv-21).**
Repeated Krylov time; iterations identical, warm solves gain the same factor.

| Preset | p=2 (108 it) | p=4 (83 it) | p=6 (72 it) | p=9 (61 it) |
|---|---|---|---|---|
| legacy | 788 ms | 535 ms | 391 ms | 447 ms |
| packed | 362 ms | 328 ms | 271 ms | 392 ms |
| **packed-fused** | **316 ms, 2.49×** | **295 ms, 1.81×** | **270 ms, 1.45×** | **318 ms, 1.41×** |

`graph` matches `packed-fused` at this size, because kernel time dominates.

**Not changed here.** The AMGX p=0 W-cycle is the remaining non-kernel cost (the share is in
slv-24). Its levers change the algorithm, not the kernels: a V-cycle, fewer `coarsest_sweeps`, a
capped `max_levels`, or an explicit-inverse coarse solve in the AMGX fork. They are left for a
decision.

## GPU budget

The campaign reserves no GPU of its own. It runs as stages in a GPU lane (`campaigns/gpu_lane/`),
which holds one H200.

## Systems

The unified campaign's caches (`/scratch/user/hybridge/adr_unified_l5/cache/<system id>`) are
read, never written. The campaign's `summary.json` supplies the case, degree and size, and the
campaign's own timings for the same system and candidate.

The 32k-triangle level (L3) is used first. The roughly 130k and 400k levels (L4, L5) can be used
once the unified campaign has cached them.

## Stages

Bulk output goes to `/scratch/user/hybridge/solver_h200/<step>`; records are copied to
`run_outputs/campaigns/solver_h200/<step>`. The kept records are in
`run_outputs/campaigns/solver_h200/`, the slv-21 results in
`21-gc-poisson-results.tar.gz`, and the H200 Nsight Systems reports (slv-00, slv-11, slv-24), the
CUDA tuning cache, the DILU 6/9 comparison arrays and the final slv-07 `result.json` files in
`h200-nsys-profiles-and-references-20261108.tar`. Solution arrays and SQLite exports were not kept.

1. `slv-00-profile`: `profile_campaign_job.sh` reruns finished campaign solves with the frozen
   worker under Nsight Systems, at 32k triangles, p=6, advection-dominated. It covers ASM-GMRES
   (`asm_d24`), AMGX FGMRES+DILU, AMGX FGMRES+AMG and native hp-MG. Outputs are the kernel, API and
   memcpy summaries plus the per-launch timeline.
2. `slv-01-primitives-l3`: `scripts/gpu/sweep_face_dense_solver.py --phase primitives`. It times
   one application each of:
   - the operator (matmul / raw / raw_fused);
   - ASM (matmul / raw / fused);
   - block-Jacobi (matmul / raw);

   at 64–1024 threads per block (`HYBRIDGE_FACE_DENSE_THREADS`, default 256, unchanged), at
   p=2, 4, 6. Each row records the median and minimum, the streamed bytes, effective GB/s, the
   share of peak, and parity against the 256-thread result.
3. `slv-02-solve-l3`: `--phase solve` reruns the campaign's native candidates on the hybridge
   classes with the campaign protocol: 1 warm-up and 3 repetitions, each with a timed setup, two
   zero-guess solves and the warm series at δ = 1e-4, 1e-3, 1e-2 over 4 steps. It reports setup,
   cold, repeated, amortized and per-δ warm timings with iterations, next to the campaign record.
   This step checks that the hybridge stack reproduces the vendored one before anything changes.

4. `slv-03-dilu69` (`campaigns/gpu_lane/stages/amgx_dilu_check.sh`): AMGX MULTICOLOR_DILU at block
   sizes 6 and 9 (p=5, p=8). The production build has no such case and aborts ("block size was
   not enabled"), which caused the campaign's 66 `process_error` jobs.
   - The AMGX fork adds the cases to all four dispatch switches, reusing the generic `_large`
     kernels (38–40 registers at FP64, no spills).
   - It is built only into `AMGX-build-cuda13.1-dilu69`. The campaign's `AMGX-build-cuda13.1` is
     not touched.
   - The stage checks it with `scripts/gpu/check_amgx_dilu_blocks.py`:
     - DILU sweeps and FGMRES+DILU at blocks 1–10;
     - block b against block b+1 with a decoupled identity component (same colouring, same factor);
     - bitwise identity with the production library on blocks 1–5, 7, 8 and 10.
   - It then reruns the 8 aborted L1 advection-dominated jobs with the new library.
5. `slv-04-target-l3` (`campaigns/gpu_lane/stages/gmres_target_study.sh`): the opt-in GMRES option
   `estimate_target='true_target'`. It first runs the GPU GMRES tests as a gate, then reruns the
   slv-02 systems and candidates with the scaled trigger.

6. `slv-05-coalesced-primitives` and `slv-06-coalesced-solve-l3` (`campaigns/gpu_lane/stages/solver_gated.sh`):
   the opt-in coalesced kernels.
   - The variants are operator `implementation="coalesced"` and ASM/block-Jacobi `application="coalesced"`.
   - Each uses one generic kernel, `GroupedGatherGemv` in `hybridge/linalg/gpu/face_dense.py`.
     A group of up to 32 lanes per face or element reads the contiguous matrix in order, and a
     fixed-order butterfly reduces the row sums.
   - Offline sm_90 compile: 20–72 registers and no spills, from operator p=1 to ASM p=9.
   - Results agree with the row-per-thread kernels to round-off, not bitwise.
   - The gate is `tests/test_cupy_face_dense_coalesced.py`.
   - slv-05 measures the primitives against `raw_fused`/`fused`/`raw` at 128, 256 and 512 threads.
   - slv-06 reruns the L3 native solves with both variants swapped in (`--operator-variant` and
     `--application-variant`).

Later stages follow the assembly study's rule: measure first, then enable behind an option, with
parity tests and a gate.

- **Variants.** A direct `cublasDgemvStridedBatched` operator and ASM path on the face-major
  layout, which is the paper's formulation without the gather and copy passes. Also a coalesced
  custom kernel (a warp per face row, or the matrix stored transposed).
- **Warm-start trigger.** Implemented as `estimate_target='true_target'` (on `restarted_gmres_cupy`,
  `CuPyRestartedGMRESSolver` and `CuPyProductionGMRESOptions`); the default is unchanged.
  - It scales the preconditioned-estimate trigger to `target·beta/‖r_cycle‖`, so warm starts and
    later restart cycles stop near `rtol·‖b‖` instead of round-off.
  - For the first cycle from a zero guess it equals the default.
  - It is measured in slv-04.
- **AMGX configuration.** Residual monitoring interval and DILU colouring, against the profile's
  launch and synchronisation counts.
- **Multiple right-hand sides.** A block-RHS application, GEMM instead of GEMV, at about
  0.25·k flop/byte, as the route to a compute-bound solve on the H200.
- **Scale.** L4 and L5 reruns with the best variants, at the same protocol.

## Reporting

Every timing report separates:
- preconditioner setup;
- cold solve (setup + first solve);
- the repeated solve on a reused setup;
- warm-started solves per δ, with iterations;
- the amortized cost.

The comparison references are the campaign's H200 records and the Blackwell `scaling.csv`
(`docs/research/solver_studies/adr_scaling_2026_09_17`).
