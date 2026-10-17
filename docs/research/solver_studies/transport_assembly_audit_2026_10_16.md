# Transport assembly cost audit, p = 7–9 (guiding-centre SI-BDF2 step), 2026-10-16

Audit by a research agent (read-only: code, run records, host counts, NVRTC compile checks; no simulation,
benchmark, test suite or AMGX build). Data: the D8-S smoke campaign
(`gc-hdg-article/run_outputs/diocotron_order_smoke_20261016`, 9.0 M element unknowns at every p,
RTX PRO 5000 Blackwell 48 GB). Legend: **[V]** verified by reading code / run records / host checks;
**[I]** inferred (model, scaling, estimate).

## Executive summary

1. The per-element transport assembly cost grows as the kernels' FP64 flop count: 103 k FMA (p=6), 194 k (p=7), 406 k (p=8), 696 k (p=9) per element, and every production kernel runs at a nearly constant 0.46 TFLOP/s (43–45 % of the card's ≈1.05 TFLOP/s FP64 peak, 62–64 % of its measured 0.74 TFLOP/s DGEMM). Time/element: 0.44, 0.84, 1.72, 3.06 µs — the measured growth is the flop growth; nothing is memory-bandwidth-bound (≤3 % of time).
2. Two things make p ≥ 8 steeper than the p ≤ 7 trend: (i) at p=8 the compact Dunavant table (degree ≤ 14) ends, the default rule becomes the (2p+2)² collapsed rule (324/400 points), and the kernels switch from the Q·n² quadrature operator to the 2n·n² reference-tensor contraction, so the effective "point count" of the volume term jumps from 42 (p=7) to 90 (p=8) / 110 (p=9); (ii) the local-solve work (n³/3 + 31·n² column solves) is executed as 31 serial dependent chains per element (31 of 256 threads active), which is latency-bound: the split3 solve stage reaches 0.24 TFLOP/s (204 of 503 ms at p=9).
3. Affine reference tensors are already implemented and are what runs at p=8–9 (`advection_tensor` (2,NB,NEL,NEL), 2.66 MB at p=9, L2-resident; mass is diagonal for the `dub_orth` basis). It is the cheapest exact operator at p=9 (333 k FMA) unless its structure is exploited: a degree-prefix bound on k removes 10 % of it exactly, and the x-direction tensor is only 22 % dense (y: 50 %).
4. Quadrature exactness: the default `volume_degree = 2p` is exact for the mass only; with the degree-p raw drift the advection integrand has degree 3p−1, so p ≤ 7 (Dunavant 2p) is under-integrated (degree 14 rule vs 20 needed at p=7) while p=8–9 (collapsed rule exact to 4p+2) is exact. A (p+1)² collapsed rule is exact to 2p (not 2p+1; the Duffy Jacobian costs one degree), would under-integrate p=8–9 and would not change the p=8–9 assembly cost at all (the tensor path is kept). `--volume-quad-1d 14` (p=9) / `13` (p=8) keeps the operator exact and halves every NQV-proportional diagnostic.
5. The Poisson RHS refresh (87 ms at p=9, 49 ms at p=7) is 17× above both its flop and byte floors: with cached Schur-Cholesky factors, phase 1 of `poisson_local` performs one n² forward/backward substitution on a single thread per element (3 active FP64 chains per SM). A warp-parallel substitution (code exists for ADR) should bring it to ≈10–15 ms.
6. The fused kernel is "validated to p=7" as a qualification-scope statement (docstring, docs, experimental test ids), not a known defect; at p=9 it needs 56.6 KB of shared memory → one 128-thread CTA per SM (4 warps), which is why split3 (two 256-thread CTAs per SM per stage) is 44 % faster there. No register spills at p=8–9 (48–80 regs, 0 local bytes); the p=7 fused kernel uses 254 registers with 288 B of spill.
7. Best gain/effort for p=9: (1) warp-parallel column solves in the split3 solve stage (or MAGMA batched LU, measured 1.9× faster than the cooperative kernel at p=9): −90…−100 ms; (2) parallel triangular solve in Poisson phase 1: −70 ms; (3) exploit the tensor's degree/sparsity structure in the build stage: −40…−60 ms. Expected p=9 step 810 → ≈590 ms (−27 %); a block-Jacobi preconditioner and/or a looser transport tolerance could remove another 30–60 ms.
8. Out-of-step diagnostics at p=9 (enstrophy 209 ms, other 309 ms) are dominated by E×NEL×NQV evaluation passes on the 400-point rule (gradients, divergence, extrema, azimuthal and positivity sampling) plus 0.5–1 GB CuPy temporaries; the enstrophy budget runs every step regardless of `--diagnostics-every`.
9. For a 16,000-step p=9 run: `--diagnostics-every 10` costs 8 min (2.3 % of the 3.6 h of stepping) and gives 100 records in the [20,45] window; the enstrophy budget costs 56 min if left on — keep it only for validation runs, or make it cheap (`--volume-quad-1d 14`, or compute the volume term from the assembled local operator).
10. First validation for every kernel change: condensed BSR values/RHS against the current kernels on a small host mesh (the degree-prefix/sparsity change and the Poisson/warp solves agree to ≈1e-15 relative — not bitwise, since summation order changes; MAGMA pivoting differs, so ≤1e-10 relative at p=9), then the campaign's consistency check (mass drift 6e-16, L2-identity defect 2e-16).

## 1. What the kernels do per element

### 1a. Quadrature policy (the solution space sets everything) [V]

* `hybridge/core/quadrature.py:168-189` (`_automatic_triangle_quadrature`): default `volume_degree = 2p`; a compact positive Dunavant rule is used while 2p ≤ 14 (`_DUNAVANT_RULE_BLOCKS` L48-96), otherwise the collapsed tensor Gauss rule `_triangle_quadrature(order, None)` with `_quadrature_point_count` = 2p+2 points per direction (L221-228). Exactness of n-point Duffy = 2n−2 (`_duffy_points_for_degree` L107-113: the Jacobian u costs one degree).
* The drift's degree is **not** accounted for: the runner builds the space with `volume_quadrature`/`volume_quad_1d` only (`scripts/guiding_center/runtime/runner.py:320-331`), and `transport/advection_field.py:1-25` states that the drift tables use the solution rule. Trace/face quadrature: 2p+1 GLL points (`DGTraceSpace`; NQF = 15 at p=7, 19 at p=9).

Host check (`ReferenceElementData` at p=1..9, `dub_orth`):

| p | NEL | NTR | NQV default (rule) | exact to | needed: mass 2p / advection 3p−1 | NQF | min. Duffy pts for 3p−1 | Dubiner tensor density (union / T0 / T1) | volume operator chosen (`use_volume_quadrature`) |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 21 | 6 | 25 (Dunavant 10) | 10 | 10 / 14 | 11 | 64 | 0.54 | quadrature |
| 6 | 28 | 7 | 33 (Dunavant 12) | 12 | 12 / 17 | 13 | 100 | 0.57 | quadrature |
| 7 | 36 | 8 | 42 (Dunavant 14) | 14 | 14 / 20 | 15 | 121 | 0.59 | quadrature |
| 8 | 45 | 9 | 324 (Duffy 18²) | 34 | 16 / 23 | 17 | 169 | 0.49 / 0.22 / 0.49 | tensor |
| 9 | 55 | 10 | 400 (Duffy 20²) | 38 | 18 / 26 | 19 | 196 | 0.50 / 0.22 / 0.50 | tensor |

So p ≤ 7 under-integrates the advection term (consistent variational crime, same discrete operator in every backend), p=8–9 is exact. The run records confirm: `transport_detail_raw_assembly_raw_volume_quadrature = 1` at p=6,7 (quadrature), tensor at p=8,9; `reference_advection_sparse_enabled = 0` everywhere (threshold `sparse_density <= 0.40`, `transport/cuda.py:221`).

### 1b. Fused kernel (`transport/raw_cuda.py`, used at p ≤ 7 in the campaign) [V]

One CUDA block per element (`assemble_advection_raw_fused`, L1311-1527; `element = blockIdx.x`), everything in dynamic shared memory: `local_lu` NEL², `local_rhs` NEL×(3NTR+1), τ/γ 2×3×NQF, coefficient caches (L1362-1385). Phases of `assemble_projected_local_advection_raw` (L685-995):
1. Coefficient cache: β coefficients (NB per component) and the affine-inverse-transformed `beta_ref0/1` (L760-766) — the J_K^{-T} contraction is folded into the coefficient vector once per element.
2. Volume operator, either by quadrature over the NQV solution-rule points in chunks of `RAW_QCHUNK` staged in the unused RHS block (L776-843), or by the reference-tensor contraction A_ij = Σ_k β_ref0[k]·T0[k,i,j] + β_ref1[k]·T1[k,i,j] (L912-950; `advection_tensor` (2,NB,NEL,NEL), read from global/L2). Reaction/mass part: `reaction_scalar·jac·mass_matrix[ii]`, diagonal for `dub_orth` (`mass_is_diagonal`), O(n).
3. Faces: τ = |β·n|, γ = τ − β·n at the 3×NQF face nodes, pre-multiplied by face Jacobian × weight (L853-895; conflict-averaged branch, zero-flux zeroing); upwind boundary mass over the upper triangle only (L957-976); trace columns B_e = ∫γ φ_i μ_j (L981-993); source column J·M·f (L900-910).
4. LU: cooperative partial-pivot LU `factor_local_lu_coop_pivot_scale_raw` (L1131-1225): block-wide pivot scan + tree reduction (log2(block) syncs), thread-0 row swap and tiny-pivot clamp (1e-30), parallel multiplier scaling and trailing update; ~5+log2(block) `__syncthreads` per pivot step, NEL steps.
5. Column solves `solve_all_columns_raw` (L1052-1090): **one thread per column**, NCOLS = 3NTR+1 (25 at p=7, 31 at p=9): each thread runs two dependent chains of ≈n²/2 FMAs with a `__syncthreads` per row.
6. Schur/emit (L1431-1525): one thread per trace row task (3NTR = 24–30 threads), lift row L = ∫τ μ φ_i by face quadrature, Schur row L·A⁻¹B, trace mass G, boundary elimination into the RHS; at NEL ≥ 45 the lift rows live in shared memory (`#if NEL >= 45`, L1377, 1433-1450). Output COO/CSR/BSR (`_raw_fused_csr_template` L1985, `_raw_fused_bsr_template` L2024).

Launch policy: fixed, no autotuner (`hdg/cuda/launch.py:35-47`: NEL ≤ 32 → 32, ≤ 64 → 64, p=8–9 → 128); `__launch_bounds__(128,3)` only for p=6/128 (`_apply_fused_assembly_launch_bounds` L1971-1983). Shared memory: `_fused_shared_sizes` L1935-1968 (21.9 KB at p=7, 40.9 KB at p=8, 56.6 KB at p=9).

### 1c. split3 = TSLE-BSR (`transport/tsle_bsr.py`, used at p=8–9) [V]

Same algebra and device routine, split into three launches, one block per element each, persistent workspace `E·(NEL² + NEL·NCOLS + 6·NQF)·8` B (6.38 GB at p=9, 5.42 GB at p=8 in the runs):
* `advection_tsle_build` (L225-311): runs `assemble_projected_local_advection_raw` and stores A_e, [B_e|f_e], τ/γ.
* `advection_tsle_solve` (L323-356): reloads A_e, cooperative LU, `solve_all_columns_raw` (the same 31-thread column solve), stores A_e⁻¹[B_e|f_e]; factors discarded.
* `advection_tsle_scatter_bsr` (L392-540): lifts, Schur rows, trace mass, boundary elimination, direct face-BSR stores (atomics on the diagonal blocks and RHS).
* Per-stage autotuner `_select_launch_blocks` (L689-890): candidates 32/64/128/256 screened on ≤32,768 elements, best two retimed on the full mesh. Selected in the runs: p=9 build 256 / solve 256 / scatter 32 or 128 (within 2 %), p=8 256/256/64; tuning wall 14.2 s (p=9), 9.0 s (p=8).
* `auto` → split3 from `RAW_SPLIT3_MIN_ORDER = 8` (`solvers/capabilities.py:465`), memory-aware split3 from p=6 when the workspace ≤ 25 % of free memory (L471-472, 475-507). `fast_profiles.py:130-133` hard-codes fused+csr (p ≤ 2), fused+bsr (3–7), split3+bsr (8–9).

### 1d. Why "validated to p = 7" [V]

`transport/raw_cuda.py:29-31`: "validated for legacy-lagrange and legendre-modal trace bases through p <= 7, with p=8--9 available for explicit experimental qualification". `docs/backends/raw_cuda.md:84-85, 111-116`: qualification sweep (Quadro RTX 6000) with seven samples for p ≤ 7 and five for p=8–9, p=8–9 "remains experimental"; the p ≥ 8 fused path is a different code path (128 threads, shared-memory lift rows, 40.9/56.6 KB dynamic shared memory above the 48 KiB default). `tests/test_advection_tsle_bsr.py:103-104` carries `p8-experimental`/`p9-experimental` parity cases (split3 vs fused). `validate_raw_cuda_supported` allows el_dof ≤ 55 (L2302, 2707, 2796). The limit is test/benchmark coverage, not a known defect; the campaign's p=8–9 path (split3) is parity-checked against fused at those orders.

### 1e. Resource facts at p = 7 and 9 (NVRTC compile only, no launch) [V]

Device: 110 SMs, 2.377 GHz, 96 MiB L2, 100 KB shared/SM (99 KB opt-in per block), 65,536 regs/SM, 384-bit bus (1,344 GB/s).

| kernel | regs | local bytes | dyn. shared | block | resident CTAs/SM (shared / regs) |
|---|---|---|---|---|---|
| p7 fused assembly | 254 | 288 (spill) | 21,952 | 64 | 4 / 4 (8 warps) |
| p7 tsle build / solve / scatter | 168 / 48 / 96 | 0 | 20,272 / 21,040 / 8,704 | 256 / 256 / 128 | 1 / 4 / 5 |
| p7 Poisson phase 1 (Cholesky) | 254 | 0 | 11,572 | 128 | 2 (regs) |
| p9 fused assembly | 80 | 0 | 56,604 | 128 | **1** (4 warps/SM) |
| p9 tsle build / solve / scatter | 48 / 56 / 126 | 0 | 41,648 / 41,388 / 15,856 | 256 / 256 / 128 | 2 / 2 / 4 |
| p9 Poisson phase 1 (Cholesky) | 162 | 0 | 25,984 | 128 | 3 |

### 1f. Poisson RHS refresh and reconstruction (`mixed/raw_cuda/poisson_source.py`, `poisson.py`) [V]

One kernel `poisson_local`, PHASE 0/1/2, one block per element, `BATCH` columns at a time. All volume integrals are reference precontractions (`poisson.py:83-100`) — no NQV dependence at run time. Phase 1 with the campaign's `schur-cholesky` policy (`RESPONSE=1`, `USE_FACTORS=1`, BATCH=1 since width=1, `configurations` L148-163): load the cached n×n Cholesky factor (L55-58), u = source (L158), **one thread** does the forward/backward substitution (L195-216), add responses·g (L218-227, g=0 here), q = J⁻¹(M⁻¹D u − …) (L229-237), lift the flux to 3NTR rows and `atomicAdd` into rhs (L260-280). Phase 2 (`RESPONSE`): no solve, u = source_solution + responses·trace, q via md, writes (u,qx,qy) (L250-256). The tuner (`_launch` L194-300) picked 128 threads at p=9, 64 at p=8, 32 at p=6; it cannot remove the serial solve.

### 1g. The operator form [V]

`scripts/guiding_center/time_schemes/si_bdf2.py:7-33` and `core/time_integration.py:43-82`: (I + (2Δt/3)·A(2v_n − v_{n−1})) ρ^{n+1} = (4ρ^n − ρ^{n−1})/3, i.e. β = (2Δt/3)(2v_n − v_{n−1}) enters A_e multiplicatively (volume term and the upwind face masses τ = |β·n|). The mass part is J·I (diagonal, reference) and is the only fixed piece; the factorization cannot be reused across steps.

## 2. Flop and byte model versus measured rates

Per-element FMA counts from the kernel loops (NB = NEL, same-space drift; NQF = 2p+1; volume = NQV·NEL² + NQV·(3NEL+2NB) when the quadrature operator is chosen, else 2·NB·NEL²) [V, counted from the code]:

| p | volume | faces (bmass, B cols, τ/γ, lifts, G) | LU | column solves | Schur | total FMA | kflop |
|---|---|---|---|---|---|---|---|
| 5 | 13,650 | 18,513 | 3,080 | 8,379 | 7,182 | 50,804 | 102 |
| 6 | 30,492 | 35,217 (15,834 / 7,644 / 2,184 / 7,644 / 1,911) | 7,308 | 17,248 | 12,936 | 103,201 | 206 |
| 7 | 61,992 | 62,010 (29,970 / 12,960 / 3,240 / 12,960 / 2,880) | 15,540 | 32,400 | 21,600 | 193,542 | 387 |
| 8 | 182,250 | 102,816 (52,785 / 20,655 / 4,590 / 20,655 / 4,131) | 30,360 | 56,700 | 34,020 | 406,146 | 812 |
| 9 | 332,750 | 162,450 (87,780 / 31,350 / 6,270 / 31,350 / 5,700) | 55,440 | 93,775 | 51,150 | 695,565 | 1,391 |

Measured (median synchronized kernel device times, run records `transport_detail_raw_assembly_raw_kernel_device`):

| p | E | kernel ms | µs/elem | Gflop | achieved TFLOP/s | split3 stages (ms): build / solve / scatter |
|---|---|---|---|---|---|---|
| 6 | 319,605 | 140.7 (fused, bs 32) | 0.440 | 66 | 0.47 | – |
| 7 | 249,084 | 208.7 (fused, bs 64) | 0.838 | 96 | 0.46 | – |
| 8 | 200,077 | 344.2 (split3) | 1.720 | 163 | 0.47 | 181 / 126 / 36 |
| 9 | 164,638 | 503.3 (split3) | 3.057 | 229 | 0.46 | 260 / 204 / 39 |

Card ceilings: FP32 65 TFLOP/s and 14,080 cores, 1,344 GB/s (NVIDIA datasheet via resellers: Puget Systems, gpupoet, SHI, NVIDIA product page); NVIDIA publishes no FP64 figure for RTX PRO cards; the 1:64 FP64 rate of GB202 and the measured DGEMM 0.74 TFLOP/s are from `advection_assembly_baseline_2026_10_03.md` → FP64 peak ≈ 14,080×2.377 GHz×2/64 ≈ 1.05 TFLOP/s. Memory bandwidth 1.344 TB/s. [I: the web check of the 1:64 ratio was cut short; the FP32/core/bandwidth figures were confirmed.]

Stage rates at p=9 [I from the counts]: build (volume + faces = 495 k FMA) 0.63 TFLOP/s = 60 % of peak, 85 % of DGEMM → compute-bound (and, with one 8-byte L2 load per FMA in the tensor contraction, ≈2.5 TB/s of L2 traffic: a possible co-limiter); solve (149 k FMA) 0.24 TFLOP/s → latency-bound; scatter (88 k FMA) 0.73 TFLOP/s. Floors at peak: 155 / 47 / 28 ms (230 ms total vs 503 measured). Bytes: ≈17 GB of workspace traffic per assembly (12.7 ms) → not memory-bound; the 2.66 MB tensor is L2-resident. NQV only matters at p ≤ 7 (quadrature operator: 32 % of the p=7 flops).

Latency calibration [I]: Poisson phase 1 at p=9 does 3,025 dependent LDS+FMA iterations on one thread per element with 3 resident CTAs/SM: 82 ms ⇒ ≈120 cycles per dependent iteration. Applied to the split3 column solves (2,970 dependent iterations per column, 2 CTAs/SM): ≈110 ms of the 204 ms solve stage; the p=7 fused column solves (1,260 iterations, 25 threads) ≈ 36 ms of 209 ms; p=6 ≈ 18 ms of 141 ms. The LU itself (55 pivot steps × ~13 block syncs, thread-0 row swaps) ≈20–25 ms at p=9.

Poisson phase 1 floors at p=9: ≈12 k FMA/elem → 4 ms; 6.3 GB (factors 3.99 GB + responses 2.17 GB) → 4.7 ms; measured 82 ms kernel (87 ms phase). Poisson reconstruction 14 ms (no solve, ≈2.4 GB) is acceptable.

Transport solve at p=9 (163 ms): row scaling 23.5 ms + AMGX BiCGSTAB 128 ms (38.5 iterations; 27–29 with the other Poisson candidate) + upload/setup/unscale/validation ≈12 ms. SpMV floor ≈1 GB per BSR pass (247,058 block rows × 5 blocks × 100 doubles) → 77 passes ≈ 57 ms; achieved ≈45 % of the SpMV bandwidth bound.

## 3. Options for p = 7–9

(a) Volume quadrature. Minimal exact rules: mass 2p (p+1 Duffy points per direction), advection 3p−1 with a degree-p drift (Duffy 14² = 196 at p=9, 13² = 169 at p=8; a degree-(p−1) compatible drift needs 3p−2 → the same Duffy counts). The (p+1)² rule is exact to 2p, not 2p+1; at p=8–9 it would under-integrate the advection term (18 < 26) **and leave the assembly cost unchanged**, because `use_volume_quadrature` (`transport/cuda.py:322-344`, margin 1.25) still keeps the tensor (p=9: quadrature 330 k vs tensor 333 k; p=8: 182 k vs 182 k). Optimal symmetric rules of degree 26 need ≈130–140 points (the repo has none above 14) → 449 k FMA > 333 k tensor, so for an exact operator the tensor stays the cheapest at p=9. Sum factorisation does not reduce the cost of forming the dense n×n local matrix with a variable coefficient; it helps matrix-free application, which static condensation does not use. Gain for the step: none at p=8–9; the only gains are in diagnostics (section 4). Risk: aliasing at p=8–9 if the rule drops below 3p−1. Effort: 0 (CLI).

(b) Affine reference tensors: already in place (`transport/advection_field.py:118-135`, `transport/cuda.py` `reference_advection_tensor_cupy`, kernel L912-950; meshes are affine: `core/mesh.py:310, 505-514`). Not exploited: (i) the degree-orthogonality rule — every nonzero of T0 and T1 has deg_k ≤ deg_i + deg_j − 1 and deg_j ≤ deg_k + deg_i − 1, modes ordered by total degree (`core/basis.py:483-485`), so a prefix bound kmax(i,j) on the k loop removes 10.2 % of the dense work exactly (0.898 of dense at p=9, 0.885 at p=7) with no indirection; (ii) per-direction sparsity: T0 (x-derivative) is 22 % dense, T1 50 %, T0's pattern ⊂ T1's; the existing sparse path (`reference_advection_sparse_cupy` L161-252) uses the union pattern for both directions, which saves nothing at p=9, which is why it is disabled. Splitting the k lists per direction gives 0.72 of the dense flops (−93 k FMA/elem) at the cost of indexed loads for T0 only. Estimated gain at p=9: −12 ms (prefix) to −50 ms (prefix + per-direction lists) of the 260 ms build. Risk: low (exact zeros only; validate to 1e-15 relative, not bitwise). Effort: 1 day (prefix table in `_advection_launch_inputs`/kernel loop bound) + 2–3 days (per-direction sparse lists + tests `test_transport_cross_space_advection`, `test_cupy_backend`).

(c) Local factorisation and Schur. The LU cannot be reused across steps (β in A_e, 1g). The dominant inefficiency is the serial column solve (`solve_all_columns_raw`, 31 threads of 256). Two remedies: (1) warp-per-column solves with multirow registers — `hdg/cuda/raw_source.py:239-345` (`RAW_WARP_COLUMN_SOLVES`, `solve_lu_columns_warp`, NEL ≤ 55) already does this for ADR; porting it into the TSLE solve kernel and the fused kernel would turn 2,970 dependent iterations per column into ≈55 shuffle steps, est. solve stage 204 → ≈90 ms (p=9), 126 → ≈65 ms (p=8), fused p=7 −30 ms; also replace the 8-step shared-memory pivot reduction with warp shuffles. Risk low (same algebra, different summation order → 1e-15); effort 2–3 days. (2) MAGMA batched `getrf/getrs` (`linalg/gpu/magma_batched.py:136-175`, `trans=`): measured stage 2 at p=9 on 117 k elements: coop 144.1 ms, cuBLAS 118.0, MAGMA 76.5 (baseline doc, stage-2 table) → 107 ms at 164,638 elements (−97 ms); at p=8 only 60.2 vs 72.5 ms, at p=7 MAGMA is slower (51 vs 35). Needs a column-major response: write `[B_e|f_e]` column-major in the build kernel (as `tsle_tensor.py:625-690` does), A_e row-major = A_eᵀ column-major → factor and solve with `trans=True`; the scatter then reads contiguous columns. Risk: moderate (optional dependency built from source; different pivot sequence → ≤1e-10 relative differences at p=9); effort 2 days. Recommendation: do (1) first (helps p=7–9 and fused), then measure MAGMA at p=9 only.

(d) Fused vs split3 at p=8–9: the code's `auto` chooses split3 from p=8 on FP64-limited cards (`capabilities.py:458-465`; measured 24 %/44 % faster; `cuda_execution.md`). Cause verified by the compile check: the fused p=9 kernel needs 56.6 KB shared → one 128-thread CTA per SM (4 warps), while split3's build/solve run two 256-thread CTAs per SM and the scatter four; register pressure is not the issue at p=8–9 (48–80 regs, zero local memory). Stage C should therefore see fused ≈1.3× (p=8) to 1.8× (p=9) slower. At p=7 the fused kernel is register-limited (254 regs, 288 B spill, 4 CTAs of 64 threads); `auto` would pick split3 there (memory-aware rule, 8–10 % faster measured) — the campaign's explicit `fused+bsr` at p=7 forgoes that.

(e) Other findings. (i) Poisson phase 1 (1f): parallelise the substitution (warp shuffle, or cuBLAS `dtrsmBatched` as in `mixed/cupy.py` `_batched_cublas_cholesky_solve`) → 87 → ≈10–15 ms at p=9, 49 → ≈8 ms at p=7; skip the `responses·trace` term when `boundary_is_zero`. Risk low, effort 1–2 days. (ii) Transport solve: per-step BSR scaling + unscaling costs 26 ms at p=9 (`transport_detail_solve_scale` 23.5 ms); the unpreconditioned scaled BiCGSTAB needs 29–38 iterations at tolerance 1e-14 (`configs/amgx/adv_rea_gpu4_hdg_bicgstab_scaled_none.json`); `pbicgstab_block_jacobi_bsr.json` / `pbicgstab_l1_bsr.json` (tolerance 1e-11) are the campaign's `bj`/`l1` modifiers — a 10×10 block-Jacobi apply is negligible per iteration, so halving the iterations and dropping the scaling is plausible (−40…−70 ms) but unmeasured; loosening 1e-14 → 1e-10 would cut ≈25–30 % of the iterations but weakens the round-off closure of the L2 identity (currently 2e-16). (iii) BSR block writes are small (900 doubles per element) — not a cost. (iv) Face terms: 162 k FMA/elem at p=9 (23 % of the kernel); a 15-point GLL face rule (exact to 27 ≥ 3p when β·n keeps its sign) instead of 19 would save ≈5 % of the build, but it is a trace-space-wide setting shared with Poisson and diagnostics — low priority. (v) The p=7 Poisson candidate `amgx-classical-csr` rebuilt its hierarchy every step (524 ms solve) — a candidate-selection artefact, not an assembly matter. (vi) `--volume-quad-1d n` forces the n² Duffy rule (exact to 2n−2): 14 at p=9 / 13 at p=8 keeps the exact tensor and halves NQV; at p ≤ 7 any exact choice switches the kernel to the tensor path (+16 % assembly flops at p=7).

## 4. Out-of-step diagnostics at p=9

Enstrophy budget (209 ms/step; 36 ms at p=6): `EnstrophyBudget.step` (`scripts/guiding_center/runtime/diagnostics.py:108-160`) calls `transport_enstrophy_terms` (`hybridge/diagnostics/hdg_identities.py:75-135`): face einsums (cheap, NQF) plus the volume term at the solution rule (L118-126): `z @ basis`, `einsum("ki,qid->kqd")` (E×NQV×2 temporaries of 1.05 GB), `bx @ advection_volume`, `by @ ...`, products → ≈1.8e10 FMA on the 400-point rule plus ≈6–8 GB of temporaries, then `field_l1_quadrature` (NQV again), five `field_l2_squared` (cheap), four field combinations and two `.copy()`. It runs **every step** (`runner.py:791`, "every step: the identity needs the accepted history"), independent of `--diagnostics-every`. [I] ≈60–70 % of it scales with NQV.

Other diagnostics (309 ms; 46 ms at p=6), gated by `diagnostics_every` (`runner.py:888-889`, 954-966): `guiding_center_field_diagnostics` (device extrema sampled at the volume points), `transport_velocity_diagnostics` (`diagnostics/guiding_center.py:400+`: divergence and speed via two `einsum("ki,qid->kqd")` on NQV points — the largest pass), `azimuthal_mode_diagnostics` (perturbation at NQV points + arctan2 of E×NQV mapped points), `positivity.measure` (volume + lattice samples + Bernstein transform per chunk), polar Fourier `DiocotronModeDiagnostics.measure` (sparse sampling matrix, 32×128 points + rfft — negligible), wall balance (faces), record writing. The E×NEL×NQV passes dominate; at p=9 each costs ≈3.6e9 FMA and a 0.5 GB temporary.

Cheapest fixes: `--volume-quad-1d 14` (p=9) / `13` (p=8) → NQV 400→196 / 324→169 with an unchanged (still exact) operator and step time; expected enstrophy ≈120 ms, other ≈180 ms [I]. Structural fix: evaluate −2(z, b·∇z)_Q + Σ∮ s z² from the assembled local operator A_e kept in the split3 workspace (zᵀA_e z minus the mass and face parts), one E×NEL² pass (≈2 ms), bit-consistent with the solve's pairing — ≈2 days, moderate risk.

`--diagnostics-every` for 16,000 steps at p=9 (0.81 s/step → 3.6 h): every 10 → 1,600 records, 8.3 min (+2.3 %), 100 samples in the [20,45] growth window (dt 0.025); every 20 → 4.1 min; every 50 → 1.7 min. Recommend 10–20. The enstrophy budget adds 56 min if on; disable the preset field `enstrophy_budget` for the long run (keep it in the short validation runs) or apply the quadrature/operator fixes above.

## 5. Recommendation

Current p=9 step (810 ms): transport assembly 508 (build 260, solve 204, scatter 39, other 5) + transport solve 163 (scale 24, Krylov 128, misc 12) + transport reconstruction 3 + Poisson RHS 87 + Poisson solve 25 + Poisson reconstruction 14 + ≈10 other.

1. Warp-parallel column solves (and shuffle pivot search) in `advection_tsle_solve` and `solve_all_columns_raw`, reusing `RAW_WARP_COLUMN_SOLVES`; then compare MAGMA stage 2 at p=9. Gain ≈ −95…−110 ms (p=9), −55 ms (p=8), −30 ms (p=7 fused). Effort 2–3 days. First validation: BSR data/RHS/local response of a 200-element host-verifiable mesh against the current split3 and fused kernels at ≤1e-15 relative (exact match impossible: summation order), `tests/test_advection_tsle_bsr.py` and `test_cupy_backend.py` parity at p=1..9; then the campaign consistency check (mass drift, L2-identity defect ≈1e-16, identical BiCGSTAB iteration counts).
2. Parallel triangular solve in Poisson phase 1 (`poisson_source.py:195-216`), skipping the zero-trace response term. Gain ≈ −72 ms (p=9), −40 ms (p=7). Effort 1–2 days. Validation: condensed RHS against the current kernel at ≤1e-13 relative on a small mesh (`tests/test_poisson_high_order_cuda.py` cached-RHS cases), then the campaign check (Poisson iterations unchanged).
3. Tensor structure in the build (degree-prefix k bound, then per-direction sparse T0). Gain ≈ −12…−50 ms at p=9, −8…−30 ms at p=8. Effort 1 + 2–3 days. Validation: condensed matrices equal to the dense-contraction kernels to 1e-15 relative (`test_transport_cross_space_advection` covers tensor vs quadrature operator parity).
4. Zero-code: `--volume-quad-1d 14/13` at p=9/8 for the diagnostics (step unchanged), `--diagnostics-every 10`, enstrophy budget off for production runs; and measure `fused+bsr` → `auto` (split3) at p=7 and the `bj` transport modifier without scaling at p=8–9.

Expected p=9 step after 1–3: ≈ 810 − 100 − 72 − 40 ≈ 600 ms (−26 %), assembly ≈ 3.06 → ≈2.2 µs/element; with a preconditioned, unscaled transport solve ≈ 540–560 ms. p=7: 211 → ≈180 ms assembly, Poisson RHS 49 → ≈8 ms. The build stage will then be the limiter at ≈0.6–0.65 TFLOP/s, i.e. within ≈1.6× of the FP64 peak of this card; further gains need fewer flops (lower-degree drift NB, or FP64-capable hardware), not better scheduling.

Sources: NVIDIA RTX PRO 5000 Blackwell datasheet figures as reproduced by Puget Systems, gpupoet, SHI and the NVIDIA product page; FP64 ratio and DGEMM rate from `docs/research/solver_studies/advection_assembly_baseline_2026_10_03.md`.
