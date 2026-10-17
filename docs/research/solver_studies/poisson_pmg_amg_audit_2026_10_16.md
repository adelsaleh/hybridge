# pMG-AMG for the HDG Poisson trace system at p = 7–9 (and 5–6): code state, measured bounds, literature, what to build — 2026-10-16

Audit by a research agent (no file edited; no simulation, GPU run, pytest or AMGX build; two CPU-only
checks on a 200-triangle mesh, scripts `pmg_small_check.py`, `pmg_eig_dist.py`, `pmg_tau_scan.py` in the
session scratchpad). Data: the D8-S smoke campaign (`gc-hdg-article/run_outputs/diocotron_order_smoke_20261016`,
9.0 M element unknowns at every p, `--poisson-tau 128000`). Scope: p = 5–9.

## Executive summary

1. hybridge already contains a pMG-AMG: `fb-hp-mg-pcg` = modal p-coarsening on the fixed face graph (Galerkin principal blocks), face-block-Jacobi Chebyshev smoothing, one fixed AMGX classical V-cycle on the p = 0 (one scalar per edge) operator, flexible PCG outside. Its kernels already support block sizes 1..10 (p ≤ 9); the only p-gate is `face_hp.py:736` (`4 <= degree <= 6`) plus preset/runner gates.
2. At the campaign's stabilization (`--poisson-tau 128000`, i.e. τh ≈ 850–4000) the current design is structurally wrong, independent of p: the p = 0 (edge-constant) coarse space misses the continuous piecewise-linear (vertex) trace modes, which dominate the low end of the spectrum when τh ≫ 1. Host check (p = 6): κ(BA) = 5.2 at τ = 1 but 583 at τ = 128000 with exactly one bad eigenvalue (≈2·10⁻³) per interior vertex. This explains the 75–127 cold iterations of `fbmg-*` at p = 4/6 in D8-S (41–49 at τ = 1 in the 300k study). A ladder ending at p = 1 (two modes per edge) keeps κ ≈ 5–7 for p = 4..9 at τ = 128000 (model: ~20–37 PCG iterations cold instead of 170–450).
3. All published HDG/SEM pMG practice coarsens to degree 1 (continuous P1 or linear traces), never to P0 (Wildey et al. 2019, Kronbichler–Wall 2018, Cockburn et al. 2014, Phillips–Kerkemeier–Fischer 2022/NekRS); block (face) smoothers are required; iteration counts are then ~p-independent (4–21 Krylov iterations).
4. D8-S at p = 9: warm Poisson Krylov is 19 ms of an 810 ms step (`amgx-cheb-csr`); transport is 681 ms (assembly 508 ms, solve 163 ms, 38.5 unpreconditioned BiCGSTAB iterations). A Poisson pMG changes nothing in the warm phase and nothing about transport.
5. The `amgx-classical-csr` results at p = 7/9 (14–15.5 warm iterations, 0.47–0.52 s/step) are a stopping-rule artifact: that JSON uses `RELATIVE_INI`, tolerance 1e-13 (`configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json`), kept verbatim by `device_solver.py:71-72`; the solves end 5 orders of magnitude below the target (`max_poisson_residual_ratio` 1.6e-5–2.1e-5). With an absolute-tolerance config (`..._classical_gs_robust_abs.json` exists) or `amgx-cheb-csr` (not yet run at p = 7/8), the p = 7 step should drop from 905 ms to ≈ 410–430 ms. This is the first thing to run.
6. Cold start at p = 9: `amgx-cheb-csr` costs 15.5 s for the first Poisson (AMGX hierarchy setup ≈ 12.1 s, Krylov 1.15 s/18 it.); `amgx-classical-csr` 2.5 s (setup 0.22 s, Krylov 0.90 s/28 it.). A native pMG at p = 9 would set up in ≈ 0.3–0.5 s (p = 6 measured: 0.16–0.24 s) and, with the p→1 ladder, solve cold in ≈ 0.25–0.45 s (25–40 it. × ~10 ms). Gain: 12–15 s per cold start vs cheb, ~0.5 s vs classical; per step in a saturated phase with a poor guess: ≈ 0.5–0.9 s saved vs classical, 0–0.9 s vs cheb (cheb's cold per-iteration cost is ambiguous, 13–64 ms).
7. For p = 5–6 the same p→1 change would cut the cold/poor-guess Poisson Krylov from 0.58–0.89 s (75–127 it.) to ≈ 0.15–0.25 s; warm steps (1–4 it.) are unchanged.
8. Memory at p = 9: pMG needs ≈ +1.4 GB (orthonormal BSR copy 0.96 GB, diagonal inverses 0.2 GB, vectors) against the AMGX cheb hierarchy's ≈ +2.8 GB (peak 31.2 vs 28.4 GB) and the scalar CSR index overhead (0.5 GB).
9. Coarse operators must be the Galerkin principal blocks (free: views of the fine BSR data), not re-assembled lower-degree HDG operators: at τ = 128000 the re-assembled P0 operator is 20–60× stiffer than the principal block (generalized eigenvalues 0.015–0.057) and makes the cycle 3–7× worse; re-assembled P1 coincides spectrally with the Galerkin P1 block in the cycle.
10. Recommendation: (1) run `amgx-cheb-csr`/`-bsr` and an absolute-tolerance classical config at p = 7–8 now (config-only); (2) build the "terminal p = 1" variant of `fb-hp-mg-pcg` with AMGX on the 2×2-block p = 1 operator and open the gate to p = 2..9 (≈ 2–3 days code + tests, device validation by the user); (3) only then compare across p. Do not extend p→0 to p = 7–9 as is.

## 1. Current state of the code (verified)

### 1.1 `fb-hp-mg-pcg` (`hybridge/linalg/multigrid/face_hp.py`, `policy.py`, `hybridge/linalg/gpu/legendre_face_bsr.py`)

* **Input and basis.** Face-BSR trace operator from the raw-CUDA Poisson assembler (block size b = p+1, modal Legendre edge basis `legendre_edge_basis`, `mixed/cupy.py:464-469`), `scale_system=False`, boundary eliminated. `FaceBlockHpMgPcgSolver.__init__` (`face_hp.py:716-809`) rescales once to the reference-orthonormal modal basis `S A S`, `S_jj = sqrt((2j+1)/2)` (`legendre_face_bsr.py:45-69`, `face_hp.py:748-753`) and keeps the Krylov workspace (8 vectors).
* **Hierarchy (p only, same face graph).** `modal_degree_schedule` (`legendre_face_bsr.py:72-88`): `"direct-to-zero"` → (p, 0) or `"halve"` → (p, p//2, …, 1, 0). Both end at p = 0; `FaceBlockPmgPrototype` asserts the terminal block size is 1 (`face_hp.py:519-521`). No h-coarsening in hybridge; all h-coarsening is delegated to AMGX at p = 0.
* **Transfers.** Nested modal injection: restriction keeps the lowest modes of each face block (`face_hp.py:630-636`), prolongation adds the coarse correction into the low modes (`face_hp.py:640-645`), `R = Pᵀ`. Coarse operators are Galerkin products `Eᵀ A E` = leading principal sub-blocks of every BSR block (`principal_modal_bsr_data`, `legendre_face_bsr.py:91-100`); derivation in `docs/algorithms/hp_amg/hp_amg.tex:225-240`.
* **Smoother (p > 0 levels).** Block-Jacobi (dense (p+1)×(p+1) face diagonal inverse, symmetrized, `face_hp.py:475-486`) preconditioned Chebyshev–Richardson of fixed order m: weights = reciprocals of the Chebyshev roots on [0.1 λ_max, λ_max], λ_max from a 12-step power iteration on D⁻¹A times a 1.2 safety factor (`face_hp.py:377-419`, `491-501`). 1+1 (standard/fast) or 2+2 (robust) sweeps; post-smoothing reverses the weight order so the cycle is symmetric. Fused warp-per-face CUDA kernels for block sizes 1..10 (`legendre_face_bsr.py:419-505`, gates at `694` and `762`); fine SpMV via CUDA-13 generic cuSPARSE BSR (`legendre_face_bsr.py:214-376`) with a raw fallback for 1..10 (`539-550`).
* **Coarsest level.** `AmgxScalarVcycle` (`face_hp.py:236-271`): requires block size 1 (`241`), builds a reusable `PyAMGXCsrDeviceSolver(fixed_amg_cycles=1, maxiter=1, tolerance=1e-30)` → exactly one AMGX classical V-cycle per application. Config `scalar_p0_amgx_config` (`policy.py:8-29`): PMIS, AHAT strength 0.40, D2 interpolation, JACOBI_L1 1+1, DENSE_LU at 128/256 rows, no aggressive level, `error_scaling 0`; robust variant 2+2 / coarsest 4 (`policy.py:32-36`).
* **Policies** (`policy.py:39-80`): `standard` = direct p→0, Chebyshev order 2, 1+1; `fast` = direct p→0, order 1, 1+1; `robust` = halve, order 4, 2+2, robust coarse config. `fbmg-standard`/`fbmg-fast` in `fast_profiles.py:94-95,157` map to these.
* **Outer iteration.** Flexible PCG (AMGX-style β, `face_hp.py:231-233`, `1066-1082`) with a true-residual refresh every 10 iterations against the original assembled matrix (`diffusion_raw_cuda.py:372-392`), best-iterate checkpoint, direction restart on a >10 % residual gap, stagnation stop after six checks; symmetry (≤1e-10) and positive-curvature gates at setup (`face_hp.py:774-793`); any failure falls back to the hybrid AMGX path and is remembered per operator key (`diffusion_raw_cuda.py:484-507`).
* **Restrictions.** `4 ≤ p ≤ 6` (`face_hp.py:736-739`); BSR + `legendre-modal` + `scale_system=False` (`diffusion_raw_cuda.py:41-73`); runner forces `amgx` outside 4..6 (`scripts/guiding_center/runtime/configuration.py:393-398`); preset ranges `range(4, 7)` (`fast_profiles.py:59-60,94-95`); capability note (`solvers/capabilities.py:349-356`). The CODEMAP/plan state the kernels are ready through p = 9 (`docs/development/plans/face_block_hp_multigrid.md`, "Long-Term Degree Extension").
* **Memory.** Orthonormal copy of the BSR data (same size as the assembled matrix), per-level principal-block copies (fine level shares), diagonal inverses N_faces·b²·8 B, 4 vectors per non-terminal level, 8 Krylov vectors, AMGX p = 0 hierarchy (~50–60 MB at 235k faces).
* **Recorded performance (τ = 1 studies).** p = 4–6 on 157k triangles: native 85–114 ms per complete call vs hybrid AMGX 96–131 vs scalar CSR 114–195; iterations 20–23 native vs 21–24 AMGX; the coarse AMGX application is 52–56 % of native time (2.07–2.39 ms per call). 315k triangles p = 6: pMG standard 41 iterations (0.34 s) vs AMGX hybrid 21 (0.29 s); Chebyshev-1 49 iterations but 0.313 s (`poisson_300k_solver_comparison_2026_09_22.md`). ITER 364k, p = 6: fast 38 it. 0.254 s vs robust 24 it. 1.359 s (`iter_repeated_poisson_2026_09_24.md`). One-shot (setup-inclusive) solves: the hybrid AMGX path won all 18 cases of the 2026-08 sweep.

### 1.2 The AMGX candidates (`scripts/guiding_center/runtime/fast_profiles.py:93-106`, `configs/amgx/`)

| candidate | config | fine format / basis | AMG | smoother | stopping |
|---|---|---|---|---|---|
| `amgx-cheb-csr` | `diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json` | scalar CSR, nodal | classical PMIS/AHAT 0.25/D2, 1 aggressive level, dense LU 2048/4096, error_scaling 3 | 0+2 sweeps of CHEBYSHEV order 2 with scalar-row JACOBI_L1, `chebyshev_lambda_estimate_mode 2` | PCGF, **ABSOLUTE 1e-12** |
| `amgx-cheb-bsr` | `…_abs_scalar_expand_bsr.json` | face BSR (generic cuSPARSE SpMV), nodal | same hierarchy built on the scalar expansion (`classical_bsr_hierarchy: scalar_expand`, fork) | same | same, `use_scalar_norm 1` |
| `amgx-classical-csr` | `diff_rea_gpu4_hdg_pcgf_classical_amg.json` | scalar CSR, nodal | classical PMIS/AHAT **0.75**/D2, no aggressive level, dense LU 2048/4096 | 1+1 symmetric MULTICOLOR_GS | PCGF, **RELATIVE_INI 1e-13**, max 2000 |
| `amgx-aggbj-bsr` (p ≤ 4) | `…_aggregation_block_jacobi_bsr.json` | BSR | aggregation SIZE_2 | 0+3 BLOCK_JACOBI 0.8 | RELATIVE_INI 1e-10 |
| `amgx-bj-bsr` (p = 5–6) | `…_block_jacobi_bsr.json` | BSR | none | BLOCK_JACOBI | RELATIVE_INI 1e-10 |

`_amgx_config_for_solve` only fills `tolerance` when the JSON has none (`hybridge/linalg/amgx/device_solver.py:62-84`), so the JSON stopping rule is what runs; the runner's rtol/atol 1e-11/1e-12 (`configuration.py:16-17`) are only the external acceptance check. Consequence (measured, §2): the classical candidate iterates to 1e-13 of the *initial* residual of a warm start and ends 5 orders below target; cheb-abs stops as soon as ‖r‖ ≤ 1e-12 absolute, i.e. after 1–2 warm iterations. An absolute-tolerance classical config already exists: `diff_rea_gpu4_hdg_pcgf_classical_gs_robust_abs.json` (PMIS 0.75, GS 2+2, dense LU 128/256, ABSOLUTE 1e-12) but was not a candidate.

### 1.3 AMGX fork block-size support (b = 8, 9, 10)

Verified in `~/src/AMGX-hdg-cuda13`:
* MULTICOLOR_DILU: compiled block sizes 1, 2, 3, 4, 5, 7, 8, 10 (`src/solvers/multicolor_dilu_solver.cu` setup switch 4011–4102, forward 3184–3398, backward 3570–3738; `LOCAL_CHANGES.md:106-121`). So DILU works at p = 7 (8) and p = 9 (10) but not p = 8 (9) nor p = 5 (6).
* BLOCK_JACOBI: 1, 2, 3, 4, 5 specialised, any b > 5 through the generic "big block" path (`block_jacobi_solver.cu:1072-1094`); fused small-block kernels only 2, 3, 5 (`1540`).
* JACOBI_L1 with `jacobi_l1_scalar_rows_for_blocks`: generic in the block dimension (`jacobi_l1_solver.cu:60-91`); CHEBYSHEV wraps any of these (`cheb_solver.cu`); CHEBYSHEV_POLY is scalar-only and rejected by hybridge for BSR (`device_solver.py:111-133`).
* Aggregation AMG on BSR: restriction/prolongation compiled for 2, 3, 4, 5, 8, 10 only (`aggregation_amg_level.cu:480-515`) → usable at p = 7 and 9, not 8 (and not 5, 6).
* Classical AMG needs scalar matrices (`csr_multiply.cu:103-105, 242-244`); on BSR the fork expands to scalars (`classical_bsr_hierarchy: scalar_expand`, `classical_amg_level.cu:1036`), so block size is unrestricted but the hierarchy is scalar.
* MULTICOLOR_GS: 1, 3, 4 specialised, generic square-block fallback (`multicolor_gauss_seidel_solver.cu:1455-1490`).

### 1.4 How the campaign ranked candidates (before 2026-10-16 evening)

`hard = step − warm Poisson Krylov + cold (first-solve) Poisson Krylov`: why `amgx-classical-csr` (hard 1.65 s) beat `amgx-cheb-csr` (1.94 s) at p = 9 despite 0.81 s vs 1.23 s per step. (Since replaced: ranking by step time, saturated-phase stage H.)

## 2. D8-S breakdown (`runs/d8s_summary.json`, medians of 6 synchronized steps; ms unless noted)

| p | Poisson cand. | triangles | trace unknowns | step | Poisson wall | Krylov | RHS | recon. | warm it. | transport wall | asm | solve | tr. it. | first Poisson wall | cold Krylov | cold it. | operator asm. | solver setup* | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | fbmg-fast | 598,613 | 4,490,620 | 274 | 76.7 | 31.3 | 31.5 | 11.2 | 4 | 197 | 82.5 | 105.2 | 20.5 | 1.39 s | 0.85 s | 127 | 0.31 s | 0.18 s | 12,644 |
| 4 | fbmg-standard | | | 272 | 88.0 | 40.5 | 32.9 | 10.8 | 3.5 | 186 | 81.8 | 99.8 | 19.5 | 1.37 s | 0.89 s | 82 | 0.25 s | 0.19 s | 13,814 |
| 6 | fbmg-fast | 319,605 | 3,356,871 | 302 | 57.3 | 7.9 | 36.5 | 9.3 | 1 | 244 | 142.7 | 95.2 | 27 | 1.25 s | 0.58 s | 119 | 0.47 s | 0.16 s | 13,292 |
| 6 | fbmg-standard | | | 321 | 70.2 | 19.7 | 36.9 | 9.4 | 2 | 251 | 143.1 | 103.7 | 28.5 | 1.54 s | 0.59 s | 75 | 0.64 s | 0.24 s | 14,458 |
| 7 | amgx-classical-csr | 249,084 | 2,990,024 | 905 | 590.8 | 516.4 | 48.5 | 15.6 | 14 | 305 | 211.1 | 87.9 | 21 | 2.46 s | 1.00 s | 28 | 1.00 s | 0.33 s | 17,636 |
| 8 | amgx-classical-csr (killed after 3 steps) | 200,077 | – | 1320–1410 | 610–720 | – | – | – | 16–18 | 490–520 | – | – | 27–32 | 3.7 s (initial) | – | 22 (step 1) | – | – | – |
| 9 | amgx-cheb-csr | 164,638 | 2,470,580 | 810 | 128.4 | 18.9 | 87.6 | 14.0 | 1.5 | 681 | 508.3 | 163.2 | 38.5 | 15.55 s | 1.15 s | 18 | 2.17 s | **12.1 s** | 31,196 |
| 9 | amgx-classical-csr | | | 1228 | 581.2 | 472.2 | 86.8 | 14.2 | 15.5 | 646 | 502.6 | 131.0 | 29 | 2.53 s | 0.90 s | 28 | 1.34 s | 0.22 s | 28,360 |

\* setup = `first_poisson_time_solve − first_poisson_krylov_time` (AMGX hierarchy build, or native hierarchy + p = 0 AMGX).

Readings:
* **Warm phase.** At p = 9 the Poisson Krylov is 2.3 % of the step; RHS assembly (87.6 ms, cached Schur-Cholesky path) + reconstruction (14 ms) are 12.5 %; transport is 84 % (assembly 63 %, solve 20 %). At p = 7 the Poisson Krylov is 57 % of the step only because of the classical config's over-solving (§1.2); with a 1–2-iteration warm solve the p = 7 step is ≈ 905 − 516 + 25 ≈ 410–430 ms.
* **Per-iteration costs (warm).** classical p = 9: 30.5 ms; p = 7: 36.9 ms; cheb p = 9: 12.6 ms (18.9/1.5); fbmg p = 6: 7.9–9.9 ms; p = 4: 7.8–11.6 ms. Cold per-iteration: classical 32 ms (p = 9), 36 ms (p = 7); fbmg 4.9–7.9 ms (p = 6), 6.7–10.9 ms (p = 4); cheb p = 9 **64 ms** (1.149 s/18) — inconsistent with its warm 12.6 ms and with p = 1/2 (cold = warm ≈ 16 ms), so the cheb cold Krylov at p = 9 most likely contains ~0.9 s of one-time work in the first solve (the `chebyshev_lambda_estimate_mode 2` spectral estimates on each level are the obvious suspect; not verified).
* **Cold start.** The cheb hierarchy setup at p = 9 is 12.1 s (strength 0.25 on 124 M nonzeros with 10×10-dense rows, D2 interpolation, aggressive level); classical (0.75) 0.22 s. Each `poisson_tau` retry rebuilds the hierarchy.
* **Transport.** 38.5 BiCGSTAB iterations at p = 9 with no preconditioner (`adv_rea_gpu4_hdg_bicgstab_scaled_none.json`, RELATIVE_INI_CORE 1e-14) cost 163 ms; split3 assembly 508 ms. A Poisson pMG does not touch any of this.

## 3. Small host check (200 triangles, unit-square mesh, Legendre-modal traces, numpy assembly; CPU, 3 min)

Script `pmg_small_check.py`: builds the reduced trace operator A_p for p ∈ {0,1,2,3,4,6,9}, τ ∈ {1, 128000}; forms the V-cycle preconditioner B as a dense matrix with the code's smoother definition — block-Jacobi Chebyshev, λ_low = 0.1 λ_max, λ_max = exact·1.2, 1+1 sweeps, reversed post-sweep, nested injection — and an **exact** coarsest solve; reports κ(BA) via eigvalsh(LᵀBL), A = LLᵀ, and the Chebyshev PCG iteration bound for 1e-8/1e-12.

**(a) Galerkin principal block vs re-assembled lower degree (same mesh, same τ).** Generalized eigenvalues of (EᵀA_pE, A_q):

| | q = 0 | q = 1 | q = p/2 |
|---|---|---|---|
| τ = 1, p = 9 | [3.25, 5.75] | [1.00, 6.39] | [1.00, 5.07] |
| τ = 128000, p = 9 | [0.0154, 0.0173] | [0.0308, 1.00] | [0.185, 1.00] |

The re-assembled P0 HDG operator is 20–60× stiffer than the constant-mode principal block at τ = 128000, and P1 is not spectrally equivalent either (ratio 10–32). In the cycle, a re-assembled A_0 coarse operator raises κ(BA) from 306 to 1929 (p = 9, Cheb-2, τ = 128000) and from 6.3 to 10.6 at τ = 1; a re-assembled A_1 gives the same κ as the Galerkin p→1 block (7.36 vs 7.32). Conclusion: keep Galerkin principal blocks.

**(b) Cycle quality vs p, ladder and Chebyshev order (exact coarse solve; PCG iterations predicted for 1e-8 / 1e-12 relative).**

| τ | ladder | Cheb | p = 4 | p = 6 | p = 9 |
|---|---|---|---|---|---|
| 1 | (p, 0) | 1 | κ 8.9 → 28/41 | 11.4 → 32/47 | 14.7 → 36/54 |
| 1 | (p, 0) | 2 | 4.3 → 19/28 | 5.2 → 21/31 | 6.3 → 23/34 |
| 1 | (p, 1) | 2 | 1.75 → 10/15 | 2.0 → 11/17 | 2.4 → 13/19 |
| 1 | (p, p/2, 1) | 2 | 1.39 → 8/12 | 1.52 → 9/13 | 1.75 → 10/15 |
| 128000 | (p, 0) | 1 | **2746 → 501/743** | 1602 → 383/567 | 836 → 277/410 |
| 128000 | (p, 0) | 2 | **1011 → 304/451** | 583 → 231/342 | 306 → 167/248 |
| 128000 | (p, p/2, …, 1, 0) | 2 | 604 → 235/348 | 274 → 159/235 | 153 → 119/176 |
| 128000 | (p, 1) | 1 | 7.3 → 25/37 | 10.4 → 30/45 | 13.6 → 35/51 |
| 128000 | (p, 1) | 2 | 4.7 → 20/29 | 5.9 → 22/33 | 7.3 → 25/37 |
| 128000 | (p, p/2, 1) | 2 | 4.2 → 19/27 | 5.1 → 21/30 | 6.0 → 23/33 |

κ(A) itself: 580/765/1039 (τ = 1) vs 1.8e5/1.3e5/9.4e4 (τ = 128000) for p = 4/6/9; κ(D⁻¹A): 500–680 vs 2.2e4–7.6e4. The growth with p of a fixed ladder is mild (κ ×1.4 from p = 4 to 9); the growth with τ of the p→0 ladder is catastrophic.

**(c) τ scan, p = 6, Cheb-2, h = 0.2** (`pmg_tau_scan.py`): κ(BA) for (6,0) = 5.2, 5.3, 5.6, 9.3, 50, 583 at τh = 0.2, 2, 20, 200, 2000, 25600; for (6,1) = 2.0, 2.0, 2.1, 3.0, 5.1, 5.9. The D8-S meshes have τh ≈ 850 (p = 1) … 4000 (p = 9); the model's (6,0) count at τh ≈ 2000 is 67 iterations (1e-8) — the same order as the 75–127 cold iterations measured for `fbmg-*` at p = 4/6.

**(d) Why.** Eigenvalue distribution of BA at p = 6 (`pmg_eig_dist.py`): with (6,0) at τ = 128000 there are exactly **81 eigenvalues ≈ 1.7e-3** (81 = number of interior vertices of the 10×10 mesh), all others ≥ 0.5; with (6,1) the same 81 modes sit in [0.17, 0.3]. The edge-constant trace space cannot represent a continuous piecewise-linear function (one degree of freedom per vertex is missing), and for τh ≫ 1 the HDG trace operator approaches the CG/EDG limit whose low-energy modes are exactly those vertex modes. At τ = 1 these modes are only mildly low-energy (eigenvalues 0.19–0.3), which is why the p→0 design passed all its τ = 1 qualification studies. On a production mesh the count of such modes is the number of interior vertices (82k at p = 9, 300k at p = 4), so PCG cannot deflate them.

Caveats: exact coarse solve (the real coarse correction is one AMGX V-cycle; on the p = 1 operator, 2 unknowns per edge, classical AMG on the scalar expansion should behave like it does on P2-type CG systems — unverified); uniform structured mesh; absolute-tolerance effects not modelled; symmetric cycle assumed.

## 4. Literature (what each source used and what transfers)

* **Cockburn, Dubois, Gopalakrishnan, Tan, "Multigrid for an HDG method", IMA JNA 34(4) 1386–1425 (2014).** Non-nested V-cycle: HDG trace space → continuous P1 on the same mesh → geometric h-MG; convergence independent of h with one smoothing step. Kronbichler–Wall report 55–75 iterations for this rediscretized-P1 variant.
* **Kronbichler & Wall, SISC 40(5) A3423–A3448 (2018), arXiv 1611.03029.** HDG trace (Q_k, hexahedra, k = 1..8, τ = 5·max κ): embedding of P1/Q1 into the trace polynomials with a **Galerkin** coarse operator (Pᵀ A P), then Trilinos ML AMG; ILU(0) smoother on the trace matrix and AMG levels; PCG to 1e-9. Iterations: 16, 15, 17, 17, 20, 20, 21, 21 for k = 1..8 (3D Cartesian), 22–46 on curved variable-coefficient meshes; "optimal multigrid performance in HDG cannot be obtained with point-relaxation smoothers … block relaxation with all dofs on a face or incomplete factorizations are necessary". Transfers: degree-1 coarse space + Galerkin + face-block smoother = the proposed design; expect ~15–25 iterations.
* **Wildey, Muralikrishnan, Bui-Thanh, SISC 41(5) (2019), arXiv 1811.09909.** 2D unstructured triangles, HDG/HMM/WG/IPDG-H; p-coarsening directly **p → 1** on the same mesh (explicitly not P0), then agglomeration h-coarsening with DtN transfers; MG-preconditioned GMRES, p = 1..10. HDG τ = 1/h_min: point-Jacobi 6–12 it., Chebyshev-Jacobi 6–13, LU-SGS 4–7, block-Jacobi 4–9 (p = 7–10: 8–9) — nearly p- and h-independent. HDG τ = 1: point smoothers lose h-scalability, block-Jacobi stays h-scalable for p ≥ 4 (6–9 it.). Transfers directly (2D triangles, trace unknowns, τ dependence).
* **Fabien, Knepley, Mills, Rivière, SISC 41(2) C73–C96 (2019), arXiv 1705.09907.** Quadrilateral HDG, Xeon Phi KNL. Nested p-MG by rediscretization p → p−1 → … → 0, p = 0 HDG projected to continuous P1, then h-MG; FSAI or additive Schwarz smoothers, ν = 2+2: V-cycle convergence factors < 0.22 for p = 2..8 with nnz(M) = nnz(A). Their p = 0 level is only a bridge to continuous P1 (so no P0 bottleneck).
* **Lu, Rupp, Kanschat, "Homogeneous multigrid for HDG", IMA JNA 42(4) 3135–3153 (2022), arXiv 2011.14018; related SISC 2022, IMA JNA 2023, JSC 2023.** Same HDG scheme on all h-levels with a stable injection; Gauss–Seidel smoother; p = 1..3: 33–39 / 9–13 / 24–25 iterations; "we expect rising counts for higher degrees, as we use a point smoother".
* **Di Pietro, Dong, Kanschat, Matalon, Rupp, "Homogeneous multigrid for hybrid discretizations: application to HHO", Numer. Methods PDE (2025), arXiv 2403.15858.** Uniform V-cycle convergence proved for skeletal methods; GS V(1,1)/V(2,2), p = 1..3; 9–21 iterations with the best injection.
* **Di Pietro, Hülsemann, Matalon, Mycek, Rüde, Ruiz, "An h-multigrid method for HHO", SISC 43(5) S839–S861 (2021).** Face unknowns kept at every h-level; same degree on all levels; robustness in polynomial order claimed (tables not retrieved).
* **Botti & Di Pietro, "p-Multilevel preconditioners for HHO discretizations of the Stokes equations with static condensation", Commun. Appl. Math. Comput. (2022), arXiv 2009.13840.** Condensed HHO; coarse operators inherited by L2-orthogonal projections on faces (Galerkin, recursive, matrix-free); ladders 3→2→1 and 6→3→1; smoother = 2 ILU-preconditioned GMRES iterations; FGMRES outer: 9–14 iterations on 2D regular meshes, "mild increase from k = 3 to k = 6".
* **Franciolini, Fidkowski, Crivellini, Computers & Fluids 203, 104542 (2020).** p-MG as FGMRES preconditioner for DG/HDG implicit compressible flow (abstract only).
* **Fortunato, Rycroft, Saye, "Efficient operator-coarsening multigrid schemes for LDG", SISC 41(6) (2019), arXiv 1808.05320.** Coarsening the discrete gradient/divergence gives geometric-MG-equivalent convergence; coarsening the primal (condensed) operator "performs poorly".
* **Phillips, Kerkemeier, Fischer, "Tuning spectral element preconditioners for parallel scalability on GPUs", SIAM PP22 (2022), arXiv 2110.07663; NekRS (Fischer et al., Parallel Computing 114, 2022, arXiv 2104.05829); CEED (Kolev et al., IJHPCA 2021, arXiv 2109.04996).** GPU pMG on hexahedral SEM: schedules (7,3,1), (7,5,3,1), (9,5,1), (9,7,5,1); Chebyshev-accelerated Jacobi / ASM / RAS smoothers of order 1–2; coarse p = 1 solved by one BoomerAMG V-cycle (or AmgX); Chebyshev bounds (0.1, 1.1)·λ̃ with λ̃ from 10 Arnoldi steps — underestimating λ_max is fatal while the λ_min fraction 0.05–0.25 is benign; iteration counts essentially p-independent (p = 3..10); "for the pMG-based methods, higher orders are generally faster" per gridpoint. Transfers: Chebyshev settings (ours: 0.1–1.2·λ̃, 12 power steps — consistent), p = 1 coarse level with one AMG cycle, 1–2 intermediate levels at most.
* **Low-order (refined) preconditioning.** Pazner, Kolev, Dohrmann SISC 45(2) (2023); Pazner, Kolev, Camier IJHPCA (2023, GPU); Pazner, arXiv 2604.00148 (2026): LOR spaces on triangles with degree-independent spectral equivalence constants, AMG (hypre) on the LOR system. An alternative coarse level, not needed here since the p = 1 Galerkin level already captures the bad modes.
* **Other.** Muralikrishnan, Bui-Thanh, Shadid JCP 407 (2020): nested-dissection coarse solver + block-Jacobi two-level for HDG traces (CPU). Henriquez, Lee, Rhebergen arXiv 2503.05918 (2025): parameter-robust preconditioners for reduced hybridized Darcy/Stokes systems. Fu & Kuang (2023/2024): optimal GMG for HDG-P0 via a Crouzeix–Raviart equivalence. No 2024–2026 paper applying GPU matrix-free p-MG to the HDG trace system was found; the closest GPU work is Cui–Kanschat's matrix-free DG/H(div) multigrid on A100s (2024/2025).

Net transfer to a 2D triangular HDG trace at p = 7–9: a two-level (p → 1) or three-level (p → p/2 → 1) modal p-MG with face-block Chebyshev smoothing of order 1–2 and one classical-AMG V-cycle on the degree-1 trace operator; expected 10–25 PCG iterations for 1e-8–1e-12 relative, p-independent to within ±20 %, provided λ_max is not underestimated.

## 5. Quantified expectation

Assumptions: fine BSR data at p = 9 ≈ 1.24 M blocks × 100 × 8 B ≈ 0.96 GB (247,058 free faces, ≤5 blocks per row); one fine-operator read ≈ 1 ms on the RTX PRO 5000 (measured fbmg p = 6 warm iteration 7.9–9.9 ms for 0.94 GB of 7×7 blocks); V-cycle with Chebyshev order m, 1+1 sweeps = 2m fine reads + 1 PCG matvec; coarse p = 1 AMGX cycle ≈ 2× the p = 0 cycle (2.1–2.4 ms per application measured at p = 4–6, 157k triangles); vector work and two host syncs per iteration ≈ 2 ms.

* **Per PCG iteration at p = 9 (native, p→1).** Cheb-1: ≈ 8–10 ms; Cheb-2: ≈ 10–12 ms. AMGX: 30 ms (classical, measured), 12.6 ms (cheb warm, measured), 64 ms (cheb cold, measured, probably inflated by one-time work).
* **Cold first solve at p = 9 (Krylov + solver setup).** Now: cheb 13.3 s (12.1 setup + 1.15), classical 1.1 s (0.22 + 0.90). Native p→1: setup ≈ 0.3–0.5 s + 25–40 iterations × 10 ms ≈ 0.25–0.45 s → **≈ 0.6–0.9 s**, i.e. 12–15 s saved vs cheb, 0.2–0.5 s vs classical, and the same saving on every τ-retry or operator rebuild.
* **Saturated phase with a poor initial guess (per step, p = 9).** If the guess is as bad as cold: classical 28 × 30 ms ≈ 0.85 s; cheb 18 × 12.6 ms ≈ 0.23 s (or up to 1.15 s if the cold per-iteration cost is real); native ≈ 0.25–0.45 s. Gain: 1.4–1.5× vs classical; vs cheb between none and 1.7× depending on the unresolved cheb cold cost.
* **p = 7 (249k triangles, 2.99 M unknowns).** Classical cold 28 × 36 ms = 1.0 s; cheb not measured (expect 18–22 it.; setup ≈ 5–8 s); native ≈ 0.25–0.4 s cold Krylov, setup ≈ 0.3 s. Warm phase: 1–2 iterations once an absolute-tolerance config is used (≈ 25 ms); the p = 7 step then ≈ 410–430 ms.
* **p = 5–6.** Cold/poor-guess Krylov now 0.58–0.89 s (75–127 it. at 5–10 ms); with p→1: ≈ 20–30 it. × 6–10 ms ≈ 0.15–0.3 s. Warm phase unchanged (1–4 it., 8–40 ms).
* **Warm phase at p = 7–9.** No measurable gain over `amgx-cheb-*`: 1–2 iterations either way; RHS assembly (49–88 ms) and reconstruction (14–16 ms) remain.
* **Iteration-count expectation (cold, relative 1e-9–1e-11).** Model (exact coarse): p→1 Cheb-2 25–37 at p = 9, 22–33 at p = 6; with a one-V-cycle AMG coarse correction expect 1.3–1.6× more, i.e. ~30–50, against 18–28 for AMGX on the full-order system and 75–127 for the present p→0 native cycle. The native cycle wins on cost per iteration (3–6×), not on iteration count.
* **Memory (p = 9).** Native: +0.96 GB orthonormal copy + 0.2 GB diagonal inverses + ≈ 0.25 GB vectors + ≈ 0.1 GB p = 1 hierarchy ≈ 1.4–1.5 GB. AMGX cheb: scalar CSR copy (0.96 GB + 0.5 GB int32 indices) + hierarchy ≈ 2.8 GB (peak 31.2 vs 28.4 GB). Both fit on the 48 GB card.
* **Transport.** At p = 9, 84 % of the step; at p = 7, 34 % now and ≈ 75 % once Poisson is fixed. Nothing in this report changes it.

## 6. What it would take (obstacles, design, effort)

Obstacles found in the code (all small):
1. Degree gate `face_hp.py:736-739` (4..6) — kernels support 1..10 (`legendre_face_bsr.py:549, 694, 762`; fused smoother chosen when block ≤ 10, `face_hp.py:505-508`). Also `fast_profiles.py:59-60, 94-95`, `configuration.py:393-398`, `capabilities.py:349-356`, `run_magnetron.py:91-96`, CODEMAP §2b/§7 rows.
2. Terminal level hard-wired to p = 0: `modal_degree_schedule` (`legendre_face_bsr.py:72-88`), assertion `face_hp.py:519-521`, `AmgxScalarVcycle` block-size-1 gate `face_hp.py:241-242`.
3. The coarse AMGX solver must accept the 2×2-block p = 1 operator: either scalarize on device with the existing `_scalarize_device_bsr_matrix` (`hybridge/linalg/gpu/sparse.py:342`) and keep `scalar_p0_amgx_config`, or upload as BSR (`PyAMGXCsrDeviceSolver.setup` already passes `block_dims`, `device_solver.py:206-230`) with `bsr_spmv_backend: cusparse_generic`, the fork's default `classical_bsr_hierarchy: scalar_expand`, and `jacobi_l1_scalar_rows_for_blocks: 1`. The first option is simpler and reuses the tuned scalar config.
4. Policy plumbing: new schedule names (`"direct-to-one"`, `"halve-to-one"`) and policies (`p1-standard` = Cheb-2 1+1, `p1-fast` = Cheb-1 1+1) in `policy.py`; candidates `fbmg-p1-standard/-fast` with `range(2, 10)` in `fast_profiles.py`; option validation in `diffusion_raw_cuda.py:56-66`.
5. Warp utilization: the fused kernels use a 32-lane warp per face with only b lanes active (10 of 32 at p = 9) — correct but ~30 % lane use; acceptable for a first version. A matrix-free application is not needed: the assembled 10×10-block BSR is 0.96 GB and the assembler already emits it.
6. Symmetry/curvature gates at τ = 128000: assembled symmetry defect ≈ 1e-11 relative in the host check; the D8-S `fbmg` runs at τ = 128000 passed with zero fallbacks, so the 1e-10 gate is fine; keep the fallback.
7. The p = 1 coarse level doubles the coarse-cycle cost and the number of AMGX levels; `scalar_p0_amgx_config` (strength 0.40, dense LU 128/256) was tuned for the p = 0 graph — re-tune strength (0.25–0.5) on the p = 1 scalar expansion.

Minimal viable design: ladder p → 1 (optionally p → ⌈p/2⌉ → 1 for p ≥ 8; the model gives only 10–20 % fewer iterations for ~25 % more work per cycle), Chebyshev order 2 (order 1 if timing favours it), 1+1 sweeps, Galerkin principal blocks, one classical AMGX V-cycle on the scalarized p = 1 operator, PCGF outer unchanged. Everything else (`FaceBlockPmgPrototype`, fused smoother, cuSPARSE BSR, true-residual logic, fallback, runner wiring, RHS/reconstruction caches) is reused as is.

Effort: code 1–2 days (≈ 150 lines across `legendre_face_bsr.py`, `face_hp.py`, `policy.py`, `fast_profiles.py`, `configuration.py`, `capabilities.py`, docs/CODEMAP); tests 0.5–1 day (extend `tests/test_legendre_face_bsr.py` and `tests/test_face_hp_mg_residual_restart.py` with p = 1 terminal levels and block sizes 8–10; a host parity test of the principal blocks); device validation (user-run) 0.5 day. Risk: low for correctness (same algebra), medium for performance (coarse p = 1 AMG quality on 0.5–0.75 M scalar unknowns is unmeasured).

## 7. Recommendation (ranked)

| rank | option | expected effect | memory | effort | risk |
|---|---|---|---|---|---|
| 1 | **Fix the AMGX candidate set at p = 7–9 first**: run `amgx-cheb-csr`, `amgx-cheb-bsr` and a classical config with `ABSOLUTE 1e-12` (`diff_rea_gpu4_hdg_pcgf_classical_gs_robust_abs.json`) at p = 7 and 8; also a cheaper cheb hierarchy (strength 0.5, dense LU 128/256) to cut the 12 s setup at p = 9 | p = 7 step 905 → ≈ 420 ms; p = 9 unchanged (810 ms); cold start 15.5 s → ≈ 3 s with a lighter hierarchy | none | hours (config + launcher) | none |
| 2 | **`fb-hp-mg-pcg` with a terminal p = 1 level, gate opened to p = 2..9** (§6) | cold/poor-guess Krylov at p = 7–9: 0.9–1.15 s → ≈ 0.25–0.45 s; solver setup 12 s → 0.3–0.5 s; at p = 5–6: 0.58–0.89 s → ≈ 0.15–0.3 s; warm phase unchanged | +1.4 GB at p = 9 (vs +2.8 GB cheb) | 2–3 days + user-run device validation | medium (coarse p = 1 AMG quality, 10-lane warps) |
| 3 | Two-level variant only (p → 1 with Chebyshev on the fine level, AMGX classical on the p = 1 operator) | same as 2; it *is* option 2 without intermediate levels — the model shows intermediate levels are not needed | same | same | same |
| 4 | Extend the existing p → 0 cycle to p = 7–9 (2-line gate change) | at τ = 128000 predicted 170–450 cold iterations — worse than AMGX | +1.4 GB | hours | high (useless at the campaign's τ) |
| 5 | Nothing beyond option 1 | warm phase already optimal (1–2 it.); cold/poor-guess phase stays at 0.85–1.15 s per step and 12 s setup with cheb | – | – | – |

First tests for option 2, in order: (1) the host check `pmg_small_check.py` extended to the real `FaceBlockPmgPrototype` smoother weights once the p = 1 terminal level exists; (2) a single device solve on a D8-S-sized operator (one SI-BDF2 step at p = 9, 7, 6 on the L3 meshes with `--poisson-candidate fbmg-p1-standard`, `-v 3`); acceptance: cold iterations ≤ 50 at p = 9 and ≤ 35 at p = 6 (τ = 128000), cold Krylov ≤ 0.5 s at p = 9, no fallback, `max_poisson_residual_ratio ≤ 1`; (3) then the campaign comparison across p = 5–9 with `fbmg-p1-standard`, `fbmg-p1-fast`, `amgx-cheb-csr`, `amgx-cheb-bsr`, classical-abs, ranked by the measured saturated-phase step (stage H), not by the first solve's Krylov time, which includes one-time costs for the cheb config.

## Appendix: facts verified vs inferred

Verified in code/data: everything in §1, §2, §3 and the file:line citations. Inferred/estimated: per-iteration and setup costs of the native cycle at p = 7–9 (scaled from measured p = 4–6), the one-time cost inside the cheb cold Krylov at p = 9, the AMG quality on the p = 1 operator, the ×1.3–1.6 factor between the exact-coarse model and a one-V-cycle coarse correction. Literature numbers come from the papers' tables as extracted; Di Pietro et al. 2021 and the Pazner 2026 result tables could not be fetched, so only their abstracts are used.
