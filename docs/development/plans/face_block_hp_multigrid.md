# Face-Block hp-Multigrid For HDG Poisson

Status: implemented for the supported p=4--6 raw-CUDA Legendre-modal
Poisson scope. The maintained solver/cache/fallback contract is documented in
[`../../backends/face_hp_mg_pcg.md`](../../backends/face_hp_mg_pcg.md); this file
retains numerical decisions, measurements, and broader qualification work. The
fine-BSR/scalar-AMGX hierarchy remains the automatic runtime fallback and the
p=1--3/unsupported-configuration default.

The precise ownership boundary between HDGFEM, direct cuSPARSE, PyAMGX, the
modified AMGX hybrid/pure-BSR paths, and the remaining custom smoother kernels
is maintained in the
[BSR and AMGX dependency map](../../backends/bsr_amgx_dependency_map.md).

## Scalar p=0 AMGX parameter tuning: 2026-09-13

Parameter tuning alone improves the native hp-BSR backend on the same **157,280-triangle Euler vortex-gas problem**. The selected configuration changes only `strength_threshold` from 0.25 to **0.40**, `dense_lu_num_rows` from 2048 to **128**, and `dense_lu_max_rows` from 4096 to **256**. These values are now the defaults in `scalar_p0_amgx_config()`. The AMGX binary, wrapper, diagnostics, fixed one-cycle application, symmetric 1+1 L1-Jacobi smoothing, no-aggressive-level policy, p-smoother, and outer PCG implementation are unchanged. No build or new kernel compilation was run.

### Validation results

Each order completed 100 SI-Euler steps with three rotating shadow candidates: the old baseline, a terminal-only 32/64 setting, and the selected strength-0.40/128/256 setting. All receive identical density states and equivalent independently owned warm guesses on a native-driven reference trajectory. The two screening stages used p=6 states 1–16; the main validation table therefore uses **84 held-out states, steps 17–100**, per configuration and order. Times are synchronized complete Poisson calls including RHS assembly, global solve, reconstruction, and wrapper overhead.

| p | Old median (ms) | Tuned median (ms) | Reduction | Old p95 (ms) | Tuned p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 84.55 | 72.72 | 14.0% | 88.22 | 75.75 |
| 5 | 94.40 | 82.60 | 12.5% | 102.18 | 83.03 |
| 6 | 113.90 | 101.21 | 11.1% | 118.20 | 101.53 |

Mean-time reductions are also positive: the mean complete calls are 85.33→72.60, 96.04→82.74, and 114.22→99.71 ms at p=4/5/6. Median and mean differ because iteration counts change along the evolving trajectory.

Mean milliseconds per call; coarse AMGX time is included in global PCG time and must not be added again.

| p | Global PCG old → tuned | Coarse AMGX old → tuned | Outer iterations old → tuned |
| ---: | ---: | ---: | ---: |
| 4 | 75.40 → 62.68 | 41.87 → 29.56 | 20.21 → 19.98 |
| 5 | 82.88 → 69.58 | 44.28 → 31.68 | 20.43 → 20.04 |
| 6 | 97.40 → 82.89 | 50.44 → 36.92 | 21.07 → 20.57 |

RHS assembly and reconstruction retain the same implementation and essentially the same cost. The terminal-only 32/64 setting also wins, but less consistently: median complete-call reductions are 12.0%, 3.4%, and 5.6% at p=4/5/6. Combining the strength change with the smaller terminal solve gives the best validated result across all three orders.

### Why the terminal setting matters

The baseline p=6 scalar hierarchy has row counts **235458 → 69371 → 13874 → 2411**. The selected hierarchy has **235458 → 69371 → 18889 → 4779 → 1129 → 240**. More levels and a modestly denser hierarchy win because the terminal LU application becomes much cheaper. Operator complexity increases from 2.15890 to 2.35681; hierarchy statistics report about 0.0533→0.0590 GB, excluding the complete application footprint.

| p | Old terminal rows / levels | Tuned terminal rows / levels | Old coarse setup (ms) | Tuned coarse setup (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 2419 / 4 | 242 / 6 | 31.64 | 15.16 |
| 5 | 2400 / 4 | 236 / 6 | 28.75 | 11.23 |
| 6 | 2411 / 4 | 240 / 6 | 29.32 | 11.36 |

Setup values are single hierarchy construction measurements, excluding the already-created native fine operator. Thresholds are controls for AMGX hierarchy construction, not a guarantee that the printed terminal matrix has that many rows. For example, the 2048/4096, 1024/2048, and 512/1024 settings all produced the same 2411-row terminal system at p=6; 256/512 and 128/256 at the old strength produced 368 rows. Always inspect actual grid statistics.

### Measured GPU costs by level

The installed AMGX Release build compiles out its internal `levelProfile` timers. Nsight Systems 2025.3.2 successfully captured CUDA/NVTX with CPU sampling disabled, requiring no rebuild. A paired p=6 profile alternated eight baseline and eight tuned one-cycle applications on the same normalized device RHS after warm-up.

Level attribution below is inferred from the fixed V-cycle traversal and verified Jacobi launch dimensions in every captured cycle; it is not a direct AMGX level timer. Descent includes restriction from that level and ascent includes prolongation back to it. Values sum GPU **kernel durations only**. Copies, host work, scheduling gaps, and synchronization are excluded.

| Level / work | Old rows | Old kernel time (μs) | Tuned rows | Tuned kernel time (μs) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 235458 | 103.06 | 235458 | 104.83 |
| 1 | 69371 | 68.26 | 69371 | 68.70 |
| 2 | 13874 | 23.93 | 18889 | 31.10 |
| 3 | 2411 | 756.05 (LU) | 4779 | 26.96 |
| 4 | — | — | 1129 | 26.11 |
| 5 | — | — | 240 | 88.98 (LU) |
| Additional solve monitoring | — | 37.21 | — | 37.14 |
| Total kernels | — | 988.51 | — | 383.82 |

Terminal LU accounts for **76.5%** of baseline kernel time and **23.2%** after tuning; its kernel time falls from **756.05 μs to 88.98 μs**. The extra sparse levels are much cheaper than that saving. The isolated profiled wrapper intervals average 1.364→0.673 ms per correction, but those repeated standalone cycles have a different cache/execution context from corrections interleaved with outer PCG. The production comparisons above use the full unprofiled matched solves and show smaller end-to-end savings. API synchronization durations overlap GPU work and are not additive overhead.

### Screening and correctness

The first screen compared 11 configurations, varying terminal thresholds, interpolation limits 2/3/4, strength 0.15/0.25/0.40, and zero/one aggressive levels. A second 12-configuration screen refined terminal sizes and combined them with strength 0.30/0.40/0.50, interpolation limit 3, or one aggressive level. Baseline and the 256/512 candidate repeat between stages, giving 21 distinct settings overall. Stage scores use states 5–16.

| Representative setting | p=6 screen median complete call (ms) | Interpretation |
| --- | ---: | --- |
| Old baseline, stage 1 | 122.95 | Reference |
| Terminal 256/512 | 113.22 | Smaller LU helps despite extra iterations |
| Terminal 4096/8192 | 168.28 | Fewer iterations, much more expensive terminal work |
| Interpolation limit 2 | 149.19 | Sparser hierarchy loses through extra iterations |
| One aggressive level | 185.96 | Lower complexity does not give a better total solve |
| Old baseline, stage 2 | 123.30 | Independent stage reference |
| Terminal 32/64 | 112.05 | Modest further gain from terminal size alone |
| Strength 0.40, terminal 128/256 | 101.80 | Selected for held-out validation |
| Strength 0.40, terminal 64/128 | 101.83 | Same hierarchy; effectively tied |

All **1268 shadow solves** across screening and validation passed the true modal residual target of 1e-12 with no native fallback. Each candidate hierarchy passed the production sampled symmetry and positive-curvature checks. The selected configuration has symmetry defects below 1.4e-18 across p=4–6 and sampled potential/qx/qy relative L2 differences below 4.2e-12 from the reference. Field parity was checked at states 1, 8, and the final state of each run. These numerical samples support the retained symmetric fixed-cycle construction; they are not a proof over all possible matrices.

The simulation settings match the earlier comparison: FP64, unit disk, signed Gaussian vortex gas with seed 17, SI-Euler, dt=0.01, tau=1, and the RTX PRO 5000 Blackwell. Every attempt stayed within 100 steps and used existing cached kernels. Candidate order rotates. Screening used separate 16-step attempts; validation used 100 steps per order. There are no independent process replicates of the final 100-step runs, so this is a measured default for this supported workload, not a universally optimal AMG configuration.

### Reproduction and adopted settings

Production defaults are centralized in `hdgfem/linalg/face_hp_multigrid.py::scalar_p0_amgx_config`. The tuning driver anchors the old three-parameter baseline explicitly, so future changes to the production default do not silently relabel the baseline. The original driver and helper used for this study are preserved with the evidence.

```sh
CUDA_PATH=/usr/local/cuda-13.0 HDGFEM_PRECISION=float64 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.guiding_center.poisson.tune_poisson_p0_amgx --order 6 --num-steps 100 --configs artifacts/poisson_p0_tuning_20260913/validation_configs.json --output-dir artifacts/poisson_p0_tuning_20260913/validate_p6_new
python3 artifacts/poisson_p0_tuning_20260913/summarize.py
```

Use p=4/5 and fresh output directories for the other orders. The first command is a simulation; the second only reads saved data. A reproduction after adoption uses the current default for the reference trajectory and the explicit old baseline for shadow comparisons; the original executed sources and hashes document the exact recorded run.

Artifacts: [raw measurements, configuration sweeps, Nsight traces, attribution, and verification](../../../artifacts/poisson_p0_tuning_20260913/README.md). The three parameter changes require no AMGX rebuild.

## Matched Euler vortex-gas comparison: 2026-09-13

Native hp-BSR has the lowest repeated Poisson time at p=4, 5, and 6 on the same **157,280-triangle** unit-disk mesh. Median complete-call savings are 11.4–13.1% against coefficient-exact hybrid BSR/CSR and 26.1–41.6% against scalar CSR AMGX. This adds repeated evolving-RHS evidence to the earlier stationary radius-5 studies; their mesh, tolerance, workload, and timing scopes differ, so their absolute timings should not be compared directly.

**Completion and failure:** p=5 and p=6 completed 100 SI-Euler steps each. At p=4, 79 steps have complete matched samples; the hybrid solve at step 80 stopped the run when the independently recomputed nodal residual was `3.803e-13` against `3.800e-13`, after 22 iterations and an AMGX success status. The approximately 0.08% excess is a true-residual acceptance failure, not evidence of divergence. No relaxed acceptance, retry, or missing-step timing is substituted. All main tables use the same **71 states, steps 9–79**, for every order and backend.

### Case, algorithms, and measurement contract

- Case: `euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr`, overridden to p=4/5/6, mesh size 0.0068 with a 150,000-triangle minimum, SI-Euler, `dt=0.01`, at most 100 steps (`T=1` when complete), and Poisson `tau=1`. The signed Gaussian vortex gas uses seed 17, counts (192,96,48,24), widths (0.008,0.016,0.032,0.064), amplitude 4, and center radius 0.96. Plotting is disabled; diagnostics run every 10 steps.
- Hardware/software: NVIDIA RTX PRO 5000 Blackwell, 48,935 MiB, driver 580.126.09, 300 W power limit; FP64; Python 3.12.3, CuPy 14.2.0, modified AMGX 2.5.0 built September 11 with CUDA 13.0. Processes ran sequentially in order p=6,5,4. The compilation guard rejects new Numba, NVRTC, or NVCC compilation; all runs used existing cached kernels. Clocks were not locked.
- Native: `fb-hp-mg-pcg`, Legendre-modal traces normalized internally, direct face BSR, direct p→0 coarsening, fused degree-2 Chebyshev/block-Jacobi smoothing with symmetric 1+1 sweeps, and generic cuSPARSE BSR standalone matvec. The dedicated scalar p=0 AMGX correction applies one fixed classical V-cycle, with L1-Jacobi 1+1, PMIS/AHAT/D2, no aggressive level, and no error scaling. No native fallback occurred in recorded samples.
- Hybrid: nodal (`legacy-lagrange`) traces, PCGF, fine BSR with generic cuSPARSE, and `classical_bsr_hierarchy=scalar_expand`: coefficient-exact scalar CSR hierarchy/transfers/coarse work. This is the established hybrid algorithm, not the experimental all-level dense-BSR hierarchy.
- Scalar AMGX: the same nodal polynomial trace space and classical PCGF configuration, assembled/stored in scalar CSR. AMGX settings: PMIS/AHAT, strength 0.25, D2, V-cycle, degree-2 Chebyshev with L1-Jacobi, one aggressive level, `interp_max_elements=4`, `error_scaling=3`, dense-LU coarse solver (2048/4096 row thresholds). The screen tests the existing 0+2 and 0+3 pre/post-sweep designs. “Best” means fastest median among those established candidates, not an exhaustive search over all AMGX configurations.
- One production native-driven trajectory generates accepted density states at each order. Independent shadow solvers receive identical states, boundaries, and equivalent physical warm guesses. The shared quadratic trace predictor uses independently owned copies of up to three accepted reference traces; the first shadow solve starts at zero. Modal/nodal basis conversion, common-residual checks, and field comparisons are outside the measured interval. Candidate execution order rotates after the first state.
- One fresh operator/hierarchy per candidate is built on state 1. States 2–8 select the lowest median complete-call time within each AMGX format. Losing configurations are closed; states 9 onward measure the three retained methods. Operator, local factors, and hierarchy are reused in every recorded measured call.
- Every timed interval synchronizes the GPU before and after `set_source`, `set_boundary_condition`, and `solve`. Complete-call wall time includes cached RHS assembly, global solver and its normal validation, reconstruction, and wrapper overhead. The extra benchmark parity checks are excluded. These are Poisson timings along a common trajectory; they do not measure three independently evolving complete simulations.
- Common acceptance: unnormalized modal true residual norm ≤1e-12, recomputed with the independent native shadow matrix. Let E evaluate modal basis functions at nodal points. Nodal coefficients satisfy `lambda_nodal = E^T lambda_modal`, and `r_modal = E r_nodal`. AMGX therefore uses absolute tolerance `1e-12 / ||E||_2`, `rtol=0`, and `use_scalar_norm=1`; native uses absolute 1e-12. Nodal targets are approximately 3.800e-13, 3.592e-13, and 3.210e-13 at p=4/5/6. This is a conservative sufficient common bound; final residuals need not be equal. The common check permits 5% rounding slack, but every recorded sample is below the nominal 1e-12 bound. The stricter adapter nodal check remains active and caused the p=4 stop.
- `metadata.json` preserves input configurations. The AMGX adapter normalizes `monitor_residual=1` and `store_res_history=1`; the adjacent effective-config files apply those documented overrides. Native reference trajectories retain the preset `rtol=1e-11, atol=1e-12`; the absolute term dominates the recorded Poisson targets.

### Repeated complete Poisson calls

Times are milliseconds. p95 uses linear interpolation of the sorted 71 observations. These are repeated states in one trajectory, not independent process replicates.

| p | Backend | Median | Mean | p95 | Iterations mean (range) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | Native hp-BSR | 84.59 | 85.80 | 88.12 | 20.37 (20–21) |
| 4 | Hybrid BSR/CSR (0+2) | 95.93 | 96.93 | 99.50 | 23.30 (23–24) |
| 4 | Scalar CSR AMGX (0+3) | 114.42 | 112.86 | 114.75 | 19.68 (19–20) |
| 5 | Native hp-BSR | 94.49 | 97.25 | 102.29 | 20.73 (20–22) |
| 5 | Hybrid BSR/CSR (0+2) | 106.70 | 108.40 | 110.93 | 21.42 (21–23) |
| 5 | Scalar CSR AMGX (0+3) | 154.34 | 153.09 | 154.74 | 17.83 (17–18) |
| 6 | Native hp-BSR | 113.82 | 115.03 | 122.44 | 21.28 (21–23) |
| 6 | Hybrid BSR/CSR (0+2) | 130.94 | 131.75 | 135.66 | 23.17 (23–24) |
| 6 | Scalar CSR AMGX (0+2) | 194.92 | 196.05 | 202.19 | 23.15 (23–24) |

| p | Trace unknowns | Native speedup over hybrid | Native speedup over CSR | Hybrid speedup over CSR |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1,177,290 | 1.134× | 1.353× | 1.193× |
| 5 | 1,412,748 | 1.129× | 1.633× | 1.446× |
| 6 | 1,648,206 | 1.150× | 1.712× | 1.489× |

Over the full available measured window (steps 9–100), p=5 medians are 94.43 / 106.63 / 154.28 ms and p=6 medians are 113.85 / 130.99 / 194.92 ms (native / hybrid / CSR). The longer windows preserve the ranking. Execution-slot median spreads in the common window are at most 0.23%, smaller than the backend differences.

### Where the time goes

Mean milliseconds per call. Iteration is a **subset** of global solve; other is `wall − RHS − global − reconstruction`, calculated per sample before averaging. Do not add iteration a second time. The AMGX iteration field is its device-adapter solve call, not an isolated CUDA-kernel timer.

| p | Backend | RHS | Global | Iteration (included) | Reconstruction | Other |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | Native hp-BSR | 6.663 | 75.891 | 75.891 | 2.209 | 1.036 |
| 4 | Hybrid BSR/CSR (0+2) | 6.662 | 87.179 | 86.046 | 2.221 | 0.867 |
| 4 | Scalar CSR AMGX (0+3) | 6.667 | 103.111 | 101.264 | 2.223 | 0.862 |
| 5 | Native hp-BSR | 9.353 | 84.073 | 84.073 | 2.724 | 1.102 |
| 5 | Hybrid BSR/CSR (0+2) | 9.353 | 95.381 | 93.957 | 2.743 | 0.921 |
| 5 | Scalar CSR AMGX (0+3) | 9.353 | 140.080 | 137.765 | 2.739 | 0.920 |
| 6 | Native hp-BSR | 12.247 | 98.230 | 98.230 | 3.404 | 1.147 |
| 6 | Hybrid BSR/CSR (0+2) | 12.253 | 115.113 | 113.327 | 3.422 | 0.963 |
| 6 | Scalar CSR AMGX (0+2) | 12.253 | 179.407 | 176.538 | 3.417 | 0.976 |

The global phase explains the performance ranking; all backends have nearly identical local RHS and reconstruction costs. Normal AMGX validation contributes 0.69/0.88/1.11 ms in hybrid and 1.41/1.77/2.20 ms in CSR at p=4/5/6, included in global time. `cached_rhs.local_solve` aliases `cached_rhs.fused_solve_flux_scatter` and must not be added separately. The existing `solver_headline_unaccounted` cached-RHS metric also aliases the whole headline because it looks up a different timing key; it is not additional work.

Native coarse AMGX time is measured by differences of the coarse adapter counters around each complete call. It includes its launch/adapter/synchronization cost and is a subset of native iteration/global time:

| p | Coarse time/call (ms) | Share of native global | Mean time/coarse application (ms) |
| ---: | ---: | ---: | ---: |
| 4 | 42.15 | 55.5% | 2.070 |
| 5 | 44.94 | 53.5% | 2.168 |
| 6 | 50.86 | 51.8% | 2.390 |

Reducing coarse-correction overhead remains a substantial native optimization target. These counters do not separate p=0 GPU arithmetic from PyAMGX dispatch; no new coarse-kernel-only claim follows from them.

### Initial call and storage

These are single initial shadow calls with zero guesses, after the reference trajectory has already initialized the GPU. They are not cold-process or replicated setup benchmarks. Times are milliseconds; hierarchy/setup and iteration are included in global solve, which is included in complete-call time. Assembly includes local factor construction.

| p | Backend | Complete initial call | Assembly | Hierarchy/setup | Iteration |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | Native hp-BSR | 353.52 | 178.98 | 49.74 | 121.51 |
| 4 | Hybrid BSR/CSR (0+2) | 857.22 | 680.64 | 34.01 | 137.16 |
| 4 | Scalar CSR AMGX (0+3) | 440.22 | 178.65 | 96.24 | 159.39 |
| 5 | Native hp-BSR | 512.50 | 307.74 | 63.79 | 136.23 |
| 5 | Hybrid BSR/CSR (0+2) | 1052.34 | 844.72 | 47.52 | 153.81 |
| 5 | Scalar CSR AMGX (0+3) | 950.43 | 306.34 | 420.92 | 216.20 |
| 6 | Native hp-BSR | 694.53 | 478.57 | 56.13 | 155.06 |
| 6 | Hybrid BSR/CSR (0+2) | 735.78 | 478.33 | 57.81 | 182.19 |
| 6 | Scalar CSR AMGX (0+2) | 1168.98 | 478.91 | 392.64 | 288.62 |

The first nodal hybrid assembly at p=4 and p=5 carries a large first-use cost: 680.64 and 844.72 ms. The later hybrid 0+3 candidate assembles in 178.22 and 305.51 ms respectively. This ordering sensitivity prevents interpreting the cold 0+2/0+3 difference as an algorithmic setup advantage. The p=6 first nodal assembly is 478.33 ms. All steady measurements exclude these first calls.

| p | Fine BSR storage, native/hybrid (MiB) | Fine CSR storage (MiB) | Shared-size compact local factors per solver (MiB) |
| ---: | ---: | ---: | ---: |
| 4 | 229.58 | 340.79 | 559.18 |
| 5 | 328.23 | 489.66 | 1009.17 |
| 6 | 444.81 | 665.43 | 1681.15 |

These are array sizes from the assembly cache, not peak GPU memory or complete solver footprints. BSR index storage is 5.38 MiB at every order; scalar CSR indices use 116.59 / 166.81 / 226.00 MiB. Native additionally owns its normalized matrix and multigrid workspaces; hybrid owns its scalar hierarchy. The local-factor column reports equal per-solver sizes, not shared allocations.

### AMGX configuration screen

States 2–8, median complete-call milliseconds; native is not retuned. All candidates keep the same absolute residual contract.

| p | Hybrid 0+2 | Hybrid 0+3 | CSR 0+2 | CSR 0+3 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 103.182 | 108.131 | 120.148 | 119.721 |
| 5 | 115.171 | 118.240 | 167.878 | 162.106 |
| 6 | 140.539 | 147.267 | 202.058 | 209.460 |

Hybrid 0+2 wins at all orders. CSR 0+3 wins at p=4/5 and 0+2 at p=6. The p=4 CSR difference is only 0.36%; treat this as a near tie given seven screening samples. These results retain the nodal Chebyshev/L1 baseline established in [the diffusion AMGX study](../../research/solver_studies/diffusion_amgx_2026_07.md).

### Correctness, limitations, and reproducibility

All 885 saved shadow results pass the common residual check. The maximum common modal residual is 9.9964e-13; the largest sampled potential/qx/qy relative L2 difference from the reference is 2.9124e-11. Physical field checks are recorded at states 1, 8, 50, and 100 when reached; p=4 has no state-100 check. Mesh node and connectivity SHA-256 hashes agree across all orders. No native fallback or measured hierarchy rebuild appears in saved samples.

The p=4 step-80 hybrid rejection is retained in `p4_matched/failure.json`; the failing result was rejected before the additional common-modal check, so there is no independently recorded common-modal residual for that failed call. A future robustness change could separate the internal iterative stopping threshold from the external true-residual acceptance threshold or refine after a borderline failure. This study does not change production solver behavior or claim p=4 completed 100 steps.

An earlier p=6 state-1 qualification attempt (`p6/`) is excluded: the harness originally borrowed a warm-start workspace and omitted aggregate BSR residual normalization. The matched runs use owned accepted history and `use_scalar_norm=1`; the discarded attempt is preserved separately. The 100-step bound is per trajectory attempt; no attempt exceeded it.

For context, the native-driven reference trajectory has median linear-step wall times 164.88 / 218.28 / 306.68 ms at p=4/5/6 over states 9–79. Its transport stage medians are 55.92 / 96.04 / 161.06 ms and Poisson stage medians 109.63 / 122.07 / 145.37 ms. Those reference solves have their own warm-start history and are not the matched shadow samples. The benchmark process elapsed times include extra candidate solves and checks and must not be interpreted as production throughput. This short study establishes solver timing and algebraic parity, not long-time turbulence fidelity or temporal convergence.

Artifacts: [`artifacts/poisson_backend_comparison_20260913`](../../../artifacts/poisson_backend_comparison_20260913/README.md), including raw samples, per-order configurations and completion/failure records, console logs, source/library hashes, `summary.json`, `summary.csv`, and an offline coverage/aggregation script. The driver reuses production runner, solver, predictor, basis, field-norm, and compilation-guard helpers:

```sh
CUDA_PATH=/usr/local/cuda-13.0 HDGFEM_PRECISION=float64 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.guiding_center.poisson.benchmark_poisson_backends --order 6 --num-steps 100 --mesh-size 0.0068 --dt 0.01 --poisson-tau 1 --output-dir artifacts/poisson_backend_comparison_20260913/p6_new
python3 artifacts/poisson_backend_comparison_20260913/summarize.py
```

Use order 4 or 5 and a fresh output directory for the other cases; the driver refuses to overwrite samples. No build is required to analyse the saved data. Simulation reruns require the existing kernel caches, and fail instead of compiling a missing kernel.

## Decision

The candidate is a fixed, symmetric face-block p/h multigrid V-cycle used as an
SPD preconditioner for PCG:

1. assemble the condensed Poisson trace operator directly in face BSR;
2. express every face in a canonical, nested, L2-orthonormal Legendre basis;
3. p-coarsen on the unchanged face graph to the constant mode;
4. apply classical scalar h-AMG only to that p=0 operator;
5. use face-block Chebyshev--block-Jacobi smoothing on p>0 levels;
6. use transpose restriction, reversed post-smoothing, and a fixed coarse
   application so the completed V-cycle is suitable for ordinary PCG.

The working name is `FB-HP-MG-PCG`. The first prototype may use PCGF while
symmetry and positive-definiteness are being measured; PCG is enabled only
after those properties pass explicit tests.

## Existing Evidence And Baseline

The coefficient-exact hybrid AMGX path keeps the fine operator in BSR and uses
a scalar hierarchy. On the 152,909-triangle radius-5 trigonometric-Poisson disk
at p=1..6 it needs 20--23 iterations and is the primary performance baseline.
The dense pure-BSR classical hierarchy needs 15--53 iterations, but is
3.05--4.45 times slower for p=3..6. The missing ingredient is therefore coarse
space quality and work distribution, not another scalar weight on the existing
whole-face interpolation.

The following have already been tested and are not the next experiment:

- identity-lifted `w_ic I_b` interpolation;
- dense block-Jacobi-smoothed interpolation on a fixed D2 support;
- larger support, repeated smoothing, right normalization, and weight damping;
- raw, normalized, and inverse-diagonal block strength metrics;
- scalar-guided whole-face promotion and block Extended+i interpolation.

The p-coarsening hypothesis has now passed its first production-size numerical
checkpoint. On the 150,209-triangle radius-5 disk at p=6, the dedicated scalar
p=0 cycle described below makes the complete modal V-cycle self-adjoint to
about 3e-18 and positive on the sampled action. The remaining work is the full
Phase-2 parameter ablation, persistent-buffer/launch optimization,
repeated-RHS timing, and qualification through p=9.

### Production-size Phase-2 checkpoint (2026-08-19)

The system had 224,862 free faces, 1,574,034 degree-6 trace unknowns, and
1,122,504 face blocks. Generic cuSPARSE BSR SpMV took 0.962 ms versus 1.019 ms
for the row-owned raw kernel, with relative parity 2.19e-16. Results below use
order-3 face-block Chebyshev smoothing, one pre- and one reversed post-sweep,
and the common true-residual tolerance 1e-8.

#### Retained GPU kernel roles

All four implementations remain in the prototype. They form a validation ladder
from a readable array formulation to the low-level production candidate.

- **Transparent CuPy reference smoother (`cupy`).** This intentionally
  materializes the residual, calls the selected BSR `matvec`, applies every
  full dense inverse diagonal block with `cupy.matmul`, and performs a
  separate vector update. With `spmv_backend=auto`, the matrix action is
  generic cuSPARSE BSR where supported. This is the simple gateway and numerical
  oracle for future dense-BSR work and is not scheduled for removal.
- **Generic cuSPARSE BSR SpMV (`cusparse-generic-bsr`).** This remains the
  default standalone action for PCG `A*p`, true-residual recomputation, and
  the CuPy reference smoother. Descriptors, preprocessing state, and workspace
  are matrix-owned and reused.
- **Row-owned raw-CUDA BSR SpMV (`raw-cuda`).** This is the race-free fallback
  and independent benchmark/parity control. It retains full dense face blocks
  and remains available when generic cuSPARSE rejects a block size or runtime.
- **Warp-owned fused dense-BSR smoother (`fused-raw-cuda`).** One warp owns
  one face row, loads each neighboring trace block once, broadcasts its entries
  with warp shuffles, and fuses dense BSR SpMV, residual formation, the complete
  dense diagonal-block inverse, and the Chebyshev update without global
  residual/update temporaries.

| p schedule | smoother | outer | iterations | V-cycle | hot solve | true relative residual |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 6 -> 3 -> 1 -> 0 | CuPy reference | PCG | 20 | 45.52 ms | 0.958 s | 4.48e-9 |
| 6 -> 0 | CuPy reference | PCG | 22 | 26.19 ms | 0.625 s | 4.27e-9 |
| 6 -> 0 | fused dense BSR | PCG | 22 | 10.67 ms | 0.280 s | 4.27e-9 |
| 6 -> 3 -> 1 -> 0 | CuPy reference | PCGF | 20 | 45.86 ms | 0.959 s | 4.48e-9 |

The two-run direct-schedule medians confirm the earlier single-run result:
fusion reduces the p=6 V-cycle by 59.3% and the hot solve by 55.3%, with unchanged
iterations and true residual. Generic cuSPARSE BSR remains the default for
standalone operator applications such as PCG `A*p`; the fused kernel is the
smoother specialization.

#### Direct p-to-zero degree/mesh sweep

The calibrated radius-5 unstructured disks contain 99,896, 124,831, and
150,209 triangles. Every mesh/degree/backend combination was run twice: 72
successful solves in total. Each reported V-cycle is itself the median of 10
warmed CUDA-event samples. Both paths use direct `p -> 0` coarsening,
order-3 Chebyshev, symmetric `1+1` smoothing, the dedicated fixed scalar-p=0
AMGX cycle, FP64 PCG, and true-residual tolerance 1e-8. CuPy and fused pairs
have identical iteration counts; all residuals pass and symmetry defects remain
at roundoff scale.

The displayed solve time is the hot outer Krylov phase only. For the custom
path it begins immediately before the first preconditioner application and ends
after final FP64 true-residual synchronization. It excludes raw-CUDA assembly,
the modal transform, p-level construction, spectral estimation, diagonal-block
inversion, the scalar-AMGX hierarchy setup, and the separate diagnostic V-cycle
samples. The historical `amgx_solve_seconds` values likewise exclude
`amgx_setup_seconds`, so the comparison below is hot-solve versus hot-solve,
not setup-inclusive or end-to-end time.

| triangles | p | it | CuPy V-cycle (ms) | fused V-cycle (ms) | V speedup | CuPy hot PCG (s) | fused hot PCG (s) | hot speedup | residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99,896 | 1 | 13 | 8.287 | 6.043 | 1.37x | 0.1193 | 0.0900 | 1.33x | 7.45e-09 |
| 99,896 | 2 | 15 | 11.606 | 6.687 | 1.74x | 0.1880 | 0.1142 | 1.65x | 4.75e-09 |
| 99,896 | 3 | 16 | 11.811 | 7.133 | 1.66x | 0.2049 | 0.1292 | 1.59x | 6.38e-09 |
| 99,896 | 4 | 17 | 15.994 | 7.925 | 2.02x | 0.2930 | 0.1558 | 1.88x | 6.74e-09 |
| 99,896 | 5 | 17 | 16.707 | 8.675 | 1.93x | 0.3072 | 0.1703 | 1.80x | 6.78e-09 |
| 99,896 | 6 | 18 | 19.974 | 9.441 | 2.12x | 0.3869 | 0.1977 | 1.96x | 5.87e-09 |
| 124,831 | 1 | 14 | 10.666 | 7.838 | 1.36x | 0.1609 | 0.1217 | 1.32x | 2.65e-09 |
| 124,831 | 2 | 15 | 14.773 | 8.493 | 1.74x | 0.2373 | 0.1436 | 1.65x | 5.74e-09 |
| 124,831 | 3 | 16 | 15.011 | 9.073 | 1.65x | 0.2563 | 0.1617 | 1.59x | 7.44e-09 |
| 124,831 | 4 | 17 | 20.302 | 10.120 | 2.01x | 0.3715 | 0.1964 | 1.89x | 5.86e-09 |
| 124,831 | 5 | 17 | 21.120 | 10.964 | 1.93x | 0.3867 | 0.2151 | 1.80x | 7.76e-09 |
| 124,831 | 6 | 18 | 25.175 | 12.017 | 2.09x | 0.4866 | 0.2510 | 1.94x | 6.38e-09 |
| 150,209 | 1 | 18 | 10.377 | 6.403 | 1.62x | 0.1808 | 0.1245 | 1.45x | 5.11e-09 |
| 150,209 | 2 | 19 | 15.679 | 6.971 | 2.25x | 0.2832 | 0.1498 | 1.89x | 7.89e-09 |
| 150,209 | 3 | 21 | 14.604 | 7.544 | 1.94x | 0.3198 | 0.1779 | 1.80x | 6.07e-09 |
| 150,209 | 4 | 21 | 20.422 | 8.185 | 2.50x | 0.4855 | 0.2058 | 2.36x | 8.20e-09 |
| 150,209 | 5 | 22 | 21.282 | 9.284 | 2.29x | 0.5087 | 0.2456 | 2.07x | 6.91e-09 |
| 150,209 | 6 | 22 | 26.188 | 10.671 | 2.45x | 0.6250 | 0.2797 | 2.23x | 4.27e-09 |

Fusion wins against the transparent CuPy reference in all 18 cases. V-cycle
speedups are 1.36--2.50x and hot-solve speedups are 1.32--2.36x. The gain
generally grows with block size and mesh size, consistent with eliminating
intermediate global vectors and reusing each dense neighboring face vector.

#### Comparison with the previous CSR and hybrid kernels

The table below joins the new hot-solve medians with measured-repeat medians
from `docs/research/solver_studies/classical_amg_bsr_sweep_samples_2026_08.csv`.
It also distinguishes BSR block nonzeros, `nnzb`, from scalar-expanded
nonzeros: `nnz = nnzb * (p+1)^2`. Trace DOFs count only free trace faces after
Dirichlet elimination. Historical paths use PCGF and their recorded
classical-AMG policy; the custom path uses the symmetric PCG/V-cycle above.
These are same-matrix, same-hardware, same-tolerance historical hot-solve
comparisons, not yet interleaved identical-runner timings.

| triangles | p | trace DOFs | BSR nnzb | scalar nnz | CuPy hot PCG (s) | fused hot PCG (s) | historical CSR hot PCGF (s) | historical hybrid hot PCGF (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99,896 | 1 | 298,952 | 745,908 | 2,983,632 | 0.1193 | 0.0900 | 0.0616 | 0.0576 |
| 99,896 | 2 | 448,428 | 745,908 | 6,713,172 | 0.1880 | 0.1142 | 0.0877 | 0.0550 |
| 99,896 | 3 | 597,904 | 745,908 | 11,934,528 | 0.2049 | 0.1292 | 0.1097 | 0.0831 |
| 99,896 | 4 | 747,380 | 745,908 | 18,647,700 | 0.2930 | 0.1558 | 0.1474 | 0.1466 |
| 99,896 | 5 | 896,856 | 745,908 | 26,852,688 | 0.3072 | 0.1703 | 0.1871 | 0.1521 |
| 99,896 | 6 | 1,046,332 | 745,908 | 36,549,492 | 0.3869 | 0.1977 | 0.2544 | 0.1843 |
| 124,831 | 1 | 373,670 | 932,529 | 3,730,116 | 0.1609 | 0.1217 | 0.0772 | 0.0451 |
| 124,831 | 2 | 560,505 | 932,529 | 8,392,761 | 0.2373 | 0.1436 | 0.1113 | 0.0688 |
| 124,831 | 3 | 747,340 | 932,529 | 14,920,464 | 0.2563 | 0.1617 | 0.1350 | 0.1011 |
| 124,831 | 4 | 934,175 | 932,529 | 23,313,225 | 0.3715 | 0.1964 | 0.1823 | 0.1821 |
| 124,831 | 5 | 1,121,010 | 932,529 | 33,571,044 | 0.3867 | 0.2151 | 0.2344 | 0.1896 |
| 124,831 | 6 | 1,307,845 | 932,529 | 45,693,921 | 0.4866 | 0.2510 | 0.3173 | 0.2290 |
| 150,209 | 1 | 449,724 | 1,122,504 | 4,490,016 | 0.1808 | 0.1245 | 0.0893 | 0.0542 |
| 150,209 | 2 | 674,586 | 1,122,504 | 10,102,536 | 0.2832 | 0.1498 | 0.1414 | 0.0815 |
| 150,209 | 3 | 899,448 | 1,122,504 | 17,960,064 | 0.3198 | 0.1779 | 0.1608 | 0.1216 |
| 150,209 | 4 | 1,124,310 | 1,122,504 | 28,062,600 | 0.4855 | 0.2058 | 0.2183 | 0.2179 |
| 150,209 | 5 | 1,349,172 | 1,122,504 | 40,410,144 | 0.5087 | 0.2456 | 0.2815 | 0.2274 |
| 150,209 | 6 | 1,574,034 | 1,122,504 | 55,002,696 | 0.6250 | 0.2797 | 0.3806 | 0.2739 |

#### Solver setup plus hot solve (assembly excluded)

For completeness, the following adds the one-time solver/preconditioner setup
to the hot Krylov time. Custom setup is `prototype_setup_seconds`: normalized
p-level extraction, dense face-diagonal inversion, spectral estimation, and
the scalar-p=0 AMGX hierarchy setup. Historical setup is
`amgx_setup_seconds`. PDE assembly and reconstruction remain excluded.
Every hot PCG/PCGF value already contains all preconditioner applications made
during Krylov iteration. Runtime/NVRTC compilation and the separately timed
diagnostic V-cycle samples are not charged to this solver total.

| triangles | p | CuPy setup + hot PCG (s) | fused setup + hot PCG (s) | historical CSR setup + hot PCGF (s) | historical hybrid setup + hot PCGF (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 99,896 | 1 | 1.3172 | 1.2902 | 0.2244 | 0.2224 |
| 99,896 | 2 | 1.3859 | 1.3156 | 0.4334 | 0.0914 |
| 99,896 | 3 | 1.4044 | 1.3494 | 0.1806 | 0.1282 |
| 99,896 | 4 | 1.4650 | 1.3352 | 0.2289 | 0.2109 |
| 99,896 | 5 | 1.4905 | 1.3398 | 0.4389 | 0.2430 |
| 99,896 | 6 | 1.5574 | 1.3779 | 0.5042 | 0.3004 |
| 124,831 | 1 | 2.2216 | 2.1876 | 0.3524 | 0.0741 |
| 124,831 | 2 | 2.3134 | 2.2359 | 0.7356 | 0.1171 |
| 124,831 | 3 | 2.3320 | 2.2337 | 0.2406 | 0.1536 |
| 124,831 | 4 | 2.4030 | 2.2256 | 0.3019 | 0.2623 |
| 124,831 | 5 | 2.4294 | 2.2591 | 0.6634 | 0.3021 |
| 124,831 | 6 | 2.5451 | 2.3071 | 0.7291 | 0.3764 |
| 150,209 | 1 | 0.4748 | 0.4179 | 0.5448 | 0.0883 |
| 150,209 | 2 | 0.5910 | 0.4502 | 1.1853 | 0.1393 |
| 150,209 | 3 | 0.6225 | 0.4702 | 0.3106 | 0.1839 |
| 150,209 | 4 | 0.7724 | 0.4980 | 0.3870 | 0.3118 |
| 150,209 | 5 | 0.7896 | 0.5363 | 0.9822 | 0.3631 |
| 150,209 | 6 | 0.9292 | 0.5954 | 1.0425 | 0.4512 |

These setup-inclusive measurements change the conclusion for one-shot solves:
the established hybrid path wins all 18 cases. The custom setup also shows
strong nonmonotonic mesh dependence in these samples, so setup optimization and
an interleaved cold/warm benchmark are required before drawing scaling claims.
The hot table remains the relevant amortized comparison when a fixed Poisson
operator and hierarchy are reused for many right-hand sides.

#### Chebyshev screen and zero-start fast path

A focused 150,209-triangle p=4--6 screen selected direct `p -> 0`, order-2
Chebyshev, symmetric `1+1` smoothing, and ordinary PCG. Relative to order 3,
order 2 needs one or two extra iterations but removes two dense-BSR smoother
stages per V-cycle and wins in hot time. Diagnostic `0+3` smoothing has
symmetry defects between 2e-6 and 2e-5, requires PCGF, and is slower. The
halving schedules reduce the iteration count to 19--20 but nearly double the
V-cycle cost. Block-L1 was not pursued because full face-block Jacobi remains
SPD and satisfies the iteration gate.

The first Phase-3 optimization exploits the provably zero correction at the
start of every V-cycle. Its first pre-smoothing stage now evaluates
`x = omega * D_face^{-1} rhs` with a dedicated warp kernel, without reading the
BSR operator or forming `A * 0`; it also avoids zero-filling that initial output
buffer. Two measurements per degree, with one p=5 tie-breaker for host jitter,
give:

| p | iterations | Cheb-2 before (s) | zero-start V-cycle (ms) | zero-start hot PCG (s) | historical hybrid (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 23 | 0.1997 | 5.407 | 0.1582 | 0.2179 |
| 5 | 24 | 0.2071 | 5.965 | 0.1847 | 0.2274 |
| 6 | 23 | 0.2257 | 6.655 | 0.2007 | 0.2739 |

True relative residuals are unchanged at 4.42e-9, 2.31e-9, and 5.56e-9. The
V-cycle improves by 23% at p=4 and about 14% at p=5,6; hot solve improves by
21%, 11%, and 11% against the pre-optimization Cheb-2 samples. Against the
historical hybrid hot medians, the optimized path is 27%, 19%, and 27% faster.
These are promising but not yet the required identical-runner interleaved
comparison.

Persistent per-level correction/scratch/residual/coarse-RHS storage then
removes solve-time V-cycle vector allocation, performs `rhs-A*x` in place,
restricts directly into the retained low-mode buffer, and adds the coarse
correction directly into the low Legendre coefficients without materializing a
full prolongation vector. The returned correction is borrowed workspace storage;
the prototype is deliberately serial and rejects reentrant application.

| triangles | p | trace DOFs | BSR nnzb | scalar nnz | workspace MiB | zero-start hot PCG (s) | persistent hot PCG (s) | gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150,209 | 4 | 1,124,310 | 1,122,504 | 28,062,600 | 27.449 | 0.1582 | 0.1562 | 1.3% |
| 150,209 | 5 | 1,349,172 | 1,122,504 | 40,410,144 | 32.596 | 0.1847 | 0.1828 | 1.0% |
| 150,209 | 6 | 1,574,034 | 1,122,504 | 55,002,696 | 37.742 | 0.2007 | 0.1977 | 1.5% |

The corresponding V-cycle medians are 5.311, 5.880, and 6.526 ms, improvements
of 1.8%, 1.4%, and 1.9%. Iterations and true residuals remain exactly unchanged.
The modest gain is consistent with CuPy's caching allocator already making raw
allocation inexpensive; the persistent storage is still required groundwork
for transfer fusion and CUDA-graph capture.

A subsequent directly restricted residual experiment computed only the retained
modal rows of `rhs-A*x`. The first one-face-per-warp version was slower. A
power-of-two subwarp revision processed four p=4--6 faces per warp, but still did
not beat the full generic-cuSPARSE BSR residual consistently:

| triangles | p | trace DOFs | BSR nnzb | scalar nnz | cuSPARSE full V-cycle (ms) | custom restricted V-cycle (ms) | cuSPARSE hot PCG (s) | custom hot PCG (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150,209 | 4 | 1,124,310 | 1,122,504 | 28,062,600 | 5.311 | 5.534 | 0.1562 | 0.1617 |
| 150,209 | 5 | 1,349,172 | 1,122,504 | 40,410,144 | 5.880 | 5.688 | 0.1828 | 0.2019 |
| 150,209 | 6 | 1,574,034 | 1,122,504 | 55,002,696 | 6.526 | 6.430 | 0.1977 | 0.1990 |

The p=5 hot custom sample contains visible host/synchronization jitter, but even
the CUDA-event V-cycle results show only small mixed changes (-4.2%, +3.3%,
+1.5%). Reducing coefficient rows did not overcome the custom kernel's lower
utilization relative to cuSPARSE. The custom kernel and configuration surface
were therefore removed; all ordinary and residual BSR SpMV now rely on the
cached generic-cuSPARSE implementation. The restored-default confirmation gives
0.1555, 0.1818, and 0.1976 s with unchanged iterations and residuals.

The degree-dependent conclusion is now:

- At p=1--3, keep the established hybrid path until the same optimization is
  measured there; fixed V-cycle and coarse-wrapper costs dominate small blocks.
- At p=4--6, direct-to-zero Cheb-2 block-Jacobi is the selected custom policy.
- The directly restricted residual was tested and rejected; generic cuSPARSE
  remains the only BSR SpMV implementation. Keep the fixed-five-slot structure
  as assembly/topology metadata, not as another SpMV kernel.
- The next hot-path targets are persistent cuSPARSE descriptor/preprocess reuse,
  fused PCG vector updates and reductions, CUDA-graph capture where the solver
  stack permits it, and an identical-runner interleaved comparison.
- Production promotion still requires setup reuse/optimization and an
  end-to-end win under the common solver contract.

The dedicated p=0 AMGX cycle is scalar classical PMIS/D2 with one
`JACOBI_L1` pre/post sweep, no aggressive level, no correction scaling, and
fixed one-cycle work. It must not inherit the full-order nodal Chebyshev preset.
The inherited preset produced a 9.61e-5 symmetry defect; AMGX-matching PCGF
still converged, but required 30 iterations and 1.477 s. This confirms the known
result that the nodal-tuned PCGF/AMG configuration should not be judged or used
as a full-order modal Legendre baseline. The fair incumbent remains full AMGX
on nodal trace coordinates.

## Discrete Representation

For degree p, one face has b=p+1 unknowns and the trace operator is

```text
A_p = [A_fg],       A_fg in R^(b x b),       at most five blocks per row
```

for a two-dimensional manifold triangular mesh. Full row storage is retained
for race-free row-owned SpMV.

The assembly basis is the current Legendre basis `P_j`. The normalized solver
basis is

```text
psi_j = sqrt((2*j + 1)/2) * P_j.
```

With `S=diag(sqrt((2*j+1)/2))`, transform once as

```text
A_modal = S A S,       g_modal = S g,       a_assembly = S a_modal.
```

Face reversal acts diagonally by `(-1)^j`; the existing modal raw-CUDA assembly
already implements this orientation convention. For a coarse degree pc, nested
injection retains modes 0..pc. Hence the Galerkin block is the leading
`(pc+1) x (pc+1)` principal subblock. This is a Galerkin trace level, not a
freshly condensed lower-degree HDG discretization.

Default degree schedule:

```text
p -> floor(p/2) -> ... -> 1 -> 0
```

The prototype must also retain `p -> 0` as an ablation.

## V-Cycle

On every p>0 level, use the SPD face diagonal `M=blockdiag(A_ff)` and a fixed
Chebyshev polynomial in `M^-1 A`. Estimate the upper spectral bound once per
level, inflate it for safety, and use a fixed lower fraction. Start with orders
2 and 3. Apply one pre-smoothing polynomial and the reversed adjoint sequence
after coarse correction. Restriction truncates high modal coefficients;
prolongation injects low coefficients.

At p=0, apply one reusable classical AMGX V-cycle on the scalar face graph. The
initial numerical prototype may synchronize and allocate in its Python wrapper;
those costs must be reported separately and must not be interpreted as the
production hot-solve time.

The outer convergence contract remains the repository contract:

```text
||b - A x|| <= max(atol, rtol * ||b||).
```

Also report `||r_k||/||r_0||`, but do not substitute it for the contract above.
Recompute the true FP64 residual periodically and before accepting a solve.

## GPU Operator Policy

Use direct face BSR throughout p-level work. For SpMV:

1. prefer CUDA 13 generic BSR `cusparseSpMV` when the runtime supports it;
2. cache matrix/dense-vector descriptors, preprocessing state, and workspace
   for the lifetime of the matrix structure;
3. retain a row-owned specialized raw-CUDA kernel as a correctness fallback and
   benchmark control;
4. specialize block sizes 1..10, prioritizing b=2..7 now and b=8..10 for
   Poisson p=7..9;
5. invalidate cached state on structure, pointer, value-type, block order,
   device, or incompatible-stream changes.

The installed CuPy wheel currently exposes generic CSR but not a public BSR
matrix class or `cusparseCreateBsr`, so the HDG prototype uses a narrow local C
binding to the loaded cuSPARSE library. It must fall back cleanly when the
symbol or runtime support is absent; changing or rebuilding CuPy is not required
for this phase.

## Phased Roadmap

### Phase 0 — freeze baselines

- Record scalar CSR, coefficient-exact hybrid fine-BSR/scalar-hierarchy, and
  dense pure-BSR results with CUDA-event timings and independent residuals.
- Treat the hybrid path as the primary target and scalar CSR as the portability
  reference.

### Phase 1 — validate modal face BSR

- Verify direct BSR/CSR matvec parity, bilinear symmetry, positive Rayleigh
  quotients, diagonal location, five-block topology, and orientation changes.
- Verify normalized-basis round trips and transformed operator equivalence.

### Phase 2 — inexpensive numerical prototype

- Build p-levels by principal modal-block extraction on CuPy arrays.
- Compare halving and direct-to-zero schedules.
- Compare order-2/order-3 Chebyshev, `1+1` and diagnostic `0+3` smoothing, and
  block Jacobi versus block-L1 only if Jacobi fails.
- Reuse one dedicated scalar AMGX hierarchy at p=0; do not inherit the
  full-order nodal smoother/coarsening preset.
- Measure V-cycle symmetry and positive curvature before selecting PCG; retain
  the AMGX Polak--Ribiere PCGF recurrence as the diagnostic fallback.
- Compare iteration counts with the fine-BSR/scalar-AMG baseline before any
  production integration.

### Phase 3 — optimize p-level primitives

- Cache generic-cuSPARSE BSR descriptors/preprocessing/workspace.
- Benchmark against specialized row-owned kernels for b=1..10.
- [x] Fuse face SpMV, residual, dense diagonal-block action, and Chebyshev
  update after the unfused reference passes parity. GPU parity covers block
  sizes 2, 5, 7, and 10; the production p=6 result is recorded above.
- Remove solve-time allocations and host synchronization.

### Phase 4 — establish the SPD contract

- Enforce `R=P^T`, reversed adjoint post-smoothing, fixed spectral intervals,
  fixed coarse work, and an SPD terminal solve.
- Add numerical preconditioner-adjointness, positive-curvature, and PCG-vs-
  trusted-solve tests.

### Phase 5 — production FB-HP-MG with AMGX at p=0

- Add a backend selected independently of the existing AMGX CSR/BSR paths.
- Reuse p-level matrices, p=0 AMG hierarchy, coarse factors, and warm trace
  guesses over repeated Poisson right-hand sides.
- Keep scalar CSR/hybrid dispatch for low degree when measurements favor it.

### Phase 6 — profile and fuse

- Capture the fixed V-cycle in CUDA graphs, fuse compatible PCG updates and
  reductions, and publish per-level timing/operator-complexity data.
- Implement a custom scalar h-hierarchy only if p=0 AMGX is a measured hot-path
  bottleneck; it is not a prerequisite for validating p-coarsening.

### Phase 7 — qualification and dispatch

- Sweep the radius-5 unstructured trigonometric-Poisson disk at production
  sizes and p=1..9.
- Select the degree-dependent dispatcher from end-to-end warm-solve evidence,
  not from SpMV alone.

## Acceptance Gates

Correctness gates:

- BSR/CSR matvec relative difference <= 5e-13 in FP64;
- bilinear symmetry defect <= 1e-13 on representative matrices;
- modal transform and transfer adjointness near FP64 rounding error;
- every Galerkin level has positive sampled Rayleigh quotients;
- final true residual satisfies the common solver contract;
- reconstructed HDG fields agree with the trusted trace solution.

Performance gates:

- initial prototype iterations no more than 1.25 times the hybrid baseline;
- p>=4 fine SpMV target at least 1.4 times scalar CSR after setup is amortized;
- projected, then measured, warm solve at least 10% faster than scalar CSR;
- production promotion requires beating the hybrid path on the same matrix,
  right-hand side, initial guess, tolerance, precision, and GPU.

No method is promoted because of a faster standalone SpMV while its complete
solve is slower or less accurate.

## Long-Term Degree Extension

The BSR route is the priority for raw-CUDA Poisson p=7,8,9. Extend local
assembly limits and shared-memory/cooperative-solve policies for element degrees
36, 45, and 55 only after recording occupancy and capacity. Direct BSR block
sizes 8, 9, and 10, modal orientation, reconstruction, cached-RHS assembly, and
BSR-vs-host/CSR parity are required. COO/expanded CSR support at those degrees
is a secondary validation/debug path, not the production optimization target.
