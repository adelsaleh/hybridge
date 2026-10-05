# AMGX Configs

> **Required post-release stack:** these configurations are qualified with
> `adelsaleh/AMGX@hdg-cuda13-integration` at `583084b` and
> `adelsaleh/pyamgx@quality-of-life` at `81efd1e`, not the upstream `main`
> branches. Follow the [fork build/install guide](../../docs/getting_started/forked_amgx_stack.md)
> before reproducing AMGX-backed results.

These JSON files are readable, reusable PyAMGX configurations for the CUDA HDG runners. The Python scripts keep embedded fallback copies, but load these files by default when present. Use `--amgx-config` to run an edited copy without changing source code.

At guiding-center runner verbosity `-v 3`, HYBRIDGE enables AMGX solve
statistics. Transport defaults to `print_solve_stats_interval=10`; an explicit
JSON value is preserved (use `1` for every iteration). The initial and final
rows and any divergence exit reason are always printed when the table is
active. The local AMGX formatter shows one aggregate block-L2 residual column,
relative-to-initial and relative-to-previous ratios, and used/held device
memory. Sampling affects printing only: convergence checks and stored residual
history still run every iteration. Direct solver verbosity `2` or `>=4` retains
the Python backend micro-timing/configuration diagnostics.

Guiding-center AMGX transport enables a native guard on the primary and every
AMGX retry. After ten startup iterations, five consecutive completed iterations
above 1000 times the best monitored residual request a fresh `b-A*x` check.
Confirmed growth returns `AMGX_ST_DIVERGED` immediately, allowing the bounded
retry policy to continue. Non-finite residuals terminate immediately, including
during startup. The reference has a floor of 64 times vector-precision epsilon
times the initial residual, so a tiny recursive residual does not make ordinary
roundoff count as explosive growth. Finite growth in restarted GMRES/FGMRES is
verified only when the current solution has been formed. Isolated spikes and
large initial residuals alone do not stop a solve.

The outer solver JSON keys are `rel_div_tolerance` (default `1000` for transport,
nonpositive disables), `divergence_patience` (`5`), and
`divergence_grace_iters` (`10`). Explicit values are preserved. Poisson keeps its
own configuration. These controls require rebuilding the local AMGX fork;
Python-side rejection alone cannot interrupt an active native solve. Independent
physical residual acceptance remains mandatory. This guard detects explosive
growth; it does not declare every plateau a failure.

The JSON basenames retain their original `adv_rea_gpu4_hdg_*` and
`diff_rea_gpu4_hdg_*` benchmark identifiers because archived run logs cite
them verbatim. This historical artifact exception does not apply to Python
packages, modules, runners, or newly generated sweep output names.

In the examples below, replace `/path/to/amgx/lib` with the directory containing
your AMGX shared library, for example `libamgxsh.so`. If AMGX is installed in a
system or environment path already known to the dynamic loader, the
`LD_LIBRARY_PATH=...` prefix is not needed.

## FP32 Guiding-Center Transport

`adv_rea_gpu4_hdg_fgmres_scaled_none.json` provides restarted FGMRES (50 vectors,
300 iterations, explicit `NOSOLVER` preconditioner, relative tolerance 5e-3).
It retains raw-CUDA BSR assembly, diagonal row scaling and independently checked
physical/scaled residuals. This is the FP32 guiding-center replacement for the
stock unpreconditioned BiCGSTAB configuration, which can diverge in FP32.
See the [FP32 run and validation notes](../../docs/development/fp32_guiding_center.md).

## Advection-Reaction HYBRIDGE

Config: `adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json`

Current working path:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python -m scripts.gpu.run_advection_reaction_cuda \
  -o 6 -ms 0.01 --basis dub_orth --trace-basis legacy-lagrange \
  --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 12 \
  --amgx-config configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json
```

Solver summary:

- Outer solver: `BICGSTAB`
- Active preconditioner: none; AMGX's plain `BICGSTAB` class does not instantiate
  the nested `preconditioner` object retained in this historical JSON.
- BSR SpMV backend: `cusparse_generic` in the outer solver scope. With the local
  CUDA-13 AMGX patch this controls BiCGStab's fine-operator `A*p` and `A*s`
  products and overrides the old 3x3/4x4 AMGX specializations.
- Default tolerance override from CLI: `--amgx-tolerance 1e-14`
- Default iteration override from CLI: `--amgx-maxiter 1500`

Correction recorded on 2026-08-24: earlier documentation described this route as
`BICGSTAB + classical AMG/ILU0 W-cycle`. Source inspection and AMGX runtime
telemetry show that it is unpreconditioned BiCGStab. The BICGSTAB JSON variants
with different nested AMG objects therefore did not compare active
preconditioners; their small timing/error differences must be treated as run
variation rather than preconditioner evidence. `PBICGSTAB` is the separate AMGX
solver that constructs and applies a configured preconditioner.

For raw-CUDA BSR input, all advection BICGSTAB/PBICGSTAB configs explicitly
request CUDA-13 generic BSR SpMV. Scalar CSR input is unchanged: the backend
setting does not convert storage and the scalar SpMV route remains active. The
generic path requires the matching local AMGX source patch and CUDA 13.0 Update
1 or newer; older or unsupported partial/distributed views fall back to legacy
cuSPARSE BSR.

Warmed matched p=1..6 measurements on nx=64, 128, and 256 structured meshes
show that BSR gains grow with problem size. At nx=256, BSR reduced AMGX solve
time by 14%, 25%, 18%, and 24% for p=1, 3, 5, and 6 respectively; p=4 was
neutral and p=2 remained 3% slower. Nsight confirmed that p=2 used CUDA-13
generic 3x3 BSR rather than AMGX's historical custom kernel. The BSR SpMV GPU
work was about 1.42x faster than CSR, but block-vector reductions and repeated
descriptor/workspace handling erased its complete-solve gain. See
[`advection_bsr_benchmark_20260824.md`](../../docs/backends/advection_bsr_benchmark_20260824.md).

The unstructured-square legacy `test2` follow-up in that document screened
PBICGSTAB preconditioners that remain available for all face-block dimensions
p=1..6. The best compatible choice was unscaled scalar-row `JACOBI_L1` with
one iteration, unit relaxation, and
`jacobi_l1_scalar_rows_for_blocks=1`. It was about twice as fast as direct
block Jacobi at p=6, `ms=0.02`, but generally did not beat plain scaled
BICGSTAB on the fine `ms=0.01` mesh. Keep BICGSTAB as the default; use
PBICGSTAB+L1 as the block-size-safe preconditioned comparison. Aggregation AMG remains unsupported for pure-BSR p=6. The repaired
`MULTICOLOR_DILU` path supports 7×7 blocks and passed its algebra tests, but
did not improve Poisson convergence in the
150k/300k study (local, untracked: `artifacts/full_bsr_convergence_20260914/`).

The absolute-tolerance Poisson fallback
`diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json` also enables
`jacobi_l1_scalar_rows_for_blocks=1` in its Chebyshev L1 preconditioner.
This is needed when a native FB-HP-MG solve falls back with p=6 face BSR
(block size 7); scalar CSR behavior is unchanged.

For pure BSR at p=1..3, the stronger measured option is
`adv_rea_gpu4_hdg_pbicgstab_dilu_bsr_p1_p3.json`: direct PBICGSTAB with one
unscaled `MULTICOLOR_DILU` application, parallel-greedy level-1 coloring, and
relaxation 0.7. On the 92,552-triangle `test2` mesh it reduced median
end-to-end wall time relative to PBICGSTAB+L1 by 5%, 20%, and 8% at p=1,2,3.
Do not use this preset as an all-order default. It was 14% slower at p=4, and
experimental 6x6/7x7 dispatches for p=5/6 did not converge and were not
retained. Run it with `--raw-matrix-format bsr --no-scale-system`; keep plain
scaled BICGSTAB as the overall default and L1 as the p=4..6 PBICGSTAB fallback.

Additional raw-CSR preconditioner checks on 2026-07-21 used `p=6`, `ms=0.01`, `dub_orth`, `legacy-lagrange`, `raw-cuda`, fused local assembly, cooperative LU, and `--amgx-tolerance 1e-10`. `adv_rea_gpu4_hdg_bicgstab_ilu0_amg_sweeps6.json` converged with 452 iterations and a 1.212 s global solve phase, compared with 454 iterations and 1.238 s for the default in that sample. Treat it as experimental: the gain is small enough to require repeated runs. `adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json` converged but did not improve iteration count or solve phase. `adv_rea_gpu4_hdg_fgmres_aggregation_dilu.json`, `adv_rea_gpu4_hdg_fgmres_amg_d2.json`, and `adv_rea_gpu4_hdg_gmres_amg_d2.json` are failed stronger-preconditioner experiments for this case; they either did not reduce the physical residual enough or were much slower.

Modal trace AMGX checks in that sweep used CuPy assembly deliberately. A follow-up validation ([raw CUDA fused cooperative LU findings](../../docs/research/solver_studies/raw_cuda_fused_coop_lu_2026_07_20.md)) validated fused raw CUDA modal trace behavior at matrix level through `p <= 8` before it is used for full modal production runs.

Guiding-center device presets use `adv_rea_gpu4_hdg_bicgstab_scaled_none.json`, whose JSON contains no inactive nested preconditioner. HYBRIDGE applies left row scaling and supplies the accepted density trace as the initial guess. The k100/k50 stress family uses an AMGX stopping tolerance of `1e-8` while independently retaining its `1e-11`/`5e-9` physical residual contract; a rejected primary first tries `PBICGSTAB` with one `JACOBI_L1` application (`adv_rea_gpu4_hdg_pbicgstab_l1_bsr.json`), then one `BLOCK_JACOBI` application (`adv_rea_gpu4_hdg_pbicgstab_block_jacobi_bsr.json`). Both start from zero, keep native BSR, inherit `transport_scale_system` (left row scaling in the device presets), use the configured transport relative tolerance, and rebuild their cheap preconditioner data for the current matrix. L1 uses `jacobi_l1_scalar_rows_for_blocks=1`; block Jacobi inverts the dense diagonal blocks. If both fail, the policy enters FGMRES with direct `MULTICOLOR_DILU` and the same scaling option, followed by at most two residual-correction solves. There is no repeated unpreconditioned retry. Because AMGX DILU is not enabled for the p=6 face block size, only that fallback expands face BSR to scalar CSR with a raw-CUDA device kernel. The stateful solver keeps the first DILU factors as a fixed FGMRES preconditioner, replaces matrix coefficients in place on later steps, and never materializes the matrix on host. On the 157,280-triangle p=6 case, the accepted primary reduced warm transport from 32 to 23 iterations and from about 0.366 s to 0.340 s; the fallback did not occur.

A recoverable block-Jacobi setup or solve exception advances immediately to the
FGMRES/`MULTICOLOR_DILU` attempt. A CUDA device memory fault can leave the context
unusable, so catching that exception alone cannot make a later fallback succeed.
The 2026-09-11 investigation reproduced out-of-bounds writes in AMGX's large-block
Jacobi setup (block sizes above five): its temporary buffer was sized by grid
blocks but indexed by threads. The local AMGX fixes remove that buffer, correct
the diagonal BSR multiply's block count and backend selection, and fix the sign
and half-warp synchronization in the large-block inverse. Rebuild `amgxsh` after
applying these native changes. `tests/test_amgx_bsr_retry_preconditioners.py`
checks a single Jacobi update against a dense per-block solve with AMGX pooling
disabled, and injects recoverable BJ errors to verify the next real FGMRES/DILU
solve succeeds. These checks use small synthetic matrices, without time stepping.
After rebuilding the patched CUDA 13 library on 2026-09-11, all 30 tests passed
in each of FP64 and FP32 under Compute Sanitizer memcheck (zero errors). The ten
large-block update cases also passed a targeted racecheck of the native Jacobi
setup kernel with zero hazards.

The optional guiding-center flag `--transport-direct-fallback cusolver-qr` adds
an unscaled device sparse QR solve after all six AMGX attempts. It is disabled
by default. Every attempt records a fresh physical `b-A*x` residual separately
from its solver-coordinate residual; failed stages also write boundary,
divergence, normal-jump and matrix row-scale diagnostics. See
[the transport investigation](../../docs/development/transport_boundary_diagnostics.md)
for tangency conditions, the localized initial-data preset, and small-test
scaling/direct-solve evidence.

Experimental advection-reaction configs retained for comparison:

Zero-flux disk-tangent AMGX screen on 2026-07-27 used `scripts/gpu/run_advection_disk_tangent_cuda.py` with `p=4`, `ms=0.01`, `dub_orth`, `legacy-lagrange`, fused raw-CUDA CSR, cooperative LU, and `--amgx-tolerance 1e-10`. The default BICGSTAB/classical-ILU0 AMG route needed about 2200 iterations. `adv_rea_gpu4_hdg_pbicgstab_aggregation_dilu_postsmooth2.json` reduced this to 73 iterations using `PBICGSTAB + aggregation AMG + MULTICOLOR_DILU` with `presweeps=0`, `postsweeps=2`. This is a stronger diagnostic config, not yet the global default: each preconditioner application is much heavier, so the wall-clock solve was slightly slower on that screen. Use it with `--no-scale-system`; external row scaling made the PBICGSTAB aggregation-DILU variants fail, and AMGX internal `BINORMALIZATION` terminated before the runner summary with device-pool leak diagnostics.

- `adv_rea_gpu4_hdg_pbicgstab_aggregation_dilu_postsmooth2.json`: strong zero-flux disk-tangent diagnostic; requires `--no-scale-system`.
- `adv_rea_gpu4_hdg_pbicgstab_l1_bsr.json`: first guiding-center retry; PBICGSTAB with the configured transport scaling with one scalar-row L1 Jacobi application directly on BSR.
- `adv_rea_gpu4_hdg_pbicgstab_block_jacobi_bsr.json`: second guiding-center retry; PBICGSTAB with the configured transport scaling with one native block-Jacobi application.
- `adv_rea_gpu4_hdg_pbicgstab_dilu_bsr_p1_p3.json`: direct block-DILU comparison for unscaled p=1..3 BSR only.
- `adv_rea_gpu4_hdg_bicgstab_classical_l1_aggressive.json`: historical unpreconditioned BICGSTAB variant; nested L1 configuration is inactive.
- `adv_rea_gpu4_hdg_bicgstab_cheb_l1_aggressive.json`: historical unpreconditioned BICGSTAB variant; nested Chebyshev/L1 configuration is inactive.
- `adv_rea_gpu4_hdg_bicgstab_ilu0_amg_sweeps6.json`: historical unpreconditioned BICGSTAB variant; nested ILU0 sweep count is inactive.
- `adv_rea_gpu4_hdg_bicgstab_scaled_none.json`: explicit guiding-center unpreconditioned BICGSTAB primary; left scaling is applied by HYBRIDGE.
- `adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json`: historical unpreconditioned BICGSTAB comparison; nested aggregation/DILU configuration is inactive.
- `adv_rea_gpu4_hdg_fgmres_aggregation_dilu.json`: failed p6/ms0.01 FGMRES+DILU experiment.
- `adv_rea_gpu4_hdg_fgmres_amg_d2.json`: failed p6/ms0.01 FGMRES+D2 experiment.
- `adv_rea_gpu4_hdg_gmres_amg_d2.json`: failed GMRES+D2 experiment.

Representative raw-CSR preconditioner sweep:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python scripts/gpu/sweep_cuda_hdg.py \
  --orders 6 --mesh-sizes 0.01 --bases dub_orth \
  --trace-bases legacy-lagrange --quad-rules default \
  --amgx-configs default \
    configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg_sweeps6.json \
    configs/amgx/adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json \
    configs/amgx/adv_rea_gpu4_hdg_fgmres_aggregation_dilu.json \
    configs/amgx/adv_rea_gpu4_hdg_fgmres_amg_d2.json \
  --assembly-backend raw-cuda --raw-local-assembly fused \
  --raw-lu-mode coop --raw-matrix-format csr --raw-block-size 32 \
  --amgx-maxiter 1500 --amgx-tolerance 1e-10 --check-rtol 1e-10
```


### Raw CUDA advection run modes

The advection runner uses the same AMGX config for CuPy, semi-fused raw, and fully fused raw assembly. Use the fully fused path when testing memory scaling:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  HYBRIDGE_CUDA_AMGX_MONITOR=0 \
  .venv/bin/python -m scripts.gpu.run_advection_reaction_cuda \
  -o 6 -ms 0.01 -mt rectangle --basis dub_orth --trace-basis legacy-lagrange \
  --assembly-backend raw-cuda --raw-local-assembly fused --raw-block-size 32 \
  --trace-ordering none --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 8 \
  --amgx-config configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json
```

Use the semi-fused path to compare against the Raw CUDA kernel that receives materialized local matrices:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  HYBRIDGE_CUDA_AMGX_MONITOR=0 \
  .venv/bin/python -m scripts.gpu.run_advection_reaction_cuda \
  -o 6 -ms 0.01 -mt rectangle --basis dub_orth --trace-basis legacy-lagrange \
  --assembly-backend raw-cuda --raw-local-assembly precomputed --raw-block-size 32 \
  --trace-ordering none --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 8 \
  --amgx-config configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json
```

Working fused stress runs on the 24 GB Quadro RTX 6000:

| p | mesh size | status |
|---:|---:|---|
| 6 | 0.006 | robust full solve, about 2.70M trace dofs |
| 6 | 0.005 | full solve fits, but AMGX residual/error degrade |
| 5 | 0.004 | robust full solve; precomputed raw OOMs during COO-to-CSR |
| 4 | 0.004 | robust full solve |

For p6/ms0.004, fused assembly completes but the current CuPy COO-to-CSR conversion OOMs before AMGX setup. The next memory target is the global sparse conversion/solver path, not local fused assembly.

## Advection-Diffusion-Reaction HYBRIDGE

- `adv_diff_rea_gpu4_hdg_fgmres_amg_block_graph_dense_dilu_bsr.json`: FGMRES
  (restart 75) preconditioned by one V-cycle of classical block-graph-dense AMG
  with one-sweep `MULTICOLOR_DILU` pre/post smoothing and a dense-LU coarse
  solve; natural face BSR is required. It is the transport-dominated winner of
  the [matched ADR solver study](../../docs/research/solver_studies/adr_solver_comparison_2026_09_17.md)
  (`K=1e-3 I`, `beta=(1,0.5)`, p=6, 99,458 triangles: 35 iterations versus 263
  for direct block DILU). The stored absolute tolerance was removed and
  convergence is `RELATIVE_INI_CORE`, so runners supply the tolerance. The ADR
  runner preset `tensor_cuda_bsr_amg` uses it. Fork-only keys: it fails on
  upstream AMGX. The study reports that transferred block AMG does not rescue
  the cellular low-diffusion or directional-anisotropy oscillatory classes.

## Diffusion-Reaction HYBRIDGE

Working configs:

- `diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json`: current relative-convergence default for nodal trace standalone diffusion runs. It sets `use_scalar_norm=1`, so face-BSR solves stop on the aggregate scalar L2 residual instead of independently over-solving every block component; scalar CSR behavior is unchanged.
- `diff_rea_gpu4_hdg_fgmres_cheb_l1_block_graph_identity_bsr.json`: validated opt-in pure-BSR classical hierarchy using a Frobenius block graph, D2 scalar weights, identity-lifted BSR transfers, weighted BSR Galerkin, and FGMRES; requires the accompanying patched AMGX source.
- `diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_identity_bsr.json`: validated PCGF path for the identity-lifted pure-BSR hierarchy. After fixing the multilevel correction overrun, all radius-5 disk cases at 99,896, 124,831, and 150,209 triangles for p=1..6 converge; its high iteration count reflects weak interpolation, not PCGF incompatibility.
- `diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_dense_bsr.json`: validated one-step/additive PCGF/Chebyshev baseline for fixed-support dense block interpolation, exact block transpose, and dense BSR Galerkin. It requires the patched AMGX source and sets `aggressive_levels=0`: aggressive D2 can leave fine block rows without interpolation support, whereas the dense mode enforces `sum_c P_ic = I_b` on every row.
- `diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_block_graph_dense_bsr.json`: opt-in fully BSR PCGF with true face-block Jacobi inside order-two Chebyshev, 0 pre-sweeps and 3 post-sweeps. It uses fixed weighted-power spectral estimates (`chebyshev_lambda_estimate_mode=4`), direct zero-start block-Jacobi corrections, and reuse of Chebyshev’s initial correction; these require the accompanying AMGX patches. The two cycle shortcuts reduced matched p=6 warm solve time, including preconditioning, from 408 to 247 ms at 157,280 triangles and from 896 to 544 ms at 315,425 triangles, with identical iteration counts and residual histories. All 80 native algebra tests passed. The preset uses absolute tolerance 1e-13 and a 300-iteration cap; runners can override the tolerance. See the cycle-cost report (local, untracked: `artifacts/full_bsr_cycle_cost_20260914/`) and earlier smoothing study (local, untracked: `artifacts/full_bsr_smoothing_steps_20260913/`).
- `diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_constant_vector_bsr.json`: experimental fully BSR interpolation for scalar Poisson in nodal coordinates. It preserves `P*1=1` and exact coarse-point injection while allowing other block components to relax. The existing additive constraint remains the default. The AMGX `constant_vector` patch is built and validated. Fresh three-step p=6 runs gave 26/21/19 iterations and 239.19 ms warm solve at 157,280 triangles, and 27/22/19 and 451.01 ms at 315,425 triangles. Hybrid still achieved 16/13/12 at both sizes, with 106.22/203.70 ms warm solve. All 106 smoother/interpolation/DILU algebra tests passed; this remains opt-in. See the 150k/300k convergence study (local, untracked: `artifacts/full_bsr_convergence_20260914/`).
- `diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_extended_i_dense_bsr.json`: rejected block Extended+i diagnostic retained for reproduction. It is correct and memory-safe, but is one to two PCGF iterations worse than projected Jacobi on the 152,909-triangle p=2/p=6 cases at thresholds 0.25 and 0.47.
- `diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_scalar_guided_dense_bsr.json`: rejected mode-aware coarse-face diagnostic retained for reproduction. Any-mode promotion regresses the production-size p=2 case from 20 to 28 iterations and its temporary scalar expansion exceeds the available p=6 setup memory.
- `diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_inverse_scaled_dense_bsr.json`: completed diagnostic for the symmetric inverse-diagonal block-action metric. It is slightly worse than raw Frobenius and has higher setup cost; retain it for reproduction, not production.
- `diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_normalized_dense_bsr.json`: completed diagnostic for diagonal-normalized Frobenius strength. It shifts the p=6 hierarchy transition but does not beat the raw-Frobenius optimum; retain it for reproduction, not as a production default.
- `diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_energy_bsr.json` and `diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_energy_strong_bsr.json`: rejected diagnostic configs retained only to reproduce the interpolation screen. Right normalization greatly increases iterations and fails on deeper levels with two or more smoothing steps; do not use these configs for production solves.
- `diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json`: same PCGF + Chebyshev/L1 hierarchy with `convergence=ABSOLUTE`; used by guiding-center Poisson presets so `poisson_solver_atol` is the AMGX stopping tolerance.
- `diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_robust_abs.json`: strong coefficient-exact hybrid BSR/CSR Poisson stage, selected by the signed and positive turbulence presets. It keeps PCGF, uses a scalar-expanded classical hierarchy, order-four `CHEBYSHEV` with scalar-row `JACOBI_L1`, symmetric 2+2 smoothing, four coarse sweeps, and no aggressive level. `chebyshev_lambda_estimate_mode=2` uses the established L1-preconditioned spectral estimate. The historical filename is retained for existing response files; the smoother must be `CHEBYSHEV`, because `CHEBYSHEV_POLY` rejects the fine face-BSR blocks and ignores a nested L1 preconditioner. The wrapper records scalar residual diagnostics without an iterate-vector history.
- `diff_rea_gpu4_hdg_pcgf_classical_gs_robust_abs.json`: pure scalar-CSR PCGF retry with symmetric 2+2 multicolor Gauss-Seidel and a stronger fixed V-cycle. The turbulence policy reuses this hierarchy for zero-start and residual-correction attempts.
- `diff_rea_gpu4_hdg_fgmres_dilu_robust_abs.json`: scalar-CSR terminal escape hatch: one FGMRES attempt with direct `MULTICOLOR_DILU`, reached only after all native, hybrid, and pure-CSR PCGF attempts fail.
- `diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json`: second nodal candidate and experimental modal PCGF candidate.
- `diff_rea_gpu4_hdg_pcgf_classical_amg.json`: conservative classical AMG baseline and modal BICGSTAB preconditioner.
- `diff_rea_gpu4_hdg_pcgf_aggregation_block_jacobi_bsr.json`: stock-AMGX face-BSR aggregation-AMG path for p=1..4 (block sizes 2..5).
- `diff_rea_gpu4_hdg_pcgf_aggregation_block_jacobi_frobenius_bsr.json`: opt-in p=1..4 diagnostic using whole-block Frobenius aggregation weights and fused 2x2/3x3/5x5 block Jacobi; requires the accompanying patched AMGX source to be rebuilt.
- `diff_rea_gpu4_hdg_pcgf_block_jacobi_bsr.json`: face-BSR fallback for p=5..6 (block sizes 6..7), which the current AMGX aggregation kernels do not instantiate.

Direct face-BSR comparison:

```bash
.venv/bin/python -m scripts.diffusion_reaction.compare_cuda_bsr_csr
```

The default comparison is trigonometric Poisson on an unstructured radius-5
disk with mesh size 0.0345 (about 153,000 triangles) at p=6. It keeps scalar
CSR on the established PCGF/classical-AMG configuration and chooses a
degree-compatible BSR configuration. On the 2026-08-18 p=6 run, BSR reduced
compressed-pattern storage from 217.2 MiB to 5.2 MiB and trace assembly from
1.569 s to 0.937 s. AMGX solve time increased from 0.447 s (26 CSR AMG
iterations) to 8.834 s (3,089 BSR block-Jacobi iterations), so p=6 BSR is
currently an assembly/storage improvement, not a faster complete solve.

For p=1..4, the stock BSR config uses aggregation AMG because AMGX classical
AMG rejects block matrices. Consequently, a CSR-classical versus
BSR-aggregation timing is not a storage-format-only comparison. The patched
Frobenius config is an A/B diagnostic for two measured block-path costs: it
uses all dense-block entries when selecting aggregates and replaces the two
legacy BSR multiplies in generic 2x2, 3x3, and 5x5 block-Jacobi sweeps with one
fused kernel. The 4x4 path was already fused. External HDG scaling remains
disabled for BSR until block-aware scaling is implemented.

Strict true-residual diagnostic:

- `diff_rea_gpu4_hdg_gmres_cheb_l1_classical_reliable.json` requires the
  `hdg-cuda13-integration` branch of AMGX and the `quality-of-life` branch of
  PyAMGX. It enables explicit residual
  verification and DGKS reorthogonalization without changing AMGX's global
  `Epsilon_conv`. It is not a production preset: on the 113,894-triangle
  Gaussian-annulus k=3 screen, restart 50 reached only a (1.158\times10^{-11})
  true relative residual after 500 iterations (13.76 s). Restart 20/DGKS and
  restart 50/ALWAYS showed the same floor after 200 iterations, so the planned
  three-step run was rejected by the (10^{-12}) qualification gate.

Current recommended nodal path:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python -m scripts.gpu.run_diffusion_reaction_cuda \
  -o 6 -ms 0.05 --basis dub_orth --trace-basis legacy-lagrange \
  --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 12 \
  --amgx-solver PCGF \
  --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json
```

Current recommended modal path:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python -m scripts.gpu.run_diffusion_reaction_cuda \
  -o 6 -ms 0.05 --basis dub_orth --trace-basis legendre-modal \
  --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 12 \
  --amgx-solver BICGSTAB \
  --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json
```

Additional device scaling diagnostics on 2026-07-26 used `scripts/gpu/diagnose_diffusion_matrix_scaling.py` with the existing PCGF Chebyshev/L1 config unchanged.  Symmetric Jacobi scaling is cheap and preserves symmetry, but it is not recommended for this config: on the p6/ms0.04 raw-CUDA CSR case it increased nodal PCGF iterations from 32 to 242, and modal PCGF still reached the 2000 iteration limit.  The modal matrix remained symmetric to roundoff, so the current evidence points to AMG/coarsening sensitivity rather than a raw-CUDA orientation bug.  Details are in `docs/research/solver_studies/diffusion_amgx_2026_07.md`.

Focused modal preconditioner sweeps on 2026-07-26 used `scripts/gpu/sweep_diffusion_amgx_preconditioners.py`.  Disabling aggressive coarsening avoids several modal setup failures and higher Chebyshev orders reduce modal PCGF iterations, but the heavier preconditioner applications did not beat the existing modal choices on the p6/ms0.04 raw-CUDA CSR case.  Representative heavy unscaled solve times were 0.452 s for nodal PCGF/Cheb-L1 aggressive, 23.978 s for modal PCGF/Cheb-L1 aggressive, 28.869 s for modal non-aggressive Cheb order 6, 31.012 s for modal non-aggressive Cheb order 10, and 5.034 s for modal BICGSTAB/classical at physical residual 1.448e-09.  Non-Chebyshev candidates from the local AMGX sources were also generated and screened at p6/ms0.18 and p6/ms0.04 with symmetric scaling controls.  The best fine non-Cheb candidate was direct `BICGSTAB + MULTICOLOR_DILU` at about 5.37 s solve and physical residual 6.70e-11, close to but not better than the existing modal BICGSTAB/classical fallback.  Symmetric diagonal scaling stayed cheap but did not improve the practical modal runs.  AMGX internal `solver.scaling` defaults to `NONE` and is not enabled in these configs; `error_scaling=3` is coarse-grid correction scaling, not hidden matrix scaling.  Details are in `docs/research/solver_studies/diffusion_amgx_2026_07.md`.

AMGX hierarchy stats for p6 nodal/modal aggressive Chebyshev/L1 runs are recorded in `docs/research/solver_studies/diffusion_amgx_2026_07.md`.  The modal hierarchy is smaller than the nodal hierarchy on the fine p6/ms0.04 case, with operator complexity about 1.008 versus 1.038, but PCGF convergence is much worse.  Lowering `dense_lu_num_rows` from 2048 to 128 fixes the modal p6/ms0.18 setup failure and reduces nodal fine setup in one sample, but it does not fix modal PCGF iteration count; repeat this before changing the default config.

Recommendation summary from the latest coarse p=4..8 sweep at `-ms 0.18`:

- Nodal best: `legacy-lagrange + PCGF + Chebyshev/L1 aggressive AMG`. Fastest average solve path; default for `scripts/gpu/run_diffusion_reaction_cuda.py` when `--amgx-config` is omitted.
- Nodal second best: `legacy-lagrange + PCGF + ChebPoly4/L1 aggressive AMG`. Similar accuracy and solve time, with heavier setup.
- Nodal most robust: `legacy-lagrange + PCGF + classical V-cycle GS AMG`. Conservative SPD baseline; sometimes wins total time when Chebyshev setup dominates small/coarse runs.
- Modal best/most robust: `legendre-modal + BICGSTAB + classical AMG`. Modal PCGF is still preconditioner-sensitive even though the assembled matrix is symmetric. Direct `BICGSTAB + MULTICOLOR_DILU` is a close diagnostic baseline, but it is not promoted because it did not beat the classical AMG fallback on the fine p6 screen.
- Modal PCGF experimental: `legendre-modal + PCGF + ChebPoly4/L1 aggressive AMG`. It can improve modal PCGF accuracy, but one p=4 coarse setup took about 95 s, so it is not a default.

Representative coarse timings, all on 5,699 triangles with `dub_orth`, `volume_quad_1d=2p`, and `error_volume_quad_1d=24`:

| p | trace/solver/config | L2 error | assembly | CSR | AMGX setup | AMGX solve | reconstruct | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | nodal PCGF classical | 1.281e-03 | 0.448 | 0.150 | 0.197 | 0.090 | 0.013 | 2.047 |
| 4 | nodal PCGF Cheb/L1 | 1.281e-03 | 0.427 | 0.129 | 0.180 | 0.065 | 0.012 | 1.931 |
| 4 | nodal PCGF ChebPoly4/L1 | 1.281e-03 | 0.456 | 0.149 | 0.190 | 0.069 | 0.013 | 2.034 |
| 4 | modal BICGSTAB classical | 1.281e-03 | 0.479 | 0.134 | 0.090 | 0.390 | 0.012 | 2.228 |
| 5 | nodal PCGF classical | 6.598e-05 | 0.521 | 0.164 | 0.204 | 0.112 | 0.019 | 2.193 |
| 5 | nodal PCGF Cheb/L1 | 6.598e-05 | 0.466 | 0.134 | 0.252 | 0.075 | 0.018 | 2.094 |
| 5 | nodal PCGF ChebPoly4/L1 | 6.598e-05 | 0.456 | 0.134 | 0.253 | 0.078 | 0.020 | 2.083 |
| 5 | modal BICGSTAB classical | 6.598e-05 | 0.525 | 0.145 | 0.086 | 0.427 | 0.017 | 2.308 |
| 6 | nodal PCGF classical | 5.464e-06 | 0.517 | 0.153 | 0.194 | 0.107 | 0.033 | 2.219 |
| 6 | nodal PCGF Cheb/L1 | 5.464e-06 | 0.501 | 0.144 | 0.223 | 0.074 | 0.033 | 2.151 |
| 6 | nodal PCGF ChebPoly4/L1 | 5.464e-06 | 0.505 | 0.161 | 0.255 | 0.086 | 0.033 | 2.261 |
| 6 | modal BICGSTAB classical | 5.464e-06 | 0.530 | 0.140 | 0.085 | 0.487 | 0.032 | 2.445 |
| 7 | nodal PCGF classical | 2.090e-07 | 0.554 | 0.147 | 0.201 | 0.211 | 0.063 | 2.329 |
| 7 | nodal PCGF Cheb/L1 | 2.090e-07 | 0.547 | 0.151 | 0.251 | 0.082 | 0.063 | 2.303 |
| 7 | nodal PCGF ChebPoly4/L1 | 2.090e-07 | 0.567 | 0.153 | 0.257 | 0.088 | 0.063 | 2.344 |
| 7 | modal BICGSTAB classical | 2.090e-07 | 0.597 | 0.150 | 0.085 | 0.516 | 0.063 | 2.591 |
| 8 | nodal PCGF classical | 1.485e-08 | 0.660 | 0.153 | 0.202 | 0.202 | 0.112 | 2.592 |
| 8 | nodal PCGF Cheb/L1 | 1.485e-08 | 0.682 | 0.154 | 0.344 | 0.096 | 0.112 | 2.653 |
| 8 | nodal PCGF ChebPoly4/L1 | 1.485e-08 | 0.661 | 0.153 | 0.338 | 0.102 | 0.113 | 2.641 |
| 8 | modal BICGSTAB classical | 1.485e-08 | 0.700 | 0.139 | 0.089 | 0.586 | 0.111 | 2.874 |

## Runtime Overrides

Both standalone runners load the JSON config first, then apply these command-line overrides:

- `--amgx-solver`, only when explicitly supplied
- `--amgx-tolerance`
- `--amgx-maxiter`

Set `HYBRIDGE_CUDA_AMGX_MONITOR=1` to enable AMGX residual/grid/timing prints for benchmark runs. Leave it unset for less noisy timing runs.

## Modal Trace Warning

The modal trace algebra is symmetric and gives the same direct-solve errors as the nodal trace in the checked cases, but modal PCGF remains sensitive to the AMG preconditioner. Device symmetric scaling normalizes the modal matrix but does not fix the p6/ms0.04 PCGF convergence issue with the Chebyshev/L1 config, and it worsens nodal PCGF for that heavy case. Use `legendre-modal + BICGSTAB + classical AMG` as the practical modal path. Use nodal `legacy-lagrange + PCGF` for the fastest robust production runs until modal AMG preconditioning is improved.
