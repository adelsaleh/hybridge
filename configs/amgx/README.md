# AMGX Configs

These JSON files are readable, reusable PyAMGX configurations for the CUDA HDG runners. The Python scripts keep embedded fallback copies, but load these files by default when present. Use `--amgx-config` to run an edited copy without changing source code.

The JSON basenames retain their original `adv_rea_gpu4_hdg_*` and
`diff_rea_gpu4_hdg_*` benchmark identifiers because archived run logs cite
them verbatim. This historical artifact exception does not apply to Python
packages, modules, runners, or newly generated sweep output names.

In the examples below, replace `/path/to/amgx/lib` with the directory containing
your AMGX shared library, for example `libamgxsh.so`. If AMGX is installed in a
system or environment path already known to the dynamic loader, the
`LD_LIBRARY_PATH=...` prefix is not needed.

## Advection-Reaction HDGFEM

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
- Preconditioner: classical AMG
- Selector: `PMIS`
- Cycle: `W`
- Smoother: `ILU0`
- Default tolerance override from CLI: `--amgx-tolerance 1e-14`
- Default iteration override from CLI: `--amgx-maxiter 1500`

Focused AMGX sweeps on 2026-07-20 did not find a safer faster replacement for this default. Keep `BICGSTAB + classical AMG/ILU0 W-cycle` as the production advection-reaction config. The closest alternative is `adv_rea_gpu4_hdg_bicgstab_classical_l1_aggressive.json`, retained only as experimental: it was about 1% faster in AMGX solve on the p6/ms0.005 stress case, but with a larger post-solve residual and about 2x larger L2 error. Details are in `run_logs/adv_rea_amgx_config_findings_20260720.md`.

Additional raw-CSR preconditioner checks on 2026-07-21 used `p=6`, `ms=0.01`, `dub_orth`, `legacy-lagrange`, `raw-cuda`, fused local assembly, cooperative LU, and `--amgx-tolerance 1e-10`. `adv_rea_gpu4_hdg_bicgstab_ilu0_amg_sweeps6.json` converged with 452 iterations and a 1.212 s global solve phase, compared with 454 iterations and 1.238 s for the default in that sample. Treat it as experimental: the gain is small enough to require repeated runs. `adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json` converged but did not improve iteration count or solve phase. `adv_rea_gpu4_hdg_fgmres_aggregation_dilu.json`, `adv_rea_gpu4_hdg_fgmres_amg_d2.json`, and `adv_rea_gpu4_hdg_gmres_amg_d2.json` are failed stronger-preconditioner experiments for this case; they either did not reduce the physical residual enough or were much slower.

Modal trace AMGX checks in that sweep used CuPy assembly deliberately. A follow-up validation (`run_logs/raw_cuda_fused_coop_lu_findings_20260720.md`) validated fused raw CUDA modal trace behavior at matrix level through `p <= 8` before it is used for full modal production runs.

Guiding-center transport presets currently use `adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json`.  Its AMGX residual history may show flat BiCGSTAB phases followed by a sharp drop, but it remains cheaper than the FGMRES aggregation/DILU variant on the long p=6 diocotron run.  Do not promote `adv_rea_gpu4_hdg_fgmres_aggregation_dilu.json` for that case without a longer multi-step benchmark; one-step smoke timings are misleading.

Experimental advection-reaction configs retained for comparison:

Zero-flux disk-tangent AMGX screen on 2026-07-27 used `scripts/gpu/run_advection_disk_tangent_cuda.py` with `p=4`, `ms=0.01`, `dub_orth`, `legacy-lagrange`, fused raw-CUDA CSR, cooperative LU, and `--amgx-tolerance 1e-10`. The default BICGSTAB/classical-ILU0 AMG route needed about 2200 iterations. `adv_rea_gpu4_hdg_pbicgstab_aggregation_dilu_postsmooth2.json` reduced this to 73 iterations using `PBICGSTAB + aggregation AMG + MULTICOLOR_DILU` with `presweeps=0`, `postsweeps=2`. This is a stronger diagnostic config, not yet the global default: each preconditioner application is much heavier, so the wall-clock solve was slightly slower on that screen. Use it with `--no-scale-system`; external row scaling made the PBICGSTAB aggregation-DILU variants fail, and AMGX internal `BINORMALIZATION` terminated before the runner summary with device-pool leak diagnostics.

- `adv_rea_gpu4_hdg_pbicgstab_aggregation_dilu_postsmooth2.json`: strong zero-flux disk-tangent diagnostic; requires `--no-scale-system`.
- `adv_rea_gpu4_hdg_bicgstab_classical_l1_aggressive.json`: L1 smoother baseline candidate.
- `adv_rea_gpu4_hdg_bicgstab_cheb_l1_aggressive.json`: Chebyshev/L1 smoother candidate.
- `adv_rea_gpu4_hdg_bicgstab_ilu0_amg_sweeps6.json`: heavier ILU0 W-cycle candidate.
- `adv_rea_gpu4_hdg_bicgstab_aggregation_dilu.json`: BICGSTAB with aggregation AMG/DILU.
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
  HDGFEM_CUDA_AMGX_MONITOR=0 \
  .venv/bin/python -m scripts.gpu.run_advection_reaction_cuda \
  -o 6 -ms 0.01 -mt rectangle --basis dub_orth --trace-basis legacy-lagrange \
  --assembly-backend raw-cuda --raw-local-assembly fused --raw-block-size 32 \
  --trace-ordering none --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 8 \
  --amgx-config configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json
```

Use the semi-fused path to compare against the Raw CUDA kernel that receives materialized local matrices:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  HDGFEM_CUDA_AMGX_MONITOR=0 \
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

## Diffusion-Reaction HDGFEM

Working configs:

- `diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json`: current relative-convergence default for nodal trace standalone diffusion runs.
- `diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json`: same PCGF + Chebyshev/L1 hierarchy with `convergence=ABSOLUTE`; used by guiding-center Poisson presets so `poisson_solver_atol` is the AMGX stopping tolerance.
- `diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json`: second nodal candidate and experimental modal PCGF candidate.
- `diff_rea_gpu4_hdg_pcgf_classical_amg.json`: conservative classical AMG baseline and modal BICGSTAB preconditioner.

Strict true-residual diagnostic:

- `diff_rea_gpu4_hdg_gmres_cheb_l1_classical_reliable.json` requires the
  `hdg/gmres-true-residual` AMGX branch. It enables explicit residual
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

Set `HDGFEM_CUDA_AMGX_MONITOR=1` to enable AMGX residual/grid/timing prints for benchmark runs. Leave it unset for less noisy timing runs.

## Modal Trace Warning

The modal trace algebra is symmetric and gives the same direct-solve errors as the nodal trace in the checked cases, but modal PCGF remains sensitive to the AMG preconditioner. Device symmetric scaling normalizes the modal matrix but does not fix the p6/ms0.04 PCGF convergence issue with the Chebyshev/L1 config, and it worsens nodal PCGF for that heavy case. Use `legendre-modal + BICGSTAB + classical AMG` as the practical modal path. Use nodal `legacy-lagrange + PCGF` for the fastest robust production runs until modal AMG preconditioning is improved.
