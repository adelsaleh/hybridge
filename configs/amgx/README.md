# AMGX Configs

These JSON files are the readable, reusable PyAMGX configurations for the standalone GPU4 HDG runners. The Python scripts keep embedded fallback copies, but when these files are present they are loaded by default. Use `--amgx-config` to run an edited copy without changing source code.

In the examples below, replace `/path/to/amgx/lib` with the directory containing
your AMGX shared library, for example `libamgxsh.so`. If AMGX is installed in a
system or environment path already known to the dynamic loader, the
`LD_LIBRARY_PATH=...` prefix is not needed.

## Advection-Reaction GPU4 HDGFEM

Config: `adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json`

Current working path:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python -m scripts.gpu.run_adv_rea_gpu4_hdg \
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

Modal trace AMGX checks in that sweep used CuPy assembly deliberately. A follow-up validation (`run_logs/raw_cuda_fused_coop_lu_findings_20260720.md`) validated fused raw CUDA modal trace behavior at matrix level through `p <= 8` before it is used for full modal production runs.


### Raw CUDA advection run modes

The advection runner uses the same AMGX config for CuPy, semi-fused raw, and fully fused raw assembly. Use the fully fused path when testing memory scaling:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  HDGFEM_GPU4_AMGX_MONITOR=0 \
  .venv/bin/python -m scripts.gpu.run_adv_rea_gpu4_hdg \
  -o 6 -ms 0.01 -mt rectangle --basis dub_orth --trace-basis legacy-lagrange \
  --assembly-backend raw-cuda --raw-local-assembly fused --raw-block-size 32 \
  --trace-ordering none --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 8 \
  --amgx-config configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json
```

Use the semi-fused path to compare against the Raw CUDA kernel that receives materialized local matrices:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  HDGFEM_GPU4_AMGX_MONITOR=0 \
  .venv/bin/python -m scripts.gpu.run_adv_rea_gpu4_hdg \
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

## Diffusion-Reaction GPU4 HDGFEM

Working configs:

- `diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json`: current default for nodal trace runs.
- `diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json`: second nodal candidate and experimental modal PCGF candidate.
- `diff_rea_gpu4_hdg_pcgf_classical_amg.json`: conservative classical AMG baseline and modal BICGSTAB preconditioner.

Current recommended nodal path:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python -m scripts.gpu.run_diff_rea_gpu4_hdg \
  -o 6 -ms 0.05 --basis dub_orth --trace-basis legacy-lagrange \
  --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 12 \
  --amgx-solver PCGF \
  --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json
```

Current recommended modal path:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python -m scripts.gpu.run_diff_rea_gpu4_hdg \
  -o 6 -ms 0.05 --basis dub_orth --trace-basis legendre-modal \
  --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 12 \
  --amgx-solver BICGSTAB \
  --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json
```

Recommendation summary from the latest coarse p=4..8 sweep at `-ms 0.18`:

- Nodal best: `legacy-lagrange + PCGF + Chebyshev/L1 aggressive AMG`. Fastest average solve path; default for `scripts/gpu/run_diff_rea_gpu4_hdg.py` when `--amgx-config` is omitted.
- Nodal second best: `legacy-lagrange + PCGF + ChebPoly4/L1 aggressive AMG`. Similar accuracy and solve time, with heavier setup.
- Nodal most robust: `legacy-lagrange + PCGF + classical V-cycle GS AMG`. Conservative SPD baseline; sometimes wins total time when Chebyshev setup dominates small/coarse runs.
- Modal best/most robust: `legendre-modal + BICGSTAB + classical AMG`. Modal PCGF is still preconditioner-sensitive even though the assembled matrix is symmetric.
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

- `--amgx-solver`
- `--amgx-tolerance`
- `--amgx-maxiter`

Set `HDGFEM_GPU4_AMGX_MONITOR=1` to enable AMGX residual/grid/timing prints for benchmark runs. Leave it unset for less noisy timing runs.

## Modal Trace Warning

The modal trace algebra is symmetric and gives the same direct-solve errors as the nodal trace in the checked cases, but modal PCGF remains sensitive to the AMG preconditioner. Use `legendre-modal + BICGSTAB + classical AMG` as the practical modal path. Use nodal `legacy-lagrange + PCGF` for the fastest robust production runs until modal AMG preconditioning is improved.
