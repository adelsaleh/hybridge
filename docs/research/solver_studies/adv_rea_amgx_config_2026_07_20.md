# Advection-Reaction AMGX Config Findings - 2026-07-20

> Historical record of 2026-07-20, moved from `run_logs/` on 2026-10-04. Its runner,
> run_adv_rea_gpu4_hdg.py, was later replaced by the runners under `scripts/gpu/`;
> the raw sweep files it cites are local, untracked evidence that is not
> distributed with the repository.

## Scope

This note records focused AMGX configuration tests for the then-current runner run_adv_rea_gpu4_hdg.py after the raw fused CUDA assembly work. The primary target was the existing production-style p6 advection-reaction case:

```bash
LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib \
HYBRIDGE_GPU4_AMGX_MONITOR=0 \
.venv/bin/python scripts/run_adv_rea_gpu4_hdg.py --case test2_legacy_gpu3 \
  -o 6 -mt rectangle --basis dub_orth --trace-basis legacy-lagrange \
  --assembly-backend raw-cuda --raw-local-assembly fused --raw-lu-mode coop \
  --raw-block-size 32 --trace-ordering none --volume-quad-1d 12 \
  --error-volume-quad-1d 24 -v 2
```

For modal trace sanity checks, I used `--assembly-backend cupy` in this AMGX sweep to keep the AMGX comparison focused and repeatable. A follow-up fused-raw modal parity check was performed afterward and is documented in
[`raw_cuda_fused_coop_lu_2026_07_20.md`](raw_cuda_fused_coop_lu_2026_07_20.md).

## Logs

Raw benchmark output and parsed JSON/CSV logs are preserved in:

- `run_logs/adv_rea_amgx_config_sweep_20260720_162358.json` (local, untracked): p6/ms0.01 legacy-lagrange raw-fused config sweep.
- `run_logs/adv_rea_amgx_config_sweep_20260720_162757.json` (local, untracked): p6/ms0.006 legacy-lagrange raw-fused config sweep.
- `run_logs/adv_rea_amgx_config_sweep_20260720_163011.json` (local, untracked): p6/ms0.005 legacy-lagrange raw-fused stress comparison.
- `run_logs/adv_rea_amgx_tolerance_sweep_20260720_163230.json` (local, untracked): p6/ms0.01 baseline tolerance sweep.
- `run_logs/adv_rea_amgx_modal_cupy_sweep_20260720_163342.json` (local, untracked): p6/ms0.01 legendre-modal CuPy assembly sanity comparison.
- [`raw_cuda_fused_coop_lu_2026_07_20.md`](raw_cuda_fused_coop_lu_2026_07_20.md): fused raw CUDA p<=8 matrix-level parity (including `legendre-modal`) and reconstruction checks.

The runner now prints `AMGX status` and `AMGX iterations` in the solver summary, and the sweep parser treats `AMGX iterations` as an integer.

## Config Sweep Results

p6/ms0.01, `legacy-lagrange`, raw fused CUDA cooperative LU:

| config | status | iters | AMGX setup | AMGX solve | scaled rel residual | L2 error | Linf error | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline BICGSTAB ILU0 W4 | success | 467 | 0.0886 | 1.0308 | 4.408e-13 | 1.705e-12 | 7.030e-12 | 6.9935 |
| BICGSTAB ILU0 V2 | success | 462 | 0.0948 | 1.0187 | 6.299e-13 | 1.263e-12 | 1.977e-11 | 6.9299 |
| BICGSTAB ILU0 W2 | success | 464 | 0.0847 | 1.0226 | 1.035e-12 | 1.636e-12 | 4.473e-11 | 6.8756 |
| BICGSTAB L1 aggressive | success | 465 | 0.0954 | 1.0245 | 1.027e-12 | 1.809e-12 | 3.323e-11 | 6.8623 |
| FGMRES L1 aggressive | not converged | 1500 | 0.2767 | 9.5119 | 3.629e-02 | 1.834e-01 | 6.111e+00 | 15.5517 |
| FGMRES aggregation DILU | not converged | 1500 | 0.2446 | 89.9752 | 1.000e+00 | 6.627e+00 | 8.243e+00 | 96.0859 |

p6/ms0.006, `legacy-lagrange`, raw fused CUDA cooperative LU:

| config | status | iters | AMGX setup | AMGX solve | scaled rel residual | L2 error | Linf error | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline BICGSTAB ILU0 W4 | success | 811 | 0.0974 | 4.7408 | 2.644e-12 | 8.618e-12 | 1.062e-10 | 19.2213 |
| BICGSTAB ILU0 V2 | success | 819 | 0.1074 | 4.7878 | 2.666e-12 | 8.417e-12 | 4.152e-11 | 19.1296 |
| BICGSTAB ILU0 W2 | success | 818 | 0.0905 | 4.7895 | 1.019e-11 | 3.747e-11 | 1.551e-10 | 19.0775 |
| BICGSTAB L1 aggressive | success | 808 | 0.0899 | 4.7327 | 8.073e-12 | 2.485e-11 | 1.828e-10 | 19.2057 |

p6/ms0.005, `legacy-lagrange`, raw fused CUDA cooperative LU:

| config | status | iters | AMGX setup | AMGX solve | scaled rel residual | L2 error | Linf error | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline BICGSTAB ILU0 W4 | success | 998 | 0.0978 | 8.2705 | 1.956e-11 | 6.121e-11 | 2.863e-10 | 29.5125 |
| BICGSTAB L1 aggressive | success | 987 | 0.1189 | 8.1854 | 4.920e-11 | 1.364e-10 | 7.052e-10 | 29.2257 |

## Tolerance Sweep

Baseline BICGSTAB ILU0 W4 at p6/ms0.01:

| AMGX tolerance | status | iters | AMGX solve | scaled rel residual | L2 error | Linf error | total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-14 | success | 465 | 1.0287 | 3.072e-12 | 1.363e-11 | 6.077e-11 | 6.9772 |
| 1e-12 | success | 462 | 1.0241 | 7.314e-13 | 2.111e-12 | 1.925e-11 | 6.9019 |
| 1e-10 | success | 458 | 1.0127 | 2.925e-11 | 3.567e-11 | 1.179e-09 | 6.8073 |
| 1e-8 | success | 443 | 0.9809 | 9.199e-09 | 2.382e-08 | 1.245e-06 | 6.7346 |

Relaxing tolerance gives only a small solve-time reduction until the error degrades visibly. I do not recommend changing the production tolerance based on these runs.

## Modal Trace Sanity Check

p6/ms0.01, `legendre-modal`, `volume_quad_1d=14`, CuPy assembly:

| config | status | iters | AMGX solve | scaled rel residual | L2 error | Linf error | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline BICGSTAB ILU0 W4 | success | 465 | 1.0283 | 8.531e-13 | 2.511e-12 | 4.408e-11 | 7.4413 |
| BICGSTAB L1 aggressive | success | 466 | 1.0320 | 7.702e-13 | 3.147e-12 | 1.286e-11 | 7.4031 |

This check used CuPy assembly deliberately to isolate the AMG configuration. Fused raw CUDA modal parity is documented separately with matrix-level parity/reconstruction checks in [`raw_cuda_fused_coop_lu_2026_07_20.md`](raw_cuda_fused_coop_lu_2026_07_20.md).

## Recommendation

Keep `configs/amgx/adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json` as the default advection-reaction AMGX configuration.

The current default remains the best safe choice: it is consistently convergent, has the best or near-best post-solve residual/error on the stress cases, and no tested alternative gave a repeatable AMGX solve-time improvement large enough to justify the worse residual/error behavior.

`configs/amgx/adv_rea_gpu4_hdg_bicgstab_classical_l1_aggressive.json` is retained as an experimental near-tie. It can be about 1 percent faster in AMGX solve on the p6/ms0.005 stress case, but it produced a larger scaled residual and roughly 2x larger L2 error there, so it should not replace the default without a broader accuracy tolerance decision.

Rejected directions from this pass:

- FGMRES with classical L1 aggressive AMG: hit the 1500-iteration cap at p6/ms0.01.
- FGMRES with aggregation/DILU AMG: hit the 1500-iteration cap and returned unusable residual/error.
- Lighter ILU0 AMG cycles/sweeps: no robust improvement on the heavier p6/ms0.006 case.
