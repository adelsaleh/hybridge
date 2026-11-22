# ADR unified iterative campaign (L1–L5)

Iterative replay of the archived ADR inventory: every archived system of the L1–L5 ladders
(`run_configs/adr_unified_l5/inventory.json`, meshes copied, matrices reassembled by the worker)
solved by the iterative candidates (AMGX, ASM and block-Jacobi with polynomial preconditioning,
native pMG-AMG) under one memory and timeout guard per run. It is the local basis of the H100 solver
campaign of the GENCI dossier (`campaigns/h100_solvers`).

| | |
|---|---|
| Measured | 2026-09-17, local GPU; the plan audit of 2026-09 is in `run_outputs/solver_studies/adr_unified_l5_plan_audit` |
| Launcher | `run_adr_unified_campaign.py` (plan, preflight or `--execute`); worker `adr_unified_worker.py` (one process per run, isolated solver tree from `vendor/adr_gmres`); `adr_unified_plan.py` (the plan); `export_adr_unified_inventory.py` (freezes archived specifications into the portable inventory, maintenance only) |
| Output directory | `run_outputs/solver_studies/adr_scaling_2026_09_17` (76 GB: assembled systems, samples, `comparison/results.csv`), `run_outputs/solver_studies/adr_unified_l5` for reruns |
| Records | `docs/research/solver_studies/adr_scaling_2026_09_17/unified/adr_unified_l5_campaign.md`, `docs/research/solver_studies/adr_scaling_2026_09_17/` (`scaling.data.json`, the synthesis), `run_configs/adr_unified_l5/inventory.json` |
| Tests | `tests/test_adr_unified_plan.py`, `tests/test_adr_unified_runner.py` |

## Commands

Plan and preflight (read-only):

```bash
.venv/bin/python -B campaigns/adr_unified/run_adr_unified_campaign.py --help
.venv/bin/python -B campaigns/adr_unified/run_adr_unified_campaign.py --output run_outputs/solver_studies/adr_unified_l5 --l5-triangles 400000
```

The full run (starts solves; only on a CUDA 13 GPU with the patched AMGX and pyamgx built and
activated) adds `--branch-root vendor/adr_gmres --maxiter 2000 --execute`; the complete command, the
timeouts and the memory guard are in `docs/research/solver_studies/adr_scaling_2026_09_17/unified/adr_unified_l5_campaign.md`.

## Host PyPardiso reference (genoa)

`host_pardiso.py` (sbatch `genoa_pardiso.sbatch`, submitter `submit_genoa_pardiso.sh`) factors
exported trace systems with PyPardiso on a whole genoa node (2x EPYC 9655, 754 GiB), with
isolated `campaigns/adr_pardiso` workers that re-hash matrix and RHS before every run.

| Output | Systems |
|---|---|
| `/scratch/user/hybridge/adr_unified_l5_pardiso_v2` | the unified device campaign's exported systems (`--device-campaign`): 190/190 planned systems covered by 274 passed thread measurements (record `docs/campaigns/adr_pardiso/records/retirement-20261031/RETIREMENT.md`, local, untracked). The device campaign stopped before its plan was complete, so this covers the systems it exported, not its full plan. |
| `/scratch/user/hybridge/adr_h200_tuned_lu` (records in `docs/campaigns/adr_pardiso/records/h200_tuned_lu`) | H200 tuned systems without LU (`--systems docs/research/solver_studies/adr_scaling_2026_09_17/h200_tuned/lu_followup_systems.csv`) |
| `/scratch/user/hybridge/adr_h200_tuned_lu_l4_lmax` | the 67 H200 systems still without LU: L4 at p = 3, 4, 5, 7, 8 (25) and Lmax (42), same list file |

PyPardiso calls MKL's 32-bit `pardiso` interface (no 64-bit path exists in hybridge). Its int32
indices bound the matrix (rows and entries), not the factors: the H200 LU follow-up factorized and solved
8M-DOF L6 systems whose factor count overflowed the int32 `iparm(18)` report (recorded as
unavailable, `factor_nnz` 0), all within the residual target. Systems whose matrix entries exceed
2^31-1 are recorded as `index_width_excluded` without running; `--max-factor-nnz` is an optional
extra cap. `--smallest-first` finishes the small systems first. Systems listed with `--systems` up to 2.5M trace DOFs take the thread counts of the nearest-size
v2 selection (`--borrow-selections`); larger ones run at 96 and 192 threads (`--fixed-threads`).
The same key names the same discrete system in both campaigns, but the matrices come from
different assembly kernels (stock versus tuned; relative differences 5e-14 to 6e-12), so
`operator_sha256` differs between them.

**Residual floor (2026-10-30).** At L7 the direct solve meets the FP64 floor of the fixed target
(L7 trigonometric p=2: true relative residual 1.2e-10 against 1e-10). Until then a miss
stopped the worker after its warm-up and discarded the timings. Now every trial and solve is
measured, and a run whose residuals all stay below `--floor-rtol` (1e-8) gets status
`residual_floor` (failure_kind `residual_floor`) with its fresh/reused times and
`worst_true_relative_residual`; worse residuals stay `numerical_failure`. No iterative refinement
is added (same protocol as v2). v2 had no residual misses, so its records are unchanged. The
memory model underestimates L7 peaks by about 1.6x at 96 threads (70.8 against 44.1 GiB) and
more at 192 (93.7 GiB); the RSS monitor, not the estimate, guards the node.

**Fresh versus the CPU pipeline (2026-10-30).** v2's fresh time is a real analysis and
factorization on every trial: the worker converts the exported blocks to CSR and calls
`pypardiso.ps.factorize` explicitly, then solves in phase 33. The `cpu_adr_scaling` campaign's
`warm_solve_s` (hybridge host pipeline, `solver='pypardiso'`) is not: `solve_pypardiso_system`
uses PyPardiso's process-wide solver, which skips factorization when the new matrix equals the
last one factored, and the campaign's warm solves re-assemble the identical system in the same
process. Its warm solve is therefore CSR assembly plus the matrix comparison plus a
phase-33 solve, about 2.3–2.9 times below v2's fresh time at L3–L4. Checked with two calls on
an identical, rebuilt 400,000-row matrix through `solve_pypardiso_system`: 0.67 s, then 0.11 s;
0.52 s again after `clear_pypardiso_cache()`. The pipeline's cold solve, the only one that
factorizes, is at or above v2's fresh time on the large systems (e.g. 0.93 against 0.80 s,
transport p=6, 32,768 triangles); at L1 it also carries first-call overhead. v2's fresh and
reused times, and the same measurements in `adr_h200_tuned_lu`, are the ones to compare with a
fresh or a repeated iterative solve.

Plan (read-only), then start from a frozen copy of the committed revision that contains this section:

```bash
.venv/bin/python -B campaigns/adr_unified/host_pardiso.py --systems <list.csv> --output /scratch/user/hybridge/adr_h200_tuned_lu \
    --borrow-selections /scratch/user/hybridge/adr_unified_l5_pardiso_v2 --fixed-threads 96 192 --smallest-first
HYBRIDGE_REPO=<copy> HYBRIDGE_PARDISO_SYSTEMS=<copy>/<list.csv> HYBRIDGE_PARDISO_OUT=/scratch/user/hybridge/adr_h200_tuned_lu \
HYBRIDGE_PARDISO_RECORDS=run_outputs/solver_studies/adr_h200_tuned_lu \
HYBRIDGE_PARDISO_ARGS='--borrow-selections /scratch/user/hybridge/adr_unified_l5_pardiso_v2 --fixed-threads 96 192 --smallest-first' \
sbatch --parsable --partition=genoa --nodes=1 --exclusive --cpus-per-task=192 --mem=0 --time=2-00:00:00 \
    --signal=B:USR1@900 --requeue --open-mode=append --job-name=adr-h200-tuned-lu \
    --output=<records>/slurm/%x-%j.out --export=ALL <copy>/campaigns/adr_unified/genoa_pardiso.sbatch
```

Resume: submit again; finished jobs are kept. Stop: `scancel`. When it is done, collect the records with
`python -m campaigns.common.records collect adr_pardiso --data /scratch/user/hybridge/adr_h200_tuned_lu --label h200_tuned_lu`.

## Additional saved output directories

- [adr_oscillatory_2026_09_18](../../docs/campaigns/adr_unified/records/adr_oscillatory_2026_09_18/DATA.md).
- [adr_scaling_2026_09_17](../../docs/campaigns/adr_unified/records/adr_scaling_2026_09_17/DATA.md).
- [adr_unified_l5_plan_audit](../../docs/campaigns/adr_unified/records/adr_unified_l5_plan_audit/DATA.md).
- [published_reports](../../docs/campaigns/adr_unified/records/published_reports/DATA.md).
