# ADR unified iterative campaign (L1–L5)

Iterative replay of the archived ADR inventory: every archived system of the L1–L5 ladders
(`run_configs/adr_unified_l5/inventory.json`, meshes copied, matrices reassembled by the worker)
solved by the iterative candidates (AMGX, ASM and block-Jacobi with polynomial preconditioning,
native pMG-AMG) under one memory and timeout guard per run. It is the local basis of the H100 solver
campaign of the GENCI dossier (`campaigns/h100_solvers`).

| | |
|---|---|
| Status | complete (2026-09-17, local GPU); the plan audit of 2026-09 is in `run_outputs/solver_studies/adr_unified_l5_plan_audit` |
| Launcher | `run_adr_unified_campaign.py` (plan, preflight or `--execute`); worker `adr_unified_worker.py` (one process per run, isolated solver tree from `vendor/adr_gmres`); `adr_unified_plan.py` (the plan); `export_adr_unified_inventory.py` (freezes archived specifications into the portable inventory, maintenance only) |
| Output directory | `run_outputs/solver_studies/adr_scaling_2026_09_17` (76 GB: assembled systems, samples, `comparison/results.csv`), `run_outputs/solver_studies/adr_unified_l5` for reruns |
| Records | `docs/research/solver_studies/adr_unified_l5_campaign.md`, `docs/research/solver_studies/adr_scaling_2026_09_17/` (`scaling.data.json`, the synthesis), `run_configs/adr_unified_l5/inventory.json` |
| Tests | `tests/test_adr_unified_plan.py`, `tests/test_adr_unified_runner.py` |

## Commands

Plan and preflight (read-only):

```bash
.venv/bin/python -B campaigns/adr_unified/run_adr_unified_campaign.py --help
.venv/bin/python -B campaigns/adr_unified/run_adr_unified_campaign.py --output run_outputs/solver_studies/adr_unified_l5 --l5-triangles 400000
```

The full run (starts solves; only on a CUDA 13 GPU with the patched AMGX and pyamgx built and
activated) adds `--branch-root vendor/adr_gmres --maxiter 2000 --execute`; the complete command, the
timeouts and the memory guard are in `docs/research/solver_studies/adr_unified_l5_campaign.md`.
