# H100/H200 solver campaign — GENCI phase 2

Compare iterative trace-system solvers and multithreaded PyPardiso on identical assembled systems.
This is question Q2 of the submitted dossier. The scope includes Poisson, transport-dominated ADR,
anisotropy and difficult closed-loop transport. Local solver results and failures are the starting
point; none is an H100/H200 result.

| Item | Value |
|---|---|
| Status (2026-10-26) | Planned; combined task registry, Slurm launcher and analysis still to build |
| Target | One H100 or H200 per GPU task; PyPardiso uses the node's allocated CPU cores |
| Planned GPU time | About 12 hours; dossier estimate of roughly 250 trials at 3 minutes each |
| Output directory | `run_outputs/campaigns/h100_solvers` locally; `$WORK/hybridge/campaigns/h100_solvers` on the cluster |
| Output selection | Future launcher: `--out` or `HYBRIDGE_CAMPAIGN_OUT`; ADR drivers already take `--output` |
| Inputs | Portable ADR inventory and meshes, archived matrices where available, Poisson mesh/source specifications |
| Records | Local baseline snapshots under `adr_unified/records`, `adr_pardiso/records`, `adr_stress/records`, `solver_records/records`; no target-machine records yet |

## Trial matrix

| Axis | Planned coverage |
|---|---|
| Equations and cases | Poisson; regular and transport-dominated ADR; anisotropic diffusion; square/annulus trap, cross and orthogonal stress cases |
| Meshes | Four sizes per selected ladder, increasing toward the measured device-memory limit |
| Degrees | p = 2, 4, 6; transport paths through p = 9 where supported |
| Iterative candidates | Five to six appropriate candidates per family: AMGX, native face pMG-AMG, ASM or block-Jacobi with polynomial preconditioning and GMRES |
| Direct reference | PyPardiso LU, or the applicable SPD path, on exactly the same saved system |
| Measurements | Assembly, setup/factorization, cold solve and repeated solves separated; physical residual, iterations, peak CPU/GPU memory, timeout and failure reason |

Candidate support differs by equation and degree: do not turn an unavailable preconditioner into a
silent fallback. Failed local stress configurations remain explicit trials rather than disappearing
from the comparison. Match precision, normalization, ordering and stopping tolerances across runs.
Use at least 16 CPU threads or all allocated cores for the large direct solves; verify the backend
thread count and observed parallel CPU use. Environment variables alone do not establish that.
Performance trials use an exclusive GPU. Accuracy preparation may share it when memory permits.

## Inputs and local evidence

- [Portable ADR inventory](../../run_configs/adr_unified_l5/inventory.json) and its mesh assets:
  freeze the inventory revision and hashes. The unified worker reassembles the systems and uses
  the vendored solver tree; it does not depend on an uncommitted sibling checkout.
- Existing large data: `run_outputs/solver_studies/adr_scaling_2026_09_17` and the stress output
  directories. Copy required matrices and source inputs to WORK before a replay. The earlier
  plan estimated the local archive at about 76 GB; the collected DATA.md files give current
  regular-file sizes. The full expanded ADR matrix cache was estimated at **840 GiB** in the
  dossier; this is storage, not device memory, and must be checked against the selected tasks.
- [Unified ADR campaign](../adr_unified/README.md), [PyPardiso reference](../adr_pardiso/README.md),
  [stress campaign](../adr_stress/README.md), and [other solver records](../solver_records/README.md).
- [Poisson local comparison](../../docs/research/solver_studies/poisson_300k_solver_comparison_2026_09_22.md),
  [Poisson pMG degree campaign](../gc_poisson_pmg_degrees/README.md), and
  [pMG/AMG audit](../../docs/research/solver_studies/poisson_pmg_amg_audit_2026_10_16.md).

The node's actual free memory determines the largest trial. H100/H200 comparisons must record
which device was used and use identical problem sizes before reporting hardware ratios.

## Existing entry points

Inspect options without starting solves:

```bash
.venv/bin/python campaigns/adr_unified/run_adr_unified_campaign.py --help
.venv/bin/python campaigns/adr_pardiso/run_adr_pardiso_campaign.py --help
.venv/bin/python campaigns/adr_stress/run_closed_loop_stress.py --help
.venv/bin/python scripts/diffusion_reaction/benchmark_poisson_pmg_degrees.py --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.benchmark_poisson_backends --help
.venv/bin/python -m scripts.guiding_center.studies.magnetron.run_poisson_cg_vs_hdg --help
```

The ADR drivers have plan/inventory modes by default and require `--execute` to start their runs;
see their READMEs for the exact inputs and flags. The Poisson drivers start solves when invoked
without an inspection option. They are building blocks, not the missing combined cluster launcher.

## Preparation still required

Build deterministic task IDs and shard plans, stage input files with hashes, cap each worker's
memory/time, preserve per-run logs and failures, write done/failed markers, skip completed trials
on resume, and compare with the local baselines. Save small JSON/CSV records here after execution;
keep matrices and field checkpoints in WORK/STORE. Record solver-stack revisions and the node's
CPU affinity and measured thread use in provenance.

The target stack is prepared by the user for Hopper (sm_90), following the
[forked stack setup](../../docs/getting_started/forked_amgx_stack.md). No AMGX build or solver trial
has been started as part of organizing this folder. There is no combined start/resume/stop command
yet. Source: submitted `genci_h100_dossier/sections/justification.tex`, phase 2 and the data-management
section, with the requested H200 alternative recorded here on 2026-10-26.
