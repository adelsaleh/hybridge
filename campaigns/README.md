# Numerical campaigns

A campaign is a set of runs started by one launcher: an assembly or solver
benchmark, or a qualification of the software on one GPU. HYBRIDGE keeps the
launchers of its numerical campaigns here. Guiding-center experiments and
their launchers live in
[guiding-center-cg-hdg](../../guiding-center-cg-hdg/README.md).

- **Source:** `campaigns/<name>/`, with a README that gives the purpose and
  the commands.
- **Results on the machine that runs a campaign:** small records, compact
  outputs and notes go to `docs/campaigns/<name>/`, which git ignores (see
  its [index](../docs/campaigns/README.md)). Bulk data goes to `run_outputs/`,
  or under `$HYBRIDGE_SCRATCH` when that variable is set.
- **What a campaign teaches:** [`docs/lessons/`](../docs/lessons/README.md),
  written so that it holds on any machine.

## Campaigns

| Campaign | What it measures | Launcher |
|---|---|---|
| [`adr_h200_tuned`](adr_h200_tuned/README.md) | Iterative ADR solves with the tuned kernels (Schwarz, block-Jacobi, AMGX, hp-multigrid) on a ladder of meshes and degrees | `lane_stage.sh`, `run_adr_h200_tuned_campaign.py` |
| [`adr_pardiso`](adr_pardiso/README.md) | Direct PyPardiso solves of the same ADR systems on CPU nodes | `run_adr_pardiso_campaign.py` |
| [`adr_recovery`](adr_recovery/README.md) | ADR primal recovery at degrees 2 to 9: assembly forms, batched solvers, reuse against rebuild | `submit.py`, `run.py`, `analyze.py` |
| [`adr_selection_cost`](adr_selection_cost/README.md) | Host cost of choosing the ADR tensor kernel, and stalls of that choice | `run.py` |
| [`adr_stress`](adr_stress/README.md) | ADR closed-loop stress cases on a square and an annulus | `run_closed_loop_stress.py` |
| [`adr_unified`](adr_unified/README.md) | Iterative ADR solvers on one archived inventory of systems | `run_adr_unified_campaign.py` |
| [`architecture_tuning`](architecture_tuning/README.md) | Qualification of the kernel tuner on one GPU | `qualify.py`, `qualify.sbatch` |
| [`cpu_adr_scaling`](cpu_adr_scaling/README.md) | Thread scaling of the host ADR pipeline: assembly, direct solve, reconstruction, postprocessing | `run.py --plan/--execute` |
| [`fork_integration`](fork_integration/README.md) | Build and regression checks of the AMGX and pyamgx forks with HYBRIDGE | `build_primary.sh`, `qualify.sh`, `pytest_repositories.sh` |
| [`gpu_lane`](gpu_lane/README.md) | Not a measurement: a batch job that holds one GPU and runs listed stages one after the other | `lane.sbatch`, `<lane>/stages` |
| [`h100_assembly`](h100_assembly/README.md) | Assembly, recovery and profiling of the three equation families | `submit.sh`, `run.py` |
| [`h100_solvers`](h100_solvers/README.md) | Iterative and direct solvers for Poisson, transport, ADR and stress systems | `plan.py`, `expanded.py` |
| [`matk`](matk/README.md) | Solvers, orderings and incomplete factorizations for one exported CSR system | `*.sbatch`, `benchmark_amgx_strong.py --list-configs` |
| [`poisson_element_cost`](poisson_element_cost/README.md) | Host cost and degree 7 to 9 kernels of the Poisson right-hand side and reconstruction | `run.py --plan`, `poisson_element_cost.sbatch` |
| [`solver_briefings`](solver_briefings/README.md) | Builds a written summary from the records of the solver campaigns | `build_briefing.py OUTDIR` |
| [`solver_h200`](solver_h200/README.md) | Face-BSR, hp-multigrid and Krylov kernel variants | stages under `gpu_lane/stages/` |

The status of a campaign, with its jobs and dates, is kept with its records on
the machine that ran it.

## Recovered numerical studies

| Campaign | Documentation |
|---|---|
| `adr_solver_reports_20260917` | [adr_solver_reports_20260917](adr_solver_reports_20260917/README.md) |
| `legacy_adr_performance` | [legacy_adr_performance](legacy_adr_performance/README.md) |
| `legacy_adr_solver_studies` | [legacy_adr_solver_studies](legacy_adr_solver_studies/README.md) |
| `legacy_adr_validation` | [legacy_adr_validation](legacy_adr_validation/README.md) |
| `legacy_diffusion_gpu_hard` | [legacy_diffusion_gpu_hard](legacy_diffusion_gpu_hard/README.md) |
| `legacy_gpu_validation` | [legacy_gpu_validation](legacy_gpu_validation/README.md) |
| `qualification_adr_p7_p9_2026_10_15` | [qualification_adr_p7_p9_2026_10_15](qualification_adr_p7_p9_2026_10_15/README.md) |
| `solver_full_bsr_amg_walkthrough_20260913` | [solver_full_bsr_amg_walkthrough_20260913](solver_full_bsr_amg_walkthrough_20260913/README.md) |
| `solver_gmres_bsr_inspiration_20260913` | [solver_gmres_bsr_inspiration_20260913](solver_gmres_bsr_inspiration_20260913/README.md) |
| `solver_poisson_100k_comparison` | [solver_poisson_100k_comparison](solver_poisson_100k_comparison/README.md) |

The [source inventory](RELOCATION.md) and [relocation manifest](relocations.json)
record recovered code and its original hashes. Historical source snapshots and
recorded commands describe their original revision and execution location.

## Campaign conventions

- Keep launchers, job scripts, environment setup and a README together. Document
  the purpose, the exact plan, start, resume and stop commands, and the output
  location.
- Name no machine in a launcher. A job script takes the repository from
  `HYBRIDGE_REPO`, or from the directory it is submitted from, which is the
  repository root. Bulk output goes under `HYBRIDGE_SCRATCH`, which
  `common/hopper_env.sh` sets to `/scratch/$USER/hybridge` unless it is already
  set, and to `run_outputs/` without it.
- Reuse `common/` for provenance, resume, stopping, timing and record collection.
  `HYBRIDGE_CAMPAIGN_OUT` or a command-line option selects generated output.
- Provide inspection before execution, skip completed tasks on resume, and record
  failures separately from scientific convergence. A completion marker alone does
  not establish accuracy or convergence.
- Collect small text records with `python -m campaigns.common.records collect NAME
  --data DIR`. `--label` preserves multiple stages and `--include` selects nested
  records. Collection preserves source bytes and records their SHA256 hashes.
- Update this index and the campaign README when adding or completing a campaign.
  Frozen checkouts retain their original source and are never edited while running.

The [repository reconciliation](../docs/development/repository_reconciliation.md)
records the upstream split, retained H200 work and cross-repository record audit.

