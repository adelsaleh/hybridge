# Campaigns

Campaign launchers, planned H100 studies, and the records of complete or partial local studies live here. A campaign is a set of runs started by one detached launcher (locally with
`setsid nohup`, on a cluster through Slurm); the drivers it starts live in `scripts/`. The H100
phases of the GENCI dossier (assembly, solvers, turbulence), the article's convergence campaign, the
ADR solver campaigns and the local guiding-center campaigns of the article are listed below with
their status and output directories.

## Index

| Campaign | Model | Target | Status | Launcher | Output directory | Records |
|---|---|---|---|---|---|---|
| [`gc_article_convergence`](gc_article_convergence/README.md) | guiding center, manufactured convergence (article Section 5, Appendices B–C) | H100 (Jean Zay, Slurm) or local GPU | prepared 2026-10-26, not started | `campaign.py list/plan/run`, `slurm/submit.sh` | `run_outputs/campaigns/gc_article_convergence`, `$WORK/hybridge/campaigns/gc_article_convergence` | `mesh_manifest.json`, `cost_estimates.json`, `dt_table*.json` |
| [`adr_unified`](adr_unified/README.md) | ADR, iterative solvers on the archived L5 inventory | local GPU | complete (2026-09-17) | `run_adr_unified_campaign.py` | `run_outputs/solver_studies/adr_scaling_2026_09_17` | `docs/research/solver_studies/adr_unified_l5_campaign.md`, `adr_scaling_2026_09_17/` |
| [`adr_pardiso`](adr_pardiso/README.md) | ADR, PyPardiso direct solves of the archived systems | local CPU | complete (2026-09-22) | `run_adr_pardiso_campaign.py` | `run_outputs/solver_studies/adr_scaling_2026_09_17/pardiso_all_2026_09_22` | `docs/research/solver_studies/adr_pardiso_campaign.md` |
| [`adr_stress`](adr_stress/README.md) | ADR closed-loop stress cases (square, annulus) | local GPU and CPU | complete | `run_closed_loop_stress.py` | `run_outputs/solver_studies/adr_closed_loop_stress_*`, `adr_square_*` | `docs/research/solver_studies/adr_closed_loop_stress_runner.md` |

| [gc_d8_diocotron_order](gc_d8_diocotron_order/README.md) | D8 diocotron order study | local GPU | Partial: phases I and X completed; the full D8 ladder remains pending. | `launcher.py --phases I` | `run_outputs/article/diocotron_order_20261016` | [records](gc_d8_diocotron_order/records/DATA.md) |
| [gc_d8s_smoke](gc_d8s_smoke/README.md) | D8-S device preset and kernel smoke study | local GPU | SMOKE_DONE present; earlier failed and out-of-memory attempts retained. | `launcher.py` | `run_outputs/article/diocotron_order_smoke_20261016` | [records](gc_d8s_smoke/records/DATA.md) |
| [gc_e7_band_evolution](gc_e7_band_evolution/README.md) | E7 control and seeded band evolution | local GPU | E7_DONE present; eight production runs completed, earlier smoke failures retained. | `launcher.py` | `run_outputs/article/e7_band_evolution_20261019` | [records](gc_e7_band_evolution/records/DATA.md) |
| [gc_e7_five_sites](gc_e7_five_sites/README.md) | E7 five-site and varicose star seeds | local GPU | X5_DONE present for the final selection; abandoned exploratory seeds were not all completed. | `launcher.py` | `run_outputs/article/e7_five_sites_20261019` | [records](gc_e7_five_sites/records/DATA.md) |
| [gc_e7_lumps](gc_e7_lumps/README.md) | E7 geometry-dependent lump seeds | local GPU | LUMPS_DONE_a/b/c present; SKIP markers identify deliberately omitted runs. | `launcher.py --lane a` | `run_outputs/article/e7_lumps_20261020` | [records](gc_e7_lumps/records/DATA.md) |
| [gc_eq_centerward](gc_eq_centerward/README.md) | Centerward equilibrium continuation | local CPU/MPI | completed.json records the accepted four geometries. | `launcher.py` | `run_outputs/article/equilibrium_centerward_20261015` | [records](gc_eq_centerward/records/DATA.md) |
| [gc_eq_dt080](gc_eq_dt080/README.md) | Equilibrium placement and high-order refinement | local CPU/MPI | Placement and high-order stages completed; refined-source limitations remain in the records. | `launcher.py placement` | `run_outputs/article/equilibrium_dt080_20261016` | [records](gc_eq_dt080/records/DATA.md) |
| [gc_eq_dt080_eps0005](gc_eq_dt080_eps0005/README.md) | Sharper equilibrium continuation | local CPU/MPI | completed.json and the recovery records describe the accepted stages and remaining failures. | `launcher.py` | `run_outputs/article/equilibrium_dt080_eps0005_20261016` | [records](gc_eq_dt080_eps0005/records/DATA.md) |
| [gc_poisson_pmg_degrees](gc_poisson_pmg_degrees/README.md) | Poisson p-multigrid degree comparison | local GPU | POISSON_PMG_DONE present; individual solver failures remain in runs/results.jsonl. | `launcher.py` | `run_outputs/article/poisson_pmg_degrees_20261017` | [records](gc_poisson_pmg_degrees/records/DATA.md) |
| [gc_d7_magnetron](gc_d7_magnetron/README.md) | D7 magnetron campaign stages | local GPU | Pilot, diagnosis and production directories have all-done markers; aborted and stale runs remain on disk. | `launch_production_v2.sh B` | `run_outputs/article/magnetron_d7_production_v2` | [records](gc_d7_magnetron/records/DATA.md) |
| [gc_e9_cg_vs_hdg](gc_e9_cg_vs_hdg/README.md) | E9 condensed CG versus HDG Poisson | local GPU | Original and retiming all-done markers present. | `launch_retime.sh` | `run_outputs/article/e9_cg_vs_hdg` | [records](gc_e9_cg_vs_hdg/records/DATA.md) |
| [gc_conv_poisson_degree](gc_conv_poisson_degree/README.md) | Poisson-degree convergence and follow-up | local GPU | Original all-done present; extension-done absent. The H100 campaign contains the new convergence reruns. | `run_campaign.sh` | `run_outputs/article/convergence_poisson_degree` | [records](gc_conv_poisson_degree/records/DATA.md) |
| [gc_conv_unstructured](gc_conv_unstructured/README.md) | Unstructured-mesh convergence | local GPU | all-done present; completed local baseline, with accurate-error reruns prepared for H100. | `run_campaign.sh` | `run_outputs/article/convergence_unstructured` | [records](gc_conv_unstructured/records/DATA.md) |
| [gc_conv_exact_drift](gc_conv_exact_drift/README.md) | BDF3 exact-drift transport reference | local GPU | Aligned marker present; per-run CSVs and check records are retained. | `run_aligned.sh` | `run_outputs/article/convergence_exact_drift` | [records](gc_conv_exact_drift/records/DATA.md) |
| [gc_diocotron_shell](gc_diocotron_shell/README.md) | Diocotron shell campaign history | local GPU | Mixed completed, superseded and interrupted batches; no single marker certifies the whole history. | `todo_20261015/launcher.py` | `run_outputs/article/diocotron` | [records](gc_diocotron_shell/records/DATA.md) |
| [gc_article_records](gc_article_records/README.md) | Additional E7 and equilibrium studies | local CPU/GPU | mixed; see saved evidence | historical launchers remain with data | `run_outputs/article/` | per-study manifests |
| [solver_records](solver_records/README.md) | Additional direct/iterative solver baselines | local CPU/GPU | mixed; see saved evidence | study drivers | `run_outputs/solver_studies/` | per-study manifests |

`common/` holds what every campaign shares: `protocol.py` (GPU lock, idle-GPU wait, nvidia-smi
sampler, contamination requeue, pause groups, provenance, resume), `reporting.py` (events, progress
and saved-result inspection of the ADR campaigns).

## Conventions

- **One folder per campaign** with the launcher(s), a `README.md` (purpose, the article section or
  dossier phase it feeds, status with date, exact start, resume and stop commands, output directory,
  what the records are, follow-ups) and `records/` for finished or partial campaigns.
- **Output directories.** H100 campaigns write to `run_outputs/campaigns/<campaign>` locally and to
  `$WORK/hybridge/campaigns/<campaign>` on Jean Zay (`HYBRIDGE_CAMPAIGN_OUT` overrides); article
  campaigns write to `run_outputs/article/<study>`, the symlink to `gc-hdg-article/run_outputs/<study>`;
  ADR campaigns to `run_outputs/solver_studies/<study>`. A launcher takes its output directory from
  an option or `HYBRIDGE_CAMPAIGN_OUT`, never from its own location.
- **Records.** `records/` keeps copies of the campaign's small record files (`*.jsonl`, `*.json`,
  `*.csv`, `*.md`, `*.txt` under 1 MiB at the top level plus selected nested summaries and inputs: run index, selected
  presets, reports, provenance, completion markers, failure lists) and a `DATA.md` with the absolute
  data path, total size, file count, run-attempt counts when available, completion evidence and status. A manifest records original paths and SHA256 hashes. Fields, snapshots, movies and logs
  stay in the output directory.
- **Running.** Every campaign runs detached (`setsid nohup <launcher> > <out>/driver_stdout.log 2>&1 < /dev/null &`),
  reports its process group (`kill -- -<pgid>` stops it), writes a completion marker and skips
  completed runs on resume. Only performance campaigns hold `.cache/gpu-campaign.lock`; accuracy
  campaigns wait for free device memory instead.
- **Reproducibility.** A launcher records the hybridge commit, the sha256 of the tracked diff and of
  the untracked sources it copies into `<out>/source/`, the GPU, driver, CUDA and Python versions and
  its own arguments (`common/protocol.py`).
- **Documentation.** The campaign's commands also appear in the study README and in
  `scripts/guiding_center/example_runs.md`; `tests/test_guiding_center_example_runs.py` checks them.

## Adding a campaign

1. Create `campaigns/<name>/` with the launcher and a README following the conventions above.
2. Give the launcher `--plan` (print every run without solving), resume by completed markers, a
   completion marker and the provenance record of `common/protocol.py`.
3. Add the row above and the commands to `scripts/guiding_center/example_runs.md` (campaign section).
4. When the campaign ends, collect its records (`records/`) and set the status here and in its README.
5. Update the `CODEMAP.md` rows.

## Local records and portability

`local_sources.json` records the original launcher paths and hashes. Their copies use the current
study module names and configured output locations. Fixed historical process-ID waits and signals
were removed; originals remain unchanged in the data directories. Historical shell versions are
listed separately from current selections in each campaign README.

`common/paths.py` resolves `HYBRIDGE_ARTICLE_ROOT`, `HYBRIDGE_CAMPAIGN_OUT`,
`HYBRIDGE_EQUILIBRIUM_REPO` and `HYBRIDGE_FENICS_PYTHON`; `common/shell.sh` supplies the same locations
to shell launchers. The HYBRIDGE interpreter is selected by `HYBRIDGE_PYTHON`; the shared protocol
accepts `HYBRIDGE_CAMPAIGN_CPU_CORES` for affinity. No local data are copied or moved by a plan command.
The local `run_outputs/article/<study>` aliases point to the sibling article checkout's output folders.
On another machine, restore the required inputs/checkpoints and set the location variables explicitly.

Collect another stage without overwriting the main snapshot:

```bash
.venv/bin/python -m campaigns.common.records collect gc_d7_magnetron --data run_outputs/article/magnetron_d7_pilot --label magnetron_d7_pilot --include 'runs/*/*_summary.json' --status 'Recorded stage; inspect completion evidence'
```

The committed records are an evidence snapshot, not a copy of the large data. Some old launchers count
failed attempts as processed: phase markers alone do not certify scientific completion. D8's selected
phases I/X are complete while its full ladder remains pending. Additional later E7 and solver studies
have data-catalog entries even when their original launchers remain with their output directories.
