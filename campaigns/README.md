# Campaigns

Every campaign of the repository lives here: its launcher, a README and the records of its finished
or partial runs. A campaign is a set of runs started by one detached launcher (locally with
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
  `*.csv`, `*.md`, `*.txt` under 1 MiB at the top level of the output directory: run index, selected
  presets, reports, provenance, completion markers, failure lists) and a `DATA.md` with the absolute
  data path, total size, run count, completion marker and status. Fields, snapshots, movies and logs
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
