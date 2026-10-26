# H100 turbulence campaign — GENCI phase 3

Measure the complete guiding-center pipeline after selecting kernels and solvers in phases 1–2.
This is question Q3 of the submitted dossier. It is separate from manufactured convergence and
from the existing local diocotron and equilibrium studies.

| Item | Value |
|---|---|
| Status (2026-10-26) | Planned; run table, Slurm launcher and analysis still to build |
| Target | One H100 per run |
| Planned GPU time | 6 hours in the submitted dossier |
| Output directory | `run_outputs/campaigns/h100_turbulence` locally; `$WORK/hybridge/campaigns/h100_turbulence` on the cluster |
| Output selection | Future launcher: `--out` or `HYBRIDGE_CAMPAIGN_OUT`; drivers take `--output-dir` or their documented output options |
| Inputs | Showcase cases/presets, refined meshes and diocotron profiles; phase-1/2 selected configurations |
| Records | No H100 run records yet; local D8 and diocotron records are linked below |

## Planned runs and measurements

| Group | Scope |
|---|---|
| Showcase reference | About 1,000 time steps of the dossier's guiding-center showcase; compare with its local timing basis of about 0.74 s per step |
| Time schemes | SI-BDF2, SI-BDF3 and IMEX-ARK3 on matched cases, mesh, tolerance and time interval; retain startup costs and stability failures |
| Large mesh | Approach one million triangles at p = 6, subject to measured memory use; the dossier's capacity model is an estimate |
| High-mode diocotron | Refined high-mode rings, with growth-rate, preservation and resolution diagnostics alongside step cost |

Separate initialization and JIT from steady step timing, retain Poisson/transport phase timings,
solver iterations and device memory, and distinguish timings on an exclusive GPU from shared-device
accuracy runs. Save fields at explicitly selected times, with conservation, extrema and positivity
settings in the run metadata. Snapshot cadence must be planned against actual output size; the
dossier anticipated large state files and STORE archives rather than committing fields to Git.

## Drivers and local references

Inspect options without evolving a case:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --help
.venv/bin/python campaigns/gc_d8_diocotron_order/launcher.py --plan
```

Use the [guiding-center command list](../../scripts/guiding_center/example_runs.md) for validated
preset names and options, the [showcase reproduction](../../docs/getting_started/gpu_showcase.md),
[D8 campaign and records](../gc_d8_diocotron_order/README.md), and
[diocotron shell history](../gc_diocotron_shell/README.md). D8's local selected phases are complete;
its full proposed ladder is still pending and must not be represented as already measured.

## Preparation still required

Select exact presets and time-step limits after the target-machine parity and solver checks.
Build the task registry, plan/list modes, Slurm launcher, per-task logging and provenance,
completion markers, skip-completed resume, and analysis that compares matched scientific accuracy.
Keep small summaries in this folder's future records directory and fields/movies in WORK/STORE.
There is no full-campaign start/resume/stop command yet.

Source: submitted `genci_h100_dossier/sections/justification.tex`, phase 3, transcribed on 2026-10-26.
Its 24-hour request also included one hour of initial checks and two hours for retries and
verification. The newer [article convergence campaign](../gc_article_convergence/README.md) was
prepared after submission and is additional work, not included in these six hours.
