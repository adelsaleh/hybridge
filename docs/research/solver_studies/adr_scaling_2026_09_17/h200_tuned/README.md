# H200 tuned kernels versus the RTX PRO 5000 Blackwell

[section.tex](section.tex) compares the H200 tuned ADR campaign
(`campaigns/adr_h200_tuned`, 28–29 October 2026) with this study's Blackwell
measurements. Both standalone wrappers (`main.tex`, `main_synthesis.tex`) now
append it after the closed-loop stress section.

## What is compared

- **Blackwell:** this study's smooth, oscillatory, focused and native-completion CSV files, plus the confirmed PyPardiso LU baseline (`scripts/reports/adr_lu_baseline.py`).
- **H200, stock kernels:** the unified ADR campaign on the same H200, for the systems and base candidates both runs share.
- **H200, tuned kernels:** the tuned campaign, exported to `docs/campaigns/adr_h200_tuned/records/solver_jobs.csv`.

Configurations are matched on geometry, case, triangle count, degree and the
candidate alias recorded in the H200 plan. The H200 systems carry the archived
operator hashes of this study's systems.

## Findings in brief

- **Same configuration:** stock kernels gain 1.1–1.5x over Blackwell. Tuned kernels gain 1.0–3.4x below 10^5 trace DOFs and up to 7.5x (ASM+PP), 7.9x (BJ+PP) and 4.1x ($p$MG–AMG) near 10^6. AMGX stays at 1.1–1.3x.
- **Where the gain comes from:** the cost per iteration falls 4.6–5.1x for ASM+PP and 2.7–2.8x for $p$MG–AMG at 10^6 DOFs. The true-target trigger also cuts iterations on 39 of 71 configurations.
- **Winners:** AMGX won the largest smooth Blackwell systems but wins none of the 25 jointly measured systems on the H200. The crossover against reused LU factors moves to smaller meshes.
- **H200 vs PyPardiso LU on a CPU node (2× EPYC 9655, 192 cores, 754 GiB; not the H200 node's own CPUs; all 114 systems, L1–L7):** 46 systems from PARDISO v2, 68 from the follow-up (records `h200_tuned_lu`). The best iterative fresh solve beats LU on 89 of the 104 systems with a passed iterative solve. LU keeps it on the smallest meshes, the cellular annulus from L5 and stress at L6 p=2 and L7. Reused LU beats the repeated iterative solve on 62 of 104. The GPU wins repeated solves on transport (15/20), variable velocity (14/20), trigonometric (11/18) and anisotropic L5 only. Cellular and stress stay LU's at every size (3.5–35x, 1.3–26x). LU at L7: 36–53 s fresh, 5–11 s reused, at most 141 GiB. Three L7 LU solves end at the residual floor (1.10–1.36e-10). PyPardiso's 32-bit interface limits the matrix indices, not the factor. LU and the iterative solvers see the same discrete systems assembled twice (stock exports for v2, tuned for the follow-up). The two hash differently and agree to 5e-14–6e-12 relative.
- **Beyond Blackwell (L5–L7, up to 20M DOFs):** AMGX is the fastest wherever it passes. Many non-passed jobs are as accurate as the best solution and only miss the fixed residual target at the FP64 round-off floor. L7 p=9 and L8–L9 do not fit on one H200.

## Files

| File | Content |
|---|---|
| `matched.csv` | One row per matched configuration: iterations, setup, cold, repeated, amortized on Blackwell, H200 stock and H200 tuned; ratios. |
| `family_best.csv` | Fastest passing member per family on each machine, winners, best iterative over P-LU. |
| `extension.csv` | H200 medians per level, degree and family, all levels. |
| `outcomes.csv` | H200 outcomes per level and family: passed, accurate at the residual floor, inaccurate, excluded. |
| `node_lu.csv` | Per H200 system: best iterative fresh/repeated and its outcome, PyPardiso LU fresh/reused on the CPU node with thread counts, ratios; `not_measured` where the LU campaign has no system. |
| `node_lu_crossover.csv` | Per case and degree: smallest trace DOFs from which iterative stays faster than LU, ratio at the largest size. |
| `end_to_end.csv` | Assembly plus one fresh solve on the 24 systems shared with `cpu_adr_scaling`. CPU node: host-pipeline assembly (warm, fastest thread count) plus the LU campaign's fresh solve. The pipeline's warm PyPardiso solve reused the cold run's factors (PyPardiso skips refactoring an identical matrix), so it is not used. H200: device assembly plus the fastest fresh iterative solve. |
| `stock_oscillatory.csv` | Classes outside the tuned ladder: best stock-kernel H200 solve (`records/stock_unified_all.csv`) against CPU-node LU, per LU-measured system. |
| `lu_followup_systems.csv` | H200 systems without an LU measurement, with the exported host system (`cache`, the PyPardiso input) and its operator hash: the input of the follow-up LU run, read back from `docs/campaigns/adr_pardiso/records/h200_tuned_lu`. |
| `*_table.tex` | Table fragments included by `section.tex`. |
| `validation.json` | Counts and status pairs of the match. |
| `assembly_kernels.csv`, `assembly_*_table.tex` | Assembly kernels (AR, DR, ADR) on the H200 against the Blackwell and per tuning step, from `scripts/reports/report_assembly_study.py --figures`. |
| `main_h200_short.tex` | Short version of `main_h200_synthesis.tex`: equations, level names, how each kernel was tuned, figures and tables only, with the assembly figures added. |

Figures are written to
`run_outputs/solver_studies/adr_scaling_2026_09_17/figures/h200_tuned/` (local,
untracked), as for the rest of the study.

## Regenerate

No solver runs. From the repository root:

```bash
# refresh the tracked H200 records from the scratch outputs (read only there)
.venv/bin/python -B -m campaigns.adr_h200_tuned.export_records
# CPU-node LU records (read only there; tracked under docs/campaigns/adr_pardiso/records/unified_l5_pardiso_v2)
.venv/bin/python -B -m campaigns.common.records collect adr_pardiso --data /scratch/user/hybridge/adr_unified_l5_pardiso_v2 --label unified_l5_pardiso_v2
# FP64 residual floor of small exported systems (CPU, PyPardiso; writes h200_tuned/fp64_floor.csv,
# which the report uses to mark NC*/FP64* runs)
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python -B scripts/advection_diffusion_reaction/diagnostics/fp64_residual_floor.py
# tables, CSV files and figures
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 .venv/bin/python -B scripts/reports/make_adr_h200_comparison.py
# assembly-kernel tables and figures (figures/h200_assembly; ADR p<=6 reads the local Q1 timings)
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 .venv/bin/python -B scripts/reports/report_assembly_study.py --figures
```

## Synthesis document (H200 versus PyPardiso LU on a CPU node)

[section_synthesis.tex](section_synthesis.tex) is the `main_synthesis` counterpart
for the H200: overview grids at p = 2, 6, 9 over L1–L7 and a degree grid at L4
(landscape pages), fresh/reused envelopes with the CPU-node LU curve, the crossover
table, and a practical interpretation. `main_h200_synthesis.tex` typesets it
followed by `section.tex`. The grids and envelopes are drawn by the Blackwell
synthesis' own functions (`make_adr_synthesis_section.comparison_overview`,
`focused_figure`), fed with H200 records; figures go to
`run_outputs/.../figures/h200_synthesis/`, the cells to `synthesis_cells.csv`.

Typesetting has not been run, as for the other sections of this study (the cluster
login host has no LaTeX distribution). For this section alone, `main_h200.tex`
is a standalone wrapper; from this folder run
`latexmk -pdf -interaction=nonstopmode -halt-on-error main_h200.tex`. It finds
the figures in the repository's `run_outputs` tree or, in a copy, in
`../figures/h200_tuned/`.
