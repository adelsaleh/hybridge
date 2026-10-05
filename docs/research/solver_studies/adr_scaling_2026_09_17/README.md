# ADR results section

[section.tex](section.tex) is the combined, insertable results section. It
progresses from smooth three-regime scaling through the focused variable-
velocity, transport-dominated, and high-diffusion comparisons, then closes with
the oscillatory cellular-geometry study. The focused comparisons report the
completed AMGX versus PP, BJ+PP, and ASM+PP measurements without short aliases.
Both standalone documents now append a shared
[closed-loop stress section](closed_loop_stress/README.md), with analytic
geometry/exact-solution figures and the preliminary strong-preset results of
22 September 2026: 8/60 passes, all on the orthogonal control, plus complete
residual coverage and converged-only timing comparisons.
Both integrate the confirmed PyPardiso LU baseline into the existing timing
figures, overview grids, and endpoint/stress tables, using the brief shared
[measurement protocol](pardiso_lu/protocol.tex). Its $\mathrm{P\!-\!LU}_n$
alias gives the selected MKL thread count; curve-point numbers give $n$.
The CPU campaign passed all 107 archived systems, including the four
trapping/crossing systems with no passing GPU iterative solve. LU grounds
the comparison of ASM/BJ+PP against AMGX; it has no separate results section
or artificial Krylov iteration count. Geometry, exact-field, accuracy and
preconditioner-only profile plots retain their original meaning.
The shared report helper `scripts/reports/adr_lu_baseline.py` matches original
suite/system identities to independently confirmed pilot-selected timings.
Both portable bundles include `pardiso_lu/confirmed_timings.csv` and
`pardiso_lu/comparisons.csv` for the CPU measurements and matched GPU records.

The portable bundle (`run_outputs/solver_studies/adr_scaling_2026_09_17/adr_results_bundle.zip`, local, untracked)
contains the combined section, standalone wrapper, vector figures, measurements,
validation records, geometric inputs and numerical source snapshots. The original
oscillatory findings are preserved inside this study, including the 22,825-triangle
case, which was **not rerun**. Old oscillatory paths are compatibility symlinks.

## Contents

- [Smooth reference data](scaling.data.json), [CSV](scaling.csv) and
  [audit](validation.json): 96 passing solver/system configurations on 24 systems.
- [Oscillatory study](oscillatory/README.md): three classes on two geometries,
  the same eight mesh/order points and per-class solver coverage; 192 configurations
  on 48 systems, including four reused square outcomes.
- [Initial oscillatory stress tests](oscillatory/initial/README.md): the original
  68 configurations, source-only control, weak anisotropy, 6,114/22,825-triangle
  annular results, conditioning, quadrature and application profiles.
- [Closed-loop stress tests, preliminary results](closed_loop_stress/README.md):
  nine-lobed trapping, crossing circulation and perpendicular through-flow;
  all 60 residuals and the eight passing configurations' timings appear with
  the analytic figures in both `main.tex` and `main_synthesis.tex`.
- [Focused comparison subsection](comparison/subsections.tex): 50 endpoint and
  144 scaling configurations, with all solver families shown in three figures
  and the largest-system table.
- Figures (`run_outputs/solver_studies/adr_scaling_2026_09_17/figures/`, local, untracked):
  PDF, SVG and PNG. Generated PDFs stay outside the documentation tree.
- Oscillatory raw outputs (`run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/`, local, untracked):
  new scaling jobs, caches, diagnostics, meshes and preserved initial runs.

The two CSV files use milliseconds. `mean`, `minimum` and `maximum` describe hot
solve time; `setup`, `fresh` and `amortized` are medians. Amortization includes two
solves per setup. Failed configurations retain their residual and iteration count
and do not enter time rankings. The JSON archives retain individual samples,
effective configurations, acceptance checks and provenance.

## Include in a larger report

Load `amsmath`, `graphicx`, `booktabs` and `hyperref` in the parent preamble. From
the repository root, insert:

```tex
\newcommand{\ADRResultsPath}{docs/research/solver_studies/adr_scaling_2026_09_17}
\newcommand{\ADRResultsFigurePath}{run_outputs/solver_studies/adr_scaling_2026_09_17/figures}
\input{\ADRResultsPath/section.tex}
```

Keep the `oscillatory/` and `comparison/` subdirectories and table fragments
beside `section.tex`.
The fragment has one `\section`, no document class or preamble, and labels with
the `adr-` prefix. The portable ZIP supplies relative paths. For just the smooth
part, use [reference_section.tex](reference_section.tex).

## Standalone typesetting

The standalone [main.tex](main.tex) supplies a compact 10-point A4 layout.
Typesetting has **not been run**, in accordance with the workspace no-compilation
rule; the page count is unverified. To compile it yourself:

```bash
cd docs/research/solver_studies/adr_scaling_2026_09_17
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/typeset \
  main.tex
```

The figures are generated and visually checked without invoking TeX. The
[delivery audit](artifact_validation.json) also verifies local links, portable TeX
dependencies, archive integrity, and preservation of the earlier measurements.

## Regenerate from saved measurements

Rendering and auditing do not run solvers. Generate the oscillatory and focused
comparison subsections first, then the combined section and portable ZIP:

```bash
# from the repository root
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python -B scripts/reports/make_closed_loop_stress_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python scripts/reports/make_oscillatory_scaling_section.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python scripts/reports/make_adr_named_comparison_report.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python scripts/reports/make_adr_scaling_section.py \
  --campaign ~/src/hdgfem-gmres/run_logs/adr_scaling_20260917 \
  --output docs/research/solver_studies/adr_scaling_2026_09_17
```

The shared `hybridge.io.figures` helper supplies publication styling and vector/raster
export. Case-specific charts and discussion remain in the report scripts/templates.
See the [oscillatory reproduction instructions](oscillatory/README.md) for the GPU
sweeps and their independent audit. The original smooth campaign is preserved in
`hdgfem-gmres/run_logs/adr_scaling_20260917`; its solver measurements were not rerun
when the reports were merged. No native build or installation was performed.

## Concise ASM/BJ+PP synthesis

[section_synthesis.tex](section_synthesis.tex) is an insertable companion that
keeps ASM+PP, BJ+PP, the fastest passing block-AMG result and the best
passing native $hp$ policy in its main comparisons. [main_synthesis.tex](main_synthesis.tex) is the standalone wrapper.
The original `section.tex` and `main.tex` remain the detailed version.

To typeset the synthesis locally:

```bash
cd docs/research/solver_studies/adr_scaling_2026_09_17
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/typeset_synthesis \
  main_synthesis.tex
```

The portable archive is `run_outputs/solver_studies/adr_scaling_2026_09_17/adr_results_synthesis_bundle.zip`.

## Native hp-BSR completion — 21 September 2026

Both `main.tex` and `main_synthesis.tex` now include the completed native
comparison. All 101 distinct matrix/right-hand-side systems were attempted;
87 have a passing native policy. The extension ran 126 timing jobs (91 passing,
35 numerical failures) and six separate successful application profiles.
The 14 systems with no native convergence remain explicit failures.

Native standard and robust policies now appear throughout the detailed smooth
and oscillatory curves. The focused and concise figures select the best passing
native policy separately for each timing metric. Existing native measurements
retain their original campaign where available; repeated measurements from
another campaign are not selected merely because they are faster. Reused times
are consistently **means**, including the earlier directional probes and tuned
ASM+PP endpoints; fresh times are medians of paired setup-plus-first-solve times.
Both solves start from zero, so reuse means reusing setup, not the initial guess.

The [native evidence directory](native_completion/) contains the full coverage
census, saved measurements, failure outcomes, validation, and application
profiles. The original campaign JSON/CSV archives remain separate. The initial
oscillatory section and its standalone bundle also include native coverage for
all 17 initial systems; original AMGX/ASM measurements were preserved.

The previous directional conclusion based only on robust native is superseded:
standard native is substantially faster at both largest endpoints, including
the annulus. Both native policies still fail on the finest low-diffusion
oscillatory meshes and the original 22,825-triangle weak-anisotropic annulus.
The manuscript does not infer implementation optimality from these timings.

Regenerate using saved data only (no solvers or TeX):

```bash
cd ~/src/hybridge
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
.venv/bin/python scripts/reports/make_adr_native_completion_report.py
.venv/bin/python scripts/reports/make_oscillatory_adr_report.py
.venv/bin/python scripts/reports/make_oscillatory_scaling_section.py
.venv/bin/python scripts/reports/make_adr_named_comparison_report.py
.venv/bin/python scripts/reports/make_adr_synthesis_section.py
.venv/bin/python scripts/reports/make_adr_scaling_section.py \
  --campaign ../hdgfem-gmres/run_logs/adr_scaling_20260917 \
  --output docs/research/solver_studies/adr_scaling_2026_09_17
```

Figures and portable ZIPs were regenerated. **Existing typeset PDFs are stale:**
no TeX compilation was performed for this update. The synthesis retains the
maximum-eight-page layout target, but its updated page count awaits typesetting.
To refresh both PDFs yourself:

```bash
cd ~/src/hybridge/docs/research/solver_studies/adr_scaling_2026_09_17
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex main_synthesis.tex
```

## Solver notation and revised overviews

The synthesis now separates mesh refinement and degree sweeps. Every solver family
has an independent outcome and time, so an AMGX failure does not hide a passing
polynomial method. Fresh and reused selections are independent. Each passing
overview cell shows time in milliseconds followed by its mean outer iteration
count in parentheses; counts use the first or second solve as appropriate.
Polynomial labels include the recorded degree; the finite-element degree is
shown separately. Generic family labels are ASM+PP and BJ+PP.
ASM and BJ denote block additive Schwarz and block Jacobi. The hierarchy
is named pMG–AMG: face-block modal polynomial multigrid with AMGX on the
scalar degree-zero coarse system. It has no geometric mesh-coarsening stage.
The degree sweep states the square/annular triangle counts.
The two detailed timing/iteration overviews follow all discussion sections
as the final figures; explicit page breaks keep preceding floats ahead of them.
`synthesis/overview_cells.csv` records the plotted cells, including NC and
not-tested outcomes. Raw solver identifiers in archived data remain unchanged.

These sources and figures have been regenerated from saved measurements; no new
solver runs or TeX compilation were performed. Existing compiled manuscripts are
stale until rebuilt, and the eight-page target still requires typesetting verification.
