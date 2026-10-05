# Closed-loop stress section: preliminary results

[section.tex](section.tex) is included by both [main.tex](../main.tex) and
[main_synthesis.tex](../main_synthesis.tex). It defines the common main-level
geometry and exact solution, distinguishes trapping/crossing/orthogonal
transport and briefly explains the difficulty. Its shared
[results.tex](results.tex) reports the completed 22 September 2026 strong-preset
campaign: **8/60 passed**, all on the orthogonal control; **0/36 AMGX passes**.
The tables include all 60 final physical residuals and setup/fresh/reused timings
for the eight converged configurations. Failed warmup costs are not ranked.

Source records: campaign manifest (`run_outputs/solver_studies/adr_closed_loop_stress_strong_numba_h150k/manifest.json`, local, untracked)
and individual jobs (`run_outputs/solver_studies/adr_closed_loop_stress_strong_numba_h150k/jobs/`, local, untracked).
The report uses actual meshes of 98,699 and 148,848 triangles, PP degree 96,
restart 150, and a 2,000-iteration cap. Setup/fresh statistics are medians;
reused time is the mean second solve across three measured setups. The
large systems had no recorded direct reference, so the results remain
preliminary. Discussion is restricted to solver performance and accuracy,
with particular emphasis on ASM+PP.

Both documents include two shared figures:

- `geometry_fields`: ideal nine-lobed boundary, analytic close-up of the
  0.02-wide neck, and the same neck in the saved **98,699-triangle** campaign
  mesh. The NPZ is authenticated against `mesh.json`; no coordinates or
  connectivity are changed. Equal axis scale preserves the actual triangle
  shapes and nonuniform sizes.
- `operator_terms`: three columns for trapping/crossing/orthogonal and two rows
  for advection `beta · grad(u_star)` and diffusion `-div(K grad(u_star))`.
  The full tensor-derivative contribution is included. Each row shares signed
  symmetric-log color limits across cases, linear within [-1, 1]; no panel is
  independently normalized. Advection arrows and headless strong-diffusion
  axes have fixed length and communicate direction, not magnitude. The equal
  diffusion panels for trapping/crossing are intentional, since they share K.
  Frozen campaign velocity normalizations are read, never estimated anew.

The repeated `exact_solutions` panels are no longer included: all three cases
have exactly the same manufactured solution. Old exported assets are retained
on disk but are not evidence of different exact fields.

Neither figure certifies spatial or coefficient resolution.

Regenerate the figure assets alone, without meshing, assembly, solving, JIT or
TeX. By default the generator reads the completed strong campaign's
`meshes/main/100000/mesh.json`; `--mesh-record PATH` selects another saved
main-level record and never generates a replacement mesh. The sibling campaign
`normalizations/` directory is the default source of frozen velocity factors;
`--normalization-dir PATH` overrides it:

```bash
# from the repository root
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python -B scripts/reports/make_closed_loop_stress_figures.py
```

Outputs are PDF, SVG, PNG and analytic provenance (`run_outputs/solver_studies/adr_scaling_2026_09_17/figures/closed_loop_stress/`, local, untracked).
The package's publication styling/export and mesh-overlay helpers supply the
portable formats. The array-only `hybridge.io.figures.add_matplotlib_mesh` accepts
`node_coords`/`triangles` and optional `(xmin, xmax, ymin, ymax)` bounds; the
existing `hybridge.io.plot.add_matplotlib_mesh` delegates to it. Bounding-box
selection retains crossing triangles and never retriangulates holes. This
allows report rendering without importing numerical kernels. Source, figure
saved-mesh and normalization hashes, term extrema and shared color scales are
recorded in `figure_metadata.json`. The package's `add_direction_glyphs` helper
normalizes vector arrows or unoriented tensor-axis segments in physical
coordinates, omitting zero/nonfinite vectors.

Both wrappers and the synthesis-wrapper generator retain this shared section.
The bundle generators include its TeX and figures on their next invocation;
existing PDF documents and ZIP bundles are not refreshed by figure generation.
No LaTeX compilation was run for this addition, so pagination awaits the user's
build. To typeset both documents:

```bash
cd docs/research/solver_studies/adr_scaling_2026_09_17
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/typeset \
  main.tex main_synthesis.tex
```

Figure assets and their provenance are regenerated together. Existing linear
systems, campaign results and meshes are read-only inputs.
Static report tests check both document inclusions, balanced environments,
unique labels, saved-mesh authentication, crop connectivity, the common exact
field, term/source consistency, tensor axes, frozen normalizations, shared
color scales, all six image exports, and (when the saved campaign exists)
all residual/timing cells
against the original job JSON. No solver, JIT or TeX build is needed:

```bash
# from the repository root
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_closed_loop_stress_report.py
```

Preserve this campaign separately from the earlier 50k/100k, PP(48), restart-75
attempts and from subsequent sampling or stabilization changes. Failure to
converge does not establish singularity; a coarse direct-solver reference and
further residual and accuracy checks remain outstanding.
