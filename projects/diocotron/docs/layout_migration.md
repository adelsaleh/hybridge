# Layout migration

The diocotron application is grouped by implementation under one scientific
project. HYBRIDGE's general library, packaging, tests, and development tools
remain at the repository root. Numerical algorithms and CLI options are
preserved; the Python module and file locations have changed.

## Entry points

Run these modules from the repository root:

| Previous runner | Current module after `python -m` |
| --- | --- |
| `scripts.diocotron_dolfinx.equiband` | `projects.diocotron.dolfinx.equiband` |
| `torsion_reduced_optimization_homotopy.py` | `projects.diocotron.dolfinx.torsion.optimization.homotopy` |
| `dolfinx_torsion_initialized_window_reduced_optimization.py` | `projects.diocotron.dolfinx.torsion.optimization.reduced` |
| `dolfinx_torsion_h1_projection_reduced_optimization.py` | `projects.diocotron.dolfinx.torsion.optimization.h1_projection` |
| `dolfinx_torsion_fractional_phi_target_window_reduced_optimization.py` | `projects.diocotron.dolfinx.torsion.initialization.fractional_target` |
| `dolfinx_torsion_frozen_frontier_window_reduced_optimization.py` | `projects.diocotron.dolfinx.torsion.initialization.frozen_frontier_run` |
| `dolfinx_torsion_adaptive_bisection_window_reduced_optimization.py` | `projects.diocotron.dolfinx.torsion.search.adaptive` |
| `dolfinx_torsion_bruteforce_window_reduced_optimization.py` | `projects.diocotron.dolfinx.torsion.search.brute_force` |
| `canonical_geometries.py` | `projects.diocotron.dolfinx.geometry.canonical` |
| `torsion_center_audit.py` | `projects.diocotron.dolfinx.geometry.center_audit` |
| `dolfinx_torsion_guiding_center_supg.py` | `projects.diocotron.dolfinx.guiding_center.supg` |
| `hdg_torsion_initialized_newton.py` | `projects.diocotron.hdg.equilibrium.newton` |
| `torsion_optimizer_numerical_tests.py` | `projects.diocotron.studies.torsion_optimizer.run` |

The former flat-script imports have been replaced with package imports.
Existing external command files should be updated using this table; the old
script tree is not kept as a second implementation. Shipped commands and
documentation use the new paths.

## Reports and artifacts

The article sources, selected figures, data, and manifest moved from
`docs/research/torsion_reduced_optimizer_article_study/` to
`projects/diocotron/studies/torsion_optimizer/report/`.
The build entry point sends PDFs and auxiliary files to `build/torsion_optimizer/`.
Existing root and report-directory build products were preserved separately
under `build/torsion_optimizer/previous_root/` and `previous_report/`.
Old bytecode/editor remnants and previous derivation builds are also retained
under `build/`, rather than mixed with maintained sources.

Raw equiband output moved from `output/equiband/` to `runs/equiband/`.
The optimizer archive moved from
`run_outputs/torsion_reduced_optimizer_numerical_tests/` to
`runs/torsion_optimizer/`. Older experiment logs live in `runs/legacy/`,
scratch experiments in `runs/scratch/`, and FreeFEM output in `runs/freefem/`.
The relative `runs/` and `build/` paths in this paragraph are project-local.

Historical logs, field files, provenance, hashes, and recorded commands are
preserved. [archive_paths.json](../archive_paths.json) records relocated
locations. The project path resolver and study-manifest loader translate old
paths in memory, without rewriting original numerical evidence. Existing
portable v2 checkpoint format identifiers are unchanged.

## Verification ownership

Root `tests/` retains the HYBRIDGE dependency and packaging contract checks.
Application tests live under `projects/diocotron/tests/`, grouped by backend,
studies, and comparisons. The DOLFINx checkpoint reader/writer can be imported
without HYBRIDGE; HDG projection is isolated under `comparisons/`.

The original standalone Newton experiment used a polygonal cosine-star mesh,
while the newer canonical generator uses a spline/sine-star. Its former
HYBRIDGE mesh dependency has been replaced by a local writer preserving the
original OCC construction and Gmsh sizing options, rather than changing the
experiment's geometry.

## Migration verification

The non-rendering application suite finished with 437 passed, 7 skipped, and
two failures. Both failures were also reproduced against the saved
pre-migration source, so they are not treated as layout regressions:

- `test_degree_matched_audit_on_curved_horseshoe` expects curved geometry to
  be rejected, but the current implementation accepts it.
- `test_horseshoe_band_screen_is_isolated_deterministic_mumps_pilot` expects
  `--save-trajectory` to be absent, but the current study configuration includes it.

Separate checks passed for headless rendering (2 tests), opt-in MPI regressions
(6 tests), and the optimizer-to-guiding-center checkpoint handoff plus affected
figure consumers (17 tests). The handoff test explicitly disables interactive
plotting; rendering is covered separately. Native guiding-center, packaging,
documentation, dependency-boundary, and project-layout checks also passed
(56 tests). The extracted star-mesh writer reproduces the original mesh, and
the DOLFINx-to-HDG field-import check reported a maximum error of `5.640e-14`.

Module and direct-script CLI entry points were checked. The numerical article
was rebuilt successfully with:

```bash
python -m projects.diocotron.studies.torsion_optimizer.build_report
```

The output is `projects/diocotron/build/torsion_optimizer/numerical_tests.pdf`.
FreeFEM source paths and output links were reorganized, but its solvers were
not executed as part of this migration.
