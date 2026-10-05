# Equiband implementation checks — 2026-09-07

These are local development checks of the new
[fixed-threshold module](../../docs/equiband.md), not a production
performance guarantee or a completed ITER study.

Historical-input note (2026-09-09): the maintained example configurations
now use relative smoothing 0.08. Results and commands recorded here used
the absolute epsilon values stated for each test. Reproduction/restart
requires those original inputs, not the changed examples; the configuration
is preserved in each run's `run.json`.

## Environment

The exercised `fenicsx-dgfem` environment reported DOLFINx 0.11.0,
PETSc/petsc4py 3.25.3, Open MPI 5.0.10, mpi4py 4.1.2, NumPy 2.4.6,
Numba 0.65.1, SciPy 1.18.0, UFL 2026.1.0 and Gmsh 4.15.2.
Numerical-library threads were limited to one.

Some runs emitted a PETSc mixed-OpenMPI/MPICH diagnostic from the existing
environment, even though the reported active MPI library was Open MPI and
the one-/two-rank results below agreed. This is **not** evidence that mixing
MPI builds is safe. The runtime/library paths require a clean-environment
audit before large production runs. PETSc explicitly checks for incompatible
MPI implementations in its
[initialization code](https://petsc.org/release/src/sys/objects/pinit.c.html).
No installed environment packages were modified for this work.

The initial implementation suite passed 27 checks, including the optional MPI
subprocess regression and local documentation links. A separate host run
also exercised the repository packaging contract: its package-directory
scan flags pre-existing `hybridge/solvers/.cache/hybridge/meshes` directories
dated July 23, 2026. Those legacy cache files were not changed or deleted.
The base runtime dependency set remains NumPy, Numba and SciPy. Equiband
now lives in `projects/diocotron/dolfinx/equiband`, outside the installed
library; its former installation extra has been removed. The commands
below use the script-side entry point.

After the source relocation, 61 host-side checks and 30 FEniCSx/MPI checks
passed. These include the new package-boundary regression, CLI discovery
without HYBRIDGE or DOLFINx, native guiding-center checkpoint consumers,
one-/two-rank target agreement and a new-process restart. The boundary
regression is included in the repository's `host-fast` lane. A further 15
test-matrix/packaging checks passed; the known cache-directory scan above
was excluded from that targeted run.

The relocated `dolfinx_checkpoint.py` / `dolfinx_equilibrium.py` adapters
also passed real scalar-field and equilibrium round trips on one and two
MPI ranks using `projects/diocotron/comparisons/check_field_import.py`. Both runs
recovered the polynomial fields with maximum error about
\(5.64\times10^{-14}\), and retained the explicit nonconverged-equilibrium
import policy. The move does not change the v2 checkpoint schemas.

## Disk target and MPI comparison

Interactive CLI follow-up: the default is now live plotting at `-v 2`.
The expanded targeted suite passed 45 tests, covering accepted-state plot
sampling, in-place arrays/cameras, live-event and Enter handling on an
off-screen VTK backend, window-close cleanup, one-/two-rank CLI behavior,
and collective recovery after an injected rank-zero render failure.
Actual plots were visually inspected for annotations and moving interfaces.
The explicit disk CLI run with `--plot-off-screen --save-frames -v 2`
retained the target values below. Rendering used PyVista 0.48.4 / VTK 9.6.2;
these checks do not measure production GUI latency. The existing mixed-MPI
warning described above still appeared in that environment.

Configuration: `projects/diocotron/examples/equiband/disk_logistic.toml`, unit disk,
affine mesh size 0.10, CG2, quadrature degree 12, 64 rays,
`samples_per_ray=200`, \(\delta_\star=0.02\),
\(\epsilon_\star=0.002\), \(d_\star=0.60\), scan endpoint \(m=0.060\).

| Execution | Recovered midpoint | Attained normalized distance |
| --- | ---: | ---: |
| One rank | 0.057666004327844904 | 0.59999970788437 |
| Two ranks | 0.05766600432784538 | 0.59999970788437 |

The target error was approximately \(2.92\times10^{-7}\).
This is the **discrete target error**, not the error relative to the
continuum problem. At the initial midpoint \(0.05362116975655636\),
the independent radial reference gave distance \(0.6408404057\);
the 2D calculation gave \(0.6403268051\).

Separate integration tests check the radial comparison at mesh sizes 0.15
and 0.075, the exact midpoint derivative against centered differences,
both smooth source families, genuine SNES rollback, checkpoint reload,
VTK output and optional SLEPc Hessian labels.

The MPI restart check was performed in a new process. A completed scan and
target were loaded and audited without new accepted nonlinear solves.
The restart implementation requires identical mesh/configuration and MPI
partition; this does not validate changing rank count on restart.

## Terminal logging and output-directory reuse

The logging follow-up adds `--save-terminal-log` and the existing-directory
fresh/resume/cancel prompt. It does not alter nonlinear solves, branch guards
or target-search direction. Fresh reuse archives the old run to a sibling
backup; explicit `--overwrite-output` and `--restart` support unattended jobs.
The three existing `disk_interactive*` runs were not modified during testing.

Host regressions exercise early import capture, Python and native descriptors,
buffered C stdout, normal interpreter finalizers, tracebacks, SIGINT/SIGTERM/
SIGHUP markers, log setup failure, output reservations, resume, cancellation,
EOF and byte-preserving archival. All 60 targeted host checks passed.

The FEniCSx/MPI/off-screen suite passed 13 checks in about 68 seconds, using
small temporary runs (not the user-edited disk example configuration).
The MPI disk regression uses mesh size 0.15, 16 rays, delta 0.02, epsilon
0.002, target 0.60 and scan endpoint 0.060. One and two ranks agree to the
test's `1e-10` tolerances; the two-rank target distance is approximately
0.599999336488. The final checkpoint count includes all eight committed
states, including scalar target substeps.

The same test resumes via MPI-forwarded `r` input without changing old logs
or the checkpoint ledger, and verifies that `--overwrite-output` preserves
the previous complete run in a backup. Separate two-rank tests check that
native stderr is retained only in its originating rank's transcript, and
that a failed rank-one log open terminates collectively without a hang.

Two broader documentation-tree tests remain unrelated failures: existing
extra top-level documentation directories and legacy algorithm directories
do not match their prescribed directory sets. Logging work does not remove
or reorganize those research files.

## Fold continuation and the two later interactive runs

The later transcripts examined were:

- `disk_interactive/logs/20260907T193609.600190Z_a7a3c691/terminal_rank0000.log`;
- `disk_interactive1/logs/20260907T194541.756701Z_9e6ed4f5/terminal_rank0000.log`.

Both are under `projects/diocotron/runs/equiband`. Their recorded inputs, not today's editable
example defaults, were used for the audit: unit disk, mesh size 0.02, CG2,
18,367 cells, 37,050 DOFs, quadrature degree 12, 64 rays, delta 0.003,
epsilon 0.001 and one rank/thread. The first requested distance 0.1; the
second requested **0.8**. Both increased the midpoint toward 0.060 from
the same radial seed and committed the same 21 physical states:

| Quantity | Seed | Last accepted state |
| --- | ---: | ---: |
| Midpoint | 0.0248773791586 | 0.0271106330635 |
| Fixed-ray normalized distance | 0.531530093359 | 0.375607792630 |
| Midpoint distance derivative | -36.9767 | approximately -8337.95 |

All stored states passed their PDE and two-interface tests. Each invocation
had 13 failed Newton trials and 14 predictor-guard rejections. The stop was
an exhausted regular midpoint chart near a fold, not numerical collapse of
the last accepted ring. For the 0.8 target, that scan also went in the wrong
direction: the nearest saved state was the seed, with gap 0.268469906641.
Output archival in the second invocation worked as intended; the prior run
is retained in its session-named sibling backup.

An independent radial shooting audit writes \(u=\phi-m\), prescribes
\(u(0)=a\), integrates
\(u''+u'/r=-W(u;0,\delta_\star)\), and recovers \(m=-u(1)\).
The middle distance is the first zero of \(u\). For these fixed inputs:

- the radial midpoint fold is near \(m=0.0271116216\), \(D_T=0.375022411\);
- the inner hole closes at \(a=\delta_\star/2\), near
  \(m=0.0180743278\), \(D_T=0.107925166\);
- a radial middle distance 0.1 requires \(a<\delta_\star/2\), so it has
  no upper-interface hole and is not an admissible two-interface band.

Direct in-memory 2D checks confirmed this distinction: the radial 0.1 seed
corrected to a PDE equilibrium at distance 0.10000714 but failed
`NO_TWO_INTERFACE_BAND`; the 0.12 seed corrected to an admissible ring at
0.12000830. These are findings about this explored family, **not** a
universal threshold-width/distance compatibility formula.

The new bordered solver was then exercised at the original fine resolution,
loading the old checkpoints read-only and continuing in memory:

| Target | New outcome | Midpoint | Distance |
| --- | --- | ---: | ---: |
| 0.8 | `TARGET_REACHED`, outward from the old seed | 0.0130320571404 | 0.800014410994 |
| 0.1 | Passed one fold, stopped at `NO_TWO_INTERFACE_BAND` | 0.0180740015744 | 0.107927622517 |

The remaining gap for 0.1 is 0.007927622517. The trace kept every accepted
state admissible and never changed delta, epsilon, ray weights or distance
convention. The largest accepted dual residual was approximately
\(1.20\times10^{-11}\) for the 0.8 search and \(6.43\times10^{-12}\)
for the 0.1 continuation, below the configured \(10^{-9}\) tolerance.
The two in-memory continuations took about 33.3 and 42.1 seconds, respectively,
excluding common setup, checkpoint writes and plotting.

A complete fresh CLI run in a **new temporary directory**, with full terminal
capture, seed/final off-screen PNGs and VTK export, reproduced the 0.8 target:
28 committed checkpoints, 10 safely rejected predictors, final PDE residual
\(7.24\times10^{-13}\), distance error \(1.45\times10^{-5}\), and
`RUN_END status=TARGET_REACHED exit_code=0`. Its elapsed time was 41.805 seconds.
This is not an apples-to-apples speedup over the old runs: the search path and
termination condition changed, and no interactive final pause was included.
Neither original interactive output directory was modified by these checks.

Regression coverage includes a controlled algebraic fold, a real disk fold
to distance 0.12, full bordered-Jacobian finite differences (including
Dirichlet rows), a guard-limited continuation toward 0.1, parent-secant
restart, equal-midpoint arclength brackets, tangential targets, finite arc
budgets, and refusal to reinterpret an arclength history as a midpoint chart.
One and two MPI ranks agree through the disk fold to the tests' `1e-10`
midpoint/distance tolerances. This does not validate arbitrary bifurcations
or exhaustive branch/root discovery.

The final targeted pass completed 79 host/documentation-link checks and 18
actual FEniCSx/MPI/off-screen checks. Per-invocation arclength controls and
search stop reasons are saved in versioned summaries even when terminal-log
capture is disabled. The pre-existing mixed-MPI environment warning and
unrelated documentation-tree issues noted above were not hidden or repaired.

### Monitor and plotting fixes

The earlier logs had no Newton iteration callbacks after the first failed
solve, despite maximum verbosity. A controlled PETSc failure reproduced the
problem: petsc4py's cached monitor list was nonempty, but the native callback
no longer fired. Cancel-and-reinstall before each solve restores exactly one
callback per iteration. Tests cover both ordinary and augmented SNES recovery.

The earlier 0.1 run spent 157.628 seconds in 22 plot updates (median 7.028
seconds). Profiling an initialized live-update window forced off-screen
showed repeated Matplotlib/VTK text layout: about 95,676 font lookups and
1,876 math-text constructions per update. This was not a ray-kernel cost.

After fixed-size plain-text annotations, scoped FreeType rendering and
contour-actor reuse, the same 18,367-cell, 293,872-display-triangle setup
reported 0.399 seconds for the first initialized-window update and
0.093, 0.085 and 0.081 seconds for three subsequent updates. The profiled
whole emission took 0.114 seconds including coefficient exchange, with no
repeated font-fitting hotspot. The unchanged full-resolution panels were
visually inspected. These are local off-screen measurements, not a guarantee
for every desktop/driver. PNG encoding and interactive waiting are separate
costs. A root-decided live cadence now skips unnecessary field gathers;
forced/final, explicitly paused and saved-frame updates remain honored.

## Warm crossing kernel

Command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m projects.diocotron.dolfinx.equiband.benchmark \
  --rays 512 --segments 400 --threads 1 4
```

The benchmark checks numerical agreement before measuring ten warm calls
on 204,800 quadratic segments and three thresholds:

| Backend | Median |
| --- | ---: |
| Vectorized NumPy | 71.5 ms |
| Numba, one thread | 3.29 ms |
| Numba, four threads | 1.00 ms |

JIT compilation, coefficient extraction, ray construction, MPI transfers,
PDE solves, topology audits and output are **excluded**. These numbers
must not be represented as end-to-end solver speedups.

## Noncircular geometry

The documented ellipse scan, with minor/major axis ratio 0.7 and mesh size
0.08, ran from \(m=0.035\) to \(0.040\) using the source homotopy.
Its primary distance changed from approximately \(0.73568\) to \(0.68681\);
the ring and whole-mesh contour guards passed. This was a branch scan,
not a claim that the configured target 0.60 was attained in that interval.

A coarse tagged ITER mesh generated from `projects/diocotron/freefem/msh/iter.geo` with size 0.30
was successfully imported, but its P2 torsion audit reported unresolved
secondary critical candidates. The single-center atlas was correctly
refused before a target solve. This is not a continuum nonexistence
result and does not justify bypassing the guard: critical-point/refinement
analysis is still required for that geometry.
