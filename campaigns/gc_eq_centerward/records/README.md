# Farther inward equilibrium bands — 2026-10-15

The October 12 bands did move, but by only about 0.008 in mean wall clearance
on the unit-scale shapes. The change was too small to see clearly and did not
meet the requested placement. These new bands use substantially larger midpoint
increases, compared directly with that last family.

![October 12 above; new farther-inward equilibria below](inward_comparison.png)

| Geometry | Old → new midpoint | Old → new mean clearance | Old → new minimum clearance | Mean-clearance factor | Old → new mean width | Certified dual PDE residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Star | 0.0028 → 0.0045 | 0.05501 → 0.10240 | 0.04340 → 0.07311 | 1.86 | 0.03919 → 0.03958 (+0.995%) | 1.13e-15 |
| Ugly Pac-Man | 0.0025 → 0.004 | 0.05052 → 0.09805 | 0.02909 → 0.04847 | 1.94 | 0.03954 → 0.03965 (+0.27%) | 5.17e-15 |
| ITER | 0.0224 → 0.036 | 0.15126 → 0.26689 | 0.12130 → 0.21487 | 1.76 | 0.10943 → 0.10938 (-0.05%) | 8.80e-16 |
| Horseshoe | 0.0028 → 0.0045 | 0.05723 → 0.10534 | 0.04701 → 0.07917 | 1.84 | 0.03915 → 0.03919 (+0.10%) | 1.78e-15 |

Width and smoothing in potential coordinates are unchanged: delta=0.001 and
epsilon=0.00008 on star, Ugly Pac-Man and horseshoe; 0.008 and 0.00064 on ITER.
Physical widths are measured rather than assumed constant. The same affine
boundaries and fine meshes are used: 828,436 star, 613,674 Ugly Pac-Man,
714,164 ITER and 720,448 horseshoe triangles, with CG2 potential and P4 torsion.
The full-atlas cases additionally use CG3 recovered gradients. Mean/minimum
clearances and widths are torsion-flow arclengths for those three geometries,
and Euclidean contour distances for ITER. Compare each shape with itself.

All four solves used MPI/MUMPS on eight ranks, one numerical-library thread
per rank. Certificates record actual solver choice, rank affinity and numerical
thread counts; per-case execution JSON files record sampled parallel CPU use.
Each run first reconverged its archived source midpoint. Star, Ugly Pac-Man and
horseshoe then followed guarded midpoint continuation and passed the full
selected geometry checks; ITER used adaptive fixed-midpoint Newton continuation
and passed the PDE and three closed-contour checks. ITER has no auxiliary
single-center flow-atlas certificate. Source acceptance criteria were retained.

The initial star predictor was too large. Its PDE solve converged, but the
predictor-correction gate rejected that trial and halved the step. Smaller
accepted steps then reached the requested midpoint. Subsequent full-atlas
cases started with smaller numerical continuation steps. These changes affect
the continuation path, not epsilon, width, mesh or PDE acceptance tolerance.

The source states are independently archived as
`GEOMETRY_centerward_20261015_01`. The older families remain available. Saved
CG2 potentials and their frozen windows can be reused for HDG initialization
or later equilibrium refinement; the convergence certificate describes the CG
problem. **No new guiding-center drift tests were run for this family.** The
October 12 degree-6 drift measurements apply to their original, closer-to-wall
bands and must not be assigned to these new initial data.

## Files and reproduction

- [Comparison measurements](comparison.json), [density gallery](equilibrium_gallery.png),
  [catalog CSV](catalog.csv), and each geometry's potential/density PNG and portable checkpoint.
- [Reusable archive](../../equilibria/README.md) and [archive verification](archive_verification.json).
- [Producer sources and settings](producer_provenance.json), [runner tests](runner_tests.log).
- `launcher.py` is a detached, restartable sequential launcher; `driver.json`
  records the process group and stop command. Per-case completion files allow
  completed runs to be skipped. `completed.json` confirms all solves, plotting
  and archive verification. Intermediate guarded branch states remain in each
  run's `branch/` directory; only final requested states enter the archive.

The fixed-window Newton/PDE solve does not optimize Jaccard or threshold
functionals. Its residual and geometry checks should not be confused with the
separate criterion that optimizer losses first drop and then stabilize.
