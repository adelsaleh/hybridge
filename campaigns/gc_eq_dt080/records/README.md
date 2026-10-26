# Equilibrium bands at the mode-128 torsion-flow distance

The requested placement is **D_T = 0.8** on star, Ugly Pac-Man, ITER and
horseshoe. The article's mode-128 Gaussian ring is centered at r = 0.8 in
the unit disk (sections/06_diocotron_disk.tex). Its width parameter is
0.0025. This study matches its placement, while keeping the existing
localized smooth-window equilibrium family on each general geometry.

Distance uses the established definition zeta_T = s/L along a torsion-flow
trajectory, from the torsion center to the wall. D_T is the average of the
middle-level crossings with fixed boundary-arclength seed weights. Thus
the normalized remaining distance to the wall is 1 - D_T = 0.2. No
equal-area normalization or Euclidean distance target is used. Individual
ray coordinates and physical wall gaps may vary on a noncircular domain.

## Computation and current status

The active computation is the `high_order_launcher.py` campaign. It runs
CG4 followed by CG6 for each geometry, warm-starting CG4 from an accepted CG2
archive and CG6 from its CG4 result. The user clarified that high degree and
up to roughly 300,000 triangles apply to **DOLFINx**, and explicitly requested
that no HDG preservation tests be run.

The accepted placement meshes have 211,768 (star), 158,358 (Ugly Pac-Man),
182,558 (ITER), and 185,876 (horseshoe) triangles. Their affine walls and
P4 torsion/CG3 recovery remain fixed for the first degree comparison. The
high-order nonlinear forms use quadrature degree 32. All solves use PETSc
MUMPS with eight MPI ranks and one numerical-library thread per rank.
Delta and epsilon remain 0.001 and 0.00008 on the unit-scale geometries,
and 0.008 and 0.00064 on ITER. The new target-fit tolerance is 2e-7;
the normalized dual PDE residual tolerance remains 1e-11.

Per-case `completed.json` files identify accepted archived fields;
`failed.json` files retain rejected attempts. The final driver marker is
`high_order.completed.json`; inspect its status and failure list together.
`cg4/` and `cg6/` contain the new degree-refinement states. Offline comparisons
integrate the complete potential, its gradient and `W(phi)` with 8 and 12
Gauss points per Duffy coordinate. These comparisons, not the algebraic
residual alone, determine the spatial-accuracy claim.

All four earlier CG2 placement solves passed, with target errors below 5e-8.
Their [density preview](placement/density_bands.png) shows the intermediate
fields. [RESULTS.md](RESULTS.md) and `summary.json` collect accepted states
and available refinement checks. The earlier CG2 refinement completed star
(823,806 triangles), Ugly Pac-Man (732,484) and horseshoe (876,284).
The planned still-larger CG2 meshes were cancelled in favor of degree 6
and the requested mesh range; `refinement_replanned.json` records that stop.
No `ultra/` state was produced.

The refined ITER CG2 attempt was rejected before its equilibrium solve:
neighboring flow labels approached below the numerical separation threshold
about 1.94e-11, leaving a resolved torsion fraction of 0.116 rather than the
required 0.5. Its earlier accepted placement atlas remains available. The
higher-order solves first retain that known admissible placement mesh;
raising the equilibrium degree does not itself refine the torsion atlas.

The archived potentials and all degree-dependent crossing, topology,
comparison-principle and contour-export checks now retain the complete
polynomial. Degree-4/6 topology uses enclosing Bernstein bounds and edge
sign-variation checks. These are numerical guards; they do not establish
mesh convergence. The coarse sharp-transition smoke case was correctly
rejected by its upper-contour guard. A resolved, smoother disk passed the
CG4/CG6 two-rank MUMPS solve, distance derivative and export tests.

No guiding-center evolution is performed for this family. Independent
boundary-label and center-stop-radius checks remain separate from the target
fit on the fixed production atlas; a failed finer atlas cannot be described
as a certified sampling check.

## Additional degree-6 mesh check

`high_order_refinement_launcher.py` waits for the primary degree comparison,
retries unfinished cases, and warm-starts CG6 on meshes graded around both
transition contours and sharp corners. It skips accepted states on resume,
rejects generated meshes above 300,000 triangles, and writes its own driver
and completion markers. Requested unit-scale sizes are 0.006 in the bulk,
0.0025 in the transition strips (buffer 0.003), and 0.00001 at sharp corners.
ITER uses bulk size 0.02 and three times the other lengths. The first such
Ugly Pac-Man mesh has 215,246 triangles. These checks are motivated by its
larger CG4-to-CG6 gradient change; the degree comparison alone is not used as
an absolute error estimate for the CG6 solution.

`high_order_checks.py` performs saved-field trajectory checks, final rendering
and archive verification after refinement. It selects an accepted refined
CG6 field when available, otherwise retains the primary CG6 field. The
selection is recorded explicitly in `selected.json`; `final/` contains only
relative links to those accepted case directories. No PDE solve or guiding-
center integration is launched by that postprocessor.

## ITER and critical-point accuracy

The first ITER attempt stopped on a purported secondary torsion critical
point about 1e-11 from a zero-Dirichlet corner. A small analytic test exposed
scale-dependent root accuracy: the root finder had used an absolute vector
residual despite very small gradients in refined corner cells. Polynomial
vectors are now normalized cellwise before the Newton root search. Scaled
Dirichlet-bubble tests pass, and genuine near-corner interior roots remain
rejected. The atlas schema is incremented to v5; prior archives are intact.
ITER also needed zero recovered gradient at its four acute convex vertices.
For a differentiable zero-Dirichlet field, the gradient is normal to both
incident wall tangents and therefore vanishes at such a vertex. Reentrant
vertices are excluded because their gradients can be singular. The optional
90-degree convex-angle cutoff changes gradient recovery, not the equilibrium
equation. With CG3 recovery and trajectory tolerance 1e-14, the full flow
atlas resolves the torsion maximum and the placement solve reaches
D_T = 0.79999999894342. Its previous PDE/contour-only certificates did not
certify D_T; the new placement certificate includes the flow checks.

The first CG4 ITER attempt also failed the trajectory-separation guard after
changing shared quadrature from 24 to 32, despite keeping the same mesh.
The solver now supports optional `auxiliary_quadrature_degree`; setting it
to 24 keeps the mathematically exact P4 torsion/CG3 recovery integration
fixed while the nonlinear forms use degree 32. The CG4 retry passed all
checks and reached D_T=0.799999999999895. This avoids an unrelated rounding
change in the tightly focused auxiliary flows; no separation guard is relaxed.
The rejected attempt remains under `cg4/iter_auxiliary32_attempt_*`.

Ugly Pac-Man's first trajectory tolerance 1e-11 limited the resolved core
and stopped continuation near D_T = 0.8644. A saved-field audit justified
retrying at 1e-13, while keeping the same acceptance guards. Horseshoe
midpoint continuation developed large sensitivity and shrinking steps near
D_T = 0.82; its saved branch is continued with the existing pseudo-arclength
corrector to allow a possible fold. The accepted horseshoe placement reached D_T=0.8
without crossing a fold. Failed attempts remain in this directory.

One Ugly Pac-Man trial put the contour within tolerance of a sampled vertex.
The topology guard correctly rejected it, but first refined every mesh cell
through subdivision factor 16. The audit now selects candidate cells using
exact CG2 extrema before building refined connectivity. It still detects
hidden loops, near-level vertices, open contours and extra components.
Comparisons against the previous whole-mesh calculation preserve statuses
and diagnostics, including a loop hidden inside one original triangle.

## Reproduction and process control

Producer: `hybridge-diocotron`, branch `diocotron-dolfinx`. Runner:
`projects.diocotron.studies.equilibrium_catalog.run_distance`. The reusable
archive is `../../equilibria`; accepted portable checkpoints retain the
complete potential and frozen density law for future HDG import.

The launcher is detached and skips completed cases. Each phase writes its
PID, process group and stop command to `<phase>.driver.json`. Each run
writes its full command, measured parallel CPU use, memory high-water mark
and exit code to an execution record. Source snapshots and SHA256 records
accompany jobs launched after the initial placement attempt.

Execution records containing `monitored_argument` measure the named case.
Earlier records sampled every active study worker; when two jobs overlapped,
their CPU and memory figures are combined and their active-process count is
not an MPI-rank count. The per-case `parallel_runtime` records give the actual
eight ranks, affinity, and numerical-library thread counts.

From this directory:

```sh
setsid nohup /usr/bin/python3 high_order_launcher.py > high_order_driver_stdout.log 2>&1 < /dev/null &
```

Do not start a second copy of an active campaign; a process lock also prevents
this. `high_order.driver.json` records the process group and stop command.
The old `launcher.py refinement` is retained for provenance and should not
be resumed: it queues the superseded large CG2 meshes. The current launcher
performs DOLFINx equilibrium solves and offline comparisons/figures only.

`summarize.py` refreshes `RESULTS.md` and `summary.json` from accepted files
and saved checks, without a solve. Run it with the DOLFINx environment's
Python, which supplies NumPy.
