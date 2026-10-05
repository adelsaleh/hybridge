# Equiband smooth-star test — 2026-09-08

This is a successful **discrete fixed-mesh smoke test**, not a completed
mesh/ray/quadrature convergence study or a proof of branch uniqueness.
The existing equiband solver was used unchanged, with a new
[star configuration](../../examples/equiband/star_logistic.toml).
The [example commands](../../examples/equiband/README.md#smooth-five-lobed-star)
reproduce the mesh and run.

Historical-input note (updated 2026-09-10): the maintained example now selects
threshold width 0.002, relative smoothing 0.01 and target distance 0.50, not this test's absolute
epsilon 0.001 and target 0.8. To reproduce the numerical results below, use
the configuration saved in the original run's `run.json` (or explicitly
restore absolute mode, epsilon 0.001, target 0.8, and omit `relative_epsilon`
in a separate input file). The mesh command remains unchanged; the maintained
relative-mode input is a different experiment.

## Geometry and fixed inputs

The simply connected, nonconvex five-lobed domain has polar boundary

\[
r(\theta)=1+0.2\sin(5\theta),\qquad 0\le\theta<2\pi.
\]

The existing `projects.diocotron.dolfinx.geometry.canonical` builder uses a
closed spline through 200 boundary samples, then generates first-order
triangular geometry. No new domain-specific source or equilibrium constraint
was introduced. Input mesh:
`projects/diocotron/runs/equiband/meshes/star_r1_a020_h003.msh`.

| Input/measurement | Value |
| --- | ---: |
| Requested Gmsh size | 0.03 |
| Measured maximum triangle edge | 0.04017646608 |
| Imported FE cells / CG2 DOFs | 8,357 / 16,970 |
| Discrete area | 3.20395663351 |
| Quadrature degree | 12 |
| Rays / trajectory sampling control | 128 / 300 |
| Threshold width delta | 0.003 |
| Absolute smoothing epsilon | 0.001 |
| Requested normalized distance | 0.8 |
| PDE / distance tolerances | 1e-9 / 1e-4 |
| MPI ranks / numerical-library threads | 1 / 1 |

The mesh SHA-256 is
`873a3548c95dbd8d1260e3a5239c21328dd8ffd5ef13a93ad5d242faa102f223`.
The accompanying `.msh.json` records shape parameters and generation settings.
FE counts above are from the imported domain: the generator's raw node count
also includes spline construction points that are not retained in the physical
mesh export.

The noncircular initializer was the existing constant-source homotopy at
midpoint 0.020 (`--seed homotopy --m-start 0.020`). Only its lambda=1 endpoint
was admitted as an equilibrium. Subsequent continuation used the ordinary
target-directed pseudo-arclength mode with unchanged default arc controls.

## Outcome and guard audit

The run ended with `TARGET_REACHED`, exit code 0, 24 committed checkpoints,
and elapsed time 31.926 seconds, including homotopy, terminal capture,
checkpointing, six off-screen PNGs and final VTK output. This timing is not a
performance guarantee and includes startup costs for this invocation.

| Final quantity | Value |
| --- | ---: |
| Solved midpoint | 0.011453031849358825 |
| Primary fixed-ray distance | 0.8000624182410174 |
| Absolute distance error | 6.2418241e-5 |
| Independently certified stiffness-dual PDE residual | 3.4770632e-16 |
| Minimum normalized transversality | 0.13995653788 |
| Core-hole margin, phi(x_T) - c_plus | 0.0041616825019 |
| Source mass | 0.42170414830 |

The torsion center was approximately `(1.2241e-7, 1.2631e-7)`, with
`T_max=0.2123175996`. All 128 rays passed the atlas checks. Recovered paths
had lengths between 0.8007163 and 1.1996878; their weights sum to one.

Re-reading all checkpoint arrays confirmed unique upper/middle/lower
crossings and negative outward slopes. All accepted states were marked
admissible, retained a positive inner-hole margin, and had a certified PDE
residual at most 1.12e-14. Reconstructing every primary distance from saved
crossings and fixed ray weights reproduced the recorded values exactly.

There were eight predictor-size rejections and one
`CONTOUR_AUDIT_UNRESOLVED` rejection. The latter was not accepted or bypassed:
the solver preserved the previous state, halved the step and proceeded through
admissible states. No fold crossing was needed for this outward target search.
Stability eigenvalues were not requested and remain `UNASSESSED`.

## The distance is an average, not a uniform offset

The primary convention remains

\[
\zeta_T(\gamma_j(s))=s/L_j,\qquad
D_T=\sum_j\omega_j\,s_{m,j}/L_j,\qquad
\phi(\gamma_j(s_{m,j}))=m.
\]

On the final star equilibrium, individual middle distances range from
0.7503645 to 0.8440445, with weighted standard deviation 0.0319780. This
variation is allowed: the target fixes the weighted mean, not every ray's
middle position. The contour-arclength audit is 0.8160830, using a different
averaging measure; it is not silently substituted for the primary distance.

Spatial thickness is also nonuniform: its derived mean is 0.0677484, with
ray values from 0.0535650 to 0.0770584. It remains a diagnostic. No physical
width equation, width/distance compatibility constraint, or change to delta
or epsilon was used to reach the target. `rho` is reserved for density W(phi).

## Output and reproduction

Run directory: `projects/diocotron/runs/equiband/star_interactive_h003`.

- Terminal transcript:
  `logs/20260908T161947.927216Z_f58b85bb/terminal_rank0000.log`.
- Final record and VTK field:
  `checkpoints/b9b42675f12c4a8181f0252d9e242b60/`.
- Final PNG:
  `frames/equiband_a9ea24f97534_00005_TARGET_REACHED.png`.
- Configuration, audit mesh, per-rank rays, branch ledger and search summary
  are retained alongside those files.

The final image was visually inspected. The same potential-defined interfaces
are overlaid on T, phi and rho=W(phi); they were not imposed as torsion levels.
Use the documented interactive command with `--restart` to inspect this saved
target without retracing the branch, or choose a fresh output directory to
repeat the solve. Host configuration/CLI/package-boundary checks passed (19
tests). The existing PETSc mixed-OpenMPI/MPICH warning is still emitted by the
installed environment; no installed packages or solver guards were changed.

Finer meshes, independent quadrature/ray refinement, and additional initial
branches are still needed before interpreting this as a converged scientific
star-domain study.

## Sharp-window follow-up — 2026-09-10

The later `star_m005` experiments are different physical inputs and must not
be compared as reruns of the September 8 result. For threshold width 0.001,
relative smoothing 0.01 (`epsilon=1e-5`) and `m=0.005`, the original affine
h=0.03, P2-torsion/CG1-recovery, quadrature-12 seed reached its 40-iteration
cap at homotopy lambda 0.95. With 80 iterations that stage converged at
iteration 42, but smaller rollback-safe homotopy steps then stagnated near
lambda 0.987. This is why a larger iteration cap alone is not sufficient.

The following controlled seed checks all retained CG2 for the equilibrium:

| Mesh / geometry | Torsion / recovery | Quadrature | Outcome at lambda=1 | Middle distance |
| --- | --- | ---: | --- | ---: |
| h=0.03 affine | P2 / CG1 | 24 | converged | 0.8747766901 |
| h=0.03 affine | P4 / CG3 | 12 | converged | 0.8747375080 |
| h=0.03 affine | P4 / CG3 | 24 | converged | 0.8747764436 |
| h=0.015 cubic | P4 / CG3 | 24 | converged | 0.8748067428 |

The maintained sharp-star example therefore uses P4/CG3, quadrature 24 and
80 nonlinear iterations for its current width-0.002 problem. In the separate
width-0.001 stress test, the coarse-to-fine distance difference is
`3.03e-5`, below the configured `1e-4` distance tolerance. This is a
two-level consistency check, not an asymptotic convergence result; higher
equilibrium degree remains unimplemented and must not be claimed as tested.

A separate run used threshold width 0.002, relative smoothing 0.01 and target
distance 0.5. It seeded successfully at `m=0.005`, `D_T=0.9147871555`, then
ended `TARGET_NOT_ATTAINED` after 208 committed checkpoints. This was an
`ARC_STEP_LIMIT`, not a PDE failure or branch-exit certificate. The explored
distance range was `[0.8758729144, 0.9147871555]`, the nearest recorded point
had `m=0.0069543164`, and its feasibility gap was `0.3758729144`. The branch
used only `0.0444404` of the configured arc-length budget 2.0 before the
200-step cap; 79 `BRANCH_JUMP_SUSPECT` rejections and one unresolved contour
audit forced conservative step reductions. The last accepted correctors took
only three or four Newton iterations, so increasing the Newton cap does not
address that termination. Extending this exact chart requires a matching
restart plus a larger `--arc-max-steps`; changing quadrature, mesh, torsion
order, width or smoothing instead requires a fresh run.

The upgraded maintained width-0.002 configuration (P4/CG3, quadrature 24,
80 iterations) independently reseeded at `m=0.005` with
`D_T=0.9147705607`. The difference from the old run's seed was `1.66e-5`.
This confirms the starting equilibrium but does not turn the old run's
step-limited result into a nonexistence statement.

A fresh, uninterrupted interactive chart with those upgraded controls
subsequently reached distance 0.50. The accepted target had

- `m=0.01844870452607754`;
- `D_T=0.49999037924452683`;
- distance error `9.6207555e-6`;
- PDE residual `6.95e-16`;
- minimum middle-crossing transversality `1.613e-2`.

The chart used 102 committed checkpoints, including its seed and final target
correction. It encountered 39 safely rolled-back `BRANCH_JUMP_SUSPECT`
trials, no detected fold, and placed the final target at cumulative arclength
`0.3521189`. Its full explored span was `0.3534699` because target correction
uses an accepted bracketing point beyond the target. A separate headless run
split across one restart recovered a midpoint within `1.45e-7` and a distance
within `5.96e-6` of this result.

Thus the old 208-checkpoint outcome was caused by conservative continuation
steps exhausting its accepted-step limit, not by an unattainable target on
the upgraded discrete branch. The last old correctors needed only three or
four Newton iterations, so raising their Newton budget would not have changed
that termination.

At the coarse target, the independent middle-contour arclength audit was
`0.5637745`, whereas the prescribed fixed boundary-arclength ray-label average
was `0.4999904`; the raywise distance variance was `0.0037940`. This is not a
change of target convention: it records the expected distinction between the
two averaging measures on a noncircular band. The derived mean physical
thickness was `0.0620369` and remained a diagnostic only.

The same target was then recomputed on a smaller cubic-geometry h=0.015 mesh,
with P4 torsion, CG3 recovery, CG2 equilibrium, quadrature degree 24, 128 ray
labels and four MPI ranks. The mesh contained 33,102 cells and 66,715
equilibrium degrees of freedom. The refined accepted state had

- `m=0.018442001804205965`;
- `D_T=0.4999993677761396`;
- distance error `6.3222386e-7`;
- PDE residual `7.02e-16`;
- minimum transversality `1.762e-2`;
- contour-audit distance `0.5639684`;
- derived mean physical thickness `0.0619227`.

The refined chart used 120 accepted checkpoints, 46 rollback-safe predictor
rejections, and no fold. The midpoint changed by `6.70e-6` from the coarse
interactive result; the contour audit changed by `1.94e-4` and the derived
mean thickness by `1.14e-4`. Both primary distance residuals satisfy the
configured `1e-4` tolerance. This is a two-level discretization check, not a
measured asymptotic convergence rate; an additional guard-clean level and ray
refinement remain appropriate for a publication-level error study.
