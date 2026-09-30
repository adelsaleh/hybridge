# Fixed-Mesh H0-1 Torsion Projection Optimizer

`projects/diocotron/dolfinx/torsion/optimization/h1_projection.py`
tracks one locally selected semilinear equilibrium branch and minimizes

```text
J = 0.5 * ||grad(phi-phi_T)||^2 / E_T,
E_T = ||grad(phi_T)||^2.
```

The reference `phi_T` is the Poisson potential of the **sharp** torsion-band
indicator. Normalized leakage `L` and missing area `M` are hard admissibility
constraints. They are never added to `J` as weighted penalties. The initial
implementation keeps the mesh fixed.

Every initializer candidate, trial, accepted state, finite-difference state,
and final state uses damped Newton with the same requested
`--newton-tol`, measured as `sqrt(R^T K^-1 R)`. There is no inexact-Newton
option in this mode. A sensitivity predictor only supplies the Newton initial
iterate; it never replaces strict nonlinear projection.

## Required scientific bounds

All six bounds are required so a run cannot silently inherit a different
scientific admissible set:

- `--center-min`, `--center-max`: box for the window center `m`.
- `--width-min`, `--width-max`: box for the positive window width `d`.
- `--leakage-max`: upper bound for normalized leakage `L`.
- `--missing-max`: upper bound for normalized missing area `M`.

The thresholds and smoothing width are always
`c1=m-d/2`, `c2=m+d/2`, and `epsilon=--eps-ratio*d`. No additional potential
range bounds are imposed on `c1` or `c2`.

## Mesh, design, and solver controls

- `--mesh` loads a fixed Gmsh mesh. Without it, `--mesh-size`, `--star-n`,
  `--star-r0`, `--star-amp`, `--star-mode`, `--gmsh-verbosity`, and
  `--gmsh-algorithm` define the generated smooth star.
- `--order` selects the continuous Lagrange degree. `--quad-degree` overrides
  the default `max(2*order+8,12)` quadrature degree.
- `--alpha-t1`, `--alpha-t2`, and `--rho-amp` define the sharp torsion target;
  `--eps-ratio` defaults to `0.08`.
- `--linear-solver`, `--ksp-type`, `--linear-rtol`, `--linear-atol`,
  `--linear-max-it`, and `--iterative-fallback-solver` configure PETSc. PDE
  state and sensitivity solves remain distributed; only the two-variable
  algebraic subproblem is solved on rank zero and broadcast.

## Strict Newton controls

- `--newton-tol` (default `1e-10`) is the sole production nonlinear tolerance.
- `--max-newton-it` bounds damped Newton iterations.
- `--tol-step`, `--beta-ls`, `--armijo-c`, `--alpha-min`, and
  `--max-backtrack` control the existing residual line search.

A trial is rejected unless Newton reports convergence at `--newton-tol`.
There is no relaxed-residual acceptance factor.
`--trial-max-newton-it` may stop an unproductive outer corrector early, but
that trial is rejected unless it has already met the same `--newton-tol`.

## Initializer controls

The initializer has one three-stage strategy:

1. an exact discrete, sorted hard-window scan of `phi_T` quadrature samples;
2. an SLSQP refinement of frozen smooth missing area subject to frozen
   leakage;
3. a frozen topology check/repair, followed by strict projection of a
   shortlist of one to five nearby pairs.

`--init-leakage` is the stricter frozen leakage budget (default
`0.8*leakage-max`). `--init-refine-max-it` and `--init-refine-ftol` control the
scalar refinement. `--init-shortlist`, `--init-width-perturbation`, and
`--init-center-perturbation` control the shortlist.
The frozen leakage optimum can already consist of several disconnected
activity lobes even when the sharp target is connected. With
`--init-topology-repair` (the default), the initializer detects this before
any nonlinear solve. It first expands the two interval edges in a
scan and then searches the full bounded center--width grid, including
translations and contractions. Correct component count and
`--init-min-coverage` are hard filters. The remaining candidates are ranked
first by their total positive `L`/`M` violation, then by higher target
coverage and finally by threshold-edge displacement from the frozen optimum.
This geometrically ranked primary pair and its direct local shortlist remain
unchanged by the continuation fallback.

If every direct projection is unusable, a separate diversified homotopy
shortlist is activated. It retains the primary geometric pair and then draws
distinct candidates from a clean nearest outward enclosure, a global
topology-continuation ranking, the frozen Jaccard ranking used by the earlier
frontier initializer, and target coverage. The continuation ranking favors
the expected significant component count, no additional raw fragments,
little unexpected core mass, and noncollapsed coverage before geometric
violation. Thus an initially leaky but clean enclosing interval may serve as
a continuation anchor without replacing the geometrically ranked primary
initializer. `--init-topology-repair-samples` controls both scalar scans and
`--init-shortlist` remains the total fallback-anchor cap.
`--init-candidate-max-newton-it` limits direct screening work, but a capped
candidate is rejected unless it already satisfies the unchanged strict
`--newton-tol`; the cap does not enable inexact acceptance. If all candidates fail,
`--init-shrink-factor` and `--init-fallback-stages` control strict width
shrinkage and continuation back toward the frozen pair.

If direct projection stalls or reaches the exact zero equilibrium,
`--init-homotopy-fallback` starts at the refined frozen pair and alternates
source continuation with small center--width SQP/restoration updates. After
every accepted nonzero source stage, at least one threshold micro-step is
attempted using the partial Jacobian
`K-lambda*rho_amp*W_s`, the correctly scaled threshold right-hand sides, and
the `J/lambda^2` model. If
`--init-homotopy-threshold-steps-per-stage` exceeds one, its remaining
micro-steps continue only while the active reduced merit makes meaningful
progress. While geometry is infeasible this merit is
`max(0,L-Lmax)+max(0,M-Mmax)`; after feasibility it is the scaled projection
objective. `--init-homotopy-threshold-stagnation-atol` and
`--init-homotopy-threshold-stagnation-rtol` define meaningful progress.
Consequently, an unattainable intermediate leakage value cannot force
arbitrarily many threshold corrections. A failed micro-step at the minimum
threshold trust radius also counts as stagnation. Failed-source restoration
is skipped at the same accepted lambda after stagnation, and the learned
threshold trust radius persists across source stages rather than being reset
to a large value.
The two threshold edges may move by at most
`--init-homotopy-threshold-edge-step-fraction` times the current width in one
micro-step, so a narrow band cannot jump merely because the global parameter
box is wide. Every accepted source or threshold state satisfies the same
`--newton-tol`, the symmetric density-continuity gate, and the discrete
topology gate; this is a globalization path, not inexact Newton. If a source
corrector fails, or if its exact corrected activity fails either branch gate,
one configurable threshold restoration is attempted at the last exact state
before the source increment is shortened. A topology-rejected source step
bypasses ordinary leakage restoration, because narrowing the accepted window
can reduce rather than improve its connectivity margin. The source tangent is
then recomputed at the updated pair.

`--init-homotopy-easy-newton-it` controls when a fully converged, undamped
strict corrector is inexpensive enough to grow the next source increment. Its
default is six iterations. This is only continuation-step scheduling: every
stage still uses the unchanged `--newton-tol`, so it does not enable inexact
Newton acceptance.

`--init-homotopy-threshold-steps-per-stage`,
`--init-homotopy-threshold-max-trials`, the three
`--init-homotopy-threshold-trust-*` options, and
`--init-homotopy-threshold-edge-step-fraction` and
`--init-homotopy-threshold-rescue-attempts` control this alternating step.
`--init-min-coverage` still makes a collapsed zero-density equilibrium
ineligible. Fixed-center geometric width shrinkage is retained only when no
frozen topology repair was needed. After a topology repair, the ranked
topology-compatible anchors replace that collapse-prone fallback.

## Trust-region, filter, and KKT controls

- `--max-opt-it` and `--max-trials-per-iteration` bound outer work.
- `--trust-radius`, `--trust-radius-min`, `--trust-radius-max`,
  `--trust-shrink`, and `--trust-grow` define the scaled infinity-norm trust
  region.
- `--acceptance-eta` and `--acceptance-grow-eta` test actual versus predicted
  objective reduction at feasible points.
- `--gn-regularization` regularizes the positive-semidefinite Gauss--Newton
  matrix.
- `--restoration-fraction`, `--filter-objective-margin`, and
  `--filter-violation-margin` control infeasible funnel restoration. Strictly
  improving infeasible iterates may be accepted until feasibility is reached;
  a feasible branch iterate is never allowed to become infeasible.
- `--functional-stagnation-atol`, `--functional-stagnation-rtol`, and
  `--functional-stagnation-patience` stop the outer iteration when its active
  merit ceases to improve. The active merit is normalized geometric violation
  while infeasible and the normalized H01 projection objective while feasible.
- `--constraint-tol`, `--active-tol`, and `--kkt-tol` control feasibility,
  active-set identification, and the final dimensionless reduced KKT test.

Every reported terminal result requires a final strict projected state and a
fresh reduced-gradient/KKT evaluation. `CONVERGED_REDUCED_KKT` is the strongest
feasible stationarity result. `STAGNATED_FEASIBLE` records observed functional
stagnation with the safeguards satisfied. `STAGNATED_INFEASIBLE` records
functional stagnation when one or both requested geometric bounds could not be
attained on the tracked branch; it is a valid computational stopping state,
but not a feasible constrained optimum. Thus the bounds remain safeguards and
diagnostics without being assumed attainable.

## Branch and optional coercivity diagnostics

- `--branch-overlap-min` bounds the symmetric soft-Dice similarity of
  consecutive nonlinear activity fields. It gates both threshold corrections
  and strictly corrected source-homotopy steps; the default is `0.8`.
- `--branch-topology-guard` independently requires each accepted activity to
  have the expected significant connected-component count. The default count
  is inferred from the discrete sharp target; it may be overridden with
  `--branch-topology-expected-components`. Components are cores above
  `--branch-topology-core-level` connected through cells above
  `--branch-topology-bridge-level`; islands smaller than
  `--branch-topology-min-component-fraction` are ignored. This hysteretic
  cell-graph test catches a gradual neck pinch that consecutive Dice checks
  alone can miss.
- `--predictor-correction-absolute` and
  `--predictor-correction-factor` bound the corrected-state distance from the
  sensitivity predictor after normalization by `sqrt(E_T)`.
- `--predictor-max-change` declares an extreme/nonfinite predictor unusable;
  Newton then starts from the last accepted state.
- `--report-coercivity` enables the generalized Jacobian eigenvalue report.
  `--coercivity-min` additionally makes its conservative
  `mu_min-eigen_error` value an acceptance gate. The eigensolver controls are
  `--coercivity-eig-tol`, `--coercivity-eig-max-it`,
  `--coercivity-zero-tol`, and `--coercivity-inertia`.

## Verification and output

- `--verify-window-derivatives` writes the default pointwise centered-
  difference check. `--verify-reduced-derivatives` adds four strict projected
  states to check `grad J`, `grad L`, and `grad M`; `--fd-relative-step` and
  `--fd-signal-factor` control that check.
- `--run-dir` and `--run-tag` select output location. `--write-xdmf` and
  `--make-plots` toggle field and summary-plot output.
- `--save-terminal-log` mirrors live stdout and stderr from all MPI ranks to
  `out/terminal.log` while preserving the interactive terminal display.
- Repeating `-v` raises verbosity. At level three (the existing `-vv`
  high-verbosity commands), every frozen topology candidate reports its
  component and coverage guards, each continuation tier reports whether it
  selected or skipped a candidate, and every strict projection, source
  stage, threshold micro-step, and outer trial reports its individual guards
  and the action taken. Frozen geometry is explicitly marked `RANK_ONLY` or
  `DIAGNOSTIC_ONLY` wherever it is not an acceptance gate.
- `--plot --plot-mode nonblocking` gathers complete distributed fields and
  updates one live PyVista window on rank zero. `--plot-design`,
  `--plot-initial-candidates`, `--plot-homotopy-stages`, and
  `--plot-accepted-states` select live stages; homotopy plotting updates after
  each accepted exact continuation correction so the GUI remains interactive;
  `--plot-fields full|state|density` selects the panel set. By default,
  `--plot-final-blocking` pauses at the final state for interactive inspection.
- `--fail-on-nonconvergence` accepts reduced-KKT convergence and either
  explicitly labelled functional-stagnation status as completed numerical
  outcomes. It still returns nonzero for iteration-budget exhaustion and
  solver/branch failures. A failed mandatory final Newton projection always
  returns nonzero.

Each run writes `logs/initialization.csv`, `logs/homotopy_threshold.csv`,
`logs/outer_iterations.csv`, `logs/newton.csv`, initialization/derivative JSON diagnostics, a portable
`out/final_equilibrium.npz`, a JSON summary, XDMF fields, and a multi-panel
history plot. Higher-order continuous fields are interpolated to CG1 and the
sharp/smooth densities are L2-projected to DG0 because XDMF output on the
linear geometry cannot directly represent arbitrary higher-order fields.

## Reproducible small example

The repository rank policy assigns one MUMPS rank to `(mesh_size=0.30,
order=2)`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run -n fenicsx-dgfem python \
  projects/diocotron/dolfinx/torsion/optimization/h1_projection.py \
  --mesh-size 0.30 --order 2 --linear-solver mumps \
  --center-min 0.015 --center-max 0.030 \
  --width-min 0.001 --width-max 0.006 \
  --leakage-max 0.60 --missing-max 0.50 \
  --verify-reduced-derivatives \
  --run-tag h1_projection_h030_p2
```

This optimizer follows and projects onto one locally selected equilibrium
branch. A successful run is not evidence of global convergence, global
uniqueness, or global uniqueness of the closest equilibrium.
