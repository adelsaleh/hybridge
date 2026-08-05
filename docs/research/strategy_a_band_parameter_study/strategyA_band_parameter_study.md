# Torsion-Initialized Newton Band Parameter Study

Date: 2026-07-12

This study checks how the torsion-initialized Newton band parameters
`alphaT1`, `alphaT2`, `betaPhi1`, and `betaPhi2` affect the agreement between
the torsion-designed density band and the final converged equilibrium density
band in `scripts/diocotron_dolfinx/dolfinx_torsion_initialized_newton.py`.

## Setup

- Solver: DOLFINx torsion-initialized Newton runner.
- Polynomial order: `p=4`.
- Mesh: fixed smooth-star Gmsh mesh,
  `run_outputs/strategyA_band_study_20260712/fixed_mesh/smooth_star_h007_n300.msh`.
- Mesh size: `nt=16,238`, `ndof=130,505`.
- Continuation: default `epsPhi` ratios `(0.11, 0.08, 0.06)`.
- Linear solver: default direct `mumps`.
- Plots: disabled for all runs.
- Verbosity: `-v 2`, so accepted Newton steps and line-search diagnostics are
  preserved in each run's `terminal.log`.

The fixed mesh was used for every run. This is important because the band
metrics are geometric and should not be compared across different meshes.

## Metrics

The study ranks closeness primarily by

```text
relRhoDesign = ||rho_final - rho_design||_L2 / ||rho_design||_L2
```

and cross-checks the ranking with:

- `rhoDesignDiffL2`: absolute L2 difference between final and design densities.
- `activeJaccard`: Jaccard overlap of active sets
  `rho > 0.05 rho_amp` and `rho_design > 0.05 rho_amp`.
- `massRhoMinusDesign`: signed final-design mass difference.
- line-search behavior from the `-v 2` logs.

The plateau Jaccard was also logged, but it was zero in these runs because the
very high-density plateau locations did not overlap under the `0.90 rho_amp`
threshold. The active-set metric was more informative for this sweep.

## Recommended Parameters

The current practical recommendation is to use the classical `phi-design`
window, where

```text
c1Phi = betaPhi1 * max(phiDesign)
c2Phi = betaPhi2 * max(phiDesign)
epsPhi = eps_phi_ratio * (c2Phi - c1Phi)
```

The best balanced dense-sweep choice is

```text
alphaT1  = 0.40
alphaT2  = 0.50
betaPhi1 = 0.75625
betaPhi2 = 0.89375
```

In the width/shift notation used by the dense beta study,

```text
alpha_center = 0.45
alpha_width  = 0.10
gamma        = 1.375
delta        = 0.375
beta_width   = gamma * alpha_width = 0.1375
beta_center  = alpha_center + delta = 0.825
beta         = (0.75625, 0.89375)
```

This choice had nearly the same relative-density mismatch as the second-best
relative-L2 case, better active overlap than the pure relative-L2 winner, and
almost zero mass mismatch.  The more aggressive relative-L2 winner is

```text
alphaT1  = 0.40
alphaT2  = 0.50
betaPhi1 = 0.80
betaPhi2 = 0.95
gamma    = 1.50
delta    = 0.425
```

but it lies close to the high-beta robustness boundary.  If active-band overlap
is the main target, use

```text
alphaT1  = 0.40
alphaT2  = 0.50
betaPhi1 = 0.60625
betaPhi2 = 0.69375
gamma    = 0.875
delta    = 0.20
```

The newer torsion-window form is also available:

```text
base_width = (alphaT2 - alphaT1) * max(T)
c1Phi = alphaT1 * max(T) + shift_scale * base_width
c2Phi = c1Phi + width_scale * base_width
```

with command-line flags

```text
--phi-window-source torsion
--phi-window-torsion-shift-scale <shift>
--phi-window-torsion-width-scale <width>
```

This torsion-scaled form is not the recommended production choice for the
current smooth-star setup.  The tested `p=5` cases near the torsion thresholds
had `c1Phi` far above `max(phiDesign)` and ended nonconverged with zero or
near-zero final density.  The successful equilibria in this study use the
classical `phi-design` beta window above.  A compact runnable summary is kept
in `docs/research/strategy_a_band_parameter_study/recommended_strategyA_parameters.md`.

### Torsion-Scale `c1,c2` Retest Around the Original FreeFEM Window

We retested nonlinear windows whose absolute thresholds are close to the
original torsion fractions

```text
alphaT1 = 0.60,
alphaT2 = 0.70,
c1 = alphaT1 * max(T),
c2 = alphaT2 * max(T).
```

In these runs the torsion-designed initializer stayed fixed at
`alphaT=(0.60,0.70)`, while the semilinear nonlinear window used
`--phi-window-source torsion`.  Thus `betaPhi1,betaPhi2` did not define the
nonlinear thresholds.  The runs used the Dolfinx no-adapt runner with `p=4`,
`starN=300`, `meshSize=0.07`, MUMPS, no plotting, and the same continuation
ratios as the rest of this study.  The raw rows are in
`docs/research/strategy_a_band_parameter_study/torsion_window_retest_results.csv`.

The result is unambiguous: these thresholds live on the torsion scale, while
the semilinear unknown `phi` lives on the smaller `phiDesign` scale.  Here
`max(T)=0.4677279` and `max(phiDesign)=0.04977098`, so even the lowest retested
`c1` is more than five times larger than `max(phiDesign)`.  The nonlinear
window therefore sees essentially no density and the solver resets to the
design state at each continuation level.

| c1/max(T) | c2/max(T) | shift | width scale | c1 | c2 | c1/max(phiDesign) | c2/max(phiDesign) | max rho | mass rho | relRhoDesign | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0.58` | `0.68` | `-0.2` | `1.0` | 0.271282 | 0.318055 | 5.451 | 6.390 | 0.0 | 0.0 | 1.0 | `NONCONVERGED` |
| `0.59` | `0.69` | `-0.1` | `1.0` | 0.275959 | 0.322732 | 5.545 | 6.484 | 0.0 | 0.0 | 1.0 | `NONCONVERGED` |
| `0.60` | `0.70` | `0.0` | `1.0` | 0.280637 | 0.327410 | 5.639 | 6.578 | 0.0 | 0.0 | 1.0 | `NONCONVERGED` |
| `0.61` | `0.71` | `0.1` | `1.0` | 0.285314 | 0.332087 | 5.733 | 6.672 | 0.0 | 0.0 | 1.0 | `NONCONVERGED` |
| `0.62` | `0.72` | `0.2` | `1.0` | 0.289991 | 0.336764 | 5.827 | 6.766 | 0.0 | 0.0 | 1.0 | `NONCONVERGED` |

This confirms that reusing `alpha_i*max(T)` directly as the semilinear
`c_i` thresholds is not scale-compatible for this formulation.  The successful
windows must be chosen on the `phiDesign` scale, or the torsion-scale thresholds
must first be mapped through the Poisson initializer before being used as
semilinear `phi` thresholds.

### Fitted Torsion-Design Window v2

The separate v2 Dolfinx runner

```text
scripts/diocotron_dolfinx/dolfinx_torsion_initialized_window_fit_newton.py
```

has no `--phi-window-source` switch.  The fitted path is unconditional.  The
original v1 runner, `scripts/diocotron_dolfinx/dolfinx_torsion_initialized_newton.py`,
keeps the previous `phi-design` and `torsion` window-source interface.  In v2,
the fitted mode keeps the torsion-designed density `rhoDesign=f(T;c1T,c2T)`
unchanged, solves the Poisson initializer `-Delta phiDesign=rhoDesign`, and
then chooses `c1Phi,c2Phi` by minimizing

```text
|| f(phiDesign; c1Phi, c2Phi, epsFit) - rhoDesign ||_L2.
```

The fit is done on Dolfinx/Basix quadrature samples.  The search itself uses a
weighted `phiDesign` histogram so the candidate scan is fast, then the selected
pair is evaluated once against the full quadrature samples.  In the p=4,
`starN=300`, `meshSize=0.07` comparison below, the fit used 893,090 quadrature
samples, 4,096 histogram bins, and took 0.305 seconds.

The important result is negative: fitting the initializer mismatch is not the
same as fitting the final nonlinear fixed point.  The fitted window converges
as a nonlinear solve, but its final density band separates badly from the
torsion-designed band.  It is therefore not competitive with the best
hand-shifted `phi-design` windows from the sweep.

Raw rows are in
`docs/research/strategy_a_band_parameter_study/fitted_window_v2_comparison.csv`.

| case | alphaT | phi window | c1Phi/max(phiDesign) | c2Phi/max(phiDesign) | relRhoDesign | mass diff | activeJaccard | plateauJaccard | time (s) | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| fitted v2 | `(0.60,0.70)` | fitted to `rhoDesign` | 0.855610 | 0.958922 | 1.262052 | -0.157657 | 0.072171 | 0.000000 | 76.01 | `OK` |
| best balanced beta | `(0.40,0.50)` | `(0.75625,0.89375)*max(phiDesign)` | 0.756250 | 0.893750 | 0.409461 | -0.002107 | 0.664255 | 0.541667 | 60.12 | `OK` |
| best relRhoDesign beta | `(0.40,0.50)` | `(0.80,0.95)*max(phiDesign)` | 0.800000 | 0.950000 | 0.384985 | 0.018011 | 0.582051 | 0.648425 | 92.95 | `OK` |
| best activeJaccard beta | `(0.40,0.50)` | `(0.60625,0.69375)*max(phiDesign)` | 0.606250 | 0.693750 | 0.547906 | -0.130093 | 0.758497 | 0.364937 | 60.00 | `OK` |

The fitted v2 window is useful diagnostically because it gives an automated
map from the torsion-designed density to a `phi`-scale nonlinear window.
However, on this test it maps to a high `phi` window and converges to a much
smaller active overlap.  The next automated approach should therefore fit a
closed-loop criterion, for example an outer search over `c1Phi,c2Phi` using the
final converged density diagnostics, not only the initializer mismatch.

## Completed Runs

| rank | parameters | relRhoDesign | rhoDesignDiffL2 | activeJaccard | mass diff | time (s) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `alpha=(0.45,0.55)`, `beta=(0.60,0.70)` | 1.101402 | 0.834840 | 0.513855 | -0.031767 | 66.61 |
| 2 | `alpha=(0.50,0.60)`, `beta=(0.60,0.70)` | 1.226505 | 0.916113 | 0.408180 | -0.008784 | 69.55 |
| 3 | `alpha=(0.55,0.65)`, `beta=(0.60,0.70)` | 1.295088 | 0.956198 | 0.308372 | 0.009689 | 60.63 |
| 4 | `alpha=(0.60,0.70)`, `beta=(0.60,0.70)` | 1.328580 | 0.972271 | 0.241929 | 0.024510 | 52.02 |
| 5 | `alpha=(0.65,0.75)`, `beta=(0.60,0.70)` | 1.345975 | 0.978598 | 0.201530 | 0.035878 | 57.13 |
| 6 | `alpha=(0.60,0.70)`, `beta=(0.65,0.80)` | 1.443650 | 1.056480 | 0.162580 | 0.226901 | 75.15 |
| 7 | `alpha=(0.60,0.70)`, `beta=(0.60,0.75)` | 1.480901 | 1.083741 | 0.075095 | 0.278467 | 79.13 |

Supporting CSV files:

- `docs/research/strategy_a_band_parameter_study/results.csv`
- `docs/research/strategy_a_band_parameter_study/interrupted_cases.csv`

## Interrupted Cases

Two low-beta cases were stopped because their line searches entered poor
regimes and were no longer useful as successful candidates:

| parameters | last stage | last residual | last alpha | backtracks | relRhoDesign | activeJaccard |
|---|---:|---:|---:|---:|---:|---:|
| `alpha=(0.60,0.70)`, `beta=(0.50,0.65)` | `ieps=1, k=2` | 4.642694e-03 | 2.384e-07 | 22 | 1.435762 | 0.004967 |
| `alpha=(0.60,0.70)`, `beta=(0.55,0.70)` | `ieps=0, k=18` | 3.521832e-03 | 6.250e-02 | 4 | 1.430719 | 0.027175 |

The first of these was especially poor: accepted overlap had essentially
collapsed and the line search repeatedly reached `alpha_min`.

## Observed Relation

For fixed `beta=(0.60,0.70)`, lowering the torsion window improved the final
agreement monotonically over the tested range:

```text
alpha=(0.65,0.75): rel=1.345975, activeJ=0.201530
alpha=(0.60,0.70): rel=1.328580, activeJ=0.241929
alpha=(0.55,0.65): rel=1.295088, activeJ=0.308372
alpha=(0.50,0.60): rel=1.226505, activeJ=0.408180
alpha=(0.45,0.55): rel=1.101402, activeJ=0.513855
```

For fixed `alpha=(0.60,0.70)`, increasing the beta window worsened the final
agreement:

```text
beta=(0.60,0.70): rel=1.328580, activeJ=0.241929
beta=(0.60,0.75): rel=1.480901, activeJ=0.075095
beta=(0.65,0.80): rel=1.443650, activeJ=0.162580
```

Lowering the beta window was not robust in this setup. The two low-beta runs
showed small accepted steps, many backtracks, and very small active-set overlap.

## Conclusion

Within this tested range, the closest design band to the final equilibrium band
is produced by

```text
alphaT1 = 0.45
alphaT2 = 0.55
betaPhi1 = 0.60
betaPhi2 = 0.70
```

This is a tested-range result, not a global optimum proof. The monotone
improvement as `alpha` is lowered means the next useful sweep should keep
`beta=(0.60,0.70)` fixed and test a bracket below the current best, for example
`alpha=(0.40,0.50)`, `(0.425,0.525)`, and `(0.45,0.55)`. If that lower bracket
turns around, then a narrower alpha-only refinement would be justified.

## Appended Coarse-Mesh Plotted Checks

Additional checks were run on a slightly coarser fixed mesh to make plotted
parameter checks cheaper:

- Mesh:
  `run_outputs/strategyA_band_study_20260712/fixed_mesh/smooth_star_h010_n220.msh`.
- Mesh size: `nt=9,110`, `ndof=73,321` for `p=4`.
- Plotting: final PyVista figure enabled with
  `--plot --plot-off-screen --save-frames --plot-final`.
- Saved frames: every completed run has exactly one final PNG. The image paths
  are recorded in
  `docs/research/strategy_a_band_parameter_study/coarse_final_frame_manifest.csv`, and
  copies are in `docs/research/strategy_a_band_parameter_study/final_frames_coarse/`.

This second family checked lower-alpha refinements and coupled beta shifts.
Because this mesh differs from the first study mesh, the following numbers
should be compared within this coarse-mesh block, not directly as a strict
replacement for the fine-mesh values.

| rank | mesh nt | parameters | relRhoDesign | rhoDesignDiffL2 | activeJaccard | mass diff | frame count |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 9,110 | `alpha=(0.40,0.50)`, `beta=(0.675,0.775)` | 0.484019 | 0.373529 | 0.582366 | -0.113546 | 1 |
| 2 | 9,110 | `alpha=(0.40,0.50)`, `beta=(0.65,0.75)` | 0.484042 | 0.373547 | 0.743608 | -0.096587 | 1 |
| 3 | 9,110 | `alpha=(0.375,0.475)`, `beta=(0.65,0.75)` | 0.497609 | 0.387993 | 0.631468 | -0.112520 | 1 |
| 4 | 9,110 | `alpha=(0.425,0.525)`, `beta=(0.65,0.75)` | 0.587736 | 0.449318 | 0.772225 | -0.082539 | 1 |
| 5 | 9,110 | `alpha=(0.35,0.45)`, `beta=(0.60,0.70)` | 0.694458 | 0.547477 | 0.659600 | -0.094118 | 1 |
| 6 | 9,110 | `alpha=(0.40,0.50)`, `beta=(0.625,0.725)` | 0.704700 | 0.543834 | 0.709753 | -0.078515 | 1 |
| 7 | 9,110 | `alpha=(0.45,0.55)`, `beta=(0.65,0.75)` | 0.730655 | 0.553824 | 0.747043 | -0.069589 | 1 |
| 8 | 9,110 | `alpha=(0.40,0.50)`, `beta=(0.60,0.70)` | 0.904823 | 0.698274 | 0.616926 | -0.059942 | 1 |
| 9 | 9,110 | `alpha=(0.35,0.45)`, `beta=(0.55,0.65)` | 0.959001 | 0.756030 | 0.520284 | -0.057169 | 1 |
| 10 | 9,110 | `alpha=(0.425,0.525)`, `beta=(0.60,0.70)` | 1.012675 | 0.774179 | 0.562224 | -0.045469 | 1 |
| 11 | 9,110 | `alpha=(0.45,0.55)`, `beta=(0.60,0.70)` | 1.101468 | 0.834893 | 0.509646 | -0.031623 | 1 |
| 12 | 9,110 | `alpha=(0.40,0.50)`, `beta=(0.55,0.65)` | 1.127332 | 0.869989 | 0.406804 | -0.020495 | 1 |
| 13 | 9,110 | `alpha=(0.45,0.55)`, `beta=(0.55,0.65)` | 1.251967 | 0.948969 | 0.299070 | 0.009687 | 1 |

Supporting appended artifacts:

- `docs/research/strategy_a_band_parameter_study/coarse_results.csv`
- `docs/research/strategy_a_band_parameter_study/coarse_final_frame_manifest.csv`
- `docs/research/strategy_a_band_parameter_study/final_frames_coarse/`

The two best relative-L2 cases are effectively tied:

```text
alpha=(0.40,0.50), beta=(0.675,0.775): rel=0.484019, activeJ=0.582366
alpha=(0.40,0.50), beta=(0.65,0.75):  rel=0.484042, activeJ=0.743608
```

The difference in `relRhoDesign` is only `2.3e-05`, while the active-band
overlap is much better for `beta=(0.65,0.75)`. On the coarse plotted mesh, the
more balanced choice is therefore

```text
alphaT1 = 0.40
alphaT2 = 0.50
betaPhi1 = 0.65
betaPhi2 = 0.75
```

This appended family changes the earlier interpretation: once `alpha` is moved
down into the `0.40-0.50` range, the preferred beta window shifts upward rather
than staying at `(0.60,0.70)`. The best next check is to repeat the two leading
coarse candidates on the finer `nt=16,238` mesh.

## Appended Extreme-Range Checks

A further coarse-mesh sweep tested windows closer to zero and closer to one,
again on
`run_outputs/strategyA_band_study_20260712/fixed_mesh/smooth_star_h010_n220.msh`
with `nt=9,110`, `ndof=73,321`, and `p=4`.

Each successful run saved one final PyVista PNG. Those final-frame copies are
in `docs/research/strategy_a_band_parameter_study/final_frames_extremes/`, with paths in
`docs/research/strategy_a_band_parameter_study/coarse_extreme_frame_manifest.csv`.

| rank | mesh nt | parameters | relRhoDesign | rhoDesignDiffL2 | activeJaccard | mass diff | frame count |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 9,110 | `alpha=(0.70,0.80)`, `beta=(0.70,0.80)` | 0.826071 | 0.597888 | 0.816894 | -0.054066 | 1 |
| 2 | 9,110 | `alpha=(0.05,0.15)`, `beta=(0.05,0.15)` | 0.971700 | 0.904201 | 0.494615 | -0.386779 | 1 |
| 3 | 9,110 | `alpha=(0.10,0.20)`, `beta=(0.10,0.20)` | 1.105575 | 1.000852 | 0.324523 | -0.240512 | 1 |
| 4 | 9,110 | `alpha=(0.20,0.30)`, `beta=(0.20,0.30)` | 1.224105 | 1.045609 | 0.191160 | -0.068719 | 1 |
| 5 | 9,110 | `alpha=(0.80,0.90)`, `beta=(0.40,0.50)` | 1.520418 | 1.095090 | 0.000000 | 0.307132 | 1 |

The following extreme/cross-extreme cases did not reach final convergence
within the 210-second guard:

| parameters | status | guard (s) | final equilibrium PNG |
|---|---|---:|---|
| `alpha=(0.80,0.90)`, `beta=(0.80,0.90)` | timeout | 210 | none |
| `alpha=(0.90,0.98)`, `beta=(0.90,0.98)` | timeout | 210 | none |
| `alpha=(0.10,0.20)`, `beta=(0.65,0.75)` | timeout | 210 | none |
| `alpha=(0.20,0.30)`, `beta=(0.65,0.75)` | timeout | 210 | none |
| `alpha=(0.40,0.50)`, `beta=(0.85,0.95)` | timeout | 210 | none |

For these timed-out cases I also ran a one-step diagnostic pass with
`--eps-ratios 0.11 --max-it 1` and saved the resulting nonconverged PyVista
state. These images are not final equilibrium results; they are only visual
diagnostics for the parameter windows that did not finish. They are stored in
`docs/research/strategy_a_band_parameter_study/diagnostic_frames_extreme_timeouts/`, with
metadata in
`docs/research/strategy_a_band_parameter_study/coarse_extreme_diagnostic_frames.csv`.

Supporting extreme-range artifacts:

- `docs/research/strategy_a_band_parameter_study/coarse_extreme_results.csv`
- `docs/research/strategy_a_band_parameter_study/coarse_extreme_frame_manifest.csv`
- `docs/research/strategy_a_band_parameter_study/coarse_extreme_timeouts.csv`
- `docs/research/strategy_a_band_parameter_study/coarse_extreme_diagnostic_frames.csv`
- `docs/research/strategy_a_band_parameter_study/final_frames_extremes/`
- `docs/research/strategy_a_band_parameter_study/diagnostic_frames_extreme_timeouts/`

Extreme-range interpretation:

- Very low diagonal windows do converge on the coarse mesh, but they are not
  competitive with the current best `alpha=(0.40,0.50), beta=(0.65,0.75)`.
  They also carry large negative mass differences.
- Moderate high diagonal `alpha=(0.70,0.80), beta=(0.70,0.80)` converges and has
  strong active overlap, but its relative density mismatch is still worse than
  the best middle-window family.
- Near-one diagonal windows `0.80-0.90` and `0.90-0.98` are too slow/non-robust
  under the guard.
- Low alpha with high beta, `alpha=(0.10,0.30), beta=(0.65,0.75)`, is also
  non-robust. This supports a lower usable alpha boundary somewhere above
  `0.30`.
- Pushing beta to `0.85-0.95` with the good alpha window also times out, so the
  useful beta upper boundary appears to be below `0.85`.

After these extreme checks, the best practical coarse-mesh candidate remains
`alpha=(0.40,0.50), beta=(0.65,0.75)`.

## Appended Beta Width/Shift Study

The next sweep fixed the best practical alpha window from the coarse study:

```text
alpha1 = 0.40
alpha2 = 0.50
```

The beta band was parameterized by a width multiplier `gamma` and a center
shift `delta`:

```text
alpha_width = alpha2 - alpha1 = 0.10
alpha_center = 0.45
beta_width = gamma * alpha_width
beta_center = alpha_center + delta
beta1 = beta_center - 0.5 * beta_width
beta2 = beta_center + 0.5 * beta_width
```

The first grid used `gamma={0.50,0.75,1.00,1.25,1.50}` and
`delta={0.20,0.25,0.30}`. Since the best point was internal, I refined locally
with `gamma={0.90,1.00,1.10}` and `delta={0.235,0.250,0.265}`.

All 23 runs completed on the same coarse mesh (`nt=9,110`, `ndof=73,321`) and
each saved one final PyVista PNG. The figure copies are in
`docs/research/strategy_a_band_parameter_study/final_frames_beta_width_shift/`.

| rank | gamma | delta | beta band | relRhoDesign | rhoDesignDiffL2 | activeJaccard | mass diff | frames |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.265 | `(0.665,0.765)` | 0.443000 | 0.341873 | 0.648114 | -0.107128 | 1 |
| 2 | 1.00 | 0.250 | `(0.650,0.750)` | 0.484042 | 0.373547 | 0.743608 | -0.096587 | 1 |
| 3 | 0.90 | 0.235 | `(0.640,0.730)` | 0.484171 | 0.373647 | 0.618766 | -0.140009 | 1 |
| 4 | 0.90 | 0.250 | `(0.655,0.745)` | 0.579720 | 0.447384 | 0.513072 | -0.149851 | 1 |
| 5 | 1.00 | 0.235 | `(0.635,0.735)` | 0.611210 | 0.471686 | 0.735865 | -0.086059 | 1 |
| 6 | 1.10 | 0.265 | `(0.660,0.770)` | 0.627685 | 0.484399 | 0.725000 | -0.055101 | 1 |
| 7 | 1.00 | 0.300 | `(0.700,0.800)` | 0.669046 | 0.516319 | 0.431572 | -0.130029 | 1 |
| 8 | 0.75 | 0.200 | `(0.6125,0.6875)` | 0.676799 | 0.522302 | 0.455162 | -0.202082 | 1 |
| 9 | 0.90 | 0.265 | `(0.670,0.760)` | 0.697331 | 0.538147 | 0.420288 | -0.159394 | 1 |
| 10 | 1.25 | 0.300 | `(0.6875,0.8125)` | 0.712958 | 0.550207 | 0.692189 | -0.006361 | 1 |

Supporting artifacts:

- `docs/research/strategy_a_band_parameter_study/coarse_beta_width_shift_results.csv`
- `docs/research/strategy_a_band_parameter_study/coarse_beta_width_shift_frame_manifest.csv`
- `docs/research/strategy_a_band_parameter_study/final_frames_beta_width_shift/`

Interpretation:

- The best relative density mismatch is obtained by
  `gamma=1.00`, `delta=0.265`, giving `beta=(0.665,0.765)`.
- The previous `beta=(0.650,0.750)` remains the best active-overlap compromise:
  it has slightly worse relative mismatch, `0.484042` instead of `0.443000`,
  but better active Jaccard, `0.743608` instead of `0.648114`.
- Widths much narrower than alpha (`gamma=0.50`) are poor: they gave zero
  active overlap in this sweep.
- Widths much wider than alpha (`gamma=1.50`) were also poor, mostly by
  increasing mass mismatch and relative density mismatch.

For matching the torsion-designed density in relative L2, the best tested
choice is therefore

```text
alpha=(0.40,0.50)
gamma=1.00
delta=0.265
beta=(0.665,0.765)
```

For a more geometric active-band match, `beta=(0.650,0.750)` remains preferable.

## Dense Beta Width/Shift Sweep With Negative Delta

The previous beta width/shift sweep was still too narrow because it mostly used
positive center shifts. I therefore ran a dense sweep for the same fixed alpha
window,

```text
alpha=(0.40,0.50)
```

with

```text
gamma = 0.50, 0.625, 0.75, 0.875, 1.00, 1.125, 1.25, 1.375, 1.50
delta = -0.25, -0.20, ..., 0.35
```

and then extended the high-shift side with

```text
gamma = 0.875, 1.00, 1.125, 1.25, 1.375, 1.50
delta = 0.375, 0.40, 0.425, 0.45
```

This gives 141 attempted parameter pairs. Of these, 122 completed with final
PyVista figures and 19 did not reach a final state within the guard. Every
completed run has one copied final PNG in
`docs/research/strategy_a_band_parameter_study/final_frames_beta_width_shift_wide/`.

Supporting dense-grid artifacts:

- `docs/research/strategy_a_band_parameter_study/coarse_beta_width_shift_wide_results.csv`
- `docs/research/strategy_a_band_parameter_study/coarse_beta_width_shift_wide_frame_manifest.csv`
- `docs/research/strategy_a_band_parameter_study/coarse_beta_width_shift_wide_timeouts.csv`
- `docs/research/strategy_a_band_parameter_study/coarse_beta_width_shift_wide_rel_heatmap.png`
- `docs/research/strategy_a_band_parameter_study/coarse_beta_width_shift_wide_activej_heatmap.png`
- `docs/research/strategy_a_band_parameter_study/coarse_beta_width_shift_wide_massdiff_heatmap.png`
- `docs/research/strategy_a_band_parameter_study/coarse_beta_width_shift_wide_timeout_map.png`

Top completed cases by relative density mismatch:

| rank | gamma | delta | beta band | relRhoDesign | activeJaccard | mass diff |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1.500 | 0.425 | `(0.8000,0.9500)` | 0.384392 | 0.583595 | 0.017897 |
| 2 | 1.250 | 0.350 | `(0.7375,0.8625)` | 0.408160 | 0.605462 | -0.041988 |
| 3 | 1.375 | 0.375 | `(0.7562,0.8938)` | 0.409392 | 0.666506 | -0.002054 |
| 4 | 1.125 | 0.300 | `(0.6937,0.8063)` | 0.425724 | 0.681023 | -0.067517 |
| 5 | 1.375 | 0.400 | `(0.7812,0.9187)` | 0.450885 | 0.536833 | -0.018853 |
| 6 | 1.500 | 0.400 | `(0.7750,0.9250)` | 0.469656 | 0.671479 | 0.035169 |
| 7 | 1.000 | 0.250 | `(0.6500,0.7500)` | 0.484042 | 0.743608 | -0.096587 |
| 8 | 0.875 | 0.200 | `(0.6062,0.6937)` | 0.548298 | 0.757761 | -0.129805 |

Top completed cases by active-band overlap:

| rank | gamma | delta | beta band | activeJaccard | relRhoDesign | mass diff |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 0.875 | 0.200 | `(0.6062,0.6937)` | 0.757761 | 0.548298 | -0.129805 |
| 2 | 0.750 | 0.150 | `(0.5625,0.6375)` | 0.751513 | 0.594415 | -0.168268 |
| 3 | 0.625 | 0.100 | `(0.5188,0.5813)` | 0.743888 | 0.606611 | -0.213257 |
| 4 | 1.000 | 0.250 | `(0.6500,0.7500)` | 0.743608 | 0.484042 | -0.096587 |
| 5 | 1.375 | 0.350 | `(0.7312,0.8688)` | 0.699553 | 0.591821 | 0.015856 |
| 6 | 1.250 | 0.300 | `(0.6875,0.8125)` | 0.692189 | 0.712958 | -0.006361 |

Dense-grid interpretation:

- Negative `delta` values were necessary to test, but they did not beat the
  positive-shift families. They are generally worse in relative density
  mismatch and include several guarded timeouts for wider `gamma`.
- For relative L2 closeness to the torsion-designed density, the best tested
  band is now

  ```text
  gamma = 1.50
  delta = 0.425
  beta = (0.8000, 0.9500)
  ```

- That best relative-L2 band lies near the high-shift robustness boundary:
  `gamma=1.50, delta=0.45` timed out.
- A more balanced choice is

  ```text
  gamma = 1.375
  delta = 0.375
  beta = (0.7562, 0.8938)
  ```

  because it has nearly the same relative mismatch as the second-best case,
  better active overlap than the pure relative-L2 best, and almost zero mass
  difference.
- If active-band overlap is the primary criterion, the best tested band is
  `gamma=0.875, delta=0.20`, i.e. `beta=(0.6062,0.6937)`.

## Appendix: Tested Window Parameters

The full normalized parameter and diagnostics table is recorded in
`docs/research/strategy_a_band_parameter_study/tested_parameter_values.csv`.  It contains
one row per unique attempted combination of
`alphaT1, alphaT2, betaPhi1, betaPhi2`, including completed, interrupted, and
timed-out runs.  For completed runs the same row also records the measured
diagnostics: runtime, final residual, density-design mismatch, mass difference,
active-band overlap, plateau overlap, and the final annular overshoot measure.

The numerical values of the thresholds are run-dependent because they are
computed from the torsion maximum and the maximum of the torsion-designed
semilinear initialization.  In normalized form the torsion window is

```text
c1T(alphaT1, alphaT2) = alphaT1 * Tmax,
c2T(alphaT1, alphaT2) = alphaT2 * Tmax.
```

After solving the torsion-designed initialization for a fixed torsion window,
let

```text
PhiDmax(alphaT1, alphaT2) = max(phiDesign).
```

The nonlinear semilinear window is then

```text
c1Phi(alphaT1, alphaT2, betaPhi1) = betaPhi1 * PhiDmax(alphaT1, alphaT2),
c2Phi(alphaT1, alphaT2, betaPhi2) = betaPhi2 * PhiDmax(alphaT1, alphaT2).
```

Equivalently, the normalized columns in
`tested_parameter_values.csv` satisfy
`c1T_over_Tmax = alphaT1`, `c2T_over_Tmax = alphaT2`,
`c1Phi_over_phiDesignMax = betaPhi1`, and
`c2Phi_over_phiDesignMax = betaPhi2`.

The diagnostic columns in the CSV are attached directly to the tested parameter
pair.  In particular:

- `diagnostic_status` is the representative run status used for the diagnostics.
- `statuses` lists all observed statuses for that exact parameter combination
  when it appeared in more than one sweep.
- `resEuclid`, `relRhoDesign`, `rhoDesignDiffL2`, `massRhoMinusDesign`,
  `activeJaccard`, `plateauJaccard`, and `annularPhiMinusC2` are copied from
  the completed run whenever available.
- `diagnostic_source` identifies which sweep file provided the representative
  diagnostics, while `sources` lists every sweep file containing the parameter
  pair.

Grouped by torsion window, the tested values were:

| alphaT1 | alphaT2 | c1T | c2T | unique beta windows tested | betaPhi windows |
|---:|---:|---:|---:|---:|---|
| `0.05` | `0.15` | `0.05*Tmax` | `0.15*Tmax` | 1 | `(0.05,0.15)` |
| `0.10` | `0.20` | `0.10*Tmax` | `0.20*Tmax` | 2 | `(0.10,0.20)`, `(0.65,0.75)` |
| `0.20` | `0.30` | `0.20*Tmax` | `0.30*Tmax` | 2 | `(0.20,0.30)`, `(0.65,0.75)` |
| `0.35` | `0.45` | `0.35*Tmax` | `0.45*Tmax` | 2 | `(0.55,0.65)`, `(0.60,0.70)` |
| `0.375` | `0.475` | `0.375*Tmax` | `0.475*Tmax` | 1 | `(0.65,0.75)` |
| `0.40` | `0.50` | `0.40*Tmax` | `0.50*Tmax` | 151 | see `tested_parameter_values.csv` |
| `0.425` | `0.525` | `0.425*Tmax` | `0.525*Tmax` | 2 | `(0.60,0.70)`, `(0.65,0.75)` |
| `0.45` | `0.55` | `0.45*Tmax` | `0.55*Tmax` | 3 | `(0.55,0.65)`, `(0.60,0.70)`, `(0.65,0.75)` |
| `0.50` | `0.60` | `0.50*Tmax` | `0.60*Tmax` | 1 | `(0.60,0.70)` |
| `0.55` | `0.65` | `0.55*Tmax` | `0.65*Tmax` | 1 | `(0.60,0.70)` |
| `0.60` | `0.70` | `0.60*Tmax` | `0.70*Tmax` | 5 | `(0.50,0.65)`, `(0.55,0.70)`, `(0.60,0.70)`, `(0.60,0.75)`, `(0.65,0.80)` |
| `0.65` | `0.75` | `0.65*Tmax` | `0.75*Tmax` | 1 | `(0.60,0.70)` |
| `0.70` | `0.80` | `0.70*Tmax` | `0.80*Tmax` | 1 | `(0.70,0.80)` |
| `0.80` | `0.90` | `0.80*Tmax` | `0.90*Tmax` | 2 | `(0.40,0.50)`, `(0.80,0.90)` |
| `0.90` | `0.98` | `0.90*Tmax` | `0.98*Tmax` | 1 | `(0.90,0.98)` |

The following roll-up table gives the best completed diagnostics within each
torsion window.  The complete per-pair diagnostics are in
`tested_parameter_values.csv`.

| alphaT1 | alphaT2 | pairs | OK | non-OK | best beta by relRhoDesign | relRhoDesign | mass diff | activeJaccard | best beta by activeJaccard | activeJaccard | relRhoDesign |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| `0.05` | `0.15` | 1 | 1 | 0 | `(0.05,0.15)` | 0.9717 | -0.386779 | 0.494615 | `(0.05,0.15)` | 0.494615 | 0.9717 |
| `0.1` | `0.2` | 2 | 1 | 1 | `(0.1,0.2)` | 1.10557 | -0.240512 | 0.324523 | `(0.1,0.2)` | 0.324523 | 1.10557 |
| `0.2` | `0.3` | 2 | 1 | 1 | `(0.2,0.3)` | 1.2241 | -0.0687194 | 0.19116 | `(0.2,0.3)` | 0.19116 | 1.2241 |
| `0.35` | `0.45` | 2 | 2 | 0 | `(0.6,0.7)` | 0.694458 | -0.0941182 | 0.6596 | `(0.6,0.7)` | 0.6596 | 0.694458 |
| `0.375` | `0.475` | 1 | 1 | 0 | `(0.65,0.75)` | 0.497609 | -0.11252 | 0.631468 | `(0.65,0.75)` | 0.631468 | 0.497609 |
| `0.4` | `0.5` | 151 | 132 | 19 | `(0.8,0.95)` | 0.384392 | 0.0178971 | 0.583595 | `(0.60625,0.69375)` | 0.757761 | 0.548298 |
| `0.425` | `0.525` | 2 | 2 | 0 | `(0.65,0.75)` | 0.587736 | -0.0825392 | 0.772225 | `(0.65,0.75)` | 0.772225 | 0.587736 |
| `0.45` | `0.55` | 3 | 3 | 0 | `(0.65,0.75)` | 0.730655 | -0.0695894 | 0.747043 | `(0.65,0.75)` | 0.747043 | 0.730655 |
| `0.5` | `0.6` | 1 | 1 | 0 | `(0.6,0.7)` | 1.22651 | -0.00878374 | 0.40818 | `(0.6,0.7)` | 0.40818 | 1.22651 |
| `0.55` | `0.65` | 1 | 1 | 0 | `(0.6,0.7)` | 1.29509 | 0.00968927 | 0.308372 | `(0.6,0.7)` | 0.308372 | 1.29509 |
| `0.6` | `0.7` | 5 | 3 | 2 | `(0.6,0.7)` | 1.32858 | 0.02451 | 0.241929 | `(0.6,0.7)` | 0.241929 | 1.32858 |
| `0.65` | `0.75` | 1 | 1 | 0 | `(0.6,0.7)` | 1.34598 | 0.0358783 | 0.20153 | `(0.6,0.7)` | 0.20153 | 1.34598 |
| `0.7` | `0.8` | 1 | 1 | 0 | `(0.7,0.8)` | 0.826071 | -0.0540661 | 0.816894 | `(0.7,0.8)` | 0.816894 | 0.826071 |
| `0.8` | `0.9` | 2 | 1 | 1 | `(0.4,0.5)` | 1.52042 | 0.307132 | 0 | `(0.4,0.5)` | 0 | 1.52042 |
| `0.9` | `0.98` | 1 | 0 | 1 | - | - | - | - | - | - | - |
