# Strategy A Band Parameter Study

Date: 2026-07-12

This study checks how the Strategy A band parameters
`alphaT1`, `alphaT2`, `betaPhi1`, and `betaPhi2` affect the agreement between
the torsion-designed density band and the final converged equilibrium density
band in `scripts/strategyA_dolfinx_noadapt_torsion_newton.py`.

## Setup

- Solver: DOLFINx Strategy A torsion/Newton runner.
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
in `docs/strategyA_band_parameter_study/recommended_strategyA_parameters.md`.

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

- `docs/strategyA_band_parameter_study/results.csv`
- `docs/strategyA_band_parameter_study/interrupted_cases.csv`

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
  `docs/strategyA_band_parameter_study/coarse_final_frame_manifest.csv`, and
  copies are in `docs/strategyA_band_parameter_study/final_frames_coarse/`.

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

- `docs/strategyA_band_parameter_study/coarse_results.csv`
- `docs/strategyA_band_parameter_study/coarse_final_frame_manifest.csv`
- `docs/strategyA_band_parameter_study/final_frames_coarse/`

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
in `docs/strategyA_band_parameter_study/final_frames_extremes/`, with paths in
`docs/strategyA_band_parameter_study/coarse_extreme_frame_manifest.csv`.

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
`docs/strategyA_band_parameter_study/diagnostic_frames_extreme_timeouts/`, with
metadata in
`docs/strategyA_band_parameter_study/coarse_extreme_diagnostic_frames.csv`.

Supporting extreme-range artifacts:

- `docs/strategyA_band_parameter_study/coarse_extreme_results.csv`
- `docs/strategyA_band_parameter_study/coarse_extreme_frame_manifest.csv`
- `docs/strategyA_band_parameter_study/coarse_extreme_timeouts.csv`
- `docs/strategyA_band_parameter_study/coarse_extreme_diagnostic_frames.csv`
- `docs/strategyA_band_parameter_study/final_frames_extremes/`
- `docs/strategyA_band_parameter_study/diagnostic_frames_extreme_timeouts/`

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
`docs/strategyA_band_parameter_study/final_frames_beta_width_shift/`.

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

- `docs/strategyA_band_parameter_study/coarse_beta_width_shift_results.csv`
- `docs/strategyA_band_parameter_study/coarse_beta_width_shift_frame_manifest.csv`
- `docs/strategyA_band_parameter_study/final_frames_beta_width_shift/`

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
`docs/strategyA_band_parameter_study/final_frames_beta_width_shift_wide/`.

Supporting dense-grid artifacts:

- `docs/strategyA_band_parameter_study/coarse_beta_width_shift_wide_results.csv`
- `docs/strategyA_band_parameter_study/coarse_beta_width_shift_wide_frame_manifest.csv`
- `docs/strategyA_band_parameter_study/coarse_beta_width_shift_wide_timeouts.csv`
- `docs/strategyA_band_parameter_study/coarse_beta_width_shift_wide_rel_heatmap.png`
- `docs/strategyA_band_parameter_study/coarse_beta_width_shift_wide_activej_heatmap.png`
- `docs/strategyA_band_parameter_study/coarse_beta_width_shift_wide_massdiff_heatmap.png`
- `docs/strategyA_band_parameter_study/coarse_beta_width_shift_wide_timeout_map.png`

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
