# Saved equilibrium results

Updated 2026-10-16T14:42:00.026950+00:00.

Target: D_T = 0.8 in the established torsion-flow coordinate. Only completed, archived states appear below. Placement is intermediate; the finer levels and their comparisons determine spatial resolution. Degree 6 and the mesh-size request apply to DOLFINx only. HDG preservation tests are excluded from this campaign.

| Geometry | Level | Triangles | FE DOFs | D_T | PDE residual | Mean width along flow |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| star | placement (CG2) | 211,768 | 427,363 | 0.799999995 | 8.94e-16 | 0.0399154 |
| uglypacman | placement (CG2) | 158,358 | 320,829 | 0.799999968 | 8.39e-16 | 0.0402460 |
| iter | placement (CG2) | 182,558 | 368,459 | 0.799999999 | 7.80e-16 | 0.1109060 |
| horseshoe | placement (CG2) | 185,876 | 376,613 | 0.799999955 | 1.11e-15 | 0.0409021 |
| star | resolved (CG2) | 823,806 | 1,651,439 | 0.800000059 | 1.54e-15 | 0.0399157 |
| uglypacman | resolved (CG2) | 732,484 | 1,469,129 | 0.799999999 | 1.69e-13 | 0.0402466 |
| horseshoe | resolved (CG2) | 876,284 | 1,757,659 | 0.799999975 | 2.44e-15 | 0.0409011 |
| star | cg4 (CG4) | 211,768 | 1,701,797 | 0.800000076 | 2.20e-15 | 0.0399157 |
| uglypacman | cg4 (CG4) | 158,358 | 1,275,089 | 0.799999999 | 1.12e-13 | 0.0402467 |
| iter | cg4 (CG4) | 182,558 | 1,467,149 | 0.800000000 | 1.93e-15 | 0.1109073 |
| horseshoe | cg4 (CG4) | 185,876 | 1,496,729 | 0.799999975 | 2.72e-15 | 0.0409017 |
| star | cg6 (CG6) | 211,768 | 3,823,303 | 0.800000076 | 4.58e-15 | 0.0399157 |
| uglypacman | cg6 (CG6) | 158,358 | 2,862,781 | 0.800000000 | 4.11e-15 | 0.0402467 |
| iter | cg6 (CG6) | 182,558 | 3,296,071 | 0.800000033 | 3.61e-15 | 0.1109073 |
| horseshoe | cg6 (CG6) | 185,876 | 3,360,349 | 0.799999976 | 6.54e-15 | 0.0409017 |
| star | cg6_refined (CG6) | 264,268 | 4,768,303 | 0.800000078 | 4.51e-15 | 0.0399157 |
| uglypacman | cg6_refined (CG6) | 215,246 | 3,887,077 | 0.799999969 | 5.10e-15 | 0.0402467 |
| iter | cg6_refined (CG6) | 250,997 | 4,529,746 | 0.800000000 | 4.34e-15 | 0.1109073 |
| horseshoe | cg6_refined (CG6) | 257,368 | 4,648,495 | 0.799999975 | 6.35e-15 | 0.0409017 |

## Refinement evidence

Relative L2 differences use the finer field as denominator and the higher quadrature order saved in each comparison (CG4 against CG6 for the new family). Widths and epsilon are fixed between levels; midpoint is fitted again at each level.

| Geometry | Potential | Density | Gradient | Density error estimate: quadrature change | Smallest transition length / crossing-cell diameter | Distance sampling check |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| star / cg6 | 3.023e-08 | 6.913e-07 | 5.084e-05 | 2.258e-05 | 0.295 | pending |
| uglypacman / cg6 | 6.719e-06 | 8.832e-05 | 2.213e-03 | 2.962e-09 | 0.276 | pending |
| iter / cg6 | 3.746e-07 | 4.114e-06 | 3.135e-04 | 7.445e-07 | 0.260 | pending |
| horseshoe / cg6 | 2.468e-08 | 6.920e-07 | 2.535e-05 | 2.982e-05 | 0.297 | pending |
| star / cg6_refined | 6.220e-09 | 1.429e-08 | 1.941e-05 | 1.037e-04 | 0.565 | passed |
| uglypacman / cg6_refined | 4.456e-06 | 5.961e-05 | 1.820e-03 | 1.923e-08 | 0.531 | passed |
| iter / cg6_refined | 1.169e-05 | 2.223e-04 | 2.249e-04 | 1.299e-08 | 0.563 | failed |
| horseshoe / cg6_refined | 4.733e-09 | 1.421e-08 | 1.041e-05 | 1.472e-04 | 0.587 | passed |

## Reusable archive identifiers

- star / placement: `star_dt080_placement_20261016-f270e1ce2ee9`
- uglypacman / placement: `uglypacman_dt080_placement_20261016-19ba0cf9191b`
- iter / placement: `iter_dt080_placement_20261016-f04eb12fe1aa`
- horseshoe / placement: `horseshoe_dt080_placement_20261016-6b6203fc1cd9`
- star / resolved: `star_dt080_resolved_20261016-a8e6e7066e35`
- uglypacman / resolved: `uglypacman_dt080_resolved_20261016-4c8533644a6b`
- horseshoe / resolved: `horseshoe_dt080_resolved_20261016-5f7c3a15cccd`
- star / cg4: `star_dt080_cg4_20261016-7114541cd669`
- uglypacman / cg4: `uglypacman_dt080_cg4_20261016-f67dd8e4ad89`
- iter / cg4: `iter_dt080_cg4_20261016-0ff5f7c07a0e`
- horseshoe / cg4: `horseshoe_dt080_cg4_20261016-58d38a7c2eac`
- star / cg6: `star_dt080_cg6_20261016-5f14fbfa5f52`
- uglypacman / cg6: `uglypacman_dt080_cg6_20261016-8be04755cdf7`
- iter / cg6: `iter_dt080_cg6_20261016-3047fd8f0454`
- horseshoe / cg6: `horseshoe_dt080_cg6_20261016-acf46db49c20`
- star / cg6_refined: `star_dt080_cg6_refined_20261016-6ee44236c6bf`
- uglypacman / cg6_refined: `uglypacman_dt080_cg6_refined_20261016-48f4dc07ef06`
- iter / cg6_refined: `iter_dt080_cg6_refined_20261016-ca20430799b1`
- horseshoe / cg6_refined: `horseshoe_dt080_cg6_refined_20261016-d443f5e088d6`

These are equilibrium inputs. No guiding-center preservation result has been computed for this family.
