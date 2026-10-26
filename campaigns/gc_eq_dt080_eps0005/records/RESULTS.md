# Smaller-smoothing CG6 equilibria

Target epsilon/delta = 0.005, target D_T = 0.8. Each solve starts from an accepted archived potential.
Only accepted target-epsilon equilibria appear below. No HDG evolution is performed.

| Geometry | Triangles | Quadrature degree | D_T | PDE residual | Transition length / cell diameter |
| --- | ---: | ---: | ---: | ---: | ---: |
| star | 264,268 | 64 | 0.799999999 | 2.852e-15 | 0.0372 |
| uglypacman | 215,246 | 64 | 0.799999987 | 4.289e-15 | 0.0337 |
| iter | 250,997 | 64 | 0.799999999 | 3.898e-15 | 0.0344 |
| horseshoe | 257,368 | 64 | 0.800000002 | 9.838e-15 | 0.0355 |

## Quadrature comparisons

These compare two nonlinear assembly rules at fixed mesh, degree, smoothing and target placement.
They do not establish spatial mesh convergence of the sharper transitions.

| Geometry | Potential relative L2 difference | Density relative L2 difference | Gradient relative L2 difference |
| --- | ---: | ---: | ---: |
| star | 8.655e-10 | 7.567e-08 | 1.255e-07 |
| uglypacman | 1.545e-09 | 9.120e-08 | 1.032e-07 |
| iter | 8.576e-10 | 7.942e-08 | 1.171e-07 |
| horseshoe | 1.174e-08 | 5.491e-07 | 1.078e-07 |

## Reusable archive entries

- star: `star_dt080_eps0p005_q64_cg6_20261016-50344ef8ba2a`
- uglypacman: `uglypacman_dt080_eps0p005_q64_cg6_20261016-c5665190d632`
- iter: `iter_dt080_eps0p005_q64_cg6_20261016-7a52ddd5ee71`
- horseshoe: `horseshoe_dt080_eps0p005_q64_cg6_20261016-0cd7b1251dd5`
