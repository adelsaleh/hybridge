# E7 band evolution: summary

## Preservation (controls)

ρ drift at t = 1 is the edge relaxation of the projected profile (angle independent); the angular departure max_{m≥1} A_m is what the instability grows from.

| Domain | transfer: HDG/CG potential | quadrature sensitivity | ρ drift t=1 | φ drift t=1 / 10 / 50 | angular departure t=1 / 10 / 50 | reaches 1e-3 / 1e-2 / 1e-1 at t | φ drift t=200 |
|---|---|---|---|---|---|---|---|
| Star | 1.06e-05 | 9.30e-08 | 1.15e-02 | 9.60e-06 / 1.03e-05 / 1.08e-05 | 3.01e-08 / 3.92e-08 / 2.60e-05 | 68.9 / 81.3 / 94.1 | 2.27e-01 |
| Ugly Pac-Man | 1.49e-05 | 9.14e-08 | 1.14e-02 | 1.34e-05 / 1.45e-05 / 2.05e-05 | 2.13e-08 / 6.73e-08 / 9.89e-05 | 62.8 / 75.7 / 88.7 | 3.13e-01 |
| ITER | 3.54e-06 | 1.38e-07 | 1.23e-02 | 3.22e-06 / 3.48e-06 / 6.01e-06 | 3.33e-08 / 4.72e-08 / 6.81e-05 | 63.9 / 76.2 / 88.5 | 2.71e-01 |
| Horseshoe | 1.48e-05 | 8.55e-08 | 1.14e-02 | 1.32e-05 / 1.44e-05 / 1.90e-05 | 2.49e-08 / 2.99e-08 / 8.64e-05 | 62.1 / 74.1 / 87.2 | 4.78e-01 |

## Band modes

| Domain | ℓ | seed A_ℓ(0) | γ_ℓ seed (window) | vortex-strip γ_ℓ | control fastest m (γ) | unstable m (count) | active modes t=40 / 100 / 200 (seed) | vortices t=0 → T (seed) |
|---|---|---|---|---|---|---|---|---|
| Star | 16 | 2.35e-02 | 0.2013 (2.0–7.5) | 0.2012 | 17 (0.1864) | 1–80 (80) | – / 80 / 79 | 1 → 8 (max 16) |
| Ugly Pac-Man | 16 | 2.29e-02 | 0.2012 (2.0–7.1) | 0.2012 | 16 (0.1844) | 1–80 (80) | – / 79 / 79 | 1 → 8 (max 16) |
| ITER | 18 | 2.26e-02 | 0.2094 (2.0–7.6) | 0.2011 | 19 (0.1950) | 1–80 (80) | – / 79 / 79 | 1 → 11 (max 18) |
| Horseshoe | 20 | 2.25e-02 | 0.2121 (2.0–6.9) | 0.2010 | 19 (0.2130) | 1–80 (80) | – / 80 / 80 | 1 → 8 (max 20) |

## Conservation

| Domain | run | max mass drift | energy drift | enstrophy drift | max L² identity defect | ρ min | wall time (h) |
|---|---|---|---|---|---|---|---|
| Star | control | 2.27e-13 | 1.24e-03 | 3.02e-01 | 6.32e-16 | -0.532 | 0.78 |
| Star | seed | 2.27e-13 | 1.67e-03 | 3.96e-01 | 9.47e-16 | -0.580 | 1.04 |
| Ugly Pac-Man | control | 2.15e-13 | 1.24e-03 | 3.05e-01 | 6.26e-16 | -0.747 | 1.14 |
| Ugly Pac-Man | seed | 2.10e-13 | 2.16e-03 | 4.77e-01 | 6.64e-16 | -0.555 | 1.60 |
| ITER | control | 3.01e-13 | 1.15e-03 | 2.69e-01 | 7.23e-16 | -0.553 | 0.77 |
| ITER | seed | 3.96e-13 | 1.75e-03 | 4.09e-01 | 9.27e-16 | -0.543 | 0.74 |
| Horseshoe | control | 2.31e-13 | 1.44e-03 | 3.70e-01 | 6.75e-16 | -0.761 | 0.65 |
| Horseshoe | seed | 2.19e-13 | 2.06e-03 | 5.44e-01 | 9.47e-16 | -1.096 | 0.69 |
