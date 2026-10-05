# Static Study: KKT Positivity Projection for Guiding-Center Densities (2026-10-05)

This note records the static evidence behind the
[positivity plan](../../development/plans/positivity_kkt_bdf2.md). No time
integration was run. The projections were applied to density checkpoints that
earlier unconstrained SI-BDF2 recordings had written, and to L2 projections of
a known function.

Reproduce both parts with
`scripts/guiding_center/diagnostics/positivity_projection_study.py`:

```bash
python -m scripts.guiding_center.diagnostics.positivity_projection_study states \
  outputs/readme_showcase/chk_positive_global_h005_dt0015625.restart.npz \
  outputs/readme_showcase/chk_positive_global_h005_dt00078125.restart.npz \
  outputs/readme_showcase/chk_positive_global_h005_dt000390625.restart.npz \
  outputs/readme_showcase/positive_c5_dt000390625.restart.npz --workers 22
python -m scripts.guiding_center.diagnostics.positivity_projection_study order --workers 22
```

The checkpoints are local and untracked (`outputs/readme_showcase/`). They hold
the positive guiding-center showcase (star domain, 360,379 triangles, p = 6,
`dub_orth` basis) at t = 0.5 for three time steps and at t = 6.4.

## Where the negative values come from

Positivity is checked at the set S of each element: the 33 volume quadrature
points plus the 28 points of the equispaced p-lattice (vertices, edge and
interior nodes), 61 points in all.

| State | Elements negative at S | Negative cell mean | min at S | Negative mass / mass |
|---|---:|---:|---:|---:|
| t = 0.5, dt = 0.0015625 | 37,719 | 16,869 | −0.815 | 2.4e-4 |
| t = 0.5, dt = 0.00078125 | 34,974 | 15,032 | −0.250 | 1.0e-5 |
| t = 0.5, dt = 0.000390625 | 35,043 | 15,001 | −0.0017 | 6.2e-9 |
| t = 6.4, dt = 0.000390625 | 118,621 | 26,964 | −6.09 | 4.2e-3 |

The undershoot depths separate three sources:

- **Temporal.** Elements below −1e-3 at t = 0.5 fall from 5,228 to 833 to 4 as
  dt halves twice. The SI-BDF2 history term (4ρⁿ − ρⁿ⁻¹)/3 is negative wherever
  ρⁿ⁻¹ > 4ρⁿ, just behind a filament moving into near-zero density. No monotone
  implicit operator can repair a negative source.
- **Spatial.** About 7,000 elements between −1e-6 and −1e-3 do not change with
  dt. This is ringing of the high-order DG solution at filament and blob tails.
- **Background.** About 25,000 elements are within 1e-6 of zero, among the
  roughly 79,000 near-zero background elements.

By t = 6.4 the unconstrained run has accumulated 94,000 elements below −1e-3.
About 40% of the flagged elements have a negative cell mean. A
cell-average-preserving scaling limiter (Zhang–Shu) therefore cannot be the
whole answer: some mass has to move between elements.

## Projection applied to the recorded states

The projection, element by element, is:

- for a nonnegative mean, the mass-matrix-closest polynomial that is
  nonnegative at S and keeps the element mean (exact local conservation);
- for a negative mean, the closest polynomial nonnegative at S, without a mean
  constraint.

The total mass this adds is then returned by one global multiplicative factor,
which keeps the signs at S.

| State | ‖Δρ‖/‖ρ‖, KKT | ‖Δρ‖/‖ρ‖, Zhang–Shu | Mass returned | min at S after | CPU (22 workers) |
|---|---:|---:|---:|---:|---:|
| t = 0.5, dt = 0.0015625 | 3.44e-3 | 3.81e-3 | 2.1e-4 | −1.0e-13 | 0.46 s |
| t = 0.5, dt = 0.00078125 | 5.36e-4 | 7.35e-4 | 6.2e-6 | −5.5e-15 | 0.45 s |
| t = 0.5, dt = 0.000390625 | 1.63e-7 | 2.92e-6 | 3.2e-9 | −6.3e-17 | 0.46 s |
| t = 6.4, dt = 0.000390625 | 2.44e-2 | 3.45e-2 | 1.6e-3 | −1.5e-13 | 0.91 s |

The Zhang–Shu column applies Zhang–Shu scaling to elements with a nonnegative
mean and zeros the others, with the same mass return.

- **Mass and positivity.** Global mass is preserved to round-off (≤ 6e-16
  relative), and positivity at S holds to round-off.
- **Correction stays below the scheme's error.** At t = 0.5 the correction
  shrinks with dt as fast as the undershoots. It stays below the
  unconstrained scheme's own time-discretization error (2.2% and 0.75% density
  difference between successive dt levels in the
  [showcase checks](../../getting_started/gpu_showcase.md#time-step)).
- **Smaller than Zhang–Shu.** The KKT correction is 1.1–18 times smaller than
  Zhang–Shu scaling, most clearly where the solution is accurate.
- **Points outside S.** Points not in S can stay slightly negative: −9.4e-4 on
  the 120-point (2p+2)-lattice at the finest t = 0.5 state. Adding that
  lattice to S makes those points exact (−2e-16). It costs about three times
  the element work and flags almost no extra elements.

## Element solver

With 61 points and 27 free modes (28 without the mean constraint), the dual
Gram matrix is rank-deficient. At the optimum an element has 14–20 active
constraints on average and up to all 61. These are near-zero background
elements whose projection is essentially their constant mean.

- **Plain semismooth Newton (primal-dual active set).** It cycles on 57–73%
  of the flagged elements of the three t = 0.5 states.
- **Regularizing the dual (Gram matrix + δI).** On a 400-element sample of the
  dt = 0.0015625 state it still cycles on 18–62% of elements for δ ≤ 1e-4.
  At δ = 1e-3 it stops 4% (median) above the optimum.
- **Bound-constrained Newton on the 28 lattice values.** It is robust (2.5
  iterations on average, at most 6), but leaves −0.78 at the quadrature
  points.
- **Lawson–Hanson NNLS solution of the least-distance problem.** This is an
  active-set Newton method with a step-length safeguard. It is exact and
  finite, and takes 0.12 ms per element in serial Python.

Elements whose mean is negligible, below 1e-8 of their point values, take the
constant mean, which is the only feasible polynomial. Zhang–Shu scaling serves
as the always-feasible fallback.

## Spatial order of the projection

The test density is f = sin²(3(x² + y²) + x) on the unit square. It is
nonnegative, not a polynomial, and touches zero along curves.

| p | nx | L2 projection | Point KKT (S) | Bernstein coefficients ≥ 0 |
|---|---:|---:|---:|---:|
| 3 | 16 | 9.16e-4 | 9.30e-4 | 1.05e-2 |
| 3 | 32 | 5.98e-5 (3.9) | 6.01e-5 (4.0) | 1.98e-3 (2.4) |
| 3 | 64 | 3.78e-6 (4.0) | 3.81e-6 (4.0) | 3.44e-4 (2.5) |
| 6 | 8 | 1.34e-4 | 1.42e-4 | 1.87e-2 |
| 6 | 16 | 1.26e-6 (6.7) | 1.28e-6 (6.8) | 3.40e-3 (2.5) |
| 6 | 32 | 1.02e-8 (6.9) | 1.02e-8 (7.0) | 6.05e-4 (2.5) |

Observed rates are in parentheses.

- **Point constraints keep the order.** They keep the p + 1 order and stay
  within 1–6% of the plain projection. This also holds with 472 points per
  element at p = 6 (rate 7.0).
- **Bernstein constraints do not.** Positivity everywhere on the element via
  Bernstein coefficients ≥ 0 drops to order ≈ 2.5 for any p. Near a zero of
  the density the Bernstein coefficients are negative by O(h²).

## Limits

- **No time integration.** The temporal order of SI-BDF2 with the projection
  in the loop is untested; the plan makes it a gate.
- **Inflated counts.** The flagged-element counts come from unconstrained
  trajectories in which violations accumulate. With the projection active from
  t = 0, per-step counts should be smaller and must be measured.
- **CPU timings.** The timings are for a Python/SciPy prototype on the CPU, not
  for the device implementation.
