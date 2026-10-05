# Order Gates: KKT-Projected SI-BDF2 on Manufactured Problems (2026-10-05)

This note records the Phase 2 order gates of the
[positivity plan](../../development/plans/positivity_kkt_bdf2.md). The question
is whether projecting every transported density onto nonnegativity changes the
convergence order of SI-BDF2. The static evidence is in the
[static study](positivity_projection_static_2026_10_05.md).

All runs are small host runs on `rectangle_mesh(n, n)` of [−1, 1]², with at
most 8,192 elements and the `dub_orth` basis. Each run uses the production
pieces:

- the `SIBDF2Stepper`, with its SI-Euler startup step and the
  `density_projector` hook;
- the guiding-center runner's transport stage, which is host HDG transport
  with Numba assembly and PyPardiso on 16 verified MKL threads, with the exact
  density as inflow data;
- the host `DensityPositivityProjector`, which matches the device kernel to
  2e-16.

Reproduce the gates with
`scripts/guiding_center/diagnostics/positivity_order_gates.py`:

```bash
python -m scripts.guiding_center.diagnostics.positivity_order_gates time --case rotation --order 6 --cells 32
python -m scripts.guiding_center.diagnostics.positivity_order_gates space --order 3 --cells 8 16 32 64
python -m scripts.guiding_center.diagnostics.positivity_order_gates space --order 6 --cells 8 16 32 64
python -m scripts.guiding_center.diagnostics.positivity_order_gates time --case vortex --order 6 --cells 32
```

## Problems

Every density is smooth and nonnegative. Its zero set is a circle of
quadratic contact, inside a Gaussian far field that is nearly zero. The
unconstrained discrete solutions therefore undershoot at both kinds of places.

- **Rotation.** The bump ((q − a²)/a²)² exp(−q/a²), with q = |x − (0.3, 0)|²
  and a = 0.12, rotates rigidly under β = (−y, x).
  - A Poisson stand-in returns the fixed flux q = (x, y), so the stepper's
    drift dt (−q_y, q_x) is the rotation.
  - The exact solution is the rotated bump.
- **Steady ring.** The radial ring ((u² − b²)/b²)² exp(−u²/b²), with
  u = r² − 0.25 and b = 0.1, rotates under the same β.
  - It is an exact steady solution, so no temporal error enters.
  - The projection still corrects the spatial undershoots every step.
- **Translating vortex.** This is a coupled guiding-center solution.
  - The density is ρ = 10 ((r² − a²)/a²)² exp(−r²/a²), with a = 0.12, about
    a centre moving from (−0.15, 0) at speed U = 0.3.
  - The potential is φ = Φ(r) + U y. The radial Φ solves −ΔΦ = ρ and is
    written in closed form with the entire exponential integral Ein.
  - The drift of φ is the uniform translation plus the vortex's own
    azimuthal drift, which does not advect a radial density.
  - The real HDG Poisson solver (Numba, τ = 1, exact Dirichlet data) supplies
    the drift every step. The test therefore includes the extrapolated
    velocity 2vⁿ − vⁿ⁻¹ and the drift computed from the projected density.

## Time gate: rotation, p = 6, 2,048 triangles, T = 1

| N | Error, none | Rate | Error, KKT | Rate | KKT / none | Worst min before KKT |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 4.169e-1 | | 4.167e-1 | | 0.9996 | −6.1e-3 |
| 20 | 2.018e-1 | 1.05 | 2.013e-1 | 1.05 | 0.9972 | −1.2e-2 |
| 40 | 6.663e-2 | 1.60 | 6.618e-2 | 1.60 | 0.9932 | −4.6e-3 |
| 80 | 1.807e-2 | 1.88 | 1.796e-2 | 1.88 | 0.9939 | −1.3e-3 |
| 160 | 4.608e-3 | 1.97 | 4.590e-3 | 1.97 | 0.9961 | −3.0e-4 |

Errors are relative L2 errors against the exact density at T. The projector
flags 850–1,100 of the 2,048 elements every step.

- The worst undershoot and the largest correction (‖Δρ‖/‖ρ‖ from 4.2e-3 at
  N = 10 to 3.0e-5 at N = 160) shrink like dt².
- From N = 80 on, no mass needs returning, because negative element means
  have disappeared.
- Differences of successive-dt solutions give the same rates for both runs:
  0.85, 1.53 and 1.86.
- The projection keeps BDF2's second order, and its error is never larger
  than the unconstrained one.

## Space gate: steady ring, 20 steps of dt = 0.05

| p | Cells | Error, none | Rate | Error, KKT | Rate | KKT / none | Flagged per step | Worst min before KKT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 8 | 3.44e-1 | | 3.79e-1 | | 1.10 | 104 / 128 | −0.75 |
| 3 | 16 | 9.00e-2 | 1.94 | 9.19e-2 | 2.04 | 1.02 | 346 / 512 | −7.0e-2 |
| 3 | 32 | 7.79e-3 | 3.53 | 8.19e-3 | 3.49 | 1.05 | 1,122 / 2,048 | −1.7e-2 |
| 3 | 64 | 3.99e-4 | 4.29 | 4.06e-4 | 4.33 | 1.02 | 3,745 / 8,192 | −7.7e-4 |
| 6 | 8 | 1.11e-1 | | 1.21e-1 | | 1.10 | 108 / 128 | −0.21 |
| 6 | 16 | 3.52e-3 | 4.98 | 3.77e-3 | 5.01 | 1.07 | 322 / 512 | −4.2e-3 |
| 6 | 32 | 4.40e-5 | 6.32 | 4.44e-5 | 6.41 | 1.01 | 944 / 2,048 | −4.8e-5 |
| 6 | 64 | 3.913e-7 | 6.81 | 3.914e-7 | 6.83 | 1.0002 | 2,744 / 8,192 | −4.5e-8 |

The projected runs keep the unconstrained h rates, which approach p + 1. The
largest correction falls from 1.2e-2 to 8.0e-10 at p = 6.

The KKT error can exceed the unconstrained one slightly here, by 10% on the
coarsest meshes and by 0.02% at p = 6 on 64 × 64 squares. The L2 projection of
the exact density itself undershoots at the points, so it lies outside the
constraint set. Non-expansiveness therefore bounds the KKT error only by the
unconstrained error plus that O(h^(p+1)) distance. This changes the constant,
not the order.

## Coupled time gate: translating vortex, p = 6, 2,048 triangles, T = 1

| N | Error, none | Rate | Error, KKT | Rate | KKT / none | Worst min before KKT |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 4.290e-1 | | 4.293e-1 | | 1.0008 | −6.3e-2 |
| 20 | 1.923e-1 | 1.16 | 1.920e-1 | 1.16 | 0.9982 | −1.3e-1 |
| 40 | 5.983e-2 | 1.68 | 5.963e-2 | 1.69 | 0.9968 | −3.6e-2 |
| 80 | 1.575e-2 | 1.93 | 1.571e-2 | 1.92 | 0.9974 | −9.6e-3 |
| 160 | 3.988e-3 | 1.98 | 3.982e-3 | 1.98 | 0.9986 | −1.6e-3 |

The density peaks at 10, so the worst undershoot is 1.3% of the peak. The
projector flags 910–980 of the 2,048 elements every step.

- The largest correction falls from 4.2e-3 to 1.7e-5.
- Mass is returned only at the coarse steps (8.5e-4 at N = 10), and none from
  N = 80 on.
- Without the projection, the final density still dips to −8.5e-3 at
  N = 160.
- Differences of successive-dt solutions give the rates 1.00, 1.65 and 1.92
  for both runs.
- The coupled scheme keeps its second order with the projection. That covers
  the drift computed from projected densities and the extrapolated velocity.

## Conclusion

On these manufactured problems, projecting every transported density keeps
SI-BDF2's second order in time, alone and coupled to the HDG Poisson drift. It
also keeps the p + 1 order in space. The corrections shrink at the rate of the
discretization error they repair: dt² in time and h^(p+1) in space.

The full-size positive-case check of the plan has not been run. It repeats
the three-dt check to t = 0.5 at 360,379 triangles on the GPU and is the
user's to run:

```bash
python -m scripts.reports.record_gpu_showcase --name chk_positive_kkt_h005_dt0015625 \
  --strength-mode positive --profile run_configs/guiding_center/profiles/readme_positive_c5_h005.npz \
  --cutoff 5 --h 0.005 --dt 0.0015625 --steps 320 --every 32 --seconds 7200 \
  --max-loss 0.9 --poisson-tau global --density-positivity kkt

python -m scripts.reports.record_gpu_showcase --name chk_positive_kkt_h005_dt00078125 \
  --strength-mode positive --profile run_configs/guiding_center/profiles/readme_positive_c5_h005.npz \
  --cutoff 5 --h 0.005 --dt 0.00078125 --steps 640 --every 32 --seconds 7200 \
  --max-loss 0.9 --poisson-tau global --density-positivity kkt

python -m scripts.reports.record_gpu_showcase --name chk_positive_kkt_h005_dt000390625 \
  --strength-mode positive --profile run_configs/guiding_center/profiles/readme_positive_c5_h005.npz \
  --cutoff 5 --h 0.005 --dt 0.000390625 --steps 1280 --every 32 --seconds 7200 \
  --max-loss 0.9 --poisson-tau global --density-positivity kkt

python -m scripts.reports.compare_showcase_states --output positive_kkt_dt.json --series \
  outputs/readme_showcase/chk_positive_kkt_h005_dt0015625.restart.npz \
  outputs/readme_showcase/chk_positive_kkt_h005_dt00078125.restart.npz \
  outputs/readme_showcase/chk_positive_kkt_h005_dt000390625.restart.npz

# Unconstrained checkpoints of the same check, from the static study.
python -m scripts.reports.compare_showcase_states --output positive_kkt_vs_none.json \
  --pair <unconstrained>/chk_positive_global_h005_dt0015625.restart.npz outputs/readme_showcase/chk_positive_kkt_h005_dt0015625.restart.npz \
  --pair <unconstrained>/chk_positive_global_h005_dt00078125.restart.npz outputs/readme_showcase/chk_positive_kkt_h005_dt00078125.restart.npz \
  --pair <unconstrained>/chk_positive_global_h005_dt000390625.restart.npz outputs/readme_showcase/chk_positive_kkt_h005_dt000390625.restart.npz
```
