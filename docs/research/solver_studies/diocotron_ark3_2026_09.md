# Disk diocotron: smooth-first IMEX-ARK3 qualification

The first case is a smooth positive annulus with a small mode-9 perturbation.
The steeper Zoni–Güçlü profile is retained as a later test. Mesh adaptivity is a
future option for the thin interfaces and nonlinear filaments, not part of this
change. No positivity limiter has been added.

Reference: [Zoni & Güçlü, arXiv:1909.05005, §6.3](https://arxiv.org/abs/1909.05005).
The paper uses inner/outer radii 0.45/0.50, mode 9, perturbation 1e-4, radial
power 50, and fits the potential perturbation amplitude over approximately
t=20–50. Its equation (34) gives gamma=0.1796309594 and Re(omega)=0.4275008105.
Modes 2–13 are unstable for that sharp annulus. Equation (35), implemented
literally, cuts the super-Gaussian off at the two radii; its limiting value just
inside either cutoff is exp(-1). It is not a globally smooth compact profile.

## First case and references

The smooth case uses

```
rho0(r) = exp(-((r - 0.475)/0.025)^4)
rho(0,r,theta) = rho0(r) * (1 + 1e-4*cos(9*theta))
```

There is no radial cutoff. Radial power 4 describes the analytic profile; DG
polynomial order remains 6. Potential is zero on the unit-circle wall and the
transport boundary has zero flux.

Changing the radial profile changes its linearized operator. The sharp-annulus
formula remains a useful comparison but is not an exact reference for the
smooth profile. We derive a separate radial eigenproblem from the same disk
Poisson/transport equations, using `exp(i*(m*theta-omega*t))`:

```
omega*rho_m = m*Omega(r)*rho_m + m*rho0'(r)/r * phi_m
Omega(r) = integral_0^r s*rho0(s) ds / r^2
phi_m(r) = integral_0^1 G_m(r,s)*rho_m(s)*s ds
G_m(r,s) = ((min(r,s)/max(r,s))^m - (r*s)^m)/(2*m)
```

The reference uses Gauss Nyström quadrature with four radial segments. For the
smooth mode-9 case, 128, 256, and 512 total nodes give growth rates 0.17947914,
0.17961693, and 0.17964991. The last change is 0.0184%; the final rotation
frequency is approximately 0.39305972. These are converged numerical radial
references, not closed-form analytical rates. Tests reproduce both the analytic
radial Poisson solution for a polynomial source and the paper's dispersion
relation from the two sharp surface sheets.

## Implemented measurements

- The existing full-domain `||phi-phi_eq||_L2`, plus radial L2 norms of individual
  angular Fourier modes. The latter prevent axisymmetric equilibrium drift and
  other modes from being mistaken for growth of the seeded mode. Polar sampling
  covers the mesh's recorded inscribed-circle radius; the whole-domain norm
  still covers the full polygon.
- Fits of log **amplitude**, not squared amplitude, over the declared complete
  window. The default also reports the two fixed subwindows 20–35 and 35–50,
  R², the 2m/m harmonic ratio, and agreement with the appropriate reference.
  Incomplete or invalid data never produce a successful growth fit.
- Initial, every-ARK-stage, and accepted-state positivity measurements. Volume
  quadrature plus a denser reference lattice including vertices/edges supplies
  negative witnesses. Bernstein coefficients bound each complete element
  polynomial to the stated floating-point tolerance. A negative Bernstein lower
  bound alone is inconclusive. The negative mass/L2 are quadrature estimates.
- Mass, energy `0.5*integral(|q_h|^2)`, and enstrophy
  `0.5*integral(rho_h^2)`, with drift from the numerical initial state. These
  conservation figures do not hide initialization error.
- Immediate initial-projection and tau-recovery records, including on failed
  runs; all stage checks and solve counts in the existing timing JSONL. Solver
  logs and verbosity 0–3 use the existing runner and terminal tee.

A 5% rate band, R² >= 0.995, control contribution <= 10%, and second harmonic
<= 10% are diagnostic gates, not a proof of convergence. Matched unperturbed,
half-seed, smaller-timestep, and spatial/diagnostic-quadrature refinement checks
are required before drawing accuracy conclusions. The analyzer refuses
incompatible control configurations and flags tau changes. It never subtracts
scalar amplitudes as a substitute for subtracting complex fields.

## Initialization and device reuse

`hdgfem.assembly.projection.project_callable` now exposes the package's existing
host/device projection with an optional richer initialization rule. The supplied
presets use 32x32 quadrature for initial and equilibrium density only. The
transport and Poisson quadrature, matrices and cache policy remain fixed.
The device projection caches the reference projection operator and mesh data;
mapped coordinates are processed in bounded batches. It does not keep a large
whole-mesh overintegration table alive throughout evolution.

The additional Fourier diagnostics reuse `RasterGeometry` and
`DeviceRasterSampler`. Their sampling map, equilibrium samples, radial weights,
and FFT size are primed once. Positivity caches its basis maps and reductions;
changing coefficients stay on the GPU and only small scalar results are copied
back. Mesh geometry, the fixed reference-basis conversion, and the independent
small radial eigenproblem are host setup/analysis work. Transport and Poisson
solves remain on the configured device backends with their existing nearest-time
warm starts and operator reuse.

## Measured screening results

The agent ran no more than 100 steps in an individual test and stopped GPU work
before the user's qualitative heavy run. The following screening runs use
**dt=0.5**, h=0.008, DG order 6, 113,894 triangles, T=50. They are not the supplied
full accuracy runs (dt=0.05, T=70).

| Quantity | Steep paper profile | Smooth radial power 4 |
|---|---:|---:|
| Initial relative L2 projection error, richer quadrature | 3.19e-2 | 7.71e-8 |
| Initial checked minimum | -0.755 | -1.36e-8 |
| Target-mode fitted growth, t=20–50 | 0.1784700 | 0.1778444 |
| Difference from applicable reference | -0.646% | -1.005% |
| Final relative mass drift | 9.82e-15 | 9.44e-15 |
| Final relative energy drift | -7.16e-5 | -3.41e-5 |
| Final relative enstrophy drift | -3.31e-2 | -2.17e-3 |
| Tau recovery events | 2 | 0 |

The smooth run retained tau=128,000. Its predeclared 20–35 subwindow fitted
0.179426 (0.125% below the radial reference); the 35–50 fit was 0.174590,
consistent with late-window curvature that requires the half-seed control.
The maximum potential 2m/m ratio in the full window was 0.0940. No unperturbed
or half-seed evolution was run by the agent, so mesh-noise/linearity qualification
remains open.

Strict positivity fails even for the smooth run. Across 400 stage/endpoint
checks, the worst value was -0.001277 and the minimum cell average was -0.000874;
maximum quadrature negative mass was 5.64e-6. A future conservative positivity
capability would need to handle negative cell averages as well as nodal
undershoots; simple pointwise clipping would change mass and the method.

Raw measurements, input hashes, plots, and fit reports are in
`../../../artifacts/diocotron_validation/` (`artifacts/diocotron_validation/`, local, untracked):
`smooth_screening/report.md`, `sharp_screening/report.md`,
`smooth_projection.json`, and `projection_q32.json`.

## User-run commands

From the repository root, in a new terminal if CUDA libraries need setup:

```bash
export CUDA_PATH=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=~/src/AMGX-build-cuda13:~/src/AMGX-install-cuda13/lib:/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HDGFEM_PRECISION=float64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

First qualitative run, h=0.008, DG p=6, dt=0.05, T=70, tau initially 128,000:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_smooth_m9_ark3_p6_h008_dt005_t70.args \
  --verbosity 3
```

Run matched controls and timestep refinement separately, after the qualitative
run. The controls use the same initial Poisson tau and projection rule.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_smooth_m9_control_ark3_p6_h008_dt005_t70.args \
  --verbosity 2

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_smooth_m9_half_seed_ark3_p6_h008_dt005_t70.args \
  --verbosity 2

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_smooth_m9_ark3_p6_h008_dt005_t70.args \
  --dt 0.025 --num-steps 2800 --plot-every 0 --verbosity 2 \
  --diagnostics-prefix diocotron_smooth_m9_dt0025

.venv/bin/python -m scripts.guiding_center.diagnostics.analyze_diocotron \
  run_outputs/guiding_center/diocotron_smooth_m9_ark3_p6_h008_dt005_t70.jsonl \
  --control run_outputs/guiding_center/diocotron_smooth_m9_control_ark3_p6_h008_dt005_t70.jsonl \
  --half-seed run_outputs/guiding_center/diocotron_smooth_m9_half_seed_ark3_p6_h008_dt005_t70.jsonl \
  --refined run_outputs/guiding_center/diocotron_smooth_m9_dt0025.jsonl \
  --output-dir artifacts/diocotron_smooth_comparison
```

An analysis can also run on a single completed file; missing controls are stated
explicitly. Use distinct output prefixes for each h/p/dt or smoothing change.
To assess Fourier integration, double `--diocotron-radial-points` and
`--diocotron-angular-points` in a matched repeat.

## Hardening and the high-mode ladder

First increase radial power through 6, 10, 20, then 50 with `truncate=false`,
using the matching smooth radial reference each time. The literal paper profile
is available as `diocotron_zg_m9_ark3_p6_h008_dt005_t70.args`; it starts at tau
32,000 and may trigger recovery. Its control, half-seed, and sharp-theory stable
mode-14 presets are also provided.

Mode 64 or 128 on the original sharp annulus is stable. High-mode candidates
therefore use a thinner ring centered at radius 0.8, with
`m*(1-(r_in/r_out)^2)≈1.59`. This is a design estimate; smoothing changes the
spectrum and the analyzer computes its separate reference.

Static smooth-profile checks on h=0.008, p=6 gave:

| Mode | Ring width | Cells across ring | Cells per angular wavelength | Relative projection L2 error | Checked minimum |
|---|---:|---:|---:|---:|---:|
| 32 | 0.020385 | 2.55 | 19.65 | 3.10e-5 | -4.02e-4 |
| 64 | 0.010063 | 1.26 | 9.83 | 1.78e-3 | -4.05e-2 |
| 128 | 0.005000 | 0.63 | 4.91 | 3.24e-2 | -4.44e-1 |

These counts use the 95th percentile of actual cell diameters near the ring;
they do not claim one degree of freedom equals one resolved physical scale.
Start with m=32 after the mode-9 checks. Treat m=64 as a resolution experiment;
m=128 already has substantial projection error/undershoots. None has been
qualified for nonlinear coherent-structure counts. Ring-focused refinement and
subsequent filament refinement are more promising than increasing angular mode
alone. Separate provisional response files exist for all three modes.

The static check uses no Poisson solve or time integration:

```bash
.venv/bin/python -m scripts.guiding_center.diagnostics.preflight_diocotron \
  --radial-power 4 --no-truncate --mesh-size 0.008 --order 6 \
  --modes 9 32 64 128 --initial-projection-quad-1d 32 \
  --error-quadrature 40 --output artifacts/diocotron_smooth_projection.json
```

## High-mode follow-up

The [m=64 and m=128 Gaussian screening](diocotron_high_modes_2026_09.md) records
the article-based spectra, smoother-profile references, bounded evolution tests,
control failures, and current heavy-run commands.
