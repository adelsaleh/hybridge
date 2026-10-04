# High-mode disk diocotron screening: m=64 and m=128

Date: 2026-09-13. IMEX-ARK3, h=0.008, DG order 6, 113,894 triangles, float64. Three runs completed 100 steps each with dt=0.5 and T=50: the two perturbed cases and the m=64 control. The m=128 control exhausted initialization recovery before time integration. No build or installation was run; runtime JIT was used under the user's authorization.

The m=64 Gaussian case is the better analysis candidate. The m=128 Gaussian case is a useful resolution stress test, with a contaminated modal signal and substantial negative density. Neither has been qualified for nonlinear coherent-structure counts.

## Zoni–Güçlü analytical growth rate

[Zoni & Güçlü, section 6.3, equation (34)](https://arxiv.org/abs/1909.05005) gives, for a unit disk and a unit-density sharp annulus,

\[
(\omega/\omega_D)^2-b_m(\omega/\omega_D)+c_m=0,\qquad \omega_D=1/2,
\]

\[
q=r_-/r_+,\quad a=1-q^2,\quad
b_m=ma+r_+^{2m}-r_-^{2m},
\]

\[
c_m=ma(1-r_-^{2m})-(1-q^{2m})(1-r_+^{2m}),\qquad
\gamma_m=\frac14\sqrt{4c_m-b_m^2}
\]

when the discriminant is negative; otherwise the analytical exponential growth rate is zero. The unstable branch has Re(omega)=b_m/4. These are amplitude growth rates; squared amplitudes grow with exponent 2 gamma.

Modes 64 and 128 are stable on the original r_-=0.45, r_+=0.50 sharp annulus. To make high modes unstable, the new rings are centered at 0.8 and use q=sqrt(1-1.59/m). The analytical spectrum is evaluated at the resulting radii, rather than reusing the paper's mode-9 rate.

| Seeded mode | Inner radius | Outer radius | Sharp-layer gamma | Sharp unstable modes | Sharp fastest mode |
|---|---:|---:|---:|---|---:|
| 64 | 0.794968553459 | 0.805031446541 | 0.1986490623 | 2–102 | 64 |
| 128 | 0.797500073855 | 0.802499926145 | 0.1999249921 | 2–205 | 128 |

Re(omega)=0.3975000000 for both selected modes, to the displayed precision. The formula was checked independently against the two-interface radial Green-function eigenproblem.

## Smooth initial profile and radial resolution

The tested initial condition is

\[
\rho(r,\theta,0)=\exp[-((r-0.8)/d)^2]\,[1+10^{-4}\cos(m\theta)],
\qquad d=(r_+-r_-)/2.
\]

It is an untruncated Gaussian (radial power 2), independent of DG order 6. The sharp-annulus formula is therefore a separate comparison, not an exact reference for this smooth equilibrium. The numerical radial reference uses the existing Green-function Nyström eigenproblem at 128, 256, 512, and 1,024 total radial nodes. The last growth-rate changes are 0.0147% for m=64 and 0.0287% for m=128.

| Mode | Gaussian reference gamma | Gaussian Re(omega) | Width | Cells across nominal ring | Cells per angular wavelength | Initial relative projection L2 error | Initial checked minimum |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.1839994596 | 0.3530890000 | 0.0100628931 | 1.26 | 9.83 | 3.591e-5 | -1.713e-6 |
| 128 | 0.1850756901 | 0.3526900862 | 0.0049998523 | 0.63 | 4.91 | 2.763e-3 | -2.101e-2 |

The geometric counts use the 95th percentile of actual cell diameters near the ring. They are not resolution guarantees. Overintegration uses 32x32 initial projection quadrature and 40x40 independent error quadrature. A radial-power-4 comparison had projection errors 1.780e-3 and 3.245e-2, and minima -0.04048 and -0.4443 respectively; this motivated the extra smoothing.

An exploratory smooth full-band scan at 256 radial nodes places the largest growth near modes 78 and 157 for the two respective rings. Thus smoothing also shifts the preferred angular mode. This scan is not a certified near-neutral stability cutoff; only the separately refined target rates above meet the reference refinement check.

## Measured instability and invariants

The fit uses all 61 samples in the predeclared window 20 <= t <= 50. End-of-window harmonic generation and competing modes prevent treating a close slope alone as a qualification pass.

| Observable | m=64 | m=128 |
|---|---:|---:|
| Fitted target-mode gamma | 0.1812491766 | 0.1762778061 |
| Difference from smooth gamma | -1.4947% | -4.7537% |
| Difference from sharp-layer gamma | -8.7591% | -11.8280% |
| Fit R² | 0.99976265 | 0.99836402 |
| Whole-potential norm gamma | 0.1819594461 | 0.1698762257 |
| Frequency from unwrapped target phase | 0.3532676071 | 0.3527260181 |
| Largest competing nonharmonic mode / target in fit window | 0.1278% (mode 105) | 32.6500% (mode 179) |
| Final relative mass drift | 1.2383e-15 | 6.2309e-16 |
| Final relative energy drift | -0.0014382% | -0.0022078% |
| Final relative enstrophy drift | -0.2703823% | -2.4612272% |
| Worst sampled density, all stages | -0.0578594 | -0.6152722 |
| Worst cell average, all stages | -0.0002046 | -0.0079868 |

Energy is 0.5 integral |q_h|²; enstrophy is 0.5 integral rho_h². Positivity is measured without a limiter. Negative samples prove violations, while negative Bernstein bounds alone do not. Stage extrema include 400 checks per completed run.

Both perturbed runs retained tau=128000, took 300 transport and 400 time-step Poisson solves, and recorded no tau retries. The solver success does not establish positivity or discretization accuracy. Smaller-dt runs, half-seed comparisons, and spatial/modal-quadrature refinement remain open.

For m=64, the fixed subwindows 20–35 and 35–50 give gamma=0.18386631 and 0.17589875. The second harmonic reaches 15.56% of the target near the end of the full window. The earlier subwindow supports good linear growth, but does not replace the predeclared full-window result.

## Controls and the remaining rank issue

The unperturbed m=64 control completed 100 steps. Its target-mode amplitude stays below 0.03555% of the perturbed amplitude over 20–50. However, initial trace-rank recovery changed tau from 128000 through 256000 and 512000 to 1024000. This is not a fixed-tau matched control and is marked provisional by the analyzer.

The m=128 control failed before the initial explicit residual could be accepted. Four Poisson-tau increases reached 2048000; the final reported edge 76260 had only six active inflow quadrature nodes for seven trace degrees of freedom. The failure and every retry are preserved in the recovery JSONL and console log. No control growth rate can be extracted. No residual was silently accepted and no limiter, clipping, or altered trace rule was introduced.

This distinguishes the two uses: m=64 is the stronger quantitative candidate; m=128 probes the present radial-resolution and trace-robustness limits. Increasing the seeded angular mode does not guarantee that the nonlinear flow produces or preserves that many coherent vortices.

## User-run heavy cases

Each new preset uses h=0.008, DG p=6, dt=0.05, 1,400 steps (T=70), tau=128000, radial power 2, seed 1e-4, and every-step/stage diagnostics. These smaller-dt heavy runs have not been executed here. Use the existing runner for identical terminal/log verbosity and cached device solves.

```bash
# from the repository root
source .env

# Better resolved high-mode analysis candidate.
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m64_ark3_p6_h008_dt005_t70.args \
  --verbosity 3 --plot-width 2048 --plot-height 2048

# Very-high-mode radial-resolution stress case.
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m128_ark3_p6_h008_dt005_t70.args \
  --verbosity 3 --plot-width 2048 --plot-height 2048
```

Record distinct prefixes when changing h, p, dt, epsilon, or tau. A future fixed-tau m=64 control comparison should repeat both perturbed and unperturbed runs with the same initial tau (1024000 is the value reached by the successful control here). The existing m=128 control failure must be resolved before claiming a complete noise-floor check for that case.

Compute the article's prediction independently of time integration:

```bash
.venv/bin/python -m scripts.guiding_center.diagnostics.diocotron_reference \
  --preset diocotron_gaussian_m128_ark3_p6_h008_dt005_t70 \
  --output artifacts/diocotron_m128_reference.json
```

Analyze a completed run, retaining the sharp and smooth references separately:

```bash
.venv/bin/python -m scripts.guiding_center.diagnostics.analyze_diocotron \
  run_outputs/guiding_center/diocotron_gaussian_m128_ark3_p6_h008_dt005_t70.jsonl \
  --output-dir artifacts/diocotron_m128_full_analysis
```

## Implementation and validation

The existing cached polar sampler and FFT now retain all angular modes 1..3m, rather than only low modes and three selected harmonics. The analyzer flags competing nonharmonic modes, includes the worst competitor in its plots, and reports the sharp-layer comparison explicitly. The existing smooth reference adds one radial refinement level automatically for high modes. No transport or Poisson kernels were rewritten.

CPU analytical/polynomial/synthetic-data checks: 19 passed, 3 deselected. They include the independent interface spectrum check and detection of competing-mode contamination with a weak second harmonic. Actual device diagnostics were exercised in the three completed 100-step runs. Full h/dt convergence, nonlinear mode-shape qualification, and a positivity-preserving discretization are not claimed.

Raw run data, reference spectra, control recovery logs, and plots are stored in `artifacts/diocotron_high_modes_20260913`. `comparison.json` records input hashes. The individual analysis directories retain the complete fit results and qualification notes.

Validation note: all new study/report links and documented paths passed scoped checks. The repository-wide documentation link check reports nine pre-existing missing older transport/temporal-CFL/diocotron artifacts. An additional pre-existing star-hole screenshot link in run_configs/README.md is also missing. These unrelated links were left unchanged. The repository-wide documented-script/path check passed.
