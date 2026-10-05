# Run Configs

This directory stores small, version-controlled benchmark and solver presets.
The goal is to keep reproducible commands and the best observed settings close
to the code, without committing large console logs.

Recommended practice:

- Store the exact CLI arguments, not just a prose description.
- Record the mesh/problem parameters that make runs comparable.
- Record enough result metrics to rank the configuration: total time, global
  solve time, iterations, residual, and error.
- Keep failed or unavailable solver attempts when they explain why a path is
  not currently used.
- Treat timings as machine-local observations. Re-run the benchmark after major
  assembly, solver, PETSc, SciPy, or hardware changes.

Current files:

- `diff_rea_p6_lc01_solver_benchmarks.json`: archived solver sweep for
  `scripts/diffusion_reaction/experiments/bootstrap_initial_guess.py --test 3 -p 6 --lc 0.1 --tau 19`. The JSON basename is retained as a historical result
  identifier.

## Guiding-Center Response Files

The disk Euler-gas BDF2 comparison files
`guiding_center/euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6.args` and
`guiding_center/euler_vortex_gas_si_bdf2_p6_poisson_p5_l2_closest_p6.args`
use density DG(6), Poisson potential/flux DG(5), and recovered flux DG(6).
Both use cached raw CUDA recovery for transport, including startup and retries;
no extra field L2 convergence order is claimed. See the
[runner documentation](../scripts/guiding_center/README.md#bdf2-with-reduced-order-poisson-and-recovered-electric-field)
for commands and validation status.

`run_guiding_center_cases.py` accepts argparse response files with `@path`.
Use these for long GPU commands so terminal copy/paste cannot insert a newline
between an option and its value.  Lines may contain normal shell-like quoting,
blank lines, and `#` comments.

Append normal CLI overrides after the response file, for example `--verbosity 2`, `--mesh-size 0.008`, or `--transport-initial-guess initial-density-trace`.

When Poisson or transport assembly uses `cupy` or `raw-cuda`, initial density
and equilibrium density are projected with the existing `CupyDGSpace` API.
Quadrature mapping, callable evaluation and coefficient projection run on the
GPU, and the resulting field keeps its coefficients on the device. Host assembly
profiles use the NumPy projection. Initialization logs show `(cupy)` or `(numpy)`;
the initial diagnostics row also records `initial_projection_backend`,
`initial_density_projection_time` and `equilibrium_density_projection_time`.
Device projection timings synchronize the current CUDA stream so they measure
completed work, including first-use setup of the cached device space.

Terminal output is drained to the log and original stdout/stderr by a separate
process. This allows verbose native solver output to exceed pipe capacity even
when PyAMGX retains the Python GIL. A Python-thread drainer could deadlock on a
long iteration log before AMGX returned and the configured retries could run.
The helper closes and is reaped when the run exits; it does not initialize CUDA.

Examples from the repository root:

```bash
scripts/guiding_center/run_local_amgx_cases.sh @run_configs/guiding_center/diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx.args
scripts/guiding_center/run_local_amgx_cases.sh @run_configs/guiding_center/diocotron_k100_p6_dt01_smoke_raw_cuda_amgx.args
scripts/guiding_center/run_local_amgx_cases.sh @run_configs/guiding_center/gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx.args
```

The [disk diocotron benchmark](../docs/research/solver_studies/diocotron_ark3_2026_09.md)
starts with a smooth mode-9 annulus (`diocotron_smooth_m9_ark3_p6_h008_dt005_t70.args`).
It includes control/half-seed cases, the steeper paper profile, and provisional
m=32/64/128 thin-ring candidates. The same runner records every-stage positivity,
Fourier modes, mass, energy and enstrophy. Initial projection can use richer
quadrature through `--initial-projection-quad-1d` without changing evolution
quadrature. See the benchmark document for commands and measured limitations.

Current guiding-center response files:

- `guiding_center/positive_turbulence_iter_si_bdf2_p6_poisson_p5_rt_p6_fast.args`: quiet ITER positive turbulence, h=0.014, 300k+ triangles, SI-BDF2 dt=0.005 to T=50; density DG(6), Poisson DG(5), RT recovery into DG(6), and conflict-averaged upwind. Inherits robust Poisson/transport retries; disables diagnostics, plots, timing files, and AMGX residual history.

- `guiding_center/euler_vortex_gas_si_bdf2_p6_poisson_p5_l2_closest_p6_fast.args`: matching quiet L2-closest recovery variant with the same mesh, timestep, output controls, and solver tolerances.
- `guiding_center/euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6_fast.args`: quiet RT-recovered SI-BDF2, h=0.0068, at least 100k triangles, p=6, dt=0.05 to T=50. Disables field diagnostics, timing files, plots, and AMGX residual history; retains conflict averaging and convergence checks. Throughput is not yet measured.

- `guiding_center/euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr.args`: heavy IMEX-ARK3 Euler vortex-gas disk, h=0.008, p=6, tau=1000, dt=0.005 to T=50. Three transport solves sharing one operator and four Poisson solves per step, with embedded error diagnostics; see [IMEX-ARK3](../docs/algorithms/advection_reaction/imex_ark3.md).

- `guiding_center/diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx.args`: full `T=50` k=100 sharp annular-band raw-CUDA/AMGX stress run.
- `guiding_center/diocotron_k100_p6_dt01_smoke_raw_cuda_amgx.args`: short launch check for the same k=100 configuration.
- `guiding_center/gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx.args`: legacy Gaussian-annulus k=3 run using the full raw-CUDA/AMGX stack.
- `guiding_center/diocotron_multimode_p6_50k_dt01_t100_raw_cuda_bsr.args`: annular mixing candidate with modes 3-7, p=6, at least 50k triangles, dt=0.1 and T=100. Uses native FB-HP-MG Poisson and AMGX BSR transport, with plots and diagnostics every 20 steps.
- `guiding_center/euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr.args`: 360 signed Gaussian vortices across the disk at four core sizes; p=6, at least 50k triangles, semi-implicit Euler dt=0.01 to T=50. Density is Euler vorticity, with a fixed symmetric red/blue color scale.
- `guiding_center/positive_turbulence_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr.args`: 360 positive compact Gaussian density blobs in the disk, with an exact width-0.04 zero-density wall annulus; p=6, h=0.008, IMEX-ARK3 dt=0.005 to T=50. Initial, stage, and endpoint positivity diagnostics are enabled.
- `guiding_center/positive_turbulence_si_bdf2_p6_h0068_dt0005_t50_raw_cuda_bsr.args`: disk positive-turbulence SI-BDF2 run with 360 blobs, p=6, h=0.0068, 150k+ triangles and dt=0.005 to T=50. Positivity is measured initially and every 10 accepted endpoints. Poisson uses the opt-in robust PCGF ladder and one terminal FGMRES/DILU attempt.
- `guiding_center/positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr.args`: ITER-wall positive-turbulence SI-BDF2 run with 11,520 blobs, p=6, h=0.014, 300k+ triangles and dt=0.005 to T=50. Positivity is measured initially and every 10 accepted endpoints. Poisson uses the same robust PCGF-first fallback ladder and retains only its best checked candidate vector.
- `guiding_center/euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr.args`: semi-implicit Euler comparison at h=0.008, p=6, dt=0.05 to T=50 (1000 steps).
- `guiding_center/euler_vortex_gas_predictor_corrector_p6_h008_dt005_t50_raw_cuda_bsr.args`: matching predictor-corrector comparison. Both presets use identical initial data, spatial operators and tolerances, with separate output names and plots/diagnostics every 0.5 time units.
- `guiding_center/euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr.args`: heavy H1-BDF3 disk trial, h=0.008, p=6, Poisson tau=1000, dt=0.005 to T=50, Holoviz output every 0.5 time units. The default timestep passed 80 steps with the revised SI-Euler extrap3 startup; full T=50 stability is untested.
- `guiding_center/euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr.args`: matched heavy H2-BDF3 Euler vortex-gas disk, h=0.008, p=6, tau=1000, dt=0.005 to T=50. Two transport and two Poisson solves per regular step, shared SI-Euler extrapolation startup, and Holoviz output every 0.5. The qualitative run is left to the user; see [H2-BDF3](../docs/algorithms/advection_reaction/h2_bdf3.md).
- `guiding_center/h1_bdf3_host_accuracy.args`: six-step user-run H1-BDF3 host configuration (two SI-Euler extrap3 startup steps and four BDF3 steps). Existing heavy configurations accept `--time-scheme h1-bdf3`; see the [algorithm and startup constraints](../docs/algorithms/advection_reaction/h1_bdf3.md).
- `guiding_center/euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr.args`: semi-implicit BDF2 with one Euler startup step, h=0.008, p=6, dt=0.05 to T=50. Uses NVIDIA Holoviz plotting and one transport/Poisson pair per step. Append `--dt 0.02 --num-steps 2500` to retain T=50 with smaller steps.
- `guiding_center/euler_star_hole_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr.args`: multiscale signed vortices in a five-lobed nonconvex star with a circular island; h=0.008, p=6, SI-BDF2 dt=0.05 to T=50, Poisson tau=1000 and Holoviz plotting.
- `guiding_center/euler_vortex_gas_localized_p6_50k_dt001_t50_raw_cuda_bsr.args`: the same vortex gas with centers restricted to radius 0.5, leaving negligible initial vorticity near the wall.
- `guiding_center/spiral_sheet_p6_50k_dt002_t100_raw_cuda_bsr.args`: thin five-turn Gaussian spiral, p=6, at least 50k triangles, predictor-corrector dt=0.02 and T=100. Plots and diagnostics every 50 steps (one time unit).

The multimode preset uses
`rho_init = A(r) * (1 + 0.15 * sum(cos(m*theta + 0.37*m*m), m=3,...,7))`,
where `A(r) = 0.5 * (tanh((r-0.3724)/0.003) - tanh((r-0.46)/0.003))`.
The annular radii follow [Rome, Chen & Maero (2018)](https://doi.org/10.1063/1.5021577);
the smooth edges and deterministic phases are an adaptation of their initial
particle fluctuations. Turbulent development needs verification in the nonlinear
run. The existing harmonic diagnostics monitor modes 5, 10 and 15; they do not
measure the complete angular spectrum.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_multimode_p6_50k_dt01_t100_raw_cuda_bsr.args
```

For a T=50 run, append `--num-steps 500`. To start the 150k comparison, append
`--mesh-size 0.0068 --minimum-triangles 150000 --diagnostics-prefix diocotron_multimode_p6_150k`.

## Thin spiral mixing candidate

The spiral preset convolves an Archimedean curve with a Gaussian:
`gamma(a) = (0.12 + 0.63*a/(10*pi)) * (cos(a), sin(a))`, for `0 <= a <= 10*pi`,
and `rho_init(x) = integral_gamma exp(-|x-y|^2/(2*sigma^2)) ds_y / (sqrt(2*pi)*sigma)`.
Here `sigma=0.005`, giving approximately unit peak density and transverse
FWHM 0.0118, with smooth tips and near-zero density between turns. It uses the
unit disk, zero potential on the wall and zero density flux. The initial density
supports NumPy and CuPy; local quadrature discards Gaussian tails beyond nine sigma.

The spiral family is motivated by
[Rome, Chen & Maero (2016)](https://doi.org/10.1088/0963-0252/25/3/035016).
These radii and Gaussian smoothing are our adaptation. Vortex formation and
mergers still need verification in a nonlinear run; this is not an already
validated turbulent solution. No radial equilibrium or reference azimuthal
mode is assigned to this case.

Run from the repository root (the exports select this machine's installed CUDA toolkit):

```bash
export CUDA_PATH=/usr/local/cuda-13.0
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$HOME/.local/amgx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/spiral_sheet_p6_50k_dt002_t100_raw_cuda_bsr.args
```

For a first T=50 run, append `--num-steps 2500`. For a two-step launch check,
append `--num-steps 2 --plot-every 0 --diagnostics-every 1 --diagnostics-prefix spiral_sheet_smoke`.
For a T=100 time-step comparison, append
`--dt 0.01 --num-steps 10000 --plot-every 100 --diagnostics-every 100 --diagnostics-prefix spiral_sheet_dt001`.
For the 150k spatial comparison, append
`--mesh-size 0.0068 --minimum-triangles 150000 --diagnostics-prefix spiral_sheet_p6_150k`.

Local launch validation: two float64 predictor-corrector steps completed on
50,676 triangles at p=6. At T=0.04, relative mass drift was -3.24e-16,
relative electrostatic-energy drift was 4.50e-11, and minimum projected density
was -2.49e-6 (the analytic initial density is nonnegative). This short check
validates initialization and solver compatibility, not long-time turbulence.

## Euler vortex gas: many vortices throughout the disk

Use `euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr` for a domain filled with
positive and negative vorticity at several scales. Its initial field is
`rho(x) = sum_j a_j*exp(-|x-c_j|^2/(2*sigma_j^2))`, with the following blobs:

| Gaussian sigma | Number of blobs |
| --- | ---: |
| 0.008 | 192 |
| 0.016 | 96 |
| 0.032 | 48 |
| 0.064 | 24 |

Centers are independently uniform in disk area up to radius 0.96, using seed 17.
Each size has equal numbers of positive and negative blobs. Individual peak
magnitudes are sampled between 3.2 and 4.8, then negative strengths are adjusted
per size to cancel circulation over the actual unit disk (including the Gaussian
tails cut by the wall). Overlap changes the extrema of the summed field; 360
blobs does not mean 360 distinct coherent vortices. The default initial field
has over 300 strong resolved vorticity extrema distributed across the disk.

This is a random Gaussian vortex-gas initialization, a family used for decaying
2D turbulence; see [Kuznetsov (2016), slides 50-53](https://nonlinearwaves.ipfran.ru/www_2016/materials/Kuznetsov1.pdf).
Our counts, core sizes, amplitudes and disk geometry define a new test case.
The initial condition seeds interacting vortices directly. It does not impose
an inertial-range spectrum or prove a turbulent cascade.

The runner still solves the same guiding-center/Euler transport-Poisson system:
zero wall potential makes the exact rotated gradient tangent, with zero advective
density flux in the transport boundary treatment. The independently approximated
HDG flux may retain a tangency defect; see the
[boundary diagnostics](../docs/development/transport_boundary_diagnostics.md).
Vorticity is signed and need not vanish at the wall. No nonnegative background,
annular confinement or forcing is added.

Run from the repository root:

```bash
export CUDA_PATH=/usr/local/cuda-13.0
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$HOME/.local/amgx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr.args \
  --precision float64
```

For a first T=5 test, append `--num-steps 500 --diagnostics-prefix vortex_gas_t5`.
For a different realization, append `--case-param seed=18`.
Keep the seed and physical parameters fixed in mesh comparisons. The 150k
comparison uses `--mesh-size 0.0068 --minimum-triangles 150000 --diagnostics-prefix vortex_gas_p6_150k`.
At fixed T=50, the dt=0.005 comparison uses
`--dt 0.005 --num-steps 10000 --plot-every 100 --diagnostics-every 100 --diagnostics-prefix vortex_gas_dt0005`.

For this signed case, the CSV/JSONL includes `circulation`, `circulation_drift`
and `enstrophy = integral(rho^2)/2`, alongside the existing energy diagnostics.
`mass_relative_drift` is left null because division by nearly zero total
circulation is misleading. Enstrophy loss in this inviscid problem measures
numerical dissipation; conservation and convergence need checking as filaments
become finer. Plots label rho as vorticity and keep the initial symmetric color
range fixed so positive and negative structures remain comparable in time.

Validation: 73 focused tests passed. The initial density also agrees between
CPU and GPU to 8.9e-16 in the sampled comparison. An attempted p=6 run completed
237 steps to T=2.37 before an AMGX scaled residual failed its acceptance check
on the next step, despite a small physical residual. The preset now enables the
runner's bounded `amgx-robust` retry policy. After the primary solve fails,
retries now use AMGX `PBICGSTAB` with L1 Jacobi, then block Jacobi, retaining
native BSR. FGMRES/DILU with device-side scalar CSR conversion is the final
fallback, with up to two residual corrections. Both Jacobi preconditioners are
already provided by the installed AMGX fork. Their one-application settings follow the earlier BSR solver benchmarks.
Both PBICGSTAB retries now inherit `--transport-scale-system`, as FGMRES already
did. Validation uses unit tests and tiny synthetic matrices; these changes have
not been qualified by a new simulation.

## Positive turbulence: guiding-center density

`positive_turbulence_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr` reuses the
Euler gas's counts `(192, 96, 48, 24)`, Gaussian widths
`(0.008, 0.016, 0.032, 0.064)`, peak-amplitude range 3.2--4.8, and seed 17.
Unlike the Euler case, every strength is positive and the field is treated as
a guiding-center density rather than signed vorticity for labels and mass
diagnostics. Mathematically it is also a one-signed Euler-vorticity initial
condition, up to the Poisson sign convention. This is an HYBRIDGE-defined stress
test rather than a reproduction of a published benchmark; its literature
context is recorded in `scripts/guiding_center/README.md`.

The package-level Gaussian sampler truncates each blob at eight standard
deviations and samples its center uniformly in the scale-dependent disk
`r < 1 - 0.04 - 8*sigma`. Hence every support ends before `r=0.96`, making the
analytic profile nonnegative everywhere and identically zero in a genuine
neighborhood of the wall. The cutoff is part of the initial-data definition;
it does not clip the evolved solution.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr.args \
  --precision float64 --verbosity 3 --save-diagnostics
```

This provisional p=6, h=0.008 run uses IMEX-ARK3 with `dt=0.005` for 10,000
steps to `T=50`. Positivity diagnostics inspect the projected initial density,
every transported ARK stage, and every accepted endpoint. They record sampled
minima, Bernstein lower bounds, negative quadrature mass, and the first loss of
positivity; they do not alter the field or provide a positivity-preserving
limiter. The preset has not been time-integrated or stability-qualified here.

## Vortex gas in a nonconvex star with an island

`euler_star_hole_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr` extends the existing
`gmsh_smooth_star_mesh` geometry with a concentric circular hole. The outer
radius is `r(theta) = 1 + 0.35*cos(5*theta)`, sampled with 500 straight boundary
segments; the hole radius is 0.30. The fluid has five lobes, concave bays and an
inner wall, with area about 3.051 (close to the unit disk's 3.142).

The initial field contains 360 signed Gaussian vortices at the same four core
sizes as the disk vortex gas, down to sigma=0.008. Centers are sampled throughout
the fluid, including the outer lobes, at least two core widths from both walls.
Opposite-signed, equal-strength pairs are related by random nonzero rotations
through multiples of 2*pi/5. This balances circulation over the continuous
star-with-hole domain, including Gaussian tails, without forcing the entire
field to be rotationally symmetric. Mesh and projection errors can leave a
small discrete circulation. Seed 17 fixes the realization; `--case-param seed=18`
selects another. The case is an unforced vortex-interaction test, with no imposed
turbulent spectrum.

Small cores and the thin filaments that may develop around the island and in
the lobes motivate the p=6, h=0.008 resolution. The preset requires at least
100,000 triangles. Geometry-only validation generated 161,680 triangles:
4,527,040 element density DOFs and 1,703,786 face trace DOFs at p=6, before the
additional Poisson fields. This is about 42% more triangles than the 113,894-cell
disk run. This resolution still needs a mesh comparison to establish accuracy
for the evolved flow. Geometry parameters are shared by vortex seeding
and mesh generation, and can be changed with `--case-param`, e.g.
`--case-param hole_radius=0.35`.

The potential is fixed to zero on both boundary components, and transport uses
zero density flux at both walls. This prescribes the two potential constants;
it does not independently prescribe a circulation around the island. Holoviz
uses the actual mesh connectivity, leaving the hole and concave bays empty.

The preset keeps semi-implicit BDF2 with one SI-Euler startup step, dt=0.05,
1,000 steps to T=50, Poisson tau=1000, native FB-HP-MG Poisson, raw-CUDA AMGX
BSR transport with bounded retries, and plots/diagnostics every 10 steps.
Its output prefix is separate from the disk runs.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_star_hole_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr.args \
  --precision float64
```

Append `--dt 0.02 --num-steps 2500` for the same T=50 with smaller steps, and
use a distinct `--diagnostics-prefix` to keep comparison outputs. The CUDA/AMGX
environment is the same as for the disk run above. Seventeen focused checks
passed, covering circulation balance, reproducibility, both mesh boundaries,
mesh-cache separation, Holoviz's hole mask, registry integration and CLI
overrides. Numba compilation was blocked during validation; no builds or time
integration were run. The geometry-only check also produced an
[initial-field preview](../artifacts/star_hole_preset/initial_vorticity.png).

## Matched-time temporal comparison

The temporal-convergence driver supplements its manufactured solution with
`--study vortex-gas`. It inherits the star-with-hole preset and compares
`dt=0.01,0.005` over T=5 by default, sampling fields every 0.5 physical time
units. Initial conditions, spatial operators and solver tolerances are fixed.

```bash
HYBRIDGE_PRECISION=float64 .venv/bin/python -m scripts.guiding_center.benchmarks.run_guiding_center_temporal_convergence \
  --study vortex-gas --dts 0.01,0.005 --final-time 5 \
  --sample-interval 0.5 --plot-comparison --cached-kernels-only \
  --prefix star_hole_bdf2_dt_comparison
```

BDF2 is the default. Select `--scheme si-euler` or
`--scheme predictor-corrector` for the other integrators. Predictor–corrector
samples use the accepted endpoint density and its extrapolated endpoint trace.

Use the CUDA/AMGX environment above. `--dry-run` inspects the comparison;
`--preset euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr` selects the disk.
Choose a new prefix for each comparison. The cache-only option prevents native
compilation and reports any missing kernel cache. Enstrophy histories, broken
palinstrophy, HDG trace mismatch, HDG palinstrophy, accepted-velocity cell CFL,
final DG L2 differences, matched field images and exact configurations are saved.
The HDG term uses `1/h_K` (element diameter), both sides of interior faces, and
the actual numerical trace. Unused zero-flux boundary trace slots are excluded.
Final volume and trace coefficients are saved separately.

The shared scalar evaluator in `hybridge/assembly/hdg_gram.py` performs CuPy norm
reductions directly from resident device fields; raster sampling uses cuSPARSE.
Reports record the diagnostic backend. The manufactured study also uses device
AMGX solves and CuPy diagnostics, with analytic gradient, trace mismatch and
HDG H1 errors/rates for both rho and phi. Two vortex-gas dts show sensitivity,
not formal order. See the [temporal study details](../MANUAL.md#fixed-mesh-guiding-center-cases-runner)
for the norm definitions and spatial error floor; the manufactured study remains
the default.

## Localized vortex gas and transport diagnostics

The localized preset keeps the original seed, 360 vortices, core sizes, p=6,
mesh target and time settings, but restricts vortex centers to radius 0.5
(instead of 0.96). Gaussian tails remain; this is not compact support. Sampled
initial wall vorticity falls from about 2.88 to 1.42e-13. The Poisson velocity
can still extend to the wall.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_localized_p6_50k_dt001_t50_raw_cuda_bsr.args \
  --precision float64 --plot-every 30 -v 3
```

Append `--transport-direct-fallback cusolver-qr` to enable the final device
sparse QR attempt. Sparse factorization can require much more memory than the
stored transport matrix; only small matrices have been tested. The default
remains the six AMGX attempts. `--transport-scale-system off` disables scaling
for the primary and all iterative retries for a controlled comparison.

At each diagnostic output, CSV/JSONL records boundary-normal velocity,
elementwise divergence, and interior normal jumps. A rejected transport stage
writes `<diagnostics-prefix>_transport_failure.json`, including the actual
stage coefficient beta, its time-step scale, physical residuals for all attempts,
and matrix row norms. See [the investigation and validation evidence](../docs/development/transport_boundary_diagnostics.md).

The [high-mode diocotron study](../docs/research/solver_studies/diocotron_high_modes_2026_09.md)
provides `diocotron_gaussian_m64_ark3_p6_h008_dt005_t70.args` and
`diocotron_gaussian_m128_ark3_p6_h008_dt005_t70.args` in `guiding_center/`.
Both use a Gaussian radial profile and retain all modes through the third harmonic;
m=128 is a resolution stress test.

Guiding-center presets, diagnostics, Poisson studies and temporal benchmarks are
listed in the [guiding-center script guide](../scripts/guiding_center/README.md).


## Additional Euler-gas geometries and diagnostic plots

The shared IMEX-ARK3 runner now has p=6, T=50 response files for
[horseshoe](guiding_center/euler_horseshoe_gas_imex_ark3_p6_150k_t50.args),
[ITER](guiding_center/euler_iter_gas_imex_ark3_p6_300k_t50.args), and
[Pac-Man](guiding_center/euler_pacman_gas_imex_ark3_p6_150k_t50.args) domains.
Horseshoe and Pac-Man require at least 150k triangles; ITER targets 300k+
with 5,760 signed vortices. These new production cases await user execution.

Append `--save-diagnostics` to save aligned Matplotlib PNG/PDF time histories.
Use `--plot-diagnostics` to display them at the end without writing figures, or
use both flags to save and display.
For diocotron, this includes the log-log instability norm
`||phi - phi_eq||_L2` and time-dependent active-mode rankings and spectra.
See the [geometry and diagnostic plotting guide](../scripts/guiding_center/README.md#horseshoe-iter-like-and-pac-man-euler-gas)
for commands, profile definitions, and output details.
