# Guiding-center temporal convergence and CFL sensitivity — September 2026

Study date: 2026-09-12. SI Euler, semi-implicit BDF2, and predictor–corrector
are compared at fixed spatial discretization.

## Main findings

Nineteen runs completed: all twelve manufactured cases and seven of the nine
gas cases. PC dt=0.02 failed in transport after T=2.52. The final PC dt=0.005
rerun was stopped on request after accepted step 628 (T=3.14); its partial
history is retained and is not classified as numerical failure.

- Manufactured density errors show the intended first/second/second temporal
  orders, including the HDG H1 diagnostic.
- At T=5 and dt=0.005, BDF2 loses 2.77% of enstrophy versus 19.08% for SI Euler,
  with similar step cost (0.322 versus 0.316 seconds).
- BDF2 dt=0.01 and 0.005 still differ by 19.03% in volume L2. Cell CFL near one
  is only an initial accuracy screen, not evidence of temporal resolution.
- Exceptional face traces dominate the HDG mismatch in otherwise completed
  second-order runs. The failed PC matrix has an explicitly demonstrated local
  rank deficiency. Fixing trace coupling deserves priority over larger Krylov
  iteration limits.

## Experiment and diagnostics

The manufactured case is the existing translating Helmholtz wave on a
rectangle: 14,776 triangles, 413,728 density coefficients, degree 6, target mesh
size 0.025, final time 0.2, and time steps
0.04, 0.02, 0.01, 0.005. Errors use the analytic density and potential and their
physical gradients. Observed rates use successive step halvings, at the same
final physical time. These are errors of the fully discrete solution and can
reach a spatial error floor.

The vortex gas uses the existing nonconvex five-lobed star with a circular hole:
161,680 triangles, degree 6, 4,527,040 density coefficients, target mesh size
0.008, hole radius 0.3, 360 signed Gaussian vortices, seed 17, and Poisson tau
1000. Every scheme starts from the same projected initial condition and runs
to time 5 with steps 0.02, 0.01, 0.005. No exact turbulent reference solution is
available. Same-mesh differences measure temporal sensitivity. They include
interaction with spatial dissipation and do not establish spatial convergence.

For the scalar field and its actual numerical trace, the shared
[`ScalarHDGGram`](../../../hdgfem/assembly/hdg_gram.py) evaluator computes

\[
Z=\tfrac12\|\rho_h\|_{L^2}^2,\qquad
P=\tfrac12\sum_K\|\nabla\rho_h\|_{L^2(K)}^2,\qquad
J=\sum_K h_K^{-1}\|\rho_h-\widehat\rho_h\|_{L^2(\partial K)}^2,
\]

with element diameter \(h_K\) equal to its longest edge. Both sides of interior
faces contribute. Unused zero-flux boundary trace slots are excluded for gas;
prescribed boundary traces are included for the manufactured case. We report
physical enstrophy \(Z\), broken palinstrophy \(P\), and the separate face term
\(J\), together with \(P_{\mathrm{HDG}}=P+J/2\). The HDG H1 norm is
\(\sqrt{\|u_h\|^2+\|\nabla_hu_h\|^2+J}\). For manufactured errors, analytic
volume values and gradients are subtracted and the exact continuous trace
cancels in the mismatch. No Gram inverse is required.

The cell CFL is \(\Delta t\max_K(U_K/h_{K,\min})\), where the denominator is
the shortest element edge. This is distinct from the diameter in the HDG norm.
Accepted-velocity CFL is sampled at diagnostic times; these maxima are not
continuous-in-time bounds. Gas snapshots also record the actual advecting
velocity and the coefficient-scaled stage CFL. The coefficient is dt for
Euler and BDF2 startup, 2dt/3 for subsequent BDF2 steps, and dt/2 for the PC
midpoint solve. The aggregate length \(\ell_{\mathrm{HDG}}=\sqrt{Z/P_{\mathrm{HDG}}}\)
and \(\Delta t U_{\max}/\ell_{\mathrm{HDG}}\) are additional resolution indicators;
this RMS length is not a lower bound on filament width.

The implemented predictor–corrector first performs a full SI Euler prediction,
solves its Poisson problem, and averages old and predicted velocities. It then
solves \((I+\Delta t A(v_{\mathrm{mid}})/2)\rho_{\mathrm{mid}}=\rho_n\)
and accepts \(2\rho_{\mathrm{mid}}-\rho_n\), with the same combination of density
traces. Its diagnostics here use that accepted endpoint. BDF2 uses SI Euler
startup followed by
\((I+2\Delta t A(2v_n-v_{n-1})/3)\rho_{n+1}=(4\rho_n-\rho_{n-1})/3\).
These statements describe the implemented frozen-velocity stages, not an
unconditional stability assertion for the nonlinear coupled scheme.

## Execution and provenance

Jobs run sequentially on an NVIDIA RTX PRO 5000 Blackwell (48 GB), CUDA 13,
local AMGX 2.5.0, float64. Transport uses raw-CUDA assembly and device AMGX;
manufactured Poisson uses device AMGX and cached raw-CUDA Schur-LU factors.
Gas Poisson uses native face-block hp-multigrid PCG with an AMGX coarse solve.
Field diagnostics and norm differences use CuPy contractions and reductions;
raster sampling uses cupyx CSR backed by cuSPARSE. Only scalar results and
explicit output artifacts are downloaded. Interactive plotting is disabled.

The study forbids Numba/NVRTC/NVCC cache misses. No native build or compilation
is authorized or performed. Run wall times include setup, diagnostics and
snapshot writes; per-step coupled timings exclude the observer's output work.
This is a single-run comparison, not a repeated hardware benchmark.

A setup validation error (`poisson_local_backend="cupy"`) was corrected to
`"numpy"`, the valid CPU fallback setting; the independently selected raw-CUDA
reconstruction path remains active. The manufactured case selects Schur-LU
factors because its nonzero Dirichlet data are unsupported by the cached
CuPy RHS path. Preliminary manufactured
setup failures are archived separately from numerical outcomes.

The first gas batch also exposed a missing
`jacobi_l1_scalar_rows_for_blocks=1` in the absolute-tolerance Poisson fallback
configuration. This caused an unsupported 7×7 block error after the native
Poisson convergence gate failed. The existing option was enabled without
changing the discretization or tolerance. Original failed logs are retained;
reruns and any native-gate events are recorded with the completed results.

A second issue was exposed by an explicit native Poisson residual of
1.0001382842e-12 against a 1e-12 target: FP64 stopped at a recursive convergence
candidate, failed final verification, and permanently abandoned the native
hierarchy. The existing FP32 verification/restart behavior now also applies
in FP64. This preserves the residual target and restarts from explicit b-A*x
when needed. The affected SI Euler/BDF2/PC gas jobs were rerun; completed
runs that never entered this path were retained. A regression test injects a
recursive-residual underestimate and checks recovery to the true solution.

Raw logs, configurations, norms, raster snapshots, final coefficients and traces,
and the execution manifest are in
[`run_outputs/guiding_center/convergence/all_schemes_20260912/`](../../../run_outputs/guiding_center/convergence/all_schemes_20260912/).
The exact orchestration and analysis scripts are
[`run_study.py`](../../../artifacts/temporal_cfl_20260912/run_study.py) and
[`analyze_study.py`](../../../artifacts/temporal_cfl_20260912/analyze_study.py).

## Manufactured convergence

All twelve cases reached T=0.2. Final-step-halving density L2 rates are
0.998 (SI Euler), 1.999 (BDF2), and 2.000 (PC). The corresponding HDG H1
rates are 0.994, 1.950, and 2.000. The gradient/trace diagnostic therefore
supports the intended temporal orders on this smooth case. BDF2's H1 rate
weakens slightly at the finest step; a fixed-space sweep alone cannot isolate
its spatial and temporal contributions.

| Scheme | dt | Density L2 error | Rate | Density HDG H1 error | Rate |
|---|---:|---:|---:|---:|---:|
| SI Euler | 0.04 | 1.183792e-02 | — | 8.077656e-02 | — |
| SI Euler | 0.02 | 5.952603e-03 | 0.992 | 4.071397e-02 | 0.988 |
| SI Euler | 0.01 | 2.984907e-03 | 0.996 | 2.047738e-02 | 0.991 |
| SI Euler | 0.005 | 1.494656e-03 | 0.998 | 1.028396e-02 | 0.994 |
| BDF2 | 0.04 | 3.508581e-03 | — | 2.413124e-02 | — |
| BDF2 | 0.02 | 8.831246e-04 | 1.990 | 6.152324e-03 | 1.972 |
| BDF2 | 0.01 | 2.210185e-04 | 1.998 | 1.572721e-03 | 1.968 |
| BDF2 | 0.005 | 5.528136e-05 | 1.999 | 4.069446e-04 | 1.950 |
| PC | 0.04 | 1.096297e-04 | — | 1.576761e-03 | — |
| PC | 0.02 | 2.748334e-05 | 1.996 | 3.942927e-04 | 2.000 |
| PC | 0.01 | 6.876395e-06 | 1.999 | 9.854741e-05 | 2.000 |
| PC | 0.005 | 1.719539e-06 | 2.000 | 2.463932e-05 | 2.000 |

At dt=0.005, the potential errors are:

| Scheme | Potential L2 error | L2 rate | Potential HDG H1 error | H1 rate |
|---|---:|---:|---:|---:|
| SI Euler | 2.646436e-04 | 0.998 | 6.543879e-04 | 0.998 |
| BDF2 | 9.630800e-06 | 2.002 | 2.386878e-05 | 2.002 |
| PC | 6.801326e-08 | 1.999 | 3.053545e-07 | 1.999 |

All ten error measures and their rates, including separate gradient and face
errors, are in the [machine-readable table](guiding_center_temporal_cfl_samples_2026_09.csv).

![Manufactured errors](../../../run_outputs/guiding_center/convergence/all_schemes_20260912/manufactured_errors.png)

Sampled cell CFL ranges from approximately 4.716 at dt=0.04 to 0.590 at
dt=0.005. All these smooth manufactured runs completed; this establishes no
universal CFL stability threshold for vortex gas.

![Manufactured CFL sensitivity](../../../run_outputs/guiding_center/convergence/all_schemes_20260912/manufactured_cfl.png)

## Gas outcomes and diffusion

The table compares the same physical endpoint T=5. Enstrophy and energy changes
for incomplete runs are deliberately omitted here because their shorter intervals
are not comparable. The complete partial histories remain in the raw outputs. The final PC dt=0.005 rerun was interrupted at the user’s request to wrap up; it is not counted as a numerical failure.

| Scheme | dt | Outcome / last accepted time | Enstrophy loss at T=5 | Energy change at T=5 | Sampled max cell CFL |
|---|---:|---|---:|---:|---:|
| SI Euler | 0.02 | T=5 completed | 29.942% | +3.3792% | 3.446 |
| SI Euler | 0.01 | T=5 completed | 24.320% | +1.9906% | 1.761 |
| SI Euler | 0.005 | T=5 completed | 19.078% | +1.1252% | 0.889 |
| BDF2 | 0.02 | T=5 completed | 8.582% | +0.2086% | 3.595 |
| BDF2 | 0.01 | T=5 completed | 4.877% | +0.0476% | 1.799 |
| BDF2 | 0.005 | T=5 completed | 2.767% | +0.0102% | 0.900 |
| PC | 0.02 | Failed at 2.520 | — | — | 3.367 |
| PC | 0.01 | T=5 completed | 1.146% | +0.0278% | 1.799 |
| PC | 0.005 | Interrupted at 3.140 | — | — | 0.900 |

BDF2 loses much less enstrophy than SI Euler at the same dt, at similar per-step
cost. PC can retain more enstrophy, but its coarsest run fails. Energy conservation
and a small linear residual alone do not establish adequate temporal resolution.

![Gas histories](../../../run_outputs/guiding_center/convergence/all_schemes_20260912/gas_histories.png)

The all-scheme field comparison uses the latest diagnostic time shared by all
nine runs. This avoids comparing a failed run's early state with T=5 states.

![Matched gas fields](../../../run_outputs/guiding_center/convergence/all_schemes_20260912/gas_fields_common_time.png)

Completed fields at T=5 and their dt=0.01 minus dt=0.005 differences are also
saved. Empty panels indicate runs that did not reach that endpoint. Color
scales are fixed across panels; they are clipped as stated on the figures.

![Gas fields at T=5](../../../run_outputs/guiding_center/convergence/all_schemes_20260912/gas_fields_T5.png)

![Temporal field differences at T=5](../../../run_outputs/guiding_center/convergence/all_schemes_20260912/gas_field_differences_T5.png)

## HDG diagnostic and trace localization

| Scheme | dt | Broken palinstrophy P | Face mismatch J | HDG palinstrophy P+J/2 |
|---|---:|---:|---:|---:|
| SI Euler | 0.02 | 2.131629e+04 | 4.970633e+00 | 2.131877e+04 |
| SI Euler | 0.01 | 3.410290e+04 | 1.176377e+01 | 3.410878e+04 |
| SI Euler | 0.005 | 5.135996e+04 | 2.094972e+01 | 5.137043e+04 |
| BDF2 | 0.02 | 1.076165e+05 | 7.682034e+02 | 1.080006e+05 |
| BDF2 | 0.01 | 1.575611e+05 | 8.445948e+08 | 4.224550e+08 |
| BDF2 | 0.005 | 2.062241e+05 | 1.152950e+08 | 5.785371e+07 |
| PC | 0.01 | 2.762234e+05 | 9.860323e+09 | 4.930438e+09 |

The new face diagnostic exposes localized trace pathologies that enstrophy
misses. At T=5, BDF2 dt=0.01 has trace coefficients up to 101,820 on interior
edge 93,640, while adjacent volume values on that face stay below 2.27 in
absolute value. That face contributes 99.999986% of J. At dt=0.005 the same
face reaches 38,644, contributes 99.999733% of J, and the adjacent volume
values stay below 2.38. The remaining J is approximately 114.92 and 308.22,
respectively. These are actual saved numerical traces, not a midpoint/endpoint
bookkeeping mismatch. Local two-cell quadrature reproduces their contributions.

Consequently, the enormous HDG palinstrophy is chiefly a trace issue; it cannot
be read as widespread volume filament growth. The broken-gradient term remains
useful for comparing resolved volume structure. The unweighted-in-velocity
1/h_K mismatch is a valuable health check, but it is not physical enstrophy or
the upwind flux's own velocity-weighted dissipation.

Full per-face coordinates, adjacent elements, amplitudes and contributions are
in [trace_face_analysis.json](../../../run_outputs/guiding_center/convergence/all_schemes_20260912/trace_face_analysis.json).

## Same-mesh temporal differences

The following norms are evaluated from saved full volume coefficients and
actual traces through the shared Gram evaluator on the GPU. Relative values
use the finer member as denominator. No raster approximation is used.

| Scheme | Coarse dt → fine dt | Relative volume L2 difference | Relative HDG H1 difference |
|---|---:|---:|---:|
| SI Euler | 0.02 → 0.01 | 23.044% | 64.781% |
| SI Euler | 0.02 → 0.005 | 36.181% | 82.760% |
| SI Euler | 0.01 → 0.005 | 20.491% | 58.170% |
| BDF2 | 0.02 → 0.01 | 24.397% | 99.991% |
| BDF2 | 0.02 → 0.005 | 30.753% | 99.950% |
| BDF2 | 0.01 → 0.005 | 19.028% | 170.477% |

Large trace outliers dominate some H1 differences. They must be retained as
evidence of a trace problem; they do not provide a clean estimate of the
volume solution's temporal order. The exact-solution manufactured study is the
order check. The gas comparison measures sensitivity at this fixed spatial
resolution, and the finest available run is not an exact reference.

## Failure evidence and a practical CFL screen

PC dt=0.02 stops in the corrector at step 127, with last accepted time 2.52.
At the last sampled state (T=2.5), enstrophy was 4.34880 versus 4.35086 initially,
and the sampled volume range was [-12.201, 8.358]. Its failed transport matrix
is finite and has no exactly zero rows or columns. Nevertheless, edge 12,759
has seven trace unknowns and only five active inflow sample directions. Its
35×7 assembled column panel has numerical rank five: the two smallest singular
values are 7.13e-27 and 4.60e-27, compared with 7.47e-10 for the largest. A
localized weak mode has relative matrix action 3.23e-17. This directly supports
a deficient trace-coupling explanation for this solve failure; a globally small
normal-flux jump norm would not exclude such a local defect.

The [failure diagnostics and matrix snapshot](../../../run_outputs/guiding_center/convergence/all_schemes_20260912/vortex-gas_predictor-corrector_dt0p02/runs/) preserve this evidence. This is not
a measurement of a universal BDF2 or PC CFL stability boundary. Positive
rescaling by dt changes coefficient magnitudes, but does not supply the missing
inflow sampling directions for a fixed velocity field. Changing dt also changes
the evolved and extrapolated velocities, so it can move or avoid a particular
deficient face without repairing the underlying trace construction.

For this gas, BDF2 completed at sampled cell CFL approximately 3.60, 1.80,
and 0.90. A cell CFL around or below one is a reasonable **first accuracy
screen** here, not a sufficient acceptance criterion: dt=0.01 versus dt=0.005
still changes enstrophy loss, gradients and filament locations appreciably.
Check the passage time of the features of interest, dt·U/ell, alongside
step-halving comparisons. The broken-gradient RMS length can help track
resolution trends; the HDG RMS length becomes dominated by exceptional trace
modes in these runs. Neither RMS length bounds the thinnest filament.

Before drawing longer-time temporal conclusions, the local normal-flux/trace
coupling needs attention. Increasing solver iteration limits or accepting a
small recursive residual does not address the demonstrated rank deficiency.

## Cost and solver checks

| Scheme | dt | Run wall (s) | Mean coupled step (s) | Mean accepted transport iterations / step | Retried transport stages | Max accepted transport true relative residual |
|---|---:|---:|---:|---:|---:|---:|
| SI Euler | 0.02 | 93.04 | 0.3437 | 42.49 | 0 | 6.595e-13 |
| SI Euler | 0.01 | 170.71 | 0.3275 | 28.25 | 0 | 3.545e-13 |
| SI Euler | 0.005 | 322.63 | 0.3163 | 23.18 | 0 | 7.758e-14 |
| BDF2 | 0.02 | 91.20 | 0.3414 | 35.18 | 0 | 1.730e-12 |
| BDF2 | 0.01 | 173.17 | 0.3322 | 27.94 | 2 | 4.967e-12 |
| BDF2 | 0.005 | 329.04 | 0.3218 | 25.80 | 2 | 3.910e-13 |
| PC | 0.01 | 349.65 | 0.6837 | 73.46 | 193 | 2.972e-11 |

PC performs two transport and two Poisson solves per step. The accepted
iteration column sums those stages, but excludes rejected attempts; total
attempt iteration counts and rejected-attempt counts are in the CSV. True
relative residuals can exceed rtol when the separately enforced absolute target
is the controlling tolerance. The large field and Gram diagnostics report
CUDA/device backends in the machine-readable outputs.

## Reproduction and validation

From the repository root, the recorded environment is:

```bash
export CUDA_PATH=/usr/local/cuda-13.0
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$HOME/.local/amgx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDGFEM_PRECISION=float64
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
.venv/bin/python artifacts/temporal_cfl_20260912/run_study.py
.venv/bin/python artifacts/temporal_cfl_20260912/analyze_study.py --device-differences
```

The orchestrator resumes its manifest and skips completed/failed jobs; rerunning
it will restart the interrupted PC dt=0.005 case from its initial state. Preserve
the existing output directory before creating a fresh manifest for a full rerun.
Individual exact configurations, source hashes, preliminary failure archives and
terminal logs are retained. The norm/observer/native-residual-restart regression
checks passed without native compilation. No new native build is required for
the Python and JSON changes made during this study.

The 84-unknown, 7×7 BSR check of the corrected scalar-row L1 component passed
with true relative residual 1.131e-14. This component check
does not validate the full hierarchy. The complete hybrid AMGX Poisson fallback remains a limitation: after the
missing block-size flag was corrected, two preliminary reruns still exhausted
its 500 iterations. A forced tiny hierarchy probe additionally hit an unsupported
host truncation path and exited with a segmentation fault; its log is preserved
in provenance. The completed corrected gas runs recover within native PCG
instead. The added block-size option alone should not be interpreted as a
validation of all fallback configurations.
