# Matplotlib diagnostic histories

The two optional flags work independently of live field plotting and the time scheme:

- `--plot-diagnostics` opens the Matplotlib figures at the end and waits until their
  windows are closed. It does not write figure files.
- `--save-diagnostics` writes PNG/PDF figures without opening windows.
- Use both to save and display the figures.

On normal completion, interruption, or a time-step failure, saved figures go under
`<diagnostics-dir>/<prefix>_diagnostic_plots/`: a multipage `diagnostics.pdf`,
descriptively named grouped PNGs, and a manifest mapping names to panel titles. CSV/JSONL recording remains automatic.
Repeated stage metrics share panels.
Numeric scalar histories, including nested stage diagnostics, are retained;
strings, configuration metadata, and single-sample values are not time series.
Signed drifts use symmetric-log axes, invariants use linear axes, and positive
errors/residuals and perturbation norms use log-log axes. Log axes omit t=0 and
nonpositive values without replacing them with a numerical floor.

For `diocotron_k`, either option also enables cached polar Fourier diagnostics.
The primary instability curve is exactly `diocotron_phi_eq_l2`,
the whole-domain norm `||phi - phi_eq||_L2`, on log-log axes. A second semilog
view helps inspect exponential growth. It is not replaced by a time derivative.

The modal recorder retains all positive FFT modes below Nyquist and records
the three strongest active mode numbers and their amplitudes at each diagnostic
time. `active_modes.png`/`.pdf` show the log-colored normalized spectrum,
dominant-mode tracks, leading modal L2 amplitudes, and active-mode count.
Active means amplitude at least 0.001 times the strongest resolved mode;
rankings measure amplitude, not an independently fitted growth exponent.
Increasing `--diocotron-angular-points` extends the recorded angular band.
Use `--diagnostics-every 1` for every-step physics and mode tracking.

Existing JSONL output can be displayed without rerunning the simulation. Append
`--save-diagnostics` to save instead, or both flags to save and display:

```bash
.venv/bin/python -m scripts.guiding_center.runtime.diagnostic_plots \
  run_outputs/guiding_center/diocotron_gaussian_m128_ark3_p6_h008_dt005_t70.jsonl
```


Related diagnostics share aligned panels in descriptively named figures:
`conservation.png` (invariants and absolute relative conservation errors),
`conservation_drifts.png` (signed drifts), `diocotron_instability.png`,
`poisson_stabilization.png`, and solver groups such as
`poisson_solver_residuals.png`. Groups exceeding six panels continue in
numbered pages. The same grouping is used for display and the combined PDF.

LaTeX mathematical labels clarify the quantities and units using Matplotlib
MathText, without requiring an external LaTeX installation.
Relative conservation errors are `|E(t)-E(0)|/|E(0)|` and
`|M(t)-M(0)|/|M(0)|`. They reuse recorded relative drifts and use log-log
axes when the error is positive; identically zero errors use linear axes.
Relative mass error is undefined for zero-circulation Euler gas, whose records
deliberately omit relative mass drift; its absolute circulation drift is
retained instead. Older logs missing drift fields can use a finite, nonzero
initial conserved quantity.

The actual Poisson stabilization value is recorded at diagnostic times and
plotted against time, using the denser timing history when available.
Every figure title reports its recorded constant value or range. Old logs
without this history explicitly identify unrecorded values.

## Initial projection timing

With `--verbosity 3`, initial and equilibrium density projection print a compact
line with total, setup, sampling, and coefficient-projection times, and save
`<prefix>_initial_density_projection_profile.json` (and the corresponding
equilibrium profile). These files are written immediately after projection,
before the first Poisson solve; the same details enter the initial JSONL row.
The full phase breakdown is kept in these files instead of a terminal table.

The existing projection total includes richer host quadrature preparation,
backend import, device initialization or earlier queued work, and the shared
device mesh/reference mirror subsequently reused by the solvers. Additional
phases separate the projection operator, quadrature coordinate mapping,
initial-field evaluation, coefficient matrix products, and field wrapping.
First-use allocation and kernel/library setup remain charged to the phase
that triggers them; the field-evaluation phase includes Gaussian device-data
preparation. These are wall times, not pure GPU kernel times.

Detailed profiling synchronizes each phase and accumulates repeated batch
costs, so its total can differ from the normal asynchronous execution.
Use verbosity below 3 for the normal batching behavior. Existing logs contain
only the projection total and cannot recover this breakdown retrospectively.

## Override the final time

Append `--final-time 10` after a preset response file to stop at T=10.
The runner derives the step count from the effective timestep: with the ITER
preset's dt=0.05 this gives 200 steps. An explicit `--dt` takes precedence over
the preset, so `--dt 0.025 --final-time 10` gives 400 steps.

The final time must be finite, nonnegative, and an integer multiple of dt;
the fixed-step schemes retain their timestep. Incompatible values produce
an error instead of rounding the requested endpoint or changing dt.
`--final-time` and an explicit `--num-steps` are mutually exclusive.
`--final-time 0` performs initialization only, like `--num-steps 0`.

## Completion and plot display

After the last accepted step, the runner closes the diagnostic records and
prints the final summary with CSV/JSONL/log paths before closing live viewers
or opening Matplotlib windows. Plot-cleanup and diagnostic-rendering errors
are reported separately from numerical completion.

With both `--save-diagnostics` and `--plot-diagnostics`, PNG/PDF output is
saved and its directory printed before GUI figures are created. Closing the
diagnostic windows returns control to the terminal. This ordering preserves
saved diagnostics even if a Qt platform-plugin failure aborts the process.

When comparing schemes using the same preset, set different
`--diagnostics-prefix` values, e.g. `euler_gas_si_euler` and
`euler_gas_predictor_corrector`. Otherwise the preset's existing output
filenames are reused, including when `--time-scheme` is overridden.

## Compact interactive diagnostics

`--plot-diagnostics` opens at most three overview figures: conservation,
field evolution/instability, and solver convergence with essential timings.
Diocotron angular-mode activity adds one figure, for a maximum of four.
Each overview contains at most six aligned panels, omitting missing quantities.
The instability norm retains its log-log view, and Poisson tau is included.

`--save-diagnostics` still saves the complete diagnostic collection as grouped
PNG/PDF files. With both flags, the complete collection is saved first and only
the compact overview is displayed. Internal stage details, memory counters,
and other secondary histories remain available in the saved collection.

## Scaling the transport upwind stabilization

Append `--transport-upwind-factor 1.25` to a run command to use
`tau_adv = 1.25 * abs(beta.n)`. This overrides the preset stabilization and
`--transport-advection-stabilization` if both options are supplied. The default
upwind factor remains 1; the two recovered-field presets use conflict averaging.
An explicit factor replaces that policy and uses the original side velocities.
Factors must be finite and positive. Factors below 1 are accepted but lose the
usual upwind stabilization bound.

The Python solver option is:

```python
from hybridge.solvers import AdvectionReactionHDGOptions, ScaledUpwind

options = AdvectionReactionHDGOptions(
    advection_stabilization=ScaledUpwind(1.25),
)
```

The same policy is accepted by assembly/reconstruction entry points through
`advection_stabilization`. A plain scalar still specifies absolute tau, not a
multiplier. NumPy, Numba, CuPy, and raw CUDA (fused, split3, precomputed) use the
same sidewise weights `tau = factor*abs(beta.n)` and `gamma = tau-beta.n`.
Existing zero-flux boundary handling is preserved. Raw CUDA specializes the
factor in generated code; no AMGX or PyAMGX rebuild is needed.

CPU-only checks (no JIT or CUDA compilation):

```bash
NUMBA_DISABLE_JIT=1 HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=0 PYTHONPATH=. \
  .venv/bin/python -m pytest -q tests/test_transport_lax_friedrichs.py
```

To explicitly compile and check the tiny CUDA assembly/reconstruction cases:

```bash
NUMBA_DISABLE_JIT=1 HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1 PYTHONPATH=. \
  .venv/bin/python -m pytest -q tests/test_transport_lax_friedrichs.py
```

Archived noncircular band equilibria can now be imported with `--case imported_equilibrium`; see [transfer, bounded preservation checks and measured limitations](../studies/equilibrium_preservation/README.md).
