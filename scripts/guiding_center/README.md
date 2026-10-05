# Guiding-center scripts

Run commands from the repository root. All six time schemes use the same case
CLI, `run_guiding_center_cases.py`; its AMGX launcher forwards to that CLI:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases --help
HYBRIDGE_PYTHON="$PWD/.venv/bin/python" scripts/guiding_center/run_local_amgx_cases.sh --help
```

The CLI selects a preset, applies overrides, and starts the shared runner. It
contains no time-stepping equations. Guiding-center evolution composes the
package diffusion and advection HDG solvers, so its temporal schemes belong
in [`time_schemes/`](time_schemes/), alongside case policy and orchestration.
The `hybridge` package supplies the HDG solvers, assembly, field/trace operations,
and shared diagnostics and I/O.

| Directory | Contents |
|---|---|
| `cases/` | Case definitions and curated run presets. |
| `time_schemes/` | SI Euler, predictor-corrector, SI BDF2, SI BDF3, H1/H2-BDF3, IMEX-ARK3, and their common registry and stage data. |
| `runtime/` | CLI arguments, configuration, stepper construction, execution, diagnostics, plotting policy, and terminal logging. |
| `poisson/` | Poisson-tau recovery, backend comparisons, AMGX tuning, saved-system replays, and hierarchy product measurements. |
| `diagnostics/` | Diocotron analysis/reference spectra, preflight checks, timing analysis, and visualization smoke checks. |
| `benchmarks/` | Host solver stage comparisons, precision benchmarks, and temporal convergence/comparison drivers. |
| `reference/` | The independent DOLFINx CG/SUPG torsion case. |

## Schemes and tested presets

The time-scheme registry drives both `--time-scheme` choices and execution. Every
scheme has its own module; SI BDF2 and predictor-corrector reuse SI Euler's
stage-solve and accepted-history handling, and SI BDF3 extends SI BDF2. All
return the same `GuidingCenterStep` result. H1/H2 share their existing BDF3
histories and third-order startup. ARK3 retains its frozen transport operator.
All seven schemes support bounded Poisson-tau recovery.

SI BDF3 is the plain one-solve analogue of SI BDF2: one linear transport solve
with third-order extrapolated drift, then the endpoint Poisson solve,

```text
(I + 6 dt/11 A(3 v_n - 3 v_(n-1) + v_(n-2))) rho_(n+1)
    = (18 rho_n - 9 rho_(n-1) + 2 rho_(n-2)) / 11.
```

The temporal algebra is `hybridge.core.time_integration.bdf3_transport_data`. Its
startup keeps third order: step 1 is Richardson-extrapolated SI Euler
(`2 E(dt/2) - E(dt)`: three transport and two Poisson solves), step 2 is SI BDF2.
A plain Euler/BDF2 ramp would limit the global order to two. The extrapolated
first step is not positivity preserving. Recovered drift
(`--transport-electric-field postprocessed`, `--poisson-order-offset -1`) works
as for SI BDF2. Per-step records add `bdf3_startup`, `bdf3_startup_method`
(`si-euler-extrap2`, `si-bdf2`, or none) and `transport_time_order`. Canned
scalar checks in `tests/test_guiding_center_bdf3.py` measure global order 3. No
PDE timestep-stability qualification has been made; BDF3 is only A(alpha)-stable.

| `--time-scheme` | Scheme module | Tested vortex-gas preset |
|---|---|---|
| `si-euler` | `si_euler.py` | `euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr` |
| `predictor-corrector` | `predictor_corrector.py` | `euler_vortex_gas_predictor_corrector_p6_h008_dt005_t50_raw_cuda_bsr` |
| `si-bdf2` | `si_bdf2.py` | `euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr` |
| `si-bdf3` | `si_bdf3.py` | `euler_vortex_gas_si_bdf3_p6_h0068_dt005_t50` |
| `h1-bdf3` | `h1_bdf3.py` | `euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr` |
| `h2-bdf3` | `h2_bdf3.py` | `euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr` |
| `imex-ark3` | `imex_ark3.py` | `euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr` |

These are the existing tested presets and response files. Their timesteps,
startup policies, stabilization and solver settings are preserved. The H1/H2
and ARK3 rows use `dt=0.005`, `poisson_tau=1000`, and 10,000 steps to `T=50`;
the lower-order rows use their existing `dt=0.05` configurations. SI BDF3
presets mirror the corresponding SI-BDF2 presets field for field, except the
scheme and output prefix: `euler_vortex_gas_si_bdf3_p6_h0068_dt005_t50`,
`positive_turbulence_si_bdf3_p6_h0068_dt0005_t50_raw_cuda_bsr`,
`positive_turbulence_iter_fft_si_bdf3_p6_h014_dt0005_t50_raw_cuda_bsr` and
`diocotron_gaussian_m64_si_bdf3_p6_h0068_dt05_t400` (see `example_runs.md`). Host accuracy
and diocotron presets remain available through `--list-presets`, which now
shows each preset's scheme and timestep.

Inspect a tested configuration without running it:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr.args \
  --dry-run
```

The same CLI accepts a positional preset name or `--preset NAME`. Use the
scheme's own preset to retain its tested configuration; `--time-scheme` changes
only the algorithm and does not silently replace the other preset settings.
Precision selection still happens before numerical imports. No rebuild is
required for this Python refactor.

## BDF2 with reduced-order Poisson and recovered electric field

Two comparison presets now provide RT or L2-closest recovery, with reduced
accuracy accepted. Both use the existing unstructured Gmsh disk mesh and
cached, device-resident raw CUDA recovery on every Poisson solve.

The earlier superconvergence research target is density in DG(p), potential
and raw electric field in DG(p-1), and a recovered electric field in DG(p) with L2 error of
order p+1 for sufficiently smooth solutions. At p=6 this requires seventh-order
field convergence from a degree-5 Poisson solve. Recovery must not require
another global solve. All degrees must use the existing unstructured Gmsh
mesh, including the same nodes and elements; recovery may use neighbouring
elements. Changing to a structured or symmetry-enforced mesh is excluded.
Each output patch must be computable independently from the frozen lower-degree solution and
known source/boundary data, using only a small edge, triangle, or bounded
combination of neighbouring elements. Geometry-dependent operators may be
cached. Corrections must not feed into other patches: sequential sweeps,
iterative exchange between patches, and global correction iterations are
excluded. A smaller error constant or higher stored degree does not meet the
accuracy contract.

Cost is a separate acceptance condition: the lower-degree Poisson solve plus
source restriction, field recovery, transfers, and amortized recovery setup
must cost less than the original degree-p Poisson solve. A local method is not
acceptable merely because it avoids another global solve. Both convergence
and total cost must be established before selecting a replacement preset.

The existing [native BSR timing study](../../artifacts/native_hp_bsr_breakdown_20260913/README.md)
provides an indicative budget on an identical 315,425-triangle mesh (both node
and connectivity hashes match). Complete cached Poisson calls, averaged over
steps 5–12 of each `samples.jsonl`, took:

| Poisson degree | Trace unknowns | Mean complete call |
|---|---:|---:|
| 5 | 2,834,898 | 167.153 ms |
| 6 | 3,307,381 | 198.238 ms |

The difference is 31.085 ms, or 15.68% of the degree-6 call. This is a budget
for **all** added work, not just the recovery kernel. These archived runs use
SI Euler and change the density degree as well, so they do not measure the
requested BDF2 degree-6-density/degree-5-Poisson combination or establish its
speedup at the required accuracy. No new production run was performed.

**The presets below are comparison baselines; neither claims an additional
field L2 order.**

`euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6` keeps density in DG(p), solves
both Poisson potential and raw electric field in DG(p-1), then feeds the direct
RT_(p-1) reconstruction, stored exactly in DGVectorField(p), to BDF2 advection.
The default p is 6; `--order` changes p while retaining the one-degree offset.
Initial, accepted-endpoint, and retry Poisson solves all recover the field.
BDF2 extrapolates the two recovered accepted fields, and tau recovery rebuilds
both histories. The density RHS is L2-restricted with a cached same-mesh
projection that preserves Poisson test moments and device residency.

`euler_vortex_gas_si_bdf2_p6_poisson_p5_l2_closest_p6` uses the same settings
with `l2_closest` flux recovery and a separate output prefix. Both explicitly
set `poisson_postprocessing_backend="raw-cuda"`.

Both recovered-field presets also select
`transport_advection_stabilization="conflict-averaged-upwind"`.
At interior quadrature nodes with two nonnegative outward velocities `a,b`
and `a+b>0`, the numerical face velocities become `(a-b)/2` and `(b-a)/2`.
Both face weights use those velocities: `tau=abs(s)` and `gamma=abs(s)-s`.
This restores weighted trace support on the two saved failing faces without
increasing the upwind factor. Exactly inactive faces receive a zero-trace
algebraic gauge; partially deficient active faces still fail strictly.
The volume field is unchanged: this is a face-flux repair, not an H(div)
reconstruction or an energy-conservation guarantee.
See the [failure diagnosis and validation command](../../artifacts/recovered_drift_rank_20260917/README.md).
Poisson tau is unchanged. Other presets retain their original flux, and
`--transport-advection-stabilization upwind` restores it for comparison.

The quiet RT and L2-closest variants use the completed RT run's mesh/time settings
(h=0.0068, at least 100,000 triangles, p=6, dt=0.05, T=50):

```bash
.venv/bin/python scripts/guiding_center/run_guiding_center_cases.py \
  euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6_fast

.venv/bin/python scripts/guiding_center/run_guiding_center_cases.py \
  euler_vortex_gas_si_bdf2_p6_poisson_p5_l2_closest_p6_fast
```

Both presets disable field diagnostics, including initial/final reductions,
every-step timing CSV/JSONL files, plots, and AMGX residual-history collection.
They retain the terminal log and failure snapshots. Conflict-averaged upwind,
the respective recovery method, solver tolerances, convergence monitoring, and
retry policies are inherited from each original preset. AMGX convergence monitoring stays enabled:
`../AMGX/src/solvers/solver.cu` also uses that flag for tolerance-based stopping.
Both retain row scaling and its separate solver/physical residual checks.
For unscaled AMGX solves, HYBRIDGE now shares the identical solver/physical residual
calculation while checking both acceptance targets. Throughput has not been measured.

The RT and L2-closest paths (including the ITER RT preset) automatically reuse
projection data: the runner retains one `DiffusionReactionHDGSolver`, and
source/boundary updates preserve its postprocessing cache. Reference lifts,
trace-moment tables, uploaded device arrays and geometry are built once;
L2-closest also caches its metric factors. Each solve applies those maps to the
new fields and trace on the device. No extra cache flag is needed. A Poisson-tau
retry currently clears this recovery cache along with the operator; retaining
the tau-independent recovery data across that exceptional path is tracked in
[`TODO.md`](../../TODO.md#device-postprocessing).

The reusable controls are `--diagnostics-every 0`, `--no-record-timings`,
`--no-amgx-residual-history`, and `--verbosity 0`. To restore sampled field
diagnostics use, for example, `--diagnostics-every 10`; `--record-timings`
restores timing files independently. Do not add `--transport-upwind-factor 1`:
that would replace the preset's conflict averaging with ordinary upwind.

From the repository root, run either:

```bash
.venv/bin/python scripts/guiding_center/run_guiding_center_cases.py \
  @run_configs/guiding_center/euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6.args

.venv/bin/python scripts/guiding_center/run_guiding_center_cases.py \
  @run_configs/guiding_center/euler_vortex_gas_si_bdf2_p6_poisson_p5_l2_closest_p6.args
```

The raw CUDA path lifts the stabilization jump through a cached reference RT
map. L2-closest additionally minimizes in the constraint nullspace, using a
cached 5-by-5 metric Cholesky factor at Poisson degree 5. This replaces the
previous per-call full RT elimination and avoids large L2 constraint-factor
tables. Trace, raw coefficients, and recovered coefficients stay on the
GPU. A tau retry reuses geometry maps and applies the new tau to the jump.
Recovery timing includes stream synchronization and completed device work.

Stationary no-JIT matrix tests compare both maps with the existing host
postprocessors, including skew elements and trace orientations. CUDA parity
and throughput have not been run, so fastest measured performance is not
claimed. To explicitly compile and check the CUDA kernels without a simulation:

```bash
HYBRIDGE_RUN_CUDA_RECOVERY_TESTS=1 NUMBA_DISABLE_JIT=1 PYTHONPATH=. \
  .venv/bin/python -m pytest -q tests/test_diffusion_flux_recovery_maps.py
```

Inspect the configuration (no simulation):

```bash
NUMBA_DISABLE_JIT=1 .venv/bin/python -B -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6.args --dry-run
```

To run it, omit `NUMBA_DISABLE_JIT=1` and `--dry-run`. It inherits h=0.008, dt=0.05, T=50, native
FB-HP-MG Poisson, AMGX BSR transport, and Holoviz from the existing BDF2 preset.
The new spatial combination has not been qualified for that timestep. No
native library rebuild is required; normal execution JIT-compiles the new
raw CUDA recovery kernels for the selected polynomial degrees.

The equivalent policy flags for another BDF2 preset are:

```text
--poisson-order-offset -1 --poisson-hdg-postprocess flux
--poisson-flux-postprocess-space RT_projection --transport-electric-field postprocessed
```

Leave `--poisson-flux-postprocess-every` at zero: recovered transport drift
requires continuous recovery, not diagnostic-only cadence. Raw-CUDA supports
continuous flux recovery; the current CuPy Poisson solver does not. Host
backends support the recovery policy with
`--poisson-postprocessing-backend numba`. Diagnostics record both raw and
recovered norms, the actual drift choice and degree, and use that drift for
velocity diagnostics and electric-energy monitoring.

The direct RT method uses numerical normal-flux moments and low-order interior
flux moments. Merely elevating or L2-projecting the raw field into DG(p) would
leave the polynomial unchanged. Standard potential recovery produces a DG(p)
potential whose derivative is only degree p-1. The combined potential/flux
method needs another local solve and substantially more per-element cached
factor storage; RT needs no scalar recovery and has an existing CUDA path.
The cached raw CUDA flux-only path keeps evolving inputs and outputs on the
device. End-to-end speedup still needs measurement.

**Recovery is not a guarantee of electric-field superconvergence.** With
Poisson degree k=p-1, standard RT recovery retains the field's usual L2 order
k+1=p while improving normal-flux conformity; the scalar potential can
superconverge under suitable hypotheses. See
[Nguyen, Peraire and Cockburn, 2009](https://www.mit.edu/~cuongng/publication/pub8/pub8.pdf),
section 4. Lowering the Poisson degree may therefore reduce spatial accuracy
relative to the original degree-p Poisson solve. This preset is an explicit
cost/accuracy option, not a verified replacement at equal error.

A reproducible one-element comparison of direct RT, full-space constrained
flux, combined potential/flux, and differentiated potential recovery uses
only local matrices and prescribed exact trace data:

```bash
NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
  -m scripts.guiding_center.diagnostics.compare_field_recovery --density-order 6
```

For p=6 on the unit triangle with potential exp(x+y/2), the diagnostic gave:

| Recovery | Electric-field L2 error | Stored electric-field degree |
|---|---:|---:|
| Raw Poisson field | 2.7691e-7 | 5 |
| Direct RT baseline | 2.2901e-7 | 6 |
| Full-space constrained flux | 2.3445e-7 | 6 |
| Combined potential/flux | 2.1874e-7 | 6 |
| Differentiated recovered potential | 2.7764e-7 | 5 |

RT reduced this local error by 17.3%. Combined recovery reduced it a further
4.5% relative to RT, but uses a 51-constraint full-flux solve and a 29-unknown
scalar recovery, with elementwise cached factors. RT uses one 48-unknown local
moment system in the original implementation. The new cached CUDA path uses
a reference lift instead; L2-closest uses a 5-unknown metric correction.
The original system sizes describe the baseline costs, not throughput, and
do not establish suitability for the additional-order requirement.

This diagnostic cannot establish global convergence or GPU throughput. The
focused regression checks likewise require no build or time integration:

```bash
NUMBA_DISABLE_JIT=1 .venv/bin/python -B -m pytest -q --assert=plain \
  tests/test_guiding_center_recovered_field.py
```

### Electric-field convergence diagnostic

Use a fixed-domain refinement study to measure the field order. The earlier
one-triangle comparison shrinks the domain with the element and cannot serve
as this acceptance check. The following bounded diagnostic uses manufactured
Poisson data on the unit square, at most 128 triangles, NumPy assembly and
SciPy direct matrix solves, with no time integration or native compilation
(the optional `--subdivisions` can extend the check to 512 triangles):

```bash
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m scripts.guiding_center.diagnostics.check_field_recovery_order \
  --density-order 6 --jitter 0.15
```

Use `--jitter 0` for uniform triangles. The degree-p Poisson result in the
output is an accuracy reference obtained with a separate global solve; it is
not a local postprocessor. Reported rates include boundary elements and use
full-domain electric-field L2 errors. These small tests do not qualify disk
geometry, time integration, GPU precision, or throughput.

For p=6, tau=1 and jitter=0.15, the 8/32/128-triangle refinement check gave:

| Electric field | L2 error at 128 triangles | Order, 8 to 32 triangles | Order, 32 to 128 triangles |
|---|---:|---:|---:|
| Raw DG(5) | 8.1820e-8 | 5.76 | 6.10 |
| RT recovery into DG(6) | 2.5495e-8 | 5.81 | 6.01 |
| Full-space constrained flux | 5.4806e-8 | 5.74 | 6.08 |
| Combined potential/flux | 1.6767e-8 | 5.87 | 5.90 |
| Differentiated recovered potential | 9.4305e-8 | 5.75 | 6.12 |
| Separate degree-6 Poisson reference | 2.1938e-9 | 6.77 | 6.87 |

None of the existing local recoveries demonstrates the required order seven.
The p=3 checks likewise give RT orders 3.00 on uniform triangles and 2.92 on
perturbed triangles on the last refinement, versus the required four; the
separate degree-3 reference gives 3.98 and 4.08 respectively. These are measured
rates for this manufactured example, not guarantees for arbitrary data.

The diagnostic also accepts `--case poly` for potential (x+y/2)^(p+1) and
`--case harmonic` for real((x+i y)^(p+1)). It reports errors in the raw field's
DG(p-2) interior moments and the potential's cell means. These distinguish
scalar superconvergence from the stronger flux information that a cheap
moment-based patch recovery would need. For example:

```bash
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m scripts.guiding_center.diagnostics.check_field_recovery_order \
  --density-order 3 --case poly --subdivisions 4 8 16 --jitter 0.15
```

In this polynomial check with tau=1, the last-refinement raw field order was
3.03, its DG(1) moment order was 3.10, and the potential mean order was 4.24.
The moments do not supply the order-four field data needed to justify direct
polynomial recovery on these perturbed meshes.

Polynomial patch recovery needs additional analysis before use here. For
example, [Bank and Li's RT recovery analysis](https://arxiv.org/abs/1802.04963)
proves results for the lowest and second-lowest RT orders on mildly structured
meshes. That result alone does not justify degree-5 HDG recovery on the
unstructured production mesh.

An enriched global residual correction is excluded by the recovery
requirement. No such correction is enabled in the runner.

A further diagnostic explores independent Poisson solves on overlapping
patches, taking artificial-boundary data from the standard recovered DG(p)
potential and retaining the degree-p electric field only in each patch's core:

```bash
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m scripts.guiding_center.diagnostics.check_oversampled_recovery_order \
  --density-order 6 --case trig --jitter 0.15
```

This is a square-domain manufactured-problem experiment, not a production
postprocessor or a qualified BDF2 preset. It uses one degree-(p-1) base solve
and four independent degree-p patch solves, each a proper subset of the mesh.
There is no enriched global system and no iteration between patches. The
finest base mesh has 512 triangles; each patch has at most 288.

On perturbed meshes with tau=1, measured electric-field orders were:

| Density degree | Exact potential | Raw field orders | Patch field orders |
|---|---|---|---|
| 3 | (x+y/2)^4 | 2.89, 3.03 | 4.00, 4.00 |
| 6 | sin(2 pi (x+0.23)) sin(2 pi (y+0.17)) | 5.85, 6.14 | 7.10, 6.73 |
| 6 | real((x+i y)^7) | 5.93, 6.00 | 6.95, 6.63 |

These are promising limited observations, not a guarantee for the unstructured
disk or arbitrary polynomial degrees. In particular, the finest degree-6
harmonic result has L2 error 2.67e-12 and is sensitive to roundoff.

The overlap is fixed in physical units (1/4 of the unit-square side), so its
element count grows with refinement. The four patches cover 2.25 times the
original element count in total at the enriched degree, in addition to the
base solve and scalar recovery. This experiment is excluded as the proposed
production method: it has not demonstrated that recovery fits within the
savings from the smaller base solve. Replacing the overlap by a fixed number
of element layers would be a different method and would need a separate
convergence and cost check.

Simple polynomial least-squares patches were also screened. Fitting a
DG(p+1) potential to lower-degree moments and differentiating appeared to gain
an order for trigonometric data, but did not robustly retain that gain for
polynomial data. Imposing the Poisson source on that fit did not repair the
failure. These fits have not been selected as the production recovery.

Further [independent small-patch checks on Gmsh meshes](../../artifacts/recovery_small_patch_20260916/README.md)
cover vertex fits, numerical normal-flux/interior-moment fits, and source-
constrained zero-curl fits over one, four or eight elements. Polynomial
reproduction passed, but the numerical HDG inputs still did not demonstrate
an additional electric-field order. They remain diagnostic experiments.

[Local HDG-response fits and structured-mesh comparisons](../../artifacts/recovery_hdg_response_20260916/README.md)
also remain unqualified. Accounting for the discrete HDG response on eight-
element patches reduced errors but did not demonstrate the extra order.
Structured-mesh sine tests showed apparent seventh-order slopes, but polynomial
checks retained roughly sixth-order field errors. Mesh-scaled stabilization
also failed to establish the requested gain. These results do not change the
production mesh or select a recovery method.


[Compact enriched corrections and a reconstructed-operator experiment](../../artifacts/recovery_reconstructed_operator_20260916/README.md)
separate two further possibilities. Independent 4-, 8-, and 12-triangle
corrections retain the original field order. Reconstructing missing trace
modes inside the global operator gives near-seventh-order fields on a refined
Gmsh sequence, using two-triangle edge maps, but changes the Poisson
discretization: it is not postprocessing of the existing DG(p-1) solve.
Its trace-degree-five matrix has about twice the scalar nonzeros of the
standard degree-six matrix in that check. Total cost remains unqualified,
and this experiment does not select a production preset.

## Poisson-tau fallback for every scheme

SI Euler, predictor-corrector, SI BDF2, SI BDF3, H1-BDF3, H2-BDF3, and IMEX-ARK3 all
retry numerical transport failures through the same backend-independent tau
policy. This includes direct-factorization failures and iterative nonconvergence,
as well as diagnosed active trace-rank loss. Capacity, configuration, and
unrelated programming errors propagate without increasing tau.

The default is `--poisson-tau-retry-factor 2 --poisson-tau-max-retries 4`.
No additional flags are needed in existing commands. A failed transport stage
backtracks to the Poisson source, boundary time, and owning trace that supplied
its drift. Recovery doubles tau, invalidates stale Poisson and transport
operators/preconditioners, repeats that Poisson solve, and replays the whole
unaccepted timestep. Accepted densities and time remain unchanged on exhaustion.
The increased tau persists after success; `--poisson-tau-max-retries 0` disables
recovery.

BDF2 rebuilds its accepted flux history; SI BDF3 rebuilds both accepted flux
levels; H1 rebuilds the Poisson-derived residual
history; H2 rebuilds its drift history. Startup and predictor/corrector stages
also participate. IMEX-ARK3 rebuilds its frozen operator and all stage residuals.
The shared replay orchestration is in `time_schemes/recovery.py`; tau policy and
failure classification are in `poisson/poisson_recovery.py`.

Retries record their old/new tau, failure, and Poisson checkpoint immediately in
`<diagnostics-prefix>_poisson_tau_recovery.jsonl`, including on exhaustion.
Completed and failed retry work is included in step recovery diagnostics. A tau
change changes the discrete Poisson operator and should be accounted for when
comparing accuracy or conservation across runs.

## Common entry points

The examples below only display options; invoking a runner without `--help`
may perform solves or time integration.

```bash
# Saved-system hybrid hierarchy CSR/BSR feasibility study
.venv/bin/python -m scripts.guiding_center.poisson.benchmark_hybrid_hierarchy_bsr --help

# Replay existing Poisson captures
.venv/bin/python -m scripts.guiding_center.poisson.replay_poisson_bsr --help
.venv/bin/python -m scripts.guiding_center.poisson.replay_poisson_asm --help

# Temporal comparisons and precision benchmarks
.venv/bin/python -m scripts.guiding_center.benchmarks.run_guiding_center_temporal_convergence --help
.venv/bin/python -m scripts.guiding_center.benchmarks.run_precision_benchmark --help

# Analyze existing output and inspect a diocotron case
.venv/bin/python -m scripts.guiding_center.diagnostics.analyze_poisson_timings --help
.venv/bin/python -m scripts.guiding_center.diagnostics.analyze_diocotron --help
.venv/bin/python -m scripts.guiding_center.diagnostics.preflight_diocotron --help
```

Use the [CUDA launcher](../gpu/run_cuda13.sh) and the repository's configured
runtime environment for GPU work. No native library or Python package rebuild
is needed for this directory reorganization.

## Imports and saved studies

Presets come from `scripts.guiding_center.cases.guiding_center_presets`.
Time-stepping classes come from `scripts.guiding_center.time_schemes`.
They reuse `hybridge` helpers to compose the existing HDG solvers.
Programmatic study drivers import
`run_guiding_center_case` from `scripts.guiding_center.runtime.runner`, and
snapshot/result types from `scripts.guiding_center.runtime.models`. They reuse
the same execution path as the user CLI. Tests and benchmark drivers contain
validation and study orchestration, not separate scheme implementations.

Repository imports, current commands and documentation links use the new
locations. Historical files under `artifacts/` and `run_outputs/` retain their
original source paths and hashes. When adapting an archived command, add the
appropriate directory from the table above to its old module path.

## Plotting ownership

Reusable rendering belongs to `hybridge.io`. The runtime's PyVista adapter supplies
case labels and color policies to `PyVistaFieldPanels`; the package owns mesh
sampling, overlays, in-place scalar updates, linked views, headless rendering,
and screenshot output. Vorticity uses fixed symmetric initial limits, while
density and potential use their current robust ranges. The existing
`hybridge.io.holoviz` backend retains GPU sampling and asynchronous rendering.

Temporal field comparisons use `plot_scalar_raster_panels_matplotlib` with
`RasterGeometry` bounds and `scalar_color_limits`. Mesh holes stay masked, the
two vorticity panels share a symmetric scale, and their difference has its own
scale. `VorticityRaster` already uses the package's raster geometry and sampling
operators. Diagnostic histories, growth fits, and convergence-study labels
remain in the scripts because they depend on those studies' observables.
The independent DOLFINx reference retains its own field-to-VTK adapter.
Incremental diagnostic JSONL/CSV writing is shared through `hybridge.io.records`.

Other package clients can construct a live viewer without importing a runner:

```python
from hybridge.io import PyVistaFieldPanels

viewer = PyVistaFieldPanels(
    [("Scalar", field, {"symmetric_clim": True, "fixed_clim": True,
                        "robust_percentile": 100.0})],
    off_screen=True, screenshot_dir="frames",
)
try:
    viewer.update([field], step=0, time_value=0.0)
finally:
    viewer.close()
```

PyVista and Matplotlib are imported only when their rendering helpers are used.
The PyVista path uses the normal host `DGField.values_at_ref` interface; use
Holoviz for GPU-resident rendering without downloading field coefficients.

## Positive guiding-center turbulence

`positive_turbulence` is an HYBRIDGE-defined density analogue of the signed Euler
vortex gas. It uses the same 360 blobs, four Gaussian widths, amplitude range,
and seed, but all strengths are positive. The shared indexed Gaussian profile
truncates every blob at eight standard deviations. Centers are sampled in the
exact unit disk with enough scale-dependent clearance to leave `rho_0 = 0`
throughout `0.96 < r <= 1`; consequently the analytic initial density is
nonnegative on the whole disk and has no tails or cores at the wall.
Passing `geometry=iter`, `horseshoe`, or `pacman` samples the same positive
profile in that actual shaped domain, retaining the prescribed physical
`wall_gap`; mesh selection follows the case geometry automatically.

The guiding-center density equation and the incompressible Euler vorticity
equation have the same mathematical form up to the Poisson sign convention, so
this profile is equivalently one-signed Euler vorticity. Nonnegative vorticity
in domains with material boundaries is a standard analytical setting, and
same-sign vortex gases also occur in related two-dimensional turbulence work.
The particular disk, population, widths, and cutoff here are not a reproduction
of a published benchmark; they define a reproducible multiscale HYBRIDGE stress
test. See [Cai, Guo, and Qiu](https://arxiv.org/abs/1804.02365),
[Iftimie et al.](https://arxiv.org/abs/1305.0905), and
[van Kan et al.](https://arxiv.org/abs/2308.08789) for the three respective
connections. Published guiding-center positivity studies commonly use a
nonnegative diocotron annulus, as in
[Kiechle, Chudzik, and Helzel](https://doi.org/10.1016/j.jcp.2024.113693);
this gas complements the repository's existing annular cases instead of
replacing them.

The prepared IMEX-ARK3 run enables `--positivity-diagnostics`. It audits the
projected initial field, all three transported ARK stage densities, and the
accepted endpoint. This is a detection test: the current high-order transport
does not apply a positivity limiter, and a negative projected or evolved value
is reported rather than clipped.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr.args \
  --verbosity 3 --save-diagnostics
```

The preset is a user-run qualitative/positivity trial at `h=0.008`, DG p=6,
`dt=0.005`, and `T=50`; no production time integration has yet qualified its
stability or positivity behavior. The initial audit is also written to
`<diagnostics-prefix>_initial_positivity.json` before the first time step.


For a quiet ITER SI-BDF2 run with density DG(p), Poisson DG(p-1), and RT flux
recovery into DGVectorField(p), use:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_iter_si_bdf2_p6_poisson_p5_rt_p6_fast.args
```

This inherits the ITER case's 11,520 blobs, h=0.014, 300k+ triangles, dt=0.005
to T=50, and robust Poisson/transport retries. At the default p=6, Poisson uses
DG(5); recovered RT flux supplies every transport stage, including startup and
retries. The conflict-averaged upwind guard is active. Field/positivity diagnostics,
plots, timing files, and AMGX residual history are off, with convergence checks
and tolerances retained. This combination has configuration checks but no new
simulation or throughput measurement.

To enable the diagnostic output of the original command, append
`--positivity-diagnostics --diagnostics-every 10 --plot-diagnostics --plot-every 100 --verbosity 3`.
The fast preset has its own output prefix. `--order p` retains the one-degree
Poisson offset. Do not append `--transport-upwind-factor 1`, which would replace
conflict averaging with ordinary upwind.

## Horseshoe, ITER, and Pac-Man Euler gas

These presets use DG p=6, dt=0.05 and 1,000 steps to T=50 with the existing
raw-CUDA/native FB-HP-MG Poisson and AMGX transport stack. Most use IMEX-ARK3;
the denser ITER variant uses SI-BDF2 with one SI-Euler startup step.
Holoviz and physics diagnostics update every ten steps; solver timings remain
every-step. Initial projection uses 16 points per coordinate.

| Response-file preset | Mesh size | Minimum triangles | Vortices | Gaussian widths |
|---|---:|---:|---:|---|
| `euler_horseshoe_gas_imex_ark3_p6_150k_t50` | 0.0048 | 150,000 | 360 | .004, .008, .016, .032 |
| `euler_iter_gas_imex_ark3_p6_300k_t50` | 0.014 | 300,000 | 5,760 | .010, .020, .040, .080 |
| `euler_iter_gas_si_bdf2_p6_300k_t50` | 0.014 | 300,000 | 11,520 | .010, .020, .040, .080 |
| `euler_pacman_gas_imex_ark3_p6_150k_t50` | 0.006 | 150,000 | 360 | .008, .016, .032, .056 |

The horseshoe is an annular sector with radii .48 and 1 and a 60-degree gap.
Pac-Man removes a 60-degree sector from the unit disk. Both openings face +x.
ITER uses the supplied [ITER.geo](../../hybridge/core/geometries/ITER.geo),
including the detailed lower wall, in its original coordinates (area about
30.73). Gmsh meshes the original Line/BSpline/Spline curves; vortex placement
samples those curves for wall containment and clearance. The physical wall
and interior labels remain 1 and 2. The h=0.014 target is sized for 300k+
triangles; the runner checks the actual count. No prebuilt mesh is bundled.
Gaussian widths are scaled to .010, .020, .040, .080 for this larger domain.

The ITER ARK3 counts are `(3072, 1536, 768, 384)`, 16 times the disk
population. The SI-BDF2 variant doubles them to `(6144, 3072, 1536, 768)`,
or 11,520 vortices and 32 times the disk population.
Centers are independent uniform-area samples in the eligible interior, with
no imposed reflection symmetry. Each Gaussian is cut off at eight widths,
where its relative value is about 1.27e-14, and its entire support stays inside
the wall. Signed strengths balance continuous circulation at each scale.
The shared spatial index evaluates nearby blobs on NumPy/CuPy, avoiding a
full mesh-points-by-vortices allocation. This cutoff defines initial data;
it does not limit or clip the evolving field.

```bash
# from the repository root
source .env

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_horseshoe_gas_imex_ark3_p6_150k_t50.args \
  --verbosity 3 --save-diagnostics

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_iter_gas_imex_ark3_p6_300k_t50.args \
  --verbosity 3 --save-diagnostics

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_iter_gas_si_bdf2_p6_300k_t50.args \
  --verbosity 3 --save-diagnostics

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_pacman_gas_imex_ark3_p6_150k_t50.args \
  --verbosity 3 --save-diagnostics
```

These are prepared user-run cases. Production triangle counts and time-integration
stability have not been measured; the existing minimum-triangle guard rejects an
undersized mesh. Small geometry/topology, profile, CLI, and plotting checks do
not establish long-run convergence. Adjust dt and num-steps together to retain T=50.

## Matplotlib diagnostic histories

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

### Initial projection timing

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

### Override the final time

Append `--final-time 10` after a preset response file to stop at T=10.
The runner derives the step count from the effective timestep: with the ITER
preset's dt=0.05 this gives 200 steps. An explicit `--dt` takes precedence over
the preset, so `--dt 0.025 --final-time 10` gives 400 steps.

The final time must be finite, nonnegative, and an integer multiple of dt;
the fixed-step schemes retain their timestep. Incompatible values produce
an error instead of rounding the requested endpoint or changing dt.
`--final-time` and an explicit `--num-steps` are mutually exclusive.
`--final-time 0` performs initialization only, like `--num-steps 0`.

### Completion and plot display

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

### Compact interactive diagnostics

`--plot-diagnostics` opens at most three overview figures: conservation,
field evolution/instability, and solver convergence with essential timings.
Diocotron angular-mode activity adds one figure, for a maximum of four.
Each overview contains at most six aligned panels, omitting missing quantities.
The instability norm retains its log-log view, and Poisson tau is included.

`--save-diagnostics` still saves the complete diagnostic collection as grouped
PNG/PDF files. With both flags, the complete collection is saved first and only
the compact overview is displayed. Internal stage details, memory counters,
and other secondary histories remain available in the saved collection.

### Scaling the transport upwind stabilization

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
