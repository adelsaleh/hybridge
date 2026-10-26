# BDF2 with reduced-order Poisson and recovered electric field

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

The existing native BSR timing study (local, untracked: `artifacts/native_hp_bsr_breakdown_20260913/`)
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
See the failure diagnosis and validation command (local, untracked: `artifacts/recovered_drift_rank_20260917/`).
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
[`TODO.md`](../../../TODO.md#device-postprocessing).

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
  -m scripts.guiding_center.studies.recovered_field.compare_field_recovery --density-order 6
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

## Electric-field convergence diagnostic

Use a fixed-domain refinement study to measure the field order. The earlier
one-triangle comparison shrinks the domain with the element and cannot serve
as this acceptance check. The following bounded diagnostic uses manufactured
Poisson data on the unit square, at most 128 triangles, NumPy assembly and
SciPy direct matrix solves, with no time integration or native compilation
(the optional `--subdivisions` can extend the check to 512 triangles):

```bash
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m scripts.guiding_center.studies.recovered_field.check_field_recovery_order \
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
  .venv/bin/python -B -m scripts.guiding_center.studies.recovered_field.check_field_recovery_order \
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
  .venv/bin/python -B -m scripts.guiding_center.studies.recovered_field.check_oversampled_recovery_order \
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

Further independent small-patch checks on Gmsh meshes (local, untracked: `artifacts/recovery_small_patch_20260916/`)
cover vertex fits, numerical normal-flux/interior-moment fits, and source-
constrained zero-curl fits over one, four or eight elements. Polynomial
reproduction passed, but the numerical HDG inputs still did not demonstrate
an additional electric-field order. They remain diagnostic experiments.

Local HDG-response fits and structured-mesh comparisons (local, untracked: `artifacts/recovery_hdg_response_20260916/`)
also remain unqualified. Accounting for the discrete HDG response on eight-
element patches reduced errors but did not demonstrate the extra order.
Structured-mesh sine tests showed apparent seventh-order slopes, but polynomial
checks retained roughly sixth-order field errors. Mesh-scaled stabilization
also failed to establish the requested gain. These results do not change the
production mesh or select a recovery method.


Compact enriched corrections and a reconstructed-operator experiment (local, untracked: `artifacts/recovery_reconstructed_operator_20260916/`)
separate two further possibilities. Independent 4-, 8-, and 12-triangle
corrections retain the original field order. Reconstructing missing trace
modes inside the global operator gives near-seventh-order fields on a refined
Gmsh sequence, using two-triangle edge maps, but changes the Poisson
discretization: it is not postprocessing of the existing DG(p-1) solve.
Its trace-degree-five matrix has about twice the scalar nonzeros of the
standard degree-six matrix in that check. Total cost remains unqualified,
and this experiment does not select a production preset.
