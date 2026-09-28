# Device Diagnostic Reductions

Guiding-center diagnostic records use resident DG coefficients and download
compact reductions. Reporting a step does not require downloading density,
potential, electric-field coefficients, quadrature tables, traces, or matrices.
The diagnostic APIs return ordinary Python numbers suitable for JSONL/CSV.

| Operation | Device-to-host result per call |
|---|---|
| `guiding_center_field_diagnostics` | One vector of 7 scalars; up to 18 with recovered-flux energy, equilibrium errors, and density harmonics |
| `azimuthal_mode_diagnostics` | One vector of 5 scalars for a positive mode; no transfer when the mode is disabled |
| `transport_velocity_diagnostics` | One vector of 10 scalars |
| `evaluate_scalar_error(..., include_samples=False)` | One vector of 4 scalars |
| `ScalarPositivityDiagnostics.measure` | One vector of 8 scalars |
| `DiocotronModeDiagnostics.measure` | One vector containing the resolved mode amplitudes and 3 extra reductions; length depends on the configured angular resolution, not the number of mesh cells |
| `solver_result_metrics` | No field/system transfer; reads the solver's existing residual, iteration, retry, and timing summaries |

`guiding_center_field_diagnostics` reduces mass, squared density L2 norm,
standard/recovered flux norms, density/potential extrema, equilibrium error
norms, and three density harmonic amplitudes. The runner derives energy,
enstrophy, and conservation drifts from these scalar values. Its velocity
diagnostics also reduce boundary-normal flux, interior normal jumps,
divergence, and speed estimates. Growth fits and active-mode rankings consume
the recorded scalar amplitudes rather than full field histories.

The core runner downloads two vectors per record: the field diagnostics and
velocity diagnostics. Manufactured density and potential errors add one
four-scalar vector each. Optional positivity and polar Fourier diagnostics
add their respective vectors. These counts remain independent of mesh size
and are checked across repeated reports.

## Backend And Sampling Contract

The public field, density-harmonic, velocity, and scalar-error helpers accept
`backend="auto"|"host"|"device"`. `auto` selects device reductions when the
relevant fields have resident coefficients; `host` explicitly permits host
materialization. The device contract is qualified on one active CUDA device.
Initial upload of host fields and reference/geometry data remains intentional;
subsequent reports reuse resident field coefficients.

Density and potential may use different spaces on the same mesh. Standard and
recovered flux norms use each component's own mass matrix, including degree
`p+1` recovery. Device equilibrium density must use the density's DGSpace, and
equilibrium potential must use the potential's DGSpace. Velocity components
must share one scalar DGSpace. The diagnostic reductions operate on DG fields
and are independent of the solver's nodal/modal trace representation.

The public density-harmonic helper shares its calculation with the combined
guiding-center diagnostic. It evaluates the `k`, `2k`, and `3k` sine/cosine
moments of the density perturbation, normalized by the absolute equilibrium
mass. A nonpositive mode disables these moments. The tiny-denominator rule
remains the same on host and device.

Default extrema and scalar Linf errors are sampled on volume quadrature.
An explicit `sample_resolution` changes the scalar-error sampling grid.
These are sampled maxima, not certified polynomial bounds. Positivity
diagnostics separately distinguish sampled violations from Bernstein bounds.

`evaluate_scalar_error` downloads plotting samples only when
`include_samples=True`. The runner explicitly requests `False`. The shared
scalar-packing helper rejects unreduced arrays before a download. Plotting,
screenshots, failure snapshots, and explicitly requested host fields have
their own transfer policies.

## Validation

`tests/test_device_diagnostics.py` uses prescribed fields on small rectangular
meshes. It covers degrees 0, 2, 3, and 6; orthogonal and Bernstein element
bases; different density/potential orders; recovered fluxes; equilibrium
norms and harmonics; scalar exact-error metrics; optional positivity and
polar diagnostics; and the runner's JSONL/CSV recording with residual
summaries. It counts download shapes/bytes and forbids host coefficient
access, including repeated reports and two mesh sizes. Solver vectors and
matrices are guarded against access while residual summaries are recorded.
No PDE solve or time integration is needed.

The suite is included in `gpu-smoke`. Reproduce it with:

```bash
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  tests/test_device_diagnostics.py \
  tests/test_cupy_backend.py::test_guiding_center_field_diagnostics_stays_on_device_and_matches_host \
  tests/test_transport_velocity_diagnostics.py::test_device_diagnostics_match_host_without_materializing_fields
```

On 2026-09-25, all 19 GPU checks above passed in 1.89 seconds with CuPy 14.2.0
and one CUDA device. CuPy runtime JIT was explicitly authorized under the
workspace compilation rule; CPU Numba JIT remained disabled. The checks
verified numerical parity and compact transfer counts using prescribed
fields, including repeated runner records.

On 2026-09-25, 17 focused host/modal/velocity and test-manifest checks passed
with Numba JIT disabled; all changed documentation links resolve. The broader
host diagnostic selection passed 16 checks and exposed one existing
vector-error tolerance failure: an error of `3.55e-15` against `atol=3e-15`
for a constant-vector sampled maximum. That vector calculation and its
helpers are unchanged from `HEAD`; it is separate from these scalar/device
reductions.

Full guiding-center stage residency, time integration, performance sweeps,
multiple devices, and the imported DOLFINx equilibrium remain separate
[TODO items](../../TODO.md#device-residency-and-diagnostics). This diagnostic
contract does not qualify those workflows.
