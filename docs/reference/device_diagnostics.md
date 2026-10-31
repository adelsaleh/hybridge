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
norms, and three density harmonic amplitudes. Applications derive energy,
enstrophy, and conservation drifts from these scalar values. Its velocity
diagnostics also reduce boundary-normal flux, interior normal jumps,
divergence, and speed estimates. Growth fits and active-mode rankings consume
the recorded scalar amplitudes rather than full field histories.

A caller requesting both helpers downloads two vectors per record: the field diagnostics and
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
`include_samples=True`. Use `False` when only the reduced metrics are needed. The shared
scalar-packing helper rejects unreduced arrays before a download. Plotting,
screenshots, failure snapshots, and explicitly requested host fields have
their own transfer policies.

## Validation

`tests/test_device_diagnostics.py` exercises the public reductions with
prescribed fields on small meshes; it covers host/device parity and compact
transfer counts without PDE time integration. The corresponding checks in
`tests/test_cupy_backend.py` and `tests/test_transport_velocity_diagnostics.py`
cover field, velocity, and error diagnostics. GPU checks require the optional
CUDA environment.

Historical runner-recording validation, its exact test commands, and the dated
results remain in the research diagnostic report (`../../../guiding-center-cg-hdg/docs/reference/device_diagnostics.md`).
Application stage residency, integration, and campaign performance are assessed
in that repository independently of these public reduction APIs.
