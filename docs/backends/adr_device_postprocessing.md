# ADR device postprocessing

Stationary ADR supports `postprocessing_backend="numba"|"cupy"|"auto"` for
both degree-`p+1` recovery stages. `auto` selects CuPy with raw-CUDA assembly and
reconstruction, and Numba with host assembly/reconstruction. CuPy implements
both `flux_postprocess_space="l2_closest"` and `"RT_projection"`, followed by
the same coupled local Neumann primal equations as the Numba reference.
Primal recovery supports positive variable scalar and elliptic tensor diffusion,
including nonsymmetric tensors with positive definite symmetric part. It samples
and validates K at recovery quadrature, then assembles all four K^-1-weighted
constitutive blocks. It retains the total numerical flux and element mean.
For quadrature-only coefficients, samples must match the recovery quadrature;
use callable or DG components when assembly and recovery points differ.

CuPy primal recovery uses `hdgfem.mixed.postprocess.primal_raw_cuda` to assemble its
matrix and RHS in one launch, tiling matrix entries across blocks and sharing
quadrature contractions across the nine mixed volume blocks. CuPy/cuBLAS solves
the batched pivoted systems. The independent contraction-based implementation
remains a test/benchmark reference. Geometry and reference tables are cached;
variable PDE coefficients are resampled each call, never cached as factors.

## Residency and materialization

For raw-CUDA solves, set `materialize_host_solution=False` to retain the full
trace and mixed local unknowns as CuPy arrays. The raw primal, diffusive flux,
projected total flux, and requested postprocessed fields are device-backed
`DGField`/`VectorDGField` objects. Their `.coeffs` property explicitly downloads
host coefficients; existing device coefficient helpers reuse the device arrays.
The default `True` eagerly materializes returned host arrays and field coefficients.

The CuPy recovery stages sample velocity and stabilization, gather oriented
traces, construct flux moments, and solve both recoveries without downloading
solution arrays. Device-backed velocity and stabilization fields are sampled in
their own spaces. Reference quadrature/basis and mesh geometry may originate on
host and be uploaded. Scalar finiteness checks and global-solver diagnostics can
synchronize; these are not full-field downloads. Existing ADR coefficient and
assembly preparation uses device sampling for supported inputs; incompatible
callables can use the documented host fallback. Tensor DG components and retained
incidence stabilization tables are not downloaded for recovery.

Explicit `postprocessing_backend="numba"` with requested recovery and
`materialize_host_solution=False` on raw CUDA fails before assembly. Set the
backend to `auto` or `cupy`, or explicitly permit host materialization. All four
`hdg_postprocess="none"|"primal"|"flux"|"both"` modes retain their meanings;
primal-only recovery computes its intermediate total flux on device as well.

CuPy uses batched dense local systems. This change establishes residency and
correctness, not a performance recommendation or a large-mesh memory bound.
Large-mesh allocation tuning and wider order/geometry qualification remain
separate work; no new AMGX build is needed.

## Reproducible evidence

Run from the repository with the installed fork stack and CUDA runtime:

```bash
OMP_NUM_THREADS=16 .venv/bin/python -m pytest -q tests/test_adr_device_postprocessing.py tests/test_advection_diffusion_reaction.py
```

On 2026-09-28, the dedicated device suite passed 54 checks without skips:

- Both trace bases (`legacy-lagrange`, `legendre-modal`) and both flux variants;
  direct recovery parity at p=0,1,3,6 on four affine triangles with variable
  cross-space velocity, unequal side stabilization, arbitrary mixed/trace
  coefficients, and an independent element-mean check.
- Recovery forbids `cupy.asnumpy` and lazy field downloads while using device-only
  velocity and solution inputs. Returned recovered fields have no host coefficients.
- Six additional sampling checks cover upwind, scalar, callable, device field,
  device coefficient-table and device quadrature-table stabilization inputs.
- Native raw-CUDA/AMGX stationary solves at p=1 cover both materialization settings,
  all four postprocessing modes, both trace bases and both flux variants, with
  nonzero boundary values. Disabled-materialization runs allow only scalar
  `cupy.asnumpy` downloads; all returned fields, full trace and local unknowns
  match the host reference after explicit materialization.

The dedicated suite is included in `gpu-smoke`. These checks execute stationary
small systems, with CuPy/CUDA runtime JIT enabled; they run no time integration
and do not compile AMGX.

The final focused run passed 272 checks in 5.22 seconds, with no skips:

```bash
OMP_NUM_THREADS=16 .venv/bin/python -m pytest -q \
  tests/test_adr_device_postprocessing.py \
  tests/test_advection_diffusion_reaction.py \
  tests/test_advection_diffusion_reaction_manufactured_disk.py \
  tests/test_backend_capabilities.py tests/test_alpha_test_matrix.py \
  tests/test_diffusion_reaction_solver.py::test_diffusion_rt_cupy_matches_numba \
  tests/test_diffusion_reaction_solver.py::test_diffusion_rt_projection_satisfies_unisolvent_moments \
  tests/test_diagnostics.py::test_solution_trace_prefers_device_and_reduces_host_trace
```

A wider related run had 250 passes and 15
failures outside this change: diffusion reusable option forwarding
(`fb_hp_mg_true_residual_every`), the `ScaledUpwind` export expectation, and
existing documentation artifacts, links and missing docstrings. The option
failure also reproduces with the pre-change recovery helpers. No full-suite
pass is claimed.


## Tensor qualification (2026-09-29)

The tensor extension is checked by:

- 128 direct recovery parity/residency cases: seven diffusion structures plus
  cross-space device DG tensor components, p=0,1,3,6, both trace bases and both
  total-flux variants. Downloads are forbidden during recovery and every
  recovered element mean is checked independently.
- Four full matrix/RHS comparisons of fused primal assembly against independent
  CuPy contractions, including nonsymmetric off-diagonal tensor blocks.
- Native raw-CUDA scalar and variable-full-tensor integration checks for all
  modes, both trace bases, both flux variants and both materialization settings.
- Continuous sine manufactured convergence on 2x2, 4x4 and 8x8 meshes, p=1,2:
  variable scalar, diagonal, symmetric and nonsymmetric tensors on the host;
  native raw-CUDA variable-full tensor recovery with both flux variants.
  These are bounded stationary checks, without time integration.

Reproduce the qualification with:

```bash
OMP_NUM_THREADS=16 .venv/bin/python -m pytest -q \
  tests/test_adr_device_postprocessing.py tests/test_adr_tensor_numba.py \
  tests/test_adr_tensor_solver_cuda.py tests/test_backend_capabilities.py
```

Run the local performance/parity diagnostic (no global solve) with an idle GPU:

```bash
OMP_NUM_THREADS=16 .venv/bin/python -m \
  scripts.advection_diffusion_reaction.diagnostics.benchmark_tensor_postprocessing \
  --nx 16 --orders 1 3 4 6 --repeats 5
```

The benchmark warms both paths and includes tensor sampling, allocations,
assembly, and batched solution in synchronized wall timings. It excludes
reference-space creation and reports the individual samples as JSON. The
reference and production outputs must agree before timings are reported.


The final focused run passed **492 checks without skips** (73.95 seconds):

```bash
OMP_NUM_THREADS=16 .venv/bin/python -m pytest -q \
  tests/test_adr_device_postprocessing.py tests/test_adr_tensor_numba.py \
  tests/test_adr_tensor_solver_cuda.py tests/test_advection_diffusion_reaction.py \
  tests/test_backend_capabilities.py
```

An additional stabilization/input run passed **23 checks**:

```bash
OMP_NUM_THREADS=16 .venv/bin/python -m pytest -q \
  tests/test_adr_face_stabilization.py tests/test_adr_tensor_numba.py \
  -k 'stabilization or mixed_element or invalid_diffusion'
```

These checks used the installed CUDA/AMGX runtime; no AMGX rebuild or time
integration was performed. The recovery qualification is FP64 on affine
triangles; it is not a blanket performance or accuracy guarantee for all meshes,
coefficients, precisions, or large-mesh memory sizes.

### Measured primal recovery performance

On an NVIDIA RTX PRO 5000 Blackwell, 512 affine elements, FP64, five warmed
measurements per path, the final implementation measured:

| p | CuPy contraction reference (ms) | Fused production (ms) | Speedup |
|---|---:|---:|---:|
| 1 | 25.740 | 16.036 | 1.61x |
| 3 | 25.745 | 16.958 | 1.52x |
| 4 | 29.728 | 20.169 | 1.47x |
| 6 | 63.967 | 29.817 | 2.15x |

These final timings were collected with another live raw-CUDA test process on
the same GPU (100% reported utilization, P1, 180 MHz SM clock at measurement
start). They establish a matched improvement under that load, not uncontended
latency or peak throughput. Rerun the diagnostic on an idle GPU before using
these absolute times for capacity planning. The comparison includes tensor
sampling and the batched solve; it measures primal recovery, not the entire
ADR solve or the preceding total-flux recovery.
