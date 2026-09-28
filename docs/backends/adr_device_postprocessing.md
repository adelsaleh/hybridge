# ADR device postprocessing

Stationary ADR supports `postprocessing_backend="numba"|"cupy"|"auto"` for
both degree-`p+1` recovery stages. `auto` selects CuPy with raw-CUDA assembly and
reconstruction, and Numba with host assembly/reconstruction. CuPy implements
both `flux_postprocess_space="l2_closest"` and `"RT_projection"`, followed by
the same coupled local Neumann primal equations as the Numba reference.
Primal recovery still requires positive constant scalar diffusion.

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
assembly preparation remains host-based: this contract does not promise a
completely host-free solve from arbitrary device-only PDE input fields.

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
