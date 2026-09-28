# Advection-Diffusion-Reaction Methods

This topic documents the stationary conservative HDG formulation

```text
div(beta*u + q) + r*u = f,       q = -kappa*grad(u),
```

with a Dirichlet trace on the complete boundary. It covers the combined
advective-diffusive numerical flux, element-local elimination, incidence-wise
transmission assembly, and the two degree-`p+1` postprocessors.

## Derivation

- [`assembly.tex`](assembly.tex) fixes signs and normal orientation, derives the
  local mixed block and condensed trace equation, and records the stabilization
  and postprocessing choices implemented by the NumPy, fused-Numba, and raw-CUDA
  paths.

## Implementation Anchors

Fused Numba assembly and reconstruction support variable scalar and elliptic
tensor diffusion with exact per-element specialization. See the
[Numba ADR guide](../../backends/numba_adr.md) for coefficient formats,
normal-diffusivity stabilization, validation and postprocessing limits.

- `hdgfem.assembly.advection_diffusion_reaction`
- `hdgfem.kernels.advection_diffusion_reaction_fused`
- `hdgfem.backends.advection_diffusion_reaction_numba`
- `hdgfem.backends.advection_diffusion_reaction_raw_cuda`
- `hdgfem.solvers.advection_diffusion_reaction`

Supported backend combinations remain defined by
[`../../reference/backend_capabilities.md`](../../reference/backend_capabilities.md).
The [stationary ADR case catalogue and runners](../../../scripts/advection_diffusion_reaction/README.md)
collect the baseline, disk, oscillatory, stress and tensor problems with a common
`cases/` / `presets.py` / `run_cases.py` interface. Use
`python -m scripts.advection_diffusion_reaction.run_cases --list-presets` from the
repository root to inspect the available configurations.
The initial implementation and validation record is in the
[`2026-08 ADR report`](../../research/solver_studies/stationary_adr_hdg_2026_08.md).

CuPy implements both constrained full-space and RT total-flux recovery and the
coupled primal Neumann recovery. Raw-CUDA results can retain all reconstructed
and postprocessed solution arrays on device. See the
[device recovery contract](../../backends/adr_device_postprocessing.md) for options,
transfer semantics, parity evidence and limits.
