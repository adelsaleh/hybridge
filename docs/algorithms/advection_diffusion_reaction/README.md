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

- `hdgfem.assembly.advection_diffusion_reaction`
- `hdgfem.kernels.advection_diffusion_reaction_fused`
- `hdgfem.backends.advection_diffusion_reaction_numba`
- `hdgfem.backends.advection_diffusion_reaction_raw_cuda`
- `hdgfem.solvers.advection_diffusion_reaction`

Supported backend combinations remain defined by
[`../../reference/backend_capabilities.md`](../../reference/backend_capabilities.md).
The initial implementation and validation record is in the
[`2026-08 ADR report`](../../research/solver_studies/stationary_adr_hdg_2026_08.md).
