# Diffusion-Reaction Methods

This topic documents the mixed HDG diffusion-reaction formulation, local
elimination, reduced trace assembly, reconstruction, and postprocessing.

## Derivations

- [`assembly.tex`](assembly.tex) derives the continuous and mixed problems,
  LDG-H numerical flux, element matrices, static condensation, boundary
  elimination, and the NumPy and fused-Numba workflows.
- [`postprocessing.tex`](postprocessing.tex) derives the degree-`p+1` primal
  postprocess and the constrained H(div)-oriented flux postprocess, with their
  verification invariants.

## Implementation Anchors

- `hdgfem.solvers.diffusion_reaction`
- `hdgfem.kernels.diffusion_reaction_fused`
- `hdgfem.backends.diffusion_cupy`
- `hdgfem.backends.diffusion_raw_cuda`

Backend restrictions are maintained in
[`../../reference/backend_capabilities.md`](../../reference/backend_capabilities.md),
not in these derivations. Historical measurements are retained in the
[host assembly study](../../research/solver_studies/diffusion_assembly_2026_07.md)
and the
[AMGX modal-trace study](../../research/solver_studies/diffusion_amgx_2026_07.md).
