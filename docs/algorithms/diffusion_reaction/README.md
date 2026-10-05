# Diffusion-Reaction Methods

This topic documents the mixed HDG diffusion-reaction formulation, local
elimination, reduced trace assembly, reconstruction, and postprocessing.

## Derivations

- [`assembly.tex`](assembly.tex) derives the continuous and mixed problems,
  LDG-H numerical flux, element matrices, static condensation, boundary
  elimination, and the NumPy and fused-Numba workflows.
- [`postprocessing.tex`](postprocessing.tex) derives the degree-`p+1` primal
  postprocess and both selectable H(div)-oriented flux recoveries:
  `l2_closest` in the full `[P_{p+1}]^2` space and
  `RT_projection` in `[P_p]^2 + x P_p`, with their
  verification invariants.

## Implementation Anchors

- `hybridge.solvers.diffusion_reaction`
- `hybridge.mixed.local_numpy`
- `hybridge.mixed.numba` and `hybridge.mixed.numba_kernels`
- `hybridge.mixed.cupy`
- `hybridge.mixed.raw_cuda.identity`
- `hybridge.mixed.postprocess.flux` and `hybridge.mixed.postprocess.numba_kernels`

Backend restrictions are maintained in
[`../../reference/backend_capabilities.md`](../../reference/backend_capabilities.md),
not in these derivations. Historical measurements are retained in the
[host assembly study](../../research/solver_studies/diffusion_assembly_2026_07.md)
and the
[AMGX modal-trace study](../../research/solver_studies/diffusion_amgx_2026_07.md).
