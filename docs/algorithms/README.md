# Numerical Algorithms

This directory contains maintained numerical formulations and derivations.
It does not contain backend support contracts, dated benchmark conclusions,
release checklists, or application-specific research notes.

## Topics

| Topic | Contents |
|---|---|
| [`advection_diffusion_reaction/`](advection_diffusion_reaction/) | Combined conservative flux, static condensation, incidence-wise transmission assembly, and postprocessing. |
| [`advection_reaction/`](advection_reaction/) | Upwind HDG fluxes and block Gauss-Seidel preconditioning. |
| [`upwind_graph_ordering_algorithm/`](upwind_graph_ordering_algorithm/) | Adaptive deterministic upwind graph ordering, SCC residual processing, and trace-DOF permutation. |
| [`diffusion_reaction/`](diffusion_reaction/) | Mixed HDG formulation, static condensation, assembly, and postprocessing. |
| [`quadrature/`](quadrature/) | Symmetric triangle quadrature and exactness requirements. |

Each topic has a Markdown landing page for navigation and one or more TeX
sources for the long-form derivation. Generated PDFs are local build artifacts
and are not tracked.

## Ownership Boundaries

- Supported interfaces and coefficient semantics belong in
  [`../reference/`](../reference/).
- CUDA implementation paths and launch policy belong in
  [`../backends/`](../backends/).
- Qualification plans belong in [`../development/`](../development/).
- Dated timings and solver studies belong in [`../research/`](../research/).

Algorithm notes may describe implementation correspondence, but they do not
expand the supported API or backend matrix. Support claims remain governed by
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md)
and their executable tests.
