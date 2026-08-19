# Active Development Plans

This directory contains forward-looking implementation and qualification plans
that are more detailed than the project roadmap but are not yet supported
algorithm, API, backend, or release contracts.

## Plans

- [`face_block_hp_multigrid.md`](face_block_hp_multigrid.md): direct face-BSR,
  nested Legendre p-coarsening to a scalar face operator, classical h-AMG, and
  a symmetric PCG production path for repeated HDG Poisson solves.
- [`diffusion_stabilization_global_scales.md`](diffusion_stabilization_global_scales.md):
  mesh- and degree-independent global-length and Steklov-calibrated diffusion
  stabilization for diffusion-reaction and ADR.
- [`unrelated_mesh_transfer.md`](unrelated_mesh_transfer.md): conservative
  host/device DG and HDG-trace transfer between unrelated meshes, including
  geometric mismatch handling, validation, and implementation phases.
- [`unsteady_solver_validation.md`](unsteady_solver_validation.md):
  manufactured transient problems and acceptance priorities for reusable
  diffusion-reaction and advection-reaction solver classes.

## Ownership And Lifecycle

- [`../../../TODO.md`](../../../TODO.md) is the canonical prioritized roadmap.
  Plan documents provide design and acceptance detail; they do not duplicate
  task status. Every TODO checklist item covered by a detailed plan must link
  that plan directly.
- [`../alpha_test_matrix.md`](../alpha_test_matrix.md) is an executable
  qualification contract, not an implementation plan, and remains directly
  under `docs/development/`.
- Once a plan is implemented, promote its stable numerical formulation to
  [`../../algorithms/`](../../algorithms/), supported behavior to
  [`../../reference/`](../../reference/), and backend architecture to
  [`../../backends/`](../../backends/).
- Move dated measurements and conclusions to
  [`../../research/`](../../research/), and release-specific evidence to
  [`../../releases/`](../../releases/).
- Remove or archive superseded planning text after its durable content has
  moved to the owning documentation areas; do not treat an old plan as a
