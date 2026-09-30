# Advection-Reaction Methods

This topic covers the mathematical upwind HDG trace system and the graph-aware
tools used to order and precondition it.

## Numerical Flux Contract

For an interior face shared by sides `l` and `r`, retain both side
contributions. With `b_s = beta_s . n_s` and stabilization `tau_s`, the trace
mass coefficient is

```text
((tau_l - b_l) + (tau_r - b_r)) * hat(u).
```

The local coupling and right-hand-side terms must likewise use each side's
coefficient; averaging a discontinuous advection field across the face changes
the discrete operator.

On a no-through-flow boundary, `beta . n = 0` and upwind stabilization gives
`tau = 0`. The boundary numerical flux is then zero, so that boundary does not
require trace unknowns. Retaining boundary trace degrees of freedom while
removing their flux equation produces an artificial singular block.

The public boundary-mode, stabilization-input, active-DOF, ordering, and
reconstruction semantics are defined in
[`../../reference/advection_boundary_stabilization.md`](../../reference/advection_boundary_stabilization.md).

Implementation parity for discontinuous advection is recorded in
[`../../backends/raw_cuda.md`](../../backends/raw_cuda.md) and checked by the
advection conservation and CuPy/raw-CUDA parity tests.

## Upwind Graph Tools

- [Upwind Graph Ordering Algorithm](../upwind_graph_ordering_algorithm/)
  derives the directed edge graph, acyclic fast path, residual strongly
  connected components, and trace-DOF permutation.
- [`upwind_block_gauss_seidel.tex`](upwind_block_gauss_seidel.tex) derives the
  ordered block-GS preconditioner and its host/device representations.

The SCC decomposition is an ordering mechanism. It can improve ILU fill and
defines the dependency structure used by upwind block-GS, but it is not itself
a global sparse solver. A future dependency-driven SCC solve is a separate
research algorithm and must not be inferred from the current ordering API.

## Implementation Anchors

- `hdgfem.linalg.ordering`
- `hdgfem.linalg.upwind_block_gs`
- `hdgfem.linalg.upwind_block_gs_on_the_fly`
- `hdgfem.linalg.upwind_block_gs_cupy`

Dated Krylov, ILU, and upwind-GS comparisons are retained in
[`../../research/solver_studies/advection_reaction_2026_07.md`](../../research/solver_studies/advection_reaction_2026_07.md).
