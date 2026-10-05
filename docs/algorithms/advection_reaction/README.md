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

- [`upwind_scc_ordering.tex`](upwind_scc_ordering.tex) derives the directed
  edge graph, strongly connected components, condensation DAG, and trace-DOF
  permutation.
- [`upwind_block_gauss_seidel.tex`](upwind_block_gauss_seidel.tex) derives the
  ordered block-GS preconditioner and its host/device representations.

The SCC decomposition is an ordering mechanism. It can improve ILU fill and
defines the dependency structure used by upwind block-GS, but it is not itself
a global sparse solver. A future dependency-driven SCC solve is a separate
research algorithm and must not be inferred from the current ordering API.

## Guiding-center time integration

- [H1-BDF3](h1_bdf3.md): AB3 prediction with one implicit BDF3 transport solve,
  SI-Euler extrapolation startup, accepted-field HDG residuals and shared device caches.
- [H2-BDF3](h2_bdf3.md): extrapolated-drift BDF3 prediction and a BDF3 correction,
  with two transport and two Poisson solves per regular step.

- [IMEX-ARK3](imex_ark3.md): three transport solves sharing a frozen operator,
  four Poisson evaluations, and an embedded second-order error estimate.

## Implementation Anchors

- `hybridge.linalg.ordering`
- `hybridge.linalg.upwind_block_gs`
- `hybridge.linalg.upwind_block_gs_on_the_fly`
- `hybridge.linalg.gpu.upwind_block_gs`

Dated Krylov, ILU, and upwind-GS comparisons are retained in
[`../../research/solver_studies/advection_reaction_2026_07.md`](../../research/solver_studies/advection_reaction_2026_07.md).
