# Torsion-Initialized Equilibrium Research

This collection contains application-specific derivations for the
torsion-initialized semilinear equilibrium work. It is research material and
does not define the public HYBRIDGE solver API.

- [`residual_accounting.tex`](residual_accounting.tex) explains the mixed HDG
  residual blocks, reported norms, and remeshing plateau.
- [`closed_loop_window_optimization.tex`](closed_loop_window_optimization.tex)
  formulates the closed-loop state-constrained threshold problem.
- [`reduced_space_window_optimization.tex`](reduced_space_window_optimization.tex)
  derives reduced sensitivities, constrained parameter directions, and the
  DOLFINx implementation strategy.
- [`h1_projection_optimizer.md`](h1_projection_optimizer.md) documents the
  fixed-mesh H0-1 metric-projection driver, its strict-Newton contract, CLI,
  diagnostics, and reproducible example.

The associated implementation lives under `projects/diocotron/hdg/` and
`projects/diocotron/dolfinx/`. Generated PDFs are not tracked.
