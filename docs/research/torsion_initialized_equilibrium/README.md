# Torsion-Initialized Equilibrium Research

This collection contains application-specific derivations for the
torsion-initialized semilinear equilibrium work. It is research material and
does not define the public HDGFEM solver API.

- [`residual_accounting.tex`](residual_accounting.tex) explains the mixed HDG
  residual blocks, reported norms, and remeshing plateau.
- [`closed_loop_window_optimization.tex`](closed_loop_window_optimization.tex)
  formulates the closed-loop state-constrained threshold problem.
- [`reduced_space_window_optimization.tex`](reduced_space_window_optimization.tex)
  derives reduced sensitivities, constrained parameter directions, and the
  DOLFINx implementation strategy.

The associated implementation lives under `scripts/diocotron_hdg/` and
`scripts/diocotron_dolfinx/`. Generated PDFs are not tracked.
