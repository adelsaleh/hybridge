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
- [`stationarity_report_2026_08_20.md`](stationarity_report_2026_08_20.md)
  records the 2026-08-20 equilibrium handoff, its Poisson re-solve and a
  preliminary spatial refinement check.

The associated implementation lives under `scripts/torsion_equilibrium/hdg/` and
`scripts/torsion_equilibrium/dolfinx/`. Generated PDFs are not tracked.
