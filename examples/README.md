# Minimal Examples

For the optional FEniCSx fixed-threshold equilibrium solver, see
[`equiband/`](../projects/diocotron/examples/equiband/README.md). Those examples require the separate
FEniCSx/PETSc environment described in their guide.

These examples use only the base NumPy/SciPy installation and the public
package-root API. They are kept small enough to serve as documentation smoke
tests, not as performance benchmarks.

- `advection_reaction_minimal.py`: manufacture and solve
  `beta . grad(u) + r u = f` with an eliminated-boundary HDG trace system.
- `diffusion_reaction_minimal.py`: manufacture and solve
  `-div(kappa grad(u)) + r u = f` with an eliminated-boundary HDG trace system.

Run them from the repository root:

```bash
python examples/advection_reaction_minimal.py
python examples/diffusion_reaction_minimal.py
```

The same source is explained in the "Minimal End-to-End Examples" section of
`MANUAL.md`. Application code should follow these package-root imports, or use
the full-name `hdgfem.solvers.advection_reaction` and
`hdgfem.solvers.diffusion_reaction` facades. The abbreviated
`hdgfem.solvers.adv_rea` and `hdgfem.solvers.diff_rea` module names remain
compatibility paths rather than the preferred public spelling.
