# dgfem

`dgfem` is a self-contained discontinuous Galerkin / HDG package.  The
repository root is the Python package root, with mesh, reference-element,
space/field, transfer, plotting, sparse global-system, and HDG assembly code
organized into subpackages.

The canonical advection-reaction executable solver is:

```bash
python -m dgfem.solvers.adv_rea
```

The canonical diffusion-reaction executable solver is:

```bash
python -m dgfem.solvers.diff_rea
```

Legacy diffusion tests `0`, `2`, `3`, `5`, and `6` are available.  With
`--domain auto`, test `3` uses a disk and test `6` uses the L-shaped
reentrant-corner domain.

Direct script execution also works:

```bash
python solvers/adv_rea.py
```

## What Is Included

- Object-oriented mesh, quadrature, DG space, scalar field, and vector field
  types.
- Vectorized NumPy HDG/DG matrix assembly helpers.
- Reusable HDG static-condensation and trace-system assembly helpers.
- Sparse direct and Krylov trace solves with optional diagonal scaling and ILU.
- Advection-reaction and diffusion-reaction CLIs.
- Projected source, advection, and reaction coefficient paths.
- Numba-backed projected advection-reaction trace assembly and reconstruction.
- Boundary trace elimination, upwind SCC trace ordering, and matrix-pattern
  diagnostics for advection-reaction runs.

## Quick Start

Run the manufactured advection-reaction test on a Gmsh rectangle:

```bash
python -m dgfem.solvers.adv_rea -p 6 --lc 0.03 --verbosity 2
```

Plot the numerical solution, exact solution, and absolute error:

```bash
python -m dgfem.solvers.adv_rea -p 6 --lc 0.03 --plot
```

Project CLI callables into DG fields before calling the solver:

```bash
python -m dgfem.solvers.adv_rea -p 6 --lc 0.03 --project-source --project-beta --project-reaction
```

Use the fused projected-coefficient Numba assembly backend:

```bash
python -m dgfem.solvers.adv_rea -p 6 --lc 0.03 --project-source --project-beta --project-reaction --assembly-backend numba
```

Use the reusable solver class when a driver needs to keep the mesh, space,
latest matrix data, ordering, preconditioner, trace, and reconstructed field:

```python
from dgfem import AdvectionReactionHDGSolver

solver = AdvectionReactionHDGSolver(
    space,
    assembly_backend="numba",
    boundary_mode="eliminate",
    trace_ordering="upwind-scc",
)
solver.set_discrete_problem(source_h, beta_h, reaction_h, boundary)
result = solver.solve()

solver.set_source(next_source_h)
next_result = solver.solve()
```

Eliminate boundary trace unknowns and apply upwind SCC trace ordering:

```bash
python -m dgfem.solvers.adv_rea -p 6 --lc 0.03 --boundary-mode eliminate --trace-ordering upwind-scc
```

Use a sparse direct trace solve instead of the default ILU-preconditioned
`BICGSTAB` path:

```bash
python -m dgfem.solvers.adv_rea -p 4 --lc 0.08 --solver direct
```

## Solver Defaults

`dgfem.solvers.adv_rea` solves the manufactured legacy `test2` problem:

```text
beta = (x, -y)
r    = y**2
f    = y**2
u    = (a cos(m pi x y) + b sin(n pi x y)) exp(y**2 / 2) + 1
```

The default parameters are `m=5`, `n=5`, `a=2`, and `b=0`.

The default global trace solver is:

```text
solver         BICGSTAB
preconditioner ILU
rtol           1e-13
```

The default basis is `dub_orth`, matching the legacy HDG comparisons.

## Important Files

- `solvers/adv_rea.py`: compact HDG advection-reaction solver and CLI.
- `solvers/diff_rea.py`: HDG diffusion-reaction solver and CLI.
- `backends/numba.py`: package adapter for the projected Numba backend.
- `backends/numpy.py`: NumPy backend exports.
- `backends/cupy.py`: placeholder for a supported CuPy backend.
- `assembly/hdg.py`: reusable HDG static-condensation and trace assembly helpers.
- `assembly/matrices_numpy.py`: vectorized local and trace matrix assembly helpers.
- `assembly/projection.py`: package-native DG projection helpers.
- `io/plot.py`: generic DG field plotting helpers plus numerical/exact/error comparison plots.
- `io/output.py`: console table formatting helpers.
- `linalg/system.py`: sparse global trace-system assembly and solve helpers.
- `linalg/ordering.py`: upwind SCC trace ordering for advection-dominated systems.
- `kernels/`: Numba kernels used by package backends.
- `core/mesh.py`: triangular mesh data, Gmsh mesh generators, and connectivity.
- `core/quadrature.py`: reference triangle quadrature, basis values, and cached reference tensors.
- `core/space.py`: `DGSpace`, `DGField`, `VectorDGSpace`, and `VectorDGField`.
- `core/transfer.py`: field transfer/projection helpers between DG spaces.
- `run_configs/`: version-controlled benchmark and solver presets.
- `scripts/`: runnable project scripts and benchmark sweep entry points.
- `tests/`: focused package tests.

## Local Development

Because the package root is the repository root, run local commands from this
directory with the parent directory on `PYTHONPATH`, or install the project in
editable mode:

```bash
PYTHONPATH=.. python -m dgfem.solvers.adv_rea -p 2 --lc 0.30 --quiet
python -m pip install -e .
```

## Generic Field Plotting

```python
from dgfem.io.plot import plot_field, plot_fields

plot_field(result.field, resolution=20)
plot_fields((u_h, v_h), titles=("u_h", "v_h"), share_clim=True)
```

See [MANUAL.md](MANUAL.md) for detailed CLI, API, and performance notes.
