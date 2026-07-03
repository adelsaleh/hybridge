# hdgfem

`hdgfem` is a self-contained discontinuous Galerkin / HDG package.  The
repository root contains the `hdgfem/` Python package directory, with mesh,
reference-element, space/field, transfer, plotting, sparse global-system, and
HDG assembly code organized into subpackages.

The canonical advection-reaction executable solver is:

```bash
python -m hdgfem.solvers.adv_rea
```

The manufactured diffusion-reaction case runner is:

```bash
python scripts/run_diff_rea_cases.py
```

Preset definitions live in the `PRESETS` dictionary inside
`scripts/run_diff_rea_cases.py`.  List the available presets with:

```bash
python scripts/run_diff_rea_cases.py --list-presets
```

Select a preset by passing its name.  To add a new manufactured PDE case, add
it to `scripts/diff_rea_cases.py`; to add a new run configuration, add an
entry to `PRESETS` in `scripts/run_diff_rea_cases.py`.

The runner keeps numerical parameters in presets.  Command-line flags are
limited to presentation and inspection, for example:

```bash
python scripts/run_diff_rea_cases.py tensor_sine_quick --plot
python scripts/run_diff_rea_cases.py tensor_sine_gamg --print-preset
python scripts/run_diff_rea_cases.py tensor_sine_gamg --dry-run
```

Direct script execution also works:

```bash
python hdgfem/solvers/adv_rea.py
```

## What Is Included

- Object-oriented mesh, quadrature, DG space, scalar field, and vector field
  types.
- Vectorized NumPy HDG/DG matrix assembly helpers.
- Reusable HDG static-condensation and trace-system assembly helpers.
- Sparse direct and Krylov trace solves with optional diagonal scaling and ILU.
- Advection-reaction and diffusion-reaction CLIs.
- Tensor diffusion coefficients for the diffusion-reaction solver.
- Projected source, advection, and reaction coefficient paths.
- Numba-backed projected advection-reaction trace assembly and reconstruction.
- Boundary trace elimination, upwind SCC trace ordering, and matrix-pattern
  diagnostics for advection-reaction runs.

## Quick Start

Run the manufactured advection-reaction test on a Gmsh rectangle:

```bash
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 --verbosity 2
```

Plot the numerical solution, exact solution, and absolute error:

```bash
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 --plot
```

Project CLI callables into DG fields before calling the solver:

```bash
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 --project-source --project-beta --project-reaction
```

Use the fused projected-coefficient Numba assembly backend:

```bash
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 --project-source --project-beta --project-reaction --assembly-backend numba
```

Run tensor diffusion test7 through the main diffusion solver's projected
Numba tensor path with reduced quadrature:

```bash
python scripts/run_diff_rea_cases.py tensor_sine_gamg
```

An experimental hard-coded test7 fused diffusion path is also available for
kernel comparisons:

```bash
python -m hdgfem.solvers.diff_rea_test7_fused \
  --domain structured-rectangle --nx 200 --ny 200 -p 6 \
  --tau 4 --petsc --petsc-preset cg_gamg \
  --volume-quad-1d 7 --edge-quad-1d 7
```

Use the reusable solver class when a driver needs to keep the mesh, space,
latest matrix data, ordering, preconditioner, trace, and reconstructed field:

```python
from hdgfem import AdvectionReactionHDGSolver

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
python -m hdgfem.solvers.adv_rea -p 6 --lc 0.03 --boundary-mode eliminate --trace-ordering upwind-scc
```

Use a sparse direct trace solve instead of the default ILU-preconditioned
`BICGSTAB` path:

```bash
python -m hdgfem.solvers.adv_rea -p 4 --lc 0.08 --solver direct
```

## Solver Defaults

`hdgfem.solvers.adv_rea` solves the manufactured legacy `test2` problem:

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

- `hdgfem/solvers/adv_rea.py`: compact HDG advection-reaction solver and CLI.
- `hdgfem/solvers/diff_rea.py`: HDG diffusion-reaction solver and CLI.
- `hdgfem/backends/numba.py`: package adapter for the projected Numba backend.
- `hdgfem/backends/numpy.py`: NumPy backend exports.
- `hdgfem/backends/cupy.py`: placeholder for a supported CuPy backend.
- `hdgfem/assembly/hdg.py`: reusable HDG static-condensation and trace assembly helpers.
- `hdgfem/assembly/matrices_numpy.py`: vectorized local and trace matrix assembly helpers.
- `hdgfem/assembly/projection.py`: package-native DG projection helpers.
- `hdgfem/io/plot.py`: generic DG field plotting helpers plus numerical/exact/error comparison plots.
- `hdgfem/io/output.py`: console table formatting helpers.
- `hdgfem/linalg/system.py`: sparse global trace-system assembly and solve helpers.
- `hdgfem/linalg/ordering.py`: upwind SCC trace ordering for advection-dominated systems.
- `hdgfem/kernels/`: Numba kernels used by package backends.
- `hdgfem/core/mesh.py`: triangular mesh data, Gmsh mesh generators, and connectivity.
- `hdgfem/core/quadrature.py`: reference triangle quadrature, basis values, and cached reference tensors.
- `hdgfem/core/space.py`: `DGSpace`, `DGField`, `VectorDGSpace`, and `VectorDGField`.
- `hdgfem/core/transfer.py`: field transfer/projection helpers between DG spaces.
- `run_configs/`: version-controlled benchmark and solver presets.
- `scripts/`: runnable project scripts and benchmark sweep entry points.
- `tests/`: focused package tests.

## Local Development

Run local commands from the repository root, or install the project in editable
mode so `hdgfem` is importable from any working directory:

```bash
python -m hdgfem.solvers.adv_rea -p 2 --lc 0.30 --quiet
python -m pip install -e .
```

## Generic Field Plotting

```python
from hdgfem.io.plot import plot_field, plot_fields

plot_field(result.field, resolution=20)
plot_fields((u_h, v_h), titles=("u_h", "v_h"), share_clim=True)
```

See [MANUAL.md](MANUAL.md) for detailed CLI, API, and performance notes.
