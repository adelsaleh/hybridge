# dgfem

`dgfem` is a self-contained discontinuous Galerkin / HDG package under active
development inside this repository.  It contains its own mesh, reference
element, DG space, DG field, transfer, plotting, global-system, and HDG
assembly modules.

The main executable solver is:

```bash
python -m dgfem.adv_rea
```

Direct script execution also works:

```bash
python dgfem/adv_rea.py
```

## Quick Start

Install dependencies from a fresh clone:

```bash
python3 scripts/install_dependencies.py
```

or run pip directly:

```bash
python3 -m pip install -r requirements.txt
```

Run the manufactured advection-reaction test on a Gmsh rectangle:

```bash
python -m dgfem.adv_rea -p 6 --lc 0.03 --verbosity 2
```

Plot the numerical solution, exact solution, and absolute error:

```bash
python -m dgfem.adv_rea -p 6 --lc 0.03 --plot
```

Use the projected-reaction path, which projects callable reaction data into
`V_h` and assembles the reaction mass from cached reference triple products:

```bash
python -m dgfem.adv_rea -p 6 --lc 0.03 --project-reaction
```

Use a sparse direct trace solve instead of the default ILU-preconditioned
`BICGSTAB` path:

```bash
python -m dgfem.adv_rea -p 4 --lc 0.08 --solver direct
```

## Solver Defaults

`dgfem.adv_rea` solves the manufactured legacy `test2` problem:

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

- `adv_rea.py`: compact HDG advection-reaction solver and CLI.
- `global_system.py`: sparse global trace-system assembly and solve helpers.
- `hdg_assembly.py`: reusable HDG static-condensation and trace assembly helpers.
- `hdg_mats.py`: vectorized local and trace matrix assembly helpers.
- `mesh.py`: triangular mesh data, Gmsh mesh generators, and connectivity.
- `quadrature.py`: reference triangle quadrature, basis values, and cached reference tensors.
- `space.py`: `DGSpace`, `DGField`, `VectorDGSpace`, and `VectorDGField`.
- `transfer.py`: field transfer/projection helpers between DG spaces.
- `plot.py`: generic DG field plotting helpers plus numerical/exact/error comparison plots.

## Generic Field Plotting

```python
from dgfem.plot import plot_field, plot_fields

plot_field(result.field, resolution=20)
plot_fields((u_h, v_h), titles=("u_h", "v_h"), share_clim=True)
```

See [MANUAL.md](MANUAL.md) for detailed CLI, API, and performance notes.
