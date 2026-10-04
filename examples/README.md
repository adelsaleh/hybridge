# Examples

Small complete programs demonstrate the package API. Run them from the
repository root after installing `hdgfem`.

## Minimal host solves

These two examples use the base installation and are covered by the
documentation smoke tests:

- [`advection_reaction_minimal.py`](advection_reaction_minimal.py): manufacture and solve
  `beta . grad(u) + r u = f` with an eliminated-boundary HDG trace system.
- [`diffusion_reaction_minimal.py`](diffusion_reaction_minimal.py): manufacture and solve
  `-div(kappa grad(u)) + r u = f` with an eliminated-boundary HDG trace system.

```bash
python examples/advection_reaction_minimal.py
python examples/diffusion_reaction_minimal.py
```

The same source is explained in the
[manual](../MANUAL.md#minimal-end-to-end-examples). Application code should
follow these package-root imports, or use
the full-name `hdgfem.solvers.advection_reaction` and
`hdgfem.solvers.diffusion_reaction` facades. The abbreviated
`hdgfem.solvers.adv_rea` and `hdgfem.solvers.diff_rea` module names remain
compatibility paths rather than the preferred public spelling.

## GPU vortex gas

[`gpu_vortex_gas.py`](gpu_vortex_gas.py) couples reusable Poisson and transport
solvers to evolve a two-species guiding-center plasma, the electrostatic form
of a two-dimensional vortex gas, in a five-lobed star around a grounded
circular island. It shows mesh generation, an initial field, the BDF2 time
loop, and paired Holoviz charge-density/potential panels in one commented script.

```bash
python examples/gpu_vortex_gas.py
```

This example requires Gmsh and the CuPy, AMGX/PyAMGX, and Holoviz GPU runtimes.
The [reproduction guide](../docs/getting_started/gpu_showcase.md) provides setup,
numerical checks, and the recording parameters used for both README videos.
