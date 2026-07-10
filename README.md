# hdgfem

`hdgfem` is a self-contained discontinuous Galerkin / HDG package.  The
repository root contains the `hdgfem/` Python package directory, with mesh,
reference-element, space/field, transfer, adaptivity, plotting, sparse
global-system, and HDG assembly code organized into subpackages.

The manufactured advection-reaction case runner is:

```bash
python scripts/run_adv_rea_cases.py
```

The manufactured diffusion-reaction case runner is:

```bash
python scripts/run_diff_rea_cases.py
```

The Strategy A star-domain HDG Newton benchmark is now a non-adaptive
torsion-initialized run:

```bash
python scripts/diocotron_equilibrium_torsion_intialized.py \
  --star-n 260 --order 4 --hdg-tau 20 -v 2 \
  --residual-norm euclid --newton-shift-mode none
```

The runner creates a timestamped directory under
`run_logs/diocotron_equilibrium_torsion_intialized/`, writes `newton.csv`,
`frames.csv`, and `summary.txt`, and reports the split mixed HDG residual
components.  Mesh adaptivity was removed from this comparison driver; reusable
adaptive remeshing helpers live in `hdgfem.core.adaptivity`.

A DOLFINx continuous-Galerkin fixed-mesh comparison runner is also available
when the `fenics-dolfinx` environment is installed:

```bash
/home/asaleh/miniforge3/envs/fenicsx-dgfem/bin/python \
  scripts/strategyA_dolfinx_noadapt_torsion_newton.py \
  --mesh run_logs/diocotron_equilibrium_torsion_intialized/<run>/initial_mesh.msh \
  --order 2 --linear-solver mumps
```

Passing the saved HDG `initial_mesh.msh` is the preferred fair-comparison path:
both runners then use the same triangle set, while DOLFINx can vary the CG
polynomial order independently.  DOLFINx outputs are written under
`run_logs/dolfinx_torsion_noadapt/`.

Preset definitions live in the `PRESETS` dictionaries inside
`scripts/run_adv_rea_cases.py` and `scripts/run_diff_rea_cases.py`.  List the
available presets with:

```bash
python scripts/run_adv_rea_cases.py --list-presets
python scripts/run_diff_rea_cases.py --list-presets
```

Select a preset by passing its name.  To add a new manufactured PDE case, add
it to `scripts/adv_rea_cases.py` or `scripts/diff_rea_cases.py`; to add a new
run configuration, add an entry to the corresponding runner's `PRESETS`.

The runner keeps numerical parameters in presets.  Command-line flags are
limited to presentation and inspection, for example:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind --plot
python scripts/run_diff_rea_cases.py tensor_sine_gamg --print-preset
python scripts/run_diff_rea_cases.py tensor_sine_gamg --dry-run
```

The diffusion-reaction runner can request HDG post-processing through the
`hdg_postprocess` preset field (`"none"`, `"primal"`, `"flux"`, or `"both"`).
Manufactured diffusion-reaction cases provide the exact conservative flux
`q=-kappa grad u`, so the solve summary reports primal, flux, postprocessed
primal, and postprocessed flux errors when the corresponding fields are
available.  Runner timing tables also show each non-total timing as a percentage
of the total runtime.

## Optional PETSc Backend

PETSc support is optional.  The core package only depends on NumPy, SciPy, and
Numba; PETSc is imported lazily when `--petsc` or `solver="petsc"` is used.

Use a PETSc build with matching `petsc4py`.  For example, after configuring and
building PETSc 3.22.2 in `~/opt/petsc` with `PETSC_ARCH=arch-linux-c-opt`:

```bash
source .venv/bin/activate

python -m pip install --force-reinstall \
  "numpy<2.5,>=2.4" "Cython>=3.0,<3.1" "setuptools<75" "wheel<0.46"

export PETSC_DIR=$HOME/opt/petsc
export PETSC_ARCH=arch-linux-c-opt
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH

cd "$PETSC_DIR/src/binding/petsc4py"
python setup.py clean --all

cd /path/to/hdgfem
python -m pip install --no-build-isolation --no-deps \
  "$PETSC_DIR/src/binding/petsc4py"
```

The `numpy<2.5` pin keeps the current Numba dependency satisfiable.  The Cython
and setuptools pins avoid known build failures with `petsc4py` 3.22.2.  Do not
mix a system `petsc4py` package with a different virtualenv NumPy; that can
produce binary ABI errors at import time.

Verify the real PETSc import and optional solver packages:

```bash
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
python -c "from petsc4py import PETSc; k=PETSc.KSP().create(); k.getPC().setType('gamg'); print('GAMG ok')"
python -c "from petsc4py import PETSc; k=PETSc.KSP().create(); pc=k.getPC(); pc.setType('hypre'); pc.setHYPREType('boomeramg'); print('Hypre/BoomerAMG ok')"
python -c "from petsc4py import PETSc; k=PETSc.KSP().create(); pc=k.getPC(); pc.setType('lu'); pc.setFactorSolverType('mumps'); print('MUMPS ok')"
```

## What Is Included

- Object-oriented mesh, quadrature, DG space, scalar field, and vector field
  types.
- Vectorized NumPy HDG/DG matrix assembly helpers.
- Reusable HDG static-condensation and trace-system assembly helpers.
- HDG Gram assembly and condensed inverse applications for dual residual
  diagnostics.
- Sparse direct and Krylov trace solves with optional diagonal scaling and ILU.
- Advection-reaction and diffusion-reaction preset runners.
- Tensor diffusion coefficients for the diffusion-reaction solver.
- Projected source, advection, and reaction coefficient paths.
- Numba-backed projected advection-reaction trace assembly and reconstruction.
- Smooth and polygonal Gmsh star-domain mesh generators.
- Reusable structured-background Gmsh adaptivity helpers in
  `hdgfem.core.adaptivity`.
- Boundary trace elimination, upwind SCC trace ordering, and matrix-pattern
  diagnostics for advection-reaction runs.

## Quick Start

Run the manufactured advection-reaction test on a Gmsh rectangle:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 6 --lc 0.03 --verbosity 2
```

Plot the numerical solution, exact solution, and absolute error:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 6 --lc 0.03 --plot
```

The advection presets project callable coefficients into DG fields before
calling the Numba assembly backend.

```bash
python scripts/run_adv_rea_cases.py test2_petsc_bicgstab_ilu -p 6 --lc 0.03
```

Compare advection-reaction global solver configurations after one trace
assembly:

```bash
python scripts/benchmark_adv_rea_solvers.py -p 6 --lc 0.01
```

This benchmark reuses the assembled upwind-ordered trace matrix while sweeping
SciPy ILU, PETSc ILU/ASM ILU, and experimental upwind block-Gauss-Seidel
preconditioners.  The block-GS variants are useful diagnostics for flow-aware
preconditioning, but high-fill ILU is currently the practical default for the
large `test2` runs.

Reference timings cited in the manual were collected on host `23G82`, running
Ubuntu 22.04 with Linux 6.8, an Intel Core i5-10210U CPU, 4 physical cores / 8
hardware threads, and 15 GiB RAM.  Report that hardware context with any
performance numbers from this benchmark, because preconditioner setup and
Krylov timings are sensitive to CPU, memory bandwidth, and thread scheduling.

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

Inspect the HDG Gram implementation on a small rectangular space:

```bash
python scripts/hdg_gram_matrix_test.py --order 2 --nx 2 --ny 2
```

The package test `tests/test_hdg_gram.py` checks that the condensed Gram
inverse matches a sparse direct solve and that the dual norm satisfies the
energy identity.

### Strategy A Run Summary

The current Strategy A driver is a fixed-mesh HDG Newton comparison against the
FreeFEM torsion/Newton formalism.  It solves the torsion initializer, builds the
semilinear density window, then runs epsilon continuation without any preadapt
or scheduled remesh step.  This isolates the Newton convergence behavior from
mesh-transfer effects.

```text
script       scripts/diocotron_equilibrium_torsion_intialized.py
output       run_logs/diocotron_equilibrium_torsion_intialized/<run-tag>_<timestamp>/
mesh         native smooth-star Gmsh mesh
residual     selectable: euclid, hdg-local, edp-volume, or hdg
adaptivity   not used by this driver
```

DOLFINx CG comparison runner:

```text
script       scripts/strategyA_dolfinx_noadapt_torsion_newton.py
output       run_logs/dolfinx_torsion_noadapt/<run-tag>_<timestamp>/
mesh         generated smooth-star mesh or saved Gmsh mesh via --mesh
solver       mumps, lu, hypre, or gamg
adaptivity   not used
```

Use `--mesh` with a saved HDG `initial_mesh.msh` to remove Gmsh-version and
mesh-generation differences from CG/HDG timing comparisons.

For cleaner timing comparisons, omit `--plot`.  For residual accounting, keep
`--residual-norm euclid` for the full mixed coefficient residual or use
`--residual-norm hdg` when the local HDG Gram diagnostic is required.

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
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 6 --lc 0.03
```

Use a sparse direct trace solve instead of the default ILU-preconditioned
`BICGSTAB` path:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_direct -p 4 --lc 0.08
```

## Solver Defaults

The default advection runner preset solves the manufactured legacy `test2`
problem:

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

- `hdgfem/solvers/adv_rea.py`: compact HDG advection-reaction solver.
- `hdgfem/solvers/diff_rea.py`: HDG diffusion-reaction solver with optional
  degree `p+1` primal and H(div)-style flux post-processing.
- `hdgfem/backends/numba.py`: package adapter for the projected Numba backend.
- `hdgfem/backends/numpy.py`: NumPy backend exports.
- `hdgfem/backends/cupy.py`: placeholder for a supported CuPy backend.
- `hdgfem/assembly/hdg.py`: reusable HDG static-condensation and trace assembly helpers.
- `hdgfem/assembly/hdg_gram.py`: sparse and statically condensed HDG Gram
  inverse applications for dual residual norms.
- `hdgfem/assembly/matrices_numpy.py`: vectorized local and trace matrix assembly helpers.
- `hdgfem/assembly/projection.py`: package-native DG projection helpers.
- `hdgfem/io/plot.py`: generic DG field plotting helpers, PyVista comparison
  plots, and Matplotlib discontinuous contour panels for small meshes.
- `hdgfem/io/output.py`: console table formatting helpers.
- `hdgfem/linalg/system.py`: sparse global trace-system assembly and solve helpers.
- `hdgfem/linalg/ordering.py`: upwind SCC trace ordering for advection-dominated systems.
- `hdgfem/linalg/upwind_block_gs.py`: experimental level-scheduled upwind block-GS preconditioners.
- `hdgfem/kernels/`: Numba kernels used by package backends.
- `hdgfem/core/mesh.py`: triangular mesh data, Gmsh mesh generators, and connectivity.
- `hdgfem/core/quadrature.py`: reference triangle quadrature, basis values, and cached reference tensors.
- `hdgfem/core/space.py`: `DGSpace`, `DGField`, `VectorDGSpace`, and `VectorDGField`.
- `hdgfem/core/transfer.py`: field transfer/projection helpers between DG spaces.
- `hdgfem/core/adaptivity.py`: PDE-agnostic DG indicators and structured
  Gmsh background-field remeshing helpers.
- `run_configs/`: version-controlled benchmark and solver presets.
- `scripts/`: runnable project scripts and benchmark sweep entry points,
  including `diocotron_equilibrium_torsion_intialized.py` and
  `hdg_gram_matrix_test.py`.
- `tests/`: focused package tests.

## Local Development

Run local commands from the repository root, or install the project in editable
mode so `hdgfem` is importable from any working directory:

```bash
python scripts/run_adv_rea_cases.py test2_scipy_ilu_upwind -p 2 --lc 0.30 --quiet
python -m pip install -e .
```

## Generic Field Plotting

```python
from hdgfem.io.plot import plot_field, plot_fields

plot_field(result.field, resolution=20)
plot_fields((u_h, v_h), titles=("u_h", "v_h"), share_clim=True)
```

For small discontinuous contour panels with duplicated per-element vertices:

```python
from hdgfem.io.plot import plot_scalar_sample_panels_matplotlib, reference_plot_points

ref = reference_plot_points(16)
values = result.field.values_at_ref(ref)
plot_scalar_sample_panels_matplotlib(
    result.field.space.mesh,
    [("u_h", ref, values)],
    cmap="jet",
)
```

See [MANUAL.md](MANUAL.md) for detailed CLI, API, and performance notes.
