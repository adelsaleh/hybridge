# hdgfem

`hdgfem` is a discontinuous Galerkin / HDG research codebase.  The Python
package lives in `hdgfem/` and contains mesh, quadrature, DG space/field,
assembly, linear algebra, solver, plotting, and Numba backend modules.

## Scripts

Runnable experiments and test drivers live under `scripts/`, grouped by topic:

- `scripts/advection_reaction/`
- `scripts/diffusion_reaction/`
- `scripts/diffusion_reaction/experimental/`
- `scripts/diocotron_hdg/`
- `scripts/diocotron_dolfinx/` (optional DOLFINx comparison diagnostics)
- `scripts/hdg_gram/`
- `scripts/dev/`

These are test and research scripts, not package APIs.  Check each script's
module docstring and `--help` output for its assumptions, parameters, and output
paths.

Common preset runners:

```bash
python -m scripts.advection_reaction.run_adv_rea_cases --list-presets
python -m scripts.diffusion_reaction.run_diff_rea_cases --list-presets
```

## Package Contents

- `hdgfem/core/`: meshes, quadrature, DG spaces, fields, transfer, adaptivity.
- `hdgfem/assembly/`: HDG matrix, trace, projection, and Gram assembly helpers.
- `hdgfem/backends/`: NumPy and Numba backend adapters.
- `hdgfem/kernels/`: Numba kernels.
- `hdgfem/solvers/`: package-level advection-reaction and diffusion-reaction solvers.
- `hdgfem/linalg/`: sparse trace-system assembly, solves, ordering, and preconditioners.
- `hdgfem/io/`: plotting and console-output helpers.
- `tests/`: focused regression tests.
- `run_configs/`: version-controlled benchmark and solver presets.

## Local Development

Run commands from the repository root, or install the project in editable mode:

```bash
source .venv/bin/activate
python -m pip install -e .
python -m pytest
```

## Optional Runtime Stacks

The base package uses NumPy, SciPy, and Numba.  PETSc, CuPy, PyAMGX, and
DOLFINx are optional runtime stacks used only by specific scripts or solver
options.  They are imported lazily, so they do not need to be present for the
core CPU HDG tests.  DOLFINx has the lowest priority here: it is used only by
later comparison diagnostics under `scripts/diocotron_dolfinx/`, not by the
main `hdgfem` package.

Numba is a normal Python dependency.  Tune CPU parallelism before launching
Python when needed:

```bash
export NUMBA_NUM_THREADS=40
export OMP_NUM_THREADS=1
```

PETSc solves require a matched PETSc / `petsc4py` install visible to the active
Python environment:

```bash
export PETSC_DIR=$HOME/opt/petsc
export PETSC_ARCH=arch-linux-c-opt
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
```

GPU scripts under `scripts/gpu/` require a CUDA-compatible CuPy install.  AMGX
solves additionally require `pyamgx` and the AMGX shared libraries on the
dynamic loader path.  Replace `/path/to/amgx/lib` with the directory that
contains your AMGX shared library, for example `libamgxsh.so`; omit the export
if AMGX is already visible through your environment, `ldconfig`, or rpath:

```bash
export AMGX_LIB_DIR=/path/to/amgx/lib
export LD_LIBRARY_PATH=$AMGX_LIB_DIR:$LD_LIBRARY_PATH
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
python -c "import pyamgx; print('pyamgx ok')"
python -m scripts.gpu.run_adv_rea_gpu4_hdg --help
```

The GPU4 advection-reaction runner is backed by the reusable package solver in
`hdgfem.solvers.adv_rea`. Its Raw CUDA path keeps the reduced trace assembly,
AMGX solve, trace reconstruction, field reconstruction, and error evaluation on
device unless host materialization is explicitly requested. Fused Raw CUDA can
emit a reduced CSR matrix directly, avoiding the older CuPy COO-to-CSR
construction when `--raw-matrix-format csr` is selected. Use
`configs/amgx/README.md` for the current AMGX presets and sweep commands.

DOLFINx is optional and only needed for comparison diagnostics.  Prefer a
separate conda environment so its MPI/PETSc stack does not constrain the normal
HDG environment:

```bash
conda create -n fenicsx-dgfem -c conda-forge fenics-dolfinx mpich pyvista gmsh
conda activate fenicsx-dgfem
python -c "import dolfinx, basix, ufl, mpi4py, petsc4py, gmsh; print('dolfinx ok')"
python -m pip install -e .
```

No project-local build step links these libraries into `hdgfem`.  Use the
correct Python environment and make native shared libraries visible through
`LD_LIBRARY_PATH` before starting Python.  See `MANUAL.md`,
`docs/gpu_hdg_modules.md`, and `configs/amgx/README.md` for longer notes.

For advection-reaction HDGFEM, the fused Raw CUDA path is the preferred high
performance path.  Default behavior is:

- `--raw-lu-mode safe` (stable baseline),
- optional `--raw-lu-mode coop` for the cooperative LU stage,
- supported for `--raw-local-assembly fused` and trace basis
  `legacy-lagrange`/`legendre-modal`,
- tested through `p <= 8` on the current raw CUDA constraints.

Smoke-run checks:

```bash
python -m scripts.advection_reaction.run_adv_rea_cases --print-preset --dry-run
python -m scripts.diffusion_reaction.run_diff_rea_cases --print-preset --dry-run
```

Mesh generation now defaults to local caching and logs cache hit/miss events
(`.cache/hdgfem/meshes`).  Use `--gmsh-num-threads` in the GPU advection runner
to enable parallel CPU meshing.

## Notes

- `docs/gpu_hdg_modules.md` contains standalone GPU runner status and benchmark notes.
- `configs/amgx/README.md` summarizes PyAMGX recommendations.
- `run_logs/raw_cuda_fused_coop_lu_findings_20260720.md` records fused raw
  cooperation LU and modal compatibility status.
- `run_logs/raw_cuda_hdg_findings_20260719.md` is the earlier raw CUDA baseline.
- `run_logs/adv_rea_amgx_config_findings_20260720.md` records AMGX sweeps and tolerance studies.

See [MANUAL.md](MANUAL.md) for detailed CLI and configuration notes.
