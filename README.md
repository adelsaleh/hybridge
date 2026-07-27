# hdgfem

`hdgfem` is a discontinuous Galerkin / hybridizable discontinuous Galerkin
research codebase.  The package provides mesh, reference-element, DG field,
assembly, linear algebra, solver, plotting, and optional GPU backend modules for
HDG experiments.

The main package workflows are advection-reaction, diffusion-reaction, and
fixed-mesh guiding-center HDG solves.  Current performance work focuses on
raw-CUDA trace assembly, direct device CSR handoff to AMGX, reusable solver
classes for unsteady runs, tangent zero-boundary-flux transport, and
high-order guiding-center/diocotron benchmarks.

A separate interested-reader part of the repository studies diocotron-like
equilibria of the guiding-center model through the semilinear elliptic equation
`-Delta phi = f(phi)`.  Those scripts are not core package dependencies;
DOLFINx is used there only as an optional continuous-Galerkin comparison path
against the native HDG implementation.

## Quick Start

Run commands from the repository root:

```bash
source .venv/bin/activate
python -m pip install -e .
python -m pytest
```

Cheap smoke checks:

```bash
python -m scripts.advection_reaction.run_adv_rea_cases --print-preset --dry-run
python -m scripts.diffusion_reaction.run_diff_rea_cases --print-preset --dry-run
```

List packaged manufactured-run presets:

```bash
python -m scripts.advection_reaction.run_adv_rea_cases --list-presets
python -m scripts.diffusion_reaction.run_diff_rea_cases --list-presets
python scripts/guiding_center/run_guiding_center_cases.py --list-presets
```

## Main Workflows

Package-level CPU/PETSc advection-reaction runs use
`scripts/advection_reaction/run_adv_rea_cases.py`.  The reusable Python API is
in `hdgfem.solvers.adv_rea`.

```bash
python -m scripts.advection_reaction.run_adv_rea_cases test2_scipy_ilu_upwind -p 4 --lc 0.05
```

Package-level diffusion-reaction runs use
`scripts/diffusion_reaction/run_diff_rea_cases.py`.  The reusable Python API is
in `hdgfem.solvers.diff_rea`.

```bash
python -m scripts.diffusion_reaction.run_diff_rea_cases quadratic_poisson
```

The current Cupyx/upwind-GS advection performance path is:

```bash
python scripts/advection_reaction/run_adv_rea_upwind_gs_cupyx.py --help
python scripts/advection_reaction/experimental/check_upwind_block_gs_onfly_adv_rea.py --help
```

The first script is the narrow fast path: Numba eliminated assembly emits
ordered scalar COO plus dense trace-block COO, a forward upwind block-GS
preconditioner is built from those blocks, the compact preconditioner is
transferred to CuPy, and Cupyx solves the scaled ordered trace system.  The
experimental checker compares the CSR-reference, scalar-COO, and ordered
block-COO builders before using the fast path for benchmark claims.

Standalone GPU benchmark runners live under `scripts/gpu/`:

```bash
python -m scripts.gpu.run_adv_rea_gpu4_hdg --help
python -m scripts.gpu.run_diff_rea_gpu4_hdg --help
python -m scripts.gpu.sweep_adv_rea_gpu4_hdg --help
python -m scripts.gpu.check_upwind_scc_host_pyamgx_adv_rea --help
```

The advection GPU runner supports CuPy assembly, raw-CUDA fused assembly, direct
raw-CUDA CSR emission, Cupyx solver experiments, and AMGX solves through
PyAMGX.  The diffusion GPU runner supports NumPy/Numba/CuPy/raw-CUDA reduced
assembly, direct raw-CUDA CSR emission, raw-CUDA reconstruction, device primal
postprocessing, and fixed-operator/RHS-only reuse for unsteady Poisson-like
steps.  See `docs/gpu_hdg_modules.md` and `configs/amgx/README.md` for current
benchmark notes and AMGX presets.

Fixed-mesh guiding-center cases live under `scripts/guiding_center/`:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH   .venv/bin/python scripts/guiding_center/run_guiding_center_cases.py   --preset diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx   --num-steps 10 --plot-every 0
```

The runner couples diffusion-reaction Poisson solves with advection-reaction
transport, supports independent Poisson/transport backend and solver choices,
writes CSV/JSONL diagnostics, updates PyVista scalar arrays in place, and plots
density only by default.  Add `--plot-both` to show density and potential.

Optional semilinear diocotron-equilibrium scripts live under
`scripts/diocotron_hdg/` and `scripts/diocotron_dolfinx/`.  They are documented
in `MANUAL.md` and the Strategy A notes under `docs/strategyA_band_parameter_study/`.

## Package Map

- `hdgfem/core/`: meshes, mesh caching, bases, quadrature, DG spaces, DG fields,
  vector fields, transfer, and adaptivity helpers.
- `hdgfem/assembly/`: NumPy HDG local matrices, trace assembly helpers,
  projection helpers, face-dense diffusion assembly, and Gram operators.
- `hdgfem/backends/`: optional Numba, CuPy, Cupyx, raw-CUDA, PyAMGX, and fused
  benchmark adapters.  This includes table-driven Numba assembly, CuPy device
  mirrors, raw-CUDA advection/diffusion kernels, direct CSR-to-AMGX handoff,
  shared PyAMGX resource management, and device reconstruction/postprocessing
  helpers.  Optional dependencies are imported lazily.
- `hdgfem/kernels/`: low-level Numba kernels used by backend wrappers.
- `hdgfem/linalg/`: sparse trace-system assembly/solves, boundary dof
  reduction, row scaling, upwind-SCC ordering, upwind block-GS preconditioners,
  and CuPy export of compact preconditioner data.
- `hdgfem/solvers/`: advection-reaction and diffusion-reaction solver APIs,
  including reusable stateful solver classes.
- `hdgfem/io/`: plotting and console-output helpers.
- `scripts/`: command-line runners, benchmarks, diagnostics, and experiments.
- `tests/`: focused regression tests.

## Optional Runtime Stacks

The base package uses NumPy, SciPy, and Numba.  PETSc, CuPy/Cupyx, PyAMGX/AMGX,
and DOLFINx are optional and needed only by selected solver paths or scripts.

Numba CPU thread controls are read when Python starts:

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

CuPy/Cupyx and PyAMGX are separate GPU layers:

```bash
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
python -c "import cupyx.scipy.sparse, cupyx.scipy.sparse.linalg; print('cupyx sparse ok')"
python -c "import pyamgx; print('pyamgx ok')"
```

CuPy provides device arrays, sparse matrix construction, raw-kernel support, and
Cupyx Krylov solvers.  PyAMGX is needed only for AMGX-backed solves and also
requires the AMGX shared library to be visible through `LD_LIBRARY_PATH`,
`ldconfig`, or rpath.  Cupyx solve dtype defaults to `float64`; set
`HDGFEM_CUPYX_DTYPE=float32` only for explicit single-precision experiments.

DOLFINx is optional and intentionally outside the main HDG package path.  Use a
separate DOLFINx environment when running the guiding-center comparison scripts
so its MPI/PETSc stack does not constrain the normal `hdgfem` environment.

## Documentation

Start with [MANUAL.md](MANUAL.md) for detailed CLI, backend, API, data-model,
and algorithm notes.  The manual also indexes every project Markdown document.
Useful supporting notes include:

- [docs/gpu_hdg_modules.md](docs/gpu_hdg_modules.md): standalone GPU runner status and benchmark notes.
- [configs/amgx/README.md](configs/amgx/README.md): AMGX/PyAMGX presets and recommendations.
- [docs/algorithms/gpu_assembly_solve_paths.md](docs/algorithms/gpu_assembly_solve_paths.md): GPU assembly/solve path map and direct CSR-to-AMGX notes.
- [docs/algorithms/advection_reaction_solver_configurations.md](docs/algorithms/advection_reaction_solver_configurations.md): current advection-reaction solver/preconditioner ranking and caveats.
- [docs/algorithms/upwind_block_gs_preconditioner/](docs/algorithms/upwind_block_gs_preconditioner/): mathematical upwind block-GS preconditioner note.
- [TODO.md](TODO.md): current GPU, upwind-GS, solver API, and backend cleanup roadmap.
