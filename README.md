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

The supported early-alpha solver imports and update/failure semantics are
defined in `docs/reference/solver_api_alpha.md`. Reusable solver classes are the primary
interface; the canonical functional solvers remain supported for one-shot use.
Supported assembly, sparse-solve, reconstruction, and host/device residency
combinations are defined in `docs/reference/backend_capabilities.md`.

## Quick Start

For a minimal host install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development from a checkout:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

See [docs/getting_started/installation.md](docs/getting_started/installation.md) for dependency groups,
installed-wheel scope, clean-install qualification, and CI commands.

Cheap smoke checks:

```bash
python -m scripts.advection_reaction.run_cases --print-preset --dry-run
python -m scripts.diffusion_reaction.run_cases --print-preset --dry-run
```

Run the complete minimal host examples:

```bash
python examples/advection_reaction_minimal.py
python examples/diffusion_reaction_minimal.py
```

These examples start from a structured mesh, define manufactured PDE data,
solve the HDG trace system through the package-root API, and verify the field
error and physical residual. See [MANUAL.md](MANUAL.md#minimal-end-to-end-examples)
for the annotated source.

List checkout-only manufactured-run presets:

```bash
python -m scripts.advection_reaction.run_cases --list-presets
python -m scripts.diffusion_reaction.run_cases --list-presets
python scripts/guiding_center/run_guiding_center_cases.py --list-presets
```

## Main Workflows

Package-level CPU/PETSc advection-reaction runs use
`scripts/advection_reaction/run_cases.py`. Application code should
import the reusable solver API from `hdgfem` or
`hdgfem.solvers.advection_reaction`; `hdgfem.solvers.adv_rea` remains an
explicit compatibility shim.

```bash
python -m scripts.advection_reaction.run_cases test2_scipy_ilu_upwind -p 4 --lc 0.05
```

Package-level diffusion-reaction runs use
`scripts/diffusion_reaction/run_cases.py`. Application code should
again use the package-root exports or `hdgfem.solvers.diffusion_reaction`;
`hdgfem.solvers.diff_rea` remains a compatibility path.

```bash
python -m scripts.diffusion_reaction.run_cases quadratic_poisson
```

The current Cupyx/upwind-GS advection performance path is:

```bash
python scripts/advection_reaction/run_upwind_gs_cupyx.py --help
python scripts/advection_reaction/experiments/check_upwind_block_gs_on_the_fly.py --help
```

The first script is the narrow fast path: Numba eliminated assembly emits
ordered scalar COO plus dense trace-block COO, a forward upwind block-GS
preconditioner is built from those blocks, the compact preconditioner is
transferred to CuPy, and Cupyx solves the scaled ordered trace system.  The
experimental checker compares the CSR-reference, scalar-COO, and ordered
block-COO builders before using the fast path for benchmark claims.

Standalone GPU benchmark runners live under `scripts/gpu/`:

```bash
python -m scripts.gpu.run_advection_reaction_cuda --help
python -m scripts.gpu.run_diffusion_reaction_cuda --help
python -m scripts.gpu.sweep_cuda_hdg --help
python -m scripts.gpu.check_advection_upwind_scc_host_pyamgx --help
```

The advection GPU runner supports CuPy assembly, raw-CUDA fused assembly, direct
raw-CUDA CSR emission, Cupyx solver experiments, and AMGX solves through
PyAMGX.  The diffusion GPU runner supports NumPy/Numba/CuPy/raw-CUDA reduced
assembly, direct raw-CUDA CSR emission, raw-CUDA reconstruction, device primal
postprocessing, and fixed-operator/RHS-only reuse for unsteady Poisson-like
steps. Public raw-CUDA solver and runner defaults use the equation- and
order-aware `raw_block_size="auto"` policy documented in
`docs/algorithms/raw_cuda_launch_policy.md`; explicit launch sizes remain
available for benchmark reproduction. See `docs/backends/cuda_runners.md` and
`configs/amgx/README.md` for current benchmark notes and AMGX presets.

Fixed-mesh guiding-center cases live under `scripts/guiding_center/`:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH   .venv/bin/python scripts/guiding_center/run_guiding_center_cases.py   --preset diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx   --num-steps 10 --plot-every 0
```

The runner couples diffusion-reaction Poisson solves with advection-reaction
transport and supports `--time-scheme si-euler|predictor-corrector`. It accepts
independent Poisson/transport backend and solver choices, writes CSV/JSONL
diagnostics, updates PyVista scalar arrays in place, and plots density only by
default. Add `--plot-both` to show density and potential. Raw-CUDA stages pass
projected/current trace guesses directly on device; `--transport-retry-policy
amgx-robust` retries a failed assembled transport system without host CSR
materialization. Compare both temporal schemes with
`scripts/guiding_center/run_guiding_center_temporal_convergence.py`.

Optional semilinear diocotron-equilibrium scripts live under
`scripts/diocotron_hdg/` and `scripts/diocotron_dolfinx/`.  They are documented
in `MANUAL.md` and the Strategy A notes under `docs/research/strategy_a_band_parameter_study/`.

## Package Map

- `hdgfem/core/`: meshes, mesh caching, bases, quadrature, DG spaces, DG fields,
  vector fields, transfer, and adaptivity helpers.
- `hdgfem/assembly/`: NumPy HDG local matrices, trace assembly helpers,
  projection helpers, face-dense diffusion assembly, and Gram operators.
- `hdgfem/backends/`: optional Numba, CuPy, Cupyx, raw-CUDA, PyAMGX, and fused
  benchmark adapters.  This includes table-driven Numba assembly, CuPy device
  mirrors, raw-CUDA advection/diffusion kernels, direct CSR-to-AMGX handoff,
  shared PyAMGX resource management, and device reconstruction/postprocessing
  helpers. Optional dependencies are imported lazily. The role map and naming
  migration policy are in [docs/backends/README.md](docs/backends/README.md).
- `hdgfem/kernels/`: low-level Numba kernels used by backend wrappers.
- `hdgfem/linalg/`: sparse trace-system assembly/solves, boundary dof
  reduction, row scaling, upwind-SCC ordering, upwind block-GS preconditioners,
  and CuPy export of compact preconditioner data.
- `hdgfem/solvers/`: advection-reaction and diffusion-reaction solver APIs,
  including reusable stateful solver classes in descriptive full-name
  implementation modules and temporary abbreviated compatibility shims.
- `hdgfem/io/`: plotting and console-output helpers.
- `scripts/`: command-line runners, benchmarks, diagnostics, and experiments.
- `tests/`: focused regression tests.

Portable repository extras are declared for tests, Gmsh meshes, plotting, and
release tooling. Gmsh is an optional dependency so core structured-mesh paths
remain lean, but the `mesh` extra is highly recommended because most scripts,
tests, and realistic configurations use Gmsh:

```bash
python -m pip install -e '.[test,mesh,plot]'
```

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

The optional `pypardiso` backend provides a multithreaded oneMKL PARDISO direct
solve for host CSR systems assembled by the NumPy, Numba, CuPy, or raw-CUDA
paths that support host materialization:

```bash
python -m pip install -e '.[pardiso]'
export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=1
```

Use `solver="pypardiso"` or `solver="pardiso"` for general real matrices. For
mathematically symmetric-positive-definite systems, `solver="pypardiso-spd"`
(or `"pardiso-spd"`) validates symmetry, stores only the upper triangle, and
selects PARDISO `mtype=2`. Do not use the SPD aliases for advection operators,
symmetric-indefinite systems, or matrices whose definiteness is unknown.

The import is lazy, all completed solves are checked against the original full,
unscaled system, and `hdgfem.linalg.clear_pypardiso_cache()` releases every
process-global cached factorization. Run
`python scripts/dev/check_pypardiso.py --side 100 --repeats 3` for local HDG
parity and host timing evidence. The matched order-6, 51,200-triangle Poisson
presets are `trigonometric_poisson_50k_scipy_direct` and
`trigonometric_poisson_50k_pypardiso_spd`; measured evidence is recorded in
`docs/releases/early_alpha.md`. Set thread counts before Python starts and avoid
concurrent PARDISO calls in one process; HDGFEM serializes this shared backend.


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

Residency is an executable alpha contract, not an inference from the selected
backend name. Host-assembled Cupyx solves intentionally upload the sparse
matrix and download the solution. Raw-CUDA CSR-to-AMGX advection and diffusion
solves keep the reduced system, trace, and reconstructed fields on device when
host materialization is disabled. The release GPU lane counts these transfer
boundaries and independently checks the original-system residual; see
[docs/reference/backend_capabilities.md](docs/reference/backend_capabilities.md).

DOLFINx is optional and intentionally outside the main HDG package path.  Use a
separate DOLFINx environment when running the guiding-center comparison scripts
so its MPI/PETSc stack does not constrain the normal `hdgfem` environment.

## Early-Alpha Validation

Run the bounded release lanes from the repository root:

```bash
python scripts/dev/alpha_test_matrix.py host-fast
python scripts/dev/alpha_test_matrix.py install-smoke
python scripts/dev/alpha_test_matrix.py cpu-parity
python scripts/dev/alpha_test_matrix.py gpu-smoke
```

The GPU lane is opt-in during development but required on the production CUDA
environment before an alpha tag. Exact targets, cadence, acceptance rules,
scheduled performance commands, current evidence, and known gaps are documented
in [docs/development/alpha_test_matrix.md](docs/development/alpha_test_matrix.md) and
[docs/releases/early_alpha.md](docs/releases/early_alpha.md).

The scheduled lane requires the recommended Gmsh runtime, preflights its import,
and enables the opt-in diffusion geometry parity cases instead of silently
recording them as skips.

The current package candidate is `0.1.0a1`. On 2026-08-05 the candidate passed
all four local release lanes: 490 host tests, the isolated wheel smoke, 14 CPU
parity cases, and 10 GPU smoke cases. The earlier Gmsh-enabled broad repository
suite passed 606 tests with zero skips, and the four focused Gmsh parameters
cover 16 geometry/order combinations. The release record retains
dependency-isolated installation and distribution metadata evidence. The first
hosted Python 3.10/3.12 CI run remains an explicit pre-tag gate; local success
is not recorded as hosted evidence.

Linear solve acceptance is backend-neutral: native success, finite values, and
both solver-system and unscaled physical residual targets must pass. See
[docs/reference/solver_convergence_contract.md](docs/reference/solver_convergence_contract.md).


## Roadmap

The near-term goal is a bounded early-alpha production surface, not a freeze of
every research backend. Work is ordered as follows:

1. **Release quality:** keep package-root solver imports, failure semantics,
   backend capabilities, executable examples, clean-install checks, and hosted
   Python 3.10/3.12 CI synchronized.
2. **Backend consolidation:** continue splitting the canonical CUDA modules by
   assembly, sparse-solve, reconstruction, and device-data roles without
   reintroducing numeric or equation-abbreviated filenames.
3. **Extensible HDG formalism:** add an `HDGTraceField` and user-defined local
   bilinear/numerical-flux interfaces so new equations and transmission
   conditions can use the same host/device backend machinery.
4. **Density transport study:** compare host SCC ordering with ILU/upwind-GS,
   dependency-driven Numba upwind solves, and guiding-center Poisson/transport
   combinations including CG, SUPG, and HDG electric fields.

The detailed, actively maintained task breakdown is in [TODO.md](TODO.md).

## Documentation

Start with [MANUAL.md](MANUAL.md) for detailed CLI, backend, API, data-model,
and end-to-end examples. [docs/README.md](docs/README.md) is the organized
index for contracts, backend guides, algorithm notes, and research outputs.
Useful supporting notes include:

- [docs/reference/solver_api_alpha.md](docs/reference/solver_api_alpha.md): bounded early-alpha public solver API and compatibility contract.
- [docs/reference/solver_convergence_contract.md](docs/reference/solver_convergence_contract.md): normalized status, residual acceptance, retry, and cleanup contract.
- [docs/getting_started/installation.md](docs/getting_started/installation.md): package dependency groups, wheel scope, install smoke, and CI qualification.
- [docs/reference/backend_capabilities.md](docs/reference/backend_capabilities.md): authoritative early-alpha backend and residency matrix.
- [docs/development/alpha_test_matrix.md](docs/development/alpha_test_matrix.md): executable host, CPU parity, GPU smoke, and scheduled validation matrix.
- [docs/releases/early_alpha.md](docs/releases/early_alpha.md): current release evidence, reviewed skips, and known gaps.
- [docs/backends/cuda_runners.md](docs/backends/cuda_runners.md): standalone GPU runner status and benchmark notes.
- [docs/backends/README.md](docs/backends/README.md): backend role map, module ownership, and naming rules.
- [configs/amgx/README.md](configs/amgx/README.md): AMGX/PyAMGX presets and recommendations.
- [docs/algorithms/gpu_assembly_solve_paths.md](docs/algorithms/gpu_assembly_solve_paths.md): GPU assembly/solve path map and direct CSR-to-AMGX notes.
- [docs/algorithms/advection_reaction_solver_configurations.md](docs/algorithms/advection_reaction_solver_configurations.md): current advection-reaction solver/preconditioner ranking and caveats.
- [docs/algorithms/upwind_block_gs_preconditioner/](docs/algorithms/upwind_block_gs_preconditioner/): mathematical upwind block-GS preconditioner note.
- [TODO.md](TODO.md): current GPU, upwind-GS, solver API, and backend cleanup roadmap.
