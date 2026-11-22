# hybridge Manual

This manual describes the current `hybridge` package, its command-line workflows,
and the reusable Python formalism behind the solvers.  The repository is a
research codebase for discontinuous Galerkin and hybridizable discontinuous
Galerkin methods, with current emphasis on advection-reaction and
diffusion-reaction HDG trace systems.

The main package path is self-contained around NumPy, SciPy, and Numba.  PETSc,
CuPy/Cupyx and PyAMGX/AMGX are optional stacks used by selected runners.
The [installation guide](docs/getting_started/installation.md) separates the
base numerical package from optional GPU, sparse-solver, and plotting tools.

## Documentation Map

The detailed navigation hub is [docs/README.md](docs/README.md). It separates
user documentation, executable alpha contracts, backend guides, algorithm
notes, and generated research outputs. The release-critical entry points are:

- [README.md](README.md): overview, quick start, release status, and roadmap.
- [docs/reference/solver_api_alpha.md](docs/reference/solver_api_alpha.md): supported public solver API.
- [docs/reference/backend_capabilities.md](docs/reference/backend_capabilities.md): supported backend and residency combinations.
- [docs/reference/solver_convergence_contract.md](docs/reference/solver_convergence_contract.md): solve acceptance and failure semantics.
- [docs/development/alpha_test_matrix.md](docs/development/alpha_test_matrix.md): executable release gates.
- [docs/releases/early_alpha.md](docs/releases/early_alpha.md): current evidence and open pre-tag gates.
- [docs/backends/README.md](docs/backends/README.md): backend roles, module ownership, and naming rules.
- [TODO.md](TODO.md): detailed active roadmap.

## Package Architecture

```text
.
  hybridge/
    __init__.py   public package exports and lazy solver imports
    runtime/      optional-dependency gates, precision, logging, errors, devices, threads
    core/         mesh/cache, basis, quadrature, DG spaces/fields, transfer, adaptivity,
                  mass matrices, L2 projection, CuPy mirrors (device/)
    cases/        analytic coefficient sets and initial profiles
    linalg/       solve dispatch and results, reduction, ordering, scaling, preconditioners;
                  amgx/ (PyAMGX), gpu/ (CuPy/Cupyx), multigrid/ (face-block hp-MG)
    hdg/          shared HDG layer: condensation, coefficients, advection stabilization,
                  trace maps, Gram operators, Numba helpers, cuda/ source library
    transport/    first-order HDG (advection-reaction) on all backends
    mixed/        mixed HDG (diffusion-reaction, ADR) on all backends, postprocess/
    conforming/   continuous-Galerkin Poisson assembly and condensation
    solvers/      advection-reaction, diffusion-reaction, ADR and CG Poisson APIs, capabilities
    diagnostics/  error, solver-result, and guiding-center diagnostics
    io/           output formatting and plotting helpers
  scripts/        command-line runners and research harnesses
  examples/       copy-runnable base-install solver examples
  docs/           index, alpha contracts, backend guides, algorithms, releases
  run_configs/    version-controlled benchmark and solver presets
  configs/        backend configuration files, currently AMGX presets
  tests/          focused regression tests
```

The `hybridge/` subpackages are listed in layer order. A module imports only
from its own layer or from layers above it in this list; `transport` and
`mixed` share a layer and never import each other.
`tests/test_package_layering.py` enforces the rule with no allowed violations.

Important module groups:

- `hybridge.core.mesh`: `DGMesh`, structured rectangle meshes, Gmsh-backed
  rectangle/disc/star/smooth-star meshes, mesh caching, and Gmsh thread
  controls.
- `hybridge.core.basis`, `hybridge.core.quadrature`, and `hybridge.core.space`:
  reference bases, quadrature tables, `DGSpace`, `DGTraceSpace`, `DGField`, and
  `VectorDGField`.
- `hybridge.core.field_ops` and `hybridge.core.trace_transfer`: reusable scalar/vector
  field combinations, host/device solution and trace extraction, skeleton
  projection, and trace-degree prolongation.
- `hybridge.core.transfer` and `hybridge.core.adaptivity`: field transfer between
  meshes and PDE-agnostic indicator/remeshing utilities.
- `hybridge.diagnostics`: reusable scalar error reports, drift calculations,
  solver/timing summaries, and guiding-center modal diagnostics.
- `hybridge.io.comparison` and `hybridge.io.plot`: sampled comparisons and
  plotting support used by runners. AMGX configuration loading is in
  `hybridge.linalg.amgx.config`.
- `hybridge.core.mass`, `hybridge.hdg.matrices`, `hybridge.transport.local_numpy`,
  and `hybridge.mixed.local_numpy`: reference NumPy mass, trace-stabilization,
  advection, and mixed local matrices, with output-buffer accumulation routines
  used by CPU solvers and validation code.
- `hybridge.hdg.condensation`: reusable static-condensation and trace-assembly
  helpers shared by solver implementations.
- `hybridge.transport.numba`: projected advection-reaction trace assembly,
  boundary-eliminated and zero-flux assembly, optional ordered block-COO
  emission, and Numba reconstruction helpers. `hybridge.mixed.numba` is the
  diffusion counterpart, with persistent Schur-LU/Cholesky factors for identity
  diffusion; see the [host cache guide](docs/backends/numba_diffusion.md).
- `hybridge.runtime.optional`, `hybridge.core.device`, `hybridge.linalg.gpu`, and
  `hybridge.linalg.amgx`: CuPy/Cupyx/PyAMGX import guards, device mirrors of mesh
  and reference data, host/device sparse conversion and row scaling, Cupyx
  Krylov wrappers, device ILU(1), host-ILU export to device triangular solves,
  and host and device PyAMGX CSR handoff.
- `hybridge.transport.cupy`, `hybridge.transport.cuda`,
  `hybridge.transport.raw_cuda`, `hybridge.mixed.cupy`, and
  `hybridge.mixed.raw_cuda`: canonical CuPy and raw-CUDA assembly,
  reconstruction, and device-data paths for the two HDG families. The device
  AMGX solve shared by both is `hybridge.linalg.amgx.device_solver`.
- `hybridge.linalg.system`: global trace matrix assembly and solve routing to
  SciPy, PyPardiso, PETSc, Cupyx, or AMGX. Boundary dof elimination/expansion
  is in `hybridge.linalg.reduction`; diagonal row scaling, `SolveResult`, and
  residual diagnostics are in `hybridge.linalg.results`.
- `hybridge.linalg.ordering`: upwind-SCC graph ordering for trace edges and
  sparse-pattern plotting diagnostics.
- `hybridge.linalg.upwind_block_gs`: CSR-reference level-scheduled upwind block
  Gauss-Seidel preconditioner builder.
- `hybridge.linalg.upwind_block_gs_on_the_fly`: scalar-COO and ordered block-COO
  builders that construct the same forward upwind block-GS preconditioner
  without scanning a finished CSR matrix.
- `hybridge.linalg.gpu.upwind_block_gs`: device application of compact host-built
  upwind block-GS data through a Cupyx `LinearOperator`.

Public application code should import solver classes and functions from
`hybridge`, `hybridge.solvers`, or the canonical implementation modules
`hybridge.solvers.advection_reaction` and
`hybridge.solvers.diffusion_reaction`. The abbreviated `adv_rea` and `diff_rea`
modules are compatibility shims for the first alpha; production backends and
kernels use descriptive names. Internal module paths below the package root and
`hybridge.solvers` carry no compatibility guarantee; the package reorganization
moved them without re-exports. See
[docs/backends/README.md](docs/backends/README.md) before adding a backend
module.

## Environment Setup

Install the base host package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development from a checkout:

```bash
python -m pip install -e '.[test,mesh,plot]'
python -m pytest
```

The `test` extra includes pytest, Matplotlib for plotting paths exercised by
the host matrix, and `tomli` on Python 3.10 for package-metadata validation.

The installed wheel contains the `hybridge` library; runners, configs, and
benchmarks remain checkout-only. See `docs/getting_started/installation.md` for
the complete dependency and release-qualification contract.

HYBRIDGE was called `hdgfem` before `0.1.0a2`; import it as `hybridge`. Its
environment variables are `HYBRIDGE_*` (for example `HYBRIDGE_PRECISION`,
`HYBRIDGE_HOST_THREADS` and `HYBRIDGE_MAGMA_ROOT`). For this release the old
`HDGFEM_*` names are still read, with a `DeprecationWarning`.

The base package uses NumPy, SciPy, and Numba.  Tune Numba CPU parallelism
before Python starts:

```bash
export NUMBA_NUM_THREADS=40
export OMP_NUM_THREADS=1
```

Mesh generation defaults to local caching under `.cache/hybridge/meshes` and logs
cache hits, misses, and fallbacks.  GPU advection runners also accept
`--gmsh-num-threads` for parallel CPU meshing.

Gmsh remains optional, but installing the `mesh` extra is highly recommended:
most runners, geometry tests, and realistic configurations use Gmsh meshes.

```bash
python -m pip install -e '.[mesh]'
```

### PETSc

PETSc is optional.  `petsc4py` is imported only when a PETSc solve is requested.
A PETSc solve needs a matched PETSc / `petsc4py` pair visible to the active
Python environment:

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-linux-c-opt
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
python -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
```

Build or install `petsc4py` from the matching PETSc checkout or exact matching
release.  Avoid mixing system `petsc4py` with virtualenv NumPy.  For the known
PETSc 3.22.2 setup used in this repository, keep NumPy below the version limit
required by the active Numba release and use a Cython/setuptools pair compatible
with that PETSc binding.

Useful backend checks:

```bash
python -c "from petsc4py import PETSc; k=PETSc.KSP().create(); k.getPC().setType('gamg'); print('GAMG ok')"
python -c "from petsc4py import PETSc; k=PETSc.KSP().create(); pc=k.getPC(); pc.setType('hypre'); pc.setHYPREType('boomeramg'); print('Hypre/BoomerAMG ok')"
python -c "from petsc4py import PETSc; k=PETSc.KSP().create(); pc=k.getPC(); pc.setType('lu'); pc.setFactorSolverType('mumps'); print('MUMPS ok')"
```
### PyPardiso Host Direct Solver

The optional `pardiso` extra installs `pypardiso`, its oneMKL runtime, and the
host direct-solver adapter:

```bash
python -m pip install -e '.[pardiso]'
export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=1
```

```python
from hybridge.linalg import clear_pypardiso_cache, solve_global_system

result = solve_global_system(rows, cols, data, rhs, size, solver="pypardiso")
assert result.converged and result.physical_residual_target_met
clear_pypardiso_cache()
```

`solver="pardiso"` is an alias for the general real-matrix path. The
`pypardiso-spd` and `pardiso-spd` aliases are reserved for mathematically
symmetric-positive-definite matrices. The SPD path verifies matrix symmetry,
converts the full CSR input to upper-triangular CSR storage, and selects
PARDISO `mtype=2`. Symmetry is checked by HYBRIDGE; positive definiteness remains
a property the caller and PDE discretization must guarantee.

Both paths canonicalize inputs to sorted `float64` CSR, import `pypardiso` only
when selected, and return the common `SolveResult`. Native success is never
enough: the solution is checked against the original full, unscaled matrix and
right-hand side, which also catches silent bad results from singular systems.

`pypardiso` owns process-global solvers and can reuse the latest factorization
for each matrix type. HYBRIDGE serializes calls; call `clear_pypardiso_cache()`
when the factorizations are no longer needed. Set MKL and OpenMP thread counts
before Python starts, especially when Numba is also active, to prevent nested
oversubscription. The backend is available to NumPy/Numba host paths and to
device assembly paths that explicitly materialize the reduced system on host.

Run `python scripts/dev/check_pypardiso.py --side 100 --repeats 3` for compact
advection/diffusion parity and timing checks. For the matched p=6 HDG Poisson
benchmark with 51,200 triangles, run:

```bash
python scripts/diffusion_reaction/run_cases.py trigonometric_poisson_50k_scipy_direct
python scripts/diffusion_reaction/run_cases.py trigonometric_poisson_50k_pypardiso_spd
```

The measured machine-specific comparison is recorded in
`docs/releases/early_alpha.md`.


### CuPy, Cupyx, and AMGX

For a compact configuration using generic CUDA/AMGX/PyAMGX directories, see
[GPU runtime](docs/getting_started/installation.md#gpu-runtime).

The supported AMGX path uses our maintained
[AMGX](https://github.com/adelsaleh/AMGX/tree/hdg-cuda13-integration) and
[PyAMGX](https://github.com/adelsaleh/pyamgx/tree/quality-of-life) forks, with
HDG block-system extensions and GPU diagnostics. See the
[fork setup guide](docs/getting_started/forked_amgx_stack.md) for supported
revisions and installation.

CuPy/Cupyx and PyAMGX are separate optional GPU layers:

```text
CuPy arrays/raw kernels      device storage, element kernels, reconstruction kernels
Cupyx sparse/linalg         CSR/COO matrices, Krylov solvers, sparse triangular solves
PyAMGX bridge               AMGX setup/solve from CuPy CSR data and device pointers
```

Cupyx-only solves do not use AMGX.  They need a CUDA-compatible CuPy install
whose sparse modules are present:

```bash
python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
python -c "import cupyx.scipy.sparse, cupyx.scipy.sparse.linalg; print('cupyx sparse ok')"
```

AMGX solves additionally need `pyamgx` and the AMGX shared libraries:

```bash
export AMGX_LIB_DIR=/path/to/amgx/lib
export LD_LIBRARY_PATH=$AMGX_LIB_DIR:$LD_LIBRARY_PATH
python -c "import pyamgx; print('pyamgx ok')"
```

The shared GPU utilities are:

```text
hybridge.runtime.optional
  require_cupy                       import guard for CuPy
  require_cupyx_sparse               import guard for cupyx.scipy.sparse
  require_cupyx_sparse_linalg        import guard for cupyx.scipy.sparse.linalg
hybridge.linalg.gpu.sparse
  scipy_csr_to_cupy                  copy a SciPy CSR matrix to CuPy CSR
  scipy_coo_to_cupy_csr              copy host COO arrays and build CuPy CSR on device
hybridge.linalg.gpu.cupyx
  solve_cupyx_csr                    run Cupyx cg, bicgstab, cgs, or gmres
  build_cupyx_ilu_preconditioner     build device ILU(1) with Cupyx spilu
  build_cupyx_exported_host_ilu_preconditioner
                                     build SciPy SuperLU ILU and apply L/U on device
hybridge.linalg.amgx.host
  solve_pyamgx_csr                   run AMGX through PyAMGX from CuPy CSR data
```

Cupyx solves default to double precision.  Set `HYBRIDGE_CUPYX_DTYPE=float32` or
`HYBRIDGE_CUPYX_DTYPE=float64` before starting Python to force the device matrix
dtype in `hybridge.linalg.gpu.cupyx.solve_cupyx_system`:

```bash
HYBRIDGE_CUPYX_DTYPE=float64 python scripts/advection_reaction/run_upwind_gs_cupyx.py -o 6 -ms 0.01
```

Do not use Cupyx CG with the default left row scaling: left scaling does not
preserve symmetry.  The code rejects `cupyx_solver="cg"` when
`scale_system=True`; use BiCGSTAB/GMRES or disable scaling for a genuinely SPD
operator.

### Backend Implementation Summary

The authoritative public solver support and host/device residency matrix is
[docs/reference/backend_capabilities.md](docs/reference/backend_capabilities.md). The summary below
also names lower-level research capabilities and must not be read as a promise
that every assembly/solve/reconstruction cross-product is supported. The
modules implementing each backend are listed in
[docs/backends/README.md](docs/backends/README.md).

The solver APIs expose multiple assembly and solve paths. The important rule is
that fast Numba and raw-CUDA kernels are table driven: coefficients must already
be represented as `DGField` or `VectorDGField` objects, usually via
`space.project_callable(...)`, `space.constant(...)`, or `space.zeros(...)`.
For advection-reaction, they may instead be given at the quadrature points
(`QuadratureVectorField`, `QuadratureScalar`; see Coefficient Ownership below).
NumPy and CuPy paths can still sample analytic callables directly.

Advection-reaction backends:

```text
NumPy          reference assembly/reconstruction; accepts callables and DG fields
Numba          projected/table assembly and reconstruction; supports boundary elimination,
               tangent zero-boundary-flux, upwind-SCC ordering, and compact zero/constant descriptors
CuPy           device assembly baseline; accepts CuPy-compatible callables and DG fields
raw-CUDA       fused projected assembly; supports COO and direct CSR emission, safe or cooperative LU,
               legendre-modal and legacy-lagrange traces, device AMGX solves, and zero-flux boundaries;
               fused and split3 also take quadrature-point advection and reaction values
```

Diffusion-reaction backends:

```text
NumPy          reference reduced assembly, reconstruction, and host postprocessing
Numba          trace-space-aware projected assembly/reconstruction for legacy and modal traces
CuPy           device assembly/reconstruction and CuPy RT_projection flux recovery; primal
               postprocessing runs on host Numba
raw-CUDA       identity-diffusion/zero-reaction fast path through p <= 6; supports legacy-lagrange
               and legendre-modal traces, COO or direct CSR emission, raw-CUDA reconstruction,
               full mixed local unknown output, and RHS-only rebuilds for cached fixed operators
```

AMGX/PyAMGX integration:

```text
PyAMGX CSR handoff       uploads CuPy CSR pointers without materializing a host CSR matrix
shared AMGX resources    all live PyAMGX solvers share one process-wide Resources handle, which
                         allows persistent Poisson setup and transient transport solves to coexist
initial guesses          device initial guesses are passed through Vector.upload_raw; no host guess copy is required
fixed Poisson operators  raw-CUDA diffusion can cache CSR data and AMGX setup, then rebuild only RHS per step
transport operators      transport matrices change with beta, so AMGX setup is rebuilt each step for now
```

Transfer behavior is tested at the public solve boundary. A host-assembled
Cupyx solve performs one sparse-matrix upload and one solution download. With
host materialization disabled, raw-CUDA advection and diffusion direct-CSR
AMGX solves perform no `cp.asnumpy` full-array downloads; reduced traces and
reconstructed DG fields remain device backed, and the normalized solve result
still validates the original unscaled residual. Diagnostics or explicit host
result requests intentionally opt back into materialization.

The raw-CUDA diffusion cache is intentionally conservative.  It preserves the
operator only when mesh, order, trace basis, boundary mode, diffusion, reaction,
stabilization, matrix format, and raw block size are unchanged.  Source and
Dirichlet trace changes invalidate only the RHS.  For zero potential boundary
conditions, the cached RHS kernel uses a source-only column layout rather than a
matrix-sized dummy data buffer.

For AMGX convergence diagnostics, distinguish the values printed by AMGX from
post-solve checks.  AMGX `RELATIVE_INI` reports reduction relative to the
initial residual of the supplied initial guess, while runner diagnostics also
record `solver_residual`, `solver_residual_target`, and residuals relative to
`||b||`.  The guiding-center Poisson presets use an absolute-convergence PCGF
config so `poisson_solver_atol` controls the AMGX stop target directly.

## Command-Line Workflows

### Advection-Reaction Presets

Manufactured advection-reaction presets live in
`scripts/advection_reaction/run_cases.py` and problem factories live in
`scripts/advection_reaction/cases.py`.

```bash
python -m scripts.advection_reaction.run_cases --list-presets
python -m scripts.advection_reaction.run_cases --print-preset --dry-run
python -m scripts.advection_reaction.run_cases test2_scipy_ilu_upwind -p 2 --lc 0.30
python -m scripts.advection_reaction.run_cases test2_scipy_ilu_upwind -p 6 --lc 0.03 --verbosity 2
```

The default manufactured `test2` problem solves

```text
beta . grad(u) + r u = f
beta_x = x
beta_y = -y
r      = y**2
f      = y**2
u(x,y) = (a cos(m pi x y) + b sin(n pi x y)) exp(y**2 / 2) + 1
```

with default parameters:

```text
m = 10
n = 15
a = 2
b = 2
```

Sensitivity variants around `test2` are available for solver/preconditioner
robustness checks:

```text
test2_minus10    m, n, a, b reduced by 10 percent
test2_plus10     m, n, a, b increased by 10 percent
test2_amp_skew   same frequencies as test2 with mildly skewed amplitudes
```

Useful runner options:

```text
preset                   preset name from run_cases.py
--list-presets           print available advection presets
--print-preset           print selected preset fields
--dry-run                validate and print the selected preset without solving
-p, --order              override uniform DG polynomial degree
--lc, --mesh-size        override Gmsh target mesh size
--boundary-mode          penalty or eliminate
--trace-ordering         none or upwind-scc
--ilu-permc-spec         SuperLU column permutation for SciPy ILU
--assembly-backend       numpy, numba, or auto
--plot                   show numerical/exact/error plots
--plot-resolution        samples per reference direction for plotting
--verbosity              0 quiet, 1 major phases, 2 substeps
```

A one-assembly solver benchmark is available for preconditioner work:

```bash
python -m scripts.advection_reaction.benchmark_solvers -p 6 --lc 0.01
```

It assembles the `test2` upwind-ordered trace matrix once, builds a reusable
SciPy CSR matrix, and runs selected global solver configurations against the
same matrix and RHS.  That removes mesh generation and assembly from the
per-solver comparison.

Default iterative benchmark configurations include SciPy BICGSTAB/GMRES with
ILU or upwind block-GS, PETSc BICGSTAB/GMRES with ILU, and PETSc ASM/ILU
variants.  Use `--config NAME`, `--include-direct`, `--json-out PATH`,
`--upwind-bgs-apply-mode MODE`, and `--upwind-bgs-sweep SWEEP` to narrow or
expand a benchmark run.

### Package-Backed Upwind-GS/Cupyx Advection Runner

`scripts/advection_reaction/run_upwind_gs_cupyx.py` is a specialized CLI front
end to `AdvectionReactionHDGSolver`; it no longer owns a second assembly,
preconditioner, sparse-solve, or reconstruction implementation.

The selected public solver configuration is:

```text
1. Project source, beta, and reaction into one DGSpace.
2. Assemble the eliminated trace system with the Numba backend.
3. Order active trace edges with package upwind-SCC ordering.
4. Build the package forward upwind block-GS preconditioner.
5. Solve with the selected Cupyx Krylov method.
6. Reconstruct through the solver class and evaluate package error diagnostics.
```

Typical run:

```bash
python scripts/advection_reaction/run_upwind_gs_cupyx.py \
  -o 6 -ms 0.01 \
  --basis dub_orth \
  --trace-basis legacy-lagrange \
  --cupyx-solver bicgstab \
  --maxiter 1500 \
  --rtol 1e-13 \
  --check-rtol 1e-10 \
  -v 2
```

Important controls:

```text
--case                         manufactured case key from cases.py
--mesh-type                    rectangle or structured-rectangle
--basis                        dub_orth, hierarchical C0, or Bernstein element basis
--trace-basis                  legacy-lagrange or legendre-modal
--cupyx-solver                 bicgstab, gmres, cg, or cgs
--gmres-restart                optional GMRES restart forwarded through the solver API
--diagonal-regularization      regularize singular/near-singular edge diagonal blocks
--trace-ordering-flux-tolerance ignore graph fluxes below this magnitude
--numba-threads                runtime Numba thread count within NUMBA_NUM_THREADS
--check-rtol                   independently accepted physical residual threshold
--json-output                  write inputs, diagnostics, timings, and errors
```

This runner requires Numba, CuPy, and Cupyx sparse linear algebra, but not
PyAMGX. Low-level builder tuning belongs to the experiment below; the production
runner deliberately exposes only controls supported by the reusable solver API.

The structural check harness compares the CSR-reference, scalar-COO, and
ordered block-COO preconditioner builders:

```bash
python scripts/advection_reaction/experiments/check_upwind_block_gs_on_the_fly.py \
  -o 6 -ms 0.01 \
  --basis dub_orth \
  --trace-basis legacy-lagrange \
  --skip-solve \
  -v 2
```

It can also export the host-built block-GS object to CuPy and run a Cupyx solve:

```bash
python scripts/advection_reaction/experiments/check_upwind_block_gs_on_the_fly.py \
  -o 6 -ms 0.01 \
  --basis dub_orth \
  --trace-basis legacy-lagrange \
  --cupyx-solve \
  --rtol 1e-13 \
  --check-rtol 1e-10 \
  -v 2
```

### Diffusion-Reaction Presets

Manufactured diffusion-reaction presets live in
`scripts/diffusion_reaction/run_cases.py` and problem factories live in
`scripts/diffusion_reaction/cases.py`.

```bash
python -m scripts.diffusion_reaction.run_cases --list-presets
python -m scripts.diffusion_reaction.run_cases quadratic_poisson
python -m scripts.diffusion_reaction.run_cases tensor_sine_quick --plot
python -m scripts.diffusion_reaction.run_cases tensor_sine_gamg --print-preset
python -m scripts.diffusion_reaction.run_cases tensor_sine_gamg --dry-run
```

Diffusion-reaction presets control optional HDG post-processing with
`hdg_postprocess="none"`, `"primal"`, `"flux"`, or `"both"`.  The primal
postprocessor recovers a degree `p+1` scalar field.  The flux postprocessor
recovers a degree `p+1` vector field whose normal moments match the HDG
numerical flux and whose interior moments match the raw HDG flux against
`[P_{p-1}]^d`.  The host postprocessor supports
`trace_basis="legacy-lagrange"`, `"legendre-modal"`, and `"bernstein"`;
modal traces use signed odd modes on reversed edges.  Numba diffusion assembly is
currently enabled for legacy and modal traces.  Manufactured cases return
the exact conservative flux `q=-kappa grad u`; the runner reports raw and
postprocessed flux errors when available.  To compare the HDG solution, the
postprocessed field and the exact solution in one figure, call
`plot_solution_comparison(result.field, exact, postprocessed=result.postprocessed_field)`
from `hybridge.io`, as in the README's first solve.

Create a new manufactured PDE by adding a factory and `CASE_DEFINITIONS` entry
in `scripts/diffusion_reaction/cases.py`.  Create a new run
configuration by adding a `DiffusionReactionRunPreset` entry to `PRESETS` in
`scripts/diffusion_reaction/run_cases.py`.

Use the optional Numba local-solver block builder by setting
`local_backend="numba"` in a preset.  Tensor diffusion test7 through the main
projected Numba tensor path is available with:

```bash
python -m scripts.diffusion_reaction.run_cases tensor_sine_gamg
```

Experimental hard-coded tensor test7 fused path:

```bash
python -m scripts.diffusion_reaction.experiments.test7_fused \
  --domain structured-rectangle --nx 200 --ny 200 -p 6 \
  --tau 4 --petsc --petsc-preset cg_gamg \
  --volume-quad-1d 7 --edge-quad-1d 7
```

### Standalone GPU Runners

High-performance advection and diffusion runs are driven by standalone scripts
in `scripts/gpu/`:

```bash
python -m scripts.gpu.run_advection_reaction_cuda --help
python -m scripts.gpu.run_diffusion_reaction_cuda --help
python -m scripts.gpu.sweep_cuda_hdg --help
python -m scripts.gpu.check_advection_upwind_scc_host_pyamgx --help
```

There are no root-level GPU compatibility wrappers in `scripts/`; run these
scripts through the `scripts.gpu` module paths above or by explicit
`scripts/gpu/*.py` file paths.

The fused raw-CUDA advection path is the CUDA memory-scaling default.  It
supports `p <= 8` under fused mode with `legacy-lagrange` and `legendre-modal`
trace bases.  Default behavior is:

```text
--raw-lu-mode coop       cooperative LU (default for fused/split3)
--raw-lu-mode safe       historical serial-handoff baseline
--raw-local-assembly fused
--raw-local-assembly auto   split3 from p=8 on (face-BSR, device AMGX), fused below
--raw-matrix-format csr  direct reduced CSR emission when selected
```

The CuPy/PyAMGX advection runner is backed by the reusable package solver in
`hybridge.solvers.advection_reaction`. Its raw-CUDA path keeps reduced trace
assembly, AMGX solve, trace reconstruction, field reconstruction, and error
evaluation on device unless host materialization is explicitly requested.

The diffusion runner is likewise a thin `DiffusionReactionHDGSolver` front end.
It selects CuPy or raw-CUDA assembly and AMGX, while package code owns scaling,
device CSR handoff, reconstruction, error evaluation, timings, and plotting
samples. With CuPy assembly, optional primal HDG postprocessing runs on host
Numba. The raw-CUDA solver supports device-resident flux-only recovery with
`RT_projection` or `l2_closest`; its compatible recovery cache survives scalar
tau-only retries.
See the [recovery cache contract](docs/backends/raw_cuda.md#flux-only-recovery-and-scalar-tau-retries).

[The AMGX guide](configs/amgx/README.md) contains current presets. The
[July 2026 advection solver study](docs/research/solver_studies/advection_reaction_2026_07.md)
retains dated SCC, ILU, upwind-GS, Krylov, and AMGX measurements; it is
research evidence rather than the current support contract.

`check_advection_upwind_scc_host_pyamgx.py` is the broader ordering/solver
comparison harness and also covers Cupyx-only solves. Its
`--cupyx-preconditioner` options are:

```text
none              unpreconditioned Cupyx Krylov
host-ilu-export   build SciPy SuperLU ILU on host, export L/U/permutations to device
cupyx-ilu1        build fill_factor=1 ILU directly with Cupyx on device
```

## Programmatic Use

The reusable formalism is intentionally explicit.  User code builds the same
objects that the command-line runners use, so solver internals can be reused in
new experiments without copying legacy scripts.

The normal data flow is:

```text
DGMesh -> DGSpace -> DGField/VectorDGField -> local HDG assembly
       -> static condensation to a trace system
       -> optional boundary elimination and trace ordering
       -> sparse global trace solve
       -> local reconstruction into DGField/VectorDGField
       -> diagnostics, plotting, transfer, or adaptivity
```

### Reusable Field, Trace, And Diagnostic Operations

Runner-independent postprocessing is part of the package API:

```python
from hybridge import evaluate_scalar_error, solution_field, solution_trace

field = solution_field(result, space)
trace_guess = solution_trace(result, space, reduced=True)
report = evaluate_scalar_error(field, exact, include_samples=False)

mass = field.integral()
minimum, maximum = field.min_max()
difference = space.l2_diff(field, reference_field)
```

`DGSpace.l2_diff(left, right)` accepts any `DGField | Callable` pairing and
evaluates both operands on the receiving space quadrature. Same-mesh fields may
have different polynomial orders or bases. `solution_field` and
`solution_trace` preserve device-backed results when available, so unsteady
applications can feed solver outputs into the next solve without recreating the
old runner-specific extraction helpers. Every solve result also exposes the
solution directly as `result.field`; after a device-resident solve it is a lazy
device-backed field that downloads only when host coefficients are read.
Results own their arrays, so a later solve never overwrites an earlier result.
Reusable solvers warm-start from their previous trace when no `initial_guess`
is passed, and work as context managers that call `close()` on exit. Scalar error reports, field
combinations, trace projections, drift metrics, solver summaries, AMGX config
loading, and plotting comparisons follow the same ownership rule: reusable
numerics live under `hybridge`; scripts select cases and present results.

### Minimal End-to-End Examples

The two examples below use only the base NumPy/SciPy installation. They cover
mesh construction, a DG space, manufactured PDE coefficients, an eliminated
HDG boundary trace, the global solve, local reconstruction, and independent
error/residual checks. Identical executable sources live in `examples/` and
are run by `tests/test_documented_examples.py`.

#### Advection-Reaction

For `u = 1 + x + y`, `beta = (1, 1/2)`, and `r = 2`, the source is
`f = beta . grad(u) + r u = 3/2 + 2u`.

```python
from hybridge import DGSpace, rectangle_mesh, solve_advection_reaction_hdg

mesh = rectangle_mesh(6, 6, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
space = DGSpace(mesh, 2, basis_type="dub_orth")

exact = lambda x, y: 1.0 + x + y
beta_x = lambda x, y: 1.0 + 0.0 * x
beta_y = lambda x, y: 0.5 + 0.0 * y
reaction = lambda x, y: 2.0 + 0.0 * x
source = lambda x, y: 1.5 + 2.0 * exact(x, y)

result = solve_advection_reaction_hdg(
    source,
    (beta_x, beta_y),
    reaction,
    exact,
    space,
    solver="direct",
    preconditioner=None,
    boundary_mode="eliminate",
    verbose=False,
)

error = result.field.l2_error(exact)
linear_solve = result.global_solve_result
assert linear_solve is not None and linear_solve.converged
assert error < 1.0e-10
assert linear_solve.physical_relative_residual_norm < 1.0e-10
```

`result.field` is the reconstructed scalar `DGField`; `result.trace` contains
the full trace coefficients after prescribed boundary values are restored.
`result.timings` separates preparation, assembly, global solve, and
reconstruction costs.

#### Diffusion-Reaction

For identity diffusion, zero reaction, and `u = 1 + x^2 + y^2`, the source is
`f = -Delta u = -4`. The result also contains the conservative flux
`q_h = -grad(u_h)` as a two-component `VectorDGField`.

```python
from hybridge import DGSpace, rectangle_mesh, solve_diffusion_reaction_hdg

mesh = rectangle_mesh(6, 6, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
space = DGSpace(mesh, 2, basis_type="dub_orth")

exact = lambda x, y: 1.0 + x**2 + y**2
reaction = lambda x, y: 0.0 * x
source = lambda x, y: -4.0 + 0.0 * x

result = solve_diffusion_reaction_hdg(
    source,
    reaction,
    exact,
    space,
    diffusion=1.0,
    stabilization=1.0,
    solver="direct",
    preconditioner=None,
    boundary_mode="eliminate",
    verbose=False,
)

error = result.field.l2_error(exact)
linear_solve = result.global_solve_result
assert linear_solve is not None and linear_solve.converged
assert error < 1.0e-10
assert linear_solve.physical_relative_residual_norm < 1.0e-10
```

Use the reusable `AdvectionReactionHDGSolver` and
`DiffusionReactionHDGSolver` classes shown later when coefficients or right-hand
sides change repeatedly. The one-shot functions are the clearest starting
point and remain part of the alpha API.

### Core Objects

A `DGMesh` owns geometry and connectivity.  A `DGSpace` pairs one mesh with a
reference basis and quadrature rule.  A `DGField` stores element-major
coefficients with shape `(num_elements, el_dof)`.  A `VectorDGField` is a tuple
of scalar DG fields on the same mesh.  A `DGTraceSpace` stores the per-edge
trace basis and face-coupling tables used by HDG static condensation.

```python
from hybridge import DGField, DGSpace, gmsh_rectangle_mesh

mesh = gmsh_rectangle_mesh(0.05, verbosity=0)
space = DGSpace(
    mesh,
    4,
    basis_type="dub_orth",
    volume_quad_1d=None,
    edge_quad_1d=None,
)
trace_space = space.trace_space("legacy-lagrange")

u_h = space.project_callable(lambda x, y: x + y, name="u_h")
values = u_h.values()
du_dx, du_dy = u_h.grad_values()
error = u_h.l2_error(lambda x, y: x + y)
```

Prefer `space.project_callable(...)`, `space.field(...)`, and
`(space * space).field(...)` in new code.  The direct `DGField(data, space)`
constructor is also supported for callables and coefficient arrays.

### Coefficient Ownership

Coefficient inputs have two distinct meanings.  A Python callable is an
analytic PDE coefficient; NumPy and CuPy assembly paths may sample it directly
on their quadrature rules.  A `DGField` or `VectorDGField` is a discrete
coefficient in the chosen DG space; when it comes from `project_callable`, it is
the L2 projection of the callable, not the exact callable itself.  Use explicit
projection when repeat solves should reuse the same discrete coefficient or
when a backend requires table data:

```python
from scripts.advection_reaction.cases import test2

beta_x, beta_y, reaction, source, exact = test2()

source_h = space.project_callable(source, name="source_h")
beta_x_h = space.project_callable(beta_x, name="beta_x_h")
beta_y_h = space.project_callable(beta_y, name="beta_y_h")
beta_h = (space * space).field((beta_x_h, beta_y_h), name="beta_h")
reaction_h = space.project_callable(reaction, name="reaction_h")
```

Lazy exact constants should be created with `space.zeros(...)` or
`space.constant(value, ...)`.  These fields carry zero/constant metadata and do
not build a full host coefficient table until `.coeffs` or `.asarray()` is
requested.

Exact coefficient values at the quadrature points are a third form, for the
advection-reaction solver only. `QuadratureVectorField.from_callable(law, space,
trace_space=space.trace_space(kind))` evaluates the advection field at the volume points
and its normal flux at the face points. The normal flux is single valued: each
edge point is evaluated once and its neighbour gets the negated value.
`QuadratureScalar.from_callable(law, space)` does the same for a reaction or a
source; a source enters through its moments. Nothing is projected:

```python
from hybridge import QuadratureVectorField, QuadratureScalar

trace = space.trace_space("legacy-lagrange")
beta_q = QuadratureVectorField.from_callable(lambda x, y: (beta_x(x, y), beta_y(x, y)), space, trace_space=trace)
reaction_q = QuadratureScalar.from_callable(reaction, space)
```

Backend coefficient support is deliberately explicit:

```text
NumPy assembly              callables, DGField/VectorDGField, arrays, lazy constants, quadrature-point values
CuPy assembly               CuPy-compatible callables, DGField/VectorDGField, lazy constants
Numba assembly              DGField/VectorDGField, compact zero/constant descriptors, or quadrature-point values
raw-CUDA advection-reaction projected source, beta, and reaction fields; quadrature-point values with fused/split3
raw-CUDA diffusion          current scalar path uses device source/boundary data and zero reaction
```

For device workflows, `DGField.coeffs` always means host NumPy coefficients.
CuPy-backed fields can keep a cached device table; CuPy backends access that
through backend helpers instead of downloading through `.coeffs`.

### One-Shot Advection-Reaction Solve

```python
from hybridge import solve_advection_reaction_hdg

result = solve_advection_reaction_hdg(
    source,
    (beta_x, beta_y),
    reaction,
    exact,
    space,
    solver="BICGSTAB",
    preconditioner="ilu",
    boundary_mode="penalty",
    verbose=2,
)

print(result.field.l2_error(exact))
print(result.trace.shape)
print(result.timings)
```

Use projected coefficients and the projected Numba backend when projection is
managed outside the solver.  The Numba and raw-CUDA advection-reaction paths
reject Python callables for source, reaction, and advection because their hot
kernels are table driven:

```python
result = solve_advection_reaction_hdg(
    source_h,
    beta_h,
    reaction_h,
    exact,
    space,
    assembly_backend="numba",
    boundary_mode="eliminate",
    trace_ordering="upwind-scc",
    solver="BICGSTAB",
    preconditioner="ilu",
    verbose=2,
)
```

Request legacy-like arrays or matrix-only output with `return_` and
`matrix_pattern_only`:

```python
trace, rows, cols, data = solve_advection_reaction_hdg(
    source,
    (beta_x, beta_y),
    reaction,
    exact,
    space,
    return_=("trace", "matrix_rows", "matrix_cols", "matrix_data"),
    verbose=False,
)

assembled = solve_advection_reaction_hdg(
    source_h,
    beta_h,
    reaction_h,
    exact,
    space,
    assembly_backend="numba",
    boundary_mode="eliminate",
    matrix_pattern_only=True,
)
```

The advection solver supports `trace_basis="legacy-lagrange"` and
`trace_basis="legendre-modal"` across the NumPy, CuPy, Numba, and raw-CUDA
assembly/reconstruction paths.  `trace_basis="bernstein"` remains unwired for
advection. With DG velocity fields or coefficient arrays, the default
`advection_stabilization=None` selects conflict-averaged upwind, including the
raw-CUDA kernels. Two analytic velocity callables keep standard sidewise
upwinding. Select `"upwind"` or `ScaledUpwind(1.)` to request the original
sidewise policy explicitly.

### Stateful Advection-Reaction Solver

Use `AdvectionReactionHDGSolver` for continuation, sweeps, adaptivity, and
benchmarks that need to keep the latest matrices, ordering, preconditioner,
trace, and reconstructed field attached to one object.

```python
from hybridge import AdvectionReactionHDGOptions, AdvectionReactionHDGSolver

options = AdvectionReactionHDGOptions(
    assembly_backend="numba",
    boundary_mode="eliminate",
    trace_ordering="upwind-scc",
    solver="BICGSTAB",
    preconditioner="ilu",
    verbose=2,
)
solver = AdvectionReactionHDGSolver(space, options=options)
solver.set_discrete_problem(source_h, beta_h, reaction_h, exact)
result = solver.solve()

rows = solver.solve_rows
cols = solver.solve_cols
data = solver.solve_data
rhs = solver.solve_rhs
ordering = solver.ordering_result
preconditioner = solver.preconditioner
field = solver.field
```

The class invalidates cached data conservatively.  Updating coefficients clears
assembled matrices, preconditioners, traces, and fields:

```python
solver.set_source(next_source_h)
next_result = solver.solve()
```

Change solver controls in place with `with_options` or per-call overrides:

```python
solver.with_options(solver="GMRES", preconditioner="ilu", maxiter=500)
gmres_result = solver.solve()

matrix_only = solver.assemble_trace_system()
```

For mesh adaptivity, install a new space and then provide coefficient data on
that space:

```python
solver.set_space(new_space)
solver.set_discrete_problem(new_source_h, new_beta_h, new_reaction_h, exact)
adapted_result = solver.solve()
```

### Diffusion-Reaction Solve and Postprocessing

```python
from hybridge import DiffusionReactionHDGSolver, solve_diffusion_reaction_hdg
from scripts.diffusion_reaction.cases import quadratic_poisson_case

diffusion, reaction, source, exact = quadratic_poisson_case()

result = solve_diffusion_reaction_hdg(
    source,
    reaction,
    exact,
    space,
    diffusion=diffusion,
    stabilization=1.0,
    solver="BICGSTAB",
    hdg_postprocess="both",
)

print(result.field.l2_error(exact))
print(result.flux.as_component_first().shape)
print(result.postprocessed_field)
print(result.postprocessed_flux)
```

The stateful diffusion solver can reuse a cached Numba operator and, for Cupyx
repeated RHS-only solves, a cached device CSR matrix when `cache_device_matrix`
is enabled:

```python
diff_solver = DiffusionReactionHDGSolver(
    space,
    source=source,
    reaction=reaction,
    boundary_condition=exact,
    diffusion=1.0,
    stabilization=1.0,
    assembly_backend="numba",
    solver="cupyx_bicgstab",
    cache_device_matrix=True,
    boundary_mode="eliminate",
)
first = diff_solver.solve()

diff_solver.set_source(next_source)
second = diff_solver.solve()
```

### Static Condensation by Hand

`hybridge.hdg.condensation` exposes the reusable formalism beneath the solver
classes.  This is the right level when a user wants to build a new PDE driver
that still uses the package's trace-system conventions.

```python
from hybridge.hdg import condensation as hdg_assembly
from hybridge.linalg.system import solve_global_system

source_rhs = hdg_assembly.source_moments(source, space)
trace_blocks = hdg_assembly.element_to_trace_matrix(
    local_solver,
    element_boundary_mats,
    space,
)
rows, cols = hdg_assembly.trace_matrix_indices(space)
data = hdg_assembly.trace_matrix_data(
    trace_blocks,
    space,
    boundary_penalty=1e20,
)
rhs, boundary_trace = hdg_assembly.global_rhs(
    source_rhs,
    local_solver,
    boundary_condition,
    space,
    1e20,
)

solve_result = solve_global_system(
    rows,
    cols,
    data,
    rhs,
    rhs.size,
    solver="BICGSTAB",
    preconditioner="ilu",
    scale_system=True,
    raise_on_nonconvergence=True,
)
u_h = hdg_assembly.reconstruct_field(
    solve_result.x,
    source_rhs,
    local_solver,
    element_boundary_mats,
    space,
)
```

For callers that already have local solver and element-boundary matrices, the
same trace system can be assembled in one call:

```python
trace_system = hdg_assembly.assemble_trace_system(
    local_solver,
    element_boundary_mats,
    source_rhs,
    boundary_condition,
    space,
)
```

Mixed local systems, such as diffusion-reaction, use block helpers:

```python
source_rhs = hdg_assembly.block_source_moments(source, space, num_blocks=3)
unknowns = hdg_assembly.reconstruct_local_unknowns(
    trace,
    source_rhs,
    local_solver,
    element_boundary_mats,
    space,
)
```

`trace_matrix_indices(..., interior_mass_mode="face")` and
`trace_matrix_data(..., interior_mass_mode="face", interior_mass_blocks=...)`
support operators whose stabilization trace mass is contributed once per
element-side incidence rather than once per global edge.

### Sparse Solves and Cupyx Routes

`hybridge.linalg.system.solve_global_system` is the common global trace solver
entry point.  It can build a SciPy sparse matrix from COO data, apply diagonal
row scaling, eliminate/expand known dofs through helper functions, and route to
SciPy, PyPardiso, PETSc, AMGX, or Cupyx solve paths.

Every completed solve returns the normalized `SolveResult` contract described
in `docs/reference/solver_convergence_contract.md`. A backend success code is accepted
only when the solution and both scaled solver-system and original-system
residual diagnostics are finite and meet their targets. Set
`raise_on_nonconvergence=True` to raise
`LinearSolveConvergenceError`; inspect `exception.result` for the same
normalized status, native `backend_info`, iteration count, and residuals.

Cupyx aliases include `solver="cupyx"`, `solver="cupyx_bicgstab"`,
`solver="cupyx_gmres"`, `solver="cupyx_cg"`, and `solver="cupyx_cgs"`.
The `cupyx_solver` argument is used when `solver="cupyx"` is selected directly.

Supported Cupyx preconditioner strings:

```text
None                         unpreconditioned Cupyx Krylov
host_ilu_export              SciPy SuperLU ILU on host, apply exported L/U on device
host_ilu                     alias for host_ilu_export
ilu                          host export unless ilu_fill_factor is exactly 1.0
cupyx_ilu1 / device_ilu1     Cupyx device-side ILU with fill_factor=1.0
upwind_block_gs              host-built upwind block-GS exported to a Cupyx LinearOperator
```

Use `host_ilu_export` for stronger SuperLU-style ILU with
`ilu_fill_factor > 1`.  Use `cupyx_ilu1` only for the device ILU(1) experiment;
it rejects fill factors other than `1.0`.  `solve_cupyx_csr` returns
`(solution, info, iteration_count)`.  For restarted GMRES, the iteration count
is a callback/restart count, not necessarily every inner Arnoldi step.

### Upwind Ordering and Block-GS Reuse

The upwind-SCC ordering is an edge-block ordering.  For boundary-eliminated
advection-reaction systems, pass only interior/free trace edges as active edges:

```python
import numpy as np
from hybridge.hdg.coefficients import advective_boundary_normal
from hybridge.linalg.ordering import upwind_scc_trace_ordering

beta_dot_normal = advective_boundary_normal(beta_h, space)
active = np.ones(space.mesh.num_edg, dtype=bool)
active[space.mesh.bnd_edges_inds] = False
ordering = upwind_scc_trace_ordering(
    space.mesh,
    beta_dot_normal,
    space.quad_data.edg_dof,
    active_edges=np.flatnonzero(active),
)
```

The CSR-reference preconditioner builder expects the matrix to already be in
the ordered scalar trace layout:

```python
from hybridge.linalg.upwind_block_gs import build_upwind_block_gs_preconditioner

preconditioner = build_upwind_block_gs_preconditioner(
    ordered_scaled_csr,
    block_size=space.quad_data.edg_dof,
    level_widths=ordering.diagnostics.level_widths,
    sweep="forward",
    apply_mode="auto",
)
M = preconditioner.operator
```

The on-the-fly module builds the same `UpwindBlockGSPreconditioner` from matrix
triplets or assembly-emitted block data.  Use scalar COO when the matrix stream
is natural-order scalar triplets:

```python
from hybridge.linalg.upwind_block_gs_on_the_fly import build_forward_upwind_block_gs_from_coo

preconditioner = build_forward_upwind_block_gs_from_coo(
    rows,
    cols,
    data,
    rhs.size,
    dof_permutation=ordering.dof_permutation,
    block_size=space.quad_data.edg_dof,
    level_widths=ordering.diagnostics.level_widths,
)
```

Use ordered block COO when the Numba assembly kernel emits dense trace blocks:

```python
from hybridge.transport.numba import assemble_projected_trace_system_eliminated_numba
from hybridge.linalg.upwind_block_gs_on_the_fly import (
    build_forward_upwind_block_gs_from_ordered_block_coo,
    scale_ordered_trace_coo_from_block_gs,
)

assembly = assemble_projected_trace_system_eliminated_numba(
    source_h,
    beta_h,
    reaction_h,
    exact,
    space,
    edge_order=ordering.edge_order,
    return_block_coo=True,
)
trace_system = assembly.trace_system
num_blocks = trace_system.rhs.size // space.quad_data.edg_dof

preconditioner = build_forward_upwind_block_gs_from_ordered_block_coo(
    assembly.block_rows,
    assembly.block_cols,
    assembly.block_data,
    num_blocks,
    level_widths=ordering.diagnostics.level_widths,
)
scaled_data, scaled_rhs = scale_ordered_trace_coo_from_block_gs(
    trace_system.rows,
    trace_system.data,
    trace_system.rhs,
    preconditioner,
)
```

The ordered block-COO builder computes the same left Jacobi row scale used by
`hybridge.linalg.results.diagonal_scale_system`.  Reuse that scale so the
preconditioner and Cupyx matrix see the same scaled operator.

Export a host-built forward upwind block-GS preconditioner to CuPy with:

```python
from hybridge.linalg.gpu.upwind_block_gs import cupy_upwind_block_gs_from_host_preconditioner

M_cp = cupy_upwind_block_gs_from_host_preconditioner(preconditioner, warm_start=True)
```

The returned object is a Cupyx `LinearOperator`.  Only forward sweeps are
supported on the CuPy export path at the moment.

### Transfer and Adaptivity

Reusable mesh-adaptivity utilities live in `hybridge.core.adaptivity`.  They are
PDE-agnostic helpers: build an indicator, convert it to a native Gmsh structured
background size field, remesh, then transfer fields with `hybridge.core.transfer`.

```python
from hybridge.core import (
    SmoothStarGeometry,
    StructuredSizeOptions,
    gradient_weighted_indicator,
    remesh_smooth_star_from_indicator,
)
from hybridge.core.mesh import mesh_edge_min_max

hmin, hmax = mesh_edge_min_max(space.mesh)
indicator = gradient_weighted_indicator(u_h, hmin, grad_weight=10.0)
new_mesh, info = remesh_smooth_star_from_indicator(
    space,
    indicator,
    geometry=SmoothStarGeometry(),
    hmin=hmin,
    hmax=hmax,
    options=StructuredSizeOptions(size_sensitivity=100.0),
)
new_space = DGSpace(new_mesh, space.order, basis_type="dub_orth")
u_new, plan = u_h.project_to(new_space)
```

Reuse a transfer plan when moving multiple fields between the same source and
target spaces:

```python
plan = new_space.transfer_plan_from(space)
u_new, _ = u_h.project_to(new_space, plan=plan)
reaction_new, _ = reaction_h.project_to(new_space, plan=plan)
```

## Data Model and Assembly Details

### DGMesh

`DGMesh` stores geometry and connectivity needed by HDG assembly:

```text
node_coords       (num_nodes, 2)
triangles         (num_elements, 3)
edges             (num_edges, 2)
loc2glob_edge     (num_elements, 3)
orientations      (num_elements, 3)
aff_mats          (num_elements, 2, 2)
aff_vecs          (num_elements, 2)
aff_jacs          (num_elements,)
normals           (num_elements, 3, 2)
jacs_el_fc        (num_elements, 3)
```

The mesh also caches `loc2oriented_face_coupling`, `interior_elements`,
`interior_faces`, and `edge_jacs`.  Legacy aliases `sigma`, `sigma_1`, and
`eta` remain available, but new code should prefer explicit attribute names.

### ReferenceElementData and DGSpace

`ReferenceElementData` stores quadrature, basis values, and reference tensors.
Important attributes include:

```text
Krf_quads                  (num_volume_quads, 2)
Krf_w                      (num_volume_quads,)
bas_of_quads               (el_dof, num_volume_quads)
dbas_of_quads              (2, el_dof, num_volume_quads)
bas_of_bd_quads            (3, el_dof, num_face_quads)
bas1d_of_ref_edg_qds       (edg_dof, num_face_quads)
MKrf                       (el_dof, el_dof)
MKrf_inv                   (el_dof, el_dof)
face_element_test_trace_trial            (3, el_dof, edg_dof)
face_element_test_trace_trial_reversed   (3, el_dof, edg_dof)
face_trace_test_element_trial_oriented   (6, edg_dof, el_dof)
face_element_test_element_trial          (3, el_dof, el_dof)
weighted_phi               (num_volume_quads, el_dof)
weighted_phi_phi_flat      (num_volume_quads, el_dof * el_dof)
weighted_triple_phi_flat   (el_dof, el_dof * el_dof)
```

Face-coupling table names encode test/trial convention.  For example,
`face_element_test_trace_trial[f, i, a]` couples element test basis `phi_i` to
trace trial basis `mu_a` on local face `f`, while
`face_trace_test_element_trial_oriented[o, a, i]` is the orientation-aware
transpose used for trace-tested assembly.

By default, volume and edge quadrature use `2 * order + 2` one-dimensional
Gauss points.  Override counts through `DGSpace` when needed:

```python
space = DGSpace(mesh, 6, basis_type="dub_orth", volume_quad_1d=7, edge_quad_1d=7)
```

### Local Matrix Assembly

The NumPy reference local matrices are split by owner: generic mass matrices
in `hybridge.core.mass`, advection matrices in `hybridge.transport.local_numpy`,
advective trace weights in `hybridge.hdg.stabilization`, and trace-stabilization
blocks in `hybridge.hdg.matrices`. They keep two API styles.

Return-style reference functions:

```python
from hybridge.core.mass import weighted_mass
from hybridge.transport.local_numpy import advection_mats, boundary_mass

mass = weighted_mass(space, reaction)
adv = advection_mats(space, beta_h)
bd = boundary_mass(space, beta_h)
```

Output-buffer accumulation functions:

```python
from hybridge.core.mass import add_reaction_mass
from hybridge.hdg.matrices import boundary_mass_from_trace_stabilization
from hybridge.hdg.stabilization import advection_trace_weights_from_normal_flux
from hybridge.transport.local_numpy import add_advection_mats

tau_face, gamma_face = advection_trace_weights_from_normal_flux(
    space,
    beta_dot_normal,
    stabilization=None,
)
local = boundary_mass_from_trace_stabilization(space, tau_face)
local = np.ascontiguousarray(local)
scratch = np.empty_like(local)

add_reaction_mass(local, reaction_h, space, scratch=scratch)
add_advection_mats(local, space, beta_h, scale=-1.0)
```

For advection-reaction, `stabilization` is the element-side trace stabilization
`tau`. The public transport default `None` uses conflict-averaged upwind for
DG velocity fields and coefficient arrays, and standard sidewise upwind for
two analytic callables. Explicit `"upwind"` or `ScaledUpwind(1.)` selects
`abs(beta_h . n)` on each side. NumPy and CuPy accept
scalars, callables, `DGField` objects, coefficient arrays, per-face constants,
or already evaluated face-quadrature values. Callables are sampled directly;
DG fields instead use coefficient contractions with the face reference tables
of their own `DGSpace`, with CuPy retaining device-backed coefficients. The
fused Numba path accepts `None`, scalars, or projected same-space
`DGField`/coefficient data and evaluates DG `tau` on face quadrature inside the
kernel.

Raw-CUDA supports the automatic default and explicit upwind, ScaledUpwind,
lax-friedrichs, and conflict-averaged-upwind policies. The complete
backend and boundary interaction is defined in
[the advection boundary and stabilization contract](docs/reference/advection_boundary_stabilization.md).

The solver uses accumulation style so it does not keep three full local element
tensors alive at the same time.  Return-style functions remain useful for tests
and profiling.

### Projected Numba Assembly

`hybridge.transport.numba` adapts package objects to the fused kernels in
`hybridge.transport.numba_kernels`; `hybridge.mixed.numba` does the same for
diffusion with `hybridge.mixed.numba_kernels`. The fused projected trace assembly
path performs local operator build, local solve, and global COO scatter inside
the Numba kernel.

At `--verbosity 2`, Numba assembly timing is split into:

```text
coefficients     shape validation and coefficient normalization
boundary/flux    boundary trace projection plus beta_h . n face samples
kernel           fused local assembly, local solve, and COO scatter
rhs              dense RHS finalization from indexed contributions
```

`boundary/flux` is outside the fused kernel because `beta_h . n` is also reused
by boundary elimination, upwind-SCC ordering, and diagnostics.  When comparing
with legacy scripts, compare the `kernel` entry with the legacy fused assembly
timer; package-level assembly also includes wrapper work needed by solvers.

### Boundary Elimination and Upwind Ordering

The advection-reaction solver supports three boundary modes:

```text
penalty     keep all trace unknowns and impose Dirichlet values with a large diagonal penalty
eliminate   remove known boundary trace dofs before the global solve
zero-flux   solve interior traces only and force exterior numerical fluxes to zero
```

Penalty and elimination require boundary data. Elimination projects the data
into the selected trace basis, solves only for interior edges, and reinserts
the prescribed boundary coefficients before reconstruction. Zero-flux never
samples boundary data; its full-trace boundary slots are zero placeholders and
its boundary lift weights are zero. It is therefore a flux condition, not
homogeneous Dirichlet data. See
[the complete contract](docs/reference/advection_boundary_stabilization.md).

`trace_ordering="upwind-scc"` builds a directed graph from signs of
`beta_h . n`, computes strongly connected components, topologically orders the
component DAG, and converts that order to a trace-dof permutation.  On acyclic
advection-dominated cases this can expose nearly triangular structure to ILU or
upwind block-GS.

For penalty mode the graph contains all trace edges. Elimination and zero-flux
order only active interior edges. Numba supports this reduced SCC ordering;
raw-CUDA requires `trace_ordering="none"`.

Matrix-pattern diagnostics can be generated with `--plot-matrix-pattern`.  Those
images are run artifacts and should generally not be committed unless a specific
documentation change needs them.

### HDG Gram Dual Norms

`hybridge.hdg.gram` builds the Gram matrix associated with the HDG tuple
`(q_x, q_y, u, uhat)`:

```text
sum_K ||q||^2_K + sum_K ||grad u||^2_K
  + sum_{K,F subset dK} jump_weight * ||u - uhat||^2_F
```

Boundary trace degrees of freedom are eliminated, so the trace block is the
interior HDG trace space.  Two inverse-application paths are available:

```python
from hybridge.hdg.gram import (
    assemble_hdg_gram,
    build_condensed_hdg_gram_inverse,
    build_ilu_bicgstab_inverse,
    build_krylov_hdg_gram_inverse,
)

gram = assemble_hdg_gram(space, sigma=10.0, jump_weight="unit")
inverse = build_condensed_hdg_gram_inverse(
    space,
    sigma=10.0,
    jump_weight="unit",
    cg_rtol=1e-8,
    cg_maxiter=200,
)
hminus2, diagnostics = inverse.dual_norm_squared(residual)
```

`build_krylov_hdg_gram_inverse` provides reusable Jacobi, ILU, or unpreconditioned
CG/GMRES/BiCGSTAB applications for an assembled Gram matrix.
`build_condensed_hdg_gram_inverse` never forms or factors the full Gram matrix.
It inverts the local flux/scalar block elementwise and applies CG to the trace
Schur complement with an edge-block Jacobi preconditioner.

Focused checks:

```bash
python -m scripts.dev.check_hdg_gram_matrix --order 2 --nx 2 --ny 2
python -m pytest tests/test_hdg_gram.py
```

## Plotting and Output Interpretation

Plotting helpers are generic over `DGField`:

```python
from hybridge.io.plot import (
    plot_field,
    plot_fields,
    plot_solution_comparison,
    refined_field_polydata,
    sample_field_on_elements,
)

plot_field(result.field, resolution=20, title="u_h")
plot_fields((result.field, result.postprocessed_field), titles=("u_h", "u_star"), share_clim=True)
plot_solution_comparison(result.field, exact)

ref_points, xy, values = sample_field_on_elements(result.field, resolution=16)
poly = refined_field_polydata(result.field, resolution=16, scalar_name="u_h")
```

Small-mesh Matplotlib contour panels duplicate refined vertices per physical
element, so discontinuous DG fields are not averaged across element boundaries.

At `--verbosity 2`, solver output separates preparation, assembly, global solve,
reconstruction, residual diagnostics, and timing percentages.  Summary tables
are grouped into run/mesh, options, solver, errors, and timings sections.
Non-total timing rows include their percentage of total runtime.

The default advection `BICGSTAB` path uses `hybridge.linalg.system.solve_global_system`
with diagonal scaling and ILU.  Explicit sparse zeros are removed before ILU
factorization; this matters for large trace systems.

## Guiding-Center BDF2 Showcase

The small [showcase](scripts/guiding_center/README.md) couples the public Poisson
and transport solvers on the README's star-shaped domain with an island. It
supports balanced vortex gas and positive turbulence using BDF2, with one Euler
startup step. The latter applies the library's density-positivity projector.

```bash
.venv/bin/python -m scripts.guiding_center.showcase
.venv/bin/python -m scripts.guiding_center.showcase --case positive-turbulence
.venv/bin/python -m scripts.guiding_center.showcase --movie run_outputs/vortex-gas.mp4
```

Use `--mesh-size`, `--dt`, `--steps`, `--every`, and `--fps` to adjust a
showcase. `--help` lists options without importing the numerical stack.
The [GPU showcase guide](docs/getting_started/gpu_showcase.md) explains the
problem and the provenance of the existing README recordings.

The guiding-center CG/HDG repository (`../guiding-center-cg-hdg/README.md`) owns
research cases, alternative schemes, convergence campaigns, equilibrium
construction, restart workflows, and their saved data. The general solvers,
field operations, diagnostics, and plotting APIs remain in HYBRIDGE.

## Performance Notes

- Element axis is kept first, so local tensors use shape
  `(num_elements, el_dof, el_dof)`.
- Reference products such as `weighted_phi_phi_flat` and
  `weighted_triple_phi_flat` are precomputed once per reference element.
- Advection assembly caches `beta_h . n` once per solve.  The corrected trace
  assembly forms side-wise `tau` and `tau - beta_h . n` weights, so projected
  discontinuous beta fields do not collapse to an unweighted edge average.
- The NumPy path reduces persistent temporaries and memory pressure, but it is
  not a true fused element kernel.
- The projected Numba advection-reaction backend is the current fused package
  path and is fastest when source, beta, and reaction fields are already
  projected and reused across solves.
- Projection costs are intentionally reported separately from solve time in
  benchmark scripts.  Compare timing scopes carefully.
- Absolute timings should record CPU model/core count, thread count, memory,
  CUDA device, and whether Numba kernels were already JIT compiled.

## Development Checks

Use the executable early-alpha matrix for release work:

```bash
python scripts/dev/alpha_test_matrix.py host-fast
python scripts/dev/alpha_test_matrix.py install-smoke
python scripts/dev/alpha_test_matrix.py cpu-parity
python scripts/dev/alpha_test_matrix.py gpu-smoke
```

`host-fast` runs on every change and does not collect device test modules.
`host-fast` also requires a descriptive docstring on every package function,
method, and Numba kernel through a package-wide AST check.
`install-smoke` builds and imports a wheel outside the source tree for every
release candidate. `cpu-parity` covers both production trace bases.
`gpu-smoke` is opt-in for ordinary development but is
required on the production CUDA/PyAMGX environment before an alpha tag; runtime
skips do not count as GPU evidence.

`scheduled-evidence` preflights the recommended Gmsh runtime and enables its
opt-in geometry parity cases; those cases may not be counted as scheduled skips.

The current package candidate is `0.1.0a2`. On 2026-10-05 it passed 825
`host-fast` tests, the isolated wheel smoke, 136 CPU parity cases, and 272 GPU
smoke cases, plus the hosted Python 3.10/3.12 workflow.
The 2026-08-05 Gmsh-enabled broad suite passed 613 tests with zero skips. The four
focused Gmsh parameters cover 16 geometry/order combinations. The evidence
record retains the stronger dependency-isolated install smoke and wheel/sdist
metadata checks. The first hosted `early-alpha` workflow execution on Python
3.10 and 3.12 remains a separate pre-tag gate and must be linked from the
release evidence record.

The normalized solver status and true-residual acceptance rules are documented
in `docs/reference/solver_convergence_contract.md`.

Inspect the longer release-candidate commands without running them using:

```bash
python scripts/dev/alpha_test_matrix.py scheduled-evidence --dry-run
```

See `docs/development/alpha_test_matrix.md` for exact targets and acceptance rules, and
`docs/releases/early_alpha.md` for current results, warnings, skips, hardware,
and known gaps.

Run the broad repository test suite when changing shared behavior:

```bash
env MPLCONFIGDIR=/tmp python -m pytest tests -q
```

Run syntax checks:

```bash
python -m compileall -q hybridge tests scripts
```

Run CLI smoke tests:

```bash
python -m scripts.advection_reaction.run_cases test2_scipy_ilu_upwind -p 2 --lc 0.30 --quiet
python -m scripts.diffusion_reaction.run_cases quadratic_poisson --dry-run
```

Run focused Gram and solver-class checks:

```bash
python -m pytest tests/test_hdg_gram.py tests/test_advection_reaction_solver.py tests/test_diffusion_reaction_solver.py
```

### Stationary ADR device recovery

Raw-CUDA ADR selects CuPy postprocessing by default for both `l2_closest` and
`RT_projection` total-flux recovery and coupled primal recovery. Set
`materialize_host_solution=False` to keep trace/local-unknown arrays and all
returned field coefficients on device; field `.coeffs` access explicitly
downloads that field. See the [ADR device postprocessing contract](docs/backends/adr_device_postprocessing.md)
for supported combinations, transfer accounting and small-mesh parity evidence.
