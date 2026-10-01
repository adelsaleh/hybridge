# hdgfem

`hdgfem` is a discontinuous Galerkin / hybridizable discontinuous Galerkin
research codebase.  The package provides mesh, reference-element, DG field,
assembly, linear algebra, solver, plotting, and optional GPU backend modules for
HDG experiments.

The main package workflows are advection-reaction, diffusion-reaction,
stationary advection-diffusion-reaction, and fixed-mesh guiding-center HDG
solves. Current performance work focuses on raw-CUDA CSR/BSR trace assembly,
face-block Poisson preconditioners, mesh-independent diffusion stabilization,
reusable solvers for unsteady runs, and high-order guiding-center/diocotron
benchmarks.

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

> **Post-release dependency requirement:** every HDG commit after the
> `v0.1.0a1` release tag is developed and qualified with
> [`adelsaleh/AMGX@hdg-cuda13-integration`](https://github.com/adelsaleh/AMGX/tree/hdg-cuda13-integration)
> and
> [`adelsaleh/pyamgx@quality-of-life`](https://github.com/adelsaleh/pyamgx/tree/quality-of-life),
> not the corresponding upstream `main` branches. The exact revisions qualified
> with this checkout are AMGX `583084b` and PyAMGX `81efd1e`. Host-only paths
> keep their lazy optional imports, but the supported post-release development
> stack uses these forks. See the
> [forked AMGX stack guide](docs/getting_started/forked_amgx_stack.md) for clone,
> build, install, runtime-library, and verification commands.

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

The `test` extra includes pytest, Matplotlib for exercised plotting paths, and
the TOML compatibility dependency required by the Python 3.10 packaging tests.

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
error and physical residual. Reusable post-solve operations such as
`evaluate_scalar_error`, `solution_field`, `solution_trace`, `DGSpace.l2_diff`,
`DGField.integral`, and `DGField.min_max` also live in the package rather than
in runner scripts. See [the minimal examples](MANUAL.md#minimal-end-to-end-examples)
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

Use `--flux-postprocess-space l2_closest|RT_projection` to choose the
pure-diffusion flux recovery and
`--postprocessing-backend auto|numba|cupy` to select the supported
execution path. The full-space minimum-distance recovery is host Numba; the RT
moment projection is available through host Numba and batched CuPy.

Stationary combined advection-diffusion-reaction is available through
`AdvectionDiffusionReactionHDGSolver` and
`solve_advection_diffusion_reaction_hdg`. NumPy is the dense reference, Numba is
the default multithreaded host assembly, and raw CUDA provides device assembly,
AMGX solve, and device reconstruction for positive constant scalar diffusion.
Whole-boundary Dirichlet elimination, upwind advection stabilization, the
mesh-independent default `tau_d=kappa/L_Omega`, and total-flux-first
degree-`p+1` ADR primal/flux postprocessing are included. See the
[maintained derivation](docs/algorithms/advection_diffusion_reaction/README.md)
and [implementation report](docs/research/solver_studies/stationary_adr_hdg_2026_08.md).

Run the steady constant-diffusivity disk manufactured case at the mildly
advection-dominated default `Pe=10` with its degree-`p+1` comparison plot:

```bash
python -m scripts.advection_diffusion_reaction.manufactured_disk --plot
```

The runner defaults to multithreaded Numba assembly/reconstruction, host Numba
postprocessing, and nonsymmetric oneMKL PARDISO. Use `--no-plot` for a
diagnostics-only run; `-v 0|1|2` selects summary-only, phase, or detailed
logging. `--assembly-backend`, `--reconstruction-backend`, and
`--postprocessing-backend` expose the currently supported stage paths. Use
`--flux-postprocess-space rt-p` to test the experimental
`RT_p=[P_p]^2+x P_p` total-flux reconstruction; it supports host Numba and
batched CuPy moment solves, while the coupled primal recovery remains host
Numba. Diffusion stabilization defaults to
`GlobalLengthDiffusion(gamma_d=1, domain_length=1)` for this known unit disk,
namely `tau_d=kappa`. Generic solvers use the mesh-derived
`L_Omega=2*area/boundary_length` fallback when no physical length is supplied.
Use
`--diffusion-stabilization-mode inverse-h` only for legacy comparisons, or
`--diffusion-stabilization VALUE` for an explicit constant.

The current Cupyx/upwind-GS advection performance path is:

```bash
python scripts/advection_reaction/run_upwind_gs_cupyx.py --help
python scripts/advection_reaction/experiments/check_upwind_block_gs_on_the_fly.py --help
```

The first script is a thin front end to `AdvectionReactionHDGSolver`: it selects
Numba eliminated assembly, upwind-SCC trace ordering, the package-owned forward
upwind block-GS preconditioner, and a Cupyx Krylov solve. The experimental
checker remains available for low-level parity and performance comparisons of
the CSR-reference, scalar-COO, and ordered block-COO builders.

Standalone GPU benchmark runners live under `scripts/gpu/`:

```bash
python -m scripts.gpu.run_advection_reaction_cuda --help
python -m scripts.gpu.run_diffusion_reaction_cuda --help
python -m scripts.gpu.sweep_cuda_hdg --help
python -m scripts.gpu.check_advection_upwind_scc_host_pyamgx --help
```

The advection GPU runner supports CuPy assembly, raw-CUDA fused assembly, direct
raw-CUDA CSR emission, Cupyx solver experiments, and AMGX solves through
PyAMGX. The diffusion GPU runner is a thin `DiffusionReactionHDGSolver` front
end for CuPy or raw-CUDA assembly with AMGX; the solver class owns assembly,
scaling, device CSR handoff, reconstruction, diagnostics, and optional primal
postprocessing, which runs on host Numba after CuPy assembly. It defaults to `tau_d=kappa/L_Omega`; use
`--tau VALUE` for an explicit constant. Raw-CUDA diffusion uses direct CSR
and currently omits
solver-call HDG postprocessing. Public raw-CUDA solver and runner defaults use
the equation- and
order-aware `raw_block_size="auto"` policy documented in
[the raw-CUDA backend guide](docs/backends/raw_cuda.md); explicit launch sizes
remain available for benchmark reproduction. See
[CUDA execution paths](docs/backends/cuda_execution.md) and
[the AMGX configuration guide](configs/amgx/README.md) for operational details
and current presets.

Three experimental block-structured Poisson paths are retained for research
and matched benchmarking. Direct raw-CUDA BSR assembly can feed the modified
local AMGX classical hierarchy in either coefficient-exact hybrid
fine-BSR/scalar-hierarchy mode or pure block-graph mode. The independent
`FB-HP-MG-PCG` prototype transforms traces to normalized Legendre modes,
p-coarsens dense face blocks to the constant mode, and uses scalar AMGX only
for the reduced p=0 correction; ordinary BSR SpMV is cuSPARSE-backed. Finally,
the face-dense CuPy solver applies restarted GMRES with block-Jacobi or
element-patch ASM and optional polynomial preconditioning without calling
AMGX. These are benchmark paths, not supported backend-matrix rows. See the
[AMGX/BSR ownership map](docs/backends/bsr_amgx_dependency_map.md),
[face-block hp-MG plan](docs/development/plans/face_block_hp_multigrid.md), and
[face-dense GPU guide](docs/backends/face_dense_gpu.md). Reproduction entry
points are:

```bash
python -m scripts.diffusion_reaction.compare_cuda_bsr_csr --help
python -m scripts.diffusion_reaction.face_block_hp_mg_prototype --help
python -m scripts.diffusion_reaction.validate_face_dense_gpu_solver --help
python -m scripts.diffusion_reaction.benchmark_face_dense_primitives --help
```

Fixed-mesh guiding-center cases live under `scripts/guiding_center/`:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH   .venv/bin/python scripts/guiding_center/run_guiding_center_cases.py   --preset diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx   --num-steps 10 --plot-every 0
```

The runner couples diffusion-reaction Poisson solves with advection-reaction
transport and supports `--time-scheme si-euler|predictor-corrector`. It accepts
independent Poisson/transport backend and solver choices, writes CSV/JSONL
diagnostics, updates PyVista scalar arrays in place, and plots density only by
default. Add `--plot-both` to show density and potential. When `DISPLAY` is
unavailable, plotting automatically uses render-only EGL/off-screen updates and
writes frames below `<diagnostics-dir>/<diagnostics-prefix>_frames/`; this avoids
polling an X interactor from a headless render window. Runner level `-v 3` prints
a compact native AMGX residual/memory table plus balanced transport and Poisson
stage summaries. Backend micro-timing dumps remain available to direct solver
calls at `verbose=2` or `verbose>=4`. Raw-CUDA stages pass projected/current
trace guesses directly on device;
`--transport-retry-policy
amgx-robust` retries a failed assembled transport system without host CSR
materialization. The primary BICGSTAB solve and robust FGMRES/DILU retries all
use the configured transport row scaling, retain device trace guesses, and are
accepted only after checking the original unscaled physical residual. Compare
both temporal schemes with
`scripts/guiding_center/benchmarks/run_guiding_center_temporal_convergence.py`.

Optional semilinear diocotron-equilibrium scripts live under
`scripts/diocotron_hdg/` and `scripts/diocotron_dolfinx/`.  They are documented
in `MANUAL.md` and the Strategy A notes under `docs/research/strategy_a_band_parameter_study/`.

## Package Map

Subpackages form a one-way layering, checked with zero allowed violations by
`tests/test_package_layering.py`:

```text
runtime → core → cases → linalg → hdg → {transport, mixed} → solvers → diagnostics → io
```

A module imports only from its own layer or from layers to its left;
`transport` and `mixed` never import each other. The package root is the
public facade over all layers. Module ownership per backend and the naming
rules are in [docs/backends/README.md](docs/backends/README.md).

- `hdgfem/runtime/`: optional-dependency gates and Numba fallbacks, precision,
  logging/timing, error types, CUDA device inventory, host threads.
- `hdgfem/core/`: meshes, bases, quadrature, DG spaces/fields, transfer,
  adaptivity, mass matrices, L2 projection, and CuPy mirrors (`core/device.py`).
- `hdgfem/cases/`: analytic coefficient sets and initial profiles for
  manufactured and stress cases.
- `hdgfem/linalg/`: global solve dispatch and results, dof reduction, direct and
  iterative solves, orderings, preconditioners; `amgx/`, `gpu/`, `multigrid/`.
- `hdgfem/hdg/`: equation-independent HDG layer: condensation, coefficient
  sampling, advection stabilization, trace maps, Gram operators, `cuda/` sources.
- `hdgfem/transport/`: first-order HDG (advection-reaction) local matrices and
  NumPy, Numba, CuPy, and raw-CUDA assembly/reconstruction.
- `hdgfem/mixed/`: mixed HDG for diffusion-reaction (ADR with β = 0) and ADR, all
  backends side by side, plus degree-`p+1` recovery in `mixed/postprocess/`.
- `hdgfem/solvers/`: public AR/DR/ADR solver APIs, the backend capability
  contract, device pipelines, and the `adv_rea`/`diff_rea` compatibility shims.
- `hdgfem/diagnostics/`: solution-error, solver-result, and guiding-center
  diagnostics, re-exported from `hdgfem.diagnostics`.
- `hdgfem/io/`: plotting, comparison figures, rasters, Holoviz panels, movies,
  and JSONL/CSV records.
- `scripts/`: command-line runners, benchmarks, diagnostics, and experiments.
- `tests/`: focused regression tests.

Optional dependencies are imported lazily: importing `hdgfem` must not require
CUDA, AMGX, PETSc, PARDISO, Gmsh, or DOLFINx.

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

The host lane also requires a descriptive docstring on every package function,
method, and Numba kernel through a package-wide AST check.

The scheduled lane requires the recommended Gmsh runtime, preflights its import,
and enables the opt-in diffusion geometry parity cases instead of silently
recording them as skips.

The current package candidate is `0.1.0a1`. On 2026-08-05 the candidate passed
all four local release lanes: 495 host tests, the isolated wheel smoke, 14 CPU
parity cases, and 10 GPU smoke cases. The earlier Gmsh-enabled broad repository
suite passed 613 tests with zero skips, and the four focused Gmsh parameters
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
2. **Backend consolidation:** keep the family/stage/backend layout and its
   layering rule, and split the large solver modules into per-backend drivers
   without reintroducing numeric or equation-abbreviated filenames.
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

- [docs/reference/advection_boundary_stabilization.md](docs/reference/advection_boundary_stabilization.md): boundary modes, stabilization inputs, active trace ownership, ordering, and reconstruction semantics.
- [docs/reference/solver_api_alpha.md](docs/reference/solver_api_alpha.md): bounded early-alpha public solver API and compatibility contract.
- [docs/reference/solver_convergence_contract.md](docs/reference/solver_convergence_contract.md): normalized status, residual acceptance, retry, and cleanup contract.
- [docs/getting_started/installation.md](docs/getting_started/installation.md): package dependency groups, wheel scope, install smoke, and CI qualification.
- [docs/getting_started/forked_amgx_stack.md](docs/getting_started/forked_amgx_stack.md): required post-release AMGX/PyAMGX forks, exact qualified revisions, and build/install instructions.
- [docs/reference/backend_capabilities.md](docs/reference/backend_capabilities.md): authoritative early-alpha backend and residency matrix.
- [docs/development/plans/](docs/development/plans/): indexed active implementation and qualification plans; task priority remains in `TODO.md`.
- [docs/development/alpha_test_matrix.md](docs/development/alpha_test_matrix.md): executable host, CPU parity, GPU smoke, and scheduled validation matrix.
- [docs/releases/early_alpha.md](docs/releases/early_alpha.md): current release evidence, reviewed skips, and known gaps.
- [docs/backends/README.md](docs/backends/README.md): backend role map, module ownership, and naming rules.
- [docs/backends/cuda_execution.md](docs/backends/cuda_execution.md): CUDA assembly, matrix handoff, runner, and direct-CSR notes.
- [docs/backends/raw_cuda.md](docs/backends/raw_cuda.md): raw-CUDA kernel ownership, launch policy, and parity audits.
- [docs/algorithms/advection_reaction/](docs/algorithms/advection_reaction/): upwind flux, SCC ordering, and block-GS derivations.
- [docs/algorithms/diffusion_reaction/](docs/algorithms/diffusion_reaction/): mixed HDG assembly and postprocessing derivations.
- [docs/algorithms/hp_amg/](docs/algorithms/hp_amg/): shared HDG multigrid formalism, native pMG-AMG, and proposed geometric/algebraic hp hierarchies.
- [docs/algorithms/quadrature/](docs/algorithms/quadrature/): symmetric triangle quadrature derivation and validation requirements.
- [docs/research/solver_studies/](docs/research/solver_studies/): dated solver and preconditioner evidence, not current support contracts.
- [configs/amgx/README.md](configs/amgx/README.md): AMGX/PyAMGX presets and recommendations.
- [TODO.md](TODO.md): current GPU, upwind-GS, solver API, and backend cleanup roadmap.
