# hdgfem

**High-order hybridizable discontinuous Galerkin methods in Python, on CPUs
and NVIDIA GPUs.**

`hdgfem` is a research library for transport, diffusion, and coupled flow
problems on two-dimensional triangular meshes. Build a mesh, define polynomial
fields, and assemble HDG solves through a common Python interface. Reusable
solvers support both individual boundary-value problems and the repeated solves
needed in time-dependent applications.

<!-- showcase-video: vortex_gas -->

https://github.com/user-attachments/assets/e67a723f-f34b-4c06-a7e6-16641e62185d

*960 interacting vortices in a star with a circular island: 360,379 triangles,
degree-6 HDG, and 10.1 million scalar unknowns. Vorticity and potential appear
side by side with fixed colorbars.
Recorded on an NVIDIA RTX PRO 5000 Blackwell using the application below.*

<!-- showcase-video: positive_density -->

https://github.com/user-attachments/assets/2ea0e59d-124b-40c0-89fe-9980ae3d0515

*The same geometry with initially positive density: density and potential evolve
side by side. Negative undershoots can occur with unlimited high-order HDG.
Both animations use the original showcase’s physical playback speed.*

[Get started](#installation) · [Run the example](#a-gpu-vortex-gas-in-python) ·
[User manual](MANUAL.md) · [Documentation](docs/README.md)

## What you can build

HDG represents the solution with discontinuous element polynomials and couples
neighboring elements through a shared trace on their edges. Element-local
unknowns are eliminated before the global solve, then reconstructed from the
trace. This gives high-order local approximations while concentrating the
globally coupled system on the mesh skeleton
([Cockburn et al., 2009](https://doi.org/10.1137/070706616);
[Nguyen et al., 2009](https://doi.org/10.1016/j.jcp.2009.01.030);
[Cockburn et al., 2010](https://doi.org/10.1090/S0025-5718-10-02334-3)).

The library provides three complementary PDE solvers:

| Problem | Typical uses | Python interface |
|---|---|---|
| Advection–reaction | Conservative transport, implicit transport steps | `AdvectionReactionHDGSolver` |
| Diffusion–reaction | Poisson equations, elliptic boundary-value problems | `DiffusionReactionHDGSolver` |
| Advection–diffusion–reaction | Combined transport and diffusion, scalar and tensor diffusion coefficients | `AdvectionDiffusionReactionHDGSolver` |

Use these building blocks directly, or combine them into an application as in
the vortex example. Each solver has a reusable class for repeated solves and a
corresponding `solve_*_hdg` function for one-shot use. Solver results expose the
computed fields, trace, timings, and convergence information, so application
code can inspect the numerical result as well as plot it.

The surrounding tools cover the usual steps of an HDG experiment:

- **Geometry and approximation.** Structured meshes and Gmsh geometries,
  triangular polynomial spaces, scalar and vector DG fields, L2 projection,
  quadrature, and field transfer.
- **High-order recovery.** Mixed formulations reconstruct scalar and flux
  fields locally; supported diffusion and ADR paths also offer degree-`p+1`
  recovery and Raviart–Thomas flux reconstruction.
- **CPU and GPU execution.** NumPy reference implementations, multithreaded
  Numba kernels, and CuPy/raw-CUDA assembly and reconstruction share the solver
  interface. GPU paths can retain the trace and reconstructed fields on the
  device between solves.
- **Diagnostics and visualization.** Field integrals, error norms, physical
  residual checks, scalar plots, live panels, and movie output help connect
  solver behavior to the solution being studied.

Backend support depends on the equation, coefficient representation, polynomial
degree, and requested recovery. The
[capability table](docs/reference/backend_capabilities.md) gives the supported
combinations; the [numerical formulations](docs/algorithms/README.md) explain
their discretizations. Those references are the place to check a new problem
before choosing its execution path.

## Installation

From a checkout, with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mesh,plot]'
```

The base package uses NumPy, SciPy, and Numba. The command above also installs
Gmsh for geometry generation and Matplotlib/PyVista for visualization. Use
`python -m pip install .` for the smaller base installation; the two minimal
host examples below need only that base environment.

Optional runtimes depend on the workflow:

| Workflow | Runtime and setup |
|---|---|
| Large CPU direct solves | [PyPardiso / oneMKL](MANUAL.md#pypardiso-host-direct-solver) |
| GPU solves | [CuPy and the supported AMGX/PyAMGX forks](docs/getting_started/forked_amgx_stack.md) |
| Live GPU visualization | [NVIDIA Holoviz](docs/backends/holoviz.md) |
| PETSc solves | [PETSc and petsc4py](MANUAL.md#petsc) |

For AMGX-based GPU solves, we maintain our own forks of
[NVIDIA AMGX](https://github.com/adelsaleh/AMGX/tree/hdg-cuda13-integration) and
[PyAMGX](https://github.com/adelsaleh/pyamgx/tree/quality-of-life), with extensions
for HDG block systems and GPU diagnostics. The
[fork setup guide](docs/getting_started/forked_amgx_stack.md) lists the supported
branches, revisions, and installation steps.

The GPU example also uses Gmsh, CuPy, and Holoviz. Importing `hdgfem` does not
require these optional runtimes. The
[installation guide](docs/getting_started/installation.md) covers all dependency
groups, editable development installs, and wheel contents.


### Connect the GPU backends

On Linux, activate the same virtual environment used for `hdgfem`. After
building our AMGX fork as described in the [setup guide](docs/getting_started/forked_amgx_stack.md),
replace the paths below with your CUDA toolkit, AMGX source/build directories,
and PyAMGX checkout. This example uses CUDA 13:

```bash
export CUDA_PATH=/path/to/cuda-13
export AMGX_DIR=/path/to/AMGX                    # Fork source, including headers.
export AMGX_BUILD_DIR=/path/to/AMGX-build        # Build tree searched by PyAMGX.
export AMGX_LIB_DIR=/path/to/AMGX-build          # Directory containing libamgxsh.so.
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$AMGX_LIB_DIR:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pip install 'cupy-cuda13x>=14,<15' cython setuptools wheel
python -m pip install --no-build-isolation --no-deps /path/to/pyamgx
python -c "import hdgfem, cupy, pyamgx; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s)')"
```

CuPy and PyAMGX are installed into the active Python environment; PyAMGX links
to AMGX at installation, and `LD_LIBRARY_PATH` exposes its shared library at
runtime. `hdgfem` discovers these backends through Python imports, so no separate
link command is needed. Use a [CuPy wheel matching your CUDA toolkit](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi)
and reinstall PyAMGX if you change the AMGX library or ABI. Select the backend
in solver options, as the example below shows.

## A GPU vortex gas in Python

The animation follows a collection of positive and negative vortices in a
closed, star-shaped domain. The transported density is signed vorticity in
this example. A Poisson solve recovers its potential, and an advection solve
advances the density. The same coupling appears in the two-dimensional
guiding-center model.

**Continuous model.** Solve Poisson for the potential, then rotate its gradient
to obtain the transport velocity:

$$
\begin{aligned}
\partial_t\rho+\nabla\!\cdot(\rho\mathbf{u})&=0,\quad
-\Delta\phi=\rho,\quad \mathbf{u}=(\partial_y\phi,-\partial_x\phi),\\
\phi|_{\Gamma}&=0,\quad \mathbf{u}\!\cdot\mathbf{n}|_{\Gamma}=0,\quad
\rho(0)=\rho_0,\quad \Gamma=\Gamma_{\mathrm{outer}}\cup\Gamma_{\mathrm{island}}.
\end{aligned}
$$

**Time discretization.** Advance density with extrapolated velocity, then solve
Poisson again to recover the new velocity (Euler startup, then BDF2):

$$
\begin{aligned}
\frac{3\rho^{n+1}-4\rho^n+\rho^{n-1}}{2\Delta t}
+\nabla\!\cdot(\rho^{n+1}\mathbf{u}^{*,n+1})&=0,\quad
\mathbf{u}^{*,n+1}=2\mathbf{u}^n-\mathbf{u}^{n-1},\\
-\Delta\phi^{n+1}&=\rho^{n+1},\quad \phi^{n+1}|_{\Gamma}=0,\quad
\mathbf{u}^{n+1}=(\partial_y\phi^{n+1},-\partial_x\phi^{n+1}),\\
\mathbf{q}_h^{n+1}&\approx-\nabla\phi_h^{n+1},\quad
\mathbf{u}_h^{n+1}=(-q_{h,y}^{n+1},q_{h,x}^{n+1}).
\end{aligned}
$$

The last row shows velocity recovery from the reconstructed HDG Poisson flux.

The initial field contains equal numbers of
positive and negative Gaussian vortices with several core sizes. Their
positions are sampled inside the fluid mesh with clearance from both walls.
The fixed random seed makes the initial condition reproducible.

The complete example below shows the mesh, initial condition, two solver
objects, and time loop. A semi-implicit BDF2 step uses the previous fields to
form the transport source and extrapolated velocity, with an Euler step at
startup. The Poisson operator is reused as the source changes. The application
coupling stays visible in the script, while assembly, linear solves,
reconstruction, and rendering belong to the library.

```python
from contextlib import closing
import hdgfem as hdg
from hdgfem.io import HolovizScalarPanels

# Mesh and degree-6 approximation.
mesh = hdg.gmsh_smooth_star_mesh(
    0.005, radius=1., amplitude=.35, mode=5,              # Decrease h to refine.
    hole_radius=.3, boundary_points=500, num_threads=16)  # Circular island.
space = hdg.DGSpace(mesh, 6, basis_type="dub_orth")

# Multiscale vortices: the counts and widths control the initial structure.
# For positive density, set strength_mode="positive" and dt=0.0015625.
# Unlimited high-order HDG can produce negative undershoots.
initial = hdg.sample_gaussian_blob_field(
    hdg.MeshDomain(mesh),                               # Respect both walls.
    (512, 256, 128, 64), (.008, .016, .024, .032),        # Counts and widths.
    seed=17, strength_mode="balanced")                  # Reproducible profile.
rho = hdg.project_callable(initial, space, backend="device")

# Reuse the Poisson operator; rebuild transport as the velocity changes.
gpu = dict(assembly_backend="raw-cuda", raw_matrix_format="bsr",
           solver_rtol=1e-9, solver_atol=1e-10, verbose=False)
poisson = hdg.DiffusionReactionHDGSolver(
    space, source=rho, reaction=space.zeros(),
    boundary_condition=0.,                             # Zero wall potential.
    solver="fb-hp-mg-pcg", trace_basis="legendre-modal", scale_system=False,
    stabilization=1000., boundary_mode="eliminate",
    cache_local_factors="schur-cholesky", **gpu)
transport = hdg.AdvectionReactionHDGSolver(
    space, solver="amgx", boundary_mode="zero-flux",     # Impermeable walls.
    raw_local_assembly="fused", materialize_host_solution=False,
    scale_system=True, **gpu)

dt, steps = 0.00625, 1200                               # Final time: steps*dt.
previous_rho = previous_velocity = trace = None         # Euler startup, then BDF2.

# Fixed scales for the live GPU view; recordings use Matplotlib colorbars.
# For positive density, use limits=((-5.75, 23.), (0., .25)).
limits = ((-18., 18.), (-.066, .066))                    # Density, potential.
with closing(poisson), closing(transport), HolovizScalarPanels(
    (space, space), ("Vorticity / density", "Potential"), cmap="RdBu_r",
    width=640, height=640, show_mesh=False) as plot:
    potential = poisson.solve()                        # Recover phi from rho.
    velocity = hdg.perpendicular_vector_field(          # u=(-q_y,q_x), q=-grad(phi).
        potential.flux, 1., space)
    plot.update_fields((rho, potential.field), limits=limits)

    for step in range(1, steps + 1):
        source, beta, _ = hdg.bdf2_transport_data(       # History and velocity extrapolation.
            rho, velocity, dt,
            previous_field=previous_rho, previous_velocity=previous_velocity)
        result = transport.solve(
            source=source, beta=beta, reaction=space.constant(1.),
            initial_guess=trace)                       # Reuse the transport trace.

        next_rho = hdg.solution_field(result, space).copy()  # Own reused-buffer data.
        potential = poisson.set_source(next_rho).solve(
            initial_guess=hdg.solution_trace(potential, space))

        # Commit history only after both solves succeed.
        previous_rho, previous_velocity, rho = rho, velocity, next_rho
        velocity = hdg.perpendicular_vector_field(potential.flux, 1., space)
        trace = hdg.solution_trace(result, space).copy()
        if step % 2 == 0:                               # Draw every other step.
            plot.update_fields((rho, potential.field), step=step,
                               time_value=step * dt, limits=limits)
```

Run the same source from the repository root:

```bash
python examples/gpu_vortex_gas.py
```

The live Holoviz view keeps fields on the GPU and displays density and potential
with independent fixed scales. The recording uses Matplotlib panels with
colorbars; only sampled scalar images are downloaded for drawing. The
[commented example](examples/gpu_vortex_gas.py) explains the main settings,
solver reuse, and history updates.

The [reproduction guide](docs/getting_started/gpu_showcase.md) records the mesh,
time step, displayed physical interval, hardware, measured runtime, numerical
checks, and command used to make the GIF. It also explains the recording
settings. The README animation is an illustration of a particular run;
resolution and time-step checks remain part of setting up a new experiment.

For a first run without a GPU, start with either of the small host examples:

```bash
python examples/advection_reaction_minimal.py
python examples/diffusion_reaction_minimal.py
```

Both construct a structured mesh, define a manufactured solution, solve the
HDG problem, and print an L2 error and physical relative residual. Their short
source files are useful starting points for changing coefficients, boundary
data, or polynomial degree. The
[annotated examples](MANUAL.md#minimal-end-to-end-examples) explain each step.

## Explore further

The package is the reusable library; `examples/` contains small Python
applications, and `scripts/` contains fuller case runners and research studies.
The latter include manufactured solutions, convergence studies, guiding-center
time integration, and GPU solver comparisons. Examples and runners are used
from a checkout; the installed wheel contains the library.

| Starting point | What you will find |
|---|---|
| [Examples](examples/README.md) | Small complete programs, including the GPU vortex gas |
| [User manual](MANUAL.md) | Solver construction, coefficients, boundary conditions, plotting, and runner commands |
| [Solver API](docs/reference/solver_api_alpha.md) | Public imports, reusable-solver updates, result objects, and failure behavior |
| [Coefficient inputs](docs/reference/coefficient_inputs.md) | Constants, callables, projected fields, and coefficient representations |
| [Backend capabilities](docs/reference/backend_capabilities.md) | Supported assembly, solve, recovery, and host/device combinations |
| [Numerical methods](docs/algorithms/README.md) | HDG formulations, stabilization, postprocessing, quadrature, and multigrid |
| [Research studies](docs/research/README.md) | Dated application and solver investigations with their supporting evidence |

For steady problems, the advection, diffusion, and combined ADR runners expose
manufactured cases and named configurations. Their `--list-presets` and
`--dry-run` options let you inspect a configuration before executing it. Start
with the manual for the equation of interest; backend-specific guides contain
the detailed tuning and implementation notes.

For unsteady problems, retain the solver objects between steps and update their
problem data. The guiding-center applications demonstrate this pattern with
Poisson and transport solves, while the example above keeps the essential
coupling short enough to read in one place. Field operations, plotting, and
diagnostics are available independently of those runners.

A small set of optional diocotron-equilibrium studies also uses DOLFINx as a
continuous-Galerkin comparison. DOLFINx is separate from the HDG library; see
the [manual](MANUAL.md#optional-guiding-center-and-diocotron-equilibria) for that
research workflow.

## Development status

`hdgfem` is an **early-alpha research package**. The public solver API and
backend support are documented, while numerical methods, performance paths,
and application studies continue to evolve. The
[roadmap](TODO.md) tracks current work; the
[release record](docs/releases/early_alpha.md) distinguishes qualified evidence
from open gaps.

To install the test dependencies and run the bounded host checks:

```bash
python -m pip install -e '.[test]'
python scripts/dev/alpha_test_matrix.py host-fast
```

Additional installation, CPU-parity, and GPU lanes are described in the
[validation guide](docs/development/alpha_test_matrix.md). Linear solves are
accepted using backend-neutral checks that include the residual of the
original, unscaled system; the
[convergence contract](docs/reference/solver_convergence_contract.md) describes
what success and failure mean. Numerical accuracy still depends on the mesh,
polynomial degree, stabilization, and time discretization chosen for the
problem.


## Scientific references

Foundational HDG work by Cockburn and collaborators:

- **Hybridization:** B. Cockburn, J. Gopalakrishnan, and R. Lazarov (2009).
  [Unified Hybridization of Discontinuous Galerkin, Mixed, and Continuous Galerkin Methods for Second Order Elliptic Problems](https://doi.org/10.1137/070706616).
  *SIAM Journal on Numerical Analysis* **47**(2), 1319–1365.
- **Convection–diffusion:** N. C. Nguyen, J. Peraire, and B. Cockburn (2009).
  [An implicit high-order hybridizable discontinuous Galerkin method for linear convection–diffusion equations](https://doi.org/10.1016/j.jcp.2009.01.030).
  *Journal of Computational Physics* **228**(9), 3232–3254.
- **Error analysis:** B. Cockburn, J. Gopalakrishnan, and F.-J. Sayas (2010).
  [A projection-based error analysis of HDG methods](https://doi.org/10.1090/S0025-5718-10-02334-3).
  *Mathematics of Computation* **79**(271), 1351–1367.

## Acknowledgements

The GPU backend builds on the work of these upstream projects:

- [CuPy](https://github.com/cupy/cupy), developed by Preferred Networks and
  community contributors, provides GPU arrays and CUDA kernel integration.
- [NVIDIA AMGX](https://github.com/NVIDIA/AMGX), developed by NVIDIA and its
  contributors, provides GPU sparse linear solvers and algebraic multigrid.
- [PyAMGX](https://github.com/shwina/pyamgx), created by Ashwin Srinath and
  extended by contributors, provides the Python bindings to AMGX.

Our AMGX and PyAMGX versions are downstream forks of these projects, with
additional support for HDG workloads.
