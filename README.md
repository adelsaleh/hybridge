# HYBRIDGE

**HYBRIdizable Discontinuous Galerkin Environment: HDG methods in Python, on
CPUs and NVIDIA GPUs.**

[Install](#installation) · [First solve](#a-first-solve-on-the-cpu) ·
[GPU example](#a-gpu-vortex-gas-in-python) · [User manual](MANUAL.md) ·
[Documentation](docs/README.md)

HYBRIDGE aims to make hybridizable discontinuous Galerkin (HDG) methods
([Cockburn et al., 2009](https://doi.org/10.1137/070706616);
[Nguyen et al., 2009](https://doi.org/10.1016/j.jcp.2009.01.030);
[Cockburn et al., 2010](https://doi.org/10.1090/S0025-5718-10-02334-3))
accessible to newcomers, in the spirit of FreeFEM and DOLFINx. Build a mesh,
define polynomial fields, and assemble and solve transport, diffusion, and
coupled advection–diffusion–reaction problems on two-dimensional triangular
meshes through one Python interface, on the CPU or entirely on the GPU.
Reusable solvers serve single boundary-value problems and the repeated solves
of time-dependent applications. The import package is `hybridge`.

HYBRIDGE is the companion HDG project of SOLEDGE-HDG, the high-order HDG code
for tokamak edge plasmas in realistic geometry
([Giorgiani et al., 2018](https://doi.org/10.1016/j.jcp.2018.07.028)). GPU
assembly and solver techniques developed here are meant to be transferred to
SOLEDGE-HDG, first in 2D and eventually in 3D. Guiding-center plasma dynamics,
shown below, is one application of the library.

<!-- showcase-video: vortex_gas -->

https://github.com/user-attachments/assets/d3e1d030-26db-4acd-a6b6-e718dd7f477b

<div>

*Positive and negative charge drifting between grounded walls: the plasma form
of a two-dimensional vortex gas, from t = 0 to 15.9; click the image to play the
video. Degree-6 HDG on 360,379 triangles carries 10.1 million element unknowns
per field and couples 3.77 million trace unknowns in each of its two solves per
step, at 0.74 s per time step, including rendering, on one NVIDIA RTX PRO 5000
Blackwell. The total charge changes by 2.2e-13 and the energy drifts by 2.9e-4,
while 29.9% of the enstrophy is dissipated as filaments reach the grid.*

</div>

## Installation

From a checkout, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mesh,plot]'
```

The base package needs NumPy, SciPy, and Numba; `python -m pip install .`
installs only those, which is enough for the [first solve](#a-first-solve-on-the-cpu).
The command above adds Gmsh for geometry and Matplotlib/PyVista for plotting.
Optional runtimes depend on the workflow:

| Workflow | Runtime and setup |
|---|---|
| Large CPU direct solves | [PyPardiso / oneMKL](MANUAL.md#pypardiso-host-direct-solver) |
| GPU solves | Linux, CUDA 13, CuPy, and the [forked AMGX/PyAMGX stack](docs/getting_started/forked_amgx_stack.md), connected as in the [GPU runtime guide](docs/getting_started/installation.md#gpu-runtime) |
| Live GPU visualization | [NVIDIA Holoviz](docs/backends/holoviz.md) (`holoviz` extra) |
| PETSc solves | [PETSc and petsc4py](MANUAL.md#petsc) |

The GPU solves use our forks of [NVIDIA AMGX](https://github.com/adelsaleh/AMGX/tree/hdg-cuda13-integration)
and [PyAMGX](https://github.com/adelsaleh/pyamgx/tree/quality-of-life), which
add HDG block systems and GPU diagnostics. The
[fork setup guide](docs/getting_started/forked_amgx_stack.md) lists the tested
revisions and build steps. Importing `hybridge` needs none of the optional
runtimes; the [installation guide](docs/getting_started/installation.md) covers
every dependency group.

## What you can build

The library provides three complementary PDE solvers:

| Problem | Typical uses | Python interface |
|---|---|---|
| Advection–reaction | Conservative transport, implicit transport steps | `AdvectionReactionHDGSolver` |
| Diffusion–reaction | Poisson equations, elliptic boundary-value problems | `DiffusionReactionHDGSolver` |
| Advection–diffusion–reaction | Combined transport and diffusion, scalar and tensor diffusion coefficients | `AdvectionDiffusionReactionHDGSolver` |

Each solver has a reusable class for repeated solves and a matching
`solve_*_hdg` function for one-shot use. Results expose the computed field,
trace, timings, and convergence information, so application code can inspect
the numerical result as well as plot it.

The surrounding tools cover the usual steps of an HDG experiment:

- **Geometry and approximation.** Structured meshes and Gmsh geometries,
  triangular polynomial spaces, scalar and vector DG fields, L2 projection,
  quadrature, and field transfer.
- **High-order recovery.** Mixed formulations reconstruct scalar and flux
  fields locally; supported diffusion and ADR paths also offer degree-`p+1`
  recovery and Raviart–Thomas flux reconstruction.
- **CPU and GPU execution.** NumPy reference implementations, multithreaded
  Numba kernels, and CuPy/raw-CUDA assembly and reconstruction share the solver
  interface. GPU solves keep traces and fields on the device between steps.
- **Diagnostics and visualization.** Field integrals, error norms, physical
  residual checks, transport-velocity compatibility measures, scalar plots,
  live GPU panels, and movie output.

Backend support depends on the equation, coefficient representation,
polynomial degree, and requested recovery. The
[capability table](docs/reference/backend_capabilities.md) lists the supported
combinations, and the [numerical formulations](docs/algorithms/README.md)
explain the discretizations.

## A first solve on the CPU

A manufactured Poisson problem on the square [-4, 4]². The solve needs only the
base installation; the plot also needs the `plot` extra:

```python
import numpy as np
from hybridge import DGSpace, rectangle_mesh, solve_diffusion_reaction_hdg
from hybridge.io import plot_solution_comparison

mesh = rectangle_mesh(10, 10, xlim=(-4., 4.), ylim=(-4., 4.))
space = DGSpace(mesh, 4, basis_type="dub_orth")
exact = lambda x, y: np.sin(x**2 + y**2) + np.sin(x*y)
source = lambda x, y: ((x**2 + y**2) * (4*np.sin(x**2 + y**2) + np.sin(x*y))
                       - 4*np.cos(x**2 + y**2))     # -Δu = source, u = exact on the boundary.

result = solve_diffusion_reaction_hdg(
    source, lambda x, y: 0.*x, exact, space, stabilization=1., solver="direct",
    preconditioner=None, boundary_mode="eliminate", hdg_postprocess="primal", verbose=False)
print(f"L2 error {result.field.l2_error(exact):.1e} on {mesh.num_tri} triangles")
print(f"after postprocessing {result.postprocessed_field.l2_error(exact):.1e}")
plot_solution_comparison(result.field, exact, postprocessed=result.postprocessed_field,
                         exact_resolution=48, backend="matplotlib", show_error=False)
```

<div>

![HDG solution, postprocessed field and exact solution](docs/getting_started/media/first_solve_light.png#gh-light-mode-only)
![HDG solution, postprocessed field and exact solution](docs/getting_started/media/first_solve_dark.png#gh-dark-mode-only)

*The example above: the HDG solution with polynomials of degree 4, its
postprocessed field of degree 5, and the exact solution.*

</div>

The [minimal examples](MANUAL.md#minimal-end-to-end-examples) add
advection and an independent residual check; run them with
`python examples/diffusion_reaction_minimal.py` and
`python examples/advection_reaction_minimal.py`.

## A GPU vortex gas in Python

The video follows a two-dimensional guiding-center plasma: positive and
negative charge density ρ drifting with the E×B velocity between grounded
conducting walls. These drift–Poisson equations coincide with the
vorticity form of two-dimensional Euler flow, which makes the plasma a vortex
gas. A Poisson solve recovers the potential from the charge and an advection
solve moves the charge.

**Model.** With Γ the outer wall and the island boundary,

```math
\partial_t\rho+\nabla\cdot(\rho\mathbf{u})=0,\qquad
-\Delta\phi=\rho,\qquad
\mathbf{u}=(\partial_y\phi,\,-\partial_x\phi),\qquad
\phi|_\Gamma=0,
```

so that u·n = 0 on both walls. In the Euler reading, a zero island potential
lets the circulation around the island follow the charge distribution; a
Kelvin-consistent Euler flow would instead hold that circulation fixed with a
floating island potential.

**Discretization.** Each step solves for the new density with a linearly
extrapolated velocity, then recovers the new potential (one Euler step starts
the second-order scheme):

```math
\frac{3\rho^{n+1}-4\rho^{n}+\rho^{n-1}}{2\Delta t}
+\nabla\cdot\left(\rho^{n+1}\mathbf{u}^{*}\right)=0,\qquad
\mathbf{u}^{*}=2\mathbf{u}^{n}-\mathbf{u}^{n-1},\qquad
-\Delta\phi^{n+1}=\rho^{n+1}.
```

The velocity is the rotated HDG flux, and dividing the transport step by its
leading coefficient gives the form the solver receives:

```math
\mathbf{u}_h=(-q_{h,y},\,q_{h,x}),\quad \mathbf{q}_h\approx-\nabla\phi_h,\qquad
\rho^{n+1}+\nabla\cdot(\boldsymbol{\beta}\,\rho^{n+1})=s,\quad
\boldsymbol{\beta}=\frac{2\Delta t}{3}\mathbf{u}^{*},\quad
s=\frac{4\rho^{n}-\rho^{n-1}}{3}.
```

The reaction coefficient is therefore one. The loop below forms `s` and `beta`
with field arithmetic, which stays on the GPU; `hdg.bdf2_transport_data`
packages the same step with input checks.

The whole application is the script below. Assembly, linear solves,
reconstruction, and drawing belong to the library; the coupling and the time
history stay visible:

```python
import hybridge as hdg
from hybridge.io import HolovizScalarPanels

# A five-lobed star around a circular island, with degree-6 elements.
mesh = hdg.gmsh_smooth_star_mesh(
    0.005, radius=1., amplitude=.35, mode=5,               # Decrease h to refine.
    hole_radius=.3, boundary_points=500, num_threads=16)
space = hdg.DGSpace(mesh, 6, basis_type="dub_orth")

# 960 Gaussian charges of both signs, starting as close as five widths to a wall.
initial = hdg.sample_gaussian_blob_field(
    hdg.MeshDomain(mesh), counts=(512, 256, 128, 64),
    sigmas=(.008, .016, .024, .032), cutoff=5., seed=17, strength_mode="balanced")
rho = hdg.project_callable(initial, space, backend="device")

gpu = dict(assembly_backend="raw-cuda", solver_rtol=1e-9, solver_atol=1e-10,
           verbose=False)
# Poisson: -Δφ = ρ with φ = 0 on both walls. Transport: no flux through them.
poisson = hdg.DiffusionReactionHDGSolver(
    space, source=rho, boundary_condition=0., solver="fb-hp-mg-pcg",
    cache_local_factors="schur-cholesky", **gpu)
transport = hdg.AdvectionReactionHDGSolver(
    space, boundary_mode="zero-flux", solver="amgx", **gpu)
one = space.constant(1.)                                   # BDF2 as ρ + ∇·(βρ) = s.

dt, steps = 0.003125, 5080                                 # Final time 15.875.
previous_rho = previous_velocity = None                    # Euler startup, then BDF2.
limits = ((-18., 18.), (-.07, .07))                        # Fixed color scales.
with poisson, transport, HolovizScalarPanels(
        (space, space), ("Charge density", "Potential"), cmap="RdBu_r",
        width=640, height=640, show_mesh=False) as plot:
    potential = poisson.solve()
    velocity = hdg.perpendicular_vector_field(potential.flux)  # u = (-q_y, q_x).
    plot.update_fields((rho, potential.field), limits=limits)

    for step in range(1, steps + 1):
        if previous_rho is None:                               # Euler startup.
            source, beta = rho, dt * velocity
        else:                                                  # BDF2.
            source = (4 * rho - previous_rho) / 3
            beta = (2 * dt / 3) * (2 * velocity - previous_velocity)
        next_rho = transport.solve(source=source, beta=beta, reaction=one).field
        potential = poisson.set_source(next_rho).solve()       # Both solves warm-start.

        # Commit history only after both solves succeed.
        previous_rho, previous_velocity, rho = rho, velocity, next_rho
        velocity = hdg.perpendicular_vector_field(potential.flux)
        if step % 4 == 0:                                      # Draw every fourth step.
            plot.update_fields((rho, potential.field), step=step,
                               time_value=step * dt, limits=limits)
```

Run the same source from the repository root:

```bash
python examples/gpu_vortex_gas.py
```

It needs the GPU runtime, the `mesh` extra, and the `holoviz` extra for the
live panels, which stay on the GPU. At this resolution the run uses about
29 GiB of GPU memory; a larger mesh size `h` suits smaller
GPUs.

<!-- showcase-video: positive_density -->
https://github.com/user-attachments/assets/e47ec83d-15d5-4ad1-b105-64af5f8219aa

<div>
*The same domain with positive charge only and KKT positivity preservation, from
t = 0 to 6.4; click the image to play the video. Like-signed charge rolls up and
merges into larger vortices. After each transport step, a KKT projection
restores ρ ≥ 0 at constrained points in every element while conserving the total
charge, so no pixel falls into the pink used for negative density. The total
charge changes by 2.4e-11, the energy drifts by 4.1e-5, and 8.2% of the
enstrophy is dissipated. The KKT projection is on the `positivity-kkt`
development branch and is not part of this release.*

</div>

The [reproduction guide](docs/getting_started/gpu_showcase.md) gives the
recording commands, the time-step, mesh, and stabilization checks behind both
videos, and their measured diagnostics.

## Explore further

The package is the reusable library; `examples/` holds small applications and
`scripts/` fuller case runners and research studies, including manufactured
solutions, convergence studies, guiding-center time integration, and GPU
solver comparisons. Examples and runners are used from a checkout; the
installed wheel contains the library.

| Starting point | What you will find |
|---|---|
| [Examples](examples/README.md) | Small complete programs, including the GPU vortex gas |
| [User manual](MANUAL.md) | Solver construction, coefficients, boundary conditions, plotting, and runner commands |
| [Solver API](docs/reference/solver_api_alpha.md) | Public imports, reusable-solver updates, result objects, and failure behavior |
| [Coefficient inputs](docs/reference/coefficient_inputs.md) | Constants, callables, projected fields, and coefficient representations |
| [Backend capabilities](docs/reference/backend_capabilities.md) | Supported assembly, solve, recovery, and host/device combinations |
| [Numerical methods](docs/algorithms/README.md) | HDG formulations, stabilization, postprocessing, quadrature, and multigrid |
| [Research studies](docs/research/README.md) | Dated application and solver investigations with their supporting evidence |

- **Steady problems.** The advection, diffusion, and combined ADR runners expose
  manufactured cases and named configurations; `--list-presets` and
  `--dry-run` show a configuration before running it.
- **Unsteady problems.** Keep the solver objects between steps and update their
  data, as in the example above; the guiding-center applications use the same
  pattern at research scale.
- **Diocotron equilibria.** A few optional studies use DOLFINx as a
  continuous-Galerkin comparison, outside the HDG library; see the
  [manual](MANUAL.md#optional-guiding-center-and-diocotron-equilibria).

## Development status

HYBRIDGE is an **early-alpha research package**. The public solver API and
backend support are documented, while numerical methods, performance paths,
and application studies continue to evolve. The [roadmap](TODO.md) tracks
current work, and the [release record](docs/releases/early_alpha.md)
separates qualified evidence from open gaps.

To install the test dependencies and run the bounded host checks:

```bash
python -m pip install -e '.[test]'
python scripts/dev/alpha_test_matrix.py host-fast
```

The [validation guide](docs/development/alpha_test_matrix.md) describes the
installation, CPU-parity, and GPU lanes. Linear solves are accepted by
backend-neutral checks that include the residual of the original, unscaled
system; the [convergence contract](docs/reference/solver_convergence_contract.md)
defines success and failure. Discretization accuracy still depends on the
mesh, polynomial degree, stabilization, and time step chosen for a problem.

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

The companion project:

- **SOLEDGE-HDG:** G. Giorgiani, H. Bufferand, G. Ciraolo, P. Ghendrih,
  F. Schwander, E. Serre, and P. Tamain (2018).
  [A hybrid discontinuous Galerkin method for tokamak edge plasma simulations in global realistic geometry](https://doi.org/10.1016/j.jcp.2018.07.028).
  *Journal of Computational Physics* **374**, 515–532.

## Acknowledgements

The GPU backend builds on these upstream projects:

- [CuPy](https://github.com/cupy/cupy), developed by Preferred Networks and
  community contributors, provides GPU arrays and CUDA kernel integration.
- [NVIDIA AMGX](https://github.com/NVIDIA/AMGX), developed by NVIDIA and its
  contributors, provides GPU sparse linear solvers and algebraic multigrid.
- [PyAMGX](https://github.com/shwina/pyamgx), created by Ashwin Srinath and
  extended by contributors, provides the Python bindings to AMGX.

Our AMGX and PyAMGX versions are downstream forks of these projects, with
additional support for HDG workloads.

## License

HYBRIDGE is distributed under the [BSD 3-Clause License](LICENSE). Its authors,
collaborators, and their affiliations are listed in [`AUTHORS.md`](AUTHORS.md).
