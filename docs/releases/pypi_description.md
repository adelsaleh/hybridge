# HYBRIDGE

**HYBRIdizable Discontinuous Galerkin Environment: HDG methods in Python, on
CPUs and NVIDIA GPUs.**

HYBRIDGE aims to make hybridizable discontinuous Galerkin (HDG) methods
accessible to newcomers, in the spirit of FreeFEM and DOLFINx. Build a mesh,
define polynomial fields, and assemble and solve transport, diffusion, and
coupled advection–diffusion–reaction problems on two-dimensional triangular
meshes through one Python interface, on the CPU or entirely on the GPU.
Reusable solvers serve single boundary-value problems and the repeated solves
of time-dependent applications.

[![Two-species guiding-center plasma: charge density and potential at t = 6. Click to play the video.](https://raw.githubusercontent.com/adelsaleh/hybridge/v0.1.0a2/docs/getting_started/media/vortex_gas_light.png)](https://github.com/adelsaleh/hybridge/issues/1#issuecomment-6020124224)

*A guiding-center plasma drifting between grounded walls, computed with
degree-6 HDG on 360,379 triangles and two solves per time step on one GPU.
Click the image to play the video; the
[project page](https://github.com/adelsaleh/hybridge#readme) shows the
complete script.*

## Installation

With Python 3.10 or newer:

```bash
python -m pip install hybridge
```

The base package needs NumPy, SciPy, and Numba, which is enough for the first
solve below. Extras add optional features, for example
`python -m pip install "hybridge[mesh,plot]"`:

| Extra | Adds |
|---|---|
| `plot` | Matplotlib and PyVista plotting |
| `mesh` | Gmsh geometries |
| `pardiso` | PyPardiso direct solves on Intel oneMKL |
| `holoviz` | Live GPU visualization with NVIDIA Holoviz |

GPU solves also need Linux, CUDA 13, CuPy, and our forks of NVIDIA AMGX and
PyAMGX, which add HDG block systems. The
[installation guide](https://github.com/adelsaleh/hybridge/blob/v0.1.0a2/docs/getting_started/installation.md)
covers every runtime.

## A first solve

A manufactured Poisson problem on the square [-4, 4]². The plot needs the
`plot` extra:

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

![The HDG solution, its postprocessed field and the exact solution](https://raw.githubusercontent.com/adelsaleh/hybridge/v0.1.0a2/docs/getting_started/media/first_solve_light.png)

*The HDG solution with polynomials of degree 4, its postprocessed field of
degree 5, and the exact solution.*

## Documentation

| Starting point | What you will find |
|---|---|
| [Project page](https://github.com/adelsaleh/hybridge#readme) | Overview, the GPU examples and their videos |
| [User manual](https://github.com/adelsaleh/hybridge/blob/v0.1.0a2/MANUAL.md) | Solver construction, coefficients, boundary conditions, plotting, and runner commands |
| [Examples](https://github.com/adelsaleh/hybridge/blob/v0.1.0a2/examples/README.md) | Small complete programs, run from a checkout |
| [Solver API](https://github.com/adelsaleh/hybridge/blob/v0.1.0a2/docs/reference/solver_api_alpha.md) | Public imports, reusable-solver updates, result objects, and failure behavior |
| [Backend capabilities](https://github.com/adelsaleh/hybridge/blob/v0.1.0a2/docs/reference/backend_capabilities.md) | Supported assembly, solve, recovery, and host/device combinations |
| [Release notes](https://github.com/adelsaleh/hybridge/blob/v0.1.0a2/docs/releases/0.1.0a2.md) | Upgrading from `hdgfem` 0.1.0a1, new features, and known gaps |

## Status

HYBRIDGE is an **early-alpha research package**. The public solver API and
backend support are documented, while numerical methods, performance paths,
and application studies continue to evolve. The
[release record](https://github.com/adelsaleh/hybridge/blob/v0.1.0a2/docs/releases/early_alpha.md)
separates qualified evidence from open gaps.

HYBRIDGE is the companion HDG project of SOLEDGE-HDG, the high-order HDG code
for tokamak edge plasmas in realistic geometry
([Giorgiani et al., 2018](https://doi.org/10.1016/j.jcp.2018.07.028)).

## License

HYBRIDGE is distributed under the BSD 3-Clause License. Its authors,
collaborators, and their affiliations are listed in
[`AUTHORS.md`](https://github.com/adelsaleh/hybridge/blob/v0.1.0a2/AUTHORS.md).
