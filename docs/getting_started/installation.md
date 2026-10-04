# Installation And Release Qualification

This document defines the bounded installation surface for the early-alpha
package. It distinguishes the installable `hdgfem` library from repository-only
runners, benchmark data, AMGX configurations, and research scripts.

The current package candidate is the PEP 440 prerelease `0.1.0a1`. Its local
release matrix is recorded in `docs/releases/early_alpha.md`; the exact release
commit and hosted Python 3.10/3.12 workflow run remain required before tagging.

## Base Host Install

The base package requires Python 3.10 or newer and installs NumPy, SciPy, and
Numba:

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

The installed wheel contains the `hdgfem` package. The `scripts/`, `configs/`,
`run_configs/`, tests, and benchmark artifacts remain repository workflows and
are intentionally not installed as package modules.

A minimal installed-API check is:

```python
import numpy as np

from hdgfem import DGSpace, rectangle_mesh, solve_global_system

space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
rows = np.array([0, 0, 1, 1])
cols = np.array([0, 1, 0, 1])
data = np.array([4.0, 1.0, 2.0, 3.0])
rhs = np.array([1.0, 2.0])
result = solve_global_system(rows, cols, data, rhs, 2, solver="direct")
assert result.converged and result.physical_residual_target_met
```

## Optional Dependency Groups

The package metadata defines these pip-resolvable groups. The `pardiso` and
`holoviz` groups require a supported platform; the others are portable:

| Extra | Contents | Purpose |
|---|---|---|
| `test` | pytest, Matplotlib, and tomli on Python 3.10 | Contract and regression tests, including exercised plotting and package-metadata paths |
| `mesh` | gmsh | Recommended Gmsh geometry paths (optional dependency) |
| `manufactured` | SymPy | Regenerate continuous n–Gamma manufactured forcing; not needed to evaluate committed data |
| `plot` | Matplotlib, PyVista | Plotting and visualization |
| `holoviz` | CUDA 13 Holoscan and CuPy, Matplotlib, Pillow, imageio-ffmpeg | NVIDIA GPU scalar panels and movie recording; see the [Holoviz guide](../backends/holoviz.md) |
| `release` | build, twine | Distribution construction and metadata checks |
| `pardiso` | pypardiso | Optional oneMKL PARDISO host direct solver |
| `all` | test, mesh, and plot dependencies, including the Python 3.10 TOML backport | Repository development convenience |

Gmsh remains optional so the structured-mesh and core solver paths stay lean,
but the `mesh` extra is highly recommended because most scripts and realistic configurations use it.

For example:

```bash
python -m pip install -e '.[test,mesh,plot]'
```

To regenerate the repository's n–Gamma continuous manufactured evaluators:

```bash
python -m pip install -e '.[manufactured]'
python -m scripts.n_gamma.manufactured
python -m scripts.n_gamma.manufactured --check
```

The committed `scripts/n_gamma/cases/forcing.py` uses NumPy or CuPy and never
imports SymPy. Its coordinates are `x=R-3, y=Z`; use `stationary=True` to freeze
the fields at zero phase and remove their time derivatives. Sources include
the cylindrical divergence and take no timestep or numerical history inputs.

PETSc/petsc4py, CUDA-specific CuPy wheels, PyAMGX/AMGX, and DOLFINx are not
declared as generic extras because they require ABI-, CUDA-, MPI-, or
distribution-specific installation. Install those stacks in matched
environments and use `docs/reference/backend_capabilities.md` to select a supported
solver combination.

All HDG commits after `v0.1.0a1` require the forked AMGX/PyAMGX development
stack when qualifying the checkout. Do not substitute the upstream `main`
branches. The required branches, exact tested commits, build commands, and
PyAMGX reinstall procedure are in
[`forked_amgx_stack.md`](forked_amgx_stack.md).

`python -m pip install -e '.[pardiso]'` installs the optional `pypardiso`
adapter and oneMKL runtime. The upstream package currently targets x86-64 Linux
and Windows. Select `solver="pypardiso"` or its `"pardiso"` alias for general
real matrices; use `"pypardiso-spd"` or `"pardiso-spd"` only for verified
symmetric-positive-definite systems. Set `MKL_NUM_THREADS` plus
`OMP_NUM_THREADS` before Python starts. This extra is not
included in `all` so the portable development install does not acquire a large
platform-specific runtime.

## GPU Runtime

The GPU paths need Linux, an NVIDIA GPU supported by CUDA 13, the CUDA 13
toolkit, CuPy built for CUDA 13, and the forked AMGX and PyAMGX builds. Build
AMGX as described in [`forked_amgx_stack.md`](forked_amgx_stack.md). Then, in
the virtual environment used for `hdgfem`, point the PyAMGX build at the
toolkit and at the AMGX source and build trees:

```bash
export CUDA_PATH=/path/to/cuda-13
export AMGX_DIR=/path/to/AMGX                  # Fork source, including headers.
export AMGX_BUILD_DIR=/path/to/AMGX-build      # Build tree containing libamgxsh.so.
export PATH="$CUDA_PATH/bin:$PATH"

python -m pip install 'cupy-cuda13x>=14,<15' cython setuptools wheel
python -m pip install --no-build-isolation --no-deps /path/to/pyamgx
python -c "import hdgfem, cupy, pyamgx; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s)')"
```

PyAMGX records the AMGX build directory as its runtime library path, so
`libamgxsh.so` needs no `LD_LIBRARY_PATH` entry. Add `$CUDA_PATH/lib64` only if
the CUDA runtime libraries are not on the default loader path. Reinstall
PyAMGX after rebuilding, moving, or changing the ABI of the AMGX library. Use a
[CuPy wheel matching your CUDA toolkit](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi).
`hdgfem` finds these backends through ordinary imports and selects them through
solver options; importing `hdgfem` itself needs none of them. The live
Holoviz viewer additionally needs the `holoviz` extra.

## Install Smoke

The release matrix contains a release-blocking `install-smoke` lane:

```bash
python scripts/dev/alpha_test_matrix.py install-smoke
```

It performs these checks without network access:

1. builds one wheel through the declared PEP 517 backend without build
   isolation;
2. installs that wheel into a temporary target without dependencies;
3. starts Python from outside the source tree;
4. verifies that `hdgfem.__file__` is inside the temporary installed target;
5. runs a public sparse solve with normalized true-residual acceptance;
6. constructs a DG mesh and space through package-level imports.

The default lane reuses the active environment's NumPy, SciPy, and Numba. A
networked clean environment must additionally verify dependency resolution:

```bash
python scripts/dev/clean_install_smoke.py --with-dependencies
```

That mode resolves dependencies into the temporary target and starts Python
with `-S`, preventing fallback to the active environment's site-packages.

## Distribution And CI Checks

Host CI runs the release-blocking host and CPU parity lanes on Python 3.10 and
3.12. A separate packaging job runs:

```bash
python -m pip install '.[test,release]'
python -m build
python -m twine check dist/*
python scripts/dev/clean_install_smoke.py --with-dependencies
```

Before an alpha tag, run all release-blocking lanes and record exact results in
`docs/releases/early_alpha.md`. The GPU lane must run on the production
CUDA/PyAMGX machine with zero runtime skips. Passing the host install smoke does
not qualify optional PETSc, CUDA, AMGX, DOLFINx, Gmsh, plotting, or pypardiso
paths; each optional runtime needs its own recorded check.
