# hdgfem

`hdgfem` is a discontinuous Galerkin / HDG research codebase.  The Python
package lives in `hdgfem/` and contains mesh, quadrature, DG space/field,
assembly, linear algebra, solver, plotting, and Numba backend modules.

## Scripts

Runnable experiments and test drivers live under `scripts/`, grouped by topic:

- `scripts/advection_reaction/`
- `scripts/diffusion_reaction/`
- `scripts/diffusion_reaction/experimental/`
- `scripts/diocotron_dolfinx/`
- `scripts/diocotron_hdg/`
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
python -m pip install -e .
python -m pytest
```

Small runner smoke checks:

```bash
python -m scripts.advection_reaction.run_adv_rea_cases --print-preset --dry-run
python -m scripts.diffusion_reaction.run_diff_rea_cases --print-preset --dry-run
```

See [MANUAL.md](MANUAL.md) for detailed CLI, API, solver, and performance notes.
