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
source .venv/bin/activate
python -m pip install -e .
python -m pytest
```

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
