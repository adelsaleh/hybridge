# Production diffusion-reaction GPU API and campaign runner

## Scope

The ordinary diffusion-reaction entry point now accepts
`solver="gpu_face_dense"`.  The local HDG matrices and reconstruction share the
existing implementation; the condensed trace problem uses the validated
face-dense CUDA operator, robust restarted GMRES, GPU block-Jacobi or
additive-Schwarz, and the harmonic-Ritz polynomial preconditioner.

The integration deliberately keeps the GPU implementation explicit.  It does
not silently replace the CPU sparse path, and all previous solver names retain
their behavior.

## One-shot API

```python
from hdgfem.solvers.diff_rea import solve_diffusion_reaction_hdg
from hdgfem.solvers.diff_rea_gpu import DiffusionReactionGPUOptions

options = DiffusionReactionGPUOptions(
    operator="auto",
    preconditioner="asm_poly",
    asm_application="auto",
    polynomial_degree=18,
    restart=75,
    autotune=True,
    autotune_cache_file="results/gpu_autotune.json",
)

result = solve_diffusion_reaction_hdg(
    source,
    reaction,
    boundary_condition,
    space,
    solver="gpu_face_dense",
    solver_rtol=1.0e-8,
    maxiter=2000,
    boundary_mode="eliminate",
    gpu_options=options,
)
```

The reconstructed field, flux, trace, and standard `SolveResult` remain
available through `DiffusionReactionResult`.  CUDA-specific information is
stored in `result.gpu_diagnostics`, including the selected kernels, cache hit,
setup and solve times, device workspaces, and the complete GPU GMRES result.

## Stateful API

`DiffusionReactionHDGSolver` forwards the same GPU options:

```python
solver = DiffusionReactionHDGSolver(
    space,
    source=source,
    reaction=reaction,
    boundary_condition=boundary_condition,
    solver="gpu_face_dense",
    boundary_mode="eliminate",
    gpu_options=options,
)
result = solver.solve()
```

This stage integrates the interface but does not yet retain the CUDA operator
between calls to the stateful solver.  The persistent autotuning cache is
reused; operator/preconditioner object reuse remains a later improvement for
Newton or time-stepping loops.

## Integrated validation

Small direct-reference comparison:

```bash
PYTHONPATH=. python scripts/validate_diff_rea_gpu_api.py \
    --mesh 4 --order 2 --compare-direct \
    --cache-file results/t600_autotune_cache.json \
    --output results/t600_gpu_api_small.json
```

Representative solve:

```bash
PYTHONPATH=. python scripts/validate_diff_rea_gpu_api.py \
    --mesh 64 --order 4 --restart 100 \
    --polynomial-degree 18 \
    --cache-file results/t600_autotune_cache.json \
    --output results/t600_gpu_api_64_p4.json
```

Run the same command twice.  The first run should tune on a cache miss; the
second should load the architecture-compatible selection.

## Mesocentre campaign runner

The runner records `nvidia-smi`, Python and CuPy configuration, writes one log
per command, and produces `campaign_summary.json`.

Check the command matrix without consuming GPU time:

```bash
PYTHONPATH=. python scripts/run_gpu_campaign.py \
    --campaign full \
    --output results/v100_campaign \
    --dry-run
```

Run levels:

```bash
# Short correctness and setup check
PYTHONPATH=. python scripts/run_gpu_campaign.py \
    --campaign smoke --output results/v100_smoke

# Numerical matrix, autotuning and polynomial comparisons
PYTHONPATH=. python scripts/run_gpu_campaign.py \
    --campaign medium --output results/v100_medium

# Adds local assembly and the full mesh/order scaling sweep
PYTHONPATH=. python scripts/run_gpu_campaign.py \
    --campaign full --output results/v100_full
```

Use `smoke` at the start of a limited allocation.  Proceed to `medium` or
`full` only after it completes successfully.  Each GPU architecture writes its
own autotuning entries because the cache key contains the device and CUDA
fingerprint.
