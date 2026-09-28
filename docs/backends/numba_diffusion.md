# Host Numba diffusion Schur assembly

`DiffusionReactionHDGSolver(assembly_backend="numba")` supports
`cache_local_factors="none"`, `"schur-lu"`, and `"schur-cholesky"` for
identity diffusion, eliminated Dirichlet traces, and the `legacy-lagrange`
and `legendre-modal` trace bases. The default remains `none`.
Persistent factors require `cache_device_matrix=True` (the existing option
also controls host operator retention). The one-shot functional solver rejects
persistent factor policies; use the reusable solver to own their lifetime.
Cholesky requires positive scalar stabilization at the solver boundary and
checks finite, symmetric positive-definite local Schur blocks. Variable reaction
in the solution DG space is supported when those blocks remain positive definite.
Tensor diffusion remains on the existing uncached path.

## Algebra and storage

All policies share the projected-coefficient kernels, trace orientation,
strong boundary elimination, and COO emission. The local scalar block is

```
S = M_tau + J^-1 [(N_x-D_x) M_ref^-1 D_x + (N_y-D_y) M_ref^-1 D_y].
```

The uncached path forms and factors `S` inside each element's assembly or
recovery. LU caching retains one pivoted `nel × nel` factor and `nel` integer
pivots per element. Cholesky retains one `nel × nel` factor, using its lower
triangle, without pivot storage. Neither retains a global batch of mixed
`3*nel × 3*nel` matrices or trace-response tensors. Element-private work arrays
and `prange` keep local solves independent. Cached application rebuilds only
the quadratic-cost geometric coupling data and skips cubic Schur construction
and factorization.

RHS updates combine source and prescribed boundary trace contributions into
one local RHS and solve it once per element. The same helper reconstructs the
primal and both flux components after the global solve. Assembly emits COO
contributions; the existing sparse solver adapters sum duplicates and form CSR.
The benchmark measures that conversion separately.

## Cache lifetime

`set_source` and `set_boundary_condition` preserve factors and the reduced
operator, invalidating RHS/solution state. `set_reaction`, `set_space`,
`with_options`, and `clear_cache` discard the local factors. Source data and
boundary values do not enter the factor signature. Geometry, reference
matrices, reaction, and stabilization do. Backend entry points reject a stale
signature before applying supplied factors, including in-place coefficient
changes. Follow the public API rule to call `clear_cache()` after external
mutation of stored objects.

The backend helpers `build_diffusion_schur_cache_numba` and the existing
assembly/RHS/reconstruction adapters accept a `NumbaDiffusionSchurCache` via
`cached_factors`. These are backend implementation APIs. Solver timing details
report factor bytes and reuse; initial factor construction is included in initial
assembly time. Host and device caches have distinct storage and cannot be
interchanged.

## Reproducible checks and measurements

```bash
NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_numba_diffusion_schur_cache.py \
  tests/test_numba_diffusion_schur_cache_cuda.py

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m \
  scripts.diffusion_reaction.benchmark_numba_schur \
  --output run_outputs/numba_schur.jsonl
```

The benchmark is assembly-only, with no global solve or time integration.
Each mesh/order/trace/policy/thread case has a separate process. Every measured
phase is warmed first; records include all sample times, CPU/wall ratio,
Numba's actual thread count and runtime, retained factor bytes, and process
peak RSS including imports, discarded JIT warmup and untimed parity checks. Factor construction,
assembly using retained factors, RHS update, recovery and COO-to-CSR conversion
are separate phases. Fresh cached setup costs factor construction **plus**
assembly. The synthetic reconstruction trace is used for timing, while the
small-matrix tests verify solved traces and physical residuals independently.

Qualification results and their limits are recorded in the
[host Schur study](../research/solver_studies/numba_diffusion_schur_2026_09_25.md).
