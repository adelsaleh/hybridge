# Host diffusion Schur qualification, September 25, 2026

The Numba diffusion backend now shares its fused assembly/recovery algebra
between uncached, persistent scalar Schur-LU, and persistent scalar
Schur-Cholesky paths. Source/boundary changes reuse factors; reaction, geometry,
space, and option changes invalidate them. The [backend guide](../../backends/numba_diffusion.md)
defines the supported API and cache contract. The default remains `none`.

## Numerical evidence

```bash
NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_numba_diffusion_schur_cache.py \
  tests/test_diffusion_reaction_assembly_parity.py \
  -k 'schur_cache or modal_numba_solve'

NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_numba_diffusion_schur_cache_cuda.py
```

The first command passed **54 tests**, with 91 unrelated cases deselected.
The second passed **18 tests**, with no skips. These are stationary, small
matrix checks; no AMGX compilation, global GPU solve or time integration was
needed. Numba and CuPy runtime JIT were enabled.

Covered scope:

- All three host policies, legacy nodal and Legendre modal traces; assembly,
  changed-source RHS and reconstruction at p=0,1,3,6 against uncached Numba.
- Cached LU/Cholesky matrix, RHS, trace, primal and flux parity with NumPy on
  sheared meshes at p=0,2,6, including variable projected reaction and nonzero
  Dirichlet data, source/boundary updates and cache invalidation, both cold and
  assemble-then-solve entry paths.
- Independent local factor-action checks at p=1,3,6 on stretched meshes;
  nonfinite/invalid Cholesky input and stale in-place coefficient rejection.
- Independent reduced physical residuals below `1e-9` and unscaled mixed
  equation residual norms below `1e-9`, using the shared `hdg_residual` helper,
  extended to accept reaction and trace-space selection.
- Raw-CUDA CSR matrix, changed-source RHS, full mixed reconstruction and
  original-system residual parity for p=1,3,6 and both trace bases. The LU
  cases reuse raw device factors; host Cholesky factors additionally reproduce
  the CuPy Cholesky scalar operator.

The sheared p=6 nodal-volume reference has Schur condition numbers about
61,000--62,000. Its explicitly assembled NumPy local inverse differs from
Numba recovery by about `1.02e-8` in some flux coefficients. Measured unscaled
mixed residual norms were `1.89e-10` for NumPy, `2.62e-12` for host LU and
`2.00e-12` for host Cholesky. The p=6 NumPy field comparison therefore uses
`atol=2e-8`, while retaining the independent `1e-9` equation-residual gate;
other field comparisons use tighter tolerances.

These checks also exposed and fixed an existing NumPy assemble-then-solve
cache defect: the solver must retain full matrix boundary couplings when it
returns a reduced assembly, so subsequent source/boundary updates can eliminate
known traces correctly. Both reduced and full representations are now retained
in their intended slots.

A broader solver/API run initially had 85 passes and one unrelated failure:
`test_supported_solver_exports_are_identical_at_both_package_levels` expects
an export set without the existing `ScaledUpwind`. That export mismatch is not
part of this change. After the cache fix, the solver/API, lane and documentation
selection had 79 passes, one export test deselected and five documentation
failures from existing workspace content: the extra `docs/diocotron` directory,
LaTeX output files/PDFs, missing diagnostic artifact links, and missing docstrings
outside the added code. These files were preserved. A pre-existing `ComplexWarning` in a postprocessing test
also occurred; this study does not claim a clean repository-wide suite.
The final solver/API/test-lane selection passed **75 tests**, with that single
export test deselected and the same warning:

```bash
NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_solver_api_contract.py tests/test_diffusion_reaction_solver.py \
  tests/test_alpha_test_matrix.py \
  -k 'not supported_solver_exports_are_identical_at_both_package_levels'
```

## Thread scaling and phase costs

Reproduction:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m \
  scripts.diffusion_reaction.benchmark_numba_schur \
  --meshes 16 32 --orders 2 4 6 --threads 1 4 16 \
  --bases legacy-lagrange legendre-modal \
  --policies none schur-lu schur-cholesky --repeats 3 \
  --minimum-seconds 0.03 --output run_outputs/numba_schur.jsonl
```

All **108 cases** passed their untimed matrix/RHS/recovery parity checks.
The [complete JSONL records](numba_diffusion_schur_2026_09_25.jsonl) retain
three warmed samples per phase, batching counts, runtime settings, CPU/wall
ratios, retained factor bytes and peak process RSS. Each case ran in a separate
sequential process. Imports, first compilation/cache loading and warmups are
excluded from phase timings. Peak RSS includes them and the untimed parity
checks, so it is a process high-water mark, not isolated kernel workspace.

Machine: Intel Xeon w7-3455, 24 cores, one hardware thread per core, 2 MiB L2
per core; affinity allowed CPUs 0--23. Python 3.12.3, NumPy 2.5.3, Numba 0.67.0,
SciPy 1.18.1, Numba OpenMP runtime; OpenBLAS/MKL each used one thread. GPU
checks used an NVIDIA RTX PRO 5000 Blackwell, driver 580.126.09.

Selected p=6, 2,048-triangle, Legendre-modal results (median milliseconds):

| Policy | Threads | Factor build | Assembly | RHS update | Reconstruction | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| none | 1 | — | 226.362 | 79.989 | 81.652 | 286.6 |
| none | 4 | — | 61.076 | 20.914 | 22.152 | 285.9 |
| none | 16 | — | 17.927 | 6.102 | 5.769 | 286.4 |
| Schur-LU | 1 | 73.314 | 151.870 | 9.449 | 8.104 | 304.0 |
| Schur-LU | 4 | 19.487 | 43.540 | 3.123 | 2.268 | 304.2 |
| Schur-LU | 16 | 5.963 | 13.330 | 1.598 | 1.087 | 304.4 |
| Schur-Cholesky | 1 | 71.113 | 146.679 | 8.493 | 8.191 | 304.3 |
| Schur-Cholesky | 4 | 18.865 | 42.081 | 3.206 | 2.366 | 303.9 |
| Schur-Cholesky | 16 | 9.976 | 13.402 | 1.597 | 1.701 | 303.8 |

Cached assembly uses retained factors: fresh setup is **factor build plus
assembly**. At 16 threads LU gives about 3.82x faster RHS updates and 5.31x
faster reconstruction, but fresh setup is about 19.29 ms versus 17.93 ms
uncached. LU retains 12.69 MiB of factors/pivots; Cholesky retains 12.25 MiB.
The assembly CPU/wall ratio was approximately 1, 4 and 16 at the configured
thread counts, confirming actual parallel CPU consumption.

COO-to-CSR conversion cost 14--19 ms in these p=6 modal cases and is explicitly
separate from the assembly table. It can dominate setup once local assembly
is parallelized. No global sparse solve was timed; the benchmark uses an
explicit synthetic trace for reconstruction and parity tolerances
`rtol=2e-8`, `atol=2e-10`.

For p=2 and small meshes the signature/preparation overhead can make cached
updates slower than uncached execution, especially with many threads. The
Cholesky and LU rankings also vary between samples. These results support
opt-in reuse for repeated high-order work, not a new universal default.

## Implementation choices and limits

Contiguous element-major arrays, element-private scratch and `prange` partition
work across cores. One element and its trace columns form a bounded batch:
at p=6 the assembly scratch is roughly 87 KiB per worker, below this CPU's
2 MiB L2. RHS/recovery use just one column. Reuse avoids cubic local Schur
construction/factorization and does not retain full mixed inverse/response
batches. Wider column batching, packed triangular storage, NUMA placement,
other CPU architectures, p>6 performance and tensor-diffusion caching have no
performance qualification here. None is selected automatically.

The numerical sources were unchanged throughout benchmark collection. Their
SHA-256 hashes, recorded after collection, are:

- `hybridge/backends/numba.py`: `1df0e2fad89b8cb9681aa7e113d3eba17d199b2f1bfcd00ab99c644c8de153e9`
- `hybridge/kernels/diffusion_reaction_fused.py`: `4dba8f6db239396c1c3db8836b098c7767be6d754ecf0162d9ca0e5f2ea897a0`
- `hybridge/kernels/common.py`: `656977195abba567ee251c27b8fd6d9917d9ea13b09283226360634044ecd78d`
