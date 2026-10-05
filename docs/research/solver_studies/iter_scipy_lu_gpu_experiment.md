# Temporary SciPy LU to GPU experiment

This user-run experiment loads the saved 364,468-triangle, p=6 ITER Poisson
matrix (3,820,649 trace unknowns), computes explicit CPU factors once, and times
repeated FP64 solves with the factors on the GPU. It uses the original captured
matrix, RHS, reference solution, and residual tolerances. An optional smaller
ITER case assembles a fresh stationary operator from a cached mesh. There is
no time integration. Production solver defaults are unchanged.

The user explicitly authorized SciPy for this isolated experiment. SciPy uses
the serial SuperLU driver: `--threads 24` sets affinity and BLAS/OpenMP limits.
Dense BLAS work can use multiple threads, but the sparse driver is not SuperLU_MT.
CPU setup may take a long time;
the report records CPU time, wall time, and effective core usage separately.
The usual large CPU solver remains multithreaded PyPardiso.

## Smaller ITER rerun, September 25

The cached h=0.2 mesh has **2,095 triangles and 21,532 trace unknowns at p=6**.
The run assembles the production Poisson operator using a constant unit source
and homogeneous Dirichlet data. This changes the RHS from the original density
capture; the LU and pMG-AMG comparisons below share the same new matrix and RHS.
No new mesh or kernel was generated. Command used:

```bash
env PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 \
  CUDA_PATH=/usr/local/cuda-13.0 LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  .venv/bin/python -u -B \
  -m scripts.guiding_center.poisson.benchmark_scipy_lu_gpu \
  --iter-mesh-size 0.2 --max-capture-dofs 25000 \
  --output run_outputs/solver_studies/iter_small_lu_gpu_20260925 \
  --threads 24 --factor-driver splu \
  --methods cupyx spsv spsv-graph spsm spsm-graph \
  --repeats 10 --timeout 300 --max-rss-gib 12 --live-output
```

The full CPU LU took **0.127872 s**. The saved L and U each have 1,651,776
nonzeros; together their FP64/int32 CSR storage occupies **37.97 MiB**.
Three factor-action checks had relative errors below `4.4e-16`. All repeated
GPU solutions passed the original-system `1e-10` absolute residual target;
the largest LU residual was `3.84e-13`, with no refinement needed.

Ten measured solves followed warmups. Setup, transfers, and host validation are
excluded; the checked time includes the original GPU residual check.

| Method | Raw solve (ms) | Checked solve (ms) |
|---|---:|---:|
| CuPy triangular wrapper | 126.342 | 126.771 |
| Cached SpSV | 7.166 | **7.602** |
| Cached SpSV with graph replay | 7.507 | 8.090 |
| Cached SpSM | 12.211 | 12.472 |
| Cached SpSM with graph replay | 11.632 | 12.176 |
| Fast pMG-AMG | — | 42.381 |

Retaining SpSV analysis was **5.57 times faster than pMG-AMG** for this small
case, whereas the ordinary CuPy triangular wrapper was slower. Graph replay
did not improve the measured means. These results do not establish timings
for the original 3.82-million-unknown matrix.

Results and reusable factors are in
`run_outputs/solver_studies/iter_small_lu_gpu_20260925/`. To repeat directly from
the saved factors, with live terminal output, run:

```bash
env PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 \
  CUDA_PATH=/usr/local/cuda-13.0 LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  .venv/bin/python -u -B \
  -m scripts.guiding_center.poisson.benchmark_scipy_lu_gpu \
  --capture run_outputs/solver_studies/iter_small_lu_gpu_20260925/poisson_capture \
  --factor-cache run_outputs/solver_studies/iter_small_lu_gpu_20260925/factors \
  --output run_outputs/solver_studies/iter_small_lu_gpu_20260925_replay \
  --threads 24 --repeats 10 --live-output
```

Each command needs a new empty output directory. The default five-method sweep
includes the matched pMG-AMG control. `--live-output` is now the default;
`--no-live-output` retains file logging without echoing worker output. Stages
and individual GPU timings are printed as they happen, with a heartbeat every
10 seconds (`--heartbeat-seconds` changes the interval). Caught Ctrl-C
interruptions are recorded in the worker result and benchmark summary.

`--iter-mesh-size` requires an existing cached mesh and caps new captures at
100,000 trace unknowns by default; the recorded run used a stricter 25,000 cap.
Reuse `--capture` and `--factor-cache` to avoid repeating assembly and LU setup.

## Larger ITER result, September 25

The user ran the h=0.03, p=6 case: **80,843 triangles and 845,915 trace unknowns**,
39.3 times the unknown count of the small case. The saved report is
`run_outputs/solver_studies/iter_h003_lu_gpu_20260925/summary.json`. It uses the
same constant-source capture procedure and matched pMG-AMG control.

| Method | Raw solve (ms) | Checked solve (ms) |
|---|---:|---:|
| Cached SpSV | 140.302 | 140.816 |
| Cached SpSV with graph replay | 142.833 | 143.373 |
| Fast pMG-AMG | — | **72.408** |

All ten measured solves per method passed. GPU LU needed no refinement and
had a maximum original-system residual of `1.65e-12`; the pMG-AMG residual was
`6.01e-11`, both below the shared `1e-10` target. The reported GPU LU speedup
is **0.514x**, meaning **pMG-AMG is 1.94 times faster**. Graph replay did not
improve the mean. The earlier small-mesh speedup does not persist at this size.

Memory capacity was sufficient: L and U together contain 252,723,408 nonzeros
and occupy **2.83 GiB**. Retained SpSV buffers/workspace occupy **2.96 GiB**;
the original CSR matrix adds **0.33 GiB**. These total about **6.1 GiB**, excluding
additional vectors, CUDA context, and allocator overhead.

CPU factorization took **76.394 s**, with 786.974 process CPU seconds (10.30
effective cores during that interval under the 24-thread limit). Each GPU
worker spent about **0.90 s** uploading data and **5.76 s** preparing retained
analysis. Those costs are excluded from the steady-state comparison above.
The measured raw GPU solve accounts for almost all of the checked LU time.

This experiment favors pMG-AMG for the larger single-RHS workload. It establishes
that the explicit factors fit on the device, but fitting alone does not make
their application faster than multigrid. It does not locate the crossover size
between the two measured meshes or establish full 3.82-million-unknown timings.

## Final near-capacity candidate (prepared, not run)

The selected final candidate is the existing **h=0.014, p=6 capture: 364,468
triangles and 3,820,649 unknowns**, 4.52 times the last benchmark's unknown count.
It reuses the original initial-density RHS and reference; the pMG-AMG control
uses that identical matrix and RHS. There is no fresh assembly in this command.

A rough sizing model fitted to the two completed meshes gives factor storage
proportional to `N^1.181`. The measured SpSV workspace/factor-storage ratio is
1.045. Applying those empirical ratios, plus the original matrix and external
vectors, gives:

| Cached mesh | Unknowns | Estimated GPU storage |
|---|---:|---:|
| h=0.02 | 1,883,511 | 15.7 GiB |
| **h=0.014** | **3,820,649** | **36.0 GiB** |
| h=0.01 | 7,514,101 | 79.5 GiB |

These are extrapolations, not measured sizes or upper bounds. The selected
case is the largest cached candidate expected to fit in the approximately
45.5 GiB free at preparation time. The no-drop factor driver can produce
different fill from standard `splu`. Actual factor uploads and analysis
workspaces must pass the runtime memory checks, with a **4 GiB reserve**.

Standard `splu` cannot be used unchanged on this input: its initial estimate
is `30 * 133,547,687 = 4,006,430,610` slots, above the signed 32-bit integer
maximum `2,147,483,647`. This is sparse-index/allocation bookkeeping; matrix
values remain FP64. The no-drop driver starts with `5 * nnz = 667,738,435`
slots and can grow while retaining all fill, subject to the same eventual
32-bit factor-index limits. It does not change or approximate the Poisson
matrix. The earlier run of this driver was still unfinished after 2 h 36 min;
setup may take hours again. This command allows 12 hours per worker.

```bash
env PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 \
  CUDA_PATH=/usr/local/cuda-13.0 LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  .venv/bin/python -u -B \
  -m scripts.guiding_center.poisson.benchmark_scipy_lu_gpu \
  --capture run_outputs/solver_studies/iter_repeated_poisson_20260924 \
  --output run_outputs/solver_studies/iter_final_lu_gpu_20260925 \
  --threads 24 --factor-driver spilu-nodrop --initial-fill 5 \
  --methods spsv spsv-graph --repeats 10 --live-output \
  --gpu-reserve-gib 4 --max-rss-gib 85 --reserve-gib 16 \
  --timeout 43200 --heartbeat-seconds 30
```

The capture paths and command parser were checked without starting a large
factorization. Completed factors will be saved under this new output directory
before GPU timing. The incomplete September 24 factor directory is not reused.

## Original large-matrix command

The September 24 large run ended during factorization after its last heartbeat
at 22:54:49 Paris time (2 h 36 min elapsed, 33.35 GiB RSS). It produced no
completed factor cache or GPU timing. Its termination reason was not recorded.

From the repository root, run:

```bash
env PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 \
  CUDA_PATH=/usr/local/cuda-13.0 LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  .venv/bin/python -u -B \
  -m scripts.guiding_center.poisson.benchmark_scipy_lu_gpu \
  --capture run_outputs/solver_studies/iter_repeated_poisson_20260924 \
  --output run_outputs/solver_studies/iter_scipy_lu_gpu_20260924 \
  --threads 24 --factor-driver auto --repeats 10
```

Choose a new empty output directory for each run. The command monitors each
worker with a six-hour timeout, 85 GiB process RSS limit, and 16 GiB host-memory
reserve. These can be changed using `--timeout`, `--max-rss-gib`, and
`--reserve-gib`. GPU uploads require space for both factors, the original CSR
matrix, vectors, and a 4 GiB reserve. Each cached triangular analysis additionally
checks its workspace requirement. Workers run sequentially in separate processes.
The runner only permits already-cached GPU kernels; it does not compile kernels
or invoke a build.

Completed factors are saved under `OUTPUT/factors`. To repeat GPU timings without
another CPU factorization, add this option to the command and use a new output:

```bash
--factor-cache run_outputs/solver_studies/iter_scipy_lu_gpu_20260924/factors
```

`factors/metadata.json` marks a completely written, validated cache. A partial
cache is never silently reused or overwritten. An interrupted factorization
cannot be resumed; use a new factor-cache directory if no completed cache exists.
GPU failures leave a completed CPU cache available for another attempt.

## Factorization detail

The installed SciPy 1.18.1 SuperLU uses 32-bit sparse indices. Its normal `splu`
driver initially estimates storage as `30 * nnz`; for this matrix that exceeds
the 32-bit limit before numerical factorization starts. The `auto` driver uses
`spilu` with **all dropping disabled** in this case: `drop_tol=0`,
`ILU_DropRule=0`, `ILU_FillTol=0`, `ILU_MILU=SILU`, and no equilibration.
`--initial-fill 5` controls the initial allocation, which may grow; it is not a
limit on retained fill. Small inputs use normal `splu` by default.

The alternate driver is accepted only if three independent action probes verify
`Pr A Pc = L U` to relative error below `1e-12`. Every GPU solution must also pass
the original BSR residual check and agree with the captured reference. Factors
are canonical CSR with unit-diagonal L and explicit-diagonal U. Both row and
column permutations are applied; no inverse is constructed. The SuperLU/CuPy
32-bit factor-index limits still apply if fill becomes too large.

## GPU paths and timing

The default sweep measures five candidates:

- `spsv`: cuSPARSE SpSV with analysis and workspace retained for each factor.
- `spsv-graph`: the same pair of solves and permutations captured in a CUDA graph.
- `spsm`: cuSPARSE SpSM with one RHS column and retained analysis/workspace.
- `spsm-graph`: the same SpSM operations replayed through a CUDA graph.
- `cupyx`: two ordinary `cupyx.scipy.sparse.linalg.spsolve_triangular` calls.

The cached paths use `hybridge.backends.cupy_triangular.ReusableCuPyLUSolve`.
CuPy's high-level wrapper repeats analysis on each call; the reusable helper
retains it. The native CUDA bindings also allow graph capture where CuPy's
stream-setting wrapper rejects it. Which path wins depends on factor structure
and triangular dependencies; the larger 845,915-unknown result above favors
pMG-AMG over both measured SpSV paths.

CPU factorization, factor transfer, triangular analysis, and graph setup are
reported separately. Steady-state raw wall time and CUDA-event time include
device RHS copying, both permutations, the two triangular solves, and copying
into the caller's device output. Checked wall time additionally includes the
original GPU residual and up to three LU iterative-refinement corrections if
needed. No tolerance is relaxed. Independent CPU BSR validation and device-to-host
validation copies are outside the solve timer.

The final control runs the existing fast pMG-AMG solver on the same captured
matrix and RHS, from a zero initial guess with a reused hierarchy. The summary
selects the fastest validated GPU LU by checked wall time and compares it with
the measured pMG-AMG time. Use `--no-pmg` to omit the control, or `--methods` to
select a subset of GPU candidates. Ten measured solves follow the warmups.

Send `OUTPUT/summary.json` after the run; it contains the per-method timings,
residuals, factor memory, setup costs, failures, and matched speedup. Detailed
progress is in `OUTPUT/<candidate>/worker.log` and `result.json`.

## Sources and validation

- [NVIDIA cuSPARSE SpSV](https://docs.nvidia.com/cuda/cusparse/#cusparsespsv)
  and [SpSM](https://docs.nvidia.com/cuda/cusparse/#cusparsespsm): analysis/solve
  separation, persistent workspaces, and CUDA graph support.
- [CuPy triangular solve](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.spsolve_triangular.html):
  supported public baseline; installed CuPy source was also checked for repeated
  descriptor creation and analysis.
- [SuperLU project](https://portal.nersc.gov/project/sparse/superlu/): serial,
  multithreaded, and distributed SuperLU are separate implementations.
- [SciPy spilu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spilu.html),
  [vendored allocation code](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/sparse/linalg/_dsolve/SuperLU/SRC/dmemory.c),
  [allocation defaults](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/sparse/linalg/_dsolve/SuperLU/SRC/sp_ienv.c),
  and [no-drop factorization code](https://raw.githubusercontent.com/scipy/scipy/v1.18.1/scipy/sparse/linalg/_dsolve/SuperLU/SRC/dgsitrf.c).

Unit validation uses a 48-by-48 nonsymmetric matrix with nontrivial permutations,
changing right-hand sides, both factor drivers, disk-cache reload, and all five
GPU paths. The 2,095-triangle stationary run passed all five paths and the
matched pMG-AMG control. The user's 80,843-triangle run passed both SpSV paths
and the matched control. Live output and caught-interruption persistence are
tested with non-numerical subprocesses. The full 364,468-triangle ITER GPU LU
performance comparison remains unmeasured.
