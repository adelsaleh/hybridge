# Repeated ITER Poisson solves — 2026-09-24

On the stopped run's cached **364,468-triangle, p=6 ITER mesh**, the new `fast`
native policy reduced full repeated Poisson wall time from **1.35879 s to
0.25417 s (5.35x)** on two fixed density variations. Krylov time fell from
**1.31363 s to 0.21399 s (6.14x)**. These are static solver diagnostics, not a
replayed or newly integrated trajectory.

The stopped run already reused its assembled operator and hierarchy. Its
later solves spent 1.27–1.38 s in 23–25 robust PCGF iterations, plus about
0.029 s in RHS condensation and 0.008 s in reconstruction. The expensive
cycle, rather than repeated setup, was the main target.

## Confirmed comparison

Both candidates used the identical resident matrix, local factors, source
fields, boundary conditions and initial trace guess. The reference is the
initial physical density of the ITER preset. The two other sources multiply
its DG coefficients in each cell by `1 + 0.05*sin(1.7*x_c + 0.9*y_c)` and
`1 + 0.05*cos(2.3*x_c - 1.1*y_c)`, respectively. They are prescribed static
density variations, not approximations to particular saved time steps.

One two-source warmup was discarded before three measured repetitions per
policy. Hardware was the NVIDIA RTX PRO 5000 Blackwell with CUDA 13. Both
policies used the new explicit fixed-cycle coarse wrapper, so their comparison
measures the effect of multigrid tuning with that wrapper held constant.

| Policy | Full repeated wall (s) | Krylov (s) | Iterations | Worst original-matrix residual |
|---|---:|---:|---:|---:|
| robust | 1.358791 | 1.313632 | 24 | 7.201e-11 |
| fast | 0.254169 | 0.213994 | 38 | 8.299e-11 |

The target remains `max(1e-10, 1e-11*||b||)`, equal to `1e-10` for these
samples. An independent SciPy BSR product checked every returned trace.
The largest relative trace difference from the robust result was **2.821e-13**.
Every solve reused the same matrix allocation and native hierarchy; none
entered fallback. Matrix and RHS hashes were checked across candidates, and
the original matrix values were checked again after repeated solves.

`fast` uses `p6 -> p0`, order-1 Chebyshev, balanced 1+1 fine sweeps and
balanced 1+1 scalar-AMGX sweeps. `robust` uses `p6 -> p3 -> p1 -> p0`,
order-4 Chebyshev and balanced 2+2 sweeps. More outer iterations are worthwhile
because each fast cycle costs much less.

The p=6 ITER positive-turbulence preset now selects `fast`. Its robust
hybrid/CSR/FGMRES fallback ladder and independently checked residual criteria
remain enabled. The separate p=5 ITER recovered-field preset explicitly
retains `robust`; this experiment did not validate that operator.

The scalar-AMGX wrapper now receives `fixed_amg_cycles=1`, disabling inner
residual stopping and history. Outer PCGF still monitors and verifies its
residuals. Both coarse cycles returned exact zero for zero RHS; scaling the
RHS by `1e-40` gave relative linearity errors below `5.6e-16`.

## Evidence and reproduction

- Confirmation records (`run_outputs/solver_studies/iter_repeated_poisson_shared_confirmation_20260924/results.json`, local, untracked)
  include all samples, matrix/RHS hashes, numerical gates and source hashes.
- Initial policy screen (`run_outputs/solver_studies/iter_repeated_poisson_20260924/probe.json`, local, untracked)
  used one captured physical RHS and two algebraic perturbations. Standard
  order-2 smoothing took about 0.234 s and order-1 about 0.192 s, versus
  robust's 1.15 s, for the nearby algebraic samples. This screen selected the
  candidate before the full-path confirmation above.
- The first confirmation attempt rejected independently assembled matrix
  copies at its strict hash gate. Those incomplete records are retained in
  `iter_repeated_poisson_confirmation_20260924`; its partial timings are not
  included in the confirmed comparison. The successful comparison shares one
  fixed operator and its local factors across policies.
- 98 host-only configuration, small-matrix residual, fallback and preset tests
  passed with Numba JIT disabled. No builds, new kernel compilation, mesh
  generation or time integration were run.

From the repository root, choose a new empty output directory:

```bash
env PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_PATH=/path/to/cuda-13 LD_LIBRARY_PATH=/path/to/cuda-13/lib64 \
  .venv/bin/python -u -B \
  -m scripts.guiding_center.poisson.benchmark_repeated_poisson \
  --output run_outputs/solver_studies/iter_repeated_poisson_repeat \
  --repeats 3 --warmup 1
```

This diagnostic requires the existing ITER mesh and kernel caches and refuses
to generate a mesh or compile a missing kernel. It includes cached RHS
assembly, the global solve, device residual verification and reconstruction
in wall time. Independent host validation is excluded. The `cold` records
are setup diagnostics, not a matched first-solve comparison: after the first
policy, the operator is retained while the native hierarchy changes.

The existing production run command selects `fast` through its preset; no
build is needed. `--poisson-fb-hp-mg-preconditioner-policy robust` restores the
previous policy for comparison. Timing and convergence over an evolved
trajectory remain for the user's next run.

## Dense inverse memory gate

The trace system has **3,820,649 unknowns**. A full FP64 inverse alone needs
**116,778,870,249,608 bytes (106.2 TiB)**, before factorization or working
storage. FP32 still needs 53.1 TiB. The machine has about 125.3 GiB host RAM
and 47.8 GiB GPU memory. The requested dense-inverse trial was therefore
rejected before allocation; the user agreed to examine sparse LU instead.

See memory-gate record (`run_outputs/solver_studies/iter_repeated_poisson_20260924/inverse_memory_gate.json`, local, untracked).

## Sparse LU follow-up

The verified **16-thread PyPardiso** rerun successfully factored the full captured
matrix with `mtype=11`. Factorization took **12.24927 s**, and three measured
solves with retained factors averaged **2.82712 s** after one discarded warmup.
The MKL API reported a maximum of 16 threads, and the worker was restricted to
16 physical cores of the 24-core Intel Xeon w7-3455. Factorization consumed
**115.149 CPU-seconds in 12.249 wall-seconds**, or 9.40 cores on average across
its serial and parallel phases. This verifies actual parallel execution,
in addition to checking environment variables.

Every solve used phase 33 and passed the original-matrix target; the largest
absolute residual was `3.670e-11`. Peak process HWM was **19.01 GiB**. MKL
reported **1,231,945,211 factor nonzeros**, a **9.22x fill ratio**, and about
**16.10 GiB** estimated in-core peak. Reused solve times include PyPardiso's
matrix-hash and CSR-index preparation; they do not isolate native triangular
substitution. The first 16-thread-setting run gave similar results (12.094 s
factorization and 2.87418 s reused solves), before CPU-use instrumentation.

The **24-thread follow-up** used the identical captured matrix and RHS from
the full 364,468-triangle mesh (3,820,649 trace unknowns). MKL reported 24
threads and the worker's affinity included all 24 physical cores. The measured
factorization consumed 160.045 CPU-seconds in 12.625 wall-seconds, or 12.68
cores on average. Every returned solution passed the original `1e-10` target;
the largest absolute residual was `3.667e-11`.

| PyPardiso threads | Factorization (s) | Mean reused solve (s) | Peak process HWM (GiB) |
|---|---:|---:|---:|
| 16 | 12.249271 | 2.827124 | 19.009 |
| 24 | 12.625071 | 2.871935 | 19.236 |

Each setting has one timed factorization and three measured reused solves
after one warmup. The 24-thread solve samples were 2.865686, 2.872010 and
2.878110 seconds. Compared with the verified 16-thread run, setup was about
3.1% slower and reused solves about 1.6% slower. These small differences do
not establish a universal optimum; this follow-up shows no improvement from
24 threads for the measured PyPardiso wrapper path. Matrix and RHS hashes
match exactly across the two runs.

See the 24-thread records (`run_outputs/solver_studies/iter_poisson_lu_24threads_20260924/pardiso/result.json`, local, untracked).
To repeat that configuration, use `--threads 24` and a new output directory in
the reproduction command below.

The final comparison also ran native GPU multigrid on the **identical captured
matrix and initial physical RHS**, starting from zero every time:

| Reused solver | Mean solve (s) | Worst original-matrix residual |
|---|---:|---:|
| PyPardiso CPU LU, 16 threads | 2.827124 | 3.669e-11 |
| Fast native GPU PCGF, 46 iterations | 0.258692 | 7.343e-11 |

Each mean excludes one warmup and includes three measured solves. Both meet
the same `1e-10` absolute target. Their inputs already reside on their respective
devices; the table excludes host/device transfers, source condensation and
field reconstruction. GPU PCGF includes its original-matrix residual checks;
independent host validation is excluded from both timings. This compares the
implemented solver paths, not intrinsic CPU/GPU triangular-solve throughput.
The full Poisson-path comparison remains the changed-source experiment above.

- Verified parallel LU and matched GPU control (`run_outputs/solver_studies/iter_poisson_lu_parallel_control_20260924/summary.json`, local, untracked)
- Original PyPardiso run (`run_outputs/solver_studies/iter_poisson_lu_20260924/pardiso/result.json`, local, untracked)

PyPardiso exposes an opaque MKL factor handle, not transferable `L` and `U`
arrays. Intel's documented [`pardiso_export`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/pardiso-export.html)
supports a Schur complement, not general LU export. **No full-size GPU LU solve
was completed or timed.** CPU sparse LU fits in host memory, but a PyPardiso-to-CuPy
factor transfer is unavailable through the supported interface.

Earlier exploratory SciPy/SuperLU records are retained for audit. The normal
call failed before factorization completed. An alternate export attempt ran
serially despite 16-thread BLAS/OpenMP settings and was stopped after **22.5
minutes** following the user's correction. It produced no full factors or GPU
solve. [SuperLU's documentation](https://portal.nersc.gov/project/sparse/superlu/)
distinguishes the serial library used by SciPy from its separate parallel
implementations. Both SciPy factorization candidates have been removed from
this diagnostic. Large CPU factorizations and direct solves must use
multithreaded PyPardiso, with 16 threads or all available cores. This rule is
also recorded in the workspace instructions. SciPy supplies only sparse storage
and independent residual products in the retained diagnostic.

To reproduce the CPU LU benchmark, choose a new empty output directory:

```bash
env PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 \
  .venv/bin/python -u -B \
  -m scripts.guiding_center.poisson.benchmark_poisson_lu \
  --capture run_outputs/solver_studies/iter_repeated_poisson_20260924 \
  --output run_outputs/solver_studies/iter_poisson_lu_repeat \
  --candidates pardiso --threads 16 --repeats 3
```

The captured BSR arrays and RHS come from the retained initial policy screen.
The benchmark rejects unsupported thread counts before loading a large matrix,
verifies MKL's runtime limit and records process CPU time. CPU factorization and
solve are always PyPardiso. An optional `--candidates pardiso pmg-fast` adds the
existing GPU iterative control and requires the CUDA environment shown above.
Six additional host-only checks cover exclusion of serial candidates and the
large-matrix thread gate, without loading a large matrix or compiling code.
