# Face-dense GPU profiling

This stage adds systematic profiling without changing the validated numerical
kernels.  The production configuration is the directly eliminated Dirichlet
system; the penalty system remains available through `--boundary-mode penalty`
for regression comparisons.

## Timing rules

Device operations are measured with CUDA events.  The benchmark helper:

1. runs warm-up calls;
2. synchronizes once before measurement;
3. records all event pairs back-to-back on the current stream;
4. synchronizes once after the final repetition;
5. reports minimum, median, mean, standard deviation, and 90th percentile.

This avoids a host synchronization between samples.  Inputs, outputs, and
operator workspaces are allocated before the timed region.

Setup is measured with synchronized wall-clock time because it contains CPU
work, host-to-device transfers, CUDA-library initialization, and device work.
GMRES time-to-solution is also synchronized wall time because the algorithm
intentionally alternates GPU vector work with CPU Hessenberg updates.

## Running the benchmark matrix

```bash
PYTHONPATH=. python scripts/profile_face_dense_gpu.py \
    --orders 1 2 3 4 \
    --meshes 8 16 32 \
    --boundary-mode eliminate \
    --matvec-implementations matmul raw \
    --preconditioners none block_jacobi asm \
    --local-solver cublas_inverse \
    --warmup 10 \
    --repeats 100 \
    --restart 50 \
    --max-iterations 1000 \
    --rtol 1e-8 \
    --output-prefix results/face_dense_gpu
```

The script writes:

```text
results/face_dense_gpu.csv
results/face_dense_gpu.json
```

The CSV is convenient for tables and plotting.  The JSON additionally stores
the detailed per-operation GMRES profile.

## Quantities measured

### Assembly and setup

- CPU local and face-dense assembly;
- face-block layout conversion and host-to-device transfer;
- GPU operator workspace allocation;
- block-Jacobi setup;
- additive-Schwarz local-matrix construction, transfer, and inversion;
- cuBLAS `getrfBatched`/`getriBatched` setup when selected.

### Face-dense operator

- neighbour gather;
- dense face-row product;
- complete `matvec_into`;
- estimated floating-point operations and median GFLOP/s.

For each face, the estimated dense work is

\[
2 b(Sb),
\]

so a complete application is estimated as

\[
2N_F b^2S
\]

floating-point operations.

### Preconditioners

Block-Jacobi reports its complete batched local product.

ASM reports:

- restriction/gather;
- batched local inverse application or solve;
- prolongation/scatter-add;
- complete application.

### GMRES

The primary time-to-solution number comes from an **uninstrumented** solve.
Unless `--skip-detailed-gmres` is supplied, the script then runs a second,
separately instrumented solve and records:

- matrix-vector products;
- preconditioner applications;
- cuBLAS dot products;
- cuBLAS norms;
- AXPY;
- scaling;
- vector copies;
- basis GEMV updates;
- restart-coefficient host-to-device copies;
- CPU Hessenberg/Givens work;
- CPU triangular back substitution.

Dot products and norms return scalars to the CPU and therefore synchronize the
host.  For those categories, the report includes both CUDA-event time and host
wall time.  Their difference is an approximate synchronization/launch overhead,
not a rigorous hardware decomposition.

The detailed solve is capped by `--detailed-max-iterations` (default: 100) to
avoid creating an excessive number of CUDA events.  It may therefore stop before
convergence.  The uninstrumented solve still uses `--max-iterations` and remains
the authoritative time-to-solution run.

## Important interpretation rules

1. **Use uninstrumented GMRES wall time for time-to-solution.** Fine-grained
   event insertion perturbs the solve.
2. **Use median or minimum device time for kernels.** The mean is more affected
   by system noise.
3. **Compare preconditioners by total solve time, not only iteration count.**
   ASM is stronger but more expensive per application.
4. **Treat `gpu_solve` separately.** Public `cupy.linalg.solve` can allocate and
   refactorize on every application. CUDA-event time captures device work but
   not all host allocation overhead; synchronized GMRES wall time captures the
   practical consequence.
5. **Run enough work.** Very small meshes are dominated by launch latency and
   do not represent throughput behavior.
6. **Record hardware and software versions externally** when publishing final
   results: GPU model, CUDA driver/runtime, CuPy version, precision, clock/power
   settings, and mesh/order configuration.

## Programmatic API

```python
from hdgfem.backends.cupy_profiling import (
    CuPyGMRESProfiler,
    profile_additive_schwarz,
    profile_block_jacobi,
    profile_face_dense_operator,
)
```

The low-perturbation operator call is:

```python
profile = profile_face_dense_operator(
    operator,
    x_gpu,
    y_gpu,
    warmup=10,
    repeats=100,
)

print(profile.total.median_ms)
print(profile.gather.median_ms)
print(profile.dense_product.median_ms)
```

Fine-grained GMRES attribution is opt-in:

```python
profiler = CuPyGMRESProfiler(device_id=operator.device_id)
result = restarted_gmres_cupy(
    operator,
    rhs_gpu,
    preconditioner=asm,
    profiler=profiler,
)
summary = profiler.finalize()
```

Do not reuse a finalized profiler for another solve.
