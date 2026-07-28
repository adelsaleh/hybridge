# T600 hot-path optimization, stage 1

The first real CUDA profile exposed three implementation issues that can be
addressed before access to the final target GPU.

1. `cupy.shares_memory` was called inside every matrix-vector and
   preconditioner application.  CuPy implements this generic overlap test with
   GPU sorting/search kernels.  Nsight therefore showed hundreds of
   `stable_sort`, radix-sort, and search kernels in the GMRES hot path.  The
   solver now uses a host-only pointer-range check after validating that all
   vectors are C-contiguous.

2. The profiling script used the first value supplied through
   `--matvec-implementations` as the GMRES operator.  A command containing
   `--matvec-implementations matmul raw` therefore timed the raw operator but
   still solved with `matmul`.  GMRES selection is now explicit through
   `--gmres-matvec-implementation`.

3. The face operator's raw kernel substantially outperformed batched
   `cupy.matmul` on the T600.  The same alternative is now available for the
   inverse-block application in block-Jacobi and additive Schwarz through
   `application="raw"` or `--preconditioner-application raw`.

## Validation

Run the host-accessible tests:

```bash
PYTHONPATH=. pytest -q \
    tests/test_cupy_hot_path.py \
    tests/test_cupy_profiling.py \
    tests/test_cupy_raw_preconditioner.py
```

Run the actual CUDA comparisons:

```bash
PYTHONPATH=. pytest -q tests/test_cupy_raw_preconditioner.py
```

## Recommended T600 rerun

```bash
PYTHONPATH=. python scripts/profile_face_dense_gpu.py \
    --orders 1 2 3 4 \
    --meshes 8 16 32 64 96 128 \
    --boundary-mode eliminate \
    --matvec-implementations raw matmul \
    --gmres-matvec-implementation raw \
    --preconditioners none block_jacobi asm \
    --local-solver cublas_inverse \
    --preconditioner-application raw \
    --warmup 10 \
    --repeats 100 \
    --restart 30 \
    --max-iterations 1000 \
    --rtol 1e-8 \
    --detailed-max-iterations 50 \
    --output-prefix results/t600_hot_path_stage1
```

The next optimization stage should replace scalar-by-scalar MGS2
orthogonalization with a CGS2/GEMV path.  The profile showed thousands of dot
products and device-to-host scalar transfers; the batched orthogonalization
should reduce these from quadratic to linear in the restart dimension.
