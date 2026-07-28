# Explicit cuBLAS batched inversion for HDG preconditioners

This stage adds a fourth local-solver setup mode:

```python
local_solver="cublas_inverse"
```

It is available for both the face block-Jacobi and one-element additive-Schwarz
preconditioners.

## Setup algorithm

For a batch of small dense matrices

\[
P_i \in \mathbb{R}^{n\times n},\qquad i=0,\ldots,N_b-1,
\]

the backend performs:

1. Create a device array of pointers to the beginning of every matrix.
2. Call pivoted `cublas<t>getrfBatched` in place on a copy of the matrices.
3. Copy and inspect the per-batch factorization `info` array.
4. Call `cublas<t>getriBatched` into a distinct inverse batch.
5. Copy and inspect the per-batch inversion `info` array.
6. Verify

   \[
   \|P_iP_i^{-1}-I\|_\infty
   \]

   for every batch matrix.

The pointer, pivot, and status arrays are owned by CuPy. The low-level calls use
the cuBLAS handle and current stream managed by CuPy.

## C-order storage and cuBLAS column-major interpretation

The HDG local matrices are stored as C-contiguous batches. cuBLAS interprets
each raw matrix buffer as column-major, so it sees the transpose of the logical
C-order matrix. It consequently writes the inverse transpose in column-major
storage. Those bytes are exactly the C-order representation of the desired
logical inverse. No transpose copy is required.

The implementation does not rely on this argument silently: every setup checks
the residual of the logical C-order matrices against the returned inverses.

## Usage

### Block-Jacobi

```python
preconditioner = CuPyFaceBlockJacobiPreconditioner.from_system(
    system,
    local_solver="cublas_inverse",
    inverse_residual_tolerance=1.0e-10,
)
```

### Additive Schwarz

```python
preconditioner = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
    system,
    element_blocks,
    loc2glob_face,
    local_solver="cublas_inverse",
    inverse_residual_tolerance=1.0e-10,
)
```

The repeated preconditioner application remains the same as for
`gpu_inverse`: an allocation-free batched dense matrix-vector multiplication.
The low-level change affects setup only.

## Diagnostics

Both preconditioners expose:

```python
preconditioner.factorization_info
preconditioner.inversion_info
preconditioner.maximum_inverse_residual
```

For a successful setup, both status arrays contain only zeros. A positive
factorization or inversion status raises `numpy.linalg.LinAlgError` with the
first failing batch index. A negative status is reported as an invalid cuBLAS
argument.

## Validation

On a CUDA machine:

```bash
PYTHONPATH=. pytest -q \
    tests/test_cublas_batched_inverse.py \
    tests/test_cupy_local_factorization.py
```

Compare all setup modes and their timing:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_local_factorization.py
```

The report compares:

- `cpu_inverse`;
- `gpu_inverse` through the public CuPy solver;
- `cublas_inverse` through explicit `getrfBatched/getriBatched`;
- `gpu_solve`, which refactorizes at every application.

## Scope

The explicit backend is NVIDIA CUDA-specific. The higher-level preconditioner
interface remains backend-independent, so a later HIP implementation can use
hipBLAS or rocSOLVER without changing GMRES.
