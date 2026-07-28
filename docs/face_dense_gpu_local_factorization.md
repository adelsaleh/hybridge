# GPU local factorization for face-dense HDG preconditioners

This stage separates **local matrix construction** from **local factorization**.
The same block-Jacobi diagonal matrices and one-element additive-Schwarz matrices
can now be handled by three CUDA strategies.

## Available strategies

### `cpu_inverse`

The validated NumPy setup computes the inverse matrices on the CPU and transfers
them once. Application is a batched dense matrix-vector multiplication on the
GPU. This remains the cross-check reference.

### `gpu_inverse`

The original local matrices are transferred to the GPU. During setup, the code
solves

\[
P_e X_e = I
\]

for all batches with `cupy.linalg.solve`, producing and validating the inverse
matrices on the GPU. Every GMRES application then uses batched `cupy.matmul`:

\[
z_e = P_e^{-1} r_e.
\]

This is the preferred CuPy production path for the current stage: setup is on
the accelerator and the repeated application is allocation-free.

### `gpu_solve`

The code retains the original local matrices and calls

```python
cupy.linalg.solve(local_matrices, local_rhs)
```

for every preconditioner application. CuPy's public solve API accepts batches,
but it does not expose a reusable batched LU-factor object or an `out` argument.
Consequently this mode refactorizes and allocates during every application. It
is intentionally kept only as an independent correctness and timing baseline.

## Additive-Schwarz data flow

For each GMRES application:

1. Restrict the global face vector into element-local vectors.
2. Apply either the stored inverse or the direct batched solve.
3. Scatter-add element corrections to the global face vector.

The CUDA restriction and prolongation kernels are unchanged by the selected
local solver.

## Usage

```python
asm = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
    system,
    element_blocks,
    loc2glob_face,
    local_solver="gpu_inverse",
)
```

For a comparison solve:

```python
asm_solve = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
    system,
    element_blocks,
    loc2glob_face,
    local_solver="gpu_solve",
)
```

Block-Jacobi exposes the same `local_solver` choices.

## Validation

Run CPU and CUDA tests:

```bash
PYTHONPATH=. pytest -q \
    tests/test_face_additive_schwarz.py \
    tests/test_cupy_local_factorization.py
```

On a CUDA machine, compare setup time, application time, and errors:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_local_factorization.py
```

## Current limitation and next low-level step

`gpu_inverse` uses CuPy's public batched dense solver during setup. A later
low-level backend can call cuBLAS `getrfBatched` + `getriBatched` explicitly,
retain status arrays, and control pointer layouts and streams. The current class
interfaces and face/element storage do not need to change for that replacement.
