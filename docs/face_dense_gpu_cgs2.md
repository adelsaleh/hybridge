# Batched CGS2 orthogonalization for GPU GMRES

## Motivation

The original GPU GMRES uses modified Gram--Schmidt (MGS). At Arnoldi column
`j`, one MGS pass executes `j+1` cuBLAS DOT calls and `j+1` cuBLAS AXPY calls.
Every DOT coefficient is needed by the CPU Hessenberg update, so each DOT also
introduces a device-to-host scalar synchronization.

For restart dimension `m`, one full MGS restart performs

\[
\frac{m(m+1)}{2}
\]

DOT/AXPY pairs. MGS with reorthogonalization doubles that count.

Classical Gram--Schmidt batches the same algebra:

\[
h = V_j w,
\qquad
w \leftarrow w - V_j^T h,
\]

where the Arnoldi basis is stored by rows,

\[
V_j \in \mathbb{R}^{(j+1)\times N}.
\]

Each pass therefore uses:

1. one cuBLAS GEMV for all projection coefficients;
2. one cuBLAS GEMV for the correction;
3. one short device-to-host coefficient-vector transfer.

CGS2 repeats this pass once. It reduces the synchronization count from
`2*(j+1)` scalar transfers at column `j` for MGS2 to two short-vector transfers.

## Available modes

`restarted_gmres_cupy` now accepts:

```python
orthogonalization="mgs"
orthogonalization="mgs2"
orthogonalization="cgs"
orthogonalization="cgs2"
```

The old argument remains available:

```python
reorthogonalize=True
```

and maps to `mgs2`. Do not combine it with an explicit `orthogonalization`
argument.

## Device and CPU responsibilities

For CGS2, the GPU performs:

```text
w = M^{-1} A v_j
h_1 = V_j w              (GEMV)
w  -= V_j^T h_1          (GEMV)
h_2 = V_j w              (GEMV)
w  -= V_j^T h_2          (GEMV)
||w||_2                   (NRM2)
```

The CPU receives only `h_1` and `h_2`, accumulates

\[
H_{0:j,j} \leftarrow h_1+h_2,
\]

then updates the small Hessenberg problem with Givens rotations.

No Krylov vector is transferred to the CPU.

## Counters

The GMRES result now records:

```python
result.orthogonalization
result.dot_count
result.axpy_count
result.basis_projection_count
result.basis_correction_count
result.coefficient_d2h_count
```

For CGS2, normally:

```python
result.dot_count == 0
result.basis_projection_count == 2 * result.iterations
result.basis_correction_count == result.basis_projection_count
result.coefficient_d2h_count == result.basis_projection_count
```

AXPY is still used for true-residual construction, so `axpy_count` is not zero.

## Validation

Run the CUDA tests:

```bash
PYTHONPATH=. pytest -q \
    tests/test_cupy_gmres.py \
    tests/test_cupy_profiling.py
```

Compare all four methods on one system:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_orthogonalization.py \
    --mesh 32 \
    --order 3 \
    --restart 30 \
    --max-iterations 1000 \
    --rtol 1e-8 \
    --preconditioner asm
```

The script reports time, iterations, residual, BLAS-operation counts, transfer
counts, and solution differences.

## Systematic T600 comparison

Use the profiling script once for each mode:

```bash
for ORTH in mgs mgs2 cgs cgs2; do
    PYTHONPATH=. python scripts/profile_face_dense_gpu.py \
        --orders 2 3 4 \
        --meshes 16 32 64 \
        --boundary-mode eliminate \
        --matvec-implementations raw \
        --gmres-matvec-implementation raw \
        --preconditioners asm \
        --local-solver cublas_inverse \
        --preconditioner-application raw \
        --orthogonalization "$ORTH" \
        --warmup 10 \
        --repeats 100 \
        --restart 30 \
        --max-iterations 1000 \
        --rtol 1e-8 \
        --detailed-max-iterations 50 \
        --output-prefix "results/t600_${ORTH}"
done
```

Important quantities are:

- total GMRES wall time;
- iteration count and final true residual;
- `dot_count` versus `coefficient_d2h_count`;
- profiler categories `dot`, `basis_projection`, `basis_correction`, and
  `orthogonalization_d2h`;
- agreement between CGS2 and MGS2 solutions.

## Interpretation

CGS2 is an orthogonalization optimization. It should reduce synchronization and
launch overhead, but it does not improve the spectrum of the preconditioned
HDG operator. If GMRES(30) stagnates on refined meshes, larger restart lengths
or stronger preconditioning are still required.
