# Face-dense CUDA restarted GMRES

This stage moves the Krylov vectors and Arnoldi basis to the GPU while retaining
the small Hessenberg problem on the CPU.

## Data placement

GPU:

- face-dense operator blocks and neighbors;
- right-hand side and solution;
- residual, work vectors, and matrix-vector workspace;
- Arnoldi basis `V[restart + 1, Ndof]`;
- the short restart coefficient vector used by the final basis update.

CPU:

- Hessenberg matrix `H[restart + 1, restart]`;
- Givens cosines and sines;
- transformed least-squares right-hand side;
- upper-triangular back substitution.

## GPU BLAS operations

`CuPyVectorBLAS` explicitly uses `cupy.cublas`:

- `dot` for Arnoldi coefficients;
- `nrm2` for residual and Arnoldi norms;
- `axpy` for modified Gram--Schmidt corrections;
- `scal` for basis normalization;
- `gemv` for `x += V.T @ y` at the end of a restart cycle.

Every dot product and norm produces one scalar required immediately by the CPU
Hessenberg update, so the first implementation accepts that synchronization.
A later CGS2 path can collect all coefficients with a GEMV and reduce the number
of synchronizations.

## Convergence policy

The Givens residual is a preconditioned residual estimate. It is used only as a
cheap trigger. A true residual `||b - A x||` is recomputed after every completed
or early-terminated restart cycle before convergence is accepted.

## Block-Jacobi

`CuPyFaceBlockJacobiPreconditioner` stores the inverse diagonal face blocks on
the GPU and applies them using batched dense matrix-vector multiplication. In
this first correctness version, the already validated CPU setup computes the
small inverses and transfers them once. GPU batched factorization is a later
setup optimization and does not change the GMRES interface.

## Commands

```bash
PYTHONPATH=. pytest -q tests/test_cupy_gmres.py
PYTHONPATH=. python scripts/validate_face_dense_gpu_gmres.py
```

The actual CUDA tests skip automatically when CuPy or a usable GPU is absent.
