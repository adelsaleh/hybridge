# Experimental Face-Dense GPU Solver

## Scope

This path preserves the natural HDG trace blocking for two-dimensional
triangular diffusion problems. A global face row stores at most five dense
blocks of shape `(p + 1, p + 1)` plus its neighbor-face indices. It is a
research solver and benchmark path; it does not replace the supported sparse
CSR/BSR backends.

The imported implementation is the validated subset of historical commit
`ea5ad26281f9e988194d9352399ccc4354a6633e`. Its dated Poisson results are in
[`../research/solver_studies/amgx_vs_face_dense_2026_08.md`](../research/solver_studies/amgx_vs_face_dense_2026_08.md)
and
[`../research/solver_studies/face_dense_primitives_2026_08.md`](../research/solver_studies/face_dense_primitives_2026_08.md).

## Ownership and dependencies

The face-dense solver does **not** depend on AMGX or PyAMGX:

- `hdgfem.mixed.face_dense` owns fixed-slot face topology, face assembly, and
  boundary elimination/penalty normalization.
- `hdgfem.linalg.face_dense` owns `FaceDenseSystem`, reference matvecs,
  residuals, and validation materialization.
- `hdgfem.solvers.diffusion_face_dense` builds the face-dense diffusion system
  from the mixed local solvers.
- `hdgfem.linalg.gmres`, `additive_schwarz`, `block_jacobi`, and `polynomial`
  provide NumPy reference algorithms.
- `hdgfem.linalg.gpu.face_dense` owns the CuPy operator and its raw/fused
  device variants.
- `hdgfem.linalg.gpu.gmres` owns restarted device GMRES.
- `hdgfem.linalg.gpu.preconditioners` owns face block-Jacobi and
  element-patch additive Schwarz.
- `hdgfem.linalg.gpu.polynomial` owns Arnoldi/harmonic-Ritz setup and the
  polynomial preconditioner.
- `hdgfem.linalg.gpu.cublas_batched` supplies optional batched dense inverses;
  `hdgfem.linalg.gpu.production_gmres` composes the production-oriented
  experimental interface and convergence safeguards.

CuPy supplies device arrays and kernel compilation, while cuBLAS is used for
selected dense operations. AMGX enters only as an independently timed
comparison solver in the dated study; it is not called by the face-dense
operator, ASM, polynomial preconditioner, or GMRES iteration.

## Data flow

1. Assemble element trace contributions into fixed face rows.
2. Eliminate prescribed Dirichlet faces or normalize penalty rows.
3. Construct `CuPyFaceDenseOperator` without materializing scalar CSR.
4. Build block-Jacobi or additive-Schwarz data, optionally wrap it in the
   polynomial preconditioner, and run restarted GMRES.
5. Expand an eliminated trace and reconstruct the HDG volume fields.

CPU implementations and small dense materialization remain correctness
references only. The hot device path must avoid per-application allocation,
implicit host transfers, and scalar CSR conversion.

## Validation and benchmark entry points

Use the focused tests:

```bash
python -m pytest -q   tests/test_face_dense.py   tests/test_face_additive_schwarz.py   tests/test_face_block_jacobi.py   tests/test_restarted_gmres.py   tests/test_polynomial_preconditioner.py   tests/test_cublas_batched_inverse.py   tests/test_cupy_face_dense.py   tests/test_cupy_gmres.py   tests/test_cupy_additive_schwarz.py   tests/test_cupy_polynomial.py   tests/test_cupy_solver.py
```

The canonical runners are:

```bash
python -m scripts.diffusion_reaction.validate_face_dense_gpu_solver --help
python -m scripts.diffusion_reaction.benchmark_face_dense_primitives --help
```

The benchmark runner writes machine-readable JSON. Dated timings must remain
in `docs/research/solver_studies/`; they are evidence, not current defaults.
