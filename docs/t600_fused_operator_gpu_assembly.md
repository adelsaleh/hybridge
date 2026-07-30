# T600 development stage: fused face operator and GPU global assembly

This stage is designed for development on the NVIDIA T600 before limited runs
on the mesocentre GPU.

## 1. Fused face-dense matvec

The previous raw path used two CUDA kernels:

1. gather all neighbour-face vectors into `x_extended[NF,S,b]`;
2. multiply each dense face row by the gathered vector.

The new `raw_fused` path performs both operations in one face-row kernel:

\[
y_{f,i}=\sum_{s=0}^{S-1}\sum_{j=0}^{b-1}
K_{f,i,s,j}\,x_{\mathrm{neighbors}[f,s],j}.
\]

Unused neighbour slots are skipped. The fused path removes:

- one kernel launch per matrix-vector product;
- the complete write and reread of `x_extended`;
- the `NF*S*b` gather workspace.

It is selected with:

```python
operator = CuPyFaceDenseOperator.from_system(
    system,
    implementation="raw_fused",
)
```

The original `raw` and `matmul` paths remain available as independent
validation references.

## 2. Current buffer architecture

The solver already uses fixed reusable storage rather than per-iteration
allocation.

### GMRES

- `matvec_buffer`: output of the global face operator;
- `work`: preconditioned Arnoldi candidate;
- `preconditioned`: preconditioned cycle residual;
- `residual`: true residual;
- `basis`: immutable Arnoldi rows during a restart cycle.

This is not a simple two-vector ping-pong algorithm because GMRES must retain
all basis vectors. The work vectors nevertheless alternate roles without new
allocation.

### Polynomial preconditioner

- `q`: current recurrence vector;
- `Bq`: first operator image;
- `B2q`: second image needed only for a conjugate pair;
- `operator_output`: temporary `Aq` when a base preconditioner is present.

Real-root and conjugate-pair updates are fused CUDA kernels. Updating `q` in
place is cheaper than a literal `q_old/q_new` ping-pong pair because the kernel
already reads each old value before writing the new one. A second full-size
`q_next` buffer would add memory traffic without removing an operator call.

## 3. GPU global face assembly

`CuPyGlobalFaceAssembler` assembles complete elemental blocks

\[
A^e_{r,c}\in\mathbb{R}^{b\times b}
\]

directly into global fixed-width face rows.

A global face row is incident to at most two elements on the supported
manifold triangular meshes. Setup builds a contribution table

```text
(element, row_local_face, column_local_face)[global_face, slot, side]
```

with two possible contributions. The CUDA kernel assigns one thread to each
scalar global block entry and gathers at most two values. This avoids atomics,
has deterministic accumulation order, and preserves the existing face-dense
layout.

This stage moves the global accumulation to the GPU. It does **not yet** move
quadrature, local mixed-matrix construction, local inversion, trace lifting, or
Schur condensation to the GPU. Those are the next assembly layers.

## 4. Direct GPU assembly-to-operator path

`CuPyFaceDenseOperator.from_device_blocks` accepts GPU-resident blocks with
shape `(NF,S,b,b)`. It performs the one-time device-side permutation to
`(NF,b,S*b)` and never copies the assembled matrix back to the CPU.

## 5. Validation

Run:

```bash
PYTHONPATH=. pytest -q \
    tests/test_cupy_face_dense.py \
    tests/test_cupy_assembly.py \
    tests/test_cupy_gmres.py \
    tests/test_cupy_polynomial.py
```

Then benchmark fusion and global assembly:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_fusion_assembly.py \
    --mesh 64 \
    --order 1 \
    --dtype float64 \
    --warmup 20 \
    --repeats 500
```

Repeat at `p=4` and on at least one smaller mesh. The T600 results decide
whether `raw_fused` becomes the default before mesocentre runs.

## 6. Development sequence before mesocentre access

1. Fused face matvec and deterministic GPU global assembly.
2. Fused vector updates in GMRES: residual formation and copy-normalize.
3. NVTX ranges and reproducible Nsight Systems/Compute run configurations.
4. GPU construction of complete elemental blocks.
5. GPU batched local Schur condensation and trace-block construction.
6. GPU boundary RHS construction, penalty row replacement, and direct
   elimination.
7. Strong validation matrix across dtype, order, mesh, boundary treatment,
   restart, preconditioner, and polynomial degree.
8. Mesocentre runs restricted to architecture-specific tuning and final scale.

The T600 should be used to settle algorithms, memory ownership, launch count,
correctness, and benchmark automation. The mesocentre allocation should not be
spent discovering basic CUDA bugs.

## 7. Strong numerical regression matrix

The small direct-reference matrix can be run entirely on the T600:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_numerics.py \
    --meshes 2 4 8 \
    --orders 1 2 3 \
    --boundary-modes eliminate penalty \
    --dtypes float64 \
    --methods none block_jacobi asm asm_poly \
    --polynomial-degree 8 \
    --restart 50 \
    --max-iterations 2000 \
    --rtol 1e-8 \
    --strict \
    --output-prefix results/t600_numerical_regression
```

This checks, for every configured case:

- deterministic GPU global assembly against NumPy assembly;
- `raw`, `raw_fused`, and CPU face matvec agreement;
- penalty and direct-elimination systems;
- GMRES true residual;
- trace solution against a direct dense solve;
- block-Jacobi, ASM, and ASM-polynomial variants.

Float32 should be run as a separate campaign with an appropriate tolerance,
for example `--dtypes float32 --rtol 2e-5`. It must not be mixed into claims
about the float64 production accuracy.
