# Two-kernel fused additive Schwarz on CUDA

## Motivation

The original GPU one-element additive Schwarz application used three numerical stages:

1. restrict the global face vector to an element-local vector;
2. multiply every local vector by the corresponding dense inverse;
3. prolong with atomic additions.

That path is correct, but it stores both the restricted right-hand side and the local correction, launches separate restriction and local-product kernels, and uses atomics during prolongation.

The fused path is selected with:

```python
application="fused"
```

It reduces an inverse-based ASM application to two kernels.

## Kernel 1: fused restriction and local inverse product

For each element-local output row, the kernel directly evaluates

\[
z^e_i = \sum_j (P_e^{-1})_{ij}\,r_{\sigma(e,\ell(j)),d(j)}.
\]

A directly eliminated boundary face has system index `-1` and contributes zero. The restricted vector is never materialized.

The old storage was

\[
2N_E( N_{lfe} b )\,\text{sizeof(scalar)}
\]

for restricted and corrected element vectors. The fused path retains only the corrected vector, cutting this workspace in half.

## Kernel 2: face-owned race-free prolongation

A host setup routine builds

```text
face_element_slots[num_system_faces, 2]
```

Each active entry is a flattened `(element, local_face)` incidence. A boundary face has one entry and an interior manifold face has two.

One CUDA thread owns one global face degree of freedom and sums its one or two local contributions. This removes:

- `atomicAdd`;
- the output zero-fill kernel;
- non-deterministic accumulation order.

Meshes with more than two incident elements at a face are rejected by the fused setup. The original atomic path remains available for non-manifold experiments.

## Profiling semantics

For `application="fused"`, the ASM profile reports:

- `restriction = 0`, because restriction is folded into the local kernel;
- `local_solve`, meaning fused restriction plus inverse multiplication;
- `prolongation`, meaning face-owned race-free gathering;
- `total`, meaning both kernels together.

## Numerical validation

The new CUDA tests compare fused ASM against:

- the validated CPU ASM implementation;
- the existing raw three-stage GPU implementation;
- direct face-dense solutions through restarted GMRES;
- penalty and direct-elimination boundary systems;
- float32 and float64.

Race-free prolongation is also run twice and required to be bitwise repeatable.

## Strengthened solver diagnostics

The numerical validation script now additionally reports, for small systems:

\[
\kappa_2(A),
\qquad
\eta = \frac{\|b-Ax\|_2}{\|A\|_2\|x\|_2+\|b\|_2},
\qquad
\kappa_2(A)\eta.
\]

The residual is recomputed in float64 from a materialized reference matrix. This distinguishes a small backward error from a larger forward error caused by conditioning, which is particularly important for float32 and penalty systems.

## Commands

Standalone ASM and ASM-polynomial comparison:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_fused_asm.py \
    --mesh 64 \
    --order 1 \
    --boundary-mode eliminate \
    --operator raw \
    --restart 75 \
    --polynomial-degree 18
```

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_fused_asm.py \
    --mesh 64 \
    --order 4 \
    --boundary-mode eliminate \
    --operator raw \
    --restart 100 \
    --polynomial-degree 18
```

Systematic profiling:

```bash
PYTHONPATH=. python scripts/profile_face_dense_gpu.py \
    --orders 1 2 3 4 \
    --meshes 32 64 96 128 \
    --matvec-implementations raw_fused raw \
    --gmres-matvec-implementation raw \
    --preconditioners asm \
    --asm-application fused \
    --orthogonalization cgs \
    --restart 75 \
    --max-iterations 2000
```

Polynomial study with fused ASM:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_polynomial.py \
    --mesh 64 \
    --order 1 \
    --degrees 12 14 16 18 20 \
    --methods asm asm_poly \
    --setup-mode shared \
    --asm-application fused \
    --restart 75
```

## Next development stage

The next assembly step should move the construction of the element trace blocks to the GPU. The current GPU global assembler begins with complete element blocks already formed on the CPU. A controlled port should therefore proceed through:

1. GPU transfer of quadrature and geometry tensors;
2. batched construction of local mixed matrices;
3. batched local factorization/solve;
4. local Schur products into complete element trace blocks;
5. the already validated GPU global face assembly.

Ping-pong buffers are most useful in this local-assembly pipeline, where successive tensor contractions and solves can alternate between two reusable element batches.
