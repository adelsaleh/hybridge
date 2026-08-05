# Diffusion Assembly Study: July 2026

Status: historical performance research, not a backend recommendation.

## Question

The study compared vectorized NumPy and fused Numba construction of the same
boundary-eliminated diffusion-reaction HDG trace operator. It measured whether
forming local condensed blocks in compiled element loops reduced the temporary
storage and memory traffic introduced by global NumPy tensor materialization.

The numerical workflows and equivalence requirements are maintained in the
[diffusion-reaction assembly derivation](../../algorithms/diffusion_reaction/assembly.tex).

## Setup

- rectangle mesh with 92,552 triangles at target mesh size `0.01`;
- polynomial orders 3 through 7;
- five timed repeats;
- `trace_ordering=none`;
- first-call Numba compilation excluded;
- Intel Xeon w5-3535X, 20 physical and 40 logical CPUs;
- Numba TBB, OpenBLAS, and OpenMP each allowed 40 logical threads.

The timing scope included local-factor construction, local inverse construction
or fused local inverse action, trace COO/RHS assembly, and Dirichlet trace
elimination. It excluded mesh generation, sparse-matrix construction for the
global solve, ILU/Krylov work, error evaluation, and plotting.

## Results

| Order | NumPy assembly | Numba assembly | Speedup | NumPy peak | Numba peak |
|---:|---:|---:|---:|---:|---:|
| 3 | 3.069 s | 0.096 s | 31.81x | 2,312 MiB | 449 MiB |
| 4 | 6.306 s | 0.166 s | 38.03x | 4,609 MiB | 685 MiB |
| 5 | 11.333 s | 0.349 s | 32.47x | 9,032 MiB | 972 MiB |
| 6 | 19.480 s | 0.644 s | 30.26x | 16,056 MiB | 1,310 MiB |
| 7 | 30.606 s | 1.064 s | 28.77x | 26,540 MiB | 1,699 MiB |

Peak memory is the Python-tracked peak reported by the benchmark harness, not a
complete process or system memory measurement.

## Interpretation

The local scalar dimension was 10 through 36 and the mixed local block
dimension was 30 through 108. This is a many-small-dense-operations regime,
rather than one large BLAS-3 operation.

The NumPy workflow exposed those operations by allocating element batches for
local inverses, boundary blocks, trace-lifted Schur blocks, orientation copies,
COO data, and RHS work. The fused Numba workflow kept most intermediate state
element-local, applied orientation before insertion, and wrote reduced entries
directly. The observed improvement therefore came primarily from execution
model and memory traffic, not from granting Numba a larger CPU budget.

These measurements justified retaining the fused host assembly path, but they
do not establish current timings on other hardware, meshes, coefficient types,
orders, or thread allocations.

## Reproduction

The historical comparison driver remains:

```bash
.venv/bin/python scripts/advection_reaction/compare_assembly_backends.py \
  --problem diff --domain rectangle --lc 0.01 \
  --mode eliminate --trace-ordering none --repeats 5 \
  --order 6 --basis dub_orth --breakdown
```

Repeat with `--order 3` through `--order 7`. Record the generated element count,
thread environment, warmup policy, timing distribution, and process-level
memory if using new runs to support a current recommendation.
