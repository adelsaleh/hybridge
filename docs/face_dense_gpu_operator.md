# First GPU layer for the HDG face-dense operator

This stage moves only the condensed face operator to CUDA. Assembly, boundary
elimination, and the CPU validation solvers remain unchanged.

The production data path is

```text
FaceDenseSystem on CPU
        |
        | one setup transfer
        v
matrix_batches on GPU  (NF, b, S*b)
neighbors on GPU       (NF, S)
        |
        | every Krylov matvec
        v
neighbor gather        x_extended[f,s,:] = x[neighbors[f,s],:]
        |
        v
batched dense product  y[f,:] = matrix_batches[f] @ x_extended[f].ravel()
```

No COO or CSR matrix is constructed on the device.

## Installation

CuPy is optional because its wheel must match the CUDA toolkit available on the
machine. For a CUDA 12 installation, for example:

```bash
python -m pip install cupy-cuda12x
```

Use the corresponding CuPy package for another CUDA version.

## Creating the operator

```python
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator

system = face_assembly.eliminated_system

operator = CuPyFaceDenseOperator.from_system(
    system,
    implementation="matmul",
)

rhs_gpu = operator.to_device(system.rhs)
x_gpu = operator.to_device(
    np.zeros_like(system.rhs),
)
y_gpu = operator.matvec(x_gpu)
```

`implementation="matmul"` uses `cupy.matmul` on the batch
`(NF, b, S*b) @ (NF, S*b, 1)`. CuPy dispatches this operation to its CUDA dense
linear-algebra backend. `implementation="raw"` uses an independent simple CUDA
kernel and exists primarily for correctness comparisons.

## Allocation-free iteration interface

The convenience method `matvec` allocates the output. GMRES should allocate its
work vectors once and call:

```python
operator.matvec_into(v_gpu, w_gpu)
```

The operator itself owns and reuses the gathered vector buffer
`operator.x_extended`. The input and output may not alias.

All Krylov vectors passed to the operator must:

- already be CuPy arrays;
- reside on the same CUDA device as the operator;
- be C-contiguous;
- use the operator's `float32` or `float64` dtype;
- have shape `(NF*b,)` or `(NF, b)`.

These checks prevent accidental CPU/GPU copies inside a GMRES iteration.

## Layout transformation performed at setup

The CPU array is

```text
blocks[f, s, i, j]          shape (NF, S, b, b)
```

The GPU dense batch is prepared once as

```text
matrix_batches[f, i, s*b+j] shape (NF, b, S*b)
```

Thus each face matrix is the horizontal concatenation

```text
[K_f,0  K_f,1  ...  K_f,S-1].
```

The gather kernel writes

```text
x_extended[f, s, j] = x[neighbors[f,s], j]
```

and writes zero for an unused `-1` slot.

## Validation

CPU-only layout tests run on every machine. GPU tests are skipped when CuPy or
a CUDA device is unavailable:

```bash
PYTHONPATH=. pytest -q tests/test_cupy_face_dense.py
```

On a GPU machine, run the readable comparison of the CuPy dense-matmul path,
the raw CUDA path, and the NumPy reference:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_operator.py
```

The test covers both the penalty system and the directly eliminated Dirichlet
system, flat and face-major vectors, unused-neighbour padding, and reuse of a
preallocated output vector.

## Deliberate limits of this stage

This code does not yet implement:

- GPU Arnoldi or restarted GMRES;
- GPU dot products, norms, or AXPY operations;
- GPU block-Jacobi or additive Schwarz;
- direct GPU assembly of the face matrix;
- profiling or kernel selection.

Those should be added only after this operator gives roundoff-level agreement
with `face_dense_matvec` on the target GPU.
