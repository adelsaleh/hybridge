# Reusable GPU GMRES workspace and multi-right-hand-side solves

## Motivation

The face-dense matrix, additive-Schwarz factors, harmonic-Ritz probe, and
polynomial roots are independent of the right-hand side.  They should be built
once and reused.  The previous GMRES entry point nevertheless allocated a new
Arnoldi basis and work vectors on every call.  CuPy's memory pool makes repeated
allocation cheaper than raw `cudaMalloc`, but it still creates Python/CuPy array
objects and obscures the real steady-state cost of solving several right-hand
sides.

This stage separates numerical setup from solve storage.

## `CuPyGMRESWorkspace`

A workspace owns the large device arrays

\[
V\in\mathbb{R}^{(m+1)\times N},
\qquad
w_A,r,w,M^{-1}w\in\mathbb{R}^{N},
\]

and the short device coefficient vectors used by CGS/CGS2 and the final basis
update.  It also owns reusable CPU arrays for

\[
H\in\mathbb{R}^{(m+1)\times m},
\]

Givens rotations, the least-squares right-hand side, and coefficient transfer
buffers.

```python
from hdgfem.backends.cupy_gmres import CuPyGMRESWorkspace

workspace = CuPyGMRESWorkspace.allocate(
    num_dofs=operator.num_dofs,
    restart_capacity=100,
    dtype=operator.dtype,
    device_id=operator.device_id,
)
```

A solve may use any restart not larger than `restart_capacity`.

```python
result = restarted_gmres_cupy(
    operator,
    rhs,
    restart=75,
    preconditioner=asm_polynomial,
    orthogonalization="cgs",
    workspace=workspace,
    solution_out=solution,
)
```

`solution_out` avoids allocating the solution vector.  It must have the same
shape, dtype, and device as `rhs`, and it must not overlap `rhs` or any workspace
array.

## `CuPyRestartedGMRESSolver`

The reusable wrapper stores the numerical configuration and workspace:

```python
from hdgfem.backends.cupy_gmres import CuPyRestartedGMRESSolver

solver = CuPyRestartedGMRESSolver(
    operator,
    restart=100,
    max_iterations=2000,
    rtol=1.0e-8,
    preconditioner=asm_polynomial,
    orthogonalization="cgs",
)

solutions = cp.empty((number_of_rhs, operator.num_dofs), dtype=operator.dtype)
for index in range(number_of_rhs):
    result = solver.solve(
        right_hand_sides[index],
        solution_out=solutions[index],
    )
```

The returned solution is a view of `solution_out`.  Reusing the same output
vector overwrites the preceding solution, so callers that need all solutions
must provide separate rows, as in the example above.

The solver and workspace are sequential objects.  They are not safe for
concurrent solves because every solve reuses the same basis and work buffers.

## Storage

The dominant workspace storage is the Arnoldi basis:

\[
8(m+1)N\ \text{bytes}
\]

in double precision.  Four full-size vector buffers add approximately

\[
32N\ \text{bytes}.
\]

The exact amount is available as

```python
solver.workspace_device_bytes
```

or

```python
workspace.device_bytes
```

## Multi-RHS benchmark

The validation script builds the following objects once:

1. face-dense operator;
2. additive-Schwarz preconditioner;
3. shared harmonic-Ritz probe;
4. one polynomial candidate;
5. reusable GMRES workspace.

It then solves scaled versions of the same right-hand side.  Scaling preserves
the Krylov convergence pattern while ensuring that the output changes for every
solve.

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_multi_rhs.py \
    --mesh 64 \
    --order 4 \
    --degree 18 \
    --restart 100 \
    --rhs-counts 1 2 5 10 20 \
    --repeats 3 \
    --output-prefix results/t600_multi_rhs_64_p4
```

The script compares:

- `legacy`: a fresh GMRES workspace is constructed inside every solver call;
- `reused`: one workspace and preallocated solution matrix are reused.

Both modes reuse the matrix, ASM factors, harmonic-Ritz probe, and polynomial
preconditioner.  The comparison therefore isolates solve-storage management.

## Correctness checks

CUDA tests verify that:

- reusable and legacy GMRES produce the same solution;
- one workspace can solve several scaled right-hand sides;
- all workspace device pointers remain unchanged;
- a preallocated output is returned directly;
- insufficient restart capacity is rejected;
- right-hand-side, solution, and workspace aliasing are rejected.

The existing API remains backward compatible.  Calls without `workspace` and
`solution_out` follow the previous allocation behavior.
