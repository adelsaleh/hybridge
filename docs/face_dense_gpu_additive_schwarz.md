# CUDA one-element additive Schwarz for the face-dense HDG system

## Scope

This stage ports only the **application** of the already validated one-element
additive-Schwarz preconditioner to CUDA. The local Schwarz matrices are still
constructed and inverted on the CPU during setup. Their inverses and the
restriction map are transferred once to the GPU.

No COO or CSR matrix is used.

## Algebra

For each element subdomain \(e\), let \(R_e\) restrict the global face vector to
the element's local faces and let \(P_e\) be the enriched dense local Schwarz
matrix. The preconditioner is

\[
M_{\mathrm{ASM}}^{-1}
= \sum_e R_e^T P_e^{-1}R_e.
\]

Its application is split into three GPU stages:

1. Restriction:
   \[
   r_e = R_e r.
   \]
2. Batched local dense products:
   \[
   z_e = P_e^{-1}r_e.
   \]
3. Prolongation with overlap accumulation:
   \[
   z = \sum_e R_e^Tz_e.
   \]

For triangular elements and \(b\) trace unknowns per face, every local matrix
has size \((3b)\times(3b)\).

## Device arrays

The CUDA preconditioner stores

```text
inverse_matrices     (NE, Nlfe*b, Nlfe*b)
element_system_faces (NE, Nlfe)          int32
```

The second array maps each element-local face to the row numbering of the
selected face-dense system. In the direct-elimination formulation, prescribed
boundary faces are represented by `-1`.

Two iteration workspaces are allocated once:

```text
element_rhs      (NE, Nlfe*b)
local_solution   (NE, Nlfe*b, 1)
```

## Restriction kernel

One CUDA thread handles one element-local trace degree of freedom. It reads

```text
element_system_faces[e, lf]
```

and either gathers the corresponding global face value or writes zero when the
mapping is `-1`.

## Local dense products

The local inverse application is evaluated by

```python
cupy.matmul(
    inverse_matrices,
    element_rhs[..., None],
    out=local_solution,
)
```

This is the correctness-oriented batched dense path. A later setup stage can
replace explicit inverses with batched LU factors and solves without changing
the restriction or prolongation kernels.

## Prolongation kernel

One CUDA thread again handles one element-local degree of freedom. Active
entries are accumulated into the global output using `atomicAdd`, because an
interior face belongs to two element subdomains.

The implementation includes a double-precision atomic fallback for devices
with compute capability below 6.0.

## Boundary formulations

### Direct Dirichlet elimination

Eliminated boundary faces have mapping `-1`. Restriction writes zero, and
prolongation ignores their local values. The CPU builder inserts identity blocks
for those inactive local rows and columns so that all element matrices retain a
uniform size.

### Penalty rows

Boundary trace unknowns remain in the global system. The CPU builder reproduces
the penalty row in each local matrix while preserving boundary columns in
interior rows. The GPU restriction and prolongation therefore use ordinary
active face indices.

## Usage

```python
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_preconditioners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
)

system = face_assembly.eliminated_system

operator = CuPyFaceDenseOperator.from_system(system)
asm = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
    system,
    face_assembly.element_blocks,
    space.mesh.loc2glob_edge,
    device_id=operator.device_id,
)

rhs_gpu = operator.to_device(system.rhs)
result = restarted_gmres_cupy(
    operator,
    rhs_gpu,
    restart=30,
    max_iterations=1000,
    rtol=1.0e-10,
    preconditioner=asm,
    reorthogonalize=True,
)
```

The solution remains on the device until an explicit transfer is requested.

## Validation

CPU-only tests validate the host batch layout against the existing CPU ASM
builder. CUDA tests, when a device is available, compare:

- GPU restriction against CPU restriction;
- GPU atomic prolongation against CPU scatter-add;
- GPU ASM application against the CPU ASM action;
- GPU ASM-GMRES against the direct face-dense solution;
- both penalty and directly eliminated boundary systems;
- flat and face-major vectors;
- reuse of preallocated iteration buffers.

Run:

```bash
PYTHONPATH=. pytest -q tests/test_cupy_additive_schwarz.py
```

For a readable report:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_additive_schwarz.py
```

## Deliberate limitations of this stage

- Local matrices are assembled and inverted on the CPU.
- Explicit inverse matrices are stored.
- Prolongation uses atomics rather than a face-owned reduction strategy.
- No performance claim should be made before the CUDA validation and profiling
  stages are completed.
