# BSR And AMGX Dependency Map

This is the maintained ownership map for HDG Poisson BSR solver paths. It
separates matrix assembly/storage, sparse matrix-vector products (SpMV),
multigrid hierarchy construction, and the outer Krylov solve. Calling a path
"BSR" does not by itself mean that it uses AMGX.

## Component Boundaries

```text
HDGFEM raw-CUDA diffusion assembly
  -> CuPy-resident face-BSR: indptr, indices, dense b x b blocks, rhs
       |
       +-> AMGX paths
       |    -> PyAMGX uploads device buffers and block dimensions
       |    -> libamgxsh owns hierarchy setup, V-cycle, and Krylov solve
       |    -> cuSPARSE BSR SpMV is selected and invoked inside AMGX
       |
       +-> FB-HP-MG prototype
            -> HDGFEM owns Legendre p-levels, PCG/PCGF, and workspaces
            -> HDGFEM invokes cuSPARSE BSR SpMV directly
            -> AMGX supplies only the scalar p=0 h-AMG cycle by default
```

| Layer | Responsibility |
|---|---|
| HDGFEM | HDG algebra, boundary elimination, direct BSR assembly, modal transforms, diagnostics, and experimental p-multigrid/PCG. |
| CuPy/CUDA/cuSPARSE | Device arrays, raw-kernel compilation, and generic BSR SpMV. |
| PyAMGX | Python/C-API bridge that uploads buffers and wraps AMGX handles. It implements no AMG algorithm. |
| AMGX (`libamgxsh.so`) | Krylov iteration, AMG hierarchy construction, smoothers, transfers, Galerkin products, coarse solves, and AMGX-internal cuSPARSE dispatch. |

## Path-By-Path Dependencies

| Path | BSR assembly | Fine BSR action | Hierarchy and solve | PyAMGX/AMGX dependency |
|---|---|---|---|---|
| Direct raw-CUDA BSR assembly | HDGFEM raw CUDA | Not applicable | Not applicable | None |
| Scalar CSR classical AMG | HDGFEM raw CUDA | Scalar CSR inside AMGX | AMGX classical AMG and PCGF | Complete |
| Hybrid fine-BSR/scalar-hierarchy AMG | HDGFEM raw CUDA | cuSPARSE invoked inside AMGX | Modified AMGX scalar expansion and classical hierarchy | Complete |
| Pure-BSR identity hierarchy | HDGFEM raw CUDA | cuSPARSE invoked inside AMGX | Modified AMGX block graph, `w_ic I_b` transfers, BSR Galerkin products, and Krylov solve | Complete |
| Pure-BSR dense hierarchy | HDGFEM raw CUDA | cuSPARSE invoked inside AMGX | Modified AMGX block graph, dense transfers, BSR Galerkin products, and Krylov solve | Complete |
| FB-HP-MG, CuPy reference smoother | HDGFEM raw CUDA | Direct HDGFEM cuSPARSE wrapper | HDGFEM p-level V-cycle and PCG; AMGX h-AMG at p=0 | Only default p=0 correction |
| FB-HP-MG, fused smoother | HDGFEM raw CUDA | cuSPARSE for standalone actions; fused HDGFEM traversal in smoothing | HDGFEM p-level V-cycle and PCG; AMGX h-AMG at p=0 | Only default p=0 correction |
| FB-HP-MG, diagnostic CuPyX p=0 CG | HDGFEM raw CUDA | Direct HDGFEM cuSPARSE wrapper | HDGFEM p-levels and CuPyX scalar CG | None |

The direct assembly kernel produces the same face graph and dense blocks for
the AMGX and FB-HP-MG branches. AMGX is downstream of assembly and is not
needed to form, inspect, transform, or directly apply the matrix.

## Hybrid AMGX

The coefficient-exact hybrid uploads the fine matrix to AMGX as square blocks
of size `b=p+1`. During AMGX setup, the modified classical hierarchy temporarily
expands it into an exactly equivalent scalar CSR operator. Classical strength,
PMIS/aggressive selection, D2 interpolation, and Galerkin products use that
scalar representation. The retained fine operator remains BSR; retained
transfers and coarse operators are scalar CSR.

During a solve, AMGX:

1. applies the fine matrix in BSR form;
2. invokes generic cuSPARSE when
   `bsr_spmv_backend="cusparse_generic"` is selected;
3. connects the block fine vector to scalar transfer operators without changing
   its coefficients;
4. executes all coarse work in scalar CSR and owns the outer PCGF state.

Thus this path has no retained fine CSR matrix during the solve, but it is
neither AMGX-free nor a pure-BSR hierarchy.

## Pure-BSR AMGX

`block_graph_identity` and `block_graph_dense` are implemented in the local
AMGX source, not in HDGFEM or PyAMGX. They use a compact scalar graph during
setup but retain block matrices and block transfers in the accepted hierarchy.

- `block_graph_identity` lifts weights as `P_ic = w_ic I_b`.
- `block_graph_dense` constructs dense `b x b` interpolation blocks and applies
  its configured block constraint.
- AMGX owns block transfers, weighted BSR Galerkin products, smoothing, the
  coarse solve, and the outer Krylov method.

These modes require the matching locally built AMGX library. A stock PyAMGX
binding does not supply them: PyAMGX only forwards configuration and device
buffers to `libamgxsh.so`.

## Face-Block p/h-Multigrid

FB-HP-MG moves the high-order hierarchy and outer Krylov method out of AMGX.
HDGFEM owns the normalized Legendre basis, principal-block Galerkin p-levels,
modal transfers, dense block inverses, Chebyshev weights, persistent V-cycle
buffers, FP64 PCG/diagnostic PCGF, and independent residual/symmetry checks.

The production prototype currently delegates only the `p=0` face-average
operator to `AmgxScalarVcycle`. It uploads a block-size-one operator through
PyAMGX, constructs one reusable scalar classical-AMG hierarchy, and applies
exactly one zero-initialized AMGX cycle per coarse correction.

`CupyxCgScalarSolve` is an AMGX-free diagnostic replacement. It proves that the
p-level architecture is not tied to AMGX, but it is not yet the efficient,
scalable scalar h-multigrid replacement needed for production.

## Two Independent cuSPARSE Routes

1. **AMGX-owned cuSPARSE:** AMGX reads
   `bsr_spmv_backend="cusparse_generic"`, creates/caches descriptors, invokes
   cuSPARSE, and owns the surrounding solver state. PyAMGX and AMGX are needed.
2. **HDGFEM-owned cuSPARSE:** `LegendreFaceBsrOperator` uses a narrow binding
   around CuPy's cuSPARSE handle. HDGFEM owns descriptors, preprocessing,
   workspace, and the stream restriction. AMGX and PyAMGX do not participate.

These routes call the same CUDA library but share no descriptors, workspace,
hierarchy state, or configuration switches.

## Remaining Custom CUDA Kernels

"Use cuSPARSE for BSR SpMV" does not mean "there are no custom CUDA kernels."

| Kernel | Owner | BSR traversal | AMGX dependency | Status |
|---|---|---:|---:|---|
| Direct reduced BSR assembly | HDGFEM | Writes BSR | None | Required assembly path |
| Generic standalone BSR SpMV | cuSPARSE through HDGFEM | Yes | None | Default standalone action |
| AMGX BSR SpMV | cuSPARSE through AMGX | Yes | Complete | Default AMGX action |
| Row-owned raw BSR SpMV | HDGFEM `RawKernel` | Yes | None | Availability/parity fallback, not selected production action |
| Fused block-Jacobi--Chebyshev | HDGFEM `RawKernel` | Yes | None | Selected p-level smoother optimization |
| Zero-start block-Jacobi | HDGFEM `RawKernel` | No | None | Selected first-stage optimization |
| Directly restricted residual | Removed | Formerly | None | Rejected after failing to beat cuSPARSE consistently |

The fused smoother performs `A*x`, residual formation, the dense diagonal-block
inverse, and the update in one kernel. It is not a general SpMV backend, but it
is custom BSR traversal. Forbidding every custom traversal would require
separate cuSPARSE SpMV, residual, dense-block, and update operations and would
forfeit the measured smoother-fusion gain.

The raw SpMV remains only as a diagnostic/availability fallback. A strict
cuSPARSE-only standalone run should request `backend="cusparse"`; it then raises
rather than silently using the raw fallback.

## Identifying The Active Runtime Path

- `raw_matrix_format="bsr"` describes only assembly/storage.
- AMGX `classical_bsr_hierarchy="scalar_expand"` selects the hybrid hierarchy;
  it is also the default BSR hierarchy mode.
- `block_graph_identity` or `block_graph_dense` selects pure-BSR AMGX.
- AMGX `bsr_spmv_backend="cusparse_generic"` affects only work inside AMGX.
- `LegendreFaceBsrOperator.backend_used="cusparse-generic-bsr"` identifies the
  independent HDGFEM cuSPARSE route.
- FB-HP-MG reports `spmv_backend` and `smoother_backend` separately.
- Coarse diagnostics must state `AMGX scalar p=0` or `CuPyX scalar CG`; the fine
  backend alone does not reveal the coarse dependency.

## Removing AMGX Completely

Direct BSR assembly and all p-level machinery already work without AMGX. A
production AMGX-free FB-HP-MG needs only an efficient reusable scalar p=0
h-hierarchy providing coarsening, interpolation and transpose restriction,
Galerkin coarse operators, symmetric smoothing, an SPD terminal solve,
persistent device storage, and fixed-work application suitable for PCG.

Removing AMGX from the historical hybrid or pure-BSR solvers is much larger: it
would require reimplementing their hierarchy construction, V-cycle, and outer
Krylov orchestration, not merely replacing SpMV.
