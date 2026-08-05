# GPU Assembly And Solve Paths

This note maps the currently supported GPU-oriented HDG assembly and solve paths. It is a roadmap document, not a public API guarantee.

## Current Paths

| Path | Assembly storage | Global solve handoff | Current role |
| --- | --- | --- | --- |
| CuPy assembly | CuPy COO/CSR construction from vectorized device arrays | Cupyx sparse solvers or PyAMGX after CSR construction | Reference GPU backend and correctness comparison path |
| Raw-CUDA COO | Raw kernels emit reduced COO triplets and RHS | Converted to CSR before iterative solve/AMGX | Simple debug/correctness path paired with CSR tests |
| Raw-CUDA CSR | Raw kernels emit reduced CSR data directly into the known reduced sparsity pattern | Direct device CSR view can be passed to PyAMGX | Preferred production direction for large GPU runs |
| Direct CSR-to-AMGX | Device CSR arrays are wrapped without staging through SciPy/CuPy COO-to-CSR reconstruction | PyAMGX `upload_CSR` consumes the device CSR view | Avoids the expensive COO-to-CSR reconstruction bottleneck |
| Cupyx solver path | CuPy CSR matrix plus cupyx Krylov/preconditioner objects | Cupyx iterative solvers | Experimental solver/preconditioner comparison path |

## Preferred Raw-CUDA Target

The preferred raw-CUDA target is fused/cooperative local assembly with direct CSR emission and direct AMGX handoff. In diffusion this means the cooperative element kernel builds local diffusion blocks on the fly, solves the local condensed systems, applies boundary elimination, and writes the reduced trace operator directly as COO or CSR. In advection-reaction the long-term target is the fused cooperative-LU path with direct CSR writes.

The direct CSR path is the release-quality direction because it avoids both large temporary local dense tensors and global COO-to-CSR reconstruction. COO emission remains valuable because it is easier to inspect and is kept paired with CSR equivalence tests.

## Compatibility And Debug Paths

The advection-reaction `safe` LU mode and `precomputed` raw assembly mode are compatibility/debug paths. They should remain available while cooperative-LU/direct-CSR is still being validated, but new performance work should focus on the fused cooperative path.

The diffusion raw-CUDA path currently supports identity diffusion, scalar zero reaction, nodal `legacy-lagrange` trace coordinates, and `p <= 6`. Nonzero reaction tables, tensor diffusion, per-face stabilization tables, and non-legacy trace bases remain implementation work.

## Tensor Diffusion Plan

Tensor diffusion should extend the validated scalar diffusion path without changing the preferred solve handoff. The raw-CUDA kernel should keep direct COO/CSR emission and direct CSR-to-AMGX as the target, while adding an explicit device table for the symmetric tensor entries sampled or projected on the volume quadrature rule. The initial scope should use three scalar entries per quadrature point, `(a00, a01, a11)`, with identity diffusion represented by the existing scalar fast path.

The local mixed block construction is the main kernel change. The kernel must replace the identity-gradient contractions by tensor-weighted contractions, keep the cooperative local factorization/solve unchanged where possible, and measure shared-memory pressure separately for `p <= 6`. Boundary trace handling can remain unchanged for the first tensor step if the HDG numerical flux still uses scalar `tau`; per-face tensor-dependent stabilization should remain a later item.

Validation should be staged in the same order as the scalar path: NumPy/CuPy reference parity for a constant anisotropic tensor, raw-CUDA COO/CSR parity against the reference on small deterministic meshes, then manufactured tensor-diffusion solves with direct CSR/AMGX timing. Only after that should variable tensor tables and per-face `tau` tables be treated as production support.

## Direct CSR Timing Anchor

A paired diffusion raw-CUDA/AMGX run on 2026-07-25 used `p=6`, `mesh_size=0.08`, `legacy-lagrange`, symmetric volume quadrature, block size 128, and the PCGF Chebyshev/L1 AMGX config. The matrix had 10,412,451 nonzeros and 298,599 reduced trace dofs.

| Raw matrix format | Matrix handoff | CSR/view time | AMGX setup/upload | AMGX solve | AMGX subtotal | Raw assembly total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CSR | direct device CSR view | 0.00019 s | 0.53684 s | 0.17208 s | 0.709 s | 0.245 s |
| COO | CuPy COO-to-CSR reconstruction | 0.28348 s | 0.50783 s | 0.17192 s | 0.963 s | 0.210 s |

The direct CSR path removed essentially all CSR reconstruction time. On this medium run, the direct handoff saved about 0.283 s in matrix conversion and about 0.254 s in the AMGX subtotal, while solver iterations and error norms were unchanged.

## Validation Anchors

- Diffusion raw-CUDA COO and CSR are checked against NumPy, Numba, and CuPy reduced matrix/RHS assembly through `p <= 6` in `tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_assembly_backends_match_numpy_for_p_le_6`.
- Diffusion raw-CUDA COO and CSR are checked against each other in the same parity test.
- Advection-reaction fused raw-CUDA CSR is checked against fused raw-CUDA COO in `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo`, including `legacy-lagrange` with `safe` and `coop` LU modes and a `legendre-modal` safe-mode case.
- Current raw-CUDA diffusion large-run timing is exposed by `scripts/gpu/run_diffusion_reaction_cuda.py`, including raw input prep, map/setup, zero, kernel, AMGX setup, AMGX solve, reconstruction, and plot/error timing rows.
