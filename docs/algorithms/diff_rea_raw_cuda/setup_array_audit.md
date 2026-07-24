# Diffusion Raw-CUDA Setup Array Audit

Date: 2026-07-24

This note audits the current `hdgfem/backends/cupy_diff_rea_raw.py` path against
the package NumPy/Numba diffusion-reaction assembly pipeline. The goal is to
make clear which expensive work is now inside the raw CUDA element kernel and
which setup arrays are still prepared outside that kernel.

## Scope Audited

The raw-CUDA backend is intentionally narrower than the full NumPy/Numba solver
pipeline:

- identity diffusion only;
- scalar zero reaction in the standalone GPU runner path;
- nodal `legacy-lagrange` trace basis only;
- polynomial orders with `el_dof <= 28`, i.e. `p <= 6`;
- reduced COO emission or direct reduced CSR emission;
- reconstruction through the matching raw-CUDA reconstruction kernel.

Unsupported cases fall back to the CuPy path in
`scripts/gpu/run_diff_rea_gpu4_hdg.py`.

## Pipeline Comparison

The projected Numba diffusion path validates or prepares the following host
data before calling its fused kernel:

| Data | Numba path | Raw-CUDA diffusion path |
| --- | --- | --- |
| Source | Same-space source coefficients on host | `source_rhs` block moments built by CuPy before raw kernel |
| Reaction | scalar or same-space coefficients | not generalized yet; current raw path targets zero reaction |
| Tensor diffusion | projected inverse tensor coefficients | not supported yet |
| Stabilization `tau` | scalar or `(num_tri, 3)` host table | scalar `tau` only in raw kernel |
| Boundary trace | host `boundary_trace_coefficients` | CuPy boundary interpolation/projection before raw kernel |
| Derivative matrices | host reference `D0`, `D1` | CuPy `D0`, `D1` reference matrices before raw kernel |
| Face mass/lift tables | reference arrays passed to kernel | mirrored reference arrays plus a CuPy `face_element_mass` |
| Reduction maps | host free-edge maps and offsets | COO: host NumPy maps copied to device; CSR: raw-CUDA CSR pattern builder |
| Final RHS accumulation | host `np.add.at` after kernel | done in raw kernel with atomics into reduced RHS |

## Work Now Inside Raw CUDA

The raw assembly kernel now owns the expensive element-local part:

- builds the identity-diffusion condensed Schur matrix per element;
- solves trace/source columns without materializing per-element dense global
  tensors;
- applies boundary-column elimination during emission;
- emits the reduced trace matrix as either COO entries or direct CSR values;
- accumulates the reduced RHS on device;
- supports a cooperative block mode for local assembly, factorization, solves,
  and emission.

This means the old `local_lhs`, `element_boundary`, `solved_el_bd`,
`solved_src`, `trace_blocks`, and `faces` CuPy intermediates from the pure CuPy
path are no longer materialized for the raw path.

## Setup Arrays Still Built Outside the Hot Kernel

The following setup arrays remain outside `assemble_diffusion_raw_coop` /
`assemble_diffusion_raw_csr`.

Important distinction: "outside the hot kernel" does not always mean "built on
the host." In the current runner, `source_rhs`, `boundary_trace`,
`d0_reference`, `d1_reference`, and `face_element_mass` are produced by CuPy
device operations. The reduced COO maps are still NumPy host arrays copied to
the device. The reduced CSR sparsity pattern and CSR block-position maps are
already constructed on device before the assembly launch.

| Array or table | Built by | Location | Notes |
| --- | --- | --- | --- |
| `source_rhs` | CuPy quadrature evaluation and projection | `source_moments_cupy` in `scripts/gpu/run_diff_rea_gpu4_hdg.py` | Device array, but still outside raw kernel. For fully table-driven production, accept projected/source moment tables directly from caller. |
| `boundary_trace` | CuPy interpolation/projection of exact boundary callable | `boundary_trace_values_cupy` in `scripts/gpu/run_diff_rea_gpu4_hdg.py` | Device array for boundary edges only. Raw backend expands it to `boundary_trace_full` before launching the kernel. |
| `boundary_trace_full` | CuPy scatter into full edge table | `assemble_projected_diffusion_trace_system_eliminated_raw_cuda` | Device array sized `(num_edges, ntr)`. Could be avoided by passing boundary slots or compact boundary arrays into the kernel. |
| `d0_reference`, `d1_reference` | CuPy `einsum` over reference quadrature tables | `reference_derivative_mats` in `scripts/gpu/run_diff_rea_gpu4_hdg.py` | Small order-dependent device matrices. Good candidate for caching on `CupyReferenceElementData`. |
| `face_element_mass` | CuPy `einsum` over trace face tables | `face_element_mass` in `scripts/gpu/run_diff_rea_gpu4_hdg.py` | Small trace-reference tensor. Also a good cache candidate. |
| Mesh/reference mirrors | Host-to-device conversion | `CupyDGSpace`, `CupyReferenceElementData`, `CupyDGMesh` | Static data copied once per CuPy space construction. Not part of per-assembly hot kernel, but still a host-origin setup cost. |
| COO `edge_to_solve_edge` | NumPy on host, then copied to device | `_edge_to_solve_edge` in `cupy_diff_rea_raw.py` | Only used by `matrix_format="coo"`. |
| COO `interior_side_index` | NumPy on host, then copied to device | `_interior_side_index` in `cupy_diff_rea_raw.py` | Only used by `matrix_format="coo"`. |
| COO `side_flux_offsets` | NumPy on host, then copied to device | `_side_flux_offsets` in `cupy_diff_rea_raw.py` | Only used by `matrix_format="coo"`. |
| CSR `indptr`, `indices` | raw CUDA pattern kernels plus CuPy prefix ops | `build_reduced_csr_pattern_raw` in `cupy_adv_rea_raw.py` | Device-side construction. It still runs before assembly but no longer needs host COO-to-CSR reconstruction. |
| CSR block-position maps | raw CUDA pattern kernels | `side_csr_block_pos`, `mass_csr_block_pos` | Device-side maps consumed by direct CSR emission. |

## Remaining Gaps

The audit does not make the raw path fully general. The next implementation
items are:

1. Add a proper table-driven coefficient interface for reaction, tensor
   diffusion, and per-face `tau`.
2. Cache `d0_reference`, `d1_reference`, and `face_element_mass` on the device
   reference object rather than rebuilding them per assembly call.
3. Remove the COO-only host map builders or route COO through a device pattern
   helper analogous to the CSR path.
4. Avoid `boundary_trace_full` for large meshes by passing compact boundary
   trace data plus an edge-to-boundary-slot map.
5. Add an automated parity sweep against the Numba path for matrix/RHS,
   solution, and reconstruction error across `p`, mesh size, and supported
   trace cases.

## TODO Status

The audit item in `TODO.md` can be marked complete because the remaining
outside-kernel setup arrays are now recorded here. The implementation work
identified by the audit remains tracked by the later coefficient-generalization
and automated-validation TODO items.
