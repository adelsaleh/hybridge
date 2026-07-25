# Advection-Reaction Discontinuous Device Assembly Audit

This note records the current discontinuous-advection handling in the GPU-oriented advection-reaction assembly paths.

## Required Face Form

For an interior face shared by element sides `(K_l, f_l)` and `(K_r, f_r)`, the trace equation must keep both side contributions. With side normal fluxes `b_l = beta_l . n_l` and `b_r = beta_r . n_r`, and side stabilizations `tau_l` and `tau_r`, the trace mass contribution multiplying `hat u` is

```text
((tau_l - b_l) + (tau_r - b_r)) * hat u
```

against the oriented face test function. The matching right-hand-side and local coupling terms must also use each element-side value, not an averaged edge value. The row lift multiplying solved local columns uses the side upwind weight `tau = |beta_K . n_K|`, while the trace-trace mass contribution uses `gamma = tau - beta_K . n_K`.

## Current Device Paths

- CuPy assembly computes `beta_dot_normal` as an element-side array with shape `(K, f, q)`. `boundary_mass_cupy`, `element_boundary_mats_cupy`, tau-weighted trace lift assembly, trace data assembly, and interior mass assembly consume those side weights directly, so discontinuities are preserved until the two sides are summed into the global edge row.
- Raw-CUDA precomputed assembly receives CuPy-built `local_mats`, `element_boundary`, `source_rhs`, side trace mass blocks, and tau-weighted trace lifts. Since those inputs already contain side-wise `tau`/`gamma` weights, the raw kernel only performs local solves, orientation handling, boundary elimination, and reduced emission.
- Raw-CUDA fused COO computes `tau_face = abs(beta_K . n_K)` and `gamma_face = tau_face - beta_K . n_K` inside the element block for each `(K, f, q)`. It forms the tau-weighted row lift and gamma trace mass from those face-quadrature values, then emits each interior side contribution separately into the global trace row, where duplicate edge entries represent the left/right sum.
- Raw-CUDA fused CSR uses the same fused assembly source as COO, with only the final write sites replaced by direct CSR positions and `atomicAdd`. The `tau_face`/`gamma_face` computation is therefore identical to the COO path.
- Device reconstruction is element-local. The fused reconstruction kernel rebuilds local matrices from the element's own projected beta and the final trace vector; it does not form an interior face conservation row, so there is no edge averaging step to audit there.

## Validation Anchors

- `tests/test_adv_rea_numba.py::test_discontinuous_beta_uses_side_weighted_trace_mass` checks that the side-weighted formulation differs from the old unweighted/averaged trace mass.
- `tests/test_cupy_backend.py::test_advection_reaction_cupy_discontinuous_beta_matrix_matches_numpy` compares CuPy full/reduced matrix and RHS assembly against NumPy for a discontinuous projected beta field.
- `tests/test_cupy_backend.py::test_advection_reaction_raw_cuda_discontinuous_beta_matrix_matches_numpy` compares raw-CUDA precomputed, fused safe, and fused cooperative matrix/RHS assembly against NumPy for the same discontinuous-beta class of inputs.
- `tests/test_cupy_backend.py::test_raw_cuda_fused_modal_trace_assembly_matches_cupy_discontinuous_beta` compares fused raw-CUDA cooperative assembly against the CuPy reference for a discontinuous projected beta field.
- `tests/test_cupy_backend.py::test_raw_fused_csr_assembly_matches_coo_discontinuous_beta` compares fused raw-CUDA direct CSR against fused raw-CUDA COO for the same discontinuous-beta class of inputs.
- `tests/test_adv_rea_conservation.py` checks the end-to-end HDG global conservation balance, including a high-order no-through-flow case and the legacy non-tangent boundary case.
