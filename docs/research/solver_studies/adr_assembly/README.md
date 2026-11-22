# ADR assembly and implementation studies

Dated qualification of the stationary ADR assembly and reconstruction backends; the solver
studies built on them are in [`../adr_scaling_2026_09_17/`](../adr_scaling_2026_09_17/README.md).

- [Stationary ADR HDG implementation, 2026-08](stationary_adr_hdg_2026_08.md): the conservative
  formulation and its NumPy, fused Numba and raw-CUDA paths.
- [Numba ADR tensor specialization, 2026-09-25](numba_adr_tensor_2026_09_25.md): variable
  scalar/tensor parity, manufactured convergence and warmed comparisons
  ([samples](numba_adr_tensor_2026_09_25.jsonl)).
- [Raw CUDA tensor ADR qualification, 2026-09-28](raw_cuda_adr_tensor_2026_09_28.md): FP64 parity,
  native CSR/BSR solves and matched assembly timings
  ([samples](raw_cuda_adr_tensor_2026_09_28.jsonl), [manifest](raw_cuda_adr_tensor_2026_09_28_manifest.json),
  [diffusion regression](raw_cuda_adr_diffusion_regression_2026_09_28.jsonl)).
- [Raw CUDA tensor ADR shared-memory budget, 2026-09-29](raw_cuda_adr_tensor_shared_memory_2026_09_29.md)
  ([data](raw_cuda_adr_tensor_shared_memory_2026_09_29.json)).
- [Higher-order raw CUDA ADR qualification, 2026-10-15](raw_cuda_adr_high_order_2026_10_15.md): p=0–9
  parity, p=7–9 comparisons and sanitizer runs.
