# Assembly on the NVIDIA RTX PRO 5000 Blackwell (sm_120)

Index of the assembly measurements on the FP64-limited Blackwell workstation
GPU, September-October 2026, and how they compare with the H200. The
algorithms are described in [`../README.md`](../README.md); the detailed
records stay in `docs/research/solver_studies/` (linked below) and are the
source of truth for every number on this machine.

## Platform

| Item | Value |
|---|---|
| GPU | RTX PRO 5000 Blackwell, sm_120, 110 SMs, 48 GB GDDR7, 96 MiB L2, **99 KiB** opt-in shared memory per block |
| FP64 | 1/64 of FP32; measured DGEMM ~0.74 TFLOP/s (about 1/40 of the H200 NVL) |
| Software | CUDA 13.0-13.2, CuPy 14.2, cuTENSOR 2.8.1, MAGMA 2.10 (sm_120 build) |

Consequences for assembly: the dense local algebra is FP64-bound here, so
FP64 tensor-core paths gain little (tensor AR wins only at p=8-9), the
99 KiB shared limit forces the external-mass layout for p>=7 variable ADR
tensors, and launch priors tuned here (e.g. retained coupling when FP32:FP64
>= 16) do not transfer to FP64-rich GPUs. The fused AR kernel compiles to
80-88 registers here (255 on sm_80-90).

## Records

| Family | Study | Headline (FP64) |
|---|---|---|
| AR | [`advection_assembly_baseline_2026_10_03.md`](../../../research/solver_studies/advection_assembly_baseline_2026_10_03.md) | 117,439-triangle star, p=9: fused 699 ms, split3 389 ms, tensor + MAGMA (cuTENSOR) 299 ms; split3 fastest at p=5-7, library paths at p=8-9 |
| DR | [`poisson_raw_cupy_2026_10_14.md`](../../../research/solver_studies/poisson_raw_cupy_2026_10_14.md) | 100k triangles: raw CSR 1.40-3.95x faster than CuPy Schur-Cholesky; raw kernel none / Cholesky p=6 98.4 / 125.4 ms, p=9 749 / 786 ms |
| DR | [`tensor_schur_prototype_2026_09_25.md`](../../../research/solver_studies/tensor_schur_prototype_2026_09_25.md), [`raw_assembly_shared_memory_2026_09_25.md`](../../../research/solver_studies/raw_assembly_shared_memory_2026_09_25.md) | tensor Schur prototype, FP32 reference, shared-memory investigation |
| ADR | [`raw_cuda_adr_tensor_2026_09_28.md`](../../../research/solver_studies/adr_assembly/raw_cuda_adr_tensor_2026_09_28.md) | 32k triangles, BSR: constant-full p=2/4/6 4.79 / 32.7 / 180.4 ms, variable-full 8.23 / 61.6 / 539.6 ms |
| ADR | [`raw_cuda_adr_high_order_2026_10_15.md`](../../../research/solver_studies/adr_assembly/raw_cuda_adr_high_order_2026_10_15.md) | p=7-9 qualification (508 tests, sanitizers clean); no p>=7 timings |

## H200 vs. this GPU (same problems)

| Path | Ratio H200 / RTX PRO 5000 (speed-up) |
|---|---|
| AR raw kernels (split3/fused), p=6-9 | 4-5x |
| AR tensor path, p=6-9 (H200 with MAGMA) | 8-10x; with the H200 fused solve p=9 is 22.4 vs 298.5 ms (13x) |
| DR raw `poisson_local` (quick-tuned on both) | 2.6x at p=2, 5.5-7.3x at p=6, 7.9-9.4x at p=9 |
| DR CuPy | 3.4-4.8x |
| ADR tensor kernel, p=6, 32k | 14.4-15x (12.1 vs 181 ms scalar, 37.6 vs 540 ms variable-full) |

The gaps are far below the ~40x FP64 ratio wherever the H200 kernels are
latency- or occupancy-bound; see [`../h200_sm90/README.md`](../h200_sm90/README.md).

## Open items on this GPU

* No ADR timings at p>=7 and no block-size-1 data exist.
* The H200 changes (fused AR local solve, chunked tensor path, ADR explicit
  mass inverse, Poisson FP64-rich priors) are gated to sm_90 or FP64-rich
  devices; whether the explicit mass inverse helps with the external-mass
  layout this GPU uses has not been measured.
