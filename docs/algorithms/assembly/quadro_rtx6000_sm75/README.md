# Assembly on the NVIDIA Quadro RTX 6000 (Turing, sm_75)

Index of the July 2026 measurements on the 24 GB Quadro RTX 6000 test GPU,
which set the first raw-CUDA launch recommendations. The algorithms are
described in [`../README.md`](../README.md); the records below are the source
of truth for this machine.

## Records

| Study | Content |
|---|---|
| [`raw_cuda_fused_coop_lu_2026_07_20.md`](../../../research/solver_studies/raw_cuda_fused_coop_lu_2026_07_20.md) | fused raw-CUDA advection kernel with the cooperative LU; full solves on the Quadro RTX 6000 |
| [`diffusion_assembly_2026_07.md`](../../../research/solver_studies/diffusion_assembly_2026_07.md) | diffusion assembly study (July 2026) |
| [`advection_reaction_2026_07.md`](../../../research/solver_studies/advection_reaction_2026_07.md) | advection-reaction solver study; excludes assembly from its comparisons |

## Recommendations that originate here

The raw-CUDA block-size table in
[`../../../backends/raw_cuda.md`](../../../backends/raw_cuda.md) (advection-reaction
32 for p<=6, 64 at p=7, 128 at p=8-9; ADR 32 / 64 / 128 for p<=2 / 4 / 9) comes
from SM75 and Blackwell sweeps; the p=8-9 fused recommendation is additionally
backed by a CUDA-13 SM75 run. These are not cross-architecture claims: on the
H200 the measured optima differ (ADR p>=8 variable tensors use 256-512
threads, DR assembly 32-256 threads with full-width column batches), see
[`../h200_sm90/README.md`](../h200_sm90/README.md).

## Open items on this GPU

None of the October 2026 paths (tensor AR with the fused solve, chunking,
the ADR explicit mass inverse) has been run on sm_75.
