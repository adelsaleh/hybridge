# Fused diffusion assembly: tensor Schur prototype and FP32 reference

Measured on NVIDIA RTX PRO 5000 Blackwell, CUDA runtime 13.2, CuPy 14.2.0.
The mesh has 32,768 triangles, Legendre modal basis, direct BSR assembly.
Each entry is the best median kernel time over blocks 32, 64, 128,
with two warmups and seven timed repetitions. Compilation is excluded.

| p | Original FP64 ms (block) | Original FP32 ms (block) | TF32 ms (block) | TF32x3 ms (block) |
|---|---:|---:|---:|---:|
| 2 | 2.076 (64) | 0.342 (32) | 2.108 (64) | 2.421 (64) |
| 4 | 14.526 (32) | 1.900 (64) | 14.224 (128) | 15.134 (64) |
| 6 | 49.204 (64) | 5.997 (128) | 47.811 (64) | 53.407 (128) |

The tensor prototype replaces only the two matrix products that form the
local Schur matrix before LU. WMMA executes inside the fused kernel.
LU and triangular solves retain the original FP64 CUDA-core code.
TF32x3 uses three products of high/low TF32 components and accumulates
partial results in FP64. The FP32 reference specializes the entire original
kernel to single precision, including its CUDA-core LU and triangular solves.
Production defaults are unchanged.

Disassembly confirms HMMA.1684.F32.TF32 instructions in both tensor variants,
and no HMMA in either original-kernel reference. At p=6 the tensor variants
reuse dead shared workspace and retain the original 32,232-byte dynamic
shared allocation; FP32 uses 16,300 bytes. Static instruction counts are not
execution counts or utilization measurements.

## Accuracy

Residuals below are evaluated against the original FP64 system on a sheared
eight-triangle mesh, after a tiny dense host solve. They measure assembly
perturbation, not iterative solver convergence.

| p | FP64 residual | FP32 residual | TF32 residual | TF32x3 residual |
|---|---:|---:|---:|---:|
| 2 | 1.03e-15 | 1.21e-6 | 3.59e-3 | 8.86e-7 |
| 4 | 1.61e-15 | 4.01e-6 | 5.84e-3 | 3.56e-6 |
| 6 | 2.57e-15 | 6.29e-6 | 5.61e-3 | 9.80e-6 |

None of the reduced-precision variants passes the experimental 1e-9 gate.
At p=6, original FP32 is about 8.2 times faster than original FP64, whereas
moving these two products to TF32 gains only about 2.8%. TF32x3 is slower.
This does not rule out tensor acceleration of other assembly products.
Block choice depends on degree and precision; the experimental harness
selects the fastest passing measured candidate per degree/device/configuration.
These short desktop-GPU measurements do not establish a production tuning policy.

For 300,000 triangles at p=6, linear extrapolation gives 54.9 ms for the
FP32 fused kernel. The measured complete wrapper median is 16.31 ms at
32,768 triangles, giving a rough 149 ms linear extrapolation. The wrapper
includes source projection and setup; its fixed and mesh-dependent costs
have not been separated. Neither extrapolation is a measurement at 300,000
triangles, and neither includes solving the assembled global system.

## Reproduction

From the repository root, use a fresh output directory for each run:

```sh
.venv/bin/python scripts/diffusion_reaction/experiments/tensor_schur.py --order 6 --nx 128 --blocks 32 64 128 --modes fp64 tf32 tf32x3 --output /tmp/tensor-p6-new
.venv/bin/python scripts/diffusion_reaction/experiments/tensor_schur.py --order 6 --nx 128 --reference-only --reference-dir /tmp/ref-p6-new --output /tmp/ref-p6-new
HYBRIDGE_PRECISION=float32 .venv/bin/python scripts/diffusion_reaction/experiments/tensor_schur.py --order 6 --nx 128 --blocks 32 64 128 --modes fp32 --reference-dir /tmp/ref-p6-new --output /tmp/fp32-p6-new
```

Repeat for p=2 and p=4. Compute Sanitizer racecheck passed the p=6,
block=128 TF32x3 smoke check with zero hazards; memcheck passed p=2,
block=32 TF32 and TF32x3 with zero errors. Source assertions also verify
that the tensor specialization leaves factorization and triangular solves intact.

[All timing samples, errors and resources](tensor_schur_prototype_2026_09_25.json).
[Compiled instruction evidence](tensor_schur_instruction_evidence_2026_09_25.json).

## Measured 300,000-triangle FP32 assembly

A 500 by 300 rectangular grid gives exactly 300,000 triangles. At p=6,
seven warmed repetitions of the original FP32 fused BSR assembly gave:

| Block size | Kernel median ms | Complete wrapper median ms |
|---|---:|---:|
| 32 | 82.850 | 124.207 |
| 64 | 66.533 | 107.978 |
| 128 | 60.473 | 103.707 |

The block=128 kernel samples ranged from 58.920 to 64.643 ms. The complete
wrapper includes source projection, allocations and assembly setup, with
synchronization around each call. Initial mesh construction and JIT compilation
are excluded. No global solve was performed. The matrix-data relative error
against separately assembled FP64 was 2.90e-6; RHS relative error was 4.61e-6.
The measured 60.5 ms kernel and 103.7 ms wrapper replace the earlier 54.9 ms
and 149 ms extrapolations for this mesh. Block 128 is the fastest tested
configuration for FP32 p=6; this is not a universal block-size choice.

[Complete measured samples](fp32_poisson_300k_2026_09_25.json).
Use the reproduction commands above with `--nx 500 --ny 300` in both
reference and FP32 processes, using fresh matching output directories.
