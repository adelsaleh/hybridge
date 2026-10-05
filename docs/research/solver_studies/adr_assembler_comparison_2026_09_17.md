# Matched ADR assembler comparison, 17 September 2026

For the next solver comparison on this machine, use the current master Numba assembler
with eight threads for constant scalar diffusion and the GMRES branch CuPy assembler
for tensor diffusion at the measured moderate/high orders. This is a measured choice
for these cases, not a package default change or a claim about every mesh/order.

## Algebraic compatibility

The branches implement compatible stationary conservative ADR algebra on the tested
cases. Apparent matrix differences came from stabilization, trace basis, and quadrature:

- Branch: Bernstein traces, Duffy volume quadrature, and diffusive stabilization 1
  plus `max(beta.n, 0)`.
- Current master: default nodal Lagrange traces, automatic volume quadrature, and
  `kappa/global_length + abs(beta.n)` stabilization.

The diagnostic explicitly uses Duffy volume quadrature with `2*p+2` points per
direction, the same mesh and Dubiner volume basis, diffusion stabilization 1, and
`max(beta.n, 0)` on both sides. Existing native trace face quadrature is retained;
it integrates the polynomial face terms exactly for these cases. Constant/linear
velocities are exactly representable under master's coefficient projection. The
quadratic boundary data are exactly representable and trigonometric boundary
data vanish on this rectangle. This is not a parity claim for arbitrary nonlinear
coefficients or non-polynomial nonzero Dirichlet traces.

For native trace coordinate conversion, let T map Bernstein coefficients to
Lagrange coefficients. Compare `T.T @ A_L @ T` and `T.T @ b_L` with the branch
Bernstein system. No row scaling or sign correction is used.

The small gate covers all five manufactured cases at p=2,4 on 32 triangles:
42 supported combinations passed; eight tensor/fused combinations are explicitly
unsupported. Maximum relative matrix/RHS differences were `5.55e-15` / `4.46e-15`;
maximum relative trace-solution difference was `2.61e-14`. The timing sweeps
also validate every assembled matrix and RHS. The largest matrix difference
across the completed sweeps is `2.13e-14`.

## Complete assembly timing

The table uses 8,192 triangles, p=4, and 60,800 free trace DOFs. Values are median
wall times in milliseconds after two warmups and five measured repetitions.
The total includes native assembly, canonical host CSR conversion, and the
Bernstein change of basis for master. Mesh/space construction is excluded.
Numba uses eight threads; BLAS/OpenMP uses one. GPU operations are synchronized.

| Case | Backend | Assembly (ms) | Host CSR + basis (ms) | Total (ms) |
|---|---|---:|---:|---:|
| advection_dominated | branch-numpy | 816.393 | 26.425 | 843.782 |
| advection_dominated | branch-cupy | 556.830 | 26.361 | 582.770 |
| advection_dominated | master-numpy | 1135.811 | 68.355 | 1201.916 |
| advection_dominated | master-numba | 284.023 | 72.376 | 356.399 |
| advection_dominated | master-raw | 416.661 | 44.059 | 460.720 |
| anisotropic | branch-numpy | 813.728 | 27.067 | 840.138 |
| anisotropic | branch-cupy | 558.405 | 26.033 | 583.466 |
| anisotropic | master-numpy | 1139.893 | 70.211 | 1209.683 |
| anisotropic | master-numba | unsupported | — | — |
| anisotropic | master-raw | unsupported | — | — |

Medians of individual stages need not sum to the median of the per-run total.

Scalar diffusion: Numba is about 1.64x faster than branch CuPy for this common
output. Tensor diffusion: branch CuPy is about 1.44x faster than branch NumPy
and 2.07x faster than master NumPy. At 2,048 triangles, p=2,4,6, the native
assembly sweep also favored Numba for scalar p=4,6 and branch CuPy for tensor
p=4,6; at low order the differences are smaller. A device-resident consumer may
benefit from avoiding host CSR conversion, so this table does not establish
an end-to-end GPU solver winner.

Raw CUDA is measured by intercepting its existing solver boundary, after
device CSR construction but before AMGX. This captures preparation, boundary
maps, transfers, kernels, and sparse conversion. It avoids relying on master's
current `trace_assembly` field, which measures only the raw kernel. No production
assembler code was modified. Peak process RSS in worker JSON includes imports,
JIT and warmup; it is not a scoped assembly or GPU peak-memory measurement.

## Evidence and reproduction

The two diagnostic scripts reuse the existing assemblers:

- [`vendor/adr_gmres/scripts/compare_adr_assemblers.py`](../../../vendor/adr_gmres/scripts/compare_adr_assemblers.py): sequential orchestration and matrix/RHS/trace parity.
- [`vendor/adr_gmres/scripts/compare_adr_assembly_worker.py`](../../../vendor/adr_gmres/scripts/compare_adr_assembly_worker.py): isolated source-root imports and synchronized timings.

Local evidence is in `run_logs/adr_assembly_comparison_20260917/`:

- `small`: 42 passed, 8 unsupported; all five cases, n=4, p=2,4.
- `timing`: 24 passed, 6 unsupported; low diffusion/tensor, n=32, p=2,4,6.
- `larger`: 8 passed, 2 unsupported; low diffusion/tensor, n=64, p=4.
- `common_basis`: 8 passed, 2 unsupported; repeated timing with common output basis.

Each directory contains completion/summary JSON, per-worker logs and raw timing
samples, source hashes/revisions, sparse matrices, RHS, and geometry/basis tables.
The current master is a dirty working tree at e40ddd2; its source hash, rather
 than that commit alone, identifies the actual tested code. The branch is d44acce.
Environment details and package-version deviations are recorded in
`docs/adr_machine_baseline_2026_09_17.md`.

```bash
cd /path/to/hybridge
python vendor/adr_gmres/scripts/compare_adr_assemblers.py \
  --master-root . \
  --output run_outputs/adr_assembly_comparison_repeat \
  --cases advection_dominated anisotropic --meshes 64 --degrees 4 \
  --warmup 2 --repeats 5 --canonical-bernstein
```

## AMGX readiness and remaining work

The installed CUDA/AMGX guard passes using `/usr/local/cuda-13.0`,
`~/src/AMGX-build-cuda13`, and
`~/src/AMGX-install-cuda13`. The configured source is
`AMGX-hdg-cuda13` at 583084b; the PyAMGX source checkout is at 81efd1e.
No native build or package install was performed.

AMGX FGMRES + MULTICOLOR_DILU solved the exact cached branch low-diffusion
system at n=8,p=2 from a zero guess with no scaling. It converged in 51
iterations with independently measured true relative residual `7.34e-13`
and relative trace-reference error `8.74e-13`, passing the common `1e-12`
residual target. Configuration, validation, memory counters and extension
path are in `amgx_smoke.json`; this is readiness evidence, not a timing comparison.

The subsequent recorded sweep, large-mesh confirmation, memory/application
profiles and solver conclusions are in [the ADR solver comparison](adr_solver_comparison_2026_09_17.md).
The separate solver-optimization task remains open.
