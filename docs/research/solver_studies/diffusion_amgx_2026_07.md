# Diffusion Modal-Trace AMGX Study: July 2026

Status: historical performance research, not a backend recommendation.

## Question

The p6 `dub_orth + legendre-modal` diffusion trace system converged much more
slowly than the nodal `legacy-lagrange` system under the existing
`PCGF + Chebyshev/L1` AMGX hierarchy. The investigation tested three
hypotheses:

1. the modal matrix was asymmetric or badly scaled;
2. aggressive coarsening built an unsuitable hierarchy;
3. a stronger or different smoother/preconditioner would recover performance.

The raw-CUDA matrix, discretization, and physical-residual checks were held
fixed while scaling and AMGX configurations changed.

## Common Setup

- trigonometric Poisson case on a disc mesh;
- polynomial order 6 and `dub_orth` element basis;
- symmetric volume quadrature;
- raw-CUDA direct CSR assembly;
- block size 128;
- requested AMGX tolerance `1e-12`.

Historical configuration and log filenames retain their original abbreviated
names for reproducibility.

## Matrix Scaling Result

On the 113,878-triangle case, the reduced matrix had 1,192,968 trace DOFs and
41,676,852 nonzeros.

| Trace basis | Scaling | Symmetry defect | PCGF iterations | AMGX solve |
|---|---|---:|---:|---:|
| legacy-lagrange | none | 8.034e-15 | 32 | 0.453 s |
| legacy-lagrange | symmetric | 9.474e-15 | 242 | 3.341 s |
| legendre-modal | none | 7.028e-15 | not repeated | not repeated |
| legendre-modal | symmetric | 1.241e-14 | 2000 | 24.847 s |

The matrices were symmetric to roundoff. Symmetric Jacobi scaling normalized
the diagonal and row/column magnitudes, but it worsened the nodal hierarchy and
did not rescue the modal solve. The scaling kernel cost about 0.02 s; the loss
came from changed AMGX hierarchy and preconditioner behavior.

Small p2 condition estimates likewise showed no modal singularity or
orientation defect. Scaling reduced the nodal estimate but slightly worsened
the modal estimate.

## Hierarchy Result

The aggressive Chebyshev/L1 hierarchy produced:

| Mesh size | Trace basis | Levels | Coarse rows | Operator complexity | Iterations | Solve |
|---:|---|---:|---:|---:|---:|---:|
| 0.18 | legacy-lagrange | 2 | 3,738 | 1.02278 | 31 | 0.0875 s |
| 0.18 | legendre-modal | setup failed | - | - | - | - |
| 0.04 | legacy-lagrange | 4 | 5,601 | 1.03785 | 32 | 0.4494 s |
| 0.04 | legendre-modal | 4 | 2,308 | 1.00762 | 1,923 | 23.8678 s |

The modal hierarchy was substantially smaller, not larger. Its low operator
complexity did not represent high-order modal edge error effectively enough
for PCGF.

The coarse modal setup failure came from the DenseLU trigger. With
`dense_lu_num_rows=2048`, aggressive coarsening could stop at a block that
was too large for the selected dense coarse solve. A diagnostic threshold of
128 removed that setup failure and preserved nodal iterations, but modal
fine-grid convergence remained at roughly 1,922 iterations. This was a
robustness finding, not sufficient evidence to change the production config.

## Preconditioner Sweep

Representative unscaled results:

| Scale | Variant | Trace | Iterations | Setup | Solve | Physical residual |
|---|---|---|---:|---:|---:|---:|
| 28,569 triangles | nodal Cheb/L1 aggressive | legacy | 32 | 0.541 s | 0.174 s | 6.075e-13 |
| 28,569 triangles | modal Cheb/L1 aggressive | modal | 754 | 0.409 s | 3.565 s | 9.714e-13 |
| 28,569 triangles | modal BICGSTAB classical | modal | 2000 | 0.097 s | 1.368 s | 9.117e-14 |
| 113,878 triangles | nodal Cheb/L1 aggressive | legacy | 32 | 0.449 s | 0.452 s | 7.503e-13 |
| 113,878 triangles | modal Cheb/L1 aggressive | modal | 1923 | 0.231 s | 23.978 s | 9.514e-13 |
| 113,878 triangles | modal BICGSTAB classical | modal | 2000 | 0.095 s | 5.034 s | 1.448e-09 |
| 113,878 triangles | modal no-aggressive Cheb order 10 | modal | 539 | 0.437 s | 31.012 s | 9.571e-13 |

Higher Chebyshev order reduced iteration counts but increased V-cycle cost, so
wall time became worse. Non-Chebyshev aggregation GS/DILU and direct DILU
variants looked inexpensive on the coarse mesh but reached the iteration cap
on the fine mesh. PCGF with classical MULTIPASS/GS reached a strict residual
in fewer iterations but was slower still.

## Interpretation

- Raw-CUDA modal assembly symmetry and orientation were not the cause.
- Scalar diagonal scaling did not address modal coarse-space representation.
- Aggressive coarsening generated an inexpensive but ineffective modal
  hierarchy.
- Stronger smoothing traded fewer iterations for more expensive cycles.
- DenseLU threshold selection affected setup robustness independently of the
  modal convergence problem.

## July 2026 Conclusion

- Keep the nodal `legacy-lagrange + PCGF + Chebyshev/L1` path as the measured
  fast baseline for this study.
- Do not promote symmetric scaling or generated no-aggressive Chebyshev
  variants from these measurements.
- Treat `legendre-modal + BICGSTAB + classical AMG` as a practical historical
  fallback only; its strict-residual behavior remained mesh dependent.
- Revisit modal traces with basis-aware edge-block scaling or mass-normalized
  coordinates before tuning more scalar AMGX options.

These conclusions are dated. Current support and defaults are documented in
[`../../reference/backend_capabilities.md`](../../reference/backend_capabilities.md).

## Reproducibility Evidence

Primary tools:

- `scripts/gpu/diagnose_diffusion_matrix_scaling.py`
- `scripts/gpu/inspect_diffusion_amgx_hierarchy.py`
- `scripts/gpu/sweep_diffusion_amgx_preconditioners.py`

Primary logs:

- `run_logs/diff_rea_amgx_hierarchy_o6_ms0p18_ms0p04_20260726_223440.csv`
- `run_logs/diff_rea_amgx_hierarchy_o6_ms0p18_ms0p04_20260726_223752.csv`
- `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p18_20260726_205823.csv`
- `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p08_20260726_205902.csv`
- `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p04_20260726_210015.csv`
- `run_logs/diff_rea_modal_amgx_preconditioners_o6_ms0p04_20260726_220323.csv`
