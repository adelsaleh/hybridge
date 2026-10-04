# Initial oscillatory stress-case evidence

The recommended combined stress case is **`cellular7_anisotropic` on the five-lobed annulus with a central hole**, at degree 6. It has a 0.07-wide minimum neck, rapidly oscillating velocity, and a rotated diffusion tensor with eigenvalues `1e-4` and `1e-6`. The more severe `cellular7_weak` variant uses isotropic `1e-5` diffusion.

On 22,825 triangles / 236,502 free trace unknowns, the anisotropic case takes **150 ASM+PP–GMRES iterations, averaging 758.8 ms**, versus 14 iterations / 72.4 ms for the smooth reference on the same mesh. Its primal L2 error is `4.99e-10`. Block-AMG/FGMRES needs 230 iterations / 1,850.6 ms; block-AMG/BiCGSTAB needs 162 / 2,293.8 ms. Direct block-DILU/BiCGSTAB fails at 1,000 iterations. These are the tested frozen configurations, not an exhaustive AMGX tuning result.

| Fine-annulus case | ASM+PP / GMRES | AMG / FGMRES | AMG / BiCGSTAB | DILU / BiCGSTAB |
|---|---:|---:|---:|---:|
| Smooth reference | 14 it. / 72.4 ms | 24 / 173.1 ms | 13 / 177.4 ms | 90 / 54.5 ms |
| Oscillatory anisotropic | 150 / 758.8 ms | 230 / 1,850.6 ms | 162 / 2,293.8 ms | NC |
| Oscillatory low isotropic | 300 / 1,514.8 ms | NC | NC | NC |

NC means the common acceptance contract failed; it is not a timing rank. Each reported time averages three measured second solves, with an explicit zero guess for every solve. In the fine anisotropic case the timing ranges are 757.8–760.0 ms, 1,849.6–1,851.7 ms, and 2,290.4–2,295.8 ms for ASM, AMG/FGMRES and AMG/BiCGSTAB. Their median setup times are 322.1, 202.4 and 202.1 ms, respectively.

This initial study is preserved inside the [combined oscillatory study](../README.md) and [combined ADR report](../../README.md). The main report now treats the oscillatory experiment as a subsection with three diffusion classes.

## Report and evidence

- [Insertable LaTeX section](section.tex), with a [standalone wrapper](preview.tex); intended to fit within the requested 3–4-page limit. Typesetting and actual pagination have not been checked: compilation is left to the user under the workspace rule.
- [Portable report bundle](../../../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/initial/adr_oscillatory_bundle.zip), including PDF/SVG/PNG figures, TeX, data, source snapshots and exact mesh arrays.
- [Geometry and fields](../../../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/initial/figures/geometry_fields.png) and [solver heatmap](../../../../../../run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/initial/figures/solver_comparison.png).
- [Tabular results](comparison.csv), [full measurements/configurations](results.data.json), [verification audit](verification.json) and [figure hashes](figure_hashes.json).
- The earlier [h/p scaling section](../../README.md) retains the larger smooth ADR campaign, including high diffusion and native hp-BSR. This note adds the combined geometry/coefficient stress case.

From this directory, typeset the supplied wrapper with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error preview.tex
```

To insert the section into a larger report, load `amsmath`, `graphicx`, `booktabs` and `hyperref`. Set `\ADRStressPath` to the directory containing `section.tex`, and `\ADRStressFigurePath` to the figure directory, then input `\ADRStressPath/section.tex`. The ZIP uses local `figures/` paths. Generated figure PDFs stay outside `docs/`.

## Exact problem

The conservative stationary equation is

```text
-div(K grad u) + div(beta u) + c u = f,     u = u* on every boundary edge,
u* = sin(6*pi*x)sin(5*pi*y) + 0.35 sin(11*pi*x)sin(9*pi*y),
beta = (1 + 4 sin(7*pi*x)cos(7*pi*y), 0.5 - 4 cos(7*pi*x)sin(7*pi*y)),
c = 0.01,
K = R(pi/6) diag(1e-4,1e-6) R(pi/6)^T,     or K = 1e-5 I,
f = -Kxx u*xx - 2 Kxy u*xy - Kyy u*yy + beta.grad(u*) + c u*.
```

The divergence of beta is zero. The domain is `0.58 < r < 1 + 0.35 cos(5 theta)`. The actual affine meshes have a 400-segment outer polygon and a polygonal circle; exact Dirichlet data is evaluated on all discrete boundary edges. The field is not constrained to be tangent to the boundary. This is a manufactured transport problem, not an impermeable-wall fluid model.

The smooth comparison uses `u*=sin(pi*x)sin(pi*y)`, `beta=(1,0.5)`, `K=1e-3 I`, and `c=1`. An additional control changes only the exact solution/source: its matrix is byte-identical to the smooth matrix, and ASM iterations stay at 8 on the coarse square and 9 on the coarse annulus. Oscillatory forcing alone does not change matrix conditioning.

The velocity is adapted from [Haynes and Vanneste's cellular-flow family](https://arxiv.org/abs/1401.6666). The Fourier solution, drift, reaction, anisotropy and star annulus are our manufactured extensions; this complete benchmark is not claimed to appear in that paper. [Le Bris, Legoll and Madiot](https://arxiv.org/abs/1710.09331) provide a related published family of advection-dominated problems in perforated domains and explain why the conditions on holes matter. We do not infer a universal geometry penalty by comparing different domains and mesh sizes.

## Protocol and validation

All methods share FP64 matrices/RHS, unchanged Bernstein trace coordinates, fixed stabilization `tau = 1 + max(beta.n, 0)`, and zero guesses. Solver success and independent physical relative residual at most `1e-10` are required; the internal target is `1e-11`, the iteration cap 1,000, and GMRES restart length 75. ASM uses PP degree 24. AMGX uses natural face BSR blocks, classical `block_graph_dense` AMG with DILU smoothing or direct DILU, with exactly matched AMG for FGMRES/BiCGSTAB. The available prebuilt AMGX and pyamgx include local modifications; their actual binary hashes are in the audit, so these results are not attributed to an unmodified upstream release.

The four campaigns cover 17 systems and 68 solver configurations. Of these, 55 pass and 13 numerical failures are retained. There are 390 solves in passing configurations and 403 attempted solves including failures. One warmup setup is excluded; each measured setup runs two zero-guess solves. The degree-4 screening uses two measured setups, and all degree-6 comparisons use three. Hot times exclude assembly, setup and validation. `comparison.csv` records hot mean/min/max, median setup/fresh/amortized times in ms, iteration counts, PDE error and physical residual; failed rows preserve their last iteration/residual without a ranked timing.

Validation includes:

- 21 CPU tests passed, including independent complex-step checks of the manufactured source, divergence and derivatives, SPD diffusion, and an assembled unchanged-matrix control.
- 14 independent CPU assembly/direct-solution references. Maximum CPU/GPU block and RHS differences are `1.05e-14` and `6.29e-15` relative. Fine annular systems use physical residual and analytic PDE error; no fine-mesh direct reference is claimed.
- Worst physical residual among passing configurations: `9.98e-12`. Saved traces agree across passing solvers within `3.06e-9` relative to ASM.
- Coarse-to-fine annular ASM primal L2 errors: `9.04e-8 -> 4.99e-10` (anisotropic), `1.61e-7 -> 7.77e-10` (low isotropic). This is resolution evidence, not a formal convergence-order claim on just two polygonal meshes.
- Doubled volume/edge quadrature, 14 -> 28 points: field changes are `2.35e-11` / `5.32e-11`, or 0.026% / 0.033% of coarse discretization error. All four anisotropic solver outcomes and iteration counts are unchanged in the separate sensitivity probe. Matrix coefficients change by about `4e-5` relative; no roundoff-level quadrature agreement is claimed.
- On the coarse annulus, reproducible lower estimates of matrix kappa_1 are `1.47e4` (base), `1.51e5` (anisotropic) and `4.14e5` (low isotropic). They use sparse LU and `onenormest(t=4,itmax=8,seed=1729)`, in fixed Bernstein coordinates; they are not exact kappa_2 or preconditioned condition numbers. Asymmetry, nonnormality and block-coupling measures are also archived.
- Separate anisotropic ASM+PP profiles give complete application means of **1.063 / 4.495 ms** on 6,114 / 22,825 triangles, using 10 samples after 3 warmups. The preconditioner accounts for **83.2% / 90.5%** of attributed GPU operation time. Instrumented profiles are excluded from rankings; host synchronization waits overlap GPU work.

The requested investigation of expensive ASM+PP applications, first using library operations and then custom CUDA kernels, remains in [the project TODO](../../../../../../TODO.md#face-dense-polynomialasm-solver-comparison).

## Reproduction

The recorded implementation is copied under `vendor/adr_gmres/`: [case factory](../../../../../../vendor/adr_gmres/scripts/oscillatory_adr_cases.py), [campaign runner](../../../../../../vendor/adr_gmres/scripts/run_oscillatory_adr_study.py), [matrix diagnostics](../../../../../../vendor/adr_gmres/scripts/diagnose_oscillatory_adr_matrix.py), [quadrature check](../../../../../../vendor/adr_gmres/scripts/check_oscillatory_adr_quadrature.py) and [audit](../../../../../../vendor/adr_gmres/scripts/audit_oscillatory_adr_study.py). The master checkout supplies the [canonical mesh generator wrapper](../../../../../../scripts/advection_diffusion_reaction/meshes/make_oscillatory_geometry.py) and [report renderer](../../../../../../scripts/reports/make_oscillatory_adr_report.py).

Original meshes, specs, source snapshots, job logs, cached systems and solutions are in the sibling worktree's `run_logs/adr_oscillatory_*_20260918` directories, with complete paths in the JSON archive. Star `n56` / `n107` job labels are conservative memory-planning bounds, **not structured mesh dimensions**; use the recorded actual triangle/trace counts. Preserve the archived NPZ meshes for an identical comparison. The earliest square screening snapshot predates the optional NPZ-mesh argument; inspection confirms its rectangular fallback and numerical solver are unchanged.

Example rerun of the selected fine case, on this machine with the existing prebuilt dependencies:

```sh
cd /home/adelsaleh/src/hdgfem-gmres
source run_logs/adr_baseline_20260917/environment.sh
HDGFEM_CUDA13_ROOT=/usr/local/cuda-13.0 \
HDGFEM_AMGX_BUILD_ROOT=/home/adelsaleh/src/AMGX-build-cuda13 \
HDGFEM_AMGX_INSTALL_ROOT=/home/adelsaleh/src/AMGX-install-cuda13 \
/home/adelsaleh/src/hybridge/scripts/gpu/run_cuda13.sh python -m scripts.run_oscillatory_adr_study \
  --output run_logs/adr_oscillatory_star_fine_rerun \
  --cases cellular7_anisotropic cellular7_weak \
  --mesh-path run_logs/adr_oscillatory_geometry_20260918/star_h0.02.npz \
  --p 6 --warmup 1 --repeats 3 --maxiter 1000 --timeout 180
```

To re-audit and regenerate the report from the archived campaigns, without solver runs:

```sh
cd /home/adelsaleh/src/hdgfem-gmres
/home/adelsaleh/src/hybridge/.venv/bin/python -m scripts.audit_oscillatory_adr_study
cd /home/adelsaleh/src/hybridge
.venv/bin/python scripts/reports/make_oscillatory_adr_report.py
```

Only the explicitly authorized runtime GPU JIT was used for these stationary tests and benchmarks. No native build/install or time integration was performed.

## Native completion — 21 September 2026

The section and solver heatmap now include native hp-BSR, with both standard
and robust policies attempted on all 17 initial systems (13 systems converge).
`native_completion_table.tex` separates fresh and reused costs, and
`native_results.json` retains each policy's original samples and failures.
The original 68 AMGX/ASM configuration measurements are unchanged. Both native
policies fail on the two fine-annulus stress cases, preserving the ASM+PP
advantage. The updated sources and figures are bundled; TeX was not compiled.
