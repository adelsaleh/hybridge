# Oscillatory ADR: three diffusion classes

This [subsection](subsection.tex) extends the [combined ADR results section](../README.md).
It uses the same oscillatory exact solution, cellular velocity and reaction in
three distinct classes, so the change in diffusion can be compared directly.
The earlier uniformly weak anisotropy remains an additional stress test.

| Class | Diffusion tensor | Physical character |
|---|---|---|
| High diffusion | `K = I` | Diffusion dominated |
| Low diffusion | `K = 1e-5 I` | Transport dominated |
| Directional anisotropy | `K = R(π/6) diag(1, 1e-3) R(π/6)ᵀ` | Strong diffusion in one direction; weak diffusion across it |

The exact solution is `sin(6πx)sin(5πy) + 0.35 sin(11πx)sin(9πy)`.
The divergence-free velocity is `(1+4sin(7πx)cos(7πy), 0.5−4cos(7πx)sin(7πy))`,
and reaction is `0.01`. The source is manufactured analytically and exact Dirichlet
values are imposed on all discrete boundary edges. The five-lobed annulus is
`0.58 < r < 1+0.35cos(5θ)`, with a hole and a minimum nominal passage width of 0.07.

The cellular-flow family comes from [Haynes and Vanneste](https://arxiv.org/abs/1401.6666).
The complete manufactured problem, drift, tensors and geometry are our extensions.
[Le Bris, Legoll and Madiot](https://arxiv.org/abs/1710.09331) provide related
context on advection in perforated domains and the importance of boundary conditions.

## Matched h/p coverage

| Sweep | Square triangles | Annular triangles | Degree |
|---|---:|---:|---:|
| Coarse h point | 2,048 | 2,043 | 6 |
| Fixed mesh for p sweep | 8,192 | 8,039 | 1, 2, 3, 4, 6 |
| Intermediate h point | 32,768 | 33,174 | 6 |
| Finest h point | 99,458 | 99,984 | 6 |

These are exactly the original square points; annular counts match within 1.9%.
The coarsest annular polygon uses 220 outer segments, the others 400. A fixed
400-segment wall prevents a mesh as small as the 2k target. All meshes are connected
with two closed boundary loops. Their areas and trace dimensions differ from the
square, so equal triangle counts do not isolate geometry as the only cause of a
solver difference.

The protocol uses FP64, unscaled Bernstein traces, zero guesses, physical residual
`<=1e-10`, internal tolerance `1e-11`, GMRES restart 75 and a 1,000-iteration cap.
One setup is discarded, followed by three measured fresh setups with two solves
each; hot time averages each setup's second solve. Assembly and validation are
outside that timer. GPU jobs run serially.

The same configurations as the smooth scaling study are frozen along each curve:

- High: ASM+PP degree 24; block-AMG/Jacobi under FGMRES and BiCGSTAB; native hp-BSR
  under GMRES on the full operator, with its symmetric part used for preconditioning.
- Low: ASM+PP degree 24; block-AMG/DILU and direct block-DILU under both outer methods.
- Directional: ASM+PP degree 48; block-AMG/Jacobi under both outer methods.

Every AMGX path uses natural `(p+1) × (p+1)` BSR blocks. The native high-diffusion
adapter and the prebuilt, locally modified AMGX/pyamgx libraries are unchanged.
The native worker now accepts the oscillatory exact solution and recorded annular
mesh. These experiments do not establish a universal optimum across all available
AMGX hierarchies or smoother settings.

## Results and checks

The complete ladder has **121 passing and 71 failed configurations**. All 64 high-diffusion configurations pass; low diffusion passes 41/80, and directional anisotropy passes 16/48 (all ASM+PP). Failures remain in the data and figures.

Fastest tested passing configurations at the finest mesh, `p=6`:

| Class | Square: mean hot time / iterations | Annulus: mean hot time / iterations |
|---|---|---|
| High diffusion | AMG/BiCGSTAB: **0.258 s / 29** | Native hp/GMRES: **0.133 s / 32** |
| Low diffusion | ASM+PP(24)/GMRES: **6.915 s / 300** | ASM+PP(24)/GMRES: **12.369 s / 525** |
| Directional anisotropy | ASM+PP(48)/GMRES: **13.159 s / 300** | ASM+PP(48)/GMRES: **6.733 s / 150** |

For high diffusion on the finest annulus, setup changes the short-run choice: the setup-plus-hot-time model is 0.782 s for AMG/FGMRES versus 1.371 s for native hp. Native hp overtakes this AMG variant at about five reused right-hand sides. This is an arithmetic model, not a measured multi-RHS pipeline.

On the finest annular transport case, AMG/FGMRES and AMG/BiCGSTAB take 16.39 and 22.04 s, versus 12.37 s for ASM+PP. The ASM advantage over the faster passing AMG variant drops from 9.8× at the coarsest annular mesh to 1.33× at the finest. Transferred block-AMG/Jacobi fails every directional-anisotropy point; replacing its smoother with block DILU in a separate square probe does not resolve the failure.

Separate annular profiles give complete ASM+PP application means of **21.57 ms (high), 21.53 ms (low), and 42.61 ms (directional)**. Preconditioning accounts for **92.6–96.1%** of attributed GPU operation time. These are instrumented diagnostics, not ranked solve measurements.

[scaling.csv](scaling.csv) contains every configuration, including failures;
[scaling.data.json](scaling.data.json) preserves samples, effective settings,
assembly checks, profiles and the complete initial stress-study data.
[validation.json](validation.json) audits the common inputs, source/binary hashes,
independent CPU references, physical residuals, solution agreement and mesh topology.
[diagnostics.json](diagnostics.json) contains separately instrumented application
profiles, condition lower estimates, quadrature checks and the two failed DILU
smoother probes. Instrumented solves never enter the timing ranks.

The earlier [initial study](initial/README.md) remains intact as historical evidence.
Its `R diag(1e-4, 1e-6) Rᵀ` tensor is distinct from the new directional class.
**No 22,825-triangle assembly, solve or profile was repeated.** Four already recorded
square low-diffusion outcomes were reused; the new ladder adds 188 configuration attempts.

## Artifact locations

All oscillatory material is now inside the preceding study's directories:

- Documentation and TeX: this directory, with the original artifacts in `initial/`.
- Outputs (`run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/`, local, untracked):
  `scaling/square`, `scaling/annulus`, `diagnostics`, `smoother_probe`, `meshes`,
  `initial` (original figures/ZIP) and `initial_runs` (original jobs/caches).
- Combined figure directory (`run_outputs/solver_studies/adr_scaling_2026_09_17/figures/oscillatory/`, local, untracked):
  mesh and order timing/iteration curves, accuracy curves and preserved geometry plots.
- Portable combined ZIP (`run_outputs/solver_studies/adr_scaling_2026_09_17/adr_results_bundle.zip`, local, untracked):
  the full results section, data, selected meshes and source snapshots; large matrix
  caches remain in the local output directory.

Legacy paths are compatibility symlinks. The [artifact migration inventory](migration.json)
and raw-output migration inventory (`run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/raw_migration.json`, local, untracked)
record the move. Raw file bytes and modification times are preserved. Original
pre-migration documentation/figure bytes are also retained in
`run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/premerge_artifacts.zip` (local, untracked);
only document paths and heading levels were adjusted for incorporation.

## Reproduce or audit

The following commands use the existing environment and prebuilt libraries.
Runtime CUDA/CuPy JIT is required and was explicitly authorized; no native build,
installation or time integration is involved. `--resume` reads completed jobs,
including failures; use a new output directory for a fresh campaign.

```bash
cd /home/adelsaleh/src/hdgfem-gmres
source run_logs/adr_baseline_20260917/environment.sh
export HDGFEM_CUDA13_ROOT=/usr/local/cuda-13.0
export HDGFEM_AMGX_BUILD_ROOT=/home/adelsaleh/src/AMGX-build-cuda13
export HDGFEM_AMGX_INSTALL_ROOT=/home/adelsaleh/src/AMGX-install-cuda13

/home/adelsaleh/src/hybridge/scripts/gpu/run_cuda13.sh \
  python -m scripts.run_oscillatory_adr_scaling --geometries square --resume \
  --output /home/adelsaleh/src/hybridge/run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/scaling/square

/home/adelsaleh/src/hybridge/scripts/gpu/run_cuda13.sh \
  python -m scripts.run_oscillatory_adr_scaling --geometries annulus --resume \
  --mesh-manifest /home/adelsaleh/src/hybridge/run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/meshes/meshes.json \
  --output /home/adelsaleh/src/hybridge/run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory/scaling/annulus

/home/adelsaleh/src/hybridge/scripts/gpu/run_cuda13.sh \
  python -m scripts.audit_oscillatory_adr_scaling \
  --study-root /home/adelsaleh/src/hybridge/run_outputs/solver_studies/adr_scaling_2026_09_17/oscillatory \
  --output /home/adelsaleh/src/hybridge/docs/research/solver_studies/adr_scaling_2026_09_17/oscillatory/validation.json
```

Run the CPU derivative/ADR checks with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_oscillatory_adr_cases.py tests/test_adv_diff_rea.py -k 'not gpu'
```

`run_oscillatory_scaling_diagnostics.py --study-root ...` reruns only the separate
99,984-triangle application profiles and 8,039-triangle quadrature/conditioning
checks after the timing sweeps finish. The archived 22k case is excluded. Renderer
commands and the unrun TeX compilation command are in the [combined README](../README.md).
