# Advection-Reaction Solver Configurations

This note records the current provisional solver recommendations for the HDG advection-reaction path.  The list is not final: the `upwSCC + forward upwGS + Cupyx Krylov` path shows degree- and mesh-sensitive behavior that still needs explanation before it becomes a universal default.

Benchmark comparisons in this document are solver/preconditioner comparisons.  They intentionally exclude mesh setup, DG-space setup, coefficient projection, trace assembly, host/device transfers, COO-to-CSR conversion, reconstruction, and field-error evaluation unless a row explicitly says otherwise.

Current benchmark evidence is split across these reports:

- `run_logs/adv_rea_solver_benchmarks_p6_ms001_20260722.md`: p=6, mesh size 0.01 baseline comparison.
- `run_logs/adv_rea_solver_robustness_ms0008_20260722.md`: finer mesh size 0.008 degree and parameter sweep.
- `run_logs/adv_rea_upwgs_variability_20260722.md`: targeted upwGS variability follow-up, including BiCGSTAB and mesh sensitivity.
- `run_logs/adv_rea_amgx_variability_20260722.md`: matching raw-CUDA direct-CSR AMGX degree, mesh, and manufactured-parameter sensitivity follow-up.

## Current Ranking

| Rank | Configuration | Recommended use | Main reason |
| --- | --- | --- | --- |
| 1 | upwind-SCC + forward upwind block-GS + Cupyx GMRES | Fast experimental advection-reaction solves when tight residuals are needed and GMRES cycle counts are benign | Best observed solve-side cost on several cases, but p6/fine-mesh variability is not explained yet |
| 2 | raw-CUDA direct CSR + AMGX | Production GPU baseline and robustness reference | Stable device solve path with strong AMG preconditioning and field-error validation |
| 3 | upwind-SCC + forward upwind block-GS + Cupyx BiCGSTAB | Fast screening solve and robustness comparison for the upwGS preconditioner | Steadier p6/fine-mesh Krylov time than GMRES, but sometimes looser residuals |
| 4 | upwind-SCC + Cupyx ILU(1) + Cupyx BiCGSTAB | GPU-native ILU experiment | Fully device-side ILU(1) preconditioner with short BiCGSTAB solve |
| 5 | upwind-SCC + Cupyx ILU(1) + Cupyx GMRES | GPU-native ILU experiment for nonsymmetric robustness | Stronger Krylov method but slower than BiCGSTAB in the current p6/ms0.01 timings |

## Recommended Presets

### upwSCC + upwGS + Cupyx GMRES

Use this as the current performance research preset, not yet as an unconditional default.  The matrix is ordered by free trace-edge upwind SCC levels, then a forward upwind block-GS preconditioner is built from the ordered trace blocks.  The Krylov work is run by Cupyx GMRES on the device.

Strengths:

- Very low Krylov cost on favorable cases: one restarted GMRES cycle with a residual around `1e-14` at p=6/ms=0.01, p=5/ms=0.008, and p=7/ms=0.008.
- Preconditioner is much cheaper than strong SciPy ILU for the tested matrix.
- The preconditioner structure matches the advection direction and the HDG trace block formalism.

Weaknesses and caveats:

- Current fast runner still builds the preconditioner on the host and still pays COO-to-CSR separately when running end to end.
- The best next implementation target is constructing upwGS during assembly, first in Numba and later in raw CUDA.
- GMRES iteration counts in Cupyx are callback counts, which for restarted GMRES count restart cycles rather than every inner Arnoldi step.
- p=6 is sensitive on finer meshes in the current data: GMRES(50) needs 1 cycle at mesh size 0.010, 2 cycles at 0.009, and 6 cycles at 0.008.
- Mild manufactured-solution changes also affect p=6/ms=0.008: `test2_minus10` needed 8 GMRES cycles, while `test2_plus10` needed 2 cycles.

Example:

```bash
.venv/bin/python scripts/advection_reaction/run_upwind_gs_cupyx.py \
  -o 6 -ms 0.01 \
  --basis dub_orth \
  --trace-basis legacy-lagrange \
  --cupyx-solver gmres \
  --gmres-restart 50 \
  --maxiter 1500 \
  --rtol 1e-13 \
  --check-rtol 1e-10 \
  -v 2
```

### upwSCC + upwGS + Cupyx BiCGSTAB

Use this as a fast comparison solve for the same upwGS preconditioner.  It is particularly useful when GMRES restart-cycle counts jump on fine p=6 runs.

Strengths:

- More stable p=6 fine-mesh Krylov time in the current data: roughly 1.1-1.3 seconds over the p=6 mesh/case variants at mesh sizes 0.008 and 0.009.
- Uses the same upwind-SCC ordering and forward upwGS preconditioner as the GMRES path, so it isolates the Krylov-method choice.
- Much cheaper Krylov time than GMRES(50) for the p=6/ms=0.008 outlier.

Weaknesses and caveats:

- The reported scaled residual can be looser than GMRES on p=6 fine meshes, around `1e-12` in several runs even with `info=0`.
- It should not replace GMRES for accuracy-sensitive runs until residual semantics and stopping behavior are better characterized.
- Like the GMRES upwGS path, it still relies on a host-built preconditioner in the current runner.

Example:

```bash
.venv/bin/python scripts/advection_reaction/run_upwind_gs_cupyx.py \
  -o 6 -ms 0.008 \
  --basis dub_orth \
  --trace-basis legacy-lagrange \
  --cupyx-solver bicgstab \
  --maxiter 1500 \
  --rtol 1e-13 \
  --check-rtol 1e-10 \
  -v 2
```

### raw-CUDA direct CSR + AMGX

Use this as the production GPU baseline while the Cupyx/upwGS path is still experimental.  The preferred assembly target is the cooperative LU fused raw-CUDA kernel writing directly into CSR, then passing device CSR arrays to AMGX with no avoidable reconstruction.

Strengths:

- Most mature GPU solve path in the package.
- AMGX gives a strong preconditioner and predictable convergence for the current test family.
- Field reconstruction and device error evaluation have already been exercised in the normal GPU runner.

Weaknesses and caveats:

- The current BICGSTAB+AMG setup requires hundreds of iterations: 472 at p=6/ms=0.010 and about 609-611 at p=6/ms=0.008 for the tested manufactured-parameter variants.
- Row scaling remains important for convergence and should be folded into assembly whenever possible.
- It is not always the fastest pure iterate path: at p=6/ms=0.008, upwGS+BiCGSTAB had lower Krylov time, while AMGX was much steadier than the upwGS+GMRES outlier.
- Direct pointer handoff to PyAMGX/AMGX should be preferred over rebuilding CSR arrays.

Example:

```bash
LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib \
.venv/bin/python scripts/gpu/run_advection_reaction_cuda.py \
  -o 6 -ms 0.01 \
  --basis dub_orth \
  --trace-basis legacy-lagrange \
  --assembly-backend raw-cuda \
  --raw-local-assembly fused \
  --raw-lu-mode coop \
  --raw-matrix-format csr \
  --raw-block-size 32 \
  --amgx-maxiter 1500 \
  --check-rtol 1e-10 \
  -v 2
```

### upwSCC + Cupyx ILU(1) + Cupyx BiCGSTAB

Use this as the first GPU-native ILU comparison.  The upwind-SCC permutation and diagonal scaling are done before the solve, Cupyx builds ILU(1) directly on the device, and Cupyx BiCGSTAB performs the Krylov iteration.

Strengths:

- Avoids SciPy ILU for the preconditioner build.
- BiCGSTAB is faster than GMRES in the current ILU(1) test, with only a few callback iterations.
- Useful as a device-side lower-complexity baseline against AMGX.

Weaknesses and caveats:

- ILU(1) build can still be expensive relative to the solve.
- ILU(1) robustness has not yet been established across basis, order, mesh, and discontinuous-coefficient cases.
- This path is currently a benchmark harness, not the main production solver interface.

Example:

```bash
.venv/bin/python scripts/gpu/check_advection_upwind_scc_host_pyamgx.py \
  -o 6 -ms 0.01 \
  --basis dub_orth \
  --trace-basis legacy-lagrange \
  --cupyx-preconditioner cupyx-ilu1 \
  --cupyx-solver bicgstab \
  --cupyx-maxiter 1500 \
  --cupyx-tolerance 1e-13 \
  --check-rtol 1e-10 \
  --solve-variants ordered \
  -v 2
```

### upwSCC + Cupyx ILU(1) + Cupyx GMRES

Use this when comparing GMRES against BiCGSTAB under the same device-side ILU(1) preconditioner.  It is not the fastest current ILU(1) option, but it is useful for checking whether GMRES handles a harder variant more smoothly.

Example:

```bash
.venv/bin/python scripts/gpu/check_advection_upwind_scc_host_pyamgx.py \
  -o 6 -ms 0.01 \
  --basis dub_orth \
  --trace-basis legacy-lagrange \
  --cupyx-preconditioner cupyx-ilu1 \
  --cupyx-solver gmres \
  --cupyx-maxiter 1500 \
  --cupyx-tolerance 1e-13 \
  --check-rtol 1e-10 \
  --solve-variants ordered \
  -v 2
```

## Implementation Priorities

1. Move the forward upwGS preconditioner build into the Numba assembly loop.
2. Add device-only upwGS builders, first in CuPy and then in raw CUDA.
3. Push the cooperative raw-CUDA direct-CSR assembly kernel toward release quality.
4. Keep AMGX as the production baseline until the Cupyx/upwGS path is integrated into the reusable solver API and validated across basis/backend combinations; AMGX currently has the best robustness story across the p=6 mesh/parameter sensitivity checks.
5. Convert the recommended configurations above into named presets once the backend module cleanup is complete.
