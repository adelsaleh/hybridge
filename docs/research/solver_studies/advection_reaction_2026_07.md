# Advection-Reaction Solver Study: July 2026

Status: historical performance research, not a supported-backend contract.

This study records the July 2026 HDG advection-reaction solver comparisons.
The `upwSCC + forward upwGS + Cupyx Krylov` path showed degree- and
mesh-sensitive behavior that still needs explanation. Current production
choices must be taken from the solver API and backend capability documents,
not from this dated ranking.

Benchmark comparisons in this document are solver/preconditioner comparisons.  They intentionally exclude mesh setup, DG-space setup, coefficient projection, trace assembly, host/device transfers, COO-to-CSR conversion, reconstruction, and field-error evaluation unless a row explicitly says otherwise.

Benchmark evidence for the study is split across these reports:

- `run_logs/adv_rea_solver_benchmarks_p6_ms001_20260722.md`: p=6, mesh size 0.01 baseline comparison.
- `run_logs/adv_rea_solver_robustness_ms0008_20260722.md`: finer mesh size 0.008 degree and parameter sweep.
- `run_logs/adv_rea_upwgs_variability_20260722.md`: targeted upwGS variability follow-up, including BiCGSTAB and mesh sensitivity.
- `run_logs/adv_rea_amgx_variability_20260722.md`: matching raw-CUDA direct-CSR AMGX degree, mesh, and manufactured-parameter sensitivity follow-up.

## Observed Ranking

| Rank | Configuration | Studied role | Main reason |
| --- | --- | --- | --- |
| 1 | upwind-SCC + forward upwind block-GS + Cupyx GMRES | Fast experimental advection-reaction solves when tight residuals are needed and GMRES cycle counts are benign | Best observed solve-side cost on several cases, but p6/fine-mesh variability is not explained yet |
| 2 | raw-CUDA direct CSR + AMGX | Robustness baseline in this study | Stable device solve path with strong AMG preconditioning and field-error validation |
| 3 | upwind-SCC + forward upwind block-GS + Cupyx BiCGSTAB | Fast screening solve and robustness comparison for the upwGS preconditioner | Steadier p6/fine-mesh Krylov time than GMRES, but sometimes looser residuals |
| 4 | upwind-SCC + Cupyx ILU(1) + Cupyx BiCGSTAB | GPU-native ILU experiment | Fully device-side ILU(1) preconditioner with short BiCGSTAB solve |
| 5 | upwind-SCC + Cupyx ILU(1) + Cupyx GMRES | GPU-native ILU experiment for nonsymmetric robustness | Stronger Krylov method but slower than BiCGSTAB in the study's p6/ms0.01 timings |

## Experimental Configurations

### upwSCC + upwGS + Cupyx GMRES

This was the main performance-research preset in the study. The matrix was ordered by free trace-edge upwind SCC levels, then a forward upwind block-GS preconditioner was built from the ordered trace blocks. Cupyx GMRES performed the Krylov work on the device.

Strengths:

- Very low Krylov cost on favorable cases: one restarted GMRES cycle with a residual around `1e-14` at p=6/ms=0.01, p=5/ms=0.008, and p=7/ms=0.008.
- Preconditioner is much cheaper than strong SciPy ILU for the tested matrix.
- The preconditioner structure matches the advection direction and the HDG trace block formalism.

Weaknesses and caveats:

- The study runner built the preconditioner on the host and paid COO-to-CSR separately when running end to end.
- The proposed next target was constructing upwGS during assembly, first in Numba and later in raw CUDA.
- GMRES iteration counts in Cupyx are callback counts, which for restarted GMRES count restart cycles rather than every inner Arnoldi step.
- p=6 was sensitive on finer meshes in the study data: GMRES(50) needed 1 cycle at mesh size 0.010, 2 cycles at 0.009, and 6 cycles at 0.008.
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

This served as the fast comparison solve for the same upwGS preconditioner, particularly where GMRES restart-cycle counts jumped on fine p=6 runs.

Strengths:

- More stable p=6 fine-mesh Krylov time in the study data: roughly 1.1-1.3 seconds over the p=6 mesh/case variants at mesh sizes 0.008 and 0.009.
- Uses the same upwind-SCC ordering and forward upwGS preconditioner as the GMRES path, so it isolates the Krylov-method choice.
- Much cheaper Krylov time than GMRES(50) for the p=6/ms=0.008 outlier.

Weaknesses and caveats:

- The reported scaled residual can be looser than GMRES on p=6 fine meshes, around `1e-12` in several runs even with `info=0`.
- The study did not justify replacing GMRES for accuracy-sensitive runs because residual semantics and stopping behavior remained incompletely characterized.
- Like the GMRES upwGS path, it relied on a host-built preconditioner in the study runner.

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

This served as the July 2026 GPU robustness baseline while the Cupyx/upwGS path remained experimental. The measured assembly target was the cooperative-LU fused raw-CUDA kernel writing directly into CSR and passing device arrays to AMGX without avoidable reconstruction.

Strengths:

- Most mature GPU solve path in the package at the time of the study.
- AMGX supplied a strong preconditioner and predictable convergence for the tested family.
- Field reconstruction and device error evaluation were exercised in the normal GPU runner.

Weaknesses and caveats:

- The tested BICGSTAB+AMG setup required hundreds of iterations: 472 at p=6/ms=0.010 and about 609-611 at p=6/ms=0.008 for the manufactured-parameter variants.
- Row scaling materially affected convergence in these runs.
- It is not always the fastest pure iterate path: at p=6/ms=0.008, upwGS+BiCGSTAB had lower Krylov time, while AMGX was much steadier than the upwGS+GMRES outlier.
- Direct pointer handoff to PyAMGX/AMGX avoided rebuilding CSR arrays.

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

This provided the first GPU-native ILU comparison. The study applied the upwind-SCC permutation and diagonal scaling before Cupyx built ILU(1) on the device and ran BiCGSTAB.

Strengths:

- Avoids SciPy ILU for the preconditioner build.
- BiCGSTAB was faster than GMRES in the study's ILU(1) test, with only a few callback iterations.
- Useful as a device-side lower-complexity baseline against AMGX.

Weaknesses and caveats:

- ILU(1) build was expensive relative to the solve.
- The study did not establish ILU(1) robustness across basis, order, mesh, and discontinuous-coefficient cases.
- This path was a benchmark harness, not the main production solver interface.

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

This variant compared GMRES against BiCGSTAB under the same device-side ILU(1) preconditioner. It was not the fastest studied ILU(1) option but helped test whether GMRES handled harder variants more smoothly.

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

## Follow-Up Priorities Recorded In July 2026

1. Move the forward upwGS preconditioner build into the Numba assembly loop.
2. Add device-only upwGS builders, first in CuPy and then in raw CUDA.
3. Push the cooperative raw-CUDA direct-CSR assembly kernel toward release quality.
4. Retain AMGX as the study robustness baseline until the Cupyx/upwGS path is integrated into the reusable solver API and validated across basis/backend combinations.
5. Convert the promising study configurations above into named presets after backend cleanup.
