# ADR branch baseline on the RTX PRO 5000

## Scope and checkout

The first incremental step toward comparing ASM + polynomial-preconditioned
GMRES with AMGX is to qualify the existing ADR runners on this machine.
The baseline is `gpu_gmres_precondit` at
`d44acce873b18daa6507e0c89b20a9e4e5ab913e`, checked out in
`~/src/hdgfem-gmres`. The main checkout remains on `master` with its
uncommitted work preserved. No assembler or solver code was changed for this baseline.

## Environment

The tests reuse `.venv/bin/python` without installing
or replacing packages. Python is 3.12.3, NumPy 2.5.3, SciPy 1.18.1,
Numba 0.67.0, CuPy CUDA 13 package 14.2.0, and pytest 9.1.1.
The GPU is an NVIDIA RTX PRO 5000 Blackwell with approximately 48 GiB VRAM;
the driver is 580.126.09. CuPy reports CUDA runtime 13.2 and NVRTC 13.0.
`pip check` passed. BLAS/OpenMP thread counts are fixed to one.
Runtime CUDA/Numba JIT was explicitly authorized for these ADR tests and
benchmarks; no package or native-library build/install was performed.

This is a tested existing-machine environment, not a reproduction of the
branch's locked environment. `scripts.validate_gpu_environment` rejects it
because NumPy and Numba exceed its declared version ranges and the Python
`cuda-toolkit` distribution is absent. A system CUDA toolkit exists at
`/usr/local/cuda-13.0`; the actual CUDA tests pass. The branch targets Python
3.13, while this baseline uses 3.12.3. Keep these deviations attached to results.

Local artifacts are under `run_logs/adr_baseline_20260917/` (Git-ignored).
They include `environment.sh`, `pip-freeze.txt`, the strict preflight log,
pytest log/XML, and campaign output directories with environment metadata,
raw measurements, and summaries. To use the same environment:

```bash
cd ~/src/hdgfem-gmres
source run_logs/adr_baseline_20260917/environment.sh
```

## Commands and results

The following are the executed command arguments after selecting the above
environment. Output directories already contain results; use new directories
for a fresh run, or the performance runner's documented resume mechanism.

```bash
python -m pytest -ra tests/test_adv_diff_rea.py \
  tests/test_adv_diff_rea_performance.py \
  --junitxml=run_logs/adr_baseline_20260917/pytest.xml

python scripts/run_adv_diff_rea_campaign.py \
  --meshes 4 8 16 --orders 1 2 --repeats 3 --profile \
  --output run_logs/adr_baseline_20260917/validation_cpu

python scripts/run_adv_diff_rea_campaign.py \
  --cases quadratic variable_velocity trigonometric \
  --meshes 4 8 --orders 2 --assembly-backend cupy --solver gpu \
  --preconditioner asm_poly --polynomial-degree 6 --restart 75 \
  --rtol 1e-12 --warmup 2 --repeats 5 --validate-gpu \
  --output run_logs/adr_baseline_20260917/validation_gpu

python scripts/run_adv_diff_rea_performance.py \
  --level smoke --output run_logs/adr_baseline_20260917/performance_smoke
```

- Both test files: **37 passed, no skips**, in 8.72 seconds. This includes
  CPU/GPU assembly parity, six GPU preconditioner families, and a CUDA
  performance-worker measurement/profile check.
- CPU validation: **30/30 cases passed**, maximum reported true relative
  residual `2.41625e-14`.
- CUDA validation with ASM + PP + GMRES: **6/6 cases passed**, maximum reported
  true relative residual `2.74847e-13`.
- GPU performance smoke: **completed**, **60/60 eligible solver candidates**,
  no issues. All **190 jobs passed**: 10 assemblies, 60 solver measurements,
  60 profiles, and 60 end-to-end confirmations. All five cases were tested
  at p=2 on 4x4 and 8x8 rectangular-cell meshes, using six preconditioner
  families. The worst measured candidate true relative residual was
  `9.58365e-13`, below the `1e-12` target. Each solver measurement used two
  warmups and five timed repetitions with fresh algebraic setup.

For ASM + PP at p=2 on the 8x8 mesh (528 free trace DOFs), the measured
median fresh-setup-plus-solve times were:

| Case | Setup + solve (ms) | Worst true relative residual | L2 error |
|---|---:|---:|---:|
| quadratic | 12.479 | 1.05603e-13 | 6.04467e-13 |
| variable_velocity | 12.610 | 6.19505e-14 | 4.59165e-13 |
| trigonometric | 12.034 | 2.74847e-13 | 7.54392e-3 |
| advection_dominated | 10.055 | 2.48385e-13 | 7.22260e-3 |
| anisotropic | 14.804 | 5.75526e-13 | 5.82460e-3 |

These configurations use `raw_fused` operator application, fused ASM,
`cublas_inverse`, polynomial degree 6, restart 30, and CGS2 for both GMRES
and polynomial setup. See `performance_smoke/completion.json`,
`coverage.json`, `candidates.csv`, and `jobs/` for authoritative evidence.

## Interpretation and next comparison

The smoke cases include smooth manufactured transport-dominated diffusion
`K=1e-3 I` and anisotropic diffusion `K=diag(1, 0.01)`, with velocity `(1, 0.5)`
and reaction `1`. They do not validate unresolved boundary layers. Small-mesh
performance smoke results are qualification evidence, not a large-system
solver recommendation.

The two branches have materially different ADR implementations and defaults:

- This branch prepares coefficient integrals on CPU, transfers dense local
  matrices to CuPy for inversion/condensation/global assembly, and transfers
  data back for elimination and reconstruction. It supports SPD tensor
  diffusion in this path and defaults to diffusive stabilization `1` plus
  `max(beta.n, 0)`.
- The current master implementation offers NumPy, fused Numba, and raw-CUDA
  paths, with default advective stabilization `abs(beta.n)` and
  diffusion stabilization `kappa / global_length`. Its TODO still tracks
  extending fused Numba/raw-CUDA ADR to variable/tensor diffusion.

Before selecting an assembler, match the mesh, polynomial and trace bases,
quadrature/coefficient representation, stabilization, and boundary elimination;
then check matrix/RHS and reconstructed-solution parity. Compare warmed
assembly timings and memory separately from solver setup/solve. For the
AMGX comparison, give both solvers the same reduced operator and right-hand
side, record an AMGX configuration sweep, and use the same initial guess,
precision, scaling, and independently checked physical-residual target.
At this baseline stage, assembler selection and the larger solver comparison
were pending. They are now recorded in the follow-up reports; solver
optimization remains a separate open task.

The next stage is recorded in [the matched assembler comparison](adr_assembler_comparison_2026_09_17.md).

The completed large study is in [the ADR solver comparison](adr_solver_comparison_2026_09_17.md).
