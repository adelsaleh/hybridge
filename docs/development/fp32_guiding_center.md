# FP32 guiding-center benchmark

The guiding-center CLI accepts `--precision float32`. The experimental path is
qualified with the Gaussian-annulus k=3 preset, p=6, raw-CUDA assembly, native
FB-HP-MG-PCG Poisson, and AMGX FGMRES transport. FP64 remains the default
precision. FP32 automatically replaces the stock scaled, unpreconditioned
BiCGSTAB transport profile with FGMRES (restart 50, maximum 300 iterations).
An explicit `--transport-amgx-config` always takes precedence.

Numerical arrays for geometry, reference tables, fields, local factors, sparse
matrices, Krylov workspaces, reconstruction and diagnostic reductions use FP32.
CUDA kernels specialize their real types, math calls, literals and shared-memory
sizes. cuBLAS uses single-precision operations; both AMGX transport and the
Poisson coarse solver use `dFFI`. Array checks reject mixed precision, and the
native Poisson path fails explicitly if it cannot meet the requested residual.

This does not mean every instruction in the process is FP32: Gmsh and host
quadrature generators retain their native preprocessing precision, and Python
control/formatting scalars remain Python floats. Native AMGX also retains scalar
setup comparisons/conversions involving doubles. A three-step Nsight trace of
the earlier BiCGSTAB path showed FP32 numerical kernels throughout assembly,
factors, solves and reconstruction; the AMGX strength-setup kernel retains double comparisons, not
double matrix arithmetic. No TF32 or FP16 mode is requested.

## Local binding

The qualified PyAMGX binding checks for FP64 arrays regardless of AMGX mode.
Build the mode-aware variant locally using the existing development tools:

```bash
.venv/bin/python scripts/dev/build_pyamgx_precision.py \
  --source "$HOME/src/pyamgx-hdg-cuda13" \
  --output .cache/pyamgx-fp32 \
  --amgx-source "$HOME/src/AMGX-hdg-cuda13" \
  --amgx-build "$HOME/src/AMGX-build-cuda13"
```

This builds a separate extension; it does not install a package or edit the
source checkout. `--precision float32` loads this extension automatically and
isolates Numba, CuPy and mesh caches. Use a separate process for each precision.

## Run

From the repository root, select the existing CUDA/AMGX environment:

```bash
export HYBRIDGE_CUDA13_ROOT=/usr/local/cuda-13.0
export HYBRIDGE_AMGX_BUILD_ROOT="$HOME/src/AMGX-build-cuda13"
export HYBRIDGE_AMGX_INSTALL_ROOT="$HOME/src/AMGX-install-cuda13"
```

Start with approximately 12k triangles. Mesh size 0.025 produced 11,776 triangles
on the development machine. The minimum-triangle flag validates the generated
count; it does not itself refine the mesh.

```bash
bash scripts/gpu/run_cuda13.sh .venv/bin/python \
  -m scripts.guiding_center.run_guiding_center_cases \
  --preset diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx \
  --precision float32 \
  --mesh-size 0.025 --minimum-triangles 10000 \
  --dt 0.1 --num-steps 500 \
  --poisson-solver-rtol 2e-3 --poisson-solver-atol 0 \
  --transport-solver-rtol 5e-3 --transport-solver-atol 0 \
  --transport-amgx-tolerance 5e-3 \
  --transport-amgx-config configs/amgx/adv_rea_gpu4_hdg_fgmres_scaled_none.json \
  --plot-every 10 --plot-both --no-plot-mesh --verbosity 2 \
  --diagnostics-dir artifacts/guiding_center/fp32_fgmres_12k_plot \
  --diagnostics-prefix gaussian_k3_p6_fp32_fgmres \
  --screenshot-dir artifacts/guiding_center/fp32_fgmres_12k_plot/frames
```

These tolerances are also the FP32 CLI defaults. AMGX's tolerance applies to the
scaled solver system; the independent transport residual checks use the physical
system. Poisson's explicit residual is checked before acceptance. The FP32
iteration limits default to 500 for Poisson and 300 for transport.

The output directory contains diagnostic CSV/JSONL, per-step `_timings.csv` and
`_timings.jsonl`, a terminal log, and `_precision.json` with stage audits and
configured tolerances and the actual transport AMGX configuration. The command
displays density and potential and saves frames every 10 steps. Use
`--plot-off-screen` for PNGs without a live window, or `--plot-every 0` to
measure solver timings without rendering. For a short smoke run, replace `--num-steps 500` with `30`.

The original unpreconditioned FP32 BiCGSTAB run diverged at step 18 despite
passing earlier short checks. Increasing the accepted residual is insufficient
when the recurrence diverges. FGMRES completed all 500 steps to T=50 on 11,776
triangles with the same tolerances, plotting and phase logging. All recorded
HYBRIDGE precision audits remained float32 and AMGX used dFFI.

Validation on 2026-09-11:

- Transport iterations: median 8, maximum 38.
- Maximum independent relative transport residual: 6.97e-5 (physical system),
  7.88e-4 (scaled system); both below the 5e-3 requirement.
- Maximum Poisson relative residual: 1.9994e-3, below 2e-3.
- Final relative mass drift: 1.97e-4; energy drift: 1.283%.
- Final normalized mode amplitude: 0.70493, versus 0.70814 in the user's FP64
  BiCGSTAB run with the same mesh and tolerances.

The run artifacts are under `artifacts/fp32-fix/fgmres_12k_plot/`. Another GPU run
was active during validation, so its timings do not establish a speedup. For a
precision timing comparison, run sequentially with the same explicit FGMRES
configuration, mesh, tolerances and plotting settings in both precisions. Output
names do not choose precision: use separate directories and check `--precision`
and the run summary. These results do not qualify convergence on larger meshes;
the earlier 157,280-triangle FP32 Poisson residual plateaued around 8e-3.

For separate-process field exports and precision comparisons, use
`scripts/guiding_center/benchmarks/run_precision_benchmark.py`. It defaults to this same
smaller mesh, p=6, dt=0.1, 500 steps and the FP32 tolerances above. Override
`--precision float64 --rtol 1e-8 --poisson-rtol 1e-8 --atol 0` for an accuracy
reference. Timing comparisons must state the tolerances and iteration counts.
