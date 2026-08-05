# Run Configs

This directory stores small, version-controlled benchmark and solver presets.
The goal is to keep reproducible commands and the best observed settings close
to the code, without committing large console logs.

Recommended practice:

- Store the exact CLI arguments, not just a prose description.
- Record the mesh/problem parameters that make runs comparable.
- Record enough result metrics to rank the configuration: total time, global
  solve time, iterations, residual, and error.
- Keep failed or unavailable solver attempts when they explain why a path is
  not currently used.
- Treat timings as machine-local observations. Re-run the benchmark after major
  assembly, solver, PETSc, SciPy, or hardware changes.

Current files:

- `diff_rea_p6_lc01_solver_benchmarks.json`: archived solver sweep for
  `scripts/diffusion_reaction/experiments/bootstrap_initial_guess.py --test 3 -p 6 --lc 0.1 --tau 19`. The JSON basename is retained as a historical result
  identifier.

## Guiding-Center Response Files

`run_guiding_center_cases.py` accepts argparse response files with `@path`.
Use these for long GPU commands so terminal copy/paste cannot insert a newline
between an option and its value.  Lines may contain normal shell-like quoting,
blank lines, and `#` comments.

Append normal CLI overrides after the response file, for example `--verbosity 2`, `--mesh-size 0.008`, or `--transport-initial-guess initial-density-trace`.

Examples from the repository root:

```bash
scripts/guiding_center/run_local_amgx_cases.sh @run_configs/guiding_center/diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx.args
scripts/guiding_center/run_local_amgx_cases.sh @run_configs/guiding_center/diocotron_k100_p6_dt01_smoke_raw_cuda_amgx.args
scripts/guiding_center/run_local_amgx_cases.sh @run_configs/guiding_center/gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx.args
```

Current guiding-center response files:

- `guiding_center/diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx.args`: full `T=50` k=100 sharp annular-band raw-CUDA/AMGX stress run.
- `guiding_center/diocotron_k100_p6_dt01_smoke_raw_cuda_amgx.args`: short launch check for the same k=100 configuration.
- `guiding_center/gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx.args`: legacy Gaussian-annulus k=3 run using the full raw-CUDA/AMGX stack.
