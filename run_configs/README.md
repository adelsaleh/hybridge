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

- `diff_rea_p6_lc01_solver_benchmarks.json`: solver sweep for
  `solvers/diff_rea_w_boostrap.py --test 3 -p 6 --lc 0.1 --tau 19`.
