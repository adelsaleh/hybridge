# Archived IMEX-ARK3 comparison runner

Restored from `/tmp/gc-stepper-refactor-jtrkkr6d`, the snapshot immediately before
splitting the monolithic guiding-center runner into the current runtime modules.
`provenance.json` records the source and restored SHA-256 hashes of every file.
Only import/file-location adjustments and an ARK-only CLI guard were applied.
The archived runner, presets, ARK stepper, stage support, and rank-only tau policy
are independent of the current temporal registry and runtime runner.

This is the Kennedy–Carpenter IMEX-ARK3 implementation, not a different RK method.
It shares the installed HDGFEM assembly, field operations, solver backends, case
builders, AMGX configurations, and precision/bootstrap support. It is not a full
historical environment. No compiled libraries or numerical caches were restored.

From the repository root:

```bash
# Euler vortex-gas turbulence — disk — legacy IMEX-ARK3 — p=6, dt=0.5, T=500
.venv/bin/python -m scripts.guiding_center.reference.legacy_ark3.runner \
  @run_configs/guiding_center/euler_disk_vortex_gas_legacy_imex_ark3_p6_h0068_dt05_t500.args
```

Append `--dry-run` for configuration inspection. The supplied args use a separate
output prefix and record diagnostics every accepted step. The archived CLI
supports PyVista live plots; it predates `--plot-diagnostics`. Saved diagnostic
histories can be rendered afterward with the current diagnostic plotting tool.

For the same T=500 at dt=1.0, append `--dt 1 --num-steps 500` and
`--diagnostics-prefix euler_disk_vortex_gas_legacy_imex_ark3_p6_h0068_dt1_t500`.

The old recovery policy doubles tau only for diagnosed active trace-rank loss.
A generic failed transport solve without those diagnostics propagates immediately.
The current production runner retains recovery for numerical transport failures.

Validation: CLI import/argument resolution and dry-run succeeded. Four canned-solver
checks compare old/current stage inputs, drift, endpoint values, reuse flags, and
rank-failure replay at dt=0.5 and 1.0. No mesh generation, builds, JIT kernels, PDE
solves, or time integration were run. Full host/GPU runtime compatibility is not
yet established.
