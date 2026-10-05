# ARK3 runner wiring audit

Read-only comparison plus restoration of an isolated legacy runner. No builds or
simulations were run. This audit does not identify a wiring regression or prove
that the current GPU calculation reproduces an earlier successful calculation.

## Pasted failing run

The supplied command uses p=6, dt=0.5, 1,000 steps (T=500), h=0.0068 and
157,280 triangles. At the attempted endpoint t=1.5, explicit trace-rank failures
increase tau from 1,000 to 64,000 over six doublings. The step is then accepted.
The next step fails in its first implicit transport solve. Eight further
doublings increase tau from 64,000 to 16,384,000 without an accepted step. The
last accepted state is t=1.5; the per-step limit correctly resets after success.
`edges=[]` at this later failure means no rank-deficient face was established by
the available diagnostic classifier, not proof of a nonsingular matrix.

## Source comparison

- The current ARK3 module and the pre-refactor snapshot are AST-identical after
  normalizing import relocations, docstrings, and the broadened transport-failure
  handler. Tableau, split, scaling, stages, accepted update, and replay are unchanged.
- `_make_poisson_options` is text-identical. `_make_transport_options` differs only
  in the repository-root path depth after relocation. Resolved options compare
  equal at dt=0.5 and 1.0 with the supplied mesh and tau settings.
- The current factory selects IMEXARK3Stepper with the same boundary callbacks,
  residual workspace, dt, Poisson solver and tau policy inputs as the old branch.
  The callback refreshes the matrix at stage 2 and changes only the RHS for the
  remaining two stages, as in the archived runner.
- `hybridge/assembly/advection_residual.py`, `hybridge/core/field_ops.py`,
  `hybridge/backends/diffusion_cupy.py`, and `hybridge/solvers/advection_reaction.py`
  are byte-identical to `/tmp/gc-stepper-refactor-jtrkkr6d`. This comparison does
  not cover every shared module, binary, configuration file or earlier version.
- Four canned-solver checks at dt=0.5 and 1.0, with and without injected trace-rank
  failure, produce identical stage sources, betas, reuse flags, Poisson sources,
  warm starts, and accepted endpoints. These are orchestration checks, not physics
  or stability tests.

## Earlier completed-run evidence

The preserved `imex_ark3_t50_audit` under
`/tmp/hybridge-cleanup-20260913-wnc2yhb3/artifacts/` records a completed run with
exactly two tau increases. Its parameters were dt=0.05, h=0.008, p=6,
113,894 triangles and T=50. Tau increased 1,000 -> 2,000 at step 215 and
2,000 -> 4,000 at step 546. This verifies that successful run, without excluding
other successful dt=0.5 or 1.0 runs whose logs have not been identified.

The production output prefix is reused across runs. During this inspection its
on-disk timing history was replaced with a dt=0.1 run, so those files cannot be
used as an immutable record of the pasted dt=0.5 run. Use distinct output prefixes
for the legacy and current comparisons.

## Restored command

See [the legacy runner](../../../scripts/guiding_center/reference/legacy_ark3/README.md)
for the runnable dt=0.5, T=500 command, source hashes and compatibility limits.
It preserves the pre-refactor runner and original rank-only tau policy, while
using the currently installed numerical backends. Its CLI dry-run passed; no
claim of full GPU runtime validation is made.
