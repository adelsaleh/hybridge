# AMGX builds after consolidation

Maintained matK solve, ordering, checkpoint and natural-ILU(k) launchers inherit
`campaigns/common/hopper_env.sh`. Its primary AMGX build is selected unless
`HYBRIDGE_AMGX_BUILD_ROOT` explicitly overrides it. Numerical arguments and
acceptance tolerances are unchanged. **ILU(k) remains experimental; automatic
tuning of its new options is deferred.** Running these launchers against a new
library does not reproduce an old library's timing or numerical identity.

Historical comparisons use the captured runtime fixtures under
`run_outputs/fork-integration-baselines/20261031/{main,bsr}`. Set
`HYBRIDGE_AMGX_BASELINE_ROOT` to relocate that fixture directory; its recorded
library bytes must remain unchanged. The main baseline is pinned to SHA256
`8fdb9d2377b2d065a90a87b30cf5e7ff8a0c94b2e8b09bb293e0f106e0a2154e`.
Comparison launchers reject a candidate identical to that baseline.

The historical `ilu_capacity`, `ilu_capacity_diagnostics`, `ilu_bfs_capacity`,
`ilu_numerics`, `ilu_sync`, `ilu_racecheck` and `ilu_paired_inverse` Slurm wrappers
require `MATK_ILU_BUILD` explicitly. Recreate the intended source variant before
using them; an old variant name is never an alias for the promoted primary.
The sync/paired inverse comparisons also accept `MATK_ILU_INVERSE_REFERENCE`
for the preserved inverse-reference report. Their original configurations, expected
outcomes and tolerances remain unchanged. Current explicit qualification is
documented in [`../fork_integration/EXPERIMENTAL.md`](../fork_integration/EXPERIMENTAL.md)
and uses `ilu_qualification.sbatch` or `truncation.sbatch` with the selected
builds and checker. These commands are for deliberate qualification.

`point2_qualify.sh` checks borrowed BSR/CSR/vector support with the selected
primary build and interpreter. `HYBRIDGE_NATIVE_TEST_LAUNCHER` overrides its
native executable. `amgx_dilu_check.sh STEP` compares the selected DILU build
with the captured main baseline. Supplying historical job keys additionally
requires `HYBRIDGE_HISTORICAL_SOURCE` to name the explicitly recreated worker
snapshot. The same requirement applies to `dilu69_reruns.sh`,
`rerun_campaign_jobs.sh` and `profile_campaign_job.sh`;
`HYBRIDGE_HISTORICAL_CAMPAIGN_OUT` selects preserved
specifications, and `HYBRIDGE_PYTHON` can select their matching interpreter.
The removed frozen source trees are never implicit defaults.

`unified_resume.sh` also requires that historical source when unfinished work
remains and selects the pinned main baseline for the original campaign. It
skips a completed output without requiring a source snapshot. The maintained
`adr_h200_tuned/lane_stage.sh` inherits the current primary build or its explicit
override.

Original `commands*.txt`, source snapshots and results remain unchanged.
Build snapshots and recipes are archived in
`docs/campaigns/fork_integration/records/build-layout-20261031/` (local, untracked).
They describe historical inputs; recreate a build in the shared `builds/AMGX/`
directory when exact historical reproduction is needed.
