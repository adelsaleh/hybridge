# Experimental AMGX consolidation

Both qualification lanes passed and their branches are
merged in the AMGX fork. The native ILU lane passed 28 capacity cases, 40
numerical cases, 40 synccheck and 32 racecheck invocations. The truncation lane
passed 191 checks plus three native modes. Exact candidate identities and
limits are in the ILU record (`docs/campaigns/matk/ilu_synchronization_qualification.md`, local, untracked)
and truncation record (in `docs/campaigns/fork_integration/records/`, local, untracked).
See the final promotion record (`docs/campaigns/fork_integration/primary_promotion.md`, local, untracked) and
[installation workflow](README.md).

The existing determinant threshold and fallback arithmetic are preserved.
Diagnostic counters are disabled by default. Other AMGX TODO items, including
the numerical conditioning limitations of native block inversion, are separate
from these synchronization and launch corrections.

Supplied-order ILU(k) remains experimental and its autotuning is
deferred. Direct
coverage of symbolic subwarp variants and deliberately induced hash-failure
exits remains pending as a separate native ILU0/1 coverage limit, recorded
explicitly in the qualification evidence.

## Historical runtime fixtures

Before rebuilding the main runtime, preserve minimal comparison inputs from
the older builds. These fixtures contain libraries, toolchain metadata and one
native test executable, not full build trees or source worktrees:

```bash
.venv/bin/python campaigns/fork_integration/capture_baselines.py \
  --main-build ../builds/AMGX/cuda13.1 \
  --bsr-build ../builds/AMGX/cuda13.1-bsr \
  --output run_outputs/fork-integration-baselines/20261031 --plan
# Repeat without --plan to capture and hash-check the files.
```

Capture the fixtures before runtime promotion. Do not recapture
the historical baseline from the now-promoted primary build. The output's `manifest.json` records
each original path and SHA256. The capture tool refuses to replace changed fixtures; the qualification
runner verifies the preserved copies directly. Preserve these local fixtures
when retiring the old build directories. Historical binary hashes are not
evidence for a different source revision.

`qualify.sh` takes `HYBRIDGE_AMGX_BASELINE_ROOT` (defaulting to this workspace's
`run_outputs/fork-integration-baselines/20261031`) and verifies its manifest.
The recorded baseline identity participates in resume checking. A candidate
equal to the baseline is rejected, preventing a meaningless comparison after
the main runtime is promoted. `HYBRIDGE_WORKSPACE_ROOT` selects the source
workspace when the runner executes from a frozen Hybridge worktree.

## Qualification lanes

ILU qualification covers setup and triangular-application synchronization,
capacity boundaries, FP32/FP64, both block layouts, independent application and
inverse checks, and zero-warning CUDA racecheck/synccheck. The optional counters
are checked enabled and disabled without changing the numerical policy.

Truncation qualification uses the maintained native oracle and
`campaigns/matk/qualify_truncation.py`. `--plan` lists cases without initializing
CUDA; `--fixed-build`, `--baseline-build` and `--executable` select immutable
inputs. Its completion markers are bound to source/binary identities and log
hashes. Resume uses the same output path and inputs; stop with `scancel JOBID`.
The baseline must reproduce the documented shared-memory fault, while fixed
tests cover every cap 1–32, supported real precisions, partial blocks, launch
geometry boundaries and grid-stride iterations.

Check all running and pending allocations before submission. Each lane uses
one H200; the workspace-wide limit is five GPUs. Qualification outputs remain
under `run_outputs/`; compact evidence belongs in
`docs/campaigns/fork_integration/records/`. No pass or merge is inferred from a
plan or a successful build alone.

The maintained truncation runner defaults to the selected primary build and
captured historical main baseline. Its standalone checker must be compiled
against the selected source and passed with `--executable`; use ELF RUNPATH
(`--enable-new-dtags`) so baseline selection can override its library path.
The archived truncation build/link records contain the exact original recipe.
Ordinary merged-runtime checks use the CMake-registered `TruncateMaxElements`
regression without rerunning the historical fault campaign.
