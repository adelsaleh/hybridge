# Native ILU synchronization qualification

The candidate AMGX build passed the H200 qualification: 28 capacity cases, 40 native oracle cases, 40 synccheck
and 32 zero-warning/error racecheck runs, plus four smoke checks. The
committed result (`docs/campaigns/matk/ilu_synchronization_qualification.md`, local, untracked)
records source/library identities and evidence limits. Source scope and
historical failures are documented in AMGX's
`docs/ilu_synchronization_qualification.md`.

The distinct supplied-order ILU(k) backend stays experimental and its automatic
architecture tuning is deferred. The 2/4/8/16-thread symbolic overloads and induced hash-failure exits remain
unqualified on GPU; their fixture is inactive deferred work. This campaign
qualifies corrective native multicolor ILU(0/1) changes, not a new default
preconditioner or altered numerical settings.

`check_ilu_synchronization.py` extends the existing native C-API oracle. It
checks coupled zero/nonzero-guess applications, external-diagonal ILU1,
8193-row paired inverses, 16385 rows per color for application workspace
reuse, and optional inverse counters with repeated setup. Its independent
two-color block algebra is cross-checked against the existing sparse block
factor reference by `--host-check`.

The scaled inverse cases preserve the existing determinant cutoff and diagonal
fallback. A successful policy check does not assert an exact inverse for those
blocks. Deliberately zero/NaN/Inf blocks test counter classification only.
Both real precisions and block layouts are covered; complex and distributed
execution are outside this qualification.

The maintained runner `qualify_ilu_synchronization.py` supports `--plan` and
resumes completed steps when source, library and commands match. It saves
source/library hashes, host/device/toolchain provenance, per-step JSON/logs
and `completion.json`. Racecheck passes require a completed summary with zero
warnings and errors. Synccheck passes require zero errors. Every worker also
checks its numerical or diagnostic contract.

From the Hybridge repository:

```bash
.venv/bin/python -B campaigns/matk/check_ilu_synchronization.py --host-check
.venv/bin/python -B campaigns/matk/qualify_ilu_synchronization.py --plan --stages smoke
sbatch --time=00:20:00 campaigns/matk/ilu_qualification.sbatch --stages smoke
sbatch campaigns/matk/ilu_qualification.sbatch --stages numerical racecheck synccheck
sbatch campaigns/matk/ilu_qualification.sbatch --stages reuse-racecheck
```

Check all running and pending GPU allocations before each submission. Each job
uses one GPU; the coordinated ILU lane may use up to three concurrent jobs,
while respecting the workspace's shared five-GPU cap and other agents' allocations.
The last two commands are separate stages of one qualification;
all must pass. Set `MATK_ILU_BUILD` to an isolated committed candidate build;
the default is `../builds/AMGX/qualify-ilu`. Set `HYBRIDGE_CAMPAIGN_OUT` to resume
the exact same stage selection. Default output is
`run_outputs/matk/ilu-qualification-JOBID`. Stop with `scancel JOBID`; the
runner records interruption and terminates its worker process group. Keep
source and the linked library unchanged while a job is running. Changed source
or candidates require a new build/output identity.

Collect compact evidence under `docs/campaigns/matk/`, retaining complete logs
and arrays in `run_outputs/`. The earlier failed sanitizer records remain
historical evidence and are not overwritten by this qualification.
