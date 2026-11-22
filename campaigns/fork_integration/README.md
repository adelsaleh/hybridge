# Combined AMGX/PyAMGX integration qualification

The workflow below separates the CPU build/install from GPU qualification.
The existing ILU(k) regression checks do not qualify automatic ILU(k) tuning;
that work is deferred and the supplied-order backend remains experimental.

From the Hybridge repository root:

```bash
export HYBRIDGE_WORKSPACE_ROOT=/path/to/workspace
export HYBRIDGE_BUILDS_ROOT="$HYBRIDGE_WORKSPACE_ROOT/builds"
export HYBRIDGE_STACK_OUT="$PWD/run_outputs/fork-integration-build-$(git -C "$HYBRIDGE_WORKSPACE_ROOT/AMGX" rev-parse --short HEAD)"
bash campaigns/fork_integration/build_primary.sh --plan
sbatch campaigns/fork_integration/build_primary.sbatch
# After the CPU job passes and writes BUILD.json and COMPLETE:
bash campaigns/fork_integration/qualify.sh --plan
sbatch campaigns/fork_integration/qualify.sbatch
# Follow output and individual build/test logs:
tail -F fork-stack-build-JOBID.out
tail -F fork-integration-JOBID.out
# Resume either job with the same source and output settings:
sbatch campaigns/fork_integration/build_primary.sbatch
HYBRIDGE_CAMPAIGN_OUT="$PWD/run_outputs/fork-integration-JOBID" sbatch campaigns/fork_integration/qualify.sbatch
# Stop:
scancel JOBID
```

`build_primary.sh` uses a CPU allocation and requires clean committed AMGX,
PyAMGX and Hybridge checkouts. It configures Release, sm90, CUDA13.1, non-MPI;
builds `amgx`, `amgxsh`, `amgx_capi` and `amgx_tests_launcher`; and installs the
complete stack under `builds/AMGX/install-cuda13.1`. The existing primary
`AMGX-build-cuda13.1` alias is used for its CMake cache. Override it explicitly
with `HYBRIDGE_INTEGRATION_BUILD`; `HYBRIDGE_AMGX_INSTALL_ROOT` selects the
install prefix. Do not rebuild the primary library while a job uses it.

The CPU job archives the committed PyAMGX source into
`HYBRIDGE_STACK_OUT/pyamgx-source.tar`, builds a fresh wheel under `wheelhouse/`,
and installs that wheel under `python/` for isolated qualification. Cached
objects and generated C in the PyAMGX working checkout are not inputs.
`BUILD.json` binds the fork commits/trees, build recipes, toolchain, native
artifacts and wheel hashes. The extension must have RUNPATH to the canonical
grouped build. CPU outputs default to
`hybridge/run_outputs/fork-integration-build`; use a new `HYBRIDGE_STACK_OUT`
when inputs change. These local reproducible artifacts remain outside Git.

`qualify.sh` verifies `BUILD.json` and uses the isolated extension. It performs
no compilation or Python-environment installation. Check all running and
pending allocations against the shared five-GPU ceiling before submitting its
one-H200 job. GPU outputs default to `run_outputs/fork-integration-JOBID`,
selectable with `HYBRIDGE_CAMPAIGN_OUT`. Stage markers bind commands and log
hashes; resume rejects changed identities. Shared protocol/lifecycle helpers
record provenance and stop child process groups. `sources.json` records the
actual executing Hybridge checkout, including a frozen Git worktree, rather
than assuming it is the workspace's main checkout. Set `HYBRIDGE_REPO` to that
worktree when submitting and retain explicit workspace/build/output settings.

Historical runtime fixtures are independently verified through
`HYBRIDGE_AMGX_BASELINE_ROOT`; see [EXPERIMENTAL.md](EXPERIMENTAL.md). Their
recorded identity participates in resume, and candidate/baseline self-comparison
is rejected. Preserve those fixtures when retiring old build directories.

After GPU acceptance, install the exact recorded wheel in both environments:

```bash
wheel=$(.venv/bin/python -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["wheel"])' "$HYBRIDGE_STACK_OUT/BUILD.json")
.venv/bin/python -m pip install --no-index --no-deps --force-reinstall "$wheel"
../hybridge-venv-bsr/bin/python -m pip install --no-index --no-deps --force-reinstall "$wheel"
```

Use each environment's `python -m pip`: the copied BSR environment's `pip`
script points to the main interpreter. Confirm both extension RUNPATHs and
actual loaded libraries resolve to the primary grouped build before removing
retired builds. The same wheel is suitable for the currently matching Python
3.12/NumPy environments; verify ABI compatibility if either environment changes.

Coverage: native borrowed CSR/BSR/vector ownership, memory diagnostics, DILU
block sizes, reliable GMRES/BiCGSTAB/BSR and `TruncateMaxElements` tests; the PyAMGX suite; Hybridge borrowed
matrix and fixed-cycle tests; independent ILU(k) oracles including k5/10/20;
native ILU capacity checks; DILU embedding checks; representative GPU smoke.
These are bounded regression checks, not a guarantee over every AMGX mode.

The [native ILU synchronization lane](../matk/ILU_QUALIFICATION.md) and explicit
truncation lane qualify the remaining archived patches before consolidation.

Results of the qualified integration build: 18 native, 176 PyAMGX, 34 Hybridge borrowing and 272 GPU smoke checks;
92 ILU(k) oracle cases,20 capacity cases,10 DILU block sizes;16 unchanged-size
DILU/FGMRES arrays bitwise equal to the previous library. The existing dDFI
smoother rejection is reproduced separately against the old BSR build. These
baseline paths are local to this workspace. See
the committed audit (`docs/campaigns/fork_integration/README.md`, local, untracked).

## Incremental experimental builds

`build_variant.py` builds explicitly declared CUDA units from a clean, committed
AMGX worktree and links them with the unchanged objects from the qualified
integration build. This is a historical reproduction helper: its default
`cuda13.1-integration` baseline and `qualification.json` refer to the qualified integration build.
Once that build is retired, reproduction requires reconstructing its matching
source/library/dependency baseline. Do not point it at the new primary library
while retaining the old qualification record. The separate
[experimental qualification lanes](EXPERIMENTAL.md) record acceptance evidence.

From the Hybridge root, inspect a plan before running the same command without
`--plan` inside a Hopper CPU allocation. This helper neither allocates nor uses
a GPU, and compilation should not run on a login node. For the initial
truncation candidate:

```bash
export HYBRIDGE_BUILDS_ROOT=../builds
.venv/bin/python campaigns/fork_integration/build_variant.py \
  --source ../worktrees/AMGX/truncation-qualification \
  --output "$HYBRIDGE_BUILDS_ROOT/AMGX/qualify-truncation-COMMIT" \
  --replace src/truncate.cu --plan
```

The ILU candidate uses source worktree `ilu-qualification`, output
`qualify-ilu-COMMIT` (COMMIT is the candidate's AMGX commit), and two replacement arguments:
`--replace src/solvers/multicolor_ilu_solver.cu --replace src/csr_multiply_detail.cu`.
Choose a new output directory for every changed source/toolchain identity.
Resume an interrupted build by adding `--resume` to its exact original command;
stop its allocation with `scancel JOBID` or send SIGINT/SIGTERM to the helper.
Shared lifecycle cleanup stops the compiler process group before returning.

The plan validates the qualified integration build's source-manifest hash, checks its contents against
the committed integration source, and verifies the baseline library hash.
It audits all 164 library objects and their dependency files, expands the exact
CMake compile/link response files, and replaces every primary-checkout include
root with the candidate root. It rejects undeclared library units, shared-header
changes, compiler environment overrides and configuration changes; use a fresh
CMake build for these. Independent `src/tests/` additions and a registration
change confined to `tests_all` are recorded but are not built into this library.
Native test/checker compilation remains part of the qualification lane.

`--baseline`, `--qualification` and `--source-manifest` select relocated inputs;
the expected recorded hashes still apply. `--plan` writes nothing and prints
the full JSON inventory, so redirect it when a compact terminal view is wanted.
Planning hashes the baseline and dependencies and can take tens of seconds.
The output directory holds `plan.json` (source/object/dependency/toolchain hashes
and exact commands), shared `provenance.jsonl`, per-unit logs, verified
`.done.json` stage records, and `COMPLETE` with the resulting library hash.
The copied `CMakeCache.txt` supplies toolkit metadata to existing runtime
wrappers; this output is not a standalone CMake build tree. A completed build
still requires numerical qualification before any runtime promotion.

## Installed-stack tests in both repositories

After installing the exact qualified wheel, run the AMGX-related pytest suites
in Hybridge and the research repository on one H200:

```bash
export HYBRIDGE_REPO=/path/to/hybridge
export HYBRIDGE_RESEARCH_REPO=../guiding-center-cg-hdg
export HYBRIDGE_AMGX_BUILD_ROOT=../builds/AMGX/cuda13.1
bash campaigns/fork_integration/pytest_repositories.sh --plan
sbatch campaigns/fork_integration/pytest_repositories.sbatch
# Stop the allocation if needed:
scancel JOBID
```

The runner selects 456 Hybridge and 356 research tests, including wrapper,
precision, device-residency and downstream AMGX solves. It verifies the installed
binding and actually mapped library, records source/runtime hashes, and runs
both repositories even if one fails. Logs, JUnit counts, skip reasons and a final
completion record default to `run_outputs/fork-pytest-JOBID`; override with
`HYBRIDGE_CAMPAIGN_OUT`. Sources must be clean and remain unchanged during the
run. This runner does not resume: after interruption use a fresh output path.
Use `--only research` or `--only hybridge` to rerun an affected repository;
the default still runs both, and every report records its selected scope.
A successful run with skips is recorded separately and requires reviewing the
reported reasons before making a coverage claim.

## Dense LU cap check

`check_dense_lu_cap.py` tests whether `dense_lu_max_rows` applies on each exit from AMG coarsening,
on simple Laplacians and the article's Poisson configuration. Result: the cap
is ignored when coarsening ends by discarding a next level. Findings are in `AMGX/TODO.md`.

```bash
python campaigns/fork_integration/check_dense_lu_cap.py --plan
sbatch --job-name=amgx-dense-lu-cap campaigns/fork_integration/check_dense_lu_cap.sbatch [--output DIR]
scancel <job>   # stop; resubmitting resumes from each case's result.json
```

Output: `run_outputs/fork_integration/dense_lu_cap/` (or `HYBRIDGE_CAMPAIGN_OUT`), with final marker
`complete.json`. Compact records: `docs/campaigns/fork_integration/records/dense_lu_cap/`.
