# matK CPU/GPU solver investigation

Generic MATLAB HDF5 CSR/RHS benchmarks with verbosity 0–3. `common.py`
validates index bases, CSR structure, the supplied RHS and original-system
residuals. Solver choices, precision and tolerances are explicit arguments.

The preserved zero-start ILU(10)/PFGMRES result
has original relative residual 1.65e-11. Later experimental results retain
their own acceptance outcomes and pinned library hashes.

See [AMGX build retirement and overrides](RETIREMENT.md) before rerunning old
commands. Maintained launchers use the selected primary build; historical
comparisons retain pinned baselines. ILU(k) remains experimental and its new
automatic tuning is deferred.

## Source and records

- `benchmark_pardiso.py`: CPU sparse LU and phase timings; see `commands.txt`.
- `benchmark_amgx.py`: the initial AMGX sweep.
- `benchmark_amgx_strong.py`: isolated candidates, scaling, optional PARDISO
  permutation, checked checkpoints and original-system residual acceptance.
- `native_*proposals.json`: tracked candidate definitions;
  `converged_pardiso_ilu10_fgmres.json` is the accepted configuration.
- `commands*.txt` and `*.sbatch`: existing study reproductions and launchers.
- `test_*.py`, `check_*.py` and native checks: small loader, permutation,
  monitoring, configuration and independent solver qualification.
- [ILU synchronization qualification](ILU_QUALIFICATION.md): independent
  application oracles and CUDA sanitizer gates for the archived corrective patches.

Canonical source is `campaigns/matk/`. The tracked compatibility link
`scripts/matk -> ../campaigns/matk` preserves historical commands. Compact
reports, audits and checkpoint manifests live in
`docs/campaigns/matk/` (local, untracked); source-directory
links retain their old names. The relocation manifest records original
paths and hashes. Historical records and command files retain their bytes.

Large input, checkpoints and raw logs stay outside Git. The local preserved
inventory is `run_outputs/matk/legacy/{inputs,saved_iterates,logs,cache}/`.
Compatibility links keep `campaigns/matk/matK.h5` and the recorded checkpoint
paths usable. The input and binary checkpoints are not distributed with the
repository; `MATK_MATRIX`/`--matrix` selects a supplied input. See
`saved_iterates/manifest.json` (`docs/campaigns/matk/records/saved_iterates/manifest.json`, local, untracked)
for checkpoint sizes and SHA256 hashes.

## Inspect and plan

Run from the Hybridge repository root. Listing candidates requires no matrix
load or GPU allocation; this is the existing plan interface:

```bash
mkdir -p run_outputs/matk/logs
.venv/bin/python campaigns/matk/benchmark_amgx_strong.py \
  --list-configs --output run_outputs/matk/plan-only.json \
  > run_outputs/matk/candidate-plan.json
.venv/bin/python -m unittest discover -s campaigns/matk -p 'test_*.py' -v
```

`--inspect-only` loads the full 7.1 GiB input to check it without factorization.
The small tests use generated fixtures. Historical experimental libraries and
saved permutations must be available for their corresponding launch commands;
rebuilding the current fork does not reproduce every old library hash.

## Start and outputs

Check all running **and pending** allocations before submitting. The shared
limit is **five GPUs total**, including work in other repositories. These
examples are independent submissions; each AMGX job requests one H200.

```bash
squeue -u "$USER" -o '%.18i %.32j %.10T %.24b %.18N'

# One explicit full AMGX candidate; additional choices are in candidate-plan.json.
sbatch --output=run_outputs/matk/logs/amgx-strong-%j.out \
  campaigns/matk/amgx_strong.sbatch \
  --candidates fgmres_natural_ilu0_b20 --verbosity 3

# CPU sparse LU, using the existing exclusive Genoa allocation.
sbatch --output=run_outputs/matk/logs/pardiso-%j.out \
  campaigns/matk/pardiso.sbatch --verbosity 2
```

AMGX and PARDISO reports default to `run_outputs/matk/*-JOBID.json`; worker
logs/configurations live beside the corresponding report. `MATK_OUTPUT`
overrides a single-report launch, and `--output` selects direct Python output.
Checkpoint generation defaults to
`run_outputs/matk/saved_iterates/reference-JOBID/`; `MATK_CHECKPOINT_DIR`
overrides it. Set `MATK_REPO` when submitting outside the repository root.
The more specialized launchers document their output subdirectories in their
adjacent `commands*.txt` files.

Reports record native status and independently recomputed original residuals;
only passing candidates enter the ranking. A final sweep status such as
`completed`, `no_converged_candidate` or `smoke_failed` records what finished.
Checkpoint generation ends with `index.json` status `saved`.

## Resume and stop

These historical drivers have **manual resume**. Inspect the previous JSON's
`results`, `planned_candidates` and `untried_candidates`, then submit the
remaining names with `--candidates` to a fresh output. Reusing a sweep's output
path overwrites that report; checkpoint generation refuses an existing
`index.json`. Integration with the shared campaign lifecycle remains future
work; no automatic completion-marker skipping is claimed here.

```bash
# Resume selected unfinished candidates under a fresh job/report identity.
sbatch --output=run_outputs/matk/logs/amgx-resume-%j.out \
  campaigns/matk/amgx_strong.sbatch \
  --candidates fgmres_natural_ilu0_b20 --verbosity 3

# Continue from a saved iterate only for the same checked matrix and RHS.
sbatch --output=run_outputs/matk/logs/amgx-warm-%j.out \
  campaigns/matk/amgx_strong.sbatch \
  --candidates fgmres_natural_ilu0_b20 --scaling column-max \
  --initial-guess "$PWD/campaigns/matk/saved_iterates/reference-JOBID/bicgstab_unpreconditioned.solution.npz" \
  --save-solution

# Stop the selected allocation, including its worker processes.
scancel --signal=TERM JOBID
```

Warm starts verify matrix/RHS and vector hashes and preserve the original
residual target. They are numerical continuation, separate from skipping
completed trials. Historical GPU-limit comments in command transcripts are
superseded by the current five-GPU rule above.
