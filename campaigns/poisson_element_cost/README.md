# Poisson per-element cost on an H200

Purpose: find what limits the two per-element costs of the HDG Poisson solve on the real
article path (compatible drift, default AMG, Hybridge of 2026-10-31), test removing them, and
report measured savings with numerical parity.

1. The fixed host cost of RHS assembly and reconstruction (about 3.4 and 3.3 ms per solve).
2. The p=7-9 RHS and reconstruction device kernels on N=92,552.

Result: all tasks finished without failure, in 50.8 GPU-minutes on one H200. Results, parity
evidence and the ranked recommendation are in [Poisson local solves](../../docs/lessons/poisson_local_solves.md);
the full findings are in `docs/campaigns/poisson_element_cost/FINDINGS.md` (local, untracked), whose
section numbers the table below cites. Every experimental switch of the study defaulted to production
behaviour. The code of the study is now in the package; the section "Where the code is now" below
says which parts are production and which are switches.

Follow-up stages `factors_smoke`, `factors` and `factors_final` measure cached LU, stored responses,
stored inverses and the recovery operator's forms against the `revised` tree. Result in FINDINGS
section 8: the option `cache_local_factors="schur-lu+responses"` (tag `article-snapshot-20261102`).

Re-running: the trees compared by `run.py` (`TREES`) are local worktrees of development revisions
listed in the local records (`docs/campaigns/poisson_element_cost/HISTORY.md`). Recreate the ones a
stage uses before running it; the comparison can only be replayed where that history exists. The
current package cannot stand in for these trees: it has no `fixed` behaviour to compare against,
because the successful switches are always on.

## Layout

| File | What it is |
|---|---|
| `run.py` | Launcher: `--plan`, stages, resume from `DONE.json`, provenance through `campaigns/common/protocol.py`, `COMPLETE_<stage>.json` |
| `harness.py` | One process, one article row: `worker.run` from `gc_article_h200` plus spans, cProfile or kernel replay, all by wrapping module attributes |
| `analyze.py` | CPU-only tables: `arms`, `spans`, `probe` |
| `poisson_element_cost.sbatch` | One H200, environment of `campaigns/common/hopper_env.sh` and the article's tuning settings |

## Commands

From `/path/to/hybridge`. Stages: `smoke`, `attribution`, `host`, `kernels_p9`,
`kernels_p78` (not run), `round2`, `device`, `device2`, `lowdegree`, `response_check`, `boundary`.

```bash
# Inspect
.venv/bin/python -B campaigns/poisson_element_cost/run.py STAGE --plan
# Start (plain sbatch, no dependency)
sbatch --job-name=poisson-element-cost-STAGE --time=00:30:00 \
  --output=$PWD/run_outputs/poisson_element_cost/logs/STAGE-%j.log \
  campaigns/poisson_element_cost/poisson_element_cost.sbatch STAGE
# Resume: the same command skips tasks that have DONE.json (remove FAILED.json tasks by hand to retry)
# Stop
scancel JOBID
# Report
PYTHONPATH=../guiding-center-cg-hdg .venv/bin/python -B campaigns/poisson_element_cost/analyze.py arms run_outputs/poisson_element_cost/host
```




## Where the code is now (2026-11-18)

Not everything in this study was an experiment that failed. The parts that succeeded are
production code and have no switch. The parts that were declined or that failed are kept behind
default-off switches in `hybridge/mixed/raw_cuda/poisson.py` (kernel text in `poisson_source.py`), so
that they can be measured again. With no variable set, all 288 production kernel variants (policy,
phase, format, batch, retained coupling, two sizes) are token for token the kernels from before the
switches were merged.

| Item (switch name in the campaign) | Result | Where it is now |
|---|---|---|
| `signature` | Success, rank 1 | Production: `device_signature` in `runtime/cuda_tuning.py` (2026-11-02) |
| `resolved`, `nocheck` | Success, rank 2 | Production: `_launch` and `_check_failures` (2026-11-02) |
| `SKIP_INTERIOR` | Success, rank 3 | Production (2026-11-02) |
| `WARP_SOLVE`, `DIRECT_FACTORS` for cached LU | Success, rank 4 | Production (2026-11-02) |
| `RESPONSES` | Success (FINDINGS 8) | Production option `cache_local_factors="schur-lu+responses"` (tag `article-snapshot-20261102`) |
| `INVERSE` for cached LU | Works, not promoted (FINDINGS 6.1, 8) | Switch `HYBRIDGE_POISSON_LOCAL_FLAGS=INVERSE` |
| `WARP_SOLVE` for `schur-cholesky` | Works, not promoted: the policy is not at parity (FINDINGS 5) | Switch `CHOLESKY_WARP` |
| `boundary` | Works, small gain, not promoted (FINDINGS 2.1) | Switch `HYBRIDGE_POISSON_HOST_FLAGS=boundary` |
| `CONTRACT_Q`, `CONTRACT_COUPLING`, `LOAD_WIDE=N` | Failed: no gain (FINDINGS 4.1) | Switches of the same names |
| `--ptxas-options=-O1`, `--maxrregcount=80` | Failed: slower, and a small gain that the warp solve superseded (FINDINGS 4.1) | `HYBRIDGE_POISSON_KERNEL_OPTIONS` |
| `PHASE_CLOCK`, `STOP=k`, `NO_RESPONSE` | Diagnostics, not experiments | Switches of the same names |

The kernel switches go in `HYBRIDGE_POISSON_KERNEL_FLAGS` as a comma list of `NAME` or `NAME=VALUE`
(macro `EXP_NAME`). They apply to the RHS and reconstruction kernels, never to assembly.

- A promoted name (`signature`, `resolved`, `nocheck`, `WARP_SOLVE`, `DIRECT_FACTORS`, `SKIP_INTERIOR`,
  `RESPONSES`) is accepted with a warning and has no effect. An unknown name is an error.
- `INVERSE` is set through `HYBRIDGE_POISSON_LOCAL_FLAGS` only, for `schur-lu` and `schur-lu+responses`.
  The Cholesky arm of the first study (`cholinverse`) was not carried over, because the Cholesky cache
  hands the same factor array to other consumers.
- `LOAD_WIDE` acts on the factor copy to shared memory. Cached LU reads its factors in place now, so
  only the `schur-cholesky` kernels still make that copy.
- `CHOLESKY_WARP` needs a block of at least 32 threads.
- `PHASE_CLOCK` and `STOP` write into the kernel's validation argument. Only the kernel replay of
  `harness.py` allocates it, so a solve refuses these two.
- Kernels built with a switch have their own tuning records and never reuse a production record.

`tests/test_poisson_launch_resolution.py` checks on a device that each switch, other than the two
replay diagnostics, reproduces the production RHS and local unknowns to 2e-10.

The tuned launches of the recovery, Oswald averaging and gradient kernels were a separate study
(`guiding-center-cg-hdg`, campaign `gc_article_kernel_tuning`). Their first form made compatible
steps a median 9.7% slower. The memoized form was at parity and is production since 2026-11-08, as
`launch_tuned` with `resolved_selection` in `runtime/cuda_tuning.py`.

## Output

Bulk output: `run_outputs/poisson_element_cost/<stage>/` (`HYBRIDGE_CAMPAIGN_OUT` or `--out` overrides it).
Compact results: `docs/campaigns/poisson_element_cost/`.
The json, jsonl, csv, log, py, diff and pstats files of the 2026-11 bulk output are in
`run_outputs/poisson_element_cost/poisson_element_cost-text-20261108.tar.gz`; its 44G of `.npz`
factor and solution arrays were not kept.
Run with working directory `guiding-center-cg-hdg` (mesh cache), never a frozen tree.
