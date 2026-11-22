# ADR selection cost

Purpose: find and remove the per-assembly cost of ADR `raw.kernel.jit`, and test whether assembly stalls are recalibrations. Findings: `docs/campaigns/adr_selection_cost/FINDINGS.md` (local, untracked). The measured source is the recorded diff in each run's `source/`. The three compared trees (`TREES` in `run.py`: `pre`, `signature`, `resolved`) are two local worktrees of development revisions listed in the local records, and this tree; the comparison can only be replayed where that history exists.

Start or resume, from this checkout (completed tasks are skipped through `DONE.json`; stop with `scancel`):

    OUT=run_outputs/adr_selection_cost
    .venv/bin/python campaigns/adr_selection_cost/run.py first --plan
    sbatch --job-name=adr-jit-cost-first --time=00:40:00 --output=$OUT/logs/%x-%j.out campaigns/adr_selection_cost/adr_selection_cost.sbatch first --out $OUT/study
    sbatch --job-name=adr-jit-cost-timing --time=00:50:00 --output=$OUT/logs/%x-%j.out campaigns/adr_selection_cost/adr_selection_cost.sbatch second --out $OUT/study
    python campaigns/adr_selection_cost/analyze.py $OUT/study --records docs/campaigns/adr_selection_cost/outputs

Worktrees have no `.venv`; use `.venv/bin/python`. Output: `$OUT/study`; parity arrays: `/scratch/user/hybridge/adr_selection_cost`.
