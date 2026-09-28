# Full archived ADR PyPardiso campaign

Entry point: `scripts/advection_diffusion_reaction/campaigns/pardiso/run_adr_pardiso_campaign.py`.
This is a separate, CPU-only campaign. It does not edit either manuscript,
rerun iterative solvers, regenerate meshes, evaluate coefficients, or change
the archived matrices or reference solutions. Run it **after the other campaign
finishes**, on an otherwise idle machine, to avoid CPU/RAM contention.

## Scope

The default read-only inventory contains **107 distinct matrix/RHS systems**:
101 from the completed report coverage registry and six from the completed
nine-lobed annular stress campaign. It includes everything in the synthesis,
plus the earlier exploratory controls retained in the full report.

- Smooth and oscillatory square/annulus L1–L4 ladders, not only endpoints.
- Degree sweeps at p = 1, 2, 3, 4, 6; their p = 6 / L2 overlaps are solved once.
- All six near-million-DOF p = 3, 4 focused endpoints.
- Original oscillatory/weak-anisotropy controls, including degree four.
- Trapping, crossing, and orthogonal stress systems at both 98,699 and 148,848
  actual triangles (100k/150k nominal), up to 1,547,455 trace DOFs.

Every report occurrence remains in the manifest's `origins`. Identical archived
systems shared by different studies are deduplicated by operator/RHS identity.
The ongoing **square stress campaign is not included** in these manuscript
archives; its results will be considered separately when it finishes.

## Commands

From `/home/adelsaleh/src/hdgfem`, inspect the plan without writing files or
starting a solver:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B scripts/advection_diffusion_reaction/campaigns/pardiso/run_adr_pardiso_campaign.py \
  --autotune-threads --threads 1 2 4 6 8 12 16 24 --tuning-repeats 3 --repeats 5
```

Launch after the current campaign finishes; **no CUDA wrapper is required**:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u -B scripts/advection_diffusion_reaction/campaigns/pardiso/run_adr_pardiso_campaign.py \
  --output run_outputs/solver_studies/adr_scaling_2026_09_17/pardiso_tuned_2026_09_22 \
  --autotune-threads --threads 1 2 4 6 8 12 16 24 \
  --tuning-repeats 3 --warmup 1 --repeats 5 \
  --max-dofs 2000000 --max-rss-gib 64 --reserve-gib 16 \
  --timeout 7200 --execute
```

Thread counts are tuned **independently for every matrix**, including the
coarsest cases. A fixed 24-thread baseline is not the recommended comparison.
Omitting the explicit list with `--autotune-threads` uses the same ladder up
to the detected physical-core count. Serial execution is always a default
candidate; no DOF-based thread cutoff or presumed optimum is imposed.

The recommended sweep has 856 pilot jobs and 107–214 independent confirmation
jobs if every system has a passing candidate. Each job is sequential and
process-isolated. This is more expensive than the previous fixed-thread run.
The thread count found is **best among those tested**, not a guaranteed global
optimum. For an exhaustive integer-count sweep on this 24-core machine,
replace the list by `--threads {1..24}` in Bash or Zsh; use a separate output
directory. Hyperthreads are not part of the default physical-core ladder.

The worker sets and checks MKL's thread limit, disables dynamic adjustment,
sets OpenBLAS to one thread, and clears inherited MKL per-domain thread limits.
OMP's thread limit is set to the candidate count. Numba JIT remains disabled.
Without `--autotune-threads`, the original fixed-count / explicit-list
measurement mode is still available; it does not claim to select an optimum.

For progress, run the same script with `--output ... --status`. Resume with
the **identical execution command plus `--resume`**. Terminal failures are
preserved; add `--retry-failed` to retry them into new attempt directories.
Changing sources, cache metadata, thread counts, or protocol requires a new
output directory. A lock prevents concurrent drivers using the same output.

## Timing and correctness protocol

Thread autotuning has two disjoint stages:

1. **Pilots:** all candidate counts receive one discarded warmup followed by
   three measured setups. Their order is reproducibly shuffled per system.
   Failed or incomplete pilots cannot win.
2. **Confirmation:** choose the lowest pilot median fresh time and lowest pilot
   mean reused time separately. Each distinct winning count gets a **new worker
   process**, one discarded warmup and five new measured setups. If the same
   count wins both objectives, one confirmation job supplies both. Exact pilot
   ties prefer fewer threads. Confirmation timing does not reselect the winner.

Pilot minima are never the reported CPU baseline. Only the preselected
objective of its confirmation may produce a CPU/GPU speedup. In particular,
a confirmation selected only for fresh solves cannot be substituted as the
reused winner (or vice versa), even if it happens to measure faster.
This separates parameter choice from performance reporting. Raw sample ranges
are retained; close differences need follow-up repeats, not categorical claims.

Each setup in either stage:

1. Releases old factors and creates scalar CSR from the same saved face blocks.
2. Performs real-nonsymmetric in-core analysis and LU factorization (mtype 11).
3. Solves the physical RHS twice, reusing the same factors for the second solve.

Fresh time is the median of **CSR conversion + analysis/factorization + first
solve**; reused time is the mean of the second solves. Conversion and
factorization are also recorded separately. Both calls must run PARDISO phase
33 after the explicit factorization. Solve times come from the package wrapper,
including PyPardiso's factor-reuse checks but excluding residual evaluation.
I/O, cache hashing, validation, memory snapshots, and factor cleanup are not
included. Input disk loading is not part of fresh time, matching the archived
linear-solver-only convention. The same physical RHS is used twice; this tests
factor reuse, not performance across a distribution of right-hand sides.

The worker verifies the archived matrix/RHS hash and an independently frozen
topology hash. Each solve must meet 1e-10 relative residual against **both CSR
and original face-block storage**. An additional, untimed planted random
solution test checks relative solution error <= 1e-6. Empty rows, invalid
sparse indices, nonfinite entries, factorization errors, and inaccurate solves
are recorded as failures. Success is evidence about this discrete linear system,
not a proof of nonsingularity, conditioning, or manufactured PDE accuracy.
The saved solution is the eliminated trace; this script does not reconstruct
the primal field or compute its L2 error.

`comparisons.json` / `comparisons.csv` pair every archived iterative configuration
with the identical matrix/RHS. Each source, configuration, status, and accepted
fresh/reused timing is retained; warmup-only failures are never assigned a
ranked time. Tuned ASM endpoints and completed pMG-AMG results are included.
Speedup is **iterative time / PyPardiso time** (>1 favors PyPardiso), reported
separately for fresh and reused measurements. These are new CPU measurements
against archived GPU measurements, not a simultaneous hardware-neutral ranking.
The recommended tuned campaign uses five confirmation setups, while archived
iterative protocols used two or three; sample counts and timing ranges remain
explicit. Tuning overhead and startup costs are saved in the worker wall times
but are not included in steady-state solver timings.

Interpret fresh and reused results separately. Do not combine setup from one
thread configuration with reused solves from another into an unmeasured policy.
A crossover depends on the number of right-hand sides per unchanged matrix,
accepted residual, memory budget, problem, and hardware. A win on these tests
does not establish a universal GPU or CPU advantage for all HDG/ADR problems.
CPU measurements remain comparisons against archived GPU runs, not new
concurrent measurements of the iterative methods.

## LU memory diagnostics and guards

`timings.csv` and each job's `result.json` record:

- Input face/CSR storage and PyPardiso's additional matrix copy/hash.
- Symbolic and numerical memory estimates, factor nonzeros and fill ratio.
- Actual process RSS and cumulative peak RSS at stage boundaries.
- External monitored RSS/high-water marks and minimum available host RAM.
- Perturbed pivots, refinement steps, phase, versions, and actual MKL threads.

See [the package memory-statistics definitions](../../reference/pardiso_diagnostics.md).
The default 64 GiB worker RSS ceiling and 16 GiB available-memory reserve are
checked every 0.2 seconds, with 30-second console heartbeats. These are
**best-effort monitoring guards, not OS-enforced allocation limits**; allocation
bursts can overshoot between checks. Timeout/RAM termination is retained as a
resource failure, never mistaken for a singular-system diagnosis. Out-of-core
mode is disabled; no automatic disk spill or tolerance relaxation is attempted.

Outputs remain under the new directory: provenance and coverage in
`manifest.json`, incremental `summary.json`, `timings.csv`, `comparisons.*`,
and per-attempt specs, logs, events, results, and accepted eliminated solutions
under `jobs/`. Raw timings and memory snapshots survive failed jobs.
In tuning mode, `thread_selection.json` freezes the pilot winners and source
attempts, `tuning.csv` keeps all pilot metrics, and `confirmed_timings.csv`
contains the independent confirmations. `comparisons.*` excludes pilots and
leaves speedup blank for objectives that did not select that confirmation.
`summary.json` distinguishes pilot/confirmation counts and the number of systems
with both confirmed baselines; the initial scheduled count is an upper bound.

Resume preserves pilots as well as confirmations. Retrying a failed pilot
changes its selection fingerprint, so old confirmations cannot silently serve
as confirmation of a new selection. Earlier attempts remain on disk.

Validation during implementation is limited to read-only archive discovery,
mocked orchestration, and tiny CPU matrix checks. Full-size timings and LU
memory findings await the user-launched campaign; neither report is updated yet.
