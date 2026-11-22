# cpu_adr_scaling: host ADR pipeline thread scaling on the device campaign's systems

## Purpose

How far the CPU (genoa: 2 × AMD EPYC 9655, 192 cores, 8 NUMA domains of 24 cores) carries
the stationary ADR pipeline, stage by stage, on the very systems of the unified device ADR
campaign (`adr_unified`): same plan ids, cases (including the closed-loop stress
cases whose iterative solves failed there: `stress_main_trap`, `stress_main_cross`), meshes,
degrees and quadrature.

Each point is one system at one thread count, run by an isolated worker (`point.py`) on
hybridge's host pipeline only: Numba assembly of the global trace system, PyPardiso solve,
Numba reconstruction, Numba total-flux and primal postprocessing (`hdg_postprocess='both'`),
with one cold solve then `--repeats` warm solves, each with a fresh solver object. Every
stage and substep the solver times is recorded, with process CPU time, RSS and L2 errors.
Case coefficients are compiled, multithreaded `PointwiseLaw`s: stress cases via
`hybridge/cases/closed_loop_pointwise.py`, the sine, cellular and quadratic catalogue cases via
`hybridge/cases/manufactured_pointwise.py` (`catalogue.compiled_case`); the NumPy closures sample
on one core. Each result records `coefficients` (`pointwise_law` or `numpy`).
The vendored `vendor/adr_gmres` stack is not used: it serves only the GPU campaign's
face-dense GMRES solvers.

Tiers (`--tiers`): `compile` (JIT compile and cache-load cost per degree on the smallest
square), `scaling` (full degree range of the representative and stress classes, threads
from an estimated lower bound up to the whole node), `coverage` (every system the device
campaign assembled plus the stress classes, whole node, largest first). Before any timed
run, an untimed `populate` run per (case, degree) on a tiny cut-out of the system's
mesh compiles every Numba kernel into the shared cache. Every run records the Numba cache
files it added (`numba_cache_new_files`); a timed run that compiles anything logs a WARNING.

Thread settings per worker (pinned to cores `0..t-1`): `NUMBA_NUM_THREADS`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS` = t; `OPENBLAS_NUM_THREADS` = min(t, 64) (NumPy's
OpenBLAS build caps at 64); scipy's OpenBLAS limited to 1 in the worker (threadpoolctl).

## Results

A genoa smoke passed 18/18: stress L4_p9 warm 102.3 s at 16 threads, 17.4 s at 192 (252 s /
173 s before the Numba ports). The single-core work constants are fitted from its 16/192
pairs (T = S + P/t): `--work 1.56e-7 6.3e-8`. The remaining serial copies in `PointwiseLaw`
evaluation were then fixed and the catalogue cases' coefficients compiled.

Scaling tier: 488 runs passed; coverage tier: 193 timed runs passed (2026-10-28/29). The text
outputs (`summary.json`, `timings.csv`) are in `run_outputs/solver_studies/cpu_adr_scaling/` and
the records in `docs/campaigns/cpu_adr_scaling/outputs/`.

**Caveat on the global solve of these runs (found 2026-10-30).** Their warm solves reused the
cold solve's factorization: `solve_pypardiso_system` goes through PyPardiso's process-wide
solver, which skips analysis and factorization when the matrix equals the last one factored, and
each warm solve re-assembled the identical system in the same process. `warm_solve_s` is
therefore CSR assembly, PyPardiso's matrix comparison and a phase-33 solve, not a fresh LU; the
warm pipeline totals include that reuse too. `cold_solve_s` is the one solve that factorizes
(with first-call overhead at small sizes). Fresh and reused LU times per thread count are in the
PARDISO v2 campaign (`docs/campaigns/adr_pardiso/records/unified_l5_pardiso_v2`); its thread tuning shows fresh LU gaining 1.0–1.2x from 16 to 96 threads.
The worker now calls `clear_pypardiso_cache()` before every solve, so a rerun measures a fresh
LU in every warm solve. Numba assembly, reconstruction and postprocessing are not affected.

## Commands

Frozen copy: cut from the committed revision that contains this campaign (record it here and
in the copy's `COMMIT`); bytecode precompiled under
`/scratch/user/hybridge/pycache/<copy>` (`python -m compileall` over the stdlib, the venv
and the copy with `PYTHONPYCACHEPREFIX` set).

Plan only (needs the device campaign's exported systems under `--device-campaign`; regenerate them first if they are missing):

    .venv/bin/python -B campaigns/cpu_adr_scaling/run.py --device-campaign /scratch/user/hybridge/adr_unified_l5 \
        --output /scratch/user/hybridge/cpu_adr_scaling --tiers compile scaling coverage --plan

Start (one whole genoa node per job; scaling and coverage can run as two jobs):

    cd campaigns/cpu_adr_scaling
    HYBRIDGE_REPO=<copy> CPU_ADR_OUT=/scratch/user/hybridge/cpu_adr_scaling \
    CPU_ADR_RECORDS=run_outputs/solver_studies/cpu_adr_scaling \
    CPU_ADR_ARGS="--tiers compile scaling" \
    sbatch --job-name=cpu-adr-scaling --output="$CPU_ADR_RECORDS/slurm/%x-%j.out" --export=ALL genoa_cpu_adr.sbatch

Resume: submit the same command again; finished runs (`runs/<run>/result.json` with a final
status) are kept. The Slurm time-limit warning makes the runner save and exit 2, and the
wrapper requeues itself. Stop: `scancel <job>`; rerun to resume.

Output directory: `$CPU_ADR_OUT` (`plan.json`, `provenance.jsonl`, `runs/`, `summary.json`,
`timings.csv`, `completion.json` when every run of the plan has a final status). Records are
rsynced to `$CPU_ADR_RECORDS` every 30 min and at exit.
