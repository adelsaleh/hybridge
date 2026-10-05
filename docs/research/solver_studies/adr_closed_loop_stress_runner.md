# ADR closed-loop stress runner

The runner implements the [September 21 proposal](adr_closed_loop_stress_proposal_2026_09_21.md).
Implementation tests cover analytic coefficients, normalization sampling,
mocked orchestration and logging. They do not generate stress meshes, run
solver comparisons or certify convergence of a user's numerical campaign.

## Plan and execute

From the repository root, inspect the default plan:

```bash
.venv/bin/python -B scripts/advection_diffusion_reaction/campaigns/stress/run_closed_loop_stress.py \
  --output run_outputs/solver_studies/adr_closed_loop_stress_main
```

Without `--execute` or `--prepare-only`, this command reads configuration/source
files and prints JSON. It creates no output directory, meshes or numerical jobs.
The default covers `trap`, `cross` and `orthogonal`, the main severity level,
p=6, and 50k/100k triangle targets: 60 solver comparisons plus separate profiles.

To execute that plan using the already installed ADR environment:

```bash
# from the repository root
HYBRIDGE_CUDA13_ROOT=/path/to/cuda-13 \
HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13 \
HYBRIDGE_AMGX_INSTALL_ROOT=../AMGX-install-cuda13 \
  scripts/gpu/run_cuda13.sh .venv/bin/python -B \
  scripts/advection_diffusion_reaction/campaigns/stress/run_closed_loop_stress.py \
  --output run_outputs/solver_studies/adr_closed_loop_stress_main --execute
```

These Python changes require no package/native rebuild. An actual numerical
execution can invoke the existing CUDA/Numba runtime JIT. No build or numerical
execution is part of the default planning command.

The assembler, ASM/BJ/AMGX workers and pure campaign helpers come from
`vendor/adr_gmres/` (`--branch-root` can select another compatible snapshot).
The native hp adapter imports the main package implementation. Companion changes
in the vendored `adv_diff_rea_cases.py`, `adr_solver_comparison_worker.py` and
`adr_native_hp_worker.py` supply process-local case registration, optional
residual histories, matrix identity checks and consistent L2-bound enforcement.
The run archives source snapshots and hashes.

## Square counterpart and direct checks

Use `--geometry square` for the smooth counterpart on `[-1,1]^2`. It uses
the existing package Gmsh rectangle helper with count matching within 3%,
one boundary component, and the same mesh shared across all three variants.
Annular geometry remains the default. There is no neck on the square:
omit `--require-neck-screen`, `--neck-width`, `--neck-elements` and
`--boundary-points`. The first two are rejected for square geometry;
the mesh-sizing-only controls do not affect a square mesh.

The [square coefficient definitions](../../reference/square_stress_coefficients.md)
are implemented in the package, not duplicated in the runner. The exact field
and trapping/crossing streamfunctions are smooth Cartesian counterparts.
The tensor is regularized at stagnation points and retains weak cross-contour
diffusion. This is not a geometry-only experiment or a claim of equal difficulty
to the annular cases. The orthogonal control retains its constant tensor and
transverse velocity. Case IDs start with `stress_square_`; annular IDs and
stored coefficient records are unchanged.

For future exploratory square runs, use the same 60 strong-preset comparisons
with six additional CPU direct checks (both mesh sizes):

```bash
# from the repository root
HYBRIDGE_CUDA13_ROOT=/path/to/cuda-13 \
HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13 \
HYBRIDGE_AMGX_INSTALL_ROOT=../AMGX-install-cuda13 \
  scripts/gpu/run_cuda13.sh .venv/bin/python -B \
  scripts/advection_diffusion_reaction/campaigns/stress/run_closed_loop_stress.py \
  --geometry square \
  --output run_outputs/solver_studies/adr_square_stress_strong_numba_h150k_pardiso_all \
  --solver-strength strong --maxiter 2000 --orders 6 \
  --triangles 100000 150000 --max-triangles 175000 \
  --assembly-backend numba --numba-threads 24 \
  --pardiso-all --pardiso-threads 24 --pardiso-max-dofs 2000000 \
  --pardiso-max-rss-gib 32 --pardiso-reserve-gib 8 \
  --skip-profiles --timeout 7200 --execute
```

Remove `--execute` to inspect a read-only plan. Use a **new output directory**;
changed sources cannot resume an annular campaign. No package/native rebuild
is required. Actual execution may invoke existing Numba/CUDA runtime compilation.

`--pardiso-all` selects every triangle target for every case and polynomial order.
It adds six checks to a three-case, two-mesh, one-order campaign; it does not
change the 60 iterative comparisons. Existing `--pardiso-coarse` commands keep
their coarse-only behavior. The two flags are mutually exclusive, and omitting
both disables direct checks. This option also works for the annular geometry.
Keep the DOF/RAM/time guards enabled for fine meshes; direct-factor memory and
runtime can grow substantially with refinement. A gated or failed check is
reported, not silently counted as a successful solve. Use a new output directory;
a running coarse-only campaign is not expanded by changing the runner source.

`--pardiso-coarse` selects the smallest requested triangle target for every
case and polynomial order (100k here, not a separate smaller diagnostic mesh).
After successful matrix preparation and before the iterative comparisons, it
runs the existing cached-system PyPardiso checker on that exact matrix/RHS.
This path has a separate DOF limit, so the ordinary 100,000-DOF reference gate
does not silently suppress it. The existing real-nonsymmetric package backend
checks residuals in CSR and original face storage, backward error, pivot
perturbations, and a planted-solution probe. Passing is numerical evidence
for that discrete system, not proof of nonsingularity or PDE accuracy.

The default direct thread count is the affinity-visible physical-core count
(24 on the current machine); `--pardiso-threads` overrides it. It is a starting
point, not a measured optimum. Direct checks run serially in isolated processes
with MKL dynamic threading disabled and single-threaded OpenBLAS. The existing
checker monitors worker RSS, available host memory, and `--timeout`.
RSS limits are sampled, not hard allocation guarantees. Failures/timeouts are
retained and do not prevent the iterative comparisons.

Results live in `pardiso_checks.json`, per-check `jobs/pardiso_*.json`, and
`pardiso/<case-point>/attempt_N/` (detailed results, events, worker log and a
saved eliminated trace on success). They do not overwrite cached reference
solutions or enter the 60-job GPU timing ranking. `completion.json` reports
direct-check counts separately; a failed direct check makes the campaign exit
nonzero even if iterative solves pass. Identical-source `--resume` reuses
terminal direct results, including failures, and preserves interrupted attempts.

## Controls

Append controls to either planning or execution commands:

- `--levels entry main severe`: the complete severity ladder.
- `--triangles 25000 50000 100000 --orders 2 4 6`: an h/p product; each level
  reuses exactly one mesh per triangle target for all variants and orders.
- `--levels main --epsilon 1e-4`: an anisotropy-only control. `--speed` and
  `--neck-width` are also available; overrides require a single base level.
- `--boundary-points 3600`: refine the geometric boundary separately.
- `--quadrature 32 --edge-quadrature 32`: a separate coefficient-quadrature
  control. Defaults are `2*p+4` in both directions. Use a new output directory
  and compare errors across runs; the runner does not certify quadrature convergence.
- `--candidates asm_pp bj_pp native_hp_standard native_hp_robust`: a subset;
  the printed plan lists all AMGX IDs as well.
- `--solver-strength strong`: PP96/restart150, block-AMG W cycles with 2+2
  sweeps, two direct-DILU inner iterations, native Chebyshev8 with 3+3 p-level
  sweeps and 3+3 coarse-AMG sweeps/W cycles. Baseline remains PP48/restart75.
  Individual `--pp-degree`, `--restart`, `--amg-sweeps`, `--amg-cycle`,
  `--amg-relaxation`, `--dilu-iterations`, `--dilu-relaxation`,
  `--native-chebyshev-order`, `--native-sweeps`, `--native-coarse-sweeps` and
  `--native-coarse-cycle` flags override the preset. Tolerances are unchanged.
- `--pp-degree 24`: change the frozen polynomial degree independently.
- `--maxiter 4000`: raise the shared outer-iteration cap above its default 2000
  (doubled from the first campaign's 1000). Applies to comparisons and profiles
  for every solver family; preconditioner cycle counts and tolerances are unchanged.
  The independent per-worker `--timeout` remains 1800 seconds by default.
- `--reference-max-dofs 100000`: CPU direct-reference ceiling, also triggering
  CPU/GPU assembly parity checks below that size; zero disables direct references.
- `--l2-bound VALUE`: additionally require a manufactured primal L2 error bound.
- `--skip-profiles`: omit separate instrumented applications/solves.
- `--heartbeat-seconds 30`: worker elapsed-time reporting interval (default 30).
- `--prepare-only`: evaluate normalization and generate meshes without assembly
  or solves. Later use `--execute --resume` with otherwise identical settings.

The default geometry is the nine-lobed annulus. A structured background size field caps
the local size by the radial gap divided by `--neck-elements` (default 8).
The shared Gmsh star helper supplies the central hole. Count matching varies
the bulk size while retaining neck refinement, accepts counts within 3%, and
never selects more than `--max-triangles` (default 100,000; larger explicit
budgets are supported). Trial meshes are retained, including
trials above that selection budget. If the target cannot be reached, preparation
fails with the trial record intact. The budget is not enforced by degrading cells.

`mesh.json` records shape quality, two closed boundary components, a sampled
polygon error and the minimum gap/element-diameter ratio near the necks.
It also samples normal velocity at five points per polygon edge for each
variant; summary rows report the maximum physical value and its ratio to U.
`neck_size_screen_passed` means that measured ratio is at least 6; it is a
resolution screen, not evidence of h/p convergence. All walls use manufactured
Dirichlet data on affine edges. Curved analytic trapping/cellular velocities
need not be exactly impermeable on those polygonal walls; orthogonal transport
intentionally has through-flow at both walls.

`--require-neck-screen` stops before assembly if the screen fails. Increasing
the global count alone does not ensure neck resolution; increase
`--neck-elements` too. This screen still does not establish field convergence.

## Stronger, better-resolved campaign

Use a **new output directory**; changed code/settings cannot resume the v2 run.
The following runs the 100k/150k ladder with a 175k selected-mesh ceiling,
stronger preconditioners, refined necks and boundary, and no duplicate profile
jobs. Solves still run on the GPU. CPU assembly avoids the monolithic GPU
assembly estimate of about 44.7 GiB at 150k/p6, above the default 80% budget
on the recorded 47.2 GiB device. Live host RAM, VRAM and disk guards remain on.

```bash
# from the repository root
HYBRIDGE_CUDA13_ROOT=/path/to/cuda-13 \
HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13 \
HYBRIDGE_AMGX_INSTALL_ROOT=../AMGX-install-cuda13 \
  scripts/gpu/run_cuda13.sh .venv/bin/python -B \
  scripts/advection_diffusion_reaction/campaigns/stress/run_closed_loop_stress.py \
  --output run_outputs/solver_studies/adr_closed_loop_stress_strong_numba_h150k \
  --solver-strength strong --maxiter 2000 \
  --triangles 100000 150000 --max-triangles 175000 \
  --neck-elements 12 --boundary-points 3600 --require-neck-screen \
  --assembly-backend numba --numba-threads 8 \
  --skip-profiles --timeout 7200 --execute
```

For a same-geometry solver-strength control, use a different output directory
and `--triangles 50000 --max-triangles 100000 --neck-elements 8
--boundary-points 1800`, omitting `--require-neck-screen`. The strong preset
changes several parameters together; it does not isolate their individual
effects and does not guarantee convergence.

### Parallel CPU assembly

`--assembly-backend numba` uses cached `parallel=True` element kernels for
unprojected tensor inverse-mass/advection integration, LAPACK local inverses
and static condensation. It reuses the existing coefficient callbacks,
quadrature, trace orientation, face accumulation and Dirichlet elimination;
those remaining stages are not all JIT-compiled. It is not the scalar-only
projected ADR Numba backend from the earlier assembler comparison.

The runner sets `NUMBA_NUM_THREADS=8` by default and BLAS/OpenMP environment
thread counts to one, preventing nested BLAS oversubscription. Numba's default
scheduler selects an available threading layer; `--numba-threading-layer`
can pin `tbb`, `omp` or `workqueue`. Eight threads is the best recorded starting
point from the earlier scalar CPU benchmark, **not a measured optimum for this
new tensor path**. Fastmath is disabled. No throughput claim is made before
compiled validation and timing on the stress case.

Tiny signature warmup/cache-loading runs before timed assembly, with its own
`numba_warmup_ms`; process wall time still includes it. Metadata records the
actual thread count, scheduler and Numba version. Detailed host timings now
separate coefficient sampling, diffusion geometry, inverse mass, advection/local
matrix, trace coupling, source and finite checks (substeps of host preparation;
do not sum them again with the host-preparation total). Below the direct-reference
ceiling, the new backend is independently compared to NumPy; larger production
matrices explicitly report that this check was not run.

Before the large campaign, the user can run this small assembly-only compiled
parity test (no global solve). This invokes Numba JIT on first use:

```bash
cd /path/to/hybridge/vendor/adr_gmres
NUMBA_DISABLE_JIT=0 NUMBA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_adv_diff_rea.py -k numba_tensor_adr_matches_numpy
```

Implementation diagnostics ran those six small-matrix cases with **all JIT
disabled**: p=1/3/6, both bases, variable anisotropy, nonpolynomial nonzero
Dirichlet data and reversed face orientations. Compiled parity, parallel
execution and performance remain to be checked by the user; no builds or
numerical campaigns were launched during these changes.

## Coefficients and solver protocol

The source uses analytic Cartesian gradients/Hessians and includes
`-(div K).grad(u)` for both curved-tensor cases. Independent finite-difference
checks cover tensor derivatives, conservative flux divergence, all three
severity levels, neck locations and the polar branch cut. The orthogonal
variant keeps its constant tensor and exactly perpendicular transport.

For each curved case, `C_psi` is estimated on nested polar grids independent of
the solver mesh, starting at 256 angles and 64 radial intervals. Two successive
relative peak changes must be below `--normalization-rtol` (default 1e-3), with
at most `--normalization-refinements` doublings (default 5). Failure to converge
stops preparation. The full sampling history and one frozen constant are
stored in `normalizations/` and included in every worker specification. This
is empirical convergence, not a certified upper bound on the continuum peak.
Orthogonal transport uses the analytic constant 1.

The baseline frozen candidate set contains fused ASM+PP and raw BJ+PP with CGS2/restart
75, block-graph AMG with block Jacobi and multicolor DILU smoothing under both
FGMRES and PBICGSTAB, direct BSR DILU under both outer methods, and standard/
robust native hp-BSR. These reuse existing optimized application paths/presets;
their relative performance on the new stress problems is unmeasured.

Each case/mesh/order is assembled once. All candidates receive the same saved
Bernstein matrix/RHS, FP64, explicit zero guesses, a physical relative residual
threshold of 1e-10, internal target 1e-11 and default iteration cap 2000
(`--maxiter` overrides it). One excluded
warmup setup and three measured fresh setups each perform two zero-guess
solves. Setup, fresh setup+solve and reused solve times remain separate.

Residual histories are saved after timing. Native histories report true norms
at restart boundaries plus estimated preconditioned norms; AMGX histories are
its reported absolute L2 norms, with the final physical residual independently
checked. CPU reference-trace differences and manufactured primal L2 errors
are recorded where applicable. Algebraic `passed` status does not establish
field accuracy without h/p/quadrature checks (or the supplied L2 bound).

Profiles run in separate processes and are excluded from timing comparisons.
They are attempted even for nonconvergent candidates; profiling failures are
retained. Native hp profiles use the existing full-preconditioner/GMRES
instrumentation, ASM/BJ use the campaign component profiler, and AMGX uses its
existing native event-timing adapter.

## Artifacts and resume

`manifest.json` freezes parameters, solver configurations and source hashes;
`source_snapshot/` retains both implementations. `specs/`, `jobs/` and `logs/`
retain every worker request and outcome. `assemblies.json`, `summary.json`,
`profiles.json` and `completion.json` distinguish numerical failures, assembly
failures, process errors and timeouts. Per-job samples contain timing, residual
histories and field errors. Failed cases remain in summaries and the process
returns nonzero when any comparison/profile fails.

`--resume` requires identical parameters and source hashes. Completed successes
and terminal failures are reused, interrupted worker attempts are archived,
and changed mesh files are rejected. A new output directory is required to
retry terminal failures with changed settings. Existing campaigns are not
overwritten by a fresh execution.

## Progress, timings and debugging

Inspect an existing campaign, including one still running:

```bash
.venv/bin/python -B scripts/advection_diffusion_reaction/campaigns/stress/run_closed_loop_stress.py \
  --output run_outputs/solver_studies/adr_closed_loop_stress_main --status
```

`--status` is read-only: no GPU initialization, numerical worker, mesh generation,
source-hash comparison, or output writes. It works on older campaign artifacts
and does not require the companion checkout. It lists saved jobs, not all
future unscheduled specifications; a saved `running` state is not proof that
the corresponding process is still alive. It also reads AMGX instrumented
sidecars when the profiling summary itself lacks solve details.

New executions print UTC-stamped phase starts/completions and elapsed time for
normalization, mesh preparation and each worker. While a worker runs, the
heartbeat reports its wall time and latest **completed** solve, if recorded.
It does not sample the GPU or provide live iteration counts within a solve.
Workers use unbuffered output in `logs/`; the log header records the exact
command, working directory and specification path.

At job completion, the terminal and per-job log show:

- Status, process exit code, total worker wall time and log location.
- Whether the last solve belongs to warmup or measured samples, its iteration
  count/cap, physical residual and target, internal target, and field errors.
- Last setup, solve and fresh setup+solve times, measured setup median and
  reused-solve mean, plus setup/assembly stage timings where the worker supplies them.
- Residual-history type, initial/best/final values and tail; native restart
  norms and AMGX reported norms are explicitly distinguished from the final
  independently checked physical relative residual.
- Failure explanations, such as reaching the iteration cap or exceeding the
  physical-residual target. A profiler's `profiling warmup did not converge`
  exception is identified separately from the comparison failure.
- Actual trace DOFs and whether CPU reference and CPU/GPU assembly checks ran.
  In particular, assembly `passed` **does not mean** a CPU reference was checked
  when the matrix exceeds `--reference-max-dofs`.

Wall time includes process startup, imports, cache reads, verification and
cleanup; it is not a substitute for synchronized solver-only timings.
Failed excluded warmups retain their timings but never become measured
benchmark repetitions. Profile timings remain separate from comparisons.
Neck-resolution screen failures now produce an explicit warning.

`events.jsonl` appends structured UTC-stamped preparation, job-start, heartbeat,
job-finish, reuse and campaign-progress records. Job results also retain
`runner_wall_seconds` and `runner_started_utc`. These logging additions do not
change solver tolerances, iteration limits, coefficients or profile policy.
An already running parent process keeps its original logging code; use
`--status` to inspect that run. Source hashes still protect resume, so a
campaign created before these code changes requires a new output directory
for execution with the updated runner.

The no-build, no-simulation diagnostic command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_closed_loop_stress.py
```

End-to-end GPU convergence and performance must be assessed from the user's
numerical runs, separately from these no-build diagnostics.

## Direct solvability diagnostic on a cached coarse system

`check_cached_adr_pardiso.py` reuses the existing
`hybridge.linalg.system.solve_pypardiso_system` backend (real nonsymmetric mode,
not SPD mode). It reads only the saved face blocks, neighbors, and eliminated
RHS: no reassembly, JIT compilation, GPU work, or writes to the campaign cache.
It pre-factorizes the existing backend's own PyPardiso instance to report
analysis/factorization separately from solve time, then reuses those factors
through the existing HYBRIDGE solve/residual-validation interface.

For the original 50k-target trapping case (49,645 triangles, 511,574 trace DOFs),
run this **after the active campaign finishes**, so CPU timing is uncontended:

```bash
# from the repository root
.venv/bin/python -B scripts/advection_diffusion_reaction/diagnostics/check_cached_adr_pardiso.py \
  --spec run_outputs/solver_studies/adr_closed_loop_stress_main_v2/specs/assemble_stress_main_trap_t50000_p6.json \
  --output run_outputs/solver_studies/adr_pardiso_coarse_50k \
  --threads 24 --execute
```

Omit `--execute` for a read-only plan. Every execution requires a new output
directory. For the newer campaign's 100k-target coarse case, point `--spec` at
`adr_closed_loop_stress_strong_numba_h150k/specs/assemble_stress_main_trap_t100000_p6.json`
under the same studies directory, use a different output, and explicitly raise
the safety gate with `--max-dofs 1200000`.

The default thread count is the affinity-visible physical-core count (24 on this
machine), **not a measured optimum**. To select the best measured count, replace
`--threads 24` by `--threads 8 16 24 --repeats 3`, with a fresh output directory.
Each repetition uses a new process and fresh factors; `summary.json` ranks
median setup-plus-solve time only for counts whose repetitions all passed.
The worker sets MKL/OMP threads explicitly, disables MKL dynamic adjustment,
keeps OpenBLAS single-threaded, and checks MKL's reported maximum thread count.

Results include the physical CSR residual, a second residual computed directly
from the original face blocks, normwise backward error, perturbed-pivot count,
and a planted random-solution recovery check using the same factors. Success
requires both residual checks at `1e-10` (configurable with `--rtol`) and planted
solution relative error at most `1e-6`. A small residual is evidence that this
discrete RHS is solvable; even the additional probe is **not a proof of
nonsingularity**, nor a manufactured-solution accuracy or mesh-convergence test.
Only successful solutions are saved as `eliminated_solution.npy` in the
diagnostic directory; they do not silently become campaign reference solutions.

The parent logs a heartbeat every 30 seconds and polls resource use every 0.5 s.
Defaults terminate the worker above 32 GiB RSS, below 8 GiB available host RAM,
or after 1800 s (`--max-rss-gib`, `--reserve-gib`, `--timeout`). These are polling
guards, not hard allocation limits; PARDISO fill-in is not known in advance.
Reported PARDISO memory counters use PyPardiso's one-based indices 15--17;
see [Intel's parameter reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2026-0/pardiso-iparm-parameter.html).

Small-matrix backend and wrapper checks, with JIT disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 MKL_NUM_THREADS=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_cached_adr_pardiso.py tests/test_pypardiso_backend.py
```
