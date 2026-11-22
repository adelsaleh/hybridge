# H200 tuned ADR campaign (`adr_h200_tuned`, started 2026-10-28)

Where do iterative solvers start to beat PyPardiso LU on the H200, and did the H200 kernel work
move the Blackwell crossover? The ladder pushes AMG (AMGX classical AMG, native pMG-AMG) and
ASM+PP (with BJ+PP as the cheap-setup reference) from 2k triangles to 80M trace DOFs, with the
tuned kernels of the merged H200 branches, and compares with PyPardiso on the same systems.

| | |
|---|---|
| Branch | `adr-h200-tuned`: merges `h200-solver-kernels`, `cupy-adr-device`, `h200-asm-tuning`, `ar-batched-lu` |
| Plan | `adr_h200_tuned_plan.py` (pure); launcher `run_adr_h200_tuned_campaign.py` (unified runner's `execute`) |
| Lane stage | `lane_stage.sh OUTPUT [options]`: calibration (device and host fits), plan, preflight, `--execute --resume` |
| Outcomes | Tier 1 (L1–L5, `50-l1-l5`) 357/357 jobs; tier 2 (L6–L7, `51-l6-l7`) 150/150; tier 3 (L8–L9, `70-l8-l9`) all systems memory-excluded on one H200 (raw-CUDA assembly rows ran at L8 p=2/6). Lmax degree sweep (p = 2–9 at the largest size per degree; `82-lmax-p2-p5`, `83-lmax-p3-p7`, `84-lmax-p4-p8-p9`): 133 solver jobs on 42 systems: 57 passed, 39 stopped at the FP64 residual floor, 37 not passed; no LU at these sizes (`lu_followup_systems.csv`). L4 degree fill (p = 3, 4, 5, 7, 8 at L4): 125 solver jobs on 30 systems: 80 passed, 4 at the FP64 floor, 11 not passed. See "Lmax" and "L4 degree fill" below. |
| Output | `/scratch/user/hybridge/adr_h200_tuned/<step>`; small run records in `run_outputs/campaigns/adr_h200_tuned/<step>` |
| Records | `docs/docs/campaigns/adr_h200_tuned/records/` (`export_records.py`: per-job medians, systems, stock-kernel rows of the unified campaign); comparison with the Blackwell study in `docs/research/solver_studies/adr_scaling_2026_09_17/h200_tuned/` |
| Tests | `tests/test_adr_h200_tuned_plan.py` |

## Reference points

- **Blackwell (RTX PRO 5000 vs PyPardiso, Sept 2026, `adr_scaling_2026_09_17/synthesis/overview_cells.csv`).**
  - Fresh solves (setup + first solve): iterative first wins at 21k trace DOFs (BJ+PP on transport squares).
  - Reused solves: LU wins below about 80k DOFs, and the crossover is at 82-85k DOFs (p=6, L2).
  - At 1.04M DOFs AMG / pMG-AMG win by 3.5-5.7x.
  - Except cellular7_weak: only ASM+PP converges there, and LU still wins (4.2 s fresh / 0.69 s reused vs 13.8 / 12.4 s).
  - Largest ADR PARDISO system measured: 1.55M DOFs.
- **H200 unified campaign (`adr_unified` device runs):** repeated solves only 1.0-1.9x the
  Blackwell before the kernel work.
- **H200 kernel work:** ASM+PP up to 9x the Blackwell at 98k p=6 (`campaigns/solver_h200/README.md`).

## Systems

Six classes, p = 2, 6, 9:
- square trigonometric, advection_dominated, variable_velocity and anisotropic;
- annulus cellular7_weak and stress_main_orthogonal.

| Level | Size | Shared with the unified device and PyPardiso runs |
|---|---|---|
| L1-L4, L4b | archived meshes (2k-100k triangles) | yes (anisotropic: no, not in the unified h-p grid) |
| L5 | 400k triangles | yes |
| L6 / L7 | 8M / 20M trace DOFs | yes |
| L8 / L9 | 40M / 80M trace DOFs | new, beyond any LU |

There are 156 systems and 657 jobs. 105 systems keep their unified plan ids, so PARDISO records join on `key`.

Two classes are left out, because no candidate converges on them in the unified campaign or on the Blackwell:
- stress_main_trap;
- stress_main_cross.

## Candidates (fastest converging per class in the unified campaign, then tuned)

| Class | ASM+PP | BJ+PP | AMGX AMG | pMG-AMG |
|---|---|---|---|---|
| trigonometric | asm_d24 | bj d12 | PBiCGStab+AMG | standard |
| advection_dominated | asm_d24 | bj d12 | PBiCGStab+AMG | standard |
| variable_velocity | asm d18 | bj d18 | PBiCGStab+AMG | standard |
| anisotropic | asm_d48 | — | PBiCGStab+AMG | standard |
| cellular7_weak | asm_d24 | — | PBiCGStab+AMG | — |
| stress_main_orthogonal | asm d96 | bj d96 | — | — |

The per-candidate evidence is in `CLASS_CANDIDATES`. Exclusions and their reasons are in `EXCLUDED`:
- AMGX smoother-only candidates: 0 converged on the stress cases;
- AMGX AMG on cellular7_directional: 0/28;
- AMGX DILU and FGMRES+AMG on cellular7_weak;
- native_hp on cellular7_weak: 0 converged from 22k triangles up;
- native_hp on stress_main_orthogonal: about 90x slower than ASM;
- polynomial-only PP.

Tuned options (`TUNED`; slv-11..25):
- **ASM+PP:** interleaved operator, `interleaved_f32` ASM application, `estimate_target='true_target'`.
- **BJ+PP:** interleaved operator and `interleaved` FP64 BJ application, with true_target. FP32 raised the BJ warm iterations.
- **pMG-AMG:** `graph` kernel preset, interleaved outer operator, true_target.
- **AMGX:** the production build (`AMGX-build-cuda13.1`, as in the unified campaign). The ladder's block sizes (3, 7, 10) need none of the DILU 6/9 additions, and the two builds are bitwise equal there.

Each tuned candidate keeps `base_candidate` (the unified campaign's id) for joins.

The tuned ASM/BJ path (`tuned_face_dense.py`) is the vendored `measure` with the hybridge-stack builder. It keeps the same timings, warm series and PDE-error checks.

## Assembly

Systems are assembled by the vendored assembler's device route (`cupy-adr-device`). Settings:
- `HDGFEM_CUPY_ADR_WORKSPACE_GIB=4`;
- 8 prefault threads;
- pageable outputs.

That route is the only one whose cache the solvers and PyPardiso can use.

The raw-CUDA ADR tensor kernel of `h200-asm-tuning` is 10-30x faster, but it builds a different discretization:
- Lagrange or Legendre traces instead of Bernstein;
- a different quadrature and stabilization;
- no `element_blocks` and no `local_inverse`.

So every system gets a `raw_assembly` row instead (`raw_assembly_timing.py`). It times device coefficient sampling, the tensor kernel and reconstruction on the same mesh, degree and case. The sm_90 settings per degree are:
- block 32 at p<=4 (`auto` picks 64 at p=3-4, slower);
- `auto` elsewhere: 128 at p=5-7, 256 at p=8, and 512 + non-compact + maxrregcount 128 at p=9 for variable kinds;
- explicit kind-6 mass inverse at p>=7;
- BSR with `schur-lu+mass` factor caching.

It records `raw.kernel.num_regs` and `local_size_bytes`, against the open spill list in `docs/algorithms/assembly/h200_sm90/README.md` (ADR tensor p=2 spills, 64 B local at p=7-9).

## Device-only execution and timing diagnostics (v2, 2026-10-28)

The first run spent 36-87% of each job's
wall time on the host, outside the timed solves. Examples at L6 (520k triangles, p=9):
- 83 s per solve verifying the solution with NumPy;
- 65-115 s sampling the stress coefficients, which rejected CuPy arrays.

v2 keeps every tensor operation on the GPU.

| Work | Before | v2 |
|---|---|---|
| Solution checks (residual, reconstruction, L2) | NumPy, DG space rebuilt per job | `device_verify`: u = w + M t with w, M formed once per job; geometry saved at assembly |
| Warm series RHS, guesses, residuals | host | device |
| AMGX BSR | SciPy on host, timed in setup | CuPy, rows sorted on device, device-pointer upload (bitwise the SciPy BSR) |
| hp-MG symmetric part, sort, modal products, diagonal search | SciPy + Python loop, timed in setup | `BernsteinHpSymmetricPartPreconditioner.from_face_dense` (application identical in the gate) |
| Closed-loop stress coefficients | NumPy only, host fallback in both assembly paths | array module of the inputs (NumPy bitwise unchanged, CuPy = NumPy to 3.5e-13) |
| Cache identity per job | SHA-256 of blocks and RHS, twice | size and mtime against the assembly record (SHA-256 kept from assembly) |

Every record carries:
- `phases` (wall seconds per phase, device-synchronized) and `worker_phases`;
- `gpu_utilization`: an NVML timeline at 1 s of SM and memory-controller busy %, with the mean, the median, the 10th percentile and the share below 10%.

The stage logs print one `phases (...)` and one `gpu: SM busy ...` line per job. Raw-assembly rows warn on any host coefficient sampling (`host_fallbacks`).

Still on the host, and none of it tensor work:
- the per-job interpreter and CUDA start-up;
- mesh generation (gmsh);
- one DG-space build and the cache write per system.

Stages:
- `tuned-20-gate` (gpu_gate.py and the hp-MG and interleaved suites), `tuned-22-smoke-l1`, then `tuned-30-l1-l5`;
- `tuned-21-smoke-stress`, then `tuned-31-l6-l9`.

The tiers run only when the gate and the smokes exit 0.

## No device-host-device round trip (v3, 2026-10-28)

v2 still passed every system through the host: the vendored assembler copied the local
inverses, boundary matrices, element blocks and trace system back, wrote them to the cache on
scratch, and each solver job (one process per candidate) read them back and uploaded them
(the 20.9 s verifier setup at stress L4 p=9 was that upload). v3 keeps the assembled system on
the GPU:

- `assemble_adr_device(device_outputs=True)`: a new mode of the vendored route. Nothing is
  copied back; the chunk hook forms the verification data `w = L_u s`, `M = L_u B` from each
  chunk's local inverse on the device before the inverse is freed (`device_system`).
- One worker per system (`system_worker`): assembly, then every candidate in the same process
  from the device arrays. The operator layout, block-Jacobi, element ASM (local matrices and
  incidence slots vectorized on the device), the hp-MG hierarchy input and the AMGX BSR are all
  built from device memory. Each candidate keeps its own SIGALRM budget (`--timeout`), NVML
  window, record file (`jobs/<system>_<candidate>/result.json`) and resume.
- The cache receives `system_*.npy` once, device to host to disk, for the host PyPardiso
  reference (`host_pardiso.py` reads `jobs/assemble_<id>/result.json` and those files as
  before); local inverses and element blocks are never written.
- Candidates run ASM first; the element blocks are released after the last ASM setup.

The scratch file system stalls some write-backs for a fixed ~307 s, size-independent (seen
with memory maps and plain writes alike). The export therefore runs in a background thread
beside the candidate solves and is joined after the last candidate; its per-file copy, write
and close times are logged, and the log line `[system] export finished ... waited M s` gives
the only GPU-idle cost that remains from it.

Host work left per system, none of it tensor work: the interpreter and CUDA start-up of one
worker (not one per candidate any more), mesh generation, one DG-space build (reference
quadrature tables and mesh topology), and the one-way export.

Gates: `tests/test_adr_h200_tuned_device_system.py` (NumPy array module: device-output route
== host-output route bitwise, device-built layouts == host builders, export hash, verifier
parity) and `gpu_gate_device_system.py` (the same on the GPU against v2 caches, plus the
preconditioner applications).

Audit of the first 43 tier-1 records (`50-l1-l5`, systems through L3): no
host callable fallbacks in any assembly; every solver candidate ran in-process on the device
with `exact_on_device`, `verifier_M_resident` and `device_verification` true; the background
PyPardiso export never delayed a solve (`export_join_wait` 0.0 s in every system, while the
export itself took up to 1229 s inside a scratch write-back stall). The remaining copy per AMGX
candidate is the device-to-device face-BSR upload, removed by point 2 below.

## AMGX zero-copy CSR (point 3, 2026-10-28)

The CUDA-13 fork's `AMGX_matrix_attach_csr` / pyamgx `Matrix.attach_CSR` lets AMGX alias the
caller's device CSR instead of copying it. Nothing in hybridge used it. Now
`PyAMGXCsrDeviceSolver(borrow_csr=True)` borrows a matrix that meets the fork's contract
(single GPU, scalar, int32, canonical, an explicit diagonal in every row; checked on the device)
and uploads anything else, recording why in `borrow_note`. It is on for the hp-MG scalar coarse
cycle (`AmgxScalarVcycle`, this campaign's native_hp candidates) and opt-in for one-shot
raw-CUDA CSR solves (`amgx_borrow_csr`). GPU record: `59-gputests-point3` / `61-gputests-point3`
(tests `test_amgx_borrowed_csr.py`, `test_amgx_fixed_cycles.py`, `test_hp_mg_packed_kernels.py`
with the production pyamgx, which has `attach_CSR` but no `attach_BSR`).

## AMGX zero-copy BSR (point 2, 2026-10-28)

The fork gained `AMGX_matrix_attach_bsr` (square blocks, same row contract as the CSR attach;
`include/borrowed_matrix_api.h`, native suite `src/tests/borrowed_bsr.cu`) and pyamgx
`Matrix.attach_BSR` (resolved with `dlsym`, so a pyamgx built against the extended library
reports `has_attach_bsr()`). `PyAMGXCsrDeviceSolver(borrow_csr=True)` borrows a device BSR that
meets the contract (`_attachable_bsr_view`) and otherwise uploads with the reason; a borrowed
`replace_coefficients` writes the new values into the aliased buffer and repeats the setup.
`_validate_amgx_block_configuration` now refuses plain `JACOBI_L1` on block rows (the fork's
message was the raw "Unsupported block size for the algorithm").

What the attach adds over the device-to-device `upload` the guiding-center studies always used:
no second copy of the fine-level matrix in GPU memory (about 8 GB at L7 p=9, 32 GB at L9 p=9,
on top of the AMG hierarchy) and in-place coefficient updates. Not time: the upload costs 0.3 to
3 ms against hierarchy setups of 100 to 900 ms in the tier-1/2 records, so the v3 timings stand.
The campaign now reaches pyamgx only through `PyAMGXCsrDeviceSolver` (CSR or BSR,
attached or uploaded); `solve_ms` of AMGX rows then includes the RHS/solution hand-off by device
pointer, which the record notes.

Qualification (`60-point2-qualify`, `campaigns/gpu_lane/stages/point2_qualify.sh`): native suites
BorrowedBSR, BorrowedCSR, BorrowedVectors (9 runs, 0 failed; mixed-precision modes are refused by
the fork and counted as not failed) with `../AMGX-build-cuda13.1-bsr`, then the
hybridge GPU tests (28 passed) with `../hybridge-venv-bsr`. The campaign's AMGX
candidates borrow the face-BSR when the plan sets `amgx_borrow_bsr` (`TUNED['amgx']`) and the
copy runs with that build and venv; a copy with the production stack uploads device-to-device
and records `bsr_borrowed: false`.

The two test stages and the qualification shared the GPU with tier 1 for a few minutes. Tier-1
candidates measured in that time (L3 p=9, small systems) shared the GPU for under a minute each;
their records' GPU windows show it.

## Lmax: the largest size per degree, p = 2–9 (2026-10-30)

The ladder ran p = 2, 6, 9 only. This tier fills p = 3, 4, 5, 7, 8 and moves p = 2 and p = 9 to
their own limits, each at the largest mesh one H200 takes. `build_plan(largest=True)`
(`--largest`) adds the level `Lmax`. For each degree its size is the largest multiple of 250k
trace DOFs whose system job passes the worker's device check on every class. That check is
`system_device_need`: the bytes retained during ASM plus the largest candidate's planning solver
bytes, against 0.97 × the 125.36 GiB budget the worker measured in `51-l6-l7`. The function
reproduces the logged L7 p=9 exclusion (167.74 GiB). cellular7_weak (AMGX on the annulus) binds
at p ≥ 3 and the stress class (restart-96 basis) at p = 2.

| p | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| Lmax trace DOFs | 34.0M | 30.0M | 25.5M | 22.0M | 19.25M | 17.25M | 15.5M | 14.0M |
| square triangles | 7,558,272 | 5,005,448 | 3,406,050 | 2,446,472 | — | 1,441,602 | 1,152,162 | 935,712 |

p = 6 is not rerun: L7 (20M) already ran there and fits only because the worker's budget has no
margin. Candidates, protocol and acceptance are the ladder's (3 repeats after one warm-up,
maxiter 2000, true relative residual 1e-10), so Lmax extends L1–L7 like for like. The PP
degrees are the ladder's per-class choices; they were never swept with the tuned kernels.

The frozen copy's `.venv` is a copy of `hybridge-venv-bsr` with the editable finder repointed at
the copy. The stages set `HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13.1-bsr`, which also has
MULTICOLOR_DILU blocks 6 and 9 (p = 5 and 8; compiled from the fork working tree).

| `--mem` | Smoke (repeats 1): L1 and stress L4 | Lmax output | Estimate |
|---|---|---|---|
| 256G | p = 5: `80-smoke-l1-p5`, `80-smoke-stress-l4-p5` | `82-lmax-p2-p5` | ~15 h |
| 320G | p = 3, 7: `80-smoke-l1-p3-p7`, `80-smoke-stress-l4-p3-p7` | `83-lmax-p3-p7` | ~13 h |
| 384G | p = 4, 8: `80-smoke-l1-p4-p8`, `80-smoke-stress-l4-p4-p8` | `84-lmax-p4-p8-p9` | ~17 h |

The estimates scale the L7 tier's wall time (about 1 h per system at 20M DOFs) to the Lmax
sizes. The host bounds `--mem`: the calibrated host fit at p = 9 is 273 KB per triangle, about
295 GB with the runner's 15% margin at 935k triangles. Each Lmax stage runs only if
`outcome_gate.py` passes on its two smokes: both completed, and no record ended outside
passed, not_converged or numerical_failure. The annulus meshes are prefetched on CPU nodes
(`mesh_prefetch.sbatch`; `--retry-mesh-errors` retries failed meshes). The p = 9 cellular annulus
(target 933,334) once failed: the five-lobed triangle count rises 7.9% between sizes 0.002652 and
0.002633, so the size**-2 step jumped over the 3% band and cycled 894k/1047k/867k.
`match_triangle_target` now bisects in log size once a step leaves the bracket of too-coarse and
too-fine sizes; it lands at 923k triangles. The exports are PyPardiso's input: its 32-bit MKL
interface limits the matrix's indices (fewer than 2^31 entries), not the factor, so L6 and L7
factorize (in the LU follow-up).

```bash
COPY=../<frozen copy of a committed revision that contains this campaign>
OUT=/scratch/user/hybridge/adr_h200_tuned
# annulus meshes first, on CPU nodes (one job per Lmax output)
sbatch --job-name=adr-tuned-meshes-lmax-a --output=$OUT/82-lmax-p2-p5/mesh-prefetch-%j.out \
    $COPY/campaigns/adr_h200_tuned/mesh_prefetch.sbatch $COPY $OUT/82-lmax-p2-p5 --levels Lmax --largest --degrees 2 5 --timeout 21600
# stages of one GPU lane (<lane-dir>/stages); the other degree groups likewise
tuned-80-smoke-p5 bash -c 'export HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13.1-bsr HYBRIDGE_TUNED_CALIBRATION_SIZES="64 128" HYBRIDGE_TUNED_CALIBRATION_DEGREES=5; S=$COPY/campaigns/adr_h200_tuned/lane_stage.sh; $S $OUT/80-smoke-l1-p5 --levels L1 --degrees 5 --repeats 1 --timeout 3600; exec $S $OUT/80-smoke-stress-l4-p5 --levels L4 --classes annulus:stress_main_orthogonal --degrees 5 --repeats 1 --timeout 3600'
tuned-82-lmax-p2-p5 bash -c 'cd $COPY && .venv/bin/python -B campaigns/adr_h200_tuned/outcome_gate.py $OUT/80-smoke-l1-p5 $OUT/80-smoke-stress-l4-p5 || exit 1; exec env HYBRIDGE_TUNED_JOB_TIMEOUT=21600 HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13.1-bsr HYBRIDGE_TUNED_CALIBRATION_DEGREES="2 5" $COPY/campaigns/adr_h200_tuned/lane_stage.sh $OUT/82-lmax-p2-p5 --levels Lmax --largest --degrees 2 5'
# lane: sbatch --job-name=gpu-lane-<lane> --mem=256G --output=<lane-dir>/slurm-%j.out \
#     --export=ALL,LANE=<lane-dir>,LANE_SBATCH_OPTS=--mem=256G campaigns/gpu_lane/lane.sbatch
```

## L4 degree fill, p = 3, 4, 5, 7, 8 (2026-10-31)

The L4 degree sweep (synthesis Figures 7 and 9) had p = 2, 6, 9 only. This tier runs the other
five degrees at L4 (the shared 99,458-triangle square mesh and the two annular L4 meshes; 0.59M
to 1.34M trace DOFs): 30 systems, 125 solver jobs, the ladder's candidates and protocol, from
the same campaign code as the Lmax tier. Each stage runs only if
`outcome_gate.py` passes on the smokes of its degrees (the Lmax tier's `80-smoke-*`).

| `--mem` | Degrees | Output | Estimate |
|---|---|---|---|
| 128G | 3, 7 | `85-l4-p3-p7` | ~1.1 h |
| 128G | 4, 8 | `86-l4-p4-p8` | ~1.3 h |
| 128G | 5 | `87-l4-p5` | ~0.5 h |

The estimates scale the ladder's L4 wall time (0.12 h at p = 2, 1.26 h at p = 9, all classes).
The exports are the PyPardiso inputs of the LU follow-up for these systems.

```bash
COPY=../<frozen copy of a committed revision that contains this campaign>
OUT=/scratch/user/hybridge/adr_h200_tuned
# p = 3, 7 (<lane-dir>/stages); likewise p = 4, 8 with the p4-p8 smokes and p = 5 with the p5 smokes
tuned-85-l4-p3-p7 bash -c 'cd $COPY && .venv/bin/python -B campaigns/adr_h200_tuned/outcome_gate.py $OUT/80-smoke-l1-p3-p7 $OUT/80-smoke-stress-l4-p3-p7 || exit 1; exec env HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13.1-bsr HYBRIDGE_TUNED_CALIBRATION_DEGREES="3 7" $COPY/campaigns/adr_h200_tuned/lane_stage.sh $OUT/85-l4-p3-p7 --levels L4 --degrees 3 7'
# sbatch --job-name=gpu-lane-<lane> --mem=128G --output=<lane-dir>/slurm-%j.out \
#     --export=ALL,LANE=<lane-dir>,LANE_SBATCH_OPTS=--mem=128G campaigns/gpu_lane/lane.sbatch
```

## Memory

The memory guard has three fixes:
- **Page cache:** `hybridge/runtime/devices.py` counts clean page cache as available. The unified campaign excluded L4 p=9 because about 148 GiB of reclaimable cache was counted as used.
- **Host fits:** `calibrate_assembly_memory.py` also fits each point's host peak (RSS). The runner uses that fit instead of the 6x host planning model, and drops the memory-mapped local inverses from the solve-time host estimate.
- **Single guard:** `HYBRIDGE_ADR_SINGLE_GUARD=1` keeps the vendored guard's estimates but drops its planning-model veto.

The host is the real ceiling at high p. The worker contract stores every element's local inverse on the host: 218 KB per triangle at p=9 and 56 KB at p=6. With `--mem=256G`, that blocks:
- p=9 from L7 (about 290 GB);
- p=6 at L9 (about 430 GB).

The H200 node has 2.27 TB, so a job with a larger `--mem` moves the ceiling to the H200 itself. Whichever limit binds is recorded as `memory_excluded` or `out_of_memory`.

## Running

Smoke stages run from the development checkout:
- `tuned-00-smoke-l1`: L1, trigonometric and cellular7_weak, p = 2, 6, 9;
- `tuned-01-smoke-stress-l4`: stress_main_orthogonal, L4, p=2.

The full ladder runs from a frozen copy (never edit it while its jobs run), in tiers:

```bash
lane_stage.sh /scratch/user/hybridge/adr_h200_tuned/10-l1-l5 --levels L1 L2 L3 L4 L4b L5
HYBRIDGE_TUNED_JOB_TIMEOUT=14400 lane_stage.sh /scratch/user/hybridge/adr_h200_tuned/11-l6-l9 --levels L6 L7 L8 L9
```

Tier 3 (L8, L9) uses the BSR-attach stack: the copy's `.venv` is a copy of `hybridge-venv-bsr`
and the stages set `HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13.1-bsr`. A borrowed-BSR smoke
gave the p=6 AMGX row the same iterations, L2 error and residual as the device-to-device
upload. Tier 3 runs after the smoke `73-smoke-cellular-v5` (L4 p=2 and p=6 cellular7_weak: the
annulus class with an AMGX candidate, the stress class has none; its gate requires
`bsr_borrowed: true` on every AMGX setup, warm-ups included, so a non-converged row still counts):

```bash
# meshes first, on a CPU node (gmsh; L9 p=2 is 17.8M triangles)
run_adr_h200_tuned_campaign.py --output /scratch/user/hybridge/adr_h200_tuned/70-l8-l9 --levels L8 L9 --mesh-only
HYBRIDGE_TUNED_JOB_TIMEOUT=21600 HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13.1-bsr \
    lane_stage.sh /scratch/user/hybridge/adr_h200_tuned/70-l8-l9 --levels L8 L9
```

The stress mesh generator once floored the bulk size at the neck size (and a retry replayed the
retained trials against the attempt budget); both were fixed on 2026-10-28, and the L7 stress
meshes for p=2 and p=6 in `51-l6-l7` were built with the fix.

Resume across checkouts: a job's `specification_sha256` covered the checkout paths in its spec
(`branch_root`, `inventory_directory`), so a mesh built by another checkout made the runner stop
with "Changed specification".
`reconcile_mesh_specs.py OUTPUT TARGET_COPY [--apply]` rewrites such records to the hash the
target copy computes, after checking the method on the copy's own records, and keeps the old
record as `result.before_spec_reconcile.json`; the mesh files and their `mesh_sha256` are
untouched (gmsh is deterministic).

### PyPardiso on the same systems

The unified PyPardiso run snapshotted its systems at launch and took about 24 min per system (about 2 min of start-up per worker), so it does not reach the large levels. Run the host reference on this campaign's output in a new directory, so that it never resumes the unified run's results by key:

```bash
bash campaigns/adr_unified/submit_genoa_pardiso.sh /scratch/user/hybridge/adr_h200_tuned/10-l1-l5 \
    /scratch/user/hybridge/adr_h200_tuned_pardiso/10-l1-l5 run_outputs/campaigns/adr_h200_tuned_pardiso/10-l1-l5
```

The job resubmits itself every 6 h while the device campaign is still assembling. L6 and above were expected to hit an int32 factor-index limit (projected about 6.5M DOFs at p=6). That was wrong: the 32-bit limit is on the matrix indices, and the LU follow-up factorized L6 p=6. Memory remains a possible limit; those outcomes are recorded as `failure_kind`.

## Reporting

Report per system and candidate, separately:
- setup;
- cold (setup + first solve);
- repeated (second solve on the same setup);
- warm per δ (1e-4, 1e-3, 1e-2), with iterations;
- amortized;
- for every job, the first (warm-up) solve's accuracy and where it came within 2× of its lowest true
  residual (`first_floor_*` in `docs/docs/campaigns/adr_h200_tuned/records/solver_jobs.csv`, from `export_records.floor_point`): iterations,
  exact or an upper bound, and the setup plus solve time to that point. The report plots it for jobs
  stopped at the FP64 residual floor (NC*), which have no timed sample.

Set them against PyPardiso:
- fresh: analysis + factorization + first solve;
- reused: repeated solves on the same factors.

Also report the device-route and raw-CUDA assembly times, and the H200/Blackwell ratios on matched systems. The crossover is stated per metric, per class and per p.
