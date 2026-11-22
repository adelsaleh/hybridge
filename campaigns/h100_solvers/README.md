# H100/H200 solver campaign — GENCI phase 2

Compare iterative trace-system solvers and multithreaded PyPardiso on identical assembled systems.
This is question Q2 of the submitted dossier. The scope includes Poisson, transport-dominated ADR,
anisotropy and difficult closed-loop transport. Local solver results and failures are the starting
point; none is an H100/H200 result.

The [smoke workflow](../common/HOPPER.md) queues solver smoke after assembly smoke succeeds.
It checks all candidate implementations and matched PARDISO references on small, well-conditioned
ADR systems plus a Poisson multigrid system. The production matrix below remains a separate stage.

| Item | Value |
|---|---|
| Target | One H100 or H200 per device task; full host preparation and PyPardiso run separately on CPU nodes (`genoa`) |
| GPU time | Expanded beyond the original 12-hour/~250-trial dossier matrix; measured costs pending |
| Output directory | `run_outputs/campaigns/h100_solvers` locally; `$WORK/hybridge/campaigns/h100_solvers` on the cluster |
| Output selection | Future launcher: `--out` or `HYBRIDGE_CAMPAIGN_OUT`; ADR drivers already take `--output` |
| Inputs | Portable ADR inventory and meshes, archived matrices where available, Poisson mesh/source specifications |
| Records | Local baseline snapshots under `adr_unified/records`, `adr_pardiso/records`, `adr_stress/records` |

## Trial matrix

Audit the portable ADR inventory before selecting production trials (no CUDA or file writes):

```bash
.venv/bin/python -m campaigns.h100_solvers.audit
.venv/bin/python -m campaigns.h100_solvers.plan --campaign host
.venv/bin/python -m campaigns.h100_solvers.plan --campaign device
.venv/bin/python -m campaigns.h100_solvers.expanded --summary
.venv/bin/python -m campaigns.h100_solvers.plan --scope archive --campaign host
```

The default plan is now the **expanded** study: all 43 checked-in AMGX files, archived
definitions, every implementation path returned by the existing native configuration builder,
three pMG policies, and CuPy Krylov controls. Identical definitions are deduplicated while source
aliases remain. There are **214 configurations** before equation/degree capability probes,
with **31 problem classes**: 25 ADR (including all archived stress classes and tensor cases),
four pure Poisson geometries, and two pure transport fixtures. Degrees p1–p9 apply throughout.
Symmetric CG/PCG configurations are not applied to nonsymmetric problems. AMGX filenames are
not trusted as preconditioner descriptions: effective BICGSTAB behavior is recorded explicitly.

Screen every eligible configuration/degree first. Then measure four mesh sizes at 3%, 15%, 40%
and 75% of the calibrated usable device memory, followed by bounded capacity probes up to 90%
with at least 8 GiB reserved. Retain nonconvergence, memory exclusions and earlier local failures.
Capacity search stops at an explicit resource/time bound, not because PARDISO wins at a smaller
mesh. PP degree, restart, orthogonalization, pMG policy and iteration-cap tuning are confirmed
independently at each HDG degree. Raw sample medians/IQR and setup/reuse crossover counts matter;
there is no universal fastest preset across degrees and regimes.

The exhaustive design has 53,487 screening points and 213,948 four-size scaling points before
capability exclusions and measured resource decisions. These are **coverage bounds**, not
submitted jobs or runtime predictions. The Q1 assembly measurements supply the kernel/memory
choices for its adapters. Generic cached-system adapters are still required for pure Poisson, pure transport,
modern ADR tensors and host-precomputed inverse/basis data.

The default replay expands 107 archived systems with 21 L5 extensions into **947 iterative
jobs**, with about **835.8 GiB** of modeled matrix caches. At the dossier's assumed three
minutes per trial that would mean **47.35 hours**, not a measured target-machine duration.
It excludes the additional Poisson, transport and PARDISO trials. This remains an archival
cost scenario, not the expanded study's time limit or a measured host/device breakdown.
It reports candidate families, degrees, system IDs, and conservative memory/storage estimates
per problem class, preserving stress cases. The estimates require calibration on the target GPU.

The `--scope archive` registries separate host and device tasks. For the complete, unfiltered archive,
the host registry contains 128 input-preparation tasks and 128 PARDISO replays on `genoa` with no
GPU allocation. The device registry contains 128 GPU assemblies and 947 iterative trials on
`hopper`, with no CPU assembly fallback or inline PARDISO calls. These counts describe coverage,
not the final selected matrix, and `launch_ready` remains false until stage runners and calibration
are complete. The 47.35-hour scenario counts iterative trials only; it does not measure host time.

Dependencies are per system: CPU input preparation → GPU assembly → independent GPU iterative
and CPU PARDISO trials. Both consumers require the same saved matrix/RHS and topology hashes;
the host replay must not reassemble its own reference system. Host timings, device timings and
transfer/staging costs must be reported separately. Full host jobs must use CPU nodes, including
when a GPU node could offer a CPU-only allocation. The placement guard in
`campaigns/common/production_placement.py` rejects those allocations and must be called by the
future full stage runners before worker imports. Smoke jobs are exempt and keep their PARDISO
checks in the GPU allocation.

| Axis | Planned coverage |
|---|---|
| Equations and cases | Poisson; regular and transport-dominated ADR; anisotropic diffusion; square/annulus trap, cross and orthogonal stress cases |
| Meshes | Four calibrated sizes per degree/path plus bounded near-capacity probes |
| Degrees | p = 1–9 across assembly, iterative solves, reconstruction and postprocessing; unsupported combinations explicit |
| Iterative candidates | All registered AMGX/native/CuPy configurations, ASM/BJ and PP implementation paths, pMG policies; degree-specific tuning |
| Direct reference | Separate CPU campaign: PyPardiso LU, or the applicable SPD path, on exactly the same saved system |
| Measurements | Assembly, setup/factorization, cold solve and repeated solves separated; physical residual, iterations, peak CPU/GPU memory, timeout and failure reason |

Candidate support differs by equation and degree: do not turn an unavailable preconditioner into a
silent fallback. Failed local stress configurations remain explicit trials rather than disappearing
from the comparison. Match precision, normalization, ordering and stopping tolerances across runs.
Use at least 16 CPU threads or all allocated cores on the CPU nodes for the large direct solves; verify the backend
thread count and observed parallel CPU use. Environment variables alone do not establish that.
Performance trials use an exclusive GPU. Accuracy preparation may share it when memory permits.

## Inputs and local evidence

- [Portable ADR inventory](../../run_configs/adr_unified_l5/inventory.json) and its mesh assets:
  freeze the inventory revision and hashes. The unified worker reassembles the systems and uses
  the vendored solver tree; it does not depend on an uncommitted sibling checkout.
- Existing large data: `run_outputs/solver_studies/adr_scaling_2026_09_17` and the stress output
  directories. Copy required matrices and source inputs to WORK before a replay. The earlier
  plan estimated the local archive at about 76 GB; the collected DATA.md files give current
  regular-file sizes. The full expanded ADR matrix cache was estimated at **840 GiB** in the
  dossier; this is storage, not device memory, and must be checked against the selected tasks.
- [Unified ADR campaign](../adr_unified/README.md), [PyPardiso reference](../adr_pardiso/README.md),
  [stress campaign](../adr_stress/README.md), and [guiding-center replay records](../../../guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_solver_records/README.md).
- [Poisson local comparison](../../../guiding-center-cg-hdg/docs/research/solver_studies/poisson_300k_solver_comparison_2026_09_22.md),
  [Poisson pMG degree campaign](../../../guiding-center-cg-hdg/guiding_center_cg_hdg/campaigns/gc_poisson_pmg_degrees/README.md), and
  [pMG/AMG audit](../../../guiding-center-cg-hdg/docs/research/solver_studies/poisson_pmg_amg_audit_2026_10_16.md).

The node's actual free memory determines the largest trial. H100/H200 comparisons must record
which device was used and use identical problem sizes before reporting hardware ratios.
At sizes beyond PARDISO's memory or sparse-index limits, retain independently validated GPU
results and report the direct reference as unavailable; do not manufacture a speedup ratio.
Choose the guiding-center configuration from total assembly/setup/solve/recovery cost and
captured early/filamented/saturated Poisson and transport systems for BDF2, BDF3 and IMEX-ARK3.
Poisson hierarchy reuse, transport matrix changes, zero guesses and previous-solution guesses
are separate experiments. Never reuse factors for a changed matrix without an explicit policy.

## Existing entry points

Inspect options without starting solves:

```bash
.venv/bin/python campaigns/adr_unified/run_adr_unified_campaign.py --help
.venv/bin/python campaigns/adr_pardiso/run_adr_pardiso_campaign.py --help
.venv/bin/python campaigns/adr_stress/run_closed_loop_stress.py --help
```

The ADR drivers have plan/inventory modes by default and require `--execute` to start their runs;
see their READMEs for the exact inputs and flags. Research Poisson comparisons
and their launch commands live in the sibling `guiding-center-cg-hdg` repository.
The combined cluster launcher remains to be implemented.

## Preparation still required

The expanded catalogue, archive host/device registries and placement contract are prepared. Implement the expanded adapters,
build shard plans and separate stage runners, stage input files with hashes, cap each worker's
memory/time, preserve per-run logs and failures, write done/failed markers, skip completed trials
on resume, and compare with the local baselines. Save small JSON/CSV records here after execution;
keep matrices and field checkpoints in WORK/STORE. Record solver-stack revisions and the node's
CPU affinity and measured thread use in provenance.

The target stack for Hopper (sm_90) follows the
[forked stack setup](../../docs/getting_started/forked_amgx_stack.md). There is no combined
start/resume/stop command yet. Source: submitted `genci_h100_dossier/sections/justification.tex`,
phase 2 and the data-management section.
