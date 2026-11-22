# GPU smoke campaigns

The smoke campaigns check that a GPU node has everything the H200 studies need
before a long campaign is started: cuTENSOR, MAGMA, AMGX, PyAMGX and PyPardiso,
and correct results from each of them on the GPU.

From the repository root:

```bash
bash campaigns/common/setup_hopper.sh
bash campaigns/common/submit_hopper_smoke.sh --dry-run
bash campaigns/common/submit_hopper_smoke.sh
```

## Setup

Setup installs `cutensor-cu13==2.8.1`, downloads MAGMA 2.10.0, and builds the shared `magma`
target with CUDA 13.1, `GPU_TARGET=sm_90`, LP64 OpenBLAS, and four compiler workers.
It checks the cubins, every MAGMA symbol used by Hybridge, and CuPy's cuTENSOR import
(including the `libcutensorMg` preload). This is a host check, not a GPU correctness result.
Build and configure logs and the downloaded source digest go under `.cache/hopper-setup/`.
The shared library is installed in the workspace's `builds/magma/cuda13.1-sm90/lib/` directory;
`HYBRIDGE_BUILDS_ROOT` changes the workspace build root.

Build MAGMA on a node that has the CUDA toolkit. `hopper_build.sbatch` asks for CPUs on the
GPU partition without a GPU, resumes an already configured build and checks the cubins and the
linkage. `submit_hopper_smoke.sh --after-build JOB_ID` makes the smoke jobs wait for that build.

Overrides: `HYBRIDGE_MAGMA_SOURCE` for an existing official release source tree,
`HYBRIDGE_CUTENSOR_WHEEL` for an offline wheel, `HYBRIDGE_MAGMA_ROOT` for the build directory,
`HYBRIDGE_MAGMA_LAPACK` for an LP64 BLAS/LAPACK library, and `HYBRIDGE_BUILD_JOBS` for compiler
parallelism. `hopper_env.sh` sets the runtime environment for every Slurm job.

## Submission

Each job asks for partition `hopper`, one `gpu:h200_nvl`, 16 CPU cores, 128 GiB host memory and
two hours. It reserves one GPU, not a full node. `CAMPAIGN_PARTITION` and `CAMPAIGN_GRES` change
the partition and the GPU request. The Slurm account is passed only when `CAMPAIGN_ACCOUNT` is
set; otherwise Slurm uses `SBATCH_ACCOUNT` or your default account. Other clusters also need
their own modules, QoS and resource settings.

Assembly smoke runs first. Solver smoke is submitted with `afterok:<assembly-job-id>` and
`--kill-on-invalid-dep=yes`, so a failed assembly smoke prevents solver smoke from starting.
Each job first requires FP64/FP32 cuTENSOR contractions and MAGMA batched LU/solve
and fused solves to agree with CPU references. Missing libraries are failures, never skips.

## What the smoke jobs cover

Assembly smoke covers tensor parity, native transport FP64/FP32 at p4/p9, every implemented
cuBLAS/cuTENSOR + cooperative/cuBLAS/MAGMA configuration, DR p2/p6/p9 assembly parity,
ADR p2/p6 scalar/full/variable tensor coefficients and reconstruction, and one Nsight capture.
Counter permissions are tested by the capture itself; its failure blocks readiness.
Solver smoke covers all stress-driver candidates and PARDISO on identical cached operators
for well-conditioned square p2/p6 and annulus p2 cases, plus three Poisson
candidates at p6 on the archived L2 mesh. The Poisson mesh has about 107k triangles and
exercises the full multigrid hierarchy; saturated snapshots are disabled.

Smoke output shows that a node is ready. It does not answer the assembly (Q1) or solver (Q2)
study questions, and it does not validate their memory limits.

## Outputs

The output root defaults to `run_outputs/campaigns/hopper_smoke_<timestamp>/`
(`HYBRIDGE_SMOKE_ROOT` sets it). It contains `jobs.txt`, `slurm/`, and `assembly/` and
`solvers/` subdirectories. Each suite keeps separate `attempt_<n>/` directories with source
archives and hashes, environment, GPU and CPU provenance, exact commands, task logs and
results, and `completion.json`. An incomplete attempt is never reused as success.
Submitting again with the same `HYBRIDGE_SMOKE_ROOT` makes a new attempt. Follow and stop
the jobs with:

```bash
squeue -u "$USER"
cat <output-root>/jobs.txt
tail -f <output-root>/slurm/<job-log>
scancel <assembly-job-id> <solver-job-id>
```

## Placement of full campaigns

Full host preparation and CPU PARDISO campaigns run on CPU nodes (`genoa`), never on GPU
nodes, even with a CPU-only allocation (`production_placement.py` checks this). Full device
campaigns contain GPU assembly and iterative solves with no inline host reference or CPU
assembly fallback. The two exchange immutable systems and check that the matrix, right-hand
side and topology hashes match. The smoke jobs, including the CPU-only build on a GPU node and
the PARDISO checks inside the GPU allocation, are exempt.
