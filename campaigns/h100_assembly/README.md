# H100/H200 assembly and recovery campaign — GENCI Q1

Compare native CUDA assembly with batched-library implementations on the target GPU. This is
question Q1 of the submitted GENCI dossier, separate from the article convergence campaign.
The dossier's three-hour estimate is historical. The H200 study covers all degrees, kernel
families, layouts and recovery stages; costs require measurement.

The [smoke workflow](../common/HOPPER.md) installs cuTENSOR, compiles MAGMA for sm_90,
and queues assembly smoke before solver smoke. Its setup and submission require a network-enabled
login shell. These smoke jobs do not launch the production task matrix below.

| Item | Value |
|---|---|
| Target | One H200 per task; H100 comparison requires a separately recorded identical experiment |
| GPU time | Expanded matrix; two-hour jobs with 110-minute batch budgets, at most two requeues per shard |
| Output directory | `run_outputs/campaigns/h200_expanded_20261026/{pilot,assembly}` for the submitted pipeline |
| Output selection | `run.py --output`; `plan.py --output-root` prints paths without writing |
| Inputs | Existing mesh/case builders and the committed local assembly baseline |
| Records | Collect JSON/CSV summaries and provenance after execution |

## Task matrix

The deterministic production registry is inspectable without CUDA, file writes or submission:

```bash
.venv/bin/python -m campaigns.h100_assembly.plan
.venv/bin/python -m campaigns.h100_assembly.plan --group transport --format shell
.venv/bin/python -m campaigns.h100_assembly.plan --group recovery --order 9
```

The expanded registry contains **2,997 task records**, each bounded and isolated:

| Group | Tasks | Coverage |
|---|---:|---|
| Transport | 504 | p1–p9, two mesh sizes, two trace bases, native FP64/FP32 and all 12 tensor precision/contraction/LU combinations; response reconstruction included |
| Tensor layouts | 432 | Cooperative tensor solve at 32/64/128 threads, both precisions, cuBLAS/cuTENSOR, every degree/basis/mesh |
| Native layouts | 108 | Paired fused/split3 qualification, registers/shared/local memory, explicit block sizes |
| Partially fused transport | 144 | CuPy local construction plus CUDA LU/condensation, COO, 1/32/64/128 threads; 48 p7–p9 records are known unsupported and do not launch kernels |
| Matched transport controls | 432 | Fully fused and split3 on the same rectangular Dirichlet fixture as the partial path; legal COO/CSR/BSR combinations |
| Poisson comparison | 18 | p1–p9, both trace bases, raw versus CuPy, 100k triangles |
| Poisson kernels | 324 | All degrees, meshes, bases, formats and factor-cache policies; thorough tuning, RHS reuse, reconstruction and legacy comparison where it fits |
| Tensor ADR | 162 | p1–p9, scalar/full/variable tensors, three cache policies, serial/cooperative kernels, both bases and all sparse formats |
| Recovery | 756 | Device reconstruction, Poisson primal recovery, ADR L2/RT flux and primal postprocessing; cache-policy controls |
| Primal parity | 18 | Independent CuPy contractions versus fused tensor primal recovery through p9 |
| Raw profiles | 18 | Nsight assembly/cache/RHS/reconstruction across every degree |
| Recovery profiles | 27 | Nsight Poisson and ADR L2/RT recovery across every degree |
| Tensor profiles | 54 | Nsight cuBLAS/cuTENSOR with cooperative/cuBLAS/MAGMA local solves across every degree |

IDs/specification hashes do not depend on the output root. The JSON remains `launch_ready=false`
until the smoke/pilot checks pass. Per-task timeouts are caps, not runtime estimates. ADR recovery
above p6 is explicitly labeled a high-order probe, not previously established accuracy evidence.
Partial transport only supports p<=6 and COO; the plan never labels its output as CSR/BSR.

| Family | Parameters | Implementations / measurements |
|---|---|---|
| AR transport | p = 1–9; FP64 and diagnostic FP32; coarse and approximately 117k-triangle meshes | Native fused, partially fused and split3; tensor contractions through cuTENSOR/cuBLAS; cooperative/cuBLAS/MAGMA LU |
| DR Poisson | Low and high degrees through p = 9; fixed mesh and trace bases | Raw CUDA and CuPy, COO/CSR/BSR, factor-cache policies, assembly and repeated-source/reconstruction phases |
| ADR | p = 1–9; scalar, constant-full and variable-full diffusion tensors | Native raw assembly, matrix formats, trace bases, block sizes, factor caching and reconstruction |
| Postprocessing | p = 1–9 | Primal recovery, L2-closest/RT flux recovery and independent contraction parity |
| Profiling | Every degree and kernel family | Registers, shared/local memory, spills, occupancy and measured compute/DRAM counters; profile timings excluded from rankings |

FP32 is a diagnostic comparison, not a production-accuracy replacement. Warm the JIT kernels before
measurement, retain repeat samples and peak memory, and check assembly/reconstruction agreement at
the same quadrature and precision. Performance measurements run alone on the device. A shared-memory
or memory-capacity failure is a recorded result; it must not silently change the matrix or kernel.
The full Q1 registry is a device campaign. Any CPU-only assembly comparisons or host preparation
must run in a separate CPU-node campaign, never as CPU benchmarks inside a GPU allocation.

## Existing entry points

These commands only show options:

```bash
.venv/bin/python scripts/gpu/benchmark_advection_tsle_tensor.py --help
.venv/bin/python scripts/gpu/benchmark_poisson_assembly.py --help
.venv/bin/python scripts/gpu/benchmark_fused_raw_assembly_kernels.py --help
.venv/bin/python scripts/gpu/capture_raw_assembly_profiles.py --help
.venv/bin/python scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_raw_cuda.py --help
.venv/bin/python -m scripts.gpu.benchmark_hdg_recovery --help
```

The task builder will use these argument sets, with one process per precision and bounded task:

| Driver | Task arguments / output |
|---|---|
| `scripts/gpu/benchmark_advection_tsle_tensor.py` | One degree/basis/mesh/configuration per process; native FP32 uses its own process; tensor FP32 uses an FP64 process/reference. Raw samples and reconstruction are saved. |
| `scripts/gpu/benchmark_poisson_assembly.py` | `--orders 2,4,6,7,8,9 --nx 250 --ny 200 --output FILE`; `--dry-run` prints cases without initializing CUDA |
| `scripts/gpu/benchmark_fused_raw_assembly_kernels.py` | One `--case`, `--order`, `--matrix-format`, `--phase` and `--coefficient-case` per task; preserve stdout and profile metadata |
| `scripts/gpu/capture_raw_assembly_profiles.py` | `--cases poisson transport adr --mode both --output DIR`; profiler runs are separate from unprofiled timing trials |
| `scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_raw_cuda.py` | p1–p9, scalar/full/variable coefficients, cache none/LU/LU+mass, serial and cooperative blocks, reconstruction |

Local reference: [assembly baseline and precision findings](../../docs/research/solver_studies/advection_assembly_baseline_2026_10_03.md),
[baseline JSON](../../docs/research/solver_studies/advection_assembly_baseline_2026_10_03.data.json), and
high-degree transport audit (`../guiding-center-cg-hdg/docs/research/solver_studies/transport_assembly_audit_2026_10_16.md`; saved artifact).
Record the exact baseline revision and match its mesh and options before calculating speedups.

## Submission, resume and records

The dependency chain is solver smoke → pilot → array. The 11-task pilot checks new recovery,
partial transport and tensor paths before the full eight-shard array is eligible. The array uses
one H200 at a time (`%1`). `submit.sh` can change concurrency with `CAMPAIGN_MAX_CONCURRENT`.

Inspect a fresh submission without sending it:

```bash
bash campaigns/h100_assembly/submit.sh SMOKE_JOB_ID run_outputs/campaigns/hopper_smoke_STAMP/solvers 8 run_outputs/campaigns/h200_expanded_STAMP --dry-run
.venv/bin/python -m campaigns.h100_assembly.run --help
```

`run.py` refuses a CPU allocation and requires the latest smoke completion to pass. Each task
keeps `tasks/<id>/attempt_<n>/`, commands, logs, artifacts and `result.json`. Successful outputs
are skipped only when specification, sources, stack binaries, GPU model/driver and artifact
hashes agree. Known unsupported paths get explicit unmeasured records. Other failures remain
terminal unless explicitly retried. A time-budget interruption is retryable; the batch script
requeues an incomplete shard at most twice. Stop the pipeline with `scancel PILOT_JOB_ID ARRAY_JOB_ID`.

Batch source archives/provenance and completion summaries live under `batches/`. Mesh fingerprints
identify actual geometry/connectivity. Profile artifacts and profile timing rows are separate from
ranking samples. Source changes invalidate the batch. Do not edit numerical sources during runs;
use a new output root for changed specifications or software. Hardware failure/timeout and an
unsupported path are not successful numerical results. Comparison tables still require the runs.

Source: the local submitted `genci_h100_dossier/sections/justification.tex` phase table and
`sections/methode.tex`, preserved in this task matrix on 2026-10-26. The later article convergence
runs are additional work, not part of this phase's three-hour estimate.
