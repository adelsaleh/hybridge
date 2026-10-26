# H100 assembly campaign — GENCI phase 1

Compare native CUDA assembly with batched-library implementations on the target GPU. This is
question Q1 of the submitted GENCI dossier, separate from the article convergence campaign.
The dossier's three-hour estimate comes from local timings; no H100 result is claimed here.

| Item | Value |
|---|---|
| Status (2026-10-26) | Planned; task registry, Slurm launcher and resume logic still to build |
| Target | One H100 per task; the dossier requested Jean Zay gpu_p6 |
| Planned GPU time | 3 hours, after phase 0's one hour of GPU parity checks |
| Output directory | `run_outputs/campaigns/h100_assembly` locally; `$WORK/hybridge/campaigns/h100_assembly` on the cluster |
| Output selection | Future launcher: `--out` or `HYBRIDGE_CAMPAIGN_OUT`; existing drivers require the output option shown below |
| Inputs | Existing mesh/case builders and the committed local assembly baseline |
| Records | None from H100 yet; collect JSON/CSV summaries and provenance after execution |

## Task matrix

| Family | Parameters | Implementations / measurements |
|---|---|---|
| AR transport | p = 4, 5, 6, 7, 8, 9; FP64 and FP32; baseline mesh near 117k triangles | Native fused and split3, tensor contractions through cuTENSOR/cuBLAS, batched local solves through MAGMA or native cooperative kernels |
| DR Poisson | Low and high degrees through p = 9; fixed mesh and trace bases | Raw CUDA and CuPy, COO/CSR/BSR, factor-cache policies, assembly and repeated-source/reconstruction phases |
| ADR | p = 2, 4, 6; scalar, constant-full and variable-full diffusion tensors | Native raw assembly, matrix formats, trace bases, block sizes and reconstruction |
| Profiling | Representative low/high degrees and coefficient cases | Nsight Compute captures of assembly, cache, RHS and reconstruction; keep unprofiled timing references separate |

FP32 is a diagnostic comparison, not a production-accuracy replacement. Warm the JIT kernels before
measurement, retain repeat samples and peak memory, and check assembly/reconstruction agreement at
the same quadrature and precision. Performance measurements run alone on the device. A shared-memory
or memory-capacity failure is a recorded result; it must not silently change the matrix or kernel.

## Existing entry points

These commands only show options:

```bash
.venv/bin/python scripts/gpu/benchmark_advection_tsle_tensor.py --help
.venv/bin/python scripts/gpu/benchmark_poisson_assembly.py --help
.venv/bin/python scripts/gpu/benchmark_fused_raw_assembly_kernels.py --help
.venv/bin/python scripts/gpu/capture_raw_assembly_profiles.py --help
.venv/bin/python scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_raw_cuda.py --help
```

The task builder will use these argument sets, with one process per precision and bounded task:

| Driver | Task arguments / output |
|---|---|
| `scripts/gpu/benchmark_advection_tsle_tensor.py` | `--orders 4,5,6,7,8,9 --native fused,split3 --repeats 5 --output-json FILE`; use separate processes with `HYBRIDGE_PRECISION=float64` and `float32` |
| `scripts/gpu/benchmark_poisson_assembly.py` | `--orders 2,4,6,7,8,9 --nx 250 --ny 200 --output FILE`; `--dry-run` prints cases without initializing CUDA |
| `scripts/gpu/benchmark_fused_raw_assembly_kernels.py` | One `--case`, `--order`, `--matrix-format`, `--phase` and `--coefficient-case` per task; preserve stdout and profile metadata |
| `scripts/gpu/capture_raw_assembly_profiles.py` | `--cases poisson transport adr --mode both --output DIR`; profiler runs are separate from unprofiled timing trials |
| `scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_raw_cuda.py` | `--orders 2 4 6 --coefficients scalar constant-full variable-full --reconstruction --output FILE` |

Local reference: [assembly baseline and precision findings](../../docs/research/solver_studies/advection_assembly_baseline_2026_10_03.md),
[baseline JSON](../../docs/research/solver_studies/advection_assembly_baseline_2026_10_03.data.json), and
[high-degree transport audit](../../docs/research/solver_studies/transport_assembly_audit_2026_10_16.md).
Record the exact baseline revision and match its mesh and options before calculating speedups.

## Preparation still required

Build the task registry, safe list/plan commands, Slurm submission, per-task logs and done/failed
markers, skip-completed resume, provenance, and comparison tables. Use the
[convergence campaign](../gc_article_convergence/README.md) as the launcher pattern. Keep the
small records here and the profiler captures and large arrays in the output directory.

The user prepares the target CUDA environment and any AMGX/MAGMA builds for sm_90. Follow the
[forked stack setup](../../docs/getting_started/forked_amgx_stack.md); no build is launched by
these instructions. The submitted dossier's software assumptions must be checked on the assigned
cluster before execution. There is no start/resume/stop command for the full campaign yet.

Source: the local submitted `genci_h100_dossier/sections/justification.tex` phase table and
`sections/methode.tex`, preserved in this task matrix on 2026-10-26. The later article convergence
runs are additional work, not part of this phase's three-hour estimate.
