# ADR recovery on H200, p2–p9

Measured 2026-11-01: the numerical gates took 28 s, the eight-degree runner smoke 1m38s, and
the large run **19m23s on one H200**, including its 36-test numerical gate; the 35
host/runner/layering checks passed. A run's output is `run_outputs/adr-recovery-*/`; progress
is in the Slurm log, each degree's `progress.jsonl`, and the incrementally generated
`REPORT.md`. No production turbulence configuration is changed automatically.

## Measured outcome

Every degree used **524,288 distorted affine triangles**. The values below are
speedups over the best measured raw-assembly reference with its selected dense
solver. Projection into Pp is included. The detailed phase timings, full costs,
resource attributes and rejected candidates remain in the immutable run report.

| p | Reuse speedup | Rebuild speedup | Selected dense solver |
|---:|---:|---:|---|
| 2 | 1.76× | 1.38× | fused |
| 3 | 1.79× | 1.31× | fused |
| 4 | 1.67× | 1.16× | fused |
| 5 | 1.69× | 1.22× | fused |
| 6 | 1.89× | 1.47× | fused |
| 7 | 3.85× | 2.93× | MAGMA |
| 8 | 4.76× | 3.50× | MAGMA |
| 9 | 5.22× | 3.46× | MAGMA |

Tensor assembly won the reuse comparison at every degree. For rebuilding,
compact raw assembly won p2–p4 and tensor assembly won p5–p9. Retaining flux
maps won the reuse cases; when rebuilding, direct uncached flux recovery won
p4–p9. The tensor configurations retain a constructor `threads=128` placeholder;
that parameter is used only by raw/compact assembly. GEMM launch choices belong
to cuBLAS, not this thread-count field.

The largest full-mesh relative coefficient difference was **3.72e-14**, and the
largest relative cell-mean defect was **1.85e-14**. The largest scaled dense
backward error in the pilot solver checks was **1.64e-16**. Every cell was
visited under all four full-mesh path/policy combinations; all markers and
230 frozen source hashes were authenticated by `analyze.py`.

**Memory scope:** reuse retains the active chunk's cache across repeated calls;
the benchmark does not retain every chunk simultaneously. The p9 full-mesh flux
maps alone would occupy **127.36 GiB**, excluding other simulation data. Its
reuse result is therefore conditional on cache availability. The measured
chunked run peaked at **22.62 GiB** in sampled device telemetry. The p9 chunk
contained 6,016 cells; other degrees used 8,192. Rebuild timings remain useful
when full-mesh cache storage is impractical.

Compact audited artifacts (local, untracked, in `docs/campaigns/adr_recovery/outputs/<run>/`):
timing CSV `timings.csv`, audit JSON `audit.json`, and the plot `adr_recovery_h200` as PNG, PDF
and SVG.
Numerical kernels live in `hybridge/mixed/postprocess/adr_recovery.py`; the
production solver accepts an explicit recovery workspace, with its existing
default preserved. This campaign compares raw tiled, compact tiled and
reference-product GEMM assembly and cuBLAS, MAGMA and fused dense solves.

Every degree compares two lifecycle policies: reuse fixed geometry and flux
correction maps, or rebuild those per-element data on every invocation.
Both retain immutable reference-basis tables; their cold construction is
reported separately rather than charged to rebuilding local element data.
Velocity, stabilization, diffusion and RHS are refreshed in either policy.
Both start from the same already reconstructed ADR unknowns and trace; upstream
PDE assembly, the global solve and mixed reconstruction are outside these
recovery timings. No p-degree factor is assumed to be a valid p+1 recovery
factor. Setup costs and retained memory are reported separately.

Bulk outputs belong in `run_outputs/adr-recovery-*`; compact evidence belongs in
`docs/campaigns/adr_recovery/`. Qualification uses one H200 for at most 20 minutes;
the production timing allocation is bounded separately to 2h30m. Submit only
after checking the shared five-GPU limit, with a unique descriptive job name.

## Commands

Use the primary checkout and the maintained Python environment:

```bash
cd /path/to/hybridge
export PYTHONPATH=$PWD
PY=.venv/bin/python
$PY -B -m campaigns.adr_recovery.run --plan
$PY -B -m campaigns.adr_recovery.submit --smoke --execute
$PY -B -m campaigns.adr_recovery.submit --execute
```

Without `--execute`, submission prints the concrete plan. The launcher freezes
source hashes, saves changed source bytes and patches, counts running/pending
GPUs across the user, and submits one uniquely named H200 job. Highest verbosity
is always enabled: every candidate, warmup, repetition, rejection, chunk and
phase is flushed to progress JSONL, degree logs and the Slurm log.

The main campaign has an 8,700-second cumulative budget, including qualification
and setup; Slurm caps its allocation at 2h30m. Each degree is isolated in a child
process with a smaller deadline, so a difficult p cannot consume later degrees'
budget. Reports and validated degree markers survive an interruption. A worker
that reaches its budget remains incomplete, never silently counted as success.

The mesh is adaptively chosen between 192×192 and 512×512 rectangular cells
(73,728–524,288 triangles) using a pilot measured at the maximum mesh's physical
scale. Memory is bounded independently: chunks contain at most 8,192 elements,
with a conservative peak estimate capped at 24 GiB and 35% of initially free
device memory. Higher degrees can use smaller chunks. Full-mesh samples sum
actual chunk executions; they are not timing extrapolations.

Raw and compact assembly test 32/64/128/256/512 threads; illegal compiled
configurations are reported. Tensor assembly uses FP64 GEMM of reference
products; no lower precision or automatic tensor-core acceleration is claimed.
cuBLAS, MAGMA and four deduplicated fused configurations compete on identical
unfactored systems. Matrix/RHS parity and dense backward error gate selection.
Best assembly and flux policy are selected by complete recovery timings for
each lifecycle. The full-mesh raw reference uses its best measured launch and
solver, uncached flux constraints and the original duplicated coefficient
sampling; optimized paths share those samples. Projection back to Pp is included
in full-mesh times. Input generation, host/GPU qualification and cache setup are
recorded separately; rebuild timings include setup on every call.

Resume with the *same* frozen options and output after the previous job is
terminal, for example (obtain a fresh unique name and check `squeue -u "$USER"`
against the shared five-GPU limit first):

```bash
export HYBRIDGE_REPO=$PWD
export HYBRIDGE_PYTHON=.venv/bin/python
export HYBRIDGE_CAMPAIGN_OUT=/absolute/path/from/launch.json
sbatch --job-name="adr-recovery-p2-p9-resume-$(date -u +%Y%m%dT%H%M%SZ)" \
  --output="$HYBRIDGE_CAMPAIGN_OUT/slurm-%j.log" --export=ALL campaigns/adr_recovery/run.sbatch
```

Resume verifies sources and result hashes, skips completed degrees, archives
failed attempts, and deducts prior runtime from the cumulative budget. To stop,
use `scancel JOBID` with the ID in `launch.json`. Regenerate the report without
a GPU using `$PY -B -m campaigns.adr_recovery.run --report --output OUTPUT`.
The report lists per-degree choices, resource attributes, phase timings,
reuse/rebuild totals, cache bytes, numerical defects and any rejected candidates.

Authenticate a completed run and regenerate its compact table and figure:

```bash
$PY -B -m campaigns.adr_recovery.analyze --output OUTPUT \
  --destination docs/campaigns/adr_recovery/outputs/JOBID
```

This analysis reads completed run artifacts without changing them. The added
analysis module is a post-run addition; original numerical sources and their
snapshots remain intact. Start a new timing run with a fresh output directory
to include it in a new manifest.
