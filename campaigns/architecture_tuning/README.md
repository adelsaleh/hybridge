# Architecture tuning qualification

On the H200 (2026-10-31) the initial qualification passed 72 selection/tensor and 7 Poisson
checks, and the extended full AR, split3, ADR, solver and selected-FP32 qualification passed
372 checks. The evidence is in `docs/campaigns/architecture_tuning/README.md` (local, untracked).
This numerical campaign belongs in Hybridge. It checks the bounded tuner, persistent reuse,
actual batched local solves against NumPy, tensor assembly/response parity,
Poisson p=9 formats/caches, full AR/standalone split3, ADR representation reuse,
and solver true residuals. Each invocation uses one allocated GPU; count all running
and pending GPUs against the shared five-GPU limit before submitting.

From the repository root:

```bash
.venv/bin/python -B campaigns/architecture_tuning/qualify.py --plan
sbatch campaigns/architecture_tuning/qualify.sbatch
# Separate independent lanes, each with a distinct output and one GPU:
sbatch campaigns/architecture_tuning/qualify.sbatch --stages selection transport poisson
sbatch campaigns/architecture_tuning/qualify.sbatch --stages adr
sbatch campaigns/architecture_tuning/qualify.sbatch --stages solvers
sbatch campaigns/architecture_tuning/qualify.sbatch --stages precision32
# Resume the same source, software/device and output path:
HYBRIDGE_CAMPAIGN_OUT="$PWD/run_outputs/architecture-tuning-JOBID" sbatch campaigns/architecture_tuning/qualify.sbatch
# Stop:
scancel JOBID
```

For another allocated architecture, run `qualify.py --output PATH` with that
machine's supported CUDA environment. The Slurm wrapper is specific to Hopper.
A frozen checkout must come from a committed revision; set `HYBRIDGE_REPO` to
it and `HYBRIDGE_PYTHON` to the installed HDG environment. Stage `.done` markers
support resume, changed source/device/command identity is rejected, and
`COMPLETE` means every requested stage passed. The shared campaign protocol records
commit, source hashes, host, device, toolchain and fork revisions. Logs and
tuning JSON go into `HYBRIDGE_CAMPAIGN_OUT`; compact qualification evidence is
committed under `docs/campaigns/architecture_tuning/`.
Each stage runs in a separate process with recorded precision: FP64 by default,
FP32 for `precision32`. Pytest temporary caches are retained with the output.

This is a correctness gate. Full-mesh performance transfer and Quadro/Blackwell
qualification are separate from a Hopper pass; no speedup is inferred here.
