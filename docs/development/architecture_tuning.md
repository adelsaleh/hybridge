# Architecture-aware kernel tuning

The repository split preserves architecture optimization in **Hybridge**.
The research repository supplies scientific cases and validation evidence;
Hybridge imports no research package to choose or execute a kernel.

The user confirmed this policy on 2026-10-31:

- An unseen GPU/workload receives automatic, correctness-checked calibration.
- The first-use limit is **30 seconds and eight candidates**, including the
  conservative reference. A running compilation or CUDA launch cannot be
  preempted; no further candidates start after the time limit.
- Equivalent algorithms, layouts and launch sizes may compete. Precision,
  discretization, quadrature, solver/preconditioner and convergence tolerances
  remain fixed. Reduced precision is never introduced by calibration.
- A persistent machine-local cache amortizes calibration. Explicit numerical
  and benchmark overrides remain available.
- Calibration preserves device residency: sampled arrays and independent
  reference calculations stay on the GPU; only scalar classification and
  acceptance results return to the host.
- The new supplied-order AMGX ILU(k) path remains **experimental**. Its
  autotuning is deferred by the user's subsequent instruction on 2026-10-31.
  It requires an explicit AMGX configuration; automatic calibration does not
  select ILU(k), change its fill level, or tune its kernels.

## What the three architectures taught us

The settings measured on each architecture and the lessons, with their numbers,
are in [kernel tuning by GPU architecture](../lessons/kernel_tuning_by_architecture.md).
The table below keeps what each one means for the automatic selection.

| Evidence | Consequence for automatic selection |
|---|---|
| [Quadro sm_75](../algorithms/assembly/quadro_rtx6000_sm75/README.md): launch recommendations came from resource-constrained cooperative kernels; the newer October paths were not qualified there | Keep a conservative supported baseline; check actual compiled resources. Newer paths need qualification, not an architecture-name substitution. |
| [Blackwell sm_120](../algorithms/assembly/rtx_pro_5000_sm120/README.md): FP64-limited assembly, 99 KiB shared limit, cooperative/MAGMA tradeoffs varying with degree | Precision and FP32:FP64 throughput belong in selection; screen memory layout and benchmark competing local solvers. FP32 proxy results cannot choose production FP64 arithmetic. |
| [Hopper sm_90](../algorithms/assembly/h200_sm90/README.md): old Poisson quick ranking missed faster wide batches; tensorized AR and fused local solves help, while high-order ADR benefits from wider launches | Use measurements as candidate-order priors. Re-measure on the actual GPU/workload; H200 defaults are not universal winners. |
| [H200 solver kernels](../../campaigns/solver_h200/README.md): packed/coalesced layouts help some operators/degrees and regress others | Tune each bandwidth-sensitive operation by shape, layout and reuse regime. Keep end-to-end residual/iteration checks separate from microkernel timing. |

## Shared implementation and present coverage

`hybridge/runtime/cuda_tuning.py` is below linalg and HDG in the dependency
order. `hybridge/hdg/cuda/tuning.py` retains compatibility imports. The shared
`select_variant` protocol supplies named candidates, resource-checked compile
callbacks, isolated trial outputs and optional independent validation. A
`before` callback restores destructive inputs outside the measured event
interval. `trial_prepare` constructs private samples and warms plans only on
misses. Related stages can share an absolute calibration deadline without
putting transient timestamps into persistent identities. Compile/resource unavailability rejects a candidate; fatal kernel
execution errors propagate. A failed conservative reference aborts calibration.

The tuner validates the baseline, rejects nonfinite or discrepant candidates,
and measures warm CUDA-event medians. A candidate must improve the current
winner by at least 3% to displace it, limiting changes caused by timing noise.
The budget may leave a valid baseline without any measured alternatives; the
record identifies that outcome and can be explicitly refreshed.

Cache identity includes schema/policy, device name and UUID, architecture,
resource limits, SM count, FP64 ratio, driver/runtime/CuPy versions, workload,
precision, source identity, candidate roster and calibration budget. Workload
adapters add dimensions, bases, sparse format, cache policy, coefficient or
boundary branches and size buckets as relevant. Atomic JSON records include
selection, candidate timings/rejections, inputs and budget outcome. Corrupt or
unwritable caches do not prevent a solve. Cache hits still screen resource
availability and never allocate trial buffers.

| Path | Current behavior |
|---|---|
| Poisson assembly, RHS refresh, reconstruction | Uses the shared tuner; up to 2,048 boundary/interior sampled elements, isolated outputs and factorization-status checks. The conservative reference is independent of FP64-rich candidate priors. |
| Tensor AR local solve (`RAW_TENSOR_LOCAL_SOLVER=auto`, default) | Same-precision cuBLAS, optional MAGMA, cooperative LU and several fused Gauss–Jordan configurations compete. Checks status, finite outputs, solution parity and an independent per-system backward error. Standalone selector samples up to 2,048 actual systems; the tensor adapter builds a prefix of at most 2,048 elements. Cache hits skip that preparation. |
| AR fused/split3/tensor pipeline and contraction engine | `transport/pipeline_tuning.py` measures complete assembly of a private actual-element patch: preparation, local solve, condensation, scatter and requested response retention. Launch/workspace limits screen the production size. FP64 tensor candidates require compatible DG coefficients; pointwise/cross-space inputs retain fused/split3. Explicit inner choices prevent nested calibration. |
| Standalone split3 stage tuner | `transport/tsle_tuning.py` measures build/solve/scatter with private patches from `transport/tuning_sample.py`; checks solve backward error and an independent scatter oracle. Stages share one deadline and eight candidates (default 3/3/2, baselines included). |
| ADR mass placement/inverse and launch shape | `mixed/raw_cuda/tensor_tuning.py` compares compact/external storage, LU/inverse representations, batches and launches on bounded actual elements spanning diffusion classes. Independent mixed-equation and reconstruction checks gate selection. Retained mass tuples carry representation and layout identity. |
| Face-dense, block-Jacobi, Schwarz, BSR and hp-MG smoother kernels | `linalg/gpu/solver_tuning.py` and `legendre_face_bsr.py` compare same-precision raw/vendor/interleaved/coalesced/packed operations on bounded actual coefficients. Automatic hp-MG preserves its transfers, recurrence, precision and stopping rules; graph replay remains explicit. |

The AR local-solve candidate timing includes any layout conversion needed by
its cooperative adapter. Native column-major variants avoid these copies.
This establishes a common contract; it does not imply that a converted
cooperative solve beats a pipeline built natively in row-major form.

Full AR pipeline samples preserve original boundary data and use zero trace on
artificial cut boundaries. They preserve basis, quadrature, stabilization and
coefficient representation. Sampling is capped at 2,048 elements and reduced
further using a 64 MiB scratch estimate. Standalone explicitly selected tensor
`coop` retains its legacy process-local launch tuner; automatic tensor local
solves and full-pipeline trials use the shared bounded protocol instead.

ADR mass tuples remain indexable/unpackable pairs. A tagged compatible cache
keeps its representation; an explicit conflicting representation rebuilds it.
An ambiguous historical high-order kind-6 pair is rebuilt once rather than
interpreted speculatively. This adds a one-time setup cost to those old caches.

Sampling bounds calibration cost but cannot prove a full-mesh performance
winner. The launch/runtime numerical checks remain active. Calibration parity
is not a substitute for manufactured-solution, reconstruction, conservation,
and true-residual qualification in the library and scientific campaigns.

## Controls and reproducibility

```bash
# Defaults:
export HYBRIDGE_CUDA_TUNING=auto
export HYBRIDGE_CUDA_TUNING_SECONDS=30
export HYBRIDGE_CUDA_TUNING_CANDIDATES=8
export HYBRIDGE_CUDA_TUNING_CACHE_DIR="$HOME/.cache/hybridge/cuda-tuning"
```

`CUDA_TUNING=off` uses the adapter's deterministic first candidate without
trials. `refresh` ignores saved measurements; it calibrates on each selection,
so use it for a deliberate refresh run, then restore `auto`. Explicit Poisson
block sizes and `RAW_TENSOR_LOCAL_SOLVER=coop/cublas/magma/fused` preserve their
existing meaning. The Poisson `thorough` tool increases the candidate count
and repeats, retaining the time cap; raise `CUDA_TUNING_SECONDS` for an
exhaustive diagnostic run.

The numerical [qualification campaign](../../campaigns/architecture_tuning/README.md)
records source hashes, environment, candidate caches and correctness results.
Keep calibration JSON with campaign evidence when reproducing a numerical
choice; shared caches are disposable performance data, not scientific inputs.

## Acceptance gates and further performance work

The user requested implementation of all remaining adapters during this
reconciliation. The adapters above are implemented; their extended H200
qualification passed. Quadro and Blackwell qualification remains
explicitly pending, as agreed. The architecture names order candidates;
they do not certify a numerical or performance result on an untested device.

On each architecture run bounded correctness gates, then representative
   sizes/degrees and reuse regimes. Compare default first-use cost, cache-hit
   cost, best qualified override, peak workspace and end-to-end results. Record
   unknown/untested combinations explicitly rather than extending a performance
   claim from architecture names alone.

## Qualification status

An H200 run passed 72 selection/tensor and 7 high-order Poisson checks on
2026-10-31; see the preserved report (`docs/campaigns/architecture_tuning/README.md`, local, untracked).
Extended qualification passed **372 checks**: 38 shared selection, 102
transport, 70 ADR, 152 solver, 7 Poisson and 3 selected-FP32 checks. The runs
used development revisions of 2026-10-31; the ADR run used a revision with
identical ADR numerical code. The preserved evidence (`docs/campaigns/architecture_tuning/README.md`, local, untracked)
includes source hashes, logs, per-stage environment and calibration records.
These bounded correctness results do not establish full-mesh performance
transfer to other architectures.
