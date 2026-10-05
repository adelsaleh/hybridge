# Unified ADR L1–L5 campaign

This campaign is intended to rerun the iterative comparisons in
`adr_scaling_2026_09_17/main_synthesis.tex`, not replay their old timings.
PyPardiso is excluded. The runner supports the original `cusparse_generic`
backend and the explicitly approved `legacy` AMGX BSR backend for CUDA-12/V100
runs. Generic BSR remains the default; no automatic AMGX backend substitution
is allowed. CPU-only checks pass, but CUDA-12 builds and numerical compatibility
on AMU remain to be validated by the user. No builds or simulations were run
while preparing this runner.

## Current implementation status

The frozen portable inventory is `run_configs/adr_unified_l5/inventory.json`,
with eight hashed mesh assets (about 14 MB including the JSON). It preserves
107 original systems and 757 distinct per-system iterative configurations.
This intentionally includes the exploratory controls from the earlier detailed
report as well as the synthesis; no historical failures are filtered out.
`export_adr_unified_inventory.py` reproduces it from local archived specifications
without importing numerical packages or reading old matrix caches.

`adr_unified_plan.py` extends each of the 21 coefficient/geometry classes with
L5: 128 systems and 947 solver jobs in the default plan. It retains original
meshes, degree sweeps, and endpoint configurations, and uses a shared 2000-step
cap without silently changing restart or polynomial degree. The unified runner
now dispatches mesh preparation, assembly, and isolated solver jobs. Eight pure
planning tests, five device-inventory tests and seventeen orchestration tests pass.
The orchestration tests include all 947 planned dispatches in each backend mode
using mocks, plus
real tiny JSON-writing subprocesses for resume/crash/timeout behavior. They do
not execute numerical kernels or establish numerical/build compatibility.

### Available entry point

From the HYBRIDGE repository, create a plan without CUDA, meshes or solves:

```sh
.venv/bin/python -B scripts/advection_diffusion_reaction/campaigns/unified/run_adr_unified_campaign.py \
  --output run_outputs/solver_studies/adr_unified_l5 \
  --l5-triangles 400000
```

Only on a compatible CUDA-13.0-Update-1+ GPU with the patched libraries already
built and activated, the full-run command is:

```sh
.venv/bin/python -B scripts/advection_diffusion_reaction/campaigns/unified/run_adr_unified_campaign.py \
  --branch-root vendor/adr_gmres \
  --output run_outputs/solver_studies/adr_unified_l5 \
  --l5-triangles 400000 --maxiter 2000 \
  --assembly-backend auto --numba-threads 24 \
  --memory-fraction 0.8 --reserve-gpu-gib 1 \
  --timeout 7200 --execute
```

Add `--resume` after interruption. Use `--preflight` instead of `--execute` for
an import/configuration/device check. Neither of those commands has been run as
part of preparing this runner. For AMU use the CUDA-12/legacy commands below,
not the generic-BSR command above.

On a compatible allocated GPU, `--preflight` imports dependencies and checks
runtime/library compatibility without solving. `--execute` explicitly enables
mesh generation and numerical jobs. `--resume` preserves terminal results
(including numerical failures and memory exclusions), retries interrupted or
crashed workers, and retains every attempt log. Changed settings, source hashes
or GPU model/memory/library hashes are rejected. Use a fresh output directory
for a different machine or numerical configuration.

The alternate ADR workers and solver package are included in
`vendor/adr_gmres/`; the runner selects that directory by default.
`--branch-root` can select a different compatible snapshot.
The inventory's small mesh assets travel with the main repository. Old
`run_logs` and matrix caches are unnecessary. The runner never installs or
builds dependencies, nor launches PyPardiso or CPU direct reference solves.

`--amgx-backend legacy` rewrites every explicit `bsr_spmv_backend` selector,
including nested preconditioner/smoother scopes, without changing numerical
solver parameters. The archive stays unchanged. Effective solver hashes and
the chosen backend are recorded in the plan, manifest, job specifications,
results and summary. A backend switch requires a fresh output directory.

The backend flag applies to **AMGX BSR**, not every solver's matrix multiply.
pMG-AMG's existing `auto` BSR operator uses its raw-CUDA fallback on CUDA 12;
its scalar coarse problem still uses AMGX. ASM/BJ+PP retain their recorded
implementations. Thus the AMU run is a separately labelled hardware/software
comparison, not an unchanged generic-cuSPARSE comparison.

Assembly `auto` chooses CuPy when the conservative estimate fits, otherwise
Numba on the host; this does not change the common discretization. Both host
and device budgets are checked against live availability before numerical
work, including cgroup limits. Results distinguish memory exclusion, runtime
OOM, timeout, process error and numerical failure. The worker records maximum
process RSS and 1-second whole-device memory samples. These device samples are
not allocator-exact peaks and can include other processes; request an exclusive
GPU allocation for comparisons. Existing worker timing/residual protocols remain
authoritative. Memory estimates do not guarantee absence of transient OOM.

Preliminary conservative L5 planning estimates are roughly 22 GiB of device
memory for some oscillatory AMG configurations and 27 GiB for the strongest
stress configurations. These are estimates, not measured peaks. A 16 GiB V100
may therefore exclude some 400k configurations; query the actual memory SKU.
Do not reduce the common problem size per solver to make those jobs appear to
pass. A 32 GiB allocation still needs live free-memory checks and runtime guards;
the default 80% budget can exclude a 27 GiB estimate even on a 32 GiB GPU.

## Required coverage

- Smooth h/p sweeps: transport, high diffusion, anisotropy.
- Focused variable-velocity/transport/high-diffusion h/p sweeps and original
  million-DOF p=3/p=4 endpoints, including BJ+PP and polynomial controls.
- Oscillatory transport/high diffusion/directional anisotropy on square and
  five-lobed annulus, retaining all original degree-sweep points.
- Original weak-anisotropy and transport controls on the 22,825-triangle annulus,
  plus published tuned directional ASM configurations.
- Nine-lobed trapping/crossing/orthogonal cases at the original approximately
  100k and 150k triangle sizes, plus L5.
- Both pMG-AMG policies and the recorded AMGX outer/preconditioner variants.
  Historical failures must not be filtered out of the new plan.

L5 defaults to approximately **400,000 triangles at p=6**, with an explicit
triangle-target override. Structured squares need a nearby integer grid size;
record actual triangles and trace DOFs. This is four times the previous ~100k
L4, not a relabeling of the existing 150k stress point. Include L5 for every
coefficient/geometry class; endpoint and p-sweep points remain additional tests.
Every candidate on a point uses the identical assembled matrix and RHS. Never
reduce an individual solver's mesh or restart silently to fit VRAM. Preserve
the saved operators for a later identical-system direct-solver comparison.

## Cluster constraints

The supplied AMU inventory has K80, P100 and V100 nodes. Prefer V100 based on
non-tensor FP64 throughput among GPUs actually allocated by the scheduler.
Device memory is per CUDA device, not summed over boards or nodes. K80 is a
dual-GPU board. Query actual memory: do not assume a V100 memory SKU.

`hybridge.backends.device_inventory` provides lazy, runtime-visible enumeration
and estimated FP64 peak ranking. It leaves CUDA_VISIBLE_DEVICES unchanged,
uses allocation-local ordinals, records free memory, and restores the caller's
selected device. Unknown throughput requires an explicit override rather than
silently ranking an unknown GPU last. It does not benchmark or compile kernels.
Peak throughput is not a forecast of sparse-solver speed.

CUDA 13 cannot compile the Pascal/Volta kernels needed for P100/V100. Plan for a
compatible CUDA 12.x environment on those nodes; K80 requires an older stack.
The runner must preflight CuPy, AMGX and both HYBRIDGE source trees, and must not
invoke installation or build scripts. Source paths must be configurable; no
machine-specific home-directory paths or local matrix caches may be required on AMU.

### Confirmed AMGX compatibility constraint

The benchmarked patched AMGX requests `bsr_spmv_backend=cusparse_generic`.
NVIDIA added generic-API BSR SpMV in CUDA 13.0 Update 1, after cuSPARSE dropped
Pascal/Volta support in CUDA 13.0. None of the supplied AMU GPUs can run that
same library path. A newer driver or container does not fix this restriction.

In `AMGX-hdg-cuda13/src/amgx_cusparse.cu`, the generic full-matrix and
diagonal-block paths are gated by `CUDART_VERSION >= 13000`; legacy `bsrmv`
remains outside that gate. A CUDA 12 rebuild may therefore support legacy BSR
on V100, but compilation and numerical compatibility are untested. The approved
mode is labelled **legacy BSR** and explicitly rewrites the effective JSON;
it does not rely on AMGX's implicit fallback for an incompatible generic request.
The existing local build targets SM 120 and cannot be copied as-is to V100.

Preflight rejects generic BSR on CUDA 12, CUDA 13 on P100/V100, K80 with this
patched stack, and mixed loaded CUDA runtime majors. It checks the patched
PyAMGX telemetry API, parses every effective AMGX configuration, and records
loaded library hashes. Import/configuration checks cannot prove compiled kernel
coverage or numerical correctness; those checks require the user's first run.

Evidence: [CUDA 13.0 Update 1 release notes](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html).

References: [CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.2/pdf/CUDA_Toolkit_Release_Notes.pdf),
[CUDA device attributes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__TYPES.html),
[NVIDIA architecture lane counts](https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h).

## AMU V100: user-run build and launch

Use a V100 allocation with at least 24 allocated CPU threads for the commands
below (or lower `--numba-threads` to the allocation). Prefer the larger VRAM SKU
when available. The campaign uses one scheduler-visible GPU at a time, not a
distributed solver. Do not unset the scheduler's `CUDA_VISIBLE_DEVICES`.

Transfer the HYBRIDGE source tree, the patched `AMGX-hdg-cuda13` and
`pyamgx-hdg-cuda13` **sources**, and `run_configs/adr_unified_l5/` with its meshes.
The source-directory suffix does not require building with CUDA 13. Use a
cluster-local Python 3.11+ environment with the study's numerical dependencies,
CuPy built for CUDA 12 (`cupy-cuda12x`, not a co-installed CUDA-13 wheel), NumPy,
Cython and setuptools. Select the site's CUDA-12 toolkit and compatible host
compiler/driver modules first; module and scheduler names are site-specific.
The driver must also support the selected toolkit's JIT path used by CuPy.

Set these absolute paths to the transferred trees and your CUDA-12 module.
Use new build directories; do not overwrite the workstation's CUDA-13 build:

```sh
export HYBRIDGE_SRC=/path/to/hybridge
export HYBRIDGE_GMRES_SRC="$HYBRIDGE_SRC/vendor/adr_gmres"
export HYBRIDGE_CUDA12_ROOT=/path/to/cuda-12.x
export HYBRIDGE_AMGX_SOURCE=/path/to/AMGX-hdg-cuda13
export HYBRIDGE_PYAMGX_SOURCE=/path/to/pyamgx-hdg-cuda13
export HYBRIDGE_AMGX_V100_BUILD=/path/to/AMGX-build-cuda12-v100
export HYBRIDGE_PYAMGX_V100_BUILD=/path/to/pyamgx-build-cuda12-v100
export HYBRIDGE_PYTHON="$HYBRIDGE_SRC/.venv/bin/python"
export HYBRIDGE_BUILD_JOBS=8

cmake -S "$HYBRIDGE_AMGX_SOURCE" -B "$HYBRIDGE_AMGX_V100_BUILD" \
  -DCMAKE_CUDA_COMPILER="$HYBRIDGE_CUDA12_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$HYBRIDGE_CUDA12_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_NO_MPI=ON
cmake --build "$HYBRIDGE_AMGX_V100_BUILD" --target amgxsh \
  --parallel "$HYBRIDGE_BUILD_JOBS"

(
  cd "$HYBRIDGE_PYAMGX_SOURCE" || exit 1
  AMGX_DIR="$HYBRIDGE_AMGX_SOURCE" AMGX_BUILD_DIR="$HYBRIDGE_AMGX_V100_BUILD" \
    "$HYBRIDGE_PYTHON" setup.py build_ext \
    --build-lib "$HYBRIDGE_PYAMGX_V100_BUILD" \
    --build-temp "$HYBRIDGE_PYAMGX_V100_BUILD/objects"
)
```

Those are build instructions for the user, **not commands already executed**.
For a P100 build the target is SM 60; it needs a separate correctly targeted
build (or an explicitly multi-architecture build). K80 is not supported by
this fork's CUDA-12-minimum build.

Activate the V100 libraries in the allocated job environment. Do not source
the workstation's CUDA-13 environment or use `scripts/gpu/run_cuda13.sh`:

```sh
export CUDA_PATH="$HYBRIDGE_CUDA12_ROOT"
export CUDA_HOME="$HYBRIDGE_CUDA12_ROOT"
export PATH="$HYBRIDGE_CUDA12_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HYBRIDGE_AMGX_V100_BUILD:$HYBRIDGE_CUDA12_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$HYBRIDGE_PYAMGX_V100_BUILD${PYTHONPATH:+:$PYTHONPATH}"
cd "$HYBRIDGE_SRC"

"$HYBRIDGE_PYTHON" -B scripts/advection_diffusion_reaction/campaigns/unified/run_adr_unified_campaign.py \
  --branch-root "$HYBRIDGE_GMRES_SRC" \
  --output run_outputs/solver_studies/adr_unified_l5_v100_legacy \
  --amgx-backend legacy --l5-triangles 400000 --preflight
```

After preflight passes, launch:

```sh
"$HYBRIDGE_PYTHON" -B scripts/advection_diffusion_reaction/campaigns/unified/run_adr_unified_campaign.py \
  --branch-root "$HYBRIDGE_GMRES_SRC" \
  --output run_outputs/solver_studies/adr_unified_l5_v100_legacy \
  --amgx-backend legacy --l5-triangles 400000 --maxiter 2000 \
  --assembly-backend auto --numba-threads 24 \
  --memory-fraction 0.8 --reserve-gpu-gib 1 \
  --timeout 7200 --execute
```

Append `--resume` to that exact command after an interruption. Preflight always
gets a fresh attempt; it does not require `--resume` before the first execution.
Planning without a GPU uses the same flags with neither `--preflight` nor
`--execute`. Allocate ample scratch disk: uncompressed per-system operator
caches are retained for later matched direct-solver runs. The default plan's
cache estimate totals approximately 836 GiB, excluding logs and temporary
files; reserve at least 1 TiB and check the site's quota. Each assembly also
checks live free disk before writing its cache.

## Completion gates

1. A portable, frozen case/configuration inventory covering the above report.
2. One plan/execute/resume entry point, with no PyPardiso jobs.
3. Common physical residual and fresh/reused timing protocol, zero guesses,
   deterministic candidate ordering, isolated workers and retained failures.
4. L5 mesh generation without the old 100k hard cap; all actual sizes recorded.
5. Live host/VRAM admission checks plus measured memory diagnostics. Memory
   exclusions, OOM, timeout and numerical nonconvergence must remain distinct.
6. Scheduler-visible GPU ranking and software compatibility preflight.
7. Interrupt-safe resume with immutable settings/code provenance and no reuse
   of stale or incomplete results.
8. CPU-only orchestration tests, full plan coverage audit and exact cluster
   commands. Builds, kernels and simulations remain user-run.
