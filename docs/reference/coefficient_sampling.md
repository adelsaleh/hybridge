# GPU-first bundled coefficient sampling

## Integration status

The reusable package modules and non-compiling tests are present. The optional
velocity-sampling hooks in the branch assembler are applied. The remaining
runner/ADR integration is supplied as
`patches/adr_gpu_first_coefficient_sampling_20260922.patch`, **not yet applied**.
Existing-file edits were blocked by a patch-tool filesystem failure; a whole-file
fallback was rejected by automated review. Only the specifically user-approved
advection-hook edit was performed. The active campaign still uses its original
coefficient callbacks and settings.

From the directory that holds both checkouts (the parent of the repository
root), inspect and apply the remaining small patch:

```bash
cd ..
git apply --check hybridge/patches/adr_gpu_first_coefficient_sampling_20260922.patch
git apply hybridge/patches/adr_gpu_first_coefficient_sampling_20260922.patch
```

No build, JIT compilation, CUDA numerical execution, or campaign was launched
while preparing this implementation. Real compiled CPU/GPU parity and performance
remain to be established by the user.

## Existing functionality and scope

The package already provides device mesh/reference mirrors in `core/device.py`,
chunked callable projection through `CupyDGSpace.project_callable`, backend-aware
profiles in `cases/profiles.py`, and Numba element-parallel integration kernels.
These do not provide bundled analytic coefficient sampling with an OOM-aware CPU
fallback. Projection is deliberately not substituted for direct sampling: that
would change the stress operator and source.

`hybridge.hdg.coefficient_sampling.CoefficientSampler` fills this narrower gap.
It does not replace matrix integration, inversion, condensation, or solver kernels.
The GPU performs the coefficient expressions; sampled outputs return to host
because the present ADR preparation consumes NumPy arrays. This avoids retaining
a second full coefficient set on the device. Host output memory is still required;
GPU batching is not a host-memory or full-assembly memory-budget guarantee.

## Reusable callable contract

```python
from hybridge.hdg.coefficient_sampling import CoefficientSampler

def coefficients(x, y, parameters):
    amplitude, = parameters
    return amplitude*x + y, x*y

sampler = CoefficientSampler(backend="auto", device=0)
values = sampler.sample(coefficients, x, y, (2.0,), components=2)
```

- `x` and `y` are broadcastable host coordinates. Rank-two arrays are interpreted
  as `(elements, quadrature_points)`; batching preserves whole elements.
- The callable returns a fixed-length tuple. Outputs must broadcast to the input
  shape. Results have shape `(components, *broadcast_shape)` and dtype FP64.
- CuPy-compatible functions must avoid host coercion such as `np.asarray` on
  device arrays. Arithmetic and supported NumPy ufuncs dispatch to CuPy. A
  NumPy-only arbitrary callable cannot automatically become GPU-compatible.
- For Numba, the same callable must accept scalars and be nopython-compilable;
  helper functions must be jitable. Parameters should be numeric tuples/arrays,
  not Python configuration objects. Unsupported functions raise rather than
  silently entering object mode or a slow Python fallback.
- Numba uses `prange` over elements, evaluates quadrature points within each
  element, and disables fastmath. `NUMBA_NUM_THREADS` controls this pool; inner
  BLAS should remain single-threaded when called from element-parallel kernels.
- `numpy` is an explicit independent/reference backend, not the automatic CPU
  fallback. `auto` tries CuPy, then Numba; `cupy` requires a usable GPU.

## Memory management and failures

The default target is 131,072 points per batch, a 512 MiB device reserve, and 50%
of remaining available memory. An estimated 256 scratch arrays per point bounds
the starting batch conservatively; callers can adjust `scratch_arrays` for other
expressions. This estimate is not an exact allocator guarantee.

A private CuPy memory pool isolates the sampler from other device caches. An
allocation failure reduces the batch size and retries the same rows. At one
element, `auto` switches to the compiled CPU sampler; forced `cupy` reports the
failure. Invalid device selection, coefficient exceptions, component mismatches,
and nonfinite outputs remain errors. The sampler never purges another component's
device pool or changes the mathematical expressions to recover from memory pressure.

The `stats` dictionary records backend request, GPU/CPU batches, OOM retries,
fallback reason, point counts, CPU compilation time, evaluation time, transfers,
and actual Numba thread configuration when that backend is used. CPU compilation
is reported separately from evaluation but is included in the first enclosing
assembly wall time. First-use CuPy kernel compilation may likewise affect the
first GPU evaluation. Use warm repetitions for performance comparisons.

## Stress-case adapter

`hybridge.cases.closed_loop_coefficients` supplies reusable scalar/array formulas for
the corrugated-annulus family. Volume evaluation returns the symmetric tensor's
three distinct entries, both velocity components, and the manufactured source
together, sharing one geometry evaluation. Faces request only velocity. The old
NumPy formulas remain an independent correctness reference.

The script adapter loads these array-only modules by path because the campaign
isolates master's native solver from the other worktree's ADR assembler. It does
not import master's full numerical package into a branch worker.

After applying the integration patch, add these flags to a **new output directory**:

```text
--coefficient-backend auto --coefficient-chunk-points 131072
--coefficient-memory-fraction 0.5 --assembly-backend numba --numba-threads 24
```

`legacy` remains the runner default to preserve existing specifications. The
new options enter specifications/manifests; new package files also change source
fingerprints, so do not resume an old campaign across this code change. The small
assembly reference path removes the accelerated sampler and compares against
the original NumPy callbacks, not against a second use of the same sampler.

Source evaluation moves into the bundled coefficient stage; the later source
stage measures integration/embedding of sampled values. Compare total assembly
time, not the old source-stage label alone. Boundary/exact-solution callbacks and
post-solve error evaluation are not accelerated by this change.

## Validation commands

Non-compiling checks (safe during development, no actual GPU work):

```bash
# from the repository root
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m pytest -q tests/test_coefficient_sampling.py
```

Preparation checks: 9 sampler tests passed (6 compiled CPU/GPU cases skipped).
Eight tiny branch adapter/local-matrix tests also passed, with one pending-integration
test skipped. The branch has a pre-existing no-JIT import issue: `basis.py` accesses
`.py_func` unconditionally. The diagnostic run used a process-local decorator shim
that attached `.py_func` to interpreted functions; no production files were changed
for that workaround. An ordinary branch test run with JIT disabled may therefore
fail during collection. After the user authorizes compilation, run the branch
hook suite normally to validate the integrated path.

The following commands explicitly compile/execute small coefficient kernels;
they are provided for the user to run after the campaign, not executed by the agent:

```bash
# from the repository root
NUMBA_DISABLE_JIT=0 NUMBA_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  HYBRIDGE_TEST_SAMPLING_JIT=1 .venv/bin/python -B -m pytest -q \
  tests/test_coefficient_sampling.py -k compiled_numba_parity

HYBRIDGE_CUDA13_ROOT=/path/to/cuda-13 HYBRIDGE_TEST_SAMPLING_CUDA=1 \
  scripts/gpu/run_cuda13.sh .venv/bin/python -B -m pytest -q \
  tests/test_coefficient_sampling.py -k cupy_parity
```

Only after these pass should a new campaign compare legacy sampling, explicit
Numba sampling, and GPU-first sampling with unchanged meshes, quadrature, and
solver parameters. No speedup or all-core utilization has yet been measured.
