# Advection Assembly Baseline: RTX PRO 5000 Blackwell (2026-10-03)

This is the reference measurement for comparing advection-reaction assembly
configurations across GPUs, starting with H100 trials. Every number and every
"fastest" verdict below belongs to the machine and software stack listed here.
The FP32 tables only predict FP64-capable hardware; the H100 runs must confirm
or refute those predictions. Raw rows and environment metadata are in
[`advection_assembly_baseline_2026_10_03.data.json`](advection_assembly_baseline_2026_10_03.data.json).

## Machine and Software

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 5000 Blackwell, sm_120, 110 SMs, 48 GB GDDR7 (384-bit), 96 MiB L2, 99 KiB opt-in shared memory per block, 300 W limit |
| FP64 throughput | 1/64 of FP32; measured DGEMM about 0.74 TFLOP/s |
| Driver / CUDA | NVIDIA 580.126.09; CUDA runtime 13.0.2 (CuPy), NVRTC 13.0, nvcc 13.0.88 |
| Libraries | CuPy 14.2.0 (`cupy-cuda13x`); cuBLAS 13.0.2; cuTENSOR 2.8.1 (`cutensor-cu13` wheel); MAGMA 2.10.0 built from source for sm_120 against the venv MKL 2026.1 |
| Host | Intel Xeon w7-3455 (24 logical CPUs), Linux 6.17, Python 3.12.3, NumPy 2.5.3 |
| Code | `cb3dcb1` plus uncommitted work: the coop LU default, split3 weight folding, tensorized TSLE, and the batched-LU helpers |

## Problem and Method

- **Mesh:** the guiding-center showcase star mesh, `showcase_mesh(0.0095)`,
  with 117,439 triangles (117,443 in the FP32 package mode).
- **Operator:** one BDF2 transport step (`dt = 0.000390625`), with density
  `1 + exp(-20((x-0.3)^2 + y^2))` and velocity
  `(-2 cos 3x sin 2y, 3 sin 3x cos 2y)`.
- **Options:** `positive_transport_options()`, meaning zero-flux walls,
  conflict-averaged upwind, the `legacy-lagrange` trace, the Dubiner basis,
  and face-BSR output.
- **Timing:** CUDA-event assembly time only. Pattern construction, AMGX, and
  reconstruction are excluded. Each value is the median of five timed
  assemblies after one warm-up. Block sizes for split3 and the cooperative LU
  are autotuned per signature.
- **FP32 runs:** native FP32 rows run in the `HYBRIDGE_PRECISION=float32`
  package mode. Tensor FP32 rows run in an FP64 process: FP64 face tables and
  coefficient rows, FP32 contractions, LU, and Schur rows, then FP64 BSR
  accumulation.
- **Configuration classes:**
  - **native:** HYBRIDGE raw kernels only (`fused`, `split3`);
  - **hybrid:** library contractions with the HYBRIDGE cooperative LU;
  - **pure library:** library contractions and library LU, with HYBRIDGE code
    limited to the face-weight and coefficient-row kernels and the BSR
    scatter.

## FP64 Results (production precision on this machine)

Times in ms. The "est." column sums measured split3 build and scatter stages
with the standalone MAGMA solve; that path is not implemented.

| p | fused | split3 | split3 + MAGMA (est.) | tensor + coop, cuTENSOR | tensor + coop, cuBLAS GEMM | tensor + MAGMA, cuTENSOR | Fastest on this machine |
|---|---:|---:|---:|---:|---:|---:|---|
| 4 | 17.0 | 17.0 | 16.7 | 20.3 | 24.1 | 19.4 | native (fused = split3) |
| 5 | 32.7 | 32.3 | 37.7 | 32.3 | 46.3 | 37.3 | native split3 (tensor + coop ties) |
| 6 | 63.0 | 60.8 | 63.8 | 65.2 | 78.9 | 66.8 | native split3 |
| 7 | 118.7 | 114.6 | 130.8 | 115.5 | 140.4 | 129.5 | native split3 |
| 8 | 286.8 | 217.4 | 205.5 | 212.1 | 246.7 | **198.6** | pure library |
| 9 | 699.4 | 388.8 | 321.7 | 366.9 | 415.3 | **298.5** | pure library |

Tensorized workspaces at p=8 and p=9 are 5.2 and 7.1 GiB, against 3.0 and 4.2
GiB for split3. On this machine the FP64 contractions run into the same FP64
limit as the raw kernels. The tensorized gain at p=8 and p=9 comes from
MAGMA's local LU.

## FP32 Results (proxy for FP64-capable GPUs)

Times in ms. The contraction engine and LU are named in parentheses; each
tensorized column shows the best combination for that order.

| p | fused | split3 | tensor + coop (hybrid) | tensor + library LU (pure library) | Fastest on this machine |
|---|---:|---:|---:|---:|---|
| 4 | **2.79** | 3.73 | 5.14 (cuTENSOR) | 3.93 (cuTENSOR + MAGMA) | native fused |
| 5 | **5.83** | 8.07 | 6.43 (cuBLAS) | 8.83 (cuTENSOR + cuBLAS LU) | native fused |
| 6 | 11.5 | 15.8 | **11.1** (cuBLAS) | 12.1 (cuBLAS + MAGMA) | tie: fused, hybrid, library within 9% |
| 7 | 19.3 | 29.5 | **17.3** (cuBLAS) | 24.5 (cuTENSOR + cuBLAS LU) | hybrid |
| 8 | 45.9 | 62.2 | **30.9** (cuBLAS) | 33.8 (cuTENSOR + MAGMA) | hybrid (library within 9%) |
| 9 | 84.5 | 107.6 | 49.7 (cuBLAS) | **46.9** (cuTENSOR + cuBLAS LU) | pure library |

An earlier split3 FP32 run reproduced these split3 times to within 3%.

## Stage 2 Alone (local LU and all-column solve)

The inputs are the FP64 `A_e` and `[B_e | f_e]` built by split3. Times in ms.

| p | coop FP64 | cuBLAS getrs FP64 | cuBLAS inverse FP64 | MAGMA FP64 | coop FP32 | cuBLAS getrs FP32 | cuBLAS inverse FP32 | MAGMA FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 6.6 | 10.4 | 9.4 | 6.3 | 1.4 | 1.7 | 3.8 | 0.9 |
| 5 | 10.6 | 25.2 | 23.8 | 16.0 | 2.5 | 4.2 | 6.9 | 4.8 |
| 6 | 18.3 | 35.5 | 30.9 | 21.3 | 5.2 | 6.3 | 10.0 | 5.7 |
| 7 | 34.9 | 67.9 | 78.9 | 51.1 | 9.5 | 16.1 | 23.1 | 17.5 |
| 8 | 72.5 | 88.2 | 93.3 | 60.2 | 19.4 | 21.2 | 30.4 | 21.2 |
| 9 | 144.1 | 118.0 | 156.4 | 76.5 | 33.9 | 29.3 | 48.4 | 31.8 |

The "inverse" columns use getrf, getri, and a batched GEMM; that approach was
never fastest, so the repository runner omits it. MAGMA `gesv_batched` timed
the same as `getrf` followed by `getrs`.

## Accuracy

- **FP64 tensor vs FP64 split3:** at most 4e-13 for p=4--8 and about 1e-11 at
  p=9 (max-norm, relative to the largest entry). At p=9 the FP64 LU solvers
  already disagree with each other by that much, which indicates local
  condition numbers of roughly 1e3 at p=8 and 1e5 at p=9.
- **FP32 tensor vs FP64 split3, end to end:** relative max-norm error of the
  BSR matrix, ranging over all FP32 configurations:

  | p | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|
  | error | 3e-7 | 1e-6 | 1e-5 | 0.7--1.4e-4 | 0.9--1.5e-4 | 2.5e-3--1.1e-2 |

  The p=9 range depends on the LU: 2.5e-3 with MAGMA and 1.1e-2 with the
  cooperative kernel. RHS and local-response errors are similar. Stage 2 alone
  on FP64-built systems gives 3e-7 to 2e-3. FP32 is a throughput proxy, not a
  production precision.
- **Not tested here:** cuTENSOR `3xTF32` lost accuracy (1e-4) without a
  speedup. cuBLAS `BF16x9` behaved as plain SGEMM under the default emulation
  strategy.

## Machine-Specific Settings

- `RAW_SPLIT3_MIN_ORDER = 8` (`hybridge/solvers/capabilities.py`) comes from the
  FP64 rows above. In FP32 the native ranking reverses: fused beats split3 at
  every order.
- The split3 and cooperative-LU block sizes are autotuned per process and are
  not portable.
- Treating FP32 here as a stand-in for FP64 elsewhere is an assumption. FP64
  doubles the bytes moved. On GPUs with FP64 tensor cores, cuBLAS and cuTENSOR
  FP64 contractions can exceed the proxy, while MAGMA's batched LU runs on the
  ordinary FP64 cores.

## Predictions to Test on H100

1. FP64 fused is at least as fast as split3 at every p. If confirmed,
   `RAW_SPLIT3_MIN_ORDER` must become device-dependent.
2. FP64 fused is fastest for p<=5, the classes tie at p=6, and tensorized
   assembly (hybrid or pure library) is fastest from p=7.
3. Once the build is tensorized, the local LU takes at least half the time.
   The library batched LUs remain slower than the cooperative kernel at p=7--8
   and are at most comparable at p=9. In FP32 here they took 16--18 vs 9.5 ms
   at p=7, 21 vs 19 ms at p=8, and 29--32 vs 34 ms at p=9.
4. The library contractions (stage 1, and stage 3 apart from the scatter) gain
   more than this proxy shows, because they run on FP64 tensor cores.
5. FP64 tensor matches FP64 split3 to at most 1e-12 relative for p<=8.

## Reproducing on H100

1. **cuTENSOR.** Install the wheel that matches CuPy's CUDA major version, for
   example `pip install cutensor-cu13` (or `cutensor-cu12` with CuPy for
   CUDA 12). `hybridge.runtime.optional.require_cutensor` preloads
   `libcutensorMg`.
2. **MAGMA.** Build a release that supports the installed toolkit (2.10 for
   CUDA 13) for sm_90:

   ```bash
   cmake <magma-src> -DMAGMA_ENABLE_CUDA=ON -DGPU_TARGET=sm_90 \
     -DUSE_FORTRAN=OFF -DFORTRAN_CONVENTION=-DADD_ \
     -DLAPACK_LIBRARIES=<path to libmkl_rt.so or libopenblas.so> \
     -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
   make -j magma
   export HYBRIDGE_MAGMA_ROOT=<build directory containing lib/libmagma.so>
   ```

3. **Correctness.** Check with
   `python -m pytest -q tests/test_advection_tsle_tensor.py tests/test_advection_tsle_bsr.py`.
4. **FP64 run:** native paths and every tensor configuration on the same mesh.

   ```bash
   python scripts/gpu/benchmark_advection_tsle_tensor.py --mesh-size 0.0095 \
     --native fused,split3 \
     --configs float64/cutensor/default/coop,float64/cutensor/default/cublas,float64/cutensor/default/magma,float64/cublas/default/coop,float64/cublas/default/cublas,float64/cublas/default/magma,float32/cutensor/default/coop,float32/cutensor/default/cublas,float32/cutensor/default/magma,float32/cublas/default/coop,float32/cublas/default/cublas,float32/cublas/default/magma \
     --output-json run_logs/h100/advection_assembly_fp64_process.json
   ```

5. **FP32 native run** (a separate process):

   ```bash
   HYBRIDGE_PRECISION=float32 python scripts/gpu/benchmark_advection_tsle_tensor.py \
     --mesh-size 0.0095 --native fused,split3 --configs none \
     --output-json run_logs/h100/advection_assembly_fp32_native.json
   ```

Both JSON files record the GPU, driver, CUDA, cuBLAS, cuTENSOR, MAGMA, and git
state. Compare them with the `native` and `tensor` rows of this baseline's
data file. On this machine the baseline rows came from scratch scripts that
used the same solver paths and statistics as the runner. Recapturing them with
the runner first gives a like-for-like diff:

```bash
HYBRIDGE_MAGMA_ROOT=../magma-build-cuda13 .venv/bin/python scripts/gpu/benchmark_advection_tsle_tensor.py <same arguments as step 4>
```
