# GPU HDG Module Notes

This note documents the standalone GPU HDG runners and legacy comparison modules added during the advection-reaction and diffusion-reaction porting work. These scripts are intentionally separate from the preset drivers while the GPU paths are being tuned.

## Environment

Use the project virtual environment and expose the AMGX shared-library
directory before running the GPU benchmarks. Replace `/path/to/amgx/lib` with
the directory containing your AMGX shared library, for example `libamgxsh.so`;
skip the prefix when AMGX is already visible through the system loader:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python <script> <args>
```

The runners disable CuPy memory pools in their main paths so AMGX memory reports are comparable with the legacy scripts.

## Legacy GPU Modules

- `2d/adv_rea_vec_gpu4.py`: solve-based legacy advection-reaction GPU runner. It avoids materializing local inverses and uses batched `cp.linalg.solve` for local trace/source solves.
- `2d/diff_rea_gpu_v2.py`: renamed legacy diffusion GPU v2 baseline, kept for regression and memory-limit comparisons.
- `2d/diff_rea_gpu_v3.py`: renamed legacy diffusion GPU v3 baseline, kept for comparison with v2/v4.
- `2d/diff_rea_gpu_v4.py`: solve-based legacy diffusion-reaction GPU runner. It avoids explicit local inverse construction, improves boundary dof elimination, and is the baseline for the HDGFEM diffusion port.

## Standalone HDGFEM GPU Runners

### `scripts/gpu/run_adv_rea_gpu4_hdg.py`

Standalone CuPy/PyAMGX advection-reaction HDG runner using `hdgfem` mesh, reference-element, quadrature, and basis data. It mirrors the fast legacy `2d/adv_rea_vec_gpu4.py` path while keeping the production preset runner isolated.

Useful baseline command:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python scripts/gpu/run_adv_rea_gpu4_hdg.py \
  -o 6 -ms 0.01 --basis dub_orth --trace-basis legacy-lagrange \
  --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 12
```

Best observed p6/ms0.01 HDGFEM result matched or beat the legacy GPU4 baseline in total measured time while preserving the expected L2 error scale. The safest production trace basis remains `legacy-lagrange`; modal trace support was useful in sweeps but is sensitive to orientation conventions.


### Raw CUDA advection backends

The advection runner has three assembly modes:

- `--assembly-backend cupy`: original HDGFEM/CuPy assembly path.
- `--assembly-backend raw-cuda --raw-local-assembly precomputed`: semi-fused path. CuPy materializes `local_mats`, `element_boundary`, and `source_rhs`; a Raw CUDA cooperative kernel performs local solves, boundary elimination, and reduced COO/RHS emission.
- `--assembly-backend raw-cuda --raw-local-assembly fused`: fully fused path. The Raw CUDA kernel receives projected coefficients, builds local matrices/RHS columns in shared memory, solves locally, emits reduced COO/RHS, and fused reconstruction rebuilds from projected coefficients. This path avoids the large local dense tensors and is the current memory-scaling direction. The fused path supports `--raw-lu-mode safe` by default and opt-in `--raw-lu-mode coop` for the experimental cooperative LU stage; fused raw advection is validated through p <= 8 for `legacy-lagrange` and `legendre-modal` trace bases.

Current safe-default fused p6/ms0.01 working command:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  HDGFEM_GPU4_AMGX_MONITOR=0 \
  .venv/bin/python scripts/gpu/run_adv_rea_gpu4_hdg.py \
  -o 6 -ms 0.01 -mt rectangle --basis dub_orth --trace-basis legacy-lagrange \
  --assembly-backend raw-cuda --raw-local-assembly fused --raw-block-size 32 \
  --trace-ordering none --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 8
```

Matched semi-fused comparison command:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  HDGFEM_GPU4_AMGX_MONITOR=0 \
  .venv/bin/python scripts/gpu/run_adv_rea_gpu4_hdg.py \
  -o 6 -ms 0.01 -mt rectangle --basis dub_orth --trace-basis legacy-lagrange \
  --assembly-backend raw-cuda --raw-local-assembly precomputed --raw-block-size 32 \
  --trace-ordering none --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 8
```

Earlier warm p6/ms0.01 end-to-end comparison before the p<=8 cooperative-LU update:

| backend | local assembly | assembly | raw kernel | AMGX solve | reconstruction | L2 error | total measured |
|---|---|---:|---:|---:|---:|---:|---:|
| raw CUDA | fused | 0.413 s | 0.227 s | 1.041 s | 0.185 s | 2.720e-12 | 7.196 s |
| raw CUDA | precomputed | 0.464 s | 0.074 s | 1.025 s | 0.153 s | 1.217e-12 | 7.098 s |
| legacy GPU4 remembered | legacy | 0.564 s | n/a | 1.036 s | 0.241 s | 2.177e-12 | 8.240 s |

Large fused Raw CUDA limit probes on the Quadro RTX 6000 showed that assembly is no longer the first memory limit. The p6/ms0.004 case assembled 578,270 triangles in 1.755 s, then failed in CuPy COO-to-CSR conversion due to a transient allocation. Practical full-solve working points were:

| p | mesh size | triangles | trace dofs | nnz | assembly | AMGX solve | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 0.006 | 258,002 | 2.704M | 94.5M | 0.866 s | 4.746 s | robust |
| 6 | 0.005 | 369,790 | 3.877M | 135.5M | 1.160 s | 8.233 s | fits, solve accuracy degraded |
| 5 | 0.004 | 578,270 | 5.198M | 155.8M | 1.986 s | 11.790 s | robust; precomputed OOMs before solve |
| 4 | 0.004 | 578,270 | 4.332M | 108.2M | 0.680 s | 8.657 s | robust |

The fused p5/ms0.004 full solve succeeds where the precomputed raw path OOMs during COO-to-CSR conversion because the precomputed path still holds materialized local dense tensors. p4/ms0.0035 fits memory but AMGX returned `nan`, so solver robustness becomes a limit before raw assembly memory there.


### Fused Raw Cooperative LU and p <= 8 Status

The fused advection kernel now has a documented cooperative LU option selected
with `--raw-lu-mode coop`. The default remains `safe`; `coop` is available only
for `--raw-local-assembly fused` and is rejected for the precomputed raw path.

Implementation summary:

- shared coefficient caches hold source, beta_x, beta_y, transformed beta_ref0,
  transformed beta_ref1, and optional reaction coefficients per element;
- volume advection uses transformed beta coefficients to avoid repeated affine
  inverse multiplies in the matrix-entry loop;
- Schur/RHS emission computes each oriented lift row once and reuses it for the
  source RHS and all trace-column dot products;
- cooperative LU uses a shared-memory pivot reduction plus parallel multiplier
  scaling and trailing Schur updates;
- row swaps remain serialized on thread 0 because fully parallel row-swap
  variants reproduced illegal-address failures in the fused kernel.

Warmed safe-vs-coop raw fused kernel comparison at `ms=0.03`, block size 32:

| p | safe kernel | coop kernel | speedup |
|---:|---:|---:|---:|
| 3 | 0.002964 s | 0.002478 s | 1.20x |
| 4 | 0.006965 s | 0.005574 s | 1.25x |
| 5 | 0.013598 s | 0.009422 s | 1.44x |
| 6 | 0.023749 s | 0.016519 s | 1.44x |
| 7 | 0.041272 s | 0.032567 s | 1.27x |
| 8 | 0.083809 s | 0.066202 s | 1.27x |

CuPy-vs-raw fused cooperative comparison at `ms=0.03`:

| p | CuPy assembly | raw fused coop assembly | assembly speedup | CuPy total | raw total |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.3749 s | 0.1346 s | 2.79x | 2.368 s | 2.117 s |
| 4 | 0.3451 s | 0.1386 s | 2.49x | 2.309 s | 2.078 s |
| 5 | 0.3466 s | 0.1475 s | 2.35x | 2.272 s | 2.144 s |
| 6 | 0.3810 s | 0.1688 s | 2.26x | 2.405 s | 2.190 s |
| 7 | 0.4156 s | 0.2231 s | 1.86x | 2.544 s | 2.304 s |
| 8 | 0.4755 s | 0.3596 s | 1.32x | 2.834 s | 2.651 s |

Larger p8 checks on the 24 GB Quadro RTX 6000:

| p | mesh size | trace dofs | nnz | raw assembly | AMGX solve | total | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 0.010 | 1.246M | 55.9M | 0.945 s | 1.567 s | 8.910 s | fits |
| 8 | 0.008 | 1.949M | 87.5M | 1.319 s | 3.154 s | 13.924 s | fits |
| 8 | 0.006 | 3.477M | 156.2M | 2.159 s | 7.331 s | 25.242 s | fits |
| 8 | 0.005 | about 4.98M | about 224M | 2.982 s | n/a | n/a | COO-to-CSR OOM |

The local fused assembly kernel is no longer the dominant p7/p8 cost. The next
high-impact target is global COO-to-CSR construction, memory pressure before
AMGX upload, and AMGX solve behavior on very large trace systems.

Modal trace compatibility was added after the p<=8 cooperative-LU update. The
raw CUDA orientation rule now distinguishes nodal reversal from modal parity:
negative legacy-lagrange columns reverse dof order, while negative Legendre-modal
columns keep the mode index and multiply by `(-1)^j`. Assembly comparisons
against the CuPy modal reference matched exact COO rows/columns with max
matrix/RHS differences at roundoff for p=3,4,6,8 on `ms=0.2` and for p=6 on
`ms=0.03`. Raw fused reconstruction also matched CuPy modal reconstruction to
about 1e-14 for p=3,6,8.

Representative modal full solves with fused raw cooperative LU:

| p | mesh size | triangles | trace dofs | nnz | raw assembly | AMGX solve | L2 error | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.05 | 3,704 | 21,904 | 432,960 | 0.150 s | 0.023 s | 1.972e-02 | fits |
| 6 | 0.05 | 3,704 | 38,332 | 1.326M | 0.155 s | 0.034 s | 3.027e-05 | fits |
| 8 | 0.05 | 3,704 | 49,284 | 2.192M | 0.287 s | 0.038 s | 3.132e-07 | fits |
| 8 | 0.03 | 10,472 | 140,166 | 6.264M | 0.350 s | 0.094 s | 3.346e-09 | fits |

### `scripts/gpu/sweep_adv_rea_gpu4_hdg.py`

Subprocess sweep driver for `run_adv_rea_gpu4_hdg.py`. It resets CuPy/AMGX state between cases, writes JSON/CSV logs under `run_logs`, and is the reference for basis/quadrature sweep methodology.

Example:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python scripts/gpu/sweep_adv_rea_gpu4_hdg.py --quick
```

### `scripts/gpu/run_diff_rea_gpu4_hdg.py`

Standalone CuPy/PyAMGX diffusion-reaction HDG runner using `hdgfem` core data. It mirrors the solve-based legacy `2d/diff_rea_gpu_v4.py` path and currently supports the identity-diffusion test cases used in the legacy benchmark suite.

Useful baseline command:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python scripts/gpu/run_diff_rea_gpu4_hdg.py \
  -o 6 -ms 0.05 --basis dub_orth --trace-basis legacy-lagrange \
  --volume-quad-1d 12 --error-volume-quad-1d 24 -pr 12
```

Current best robust diffusion configuration:

```bash
--basis dub_orth --trace-basis legacy-lagrange --volume-quad-1d 12
```

The `dub_orth + legacy-lagrange` p6/ms0.05 run matches the legacy v4 numerical error and solve speed while using the modern reference-element and basis infrastructure. A q14 quadrature rule was sometimes marginally faster in isolated sweeps, but q12 (`2p`) is the conservative default because the difference is within benchmark noise and it is consistent with the HDGFEM quadrature default.

## Modal Trace Status for Diffusion

The diffusion runner now orients trace-column couplings directly through the active `DGTraceSpace` tables.  NumPy, CuPy, Numba, and raw-CUDA COO/CSR assembly are validated against the same modal trace operator, and raw-CUDA reconstruction supports `legendre-modal` for the current p <= 6 identity-diffusion/zero-reaction path.  The raw-CUDA path still falls back for p > 6 and does not enable Bernstein traces yet.

Low-order runner checks show the modal trace algebra is consistent with AMGX handoff:

- p2, structured rectangle, `dub_orth + legendre-modal`, CuPy COO-to-CSR: correct error.
- p2, structured rectangle, `dub_orth + legendre-modal`, raw-CUDA direct CSR with cooperative local solve: correct error.

Large p6 modal AMGX robustness should still be benchmarked separately before treating modal traces as the preferred production setting.

## Defaults

The core HDGFEM defaults were updated to use the Dubiner-style orthogonal element basis and the `2p` volume quadrature rule. These defaults are the best average choice from the advection and diffusion sweeps so far:

- Element basis: `dub_orth`
- Volume quadrature: `max(2*p, 2)`
- Diffusion trace basis: `legacy-lagrange`

## Benchmark Logs

Relevant summaries and raw sweep outputs are stored under `run_logs`, including:

- `run_logs/adv_rea_gpu4_hdg_findings_20260718.md`
- `run_logs/diff_rea_gpu4_hdg_modal_trace_findings_20260718.md`
- `run_logs/raw_cuda_hdg_findings_20260719.md`
- `run_logs/raw_cuda_fused_coop_lu_findings_20260720.md`
- `run_logs/diff_rea_gpu4_hdg_nodal_trace_quad_basis_sweep_20260718_163432.json`
- `run_logs/diff_rea_gpu4_hdg_nodal_trace_quad_basis_sweep_20260718_163432.txt`
