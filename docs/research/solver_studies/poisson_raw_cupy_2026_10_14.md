# Poisson raw CUDA and CuPy comparison on 14 October 2026

Raw CUDA CSR assembly is **1.40–3.95× faster** than CuPy Schur-Cholesky on
100,000 triangles at the tested orders p=2,4,6,7,8,9. The matched lower-order
follow-up gives **2.91×, 3.95×, and 2.25×** at p=2,4,6, compared with
**2.41×, 1.75×, and 1.40×** at p=7,8,9. The advantage is generally larger
at lower orders, but it is not strictly monotonic: p=4 leads, and p=7 has
a slightly larger ratio than p=6. At p=9, raw still saves 28.4% of assembly
time. CSR and face BSR have nearly identical assembly times.

The advantage also depends on the phase. Raw cached RHS refresh is slower
at p=2 and p=4. Reusing the raw Cholesky trace response gives a 5.5–7.0×
reconstruction gain at p=7–9, but only 1.08–1.16× at p=2,4.

The [recorded samples](poisson_raw_cupy_2026_10_14.data.json) preserve all
three measured repetitions, device metadata, selected launch configurations,
cache sizes, and numerical comparisons from both runs. These are stationary assembly and
reconstruction measurements; they do not measure a global solve or change
solver defaults.

## Problem and measurement

- Mesh: `rectangle_mesh(250, 200)` on `[-1,1]²`, exactly 100,000 affine
  triangles, with the same geometry at each order.
- Discretization: FP64, `dub_orth` elements, `legendre-modal` traces,
  identity diffusion, zero reaction, scalar stabilization 1.3, source 1,
  and zero Dirichlet data. Reduced trace sizes are 448,650 / 747,750 /
  1,046,850 / 1,196,400 / 1,345,950 / 1,495,500 at p=2 / 4 / 6 / 7 / 8 / 9.
- Device: NVIDIA RTX PRO 5000 Blackwell, SM 12.0, 110 SMs; CuPy 14.2.0,
  CUDA runtime 13.2 (`13020`), driver API `13000`.
- Reservation: existing CUDA compute processes finished before the run;
  `.cache/gpu-campaign.lock` was held for the campaign. Clocks were not locked.
- Timing: one warmup followed by three repetitions, reported as medians
  of synchronized wall time. Mesh/space creation, first-use compilation,
  and automatic tuning are excluded. Each timed assembly rebuilds matrix
  values, RHS, and requested local caches; prepared topology/reference data
  can be reused. CuPy runs first at each order, followed by the raw variants.

The p=2,4,6 follow-up used the same device, software versions, problem,
timing procedure, and five raw variants as the earlier p=7–9 run. These
were separate reservations on the same day, not an interleaved sweep.
The generated Poisson kernel-source hash is identical in both records.
Only the follow-up captured a complete source archive and verified that
none of its files changed during execution.

The CuPy baseline is the existing arbitrary-order assembly wrapper with
`use_schur_cholesky=True`, including its COO-to-CSR construction. The raw
wrapper uses `block_size="auto"` and emits the requested sparse format
directly. Thus the assembly table includes wrapper preparation and sparse
emission, not just the isolated local kernel. COO conversion to CSR is
excluded from the raw COO timing. No NumPy assembly is used.

RHS refresh retains the operator and repeats the same constant source and
zero boundary data. Reconstruction uses the prescribed full trace
`sin(0.273*j)` and returns all three mixed components `(u, q_x, q_y)`.
CuPy gathers element traces and calls its scalar-Schur/cuBLAS solve; raw
calls the matching reconstruction kernel and also copies out the scalar
field. This measures local recovery independently of a global solve.

## Assembly with Schur-Cholesky caching

All times are milliseconds. Each path creates its own Cholesky cache during
assembly; the retained data differ because raw also stores a scalar trace
response and source solution for subsequent recovery.

| p | CuPy CSR | Raw COO | Raw CSR | Raw BSR | CuPy / raw CSR |
|---:|---:|---:|---:|---:|---:|
| 2 | 33.309 | 21.392 | 11.439 | 11.475 | 2.91× |
| 4 | 130.921 | 46.000 | 33.110 | 33.007 | 3.95× |
| 6 | 293.003 | 140.302 | 130.255 | 129.742 | 2.25× |
| 7 | 644.378 | 273.350 | 267.323 | 267.067 | 2.41× |
| 8 | 829.141 | 489.953 | 472.677 | 471.614 | 1.75× |
| 9 | 1106.391 | 802.956 | 792.125 | 792.022 | 1.40× |

CSR and BSR differ by at most 0.40% in these medians. Their assembly times
alone give no useful reason to prefer one storage format. The raw local
algebra is shared across formats; the global solver's format requirements
remain relevant to that choice.

## RHS refresh and reconstruction

These measurements use the CSR assemblies and retained Schur-Cholesky
caches. Times are milliseconds.

| p | CuPy RHS | Raw RHS | CuPy reconstruction | Raw reconstruction |
|---:|---:|---:|---:|---:|
| 2 | 4.090 | 4.818 | 3.617 | 3.336 |
| 4 | 5.001 | 7.665 | 4.751 | 4.086 |
| 6 | 32.948 | 13.342 | 32.844 | 4.849 |
| 7 | 44.294 | 20.375 | 52.926 | 7.545 |
| 8 | 45.160 | 28.886 | 45.155 | 8.214 |
| 9 | 60.546 | 53.527 | 60.839 | 9.158 |

At p=9, raw cached RHS refresh is only 1.13× faster, while reconstruction
is 6.64× faster. The reconstruction advantage includes reuse of the
precomputed scalar trace response and source solution, so it is a comparison
of complete cache strategies as well as kernel execution.

At p=2 and p=4, raw RHS refresh takes 1.18× and 1.53× the CuPy time,
respectively. The assembly speedup therefore does not imply that every
phase improves; the measured refresh and reconstruction costs should be
considered separately for repeated-source workloads.

The p=9 raw CSR cache-policy comparison makes that tradeoff explicit:

| Raw cache policy | Assembly ms | RHS refresh ms | Reconstruction ms | Retained local cache GB |
|---|---:|---:|---:|---:|
| none | 756.235 | 111.809 | 107.938 | 0 |
| schur-lu | 755.922 | 61.402 | 58.331 | 2.442 |
| schur-cholesky | 792.125 | 53.527 | 9.158 | 3.806 |

GB means decimal bytes divided by 1e9. The counts include raw factors and
pivots, plus the response and source solution for Cholesky; they exclude
the global matrix, mesh, source arrays, and allocator reserves. They are
retained cache sizes, not peak device memory. All-order cache-policy samples
are in the JSON.

## Numerical comparison

All three raw formats and all tested cache policies agree with the CuPy
reference. The maximum relative Euclidean differences across both runs are:

| Quantity | Maximum relative difference |
|---|---:|
| Matrix action on `sin(0.731*j)` | 6.363e-13 |
| Assembled or refreshed RHS | 3.578e-14 |
| Reconstructed mixed field | 1.736e-13 |

For the lower-order follow-up alone, the maxima are 4.837e-14 for matrix
action, 1.873e-14 for RHS, and 3.169e-14 for mixed reconstruction.

The large-mesh matrix check compares one prescribed matrix-vector product,
not every matrix entry. Reconstruction and RHS refresh were checked for
CSR with each raw cache policy; COO and BSR were checked for assembly with
Cholesky caching. The broader small-matrix, trace-basis, boundary-update,
and sanitizer qualification is recorded in the
[raw CUDA backend notes](../../backends/raw_cuda.md#poisson-qualification-2026-10-14).

## Relation to earlier results

The [20 July advection-reaction comparison](raw_cuda_fused_coop_lu_2026_07_20.md#fused-raw-cooperative-vs-vectorized-cupy)
already showed a declining raw-over-CuPy assembly advantage on 10,472
triangles: 2.79× / 2.49× / 2.35× / 2.26× / 1.86× / 1.32× for p=3 through 8.
Today's Poisson measurements support the same broad tendency, with the
nonmonotonic order dependence stated above, while retaining a useful
advantage at p=9. The equations, mesh, implementation,
hardware, and CuPy local-solver baseline differ, so the two studies do not
form a single order sweep. The matched p=2,4,6 follow-up now quantifies the
current Poisson advantage at those lower orders directly.

The same-day [Poisson kernel qualification](../../backends/raw_cuda.md#poisson-qualification-2026-10-14)
reported 4.1–6.7× faster assembly at p=7–9 against the **previous raw layout**
on 2,048 triangles, using CUDA-event kernel medians without factor caching.
That is a different baseline and timing scope from the 1.4–2.4× gains over
CuPy reported here; both results remain applicable to their stated workloads.

Increasing local work provides a plausible explanation for the shrinking
relative gain. A triangular scalar space has `(p+1)(p+2)/2` DOFs, and dense
factorization scales cubically in that dimension. The increase from 36 to
55 DOFs between p=7 and p=9 implies about 3.6× the factorization work.
Launch and intermediate-array overhead can become a smaller fraction of
CuPy time, while shared-memory demand and synchronization can limit the raw
schedule. This quick timing run does not isolate those hardware costs.

## Evidence and reproduction

The adjacent JSON preserves the original p=7–9 extract at the top level
and the p=2,4,6 follow-up under `matched_low_orders`. Each retains every
timing sample, numerical result, selected configuration and workload key;
the full local artifacts also contain tuning-candidate details.

The original artifacts are `artifacts/poisson_100k_comparison.{json,py,log}`.
Their extract records hashes of the JSON and driver, but no complete Git
revision or dirty-source snapshot was captured at execution. That limitation
applies only to the historical p=7–9 run.

The follow-up artifacts are `artifacts/poisson_100k_low_orders.json`, `.log`,
and `.source.tar.gz`. Its manifest records Git HEAD and dirty-code digest,
package versions, per-file SHA-256 hashes, and the archive hash. The archive
contains all package Python sources, the maintained runner, its provenance
helper and `pyproject.toml`, including untracked source files. No archived
source changed during the run. The artifacts themselves are Git-ignored;
the manifest and compact samples are retained in this study's JSON.

The maintained [comparison runner](../../../scripts/gpu/benchmark_poisson_assembly.py)
reproduces the measurement with configurable orders and rectangle size:

```bash
.venv/bin/python scripts/gpu/benchmark_poisson_assembly.py \
  --orders 2,4,6 --nx 250 --ny 200 --warmups 1 --repeats 3 \
  --output artifacts/poisson_100k_low_orders.json
```

Use `--orders 7,8,9` to repeat the high-order cases, or `--dry-run` to inspect
the configuration without device work. The runner acquires
`.cache/gpu-campaign.lock` itself, then waits for existing compute processes
before initializing CUDA; do not wrap it in another acquisition of that lock.
`--gpu-lock` selects a shared reservation file and `--wait-timeout` bounds
the wait. Each run writes samples incrementally, a flushed log, and an exact
source archive. The default verbosity is 1; levels 2 and 3 add phase starts
and individual samples. It performs no global solve or NumPy assembly.
