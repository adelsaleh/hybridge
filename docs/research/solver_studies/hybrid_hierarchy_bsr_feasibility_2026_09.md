# Hybrid hierarchy executed in BSR: feasibility experiment

Completed on 2026-09-14 after the user rebuilt AMGX, using the NVIDIA RTX PRO
5000 Blackwell (48 GB), driver 580.126.09, and the existing CUDA 13 stack.
Do not implement an entirely BSR coarse hierarchy on this evidence. The best
of the eight tested combinations was RCM with block size 2 on both meshes:

| Triangles | Normal solve ms | Projected solve ms | Projected change | Coarse A/P/R storage | Conversion + permutation s |
|---|---:|---:|---:|---:|---:|
| 157,280 | 119.197 | 138.249 | +15.98% | 1.421× CSR | 2.399 |
| 315,425 | 227.027 | 273.205 | +20.34% | 1.421× CSR | 5.017 |

These are projections, holding unaffected solver work constant. Even the best
full-BSR candidates lose beyond the observed timing ranges. Normal timing is
the mean of three per-RHS medians, with five repeats each. The warm native
product profiles attribute 44.163 ms (37.05%) and 83.232 ms (36.66%) respectively
to the coarse operators and transfers that could change. Total retained
operator storage (including unchanged fine BSR A) grows from 0.928 to 1.136 GiB
and from 1.865 to 2.282 GiB. The best candidates add approximately 40.0 million
and 80.3 million zero entries; dimension padding is only 82 and 172 entries.

Some original-order block-2 operators do benefit individually: L1.P/L1.R/L2.A/
L2.P on the smaller mesh, and L1.P/L1.R/L2.P/L2.R/L3.P on the larger mesh.
Changing only those operators projects savings of 0.868 ms (0.73%) and 1.003 ms
(0.44%). Their conservative saving intervals are positive, but generic CSR on
the same operators projects larger savings of 2.709 ms and 1.541 ms. These small
selective results warrant separate confirmation if pursued; they do not
establish an advantage from BSR storage itself or an integrated solver speedup.

The numerical hierarchy, coarse-point selection and smoothing are unchanged.
Only offline copies are reordered and packed into BSR. The fine level retains
its original ordering and BSR operator. Each mesh reproduced the saved hybrid
iteration sequence 16/13/12 and passed physical-residual/reference checks.

Full per-mesh comparison tables (`artifacts/hybrid_hierarchy_bsr_feasibility_20260914/report.md`, local, untracked)
and machine-readable summary/provenance (`artifacts/hybrid_hierarchy_bsr_feasibility_20260914/summary.json`, local, untracked)
include every block size, ordering, storage component, timing sample and
projection. Raw exports, failed-path diagnostics and cold profiling passes are
retained beside the final results for audit.

The user ran this one AMGX build; no PyAMGX rebuild was needed. For reproduction:

```bash
cmake --build /home/adelsaleh/src/AMGX-build-cuda13 --target amgxsh --parallel 2
```

After the build, run the saved-system experiment from the existing environment:

```bash
source /home/adelsaleh/src/hybridge/.env
cd /home/adelsaleh/src/hybridge
scripts/gpu/run_cuda13.sh .venv/bin/python -m scripts.guiding_center.poisson.benchmark_hybrid_hierarchy_bsr \
  --output-dir artifacts/hybrid_hierarchy_bsr_feasibility_20260914
```

The runner never builds or compiles kernels. Cache misses fail explicitly. It
uses exactly the p=6 captures for 157,280 and 315,425 triangles in
`full_bsr_convergence_20260914/p6_150k_screen1` and `p6_300k_controls`.
Each has three saved systems; no assembly or time integration is involved.

Normal replay has one warmup round followed by five rounds of the same three
RHS/initial guesses. A separate diagnostic setup exports the hierarchy, then
warms each system once before replaying it with product profiling. Iteration count, residual
history, hashes, physical residual and saved-reference parity are checked.
Diagnostic solve time is excluded from the normal baseline.

`--phase baseline` runs only the normal replay and works without the new export
hook. `--phase measure` resumes from an existing normal replay. Completed
replays can be reused; incomplete replays require inspection and a fresh output
directory. Product measurement is rerun as a unit after interruption.

The native setting `classical_hierarchy_export_prefix` is a string in the AMG
preconditioner scope, empty by default. One setup owns one prefix. The existing
parent directory must be writable. AMGX limits configuration strings to 64
characters; the replay runner uses a temporary short symlink for long paths. Each transfer level L writes `L.P`, `L.R`,
and `(L+1).A`, as JSON sidecars plus native-endian binary `indptr`, `indices`,
`values`, and `diag` arrays. Sidecars describe FP/index types, block dimensions,
block order, diagonal placement, array counts and level identities. Original
arrays include explicit zeros and allocated value tails. Readers use only the
logical matrix entries, preserve external diagonals and reject malformed data.
An existing sidecar is an error, preventing silent hierarchy replacement.

`AMGX_CLASSICAL_PROFILE_PATH` enables synchronous CUDA-event records for tagged
`multiply()` actions during a diagnostic solve. The replay runner scopes this
environment variable to individual solves. Normal replay leaves it unset.
This measures actual call counts and event envelopes, including host submission
gaps. Fused smoother kernels, fine-level BSR work, vector/reduction work, and
coarse factorization/solve remain unchanged in the projection.

Every exported matrix is processed sequentially for block sizes 2, 4, 7 and 8
in original ordering and RCM ordering. RCM is constructed from each coarse A's
symmetrized structural graph. Adjacent P/R use the matching row/column
permutations. No GPU permutation is included in a timed product: the projection
assumes an integrated implementation retains vectors in each level's ordering.

Validation undoes padding and permutations, checks every original coefficient
exactly (including stored zeros), and verifies no extra nonzero was introduced.
The lossless source arrays preserve the distinction between original structural
zeros and BSR fill zeros. Three deterministic vectors exercise every candidate
against independent SciPy FP64 products. CSR and BSR use preallocated vectors,
descriptors and workspaces, 10 warmups, and five alternating batches of 50
products. Generic cuSPARSE CSR is labeled separately from native AMGX CSR.
The existing AMGX C upload API supports the square-A baseline; P/R instead use
in-hierarchy native event samples for the current-solver projection.

The root `report.md` combines one comparison table per mesh. Detailed
`products/*.json` files contain occupied blocks/scalar slots, original zeros,
added interior zeros, padding entries/dimensions, CSR/BSR bytes, vector padding,
workspaces, conversion/permutation/device-setup times, correctness checks,
product samples and spreads. Storage inflation in the comparison table covers
exported coarse matrices and transfers; unchanged fine BSR storage is excluded
from that ratio. `projection.json` records measured operation frequencies and
baseline attribution. Export/setup time is separate from solve time.

The recommendation gate requires projected savings to exceed the full observed
product ranges plus normal solve variability. These are sensitivity bounds,
not statistical confidence intervals. If profiled or modeled affected work
exceeds normal solve time, attribution is inconsistent and the runner declines
to recommend integration. Locally favorable operators are listed even when the
whole BSR hierarchy is projected to lose. A positive result remains a projection
until an integrated solver confirms it.

Pre-build validation: 14 host tests passed. Twelve small GPU cases (9×5, 5×9,
9×9; all four block sizes) passed the three-vector checks with kernel compilation
blocked. Square cases also passed the existing native AMGX CSR C-API checks.
Post-build validation passed for all 30 exported operators, 240 matrix variants
and 1,680 independent FP64 product checks. The warm profiling setups reproduced
every measured operator exactly. No compilation or time integration was run by
the agent.
