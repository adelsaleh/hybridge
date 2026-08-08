# Implementation Plan: Parallel DG and HDG-Trace Transfer Between Unrelated Triangular Meshes

## 0. Scope and required behavior

Implement conservative/high-order transfer between two unrelated triangular meshes over the same intended physical domain:

1. **DG volume transfer**
   \[
   V_h^{DG}\to V_{h'}^{DG},\qquad u_h\mapsto u_{h'}.
   \]

2. **HDG/skeleton trace transfer**
   \[
   \widehat V_h\to \widehat V_{h'}.
   \]

3. Support a **small mismatch between the polygonal discrete domains** \(\Omega_h\) and \(\Omega_{h'}\) near the physical boundary.

4. Provide separate **host/CPU** and **device/CUDA** implementations.

5. Support and optimize triangular polynomial degrees up to **\(p_{\max}=14\)** for both source and target spaces. Degrees \(p\le 10\) are expected to be the most common production case, but they are **not** an implementation limit. Compile/test specializations through degree 14. For degrees above 14, use an explicit unsupported/fallback path unless a later optimization extends the design.

### Non-goals / constraints

- Do **not** assemble a global projection matrix.
- Do **not** build full supermesh connectivity unless another part of the code explicitly needs it.
- Do **not** use direct old-skeleton/new-skeleton intersections for unrelated meshes: generic skeleton faces intersect in zero \((d-1)\)-measure.
- DG target cells and target trace faces are independent local projection problems; exploit this fully.
- All geometry tolerances must be **scale-aware**; do not use one hard-coded absolute epsilon.

### `DGMesh` geometry prerequisite

`DGMesh` must retain enough physical-geometry information to distinguish the
intended domain from its current polygonal discretization. At minimum, preserve
boundary entity tags and geometry provenance; for curved or high-order domains,
retain the boundary/element geometry maps or their reconstructible parameters.

The transfer must use this information to determine whether two meshes:

1. exactly cover the same discrete domain;
2. approximate the same intended physical domain with a small boundary mismatch;
3. cover genuinely different domains.

For nonidentical coverage, report overlap, uncovered target measure, discarded
source measure, and the associated mass separately. Conservation is defined on
the actual overlap unless an explicit geometry-aware extension or correction
policy is selected. If the stored geometry cannot establish the relationship
between the domains, reject the transfer instead of inferring it from nearby
linearized boundary facets alone.

---

## 1. Common mathematical definition: DG volume transfer

Let \(K\in\mathcal T_h\) be an old/source triangle and \(K'\in\mathcal T_{h'}\) a new/target triangle. On each target cell solve the local \(L^2\) projection

\[
M_{K'}U_{K'}'=b_{K'},
\]

with

\[
(M_{K'})_{ij}=\int_{K'}\phi_i'\phi_j'\,dx,
\]

and, when the two discrete domains coincide,

\[
(b_{K'})_i=
\sum_{K:\,|K\cap K'|>0}
\int_{K\cap K'}u_h|_K\,\phi_i'\,dx.
\]

For affine triangles and a fixed reference basis,

\[
M_{K'}=|\det J_{K'}|\,\widehat M.
\]

Precompute \(\widehat M^{-1}\). If the reference basis is orthonormal, the local solve is only a scalar scaling.

For source degree \(p\) and target degree \(p'\), use intersection quadrature exact for degree at least

\[
p+p'.
\]

For the supported range \(p,p'\le 14\), pretabulate suitable reference-triangle quadrature rules. In the equal-degree worst case this requires exactness through degree \(28\).

---

## 2. Triangle intersection representation

For every candidate pair \((K,K')\):

1. Compute the convex polygon
   \[
   P=K\cap K'.
   \]
2. Two triangles produce at most a 6-vertex convex polygon.
3. Reject intersections whose scale-aware area is numerically zero.
4. Fan-triangulate \(P\) into at most four triangles.
5. Perform quadrature immediately on these subtriangles.

Do not persist supermesh topology. Only transient intersection vertices/quadrature data are needed.

Use fixed-size stack/register/thread-local storage for at most 6 polygon vertices; no allocation inside hot loops.

---

## 3. Boundary mismatch: \(\Omega_h\ne\Omega_{h'}\)

### 3.1 Cases

- If \(\Omega_h=\Omega_{h'}\) exactly as discrete polygonal domains: use the ordinary intersection projection; there is no special transfer issue.
- If the two meshes approximate the same physical domain but their polygonal boundaries differ slightly: use the extension treatment below.
- If the mismatch is not small relative to the local mesh size, or topology/physical boundary has genuinely changed: **do not extrapolate blindly**. Require a common physical geometry map, a problem-specific boundary extension, or a remeshing-specific fallback.

### 3.2 Boundary extension polynomial

For each affected target boundary cell \(K'\), associate an old/source boundary cell \(K_b\). Define the local extension by polynomial continuation:

\[
E_bu_h(x)=u_h|_{K_b}(x)
=\sum_j U_{K_b,j}\,\widehat\phi_j(F_{K_b}^{-1}(x)),
\]

allowing \(F_{K_b}^{-1}(x)\) to lie slightly outside the reference triangle.

Only use this for a small boundary mismatch. Store/compute a dimensionless extension distance such as

\[
\delta=\frac{\operatorname{dist}(x,\Gamma_h)}{h_{K_b}},
\]

and flag/fallback if it exceeds a configurable threshold.

A safer optional mode is constant-normal/closest-boundary extension instead of high-order polynomial continuation.

### 3.3 GPU/CPU-friendly identity: never construct \(K'\setminus\Omega_h\)

For every target boundary cell use

\[
\boxed{
(b_{K'})_i
=
\int_{K'}E_bu_h\,\phi_i'\,dx
+
\sum_K
\int_{K'\cap K}
\left(u_h|_K-E_bu_h\right)\phi_i'\,dx.
}
\]

This is exactly

\[
\int_{K'\cap\Omega_h}u_h\phi_i'\,dx
+
\int_{K'\setminus\Omega_h}E_bu_h\phi_i'\,dx,
\]

without geometrically constructing the complement.

It is safe to apply this formula to every target boundary element. If \(K'\subset\Omega_h\), the extension terms cancel up to quadrature/roundoff.

### 3.4 Coverage diagnostic

While processing intersections accumulate

\[
A_\cap(K')=\sum_K |K\cap K'|.
\]

For a target cell that is not marked as a boundary cell, require

\[
A_\cap(K')/|K'|\approx 1.
\]

If this fails, place the cell in an error/fallback list. Possible causes: missed BVH candidates, bad clipping, unexpected boundary mismatch, or invalid mesh geometry.

---

# PART A — HOST / CPU IMPLEMENTATION

## 4. Host design principle

On CPU, prefer a **fused target-centric traversal**:

\[
\text{target cell}\to\text{BVH query}\to\text{exact clipping}\to\text{quadrature}\to\text{local solve}.
\]

Unlike the GPU path, do not automatically materialize a candidate CSR list on the host. Branchy geometry and variable work are handled well by CPU task scheduling, and fusing avoids extra memory traffic.

Build the old/source element spatial index once per source mesh and reuse it for both volume and trace transfer.

Recommended default spatial structure: a flat AABB BVH over source triangles. A uniform spatial hash can be an optional fast path for quasi-uniform meshes, but the BVH should be the general implementation.

Also build a small boundary-edge spatial index for locating \(K_b\) for target boundary cells/faces.

---

## 5. Host DG volume projection

Parallelize over target triangles. Each worker owns one \(K'\) at a time and writes only \(U_{K'}'\).

Pseudocode:

```text
parallel_for target K' with dynamic/task-stealing schedule:
    rhs[0:Np_new] = 0
    covered_area = 0

    if K' is boundary:
        Kb = precomputed_extension_owner[K']
        rhs += integrate_full_target_triangle(E_b u_h * phi_new)

    traverse source-element BVH with AABB(K'):
        for candidate old K:
            P = exact_triangle_intersection(K, K')
            if P has zero area: continue

            covered_area += area(P)

            for subtriangle S in fan_triangulation(P):
                for quadrature point xq,wq in S:
                    uold = eval_old_polynomial(K, xq)
                    if K' is boundary:
                        uext = eval_old_polynomial(Kb, xq)
                        value = uold - uext
                    else:
                        value = uold

                    accumulate rhs_i += wq * value * phi_new_i(xq)

    apply reference local mass inverse
    write U_new[K']
    run coverage diagnostic
```

### CPU scheduling for heavy multithreading

Use exactly one outer parallel layer. Do **not** call a threaded BLAS from inside it.

Preferred scheduling:

- TBB/task runtime: `parallel_for` + work stealing / auto partitioner.
- OpenMP: start with `schedule(dynamic, 8)` or `schedule(guided, 8)`; benchmark chunk sizes roughly 4–32.
- If the runtime only offers static scheduling (for example some `prange` paths), precompute or cheaply estimate candidate counts and bucket target cells by workload before the parallel loop.

Start with one worker per **physical core**. Benchmark SMT/2 threads per core only after the physical-core configuration; branchy BVH traversal can sometimes benefit from SMT, but it is not guaranteed.

For multi-socket NUMA systems:

- partition target elements by NUMA node;
- first-touch target/output storage on the owning socket;
- keep thread affinity fixed;
- avoid cross-socket migration.

### CPU memory/layout rules

- Element-major coefficient storage:
  `coeff[element * Np + mode]`.
- Prefer SoA/flat arrays for triangle geometry and BVH nodes.
- Keep `rhs`, polygon vertices, Jacobians and quadrature scratch thread-local.
- No heap allocation in the hot loop.
- Align frequently accessed arrays to cache-line boundaries.
- Batch basis evaluation over several quadrature points to enable SIMD; do not try to SIMD-vectorize the branch-heavy clipping logic first.
- For \(p\le 14\), \(N_p\le 120\), so one local FP64 RHS is at most 960 bytes and still fits comfortably in cache/thread-local scratch.

### CPU load-imbalance fallback

Dynamic/task scheduling should handle most adaptive meshes. Only add a pair-centric/heavy-cell fallback if profiling shows a severe long tail in the number of intersections per target element.

If needed:

1. classify cells with `candidate_count > heavy_threshold`;
2. split a heavy target cell into chunks of source candidates;
3. compute partial RHS vectors in independent tasks;
4. reduce the few partial vectors into the target RHS.

Do not implement this before profiling proves it is needed.

---

## 6. Host trace/skeleton transfer

### 6.1 Do not intersect unrelated skeletons directly

For unrelated meshes, old and new edges generally intersect only at points, so a direct \(L^2\) skeleton-to-skeleton projection is not defined by edge intersections.

Use the old volume field as a mediator:

\[
\widehat V_h
\xrightarrow{\mathcal R_h}
V_h^{DG}
\xrightarrow{\text{restriction to new skeleton}}
\widehat V_{h'}.
\]

If the old primal \(u_h\) is already available, reuse it and skip reconstruction.

If only \(\widehat u_h\) is stored, reconstruct the old element field once with the existing HDG local reconstruction operator. Do not invent a new lifting if the code already has the HDG local solve/reconstruction.

### 6.2 Local target-edge projection

For each new target edge \(F'\):

\[
M_{F'}\widehat U_{F'}'
=
\sum_{K:\,|F'\cap K|>0}
\int_{F'\cap K}u_h|_K\,\widehat\psi'\,ds.
\]

In 2D each \(F'\cap K\) is a line segment. Query the same old-triangle BVH with `AABB(F')`, compute segment-triangle intersections, and integrate on each segment.

If \(F'\) is a new physical boundary face and the PDE prescribes a Dirichlet datum \(g\), prefer

\[
\int_{F'}\widehat u_{h'}\widehat\mu'\,ds
=
\int_{F'}g\widehat\mu'\,ds
\]

instead of transferring the old numerical trace.

### 6.3 Boundary mismatch on target edges

For a target boundary edge that extends slightly outside \(\Omega_h\), use the analogous identity

\[
\boxed{
(\widehat b_{F'})_i
=
\int_{F'}E_bu_h\,\widehat\psi_i'\,ds
+
\sum_K
\int_{F'\cap K}(u_h|_K-E_bu_h)\widehat\psi_i'\,ds.
}
\]

Again, do not construct the uncovered piece explicitly.

### 6.4 Host parallelism

Parallelize over target edges with dynamic/task-stealing scheduling. Each edge owns its local RHS and local mass solve, so there are no atomics.

---

# PART B — DEVICE / CUDA IMPLEMENTATION

## 7. Device design principle

On GPU, separate irregular search from the expensive projection kernel:

\[
\boxed{
\text{BVH query/count}
\to
\text{prefix scan}
\to
\text{candidate CSR fill}
\to
\text{projection kernel}.
}
\]

This makes the projection kernel operate on compact contiguous candidate ranges.

Device data layout:

```text
old_vertices / old_x0 / old_Jinv
old_coeff
new_vertices / new_x0 / new_Jinv
candidate_offsets[Ntarget+1]
candidate_cells[Npairs]
new_boundary_flag[Ntarget]
extension_cell[Ntarget]
```

Store coefficients element-major:

```text
coeff[element * Np + mode]
```

so a warp/block reading all modes of one element is coalesced.

Build/reuse:

- one device BVH/LBVH over old triangles;
- one device spatial index over old boundary edges;
- pretabulated quadrature and reference-basis/mass data in read-only/constant storage when practical.

---

## 8. Device candidate search

### Kernel A: count candidates

Use one warp per target triangle:

```cpp
blockDim.x = 128;              // 4 warps
 gridDim.x = ceil(Ntarget/4);  // 4 target cells/block
```

Each warp traverses the old-triangle BVH using `AABB(K')` and writes `candidate_count[K']`.

Run a device exclusive scan to produce `candidate_offsets`.

### Kernel B: fill candidate CSR

Use the same launch configuration and repeat the BVH traversal, now writing candidate old-cell indices into the CSR range for each target cell.

Exact triangle clipping can remain in the projection kernel.

### Boundary-extension owner kernel

For target boundary cells, use one warp per target boundary cell to locate a nearby/compatible old boundary edge and store its adjacent source cell as `extension_cell[K']`.

Use geometry/normal checks to avoid selecting the wrong side of a thin or non-convex region. If the association is ambiguous or too far away, flag the target cell for fallback.

---

## 9. Device DG projection: degree-dependent kernels

Compile/template-specialize supported source/target degree combinations through `MAX_TRANSFER_DEGREE = 14`. Do not silently run an untested generic path above 14; return an explicit unsupported/fallback status unless higher-degree support is added deliberately.

### 9.1 Low degree \(p\le 6\)

For triangular \(P_p\),

\[
N_p=\frac{(p+1)(p+2)}2,\qquad N_6=28<32.
\]

A warp can map one active lane to one modal coefficient.

Initial dispatch:

```text
p = 0..2 : 1 warp / target
p = 3..4 : 2 warps / target
p = 5..6 : 4 warps / target
```

For the 4-warp case use

```cpp
blockDim.x = 128;
gridDim.x  = Ntarget;
```

with one block per target cell. Warps split the source-candidate list and reduce partial RHS vectors through shared memory.

Special-case \(P_0\) if worthwhile; the generic modal-per-lane mapping is correct but wasteful there.

### 9.2 High-order supported range \(7\le p\le 14\)

Use one target triangle per block. Start with

```cpp
blockDim.x = 128;   // 4 warps
gridDim.x  = Ntarget;
```

and retain a 256-thread specialization for benchmarking/heavy cells.

For reference:

\[
N_7=36,\quad N_8=45,\quad N_9=55,\quad N_{10}=66,\quad N_{12}=91,\quad N_{14}=120.
\]

Use `p_work = max(p_old, p_new)` for the conservative launch-policy decision because source degree controls source-polynomial evaluation cost while target degree controls RHS/moment count. The target degree `p_new` determines the number of target coefficients.

The 128-thread design remains the default through roughly \(p=12\). For \(p=13\)–14, provide both 128-thread and 256-thread specializations and benchmark them. Degrees \(p\le 10\) remain the likely common workload, but the implementation must not impose a degree-10 limit.

### 9.3 Cooperative 128-thread algorithm for \(p\ge7\)

Use four warps to process four quadrature points concurrently.

For source evaluation at quadrature point \(q_w\), warp \(w\) computes

\[
u_h(x_{q_w})=\sum_j U_j\phi_j(x_{q_w}).
\]

Lane \(\ell\) handles

\[
j=\ell,\ell+32,\ell+64,\ldots
\]

then use warp-shuffle reduction. Store the four values in shared memory.

For target moment accumulation, flatten the \((q,i)\) tile for the four quadrature points:

```text
r = q * Np_new + i
```

and let thread `t` process

```cpp
for (r = threadIdx.x; r < 4*Np_new; r += 128) {
    q = r / Np_new;
    i = r % Np_new;
    ... accumulate contribution to rhs[i] ...
}
```

For \(p'=10\), `4*Np_new = 264`; for \(p'=14\), \(N_{14}=120\) and `4*Np_new = 480`. Both map naturally to a 128-thread block, with each thread processing multiple flattened `(q,i)` entries at the higher degree.

Avoid atomics inside a target block. Use shared-memory/tiled reduction or per-thread/per-mode partials as appropriate.

### 9.4 Optional 256-thread variant

For heavier intersection/quadrature work:

```cpp
blockDim.x = 256;   // 8 warps
gridDim.x  = Ntarget;
```

Process eight quadrature points concurrently and flatten the `8 * Np_new` target-moment tile.

Use 256 threads as an optional benchmarked alternative for \(p=7\)–12, and as a required comparison candidate for \(p=13\)–14. If support is later extended to \(p\ge 15\), 256 threads should become the default starting point.

Recommended device dispatch (using `p_work = max(p_old, p_new)`):

```text
p_work = 0..2 : 1 warp / target
p_work = 3..4 : 2 warps / target
p_work = 5..6 : 4 warps / target
p_work = 7..12: 128 threads / target (default)
p_work = 13..14: benchmark/select 128 vs 256 threads / target
p_work > 14: explicit unsupported/fallback unless support is extended
```

---

## 10. Device boundary-mismatch path

For target boundary cells, first compute the full-cell extension contribution

\[
\int_{K'}E_bu_h\phi_i'\,dx.
\]

Then process the normal intersection CSR, but replace source values by

\[
u_h|_K(x_q)-E_bu_h(x_q).
\]

The same target-local kernel can therefore handle both exact-domain and boundary-mismatch cases with only a boundary flag and `extension_cell[K']`.

Important: do not try to classify/construct the exact uncovered polygon on device.

Accumulate `covered_area` during clipping and write diagnostics/fallback flags for invalid non-boundary coverage.

---

## 11. Device local mass solve

For affine target triangles:

\[
M_{K'}=|\det J_{K'}|\widehat M.
\]

Precompute \(\widehat M^{-1}\) per target degree.

- Orthonormal modal basis: one scaling per coefficient.
- Otherwise apply the small fixed reference inverse locally.

Never launch a global sparse solve for DG projection.

---

## 12. Device heavy-cell fallback

Record `candidate_count[K']` from the search stage.

Default: target-centric projection.

Only if the distribution has a serious long tail, classify heavy cells and use a pair/chunk-centric fallback:

1. one warp/block handles one or several \((K',K)\) intersections;
2. compute partial target RHS contributions;
3. segmented-reduce partials by \(K'\).

Do not implement this initially unless profiling shows target-block stragglers.

---

## 13. Device trace/skeleton transfer

### 13.1 Source field

If old \(u_h\) exists, reuse it.

If only \(\widehat u_h\) exists, reconstruct old volume coefficients first with the existing HDG local reconstruction, in parallel over old elements.

Then transfer from the old volume field to the new skeleton.

### 13.2 Candidate search for new target edges

Query the same old-triangle BVH using each target edge AABB.

Use the same two-pass count/scan/fill CSR design if the edge-candidate list is needed on device.

### 13.3 Trace projection kernel for supported \(p\le14\)

A scalar polynomial trace on an edge has only

\[
N_{\widehat p}=\widehat p+1\le 11
\]

modes for \(\widehat p\le14\), i.e. at most 15 target trace modes. Therefore use **one warp per target edge** rather than one whole block per edge. The source-volume evaluation may involve up to \(N_{14}=120\) modes; one warp can evaluate these by striding over modal indices and reducing.

Recommended launch:

```cpp
blockDim.x = 128;             // 4 warps
gridDim.x  = ceil(Nfaces/4); // 4 target edges/block
```

Each warp:

1. loops through the candidate old triangles for its target edge;
2. computes `segment ∩ triangle`;
3. performs 1D Gauss quadrature on the resulting segment;
4. evaluates old volume polynomial and new edge basis;
5. accumulates the local edge RHS in registers;
6. applies the local edge mass inverse;
7. writes the new trace coefficients.

For target boundary edges with geometric mismatch use

\[
\int_{F'}E_bu_h\widehat\psi_i'\,ds
+
\sum_K\int_{F'\cap K}(u_h-E_bu_h)\widehat\psi_i'\,ds.
\]

For prescribed Dirichlet boundaries, project the exact boundary datum directly instead.

If a target edge crosses an extreme number of source cells, optionally promote that edge to a block-per-edge heavy path.

---

## 14. Optional global conservation correction

If transferring a density and exact global mass conservation is required, compute

\[
m_h=\int_{\Omega_h}u_h\,dx,
\qquad
m_{h'}=\int_{\Omega_{h'}}u_{h'}\,dx.
\]

The \(L^2\)-minimal global constant correction is

\[
\alpha=\frac{m_h-m_{h'}}{|\Omega_{h'}|},
\qquad
u_{h'}\leftarrow u_{h'}+\alpha.
\]

For a modal basis this modifies only the constant mode (with the appropriate normalization).

Implement as an optional feature because it changes the local projection to enforce a global invariant. If positivity or another invariant is required, do not assume a constant correction is always acceptable; use the problem-specific conservative limiter/correction instead.

---

## 15. Recommended reusable interfaces

Exact names can follow the existing codebase, but keep these responsibilities separate:

```text
build_source_element_spatial_index(mesh_old)
build_source_boundary_spatial_index(mesh_old)

project_dg_host(mesh_old, coeff_old, mesh_new, p_old, p_new, options)
project_dg_device(mesh_old_dev, coeff_old_dev, mesh_new_dev, p_old, p_new, options)

project_trace_host(mesh_old, old_volume_or_trace_state, mesh_new, ...)
project_trace_device(mesh_old_dev, old_volume_or_trace_state_dev, mesh_new_dev, ...)
```

Common geometry helpers:

```text
triangle_triangle_intersection()
segment_triangle_intersection()
fan_triangulate_convex_polygon()
map_physical_to_reference()
evaluate_modal_basis()
locate_boundary_extension_owner()
```

Keep CPU and CUDA geometry kernels mathematically identical where possible so they can be cross-validated.

---

## 16. Validation tests Codex must add

### Geometry/intersection tests

1. identical triangles;
2. disjoint triangles;
3. one triangle contained in another;
4. partial overlap producing 3, 4, 5 and 6-vertex polygons;
5. edge/vertex touching only: zero-area intersection;
6. random triangle pairs compared CPU vs GPU;
7. total intersection area reproduces target-cell area for interior cells.

### DG projection tests

For unrelated meshes over the same exact polygonal domain:

1. project constants: reproduce exactly to tolerance;
2. project every polynomial representable in both spaces up to \(\min(p,p')\): reproduce to quadrature/roundoff tolerance;
3. compare CPU and GPU coefficients;
4. verify expected \(L^2\) convergence for a smooth non-polynomial field;
5. verify mass preservation for equal domains.

### Boundary-mismatch tests

1. two independently polygonized approximations of the same curved domain;
2. constant field: extension transfer must remain constant;
3. smooth polynomial field with a very small boundary displacement;
4. verify that the boundary formula matches an explicitly constructed uncovered-region reference on small test meshes;
5. verify extension-distance guard/fallback;
6. verify optional global mass correction.

### Trace-transfer tests

1. old and new unrelated skeletons;
2. constant/low-degree old volume field gives correct new trace projection;
3. CPU vs GPU trace coefficients;
4. if old trace-only state is used, reconstruct then transfer and compare against transfer using the already reconstructed old primal;
5. prescribed Dirichlet target edges use the boundary datum rather than old trace values.

---

## 17. Profiling metrics

Record separately:

```text
spatial-index build time
candidate search time
candidate count distribution
exact clipping time
quadrature/basis-evaluation time
local mass solve time
boundary-extension overhead
trace-transfer time
CPU thread scaling
GPU occupancy/register/shared-memory usage
heavy-cell count
```

For GPU kernels use 128 threads as the primary configuration for \(p=7\)–12, and compare 128 vs 256 threads explicitly for \(p=13\)–14. It is also useful to retain selected 128-vs-256 measurements below degree 13 to detect workload-dependent crossovers.

For CPU compare physical-core count vs SMT-enabled count and dynamic/guided chunk sizes. Do not assume maximum logical-thread count is automatically fastest.

---

## 18. Implementation order

Implement in this order so every step has a reference test:

1. Robust CPU triangle-triangle intersection and polygon quadrature.
2. Serial CPU DG projection on equal discrete domains.
3. Multithreaded host DG projection with fused BVH traversal.
4. Host boundary-extension identity and coverage diagnostics.
5. Host trace transfer via old volume field.
6. CPU test suite as numerical reference.
7. Device old-triangle BVH + candidate count/scan/fill CSR.
8. Device interior DG projection for low degree.
9. Device \(p\le6\) specialized warp mapping.
10. Device cooperative high-order kernels through \(p=14\): 128-thread default for \(p=7\)–12, plus both 128- and 256-thread variants for \(p=13\)–14.
11. Device boundary-extension path and diagnostics.
12. Device trace candidate CSR + warp-per-target-edge kernel.
13. CPU/GPU equivalence tests.
14. Benchmark 128 vs 256 high-order kernels.
15. Add optional conservation correction.
16. Add heavy-cell fallback only if profiling demonstrates a need.

---

## 19. Default performance policy for the supported range \(p\le14\)

### Host

- Flat BVH over old triangles.
- Fused query + clipping + quadrature + local solve.
- One target cell/task.
- Dynamic/task-stealing scheduling.
- All physical cores first; benchmark SMT separately.
- Thread-local fixed-size scratch; no atomics; no nested threaded BLAS.
- SIMD/batch basis evaluation over quadrature points.

### Device

- BVH query -> count -> prefix scan -> candidate CSR -> projection.
- \(p\le2\): 1 warp/target.
- \(p=3\)–4: 2 warps/target.
- \(p=5\)–6: 4 warps/target.
- \(p=7\)–12: one target/block, **128 threads as the default**; retain a 256-thread variant for profiling/heavy workloads.
- \(p=13\)–14: one target/block; compile **both 128- and 256-thread variants** and choose from benchmark/profile data.
- Trace faces for \(p\le14\): one warp/target face, 4 faces per 128-thread block.
- No atomics in the normal target-centric projection paths.

This is the implementation baseline. Optimize or introduce pair-centric fallbacks only after profiling the real remeshing workloads.
