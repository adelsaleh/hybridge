# n–Gamma manufactured data and geometry

Four manufactured problems (`stationary_baseline`, `stationary_stress`,
`transient_baseline`, `transient_stress`) exist in two geometries, selected
explicitly (there is no default):

* `geometry="cartesian"`: the poloidal-plane model on `(x, y)` with the plain
  divergence. Baseline B is `(-1,1)^2`; the star H is centred at the origin.
* `geometry="axisymmetric"`: the toroidally symmetric model on `(R, Z)` with
  `x = R - 3` and the axisymmetric divergence. B is `(2,4) x (-1,1)`; H is
  centred at `(3, 0)`.

These modules provide data and meshes for the
[D-BDF2 plan](../../docs/development/plans/n_gamma_d_bdf2.md); the stepper
lives in `stepper.py` and the runner in `run_d_bdf2.py`.

```python
from scripts.n_gamma.cases import get_case

case = get_case("transient_stress", geometry="cartesian")
record = case.build_mesh(h=0.1, outer_vertices=160, hole_vertices=40)
mesh = record.mesh
print(record.metadata)  # domain, geometry, requested h, actual largest edge, counts
n_dirichlet = case.density_boundary_at(0.2)      # (first, second) -> values
source = case.density_source_at(0.2)             # continuous, unweighted
b_1, b_2 = case.b_poloidal(*mesh.node_coords.T)  # non-normalized, |b_p| < 1
```

Case functions take mesh coordinates, `(x, y)` or `(R, Z)`, and accept NumPy or
CuPy arrays. Dirichlet functions apply to **every** exterior edge, including
the hole; `boundary_mode="eliminate"` and `bohm_conditions=False` describe this
contract. The stationary cases freeze both fields and their time derivatives.
`density_source` and `momentum_source` return continuous unweighted forcing
(`forcing.S_n` / `forcing.S_Gamma` with a required `geometry=`); the coefficient
builder applies the measure weight (1 or R) and subtracts the **numerical**
new-density pressure gradient from the momentum RHS.

## Stepper and study runner

`stepper.NGammaBDF2Stepper` advances `(n, Gamma)` with two ADR solves per step;
`coefficients.py` builds the weighted coefficients and `diagnostics.py` the
error norms and records. `run_d_bdf2.py` runs the plan's studies; it performs
time integration, so run it only when authorized. Plan first with `--dry-run`:

```bash
.venv/bin/python -m scripts.n_gamma.run_d_bdf2 --geometry cartesian --dry-run
.venv/bin/python -m scripts.n_gamma.run_d_bdf2 --geometry cartesian --case transient_baseline
.venv/bin/python -m scripts.n_gamma.run_d_bdf2 --geometry cartesian --study startup --case transient_baseline
```

The transient study adds an `h/2` spatial-contamination check run (4× the
elements, refined up to twice more when the error changes by more than 10%).
`--no-spatial-check` skips it, for large fixed meshes whose `h/2` level does not
fit in memory; the summary then records `spatially_resolved: null`.

Defaults: p=4, raw-CUDA assembly with device-resident fields on face BSR,
`solver_rtol=1e-11`, the 42-point degree-14 Dunavant volume rule
(`--volume-degree`, or `--volume-quad-1d` for Duffy), p+7 Duffy projection
and error rules, density floor `1e-8`, no primal postprocessing.

Linear solver: `--amgx-config` defaults to
`configs/amgx/adv_rea_gpu4_hdg_pbicgstab_l1_bsr.json` (PBICGSTAB + L1 Jacobi).
A failed solve is retried once, at the same step and `dt`, with
`--amgx-fallback-config` (default FGMRES + block AMG,
`adv_diff_rea_gpu4_hdg_fgmres_amg_block_graph_dense_dilu_bsr.json`), and the
retry is recorded as `*_solve.fallback`. Bounded measurements on 2026-09-29 (p=4,
finest stress mesh, 16k triangles, both geometries, three steps at
`dt=0.02` and `0.0025`):

| AMGX config (face BSR) | Iterations | Warm time per solve |
|---|---:|---:|
| PBICGSTAB + L1 Jacobi (default) | 66–209 | 0.08–0.11 s |
| FGMRES + block AMG / DILU (fallback) | 15–18 | 0.51–0.55 s |
| PBICGSTAB + block Jacobi | 64–76 | 0.09 s; failed at `dt=0.02` |
| PBICGSTAB + block DILU (p1–p3 config) | – | failed |
| built-in FGMRES + MULTICOLOR_DILU, CSR | – | failed on the p=2 stress mesh |

All converged runs gave identical errors. A full default step (two solves plus
assembly and reconstruction) took 0.21–0.23 s on the 16k-triangle mesh, and
0.63 s on its h/2 check mesh, where L1 Jacobi needs 300–390 iterations.
Cross-step caching (2026-09-29) now keeps the time-independent diffusion data,
sparsity pattern and factored mass, and the AMGX solver objects: `--amgx-reuse`
(`preconditioner` by default: coefficients are replaced and the previous setup
is kept, refreshed every `--amgx-refresh-interval` solves, on iteration growth
beyond `--amgx-refresh-growth`, and after a failed stale solve; `solver` redoes
the setup on kept objects; `none` disables). Warm step medians over 12 steps
on the 16k-triangle p=4 stress mesh (`dt=0.0025`):

| Config | no reuse | `solver` | `preconditioner` |
|---|---:|---:|---:|
| PBICGSTAB + L1 (default) | 157 ms | 85 ms | 85 ms |
| FGMRES + block AMG (fallback) | 1004–1059 ms | 886–936 ms | 179 ms |

Density iterations were unchanged by reuse (L1 66–77, block AMG 15), and fields
agreed with no-reuse runs to about 1e-9.
The vendored `hdgfem-gmres` pMG, ASM and block Jacobi + polynomial solvers
are not wired into the package ADR solver and cannot be selected here.

Outputs go to `run_outputs/n_gamma/<geometry>/<case>/`: per-step
`*_steps.csv/.jsonl`, `*_summary.json/.md` convergence tables (spatial orders
use the measured largest edge), and `*_runs.json`.

## Presets and host/device execution

`presets.py` defines one case, `MMS_XY_P6`: the plan's manufactured solution in
the Cartesian poloidal plane `(x, y)`, p=6 (`dub_orth`), both domains B and H,
stationary and transient, the degree-17 (Duffy, 100-point) volume rule, p+7
error rules, and HDG post-processing of the final step of every run, so the
tables also report `n_post_l2`/`Gamma_post_l2` for the degree-7 recovered
fields. Two presets run it:

| Preset | Execution |
|---|---|
| `mms_xy_p6_numba_pardiso` | PyVista panels every step; host Numba on all available threads; oneMKL PARDISO with `--pardiso-threads auto` (8 threads up to 15,000 reduced unknowns, 16 above, capped by the CPU count); reused PARDISO analysis, static preparation cache, reconstruction from assembly columns |
| `mms_xy_p6_device` | Holoviz panels every step; raw CUDA, device-resident fields and CuPy post-processing; face-BSR AMGX PBICGSTAB + L1 with block-AMG fallback; persistent AMGX setups and cached static device data |

```bash
.venv/bin/python -m scripts.n_gamma.run_d_bdf2 --list-presets
.venv/bin/python -m scripts.n_gamma.run_d_bdf2 --preset mms_xy_p6_device --print-preset
.venv/bin/python -m scripts.n_gamma.run_d_bdf2 --preset mms_xy_p6_device --dry-run
.venv/bin/python -m scripts.n_gamma.run_d_bdf2 --preset mms_xy_p6_numba_pardiso --case transient_baseline
```

Options given with a preset override it. Host runs set `OPENBLAS_NUM_THREADS=1`
before NumPy loads, because idle OpenBLAS workers spin; on 24 cores this halved
a p=6 host step (301 → 155 ms at 944 triangles). This costs the coefficient
preparation nothing: its NumPy work is elementwise ufuncs and stacks of tiny
matrix products, which OpenBLAS never threads (24 OpenBLAS threads kept 16
cores busy with no wall-time change). Instead, `hdgfem.core.host_threads` runs
that work in element chunks on a thread pool sized to the Numba thread count.
On 2026-09-29 (transient baseline, h=0.025, 14,776 triangles, p=6, 24 threads)
the two preparations of one step took 0.50 s instead of 1.71 s, matching the
serial result to 5e-16. The manufactured forcing takes about 140 ms of that;
its hundreds of ufunc calls per evaluation are GIL-bound at about 6x.

Host runs therefore default to `--compiled-coefficients`:
`compiled.CompiledCoefficients` evaluates the diffusion tensor, advection and
both sources as Numba `cfunc` pointwise coefficients built on the generated
scalar evaluators in `cases/forcing_numba.py`, compiled once (disk-cached) and
rebound every step. They match the NumPy coefficients to 1 ulp; the NumPy
coefficients remain for the device path and `--no-compiled-coefficients`.
On 2026-09-30 (transient baseline, `--fixed-mesh-size 0.0055`, 306,072 triangles,
p=6, 3.2M trace unknowns per solve, three steps, 24 Numba/host and 16 PARDISO
threads, `--no-spatial-check`):

| Host path | Step 1 | Warm step | Warm preparation per solve | Run | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Serial NumPy preparation building the dense face tables (emulated pre-2026-09-29 path) | 169.8 s | 82.7–109.7 s | 17.9–48.6 s | 6:29 | 117 GB (swapped) |
| Threaded NumPy, fused face tables | 93.1 s | 36.1–36.3 s | 3.2–4.6 s | 2:53 | 71 GB |
| + parallel PARDISO pattern/gather, threaded diffusion tables, NumPy coefficients | 59.1 s | 33.4–33.8 s | 3.2–4.6 s | 2:14 | 68 GB |
| + compiled coefficients (default) | 51.7 s | 30.1–30.5 s | 2.0–2.6 s | 2:00 | 68 GB |

All four runs give the same errors. A warm solve is now 6.5 s of assembly (all
24 cores busy), 5.0 s of PARDISO (3.5 s factorization on 16 threads; MKL's
default two iterative-refinement steps cost 1.3 s of the 1.75 s triangular
solves), 2.0–2.6 s of preparation and 0.9 s of reconstruction. Every run
records the Numba, host NumPy, PARDISO and OpenBLAS thread counts and the
stepping CPU/wall ratio, and every
step records which caches its two solves reused (`*_solve.static_reused`,
`analysis_reused` (PARDISO), `setup_reused` (AMGX), `reconstruction_reused`).
A bounded six-step check on 2026-09-29 (star mesh, 4,444 triangles, p=6,
`dt=0.0025`) took 0.77 s per warm host step (24 Numba and 16 PARDISO threads,
CPU/wall 7.0) and 0.08 s per warm device step; the first two device steps paid
one-time CuPy compilation of the p=6 kernels (about 25 s each).

## Plotting and verbosity

`--plot-every N` (the presets use 1) shows six panels every `N` steps of every
run, including the initial state: exact, numerical and error fields of `n`
(top row) and `Gamma` (bottom row). `--plot-backend auto` selects Holoviz on
the device path, sampling the device-resident fields and the exact solution on
the GPU, and PyVista on the host path, evaluating the exact solution pointwise
at the plot points. Neither backend projects the exact solution. Without a
display, PyVista saves frames under `<output>/<geometry>/<case>/frames/<run>`;
Holoviz saves frames only with `--plot-dir` or a movie with `--plot-movie`.
`--plot-off-screen` renders without a window. Other options:
`--plot-width/--plot-height` (total window size), `--plot-resolution` (PyVista
sub-samples per edge), `--plot-show-mesh` and `--plot-max-fps` (Holoviz preview
cap; saved frames are never dropped).

`-v/--verbosity` (or `--quiet`) selects:

| Level | Output |
|---|---|
| 0 | nothing (errors still raise) |
| 1 (default) | case headers, one line per run (mesh, p, dt, steps, PARDISO threads), progress every tenth of a run, final errors and the convergence tables |
| 2 | also one line per step: stage, solve iterations, reused caches (`S` static, `A` PARDISO analysis, `P` AMGX setup, `R` reconstruction) per equation, fallbacks, floor clamps, minimum sampled density, step time |
| 3 | also the ADR solver's own stage lines inside every step |

Plotting time is excluded from the reported seconds per step and recorded
separately (`stepping.plot_seconds` in `*_runs.json`).

Plot cost per update, measured at p=6: PyVista about 30 ms, or about 115 ms
when frames are saved as PNG (headless); Holoviz about 3 ms, and preview
frames dropped by `--plot-max-fps` or a busy renderer are skipped before any
sampling. For long runs, raise `--plot-every` or save frames only on request.
VTK text uses the FreeType backend and the panel captions are updated in
place, since matplotlib mathtext rendering and rebuilt caption actors cost
several seconds per update.

Ctrl-C stops a run at the next Python instruction: the runner closes the
solvers and the plot window, prints that it was interrupted, and exits with
status 130. Completed runs keep their outputs. Holoscan installs its own SIGINT
handler; when it stops the renderer first, the panels raise the same
interrupt. Gmsh sessions are opened with `interruptible=False`, because gmsh's
interruptible mode leaves SIGINT at the default action after `finalize()`.

## Geometry

`build_case_mesh(domain, h, geometry=...)` uses the shared Gmsh mesh/cache
machinery. In the axisymmetric frame B is the rectangle `(2,4) × (-1,1)` and H
uses star center `(3,0)`, radius `0.70`, absolute modulation `0.224`, five
lobes, and an offset hole at `(3.28,0.10)` of radius `0.12`; the Cartesian frame
is the same geometry shifted by `-3` in the first coordinate. The table below
was measured in the axisymmetric frame; Cartesian meshes record their own
metadata, which the runner stores with every result.

Both are affine triangular meshes. H's outer wall and hole are independently
polygonized, and exact boundary data are evaluated on these actual straight
segments. Vertex counts describe the geometry; Gmsh may subdivide segments.
Errors belong to the actual polygonal domain, not the smooth limiting domain.

The shared `gmsh_smooth_star_mesh` accepts `boundary_points`, `hole_center`
(absolute coordinates), and `hole_boundary_points`. Its existing default is
still a concentric circular CAD hole. The entire disk must have strict
clearance from the sampled outer polygon. Hole geometry enters cache identity.

Diagnostic meshes generated on 2026-09-29 with Gmsh 4.15.2 and one meshing
thread gave the following values. Recompute metadata for each run; the target
Gmsh size is not the measured largest edge, particularly when boundary
polygonization forces smaller elements.

| Domain | Target h | Outer/hole vertices | Actual largest edge | Triangles |
|---|---:|---:|---:|---:|
| B | 0.20 | 4/0 | 0.232711915 | 244 |
| B | 0.10 | 4/0 | 0.139711001 | 946 |
| B | 0.05 | 4/0 | 0.062645674 | 3706 |
| H | 0.20 | 80/20 | 0.111310684 | 1160 |
| H | 0.10 | 160/40 | 0.063386866 | 4442 |
| H | 0.05 | 320/80 | 0.032751305 | 16434 |

## Regeneration and validation

SymPy is only needed to regenerate the committed forcing module:

```bash
.venv/bin/python -m pip install -e '.[manufactured]'
.venv/bin/python -m scripts.n_gamma.manufactured
.venv/bin/python -m scripts.n_gamma.manufactured --check
OMP_NUM_THREADS=16 .venv/bin/python -m pytest -q \
  tests/test_n_gamma_manufactured.py tests/test_n_gamma_geometry.py \
  tests/test_n_gamma_coefficients.py tests/test_n_gamma_stepper.py \
  tests/test_n_gamma_diagnostics.py tests/test_n_gamma_runner.py \
  tests/test_guiding_center_star_hole.py tests/test_mesh_cache.py
```

On 2026-09-29 the n–Gamma tests passed (61 in the six `test_n_gamma_*` files, no skips),
and the generator drift check passed.

The forcing tests independently differentiate the continuous PDE with finite
differences, test frozen stationary sources, density bounds and magnetic
projector eigenvalues, compare NumPy/CuPy results without host downloads, and
check generation drift (skipped if SymPy is absent). Mesh diagnostics check
boundary membership, clearance, polygon area, annulus topology, inward-to-hole
fluid normals, cache separation, and positive R for all four cases. These are
data/geometry diagnostics, with no PDE solve or time integration.

Repository-wide documentation checks currently also fail on unrelated
historical artifact links, generated documentation files, missing existing
docstrings, and the planned but unimplemented n–Gamma runner path. A packaging
check fails on the existing `hdgfem/core/geometries` directory lacking an
`__init__.py`. These broader checks are not claimed as passing.
