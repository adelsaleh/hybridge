# Reproducing the GPU showcase

The [README](../../README.md) shows two degree-6 HDG calculations on the same
five-lobed star with a circular island. The signed run is exactly the
simulation of the [example script](../../examples/gpu_vortex_gas.py); the
recorder `scripts/reports/record_gpu_showcase.py` builds the same solvers
through `scripts/reports/gpu_showcase_setup.py`, runs the same time loop
(forming the BDF2 data through `hdg.bdf2_transport_data`, which computes the
example's `s` and `beta` with input checks), and
adds Matplotlib rendering, diagnostics, a final checkpoint, and provenance.
The published positive run also applies a KKT positivity projection after each
transport step. It was recorded on the `positivity-kkt` development branch,
which is not part of this release; see
[the positive run with KKT positivity](#the-positive-run-with-kkt-positivity).

## Problem

Both runs evolve a guiding-center plasma between grounded conducting walls:

```text
∂t ρ + div(ρ u) = 0,   -Δφ = ρ,   u = (∂y φ, -∂x φ),   φ = 0 on both walls.
```

The signed run holds equal amounts of positive and negative charge (a
two-species plasma), the positive run one species. Both walls are
equipotentials, so u·n = 0 there. The same equations are two-dimensional Euler
flow in vorticity form, with one caveat for the island: with φ = 0 there, the
circulation around the island equals −∫ h ρ dx, where h is harmonic with h = 1
on the island and h = 0 on the outer wall, so it changes as charge moves
relative to the island. A Kelvin-consistent Euler flow would instead hold that
circulation fixed with a floating island potential φ = c(t).

| Parameter | Both recordings |
|---|---|
| Outer wall | `r(θ) = 1 + 0.35 cos(5θ)`, 500 boundary points |
| Island | Centered circle of radius `0.3` |
| Mesh | Gmsh size `0.005`; **360,379** straight-sided triangles |
| Approximation | Dubiner basis, **p = 6** |
| Element unknowns per scalar field | **10,090,612** |
| Interior trace unknowns per solve | **3,774,995** |
| Initial charges | **960** Gaussians, counts `(512, 256, 128, 64)`, widths `(0.008, 0.016, 0.024, 0.032)` |
| Profile | Amplitude `4`, seed `17`, cutoff and wall clearance **5 widths** |
| Poisson stabilization | Library default `τ = κ/L_Ω`, with `L_Ω` twice the area over the boundary length; here **τ ≈ 1.913** |

`MeshDomain` samples centers in proportion to triangle area, keeping each
Gaussian's support, five widths, inside the fluid; every scale can start in
the narrow necks between the lobes and the island. The signed profile balances
positive and negative strengths at each scale. Both saved profiles,
`initial_balanced_c5_h005.npz` and `initial_positive_c5_h005.npz`, are reused by
every check below.

## Solvers

The Poisson solve uses raw-CUDA face-BSR assembly and the native face-block
hp-multigrid preconditioned CG with an AMGX coarse solve; its operator and
local Cholesky factors are cached and only the right-hand side changes between
steps. Transport uses fused raw-CUDA BSR assembly with AMGX and zero-flux
walls. The discrete velocity is the rotated HDG flux, which is discontinuous
across faces, so both outward traces of a face can point outward; the default
`conflict-averaged-upwind` policy repairs those faces. Both solves request a
relative tolerance of `1e-9` and an absolute tolerance of `1e-10`,
warm-start from the previous trace, and keep fields on the GPU; only two sampled
rasters per frame are downloaded for drawing.

The positive run reuses the guiding-center runner's bounded Jacobi, FGMRES,
and residual-correction retries. If those were exhausted, an unscaled
nonsymmetric host PyPardiso solve, under a verified 16-thread MKL limit with
measured CPU use, would recover the same corrected discrete problem before
either time history is committed; the metadata records any such event.
The host recovery itself was exercised on 40,715 triangles at p = 6 by
forcing the recovery branch: its field differed from the accepted GPU result by
`8.53e-11` in relative coefficient norm, MKL reported 16 threads with about 16
cores in use, and field and trace returned as device arrays with a physical
residual of `4.57e-19` against a `1e-10` target.
In the published positive run the bounded GPU retries always sufficed: it
recorded no host recovery.

GPU reductions are not bitwise reproducible: repeating a solve changes the
result by about `1e-13`. Over a long chaotic run such differences grow, so two
recordings agree statistically and in their conservation diagnostics, not
frame by frame.

## Checks behind the settings

These checks ran without the KKT projection. Every check starts from the saved
profiles on the published mesh, and compares
final states with `scripts/reports/compare_showcase_states.py` (physical DG L2
norms on one mesh, shared sampled points across meshes).

### Poisson stabilization

`scripts/reports/probe_poisson_tau.py` solves the initial Poisson problem for
each τ and compares the velocity with a degree-8 host reference on the same
mesh (Numba assembly, 16-thread PyPardiso):

| τ | Velocity error, all | Within 0.05 of a wall | Interior | Double-outflow face measure | Iterations |
|---:|---:|---:|---:|---:|---:|
| 1000 | 2.6e-5 / 7.4e-5 | 7.0e-5 / 1.3e-4 | 8.2e-9 / 6.0e-9 | 1.2e-7 / 7.2e-7 | 29 / 31 |
| 100 | 2.6e-5 / 7.5e-5 | 7.1e-5 / 1.3e-4 | 8.3e-9 / 6.9e-9 | 1.2e-7 / 9.4e-7 | 29 / 30 |
| 10 | 2.6e-5 / 7.5e-5 | 7.2e-5 / 1.3e-4 | 8.3e-9 / 7.0e-9 | 1.2e-7 / 9.4e-7 | 30 / 31 |
| 1.913 (default) | 2.6e-5 / 7.5e-5 | 7.2e-5 / 1.3e-4 | 8.3e-9 / 7.0e-9 | 1.2e-7 / 9.4e-7 | 30 / 31 |
| 1 | 2.6e-5 / 7.5e-5 | 7.2e-5 / 1.3e-4 | 8.3e-9 / 7.0e-9 | 1.2e-7 / 9.4e-7 | 30 / 31 |
| 0.1 | 2.6e-5 / 7.5e-5 | 7.2e-5 / 1.3e-4 | 8.3e-9 / 7.0e-9 | 5.4e-3 / 2.0e-3 | 30 / 31 |

For τ from 1 to 1000 the velocity is the same to about 2%, and the solver
cost does not change; only τ = 0.1 degrades face compatibility. The error sits
almost entirely next to the walls, four orders of magnitude above the
interior. That is consistent with the weak corner singularities of a
straight-sided wall, which curved boundary elements would remove; it is not a
τ effect. Over a short run the choice is invisible too: τ = 1000 and the
default τ differ at t = 1 by 1.3e-6 in density (signed) and at
t = 0.5 by 1.8e-7 (positive), far below the time-step error.

### Time step

Each case was run with three halved time steps to a short endpoint:

| Case | dt | Density difference to dt/2 | Potential difference | Enstrophy loss | Energy drift | min ρ |
|---|---:|---:|---:|---:|---:|---:|
| Signed, t = 1 | 0.00625 | 2.98% | 0.0122% | 0.2573% | 1.1e-04 | – |
| Signed, t = 1 | 0.003125 | 1.18% | 0.0030% | 0.0603% | 2.4e-05 | – |
| Signed, t = 1 | 0.0015625 | – | – | 0.0138% | 5.4e-06 | – |
| Positive, t = 0.5 | 0.0015625 | 2.22% | 0.0019% | 0.0958% | 7.3e-06 | -0.814 |
| Positive, t = 0.5 | 0.00078125 | 0.75% | 0.0005% | 0.0198% | 1.5e-06 | -0.25 |
| Positive, t = 0.5 | 0.000390625 | – | – | 0.0041% | 3.4e-07 | -0.000348 |

The potential converges at second order. The density, dominated by the
positions of the smallest charges, shows an observed order of
1.33 (signed) and 1.56 (positive); its
Richardson-estimated error is 2.0% at the recorded signed step
and 0.4% (extrapolated) at the recorded positive step. Enstrophy loss and
energy drift fall by four for each halving, so at these times they are
time-discretization errors. In the positive case the negative undershoots fall
even faster: the extrapolated-velocity BDF2 step, not the high-order space
discretization, produced most of them. The recordings therefore use
**dt = 0.003125** (signed, half the step of the earlier GIF) and
**dt = 0.000390625** (positive, a quarter).

### Mesh

A coarser mesh (h = 0.007, 197,624 triangles, 1.4 times the published size) was run with the finest time steps. At 475,012 shared points its states differ from the published mesh by 0.36% in density and 0.0023% in potential (signed, t = 1) and by 0.029% and 0.0019% (positive, t = 0.5). The signed enstrophy loss changes from 0.0138% to 0.0161% and the positive minimum from −0.00035 to −0.0066. These differences bound the coarser mesh's error; at degree 6 the published mesh's own error is much smaller, and well below the time-step errors above.

These checks cover short intervals. They do not, and for a turbulent flow
cannot, establish pointwise convergence of the whole recorded trajectory;
the conservation diagnostics below are the long-time evidence.

## Recorded runs

| | Signed | Positive, with KKT |
|---|---:|---:|
| Time step | 0.003125 | 0.000390625 |
| Steps / final time | 5,080 / 15.9 | 16,384 / 6.4 |
| Frames / playback | 1,271 / 26 s (2× speed) | 1,025 / 21 s |
| Wall time, s per step (with rendering) | 63 min, 0.74 | 152 min, 0.56 (two segments) |
| GPU memory in use at finish | 29 GiB | 28 GiB |
| Change in total charge | 2.2e-13 | 2.4e-11 |
| Energy drift | 2.9e-4 | -4.1e-5 |
| Enstrophy loss | 29.9% | 8.2% |
| Density range at the end | -17.7 to 13.2 | -7e-16 to 21.1 |
| Negative charge (share of total) | – | below 1.3e-9 throughout |
| Published videos, light / dark | 5.4 / 5.0 MB | 6.3 / 6.4 MB |

Circulation `∫ρ`, energy `½∫|q_h|²` and enstrophy `½∫ρ²` are computed on the
device every frame. The signed charge balance is set by the profile, not by
round-off. Energy is nearly conserved while enstrophy decays: the filaments of
the forward enstrophy cascade reach the grid and are dissipated by upwinding,
while energy gathers in fewer, larger vortices. The KKT projection removes a
little energy and enstrophy whenever it acts: without it, the same positive run
drifted by +2.2e-6 in energy and lost 7.3% of its enstrophy, with the density
reaching -3.1 and 0.42% of the charge in negative regions.

### The positive run with KKT positivity

The run was recorded on the `positivity-kkt` branch (base `a11a741` with
uncommitted changes; the run metadata keeps their diff digest). The KKT
projection was committed afterwards as `5da4901`. It was stopped at step 5,273
and resumed from its checkpoint. Before the restart (t ≤ 2.06) the projection
constrained the density at element quadrature and lattice points; between
them a denser diagnostic sampling still found values down to -2.88 (t = 1.6).
From the restart on, a denser point set was added and no sampled value was
negative. No displayed pixel ever fell below the pink threshold. On that
branch:

```bash
python -m scripts.reports.record_gpu_showcase --name positive_kkt_c5_dt000390625 \
  --strength-mode positive --profile run_configs/guiding_center/profiles/readme_positive_c5_h005.npz \
  --cutoff 5 --h 0.005 --dt 0.000390625 --steps 16384 --every 16 --fps 48 --movie \
  --seconds 43200 --max-loss 0.9 --poisson-tau global \
  --density-positivity kkt --density-positivity-points quadrature+lattice
```

The continuation used `--name positive_kkt_c5_dt000390625_part2 --resume
outputs/readme_showcase/positive_kkt_c5_dt000390625.restart.npz --steps 11111
--density-positivity-points quadrature+lattice+dense` with the same other
settings.

## Record the videos

The recordings need the [forked GPU stack](forked_amgx_stack.md),
Matplotlib, Pillow and imageio-ffmpeg; they run without a display and do not
build AMGX. From the repository root, sequentially:

```bash
python -m scripts.reports.record_gpu_showcase --name signed_c5_dt003125 \
  --strength-mode balanced --profile outputs/readme_showcase/initial_balanced_c5_h005.npz \
  --cutoff 5 --h 0.005 --dt 0.003125 --steps 5080 --every 4 --fps 24 --movie \
  --seconds 14400 --max-loss 0.9 --poisson-tau global

python -m scripts.reports.record_gpu_showcase --name positive_c5_dt000390625 \
  --strength-mode positive --profile outputs/readme_showcase/initial_positive_c5_h005.npz \
  --cutoff 5 --h 0.005 --dt 0.000390625 --steps 16384 --every 16 --fps 48 --movie \
  --seconds 25200 --max-loss 0.9 --poisson-tau global
```

A missing `--profile` file is sampled on the run's mesh and saved. Both
recordings advance **0.3 physical-time units per playback second**: the signed
case draws every fourth step at 24 frames per second, the positive case every
sixteenth step at 48. The signed video is published at twice that speed. Frames are 1600 × 800 pixels with tick-free light-grey coordinate
boxes, wall outlines and fixed colorbars. Signed charge uses a symmetric
diverging map; positive density uses a sequential map from zero in which
values below −1% of the color scale appear pink, so lost positivity stays
visible. `--gif-mb` adds an optional byte-capped GIF; GIFs are not published.
`--save-rasters` archives the sampled display rasters so that
`scripts/reports/restyle_gpu_showcase.py` can repaint a run on the CPU; the
published runs omit it to keep per-frame data off disk.

Output lives in the ignored `outputs/readme_showcase/` directory: the MP4, the
latest PNG frame, JSON metadata with provenance (git revision and a digest of
uncommitted code, PyAMGX and AMGX fork revisions from the loaded library,
package versions), per-frame JSONL diagnostics, and one atomic full-precision
`<name>.restart.npz` endpoint checkpoint. `--resume <checkpoint>` validates the
mesh and every run parameter, takes one Euler startup step, then resumes BDF2.

Publish a finished run, or a run and its continuations:

```bash
python -m scripts.reports.publish_gpu_showcase outputs/readme_showcase/signed_c5_dt003125 \
  --as vortex_gas --poster-time 6 --speed 2
python -m scripts.reports.publish_gpu_showcase outputs/readme_showcase/positive_kkt_c5_dt000390625 \
  outputs/readme_showcase/positive_kkt_c5_dt000390625_part2 --as positive_density --poster-time 6
```

The tool checks each seam (a continuation's restart frame, off the frame
cadence, is dropped) and writes light and dark MP4s to the ignored
`outputs/readme_showcase/published/` for upload as GitHub attachments. Into
`docs/getting_started/media` it writes transparent light and dark posters with a
play button, and a JSON record with the run settings, diagnostics, provenance
and the digests of all four files, without local paths. The dark movie and the
posters redraw labels, colorbars and the fill from the rebuilt Matplotlib
layout (`scripts/reports/showcase_media.py`); field pixels keep their recorded
colors. The README shows the poster matching the reader's theme, and each
poster links to the issue where the videos play.

## Rerun the checks

```bash
python -m scripts.reports.probe_poisson_tau --name tau_probe_signed_c5 \
  --profile outputs/readme_showcase/initial_balanced_c5_h005.npz --cutoff 5

python -m scripts.reports.record_gpu_showcase --name chk_signed_global_h005_dt003125 \
  --strength-mode balanced --profile outputs/readme_showcase/initial_balanced_c5_h005.npz \
  --cutoff 5 --h 0.005 --dt 0.003125 --steps 320 --every 32 --seconds 7200 \
  --max-loss 0.9 --poisson-tau global

python -m scripts.reports.compare_showcase_states --output signed_dt.json --series \
  outputs/readme_showcase/chk_signed_global_h005_dt00625.restart.npz \
  outputs/readme_showcase/chk_signed_global_h005_dt003125.restart.npz \
  outputs/readme_showcase/chk_signed_global_h005_dt0015625.restart.npz
```

Vary `--dt`, `--steps`, `--h`, `--strength-mode` and `--poisson-tau` for the other
rows; `--pair A B` compares any two checkpoints, on one mesh or two.

## Preview the README and manual locally

```bash
python -m pip install markdown-it-py pygments matplotlib
python -m scripts.reports.render_docs_preview --output-dir /path/to/preview
```

Open `hybridge-readme-portable.html` in that directory. It links to
`hybridge-manual.html`; keep both files together when copying them. Posters,
figures and equations are embedded for offline viewing; posters open the
published videos when `outputs/readme_showcase/published/` holds them.
The theme follows the browser preference, with a small Auto/Light/Dark override.
Repository source links resolve against the checkout where the preview was built.
`python -m scripts.reports.make_readme_first_solve_figure` regenerates the
README's first-solve figure from the README's own code block.
