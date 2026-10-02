# Reproducing the GPU turbulence showcase

The [README](../../README.md) shows two real degree-6 HDG calculations on the
same star-shaped domain with a circular island. The
[commented example](../../examples/gpu_vortex_gas.py) contains the mesh,
initial field, coupled solves, and live Holoviz plot. The recording workflow
adds Matplotlib colorbars, measurements, and file output.

## Problem and resolution

The coupling is `−Δφ = ρ`, `q = −∇φ`, `u = (−q_y, q_x)`, and
`∂tρ + div(ρu) = 0`. Constant-step semi-implicit BDF2 starts with one
semi-implicit Euler step. Both walls have zero potential; transport uses the
zero-flux boundary mode. The island has no independently prescribed circulation.
Signed density describes Euler vorticity; initially positive density gives the
guiding-center interpretation of the same equations.

| Parameter | Both recordings |
|---|---|
| Outer wall | `r(θ) = 1 + 0.35 cos(5θ)`, 500 boundary points |
| Island | Centered circle of radius `0.3` |
| Mesh | Gmsh size `0.005`; **360,379 triangles** |
| Approximation | Dubiner basis, **p = 6** |
| Scalar unknowns | **10,090,612** |
| Interior scalar trace unknowns | **3,774,995** |
| Initial blobs | **960**, counts `(512, 256, 128, 64)` |
| Core widths | `(0.008, 0.016, 0.024, 0.032)` |
| Profile | Amplitude `4`, seed `17`, no uniform background |

`MeshDomain` samples triangle interiors in proportion to area, with clearance
from both walls. Gaussian supports are cut off at eight core widths. The
signed profile balances positive and negative strengths at each scale; the
positive profile uses positive strengths throughout. Saved centers and strengths
are reused for refinement checks. The unlimited high-order discretization can
produce negative undershoots from positive initial data. These values remain
in the numerical fields and diagnostics; no clipping or flux limiter is applied.

The Poisson solver uses raw-CUDA face-BSR assembly and native face-block hp
multigrid with an AMGX coarse solve. Transport uses fused raw-CUDA BSR assembly
and AMGX. DG velocities select `conflict-averaged-upwind` by default, using the
existing corrected face assembly to repair conflicting outflow classifications.
The positive case reuses the guiding-center runner's bounded Jacobi, FGMRES, and residual
correction retries. If those fail to converge, an unscaled nonsymmetric host
PyPardiso solve recovers the same corrected discrete problem. Finite-element
assembly stays in the fused raw-CUDA kernel; only the assembled trace system
is downloaded to the host. The solve uses a verified
16-thread MKL limit and measured CPU use. The accepted trace is uploaded for
GPU reconstruction before either time history is committed. Recovery events,
residuals, thread count, and CPU use are recorded in metadata. The signed case
retains its original linear-solver policy without the host recovery.

Both request relative tolerance `1e-9` and absolute tolerance `1e-10`.
Fields and warm-start traces remain on the GPU. The recording downloads sampled
rasters for Matplotlib, with independent, fixed colorbars for density and
potential. The live example uses Holoviz and independent panel ranges.

## Time-step and mesh checks

The selected steps are **0.00625 for signed vorticity** and **0.0015625 for
initially positive density**. Each was compared with a half-step reference on
the displayed mesh, using exactly the same initial profile. These checks used
the original upwind policy; they predate the corrected positive-case recording:

| Case / comparison endpoint | Density relative L2 difference | Potential relative L2 difference |
|---|---:|---:|
| Signed, `t = 1`, `dt = 0.00625` versus `0.003125` | 3.6673% | 0.01360% |
| Positive, `t = 0.5`, `dt = 0.0015625` versus `0.00078125` | 3.1802% | 0.002636% |

These are physical volume L2 norms. At the selected steps, the short signed
check lost **0.3357% enstrophy** with **+0.01462% energy drift**; the positive
check lost **0.1551% enstrophy** with **+0.0009363% energy drift**.
Here enstrophy is `½∫ρ²`, energy is `½∫|q_h|²`, and circulation is `∫ρ`.

A separate spatial comparison uses the half-step reference time steps and a
coarser mesh (`h = 0.006`, 264,421 triangles). At the same endpoints, differences
sampled at 100,000 common area-uniform fluid points were **0.09881% / 0.0004269%**
for signed density / potential and **0.009302% / 0.0006889%** for positive density /
potential. These sampled norms support the chosen mesh for this demonstration.
The checks cover short intervals; they do not establish convergence of the
entire extended turbulent trajectory.

## Record both GIFs

Use the [forked GPU stack](forked_amgx_stack.md), plus Matplotlib and Pillow.
The recordings run without a desktop display. They use an NVIDIA RTX PRO 5000
Blackwell with 48 GB VRAM, CUDA 13, and the existing AMGX 2.5 build. These
commands do not build AMGX.

From the repository root, run the two cases sequentially:

```bash
python -m scripts.reports.record_gpu_showcase \
  --h 0.005 --dt 0.00625 --steps 100000 --every 2 \
  --seconds 14400 --max-loss 0.5 --gif-mb 200 --fps 24 \
  --movie --name signed_200mb

python -m scripts.reports.record_gpu_showcase \
  --strength-mode positive --h 0.005 --dt 0.0015625 \
  --steps 100000 --every 4 --seconds 14400 --max-loss 0.5 \
  --gif-mb 200 --fps 48 --movie --name positive_200mb
```

For refinement studies, pass `--profile PATH` to reuse the saved profile rather
than resampling centers on a different mesh. Use a fresh `--name` for each run.

The recorder stops before a real frame would exceed **200,000,000 bytes per
GIF**. No padding, duplicated states, or interpolated states are added to fill
the budget. Additional time and enstrophy limits guard the calculation; the
metadata records which limit ended the run. The measured endpoint that would
exceed a limit is excluded from the animation, so use `last_rendered` for its
final displayed diagnostics.

Both animations advance approximately **0.30 physical-time units per playback
second**, matching the original README GIF. Solver time steps and playback
speed are independent: the signed case captures every two steps at 24 FPS;
the positive case captures every four steps at 48 FPS. GIF timing alternates
between adjacent centisecond durations to preserve the requested average rate.

Output lives in the ignored `outputs/readme_showcase/` directory: a GIF, optional
MP4, latest PNG, JSON metadata, scalar JSONL diagnostics, saved initial profile,
and optional sampled raster archives. The rasters permit recoloring without
repeating the GPU solves. Each published GIF is 1600 × 800 pixels with two
fields, physical boundaries, and colorbars. The accompanying evidence files
record actual duration, runtime, size, conservation diagnostics, and calibration.

The host recovery was exercised separately on 40,715 triangles at p = 6,
after deliberately triggering the recovery branch. Its field differed from
the accepted GPU result by `8.53e-11` in relative coefficient norm. MKL reported
16 threads; observed peak process CPU use was approximately 16 cores. Both
field and trace returned as device arrays, and the physical residual was
`4.57e-19` against a `1e-10` target.

The published signed recording ends at `t = 15.8625` (final solver state
`t = 15.875`), took 32.4 minutes, and occupies 199.92 MB. The positive recording
ends at `t = 6.41875` (final solver state `t = 6.425`), took 48.8 minutes, and
occupies 199.94 MB. Each run saved one final endpoint checkpoint; these extended
runs did not archive per-frame rasters.

## Change the animation background without rerunning the simulation

Both published animations use black coordinate boxes, with white outer
figure margins and black labels/colorbar text. Physical wall outlines remain
visible against the coordinate-box background. Density/potential color scales and frame
timing are unchanged. For earlier runs recorded with `--save-rasters`, restyle
the archived display samples on the CPU:

```bash
python -m scripts.reports.restyle_gpu_showcase \
  outputs/readme_showcase/positive_corrected \
  outputs/readme_showcase/positive_dark --background black --movie

python -m scripts.reports.restyle_gpu_showcase \
  outputs/readme_showcase/signed_final \
  outputs/readme_showcase/signed_dark --background black --movie
```

The command rebuilds only geometry for boundary outlines. It performs no
finite-element assembly, linear solves, or time integration. The real recorded
states are rendered from saved float32 display samples, and the resulting GIF
still respects the 100 MB ceiling.

Recordings always save one atomic `<name>.restart.npz` checkpoint at the final
accepted time, including on a handled interruption. It contains full-precision
endpoint coefficients, mesh and solver traces, with no sequence of DG states or
previous-time coefficients. `--resume <checkpoint>` validates the mesh and run
parameters, takes one Euler startup step, then resumes BDF2. Frames are streamed
to GIF/MP4 encoders; `--save-rasters` is optional and is omitted for the extended
200 MB recordings to avoid retaining per-frame snapshots.


## Preview the README and manual locally

```bash
python -m pip install markdown-it-py pygments matplotlib pillow imageio-ffmpeg
python -m scripts.reports.render_docs_preview --output-dir /path/to/preview
```

Open `hdgfem-readme-portable.html` in that directory. It links to
`hdgfem-manual.html`; keep both files together when copying them. Videos and
equations are embedded for offline viewing. Videos start paused and do not loop.
The theme follows the browser preference, with a small Auto/Light/Dark override.
Repository source links resolve against the checkout where the preview was built.
