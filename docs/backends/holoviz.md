# Holoviz plotting

`hybridge.io.holoviz` provides NVIDIA Holoviz panels for any scalar DG field.
The BDF2 turbulence showcase uses them for live updates; the static helpers
mirror the PyVista functions in `hybridge.io.plot`:

| PyVista (`hybridge.io.plot`) | Holoviz (`hybridge.io.holoviz`) |
|---|---|
| `plot_field` | `plot_field_holoviz` |
| `plot_fields` | `plot_fields_holoviz` (`share_clim` supported) |
| `plot_solution_comparison` | `plot_solution_comparison_holoviz` |

`HolovizScalarPanels(spaces, labels, ...)` is the shared viewer: one panel per
DG space, `update_fields(...)` or `submit(images, captions=...)` per frame, and
`wait_until_closed()`/`close()`. The static helpers block until the window is
closed when `show=True`, or return the open viewer with `show=False`.
Sampling cost follows `width x height` pixels per panel, not the mesh size,
which makes these helpers suitable for meshes with hundreds of thousands of
elements where refined PyVista meshes become slow.

Static-helper limits: one colormap per window (Holoviz applies a single
colour table), no colour bar — each panel caption shows its value range, which
downloads two scalars once — and the exact-solution callable is evaluated on
the host at the owned pixel centers and uploaded once. In the comparison, the
numerical and exact panels share one range and the error panel starts at zero.
The ADR runner selects it with `--plot --plot-backend holoviz`
(`--plot-width`/`--plot-height` per panel).

## BDF2 showcase

Run `python -m scripts.guiding_center.showcase` for live density and potential
panels. See the [showcase guide](../getting_started/gpu_showcase.md) for the
positive case and MP4 output. Holoscan is imported only for live display.

## Installation

The tested CUDA 13 environment has `holoscan-cu13==4.6.0` and
`cupy-cuda13x==14.2.0`. The Holoscan wheel includes Holoviz and the Python
bindings. CuPy connects device arrays through the CUDA array interface; no
separate connector or custom C++ extension is needed. This is NVIDIA Holoviz,
not the HoloViz browser plotting ecosystem.

For a fresh compatible Python environment, install prebuilt packages:

```bash
.venv/bin/python -m pip install --only-binary=:all: \
  'holoscan-cu13==4.6.0' 'cupy-cuda13x==14.2.0' matplotlib pillow
```

HYBRIDGE also declares these dependencies in its optional `holoviz` extra.
The machine needs an NVIDIA driver with CUDA/Vulkan interoperability and a
compatible CUDA 13 runtime/toolkit. See NVIDIA's
[installation guidance](https://docs.nvidia.com/holoscan/sdk-user-guide/faq/faq).
An X11 or Wayland display is required for a window; headless rendering still
requires the NVIDIA Vulkan driver.

## Viewer options

The shared `HolovizScalarPanels` API accepts panel dimensions, labels, color
limits and a frame-rate cap. `update_fields(...)` samples resident fields;
`close()` drains accepted frames. The BDF2 showcase exposes `--every` and
`--fps`; ADR uses `--plot --plot-backend holoviz` with its own plotting flags.
Use each runner's `--help` for its supported options.

## Device and sampling contract

Mesh geometry and basis tables are used once on the CPU to build a sparse
coefficient-to-pixel map. Changing DG coefficients are sampled with cuSPARSE
on the GPU. Each pixel belongs to one actual triangle; values are not averaged
across DG interfaces, and holes remain masked. The view and sampling grid are
fixed for the lifetime of the viewer.

Colour limits are device reductions. Vorticity uses a symmetric range fixed
from the initial image and `RdBu_r`; ordinary density uses per-frame limits
and `viridis`. Potential uses its own per-frame range and the same colour map
as the first panel. Only display indices are converted to FP32. Holoviz
applies a 256-entry colour table in its Vulkan shader. This first backend
provides scalar panels and mesh overlays; it does not provide a numeric
colour bar, field probing, or camera-dependent resampling.

Live updates do not download coefficients, sampled images, or colour-limit
scalars. Host fields can be uploaded, but fields already resident on another
CUDA device are rejected rather than staged through host memory. The viewer
uses the caller's current CUDA device. This transfer contract applies to the
plotting path; solver diagnostics retain their separate transfer behavior.

The renderer holds owned GPU images until its CUDA work finishes. Its queue
is bounded, and busy or rate-limited live updates are skipped. The first
frame waits for initialization errors; later previews are asynchronous.
Closing the runner drains accepted frames and releases the rendering graph.

Each distinct DG space needs a sampling map, shared between panels using the
same space. The map is limited to 512 MiB per space. For high polynomial order,
reduce pixel dimensions if this limit is exceeded. Pixel count, rather than
mesh-node count, controls the per-frame sampling work. Fixed-grid sampling
can miss structures smaller than a pixel.

## Static checks

`tests/test_holoviz.py` covers polynomial sampling, discontinuities, holes,
memory limits, device checks and explicit saving with synthetic data.
The archived manual GPU smoke and animation scripts are maintained with the
research viewer documentation (`../../../guiding-center-cg-hdg/docs/backends/holoviz.md`).

## Window focus and workspace switching

The source ticks independently of simulation updates. It re-presents the last
completed GPU image while the solver is busy and retries an unacknowledged
image if Holoviz skipped rendering while minimized. Holoviz returns without a
framebuffer in this case, so stopping all submissions until completion would
also prevent it from processing restoration.

Frame ownership travels through the graph in local Holoscan metadata. The
completion sink acknowledges that exact frame, so delayed retries cannot
release a newer frame or save the same screenshot twice. Idle redraws do not
re-sample DG fields or count as new simulation frames. Sampling readiness is
queried without blocking the rendering thread on the solver's CUDA stream.

This addresses the queue deadlock; it cannot guarantee responsiveness during
a driver stall or a native solver operation holding Python's GIL. Workspace
switching, minimizing/restoring, and close behavior still require a desktop
check. For unattended showcase output, use `--movie PATH`.
A plot-close failure during solver-error cleanup is logged separately and
does not replace the original solver exception.

## PyAMGX can starve the Python rendering operators

The local PyAMGX source used by this environment originally called
`AMGX_solver_setup` and `AMGX_solver_solve` while holding Python's GIL.
Holoscan `run_async()` uses another thread in the same process, so its Python
source and completion operators cannot execute during those calls. Repeating
frames fixes the minimized-frame queue deadlock but does not release this lock.

The source patch is retained in
[pyamgx-release-gil.patch](../../patches/pyamgx-release-gil.patch) and has been
applied to `../pyamgx-hdg-cuda13`. It releases the GIL around
native setup and both solve variants, then reacquires it for error handling.
The AMGX print callback explicitly acquires the GIL before calling Python.
This does not make concurrent access to the same AMGX handles safe: keep
solver resources and vectors on the solver thread.

The installed binary is unchanged until the user rebuilds the binding:

```bash
AMGX_DIR=../AMGX-hdg-cuda13 \
AMGX_BUILD_DIR=../AMGX-build-cuda13 \
.venv/bin/python -m pip install --no-build-isolation --no-deps --force-reinstall \
  ../pyamgx-hdg-cuda13
```

Restart the simulation process after rebuilding. This reuses the existing
AMGX library; it rebuilds only the Python binding. Neither compilation nor
workspace-switch testing has been performed as part of this edit. A Vulkan
presentation stall or GPU contention could still require separate diagnosis.

## Native minimized-window event loop

The upstream Holoviz GLFW implementation has a separate persistent-stall path:
`HolovizOp::compute` returns early when `WindowIsMinimized()` is true, whereas
`glfwPollEvents()` runs in `GLFWWindow::begin()`. Once minimized, the operator
can stop reaching the poll that would deliver the restore event and clear the
cached minimized flag. Periodic Python frame retries and releasing PyAMGX's
GIL do not repair this native control flow.

[holoviz-poll-events-when-minimized.patch](../../patches/holoviz-poll-events-when-minimized.patch)
adds a call to the existing, mutex-protected `begin()` event poll before
checking the minimized flag. It targets the Holoscan source file
`modules/holoviz/src/glfw_window.cpp`. This is a source patch for rebuilding
`libholoscan_viz`, not an applied modification to the installed wheel.
It has not been compiled or tested.

Apply from a compatible Holoscan SDK source checkout:

```bash
git apply patches/holoviz-poll-events-when-minimized.patch
```

The installed wheel statically incorporates GLFW without exporting its
`glfwPollEvents` symbol. Loading another system GLFW library from Python
would not process Holoviz's window and is not a valid workaround.

The showcase's `--movie PATH` uses the raster recorder without opening a
live window. A native Holoviz patch still requires separate desktop testing.

Sources:
- [Holoviz operator](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/src/operators/holoviz/holoviz.cpp)
- [GLFW window](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/modules/holoviz/src/glfw_window.cpp)

Live density and potential panels (including saved screenshots) show the displayed
simulation time `t`, time-step size `dt`, and accepted iteration `n/N`. The count
is the integration step, not the number of rendered frames. The showcase displays Density and Potential side by side.
Holoviz frame-specific captions use the SDK [dynamic input specifications](https://docs.nvidia.com/holoscan/sdk-user-guide/operators/visualization).

## Holoviz over SSH: session-local compatibility launcher

From a live `ssh -Y` terminal, preserve the original SSH `DISPLAY` and
`XAUTHORITY` and launch the command through:

```bash
.venv/bin/python -m hybridge.io.holoviz_ssh -- \
  .venv/bin/python -m scripts.guiding_center.showcase --every 1
```

The launcher checks the upstream display and its authorization before starting
the command, then creates an authenticated loopback proxy that hides only the
`NV-GLX` extension query. This enables the Holoviz presentation path verified in
our SSH session while retaining GPU rendering on the server. It also raises
the stack soft limit to 32 MiB when the hard limit permits. The proxy and its
private authorization file are removed when the command exits. Server tools
`xdpyinfo` (`x11-utils`) and `xauth` are required.

Do not reuse an old `DISPLAY=localhost:99` or temporary `XAUTHORITY` path:
an SSH reconnect can change the upstream display number, leaving an old proxy
pointing at a closed port. This produces GLFW initialization failures before
Vulkan device selection. Start from a fresh SSH terminal if the launcher says
the original display cannot be opened.

Optional movie capture uses Holoviz's rendered framebuffer output and streams
RGBA images into H.264 MP4 via `imageio-ffmpeg` (included in the `holoviz` extra).
`GuidingCenterHolovizPanels(movie_path=..., movie_fps=20)` enables it; `None`
keeps display-only updates free of framebuffer downloads and encoder processes.
Frames are acknowledged after encoding submission, preventing queue growth and
avoiding duplicate frames from idle redraws. No intermediate PNGs are needed.
The movie includes all panels and overlays. H.264 B-frames are disabled so decoding and presentation order agree. Normal
`close()` drains the encoder and writes a standard fast-start MP4 index for
player compatibility. An abrupt process kill before finalization can leave an
unplayable file.
This framebuffer API is available to application callers. The simple showcase
uses `--movie` with the separate raster recorder; `--fps` sets its frame rate.

Holoviz color ranges start with 10% padding and remain unchanged while all
sampled field values fit. An exceeded bound expands with fresh padding; ranges
never shrink. Density and potential track independent ranges, and signed
vorticity retains a symmetric range around zero. The shared
`hybridge.io.live.expanding_color_limits` helper supports NumPy and CuPy; Holoviz
keeps extrema and range updates on the device without copying field coefficients.
