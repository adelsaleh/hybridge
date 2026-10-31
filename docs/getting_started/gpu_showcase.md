# BDF2 turbulence showcases

The [README](../../README.md) shows degree-6 HDG guiding-center calculations on
a five-lobed star with a circular island. The small
[showcase script](../../scripts/guiding_center/showcase.py) generates balanced
vortex gas or positive turbulence using the public Poisson and transport APIs.
The [standalone example](../../examples/gpu_vortex_gas.py) shows the same coupling
as a short script.

The model is

```text
∂t ρ + div(ρ u) = 0,   -Δφ = ρ,   u = (∂y φ, -∂x φ),   φ = 0 on both walls.
```

These are grounded conducting walls. With a fixed zero potential on the island,
its circulation can change; this differs from Euler flow with prescribed island
circulation. BDF2 uses one Euler startup step. The positive case applies the
library's density-positivity projector after the transport solve.

Install the [forked GPU stack](forked_amgx_stack.md) and optional plotting tools
from the [installation guide](installation.md). From the HYBRIDGE checkout:

```bash
.venv/bin/python -m scripts.guiding_center.showcase
.venv/bin/python -m scripts.guiding_center.showcase --case positive-turbulence
.venv/bin/python -m scripts.guiding_center.showcase --movie run_outputs/vortex-gas.mp4
```

Live display uses NVIDIA Holoviz. `--movie` writes an MP4 using the shared raster
recorder. `--mesh-size`, `--dt`, `--steps`, `--every`, and `--fps` control the
size and duration; `--help` is read-only. The defaults use the README geometry,
p=6, seed 17, and multiscale Gaussian charges. The signed and positive default
time steps are 0.003125 and 0.000390625, respectively. Solver fields stay on the
GPU; movie output downloads sampled rasters.

The existing README videos were produced by earlier recording campaigns. Their
[published signed metadata](media/vortex_gas.json) and
[published positive metadata](media/positive_density.json), posters, measurements,
and original scientific interpretation are unchanged. They were not rerun during
the repository reorganization. The positive recording used two segments with
different positivity sampling sets, as documented in its original metadata.

The complete recording and validation report (`../../../guiding-center-cg-hdg/docs/getting_started/gpu_showcase.md`),
checkpoint continuation, comparisons, and publication tools now live in the
guiding-center CG/HDG repository (`../../../guiding-center-cg-hdg/README.md`).
Its archived commands describe those recordings; the small example here does
not promise frame-by-frame reproduction of a chaotic trajectory.

The documentation preview remains available with
`python -m scripts.reports.render_docs_preview`. It reads the published posters
and metadata without launching a solver.
