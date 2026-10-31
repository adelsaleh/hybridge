# BDF2 turbulence showcases

This small example couples the public Poisson and transport solvers on the
star-shaped domain with an island shown in the main README. It supports a
balanced vortex gas and nonnegative turbulence, with the library's positivity
projector for the latter. BDF2 uses one Euler startup step.

From a checkout with the GPU and plotting dependencies installed:

```bash
.venv/bin/python -m scripts.guiding_center.showcase
.venv/bin/python -m scripts.guiding_center.showcase --case positive-turbulence
.venv/bin/python -m scripts.guiding_center.showcase --movie run_outputs/vortex-gas.mp4
```

The default degree, mesh size, Gaussian blobs, and time steps match the README
showcases. `--mesh-size`, `--dt`, and `--steps` allow smaller demonstrations;
`--help` only lists options. The existing README videos retain their original
recording metadata. Full campaigns, convergence studies, alternative time
schemes, checkpoint recovery, and publication workflows live in the separate
`guiding-center-cg-hdg` repository.
