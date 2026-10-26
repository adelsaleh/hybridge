# Plotting ownership

Reusable rendering belongs to `hybridge.io`. The runtime's PyVista adapter supplies
case labels and color policies to `PyVistaFieldPanels`; the package owns mesh
sampling, overlays, in-place scalar updates, linked views, headless rendering,
and screenshot output. Vorticity uses fixed symmetric initial limits, while
density and potential use their current robust ranges. The existing
`hybridge.io.holoviz` backend retains GPU sampling and asynchronous rendering.

Temporal field comparisons use `plot_scalar_raster_panels_matplotlib` with
`RasterGeometry` bounds and `scalar_color_limits`. Mesh holes stay masked, the
two vorticity panels share a symmetric scale, and their difference has its own
scale. `VorticityRaster` already uses the package's raster geometry and sampling
operators. Diagnostic histories, growth fits, and convergence-study labels
remain in the scripts because they depend on those studies' observables.
The independent DOLFINx reference retains its own field-to-VTK adapter.
Incremental diagnostic JSONL/CSV writing is shared through `hybridge.io.records`.

Other package clients can construct a live viewer without importing a runner:

```python
from hybridge.io import PyVistaFieldPanels

viewer = PyVistaFieldPanels(
    [("Scalar", field, {"symmetric_clim": True, "fixed_clim": True,
                        "robust_percentile": 100.0})],
    off_screen=True, screenshot_dir="frames",
)
try:
    viewer.update([field], step=0, time_value=0.0)
finally:
    viewer.close()
```

PyVista and Matplotlib are imported only when their rendering helpers are used.
The PyVista path uses the normal host `DGField.values_at_ref` interface; use
Holoviz for GPU-resident rendering without downloading field coefficients.
