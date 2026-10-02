"""Verify persistent colorbars, missing pixels, and owning CPU captures."""

import numpy as np
import pytest

from hdgfem.io.matplotlib import MatplotlibRasterPanels


@pytest.mark.parametrize("background", ("white", "black"))
def test_updates_preserve_masks_scales_and_colorbars(background):
    """Changing data leaves physical scales and colorbar artists unchanged."""
    import matplotlib

    matplotlib.use("Agg")
    values = np.array([[np.nan, 1.], [2., 3.]])
    panels = [("Density", values, {"clim": (0., 4.), "cmap": "viridis"}),
              ("Potential", values/10, {"clim": (0., .4), "cmap": "cividis"})]
    with MatplotlibRasterPanels(panels, (0., 1., 0., 1.), size=(600, 300),
                                font_size=8, background=background) as viewer:
        axes = tuple(viewer.figure.axes)
        original = viewer.capture()
        viewer.update((values*2, values/20), caption="t = 1")
        updated = viewer.capture()
        assert tuple(viewer.figure.axes) == axes
        assert len(axes) == 4
        assert viewer.images[0].get_clim() == (0., 4.)
        assert viewer.images[1].get_clim() == (0., .4)
        assert np.ma.getmaskarray(viewer.images[0].get_array())[0, 0]
        assert original.shape == updated.shape == (300, 600, 4)
        assert original.dtype == updated.dtype == np.uint8
        assert np.all(original[..., 3] == 255)
        np.testing.assert_array_equal(original[0, 0, :3], 255)
        x, y = viewer.images[0].axes.transData.transform((.25, .75))
        pixel = original[original.shape[0]-1-int(y), int(x), :3]
        np.testing.assert_array_equal(pixel, 255 if background == "white" else 0)
        assert not np.shares_memory(original, updated)
        assert not np.array_equal(original, updated)
        with pytest.raises(ValueError, match="shape changed"):
            viewer.update((np.zeros((1, 1)), values))
    with pytest.raises(RuntimeError, match="closed"):
        viewer.capture()
