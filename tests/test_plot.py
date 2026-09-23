import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from hdgfem import rectangle_mesh
from hdgfem.io.plot import plot_scalar_sample_panels_matplotlib, reference_plot_points


def _sample_values(mesh, reference_points, offset=0.0, scale=1.0):
    base = np.linspace(-1.0, 1.0, reference_points.shape[0], dtype=np.float64)
    return np.broadcast_to(offset + scale * base[None, :], (mesh.num_tri, reference_points.shape[0]))


def _expected_robust_clim(values, *, percentile=95.0, zero_min=False):
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    if zero_min:
        minimum = 0.0
        maximum = float(np.percentile(finite, percentile))
    elif percentile == 100.0:
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
    else:
        tail = 0.5 * (100.0 - percentile)
        minimum, maximum = (float(v) for v in np.percentile(finite, (tail, 100.0 - tail)))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return (0.0, 1.0)
    if minimum == maximum:
        maximum = minimum + 1.0
    return (minimum, maximum)


def test_matplotlib_panel_colorbars_honor_explicit_clim():
    import matplotlib.pyplot as plt

    mesh = rectangle_mesh(1, 1)
    reference_points = reference_plot_points(5)
    first_values = _sample_values(mesh, reference_points, offset=0.0, scale=2.0)
    second_values = _sample_values(mesh, reference_points, offset=10.0, scale=1.0)

    fig = plot_scalar_sample_panels_matplotlib(
        mesh,
        (
            ("first", reference_points, first_values, {"clim": (-3.0, 3.0)}),
            ("second", reference_points, second_values, {"clim": (8.0, 12.0)}),
        ),
        levels=17,
        share_clim=False,
        show=False,
    )

    try:
        colorbar_axes = fig.axes[2:]
        assert len(colorbar_axes) == 2
        np.testing.assert_allclose(colorbar_axes[0].get_ylim(), (-3.0, 3.0))
        np.testing.assert_allclose(colorbar_axes[1].get_ylim(), (8.0, 12.0))
    finally:
        plt.close(fig)


def test_matplotlib_shared_colorbar_matches_combined_values():
    import matplotlib.pyplot as plt

    mesh = rectangle_mesh(1, 1)
    reference_points = reference_plot_points(5)
    first_values = _sample_values(mesh, reference_points, offset=-4.0, scale=1.0)
    second_values = _sample_values(mesh, reference_points, offset=6.0, scale=2.0)

    fig = plot_scalar_sample_panels_matplotlib(
        mesh,
        (
            ("first", reference_points, first_values),
            ("second", reference_points, second_values),
        ),
        levels=17,
        share_clim=True,
        show=False,
    )

    try:
        colorbar_axes = fig.axes[2:]
        assert len(colorbar_axes) == 1
        expected = _expected_robust_clim(np.concatenate((first_values.reshape(-1), second_values.reshape(-1))))
        np.testing.assert_allclose(colorbar_axes[0].get_ylim(), expected)
    finally:
        plt.close(fig)


def test_matplotlib_unshared_colorbars_default_to_each_panel_values():
    import matplotlib.pyplot as plt

    mesh = rectangle_mesh(1, 1)
    reference_points = reference_plot_points(5)
    bounded_exact = _sample_values(mesh, reference_points, offset=0.0, scale=1.0)
    overshot_numerical = _sample_values(mesh, reference_points, offset=10.0, scale=14.0)

    fig = plot_scalar_sample_panels_matplotlib(
        mesh,
        (
            ("numerical", reference_points, overshot_numerical),
            ("exact", reference_points, bounded_exact, {"show_mesh": False}),
        ),
        levels=17,
        share_clim=False,
        show=False,
    )

    try:
        colorbar_axes = fig.axes[2:]
        assert len(colorbar_axes) == 2
        np.testing.assert_allclose(colorbar_axes[0].get_ylim(), _expected_robust_clim(overshot_numerical))
        np.testing.assert_allclose(colorbar_axes[1].get_ylim(), _expected_robust_clim(bounded_exact))
    finally:
        plt.close(fig)


def test_gpu_diffusion_matplotlib_exact_panel_keeps_own_color_range():
    import matplotlib.pyplot as plt

    from hdgfem.io.comparison import plot_sampled_solution_comparison

    mesh = rectangle_mesh(1, 1)
    reference_points = reference_plot_points(5)
    numerical_values = _sample_values(mesh, reference_points, offset=10.45, scale=13.55)
    exact_values = _sample_values(mesh, reference_points, offset=0.0, scale=1.0)
    post_values = _sample_values(mesh, reference_points, offset=0.0, scale=1.25)

    def exact_solution(x, y):
        return x + 0.5 * y

    _, _, exact_display = __import__("hdgfem.io.plot", fromlist=["sample_callable_on_elements"]).sample_callable_on_elements(
        mesh,
        exact_solution,
        reference_points=reference_points,
    )

    fig = plot_sampled_solution_comparison(
        mesh,
        exact_solution,
        {
            "reference_points": reference_points,
            "numerical_values": numerical_values,
            "exact_values_for_error": exact_values,
        },
        numerical_resolution=5,
        exact_resolution="same",
        polynomial_order=1,
        postprocessed_samples={
            "reference_points": reference_points,
            "postprocessed_values": post_values,
            "exact_values_for_error": exact_values,
        },
        show=False,
    )

    try:
        colorbar_axes = fig.axes[4:]
        assert len(colorbar_axes) == 4
        np.testing.assert_allclose(colorbar_axes[0].get_ylim(), _expected_robust_clim(numerical_values))
        np.testing.assert_allclose(colorbar_axes[1].get_ylim(), _expected_robust_clim(post_values))
        np.testing.assert_allclose(colorbar_axes[2].get_ylim(), _expected_robust_clim(exact_display))
        np.testing.assert_allclose(
            colorbar_axes[3].get_ylim(),
            _expected_robust_clim(np.abs(post_values - exact_values), zero_min=True),
        )
    finally:
        plt.close(fig)


def test_raster_panels_preserve_holes_orientation_and_separate_difference_scale():
    import matplotlib.pyplot as plt
    from hdgfem.io import plot_scalar_raster_panels_matplotlib

    left = np.array([[np.nan, 2.0], [-4.0, 1.0]])
    right = left + .25
    bounds = (-2., 3., -5., 7.)
    fig = plot_scalar_raster_panels_matplotlib(
        (("first", left, {"clim": (-4., 4.)}),
         ("second", right, {"clim": (-4., 4.)}),
         ("difference", left - right)),
        bounds, symmetric=True, robust_percentile=100, share_clim=False, show=False,
    )
    try:
        assert len(fig.axes) == 6
        for axis, expected, limits in zip(fig.axes[:3], (left, right, left-right),
                                          ((-4., 4.), (-4., 4.), (-.25, .25))):
            image = axis.images[0]
            np.testing.assert_array_equal(image.get_array().mask, np.isnan(expected))
            np.testing.assert_allclose(image.get_array().compressed(), expected[np.isfinite(expected)])
            assert image.origin == "upper"
            assert image.get_interpolation() == "nearest"
            assert tuple(image.get_extent()) == bounds
            assert image.get_clim() == limits
            assert tuple(axis.get_ylim()) == bounds[2:]
    finally:
        plt.close(fig)


def test_raster_shared_colorbar_and_masked_outliers():
    import matplotlib.pyplot as plt
    from hdgfem.io import plot_scalar_raster_panels_matplotlib

    first = np.ma.array([[2., 1.e9]], mask=[[False, True]])
    second = np.array([[-5., np.inf]])
    fig = plot_scalar_raster_panels_matplotlib(
        [("first", first), ("second", second)], (0., 2., 0., 1.),
        symmetric=True, robust_percentile=100, show=False,
    )
    try:
        assert len(fig.axes) == 3
        for axis in fig.axes[:2]:
            assert axis.images[0].get_clim() == (-5., 5.)
        np.testing.assert_allclose(fig.axes[-1].get_ylim(), (-5., 5.))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("panels, bounds, message", [
    ([], (0, 1, 0, 1), "at least one panel"),
    ([("bad", [1, 2])], (0, 1, 0, 1), "two-dimensional"),
    ([("bad", [[1]])], (1, 0, 0, 1), "bounds"),
])
def test_raster_panels_reject_invalid_input(panels, bounds, message):
    from hdgfem.io import plot_scalar_raster_panels_matplotlib

    with pytest.raises(ValueError, match=message):
        plot_scalar_raster_panels_matplotlib(panels, bounds, show=False)
