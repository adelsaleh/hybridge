import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from hdgfem import rectangle_mesh
from hdgfem.io.plot import (
    coarse_mesh_polydata,
    map_element_plot_points,
    matplotlib_discontinuous_triangulation,
    plot_scalar_sample_panels_matplotlib,
    reference_plot_points,
    refined_sample_polydata,
)


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


def test_curved_geometry_map_controls_triangulation_overlay_and_output(tmp_path):
    import matplotlib.pyplot as plt

    mesh = rectangle_mesh(1, 1)
    reference_points = reference_plot_points(7)
    affine_points = mesh.map_reference_points(reference_points)

    def curved_map(points):
        mapped = mesh.map_reference_points(points).copy()
        mapped[..., 1] += 0.15 * (1.0 - points[:, 0] ** 2)[None, :]
        return mapped

    physical_points = map_element_plot_points(
        mesh,
        reference_points,
        geometry_map=curved_map,
    )
    assert np.max(np.abs(physical_points - affine_points)) > 0.1

    triangulation = matplotlib_discontinuous_triangulation(
        mesh,
        reference_points,
        geometry_map=curved_map,
    )
    np.testing.assert_allclose(triangulation.x, physical_points[..., 0].reshape(-1))
    np.testing.assert_allclose(triangulation.y, physical_points[..., 1].reshape(-1))

    values = physical_points[..., 0] + physical_points[..., 1]
    output = tmp_path / "curved-elements.png"
    figure = plot_scalar_sample_panels_matplotlib(
        mesh,
        (("curved", reference_points, values, {"geometry_map": curved_map}),),
        show=False,
        output=output,
        mesh_edge_resolution=9,
    )
    try:
        assert output.is_file()
        assert output.stat().st_size > 0
        assert len(figure.axes[0].collections) >= 2
    finally:
        plt.close(figure)

    with pytest.raises(ValueError, match="geometry_map must return"):
        map_element_plot_points(
            mesh,
            reference_points,
            geometry_map=lambda points: np.zeros((points.shape[0], 2)),
        )


def test_curved_geometry_map_builds_refined_pyvista_surface_and_wireframe():
    pytest.importorskip("pyvista")

    mesh = rectangle_mesh(1, 1)
    reference_points = reference_plot_points(6)

    def curved_map(points):
        mapped = mesh.map_reference_points(points).copy()
        mapped[..., 1] += 0.1 * (1.0 - points[:, 0] ** 2)[None, :]
        return mapped

    physical_points = curved_map(reference_points)
    values = physical_points[..., 0] - physical_points[..., 1]
    refined = refined_sample_polydata(
        mesh,
        reference_points,
        values,
        scalar_name="u",
        geometry_map=curved_map,
    )
    np.testing.assert_allclose(refined.points[:, :2], physical_points.reshape(-1, 2))
    np.testing.assert_allclose(refined.point_data["u"], values.reshape(-1))

    edge_resolution = 8
    wireframe = coarse_mesh_polydata(
        mesh,
        geometry_map=curved_map,
        edge_resolution=edge_resolution,
    )
    assert wireframe.n_lines == 3 * mesh.num_tri
    assert wireframe.n_points == 3 * edge_resolution * mesh.num_tri
