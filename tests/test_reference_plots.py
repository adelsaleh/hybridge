"""Figures of ``hybridge.io.reference_plots`` render without a display and write non-empty files."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from hybridge.core.simplex.nodes import simplex_nodes  # noqa: E402
from hybridge.core.simplex.reference import reference_simplex  # noqa: E402
from hybridge.io import reference_plots as rp  # noqa: E402


def _save(figure, path) -> int:
    """Save ``figure`` to ``path`` and return the file size in bytes."""
    figure.savefig(path)
    return path.stat().st_size


def test_node_figures(tmp_path):
    """Triangle nodes, tetrahedron nodes and the unfolded tetrahedron faces draw on one figure."""
    figure = rp.new_figure(6.0, 2.0)
    rp.plot_triangle_nodes(figure.add_subplot(1, 3, 1), simplex_nodes("fekete", 2, 5), title="triangle")
    points = simplex_nodes("warp_blend", 3, 4)
    rp.plot_tetrahedron_nodes(figure.add_subplot(1, 3, 2, projection="3d"), points, title="tetrahedron")
    simplex = reference_simplex(3)
    ax = figure.add_subplot(1, 3, 3)
    rp.plot_tetrahedron_net(ax, simplex.barycentric(points), simplex.face_vertices, title="faces")
    assert len(figure.axes) == 3
    assert len(ax.collections) == 4  # one scatter per face
    assert _save(figure, tmp_path / "nodes.png") > 1000


def test_tetrahedron_net_folds_each_face_across_the_bottom_face():
    """Each side face of the net shares its two bottom vertices with the central triangle."""
    centre, positions = rp.tetrahedron_net(reference_simplex(3).face_vertices)
    for face, corners in zip(reference_simplex(3).face_vertices, positions):
        for vertex, corner in zip(face, corners):
            if vertex != 3:
                assert np.allclose(corner, centre[vertex])


def test_matrix_curve_function_and_table_figures(tmp_path):
    """Matrix magnitudes, curves, a function on the triangle and a text table render."""
    from hybridge.core.quadrature import reference_plot_points

    figure = rp.new_figure(8.0, 2.0)
    image = rp.plot_matrix_magnitude(figure.add_subplot(1, 4, 1), np.diag([1.0, 0.0, 1e-3]), title="M")
    assert np.isnan(np.ma.getdata(image.get_array())[1, 1])  # zero entry left blank
    rp.plot_series(figure.add_subplot(1, 4, 2), [1, 2, 3], {"a": [1, 10, 100], "b": [2, 3, 4]}, title="curves")
    points = reference_plot_points(20)
    rp.plot_triangle_function(figure.add_subplot(1, 4, 3), points, points[:, 0] * points[:, 1], title="xy")
    rp.plot_text_table(figure.add_subplot(1, 4, 4), [["1", "2"]], ["a", "b"])
    assert _save(figure, tmp_path / "misc.png") > 1000


def test_more_than_eight_series_are_refused():
    """A curve plot takes at most eight series (one categorical color each)."""
    figure = rp.new_figure(2.0, 2.0)
    with pytest.raises(ValueError, match="at most"):
        rp.plot_series(figure.add_subplot(1, 1, 1), [0, 1], {str(k): [1, 2] for k in range(9)})
