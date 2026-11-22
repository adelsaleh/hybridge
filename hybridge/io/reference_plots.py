"""Matplotlib figures of reference-simplex data: node sets, basis functions, matrices and curves.

Every function draws on a Matplotlib ``Axes`` given by the caller and takes
plain arrays, so the figures do not depend on how the data was computed. The
report ``scripts/reports/make_reference_simplex_report.py`` assembles them.

Colors follow one fixed scheme: series of a curve plot take the categorical
colors in a fixed order (at most eight per axes, never cycled), magnitudes such
as ``log10 |a_ij|`` use one blue ramp from light (small) to dark (large), and
signed fields use a blue-gray-red scale centered at zero.
"""

from __future__ import annotations

import numpy as np

# Categorical colors, used in this order for the series of one axes.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# Marker shapes paired with the categorical colors, so that series differ by more than color.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
# One-hue blue ramp for magnitudes, light (small) to dark (large).
SEQUENTIAL_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
# Diverging scale for signed values: blue (negative), gray (zero), red (positive). The red
# stops have the OKLab lightness of the matching blue stops (0.48 and 0.76), so both arms
# change lightness at the same rate.
DIVERGING = ("#1c5cab", "#86b6ef", "#f0efec", "#e49d97", "#ad1722")
TEXT_SECONDARY = "#52514e"
OUTLINE = "#52514e"


def new_figure(width: float, height: float):
    """Return a Matplotlib ``Figure`` of ``width x height`` inches attached to an Agg canvas (no pyplot)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def sequential_colormap():
    """Return the blue ramp of ``SEQUENTIAL_BLUE`` as a continuous Matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("hybridge_blue", SEQUENTIAL_BLUE)


def diverging_colormap():
    """Return the blue-gray-red scale of ``DIVERGING`` as a continuous Matplotlib colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("hybridge_diverging", DIVERGING)


def draw_simplex_outline(ax, vertices: np.ndarray, edges) -> None:
    """Draw the edges ``(a, b)`` of a simplex with vertex coordinates ``vertices`` (2D or 3D axes)."""
    vertices = np.asarray(vertices, dtype=float)
    for a, b in edges:
        segment = vertices[[a, b]]
        ax.plot(*segment.T, color=OUTLINE, linewidth=0.8, zorder=1)


def plot_triangle_nodes(ax, nodes: np.ndarray, *, title: str = "", color: str = CATEGORICAL[0]) -> None:
    """Scatter the nodes ``(n, 2)`` inside the reference triangle ``(-1,-1), (1,-1), (-1,1)``."""
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    draw_simplex_outline(ax, vertices, ((0, 1), (1, 2), (2, 0)))
    nodes = np.asarray(nodes, dtype=float)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=12, color=color, edgecolors="white", linewidths=0.5, zorder=2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)


def plot_tetrahedron_nodes(ax3d, nodes: np.ndarray, *, title: str = "", color: str = CATEGORICAL[0]) -> None:
    """Scatter the nodes ``(n, 3)`` of the reference tetrahedron on 3D axes, with its six edges."""
    vertices = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    draw_simplex_outline(ax3d, vertices, ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)))
    nodes = np.asarray(nodes, dtype=float)
    ax3d.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=8, color=color, depthshade=True)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=18, azim=-60)
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.set_title(title)


def tetrahedron_net(face_vertices) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the 2D positions of the unfolded faces of a tetrahedron.

    The face that does not contain vertex 3 is drawn as a central equilateral
    triangle; each other face is folded out across the edge it shares with it.
    Returns the central-triangle corners ``(3, 2)`` (for vertices 0, 1, 2) and, for
    every face in ``face_vertices`` order, the net positions ``(3, 2)`` of its three
    listed vertices.
    """
    centre = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    positions = []
    for face in face_vertices:
        if 3 not in face:
            positions.append(centre[list(face)])
            continue
        a, b = (v for v in face if v != 3)
        opposite = ({0, 1, 2} - {a, b}).pop()
        midpoint = 0.5 * (centre[a] + centre[b])
        apex = 2.0 * midpoint - centre[opposite]
        positions.append(np.array([centre[v] if v != 3 else apex for v in face]))
    return centre, positions


def plot_tetrahedron_net(ax, barycentric: np.ndarray, face_vertices, *, title: str = "",
                         color: str = CATEGORICAL[0], tolerance: float = 1e-10) -> None:
    """Draw the unfolded faces of the tetrahedron with the nodes that lie on each face.

    ``barycentric`` holds the barycentric coordinates ``(n, 4)`` of the nodes. A node lies on face ``f`` when its coordinate of the vertex not in
    ``face_vertices[f]`` is below ``tolerance``; it is drawn at the same barycentric
    position inside the unfolded face. Interior nodes are not drawn.
    """
    _, positions = tetrahedron_net(face_vertices)
    barycentric = np.asarray(barycentric, dtype=float)
    for face, corners in zip(face_vertices, positions):
        closed = np.vstack((corners, corners[:1]))
        ax.plot(closed[:, 0], closed[:, 1], color=OUTLINE, linewidth=0.8, zorder=1)
        missing = ({0, 1, 2, 3} - set(face)).pop()
        on_face = barycentric[:, missing] < tolerance
        weights = barycentric[on_face][:, list(face)]
        points = weights @ corners
        ax.scatter(points[:, 0], points[:, 1], s=9, color=color, edgecolors="white", linewidths=0.4, zorder=2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)


def plot_matrix_magnitude(ax, matrix: np.ndarray, *, title: str = "", vmin: float = -8.0):
    """Show ``log10(|a_ij| / max |a|)`` of a matrix on the blue ramp; entries below ``10**vmin`` of the largest stay blank.

    The scale runs from ``vmin`` (lightest) to 0 (darkest, the largest entry), so
    matrices of different size and scaling share one color scale and the sparsity
    pattern is visible. Returns the image, for a color bar.
    """
    matrix = np.asarray(matrix, dtype=float)
    magnitude = np.abs(matrix)
    scale = magnitude.max() if magnitude.size and magnitude.max() > 0 else 1.0
    relative = magnitude / scale
    with np.errstate(divide="ignore"):
        logs = np.where(relative > 10.0 ** vmin, np.log10(np.maximum(relative, 1e-300)), np.nan)
    image = ax.imshow(logs, cmap=sequential_colormap(), interpolation="nearest", vmin=vmin, vmax=0.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_series(ax, x, series: dict, *, logy: bool = True, xlabel: str = "", ylabel: str = "", title: str = "",
                legend: bool = True) -> None:
    """Plot one curve per entry of ``series`` (name -> y values) against ``x``.

    Colors and markers are taken in the fixed order of ``CATEGORICAL`` and
    ``MARKERS``; more than eight series raise ``ValueError`` (split them into
    several axes instead).
    """
    if len(series) > len(CATEGORICAL):
        raise ValueError(f"at most {len(CATEGORICAL)} series per axes; got {len(series)}")
    for index, (name, values) in enumerate(series.items()):
        xs, ys = (np.asarray(x[name], dtype=float), np.asarray(values, dtype=float)) if isinstance(x, dict) \
            else (np.asarray(x, dtype=float), np.asarray(values, dtype=float))
        ax.plot(xs, ys, color=CATEGORICAL[index], marker=MARKERS[index], markersize=4, linewidth=1.5, label=name)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major", linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend and len(series) > 1:
        ax.legend(frameon=False)


def plot_triangle_function(ax, points: np.ndarray, values: np.ndarray, *, title: str = "", levels: int = 15,
                           bound: float | None = None):
    """Filled contours of a function sampled at ``points`` ``(n, 2)`` of the reference triangle.

    The color scale is the diverging blue-gray-red scale on ``[-bound, bound]``
    (default: the largest ``|value|``); pass one ``bound`` to several axes so that
    they share a color bar. Values at the ends of the range are drawn (a constant
    function fills the triangle with one color). Returns the contour set.
    """
    import matplotlib.tri as mtri

    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    if bound is None:
        bound = np.abs(values).max()
    bound = max(float(bound), 1e-300) * (1.0 + 1e-9)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1])
    contours = ax.tricontourf(triangulation, values, levels=np.linspace(-bound, bound, levels),
                              cmap=diverging_colormap(), extend="both")
    draw_simplex_outline(ax, np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]), ((0, 1), (1, 2), (2, 0)))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    return contours


def plot_text_table(ax, rows, columns, *, title: str = "") -> None:
    """Draw a small table of strings (``rows`` is a list of lists) on an axes without frame."""
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)
    ax.set_title(title)


__all__ = [
    "CATEGORICAL",
    "DIVERGING",
    "MARKERS",
    "SEQUENTIAL_BLUE",
    "diverging_colormap",
    "draw_simplex_outline",
    "new_figure",
    "plot_matrix_magnitude",
    "plot_series",
    "plot_tetrahedron_net",
    "plot_tetrahedron_nodes",
    "plot_text_table",
    "plot_triangle_function",
    "plot_triangle_nodes",
    "sequential_colormap",
    "tetrahedron_net",
]
