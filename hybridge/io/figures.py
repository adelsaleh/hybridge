"""Consistent Matplotlib styling and portable publication-figure export."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def publication_style(*, font_size=9.0):
    """Use embedded serif fonts and restrained axes without external TeX."""
    import matplotlib as mpl

    with mpl.rc_context({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': font_size,
        'text.usetex': False, 'axes.titlesize': font_size + 1,
        'axes.labelsize': font_size, 'legend.fontsize': font_size - 1,
        'xtick.labelsize': font_size - 1, 'ytick.labelsize': font_size - 1,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6, 'axes.edgecolor': '#667085',
        'axes.axisbelow': True, 'grid.color': '#D0D5DD', 'grid.linewidth': 0.5,
        'lines.linewidth': 1.5, 'lines.markersize': 4,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path',
    }):
        yield


def apply_figure_theme(figure, foreground, *, transparent=True):
    """Recolor text, ticks and frames for a page theme, optionally on a clear background.

    Plotted data and colormaps are unchanged. A transparent figure takes the
    color of the page it is shown on, so a dark ``foreground`` suits dark pages
    and a light one suits dark-mode pages.
    """
    import matplotlib as mpl

    if transparent:
        figure.patch.set_alpha(0)
    for ax in figure.axes:
        if transparent:
            ax.set_facecolor('none')
        ax.tick_params(colors=foreground, which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor(foreground)
    for text in figure.findobj(mpl.text.Text):
        text.set_color(foreground)
    return figure


def save_publication_figure(figure, stem, *, formats=('pdf', 'svg', 'png'), dpi=220):
    """Export one figure as embedded-font PDF, portable SVG and/or PNG."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        if extension not in ('pdf', 'svg', 'png'):
            raise ValueError('Supported figure formats are pdf, svg and png')
        path = stem.with_suffix('.' + extension)
        metadata = ({'Creator': 'HYBRIDGE', 'CreationDate': None, 'ModDate': None}
                    if extension == 'pdf' else {'Date': None} if extension == 'svg' else None)
        figure.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.035,
                       metadata=metadata)
        paths.append(path)
    return tuple(paths)


def add_matplotlib_mesh(ax, mesh, *, color="black", linewidth=0.65, alpha=0.55, bounds=None):
    """Overlay existing physical triangles without retriangulation.

    Accept a DGMesh or any object with node_coords (N, 2) and integer
    triangles (E, 3) arrays. No numerical package, basis, or JIT import is
    needed. Optional bounds=(xmin, xmax, ymin, ymax) conservatively selects
    triangles whose bounding boxes intersect the view, including triangles
    crossing it without a vertex inside. Coordinates and connectivity are
    never modified; callers set axes limits and equal aspect themselves.
    Return Matplotlib's line artists, or an empty list for an empty crop.
    """
    import numpy as np
    import matplotlib.tri as mtri

    nodes = np.asarray(mesh.node_coords)
    triangles = np.asarray(mesh.triangles)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise ValueError("node_coords must contain finite (N, 2) coordinates")
    if (triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)):
        raise ValueError("triangles must contain integer (E, 3) connectivity")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(nodes)):
        raise ValueError("triangle index outside node_coords")
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if (bounds.shape != (4,) or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
            raise ValueError("bounds must be finite (xmin, xmax, ymin, ymax) with positive extents")
        vertices = nodes[triangles]
        lower, upper = vertices.min(axis=1), vertices.max(axis=1)
        keep = ((upper[:, 0] >= bounds[0]) & (lower[:, 0] <= bounds[1])
                & (upper[:, 1] >= bounds[2]) & (lower[:, 1] <= bounds[3]))
        triangles = triangles[keep]
    if not len(triangles):
        return []
    coarse = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    return ax.triplot(coarse, color=color, linewidth=linewidth, alpha=alpha)


def add_direction_glyphs(ax, x, y, vx, vy, *, length=0.08, headless=False,
                         color="black", alpha=0.7):
    """Draw equal-length direction glyphs in physical coordinates.

    Nonfinite samples and zero vectors are omitted. Arrows indicate an oriented
    vector; headless segments indicate an unoriented tensor axis. Glyph length
    carries no magnitude information. Callers should use equal axes aspect.
    Returns the Quiver or LineCollection artist; inputs are not modified.
    """
    import numpy as np

    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive")
    x, y, vx, vy = np.broadcast_arrays(x, y, vx, vy)
    magnitude = np.hypot(vx, vy)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(magnitude) & (magnitude > 0)
    points = np.column_stack((x[keep], y[keep]))
    vectors = np.column_stack((vx[keep], vy[keep])) * (length / magnitude[keep])[:, None]
    if headless:
        from matplotlib.collections import LineCollection
        artist = LineCollection(np.stack((points-vectors/2, points+vectors/2), axis=1),
                                colors=color, linewidths=0.7, alpha=alpha)
        ax.add_collection(artist)
        return artist
    return ax.quiver(points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1],
                     angles="xy", scale_units="xy", scale=1, pivot="mid",
                     color=color, alpha=alpha, width=.004)


__all__ = ['apply_figure_theme', 'publication_style', 'save_publication_figure', 'add_matplotlib_mesh', 'add_direction_glyphs']
