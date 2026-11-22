"""Problem definition read by ``scripts/reference_simplex/plot_alpha_u_f.py``: ``alpha u = f`` on the reference simplex.

Edit this file, or copy it and pass the copy with ``--problem``. The driver reads:

``alpha``
    The coefficient: a positive number, or a function ``alpha(x)`` of the points
    ``x`` (array of shape ``(n, dim)``, reference coordinates) returning ``n``
    positive values.
``f``
    The right-hand side: a function ``f(x)`` returning ``n`` values.
``DIM`` (optional, default 3)
    2 for the reference triangle ``conv{(-1,-1), (1,-1), (-1,1)}``, 3 for the
    reference tetrahedron ``conv{(-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)}``.
``DEGREES`` (optional)
    Polynomial degrees ``p`` of ``u_h`` used when ``--degrees`` is not given.
``TAU`` (optional, default 1)
    The HDG stabilization ``tau > 0``.

The exact solution is ``u = f / alpha``; it is also the trace data on every face.
Functions must accept arrays of points of either dimension that ``DIM`` selects;
``x[:, 0]``, ``x[:, 1]`` (and ``x[:, 2]`` in 3D) are the coordinates.
"""

import numpy as np

DIM = 3
DEGREES = (1, 2, 3, 4, 6, 8)
TAU = 1.0


def alpha(x):
    """A smooth positive coefficient, 1 + |x|^2 / 2."""
    return 1.0 + 0.5 * np.sum(x * x, axis=1)


def f(x):
    """Right-hand side chosen so that u = f / alpha = exp((x + y + z) / 2) cos(pi x / 2) is smooth and not a polynomial."""
    u = np.exp(0.5 * np.sum(x, axis=1)) * np.cos(0.5 * np.pi * x[:, 0])
    return alpha(x) * u
