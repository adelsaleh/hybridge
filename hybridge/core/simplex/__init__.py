"""Reference simplices (interval, triangle, tetrahedron): geometry, quadrature and bases.

This subpackage is the lowest level of :mod:`hybridge.core` for reference-element
data. It does not import :mod:`hybridge.core.quadrature` or :mod:`hybridge.core.space`;
those modules build :class:`~hybridge.core.quadrature.ReferenceElementData` and the DG
spaces on top of it.
"""
