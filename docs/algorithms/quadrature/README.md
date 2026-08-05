# Triangle Quadrature

[`symmetric_triangle_quadrature.tex`](symmetric_triangle_quadrature.tex)
documents the reference triangle, polynomial basis families, exactness target,
symmetric rule construction, and validation matrix.

For products of degree-`p` trial and test functions, the baseline exactness
target is total degree `2p`. The automatic policy uses compact tabulated
Dunavant rules through polynomial order 7 and then selects the lower-point
admissible fallback. The generated positive-weight symmetric rule remains an
explicit option; the collapsed Duffy rule remains available for compatibility
and higher-order fallback.

The implementation lives in `hdgfem.core.quadrature` and is validated by
`tests/test_symmetric_triangle_quadrature_host.py`. Quadrature policy changes
must preserve monomial exactness, reference mass/stiffness matrices, and
manufactured-solution parity before changing a solver default.
