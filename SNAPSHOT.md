# Snapshot of the revision used by the guiding-center article

This tag holds the Hybridge package exactly as it was when the timings and
tables of the guiding-center article were computed (2026-11-02). It contains
the package `hybridge/`, the AMGX configurations in `configs/` that the
package reads, and the files needed to install it. Tests, documentation,
campaign launchers and examples are not part of the snapshot; they are on
`master`.

The Poisson runs of the article use
`poisson_cache_local_factors="schur-lu+responses"` and an AMG coarse grid
solved by dense LU up to 256 rows.

Install it as usual:

    pip install -e .
