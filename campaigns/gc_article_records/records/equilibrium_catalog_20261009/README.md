# Equilibrium inputs for guiding-center preservation tests

This directory is the canonical home of the complete 2026-10-09 equilibrium
campaign: twelve thin near-wall equilibria, three placements each on star,
Ugly Pacman, ITER and horseshoe. It contains the portable CG2 potentials,
tagged meshes, window parameters, certificates, branch checkpoints, images,
logs, coarse initializer and earlier attempts.

- [Density gallery](equilibrium_gallery.png)
- [Results and certificate limits](results_2026_10_09.md)
- [Catalog index and checkpoint checksums](catalog.json)
- [Numerical table](catalog.csv)
- [Original validation record](validation.json)
- [Relocation inventory and checksums](relocation.json)
- [Solver method and reproduction](../../../hybridge-diocotron/projects/diocotron/studies/equilibrium_catalog/README.md)
- [Reusable equilibrium library and restart preparation](../../equilibria/README.md):
  independent copies of all twelve selected states plus the coarse ITER initializer.

## Using the equilibria

Resolve each `catalog.json` entry's `checkpoint`, `metadata`, `image` and
`tagged_mesh` paths relative to this directory. All required mesh and field
data are bundled here. The portable checkpoints contain the CG2 potential
and its complete affine mesh. Evaluate that potential at the HDG density
quadrature points, apply the saved nonlinear window, then project the density.

Nine fields pass the full Equiband guards. The three ITER fields pass the
semilinear PDE and closed-contour checks; their torsion-flow certificate is
unavailable, as recorded in the report. Guiding-center preservation tests
remain to be performed.

## Relocation

All 150 original output files (1,184,287,861 bytes) were moved from
`hybridge-diocotron/projects/diocotron/runs/equilibrium_catalog_20261009/`.
The original location is now a compatibility symlink to this directory.
Checkpoint paths in the convenience manifests and CSV point to the article;
their original versions are retained under `relocation_originals/`.
All remaining original files, including checkpoints, meshes, certificates,
input configurations, logs and figures, retain their original bytes.

Historical source paths inside configurations and checkpoints are provenance:
they are retained so recorded configuration and restart hashes remain valid.
For new consumers, use the relative paths in `catalog.json`. The solver and
rendering scripts remain in the `hybridge-diocotron` worktree.
