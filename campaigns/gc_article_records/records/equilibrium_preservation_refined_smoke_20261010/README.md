# Equilibrium import and preservation: short checks, 2026-10-10

The transfer works, but these modest HDG resolutions do not accurately preserve
the thin archived bands. Density drift is 2.60–3.10% after four steps on the finer
smoke level. This is preliminary evidence, not a long-time equilibrium certificate.

| Geometry / archived state | p=4 triangles | Euler density drift | BDF2 density drift | BDF2 potential drift |
|---|---:|---:|---:|---:|
| Star / star_02 | 7407 | 2.914% | 3.097% | 0.2424% |
| Ugly Pac-Man / uglypacman_04 | 6600 | 2.605% | 2.768% | 0.2723% |
| ITER / iter_02 | 5571 | 2.868% | 3.044% | 0.0884% |
| Horseshoe / horseshoe_02 | 7504 | 2.760% | 2.934% | 0.2841% |

Each run takes four steps with dt=0.05 to T=0.2, raw HDG drift, tau=1,
conflict-averaged upwind flux, zero-flux walls, no KKT or equilibrium polish.
The fine archived CG2 potential is evaluated before applying its frozen window
and L2-projecting into DG. The target interior mesh is coarser, but every archived
polygonal wall segment is retained. Source meshes were not evolved.

The p=4 initial semilinear mismatch is 2.32–4.60%; relative CG/HDG potential
mismatch is 0.201–0.658%; the normalized mass-inverted HDG transport residual is
0.486–0.535 per unit time. Projection sensitivity under 16 -> 20 Duffy points is
2.56e-5–1.14e-4. Poisson relative residuals stay below 7.7e-14 and relative mass
drift below 5.1e-16. Final quadrature-sampled densities include negative values
from -0.229 to -0.192. A nonnegative source window does not imply a positive DG field.

The [coarse p=2 checks](../equilibrium_preservation_smoke_20261010/summary.json)
drift by 7.75–9.15% over the same interval. Both h and p change; no convergence
order or time-step accuracy ranking follows from these two levels.

Files:

- [summary.json](summary.json): all eight p=4 runs, source hashes, frozen windows, residuals, final drifts and CPU observations.
- Per-run `*_config.json`, `.jsonl`, `.csv`, timings and `_fields.npz`: exact configs, all five diagnostic states, and native initial/final fields with their actual mesh.
- [import_validation.json](import_validation.json): all 12 fine archived equilibria plus the coarse ITER initializer pass legacy-v2 reading, checksums and CG2 sampling at 129 saved nodes each; maximum potential error <1.7e-16. Only the four listed states were evolved.
- [thread comparison](../equilibrium_preservation_threads16_20261010/thread_parity.json): repeated star p=4 BDF2 with verified Numba/host/Pardiso counts all 16; final density coefficients differ by at most 2.84e-14 from the earlier Numba=1/Pardiso=16 result. Sampled CPU activity confirms multiple active workers. The driver now defaults to all three counts=16. Original data were not relabelled as 16-thread assembly runs.
- [consumer_provenance.json](consumer_provenance.json), `consumer_source/`: revision and hash inventory of the consumer sources after validation, including the thread-control correction; source worktree has uncommitted changes.
- [console.log](console.log): full terminal output. `setup_attempt/` retains the initial setup failure (optional psutil unavailable; replaced with standard-library /proc sampling before any solve).
- [article figure](../../figures/equilibrium_preservation_smoke.pdf) and [table](../../tables/equilibrium_preservation_smoke.tex): generated from the saved runs; described in article Section 7.

Reproduction is documented in
[the runner guide](../../../hybridge/scripts/guiding_center/benchmarks/equilibrium_preservation.md).
Future work: full-source-mesh checks, separate h/p/dt ladders, compatible velocity,
HDG equilibrium correction with frozen window parameters, KKT and long-time tests.
No further time integration was performed in this pass after the one thread-parity check.
