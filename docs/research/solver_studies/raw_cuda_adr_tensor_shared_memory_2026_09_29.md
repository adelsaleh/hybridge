# Raw CUDA tensor ADR shared-memory budget — 2026-09-29

Scope: host-side sizing of the cooperative tensor ADR assembly/reconstruction
workspace (`hybridge/backends/adr_tensor_raw_cuda.py`) against its 48 KiB dynamic
shared-memory limit, for p=0--6, every exact diffusion kind, and the default
and overintegrated quadrature rules of the
[n-Gamma D-BDF2 plan](../../development/plans/n_gamma_d_bdf2.md). No GPU
assembly, AMGX solve, simulation or time integration was run for this record.
GPU parity at the overintegrated volume rule is covered by
`tests/test_adr_tensor_raw_cuda.py::test_tensor_overintegrated_quadrature`
(p=4 and 6, Duffy `p+5` and Dunavant degree-14 volume rules, both production
trace bases, all eight tensor representations, CSR matrix/RHS and reconstruction
against Numba) and
`::test_oversized_quadrature_fails_before_launch`: the 64 overintegration cases
plus the workspace checks passed on 2026-09-29 (FP64, CUDA 13 device).

Regenerate (JSON beside this note):

```bash
.venv/bin/python -m scripts.advection_diffusion_reaction.diagnostics.tensor_shared_memory_budget \
    --json docs/research/solver_studies/raw_cuda_adr_tensor_shared_memory_2026_09_29.json
HYBRIDGE_PRECISION=float32 .venv/bin/python -m \
    scripts.advection_diffusion_reaction.diagnostics.tensor_shared_memory_budget
```

Rules are given as 1D Gauss counts `volume_quad_1d/edge_quad_1d`: a collapsed
(Duffy) volume rule has `NQ = n^2` points; `auto` is the default Dunavant rule
(exact degree 2p). `deg:D` selects `DGSpace(volume_degree=D)`: the smallest
compact positive symmetric Dunavant rule of at least degree D (even degrees up
to 14; degree 13 or 14 has 42 points) and otherwise the minimal Duffy rule.
`p+5/p+4` is the plan's overintegration read as 1D counts; a collapsed rule
with `n` points per direction is exact to degree `2n-3`, so Duffy `p+5` is
exact to `2p+7` (15 at p=4) with `(p+5)^2` points.
**The face rule is not selectable for the production traces:**
`legacy-lagrange` and `legendre-modal` (`DGTraceSpace.from_space`) always use
`NFQ = 2p+1` Gauss--Lobatto points, exact to degree `4p-1`; `edge_quad_1d`
only changes `bernstein` traces. The tables use `legacy-lagrange`
(`legendre-modal` is identical).
Cells give the selected batch width / KiB; `-` means one element does not fit.

## FP64

| p | rule (vol/edge 1D) | NQ | NFQ | constant-isotropic | constant-diagonal | constant-full | variable-isotropic | variable-diagonal | variable-symmetric | variable-full | max NQ (variable-full) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | auto (symmetric) | 3 | 1 | 8/0.5 | 8/0.5 | 8/0.5 | 8/0.5 | 8/0.5 | 8/0.5 | 8/0.5 | 1533 |
| 0 | deg:2p+5 (symmetric degree>=5) | 12 | 1 | 8/0.7 | 8/0.7 | 8/0.7 | 8/0.7 | 8/0.7 | 8/0.7 | 8/0.7 | 1533 |
| 0 | deg:14 (symmetric degree>=14) | 42 | 1 | 8/1.5 | 8/1.5 | 8/1.5 | 8/1.5 | 8/1.5 | 8/1.5 | 8/1.5 | 1533 |
| 0 | p+3/default (3/default) | 9 | 1 | 8/0.6 | 8/0.6 | 8/0.6 | 8/0.6 | 8/0.6 | 8/0.6 | 8/0.6 | 1533 |
| 0 | p+5/p+4 (5/4) | 25 | 1 | 8/1.0 | 8/1.0 | 8/1.0 | 8/1.0 | 8/1.0 | 8/1.0 | 8/1.0 | 1533 |
| 0 | p+7/p+4 (7/4) | 49 | 1 | 8/1.7 | 8/1.7 | 8/1.7 | 8/1.7 | 8/1.7 | 8/1.7 | 8/1.7 | 1533 |
| 0 | 2p+2/default (2/default) | 4 | 1 | 8/0.5 | 8/0.5 | 8/0.5 | 8/0.5 | 8/0.5 | 8/0.5 | 8/0.5 | 1533 |
| 1 | auto (symmetric) | 3 | 3 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 1518 |
| 1 | deg:2p+5 (symmetric degree>=7) | 16 | 3 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 1518 |
| 1 | deg:14 (symmetric degree>=14) | 42 | 3 | 8/2.3 | 8/2.3 | 8/2.3 | 8/2.3 | 8/2.3 | 8/2.3 | 8/2.3 | 1518 |
| 1 | p+3/default (4/default) | 16 | 3 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 1518 |
| 1 | p+5/p+4 (6/5) | 36 | 3 | 8/2.1 | 8/2.1 | 8/2.1 | 8/2.1 | 8/2.1 | 8/2.1 | 8/2.1 | 1518 |
| 1 | p+7/p+4 (8/5) | 64 | 3 | 8/2.8 | 8/2.8 | 8/2.8 | 8/2.8 | 8/2.8 | 8/2.8 | 8/2.8 | 1518 |
| 1 | 2p+2/default (4/default) | 16 | 3 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 8/1.8 | 1518 |
| 2 | auto (symmetric) | 6 | 5 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/4.2 | 1468 |
| 2 | deg:2p+5 (symmetric degree>=9) | 25 | 5 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/4.2 | 1468 |
| 2 | deg:14 (symmetric degree>=14) | 42 | 5 | 8/4.0 | 8/4.0 | 8/4.0 | 8/4.0 | 8/4.0 | 8/4.0 | 8/4.3 | 1468 |
| 2 | p+3/default (5/default) | 25 | 5 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/4.2 | 1468 |
| 2 | p+5/p+4 (7/6) | 49 | 5 | 8/4.1 | 8/4.1 | 8/4.1 | 8/4.1 | 8/4.1 | 8/4.1 | 8/4.5 | 1468 |
| 2 | p+7/p+4 (9/6) | 81 | 5 | 8/4.9 | 8/4.9 | 8/4.9 | 8/4.9 | 8/4.9 | 8/4.9 | 8/5.2 | 1468 |
| 2 | 2p+2/default (6/default) | 36 | 5 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/3.8 | 8/4.2 | 1468 |
| 3 | auto (symmetric) | 12 | 7 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.4 | 8/8.9 | 1353 |
| 3 | deg:2p+5 (symmetric degree>=11) | 33 | 7 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.4 | 8/8.9 | 1353 |
| 3 | deg:14 (symmetric degree>=14) | 42 | 7 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.4 | 8/8.9 | 1353 |
| 3 | p+3/default (6/default) | 36 | 7 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.0 | 8/7.4 | 8/8.9 | 1353 |
| 3 | p+5/p+4 (8/7) | 64 | 7 | 8/7.4 | 8/7.4 | 8/7.4 | 8/7.4 | 8/7.4 | 8/7.8 | 8/9.2 | 1353 |
| 3 | p+7/p+4 (10/7) | 100 | 7 | 8/8.2 | 8/8.2 | 8/8.2 | 8/8.2 | 8/8.2 | 8/8.6 | 8/10.1 | 1353 |
| 3 | 2p+2/default (8/default) | 64 | 7 | 8/7.4 | 8/7.4 | 8/7.4 | 8/7.4 | 8/7.4 | 8/7.8 | 8/9.2 | 1353 |
| 4 | auto (symmetric) | 16 | 9 | 8/12.0 | 8/12.0 | 8/12.0 | 8/12.0 | 8/12.0 | 8/13.8 | 8/17.2 | 1131 |
| 4 | deg:2p+5 (symmetric degree>=13) | 42 | 9 | 8/12.0 | 8/12.0 | 8/12.0 | 8/12.0 | 8/12.0 | 8/13.8 | 8/17.2 | 1131 |
| 4 | deg:14 (symmetric degree>=14) | 42 | 9 | 8/12.0 | 8/12.0 | 8/12.0 | 8/12.0 | 8/12.0 | 8/13.8 | 8/17.2 | 1131 |
| 4 | p+3/default (7/default) | 49 | 9 | 8/12.0 | 8/12.0 | 8/12.0 | 8/12.0 | 8/12.0 | 8/13.8 | 8/17.2 | 1131 |
| 4 | p+5/p+4 (9/8) | 81 | 9 | 8/12.4 | 8/12.4 | 8/12.4 | 8/12.4 | 8/12.4 | 8/14.2 | 8/17.6 | 1131 |
| 4 | p+7/p+4 (11/8) | 121 | 9 | 8/13.4 | 8/13.4 | 8/13.4 | 8/13.4 | 8/13.4 | 8/15.1 | 8/18.5 | 1131 |
| 4 | 2p+2/default (10/default) | 100 | 9 | 8/12.9 | 8/12.9 | 8/12.9 | 8/12.9 | 8/12.9 | 8/14.6 | 8/18.0 | 1131 |
| 5 | auto (symmetric) | 25 | 11 | 8/19.5 | 8/19.5 | 8/19.5 | 8/19.5 | 8/20.4 | 8/23.9 | 8/30.6 | 748 |
| 5 | deg:2p+5 (duffy degree>=15) | 81 | 11 | 8/19.5 | 8/19.5 | 8/19.5 | 8/19.5 | 8/20.5 | 8/24.0 | 8/30.7 | 748 |
| 5 | deg:14 (symmetric degree>=14) | 42 | 11 | 8/19.5 | 8/19.5 | 8/19.5 | 8/19.5 | 8/20.4 | 8/23.9 | 8/30.6 | 748 |
| 5 | p+3/default (8/default) | 64 | 11 | 8/19.5 | 8/19.5 | 8/19.5 | 8/19.5 | 8/20.4 | 8/23.9 | 8/30.6 | 748 |
| 5 | p+5/p+4 (10/9) | 100 | 11 | 8/20.0 | 8/20.0 | 8/20.0 | 8/20.0 | 8/21.0 | 8/24.4 | 8/31.1 | 748 |
| 5 | p+7/p+4 (12/9) | 144 | 11 | 8/21.0 | 8/21.0 | 8/21.0 | 8/21.0 | 8/22.0 | 8/25.5 | 8/32.2 | 748 |
| 5 | 2p+2/default (12/default) | 144 | 11 | 8/21.0 | 8/21.0 | 8/21.0 | 8/21.0 | 8/22.0 | 8/25.5 | 8/32.2 | 748 |
| 6 | auto (symmetric) | 33 | 13 | 8/30.2 | 8/30.2 | 8/30.2 | 8/30.2 | 8/33.1 | 8/39.2 | 4/47.4 | 143 |
| 6 | deg:2p+5 (duffy degree>=17) | 100 | 13 | 8/30.4 | 8/30.4 | 8/30.4 | 8/30.4 | 8/33.3 | 8/39.4 | 2/47.5 | 143 |
| 6 | deg:14 (symmetric degree>=14) | 42 | 13 | 8/30.2 | 8/30.2 | 8/30.2 | 8/30.2 | 8/33.1 | 8/39.2 | 4/47.4 | 143 |
| 6 | p+3/default (9/default) | 81 | 13 | 8/30.2 | 8/30.2 | 8/30.2 | 8/30.2 | 8/33.1 | 8/39.2 | 3/47.7 | 143 |
| 6 | p+5/p+4 (11/10) | 121 | 13 | 8/30.9 | 8/30.9 | 8/30.9 | 8/30.9 | 8/33.8 | 8/39.9 | 2/48.0 | 143 |
| 6 | p+7/p+4 (13/10) | 169 | 13 | 8/32.0 | 8/32.0 | 8/32.0 | 8/32.0 | 8/34.9 | 8/41.0 | - | 143 |
| 6 | 2p+2/default (14/default) | 196 | 13 | 8/32.7 | 8/32.7 | 8/32.7 | 8/32.7 | 8/35.5 | 8/41.6 | - | 143 |

## FP32 (p=6 only; every lower order fits more easily)

| p | rule (vol/edge 1D) | NQ | NFQ | constant-isotropic | constant-diagonal | constant-full | variable-isotropic | variable-diagonal | variable-symmetric | variable-full | max NQ (variable-full) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | auto (symmetric) | 33 | 13 | 8/15.3 | 8/15.3 | 8/15.3 | 8/15.3 | 8/16.8 | 8/19.8 | 8/25.8 | 1665 |
| 6 | deg:2p+5 (duffy degree>=17) | 100 | 13 | 8/15.4 | 8/15.4 | 8/15.4 | 8/15.4 | 8/16.9 | 8/19.9 | 8/25.9 | 1665 |
| 6 | deg:14 (symmetric degree>=14) | 42 | 13 | 8/15.3 | 8/15.3 | 8/15.3 | 8/15.3 | 8/16.8 | 8/19.8 | 8/25.8 | 1665 |
| 6 | p+3/default (9/default) | 81 | 13 | 8/15.3 | 8/15.3 | 8/15.3 | 8/15.3 | 8/16.8 | 8/19.8 | 8/25.8 | 1665 |
| 6 | p+5/p+4 (11/10) | 121 | 13 | 8/15.7 | 8/15.7 | 8/15.7 | 8/15.7 | 8/17.1 | 8/20.2 | 8/26.2 | 1665 |
| 6 | p+7/p+4 (13/10) | 169 | 13 | 8/16.2 | 8/16.2 | 8/16.2 | 8/16.2 | 8/17.7 | 8/20.7 | 8/26.7 | 1665 |
| 6 | 2p+2/default (14/default) | 196 | 13 | 8/16.6 | 8/16.6 | 8/16.6 | 8/16.6 | 8/18.0 | 8/21.0 | 8/27.1 | 1665 |

## Findings

- p<=5 fits every kind and every listed rule at the full batch width 8; the
  largest p=5 `variable-full` workspace is 32.2 KiB.
- p=4 with the plan's `p+5` volume rule (NQ=81, NFQ=9) uses at most 17.6 KiB,
  so the n-Gamma stationary and transient studies are not constrained.
- The degree-14 Dunavant rule needs 42 volume points at every order, against 81
  (p=4) and 121 (p=6) Duffy `p+5` points, one degree less exact at p=4
  (14 versus 15). It fits every kind at p<=5 at batch 8 and p=6 `variable-full`
  at batch 4. GPU matrix/RHS and reconstruction parity against Numba passes for
  both rules (`test_tensor_overintegrated_quadrature`, 64 cases).
- The plan's "faces `p+4`" cannot be requested for production traces. The
  fixed 2p+1 GLL face rule is exact to degree `4p-1`, which equals a `p+4`
  Gauss rule (`2p+7`) at p=4, exceeds it for p>4 and is weaker for p<4.
- The only limited class is FP64 p=6 `variable-full` (general nonsymmetric
  variable tensor): at most NQ=143 volume points fit with NFQ=13. The batch
  width falls to 4, 3 and 2 for NQ=33, 81 and 121, and the `p+7` and
  default-Duffy `2p+2` rules do not fit.
  `variable-symmetric` fits all rules at batch 8; the n-Gamma tensors
  `R*D*P` and `R*mu*P` are symmetric and use that kind when supplied as the
  three components `(k00, k01, k11)`.
- FP32 halves every workspace; p=6 `variable-full` then fits up to NQ=1665.
- Oversized configurations now raise `TensorWorkspaceError` (an
  `UnsupportedBackendConfigurationError` and `ValueError`) before upload,
  compilation or launch, naming p, NEL, the kind, NQ/NFQ, the required bytes
  and the largest NQ that fits.

Not done: raising the limit through the opt-in dynamic shared-memory attribute
(`cudaFuncAttributeMaxDynamicSharedMemorySize`), and performance of the reduced
batch widths at p=6 `variable-full`.
