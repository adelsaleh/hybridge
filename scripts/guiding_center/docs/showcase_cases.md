# Showcase cases

The signed Euler vortex gases and the positive turbulence cases of the guiding-center runner: how they are
set up, which presets exist and how their diagnostics and figures are produced. Commands are in
[`example_runs.md`](../example_runs.md).

## Positive guiding-center turbulence

`positive_turbulence` is an HYBRIDGE-defined density analogue of the signed Euler
vortex gas. It uses the same 360 blobs, four Gaussian widths, amplitude range,
and seed, but all strengths are positive. The shared indexed Gaussian profile
truncates every blob at eight standard deviations. Centers are sampled in the
exact unit disk with enough scale-dependent clearance to leave `rho_0 = 0`
throughout `0.96 < r <= 1`; consequently the analytic initial density is
nonnegative on the whole disk and has no tails or cores at the wall.
Passing `geometry=iter`, `horseshoe`, or `pacman` samples the same positive
profile in that actual shaped domain, retaining the prescribed physical
`wall_gap`; mesh selection follows the case geometry automatically.

The guiding-center density equation and the incompressible Euler vorticity
equation have the same mathematical form up to the Poisson sign convention, so
this profile is equivalently one-signed Euler vorticity. Nonnegative vorticity
in domains with material boundaries is a standard analytical setting, and
same-sign vortex gases also occur in related two-dimensional turbulence work.
The particular disk, population, widths, and cutoff here are not a reproduction
of a published benchmark; they define a reproducible multiscale HYBRIDGE stress
test. See [Cai, Guo, and Qiu](https://arxiv.org/abs/1804.02365),
[Iftimie et al.](https://arxiv.org/abs/1305.0905), and
[van Kan et al.](https://arxiv.org/abs/2308.08789) for the three respective
connections. Published guiding-center positivity studies commonly use a
nonnegative diocotron annulus, as in
[Kiechle, Chudzik, and Helzel](https://doi.org/10.1016/j.jcp.2024.113693);
this gas complements the repository's existing annular cases instead of
replacing them.

The prepared IMEX-ARK3 run enables `--positivity-diagnostics`. It audits the
projected initial field, all three transported ARK stage densities, and the
accepted endpoint. This is a detection test: the current high-order transport
does not apply a positivity limiter, and a negative projected or evolved value
is reported rather than clipped.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr.args \
  --verbosity 3 --save-diagnostics
```

The preset is a user-run qualitative/positivity trial at `h=0.008`, DG p=6,
`dt=0.005`, and `T=50`; no production time integration has yet qualified its
stability or positivity behavior. The initial audit is also written to
`<diagnostics-prefix>_initial_positivity.json` before the first time step.


For a quiet ITER SI-BDF2 run with density DG(p), Poisson DG(p-1), and RT flux
recovery into DGVectorField(p), use:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_iter_si_bdf2_p6_poisson_p5_rt_p6_fast.args
```

This inherits the ITER case's 11,520 blobs, h=0.014, 300k+ triangles, dt=0.005
to T=50, and robust Poisson/transport retries. At the default p=6, Poisson uses
DG(5); recovered RT flux supplies every transport stage, including startup and
retries. The conflict-averaged upwind guard is active. Field/positivity diagnostics,
plots, timing files, and AMGX residual history are off, with convergence checks
and tolerances retained. This combination has configuration checks but no new
simulation or throughput measurement.

To enable the diagnostic output of the original command, append
`--positivity-diagnostics --diagnostics-every 10 --plot-diagnostics --plot-every 100 --verbosity 3`.
The fast preset has its own output prefix. `--order p` retains the one-degree
Poisson offset. Do not append `--transport-upwind-factor 1`, which would replace
conflict averaging with ordinary upwind.

## Horseshoe, ITER, and Pac-Man Euler gas

These presets use DG p=6, dt=0.05 and 1,000 steps to T=50 with the existing
raw-CUDA/native FB-HP-MG Poisson and AMGX transport stack. Most use IMEX-ARK3;
the denser ITER variant uses SI-BDF2 with one SI-Euler startup step.
Holoviz and physics diagnostics update every ten steps; solver timings remain
every-step. Initial projection uses 16 points per coordinate.

| Response-file preset | Mesh size | Minimum triangles | Vortices | Gaussian widths |
|---|---:|---:|---:|---|
| `euler_horseshoe_gas_imex_ark3_p6_150k_t50` | 0.0048 | 150,000 | 360 | .004, .008, .016, .032 |
| `euler_iter_gas_imex_ark3_p6_300k_t50` | 0.014 | 300,000 | 5,760 | .010, .020, .040, .080 |
| `euler_iter_gas_si_bdf2_p6_300k_t50` | 0.014 | 300,000 | 11,520 | .010, .020, .040, .080 |
| `euler_pacman_gas_imex_ark3_p6_150k_t50` | 0.006 | 150,000 | 360 | .008, .016, .032, .056 |

The horseshoe is an annular sector with radii .48 and 1 and a 60-degree gap.
Pac-Man removes a 60-degree sector from the unit disk. Both openings face +x.
ITER uses the supplied [ITER.geo](../../../hybridge/core/geometries/ITER.geo),
including the detailed lower wall, in its original coordinates (area about
30.73). Gmsh meshes the original Line/BSpline/Spline curves; vortex placement
samples those curves for wall containment and clearance. The physical wall
and interior labels remain 1 and 2. The h=0.014 target is sized for 300k+
triangles; the runner checks the actual count. No prebuilt mesh is bundled.
Gaussian widths are scaled to .010, .020, .040, .080 for this larger domain.

The ITER ARK3 counts are `(3072, 1536, 768, 384)`, 16 times the disk
population. The SI-BDF2 variant doubles them to `(6144, 3072, 1536, 768)`,
or 11,520 vortices and 32 times the disk population.
Centers are independent uniform-area samples in the eligible interior, with
no imposed reflection symmetry. Each Gaussian is cut off at eight widths,
where its relative value is about 1.27e-14, and its entire support stays inside
the wall. Signed strengths balance continuous circulation at each scale.
The shared spatial index evaluates nearby blobs on NumPy/CuPy, avoiding a
full mesh-points-by-vortices allocation. This cutoff defines initial data;
it does not limit or clip the evolving field.

```bash
# from the repository root
source .env

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_horseshoe_gas_imex_ark3_p6_150k_t50.args \
  --verbosity 3 --save-diagnostics

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_iter_gas_imex_ark3_p6_300k_t50.args \
  --verbosity 3 --save-diagnostics

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_iter_gas_si_bdf2_p6_300k_t50.args \
  --verbosity 3 --save-diagnostics

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_pacman_gas_imex_ark3_p6_150k_t50.args \
  --verbosity 3 --save-diagnostics
```

These are prepared user-run cases. Production triangle counts and time-integration
stability have not been measured; the existing minimum-triangle guard rejects an
undersized mesh. Small geometry/topology, profile, CLI, and plotting checks do
not establish long-run convergence. Adjust dt and num-steps together to retain T=50.
