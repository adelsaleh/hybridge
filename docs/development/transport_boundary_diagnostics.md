# Transport tangency, scaling and last-resort device QR

The six rejected AMGX attempts establish nonconvergence, not singularity.
The saved step-555 diagnostics show a small, persistent boundary defect and
larger interior compatibility defects as the vortex gas evolves. The new
localized preset and diagnostics support a controlled investigation; they do
not establish that localization cures the failure.

## Confirmed BDF2 rank loss at step 208 (2026-09-12)

An authorized quiet replay captured the actual BSR matrices **before AMGX** at
steps 207 and 208. On interior edge 156512, the seven trace columns have rank
7 at step 207 and rank 6 at step 208; their smallest/largest singular-value
ratio drops from 1.74e-5 to 1.23e-17. The latter matrix has a face-local null
mode to roundoff, despite having no zero rows or columns.

The two discontinuous outward velocity traces both classify parts of this
nearly tangent face as outflow. With `gamma = abs(beta.n) - beta.n`, only six
of the thirteen face quadrature nodes then couple the degree-6 trace to either
element. A nonzero trace polynomial vanishes at all six nodes, disconnecting
one mode. Root-split quadrature or a shared numerical normal velocity restores
full rank in saved-face tests. A production flux remedy has not yet been
implemented or validated.

See the [matrix evidence, small face experiments and replay record](
../../artifacts/transport_diagnostics_20260912/README.md). The device failure
path now preserves a system archive for host-only analysis without replay.
This confirms the BDF2 failure mechanism; the earlier predictor failures below
were not replayed and retain their original evidence limits.

## When the velocity is tangent

The code uses `q = -grad(phi)` and rotates it as `v = (-q_y, q_x)`.
For outward normal `n` and tangent `t = (-n_y, n_x)`, this gives

```text
v = (phi_y, -phi_x)
v . n = grad(phi) . t = d(phi)/ds.
```

Thus the exact velocity is tangent when phi is constant along each connected
boundary component. The constants may differ between components and may depend
on time. This is the usual impermeable streamfunction condition; see the
[NYU streamfunction formulation](https://math.nyu.edu/~goodman/teaching/NumericalMethodsII2020/assignments/assignment4.pdf).
The disk cases `diocotron_gaussian_annulus`, `diocotron_k`, `spiral_sheet` and
`euler_vortex_gas` prescribe zero wall potential. The manufactured
`rho_helm_wave` case prescribes a nonconstant potential and uses a different
transport boundary treatment; it has no corresponding general tangency guarantee.

This condition is independent of the time-step size. It must hold for the
velocity used at every stage. Linear combinations of tangent velocities remain
tangent. The predictor uses `beta = dt*v^n`, and the corrector uses
`beta = (dt/2)*v_mid`, where the midpoint flux averages two Poisson fluxes.
Reducing dt reduces the scaled defect in beta, but does not make v tangent.
Vorticity need not vanish at the wall, and localizing vorticity does not localize
the Poisson field.

The discrete qualification matters: HDG approximates q independently of phi.
Setting the potential trace to zero does not force the computed `q_h . t` to
zero. Rotating q_h therefore need not produce a field that is simultaneously:

- tangent at the exterior boundary;
- divergence free inside each element;
- continuous in its normal component across interior edges.

The zero-flux transport treatment suppresses the exterior numerical flux; it
does not project the element velocity onto a tangent field. An H(div)/RT
postprocess of q controls its normal component, whereas tangency of the rotated
velocity requires its tangential component. The actual stage builders currently
rotate the standard Poisson flux; the new diagnostics measure that same flux.
A compatible rotated gradient of a continuous potential with constant boundary
values would address these three constraints, but is a separate discretization
change.

## Evidence at the failed predictor

The [preserved failure report](../../artifacts/transport_diagnostics_20260911/predictor_step555_failure.json)
was read from the workspace output produced on 2026-09-11. It records predictor
step 555, t=5.55, of the original radius-0.96 vortex gas. All six attempts returned
finite iterates and missed the physical residual target; there was no BJ CUDA
memory error in this report.

The failure file measures **beta**, with `beta_scale=0.01`; ordinary accepted-step
diagnostics measure **v**. Dividing the failure norms by that scale gives:

| Diagnostic | Initial state | Failed predictor velocity |
| --- | ---: | ---: |
| Boundary normal L2 / boundary speed L2 | 8.908e-4 | 8.914e-4 |
| Boundary normal L2 | 3.001e-4 | 2.840e-4 |
| Elementwise divergence L2 | 1.343e-2 | 5.702e-2 |
| Interior normal jump L2 | 3.092e-4 | 1.319e-3 |

The boundary leakage remains about **0.089%**, rather than suddenly increasing.
Interior divergence and normal jumps increased by about a factor of four.
This weakens a sudden boundary-tangency-loss explanation; it does not exclude
an accumulated discrete compatibility problem. The
[accepted-step evidence](../../artifacts/transport_diagnostics_20260911/accepted_step_diagnostics.json)
also preserves the values at step 550.

The trace matrix has 530,264 unknowns and 18,507,888 stored scalar entries. Its
row L1 norms range from 1.36e-10 to 1.72e-5, a ratio of about 1.26e5, with no
zero rows. That ratio measures equation scales, **not** a condition number;
nearly dependent rows can exist without any zero row.

Scaled L1 PBICGSTAB reaches physical relative residual 5.50e-2, scaled BJ
PBICGSTAB diverges, and FGMRES/DILU plus two corrections reaches 1.43e-4. The
last physical absolute residual is 3.38e-7 against a 1e-12 target. The report
confirms that FGMRES reused DILU factors. Stale factors are another possible
contributor: a fresh factorization of the same failed matrix is a useful future
comparison, without increasing DILU strength. No condition estimate or direct
factorization of this large failed matrix has been performed.

## Localized initial data

Use `euler_vortex_gas_localized_p6_50k_dt001_t50_raw_cuda_bsr`, or its
[response file](../../run_configs/guiding_center/euler_vortex_gas_localized_p6_50k_dt001_t50_raw_cuda_bsr.args).
It keeps seed 17, all 360 vortices, Gaussian widths 0.008/0.016/0.032/0.064,
p=6, the 50k mesh target and dt=0.01, but changes the center radius from 0.96
to 0.5. Its output prefix is distinct from the original preset.

Initial-field sampling at 2,048 wall angles gives maximum absolute vorticity
2.88 for the original data and 1.42e-13 for the localized data. Sampling 20 radii
from 0.85 to 1 gives maxima 7.63 and 9.05e-7 respectively. These are Gaussian
tails, not compact support. Concentrating the same vortices also changes their
overlap and interactions; no evolved localized solution was tested.

## Scaling preconditioned methods

Both PBICGSTAB retries now inherit `transport_scale_system`, which was previously
hardcoded off for those retries. FGMRES and its corrections already inherited
that option. With the device presets, all six iterative attempts now use left
row-diagonal scaling. `--transport-scale-system off` disables it for all six.
The retry count, order and DILU strength are unchanged.

Row scaling solves `D^-1*A*x = D^-1*b`. It preserves the exact solution, but
changes residual weighting and can change preconditioner construction and
finite-precision behavior. An ideal preconditioner can already cancel the row
scale, so extra scaling is not guaranteed to reduce iterations. Always compare
the original-system residual as well as AMGX's monitored residual.

The device solver now restores the physical coefficients and explicitly
recomputes `b-A*x` for validation. Each attempt records physical and solver
residuals separately; correction acceptance uses the combined solution and
original RHS. No convergence tolerance was relaxed.

A [reproducible 56-unknown comparison](../../artifacts/transport_diagnostics_20260911/scaling_probe.py)
used nonsymmetric 7x7 blocks, row multipliers from 1e-4 to 1e4, FP64, zero guesses,
and a fresh preconditioner for each solve. All six solves converged:

| Method | Iterations: unscaled / scaled | Relative solution error: unscaled / scaled |
| --- | ---: | ---: |
| PBICGSTAB + L1 | 11 / 11 | 8.28e-13 / 1.28e-13 |
| PBICGSTAB + BJ | 8 / 9 | 6.25e-13 / 1.55e-13 |
| FGMRES + DILU | 7 / 8 | 2.03e-11 / 1.11e-13 |

The [raw results](../../artifacts/transport_diagnostics_20260911/scaling_probe.json)
show improved solution accuracy here, without an iteration-count improvement.
This synthetic example is not a performance prediction for the vortex gas.

## Optional device direct fallback

Append `--transport-direct-fallback cusolver-qr` to enable a seventh, final
attempt. It uses the installed CuPy/cuSOLVER sparse QR interface, expands BSR
to scalar CSR on device, and solves a separate canonical CSR copy of the
unscaled system. There is no host sparse-solver fallback. It is lazy: successful
AMGX solves do not invoke QR. Singular warnings, nonfinite solutions and failed
physical residual checks are rejected.

CuPy's [binding](https://raw.githubusercontent.com/cupy/cupy/v14.2.0/cupyx/_cusolver.pyx)
returns a device solution and reports detected singularity through a warning.
Its `tol` parameter is a pivot threshold, distinct from the physical residual
tolerance. The fallback uses zero for that threshold and applies the existing
residual contract afterward. This is not a rank or conditioning certificate.

The installed sparse QR API is deprecated in
[CUDA 13](https://docs.nvidia.com/cuda/archive/13.0.0/cusolver/index.html#cusolver-lt-t-gt-csrlsvqr).
It uses host and device workspace internally and factorization fill can greatly
exceed matrix storage. cuDSS is a future replacement; its Python bindings are
not installed here. Only small direct solves were tested, not the 530k-unknown
system. A direct solver cannot restore a CUDA context after a device memory
fault or cure an inconsistent singular system.

## Validation scope (2026-09-11 investigation)

Validation was restricted to diagnostics, configuration checks, analytic initial
fields and small matrices. No simulation was launched for this investigation.
Tests cover manufactured tangency/divergence/jumps, host/device diagnostic
agreement, failed-stage reports, localized initial data, scaled and unscaled
native BSR preconditioners, direct CSR/BSR solves, lazy fallback, singular and
nonfinite rejection, and the physical convergence contract. The large-run
observations above came from reading existing workspace output.

The focused FP64 suite passed all 115 tests. The FP32 diagnostic, direct-solve
and native preconditioner suite passed all 70 tests. Test logs are retained
with the [investigation artifacts](../../artifacts/transport_diagnostics_20260911/README.md).

Documentation links pass. The repository area-inventory check still flags the
existing `docs/diocotron` directory, which was not changed by this work.
