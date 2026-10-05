# CIP screening and large-step equilibrium stress test (2026-08-20)

## Setup

- Equilibrium: `equilibrium_h002_p4_converged_gridseed_20260820/out/equilibrium_portable.npz`
- Guiding-center discretization: P4, 690,677 degrees of freedom, 86,112 triangles
- MPI: 16 ranks, MUMPS for transport and Poisson solves
- SUPG scale: 0.1
- CIP parameter is `--flux-stabilization`

## CIP screen at dt=0.025, T=0.1

| CIP | relative rho L2 drift | rho Linf change | relative advective defect | rho minimum | step time |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.28909e-4 | 1.93852e-3 | 7.52145e-3 | -7.00735e-5 | 4.16 s |
| 1e-5 | about 1.265e-4 | -- | about 7.126e-3 | -5.244e-5 | about 15.5 s |
| 1e-4 | 1.18029e-4 | 1.57085e-3 | 5.36987e-3 | -6.72646e-7 | 14.98 s |
| 1e-3 | 1.45832e-4 | 1.30200e-3 | 4.54921e-3 | -1.48252e-6 | 16.82 s |

`1e-4` is the best compromise in this screen. Relative to no CIP it lowers the
rho L2 drift by 8.4%, lowers the advective defect by 28.6%, and reduces the
negative undershoot by about 104x. It is nevertheless about 3.6x slower per
step because the facet coupling changes the MUMPS factorization workload.
`1e-3` lowers the residual further but increases the overall rho L2 drift by
13.1%, indicating over-stabilization for equilibrium preservation.

The interrupted `1e-5` screen did not flush its CSV; its values above are from
the live step-4 diagnostics. It had already shown too little benefit to merit a
full run.

## Large-step stress test

Run directory:

`projects/diocotron/runs/guiding_center/dolfinx_torsion_supg/equilibrium_h002_p4_supg01_cip1e4_dt05_T20_20260820`

Parameters: dt=0.5, 40 steps, T=20, SUPG=0.1, CIP=1e-4.

At T=20:

- relative rho L2 drift: 8.04872e-4
- rho Linf change: 3.37813e-3
- relative advective defect: 5.05622e-3 (down from 6.20099e-3 initially)
- rho range: [-3.58717e-4, 9.96435e-1]
- relative mass drift: 5.02072e-14
- relative energy drift: -1.92754e-11
- median step time: 15.56 s

No equilibrium break or instability onset was observed through T=20. The rho
drift grew sublinearly and its increment per step decreased, while the
advective defect declined. The extrema remained bounded and the initial
negative undershoot peaked early (near 6.9e-4) before shrinking. This behavior
is consistent with a small discretization-induced relaxation toward a nearby
discrete state, not exponential instability growth.

This test does not establish physical stability to arbitrary perturbations:
it starts from the equilibrium with only projection/roundoff errors, and SUPG
plus CIP can damp weak grid-scale modes. A deliberate modal perturbation and a
smaller-dt comparison would be the appropriate next instability experiment.
