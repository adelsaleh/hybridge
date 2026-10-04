# Recommended Torsion-Initialized Newton Parameters

These recommendations are for
`scripts/torsion_equilibrium/dolfinx/dolfinx_torsion_initialized_newton.py` on the fixed smooth-star
tests in this study.

## Preferred Phi-Design-Scaled Window

Use the `phi-design` nonlinear window for production runs.  In this mode,

```text
c1Phi = betaPhi1 * max(phiDesign)
c2Phi = betaPhi2 * max(phiDesign)
epsPhi = eps_phi_ratio * (c2Phi - c1Phi)
```

The best balanced choice from the dense coarse-mesh sweep is:

```text
alphaT1  = 0.40
alphaT2  = 0.50
betaPhi1 = 0.75625
betaPhi2 = 0.89375
```

Equivalent width/shift form:

```text
alpha_center = 0.45
alpha_width  = 0.10
gamma        = 1.375
delta        = 0.375
beta_width   = gamma * alpha_width = 0.1375
beta_center  = alpha_center + delta = 0.825
beta         = (0.75625, 0.89375)
```

The best relative-L2 point was more aggressive and closer to the robustness
boundary:

```text
alphaT1  = 0.40
alphaT2  = 0.50
betaPhi1 = 0.80
betaPhi2 = 0.95

gamma = 1.50
delta = 0.425
```

For active-band overlap, use:

```text
alphaT1  = 0.40
alphaT2  = 0.50
betaPhi1 = 0.60625
betaPhi2 = 0.69375

gamma = 0.875
delta = 0.20
```

## Command Form

Balanced run:

```bash
XDG_CACHE_HOME=/tmp/hdgfem_fenics_cache \
MPLCONFIGDIR=/tmp/hdgfem_mpl_cache \
/home/as305/miniforge3/envs/fenicsx-dgfem/bin/python \
  scripts/torsion_equilibrium/dolfinx/dolfinx_torsion_initialized_newton.py \
  --mesh run_outputs/strategyA_band_study_20260712/fixed_mesh/smooth_star_h010_n220.msh \
  --order 5 \
  --alphaT1 0.40 \
  --alphaT2 0.50 \
  --betaPhi1 0.75625 \
  --betaPhi2 0.89375 \
  --phi-window-source phi-design \
  --eps-t-ratio 0.06 \
  --eps-phi-ratios 0.11,0.08,0.06 \
  --save-frames \
  --plot-off-screen \
  --no-plot-initial \
  --no-plot-design \
  --no-plot-newton \
  --plot-final \
  -v 2
```

## Torsion-Scaled Diagnostic Window

The newer torsion-window mode is available for diagnostics:

```text
base_width = (alphaT2 - alphaT1) * max(T)
c1Phi = alphaT1 * max(T) + shift_scale * base_width
c2Phi = c1Phi + width_scale * base_width
```

Use:

```bash
--phi-window-source torsion
--phi-window-torsion-shift-scale <shift>
--phi-window-torsion-width-scale <width>
```

This is not the recommended production form for the current smooth-star setup.
For `alpha=(0.40,0.50)`, the tested torsion-scaled windows place `c1Phi` far
above `max(phiDesign)`, leaving the nonlinear source essentially empty.  The
tested `p=5` cases

```text
shift= 0.00, width=1.00
shift=-0.25, width=1.05
shift= 0.25, width=1.05
```

all ended nonconverged with zero or near-zero final density.  The torsion form
is useful for scale diagnostics; the phi-design-scaled beta window is the form
that produced the successful equilibria in this study.
