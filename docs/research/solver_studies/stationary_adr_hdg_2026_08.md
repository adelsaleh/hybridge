# Stationary ADR HDG Implementation — 2026-08

## Scope

This change adds the conservative stationary problem
`div(beta*u + q) + r*u = f`, `q = -kappa*grad(u)`, with Dirichlet data on the
whole boundary. The maintained derivation is
[`docs/algorithms/advection_diffusion_reaction/assembly.tex`](../../algorithms/advection_diffusion_reaction/assembly.tex).

The implementation was built in this order:

1. a dense NumPy element-local reference;
2. a fused multithreaded Numba `prange` assembly and reconstruction;
3. a readable one-thread-per-element raw-CUDA assembly and reconstruction,
   with device COO-to-CSR and direct PyAMGX handoff.

All three use the total numerical flux

```text
q.n + (beta.n)*uhat + (tau_adv + tau_diff)*(u - uhat).
```

The default advection penalty is upwind, `abs(beta.n)`. The default diffusion
penalty is `C_tau*(p+1)^2*kappa/h_F`, with `C_tau=1`. Explicit scalar or
per-element/per-face diffusion stabilization remains available.

## Incidence-Wise Transmission Assembly

For `tau=tau_adv+tau_diff` and `gamma=tau-beta.n`, every element side emits its
own `gamma`-weighted global edge mass block. This happens inside the
element/local-face loop in both fused Numba and raw CUDA. No neighboring-side
average and no hard-coded `2*tau` shortcut is used. A regression uses unequal
stabilization on the two incidences of every interior edge and checks the
resulting NumPy, Numba, and raw-CUDA systems/solutions.

## Coefficient And Postprocessing Semantics

Source, reaction, and velocity are sampled from their own coefficient objects
onto the solve-space volume/face quadrature. A regression solves with these
three fields in different DG orders on the same mesh.

Postprocessing supports `none`, `primal`, `flux`, and `both`. It first builds a
degree-`p+1` conservative reconstruction `q_T_star` of `q + beta*u`, whose
face-normal moments match the complete advective-diffusive numerical flux. The
primal recovery then solves the coupled degree-`p+1` element HDG Neumann
problem for `(u_star, q_star, phi_star)`, driven by the divergence and boundary
normal moments of `q_T_star`, with `(u_star, 1)_K = (u_h, 1)_K`. It therefore
includes advection and the total stabilization; it is not the diffusion-only
`q_h` gradient recovery. In `primal` output mode the prerequisite total-flux
reconstruction is computed internally but is not returned.

The default total-flux reconstruction is the constrained minimum-`L2` method
in full `[P_{p+1}]^2`. The selectable experimental `rt-p` method instead uses
`RT_p=[P_p]^2+x P_p`; its `P_p(F)` normal moments and `[P_{p-1}]^2` interior
moments form a square unisolvent system. It has host Numba and batched CuPy
implementations. The coupled primal recovery remains host Numba. The raw-CUDA
path currently materializes its degree-`p` reconstruction on the host before
postprocessing, so selecting CuPy presently incurs a host-to-device re-upload.

## RT Projection And Stabilization Comparison (2026-08-14)

The RT reconstruction implements the two moment equations directly: its face
target is the complete numerical total flux and its interior target is
`q_h+beta*u_h` sampled on the postprocessing quadrature. No preliminary
degree-`p` projection is used in the RT interior right-hand side.

Matched `Pe=10`, `p=3` disk runs at mesh-size requests `0.4`, `0.2`, and `0.1`
show that reconstruction space mainly changes the error constant under the
default stabilization:

| reconstruction | post primal errors | post total-flux errors |
| --- | --- | --- |
| full `[P_{p+1}]^2` | `4.8526e-4`, `2.9155e-5`, `2.0010e-6` | `2.0266e-3`, `2.3679e-4`, `3.1652e-5` |
| `RT_p` | `4.9141e-4`, `2.9396e-5`, `2.0057e-6` | `2.2203e-3`, `2.5787e-4`, `3.4063e-5` |

The slightly larger RT values do not contradict optimal convergence: optimality
specifies an asymptotic order, not the smallest error constant among different
reconstruction spaces. Host Numba and CuPy RT coefficients agree within the
regression tolerances. The full-space minimum-distance reconstruction therefore
remains the default.

Stabilization scaling has a much larger effect. Repeating both disk
reconstructions with constant `tau_diff=0.1` gives:

| reconstruction / tau_diff | post primal errors | post total-flux errors |
| --- | --- | --- |
| full / constant `0.1` | `1.9513e-4`, `8.2740e-6`, `3.0542e-7` | `8.3866e-4`, `6.6703e-5`, `5.1910e-6` |
| `RT_p` / constant `0.1` | `2.1243e-4`, `9.5442e-6`, `3.5146e-7` | `8.1887e-4`, `6.9449e-5`, `5.0778e-6` |

Because those disk meshes are neither geometrically exact nor uniformly nested,
a second study used affine nested square meshes and a smooth trigonometric
solution. For `p=3`, successive refinements gave:

| reconstruction / tau_diff | post primal rates | post total-flux rates |
| --- | --- | --- |
| full / `(p+1)^2*kappa/h_F` | `4.36`, `4.18` | `3.18`, `3.08` |
| `RT_p` / `(p+1)^2*kappa/h_F` | `4.36`, `4.18` | `3.17`, `3.07` |
| full / constant `0.1` | `4.925`, `4.969` | `3.954`, `3.978` |
| `RT_p` / constant `0.1` | `4.925`, `4.969` | `3.959`, `3.980` |

Thus the Cockburn-style expected rates are recovered by both flux spaces with
h-independent stabilization: approximately `p+2=5` for the primal and
`p+1=4` for the total flux. In the constant-beta, constant-tau square problem,
the two primal reconstructions agree to roundoff because the normal and
divergence moments driving the coupled local primal problem coincide. Their
flux fields differ slightly. At the time of this measurement, the inverse-h
rule was retained as the public default pending further qualification. That
default decision is now superseded: the supported production default is
`tau_d=kappa/L_Omega`, while inverse-h remains an explicit comparison mode.
The manufactured runner exposes fixed-constant comparisons through
`--diffusion-stabilization 0.1` and the legacy rule through
`--diffusion-stabilization-mode inverse-h`.

## Validation Record

Focused checks through 2026-08-14 gave:

- variable-coefficient degree-2 NumPy/Numba matrix difference `2.66e-15`, RHS
  difference `3.55e-15`, and reconstruction difference `7.50e-15`;
- an affine degree-1 manufactured solution error `1.55e-15`, with both
  degree-2 postprocessors present and physical relative residual `1.53e-16`;
- asymmetric degree-2 element/face penalties: NumPy/Numba trace difference
  `1.11e-15`, local difference `6.92e-15`; NumPy/raw-CUDA trace difference
  `9.76e-15`, local difference `1.34e-14`, raw physical residual `1.19e-15`;
- focused ADR and manufactured-runner tests: 24 passed, including positive
  scalar-diffusion Raw CUDA/PyAMGX parity of both postprocessed fields, crossed
  NumPy/Numba host stages, and the two-space nested-mesh convergence check; the
  combined
  solver API, backend capability, and documentation contracts passed 246 tests;
- for the steady disk at `Pe=10`, `p=3`, the corrected coupled recovery reduced
  primal errors from `8.8624e-4`, `5.9118e-5`, and `4.0944e-6` to `4.8525e-4`,
  `2.9155e-5`, and `2.0010e-6` at mesh-size requests `0.4`, `0.2`, and `0.1`.

These are correctness/smoke results, not performance claims.

## Bounded Limitations

- Fused Numba and raw CUDA support positive constant scalar diffusion; NumPy
  remains the reference path for variable scalar or tensor diffusion accepted by
  the existing mixed local operator. Automatic diffusion stabilization currently
  requires a positive scalar constant; tensor diffusion needs explicit side penalties.
- Host NumPy/Numba assembly and reconstruction stages may be mixed. Raw CUDA
  assembly currently requires Raw CUDA reconstruction. Full-space
  postprocessing uses host Numba; RT total-flux reconstruction can use Numba or
  CuPy, but coupled primal recovery remains host Numba and current Raw CUDA data
  are materialized on host before any CuPy re-upload.
- Raw CUDA currently uses one thread per element and device COO-to-CSR. A
  cooperative kernel and direct CSR emission need separate performance work.
- Raw CUDA returns host-materialized fields and performs degree-`p+1`
  postprocessing on the host after device reconstruction.
- The currently implemented boundary mode is whole-boundary Dirichlet
  elimination.
