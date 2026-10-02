# Coefficient and Stabilization Inputs

This is the user-facing alpha matrix for PDE coefficients and HDG
stabilization. Internal quadrature tables and compact descriptors are backend
implementation data. Users normally provide formulas, scalar values, or DG
fields. A user should construct a quadrature table only when deliberately using
an advanced evaluated-table interface.

The mathematical input and execution backend are separate choices:

- NumPy and supported CuPy reference paths evaluate compatible vectorized
  analytic formulas at quadrature points.
- Numba and raw-CUDA kernels never call Python. Adapters require or prepare DG
  fields, constant descriptors, or evaluated tables before compiled dispatch.
- A CuPy-incompatible callable is rejected; there is no silent host evaluation
  and device upload.

See [Coefficient Inputs](coefficient_inputs.md) for field materialization and
projection semantics and [Backend Capabilities](backend_capabilities.md) for
valid assembly, solve, and reconstruction combinations.

## PDE coefficient matrix

| Solver | Parameter | NumPy | CuPy | Numba | raw CUDA |
| --- | --- | --- | --- | --- | --- |
| Advection-reaction | source, reaction | Vectorized callable, scalar, or DGField | CuPy-compatible callable, scalar, or resident/host DGField | Same-space DGField; lazy zero/constant fields are supported | Same-space projected DGField |
| Advection-reaction | beta | Pair of callables or VectorDGField | Pair of CuPy-compatible callables or VectorDGField | Same-space VectorDGField | Same-space projected VectorDGField |
| Diffusion-reaction | source, reaction | Vectorized callable, scalar, or DGField | Reusable device path projected/constant subset | Same-space DGField | Same-space projected fields within the raw-kernel subset |
| Diffusion-reaction | diffusion | Positive scalar, supported tensor forms, or supported callable tensor components | Identity production subset | Identity or adapter-projected inverse tensor | Identity production subset |
| ADR | source, reaction | Callable, DG input or `ElementCoefficient` accepted by common preparation | No standalone CuPy assembly path | Common host preparation lowers the accepted input to tables | Common device preparation lowers the input to device tables; `ElementCoefficient` and CuPy source moments/values and reaction values stay on the device |
| ADR | beta | Pair of formulas, VectorDGField or two-component `ElementCoefficient` (sampled directly at volume, per-incidence face and recovery points); common preparation owns sampling | No standalone CuPy assembly path | Common preparation supplies sampled vector data | Common preparation supplies sampled device data (`ElementCoefficient` evaluated with `xp=cupy`) |
| ADR | diffusion | Positive scalar or elliptic tensor; constants, callables, DG fields | CuPy postprocessing only | Same inputs as NumPy, with exact per-element structural specialization | Same tensor inputs; p=0--6 assembly/reconstruction, tensor primal and both total-flux recoveries |

For variable/tensor ADR diffusion, the default stabilization uses the maximum
sampled `n^T K n` independently on each element-face incidence, divided by the
global physical length. NumPy/Numba and raw CUDA support all four recovery modes.
Primal recovery samples the inverse tensor at degree-p+1 recovery quadrature;
CuPy uses fused device assembly and batched pivoted LU. See the [Numba ADR guide](../backends/numba_adr.md).

Boundary conditions are solver-specific. Current diffusion-reaction and ADR
Dirichlet APIs accept scalar or callable trace data and project it internally.
Advection-reaction boundary modes are defined in
[Advection boundary and stabilization](advection_boundary_stabilization.md).

A vectorized scalar callable operates elementwise on array inputs:

    def reaction(x, y):
        return 1.5 + 0.2 * x**2 + 0.1 * x * y

Prefer array expressions such as where for short piecewise formulas. A Python
if statement branches on the whole array and is generally invalid. For many
regions or expensive branches, project once to a DGField and reuse it. A CuPy
direct-evaluation formula must use its received array namespace rather than
capturing NumPy-only operations.

## Stabilization matrix

| Solver/path | Advection stabilization | Diffusion stabilization |
| --- | --- | --- |
| Advection-reaction NumPy/CuPy | None selects conflict-averaged upwind for DG velocities, standard upwind for analytic callables; explicit upwind/ScaledUpwind/LF/conflict policy, scalar, callable, DGField, compatible incidence constants, or face table | Not applicable |
| Advection-reaction Numba | None selects conflict-averaged upwind for DG velocities; explicit upwind/ScaledUpwind/LF/conflict policy, scalar, or projected DGField | Not applicable |
| Advection-reaction raw CUDA | None selects conflict-averaged upwind; explicit upwind, ScaledUpwind, lax-friedrichs, or conflict-averaged-upwind | Not applicable |
| Diffusion-reaction NumPy/Numba | Not applicable | Default GlobalLengthDiffusion; positive scalar, shape (K,), or shape (K,3) explicit inputs |
| Diffusion-reaction CuPy/raw CUDA | Not applicable | Default GlobalLengthDiffusion resolved before dispatch, or an explicit positive scalar |
| ADR NumPy/Numba/raw CUDA | None for sidewise abs(beta.n), or a supported prepared input | Default GlobalLengthDiffusion; positive scalar, shape (K,), shape (K,3), same-mesh DGField, geometry/incidence callable, or face quadrature samples; explicit legacy inverse-h remains supported |

ADR keeps the advective and diffusive stabilization separate through
preparation and adds them only in the numerical flux. Every element/local-face
incidence retains its own value. Two elements sharing an interior face may
therefore supply unequal side values. Their weighted trace-mass contributions
are assembled separately and are never averaged implicitly.

The planned canonical callable signature is:

    def tau(x, y, *, element, local_face, normal, t=None):
        # x, y have shape (elements, local faces, face quadrature points)
        # normal is outward from the current element
        return ...

The output is scalar or broadcast-compatible with the (K,e,q) axes. Geometry
only tau(x,y) and legacy tau(x,y,K,e) adapters remain available where the
matrix says callable stabilization is currently supported. The complete common
incidence-aware adapter remains a TODO. Compiled consumers will receive its
evaluated table rather than the callable itself.

## Global physical-length diffusion policy

For positive constant isotropic diffusion, diffusion-reaction and ADR default
to and explicitly accept:

    from hdgfem import GlobalLengthDiffusion

    policy = GlobalLengthDiffusion(
        gamma_d=1.0,
        domain_length="auto",
    )

The policy resolves to

\[
\tau_d=\gamma_d\frac{\kappa}{L_\Omega},
\qquad
L_\Omega=\frac{2|\Omega|}{|\partial\Omega|}.
\]

An explicit positive domain length takes precedence over geometry. The
automatic length is recomputed from the current solver mesh, so changing the
mesh naturally invalidates the effective value. It is independent of
production mesh size and HDG degree. Scalar values, symmetric-component
isotropic tensors `(kappa, 0, kappa)`, and isotropic 2-by-2 matrices are
supported. Variable and anisotropic tensor normal diffusivity are not silently
approximated.

This policy is now the production default for diffusion-reaction and ADR.
ADR's legacy inverse-h rule remains available explicitly as
`--diffusion-stabilization-mode inverse-h`; `None` is retained only as a
core compatibility alias. Explicit positive scalar/incidence inputs continue
to override the policy.

The ADR manufactured runner exposes:

    --diffusion-stabilization-mode global-length
    --diffusion-domain-length auto
    --diffusion-stabilization-gamma 1

The diffusion-reaction runner exposes the corresponding flags.

## Lowering, caching, and transfers

| Input | NumPy/CuPy reference lowering | Numba/raw-CUDA lowering | Invalidation |
| --- | --- | --- | --- |
| Analytic PDE callable | Direct quadrature evaluation where supported | Projected fields; stabilization adapters may evaluate internally | Formula dependencies, coefficient identity, or time |
| DGField or VectorDGField | Basis contraction in its compatible space | Compact descriptor or host/device coefficient table | Field, space, or mesh |
| Scalar or lazy constant | Scalar/reference fast path | Scalar descriptor or existing scalar kernel argument | Value |
| GlobalLengthDiffusion | Resolve from diffusion and physical geometry | Pass the same resolved scalar to compiled/device code | Diffusion, mesh, explicit length, or gamma |
| Incidence callable, planned | Direct vectorized (K,e,q) evaluation | Cached internal (K,e,q) table | Formula dependencies, time, geometry, or quadrature |

Matching-residency CuPy/device fields are reused without a host download.
Host-born DG fields upload at an explicit backend boundary. Numba consumes host
storage. Formula projection or evaluation time belongs to preparation
diagnostics. Ordinary users do not manage the resulting quadrature tables.
