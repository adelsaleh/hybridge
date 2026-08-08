# Advection Boundary And Stabilization Contract

This reference defines how advection-reaction boundary modes, stabilization,
trace ownership, ordering, and reconstruction interact. Backend availability
remains governed by [`backend_capabilities.md`](backend_capabilities.md).

## Boundary Modes

| Mode | Exterior condition | Trace unknowns in the solve | Boundary data | Full trace after solve | Upwind-SCC graph |
|---|---|---|---|---|---|
| `penalty` | Dirichlet data imposed with the legacy large diagonal penalty | All global edges | Required and sampled | Already full; boundary coefficients are solved under the penalty equation | All trace edges |
| `eliminate` | Prescribed Dirichlet trace | Interior edges only | Required; nodal traces interpolate and modal traces project the data | Prescribed boundary coefficients are reinserted | Active interior edges only |
| `zero-flux` | Exterior numerical flux is forced to zero | Interior edges only | Must be `None`; any supplied callable or constant is rejected | Boundary slots are zero placeholders; their values do not enter reconstruction because boundary flux/lift weights are zero | Active interior edges for Numba; raw-CUDA requires `trace_ordering="none"` |

`zero-flux` is a physical numerical-flux choice, not shorthand for homogeneous
Dirichlet data. Use it only when the intended exterior flux is zero, such as a
tangent/no-through-flow boundary. Its public API requires
`boundary_condition=None`; it rejects rather than ignores supplied boundary
data. Omitting the boundary trace equations while
retaining their unknowns would create a singular block, so those degrees of
freedom are excluded from the reduced solve.

The public default remains `penalty` for compatibility. Performance-oriented
and new Dirichlet workflows should request `eliminate` explicitly until the
separate default-change task is completed.

## Stabilization

With side-normal flux `b = beta . n`, `advection_stabilization=None` selects
the upwind value `tau = abs(b)`. Assembly uses both `tau` and
`gamma = tau - b` on each element side. For an interior face, left and right
contributions remain distinct; discontinuous advection is not averaged.

| Assembly backend | Boundary modes | Accepted explicit `tau` inputs | Ordering |
|---|---|---|---|
| NumPy | `penalty`, `eliminate` | Scalar, callable `tau(x, y)` or `tau(x, y, K, e)`, `DGField`, compatible coefficient arrays, per-face constants, or evaluated face-quadrature tables | `none`, `upwind-scc` |
| Numba | `penalty`, `eliminate`, `zero-flux` | Scalar or same-space projected `DGField`; callable stabilization must be projected first | `none`, `upwind-scc` |
| CuPy | `penalty`, `eliminate` | Same forms as NumPy | `none`, `upwind-scc` with host graph construction |
| Raw CUDA | `eliminate`, `zero-flux` | None; explicit stabilization is rejected before device setup | `none` only; zero-flux also requires fused local assembly |

NumPy and CuPy sample analytic stabilization callables directly at face
quadrature points. A `DGField` has discrete rather than analytic semantics:
both backends contract its coefficient table with face-basis reference tables
from that field's `DGSpace`. CuPy performs this contraction on the device and
reuses device-resident coefficients when available.

For `zero-flux`, boundary-side `tau` and `gamma` weights are set to zero after
normal-flux evaluation. Interior-face stabilization is unchanged.

## Ordering, Reconstruction, And Residency

`trace_ordering="upwind-scc"` is built on exactly the degrees of freedom in
the reduced solve: all edges for `penalty`, and interior edges for
`eliminate` or `zero-flux`. It is a symmetric permutation of the trace system,
not an additional boundary condition or solver.

Host reconstruction consumes a full trace. Eliminated solves expand the
reduced solution with the prescribed boundary coefficients. Zero-flux solves
expand with zeros on boundary slots and use zero boundary lift weights, so
those placeholders cannot affect the reconstructed element field.

CuPy reconstruction consumes the full device trace directly. It gathers and
orients element traces on-device (nodal reversal or modal parity), rebuilds the
local operator and trace coupling from coefficient/reference tables, and uses a
batched device solve. Dense local inverses and element-boundary matrices are
discarded after condensation unless `cache_local_solvers=True` or an explicit
return key requests them. Host field/trace arrays are optional final copies.

For a compatible Cupyx global solve, CuPy also retains the assembled trace COO
values and RHS on-device. No host matrix/RHS copy occurs with no preconditioner,
a supplied device operator, or device ILU(1). A host solver, host ILU export,
matrix diagnostics, or `materialize_host_system=True` is an explicit residency
boundary and materializes the trace system before solving.

Raw-CUDA direct-CSR/AMGX runs may keep the reduced trace and reconstructed
field on the device. Host materialization changes residency only; it does not
change boundary ownership or stabilization semantics.
