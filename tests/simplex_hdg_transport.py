"""Test helpers: HDG local blocks of ``beta . grad u + alpha u = f`` on affine simplices.

These helpers are test oracles for the derivative, mass, face-mass and
element-trace tables of :class:`hybridge.core.quadrature.ReferenceElementData`
on the triangle (``d = 2``) and the tetrahedron (``d = 3``). They form dense
local blocks with numpy and are not a second implementation of the transport
assembly of :mod:`hybridge.solvers.advection_host`.

**Form.** The velocity ``beta`` and the reaction ``alpha`` are constant on each
element. The local problem on an element ``K`` is the one assembled by
``assemble_transport_numpy`` in ``hybridge/solvers/advection_host.py``:

    -(u, beta . grad v)_K + <beta.n uhat + tau (u - uhat), v>_dK + (alpha u, v)_K = (f, v)_K

for every ``v`` in ``P_p(K)``, and the transmission row of a face ``F`` sums

    <beta.n_K uhat + tau_K (u_K - uhat), mu>_F

over the elements ``K`` that contain ``F``. Here ``n`` is the unit outward
normal of ``K``. The stabilization is that of
:func:`hybridge.hdg.stabilization.advection_trace_stabilization_values`: the
upwind value ``tau = factor |beta.n|`` (factor 1 by default, 2 for
``"lax-friedrichs"``, any positive factor for ``ScaledUpwind``) or an absolute
``tau`` given by the caller. The trace unknown is weighted by
``gamma = tau - beta.n`` (:func:`advection_trace_weights_from_normal_flux`).
The conflict-averaged repair of ``"conflict-averaged-upwind"`` changes nothing
for a velocity that is constant on both sides of a face, because there
``beta.n_l + beta.n_r = 0``.

**Blocks.** In the sign convention of TODO.md, "Form-Driven HDG Assembly",
``A u - B uhat = F`` and ``D uhat - C u = G``, with ``S = D - C A^{-1} B``:

    A = alpha |det J| M - |det J| sum_k b_k (D^k)^T + sum_f tau_f r_f E^f,
    B[:, f] = gamma_f r_f element_trace_coupling[o_f, f],
    C[f, :] = tau_f r_f trace_element_coupling[o_f (d + 1) + f],
    D = blockdiag_f(gamma_f r_f G),

where ``M``, ``D^k_ij = int phi_i d_k phi_j``, ``E^f``, the couplings and ``G``
are the reference tables, ``b = J^{-1} beta`` is the velocity in reference
coordinates (the physical gradient is ``grad v = J^{-T} grad^ v^``, so
``beta . grad v = b . grad^ v^``), ``r_f = |F_f| / |F^|`` is the ratio of the
measure of face ``f`` of ``K`` to the measure of the reference face, and
``o_f`` is the orientation of face ``f`` (the index of its vertex permutation
in :func:`hybridge.core.simplex.reference.face_permutations`). ``tau_f``,
``gamma_f`` and ``beta.n_f`` are constant on each face because ``beta`` is
constant and the element is affine.

**Layout.** Every geometric input has a leading element axis of length ``E``
and every block is returned stacked over it, so that one call serves many
elements. The defaults describe the reference element itself: ``J = I``,
``det J = 1``, the reference face ratios and unit outward normals, and the
identity orientation on every face.
"""

from __future__ import annotations

import numpy as np

from hybridge.core.quadrature import ReferenceElementData, _evaluate_basis
from hybridge.hdg.stabilization import upwind_factor
from simplex_hdg_helpers import monomial_exponents


def _stacked(value, default, tail_ndim: int) -> np.ndarray:
    """Return ``value`` (or ``default`` when ``value`` is None) as float64 with a leading element axis.

    An array with ``tail_ndim`` dimensions describes one element and gains a
    leading axis of length 1; an array with one more dimension is returned as
    it is.
    """
    array = np.asarray(default if value is None else value, dtype=np.float64)
    if array.ndim == tail_ndim:
        return array[None]
    if array.ndim != tail_ndim + 1:
        raise ValueError(f"expected an array with {tail_ndim} or {tail_ndim + 1} dimensions; got shape {array.shape}")
    return array


def element_geometry(
        reference: ReferenceElementData,
        *,
        beta,
        alpha,
        inverse_jacobians=None,
        determinants=None,
        face_ratios=None,
        normals=None,
        orientations=None,
) -> dict[str, np.ndarray]:
    """Return the coefficients and the affine geometry of ``E`` elements as arrays with a common element axis.

    Inputs, each for one element or stacked over ``E`` elements:
    ``beta`` ``(d,)`` or ``(E, d)``, the physical velocity; ``alpha`` scalar or
    ``(E,)``; ``inverse_jacobians`` ``(d, d)`` or ``(E, d, d)``, the inverse of
    the Jacobian ``J = dx/dxi`` of the affine map ``x = J xi + c``;
    ``determinants`` scalar or ``(E,)``, ``|det J|``; ``face_ratios``
    ``(d + 1,)`` or ``(E, d + 1)``, ``|F_f| / |F^|``; ``normals`` ``(d + 1, d)`` or
    ``(E, d + 1, d)``, the physical unit outward normals; ``orientations``
    ``(d + 1,)`` or ``(E, d + 1)``, face orientation indices in
    ``range(d!)``. The defaults are those of the reference element. ``E`` is the
    largest leading length; inputs of length 1 are repeated. Returns a dict with
    these keys and arrays of shapes ``(E, d)``, ``(E,)``, ``(E, d, d)``,
    ``(E,)``, ``(E, d + 1)``, ``(E, d + 1, d)`` and ``(E, d + 1)`` (int64).
    """
    dim = reference.dim
    arrays = {
        "beta": _stacked(beta, None, 1),
        "alpha": _stacked(alpha, None, 0),
        "inverse_jacobians": _stacked(inverse_jacobians, np.eye(dim), 2),
        "determinants": _stacked(determinants, 1.0, 0),
        "face_ratios": _stacked(face_ratios, reference.face_measure_ratios, 1),
        "normals": _stacked(normals, reference.reference_normals, 2),
        "orientations": _stacked(orientations, np.zeros(dim + 1), 1),
    }
    count = max(array.shape[0] for array in arrays.values())
    out = {}
    for name, array in arrays.items():
        if array.shape[0] not in (1, count):
            raise ValueError(f"{name} has {array.shape[0]} elements; expected 1 or {count}")
        out[name] = np.ascontiguousarray(np.broadcast_to(array, (count,) + array.shape[1:]))
    orientations = out["orientations"]
    if np.any(orientations != np.round(orientations)) or orientations.min() < 0 or orientations.max() >= reference.num_face_orientations:
        raise ValueError(f"orientations must be integers in range({reference.num_face_orientations})")
    out["orientations"] = orientations.astype(np.int64)
    if out["beta"].shape[1] != dim or out["normals"].shape[1:] != (dim + 1, dim):
        raise ValueError(f"beta must have {dim} components and normals shape (d + 1, d) = ({dim + 1}, {dim})")
    return out


def transport_face_weights(beta_dot_normal, stabilization=None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(tau, gamma)`` on every face from ``beta . n`` and the stabilization, as in hybridge.

    ``beta_dot_normal`` has shape ``(E, d + 1)``. ``stabilization`` takes the
    values of ``advection_stabilization`` in
    :func:`hybridge.hdg.stabilization.advection_trace_stabilization_values`:
    ``None``, ``"upwind"``, ``"conflict-averaged-upwind"`` (``tau = |beta.n|``),
    ``"lax-friedrichs"`` (``tau = 2 |beta.n|``), ``ScaledUpwind(factor)``
    (``tau = factor |beta.n|``), or an absolute ``tau`` as a scalar or an array
    that broadcasts to ``(E, d + 1)``. Returns ``tau`` and
    ``gamma = tau - beta.n``, both of shape ``(E, d + 1)``.
    """
    normal_flux = np.asarray(beta_dot_normal, dtype=np.float64)
    factor = upwind_factor(stabilization)
    if factor is not None:
        tau = factor * np.abs(normal_flux)
    else:
        tau = np.broadcast_to(np.asarray(stabilization, dtype=np.float64), normal_flux.shape).copy()
    return tau, tau - normal_flux


def advection_reaction_blocks(
        reference: ReferenceElementData,
        trace_basis: str,
        *,
        beta,
        alpha,
        stabilization=None,
        inverse_jacobians=None,
        determinants=None,
        face_ratios=None,
        normals=None,
        orientations=None,
):
    """Return ``(A, B, C, D)`` of the HDG discretization of ``beta . grad u + alpha u = f`` on ``E`` affine elements.

    The form, the stabilization and the block formulas are those of the module
    docstring; the geometric inputs are those of :func:`element_geometry` and
    default to the reference element, and ``stabilization`` is that of
    :func:`transport_face_weights`. With ``n_u`` volume and ``n_t`` trace
    functions per face, the shapes are ``A`` ``(E, n_u, n_u)``, ``B``
    ``(E, n_u, (d + 1) n_t)``, ``C`` ``(E, (d + 1) n_t, n_u)`` and ``D``
    ``(E, (d + 1) n_t, (d + 1) n_t)``; trace columns and rows are face-major. The
    trace basis of face ``f`` is the basis ``trace_basis`` in the global vertex
    order of the face, seen through the orientation ``o_f``.
    """
    geometry = element_geometry(
        reference, beta=beta, alpha=alpha, inverse_jacobians=inverse_jacobians, determinants=determinants,
        face_ratios=face_ratios, normals=normals, orientations=orientations,
    )
    tables = reference.trace_tables(trace_basis)
    num_faces = reference.num_faces
    count = geometry["beta"].shape[0]
    mass = reference.exact_mass.astype(np.float64)
    derivative = reference.derivative_matrices.astype(np.float64)
    face_mass = reference.face_mass.astype(np.float64)
    trace_mass = tables.trace_mass.astype(np.float64)
    n_u, n_t = mass.shape[0], trace_mass.shape[0]
    faces = np.arange(num_faces)[None, :]
    orient = geometry["orientations"]
    coupling = tables.element_trace_coupling.astype(np.float64)[orient, faces]  # (E, d+1, n_u, n_t)
    lift = tables.trace_element_coupling.astype(np.float64)[orient * num_faces + faces]  # (E, d+1, n_t, n_u)

    det = geometry["determinants"]
    ratios = geometry["face_ratios"]
    reference_velocity = np.einsum("Ekl,El->Ek", geometry["inverse_jacobians"], geometry["beta"])
    normal_flux = np.einsum("Efk,Ek->Ef", geometry["normals"], geometry["beta"])
    tau, gamma = transport_face_weights(normal_flux, stabilization)

    a = (
            (geometry["alpha"] * det)[:, None, None] * mass
            - det[:, None, None] * np.einsum("Ek,kji->Eij", reference_velocity, derivative)
            + np.einsum("Ef,fij->Eij", tau * ratios, face_mass)
    )
    b = np.einsum("Ef,Efia->Eifa", gamma * ratios, coupling).reshape(count, n_u, num_faces * n_t)
    c = np.einsum("Ef,Efai->Efai", tau * ratios, lift).reshape(count, num_faces * n_t, n_u)
    d = np.einsum("Ef,fg,ab->Efagb", gamma * ratios, np.eye(num_faces), trace_mass).reshape(
        count, num_faces * n_t, num_faces * n_t
    )
    return a, b, c, d


def volume_loads(reference: ReferenceElementData, source, *, jacobians=None, offsets=None, rule=None) -> np.ndarray:
    """Return ``F_i = int_K f phi_i dx`` on ``E`` affine elements, shape ``(E, n_u)``.

    ``source`` maps physical points ``(n, d)`` to values ``(n,)``. The element
    ``K`` is the image of the reference element under ``x = J xi + c`` with
    ``jacobians`` ``J`` (``(d, d)`` or ``(E, d, d)``, default the identity) and
    ``offsets`` ``c`` (``(d,)`` or ``(E, d)``, default zero), so
    ``F_i = |det J| int_K^ f(J xi + c) phi_i(xi) dxi``. ``rule`` is
    ``(points, weights)`` on the reference element; the default is exact for
    degree ``2p``, so pass a finer rule when ``f`` is not a polynomial of degree
    at most ``p``.
    """
    dim = reference.dim
    points, weights = reference._generic_volume_rule if rule is None else rule
    jacobian = _stacked(jacobians, np.eye(dim), 2)
    offset = _stacked(offsets, np.zeros(dim), 1)
    count = max(jacobian.shape[0], offset.shape[0])
    jacobian = np.broadcast_to(jacobian, (count, dim, dim))
    offset = np.broadcast_to(offset, (count, dim))
    mapped = offset[:, None, :] + np.einsum("Ekl,ql->Eqk", jacobian, points)
    values = np.asarray(source(mapped.reshape(-1, dim)), dtype=np.float64).reshape(count, len(weights))
    basis = np.asarray(_evaluate_basis(reference.basis_type, reference.order, points), dtype=np.float64)
    return np.abs(np.linalg.det(jacobian))[:, None] * ((values * weights) @ basis)


def local_solve(a, b, loads, uhat) -> np.ndarray:
    """Return the element coefficients ``u = A^{-1} (F + B uhat)`` of every element, shape ``(E, n_u)``."""
    rhs = np.asarray(loads, dtype=np.float64) + np.einsum("Eia,Ea->Ei", b, np.asarray(uhat, dtype=np.float64))
    return np.linalg.solve(a, rhs[..., None])[..., 0]


def polynomial_gradient(dim: int, degree: int, seed: int):
    """Return the gradient of :func:`simplex_hdg_helpers.random_polynomial` ``(dim, degree, seed)``.

    The returned callable maps points ``(n, dim)`` to gradients ``(n, dim)``. It
    draws the same coefficients as ``random_polynomial``: one standard normal
    number per exponent of :func:`simplex_hdg_helpers.monomial_exponents`, in
    that order, from ``numpy.random.default_rng(seed)``.
    """
    rng = np.random.default_rng(seed)
    exponents = np.asarray(monomial_exponents(dim, degree))
    coefficients = rng.standard_normal(len(exponents))

    def evaluate(points: np.ndarray) -> np.ndarray:
        """Evaluate the gradient at ``points`` of shape ``(n, dim)``."""
        points = np.asarray(points, dtype=np.float64)
        gradient = np.zeros(points.shape)
        for c, a in zip(coefficients, exponents):
            for k in range(dim):
                if a[k] == 0:
                    continue
                lowered = a.copy()
                lowered[k] -= 1
                gradient[:, k] += c * a[k] * np.prod(points ** lowered, axis=1)
        return gradient

    return evaluate
