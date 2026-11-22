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

**Meshes.** :func:`mesh_transport_blocks` takes the geometry of a
:class:`simplex_hdg_mesh.SimplexMesh` (``J = A_K``), :func:`solve_transport`
solves on a mesh with the trace prescribed on every boundary face (as
``boundary_mode="eliminate"`` of ``solve_advection_reaction_hdg``), and
:func:`transport_convergence` measures L2 errors on Kuhn box meshes for a smooth
solution.
"""

from __future__ import annotations

import numpy as np

from hybridge.core.quadrature import ReferenceElementData
from hybridge.hdg.stabilization import upwind_factor
from simplex_hdg_helpers import monomial_exponents
from simplex_hdg_mesh import (
    SimplexMesh,
    cell_load,
    cell_weighted_mass,
    face_rule,
    face_trace_projection,
    kuhn_box_mesh,
    l2_error,
    reference_element,
    smooth_solution,
    solve_condensed,
    volume_rule,
)


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


# ---------------------------------------------------------------------------
# Meshes: blocks with the geometry of a SimplexMesh, global solve, convergence
# ---------------------------------------------------------------------------


def mesh_transport_blocks(mesh: SimplexMesh, reference: ReferenceElementData, trace_basis: str, *, beta, alpha,
                          stabilization=None):
    """Return ``(A, B, C, D)`` of every cell of ``mesh``, stacked over the cells (shapes of :func:`advection_reaction_blocks`).

    The geometry is that of :class:`simplex_hdg_mesh.SimplexMesh`: the inverse
    ``A_K^{-1}`` of the matrix of the affine map ``x_K(xi) = A_K xi + b_K``, the
    Jacobian ``|det A_K|``, the ratios ``r_{K,f}``, the unit outward normals and the
    orientation of every cell face. ``beta`` ``(d,)`` or ``(n_K, d)`` and ``alpha``
    (scalar or ``(n_K,)``) are constant on each cell.
    """
    return advection_reaction_blocks(
        reference, trace_basis, beta=beta, alpha=alpha, stabilization=stabilization,
        inverse_jacobians=np.linalg.inv(mesh.affine_matrices), determinants=mesh.jacobians,
        face_ratios=mesh.face_ratios, normals=mesh.normals, orientations=mesh.orientations,
    )


def solve_transport(mesh: SimplexMesh, reference: ReferenceElementData, trace_basis: str, *, beta, alpha, source,
                    boundary, stabilization=None, rule=None, face_rule=None):
    """Solve the HDG discretization of ``beta . grad u + alpha u = f`` on ``mesh``; return a ``CondensedSolution``.

    The blocks are those of :func:`mesh_transport_blocks`, the loads
    ``int_K f phi_i`` (:func:`simplex_hdg_mesh.cell_load` on ``rule``), and the
    trace on every boundary face, inflow and outflow, is the L2 projection of
    ``g = boundary`` (:func:`simplex_hdg_mesh.face_trace_projection` on
    ``face_rule``); :func:`simplex_hdg_mesh.solve_condensed` solves for the
    interior faces. This is the treatment of ``boundary_mode="eliminate"`` of
    :func:`hybridge.solvers.advection_reaction.solve_advection_reaction_hdg`, which
    prescribes the projected data on every boundary edge (``"penalty"`` does the
    same through a large diagonal entry). With the upwind ``tau = |beta.n|`` an
    outflow face (``beta.n > 0``) has ``gamma = 0``, so its columns of ``B`` and
    ``D`` vanish and the prescribed trace does not enter ``u_h``: the flux there is
    ``beta.n u``, the natural outflow condition. With ``tau = k |beta.n|``, ``k > 1``,
    or an absolute ``tau`` the trace of an outflow face enters ``u_h``, as in hybridge.
    ``source`` and ``boundary`` are callables of physical points ``(n, d)``.
    """
    blocks = mesh_transport_blocks(mesh, reference, trace_basis, beta=beta, alpha=alpha, stabilization=stabilization)
    loads = (cell_load(mesh, reference, source, rule), np.zeros((mesh.num_cells, blocks[1].shape[2])))
    boundary_trace = face_trace_projection(mesh, reference, trace_basis, boundary, face_rule)
    return solve_condensed(mesh, blocks, loads, boundary_trace)


def interior_trace_matrix(mesh: SimplexMesh, solution) -> np.ndarray:
    """Return the dense matrix ``S_II`` of the trace system restricted to the interior faces (the matrix that is solved)."""
    n_t = solution.uhat.shape[1]
    free = np.flatnonzero(np.repeat(~mesh.boundary_faces, n_t))
    return solution.matrix[free][:, free].toarray()


def smooth_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient ``(n, d)`` of :func:`simplex_hdg_mesh.smooth_solution` ``u = exp(s / 2) cos(pi x_1 / 2)``, ``s = x_1 + ... + x_d``.

    ``d_1 u = exp(s / 2) (cos(pi x_1 / 2) / 2 - (pi / 2) sin(pi x_1 / 2))`` and
    ``d_k u = u / 2`` for ``k >= 2``.
    """
    x = np.asarray(x, dtype=np.float64)
    growth = np.exp(0.5 * np.sum(x, axis=1))
    angle = 0.5 * np.pi * x[:, 0]
    gradient = np.repeat((0.5 * growth * np.cos(angle))[:, None], x.shape[1], axis=1)
    gradient[:, 0] -= 0.5 * np.pi * growth * np.sin(angle)
    return gradient


def smooth_transport_source(beta, alpha: float):
    """Return the callable ``f = beta . grad u + alpha u`` of :func:`simplex_hdg_mesh.smooth_solution` at points ``(n, d)``."""
    beta = np.asarray(beta, dtype=np.float64)

    def source(x: np.ndarray) -> np.ndarray:
        """Evaluate ``f`` at physical points ``(n, d)``."""
        return smooth_gradient(x) @ beta + alpha * smooth_solution(x)

    return source


def transport_convergence(dim: int, order: int, subdivisions, *, beta, alpha: float = 1.0, stabilization=None,
                          basis: str = "orthonormal", trace_basis: str = "orthonormal"):
    """L2 errors of ``u_h`` and of the L2 projection of ``u`` on Kuhn box meshes of ``[-1, 1]^d``.

    The problem is ``beta . grad u + alpha u = f`` with the solution
    :func:`simplex_hdg_mesh.smooth_solution` (also the boundary data ``g``) and
    the source :func:`smooth_transport_source`; ``subdivisions`` lists the numbers
    ``N`` of cubes per direction. Loads, boundary projections and errors use rules
    exact for degree ``2p + 6``. Returns two lists of floats, the errors of ``u_h``
    and of the projection, one per ``N``.
    """
    rule, rule_on_faces = volume_rule(dim, 2 * order + 6), face_rule(dim, 2 * order + 6)
    reference = reference_element(dim, order, basis)
    source = smooth_transport_source(beta, alpha)
    errors, projection_errors = [], []
    for n in subdivisions:
        mesh = kuhn_box_mesh(dim, n)
        solution = solve_transport(mesh, reference, trace_basis, beta=beta, alpha=alpha, source=source,
                                   boundary=smooth_solution, stabilization=stabilization, rule=rule,
                                   face_rule=rule_on_faces)
        errors.append(l2_error(mesh, reference, solution.u, smooth_solution, rule))
        mass = cell_weighted_mass(mesh, reference, 1.0, rule)
        projection = np.linalg.solve(mass, cell_load(mesh, reference, smooth_solution, rule)[:, :, None])[:, :, 0]
        projection_errors.append(l2_error(mesh, reference, projection, smooth_solution, rule))
    return errors, projection_errors
