r"""Reference-triangle quadrature, basis tabulation, and cached tensors.

The reference element is

.. math::

    \hat K = \operatorname{conv}\{(-1,-1), (1,-1), (-1,1)\}.

This module owns the reference-element data used by :class:`hdgfem.core.space.DGSpace`.
It does not know about a physical mesh; geometric scaling by element Jacobians
is applied later by mesh- and assembly-level code.  All arrays here are
therefore reference arrays, stored once per polynomial order and basis family.

The default basis is the Bernstein basis of total degree ``p``.  Bernstein
functions form a partition of unity, are easy to evaluate and differentiate in
vectorized NumPy, and provide a stable first internal basis.  The module also
supports the legacy hierarchical C0 and Dubiner-style bases.  Public attribute
names intentionally mirror the legacy quadrature object where that keeps
assembly kernels simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from . import basis as basis_module


_VOLUME_QUADRATURE_ALIASES = {
    "auto": "auto",
    "default": "auto",
    "symmetric": "symmetric",
    "dunavant": "symmetric",
    "duffy": "duffy",
    "legacy": "duffy",
    "collapsed": "duffy",
}

# Positive-weight Dunavant rules on the unit triangle.  Each entry is
# ``(barycentric representative, weight per orbit point)``; permutations are
# expanded below and weights are scaled by two for our area-two reference
# triangle.
_DUNAVANT_RULE_BLOCKS = {
    2: (((2 / 3, 1 / 6, 1 / 6), 1 / 3),),
    4: (
        ((0.108103018168070, 0.445948490915965, 0.445948490915965), 0.223381589678011),
        ((0.816847572980459, 0.091576213509771, 0.091576213509771), 0.109951743655322),
    ),
    6: (
        ((0.501426509658179, 0.249286745170910, 0.249286745170910), 0.116786275726379),
        ((0.873821971016996, 0.063089014491502, 0.063089014491502), 0.050844906370207),
        ((0.053145049844816, 0.310352451033785, 0.636502499121399), 0.082851075618374),
    ),
    8: (
        ((1 / 3, 1 / 3, 1 / 3), 0.144315607677787),
        ((0.081414823414554, 0.459292588292723, 0.459292588292723), 0.095091634267285),
        ((0.658861384496480, 0.170569307751760, 0.170569307751760), 0.103217370534718),
        ((0.898905543365938, 0.050547228317031, 0.050547228317031), 0.032458497623198),
        ((0.008394777409958, 0.263112829634638, 0.728492392955404), 0.027230314174435),
    ),
    10: (
        ((1 / 3, 1 / 3, 1 / 3), 0.090817990382754),
        ((0.028844733232685, 0.485577633383657, 0.485577633383657), 0.036725957756467),
        ((0.781036849029926, 0.109481575485037, 0.109481575485037), 0.045321059435528),
        ((0.141707219414880, 0.307939838764121, 0.550352941820999), 0.072757916845420),
        ((0.025003534762686, 0.246672560639903, 0.728323904597411), 0.028327242531057),
        ((0.009540815400299, 0.066803251012200, 0.923655933587500), 0.009421666963733),
    ),
    12: (
        ((0.023565220452390, 0.488217389773805, 0.488217389773805), 0.025731066440455),
        ((0.120551215411080, 0.439724392294460, 0.439724392294460), 0.043692544538038),
        ((0.457579229975768, 0.271210385012116, 0.271210385012116), 0.062858224217885),
        ((0.744847708916828, 0.127576145541586, 0.127576145541586), 0.034796112930709),
        ((0.957365299093580, 0.021317350453210, 0.021317350453210), 0.006166261051559),
        ((0.115343494534698, 0.275713269685514, 0.608943235779788), 0.040371557766381),
        ((0.022838332222257, 0.281325580989940, 0.695836086787803), 0.022356773202303),
        ((0.025734050548330, 0.116251915907597, 0.858014033544073), 0.017316231108659),
    ),
    14: (
        ((0.022072179275643, 0.488963910362179, 0.488963910362179), 0.021883581369429),
        ((0.164710561319092, 0.417644719340454, 0.417644719340454), 0.032788353544125),
        ((0.453044943382323, 0.273477528308839, 0.273477528308839), 0.051774104507292),
        ((0.645588935174913, 0.177205532412543, 0.177205532412543), 0.042162588736993),
        ((0.876400233818255, 0.061799883090873, 0.061799883090873), 0.014433699669777),
        ((0.961218077502598, 0.019390961248701, 0.019390961248701), 0.004923403602400),
        ((0.057124757403648, 0.172266687821356, 0.770608554774996), 0.024665753212564),
        ((0.092916249356972, 0.336861459796345, 0.570222290846683), 0.038571510787061),
        ((0.014646950055654, 0.298372882136258, 0.686980167808088), 0.014436308113534),
        ((0.001268330932872, 0.118974497696957, 0.879757171370171), 0.005010228838501),
    ),
}


def _normalize_volume_quadrature(name: str) -> str:
    """Normalize a volume quadrature alias to its canonical name."""
    try:
        return _VOLUME_QUADRATURE_ALIASES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError("volume_quadrature must be 'auto', 'symmetric', or 'duffy'") from exc


def _symmetric_triangle_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a positive-weight symmetric rule exact for ``P_(2*order)``.

    Compact Dunavant data is used when available.  At higher orders, a
    minimally exact ``(order + 1)^2`` Duffy rule is expanded into complete
    barycentric permutation orbits.  Orbit symmetrization preserves the
    original rule's polynomial exactness and positivity while providing a
    fully symmetric rule for arbitrary nonnegative ``order``.
    """
    exact_degree = max(2, 2 * order)
    blocks = _DUNAVANT_RULE_BLOCKS.get(exact_degree)
    if blocks is None:
        points, base_weights = _triangle_quadrature(order, order + 1)
        barycentric = np.column_stack(
            (
                -0.5 * (points[:, 0] + points[:, 1]),
                0.5 * (points[:, 0] + 1.0),
                0.5 * (points[:, 1] + 1.0),
            )
        )
        orbit_points = []
        orbit_weights = []
        for bary, weight in zip(barycentric, base_weights):
            orbit = sorted(set(permutations(map(float, bary))))
            orbit = np.asarray(orbit, dtype=np.float64)
            orbit_points.append(
                np.column_stack((-1.0 + 2.0 * orbit[:, 1], -1.0 + 2.0 * orbit[:, 2]))
            )
            orbit_weights.append(np.full(len(orbit), weight / len(orbit), dtype=np.float64))
        return (
            np.ascontiguousarray(np.vstack(orbit_points)),
            np.ascontiguousarray(np.concatenate(orbit_weights)),
        )

    barycentric = []
    weights = []
    for bary, weight in blocks:
        orbit = sorted(set(permutations(map(float, bary))))
        barycentric.extend(orbit)
        weights.extend([2.0 * weight] * len(orbit))
    barycentric = np.asarray(barycentric, dtype=np.float64)
    points = np.column_stack((-1.0 + 2.0 * barycentric[:, 1], -1.0 + 2.0 * barycentric[:, 2]))
    return np.ascontiguousarray(points), np.ascontiguousarray(weights, dtype=np.float64)


def _automatic_triangle_quadrature(order: int) -> tuple[str, np.ndarray, np.ndarray]:
    """Select the default volume rule for one polynomial order.

    The automatic policy prefers compact admissible Dunavant data.  Once that
    table is exhausted, it returns the legacy collapsed tensor-product rule
    rather than the generated symmetric rule, whose point count is larger than
    Duffy at high order.  Callers can still force the generated rule with
    ``volume_quadrature="symmetric"``.
    """
    exact_degree = max(2, 2 * order)
    if exact_degree in _DUNAVANT_RULE_BLOCKS:
        points, weights = _symmetric_triangle_quadrature(order)
        return "symmetric", points, weights
    points, weights = _triangle_quadrature(order, None)
    return "duffy", points, weights


_BASIS_ALIASES = {
    "bernstein": "bernstein",
    "bern": "bernstein",
    "hier_c0": "hier_C0",
    "hierarchical_c0": "hier_C0",
    "c0": "hier_C0",
    "cg": "hier_C0",
    "dub_orth": "dub_orth",
    "dubiner": "dub_orth",
    "dubiner_orthonormal": "dub_orth",
}


def _normalize_basis_type(name: str) -> str:
    """Return the canonical basis-family name used internally.

    User-facing constructors accept a small set of aliases such as ``"bern"``
    and ``"dubiner"``.  This helper converts those aliases to the exact names
    dispatched by :func:`_evaluate_basis` and :func:`_evaluate_gradients`.
    """
    try:
        return _BASIS_ALIASES[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported basis_type {name!r}; supported bases are "
            "'bernstein', 'hier_C0', and 'dub_orth'"
        ) from exc


def _quadrature_point_count(order: int, num_1d: int | None) -> int:
    """Resolve and validate the one-dimensional quadrature point count."""
    if num_1d is None:
        return max(2 * order + 2, 2)
    count = int(num_1d)
    if count < 1:
        raise ValueError("quadrature point count must be positive")
    return count


def _triangle_quadrature(order: int, num_1d: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build a collapsed tensor-product Gauss rule on the reference triangle.

    A tensor Gauss-Legendre rule on ``[-1, 1]^2`` is mapped to the reference
    triangle with the Duffy-style transform

    ``x = -1 + 2*u*(1-v)``, ``y = -1 + 2*u*v``.

    The returned points have shape ``(num_quads, 2)`` and weights have shape
    ``(num_quads,)``.  The weights integrate on the reference triangle only;
    callers multiply by each physical element Jacobian when assembling.
    """
    num_1d = _quadrature_point_count(order, num_1d)
    s, ws = np.polynomial.legendre.leggauss(num_1d)
    t, wt = np.polynomial.legendre.leggauss(num_1d)
    ss, tt = np.meshgrid(s, t, indexing="xy")
    u = 0.5 * (ss + 1.0)
    v = 0.5 * (tt + 1.0)
    x = -1.0 + 2.0 * u * (1.0 - v)
    y = -1.0 + 2.0 * u * v
    weights = (ws[None, :] * wt[:, None] * u).ravel()
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    return np.ascontiguousarray(points), np.ascontiguousarray(weights)


def _edge_quadrature(order: int, num_1d: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build a one-dimensional Gauss rule for reference edges.

    The same rule is used on all three reference faces.  Points and weights are
    returned in the canonical interval coordinate ``t in [-1, 1]`` with shape
    ``(num_face_quads,)``.
    """
    num_1d = _quadrature_point_count(order, num_1d)
    points, weights = np.polynomial.legendre.leggauss(num_1d)
    return np.ascontiguousarray(points), np.ascontiguousarray(weights)


def _edge_points(edge_points_1d: np.ndarray) -> np.ndarray:
    """Map 1D edge quadrature nodes onto the three reference-triangle faces.

    The output has shape ``(num_face_quads, 3, 2)``.  The face axis uses this
    ordering:

    ``0``: bottom edge ``(t, -1)``
    ``1``: diagonal edge ``(-t, t)``
    ``2``: left edge ``(-1, -t)``

    This orientation is paired with ``_edge_points(-edge_points_1d)`` below to
    build plus/minus edge coupling tables for neighboring elements.
    """
    t = edge_points_1d
    return np.ascontiguousarray(
        np.stack(
            (
                np.stack((t, -np.ones_like(t)), axis=1),
                np.stack((-t, t), axis=1),
                np.stack((-np.ones_like(t), -t), axis=1),
            ),
            axis=1,
        )
    )


def _legendre_gauss_lobatto(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre-Gauss-Lobatto nodes and weights on ``[-1, 1]``."""
    if num_points < 1:
        raise ValueError("Gauss-Lobatto rule needs at least one point")
    if num_points == 1:
        return np.array([0.0]), np.array([2.0])
    if num_points == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    poly = np.polynomial.legendre.Legendre.basis(num_points - 1)
    interior = np.sort(poly.deriv().roots())
    points = np.concatenate(([-1.0], interior, [1.0]))
    values = poly(points)
    weights = 2.0 / ((num_points - 1) * num_points * values * values)
    return np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(weights, dtype=np.float64)


def _lagrange_basis(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate 1D nodal Lagrange basis functions at ``points``."""
    nodes = np.asarray(nodes, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    values = np.ones((nodes.size, points.size), dtype=np.float64)
    for i in range(nodes.size):
        for j in range(nodes.size):
            if i != j:
                values[i] *= (points - nodes[j]) / (nodes[i] - nodes[j])
    return np.ascontiguousarray(values)


def _edge_basis(order: int, edge_points_1d: np.ndarray) -> np.ndarray:
    """Evaluate the default nodal Lagrange trace basis on reference-edge points.

    The trace degrees of freedom are located at ``order + 1``
    Legendre-Gauss-Lobatto nodes on ``[-1, 1]``.  This keeps package solvers on
    the same nodal trace convention as the standalone GPU runners'
    ``legacy-lagrange`` trace basis.
    """
    nodes, _ = _legendre_gauss_lobatto(order + 1)
    return _lagrange_basis(nodes, edge_points_1d)


def _evaluate_basis(basis_type: str, order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the selected 2D basis at reference points.

    ``points`` must have shape ``(num_points, 2)`` in reference coordinates.
    The result has shape ``(num_points, el_dof)`` and is contiguous whenever the
    underlying basis evaluator returns contiguous storage.
    """
    if basis_type == "bernstein":
        return basis_module.evaluate_bernstein_basis(order, points)
    if basis_type == "hier_C0":
        return basis_module.evaluate_hierarchical_c0_basis(order, points)
    if basis_type == "dub_orth":
        return basis_module.evaluate_dubiner_basis(order, points)
    raise ValueError(f"unsupported basis_type {basis_type!r}")


def _evaluate_gradients(basis_type: str, order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate reference-coordinate gradients of the selected basis.

    The returned array has shape ``(num_points, el_dof, 2)``.  The last axis is
    ``(d/dxi, d/deta)`` on the reference triangle; physical gradients are
    obtained later by multiplying with the inverse-transpose element map.
    """
    if basis_type == "bernstein":
        return basis_module.evaluate_bernstein_gradients(order, points)
    if basis_type == "hier_C0":
        return basis_module.evaluate_hierarchical_c0_gradients(order, points)
    if basis_type == "dub_orth":
        return basis_module.evaluate_dubiner_gradients(order, points)
    raise ValueError(f"unsupported basis_type {basis_type!r}")


@dataclass(frozen=True)
class ReferenceElementData:
    """Reference data for one triangular polynomial space.

    A :class:`ReferenceElementData` instance is immutable after construction and
    contains only reference-element quantities.  It is shared by DG spaces with
    the same polynomial order and basis family and is deliberately shaped for
    vectorized assembly.

    Important shapes
    ----------------
    ``el_dof``
        Number of scalar element-local basis functions,
        ``(order + 1) * (order + 2) // 2``.
    ``edg_dof``
        Number of trace basis functions per edge, ``order + 1``.
    ``Krf_quads`` / ``Krf_w``
        Volume quadrature points and weights on the reference triangle with
        shapes ``(num_quads, 2)`` and ``(num_quads,)``.
    ``phi`` / ``bas_of_quads``
        Volume basis values in two layouts: ``phi`` is
        ``(num_quads, el_dof)`` and ``bas_of_quads`` is
        ``(el_dof, num_quads)``.
    ``gphi`` / ``dbas_of_quads``
        Reference gradients in layouts ``(num_quads, el_dof, 2)`` and
        ``(2, el_dof, num_quads)``.
    ``weighted_phi_phi_flat``
        Flattened ``q``-indexed tables for mass-like contractions,
        ``sum_q w_q f_q phi_i(q) phi_j(q)``.
    ``weighted_triple_phi_flat``
        Flattened triple products used when a same-space DG field is the
        coefficient in a weighted mass matrix.

    Face coupling tables
    --------------------
    The following tables are reference-element integrals over faces of the
    reference triangle.  The names describe the row/test and column/trial
    convention used by assembly code:

    ``face_element_test_trace_trial[f, i, a]``
        Coupling between element test basis ``phi_i`` and trace trial basis
        ``mu_a`` on local face ``f`` with the positive local trace orientation.
        Shape ``(3, el_dof, edg_dof)``.
    ``face_element_test_trace_trial_reversed[f, i, a]``
        Same coupling with the trace parametrization reversed.  Shape
        ``(3, el_dof, edg_dof)``.
    ``face_trace_test_element_trial_oriented[o, a, i]``
        Oriented transpose table used for trace-tested quantities.  The
        orientation-table index ``o`` uses ``0..2`` for positive local faces
        and ``3..5`` for reversed local faces.  Shape
        ``(6, edg_dof, el_dof)``.
    ``face_element_test_element_trial[f, i, j]``
        Face mass table between element test and element trial bases restricted
        to local face ``f``.  Shape ``(3, el_dof, el_dof)``.

    Legacy attributes ``MKrfe_lst_p``, ``MKrfe_lst_n``, ``MKrfe_lst``, and
    ``MbdeKrf_lst`` are aliases of these semantic names.
    """

    order: int
    basis_type: str = "bernstein"
    verbosity: int = 0
    cache: bool = True
    volume_quadrature: str = "auto"
    volume_quad_1d: int | None = None
    edge_quad_1d: int | None = None
    el_dof: int = field(init=False)
    edg_dof: int = field(init=False)
    Krf_quads: np.ndarray = field(init=False)
    Krf_w: np.ndarray = field(init=False)
    bas_of_quads: np.ndarray = field(init=False)
    dbas_of_quads: np.ndarray = field(init=False)
    ref_tri_verts: np.ndarray = field(init=False)
    quads_JGL: np.ndarray = field(init=False)
    weights_JGL: np.ndarray = field(init=False)
    rf_edg_lag_nodes: np.ndarray = field(init=False)
    pts_fc: np.ndarray = field(init=False)
    bas_of_bd_quads: np.ndarray = field(init=False)
    bas1d_of_ref_edg_qds: np.ndarray = field(init=False)
    face_element_test_trace_trial: np.ndarray = field(init=False)
    face_element_test_trace_trial_reversed: np.ndarray = field(init=False)
    face_trace_test_element_trial_oriented: np.ndarray = field(init=False)
    face_element_test_element_trial: np.ndarray = field(init=False)
    MKrfe_lst_p: np.ndarray = field(init=False)
    MKrfe_lst_n: np.ndarray = field(init=False)
    MKrfe_lst: np.ndarray = field(init=False)
    M_rf_fc: np.ndarray = field(init=False)
    M_rf_fc_f: np.ndarray = field(init=False)
    MbdeKrf_lst: np.ndarray = field(init=False)
    MKrf: np.ndarray = field(init=False)
    MKrf_inv: np.ndarray = field(init=False)
    phi: np.ndarray = field(init=False)
    gphi: np.ndarray = field(init=False)
    phi_f: np.ndarray = field(init=False)
    mu: np.ndarray = field(init=False)
    weighted_phi: np.ndarray = field(init=False)
    weighted_phi_phi_flat: np.ndarray = field(init=False)
    weighted_triple_phi_flat: np.ndarray = field(init=False)
    weighted_bas_of_bd_quads: np.ndarray = field(init=False)
    weighted_bas1d_of_ref_edg_qds: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Validate options and tabulate all reference-element data."""
        order = int(self.order)
        if order < 0:
            raise ValueError("order must be nonnegative")
        basis_type = _normalize_basis_type(self.basis_type)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "basis_type", basis_type)
        volume_quadrature = _normalize_volume_quadrature(self.volume_quadrature)
        if self.volume_quad_1d is not None:
            volume_quadrature = "duffy"
        object.__setattr__(self, "el_dof", (order + 1) * (order + 2) // 2)
        object.__setattr__(self, "edg_dof", order + 1)

        if volume_quadrature == "auto":
            volume_quadrature, q_points, q_weights = _automatic_triangle_quadrature(order)
        elif volume_quadrature == "symmetric":
            q_points, q_weights = _symmetric_triangle_quadrature(order)
        else:
            q_points, q_weights = _triangle_quadrature(order, self.volume_quad_1d)
        object.__setattr__(self, "volume_quadrature", volume_quadrature)
        basis = _evaluate_basis(basis_type, order, q_points)
        gradients = _evaluate_gradients(basis_type, order, q_points)
        phi = np.ascontiguousarray(basis)
        basis_iq = np.ascontiguousarray(basis.T)
        gradients_qid = np.ascontiguousarray(gradients)
        gradients_diq = np.ascontiguousarray(gradients.swapaxes(0, 2))
        object.__setattr__(self, "Krf_quads", q_points)
        object.__setattr__(self, "Krf_w", q_weights)
        object.__setattr__(self, "bas_of_quads", basis_iq)
        object.__setattr__(self, "dbas_of_quads", gradients_diq)
        object.__setattr__(
            self,
            "ref_tri_verts",
            np.array(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)), dtype=np.float64),
        )

        edge_points, edge_weights = _edge_quadrature(order, self.edge_quad_1d)
        face_points = _edge_points(edge_points)
        face_basis = _evaluate_basis(basis_type, order, face_points.reshape(-1, 2))
        face_basis = face_basis.reshape(edge_points.size, 3, self.el_dof).transpose(1, 2, 0)
        negative_face_points = _edge_points(-edge_points)
        negative_face_basis = _evaluate_basis(basis_type, order, negative_face_points.reshape(-1, 2))
        negative_face_basis = negative_face_basis.reshape(edge_points.size, 3, self.el_dof).transpose(1, 2, 0)
        object.__setattr__(self, "quads_JGL", edge_points)
        object.__setattr__(self, "weights_JGL", edge_weights)
        trace_nodes, _ = _legendre_gauss_lobatto(order + 1)
        object.__setattr__(self, "rf_edg_lag_nodes", trace_nodes)
        object.__setattr__(self, "pts_fc", face_points)
        object.__setattr__(self, "bas_of_bd_quads", np.ascontiguousarray(face_basis))
        edge_basis = _edge_basis(order, edge_points)
        object.__setattr__(self, "bas1d_of_ref_edg_qds", edge_basis)
        object.__setattr__(
            self,
            "weighted_bas_of_bd_quads",
            np.ascontiguousarray(face_basis * edge_weights[None, None, :]),
        )
        object.__setattr__(
            self,
            "weighted_bas1d_of_ref_edg_qds",
            np.ascontiguousarray(edge_basis * edge_weights[None, :]),
        )

        face_element_test_trace_trial = np.einsum(
            "q,fiq,jq->fij",
            edge_weights,
            face_basis,
            edge_basis,
            optimize=True,
        )
        face_element_test_trace_trial_reversed = np.einsum(
            "q,fiq,jq->fij",
            edge_weights,
            negative_face_basis,
            edge_basis,
            optimize=True,
        )
        face_element_test_trace_trial = np.ascontiguousarray(face_element_test_trace_trial)
        face_element_test_trace_trial_reversed = np.ascontiguousarray(face_element_test_trace_trial_reversed)
        face_trace_test_element_trial_oriented = np.ascontiguousarray(
            np.concatenate(
                (
                    face_element_test_trace_trial.transpose(0, 2, 1),
                    face_element_test_trace_trial_reversed.transpose(0, 2, 1),
                ),
                axis=0,
            )
        )
        object.__setattr__(self, "face_element_test_trace_trial", face_element_test_trace_trial)
        object.__setattr__(
            self,
            "face_element_test_trace_trial_reversed",
            face_element_test_trace_trial_reversed,
        )
        object.__setattr__(
            self,
            "face_trace_test_element_trial_oriented",
            face_trace_test_element_trial_oriented,
        )
        object.__setattr__(self, "MKrfe_lst_p", face_element_test_trace_trial)
        object.__setattr__(self, "MKrfe_lst_n", face_element_test_trace_trial_reversed)
        object.__setattr__(self, "MKrfe_lst", face_trace_test_element_trial_oriented)
        edge_mass = np.einsum(
            "q,iq,jq->ij",
            edge_weights,
            edge_basis,
            edge_basis,
            optimize=True,
        )
        object.__setattr__(self, "M_rf_fc", np.ascontiguousarray(edge_mass))
        object.__setattr__(self, "M_rf_fc_f", np.ascontiguousarray(edge_mass.ravel()))
        face_element_test_element_trial = np.einsum(
            "q,fiq,fjq->fij",
            edge_weights,
            face_basis,
            face_basis,
            optimize=True,
        )
        face_element_test_element_trial = np.ascontiguousarray(face_element_test_element_trial)
        object.__setattr__(self, "face_element_test_element_trial", face_element_test_element_trial)
        object.__setattr__(self, "MbdeKrf_lst", face_element_test_element_trial)

        weighted_phi = np.ascontiguousarray(q_weights[:, None] * phi)
        weighted_phi_phi = np.einsum("q,qi,qj->qij", q_weights, phi, phi, optimize=True)
        weighted_triple_phi = np.einsum(
            "q,qk,qi,qj->kij",
            q_weights,
            phi,
            phi,
            phi,
            optimize=True,
        )
        mass = np.einsum("q,iq,jq->ij", q_weights, basis_iq, basis_iq, optimize=True)
        object.__setattr__(self, "MKrf", np.ascontiguousarray(mass))
        object.__setattr__(self, "MKrf_inv", np.ascontiguousarray(np.linalg.inv(mass)))
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "gphi", gradients_qid)
        object.__setattr__(self, "phi_f", np.ascontiguousarray(face_basis.swapaxes(1, 2)))
        object.__setattr__(self, "mu", np.ascontiguousarray(edge_basis.T))
        object.__setattr__(self, "weighted_phi", weighted_phi)
        object.__setattr__(
            self,
            "weighted_phi_phi_flat",
            np.ascontiguousarray(weighted_phi_phi.reshape(q_weights.size, self.el_dof * self.el_dof)),
        )
        object.__setattr__(
            self,
            "weighted_triple_phi_flat",
            np.ascontiguousarray(weighted_triple_phi.reshape(self.el_dof, self.el_dof * self.el_dof)),
        )

    @classmethod
    def triangle(
            cls,
            polynomial_order: int,
            *,
            basis_type: str = "bernstein",
            verbosity: int = 0,
            cache: bool = True,
            volume_quadrature: str = "auto",
            volume_quad_1d: int | None = None,
            edge_quad_1d: int | None = None,
    ) -> "ReferenceElementData":
        """Build reference data for a triangular DG space.

        This classmethod mirrors the historical construction API used by the
        rest of the package.  ``polynomial_order`` is passed through as
        ``order`` and the returned object contains all volume, edge, and
        preweighted reference tensors needed by :class:`DGSpace`.
        """
        return cls(
            polynomial_order,
            basis_type=basis_type,
            verbosity=verbosity,
            cache=cache,
            volume_quadrature=volume_quadrature,
            volume_quad_1d=volume_quad_1d,
            edge_quad_1d=edge_quad_1d,
        )

    @property
    def quadrature(self) -> "ReferenceElementData":
        """Compatibility property: reference data is its own quadrature object.

        Some legacy-facing assembly code expects a separate ``quadrature``
        object.  In the reorganized core, the quadrature data and reference
        basis tables live on the same object, so this property returns ``self``.
        """
        return self

    def basis_at(self, reference_points: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at reference points.

        Passing ``self.Krf_quads`` by object identity returns the precomputed
        ``phi`` table instead of retabulating.  Other point arrays are evaluated
        directly and are not cached at this level; :class:`DGSpace` maintains a
        small identity-based cache for repeated user-level calls.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(num_points, el_dof)``.
        """
        points = np.asarray(reference_points, dtype=np.float64)
        if points is self.Krf_quads:
            return self.phi
        return _evaluate_basis(self.basis_type, self.order, points)

    def gradients_at(self, reference_points: np.ndarray) -> np.ndarray:
        """Evaluate reference gradients at reference points.

        Passing ``self.Krf_quads`` by object identity returns the precomputed
        ``gphi`` table.  The gradients are with respect to reference coordinates
        only; physical gradient conversion is handled by :meth:`DGField.grad_at_ref`.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(num_points, el_dof, 2)``.
        """
        points = np.asarray(reference_points, dtype=np.float64)
        if points is self.Krf_quads:
            return self.gphi
        return _evaluate_gradients(self.basis_type, self.order, points)

    evaluate_basis = basis_at
    evaluate_gradients = gradients_at


TriangleQuadratureData = ReferenceElementData


def as_reference_element(reference: ReferenceElementData) -> ReferenceElementData:
    """Validate and return a :class:`ReferenceElementData` instance.

    This small adapter keeps call sites explicit when they accept "reference
    data" conceptually but currently support only the concrete
    :class:`ReferenceElementData` implementation.
    """
    if isinstance(reference, ReferenceElementData):
        return reference
    raise TypeError(f"expected ReferenceElementData, got {type(reference)!r}")
