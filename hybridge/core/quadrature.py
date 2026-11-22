r"""Reference-triangle quadrature, basis tabulation, and cached tensors.

The reference element is

.. math::

    \hat K = \operatorname{conv}\{(-1,-1), (1,-1), (-1,1)\}.

This module owns the reference-element data used by :class:`hybridge.core.space.DGSpace`.
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

from hybridge.runtime.precision import REAL_DTYPE

import dataclasses
from dataclasses import dataclass, field
from functools import cached_property
from math import comb, factorial

import numpy as np

from hybridge.core import basis as basis_module
from hybridge.core.simplex.polynomials import _lagrange_basis
from hybridge.core.simplex.reference import (
    _DUNAVANT_RULE_BLOCKS,
    _compact_dunavant_degree,
    _duffy_points_for_degree,
    _edge_quadrature,
    _legendre_gauss_lobatto,
    _quadrature_point_count,
    _symmetric_triangle_quadrature,
    _triangle_quadrature,
    triangle_edge_points,
)

# Former names of the triangle edge map, kept for callers in other modules.
_edge_points = triangle_edge_points
_reference_edge_points_from_1d = triangle_edge_points


_VOLUME_QUADRATURE_ALIASES = {
    "auto": "auto",
    "default": "auto",
    "symmetric": "symmetric",
    "dunavant": "symmetric",
    "duffy": "duffy",
    "legacy": "duffy",
    "collapsed": "duffy",
}

def _normalize_volume_quadrature(name: str) -> str:
    """Normalize a volume quadrature alias to its canonical name."""
    try:
        return _VOLUME_QUADRATURE_ALIASES[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError("volume_quadrature must be 'auto', 'symmetric', or 'duffy'") from exc


def _automatic_triangle_quadrature(order: int, exact_degree: int | None = None) -> tuple[str, np.ndarray, np.ndarray]:
    """Select the default volume rule for one polynomial order.

    The automatic policy prefers compact admissible Dunavant data.  Once that
    table is exhausted, it returns the legacy collapsed tensor-product rule
    rather than the generated symmetric rule, whose point count is larger than
    Duffy at high order.  Callers can still force the generated rule with
    ``volume_quadrature="symmetric"``.
    """
    if exact_degree is not None:
        # Requested exactness: compact Dunavant when tabulated, else minimal Duffy.
        if _compact_dunavant_degree(max(1, int(exact_degree))) is not None:
            points, weights = _symmetric_triangle_quadrature(order, exact_degree)
            return "symmetric", points, weights
        points, weights = _triangle_quadrature(order, _duffy_points_for_degree(exact_degree))
        return "duffy", points, weights
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
    "orthonormal": "orthonormal",
    "lagrange_equispaced": "lagrange_equispaced",
    "lagrange_warp_blend": "lagrange_warp_blend",
    "lagrange_recursive": "lagrange_recursive",
    "lagrange_fekete": "lagrange_fekete",
    "lagrange_fekete_free": "lagrange_fekete_free",
}

# Bases that the 2D DG spaces, transfers and compiled kernels support. The other
# names of ``_BASIS_ALIASES`` exist for reference-element data only
# (``ReferenceElementData``, its tables and the reference-simplex studies).
LEGACY_BASIS_TYPES = ("bernstein", "hier_C0", "dub_orth")


def _normalize_basis_type(name: str) -> str:
    """Return the canonical basis-family name used internally.

    User-facing constructors accept a small set of aliases such as ``"bern"``
    and ``"dubiner"``.  This helper converts those aliases to the exact names
    dispatched by :func:`_evaluate_basis` and :func:`_evaluate_gradients`, and
    raises ``ValueError`` listing the accepted names for any other string.
    """
    try:
        return _BASIS_ALIASES[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported basis_type {name!r}; supported bases are "
            + ", ".join(repr(value) for value in dict.fromkeys(_BASIS_ALIASES.values()))
        ) from exc


def _evaluate_simplex_basis(basis_type: str, dim: int, order: int, points: np.ndarray, *, gradients: bool):
    """Evaluate one basis of :mod:`hybridge.core.simplex` on the reference simplex of dimension ``dim``.

    ``basis_type`` is a canonical name of ``_BASIS_ALIASES``: a name starting with
    ``lagrange_`` selects the nodal Lagrange basis of that node family
    (:mod:`hybridge.core.simplex.nodes`), any other name a modal basis
    (:mod:`hybridge.core.simplex.polynomials`). ``points`` has shape
    ``(num_points, dim)``. Returns values of shape ``(num_points, n)`` or, with
    ``gradients=True``, reference gradients of shape ``(num_points, n, dim)``,
    as ``REAL_DTYPE``.
    """
    from hybridge.core.simplex import nodes, polynomials

    if basis_type.startswith("lagrange_"):
        family = basis_type[len("lagrange_"):]
        evaluate = nodes.lagrange_gradients if gradients else nodes.lagrange_basis
        values = evaluate(family, int(dim), int(order), points)
    else:
        evaluate = polynomials.modal_gradients if gradients else polynomials.modal_basis
        values = evaluate(basis_type, int(dim), int(order), points)
    return np.ascontiguousarray(values, dtype=REAL_DTYPE)


def _edge_basis(order: int, edge_points_1d: np.ndarray) -> np.ndarray:
    """Evaluate the default nodal Lagrange trace basis on reference-edge points.

    The trace degrees of freedom are located at ``order + 1``
    Legendre-Gauss-Lobatto nodes on ``[-1, 1]``.  This keeps package solvers on
    the same nodal trace convention as the standalone GPU runners'
    ``legacy-lagrange`` trace basis.
    """
    nodes, _ = _legendre_gauss_lobatto(order + 1)
    return _lagrange_basis(nodes, edge_points_1d)


def _point_dimension(points) -> int:
    """Return the number of coordinates of a point array of shape ``(num_points, dim)``."""
    shape = np.shape(points)
    return int(shape[-1]) if len(shape) == 2 else -1


def _evaluate_basis(basis_type: str, order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the selected basis at reference points.

    ``points`` has shape ``(num_points, dim)`` in reference coordinates; ``dim``
    is 2 for the reference triangle and 3 for the reference tetrahedron. The
    three legacy bases on the triangle use the compiled kernels of
    :mod:`hybridge.core.basis`; every other basis or dimension is evaluated by
    :mod:`hybridge.core.simplex`. The result has shape ``(num_points, el_dof)``.
    """
    if _point_dimension(points) in (-1, 2) and basis_type in LEGACY_BASIS_TYPES:
        if basis_type == "bernstein":
            return basis_module.evaluate_bernstein_basis(order, points)
        if basis_type == "hier_C0":
            return basis_module.evaluate_hierarchical_c0_basis(order, points)
        return basis_module.evaluate_dubiner_basis(order, points)
    if basis_type not in _BASIS_ALIASES.values():
        raise ValueError(f"unsupported basis_type {basis_type!r}")
    return _evaluate_simplex_basis(basis_type, _point_dimension(points), order, points, gradients=False)


def _evaluate_gradients(basis_type: str, order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate reference-coordinate gradients of the selected basis.

    The returned array has shape ``(num_points, el_dof, dim)``. The last axis
    holds the derivatives with respect to the reference coordinates; physical
    gradients are obtained later by multiplying with the inverse-transpose
    element map. Dispatch is as in :func:`_evaluate_basis`.
    """
    if _point_dimension(points) in (-1, 2) and basis_type in LEGACY_BASIS_TYPES:
        if basis_type == "bernstein":
            return basis_module.evaluate_bernstein_gradients(order, points)
        if basis_type == "hier_C0":
            return basis_module.evaluate_hierarchical_c0_gradients(order, points)
        return basis_module.evaluate_dubiner_gradients(order, points)
    if basis_type not in _BASIS_ALIASES.values():
        raise ValueError(f"unsupported basis_type {basis_type!r}")
    return _evaluate_simplex_basis(basis_type, _point_dimension(points), order, points, gradients=True)


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
    volume_degree: int | None = None
    dim: int = 2
    face_degree: int | None = None
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
        """Check the options and tabulate the reference-element data.

        ``dim = 2`` (the triangle) runs the historical construction unchanged.
        ``dim = 3`` (the tetrahedron) tabulates the volume rule, the basis and
        the mass matrix, and leaves the triangle-only fields as ``None``.
        """
        order = int(self.order)
        if order < 0:
            raise ValueError("order must be nonnegative")
        dim = int(self.dim)
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 (triangle) or 3 (tetrahedron); got {self.dim!r}")
        object.__setattr__(self, "dim", dim)
        if self.face_degree is not None:
            if int(self.face_degree) < 0:
                raise ValueError("face_degree must be nonnegative")
            object.__setattr__(self, "face_degree", int(self.face_degree))
        basis_type = _normalize_basis_type(self.basis_type)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "basis_type", basis_type)
        if dim == 3:
            self._tabulate_tetrahedron()
            return
        volume_quadrature = _normalize_volume_quadrature(self.volume_quadrature)
        volume_degree = self.volume_degree
        if volume_degree is not None:
            volume_degree = int(volume_degree)
            if volume_degree < 0:
                raise ValueError("volume_degree must be nonnegative")
            if self.volume_quad_1d is not None:
                raise ValueError("choose either volume_degree or volume_quad_1d, not both")
            object.__setattr__(self, "volume_degree", volume_degree)
        if self.volume_quad_1d is not None:
            volume_quadrature = "duffy"
        object.__setattr__(self, "el_dof", (order + 1) * (order + 2) // 2)
        object.__setattr__(self, "edg_dof", order + 1)

        if volume_quadrature == "auto":
            volume_quadrature, q_points, q_weights = _automatic_triangle_quadrature(order, volume_degree)
        elif volume_quadrature == "symmetric":
            q_points, q_weights = _symmetric_triangle_quadrature(order, volume_degree)
        else:
            count = self.volume_quad_1d if volume_degree is None else _duffy_points_for_degree(volume_degree)
            q_points, q_weights = _triangle_quadrature(order, count)
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
            np.array(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)), dtype=REAL_DTYPE),
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

    def _tabulate_tetrahedron(self) -> None:
        """Tabulate the volume data of the reference tetrahedron (``dim = 3``).

        The volume rule is the collapsed Gauss-Jacobi rule of
        :func:`hybridge.core.simplex.reference.gauss_jacobi_collapsed_rule`, exact
        for total degree ``volume_degree`` (default ``2 * order``). Sets ``el_dof``
        (``C(p+3, 3)``), ``edg_dof`` (the trace dimension per face, ``C(p+2, 2)``),
        ``Krf_quads``/``Krf_w``, the basis values ``phi``/``bas_of_quads``, the
        gradients ``gphi``/``dbas_of_quads``, ``weighted_phi`` and the mass matrix
        ``MKrf`` with its inverse. Every other field, which describes triangle
        edges, is set to ``None``. Raises ``ValueError`` for options that only
        exist on the triangle (``volume_quad_1d``, ``edge_quad_1d``, the
        ``"symmetric"`` volume rule).
        """
        from hybridge.core.simplex.reference import gauss_jacobi_collapsed_rule

        order = self.order
        if self.volume_quad_1d is not None or self.edge_quad_1d is not None:
            raise ValueError(
                "volume_quad_1d and edge_quad_1d exist only on the triangle; "
                "use volume_degree and face_degree on the tetrahedron"
            )
        volume_quadrature = _normalize_volume_quadrature(self.volume_quadrature)
        if volume_quadrature == "symmetric":
            raise ValueError(
                "no symmetric volume rule exists on the tetrahedron yet; use "
                "volume_quadrature='auto' or 'duffy' (collapsed Gauss-Jacobi)"
            )
        if self.volume_degree is not None and int(self.volume_degree) < 0:
            raise ValueError("volume_degree must be nonnegative")
        degree = 2 * order if self.volume_degree is None else int(self.volume_degree)
        for item in dataclasses.fields(self):
            if not item.init:
                object.__setattr__(self, item.name, None)
        object.__setattr__(self, "volume_quadrature", "duffy")
        if self.volume_degree is not None:
            object.__setattr__(self, "volume_degree", int(self.volume_degree))
        object.__setattr__(self, "el_dof", comb(order + 3, 3))
        object.__setattr__(self, "edg_dof", comb(order + 2, 2))
        points, weights = gauss_jacobi_collapsed_rule(3, max(degree, 1))
        points = np.ascontiguousarray(points, dtype=REAL_DTYPE)
        weights = np.ascontiguousarray(weights, dtype=REAL_DTYPE)
        basis = _evaluate_basis(self.basis_type, order, points)
        gradients = _evaluate_gradients(self.basis_type, order, points)
        mass = np.einsum("q,qi,qj->ij", weights.astype(np.float64), basis.astype(np.float64),
                         basis.astype(np.float64), optimize=True)
        object.__setattr__(self, "Krf_quads", points)
        object.__setattr__(self, "Krf_w", weights)
        object.__setattr__(self, "phi", np.ascontiguousarray(basis))
        object.__setattr__(self, "bas_of_quads", np.ascontiguousarray(basis.T))
        object.__setattr__(self, "gphi", np.ascontiguousarray(gradients))
        object.__setattr__(self, "dbas_of_quads", np.ascontiguousarray(gradients.transpose(2, 1, 0)))
        object.__setattr__(self, "weighted_phi", np.ascontiguousarray(weights[:, None] * basis))
        object.__setattr__(self, "MKrf", np.ascontiguousarray(mass, dtype=REAL_DTYPE))
        object.__setattr__(self, "MKrf_inv", np.ascontiguousarray(np.linalg.inv(mass), dtype=REAL_DTYPE))

    # ------------------------------------------------------------------
    # Dimension-generic reference integrals (triangle and tetrahedron).
    #
    # These are computed on first access with their own rules, exact for the
    # integrand degree, so they do not depend on the volume rule the caller
    # chose for ``Krf_quads``. They are computed in float64 and stored as
    # ``REAL_DTYPE``. Face integrals are taken over the reference face F^ (the
    # reference simplex of one dimension lower); the integral over the face
    # F_f of the reference element is the table times ``face_measure_ratios[f]``.
    # ------------------------------------------------------------------

    @cached_property
    def simplex(self):
        """Geometry of the reference simplex (:class:`hybridge.core.simplex.reference.ReferenceSimplex`)."""
        from hybridge.core.simplex.reference import reference_simplex

        return reference_simplex(self.dim)

    @property
    def num_faces(self) -> int:
        """Number of faces of the reference element, ``dim + 1``."""
        return self.dim + 1

    @property
    def face_dof(self) -> int:
        """Dimension of the trace space on one face, ``C(order + dim - 1, dim - 1)``."""
        return comb(self.order + self.dim - 1, self.dim - 1)

    @property
    def num_face_orientations(self) -> int:
        """Number of vertex permutations of one face, ``dim!`` (2 on the triangle, 6 on the tetrahedron)."""
        return factorial(self.dim)

    @property
    def face_vertices(self) -> tuple[tuple[int, ...], ...]:
        """Local vertex numbers of every face, in the order that defines its face map."""
        return self.simplex.face_vertices

    @property
    def face_measure_ratios(self) -> np.ndarray:
        """Ratios ``|F_f| / |F^|`` of the face measures to the reference-face measure, shape ``(dim + 1,)``."""
        return self.simplex.face_measure_ratios

    @property
    def reference_normals(self) -> np.ndarray:
        """Outward unit normals of the faces of the reference element, shape ``(dim + 1, dim)``."""
        return self.simplex.outward_normals

    @cached_property
    def _generic_volume_rule(self) -> tuple[np.ndarray, np.ndarray]:
        """Volume rule exact for total degree ``2 * order``, in float64 (points ``(nq, dim)``, weights ``(nq,)``)."""
        from hybridge.core.simplex.reference import simplex_volume_rule

        points, weights = simplex_volume_rule(self.dim, max(2 * self.order, 1))
        return np.asarray(points, dtype=np.float64), np.asarray(weights, dtype=np.float64)

    @cached_property
    def _generic_face_rule(self) -> tuple[np.ndarray, np.ndarray]:
        """Reference-face rule exact for degree ``face_degree`` (default ``2 * order``), in float64."""
        from hybridge.core.simplex.reference import simplex_face_rule

        degree = 2 * self.order if self.face_degree is None else self.face_degree
        points, weights = simplex_face_rule(self.dim, max(degree, 1))
        return np.asarray(points, dtype=np.float64).reshape(len(weights), self.dim - 1), np.asarray(weights, dtype=np.float64)

    @property
    def face_quad_points(self) -> np.ndarray:
        """Quadrature points on the reference face, shape ``(nfq, dim - 1)``."""
        return np.ascontiguousarray(self._generic_face_rule[0], dtype=REAL_DTYPE)

    @property
    def face_quad_weights(self) -> np.ndarray:
        """Quadrature weights on the reference face, shape ``(nfq,)``; they sum to ``|F^|``."""
        return np.ascontiguousarray(self._generic_face_rule[1], dtype=REAL_DTYPE)

    @cached_property
    def face_points(self) -> np.ndarray:
        """Face quadrature points mapped onto every face of the reference element, shape ``(dim + 1, nfq, dim)``."""
        points = self._generic_face_rule[0]
        return np.ascontiguousarray(
            np.stack([self.simplex.face_map(f, points) for f in range(self.num_faces)]), dtype=REAL_DTYPE
        )

    @cached_property
    def face_quadrature_permutation(self) -> np.ndarray:
        """Index permutation of the face quadrature points for every face orientation, shape ``(dim!, nfq)``.

        Row ``o`` is the permutation returned by
        :func:`hybridge.core.simplex.reference.face_quadrature_permutation` for the
        orientation ``face_permutations(dim)[o]``; the face rules are invariant
        under every vertex permutation, so each row is a permutation of ``range(nfq)``.
        """
        from hybridge.core.simplex.reference import face_permutations, face_quadrature_permutation

        points = self._generic_face_rule[0]
        return np.ascontiguousarray(
            np.stack([face_quadrature_permutation(self.dim, points, sigma) for sigma in face_permutations(self.dim)]),
            dtype=np.int64,
        )

    @cached_property
    def _generic_volume_tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weights, basis values ``(nq, n_u)`` and gradients ``(nq, n_u, dim)`` on the exact volume rule, float64."""
        points, weights = self._generic_volume_rule
        basis = np.asarray(_evaluate_basis(self.basis_type, self.order, points), dtype=np.float64)
        gradients = np.asarray(_evaluate_gradients(self.basis_type, self.order, points), dtype=np.float64)
        return weights, basis, gradients

    @cached_property
    def element_basis_on_faces(self) -> np.ndarray:
        """Volume basis values at the face quadrature points of every face, shape ``(dim + 1, n_u, nfq)``."""
        points = self._generic_face_rule[0]
        tables = [
            np.asarray(_evaluate_basis(self.basis_type, self.order, self.simplex.face_map(f, points)), dtype=np.float64).T
            for f in range(self.num_faces)
        ]
        return np.ascontiguousarray(np.stack(tables), dtype=REAL_DTYPE)

    @cached_property
    def exact_mass(self) -> np.ndarray:
        """Mass matrix ``M_ij = int_K^ phi_i phi_j`` on a rule exact for degree ``2p``, shape ``(n_u, n_u)``."""
        weights, basis, _ = self._generic_volume_tables
        return np.ascontiguousarray(np.einsum("q,qi,qj->ij", weights, basis, basis, optimize=True), dtype=REAL_DTYPE)

    @cached_property
    def face_mass(self) -> np.ndarray:
        """Face mass matrices ``E^f_ij = int_F^ (phi_i phi_j) o F_f``, shape ``(dim + 1, n_u, n_u)``."""
        _, weights = self._generic_face_rule
        values = self.element_basis_on_faces.astype(np.float64)
        return np.ascontiguousarray(np.einsum("q,fiq,fjq->fij", weights, values, values, optimize=True), dtype=REAL_DTYPE)

    @cached_property
    def derivative_matrices(self) -> np.ndarray:
        """Derivative matrices ``D^k_ij = int_K^ phi_i d(phi_j)/d(xi_k)``, shape ``(dim, n_u, n_u)``.

        Note the convention: the derivative falls on the second (trial) index.
        :func:`hybridge.hdg.reference._reference_derivative_matrices` stores the
        transposes, computed on the caller's volume rule; these tables use a rule
        exact for degree ``2p``.
        """
        weights, basis, gradients = self._generic_volume_tables
        return np.ascontiguousarray(
            np.einsum("q,qi,qjk->kij", weights, basis, gradients, optimize=True), dtype=REAL_DTYPE
        )

    @cached_property
    def stiffness_components(self) -> np.ndarray:
        """Stiffness components ``K^{kl}_ij = int_K^ d(phi_i)/d(xi_k) d(phi_j)/d(xi_l)``, shape ``(dim, dim, n_u, n_u)``."""
        weights, _, gradients = self._generic_volume_tables
        return np.ascontiguousarray(
            np.einsum("q,qik,qjl->klij", weights, gradients, gradients, optimize=True), dtype=REAL_DTYPE
        )

    def trace_tables(self, trace_basis: str = "lagrange_warp_blend") -> "ReferenceTraceTables":
        """Return the element-trace tables of this volume basis with one trace basis.

        ``trace_basis`` names a basis of the reference simplex of dimension
        ``dim - 1`` (same names as ``basis_type``). On the triangle,
        ``"lagrange_warp_blend"`` (or any ``lagrange_*`` family) is the
        Gauss-Lobatto Lagrange basis of the ``legacy-lagrange`` trace,
        ``"dub_orth"`` the Legendre basis of ``legendre-modal`` and
        ``"bernstein"`` the Bernstein trace basis. One table set is built per
        trace basis and kept on this instance. See :class:`ReferenceTraceTables`.
        """
        name = _normalize_basis_type(trace_basis)
        cache = self.__dict__.setdefault("_trace_tables_cache", {})
        if name not in cache:
            cache[name] = _build_reference_trace_tables(self, name)
        return cache[name]

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
            volume_degree: int | None = None,
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
            volume_degree=volume_degree,
        )

    @classmethod
    def tetrahedron(
            cls,
            polynomial_order: int,
            *,
            basis_type: str = "orthonormal",
            volume_quadrature: str = "auto",
            volume_degree: int | None = None,
            face_degree: int | None = None,
    ) -> "ReferenceElementData":
        """Build reference data for the reference tetrahedron.

        The tetrahedron is ``conv{(-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)}``.
        ``basis_type`` is any name accepted by ``ReferenceElementData``;
        ``volume_degree`` and ``face_degree`` set the exactness of the volume and
        face rules (default ``2 * polynomial_order``). Only reference-element data
        exists in three dimensions; there is no tetrahedral mesh or DG space yet.
        """
        return cls(
            polynomial_order,
            basis_type=basis_type,
            volume_quadrature=volume_quadrature,
            volume_degree=volume_degree,
            face_degree=face_degree,
            dim=3,
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
        points = np.asarray(reference_points, dtype=REAL_DTYPE)
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
            Array with shape ``(num_points, el_dof, dim)``.
        """
        points = np.asarray(reference_points, dtype=REAL_DTYPE)
        if points is self.Krf_quads:
            return self.gphi
        return _evaluate_gradients(self.basis_type, self.order, points)

    evaluate_basis = basis_at
    evaluate_gradients = gradients_at


TriangleQuadratureData = ReferenceElementData


@dataclass(frozen=True)
class ReferenceTraceTables:
    """Element-trace integrals of one volume basis and one trace basis on the reference element.

    Let ``phi_i`` (``i < n_u``) be the volume basis, ``mu_a`` (``a < n_t``) the
    trace basis on the reference face ``F^``, ``F_f`` the map of ``F^`` onto face
    ``f`` and ``sigma_o`` the face orientations of
    :func:`hybridge.core.simplex.reference.face_permutations` (``dim!`` of them,
    identity first). The trace basis is defined in the global vertex order of a
    face; an element sees it through ``sigma_o``, so its trace values at the face
    point ``s`` are ``mu_a(sigma_o s)``. All integrals are over ``F^``; multiply
    by ``face_measure_ratios[f]`` for the integral over face ``f`` itself.

    ``trace_values``
        ``mu_a`` at the reference-face quadrature points, shape ``(n_t, nfq)``.
    ``element_trace_coupling[o, f, i, a]``
        ``int_F^ phi_i(F_f s) mu_a(sigma_o s) ds`` (the u-uhat block),
        shape ``(dim!, dim + 1, n_u, n_t)``.
    ``trace_element_coupling[o * (dim + 1) + f, a, i]``
        The same integrals with the trace function as test function (the
        uhat-u block), shape ``(dim! (dim + 1), n_t, n_u)``. On the triangle this
        is the layout of ``face_trace_test_element_trial_oriented``.
    ``trace_mass[a, b]``
        ``int_F^ mu_a mu_b ds`` (the uhat-uhat block), shape ``(n_t, n_t)``.
    ``orientation_matrices[o]``
        The matrix ``R`` with ``mu_a(sigma_o s) = sum_b R[a, b] mu_b(s)``, shape
        ``(dim!, n_t, n_t)``. It is a permutation matrix for a nodal trace basis on
        a symmetric node set and an orthogonal matrix for an orthonormal one.
    """

    trace_basis: str
    dim: int
    order: int
    trace_values: np.ndarray
    element_trace_coupling: np.ndarray
    trace_element_coupling: np.ndarray
    trace_mass: np.ndarray
    orientation_matrices: np.ndarray

    @property
    def face_dof(self) -> int:
        """Dimension ``n_t`` of the trace space on one face."""
        return int(self.trace_mass.shape[0])


def _build_reference_trace_tables(reference: ReferenceElementData, trace_basis: str) -> ReferenceTraceTables:
    """Compute the :class:`ReferenceTraceTables` of ``reference`` with the trace basis ``trace_basis``.

    The face integrals use the reference-face rule of ``reference`` (exact for
    degree ``face_degree``, default ``2p``), evaluated in float64. The
    orientation matrices solve ``R G = int_F^ mu(sigma s) mu(s)^T`` with the trace
    mass ``G``, which is exact because a vertex permutation maps the trace space
    onto itself.
    """
    from hybridge.core.simplex.reference import face_permutations, permute_face_points

    dim = reference.dim
    order = reference.order
    points, weights = reference._generic_face_rule
    face_dim = dim - 1

    def trace_at(face_points: np.ndarray) -> np.ndarray:
        """Trace basis values at reference-face points ``(n, dim - 1)``, shape ``(n_t, n)``."""
        values = _evaluate_simplex_basis(trace_basis, face_dim, order, face_points, gradients=False)
        return np.asarray(values, dtype=np.float64).T

    mu = trace_at(points)
    mass = np.einsum("q,aq,bq->ab", weights, mu, mu, optimize=True)
    phi_faces = reference.element_basis_on_faces.astype(np.float64)
    couplings = []
    rotations = []
    for sigma in face_permutations(dim):
        mu_sigma = trace_at(permute_face_points(sigma, points))
        couplings.append(np.einsum("q,fiq,aq->fia", weights, phi_faces, mu_sigma, optimize=True))
        cross = np.einsum("q,aq,bq->ab", weights, mu_sigma, mu, optimize=True)
        rotations.append(np.linalg.solve(mass.T, cross.T).T)
    element_trace = np.stack(couplings)
    trace_element = element_trace.transpose(0, 1, 3, 2).reshape(
        len(couplings) * (dim + 1), mu.shape[0], phi_faces.shape[1]
    )

    def stored(array: np.ndarray) -> np.ndarray:
        """Return ``array`` as a contiguous ``REAL_DTYPE`` array."""
        return np.ascontiguousarray(array, dtype=REAL_DTYPE)

    return ReferenceTraceTables(
        trace_basis=trace_basis,
        dim=dim,
        order=order,
        trace_values=stored(mu),
        element_trace_coupling=stored(element_trace),
        trace_element_coupling=stored(trace_element),
        trace_mass=stored(mass),
        orientation_matrices=stored(np.stack(rotations)),
    )


def as_reference_element(reference: ReferenceElementData) -> ReferenceElementData:
    """Validate and return a :class:`ReferenceElementData` instance.

    This small adapter keeps call sites explicit when they accept "reference
    data" conceptually but currently support only the concrete
    :class:`ReferenceElementData` implementation.
    """
    if isinstance(reference, ReferenceElementData):
        return reference
    raise TypeError(f"expected ReferenceElementData, got {type(reference)!r}")


def reference_plot_points(resolution: int) -> np.ndarray:
    r"""Return a uniform plotting grid on :math:`\hat K`.

    The reference triangle is
    :math:`\hat K=\{(r,s)\in[-1,1]^2:\ s\le -r\}`.  The returned array has
    shape ``(num_plot_points, 2)`` and is C-contiguous.
    """
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    axis = np.linspace(-1.0, 1.0, int(resolution))
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    inside = yy <= -xx
    return np.ascontiguousarray(np.column_stack((xx[inside], yy[inside])), dtype=np.float64)
