r"""Self-contained reference-triangle data for :mod:`dgfem`.

The reference element is

.. math::

    \hat K = \operatorname{conv}\{(-1,-1), (1,-1), (-1,1)\}.

The default basis is the Bernstein basis of total degree ``p`` on this
triangle. Bernstein functions form a partition of unity, are easy to evaluate
and differentiate in vectorized NumPy, and provide a stable first internal
basis for the new package. The public attribute names intentionally mirror the
legacy quadrature object where they are useful for assembly kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial

import numpy as np

from . import basis as basis_module


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
    try:
        return _BASIS_ALIASES[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported basis_type {name!r}; supported bases are "
            "'bernstein', 'hier_C0', and 'dub_orth'"
        ) from exc


def _triangle_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss rule collapsed onto the reference triangle."""
    num_1d = max(2 * order + 2, 2)
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


def _edge_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    num_1d = max(2 * order + 2, 2)
    points, weights = np.polynomial.legendre.leggauss(num_1d)
    return np.ascontiguousarray(points), np.ascontiguousarray(weights)


def _edge_points(edge_points_1d: np.ndarray) -> np.ndarray:
    """Reference face points with shape ``(num_face_quads, 3, 2)``."""
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


def _edge_basis(order: int, edge_points_1d: np.ndarray) -> np.ndarray:
    """One-dimensional Bernstein edge basis with shape ``(edg_dof, nq)``."""
    r = 0.5 * (edge_points_1d + 1.0)
    values = np.empty((order + 1, edge_points_1d.size), dtype=np.float64)
    for j in range(order + 1):
        coeff = factorial(order) / (factorial(j) * factorial(order - j))
        values[j] = coeff * (1.0 - r) ** (order - j) * r ** j
    return np.ascontiguousarray(values)


def _evaluate_basis(basis_type: str, order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the selected basis at reference points."""
    if basis_type == "bernstein":
        return basis_module.evaluate_bernstein_basis(order, points)
    if basis_type == "hier_C0":
        return basis_module.evaluate_hierarchical_c0_basis(order, points)
    if basis_type == "dub_orth":
        return basis_module.evaluate_dubiner_basis(order, points)
    raise ValueError(f"unsupported basis_type {basis_type!r}")


def _evaluate_gradients(basis_type: str, order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate gradients of the selected basis at reference points."""
    if basis_type == "bernstein":
        return basis_module.evaluate_bernstein_gradients(order, points)
    if basis_type == "hier_C0":
        return basis_module.evaluate_hierarchical_c0_gradients(order, points)
    if basis_type == "dub_orth":
        return basis_module.evaluate_dubiner_gradients(order, points)
    raise ValueError(f"unsupported basis_type {basis_type!r}")


@dataclass(frozen=True)
class ReferenceElementData:
    """Reference triangle quadrature, basis values, and reference matrices."""

    order: int
    basis_type: str = "bernstein"
    verbosity: int = 0
    cache: bool = True
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
        order = int(self.order)
        if order < 0:
            raise ValueError("order must be nonnegative")
        basis_type = _normalize_basis_type(self.basis_type)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "basis_type", basis_type)
        object.__setattr__(self, "el_dof", (order + 1) * (order + 2) // 2)
        object.__setattr__(self, "edg_dof", order + 1)

        q_points, q_weights = _triangle_quadrature(order)
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

        edge_points, edge_weights = _edge_quadrature(order)
        face_points = _edge_points(edge_points)
        face_basis = _evaluate_basis(basis_type, order, face_points.reshape(-1, 2))
        face_basis = face_basis.reshape(edge_points.size, 3, self.el_dof).transpose(1, 2, 0)
        negative_face_points = _edge_points(-edge_points)
        negative_face_basis = _evaluate_basis(basis_type, order, negative_face_points.reshape(-1, 2))
        negative_face_basis = negative_face_basis.reshape(edge_points.size, 3, self.el_dof).transpose(1, 2, 0)
        object.__setattr__(self, "quads_JGL", edge_points)
        object.__setattr__(self, "weights_JGL", edge_weights)
        object.__setattr__(self, "rf_edg_lag_nodes", edge_points)
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

        mkrfe_p = np.einsum(
            "q,fiq,jq->fij",
            edge_weights,
            face_basis,
            edge_basis,
            optimize=True,
        )
        mkrfe_n = np.einsum(
            "q,fiq,jq->fij",
            edge_weights,
            negative_face_basis,
            edge_basis,
            optimize=True,
        )
        object.__setattr__(self, "MKrfe_lst_p", np.ascontiguousarray(mkrfe_p))
        object.__setattr__(self, "MKrfe_lst_n", np.ascontiguousarray(mkrfe_n))
        object.__setattr__(
            self,
            "MKrfe_lst",
            np.ascontiguousarray(
                np.concatenate(
                    (mkrfe_p.transpose(0, 2, 1), mkrfe_n.transpose(0, 2, 1)),
                    axis=0,
                )
            ),
        )
        edge_mass = np.einsum(
            "q,iq,jq->ij",
            edge_weights,
            edge_basis,
            edge_basis,
            optimize=True,
        )
        object.__setattr__(self, "M_rf_fc", np.ascontiguousarray(edge_mass))
        object.__setattr__(self, "M_rf_fc_f", np.ascontiguousarray(edge_mass.ravel()))
        mbde = np.einsum("q,fiq,fjq->fij", edge_weights, face_basis, face_basis, optimize=True)
        object.__setattr__(self, "MbdeKrf_lst", np.ascontiguousarray(mbde))

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
    ) -> "ReferenceElementData":
        """Build reference data for a triangular DG space."""
        return cls(polynomial_order, basis_type=basis_type, verbosity=verbosity, cache=cache)

    @property
    def quadrature(self) -> "ReferenceElementData":
        """Compatibility property: reference data is its own quadrature object."""
        return self

    def basis_at(self, reference_points: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at reference points.

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
    """Normalize reference data to :class:`ReferenceElementData`."""
    if isinstance(reference, ReferenceElementData):
        return reference
    raise TypeError(f"expected ReferenceElementData, got {type(reference)!r}")
