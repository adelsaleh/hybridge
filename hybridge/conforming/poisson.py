r"""Statically condensed continuous Galerkin Poisson operator (layer 4).

Find :math:`\phi_h \in S_h^k` with :math:`\phi_h = g` at the boundary nodes and

.. math::

    (\nabla\phi_h, \nabla\psi) = (\rho_h, \psi) \qquad \forall \psi \in S_{h,0}^k,

where :math:`S_h^k` is the continuous Lagrange space of
:class:`hybridge.core.conforming.ContinuousLagrangeSpace` and :math:`\rho_h` a DG field on the same
mesh. On an affine triangle :math:`x = A\xi + b` the element matrix is

.. math::

    K_e = |\det A| \sum_{a,b} M_{ab} S^{ab}, \qquad M = A^{-1}A^{-T},\quad
    S^{ab}_{ij} = \int_{\hat K} \partial_a\hat\varphi_i\,\partial_b\hat\varphi_j ,

with reference tables computed once. The element-interior (bubble) unknowns are eliminated element
by element (Numba, parallel over elements), so the global system couples only the vertex and edge
unknowns (the skeleton). The
operator is constant: element matrices, condensation and the global matrix are built once in
:class:`CondensedCGPoissonOperator`; every solve costs the load ``(rho_h, phi_i)``, the condensed
right-hand side, one global solve and the bubble back-substitution. Per-solve work runs on the host
(NumPy/SciPy) or on the device (CuPy) with the arrays' residency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse

from hybridge.conforming.numba_kernels import condense_cg_elements
from hybridge.core.conforming import ContinuousField, ContinuousLagrangeSpace
from hybridge.core.quadrature import _automatic_triangle_quadrature
from hybridge.core.space import DGField, DGSpace
from hybridge.runtime.precision import REAL_DTYPE

__all__ = ["CondensedCGPoissonOperator", "CGPoissonSystem"]


@dataclass(frozen=True)
class CGPoissonSystem:
    """Free-skeleton system of the condensed operator: ``matrix @ u_free = rhs``."""

    matrix: object  # scipy.sparse.csr_matrix (host) or cupyx csr_matrix (device)
    rhs: object
    num_skeleton: int


class CondensedCGPoissonOperator:
    r"""Element matrices, static condensation and the global skeleton matrix of CG Poisson.

    ``degree`` is the continuous degree ``k`` (``k = p + 1`` for the DG(p) velocity of the
    guiding-center transport); ``source_space`` the DG space of the right-hand sides. The
    quadrature is exact for the stiffness (degree ``2k - 2``) and the load (degree ``k + p``).
    """

    def __init__(self, source_space: DGSpace, degree: int, *, device: bool = False) -> None:
        k = int(degree)
        if k < 1:
            raise ValueError("the CG degree must be at least 1")
        self.source_space = source_space
        self.cspace = ContinuousLagrangeSpace.on(source_space.mesh, k)
        self.degree = k
        mesh = self.cspace.mesh
        num_elements = mesh.num_tri
        n_loc = self.cspace.num_local
        self.num_skeleton_local = 3 * k  # vertices + edge nodes, first in the lattice order
        nb = self.num_skeleton_local
        ni = n_loc - nb

        # Exact for the stiffness (2k - 2) and the load (k + p).
        _, points, weights = _automatic_triangle_quadrature(k, exact_degree=max(2 * k, k + source_space.order))
        values = self.cspace.lagrange_basis_at(points)  # (nq, n_loc)
        gradients = self.cspace.lagrange_gradients_at(points)  # (nq, n_loc, 2)
        reference = np.einsum("q,qia,qjb->abij", weights, gradients, gradients)
        source_basis = source_space.basis_at(points)  # (nq, n_dg)
        self._load = np.ascontiguousarray((source_basis * weights[:, None]).T @ values, dtype=REAL_DTYPE)

        jacobian = np.abs(np.linalg.det(mesh.aff_mats))
        inverse = np.linalg.inv(mesh.aff_mats)
        metric = np.einsum("kad,kbd->kab", inverse, inverse)  # A^{-1} A^{-T}
        self._jacobian = np.ascontiguousarray(jacobian, dtype=REAL_DTYPE)

        cell_dofs = self.cspace.cell_dofs
        self.num_skeleton = int(cell_dofs[:, :nb].max()) + 1 if nb else 0
        if ni and int(cell_dofs[:, nb:].min()) < self.num_skeleton:  # pragma: no cover - numbering invariant
            raise AssertionError("continuous numbering does not place interior nodes after the skeleton")
        self._bubble_inverse = np.zeros((num_elements, ni, ni), dtype=np.float64)
        self._bubble_coupling = np.zeros((num_elements, ni, nb), dtype=np.float64)  # K_ii^{-1} K_ib
        schur = np.empty((num_elements, nb, nb), dtype=np.float64)
        status = np.zeros(num_elements, dtype=np.int64)
        condense_cg_elements(np.ascontiguousarray(metric, dtype=np.float64),
                             np.ascontiguousarray(jacobian, dtype=np.float64),
                             np.ascontiguousarray(reference, dtype=np.float64), nb, schur,
                             self._bubble_inverse, self._bubble_coupling, status)
        failed = np.flatnonzero(status)
        if failed.size:
            raise np.linalg.LinAlgError(f"interior CG block not SPD on {failed.size} elements "
                                        f"(first element {int(failed[0])}, status {int(status[failed[0]])})")
        skeleton_dofs = np.ascontiguousarray(cell_dofs[:, :nb])
        rows = np.repeat(skeleton_dofs, nb, axis=1).ravel()
        cols = np.tile(skeleton_dofs, (1, nb)).ravel()
        full = scipy.sparse.coo_matrix((schur.ravel(), (rows, cols)),
                                       shape=(self.num_skeleton, self.num_skeleton)).tocsr()
        full.sum_duplicates()
        boundary = self.cspace.boundary_dofs
        is_free = np.ones(self.num_skeleton, dtype=bool)
        is_free[boundary] = False
        self.free = np.flatnonzero(is_free)
        self.boundary = np.asarray(boundary, dtype=np.int64)
        self.matrix_free = full[self.free][:, self.free].tocsr()
        self.matrix_boundary = full[self.free][:, self.boundary].tocsr()
        # Scatter matrix: skeleton rhs = P @ (element condensed rhs).ravel().
        self._scatter = scipy.sparse.csr_matrix(
            (np.ones(skeleton_dofs.size, dtype=REAL_DTYPE), (skeleton_dofs.ravel(), np.arange(skeleton_dofs.size))),
            shape=(self.num_skeleton, skeleton_dofs.size))
        self._skeleton_dofs = skeleton_dofs
        self._interior_dofs = np.ascontiguousarray(cell_dofs[:, nb:])
        self.device = bool(device)
        self._device_tables = None
        if self.device:
            self._upload()

    # ----------------------------------------------------------------- device
    def _upload(self) -> None:
        import cupyx.scipy.sparse as csparse

        from hybridge.runtime.optional import require_cupy

        cp = require_cupy()
        self._device_tables = {
            "load": cp.asarray(self._load), "jacobian": cp.asarray(self._jacobian),
            "bubble_inverse": cp.asarray(self._bubble_inverse),
            "bubble_coupling": cp.asarray(self._bubble_coupling),
            "scatter": csparse.csr_matrix(self._scatter), "free": cp.asarray(self.free),
            "boundary": cp.asarray(self.boundary),
            "matrix_free": csparse.csr_matrix(self.matrix_free),
            "matrix_boundary": csparse.csr_matrix(self.matrix_boundary),
            "skeleton_dofs": cp.asarray(self._skeleton_dofs), "interior_dofs": cp.asarray(self._interior_dofs),
        }

    def _tables(self, xp):
        if xp is np:
            return {"load": self._load, "jacobian": self._jacobian, "bubble_inverse": self._bubble_inverse,
                    "bubble_coupling": self._bubble_coupling, "scatter": self._scatter, "free": self.free,
                    "boundary": self.boundary, "matrix_free": self.matrix_free,
                    "matrix_boundary": self.matrix_boundary, "skeleton_dofs": self._skeleton_dofs,
                    "interior_dofs": self._interior_dofs}
        if self._device_tables is None:
            self._upload()
        return self._device_tables

    # ------------------------------------------------------------------ solve
    def boundary_values(self, boundary_condition, xp=np):
        """Nodal Dirichlet data ``g`` at the boundary nodes (zero for ``None``)."""
        if boundary_condition is None:
            return xp.zeros(self.boundary.size, dtype=REAL_DTYPE)
        coords = self.cspace.dof_coords[self.boundary]
        values = np.broadcast_to(np.asarray(boundary_condition(coords[:, 0], coords[:, 1]), dtype=REAL_DTYPE),
                                 (self.boundary.size,))
        return xp.asarray(np.ascontiguousarray(values))

    def condensed_system(self, source: DGField, boundary_values, *, xp=np):
        """Return the free-skeleton system and the element loads for ``source`` and data ``g``."""
        if source.space.mesh is not self.source_space.mesh or source.space.order != self.source_space.order:
            raise ValueError("source must live in the operator's source space")
        tables = self._tables(xp)
        coefficients = self._coefficients(source, xp)
        loads = tables["jacobian"][:, None] * (coefficients @ tables["load"])  # (K, n_loc)
        nb = self.num_skeleton_local
        load_b, load_i = loads[:, :nb], loads[:, nb:]
        if load_i.shape[1]:
            condensed = load_b - xp.einsum("kib,ki->kb", tables["bubble_coupling"], load_i)
        else:
            condensed = load_b
        skeleton = tables["scatter"] @ condensed.ravel()
        rhs = skeleton[tables["free"]] - tables["matrix_boundary"] @ boundary_values
        return CGPoissonSystem(tables["matrix_free"], rhs, self.num_skeleton), loads

    def reconstruct(self, free_values, boundary_values, loads, *, xp=np, name: str = "phi_cg") -> ContinuousField:
        """Back-substitute the bubbles and return the continuous solution."""
        tables = self._tables(xp)
        skeleton = xp.zeros(self.num_skeleton, dtype=REAL_DTYPE)
        skeleton[tables["free"]] = free_values
        skeleton[tables["boundary"]] = boundary_values
        dofs = xp.zeros(self.cspace.num_dofs, dtype=REAL_DTYPE)
        dofs[: self.num_skeleton] = skeleton
        nb = self.num_skeleton_local
        if loads.shape[1] > nb:
            local_b = skeleton[tables["skeleton_dofs"]]
            interior = (xp.einsum("kij,kj->ki", tables["bubble_inverse"], loads[:, nb:])
                        - xp.einsum("kib,kb->ki", tables["bubble_coupling"], local_b))
            dofs[tables["interior_dofs"].ravel()] = interior.ravel()
        return ContinuousField(self.cspace, dofs, name=name)

    @staticmethod
    def _coefficients(field: DGField, xp):
        if xp is np:
            return np.asarray(field.coeffs, dtype=REAL_DTYPE)
        from hybridge.core.device import as_cupy_coefficients, as_cupy_space

        return as_cupy_coefficients(field, as_cupy_space(field.space))

    def nonzeros(self) -> int:
        """Nonzeros of the free-skeleton matrix."""
        return int(self.matrix_free.nnz)
