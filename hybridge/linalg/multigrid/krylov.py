"""Experimental native hp-BSR preconditioning for nonsymmetric trace solves."""
from __future__ import annotations

import numpy as np

from hybridge.runtime.optional import require_cupy
from hybridge.linalg.gpu.legendre_face_bsr import legendre_orthonormal_scales
from hybridge.core.space import _bernstein_edge_basis, _legendre_edge_basis
from hybridge.linalg.multigrid.face_hp import (
    AmgxScalarVcycle,
    FaceBlockPmgPrototype,
    face_pmg_kernel_preset,
)
from hybridge.linalg.multigrid.policy import face_hp_mg_preconditioner_parameters


def _bernstein_to_modal_transform(degree):
    """T maps orthonormal Legendre coefficients to Bernstein coefficients on one edge."""
    q = int(degree) + 1
    points, _ = np.polynomial.legendre.leggauss(q)
    bernstein = _bernstein_edge_basis(degree, points)
    modal = _legendre_edge_basis(degree, points)
    modal = modal * legendre_orthonormal_scales(q)[:, None]
    return np.linalg.solve(bernstein.T, modal.T)


class BernsteinHpSymmetricPartPreconditioner:
    """Apply the native hp cycle to the symmetric part in modal coordinates.

    Intended for GMRES on the original, nonsymmetric ADR operator. The native
    diffusion-oriented smoother must be suitable for the symmetric part;
    successful setup alone does not certify positive definiteness. The existing
    scalar AMGX V-cycle supplies the p=0 coarse correction.
    """

    def __init__(self, matrix, *, degree, policy="standard", tuning=None, kernels=None):
        """Build from host Bernstein face-BSR data, retaining device buffers.

        ``kernels`` names a ``FACE_PMG_KERNEL_PRESETS`` entry; ``None`` reads
        ``HYBRIDGE_HP_MG_KERNELS`` (default ``legacy``).
        """
        cp = require_cupy()
        q = int(degree) + 1
        if getattr(matrix, "blocksize", None) != (q, q):
            raise ValueError("Require natural square Bernstein face blocks")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Require a square trace operator")
        symmetric = ((matrix + matrix.T) * 0.5).tobsr(blocksize=(q, q))
        symmetric.sum_duplicates()
        symmetric.sort_indices()

        transform = _bernstein_to_modal_transform(degree)
        data = np.ascontiguousarray(transform.T @ symmetric.data @ transform)
        faces = matrix.shape[0] // q
        positions = np.empty(faces, dtype=np.int32)
        for row in range(faces):
            start, end = symmetric.indptr[row:row + 2]
            found = np.flatnonzero(symmetric.indices[start:end] == row)
            if found.size != 1:
                raise ValueError("Every face requires a diagonal block")
            positions[row] = start + found[0]
        self._build(cp, cp.asarray(symmetric.indptr, dtype=cp.int32), cp.asarray(symmetric.indices, dtype=cp.int32),
                    cp.asarray(data), cp.asarray(positions), transform, degree=degree, policy=policy,
                    tuning=tuning, kernels=kernels)

    @classmethod
    def from_face_dense(cls, blocks, neighbors, *, degree, policy="standard", tuning=None, kernels=None):
        """Build from device face-dense blocks ``(faces, slots, q, q)`` and ``neighbors``.

        The same hierarchy as the host constructor, with the symmetric part, the row sort,
        the modal transform and the diagonal search on the device. The symmetric blocks
        ``(A_ij + A_ji^T) * 0.5`` equal the host's bitwise; the modal products use cuBLAS.
        """
        cp = require_cupy()
        q = int(degree) + 1
        faces, slots = neighbors.shape
        if blocks.shape != (faces, slots, q, q):
            raise ValueError("Require natural square Bernstein face blocks")
        rows = cp.arange(faces)
        valid = neighbors >= 0
        columns = cp.where(valid, neighbors, 0)
        # Slot of the transposed block: the position of row i in the neighbour list of j.
        partner = (neighbors[columns] == rows[:, None, None]) & valid[:, :, None]
        found = partner.any(axis=2)
        slot = cp.argmax(partner, axis=2)
        transposed = blocks[columns, slot].swapaxes(-1, -2) * found[:, :, None, None]
        symmetric = (blocks + transposed) * 0.5
        order = cp.argsort(cp.where(valid, neighbors, cp.iinfo(cp.int64).max), axis=1)
        sorted_columns = cp.take_along_axis(neighbors, order, axis=1)
        sorted_valid = cp.take_along_axis(valid, order, axis=1)
        counts = sorted_valid.sum(axis=1)
        indptr = cp.concatenate((cp.zeros(1, dtype=cp.int64), cp.cumsum(counts))).astype(cp.int32)
        diagonal = (sorted_columns == rows[:, None]) & sorted_valid
        if not bool(cp.all(diagonal.sum(axis=1) == 1)):
            raise ValueError("Every face requires a diagonal block")
        positions = (indptr[:-1] + cp.argmax(diagonal, axis=1)).astype(cp.int32)
        indices = cp.ascontiguousarray(sorted_columns[sorted_valid].astype(cp.int32))
        sorted_blocks = symmetric[rows[:, None], order][sorted_valid]
        transform = _bernstein_to_modal_transform(degree)
        device_transform = cp.asarray(transform)
        data = cp.ascontiguousarray(cp.matmul(cp.matmul(device_transform.T, sorted_blocks), device_transform))
        self = cls.__new__(cls)
        self._build(cp, indptr, indices, data, positions, transform, degree=degree, policy=policy,
                    tuning=tuning, kernels=kernels)
        return self

    def _build(self, cp, indptr, indices, data, positions, transform, *, degree, policy, tuning, kernels):
        """Build the face hp-multigrid hierarchy and its device workspaces."""
        q = int(degree) + 1
        faces = int(positions.size)
        parameters = face_hp_mg_preconditioner_parameters(policy, overrides=tuning)
        self.parameters = parameters
        self.cp = cp
        self.num_dofs = faces * q
        self.dtype = cp.dtype(data.dtype)
        self.device_id = int(cp.cuda.runtime.getDevice())
        self.block_size = q
        self.transform = cp.asarray(transform)
        self.modal_rhs = cp.empty((faces, q), dtype=self.dtype)
        smoother, transfers, graphs = face_pmg_kernel_preset(kernels)
        self.cycle = FaceBlockPmgPrototype(
            indptr=indptr,
            indices=indices,
            orthonormal_data=data,
            degree=degree,
            diagonal_positions=positions,
            schedule=parameters["schedule"],
            chebyshev_order=parameters["chebyshev_order"],
            presweeps=parameters["presweeps"],
            postsweeps=parameters["postsweeps"],
            spmv_backend="auto" if smoother == "auto" else "legacy",
            smoother_backend=smoother,
            transfer_backend=transfers,
            cuda_graphs=graphs,
            coarse_factory=lambda op: AmgxScalarVcycle(
                op, config=parameters["coarse_config"]
            ),
        )
        self.application_count = 0

    def apply_into(self, rhs, out):
        """Apply T M_modal^-1 T^T in the original outer coordinates."""
        self.cp.matmul(
            rhs.reshape((-1, self.block_size)),
            self.transform,
            out=self.modal_rhs,
        )
        correction = self.cycle.apply(self.modal_rhs.ravel())
        self.cp.matmul(
            correction.reshape((-1, self.block_size)),
            self.transform.T,
            out=out.reshape((-1, self.block_size)),
        )
        self.application_count += 1

    def close(self):
        """Release the hierarchy and its owned scalar AMGX coarse solver."""
        self.cycle.close()


__all__ = ["BernsteinHpSymmetricPartPreconditioner"]
