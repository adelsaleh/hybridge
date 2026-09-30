"""Experimental native hp-BSR preconditioning for nonsymmetric trace solves."""
from __future__ import annotations

import numpy as np

from hdgfem.backends.cupy import require_cupy
from hdgfem.backends.legendre_face_bsr import legendre_orthonormal_scales
from hdgfem.core.space import _bernstein_edge_basis, _legendre_edge_basis
from hdgfem.linalg.face_hp_multigrid import AmgxScalarVcycle, FaceBlockPmgPrototype
from hdgfem.linalg.face_hp_policy import face_hp_mg_preconditioner_parameters


class BernsteinHpSymmetricPartPreconditioner:
    """Apply the native hp cycle to the symmetric part in modal coordinates.

    Intended for GMRES on the original, nonsymmetric ADR operator. The native
    diffusion-oriented smoother must be suitable for the symmetric part;
    successful setup alone does not certify positive definiteness. The existing
    scalar AMGX V-cycle supplies the p=0 coarse correction.
    """

    def __init__(self, matrix, *, degree, policy="standard", tuning=None):
        """Build from host Bernstein face-BSR data, retaining device buffers."""
        cp = require_cupy()
        q = int(degree) + 1
        if getattr(matrix, "blocksize", None) != (q, q):
            raise ValueError("Require natural square Bernstein face blocks")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Require a square trace operator")
        symmetric = ((matrix + matrix.T) * 0.5).tobsr(blocksize=(q, q))
        symmetric.sum_duplicates()
        symmetric.sort_indices()

        points, _ = np.polynomial.legendre.leggauss(q)
        bernstein = _bernstein_edge_basis(degree, points)
        modal = _legendre_edge_basis(degree, points)
        modal = modal * legendre_orthonormal_scales(q)[:, None]
        # T maps orthonormal Legendre coefficients to Bernstein coefficients.
        transform = np.linalg.solve(bernstein.T, modal.T)
        data = np.ascontiguousarray(transform.T @ symmetric.data @ transform)
        faces = matrix.shape[0] // q
        positions = np.empty(faces, dtype=np.int32)
        for row in range(faces):
            start, end = symmetric.indptr[row:row + 2]
            found = np.flatnonzero(symmetric.indices[start:end] == row)
            if found.size != 1:
                raise ValueError("Every face requires a diagonal block")
            positions[row] = start + found[0]

        parameters = face_hp_mg_preconditioner_parameters(policy, overrides=tuning)
        self.parameters = parameters
        self.cp = cp
        self.num_dofs = matrix.shape[0]
        self.dtype = cp.dtype(data.dtype)
        self.device_id = int(cp.cuda.runtime.getDevice())
        self.block_size = q
        self.transform = cp.asarray(transform)
        self.modal_rhs = cp.empty((faces, q), dtype=self.dtype)
        self.cycle = FaceBlockPmgPrototype(
            indptr=cp.asarray(symmetric.indptr, dtype=cp.int32),
            indices=cp.asarray(symmetric.indices, dtype=cp.int32),
            orthonormal_data=cp.asarray(data),
            degree=degree,
            diagonal_positions=cp.asarray(positions),
            schedule=parameters["schedule"],
            chebyshev_order=parameters["chebyshev_order"],
            presweeps=parameters["presweeps"],
            postsweeps=parameters["postsweeps"],
            spmv_backend="auto",
            smoother_backend="fused-raw-cuda",
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
