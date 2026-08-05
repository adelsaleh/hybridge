"""Sparse HDG Gram matrices and dual-norm inverse applications."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.linalg import LinearOperator, bicgstab, cg, spilu

try:  # pragma: no cover - import behavior depends on optional runtime package.
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        """Provide an identity decorator when Numba is unavailable."""
        if args and callable(args[0]):
            return args[0]

        def decorate(func):
            """Return the decorated function unchanged."""
            return func

        return decorate

from ..core.space import DGSpace


@dataclass(frozen=True)
class HDGGram:
    """Sparse HDG Gram matrix with compatible residual-vector dimensions."""

    matrix: object
    local_dofs: int
    trace_dofs: int

    @property
    def ndof(self) -> int:
        """Total number of Gram/residual degrees of freedom."""
        return int(self.matrix.shape[0])


@dataclass(frozen=True)
class GramSolveDiagnostics:
    """Diagnostics from an iterative Gram inverse application."""

    info: int
    iterations: int
    relative_residual: float
    elapsed: float


@dataclass(frozen=True)
class ILUBiCGSTABGramInverse:
    """Reusable ``spilu`` preconditioner for ``G z = r`` solved by BiCGSTAB."""

    gram: object
    preconditioner: LinearOperator
    setup_seconds: float
    fill_ratio: float

    def solve(
            self,
            rhs: np.ndarray,
            *,
            rtol: float = 1.0e-11,
            atol: float = 0.0,
            maxiter: int | None = None,
    ) -> tuple[np.ndarray, GramSolveDiagnostics]:
        """Solve ``G z = rhs`` with ILU-preconditioned BiCGSTAB."""
        rhs = np.asarray(rhs, dtype=np.float64)
        iterations = 0

        def callback(_):
            """Count one Krylov iteration."""
            nonlocal iterations
            iterations += 1

        start = time.perf_counter()
        z, info = bicgstab(
            self.gram,
            rhs,
            M=self.preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            callback=callback,
        )
        elapsed = time.perf_counter() - start
        residual = rhs - self.gram @ z
        relative_residual = np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-300)
        return z, GramSolveDiagnostics(
            info=int(info),
            iterations=int(iterations),
            relative_residual=float(relative_residual),
            elapsed=float(elapsed),
        )

    def dual_norm_squared(
            self,
            residual: np.ndarray,
            *,
            rtol: float = 1.0e-11,
            atol: float = 0.0,
            maxiter: int | None = None,
    ) -> tuple[float, GramSolveDiagnostics]:
        """Return ``residual.T @ G^{-1} @ residual``."""
        z, diagnostics = self.solve(residual, rtol=rtol, atol=atol, maxiter=maxiter)
        value = float(np.asarray(residual, dtype=np.float64) @ z)
        if value < 0.0 and abs(value) < 1.0e-12:
            value = 0.0
        if value < 0.0:
            raise FloatingPointError(f"negative HDG dual-norm square {value}")
        return value, diagnostics


@dataclass(frozen=True)
class CondensedHDGGramInverse:
    """Static-condensed inverse application for the HDG Gram matrix.

    The full Gram matrix is never factored.  The ``q`` and element ``u`` blocks
    are inverted locally, while the trace Schur complement is solved by CG with
    an edge-block Jacobi preconditioner.
    """

    local_dofs: int
    trace_dofs: int
    element_count: int
    element_dofs: int
    edge_dofs: int
    aff_jacs: np.ndarray
    reference_mass: np.ndarray
    reference_mass_inverse: np.ndarray
    u_block: np.ndarray
    u_block_inverse: np.ndarray
    face_u_trace: np.ndarray
    face_trace_trace: np.ndarray
    face_free_edges: np.ndarray
    interior_face_flat: np.ndarray
    face_free_flat: np.ndarray
    edge_block_inverse: np.ndarray
    setup_seconds: float
    default_rtol: float = 1.0e-11
    default_atol: float = 0.0
    default_maxiter: int | None = None
    verbose_every: int = 0
    verify_residual: bool = True
    fill_ratio: float = 0.0
    numba_enabled: bool = NUMBA_AVAILABLE

    @property
    def ndof(self) -> int:
        """Total compatible residual-vector size."""
        return int(self.local_dofs + self.trace_dofs)

    def _trace_matvec(self, trace_vector: np.ndarray) -> np.ndarray:
        """Apply the statically condensed trace Schur complement."""
        trace_by_edge = np.asarray(trace_vector, dtype=np.float64).reshape(-1, self.edge_dofs)
        element_contrib = _condensed_schur_element_contrib(
            trace_by_edge,
            self.u_block_inverse,
            self.face_u_trace,
            self.face_trace_trace,
            self.face_free_edges,
        )
        out = np.zeros((self.trace_dofs // self.edge_dofs, self.edge_dofs), dtype=np.float64)
        np.add.at(
            out,
            self.face_free_flat,
            element_contrib.reshape(-1, self.edge_dofs)[self.interior_face_flat],
        )
        return out.reshape(-1)

    def _trace_preconditioner(self, trace_vector: np.ndarray) -> np.ndarray:
        """Apply the edge-block Jacobi inverse to a trace vector."""
        trace_by_edge = np.asarray(trace_vector, dtype=np.float64).reshape(-1, self.edge_dofs)
        out = _apply_edge_block_inverse(trace_by_edge, self.edge_block_inverse)
        return out.reshape(-1)

    def solve(
            self,
            rhs: np.ndarray,
            *,
            rtol: float | None = None,
            atol: float | None = None,
            maxiter: int | None = None,
    ) -> tuple[np.ndarray, GramSolveDiagnostics]:
        """Solve ``G z = rhs`` using HDG static condensation."""
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape != (self.ndof,):
            raise ValueError(f"rhs must have shape ({self.ndof},); got {rhs.shape}")
        rtol = self.default_rtol if rtol is None else float(rtol)
        atol = self.default_atol if atol is None else float(atol)
        maxiter = self.default_maxiter if maxiter is None else maxiter

        if self.verbose_every > 0:
            print(
                f"GRAM_CONDENSE_START ndof={self.ndof} traceDofs={self.trace_dofs} "
                f"rtol={rtol:.3e} atol={atol:.3e} maxiter={maxiter}",
                flush=True,
            )
        local_rhs = rhs[:self.local_dofs].reshape(self.element_count, 3 * self.element_dofs)
        trace_rhs = rhs[self.local_dofs:]
        local_out = np.empty_like(local_rhs)
        tmp_u = np.empty((self.element_count, self.element_dofs), dtype=np.float64)
        condense_start = time.perf_counter()
        _apply_local_rhs_inverse(
            local_rhs,
            self.aff_jacs,
            self.reference_mass_inverse,
            self.u_block_inverse,
            local_out,
            tmp_u,
        )

        rhs_trace_by_edge = trace_rhs.reshape(-1, self.edge_dofs).copy()
        rhs_element_contrib = _trace_rhs_element_contrib(tmp_u, self.face_u_trace, self.face_free_edges)
        np.add.at(
            rhs_trace_by_edge,
            self.face_free_flat,
            rhs_element_contrib.reshape(-1, self.edge_dofs)[self.interior_face_flat],
        )
        condensed_rhs = rhs_trace_by_edge.reshape(-1)
        if self.verbose_every > 0:
            print(
                f"GRAM_CONDENSE_DONE rhsNorm={np.linalg.norm(rhs):.6e} "
                f"traceRhsNorm={np.linalg.norm(condensed_rhs):.6e} "
                f"elapsed={time.perf_counter() - condense_start:.3f}",
                flush=True,
            )

        iterations = 0
        start = time.perf_counter()

        def callback(_):
            """Count one Krylov iteration."""
            nonlocal iterations
            iterations += 1
            if self.verbose_every > 0 and iterations % self.verbose_every == 0:
                print(
                    f"GRAM_CG_PROGRESS iterations={iterations} elapsed={time.perf_counter() - start:.3f}",
                    flush=True,
                )

        schur = LinearOperator(
            (self.trace_dofs, self.trace_dofs),
            matvec=self._trace_matvec,
            dtype=np.float64,
        )
        preconditioner = LinearOperator(
            (self.trace_dofs, self.trace_dofs),
            matvec=self._trace_preconditioner,
            dtype=np.float64,
        )
        if self.verbose_every > 0:
            print("GRAM_CG_START", flush=True)
        trace_solution, info = cg(
            schur,
            condensed_rhs,
            M=preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            callback=callback,
        )
        elapsed = time.perf_counter() - start
        if self.verbose_every > 0:
            print(f"GRAM_CG_DONE info={info} iterations={iterations} elapsed={elapsed:.3f}", flush=True)
        if info != 0:
            print(f"GRAM_CG_WARNING info={info} iterations={iterations}", flush=True)

        if self.verbose_every > 0:
            print("GRAM_RECOVER_START", flush=True)
        recover_start = time.perf_counter()
        trace_by_edge = trace_solution.reshape(-1, self.edge_dofs)
        _recover_local_u(
            tmp_u,
            trace_by_edge,
            self.u_block_inverse,
            self.face_u_trace,
            self.face_free_edges,
            local_out,
        )
        z = np.concatenate((local_out.reshape(-1), trace_solution))
        if self.verbose_every > 0:
            print(f"GRAM_RECOVER_DONE elapsed={time.perf_counter() - recover_start:.3f}", flush=True)

        rhs_norm = max(float(np.linalg.norm(rhs)), 1.0e-300)
        if self.verify_residual:
            if self.verbose_every > 0:
                print("GRAM_VERIFY_START", flush=True)
            verify_start = time.perf_counter()
            local_residual, trace_residual = self.residual(rhs, z)
            residual_norm = np.sqrt(float(local_residual @ local_residual + trace_residual @ trace_residual))
            relative_residual = float(residual_norm / rhs_norm)
            if self.verbose_every > 0:
                print(
                    f"GRAM_VERIFY_DONE rel={relative_residual:.3e} "
                    f"elapsed={time.perf_counter() - verify_start:.3f}",
                    flush=True,
                )
        else:
            relative_residual = float("nan")
        return z, GramSolveDiagnostics(
            info=int(info),
            iterations=int(iterations),
            relative_residual=relative_residual,
            elapsed=float(elapsed),
        )

    def residual(self, rhs: np.ndarray, solution: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``rhs - G solution`` without assembling the full Gram matrix."""
        rhs = np.asarray(rhs, dtype=np.float64)
        solution = np.asarray(solution, dtype=np.float64)
        local_solution = solution[:self.local_dofs].reshape(self.element_count, 3 * self.element_dofs)
        trace_solution = solution[self.local_dofs:].reshape(-1, self.edge_dofs)
        local_applied = _apply_local_gram(
            local_solution,
            trace_solution,
            self.aff_jacs,
            self.reference_mass,
            self.u_block,
            self.face_u_trace,
            self.face_free_edges,
        )
        trace_element_contrib = _trace_apply_element_contrib(
            local_solution,
            trace_solution,
            self.face_u_trace,
            self.face_trace_trace,
            self.face_free_edges,
        )
        trace_applied = np.zeros((self.trace_dofs // self.edge_dofs, self.edge_dofs), dtype=np.float64)
        np.add.at(
            trace_applied,
            self.face_free_flat,
            trace_element_contrib.reshape(-1, self.edge_dofs)[self.interior_face_flat],
        )
        return (
            rhs[:self.local_dofs] - local_applied.reshape(-1),
            rhs[self.local_dofs:] - trace_applied.reshape(-1),
        )

    def dual_norm_squared(
            self,
            residual: np.ndarray,
            *,
            rtol: float | None = None,
            atol: float | None = None,
            maxiter: int | None = None,
    ) -> tuple[float, GramSolveDiagnostics]:
        """Return ``residual.T @ G^{-1} @ residual``."""
        z, diagnostics = self.solve(residual, rtol=rtol, atol=atol, maxiter=maxiter)
        value = float(np.asarray(residual, dtype=np.float64) @ z)
        if value < 0.0 and abs(value) < 1.0e-12:
            value = 0.0
        if value < 0.0:
            raise FloatingPointError(f"negative HDG dual-norm square {value}")
        return value, diagnostics


@njit(cache=True, parallel=True)
def _apply_local_rhs_inverse(local_rhs, aff_jacs, reference_mass_inverse, u_block_inverse, local_out, tmp_u):
    """Apply the element-local Gram inverse to the local RHS blocks."""
    element_count = local_rhs.shape[0]
    element_dofs = reference_mass_inverse.shape[0]
    for element in prange(element_count):
        for i in range(element_dofs):
            value = 0.0
            for j in range(element_dofs):
                value += u_block_inverse[element, i, j] * local_rhs[element, j]
            tmp_u[element, i] = value
            local_out[element, i] = value
        inv_jac = 1.0 / aff_jacs[element]
        for component in range(2):
            offset = (component + 1) * element_dofs
            for i in range(element_dofs):
                value = 0.0
                for j in range(element_dofs):
                    value += reference_mass_inverse[i, j] * local_rhs[element, offset + j]
                local_out[element, offset + i] = inv_jac * value


@njit(cache=True, parallel=True)
def _trace_rhs_element_contrib(tmp_u, face_u_trace, face_free_edges):
    """Form element contributions to the condensed trace RHS."""
    element_count, face_count, element_dofs, edge_dofs = face_u_trace.shape
    out = np.zeros((element_count, face_count, edge_dofs), dtype=np.float64)
    for element in prange(element_count):
        for face in range(face_count):
            if face_free_edges[element, face] >= 0:
                for a in range(edge_dofs):
                    value = 0.0
                    for i in range(element_dofs):
                        value += face_u_trace[element, face, i, a] * tmp_u[element, i]
                    out[element, face, a] = value
    return out


@njit(cache=True, parallel=True)
def _condensed_schur_element_contrib(trace_by_edge, u_block_inverse, face_u_trace, face_trace_trace, face_free_edges):
    """Apply each element condensed Schur block to a trace vector."""
    element_count, face_count, element_dofs, edge_dofs = face_u_trace.shape
    out = np.zeros((element_count, face_count, edge_dofs), dtype=np.float64)
    local_u = np.zeros((element_count, element_dofs), dtype=np.float64)
    solved_u = np.zeros((element_count, element_dofs), dtype=np.float64)
    for element in prange(element_count):
        for face in range(face_count):
            edge = face_free_edges[element, face]
            if edge >= 0:
                for i in range(element_dofs):
                    value = 0.0
                    for a in range(edge_dofs):
                        value += face_u_trace[element, face, i, a] * trace_by_edge[edge, a]
                    local_u[element, i] += value
        for i in range(element_dofs):
            value = 0.0
            for j in range(element_dofs):
                value += u_block_inverse[element, i, j] * local_u[element, j]
            solved_u[element, i] = value
        for face in range(face_count):
            edge = face_free_edges[element, face]
            if edge >= 0:
                for a in range(edge_dofs):
                    value = 0.0
                    for b in range(edge_dofs):
                        value += face_trace_trace[element, face, a, b] * trace_by_edge[edge, b]
                    for i in range(element_dofs):
                        value -= face_u_trace[element, face, i, a] * solved_u[element, i]
                    out[element, face, a] = value
    return out


@njit(cache=True, parallel=True)
def _apply_edge_block_inverse(trace_by_edge, edge_block_inverse):
    """Apply independent inverse diagonal blocks to edge trace values."""
    edge_count, edge_dofs = trace_by_edge.shape
    out = np.empty_like(trace_by_edge)
    for edge in prange(edge_count):
        for a in range(edge_dofs):
            value = 0.0
            for b in range(edge_dofs):
                value += edge_block_inverse[edge, a, b] * trace_by_edge[edge, b]
            out[edge, a] = value
    return out


@njit(cache=True, parallel=True)
def _recover_local_u(tmp_u, trace_by_edge, u_block_inverse, face_u_trace, face_free_edges, local_out):
    """Recover the local scalar field after the condensed trace solve."""
    element_count, face_count, element_dofs, edge_dofs = face_u_trace.shape
    for element in prange(element_count):
        local_u = np.zeros(element_dofs, dtype=np.float64)
        solved_u = np.zeros(element_dofs, dtype=np.float64)
        for face in range(face_count):
            edge = face_free_edges[element, face]
            if edge >= 0:
                for i in range(element_dofs):
                    value = 0.0
                    for a in range(edge_dofs):
                        value += face_u_trace[element, face, i, a] * trace_by_edge[edge, a]
                    local_u[i] += value
        for i in range(element_dofs):
            value = 0.0
            for j in range(element_dofs):
                value += u_block_inverse[element, i, j] * local_u[j]
            solved_u[i] = value
        for i in range(element_dofs):
            local_out[element, i] = tmp_u[element, i] + solved_u[i]


@njit(cache=True, parallel=True)
def _apply_local_gram(local_solution, trace_by_edge, aff_jacs, reference_mass, u_block, face_u_trace, face_free_edges):
    """Apply the local blocks of the HDG Gram operator."""
    element_count, face_count, element_dofs, edge_dofs = face_u_trace.shape
    out = np.zeros_like(local_solution)
    for element in prange(element_count):
        jac = aff_jacs[element]
        u_offset = 0
        for i in range(element_dofs):
            value = 0.0
            for j in range(element_dofs):
                value += u_block[element, i, j] * local_solution[element, u_offset + j]
            out[element, u_offset + i] = value
        for component in range(2):
            offset = (component + 1) * element_dofs
            for i in range(element_dofs):
                value = 0.0
                for j in range(element_dofs):
                    value += reference_mass[i, j] * local_solution[element, offset + j]
                out[element, offset + i] = jac * value
        for face in range(face_count):
            edge = face_free_edges[element, face]
            if edge >= 0:
                for i in range(element_dofs):
                    value = 0.0
                    for a in range(edge_dofs):
                        value += face_u_trace[element, face, i, a] * trace_by_edge[edge, a]
                    out[element, u_offset + i] -= value
    return out


@njit(cache=True, parallel=True)
def _trace_apply_element_contrib(local_solution, trace_by_edge, face_u_trace, face_trace_trace, face_free_edges):
    """Form element contributions to the trace block of a Gram product."""
    element_count, face_count, element_dofs, edge_dofs = face_u_trace.shape
    out = np.zeros((element_count, face_count, edge_dofs), dtype=np.float64)
    u_offset = 0
    for element in prange(element_count):
        for face in range(face_count):
            edge = face_free_edges[element, face]
            if edge >= 0:
                for a in range(edge_dofs):
                    value = 0.0
                    for i in range(element_dofs):
                        value -= face_u_trace[element, face, i, a] * local_solution[element, u_offset + i]
                    for b in range(edge_dofs):
                        value += face_trace_trace[element, face, a, b] * trace_by_edge[edge, b]
                    out[element, face, a] = value
    return out


def _physical_stiffness_blocks(space: DGSpace) -> np.ndarray:
    """Assemble physical-element stiffness blocks for the scalar field."""
    mesh = space.mesh
    q = space.quad_data
    gradients = np.einsum(
        "KcD,Diq->Kciq",
        mesh.inv_aff_mats_t,
        q.dbas_of_quads,
        optimize=True,
    )
    return np.einsum(
        "K,q,Kciq,Kcjq->Kij",
        mesh.aff_jacs,
        q.Krf_w,
        gradients,
        gradients,
        optimize=True,
    )


def _face_jump_weight(space: DGSpace, sigma: float, jump_weight: str) -> np.ndarray:
    """Return unit or mesh-scaled trace jump weights for every face."""
    if jump_weight == "unit":
        return np.ones_like(space.mesh.jacs_el_fc, dtype=np.float64)
    if jump_weight != "scaled":
        raise ValueError("jump_weight must be 'unit' or 'scaled'")
    p = max(1, int(space.order))
    face_length = 2.0 * space.mesh.jacs_el_fc
    return float(sigma) * p * p / face_length


def assemble_hdg_gram(space: DGSpace, *, sigma: float = 10.0, jump_weight: str = "unit"):
    """Assemble the sparse HDG Gram matrix on ``(q_x, q_y, u, uhat)``.

    Boundary trace degrees of freedom are eliminated, so ``uhat`` contains
    only interior-edge trace coefficients.
    """
    mesh = space.mesh
    q = space.quad_data
    tau = _face_jump_weight(space, sigma, jump_weight)
    tau_jac = tau * mesh.jacs_el_fc

    mass = mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
    stiffness = _physical_stiffness_blocks(space)
    face_uu = tau_jac[:, :, None, None] * q.face_element_test_element_trial[None, :, :, :]
    local_uu = stiffness + np.sum(face_uu, axis=1)
    oriented = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    face_u_trace = tau_jac[:, :, None, None] * oriented.swapaxes(2, 3)
    face_trace_trace = tau_jac[:, :, None, None] * q.M_rf_fc[None, None, :, :]

    edge_to_free = np.full(mesh.num_edg, -1, dtype=np.int64)
    edge_to_free[mesh.int_edges_inds] = np.arange(mesh.int_edges_inds.size)

    num_elements = mesh.num_tri
    el_dof = q.el_dof
    edg_dof = q.edg_dof
    local_stride = 3 * el_dof
    local_dofs = num_elements * local_stride
    trace_dofs = mesh.int_edges_inds.size * edg_dof
    total_dofs = local_dofs + trace_dofs

    i_el, j_el = np.meshgrid(np.arange(el_dof), np.arange(el_dof), indexing="ij")
    element_base = np.arange(num_elements)[:, None, None, None] * local_stride
    block_offsets = np.array([0, el_dof, 2 * el_dof], dtype=np.int64)
    local_rows = element_base + block_offsets[None, :, None, None] + i_el[None, None, :, :]
    local_cols = element_base + block_offsets[None, :, None, None] + j_el[None, None, :, :]
    local_data = np.stack((local_uu, mass, mass), axis=1)

    face_free_edges = edge_to_free[mesh.loc2glob_edge]
    interior_side = face_free_edges >= 0
    trace_base = local_dofs + face_free_edges[:, :, None] * edg_dof
    trace_ids = trace_base + np.arange(edg_dof)[None, None, :]
    trace_ids = np.where(interior_side[:, :, None], trace_ids, 0)
    u_ids = (
        np.arange(num_elements)[:, None, None] * local_stride
        + np.arange(el_dof)[None, None, :]
    )

    u_trace_rows = np.broadcast_to(u_ids[:, :, :, None], (num_elements, 3, el_dof, edg_dof))
    u_trace_cols = np.broadcast_to(trace_ids[:, :, None, :], (num_elements, 3, el_dof, edg_dof))
    u_trace_data = np.where(interior_side[:, :, None, None], -face_u_trace, 0.0)
    trace_u_rows = np.swapaxes(u_trace_cols, 2, 3)
    trace_u_cols = np.swapaxes(u_trace_rows, 2, 3)
    trace_u_data = np.swapaxes(u_trace_data, 2, 3)

    trace_trace_rows = np.broadcast_to(trace_ids[:, :, :, None], (num_elements, 3, edg_dof, edg_dof))
    trace_trace_cols = np.broadcast_to(trace_ids[:, :, None, :], (num_elements, 3, edg_dof, edg_dof))
    trace_trace_data = np.where(interior_side[:, :, None, None], face_trace_trace, 0.0)

    rows = np.concatenate((
        local_rows.ravel(),
        u_trace_rows.ravel(),
        trace_u_rows.ravel(),
        trace_trace_rows.ravel(),
    ))
    cols = np.concatenate((
        local_cols.ravel(),
        u_trace_cols.ravel(),
        trace_u_cols.ravel(),
        trace_trace_cols.ravel(),
    ))
    data = np.concatenate((
        local_data.ravel(),
        u_trace_data.ravel(),
        trace_u_data.ravel(),
        trace_trace_data.ravel(),
    ))
    matrix = coo_array((data, (rows, cols)), shape=(total_dofs, total_dofs)).tocsr()
    matrix.eliminate_zeros()
    return HDGGram(matrix=matrix, local_dofs=local_dofs, trace_dofs=trace_dofs)


def build_condensed_hdg_gram_inverse(
        space: DGSpace,
        *,
        sigma: float = 10.0,
        jump_weight: str = "unit",
        cg_rtol: float = 1.0e-11,
        cg_atol: float = 0.0,
        cg_maxiter: int | None = None,
        verbose_every: int = 0,
        verify_residual: bool = True,
) -> CondensedHDGGramInverse:
    """Build a memory-light static-condensed inverse for HDG Gram dual norms."""
    start = time.perf_counter()
    mesh = space.mesh
    q = space.quad_data
    tau = _face_jump_weight(space, sigma, jump_weight)
    tau_jac = tau * mesh.jacs_el_fc

    stiffness = _physical_stiffness_blocks(space)
    face_uu = tau_jac[:, :, None, None] * q.face_element_test_element_trial[None, :, :, :]
    u_block = np.ascontiguousarray(stiffness + np.sum(face_uu, axis=1), dtype=np.float64)
    u_block_inverse = np.ascontiguousarray(np.linalg.inv(u_block), dtype=np.float64)

    oriented = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    face_u_trace = np.ascontiguousarray(
        tau_jac[:, :, None, None] * oriented.swapaxes(2, 3),
        dtype=np.float64,
    )
    face_trace_trace = np.ascontiguousarray(
        tau_jac[:, :, None, None] * q.M_rf_fc[None, None, :, :],
        dtype=np.float64,
    )

    edge_to_free = np.full(mesh.num_edg, -1, dtype=np.int64)
    edge_to_free[mesh.int_edges_inds] = np.arange(mesh.int_edges_inds.size)
    face_free_edges = np.ascontiguousarray(edge_to_free[mesh.loc2glob_edge], dtype=np.int64)
    interior_face_mask = face_free_edges.reshape(-1) >= 0
    interior_face_flat = np.flatnonzero(interior_face_mask).astype(np.int64)
    face_free_flat = np.ascontiguousarray(face_free_edges.reshape(-1)[interior_face_flat], dtype=np.int64)

    a_inv_b = np.einsum("Kij,Kfja->Kfia", u_block_inverse, face_u_trace, optimize=True)
    same_face_schur = face_trace_trace - np.einsum("Kfia,Kfib->Kfab", face_u_trace, a_inv_b, optimize=True)
    edge_blocks = np.zeros((mesh.int_edges_inds.size, q.edg_dof, q.edg_dof), dtype=np.float64)
    np.add.at(
        edge_blocks,
        face_free_flat,
        same_face_schur.reshape(-1, q.edg_dof, q.edg_dof)[interior_face_flat],
    )
    try:
        edge_block_inverse = np.ascontiguousarray(np.linalg.inv(edge_blocks), dtype=np.float64)
    except np.linalg.LinAlgError as exc:
        min_diag = float(np.min(np.diagonal(edge_blocks, axis1=1, axis2=2)))
        raise np.linalg.LinAlgError(
            f"failed to invert condensed Gram edge-block preconditioner; min diagonal={min_diag:.6e}"
        ) from exc

    element_count = mesh.num_tri
    element_dofs = q.el_dof
    edge_dofs = q.edg_dof
    local_dofs = element_count * 3 * element_dofs
    trace_dofs = mesh.int_edges_inds.size * edge_dofs
    local_storage = (
        u_block.size
        + u_block_inverse.size
        + face_u_trace.size
        + face_trace_trace.size
        + edge_block_inverse.size
    )
    compatible_sparse_nnz = max(
        element_count * (3 * element_dofs * element_dofs + 6 * element_dofs * edge_dofs)
        + mesh.int_edges_inds.size * edge_dofs * edge_dofs,
        1,
    )
    return CondensedHDGGramInverse(
        local_dofs=local_dofs,
        trace_dofs=trace_dofs,
        element_count=element_count,
        element_dofs=element_dofs,
        edge_dofs=edge_dofs,
        aff_jacs=np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        reference_mass=np.ascontiguousarray(q.MKrf, dtype=np.float64),
        reference_mass_inverse=np.ascontiguousarray(q.MKrf_inv, dtype=np.float64),
        u_block=u_block,
        u_block_inverse=u_block_inverse,
        face_u_trace=face_u_trace,
        face_trace_trace=face_trace_trace,
        face_free_edges=face_free_edges,
        interior_face_flat=interior_face_flat,
        face_free_flat=face_free_flat,
        edge_block_inverse=edge_block_inverse,
        setup_seconds=time.perf_counter() - start,
        default_rtol=float(cg_rtol),
        default_atol=float(cg_atol),
        default_maxiter=cg_maxiter,
        verbose_every=int(verbose_every),
        verify_residual=bool(verify_residual),
        fill_ratio=float(local_storage / compatible_sparse_nnz),
    )


def build_flux_jump_gram_inverse(
        space: DGSpace,
        *,
        cg_rtol: float = 1.0e-11,
        cg_atol: float = 0.0,
        cg_maxiter: int | None = None,
        verbose_every: int = 0,
        verify_residual: bool = True,
) -> CondensedHDGGramInverse:
    r"""Build the condensed inverse for the unit flux-plus-trace-jump Gram.

    This is the HDG dual-norm inverse used when a residual vector is ordered as
    local blocks ``[u_h, q_{x,h}, q_{y,h}]`` followed by interior trace degrees
    of freedom, and the primal control norm is

    .. math::

        \|q_h\|_{L^2(\Omega)}^2
        + \|u_h-\widehat u_h\|_{L^2(\partial\mathcal T_h)}^2.

    It is a named wrapper around :func:`build_condensed_hdg_gram_inverse` with
    unit face-jump weights.
    """
    return build_condensed_hdg_gram_inverse(
        space,
        sigma=1.0,
        jump_weight="unit",
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
        verbose_every=verbose_every,
        verify_residual=verify_residual,
    )


def build_ilu_bicgstab_inverse(
        gram: HDGGram,
        *,
        drop_tol: float = 1.0e-12,
        fill_factor: float = 50.0,
) -> ILUBiCGSTABGramInverse:
    """Build a reusable ILU preconditioner for repeated HDG dual norms."""
    start = time.perf_counter()
    ilu = spilu(
        gram.matrix.tocsc(),
        drop_tol=float(drop_tol),
        fill_factor=float(fill_factor),
        permc_spec="COLAMD",
    )
    setup_seconds = time.perf_counter() - start
    preconditioner = LinearOperator(gram.matrix.shape, matvec=ilu.solve, dtype=np.float64)
    fill_ratio = float((ilu.L.nnz + ilu.U.nnz) / max(gram.matrix.nnz, 1))
    return ILUBiCGSTABGramInverse(
        gram=gram.matrix,
        preconditioner=preconditioner,
        setup_seconds=setup_seconds,
        fill_ratio=fill_ratio,
    )


__all__ = [
    "HDGGram",
    "ILUBiCGSTABGramInverse",
    "GramSolveDiagnostics",
    "CondensedHDGGramInverse",
    "assemble_hdg_gram",
    "build_condensed_hdg_gram_inverse",
    "build_flux_jump_gram_inverse",
    "build_ilu_bicgstab_inverse",
]
