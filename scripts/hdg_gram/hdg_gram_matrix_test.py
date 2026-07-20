"""Assemble and test an HDG energy Gram matrix.

The norm is defined on the HDG tuple ``(q_x, q_y, u, uhat)``:

    ||(q, u, uhat)||^2 =
        sum_K ||q||^2_K
      + sum_K ||grad u||^2_K
      + sum_K sum_{F subset dK} tau_F ||u - uhat||^2_F.

Boundary trace unknowns are eliminated, so ``uhat`` lives in ``M_h^0`` and
vanishes on boundary faces.  The script assembles the same sparse Gram matrix
with a vectorized NumPy path and a parallel Numba path, then checks that the
dual norm computed by solving ``G z = r`` satisfies
``||G x||_{G^{-1}} = ||x||_G``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_array, save_npz
from scipy.sparse.linalg import LinearOperator, cg, eigsh, gmres, spilu, spsolve, bicgstab

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace

try:  # pragma: no cover - exercised by the script when numba is installed.
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range


@dataclass(frozen=True)
class GramAssembly:
    """Sparse COO Gram matrix data and the associated HDG dimensions."""

    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]
    local_dofs: int
    trace_dofs: int

    def matrix(self):
        """Return the assembled CSR matrix with duplicate entries summed."""
        matrix = coo_array((self.data, (self.rows, self.cols)), shape=self.shape).tocsr()
        matrix.eliminate_zeros()
        return matrix


@dataclass(frozen=True)
class IterativeSolveDiagnostics:
    """Diagnostics for an iterative Gram inverse application."""

    info: int
    iterations: int
    relative_residual: float
    elapsed: float


@dataclass(frozen=True)
class GramInverse:
    """Reusable sparse approximate inverse setup for HDG Gram dual norms."""

    gram: object
    preconditioner: LinearOperator | None
    preconditioner_name: str
    setup_seconds: float
    fill_ratio: float

    def solve(
            self,
            rhs: np.ndarray,
            *,
            method: str = "gmres",
            rtol: float = 1.0e-11,
            atol: float = 0.0,
            maxiter: int | None = None,
    ) -> tuple[np.ndarray, IterativeSolveDiagnostics]:
        """Solve ``G z = rhs`` using a Krylov method and reusable preconditioner."""
        rhs = np.asarray(rhs, dtype=np.float64)
        iterations = 0

        def callback(_):
            nonlocal iterations
            iterations += 1

        start = time.perf_counter()
        if method == "gmres":
            z, info = gmres(
                self.gram,
                rhs,
                M=self.preconditioner,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                callback=callback,
                callback_type="pr_norm",
            )
        elif method == "cg":
            if self.preconditioner_name == "spilu":
                raise ValueError("CG requires an SPD preconditioner; use --preconditioner jacobi/none or --krylov gmres")
            z, info = cg(
                self.gram,
                rhs,
                M=self.preconditioner,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                callback=callback,
            )
        elif method == "bicgstab":
            z, info = bicgstab(
                self.gram,
                rhs,
                M=self.preconditioner,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                callback=callback,
            )
        else:
            raise ValueError("method must be 'gmres' or 'cg'")
        elapsed = time.perf_counter() - start

        residual = rhs - self.gram @ z
        relative_residual = np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0e-300)
        return z, IterativeSolveDiagnostics(
            info=int(info),
            iterations=int(iterations),
            relative_residual=float(relative_residual),
            elapsed=float(elapsed),
        )

    def dual_norm(
            self,
            residual: np.ndarray,
            *,
            method: str = "gmres",
            rtol: float = 1.0e-11,
            atol: float = 0.0,
            maxiter: int | None = None,
    ) -> tuple[float, IterativeSolveDiagnostics]:
        """Return ``sqrt(residual.T @ G^{-1} @ residual)``."""
        z, diagnostics = self.solve(residual, method=method, rtol=rtol, atol=atol, maxiter=maxiter)
        value = float(residual @ z)
        if value < 0.0 and abs(value) <= 1.0e-12 * max(1.0, abs(value)):
            value = 0.0
        if value < 0.0:
            raise FloatingPointError(f"computed negative dual-norm square {value}")
        return float(np.sqrt(value)), diagnostics


def build_ilu_inverse(
        gram,
        *,
        drop_tol: float,
        fill_factor: float,
) -> GramInverse:
    """Build a reusable SuperLU ILU preconditioner for ``G z = r``."""
    start = time.perf_counter()
    gram_csc = gram.tocsc()
    ilu = spilu(
        gram_csc,
        drop_tol=float(drop_tol),
        fill_factor=float(fill_factor),
        permc_spec="COLAMD",
    )
    setup_seconds = time.perf_counter() - start
    preconditioner = LinearOperator(gram.shape, matvec=ilu.solve, dtype=np.float64)
    fill_ratio = float((ilu.L.nnz + ilu.U.nnz) / max(gram.nnz, 1))
    return GramInverse(
        gram=gram,
        preconditioner=preconditioner,
        preconditioner_name="spilu",
        setup_seconds=setup_seconds,
        fill_ratio=fill_ratio,
    )


def build_jacobi_inverse(gram) -> GramInverse:
    """Build a diagonal SPD preconditioner for ``G z = r``."""
    start = time.perf_counter()
    diagonal = np.asarray(gram.diagonal(), dtype=np.float64)
    min_diagonal = float(np.min(diagonal))
    if min_diagonal <= 0.0:
        raise ValueError(f"Jacobi preconditioner requires positive diagonal entries; min={min_diagonal}")
    inv_diagonal = 1.0 / diagonal
    preconditioner = LinearOperator(gram.shape, matvec=lambda x: inv_diagonal * x, dtype=np.float64)
    return GramInverse(
        gram=gram,
        preconditioner=preconditioner,
        preconditioner_name="jacobi",
        setup_seconds=time.perf_counter() - start,
        fill_ratio=float(diagonal.size / max(gram.nnz, 1)),
    )


def build_unpreconditioned_inverse(gram) -> GramInverse:
    """Build an unpreconditioned Gram inverse wrapper."""
    return GramInverse(
        gram=gram,
        preconditioner=None,
        preconditioner_name="none",
        setup_seconds=0.0,
        fill_ratio=0.0,
    )


def build_gram_inverse(
        gram,
        *,
        preconditioner: str,
        drop_tol: float,
        fill_factor: float,
) -> GramInverse:
    """Build a reusable preconditioner for Gram inverse applications."""
    if preconditioner == "spilu":
        return build_ilu_inverse(gram, drop_tol=drop_tol, fill_factor=fill_factor)
    if preconditioner == "jacobi":
        return build_jacobi_inverse(gram)
    if preconditioner == "none":
        return build_unpreconditioned_inverse(gram)
    raise ValueError("preconditioner must be 'spilu', 'jacobi', or 'none'")


def _physical_stiffness_blocks(space: DGSpace) -> np.ndarray:
    """Assemble ``int_K grad(phi_i).grad(phi_j) dx`` for every element."""
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


def _face_tau(space: DGSpace, sigma: float) -> np.ndarray:
    """Return the per-element-face HDG stabilization scale."""
    p = max(1, int(space.order))
    face_length = 2.0 * space.mesh.jacs_el_fc
    return float(sigma) * p * p / face_length


def _gram_blocks(space: DGSpace, sigma: float):
    """Build local volume and face blocks used by both assembly paths."""
    mesh = space.mesh
    q = space.quad_data
    tau = _face_tau(space, sigma)
    tau_jac = tau * mesh.jacs_el_fc

    mass = mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
    stiffness = _physical_stiffness_blocks(space)
    face_uu = tau_jac[:, :, None, None] * q.face_element_test_element_trial[None, :, :, :]
    local_uu = stiffness + np.sum(face_uu, axis=1)

    oriented = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    face_u_trace = tau_jac[:, :, None, None] * oriented.swapaxes(2, 3)
    face_trace_trace = tau_jac[:, :, None, None] * q.M_rf_fc[None, None, :, :]
    return (
        np.ascontiguousarray(mass),
        np.ascontiguousarray(local_uu),
        np.ascontiguousarray(face_u_trace),
        np.ascontiguousarray(face_trace_trace),
    )


def _interior_edge_map(space: DGSpace) -> np.ndarray:
    """Map global mesh edges to compact ``M_h^0`` edge ids, or -1 on boundary."""
    edge_to_free = np.full(space.mesh.num_edg, -1, dtype=np.int64)
    edge_to_free[space.mesh.int_edges_inds] = np.arange(space.mesh.int_edges_inds.size)
    return edge_to_free


def assemble_hdg_gram_numpy(space: DGSpace, *, sigma: float = 10.0) -> GramAssembly:
    """Assemble the HDG Gram matrix with vectorized NumPy operations."""
    mesh = space.mesh
    q = space.quad_data
    mass, local_uu, face_u_trace, face_trace_trace = _gram_blocks(space, sigma)
    edge_to_free = _interior_edge_map(space)

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
    local_data = np.stack((mass, mass, local_uu), axis=1)

    local_rows = local_rows.ravel()
    local_cols = local_cols.ravel()
    local_data = local_data.ravel()

    face_free_edges = edge_to_free[mesh.loc2glob_edge]
    interior_side = face_free_edges >= 0
    trace_base = local_dofs + face_free_edges[:, :, None] * edg_dof
    trace_ids = trace_base + np.arange(edg_dof)[None, None, :]
    trace_ids = np.where(interior_side[:, :, None], trace_ids, 0)

    u_ids = (
        np.arange(num_elements)[:, None, None] * local_stride
        + 2 * el_dof
        + np.arange(el_dof)[None, None, :]
    )
    u_trace_rows = np.broadcast_to(u_ids[:, :, :, None], (num_elements, 3, el_dof, edg_dof))
    u_trace_cols = np.broadcast_to(trace_ids[:, :, None, :], (num_elements, 3, el_dof, edg_dof))
    u_trace_data = -face_u_trace
    u_trace_data = np.where(interior_side[:, :, None, None], u_trace_data, 0.0)

    trace_u_rows = np.swapaxes(u_trace_cols, 2, 3)
    trace_u_cols = np.swapaxes(u_trace_rows, 2, 3)
    trace_u_data = np.swapaxes(u_trace_data, 2, 3)

    trace_trace_rows = np.broadcast_to(trace_ids[:, :, :, None], (num_elements, 3, edg_dof, edg_dof))
    trace_trace_cols = np.broadcast_to(trace_ids[:, :, None, :], (num_elements, 3, edg_dof, edg_dof))
    trace_trace_data = np.where(
        interior_side[:, :, None, None],
        face_trace_trace,
        0.0,
    )

    rows = np.concatenate(
        (
            local_rows,
            u_trace_rows.ravel(),
            trace_u_rows.ravel(),
            trace_trace_rows.ravel(),
        )
    )
    cols = np.concatenate(
        (
            local_cols,
            u_trace_cols.ravel(),
            trace_u_cols.ravel(),
            trace_trace_cols.ravel(),
        )
    )
    data = np.concatenate(
        (
            local_data,
            u_trace_data.ravel(),
            trace_u_data.ravel(),
            trace_trace_data.ravel(),
        )
    )
    return GramAssembly(rows, cols, data, (total_dofs, total_dofs), local_dofs, trace_dofs)


if njit is not None:

    @njit(parallel=True, cache=True)
    def _assemble_hdg_gram_numba_kernel(
            mass,
            local_uu,
            face_u_trace,
            face_trace_trace,
            loc2glob_edge,
            edge_to_free,
            local_dofs,
            rows,
            cols,
            data,
    ) -> None:
        num_elements = mass.shape[0]
        el_dof = mass.shape[1]
        edg_dof = face_trace_trace.shape[2]
        local_stride = 3 * el_dof
        local_entries_per_element = 3 * el_dof * el_dof
        face_entries_per_element = 3 * (2 * el_dof * edg_dof + edg_dof * edg_dof)
        face_offset = num_elements * local_entries_per_element

        for k in prange(num_elements):
            local_start = k * local_entries_per_element
            cursor = local_start
            for block in range(3):
                block_offset = block * el_dof
                for i in range(el_dof):
                    row = k * local_stride + block_offset + i
                    for j in range(el_dof):
                        rows[cursor] = row
                        cols[cursor] = k * local_stride + block_offset + j
                        if block == 0 or block == 1:
                            data[cursor] = mass[k, i, j]
                        else:
                            data[cursor] = local_uu[k, i, j]
                        cursor += 1

            cursor = face_offset + k * face_entries_per_element
            for f in range(3):
                free_edge = edge_to_free[loc2glob_edge[k, f]]
                is_interior = free_edge >= 0
                for i in range(el_dof):
                    row = k * local_stride + 2 * el_dof + i
                    for a in range(edg_dof):
                        if is_interior:
                            col = local_dofs + free_edge * edg_dof + a
                            value = -face_u_trace[k, f, i, a]
                        else:
                            col = 0
                            value = 0.0
                        rows[cursor] = row
                        cols[cursor] = col
                        data[cursor] = value
                        cursor += 1

                for a in range(edg_dof):
                    if is_interior:
                        row = local_dofs + free_edge * edg_dof + a
                    else:
                        row = 0
                    for i in range(el_dof):
                        col = k * local_stride + 2 * el_dof + i
                        rows[cursor] = row
                        cols[cursor] = col
                        data[cursor] = -face_u_trace[k, f, i, a] if is_interior else 0.0
                        cursor += 1

                for a in range(edg_dof):
                    if is_interior:
                        row = local_dofs + free_edge * edg_dof + a
                    else:
                        row = 0
                    for b in range(edg_dof):
                        if is_interior:
                            col = local_dofs + free_edge * edg_dof + b
                            value = face_trace_trace[k, f, a, b]
                        else:
                            col = 0
                            value = 0.0
                        rows[cursor] = row
                        cols[cursor] = col
                        data[cursor] = value
                        cursor += 1


def assemble_hdg_gram_numba(space: DGSpace, *, sigma: float = 10.0) -> GramAssembly:
    """Assemble the HDG Gram matrix with a parallel Numba kernel."""
    if njit is None:
        raise RuntimeError("numba is not installed")

    mesh = space.mesh
    q = space.quad_data
    mass, local_uu, face_u_trace, face_trace_trace = _gram_blocks(space, sigma)
    edge_to_free = _interior_edge_map(space)

    num_elements = mesh.num_tri
    el_dof = q.el_dof
    edg_dof = q.edg_dof
    local_stride = 3 * el_dof
    local_dofs = num_elements * local_stride
    trace_dofs = mesh.int_edges_inds.size * edg_dof
    total_dofs = local_dofs + trace_dofs

    local_nnz = num_elements * 3 * el_dof * el_dof
    face_nnz = num_elements * 3 * (2 * el_dof * edg_dof + edg_dof * edg_dof)
    rows = np.empty(local_nnz + face_nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(rows.size, dtype=np.float64)

    _assemble_hdg_gram_numba_kernel(
        mass,
        local_uu,
        face_u_trace,
        face_trace_trace,
        mesh.loc2glob_edge,
        edge_to_free,
        local_dofs,
        rows,
        cols,
        data,
    )
    return GramAssembly(rows, cols, data, (total_dofs, total_dofs), local_dofs, trace_dofs)


def _relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def _check_spectrum(gram, *, dense_check_dofs: int, mode: str) -> None:
    """Check or estimate the Gram spectrum without forcing dense storage for large runs."""
    if mode == "skip":
        print("spectrum check skipped")
        return

    use_dense = mode == "dense" or (mode == "auto" and gram.shape[0] <= dense_check_dofs)
    use_eigsh = mode == "eigsh"

    if use_dense:
        if gram.shape[0] > dense_check_dofs and mode != "dense":
            print("dense spectrum check skipped")
            return
        dense = gram.toarray()
        eigvals = np.linalg.eigvalsh(dense)
        min_eig = float(eigvals[0])
        max_eig = float(eigvals[-1])
        cond = max_eig / min_eig
        print(f"dense eigenvalues: min={min_eig:.6e}, max={max_eig:.6e}, cond={cond:.6e}")
        if min_eig <= 0.0:
            raise AssertionError(f"Gram matrix is not positive definite; min_eig={min_eig}")
        return

    if mode == "auto" and not use_eigsh:
        print(f"spectrum check skipped for {gram.shape[0]} dofs; use --spectrum eigsh to estimate it")
        return

    try:
        min_eig = float(eigsh(gram, k=1, which="SA", return_eigenvectors=False)[0])
        max_eig = float(eigsh(gram, k=1, which="LA", return_eigenvectors=False)[0])
    except Exception as exc:
        print(f"sparse eigenvalue check skipped: {exc}")
        return
    cond = max_eig / min_eig
    print(f"sparse eigenvalues: min={min_eig:.6e}, max={max_eig:.6e}, cond={cond:.6e}")
    if min_eig <= 0.0:
        raise AssertionError(f"Gram matrix is not positive definite; min_eig={min_eig}")


def _direct_dual_norm(gram, residual: np.ndarray) -> tuple[float, float]:
    """Compute ``sqrt(r.T G^-1 r)`` with sparse direct solve."""
    start = time.perf_counter()
    z = spsolve(gram, residual)
    elapsed = time.perf_counter() - start
    value = float(residual @ z)
    if value < 0.0:
        raise FloatingPointError(f"direct solve produced negative dual-norm square {value}")
    return float(np.sqrt(value)), elapsed


def test_gram(
        space: DGSpace,
        *,
        sigma: float,
        seed: int,
        use_numba: bool,
        save_path: Path | None,
        dense_check_dofs: int,
        direct_check_dofs: int,
        preconditioner: str,
        ilu_drop_tol: float,
        ilu_fill_factor: float,
        krylov: str,
        krylov_rtol: float,
        krylov_atol: float,
        krylov_maxiter: int | None,
        spectrum: str,
) -> None:
    """Assemble the Gram matrix and print numerical consistency checks."""
    print(f"mesh: elements={space.mesh.num_tri}, edges={space.mesh.num_edg}, interior_edges={space.mesh.int_edges_inds.size}")
    print(f"space: order={space.order}, basis={space.reference.basis_type}, el_dof={space.el_dof}, edg_dof={space.quad_data.edg_dof}")
    print(f"stabilization: tau_F = {sigma:g} * p^2 / h_F")

    start = time.perf_counter()
    numpy_assembly = assemble_hdg_gram_numpy(space, sigma=sigma)
    numpy_time = time.perf_counter() - start
    gram = numpy_assembly.matrix()
    print(f"numpy assembly: nnz_raw={numpy_assembly.data.size}, nnz_csr={gram.nnz}, time={numpy_time:.4f}s")
    print(f"dofs: local={numpy_assembly.local_dofs}, trace_interior={numpy_assembly.trace_dofs}, total={gram.shape[0]}")
    sparse_mb = (gram.data.nbytes + gram.indices.nbytes + gram.indptr.nbytes) / 1024.0**2
    print(f"sparse storage: csr_bytes={sparse_mb:.3f} MiB")

    if save_path is not None:
        save_npz(save_path, gram)
        print(f"saved sparse Gram matrix: {save_path}")

    if use_numba:
        start = time.perf_counter()
        numba_assembly = assemble_hdg_gram_numba(space, sigma=sigma)
        first_numba_time = time.perf_counter() - start
        start = time.perf_counter()
        numba_assembly = assemble_hdg_gram_numba(space, sigma=sigma)
        numba_time = time.perf_counter() - start
        numba_gram = numba_assembly.matrix()
        diff = (numba_gram - gram).tocoo()
        max_diff = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
        print(
            "numba assembly: "
            f"nnz_raw={numba_assembly.data.size}, nnz_csr={numba_gram.nnz}, "
            f"compile_run={first_numba_time:.4f}s, warm_run={numba_time:.4f}s, "
            f"max_abs_diff={max_diff:.3e}"
        )
        if max_diff > 1.0e-11:
            raise AssertionError(f"NumPy and Numba Gram matrices differ by {max_diff}")

    symmetry_error = gram - gram.T
    symmetry_norm = 0.0 if symmetry_error.nnz == 0 else float(np.max(np.abs(symmetry_error.data)))
    print(f"symmetry max_abs={symmetry_norm:.3e}")
    if symmetry_norm > 1.0e-11:
        raise AssertionError(f"Gram matrix is not symmetric; max_abs={symmetry_norm}")

    _check_spectrum(gram, dense_check_dofs=dense_check_dofs, mode=spectrum)

    if krylov == "cg" and preconditioner == "spilu":
        raise ValueError("CG requires an SPD preconditioner; use --preconditioner jacobi/none or --krylov gmres")
    inverse = build_gram_inverse(
        gram,
        preconditioner=preconditioner,
        drop_tol=ilu_drop_tol,
        fill_factor=ilu_fill_factor,
    )
    if inverse.preconditioner_name == "spilu":
        print(
            "spilu preconditioner: "
            f"drop_tol={ilu_drop_tol:.1e}, fill_factor={ilu_fill_factor:g}, "
            f"fill_ratio={(inverse.fill_ratio):.3f}, setup={inverse.setup_seconds:.4f}s"
        )
    else:
        print(
            f"{inverse.preconditioner_name} preconditioner: "
            f"fill_ratio={(inverse.fill_ratio):.3f}, setup={inverse.setup_seconds:.4f}s"
        )

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(gram.shape[0])
    residual = gram @ x
    primal_norm = float(np.sqrt(x @ residual))
    dual_norm, diag = inverse.dual_norm(
        residual,
        method=krylov,
        rtol=krylov_rtol,
        atol=krylov_atol,
        maxiter=krylov_maxiter,
    )
    riesz_error = _relative_error(primal_norm, dual_norm)
    print(
        f"iterative riesz check: ||x||_G={primal_norm:.12e}, "
        f"||Gx||_G^-1={dual_norm:.12e}, rel={riesz_error:.3e}, "
        f"{krylov}_iters={diag.iterations}, solve={diag.elapsed:.4f}s, "
        f"linear_relres={diag.relative_residual:.3e}, info={diag.info}"
    )
    if diag.info != 0:
        raise AssertionError(f"{krylov} did not converge; info={diag.info}")
    if riesz_error > 1.0e-9:
        raise AssertionError("Iterative Riesz inverse check failed")

    r = rng.standard_normal(gram.shape[0])
    dual_norm, diag = inverse.dual_norm(
        r,
        method=krylov,
        rtol=krylov_rtol,
        atol=krylov_atol,
        maxiter=krylov_maxiter,
    )
    print(
        "iterative residual dual norm: "
        f"sqrt(r^T G^-1 r)={dual_norm:.12e}, "
        f"{krylov}_iters={diag.iterations}, solve={diag.elapsed:.4f}s, "
        f"linear_relres={diag.relative_residual:.3e}, info={diag.info}"
    )
    if diag.info != 0:
        raise AssertionError(f"{krylov} did not converge on random residual; info={diag.info}")

    if gram.shape[0] <= direct_check_dofs:
        direct_norm, direct_time = _direct_dual_norm(gram, r)
        direct_error = _relative_error(dual_norm, direct_norm)
        print(
            "direct sparse reference: "
            f"sqrt(r^T G^-1 r)={direct_norm:.12e}, solve={direct_time:.4f}s, "
            f"iterative_rel={direct_error:.3e}"
        )
        if direct_error > 1.0e-9:
            raise AssertionError("Iterative and direct sparse dual norms differ")
        if gram.shape[0] <= dense_check_dofs:
            dense = gram.toarray()
            dense_norm = float(np.sqrt(r @ np.linalg.solve(dense, r)))
            dense_error = _relative_error(direct_norm, dense_norm)
            print(f"dense reference: sqrt(r^T G^-1 r)={dense_norm:.12e}, direct_rel={dense_error:.3e}")
            if dense_error > 1.0e-11:
                raise AssertionError("Sparse direct and dense inverse dual norms differ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=3, help="number of rectangular cells in x")
    parser.add_argument("--ny", type=int, default=None, help="number of rectangular cells in y; defaults to nx")
    parser.add_argument("--order", type=int, default=2, help="DG polynomial order")
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--sigma", type=float, default=10.0, help="HDG face stabilization constant")
    parser.add_argument("--seed", type=int, default=7, help="random seed for inverse checks")
    parser.add_argument("--no-numba", action="store_true", help="skip the Numba assembly path")
    parser.add_argument("--save-gram", type=Path, default=None, help="optional path for scipy.sparse.save_npz output")
    parser.add_argument(
        "--dense-check-dofs",
        type=int,
        default=1000,
        help="maximum dofs for dense eigen/dense inverse reference checks",
    )
    parser.add_argument(
        "--spectrum",
        choices=("auto", "dense", "eigsh", "skip"),
        default="auto",
        help="spectrum check mode; auto uses dense checks only for small matrices",
    )
    parser.add_argument(
        "--direct-check-dofs",
        type=int,
        default=20000,
        help="maximum dofs for sparse direct-solve reference checks",
    )
    parser.add_argument(
        "--ilu-drop-tol",
        type=float,
        default=1.0e-12,
        help="SuperLU ILU drop tolerance; smaller is more accurate and uses more memory",
    )
    parser.add_argument(
        "--ilu-fill-factor",
        type=float,
        default=50.0,
        help="SuperLU ILU fill factor; larger is more accurate and uses more memory",
    )
    parser.add_argument(
        "--preconditioner",
        choices=("jacobi", "spilu", "none"),
        default="jacobi",
        help="Krylov preconditioner; spilu is nonsymmetric and should be used with gmres",
    )
    parser.add_argument("--krylov", choices=("gmres", "cg", "bicgstab"), default="cg", help="iterative method for G z = r")
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-11, help="relative tolerance for Krylov solves")
    parser.add_argument("--krylov-atol", type=float, default=0.0, help="absolute tolerance for Krylov solves")
    parser.add_argument("--krylov-maxiter", type=int, default=None, help="maximum Krylov iterations")
    args = parser.parse_args()
    if args.krylov == "cg" and args.preconditioner == "spilu":
        parser.error("CG requires an SPD preconditioner; use --preconditioner jacobi/none or --krylov gmres")
    return args


def main() -> None:
    args = parse_args()
    mesh = rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, args.order, basis_type=args.basis)
    test_gram(
        space,
        sigma=args.sigma,
        seed=args.seed,
        use_numba=not args.no_numba,
        save_path=args.save_gram,
        dense_check_dofs=args.dense_check_dofs,
        direct_check_dofs=args.direct_check_dofs,
        preconditioner=args.preconditioner,
        ilu_drop_tol=args.ilu_drop_tol,
        ilu_fill_factor=args.ilu_fill_factor,
        krylov=args.krylov,
        krylov_rtol=args.krylov_rtol,
        krylov_atol=args.krylov_atol,
        krylov_maxiter=args.krylov_maxiter,
        spectrum=args.spectrum,
    )


if __name__ == "__main__":
    main()
