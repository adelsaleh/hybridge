#!/usr/bin/env python3
"""Assemble and validate an HDG energy Gram matrix through package APIs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import save_npz
from scipy.sparse.linalg import eigsh, spsolve

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hdgfem.assembly.hdg_gram import assemble_hdg_gram, build_krylov_hdg_gram_inverse
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace


def _relative_error(a: float, b: float) -> float:
    """Return a symmetric relative error for two scalars."""
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def _check_spectrum(gram, *, dense_check_dofs: int, mode: str) -> None:
    """Check or estimate positivity without forcing dense storage for large runs."""
    if mode == "skip":
        print("spectrum check skipped")
        return
    use_dense = mode == "dense" or (mode == "auto" and gram.shape[0] <= dense_check_dofs)
    if use_dense:
        eigenvalues = np.linalg.eigvalsh(gram.toarray())
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        print(f"dense eigenvalues: min={minimum:.6e}, max={maximum:.6e}, cond={maximum / minimum:.6e}")
        if minimum <= 0.0:
            raise AssertionError(f"Gram matrix is not positive definite; min_eig={minimum}")
        return
    if mode == "auto":
        print(f"spectrum check skipped for {gram.shape[0]} dofs; use --spectrum eigsh to estimate it")
        return
    try:
        minimum = float(eigsh(gram, k=1, which="SA", return_eigenvectors=False)[0])
        maximum = float(eigsh(gram, k=1, which="LA", return_eigenvectors=False)[0])
    except Exception as exc:
        print(f"sparse eigenvalue check skipped: {exc}")
        return
    print(f"sparse eigenvalues: min={minimum:.6e}, max={maximum:.6e}, cond={maximum / minimum:.6e}")
    if minimum <= 0.0:
        raise AssertionError(f"Gram matrix is not positive definite; min_eig={minimum}")


def _direct_dual_norm(gram, residual: np.ndarray) -> tuple[float, float]:
    """Compute ``sqrt(r.T G^-1 r)`` with a sparse direct solve."""
    started = time.perf_counter()
    solution = spsolve(gram, residual)
    value = float(residual @ solution)
    if value < 0.0:
        raise FloatingPointError(f"direct solve produced negative dual-norm square {value}")
    return float(np.sqrt(value)), time.perf_counter() - started


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
    """Assemble with hdgfem and print numerical consistency checks."""
    print(f"mesh: elements={space.mesh.num_tri}, edges={space.mesh.num_edg}, interior_edges={space.mesh.int_edges_inds.size}")
    print(f"space: order={space.order}, basis={space.reference.basis_type}, el_dof={space.el_dof}, edg_dof={space.quad_data.edg_dof}")
    print(f"stabilization: tau_F = {sigma:g} * p^2 / h_F")
    if use_numba:
        print("assembly implementation: hdgfem package (backend selection is package-owned)")

    started = time.perf_counter()
    assembled = assemble_hdg_gram(space, sigma=sigma, jump_weight="scaled")
    assembly_seconds = time.perf_counter() - started
    gram = assembled.matrix
    print(f"package assembly: nnz={gram.nnz}, time={assembly_seconds:.4f}s")
    print(f"dofs: local={assembled.local_dofs}, trace_interior={assembled.trace_dofs}, total={assembled.ndof}")
    sparse_mib = (gram.data.nbytes + gram.indices.nbytes + gram.indptr.nbytes) / 1024.0**2
    print(f"sparse storage: csr_bytes={sparse_mib:.3f} MiB")

    if save_path is not None:
        save_npz(save_path, gram)
        print(f"saved sparse Gram matrix: {save_path}")

    symmetry = gram - gram.T
    symmetry_error = 0.0 if symmetry.nnz == 0 else float(np.max(np.abs(symmetry.data)))
    print(f"symmetry max_abs={symmetry_error:.3e}")
    if symmetry_error > 1.0e-11:
        raise AssertionError(f"Gram matrix is not symmetric; max_abs={symmetry_error}")
    _check_spectrum(gram, dense_check_dofs=dense_check_dofs, mode=spectrum)

    inverse = build_krylov_hdg_gram_inverse(
        assembled,
        preconditioner=preconditioner,
        drop_tol=ilu_drop_tol,
        fill_factor=ilu_fill_factor,
    )
    print(
        f"{inverse.preconditioner_name} preconditioner: "
        f"fill_ratio={inverse.fill_ratio:.3f}, setup={inverse.setup_seconds:.4f}s"
    )

    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(gram.shape[0])
    residual = gram @ vector
    primal_norm = float(np.sqrt(vector @ residual))
    dual_squared, diagnostics = inverse.dual_norm_squared(
        residual,
        method=krylov,
        rtol=krylov_rtol,
        atol=krylov_atol,
        maxiter=krylov_maxiter,
    )
    dual_norm = float(np.sqrt(dual_squared))
    riesz_error = _relative_error(primal_norm, dual_norm)
    print(
        f"iterative riesz check: ||x||_G={primal_norm:.12e}, "
        f"||Gx||_G^-1={dual_norm:.12e}, rel={riesz_error:.3e}, "
        f"{krylov}_iters={diagnostics.iterations}, solve={diagnostics.elapsed:.4f}s, "
        f"linear_relres={diagnostics.relative_residual:.3e}, info={diagnostics.info}"
    )
    if diagnostics.info != 0 or riesz_error > 1.0e-9:
        raise AssertionError("iterative Riesz inverse check failed")

    random_residual = rng.standard_normal(gram.shape[0])
    dual_squared, diagnostics = inverse.dual_norm_squared(
        random_residual,
        method=krylov,
        rtol=krylov_rtol,
        atol=krylov_atol,
        maxiter=krylov_maxiter,
    )
    dual_norm = float(np.sqrt(dual_squared))
    print(
        f"iterative residual dual norm: sqrt(r^T G^-1 r)={dual_norm:.12e}, "
        f"{krylov}_iters={diagnostics.iterations}, solve={diagnostics.elapsed:.4f}s, "
        f"linear_relres={diagnostics.relative_residual:.3e}, info={diagnostics.info}"
    )
    if diagnostics.info != 0:
        raise AssertionError(f"{krylov} did not converge; info={diagnostics.info}")

    if gram.shape[0] <= direct_check_dofs:
        direct_norm, direct_seconds = _direct_dual_norm(gram, random_residual)
        direct_error = _relative_error(dual_norm, direct_norm)
        print(f"direct sparse reference: norm={direct_norm:.12e}, solve={direct_seconds:.4f}s, iterative_rel={direct_error:.3e}")
        if direct_error > 1.0e-9:
            raise AssertionError("iterative and direct sparse dual norms differ")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=3)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-numba", action="store_true", help="compatibility flag; assembly backend is package-owned")
    parser.add_argument("--save-gram", type=Path, default=None)
    parser.add_argument("--dense-check-dofs", type=int, default=1000)
    parser.add_argument("--spectrum", choices=("auto", "dense", "eigsh", "skip"), default="auto")
    parser.add_argument("--direct-check-dofs", type=int, default=20000)
    parser.add_argument("--ilu-drop-tol", type=float, default=1.0e-12)
    parser.add_argument("--ilu-fill-factor", type=float, default=50.0)
    parser.add_argument("--preconditioner", choices=("jacobi", "spilu", "none"), default="jacobi")
    parser.add_argument("--krylov", choices=("gmres", "cg", "bicgstab"), default="cg")
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-11)
    parser.add_argument("--krylov-atol", type=float, default=0.0)
    parser.add_argument("--krylov-maxiter", type=int, default=None)
    args = parser.parse_args()
    if args.krylov == "cg" and args.preconditioner == "spilu":
        parser.error("CG requires an SPD preconditioner; use jacobi/none or gmres/bicgstab")
    return args


def main() -> None:
    """Build a small structured problem and run Gram validation."""
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
