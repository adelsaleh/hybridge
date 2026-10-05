import time
from dataclasses import dataclass
from typing import Callable, Union, Optional, Literal
import numba as nb
import numpy as np
from numpy.typing import NDArray
# from diff_rea_nb_params import r_eval, d_eval_diag, f_eval, bd_cond
from global_system import solve_global_system
from scipy.sparse.linalg import LinearOperator
from hdg_mats import dg_moments
from hdg_numba_helpers import (
    map_ref_to_phys_m11, lu_factor_inplace, lu_solve_inplace, map_edge_dof_bool,
    _zero_mat, _zero_vec, _gather_geom
)
from mesh_utils import Triangulation
from quadratures_new import TriangleQuadratureData

@nb.njit(cache=True, inline="always")
def r_eval(x, y):
    return 0.0  # pure Poisson default


@nb.njit(cache=True, inline="always")
def f_eval(x, y):
    # return 4
    return -(4 * np.cos(x ** 2 + y ** 2) - (x ** 2 + y ** 2) * (np.sin(x * y) + 4 * np.sin(x ** 2 + y ** 2)))
    # r = np.sqrt(x * x + y * y)
    # theta = np.arctan2(y, x)
    # return (1.0 + 0.05 * np.cos(3 * theta)) * np.exp(-((r - 0.45) ** 2) / (2.0 * 0.03 ** 2)) * (r<0.9)

@nb.njit(cache=True, inline="always")
def d_eval_diag(x, y):
    # diffusion tensor diag(d00,d11)
    return 1.0, 1.0


@nb.njit(cache=True, inline="always")
def bd_cond(x, y):
    return u_exact(x, y)

@nb.njit(cache=True, inline="always")
def u_exact(x, y):
    # return (1-x*x -y*y)
    # return np.sin(x ** 2 + y ** 2) + np.sin(x * y)
    return 0.0 * x * y


@dataclass
class LocalSolverCache:
    """
    Per-element cached local solver data for repeated solves on a fixed operator.

    Shapes
    ------
    nK   : number of elements
    nel  : number of elemental u-dofs
    ntr  : number of edge trace dofs per face
    m    : 3*ntr
    """
    Mhat: NDArray[np.float64]   # (nK, 3*nel, 3*ntr)
    G0: NDArray[np.float64]     # (nK, nel, nel)
    G1: NDArray[np.float64]     # (nK, nel, nel)
    LU0: NDArray[np.float64]    # (nK, nel, nel)
    piv0: NDArray[np.int32]     # (nK, nel)
    LU1: NDArray[np.float64]    # (nK, nel, nel)
    piv1: NDArray[np.int32]     # (nK, nel)
    X0: NDArray[np.float64]     # (nK, nel, nel) = D0^{-1} F0
    X1: NDArray[np.float64]     # (nK, nel, nel) = D1^{-1} F1
    LUS: NDArray[np.float64]    # (nK, nel, nel) LU factors of Schur matrix S
    pivS: NDArray[np.int32]     # (nK, nel)


@dataclass
class ElementSchwarzData:
    """
    Factorized weighted additive-Schwarz data for the HDG trace matrix.

    Each element contributes one overlapping patch containing the trace DOFs
    on its three edges.  The patch matrix is built during the normal element
    scatter and stored as an in-place LU factorization.

    Shapes
    ------
    nK : number of elements
    m  : 3*ntr, number of trace DOFs in one element patch
    """
    lu: NDArray[np.float64]      # (nK, m, m), LU factors of local patch matrices
    piv: NDArray[np.int32]       # (nK, m), LU pivots
    dofs: NDArray[np.int64]      # (nK, m), global trace DOFs for each patch
    weights: NDArray[np.float64] # (nK, m), overlap weights



def edge_patch_counts_from_sigma(sigma: NDArray[np.int64], n_edges: int) -> NDArray[np.int64]:
    """
    Count how many element patches contain each mesh edge.

    For a conforming triangular mesh this is usually 1 on boundary edges and
    2 on interior edges.  The count is used both for overlap weights and for
    distributing the global edge-diagonal contribution across element patches.
    """
    counts = np.zeros(n_edges, dtype=np.int64)
    for K in range(sigma.shape[0]):
        for e in range(3):
            counts[sigma[K, e]] += 1
    counts[counts == 0] = 1
    return counts


def allocate_element_schwarz_data(n_elements: int, ntr: int) -> ElementSchwarzData:
    """Allocate storage for one factorized Schwarz patch per element."""
    m = 3 * ntr
    return ElementSchwarzData(
        lu=np.empty((n_elements, m, m), dtype=np.float64),
        piv=np.empty((n_elements, m), dtype=np.int32),
        dofs=np.empty((n_elements, m), dtype=np.int64),
        weights=np.empty((n_elements, m), dtype=np.float64),
    )


@nb.njit(cache=True, parallel=True, fastmath=True)
def _element_schwarz_local_corrections(r, AS_lu, AS_piv, AS_dofs, AS_w, Yloc):
    """
    Parallel stage of weighted additive Schwarz.

    Each element patch independently solves

        A_K z_K = R_K r

    and stores the weighted local correction in Yloc[K,:].  This stage is race
    free because every thread writes only to its own element row in Yloc.
    """
    nK = AS_lu.shape[0]
    m = AS_lu.shape[1]

    for K in nb.prange(nK):
        rhs = np.empty((m, 1), dtype=np.float64)

        for a in range(m):
            rhs[a, 0] = r[AS_dofs[K, a]]

        lu_solve_inplace(AS_lu[K], AS_piv[K], rhs)

        for a in range(m):
            Yloc[K, a] = AS_w[K, a] * rhs[a, 0]


@nb.njit(cache=True)
def _element_schwarz_reduce(AS_dofs, Yloc, y):
    """
    Deterministic serial reduction of local Schwarz corrections.

    This is intentionally not parallel: neighboring element patches share edge
    trace DOFs, so direct parallel scatter would race.  The cost is O(nK*3*ntr),
    usually much cheaper than the local patch solves.
    """
    nK = AS_dofs.shape[0]
    m = AS_dofs.shape[1]

    for i in range(y.shape[0]):
        y[i] = 0.0

    for K in range(nK):
        for a in range(m):
            y[AS_dofs[K, a]] += Yloc[K, a]


class ElementSchwarzPreconditioner:
    """
    SciPy LinearOperator wrapper for weighted element additive Schwarz.

    The local patch solves are parallelized by Numba.  The final scatter/reduce
    is serial to avoid write races on shared edge trace DOFs.  The matvec returns
    a copy because SciPy Krylov methods may keep references to previous outputs.
    """

    def __init__(self, data: ElementSchwarzData, ndof: int):
        self.data = data
        self.ndof = ndof
        self.Yloc = np.empty((data.lu.shape[0], data.lu.shape[1]), dtype=np.float64)
        self.y = np.empty(ndof, dtype=np.float64)
        self.linear_operator = LinearOperator(
            shape=(ndof, ndof),
            matvec=self.matvec,
            dtype=np.float64,
        )

    def matvec(self, r):
        r = np.asarray(r, dtype=np.float64)
        _element_schwarz_local_corrections(
            r,
            self.data.lu,
            self.data.piv,
            self.data.dofs,
            self.data.weights,
            self.Yloc,
        )
        _element_schwarz_reduce(self.data.dofs, self.Yloc, self.y)
        return self.y.copy()


class DiffReaHDGSolver:
    """
    Reusable HDG Poisson / diffusion-reaction trace solver.

    This class is intended for workloads where the condensed global trace
    matrix stays fixed across multiple solves, while the source term changes.

    Cached data:
      - mesh and quadrature metadata
      - basis evaluation tables for the source space
      - boundary mask
      - condensed global matrix in COO triplet form
      - optional iterative-solver preconditioner

    Typical usage
    -------------
    solver = DiffReaHDGSolver(msh, quad_u, quad_f, tau_const=1.0)
    solver.build_global_matrix()

    uh1, qh1 = solver.solve(fcoef1)
    uh2, qh2 = solver.solve(fcoef2, reuse_prec=True)
    """

    def __init__(
            self,
            msh: Triangulation,
            quad_u: TriangleQuadratureData,
            quad_f: Optional[TriangleQuadratureData] = None,
            tau_const: float = 1.0,
            penalty: float = 1e20,
    ):
        self.msh = msh
        self.quad_u = quad_u
        self.quad_f = quad_u if quad_f is None else quad_f
        self.tau_const = tau_const
        self.penalty = penalty

        self.bd_mask = np.zeros(msh.num_edg, dtype=np.bool_)
        self.bd_mask[msh.bnd_edges_inds] = True
        self.local_cache: Optional[LocalSolverCache] = None

        # Basis tables for the source DG space.
        if self.quad_f is self.quad_u:
            self.psi_f = np.ascontiguousarray(self.quad_u.bas_of_quads.T)
            self.psi_f_on_uq = np.ascontiguousarray(self.quad_u.bas_of_quads.T)
        else:
            self.psi_f, self.psi_f_on_uq = build_rhs_dg_tables(self.quad_u, self.quad_f)

        self.nK = msh.num_tri
        self.nE = msh.num_edg
        self.ntr = quad_u.edg_dof
        self.ndofT = self.nE * self.ntr

        self.nnzK = 9 * self.ntr * self.ntr
        self.nnz = self.nK * self.nnzK + self.nE * (self.ntr * self.ntr)

        self.I: Optional[NDArray[np.int64]] = None
        self.J: Optional[NDArray[np.int64]] = None
        self.V: Optional[NDArray[np.float64]] = None

        # Cached global iterative preconditioner.
        #
        # Important:
        #   None means "no cached preconditioner currently available".
        #   It does not mean "solve without preconditioning".
        self.prec = None

        # Additive-Schwarz preconditioner data.  This is built during global
        # trace assembly, not by extracting blocks from the assembled sparse matrix.
        self.schwarz_data: Optional[ElementSchwarzData] = None
        self.schwarz_operator: Optional[ElementSchwarzPreconditioner] = None
        self.schwarz_prec = None

    # --------------------------------------------------------
    # Source normalization
    # --------------------------------------------------------
    def source_to_fcoef(self, src: Union[NDArray, Callable]) -> NDArray[np.float64]:
        """
        Normalize `src` to DG coefficients of shape (nK, nel_f).

        Accepted inputs:
          - ndarray already containing DG coefficients
          - Python callable src(x,y), projected onto V_f
        """
        if isinstance(src, np.ndarray):
            if src.ndim != 2:
                raise ValueError("Expected `src` as DG coefficients with shape (nK, nel_f).")
            if src.shape[0] != self.nK:
                raise ValueError(f"src.shape[0] = {src.shape[0]} but expected nK = {self.nK}.")
            if src.shape[1] != self.quad_f.el_dof:
                raise ValueError(
                    f"src.shape[1] = {src.shape[1]} but expected quad_f.el_dof = {self.quad_f.el_dof}."
                )
            return np.ascontiguousarray(src, dtype=np.float64)

        if callable(src):
            return dg_moments(src, self.quad_f, self.msh)

        raise TypeError("`src` must be either a DG coefficient array or a callable src(x,y).")

    # --------------------------------------------------------
    # Matrix / RHS allocation helpers
    # --------------------------------------------------------
    def _alloc_matrix_triplets(self):
        I = np.empty(self.nnz, dtype=np.int64)
        J = np.empty(self.nnz, dtype=np.int64)
        V = np.empty(self.nnz, dtype=np.float64)
        return I, J, V

    def _alloc_rhs_triplets(self):
        n_rhs_entries = self.nK * 3 * self.ntr + self.nE * self.ntr
        Ib = np.empty(n_rhs_entries, dtype=np.int64)
        Vb = np.empty(n_rhs_entries, dtype=np.float64)
        return Ib, Vb

    # --------------------------------------------------------
    # Fixed operator setup
    # --------------------------------------------------------
    def build_global_matrix(
            self,
            force: bool = False,
            verbose: bool = True,
            build_schwarz: bool = False,
            schwarz_reg: float = 0.0,
    ):
        """
        Assemble and cache the condensed global trace matrix.

        If build_schwarz=True, the element additive-Schwarz patch factors are
        built on the fly inside the element scatter.  Because the Schwarz patch
        is constructed during assembly, requesting it after a matrix-only build
        requires a rebuild of the trace matrix.
        """
        matrix_ready = (self.I is not None) and (self.J is not None) and (self.V is not None)
        schwarz_ready = (self.schwarz_prec is not None)

        if matrix_ready and (not force) and ((not build_schwarz) or schwarz_ready):
            return

        zero_fcoef = np.zeros((self.nK, self.quad_f.el_dof), dtype=np.float64)

        I, J, V = self._alloc_matrix_triplets()
        Ib, Vb = self._alloc_rhs_triplets()

        schwarz_data = None
        if build_schwarz:
            schwarz_data = allocate_element_schwarz_data(self.nK, self.ntr)

        t0 = time.time()
        if verbose:
            msg = "assembling cached global matrix"
            if build_schwarz:
                msg += " + element-Schwarz preconditioner"
            print(msg + " ...", end=" ", flush=True)

        assemble_trace_system_from_fcoef(
            self.msh, self.quad_u,
            self.bd_mask,
            self.tau_const, self.penalty,
            self.psi_f_on_uq, zero_fcoef,
            I, J, V, Ib, Vb,
            schwarz_data=schwarz_data,
            schwarz_reg=schwarz_reg,
        )

        self.I, self.J, self.V = I, J, V

        # The global operator changed, so any cached preconditioner is stale.
        self.prec = None
        self.schwarz_data = schwarz_data
        self.schwarz_operator = None
        self.schwarz_prec = None

        if schwarz_data is not None:
            self.schwarz_operator = ElementSchwarzPreconditioner(schwarz_data, self.ndofT)
            self.schwarz_prec = self.schwarz_operator.linear_operator

        if verbose:
            print(f"done in {time.time() - t0:.2f}s")

    # --------------------------------------------------------
    # Preconditioner policy
    # --------------------------------------------------------
    def _select_preconditioner(
            self,
            itr_solv: Optional[str],
            reuse_prec: bool,
            prec_type: Literal["ilu", "element_schwarz", "schwarz", "none"] = "ilu",
    ):
        """
        Select the preconditioner passed to solve_global_system.

        Choices
        -------
        prec_type="ilu"
            Use cached ILU if available and requested; otherwise ask
            solve_global_system to build a fresh ILU.
        prec_type="element_schwarz" or "schwarz"
            Use the on-the-fly element additive-Schwarz LinearOperator.
        prec_type="none"
            No preconditioner.
        """
        if itr_solv is None or itr_solv == "direct":
            return None

        if prec_type == "none":
            return None

        if prec_type == "element_schwarz" or prec_type == "schwarz":
            if self.schwarz_prec is None:
                # The Schwarz blocks can only be built during the element scatter.
                self.build_global_matrix(force=True, verbose=False, build_schwarz=True)
            return self.schwarz_prec

        if reuse_prec and self.prec is not None:
            return self.prec

        return "ilu"

    # --------------------------------------------------------
    # Solve / reconstruct
    # --------------------------------------------------------
    def solve_trace(
            self,
            src: Union[NDArray, Callable],
            itr_solv: str = "CGS",
            reuse_prec: bool = True,
            update_prec: bool = True,
            ini_guess: Optional[np.ndarray] = None,
            return_resd: bool = False,
            verbose: bool = True,
            rtol: float = 1e-13,
            ilu_drop_tol: float = 1e-10,
            ilu_fill_factor: float = 35,
            raise_on_nonconvergence: bool = False,
            prec_type: Literal["ilu", "element_schwarz", "schwarz", "none"] = "ilu",
            schwarz_reg: float = 0.0,
    ):
        """
        Solve only the condensed trace system and return `uhat`.

        Parameters
        ----------
        src
            Either DG coefficients `(nK, nel_f)` or a callable `src(x,y)`.
        itr_solv
            Iterative solver name. Use None or "direct" for a direct sparse solve.
        reuse_prec
            Reuse the cached preconditioner if available.
        update_prec
            Store the preconditioner returned by `solve_global_system`.
        ini_guess
            Initial guess for iterative solver.
        return_resd
            Also return the residual norm.
        verbose
            Print timing information.
        rtol
            Relative tolerance passed to the iterative solver.
        ilu_drop_tol
            Drop tolerance used when building a fresh ILU preconditioner.
        ilu_fill_factor
            Fill factor used when building a fresh ILU preconditioner.
        raise_on_nonconvergence
            If True, raise RuntimeError when the iterative solver reports nonzero info.
        prec_type
            "ilu", "element_schwarz"/"schwarz", or "none".
        schwarz_reg
            Optional diagonal regularization added to each local Schwarz block before LU.
        """
        use_schwarz = prec_type == "element_schwarz" or prec_type == "schwarz"
        self.build_global_matrix(verbose=verbose, build_schwarz=use_schwarz, schwarz_reg=schwarz_reg)

        f_coef = self.source_to_fcoef(src)

        t0 = time.time()
        rhs = self.assemble_rhs_from_fcoef(f_coef)
        rhs_time = time.time() - t0

        preconditioner = self._select_preconditioner(
            itr_solv=itr_solv,
            reuse_prec=reuse_prec,
            prec_type=prec_type,
        )

        t1 = time.time()

        solve_kwargs = dict(
            row_indices=self.I,
            col_indices=self.J,
            matrix_values=self.V,
            rhs=rhs,
            system_size=self.ndofT,
            solver="direct" if itr_solv is None else itr_solv,
            preconditioner=preconditioner,
            initial_guess=ini_guess,
            rtol=rtol,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            raise_on_nonconvergence=raise_on_nonconvergence,
            verbose=verbose,
        )

        # The element-Schwarz operator is built for the unscaled trace matrix.
        # If your global_system.solve_global_system supports scale_system=False,
        # use it.  If not, we fall back while emitting a warning because using
        # this preconditioner on D^{-1}A is only an approximation to the intended
        # operator.
        if use_schwarz:
            solve_kwargs["scale_system"] = False

        try:
            result = solve_global_system(**solve_kwargs)
        except TypeError as exc:
            if use_schwarz and "scale_system" in solve_kwargs:
                solve_kwargs.pop("scale_system")
                if verbose:
                    print(
                        "warning: solve_global_system does not accept scale_system=False; "
                        "using element-Schwarz with the solver's default scaling.",
                        flush=True,
                    )
                result = solve_global_system(**solve_kwargs)
            else:
                raise exc

        solve_time = time.time() - t1

        uhat = result.x

        if update_prec and (not use_schwarz) and result.preconditioner is not None:
            self.prec = result.preconditioner

        if verbose:
            total_elapsed = getattr(result, "total_elapsed_seconds", solve_time)
            scale_elapsed = getattr(result, "scale_elapsed_seconds", None)
            prec_elapsed = getattr(result, "preconditioner_elapsed_seconds", None)
            krylov_elapsed = getattr(result, "solve_elapsed_seconds", None)

            print(f" rhs assembly : {rhs_time:.2f}s")
            print(f" trace total  : {total_elapsed:.2f}s")

            if scale_elapsed is not None:
                print(f"   scaling    : {scale_elapsed:.2f}s")

            if prec_elapsed is not None:
                print(f"   prec build : {prec_elapsed:.2f}s")

            if krylov_elapsed is not None:
                print(f"   krylov     : {krylov_elapsed:.2f}s")

            if result.info == 0:
                print(f" residual     : {result.residual_norm:.3e}")
            elif result.info is not None:
                print(f" solver info  : {result.info}")
                print(f" residual     : {result.residual_norm:.3e}")

        if return_resd:
            return uhat, f_coef, result.residual_norm

        return uhat, f_coef

    def recons_interior_vars(
            self,
            uhat: NDArray[np.float64],
            f_coef: NDArray[np.float64],
    ):
        """
        Reconstruct the element unknowns `(u_h, q_h)` from the global trace
        solution and source coefficients.
        """
        return reconstruct_uq_from_fcoef(
            self.msh, self.quad_u,
            self.bd_mask,
            self.tau_const,
            uhat, self.psi_f_on_uq, f_coef,
        )

    def solve(
            self,
            src: Union[NDArray, Callable],
            itr_solv: str = "BICGSTAB",
            reuse_prec: bool = True,
            update_prec: bool = True,
            return_trace: bool = False,
            return_resd: bool = False,
            ini_guess: NDArray = None,
            verbose: bool = True,
            rtol: float = 1e-13,
            ilu_drop_tol: float = 1e-10,
            ilu_fill_factor: float = 35,
            raise_on_nonconvergence: bool = False,
            prec_type: Literal["ilu", "element_schwarz", "schwarz", "none"] = "ilu",
            schwarz_reg: float = 0.0,
    ):
        """
        High-level solve: assemble RHS, solve trace system, reconstruct `(u_h, q_h)`.

        Returns
        -------
        By default:
            uh, qh

        If `return_trace=True`:
            uh, qh, uhat

        If `return_resd=True`:
            the residual norm is appended at the end.
        """
        solve_out = self.solve_trace(
            src,
            itr_solv=itr_solv,
            reuse_prec=reuse_prec,
            update_prec=update_prec,
            ini_guess=ini_guess,
            return_resd=return_resd,
            verbose=verbose,
            rtol=rtol,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            raise_on_nonconvergence=raise_on_nonconvergence,
            prec_type=prec_type,
            schwarz_reg=schwarz_reg,
        )

        if return_resd:
            uhat, f_coef, resd = solve_out
        else:
            uhat, f_coef = solve_out
            resd = None

        t0 = time.time()
        if verbose:
            print("recovering uh ... ", end="", flush=True)

        uh, qh = self.recons_interior_vars(uhat, f_coef)

        if verbose:
            print(f"done in {time.time() - t0:.2f}s")

        ret = [uh, qh]

        if return_trace:
            ret.append(uhat)

        if return_resd:
            ret.append(resd)

        return tuple(ret)

    def build_local_solvers(self, force: bool = False, verbose: bool = True):
        """
        Build and cache the local Schur solver data for all elements.

        This is useful for repeated solves when the local operator is fixed.
        """
        if (self.local_cache is not None) and (not force):
            return

        t0 = time.time()
        if verbose:
            print("building cached local solvers ...", end=" ", flush=True)

        self.local_cache = build_local_solver_cache(
            self.msh, self.quad_u, self.bd_mask, self.tau_const,
            self.quad_f, self.psi_f_on_uq,
        )

        if verbose:
            print(f"done in {time.time() - t0:.2f}s")

    def clear_local_solvers(self):
        """Drop the cached local solver data."""
        self.local_cache = None

    def clear_preconditioner(self):
        """Drop cached ILU and element-Schwarz preconditioners."""
        self.prec = None
        self.schwarz_data = None
        self.schwarz_operator = None
        self.schwarz_prec = None

    def assemble_rhs_from_fcoef(self, f_coef: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Assemble the condensed rhs for the provided DG source coefficients.

        Uses cached local solver data if available.
        """
        if self.local_cache is not None:
            return assemble_trace_rhs_from_fcoef_cached(
                self.msh, self.quad_u,
                self.bd_mask,
                self.tau_const, self.penalty,
                self.psi_f_on_uq, f_coef,
                self.local_cache,
            )

        Ib, Vb = self._alloc_rhs_triplets()

        assemble_trace_rhs_from_fcoef(
            self.msh, self.quad_u,
            self.bd_mask,
            self.tau_const, self.penalty,
            self.psi_f_on_uq, f_coef,
            Ib, Vb,
        )

        return coo_rhs_to_dense(Ib, Vb, self.ndofT)

    def reconstruct_from_fcoef(
            self,
            uhat: NDArray[np.float64],
            f_coef: NDArray[np.float64],
    ):
        """
        Reconstruct the element unknowns from the trace solution and DG source coefficients.

        Uses cached local solver data if available.
        """
        if self.local_cache is not None:
            return reconstruct_uq_from_fcoef_cached(
                self.msh, self.quad_u,
                self.bd_mask,
                self.tau_const,
                uhat, self.psi_f_on_uq, f_coef,
                self.local_cache,
            )

        return reconstruct_uq_from_fcoef(
            self.msh, self.quad_u,
            self.bd_mask,
            self.tau_const,
            uhat, self.psi_f_on_uq, f_coef,
        )

@nb.njit(cache=True, inline="always")
def _assemble_local_rhs_only_poisson_fcoef(
        x0, y0, x1, y1, x2, y2,
        geom5,
        xi_eta_u, wq_u, phi_u,
        psi_f_on_uq,
        fcoefK,
        rhs_loc
):
    """
    Assemble only the source-dependent local rhs.

    For the present formulation, the source contributes only to the u-block:
        rhs_loc = [rhs_u, 0, 0].
    """
    absdetJ = geom5[0]

    ex1 = x1 - x0
    ey1 = y1 - y0
    ex2 = x2 - x0
    ey2 = y2 - y0

    nq = wq_u.shape[0]
    nel = phi_u.shape[1]
    nel_f = fcoefK.shape[0]

    _zero_vec(rhs_loc)

    for q in range(nq):
        xi = xi_eta_u[q, 0]
        eta = xi_eta_u[q, 1]
        x, y = map_ref_to_phys_m11(x0, y0, ex1, ey1, ex2, ey2, xi, eta)

        w = wq_u[q] * absdetJ

        fhq = 0.0
        for j in range(nel_f):
            fhq += fcoefK[j] * psi_f_on_uq[q, j]

        for i in range(nel):
            rhs_loc[i] += w * fhq * phi_u[q, i]


@nb.njit(cache=True, inline="always")
def _build_local_solver_cache_from_blocks(
        H, G0, G1, F0, F1, D0, D1,
        Mhat_loc,
        LU0, piv0, LU1, piv1,
        X0, X1,
        S, LUS, pivS
):
    """
    Build the fixed local cached solver data from local operator blocks.

    Outputs:
      - LU factors of D0 and D1
      - X0 = D0^{-1} F0
      - X1 = D1^{-1} F1
      - LU factors of Schur matrix S = H + G0 X0 + G1 X1
    """
    nel = H.shape[0]

    for i in range(nel):
        for j in range(nel):
            LU0[i, j] = D0[i, j]
            LU1[i, j] = D1[i, j]
    lu_factor_inplace(LU0, piv0)
    lu_factor_inplace(LU1, piv1)

    for i in range(nel):
        for j in range(nel):
            X0[i, j] = F0[i, j]
            X1[i, j] = F1[i, j]
    lu_solve_inplace(LU0, piv0, X0)
    lu_solve_inplace(LU1, piv1, X1)

    for i in range(nel):
        for j in range(nel):
            s = H[i, j]
            t = 0.0
            for k in range(nel):
                t += G0[i, k] * X0[k, j]
            s += t
            t = 0.0
            for k in range(nel):
                t += G1[i, k] * X1[k, j]
            s += t
            S[i, j] = s

    for i in range(nel):
        for j in range(nel):
            LUS[i, j] = S[i, j]
    lu_factor_inplace(LUS, pivS)

@nb.njit(cache=True, inline="always")
def _apply_cached_local_solver(
        Mhat_loc,
        G0, G1,
        LU0, piv0,
        LU1, piv1,
        X0, X1,
        LUS, pivS,
        rhs_loc,
        uhat_loc,
        r_loc,
        y0, y1,
        red_rhs,
        u, q0, q1
):
    """
    Apply cached local solver data to the rhs and local trace.

    This routine is shared by:
      - condensed rhs assembly   (use uhat_loc = 0)
      - local reconstruction     (use gathered local trace)

    It computes:
      r_loc   = rhs_loc + Mhat_loc @ uhat_loc
      y0      = D0^{-1} r_q0
      y1      = D1^{-1} r_q1
      red_rhs = r_u + G0 y0 + G1 y1
      u       = S^{-1} red_rhs
      q0      = X0 u - y0
      q1      = X1 u - y1
    """
    nel = G0.shape[0]
    m = uhat_loc.shape[0]

    # r_loc = rhs_loc + Mhat_loc @ uhat_loc
    for i in range(3 * nel):
        s = rhs_loc[i]
        for j in range(m):
            s += Mhat_loc[i, j] * uhat_loc[j]
        r_loc[i] = s

    # y0 = D0^{-1} r_q0 ; y1 = D1^{-1} r_q1
    for i in range(nel):
        y0[i] = r_loc[nel + i]
        y1[i] = r_loc[2 * nel + i]

    tmp0 = y0.reshape(nel, 1)
    tmp1 = y1.reshape(nel, 1)
    lu_solve_inplace(LU0, piv0, tmp0)
    lu_solve_inplace(LU1, piv1, tmp1)

    # red_rhs = r_u + G0 y0 + G1 y1
    for i in range(nel):
        s = r_loc[i]
        t = 0.0
        for k in range(nel):
            t += G0[i, k] * y0[k]
        s += t
        t = 0.0
        for k in range(nel):
            t += G1[i, k] * y1[k]
        s += t
        red_rhs[i] = s

    # u = S^{-1} red_rhs
    tmpu = red_rhs.reshape(nel, 1)
    lu_solve_inplace(LUS, pivS, tmpu)
    for i in range(nel):
        u[i] = red_rhs[i]

    # q0 = X0 u - y0 ; q1 = X1 u - y1
    for i in range(nel):
        s0 = 0.0
        s1 = 0.0
        for k in range(nel):
            uk = u[k]
            s0 += X0[i, k] * uk
            s1 += X1[i, k] * uk
        q0[i] = s0 - y0[i]
        q1[i] = s1 - y1[i]

def build_local_solver_cache(msh, quad_u, bd_mask, tau_const, quad_f, psi_f_on_uq):
    """
    Build and return cached local solver data for all elements.

    This uses the existing validated local block assembly with a zero source,
    since the fixed local operator does not depend on the source term.
    """
    nK = msh.num_tri
    nel = quad_u.el_dof
    ntr = quad_u.edg_dof
    m = 3 * ntr

    Mhat = np.empty((nK, 3 * nel, m), dtype=np.float64)
    G0all = np.empty((nK, nel, nel), dtype=np.float64)
    G1all = np.empty((nK, nel, nel), dtype=np.float64)

    LU0all = np.empty((nK, nel, nel), dtype=np.float64)
    piv0all = np.empty((nK, nel), dtype=np.int32)
    LU1all = np.empty((nK, nel, nel), dtype=np.float64)
    piv1all = np.empty((nK, nel), dtype=np.int32)

    X0all = np.empty((nK, nel, nel), dtype=np.float64)
    X1all = np.empty((nK, nel, nel), dtype=np.float64)

    LUSall = np.empty((nK, nel, nel), dtype=np.float64)
    pivSall = np.empty((nK, nel), dtype=np.int32)

    phi_u = np.ascontiguousarray(quad_u.bas_of_quads.T)
    gphi_u = np.ascontiguousarray(quad_u.dbas_of_quads.swapaxes(0, 2))
    phi_f_bd = np.ascontiguousarray(quad_u.bas_of_bd_quads.swapaxes(1, 2))
    mu = np.ascontiguousarray(quad_u.bas1d_of_ref_edg_qds.T)

    zero_fcoefK = np.zeros(quad_f.el_dof, dtype=np.float64)

    for K in range(nK):
        geom5 = np.empty(5, dtype=np.float64)
        edge3x3 = np.empty((3, 3), dtype=np.float64)
        x0, y0, x1, y1, x2, y2 = _gather_geom(msh.node_coords, msh.triangles, K, geom5, edge3x3)

        H = np.empty((nel, nel), dtype=np.float64)
        G0 = np.empty((nel, nel), dtype=np.float64)
        G1 = np.empty((nel, nel), dtype=np.float64)
        F0 = np.empty((nel, nel), dtype=np.float64)
        F1 = np.empty((nel, nel), dtype=np.float64)
        D0 = np.empty((nel, nel), dtype=np.float64)
        D1 = np.empty((nel, nel), dtype=np.float64)
        Mhat_loc = np.empty((3 * nel, m), dtype=np.float64)
        rhs_loc = np.empty(3 * nel, dtype=np.float64)

        _assemble_local_blocks_hat_rhs_poisson_fcoef(
            x0, y0, x1, y1, x2, y2,
            geom5, edge3x3,
            tau_const,
            quad_u.Krf_quads, quad_u.Krf_w, phi_u, gphi_u,
            psi_f_on_uq, zero_fcoefK,
            quad_u.quads_JGL, quad_u.weights_JGL, phi_f_bd, mu,
            H, G0, G1, F0, F1, D0, D1,
            Mhat_loc, rhs_loc
        )

        LU0 = np.empty((nel, nel), dtype=np.float64)
        piv0 = np.empty(nel, dtype=np.int32)
        LU1 = np.empty((nel, nel), dtype=np.float64)
        piv1 = np.empty(nel, dtype=np.int32)
        X0 = np.empty((nel, nel), dtype=np.float64)
        X1 = np.empty((nel, nel), dtype=np.float64)
        S = np.empty((nel, nel), dtype=np.float64)
        LUS = np.empty((nel, nel), dtype=np.float64)
        pivS = np.empty(nel, dtype=np.int32)

        _build_local_solver_cache_from_blocks(
            H, G0, G1, F0, F1, D0, D1,
            Mhat_loc,
            LU0, piv0, LU1, piv1,
            X0, X1,
            S, LUS, pivS
        )

        Mhat[K] = Mhat_loc
        G0all[K] = G0
        G1all[K] = G1
        LU0all[K] = LU0
        piv0all[K] = piv0
        LU1all[K] = LU1
        piv1all[K] = piv1
        X0all[K] = X0
        X1all[K] = X1
        LUSall[K] = LUS
        pivSall[K] = pivS

    return LocalSolverCache(
        Mhat=Mhat,
        G0=G0all,
        G1=G1all,
        LU0=LU0all,
        piv0=piv0all,
        LU1=LU1all,
        piv1=piv1all,
        X0=X0all,
        X1=X1all,
        LUS=LUSall,
        pivS=pivSall,
    )

def assemble_trace_rhs_from_fcoef_cached(
        msh, quad_u,
        bd_mask,
        tau_const, penalty,
        psi_f_on_uq, f_coef,
        local_cache: LocalSolverCache
):
    """
    Assemble condensed global rhs using cached local solvers.

    This avoids rebuilding and refactorizing the local operator on each solve.
    """
    nK = msh.num_tri
    nE = msh.num_edg
    nel = quad_u.el_dof
    ntr = quad_u.edg_dof
    m = 3 * ntr

    Ib = np.empty(nK * 3 * ntr + nE * ntr, dtype=np.int64)
    Vb = np.empty(nK * 3 * ntr + nE * ntr, dtype=np.float64)

    phi_u = np.ascontiguousarray(quad_u.bas_of_quads.T)

    base_rhs_elem = 0
    base_rhs_bc = nK * 3 * ntr

    zero_uhat = np.zeros(m, dtype=np.float64)

    for K in range(nK):
        geom5 = np.empty(5, dtype=np.float64)
        edge3x3 = np.empty((3, 3), dtype=np.float64)
        x0, y0, x1, y1, x2, y2 = _gather_geom(msh.node_coords, msh.triangles, K, geom5, edge3x3)

        rhs_loc = np.empty(3 * nel, dtype=np.float64)
        _assemble_local_rhs_only_poisson_fcoef(
            x0, y0, x1, y1, x2, y2,
            geom5,
            quad_u.Krf_quads, quad_u.Krf_w, phi_u,
            psi_f_on_uq,
            f_coef[K],
            rhs_loc
        )

        r_loc = np.empty(3 * nel, dtype=np.float64)
        y0v = np.empty(nel, dtype=np.float64)
        y1v = np.empty(nel, dtype=np.float64)
        red_rhs = np.empty(nel, dtype=np.float64)
        u = np.empty(nel, dtype=np.float64)
        q0 = np.empty(nel, dtype=np.float64)
        q1 = np.empty(nel, dtype=np.float64)

        _apply_cached_local_solver(
            local_cache.Mhat[K],
            local_cache.G0[K], local_cache.G1[K],
            local_cache.LU0[K], local_cache.piv0[K],
            local_cache.LU1[K], local_cache.piv1[K],
            local_cache.X0[K], local_cache.X1[K],
            local_cache.LUS[K], local_cache.pivS[K],
            rhs_loc,
            zero_uhat,
            r_loc,
            y0v, y1v,
            red_rhs,
            u, q0, q1
        )

        _scatter_element_rhs_poisson(
            K, msh.sigma, msh.orientations,
            edge3x3, quad_u.MKrfe_lst_p, tau_const,
            u, q0, q1,
            Ib, Vb,
            base_rhs_elem
        )

    for Gamma in range(nE):
        _diag_bc_rhs_poisson(
            Gamma,
            msh.node_coords, msh.edges, bd_mask,
            penalty,
            quad_u.rf_edg_lag_nodes,
            Ib, Vb,
            base_rhs_bc,
            ntr
        )

    return coo_rhs_to_dense(Ib, Vb, msh.num_edg * quad_u.edg_dof)

def reconstruct_uq_from_fcoef_cached(
        msh, quad_u,
        bd_mask,
        tau_const,
        uhat,
        psi_f_on_uq, f_coef,
        local_cache: LocalSolverCache
):
    """
    Reconstruct local fields using cached local solvers.
    """
    nK = msh.num_tri
    nel = quad_u.el_dof
    ntr = quad_u.edg_dof
    m = 3 * ntr

    uhat = uhat.copy()
    apply_bd_to_uhat_inplace(uhat, msh.node_coords, msh.edges, bd_mask, quad_u.rf_edg_lag_nodes, ntr)

    uK = np.empty((nK, nel), dtype=np.float64)
    qK = np.empty((nK, nel, 2), dtype=np.float64)

    phi_u = np.ascontiguousarray(quad_u.bas_of_quads.T)

    for K in range(nK):
        geom5 = np.empty(5, dtype=np.float64)
        edge3x3 = np.empty((3, 3), dtype=np.float64)
        x0, y0, x1, y1, x2, y2 = _gather_geom(msh.node_coords, msh.triangles, K, geom5, edge3x3)

        rhs_loc = np.empty(3 * nel, dtype=np.float64)
        _assemble_local_rhs_only_poisson_fcoef(
            x0, y0, x1, y1, x2, y2,
            geom5,
            quad_u.Krf_quads, quad_u.Krf_w, phi_u,
            psi_f_on_uq,
            f_coef[K],
            rhs_loc
        )

        uhat_loc = np.empty(m, dtype=np.float64)
        _gather_uhat_local(K, msh.sigma, msh.orientations, uhat, ntr, uhat_loc)

        r_loc = np.empty(3 * nel, dtype=np.float64)
        y0v = np.empty(nel, dtype=np.float64)
        y1v = np.empty(nel, dtype=np.float64)
        red_rhs = np.empty(nel, dtype=np.float64)
        u = np.empty(nel, dtype=np.float64)
        q0 = np.empty(nel, dtype=np.float64)
        q1 = np.empty(nel, dtype=np.float64)

        _apply_cached_local_solver(
            local_cache.Mhat[K],
            local_cache.G0[K], local_cache.G1[K],
            local_cache.LU0[K], local_cache.piv0[K],
            local_cache.LU1[K], local_cache.piv1[K],
            local_cache.X0[K], local_cache.X1[K],
            local_cache.LUS[K], local_cache.pivS[K],
            rhs_loc,
            uhat_loc,
            r_loc,
            y0v, y1v,
            red_rhs,
            u, q0, q1
        )

        uK[K, :] = u
        qK[K, :, 0] = q0
        qK[K, :, 1] = q1

    return uK, qK

# ============================================================
# 1) Projection driver: f(x,y) -> f_coef (nK, nel_f)
# ============================================================
@nb.njit(cache=True, parallel=True, fastmath=True)
def project_src_func_to_dg_coeffs(
        nodes, tris,
        xi_eta_f, wq_f, psi_f,  # psi_f: (nq_f, nel_f)
        MKrf_inv_f,  # (nel_f, nel_f)  = TriangleQuadratureData(p_f).MKrf_inv
        f_coef  # (nK, nel_f) output
):
    """
    L2-project f_eval(x,y) onto DG space V_f on each element.
    Solves on reference: MKrf_ref * f_coef = b_ref, using MKrf_inv_f.

    b_ref[i] = Σ_q wq_f[q] * f(x_q^K) * ψ_i(ξ_q)
    f_coef[K,:] = MKrf_inv_f @ b_ref
    """
    nK = tris.shape[0]
    nq = wq_f.shape[0]
    nel_f = psi_f.shape[1]

    for K in nb.prange(nK): # TODO: can be included in the assembly loop?
        geom5 = np.empty(5, dtype=np.float64)
        edge3 = np.empty((3, 3), dtype=np.float64)  # unused but required by _gather_geom
        x0, y0, x1, y1, x2, y2 = _gather_geom(nodes, tris, K, geom5, edge3)

        ex1 = x1 - x0; ey1 = y1 - y0
        ex2 = x2 - x0; ey2 = y2 - y0

        b_ref = np.zeros(nel_f, dtype=np.float64)
        for q in range(nq):
            xi = xi_eta_f[q, 0]
            eta = xi_eta_f[q, 1]
            x, y = map_ref_to_phys_m11(x0, y0, ex1, ey1, ex2, ey2, xi, eta)
            fq = f_eval(x, y)
            w = wq_f[q]
            for j in range(nel_f):
                b_ref[j] += w * fq * psi_f[q, j]

        for i in range(nel_f):
            s = 0.0
            for j in range(nel_f):
                s += MKrf_inv_f[i, j] * b_ref[j]
            f_coef[K, i] = s


# ============================================================
# 2) Local assembly (blocks + hatM + rhs) with DG source f_coef
# ============================================================
@nb.njit(cache=True, inline="always")
def _assemble_local_blocks_hat_rhs_poisson_fcoef(
        x0, y0, x1, y1, x2, y2,
        geom5, edge3x3,
        tau_const,
        # u-space quad/basis (degree p_u)
        xi_eta_u, wq_u, phi_u, gphi_u,
        # f-space basis evaluated at u-space quad points (bridge)
        psi_f_on_uq,  # (nq_u, nel_f)
        fcoefK,  # (nel_f,)
        # edge quad/basis (from u-space TriangleQuadratureData(p_u))
        sq, wf, phi_f, mu,
        # outputs (nel_u x nel_u blocks)
        H, G0, G1, F0, F1, D0, D1,
        # outputs
        Mhat_loc,  # (3*nel_u, 3*ntr)
        rhs_loc  # (3*nel_u,)
):
    """
    Same as _assemble_local_blocks_hat_rhs_poisson, except:
      - volume load uses DG field f_h given by fcoefK in space V_f,
      - and we evaluate f_h at u-quadrature points using psi_f_on_uq.

    rhs_u[i] = ∫_K f_h * φ_i dx
             ~ absdetJ * Σ_q wq_u[q] * f_h(x_q^K) * φ_i(ξ_q)
    with f_h(x_q^K) = Σ_j fcoefK[j] * ψ_j(ξ_q), where ψ_j evaluated at u-quads.
    """
    absdetJ = geom5[0]
    iJT00 = geom5[1]; iJT01 = geom5[2]
    iJT10 = geom5[3]; iJT11 = geom5[4]

    ex1 = x1 - x0; ey1 = y1 - y0
    ex2 = x2 - x0; ey2 = y2 - y0

    nq = wq_u.shape[0]
    nqf = wf.shape[0]
    nel = phi_u.shape[1]
    ntr = mu.shape[1]
    nel_f = fcoefK.shape[0]

    # zero
    _zero_mat(H); _zero_mat(G0); _zero_mat(G1); _zero_mat(F0); _zero_mat(F1); _zero_mat(D0); _zero_mat(D1)
    _zero_mat(Mhat_loc)
    _zero_vec(rhs_loc)

    # -------------------------
    # Volume: build rhs_u from f_h (different DG space)
    # -------------------------
    for q in range(nq):
        xi = xi_eta_u[q, 0]
        eta = xi_eta_u[q, 1]
        x, y = map_ref_to_phys_m11(x0, y0, ex1, ey1, ex2, ey2, xi, eta)

        rq = r_eval(x, y)

        d00, d11 = d_eval_diag(x, y)
        dinv0 = 1.0 / d00
        dinv1 = 1.0 / d11

        w_r = wq_u[q] * absdetJ * rq
        w = wq_u[q] * absdetJ

        # evaluate f_h at this quad point using fcoefK and psi_f_on_uq
        fhq = 0.0
        for j in range(nel_f):
            fhq += fcoefK[j] * psi_f_on_uq[q, j]

        # rhs: f only in u block
        for i in range(nel):
            rhs_loc[i] += w * fhq * phi_u[q, i]

        # assemble element blocks (same as your validated convention)
        for i in range(nel):
            gri0 = gphi_u[q, i, 0]
            gri1 = gphi_u[q, i, 1]
            dphi_i_dx = iJT00 * gri0 + iJT01 * gri1
            dphi_i_dy = iJT10 * gri0 + iJT11 * gri1

            phi_i = phi_u[q, i]

            for a in range(nel):
                phi_a = phi_u[q, a]

                # H reaction
                H[i, a] += w_r * phi_a * phi_i

                # D0, D1 positive (minus applied by block structure)
                D0[i, a] += w * dinv0 * phi_a * phi_i
                D1[i, a] += w * dinv1 * phi_a * phi_i

                # (A0^T)_{i,a} = int phi_a phi_i ; (A1^T) similarly
                val0 = w * phi_a * dphi_i_dx
                val1 = w * phi_a * dphi_i_dy

                F0[i, a] += val0
                F1[i, a] += val1

                G0[i, a] += -val0
                G1[i, a] += -val1

    # -------------------------
    # Boundary (same as your working code)
    # -------------------------
    for e in range(3):
        nx = edge3x3[e, 0]
        ny = edge3x3[e, 1]
        Jsc = edge3x3[e, 2]
        col0 = e * ntr

        for qf in range(nqf):
            tau = tau_const
            ws = wf[qf] * Jsc

            wtau = ws * tau
            wn0 = ws * nx
            wn1 = ws * ny

            # H += M_tau^{∂K}
            for i in range(nel):
                phi_i = phi_f[e, qf, i]
                for a in range(nel):
                    H[i, a] += wtau * phi_f[e, qf, a] * phi_i

            # Gℓ += M_{nℓ}^{∂K}
            for i in range(nel):
                phi_i = phi_f[e, qf, i]
                for a in range(nel):
                    phi_a = phi_f[e, qf, a]
                    G0[i, a] += wn0 * phi_a * phi_i
                    G1[i, a] += wn1 * phi_a * phi_i

            # Mhat_loc
            for i in range(nel):
                phi_i = phi_f[e, qf, i]
                for j in range(ntr):
                    muj = mu[qf, j]
                    Mhat_loc[i, col0 + j] += wtau * muj * phi_i
                    Mhat_loc[nel + i, col0 + j] += wn0 * muj * phi_i
                    Mhat_loc[2 * nel + i, col0 + j] += wn1 * muj * phi_i

@nb.njit(cache=True, inline="always")
def _scatter_element_rhs_poisson(
        K, sigma, orien,
        edge3x3, Med_ref, tau_const,
        u, q0, q1,
        Ib, Vb,
        base_rhs_elem
):
    """
    Scatter only the element contribution to the condensed global RHS.

    This is the rhs-only analogue of `_scatter_element_blocks_poisson`.
    It is used when the global matrix is already cached and only the
    source term changes between solves.
    """
    ntr = Med_ref.shape[2]
    nel = Med_ref.shape[1]

    g = np.zeros(ntr, dtype=np.float64)

    for er in range(3):
        Jedge = edge3x3[er, 2]
        nx = edge3x3[er, 0]
        ny = edge3x3[er, 1]

        Gamma_row = sigma[K, er]
        ispos_row = orien[K, er]

        for j in range(ntr):
            g[j] = 0.0

        for a in range(nel):
            comb_y = tau_const * u[a] + nx * q0[a] + ny * q1[a]
            for jloc in range(ntr):
                wj = Jedge * Med_ref[er, a, jloc]
                g[jloc] += wj * comb_y

        qb0 = base_rhs_elem + (K * 3 + er) * ntr
        for jloc in range(ntr):
            jg = map_edge_dof_bool(ispos_row, jloc, ntr)
            row = Gamma_row * ntr + jg
            Ib[qb0 + jloc] = row
            Vb[qb0 + jloc] = g[jloc]


@nb.njit(cache=True, inline="always")
def _diag_bc_rhs_poisson(
        Gamma,
        nodes, edge_nodes, bd_mask,
        penalty,
        rg_edg_lag_nodes,
        Ib, Vb,
        base_rhs_bc,
        ntr
):
    """
    Scatter only the boundary-condition contribution to the condensed RHS.

    Interior edges contribute zero here. Boundary edges contribute the
    large-penalty Dirichlet term exactly as in `_diag_block_and_bc_poisson`.
    """
    a = edge_nodes[Gamma, 0]
    bnode = edge_nodes[Gamma, 1]
    xa = nodes[a, 0]; ya = nodes[a, 1]
    xb = nodes[bnode, 0]; yb = nodes[bnode, 1]

    qb_bc = base_rhs_bc + Gamma * ntr
    is_bd = bd_mask[Gamma]

    for i in range(ntr):
        row = Gamma * ntr + i
        if is_bd:
            s = 0.0 if ntr == 1 else rg_edg_lag_nodes[i]
            x = 0.5 * ((1.0 - s) * xa + (1.0 + s) * xb)
            y = 0.5 * ((1.0 - s) * ya + (1.0 + s) * yb)
            gval = bd_cond(x, y)
            Ib[qb_bc + i] = row
            Vb[qb_bc + i] = penalty * gval
        else:
            Ib[qb_bc + i] = 0
            Vb[qb_bc + i] = 0.0



# ============================================================
# 3) Local Schur-condensation solver multi-RHS (no SPD assumption)
# ============================================================
@nb.njit(cache=True, inline="always")
def _schur_solve_local_multi_rhs(
        H, G0, G1, F0, F1, D0, D1,
        Mhat_loc, rhs_loc,
        LU0, piv0, LU1, piv1,
        S, LUS, pivS,
        X0, X1, Y0, Y1,
        Ru, U,
        T0, T1,
        Sol
):
    nel = H.shape[0]
    m = Mhat_loc.shape[1]
    nrhs = m + 1

    for i in range(nel):
        for c in range(m):
            Ru[i, c] = Mhat_loc[i, c]
            Y0[i, c] = Mhat_loc[nel + i, c]
            Y1[i, c] = Mhat_loc[2 * nel + i, c]
        Ru[i, m] = rhs_loc[i]
        Y0[i, m] = rhs_loc[nel + i]
        Y1[i, m] = rhs_loc[2 * nel + i]

    for i in range(nel):
        for j in range(nel):
            LU0[i, j] = D0[i, j]
            LU1[i, j] = D1[i, j]
    lu_factor_inplace(LU0, piv0)
    lu_factor_inplace(LU1, piv1)

    for i in range(nel):
        for j in range(nel):
            X0[i, j] = F0[i, j]
            X1[i, j] = F1[i, j]
    lu_solve_inplace(LU0, piv0, X0)
    lu_solve_inplace(LU1, piv1, X1)

    lu_solve_inplace(LU0, piv0, Y0)
    lu_solve_inplace(LU1, piv1, Y1)

    for i in range(nel):
        for j in range(nel):
            s = H[i, j]
            tmp = 0.0
            for k in range(nel):
                tmp += G0[i, k] * X0[k, j]
            s += tmp
            tmp = 0.0
            for k in range(nel):
                tmp += G1[i, k] * X1[k, j]
            s += tmp
            S[i, j] = s

    for i in range(nel):
        for c in range(nrhs):
            s = Ru[i, c]
            tmp = 0.0
            for k in range(nel):
                tmp += G0[i, k] * Y0[k, c]
            s += tmp
            tmp = 0.0
            for k in range(nel):
                tmp += G1[i, k] * Y1[k, c]
            s += tmp
            Ru[i, c] = s

    for i in range(nel):
        for j in range(nel):
            LUS[i, j] = S[i, j]
    lu_factor_inplace(LUS, pivS)

    for i in range(nel):
        for c in range(nrhs):
            U[i, c] = Ru[i, c]
    lu_solve_inplace(LUS, pivS, U)

    for i in range(nel):
        for c in range(nrhs):
            s0 = 0.0
            s1 = 0.0
            for k in range(nel):
                s0 += X0[i, k] * U[k, c]
                s1 += X1[i, k] * U[k, c]
            T0[i, c] = s0 - Y0[i, c]
            T1[i, c] = s1 - Y1[i, c]

    for i in range(nel):
        for c in range(nrhs):
            Sol[i, c] = U[i, c]
            Sol[nel + i, c] = T0[i, c]
            Sol[2 * nel + i, c] = T1[i, c]


# ============================================================
# 4) Scatter + diagonal/BC
# ============================================================
@nb.njit(cache=True, inline="always")
def _scatter_element_blocks_poisson(
        K, sigma, orien,
        edge3x3, Med_ref, Mhat_ref,
        bd_mask, penalty, tau_const,
        edge_patch_counts,
        Sol,
        I, J, V,
        Ib, Vb,
        baseK, nnzK,
        base_rhs_elem,
        build_schwarz,
        schwarz_reg,
        AS_lu, AS_piv, AS_dofs, AS_w
):
    """
    Scatter one element trace block/RHS and, optionally, build the element
    additive-Schwarz patch for the same local operator.

    The scattered element contribution is

        A_K[er,jloc; ec,kloc] = -C[jloc, ec*ntr + kloc].

    If build_schwarz is true, the same entries are copied into AS_lu[K].  The
    local patch also receives a distributed share of the global edge diagonal
    term from _diag_block_and_bc_poisson:

      * boundary edge: penalty*I, assigned to its unique adjacent element;
      * interior edge: 2*tau*J_edge*Mhat_ref, split over adjacent elements.

    The split keeps the local Schwarz block algebraically close to the matrix
    assembled by the element loop plus the subsequent edge-diagonal loop.
    """
    ntr = Med_ref.shape[2]
    nel = Med_ref.shape[1]
    m = 3 * ntr

    g = np.zeros(ntr, dtype=np.float64)
    C = np.zeros((ntr, m), dtype=np.float64)

    off_u = 0
    off_q0 = nel
    off_q1 = 2 * nel

    if build_schwarz:
        # Zero local patch matrix and fill the global DOF map/overlap weights.
        for a in range(m):
            for b in range(m):
                AS_lu[K, a, b] = 0.0

        for e in range(3):
            Gamma = sigma[K, e]
            ispos = orien[K, e]
            cnt = edge_patch_counts[Gamma]
            if cnt <= 0:
                cnt = 1
            w_patch = 1.0 / cnt

            for jloc in range(ntr):
                jg = map_edge_dof_bool(ispos, jloc, ntr)
                loc = e * ntr + jloc
                AS_dofs[K, loc] = Gamma * ntr + jg
                AS_w[K, loc] = w_patch

    for er in range(3):
        Jedge = edge3x3[er, 2]
        nx = edge3x3[er, 0]
        ny = edge3x3[er, 1]

        Gamma_row = sigma[K, er]
        ispos_row = orien[K, er]

        for j in range(ntr):
            g[j] = 0.0
            for c in range(m):
                C[j, c] = 0.0

        for a in range(nel):
            yu = Sol[off_u + a, m]
            yq0 = Sol[off_q0 + a, m]
            yq1 = Sol[off_q1 + a, m]
            comb_y = tau_const * yu + nx * yq0 + ny * yq1

            for jloc in range(ntr):
                wj = Jedge * Med_ref[er, a, jloc]
                g[jloc] += wj * comb_y

                for c in range(m):
                    Xu = Sol[off_u + a, c]
                    Xq0 = Sol[off_q0 + a, c]
                    Xq1 = Sol[off_q1 + a, c]
                    comb_X = tau_const * Xu + nx * Xq0 + ny * Xq1
                    C[jloc, c] += wj * comb_X

        qb0 = base_rhs_elem + (K * 3 + er) * ntr
        for jloc in range(ntr):
            jg = map_edge_dof_bool(ispos_row, jloc, ntr)
            row = Gamma_row * ntr + jg
            Ib[qb0 + jloc] = row
            Vb[qb0 + jloc] = g[jloc]

        for ec in range(3):
            Gamma_col = sigma[K, ec]
            ispos_col = orien[K, ec]
            blk = er * 3 + ec
            c0 = ec * ntr

            for jloc in range(ntr):
                jg = map_edge_dof_bool(ispos_row, jloc, ntr)
                row = Gamma_row * ntr + jg
                loc_row = er * ntr + jloc

                for kloc in range(ntr):
                    kg = map_edge_dof_bool(ispos_col, kloc, ntr)
                    col = Gamma_col * ntr + kg
                    loc_col = ec * ntr + kloc

                    val = -C[jloc, c0 + kloc]

                    loc = ((blk * ntr + jloc) * ntr + kloc)
                    p = baseK + loc
                    I[p] = row
                    J[p] = col
                    V[p] = val

                    if build_schwarz:
                        AS_lu[K, loc_row, loc_col] = val

    if build_schwarz:
        # Add the edge-diagonal/BC contribution consistently with
        # _diag_block_and_bc_poisson, but distributed over element patches.
        for e in range(3):
            Gamma = sigma[K, e]
            Jedge = edge3x3[e, 2]
            cnt = edge_patch_counts[Gamma]
            if cnt <= 0:
                cnt = 1
            share = 1.0 / cnt
            is_bd = bd_mask[Gamma]
            row0 = e * ntr

            for i in range(ntr):
                for j in range(ntr):
                    if is_bd:
                        val = penalty if i == j else 0.0
                    else:
                        val = 2.0 * tau_const * Jedge * Mhat_ref[i, j]
                    AS_lu[K, row0 + i, row0 + j] += share * val

        if schwarz_reg > 0.0:
            # Optional diagonal regularization for debugging/robustness if a
            # patch is singular or nearly singular.  Keep default at zero so the
            # preconditioner is algebraically faithful unless explicitly changed.
            for i in range(m):
                AS_lu[K, i, i] += schwarz_reg

        lu_factor_inplace(AS_lu[K], AS_piv[K])

@nb.njit(cache=True, inline="always")
def _diag_block_and_bc_poisson(
        Gamma,
        nodes, edge_nodes, bd_mask,
        penalty,
        tau_const,
        Mhat_ref, rg_edg_lag_nodes,
        I, J, V,
        Ib, Vb,
        baseMass, base_rhs_bc,
        ntr
):
    a = edge_nodes[Gamma, 0]
    bnode = edge_nodes[Gamma, 1]
    xa = nodes[a, 0]; ya = nodes[a, 1]
    xb = nodes[bnode, 0]; yb = nodes[bnode, 1]

    dx = xb - xa
    dy = yb - ya
    JG = 0.5 * np.sqrt(dx * dx + dy * dy)

    base = baseMass + Gamma * (ntr * ntr)
    qb_bc = base_rhs_bc + Gamma * ntr
    is_bd = bd_mask[Gamma]

    for i in range(ntr):
        row = Gamma * ntr + i
        if is_bd:
            s = 0.0 if ntr == 1 else rg_edg_lag_nodes[i]
            x = 0.5 * ((1.0 - s) * xa + (1.0 + s) * xb)
            y = 0.5 * ((1.0 - s) * ya + (1.0 + s) * yb)
            gval = bd_cond(x, y)
            Ib[qb_bc + i] = row
            Vb[qb_bc + i] = penalty * gval
        else:
            Ib[qb_bc + i] = 0
            Vb[qb_bc + i] = 0.0

    for i in range(ntr):
        row = Gamma * ntr + i
        for j in range(ntr):
            col = Gamma * ntr + j
            p = base + i * ntr + j
            if is_bd:
                val = penalty if (i == j) else 0.0
            else:
                # factor 2 on both sides + tau_const scaling (as in your current code)
                val = 2.0 * tau_const * JG * Mhat_ref[i, j]
            I[p] = row
            J[p] = col
            V[p] = val


# ============================================================
# 5) Global assembly using f_coef
# ============================================================
@nb.njit(cache=True, parallel=True, fastmath=True)
def assemble_trace_system_poisson_schur_fcoef(
        nodes, tris,
        sigma, orien,
        edge_nodes,
        bd_mask,
        penalty,
        tau_const,
        # u-space quad/basis
        xi_eta_u, wq_u, phi_u, gphi_u,
        # f-basis evaluated on u-quads
        psi_f_on_uq,  # (nq_u, nel_f)
        f_coef,  # (nK, nel_f)
        # edge quad/basis (u-space)
        sq, wf, phi_f, mu,
        Med_ref, Mhat_ref, rg_edg_lag_nodes,
        edge_patch_counts,
        build_schwarz,
        schwarz_reg,
        AS_lu, AS_piv, AS_dofs, AS_w,
        I, J, V,
        Ib, Vb
):
    nK = tris.shape[0]
    nel = phi_u.shape[1]
    ntr = mu.shape[1]
    m = 3 * ntr
    nE = edge_nodes.shape[0]

    nnzK = 9 * ntr * ntr
    baseMass = nK * nnzK

    base_rhs_elem = 0
    base_rhs_bc = nK * 3 * ntr

    for K in nb.prange(nK):
        geom5 = np.empty(5, dtype=np.float64)
        edge3x3 = np.empty((3, 3), dtype=np.float64)
        x0, y0, x1, y1, x2, y2 = _gather_geom(nodes, tris, K, geom5, edge3x3)

        H = np.empty((nel, nel), dtype=np.float64)
        G0 = np.empty((nel, nel), dtype=np.float64)
        G1 = np.empty((nel, nel), dtype=np.float64)
        F0 = np.empty((nel, nel), dtype=np.float64)
        F1 = np.empty((nel, nel), dtype=np.float64)
        D0 = np.empty((nel, nel), dtype=np.float64)
        D1 = np.empty((nel, nel), dtype=np.float64)

        Mhat_loc = np.empty((3 * nel, m), dtype=np.float64)
        rhs_loc = np.empty(3 * nel, dtype=np.float64)

        _assemble_local_blocks_hat_rhs_poisson_fcoef(
            x0, y0, x1, y1, x2, y2,
            geom5, edge3x3,
            tau_const,
            xi_eta_u, wq_u, phi_u, gphi_u,
            psi_f_on_uq, f_coef[K],
            sq, wf, phi_f, mu,
            H, G0, G1, F0, F1, D0, D1,
            Mhat_loc, rhs_loc
        )

        LU0 = np.empty((nel, nel), dtype=np.float64)
        LU1 = np.empty((nel, nel), dtype=np.float64)
        piv0 = np.empty(nel, dtype=np.int32)
        piv1 = np.empty(nel, dtype=np.int32)

        S = np.empty((nel, nel), dtype=np.float64)
        LUS = np.empty((nel, nel), dtype=np.float64)
        pivS = np.empty(nel, dtype=np.int32)

        X0 = np.empty((nel, nel), dtype=np.float64)
        X1 = np.empty((nel, nel), dtype=np.float64)
        Y0 = np.empty((nel, m + 1), dtype=np.float64)
        Y1 = np.empty((nel, m + 1), dtype=np.float64)
        Ru = np.empty((nel, m + 1), dtype=np.float64)
        U = np.empty((nel, m + 1), dtype=np.float64)
        T0 = np.empty((nel, m + 1), dtype=np.float64)
        T1 = np.empty((nel, m + 1), dtype=np.float64)

        Sol = np.empty((3 * nel, m + 1), dtype=np.float64)

        _schur_solve_local_multi_rhs(
            H, G0, G1, F0, F1, D0, D1,
            Mhat_loc, rhs_loc,
            LU0, piv0, LU1, piv1,
            S, LUS, pivS,
            X0, X1, Y0, Y1,
            Ru, U,
            T0, T1,
            Sol
        )

        baseK = K * nnzK
        _scatter_element_blocks_poisson(
            K, sigma, orien,
            edge3x3, Med_ref, Mhat_ref,
            bd_mask, penalty, tau_const,
            edge_patch_counts,
            Sol,
            I, J, V,
            Ib, Vb,
            baseK, nnzK,
            base_rhs_elem,
            build_schwarz,
            schwarz_reg,
            AS_lu, AS_piv, AS_dofs, AS_w
        )

    for Gamma in nb.prange(nE):
        _diag_block_and_bc_poisson(
            Gamma,
            nodes, edge_nodes, bd_mask,
            penalty,
            tau_const,
            Mhat_ref, rg_edg_lag_nodes,
            I, J, V,
            Ib, Vb,
            baseMass, base_rhs_bc,
            ntr
        )

@nb.njit(cache=True, parallel=True, fastmath=True)
def assemble_trace_rhs_poisson_schur_fcoef(
        nodes, tris,
        sigma, orien,
        edge_nodes,
        bd_mask,
        penalty,
        tau_const,
        # u-space quad/basis
        xi_eta_u, wq_u, phi_u, gphi_u,
        # f-basis evaluated on u-quads
        psi_f_on_uq,
        f_coef,
        # edge quad/basis (u-space)
        sq, wf, phi_f, mu,
        Med_ref, rg_edg_lag_nodes,
        Ib, Vb
):
    """
    Assemble only the condensed global RHS for the Poisson HDG trace system.

    This is the efficient repeated-solve path when the global matrix is fixed
    and only the source changes.
    """
    nK = tris.shape[0]
    nel = phi_u.shape[1]
    ntr = mu.shape[1]
    nE = edge_nodes.shape[0]

    base_rhs_elem = 0
    base_rhs_bc = nK * 3 * ntr

    for K in nb.prange(nK):
        geom5 = np.empty(5, dtype=np.float64)
        edge3x3 = np.empty((3, 3), dtype=np.float64)
        x0, y0, x1, y1, x2, y2 = _gather_geom(nodes, tris, K, geom5, edge3x3)

        H = np.empty((nel, nel), dtype=np.float64)
        G0 = np.empty((nel, nel), dtype=np.float64)
        G1 = np.empty((nel, nel), dtype=np.float64)
        F0 = np.empty((nel, nel), dtype=np.float64)
        F1 = np.empty((nel, nel), dtype=np.float64)
        D0 = np.empty((nel, nel), dtype=np.float64)
        D1 = np.empty((nel, nel), dtype=np.float64)

        m = 3 * ntr
        Mhat_loc = np.empty((3 * nel, m), dtype=np.float64)
        rhs_loc = np.empty(3 * nel, dtype=np.float64)

        _assemble_local_blocks_hat_rhs_poisson_fcoef(
            x0, y0, x1, y1, x2, y2,
            geom5, edge3x3,
            tau_const,
            xi_eta_u, wq_u, phi_u, gphi_u,
            psi_f_on_uq, f_coef[K],
            sq, wf, phi_f, mu,
            H, G0, G1, F0, F1, D0, D1,
            Mhat_loc, rhs_loc
        )

        LU0 = np.empty((nel, nel), dtype=np.float64)
        LU1 = np.empty((nel, nel), dtype=np.float64)
        piv0 = np.empty(nel, dtype=np.int32)
        piv1 = np.empty(nel, dtype=np.int32)

        S = np.empty((nel, nel), dtype=np.float64)
        LUS = np.empty((nel, nel), dtype=np.float64)
        pivS = np.empty(nel, dtype=np.int32)

        X0 = np.empty((nel, nel), dtype=np.float64)
        X1 = np.empty((nel, nel), dtype=np.float64)
        y0 = np.empty(nel, dtype=np.float64)
        y1 = np.empty(nel, dtype=np.float64)

        u = np.empty(nel, dtype=np.float64)
        q0 = np.empty(nel, dtype=np.float64)
        q1 = np.empty(nel, dtype=np.float64)

        _schur_solve_local_single_rhs_nospd(
            H, G0, G1, F0, F1, D0, D1,
            rhs_loc,
            LU0, piv0, LU1, piv1,
            S, LUS, pivS,
            X0, X1,
            y0, y1,
            u, q0, q1
        )

        _scatter_element_rhs_poisson(
            K, sigma, orien,
            edge3x3, Med_ref, tau_const,
            u, q0, q1,
            Ib, Vb,
            base_rhs_elem
        )

    for Gamma in nb.prange(nE):
        _diag_bc_rhs_poisson(
            Gamma,
            nodes, edge_nodes, bd_mask,
            penalty,
            rg_edg_lag_nodes,
            Ib, Vb,
            base_rhs_bc,
            ntr
        )


# ============================================================
# 6) Reconstruction using f_coef (packed qK optional)
# ============================================================
@nb.njit(cache=True, inline="always")
def _gather_uhat_local(K, sigma, orien, uhat_global, ntr, uhat_loc):
    for e in range(3):
        Gamma = sigma[K, e]
        ispos = orien[K, e]
        baseG = Gamma * ntr
        baseL = e * ntr
        for jloc in range(ntr):
            jg = map_edge_dof_bool(ispos, jloc, ntr)
            uhat_loc[baseL + jloc] = uhat_global[baseG + jg]


@nb.njit(cache=True, inline="always")
def _apply_Mhat_plus_rhs(Mhat_loc, rhs_loc, uhat_loc, r_loc):
    nloc = rhs_loc.shape[0]
    m = uhat_loc.shape[0]
    for i in range(nloc):
        s = rhs_loc[i]
        for c in range(m):
            s += Mhat_loc[i, c] * uhat_loc[c]
        r_loc[i] = s


@nb.njit(cache=True, inline="always")
def _schur_solve_local_single_rhs_nospd(
        H, G0, G1, F0, F1, D0, D1,
        r_loc,
        LU0, piv0, LU1, piv1,
        S, LUS, pivS,
        X0, X1,
        y0, y1,
        u, q0, q1
):
    nel = H.shape[0]

    for i in range(nel):
        for j in range(nel):
            LU0[i, j] = D0[i, j]
            LU1[i, j] = D1[i, j]
    lu_factor_inplace(LU0, piv0)
    lu_factor_inplace(LU1, piv1)

    for i in range(nel):
        for j in range(nel):
            X0[i, j] = F0[i, j]
            X1[i, j] = F1[i, j]
    lu_solve_inplace(LU0, piv0, X0)
    lu_solve_inplace(LU1, piv1, X1)

    for i in range(nel):
        y0[i] = r_loc[nel + i]
        y1[i] = r_loc[2 * nel + i]
    tmp0 = y0.reshape(nel, 1)
    tmp1 = y1.reshape(nel, 1)
    lu_solve_inplace(LU0, piv0, tmp0)
    lu_solve_inplace(LU1, piv1, tmp1)

    for i in range(nel):
        for j in range(nel):
            s = H[i, j]
            t = 0.0
            for k in range(nel):
                t += G0[i, k] * X0[k, j]
            s += t
            t = 0.0
            for k in range(nel):
                t += G1[i, k] * X1[k, j]
            s += t
            S[i, j] = s

    for i in range(nel):
        s = r_loc[i]
        t = 0.0
        for k in range(nel):
            t += G0[i, k] * y0[k]
        s += t
        t = 0.0
        for k in range(nel):
            t += G1[i, k] * y1[k]
        s += t
        u[i] = s

    for i in range(nel):
        for j in range(nel):
            LUS[i, j] = S[i, j]
    lu_factor_inplace(LUS, pivS)
    tmpu = u.reshape(nel, 1)
    lu_solve_inplace(LUS, pivS, tmpu)

    for i in range(nel):
        s0 = 0.0
        s1 = 0.0
        for k in range(nel):
            uk = u[k]
            s0 += X0[i, k] * uk
            s1 += X1[i, k] * uk
        q0[i] = s0 - y0[i]
        q1[i] = s1 - y1[i]


@nb.njit(cache=True, parallel=True, fastmath=True)
def reconstruct_uq_from_trace_fcoef_packed(
        nodes, tris,
        sigma, orien,
        uhat_global,
        tau_const,
        # u-space quad/basis
        xi_eta_u, wq_u, phi_u, gphi_u,
        # f-basis on u-quads + element coefficients
        psi_f_on_uq, f_coef,
        # edge quad/basis
        sq, wf, phi_f, mu,
        # outputs
        uK, qK  # uK: (nK,nel_u), qK: (nK,nel_u,2)
):
    nK = tris.shape[0]
    nel = phi_u.shape[1]
    ntr = mu.shape[1]
    m = 3 * ntr

    for K in nb.prange(nK):
        geom5 = np.empty(5, dtype=np.float64)
        edge3x3 = np.empty((3, 3), dtype=np.float64)
        x0, y0, x1, y1, x2, y2 = _gather_geom(nodes, tris, K, geom5, edge3x3)

        H = np.empty((nel, nel), dtype=np.float64)
        G0 = np.empty((nel, nel), dtype=np.float64)
        G1 = np.empty((nel, nel), dtype=np.float64)
        F0 = np.empty((nel, nel), dtype=np.float64)
        F1 = np.empty((nel, nel), dtype=np.float64)
        D0 = np.empty((nel, nel), dtype=np.float64)
        D1 = np.empty((nel, nel), dtype=np.float64)

        Mhat_loc = np.empty((3 * nel, m), dtype=np.float64)
        rhs_loc = np.empty(3 * nel, dtype=np.float64)

        _assemble_local_blocks_hat_rhs_poisson_fcoef(
            x0, y0, x1, y1, x2, y2,
            geom5, edge3x3,
            tau_const,
            xi_eta_u, wq_u, phi_u, gphi_u,
            psi_f_on_uq, f_coef[K],
            sq, wf, phi_f, mu,
            H, G0, G1, F0, F1, D0, D1,
            Mhat_loc, rhs_loc
        )

        uhat_loc = np.empty(m, dtype=np.float64)
        _gather_uhat_local(K, sigma, orien, uhat_global, ntr, uhat_loc)

        r_loc = np.empty(3 * nel, dtype=np.float64)
        _apply_Mhat_plus_rhs(Mhat_loc, rhs_loc, uhat_loc, r_loc)

        LU0 = np.empty((nel, nel), dtype=np.float64)
        LU1 = np.empty((nel, nel), dtype=np.float64)
        piv0 = np.empty(nel, dtype=np.int32)
        piv1 = np.empty(nel, dtype=np.int32)

        S = np.empty((nel, nel), dtype=np.float64)
        LUS = np.empty((nel, nel), dtype=np.float64)
        pivS = np.empty(nel, dtype=np.int32)

        X0 = np.empty((nel, nel), dtype=np.float64)
        X1 = np.empty((nel, nel), dtype=np.float64)
        y0 = np.empty(nel, dtype=np.float64)
        y1 = np.empty(nel, dtype=np.float64)

        u = np.empty(nel, dtype=np.float64)
        q0 = np.empty(nel, dtype=np.float64)
        q1 = np.empty(nel, dtype=np.float64)

        _schur_solve_local_single_rhs_nospd(
            H, G0, G1, F0, F1, D0, D1,
            r_loc,
            LU0, piv0, LU1, piv1,
            S, LUS, pivS,
            X0, X1,
            y0, y1,
            u, q0, q1
        )

        for i in range(nel):
            uK[K, i] = u[i]
            qK[K, i, 0] = q0[i]
            qK[K, i, 1] = q1[i]


# ============================================================
# 7) Utility: apply boundary to uhat in-place
# ============================================================
@nb.njit(cache=True, parallel=True, fastmath=True)
def apply_bd_to_uhat_inplace(
        uhat,
        nodes,
        edge_nodes,
        bd_mask,
        rg_edg_lag_nodes,
        ntr
):
    nE = edge_nodes.shape[0]
    for Gamma in nb.prange(nE):
        if not bd_mask[Gamma]:
            continue
        a = edge_nodes[Gamma, 0]
        b = edge_nodes[Gamma, 1]
        xa = nodes[a, 0]; ya = nodes[a, 1]
        xb = nodes[b, 0]; yb = nodes[b, 1]
        base = Gamma * ntr
        for i in range(ntr):
            s = rg_edg_lag_nodes[i]
            x = 0.5 * ((1.0 - s) * xa + (1.0 + s) * xb)
            y = 0.5 * ((1.0 - s) * ya + (1.0 + s) * yb)
            uhat[base + i] = bd_cond(x, y)


def eval_basis_on_ref_points(basis_list, xi_eta):
    """
    Evaluate a list of scalar basis callables on reference points.

    Returns B with shape (nq, nbas): B[q,j] = basis_j(xi_q, eta_q)
    """
    nq = xi_eta.shape[0]
    nbas = len(basis_list)
    B = np.empty((nq, nbas), dtype=np.float64)
    for q in range(nq):
        xi = float(xi_eta[q, 0])
        eta = float(xi_eta[q, 1])
        for j, psi in enumerate(basis_list):
            B[q, j] = psi(xi, eta)
    return np.ascontiguousarray(B)


def build_rhs_dg_tables(quad_u, quad_f):
    """
    Build basis evaluation tables needed for:
      - projection: psi_f on f-quadrature points
      - assembly/reconstruction: psi_f evaluated on u-quadrature points
    """
    psi_f = eval_basis_on_ref_points(quad_f.basis, quad_f.Krf_quads)  # (nq_f, nel_f)
    psi_f_on_uq = eval_basis_on_ref_points(quad_f.basis, quad_u.Krf_quads)  # (nq_u, nel_f)
    return psi_f, psi_f_on_uq


def project_source_to_fcoef(msh, quad_f, psi_f):
    """
    Compute f_coef (nK, nel_f) using the Numba projection kernel.
    """
    nK = msh.num_tri
    nel_f = quad_f.el_dof
    f_coef = np.empty((nK, nel_f), dtype=np.float64)

    project_src_func_to_dg_coeffs(
        msh.node_coords, msh.triangles,
        quad_f.Krf_quads, quad_f.Krf_w, psi_f,
        quad_f.MKrf_inv,
        f_coef
    )
    return f_coef


def assemble_trace_system_from_fcoef(
        msh, quad_u,
        bd_mask,
        tau_const, penalty,
        psi_f_on_uq, f_coef,
        I, J, V, Ib, Vb,
        schwarz_data: Optional[ElementSchwarzData] = None,
        schwarz_reg: float = 0.0,
):
    """
    Assemble global trace system (COO) using DG source coefficients f_coef.

    If schwarz_data is provided, element additive-Schwarz patch factors are
    built during the same element scatter.  No sparse submatrix extraction is
    performed after assembly.
    """
    phi_u = np.ascontiguousarray(quad_u.bas_of_quads.T)
    gphi_u = np.ascontiguousarray(quad_u.dbas_of_quads.swapaxes(0, 2))
    phi_f_bd = np.ascontiguousarray(quad_u.bas_of_bd_quads.swapaxes(1, 2))
    mu = np.ascontiguousarray(quad_u.bas1d_of_ref_edg_qds.T)

    ntr = quad_u.edg_dof
    m = 3 * ntr

    build_schwarz = schwarz_data is not None
    edge_counts = edge_patch_counts_from_sigma(msh.sigma, msh.num_edg)

    if build_schwarz:
        AS_lu = schwarz_data.lu
        AS_piv = schwarz_data.piv
        AS_dofs = schwarz_data.dofs
        AS_w = schwarz_data.weights
    else:
        # Dummy arrays keep the Numba kernel monomorphic.  They are not touched
        # when build_schwarz=False.
        AS_lu = np.empty((1, m, m), dtype=np.float64)
        AS_piv = np.empty((1, m), dtype=np.int32)
        AS_dofs = np.empty((1, m), dtype=np.int64)
        AS_w = np.empty((1, m), dtype=np.float64)

    assemble_trace_system_poisson_schur_fcoef(
        msh.node_coords, msh.triangles,
        msh.sigma, msh.orientations,
        msh.edges,
        bd_mask,
        penalty,
        tau_const,
        quad_u.Krf_quads, quad_u.Krf_w, phi_u, gphi_u,
        psi_f_on_uq, f_coef,
        quad_u.quads_JGL, quad_u.weights_JGL, phi_f_bd, mu,
        quad_u.MKrfe_lst_p, quad_u.mass_ref_face(), quad_u.rf_edg_lag_nodes,
        edge_counts,
        build_schwarz,
        schwarz_reg,
        AS_lu, AS_piv, AS_dofs, AS_w,
        I, J, V,
        Ib, Vb,
    )


def assemble_trace_rhs_from_fcoef(
        msh, quad_u,
        bd_mask,
        tau_const, penalty,
        psi_f_on_uq, f_coef,
        Ib, Vb
):
    """
    Assemble only the condensed global RHS using DG source coefficients.

    Use this when the global trace matrix has already been assembled and cached.
    """
    phi_u = np.ascontiguousarray(quad_u.bas_of_quads.T)
    gphi_u = np.ascontiguousarray(quad_u.dbas_of_quads.swapaxes(0, 2))
    phi_f_bd = np.ascontiguousarray(quad_u.bas_of_bd_quads.swapaxes(1, 2))
    mu = np.ascontiguousarray(quad_u.bas1d_of_ref_edg_qds.T)

    assemble_trace_rhs_poisson_schur_fcoef(
        msh.node_coords, msh.triangles,
        msh.sigma, msh.orientations,
        msh.edges,
        bd_mask,
        penalty,
        tau_const,
        quad_u.Krf_quads, quad_u.Krf_w, phi_u, gphi_u,
        psi_f_on_uq, f_coef,
        quad_u.quads_JGL, quad_u.weights_JGL, phi_f_bd, mu,
        quad_u.MKrfe_lst_p, quad_u.rf_edg_lag_nodes,
        Ib, Vb
    )

def coo_rhs_to_dense(Ib, Vb, ndofT):
    """
    Convert RHS COO-vector (Ib,Vb) into dense rhs via bincount.
    """
    return np.bincount(Ib, weights=Vb, minlength=ndofT)


def reconstruct_uq_from_fcoef(
        msh, quad_u,
        bd_mask,
        tau_const,
        uhat,
        psi_f_on_uq, f_coef
):
    """
    Apply Dirichlet overwrite to uhat and reconstruct uK and packed qK.
    """
    nK = msh.num_tri
    nel_u = quad_u.el_dof
    ntr = quad_u.edg_dof

    apply_bd_to_uhat_inplace(uhat, msh.node_coords, msh.edges, bd_mask, quad_u.rf_edg_lag_nodes, ntr)

    uK = np.empty((nK, nel_u), dtype=np.float64)
    qK = np.empty((nK, nel_u, 2), dtype=np.float64)

    phi_u = np.ascontiguousarray(quad_u.bas_of_quads.T)
    gphi_u = np.ascontiguousarray(quad_u.dbas_of_quads.swapaxes(0, 2))
    phi_f_bd = np.ascontiguousarray(quad_u.bas_of_bd_quads.swapaxes(1, 2))
    mu = np.ascontiguousarray(quad_u.bas1d_of_ref_edg_qds.T)

    reconstruct_uq_from_trace_fcoef_packed(
        msh.node_coords, msh.triangles,
        msh.sigma, msh.orientations,
        uhat,
        tau_const,
        quad_u.Krf_quads, quad_u.Krf_w, phi_u, gphi_u,
        psi_f_on_uq, f_coef,
        quad_u.quads_JGL, quad_u.weights_JGL, phi_f_bd, mu,
        uK, qK
    )
    return uK, qK

def diff_rea_solv(src: Union[NDArray, Callable],
                  msh: Triangulation,
                  quad_u: TriangleQuadratureData,
                  quad_f: TriangleQuadratureData = None,
                  tau_const=1.0):

    if quad_f is None: quad_f = quad_u

    bd_mask = np.zeros(msh.num_edg, dtype=np.bool_)
    bd_mask[msh.bnd_edges_inds] = True

    if quad_f is None:
        psi_f, psi_sr_on_uq = quad_u.bas_of_quads, quad_u.bas_of_quads
    else:
        psi_f, psi_sr_on_uq = build_rhs_dg_tables(quad_u, quad_f)

    # for COO allocation
    nK = msh.num_tri
    nE = msh.num_edg
    ntr = quad_u.edg_dof
    nnzK = 9 * ntr * ntr
    nnz = nK * nnzK + nE * (ntr * ntr)

    I = np.empty(nnz, dtype=np.int64)
    J = np.empty(nnz, dtype=np.int64)
    V = np.empty(nnz, dtype=np.float64)

    Ib = np.empty(nK * 3 * ntr + nE * ntr, dtype=np.int64)
    Vb = np.empty(nK * 3 * ntr + nE * ntr, dtype=np.float64)

    asmb_time = time.time()
    print("assembling ...", end=" ", flush=True)
    assemble_trace_system_from_fcoef(
        msh, quad_u,
        bd_mask,
        tau_const, 1e20,
        psi_sr_on_uq, src,
        I, J, V, Ib, Vb
    )
    asmb_time = time.time() - asmb_time
    print(f"done in {asmb_time:.2f}s")

    # Build dense rhs
    ndofT = msh.num_edg * quad_u.edg_dof
    rhs = coo_rhs_to_dense(Ib, Vb, ndofT)

    # Solve
    solv_time = time.time()
    # uhat, info = solve_petsc_from_coo_new(row_inds=I, col_inds=J, data=V, rhs=rhs, solver_preset="cg_gamg")
    result = solve_global_system(
        row_indices=I,
        col_indices=J,
        matrix_values=V,
        rhs=rhs,
        system_size=rhs.size,
        solver="BICGSTAB",
        preconditioner="ilu",
        verbose=True,
    )
    uhat = result.x
    solv_time = time.time() - solv_time

    # Reconstruct
    recons_time = time.time()
    print(f"recovering uh ... ", end=" ", flush=True)
    uh, q_h = reconstruct_uq_from_fcoef(msh, quad_u, bd_mask, tau_const, uhat, psi_sr_on_uq, src)
    recons_time = time.time() - recons_time
    print(f"done in {recons_time:.2f}s")
    return uh, q_h

def diff_rea_solv_(src: Union[NDArray, Callable],
                  msh: Triangulation,
                  quad_u: TriangleQuadratureData,
                  quad_f: TriangleQuadratureData = None,
                  tau_const=1.0,
                  penalty=1e20,
                  itr_solv: str = "BICGSTAB"):
    """
    One-shot convenience wrapper.

    For repeated solves on a fixed operator, prefer instantiating
    `DiffReaHDGSolver` directly and reusing it.
    """
    solver = DiffReaHDGSolver(
        msh=msh,
        quad_u=quad_u,
        quad_f=quad_f,
        tau_const=tau_const,
        penalty=penalty,
    )
    uh, q_h = solver.solve(src, itr_solv=itr_solv, reuse_prec=True, update_prec=True, verbose=True)
    return uh, q_h

