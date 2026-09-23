"""Small CPU reference implementation of restarted GMRES.

The implementation in this module is intentionally written with NumPy and a
callable matrix-vector product.  It does not inspect or materialize the matrix,
which makes it directly compatible with the HDG face-dense operator and gives
us a clear algorithmic reference before moving the same buffers and kernels to
the GPU.

The solver uses left preconditioning when a preconditioner callable is supplied:

    M^{-1} A x = M^{-1} b.

Modified Gram--Schmidt and incremental Givens rotations are used inside every
restart cycle.  Because this is a correctness-oriented CPU implementation, the
true residual is recomputed after every Arnoldi step.  A later GPU version can
reduce that frequency and primarily use the inexpensive Givens residual
estimate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..assembly.face_dense import FaceDenseSystem, face_dense_matvec

VectorOperator = Callable[[np.ndarray], np.ndarray]
GMRESStatus = Literal["converged", "max_iterations", "breakdown"]


@dataclass(frozen=True)
class GMRESResult:
    """Result and diagnostics returned by :func:`restarted_gmres`.

    Attributes
    ----------
    solution
        Approximate solution with the same shape as the supplied right-hand
        side.
    converged
        Whether the requested true-residual tolerance was reached.
    status
        ``"converged"``, ``"max_iterations"``, or ``"breakdown"``.
    iterations
        Total number of Arnoldi steps over all restart cycles.
    restart_cycles
        Number of restart cycles entered.
    residual_norm
        Final true residual norm ``||b - A x||_2``.
    relative_residual
        Final residual divided by ``max(||b||_2, eps)``.
    residual_history
        True residual norm at the initial guess and after every Arnoldi step.
    preconditioned_residual_history
        Givens/least-squares estimate of the left-preconditioned residual.  Its
        first entry is ``||M^{-1}(b-Ax_0)||_2``.
    matvec_count
        Number of calls to the matrix-vector product.  This reference solver
        performs extra calls to verify the true residual at every step.
    preconditioner_count
        Number of calls to the preconditioner.
    """

    solution: np.ndarray
    converged: bool
    status: GMRESStatus
    iterations: int
    restart_cycles: int
    residual_norm: float
    relative_residual: float
    residual_history: np.ndarray
    preconditioned_residual_history: np.ndarray
    matvec_count: int
    preconditioner_count: int


def _validated_operator_output(
    value: np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size != size:
        raise ValueError(
            f"{name} must return {size} values; got output shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} returned non-finite values")
    return np.ascontiguousarray(array.reshape(-1))


def restarted_gmres(
    matvec: VectorOperator,
    rhs: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    restart: int = 30,
    max_iterations: int | None = None,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    preconditioner: VectorOperator | None = None,
    reorthogonalize: bool = False,
    breakdown_tolerance: float | None = None,
) -> GMRESResult:
    r"""Solve ``A x = rhs`` with restarted, left-preconditioned GMRES.

    Parameters
    ----------
    matvec
        Callable implementing ``A @ x``.  It receives and returns flat NumPy
        vectors of length ``rhs.size``.
    rhs
        Right-hand side.  It may be flat or face-major; the returned solution
        has the same shape.
    x0
        Optional initial guess with the same number of values as ``rhs``.
    restart
        Maximum Krylov dimension in one restart cycle.
    max_iterations
        Maximum total Arnoldi steps.  The default is ``10 * rhs.size``.
    rtol, atol
        Convergence requires

        ``||b - A x||_2 <= max(atol, rtol * ||b||_2)``.

    preconditioner
        Optional callable implementing the left-preconditioner action
        ``M^{-1} @ x``.  It also receives and returns flat vectors.
    reorthogonalize
        Apply a second modified Gram--Schmidt pass.  This costs more dot/AXPY
        operations but is useful when the Arnoldi basis loses orthogonality.
    breakdown_tolerance
        Relative threshold used to detect an Arnoldi happy breakdown.  The
        default is ``100 * machine_epsilon``.

    Notes
    -----
    The Hessenberg matrix, Givens coefficients, and triangular solves are all
    small dense CPU objects.  The large-vector operations are exposed only
    through ``matvec``, preconditioner, dot, norm, and AXPY-style NumPy
    operations, mirroring the intended GPU decomposition.
    """

    if not callable(matvec):
        raise TypeError("matvec must be callable")
    if preconditioner is not None and not callable(preconditioner):
        raise TypeError("preconditioner must be callable or None")
    if isinstance(restart, bool) or int(restart) != restart or restart <= 0:
        raise ValueError("restart must be a positive integer")
    restart = int(restart)
    if rtol < 0.0 or not np.isfinite(rtol):
        raise ValueError("rtol must be a finite non-negative number")
    if atol < 0.0 or not np.isfinite(atol):
        raise ValueError("atol must be a finite non-negative number")

    rhs_array = np.asarray(rhs, dtype=np.float64)
    if rhs_array.size == 0:
        raise ValueError("rhs must be non-empty")
    if not np.all(np.isfinite(rhs_array)):
        raise ValueError("rhs contains non-finite values")
    original_shape = rhs_array.shape
    b = np.ascontiguousarray(rhs_array.reshape(-1))
    size = int(b.size)

    if max_iterations is None:
        max_iterations = 10 * size
    if (
        isinstance(max_iterations, bool)
        or int(max_iterations) != max_iterations
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
    max_iterations = int(max_iterations)

    if breakdown_tolerance is None:
        breakdown_tolerance = 100.0 * np.finfo(np.float64).eps
    if breakdown_tolerance < 0.0 or not np.isfinite(breakdown_tolerance):
        raise ValueError(
            "breakdown_tolerance must be a finite non-negative number"
        )

    if x0 is None:
        x = np.zeros(size, dtype=np.float64)
    else:
        x0_array = np.asarray(x0, dtype=np.float64)
        if x0_array.size != size:
            raise ValueError(
                f"x0 must contain {size} values; got shape {x0_array.shape}"
            )
        if not np.all(np.isfinite(x0_array)):
            raise ValueError("x0 contains non-finite values")
        x = np.ascontiguousarray(x0_array.reshape(-1).copy())

    matvec_count = 0
    preconditioner_count = 0

    def apply_matvec(vector: np.ndarray) -> np.ndarray:
        nonlocal matvec_count
        matvec_count += 1
        return _validated_operator_output(
            matvec(vector),
            size=size,
            name="matvec",
        )

    def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
        nonlocal preconditioner_count
        if preconditioner is None:
            return np.ascontiguousarray(vector.copy())
        preconditioner_count += 1
        return _validated_operator_output(
            preconditioner(vector),
            size=size,
            name="preconditioner",
        )

    b_norm = float(np.linalg.norm(b))
    denominator = max(b_norm, np.finfo(np.float64).eps)
    target = max(float(atol), float(rtol) * b_norm)

    true_residual = b - apply_matvec(x)
    true_norm = float(np.linalg.norm(true_residual))
    residual_history: list[float] = [true_norm]

    preconditioned_residual = apply_preconditioner(true_residual)
    beta = float(np.linalg.norm(preconditioned_residual))
    preconditioned_history: list[float] = [beta]

    def make_result(status: GMRESStatus, iterations: int, cycles: int) -> GMRESResult:
        final_residual = b - apply_matvec(x)
        final_norm = float(np.linalg.norm(final_residual))
        # Avoid adding a duplicate to the diagnostic history while still making
        # sure the reported residual is freshly verified.
        if not residual_history or final_norm != residual_history[-1]:
            residual_history.append(final_norm)
        return GMRESResult(
            solution=np.ascontiguousarray(x.reshape(original_shape)),
            converged=status == "converged",
            status=status,
            iterations=iterations,
            restart_cycles=cycles,
            residual_norm=final_norm,
            relative_residual=final_norm / denominator,
            residual_history=np.asarray(residual_history, dtype=np.float64),
            preconditioned_residual_history=np.asarray(
                preconditioned_history,
                dtype=np.float64,
            ),
            matvec_count=matvec_count,
            preconditioner_count=preconditioner_count,
        )

    if true_norm <= target:
        return make_result("converged", 0, 0)

    iterations = 0
    cycles = 0

    while iterations < max_iterations:
        cycles += 1

        # Recompute the residual at every restart so accumulated update and
        # orthogonalization errors cannot silently drift from the true system.
        true_residual = b - apply_matvec(x)
        true_norm = float(np.linalg.norm(true_residual))
        if true_norm <= target:
            return make_result("converged", iterations, cycles)

        preconditioned_residual = apply_preconditioner(true_residual)
        beta = float(np.linalg.norm(preconditioned_residual))
        if beta <= breakdown_tolerance * max(1.0, true_norm):
            # M^{-1}r vanished although the true residual did not.  This means
            # the supplied preconditioner is singular/incompatible here.
            return make_result("breakdown", iterations, cycles)

        inner_limit = min(restart, size, max_iterations - iterations)
        basis = np.zeros((size, inner_limit + 1), dtype=np.float64)
        hessenberg = np.zeros((inner_limit + 1, inner_limit), dtype=np.float64)
        givens_cos = np.zeros(inner_limit, dtype=np.float64)
        givens_sin = np.zeros(inner_limit, dtype=np.float64)
        least_squares_rhs = np.zeros(inner_limit + 1, dtype=np.float64)

        basis[:, 0] = preconditioned_residual / beta
        least_squares_rhs[0] = beta

        cycle_candidate = x.copy()

        for column in range(inner_limit):
            # Arnoldi action for the left-preconditioned operator M^{-1} A.
            work = apply_preconditioner(apply_matvec(basis[:, column]))
            work_before_orthogonalization = float(np.linalg.norm(work))

            # Modified Gram--Schmidt.
            for row in range(column + 1):
                coefficient = float(np.dot(basis[:, row], work))
                hessenberg[row, column] = coefficient
                work -= coefficient * basis[:, row]

            if reorthogonalize:
                for row in range(column + 1):
                    correction = float(np.dot(basis[:, row], work))
                    hessenberg[row, column] += correction
                    work -= correction * basis[:, row]

            next_norm = float(np.linalg.norm(work))
            hessenberg[column + 1, column] = next_norm
            happy_breakdown = next_norm <= breakdown_tolerance * max(
                1.0,
                work_before_orthogonalization,
            )
            if not happy_breakdown:
                basis[:, column + 1] = work / next_norm

            # Apply the rotations accumulated from previous columns.
            for rotation in range(column):
                upper = hessenberg[rotation, column]
                lower = hessenberg[rotation + 1, column]
                hessenberg[rotation, column] = (
                    givens_cos[rotation] * upper
                    + givens_sin[rotation] * lower
                )
                hessenberg[rotation + 1, column] = (
                    -givens_sin[rotation] * upper
                    + givens_cos[rotation] * lower
                )

            # Construct and apply the new Givens rotation.
            diagonal = hessenberg[column, column]
            subdiagonal = hessenberg[column + 1, column]
            radius = float(np.hypot(diagonal, subdiagonal))
            if radius == 0.0:
                givens_cos[column] = 1.0
                givens_sin[column] = 0.0
            else:
                givens_cos[column] = diagonal / radius
                givens_sin[column] = subdiagonal / radius

            hessenberg[column, column] = radius
            hessenberg[column + 1, column] = 0.0

            rhs_upper = least_squares_rhs[column]
            rhs_lower = least_squares_rhs[column + 1]
            least_squares_rhs[column] = (
                givens_cos[column] * rhs_upper
                + givens_sin[column] * rhs_lower
            )
            least_squares_rhs[column + 1] = (
                -givens_sin[column] * rhs_upper
                + givens_cos[column] * rhs_lower
            )
            estimated_preconditioned_norm = abs(
                float(least_squares_rhs[column + 1])
            )
            preconditioned_history.append(estimated_preconditioned_norm)

            krylov_size = column + 1
            triangular = hessenberg[:krylov_size, :krylov_size]
            projected_rhs = least_squares_rhs[:krylov_size]
            try:
                coefficients = np.linalg.solve(triangular, projected_rhs)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(
                    triangular,
                    projected_rhs,
                    rcond=None,
                )[0]

            cycle_candidate = x + basis[:, :krylov_size] @ coefficients
            candidate_residual = b - apply_matvec(cycle_candidate)
            candidate_norm = float(np.linalg.norm(candidate_residual))
            residual_history.append(candidate_norm)
            iterations += 1

            if candidate_norm <= target:
                x = np.ascontiguousarray(cycle_candidate)
                return make_result("converged", iterations, cycles)

            if happy_breakdown:
                x = np.ascontiguousarray(cycle_candidate)
                return make_result("breakdown", iterations, cycles)

        x = np.ascontiguousarray(cycle_candidate)

    return make_result("max_iterations", iterations, cycles)


def solve_face_dense_gmres(
    system: FaceDenseSystem,
    *,
    x0: np.ndarray | None = None,
    restart: int = 30,
    max_iterations: int | None = None,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    preconditioner: VectorOperator | None = None,
    reorthogonalize: bool = False,
    breakdown_tolerance: float | None = None,
) -> GMRESResult:
    """Solve a :class:`~hdgfem.assembly.face_dense.FaceDenseSystem`.

    The only matrix operation used is :func:`face_dense_matvec`; no COO, CSR,
    or scalar dense matrix is formed.
    """

    if not isinstance(system, FaceDenseSystem):
        raise TypeError("system must be a FaceDenseSystem")

    def operator(vector: np.ndarray) -> np.ndarray:
        return face_dense_matvec(system.blocks, system.neighbors, vector)

    return restarted_gmres(
        operator,
        system.rhs,
        x0=x0,
        restart=restart,
        max_iterations=max_iterations,
        rtol=rtol,
        atol=atol,
        preconditioner=preconditioner,
        reorthogonalize=reorthogonalize,
        breakdown_tolerance=breakdown_tolerance,
    )


__all__ = [
    "GMRESResult",
    "GMRESStatus",
    "VectorOperator",
    "restarted_gmres",
    "solve_face_dense_gmres",
]