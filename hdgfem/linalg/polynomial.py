"""Harmonic-Ritz polynomial preconditioners for matrix-free operators.

This CPU module is the numerical reference for the CUDA implementation.  For
ordered roots ``theta_1, ..., theta_P`` it applies

    p(B) = sum_j [prod_{k<j}(I - B/theta_k)] / theta_j,

where ``B=A`` for a pure polynomial preconditioner and ``B=M^{-1}A`` for the
hybrid form ``p(M^{-1}A)M^{-1}``.

Complex-conjugate roots are evaluated in paired real arithmetic.  The paired
recurrence is obtained by expanding the product of the two shifted operators;
it intentionally contains both the linear ``Bq`` and quadratic ``B^2q`` terms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..assembly.face_dense import FaceDenseSystem, face_dense_matvec

VectorOperator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ArnoldiResult:
    basis: np.ndarray
    hessenberg: np.ndarray
    requested_dimension: int
    effective_dimension: int
    breakdown: bool
    orthogonality_error: float


@dataclass(frozen=True)
class PolynomialSetup:
    arnoldi: ArnoldiResult
    harmonic_ritz_values: np.ndarray
    ordered_roots: np.ndarray
    seed: int

    @property
    def degree(self) -> int:
        return int(self.ordered_roots.size)


@dataclass(frozen=True)
class RootBlock:
    first: complex
    second: complex | None = None

    @property
    def is_real(self) -> bool:
        return self.second is None

    @property
    def degree(self) -> int:
        return 1 if self.second is None else 2


def _validate_operator_output(
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


def _is_nearly_real(value: complex, tolerance: float) -> bool:
    return abs(value.imag) <= tolerance * max(1.0, abs(value))


def conjugate_root_blocks(
    roots: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> list[RootBlock]:
    """Group a real root list into real entries and conjugate pairs.

    Small conjugacy defects produced by the eigensolver are removed by
    averaging each pair.  A complex root without a matching conjugate is an
    error because the device recurrence must remain real.
    """

    values = np.asarray(roots, dtype=np.complex128)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("roots must be a non-empty one-dimensional array")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("roots contain non-finite values")

    unused = list(range(values.size))
    blocks: list[RootBlock] = []
    while unused:
        index = unused.pop(0)
        value = complex(values[index])
        if _is_nearly_real(value, tolerance):
            blocks.append(RootBlock(complex(float(value.real), 0.0)))
            continue

        target = np.conjugate(value)
        best_position: int | None = None
        best_error = np.inf
        for position, candidate_index in enumerate(unused):
            error = abs(complex(values[candidate_index]) - target)
            if error < best_error:
                best_error = error
                best_position = position
        allowed = tolerance * max(1.0, abs(value))
        if best_position is None or best_error > allowed:
            raise ValueError(f"complex root {value!r} has no conjugate partner")

        partner = complex(values[unused.pop(best_position)])
        real_part = 0.5 * (value.real + partner.real)
        imag_part = 0.5 * (abs(value.imag) + abs(partner.imag))
        positive = complex(real_part, imag_part)
        blocks.append(RootBlock(positive, np.conjugate(positive)))
    return blocks


def leja_order_conjugate_preserving(
    roots: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return a deterministic greedy Leja-style root ordering.

    Complex conjugates are kept adjacent so that polynomial application can
    use one real quadratic recurrence for each pair.
    """

    remaining = list(conjugate_root_blocks(roots, tolerance=tolerance))
    ordered: list[complex] = []
    tiny = np.finfo(np.float64).tiny

    while remaining:
        best_index = 0
        best_score = -np.inf
        for block_index, block in enumerate(remaining):
            candidates = [block.first]
            if block.second is not None:
                candidates.append(block.second)
            scores: list[float] = []
            for candidate in candidates:
                if not ordered:
                    score = np.log(max(abs(candidate), tiny))
                else:
                    score = float(
                        np.sum(
                            [
                                np.log(max(abs(candidate - chosen), tiny))
                                for chosen in ordered
                            ]
                        )
                    )
                scores.append(score)
            score = max(scores)
            if score > best_score:
                best_score = score
                best_index = block_index

        chosen = remaining.pop(best_index)
        ordered.append(chosen.first)
        if chosen.second is not None:
            ordered.append(chosen.second)

    return np.ascontiguousarray(np.asarray(ordered, dtype=np.complex128))


def arnoldi_factorization(
    operator: VectorOperator,
    *,
    size: int,
    dimension: int,
    initial_vector: np.ndarray,
    reorthogonalize: bool = True,
    breakdown_tolerance: float | None = None,
) -> ArnoldiResult:
    """Compute a fixed-length CPU Arnoldi factorization."""

    if not callable(operator):
        raise TypeError("operator must be callable")
    if isinstance(size, bool) or int(size) != size or size <= 0:
        raise ValueError("size must be a positive integer")
    if (
        isinstance(dimension, bool)
        or int(dimension) != dimension
        or dimension <= 0
        or dimension > size
    ):
        raise ValueError("dimension must be an integer in [1, size]")
    size, dimension = int(size), int(dimension)
    if breakdown_tolerance is None:
        breakdown_tolerance = 100.0 * np.finfo(np.float64).eps
    if breakdown_tolerance < 0.0 or not np.isfinite(breakdown_tolerance):
        raise ValueError("breakdown_tolerance must be finite and non-negative")

    initial = np.asarray(initial_vector, dtype=np.float64)
    if initial.size != size or not np.all(np.isfinite(initial)):
        raise ValueError("initial_vector has an invalid size or non-finite values")
    initial = np.ascontiguousarray(initial.reshape(-1))
    initial_norm = float(np.linalg.norm(initial))
    if initial_norm == 0.0:
        raise ValueError("initial_vector must be nonzero")

    basis = np.zeros((dimension + 1, size), dtype=np.float64)
    hessenberg = np.zeros((dimension + 1, dimension), dtype=np.float64)
    basis[0] = initial / initial_norm
    effective_dimension = dimension
    breakdown = False

    for column in range(dimension):
        work = _validate_operator_output(
            operator(basis[column]), size=size, name="operator"
        )
        work_before = float(np.linalg.norm(work))
        passes = 2 if reorthogonalize else 1
        for _ in range(passes):
            coefficients = basis[: column + 1] @ work
            hessenberg[: column + 1, column] += coefficients
            work -= basis[: column + 1].T @ coefficients

        next_norm = float(np.linalg.norm(work))
        hessenberg[column + 1, column] = next_norm
        threshold = breakdown_tolerance * max(work_before, 1.0)
        if next_norm <= threshold:
            effective_dimension = column + 1
            breakdown = True
            break
        basis[column + 1] = work / next_norm

    used_basis = np.ascontiguousarray(basis[: effective_dimension + 1])
    used_hessenberg = np.ascontiguousarray(
        hessenberg[: effective_dimension + 1, :effective_dimension]
    )
    active_basis = used_basis[:effective_dimension]
    gram = active_basis @ active_basis.T
    orthogonality_error = float(
        np.linalg.norm(gram - np.eye(effective_dimension), ord=np.inf)
    )
    return ArnoldiResult(
        basis=used_basis,
        hessenberg=used_hessenberg,
        requested_dimension=dimension,
        effective_dimension=effective_dimension,
        breakdown=breakdown,
        orthogonality_error=orthogonality_error,
    )


def harmonic_ritz_values(
    hessenberg: np.ndarray,
    *,
    singular_tolerance: float | None = None,
) -> np.ndarray:
    r"""Extract harmonic Ritz values from ``Hbar`` of shape ``(m+1,m)``."""

    hbar = np.asarray(hessenberg, dtype=np.float64)
    if hbar.ndim != 2 or hbar.shape[0] != hbar.shape[1] + 1:
        raise ValueError("hessenberg must have shape (m + 1, m)")
    if hbar.shape[1] == 0 or not np.all(np.isfinite(hbar)):
        raise ValueError("hessenberg must be non-empty and finite")
    dimension = int(hbar.shape[1])
    square = np.ascontiguousarray(hbar[:dimension, :dimension])
    if singular_tolerance is None:
        singular_tolerance = 100.0 * np.finfo(np.float64).eps
    if singular_tolerance < 0.0 or not np.isfinite(singular_tolerance):
        raise ValueError("singular_tolerance must be finite and non-negative")

    singular_values = np.linalg.svd(square, compute_uv=False)
    scale = max(float(singular_values[0]), 1.0)
    if float(singular_values[-1]) <= singular_tolerance * scale:
        raise np.linalg.LinAlgError(
            "Arnoldi H matrix is singular or too ill-conditioned for harmonic Ritz extraction"
        )

    last = np.zeros(dimension, dtype=np.float64)
    last[-1] = 1.0
    inverse_transpose_last = np.linalg.solve(square.T, last)
    subdiagonal = float(hbar[dimension, dimension - 1])
    harmonic_matrix = square + subdiagonal**2 * np.outer(
        inverse_transpose_last, last
    )
    values = np.linalg.eigvals(harmonic_matrix).astype(np.complex128, copy=False)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("harmonic Ritz extraction produced non-finite values")
    return np.ascontiguousarray(values)


class PolynomialPreconditioner:
    """CPU action ``p(M^{-1}A)M^{-1}`` evaluated in real arithmetic."""

    def __init__(
        self,
        matvec: VectorOperator,
        *,
        size: int,
        roots: np.ndarray,
        base_preconditioner: VectorOperator | None = None,
        pair_tolerance: float = 1.0e-10,
        setup: PolynomialSetup | None = None,
    ) -> None:
        if not callable(matvec):
            raise TypeError("matvec must be callable")
        if base_preconditioner is not None and not callable(base_preconditioner):
            raise TypeError("base_preconditioner must be callable or None")
        if isinstance(size, bool) or int(size) != size or size <= 0:
            raise ValueError("size must be a positive integer")
        roots_array = np.asarray(roots, dtype=np.complex128)
        if roots_array.ndim != 1 or roots_array.size == 0:
            raise ValueError("roots must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(roots_array)):
            raise ValueError("roots contain non-finite values")
        root_scale = max(float(np.max(np.abs(roots_array))), 1.0)
        zero_tolerance = 100.0 * np.finfo(np.float64).eps * root_scale
        if np.any(np.abs(roots_array) <= zero_tolerance):
            raise np.linalg.LinAlgError("polynomial roots cannot be zero")

        self.matvec = matvec
        self.base_preconditioner = base_preconditioner
        self.size = int(size)
        self.roots = np.ascontiguousarray(roots_array)
        self.root_blocks = tuple(
            conjugate_root_blocks(self.roots, tolerance=pair_tolerance)
        )
        self.setup = setup

    @property
    def degree(self) -> int:
        return int(self.roots.size)

    @property
    def matvecs_per_application(self) -> int:
        return self.degree

    @property
    def base_preconditioner_calls_per_application(self) -> int:
        return 0 if self.base_preconditioner is None else self.degree + 1

    def _base(self, vector: np.ndarray) -> np.ndarray:
        if self.base_preconditioner is None:
            return np.ascontiguousarray(vector.copy())
        return _validate_operator_output(
            self.base_preconditioner(vector),
            size=self.size,
            name="base_preconditioner",
        )

    def _effective(self, vector: np.ndarray) -> np.ndarray:
        applied = _validate_operator_output(
            self.matvec(vector), size=self.size, name="matvec"
        )
        return self._base(applied)

    def apply(self, vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float64)
        if array.size != self.size or not np.all(np.isfinite(array)):
            raise ValueError("vector has an invalid size or non-finite values")
        shape = array.shape
        q = self._base(np.ascontiguousarray(array.reshape(-1)))
        result = np.zeros(self.size, dtype=np.float64)

        for block in self.root_blocks:
            if block.is_real:
                theta = float(block.first.real)
                bq = self._effective(q)
                result += q / theta
                q -= bq / theta
                continue

            a = float(block.first.real)
            b = float(abs(block.first.imag))
            denominator = a * a + b * b
            bq = self._effective(q)
            b2q = self._effective(bq)
            result += (2.0 * a * q - bq) / denominator
            q += (-2.0 * a * bq + b2q) / denominator

        if not np.all(np.isfinite(result)):
            raise FloatingPointError("polynomial application produced non-finite values")
        return np.ascontiguousarray(result.reshape(shape))

    __call__ = apply


def build_polynomial_preconditioner(
    matvec: VectorOperator,
    *,
    size: int,
    degree: int,
    base_preconditioner: VectorOperator | None = None,
    seed: int = 1729,
    initial_vector: np.ndarray | None = None,
    reorthogonalize: bool = True,
    breakdown_tolerance: float | None = None,
    pair_tolerance: float = 1.0e-10,
) -> PolynomialPreconditioner:
    """Build roots from one Arnoldi cycle and return the CPU preconditioner."""

    if isinstance(degree, bool) or int(degree) != degree or degree <= 0 or degree > size:
        raise ValueError("degree must be an integer in [1, size]")
    size, degree = int(size), int(degree)

    def base(vector: np.ndarray) -> np.ndarray:
        if base_preconditioner is None:
            return np.ascontiguousarray(vector.copy())
        return _validate_operator_output(
            base_preconditioner(vector), size=size, name="base_preconditioner"
        )

    def effective(vector: np.ndarray) -> np.ndarray:
        applied = _validate_operator_output(matvec(vector), size=size, name="matvec")
        return base(applied)

    if initial_vector is None:
        initial_vector = np.random.default_rng(seed).standard_normal(size)
    arnoldi = arnoldi_factorization(
        effective,
        size=size,
        dimension=degree,
        initial_vector=initial_vector,
        reorthogonalize=reorthogonalize,
        breakdown_tolerance=breakdown_tolerance,
    )
    values = harmonic_ritz_values(arnoldi.hessenberg)
    roots = leja_order_conjugate_preserving(values, tolerance=pair_tolerance)
    setup = PolynomialSetup(
        arnoldi=arnoldi,
        harmonic_ritz_values=values,
        ordered_roots=roots,
        seed=int(seed),
    )
    return PolynomialPreconditioner(
        matvec,
        size=size,
        roots=roots,
        base_preconditioner=base_preconditioner,
        pair_tolerance=pair_tolerance,
        setup=setup,
    )


def build_face_dense_polynomial_preconditioner(
    system: FaceDenseSystem,
    *,
    degree: int,
    base_preconditioner: VectorOperator | None = None,
    seed: int = 1729,
    initial_vector: np.ndarray | None = None,
    reorthogonalize: bool = True,
    breakdown_tolerance: float | None = None,
    pair_tolerance: float = 1.0e-10,
) -> PolynomialPreconditioner:
    """Convenience builder for a :class:`FaceDenseSystem`."""

    if not isinstance(system, FaceDenseSystem):
        raise TypeError("system must be a FaceDenseSystem")

    def matvec(vector: np.ndarray) -> np.ndarray:
        return face_dense_matvec(system.blocks, system.neighbors, vector)

    return build_polynomial_preconditioner(
        matvec,
        size=system.num_dofs,
        degree=degree,
        base_preconditioner=base_preconditioner,
        seed=seed,
        initial_vector=initial_vector,
        reorthogonalize=reorthogonalize,
        breakdown_tolerance=breakdown_tolerance,
        pair_tolerance=pair_tolerance,
    )


__all__ = [
    "ArnoldiResult",
    "PolynomialPreconditioner",
    "PolynomialSetup",
    "RootBlock",
    "arnoldi_factorization",
    "build_face_dense_polynomial_preconditioner",
    "build_polynomial_preconditioner",
    "conjugate_root_blocks",
    "harmonic_ritz_values",
    "leja_order_conjugate_preserving",
]
