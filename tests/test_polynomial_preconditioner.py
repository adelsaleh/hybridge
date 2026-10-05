from __future__ import annotations

import numpy as np
import pytest

from hybridge.linalg.polynomial import (
    PolynomialPreconditioner,
    arnoldi_factorization,
    build_polynomial_preconditioner,
    conjugate_root_blocks,
    harmonic_ritz_values,
    leja_order_conjugate_preserving,
)


def _explicit_inverse_polynomial(matrix: np.ndarray, roots: np.ndarray) -> np.ndarray:
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    product = identity.copy()
    polynomial = np.zeros_like(identity)
    for root in roots:
        polynomial += product / root
        product = (identity - matrix / root) @ product
    return polynomial


def test_real_root_recurrence_matches_explicit_matrix_polynomial() -> None:
    matrix = np.array(
        [[3.0, -0.5, 0.2], [0.3, 2.0, 0.1], [0.0, -0.4, 1.5]]
    )
    roots = np.array([4.0, 2.5, 1.2])
    vector = np.array([0.4, -1.0, 2.0])
    preconditioner = PolynomialPreconditioner(
        lambda x: matrix @ x,
        size=3,
        roots=roots,
    )
    expected = _explicit_inverse_polynomial(matrix, roots) @ vector
    np.testing.assert_allclose(
        preconditioner(vector), expected.real, rtol=2.0e-14, atol=2.0e-14
    )


def test_conjugate_pair_recurrence_matches_complex_arithmetic() -> None:
    matrix = np.array([[2.0, 0.4], [-0.7, 1.5]])
    root = 1.7 + 0.8j
    roots = np.array([root, np.conjugate(root)])
    vector = np.array([0.3, -1.2])
    preconditioner = PolynomialPreconditioner(
        lambda x: matrix @ x,
        size=2,
        roots=roots,
    )
    expected = _explicit_inverse_polynomial(matrix, roots) @ vector
    assert np.max(np.abs(expected.imag)) < 1.0e-14
    np.testing.assert_allclose(
        preconditioner(vector), expected.real, rtol=2.0e-14, atol=2.0e-14
    )


def test_hybrid_polynomial_matches_explicit_p_minv_a_minv() -> None:
    matrix = np.array([[4.0, -1.0, 0.2], [0.5, 3.0, -0.4], [0.0, 0.7, 2.0]])
    inverse_mass = np.diag([0.5, 0.25, 0.8])
    effective = inverse_mass @ matrix
    roots = np.array([3.5, 1.4 + 0.6j, 1.4 - 0.6j])
    vector = np.array([1.0, -0.3, 0.7])
    preconditioner = PolynomialPreconditioner(
        lambda x: matrix @ x,
        size=3,
        roots=roots,
        base_preconditioner=lambda x: inverse_mass @ x,
    )
    expected = _explicit_inverse_polynomial(effective, roots) @ (inverse_mass @ vector)
    np.testing.assert_allclose(
        preconditioner(vector), expected.real, rtol=3.0e-14, atol=3.0e-14
    )
    assert preconditioner.matvecs_per_application == roots.size
    assert preconditioner.base_preconditioner_calls_per_application == roots.size + 1


def test_arnoldi_relation_and_full_dimension_harmonic_ritz_values() -> None:
    matrix = np.array(
        [[4.0, 1.0, 0.0, 0.2], [0.0, 3.0, -0.5, 0.0], [0.1, 0.0, 2.0, 0.7], [0.0, 0.2, -0.1, 1.5]]
    )
    initial = np.array([1.0, 0.3, -0.7, 0.8])
    result = arnoldi_factorization(
        lambda x: matrix @ x,
        size=4,
        dimension=4,
        initial_vector=initial,
        reorthogonalize=True,
    )
    dimension = result.effective_dimension
    left = matrix @ result.basis[:dimension].T
    right = result.basis[: dimension + 1].T @ result.hessenberg
    np.testing.assert_allclose(left, right, rtol=2.0e-13, atol=2.0e-13)
    assert result.orthogonality_error < 2.0e-13

    values = harmonic_ritz_values(result.hessenberg)
    np.testing.assert_allclose(
        np.sort_complex(values),
        np.sort_complex(np.linalg.eigvals(matrix)),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_leja_order_preserves_values_and_conjugate_adjacency() -> None:
    roots = np.array([2.0, 0.8 - 1.1j, 4.0, 0.8 + 1.1j, 1.2])
    ordered = leja_order_conjugate_preserving(roots)
    np.testing.assert_allclose(np.sort_complex(ordered), np.sort_complex(roots))
    blocks = conjugate_root_blocks(ordered)
    assert sum(block.degree for block in blocks) == roots.size
    for index, value in enumerate(ordered):
        if abs(value.imag) > 1.0e-12:
            assert index + 1 < ordered.size or index > 0
            neighbor = ordered[index + 1] if value.imag > 0 else ordered[index - 1]
            np.testing.assert_allclose(neighbor, np.conjugate(value))


def test_builder_is_deterministic_and_rejects_zero_roots() -> None:
    matrix = np.diag([1.0, 2.0, 3.0, 4.0]) + 0.05 * np.ones((4, 4))
    first = build_polynomial_preconditioner(
        lambda x: matrix @ x,
        size=4,
        degree=3,
        seed=831,
    )
    second = build_polynomial_preconditioner(
        lambda x: matrix @ x,
        size=4,
        degree=3,
        seed=831,
    )
    np.testing.assert_allclose(first.roots, second.roots, rtol=0.0, atol=0.0)
    with pytest.raises(np.linalg.LinAlgError, match="zero"):
        PolynomialPreconditioner(
            lambda x: matrix @ x,
            size=4,
            roots=np.array([1.0, 0.0]),
        )
