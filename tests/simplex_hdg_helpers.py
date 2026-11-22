"""Test helpers: HDG local blocks composed from reference-element tables.

These helpers are test oracles for the reference integrals of
:class:`hybridge.core.quadrature.ReferenceElementData` on the triangle and the
tetrahedron. They form dense local blocks in the sign convention of TODO.md,
"Form-Driven HDG Assembly",

    A u - B uhat = F,        D uhat - C u = G,        S = D - C A^{-1} B,

with numpy, and are not a second implementation of the static condensation of
:mod:`hybridge.hdg.condensation`, which a tetrahedral ``DGMesh`` will use later.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from hybridge.core.quadrature import ReferenceElementData, _evaluate_basis, _evaluate_simplex_basis
from hybridge.core.simplex.reference import face_permutations, permute_face_points


def monomial_exponents(dim: int, degree: int) -> list[tuple[int, ...]]:
    """Return every exponent tuple ``a`` of length ``dim`` with ``|a| <= degree``."""
    return [a for a in product(range(degree + 1), repeat=dim) if sum(a) <= degree]


def random_polynomial(dim: int, degree: int, seed: int):
    """Return a callable ``u(points (n, dim)) -> (n,)``: a polynomial of total degree ``degree`` with random coefficients."""
    rng = np.random.default_rng(seed)
    exponents = monomial_exponents(dim, degree)
    coefficients = rng.standard_normal(len(exponents))

    def evaluate(points: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at ``points`` of shape ``(n, dim)``."""
        points = np.asarray(points, dtype=np.float64)
        values = np.zeros(points.shape[0])
        for c, a in zip(coefficients, exponents):
            values += c * np.prod(points ** np.asarray(a), axis=1)
        return values

    return evaluate


def random_interior_points(dim: int, count: int, seed: int) -> np.ndarray:
    """Return ``count`` random points inside the reference simplex of dimension ``dim``, shape ``(count, dim)``."""
    rng = np.random.default_rng(seed)
    bary = rng.dirichlet(np.ones(dim + 1), size=count)
    return -1.0 + 2.0 * bary[:, 1:]


def volume_load(reference: ReferenceElementData, source) -> np.ndarray:
    """Return ``F_i = int_K^ f phi_i`` on the reference element for a callable ``f(points) -> (n,)``."""
    points, weights = reference._generic_volume_rule
    basis = np.asarray(_evaluate_basis(reference.basis_type, reference.order, points), dtype=np.float64)
    return basis.T @ (weights * source(points))


def trace_coefficients(reference: ReferenceElementData, trace_basis: str, function, orientations=None) -> np.ndarray:
    """Return the L2 projection of ``function`` onto the trace space of every face, as the element sees it.

    For face ``f`` with orientation ``o = orientations[f]`` (default 0) the element
    sees the trace ``sum_a uhat_a mu_a(sigma_o s)`` at the reference-face point
    ``s``. The coefficients solve ``G uhat = int_F^ mu(sigma_o s) g(F_f s) ds``; the
    trace mass ``G`` does not depend on ``o`` because a vertex permutation
    preserves the face measure. Returns shape ``((dim + 1) * n_t,)``, face-major.
    """
    tables = reference.trace_tables(trace_basis)
    points, weights = reference._generic_face_rule
    permutations_ = face_permutations(reference.dim)
    orientations = [0] * reference.num_faces if orientations is None else list(orientations)
    blocks = []
    for f in range(reference.num_faces):
        sigma = permutations_[orientations[f]]
        mu = np.asarray(
            _evaluate_simplex_basis(trace_basis, reference.dim - 1, reference.order,
                                    permute_face_points(sigma, points), gradients=False),
            dtype=np.float64,
        ).T
        rhs = mu @ (weights * function(reference.simplex.face_map(f, points)))
        blocks.append(np.linalg.solve(tables.trace_mass.astype(np.float64), rhs))
    return np.concatenate(blocks)


def reaction_blocks(
        reference: ReferenceElementData,
        trace_basis: str,
        *,
        alpha: float,
        tau: float,
        orientations=None,
        jacobian: float = 1.0,
        face_ratios=None,
):
    """Return ``(A, B, C, D)`` of the HDG discretization of ``alpha u = f`` on one element.

    The local problem is ``(alpha u, v)_K + <tau (u - uhat), v>_dK = (f, v)_K`` and the
    transmission row is ``<tau (u - uhat), mu>_dK``. With ``r_f`` the face-measure
    ratios (default: those of the reference element) and ``J`` the volume Jacobian
    (default 1, the reference element):

    ``A = alpha J M + tau sum_f r_f E^f`` (``n_u x n_u``),
    ``B[:, f] = tau r_f element_trace_coupling[o_f, f]`` (``n_u x (d+1) n_t``),
    ``C[f, :] = tau r_f trace_element_coupling[o_f (d+1) + f]`` (``(d+1) n_t x n_u``),
    ``D = blockdiag(tau r_f G)``.
    """
    tables = reference.trace_tables(trace_basis)
    num_faces = reference.num_faces
    ratios = reference.face_measure_ratios if face_ratios is None else np.asarray(face_ratios)
    orientations = [0] * num_faces if orientations is None else list(orientations)
    mass = reference.exact_mass.astype(np.float64)
    face_mass = reference.face_mass.astype(np.float64)
    coupling = tables.element_trace_coupling.astype(np.float64)
    lift = tables.trace_element_coupling.astype(np.float64)
    trace_mass = tables.trace_mass.astype(np.float64)
    n_t = trace_mass.shape[0]
    a = alpha * jacobian * mass + tau * np.einsum("f,fij->ij", ratios, face_mass)
    b = np.concatenate([tau * ratios[f] * coupling[orientations[f], f] for f in range(num_faces)], axis=1)
    c = np.concatenate([tau * ratios[f] * lift[orientations[f] * num_faces + f] for f in range(num_faces)], axis=0)
    d = np.zeros((num_faces * n_t, num_faces * n_t))
    for f in range(num_faces):
        d[f * n_t:(f + 1) * n_t, f * n_t:(f + 1) * n_t] = tau * ratios[f] * trace_mass
    return a, b, c, d


def schur_complement(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Return ``S = D - C A^{-1} B`` with a dense solve."""
    return d - c @ np.linalg.solve(a, b)


def block_diagonal(block: np.ndarray, count: int) -> np.ndarray:
    """Return the block-diagonal matrix with ``count`` copies of ``block``."""
    n = block.shape[0]
    out = np.zeros((count * n, count * n))
    for k in range(count):
        out[k * n:(k + 1) * n, k * n:(k + 1) * n] = block
    return out
