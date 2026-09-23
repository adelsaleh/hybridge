"""Small dense diagnostics for the proposed nonsymmetric HDG p hierarchy.

No HDG assembly, simulation, accelerator initialization, or compilation occurs.
The terminal p-level uses dense LU, NOT AMG. Fixed linear cycles permit SciPy
GMRES on A B; this does not test a production flexible-GMRES implementation.
Run from the repository root with ``python -B scripts/dev/check_nonsymmetric_pmg_algebra.py``.
"""
from __future__ import annotations

import json

import numpy as np
import scipy
from scipy.sparse.linalg import gmres


def split_indices(size, block_size, coarse_size):
    face_dofs = np.arange(size).reshape(-1, block_size)
    return face_dofs[:, coarse_size:].ravel(), face_dofs[:, :coarse_size].ravel()


def transfers(matrix, block_size, coarse_size, local):
    """Inject low modes, optionally extending on one-ring high-mode patches."""
    size = matrix.shape[0]
    faces = size // block_size
    high, low = split_indices(size, block_size, coarse_size)
    prolong = np.eye(size)[:, low]
    restrict = prolong.T.copy()
    if not local:
        return restrict, prolong
    blocks = matrix.reshape(faces, block_size, faces, block_size)
    norms = np.linalg.norm(blocks, axis=(1, 3))
    for face in range(faces):
        neighbors = np.flatnonzero((norms[face] != 0) | (norms[:, face] != 0))
        local_high = np.array([
            f * block_size + mode for f in neighbors
            for mode in range(coarse_size, block_size)
        ])
        coarse_columns = np.arange(face * coarse_size, (face + 1) * coarse_size)
        local_low = low[coarse_columns]
        patch = matrix[np.ix_(local_high, local_high)]
        prolong[np.ix_(local_high, coarse_columns)] = -np.linalg.solve(
            patch, matrix[np.ix_(local_high, local_low)]
        )
        restrict[np.ix_(coarse_columns, local_high)] = -np.linalg.solve(
            patch.T, matrix[np.ix_(local_low, local_high)].T
        ).T
    return restrict, prolong


def cycle_matrix(matrix, block_size, local, diagnostics):
    """Materialize a tiny linear V-cycle with two block-Jacobi F sweeps."""
    size = len(matrix)
    if block_size == 1:
        return np.linalg.solve(matrix, np.eye(size))
    coarse_size = max(1, block_size // 2)
    high, _ = split_indices(size, block_size, coarse_size)
    one_sweep = np.zeros_like(matrix)
    for indices in high.reshape(-1, block_size - coarse_size):
        one_sweep[np.ix_(indices, indices)] = np.linalg.solve(
            matrix[np.ix_(indices, indices)], np.eye(len(indices))
        )
    smooth = 2 * one_sweep - one_sweep @ matrix @ one_sweep
    restrict, prolong = transfers(matrix, block_size, coarse_size, local)
    coarse = restrict @ matrix @ prolong
    diagnostics.append({
        "size": size,
        "block_size": block_size,
        "coarse_condition_2": float(np.linalg.cond(coarse)),
        "restriction_transpose_defect": float(np.linalg.norm(restrict - prolong.T)),
        "coarse_numerical_nonzeros": int(np.count_nonzero(np.abs(coarse) > 1e-14)),
    })
    coarse_inverse = cycle_matrix(coarse, coarse_size, local, diagnostics)
    partial = smooth + prolong @ coarse_inverse @ restrict @ (np.eye(size) - matrix @ smooth)
    return partial + smooth @ (np.eye(size) - matrix @ partial)


def identity_checks():
    rng = np.random.default_rng(1709)
    matrix = rng.normal(size=(8, 8)) + 4 * np.eye(8)
    high, low = split_indices(8, 4, 2)
    hh = matrix[np.ix_(high, high)]
    hc = matrix[np.ix_(high, low)]
    ch = matrix[np.ix_(low, high)]
    cc = matrix[np.ix_(low, low)]
    prolong = np.eye(8)[:, low]
    prolong[high] = -np.linalg.solve(hh, hc)
    restrict = np.eye(8)[low]
    restrict[:, high] = -np.linalg.solve(hh.T, ch.T).T
    schur = cc - ch @ np.linalg.solve(hh, hc)
    high_solve = np.zeros_like(matrix)
    high_solve[np.ix_(high, high)] = np.linalg.solve(hh, np.eye(len(high)))
    inverse = high_solve + prolong @ np.linalg.solve(schur, restrict)
    errors = {
        "ideal_RAP_minus_S": float(np.linalg.norm(restrict @ matrix @ prolong - schur)),
        "ideal_RA_high": float(np.linalg.norm((restrict @ matrix)[:, high])),
        "ideal_AP_high": float(np.linalg.norm((matrix @ prolong)[high])),
        "ideal_inverse_residual": float(np.linalg.norm(matrix @ inverse - np.eye(8))),
    }
    assert max(errors.values()) < 1e-11, errors

    approximate = np.diag(1 / np.diag(hh))
    approximate_p = prolong.copy()
    approximate_r = restrict.copy()
    approximate_p[high] = -approximate @ hc
    approximate_r[:, high] = -ch @ approximate
    rap = approximate_r @ matrix @ approximate_p
    expanded = cc - 2 * ch @ approximate @ hc + ch @ approximate @ hh @ approximate @ hc
    errors["approximate_RAP_expansion_error"] = float(np.linalg.norm(rap - expanded))
    errors["approximate_RAP_minus_naive_S"] = float(np.linalg.norm(rap - (cc - ch @ approximate @ hc)))
    assert errors["approximate_RAP_expansion_error"] < 1e-11
    assert errors["approximate_RAP_minus_naive_S"] > 1e-3

    transform = np.diag(np.geomspace(.5, 2., len(matrix)))
    modal_inverse = np.linalg.solve(transform.T @ matrix @ transform, np.eye(len(matrix)))
    errors["coordinate_mapped_inverse_residual"] = float(
        np.linalg.norm(matrix @ transform @ modal_inverse @ transform.T - np.eye(len(matrix)))
    )
    assert errors["coordinate_mapped_inverse_residual"] < 1e-11

    # A nonsingular trace operator with singular injected coarse matrix AND
    # singular symmetric part. C is mode zero, H is mode one.
    counterexample = np.array([[0., 1.], [-1., 2.]])
    restrict, prolong = transfers(counterexample, 2, 1, local=True)
    errors["counterexample_determinant"] = float(np.linalg.det(counterexample))
    errors["injected_coarse_value"] = float(counterexample[0, 0])
    errors["petrov_coarse_value"] = float((restrict @ counterexample @ prolong)[0, 0])
    assert np.linalg.matrix_rank(counterexample + counterexample.T) == 1
    assert abs(errors["petrov_coarse_value"] - 0.5) < 1e-14
    inverse = cycle_matrix(counterexample, 2, True, [])
    errors["counterexample_inverse_residual"] = float(
        np.linalg.norm(counterexample @ inverse - np.eye(2))
    )
    assert errors["counterexample_inverse_residual"] < 1e-14

    indefinite_part = np.array([[1., 4.], [-1., -1.]])
    assert np.linalg.eigvalsh(indefinite_part + indefinite_part.T)[0] < 0
    inverse = cycle_matrix(indefinite_part, 2, True, [])
    errors["indefinite_symmetric_part_inverse_residual"] = float(
        np.linalg.norm(indefinite_part @ inverse - np.eye(2))
    )
    assert errors["indefinite_symmetric_part_inverse_residual"] < 1e-14

    # An invertible full face block may also have singular A_HH. This p split
    # must be rejected or changed; no AIR transfer can invert the chosen A_HH.
    bad_split = np.array([[1., 1.], [1., 0.]])
    assert np.linalg.det(bad_split) != 0 and bad_split[1, 1] == 0
    try:
        transfers(bad_split, 2, 1, local=True)
    except np.linalg.LinAlgError:
        errors["singular_high_split_rejected"] = True
    else:
        raise AssertionError("singular high-mode block was not rejected")
    return errors


def algebraic_cases():
    """Synthetic face-block matrices, not assembled PDE or HDG benchmarks."""
    faces, modes = 12, 4
    shift = np.roll(np.eye(faces), 1, axis=1)
    laplacian = 2 * np.eye(faces) - shift - shift.T
    modal = np.diag([1., 2., 4., 8.])
    mixing = np.array([[1., .2, .1, 0.], [.2, 1., .15, .1],
                       [.1, .15, 1., .2], [0., .1, .2, 1.]])
    base = np.kron(np.eye(faces), modal) + np.kron(laplacian, mixing)
    yield "symmetric_control", base, modes
    directed = np.kron(np.eye(faces) - shift, mixing)
    yield "directed_cycle", base + 5 * directed, modes
    rotation = np.array([[0., 4., 0., 1.], [-4., 0., 2., 0.],
                         [0., -2., 0., 3.], [-1., 0., -3., 0.]])
    yield "skew_coupled", base + np.kron(np.eye(faces), rotation), modes
    # Same dense face blocks, with algebraically imposed unequal edge strengths.
    incidence = np.eye(faces) - shift
    weights = np.diag(np.tile([1., 100., .1], 4))
    unequal = incidence.T @ weights @ incidence
    yield "unequal_couplings", (
        np.kron(np.eye(faces), modal) + np.kron(unequal, mixing) + 2 * directed
    ), modes


def solve_check(name, matrix, block_size, local):
    levels = []
    inverse = cycle_matrix(matrix, block_size, local, levels)
    reference = np.sin(np.arange(len(matrix)) + .3)
    rhs = matrix @ reference
    history = []
    transformed, info = gmres(
        matrix @ inverse, rhs, rtol=1e-10, atol=0., restart=20, maxiter=10,
        callback=lambda residual: history.append(float(residual)), callback_type="pr_norm",
    )
    solution = inverse @ transformed
    residual = float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
    return {
        "case": name,
        "transfer": "local_petrov" if local else "injection",
        "size": len(matrix),
        "gmres_iterations": len(history),
        "gmres_info": int(info),
        "true_relative_residual": residual,
        "relative_solution_error": float(np.linalg.norm(solution - reference) / np.linalg.norm(reference)),
        "passed_residual_check": bool(info == 0 and residual < 2e-10),
        "levels": levels,
    }


def main():
    checks = identity_checks()
    samples = [solve_check(name, matrix, size, local)
               for name, matrix, size in algebraic_cases() for local in (False, True)]
    report = {
        "scope": "dense algebra diagnostics; exact terminal solve, no AMG, HDG assembly, or GPU",
        "numpy": np.__version__, "scipy": scipy.__version__,
        "identities": checks, "samples": samples,
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    # Failure of the candidate is evidence to retain, not a reason to tune the
    # synthetic cases. Comparator failures are recorded without aborting.
    if not all(row["passed_residual_check"] for row in samples if row["transfer"] == "local_petrov"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
