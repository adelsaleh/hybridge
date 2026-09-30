from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from hdgfem.hdg.gram import assemble_hdg_gram, build_condensed_hdg_gram_inverse
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace


def test_condensed_hdg_gram_inverse_matches_sparse_direct() -> None:
    mesh = rectangle_mesh(2, 2)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    gram = assemble_hdg_gram(space, sigma=7.0)
    inverse = build_condensed_hdg_gram_inverse(space, sigma=7.0, cg_rtol=1.0e-12)

    rng = np.random.default_rng(1234)
    rhs = rng.standard_normal(gram.ndof)

    condensed, diagnostics = inverse.solve(rhs)
    direct = spsolve(gram.matrix, rhs)

    assert diagnostics.info == 0
    assert diagnostics.relative_residual < 1.0e-9
    np.testing.assert_allclose(condensed, direct, rtol=1.0e-8, atol=1.0e-9)


def test_condensed_hdg_dual_norm_matches_energy_identity() -> None:
    mesh = rectangle_mesh(2, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    gram = assemble_hdg_gram(space, sigma=4.0)
    inverse = build_condensed_hdg_gram_inverse(space, sigma=4.0, cg_rtol=1.0e-12)

    rng = np.random.default_rng(5678)
    state = rng.standard_normal(gram.ndof)
    residual = gram.matrix @ state

    dual_squared, diagnostics = inverse.dual_norm_squared(np.asarray(residual))
    energy = float(state @ residual)

    assert diagnostics.info == 0
    assert diagnostics.relative_residual < 1.0e-9
    np.testing.assert_allclose(dual_squared, energy, rtol=1.0e-8, atol=1.0e-9)


def test_configurable_krylov_gram_inverse_matches_sparse_direct() -> None:
    from hdgfem.hdg.gram import build_krylov_hdg_gram_inverse

    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 1, basis_type="dub_orth")
    gram = assemble_hdg_gram(space, sigma=3.0, jump_weight="scaled")
    inverse = build_krylov_hdg_gram_inverse(gram, preconditioner="jacobi")
    rhs = np.random.default_rng(91).standard_normal(gram.ndof)

    solution, diagnostics = inverse.solve(rhs, method="cg", rtol=1.0e-12)

    assert diagnostics.info == 0
    np.testing.assert_allclose(solution, spsolve(gram.matrix, rhs), rtol=1.0e-9, atol=1.0e-10)
