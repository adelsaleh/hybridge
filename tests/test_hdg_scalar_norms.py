"""Analytic and small-matrix checks for scalar HDG norms; no PDE solves."""
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge.hdg.gram import ScalarHDGGram, assemble_hdg_gram
from hybridge.core.field_ops import project_callable_to_trace
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.diagnostics.errors import evaluate_hdg_scalar_error


def space_for_norms(scale=1):
    return DGSpace(rectangle_mesh(2, 1, xlim=(-scale, scale), ylim=(-scale, scale)),
                   2, basis_type="dub_orth", volume_quad_1d=6)


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal", "bernstein"])
def test_continuous_polynomial_has_zero_face_mismatch_in_each_trace_basis(basis):
    space = space_for_norms()
    exact = lambda x, y: 1 + 2*x - .5*y
    field = space.project_callable(exact)
    trace = project_callable_to_trace(space, exact, trace_basis=basis, reduced=False)
    gram = ScalarHDGGram(space, trace_basis=basis, chunk_size=1)
    assert np.any(~space.mesh.orientations)
    assert gram.trace_mismatch_squared(field.coeffs, trace) < 1e-24
    assert gram.gradient_squared(field.coeffs) == pytest.approx(17)
    assert gram.l2_squared(field.coeffs) == pytest.approx(29/3)
    metrics = evaluate_hdg_scalar_error(field, trace, exact, lambda x, y: (2., -.5),
                                        trace_basis=basis, chunk_size=1)
    assert metrics.hdg_h1 < 1e-12


@pytest.mark.parametrize("weight", ["unit", "scaled", "element"])
def test_scalar_quadratic_form_agrees_with_existing_mixed_gram(weight):
    space = space_for_norms()
    mesh, q = space.mesh, space.quad_data
    rng = np.random.default_rng(401)
    c = rng.standard_normal((mesh.num_tri, q.el_dof))
    trace = np.zeros((mesh.num_edg, q.edg_dof))
    trace[mesh.int_edges_inds] = rng.standard_normal((mesh.int_edges_inds.size, q.edg_dof))
    scalar = ScalarHDGGram(space, jump_weight=weight, sigma=3)
    mixed = assemble_hdg_gram(space, jump_weight=weight, sigma=3)
    local = np.zeros((mesh.num_tri, 3, q.el_dof))
    local[:, 0] = c  # mixed Gram layout: element [u,qx,qy], then interior trace
    state = np.concatenate((local.ravel(), trace[mesh.int_edges_inds].ravel()))
    expected = float(state @ mixed.matrix @ state)
    assert scalar.norm_squared(c, trace) == pytest.approx(expected + scalar.l2_squared(c), rel=2e-13)


def test_piecewise_constants_reveal_jump_with_zero_broken_gradient_and_correct_scaling():
    # Two-sided interior contributions, plus the separate optional boundary term.
    values = np.array([1., -2., 3., -4.])
    measured = []
    for scale in (1, 3):
        space = space_for_norms(scale)
        mesh = space.mesh
        c = values[:, None] * space.constant(1).coeffs
        trace = np.zeros(mesh.num_edg*space.quad_data.edg_dof)
        interior = ScalarHDGGram(space, include_boundary=False)
        full = ScalarHDGGram(space)
        lengths = 2*mesh.jacs_el_fc
        h = lengths.max(axis=1)
        mask = np.isin(mesh.loc2glob_edge, mesh.int_edges_inds)
        expected = np.sum(values**2 * np.sum(lengths*mask, axis=1)/h)
        assert interior.gradient_squared(c) < 1e-24
        assert interior.trace_mismatch_squared(c, trace) == pytest.approx(expected)
        assert full.trace_mismatch_squared(c, trace) == pytest.approx(np.sum(values**2*lengths.sum(axis=1)/h))
        # Unused zero-flux boundary slots must not contribute, even if nonzero.
        trace.reshape(mesh.num_edg, -1)[mesh.bnd_edges_inds] = 1e3
        assert interior.trace_mismatch_squared(c, trace) == pytest.approx(expected)
        measured.append((interior.l2_squared(c), expected))
    assert measured[1][0] == pytest.approx(9*measured[0][0])
    assert measured[1][1] == pytest.approx(measured[0][1])


def test_exact_error_uses_analytic_physical_gradient_and_quadratic_sum():
    space = space_for_norms()
    field = space.zeros()
    trace = np.zeros(space.mesh.num_edg*space.quad_data.edg_dof)
    error = evaluate_hdg_scalar_error(field, trace, lambda x, y: x*x, lambda x, y: (2*x, 0.))
    assert error.l2**2 == pytest.approx(4/5)
    assert error.gradient_l2**2 == pytest.approx(16/3)
    assert error.trace_mismatch == 0
    assert error.hdg_h1**2 == pytest.approx(4/5 + 16/3)
