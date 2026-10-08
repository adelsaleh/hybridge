"""Skeleton error of numerical traces (``hybridge.diagnostics.errors.evaluate_trace_error``)."""

from __future__ import annotations

import numpy as np
import pytest

from hybridge import DGSpace
from hybridge.core.field_ops import project_callable_to_trace
from hybridge.core.mesh import gmsh_rectangle_mesh
from hybridge.diagnostics.errors import evaluate_trace_error


def _wave(x, y):
    return np.cos(3.0 * x + 2.0 * y) + 0.5 * y


@pytest.fixture(scope="module")
def space():
    return DGSpace(gmsh_rectangle_mesh(0.2, log_cache=False), 3, basis_type="dub_orth")


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
def test_projected_trace_has_zero_projection_error_in_both_bases(space, basis):
    trace = project_callable_to_trace(space, _wave, trace_basis=basis, reduced=False)
    metrics = evaluate_trace_error(space, trace, _wave, trace_basis=basis)
    assert metrics.projection_l2 < 1e-13 * metrics.exact_l2
    assert 0.0 < metrics.l2 < 1e-3 * metrics.exact_l2


def test_bases_agree_and_unit_weight_is_neutral(space):
    nodal = evaluate_trace_error(
        space, project_callable_to_trace(space, _wave, trace_basis="legacy-lagrange", reduced=False), _wave)
    modal = evaluate_trace_error(
        space, project_callable_to_trace(space, _wave, trace_basis="legendre-modal", reduced=False), _wave,
        trace_basis="legendre-modal")
    weighted = evaluate_trace_error(
        space, project_callable_to_trace(space, _wave, trace_basis="legacy-lagrange", reduced=False), _wave,
        face_weight=lambda x, y, nx, ny: nx * nx + ny * ny)
    assert modal.l2 == pytest.approx(nodal.l2, rel=1e-10)
    assert weighted.l2 == pytest.approx(nodal.l2, rel=1e-12)


@pytest.mark.parametrize("edges", ["interior", "all"])
def test_skeleton_weight_is_sum_of_squared_edge_lengths(space, edges):
    one = lambda x, y: 1.0 + 0.0 * x
    metrics = evaluate_trace_error(space, project_callable_to_trace(space, one, reduced=False), one, edges=edges)
    mesh = space.mesh
    lengths = np.linalg.norm(mesh.node_coords[mesh.edges[:, 0]] - mesh.node_coords[mesh.edges[:, 1]], axis=1)
    selected = lengths[mesh.int_edges_inds] if edges == "interior" else lengths
    assert metrics.exact_l2 ** 2 == pytest.approx(np.sum(selected ** 2), rel=1e-12)


def test_trace_error_converges_at_projection_order():
    errors = []
    for size in (0.2, 0.1):
        space = DGSpace(gmsh_rectangle_mesh(size, log_cache=False), 2, basis_type="dub_orth")
        trace = project_callable_to_trace(space, _wave, reduced=False)
        errors.append((space.mesh.num_tri, evaluate_trace_error(space, trace, _wave).l2))
    (k0, e0), (k1, e1) = errors
    rate = np.log(e0 / e1) / np.log(np.sqrt(k1 / k0))
    assert rate == pytest.approx(3.0, abs=0.4)


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
def test_enriched_edge_rule_matches_native_rule_on_smooth_data(space, basis):
    trace = project_callable_to_trace(space, _wave, trace_basis=basis, reduced=False)
    native = evaluate_trace_error(space, trace, _wave, trace_basis=basis)
    enriched = evaluate_trace_error(space, trace, _wave, trace_basis=basis, edge_quad_1d=16)
    assert enriched.exact_l2 == pytest.approx(native.exact_l2, rel=1e-12)
    assert enriched.l2 == pytest.approx(native.l2, rel=1e-3)
    # A polynomial trace of degree p is integrated exactly by both rules.
    cubic = lambda x, y: x**3 - 2.0 * x * y + y
    poly = project_callable_to_trace(space, cubic, trace_basis=basis, reduced=False)
    assert evaluate_trace_error(space, poly, cubic, trace_basis=basis, edge_quad_1d=16).l2 < 1e-12
