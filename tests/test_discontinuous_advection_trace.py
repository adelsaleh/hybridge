"""Host assembly-only diagnosis of the original discontinuous-velocity fixture."""

import numpy as np
import pytest

from hdgfem import DGSpace, VectorDGField, rectangle_mesh
from hdgfem.linalg.system import assemble_global_matrix
from hdgfem.solvers import advection_reaction
from scripts.advection_reaction.diagnose_discontinuous_trace import assemble_fixture, diagnose


@pytest.fixture(autouse=True)
def forbid_global_solve(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Rank and matrix parity checks must not invoke a global solve")
    monkeypatch.setattr(advection_reaction, "solve_global_system", reject)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_averaging_repairs_the_fixture_in_numpy_and_numba(degree, trace_basis, boundary_mode):
    options = dict(degree=degree, trace_basis=trace_basis, boundary_mode=boundary_mode)
    original = assemble_fixture(**options)
    averaged = assemble_fixture(**options, scenario="averaged")
    broken, repaired = diagnose(original), diagnose(averaged)
    for case in (broken, repaired):
        assert case["assembly_only"]
        assert case["local_min_rcond"] > 1.e-12
        assert case["reaction_min"] > 1.98
    assert broken["matrix_zero_rows"] == 0
    assert broken["matrix_zero_column_samples"] == broken["seam_trace_columns"]
    assert broken["matrix_zero_columns"] == degree + 1
    assert broken["row_scaled_rank"] == broken["matrix_size"] - degree - 1
    assert broken["seam_local_coupling_abs_max"] == broken["seam_column_abs_max"] == 0.
    assert broken["trace_inflow_diagnostics"]["double_outflow_faces"] == 1
    assert repaired["matrix_zero_columns"] == 0
    assert repaired["row_scaled_rank"] == repaired["matrix_size"]
    assert repaired["seam_local_coupling_abs_max"] > 0.
    assert repaired["trace_inflow_diagnostics"]["no_inflow_faces"] == 0
    for first, second in zip(original.beta.components, averaged.beta.components, strict=True):
        np.testing.assert_array_equal(first.coeffs, second.coeffs)
    for host in (original, averaged):
        compiled_path = assemble_fixture(**options, backend="numba", scenario=host.scenario)
        np.testing.assert_allclose(compiled_path.matrix().toarray(), host.matrix().toarray(), rtol=2.e-11, atol=2.e-12)
        np.testing.assert_allclose(compiled_path.result.solve_rhs, host.result.solve_rhs, rtol=2.e-11, atol=2.e-12)
        report = diagnose(compiled_path)
        assert report["row_scaled_rank"] == diagnose(host)["row_scaled_rank"]
        assert report["assembly_only"]


@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_positive_reaction_does_not_replace_missing_face_inflow(trace_basis, boundary_mode):
    options = dict(degree=3, trace_basis=trace_basis, boundary_mode=boundary_mode)
    stronger = diagnose(assemble_fixture(**options, scenario="reaction-20"))
    continuous = diagnose(assemble_fixture(**options, scenario="continuous"))
    assert stronger["reaction_min"] == 20.
    assert stronger["local_min_rcond"] > 1.e-12
    assert stronger["matrix_zero_columns"] == 4
    assert stronger["row_scaled_rank"] == stronger["matrix_size"] - 4
    assert continuous["matrix_zero_columns"] == 0
    assert continuous["row_scaled_rank"] == continuous["matrix_size"]
    assert continuous["trace_inflow_diagnostics"]["double_outflow_faces"] == 0


@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
def test_unaligned_two_triangle_fixture_is_a_different_rank_case(trace_basis):
    # The existing solve tests use nx=1: the jump then cuts element interiors.
    report = diagnose(assemble_fixture(degree=2, trace_basis=trace_basis, nx=1))
    assert report["triangles"] == 2 and report["seam_edges"] == []
    assert report["row_scaled_rank"] == report["matrix_size"]


@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
def test_tangent_zero_reaction_remains_assembly_only_despite_averaging(trace_basis):
    space = DGSpace(rectangle_mesh(2, 2, xlim=(-1., 1.), ylim=(-1., 1.)), 3,
                    basis_type="dub_orth", volume_quad_1d=8)
    # Curl of (1-x^2)(1-y^2): exactly divergence-free and tangent to the box.
    beta = VectorDGField((lambda x, y: -2*y*(1-x*x), lambda x, y: 2*x*(1-y*y)), space)
    for reaction, nullity in ((0., 1), (1., 0)):
        result = advection_reaction.AdvectionReactionHDGSolver(
            space, source=space.zeros(), reaction=space.constant(reaction), beta=beta,
            boundary_condition=None, boundary_mode="zero-flux", assembly_backend="numba",
            trace_basis=trace_basis, advection_stabilization="conflict-averaged-upwind",
            solver="direct", preconditioner=None, scale_system=False, verbose=False,
        ).assemble_trace_system()
        matrix = assemble_global_matrix(result.solve_matrix_rows, result.solve_matrix_cols,
                                        result.solve_matrix_data, result.solve_rhs.size).toarray()
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = max(matrix.shape) * np.finfo(matrix.dtype).eps * singular[0]
        assert np.count_nonzero(singular > tolerance) == len(matrix) - nullity
        if reaction == 0.:
            trace = space.trace_space(trace_basis)
            one_edge = np.linalg.solve(trace.M_rf_fc, trace.bas1d_of_ref_edg_qds @ trace.weights)
            constant_trace = np.tile(one_edge, len(matrix) // trace.edg_dof)
            assert np.linalg.norm(matrix @ constant_trace) <= 1.e-12 * singular[0] * np.linalg.norm(constant_trace)
        assert result.field is result.trace is result.global_solve_result is None


@pytest.mark.parametrize("options", [{"degree": 12}, {"nx": 200}, {"scenario": "unknown"}])
def test_diagnostic_stays_bounded(options):
    with pytest.raises(ValueError):
        assemble_fixture(**options)
