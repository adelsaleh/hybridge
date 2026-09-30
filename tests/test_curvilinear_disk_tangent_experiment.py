from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from hdgfem.core.space import DGSpace, VectorDGField
from hdgfem.solvers.advection_reaction import solve_advection_reaction_hdg
from scripts.advection_reaction.cases import disk_tangent_conservative

from scripts.advection_reaction.experiments.curvilinear_disk_tangent_numba import (
    GEOMETRY_KINDS,
    _face_reference_points,
    _build_geometry_controls_numpy,
    build_geometry_controls,
    build_disk_topology,
    evaluate_geometry,
    plot_finest_geometry_result,
    run_convergence,
    run_single,
)


pytest.importorskip("gmsh")
pytest.importorskip("numba")
pytest.importorskip("matplotlib")


@pytest.mark.parametrize("kind", GEOMETRY_KINDS)
def test_numba_geometry_controls_match_numpy_reference(kind: str) -> None:
    mesh = build_disk_topology(0.5, cache=False)
    controls, weights, code = build_geometry_controls(mesh, kind)
    expected_controls, expected_weights, expected_code = _build_geometry_controls_numpy(
        mesh, kind
    )

    assert code == expected_code
    np.testing.assert_allclose(controls, expected_controls, rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(weights, expected_weights, rtol=0.0, atol=2.0e-16)


def test_curved_geometry_tables_are_finite_and_orientation_preserving() -> None:
    mesh = build_disk_topology(0.7, cache=False)
    volume_points = np.array(((-0.5, -0.5), (0.0, -0.5), (-0.5, 0.0)))
    face_points = _face_reference_points(np.linspace(-1.0, 1.0, 9))

    for kind in GEOMETRY_KINDS:
        geometry = evaluate_geometry(mesh, kind, volume_points, face_points)
        assert np.all(np.isfinite(geometry.volume_points))
        assert np.all(np.isfinite(geometry.face_normals))
        assert np.min(geometry.det_jacobians) > 0.0
        np.testing.assert_allclose(
            np.linalg.norm(geometry.face_normals, axis=-1),
            1.0,
            rtol=2.0e-14,
            atol=2.0e-14,
        )


def test_curved_zero_flux_modal_and_nodal_traces_agree() -> None:
    mesh = build_disk_topology(0.5, cache=False)
    nodal = run_single(
        mesh,
        mesh_size=0.5,
        geometry_kind="rational-p2",
        solution_order=2,
        trace_basis="legacy-lagrange",
        linear_solver="scipy",
    )
    modal = run_single(
        mesh,
        mesh_size=0.5,
        geometry_kind="rational-p2",
        solution_order=2,
        trace_basis="legendre-modal",
        linear_solver="scipy",
    )

    for result in (nodal, modal):
        assert result.assembly_seconds >= 0.0
        assert result.sparse_reduction_seconds >= 0.0
        assert result.volume_quadrature == "duffy"
        assert result.num_volume_quads == 25
    assert nodal.relative_residual < 1.0e-12
    assert modal.relative_residual < 1.0e-12
    np.testing.assert_allclose(modal.l2_error, nodal.l2_error, rtol=2.0e-11, atol=1.0e-13)
    np.testing.assert_allclose(modal.field_coefficients, nodal.field_coefficients, rtol=2.0e-10, atol=2.0e-11)


def test_solution_order_exposes_affine_and_quadratic_geometry_floors() -> None:
    rows = run_convergence(
        mesh_sizes=(0.7, 0.5, 0.35),
        solution_orders=(1, 4),
        cache_mesh=False,
        linear_solver="scipy",
    )
    by_case = {
        (row.result.geometry, row.result.solution_order, row.result.mesh_size): row
        for row in rows
    }
    fine = 0.35
    affine_p1 = by_case[("p1", 1, fine)]
    affine_p4 = by_case[("p1", 4, fine)]
    polynomial_p1 = by_case[("p2", 1, fine)]
    polynomial_p4 = by_case[("p2", 4, fine)]
    rational_p4 = by_case[("rational-p2", 4, fine)]

    # Raising the HDG order cannot repair the affine polygonal boundary.
    assert affine_p4.result.l2_error > affine_p1.result.l2_error

    # Curved P2 geometry exposes the expected high-order HDG improvement.
    assert polynomial_p4.result.l2_error < 0.05 * polynomial_p1.result.l2_error
    assert polynomial_p4.observed_rate is not None
    assert polynomial_p4.observed_rate > 3.5

    # Exact rational arcs remove the remaining polynomial-P2 geometry floor.
    assert rational_p4.result.l2_error < 0.4 * polynomial_p4.result.l2_error
    assert rational_p4.observed_rate is not None
    assert rational_p4.observed_rate > 4.5

    assert all(row.result.relative_residual < 5.0e-12 for row in rows)


def test_affine_experiment_assembly_matches_production_numpy() -> None:
    mesh = build_disk_topology(0.7, cache=False)
    order = 2
    experimental = run_single(
        mesh,
        mesh_size=0.7,
        geometry_kind="p1",
        solution_order=order,
        linear_solver="scipy",
    )
    space = DGSpace(
        mesh,
        order,
        basis_type="dub_orth",
        volume_quadrature="duffy",
        volume_quad_1d=5,
    )
    beta_x, beta_y, reaction, source, _ = disk_tangent_conservative()
    reference = solve_advection_reaction_hdg(
        source,
        (beta_x, beta_y),
        reaction,
        None,
        space,
        boundary_mode="zero-flux",
        assembly_backend="numpy",
        trace_basis="legacy-lagrange",
        matrix_pattern_only=True,
        verbose=False,
    )
    reference_matrix = coo_matrix(
        (reference.solve_matrix_data, (reference.solve_matrix_rows, reference.solve_matrix_cols)),
        shape=experimental.matrix.shape,
    ).tocsr()
    reference_matrix.sum_duplicates()

    np.testing.assert_allclose(
        experimental.matrix.toarray(),
        reference_matrix.toarray(),
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(experimental.rhs, reference.solve_rhs, rtol=2.0e-11, atol=2.0e-11)


def test_cross_degree_vector_dg_beta_matches_analytic_affine_velocity() -> None:
    mesh = build_disk_topology(0.7, cache=False)
    beta_x, beta_y, _, _, _ = disk_tangent_conservative()
    beta_x_space = DGSpace(mesh, 3, basis_type="dub_orth")
    beta_y_space = DGSpace(mesh, 5, basis_type="dub_orth")
    beta = VectorDGField((
        beta_x_space.project_callable(beta_x),
        beta_y_space.project_callable(beta_y),
    ))

    analytic = run_single(
        mesh,
        mesh_size=0.7,
        geometry_kind="p1",
        solution_order=2,
        linear_solver="scipy",
    )
    discrete = run_single(
        mesh,
        mesh_size=0.7,
        geometry_kind="p1",
        solution_order=2,
        beta_field=beta,
        linear_solver="scipy",
    )

    assert tuple(component.space.order for component in beta.components) == (3, 5)
    assert discrete.relative_residual < 1.0e-12
    assert discrete.solver_backend == "scipy"
    np.testing.assert_allclose(discrete.matrix.toarray(), analytic.matrix.toarray(), rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(discrete.rhs, analytic.rhs, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(discrete.field_coefficients, analytic.field_coefficients, rtol=2e-10, atol=2e-11)


def test_runner_writes_geometry_correct_error_plot(tmp_path) -> None:
    rows = run_convergence(
        mesh_sizes=(0.7,),
        solution_orders=(1,),
        cache_mesh=False,
        linear_solver="scipy",
    )
    output = tmp_path / "curvilinear-error.png"
    returned = plot_finest_geometry_result(
        rows,
        output=output,
        resolution=9,
        quantity="error",
        show=False,
    )

    assert returned == output
    assert output.is_file()
    assert output.stat().st_size > 0
