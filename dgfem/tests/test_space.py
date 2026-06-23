from __future__ import annotations

import numpy as np
import pytest

from dgfem import DGMesh, DGSpace, gmsh_disc_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, solve_advection_reaction_hdg
from dgfem.hdg_assembly import (
    assemble_trace_system,
    as_vector_field,
    element_to_trace_matrix,
    global_rhs,
    reconstruct_field,
    source_moments,
    trace_matrix_data,
    trace_matrix_indices,
)
from dgfem.hdg_mats import (
    _vector_values_on_test_quads,
    add_advection_mats,
    add_reaction_mass,
    advection_mats,
    advective_boundary_normal,
    boundary_mass,
    boundary_mass_from_normal_flux,
    element_boundary_mats,
    element_boundary_mats_from_normal_flux,
    mass_from_field,
    weighted_mass_from_field,
)


def reference_triangle_mesh() -> DGMesh:
    nodes = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.uint64)
    return DGMesh.from_arrays(nodes, triangles)


def split_reference_triangle_mesh() -> DGMesh:
    nodes = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.uint64)
    return DGMesh.from_arrays(nodes, triangles)


def test_constant_field_evaluation_and_norm() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="hier_C0")
    u = V.field(np.ones(V.shape))

    np.testing.assert_allclose(u.values(), 1.0)
    np.testing.assert_allclose(u.grad_values()[0], 0.0, atol=1e-13)
    np.testing.assert_allclose(u.grad_values()[1], 0.0, atol=1e-13)
    np.testing.assert_allclose(u.l2_norm() ** 2, np.sum(V.quad_data.Krf_w))


def test_project_callable_constant() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="hier_C0")

    u = V.project_callable(lambda x, y: np.ones_like(x), name="one")

    np.testing.assert_allclose(u.values(), 1.0, atol=1e-12)


def test_vector_space_product_and_layouts() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="hier_C0")
    W = V * V

    coeffs = np.stack((np.ones(V.shape), 2.0 * np.ones(V.shape)))
    beta = W.field(coeffs)

    assert beta.dim == 2
    assert beta.as_component_first().shape == coeffs.shape
    np.testing.assert_allclose(beta.as_component_last()[..., 0], 1.0)
    np.testing.assert_allclose(beta.as_component_last()[..., 1], 2.0)

    component_last = np.stack((np.ones(V.shape), 3.0 * np.ones(V.shape)), axis=-1)
    gamma = V.vector_field(component_last)
    np.testing.assert_allclose(gamma.as_component_first()[0], 1.0)
    np.testing.assert_allclose(gamma.as_component_first()[1], 3.0)


def test_weighted_mass_from_field_matches_scaled_mass() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="hier_C0")
    u = V.field(2.0 * np.ones(V.shape))

    mass = V.mass()
    weighted = weighted_mass_from_field(V, lambda z: z + 1.0, u)

    np.testing.assert_allclose(weighted, 3.0 * mass)


def test_mass_from_field_matches_quadrature_path() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 3, basis_type="dub_orth")
    coeffs = np.arange(np.prod(V.shape), dtype=np.float64).reshape(V.shape) / 10.0
    reaction = V.field(coeffs)

    direct = mass_from_field(V, reaction)
    quadrature = weighted_mass_from_field(V, lambda z: z, reaction)

    np.testing.assert_allclose(direct, quadrature, rtol=1e-12, atol=1e-12)


def test_same_mesh_projection_between_orders() -> None:
    mesh = reference_triangle_mesh()
    V1 = DGSpace(mesh, 1, basis_type="hier_C0")
    V2 = DGSpace(mesh, 2, basis_type="hier_C0")
    u1 = V1.field(np.ones(V1.shape))

    u2, diag = u1.project_to(V2, verbose=False)

    assert diag.n_missed_points == 0
    np.testing.assert_allclose(u2.values(), 1.0, atol=1e-12)


def test_values_at_xy_and_cross_mesh_projection() -> None:
    source_mesh = reference_triangle_mesh()
    target_mesh = split_reference_triangle_mesh()
    V_source = DGSpace(source_mesh, 1, basis_type="bernstein")
    V_target = DGSpace(target_mesh, 1, basis_type="bernstein")
    u = V_source.field(np.ones(V_source.shape))

    np.testing.assert_allclose(u.values_at_xy(np.array([[-0.5, -0.5], [0.0, 0.0]])), 1.0)

    transferred, diag = u.project_to(V_target, verbose=False)

    assert diag.n_missed_points == 0
    np.testing.assert_allclose(transferred.values(), 1.0, atol=1e-12)


def test_object_hdg_assembly_shapes_are_self_contained() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="bernstein")
    beta = (V * V).field(np.stack((np.ones(V.shape), np.zeros(V.shape))))

    adv = advection_mats(V, beta)
    bd = boundary_mass(V, beta)
    el_bd = element_boundary_mats(V, beta)

    assert adv.shape == (mesh.num_tri, V.el_dof, V.el_dof)
    assert bd.shape == (mesh.num_tri, V.el_dof, V.el_dof)
    assert el_bd.shape == (mesh.num_tri, V.el_dof, 3 * V.quad_data.edg_dof)
    assert np.all(np.isfinite(adv))
    assert np.all(np.isfinite(bd))
    assert np.all(np.isfinite(el_bd))


def test_advection_mats_match_expanded_physical_gradient_formula() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    coeffs = np.stack((np.ones(V.shape), 0.5 * np.ones(V.shape)))
    beta = (V * V).field(coeffs)

    direct = advection_mats(V, beta)

    beta_values = _vector_values_on_test_quads(beta, V)
    grad_ref = V.quad_data.dbas_of_quads.swapaxes(0, 2)
    grad_phys = np.einsum("Krd,qid->Kqir", V.mesh.inv_aff_mats_t, grad_ref, optimize=True)
    beta_dot_grad = np.einsum("Kqr,Kqir->Kqi", beta_values, grad_phys, optimize=True)
    expanded = np.einsum(
        "K,Kqi,jq,q->Kij",
        V.mesh.aff_jacs,
        beta_dot_grad,
        V.quad_data.bas_of_quads,
        V.quad_data.Krf_w,
        optimize=True,
    )

    np.testing.assert_allclose(direct, expanded, rtol=1e-12, atol=1e-12)


def test_reference_edge_matrices_and_sigma_1() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="bernstein")
    ref = V.quad_data

    assert mesh.get_sigma_1().shape == (mesh.num_tri, 3)
    assert ref.MKrfe_lst_p.shape == (3, ref.el_dof, ref.edg_dof)
    assert ref.MKrfe_lst_n.shape == (3, ref.el_dof, ref.edg_dof)
    assert ref.MKrfe_lst.shape == (6, ref.edg_dof, ref.el_dof)
    assert ref.M_rf_fc.shape == (ref.edg_dof, ref.edg_dof)
    assert ref.MbdeKrf_lst.shape == (3, ref.el_dof, ref.el_dof)


def test_dubiner_reference_element_path() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    ref = V.quad_data
    u = V.field(np.zeros(V.shape))
    u.coeffs[:, 0] = 1.0

    assert ref.basis_type == "dub_orth"
    assert ref.bas_of_quads.shape == (ref.el_dof, ref.Krf_quads.shape[0])
    assert ref.dbas_of_quads.shape == (2, ref.el_dof, ref.Krf_quads.shape[0])
    np.testing.assert_allclose(u.values(), 1.0)


def test_three_basis_families_are_distinct_choices() -> None:
    mesh = reference_triangle_mesh()
    bernstein = DGSpace(mesh, 2, basis_type="bernstein").quad_data
    hierarchical = DGSpace(mesh, 2, basis_type="hier_C0").quad_data
    dubiner = DGSpace(mesh, 2, basis_type="dub_orth").quad_data

    assert bernstein.basis_type == "bernstein"
    assert hierarchical.basis_type == "hier_C0"
    assert dubiner.basis_type == "dub_orth"
    assert bernstein.bas_of_quads.shape == hierarchical.bas_of_quads.shape == dubiner.bas_of_quads.shape
    assert not np.allclose(bernstein.bas_of_quads, hierarchical.bas_of_quads)
    assert not np.allclose(bernstein.bas_of_quads, dubiner.bas_of_quads)


def test_hierarchical_c0_constant_projection() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="hier_C0")

    u = V.project_callable(lambda x, y: np.ones_like(x), name="one")

    np.testing.assert_allclose(u.values(), 1.0, atol=1e-12)


def test_advection_reaction_solver_smoke() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="dub_orth")
    one = lambda x, y: np.ones_like(x)
    zero = lambda x, y: np.zeros_like(x)

    result = solve_advection_reaction_hdg(one, (one, zero), 1.0, zero, V, verbose=False)

    assert result.field.coeffs.shape == V.shape
    assert result.trace.shape == (mesh.num_edg * V.quad_data.edg_dof,)
    assert np.all(np.isfinite(result.field.coeffs))
    assert np.all(np.isfinite(result.trace))


def test_hdg_assembly_helpers_build_trace_system() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="dub_orth")
    one = lambda x, y: np.ones_like(x)
    zero = lambda x, y: np.zeros_like(x)

    beta = as_vector_field((one, zero), V)
    beta_dot_normal = advective_boundary_normal(beta, V)
    local_mats = np.ascontiguousarray(boundary_mass_from_normal_flux(V, beta_dot_normal))
    scratch = np.empty_like(local_mats)
    add_reaction_mass(local_mats, 1.0, V, scratch=scratch)
    add_advection_mats(local_mats, V, beta, scale=-1.0)
    local_solver = np.linalg.inv(local_mats)
    element_bd = element_boundary_mats_from_normal_flux(V, beta_dot_normal)
    source_rhs = source_moments(one, V)

    trace_blocks = element_to_trace_matrix(local_solver, element_bd, V)
    rows, cols = trace_matrix_indices(V)
    data = trace_matrix_data(trace_blocks, V, 1e20)
    rhs, boundary_trace = global_rhs(source_rhs, local_solver, zero, V, 1e20)
    system = assemble_trace_system(local_solver, element_bd, source_rhs, zero, V)
    field = reconstruct_field(np.zeros_like(rhs), source_rhs, local_solver, element_bd, V)

    assert trace_blocks.shape == (mesh.num_tri, 3, 3, V.quad_data.edg_dof, V.quad_data.edg_dof)
    assert rows.shape == cols.shape == data.shape == system.rows.shape
    assert rhs.shape == system.rhs.shape == (mesh.num_edg * V.quad_data.edg_dof,)
    assert boundary_trace.shape == system.boundary_trace.shape == (mesh.num_edg, V.quad_data.edg_dof)
    assert field.coeffs.shape == V.shape
    np.testing.assert_array_equal(system.rows, rows)
    np.testing.assert_array_equal(system.cols, cols)
    np.testing.assert_allclose(system.data, data)
    np.testing.assert_allclose(system.rhs, rhs)


def test_gmsh_basic_shape_meshes() -> None:
    pytest.importorskip("gmsh")

    for mesh in (
        gmsh_rectangle_mesh(1.0, verbosity=0),
        gmsh_disc_mesh(1.0, verbosity=0),
        gmsh_triangle_mesh(1.0, verbosity=0),
    ):
        assert mesh.num_tri > 0
        assert mesh.num_edg > 0
        assert np.all(mesh.aff_jacs > 0.0)
