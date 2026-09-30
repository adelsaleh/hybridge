from __future__ import annotations

import numpy as np
import pytest

from hdgfem import (
    DGField,
    DGMesh,
    DGSpace,
    VectorDGField,
    gmsh_disc_mesh,
    gmsh_lshape_mesh,
    gmsh_rectangle_mesh,
    gmsh_smooth_star_mesh,
    gmsh_star_mesh,
    gmsh_triangle_mesh,
    rectangle_mesh,
    solve_advection_reaction_hdg,
)
from hdgfem.core.space import evaluate_product
from hdgfem.assembly.hdg import (
    assemble_trace_system,
    as_vector_field,
    boundary_trace_coefficients,
    element_to_trace_matrix,
    global_rhs,
    reconstruct_field,
    source_moments,
    trace_matrix_data,
    trace_matrix_indices,
)
from hdgfem.assembly.matrices_numpy import (
    _vector_values_on_test_quads,
    add_advection_mats,
    advection_mats,
    advective_boundary_normal,
    boundary_mass,
    boundary_mass_from_normal_flux,
    element_boundary_mats,
    element_boundary_mats_from_normal_flux,
)
from hdgfem.core.mass import (
    add_reaction_mass,
    mass_from_field,
    weighted_mass_from_field,
)
from hdgfem.solvers.diffusion_reaction import (
    DiffusionReactionHDGOptions,
    DiffusionReactionHDGSolver,
    impose_boundary_trace_on_guess,
    solve_diffusion_reaction_hdg,
)
from hdgfem.core.trace_transfer import prolong_trace_coefficients
from scripts.diffusion_reaction.experiments.bootstrap_initial_guess import (
    solve_diffusion_reaction_hdg as solve_diffusion_reaction_hdg_with_bootstrap,
)
from hdgfem.linalg.system import eliminate_known_dofs, expand_known_dofs, solve_global_system
from hdgfem.linalg.ordering import strongly_connected_component_order, upwind_scc_trace_ordering
from scripts.diffusion_reaction.cases import (
    lshape_singular_harmonic_case,
    quadratic_poisson_case,
    trigonometric_poisson_case,
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


def test_dgspace_accepts_explicit_quadrature_counts() -> None:
    mesh = reference_triangle_mesh()
    default = DGSpace(mesh, 3, basis_type="dub_orth")
    custom = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=5, edge_quad_1d=4)
    legacy = DGSpace(mesh, 3, basis_type="dub_orth", volume_quadrature="duffy")

    assert default.quad_data.volume_quadrature == "symmetric"
    assert default.quad_data.Krf_w.size == 12
    assert legacy.quad_data.Krf_w.size == (2 * legacy.order + 2) ** 2
    assert default.quad_data.weights_JGL.size == 2 * default.order + 2
    assert custom.quad_data.Krf_w.size == 25
    assert custom.quad_data.weights_JGL.size == 4
    assert custom.el_dof == default.el_dof
    assert custom.quad_data.edg_dof == default.quad_data.edg_dof


def test_auto_quadrature_uses_compact_p7_and_duffy_above_table() -> None:
    p7 = DGSpace(reference_triangle_mesh(), 7)
    p8 = DGSpace(reference_triangle_mesh(), 8)

    assert p7.quad_data.volume_quadrature == "symmetric"
    assert p7.quad_data.Krf_w.size == 42
    assert p8.quad_data.volume_quadrature == "duffy"
    assert p8.quad_data.Krf_w.size == (2 * p8.order + 2) ** 2
    np.testing.assert_allclose(np.sum(p7.quad_data.Krf_w), 2.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(np.sum(p8.quad_data.Krf_w), 2.0, rtol=0.0, atol=1.0e-14)


def test_explicit_symmetric_quadrature_is_generated_above_compact_table() -> None:
    space = DGSpace(reference_triangle_mesh(), 8, volume_quadrature="symmetric")

    assert space.quad_data.volume_quadrature == "symmetric"
    assert space.quad_data.Krf_w.size <= 6 * (space.order + 1) ** 2
    assert space.quad_data.Krf_w.size > (2 * space.order + 2) ** 2
    np.testing.assert_allclose(np.sum(space.quad_data.Krf_w), 2.0, rtol=0.0, atol=1.0e-14)


def test_project_callable_constant() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="hier_C0")

    u = V.project_callable(lambda x, y: np.ones_like(x), name="one")

    np.testing.assert_allclose(u.values(), 1.0, atol=1e-12)


@pytest.mark.parametrize("basis_type", ("bernstein", "hier_C0", "dub_orth"))
def test_zero_and_constant_fields_carry_constructor_metadata(basis_type: str) -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 3, basis_type=basis_type)

    zero = V.zeros(name="zero_h")
    constant = V.constant(2.5, name="constant_h")
    projected = V.project_callable(lambda x, y: 2.5 + 0.0 * x * y, name="projected_h")

    assert not zero.coefficients_materialized
    assert not constant.coefficients_materialized

    assert zero.coefficient_kind == "zero"
    assert zero.is_zero
    assert zero.is_constant
    assert zero.constant_value == 0.0
    assert not zero.coefficients_materialized
    np.testing.assert_array_equal(zero.coeffs, np.zeros(V.shape))
    assert zero.coefficients_materialized

    assert constant.coefficient_kind == "constant"
    assert constant.is_constant
    assert constant.constant_value == 2.5
    assert not constant.is_zero
    np.testing.assert_allclose(constant.values(), 2.5, atol=1.0e-12)
    assert not constant.coefficients_materialized

    table = V.field(constant.coeffs.copy(), name="table_h")
    assert constant.coefficients_materialized
    assert table.coefficient_kind == "table"
    assert not table.is_constant
    assert table.constant_value is None
    np.testing.assert_allclose(table.values(), 2.5, atol=1.0e-12)

    assert projected.coefficient_kind == "projected"
    assert not projected.is_constant
    np.testing.assert_allclose(projected.values(), 2.5, atol=1.0e-12)

    mutated = constant.copy(name="mutated")
    mutated.coeffs[0, 0] += 1.0e-3
    assert mutated.constant_value is None
    assert not mutated.is_constant


def test_vector_constant_metadata() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    beta = (V * V).constant((1.25, -0.5), name="beta_h")

    assert beta.is_constant
    assert beta.constant_values == (1.25, -0.5)
    assert not beta.is_zero
    np.testing.assert_allclose(beta.components[0].values(), 1.25, atol=1.0e-12)
    np.testing.assert_allclose(beta.components[1].values(), -0.5, atol=1.0e-12)


def test_direct_analytic_reaction_mass_is_not_projected_dg_assembly() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="dub_orth", volume_quad_1d=4)
    reaction = lambda x, y: 1.0 + x * x + 0.25 * x * y

    direct = add_reaction_mass(np.zeros((mesh.num_tri, V.el_dof, V.el_dof)), reaction, V)
    projected = add_reaction_mass(
        np.zeros((mesh.num_tri, V.el_dof, V.el_dof)),
        V.project_callable(reaction, name="reaction_h"),
        V,
    )

    assert not np.allclose(direct, projected, rtol=1.0e-10, atol=1.0e-12)


def test_numpy_source_and_reaction_fast_paths_do_not_materialize_constant_fields() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=5)
    source = V.constant(1.75, name="source_h")
    reaction = V.constant(2.25, name="reaction_h")

    rhs = source_moments(source, V)
    mass = add_reaction_mass(np.zeros((mesh.num_tri, V.el_dof, V.el_dof)), reaction, V)

    assert rhs.shape == V.shape
    assert mass.shape == (mesh.num_tri, V.el_dof, V.el_dof)
    assert not source.coefficients_materialized
    assert not reaction.coefficients_materialized


def test_dgfield_constructor_projects_callable() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    func = lambda x, y: x + y + 1.0

    direct = DGField(func, V, name="u_h")
    expected = V.project_callable(func, name="u_h")

    assert direct.name == "u_h"
    np.testing.assert_allclose(direct.coeffs, expected.coeffs, atol=1e-13, rtol=1e-13)


def test_dgfield_constructor_accepts_coefficient_array() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    coeffs = np.arange(np.prod(V.shape), dtype=np.float64).reshape(V.shape)

    field = DGField(coeffs, V, name="u_h")

    assert field.name == "u_h"
    np.testing.assert_array_equal(field.coeffs, coeffs)


def test_dgfield_mul_projects_same_space_product() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    u = DGField(lambda x, y: x + 1.0 + 0.0 * y, V, name="u")
    v = DGField(lambda x, y: y + 2.0 + 0.0 * x, V, name="v")

    product = u * v
    expected = V.project_callable(lambda x, y: (x + 1.0) * (y + 2.0), name="expected")

    np.testing.assert_allclose(product.coeffs, expected.coeffs, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(product.values(), expected.values(), atol=1e-12, rtol=1e-12)


def test_dgfield_project_product_matches_quadrature_projection() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 3, basis_type="bernstein")
    u = V.field(np.arange(np.prod(V.shape), dtype=np.float64).reshape(V.shape) / 10.0, name="u")
    v = V.field((np.arange(np.prod(V.shape), dtype=np.float64).reshape(V.shape) + 1.0) / 7.0, name="v")

    product = u.project_product(v)
    values = u.values() * v.values()
    rhs = values @ V.quad_data.weighted_phi
    expected_coeffs = rhs @ V.quad_data.MKrf_inv

    np.testing.assert_allclose(product.coeffs, expected_coeffs, atol=1e-12, rtol=1e-12)


def test_evaluate_product_returns_values_without_projection() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="bernstein")
    u = DGField(lambda x, y: x + 1.0 + 0.0 * y, V, name="u")
    v = DGField(lambda x, y: y + 2.0 + 0.0 * x, V, name="v")
    reference_points = np.array([[-0.5, -0.5], [0.0, 0.0]], dtype=np.float64)

    np.testing.assert_allclose(evaluate_product(u, v), u.values() * v.values())
    np.testing.assert_allclose(u.product_values(v), evaluate_product(u, v))
    np.testing.assert_allclose(
        evaluate_product(u, v, reference_points, reference=True),
        u.values_at_ref(reference_points) * v.values_at_ref(reference_points),
    )
    np.testing.assert_allclose(
        evaluate_product(u, v, reference_points),
        u(reference_points) * v(reference_points),
    )


def test_dgfield_cross_space_product_requires_explicit_target() -> None:
    mesh = reference_triangle_mesh()
    V1 = DGSpace(mesh, 1, basis_type="hier_C0")
    V2 = DGSpace(mesh, 2, basis_type="hier_C0")
    u = DGField(lambda x, y: x + 1.0 + 0.0 * y, V1, name="u")
    v = DGField(lambda x, y: y + 2.0 + 0.0 * x, V2, name="v")

    with pytest.raises(ValueError, match="target is required"):
        _ = u * v

    product = u.project_product(v, target=V2)
    expected = V2.project_callable(lambda x, y: (x + 1.0) * (y + 2.0), name="expected")

    np.testing.assert_allclose(product.coeffs, expected.coeffs, atol=1e-12, rtol=1e-12)


def test_dgfield_scalar_multiply_and_array_multiply_rejection() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="bernstein")
    u = V.field(np.arange(np.prod(V.shape), dtype=np.float64).reshape(V.shape), name="u")

    np.testing.assert_allclose((u * 2.5).coeffs, u.coeffs * 2.5)
    np.testing.assert_allclose((2.5 * u).coeffs, u.coeffs * 2.5)

    with pytest.raises(TypeError, match="arrays is ambiguous"):
        _ = u * np.ones(V.shape)
    with pytest.raises(TypeError, match="arrays is ambiguous"):
        _ = np.ones(V.shape) * u


def test_dgfield_linear_arithmetic_and_scalar_division() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    u_coeffs = np.arange(np.prod(V.shape), dtype=np.float64).reshape(V.shape)
    v_coeffs = np.flip(u_coeffs, axis=1).copy()
    u = V.field(u_coeffs, name="u")
    v = V.field(v_coeffs, name="v")

    result = (4.0 * u - v) / 3.0

    np.testing.assert_allclose(result.coeffs, (4.0 * u_coeffs - v_coeffs) / 3.0)
    np.testing.assert_array_equal(u.coeffs, u_coeffs)
    np.testing.assert_array_equal(v.coeffs, v_coeffs)
    with pytest.raises(ZeroDivisionError, match="divide a DGField by zero"):
        _ = u / 0.0
    with pytest.raises(TypeError, match="division by arrays is ambiguous"):
        _ = u / np.ones(V.shape)


def test_dgfield_linear_arithmetic_preserves_lazy_constants() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    current = V.constant(2.0, name="current")
    previous = V.constant(-1.0, name="previous")

    result = (4.0 * current - previous) / 3.0

    assert result.constant_value == pytest.approx(3.0)
    assert not current.coefficients_materialized
    assert not previous.coefficients_materialized
    assert not result.coefficients_materialized


def test_dgfield_linear_arithmetic_accepts_coefficient_compatible_spaces() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=4)
    equivalent_space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=5)
    u_coeffs = np.arange(np.prod(V.shape), dtype=np.float64).reshape(V.shape)
    other_coeffs = np.flip(u_coeffs, axis=1).copy()
    u = V.field(u_coeffs)
    other = equivalent_space.field(other_coeffs)

    added = u + other
    subtracted = u - other

    assert V.is_compatible(equivalent_space)
    assert added.space is V
    assert subtracted.space is V
    np.testing.assert_array_equal(added.coeffs, u_coeffs + other_coeffs)
    np.testing.assert_array_equal(subtracted.coeffs, u_coeffs - other_coeffs)


@pytest.mark.parametrize("basis_type", ("bernstein", "hier_C0", "dub_orth"))
def test_dgfield_linear_arithmetic_elevates_to_higher_order_space(
        basis_type: str,
) -> None:
    mesh = reference_triangle_mesh()
    low_space = DGSpace(mesh, 1, basis_type=basis_type)
    high_space = DGSpace(mesh, 3, basis_type=basis_type)
    low_coeffs = np.arange(np.prod(low_space.shape), dtype=np.float64).reshape(low_space.shape)
    high_coeffs = (
        0.25
        + np.arange(np.prod(high_space.shape), dtype=np.float64).reshape(high_space.shape)
    )
    low = low_space.field(low_coeffs, name="low")
    high = high_space.field(high_coeffs, name="high")

    added = low + high
    reverse_added = high + low
    subtracted = low - high
    reverse_subtracted = high - low

    elevation = high_space.degree_elevation_matrix_from(low_space)
    assert low_space.is_basis_compatible(high_space)
    assert not low_space.is_compatible(high_space)
    expected_low = low_coeffs @ elevation
    assert elevation is high_space.degree_elevation_matrix_from(low_space)
    assert all(
        result.space is high_space
        for result in (added, reverse_added, subtracted, reverse_subtracted)
    )
    np.testing.assert_allclose(added.coeffs, expected_low + high_coeffs)
    np.testing.assert_allclose(reverse_added.coeffs, high_coeffs + expected_low)
    np.testing.assert_allclose(subtracted.coeffs, expected_low - high_coeffs)
    np.testing.assert_allclose(reverse_subtracted.coeffs, high_coeffs - expected_low)

    points = np.array(((-0.8, -0.7), (0.2, -0.4), (-0.4, 0.1)))
    np.testing.assert_allclose(
        added.values_at_ref(points),
        low.values_at_ref(points) + high.values_at_ref(points),
    )

    constant_result = low_space.constant(2.0) + high_space.constant(-0.5)
    assert constant_result.space is high_space
    assert constant_result.constant_value == pytest.approx(1.5)
    assert not constant_result.coefficients_materialized

    with pytest.raises(ValueError, match="target polynomial order"):
        low_space.degree_elevation_matrix_from(high_space)


def test_dgfield_linear_arithmetic_rejects_incompatible_basis_or_mesh() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="dub_orth")
    u = V.zeros()

    with pytest.raises(ValueError, match="same basis type"):
        _ = u - DGSpace(mesh, 2, basis_type="bernstein").zeros()
    with pytest.raises(ValueError, match="same mesh object"):
        _ = u + DGSpace(reference_triangle_mesh(), 2, basis_type="dub_orth").zeros()


def test_vectordgfield_componentwise_arithmetic() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="bernstein")
    first_coeffs = np.arange(np.prod(V.shape), dtype=np.float64).reshape(V.shape)
    second_coeffs = first_coeffs + 2.0
    current = VectorDGField(
        (V.field(first_coeffs), V.field(second_coeffs)), name="current",
    )
    previous = VectorDGField(
        (V.field(0.5 * first_coeffs), V.field(0.25 * second_coeffs)), name="previous",
    )

    result = (2.0 * current - previous) / 2.0

    np.testing.assert_allclose(result.components[0].coeffs, 0.75 * first_coeffs)
    np.testing.assert_allclose(result.components[1].coeffs, 0.875 * second_coeffs)

    copied = current.copy(name="current_copy")
    assert copied.name == "current_copy"
    assert tuple(component.name for component in copied.components) == (
        "current_copy_0", "current_copy_1",
    )
    copied.components[0].coeffs[0, 0] += 1.0
    assert copied.components[0].coeffs[0, 0] != current.components[0].coeffs[0, 0]

    constant = VectorDGField((V.constant(2.0), V.constant(-3.0)))
    constant_result = 2.0 * constant - constant
    assert constant_result.constant_values == pytest.approx((2.0, -3.0))
    assert all(not component.coefficients_materialized for component in constant.components)
    assert all(not component.coefficients_materialized for component in constant_result.components)

    with pytest.raises(ValueError, match="same dimension"):
        _ = current + VectorDGField((V.zeros(),))
    equivalent_space = DGSpace(mesh, 1, basis_type="bernstein")
    compatible = VectorDGField((
        equivalent_space.field(0.25 * first_coeffs),
        equivalent_space.field(0.5 * second_coeffs),
    ))
    compatible_result = current - compatible
    assert all(component.space is V for component in compatible_result.components)
    np.testing.assert_allclose(compatible_result.components[0].coeffs, 0.75 * first_coeffs)
    np.testing.assert_allclose(compatible_result.components[1].coeffs, 0.5 * second_coeffs)
    higher_space = DGSpace(mesh, 2, basis_type="bernstein")
    higher_first_coeffs = np.arange(
        np.prod(higher_space.shape), dtype=np.float64,
    ).reshape(higher_space.shape)
    higher_second_coeffs = higher_first_coeffs + 3.0
    higher = VectorDGField((
        higher_space.field(higher_first_coeffs),
        higher_space.field(higher_second_coeffs),
    ))
    promoted = current + higher
    elevation = higher_space.degree_elevation_matrix_from(V)
    assert all(component.space is higher_space for component in promoted.components)
    np.testing.assert_allclose(
        promoted.components[0].coeffs, first_coeffs @ elevation + higher_first_coeffs,
    )
    np.testing.assert_allclose(
        promoted.components[1].coeffs, second_coeffs @ elevation + higher_second_coeffs,
    )
    incompatible_space = DGSpace(mesh, 1, basis_type="dub_orth")
    incompatible = VectorDGField((incompatible_space.zeros(), incompatible_space.zeros()))
    with pytest.raises(ValueError, match="same basis type"):
        _ = current - incompatible
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = current * previous


def test_vectordgfield_constructor_projects_callables() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_x = lambda x, y: x + 0.0 * y
    beta_y = lambda x, y: -y + 0.0 * x

    beta = VectorDGField((beta_x, beta_y), V, name="beta_h")

    assert beta.dim == 2
    np.testing.assert_allclose(beta.components[0].coeffs, V.project_callable(beta_x).coeffs, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(beta.components[1].coeffs, V.project_callable(beta_y).coeffs, atol=1e-13, rtol=1e-13)


def test_vectordgfield_constructor_accepts_scalar_coefficient_array() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    coeffs = np.arange(np.prod(V.shape), dtype=np.float64).reshape(V.shape)

    field = VectorDGField(coeffs, V, name="w_h")

    assert field.dim == 1
    np.testing.assert_array_equal(field.components[0].coeffs, coeffs)
    np.testing.assert_array_equal(field.as_component_first()[0], coeffs)
    np.testing.assert_array_equal(field.as_component_last()[..., 0], coeffs)


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


def test_dgfield_evaluate_and_call_match_existing_paths() -> None:
    mesh = reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="bernstein")
    u = V.project_callable(lambda x, y: x + 2.0 * y, name="linear")
    points = np.array([[-0.5, -0.5], [0.0, 0.0]], dtype=np.float64)

    np.testing.assert_allclose(u.evaluate(), u.values())
    np.testing.assert_allclose(u(), u.values())
    np.testing.assert_allclose(u.evaluate(points, reference=True), u.values_at_ref(points))
    np.testing.assert_allclose(u(points, reference=True), u.values_at_ref(points))
    np.testing.assert_allclose(u.evaluate(points), u.values_at_xy(points))
    np.testing.assert_allclose(u(points), u.values_at_xy(points))
    np.testing.assert_allclose(u(points[:, 0], points[:, 1]), u.values_at_xy(points))


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


def test_reference_edge_matrices_and_oriented_faces() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="bernstein")
    ref = V.quad_data

    assert mesh.loc2glob_edge.shape == (mesh.num_tri, 3)
    assert mesh.loc2oriented_ref_face.shape == (mesh.num_tri, 3)
    assert mesh.loc2oriented_face_coupling.shape == (mesh.num_tri, 3)
    np.testing.assert_array_equal(mesh.get_sigma_1(), mesh.loc2oriented_face_coupling)
    np.testing.assert_array_equal(mesh.get_oriented_face_coupling_indices(), mesh.loc2oriented_face_coupling)
    assert ref.face_element_test_trace_trial.shape == (3, ref.el_dof, ref.edg_dof)
    assert ref.face_element_test_trace_trial_reversed.shape == (3, ref.el_dof, ref.edg_dof)
    assert ref.face_trace_test_element_trial_oriented.shape == (6, ref.edg_dof, ref.el_dof)
    np.testing.assert_array_equal(ref.MKrfe_lst_p, ref.face_element_test_trace_trial)
    np.testing.assert_array_equal(ref.MKrfe_lst_n, ref.face_element_test_trace_trial_reversed)
    np.testing.assert_array_equal(ref.MKrfe_lst, ref.face_trace_test_element_trial_oriented)
    assert ref.M_rf_fc.shape == (ref.edg_dof, ref.edg_dof)
    assert ref.face_element_test_element_trial.shape == (3, ref.el_dof, ref.el_dof)
    np.testing.assert_array_equal(ref.MbdeKrf_lst, ref.face_element_test_element_trial)


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


def test_advection_reaction_solver_accepts_same_space_dg_coefficients() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="dub_orth")
    one = lambda x, y: np.ones_like(x)
    zero = lambda x, y: np.zeros_like(x)

    callable_result = solve_advection_reaction_hdg(one, (one, zero), 1.0, zero, V, verbose=False)
    beta_h = VectorDGField((one, zero), V, name="beta_h")
    source_h = DGField(one, V, name="source_h")
    reaction_h = DGField(one, V, name="reaction_h")

    field_result = solve_advection_reaction_hdg(source_h, beta_h, reaction_h, zero, V, verbose=False)
    tuple_field_result = solve_advection_reaction_hdg(
        source_h,
        beta_h.components,
        reaction_h,
        zero,
        V,
        verbose=False,
    )
    normalized_beta = as_vector_field(beta_h.components, V)

    np.testing.assert_allclose(field_result.trace, callable_result.trace, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(field_result.field.coeffs, callable_result.field.coeffs, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(tuple_field_result.trace, field_result.trace, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(tuple_field_result.field.coeffs, field_result.field.coeffs, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(normalized_beta.as_component_first(), beta_h.as_component_first())


def test_known_dof_elimination_matches_dense_schur_rhs() -> None:
    matrix = np.array(
        [
            [4.0, 1.0, 2.0],
            [3.0, 5.0, 7.0],
            [0.0, 2.0, 6.0],
        ]
    )
    rhs = np.array([1.0, 2.0, 3.0])
    rows, cols = np.nonzero(matrix)
    data = matrix[rows, cols]
    known_mask = np.array([False, True, False])
    known_values = np.array([0.0, 10.0, 0.0])

    reduced = eliminate_known_dofs(rows, cols, data, rhs, known_mask, known_values)

    expected_matrix = matrix[np.ix_(~known_mask, ~known_mask)]
    expected_rhs = rhs[~known_mask] - matrix[np.ix_(~known_mask, known_mask)] @ known_values[known_mask]
    dense_reduced = np.zeros_like(expected_matrix)
    dense_reduced[reduced.rows, reduced.cols] += reduced.data
    np.testing.assert_allclose(dense_reduced, expected_matrix)
    np.testing.assert_allclose(reduced.rhs, expected_rhs)

    full = expand_known_dofs(np.array([11.0, 12.0]), reduced)
    np.testing.assert_allclose(full, np.array([11.0, 10.0, 12.0]))


def test_permuted_global_solve_matches_unpermuted_direct_solve() -> None:
    matrix = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 2.0],
            [0.0, 2.0, 5.0],
        ]
    )
    rhs = np.array([1.0, 2.0, 3.0])
    rows, cols = np.nonzero(matrix)
    data = matrix[rows, cols]
    permutation = np.array([2, 0, 1])

    baseline = solve_global_system(rows, cols, data, rhs, rhs.size, solver="direct")
    permuted = solve_global_system(rows, cols, data, rhs, rhs.size, solver="direct", permutation=permutation)

    np.testing.assert_allclose(permuted.x, baseline.x, atol=1e-13, rtol=1e-13)
    assert permuted.permutation_size == rhs.size


def test_strongly_connected_component_order_detects_cycle() -> None:
    sources = np.array([0, 1, 2, 2], dtype=np.int64)
    targets = np.array([1, 0, 3, 4], dtype=np.int64)

    node_order, component_id, component_order, component_sizes, level_widths, timings = strongly_connected_component_order(
        5,
        sources,
        targets,
    )

    assert sorted(node_order.tolist()) == [0, 1, 2, 3, 4]
    assert component_id[0] == component_id[1]
    assert np.max(component_sizes) == 2
    assert component_order.size == component_sizes.size
    assert level_widths.num_levels >= 1
    assert timings["scc"] >= 0.0


def test_upwind_scc_trace_ordering_keeps_edge_blocks() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_dot_normal = np.zeros((mesh.num_tri, 3, V.quad_data.quads_JGL.shape[0]), dtype=np.float64)
    beta_dot_normal[:, 0, :] = -1.0
    beta_dot_normal[:, 1, :] = 1.0
    beta_dot_normal[:, 2, :] = 1.0

    ordering = upwind_scc_trace_ordering(mesh, beta_dot_normal, V.quad_data.edg_dof)

    assert ordering.edge_order.shape == (mesh.num_edg,)
    assert ordering.dof_permutation.shape == (mesh.num_edg * V.quad_data.edg_dof,)
    assert sorted(ordering.dof_permutation.tolist()) == list(range(ordering.dof_permutation.size))
    for block_start in range(0, ordering.dof_permutation.size, V.quad_data.edg_dof):
        block = ordering.dof_permutation[block_start:block_start + V.quad_data.edg_dof]
        np.testing.assert_array_equal(block, np.arange(block[0], block[0] + V.quad_data.edg_dof))


def test_advection_reaction_boundary_elimination_matches_penalty_path() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="dub_orth")
    one = lambda x, y: np.ones_like(x)
    zero = lambda x, y: np.zeros_like(x)

    penalty = solve_advection_reaction_hdg(
        one,
        (one, zero),
        1.0,
        one,
        V,
        solver="direct",
        preconditioner=None,
        boundary_mode="penalty",
        verbose=False,
    )
    reduced = solve_advection_reaction_hdg(
        one,
        (one, zero),
        1.0,
        one,
        V,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        verbose=False,
    )

    assert reduced.boundary_mode == "eliminate"
    np.testing.assert_allclose(reduced.trace, penalty.trace, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(reduced.field.coeffs, penalty.field.coeffs, atol=1e-12, rtol=1e-12)


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


def test_diffusion_reaction_solver_quadratic_smoke() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()

    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        V,
        diffusion=diffusion,
        stabilization=1.0,
        solver="direct",
        preconditioner=None,
        verbose=False,
    )

    assert result.field.coeffs.shape == V.shape
    assert result.flux.as_component_first().shape == (2,) + V.shape
    assert result.trace.shape == (mesh.num_edg * V.quad_data.edg_dof,)
    assert result.local_unknowns.shape == (mesh.num_tri, 3 * V.el_dof)
    assert result.field.l2_error(exact) < 1e-11


def test_diffusion_reaction_boundary_elimination_matches_penalty_path() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()

    penalty = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        V,
        diffusion=diffusion,
        stabilization=1.0,
        solver="direct",
        preconditioner=None,
        boundary_mode="penalty",
        verbose=False,
    )
    reduced = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        V,
        diffusion=diffusion,
        stabilization=1.0,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        verbose=False,
    )

    assert reduced.boundary_mode == "eliminate"
    np.testing.assert_allclose(reduced.trace, penalty.trace, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(reduced.field.coeffs, penalty.field.coeffs, atol=1e-11, rtol=1e-11)


def test_diffusion_reaction_boundary_elimination_weak_ilu_smoke() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()

    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        V,
        diffusion=diffusion,
        stabilization=1.0,
        solver="BICGSTAB",
        preconditioner="ilu",
        ilu_drop_tol=1.0,
        ilu_fill_factor=1.0,
        boundary_mode="eliminate",
        verbose=False,
    )

    assert result.global_solve_result is not None
    assert result.global_solve_result.info == 0
    assert result.field.l2_error(exact) < 1e-11


@pytest.mark.parametrize("solver", ("CG", "MINRES"))
def test_diffusion_reaction_symmetric_krylov_jacobi_smoke(solver) -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()

    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        V,
        diffusion=diffusion,
        stabilization=1.0,
        solver=solver,
        preconditioner="jacobi",
        scale_system=False,
        boundary_mode="eliminate",
        verbose=False,
    )

    assert result.global_solve_result is not None
    assert result.global_solve_result.info == 0
    assert result.field.l2_error(exact) < 1e-11


def test_trace_degree_elevation_from_linear_to_cubic() -> None:
    trace = np.array([2.0, 5.0, -1.0, 3.0])

    elevated = prolong_trace_coefficients(trace, source_order=1, target_order=3)

    expected = np.array([2.0, 3.0, 4.0, 5.0, -1.0, 1.0 / 3.0, 5.0 / 3.0, 3.0])
    np.testing.assert_allclose(elevated, expected)


def test_diffusion_reaction_solver_reuses_numpy_matrix_after_source_update() -> None:
    mesh = rectangle_mesh(3, 3, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    zero = lambda x, y: 0.0 * x + 0.0 * y
    options = DiffusionReactionHDGOptions(
        assembly_backend="numpy",
        boundary_mode="eliminate",
        solver="direct",
        cache_device_matrix=True,
        verbose=False,
    )
    solver = DiffusionReactionHDGSolver(
        V,
        source=lambda x, y: x + y,
        reaction=0.0,
        boundary_condition=zero,
        options=options,
    )

    first = solver.solve()
    rows = solver.solve_rows
    data = solver.solve_data
    solver.set_source(lambda x, y: 2.0 * x - y)
    second = solver.solve()

    assert solver.solve_rows is rows
    assert solver.solve_data is data
    assert second.assembly_backend == "numpy"
    assert second.timings.local_solver == 0.0
    assert second.field.l2_norm() != pytest.approx(first.field.l2_norm())


def test_diffusion_reaction_bootstrap_initial_guess_smoke() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()

    result = solve_diffusion_reaction_hdg_with_bootstrap(
        source,
        reaction,
        exact,
        V,
        diffusion=diffusion,
        stabilization=1.0,
        solver="BICGSTAB",
        preconditioner="ilu",
        bootstrap_order=1,
        verbose=False,
    )

    assert result.bootstrap_order == 1
    assert result.initial_guess is not None
    assert result.initial_guess.shape == result.trace.shape
    assert result.timings.initial_guess >= 0.0
    assert result.field.l2_error(exact) < 1e-11


def test_initial_guess_boundary_trace_is_imposed() -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 2, basis_type="dub_orth")
    guess = np.zeros(mesh.num_edg * V.quad_data.edg_dof)
    boundary_trace = np.arange(mesh.num_edg * V.quad_data.edg_dof, dtype=np.float64).reshape(
        mesh.num_edg,
        V.quad_data.edg_dof,
    )

    corrected = impose_boundary_trace_on_guess(guess, boundary_trace, V)
    corrected_edges = corrected.reshape(mesh.num_edg, V.quad_data.edg_dof)

    np.testing.assert_allclose(corrected_edges[mesh.bnd_edges_inds], boundary_trace[mesh.bnd_edges_inds])
    np.testing.assert_allclose(corrected_edges[mesh.int_edges_inds], 0.0)


@pytest.mark.parametrize("problem", (trigonometric_poisson_case, lshape_singular_harmonic_case))
def test_diffusion_reaction_nonrectangular_manufactured_cases_smoke(problem) -> None:
    mesh = split_reference_triangle_mesh()
    V = DGSpace(mesh, 1, basis_type="dub_orth")
    diffusion, reaction, source, exact = problem()

    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        V,
        diffusion=diffusion,
        stabilization=1.0,
        solver="direct",
        preconditioner=None,
        verbose=False,
    )

    assert result.field.coeffs.shape == V.shape
    assert result.flux.as_component_first().shape == (2,) + V.shape
    assert result.trace.shape == (mesh.num_edg * V.quad_data.edg_dof,)
    assert np.all(np.isfinite(result.field.coeffs))
    assert np.all(np.isfinite(result.flux.as_component_first()))
    assert np.isfinite(result.field.l2_error(exact))


def test_gmsh_basic_shape_meshes() -> None:
    pytest.importorskip("gmsh")

    for mesh in (
        gmsh_rectangle_mesh(1.0, verbosity=0),
        gmsh_disc_mesh(1.0, verbosity=0),
        gmsh_triangle_mesh(1.0, verbosity=0),
        gmsh_lshape_mesh(1.0, verbosity=0),
        gmsh_star_mesh(1.0, corners=5, verbosity=0),
        gmsh_smooth_star_mesh(1.0, boundary_points=40, radius=1.5, amplitude=0.32, mode=5, verbosity=0),
    ):
        assert mesh.num_tri > 0
        assert mesh.num_edg > 0
        assert np.all(mesh.aff_jacs > 0.0)


def test_dgspace_layout_and_trace_space_formalism() -> None:
    mesh = rectangle_mesh(1, 1, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=8)

    layout = space.layout
    assert layout.scalar_shape == space.shape
    assert layout.trace_shape == (mesh.num_edg, space.quad_data.edg_dof)
    assert layout.reduced_trace_shape == (mesh.int_edges_inds.size, space.quad_data.edg_dof)
    assert layout.trace_vector_size == mesh.num_edg * space.quad_data.edg_dof

    for kind in ("legacy-lagrange", "legendre-modal", "bernstein"):
        trace = space.trace_space(kind)
        assert trace is space.trace_space(kind)
        assert trace.kind == kind
        assert trace.bas_of_bd_quads.shape[:2] == (3, space.el_dof)
        assert trace.bas1d_of_ref_edg_qds.shape[0] == space.quad_data.edg_dof
        assert trace.face_trace_test_element_trial_oriented.shape == (6, space.quad_data.edg_dof, space.el_dof)
        boundary = trace.boundary_coefficients(lambda x, y: x + 2.0 * y)
        assert boundary.shape == layout.trace_shape
        assert np.all(np.isfinite(boundary))


def test_trace_boundary_coefficients_separate_nodal_and_modal_paths() -> None:
    mesh = rectangle_mesh(1, 1, xlim=(-0.5, 1.25), ylim=(-1.0, 0.75))
    space = DGSpace(mesh, 2, basis_type="dub_orth", volume_quad_1d=6)

    def boundary_condition(x, y):
        return 0.2 + x**3 - 0.35 * y**3 + 0.1 * x * y

    legacy = space.trace_space("legacy-lagrange")
    legacy_coeffs = legacy.boundary_coefficients(boundary_condition)
    edge_vertices = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]]
    t = legacy.interpolation_nodes
    nodal_points = 0.5 * (
        (1.0 - t)[None, :, None] * edge_vertices[:, 0:1, :]
        + (1.0 + t)[None, :, None] * edge_vertices[:, 1:2, :]
    )
    expected_nodal = boundary_condition(nodal_points[:, :, 0], nodal_points[:, :, 1])
    np.testing.assert_allclose(legacy_coeffs[mesh.bnd_edges_inds], expected_nodal, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(boundary_trace_coefficients(boundary_condition, space), legacy_coeffs)

    for kind in ("legendre-modal", "bernstein"):
        trace = space.trace_space(kind)
        coeffs = trace.boundary_coefficients(boundary_condition)
        t = trace.quads
        quad_points = 0.5 * (
            (1.0 - t)[None, :, None] * edge_vertices[:, 0:1, :]
            + (1.0 + t)[None, :, None] * edge_vertices[:, 1:2, :]
        )
        values = boundary_condition(quad_points[:, :, 0], quad_points[:, :, 1])
        rhs = (values * trace.weights[None, :]) @ trace.bas1d_of_ref_edg_qds.T
        expected_projected = rhs @ np.linalg.inv(trace.M_rf_fc)
        np.testing.assert_allclose(coeffs[mesh.bnd_edges_inds], expected_projected, rtol=1.0e-14, atol=1.0e-14)
