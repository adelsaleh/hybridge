from __future__ import annotations

import numpy as np
import pytest

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.assembly import matrices_numpy as hdg_mats
from hdgfem.backends.numba import (
    assemble_local_advection_reaction_numba,
    assemble_projected_trace_system_eliminated_numba,
    assemble_projected_trace_system_numba,
)
from hdgfem.linalg.system import assemble_global_matrix, eliminate_known_dofs
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.solvers.adv_rea import (
    AdvectionReactionHDGSolver,
    solve_advection_reaction_hdg,
)
from hdgfem.core.space import DGField, DGSpace, VectorDGField
from scripts.advection_reaction.adv_rea_cases import test2 as adv_rea_test2


pytest.importorskip("numba")


def _projected_test2_fields(space: DGSpace):
    beta_x, beta_y, reaction, source, exact = adv_rea_test2()
    return (
        VectorDGField((beta_x, beta_y), space, name="beta_h"),
        DGField(reaction, space, name="reaction_h"),
        DGField(source, space, name="source_h"),
        exact,
    )


def _elementwise_constant_field(space: DGSpace, values, *, name: str) -> DGField:
    one = space.project_callable(lambda x, y: np.ones_like(x), name=f"{name}_one")
    coeffs = np.asarray(values, dtype=np.float64)[:, None] * one.coeffs
    return space.field(np.ascontiguousarray(coeffs), name=name)


def _numpy_weighted_advection_trace_system(source_h, beta_h, reaction_h, boundary_condition, space: DGSpace):
    beta_dot_normal = hdg_mats.advective_boundary_normal(beta_h, space)
    tau_face, gamma_face = hdg_mats.advection_trace_weights_from_normal_flux(space, beta_dot_normal)
    local_mats = np.ascontiguousarray(hdg_mats.boundary_mass_from_trace_stabilization(space, tau_face))
    scratch = np.empty_like(local_mats)
    hdg_mats.add_reaction_mass(local_mats, reaction_h, space, scratch=scratch)
    hdg_mats.add_advection_mats(local_mats, space, beta_h, scale=-1.0)
    local_solver = np.linalg.inv(local_mats)
    element_boundary_mats = hdg_mats.element_boundary_mats_from_trace_weight(space, gamma_face)
    source_moments = hdg_assembly.source_moments(source_h, space)
    trace_lift = hdg_mats.advection_trace_lift_from_stabilization(space, tau_face)
    trace_blocks = hdg_assembly.element_to_trace_matrix_from_lift(
        trace_lift,
        local_solver,
        element_boundary_mats,
        space,
    )
    rows, cols = hdg_assembly.trace_matrix_indices(space, interior_mass_mode="face")
    interior_mass_blocks = hdg_mats.advection_interior_trace_mass_blocks_from_weight(space, gamma_face)
    data = hdg_assembly.trace_matrix_data(
        trace_blocks,
        space,
        1e20,
        interior_mass_mode="face",
        interior_mass_blocks=interior_mass_blocks,
    )
    rhs, boundary_trace = hdg_assembly.trace_rhs_from_lift(
        trace_lift,
        source_moments,
        local_solver,
        boundary_condition,
        space,
        1e20,
    )
    return hdg_assembly.TraceSystem(rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace)


def test_discontinuous_beta_uses_side_weighted_trace_mass() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h = VectorDGField(
        (
            _elementwise_constant_field(space, [1.0, 3.0], name="beta_x"),
            _elementwise_constant_field(space, [0.0, 0.0], name="beta_y"),
        ),
        name="beta_h",
    )

    beta_dot_normal = hdg_mats.advective_boundary_normal(beta_h, space)
    _, gamma_face = hdg_mats.advection_trace_weights_from_normal_flux(space, beta_dot_normal)
    side_blocks = hdg_mats.advection_interior_trace_mass_blocks_from_weight(space, gamma_face)

    assert side_blocks.shape == (mesh.interior_elements.size, space.quad_data.edg_dof, space.quad_data.edg_dof)
    side_sum = np.zeros((mesh.int_edges_inds.size, space.quad_data.edg_dof, space.quad_data.edg_dof))
    for side, (element, face) in enumerate(zip(mesh.interior_elements, mesh.interior_faces)):
        edge = mesh.loc2glob_edge[element, face]
        edge_pos = int(np.where(mesh.int_edges_inds == edge)[0][0])
        side_sum[edge_pos] += side_blocks[side]

    old_unweighted = mesh.edge_jacs[mesh.int_edges_inds, None, None] * space.quad_data.M_rf_fc[None]
    assert not np.allclose(side_sum, old_unweighted)


@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
def test_numba_solve_matches_numpy_with_dg_stabilization(boundary_mode: str) -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    source_h = _elementwise_constant_field(space, [1.0, 1.0], name="source_h")
    reaction_h = _elementwise_constant_field(space, [1.5, 2.0], name="reaction_h")
    beta_h = VectorDGField(
        (
            _elementwise_constant_field(space, [1.0, 3.0], name="beta_x"),
            _elementwise_constant_field(space, [0.25, -0.5], name="beta_y"),
        ),
        name="beta_h",
    )
    tau_h = _elementwise_constant_field(space, [6.0, 8.0], name="tau_h")
    zero = lambda x, y: np.zeros_like(x)

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        zero,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode=boundary_mode,
        assembly_backend="numpy",
        advection_stabilization=tau_h,
        verbose=False,
    )
    numba_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        zero,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode=boundary_mode,
        assembly_backend="numba",
        advection_stabilization=tau_h,
        verbose=False,
    )

    np.testing.assert_allclose(numba_result.trace, numpy_result.trace, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(numba_result.field.coeffs, numpy_result.field.coeffs, rtol=1e-11, atol=1e-11)


def test_numba_local_assembly_matches_numpy_projected_coefficients() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, _, _ = _projected_test2_fields(space)
    beta_dot_normal = hdg_mats.advective_boundary_normal(beta_h, space)

    numba_data = assemble_local_advection_reaction_numba(
        space,
        beta_field=beta_h,
        beta_callables=None,
        beta_dot_normal=beta_dot_normal,
        reaction=reaction_h,
    )

    numpy_local = np.ascontiguousarray(hdg_mats.boundary_mass_from_normal_flux(space, beta_dot_normal))
    scratch = np.empty_like(numpy_local)
    hdg_mats.add_reaction_mass(numpy_local, reaction_h, space, scratch=scratch)
    hdg_mats.add_advection_mats(numpy_local, space, beta_h, scale=-1.0)
    numpy_boundary = hdg_mats.element_boundary_mats_from_normal_flux(space, beta_dot_normal)

    np.testing.assert_allclose(numba_data.local_mats, numpy_local, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_data.element_boundary_mats, numpy_boundary, rtol=1e-12, atol=1e-12)


def test_numba_fused_trace_system_matches_numpy_projected_coefficients() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)
    numpy_trace_system = _numpy_weighted_advection_trace_system(source_h, beta_h, reaction_h, exact, space)

    numba_trace_system = assemble_projected_trace_system_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
    ).trace_system

    np.testing.assert_array_equal(numba_trace_system.rows, numpy_trace_system.rows)
    np.testing.assert_array_equal(numba_trace_system.cols, numpy_trace_system.cols)
    np.testing.assert_allclose(numba_trace_system.data, numpy_trace_system.data, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(numba_trace_system.rhs, numpy_trace_system.rhs, rtol=1e-11, atol=1e-11)


def test_numba_eliminated_trace_system_matches_generic_elimination() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)

    full_trace_system = assemble_projected_trace_system_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
    ).trace_system
    generic_reduction = eliminate_known_dofs(
        full_trace_system.rows,
        full_trace_system.cols,
        full_trace_system.data,
        full_trace_system.rhs,
        ~hdg_assembly.free_trace_dofs(space),
        full_trace_system.boundary_trace.ravel(),
    )

    eliminated = assemble_projected_trace_system_eliminated_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
    )
    direct_reduction = eliminated.reduction

    np.testing.assert_array_equal(direct_reduction.free_mask, generic_reduction.free_mask)
    np.testing.assert_array_equal(direct_reduction.known_mask, generic_reduction.known_mask)
    np.testing.assert_array_equal(direct_reduction.old_to_new, generic_reduction.old_to_new)
    np.testing.assert_array_equal(direct_reduction.rows, generic_reduction.rows)
    np.testing.assert_array_equal(direct_reduction.cols, generic_reduction.cols)
    np.testing.assert_allclose(direct_reduction.data, generic_reduction.data, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(direct_reduction.rhs, generic_reduction.rhs, rtol=1e-11, atol=1e-11)


def test_numba_eliminated_block_coo_reconstructs_trace_matrix() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)

    assembly = assemble_projected_trace_system_eliminated_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        return_block_coo=True,
    )
    assert assembly.block_rows is not None
    assert assembly.block_cols is not None
    assert assembly.block_data is not None

    trace_system = assembly.trace_system
    scalar_matrix = assemble_global_matrix(
        trace_system.rows,
        trace_system.cols,
        trace_system.data,
        trace_system.rhs.size,
    ).tocsr()

    edg_dof = space.quad_data.edg_dof
    local_rows = np.arange(edg_dof, dtype=np.int64)
    local_cols = np.arange(edg_dof, dtype=np.int64)
    block_scalar_rows = np.broadcast_to(
        assembly.block_rows[:, None, None] * edg_dof + local_rows[None, :, None],
        assembly.block_data.shape,
    ).ravel()
    block_scalar_cols = np.broadcast_to(
        assembly.block_cols[:, None, None] * edg_dof + local_cols[None, None, :],
        assembly.block_data.shape,
    ).ravel()
    block_matrix = assemble_global_matrix(
        block_scalar_rows,
        block_scalar_cols,
        assembly.block_data.ravel(),
        trace_system.rhs.size,
    ).tocsr()

    diff = scalar_matrix - block_matrix
    diff.sum_duplicates()
    assert diff.nnz == 0 or np.max(np.abs(diff.data)) <= 1.0e-12


@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
def test_numba_solve_matches_numpy_projected_coefficients(boundary_mode: str) -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode=boundary_mode,
        assembly_backend="numpy",
        verbose=False,
    )
    numba_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode=boundary_mode,
        assembly_backend="numba",
        verbose=False,
    )

    assert numba_result.assembly_backend == "numba"
    assert numba_result.local_solver is None
    np.testing.assert_allclose(numba_result.trace, numpy_result.trace, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(numba_result.field.coeffs, numpy_result.field.coeffs, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        numba_result.field.l2_error(exact),
        numpy_result.field.l2_error(exact),
        rtol=1e-11,
        atol=1e-13,
    )


@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
def test_numba_upwind_ordered_solve_matches_numpy_projected_coefficients(boundary_mode: str) -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)

    numpy_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode=boundary_mode,
        trace_ordering="upwind-scc",
        assembly_backend="numpy",
        verbose=False,
    )
    numba_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode=boundary_mode,
        trace_ordering="upwind-scc",
        assembly_backend="numba",
        verbose=False,
    )

    assert numba_result.global_solve_result.permutation_elapsed_seconds == 0.0
    np.testing.assert_allclose(numba_result.trace, numpy_result.trace, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(numba_result.field.coeffs, numpy_result.field.coeffs, rtol=1e-11, atol=1e-11)


def test_stateful_numba_solver_matches_function_and_caches_artifacts() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)

    function_result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        trace_ordering="upwind-scc",
        assembly_backend="numba",
        verbose=False,
    )

    solver = AdvectionReactionHDGSolver(
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        trace_ordering="upwind-scc",
        assembly_backend="numba",
        verbose=False,
    )
    solver.set_discrete_problem(source_h, beta_h, reaction_h, exact)
    class_result = solver.solve()

    np.testing.assert_allclose(class_result.trace, function_result.trace, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(class_result.field.coeffs, function_result.field.coeffs, rtol=1e-11, atol=1e-11)
    assert solver.result is class_result
    assert solver.trace is class_result.trace
    assert solver.solve_rows is class_result.solve_matrix_rows
    assert solver.solve_cols is class_result.solve_matrix_cols
    assert solver.solve_data is class_result.solve_matrix_data
    assert solver.solve_rhs is class_result.solve_rhs
    assert solver.boundary_trace is class_result.boundary_trace
    assert solver.reduction is class_result.reduction
    assert solver.ordering_result is class_result.ordering_result
    assert solver.global_solve_result is class_result.global_solve_result
    assert solver.reduction is not None


def test_stateful_solver_source_update_invalidates_previous_solution() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 1, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)

    solver = AdvectionReactionHDGSolver(
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numba",
        verbose=False,
    )
    solver.set_discrete_problem(source_h, beta_h, reaction_h, exact)
    first = solver.solve()
    assert solver.result is first

    scaled_source = space.field(1.1 * source_h.coeffs, name="scaled_source")
    solver.set_source(scaled_source)
    assert solver.result is None
    second = solver.solve()

    assert solver.result is second
    assert second.trace is not None
    assert not np.allclose(second.trace, first.trace)


def test_numba_backend_requires_projected_source_and_beta() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 1, basis_type="dub_orth")
    beta_x, beta_y, reaction, source, exact = adv_rea_test2()

    with pytest.raises(TypeError, match="projected beta"):
        solve_advection_reaction_hdg(
            source,
            (beta_x, beta_y),
            reaction,
            exact,
            space,
            solver="direct",
            preconditioner=None,
            assembly_backend="numba",
            verbose=False,
        )


def test_numba_local_solver_cache_is_explicit() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 1, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)

    uncached = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        solver="direct",
        preconditioner=None,
        assembly_backend="numba",
        verbose=False,
    )
    cached = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        solver="direct",
        preconditioner=None,
        assembly_backend="numba",
        cache_local_solvers=True,
        verbose=False,
    )

    assert uncached.local_solver is None
    assert cached.local_solver is not None
    np.testing.assert_allclose(cached.trace, uncached.trace, rtol=1e-12, atol=1e-12)
