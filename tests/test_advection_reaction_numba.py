from __future__ import annotations

import numpy as np
import pytest

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.assembly import matrices_numpy as hdg_mats
import hdgfem.core.mass as core_mass
from hdgfem.backends import UnsupportedBackendConfigurationError
from hdgfem.backends.numba import (
    assemble_local_advection_reaction_numba,
    assemble_projected_trace_system_eliminated_numba,
    assemble_projected_trace_system_numba,
    assemble_projected_trace_system_zero_flux_numba,
)
from hdgfem.linalg.system import assemble_global_matrix, eliminate_known_dofs
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.solvers.advection_reaction import (
    AdvectionReactionHDGSolver,
    solve_advection_reaction_hdg,
)
from hdgfem.core.space import DGField, DGSpace, VectorDGField
from scripts.advection_reaction.cases import test2 as adv_rea_test2


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


def test_dg_stabilization_uses_field_space_reference_tables(monkeypatch) -> None:
    mesh = rectangle_mesh(1, 1)
    test_space = DGSpace(mesh, 3, basis_type="dub_orth")
    field_space = DGSpace(mesh, 2, basis_type="dub_orth")
    tau_h = field_space.project_callable(lambda x, y: 3.0 + x - 0.25 * y, name="tau_h")
    trace_space = test_space.trace_space("legendre-modal")
    basis = hdg_mats.dg_field_basis_on_trace_faces(field_space, trace_space)
    expected = np.ascontiguousarray(np.einsum("Ki,fiq->Kfq", tau_h.coeffs, basis, optimize=True))

    def fail_generic_evaluation(*_args, **_kwargs):
        raise AssertionError("DG stabilization must use coefficient/reference-table contraction")

    monkeypatch.setattr(DGField, "values_at_ref", fail_generic_evaluation)
    beta_dot_normal = np.zeros_like(expected)
    actual = hdg_mats.advection_trace_stabilization_values(
        test_space,
        beta_dot_normal,
        tau_h,
        trace_space=trace_space,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_context_aware_callable_stabilization_receives_element_and_face_ids() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    trace_space = space.trace_space("legacy-lagrange")
    shape = (mesh.num_tri, 3, trace_space.weights.size)

    def tau(x, y, element, face):
        return 4.0 + 0.0 * x * y + element + 0.25 * face

    actual = hdg_mats.advection_trace_stabilization_values(
        space,
        np.zeros(shape, dtype=np.float64),
        tau,
        trace_space=trace_space,
    )
    expected = 4.0 + np.arange(mesh.num_tri)[:, None, None] + 0.25 * np.arange(3)[None, :, None]
    expected = np.broadcast_to(expected, shape)
    assert actual.shape == shape
    np.testing.assert_allclose(actual, expected)



def _numpy_weighted_advection_trace_system(
        source_h,
        beta_h,
        reaction_h,
        boundary_condition,
        space: DGSpace,
        *,
        trace_basis: str = "legacy-lagrange",
        zero_boundary_flux: bool = False,
):
    trace_space = space.trace_space(trace_basis)
    beta_dot_normal = hdg_mats.advective_boundary_normal(beta_h, space, trace_space=trace_space)
    tau_face, gamma_face = hdg_mats.advection_trace_weights_from_normal_flux(
        space,
        beta_dot_normal,
        trace_space=trace_space,
    )
    if zero_boundary_flux:
        edge_is_boundary = np.zeros(space.mesh.num_edg, dtype=bool)
        edge_is_boundary[space.mesh.bnd_edges_inds] = True
        boundary_faces = edge_is_boundary[space.mesh.loc2glob_edge]
        tau_face = tau_face.copy()
        gamma_face = gamma_face.copy()
        tau_face[boundary_faces] = 0.0
        gamma_face[boundary_faces] = 0.0
    local_mats = np.ascontiguousarray(
        hdg_mats.boundary_mass_from_trace_stabilization(space, tau_face, trace_space=trace_space)
    )
    scratch = np.empty_like(local_mats)
    core_mass.add_reaction_mass(local_mats, reaction_h, space, scratch=scratch)
    hdg_mats.add_advection_mats(local_mats, space, beta_h, scale=-1.0)
    local_solver = np.linalg.inv(local_mats)
    element_boundary_mats = hdg_mats.element_boundary_mats_from_trace_weight(
        space,
        gamma_face,
        trace_space=trace_space,
    )
    source_moments = hdg_assembly.source_moments(source_h, space)
    trace_lift = hdg_mats.advection_trace_lift_from_stabilization(space, tau_face, trace_space=trace_space)
    trace_blocks = hdg_assembly.element_to_trace_matrix_from_lift(
        trace_lift,
        local_solver,
        element_boundary_mats,
        space,
        trace_space=trace_space,
    )
    rows, cols = hdg_assembly.trace_matrix_indices(space, interior_mass_mode="face", trace_space=trace_space)
    interior_mass_blocks = hdg_mats.advection_interior_trace_mass_blocks_from_weight(
        space,
        gamma_face,
        trace_space=trace_space,
    )
    data = hdg_assembly.trace_matrix_data(
        trace_blocks,
        space,
        1e20,
        interior_mass_mode="face",
        interior_mass_blocks=interior_mass_blocks,
        trace_space=trace_space,
    )
    rhs, boundary_trace = hdg_assembly.trace_rhs_from_lift(
        trace_lift,
        source_moments,
        local_solver,
        boundary_condition,
        space,
        1e20,
        trace_space=trace_space,
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
    core_mass.add_reaction_mass(numpy_local, reaction_h, space, scratch=scratch)
    hdg_mats.add_advection_mats(numpy_local, space, beta_h, scale=-1.0)
    numpy_boundary = hdg_mats.element_boundary_mats_from_normal_flux(space, beta_dot_normal)

    np.testing.assert_allclose(numba_data.local_mats, numpy_local, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_data.element_boundary_mats, numpy_boundary, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
def test_numba_fused_trace_system_matches_numpy_projected_coefficients(trace_basis: str) -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)
    trace_space = space.trace_space(trace_basis)
    numpy_trace_system = _numpy_weighted_advection_trace_system(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        trace_basis=trace_basis,
    )

    numba_trace_system = assemble_projected_trace_system_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        trace_space=trace_space,
    ).trace_system

    np.testing.assert_array_equal(numba_trace_system.rows, numpy_trace_system.rows)
    np.testing.assert_array_equal(numba_trace_system.cols, numpy_trace_system.cols)
    np.testing.assert_allclose(numba_trace_system.data, numpy_trace_system.data, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(numba_trace_system.rhs, numpy_trace_system.rhs, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
def test_numba_eliminated_trace_system_matches_generic_elimination(trace_basis: str) -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, exact = _projected_test2_fields(space)
    trace_space = space.trace_space(trace_basis)

    full_trace_system = assemble_projected_trace_system_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        trace_space=trace_space,
    ).trace_system
    generic_reduction = eliminate_known_dofs(
        full_trace_system.rows,
        full_trace_system.cols,
        full_trace_system.data,
        full_trace_system.rhs,
        ~hdg_assembly.free_trace_dofs(space, trace_space=trace_space),
        full_trace_system.boundary_trace.ravel(),
    )

    eliminated = assemble_projected_trace_system_eliminated_numba(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        trace_space=trace_space,
    )
    direct_reduction = eliminated.reduction

    np.testing.assert_array_equal(direct_reduction.free_mask, generic_reduction.free_mask)
    np.testing.assert_array_equal(direct_reduction.known_mask, generic_reduction.known_mask)
    np.testing.assert_array_equal(direct_reduction.old_to_new, generic_reduction.old_to_new)
    np.testing.assert_array_equal(direct_reduction.rows, generic_reduction.rows)
    np.testing.assert_array_equal(direct_reduction.cols, generic_reduction.cols)
    np.testing.assert_allclose(direct_reduction.data, generic_reduction.data, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(direct_reduction.rhs, generic_reduction.rhs, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
def test_numba_zero_flux_trace_system_matches_numpy_zeroed_boundary_flux(trace_basis: str) -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, _ = _projected_test2_fields(space)
    trace_space = space.trace_space(trace_basis)
    zero = lambda x, y: np.zeros_like(x)

    full_trace_system = _numpy_weighted_advection_trace_system(
        source_h,
        beta_h,
        reaction_h,
        zero,
        space,
        trace_basis=trace_basis,
        zero_boundary_flux=True,
    )
    generic_reduction = eliminate_known_dofs(
        full_trace_system.rows,
        full_trace_system.cols,
        full_trace_system.data,
        full_trace_system.rhs,
        ~hdg_assembly.free_trace_dofs(space, trace_space=trace_space),
        full_trace_system.boundary_trace.ravel(),
    )

    zero_flux = assemble_projected_trace_system_zero_flux_numba(
        source_h,
        beta_h,
        reaction_h,
        space,
        trace_space=trace_space,
    )
    direct_reduction = zero_flux.reduction

    np.testing.assert_array_equal(direct_reduction.free_mask, generic_reduction.free_mask)
    np.testing.assert_array_equal(direct_reduction.known_mask, generic_reduction.known_mask)
    np.testing.assert_array_equal(direct_reduction.old_to_new, generic_reduction.old_to_new)
    np.testing.assert_array_equal(direct_reduction.rows, generic_reduction.rows)
    np.testing.assert_array_equal(direct_reduction.cols, generic_reduction.cols)
    np.testing.assert_allclose(direct_reduction.data, generic_reduction.data, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(direct_reduction.rhs, generic_reduction.rhs, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(zero_flux.trace_system.boundary_trace, 0.0)
    assert "boundary_flux_zeroing" in zero_flux.timings


def test_zero_flux_numba_requires_none_boundary_condition() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, _ = _projected_test2_fields(space)

    def raising_boundary(x, y):
        raise AssertionError("rejected boundary callables must not be sampled")

    missing_boundary_solver = AdvectionReactionHDGSolver(
        space,
        source=source_h,
        beta=beta_h,
        reaction=reaction_h,
        solver="direct",
        preconditioner=None,
        boundary_mode="zero-flux",
        assembly_backend="numba",
        verbose=False,
    )
    missing_boundary = missing_boundary_solver.solve()

    assert missing_boundary.boundary_mode == "zero-flux"
    np.testing.assert_allclose(missing_boundary.boundary_trace, 0.0)

    match = "boundary_condition must be None"
    for invalid_boundary in (raising_boundary, 0.0):
        with pytest.raises(ValueError, match=match):
            solve_advection_reaction_hdg(
                source_h,
                beta_h,
                reaction_h,
                invalid_boundary,
                space,
                solver="direct",
                preconditioner=None,
                boundary_mode="zero-flux",
                assembly_backend="numba",
                verbose=False,
            )
        with pytest.raises(ValueError, match=match):
            AdvectionReactionHDGSolver(
                space,
                source=source_h,
                beta=beta_h,
                reaction=reaction_h,
                boundary_condition=invalid_boundary,
                solver="direct",
                preconditioner=None,
                boundary_mode="zero-flux",
                assembly_backend="numba",
                verbose=False,
            )
        with pytest.raises(ValueError, match=match):
            assemble_projected_trace_system_eliminated_numba(
                source_h,
                beta_h,
                reaction_h,
                invalid_boundary,
                space,
                zero_boundary_flux=True,
            )


def test_zero_flux_numba_upwind_scc_matches_unordered() -> None:
    mesh = rectangle_mesh(2, 1)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    beta_h, reaction_h, source_h, _ = _projected_test2_fields(space)

    unordered = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        None,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode="zero-flux",
        trace_ordering="none",
        assembly_backend="numba",
        verbose=False,
    )
    ordered = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        None,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode="zero-flux",
        trace_ordering="upwind-scc",
        assembly_backend="numba",
        verbose=False,
    )

    assert ordered.ordering_result is not None
    assert ordered.solve_rhs.size == space.mesh.int_edges_inds.size * space.trace_space().edg_dof
    np.testing.assert_allclose(ordered.trace, unordered.trace, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(ordered.field.coeffs, unordered.field.coeffs, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("backend", ("numpy", "cupy", "auto"))
def test_zero_flux_rejects_backends_without_zero_flux_support(backend: str) -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 1, basis_type="dub_orth")
    source_h = space.constant(1.0, name="source_h")
    reaction_h = space.constant(1.0, name="reaction_h")
    beta_h = (space * space).constant((0.5, -0.25), name="beta_h")

    with pytest.raises(
        UnsupportedBackendConfigurationError,
        match=r"boundary_mode='zero-flux' is unsupported",
    ):
        solve_advection_reaction_hdg(
            source_h,
            beta_h,
            reaction_h,
            None,
            space,
            solver="direct",
            preconditioner=None,
            boundary_mode="zero-flux",
            assembly_backend=backend,
            verbose=False,
        )


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


@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
def test_numba_solve_matches_numpy_projected_coefficients(boundary_mode: str, trace_basis: str) -> None:
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
        trace_basis=trace_basis,
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
        trace_basis=trace_basis,
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


def test_numba_backend_requires_projected_source_reaction_and_beta() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 1, basis_type="dub_orth")
    beta_x, beta_y, reaction, source, exact = adv_rea_test2()
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_h = VectorDGField((beta_x, beta_y), space, name="beta_h")

    with pytest.raises(TypeError, match="requires source to be a DGField"):
        solve_advection_reaction_hdg(
            source,
            beta_h,
            reaction_h,
            exact,
            space,
            solver="direct",
            preconditioner=None,
            assembly_backend="numba",
            verbose=False,
        )

    with pytest.raises(TypeError, match="requires reaction to be a DGField"):
        solve_advection_reaction_hdg(
            source_h,
            beta_h,
            reaction,
            exact,
            space,
            solver="direct",
            preconditioner=None,
            assembly_backend="numba",
            verbose=False,
        )

    with pytest.raises(TypeError, match="requires beta to be a VectorDGField"):
        solve_advection_reaction_hdg(
            source_h,
            (beta_x, beta_y),
            reaction_h,
            exact,
            space,
            solver="direct",
            preconditioner=None,
            assembly_backend="numba",
            verbose=False,
        )


def test_raw_cuda_rejects_explicit_advection_stabilization_before_device_setup() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 1, basis_type="dub_orth")
    source_h = space.constant(1.0, name="source_h")
    reaction_h = space.constant(0.5, name="reaction_h")
    beta_h = (space * space).constant((0.75, -0.25), name="beta_h")
    boundary = lambda x, y: np.zeros_like(x)

    with pytest.raises(NotImplementedError, match="advection_stabilization=None"):
        solve_advection_reaction_hdg(
            source_h,
            beta_h,
            reaction_h,
            boundary,
            space,
            solver="direct",
            preconditioner=None,
            boundary_mode="eliminate",
            assembly_backend="raw-cuda",
            advection_stabilization=2.0,
            verbose=False,
        )


def test_advection_reaction_numba_constant_source_reaction_fields_stay_lazy() -> None:
    mesh = rectangle_mesh(1, 1)
    space = DGSpace(mesh, 1, basis_type="dub_orth")
    source_h = space.constant(1.0, name="source_h")
    reaction_h = space.constant(0.5, name="reaction_h")
    beta_h = (space * space).constant((0.75, -0.25), name="beta_h")
    boundary = lambda x, y: np.zeros_like(x)

    result = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        boundary,
        space,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numba",
        verbose=False,
    )

    assert result.field.coeffs.shape == space.shape
    assert not source_h.coefficients_materialized
    assert not reaction_h.coefficients_materialized


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
