from __future__ import annotations

import numpy as np
import pytest

from dgfem.assembly import hdg as hdg_assembly
from dgfem.assembly import matrices_numpy as hdg_mats
from dgfem.backends.numba import (
    assemble_local_advection_reaction_numba,
    assemble_projected_trace_system_eliminated_numba,
    assemble_projected_trace_system_numba,
)
from dgfem.linalg.system import eliminate_known_dofs
from dgfem.core.mesh import rectangle_mesh
from dgfem.solvers.adv_rea import (
    AdvectionReactionHDGSolver,
    solve_advection_reaction_hdg,
    test2 as adv_rea_test2,
)
from dgfem.core.space import DGField, DGSpace, VectorDGField


pytest.importorskip("numba")


def _projected_test2_fields(space: DGSpace):
    beta_x, beta_y, reaction, source, exact = adv_rea_test2()
    return (
        VectorDGField((beta_x, beta_y), space, name="beta_h"),
        DGField(reaction, space, name="reaction_h"),
        DGField(source, space, name="source_h"),
        exact,
    )


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
    beta_dot_normal = hdg_mats.advective_boundary_normal(beta_h, space)

    local_mats = np.ascontiguousarray(hdg_mats.boundary_mass_from_normal_flux(space, beta_dot_normal))
    scratch = np.empty_like(local_mats)
    hdg_mats.add_reaction_mass(local_mats, reaction_h, space, scratch=scratch)
    hdg_mats.add_advection_mats(local_mats, space, beta_h, scale=-1.0)
    local_solver = np.linalg.inv(local_mats)
    element_boundary_mats = hdg_mats.element_boundary_mats_from_normal_flux(space, beta_dot_normal)
    source_moments = hdg_assembly.source_moments(source_h, space)
    numpy_trace_system = hdg_assembly.assemble_trace_system(
        local_solver,
        element_boundary_mats,
        source_moments,
        exact,
        space,
    )

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
