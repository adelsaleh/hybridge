from __future__ import annotations

import numpy as np
import pytest

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGField, DGSpace
from hdgfem.solvers.diff_rea import (
    DiffusionReactionHDGSolver as DiffReaSolver,
    solve_diffusion_reaction_hdg,
)
from scripts.diffusion_reaction.diff_rea_cases import (
    quadratic_poisson_case,
    tensor_sine_diffusion_reaction_case,
    tensor_sine_exact_gradients,
    trigonometric_poisson_case,
)


def _space(order: int = 2) -> DGSpace:
    return DGSpace(rectangle_mesh(1, 1), order, basis_type="dub_orth")


def _callable_problem():
    _, reaction, source, exact = quadratic_poisson_case()
    return source, reaction, exact


def _projected_problem(space: DGSpace):
    _, reaction, source, exact = quadratic_poisson_case()
    return (
        DGField(source, space, name="source_h"),
        DGField(reaction, space, name="reaction_h"),
        exact,
    )


def _assert_solver_cache_matches_result(solver: DiffReaSolver, result) -> None:
    assert solver.result is result
    assert solver.field is result.field
    assert solver.flux is result.flux
    assert solver.postprocessed_field is result.postprocessed_field
    assert solver.postprocessed_flux is result.postprocessed_flux
    assert solver.trace is result.trace
    assert solver.timings is result.timings
    assert solver.local_unknowns is result.local_unknowns
    assert solver.rows is result.matrix_rows
    assert solver.cols is result.matrix_cols
    assert solver.data is result.matrix_data
    assert solver.rhs is result.rhs
    assert solver.solve_rows is result.solve_matrix_rows
    assert solver.solve_cols is result.solve_matrix_cols
    assert solver.solve_data is result.solve_matrix_data
    assert solver.solve_rhs is result.solve_rhs
    assert solver.boundary_trace is result.boundary_trace
    assert solver.reduction is result.reduction
    assert solver.local_solver is result.local_solver
    assert solver.element_boundary_mats is result.element_boundary_mats
    assert solver.global_solve_result is result.global_solve_result


def _legendre_gauss_lobatto(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    if num_points == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    poly = np.polynomial.legendre.Legendre.basis(num_points - 1)
    interior = np.sort(poly.deriv().roots())
    points = np.concatenate(([-1.0], interior, [1.0]))
    values = poly(points)
    weights = 2.0 / ((num_points - 1) * num_points * values * values)
    return np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(weights, dtype=np.float64)


def _edge_lagrange_basis(order: int, points: np.ndarray) -> np.ndarray:
    nodes, _ = _legendre_gauss_lobatto(order + 1)
    values = np.ones((order + 1, points.size), dtype=np.float64)
    for i in range(order + 1):
        for j in range(order + 1):
            if i != j:
                values[i] *= (points - nodes[j]) / (nodes[i] - nodes[j])
    return values


def _face_base_to_post(space: DGSpace, post_space: DGSpace) -> np.ndarray:
    q_post = post_space.quad_data
    face_points = q_post.pts_fc.reshape(-1, 2)
    base_face = space.basis_at(face_points).reshape(q_post.weights_JGL.size, 3, space.el_dof)
    base_face = base_face.transpose(1, 2, 0)
    return np.einsum(
        "q,fiq,aq->fia",
        q_post.weights_JGL,
        base_face,
        q_post.bas1d_of_ref_edg_qds,
        optimize=True,
    )


def _trace_base_to_post(space: DGSpace, post_space: DGSpace) -> np.ndarray:
    q_post = post_space.quad_data
    base_trace = _edge_lagrange_basis(space.order, q_post.quads_JGL)
    return np.einsum(
        "q,iq,aq->ia",
        q_post.weights_JGL,
        base_trace,
        q_post.bas1d_of_ref_edg_qds,
        optimize=True,
    )


def _interior_moment_tables(space: DGSpace, post_space: DGSpace) -> tuple[np.ndarray, np.ndarray]:
    if space.order == 0:
        return np.empty((0, space.el_dof)), np.empty((0, post_space.el_dof))
    low_space = DGSpace(space.mesh, space.order - 1, basis_type=space.reference.basis_type)
    q_post = post_space.quad_data
    low_basis = low_space.basis_at(q_post.Krf_quads)
    base_basis = space.basis_at(q_post.Krf_quads)
    return (
        np.einsum("q,qi,qj->ij", q_post.Krf_w, low_basis, base_basis, optimize=True),
        np.einsum("q,qi,qj->ij", q_post.Krf_w, low_basis, q_post.phi, optimize=True),
    )


def _assert_hdiv_flux_constraints(result, space: DGSpace, tau_value: float) -> None:
    flux_star = result.postprocessed_flux
    assert flux_star is not None
    post_space = flux_star.components[0].space
    assert post_space.order == space.order + 1

    unknowns = result.local_unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)
    qx_star, qy_star = flux_star.as_component_first()
    face_base = _face_base_to_post(space, post_space)
    trace_base = _trace_base_to_post(space, post_space)
    low_to_base, low_to_post = _interior_moment_tables(space, post_space)
    face_post = post_space.quad_data.face_element_test_trace_trial

    for element in range(space.mesh.num_tri):
        for face in range(3):
            edge = space.mesh.loc2glob_edge[element, face]
            orientation = space.mesh.orientations[element, face]
            trace_ids = np.arange(space.quad_data.edg_dof)
            if not orientation:
                trace_ids = trace_ids[::-1]
            trace_coeffs = result.trace[edge * space.quad_data.edg_dof + trace_ids]

            nx, ny = space.mesh.normals[element, face]
            scale = space.mesh.jacs_el_fc[element, face]
            lhs = scale * (
                nx * (qx_star[element] @ face_post[face])
                + ny * (qy_star[element] @ face_post[face])
            )
            rhs = scale * (
                nx * (unknowns[element, 1] @ face_base[face])
                + ny * (unknowns[element, 2] @ face_base[face])
                + tau_value * (
                    unknowns[element, 0] @ face_base[face]
                    - trace_coeffs @ trace_base
                )
            )
            np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)

        jac = space.mesh.aff_jacs[element]
        np.testing.assert_allclose(
            jac * (low_to_post @ qx_star[element]),
            jac * (low_to_base @ unknowns[element, 1]),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            jac * (low_to_post @ qy_star[element]),
            jac * (low_to_base @ unknowns[element, 2]),
            rtol=1e-10,
            atol=1e-10,
        )


def _vector_l2_error(vector_field, exact_flux) -> float:
    space = vector_field.components[0].space
    points = space.mapped_quads()
    exact_values = np.stack(exact_flux(points[:, :, 0], points[:, :, 1]), axis=0)
    diff = vector_field.values() - exact_values
    return float(np.sqrt(np.einsum("K,dKq,q->", space.mesh.aff_jacs, diff * diff, space.quad_data.Krf_w)))


@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
def test_diff_rea_solver_matches_function_for_callable_problem(boundary_mode: str) -> None:
    space = _space()
    source, reaction, exact = _callable_problem()
    kwargs = {
        "stabilization": 1.0,
        "solver": "direct",
        "preconditioner": None,
        "boundary_mode": boundary_mode,
        "verbose": False,
    }

    function_result = solve_diffusion_reaction_hdg(source, reaction, exact, space, **kwargs)
    solver = DiffReaSolver(space, source=source, reaction=reaction, boundary_condition=exact, **kwargs)
    class_result = solver.solve()

    np.testing.assert_allclose(class_result.trace, function_result.trace, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(class_result.field.coeffs, function_result.field.coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        class_result.flux.as_component_first(),
        function_result.flux.as_component_first(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(class_result.local_unknowns, function_result.local_unknowns, rtol=1e-12, atol=1e-12)
    assert class_result.boundary_mode == boundary_mode
    _assert_solver_cache_matches_result(solver, class_result)


def test_diff_rea_solver_discrete_problem_and_source_update_match_function() -> None:
    space = _space(order=1)
    source_h, reaction_h, exact = _projected_problem(space)
    kwargs = {
        "stabilization": 1.0,
        "solver": "direct",
        "preconditioner": None,
        "boundary_mode": "eliminate",
        "verbose": False,
    }

    function_result = solve_diffusion_reaction_hdg(source_h, reaction_h, exact, space, **kwargs)
    solver = DiffReaSolver(space, **kwargs)
    solver.set_discrete_problem(source_h, reaction_h, exact)
    class_result = solver.solve()

    np.testing.assert_allclose(class_result.trace, function_result.trace, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(class_result.field.coeffs, function_result.field.coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(class_result.local_unknowns, function_result.local_unknowns, rtol=1e-12, atol=1e-12)
    assert class_result.reduction is not None
    assert class_result.local_solver is not None
    assert class_result.element_boundary_mats is not None
    _assert_solver_cache_matches_result(solver, class_result)

    scaled_source = space.field(1.1 * source_h.coeffs, name="scaled_source")
    solver.set_source(scaled_source)
    assert solver.result is None
    assert solver.trace is None
    assert solver.rows is None

    updated_function_result = solve_diffusion_reaction_hdg(
        scaled_source,
        reaction_h,
        exact,
        space,
        **kwargs,
    )
    updated_class_result = solver.solve()

    np.testing.assert_allclose(updated_class_result.trace, updated_function_result.trace, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        updated_class_result.field.coeffs,
        updated_function_result.field.coeffs,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        updated_class_result.flux.as_component_first(),
        updated_function_result.flux.as_component_first(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert not np.allclose(updated_class_result.trace, class_result.trace)
    _assert_solver_cache_matches_result(solver, updated_class_result)


def test_diff_rea_solver_rejects_incomplete_problem_update() -> None:
    solver = DiffReaSolver(_space(order=1), solver="direct", preconditioner=None, verbose=False)
    source, reaction, exact = _callable_problem()

    with pytest.raises(ValueError, match="must be provided together"):
        solver.solve(source=source, reaction=reaction)

    solver.set_problem(source, reaction, exact)
    result = solver.solve()
    assert result.trace is not None


def test_hdg_postprocess_primal_and_flux_outputs_and_flux_moments() -> None:
    pytest.importorskip("numba")
    space = _space(order=1)
    source_h, reaction_h, exact = _projected_problem(space)
    tau = 1.3
    solver = DiffReaSolver(
        space,
        source=source_h,
        reaction=reaction_h,
        boundary_condition=exact,
        stabilization=tau,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numba",
        hdg_postprocess="both",
        verbose=False,
    )

    result = solver.solve()

    assert result.postprocessed_field is not None
    assert result.postprocessed_flux is not None
    assert result.postprocessed_field.space.order == space.order + 1
    assert result.postprocessed_field.coeffs.shape == (
        space.mesh.num_tri,
        (space.order + 2) * (space.order + 3) // 2,
    )
    assert result.postprocessed_flux.as_component_first().shape == (2,) + result.postprocessed_field.coeffs.shape
    assert result.timings.postprocessing > 0.0
    _assert_hdiv_flux_constraints(result, space, tau)
    _assert_solver_cache_matches_result(solver, result)


def test_hdg_postprocess_flux_uses_primal_reference_for_identity_diffusion() -> None:
    pytest.importorskip("numba")
    mesh = rectangle_mesh(4, 4, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, 3, basis_type="dub_orth")
    problem = trigonometric_poisson_case()
    diffusion, reaction, source, exact = problem

    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.0,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numba",
        hdg_postprocess="both",
        verbose=False,
    )

    assert result.postprocessed_flux is not None
    raw_error = _vector_l2_error(result.flux, problem.exact_flux)
    post_error = _vector_l2_error(result.postprocessed_flux, problem.exact_flux)
    assert post_error < 0.55 * raw_error
    _assert_hdiv_flux_constraints(result, space, 1.0)


def test_identity_diffusion_argument_preserves_default_solution() -> None:
    space = _space(order=2)
    diffusion, reaction, source, exact = quadratic_poisson_case()
    kwargs = {
        "stabilization": 1.0,
        "solver": "direct",
        "preconditioner": None,
        "boundary_mode": "eliminate",
        "verbose": False,
    }

    default = solve_diffusion_reaction_hdg(source, reaction, exact, space, **kwargs)
    explicit = solve_diffusion_reaction_hdg(source, reaction, exact, space, diffusion=diffusion, **kwargs)

    np.testing.assert_allclose(explicit.trace, default.trace, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(explicit.field.coeffs, default.field.coeffs, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        explicit.flux.as_component_first(),
        default.flux.as_component_first(),
        rtol=0.0,
        atol=0.0,
    )


def test_tensor_diffusion_manufactured_solution_numpy_and_projected_numba_are_accurate() -> None:
    mesh = rectangle_mesh(3, 3)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = tensor_sine_diffusion_reaction_case()
    kwargs = {
        "diffusion": diffusion,
        "stabilization": 4.0,
        "solver": "direct",
        "preconditioner": None,
        "boundary_mode": "eliminate",
        "verbose": False,
    }

    numpy_result = solve_diffusion_reaction_hdg(source, reaction, exact, space, assembly_backend="numpy", **kwargs)
    numba_result = solve_diffusion_reaction_hdg(source, reaction, exact, space, assembly_backend="numba", **kwargs)

    assert numpy_result.field.l2_error(exact) < 2.0e-2
    assert numba_result.field.l2_error(exact) < 2.0e-2
    assert numba_result.local_solver is None
    assert numba_result.element_boundary_mats is None
    np.testing.assert_allclose(numba_result.trace, numpy_result.trace, rtol=0.0, atol=1.0e-3)
    np.testing.assert_allclose(numba_result.field.coeffs, numpy_result.field.coeffs, rtol=0.0, atol=1.0e-3)


def test_tensor_diffusion_flux_uses_conservative_sign() -> None:
    mesh = rectangle_mesh(4, 4)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = tensor_sine_diffusion_reaction_case()
    gradx, grady = tensor_sine_exact_gradients()
    k11, k12, k22 = diffusion

    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=4.0,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        verbose=False,
    )

    points = space.mapped_quads()
    x = points[:, :, 0]
    y = points[:, :, 1]
    exact_qx = -(k11(x, y) * gradx(x, y) + k12(x, y) * grady(x, y))
    exact_qy = -(k12(x, y) * gradx(x, y) + k22(x, y) * grady(x, y))
    qx = result.flux.components[0].values()
    qy = result.flux.components[1].values()

    weights = space.quad_data.Krf_w
    jac = space.mesh.aff_jacs
    flux_error = np.sqrt(
        np.einsum("K,Kq,q->", jac, (qx - exact_qx) ** 2 + (qy - exact_qy) ** 2, weights, optimize=True)
    )
    flux_norm = np.sqrt(np.einsum("K,Kq,q->", jac, exact_qx**2 + exact_qy**2, weights, optimize=True))
    assert flux_error / flux_norm < 4.0e-2
