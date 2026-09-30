from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from scripts.advection_diffusion_reaction.cases import scalar_case

from hdgfem import (
    AdvectionDiffusionReactionHDGOptions,
    AdvectionDiffusionReactionHDGSolver,
    DGSpace,
    rectangle_mesh,
    solve_advection_diffusion_reaction_hdg,
)
from hdgfem.assembly.advection_diffusion_reaction import (
    assemble_numpy,
    prepare_adr_data,
    recommended_diffusion_stabilization,
)
from hdgfem.backends.advection_diffusion_reaction_numba import (
    assemble_projected_adr_trace_system_eliminated_numba,
)

pytest.importorskip("numba")


def _problem(space):
    source = space.project_callable(lambda x, y: 1.0 + 0.2 * x - 0.1 * y)
    reaction = space.project_callable(lambda x, y: 0.4 + 0.1 * x)
    beta = (space * space).field(
        (
            space.project_callable(lambda x, y: 0.8 + 0.1 * y),
            space.project_callable(lambda x, y: -0.25 + 0.07 * x),
        )
    )
    boundary = lambda x, y: 0.2 + x - 0.3 * y
    return source, beta, reaction, boundary


def _csr(system):
    matrix = scipy.sparse.coo_array(
        (system.data, (system.rows, system.cols)),
        shape=(system.rhs.size, system.rhs.size),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def test_recommended_diffusion_stabilization_scales_with_order_and_face_altitude():
    mesh = rectangle_mesh(2, 1, xlim=(0.0, 3.0), ylim=(-1.0, 1.0))
    p1 = DGSpace(mesh, 1, basis_type="dub_orth")
    p3 = DGSpace(mesh, 3, basis_type="dub_orth")
    tau1 = recommended_diffusion_stabilization(p1, 2.0, penalty_constant=1.5)
    tau3 = recommended_diffusion_stabilization(p3, 2.0, penalty_constant=1.5)
    expected_h = 2.0 * mesh.aff_jacs[:, None] / mesh.jacs_el_fc
    np.testing.assert_allclose(tau1, 1.5 * 4.0 * 2.0 / expected_h)
    np.testing.assert_allclose(tau3 / tau1, 4.0)


@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
def test_numba_matches_numpy_with_asymmetric_element_face_stabilization(trace_basis):
    space = DGSpace(rectangle_mesh(3, 2), 2, basis_type="dub_orth")
    source, beta, reaction, boundary = _problem(space)
    element = np.arange(space.mesh.num_tri)[:, None]
    face = np.arange(3)[None, :]
    tau_adv = 0.35 + 0.017 * element + 0.09 * face
    tau_diff = 1.1 + 0.031 * element + 0.13 * face
    trace_space = space.trace_space(trace_basis)
    prepared = prepare_adr_data(
        source,
        reaction,
        beta,
        space,
        advection_stabilization=tau_adv,
        diffusion_stabilization=tau_diff,
        trace_space=trace_space,
    )
    numpy = assemble_numpy(prepared, boundary, space, trace_space=trace_space)
    numba = assemble_projected_adr_trace_system_eliminated_numba(
        prepared, boundary, space, trace_space=trace_space
    )
    a = _csr(numpy.trace_system)
    b = _csr(numba.trace_system)
    np.testing.assert_array_equal(a.indptr, b.indptr)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_allclose(a.data, b.data, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        numpy.trace_system.rhs, numba.trace_system.rhs, rtol=2e-13, atol=2e-13
    )

    # This explicitly checks the incidence-wise mass rule: two sides sharing
    # an edge retain their unequal gamma-weighted contributions. The Numba
    # kernel builds these blocks itself, so they agree to round-off.
    side_blocks = prepared.interior_gamma_mass
    emitted_mass = numba.trace_system.data[-side_blocks.size:].reshape(side_blocks.shape)
    np.testing.assert_allclose(emitted_mass, side_blocks, rtol=1e-13, atol=1e-14 * np.abs(side_blocks).max())
    side_edges = space.mesh.loc2glob_edge[
        space.mesh.interior_elements, space.mesh.interior_faces
    ]
    for edge in space.mesh.int_edges_inds:
        positions = np.flatnonzero(side_edges == edge)
        assert positions.size == 2
        assert not np.allclose(side_blocks[positions[0]], side_blocks[positions[1]])


@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
@pytest.mark.parametrize("diffusion", (0.3, np.array([[1.0, 0.3], [0.3, 0.5]])))
def test_numba_builds_face_tables_in_kernel_from_light_preparation(trace_basis, diffusion):
    from hdgfem.hdg import condensation as hdg
    from hdgfem.assembly.advection_diffusion_reaction import local_solvers_numpy
    from hdgfem.backends.advection_diffusion_reaction_numba import reconstruct_projected_adr_local_unknowns_numba

    space = DGSpace(rectangle_mesh(4, 3, xlim=(0.0, 1.3), ylim=(-0.4, 0.7)), 3, basis_type="dub_orth")
    source, beta, reaction, boundary = _problem(space)
    trace_space = space.trace_space(trace_basis)
    options = dict(diffusion=diffusion, trace_space=trace_space)
    dense = prepare_adr_data(source, reaction, beta, space, **options)
    light = prepare_adr_data(source, reaction, beta, space, dense_local_matrices=False, **options)
    assert light.element_boundary is None and light.trace_lift is None
    numpy = assemble_numpy(dense, boundary, space, diffusion=diffusion, trace_space=trace_space)
    columns = np.empty((space.mesh.num_tri, 3 * space.el_dof, 3 * trace_space.edg_dof + 1))
    numba = assemble_projected_adr_trace_system_eliminated_numba(
        light, boundary, space, trace_space=trace_space, diffusion=diffusion, local_columns=columns)
    a, b = _csr(numpy.trace_system), _csr(numba.trace_system)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_allclose(b.data, a.data, rtol=1e-13, atol=1e-13 * np.abs(a.data).max())
    np.testing.assert_allclose(numba.trace_system.rhs, numpy.trace_system.rhs, rtol=1e-13, atol=1e-13)

    trace = np.random.default_rng(5).normal(size=space.mesh.num_edg * trace_space.edg_dof)
    source_block = np.zeros((space.mesh.num_tri, 3 * space.el_dof))
    source_block[:, :space.el_dof] = dense.source_rhs
    expected = hdg.reconstruct_local_unknowns(
        trace, source_block, local_solvers_numpy(dense, space, diffusion=diffusion), dense.element_boundary,
        space, trace_space=trace_space)
    for stored in (None, columns):
        actual = reconstruct_projected_adr_local_unknowns_numba(
            trace, light, space, trace_space=trace_space, diffusion=diffusion, local_columns=stored)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


def test_coefficients_may_use_different_dg_spaces_on_same_mesh():
    mesh = rectangle_mesh(2, 2)
    solve_space = DGSpace(mesh, 2, basis_type="dub_orth")
    source_space = DGSpace(mesh, 1, basis_type="dub_orth")
    reaction_space = DGSpace(mesh, 3, basis_type="dub_orth")
    beta_space = DGSpace(mesh, 1, basis_type="dub_orth")
    source = source_space.project_callable(lambda x, y: 1.0 + x)
    reaction = reaction_space.project_callable(lambda x, y: 0.5 + 0.1 * x * y)
    beta = (beta_space * beta_space).field(
        (beta_space.project_callable(lambda x, y: 0.7 + y), beta_space.constant(-0.2))
    )
    result = solve_advection_diffusion_reaction_hdg(
        source,
        beta,
        reaction,
        0.0,
        solve_space,
        assembly_backend="numba",
        solver="direct",
        preconditioner=None,
        scale_system=False,
        hdg_postprocess="both",
        verbose=False,
    )
    assert result.field.space is solve_space
    assert result.postprocessed_field is not None
    assert result.postprocessed_flux is not None
    assert np.all(np.isfinite(result.local_unknowns))
    assert np.all(np.isfinite(result.postprocessed_field.coeffs))


@pytest.mark.parametrize(
    "flux_postprocess_space", ("full-p-plus-1", "rt-p")
)
@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
@pytest.mark.parametrize(
    ("backend", "reconstruction_backend"),
    (("numpy", "numba"), ("numba", "numpy")),
)
def test_affine_manufactured_solution_and_postprocessing(
        backend,
        reconstruction_backend,
        trace_basis,
        flux_postprocess_space,
):
    space = DGSpace(rectangle_mesh(2, 2), 1, basis_type="dub_orth")
    problem = scalar_case('affine')
    exact = problem.exact
    beta = (space * space).field(tuple(space.constant(value) for value in problem.beta))
    reaction = space.constant(problem.reaction)
    source = space.project_callable(problem.source)
    result = solve_advection_diffusion_reaction_hdg(
        source,
        beta,
        reaction,
        exact,
        space,
        assembly_backend=backend,
        reconstruction_backend=reconstruction_backend,
        postprocessing_backend="numba",
        trace_basis=trace_basis,
        flux_postprocess_space=flux_postprocess_space,
        diffusion=0.1,
        solver="direct",
        preconditioner=None,
        scale_system=False,
        diffusion_stabilization=1.0,
        hdg_postprocess="both",
        verbose=False,
    )
    points = space.mapped_quads()
    np.testing.assert_allclose(
        result.field.values_at_ref(space.quad_data.Krf_quads),
        exact(points[..., 0], points[..., 1]),
        rtol=2e-13,
        atol=2e-13,
    )
    assert result.reconstruction_backend == reconstruction_backend
    assert result.postprocessing_backend == "numba"
    assert result.postprocessed_field is not None
    assert result.postprocessed_flux is not None
    assert result.postprocessed_field.space.order == space.order + 1
    assert result.postprocessed_flux.components[0].space.order == space.order + 1
    post_points = result.postprocessed_field.space.mapped_quads()
    np.testing.assert_allclose(
        result.postprocessed_field.values_at_ref(
            result.postprocessed_field.space.quad_data.Krf_quads
        ),
        exact(post_points[..., 0], post_points[..., 1]),
        rtol=2e-11,
        atol=2e-11,
    )


def test_rt_total_flux_satisfies_face_and_interior_moments():
    """Verify the unisolvent RT_p degrees of freedom against HDG targets."""
    from hdgfem.solvers.diffusion_reaction import (
        _edge_lagrange_basis,
        _trace_basis_at,
    )

    space = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
    source, beta, reaction, boundary = _problem(space)
    tau_adv = 0.4
    tau_diff = 0.7
    result = solve_advection_diffusion_reaction_hdg(
        source,
        beta,
        reaction,
        boundary,
        space,
        assembly_backend="numba",
        reconstruction_backend="numba",
        postprocessing_backend="numba",
        diffusion=0.1,
        advection_stabilization=tau_adv,
        diffusion_stabilization=tau_diff,
        flux_postprocess_space="RT_projection",
        solver="direct",
        preconditioner=None,
        scale_system=False,
        hdg_postprocess="flux",
        verbose=False,
    )
    post_flux = result.postprocessed_flux
    assert post_flux is not None
    assert result.total_flux is not None
    post = post_flux.components[0].space
    qpost = post.quad_data
    nqf = qpost.weights_JGL.size
    face_points = qpost.pts_fc.reshape(-1, 2)
    base_face = space.basis_at(face_points).reshape(
        nqf, 3, space.el_dof
    ).transpose(1, 2, 0)
    blocks = result.local_unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)
    u_face = np.einsum("Ki,fiq->Kfq", blocks[:, 0], base_face)
    qx_face = np.einsum("Ki,fiq->Kfq", blocks[:, 1], base_face)
    qy_face = np.einsum("Ki,fiq->Kfq", blocks[:, 2], base_face)
    trace_space = space.trace_space("legacy-lagrange")
    trace_basis = _trace_basis_at(trace_space, qpost.quads_JGL)
    local_trace = trace_space.element_coefficients(result.trace).reshape(
        space.mesh.num_tri, 3, trace_space.edg_dof
    )
    hat_face = np.einsum("Kfa,aq->Kfq", local_trace, trace_basis)
    beta_face = []
    for component in beta.components:
        component_basis = component.space.basis_at(face_points).reshape(
            nqf, 3, component.space.el_dof
        ).transpose(1, 2, 0)
        beta_face.append(
            np.einsum("Ki,fiq->Kfq", component.coeffs, component_basis)
        )
    normals = space.mesh.normals
    beta_n = (
        beta_face[0] * normals[..., 0, None]
        + beta_face[1] * normals[..., 1, None]
    )
    numerical = (
        normals[..., 0, None] * qx_face
        + normals[..., 1, None] * qy_face
        + beta_n * hat_face
        + (tau_adv + tau_diff) * (u_face - hat_face)
    )
    post_qx_face = np.einsum(
        "Ki,fiq->Kfq", post_flux.components[0].coeffs, qpost.bas_of_bd_quads
    )
    post_qy_face = np.einsum(
        "Ki,fiq->Kfq", post_flux.components[1].coeffs, qpost.bas_of_bd_quads
    )
    post_normal = (
        normals[..., 0, None] * post_qx_face
        + normals[..., 1, None] * post_qy_face
    )
    face_test = _edge_lagrange_basis(space.order, qpost.quads_JGL)
    face_residual = np.einsum(
        "Kf,q,aq,Kfq->Kfa",
        space.mesh.jacs_el_fc,
        qpost.weights_JGL,
        face_test,
        post_normal - numerical,
    )
    np.testing.assert_allclose(face_residual, 0.0, rtol=0.0, atol=2e-11)

    low_space = DGSpace(
        space.mesh, space.order - 1, basis_type=space.reference.basis_type
    )
    low_basis = low_space.basis_at(qpost.Krf_quads)
    for post_component, base_component in zip(
        post_flux.components, result.total_flux.components
    ):
        difference = (
            post_component.values_at_ref(qpost.Krf_quads)
            - base_component.values_at_ref(qpost.Krf_quads)
        )
        interior_residual = np.einsum(
            "K,q,qi,Kq->Ki",
            space.mesh.aff_jacs,
            qpost.Krf_w,
            low_basis,
            difference,
        )
        np.testing.assert_allclose(
            interior_residual, 0.0, rtol=0.0, atol=2e-11
        )


@pytest.mark.parametrize(
    "flux_postprocess_space", ("full-p-plus-1", "rt-p")
)
def test_flux_postprocessing_has_optimal_rates_with_h_independent_tau(
        flux_postprocess_space,
):
    """Recover order p+1 total flux and p+2 primal rates for p=3."""
    problem = scalar_case('sine')
    kappa = problem.diffusion
    beta_x, beta_y = problem.beta
    reaction = problem.reaction
    exact, source = problem.exact, problem.source

    def exact_total_flux(x, y):
        qx, qy = problem.exact_flux(x, y)
        value = exact(x, y)
        return qx + beta_x * value, qy + beta_y * value

    errors = []
    for subdivisions in (4, 8, 16):
        space = DGSpace(
            rectangle_mesh(
                subdivisions,
                subdivisions,
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
            ),
            3,
            basis_type="dub_orth",
        )
        beta = (space * space).field(
            (space.constant(beta_x), space.constant(beta_y))
        )
        result = solve_advection_diffusion_reaction_hdg(
            source,
            beta,
            reaction,
            exact,
            space,
            diffusion=kappa,
            diffusion_stabilization=0.1,
            assembly_backend="numba",
            reconstruction_backend="numba",
            postprocessing_backend="numba",
            flux_postprocess_space=flux_postprocess_space,
            solver="direct",
            preconditioner=None,
            scale_system=False,
            hdg_postprocess="both",
            verbose=False,
        )
        errors.append(
            (
                result.postprocessed_field.l2_error(exact),
                result.postprocessed_flux.l2_error(exact_total_flux),
            )
        )

    errors = np.asarray(errors)
    primal_rates = np.log2(errors[:-1, 0] / errors[1:, 0])
    flux_rates = np.log2(errors[:-1, 1] / errors[1:, 1])
    assert np.all(primal_rates > 4.6)
    assert np.all(flux_rates > 3.7)


def test_reusable_solver_and_public_options_contract():
    space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
    source, beta, reaction, boundary = _problem(space)
    options = AdvectionDiffusionReactionHDGOptions(verbose=False)
    with pytest.raises(TypeError, match="unknown advection-diffusion-reaction solver option"):
        options.with_overrides(not_an_option=True)
    solver = AdvectionDiffusionReactionHDGSolver(space, options=options)
    with pytest.raises(RuntimeError, match="no complete advection-diffusion-reaction problem"):
        solver.solve()
    result = solver.set_problem(source, beta, reaction, boundary).solve(
        solver="direct", preconditioner=None, scale_system=False, hdg_postprocess="none"
    )
    assert result.field.space is space


def test_stage_backend_preflight_rejects_unavailable_paths():
    """Reject stage combinations that do not have an implementation path."""
    space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
    source, beta, reaction, boundary = _problem(space)
    common = dict(solver="direct", preconditioner=None, scale_system=False, verbose=False)
    with pytest.raises(NotImplementedError, match="postprocessing_backend='numba'"):
        solve_advection_diffusion_reaction_hdg(
            source, beta, reaction, boundary, space,
            assembly_backend="numpy", postprocessing_backend="numpy", **common,
        )
    with pytest.raises(NotImplementedError, match="requires assembly_backend='raw-cuda'"):
        solve_advection_diffusion_reaction_hdg(
            source, beta, reaction, boundary, space,
            assembly_backend="numpy", reconstruction_backend="raw-cuda", **common,
        )
    with pytest.raises(ValueError, match="requires CuPy ADR postprocessing"):
        solve_advection_diffusion_reaction_hdg(
            source, beta, reaction, boundary, space,
            assembly_backend="raw-cuda", postprocessing_backend="numba",
            materialize_host_solution=False, **common,
        )


def _cupy_runtime_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy runtime unavailable")
def test_rt_cupy_postprocess_matches_host_numba():
    """Match the batched CuPy RT moment solve against the host kernel."""
    space = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
    source, beta, reaction, boundary = _problem(space)
    common = dict(
        assembly_backend="numba",
        reconstruction_backend="numba",
        diffusion=0.1,
        diffusion_stabilization=0.4,
        flux_postprocess_space="rt-p",
        hdg_postprocess="both",
        solver="direct",
        preconditioner=None,
        scale_system=False,
        verbose=False,
    )
    host = solve_advection_diffusion_reaction_hdg(
        source,
        beta,
        reaction,
        boundary,
        space,
        postprocessing_backend="numba",
        **common,
    )
    device = solve_advection_diffusion_reaction_hdg(
        source,
        beta,
        reaction,
        boundary,
        space,
        postprocessing_backend="cupy",
        **common,
    )
    assert device.postprocessed_field is not None
    assert device.postprocessed_flux is not None
    np.testing.assert_allclose(
        device.postprocessed_field.coeffs,
        host.postprocessed_field.coeffs,
        rtol=3e-11,
        atol=3e-12,
    )
    for device_component, host_component in zip(
        device.postprocessed_flux.components,
        host.postprocessed_flux.components,
    ):
        np.testing.assert_allclose(
            device_component.coeffs,
            host_component.coeffs,
            rtol=3e-11,
            atol=3e-12,
        )


def _raw_cuda_runtime_available():
    try:
        import cupy as cp
        import pyamgx  # noqa: F401

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.parametrize(
    "flux_postprocess_space", ("full-p-plus-1", "rt-p")
)
@pytest.mark.parametrize("postprocessing_backend", ("auto", "numba"))
@pytest.mark.skipif(not _raw_cuda_runtime_available(), reason="Raw CUDA/PyAMGX runtime unavailable")
def test_raw_cuda_matches_numpy_with_asymmetric_side_stabilization(
        flux_postprocess_space, postprocessing_backend,
):
    from hdgfem.linalg.amgx.config import load_amgx_config

    space = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
    source, beta, reaction, boundary = _problem(space)
    element = np.arange(space.mesh.num_tri)[:, None]
    face = np.arange(3)[None, :]
    tau_adv = 0.4 + 0.021 * element + 0.08 * face
    tau_diff = 1.0 + 0.037 * element + 0.11 * face
    common = dict(
        diffusion=0.1,
        advection_stabilization=tau_adv,
        diffusion_stabilization=tau_diff,
        scale_system=False,
        hdg_postprocess="both",
        flux_postprocess_space=flux_postprocess_space,
        verbose=False,
    )
    reference = solve_advection_diffusion_reaction_hdg(
        source, beta, reaction, boundary, space,
        assembly_backend="numpy", solver="direct", preconditioner=None, **common,
    )
    config, _ = load_amgx_config("configs/amgx/adv_rea_gpu4_hdg_fgmres_dilu_abs.json")
    device = solve_advection_diffusion_reaction_hdg(
        source, beta, reaction, boundary, space,
        assembly_backend="raw-cuda", solver="amgx", amgx_config=config,
        solver_rtol=1e-11, postprocessing_backend=postprocessing_backend, **common,
    )
    np.testing.assert_allclose(device.trace, reference.trace, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(
        device.local_unknowns, reference.local_unknowns, rtol=2e-11, atol=2e-12
    )
    assert device.postprocessed_field is not None
    assert device.postprocessed_flux is not None
    np.testing.assert_allclose(
        device.postprocessed_field.coeffs,
        reference.postprocessed_field.coeffs,
        rtol=2e-10,
        atol=2e-11,
    )
    for device_component, reference_component in zip(
        device.postprocessed_flux.components,
        reference.postprocessed_flux.components,
    ):
        np.testing.assert_allclose(
            device_component.coeffs,
            reference_component.coeffs,
            rtol=2e-10,
            atol=2e-11,
        )
