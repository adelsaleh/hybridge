from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import hdgfem
import hdgfem.solvers as solver_api
from hdgfem import DGSpace, VectorDGField, rectangle_mesh
from hdgfem.runtime.errors import UnsupportedBackendConfigurationError


_PRIMARY_SOLVER_EXPORTS = {
    "AdvectionDiffusionReactionHDGOptions",
    "AdvectionDiffusionReactionHDGSolver",
    "AdvectionDiffusionReactionResult",
    "AdvectionDiffusionReactionTimings",
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "DiffusionReactionAssemblyResult",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "GlobalLengthDiffusion",
    "ScaledUpwind",
    "automatic_domain_length",
    "compute_domain_length",
    "geometric_diffusion_tau",
    "mesh_domain_measures",
    "solve_advection_reaction_hdg",
    "solve_advection_diffusion_reaction_hdg",
    "solve_diffusion_reaction_hdg",
}


def _space() -> DGSpace:
    return DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")


def test_supported_solver_exports_are_identical_at_both_package_levels() -> None:
    assert _PRIMARY_SOLVER_EXPORTS <= set(hdgfem.__all__)
    assert set(solver_api.__all__) == _PRIMARY_SOLVER_EXPORTS | {
        "advection_reaction",
        "advection_diffusion_reaction",
        "adv_rea",
        "diffusion_reaction",
        "diff_rea",
    }

    for name in _PRIMARY_SOLVER_EXPORTS:
        assert getattr(hdgfem, name) is getattr(solver_api, name)


def test_solver_module_facades_and_legacy_aliases_remain_importable() -> None:
    assert (
        solver_api.advection_reaction.solve_advection_reaction_hdg
        is solver_api.solve_advection_reaction_hdg
    )
    assert (
        solver_api.diffusion_reaction.solve_diffusion_reaction_hdg
        is solver_api.solve_diffusion_reaction_hdg
    )
    assert solver_api.adv_rea.adv_rea_hdg_solv is solver_api.solve_advection_reaction_hdg
    assert solver_api.diff_rea.diff_rea_hdg_solve is solver_api.solve_diffusion_reaction_hdg
    assert "adv_rea_hdg_solv" not in solver_api.__all__
    assert "diff_rea_hdg_solve" not in solver_api.__all__


@pytest.mark.parametrize(
    "options_type, unknown_message",
    (
        (hdgfem.AdvectionReactionHDGOptions, "unknown advection-reaction solver option"),
        (
            hdgfem.AdvectionDiffusionReactionHDGOptions,
            "unknown advection-diffusion-reaction solver option",
        ),
        (hdgfem.DiffusionReactionHDGOptions, "unknown diffusion-reaction solver option"),
    ),
)
def test_options_are_immutable_and_reject_unknown_overrides(options_type, unknown_message: str) -> None:
    options = options_type(verbose=False)
    updated = options.with_overrides(verbose=True)

    assert options.verbose is False
    assert updated.verbose is True
    with pytest.raises(FrozenInstanceError):
        options.verbose = True
    with pytest.raises(TypeError, match=unknown_message):
        options.with_overrides(not_an_option=True)


@pytest.mark.parametrize(
    "public_type",
    (
        hdgfem.AdvectionReactionTimings,
        hdgfem.AdvectionReactionResult,
        hdgfem.AdvectionDiffusionReactionTimings,
        hdgfem.AdvectionDiffusionReactionResult,
        hdgfem.DiffusionReactionTimings,
        hdgfem.DiffusionReactionResult,
        hdgfem.DiffusionReactionAssemblyResult,
    ),
)
def test_public_result_and_timing_dataclasses_are_frozen(public_type) -> None:
    assert public_type.__dataclass_params__.frozen is True


def test_solver_failure_types_are_stable_before_backend_dispatch() -> None:
    adv_solver = hdgfem.AdvectionReactionHDGSolver(_space(), verbose=False)
    diff_solver = hdgfem.DiffusionReactionHDGSolver(_space(), verbose=False)

    with pytest.raises(RuntimeError, match="no complete advection-reaction problem is set"):
        adv_solver.solve()
    with pytest.raises(RuntimeError, match="no complete diffusion-reaction problem is set"):
        diff_solver.solve()
    with pytest.raises(TypeError, match="unknown advection-reaction solver option"):
        adv_solver.solve(not_an_option=True)
    with pytest.raises(TypeError, match="unknown diffusion-reaction solver option"):
        diff_solver.solve(not_an_option=True)


def test_constructor_rejects_partial_problem_bundles() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        hdgfem.AdvectionReactionHDGSolver(_space(), source=object(), verbose=False)
    with pytest.raises(ValueError, match="must be provided together"):
        hdgfem.DiffusionReactionHDGSolver(_space(), source=object(), verbose=False)


def test_with_options_is_persistent_and_clears_cached_result() -> None:
    adv_solver = hdgfem.AdvectionReactionHDGSolver(_space(), verbose=False)
    diff_solver = hdgfem.DiffusionReactionHDGSolver(_space(), verbose=False)
    sentinel = object()
    adv_solver.result = sentinel
    diff_solver.result = sentinel

    assert adv_solver.with_options(solver_rtol=1e-9) is adv_solver
    assert diff_solver.with_options(solver_rtol=1e-9) is diff_solver
    assert adv_solver.options.solver_rtol == 1e-9
    assert diff_solver.options.solver_rtol == 1e-9
    assert adv_solver.result is None
    assert diff_solver.result is None


@pytest.mark.parametrize(
    "solver_type",
    (
        hdgfem.AdvectionReactionHDGSolver,
        hdgfem.DiffusionReactionHDGSolver,
    ),
)
def test_solve_overrides_persist_but_explicit_initial_guess_does_not(solver_type) -> None:
    solver = solver_type(_space(), verbose=False)

    with pytest.raises(RuntimeError, match="no complete .* problem is set"):
        solver.solve(solver_rtol=2e-9)
    assert solver.options.solver_rtol == 2e-9

    stored_options = solver.options
    with pytest.raises(RuntimeError, match="no complete .* problem is set"):
        solver.solve(initial_guess=object())
    assert solver.options is stored_options
    assert solver.options.initial_guess is None


def _advection_problem(space: DGSpace):
    source = space.project_callable(lambda x, y: 1.0 + 0.2 * x - 0.1 * y, name="source_h")
    reaction = space.constant(2.0, name="reaction_h")
    beta = VectorDGField(
        (
            space.project_callable(lambda x, y: 0.7 + 0.1 * y, name="beta_x_h"),
            space.project_callable(lambda x, y: -0.2 + 0.05 * x, name="beta_y_h"),
        ),
        name="beta_h",
    )
    boundary = lambda x, y: x - 0.25 * y
    return source, beta, reaction, boundary


def _diffusion_problem(space: DGSpace):
    source = space.project_callable(lambda x, y: 1.0 + x - 0.5 * y, name="source_h")
    reaction = space.constant(0.5, name="reaction_h")
    boundary = lambda x, y: 0.25 + x + y
    return source, reaction, boundary


@pytest.mark.parametrize("assembly_backend", ("numpy", "numba"))
@pytest.mark.parametrize("equation", ("advection-reaction", "diffusion-reaction"))
def test_functional_and_reusable_solvers_accept_real_constant_boundary_data(equation: str, assembly_backend: str) -> None:
    space = _space()
    boundary_value = np.float64(1.25)
    common = dict(
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend=assembly_backend,
        verbose=False,
    )

    if equation == "advection-reaction":
        source, beta, reaction, _ = _advection_problem(space)
        functional = hdgfem.solve_advection_reaction_hdg(
            source,
            beta,
            reaction,
            boundary_value,
            space,
            **common,
        )
        solver = hdgfem.AdvectionReactionHDGSolver(space, **common)
        solver.set_problem(source, beta, reaction, boundary_value)
    else:
        source, reaction, _ = _diffusion_problem(space)
        functional = hdgfem.solve_diffusion_reaction_hdg(
            source,
            reaction,
            boundary_value,
            space,
            **common,
        )
        solver = hdgfem.DiffusionReactionHDGSolver(space, **common)
        solver.set_problem(source, reaction, boundary_value)

    reusable = solver.solve()
    assert callable(solver.boundary_condition)
    np.testing.assert_allclose(
        functional.boundary_trace[space.mesh.bnd_edges_inds],
        boundary_value,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(reusable.trace, functional.trace, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("equation", ("advection-reaction", "diffusion-reaction"))
@pytest.mark.parametrize("invalid_kind", ("dg-field", "object"))
def test_functional_and_reusable_solvers_reject_non_callable_non_constant_boundary_data(
    equation: str,
    invalid_kind: str,
) -> None:
    space = _space()
    invalid = space.constant(1.0, name="boundary_h") if invalid_kind == "dg-field" else object()
    message = "boundary_condition must be a callable or real scalar constant"
    common = dict(
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numpy",
        verbose=False,
    )

    if equation == "advection-reaction":
        source, beta, reaction, _ = _advection_problem(space)
        with pytest.raises(TypeError, match=message):
            hdgfem.solve_advection_reaction_hdg(
                source,
                beta,
                reaction,
                invalid,
                space,
                **common,
            )
        with pytest.raises(TypeError, match=message):
            hdgfem.AdvectionReactionHDGSolver(
                space,
                source=source,
                beta=beta,
                reaction=reaction,
                boundary_condition=invalid,
                **common,
            )
    else:
        source, reaction, _ = _diffusion_problem(space)
        with pytest.raises(TypeError, match=message):
            hdgfem.solve_diffusion_reaction_hdg(
                source,
                reaction,
                invalid,
                space,
                **common,
            )
        with pytest.raises(TypeError, match=message):
            hdgfem.DiffusionReactionHDGSolver(
                space,
                source=source,
                reaction=reaction,
                boundary_condition=invalid,
                **common,
            )


@pytest.mark.parametrize("assembly_backend", ("numpy", "numba"))
@pytest.mark.parametrize("equation", ("advection-reaction", "diffusion-reaction"))
def test_host_reusable_solver_accepts_per_call_initial_guess_and_reports_true_residual(
    equation: str,
    assembly_backend: str,
) -> None:
    space = _space()
    common = {
        "assembly_backend": assembly_backend,
        "solver": "direct",
        "preconditioner": None,
        "boundary_mode": "eliminate",
        "trace_basis": "legacy-lagrange",
        "verbose": False,
    }
    if equation == "advection-reaction":
        solver = hdgfem.AdvectionReactionHDGSolver(space, **common)
        solver.set_discrete_problem(*_advection_problem(space))
    else:
        solver = hdgfem.DiffusionReactionHDGSolver(
            space,
            hdg_postprocess="none",
            **common,
        )
        solver.set_discrete_problem(*_diffusion_problem(space))

    direct = solver.solve()
    warm = solver.solve(
        solver="GMRES",
        solver_rtol=1.0e-10,
        maxiter=100,
        initial_guess=direct.trace,
    )

    solve_result = warm.global_solve_result
    assert solver.options.solver == "GMRES"
    assert solver.options.initial_guess is None
    assert solve_result is not None
    assert solve_result.initial_residual_norm is not None
    assert solve_result.converged
    assert solve_result.physical_residual_target_met
    assert solve_result.physical_relative_residual_norm is not None
    assert solve_result.physical_relative_residual_norm <= 1.0e-10
    np.testing.assert_allclose(warm.trace, direct.trace, rtol=1.0e-9, atol=1.0e-10)


@pytest.mark.parametrize(
    "setter_name,new_value",
    (
        ("set_source", object()),
        ("set_beta", object()),
        ("set_reaction", object()),
        ("set_boundary_condition", lambda x, y: x + y),
    ),
)
def test_advection_problem_updates_clear_exposed_solve_artifacts(setter_name: str, new_value) -> None:
    space = _space()
    solver = hdgfem.AdvectionReactionHDGSolver(space, boundary_mode="eliminate", verbose=False)
    solver.set_discrete_problem(*_advection_problem(space))
    sentinel = object()
    artifact_names = (
        "result",
        "field",
        "trace",
        "rows",
        "cols",
        "data",
        "rhs",
        "solve_rows",
        "solve_cols",
        "solve_data",
        "solve_rhs",
        "boundary_trace",
        "reduction",
        "local_solver",
        "element_boundary_mats",
        "ordering_result",
        "matrix_pattern_plots",
        "global_solve_result",
        "preconditioner",
        "timings",
    )
    for name in artifact_names:
        setattr(solver, name, sentinel)

    getattr(solver, setter_name)(new_value)

    for name in artifact_names:
        assert getattr(solver, name) is None


@pytest.mark.parametrize("update_kind", ("source", "boundary"))
def test_diffusion_rhs_updates_preserve_cached_operator_but_clear_rhs_and_solution(update_kind: str) -> None:
    space = _space()
    solver = hdgfem.DiffusionReactionHDGSolver(
        space,
        assembly_backend="numba",
        boundary_mode="eliminate",
        cache_device_matrix=True,
        verbose=False,
    )
    solver.set_discrete_problem(*_diffusion_problem(space))
    rows = object()
    data = object()
    local_solver = object()
    host_matrix = object()
    solver.rows = rows
    solver.cols = rows
    solver.data = data
    solver.solve_rows = rows
    solver.solve_cols = rows
    solver.solve_data = data
    solver.local_solver = local_solver
    solver._host_solve_matrix = host_matrix
    solver.rhs = object()
    solver.solve_rhs = object()
    solver.boundary_trace = object()
    solver.result = object()
    solver.trace = object()
    solver.global_solve_result = object()
    solver._host_cached_rhs_valid = True

    if update_kind == "source":
        solver.set_source(space.constant(3.0, name="next_source_h"))
    else:
        solver.set_boundary_condition(lambda x, y: x - y)

    assert solver.rows is rows
    assert solver.cols is rows
    assert solver.data is data
    assert solver.solve_rows is rows
    assert solver.solve_cols is rows
    assert solver.solve_data is data
    assert solver.local_solver is local_solver
    assert solver._host_solve_matrix is host_matrix
    assert solver.rhs is None
    assert solver.solve_rhs is None
    assert solver.boundary_trace is None
    assert solver.result is None
    assert solver.trace is None
    assert solver.global_solve_result is None
    assert solver._host_cached_rhs_valid is False


def test_diffusion_reaction_update_invalidates_cached_operator() -> None:
    space = _space()
    solver = hdgfem.DiffusionReactionHDGSolver(
        space,
        assembly_backend="numba",
        boundary_mode="eliminate",
        cache_device_matrix=True,
        verbose=False,
    )
    solver.set_discrete_problem(*_diffusion_problem(space))
    solver.data = object()
    solver.local_solver = object()
    solver._host_solve_matrix = object()

    solver.set_reaction(space.constant(1.5, name="next_reaction_h"))

    assert solver.data is None
    assert solver.local_solver is None
    assert solver._host_solve_matrix is None


def test_diffusion_scipy_cache_reuses_host_and_scaled_csr() -> None:
    space = _space()
    solver = hdgfem.DiffusionReactionHDGSolver(
        space,
        assembly_backend="numba",
        boundary_mode="eliminate",
        solver="BICGSTAB",
        scale_system=True,
        cache_device_matrix=True,
        verbose=False,
    )
    solver.solve_rows = np.array([0, 0, 1, 1], dtype=np.int64)
    solver.solve_cols = np.array([0, 1, 0, 1], dtype=np.int64)
    solver.solve_data = np.array([2.0, -1.0, -1.0, 2.0])
    solver.solve_rhs = np.ones(2)

    host_matrix, device_matrix = solver._prepared_cupyx_operator(scale_system=True)
    scaled_matrix = solver._host_scaled_solve_matrix
    inverse_diagonal = solver._host_inverse_diagonal

    assert device_matrix is None
    assert host_matrix is not None
    assert scaled_matrix is not None
    assert inverse_diagonal == pytest.approx([0.5, 0.5])

    cached_host, cached_device = solver._prepared_cupyx_operator(scale_system=True)
    assert cached_host is host_matrix
    assert cached_device is None
    assert solver._host_scaled_solve_matrix is scaled_matrix
    assert solver._host_inverse_diagonal is inverse_diagonal


@pytest.mark.parametrize("equation", ("advection-reaction", "diffusion-reaction"))
def test_reusable_solver_preflights_before_coefficient_sampling(equation: str) -> None:
    space = _space()
    calls = 0

    def unexpected_coefficient_call(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("coefficient sampling must not run before backend preflight")

    if equation == "advection-reaction":
        solver = hdgfem.AdvectionReactionHDGSolver(
            space,
            source=unexpected_coefficient_call,
            beta=(unexpected_coefficient_call, unexpected_coefficient_call),
            reaction=unexpected_coefficient_call,
            assembly_backend="cupy",
            solver="direct",
            boundary_mode="zero-flux",
            materialize_host_solution=False,
            verbose=False,
        )
        match = "boundary_mode='zero-flux'"
    else:
        solver = hdgfem.DiffusionReactionHDGSolver(
            space,
            source=unexpected_coefficient_call,
            reaction=unexpected_coefficient_call,
            boundary_condition=unexpected_coefficient_call,
            assembly_backend="cupy",
            solver="direct",
            boundary_mode="eliminate",
            verbose=False,
        )
        match = "early-alpha support matrix"

    with pytest.raises(UnsupportedBackendConfigurationError, match=match):
        solver.solve()
    assert calls == 0


def test_diffusion_options_expose_local_factor_cache_policy() -> None:
    options = hdgfem.DiffusionReactionHDGOptions(cache_local_factors="schur-lu", verbose=False)

    assert options.cache_local_factors == "schur-lu"
    assert options.as_solve_kwargs()["cache_local_factors"] == "schur-lu"
    assert options.with_overrides(cache_local_factors="none").cache_local_factors == "none"

    cholesky = options.with_overrides(cache_local_factors="schur-cholesky")
    assert cholesky.cache_local_factors == "schur-cholesky"
    assert cholesky.as_solve_kwargs()["cache_local_factors"] == "schur-cholesky"


def test_diffusion_options_expose_flux_postprocessing_policy() -> None:
    options = hdgfem.DiffusionReactionHDGOptions(verbose=False)

    assert options.flux_postprocess_space == "l2_closest"
    assert options.postprocessing_backend == "auto"
    kwargs = options.as_solve_kwargs()
    assert kwargs["flux_postprocess_space"] == "l2_closest"
    assert kwargs["postprocessing_backend"] == "auto"
    updated = options.with_overrides(
        flux_postprocess_space="RT_projection",
        postprocessing_backend="cupy",
    )
    assert updated.flux_postprocess_space == "RT_projection"
    assert updated.postprocessing_backend == "cupy"


def test_diffusion_local_factor_cache_rejects_incompatible_configuration_before_runtime_setup() -> None:
    space = _space()
    solver = hdgfem.DiffusionReactionHDGSolver(
        space,
        source=space.constant(1.0),
        reaction=space.zeros(),
        boundary_condition=0.0,
        assembly_backend="numpy",
        cache_local_factors="schur-lu",
        verbose=False,
    )
    with pytest.raises(ValueError, match="assembly_backend='numba' or 'raw-cuda'"):
        solver.solve()

    with pytest.raises(ValueError, match="stateful DiffusionReactionHDGSolver"):
        hdgfem.solve_diffusion_reaction_hdg(
            1.0,
            0.0,
            0.0,
            space,
            cache_local_factors="schur-lu",
            verbose=False,
        )

    raw_cholesky = hdgfem.DiffusionReactionHDGSolver(
        space,
        source=space.constant(1.0),
        reaction=space.zeros(),
        boundary_condition=0.0,
        assembly_backend="numpy",
        raw_matrix_format="csr",
        boundary_mode="eliminate",
        cache_local_factors="schur-cholesky",
        verbose=False,
    )
    with pytest.raises(ValueError, match="requires assembly_backend='numba', 'cupy' or 'raw-cuda'"):
        raw_cholesky.solve()



def test_advection_options_expose_krylov_restart() -> None:
    options = hdgfem.AdvectionReactionHDGOptions(restart=37, verbose=False)

    assert options.restart == 37
    assert options.as_solve_kwargs()["restart"] == 37
    assert options.with_overrides(restart=19).restart == 19
