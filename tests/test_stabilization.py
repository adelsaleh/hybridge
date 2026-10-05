"""Global physical-length diffusion stabilization tests."""

from __future__ import annotations

import numpy as np
import pytest

from hybridge import (
    DGSpace,
    GlobalLengthDiffusion,
    automatic_domain_length,
    compute_domain_length,
    geometric_diffusion_tau,
    gmsh_disc_mesh,
    mesh_domain_measures,
    rectangle_mesh,
)
from hybridge.mixed.adr_preparation import (
    prepare_adr_data,
    recommended_diffusion_stabilization,
)
from hybridge.solvers.advection_diffusion_reaction import (
    AdvectionDiffusionReactionHDGOptions,
)
from hybridge.solvers.diffusion_reaction import (
    DiffusionReactionHDGOptions,
    DiffusionReactionHDGSolver,
    solve_diffusion_reaction_hdg,
)
from scripts.diffusion_reaction.run_cases import (
    DiffusionReactionRunPreset,
    PRESETS,
)
from scripts.gpu.run_diffusion_reaction_cuda import (
    build_arg_parser as build_cuda_diffusion_arg_parser,
)


def test_domain_length_helpers_on_two_by_one_rectangle() -> None:
    """Recover exact affine area, perimeter, and hydraulic-radius length."""
    mesh = rectangle_mesh(
        3,
        2,
        xlim=(0.0, 2.0),
        ylim=(-0.5, 0.5),
    )
    domain_measure, boundary_measure = mesh_domain_measures(mesh)
    assert domain_measure == pytest.approx(2.0, rel=0.0, abs=2.0e-15)
    assert boundary_measure == pytest.approx(6.0, rel=0.0, abs=2.0e-15)
    assert compute_domain_length(domain_measure, boundary_measure, 2) == pytest.approx(2.0 / 3.0)
    assert automatic_domain_length(mesh) == pytest.approx(2.0 / 3.0)


def test_unit_disk_global_length_is_one_up_to_polygon_geometry() -> None:
    """Recover the unit-disk scale, with explicit length giving exact tau one."""
    pytest.importorskip("gmsh")
    mesh = gmsh_disc_mesh(
        0.3,
        radius=1.0,
        verbosity=0,
        log_cache=False,
    )
    space = DGSpace(mesh, 1, basis_type="dub_orth")
    length = automatic_domain_length(space)
    assert length == pytest.approx(1.0, rel=1.5e-2)
    assert GlobalLengthDiffusion().resolve(1.0, space) == pytest.approx(1.0 / length)
    assert GlobalLengthDiffusion(domain_length=1.0).resolve(1.0, space) == 1.0


@pytest.mark.parametrize("order", (1, 3, 6))
@pytest.mark.parametrize("subdivisions", (1, 2, 5))
def test_global_length_policy_is_independent_of_h_and_p(
        order: int,
        subdivisions: int,
) -> None:
    """Keep the scalar global-length value fixed across production h/p changes."""
    space = DGSpace(
        rectangle_mesh(
            2 * subdivisions,
            subdivisions,
            xlim=(0.0, 2.0),
            ylim=(0.0, 1.0),
        ),
        order,
        basis_type="dub_orth",
    )
    policy = GlobalLengthDiffusion(gamma_d=1.5)
    assert policy.resolve(0.2, space) == pytest.approx(0.45)
    assert geometric_diffusion_tau(0.2, 2.0 / 3.0, 1.5) == pytest.approx(0.45)


def test_global_length_policy_has_correct_uniform_domain_scaling() -> None:
    """Scale L by s and the resolved stabilization by one over s."""
    base = DGSpace(
        rectangle_mesh(2, 1, xlim=(0.0, 2.0), ylim=(0.0, 1.0)),
        2,
        basis_type="dub_orth",
    )
    scaled = DGSpace(
        rectangle_mesh(2, 1, xlim=(0.0, 4.0), ylim=(0.0, 2.0)),
        2,
        basis_type="dub_orth",
    )
    policy = GlobalLengthDiffusion()
    assert automatic_domain_length(scaled) == pytest.approx(
        2.0 * automatic_domain_length(base)
    )
    assert policy.resolve(0.2, scaled) == pytest.approx(
        0.5 * policy.resolve(0.2, base)
    )


def test_global_length_policy_accepts_constant_isotropic_tensor_encodings() -> None:
    """Treat scalar, symmetric-component, and matrix isotropic data identically."""
    space = DGSpace(
        rectangle_mesh(2, 1, xlim=(0.0, 2.0), ylim=(0.0, 1.0)),
        1,
        basis_type="dub_orth",
    )
    policy = GlobalLengthDiffusion()
    expected = policy.resolve(0.2, space)
    assert policy.resolve((0.2, 0.0, 0.2), space) == pytest.approx(expected)
    assert policy.resolve(0.2 * np.eye(2), space) == pytest.approx(expected)


def test_global_length_policy_rejects_unsupported_or_nonpositive_inputs() -> None:
    """Resolve anisotropic normals and reject invalid physical scales."""
    space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
    normals = space.mesh.normals
    expected = (normals[..., 0]**2 + 2.*normals[..., 1]**2) / automatic_domain_length(space)
    np.testing.assert_allclose(GlobalLengthDiffusion().resolve(np.diag((1., 2.)), space), expected)
    with pytest.raises(ValueError, match="normal diffusivity"):
        GlobalLengthDiffusion().resolve(0.0, space)
    with pytest.raises(ValueError, match="gamma_d"):
        GlobalLengthDiffusion(gamma_d=0.0).resolve(1.0, space)
    with pytest.raises(ValueError, match="domain_length"):
        GlobalLengthDiffusion(domain_length=0.0).resolve(1.0, space)


def test_solver_defaults_select_global_length_and_inverse_h_is_explicit() -> None:
    """Make global length the default while preserving the legacy ADR rule."""
    assert DiffusionReactionHDGOptions().stabilization == "global_length"
    assert (
        AdvectionDiffusionReactionHDGOptions().diffusion_stabilization
        == "global_length"
    )
    space = DGSpace(
        rectangle_mesh(2, 1, xlim=(0.0, 2.0), ylim=(0.0, 1.0)),
        2,
        basis_type="dub_orth",
    )
    source = space.constant(1.0)
    reaction = space.constant(0.5)
    beta = (space * space).field((space.constant(0.7), space.constant(-0.2)))
    default = prepare_adr_data(source, reaction, beta, space, diffusion=0.2)
    np.testing.assert_allclose(default.tau_diffusion, 0.3)
    legacy = prepare_adr_data(
        source,
        reaction,
        beta,
        space,
        diffusion=0.2,
        diffusion_stabilization="inverse-h",
    )
    np.testing.assert_allclose(
        legacy.tau_diffusion,
        recommended_diffusion_stabilization(space, 0.2),
    )


def test_diffusion_runner_defaults_and_fixed_tau_presets_are_unambiguous() -> None:
    """Default runners to global length without changing fixed benchmark inputs."""
    generic = DiffusionReactionRunPreset(
        case="quadratic-poisson",
        description="test",
    )
    assert generic.diffusion_stabilization_mode == "global-length"
    fixed_tau = [preset for preset in PRESETS.values() if preset.tau != 1.0]
    assert fixed_tau
    assert all(
        preset.diffusion_stabilization_mode == "explicit"
        for preset in fixed_tau
    )

    cuda_defaults = build_cuda_diffusion_arg_parser().parse_args([])
    assert cuda_defaults.diffusion_stabilization_mode == "global-length"
    assert cuda_defaults.tau is None
    cuda_explicit = build_cuda_diffusion_arg_parser().parse_args(
        ["--tau", "0.7"]
    )
    assert cuda_explicit.tau == pytest.approx(0.7)


def test_adr_policy_lowers_to_constant_per_incidence_table() -> None:
    """Lower global-length ADR stabilization without altering face incidence assembly."""
    space = DGSpace(
        rectangle_mesh(2, 1, xlim=(0.0, 2.0), ylim=(0.0, 1.0)),
        2,
        basis_type="dub_orth",
    )
    source = space.constant(1.0)
    reaction = space.constant(0.5)
    beta = (space * space).field((space.constant(0.7), space.constant(-0.2)))
    prepared = prepare_adr_data(
        source,
        reaction,
        beta,
        space,
        diffusion=0.2,
        diffusion_stabilization=GlobalLengthDiffusion(gamma_d=1.5),
    )
    np.testing.assert_allclose(prepared.tau_diffusion, 0.45, rtol=0.0, atol=2.0e-15)


def test_diffusion_reusable_solver_policy_matches_explicit_scalar_assembly() -> None:
    """Resolve the policy before NumPy assembly and preserve the requested options."""
    space = DGSpace(
        rectangle_mesh(2, 1, xlim=(0.0, 2.0), ylim=(0.0, 1.0)),
        1,
        basis_type="dub_orth",
    )
    source = lambda x, y: 1.0 + 0.1 * x
    reaction = lambda x, y: 0.3 + 0.0 * x
    boundary = lambda x, y: x - 0.25 * y
    common = dict(
        diffusion=0.2,
        assembly_backend="numpy",
        boundary_mode="eliminate",
        solver=None,
        verbose=False,
    )
    policy = GlobalLengthDiffusion(gamma_d=1.5)
    policy_solver = DiffusionReactionHDGSolver(
        space,
        source=source,
        reaction=reaction,
        boundary_condition=boundary,
        options=DiffusionReactionHDGOptions(stabilization=policy, **common),
    )
    explicit_solver = DiffusionReactionHDGSolver(
        space,
        source=source,
        reaction=reaction,
        boundary_condition=boundary,
        options=DiffusionReactionHDGOptions(stabilization=0.45, **common),
    )
    policy_assembly = policy_solver.assemble_global_matrix()
    explicit_assembly = explicit_solver.assemble_global_matrix()
    assert policy_solver.options.stabilization is policy
    np.testing.assert_array_equal(policy_assembly.rows, explicit_assembly.rows)
    np.testing.assert_array_equal(policy_assembly.cols, explicit_assembly.cols)
    np.testing.assert_allclose(policy_assembly.data, explicit_assembly.data, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(policy_assembly.rhs, explicit_assembly.rhs, rtol=0.0, atol=2.0e-14)


def test_diffusion_functional_solver_policy_matches_explicit_solution() -> None:
    """Resolve the policy in the one-shot solver before validation and assembly."""
    space = DGSpace(
        rectangle_mesh(1, 1, xlim=(0.0, 2.0), ylim=(0.0, 1.0)),
        1,
        basis_type="dub_orth",
    )
    source = lambda x, y: 1.0 + 0.1 * x
    reaction = lambda x, y: 0.3 + 0.0 * x
    boundary = lambda x, y: x - 0.25 * y
    common = dict(
        diffusion=0.2,
        solver="direct",
        preconditioner=None,
        scale_system=False,
        assembly_backend="numpy",
        boundary_mode="eliminate",
        verbose=False,
    )
    policy_result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        boundary,
        space,
        stabilization=GlobalLengthDiffusion(gamma_d=1.5),
        **common,
    )
    explicit_result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        boundary,
        space,
        stabilization=0.45,
        **common,
    )
    np.testing.assert_allclose(policy_result.trace, explicit_result.trace, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        policy_result.field.coeffs,
        explicit_result.field.coeffs,
        rtol=0.0,
        atol=2.0e-13,
    )
