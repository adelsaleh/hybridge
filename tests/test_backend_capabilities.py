from __future__ import annotations

from pathlib import Path

import pytest

from hdgfem import (
    AdvectionReactionHDGSolver,
    DGSpace,
    DiffusionReactionHDGSolver,
    rectangle_mesh,
)
from hdgfem.backends import (
    BACKEND_CAPABILITIES,
    UnsupportedBackendConfigurationError,
    get_backend_capability,
)
from hdgfem.backends.capabilities import (
    normalize_solver_backend,
    render_backend_capability_table,
    validate_advection_backend_configuration,
    validate_diffusion_backend_configuration,
)
from hdgfem.solvers.advection_reaction import solve_advection_reaction_hdg
from hdgfem.solvers.diffusion_reaction import solve_diffusion_reaction_hdg


def _capability_id(capability) -> str:
    solver = capability.solver_backend or "none"
    return f"{capability.equation}-{capability.operation}-{capability.assembly_backend}-{solver}"


@pytest.mark.parametrize("capability", BACKEND_CAPABILITIES, ids=_capability_id)
def test_every_published_capability_has_an_exact_registry_lookup(capability) -> None:
    assert get_backend_capability(*capability.key) is capability
    assert capability.boundary_modes
    assert capability.trace_bases
    assert capability.assembly_residency
    assert capability.solve_residency
    assert capability.reconstruction_residency


@pytest.mark.parametrize("capability", BACKEND_CAPABILITIES, ids=_capability_id)
def test_every_published_capability_passes_preflight(capability) -> None:
    solver_names = {
        None: None,
        "scipy": "direct",
        "pypardiso": "pypardiso",
        "petsc": "petsc",
        "cupyx": "cupyx_bicgstab",
        "amgx": "amgx",
    }
    common = {
        "operation": capability.operation,
        "assembly_backend": capability.assembly_backend,
        "solver": solver_names[capability.solver_backend],
        "cupyx_solver": "bicgstab",
        "boundary_mode": capability.boundary_modes[0],
        "trace_basis": capability.trace_bases[0],
    }

    if capability.equation == "advection-reaction":
        actual = validate_advection_backend_configuration(
            **common,
            trace_ordering="none",
            materialize_host_solution=True,
            raw_local_assembly="fused",
            raw_lu_mode="safe",
            raw_matrix_format=(
                "csr"
                if capability.operation == "solve" and capability.solver_backend == "amgx"
                else "auto"
            ),
            requires_host_system=False,
            advection_stabilization_is_default=True,
        )
    else:
        actual = validate_diffusion_backend_configuration(
            **common,
            local_solver_backend="numpy",
            raw_matrix_format="csr" if capability.assembly_backend == "raw-cuda" else "coo",
            postprocess_mode="none",
            identity_diffusion=True,
            scalar_stabilization=True,
            allow_raw_device_solve=True,
        )

    assert actual is capability


@pytest.mark.parametrize(
    "capability",
    tuple(capability for capability in BACKEND_CAPABILITIES if capability.operation == "solve"),
    ids=_capability_id,
)
def test_every_published_solve_capability_constructs_through_reusable_api(capability) -> None:
    solver_names = {
        "scipy": "direct",
        "pypardiso": "pypardiso",
        "petsc": "petsc",
        "cupyx": "cupyx_bicgstab",
        "amgx": "amgx",
    }
    options = {
        "assembly_backend": capability.assembly_backend,
        "solver": solver_names[capability.solver_backend],
        "boundary_mode": capability.boundary_modes[0],
        "trace_basis": capability.trace_bases[0],
        "verbose": False,
    }
    space = DGSpace(rectangle_mesh(1, 1), 1)

    if capability.equation == "advection-reaction":
        options.update(
            materialize_host_solution=True,
            raw_local_assembly="fused",
            raw_matrix_format="csr" if capability.solver_backend == "amgx" else "auto",
        )
        solver = AdvectionReactionHDGSolver(space, **options)
    else:
        options.update(
            raw_matrix_format="csr" if capability.assembly_backend == "raw-cuda" else "coo",
            hdg_postprocess="none",
        )
        solver = DiffusionReactionHDGSolver(space, **options)

    assert solver.options.assembly_backend == capability.assembly_backend
    assert (
        normalize_solver_backend(solver.options.solver, cupyx_solver=solver.options.cupyx_solver)
        == capability.solver_backend
    )
    assert solver.options.boundary_mode == capability.boundary_modes[0]
    assert solver.options.trace_basis == capability.trace_bases[0]


def test_capability_keys_are_unique() -> None:
    keys = [capability.key for capability in BACKEND_CAPABILITIES]
    assert len(keys) == len(set(keys))


def test_checked_in_capability_table_matches_registry() -> None:
    text = Path("docs/reference/backend_capabilities.md").read_text(encoding="utf-8")
    start_marker = "<!-- BEGIN GENERATED CAPABILITY MATRIX -->"
    end_marker = "<!-- END GENERATED CAPABILITY MATRIX -->"
    generated = text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
    assert generated == render_backend_capability_table()


@pytest.mark.parametrize(
    "solver,expected",
    (
        (None, "scipy"),
        ("direct", "scipy"),
        ("GMRES", "scipy"),
        ("pypardiso", "pypardiso"),
        ("pardiso", "pypardiso"),
        ("pypardiso-spd", "pypardiso"),
        ("pardiso_spd", "pypardiso"),
        ("petsc", "petsc"),
        ("cupyx_bicgstab", "cupyx"),
        ("pyamgx", "amgx"),
    ),
)
def test_global_solver_names_map_to_stable_backend_families(solver, expected: str) -> None:
    assert normalize_solver_backend(solver) == expected


def test_unknown_global_solver_is_rejected_during_preflight() -> None:
    with pytest.raises(ValueError, match="unknown global solver"):
        normalize_solver_backend("not-a-solver")


def _advection_preflight(**overrides):
    values = {
        "operation": "solve",
        "assembly_backend": "numpy",
        "solver": "direct",
        "cupyx_solver": "bicgstab",
        "boundary_mode": "eliminate",
        "trace_basis": "legacy-lagrange",
        "trace_ordering": "none",
        "materialize_host_solution": True,
        "raw_local_assembly": "fused",
        "raw_lu_mode": "safe",
        "raw_matrix_format": "auto",
        "requires_host_system": False,
        "advection_stabilization_is_default": True,
    }
    values.update(overrides)
    return validate_advection_backend_configuration(**values)


@pytest.mark.parametrize(
    "overrides,reason",
    (
        ({"assembly_backend": "cupy", "materialize_host_solution": False}, "materialize_host_solution=True"),
        ({"assembly_backend": "numpy", "boundary_mode": "zero-flux"}, "boundary_mode='zero-flux'"),
        ({"assembly_backend": "raw-cuda", "trace_ordering": "upwind-scc"}, "trace_ordering='none'"),
        (
            {
                "assembly_backend": "raw-cuda",
                "solver": "amgx",
                "raw_matrix_format": "csr",
                "requires_host_system": True,
            },
            "no host-system diagnostics",
        ),
    ),
)
def test_unsupported_advection_combinations_use_stable_actionable_error(overrides, reason: str) -> None:
    with pytest.raises(UnsupportedBackendConfigurationError) as exc_info:
        _advection_preflight(**overrides)
    message = str(exc_info.value)
    assert "unsupported backend configuration" in message
    assert reason in message
    assert "docs/reference/backend_capabilities.md" in message


def _diffusion_preflight(**overrides):
    values = {
        "operation": "solve",
        "assembly_backend": "numpy",
        "solver": "direct",
        "cupyx_solver": "bicgstab",
        "boundary_mode": "eliminate",
        "trace_basis": "legacy-lagrange",
        "local_solver_backend": "numpy",
        "raw_matrix_format": "coo",
        "postprocess_mode": "none",
        "identity_diffusion": True,
        "scalar_stabilization": True,
        "allow_raw_device_solve": True,
    }
    values.update(overrides)
    return validate_diffusion_backend_configuration(**values)


@pytest.mark.parametrize(
    "overrides,reason",
    (
        ({"assembly_backend": "cupy"}, "not in the early-alpha support matrix"),
        ({"assembly_backend": "numba", "boundary_mode": "penalty"}, "boundary_mode='penalty'"),
        (
            {"assembly_backend": "raw-cuda", "solver": "amgx", "raw_matrix_format": "coo"},
            "raw_matrix_format='csr'",
        ),
        (
            {
                "assembly_backend": "raw-cuda",
                "solver": "amgx",
                "raw_matrix_format": "csr",
                "allow_raw_device_solve": False,
            },
            "DiffusionReactionHDGSolver",
        ),
        (
            {
                "operation": "assemble",
                "assembly_backend": "cupy",
                "solver": None,
                "identity_diffusion": False,
            },
            "identity diffusion only",
        ),
    ),
)
def test_unsupported_diffusion_combinations_use_stable_actionable_error(overrides, reason: str) -> None:
    with pytest.raises(UnsupportedBackendConfigurationError) as exc_info:
        _diffusion_preflight(**overrides)
    message = str(exc_info.value)
    assert "unsupported backend configuration" in message
    assert reason in message
    assert "docs/reference/backend_capabilities.md" in message


def test_public_solvers_preflight_before_coefficient_sampling_or_optional_backend_import() -> None:
    space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
    calls = 0

    def unexpected_coefficient_call(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("coefficient sampling must not run before backend preflight")

    with pytest.raises(UnsupportedBackendConfigurationError, match="materialize_host_solution=True"):
        solve_advection_reaction_hdg(
            unexpected_coefficient_call,
            (unexpected_coefficient_call, unexpected_coefficient_call),
            unexpected_coefficient_call,
            unexpected_coefficient_call,
            space,
            assembly_backend="cupy",
            boundary_mode="eliminate",
            solver="direct",
            materialize_host_solution=False,
            verbose=False,
        )

    with pytest.raises(UnsupportedBackendConfigurationError, match="early-alpha support matrix"):
        solve_diffusion_reaction_hdg(
            unexpected_coefficient_call,
            unexpected_coefficient_call,
            unexpected_coefficient_call,
            space,
            assembly_backend="cupy",
            boundary_mode="eliminate",
            solver="direct",
            verbose=False,
        )

    assert calls == 0


def test_invalid_solver_name_fails_before_advection_coefficient_sampling() -> None:
    space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
    calls = 0

    def unexpected_coefficient_call(*_args):
        nonlocal calls
        calls += 1
        return 0.0

    with pytest.raises(ValueError, match="unknown global solver"):
        solve_advection_reaction_hdg(
            unexpected_coefficient_call,
            (unexpected_coefficient_call, unexpected_coefficient_call),
            unexpected_coefficient_call,
            unexpected_coefficient_call,
            space,
            assembly_backend="numpy",
            boundary_mode="eliminate",
            solver="not-a-solver",
            verbose=False,
        )
    assert calls == 0


def test_unsupported_configuration_error_preserves_not_implemented_category() -> None:
    assert issubclass(UnsupportedBackendConfigurationError, NotImplementedError)
