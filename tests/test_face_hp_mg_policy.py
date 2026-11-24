"""Polynomial levels can be compared without changing the other cycle settings."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hybridge.linalg.gpu.legendre_face_bsr import modal_degree_schedule
from hybridge.linalg.multigrid.policy import face_hp_mg_preconditioner_parameters
from hybridge.solvers.diffusion_raw_cuda import solve_raw_cuda_device_amgx
from hybridge.solvers.diffusion_reaction import (
    DiffusionReactionHDGOptions,
    solve_diffusion_reaction_hdg,
)


@pytest.mark.parametrize("policy", ["standard", "fast", "robust"])
@pytest.mark.parametrize("schedule", ["direct-to-zero", "halve"])
def test_schedule_override_preserves_relaxation_and_coarse_solve(policy, schedule):
    original = face_hp_mg_preconditioner_parameters(policy)
    overrides = {"schedule": schedule}
    changed = face_hp_mg_preconditioner_parameters(policy, overrides=overrides)
    expected = dict(original, schedule=schedule)
    assert changed == expected
    assert overrides == {"schedule": schedule}
    assert face_hp_mg_preconditioner_parameters(policy) == original


@pytest.mark.parametrize(
    ("degree", "expected"),
    [(6, (6, 3, 1, 0)), (7, (7, 3, 1, 0)),
     (8, (8, 4, 2, 1, 0)), (9, (9, 4, 2, 1, 0))],
)
def test_overridden_schedule_reaches_expected_polynomial_spaces(degree, expected):
    parameters = face_hp_mg_preconditioner_parameters(
        "standard", overrides={"schedule": "halve"},
    )
    assert modal_degree_schedule(degree, parameters["schedule"]) == expected
    baseline = face_hp_mg_preconditioner_parameters("standard")
    assert modal_degree_schedule(degree, baseline["schedule"]) == (degree, 0)


@pytest.mark.parametrize("schedule", [None, True, 2, "linear", "", [6, 3, 0]])
def test_invalid_schedule_is_rejected_before_device_setup(schedule):
    options = DiffusionReactionHDGOptions(
        solver="fb-hp-mg-pcg", assembly_backend="raw-cuda",
        trace_basis="legendre-modal", raw_matrix_format="bsr",
        scale_system=False, boundary_mode="eliminate",
        fb_hp_mg_preconditioner_tuning={"schedule": schedule},
    )
    owner = SimpleNamespace(options=options, space=SimpleNamespace(order=6))
    with pytest.raises(ValueError, match="schedule must be"):
        solve_raw_cuda_device_amgx(owner)


def test_native_tuning_is_not_silently_ignored_by_amgx():
    options = DiffusionReactionHDGOptions(
        solver="amgx", assembly_backend="raw-cuda", raw_matrix_format="csr",
        fb_hp_mg_preconditioner_tuning={"schedule": "halve"},
    )
    owner = SimpleNamespace(options=options, space=SimpleNamespace(order=6))
    with pytest.raises(ValueError, match="tuning requires solver='fb-hp-mg-pcg'"):
        solve_raw_cuda_device_amgx(owner)


def test_diffusion_options_retain_tuning_in_keyword_interface():
    tuning = {"schedule": "halve", "chebyshev_order": 2}
    options = DiffusionReactionHDGOptions(fb_hp_mg_preconditioner_tuning=tuning)
    kwargs = options.as_solve_kwargs()
    assert kwargs["fb_hp_mg_preconditioner_tuning"] == tuning
    parameters = inspect.signature(solve_diffusion_reaction_hdg).parameters
    assert parameters["fb_hp_mg_preconditioner_tuning"].default is None
