"""Recovery-cache lifetime checks using host metadata; no solve or CUDA launch."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from hdgfem.backends import diffusion_flux_recovery_raw_cuda as raw
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diffusion_reaction import (
    DiffusionReactionHDGOptions, DiffusionReactionHDGSolver,
    _new_hdg_postprocess_cache,
)
from scripts.guiding_center.poisson.poisson_recovery import (
    PoissonCheckpoint, PoissonStageFailure, PoissonTauRecovery,
)


HOST_FACTORS = (
    "flux_ainv_constraint_t", "flux_schur_lu", "flux_schur_pivots",
    "primal_lu", "primal_pivots",
)


def cached_solver(monkeypatch, *, trace_basis="legendre-modal",
                  variant="l2_closest", postprocessing_backend="auto"):
    """Seed real cache containers with opaque buffers and a controllable device."""
    active = {"device": 0}
    monkeypatch.setattr(raw, "require_cupy", lambda: SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: active["device"]))))
    space = DGSpace(rectangle_mesh(1, 1), 2, basis_type="dub_orth")
    options = DiffusionReactionHDGOptions(
        stabilization=0.4, assembly_backend="raw-cuda", boundary_mode="eliminate",
        trace_basis=trace_basis, hdg_postprocess="flux", flux_postprocess_space=variant,
        postprocessing_backend=postprocessing_backend, verbose=False,
    )
    solver = DiffusionReactionHDGSolver(
        space, source=space.zeros(), reaction=space.zeros(), boundary_condition=0., options=options)
    trace_space = space.trace_space(trace_basis)
    cache = _new_hdg_postprocess_cache(space, trace_space)
    cache.raw_flux_cache = raw.FluxRecoveryDeviceCache(
        key=(id(space), id(trace_space), variant, 0), reference=object(),
        cspace=object(), arrays=tuple(object() for _ in range(6)),
        cholesky=object(), kernel=object(),
    )
    for name in HOST_FACTORS:
        setattr(cache, name, object())
    solver._hdg_postprocess_cache = cache
    return solver, cache, active


class OwnedSolver:
    """Record closure of a tau-dependent native or AMGX solver context."""

    def __init__(self):
        self.closed = 0

    def close(self, **kwargs):
        self.closed += 1


@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("variant", ["RT_projection", "l2_closest"])
@pytest.mark.parametrize("backend", ["auto", "raw-cuda"])
def test_tau_retry_preserves_recovery_but_invalidates_operators(
        monkeypatch, trace_basis, variant, backend):
    solver, original, _ = cached_solver(
        monkeypatch, trace_basis=trace_basis, variant=variant, postprocessing_backend=backend)
    invalidated = (
        "_raw_cuda_assembly_cache", "_raw_cuda_operator_key", "_raw_cuda_last_trace_reduced",
        "_cupy_assembly_cache", "_cupy_operator_key", "_cupy_last_trace_reduced",
        "_device_solve_matrix", "_host_solve_matrix", "_host_scaled_solve_matrix",
        "local_solver", "element_boundary_mats", "local_unknowns", "result", "field",
        "flux", "postprocessed_flux", "postprocessed_field", "trace", "data", "rhs",
    )
    for tau in (np.float64(0.8), 2):
        for name in invalidated:
            setattr(solver, name, object())
        solver._raw_cuda_rhs_valid = solver._cupy_rhs_valid = solver._host_cached_rhs_valid = True
        owners = [OwnedSolver() for _ in range(4)]
        solver._raw_cuda_fb_hp_mg_solver, solver._raw_cuda_amgx_solver, solver._cupy_amgx_solver = owners[:3]
        solver._raw_cuda_amgx_retry_solver_cache = {"first": owners[3], "alias": owners[3]}
        assert solver.with_options(stabilization=tau) is solver
        retained = solver._hdg_postprocess_cache
        for item in fields(original):
            if item.name in HOST_FACTORS:
                assert getattr(retained, item.name) is None
            else:
                assert getattr(retained, item.name) is getattr(original, item.name)
        assert all(getattr(solver, name) is None for name in invalidated)
        assert not solver._raw_cuda_rhs_valid and not solver._cupy_rhs_valid
        assert not solver._host_cached_rhs_valid
        assert [owner.closed for owner in owners] == [1, 1, 1, 1]
        assert solver._raw_cuda_fb_hp_mg_solver is solver._raw_cuda_amgx_solver is None
        assert solver._cupy_amgx_solver is None and not solver._raw_cuda_amgx_retry_solver_cache
        solver.set_source(solver.space.constant(tau))
        solver.set_boundary_condition(float(tau))
        assert solver._hdg_postprocess_cache is retained


@pytest.mark.parametrize("overrides", [
    {"trace_basis": "legacy-lagrange"}, {"flux_postprocess_space": "RT_projection"},
    {"postprocessing_backend": "numba"}, {"hdg_postprocess": "none"},
    {"diffusion": 2.}, {"stabilization": 2., "verbose": True},
    {"stabilization": np.ones((2, 3))}, {"stabilization": "global_length"},
    {"stabilization": float("nan")}, {"stabilization": float("inf")}, {},
])
def test_other_option_updates_clear_recovery(monkeypatch, overrides):
    solver, _, _ = cached_solver(monkeypatch)
    solver.with_options(**overrides)
    assert solver._hdg_postprocess_cache is None


@pytest.mark.parametrize("mismatch", ["space", "trace", "variant", "device", "missing_raw"])
def test_tau_retry_rejects_incompatible_recovery_cache(monkeypatch, mismatch):
    solver, cache, active = cached_solver(monkeypatch)
    if mismatch == "space":
        cache.base_space = DGSpace(solver.space.mesh, 3, basis_type="bernstein")
    elif mismatch == "trace":
        cache.trace_space = solver.space.trace_space("legacy-lagrange")
    elif mismatch == "variant":
        cache.raw_flux_cache.key = (*cache.raw_flux_cache.key[:2], "RT_projection", 0)
    elif mismatch == "device":
        active["device"] = 1
    else:
        cache.raw_flux_cache = None
    solver.with_options(stabilization=0.8)
    assert solver._hdg_postprocess_cache is None


@pytest.mark.parametrize("reset", ["clear_cache", "close", "space", "mesh", "reaction"])
def test_explicit_resets_release_recovery_cache(monkeypatch, reset):
    solver, _, _ = cached_solver(monkeypatch)
    if reset == "space":
        solver.set_space(DGSpace(solver.space.mesh, 3, basis_type="bernstein"))
    elif reset == "mesh":
        solver.set_mesh(rectangle_mesh(2, 1))
    elif reset == "reaction":
        solver.set_reaction(solver.space.constant(1.))
    else:
        getattr(solver, reset)()
    assert solver._hdg_postprocess_cache is None


@pytest.mark.parametrize("initial_tau", [0.4, "global_length"])
def test_poisson_retry_controller_preserves_recovery_from_resolved_scalar_tau(monkeypatch, initial_tau):
    solver, cache, _ = cached_solver(monkeypatch)
    solver.options = solver.options.with_overrides(stabilization=initial_tau)
    old_tau = float(solver._resolved_options().stabilization)
    checkpoint = PoissonCheckpoint(solver.source, object(), 0., "prescribed checkpoint")
    failure = PoissonStageFailure(RuntimeError("failed to converge"), checkpoint, "transport")
    retry = PoissonTauRecovery(factor=2., max_retries=2)
    for count in (1, 2):
        assert retry.increase(solver, failure, step_time=0.) == old_tau * 2**count
        assert solver._hdg_postprocess_cache.raw_flux_cache is cache.raw_flux_cache
    assert len(retry.events) == 2 and checkpoint.time == 0.


def test_tau_update_without_recovery_does_not_load_cuda(monkeypatch):
    def reject():
        raise AssertionError("Host option updates must not initialize CUDA")
    monkeypatch.setattr(raw, "require_cupy", reject)
    space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
    solver = DiffusionReactionHDGSolver(space, verbose=False)
    solver.with_options(stabilization=2.)
    assert solver._hdg_postprocess_cache is None


def test_unknown_option_does_not_discard_recovery(monkeypatch):
    solver, cache, _ = cached_solver(monkeypatch)
    options = solver.options
    with pytest.raises(TypeError, match="unknown"):
        solver.with_options(stabilization=2., unknown_option=True)
    assert solver.options is options and solver._hdg_postprocess_cache is cache
