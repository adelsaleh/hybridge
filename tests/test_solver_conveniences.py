"""Defaults and lifecycle helpers that keep reusable-solver scripts short."""

from __future__ import annotations

import numpy as np
import pytest

import hdgfem as hdg
from hdgfem import DGSpace, rectangle_mesh
from hdgfem.cases.profiles import GaussianBlobField
from hdgfem.core.field_ops import perpendicular_vector_field
from hdgfem.diagnostics import transport_velocity_diagnostics


def _space(order: int = 2, cells: int = 2) -> DGSpace:
    return DGSpace(rectangle_mesh(cells, cells), order, basis_type="dub_orth", volume_quad_1d=6)


@pytest.mark.parametrize("solver_class", (
    hdg.AdvectionReactionHDGSolver,
    hdg.DiffusionReactionHDGSolver,
    hdg.AdvectionDiffusionReactionHDGSolver,
))
def test_reusable_solvers_close_when_their_with_block_exits(monkeypatch, solver_class):
    closed = []
    # Record identities only: keeping ``self`` would resurrect solvers whose
    # finalizers also call close().
    monkeypatch.setattr(solver_class, "close", lambda self: closed.append(id(self)))
    with solver_class(_space()) as solver:
        assert not closed
    assert closed == [id(solver)]


def test_context_manager_does_not_suppress_errors(monkeypatch):
    monkeypatch.setattr(hdg.DiffusionReactionHDGSolver, "close", lambda self: None)
    with pytest.raises(RuntimeError, match="inside"):
        with hdg.DiffusionReactionHDGSolver(_space()):
            raise RuntimeError("inside")


def test_fb_hp_mg_pcg_and_device_assembly_imply_their_only_valid_options():
    options = hdg.DiffusionReactionHDGSolver(
        _space(), solver="fb-hp-mg-pcg", assembly_backend="raw-cuda").options
    assert options.trace_basis == "legendre-modal"
    assert options.scale_system is False
    assert options.boundary_mode == "eliminate"
    explicit = hdg.DiffusionReactionHDGSolver(
        _space(), solver="fb-hp-mg-pcg", assembly_backend="raw-cuda",
        trace_basis="legacy-lagrange", boundary_mode="penalty").options
    assert explicit.trace_basis == "legacy-lagrange"
    assert explicit.boundary_mode == "penalty"
    host = hdg.DiffusionReactionHDGSolver(_space(), assembly_backend="numba").options
    assert host.trace_basis == "legacy-lagrange"
    assert host.scale_system is True
    assert host.boundary_mode == "eliminate"
    numpy_host = hdg.DiffusionReactionHDGSolver(_space()).options
    assert numpy_host.boundary_mode == "penalty"


def test_options_object_disables_inference():
    options = hdg.DiffusionReactionHDGOptions(solver="fb-hp-mg-pcg", assembly_backend="raw-cuda")
    assert hdg.DiffusionReactionHDGSolver(_space(), options=options).options == options


def test_raw_zero_flux_transport_implies_auto_local_assembly():
    implied = hdg.AdvectionReactionHDGSolver(
        _space(), assembly_backend="raw-cuda", boundary_mode="zero-flux").options
    assert implied.raw_local_assembly == "auto"
    explicit = hdg.AdvectionReactionHDGSolver(
        _space(), assembly_backend="raw-cuda", boundary_mode="zero-flux",
        raw_local_assembly="split3").options
    assert explicit.raw_local_assembly == "split3"
    host = hdg.AdvectionReactionHDGSolver(_space(), boundary_mode="zero-flux").options
    assert host.raw_local_assembly == "precomputed"


def test_poisson_reaction_defaults_to_zero():
    space = _space()
    solver = hdg.DiffusionReactionHDGSolver(
        space, source=space.constant(1.0), boundary_condition=0.0, assembly_backend="numba",
        boundary_mode="eliminate", verbose=False)
    assert solver.reaction.is_zero
    with pytest.raises(ValueError, match="boundary_condition"):
        hdg.DiffusionReactionHDGSolver(space, source=space.constant(1.0))


def test_perpendicular_vector_field_defaults_to_unit_scale_and_flux_space():
    space = _space()
    x = space.project_callable(lambda x, y: x)
    one = space.constant(1.0)
    flux = (space * space).field((one.coeffs, x.coeffs), name="q")
    rotated = perpendicular_vector_field(flux)
    np.testing.assert_allclose(rotated.components[0].coeffs, -x.coeffs)
    np.testing.assert_allclose(rotated.components[1].coeffs, one.coeffs)
    assert rotated.components[0].space is space


def test_gaussian_blob_profile_round_trips_cutoff_and_amplitude(tmp_path):
    field = GaussianBlobField([[0.1, 0.2], [-0.3, 0.4]], [0.05, 0.1], [2.0, -1.0], cutoff=5.0)
    path = tmp_path / "profile.npz"
    field.save(path, amplitude=2.0)
    loaded = GaussianBlobField.load(path)
    assert loaded.cutoff == 5.0
    np.testing.assert_array_equal(loaded.centers, field.centers)
    np.testing.assert_array_equal(loaded.strengths, field.strengths)
    doubled = GaussianBlobField.load(path, amplitude=4.0)
    np.testing.assert_allclose(doubled.strengths, 2.0 * field.strengths)
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, centers=field.centers, sigmas=field.sigmas, strengths=field.strengths)
    assert GaussianBlobField.load(legacy).cutoff == 8.0


@pytest.mark.parametrize(("sign", "outflow", "inflow"), ((-1.0, True, False), (1.0, False, True)))
def test_velocity_diagnostics_classify_conflicting_face_traces(sign, outflow, inflow):
    """Opposed piecewise-constant velocities across x = 0 conflict on every face there."""
    space = _space(order=1)
    ux = space.project_callable(lambda x, y: sign * np.sign(x))
    velocity = (space * space).field((ux.coeffs, space.zeros().coeffs), name="u")
    values = transport_velocity_diagnostics(velocity, backend="host")
    mesh = space.mesh
    midpoints = mesh.node_coords[mesh.edges[mesh.int_edges_inds]].mean(axis=1)
    expected = float(np.mean(np.isclose(midpoints[:, 0], 0.0)))
    assert expected > 0.0
    assert values["velocity_double_outflow_face_fraction"] == pytest.approx(expected if outflow else 0.0)
    assert values["velocity_double_inflow_measure_fraction"] > 0.0 if inflow else (
        values["velocity_double_inflow_measure_fraction"] == 0.0)
    continuous = transport_velocity_diagnostics(
        (space * space).field((space.constant(1.0).coeffs, space.zeros().coeffs)), backend="host")
    assert continuous["velocity_double_outflow_measure_fraction"] == 0.0
    assert continuous["velocity_double_inflow_measure_fraction"] == 0.0


@pytest.mark.parametrize(("method", "reference"), (
    ("basis_at", "basis_at"), ("gradient_basis_at", "gradients_at")))
def test_basis_cache_never_serves_values_for_a_recycled_array_identity(method, reference):
    """A freed array's id() is reused by the next same-shaped array; no stale tables."""
    space = _space(order=3)
    tabulate, exact = getattr(space, method), getattr(space.reference, reference)
    collisions = 0
    for _ in range(50):
        first = np.empty((12, 2))
        first[:] = -0.25
        tabulate(first)
        address = id(first)
        del first
        second = np.empty((12, 2))
        if id(second) != address:
            continue
        collisions += 1
        second[:] = -0.75
        np.testing.assert_array_equal(tabulate(second), exact(second))
    if not collisions:
        pytest.skip("the allocator never reused a freed array address")


def test_basis_cache_reuses_live_arrays_and_forgets_dead_ones():
    space = _space(order=3)
    points = np.array([[-0.5, -0.5], [-0.2, -0.7]])
    first = space.basis_at(points)
    assert space.basis_at(points) is first
    assert len(space._basis_cache) == 1
    del points
    assert len(space._basis_cache) == 0


def _ownership_cases():
    """Host cases always; raw-CUDA cases only when CuPy and PyAMGX import."""
    cases = [("numba", {})]
    try:
        import cupy  # noqa: F401
        import pyamgx  # noqa: F401
    except ImportError:
        return cases
    return cases + [("raw-cuda", {"solver": "amgx"})]


@pytest.mark.parametrize(("backend", "extra"), _ownership_cases())
def test_a_later_solve_never_overwrites_an_earlier_result(backend, extra):
    """Results own their solution arrays; reusable workspaces are copied out."""
    space = _space(order=2, cells=3)
    rho0 = space.project_callable(lambda x, y: np.exp(-4 * (x**2 + y**2)))
    rho1 = space.project_callable(lambda x, y: np.exp(-4 * ((x - .3)**2 + y**2)))
    snapshot = lambda field: np.array(field.coeffs, copy=True)

    def host_trace(result):
        trace = hdg.solution_trace(result, space)
        return np.array(trace.get() if hasattr(trace, "get") else trace, copy=True)

    common = dict(assembly_backend=backend, solver_rtol=1e-10, solver_atol=1e-12, verbose=False, **extra)
    with hdg.DiffusionReactionHDGSolver(space, source=rho0, boundary_condition=0., **common) as poisson:
        first = poisson.solve()
        field, trace = snapshot(first.field), host_trace(first)
        poisson.set_source(rho1).solve()
        np.testing.assert_array_equal(first.field.coeffs, field)
        np.testing.assert_array_equal(host_trace(first), trace)
        velocity = perpendicular_vector_field(first.flux, 0.1)
    one = space.constant(1.0)
    with hdg.AdvectionReactionHDGSolver(space, boundary_mode="zero-flux", **common) as transport:
        first = transport.solve(source=rho0, beta=velocity, reaction=one)
        field = snapshot(first.field)
        transport.solve(source=rho1, beta=velocity, reaction=one)
        np.testing.assert_array_equal(first.field.coeffs, field)
