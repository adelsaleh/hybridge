"""Continuous Lagrange numbering, Oswald averaging and the compatible drift.

Polynomial tables, canned Poisson results and one explicit transport residual
evaluation only; no linear PDE solve and no time integration.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge.core.conforming import ContinuousLagrangeSpace, lattice_reference_nodes, oswald_average
from hybridge.core.field_ops import perpendicular_vector_field
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace, VectorDGField
from hybridge.diagnostics.guiding_center import transport_velocity_diagnostics
from hybridge.runtime.precision import REAL_DTYPE
from hybridge.transport.residual import UpwindHDGTransportResidual
from scripts.guiding_center.runtime.compatible_drift import (
    CompatibleFluxPoissonSolver,
    compatible_potential_source,
)

TOL = 5e-5 if REAL_DTYPE == np.float32 else 1e-10


def mesh():
    return rectangle_mesh(4, 3, xlim=(-1.0, 1.0), ylim=(-1.0, 0.5))


def polynomial(k):
    """A degree-k polynomial and its gradient."""
    def value(x, y):
        return (1.0 + x + 0.3 * y) ** k - 0.2 * x ** (k - 1) * y

    def gradient(x, y):
        dx = k * (1.0 + x + 0.3 * y) ** (k - 1) - (0.2 * (k - 1) * x ** max(k - 2, 0) * y if k > 1 else 0.0)
        dy = 0.3 * k * (1.0 + x + 0.3 * y) ** (k - 1) - 0.2 * x ** (k - 1)
        return dx, dy

    return value, gradient


@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_lattice_and_numbering_counts(k):
    nodes = lattice_reference_nodes(k)
    assert nodes.shape == ((k + 1) * (k + 2) // 2, 2)
    assert np.allclose(nodes[:3], [[-1, -1], [1, -1], [-1, 1]])
    m = mesh()
    space = ContinuousLagrangeSpace.on(m, k)
    assert space is ContinuousLagrangeSpace.on(m, k)
    expected = m.node_coords.shape[0] + m.num_edg * (k - 1) + m.num_tri * (k - 1) * (k - 2) // 2
    assert space.num_dofs == expected
    # Shared nodes have one physical position: re-evaluating every element's
    # lattice reproduces the global coordinates.
    physical = np.einsum("kij,nj->kni", m.aff_mats, space.reference_nodes) + m.aff_vecs[:, None, :]
    assert np.abs(physical - space.dof_coords[space.cell_dofs]).max() < 1e-12
    x, y = space.dof_coords[space.boundary_dofs].T
    on_wall = np.isclose(np.abs(x), 1.0) | np.isclose(y, -1.0) | np.isclose(y, 0.5)
    assert np.all(on_wall)
    x, y = space.dof_coords[space.free_dofs].T
    assert not np.any(np.isclose(np.abs(x), 1.0) | np.isclose(y, -1.0) | np.isclose(y, 0.5))


@pytest.mark.parametrize("basis", ["dub_orth", "bernstein"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_oswald_reproduces_continuous_data_and_its_gradient(basis, k):
    m = mesh()
    value, gradient = polynomial(k)
    field = DGSpace(m, k, basis_type=basis).project_callable(value)
    continuous = oswald_average(field, boundary_values=None)
    space = continuous.space
    assert np.abs(continuous.dofs - value(*space.dof_coords.T)).max() < TOL * 100
    assert np.abs(continuous.to_dg_field(field.space).coeffs - field.coeffs).max() < TOL * 100
    target = DGSpace(m, max(k - 1, 1), basis_type=basis)
    flux = continuous.flux_field(target)
    expected_x = target.project_callable(lambda x, y: -gradient(x, y)[0])
    expected_y = target.project_callable(lambda x, y: -gradient(x, y)[1])
    assert np.abs(flux.components[0].coeffs - expected_x.coeffs).max() < TOL * 1000
    assert np.abs(flux.components[1].coeffs - expected_y.coeffs).max() < TOL * 1000


def random_compatible_drift(basis="dub_orth", p=3, seed=0):
    m = mesh()
    density_space = DGSpace(m, p, basis_type=basis)
    potential_space = DGSpace(m, p + 1, basis_type=basis)
    rng = np.random.default_rng(seed)
    potential = potential_space.field(rng.standard_normal(potential_space.shape).astype(REAL_DTYPE))
    continuous = oswald_average(potential, boundary_values=0.0)
    return density_space, perpendicular_vector_field(continuous.flux_field(density_space), 1.0)


@pytest.mark.parametrize("basis", ["dub_orth", "bernstein"])
def test_compatible_drift_is_hdiv_divergence_free_and_wall_tangent(basis):
    density_space, drift = random_compatible_drift(basis)
    metrics = transport_velocity_diagnostics(drift, backend="host")
    scale = metrics["velocity_speed_linf"]
    assert scale > 1.0
    for key in ("boundary_normal_linf", "normal_jump_linf", "divergence_linf"):
        assert metrics[f"velocity_{key}"] < TOL * scale
    assert metrics["velocity_double_outflow_face_fraction"] == 0.0
    assert metrics["velocity_double_inflow_measure_fraction"] == 0.0


def test_boundary_values_follow_dirichlet_data():
    m = mesh()
    field = DGSpace(m, 3, basis_type="dub_orth").zeros()
    data = lambda x, y: 2.0 * x - y ** 2
    continuous = oswald_average(field, boundary_values=data)
    space = continuous.space
    x, y = space.dof_coords[space.boundary_dofs].T
    assert np.abs(continuous.dofs[space.boundary_dofs] - data(x, y)).max() < 1e-14
    assert np.all(continuous.dofs[space.free_dofs] == 0.0)


def test_constant_state_is_steady_with_the_compatible_drift():
    density_space, drift = random_compatible_drift("dub_orth", p=3, seed=4)
    residual = UpwindHDGTransportResidual(density_space, boundary_mode="zero-flux",
                                          advection_stabilization="conflict-averaged-upwind")
    constant = density_space.constant(1.7)
    rhs, _ = residual.evaluate(constant, drift)
    assert np.abs(rhs.coeffs).max() < TOL * transport_velocity_diagnostics(drift)["velocity_speed_linf"]
    # A drift with face jumps does not leave constants steady.
    rng = np.random.default_rng(1)
    raw = VectorDGField(tuple(density_space.field(rng.standard_normal(density_space.shape)) for _ in range(2)))
    raw_rhs, _ = residual.evaluate(constant, raw)
    assert np.abs(raw_rhs.coeffs).max() > 1e-3


def test_device_oswald_matches_host_without_host_staging():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    from hybridge.core.device import field_from_cupy_coefficients

    m = mesh()
    density_space = DGSpace(m, 4, basis_type="dub_orth")
    potential_space = DGSpace(m, 5, basis_type="dub_orth")
    coefficients = np.random.default_rng(3).standard_normal(potential_space.shape)
    data = lambda x, y: 0.1 * x - y
    host = oswald_average(potential_space.field(coefficients), boundary_values=data)
    device = oswald_average(field_from_cupy_coefficients(potential_space, cp.asarray(coefficients)),
                            boundary_values=data)
    assert isinstance(device.dofs, cp.ndarray)
    assert np.abs(host.dofs - cp.asnumpy(device.dofs)).max() < TOL
    host_flux = host.flux_field(density_space)
    device_flux = device.flux_field(density_space)
    assert not device_flux.components[0].coefficients_materialized
    for a, b in zip(host_flux.components, device_flux.components):
        assert np.abs(a.coeffs - b.coeffs).max() < TOL * 100


@dataclass(frozen=True)
class _CannedResult:
    """Frozen like ``DiffusionReactionResult``: the wrapper must use ``replace``."""

    field: object
    flux: object
    postprocessed_field: object = None


class _CannedPoissonSolver:
    """Minimal reusable-solver stand-in: records options, returns a canned result."""

    def __init__(self, space, result, hdg_postprocess="none"):
        self.space = space
        self.result = result
        self.boundary_condition = lambda x, y: np.zeros_like(x)
        self.options = SimpleNamespace(hdg_postprocess=hdg_postprocess, stabilization=1.0)
        self.calls = []

    def solve(self, *, initial_guess=None, postprocess_overrides=None):
        self.calls.append(dict(postprocess_overrides or {}))
        return self.result

    def with_options(self, **overrides):
        for key, value in overrides.items():
            setattr(self.options, key, value)
        return self


@pytest.mark.parametrize("configured,requested", [("none", "primal"), ("flux", "both"), ("both", "both")])
def test_wrapper_requests_primal_postprocessing_and_replaces_the_flux(configured, requested):
    m = mesh()
    density_space = DGSpace(m, 3, basis_type="dub_orth")
    potential_space = DGSpace(m, 4, basis_type="dub_orth")
    rng = np.random.default_rng(7)
    raw_flux = VectorDGField(tuple(density_space.field(rng.standard_normal(density_space.shape))
                                   for _ in range(2)))
    canned = _CannedResult(field=density_space.field(rng.standard_normal(density_space.shape)),
                           flux=raw_flux,
                           postprocessed_field=potential_space.field(rng.standard_normal(potential_space.shape)))
    solver = _CannedPoissonSolver(density_space, canned, hdg_postprocess=configured)
    wrapped = CompatibleFluxPoissonSolver(solver, density_space, potential="postprocessed")
    result = wrapped.solve(initial_guess=None)
    assert solver.calls[-1]["hdg_postprocess"] == requested
    assert result.raw_flux is raw_flux
    drift = perpendicular_vector_field(result.flux, 1.0)
    metrics = transport_velocity_diagnostics(drift, backend="host")
    assert metrics["velocity_normal_jump_linf"] < TOL * metrics["velocity_speed_linf"]
    # Attribute writes and option updates reach the wrapped solver (tau recovery).
    wrapped.with_options(stabilization=8.0)
    assert solver.options.stabilization == 8.0
    wrapped.boundary_condition = 0.0
    assert solver.boundary_condition == 0.0


def test_potential_source_resolution():
    assert compatible_potential_source("auto", "numba") == "postprocessed"
    assert compatible_potential_source("auto", "raw-cuda") == "postprocessed"
    assert compatible_potential_source("primal", "numba") == "primal"
    # The raw-CUDA Poisson postprocesses the potential on the device.
    assert compatible_potential_source("postprocessed", "raw-cuda") == "postprocessed"
    with pytest.raises(ValueError, match="compatible potential"):
        compatible_potential_source("exact", "raw-cuda")
