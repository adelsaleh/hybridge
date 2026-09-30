"""Reduced Poisson/recovered drift contracts; no simulation or time integration.

Checks use polynomial tables, local matrices, metadata, and canned callbacks.
No call to a time stepper's advance method or a numerical PDE solver occurs.
"""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import hdgfem.runtime.optional as runtime_optional
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGField, DGSpace, VectorDGField
from hdgfem.core.transfer import project_same_mesh_field
from hdgfem.diagnostics import guiding_center_field_diagnostics
from hdgfem.runtime.precision import REAL_DTYPE
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import _runtime_config, _validate_config
from scripts.guiding_center.time_schemes.si_bdf2 import SIBDF2Stepper
from scripts.guiding_center.time_schemes.recovery import StepRecoveryWork

PRESET = 'euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6'
TOL = 4000 * np.finfo(REAL_DTYPE).eps


def spaces(p=3, basis='dub_orth'):
    mesh = rectangle_mesh(1, 1)
    return (DGSpace(mesh, p, basis_type=basis), DGSpace(mesh, p-1, basis_type=basis))


def result(density_space, poisson_space, speed=4.):
    return SimpleNamespace(
        field=poisson_space.constant(1.),
        flux=VectorDGField((poisson_space.zeros(), poisson_space.constant(-100.))),
        postprocessed_flux=VectorDGField((density_space.zeros(), density_space.constant(-speed))),
        trace=np.ones(poisson_space.mesh.num_edg * (poisson_space.order+1)),
    )


def stepper(density_space, poisson_space, recovered=True):
    value = result(density_space, poisson_space)
    return SIBDF2Stepper(density_space, .1, density_space.constant(2.), value,
        np.zeros(density_space.mesh.num_edg * (density_space.order+1)),
        potential_trace=value.trace, density_boundary=lambda t: None,
        potential_boundary=lambda t: 0., use_postprocessed_flux=recovered)


@pytest.mark.parametrize('p', [1, 2, 6])
@pytest.mark.parametrize('basis', ['dub_orth', 'bernstein'])
def test_source_restriction_preserves_all_poisson_test_moments(p, basis):
    density_space, poisson_space = spaces(p, basis)
    coeffs = np.random.default_rng(42).normal(size=density_space.shape).astype(REAL_DTYPE)
    rho = density_space.field(coeffs)
    projected = project_same_mesh_field(rho, poisson_space)
    points = density_space.quad_data.Krf_quads
    difference = rho.values_at_ref(points) - projected.values_at_ref(points)
    moments = difference @ (density_space.quad_data.Krf_w[:, None] * poisson_space.basis_at(points))
    np.testing.assert_allclose(moments, 0., atol=TOL, rtol=0.)
    assert projected.integral() == pytest.approx(rho.integral(), abs=TOL)
    cached = poisson_space._same_mesh_projection_cache[density_space][0]
    project_same_mesh_field(2*rho, poisson_space)
    assert poisson_space._same_mesh_projection_cache[density_space][0] is cached
    np.testing.assert_array_equal(rho.coeffs, coeffs)


def test_bdf2_uses_recovered_fields_for_startup_and_extrapolation_without_advancing():
    density_space, poisson_space = spaces()
    scheme = stepper(density_space, poisson_space)
    source, beta, scale, _, _ = scheme._transport_data()
    assert source is scheme.density
    assert beta.components[0].space.order == density_space.order
    np.testing.assert_allclose(beta.components[0].values(), .4, atol=TOL)
    scheme.previous_density = density_space.constant(1.)
    scheme.previous_flux = result(density_space, poisson_space, speed=1.).postprocessed_flux
    source, beta, scale, _, _ = scheme._transport_data()
    np.testing.assert_allclose(source.values(), 7./3., atol=TOL)
    np.testing.assert_allclose(beta.components[0].values(), (2*.1/3)*7., atol=TOL)
    assert scheme.time == 0.  # Algebra only, no time step was executed.


def test_recovered_highest_modes_reach_transport_unchanged():
    density_space, poisson_space = spaces(4)
    scheme = stepper(density_space, poisson_space)
    electric = VectorDGField((
        density_space.project_callable(lambda x, y: x**4),
        density_space.project_callable(lambda x, y: -y**4),
    ))
    scheme.poisson_result.postprocessed_flux = electric
    _, beta, _, _, _ = scheme._transport_data()
    np.testing.assert_allclose(beta.components[0].coeffs, -.1*electric.components[1].coeffs, atol=TOL)
    np.testing.assert_allclose(beta.components[1].coeffs, .1*electric.components[0].coeffs, atol=TOL)
    assert beta.components[0].space is density_space


def test_bdf2_requires_recovery_and_rebuilds_recovered_history_without_advancing():
    density_space, poisson_space = spaces()
    scheme = stepper(density_space, poisson_space)
    scheme.previous_density = density_space.constant(1.)
    responses = [result(density_space, poisson_space, 7.), result(density_space, poisson_space, 9.)]
    work = SimpleNamespace(poisson=lambda *args, **kw: (responses.pop(0), np.ones(5)))
    scheme._rebuild_poisson_history(work)
    np.testing.assert_allclose(scheme.previous_flux.components[1].values(), -7., atol=TOL)
    np.testing.assert_allclose(scheme._drift_flux(scheme.poisson_result).components[1].values(), -9., atol=TOL)
    with pytest.raises(ValueError, match='every Poisson solve'):
        scheme._drift_flux(SimpleNamespace(flux=scheme.poisson_result.flux, postprocessed_flux=None))
    assert scheme.time == 0.


def test_poisson_callback_restricts_source_and_preserves_original_checkpoint():
    density_space, poisson_space = spaces()
    scheme = stepper(density_space, poisson_space)
    sources = []
    canned = result(density_space, poisson_space)
    solver = SimpleNamespace(space=poisson_space, set_source=sources.append,
        set_boundary_condition=lambda value: None, solve=lambda **kw: canned)
    work = StepRecoveryWork(scheme, solver)
    rho = density_space.project_callable(lambda x, y: 1+x*x*y)
    returned, trace = work.poisson(rho, 0., 'canned local contract', guess=None)
    assert returned is canned
    assert sources[0].space is poisson_space
    assert work.checkpoint.density.space is density_space
    assert sources[0].integral() == pytest.approx(rho.integral(), abs=TOL)
    assert trace.size == poisson_space.mesh.num_edg * (poisson_space.order+1)


def test_preset_and_order_override_keep_one_degree_separation():
    config = preset_by_key(PRESET)
    _validate_config(config)
    args = build_parser().parse_args([PRESET, '-p', '4', '--dry-run'])
    config = _runtime_config(config, args)
    assert config.order == 4 and config.poisson_order_offset == -1
    assert config.transport_electric_field == 'postprocessed'
    assert config.poisson_hdg_postprocess == 'flux'
    assert config.poisson_flux_postprocess_space == 'RT_projection'
    assert config.poisson_flux_postprocess_every == 0
    _validate_config(config)
    for changes in ({'order': 0}, {'poisson_order_offset': 0}, {'time_scheme': 'si-euler'},
                    {'poisson_hdg_postprocess': 'none'}, {'poisson_flux_postprocess_every': 2}):
        with pytest.raises(ValueError):
            _validate_config(replace(config, **changes))


class NumpyDevice:
    """NumPy stand-in to check device data flow without CUDA or JIT."""
    cuda = SimpleNamespace(Device=lambda device: nullcontext())
    asnumpy = staticmethod(np.asarray)

    def __getattr__(self, name):
        return getattr(np, name)


def test_device_projection_keeps_coefficients_resident(monkeypatch):
    import hdgfem.backends.cupy as backend
    import hdgfem.core.device as core_device
    density_space, poisson_space = spaces(4)
    coefficients = np.random.default_rng(123).normal(size=density_space.shape).astype(REAL_DTYPE)
    expected = project_same_mesh_field(density_space.field(coefficients), poisson_space)
    resident = DGField.from_device_coefficients(density_space, coefficients, device_id=0)
    monkeypatch.setattr(backend, 'require_cupy', lambda: NumpyDevice())
    monkeypatch.setattr(runtime_optional, 'require_cupy', lambda: NumpyDevice())
    monkeypatch.setattr(core_device, 'field_from_cupy_coefficients',
        lambda space, coeffs, device, name: DGField.from_device_coefficients(space, coeffs, device_id=device, name=name))
    projected = project_same_mesh_field(resident, poisson_space)
    assert resident._coeffs is None and projected._coeffs is None
    np.testing.assert_allclose(projected._device_coefficients_for(0), expected.coeffs, atol=TOL, rtol=TOL)


def test_device_diagnostics_accept_distinct_scalar_spaces(monkeypatch):
    import hdgfem.backends.cupy as backend
    import hdgfem.core.device as core_device
    density_space, poisson_space = spaces(3)
    rho = density_space.project_callable(lambda x, y: 2+x*x*y)
    fields = result(density_space, poisson_space)
    kwargs = dict(postprocessed_flux=fields.postprocessed_flux,
        equilibrium_potential=poisson_space.zeros(), equilibrium_density=density_space.constant(1.))
    host = guiding_center_field_diagnostics(rho, fields.field, fields.flux, backend='host', **kwargs)
    monkeypatch.setattr(backend, 'require_cupy', lambda: NumpyDevice())
    monkeypatch.setattr(runtime_optional, 'require_cupy', lambda: NumpyDevice())
    monkeypatch.setattr(core_device, 'as_cupy_space', lambda space, device=None:
        SimpleNamespace(host=space, mesh=space.mesh, quad_data=space.quad_data, device_id=0))
    def coefficients(field, device_space):
        assert field.space is device_space.host
        return field.coeffs
    monkeypatch.setattr(core_device, 'as_cupy_coefficients', coefficients)
    device = guiding_center_field_diagnostics(rho, fields.field, fields.flux, backend='device', **kwargs)
    for key in host.keys() - {'diagnostics_backend'}:
        assert device[key] == pytest.approx(host[key], abs=TOL, rel=TOL)


@pytest.mark.parametrize('density_order', [1, 3, 6])
def test_rt_recovery_has_requested_degree_and_conservative_moments(density_order):
    """Postprocess synthetic local coefficients only; no Poisson solve."""
    from hdgfem.solvers.diffusion_reaction import _postprocess_diffusion_solution
    from test_diffusion_reaction_solver import _assert_rt_flux_constraints
    _, poisson_space = spaces(density_order)
    random = np.random.default_rng(41)
    local = random.normal(size=(poisson_space.mesh.num_tri, 3*poisson_space.el_dof)).astype(REAL_DTYPE)
    trace = random.normal(size=poisson_space.mesh.num_edg*(poisson_space.order+1)).astype(REAL_DTYPE)
    _, flux, _ = _postprocess_diffusion_solution(local, trace, poisson_space, 1., 1., 'flux',
        trace_space=poisson_space.trace_space('legendre-modal'), flux_postprocess_space='RT_projection')
    assert flux.components[0].space.order == density_order
    assert flux.components[1].space.order == density_order
    _assert_rt_flux_constraints(SimpleNamespace(postprocessed_flux=flux, local_unknowns=local, trace=trace),
                               poisson_space, 1., 'legendre-modal')
