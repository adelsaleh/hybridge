"""Numba-compiled n-Gamma coefficients (scripts/n_gamma/compiled.py) against the NumPy builders.

Coarse meshes and at most one step per geometry; no study runs.
"""
import numpy as np
import pytest

pytest.importorskip('numba')
pytest.importorskip('gmsh')

from hdgfem import DGSpace  # noqa: E402
from scripts.n_gamma import coefficients as nc  # noqa: E402
from scripts.n_gamma.cases import forcing, forcing_numba, get_case  # noqa: E402
from scripts.n_gamma.compiled import CompiledCoefficients  # noqa: E402
from scripts.n_gamma.stepper import NGammaBDF2Stepper  # noqa: E402

HOST = dict(assembly_backend='numba', solver='direct')


def test_numba_forcing_matches_numpy_forcing():
    rng = np.random.default_rng(4)
    for (x, y), t in zip(rng.uniform(-1., 1., (40, 2)), rng.uniform(0., 1., 40)):
        for name in ('n_e', 'u_e', 'Gamma_e', 'b_p'):
            np.testing.assert_allclose(getattr(forcing_numba, name)(x, y, t), getattr(forcing, name)(x, y, t),
                                       rtol=1e-14, atol=1e-14)
        for variant, (geometry, stationary) in enumerate(forcing_numba.VARIANTS):
            for name in ('S_n', 'S_Gamma'):
                expected = getattr(forcing, name)(x, y, t, geometry=geometry, stationary=stationary)
                np.testing.assert_allclose(getattr(forcing_numba, name)(x, y, t, variant), expected,
                                           rtol=1e-13, atol=1e-13)


def _case_space(name, geometry, order=3):
    case = get_case(name, geometry=geometry)
    return case, DGSpace(case.build_mesh(.25).mesh, order)


@pytest.mark.parametrize('geometry', ['cartesian', 'axisymmetric'])
@pytest.mark.parametrize('name', ['transient_baseline', 'stationary_stress'])
def test_compiled_coefficients_match_numpy_builders(geometry, name):
    case, space = _case_space(name, geometry)
    trace = space.trace_space('legacy-lagrange')
    compiled = CompiledCoefficients(space, case, density_floor=1e-8, sound_speed=1.3)
    n = space.project_callable(lambda a, b: case.density(a, b, .1))
    gamma = space.project_callable(lambda a, b: case.momentum(a, b, .1))
    mapped = space.mapped_quads()
    for law, reference in zip(compiled.diffusion_tensor(.02), nc.diffusion_tensor(.02, case.b_poloidal,
                                                                                    geometry=geometry)):
        np.testing.assert_allclose(law(mapped[..., 0], mapped[..., 1]), reference(mapped[..., 0], mapped[..., 1]),
                                   rtol=1e-14, atol=1e-15)
    for t in (.3, .7):      # the second pass rebinds the compiled templates
        pairs = [(compiled.advection(n, gamma),
                  nc.advection(space, n, gamma, case.b_poloidal, 1e-8, geometry=geometry)),
                 (compiled.density_source(gamma, t), nc.density_source(space, case.density_source_at(t), gamma,
                                                                       geometry=geometry)),
                 (compiled.momentum_source(gamma, n, t), nc.momentum_source(
                     space, case.momentum_source_at(t), gamma, n, case.b_poloidal, 1.3, geometry=geometry))]
        for actual, expected in pairs:
            np.testing.assert_allclose(actual.volume_values(space), expected.volume_values(space),
                                       rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(actual.face_values(space, trace), expected.face_values(space, trace),
                                       rtol=1e-13, atol=1e-13)


def test_templates_compile_once_and_rebind():
    case, space = _case_space('transient_baseline', 'cartesian', order=2)
    compiled = CompiledCoefficients(space, case, density_floor=1e-8)
    n, gamma = space.constant(2.), space.constant(.5)
    first, second = compiled.density_source(n, .1), compiled.density_source(gamma, .2)
    assert second.functions is first.functions and second.time == .2 and second.fields == (gamma,)
    with pytest.raises(ValueError, match='floor'):
        CompiledCoefficients(space, case, density_floor=0.)


@pytest.mark.parametrize('geometry', ['cartesian', 'axisymmetric'])
def test_compiled_step_matches_numpy_step(geometry):
    case, space = _case_space('transient_baseline', geometry, order=2)
    dt = .05
    fields = [(space.project_callable(lambda a, b, t=t: case.density(a, b, t)),
               space.project_callable(lambda a, b, t=t: case.momentum(a, b, t))) for t in (0., dt)]
    results = []
    for compiled in (None, CompiledCoefficients(space, case, density_floor=1e-8)):
        stepper = NGammaBDF2Stepper(
            space, *fields[1], previous_density=fields[0][0], previous_momentum=fields[0][1], dt=dt, time=dt,
            geometry=geometry, b_poloidal=case.b_poloidal, diffusion=forcing.D, viscosity=forcing.MU,
            density_floor=1e-8, source_density=case.density_source_at, source_momentum=case.momentum_source_at,
            boundary_density=case.density_boundary_at, boundary_momentum=case.momentum_boundary_at,
            options=HOST, compiled=compiled)
        results.append(stepper.advance())
    reference, actual = results
    for name in ('density', 'momentum'):
        np.testing.assert_allclose(getattr(actual, name).field.coeffs, getattr(reference, name).field.coeffs,
                                   rtol=1e-11, atol=1e-12)
