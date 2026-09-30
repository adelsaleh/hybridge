"""Numba cfunc pointwise coefficients and laws: values, guidance errors, rebinding, solves and single kernel
signatures."""
from math import pi
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

pytest.importorskip("numba")

from hdgfem import (  # noqa: E402
    DGSpace,
    ElementCoefficient,
    pointwise_coefficient,
    pointwise_law,
    rectangle_mesh,
    solve_advection_diffusion_reaction_hdg,
)
from hdgfem.core.element_coefficients import physical_points  # noqa: E402

GLOBAL_SCALE = 4.0


@pytest.fixture(scope="module")
def space():
    return DGSpace(rectangle_mesh(6, 5), 3, basis_type="dub_orth")


def _xy(space):
    mapped = space.mesh.map_reference_points(space.quad_data.Krf_quads)
    return mapped[..., 0], mapped[..., 1]


def test_xyt_values_and_time_binding(space):
    x, y = _xy(space)
    source = pointwise_coefficient(lambda x, y, t: np.sin(pi*x)*np.cos(y + t), space, time=.3, name="source")
    np.testing.assert_allclose(source.volume_values(space), np.sin(pi*x)*np.cos(y + .3), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(source.at_time(1.).volume_values(space), np.sin(pi*x)*np.cos(y + 1.), atol=1e-14)
    np.testing.assert_allclose(source.volume_values(space, t=2.), np.sin(pi*x)*np.cos(y + 2.), atol=1e-14)


def test_field_gradient_and_parameter_values(space):
    density = space.project_callable(lambda a, b: 2. + .1*a*b)
    momentum = space.project_callable(lambda a, b: .3 + a)
    velocity = pointwise_coefficient(
        (lambda x, y, t, v: v[1]/max(v[0], v[4]), lambda x, y, t, v: v[2] + v[3]),
        space, fields=(density, momentum), gradients=(density,), params=(1e-8,), name="velocity")
    values = velocity.volume_values(space)
    d_x, d_y = density.grad_values()
    assert values.shape == density.values().shape + (2,)
    np.testing.assert_allclose(values[..., 0], momentum.values()/density.values(), rtol=1e-13)
    np.testing.assert_allclose(values[..., 1], d_x + d_y, rtol=1e-12, atol=1e-12)
    faces = velocity.face_values(space, space.trace_space("legacy-lagrange"))
    assert faces.shape[:2] == (space.mesh.num_tri, 3) and faces.shape[-1] == 2


def test_non_numpy_code_asks_to_project_first(space):
    scipy_special = pytest.importorskip("scipy.special")
    with pytest.raises(TypeError, match=r"cannot be compiled by Numba.*project_callable"):
        pointwise_coefficient(lambda x, y, t: scipy_special.j0(x), space, name="bessel")
    with pytest.raises(TypeError, match="project_callable"):
        pointwise_coefficient(lambda x, y, t: x + float(object() is None), space, name="python object")


def test_globals_are_rejected_closures_and_constants_are_not(space):
    x, _ = _xy(space)
    with pytest.raises(ValueError, match="GLOBAL_SCALE=4.0"):
        pointwise_coefficient(lambda x, y, t: GLOBAL_SCALE*x, space)

    def scaled(scale):
        return pointwise_coefficient(lambda x, y, t: scale*x*pi/pi, space)

    np.testing.assert_allclose(scaled(2.).volume_values(space), 2.*x, rtol=1e-14)
    np.testing.assert_allclose(scaled(3.).volume_values(space), 3.*x, rtol=1e-14)


def test_signature_errors_and_device_refusal(space):
    with pytest.raises(TypeError, match=r"\(x, y, t\) or \(x, y, t, v\)"):
        pointwise_coefficient(lambda x, y: x, space)
    with pytest.raises(TypeError, match="must take"):
        pointwise_coefficient(lambda x, y, t: x, space, params=(1.,))
    coefficient = pointwise_coefficient(lambda x, y, t: x, space)
    with pytest.raises(TypeError, match="host Numba kernels"):
        coefficient.volume_values(space, xp=SimpleNamespace(asarray=np.asarray))


def test_functions_outside_files_compile_with_a_warning(space):
    namespace = {}
    exec("f = lambda x, y, t: x + y", {}, namespace)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coefficient = pointwise_coefficient(namespace["f"], space)
    assert any("cannot cache" in str(item.message) for item in caught)
    x, y = _xy(space)
    np.testing.assert_allclose(coefficient.volume_values(space), x + y, rtol=1e-14)


def _element(space, function, components=1):
    return ElementCoefficient(
        lambda points, *, xp, t=None: function(*np.moveaxis(physical_points(space, points), -1, 0)),
        space.mesh, components)


def test_numba_solve_matches_numpy_element_coefficients_with_one_kernel_signature(space):
    from hdgfem.kernels.advection_diffusion_reaction_fused import (
        assemble_projected_adr_trace_system_eliminated_kernel, reconstruct_projected_adr_local_unknowns_kernel)
    from hdgfem.kernels.pointwise import sample_pointwise_xyt_kernel

    beta = pointwise_coefficient((lambda x, y, t: .8 + .1*y, lambda x, y, t: -.25 + .07*x), space)
    source = pointwise_coefficient(lambda x, y, t: np.sin(pi*x)*np.cos(y), space)
    reaction = pointwise_coefficient(lambda x, y, t: .4 + .1*x, space)
    expected = (_element(space, lambda x, y: np.sin(np.pi*x)*np.cos(y)),
                _element(space, lambda x, y: np.stack((.8 + .1*y, -.25 + .07*x), -1), 2),
                _element(space, lambda x, y: .4 + .1*x))
    boundary = lambda x, y: .2 + x - .3*y
    kernels = (assemble_projected_adr_trace_system_eliminated_kernel, reconstruct_projected_adr_local_unknowns_kernel,
               sample_pointwise_xyt_kernel)
    before = [set(kernel.signatures) for kernel in kernels]
    for diffusion, columns in ((.3, True), (np.array([[1., .3], [.3, .5]]), False)):
        options = dict(assembly_backend="numba", solver="direct", diffusion=diffusion,
                       numba_reuse_local_columns=columns, verbose=False)
        actual = solve_advection_diffusion_reaction_hdg(source, beta, reaction, boundary, space, **options)
        reference = solve_advection_diffusion_reaction_hdg(*expected, boundary, space, **options)
        np.testing.assert_allclose(actual.field.coeffs, reference.field.coeffs, rtol=1e-13, atol=1e-13)
    # Different coefficient functions and diffusion kinds reuse one compilation of each kernel.
    for kernel, known in zip(kernels, before):
        assert len(set(kernel.signatures) - known) <= 1, kernel


def test_laws_evaluate_any_shape_with_params_and_time():
    law = pointwise_law(lambda x, y, t, v: v[0]*x + np.sin(y) + t, params=(2.,), time=.5, name="law")
    x = np.linspace(0., 1., 24).reshape(2, 3, 4)
    np.testing.assert_allclose(law(x, x[::-1]), 2*x + np.sin(x[::-1]) + .5, rtol=1e-15)
    assert law(1., 2.).shape == () and law(np.arange(3.), 1.).shape == (3,)
    assert law(np.empty((0, 2)), 0.).shape == (0, 2)
    np.testing.assert_allclose(law.at_time(1.)(x, x), 2*x + np.sin(x) + 1., rtol=1e-15)
    xyt = pointwise_law(lambda x, y, t: x*y)
    np.testing.assert_allclose(xyt(x, x), x*x, rtol=1e-15)
    with pytest.raises(TypeError, match="must take"):
        pointwise_law(lambda x, y, t: x, params=(1.,))
    with pytest.raises(TypeError, match="project_callable"):
        pointwise_law(lambda x, y, t: float(object() is None))


def test_laws_are_diffusion_tensor_components(space):
    from hdgfem.assembly.diffusion_coefficients import prepare_diffusion, sample_diffusion_tensor

    laws = (pointwise_law(lambda x, y, t: 1. + .1*x), pointwise_law(lambda x, y, t: .2*y),
            pointwise_law(lambda x, y, t: 2. + x*y))
    reference = (lambda x, y: 1. + .1*x, lambda x, y: .2*y, lambda x, y: 2. + x*y)
    np.testing.assert_allclose(sample_diffusion_tensor(laws, space), sample_diffusion_tensor(reference, space),
                               rtol=1e-15)
    trace = space.trace_space("legacy-lagrange")
    np.testing.assert_allclose(sample_diffusion_tensor(laws, space, on_faces=True, trace_space=trace),
                               sample_diffusion_tensor(reference, space, on_faces=True, trace_space=trace), rtol=1e-15)
    assert prepare_diffusion(laws, space).counts["variable-symmetric"] == space.mesh.num_tri


def test_bind_swaps_point_data_without_recompiling(space):
    first, second = space.project_callable(lambda a, b: 1. + a), space.project_callable(lambda a, b: 2. + b)
    template = pointwise_coefficient(lambda x, y, t, v: v[0]*v[1] + t, space, fields=(first,), params=(3.,))
    compiled = template.functions
    bound = template.bind(fields=(second,), params=(4.,), time=.5)
    assert bound.functions is compiled
    np.testing.assert_allclose(bound.volume_values(space), 4*second.values() + .5, rtol=1e-14)
    np.testing.assert_allclose(template.volume_values(space), 3*first.values(), rtol=1e-14)
    with pytest.raises(ValueError, match="keep the number"):
        template.bind(params=(1., 2.))
    other = DGSpace(rectangle_mesh(2, 2), 1).zeros()
    with pytest.raises(ValueError, match="same mesh|coefficient's mesh"):
        template.bind(fields=(other,))
