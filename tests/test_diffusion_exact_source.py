"""`DiffusionReactionHDGSolver` takes an exact source as values at its own quadrature points.

The NumPy callable path samples the function on the solver's quadrature, so it is the
reference. Numba, CuPy and raw CUDA take a `QuadratureScalar` and must reproduce it. Device
checks run only where a CUDA device is available.
"""
from __future__ import annotations

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.quadrature_coefficients import QuadratureScalar
from hybridge.hdg.condensation import block_source_moments, source_moments
from hybridge.runtime.precision import REAL_DTYPE
from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGOptions, DiffusionReactionHDGSolver

TOLERANCE = 2e-10 if REAL_DTYPE == np.float64 else 1e-3
# The device solves are iterative and stop near this relative residual.
DEVICE_TOLERANCE = 1e-7 if REAL_DTYPE == np.float64 else 1e-2


def _law(x, y):
    """A Gaussian not representable at degree p, so that the quadrature matters."""
    return np.exp(-40.0 * ((x - 0.4) ** 2 + (y - 0.6) ** 2))


def _device_law(x, y):
    import cupy as cp

    return cp.exp(-40.0 * ((x - 0.4) ** 2 + (y - 0.6) ** 2))


def _space(order):
    return DGSpace(rectangle_mesh(4, 3), order, basis_type="dub_orth")


def _solve(space, source, backend, **options):
    options = DiffusionReactionHDGOptions(
        stabilization=1.0, assembly_backend=backend, boundary_mode="eliminate", verbose=False, **options)
    solver = DiffusionReactionHDGSolver(
        space, source=source, reaction=space.zeros(name="reaction_h"), boundary_condition=0.0, options=options)
    result = solver.solve()
    return result


def _coefficients(result):
    """Return the solution coefficients on the host, downloading a device-resident field."""
    return np.asarray(result.field.coeffs)


@pytest.mark.parametrize("order", [2, 3, 6])
def test_moments_equal_those_of_the_callable_on_the_same_quadrature(order):
    space = _space(order)
    scalar = QuadratureScalar.from_callable(_law, space)
    np.testing.assert_array_equal(source_moments(scalar, space), source_moments(_law, space))
    blocks = block_source_moments(scalar, space, num_blocks=3)
    np.testing.assert_array_equal(blocks, block_source_moments(_law, space, num_blocks=3))


def test_values_from_another_quadrature_are_rejected():
    space = _space(2)
    other = DGSpace(space.mesh, 2, basis_type="dub_orth", volume_quad_1d=8)
    scalar = QuadratureScalar.from_callable(_law, other)
    with pytest.raises(ValueError, match="another volume quadrature"):
        source_moments(scalar, space)


@pytest.mark.parametrize("order", [2, 3, 6])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_host_solves_match_the_numpy_callable(order, backend):
    space = _space(order)
    reference = _coefficients(_solve(space, _law, "numpy"))
    solved = _coefficients(_solve(space, QuadratureScalar.from_callable(_law, space), backend))
    assert np.max(np.abs(solved - reference)) <= TOLERANCE * max(np.max(np.abs(reference)), 1.0)


def test_raised_degree_uses_its_own_quadrature():
    density, raised = _space(3), _space(5)
    reference = _coefficients(_solve(raised, _law, "numpy"))
    solved = _coefficients(_solve(raised, QuadratureScalar.from_callable(_law, raised), "numba"))
    assert np.max(np.abs(solved - reference)) <= TOLERANCE * max(np.max(np.abs(reference)), 1.0)
    with pytest.raises(ValueError):
        _solve(raised, QuadratureScalar.from_callable(_law, density), "numba")


def _require_device():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001 - no usable CUDA runtime
        count = 0
    if count == 0:
        pytest.skip("No CUDA device")
    return cp


@pytest.mark.parametrize("order", [2, 3, 6])
@pytest.mark.parametrize("backend", ["cupy", "raw-cuda"])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
def test_device_solves_match_the_numpy_callable(order, backend, trace_basis):
    _require_device()
    space = _space(order)
    reference = _coefficients(_solve(space, _law, "numpy", trace_basis=trace_basis))
    source = QuadratureScalar.from_callable(_device_law, space, device=True)
    assert source.on_device
    solved = _coefficients(_solve(space, source, backend, trace_basis=trace_basis, solver="amgx",
                                  solver_rtol=1e-12, solver_atol=1e-13))
    assert np.max(np.abs(solved - reference)) <= DEVICE_TOLERANCE * max(np.max(np.abs(reference)), 1.0)
