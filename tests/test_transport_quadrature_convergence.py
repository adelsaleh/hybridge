"""Stationary HDG transport with exact coefficients at the quadrature points: convergence.

Solves ``beta . grad(u) + r u = f`` on (-1, 1)^2 with a divergence-free advection
``beta = (-d_y psi, d_x psi)`` (no stagnation point), Dirichlet data from the exact
solution (eliminated) and the advection, reaction and source given as exact values at the
quadrature points (:class:`~hybridge.core.quadrature_coefficients.QuadratureVectorField`,
:class:`~hybridge.core.quadrature_coefficients.QuadratureScalar`). Errors are L2 errors
against the exact solution with 2p + 2 collapsed-Gauss points per direction. References:
the same solve with the advection projected onto DG(p + 2), and (GPU) raw-CUDA fused assembly
and reconstruction.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.quadrature_coefficients import QuadratureVectorField, QuadratureScalar
from hybridge.core.space import VectorDGField
from hybridge.diagnostics.errors import evaluate_scalar_error
from hybridge.solvers.advection_reaction import AdvectionReactionHDGSolver

pytest.importorskip("numba")

CUDA = os.environ.get("HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS") == "1"
MESHES = (4, 8, 16)


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not (CUDA and _cuda_available()),
                               reason="set HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1 on a CUDA host")


def _laws(xp):
    """Exact solution, advection components, reaction and source as ``xp`` callables."""
    def u(x, y):
        return xp.sin(2.0 * x + 1.0) * xp.cos(1.5 * y) + 0.5 * x * y

    def bx(x, y):  # -d_y psi, psi = y (1.5 + 0.3 sin x) + 0.2 cos y
        return -(1.5 + 0.3 * xp.sin(x)) + 0.2 * xp.sin(y)

    def by(x, y):  # d_x psi
        return 0.3 * y * xp.cos(x)

    def reaction(x, y):
        return 1.0 + 0.3 * xp.cos(x * y)

    def source(x, y):
        ux = 2.0 * xp.cos(2.0 * x + 1.0) * xp.cos(1.5 * y) + 0.5 * y
        uy = -1.5 * xp.sin(2.0 * x + 1.0) * xp.sin(1.5 * y) + 0.5 * x
        return bx(x, y) * ux + by(x, y) * uy + reaction(x, y) * u(x, y)

    return u, bx, by, reaction, source


def _error(field, order):
    exact = _laws(np)[0]
    return float(evaluate_scalar_error(field, exact, volume_quad_1d=2 * order + 2, backend="host").metrics.l2)


def _solve(n, order, *, advection="pointwise", backend="numba"):
    """Solve on the n x n mesh; ``advection`` is 'pointwise' or 'projected' (DG(p + 2))."""
    device = backend == "raw-cuda"
    if device:
        import cupy as xp
    else:
        xp = np
    u, bx, by, reaction, source = _laws(xp)
    space = DGSpace(rectangle_mesh(n, n), order, basis_type="dub_orth")
    trace = space.trace_space("legacy-lagrange")
    if advection == "pointwise":
        beta = QuadratureVectorField.from_callable(lambda x, y: (bx(x, y), by(x, y)), space, trace_space=trace,
                                             device=device)
    else:
        _, host_bx, host_by, _, _ = _laws(np)
        beta = VectorDGField((host_bx, host_by), DGSpace(space.mesh, order + 2, basis_type="dub_orth"))
    # Direct trace solves on both sides: the device comparison sees the assembly and reconstruction alone.
    options = dict(solver="direct", raw_local_assembly="fused") if device else dict(solver="direct")
    solver = AdvectionReactionHDGSolver(
        space, assembly_backend=backend, source=QuadratureScalar.from_callable(source, space, device=device),
        beta=beta, reaction=QuadratureScalar.from_callable(reaction, space, device=device),
        boundary_condition=_laws(np)[0], boundary_mode="eliminate", trace_basis="legacy-lagrange", verbose=0,
        **options)
    return _error(solver.solve().field, order)


@pytest.mark.parametrize("order", [2, 3])
def test_exact_coefficients_converge_like_the_projected_drift(order):
    pointwise = np.array([_solve(n, order) for n in MESHES])
    projected = np.array([_solve(n, order, advection="projected") for n in MESHES])
    rates = np.log2(pointwise[:-1] / pointwise[1:])
    assert rates[-1] > order + 0.6, (pointwise, rates)
    # The DG(p + 2) advection differs from the exact one by O(h^(p+3)): the density errors agree.
    np.testing.assert_allclose(pointwise, projected, rtol=0.02)


@cuda_only
@pytest.mark.parametrize("order", [2, 3])
def test_device_exact_coefficients_match_host(order):
    host = np.array([_solve(n, order) for n in MESHES])
    device = np.array([_solve(n, order, backend="raw-cuda") for n in MESHES])
    np.testing.assert_allclose(device, host, rtol=1e-6)
