"""Condensed CG Poisson solver (E9: CG against HDG). Host checks use the Numba setup; AMGX is optional."""
import math

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.solvers.cg_poisson import CGPoissonOptions, CGPoissonSolver


def _space(n, order):
    return DGSpace(rectangle_mesh(n, n, xlim=(0.0, 1.0), ylim=(0.0, 1.0)), order)


def test_cubic_solution_with_dirichlet_data_is_exact():
    """phi = x^3 + x y^2 - y is in S_h^3 and -lap phi = -8x: assembly, condensation and data are exact."""
    space = _space(3, 2)

    def phi(x, y):
        return x ** 3 + x * y * y - y

    solver = CGPoissonSolver(space, boundary_condition=phi, options=CGPoissonOptions(degree=3, solver="direct"))
    result = solver.solve(space.project_callable(lambda x, y: -8.0 * x))
    coords = result.potential.space.dof_coords
    np.testing.assert_allclose(result.potential.dofs, phi(coords[:, 0], coords[:, 1]), atol=1e-11)
    qx, qy = result.flux.components  # q = -grad phi, exactly in DG(2)
    assert space.linf_diff(qx, lambda x, y: -(3.0 * x * x + y * y)) < 1e-10
    assert space.linf_diff(qy, lambda x, y: -(2.0 * x * y - 1.0)) < 1e-10


@pytest.mark.parametrize("degree", [2, 3])
def test_convergence_orders(degree):
    """Smooth phi with zero data: max-norm orders about k+1 for phi and k for q."""
    def phi(x, y):
        return np.sin(math.pi * x) * np.sin(math.pi * y)

    errors_phi, errors_q = [], []
    for n in (4, 8):
        space = _space(n, degree + 1)  # accurate source projection
        solver = CGPoissonSolver(space, options=CGPoissonOptions(degree=degree, solver="direct"),
                                 flux_space=DGSpace(space.mesh, degree - 1))
        result = solver.solve(space.project_callable(lambda x, y: 2.0 * math.pi ** 2 * phi(x, y)))
        errors_phi.append(space.linf_diff(result.potential.to_dg_field(space), phi))
        errors_q.append(solver.flux_space.linf_diff(
            result.flux.components[0], lambda x, y: -math.pi * np.cos(math.pi * x) * np.sin(math.pi * y)))
    assert math.log2(errors_phi[0] / errors_phi[1]) > degree + 0.6
    assert math.log2(errors_q[0] / errors_q[1]) > degree - 0.4


def test_pypardiso_matches_direct():
    space = _space(6, 3)
    source = space.project_callable(lambda x, y: np.exp(x) * np.cos(2.0 * y))
    potentials = []
    for name in ("direct", "pypardiso"):
        solver = CGPoissonSolver(space, options=CGPoissonOptions(solver=name))
        potentials.append(solver.solve(source).potential.dofs)
        solver.close()
    np.testing.assert_allclose(potentials[1], potentials[0], rtol=1e-10, atol=1e-13)


def test_amgx_matches_pypardiso():
    pytest.importorskip("cupy")
    pytest.importorskip("pyamgx")
    space = _space(8, 3)
    source = space.project_callable(lambda x, y: np.exp(x) * np.cos(2.0 * y))
    host = CGPoissonSolver(space, options=CGPoissonOptions(solver="pypardiso"))
    reference = host.solve(source).potential.dofs
    device = CGPoissonSolver(space, options=CGPoissonOptions(solver="amgx", rtol=1e-12))
    result = device.solve(source)
    np.testing.assert_allclose(np.asarray(result.potential.dofs.get()), reference, rtol=1e-8, atol=1e-10)
    assert result.iterations is not None and result.iterations > 0
    device.close()
    host.close()
