from __future__ import annotations

import numpy as np
import pytest

from hdgfem.backends.diff_rea_test7_fused import (
    assemble_test7_tensor_trace_system_eliminated_numba,
    reconstruct_test7_tensor_local_unknowns_numba,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.system import expand_known_dofs, solve_global_system
from hdgfem.solvers.diff_rea import solve_diffusion_reaction_hdg, split_diffusion_unknowns, test7 as diff_rea_test7


pytest.importorskip("numba")


def test_test7_fused_tensor_path_matches_numpy_tensor_solver() -> None:
    mesh = rectangle_mesh(2, 2)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = diff_rea_test7()
    tau = 4.0

    reference = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=tau,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        assembly_backend="numpy",
        verbose=False,
    )

    assembly = assemble_test7_tensor_trace_system_eliminated_numba(tau, space)
    solve = solve_global_system(
        assembly.reduction.rows,
        assembly.reduction.cols,
        assembly.reduction.data,
        assembly.reduction.rhs,
        assembly.reduction.rhs.size,
        solver="direct",
        preconditioner=None,
        scale_system=False,
        raise_on_nonconvergence=True,
        verbose=False,
    )
    trace = expand_known_dofs(solve.x, assembly.reduction)
    local_unknowns = reconstruct_test7_tensor_local_unknowns_numba(trace, tau, space)
    field, flux = split_diffusion_unknowns(local_unknowns, space)

    np.testing.assert_allclose(trace, reference.trace, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(field.coeffs, reference.field.coeffs, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        flux.as_component_first(),
        reference.flux.as_component_first(),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.isfinite(field.l2_error(exact))
