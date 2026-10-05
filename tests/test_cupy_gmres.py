from __future__ import annotations

import numpy as np
import pytest

from hybridge.linalg.face_dense import face_dense_relative_residual
from hybridge.mixed.face_dense import normalize_penalty_rows
from hybridge.runtime.optional import require_cupy_device
from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
from hybridge.linalg.gpu.gmres import (
    _apply_previous_givens,
    _back_substitute_upper,
    _compute_givens,
    _resolve_orthogonalization,
    CuPyGMRESWorkspace,
    CuPyRestartedGMRESSolver,
    CuPyVectorBLAS,
    restarted_gmres_cupy,
)
from hybridge.linalg.gpu.preconditioners import CuPyFaceBlockJacobiPreconditioner
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.solvers.diffusion_face_dense import solve_diffusion_face_dense_direct
from scripts.diffusion_reaction.cases import quadratic_poisson_case


_BOUNDARY_PENALTY = 1.0e6


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def _small_face_system(boundary_mode: str, *, nx: int = 3, ny: int = 3):
    space = DGSpace(
        rectangle_mesh(nx, ny),
        2,
        basis_type="dub_orth",
    )
    diffusion, reaction, source, exact = quadratic_poisson_case()
    direct = solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode=boundary_mode,
        boundary_penalty=_BOUNDARY_PENALTY,
    )
    return direct


def test_resolve_orthogonalization_preserves_legacy_behavior() -> None:
    assert _resolve_orthogonalization(
        orthogonalization=None,
        reorthogonalize=False,
    ) == "mgs"
    assert _resolve_orthogonalization(
        orthogonalization=None,
        reorthogonalize=True,
    ) == "mgs2"
    for mode in ("mgs", "mgs2", "cgs", "cgs2"):
        assert _resolve_orthogonalization(
            orthogonalization=mode,
            reorthogonalize=False,
        ) == mode


def test_resolve_orthogonalization_rejects_ambiguous_or_invalid_input() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_orthogonalization(
            orthogonalization="cgs2",
            reorthogonalize=True,
        )
    with pytest.raises(ValueError, match="must be one of"):
        _resolve_orthogonalization(
            orthogonalization="invalid",  # type: ignore[arg-type]
            reorthogonalize=False,
        )


def test_compute_givens_annihilates_lower_entry() -> None:
    """
    Givens rotation :
    [c s ; -s c][a b] = [r 0]
    The test checks:
    - The rotated lower value is zero.
    - The upper value is the expected radius (r).
    - (c^2+s^2=1), meaning it is a proper rotation.
    It also covers special cases such as (0, 0) and an already-zero lower entry.
    """
    for a, b in [(3.0, 4.0), (-2.5, 7.0), (0.0, 0.0), (5.0, 0.0)]:
        cosine, sine, radius = _compute_givens(a, b)
        rotated = np.array(
            [
                cosine * a + sine * b,
                -sine * a + cosine * b,
            ]
        )
        np.testing.assert_allclose(rotated, [radius, 0.0], atol=2.0e-15)
        np.testing.assert_allclose(cosine**2 + sine**2, 1.0, atol=2.0e-15)


def test_apply_previous_givens_matches_explicit_rotations() -> None:
    """
    Whenever Arnoldi adds a new Hessenberg column, all Givens rotations computed for earlier columns must also be applied to this new column.
    The test computes those rotations manually, applies the implementation, and confirms that both results agree.
    """
    hessenberg = np.zeros((5, 4))
    hessenberg[:4, 3] = np.array([1.0, -2.0, 3.0, 4.0])
    cosines = np.array([0.8, 0.6, 12.0 / 13.0, 0.0])
    sines = np.array([0.6, 0.8, 5.0 / 13.0, 0.0])

    expected = hessenberg[:, 3].copy()
    for row in range(3):
        upper, lower = expected[row], expected[row + 1]
        expected[row] = cosines[row] * upper + sines[row] * lower
        expected[row + 1] = -sines[row] * upper + cosines[row] * lower

    _apply_previous_givens(hessenberg, cosines, sines, 3)
    np.testing.assert_allclose(hessenberg[:, 3], expected, atol=2.0e-15)


def test_back_substitution_matches_numpy_solve() -> None:
    """
    After the Givens rotations, GMRES needs to solve [ Ry=g, ], (R) upper triangular.
    The test creates a random upper-triangular matrix and compares the custom back-substitution result with numpy.linalg.solve
    """
    rng = np.random.default_rng(932)
    upper = np.triu(rng.standard_normal((8, 8)))
    upper[np.diag_indices_from(upper)] += 4.0
    rhs = rng.standard_normal(8)

    result = _back_substitute_upper(
        upper,
        rhs,
        singular_tolerance=1.0e-14,
    )
    expected = np.linalg.solve(upper, rhs)
    np.testing.assert_allclose(result, expected, rtol=2.0e-15, atol=2.0e-15)


def test_back_substitution_detects_singular_diagonal() -> None:
    """
    Verifies the failure case: if one diagonal value is zero, the solver must raise LinAlgError and identify the problematic row.
    """
    upper = np.eye(4)
    upper[2, 2] = 0.0
    with pytest.raises(np.linalg.LinAlgError, match="row 2"):
        _back_substitute_upper(
            upper,
            np.ones(4),
            singular_tolerance=1.0e-14,
        )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_cupy_gmres_matches_direct_face_solution(
    boundary_mode: str,
    preconditioned: bool,
) -> None:
    """ The assembled face system is wrapped in CuPyFaceDenseOperator, which provides the GPU matrix-vector operation (Ax).
    When requested, it also creates a block-Jacobi preconditioner.
    The test verifies that:
    - GMRES reports convergence.
    - The solution remains a CuPy GPU array.
    - Its shape is preserved.
    - The true relative residual is sufficiently small.
    - Matrix-vector products, dot products, norms and AXPY operations were actually performed.
    - There is one basis update per restart cycle.
    - The preconditioner counter is positive only when preconditioning is enabled.
    Finally, the solution is copied back to the CPU and compared with the direct face-system solution.

    The unpreconditioned penalty case first divides the artificial boundary
    equations by their penalty factor.  This preserves their solution exactly
    and prevents the stopping norm from being dominated by an arbitrary row
    scale.  Its residual is also checked against the original physical system.
    """

    cp = _cupy_or_skip()
    direct = _small_face_system(boundary_mode)
    physical_system = direct.system
    system = physical_system
    if boundary_mode == "penalty" and not preconditioned:
        boundary_faces = np.flatnonzero(
            direct.assembly.topology.incidence_count == 1
        )
        system = normalize_penalty_rows(
            physical_system,
            boundary_faces,
            boundary_penalty=_BOUNDARY_PENALTY,
        )
    operator = CuPyFaceDenseOperator.from_system(system, implementation="matmul")
    rhs = operator.to_device(system.rhs)
    preconditioner = None
    if preconditioned:
        preconditioner = CuPyFaceBlockJacobiPreconditioner.from_system(
            system,
            device_id=operator.device_id,
        )

    result = restarted_gmres_cupy(
        operator,
        rhs,
        restart=12,
        max_iterations=500,
        rtol=1.0e-10,
        preconditioner=preconditioner,
        reorthogonalize=True,
    )
    operator.synchronize()

    assert result.converged, result.status
    assert isinstance(result.solution, cp.ndarray)
    assert result.solution.shape == system.rhs.shape
    assert result.relative_residual <= 1.2e-10
    assert result.matvec_count >= result.iterations + 2
    assert result.dot_count > 0
    assert result.axpy_count > 0
    assert result.norm_count > 0
    assert result.basis_update_count == result.restart_cycles
    assert result.orthogonalization == "mgs2"
    assert result.basis_projection_count == 0
    assert result.basis_correction_count == 0
    assert result.coefficient_d2h_count == 0
    if preconditioned:
        assert result.preconditioner_count > 0
    else:
        assert result.preconditioner_count == 0

    computed = operator.to_host(result.solution).reshape(-1)
    assert (
        face_dense_relative_residual(physical_system, computed)
        <= 1.2e-10
    )
    np.testing.assert_allclose(
        computed,
        direct.system_solution,
        rtol=2.0e-9,
        atol=2.0e-10,
    )


def test_cupy_gmres_exact_initial_guess_exits_without_arnoldi() -> None:
    """
    Supplies the direct solution as x0.
    Because this initial guess already satisfies (Ax=b), GMRES should immediately return:
    - Zero iterations.
    - Zero restart cycles.
    - Zero Krylov-basis updates.
    This ensures the solver checks the initial residual before doing expensive Arnoldi work.
    """
    _cupy_or_skip()
    direct = _small_face_system("eliminate", nx=2, ny=2)
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system)
    rhs = operator.to_device(system.rhs)
    x0 = operator.to_device(direct.system_solution)

    result = restarted_gmres_cupy(
        operator,
        rhs,
        x0=x0,
        restart=10,
        max_iterations=50,
        rtol=1.0e-11,
    )
    assert result.converged
    assert result.iterations == 0
    assert result.restart_cycles == 0
    assert result.basis_update_count == 0
    assert result.orthogonalization == "mgs"


def test_cupy_batched_basis_projection_and_correction_match_numpy() -> None:
    cp = _cupy_or_skip()
    rng = np.random.default_rng(20260728)
    basis_host = np.ascontiguousarray(rng.standard_normal((5, 37)))
    vector_host = np.ascontiguousarray(rng.standard_normal(37))
    basis = cp.asarray(basis_host)
    vector = cp.asarray(vector_host)
    coefficients = cp.empty(5, dtype=cp.float64)
    blas = CuPyVectorBLAS(dtype=cp.float64, device_id=int(cp.cuda.Device().id))

    blas.basis_projection(basis, vector, coefficients)
    coefficient_host = cp.asnumpy(coefficients)
    np.testing.assert_allclose(
        coefficient_host,
        basis_host @ vector_host,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    corrected = vector.copy()
    blas.basis_correction(basis, coefficients, corrected)
    np.testing.assert_allclose(
        cp.asnumpy(corrected),
        vector_host - basis_host.T @ coefficient_host,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    copied = np.empty_like(coefficient_host)
    blas.copy_device_vector_to_host(coefficients, copied)
    np.testing.assert_allclose(copied, coefficient_host, rtol=0.0, atol=0.0)


def test_cupy_cgs2_matches_mgs2_with_fewer_host_synchronizations() -> None:
    _cupy_or_skip()
    direct = _small_face_system("eliminate", nx=4, ny=4)
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system, implementation="matmul")
    rhs = operator.to_device(system.rhs)

    mgs2 = restarted_gmres_cupy(
        operator,
        rhs,
        restart=20,
        max_iterations=500,
        rtol=1.0e-10,
        orthogonalization="mgs2",
    )
    cgs2 = restarted_gmres_cupy(
        operator,
        rhs,
        restart=20,
        max_iterations=500,
        rtol=1.0e-10,
        orthogonalization="cgs2",
    )
    operator.synchronize()

    assert mgs2.converged, mgs2.status
    assert cgs2.converged, cgs2.status
    assert mgs2.orthogonalization == "mgs2"
    assert cgs2.orthogonalization == "cgs2"
    assert mgs2.dot_count > 0
    assert mgs2.basis_projection_count == 0
    assert cgs2.dot_count == 0
    assert cgs2.basis_projection_count == 2 * cgs2.iterations
    assert cgs2.basis_correction_count == cgs2.basis_projection_count
    assert cgs2.coefficient_d2h_count == cgs2.basis_projection_count
    assert cgs2.coefficient_d2h_count < mgs2.dot_count

    computed_mgs2 = operator.to_host(mgs2.solution).reshape(-1)
    computed_cgs2 = operator.to_host(cgs2.solution).reshape(-1)
    np.testing.assert_allclose(
        computed_cgs2,
        computed_mgs2,
        rtol=3.0e-9,
        atol=3.0e-10,
    )
    np.testing.assert_allclose(
        computed_cgs2,
        direct.system_solution,
        rtol=3.0e-9,
        atol=3.0e-10,
    )


def test_cupy_block_jacobi_application_matches_cpu_reference() -> None:
    """
    It applies the block-Jacobi preconditioner to the same random vector in two ways:
    1. Using the existing NumPy CPU implementation.
    2. Using CuPyFaceBlockJacobiPreconditioner on the GPU.
    """
    _cupy_or_skip()
    direct = _small_face_system("eliminate", nx=2, ny=2)
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system)
    preconditioner = CuPyFaceBlockJacobiPreconditioner.from_system(
        system,
        device_id=operator.device_id,
    )
    rng = np.random.default_rng(77)
    vector = rng.standard_normal(system.rhs.shape)

    from hybridge.linalg.block_jacobi import (
        build_face_block_jacobi_preconditioner,
    )

    expected = build_face_block_jacobi_preconditioner(system).apply(vector)
    vector_device = operator.to_device(vector)
    result_device = preconditioner.apply(vector_device)
    np.testing.assert_allclose(
        operator.to_host(result_device),
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_orthogonality_metrics_identity_and_perturbation() -> None:
    from hybridge.linalg.gpu.gmres import _orthogonality_metrics_from_gram

    identity = np.eye(4)
    assert _orthogonality_metrics_from_gram(identity) == (0.0, 0.0, 0.0)

    gram = identity.copy()
    gram[0, 1] = gram[1, 0] = 2.0e-3
    gram[2, 2] += 3.0e-4
    frobenius, offdiagonal, diagonal = _orthogonality_metrics_from_gram(gram)
    np.testing.assert_allclose(
        frobenius,
        np.sqrt(2.0 * (2.0e-3) ** 2 + (3.0e-4) ** 2),
        rtol=2.0e-15,
    )
    assert offdiagonal == pytest.approx(2.0e-3)
    assert diagonal == pytest.approx(3.0e-4)


@pytest.mark.parametrize("bad", [np.ones(3), np.ones((2, 3))])
def test_orthogonality_metrics_reject_nonsquare_input(bad: np.ndarray) -> None:
    from hybridge.linalg.gpu.gmres import _orthogonality_metrics_from_gram

    with pytest.raises(ValueError, match="square"):
        _orthogonality_metrics_from_gram(bad)


def test_cupy_gmres_optional_orthogonality_monitor_records_each_cycle() -> None:
    _cupy_or_skip()
    direct = _small_face_system("eliminate", nx=4, ny=4)
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system, implementation="raw")
    rhs = operator.to_device(system.rhs)

    result = restarted_gmres_cupy(
        operator,
        rhs,
        restart=10,
        max_iterations=200,
        rtol=1.0e-10,
        orthogonalization="cgs2",
        monitor_orthogonality=True,
    )
    operator.synchronize()

    assert result.converged, result.status
    assert len(result.orthogonality_records) == result.restart_cycles
    for cycle, record in enumerate(result.orthogonality_records, start=1):
        assert record.restart_cycle == cycle
        assert record.total_iterations > 0
        assert 1 <= record.basis_dimension <= 11
        assert record.frobenius_defect >= 0.0
        assert record.maximum_offdiagonal >= 0.0
        assert record.maximum_diagonal_error >= 0.0
        assert np.isfinite(record.frobenius_defect)


def test_cupy_gmres_orthogonality_monitor_disabled_by_default() -> None:
    _cupy_or_skip()
    direct = _small_face_system("eliminate", nx=2, ny=2)
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system, implementation="raw")
    result = restarted_gmres_cupy(
        operator,
        operator.to_device(system.rhs),
        restart=10,
        max_iterations=100,
        rtol=1.0e-10,
        orthogonalization="cgs",
    )
    assert result.orthogonality_records == ()



def test_cupy_reusable_workspace_matches_legacy_solver_for_two_rhs() -> None:
    cp = _cupy_or_skip()
    direct = _small_face_system("eliminate", nx=3, ny=3)
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system, implementation="raw")
    rhs = operator.to_device(system.rhs).reshape(-1)
    output = cp.empty_like(rhs)
    workspace = CuPyGMRESWorkspace.allocate(
        num_dofs=operator.num_dofs,
        restart_capacity=20,
        dtype=operator.dtype,
        device_id=operator.device_id,
    )
    pointers_before = tuple(int(array.data.ptr) for array in workspace.device_arrays)

    legacy = restarted_gmres_cupy(
        operator,
        rhs,
        restart=20,
        max_iterations=500,
        rtol=1.0e-10,
        orthogonalization="cgs2",
    )
    reusable = restarted_gmres_cupy(
        operator,
        rhs,
        restart=20,
        max_iterations=500,
        rtol=1.0e-10,
        orthogonalization="cgs2",
        workspace=workspace,
        solution_out=output,
    )
    operator.synchronize()

    assert legacy.converged and reusable.converged
    assert reusable.solution.data.ptr == output.data.ptr
    np.testing.assert_allclose(
        cp.asnumpy(reusable.solution),
        cp.asnumpy(legacy.solution),
        rtol=3.0e-10,
        atol=3.0e-11,
    )

    scaled_rhs = 1.7 * rhs
    second = restarted_gmres_cupy(
        operator,
        scaled_rhs,
        restart=20,
        max_iterations=500,
        rtol=1.0e-10,
        orthogonalization="cgs2",
        workspace=workspace,
        solution_out=output,
    )
    operator.synchronize()
    assert second.converged
    np.testing.assert_allclose(
        cp.asnumpy(second.solution),
        1.7 * direct.system_solution.reshape(-1),
        rtol=3.0e-9,
        atol=3.0e-10,
    )
    assert tuple(int(array.data.ptr) for array in workspace.device_arrays) == pointers_before


def test_cupy_restarted_solver_reuses_workspace_and_preallocated_output() -> None:
    cp = _cupy_or_skip()
    direct = _small_face_system("eliminate", nx=2, ny=2)
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system, implementation="raw")
    rhs = operator.to_device(system.rhs)
    output = cp.empty_like(rhs)
    solver = CuPyRestartedGMRESSolver(
        operator,
        restart=15,
        max_iterations=300,
        rtol=1.0e-10,
        orthogonalization="cgs",
    )

    first = solver.solve(rhs, solution_out=output)
    first_host = cp.asnumpy(first.solution).copy().reshape(-1)
    second = solver.solve(0.5 * rhs, solution_out=output)
    operator.synchronize()

    assert first.converged and second.converged
    assert solver.workspace_device_bytes == solver.workspace.device_bytes
    np.testing.assert_allclose(
        first_host,
        direct.system_solution,
        rtol=3.0e-9,
        atol=3.0e-10,
    )
    np.testing.assert_allclose(
        cp.asnumpy(second.solution).reshape(-1),
        0.5 * direct.system_solution,
        rtol=3.0e-9,
        atol=3.0e-10,
    )


def test_cupy_workspace_rejects_insufficient_capacity_and_aliasing() -> None:
    cp = _cupy_or_skip()
    direct = _small_face_system("eliminate", nx=2, ny=2)
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system, implementation="raw")
    rhs = operator.to_device(system.rhs).reshape(-1)
    workspace = CuPyGMRESWorkspace.allocate(
        num_dofs=operator.num_dofs,
        restart_capacity=4,
        dtype=operator.dtype,
        device_id=operator.device_id,
    )

    with pytest.raises(ValueError, match="restart_capacity"):
        restarted_gmres_cupy(
            operator,
            rhs,
            restart=5,
            max_iterations=20,
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="must not overlap rhs"):
        restarted_gmres_cupy(
            operator,
            rhs,
            restart=4,
            max_iterations=20,
            solution_out=rhs,
        )
    with pytest.raises(ValueError, match="workspace"):
        restarted_gmres_cupy(
            operator,
            workspace.residual,
            restart=4,
            max_iterations=20,
            workspace=workspace,
        )