from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import (
    DiffusionReactionHDGOptions,
    DiffusionReactionHDGSolver,
    solve_diffusion_reaction_hdg,
)
from hdgfem.solvers.diff_rea_gpu import (
    DiffusionReactionGPUOptions,
    _extract_initial_guess,
)
from scripts.diff_rea_cases import quadratic_poisson_case


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def test_gpu_options_normalize_mapping_and_validate() -> None:
    options = DiffusionReactionGPUOptions.normalize(
        {
            "operator": "raw",
            "asm_application": "fused",
            "polynomial_degree": 12,
            "autotune": False,
        }
    )
    options.validate()
    assert options.operator == "raw"
    assert options.asm_application == "fused"
    assert options.polynomial_degree == 12
    assert not options.autotune

    with pytest.raises(TypeError, match="unknown"):
        DiffusionReactionGPUOptions.normalize({"not_an_option": 1})
    with pytest.raises(ValueError, match="polynomial_degree"):
        options.with_overrides(polynomial_degree=0).validate()


def test_hdg_options_accept_gpu_configuration() -> None:
    gpu = DiffusionReactionGPUOptions(operator="raw", autotune=False)
    options = DiffusionReactionHDGOptions(
        solver="gpu_face_dense",
        gpu_options=gpu,
        boundary_mode="eliminate",
    )
    kwargs = options.as_solve_kwargs()
    assert kwargs["solver"] == "gpu_face_dense"
    assert kwargs["gpu_options"] is gpu


@dataclass
class _FakeSystem:
    block_size: int = 2
    mode: str = "eliminate"
    global_faces: np.ndarray = None

    def __post_init__(self):
        if self.global_faces is None:
            self.global_faces = np.array([1, 3], dtype=np.int64)


def test_extract_initial_guess_reduces_eliminated_faces() -> None:
    full = np.arange(10, dtype=np.float64)
    system = _FakeSystem()
    reduced = _extract_initial_guess(full, system=system, num_global_faces=5)
    np.testing.assert_array_equal(reduced, np.array([2.0, 3.0, 6.0, 7.0]))

    system.mode = "penalty"
    penalty = _extract_initial_guess(full, system=system, num_global_faces=5)
    np.testing.assert_array_equal(penalty, full)


def test_ordinary_api_delegates_gpu_solver(monkeypatch) -> None:
    space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()
    sentinel = object()
    captured = {}

    def fake_gpu(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "hdgfem.solvers.diff_rea_gpu.solve_diffusion_reaction_face_dense_gpu",
        fake_gpu,
    )
    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        solver="gpu_face_dense",
        solver_rtol=2.0e-8,
        boundary_mode="eliminate",
        gpu_options={"operator": "raw", "autotune": False},
        verbose=False,
    )
    assert result is sentinel
    assert captured["solver_rtol"] == 2.0e-8
    assert captured["boundary_mode"] == "eliminate"
    assert captured["gpu_options"]["operator"] == "raw"


def test_stateful_solver_forwards_gpu_options(monkeypatch) -> None:
    space = DGSpace(rectangle_mesh(1, 1), 1, basis_type="dub_orth")
    _, reaction, source, exact = quadratic_poisson_case()
    sentinel = object()

    def fake_solve(*args, **kwargs):
        assert kwargs["solver"] == "gpu_face_dense"
        assert kwargs["gpu_options"]["operator"] == "raw_fused"
        return sentinel

    monkeypatch.setattr("hdgfem.solvers.diff_rea.solve_diffusion_reaction_hdg", fake_solve)
    solver = DiffusionReactionHDGSolver(
        space,
        source=source,
        reaction=reaction,
        boundary_condition=exact,
        solver="gpu_face_dense",
        gpu_options={"operator": "raw_fused"},
        verbose=False,
    )
    # Avoid storing a deliberately incomplete sentinel result; this test only
    # verifies the stateful API forwards the new configuration.
    monkeypatch.setattr(solver, "_postprocess_result", lambda value: value)
    monkeypatch.setattr(solver, "_store_result", lambda value: None)
    assert solver.solve() is sentinel


@pytest.mark.parametrize("boundary_mode", ("eliminate", "penalty"))
def test_integrated_gpu_solve_matches_cpu_direct(boundary_mode: str) -> None:
    _cupy_or_skip()
    space = DGSpace(rectangle_mesh(2, 2), 1, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()

    cpu = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        solver="direct",
        preconditioner=None,
        boundary_mode=boundary_mode,
        verbose=False,
    )
    gpu = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        solver="gpu_face_dense",
        solver_rtol=1.0e-10,
        maxiter=500,
        boundary_mode=boundary_mode,
        gpu_options={
            "operator": "raw",
            "preconditioner": "asm",
            "asm_application": "raw",
            "restart": 30,
            "autotune": False,
        },
        verbose=False,
    )

    np.testing.assert_allclose(gpu.trace, cpu.trace, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(
        gpu.field.coeffs, cpu.field.coeffs, rtol=2.0e-9, atol=2.0e-9
    )
    assert gpu.linear_solver_backend == "gpu_face_dense"
    assert gpu.gpu_diagnostics is not None
    assert gpu.gpu_diagnostics.gmres_result.converged
    assert gpu.global_solve_result.iteration_count > 0
