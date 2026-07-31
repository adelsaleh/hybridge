from __future__ import annotations

import json

import numpy as np
import pytest

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_autotune import (
    FaceDenseAutotuneResult,
    KernelCandidateTiming,
    autotune_face_dense_gpu,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import diffusion_element_boundary_mats, local_solvers
from hdgfem.solvers.diff_rea_face_dense import assemble_diffusion_face_dense_components
from scripts.diff_rea_cases import quadratic_poisson_case


def _cupy_or_skip():
    try:
        return require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))


def test_autotune_result_is_json_serializable() -> None:
    row = KernelCandidateTiming("raw", 1.0, 0.8, 1024, 0.0)
    result = FaceDenseAutotuneResult(
        device_name="test",
        dtype="float64",
        num_dofs=100,
        block_size=3,
        operator_choice="raw",
        asm_choice="fused",
        operator_candidates=(row,),
        asm_candidates=(KernelCandidateTiming("fused", 2.0, 1.8, 512, 0.0),),
    )
    payload = result.to_dict()
    assert json.loads(json.dumps(payload))["operator_choice"] == "raw"


def test_gpu_autotuner_selects_numerically_equivalent_candidates() -> None:
    _cupy_or_skip()
    space = DGSpace(rectangle_mesh(2, 2), 1, basis_type="dub_orth")
    diffusion, reaction, source, boundary = quadratic_poisson_case()
    tau = 1.3
    local_solver = local_solvers(reaction, tau, space, diffusion=diffusion)
    boundary_mats = diffusion_element_boundary_mats(tau, space)
    source_rhs = hdg_assembly.block_source_moments(
        source, space, num_blocks=3, source_block=0
    )
    assembly = assemble_diffusion_face_dense_components(
        local_solver, boundary_mats, source_rhs, boundary, tau, space
    )
    result = autotune_face_dense_gpu(
        assembly.eliminated_system,
        element_blocks=assembly.element_blocks,
        loc2glob_face=space.mesh.loc2glob_edge,
        warmup=1,
        repeats=3,
    )
    assert result.operator_choice in {"raw", "raw_fused"}
    assert result.asm_choice in {"raw", "fused"}
    assert all(row.relative_error < 5e-12 for row in result.operator_candidates)
    assert all(row.relative_error < 5e-12 for row in result.asm_candidates)
