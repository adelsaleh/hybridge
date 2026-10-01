"""Device local reconstructions reject non-finite results (fix A4)."""

from __future__ import annotations

import numpy as np
import pytest

from hdgfem import DGSpace, rectangle_mesh

cp = pytest.importorskip("cupy")


def _require_device():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except Exception as exc:  # pragma: no cover - driver dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")


def test_require_finite_device_values_names_stage_and_elements():
    """Finite values pass through; NaN/Inf raise with the stage and element count."""
    _require_device()
    from hdgfem.hdg.condensation_device import require_finite_device_values

    values = cp.ones((4, 3))
    assert require_finite_device_values(values, "stage") is values
    values[1, 2] = cp.nan
    values[3, 0] = cp.inf
    with pytest.raises(np.linalg.LinAlgError, match=r"test reconstruction produced non-finite values on 2 element"):
        require_finite_device_values(values, "test reconstruction")
    host = np.ones((2, 2))
    host[0, 0] = np.nan
    with pytest.raises(np.linalg.LinAlgError, match="on 1 element"):
        require_finite_device_values(host, "host")


def test_raw_cuda_advection_reconstruction_rejects_non_finite_trace():
    """A NaN reaching the fused raw-CUDA local solve raises instead of returning NaN fields."""
    _require_device()
    from hdgfem.core.device import as_cupy_space, as_cupy_trace_space
    from hdgfem.transport.cuda import (
        assemble_reduced_system_cuda,
        project_callable_cupy,
        reconstruct_advection_field_cuda,
    )

    space = DGSpace(rectangle_mesh(2, 1), 2, basis_type="dub_orth")
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(space.trace_space("legendre-modal"), device=cspace.device_id)
    source_h = space.project_callable(lambda x, y: 1.0 + x, name="source_h")
    reaction_h = space.project_callable(lambda x, y: 2.0 + 0.0 * x, name="reaction_h")
    beta_coeffs = cp.ascontiguousarray(cp.stack((
        project_callable_cupy(lambda x, y: 1.0 + 0.0 * x, cspace),
        project_callable_cupy(lambda x, y: 0.5 + 0.0 * x, cspace),
    ), axis=0))
    assembly = assemble_reduced_system_cuda(
        source_h, reaction_h, lambda x, y: x + y, beta_coeffs, cspace, trace_ref,
        backend="raw-cuda", raw_local_assembly="fused", raw_lu_mode="coop",
        raw_block_size=32, raw_matrix_format="coo",
    )
    trace = cp.zeros(space.mesh.num_edg * cspace.edg_dof, dtype=cp.float64)
    uh, _ = reconstruct_advection_field_cuda(trace, source_h, reaction_h, beta_coeffs, assembly)
    assert bool(cp.isfinite(uh).all())

    trace[0] = cp.nan
    with pytest.raises(np.linalg.LinAlgError, match="raw-CUDA advection reconstruction"):
        reconstruct_advection_field_cuda(trace, source_h, reaction_h, beta_coeffs, assembly)
