"""Device source moments agree with the host rule for fields from other spaces."""

from __future__ import annotations

import numpy as np
import pytest

from hdgfem import DGSpace, rectangle_mesh
from hdgfem.assembly import hdg

cp = pytest.importorskip("cupy")


def _require_device():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except Exception as exc:  # pragma: no cover - driver dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")


@pytest.mark.parametrize("source_order", (1, 2, 4))
def test_device_source_moments_match_host_for_other_space_fields(source_order):
    """A same-mesh field of another order is sampled, not reused as coefficients."""
    _require_device()
    from hdgfem.backends import advection_cuda, diffusion_cupy
    from hdgfem.backends.cupy import as_cupy_space

    mesh = rectangle_mesh(3, 2)
    space = DGSpace(mesh, 2)
    source = DGSpace(mesh, source_order).project_callable(lambda x, y: np.sin(x) + x * y**2)
    expected = hdg.source_moments(source, space)
    cspace = as_cupy_space(space)

    advection = cp.asnumpy(advection_cuda.source_moments_cupy(source, cspace))
    diffusion = cp.asnumpy(diffusion_cupy.source_moments_cupy(source, cspace))[:, : space.el_dof]
    np.testing.assert_allclose(advection, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(diffusion, expected, rtol=1e-12, atol=1e-14)
