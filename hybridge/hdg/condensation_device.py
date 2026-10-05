"""hybridge.hdg.condensation_device."""

from __future__ import annotations

import numpy as np
import time
from typing import Any
from hybridge.core.device import CupyDGSpace, CupyDGTraceSpace
from hybridge.linalg.reduction import KnownDofReduction
from hybridge.runtime.precision import REAL_DTYPE
from dataclasses import dataclass, field
from hybridge.runtime.optional import array_module, require_cupy
from hybridge.runtime.logging import sync_elapsed

from hybridge.core.space import DGSpace, DGTraceSpace
from hybridge.core.device import as_cupy_space, as_cupy_trace_reference



@dataclass(frozen=True)
class CudaAdvectionAssembly:
    """Reduced trace system assembled by the CUDA path.

    Rows, columns, data, RHS, local tensors, and boundary trace are CUDA arrays.
    Use :meth:`to_host_reduction` when passing the system to host-only solver
    APIs.
    """

    rows: Any | None
    cols: Any | None
    data: Any
    rhs: Any
    local_mats: Any | None
    element_boundary: Any | None
    source_rhs: Any | None
    boundary_trace: Any
    beta_dot_normal: Any | None
    cspace: CupyDGSpace
    trace_ref: CupyDGTraceSpace
    raw: Any | None = None  # raw-CUDA result (e.g. RawAdvectionAssemblyResult) when assembled by raw kernels
    indptr: Any | None = None
    indices: Any | None = None
    matrix_format: str = "coo"
    timings: dict[str, float] = field(default_factory=dict)

    def to_host_reduction(self) -> KnownDofReduction:
        """Transfer the reduced system metadata to a host KnownDofReduction."""
        if self.matrix_format != "coo":
            raise RuntimeError("host KnownDofReduction materialization is currently supported only for COO raw systems")
        cp = require_cupy()
        space = self.cspace.host
        edg_dof = self.cspace.edg_dof
        full_size = space.mesh.num_edg * edg_dof
        free_mask = np.zeros(full_size, dtype=bool)
        local = np.arange(edg_dof, dtype=np.int64)
        free_mask[(space.mesh.int_edges_inds[:, None] * edg_dof + local[None, :]).ravel()] = True
        known_mask = ~free_mask
        known_values = np.zeros(full_size, dtype=REAL_DTYPE)
        boundary_host = np.ascontiguousarray(cp.asnumpy(self.boundary_trace), dtype=REAL_DTYPE)
        known_values[(space.mesh.bnd_edges_inds[:, None] * edg_dof + local[None, :]).ravel()] = boundary_host.ravel()
        old_to_new = np.full(full_size, -1, dtype=np.int64)
        old_to_new[free_mask] = np.arange(np.count_nonzero(free_mask), dtype=np.int64)
        return KnownDofReduction(
            rows=np.ascontiguousarray(cp.asnumpy(self.rows), dtype=np.int64),
            cols=np.ascontiguousarray(cp.asnumpy(self.cols), dtype=np.int64),
            data=np.ascontiguousarray(cp.asnumpy(self.data), dtype=REAL_DTYPE),
            rhs=np.ascontiguousarray(cp.asnumpy(self.rhs), dtype=REAL_DTYPE),
            free_mask=np.ascontiguousarray(free_mask),
            known_mask=np.ascontiguousarray(known_mask),
            known_values=np.ascontiguousarray(known_values),
            old_to_new=np.ascontiguousarray(old_to_new),
        )


def reconstruct_trace_cupy(trace_reduced, boundary_trace, cspace: CupyDGSpace, timings: dict[str, float] | None = None):
    """Expand reduced trace values into the full device trace vector."""
    cp = require_cupy()
    start = time.perf_counter()
    trace = cp.empty(cspace.mesh.num_edg * cspace.edg_dof, dtype=REAL_DTYPE)
    trace_r = trace.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    trace_r[cspace.mesh.int_edges_inds] = trace_reduced.reshape((cspace.mesh.int_edges_inds.size, cspace.edg_dof))
    trace_r[cspace.mesh.bnd_edges_inds] = boundary_trace
    if timings is not None:
        timings["reconstruct.trace"] = timings.get("reconstruct.trace", 0.0) + sync_elapsed(start)
    return trace


def element_traces_cupy(
        trace,
        space: DGSpace | CupyDGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
):
    """Gather and orient global trace coefficients entirely on-device."""
    cupy = require_cupy()
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_reference(trace_space, cspace)
    expected_size = int(cspace.mesh.num_edg * trace_ref.edg_dof)
    trace_cp = cupy.asarray(trace, dtype=REAL_DTYPE)
    if trace_cp.shape != (expected_size,):
        raise ValueError(f"trace must have shape ({expected_size},); got {trace_cp.shape}")

    traces = trace_cp.reshape((cspace.mesh.num_edg, trace_ref.edg_dof))[
        cspace.mesh.loc2glob_edge
    ].copy()
    if cspace.mesh.num_negative_orientations:
        elements = cspace.mesh.negative_orientation_elements
        faces = cspace.mesh.negative_orientation_faces
        negative = traces[elements, faces]
        if trace_ref.kind == "legendre-modal":
            signs = cupy.where(
                cupy.arange(trace_ref.edg_dof, dtype=cupy.int64) % 2 == 0,
                REAL_DTYPE(1.0),
                REAL_DTYPE(-1.0),
            )
            traces[elements, faces] = negative * signs[None, :]
        else:
            traces[elements, faces] = negative[:, ::-1]
    return cupy.ascontiguousarray(
        traces.reshape((cspace.mesh.num_tri, 3 * trace_ref.edg_dof))
    )


def require_finite_device_values(values, stage: str):
    """Raise ``numpy.linalg.LinAlgError`` if device values contain NaN or Inf.

    The raw-CUDA local LU kernels clamp tiny pivots but cannot detect a
    non-finite pivot, and a non-finite element solve poisons that element's
    outputs. The global AMGX solve already rejects non-finite systems, so local
    reconstruction outputs are where such a failure would otherwise pass
    silently. Accepts host or device arrays. Returns ``values`` unchanged.
    """
    xp = array_module(values)
    if not bool(xp.isfinite(values).all()):
        flat = values.reshape(values.shape[0], -1)
        elements = int((~xp.isfinite(flat)).any(axis=1).sum())
        raise np.linalg.LinAlgError(
            f"{stage} produced non-finite values on {elements} element(s): "
            "a local solve met a NaN/Inf pivot or input"
        )
    return values
