"""GPU-local assembly for scalar diffusion-reaction HDG systems.

This module ports the numerically expensive element-local portion of the
identity-diffusion HDG setup to CuPy/CUDA.  Python coefficient callables are
still evaluated on the host, because arbitrary Python functions cannot be
executed inside a CUDA kernel.  Once reaction mass matrices, geometry arrays,
and reference tensors are prepared, the following path remains on the GPU::

    physical derivative blocks
      -> local scalar Schur matrices
      -> batched local inversion
      -> full mixed local inverse
      -> trace lift and boundary coupling
      -> complete elemental face blocks
      -> deterministic global face assembly

Two reusable square element work arrays are alternated as ping/pong buffers
for intermediate batched matrix products.  The returned elemental blocks can
feed :class:`hdgfem.backends.cupy_assembly.CuPyGlobalFaceAssembler` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.assembly.face_dense import FaceTopology
from hdgfem.backends.cublas_batched import invert_batched_cublas
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_assembly import CuPyGlobalFaceAssembler
from hdgfem.solvers.diff_rea import _normalize_tau, _reference_derivative_matrices

LocalInverseBackend = Literal["cublas_inverse", "gpu_inverse"]


@dataclass(frozen=True)
class DiffusionLocalAssemblyHostInputs:
    """CPU metadata transferred once for GPU-local diffusion assembly."""

    tau: np.ndarray
    reaction_mass: np.ndarray
    d0_base: np.ndarray
    d1_base: np.ndarray
    aff_mats: np.ndarray
    aff_jacs: np.ndarray
    face_jacs: np.ndarray
    normals: np.ndarray
    oriented_trace_restriction: np.ndarray
    face_element_trace: np.ndarray
    face_element_mass: np.ndarray
    face_mass: np.ndarray
    mkrf_inv: np.ndarray
    orientations: np.ndarray
    interior_face_mask: np.ndarray

    @property
    def num_elements(self) -> int:
        return int(self.tau.shape[0])

    @property
    def element_dofs(self) -> int:
        return int(self.reaction_mass.shape[1])

    @property
    def face_dofs(self) -> int:
        return int(self.face_mass.shape[0])


def prepare_diffusion_local_assembly_inputs(
    reaction: Any,
    stabilization: Any,
    space: Any,
    *,
    dtype: Any = np.float64,
) -> DiffusionLocalAssemblyHostInputs:
    """Prepare host coefficient and geometry arrays for GPU assembly.

    Arbitrary callable reaction coefficients are evaluated through the existing
    validated CPU quadrature path.  All subsequent dense element algebra is
    performed by :class:`CuPyDiffusionLocalAssembler`.
    """

    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dtype must be float32 or float64")

    mesh = space.mesh
    q = space.quad_data
    d0_base, d1_base = _reference_derivative_matrices(space)
    tau = _normalize_tau(stabilization, space)
    reaction_mass = hdg_assembly.reaction_mass(reaction, space)
    oriented = q.face_trace_test_element_trial_oriented[
        mesh.loc2oriented_face_coupling
    ]

    return DiffusionLocalAssemblyHostInputs(
        tau=np.ascontiguousarray(tau, dtype=dtype),
        reaction_mass=np.ascontiguousarray(reaction_mass, dtype=dtype),
        d0_base=np.ascontiguousarray(d0_base, dtype=dtype),
        d1_base=np.ascontiguousarray(d1_base, dtype=dtype),
        aff_mats=np.ascontiguousarray(mesh.aff_mats, dtype=dtype),
        aff_jacs=np.ascontiguousarray(mesh.aff_jacs, dtype=dtype),
        face_jacs=np.ascontiguousarray(mesh.jacs_el_fc, dtype=dtype),
        normals=np.ascontiguousarray(mesh.normals, dtype=dtype),
        oriented_trace_restriction=np.ascontiguousarray(oriented, dtype=dtype),
        face_element_trace=np.ascontiguousarray(
            q.face_element_test_trace_trial.swapaxes(0, 1), dtype=dtype
        ),
        face_element_mass=np.ascontiguousarray(
            q.face_element_test_element_trial, dtype=dtype
        ),
        face_mass=np.ascontiguousarray(q.M_rf_fc, dtype=dtype),
        mkrf_inv=np.ascontiguousarray(q.MKrf_inv, dtype=dtype),
        orientations=np.ascontiguousarray(mesh.orientations, dtype=bool),
        interior_face_mask=np.ascontiguousarray(mesh.interior_face_mask, dtype=bool),
    )


@dataclass
class CuPyDiffusionAssemblyWorkspace:
    """Reusable element-local ping/pong work buffers."""

    ping: Any
    pong: Any

    @classmethod
    def allocate(
        cls,
        *,
        num_elements: int,
        element_dofs: int,
        dtype: Any,
        device_id: int,
    ) -> "CuPyDiffusionAssemblyWorkspace":
        cp = require_cupy_device()
        with cp.cuda.Device(int(device_id)):
            shape = (int(num_elements), int(element_dofs), int(element_dofs))
            return cls(
                ping=cp.empty(shape, dtype=dtype),
                pong=cp.empty(shape, dtype=dtype),
            )

    @property
    def nbytes(self) -> int:
        return int(self.ping.nbytes + self.pong.nbytes)


@dataclass(frozen=True)
class CuPyDiffusionLocalAssemblyResult:
    """Device-resident products of the local HDG assembly pipeline."""

    element_blocks: Any
    trace_blocks: Any
    local_solver: Any | None
    trace_lift: Any | None
    element_boundary_mats: Any | None
    scalar_inverse_residuals: Any
    factorization_info: np.ndarray | None
    inversion_info: np.ndarray | None

    @property
    def maximum_scalar_inverse_residual(self) -> float:
        cp = require_cupy_device()
        return float(cp.max(self.scalar_inverse_residuals).item())


_FACE_RHS_KERNEL_SOURCE = r"""
extern "C" __global__
void assemble_face_rhs_f32(
    const unsigned long long total,
    const int block_size,
    const int* __restrict__ element_ids,
    const int* __restrict__ local_faces,
    const float* __restrict__ local_face_rhs,
    float* __restrict__ global_rhs)
{
    const unsigned long long i =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    const int dof = (int) (i % block_size);
    const int face = (int) (i / block_size);
    const unsigned long long map = (unsigned long long) face * 2;
    float value = 0.0f;
    #pragma unroll
    for (int side = 0; side < 2; ++side) {
        const int element = element_ids[map + side];
        if (element < 0) continue;
        const int local_face = local_faces[map + side];
        value += local_face_rhs[
            ((unsigned long long) element * 3 + local_face) * block_size + dof
        ];
    }
    global_rhs[i] = value;
}

extern "C" __global__
void assemble_face_rhs_f64(
    const unsigned long long total,
    const int block_size,
    const int* __restrict__ element_ids,
    const int* __restrict__ local_faces,
    const double* __restrict__ local_face_rhs,
    double* __restrict__ global_rhs)
{
    const unsigned long long i =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    const int dof = (int) (i % block_size);
    const int face = (int) (i / block_size);
    const unsigned long long map = (unsigned long long) face * 2;
    double value = 0.0;
    #pragma unroll
    for (int side = 0; side < 2; ++side) {
        const int element = element_ids[map + side];
        if (element < 0) continue;
        const int local_face = local_faces[map + side];
        value += local_face_rhs[
            ((unsigned long long) element * 3 + local_face) * block_size + dof
        ];
    }
    global_rhs[i] = value;
}
"""


class CuPyDiffusionLocalAssembler:
    """Reusable GPU assembler for identity-diffusion local HDG blocks."""

    def __init__(
        self,
        host: DiffusionLocalAssemblyHostInputs,
        *,
        device_id: int = 0,
        inverse_backend: LocalInverseBackend = "cublas_inverse",
    ) -> None:
        cp = require_cupy_device()
        if inverse_backend not in {"cublas_inverse", "gpu_inverse"}:
            raise ValueError("inverse_backend must be 'cublas_inverse' or 'gpu_inverse'")
        self._cp = cp
        self.device_id = int(device_id)
        self.inverse_backend = inverse_backend
        self.dtype = cp.dtype(host.tau.dtype)
        self.num_elements = host.num_elements
        self.element_dofs = host.element_dofs
        self.face_dofs = host.face_dofs
        self.num_local_faces = 3
        self.interior_face_mask_host = np.ascontiguousarray(
            host.interior_face_mask, dtype=bool
        )

        with cp.cuda.Device(self.device_id):
            for name in (
                "tau", "reaction_mass", "d0_base", "d1_base", "aff_mats",
                "aff_jacs", "face_jacs", "normals",
                "oriented_trace_restriction", "face_element_trace",
                "face_element_mass", "face_mass", "mkrf_inv",
                "orientations", "interior_face_mask",
            ):
                value = getattr(host, name)
                setattr(self, name, cp.ascontiguousarray(cp.asarray(value)))
            self.workspace = CuPyDiffusionAssemblyWorkspace.allocate(
                num_elements=self.num_elements,
                element_dofs=self.element_dofs,
                dtype=self.dtype,
                device_id=self.device_id,
            )
            suffix = "f32" if self.dtype == cp.float32 else "f64"
            self._face_rhs_kernel = cp.RawKernel(
                _FACE_RHS_KERNEL_SOURCE, f"assemble_face_rhs_{suffix}"
            )

    @classmethod
    def from_space(
        cls,
        reaction: Any,
        stabilization: Any,
        space: Any,
        *,
        dtype: Any = np.float64,
        device_id: int = 0,
        inverse_backend: LocalInverseBackend = "cublas_inverse",
    ) -> "CuPyDiffusionLocalAssembler":
        return cls(
            prepare_diffusion_local_assembly_inputs(
                reaction, stabilization, space, dtype=dtype
            ),
            device_id=device_id,
            inverse_backend=inverse_backend,
        )

    @property
    def workspace_bytes(self) -> int:
        return self.workspace.nbytes

    def _matmul(self, first: Any, second: Any, out: Any) -> Any:
        self._cp.matmul(first, second, out=out)
        return out

    def _build_physical_matrices(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        cp = self._cp
        a = self.aff_mats
        d0 = (
            a[:, 1, 1, None, None] * self.d0_base[None]
            - a[:, 1, 0, None, None] * self.d1_base[None]
        )
        d1 = (
            -a[:, 0, 1, None, None] * self.d0_base[None]
            + a[:, 0, 0, None, None] * self.d1_base[None]
        )
        weighted_face_mass = self.face_jacs[..., None, None] * self.face_element_mass[None]
        m_tau = self.reaction_mass + cp.sum(
            self.tau[..., None, None] * weighted_face_mass, axis=1
        )
        m_n0 = cp.sum(
            self.normals[..., 0, None, None] * weighted_face_mass, axis=1
        )
        m_n1 = cp.sum(
            self.normals[..., 1, None, None] * weighted_face_mass, axis=1
        )
        jacs_inv = 1.0 / self.aff_jacs[:, None, None]
        return (
            cp.ascontiguousarray(d0), cp.ascontiguousarray(d1),
            cp.ascontiguousarray(m_tau), cp.ascontiguousarray(m_n0),
            cp.ascontiguousarray(m_n1), cp.ascontiguousarray(jacs_inv),
        )

    def _invert_scalar_matrix(self, matrix: Any) -> tuple[Any, Any, np.ndarray | None, np.ndarray | None]:
        cp = self._cp
        if self.inverse_backend == "cublas_inverse":
            result = invert_batched_cublas(
                cp.ascontiguousarray(matrix),
                label="diffusion scalar local matrices",
            )
            return (
                result.inverse_matrices,
                result.inverse_residuals,
                result.factorization_info,
                result.inversion_info,
            )
        inverse = cp.linalg.inv(matrix)
        identity = cp.eye(self.element_dofs, dtype=self.dtype)
        residuals = cp.max(
            cp.sum(cp.abs(cp.matmul(matrix, inverse) - identity[None]), axis=2),
            axis=1,
        )
        return cp.ascontiguousarray(inverse), cp.ascontiguousarray(residuals), None, None

    def assemble(
        self,
        *,
        retain_intermediates: bool = True,
    ) -> CuPyDiffusionLocalAssemblyResult:
        """Assemble complete elemental face blocks entirely on the GPU."""

        cp = self._cp
        n = self.element_dofs
        b = self.face_dofs
        ne = self.num_elements
        with cp.cuda.Device(self.device_id):
            d0, d1, m_tau, m_n0, m_n1, jacs_inv = self._build_physical_matrices()
            mn0 = cp.ascontiguousarray(m_n0 - d0)
            mn1 = cp.ascontiguousarray(m_n1 - d1)

            ping = self.workspace.ping
            pong = self.workspace.pong
            self._matmul(mn0, self.mkrf_inv, ping)
            self._matmul(ping, d0, pong)
            scalar_matrix = cp.ascontiguousarray(m_tau + jacs_inv * pong)
            self._matmul(mn1, self.mkrf_inv, ping)
            self._matmul(ping, d1, pong)
            scalar_matrix += jacs_inv * pong

            e, inverse_residuals, factor_info, inversion_info = self._invert_scalar_matrix(
                scalar_matrix
            )

            local_solver = cp.zeros((ne, 3 * n, 3 * n), dtype=self.dtype)
            blocks = local_solver.reshape(ne, 3, n, 3, n)
            blocks[:, 0, :, 0, :] = e
            j2 = jacs_inv * jacs_inv
            identity = cp.eye(n, dtype=self.dtype)[None]

            # The two square work arrays alternate for every product chain.
            self._matmul(e, mn0, ping)
            self._matmul(ping, self.mkrf_inv, pong)
            blocks[:, 0, :, 1, :] = jacs_inv * pong
            self._matmul(e, mn1, ping)
            self._matmul(ping, self.mkrf_inv, pong)
            blocks[:, 0, :, 2, :] = jacs_inv * pong

            self._matmul(d0, e, ping)
            self._matmul(self.mkrf_inv, ping, pong)
            blocks[:, 1, :, 0, :] = jacs_inv * pong
            self._matmul(ping, mn0, pong)
            self._matmul(pong, self.mkrf_inv, ping)
            blocks[:, 1, :, 1, :] = jacs_inv * (
                self.mkrf_inv[None] @ (-identity + jacs_inv * ping)
            )
            # Rebuild d0@e because ping was overwritten.
            self._matmul(d0, e, ping)
            self._matmul(ping, mn1, pong)
            self._matmul(pong, self.mkrf_inv, ping)
            blocks[:, 1, :, 2, :] = j2 * (self.mkrf_inv[None] @ ping)

            self._matmul(d1, e, ping)
            self._matmul(self.mkrf_inv, ping, pong)
            blocks[:, 2, :, 0, :] = jacs_inv * pong
            self._matmul(ping, mn0, pong)
            self._matmul(pong, self.mkrf_inv, ping)
            blocks[:, 2, :, 1, :] = j2 * (self.mkrf_inv[None] @ ping)
            self._matmul(d1, e, ping)
            self._matmul(ping, mn1, pong)
            self._matmul(pong, self.mkrf_inv, ping)
            blocks[:, 2, :, 2, :] = jacs_inv * (
                self.mkrf_inv[None] @ (-identity + jacs_inv * ping)
            )
            local_solver = cp.ascontiguousarray(local_solver)

            oriented = self.oriented_trace_restriction * self.face_jacs[..., None, None]
            trace_lift = cp.empty((ne, 3, b, 3 * n), dtype=self.dtype)
            trace_lift[..., :n] = self.tau[..., None, None] * oriented
            trace_lift[..., n:2 * n] = self.normals[..., 0, None, None] * oriented
            trace_lift[..., 2 * n:] = self.normals[..., 1, None, None] * oriented
            trace_lift = cp.ascontiguousarray(trace_lift)

            boundary = cp.zeros((ne, 3 * n, 3 * b), dtype=self.dtype)
            boundary_blocks = boundary.reshape(ne, 3, n, 3, b)
            weighted_trace = self.face_jacs[:, None, :, None] * self.face_element_trace[None]
            boundary_blocks[:, 0] = self.tau[:, None, :, None] * weighted_trace
            boundary_blocks[:, 1] = self.normals[..., 0][:, None, :, None] * weighted_trace
            boundary_blocks[:, 2] = self.normals[..., 1][:, None, :, None] * weighted_trace
            boundary = cp.ascontiguousarray(boundary)

            local_boundary = cp.matmul(local_solver, boundary)
            schur = cp.matmul(trace_lift, local_boundary[:, None, :, :])
            schur = schur.reshape(ne, 3, b, 3, b)
            # Reverse column trace basis for negatively oriented local faces.
            for local_face in range(3):
                mask = ~self.orientations[:, local_face]
                schur[mask, :, :, local_face, :] = schur[
                    mask, :, :, local_face, ::-1
                ]
            trace_blocks = cp.ascontiguousarray(schur.swapaxes(2, 3))

            element_blocks = -trace_blocks.copy()
            stabilization_mass = (
                (self.tau * self.face_jacs)[..., None, None]
                * self.face_mass[None, None, :, :]
            )
            faces = cp.arange(3)
            element_blocks[:, faces, faces] += stabilization_mass
            element_blocks = cp.ascontiguousarray(element_blocks)

            return CuPyDiffusionLocalAssemblyResult(
                element_blocks=element_blocks,
                trace_blocks=trace_blocks,
                local_solver=local_solver if retain_intermediates else None,
                trace_lift=trace_lift if retain_intermediates else None,
                element_boundary_mats=boundary if retain_intermediates else None,
                scalar_inverse_residuals=inverse_residuals,
                factorization_info=factor_info,
                inversion_info=inversion_info,
            )

    def assemble_interior_rhs(
        self,
        source_rhs: Any,
        result: CuPyDiffusionLocalAssemblyResult,
        *,
        topology: FaceTopology,
    ) -> Any:
        """Assemble the unpenalized global trace RHS on the GPU.

        ``source_rhs`` has shape ``(NE, 3*element_dofs)`` and may be a NumPy
        or CuPy array. Boundary face rows are left equal to zero; projected
        Dirichlet values can subsequently be imposed by penalty replacement or
        direct elimination.
        """

        cp = self._cp
        if result.local_solver is None or result.trace_lift is None:
            raise ValueError(
                "assemble_interior_rhs requires retain_intermediates=True"
            )
        with cp.cuda.Device(self.device_id):
            source = cp.ascontiguousarray(cp.asarray(source_rhs, dtype=self.dtype))
            expected = (self.num_elements, 3 * self.element_dofs)
            if source.shape != expected:
                raise ValueError(f"source_rhs must have shape {expected}")
            local_solution = cp.matmul(result.local_solver, source[..., None])
            local_face_rhs = cp.matmul(
                result.trace_lift, local_solution[:, None, :, :]
            ).squeeze(-1)
            local_face_rhs = cp.ascontiguousarray(local_face_rhs)

            element_ids = np.asarray(topology.adjacent_elements, dtype=np.int32).copy()
            local_faces = np.asarray(
                topology.adjacent_local_faces, dtype=np.int32
            ).copy()
            for face in range(topology.num_faces):
                for side in range(2):
                    element = int(element_ids[face, side])
                    local_face = int(local_faces[face, side])
                    if element < 0:
                        continue
                    if not bool(
                        self.interior_face_mask_host[element, local_face]
                    ):
                        element_ids[face, side] = -1
                        local_faces[face, side] = -1
            element_ids_device = cp.asarray(
                np.ascontiguousarray(element_ids), dtype=cp.int32
            )
            local_faces_device = cp.asarray(
                np.ascontiguousarray(local_faces), dtype=cp.int32
            )
            out = cp.empty((topology.num_faces, self.face_dofs), dtype=self.dtype)
            total = int(out.size)
            threads = 256
            blocks = (total + threads - 1) // threads
            self._face_rhs_kernel(
                (blocks,),
                (threads,),
                (
                    np.uint64(total),
                    np.int32(self.face_dofs),
                    element_ids_device,
                    local_faces_device,
                    local_face_rhs,
                    out,
                ),
            )
            return out

    def assemble_global_blocks(
        self,
        result: CuPyDiffusionLocalAssemblyResult,
        *,
        loc2glob_face: np.ndarray,
        topology: FaceTopology,
        active_row_faces: np.ndarray | None = None,
    ) -> Any:
        """Assemble device elemental blocks into global face-dense rows."""

        assembler = CuPyGlobalFaceAssembler.from_topology(
            loc2glob_face,
            topology,
            block_size=self.face_dofs,
            dtype=self.dtype,
            device_id=self.device_id,
            active_row_faces=active_row_faces,
        )
        return assembler.assemble(result.element_blocks)


__all__ = [
    "CuPyDiffusionAssemblyWorkspace",
    "CuPyDiffusionLocalAssembler",
    "CuPyDiffusionLocalAssemblyResult",
    "DiffusionLocalAssemblyHostInputs",
    "prepare_diffusion_local_assembly_inputs",
]
