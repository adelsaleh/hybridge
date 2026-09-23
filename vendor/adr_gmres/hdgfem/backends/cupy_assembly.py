"""CUDA assembly of global face-dense blocks from elemental HDG blocks.

This is the first GPU-assembly layer.  The complete elemental trace blocks are
assumed to exist already, with shape ``(NE, Nlfe, Nlfe, b, b)``.  The module
assembles their numerical contributions directly into the global fixed-width
face rows without COO/CSR materialization and without atomics.

For a manifold triangular mesh, a global row face belongs to at most two
elements.  A host-side setup therefore builds a fixed contribution table with
at most two elemental sources for every global ``(row_face, slot)`` block.  A
face-parallel CUDA kernel then gathers and sums these sources.  This preserves
bitwise-deterministic contribution order and avoids the races of a generic
scatter-add kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hdgfem.assembly.face_dense import FaceTopology
from hdgfem.backends.cupy import device_arrays_overlap, require_cupy_device


@dataclass(frozen=True)
class FaceAssemblyContributionLayout:
    """Host contribution maps for deterministic GPU global assembly.

    Arrays have shape ``(NF, S, 2)``.  The last axis enumerates the one or two
    incident elemental contributions.  Unused entries are ``-1``.
    """

    element_ids: np.ndarray
    row_local_faces: np.ndarray
    column_local_faces: np.ndarray
    num_elements: int
    num_local_faces: int

    @property
    def num_faces(self) -> int:
        return int(self.element_ids.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.element_ids.shape[1])

    @property
    def maximum_contributions(self) -> int:
        return int(self.element_ids.shape[2])


def prepare_face_assembly_contributions(
    loc2glob_face: np.ndarray,
    topology: FaceTopology,
    *,
    active_row_faces: np.ndarray | None = None,
) -> FaceAssemblyContributionLayout:
    """Build deterministic gather maps for GPU global face assembly."""

    loc2glob_face = np.ascontiguousarray(loc2glob_face, dtype=np.int64)
    if loc2glob_face.ndim != 2 or loc2glob_face.size == 0:
        raise ValueError("loc2glob_face must have shape (NE, Nlfe) and be non-empty")
    num_elements, num_local_faces = loc2glob_face.shape
    if np.min(loc2glob_face) < 0:
        raise ValueError("loc2glob_face cannot contain negative ids")
    if topology.element_face_slots.shape != (
        num_elements,
        num_local_faces,
        num_local_faces,
    ):
        raise ValueError("topology is incompatible with loc2glob_face")
    if topology.neighbors.shape[0] <= int(np.max(loc2glob_face)):
        raise ValueError("topology does not contain every global face")

    if active_row_faces is None:
        active = np.ones((num_elements, num_local_faces), dtype=bool)
    else:
        active = np.asarray(active_row_faces, dtype=bool)
        if active.shape != (num_elements, num_local_faces):
            raise ValueError(
                "active_row_faces must have shape "
                f"({num_elements}, {num_local_faces})"
            )

    if max(num_elements, num_local_faces, topology.num_faces, topology.num_slots) > np.iinfo(np.int32).max:
        raise OverflowError("CUDA assembly maps use int32 indices")

    shape = (topology.num_faces, topology.num_slots, 2)
    element_ids = np.full(shape, -1, dtype=np.int32)
    row_local_faces = np.full(shape, -1, dtype=np.int32)
    column_local_faces = np.full(shape, -1, dtype=np.int32)
    counts = np.zeros((topology.num_faces, topology.num_slots), dtype=np.int8)

    for element in range(num_elements):
        for row_local_face in range(num_local_faces):
            if not active[element, row_local_face]:
                continue
            row_face = int(loc2glob_face[element, row_local_face])
            for column_local_face in range(num_local_faces):
                slot = int(
                    topology.element_face_slots[
                        element, row_local_face, column_local_face
                    ]
                )
                contribution = int(counts[row_face, slot])
                if contribution >= 2:
                    raise ValueError(
                        "a global face block receives more than two elemental "
                        "contributions; the manifold gather layout is invalid"
                    )
                element_ids[row_face, slot, contribution] = element
                row_local_faces[row_face, slot, contribution] = row_local_face
                column_local_faces[row_face, slot, contribution] = column_local_face
                counts[row_face, slot] += 1

    return FaceAssemblyContributionLayout(
        element_ids=np.ascontiguousarray(element_ids),
        row_local_faces=np.ascontiguousarray(row_local_faces),
        column_local_faces=np.ascontiguousarray(column_local_faces),
        num_elements=int(num_elements),
        num_local_faces=int(num_local_faces),
    )


_GLOBAL_FACE_ASSEMBLY_KERNEL_SOURCE = r"""
extern "C" __global__
void assemble_global_face_blocks_f32(
    const unsigned long long total,
    const int num_slots,
    const int num_local_faces,
    const int block_size,
    const int* __restrict__ element_ids,
    const int* __restrict__ row_local_faces,
    const int* __restrict__ column_local_faces,
    const float* __restrict__ element_blocks,
    float* __restrict__ global_blocks)
{
    const unsigned long long i =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;

    const int column_dof = (int) (i % block_size);
    const int row_dof = (int) ((i / block_size) % block_size);
    const unsigned long long face_slot =
        i / ((unsigned long long) block_size * block_size);
    const int slot = (int) (face_slot % num_slots);
    const int row_face = (int) (face_slot / num_slots);
    const unsigned long long map_offset =
        ((unsigned long long) row_face * num_slots + slot) * 2;

    float value = 0.0f;
    #pragma unroll
    for (int contribution = 0; contribution < 2; ++contribution) {
        const int element = element_ids[map_offset + contribution];
        if (element < 0) continue;
        const int row_local = row_local_faces[map_offset + contribution];
        const int column_local = column_local_faces[map_offset + contribution];
        const unsigned long long source =
            (((((unsigned long long) element * num_local_faces + row_local)
                * num_local_faces + column_local)
              * block_size + row_dof)
             * block_size + column_dof);
        value += element_blocks[source];
    }
    global_blocks[i] = value;
}

extern "C" __global__
void assemble_global_face_blocks_f64(
    const unsigned long long total,
    const int num_slots,
    const int num_local_faces,
    const int block_size,
    const int* __restrict__ element_ids,
    const int* __restrict__ row_local_faces,
    const int* __restrict__ column_local_faces,
    const double* __restrict__ element_blocks,
    double* __restrict__ global_blocks)
{
    const unsigned long long i =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;

    const int column_dof = (int) (i % block_size);
    const int row_dof = (int) ((i / block_size) % block_size);
    const unsigned long long face_slot =
        i / ((unsigned long long) block_size * block_size);
    const int slot = (int) (face_slot % num_slots);
    const int row_face = (int) (face_slot / num_slots);
    const unsigned long long map_offset =
        ((unsigned long long) row_face * num_slots + slot) * 2;

    double value = 0.0;
    #pragma unroll
    for (int contribution = 0; contribution < 2; ++contribution) {
        const int element = element_ids[map_offset + contribution];
        if (element < 0) continue;
        const int row_local = row_local_faces[map_offset + contribution];
        const int column_local = column_local_faces[map_offset + contribution];
        const unsigned long long source =
            (((((unsigned long long) element * num_local_faces + row_local)
                * num_local_faces + column_local)
              * block_size + row_dof)
             * block_size + column_dof);
        value += element_blocks[source];
    }
    global_blocks[i] = value;
}
"""


class CuPyGlobalFaceAssembler:
    """Reusable deterministic CUDA assembler for global face blocks."""

    def __init__(
        self,
        *,
        element_ids: Any,
        row_local_faces: Any,
        column_local_faces: Any,
        num_elements: int,
        num_local_faces: int,
        block_size: int,
        dtype: Any,
        device_id: int,
    ) -> None:
        cp = require_cupy_device()
        arrays = (element_ids, row_local_faces, column_local_faces)
        if any(not isinstance(array, cp.ndarray) for array in arrays):
            raise TypeError("contribution maps must be CuPy arrays")
        if not (element_ids.shape == row_local_faces.shape == column_local_faces.shape):
            raise ValueError("contribution maps must have equal shapes")
        if element_ids.ndim != 3 or element_ids.shape[2] != 2:
            raise ValueError("contribution maps must have shape (NF, S, 2)")
        if any(array.dtype != cp.int32 for array in arrays):
            raise TypeError("contribution maps must use int32")
        if any(not array.flags.c_contiguous for array in arrays):
            raise ValueError("contribution maps must be C-contiguous")
        dtype = cp.dtype(dtype)
        if dtype not in (cp.float32, cp.float64):
            raise TypeError("dtype must be float32 or float64")
        device_id = int(device_id)
        if any(int(array.device.id) != device_id for array in arrays):
            raise ValueError("contribution maps are on the wrong CUDA device")

        self._cp = cp
        self.element_ids = element_ids
        self.row_local_faces = row_local_faces
        self.column_local_faces = column_local_faces
        self.num_elements = int(num_elements)
        self.num_local_faces = int(num_local_faces)
        self.num_faces = int(element_ids.shape[0])
        self.num_slots = int(element_ids.shape[1])
        self.block_size = int(block_size)
        self.dtype = dtype
        self.device_id = device_id
        if min(self.num_elements, self.num_local_faces, self.num_faces, self.num_slots, self.block_size) <= 0:
            raise ValueError("assembly dimensions must be positive")

        suffix = "f32" if dtype == cp.float32 else "f64"
        self._kernel = cp.RawKernel(
            _GLOBAL_FACE_ASSEMBLY_KERNEL_SOURCE,
            f"assemble_global_face_blocks_{suffix}",
        )

    @classmethod
    def from_topology(
        cls,
        loc2glob_face: np.ndarray,
        topology: FaceTopology,
        *,
        block_size: int,
        dtype: Any = np.float64,
        active_row_faces: np.ndarray | None = None,
        device_id: int | None = None,
    ) -> "CuPyGlobalFaceAssembler":
        cp = require_cupy_device()
        layout = prepare_face_assembly_contributions(
            loc2glob_face,
            topology,
            active_row_faces=active_row_faces,
        )
        selected_device = int(cp.cuda.Device().id) if device_id is None else int(device_id)
        with cp.cuda.Device(selected_device):
            return cls(
                element_ids=cp.asarray(layout.element_ids),
                row_local_faces=cp.asarray(layout.row_local_faces),
                column_local_faces=cp.asarray(layout.column_local_faces),
                num_elements=layout.num_elements,
                num_local_faces=layout.num_local_faces,
                block_size=int(block_size),
                dtype=cp.dtype(dtype),
                device_id=selected_device,
            )

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_faces,
            self.num_slots,
            self.block_size,
            self.block_size,
        )

    @property
    def mapping_bytes(self) -> int:
        return sum(
            int(array.nbytes)
            for array in (
                self.element_ids,
                self.row_local_faces,
                self.column_local_faces,
            )
        )

    def _validate_element_blocks(self, element_blocks: Any) -> Any:
        cp = self._cp
        if not isinstance(element_blocks, cp.ndarray):
            raise TypeError("element_blocks must be a CuPy array")
        expected = (
            self.num_elements,
            self.num_local_faces,
            self.num_local_faces,
            self.block_size,
            self.block_size,
        )
        if element_blocks.shape != expected:
            raise ValueError(f"element_blocks must have shape {expected}")
        if element_blocks.dtype != self.dtype:
            raise TypeError(f"element_blocks must have dtype {self.dtype}")
        if int(element_blocks.device.id) != self.device_id:
            raise ValueError("element_blocks are on the wrong CUDA device")
        if not element_blocks.flags.c_contiguous:
            raise ValueError("element_blocks must be C-contiguous")
        return element_blocks

    def _validate_output(self, out: Any) -> Any:
        cp = self._cp
        if not isinstance(out, cp.ndarray):
            raise TypeError("out must be a CuPy array")
        if out.shape != self.output_shape:
            raise ValueError(f"out must have shape {self.output_shape}")
        if out.dtype != self.dtype:
            raise TypeError(f"out must have dtype {self.dtype}")
        if int(out.device.id) != self.device_id:
            raise ValueError("out is on the wrong CUDA device")
        if not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")
        return out

    def assemble_into(self, element_blocks: Any, out: Any) -> None:
        """Assemble a device element-block batch into preallocated face rows."""

        element_blocks = self._validate_element_blocks(element_blocks)
        out = self._validate_output(out)
        if device_arrays_overlap(element_blocks, out):
            raise ValueError("element_blocks and out must not overlap")
        total = int(np.prod(self.output_shape, dtype=np.int64))
        threads = 256
        blocks = (total + threads - 1) // threads
        self._kernel(
            (blocks,),
            (threads,),
            (
                np.uint64(total),
                np.int32(self.num_slots),
                np.int32(self.num_local_faces),
                np.int32(self.block_size),
                self.element_ids,
                self.row_local_faces,
                self.column_local_faces,
                element_blocks,
                out,
            ),
        )

    def assemble(self, element_blocks: Any) -> Any:
        out = self._cp.empty(self.output_shape, dtype=self.dtype)
        self.assemble_into(element_blocks, out)
        return out


__all__ = [
    "CuPyGlobalFaceAssembler",
    "FaceAssemblyContributionLayout",
    "prepare_face_assembly_contributions",
]
