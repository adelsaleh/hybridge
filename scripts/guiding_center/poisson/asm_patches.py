"""Host-only overlapping trace patches for captured guiding-center systems.

The input is the exact element-to-reduced-face map of a triangular mesh with
Dirichlet boundary faces eliminated. These diagnostic extensions preserve its
free-face numbering and use the package's shared face incidence construction.
"""

from __future__ import annotations

import operator

import numpy as np

from hdgfem.backends.cupy_preconditionners import (
    build_face_additive_schwarz_incidence_slots,
)


def _validated_incidence(element_system_faces, num_system_faces):
    faces = np.asarray(element_system_faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError("element_system_faces must have nonempty shape (NE, 3)")
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("element_system_faces must contain integer face IDs")
    count = operator.index(num_system_faces)
    if count <= 0:
        raise ValueError("num_system_faces must be positive")
    if count > np.iinfo(np.int32).max:
        raise OverflowError("trace patch maps require int32 face IDs")
    if np.any(faces < -1) or np.any(faces >= count):
        raise ValueError("element_system_faces contains an out-of-range face ID")
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    ordered = np.sort(faces, axis=1)
    if np.any((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])):
        raise ValueError("an element must not contain duplicate active face IDs")
    slots = build_face_additive_schwarz_incidence_slots(faces, count)
    if np.any(slots < 0):
        raise ValueError("every free face must have exactly two incident elements")
    return faces, slots // 3, count


def _sorted_unique_rows(candidates, *, width, num_system_faces):
    # A sentinel larger than every active ID sorts padding to the row end.
    ordered = np.sort(np.where(candidates >= 0, candidates, num_system_faces), axis=1)
    unique = ordered != num_system_faces
    unique[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    counts = np.sum(unique, axis=1)
    if np.any(counts > width):
        raise ValueError(f"trace patch exceeds its maximum width of {width}")
    result = np.full((candidates.shape[0], width), -1, dtype=np.int32)
    rows, columns = np.nonzero(unique)
    positions = np.cumsum(unique, axis=1) - 1
    result[rows, positions[rows, columns]] = ordered[rows, columns]
    return result


def build_element_neighbor_patches(element_system_faces, num_system_faces):
    """Return each triangle's traces plus its neighbors' traces as ``(NE, 9)``.

    Active IDs are sorted and unique; trailing ``-1`` entries pad shorter
    boundary patches. A neighbor is the other element sharing one of the
    triangle's free faces. Eliminated boundary faces have no neighbor.
    """
    faces, incidence, count = _validated_incidence(element_system_faces, num_system_faces)
    paired_elements = incidence[np.maximum(faces, 0)]
    own_element = np.arange(faces.shape[0])[:, None]
    neighbors = np.where(
        paired_elements[:, :, 0] == own_element,
        paired_elements[:, :, 1],
        paired_elements[:, :, 0],
    )
    neighbors[faces < 0] = faces.shape[0]
    padded_faces = np.concatenate((faces, np.full((1, 3), -1, dtype=faces.dtype)))
    candidates = np.concatenate(
        (faces, padded_faces[neighbors].reshape(faces.shape[0], 9)), axis=1,
    )
    return _sorted_unique_rows(candidates, width=9, num_system_faces=count)


def build_face_pair_patches(element_system_faces, num_system_faces):
    """Return both incident triangles' traces for each free face as ``(NF, 5)``.

    Output row ``f`` belongs to input free face ``f``. Active IDs are sorted
    and unique, followed by ``-1`` padding near eliminated boundaries.
    """
    faces, incidence, count = _validated_incidence(element_system_faces, num_system_faces)
    candidates = faces[incidence].reshape(count, 6)
    return _sorted_unique_rows(candidates, width=5, num_system_faces=count)


__all__ = ["build_element_neighbor_patches", "build_face_pair_patches"]
