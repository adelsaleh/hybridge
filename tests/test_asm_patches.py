"""Pure NumPy topology checks for expanded guiding-center ASM patches."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.guiding_center.poisson.asm_patches import (
    build_element_neighbor_patches,
    build_face_pair_patches,
)


BUILDERS = (build_element_neighbor_patches, build_face_pair_patches)
PAIR = np.asarray([[0, -1, -1], [0, -1, -1]], dtype=np.int64)
FAN = np.asarray(
    [[0, 1, 2], [0, 3, -1], [1, 4, -1], [2, 5, -1], [3, 4, 5]],
    dtype=np.int64,
)
NINE = np.asarray(
    [[0, 1, 2], [0, 3, 4], [1, 5, 6], [2, 7, 8],
     [3, -1, -1], [4, -1, -1], [5, -1, -1],
     [6, -1, -1], [7, -1, -1], [8, -1, -1]],
    dtype=np.int64,
)


def _set_oracle(faces, *, element_neighbors):
    active = [set(row[row >= 0].tolist()) for row in faces]
    count = max(set.union(*active)) + 1
    incidence = [{e for e, patch in enumerate(active) if f in patch} for f in range(count)]
    if element_neighbors:
        return [set.union(patch, *(active[n] for f in patch for n in incidence[f]))
                for patch in active]
    return [set.union(*(active[e] for e in elements)) for elements in incidence]


@pytest.mark.parametrize("faces", (PAIR, FAN, NINE, np.vstack((PAIR, [-1, -1, -1]))))
@pytest.mark.parametrize("builder", BUILDERS)
def test_expanded_patches_match_independent_set_unions_and_cover_every_free_face(faces, builder):
    count = int(faces.max()) + 1
    expected = _set_oracle(faces, element_neighbors=builder is build_element_neighbor_patches)
    result = builder(faces, count)
    width = 9 if builder is build_element_neighbor_patches else 5
    assert result.shape == (len(expected), width)
    assert result.dtype == np.int32
    assert result.flags.c_contiguous
    for row, active in zip(result, expected):
        np.testing.assert_array_equal(row[row >= 0], sorted(active))
        assert np.all(row[len(active):] == -1)
    np.testing.assert_array_equal(np.unique(result[result >= 0]), np.arange(count))
    # Every original element/central face remains in its own patch.
    if builder is build_element_neighbor_patches:
        assert all(set(row[row >= 0]).issubset(patch) for row, patch in zip(faces, expected))
    else:
        assert all(face in patch for face, patch in enumerate(expected))


def test_patch_width_limits_are_attained_on_an_interior_triangle_and_face():
    elements = build_element_neighbor_patches(NINE, 9)
    pairs = build_face_pair_patches(NINE, 9)
    np.testing.assert_array_equal(elements[0], np.arange(9))
    np.testing.assert_array_equal(pairs[0], np.arange(5))


@pytest.mark.parametrize("builder", BUILDERS)
def test_patch_numbering_is_independent_of_local_face_order(builder):
    expected = builder(FAN, 6)
    permuted = FAN[:, [2, 0, 1]].copy()
    np.testing.assert_array_equal(builder(permuted, 6), expected)


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize(
    ("faces", "count", "error"),
    (
        ([[0, -1, -1]], 1, ValueError),                 # Only one incident element.
        ([[0, -1, -1]] * 3, 1, ValueError),             # Non-manifold incidence.
        (PAIR, 2, ValueError),                          # Missing free-face row 1.
        ([[0, 0, -1]], 1, ValueError),                  # Same element twice.
        ([[0, -2, -1], [0, -1, -1]], 1, ValueError),
        ([[1, -1, -1], [1, -1, -1]], 1, ValueError),
        (np.zeros((2, 2), dtype=np.int64), 1, ValueError),
        (np.empty((0, 3), dtype=np.int64), 1, ValueError),
        (PAIR.astype(np.float64), 1, TypeError),
        (PAIR, 0, ValueError),
    ),
)
def test_patch_maps_reject_invalid_or_incomplete_incidence(builder, faces, count, error):
    with pytest.raises(error):
        builder(faces, count)
