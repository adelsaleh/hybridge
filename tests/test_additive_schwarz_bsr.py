
"""CPU algebra checks for assembling element ASM directly from BSR storage."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import bsr_matrix

from hdgfem.assembly.face_dense import (
    assemble_global_face_blocks,
    build_face_topology,
    eliminate_dirichlet_faces,
)
from hdgfem.linalg.face_dense import materialize_face_dense_matrix
from hdgfem.linalg.additive_schwarz import (
    assemble_bsr_additive_schwarz_correction,
    assemble_bsr_face_additive_schwarz_correction,
    build_bsr_additive_schwarz_local_matrices,
    build_bsr_face_additive_schwarz_local_matrices,
    build_face_additive_schwarz_preconditioner,
)


def _triangle_problem(block_size: int, *, permuted: bool = False, cells: int = 2):
    # A square triangulation, constructed without mesh kernels or JIT.
    triangles = []
    stride = cells + 1
    for y in range(cells):
        for x in range(cells):
            a = stride * y + x
            triangles.extend(((a, a + 1, a + stride + 1), (a, a + stride + 1, a + stride)))
    edge_ids = {}
    element_faces = []
    for triangle in triangles:
        faces = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((triangle[i], triangle[j])))
            faces.append(edge_ids.setdefault(edge, len(edge_ids)))
        element_faces.append(faces)
    # An isolated triangle exercises a patch with no active reduced faces.
    element_faces.append(list(range(len(edge_ids), len(edge_ids) + 3)))
    element_faces = np.asarray(element_faces, dtype=np.int64)
    topology = build_face_topology(element_faces)
    rng = np.random.default_rng(791 + block_size)
    nlocal = 3 * block_size
    factors = rng.standard_normal((len(element_faces), nlocal, nlocal))
    elemental = factors.transpose(0, 2, 1) @ factors + 2.0 * np.eye(nlocal)
    element_blocks = elemental.reshape(-1, 3, block_size, 3, block_size).transpose(
        0, 1, 3, 2, 4
    )
    global_blocks = assemble_global_face_blocks(
        element_blocks, element_faces, topology
    )
    free_faces = np.flatnonzero(topology.incidence_count == 2)
    if permuted:
        free_faces = rng.permutation(free_faces)
    zero = np.zeros((topology.num_faces, block_size))
    system = eliminate_dirichlet_faces(
        global_blocks, topology, zero, zero, free_faces
    )
    matrix = bsr_matrix(
        materialize_face_dense_matrix(system.blocks, system.neighbors),
        blocksize=(block_size, block_size),
    )
    reference = build_face_additive_schwarz_preconditioner(
        system, element_blocks, element_faces
    )
    return matrix, reference


@pytest.mark.parametrize("block_size", [2, 7])
@pytest.mark.parametrize("permuted", [False, True])
def test_bsr_asm_matches_existing_element_asm(block_size, permuted):
    matrix, reference = _triangle_problem(block_size, permuted=permuted)
    original_data = matrix.data.copy()
    local = build_bsr_face_additive_schwarz_local_matrices(
        matrix, reference.element_system_faces
    )
    # This checks assembled face diagonals and all eliminated identity padding.
    np.testing.assert_array_equal(local.local_matrices, reference.local_matrices)
    np.testing.assert_array_equal(
        local.element_system_faces, reference.element_system_faces
    )
    np.testing.assert_array_equal(local.local_matrices[-1], np.eye(3 * block_size))
    inverses = np.linalg.inv(local.local_matrices)
    correction = assemble_bsr_face_additive_schwarz_correction(
        matrix, local, inverses
    )
    assert correction.blocksize == matrix.blocksize
    np.testing.assert_array_equal(correction.indptr, matrix.indptr)
    np.testing.assert_array_equal(correction.indices, matrix.indices)
    np.testing.assert_array_equal(matrix.data, original_data)

    rng = np.random.default_rng(721)
    vector = rng.standard_normal(matrix.shape[0])
    np.testing.assert_allclose(
        correction @ vector, reference.apply(vector), rtol=3e-14, atol=3e-14
    )
    # Independent restriction / principal solve / prolongation oracle.
    expected = np.zeros_like(vector)
    dense_matrix = matrix.toarray()
    for faces in local.element_system_faces:
        active_faces = faces[faces >= 0]
        dofs = (active_faces[:, None] * block_size + np.arange(block_size)).ravel()
        if dofs.size:
            np.add.at(
                expected,
                dofs,
                np.linalg.solve(dense_matrix[np.ix_(dofs, dofs)], vector[dofs]),
            )
    np.testing.assert_allclose(correction @ vector, expected, rtol=3e-14, atol=3e-14)
    dense_correction = correction.toarray()
    np.testing.assert_allclose(
        dense_correction, dense_correction.T, rtol=3e-14, atol=3e-14
    )
    assert np.linalg.eigvalsh(dense_correction)[0] > 0.0

    # Each active face occurs twice; in particular its two inverse diagonal
    # contributions must add rather than overwrite one another.
    for face in range(matrix.shape[0] // block_size):
        entries = np.argwhere(local.element_system_faces == face)
        assert len(entries) == 2
        expected_diagonal = sum(
            inverses[e, lf * block_size : (lf + 1) * block_size,
                     lf * block_size : (lf + 1) * block_size]
            for e, lf in entries
        )
        dofs = slice(face * block_size, (face + 1) * block_size)
        np.testing.assert_allclose(
            dense_correction[dofs, dofs], expected_diagonal, rtol=0.0, atol=0.0
        )


def test_bsr_asm_excludes_all_eliminated_inverse_entries():
    matrix, reference = _triangle_problem(2, permuted=True)
    local = build_bsr_face_additive_schwarz_local_matrices(
        matrix, reference.element_system_faces
    )
    inverses = np.linalg.inv(local.local_matrices)
    expected = assemble_bsr_face_additive_schwarz_correction(matrix, local, inverses)
    # Deliberately contaminate the unused part of supplied local matrices:
    # output must depend solely on R_e.T P_e^-1 R_e.
    for element, faces in enumerate(local.element_system_faces):
        for local_face in np.flatnonzero(faces < 0):
            dofs = slice(2 * local_face, 2 * (local_face + 1))
            inverses[element, dofs, :] = 193.0
            inverses[element, :, dofs] = -471.0
    actual = assemble_bsr_face_additive_schwarz_correction(matrix, local, inverses)
    np.testing.assert_array_equal(actual.data, expected.data)


def _replace_row(matrix, row, columns, data):
    begin, end = matrix.indptr[row : row + 2]
    indptr = matrix.indptr.copy()
    indptr[row + 1 :] += len(columns) - (end - begin)
    return bsr_matrix(
        (
            np.concatenate((matrix.data[:begin], data, matrix.data[end:])),
            np.concatenate((matrix.indices[:begin], columns, matrix.indices[end:])),
            indptr,
        ),
        shape=matrix.shape,
    )


@pytest.mark.parametrize("defect", ["unsorted", "duplicate", "missing", "extra"])
def test_bsr_asm_rejects_incompatible_block_patterns(defect):
    matrix, reference = _triangle_problem(2)
    if defect in ("unsorted", "duplicate", "missing"):
        row = int(np.argmax(np.diff(matrix.indptr)))
        begin, end = matrix.indptr[row : row + 2]
        columns = matrix.indices[begin:end].copy()
        data = matrix.data[begin:end].copy()
        if defect == "unsorted":
            columns, data = columns[::-1], data[::-1]
            message = "sorted BSR indices"
        elif defect == "duplicate":
            columns = np.insert(columns, 0, columns[0])
            data = np.concatenate((data[:1], data))
            message = "sorted BSR indices"
        else:
            remove = int(np.flatnonzero(columns != row)[0])
            columns = np.delete(columns, remove)
            data = np.delete(data, remove, axis=0)
            message = "missing an active element face pair"
    else:
        row = 0
        begin, end = matrix.indptr[row : row + 2]
        columns = matrix.indices[begin:end].copy()
        data = matrix.data[begin:end].copy()
        absent = np.setdiff1d(np.arange(len(matrix.indptr) - 1), columns)[0]
        insertion = np.searchsorted(columns, absent)
        columns = np.insert(columns, insertion, absent)
        # A stored zero outside all cliques must still fail the pattern gate.
        data = np.insert(data, insertion, np.zeros((2, 2)), axis=0)
        message = "outside the element face pairs"
    malformed = _replace_row(matrix, row, columns, data)
    with pytest.raises(ValueError, match=message):
        build_bsr_face_additive_schwarz_local_matrices(
            malformed, reference.element_system_faces
        )


def test_bsr_asm_preserves_structural_zero_blocks():
    matrix, reference = _triangle_problem(2)
    matrix.data[0] = 0.0
    local = build_bsr_face_additive_schwarz_local_matrices(
        matrix, reference.element_system_faces
    )
    correction = assemble_bsr_face_additive_schwarz_correction(
        matrix, local, np.broadcast_to(np.eye(6), local.local_matrices.shape)
    )
    np.testing.assert_array_equal(correction.indices, matrix.indices)
    np.testing.assert_array_equal(correction.indptr, matrix.indptr)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bsr_asm_accepts_real_floating_point_storage(dtype):
    matrix, reference = _triangle_problem(2)
    matrix = matrix.astype(dtype)
    local = build_bsr_face_additive_schwarz_local_matrices(
        matrix, reference.element_system_faces
    )
    inverse = np.linalg.inv(local.local_matrices).astype(dtype)
    correction = assemble_bsr_face_additive_schwarz_correction(matrix, local, inverse)
    assert local.local_matrices.dtype == np.float64
    assert correction.dtype == dtype
    np.testing.assert_allclose(
        correction.toarray(), correction.toarray().T,
        rtol=2e-6 if dtype == np.float32 else 3e-14,
        atol=2e-6 if dtype == np.float32 else 3e-14,
    )


def _wider_patches(reference, width):
    element_faces = reference.element_system_faces
    patches = np.full((len(element_faces), width), -1, dtype=np.int64)
    for element, faces in enumerate(element_faces):
        active = faces[faces >= 0]
        neighbors = [
            other for other, other_faces in enumerate(element_faces)
            if other != element and np.any(np.isin(other_faces[other_faces >= 0], active))
        ]
        if width == 5:
            neighbors = neighbors[:1]
        union = np.unique(element_faces[[element, *neighbors]].ravel())
        union = union[union >= 0]
        assert len(union) <= width
        patches[element, :len(union)] = union
    return patches


@pytest.mark.parametrize("block_size", [2, 7])
@pytest.mark.parametrize("width", [5, 9])
def test_wider_bsr_asm_matches_dense_principal_solve_and_retains_fill(block_size, width):
    matrix, reference = _triangle_problem(block_size, permuted=True, cells=4)
    patches = _wider_patches(reference, width)
    local = build_bsr_additive_schwarz_local_matrices(matrix, patches)
    assert local.num_local_faces == width
    assert local.num_elements == len(patches)
    dense_matrix = matrix.toarray()
    explicit = np.zeros_like(dense_matrix)
    expected_graph = np.zeros((local.num_system_faces, local.num_system_faces), dtype=bool)
    for patch, faces in enumerate(patches):
        active = faces >= 0
        global_dofs = (
            faces[active, None] * block_size + np.arange(block_size)
        ).ravel()
        local_dofs = (
            np.flatnonzero(active)[:, None] * block_size + np.arange(block_size)
        ).ravel()
        padded = np.flatnonzero(np.repeat(~active, block_size))
        expected_patch = np.zeros_like(local.local_matrices[patch])
        expected_patch[padded, padded] = 1.0
        expected_patch[np.ix_(local_dofs, local_dofs)] = dense_matrix[
            np.ix_(global_dofs, global_dofs)
        ]
        np.testing.assert_array_equal(local.local_matrices[patch], expected_patch)
        if global_dofs.size:
            explicit[np.ix_(global_dofs, global_dofs)] += np.linalg.solve(
                dense_matrix[np.ix_(global_dofs, global_dofs)],
                np.eye(global_dofs.size),
            )
            expected_graph[np.ix_(faces[active], faces[active])] = True

    inverses = np.linalg.inv(local.local_matrices)
    correction = assemble_bsr_additive_schwarz_correction(matrix, local, inverses)
    assert correction.blocksize == matrix.blocksize
    assert correction.has_canonical_format
    assert correction.data.shape[0] > matrix.data.shape[0]
    dense_correction = correction.toarray()
    np.testing.assert_allclose(dense_correction, explicit, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(dense_correction, dense_correction.T, rtol=2e-13, atol=2e-14)
    assert np.linalg.eigvalsh(dense_correction)[0] > 0.0
    actual_graph = np.zeros_like(expected_graph)
    for row in range(local.num_system_faces):
        actual_graph[row, correction.indices[correction.indptr[row]:correction.indptr[row + 1]]] = True
    np.testing.assert_array_equal(actual_graph, expected_graph)
    rng = np.random.default_rng(width + block_size)
    vector = rng.standard_normal(matrix.shape[0])
    np.testing.assert_allclose(correction @ vector, explicit @ vector, rtol=2e-13, atol=2e-14)
    # At least one newly introduced block must carry numeric inverse fill.
    has_numeric_fill = False
    for row in range(local.num_system_faces):
        old_columns = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
        for slot in range(correction.indptr[row], correction.indptr[row + 1]):
            if correction.indices[slot] not in old_columns:
                has_numeric_fill |= np.linalg.norm(correction.data[slot]) > 1e-16
    assert has_numeric_fill


def test_generic_bsr_asm_matches_strict_three_face_helper():
    matrix, reference = _triangle_problem(7, permuted=True)
    strict = build_bsr_face_additive_schwarz_local_matrices(
        matrix, reference.element_system_faces
    )
    generic = build_bsr_additive_schwarz_local_matrices(
        matrix, reference.element_system_faces
    )
    np.testing.assert_array_equal(generic.local_matrices, strict.local_matrices)
    inverses = np.linalg.inv(generic.local_matrices)
    actual = assemble_bsr_additive_schwarz_correction(matrix, generic, inverses)
    expected = assemble_bsr_face_additive_schwarz_correction(matrix, strict, inverses)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize("defect", ["uncovered", "duplicate", "missing_diagonal"])
def test_generic_bsr_asm_rejects_invalid_patch_inputs(defect):
    matrix, reference = _triangle_problem(2)
    patches = _wider_patches(reference, 5)
    if defect == "uncovered":
        patches[patches == 0] = -1
        message = "cover every matrix face row"
    elif defect == "duplicate":
        row = int(np.argmax(np.sum(patches >= 0, axis=1)))
        patches[row, 1] = patches[row, 0]
        message = "repeat an active system face"
    else:
        row = 0
        begin, end = matrix.indptr[row:row + 2]
        columns = matrix.indices[begin:end]
        data = matrix.data[begin:end]
        active = columns != row
        matrix = _replace_row(matrix, row, columns[active], data[active])
        message = "missing an active element face pair"
    with pytest.raises(ValueError, match=message):
        build_bsr_additive_schwarz_local_matrices(matrix, patches)


def test_generic_bsr_asm_repeated_patch_accumulation_and_padding():
    matrix, reference = _triangle_problem(2)
    patches = _wider_patches(reference, 9)
    local = build_bsr_additive_schwarz_local_matrices(matrix, patches)
    inverse = np.linalg.inv(local.local_matrices)
    expected = assemble_bsr_additive_schwarz_correction(matrix, local, inverse)
    repeated = build_bsr_additive_schwarz_local_matrices(
        matrix, np.concatenate((patches, patches))
    )
    inverse = np.concatenate((inverse, inverse))
    for patch, faces in enumerate(repeated.element_system_faces):
        for face in np.flatnonzero(faces < 0):
            dofs = slice(2 * face, 2 * (face + 1))
            inverse[patch, dofs, :] = 913.0
            inverse[patch, :, dofs] = -871.0
    actual = assemble_bsr_additive_schwarz_correction(matrix, repeated, inverse)
    np.testing.assert_allclose(actual.toarray(), 2.0 * expected.toarray(), rtol=2e-14, atol=2e-14)
