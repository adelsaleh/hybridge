from __future__ import annotations

import numpy as np

from scripts.validate_face_dense_gpu_restart import krylov_basis_mebibytes


def test_krylov_basis_memory_matches_explicit_byte_count() -> None:
    restart = 50
    num_dofs = 12345
    expected = (restart + 1) * num_dofs * 8 / 2.0**20
    assert krylov_basis_mebibytes(restart, num_dofs, np.float64) == expected


def test_krylov_basis_memory_tracks_dtype_and_restart() -> None:
    small = krylov_basis_mebibytes(20, 1000, np.float32)
    large = krylov_basis_mebibytes(40, 1000, np.float64)
    assert large > 3.0 * small
