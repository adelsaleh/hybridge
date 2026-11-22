"""Tiny CPU-only ordering, archive-integrity and coordinate-mapping checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from common import system_fingerprint
import pardiso_ordering as ordering


def tiny_system():
    rng = np.random.default_rng(81021)
    dense = rng.standard_normal((64, 64))
    dense[rng.random(dense.shape) > 0.045] = 0
    dense[np.diag_indices_from(dense)] = 0
    dense += np.diag(np.abs(dense).sum(axis=1) + 2)
    matrix = csr_matrix(dense)
    exact = rng.standard_normal(matrix.shape[0])
    return matrix, matrix @ exact, exact


class OrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pypardiso
        cls.matrix, cls.rhs, cls.exact = tiny_system()
        cls.system_sha256 = system_fingerprint(cls.matrix, cls.rhs)
        cls.phases = []
        native_call = pypardiso.PyPardisoSolver._call_pardiso

        def observe(solver, matrix, rhs):
            cls.phases.append(solver.phase)
            result = native_call(solver, matrix, rhs)
            if solver.phase == 11:
                cls.native_permutation = solver.perm.copy()
            return result

        with patch.object(pypardiso.PyPardisoSolver, "_call_pardiso", observe):
            cls.p, cls.info = ordering.compute_permutation(cls.matrix, threads=1, verbosity=0)
        cls.info["system_sha256"] = cls.system_sha256

    def test_native_analysis_only_and_one_based_conversion(self):
        self.assertEqual(self.phases, [11, -1])
        self.assertEqual(self.p.dtype, np.int32)
        np.testing.assert_array_equal(self.p + 1, self.native_permutation)
        np.testing.assert_array_equal(np.sort(self.p), np.arange(64))
        self.assertGreater(np.count_nonzero(self.p != np.arange(64)), 0)
        self.assertEqual(self.info["threads_effective"], 1)
        self.assertFalse(self.info["mkl_dynamic"])
        self.assertGreater(self.info["analysis_seconds"], 0)
        self.assertTrue(self.info["mkl_version"])
        for name in ("iparm_before", "iparm_after"):
            self.assertEqual(set(self.info[name]), {str(i) for i in range(1, 65)})
        for key, value in {1: 1, 5: 2, 31: 0, 35: 0, 36: 0}.items():
            self.assertEqual(self.info["iparm_before"][str(key)], value)

    def test_reorder_matches_independent_dense_coordinates(self):
        original = [x.copy() for x in (self.matrix.data, self.matrix.indices, self.matrix.indptr)]
        self.assertTrue(self.matrix.has_sorted_indices)
        # Exercise multiple remapping chunks and SciPy's cached sortedness flag.
        with patch.object(ordering, "CHUNK_ENTRIES", 7):
            reordered = ordering.reorder_csr(self.matrix, self.p)
        expected = self.matrix.toarray()[np.ix_(self.p, self.p)]
        np.testing.assert_array_equal(reordered.toarray(), expected)
        self.assertEqual(reordered.dtype, np.float64)
        self.assertEqual(reordered.nnz, self.matrix.nnz)
        self.assertTrue(reordered.has_sorted_indices)
        self.assertTrue(reordered.has_canonical_format)
        self.assertEqual(reordered.indices.dtype, np.int32)
        for before, after in zip(original, (self.matrix.data, self.matrix.indices, self.matrix.indptr)):
            np.testing.assert_array_equal(after, before)
        self.assertFalse(np.shares_memory(reordered.data, self.matrix.data))
        y = np.linalg.solve(reordered.toarray(), self.rhs[self.p])
        x = np.empty_like(y)
        x[self.p] = y
        np.testing.assert_allclose(x, self.exact, rtol=1e-12, atol=1e-12)

    def test_scaling_and_warm_start_map_back_to_original_system(self):
        n = self.matrix.shape[0]
        row_scale = np.logspace(-2, 2, n)
        column_scale = np.logspace(1, -1, n)
        reordered = ordering.reorder_csr(self.matrix, self.p).toarray()
        scaled = row_scale[:, None] * reordered * column_scale[None, :]
        scaled_rhs = row_scale * self.rhs[self.p]
        x0 = np.linspace(-0.5, 0.5, n)
        y0 = x0[self.p] / column_scale
        correction = np.linalg.solve(scaled, scaled_rhs - scaled @ y0)
        x = np.empty(n)
        x[self.p] = column_scale * (y0 + correction)
        np.testing.assert_allclose(self.matrix @ x, self.rhs, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(x, self.exact, rtol=1e-11, atol=1e-11)

    def test_consecutive_bsr_counts_and_original_coordinate_solution(self):
        from inspect_ordered_blocks import block_profile
        ordered = ordering.reorder_csr(self.matrix, self.p)
        # Explicit stored zero: occupancy must describe storage, not drop it.
        stored_zero = ordered.copy()
        stored_zero.data[1] = 0.
        for size in (2, 4, 8, 16):
            profile = block_profile(stored_zero, size, numerical_nnz=np.count_nonzero(stored_zero.data))
            blocked_zero = stored_zero.tobsr(blocksize=(size, size))
            # Independent dense block grid checks touched blocks from CSR pairs.
            rows = np.repeat(np.arange(64), np.diff(stored_zero.indptr))
            touched = set(zip(rows // size, stored_zero.indices // size))
            self.assertEqual(profile['stored_blocks'], len(touched))
            self.assertEqual(profile['scalar_slots'], blocked_zero.data.size)
            self.assertEqual(profile['bsr_bytes'], blocked_zero.data.nbytes + blocked_zero.indices.nbytes + blocked_zero.indptr.nbytes)
            self.assertLess(profile['numerical_occupancy'], profile['stored_occupancy'])
            blocked = ordered.tobsr(blocksize=(size, size))
            blocked.sort_indices()
            self.assertTrue(blocked.has_sorted_indices)
            expected = self.matrix.toarray()[np.ix_(self.p, self.p)]
            np.testing.assert_array_equal(blocked.toarray(), expected)
            y = np.linalg.solve(blocked.toarray(), self.rhs[self.p])
            x = np.empty_like(y)
            x[self.p] = y
            np.testing.assert_allclose(x, self.exact, rtol=1e-11, atol=1e-11)

    def test_archive_roundtrip_and_atomic_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordering.npz"
            summary = ordering.save_permutation(path, self.p, self.info)
            p, info = ordering.load_permutation(path, self.system_sha256, self.p.size)
            np.testing.assert_array_equal(p, self.p)
            self.assertEqual(info, summary)
            self.assertEqual(info["direction"], "new_to_old")
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                ordering.save_permutation(path, self.p, self.info)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            with np.load(path, allow_pickle=False) as raw:
                self.assertEqual(raw["metadata"].dtype.kind, "U")
                self.assertEqual(raw["p"].dtype, np.int32)

    def test_bad_permutations_rejected(self):
        cases = [self.p.astype(np.int64), self.p.reshape(-1, 1), self.p[:-1],
                 np.zeros_like(self.p), self.p - np.int32(1), self.p + np.int32(1)]
        for p in cases:
            with self.subTest(shape=p.shape, dtype=p.dtype):
                with self.assertRaises(ValueError):
                    ordering.reorder_csr(self.matrix, p)

    def test_archive_rejects_system_size_direction_and_checksum_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordering.npz"
            info = ordering.save_permutation(path, self.p, self.info)
            with self.assertRaisesRegex(ValueError, "system_sha256"):
                ordering.load_permutation(path, "0" * 64, self.p.size)
            with self.assertRaisesRegex(ValueError, "size"):
                ordering.load_permutation(path, self.system_sha256, self.p.size + 1)
            tampered = self.p.copy()
            tampered[:2] = tampered[1::-1]
            np.savez(path, p=tampered, metadata=np.asarray(json.dumps(info)))
            with self.assertRaisesRegex(ValueError, "checksum"):
                ordering.load_permutation(path, self.system_sha256, self.p.size)
            info["direction"] = "old_to_new"
            np.savez(path, p=self.p, metadata=np.asarray(json.dumps(info)))
            with self.assertRaisesRegex(ValueError, "direction"):
                ordering.load_permutation(path, self.system_sha256, self.p.size)

    def test_unsafe_matrix_storage_rejected(self):
        cases = []
        wide = self.matrix.copy()
        wide.indices = wide.indices.astype(np.int64)
        cases.append(wide)
        unsorted = csr_matrix([[4., 2.], [1., 3.]])
        unsorted.indices[:2] = unsorted.indices[1::-1]
        unsorted.has_sorted_indices = False
        cases.append(unsorted)
        cases.extend([self.matrix.astype(np.float32), self.matrix.tocsc(),
                      csr_matrix([[1., 0.], [0., 0.]])])
        for matrix in cases:
            with self.subTest(shape=matrix.shape, dtype=matrix.dtype):
                with self.assertRaises((ValueError, TypeError)):
                    ordering.compute_permutation(matrix, threads=1, verbosity=0)

    def test_cleanup_runs_when_native_analysis_raises(self):
        import pypardiso
        phases = []
        native_call = pypardiso.PyPardisoSolver._call_pardiso

        def fail_analysis(solver, matrix, rhs):
            phases.append(solver.phase)
            if solver.phase == 11:
                raise RuntimeError("injected analysis failure")
            return native_call(solver, matrix, rhs)

        with patch.object(pypardiso.PyPardisoSolver, "_call_pardiso", fail_analysis):
            with self.assertRaisesRegex(RuntimeError, "injected analysis failure"):
                ordering.compute_permutation(self.matrix, threads=1, verbosity=0)
        self.assertEqual(phases, [11, -1])


if __name__ == "__main__":
    unittest.main()
