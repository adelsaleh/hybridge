import unittest
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.sparse import csr_matrix, eye
from common import load_iterate, save_iterate, system_fingerprint

from benchmark_amgx_strong import block_matrix, candidates, equilibrate, expand_block_pattern, normalize_diagonal_blocks, reorder_block_rcm, run_trial, smoke_system


class StrongTests(unittest.TestCase):
    def test_equilibration_preserves_original_system(self):
        matrix, rhs, _ = smoke_system()
        original = matrix.data.copy()
        scaled, dr, dc = equilibrate(matrix, 5)
        y = np.random.default_rng(42).standard_normal(matrix.shape[0])
        np.testing.assert_allclose(scaled @ y, dr * (matrix @ (dc * y)), rtol=1e-12, atol=1e-12)
        x = dc * np.linalg.solve(scaled.toarray(), dr * rhs)
        np.testing.assert_allclose(matrix @ x, rhs, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(matrix.data, original)

    def test_block_conversion_preserves_operator(self):
        matrix, _, _ = smoke_system()
        vector = np.arange(matrix.shape[0], dtype=float)
        for size in (1, 4, 20):
            np.testing.assert_allclose(block_matrix(matrix, size) @ vector, matrix @ vector, rtol=1e-12)
        with self.assertRaises(ValueError):
            block_matrix(eye(80, format="csr"), 4)

    def test_uniform_right_multiplier_preserves_original_system(self):
        matrix, rhs, _ = smoke_system()
        for multiplier in (1e-6, 1e3, 1e6):
            scaled, dr, dc = equilibrate(matrix, 0, "column-max", multiplier)
            y = np.linalg.solve(scaled.toarray(), dr * rhs)
            np.testing.assert_allclose(matrix @ (dc * y), rhs, rtol=1e-10, atol=1e-10)
        for invalid in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                equilibrate(matrix, 0, "column-max", invalid)

    def test_supported_preconditioned_krylov_and_ilu(self):
        specs = candidates()
        self.assertGreaterEqual(len(specs), 10)
        for spec in specs.values():
            solver = spec["config"]["solver"]
            if spec.get("outer_backend") != "hybridge_gmres":
                self.assertIn(solver["solver"], ("GMRES", "FGMRES", "PBICGSTAB", "BICGSTAB", "IDRMSYNC"))
                if solver["solver"] == "IDRMSYNC":
                    self.assertEqual(spec["block_size"], 1)
                    self.assertEqual(solver["store_res_history"], 0)
                    self.assertEqual(solver["max_iters"] % (solver["subspace_dim_s"] + 1), 0)
                if solver["solver"] == "BICGSTAB":
                    self.assertEqual(solver["preconditioner"], "NOSOLVER")
            if solver["solver"] == "GMRES":
                self.assertEqual(solver["gmres_reliable_residual"], 1)
            def check(node):
                if node.get("solver") == "MULTICOLOR_ILU" and node.get("block_ilu_backend") != "cusparse_legacy":
                    self.assertEqual(spec["block_size"], 4)
                    self.assertGreater(node["coloring_level"], node["ilu_sparsity_level"])
                    self.assertEqual(node["reorder_cols_by_color"], 1)
                for value in node.values():
                    if isinstance(value, dict):
                        check(value)
            check(solver)
            preconditioner = solver.get("preconditioner", {})
            if (solver["solver"] in ("GMRES", "PBICGSTAB") and isinstance(preconditioner, dict)
                    and preconditioner.get("solver") == "AMG"):
                self.assertEqual(preconditioner["error_scaling"], 0)

    def test_zero_equilibration_is_identity(self):
        matrix = csr_matrix([[4., 1.], [2., 3.]])
        scaled, dr, dc = equilibrate(matrix, 0)
        np.testing.assert_array_equal(scaled.toarray(), matrix.toarray())
        np.testing.assert_array_equal(dr, np.ones(2))
        np.testing.assert_array_equal(dc, np.ones(2))

    def test_checkpoint_checks_equations_and_vector_integrity(self):
        matrix = csr_matrix([[4., 1.], [2., 3.]])
        rhs = np.array([1., 2.])
        fingerprint = system_fingerprint(matrix, rhs)
        self.assertEqual(system_fingerprint(matrix.copy(), rhs.copy()), fingerprint)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iterate.npz"
            x = np.array([0.1, 0.4])
            saved = save_iterate(path, x, dict(system_sha256=fingerprint, relative_residual=0.2))
            loaded, metadata = load_iterate(path, fingerprint, 2)
            np.testing.assert_array_equal(loaded, x)
            self.assertEqual(metadata["coordinates"], "original_global")
            with self.assertRaises(ValueError):
                load_iterate(path, system_fingerprint(matrix, rhs * 2), 2)
            with self.assertRaises(ValueError):
                load_iterate(path, fingerprint, 3)
            import json
            np.savez(path, x=x * 2, metadata=np.array(json.dumps(saved)))
            with self.assertRaises(ValueError):
                load_iterate(path, fingerprint, 2)
            with self.assertRaises(ValueError):
                save_iterate(path, np.array([np.nan, 1.]), dict(system_sha256=fingerprint))

    def test_diagonal_and_row_scaling_preserve_equations(self):
        matrix = csr_matrix([[-4., 1.], [2., 3.]])
        for method in ("diagonal", "row-max", "column-max", "column-diagonal", "none"):
            scaled, dr, dc = equilibrate(matrix, 5, method)
            np.testing.assert_allclose(scaled.toarray(), dr[:, None] * matrix.toarray() * dc[None, :])
            if method == "diagonal":
                np.testing.assert_allclose(scaled.diagonal(), np.ones(2))

    def test_nonfinite_native_iteration_stops_worker(self):
        with tempfile.TemporaryDirectory() as root:
            command = [sys.executable, "-c", "import time; print('  0  1.0  2.0  nan  nan', flush=True); time.sleep(30)"]
            result = run_trial(command, Path(root) / "trial.json", timeout=5, verbosity=0)
            self.assertEqual(result["status"], "numerical_failure")
            self.assertIn("Nonfinite", result["error"])

    def test_extra_fill_changes_pattern_without_changing_equations(self):
        # Sparse chain of dense 4x4 blocks has new paths at distances 2 and 3.
        from scipy.sparse import kron
        graph = csr_matrix(np.diag(np.full(6, 4.)) + np.diag(np.ones(5), 1) + np.diag(np.ones(5), -1))
        original = kron(graph, np.full((4, 4), 0.1) + np.eye(4), format="bsr").tobsr(blocksize=(4, 4))
        for power in (2, 3):
            expanded, metadata = expand_block_pattern(original, power, 3)
            self.assertGreater(metadata["expanded_blocks"], metadata["original_blocks"])
            np.testing.assert_array_equal(expanded.toarray(), original.toarray())

    def test_block_normalization_transforms_both_matrix_and_rhs(self):
        matrix, rhs, _ = smoke_system()
        normalized, inverse = normalize_diagonal_blocks(matrix, 20)
        x = np.linalg.solve(matrix.toarray(), rhs)
        transformed = np.einsum("nij,nj->ni", inverse, rhs.reshape(-1, 20)).ravel()
        np.testing.assert_allclose(normalized @ x, transformed, rtol=1e-11, atol=1e-11)

    def test_block_reordering_preserves_equations(self):
        matrix, rhs, _ = smoke_system()
        reordered, permutation = reorder_block_rcm(matrix, 20)
        x = np.linalg.solve(matrix.toarray(), rhs)
        np.testing.assert_allclose(reordered @ x[permutation], rhs[permutation], rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
