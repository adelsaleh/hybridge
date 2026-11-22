"""Small real HDF5 fixtures: indexing, global row order, RHS alignment, failures."""
import tempfile
from pathlib import Path
import unittest
from contextlib import redirect_stdout
import io

import h5py
import numpy as np

from common import load_system, log, parser, residual
from benchmark_amgx import configurations


class ImportTests(unittest.TestCase):
    def fixture(self, root, base=1, permute=False):
        path = Path(root) / "system.h5"
        # Nonsymmetric A = [[4, 2], [1, 3]], b = [8, 7].
        with h5py.File(path, "w") as f:
            g = f.create_group("mat")
            g["n"] = [2]
            g["nnz"] = [4]
            g["rowptr"] = np.array([0, 2, 4], dtype=np.int32) + base
            g["cols"] = np.array([0, 1, 0, 1], dtype=np.int32) + base
            g["vals"] = [1., 3., 4., 2.] if permute else [4., 2., 1., 3.]
            g["loc2glob"] = np.array([1, 0] if permute else [0, 1]) + base
            g["rhs_vals"] = np.array([[7., 8.]])
            g["rhs_loc2glob"] = np.array([[1, 0]]) + base
        return path

    def test_bases_and_global_mappings(self):
        with tempfile.TemporaryDirectory() as tmp:
            for base in (0, 1):
                for permute in (False, True):
                    a, b, meta = load_system(self.fixture(tmp, base, permute))
                    np.testing.assert_array_equal(a.toarray(), [[4, 2], [1, 3]])
                    np.testing.assert_array_equal(b, [8, 7])
                    self.assertEqual(a.indices.dtype, np.int32)
                    self.assertEqual(meta["input_index_base"], base)
                    self.assertTrue(residual(a, b, np.array([1., 2.]), 1e-12)["passed"])

    def test_invalid_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for dataset, value in (("cols", [0, 2, 1, 2]), ("rowptr", [1, 4, 3]),
                                   ("rhs_loc2glob", [[1, 1]]), ("loc2glob", [1, 1]),
                                   ("vals", [4., float("nan"), 1., 3.])):
                path = self.fixture(tmp)
                with h5py.File(path, "r+") as f:
                    f[f"mat/{dataset}"][...] = value
                with self.assertRaises(ValueError, msg=dataset):
                    load_system(path)

    def test_no_invented_rhs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(tmp)
            with h5py.File(path, "r+") as f:
                del f["mat/rhs_vals"]
            with self.assertRaisesRegex(ValueError, "right-hand side"):
                load_system(path)

    def test_nonsymmetric_solver_sweep(self):
        configs = configurations(1e-8, 1000, 100)
        self.assertEqual(len(configs), 11)
        for config in configs.values():
            self.assertIn(config["solver"]["solver"], ("FGMRES", "PBICGSTAB"))

    def test_verbosity_levels(self):
        p = parser("test")
        self.assertEqual(p.parse_args(["--output", "unused"]).verbosity, 2)
        self.assertEqual(p.parse_args(["--output", "unused", "--quiet"]).verbosity, 0)
        for verbosity in range(4):
            args = p.parse_args(["--output", "unused", "-v", str(verbosity)])
            output = io.StringIO()
            with redirect_stdout(output):
                for level in range(4):
                    log(args.verbosity, level, f"message_{level}")
            for level in range(4):
                self.assertEqual(f"message_{level}" in output.getvalue(), level <= verbosity)


if __name__ == "__main__":
    unittest.main()
