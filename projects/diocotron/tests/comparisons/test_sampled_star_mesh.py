"""Keep the standalone Newton experiment's geometry unchanged on relocation."""
from pathlib import Path
import subprocess
import sys

import pytest


def test_local_sampled_star_matches_original_hdg_mesh(tmp_path):
    pytest.importorskip("gmsh")
    pytest.importorskip("meshio")
    source = '''
import sys
from pathlib import Path
import meshio
import numpy as np
from hybridge.core.mesh import gmsh_smooth_star_mesh
from projects.diocotron.dolfinx.geometry.sampled_star import write_sampled_star_mesh
root = Path(sys.argv[1])
kwargs = dict(boundary_points=40, radius=1.5, amplitude=.32, mode=5,
              verbosity=0, algorithm=6, msh_file_version=2.2)
gmsh_smooth_star_mesh(.3, write_path=str(root / "original.msh"), **kwargs)
write_sampled_star_mesh(.3, write_path=root / "relocated.msh", **kwargs)
a, b = (meshio.read(root / name) for name in ("original.msh", "relocated.msh"))
np.testing.assert_array_equal(a.points, b.points)
np.testing.assert_array_equal(a.cells_dict["triangle"], b.cells_dict["triangle"])
'''
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run([sys.executable, "-c", source, str(tmp_path)], cwd=root,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
