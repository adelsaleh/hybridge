import sys
from pathlib import Path

import numpy as np
import pytest

import hybridge.core.mesh as mesh_module
from hybridge.core.mesh import DGMesh, default_mesh_cache_dir, gmsh_rectangle_mesh


class _FakeGmshOption:
    def __init__(self):
        self.values = {}

    def setNumber(self, name, value):
        self.values[name] = value

    def getNumber(self, name):
        return self.values[name]


class _FakeGmshOcc:
    def __init__(self):
        self.rectangle_calls = []
        self.synchronize_calls = 0

    def addRectangle(self, x, y, z, *, dx, dy):
        self.rectangle_calls.append((x, y, z, dx, dy))
        return 7

    def synchronize(self):
        self.synchronize_calls += 1


class _FakeGmshMesh:
    def __init__(self):
        self.generate_calls = 0

    def generate(self, dim):
        assert dim == 2
        self.generate_calls += 1

    def getNodes(self):
        return (
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0], dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    def getElementsByType(self, element_type):
        assert element_type == 2
        return np.array([1], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64)


class _FakeGmshModel:
    def __init__(self):
        self.occ = _FakeGmshOcc()
        self.mesh = _FakeGmshMesh()
        self.names = []
        self.physical_groups = []

    def add(self, name):
        self.names.append(name)

    def addPhysicalGroup(self, dim, tags, tag=None, name=None):
        self.physical_groups.append((dim, tuple(tags), tag, name))


class _FakeGmsh:
    def __init__(self):
        self.option = _FakeGmshOption()
        self.model = _FakeGmshModel()
        self.initialized = False
        self.initialize_calls = 0
        self.finalize_calls = 0

    def isInitialized(self):
        return self.initialized

    def initialize(self, **options):
        self.initialize_options = options
        self.initialized = True
        self.initialize_calls += 1

    def clear(self):
        pass

    def finalize(self):
        self.initialized = False
        self.finalize_calls += 1

    def write(self, path):
        raise AssertionError(f"unexpected gmsh.write({path!r})")


def test_default_mesh_cache_dir_is_local_relative():
    assert default_mesh_cache_dir() == Path(".cache") / "hybridge" / "meshes"


def test_gmsh_rectangle_mesh_uses_local_cache_logs_and_thread_options(monkeypatch, tmp_path, capsys):
    fake = _FakeGmsh()
    monkeypatch.setitem(sys.modules, "gmsh", fake)

    mesh1 = gmsh_rectangle_mesh(
        0.25,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        cache_dir=tmp_path,
        num_threads=4,
    )
    first_log = capsys.readouterr().out

    assert "mesh cache miss" in first_log
    assert "mesh cache stored" in first_log
    assert fake.model.mesh.generate_calls == 1
    assert fake.option.values["General.NumThreads"] == 4
    # gmsh's interruptible mode leaves SIGINT at SIG_DFL after finalize (Ctrl-C regression).
    assert fake.initialize_options == {"interruptible": False}
    assert fake.option.values["Mesh.MaxNumThreads2D"] == 4
    assert fake.option.values["Geometry.OCCParallel"] == 1
    assert list(tmp_path.glob("rectangle-*.npz"))

    mesh2 = gmsh_rectangle_mesh(
        0.25,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        cache_dir=tmp_path,
        num_threads=2,
    )
    second_log = capsys.readouterr().out

    assert "mesh cache hit" in second_log
    assert fake.model.mesh.generate_calls == 1
    np.testing.assert_allclose(mesh2.node_coords, mesh1.node_coords)
    np.testing.assert_array_equal(mesh2.triangles, mesh1.triangles)

    gmsh_rectangle_mesh(
        0.25,
        xlim=(0.0, 2.0),
        ylim=(0.0, 1.0),
        cache_dir=tmp_path,
    )
    third_log = capsys.readouterr().out

    assert "mesh cache miss" in third_log
    assert fake.model.mesh.generate_calls == 2


def test_gmsh_rectangle_mesh_falls_back_when_default_cache_write_fails(monkeypatch, tmp_path, capsys):
    fake = _FakeGmsh()
    monkeypatch.setitem(sys.modules, "gmsh", fake)
    monkeypatch.setattr(mesh_module, "_DEFAULT_MESH_CACHE_DIR", tmp_path / "primary")
    monkeypatch.setattr(mesh_module, "_fallback_mesh_cache_dir", lambda: tmp_path / "fallback")

    original_write = mesh_module._write_cached_gmsh_mesh

    def flaky_write(cache_path, mesh, cache_key_json):
        if cache_path.parent == tmp_path / "primary":
            raise OSError("simulated project-cache write failure")
        original_write(cache_path, mesh, cache_key_json)

    monkeypatch.setattr(mesh_module, "_write_cached_gmsh_mesh", flaky_write)

    mesh1 = gmsh_rectangle_mesh(0.25, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    first_log = capsys.readouterr().out

    assert "mesh cache write failed" in first_log
    assert "mesh cache stored in fallback cache" in first_log
    assert fake.model.mesh.generate_calls == 1
    assert list((tmp_path / "fallback").glob("rectangle-*.npz"))

    mesh2 = gmsh_rectangle_mesh(0.25, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    second_log = capsys.readouterr().out

    assert "mesh cache fallback hit" in second_log
    assert fake.model.mesh.generate_calls == 1
    np.testing.assert_allclose(mesh2.node_coords, mesh1.node_coords)
    np.testing.assert_array_equal(mesh2.triangles, mesh1.triangles)


def test_malformed_uniform_mesh_cache_is_rejected_and_regenerated(monkeypatch, tmp_path, capsys):
    fake = _FakeGmsh()
    monkeypatch.setitem(sys.modules, "gmsh", fake)

    expected = gmsh_rectangle_mesh(0.25, xlim=(0.0, 1.0), ylim=(0.0, 1.0), cache_dir=tmp_path)
    cache_path = next(tmp_path.glob("rectangle-*.npz"))
    malformed = DGMesh.from_arrays(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
        np.array([[0, 1, 2]]),
    )
    _, cache_key_json = mesh_module._mesh_cache_files(
        "rectangle",
        0.25,
        algorithm=None,
        cache_key_data={"geometry": "rectangle", "xlim": (0.0, 1.0), "ylim": (0.0, 1.0)},
        cache_dir=tmp_path,
    )
    mesh_module._write_cached_gmsh_mesh(cache_path, malformed, cache_key_json)
    capsys.readouterr()

    regenerated = gmsh_rectangle_mesh(0.25, xlim=(0.0, 1.0), ylim=(0.0, 1.0), cache_dir=tmp_path)
    log = capsys.readouterr().out

    assert "mesh cache read failed" in log
    assert "violates requested sizing" in log
    assert fake.model.mesh.generate_calls == 2
    np.testing.assert_allclose(regenerated.node_coords, expected.node_coords)
    np.testing.assert_array_equal(regenerated.triangles, expected.triangles)


def test_malformed_fresh_uniform_mesh_is_not_cached(monkeypatch, tmp_path):
    fake = _FakeGmsh()
    monkeypatch.setitem(sys.modules, "gmsh", fake)
    malformed = DGMesh.from_arrays(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
        np.array([[0, 1, 2]]),
    )
    monkeypatch.setattr(mesh_module, "_gmsh_model_to_mesh", lambda *args, **kwargs: malformed)

    with pytest.raises(ValueError, match="violates requested sizing"):
        gmsh_rectangle_mesh(0.25, cache_dir=tmp_path)

    assert not list(tmp_path.glob("rectangle-*.npz"))



def test_required_uniform_size_option_is_verified(monkeypatch, tmp_path):
    fake = _FakeGmsh()
    monkeypatch.setitem(sys.modules, "gmsh", fake)

    def altered_value(name):
        value = fake.option.values[name]
        return 2.0 * value if name == "Mesh.MeshSizeMax" else value

    fake.option.getNumber = altered_value

    with pytest.raises(RuntimeError, match="Mesh.MeshSizeMax.*was not applied"):
        gmsh_rectangle_mesh(0.25, cache_dir=tmp_path)

    assert fake.model.mesh.generate_calls == 0
    assert not list(tmp_path.glob("rectangle-*.npz"))
