#!/usr/bin/env python3
"""Gmsh meshes of the GC convergence campaign: manifest, bundle and check of the mesh cache.

The runs read their Gmsh meshes from the repository's mesh cache, ``<repo>/.cache/hybridge/meshes`` (the
workers run in the repository root; ``hybridge.core.mesh._mesh_cache_files`` names the files after a hash of
the mesh key). Shipping the cached files keeps the cluster runs on the meshes of the local runs and needs no
Gmsh there. Structured meshes are built deterministically (``rectangle_mesh``) and are not cached.

Commands (run from any directory):

* ``manifest``: hash the cached meshes of ``MESHES`` into ``mesh_manifest.json`` (on the machine that holds them);
* ``pack BUNDLE.tar.gz``: write them into a tarball with paths relative to the repository root;
* ``verify [FAMILY:H ...]``: check that the listed meshes (default: all) are cached with the manifest's hash.

Unpack a bundle in the repository root (``tar -xzf BUNDLE.tar.gz``), then run ``verify``.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from argparse import ArgumentParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CACHE = REPO / ".cache" / "hybridge" / "meshes"
MANIFEST = HERE / "mesh_manifest.json"
UNSTRUCTURED_ALGORITHM = 5  # Gmsh Delaunay: run_guiding_center_temporal_convergence.mesh_family_gmsh_algorithm
# family -> (Gmsh model, algorithm, geometry key, mesh sizes)
MESHES = {
    "gmsh": ("rectangle", None, "rectangle", (0.08, 0.04, 0.02, 0.0125, 0.01, 0.005, 0.0025, 0.00177)),
    "unstructured": ("rectangle", UNSTRUCTURED_ALGORITHM, "rectangle", (0.08, 0.04, 0.02, 0.01, 0.005, 0.0025)),
    "polygon": ("polygon", None, "polygon", (0.025, 0.00625)),
}


def _geometry(kind: str) -> dict:
    """Mesh-cache key data of the runner's domains: the square (-1, 1)^2 or the vortex cross polygon."""
    if kind == "rectangle":
        return {"geometry": "rectangle", "xlim": (-1.0, 1.0), "ylim": (-1.0, 1.0)}
    from hybridge.core.geometry import PolygonDomain
    from scripts.guiding_center.cases.guiding_center_cases import CROSS_POLYGON

    return {"geometry": "polygon", "vertices": PolygonDomain(CROSS_POLYGON).vertices}


def key(family: str, mesh_size: float) -> str:
    """Manifest key ``family:h`` of one mesh."""
    return f"{family}:{float(mesh_size):g}"


def cache_file(family: str, mesh_size: float) -> Path:
    """Cache file the runner reads for this mesh."""
    from hybridge.core.mesh import _mesh_cache_files

    model, algorithm, geometry, _ = MESHES[family]
    paths, _ = _mesh_cache_files(model, float(mesh_size), algorithm=algorithm, cache_key_data=_geometry(geometry),
                                 cache_dir=CACHE)
    return paths[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _triangles(path: Path) -> int:
    """Number of triangles of a cache file (magic line, key-size line, key, nodes, triangles)."""
    import numpy as np

    with path.open("rb") as handle:
        handle.readline()
        size = int(handle.readline())
        handle.read(size)
        np.load(handle, allow_pickle=False)
        return int(np.load(handle, allow_pickle=False).shape[0])


def manifest() -> Path:
    """Write ``mesh_manifest.json`` from the local cache; every mesh of ``MESHES`` must be cached."""
    import numpy as np  # noqa: F401  (fails early without NumPy)
    from scripts.guiding_center.benchmarks.run_guiding_center_temporal_convergence import mesh_family_gmsh_algorithm

    if mesh_family_gmsh_algorithm("unstructured") != UNSTRUCTURED_ALGORITHM:
        raise SystemExit("the runner's unstructured Gmsh algorithm changed; update UNSTRUCTURED_ALGORITHM")
    entries, missing = {}, []
    for family, (model, algorithm, _, sizes) in MESHES.items():
        for size in sizes:
            path = cache_file(family, size)
            if not path.exists():
                missing.append(key(family, size))
                continue
            entries[key(family, size)] = {
                "family": family, "model": model, "mesh_size": size, "algorithm": algorithm,
                "triangles": _triangles(path), "file": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if missing:
        raise SystemExit("not in the mesh cache: " + ", ".join(missing))
    MANIFEST.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"{len(entries)} meshes, {sum(e['bytes'] for e in entries.values()) / 1e6:.0f} MB -> {MANIFEST}")
    return MANIFEST


def load_manifest() -> dict:
    """Manifest entries keyed by ``family:h``."""
    return json.loads(MANIFEST.read_text())


def verify(keys=None, *, quiet: bool = False) -> list[str]:
    """Return the problems (missing file or hash mismatch) of the listed meshes, default all of the manifest."""
    entries = load_manifest()
    problems = []
    for name in (sorted(entries) if not keys else keys):
        entry = entries.get(name)
        if entry is None:
            problems.append(f"{name}: not in {MANIFEST.name}")
            continue
        path = REPO / entry["file"]
        if not path.exists():
            problems.append(f"{name}: missing {entry['file']}")
        elif path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            problems.append(f"{name}: {entry['file']} differs from the manifest")
    if not quiet:
        for problem in problems:
            print(f"[meshes] {problem}")
        print(f"[meshes] {len(keys or entries) - len(problems)} of {len(keys or entries)} meshes verified")
    return problems


def pack(bundle: Path) -> Path:
    """Write every manifest mesh into ``bundle`` (tar.gz, paths relative to the repository root)."""
    problems = verify(quiet=True)
    if problems:
        raise SystemExit("cannot pack: " + "; ".join(problems))
    with tarfile.open(bundle, "w:gz") as archive:
        for entry in load_manifest().values():
            archive.add(REPO / entry["file"], arcname=entry["file"])
    print(f"[meshes] {bundle} ({bundle.stat().st_size / 1e6:.0f} MB)")
    return bundle


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("manifest", help="hash the cached meshes into mesh_manifest.json")
    packer = commands.add_parser("pack", help="write the cached meshes into a tar.gz bundle")
    packer.add_argument("bundle", type=Path)
    checker = commands.add_parser("verify", help="check the cached meshes against the manifest")
    checker.add_argument("keys", nargs="*", help="family:h keys (default: all)")
    args = parser.parse_args()
    if args.command == "manifest":
        manifest()
    elif args.command == "pack":
        pack(args.bundle)
    elif verify(args.keys or None):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
