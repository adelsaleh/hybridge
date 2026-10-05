"""Validated, MPI-safe cache for canonical DOLFINx/Gmsh meshes.

The installed :mod:`hdgfem` package has its own cache for ``DGMesh`` arrays.
Equiband needs tagged, possibly curved ``.msh`` files instead, so this module
mirrors the same cache principles in the script layer without introducing a
DOLFINx dependency into :mod:`hdgfem`:

* JSON-stable keys include geometry, parameters, size, geometry order and the
  canonical-generator source hash;
* a cache hit validates both metadata and the complete mesh SHA-256;
* a cache miss is generated in a temporary directory and installed atomically;
* a file lock prevents independent jobs from constructing the same key at the
  same time; and
* in an MPI run, rank zero performs filesystem/Gmsh work and broadcasts the
  authoritative result to every rank.

Gmsh generation occurs in a short subprocess.  This keeps the Gmsh Python
module and its optional MPI-linked libraries out of the long-lived PETSc
process as far as the DOLFINx importer permits.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable


MESH_CACHE_VERSION = 2
MESH_CACHE_FORMAT = "equiband_dolfinx_mesh_cache_v2"
DEFAULT_MESH_CACHE_DIRECTORY = Path(__file__).resolve().parents[2] / ".cache" / "meshes"
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class MeshCacheSpec:
    """Complete deterministic input to one canonical mesh generation."""

    geometry: str
    mesh_size: float
    geometry_degree: int
    gmsh_algorithm: int
    gmsh_version: str
    parameters: dict[str, object]
    source_hash: str
    payload: dict[str, object]
    key: str


@dataclass(frozen=True)
class MeshCacheResult:
    """Resolved cache artifact and validated generator metadata."""

    path: Path
    metadata: dict[str, object]
    key: str
    status: str
    invalid_reason: str | None = None


def default_mesh_cache_directory() -> Path:
    """Return the project-local cache used when no directory is configured."""

    return DEFAULT_MESH_CACHE_DIRECTORY


def _fallback_mesh_cache_directory() -> Path:
    """Return a project-specific process-local fallback cache directory."""

    try:
        project = str(Path.cwd().resolve())
    except OSError:  # pragma: no cover - an inaccessible cwd is uncommon
        project = str(Path.cwd())
    digest = hashlib.sha256(project.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "diocotron" / "dolfinx_meshes" / digest


def _normalize(value):
    """Convert cache-key data to recursively sorted JSON scalar containers."""

    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def mesh_cache_spec(config) -> MeshCacheSpec:
    """Build the canonical cache key represented by ``config``.

    Disk and ellipse parameters live in the existing top-level configuration
    fields.  Other canonical geometries start from the shared builder defaults
    and accept explicit ``geometry_parameters`` overrides.
    """

    from projects.diocotron.dolfinx.geometry.canonical import (
        geometry_definition,
        geometry_source_hash,
    )

    definition = geometry_definition(config.geometry)
    parameters = dict(definition.parameters)
    if definition.slug == "disk":
        parameters["radius"] = float(config.radius)
    elif definition.slug == "ellipse":
        parameters.update(radius=float(config.radius),
                          ellipse_ratio=float(config.ellipse_ratio))
    if config.geometry_parameters:
        parameters.update(config.geometry_parameters)
    parameters = _normalize(parameters)
    source_hash = geometry_source_hash(definition.slug, parameters)
    try:
        gmsh_version = importlib_metadata.version("gmsh")
    except importlib_metadata.PackageNotFoundError:
        # Cache inspection and configuration tests deliberately work without
        # importing (or even installing) Gmsh.  Actual generation will then
        # fail with its normal dependency message rather than during keying.
        gmsh_version = "unavailable"
    payload = {
        "version": MESH_CACHE_VERSION,
        "generator": "projects.diocotron.dolfinx.geometry.canonical",
        "geometry": definition.slug,
        "mesh_size": float(config.mesh_size),
        "geometry_degree": int(config.geometry_degree),
        "gmsh_algorithm": int(config.gmsh_algorithm),
        "gmsh_version": gmsh_version,
        "optimize": True,
        "parameters": parameters,
        "source_hash": source_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return MeshCacheSpec(
        definition.slug,
        float(config.mesh_size),
        int(config.geometry_degree),
        int(config.gmsh_algorithm),
        gmsh_version,
        parameters,
        source_hash,
        payload,
        hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )


def _size_slug(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p").replace("+", "")


def mesh_cache_path(spec: MeshCacheSpec, directory: Path) -> Path:
    """Return a readable filename whose digest still authenticates all inputs."""

    name = (f"{spec.geometry}-h{_size_slug(spec.mesh_size)}-g{spec.geometry_degree}-"
            f"a{spec.gmsh_algorithm}-{spec.key}.msh")
    return Path(directory) / name


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metadata(path: Path, spec: MeshCacheSpec, *, require_cache_key: bool) -> dict[str, object]:
    """Read and authenticate one mesh/sidecar pair or raise ``ValueError``."""

    metadata_path = _sidecar(path)
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("mesh or metadata sidecar is missing")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid metadata sidecar: {type(error).__name__}") from error
    if metadata.get("geometry") != spec.geometry:
        raise ValueError("cached geometry does not match its key")
    try:
        size = float(metadata["requested_size"])
        degree = int(metadata["geometry_degree"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("cached size or geometry degree is invalid") from error
    if not math.isclose(size, spec.mesh_size, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("cached requested size does not match its key")
    if degree != spec.geometry_degree:
        raise ValueError("cached geometry degree does not match its key")
    if int(metadata.get("gmsh_algorithm", -1)) != spec.gmsh_algorithm:
        raise ValueError("cached Gmsh algorithm does not match its key")
    if str(metadata.get("gmsh_version", "unknown")) != spec.gmsh_version:
        raise ValueError("cached Gmsh version does not match its key")
    if metadata.get("source_hash") != spec.source_hash:
        raise ValueError("cached canonical geometry source hash changed")
    for name in ("vertices", "edges", "cells"):
        if not isinstance(metadata.get(name), int) or int(metadata[name]) <= 0:
            raise ValueError(f"cached {name} count is invalid")
    actual_hash = _sha256_file(path)
    if metadata.get("mesh_sha256") != actual_hash:
        raise ValueError("cached mesh SHA-256 does not match its sidecar")
    cache_record = metadata.get("equiband_cache")
    if require_cache_key and (
            not isinstance(cache_record, dict)
            or cache_record.get("format") != MESH_CACHE_FORMAT
            or cache_record.get("key") != spec.key
            or cache_record.get("payload") != spec.payload):
        raise ValueError("cached equiband key metadata does not match")
    return metadata


def _subprocess_generate(path: Path, spec: MeshCacheSpec) -> None:
    """Generate one canonical mesh without importing Gmsh in this process."""

    command = [
        sys.executable,
        "-m", "projects.diocotron.dolfinx.geometry.canonical",
        spec.geometry,
        "--mesh-size", repr(spec.mesh_size),
        "--geometry-degree", str(spec.geometry_degree),
        "--algorithm", str(spec.gmsh_algorithm),
        "--parameters-json", json.dumps(spec.parameters, separators=(",", ":")),
        "--output", str(path),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(REPO_ROOT) if not environment.get("PYTHONPATH")
        else f"{REPO_ROOT}{os.pathsep}{environment['PYTHONPATH']}"
    )
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def ensure_local_cached_mesh(
    config,
    directory: Path,
    *,
    rebuild: bool = False,
    generator: Callable[[Path, MeshCacheSpec], None] | None = None,
) -> MeshCacheResult:
    """Resolve one cache artifact in a single process.

    ``generator`` is injectable so cache semantics can be tested without Gmsh.
    Independent jobs serialize only the same cache key, not the whole cache.
    """

    spec = mesh_cache_spec(config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = mesh_cache_path(spec, directory)
    lock_path = path.with_suffix(path.suffix + ".lock")
    invalid_reason = None
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not rebuild:
            try:
                metadata = _validate_metadata(path, spec, require_cache_key=True)
            except ValueError as error:
                if path.exists() or _sidecar(path).exists():
                    invalid_reason = str(error)
            else:
                return MeshCacheResult(path.resolve(), metadata, spec.key, "hit")

        build_directory = Path(tempfile.mkdtemp(prefix=f".{spec.geometry}-", dir=directory))
        temporary_mesh = build_directory / "mesh.msh"
        try:
            (generator or _subprocess_generate)(temporary_mesh, spec)
            metadata = _validate_metadata(temporary_mesh, spec, require_cache_key=False)
            metadata["equiband_cache"] = {
                "format": MESH_CACHE_FORMAT,
                "key": spec.key,
                "payload": spec.payload,
            }
            _sidecar(temporary_mesh).write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_mesh, path)
            os.replace(_sidecar(temporary_mesh), _sidecar(path))
        finally:
            shutil.rmtree(build_directory, ignore_errors=True)
        metadata = _validate_metadata(path, spec, require_cache_key=True)
        status = "rebuild" if rebuild else ("invalid-regenerated" if invalid_reason else "miss-stored")
        return MeshCacheResult(path.resolve(), metadata, spec.key, status, invalid_reason)


def _candidate_directories(config) -> list[Path]:
    if config.mesh_cache_directory is not None:
        return [Path(config.mesh_cache_directory)]
    return [default_mesh_cache_directory(), _fallback_mesh_cache_directory()]


def ensure_cached_mesh(config, comm, *, report, rebuild: bool = False) -> MeshCacheResult:
    """Collectively resolve a canonical mesh, generating only on rank zero."""

    payload = None
    if comm.rank == 0:
        errors: list[str] = []
        try:
            for index, directory in enumerate(_candidate_directories(config)):
                try:
                    result = ensure_local_cached_mesh(config, directory, rebuild=rebuild)
                except OSError as error:
                    errors.append(f"{directory}: {type(error).__name__}: {error}")
                    if config.mesh_cache_directory is not None:
                        raise
                    continue
                if index:
                    result = MeshCacheResult(result.path, result.metadata, result.key,
                                             "fallback-" + result.status, result.invalid_reason)
                payload = {
                    "ok": True,
                    "path": str(result.path),
                    "metadata": result.metadata,
                    "key": result.key,
                    "status": result.status,
                    "invalid_reason": result.invalid_reason,
                    "cache_errors": errors,
                }
                break
            if payload is None:
                raise OSError("no writable mesh cache: " + "; ".join(errors))
        except Exception as error:  # broadcast before raising so peers never hang
            payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    payload = comm.bcast(payload, root=0)
    if not payload["ok"]:
        raise RuntimeError("MESH_CACHE_FAILED: " + payload["error"])
    result = MeshCacheResult(
        Path(payload["path"]), payload["metadata"], payload["key"],
        payload["status"], payload["invalid_reason"],
    )
    metadata = result.metadata
    from projects.diocotron.dolfinx.geometry.canonical import lagrange_dofs_from_metadata
    equilibrium_dofs = lagrange_dofs_from_metadata(metadata, config.degree)
    torsion_dofs = lagrange_dofs_from_metadata(metadata, config.torsion_degree)
    report(
        f"MESH_CACHE status={result.status} geometry={config.geometry} "
        f"requested_size={config.mesh_size:.10g} geometry_degree={config.geometry_degree} "
        f"key={result.key} path={result.path}",
        level=0,
    )
    if result.invalid_reason:
        report(f"MESH_CACHE_REGENERATED reason={result.invalid_reason}", level=0)
    for error in payload.get("cache_errors", []):
        report(f"MESH_CACHE_FALLBACK primary_error={error}", level=0)
    report(
        f"MESH_DOF_ESTIMATE cells={metadata['cells']} equilibrium_degree={config.degree} "
        f"equilibrium_dofs={equilibrium_dofs} torsion_degree={config.torsion_degree} "
        f"torsion_dofs={torsion_dofs}",
        level=0,
    )
    maximum_dofs = max(equilibrium_dofs, torsion_dofs)
    highest_degree = max(config.degree, config.torsion_degree)
    recommended = recommended_mpi_ranks(maximum_dofs, highest_degree)
    comparison = "12,16" if maximum_dofs > 250_000 else "none"
    status = "below" if comm.size < recommended else (
        "recommended" if comm.size == recommended else "above"
    )
    report(
        f"MPI_RANK_POLICY current_ranks={comm.size} recommended_start={recommended} "
        f"status={status} largest_scalar_space_dofs={maximum_dofs} "
        f"highest_degree={highest_degree} comparison_ranks={comparison} "
        "threads_per_rank=1 benchmark_for_this_machine=1",
        level=0,
    )
    return result


def recommended_mpi_ranks(global_dofs: int, polynomial_degree: int) -> int:
    """Apply the repository's preliminary MUMPS rank policy.

    The recommendation is intentionally a starting point rather than a claim
    of portable optimality.  The workstation policy asks users to compare 12
    and 16 ranks above 250,000 global scalar DOFs; that comparison is emitted
    separately by :func:`ensure_cached_mesh` while this function returns 8.
    """

    dofs = int(global_dofs)
    degree = int(polynomial_degree)
    if dofs < 0 or degree < 1:
        raise ValueError("global_dofs must be nonnegative and polynomial_degree positive")
    if dofs < 20_000:
        return 1
    if dofs < 40_000:
        return 4 if degree >= 5 else 2
    if dofs < 120_000:
        return 4
    return 8
