"""Readers of run records shared by the diocotron and magnetron analysis and figure scripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def records(path):
    """The JSONL records of a run (one dict per non-empty line)."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def series(records, key):
    """One record key as a float array, ``nan`` where the record has no value."""
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in records], dtype=float)


def raster(npz, pixels):
    """Raster geometry, sampling matrix and outside mask of a snapshot file's mesh and space."""
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from hybridge.io.raster import RasterGeometry

    mesh = DGMesh(npz["node_coords"], npz["triangles"])
    space = DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]))
    geometry = RasterGeometry.from_mesh(mesh, pixels, pixels)
    mask = (geometry.element_ids < 0).reshape(geometry.height, geometry.width)
    return geometry, geometry.sampling_matrix(space), mask


def image(ax, values, geometry, mask):
    """Panel ``(ax, image, extent)`` for ``draw_signed_density`` (one scale per figure)."""
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax, np.ma.array(values, mask=mask), geometry.bounds
