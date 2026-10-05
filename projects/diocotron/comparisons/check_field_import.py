#!/usr/bin/env python3
"""Standalone DOLFINx-field checkpoint and DG projection round-trip check.

Run this script with a Python environment containing DOLFINx, for example::

    /path/to/fenicsx/bin/python projects/diocotron/comparisons/check_field_import.py

The check writes generic named scalar fields, imports them without using
DOLFINx data structures, permutes mesh nodes/cells, and projects into every
supported HDGFEM DG basis.  It also exercises the ``rho``/``phi`` equilibrium
convenience wrapper and its nonconverged-state policy.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path
import shutil
import sys
import tempfile

import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hdgfem.core import DGMesh, DGSpace  # noqa: E402
from projects.diocotron.dolfinx.checkpoint import (  # noqa: E402
    write_dolfinx_checkpoint_v2,
    write_equilibrium_checkpoint_v2,
)
from projects.diocotron.comparisons.hdg_projection import (  # noqa: E402
    load_dolfinx_checkpoint,
    load_dolfinx_equilibrium,
)


def temperature_exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the degree-three generic scalar test field."""
    return 1.0 + 2.0 * x - 3.0 * y + 0.5 * x * y + 0.25 * x**3 - 0.4 * y**2


def streamfunction_exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the degree-two generic scalar test field."""
    return 0.25 + x**2 + y**2 + 0.75 * x * y


def permuted_mesh(source: DGMesh) -> DGMesh:
    """Return the same geometry with shuffled nodes, cells, and local vertices."""
    generator = np.random.default_rng(41273)
    new_to_old = generator.permutation(source.node_coords.shape[0])
    old_to_new = np.empty_like(new_to_old)
    old_to_new[new_to_old] = np.arange(new_to_old.size)
    triangles = old_to_new[source.triangles]
    triangles = triangles[generator.permutation(triangles.shape[0])].copy()
    flipped = np.arange(triangles.shape[0]) % 2 == 0
    triangles[flipped, 1], triangles[flipped, 2] = (
        triangles[flipped, 2].copy(),
        triangles[flipped, 1].copy(),
    )
    return DGMesh.from_arrays(source.node_coords[new_to_old], triangles)


def _check_generic_checkpoint(path: Path, domain, order: int) -> None:
    """Write arbitrary named DOLFINx fields and verify generic DG import."""
    element = basix.ufl.element("P", domain.basix_cell(), order)
    space = fem.functionspace(domain, element)
    temperature = fem.Function(space, name="temperature")
    streamfunction = fem.Function(space, name="streamfunction")
    temperature.interpolate(lambda x: temperature_exact(x[0], x[1]))
    streamfunction.interpolate(lambda x: streamfunction_exact(x[0], x[1]))
    write_dolfinx_checkpoint_v2(
        path,
        {"temperature": temperature, "streamfunction": streamfunction},
        metadata={"case": "standalone_generic_roundtrip", "final_status": "OK"},
    )

    checkpoint = load_dolfinx_checkpoint(path)
    target_mesh = permuted_mesh(checkpoint.to_mesh())
    reference_points = np.asarray(
        [
            [-0.83, -0.91],
            [-0.31, -0.52],
            [0.18, -0.73],
            [-0.62, 0.24],
            [-0.05, -0.08],
        ],
        dtype=np.float64,
    )
    physical = target_mesh.map_reference_points(reference_points)
    exact_temperature = temperature_exact(physical[:, :, 0], physical[:, :, 1])
    exact_streamfunction = streamfunction_exact(physical[:, :, 0], physical[:, :, 1])

    maximum_error = 0.0
    for basis_type in ("bernstein", "hier_C0", "dub_orth"):
        for target_order in (1, order, order + 1):
            target_space = DGSpace(target_mesh, target_order, basis_type=basis_type)
            imported = checkpoint.project(target_space)
            for name in checkpoint.field_names:
                if not np.all(np.isfinite(imported[name].coeffs)):
                    raise AssertionError(
                        f"non-finite {name} coefficients for {basis_type} P{target_order}"
                    )
            if target_order >= order:
                temperature_error = float(
                    np.max(
                        np.abs(
                            imported["temperature"].evaluate(reference_points, reference=True)
                            - exact_temperature
                        )
                    )
                )
                streamfunction_error = float(
                    np.max(
                        np.abs(
                            imported["streamfunction"].evaluate(reference_points, reference=True)
                            - exact_streamfunction
                        )
                    )
                )
                maximum_error = max(maximum_error, temperature_error, streamfunction_error)
            if imported.diagnostics.matched_cells != target_mesh.num_tri:
                raise AssertionError("cell mapping is incomplete")
    if maximum_error > 2.0e-10:
        raise AssertionError(f"generic DOLFINx-to-DGField round-trip error {maximum_error:.3e}")
    if domain.comm.rank == 0:
        print(
            "GENERIC_IMPORT_OK "
            f"fields={','.join(checkpoint.field_names)} cells={target_mesh.num_tri} "
            f"maxError={maximum_error:.3e}",
            flush=True,
        )


def _check_equilibrium_checkpoint(path: Path, domain, order: int) -> None:
    """Verify the equilibrium facade and default failed-state rejection."""
    element = basix.ufl.element("P", domain.basix_cell(), order)
    space = fem.functionspace(domain, element)
    rho = fem.Function(space, name="rho")
    phi = fem.Function(space, name="phi")
    rho.interpolate(lambda x: 2.0 + streamfunction_exact(x[0], x[1]))
    phi.interpolate(lambda x: temperature_exact(x[0], x[1]))
    write_equilibrium_checkpoint_v2(
        path,
        rho=rho,
        phi=phi,
        metadata={
            "case": "standalone_equilibrium_roundtrip",
            "final_status": "NONCONVERGED",
            "final_residual": 1.0e-4,
        },
    )
    try:
        load_dolfinx_equilibrium(path)
    except ValueError as exc:
        if "allow_nonconverged=True" not in str(exc):
            raise
    else:
        raise AssertionError("nonconverged equilibrium was accepted without an override")
    equilibrium = load_dolfinx_equilibrium(path, allow_nonconverged=True)
    target_space = DGSpace(equilibrium.to_mesh(), order, basis_type="bernstein")
    imported = equilibrium.project(target_space)
    if imported.density.name != "rho" or imported.potential.name != "phi":
        raise AssertionError("equilibrium facade did not preserve rho/phi field names")
    if domain.comm.rank == 0:
        print("EQUILIBRIUM_POLICY_OK status=NONCONVERGED overrideAccepted=1", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone round-trip command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="keep the generic v2 artifact here")
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--mesh-cells", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the standalone generic and equilibrium round-trip checks."""
    args = build_parser().parse_args(argv)
    if args.order < 3:
        raise ValueError("--order must be at least three for the exact cubic field check")
    if args.mesh_cells < 1:
        raise ValueError("--mesh-cells must be positive")
    comm = MPI.COMM_WORLD
    temporary_directory = None
    if comm.rank == 0:
        if args.output is None:
            temporary_directory = Path(tempfile.mkdtemp(prefix="hdgfem_dolfinx_import_"))
            generic_path = temporary_directory / "generic_fields.npz"
        else:
            generic_path = args.output.expanduser().resolve()
    else:
        generic_path = None
    generic_path = Path(comm.bcast(str(generic_path), root=0))
    equilibrium_path = generic_path.with_name(f"{generic_path.stem}.equilibrium.npz")
    domain = mesh.create_unit_square(
        comm,
        args.mesh_cells,
        args.mesh_cells,
        cell_type=mesh.CellType.triangle,
    )
    try:
        _check_generic_checkpoint(generic_path, domain, args.order)
        _check_equilibrium_checkpoint(equilibrium_path, domain, args.order)
        if comm.rank == 0:
            print(f"DOLFINX_FIELD_IMPORT_PASS artifact={generic_path}", flush=True)
    finally:
        comm.barrier()
        if comm.rank == 0 and temporary_directory is not None:
            shutil.rmtree(temporary_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
