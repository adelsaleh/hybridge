"""Inspect small discontinuous-advection trace matrices without solving a PDE.

Run with NUMBA_DISABLE_JIT=1 for host checks without native compilation.
CuPy/raw-CUDA assembly is explicit and may invoke CuPy runtime compilation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from hdgfem import DGSpace, VectorDGField, rectangle_mesh
from hdgfem.hdg import matrices as mats
import hdgfem.transport.local_numpy as transport_local_numpy
import hdgfem.hdg.coefficients as hdg_coefficients
import hdgfem.hdg.stabilization as hdg_stabilization
import hdgfem.core.mass as core_mass
from hdgfem.linalg.system import assemble_global_matrix
from hdgfem.transport.diagnostics import trace_inflow_diagnostics
from hdgfem.linalg.failure_snapshot import trace_matrix_diagnostics
from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver


AVERAGED = "conflict-averaged-upwind"
SCENARIOS = ("standard", "averaged", "reaction-20", "continuous")


def discontinuous_advection_fields(space: DGSpace):
    """Return the original GPU parity fixture, including its converging jump."""
    beta_x = lambda x, y: np.where(x < 0.0, 2.0, -1.0) + 0.2 * y
    beta_y = lambda x, y: 0.1 + 0.05 * x
    source = lambda x, y: 1.0 + 0.2 * x - 0.1 * y
    reaction = lambda x, y: 2.0 + 0.01 * x * y
    boundary = lambda x, y: 0.5 * x + 0.75 * y
    beta_h = VectorDGField(
        (space.project_callable(beta_x, name="beta_x_h"),
         space.project_callable(beta_y, name="beta_y_h")), name="beta_h",
    )
    return (source, reaction, space.project_callable(source, name="source_h"),
            space.project_callable(reaction, name="reaction_h"), beta_h, boundary)


@dataclass
class FixtureAssembly:
    """Small assembled system and its coefficient/face provenance."""

    space: DGSpace
    beta: VectorDGField
    reaction: object
    result: object
    trace_basis: str
    policy: str | None
    scenario: str

    def matrix(self):
        """Materialize the explicitly requested tiny diagnostic CSR matrix."""
        result = self.result
        return assemble_global_matrix(
            result.solve_matrix_rows, result.solve_matrix_cols, result.solve_matrix_data,
            result.solve_rhs.size,
        )


def assemble_fixture(*, degree=3, trace_basis="legacy-lagrange", backend="numpy",
                     boundary_mode="eliminate", scenario="standard", nx=2):
    """Assemble the original fixture or one controlled variation; never solve."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown diagnostic scenario: {scenario}")
    if degree not in (1, 2, 3) or nx not in (1, 2):
        raise ValueError("This diagnostic is bounded to p=1,2,3 and one or two rectangles")
    space = DGSpace(rectangle_mesh(nx, 1, xlim=(-1., 1.), ylim=(0., 1.)), degree,
                    basis_type="dub_orth", volume_quad_1d=2 * degree + 2)
    _, _, source, reaction, beta, boundary = discontinuous_advection_fields(space)
    policy = AVERAGED if scenario == "averaged" else None
    if scenario == "reaction-20":
        reaction = space.constant(20., name="reaction_h")
    elif scenario == "continuous":
        beta = VectorDGField((space.project_callable(lambda x, y: 1. + .2*y), beta.components[1]))
    solver = AdvectionReactionHDGSolver(
        space, source=source, reaction=reaction, beta=beta, boundary_condition=boundary,
        assembly_backend=backend, boundary_mode=boundary_mode, trace_basis=trace_basis,
        advection_stabilization=policy, raw_local_assembly="fused", raw_lu_mode="coop",
        raw_block_size=64, solver="direct", preconditioner=None, scale_system=False,
        materialize_host_system=True, verbose=False,
    )
    result = solver.assemble_trace_system()
    return FixtureAssembly(space, beta, reaction, result, trace_basis, policy, scenario)


def diagnose(assembly: FixtureAssembly):
    """Combine existing face audits with a row-equilibrated small-matrix SVD."""
    space, result = assembly.space, assembly.result
    mesh, trace = space.mesh, space.trace_space(assembly.trace_basis)
    matrix = assembly.matrix()
    dense = matrix.toarray()
    # Exterior penalty rows are O(1e20). Scaling them avoids misclassifying
    # ordinary O(1) interior singular values as additional nullspace modes.
    row_scale = np.max(np.abs(dense), axis=1)
    scaled = dense / np.where(row_scale > 0, row_scale, 1.)[:, None]
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = max(scaled.shape) * np.finfo(scaled.dtype).eps * singular[0]
    normal = hdg_coefficients.advective_boundary_normal(assembly.beta, space, trace_space=trace)
    face_report = trace_inflow_diagnostics(
        normal, mesh.loc2glob_edge, mesh.orientations, mesh.int_edges_inds,
        trace.bas1d_of_ref_edg_qds, trace.weights, stabilization=assembly.policy,
    )
    tau, gamma = hdg_stabilization.advection_trace_weights_from_normal_flux(
        space, normal, assembly.policy, trace_space=trace)
    local = np.ascontiguousarray(mats.boundary_mass_from_trace_stabilization(space, tau, trace_space=trace))
    core_mass.add_reaction_mass(local, assembly.reaction, space)
    transport_local_numpy.add_advection_mats(local, space, assembly.beta, scale=-1.)
    local_singular = np.linalg.svd(local, compute_uv=False)
    coupling = mats.element_boundary_mats_from_trace_weight(space, gamma, trace_space=trace)
    seam = mesh.int_edges_inds[np.all(mesh.node_coords[mesh.edges[mesh.int_edges_inds], 0] == 0., axis=1)]
    columns = (seam[:, None] * trace.edg_dof + np.arange(trace.edg_dof)[None, :]).ravel()
    if result.reduction is not None:
        columns = result.reduction.old_to_new[columns]
    seam_coupling = []
    for edge in seam:
        for side in mesh.edge_side_indices[edge]:
            element, face = divmod(int(side), 3)
            seam_coupling.append(coupling[element, :, face*trace.edg_dof:(face+1)*trace.edg_dof])
    report = trace_matrix_diagnostics({
        "matrix_format": "csr", "data": matrix.data, "indices": matrix.indices,
        "indptr": matrix.indptr, "rhs": result.solve_rhs,
    })
    report.update(
        scenario=assembly.scenario, backend=result.assembly_backend, degree=space.order,
        trace_basis=assembly.trace_basis, boundary_mode=result.boundary_mode,
        triangles=mesh.num_tri, advection_stabilization=assembly.policy or "upwind",
        row_scaled_rank=int(np.count_nonzero(singular > tolerance)),
        row_scaled_svd_tolerance=float(tolerance),
        row_scaled_smallest_singular_value=float(singular[-1]),
        local_min_rcond=float(np.min(local_singular[:, -1] / local_singular[:, 0])),
        reaction_min=float(np.min(assembly.reaction.values())),
        seam_edges=seam.tolist(), seam_trace_columns=columns.tolist(),
        seam_column_abs_max=float(np.max(np.abs(dense[:, columns]))) if columns.size else None,
        seam_local_coupling_abs_max=float(np.max(np.abs(seam_coupling))) if seam_coupling else None,
        trace_inflow_diagnostics=face_report,
        assembly_only=result.field is None and result.trace is None and result.global_solve_result is None,
    )
    return report


def main(argv=None):
    """Write reproducible host/device matrix evidence for the bounded fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=("numpy", "numba", "cupy", "raw-cuda"),
                        default=["numpy", "numba"])
    parser.add_argument("--degrees", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    reports = []
    for backend in args.backends:
        boundaries = ("eliminate",) if backend == "raw-cuda" else ("eliminate", "penalty")
        for degree in args.degrees:
            for basis in ("legacy-lagrange", "legendre-modal"):
                for boundary in boundaries:
                    for scenario in SCENARIOS:
                        assembly = assemble_fixture(degree=degree, backend=backend, trace_basis=basis,
                                                    boundary_mode=boundary, scenario=scenario)
                        reports.append(diagnose(assembly))
    evidence = {"numpy_version": np.__version__, "reports": reports}
    text = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Recorded {len(reports)} assembly-only matrix diagnostics in {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
