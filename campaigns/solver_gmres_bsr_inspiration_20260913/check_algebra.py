"""CPU-only evidence for the ASM/BSR review; this is not a solver backend.

Reuse the package's face topology, assembly, Schwarz, and polynomial references.
Numba compilation is forbidden. No CuPy kernels or time integration are run.
Dense matrices occur only as tiny algebraic oracles; applications use SciPy BSR.
"""
from __future__ import annotations

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))
_campaign_output = _campaign_data_directory('hybridge/artifacts/gmres_bsr_inspiration_20260913')


import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import bsr_matrix

ROOT = _campaign_workspace_path('hybridge')
OUTPUT = _campaign_output


def forbid_compilation(*args, **kwargs):
    raise RuntimeError("Native compilation is forbidden in this review diagnostic")


def relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300))


def pack_bsr(system):
    """Pack existing dense blocks directly, with no coefficient-expanded CSR."""
    valid = system.neighbors >= 0
    indptr = np.concatenate(([0], np.cumsum(valid.sum(axis=1)))).astype(np.int32)
    indices = np.ascontiguousarray(system.neighbors[valid], dtype=np.int32)
    values = np.ascontiguousarray(system.blocks[valid])
    return bsr_matrix((values, indices, indptr), shape=(system.num_dofs, system.num_dofs))


def terminal_shortcut(reference, vector):
    """Algebra-only prototype of omitting the unused final residual update."""
    base = reference.base_preconditioner
    q = vector.copy() if base is None else base(vector)
    result = np.zeros_like(q)
    counts = {"A": 0, "C": int(base is not None)}

    def effective(source):
        value = reference.matvec(source)
        counts["A"] += 1
        if base is not None:
            value = base(value)
            counts["C"] += 1
        return value

    for index, root in enumerate(reference.root_blocks):
        last = index == len(reference.root_blocks) - 1
        if root.is_real:
            theta = float(root.first.real)
            result += q / theta
            if not last:
                q -= effective(q) / theta
        else:
            a, b = float(root.first.real), float(abs(root.first.imag))
            denominator = a * a + b * b
            bq = effective(q)
            result += (2.0 * a * q - bq) / denominator
            if not last:
                b2q = effective(bq)
                q += (-2.0 * a * bq + b2q) / denominator
    return result, counts


def main():
    from hdgfem.assembly.face_dense import (
        assemble_global_face_blocks,
        build_face_topology,
        eliminate_dirichlet_faces,
    )
    from hdgfem.core.mesh import rectangle_mesh
    from hdgfem.linalg.additive_schwarz import build_face_additive_schwarz_preconditioner
    from hdgfem.linalg.polynomial import PolynomialPreconditioner

    rng = np.random.default_rng(20260913)
    cases = []
    pp_cases = []
    for mesh_shape in ((3, 2), (4, 3)):
        mesh = rectangle_mesh(*mesh_shape)
        connectivity = mesh.loc2glob_edge
        topology = build_face_topology(connectivity)
        ne = connectivity.shape[0]
        for q in (2, 3, 5, 6, 7):
            local = rng.standard_normal((ne, 3*q, 3*q))
            local = local @ local.transpose(0, 2, 1) + np.eye(3*q)[None] * (3*q)
            element_blocks = local.reshape(ne, 3, q, 3, q).transpose(0, 1, 3, 2, 4).copy()
            global_blocks = assemble_global_face_blocks(element_blocks, connectivity, topology)
            zeros = np.zeros((topology.num_faces, q))
            for boundary_mode in ("all_faces_active", "dirichlet_eliminated"):
                free_faces = np.arange(topology.num_faces) if boundary_mode == "all_faces_active" else np.flatnonzero(topology.incidence_count == 2)
                # Exercise a nontrivial reduced row numbering as well as overlap.
                free_faces = rng.permutation(free_faces)
                system = eliminate_dirichlet_faces(global_blocks, topology, zeros, zeros, free_faces)
                asm = build_face_additive_schwarz_preconditioner(system, element_blocks, connectivity)

                # The proposed BSR ASM matrix uses the EXISTING block assembler.
                inverse_blocks = asm.inverse_matrices.reshape(ne, 3, q, 3, q).transpose(0, 1, 3, 2, 4).copy()
                assembled_inverse = assemble_global_face_blocks(inverse_blocks, connectivity, topology)
                c_system = eliminate_dirichlet_faces(assembled_inverse, topology, zeros, zeros, free_faces)
                A, C = pack_bsr(system), pack_bsr(c_system)
                assert np.array_equal(A.indptr, C.indptr)
                assert np.array_equal(A.indices, C.indices)
                np.testing.assert_array_equal(system.neighbors, c_system.neighbors)
                a_dense, c_dense = A.toarray(), C.toarray()
                x = rng.standard_normal(system.num_dofs)
                asm_error = relative_error(C @ x, asm.apply(x))
                assert asm_error < 2e-13
                local_error = 0.0
                for e, row_faces in enumerate(asm.element_system_faces):
                    active = np.flatnonzero(row_faces >= 0)
                    if active.size:
                        ldofs = (active[:, None] * q + np.arange(q)).reshape(-1)
                        gdofs = (row_faces[active, None] * q + np.arange(q)).reshape(-1)
                        local_error = max(local_error, relative_error(asm.local_matrices[e][np.ix_(ldofs, ldofs)], a_dense[np.ix_(gdofs, gdofs)]))
                assert local_error < 2e-13
                symmetry = relative_error(c_dense, c_dense.T)
                minimum_eigenvalue = float(np.linalg.eigvalsh(c_dense).min())
                assert symmetry < 2e-13 and minimum_eigenvalue > 0.0

                # Tiny two-level oracle: fixed damped ASM, Galerkin correction,
                # and adjoint post-smoothing give a symmetric positive action.
                nf = system.num_rows
                nc = (nf + 2) // 3
                aggregate = np.arange(nf) // 3
                P = bsr_matrix((np.tile(np.eye(q), (nf, 1, 1)), aggregate.astype(np.int32), np.arange(nf + 1, dtype=np.int32)), shape=(nf*q, nc*q))
                p_dense = P.toarray()
                ac = p_dense.T @ a_dense @ p_dense
                chol_c = np.linalg.cholesky(c_dense)
                eigenvalues = np.linalg.eigvalsh(chol_c.T @ a_dense @ chol_c)
                omega = 1.0 / float(eigenvalues.max())
                S = omega * c_dense
                F = np.eye(system.num_dofs) - a_dense @ S
                G = p_dense @ np.linalg.solve(ac, p_dense.T)
                H = S + S.T - S.T @ a_dense @ S + F.T @ G @ F
                h_symmetry = relative_error(H, H.T)
                h_minimum = float(np.linalg.eigvalsh(H).min())
                z = S @ x
                z += p_dense @ np.linalg.solve(ac, p_dense.T @ (x - A @ z))
                z += S.T @ (x - A @ z)
                cycle_error = relative_error(z, H @ x)
                assert h_symmetry < 2e-13 and h_minimum > 0.0 and cycle_error < 2e-13
                cases.append({"mesh_shape": mesh_shape, "triangles": ne, "block_size": q, "boundary_mode": boundary_mode, "scalar_dimension": system.num_dofs, "face_blocks": int(C.indices.size), "same_bsr_pattern": True, "asm_action_relative_error": asm_error, "patch_principal_relative_error": local_error, "asm_symmetry_relative_error": symmetry, "asm_minimum_eigenvalue": minimum_eigenvalue, "two_level_symmetry_relative_error": h_symmetry, "two_level_minimum_eigenvalue": h_minimum, "two_level_action_relative_error": cycle_error})

                if mesh_shape == (3, 2) and boundary_mode == "dirichlet_eliminated":
                    for base_enabled in (False, True):
                        effective_matrix = c_dense @ a_dense if base_enabled else a_dense
                        scale = float(max(np.max(np.abs(np.linalg.eigvals(effective_matrix))), 1.0))
                        root_sets = [np.array([1.1]), np.array([0.8+0.3j, 0.8-0.3j]), np.array([1.1, 0.8+0.3j, 0.8-0.3j]), np.array([0.8+0.3j, 0.8-0.3j, 0.9, 1.2, 1.1])]
                        for roots in root_sets:
                            roots = roots * scale
                            observed = {"A": 0, "C": 0}
                            def mv(v):
                                observed["A"] += 1
                                return A @ v
                            def base(v):
                                observed["C"] += 1
                                return C @ v
                            reference = PolynomialPreconditioner(mv, size=system.num_dofs, roots=roots, base_preconditioner=base if base_enabled else None)
                            old = reference.apply(x)
                            old_counts = observed.copy()
                            new, counts = terminal_shortcut(reference, x)
                            q_complex = (C @ x if base_enabled else x).astype(complex)
                            exact = np.zeros_like(q_complex)
                            for theta in roots:
                                exact += q_complex / theta
                                q_complex = q_complex - effective_matrix @ q_complex / theta
                            old_expected = {"A": len(roots), "C": len(roots)+1 if base_enabled else 0}
                            new_expected = {"A": len(roots)-1, "C": len(roots) if base_enabled else 0}
                            assert old_counts == old_expected and counts == new_expected
                            shortcut_error = relative_error(new, old)
                            complex_error = relative_error(new, exact)
                            assert shortcut_error < 2e-13 and complex_error < 2e-13
                            pp_cases.append({"block_size": q, "with_asm": base_enabled, "degree": len(roots), "terminal_root_kind": "real" if reference.root_blocks[-1].is_real else "conjugate_pair", "old_calls": old_counts, "new_calls": counts, "shortcut_relative_error": shortcut_error, "complex_oracle_relative_error": complex_error})

    result = {"scope": "Small synthetic SPD element blocks on real triangular connectivity; algebra checks only, not HDG convergence or GPU performance.", "compilation": "Forbidden by active Numba compiler guard; no CUDA kernel was invoked.", "mesh_steps": "No time integration or PDE simulation.", "asm_case_count": len(cases), "polynomial_case_count": len(pp_cases), "cases": cases, "polynomial_cases": pp_cases, "max_asm_relative_error": max(c['asm_action_relative_error'] for c in cases), "max_shortcut_relative_error": max(c['shortcut_relative_error'] for c in pp_cases), "max_complex_oracle_relative_error": max(c['complex_oracle_relative_error'] for c in pp_cases), "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('hdgfem/assembly/face_dense.py', 'hdgfem/linalg/additive_schwarz.py', 'hdgfem/linalg/polynomial.py')}, "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUTPUT / 'algebra_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('cases', 'polynomial_cases', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    with patch('numba.core.dispatcher._FunctionCompiler._compile_core', forbid_compilation):
        main()
