"""Run by tests/test_adr_h200_tuned_device_system.py in its own process (the vendored ``scripts``
package must shadow master's). With NumPy as the array module:

- the vendored route's device-output mode gives the host-output mode's arrays, and the chunk
  hook's w, M equal L_u s and L_u B;
- the device-built operator layout, ASM local matrices and incidence slots equal the host builders;
- the one-way PyPardiso export round-trips and hashes like the campaign's operator_hash;
- the verifier built from the device assembly agrees with the vendored host verify.
"""
import hashlib
import sys
import types
from pathlib import Path

import numpy as np

root, scratch = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root/'vendor/adr_gmres'))
sys.path.append(str(root))

from scipy.sparse.linalg import spsolve
from scripts.adr_performance_worker import get_space, verify, save_cache, load_cache
from scripts.adv_diff_rea_cases import get_case
from hdgfem.backends.cupy_adr_assembly import assemble_adr_device
from hdgfem.solvers.adv_diff_rea import assemble_advection_diffusion_reaction, _csr
from hybridge.linalg.face_dense import FaceDenseSystem as HostSystem
from hybridge.linalg.gpu.face_dense import prepare_face_dense_batch_layout
from hybridge.linalg.additive_schwarz import build_face_additive_schwarz_local_matrices
from hybridge.linalg.gpu.preconditioners import build_face_additive_schwarz_incidence_slots
from campaigns.adr_h200_tuned.device_system import (
    asm_incidence_slots_on_device, asm_local_matrices_on_device, assemble_on_device, export_for_pardiso,
    operator_batches_on_device, retained_bytes,
)
from campaigns.adr_h200_tuned.device_verify import DeviceVerifier


class _Device:
    id = 0

    def synchronize(self):
        pass


shim = types.SimpleNamespace(**{name: getattr(np, name) for name in dir(np) if not name.startswith('_')})
shim.cuda = types.SimpleNamespace(Device=_Device)
shim.asnumpy = np.asarray

for case, p in (('trigonometric', 2), ('variable_velocity', 3), ('anisotropic', 4)):
    cache = scratch/f'{case}_{p}'
    spec = dict(n=5, p=p, case=case, rtol=1e-10, cache=str(cache), mesh_path=None)
    space = get_space(spec)
    kw, exact = get_case(case)
    P, F = space.el_dof, space.quad_data.edg_dof
    host = assemble_adr_device(kw['source'], kw['beta'], kw['reaction'], kw['boundary_condition'], space,
                               diffusion=kw['diffusion'], xp=np)
    assembly = assemble_on_device(shim, space, kw, xp=np, log=lambda *_: None)
    system = assembly.system
    for name in ('blocks', 'neighbors', 'rhs', 'global_faces', 'global_to_local', 'boundary_trace'):
        assert np.array_equal(getattr(system, name), getattr(host['system'], name)), (case, name)
    assert np.array_equal(assembly.element_blocks, host['element_blocks'])
    inv, b, s = host['local_solver'], host['boundary'], host['source_rhs']
    assert np.array_equal(assembly.w, (inv[:, :P, :] @ s[..., None])[..., 0])
    assert np.array_equal(assembly.M, inv[:, :P, :] @ b)
    reference = assemble_advection_diffusion_reaction(space=space, backend='numpy', **kw)
    assert np.allclose(system.blocks, reference.system.blocks, rtol=0, atol=1e-10*np.abs(reference.system.blocks).max())

    batches, neighbors32 = operator_batches_on_device(shim, system)
    layout = prepare_face_dense_batch_layout(host['system'].blocks, host['system'].neighbors)
    assert np.array_equal(batches, layout.matrix_batches) and np.array_equal(neighbors32, layout.neighbors)

    host_system = HostSystem(blocks=host['system'].blocks, neighbors=host['system'].neighbors, rhs=host['system'].rhs,
                             global_faces=host['system'].global_faces, global_to_local=host['system'].global_to_local,
                             boundary_trace=host['system'].boundary_trace, mode='eliminate')
    loc2glob = space.mesh.loc2glob_edge
    local = build_face_additive_schwarz_local_matrices(host_system, host['element_blocks'], loc2glob)
    matrices, faces = asm_local_matrices_on_device(shim, system, assembly.element_blocks, assembly.loc2glob)
    assert np.array_equal(matrices, local.local_matrices) and np.array_equal(faces, local.element_system_faces)
    slots = asm_incidence_slots_on_device(shim, faces, system.num_rows)
    assert np.array_equal(slots, build_face_additive_schwarz_incidence_slots(local.element_system_faces, system.num_rows))

    files, operator_sha256, _ = export_for_pardiso(shim, cache, system, chunk_rows=7, log=lambda *_: None)
    digest = hashlib.sha256()
    for name in ('system_blocks', 'system_rhs'):
        digest.update(memoryview(np.load(cache/f'{name}.npy', mmap_mode='r')).cast('B'))
    assert operator_sha256 == digest.hexdigest()
    assert set(files) == {f'system_{n}.npy' for n in ('blocks', 'neighbors', 'rhs', 'global_faces', 'global_to_local', 'boundary_trace')}
    assert np.array_equal(np.load(cache/'system_blocks.npy'), host['system'].blocks)

    # The verifier from the device assembly against the vendored host verify (host cache for the latter).
    reference_cache = scratch/f'{case}_{p}_host'
    save_cache(reference_cache, types.SimpleNamespace(system=host['system'], element_blocks=host['element_blocks'],
                                                       local_solver=inv, boundary=b, source_rhs=s), space)
    host_loaded, get = load_cache(reference_cache)
    solution = spsolve(_csr(host_loaded), np.asarray(host_loaded.rhs).reshape(-1))
    checker = DeviceVerifier.from_device(shim, spec, assembly, space, exact, log=lambda *_: None)
    assert checker.exact_on_device and checker.resident
    bn = float(np.linalg.norm(np.asarray(host_loaded.rhs)))
    for x in (solution, solution*(1+1e-6*np.random.default_rng(1).standard_normal(solution.size))):
        expected = verify(spec, space, host_loaded, get, x)
        got = checker.verify(x)
        assert expected['passed'] == got['passed'], (case, expected, got)
        assert abs(expected['true_residual_norm']-got['true_residual_norm']) <= 1e-13*bn
        assert abs(expected['l2_error']-got['l2_error']) <= max(1e-12*expected['l2_error'], 1e-15), (case, expected['l2_error'], got['l2_error'])
    sizes = retained_bytes(space.mesh.num_tri, p)
    assert sizes['after_asm'] < sizes['during_asm']
    print(case, p, 'device-output route, device solver inputs and export match the host code')
