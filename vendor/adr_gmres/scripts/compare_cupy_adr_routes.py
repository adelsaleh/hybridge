"""Parity of the device-resident CuPy ADR assembly against the host-prep CuPy, NumPy and Numba routes.

For every case and degree, assemble on a small rectangle mesh with each route
and report max relative differences ``||x-y||/||y||`` of the local matrix,
local inverse, trace coupling, source moments, condensed element blocks,
eliminated system blocks, system RHS and Dirichlet trace. Integer outputs
(neighbors, global-to-local map) must match exactly. The host-prep CuPy versus
NumPy row is the floor set by different BLAS summation orders.

    python vendor/adr_gmres/scripts/compare_cupy_adr_routes.py --output DIR
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

BRANCH = Path(__file__).resolve().parents[1]
FIELDS = ('local_matrix', 'local_solver', 'boundary', 'source_rhs', 'element_blocks',
          'system.blocks', 'system.rhs', 'system.boundary_trace')


def variable_tensor_case():
    """Variable anisotropic diffusion and velocity, nonpolynomial BC (the Numba parity test's data)."""
    import numpy as np
    angle = lambda x, y: .7*x-.3*y
    eps = 1e-6
    k00 = lambda x, y: np.cos(angle(x, y))**2+eps*np.sin(angle(x, y))**2
    k01 = lambda x, y: (1-eps)*np.cos(angle(x, y))*np.sin(angle(x, y))
    k11 = lambda x, y: np.sin(angle(x, y))**2+eps*np.cos(angle(x, y))**2
    return dict(source=lambda x, y: np.cos(2*x+y), reaction=.4,
                beta=(lambda x, y: 2+np.sin(y), lambda x, y: -1+np.cos(x)),
                diffusion=((k00, k01), (k01, k11)),
                boundary_condition=lambda x, y: np.exp(.2*x-.1*y), stabilization=.7)


def host_only_case():
    """Coefficient callables that reject CuPy arrays, to exercise the host-evaluation fallback."""
    import numpy as np
    kw = variable_tensor_case()
    source = kw['source']
    kw['source'] = lambda x, y: source(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return kw


def get(assembly, name):
    value = assembly
    for part in name.split('.'):
        value = getattr(value, part)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('HYBRIDGE_STUDY_OUT', '.')))
    parser.add_argument('--cases', nargs='+', default=['trigonometric', 'variable_velocity', 'anisotropic',
                                                      'advection_dominated', 'variable_tensor', 'host_only'])
    parser.add_argument('--degrees', type=int, nargs='+', default=list(range(1, 10)))
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--chunk', type=int, default=17, help='element chunk of the second device run')
    parser.add_argument('--no-numba', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(BRANCH))
    os.chdir(BRANCH)
    import numpy as np
    from scripts.adv_diff_rea_cases import get_case
    from hdgfem.core.mesh import rectangle_mesh
    from hdgfem.core.space import DGSpace
    from hdgfem.solvers.adv_diff_rea import assemble_advection_diffusion_reaction as assemble
    rel = lambda a, b: float(np.linalg.norm((np.asarray(a)-np.asarray(b)).ravel())/max(np.linalg.norm(np.asarray(b).ravel()), 1e-300))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'parity.jsonl'
    pairs = [('device', 'cupy_host'), ('device_chunked', 'device'), ('device', 'numpy'), ('cupy_host', 'numpy')]
    if not args.no_numba:
        pairs += [('device', 'numba'), ('numba', 'numpy')]
    worst = {}
    for case in args.cases:
        for p in args.degrees:
            if case == 'variable_tensor':
                kw = variable_tensor_case()
            elif case == 'host_only':
                kw = host_only_case()
            else:
                kw = get_case(case)[0]
            extra = 2*p+4 if case in ('variable_tensor', 'host_only') else None
            space = DGSpace(rectangle_mesh(args.n, args.n), p, basis_type='dub_orth', volume_quad_1d=extra, edge_quad_1d=extra)
            runs = dict(
                numpy=assemble(space=space, backend='numpy', **kw),
                cupy_host=assemble(space=space, backend='cupy', cupy_route='host', **kw),
                device=assemble(space=space, backend='cupy', cupy_route='device', return_local_matrix=True, **kw),
                device_chunked=assemble(space=space, backend='cupy', cupy_route='device', return_local_matrix=True,
                                        element_chunk=args.chunk, **kw))
            if not args.no_numba:
                runs['numba'] = assemble(space=space, backend='numba', **kw)
            for left, right in pairs:
                row = dict(case=case, p=p, n=args.n, triangles=2*args.n**2, left=left, right=right,
                           fallbacks=runs['device'].info.get('host_callable_fallbacks'))
                for name in FIELDS:
                    row[name] = rel(get(runs[left], name), get(runs[right], name))
                for name in ('system.neighbors', 'system.global_to_local', 'system.global_faces'):
                    row[name+'_equal'] = bool(np.array_equal(get(runs[left], name), get(runs[right], name)))
                row['max'] = max(row[name] for name in FIELDS)
                key = (left, right)
                worst[key] = max(worst.get(key, 0.), row['max'])
                with path.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                flags = all(row[k] for k in row if k.endswith('_equal'))
                print(f"{case:20s} p={p} {left:>14s} vs {right:9s} max {row['max']:.2e} "
                      + ' '.join(f"{name.split('.')[-1]}={row[name]:.1e}" for name in FIELDS)
                      + f" ints_equal={flags} fallbacks={row['fallbacks']}", flush=True)
                if not flags:
                    raise SystemExit(f'integer outputs differ: {case} p={p} {left} vs {right}')
    summary = {f'{a}_vs_{b}': v for (a, b), v in worst.items()}
    (args.output/'parity_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
