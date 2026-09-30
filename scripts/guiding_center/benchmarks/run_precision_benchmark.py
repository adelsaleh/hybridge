"""Run the complete guiding-center pipeline at explicit process precision.

FP32 includes GPU assembly, local factors, sparse solves, reconstruction, field
updates and diagnostic reductions. Run FP64 separately for a matched baseline.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """Configure precision before importing numerical code and run the case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--num-steps', type=int, default=500)
    parser.add_argument('--mesh-size', type=float, default=0.025)
    parser.add_argument('--minimum-triangles', type=int, default=10000)
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--rtol', type=float, default=5e-3)
    parser.add_argument('--poisson-rtol', type=float, default=2e-3)
    parser.add_argument('--transport-amgx-config', type=Path, default=None)
    parser.add_argument('--transport-amgx-tolerance', type=float, default=None)
    parser.add_argument('--atol', type=float, default=0.0)
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--time-scheme', choices=('si-euler', 'predictor-corrector'), default='si-euler')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'precision-benchmark')
    args = parser.parse_args()
    os.environ['HDGFEM_PRECISION'] = args.precision
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'.cache'/f'numba-{args.precision}')
    os.environ['CUPY_CACHE_DIR'] = str(ROOT/'.cache'/f'cupy-{args.precision}')
    os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '1'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', '8')
    local_binding = ROOT/'.cache'/'pyamgx-fp32'
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(local_binding))
    import cupy as cp
    import pyamgx
    import numpy as np
    from hdgfem.runtime.precision import (
            KERNEL_AUDIT,
            PIPELINE_AUDIT,
            AMGX_MODE,
            audit_arrays,
        )
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.runner import run_guiding_center_case
    from scripts.guiding_center.runtime.configuration import _with_fp32_transport_solver
    if args.precision == 'float32' and not getattr(pyamgx, 'HDGFEM_PRECISION_AWARE', False):
        raise RuntimeError('Build the local dtype-aware binding with scripts/dev/build_pyamgx_precision.py first')
    args.output.mkdir(parents=True, exist_ok=True)
    preset = 'diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx'
    base_config = preset_by_key(preset)
    if args.transport_amgx_config is None:
        base_config = _with_fp32_transport_solver(base_config)
    else:
        base_config = replace(base_config, transport_amgx_config_path=str(args.transport_amgx_config))
    config = replace(
        base_config, mesh_size=args.mesh_size,
        minimum_triangles=args.minimum_triangles, order=args.order,
        num_steps=args.num_steps, dt=args.dt, time_scheme=args.time_scheme,
        poisson_solver_rtol=args.rtol if args.poisson_rtol is None else args.poisson_rtol, poisson_solver_atol=args.atol,
        transport_solver_rtol=args.rtol, transport_solver_atol=args.atol,
        transport_amgx_tolerance=args.rtol if args.transport_amgx_tolerance is None else args.transport_amgx_tolerance,
        transport_maxiter=300, poisson_maxiter=500,
        plot_every=0, diagnostics_every=1, verbosity=args.verbosity,
        diagnostics_dir=str(args.output), diagnostics_prefix=args.precision,
    )
    cp.cuda.get_current_stream().synchronize()
    start = time.perf_counter()
    try:
        result = run_guiding_center_case(config, preset_key=preset)
    except Exception as exc:
        failed = getattr(exc, 'result', None)
        if failed is not None:
            detail = {key: value for key, value in vars(failed).items()
                      if value is None or isinstance(value, (str, bool, int, float, tuple))}
            (args.output/f'{args.precision}-failure.json').write_text(json.dumps(detail, indent=2, default=str))
        raise
    audit_arrays('final-state', result)
    cp.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter()-start
    np.savez_compressed(args.output/f'{args.precision}-fields.npz', density=result.final_density.coeffs, potential=result.final_potential.coeffs, triangles=result.mesh.triangles, coordinates=result.mesh.node_coords)
    report = dict(precision=args.precision, amgx_mode=AMGX_MODE,
                  pyamgx_path=pyamgx.__file__, wall_seconds=elapsed,
                  triangles=result.mesh.num_tri, order=args.order, steps=args.num_steps,
                  dt=args.dt, poisson_rtol=config.poisson_solver_rtol,
                  transport_rtol=config.transport_solver_rtol,
                  transport_amgx_tolerance=config.transport_amgx_tolerance,
                  transport_amgx_config_path=config.transport_amgx_config_path,
                  atol=args.atol, kernels=KERNEL_AUDIT, pipeline=PIPELINE_AUDIT, final_diagnostics=result.diagnostics[-1])
    (args.output/f'{args.precision}-audit.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({key:value for key,value in report.items() if key not in {'kernels','pipeline','final_diagnostics'}}, indent=2))


if __name__ == '__main__':
    main()
