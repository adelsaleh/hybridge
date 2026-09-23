"""Configure guiding-center precision before loading NumPy, Numba or hdgfem."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def configure_precision_cli() -> None:
    """Read only --precision early, leaving all other CLI options untouched."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--precision', choices=('float32', 'float64'), default=os.environ.get('HDGFEM_PRECISION', 'float64'))
    args, _ = parser.parse_known_args()
    os.environ['HDGFEM_PRECISION'] = args.precision
    if args.precision == 'float32':
        root = Path(__file__).resolve().parents[3]
        os.environ['NUMBA_CACHE_DIR'] = str(root/'.cache'/'numba-float32')
        os.environ['CUPY_CACHE_DIR'] = str(root/'.cache'/'cupy-float32')
        os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '1'
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('NUMBA_NUM_THREADS', '8')
        binding = root/'.cache'/'pyamgx-fp32'
        inspection_only = any(flag in sys.argv for flag in ('--help', '-h', '--dry-run', '--print-preset', '--list-presets'))
        if not tuple(binding.glob('pyamgx*.so')) and not inspection_only:
            raise RuntimeError('Build the local mode-aware binding with scripts/dev/build_pyamgx_precision.py before using --precision float32')
        sys.path.insert(0, str(binding))
