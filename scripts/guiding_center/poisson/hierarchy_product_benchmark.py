"""Preallocated native-AMGX and generic-cuSPARSE product measurements, no JIT."""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
import statistics
import time

import numpy as np

from hdgfem.backends.cupy import initialize_pyamgx_once
from hdgfem.runtime.optional import require_cupy
from hdgfem.backends.legendre_face_bsr import _CusparseGenericBsrOperator, _CusparseGenericCsrOperator
from hdgfem.linalg.hierarchy_bsr import compare_product, deterministic_vectors


class NativeCsrProduct:
    """AMGX_matrix_vector_multiply via the existing C ABI, square CSR only.

    The C upload API supplies one row count and has no independent column count.
    Vectors belong to AMGX; uploads/downloads happen only outside timed products.
    """
    def __init__(self, host):
        if host.shape[0] != host.shape[1] or host.dtype != np.float64:
            raise ValueError('Native AMGX product requires square FP64 CSR')
        initialize_pyamgx_once()
        self.lib = ct.CDLL(str(Path(os.environ['HDGFEM_AMGX_BUILD_ROOT'])/'libamgxsh.so'))
        p, i = ct.c_void_p, ct.c_int
        signatures = dict(
            AMGX_config_create=([ct.POINTER(p), ct.c_char_p]),
            AMGX_resources_create_simple=([ct.POINTER(p), p]),
            AMGX_matrix_create=([ct.POINTER(p), p, i]),
            AMGX_vector_create=([ct.POINTER(p), p, i]),
            AMGX_matrix_upload_all=([p, i, i, i, i, p, p, p, p]),
            AMGX_vector_upload=([p, i, i, p]), AMGX_vector_set_zero=([p, i, i]),
            AMGX_vector_download=([p, p]), AMGX_matrix_vector_multiply=([p, p, p]))
        for name in ('config', 'resources', 'matrix', 'vector'):
            signatures[f'AMGX_{name}_destroy'] = [p]
        for name, args in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = i
        self.owned = []
        self.size = host.shape[0]
        try:
            self.cfg = self.create('config', b'config_version=2, determinism_flag=1, exception_handling=1')
            self.resources = self.create('resources', self.cfg, suffix='_simple')
            # AMGX_mode_dDDI from the installed amgx_config.h (8193).
            self.matrix = self.create('matrix', self.resources, 8193)
            self.x = self.create('vector', self.resources, 8193)
            self.y = self.create('vector', self.resources, 8193)
            indptr, indices = host.indptr.astype(np.int32), host.indices.astype(np.int32)
            if host.nnz > np.iinfo(np.int32).max:
                raise ValueError('AMGX upload exceeds int32 capacity')
            self.call('AMGX_matrix_upload_all', self.matrix, self.size, host.nnz, 1, 1,
                      indptr.ctypes.data, indices.ctypes.data, host.data.ctypes.data, None)
            for v in (self.x, self.y):
                self.call('AMGX_vector_set_zero', v, self.size, 1)
        except BaseException:
            self.close()
            raise

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f'{name} failed with AMGX status {status}')

    def create(self, kind, *args, suffix=''):
        h = ct.c_void_p()
        self.call(f'AMGX_{kind}_create{suffix}', ct.byref(h), *args)
        self.owned.append((kind, h))
        return h

    def set_input(self, x):
        self.call('AMGX_vector_upload', self.x, self.size, 1, x.ctypes.data)

    def product(self):
        self.call('AMGX_matrix_vector_multiply', self.matrix, self.x, self.y)

    def download(self):
        out = np.empty(self.size)
        self.call('AMGX_vector_download', self.y, out.ctypes.data)
        return out

    def close(self):
        for kind, handle in reversed(self.owned):
            self.call(f'AMGX_{kind}_destroy', handle)
        self.owned.clear()


def alternating_batches(actions, *, warmups=10, batches=5, products=50):
    """CUDA-event elapsed envelope per product; no allocations in timed loops."""
    cp = require_cupy()
    if int(cp.cuda.get_current_stream().ptr) != 0:
        raise RuntimeError('AMGX comparison must use the default CUDA stream')
    for action in actions.values():
        for _ in range(warmups):
            action()
    cp.cuda.runtime.deviceSynchronize()
    events = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(batches*len(actions))]
    samples = {key: [] for key in actions}
    event = iter(events)
    names = list(actions)
    for batch in range(batches):
        for key in (names if batch % 2 == 0 else names[::-1]):
            begin, end = next(event)
            begin.record()
            for _ in range(products):
                actions[key]()
            end.record(); end.synchronize()
            samples[key].append(float(cp.cuda.get_elapsed_time(begin, end))/products)
    return {key: dict(samples_ms=values, median_ms=statistics.median(values),
                     min_ms=min(values), max_ms=max(values), spread_ms=max(values)-min(values))
            for key, values in samples.items()}


def benchmark_operator(csr, bsr):
    cp = require_cupy()
    objects = []
    started = time.perf_counter()
    try:
        gi = _CusparseGenericCsrOperator(cp.asarray(csr.indptr), cp.asarray(csr.indices),
                                        cp.asarray(csr.data), shape=csr.shape)
        objects.append(gi)
        bi = _CusparseGenericBsrOperator(cp.asarray(bsr.indptr), cp.asarray(bsr.indices),
                                        cp.asarray(bsr.data), shape=bsr.shape)
        objects.append(bi)
        x, y = cp.empty(csr.shape[1], dtype=np.float64), cp.empty(csr.shape[0], dtype=np.float64)
        bx, by = cp.empty(bsr.shape[1], dtype=np.float64), cp.empty(bsr.shape[0], dtype=np.float64)
        native = None
        if csr.shape[0] == csr.shape[1]:
            native = NativeCsrProduct(csr); objects.append(native)
        cp.cuda.runtime.deviceSynchronize()
        setup = time.perf_counter()-started
        checks = []
        for vector in deterministic_vectors(csr.shape[1]):
            padded = np.pad(vector, (0, bsr.shape[1]-csr.shape[1]))
            x.set(vector); bx.set(padded)
            gi.matvec(x, out=y); bi.matvec(bx, out=by)
            check = dict(generic_csr=compare_product(csr, vector, cp.asnumpy(y)),
                         generic_bsr=compare_product(csr, vector, cp.asnumpy(by)[:csr.shape[0]]))
            if np.any(cp.asnumpy(by)[csr.shape[0]:] != 0):
                raise AssertionError('Nonzero padded product rows')
            if native is not None:
                native.set_input(vector); native.product()
                check['native_amgx_csr'] = compare_product(csr, vector, native.download())
            checks.append(check)
        actions = dict(generic_csr=lambda: gi.matvec(x, out=y), generic_bsr=lambda: bi.matvec(bx, out=by))
        if native is not None:
            actions['native_amgx_csr'] = native.product
        timings = alternating_batches(actions)
        return dict(timings=timings, product_checks=checks, device_setup_seconds=setup,
            workspace_bytes=dict(generic_csr=gi.workspace_size, generic_bsr=bi.workspace_size),
            native_amgx_csr='measured' if native is not None else 'unsupported: rectangular upload has no column-count argument',
            timing_scope='CUDA-event envelope including host submission gaps; preallocated vectors/workspaces; alpha=1 beta=0; no transfers in timed products')
    finally:
        cp.cuda.runtime.deviceSynchronize()
        for obj in reversed(objects):
            obj.close()
