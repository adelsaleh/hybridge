"""Cooperative tensor ADR assembly and reconstruction kernels (FP64, p=0--6).

One block owns an element. Inverse-diffusion mass factors, the scalar Schur
factor, and small batches of columns share at most 48 KiB. Only coefficient
samples and reference tables cross the host/device boundary.
"""
from __future__ import annotations

import time
import numpy as np

from .raw_cuda_local import RAW_TRACE_ORIENTATION_HELPERS, RAW_COOPERATIVE_SOLVES, checked_lu_source


_MASS_ALGEBRA = r'''
__device__ void chol(double* a, int n, int* status) {
    for (int k=0; k<n; ++k) {
        if (threadIdx.x == 0) {
            double v=a[k*n+k];
            for(int j=0;j<k;++j) v-=a[k*n+j]*a[k*n+j];
            if (!isfinite(v) || v<=0.) *status=k+1;
            else a[k*n+k]=sqrt(v);
        }
        __syncthreads();
        if (*status) return;
        for(int i=k+1+threadIdx.x;i<n;i+=blockDim.x) {
            double v=a[i*n+k];
            for(int j=0;j<k;++j) v-=a[i*n+j]*a[k*n+j];
            a[i*n+k]=v/a[k*n+k];
        }
        __syncthreads();
    }
}

__device__ void triangular(const double* a, const int* piv, double* rhs,
                           int n, int count, bool cholesky) {
    for(int c=threadIdx.x;c<count;c+=blockDim.x) {
        if (!cholesky) for(int k=0;k<n;++k) {
            int p=piv[k]; double v=rhs[k*RAW_BATCH_COLS+c];
            rhs[k*RAW_BATCH_COLS+c]=rhs[p*RAW_BATCH_COLS+c];
            rhs[p*RAW_BATCH_COLS+c]=v;
        }
        for(int i=0;i<n;++i) {
            double v=rhs[i*RAW_BATCH_COLS+c];
            for(int j=0;j<i;++j) v-=a[i*n+j]*rhs[j*RAW_BATCH_COLS+c];
            rhs[i*RAW_BATCH_COLS+c]=cholesky ? v/a[i*n+i] : v;
        }
        for(int i=n-1;i>=0;--i) {
            double v=rhs[i*RAW_BATCH_COLS+c];
            for(int j=i+1;j<n;++j)
                v-=(cholesky ? a[j*n+i] : a[i*n+j])*rhs[j*RAW_BATCH_COLS+c];
            rhs[i*RAW_BATCH_COLS+c]=v/a[i*n+i];
        }
    }
    __syncthreads();
}

__device__ void inverse_mass(double* rhs, double* factor, int* piv,
        int kind, const double* tensor, const double* minv, double jac, int count) {
    if(kind<=2) {
        // Constant tensors use reference M^{-1}; factor storage is scratch.
        for(int idx=threadIdx.x;idx<2*NEL*RAW_BATCH_COLS;idx+=blockDim.x) {
            int i=idx/RAW_BATCH_COLS, c=idx%RAW_BATCH_COLS;
            if(c>=count) continue;
            int row=i%NEL, component=i/NEL;
            double x=0., y=0.;
            for(int j=0;j<NEL;++j) {
                x+=minv[row*NEL+j]*rhs[j*RAW_BATCH_COLS+c];
                y+=minv[row*NEL+j]*rhs[(NEL+j)*RAW_BATCH_COLS+c];
            }
            factor[idx]=(tensor[component*2]*x+tensor[component*2+1]*y)/jac;
        }
        __syncthreads();
        for(int idx=threadIdx.x;idx<2*NEL*RAW_BATCH_COLS;idx+=blockDim.x)
            if(idx%RAW_BATCH_COLS<count) rhs[idx]=factor[idx];
        __syncthreads();
    } else if(kind<=4) {
        triangular(factor,piv,rhs,NEL,count,true);
        triangular(factor+(kind==4 ? NEL*NEL : 0),piv,rhs+NEL*RAW_BATCH_COLS,NEL,count,true);
    } else triangular(factor,piv,rhs,2*NEL,count,kind==5);
}

__device__ __forceinline__ double derivative(int dim, int ij,
        const double* a, const double* d0, const double* d1) {
    return dim==0 ? a[3]*d0[ij]-a[2]*d1[ij] : -a[1]*d0[ij]+a[0]*d1[ij];
}

__device__ double face_moment(int face, int i, int d, const double* phi,
        const double* psi, const double* weights, const double* coefficient) {
    double v=0.;
    for(int q=0;q<NFQ;++q)
        v+=weights[q]*phi[(face*NEL+i)*NFQ+q]*psi[d*NFQ+q]
           *(coefficient ? coefficient[face*NFQ+q] : 1.);
    return v;
}
'''

_ASSEMBLY = r'''
extern "C" __global__ void assemble_adr_tensor(
        long long* rows, long long* cols, double* data, double* rhs,
        const int* indptr, const int* block_pos, const long long* offsets,
        const long long* loc2glob, const bool* orientations, const long long* sides,
        const long long* edge_map, const double* aff, const double* jacs,
        const double* face_jacs, const double* normals,
        const double* minv, const double* basis, const double* gradients,
        const double* weights, const double* phi, const double* psi,
        const double* face_weights, const double* face_mass,
        const double* d0, const double* d1, const double* reaction,
        const double* beta, const double* tau, const double* gamma,
        const double* source, const double* boundary, const long long* kinds,
        const double* constants, const double* inverse, int constant_count,
        int* statuses
#if RAW_RECONSTRUCT
        , const double* trace, double* unknowns
#endif
        ) {
    const int e=blockIdx.x, tid=threadIdx.x, kind=kinds[e];
    const double* a=aff+4*e;
    const double jac=jacs[e];
    const double* tensor=constants+4*(constant_count==1 ? 0 : e);
    const double* t=tau+e*3*NFQ;
    const double* g=gamma+e*3*NFQ;
    extern __shared__ double shared[];
    double* schur=shared;
    double* mn0=schur+NEL*NEL;
    double* mn1=mn0+NEL*NEL;
    double* factor=mn1+NEL*NEL;
    double* flux=factor+FACTOR_STORAGE;
    double* u=flux+2*NEL*RAW_BATCH_COLS;
    int* piv=reinterpret_cast<int*>(u+NEL*RAW_BATCH_COLS);
    int* mass_piv=piv+NEL;
    int* status=statuses+e;

    if(kind>=3) {
        const int n=kind>=5 ? 2*NEL : NEL;
        const int size=kind==4 ? 2*NEL*NEL : n*n;
        for(int idx=tid;idx<size;idx+=blockDim.x) {
            int i,j,component;
            if(kind<=4) {i=(idx%(NEL*NEL))/NEL;j=idx%NEL;component=idx/(NEL*NEL)*3;}
            else {i=(idx/n)%NEL;j=idx%NEL;component=(idx/n)/NEL*2+(idx%n)/NEL;}
            double v=0.;
            for(int q=0;q<NQ;++q)
                v+=jac*weights[q]*basis[i*NQ+q]*basis[j*NQ+q]*inverse[(e*NQ+q)*4+component];
            factor[idx]=v;
        }
        __syncthreads();
        if(kind==6) factor_mass(factor,mass_piv,status);
        else {
            chol(factor,n,status);
            if(kind==4 && !*status) chol(factor+NEL*NEL,NEL,status);
        }
        if(*status) {if(tid==0) *status+=1000; return;}
    }
    // Assemble scalar reaction-advection and all weighted face contractions.
    for(int idx=tid;idx<NEL*NEL;idx+=blockDim.x) {
        int i=idx/NEL,j=idx%NEL;
        double v=0., nx=0., ny=0.;
        for(int q=0;q<NQ;++q) {
            double gx=(a[3]*gradients[i*NQ+q]-a[2]*gradients[(NEL+i)*NQ+q])/jac;
            double gy=(-a[1]*gradients[i*NQ+q]+a[0]*gradients[(NEL+i)*NQ+q])/jac;
            v+=jac*weights[q]*basis[j*NQ+q]*(reaction[e*NQ+q]*basis[i*NQ+q]
               -beta[(e*NQ+q)*2]*gx-beta[(e*NQ+q)*2+1]*gy);
        }
        for(int f=0;f<3;++f) {
            double scale=face_jacs[e*3+f];
            nx+=scale*normals[(e*3+f)*2]*face_mass[f*NEL*NEL+idx];
            ny+=scale*normals[(e*3+f)*2+1]*face_mass[f*NEL*NEL+idx];
            for(int q=0;q<NFQ;++q)
                v+=scale*face_weights[q]*t[f*NFQ+q]*phi[(f*NEL+i)*NFQ+q]*phi[(f*NEL+j)*NFQ+q];
        }
        schur[idx]=v;
        mn0[idx]=nx-derivative(0,idx,a,d0,d1);
        mn1[idx]=ny-derivative(1,idx,a,d0,d1);
    }
    __syncthreads();
    // S = A + (N-D) G^{-1} D, built in bounded batches of scalar columns.
    for(int start=0;start<NEL;start+=RAW_BATCH_COLS) {
        int count=min(RAW_BATCH_COLS,NEL-start);
        for(int idx=tid;idx<2*NEL*RAW_BATCH_COLS;idx+=blockDim.x) {
            int i=idx/RAW_BATCH_COLS,c=idx%RAW_BATCH_COLS;
            if(c<count) flux[idx]=derivative(i/NEL,(i%NEL)*NEL+start+c,a,d0,d1);
        }
        __syncthreads();
        inverse_mass(flux,factor,mass_piv,kind,tensor,minv,jac,count);
        for(int idx=tid;idx<NEL*RAW_BATCH_COLS;idx+=blockDim.x) {
            int i=idx/RAW_BATCH_COLS,c=idx%RAW_BATCH_COLS;
            if(c<count) {
                double v=0.;
                for(int j=0;j<NEL;++j)
                    v+=mn0[i*NEL+j]*flux[j*RAW_BATCH_COLS+c]+mn1[i*NEL+j]*flux[(NEL+j)*RAW_BATCH_COLS+c];
                schur[i*NEL+start+c]+=v;
            }
        }
        __syncthreads();
    }
    factor_schur(schur,piv,status);
    if(*status) return;
#if RAW_RECONSTRUCT
    const int output_columns=1;
#else
    const int output_columns=NCOLS;
#endif
    for(int start=0;start<output_columns;start+=RAW_BATCH_COLS) {
        int count=min(RAW_BATCH_COLS,output_columns-start);
        for(int idx=tid;idx<NEL*RAW_BATCH_COLS;idx+=blockDim.x) {
            int i=idx/RAW_BATCH_COLS,c=idx%RAW_BATCH_COLS;
            if(c>=count) continue;
            int col=start+c;
            double x=0.,y=0.;
#if RAW_RECONSTRUCT
            double v=source[e*NEL+i];
            const int first=0, last=3*NTR;
#else
            double v=col==3*NTR ? source[e*NEL+i] : 0.;
            const int first=col, last=min(col+1,3*NTR);
#endif
            for(int tc=first;tc<last;++tc) {
                int f=tc/NTR,d=raw_trace_local_dof(orientations[e*3+f],tc%NTR,NTR);
                double scale=face_jacs[e*3+f]*raw_trace_orientation_sign(orientations[e*3+f],tc%NTR);
#if RAW_RECONSTRUCT
                scale*=trace[loc2glob[e*3+f]*NTR+tc%NTR];
#endif
                double coupling=scale*face_moment(f,i,d,phi,psi,face_weights,nullptr);
                x+=normals[(e*3+f)*2]*coupling;y+=normals[(e*3+f)*2+1]*coupling;
                v+=scale*face_moment(f,i,d,phi,psi,face_weights,g);
            }
            u[idx]=v;flux[idx]=x;flux[NEL*RAW_BATCH_COLS+idx]=y;
        }
        __syncthreads();
        inverse_mass(flux,factor,mass_piv,kind,tensor,minv,jac,count);
        for(int idx=tid;idx<NEL*RAW_BATCH_COLS;idx+=blockDim.x) {
            int i=idx/RAW_BATCH_COLS,c=idx%RAW_BATCH_COLS;
            if(c>=count) continue;
            double v=0.;
            for(int j=0;j<NEL;++j)
                v+=mn0[i*NEL+j]*flux[j*RAW_BATCH_COLS+c]+mn1[i*NEL+j]*flux[(NEL+j)*RAW_BATCH_COLS+c];
            u[idx]+=v;
        }
        __syncthreads();
        solve_diffusion_column_batch_coop_raw(schur,piv,u,count);
        for(int idx=tid;idx<2*NEL*RAW_BATCH_COLS;idx+=blockDim.x) {
            int i=(idx/RAW_BATCH_COLS)%NEL,dim=idx/(NEL*RAW_BATCH_COLS),c=idx%RAW_BATCH_COLS;
            if(c>=count) continue;
            double v=0.;
            for(int j=0;j<NEL;++j) v+=derivative(dim,i*NEL+j,a,d0,d1)*u[j*RAW_BATCH_COLS+c];
            int col=start+c;
#if RAW_RECONSTRUCT
            const int first=0,last=3*NTR;
#else
            const int first=col,last=min(col+1,3*NTR);
#endif
            for(int tc=first;tc<last;++tc) {
                int f=tc/NTR,d=raw_trace_local_dof(orientations[e*3+f],tc%NTR,NTR);
                double scale=face_jacs[e*3+f]*raw_trace_orientation_sign(orientations[e*3+f],tc%NTR);
#if RAW_RECONSTRUCT
                scale*=trace[loc2glob[e*3+f]*NTR+tc%NTR];
#endif
                v-=scale*normals[(e*3+f)*2+dim]*face_moment(f,i,d,phi,psi,face_weights,nullptr);
            }
            flux[idx]=v;
        }
        __syncthreads();
        inverse_mass(flux,factor,mass_piv,kind,tensor,minv,jac,count);
#if RAW_RECONSTRUCT
        for(int i=tid;i<NEL;i+=blockDim.x) {
            const double uh=u[i*RAW_BATCH_COLS];
            const double qx=flux[i*RAW_BATCH_COLS], qy=flux[(NEL+i)*RAW_BATCH_COLS];
            if(!isfinite(uh) || !isfinite(qx) || !isfinite(qy)) atomicExch(status,2000);
            unknowns[e*3*NEL+i]=uh;
            unknowns[e*3*NEL+NEL+i]=qx;
            unknowns[e*3*NEL+2*NEL+i]=qy;
        }
#else
        for(int task=tid;task<3*NTR*count;task+=blockDim.x) {
            int row=task/count,c=task%count,rf=row/NTR,rd=row%NTR,col=start+c;
            long long sid=sides[e*3+rf],re=edge_map[loc2glob[e*3+rf]];
            if(sid<0 || re<0) continue;
            int ld=raw_trace_local_dof(orientations[e*3+rf],rd,NTR);
            double rs=raw_trace_orientation_sign(orientations[e*3+rf],rd);
            double v=0.;
            for(int q=0;q<NFQ;++q) {
                double uh=0.,qx=0.,qy=0.;
                for(int i=0;i<NEL;++i) {
                    double p=phi[(rf*NEL+i)*NFQ+q];
                    uh+=p*u[i*RAW_BATCH_COLS+c];qx+=p*flux[i*RAW_BATCH_COLS+c];qy+=p*flux[(NEL+i)*RAW_BATCH_COLS+c];
                }
                v+=face_weights[q]*psi[ld*NFQ+q]*(t[rf*NFQ+q]*uh
                    +normals[(e*3+rf)*2]*qx+normals[(e*3+rf)*2+1]*qy);
            }
            v*=rs*face_jacs[e*3+rf];
            if(!isfinite(v)) {atomicExch(status,2000);continue;}
            if(col==3*NTR) {atomicAdd(rhs+re*NTR+rd,v);continue;}
            int cf=col/NTR,cd=col%NTR;
            long long ce=edge_map[loc2glob[e*3+cf]];
            if(ce<0) {atomicAdd(rhs+re*NTR+rd,v*boundary[loc2glob[e*3+cf]*NTR+cd]);continue;}
            double entry=-v;
            if(rf==cf) {
                int lcd=raw_trace_local_dof(orientations[e*3+cf],cd,NTR);
                double cs=raw_trace_orientation_sign(orientations[e*3+cf],cd),m=0.;
                for(int q=0;q<NFQ;++q) m+=face_weights[q]*g[rf*NFQ+q]*psi[ld*NFQ+q]*psi[lcd*NFQ+q];
                entry+=rs*cs*face_jacs[e*3+rf]*m;
            }
#if RAW_MATRIX_CSR
            const int pos=block_pos[sid*3+cf];
#if RAW_MATRIX_BSR
            const long long out=((long long)indptr[re]+pos)*NTR*NTR+rd*NTR+cd;
#else
            const long long out=(long long)indptr[re*NTR+rd]+pos*NTR+cd;
#endif
            atomicAdd(data+out,entry);
#else
            int pos=0;
            for(int f=0;f<cf;++f) if(edge_map[loc2glob[e*3+f]]>=0) ++pos;
            const long long out=offsets[sid]+(pos*NTR+rd)*NTR+cd;
            rows[out]=re*NTR+rd;cols[out]=ce*NTR+cd;data[out]=entry;
#endif
        }
#endif
        __syncthreads();
    }
}
'''


def tensor_workspace(nel, kinds):
    """Return batch width, factor/scratch slots, and bounded shared bytes."""
    maximum = int(np.max(kinds, initial=0))
    blocks = 0 if maximum <= 2 else (1 if maximum == 3 else (2 if maximum == 4 else 4))
    for batch in range(8, 0, -1):
        storage = max(blocks * nel * nel, 2 * nel * batch)
        size = 8 * (3 * nel * nel + storage + 3 * nel * batch) + 4 * 3 * nel
        if size <= 48 * 1024:
            return batch, storage, size
    raise ValueError("tensor ADR local workspace exceeds the 48 KiB resource limit")


def kernel_source(nel, ntr, nq, nfq, orientation, matrix_format, batch, storage, *, reconstruct=False):
    """Build a shape-specialized kernel using diffusion's shared local helpers."""
    defines = dict(NEL=nel, NTR=ntr, NQ=nq, NFQ=nfq, NCOLS=3*ntr+1,
                   TRACE_ORIENTATION_MODE=orientation, RAW_BATCH_COLS=batch,
                   FACTOR_STORAGE=storage, RAW_MATRIX_CSR=int(matrix_format!='coo'),
                   RAW_MATRIX_BSR=int(matrix_format=='bsr'), RAW_RECONSTRUCT=int(reconstruct))
    prefix = ''.join(f'#define {key} {value}\n' for key, value in defines.items())
    source = (prefix + RAW_TRACE_ORIENTATION_HELPERS + RAW_COOPERATIVE_SOLVES
            + checked_lu_source('factor_schur', 'NEL')
            + checked_lu_source('factor_mass', '(2 * NEL)') + _MASS_ALGEBRA + _ASSEMBLY)
    return source.replace('void assemble_adr_tensor(', 'void reconstruct_adr_tensor(') if reconstruct else source


def assemble_tensor_operator(prepared, boundary_condition, space, *, diffusion,
                             trace_space, matrix_format, block_size):
    """Prepare reference/device data and emit a reduced operator without a solve."""
    from ..assembly import hdg
    from ..assembly.diffusion_coefficients import prepare_diffusion
    from .cupy import require_cupy, as_cupy_space
    from .advection_cuda import CudaAdvectionAssembly, as_cupy_trace_space
    from .advection_raw_cuda import build_reduced_csr_pattern_raw
    from .diffusion_raw_cuda import (_edge_to_solve_edge, _interior_side_index,
                                    _side_flux_offsets, validate_raw_cuda_supported)
    from .numba import _trace_orientation_mode
    from .advection_diffusion_reaction_raw_cuda import RawADRTraceOperator

    start = time.perf_counter()
    cp = require_cupy()
    stream = cp.cuda.get_current_stream()
    timings = {'raw.coefficients.adr': prepared.preparation_seconds}
    prep_start = time.perf_counter()
    tensor = prepare_diffusion(diffusion, space)
    timings['raw.coefficients.diffusion'] = time.perf_counter() - prep_start
    nel, ntr = space.el_dof, trace_space.edg_dof
    q, mesh = space.quad_data, space.mesh
    expected = (mesh.num_tri, 3, trace_space.weights.size)
    if prepared.tau_total.shape != expected or prepared.gamma.shape != expected:
        raise ValueError('ADR preparation face quadrature does not match trace_space')
    if prepared.face_quadrature is not None and not np.array_equal(prepared.face_quadrature, trace_space.quads):
        raise ValueError('ADR preparation face quadrature does not match trace_space')
    for name in ('reaction_values', 'beta_values', 'source_rhs', 'tau_total', 'gamma'):
        if not np.all(np.isfinite(getattr(prepared, name))):
            raise ValueError(f'ADR {name} samples must be finite')
    if not np.all(np.isfinite(prepared.tau_diffusion)) or np.any(prepared.tau_diffusion <= 0):
        raise ValueError('diffusion stabilization must be finite and strictly positive')
    batch, storage, shared_bytes = tensor_workspace(nel, tensor.kinds)
    upload_start = time.perf_counter()
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(trace_space)
    validate_raw_cuda_supported(cspace, trace_ref)
    boundary_host = hdg.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_space)
    if not np.all(np.isfinite(boundary_host)):
        raise ValueError('ADR boundary trace must be finite')
    boundary = cp.asarray(boundary_host, dtype=cp.float64)
    face_mass = np.einsum('q,fiq,fjq->fij', trace_space.weights,
                          trace_space.bas_of_bd_quads, trace_space.bas_of_bd_quads, optimize=True)
    arrays = (mesh.aff_mats, mesh.aff_jacs, mesh.jacs_el_fc, mesh.normals,
              q.MKrf_inv, q.bas_of_quads, q.dbas_of_quads, q.Krf_w,
              trace_space.bas_of_bd_quads, trace_space.bas1d_of_ref_edg_qds,
              trace_space.weights, face_mass, prepared.d0_reference, prepared.d1_reference,
              prepared.reaction_values, prepared.beta_values, prepared.tau_total,
              prepared.gamma, prepared.source_rhs)
    inputs = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=cp.float64)) for value in arrays)
    kinds = cp.asarray(tensor.kinds, dtype=cp.int64)
    constants, inverse = cp.asarray(tensor.constants), cp.asarray(tensor.inverse_values)
    stream.synchronize()
    timings['raw.upload'] = time.perf_counter() - upload_start
    graph_start = time.perf_counter()
    dummy64, dummy32 = cp.empty(1, dtype=cp.int64), cp.empty(1, dtype=cp.int32)
    pattern = None
    if matrix_format == 'coo':
        edge_host = _edge_to_solve_edge(mesh)
        offsets_host = _side_flux_offsets(mesh, edge_host, ntr)
        edge_map = cp.asarray(edge_host)
        sides = cp.asarray(_interior_side_index(mesh))
        offsets = cp.asarray(offsets_host)
        rows = cp.empty(int(offsets_host[-1]), dtype=cp.int64)
        cols = cp.empty_like(rows)
        data = cp.empty(rows.size, dtype=cp.float64)
        indptr = indices = None
        row_arg, col_arg, ptr_arg, pos_arg = rows, cols, dummy32, dummy32
    else:
        pattern = build_reduced_csr_pattern_raw(cspace, timings, matrix_format=matrix_format)
        edge_map, sides = pattern.edge_to_solve_edge, pattern.interior_side_index
        indptr, indices = pattern.indptr, pattern.indices
        data = cp.zeros((pattern.num_blocks, ntr, ntr) if matrix_format == 'bsr'
                        else indices.size, dtype=cp.float64)
        rows = cols = None
        row_arg = col_arg = offsets = dummy64
        ptr_arg, pos_arg = indptr, pattern.side_csr_block_pos
    rhs = cp.zeros(mesh.int_edges_inds.size*ntr, dtype=cp.float64)
    statuses = cp.zeros(mesh.num_tri, dtype=cp.int32)
    stream.synchronize()
    timings['raw.graph'] = time.perf_counter() - graph_start
    jit_start = time.perf_counter()
    code = kernel_source(nel, ntr, q.Krf_w.size, trace_space.weights.size,
                         _trace_orientation_mode(trace_space), matrix_format, batch, storage)
    module = cp.RawModule(code=code, options=('--std=c++11',))
    kernel = module.get_function('assemble_adr_tensor')
    timings['raw.kernel.jit'] = time.perf_counter() - jit_start
    args = (row_arg, col_arg, data, rhs, ptr_arg, pos_arg, offsets,
            cspace.mesh.loc2glob_edge, cspace.mesh.orientations, sides, edge_map,
            *inputs, boundary, kinds, constants, inverse, np.int32(tensor.constants.shape[0]), statuses)
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    launch_start = time.perf_counter()
    begin.record()
    kernel((mesh.num_tri,), (block_size,), args, shared_mem=shared_bytes)
    end.record()
    end.synchronize()
    timings['raw.kernel.wall'] = time.perf_counter() - launch_start
    timings['raw.kernel.device'] = cp.cuda.get_elapsed_time(begin, end)/1000.
    check_start = time.perf_counter()
    if bool(cp.any(statuses)):
        failures = cp.asnumpy(statuses)
        element = int(np.flatnonzero(failures)[0])
        code = int(failures[element])
        stage = 'inverse-diffusion mass' if 1000 < code < 2000 else ('condensation' if code == 2000 else 'scalar Schur')
        raise np.linalg.LinAlgError(f'ADR {stage} factorization/finite-value failure in element {element} (status {code})')
    timings['raw.validation'] = time.perf_counter() - check_start
    timings.update({'raw.block_size': float(block_size), 'raw.shared_bytes': float(shared_bytes),
                    'raw.batch_columns': float(batch), 'raw.conversion': 0., 'raw.coo_to_csr.wall': 0.})
    timings['raw.wall_total'] = time.perf_counter() - start
    timings['raw.total_with_preparation'] = timings['raw.wall_total'] + prepared.preparation_seconds
    assembly = CudaAdvectionAssembly(
        rows, cols, data, rhs, None, None, inputs[-1], boundary[cspace.mesh.bnd_edges_inds],
        None, cspace, trace_ref, indptr=indptr, indices=indices,
        matrix_format=matrix_format, timings=timings)
    return RawADRTraceOperator(assembly, module, inputs, boundary,
                               csr_pattern=pattern, diffusion_structure=tensor.counts,
                               diffusion_kinds=tensor.kinds.copy(),
                               reconstruction_data=dict(args=args[:-1], nel=nel, ntr=ntr,
                                   nq=q.Krf_w.size, nfq=trace_space.weights.size,
                                   orientation=_trace_orientation_mode(trace_space),
                                   batch=batch, storage=storage, shared_bytes=shared_bytes))


def reconstruct_tensor_operator(operator, trace, *, block_size="auto"):
    """Reconstruct device [u, qx, qy] with the assembly's exact coefficient data.

    No dense local operators or coefficient samples are uploaded again. Local
    factors are rebuilt using the same kernel algebra; this is not a factor cache.
    """
    from .cupy import require_cupy
    from .raw_cuda import resolve_raw_cuda_block_size
    cp=require_cupy()
    local=operator.reconstruction_data
    if local is None:
        raise ValueError("tensor reconstruction requires the cooperative/lightweight assembly operator")
    space=operator.assembly.cspace.host
    trace=cp.ascontiguousarray(cp.asarray(trace,dtype=cp.float64)).reshape(-1)
    if trace.size!=space.mesh.num_edg*local['ntr']:
        raise ValueError("reconstruction requires the full trace, including boundary coefficients")
    if not bool(cp.all(cp.isfinite(trace))):
        raise ValueError("reconstruction trace must be finite")
    block_size=resolve_raw_cuda_block_size(block_size,equation="diffusion-reaction",order=space.order)
    shape={key:local[key] for key in ('nel','ntr','nq','nfq','orientation','batch','storage')}
    start=time.perf_counter()
    module=cp.RawModule(code=kernel_source(**shape,matrix_format='csr',reconstruct=True),options=('--std=c++11',))
    kernel=module.get_function('reconstruct_adr_tensor')
    timings={'raw.reconstruction.jit':time.perf_counter()-start}
    unknowns=cp.empty((space.mesh.num_tri,3*space.el_dof),dtype=cp.float64)
    statuses=cp.zeros(space.mesh.num_tri,dtype=cp.int32)
    begin,end=cp.cuda.Event(),cp.cuda.Event()
    start=time.perf_counter()
    begin.record()
    kernel((space.mesh.num_tri,),(block_size,),(*local['args'],statuses,trace,unknowns),shared_mem=local['shared_bytes'])
    end.record()
    end.synchronize()
    timings['raw.reconstruction.wall']=time.perf_counter()-start
    timings['raw.reconstruction.device']=cp.cuda.get_elapsed_time(begin,end)/1000.
    if bool(cp.any(statuses)):
        failures=cp.asnumpy(statuses)
        element=int(np.flatnonzero(failures)[0])
        raise np.linalg.LinAlgError(f'ADR reconstruction factorization/finite-value failure in element {element} (status {failures[element]})')
    return unknowns,timings
