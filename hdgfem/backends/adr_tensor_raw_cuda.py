"""Cooperative tensor ADR assembly and reconstruction kernels (p=0--6, FP64 or FP32).

One block owns an element. Inverse-diffusion mass factors, the scalar Schur
factor, and small batches of columns share at most 48 KiB. Only coefficient
samples and reference tables cross the host/device boundary.

One CUDA source serves both kernels and every order; ``kernel_source``
specializes it with compile-time sizes (NEL, NTR, NQ, NFQ) and switches:

* ``RAW_RECONSTRUCT`` builds ``reconstruct_adr_tensor`` (one right-hand side
  from the solved trace) instead of ``assemble_adr_tensor`` (all 3*NTR trace
  columns plus the source column, scattered into COO/CSR/BSR).
* ``RAW_WRITE_FACTORS`` (assembly) stores each element's pivoted Schur LU in a
  device cache; ``RAW_USE_FACTORS`` (reconstruction) reads it back and skips
  volume quadrature, Schur accumulation and the LU. The cache costs
  ``NEL*NEL`` reals + ``NEL`` ints per element (about 6.4 KB at p=6 in FP64).
* ``RAW_WRITE_MASS`` / ``RAW_USE_MASS`` do the same for the factored
  variable-tensor mass (``mass_factor_size`` reals per element, plus 2*NEL
  pivots for general tensors), so cached reconstruction reads no kappa samples.

The source is written in ``double``; ``precision.real_raw_module`` rewrites it
to ``float`` under ``HDGFEM_PRECISION=float32``, and every array follows
``REAL_DTYPE``.

Per element the assembly kernel (i) assembles and factors the inverse-diffusion
mass for variable tensors (kinds 3--6): point coefficients jac*w*kappa^{-1}, one
phi_i phi_j product per symmetric (i, j) pair, a Cholesky or warp-pivoted LU
whose diagonal is stored as reciprocals, and warp-per-column triangular solves
(two rows per lane when the factor is 2*NEL), (ii) folds quadrature weights, the Jacobian and
the affine gradient map into per-point coefficients so the NEL^2 contractions
need no FP64 divides, (iii) accumulates S = A + (N-D) G^{-1} D in column
batches and factors it, and (iv) solves the local columns and projects the
weighted face traces tau*u + n.q onto the trace basis, evaluating those traces
once per face point rather than once per trace row.
"""
from __future__ import annotations

import time
import numpy as np

from hdgfem.precision import REAL_DTYPE, REAL_ITEMSIZE, real_raw_module

from .capabilities import UnsupportedBackendConfigurationError
from .raw_cuda_local import (RAW_TRACE_ORIENTATION_HELPERS, RAW_COOPERATIVE_SOLVES,
                             RAW_WARP_COLUMN_SOLVES, checked_warp_lu_source)


_MASS_ALGEBRA = r'''
// Packed lower-triangular storage (rows of increasing length): entry (i, j),
// j <= i, of a symmetric factor lives at packed(i, j). Cholesky factors of
// symmetric kappa (kinds 3--5) use it, halving their shared memory.
__device__ __forceinline__ int packed(int i, int j) { return i*(i+1)/2+j; }

// Cholesky of a packed symmetric n x n shared matrix. Warp 0 reduces each
// pivot's dot product; the block then updates the column. The diagonal is
// stored as its reciprocal 1/L_kk, so columns and the triangular sweeps
// multiply instead of divide (few FP64 units make every division costly).
__device__ void chol(double* a, int n, int* status) {
    const int tid=threadIdx.x, lane=tid&31;
    const int lanes=blockDim.x<32 ? blockDim.x : 32;
    for (int k=0; k<n; ++k) {
        const double* row_k=a+packed(k,0);
        if (tid < 32) {
            const unsigned mask=__activemask();
            double s=0.;
            for(int j=lane;j<k;j+=lanes) s+=row_k[j]*row_k[j];
            for(int offset=lanes/2;offset>0;offset>>=1) s+=__shfl_xor_sync(mask,s,offset);
            if (lane == 0) {
                const double v=row_k[k]-s;
                if (!isfinite(v) || v<=0.) *status=k+1;
                else a[packed(k,k)]=1./sqrt(v);
            }
        }
        __syncthreads();
        if (*status) return;
        for(int i=k+1+tid;i<n;i+=blockDim.x) {
            double* row_i=a+packed(i,0);
            double v=row_i[k];
            for(int j=0;j<k;++j) v-=row_i[j]*row_k[j];
            row_i[k]=v*row_k[k];
        }
        __syncthreads();
    }
}

// Replace a factor's diagonal by its reciprocals (LU factors; chol does it inline).
__device__ void invert_diagonal(double* a, int n) {
    for(int i=threadIdx.x;i<n;i+=blockDim.x) a[i*n+i]=1./a[i*n+i];
    __syncthreads();
}

// Solve the batch columns with a mass factor whose diagonal holds reciprocals:
// packed Cholesky (L L^T) or full pivoted LU (unit L, U). One warp per
// column; lane r holds rows r and r+32, so both sweeps are shuffles with one
// FP64 FMA per row block per step. Forward sums keep the serial order.
template<int N, bool CHOLESKY>
__device__ __forceinline__ void triangular_warp(const double* __restrict__ a,
        const int* __restrict__ piv, double* __restrict__ rhs, int count) {
    if(blockDim.x<32) {
        for(int c=threadIdx.x;c<count;c+=blockDim.x) {
            double x[N];
            for(int i=0;i<N;++i) x[i]=rhs[i*RAW_BATCH_COLS+c];
            if(!CHOLESKY) {
                for(int k=0;k<N;++k) {
                    const int p=piv[k];
                    const double tmp=x[k]; x[k]=x[p]; x[p]=tmp;
                }
            }
            for(int i=0;i<N;++i) {
                for(int j=0;j<i;++j) x[i]-=a[CHOLESKY ? packed(i,j) : i*N+j]*x[j];
                if(CHOLESKY) x[i]*=a[packed(i,i)];
            }
            for(int i=N-1;i>=0;--i) {
                for(int j=N-1;j>i;--j) x[i]-=a[CHOLESKY ? packed(j,i) : i*N+j]*x[j];
                x[i]*=a[CHOLESKY ? packed(i,i) : i*N+i];
            }
            for(int i=0;i<N;++i) rhs[i*RAW_BATCH_COLS+c]=x[i];
        }
        __syncthreads();
        return;
    }
    constexpr int R=(N+31)/32;
    static_assert(R<=2, "warp mass solves hold at most two rows per lane (2*NEL <= 64)");
    const int lane=threadIdx.x&31;
    for(int c=threadIdx.x>>5;c<count;c+=blockDim.x>>5) {
        double x[R];
#pragma unroll
        for(int r=0;r<R;++r) {
            const int row=lane+32*r;
            x[r]=row<N ? rhs[row*RAW_BATCH_COLS+c] : 0.;
        }
        if(!CHOLESKY) {
            for(int k=0;k<N;++k) {
                const int p=piv[k];
                if(p==k) continue;
                const double xk=__shfl_sync(0xffffffffu, k<32 ? x[0] : x[R-1], k&31);
                const double xp=__shfl_sync(0xffffffffu, p<32 ? x[0] : x[R-1], p&31);
#pragma unroll
                for(int r=0;r<R;++r) {
                    const int row=lane+32*r;
                    if(row==k) x[r]=xp; else if(row==p) x[r]=xk;
                }
            }
        }
        for(int j=0;j<N;++j) {
            if(CHOLESKY) {
#pragma unroll
                for(int r=0;r<R;++r) if(lane+32*r==j) x[r]*=a[packed(j,j)];
            }
            const double xj=__shfl_sync(0xffffffffu, j<32 ? x[0] : x[R-1], j&31);
#pragma unroll
            for(int r=0;r<R;++r) {
                const int row=lane+32*r;
                const double l=(row>j && row<N) ? a[CHOLESKY ? packed(row,j) : row*N+j] : 0.;
                x[r]-=l*xj;
            }
        }
        for(int j=N-1;j>=0;--j) {
#pragma unroll
            for(int r=0;r<R;++r) if(lane+32*r==j) x[r]*=a[CHOLESKY ? packed(j,j) : j*N+j];
            const double xj=__shfl_sync(0xffffffffu, j<32 ? x[0] : x[R-1], j&31);
#pragma unroll
            for(int r=0;r<R;++r) {
                const int row=lane+32*r;
                const double u=row<j ? a[CHOLESKY ? packed(j,row) : row*N+j] : 0.;
                x[r]-=u*xj;
            }
        }
#pragma unroll
        for(int r=0;r<R;++r) {
            const int row=lane+32*r;
            if(row<N) rhs[row*RAW_BATCH_COLS+c]=x[r];
        }
    }
    __syncthreads();
}

__device__ void inverse_mass(double* rhs, double* factor, int* piv,
        int kind, const double* tensor, const double* minv, double inv_jac, int count) {
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
            factor[idx]=(tensor[component*2]*x+tensor[component*2+1]*y)*inv_jac;
        }
        __syncthreads();
        for(int idx=threadIdx.x;idx<2*NEL*RAW_BATCH_COLS;idx+=blockDim.x)
            if(idx%RAW_BATCH_COLS<count) rhs[idx]=factor[idx];
        __syncthreads();
    } else if(kind<=4) {
        triangular_warp<NEL,true>(factor,piv,rhs,count);
        triangular_warp<NEL,true>(factor+(kind==4 ? packed(NEL,0) : 0),piv,rhs+NEL*RAW_BATCH_COLS,count);
    } else if(kind==5) triangular_warp<2*NEL,true>(factor,piv,rhs,count);
    else triangular_warp<2*NEL,false>(factor,piv,rhs,count);
}

__device__ __forceinline__ double derivative(int dim, int ij,
        const double* a, const double* d0, const double* d1) {
    return dim==0 ? a[3]*d0[ij]-a[2]*d1[ij] : -a[1]*d0[ij]+a[0]*d1[ij];
}

__device__ double face_moment(int face, int i, int d, const double* phi_t,
        const double* psi, const double* weights, const double* coefficient) {
    double v=0.;
    for(int q=0;q<NFQ;++q)
        v+=weights[q]*phi_t[(face*NFQ+q)*NEL+i]*psi[d*NFQ+q]
           *(coefficient ? coefficient[face*NFQ+q] : 1.);
    return v;
}

// Coupling blocks N_d - D_d: the side normal mass minus the element derivative.
__device__ __forceinline__ void normal_minus_derivative(int idx, int e, const double* a,
        const double* face_jacs, const double* normals, const double* face_mass,
        const double* d0, const double* d1, double* mn0, double* mn1) {
    double nx=0., ny=0.;
    for(int f=0;f<3;++f) {
        const double scale=face_jacs[e*3+f]*face_mass[f*NEL*NEL+idx];
        nx+=normals[(e*3+f)*2]*scale;
        ny+=normals[(e*3+f)*2+1]*scale;
    }
    mn0[idx]=nx-derivative(0,idx,a,d0,d1);
    mn1[idx]=ny-derivative(1,idx,a,d0,d1);
}

// Moment of the weighted trace samples lambda (see RAW_RECONSTRUCT) on one face.
__device__ double trace_moment(int face, int i, const double* phi_t,
        const double* lambda, const double* coefficient) {
    double v=0.;
    for(int q=0;q<NFQ;++q)
        v+=phi_t[(face*NFQ+q)*NEL+i]*lambda[face*NFQ+q]
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
        const double* minv, const double* basis, const double* basis_t, const double* gradients,
        const double* gradients_t,
        const double* weights, const double* phi, const double* phi_t, const double* psi,
        const double* face_weights, const double* face_mass,
        const double* d0, const double* d1, const double* reaction,
        const double* beta, const double* tau, const double* gamma,
        const double* source, const double* boundary, const long long* kinds,
        const double* constants, const double* inverse, int constant_count,
        double* schur_cache, int* pivot_cache, double* mass_cache, int* mass_pivot_cache,
        int* statuses
#if RAW_RECONSTRUCT
        , const double* trace, double* unknowns
#endif
        ) {
    const int e=blockIdx.x, tid=threadIdx.x, kind=kinds[e];
    const double* a=aff+4*e;
    const double jac=jacs[e], inv_jac=1./jac;
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
    // Per-point volume/face coefficients, later the weighted face traces.
    double* coef=u+NEL*RAW_BATCH_COLS;
    double* inverse_diagonal=coef+COEF_STORAGE;  // Schur pivot reciprocals
    int* piv=reinterpret_cast<int*>(inverse_diagonal+NEL);
    int* mass_piv=piv+NEL;
    int* schur_rows=mass_piv+2*NEL;  // Schur row permutation for the warp column solves
    int* status=statuses+e;

    if(kind>=3) {
        // Variable inverse-diffusion mass G_ab = int kappa^{-1}_ab phi_i phi_j.
        // Kinds 3--5 (symmetric kappa) store packed Cholesky factors: kind 3 one
        // NEL block, kind 4 two (00 and 11), kind 5 the 2x2 block matrix (lower
        // blocks 00, 10, 11). Kind 6 stores the full 2x2 block matrix for LU.
        const int n=kind>=5 ? 2*NEL : NEL;
        const int size=kind==3 ? packed(NEL,0) : (kind==4 ? 2*packed(NEL,0) : (kind==5 ? packed(n,0) : n*n));
        double* const cached_mass=mass_cache+(long long)e*MASS_STRIDE;
        int* const cached_mass_piv=mass_pivot_cache+(long long)e*2*NEL;
#if RAW_USE_MASS
        // Cached reconstruction: reload the assembly's factored mass (and pivots).
        for(int idx=tid;idx<size;idx+=blockDim.x) factor[idx]=cached_mass[idx];
        if(kind==6) for(int i=tid;i<n;i+=blockDim.x) mass_piv[i]=cached_mass_piv[i];
        __syncthreads();
#else
        // Fold jac*w_q*kappa^{-1}_ab(q) into point coefficients (scratch: flux,
        // u and coef are contiguous), then form phi_i phi_j once per i <= j
        // pair for every needed block: each block is symmetric in (i, j).
        const int blocks=kind==3 ? 1 : (kind==4 ? 2 : (kind==5 ? 3 : 4));
        double* point=flux;
        for(int idx=tid;idx<blocks*NQ;idx+=blockDim.x) {
            const int b=idx/NQ,q=idx%NQ;
            const int component=kind==3 ? 0 : (kind==4 ? 3*b : (kind==5 ? (b==0 ? 0 : b+1) : b));
            point[idx]=jac*weights[q]*inverse[(e*NQ+q)*4+component];
        }
        __syncthreads();
        for(int pair=tid;pair<packed(NEL,0);pair+=blockDim.x) {
            int i=0,rest=pair;
            while(rest>=NEL-i) {rest-=NEL-i;++i;}
            const int j=i+rest;  // i <= j
            double v[4]={0.,0.,0.,0.};
            for(int q=0;q<NQ;++q) {
                const double product=basis_t[q*NEL+i]*basis_t[q*NEL+j];
#pragma unroll
                for(int b=0;b<4;++b) if(b<blocks) v[b]+=product*point[b*NQ+q];
            }
            if(kind<=4) {
                for(int b=0;b<blocks;++b) factor[b*packed(NEL,0)+packed(j,i)]=v[b];
            } else if(kind==5) {
                factor[packed(j,i)]=v[0];                    // block 00
                factor[packed(NEL+i,j)]=v[1];                // block 10 (kappa^{-1}_10)
                factor[packed(NEL+j,i)]=v[1];
                factor[packed(NEL+j,NEL+i)]=v[2];            // block 11
            } else {
                for(int b=0;b<4;++b) {
                    const int row=(b/2)*NEL, col=(b%2)*NEL;
                    factor[(row+i)*n+col+j]=v[b];
                    factor[(row+j)*n+col+i]=v[b];
                }
            }
        }
        __syncthreads();
        if(kind==6) {
            factor_mass(factor,mass_piv,status);
            if(!*status) invert_diagonal(factor,n);
        } else {
            chol(factor,n,status);
            if(kind==4 && !*status) chol(factor+packed(NEL,0),NEL,status);
        }
        if(*status) {if(tid==0) *status+=1000; return;}
#if RAW_WRITE_MASS
        // Keep the factored mass for reconstruct_adr_tensor (RAW_USE_MASS).
        for(int idx=tid;idx<size;idx+=blockDim.x) cached_mass[idx]=factor[idx];
        if(kind==6) for(int i=tid;i<n;i+=blockDim.x) cached_mass_piv[i]=mass_piv[i];
#endif
#endif
    }
    double* const cached_lu=schur_cache+(long long)e*NEL*NEL;
    int* const cached_piv=pivot_cache+(long long)e*NEL;
#if RAW_USE_FACTORS
    // Cached reconstruction: reload the assembly's Schur LU and rebuild only
    // the quadrature-free coupling blocks N-D. Variable-tensor mass factors
    // (kinds 3--6) were reloaded (RAW_USE_MASS) or refactored above.
    for(int idx=tid;idx<NEL*NEL;idx+=blockDim.x) {
        schur[idx]=cached_lu[idx];
        normal_minus_derivative(idx,e,a,face_jacs,normals,face_mass,d0,d1,mn0,mn1);
    }
    for(int i=tid;i<NEL;i+=blockDim.x) piv[i]=cached_piv[i];
    __syncthreads();
#else
    // Fold weights, jac and the affine gradient map into per-point
    // coefficients once, so the NEL^2 contractions carry no FP64 divides:
    // jac*w*(beta.grad phi) = w*[(bx*a3-by*a1) d0phi + (by*a0-bx*a2) d1phi].
    for(int q=tid;q<NQ;q+=blockDim.x) {
        const double w=weights[q], bx=beta[(e*NQ+q)*2], by=beta[(e*NQ+q)*2+1];
        coef[q]=jac*w*reaction[e*NQ+q];
        coef[NQ+q]=w*(bx*a[3]-by*a[1]);
        coef[2*NQ+q]=w*(by*a[0]-bx*a[2]);
    }
    for(int fq=tid;fq<3*NFQ;fq+=blockDim.x)
        coef[3*NQ+fq]=face_jacs[e*3+fq/NFQ]*face_weights[fq%NFQ]*t[fq];
    __syncthreads();
    // Assemble scalar reaction-advection and all weighted face contractions,
    // register-blocked: a thread owns row i and QUAD_COLS consecutive columns,
    // forms h_i(q) = c0 phi_i - c1 d0phi_i - c2 d1phi_i once per point and
    // accumulates phi_j(q) h_i(q) for its columns (per-entry sums keep the
    // serial order). Transposed tables make the per-point reads contiguous.
    for(int task=tid;task<NEL*QUAD_COL_BLOCKS;task+=blockDim.x) {
        const int i=task/QUAD_COL_BLOCKS, j0=(task%QUAD_COL_BLOCKS)*QUAD_COLS;
        double v[QUAD_COLS];
#pragma unroll
        for(int c=0;c<QUAD_COLS;++c) v[c]=0.;
        for(int q=0;q<NQ;++q) {
            const double h=coef[q]*basis_t[q*NEL+i]-coef[NQ+q]*gradients_t[q*NEL+i]
                           -coef[2*NQ+q]*gradients_t[(NQ+q)*NEL+i];
#pragma unroll
            for(int c=0;c<QUAD_COLS;++c) if(j0+c<NEL) v[c]+=basis_t[q*NEL+j0+c]*h;
        }
        for(int fq=0;fq<3*NFQ;++fq) {
            const double h=coef[3*NQ+fq]*phi_t[fq*NEL+i];
#pragma unroll
            for(int c=0;c<QUAD_COLS;++c) if(j0+c<NEL) v[c]+=h*phi_t[fq*NEL+j0+c];
        }
#pragma unroll
        for(int c=0;c<QUAD_COLS;++c) {
            if(j0+c>=NEL) continue;
            schur[i*NEL+j0+c]=v[c];
            normal_minus_derivative(i*NEL+j0+c,e,a,face_jacs,normals,face_mass,d0,d1,mn0,mn1);
        }
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
        inverse_mass(flux,factor,mass_piv,kind,tensor,minv,inv_jac,count);
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
#if RAW_WRITE_FACTORS
    // Keep the pivoted LU for reconstruct_adr_tensor (RAW_USE_FACTORS).
    for(int idx=tid;idx<NEL*NEL;idx+=blockDim.x) cached_lu[idx]=schur[idx];
    for(int i=tid;i<NEL;i+=blockDim.x) cached_piv[i]=piv[i];
#endif
#endif
    lu_solve_setup(schur,piv,schur_rows,inverse_diagonal);
#if RAW_RECONSTRUCT
    // lambda = J_F w_q sum_k sign_k trace_k psi_k(q): the trace on face points.
    for(int fq=tid;fq<3*NFQ;fq+=blockDim.x) {
        const int f=fq/NFQ,q=fq%NFQ;
        double v=0.;
        for(int k=0;k<NTR;++k)
            v+=raw_trace_orientation_sign(orientations[e*3+f],k)*trace[loc2glob[e*3+f]*NTR+k]
               *psi[raw_trace_local_dof(orientations[e*3+f],k,NTR)*NFQ+q];
        coef[fq]=face_jacs[e*3+f]*face_weights[q]*v;
    }
    __syncthreads();
    const int output_columns=1;
#else
    const int output_columns=NCOLS;
#endif
    for(int start=0;start<output_columns;start+=RAW_BATCH_COLS) {
        int count=min(RAW_BATCH_COLS,output_columns-start);
        for(int idx=tid;idx<NEL*RAW_BATCH_COLS;idx+=blockDim.x) {
            int i=idx/RAW_BATCH_COLS,c=idx%RAW_BATCH_COLS;
            if(c>=count) continue;
            double x=0.,y=0.;
#if RAW_RECONSTRUCT
            double v=source[e*NEL+i];
            for(int f=0;f<3;++f) {
                double coupling=trace_moment(f,i,phi_t,coef,nullptr);
                x+=normals[(e*3+f)*2]*coupling;y+=normals[(e*3+f)*2+1]*coupling;
                v+=trace_moment(f,i,phi_t,coef,g);
            }
#else
            const int col=start+c;
            double v=col==3*NTR ? source[e*NEL+i] : 0.;
            for(int tc=col;tc<min(col+1,3*NTR);++tc) {
                int f=tc/NTR,d=raw_trace_local_dof(orientations[e*3+f],tc%NTR,NTR);
                double scale=face_jacs[e*3+f]*raw_trace_orientation_sign(orientations[e*3+f],tc%NTR);
                double coupling=scale*face_moment(f,i,d,phi_t,psi,face_weights,nullptr);
                x+=normals[(e*3+f)*2]*coupling;y+=normals[(e*3+f)*2+1]*coupling;
                v+=scale*face_moment(f,i,d,phi_t,psi,face_weights,g);
            }
#endif
            u[idx]=v;flux[idx]=x;flux[NEL*RAW_BATCH_COLS+idx]=y;
        }
        __syncthreads();
        inverse_mass(flux,factor,mass_piv,kind,tensor,minv,inv_jac,count);
        for(int idx=tid;idx<NEL*RAW_BATCH_COLS;idx+=blockDim.x) {
            int i=idx/RAW_BATCH_COLS,c=idx%RAW_BATCH_COLS;
            if(c>=count) continue;
            double v=0.;
            for(int j=0;j<NEL;++j)
                v+=mn0[i*NEL+j]*flux[j*RAW_BATCH_COLS+c]+mn1[i*NEL+j]*flux[(NEL+j)*RAW_BATCH_COLS+c];
            u[idx]+=v;
        }
        __syncthreads();
        solve_lu_columns_warp(schur,schur_rows,inverse_diagonal,u,count);
        for(int idx=tid;idx<2*NEL*RAW_BATCH_COLS;idx+=blockDim.x) {
            int i=(idx/RAW_BATCH_COLS)%NEL,dim=idx/(NEL*RAW_BATCH_COLS),c=idx%RAW_BATCH_COLS;
            if(c>=count) continue;
            double v=0.;
            for(int j=0;j<NEL;++j) v+=derivative(dim,i*NEL+j,a,d0,d1)*u[j*RAW_BATCH_COLS+c];
#if RAW_RECONSTRUCT
            for(int f=0;f<3;++f) v-=normals[(e*3+f)*2+dim]*trace_moment(f,i,phi_t,coef,nullptr);
#else
            int col=start+c;
            for(int tc=col;tc<min(col+1,3*NTR);++tc) {
                int f=tc/NTR,d=raw_trace_local_dof(orientations[e*3+f],tc%NTR,NTR);
                double scale=face_jacs[e*3+f]*raw_trace_orientation_sign(orientations[e*3+f],tc%NTR);
                v-=scale*normals[(e*3+f)*2+dim]*face_moment(f,i,d,phi_t,psi,face_weights,nullptr);
            }
#endif
            flux[idx]=v;
        }
        __syncthreads();
        inverse_mass(flux,factor,mass_piv,kind,tensor,minv,inv_jac,count);
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
        // Weighted face traces tau*u + n.q, once per (face, point, column);
        // every trace row of that face then only projects onto psi.
        for(int idx=tid;idx<3*NFQ*count;idx+=blockDim.x) {
            int rf=idx/(NFQ*count),q=(idx/count)%NFQ,c=idx%count;
            double uh=0.,qx=0.,qy=0.;
            for(int i=0;i<NEL;++i) {
                double p=phi[(rf*NEL+i)*NFQ+q];
                uh+=p*u[i*RAW_BATCH_COLS+c];qx+=p*flux[i*RAW_BATCH_COLS+c];qy+=p*flux[(NEL+i)*RAW_BATCH_COLS+c];
            }
            coef[idx]=face_weights[q]*(t[rf*NFQ+q]*uh+normals[(e*3+rf)*2]*qx+normals[(e*3+rf)*2+1]*qy);
        }
        __syncthreads();
        for(int task=tid;task<3*NTR*count;task+=blockDim.x) {
            int row=task/count,c=task%count,rf=row/NTR,rd=row%NTR,col=start+c;
            long long sid=sides[e*3+rf],re=edge_map[loc2glob[e*3+rf]];
            if(sid<0 || re<0) continue;
            int ld=raw_trace_local_dof(orientations[e*3+rf],rd,NTR);
            double rs=raw_trace_orientation_sign(orientations[e*3+rf],rd);
            double v=0.;
            for(int q=0;q<NFQ;++q) v+=psi[ld*NFQ+q]*coef[(rf*NFQ+q)*count+c];
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


def coefficient_storage(nq, nfq, batch, nel=0):
    """Doubles for per-point coefficients, reused for weighted face traces.

    Variable-tensor mass assembly also stages up to 4*nq point coefficients in
    the contiguous flux/u/coef scratch (3*nel*batch doubles before this one).
    """
    return max(3 * nq + 3 * nfq, 3 * nfq * batch, 4 * nq - 3 * nel * batch)


CACHE_LOCAL_FACTORS = ("none", "schur-lu", "schur-lu+mass")


def mass_factor_size(nel, kind):
    """Reals of one element's factored inverse-diffusion mass for ``kind``.

    Kinds 3--5 hold packed Cholesky factors (one NEL block, two NEL blocks,
    one 2*NEL block); kind 6 holds the full 2*NEL LU; constant kinds need none.
    """
    packed = nel * (nel + 1) // 2
    return (0, 0, 0, packed, 2 * packed, nel * (2 * nel + 1), 4 * nel * nel)[int(kind)]


TENSOR_SHARED_MEMORY_LIMIT = 48 * 1024
TENSOR_MAX_BATCH_COLUMNS = 8


class TensorWorkspaceError(UnsupportedBackendConfigurationError, ValueError):
    """One element's tensor ADR workspace exceeds the shared-memory limit.

    Raised before any upload, compilation or launch. It remains a
    ``ValueError`` for callers of the former generic resource error.
    """


def tensor_shared_bytes(nel, kind, nq, nfq, batch):
    """Return factor/scratch slots and dynamic shared bytes of one element block.

    Bytes use the selected precision (``REAL_ITEMSIZE``); integer pivots and
    the Schur permutation take 4*nel ints. The size never decreases with
    ``nq``, ``nfq``, ``batch`` or ``kind``.
    """
    storage = max(mass_factor_size(nel, kind), 2 * nel * batch)
    size = (REAL_ITEMSIZE * (3 * nel * nel + storage + 3 * nel * batch
                             + coefficient_storage(nq, nfq, batch, nel) + nel)
            + 4 * 4 * nel)
    return storage, size


def max_tensor_volume_points(nel, kind, nfq):
    """Return the largest volume point count NQ that fits at batch width 1 (0 if none)."""
    def fits(nq):
        """Return whether ``nq`` volume points fit at batch width 1."""
        return tensor_shared_bytes(nel, kind, nq, nfq, 1)[1] <= TENSOR_SHARED_MEMORY_LIMIT

    if not fits(1):
        return 0
    low, high = 1, 2
    while fits(high):
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if fits(middle) else (low, middle)
    return low


def tensor_workspace(nel, kinds, nq=0, nfq=0, *, order=None):
    """Return batch width, factor/scratch slots, and bounded shared bytes.

    The widest batch (at most ``TENSOR_MAX_BATCH_COLUMNS``) whose workspace
    fits ``TENSOR_SHARED_MEMORY_LIMIT`` is chosen for the most expensive
    diffusion kind present. When even one column does not fit,
    ``TensorWorkspaceError`` names the order, kind, quadrature sizes and the
    largest volume rule that would fit.
    """
    from ..assembly.diffusion_coefficients import DIFFUSION_KINDS

    maximum = int(kinds.max()) if kinds.size else 0
    for batch in range(TENSOR_MAX_BATCH_COLUMNS, 0, -1):
        storage, size = tensor_shared_bytes(nel, maximum, nq, nfq, batch)
        if size <= TENSOR_SHARED_MEMORY_LIMIT:
            return batch, storage, size
    label = f"p={order}, " if order is not None else ""
    largest = max_tensor_volume_points(nel, maximum, nfq)
    remedy = (f"at most NQ={largest} volume points fit with NFQ={nfq}" if largest
              else f"no volume rule fits with NFQ={nfq}")
    raise TensorWorkspaceError(
        f"raw-CUDA tensor ADR workspace for {label}NEL={nel}, diffusion kind {maximum} "
        f"({DIFFUSION_KINDS[maximum]}), NQ={nq} volume and NFQ={nfq} face quadrature points "
        f"needs {size:,} B of shared memory with one batch column, above the "
        f"{TENSOR_SHARED_MEMORY_LIMIT:,} B limit; {remedy}. Reduce volume_quad_1d or "
        "edge_quad_1d, use the default volume rule, or select assembly_backend='numba'.")


def kernel_source(nel, ntr, nq, nfq, orientation, matrix_format, batch, storage, *,
                  reconstruct=False, cached_factors=False, mass_stride=0, reuse_mass=False):
    """Build a shape-specialized kernel using diffusion's shared local helpers.

    ``cached_factors`` makes assembly write the per-element Schur LU cache and
    makes reconstruction read it instead of refactoring. ``mass_stride > 0``
    does the same for the factored variable-tensor mass (reals per element);
    ``reuse_mass`` makes assembly read that mass cache as well, for repeated
    assemblies with an unchanged diffusion tensor. The source is written in double; ``real_raw_module`` specializes it to the
    selected precision.
    """
    # Register blocking of the scalar quadrature: 4 column blocks per row.
    quad_cols = (nel + 3) // 4
    defines = dict(NEL=nel, NTR=ntr, NQ=nq, NFQ=nfq, NCOLS=3*ntr+1,
                   QUAD_COLS=quad_cols, QUAD_COL_BLOCKS=(nel + quad_cols - 1) // quad_cols,
                   TRACE_ORIENTATION_MODE=orientation, RAW_BATCH_COLS=batch,
                   FACTOR_STORAGE=storage, COEF_STORAGE=coefficient_storage(nq, nfq, batch, nel),
                   RAW_MATRIX_CSR=int(matrix_format!='coo'),
                   RAW_MATRIX_BSR=int(matrix_format=='bsr'), RAW_RECONSTRUCT=int(reconstruct),
                   RAW_WRITE_FACTORS=int(cached_factors and not reconstruct),
                   RAW_USE_FACTORS=int(cached_factors and reconstruct),
                   RAW_WRITE_MASS=int(mass_stride > 0 and not reconstruct and not reuse_mass),
                   RAW_USE_MASS=int(mass_stride > 0 and (reconstruct or reuse_mass)),
                   MASS_STRIDE=max(int(mass_stride), 1))
    prefix = ''.join(f'#define {key} {value}\n' for key, value in defines.items())
    source = (prefix + RAW_TRACE_ORIENTATION_HELPERS + RAW_COOPERATIVE_SOLVES + RAW_WARP_COLUMN_SOLVES
            + checked_warp_lu_source('factor_schur', 'NEL')
            + checked_warp_lu_source('factor_mass', '(2 * NEL)') + _MASS_ALGEBRA + _ASSEMBLY)
    return source.replace('void assemble_adr_tensor(', 'void reconstruct_adr_tensor(') if reconstruct else source


def assemble_tensor_operator(prepared, boundary_condition, space, *, diffusion,
                             trace_space, matrix_format, block_size, cache_local_factors="schur-lu+mass",
                             prepared_diffusion=None, csr_pattern=None, mass_factors=None):
    """Prepare reference/device data and emit a reduced operator without a solve.

    ``prepared_diffusion`` (a ``PreparedDiffusion`` of the same ``diffusion``
    and space) and ``csr_pattern`` (``operator.csr_pattern`` of an earlier
    CSR/BSR assembly on the same space, trace space and format) skip their
    time-independent construction for repeated assemblies. ``mass_factors``
    (``operator.mass_factors`` of an earlier assembly with the same diffusion,
    space and ``cache_local_factors="schur-lu+mass"``) makes the kernel reload
    the factored variable-tensor mass instead of assembling and factoring it.

    ``cache_local_factors`` selects what ``reconstruct_tensor_operator`` reuses:
    ``"schur-lu"`` keeps every element's pivoted Schur LU on the device,
    ``"schur-lu+mass"`` also the factored variable-tensor mass (none is stored
    for constant tensors), and ``"none"`` refactors everything there. Arrays
    and the kernel follow the selected precision (``HDGFEM_PRECISION``).
    """
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
    # Device-prepared samples imply device kappa^{-1} tables; a CuPy-incompatible
    # diffusion callable (TypeError) falls back to host sampling.
    device_samples = isinstance(prepared.tau_total, cp.ndarray)
    if prepared_diffusion is not None:
        tensor = prepared_diffusion
        timings['raw.coefficients.diffusion.cached'] = 1.
    else:
        try:
            tensor = prepare_diffusion(diffusion, space, device=device_samples)
        except TypeError:
            if not device_samples:
                raise
            tensor = prepare_diffusion(diffusion, space)
            timings['raw.coefficients.diffusion.host_fallback'] = 1.
    stream.synchronize()
    timings['raw.coefficients.diffusion'] = time.perf_counter() - prep_start
    nel, ntr = space.el_dof, trace_space.edg_dof
    q, mesh = space.quad_data, space.mesh
    expected = (mesh.num_tri, 3, trace_space.weights.size)
    if prepared.tau_total.shape != expected or prepared.gamma.shape != expected:
        raise ValueError('ADR preparation face quadrature does not match trace_space')
    if prepared.face_quadrature is not None and not np.array_equal(prepared.face_quadrature, trace_space.quads):
        raise ValueError('ADR preparation face quadrature does not match trace_space')
    for name in ('reaction_values', 'beta_values', 'source_rhs', 'tau_total', 'gamma'):
        # Samples are host arrays (prepare_adr_data) or device arrays (prepare_adr_data_cupy).
        values = getattr(prepared, name)
        if not bool(cp.get_array_module(values).isfinite(values).all()):
            raise ValueError(f'ADR {name} samples must be finite')
    tau_d = prepared.tau_diffusion
    xp_tau = cp.get_array_module(tau_d)
    if not bool(xp_tau.isfinite(tau_d).all()) or bool((tau_d <= 0).any()):
        raise ValueError('diffusion stabilization must be finite and strictly positive')
    batch, storage, shared_bytes = tensor_workspace(nel, tensor.kinds, q.Krf_w.size, trace_space.weights.size,
                                                    order=space.order)
    upload_start = time.perf_counter()
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(trace_space)
    validate_raw_cuda_supported(cspace, trace_ref)
    boundary_host = hdg.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_space)
    if not np.all(np.isfinite(boundary_host)):
        raise ValueError('ADR boundary trace must be finite')
    boundary = cp.asarray(boundary_host, dtype=REAL_DTYPE)
    face_mass = np.einsum('q,fiq,fjq->fij', trace_space.weights,
                          trace_space.bas_of_bd_quads, trace_space.bas_of_bd_quads, optimize=True)
    # Mesh and volume tables come from the cached device space; device samples
    # pass through, so only host-prepared samples and small tables are copied.
    # The volume/face bases and gradients also go in transposed ((NQ, NEL),
    # (2, NQ, NEL), (3, NFQ, NEL)): loops whose lanes run over basis functions
    # then read contiguous rows instead of strided columns.
    cq, cmesh = cspace.quad_data, cspace.mesh
    arrays = (cmesh.aff_mats, cmesh.aff_jacs, cmesh.jacs_el_fc, cmesh.normals,
              cq.MKrf_inv, cq.bas_of_quads, cq.bas_of_quads.T, cq.dbas_of_quads,
              cq.dbas_of_quads.transpose(0, 2, 1), cq.Krf_w,
              trace_space.bas_of_bd_quads, np.swapaxes(trace_space.bas_of_bd_quads, 1, 2),
              trace_space.bas1d_of_ref_edg_qds,
              trace_space.weights, face_mass, prepared.d0_reference, prepared.d1_reference,
              prepared.reaction_values, prepared.beta_values, prepared.tau_total,
              prepared.gamma, prepared.source_rhs)
    inputs = tuple(cp.ascontiguousarray(cp.asarray(value, dtype=REAL_DTYPE)) for value in arrays)
    kinds = cp.asarray(tensor.kinds, dtype=cp.int64)
    constants = cp.ascontiguousarray(cp.asarray(tensor.constants, dtype=REAL_DTYPE))
    inverse = cp.ascontiguousarray(cp.asarray(tensor.inverse_values, dtype=REAL_DTYPE))
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
        data = cp.empty(rows.size, dtype=REAL_DTYPE)
        indptr = indices = None
        row_arg, col_arg, ptr_arg, pos_arg = rows, cols, dummy32, dummy32
    else:
        pattern = csr_pattern if csr_pattern is not None else build_reduced_csr_pattern_raw(
            cspace, timings, matrix_format=matrix_format)
        edge_map, sides = pattern.edge_to_solve_edge, pattern.interior_side_index
        indptr, indices = pattern.indptr, pattern.indices
        data = cp.zeros((pattern.num_blocks, ntr, ntr) if matrix_format == 'bsr'
                        else indices.size, dtype=REAL_DTYPE)
        rows = cols = None
        row_arg = col_arg = offsets = dummy64
        ptr_arg, pos_arg = indptr, pattern.side_csr_block_pos
    rhs = cp.zeros(mesh.int_edges_inds.size*ntr, dtype=REAL_DTYPE)
    statuses = cp.zeros(mesh.num_tri, dtype=cp.int32)
    if cache_local_factors not in CACHE_LOCAL_FACTORS:
        raise ValueError(f"cache_local_factors must be one of {CACHE_LOCAL_FACTORS}")
    cached = cache_local_factors != "none"
    max_kind = int(tensor.kinds.max()) if tensor.kinds.size else 0
    mass_stride = mass_factor_size(nel, max_kind) if cache_local_factors == "schur-lu+mass" else 0
    schur_cache = cp.empty((mesh.num_tri, nel, nel) if cached else 1, dtype=REAL_DTYPE)
    pivot_cache = cp.empty((mesh.num_tri, nel) if cached else 1, dtype=cp.int32)
    reuse_mass = bool(mass_stride and mass_factors is not None
                      and mass_factors[0].shape == (mesh.num_tri, mass_stride))
    if reuse_mass:
        mass_cache, mass_pivot_cache = mass_factors
        timings['raw.mass_factors.cached'] = 1.
    else:
        mass_cache = cp.empty((mesh.num_tri, mass_stride) if mass_stride else 1, dtype=REAL_DTYPE)
        mass_pivot_cache = cp.empty((mesh.num_tri, 2*nel) if mass_stride and max_kind == 6 else 1, dtype=cp.int32)
    stream.synchronize()
    timings['raw.graph'] = time.perf_counter() - graph_start
    jit_start = time.perf_counter()
    code = kernel_source(nel, ntr, q.Krf_w.size, trace_space.weights.size,
                         _trace_orientation_mode(trace_space), matrix_format, batch, storage,
                         cached_factors=cached, mass_stride=mass_stride, reuse_mass=reuse_mass)
    module = real_raw_module(code=code, options=('--std=c++11',))
    kernel = module.get_function('assemble_adr_tensor')
    timings['raw.kernel.jit'] = time.perf_counter() - jit_start
    args = (row_arg, col_arg, data, rhs, ptr_arg, pos_arg, offsets,
            cspace.mesh.loc2glob_edge, cspace.mesh.orientations, sides, edge_map,
            *inputs, boundary, kinds, constants, inverse, np.int32(tensor.constants.shape[0]),
            schur_cache, pivot_cache, mass_cache, mass_pivot_cache, statuses)
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
                    'raw.batch_columns': float(batch), 'raw.conversion': 0., 'raw.coo_to_csr.wall': 0.,
                    'raw.local_factor_bytes': float(sum(array.nbytes for array in (
                        schur_cache, pivot_cache, mass_cache, mass_pivot_cache) if array.size > 1))})
    timings['raw.wall_total'] = time.perf_counter() - start
    timings['raw.total_with_preparation'] = timings['raw.wall_total'] + prepared.preparation_seconds
    assembly = CudaAdvectionAssembly(
        rows, cols, data, rhs, None, None, inputs[-1], boundary[cspace.mesh.bnd_edges_inds],
        None, cspace, trace_ref, indptr=indptr, indices=indices,
        matrix_format=matrix_format, timings=timings)
    return RawADRTraceOperator(assembly, module, inputs, boundary,
                               mass_factors=(mass_cache, mass_pivot_cache) if mass_stride else None,
                               csr_pattern=pattern, diffusion_structure=tensor.counts,
                               diffusion_kinds=tensor.kinds.copy(),
                               reconstruction_data=dict(args=args[:-1], nel=nel, ntr=ntr,
                                   nq=q.Krf_w.size, nfq=trace_space.weights.size,
                                   orientation=_trace_orientation_mode(trace_space),
                                   batch=batch, storage=storage, shared_bytes=shared_bytes,
                                   cached_factors=cached, mass_stride=mass_stride))


def reconstruct_tensor_operator(operator, trace, *, block_size="auto"):
    """Reconstruct device [u, qx, qy] with the assembly's exact coefficient data.

    No dense local operators or coefficient samples are uploaded again. With
    the assembly's Schur LU cache the kernel skips quadrature and factorization;
    otherwise local factors are rebuilt using the same kernel algebra.
    """
    from .cupy import require_cupy
    from .raw_cuda import resolve_raw_cuda_block_size
    cp=require_cupy()
    local=operator.reconstruction_data
    if local is None:
        raise ValueError("tensor reconstruction requires the cooperative/lightweight assembly operator")
    space=operator.assembly.cspace.host
    trace=cp.ascontiguousarray(cp.asarray(trace,dtype=REAL_DTYPE)).reshape(-1)
    if trace.size!=space.mesh.num_edg*local['ntr']:
        raise ValueError("reconstruction requires the full trace, including boundary coefficients")
    if not bool(cp.all(cp.isfinite(trace))):
        raise ValueError("reconstruction trace must be finite")
    block_size=resolve_raw_cuda_block_size(block_size,equation="diffusion-reaction",order=space.order)
    shape={key:local[key] for key in ('nel','ntr','nq','nfq','orientation','batch','storage',
                                      'cached_factors','mass_stride')}
    start=time.perf_counter()
    module=real_raw_module(code=kernel_source(**shape,matrix_format='csr',reconstruct=True),options=('--std=c++11',))
    kernel=module.get_function('reconstruct_adr_tensor')
    timings={'raw.reconstruction.jit':time.perf_counter()-start}
    unknowns=cp.empty((space.mesh.num_tri,3*space.el_dof),dtype=REAL_DTYPE)
    statuses=cp.zeros(space.mesh.num_tri,dtype=cp.int32)
    begin,end=cp.cuda.Event(),cp.cuda.Event()
    start=time.perf_counter()
    begin.record()
    kernel((space.mesh.num_tri,),(block_size,),(*local['args'],statuses,trace,unknowns),shared_mem=local['shared_bytes'])
    end.record()
    end.synchronize()
    timings['raw.reconstruction.wall']=time.perf_counter()-start
    timings['raw.reconstruction.cached_factors']=float(bool(local['cached_factors']))
    timings['raw.reconstruction.device']=cp.cuda.get_elapsed_time(begin,end)/1000.
    if bool(cp.any(statuses)):
        failures=cp.asnumpy(statuses)
        element=int(np.flatnonzero(failures)[0])
        raise np.linalg.LinAlgError(f'ADR reconstruction factorization/finite-value failure in element {element} (status {failures[element]})')
    return unknowns,timings
