"""Compact scalar-Schur Poisson kernels; all local rows stride the block.

Reference contractions remove the directional dense matrix intermediates.
RETAIN keeps the two coupling matrices when the device can profit from them.
PHASE=0 assembles directly into COO/CSR/BSR; 1 refreshes the reduced RHS;
2 reconstructs the full mixed field. SAMPLE writes isolated element responses.
"""
from hybridge.hdg.cuda.raw_source import RAW_WARP_COLUMN_SOLVES, checked_warp_lu_source


# WARP_SOLVE: the phases that apply stored LU factors solve each local column on a warp and
# read the factors in place instead of copying them to shared memory. The one-lane sweeps on
# the shared copy took 57-59% of block cycles at p=7-9 (campaigns/poisson_element_cost,
# 2026-11-02). Block size 1, allowed up to p=6, keeps the sub-warp column solve.
#
# INVERSE and the EXP_ macros are experimental switches of that study, default 0 and NOT
# production: the top of poisson.py gives the status of each one (not promoted, failed or
# diagnostic). When all are 0, every EXP_ line preprocesses to nothing and the kernel is the
# production one token for token.
_EXPERIMENTS = r'''
#ifndef EXP_CONTRACT_Q
#define EXP_CONTRACT_Q 0
#endif
#ifndef EXP_CONTRACT_COUPLING
#define EXP_CONTRACT_COUPLING 0
#endif
#ifndef EXP_LOAD_WIDE
#define EXP_LOAD_WIDE 0
#endif
#ifndef EXP_CHOLESKY_WARP
#define EXP_CHOLESKY_WARP 0
#endif
#ifndef EXP_PHASE_CLOCK
#define EXP_PHASE_CLOCK 0
#endif
#ifndef EXP_STOP
#define EXP_STOP 0
#endif
// Diagnostic for the kernel replay only: clock64 per stage into validation[e*EXP_SLOTS+k].
#if EXP_PHASE_CLOCK
#define EXP_SLOTS 10
#define EXP_CLOCK_START long long exp_last=clock64();
#define EXP_CLOCK(k) if(tid==0 && !sample) { const long long exp_now=clock64(); \
    validation[e*EXP_SLOTS+(k)]+=(double)(exp_now-exp_last); exp_last=exp_now; }
#else
#define EXP_CLOCK_START
#define EXP_CLOCK(k)
#endif
// Diagnostic for the kernel replay only: stop after stage EXP_STOP. A truncated build keeps
// its stores alive through one sink value per element.
#if EXP_STOP
#define EXP_STOP_AT(k) if(EXP_STOP==(k)) { __syncthreads(); \
    if(tid==0) validation[slot]=storage[0]+u[0]+qx[0]+qy[0]; return; }
#else
#define EXP_STOP_AT(k)
#endif
'''

# Experimental, not promoted: warp solve of one column with a Cholesky factor, lower triangle
# in row-major order as this kernel stores it. Needs a block of at least one full warp.
_WARP_CHOLESKY = r'''
#if EXP_CHOLESKY_WARP
__device__ __forceinline__ void cholesky_solve_warp(
        const double* __restrict__ l, const double* __restrict__ inverse_diagonal, double* __restrict__ columns) {
    if(threadIdx.x<32) {
        const int lane=threadIdx.x&31;
        constexpr int R=(NEL+31)/32;
        double x[R];
#pragma unroll
        for(int r=0;r<R;++r) { const int row=lane+32*r; x[r]=row<NEL ? columns[row*BATCH] : 0.; }
        for(int j=0;j<NEL;++j) {
#pragma unroll
            for(int r=0;r<R;++r) if(lane+32*r==j) x[r]*=inverse_diagonal[j];
            const double xj=warp_row_value(x,j);
#pragma unroll
            for(int r=0;r<R;++r) { const int row=lane+32*r; if(row>j && row<NEL) x[r]-=l[row*NEL+j]*xj; }
        }
        for(int j=NEL-1;j>=0;--j) {
#pragma unroll
            for(int r=0;r<R;++r) if(lane+32*r==j) x[r]*=inverse_diagonal[j];
            const double xj=warp_row_value(x,j);
#pragma unroll
            for(int r=0;r<R;++r) { const int row=lane+32*r; if(row<j) x[r]-=l[j*NEL+row]*xj; }
        }
#pragma unroll
        for(int r=0;r<R;++r) { const int row=lane+32*r; if(row<NEL) columns[row*BATCH]=x[r]; }
    }
    __syncthreads();
}
#endif
'''

SOURCE = _EXPERIMENTS + r'''
#define RAW_BATCH_COLS BATCH
#define RAW_COOPERATIVE_ONLY (NEL > 28)
''' + checked_warp_lu_source('poisson_lu', 'NEL') + RAW_WARP_COLUMN_SOLVES + _WARP_CHOLESKY + r'''
__device__ double poisson_coupling(int i, int j, int z, const double* d,
        const double* fm, const double* a, const double* scales, const double* normals) {
    double value = -a[2*z] * d[i*NEL+j] - a[2*z+1] * d[NEL*NEL+i*NEL+j];
    for (int f=0; f<3; ++f)
        value += scales[f]*normals[2*f+z]*fm[(f*NEL+i)*NEL+j];
    return value;
}

extern "C" __global__ void poisson_local(
    const long long* elements, const int sample, const long long count,
    const double* dd, const double* fd, const double* d, const double* md,
    const double* fm, const double* lifts, const double* mlifts, const double* edge_mass,
    const double* affine, const double* jacobians, const double* scales_all,
    const double* normals_all, const long long* edges, const long long* oriented,
    const long long* side_index, const long long* solve_edges,
    const int* indptr, const int* side_positions, const long long* offsets,
    const double* source, const int source_stride, const double* trace, const double tau,
    double* factors, int* all_pivots, double* responses, double* source_solutions,
    long long* rows, long long* cols, double* data, double* rhs, double* local,
    double* validation, int* failures) {
    const long long slot=blockIdx.x;
    if (slot>=count) return;
    const long long e=sample ? elements[slot] : slot;
    const int tid=threadIdx.x;
    const double* aff=affine+4*e;
    const double a[4]={aff[3], -aff[2], -aff[1], aff[0]};
    const double* scales=scales_all+3*e;
    const double* normals=normals_all+6*e;
    const double ji=1./jacobians[e];
    extern __shared__ double storage[];
    double* s=storage;
    double* mx=s+FACTOR_STORAGE;
    double* my=mx+(RETAIN ? NEL*NEL : 0);
    double* u=my+(RETAIN ? NEL*NEL : 0);
    double* qx=u+NEL*BATCH;
    double* qy=qx+NEL*BATCH;
    double* flux=qy+NEL*BATCH;
#if EXP_CONTRACT_COUPLING || EXP_CHOLESKY_WARP
    // Experimental layouts: NEL pivot reciprocals when a warp solve runs, then the 5*NEL
    // contracted coupling tables.
    double* inverse_diagonal=flux+3*NTR*BATCH;
    double* w=inverse_diagonal+((WARP_SOLVE || EXP_CHOLESKY_WARP) ? NEL : 0);
    int* pivots=reinterpret_cast<int*>(w+(EXP_CONTRACT_COUPLING ? 5*NEL : 0));
#elif WARP_SOLVE
    double* inverse_diagonal=flux+3*NTR*BATCH;
    int* pivots=reinterpret_cast<int*>(inverse_diagonal+NEL);
#else
    int* pivots=reinterpret_cast<int*>(flux+3*NTR*BATCH);
#endif
    int* status=pivots+NEL;
#if WARP_SOLVE
    int* permuted_rows=status+1;
    const double* __restrict__ factor_view=factors+e*NEL*NEL;
#endif
#if EXP_CHOLESKY_WARP
    const double* __restrict__ factor_view=factors+e*NEL*NEL;
#endif
    EXP_CLOCK_START
#if PHASE == 1
    // The refreshed trace holds boundary values only: an element without a boundary edge has a zero lift.
    const bool has_boundary=solve_edges[edges[3*e]]<0 || solve_edges[edges[3*e+1]]<0
        || solve_edges[edges[3*e+2]]<0;
#else
    const bool has_boundary=true;
#endif
    if(tid==0) *status=0;
    __syncthreads();
    EXP_CLOCK(0)
    EXP_STOP_AT(1)

#if !(PHASE == 2 && RESPONSE)
#if EXP_LOAD_WIDE && USE_FACTORS && !WARP_SOLVE && !EXP_CHOLESKY_WARP
    // Experimental, no gain: each thread issues EXP_LOAD_WIDE independent reads of the factor,
    // then stores them.
    for(int start=tid*EXP_LOAD_WIDE; start<NEL*NEL; start+=blockDim.x*EXP_LOAD_WIDE) {
        double loaded[EXP_LOAD_WIDE];
#pragma unroll
        for(int o=0; o<EXP_LOAD_WIDE; ++o)
            loaded[o]=start+o<NEL*NEL ? factors[e*NEL*NEL+start+o] : 0.;
#pragma unroll
        for(int o=0; o<EXP_LOAD_WIDE; ++o)
            if(start+o<NEL*NEL) s[start+o]=loaded[o];
    }
    __syncthreads();
#endif
    for(int ij=tid; ij<NEL*NEL; ij+=blockDim.x) {
        const int i=ij/NEL, j=ij%NEL;
#if USE_FACTORS
#if !WARP_SOLVE && !EXP_LOAD_WIDE && !EXP_CHOLESKY_WARP
        s[ij]=factors[e*NEL*NEL+ij];
#endif
#else
        double value=0.;
        for(int f=0; f<3; ++f) value+=tau*scales[f]*fm[f*NEL*NEL+ij];
        double diff=0.;
        for(int z=0; z<2; ++z) {
            for(int v=0; v<2; ++v) {
                for(int f=0; f<3; ++f)
                    diff+=scales[f]*normals[2*f+z]*a[2*z+v]*fd[(2*f+v)*NEL*NEL+ij];
                for(int w=0; w<2; ++w)
                    diff-=a[2*z+v]*a[2*z+w]*dd[(2*v+w)*NEL*NEL+ij];
            }
        }
        s[ij]=value+ji*diff;
#if CHOLESKY
        const double cx=poisson_coupling(i,j,0,d,fm,a,scales,normals);
        const double cy=poisson_coupling(i,j,1,d,fm,a,scales,normals);
        const double dx=a[0]*d[j*NEL+i]+a[1]*d[NEL*NEL+j*NEL+i];
        const double dy=a[2]*d[j*NEL+i]+a[3]*d[NEL*NEL+j*NEL+i];
        const double coupling_scale=fmax(1.,fmax(fabs(cx),fabs(cy)));
        if(fmax(fabs(cx-dx),fabs(cy-dy))>SYMMETRY_RTOL*coupling_scale)
            atomicExch(status,NEL+2);
#endif
#endif
#if RETAIN
        mx[ij]=poisson_coupling(i,j,0,d,fm,a,scales,normals);
        my[ij]=poisson_coupling(i,j,1,d,fm,a,scales,normals);
#endif
    }
    __syncthreads();
    EXP_CLOCK(1)
    EXP_STOP_AT(2)
#if USE_FACTORS
#if !INVERSE
    for(int i=tid; i<NEL; i+=blockDim.x) pivots[i]=all_pivots[e*NEL+i];
    __syncthreads();
#if WARP_SOLVE
    lu_solve_setup(factor_view,pivots,permuted_rows,inverse_diagonal);
#endif
#endif
#if EXP_CHOLESKY_WARP
    // Experimental, not promoted: pivot reciprocals of the Cholesky factor read in place.
    for(int i=tid; i<NEL; i+=blockDim.x) inverse_diagonal[i]=1./factor_view[i*(NEL+1)];
    __syncthreads();
#endif
    EXP_CLOCK(2)
    EXP_STOP_AT(3)
#else
#if CHOLESKY
    // Validate and symmetrize disjoint matrix pairs before in-place Cholesky.
    for(int ij=tid; ij<NEL*NEL; ij+=blockDim.x) {
        const int i=ij/NEL, j=ij%NEL;
        if(i>j) {
            double scale=fmax(1.,fmax(fabs(s[i*NEL+i]),fabs(s[j*NEL+j])));
            if(fabs(s[ij]-s[j*NEL+i])>SYMMETRY_RTOL*scale) atomicExch(status,NEL+1);
            const double mean=.5*(s[ij]+s[j*NEL+i]);
            s[ij]=mean; s[j*NEL+i]=mean;
        }
    }
    __syncthreads();
    for(int k=0; k<NEL && !*status; ++k) {
        if(tid==0) {
            if(!(s[k*NEL+k]>0.) || !isfinite(s[k*NEL+k])) *status=k+1;
            else s[k*NEL+k]=sqrt(s[k*NEL+k]);
        }
        __syncthreads();
        if(*status) break;
        for(int i=k+1+tid; i<NEL; i+=blockDim.x) s[i*NEL+k]/=s[k*NEL+k];
        __syncthreads();
        for(int ij=tid; ij<NEL*NEL; ij+=blockDim.x) {
            const int i=ij/NEL, j=ij%NEL;
            if(i>=j && j>k) s[ij]-=s[i*NEL+k]*s[j*NEL+k];
        }
        __syncthreads();
    }
#else
    poisson_lu(s,pivots,status);
#endif
    if(*status) {
        if(tid==0) failures[slot]=*status;
        return;
    }
#if WRITE_FACTORS
    if(!sample) {
        for(int ij=tid; ij<NEL*NEL; ij+=blockDim.x)
            factors[e*NEL*NEL+ij]=(CHOLESKY && ij/NEL<ij%NEL) ? 0. : s[ij];
        for(int i=tid; i<NEL; i+=blockDim.x)
            all_pivots[e*NEL+i]=CHOLESKY ? i : pivots[i];
    }
#endif
#endif
#endif
    __syncthreads();
    if(tid==0) failures[slot]=0;

    const int total_columns=PHASE==0 ? 3*NTR+1 : 1;
    for(int first=0; first<total_columns; first+=BATCH) {
        const int width=min(BATCH,total_columns-first);
        // qx/qy initially hold M^-1 times the prescribed flux-block RHS.
        for(int index=tid; index<NEL*BATCH; index+=blockDim.x) {
            const int i=index/BATCH, b=index%BATCH, column=first+b;
            if(b>=width) continue;
            double value=0., bx=0., by=0.;
#if PHASE == 0
            if(column==3*NTR) value=source[e*source_stride+i];
            else {
                const int f=column/NTR, t=column%NTR;
                const long long of=oriented[3*e+f];
                const long long li=(of*NTR+t)*NEL+i;
                value=tau*scales[f]*lifts[li];
                bx=scales[f]*normals[2*f]*mlifts[li];
                by=scales[f]*normals[2*f+1]*mlifts[li];
            }
#else
#if PHASE == 2 && RESPONSE
            value=source_solutions[e*NEL+i];
#else
            value=source[e*source_stride+i];
#endif
            if(has_boundary) for(int f=0; f<3; ++f) {
                const long long edge=edges[3*e+f], of=oriented[3*e+f];
                for(int t=0; t<NTR; ++t) {
                    const double g=trace[edge*NTR+t];
                    const long long li=(of*NTR+t)*NEL+i;
#if PHASE == 2 && RESPONSE
                    value+=responses[(e*NEL+i)*(3*NTR)+f*NTR+t]*g;
#elif !RESPONSE
                    value+=tau*scales[f]*lifts[li]*g;
#endif
                    bx+=scales[f]*normals[2*f]*mlifts[li]*g;
                    by+=scales[f]*normals[2*f+1]*mlifts[li]*g;
                }
            }
#endif
            u[index]=value; qx[index]=bx; qy[index]=by;
        }
        __syncthreads();
        EXP_CLOCK(3)
        EXP_STOP_AT(4)
#if !RESPONSE || PHASE == 0
        if(has_boundary) {
#if EXP_CONTRACT_COUPLING && BATCH == 1
        // Experimental, no gain: contract q on the five reference tables first, then one
        // matrix-vector product per table.
        for(int index=tid; index<5*NEL; index+=blockDim.x) {
            const int table=index/NEL, k=index%NEL;
            const double x=qx[k], y=qy[k];
            w[index]=table==0 ? -(a[0]*x+a[2]*y) : table==1 ? -(a[1]*x+a[3]*y)
                : scales[table-2]*(normals[2*(table-2)]*x+normals[2*(table-2)+1]*y);
        }
        __syncthreads();
        for(int i=tid; i<NEL; i+=blockDim.x) {
            double value=0.;
            for(int k=0; k<NEL; ++k)
                value+=d[i*NEL+k]*w[k]+d[NEL*NEL+i*NEL+k]*w[NEL+k]
                    +fm[i*NEL+k]*w[2*NEL+k]+fm[(NEL+i)*NEL+k]*w[3*NEL+k]+fm[(2*NEL+i)*NEL+k]*w[4*NEL+k];
            u[i]+=ji*value;
        }
        __syncthreads();
#else
        for(int index=tid; index<NEL*BATCH; index+=blockDim.x) {
            const int i=index/BATCH, b=index%BATCH;
            if(b>=width) continue;
            double value=0.;
            for(int k=0; k<NEL; ++k) {
                const double cx=RETAIN ? mx[i*NEL+k] : poisson_coupling(i,k,0,d,fm,a,scales,normals);
                const double cy=RETAIN ? my[i*NEL+k] : poisson_coupling(i,k,1,d,fm,a,scales,normals);
                value+=cx*qx[k*BATCH+b]+cy*qy[k*BATCH+b];
            }
            u[index]+=ji*value;
        }
        __syncthreads();
#endif
        }
        EXP_CLOCK(4)
        EXP_STOP_AT(5)
#endif
#if INVERSE
        // Experimental, not promoted. Stored local inverse: one parallel matrix-vector product
        // replaces both sweeps.
        for(int i=tid; i<NEL; i+=blockDim.x) {
            const double* __restrict__ row=factor_view+i*NEL;
            double v=0.;
            for(int k=0; k<NEL; ++k) v+=row[k]*u[k*BATCH];
            inverse_diagonal[i]=v;
        }
        __syncthreads();
        for(int i=tid; i<NEL; i+=blockDim.x) u[i*BATCH]=inverse_diagonal[i];
        __syncthreads();
#elif WARP_SOLVE
        solve_lu_columns_warp(factor_view,permuted_rows,inverse_diagonal,u,width);
#elif EXP_CHOLESKY_WARP
        // Experimental, not promoted: warp solve with the Cholesky factor read in place.
        cholesky_solve_warp(factor_view,inverse_diagonal,u);
#elif !(PHASE == 2 && RESPONSE)
        for(int b=tid; b<width; b+=blockDim.x) {
#if !CHOLESKY
            for(int k=0; k<NEL; ++k) {
                const int p=pivots[k];
                const double tmp=u[k*BATCH+b];
                u[k*BATCH+b]=u[p*BATCH+b]; u[p*BATCH+b]=tmp;
            }
#endif
            for(int i=0; i<NEL; ++i) {
                double v=u[i*BATCH+b];
                for(int j=0; j<i; ++j) v-=s[i*NEL+j]*u[j*BATCH+b];
                u[i*BATCH+b]=CHOLESKY ? v/s[i*NEL+i] : v;
            }
            for(int i=NEL-1; i>=0; --i) {
                double v=u[i*BATCH+b];
                for(int j=i+1; j<NEL; ++j)
                    v-=(CHOLESKY ? s[j*NEL+i] : s[i*NEL+j])*u[j*BATCH+b];
                u[i*BATCH+b]=v/s[i*NEL+i];
            }
        }
        __syncthreads();
#endif
        EXP_CLOCK(5)
        EXP_STOP_AT(6)
#if PHASE == 1 && RESPONSE
        for(int i=tid; i<NEL; i+=blockDim.x) {
            if(!sample) source_solutions[e*NEL+i]=u[i*BATCH];
            double v=0.;
            if(has_boundary) for(int f=0; f<3; ++f)
                for(int t=0; t<NTR; ++t)
                    v+=responses[(e*NEL+i)*(3*NTR)+f*NTR+t]*trace[edges[3*e+f]*NTR+t];
            u[i*BATCH]+=v;
        }
        __syncthreads();
#endif
        for(int index=tid; index<NEL*BATCH; index+=blockDim.x) {
            const int i=index/BATCH, b=index%BATCH, column=first+b;
            if(b>=width) continue;
            double x=0.,y=0.;
#if EXP_CONTRACT_Q
            // Experimental, no gain: contract u with the two reference tables first.
            double r0=0.,r1=0.;
            for(int j=0; j<NEL; ++j) {
                r0+=md[i*NEL+j]*u[j*BATCH+b];
                r1+=md[NEL*NEL+i*NEL+j]*u[j*BATCH+b];
            }
            x=a[0]*r0+a[1]*r1;
            y=a[2]*r0+a[3]*r1;
#else
            for(int j=0; j<NEL; ++j) {
                x+=(a[0]*md[i*NEL+j]+a[1]*md[NEL*NEL+i*NEL+j])*u[j*BATCH+b];
                y+=(a[2]*md[i*NEL+j]+a[3]*md[NEL*NEL+i*NEL+j])*u[j*BATCH+b];
            }
#endif
            qx[index]=ji*(x-qx[index]); qy[index]=ji*(y-qy[index]);
#if PHASE == 0 && RESPONSE
            if(!sample) {
                if(column<3*NTR) responses[(e*NEL+i)*(3*NTR)+column]=u[index];
                else source_solutions[e*NEL+i]=u[index];
            }
#endif
            if(sample) {
                const long long base=(slot*total_columns+column)*(3*NEL+3*NTR);
                validation[base+i]=u[index];
                validation[base+NEL+i]=qx[index];
                validation[base+2*NEL+i]=qy[index];
            }
#if PHASE == 2
            else {
                local[e*3*NEL+i]=u[index];
                local[e*3*NEL+NEL+i]=qx[index];
                local[e*3*NEL+2*NEL+i]=qy[index];
            }
#endif
        }
        __syncthreads();
        EXP_CLOCK(6)
        EXP_STOP_AT(7)
        for(int index=tid; index<3*NTR*BATCH; index+=blockDim.x) {
            const int task=index/BATCH, b=index%BATCH;
            if(b>=width) continue;
            const int f=task/NTR, t=task%NTR;
            const long long of=oriented[3*e+f];
            double value=0.;
#if PHASE != 2
            for(int i=0; i<NEL; ++i)
                value+=scales[f]*lifts[(of*NTR+t)*NEL+i]*(tau*u[i*BATCH+b]
                    +normals[2*f]*qx[i*BATCH+b]+normals[2*f+1]*qy[i*BATCH+b]);
#endif
            flux[index]=value;
            if(sample) validation[(slot*total_columns+first+b)*(3*NEL+3*NTR)+3*NEL+task]=value;
        }
        __syncthreads();
        EXP_CLOCK(7)
        EXP_STOP_AT(8)
#if PHASE != 2
        if(!sample) for(int task=tid; task<3*NTR; task+=blockDim.x) {
            const int rf=task/NTR, r=task%NTR;
            const long long re=solve_edges[edges[3*e+rf]];
            if(re<0) continue;
#if PHASE == 1
            atomicAdd(rhs+re*NTR+r,flux[task*BATCH]);
#else
            const long long side=side_index[3*e+rf];
            for(int b=0; b<width; ++b) {
                const int column=first+b;
                if(column==3*NTR) {
                    atomicAdd(rhs+re*NTR+r,flux[task*BATCH+b]);
                    continue;
                }
                const int cf=column/NTR, c=column%NTR;
                const long long edge=edges[3*e+cf], ce=solve_edges[edge];
                double value=-flux[task*BATCH+b];
                if(rf==cf) value+=tau*scales[rf]*edge_mass[r*NTR+c];
                if(ce<0) atomicAdd(rhs+re*NTR+r,-value*trace[edge*NTR+c]);
                else {
#if FORMAT == 0
                    int preceding=0;
                    for(int f=0; f<cf; ++f) preceding+=solve_edges[edges[3*e+f]]>=0;
                    const long long out=offsets[side]+(preceding*NTR+r)*NTR+c;
                    rows[out]=re*NTR+r; cols[out]=ce*NTR+c; data[out]=value;
#elif FORMAT == 1
                    const long long out=indptr[re*NTR+r]+side_positions[side*3+cf]*NTR+c;
                    atomicAdd(data+out,value);
#else
                    const long long out=((long long)indptr[re]+side_positions[side*3+cf])*NTR*NTR+r*NTR+c;
                    atomicAdd(data+out,value);
#endif
                }
            }
#endif
        }
#endif
        __syncthreads();
        EXP_CLOCK(8)
    }
}
'''
