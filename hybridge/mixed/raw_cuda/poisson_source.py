"""Compact scalar-Schur Poisson kernels; all local rows stride the block.

Reference contractions remove the directional dense matrix intermediates.
RETAIN keeps the two coupling matrices when the device can profit from them.
PHASE=0 assembles directly into COO/CSR/BSR; 1 refreshes the reduced RHS;
2 reconstructs the full mixed field. SAMPLE writes isolated element responses.
"""
from hybridge.hdg.cuda.raw_source import checked_warp_lu_source


SOURCE = checked_warp_lu_source('poisson_lu', 'NEL') + r'''
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
    int* pivots=reinterpret_cast<int*>(flux+3*NTR*BATCH);
    int* status=pivots+NEL;
    if(tid==0) *status=0;
    __syncthreads();

#if !(PHASE == 2 && RESPONSE)
    for(int ij=tid; ij<NEL*NEL; ij+=blockDim.x) {
        const int i=ij/NEL, j=ij%NEL;
#if USE_FACTORS
        s[ij]=factors[e*NEL*NEL+ij];
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
#if USE_FACTORS
    for(int i=tid; i<NEL; i+=blockDim.x) pivots[i]=all_pivots[e*NEL+i];
    __syncthreads();
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
            for(int f=0; f<3; ++f) {
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
#if !RESPONSE || PHASE == 0
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
#if !(PHASE == 2 && RESPONSE)
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
#if PHASE == 1 && RESPONSE
        for(int i=tid; i<NEL; i+=blockDim.x) {
            if(!sample) source_solutions[e*NEL+i]=u[i*BATCH];
            double v=0.;
            for(int f=0; f<3; ++f)
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
            for(int j=0; j<NEL; ++j) {
                x+=(a[0]*md[i*NEL+j]+a[1]*md[NEL*NEL+i*NEL+j])*u[j*BATCH+b];
                y+=(a[2]*md[i*NEL+j]+a[3]*md[NEL*NEL+i*NEL+j])*u[j*BATCH+b];
            }
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
    }
}
'''
