"""Fused device assembly of the tensor ADR primal Neumann recovery.

Matrix-entry tiles assemble each element's matrix and RHS. The dense pivoted solve is
left to CuPy/cuBLAS; no coefficients or solutions leave the device. The
quadrature contractions avoid the dozens of temporary arrays and launches of
the independent einsum reference in advection_diffusion_reaction_cupy.
"""
from functools import lru_cache

from hdgfem.runtime.optional import require_cupy
from hdgfem.core.device import as_cupy_space, as_cupy_coefficients


_SOURCE = r'''
extern "C" __global__ void assemble(
 const double* jac, const double* invmap, const double* fj, const double* normal,
 const double* w, const double* phi, const double* grad,
 const double* fw, const double* fp, const double* psi,
 const double* beta, const double* bf, const double* tau, const double* kinv,
 const double* flux, const double* base, const double* mean,
 double* matrix, double* rhs) {
 const int k=blockIdx.x;
 const double* im=invmap+4*k;
 for(int entry=blockIdx.y*blockDim.x+threadIdx.x;entry<N*N && entry<(blockIdx.y+1)*blockDim.x;entry+=blockDim.x) {
  const int row=entry/N,col=entry%N;
  double v=0.;
  if(row<3*D && col<3*D) {
   // One (i,j) contraction fills all nine mixed blocks. In particular,
   // the four inverse-tensor masses share phi_i*phi_j and its weight.
   if(row>=D || col>=D) continue;
   const int i=row,j=col;
   double m00=0.,m01=0.,m10=0.,m11=0.,dx=0.,dy=0.,adv=0.;
   for(int q=0;q<Q;++q) {
    const double b=jac[k]*w[q]*phi[q*D+j],mass=b*phi[q*D+i];
    const double gx=im[0]*grad[(q*D+i)*2]+im[1]*grad[(q*D+i)*2+1];
    const double gy=im[2]*grad[(q*D+i)*2]+im[3]*grad[(q*D+i)*2+1];
    m00+=mass*kinv[(k*Q+q)*4];m01+=mass*kinv[(k*Q+q)*4+1];
    m10+=mass*kinv[(k*Q+q)*4+2];m11+=mass*kinv[(k*Q+q)*4+3];
    dx+=b*gx;dy+=b*gy;
    adv-=b*(gx*beta[(k*Q+q)*2]+gy*beta[(k*Q+q)*2+1]);
   }
   double nx=-dx,ny=-dy;
   for(int f=0;f<3;++f) for(int q=0;q<F;++q) {
    double weight=fj[k*3+f]*fw[q]*fp[(f*D+i)*F+q]*fp[(f*D+j)*F+q];
    adv+=weight*tau[(k*3+f)*F+q];
    nx+=weight*normal[(k*3+f)*2];ny+=weight*normal[(k*3+f)*2+1];
   }
   double* out=matrix+k*N*N;
   out[i*N+j]=-dx;out[i*N+D+j]=m00;out[i*N+2*D+j]=m01;
   out[(D+i)*N+j]=-dy;out[(D+i)*N+D+j]=m10;out[(D+i)*N+2*D+j]=m11;
   out[(2*D+i)*N+j]=adv;out[(2*D+i)*N+D+j]=nx;out[(2*D+i)*N+2*D+j]=ny;
   continue;
  } else if(row<3*D && col>=3*D && col<N-1) {
   const int block=row/D,i=row%D,f=(col-3*D)/E,a=(col-3*D)%E;
   for(int q=0;q<F;++q) {
    double bn=bf[((k*3+f)*F+q)*2]*normal[(k*3+f)*2]
             +bf[((k*3+f)*F+q)*2+1]*normal[(k*3+f)*2+1];
    v+=fj[k*3+f]*fw[q]*fp[(f*D+i)*F+q]*psi[a*F+q]
       *(block<2?normal[(k*3+f)*2+block]:bn-tau[(k*3+f)*F+q]);
   }
  } else if(row>=3*D && row<N-1 && col<N-1) {
   const int f=(row-3*D)/E,a=(row-3*D)%E;
   if(col<3*D || (col-3*D)/E==f) for(int q=0;q<F;++q) {
    double factor;
    if(col<D) factor=tau[(k*3+f)*F+q]*fp[(f*D+col)*F+q];
    else if(col<3*D) factor=normal[(k*3+f)*2+col/D-1]*fp[(f*D+col%D)*F+q];
    else {
     double bn=bf[((k*3+f)*F+q)*2]*normal[(k*3+f)*2]
              +bf[((k*3+f)*F+q)*2+1]*normal[(k*3+f)*2+1];
     factor=(bn-tau[(k*3+f)*F+q])*psi[(col-3*D)%E*F+q];
    }
    v+=fj[k*3+f]*fw[q]*psi[a*F+q]*factor;
   }
  } else if((row==N-1 && col<D) || (row>=2*D && row<3*D && col==N-1)) {
   const int i=row==N-1?col:row-2*D;
   for(int q=0;q<Q;++q) v+=jac[k]*w[q]*phi[q*D+i];
  }
  matrix[k*N*N+entry]=v;
 }
 for(int row=threadIdx.x;blockIdx.y==0 && row<N;row+=blockDim.x) {
  double v=0.;
  if(row>=2*D && row<3*D) {
   const int i=row-2*D;
   for(int q=0;q<Q;++q) {
    double fx=0.,fy=0.;
    for(int j=0;j<D;++j) {fx+=flux[(k*2)*D+j]*phi[q*D+j];fy+=flux[(k*2+1)*D+j]*phi[q*D+j];}
    double gx=im[0]*grad[(q*D+i)*2]+im[1]*grad[(q*D+i)*2+1];
    double gy=im[2]*grad[(q*D+i)*2]+im[3]*grad[(q*D+i)*2+1];
    v-=jac[k]*w[q]*(gx*fx+gy*fy);
   }
  }
  if(row>=2*D && row<N-1) {
   const int first=row<3*D?0:(row-3*D)/E,last=row<3*D?3:first+1;
   for(int f=first;f<last;++f) for(int q=0;q<F;++q) {
    double fn=0.;
    for(int j=0;j<D;++j) fn+=(normal[(k*3+f)*2]*flux[k*2*D+j]
       +normal[(k*3+f)*2+1]*flux[(k*2+1)*D+j])*fp[(f*D+j)*F+q];
    double test=row<3*D?fp[(f*D+row-2*D)*F+q]:psi[(row-3*D)%E*F+q];
    v+=fj[k*3+f]*fw[q]*fn*test;
   }
  }
  if(row==N-1) for(int i=0;i<B;++i) v+=jac[k]*base[k*3*B+i]*mean[i];
  rhs[k*N+row]=v;
 }
}
'''


@lru_cache(maxsize=None)
def _kernel(d, e, nq, nfq, base_dof):
    """Cache a shape-specialized FP64 recovery assembly (matching both references)."""
    cp = require_cupy()
    definitions = dict(D=d, E=e, Q=nq, F=nfq, B=base_dof, N=3*d+3*e+1)
    source = '\n'.join(f'#define {key} {value}' for key, value in definitions.items())+'\n'+_SOURCE
    return cp.RawKernel(source, 'assemble')


def primal_system_raw_cuda(local_unknowns, total_flux, space, cache, samples, inverse):
    """Assemble the constrained system in matrix-entry tiles without intermediate contractions."""
    cp = require_cupy()
    post = cache.post_space
    q = post.quad_data
    d, e = post.el_dof, q.edg_dof
    rows, n = 3*d+3*e+1, space.mesh.num_tri
    # Static uploads are scoped to the existing recovery cache and CUDA device.
    device = cp.cuda.runtime.getDevice()
    tables = getattr(cache, '_adr_primal_device_tables', None)
    if tables is None:
        tables = cache._adr_primal_device_tables = {}
    if device not in tables:
        tables[device] = tuple(cp.asarray(value, dtype=cp.float64, order="C") for value in (
            space.mesh.aff_jacs, space.mesh.inv_aff_mats_t, space.mesh.jacs_el_fc,
            space.mesh.normals, q.Krf_w, q.phi, q.gphi, q.weights_JGL,
            q.bas_of_bd_quads, q.bas1d_of_ref_edg_qds))
    flux = cp.stack([as_cupy_coefficients(f, as_cupy_space(f.space)) for f in total_flux.components], axis=1)
    arrays = tuple(cp.asarray(value, dtype=cp.float64, order="C") for value in (
        *samples, inverse, flux, local_unknowns, cache.mean_base))
    matrix, rhs = cp.empty((n, rows, rows)), cp.empty((n, rows))
    _kernel(d, e, q.Krf_w.size, q.weights_JGL.size, space.el_dof)(
        (n, (rows*rows+127)//128), (128,), (*tables[device], *arrays, matrix, rhs))
    return matrix, rhs
