"""Cached, device-resident RT and L2-closest flux recovery.

One block per triangle applies reference moment lifts. L2-closest additionally
solves a k-by-k system using cached metric Cholesky factors, not the full
constraint system. Evolving fields and traces never pass through the host.
"""
from dataclasses import dataclass
import numpy as np

from hybridge.runtime.precision import REAL_DTYPE, real_raw_kernel
from hybridge.mixed.postprocess.flux_recovery import build_flux_recovery_reference
from hybridge.core.space import VectorDGField
from hybridge.core.device import as_cupy_space, field_from_cupy_coefficients
from hybridge.runtime.optional import require_cupy

_KERNELS = {}
_SOURCE = r'''
__device__ void metric(const double *a, double *g) {
    g[0] = a[0]*a[0] + a[2]*a[2];
    g[1] = a[0]*a[1] + a[2]*a[3];
    g[2] = a[1]*a[1] + a[3]*a[3];
    const double scale = g[0]+g[2];
    for (int j=0; j<3; ++j) g[j] /= scale;
}
extern "C" __global__ void factor_flux_metric(
        int count, const double *affine, const double *gram,
        double *chol, int *status) {
    const int k = blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=count) return;
    double g[3]; metric(affine+4*k, g);
    double *l = chol+k*NN*NN;
    status[k]=0;
    for (int i=0; i<NN; ++i) {
        for (int j=0; j<=i; ++j) {
            double value=0.;
            for (int c=0; c<3; ++c) value+=g[c]*gram[(c*NN+i)*NN+j];
            for (int r=0; r<j; ++r) value-=l[i*NN+r]*l[j*NN+r];
            if (i==j) {
                if (!(value>0.) || !isfinite(value)) {status[k]=i+1; return;}
                l[i*NN+j]=sqrt(value);
            } else l[i*NN+j]=value/l[j*NN+j];
        }
    }
}
extern "C" __global__ void recover_flux(
        int count, const double *local, const double *trace,
        const long long *edges, const bool *orientation,
        const double *affine, const double *jacobian,
        const double *face_jacobian, double tau, const double *tau_faces, int scalar_tau,
        const double *embedding, const double *face_moments,
        const double *trace_moments, const double *lift,
        const double *nullspace, const double *cross,
        const double *chol, double *output) {
    const int k=blockIdx.x, tid=threadIdx.x;
    if (k>=count) return;
    __shared__ double gap[NF], alpha[NN+1];
    const double *u=local+k*3*NB;
    const double *a=affine+4*k;
    for (int row=tid; row<NF; row+=blockDim.x) {
        const int face=row/NE, moment=row%NE;
        double value=0.;
        for (int j=0; j<NB; ++j) value+=face_moments[row*NB+j]*u[j];
        const bool positive=orientation[k*3+face];
        for (int j=0; j<NE; ++j) {
            const int index=(!positive && !MODAL)?NE-1-j:j;
            const double sign=(!positive && MODAL && (j%2))?-1.:1.;
            value-=trace_moments[moment*NE+j]*sign*trace[edges[k*3+face]*NE+index];
        }
        const double tau_face=scalar_tau?tau:tau_faces[k*3+face];
        gap[row]=face_jacobian[k*3+face]*tau_face*value;
    }
    __syncthreads();
    for (int i=tid; i<NN; i+=blockDim.x) {
        double g[3]; metric(a,g);
        double value=0.;
        for (int j=0; j<NF; ++j) {
            double coefficient=0.;
            for (int c=0; c<3; ++c) coefficient+=g[c]*cross[(c*NN+i)*NF+j];
            value-=coefficient*gap[j];
        }
        alpha[i]=value;
    }
    __syncthreads();
    if (tid==0 && NN) {
        const double *l=chol+k*NN*NN;
        for (int i=0; i<NN; ++i) {
            for (int j=0; j<i; ++j) alpha[i]-=l[i*NN+j]*alpha[j];
            alpha[i]/=l[i*NN+i];
        }
        for (int i=NN-1; i>=0; --i) {
            for (int j=i+1; j<NN; ++j) alpha[i]-=l[j*NN+i]*alpha[j];
            alpha[i]/=l[i*NN+i];
        }
    }
    __syncthreads();
    for (int flat=tid; flat<2*NP; flat+=blockDim.x) {
        const int c=flat/NP, i=flat%NP;
        double x=0., y=0., raw=0.;
        for (int j=0; j<NF; ++j) {
            x+=lift[i*NF+j]*gap[j];
            y+=lift[(NP+i)*NF+j]*gap[j];
        }
        for (int j=0; j<NN; ++j) {
            x+=nullspace[i*NN+j]*alpha[j];
            y+=nullspace[(NP+i)*NN+j]*alpha[j];
        }
        if (HIERARCHICAL) {if (i<NB) raw=u[(c+1)*NB+i];}
        else for (int j=0; j<NB; ++j) raw+=embedding[i*NB+j]*u[(c+1)*NB+j];
        output[(c*count+k)*NP+i]=raw+(a[2*c]*x+a[2*c+1]*y)/jacobian[k];
    }
}
'''


@dataclass
class FluxRecoveryDeviceCache:
    """Reference lifts and geometry factors independent of the current tau."""

    key: tuple
    reference: object
    cspace: object
    arrays: tuple
    cholesky: object
    kernel: object

    def is_compatible(self, space, trace_space, flux_space, *, device=None):
        """Check space, trace, recovery variant, and active-device ownership."""
        if device is None:
            device = int(require_cupy().cuda.runtime.getDevice())
        return self.key == (id(space), id(trace_space), flux_space, int(device))


def recover_diffusion_flux_raw_cuda(local_unknowns, trace, space, trace_space,
                                     stabilization, flux_space, *, cache=None):
    """Return a device-backed recovered vector field and its reusable cache.

    Scalar stabilization may change between calls (including retries) without
    rebuilding geometry data. Element/face stabilization arrays are supported
    as dynamic data, with the same normalization as the host postprocessor.
    """
    if flux_space not in {'RT_projection', 'l2_closest'}:
        raise ValueError('Expected RT_projection or l2_closest')
    if trace_space.kind not in {'legendre-modal', 'legacy-lagrange', 'bernstein'}:
        raise NotImplementedError('Unsupported flux recovery trace basis')
    cp = require_cupy()
    device = int(cp.cuda.runtime.getDevice())
    key = (id(space), id(trace_space), flux_space, device)
    if cache is None or not cache.is_compatible(space, trace_space, flux_space, device=device):
        ref = build_flux_recovery_reference(space, trace_space, l2_closest=flux_space=='l2_closest')
        cspace = as_cupy_space(space, device=device)
        n, b, e = ref.post_space.el_dof, space.el_dof, trace_space.edg_dof
        m = ref.nullspace.shape[1]
        modal = int(trace_space.kind=='legendre-modal')
        hierarchical = int(np.allclose(ref.embedding, np.eye(n,b), rtol=0., atol=1000*np.finfo(REAL_DTYPE).eps))
        kernel_key = (n,b,e,m,modal,hierarchical)
        if kernel_key not in _KERNELS:
            defines = ''.join(f'#define {name} {value}\n' for name,value in zip(
                ('NP','NB','NE','NN','MODAL','HIERARCHICAL','NF'), (*kernel_key,3*e)))
            source = defines+_SOURCE
            _KERNELS[kernel_key] = (real_raw_kernel(source,'recover_flux'), real_raw_kernel(source,'factor_flux_metric'))
        kernel, factor = _KERNELS[kernel_key]
        arrays = tuple(cp.asarray(value, dtype=REAL_DTYPE) for value in (
            ref.embedding,ref.face_moments,ref.trace_moments,ref.lift,ref.nullspace,ref.cross))
        chol = cp.empty((space.mesh.num_tri,m,m),dtype=REAL_DTYPE)
        if m:
            status = cp.empty(space.mesh.num_tri,dtype=cp.int32)
            factor(((space.mesh.num_tri+127)//128,), (128,), (
                np.int32(space.mesh.num_tri),cspace.mesh.aff_mats,
                cp.asarray(ref.gram),chol,status))
            if bool(cp.any(status).item()):
                raise ArithmeticError('Non-positive flux recovery metric factor')
        cache = FluxRecoveryDeviceCache(key,ref,cspace,arrays,chol,kernel)
    local = cp.ascontiguousarray(cp.asarray(local_unknowns,dtype=REAL_DTYPE))
    trace = cp.ascontiguousarray(cp.asarray(trace,dtype=REAL_DTYPE))
    if local.shape != (space.mesh.num_tri,3*space.el_dof):
        raise ValueError('Invalid local HDG unknown shape for flux recovery')
    if trace.shape != (space.mesh.num_edg*trace_space.edg_dof,):
        raise ValueError('Invalid full HDG trace shape for flux recovery')
    scalar_tau = int(np.isscalar(stabilization))
    if scalar_tau:
        tau_value, tau_faces = REAL_DTYPE(stabilization), cache.arrays[0]
    else:
        from hybridge.mixed.coefficients import normalize_diffusion_stabilization
        tau_value = REAL_DTYPE(0.)
        tau_faces = cp.ascontiguousarray(cp.asarray(normalize_diffusion_stabilization(stabilization,space),dtype=REAL_DTYPE))
    mesh, post = cache.cspace.mesh, cache.reference.post_space
    output = cp.empty((2,space.mesh.num_tri,post.el_dof),dtype=REAL_DTYPE)
    cache.kernel((space.mesh.num_tri,), (64,), (
        np.int32(space.mesh.num_tri),local,trace,mesh.loc2glob_edge,mesh.orientations,
        mesh.aff_mats,mesh.aff_jacs,mesh.jacs_el_fc,tau_value,tau_faces,np.int32(scalar_tau),
        *cache.arrays,cache.cholesky,output))
    components = tuple(field_from_cupy_coefficients(post,output[c],device=device,name=f'q_h_star_{c}') for c in range(2))
    return VectorDGField(components,name='q_h_star'), cache
