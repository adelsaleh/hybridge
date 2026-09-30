"""Raw CUDA stationary tensor ADR assembly, solve, and reconstruction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from hdgfem.assembly import hdg
from hdgfem.assembly.advection_diffusion_reaction import ADRPreparedData
from hdgfem.core.space import DGSpace, DGTraceSpace
from hdgfem.backends.cupy import as_cupy_space, require_cupy
from hdgfem.backends.numba import _boundary_reduction_maps, _interior_side_index, _trace_orientation_mode


_RAW_ADR_TEMPLATE = r'''
#define NEL {nel}
#define NTR {ntr}
#define NQ {nq}
#define ORIENTATION_MODE {orientation_mode}

__device__ __forceinline__ int local_dof(bool positive, int dof) {{
#if ORIENTATION_MODE == 1
    return dof;
#else
    return positive ? dof : (NTR - 1 - dof);
#endif
}}

__device__ __forceinline__ double orientation_sign(bool positive, int dof) {{
#if ORIENTATION_MODE == 1
    return ((!positive) && ((dof & 1) == 1)) ? -1.0 : 1.0;
#else
    return 1.0;
#endif
}}

__device__ __forceinline__ int factor(double* a, int* pivots) {{
    for (int k = 0; k < NEL; ++k) {{
        int pivot = k;
        double largest = fabs(a[k*NEL+k]);
        for (int i = k+1; i < NEL; ++i) {{
            double value = fabs(a[i*NEL+k]);
            if (value > largest) {{ largest = value; pivot = i; }}
        }}
        pivots[k] = pivot;
        if (pivot != k) for (int j = 0; j < NEL; ++j) {{
            double tmp = a[k*NEL+j]; a[k*NEL+j] = a[pivot*NEL+j]; a[pivot*NEL+j] = tmp;
        }}
        double diagonal = a[k*NEL+k];
        if (!isfinite(diagonal) || fabs(diagonal) < 1.e-30) return k+1;
        for (int i = k+1; i < NEL; ++i) {{
            a[i*NEL+k] /= diagonal;
            double multiplier = a[i*NEL+k];
            for (int j = k+1; j < NEL; ++j) a[i*NEL+j] -= multiplier*a[k*NEL+j];
        }}
    }}
    return 0;
}}

__device__ __forceinline__ void solve(
        const double* lu, const int* pivots, const double* mass_inverse,
        const double* d0, const double* d1, const double* mn0, const double* mn1,
        double* rhs0, double* rhs1, double* rhs2, double* u, double* qx, double* qy,
        double* work0, double* work1, double jac_inv) {{
    for (int i=0;i<NEL;++i) {{
        double x=0.0,y=0.0;
        for (int k=0;k<NEL;++k) {{ x += mass_inverse[i*NEL+k]*rhs1[k]; y += mass_inverse[i*NEL+k]*rhs2[k]; }}
        work0[i]=x; work1[i]=y;
    }}
    for (int i=0;i<NEL;++i) {{
        double x=0.0,y=0.0;
        for (int k=0;k<NEL;++k) {{ x += mn0[i*NEL+k]*work0[k]; y += mn1[i*NEL+k]*work1[k]; }}
        u[i]=rhs0[i]+jac_inv*(x+y);
    }}
    for (int k=0;k<NEL;++k) if (pivots[k]!=k) {{ double tmp=u[k];u[k]=u[pivots[k]];u[pivots[k]]=tmp; }}
    for (int i=0;i<NEL;++i) for (int j=0;j<i;++j) u[i]-=lu[i*NEL+j]*u[j];
    for (int i=NEL-1;i>=0;--i) {{ for(int j=i+1;j<NEL;++j)u[i]-=lu[i*NEL+j]*u[j]; u[i]/=lu[i*NEL+i]; }}
    for (int i=0;i<NEL;++i) {{
        double x=-rhs1[i],y=-rhs2[i];
        for(int j=0;j<NEL;++j){{x+=d0[i*NEL+j]*u[j];y+=d1[i*NEL+j]*u[j];}}
        work0[i]=x;work1[i]=y;
    }}
    for (int i=0;i<NEL;++i) {{
        double x=0.0,y=0.0;
        for(int k=0;k<NEL;++k){{x+=mass_inverse[i*NEL+k]*work0[k];y+=mass_inverse[i*NEL+k]*work1[k];}}
        qx[i]=jac_inv*x;qy[i]=jac_inv*y;
    }}
}}

__device__ __forceinline__ double lift_value(
        const double* trace_lift, long long element, int face, int row_dof,
        const double* u, const double* qx, const double* qy) {{
    long long base=(((element*3+face)*NTR+row_dof)*3*NEL);
    double value=0.0;
    for(int i=0;i<NEL;++i){{
        value += trace_lift[base+i]*u[i];
        value += trace_lift[base+NEL+i]*qx[i];
        value += trace_lift[base+2*NEL+i]*qy[i];
    }}
    return value;
}}

__device__ __forceinline__ int build_factor(
        long long element, double diffusion, const double* aff_mats, const double* aff_jacs,
        const double* mass_inverse, const double* basis, const double* gradients,
        const double* weights, const double* reaction_values, const double* beta_values,
        const double* u_boundary_mass, const double* normal_mass_x, const double* normal_mass_y,
        const double* d0_reference, const double* d1_reference,
        double* schur,double* d0,double* d1,double* mn0,double* mn1,double* kd0,double* kd1,int* pivots) {{
    double a00=aff_mats[element*4],a01=aff_mats[element*4+1],a10=aff_mats[element*4+2],a11=aff_mats[element*4+3];
    double jac=aff_jacs[element],inv00=a11/jac,inv01=-a10/jac,inv10=-a01/jac,inv11=a00/jac;
    for(int i=0;i<NEL;++i)for(int j=0;j<NEL;++j){{
        int ij=i*NEL+j;
        d0[ij]=a11*d0_reference[ij]-a10*d1_reference[ij];
        d1[ij]=-a01*d0_reference[ij]+a00*d1_reference[ij];
        mn0[ij]=normal_mass_x[element*NEL*NEL+ij]-d0[ij];
        mn1[ij]=normal_mass_y[element*NEL*NEL+ij]-d1[ij];
        double reaction=0.0,advection=0.0;
        for(int q=0;q<NQ;++q){{
            double phii=basis[i*NQ+q],phij=basis[j*NQ+q];
            double gx=inv00*gradients[(0*NEL+i)*NQ+q]+inv01*gradients[(1*NEL+i)*NQ+q];
            double gy=inv10*gradients[(0*NEL+i)*NQ+q]+inv11*gradients[(1*NEL+i)*NQ+q];
            reaction += weights[q]*reaction_values[element*NQ+q]*phii*phij;
            advection += weights[q]*phij*(beta_values[(element*NQ+q)*2]*gx+beta_values[(element*NQ+q)*2+1]*gy);
        }}
        schur[ij]=u_boundary_mass[element*NEL*NEL+ij]+jac*(reaction-advection);
    }}
    for(int i=0;i<NEL;++i)for(int j=0;j<NEL;++j){{
        double x=0.0,y=0.0;for(int k=0;k<NEL;++k){{x+=mass_inverse[i*NEL+k]*d0[k*NEL+j];y+=mass_inverse[i*NEL+k]*d1[k*NEL+j];}}
        kd0[i*NEL+j]=x;kd1[i*NEL+j]=y;
    }}
    double jac_inv=diffusion/jac;
    for(int i=0;i<NEL;++i)for(int j=0;j<NEL;++j){{
        double x=0.0,y=0.0;for(int k=0;k<NEL;++k){{x+=mn0[i*NEL+k]*kd0[k*NEL+j];y+=mn1[i*NEL+k]*kd1[k*NEL+j];}}
        schur[i*NEL+j]+=jac_inv*(x+y);
    }}
    return factor(schur,pivots);
}}

extern "C" __global__ void assemble_adr_raw(
        long long* rows,long long* cols,double* data,double* rhs,
        const long long* loc2glob,const bool* orientations,const long long* side_index,
        const long long* edge_to_solve,const long long* side_offsets,
        const double* aff_mats,const double* aff_jacs,const double* mass_inverse,
        const double* basis,const double* gradients,const double* weights,
        const double* reaction_values,const double* beta_values,const double* u_boundary_mass,
        const double* normal_mass_x,const double* normal_mass_y,const double* d0_reference,
        const double* d1_reference,const double* element_boundary,const double* trace_lift,
        const double* gamma_mass,const double* source_rhs,const double* boundary_trace,
        long long num_elements,long long mass_offset,double diffusion,int* statuses) {{
    long long element=blockIdx.x;
    if(element>=num_elements||threadIdx.x!=0)return;
    double schur[NEL*NEL],d0[NEL*NEL],d1[NEL*NEL],mn0[NEL*NEL],mn1[NEL*NEL],kd0[NEL*NEL],kd1[NEL*NEL];
    double rhs0[NEL],rhs1[NEL],rhs2[NEL],u[NEL],qx[NEL],qy[NEL],work0[NEL],work1[NEL];int pivots[NEL];
    int failure=build_factor(element,diffusion,aff_mats,aff_jacs,mass_inverse,basis,gradients,weights,reaction_values,beta_values,u_boundary_mass,normal_mass_x,normal_mass_y,d0_reference,d1_reference,schur,d0,d1,mn0,mn1,kd0,kd1,pivots);
    if(failure) {{ statuses[element]=failure; return; }}
    double jac_inv=diffusion/aff_jacs[element];
    for(int i=0;i<NEL;++i){{rhs0[i]=source_rhs[element*NEL+i];rhs1[i]=0.0;rhs2[i]=0.0;}}
    solve(schur,pivots,mass_inverse,d0,d1,mn0,mn1,rhs0,rhs1,rhs2,u,qx,qy,work0,work1,jac_inv);
    for(int rf=0;rf<3;++rf){{
        long long sid=side_index[element*3+rf],redge=edge_to_solve[loc2glob[element*3+rf]];
        if(sid<0||redge<0)continue;
        for(int rd=0;rd<NTR;++rd)atomicAdd(&rhs[redge*NTR+rd],lift_value(trace_lift,element,rf,rd,u,qx,qy));
        long long mbase=mass_offset+sid*NTR*NTR;
        for(int i=0;i<NTR;++i)for(int j=0;j<NTR;++j){{long long out=mbase+i*NTR+j;rows[out]=redge*NTR+i;cols[out]=redge*NTR+j;data[out]=gamma_mass[sid*NTR*NTR+i*NTR+j];}}
    }}
    for(int cf=0;cf<3;++cf){{
        long long cedge=loc2glob[element*3+cf],credge=edge_to_solve[cedge];bool positive=orientations[element*3+cf];int blockpos=0;
        for(int f=0;f<cf;++f)if(edge_to_solve[loc2glob[element*3+f]]>=0)++blockpos;
        for(int cd=0;cd<NTR;++cd){{
            int ld=local_dof(positive,cd);double sign=orientation_sign(positive,cd);int column=cf*NTR+ld;
            for(int i=0;i<NEL;++i){{rhs0[i]=element_boundary[(element*3*NEL+i)*3*NTR+column];rhs1[i]=element_boundary[(element*3*NEL+NEL+i)*3*NTR+column];rhs2[i]=element_boundary[(element*3*NEL+2*NEL+i)*3*NTR+column];}}
            solve(schur,pivots,mass_inverse,d0,d1,mn0,mn1,rhs0,rhs1,rhs2,u,qx,qy,work0,work1,jac_inv);
            for(int rf=0;rf<3;++rf){{
                long long sid=side_index[element*3+rf],redge=edge_to_solve[loc2glob[element*3+rf]];if(sid<0||redge<0)continue;
                long long base=side_offsets[sid];
                for(int rd=0;rd<NTR;++rd){{double value=sign*lift_value(trace_lift,element,rf,rd,u,qx,qy);
                    if(credge>=0){{long long out=base+((long long)blockpos*NTR+rd)*NTR+cd;rows[out]=redge*NTR+rd;cols[out]=credge*NTR+cd;data[out]=-value;}}
                    else atomicAdd(&rhs[redge*NTR+rd],value*boundary_trace[cedge*NTR+cd]);
                }}
            }}
        }}
    }}
}}

extern "C" __global__ void reconstruct_adr_raw(
        double* unknowns,const double* trace,const long long* loc2glob,const bool* orientations,
        const double* aff_mats,const double* aff_jacs,const double* mass_inverse,const double* basis,
        const double* gradients,const double* weights,const double* reaction_values,const double* beta_values,
        const double* u_boundary_mass,const double* normal_mass_x,const double* normal_mass_y,
        const double* d0_reference,const double* d1_reference,const double* element_boundary,
        const double* source_rhs,long long num_elements,double diffusion,int* statuses) {{
    long long element=blockIdx.x;if(element>=num_elements||threadIdx.x!=0)return;
    double schur[NEL*NEL],d0[NEL*NEL],d1[NEL*NEL],mn0[NEL*NEL],mn1[NEL*NEL],kd0[NEL*NEL],kd1[NEL*NEL];
    double rhs0[NEL],rhs1[NEL],rhs2[NEL],u[NEL],qx[NEL],qy[NEL],work0[NEL],work1[NEL];int pivots[NEL];
    int failure=build_factor(element,diffusion,aff_mats,aff_jacs,mass_inverse,basis,gradients,weights,reaction_values,beta_values,u_boundary_mass,normal_mass_x,normal_mass_y,d0_reference,d1_reference,schur,d0,d1,mn0,mn1,kd0,kd1,pivots);
    if(failure) {{ statuses[element]=failure; return; }}
    for(int i=0;i<NEL;++i){{rhs0[i]=source_rhs[element*NEL+i];rhs1[i]=0.0;rhs2[i]=0.0;}}
    for(int f=0;f<3;++f){{long long edge=loc2glob[element*3+f];bool positive=orientations[element*3+f];for(int d=0;d<NTR;++d){{int ld=local_dof(positive,d);double value=orientation_sign(positive,d)*trace[edge*NTR+d];int col=f*NTR+ld;for(int i=0;i<NEL;++i){{rhs0[i]+=element_boundary[(element*3*NEL+i)*3*NTR+col]*value;rhs1[i]+=element_boundary[(element*3*NEL+NEL+i)*3*NTR+col]*value;rhs2[i]+=element_boundary[(element*3*NEL+2*NEL+i)*3*NTR+col]*value;}}}}}}
    solve(schur,pivots,mass_inverse,d0,d1,mn0,mn1,rhs0,rhs1,rhs2,u,qx,qy,work0,work1,diffusion/aff_jacs[element]);
    for(int i=0;i<NEL;++i){{unknowns[element*3*NEL+i]=u[i];unknowns[element*3*NEL+NEL+i]=qx[i];unknowns[element*3*NEL+2*NEL+i]=qy[i];}}
}}
'''


def _device_inputs(cp, prepared: ADRPreparedData, space: DGSpace):
    """Upload contiguous ADR coefficient and reference tables to the active device."""
    q = space.quad_data
    arrays = (
        space.mesh.aff_mats, space.mesh.aff_jacs, q.MKrf_inv, q.bas_of_quads,
        q.dbas_of_quads, q.Krf_w, prepared.reaction_values, prepared.beta_values,
        prepared.u_boundary_mass, prepared.normal_mass_x, prepared.normal_mass_y,
        prepared.d0_reference, prepared.d1_reference, prepared.element_boundary,
        prepared.trace_lift, prepared.interior_gamma_mass, prepared.source_rhs,
    )
    return tuple(cp.ascontiguousarray(cp.asarray(value)) for value in arrays)


@dataclass(frozen=True)
class RawADRTraceOperator:
    """Device trace operator and local data retained for ADR reconstruction."""

    assembly: Any
    module: Any
    device_inputs: tuple[Any, ...]
    boundary_trace: Any
    csr_pattern: Any = None
    diffusion_structure: dict[str, int] | None = None
    diffusion_kinds: Any = None
    reconstruction_data: Any = None
    mass_factors: Any = None  # (mass, pivots) of variable tensors when cached; reusable while kappa is fixed


def assemble_projected_adr_trace_operator_raw_cuda(
        prepared: ADRPreparedData, boundary_condition, space: DGSpace, *,
        diffusion=1.0, trace_space: DGTraceSpace,
        matrix_format="csr", block_size="auto", cache_local_factors="schur-lu+mass",
        prepared_diffusion=None, csr_pattern=None, mass_factors=None,
) -> RawADRTraceOperator:
    """Assemble FP64 tensor ADR in COO, direct CSR, or face-block BSR.

    Tensor assembly and reconstruction share the same local algebra;
    ``cache_local_factors`` ("none", "schur-lu", "schur-lu+mass") selects the
    local factors reconstruction reuses.
    ``block_size=1`` uses serial tensor solves in every matrix format; scalar
    CSR with dense preparation retains the original serial diagnostic kernel.
    Cooperative automatic sizing uses 32/64/128 threads for p<=2/4/6.
    """
    from hdgfem.backends.raw_cuda import resolve_raw_cuda_block_size
    from hdgfem.backends.adr_tensor_raw_cuda import assemble_tensor_operator
    if space.order > 6 or trace_space.kind not in {"legacy-lagrange", "legendre-modal"}:
        raise ValueError("raw CUDA tensor ADR supports p=0--6 and legacy-lagrange/legendre-modal traces")
    if trace_space.space is not space:
        raise ValueError("trace_space must belong to the ADR space")
    matrix_format = str(matrix_format).lower()
    if matrix_format not in {"coo", "csr", "bsr"}:
        raise ValueError("matrix_format must be 'coo', 'csr', or 'bsr'")
    block_size = resolve_raw_cuda_block_size(block_size, equation="diffusion-reaction", order=space.order)
    if (block_size == 1 and matrix_format == "csr" and prepared.element_boundary is not None
            and np.isscalar(diffusion)):
        return _assemble_scalar_serial_operator(prepared, boundary_condition, space,
                                                 diffusion=diffusion, trace_space=trace_space)
    return assemble_tensor_operator(prepared, boundary_condition, space, diffusion=diffusion,
                                    trace_space=trace_space, matrix_format=matrix_format,
                                    block_size=block_size, cache_local_factors=cache_local_factors,
                                    prepared_diffusion=prepared_diffusion, csr_pattern=csr_pattern,
                                    mass_factors=mass_factors)


def reconstruct_projected_adr_local_unknowns_raw_cuda(operator, trace, *, block_size="auto"):
    """Return device mixed local unknowns and timings from a full trace."""
    from hdgfem.backends.adr_tensor_raw_cuda import reconstruct_tensor_operator
    return reconstruct_tensor_operator(operator, trace, block_size=block_size)


def _assemble_scalar_serial_operator(
        prepared: ADRPreparedData, boundary_condition, space: DGSpace, *,
        diffusion: float = 1.0, trace_space: DGTraceSpace,
) -> RawADRTraceOperator:
    """Assemble raw ADR COO then device CSR without a global solve or recovery."""
    if not np.isscalar(diffusion) or not np.isfinite(float(diffusion)) or float(diffusion) <= 0.0:
        raise NotImplementedError("raw CUDA ADR currently requires positive constant scalar diffusion")
    cp = require_cupy()
    from cupyx.scipy import sparse
    from hdgfem.backends.advection_cuda import CudaAdvectionAssembly, as_cupy_trace_space
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(trace_space)
    mesh = space.mesh
    ntr = trace_space.edg_dof
    boundary_host = hdg.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_space)
    edge_to_solve, free_edges, _reduction = _boundary_reduction_maps(space, boundary_host, None, trace_space=trace_space)
    side_index = _interior_side_index(space)
    valid_elements = mesh.interior_elements
    counts = np.count_nonzero(edge_to_solve[mesh.loc2glob_edge[valid_elements]] >= 0, axis=1).astype(np.int64)
    offsets = np.empty(counts.size + 1, dtype=np.int64); offsets[0]=0
    np.cumsum(counts*ntr*ntr,out=offsets[1:])
    nflux=int(offsets[-1]); nnz=nflux+valid_elements.size*ntr*ntr
    rows=cp.empty(nnz,dtype=cp.int64);cols=cp.empty(nnz,dtype=cp.int64);data=cp.empty(nnz,dtype=cp.float64)
    rhs=cp.zeros(free_edges.size*ntr,dtype=cp.float64)
    boundary=cp.asarray(boundary_host,dtype=cp.float64)
    for name in ("source_rhs", "reaction_values", "beta_values", "tau_total", "gamma"):
        if not np.all(np.isfinite(getattr(prepared, name))):
            raise ValueError(f"ADR {name} samples must be finite")
    statuses=cp.zeros(mesh.num_tri,dtype=cp.int32)
    device_inputs=_device_inputs(cp,prepared,space)
    source_code=_RAW_ADR_TEMPLATE.format(nel=space.el_dof,ntr=ntr,nq=space.quad_data.Krf_w.size,orientation_mode=_trace_orientation_mode(trace_space))
    compile_start=time.perf_counter(); module=cp.RawModule(code=source_code,options=("--std=c++11",)); kernel=module.get_function("assemble_adr_raw");compile_seconds=time.perf_counter()-compile_start
    args=(rows,cols,data,rhs,cp.asarray(mesh.loc2glob_edge),cp.asarray(mesh.orientations),cp.asarray(side_index),cp.asarray(edge_to_solve),cp.asarray(offsets),*device_inputs[:15],device_inputs[15],device_inputs[16],boundary.reshape(-1),np.int64(mesh.num_tri),np.int64(nflux),np.float64(diffusion),statuses)
    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    start=time.perf_counter()
    start_event.record()
    kernel((mesh.num_tri,),(1,),args)
    end_event.record()
    end_event.synchronize()
    assembly_seconds=time.perf_counter()-start
    device_seconds=cp.cuda.get_elapsed_time(start_event, end_event)/1000.0
    if bool(cp.any(statuses)):
        failures=cp.asnumpy(statuses)
        raise np.linalg.LinAlgError(f"ADR scalar Schur factorization failed in element {int(np.flatnonzero(failures)[0])}")
    if not bool(cp.all(cp.isfinite(data))) or not bool(cp.all(cp.isfinite(rhs))):
        raise np.linalg.LinAlgError("ADR serial condensation produced nonfinite values")
    conversion_start=time.perf_counter()
    coo=sparse.coo_matrix((data,(rows,cols)),shape=(rhs.size,rhs.size));csr=coo.tocsr();csr.sum_duplicates()
    cp.cuda.get_current_stream().synchronize()
    conversion_seconds=time.perf_counter()-conversion_start
    assembly=CudaAdvectionAssembly(None,None,csr.data,rhs,None,None,device_inputs[16],boundary[mesh.bnd_edges_inds],None,cspace,trace_ref,indptr=csr.indptr,indices=csr.indices,matrix_format="csr",timings={"raw.kernel.jit":compile_seconds,"raw.kernel.wall":assembly_seconds,"raw.kernel.device":device_seconds,"raw.coo_to_csr.wall":conversion_seconds})
    from hdgfem.assembly.diffusion_coefficients import DIFFUSION_KINDS
    counts={name: mesh.num_tri if i==0 else 0 for i,name in enumerate(DIFFUSION_KINDS)}
    return RawADRTraceOperator(assembly, module, device_inputs, boundary,
                               diffusion_structure=counts, diffusion_kinds=np.zeros(mesh.num_tri,dtype=np.int64))


def _static_cache_key(space, trace_space, options):
    """Identity of the time-independent diffusion data and sparsity of one solver setup."""
    return (id(space), trace_space.kind, id(options.diffusion), id(options.diffusion_stabilization),
            float(options.diffusion_penalty_constant), str(options.raw_matrix_format))


def _amgx_cache_key(config, options, size):
    """Identity of a persistent AMGX solver: config contents, tolerances, format and size."""
    import json
    return (json.dumps(config, sort_keys=True, default=str), float(options.solver_rtol),
            None if options.maxiter is None else int(options.maxiter), str(options.raw_matrix_format),
            int(size), str(options.scale_system))


def close_raw_adr_cache(cache) -> None:
    """Release persistent AMGX/PARDISO solvers and cached data of an ADR solver cache."""
    if not cache:
        return
    for state in cache.get("amgx", {}).values():
        state["solver"].close(suppress_errors=True)
    if cache.get("pardiso") is not None:
        cache["pardiso"].close()
    cache.clear()


def assemble_projected_adr_trace_system_eliminated_raw_cuda(
        source, beta, reaction, boundary_condition, space: DGSpace, *,
        options, trace_space: DGTraceSpace, total_start: float, cache: dict | None = None,
):
    """Assemble, solve, and reconstruct an elliptic tensor ADR system on CUDA.

    Assembly starts by sampling the coefficients on the device
    (``adr_coefficients_cupy.prepare_adr_data_cupy``); those stage times are
    part of the reported assembly time and its ``raw.coefficients.*`` details.

    ``cache`` (owned by :class:`AdvectionDiffusionReactionHDGSolver`) keeps
    data across repeated solves on one space: the prepared diffusion tensor,
    the diffusion stabilization and the reduced sparsity pattern when
    ``options.reuse_static_coefficients`` is set, and persistent AMGX solvers
    when ``options.amgx_reuse`` is ``"solver"`` (fresh setup, reused objects)
    or ``"preconditioner"`` (coefficients replaced and the previous setup kept,
    refreshed every ``amgx_refresh_interval`` solves, when the iteration count
    exceeds ``amgx_refresh_iteration_growth`` times the count after the last
    refresh, and once after a failed solve with a stale setup).
    """
    if str(options.solver).lower() not in {"amgx", "pyamgx"}:
        raise ValueError("assembly_backend='raw-cuda' currently requires solver='amgx'")
    cp = require_cupy()
    from hdgfem.backends.advection_cuda import reconstruct_trace_cupy, solve_reduced_system_amgx_device
    from hdgfem.solvers.advection_diffusion_reaction import (
            _detailed_logging,
            _print_diffusion_structure,
            _print_timing_details,
            _solver_verbosity,
            _timed_substep,
        )
    from hdgfem.solvers.diffusion_reaction import _timed_call, _verbosity_level
    from hdgfem.backends.adr_coefficients_cupy import prepare_adr_data_cupy
    verbosity = _verbosity_level(options.verbose)

    static = None
    if cache is not None and options.reuse_static_coefficients:
        key = _static_cache_key(space, trace_space, options)
        static = cache.get("static")
        if static is None or static["key"] != key:
            static = cache["static"] = {"key": key}

    def assemble():
        """Sample coefficients on the device, then run the tensor assembly kernel."""
        from hdgfem.assembly.diffusion_coefficients import prepare_diffusion

        coefficient_timings: dict[str, float] = {}
        device_prepared = prepare_adr_data_cupy(
            source, reaction, beta, space, diffusion=options.diffusion,
            advection_stabilization=options.advection_stabilization,
            diffusion_stabilization=options.diffusion_stabilization,
            diffusion_penalty_constant=options.diffusion_penalty_constant,
            trace_space=trace_space, timings=coefficient_timings,
            tau_diffusion=None if static is None else static.get("tau_diffusion"))
        tensor = None if static is None else static.get("diffusion")
        if static is not None and tensor is None:
            try:
                tensor = prepare_diffusion(options.diffusion, space, device=True)
            except TypeError:  # CuPy-incompatible diffusion callable: host samples
                tensor = prepare_diffusion(options.diffusion, space)
        assembled = assemble_projected_adr_trace_operator_raw_cuda(
            device_prepared, boundary_condition, space, diffusion=options.diffusion,
            trace_space=trace_space, matrix_format=options.raw_matrix_format,
            block_size=options.raw_block_size, cache_local_factors=options.cache_local_factors,
            prepared_diffusion=tensor, csr_pattern=None if static is None else static.get("pattern"),
            mass_factors=None if static is None else static.get("mass_factors"))
        if static is not None:
            static.update(tau_diffusion=device_prepared.tau_diffusion, diffusion=tensor,
                          pattern=assembled.csr_pattern, mass_factors=assembled.mass_factors)
        assembled.assembly.timings.update(coefficient_timings)
        return device_prepared, assembled

    (prepared, operator), assembly_seconds = _timed_call(
        f"assembling reduced global trace system (raw-cuda {options.raw_matrix_format})", verbosity, assemble)
    assembly = operator.assembly
    _print_diffusion_structure(operator.diffusion_structure, verbosity)
    _print_timing_details("raw-cuda assembly timings", assembly.timings, verbosity)
    if _detailed_logging(verbosity):
        print(f"  reduced trace system: {assembly.rhs.size:,} free trace dofs, "
              f"{assembly.data.size:,} stored {assembly.matrix_format.upper()} values", flush=True)
    module, device_inputs, boundary = operator.module, operator.device_inputs, operator.boundary_trace
    cspace, mesh = assembly.cspace, space.mesh
    rhs = assembly.rhs
    amgx_config = options.amgx_config
    if amgx_config is None:
        amgx_config = {
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "solver": "FGMRES",
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI_CORE",
                "tolerance": float(options.solver_rtol),
                "max_iters": 500 if options.maxiter is None else int(options.maxiter),
                "gmres_n_restart": 100,
                "print_solve_stats": 0,
                "norm": "L2",
                "preconditioner": {"solver": "MULTICOLOR_DILU", "max_iters": 1},
            },
        }
    def solve_once(reusable=None, reuse_preconditioner=False):
        """One AMGX solve, optionally on a persistent solver."""
        return solve_reduced_system_amgx_device(
            assembly, config=amgx_config, tolerance=options.solver_rtol, atol=options.solver_atol,
            maxiter=options.maxiter, initial_guess=options.initial_guess, scale_system=options.scale_system,
            reusable_solver=reusable, reuse_primary_preconditioner=reuse_preconditioner,
            materialize_host_solution=options.materialize_host_solution, verbose=_solver_verbosity(verbosity))

    def solve():
        """Solve, reusing a cached AMGX solver per ``options.amgx_reuse``."""
        reuse = options.amgx_reuse
        if cache is None or reuse == "none":
            return solve_once()
        from hdgfem.backends.advection_cuda import PyAMGXCsrDeviceSolver
        states = cache.setdefault("amgx", {})
        key = _amgx_cache_key(amgx_config, options, assembly.rhs.size)
        state = states.get(key)
        if state is None or state["solver"].closed:
            state = states[key] = dict(
                solver=PyAMGXCsrDeviceSolver(config=amgx_config, tolerance=options.solver_rtol,
                                             maxiter=options.maxiter, verbose=_solver_verbosity(verbosity),
                                             reusable=True),
                reference_iterations=None, since_refresh=0, last_iterations=None)
        solver = state["solver"]
        stale = reuse == "preconditioner" and solver.is_setup
        if stale and (state["since_refresh"] >= int(options.amgx_refresh_interval)
                      or (state["reference_iterations"] and state["last_iterations"]
                          and state["last_iterations"] > options.amgx_refresh_iteration_growth
                          * state["reference_iterations"])):
            stale = False
        if not stale:
            solver.is_setup = False  # fresh setup of the current matrix on the reused objects
        try:
            result = solve_once(solver, reuse_preconditioner=stale)
        except Exception:
            if not stale or solver.closed:
                raise
            solver.is_setup = False  # the stale setup failed: retry once with a fresh setup
            stale = False
            result = solve_once(solver, reuse_preconditioner=False)
        iterations = result[0].iteration_count
        if stale:
            state["since_refresh"] += 1
        else:
            state["since_refresh"], state["reference_iterations"] = 0, iterations
        state["last_iterations"] = iterations
        return result

    (solve_result, reduced), solve_seconds = _timed_call(
        "solving global system (AMGX device)", verbosity, solve, multiline=verbosity >= 1)
    trace_device, _ = _timed_substep(
        "expanding reduced trace", verbosity,
        lambda: reconstruct_trace_cupy(reduced, assembly.boundary_trace, cspace))
    (unknowns, reconstruction_timings), reconstruction = _timed_call(
        "reconstructing local fields (raw-cuda)", verbosity,
        lambda: reconstruct_projected_adr_local_unknowns_raw_cuda(
            operator, trace_device, block_size=options.raw_block_size))
    _print_timing_details("raw-cuda reconstruction timings", reconstruction_timings, verbosity)
    from hdgfem.solvers.advection_diffusion_reaction import (
        AdvectionDiffusionReactionResult, AdvectionDiffusionReactionTimings,
        _postprocess_primal_from_total_flux, _postprocess_total_flux,
        _reported_postprocessing_backend, _project_total_flux,
    )
    from hdgfem.solvers.diffusion_reaction import _normalize_hdg_postprocess_mode, split_diffusion_unknowns

    field, flux = split_diffusion_unknowns(unknowns, space)
    total_flux = _project_total_flux(unknowns, prepared, space)
    post_mode = _normalize_hdg_postprocess_mode(options.hdg_postprocess)
    # Face samples returned on the result; device arrays unless host output is requested.
    face_samples = {"tau_advection": prepared.tau_advection, "tau_diffusion": prepared.tau_diffusion,
                    "beta_dot_normal": prepared.beta_dot_normal}

    def finish():
        """Postprocess on the selected backend, then materialize and synchronize outputs."""
        recovered_field = recovered_flux = None
        local, full_trace = unknowns, trace_device
        if post_mode != "none" and options.postprocessing_backend == "numba":
            local, full_trace = cp.asnumpy(unknowns), cp.asnumpy(trace_device)
        if post_mode != "none":
            recovered_flux, _ = _timed_substep(
                f"recovering total flux ({options.flux_postprocess_space})", verbosity,
                lambda: _postprocess_total_flux(
                    local, full_trace, beta, prepared, space, trace_space,
                    options.advection_stabilization, options.flux_postprocess_space,
                    options.postprocessing_backend))
        if post_mode in {"primal", "both"}:
            recovered_field, _ = _timed_substep(
                "recovering primal field", verbosity,
                lambda: _postprocess_primal_from_total_flux(
                    local, recovered_flux, beta, prepared, space, trace_space,
                    options.advection_stabilization, options.diffusion, options.postprocessing_backend))
        returned_flux = recovered_flux if post_mode in {"flux", "both"} else None

        def materialize():
            """Download requested host outputs and drain the device stream."""
            nonlocal local, full_trace
            if options.materialize_host_solution:
                if isinstance(local, cp.ndarray):
                    local = cp.asnumpy(local)
                if isinstance(full_trace, cp.ndarray):
                    full_trace = cp.asnumpy(full_trace)
                for key, value in face_samples.items():
                    face_samples[key] = cp.asnumpy(value)
                for output in (field, *flux.components, *total_flux.components,
                               recovered_field, *(returned_flux.components if returned_flux is not None else ())):
                    if output is not None:
                        _ = output.coeffs
            cp.cuda.get_current_stream().synchronize()

        _timed_substep(
            "materializing host outputs" if options.materialize_host_solution else "synchronizing device outputs",
            verbosity, materialize)
        return local, full_trace, recovered_flux, recovered_field, returned_flux

    finish_label = (f"postprocessing ({post_mode}, {options.postprocessing_backend})" if post_mode != "none"
                    else "finalizing raw-cuda outputs")
    (local_unknowns, trace, total_flux_star, post_field, post_flux), post_seconds = _timed_call(
        finish_label, verbosity, finish, multiline=_detailed_logging(verbosity))
    timings=AdvectionDiffusionReactionTimings(preparation=0.,trace_assembly=assembly_seconds,solve=solve_seconds,reconstruction=reconstruction,postprocessing=post_seconds,total=time.perf_counter()-total_start,details={**assembly.timings, **reconstruction_timings})
    return AdvectionDiffusionReactionResult(field=field,flux=flux,total_flux=total_flux,trace=trace,timings=timings,postprocessed_field=post_field,postprocessed_flux=post_flux,local_unknowns=local_unknowns,matrix_rows=assembly.rows,matrix_cols=assembly.cols,matrix_data=assembly.data,matrix_indptr=assembly.indptr,matrix_indices=assembly.indices,matrix_format=assembly.matrix_format,diffusion_structure=operator.diffusion_structure,rhs=rhs,boundary_trace=boundary,reduction=None,element_boundary_mats=prepared.element_boundary,tau_advection=face_samples["tau_advection"],tau_diffusion=face_samples["tau_diffusion"],beta_dot_normal=face_samples["beta_dot_normal"],assembly_backend="raw-cuda",reconstruction_backend="raw-cuda",postprocessing_backend=("none" if post_mode == "none" else _reported_postprocessing_backend(options.postprocessing_backend,post_mode)),global_solve_result=solve_result)


__all__=["RawADRTraceOperator", "reconstruct_projected_adr_local_unknowns_raw_cuda", "assemble_projected_adr_trace_operator_raw_cuda", "assemble_projected_adr_trace_system_eliminated_raw_cuda"]
