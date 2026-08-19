"""Raw CUDA stationary ADR assembly and reconstruction baseline."""

from __future__ import annotations

import time

import numpy as np

from ..assembly import hdg
from ..assembly.advection_diffusion_reaction import ADRPreparedData
from ..core.space import DGSpace, DGTraceSpace
from .cupy import as_cupy_space, require_cupy
from .numba import _boundary_reduction_maps, _interior_side_index, _trace_orientation_mode


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

__device__ __forceinline__ void factor(double* a, int* pivots) {{
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
        if (fabs(diagonal) < 1.e-30) {{ diagonal = diagonal < 0.0 ? -1.e-30 : 1.e-30; a[k*NEL+k] = diagonal; }}
        for (int i = k+1; i < NEL; ++i) {{
            a[i*NEL+k] /= diagonal;
            double multiplier = a[i*NEL+k];
            for (int j = k+1; j < NEL; ++j) a[i*NEL+j] -= multiplier*a[k*NEL+j];
        }}
    }}
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

__device__ __forceinline__ void build_factor(
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
    factor(schur,pivots);
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
        long long num_elements,long long mass_offset,double diffusion) {{
    long long element=blockIdx.x;
    if(element>=num_elements||threadIdx.x!=0)return;
    double schur[NEL*NEL],d0[NEL*NEL],d1[NEL*NEL],mn0[NEL*NEL],mn1[NEL*NEL],kd0[NEL*NEL],kd1[NEL*NEL];
    double rhs0[NEL],rhs1[NEL],rhs2[NEL],u[NEL],qx[NEL],qy[NEL],work0[NEL],work1[NEL];int pivots[NEL];
    build_factor(element,diffusion,aff_mats,aff_jacs,mass_inverse,basis,gradients,weights,reaction_values,beta_values,u_boundary_mass,normal_mass_x,normal_mass_y,d0_reference,d1_reference,schur,d0,d1,mn0,mn1,kd0,kd1,pivots);
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
        const double* source_rhs,long long num_elements,double diffusion) {{
    long long element=blockIdx.x;if(element>=num_elements||threadIdx.x!=0)return;
    double schur[NEL*NEL],d0[NEL*NEL],d1[NEL*NEL],mn0[NEL*NEL],mn1[NEL*NEL],kd0[NEL*NEL],kd1[NEL*NEL];
    double rhs0[NEL],rhs1[NEL],rhs2[NEL],u[NEL],qx[NEL],qy[NEL],work0[NEL],work1[NEL];int pivots[NEL];
    build_factor(element,diffusion,aff_mats,aff_jacs,mass_inverse,basis,gradients,weights,reaction_values,beta_values,u_boundary_mass,normal_mass_x,normal_mass_y,d0_reference,d1_reference,schur,d0,d1,mn0,mn1,kd0,kd1,pivots);
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


def assemble_projected_adr_trace_system_eliminated_raw_cuda(
        source, beta, reaction, boundary_condition, space: DGSpace, *, prepared: ADRPreparedData,
        options, trace_space: DGTraceSpace, preparation_seconds: float, total_start: float,
):
    """Assemble, solve, and reconstruct the identity-diffusion ADR system on CUDA."""
    if not np.isscalar(options.diffusion) or not np.isfinite(float(options.diffusion)) or float(options.diffusion) <= 0.0:
        raise NotImplementedError("raw CUDA ADR currently requires positive constant scalar diffusion")
    if str(options.solver).lower() not in {"amgx", "pyamgx"}:
        raise ValueError("assembly_backend='raw-cuda' currently requires solver='amgx'")
    cp = require_cupy()
    from cupyx.scipy import sparse
    from .advection_cuda import CudaAdvectionAssembly, as_cupy_trace_space, reconstruct_trace_cupy, solve_reduced_system_amgx_device
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
    device_inputs=_device_inputs(cp,prepared,space)
    source_code=_RAW_ADR_TEMPLATE.format(nel=space.el_dof,ntr=ntr,nq=space.quad_data.Krf_w.size,orientation_mode=_trace_orientation_mode(trace_space))
    compile_start=time.perf_counter(); module=cp.RawModule(code=source_code,options=("--std=c++11",)); kernel=module.get_function("assemble_adr_raw");compile_seconds=time.perf_counter()-compile_start
    args=(rows,cols,data,rhs,cp.asarray(mesh.loc2glob_edge),cp.asarray(mesh.orientations),cp.asarray(side_index),cp.asarray(edge_to_solve),cp.asarray(offsets),*device_inputs[:15],device_inputs[15],device_inputs[16],boundary.reshape(-1),np.int64(mesh.num_tri),np.int64(nflux),np.float64(options.diffusion))
    start=time.perf_counter();kernel((mesh.num_tri,),(1,),args);cp.cuda.get_current_stream().synchronize();assembly_seconds=time.perf_counter()-start
    coo=sparse.coo_matrix((data,(rows,cols)),shape=(rhs.size,rhs.size));csr=coo.tocsr();csr.sum_duplicates()
    assembly=CudaAdvectionAssembly(None,None,csr.data,rhs,None,None,device_inputs[16],boundary[mesh.bnd_edges_inds],None,cspace,trace_ref,indptr=csr.indptr,indices=csr.indices,matrix_format="csr",timings={"raw.kernel.jit":compile_seconds,"raw.kernel.wall":assembly_seconds})
    amgx_config = options.amgx_config
    if amgx_config is None:
        amgx_config = {
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "solver": "FGMRES",
                "monitor_residual": 1,
                "convergence": "ABSOLUTE",
                "tolerance": float(options.solver_rtol),
                "max_iters": 500 if options.maxiter is None else int(options.maxiter),
                "gmres_n_restart": 100,
                "print_solve_stats": 0,
                "norm": "L2",
                "preconditioner": {"solver": "MULTICOLOR_DILU", "max_iters": 1},
            },
        }
    start=time.perf_counter();solve_result,reduced=solve_reduced_system_amgx_device(assembly,config=amgx_config,tolerance=options.solver_rtol,atol=options.solver_atol,maxiter=options.maxiter,initial_guess=options.initial_guess,scale_system=options.scale_system,materialize_host_solution=options.materialize_host_solution,verbose=options.verbose);solve_seconds=time.perf_counter()-start
    trace_device=reconstruct_trace_cupy(reduced,assembly.boundary_trace,cspace)
    reconstruct_kernel=module.get_function("reconstruct_adr_raw");unknowns=cp.empty((mesh.num_tri,3*space.el_dof),dtype=cp.float64)
    rargs=(unknowns,trace_device,cp.asarray(mesh.loc2glob_edge),cp.asarray(mesh.orientations),*device_inputs[:14],device_inputs[16],np.int64(mesh.num_tri),np.float64(options.diffusion))
    start=time.perf_counter();reconstruct_kernel((mesh.num_tri,),(1,),rargs);cp.cuda.get_current_stream().synchronize();reconstruction=time.perf_counter()-start
    trace=cp.asnumpy(trace_device);local_unknowns=cp.asnumpy(unknowns)
    from ..solvers.advection_diffusion_reaction import AdvectionDiffusionReactionResult,AdvectionDiffusionReactionTimings,_postprocess_primal_from_total_flux,_postprocess_total_flux,_project_total_flux,_reported_postprocessing_backend
    from ..solvers.diffusion_reaction import _normalize_hdg_postprocess_mode,split_diffusion_unknowns
    field,flux=split_diffusion_unknowns(local_unknowns,space);total_flux=_project_total_flux(local_unknowns,prepared,space)
    post_mode=_normalize_hdg_postprocess_mode(options.hdg_postprocess);post_field=post_flux=total_flux_star=None;post_start=time.perf_counter()
    if post_mode != "none": total_flux_star=_postprocess_total_flux(local_unknowns,trace,beta,prepared,space,trace_space,options.advection_stabilization,options.flux_postprocess_space,options.postprocessing_backend)
    if post_mode in {"primal","both"}: post_field=_postprocess_primal_from_total_flux(local_unknowns,total_flux_star,beta,prepared,space,trace_space,options.advection_stabilization,options.diffusion)
    if post_mode in {"flux","both"}: post_flux=total_flux_star
    post_seconds=time.perf_counter()-post_start
    timings=AdvectionDiffusionReactionTimings(preparation=preparation_seconds,trace_assembly=assembly_seconds,solve=solve_seconds,reconstruction=reconstruction,postprocessing=post_seconds,total=time.perf_counter()-total_start,details={"raw.kernel.jit":compile_seconds,"raw.kernel.wall":assembly_seconds})
    return AdvectionDiffusionReactionResult(field=field,flux=flux,total_flux=total_flux,trace=trace,timings=timings,postprocessed_field=post_field,postprocessed_flux=post_flux,local_unknowns=local_unknowns,matrix_data=csr.data,rhs=rhs,boundary_trace=boundary,reduction=None,element_boundary_mats=prepared.element_boundary,tau_advection=prepared.tau_advection,tau_diffusion=prepared.tau_diffusion,beta_dot_normal=prepared.beta_dot_normal,assembly_backend="raw-cuda",reconstruction_backend="raw-cuda",postprocessing_backend=("none" if post_mode == "none" else _reported_postprocessing_backend(options.postprocessing_backend,post_mode)),global_solve_result=solve_result)


__all__=["assemble_projected_adr_trace_system_eliminated_raw_cuda"]
