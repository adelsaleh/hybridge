"""Experimental oversampled recovery on a perturbed square mesh.

This is a bounded manufactured-problem diagnostic, not a production recovery
API or a BDF2 preset. One degree-(p-1) full-domain solve is followed by four
independent degree-p patch solves. Each patch is a proper subset of the mesh;
there is no enriched full-domain solve or iteration between patches.

The physical overlap stays at 1/4 as h decreases: the patch element count
therefore grows with refinement. Total enriched element work is 2.25 times a
degree-p full-domain discretization. No cost advantage is claimed. At most
512 triangles are used in the base problem and 288 in each patch. No time
integration, builds, or native compilation are performed.
"""
import os
if os.environ.get('NUMBA_DISABLE_JIT') != '1':
    raise RuntimeError('Run this bounded matrix diagnostic with NUMBA_DISABLE_JIT=1')
import numba
if not numba.config.DISABLE_JIT:
    raise RuntimeError('Numba was imported before JIT was disabled')
import json
import argparse
import numpy as np
from hdgfem.core.mesh import DGMesh, rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.core.transfer import project_same_mesh_field, evaluate_field_at_points
from hdgfem.solvers.diffusion_reaction import solve_diffusion_reaction_hdg

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density-order', type=int, choices=range(2, 7), default=6)
    parser.add_argument('--jitter', type=float, default=0.15)
    parser.add_argument('--case', choices=('trig', 'poly', 'harmonic'), default='trig')
    args = parser.parse_args()
    p, jitter, kind = args.density_order, args.jitter, args.case
    if not 0. <= jitter <= .2:
        parser.error('--jitter must lie in [0, 0.2]')
    rows=[]
    for n in [4,8,16]:
        mesh=rectangle_mesh(n,n,xlim=(0.,1.),ylim=(0.,1.))
        if jitter:
            nodes=mesh.node_coords.copy()
            interior=np.all((nodes>0)&(nodes<1),axis=1)
            nodes[interior]+=np.random.default_rng(1907+n).uniform(-jitter/n,jitter/n,(int(interior.sum()),2))
            mesh=DGMesh.from_arrays(nodes,mesh.triangles)
        out=DGSpace(mesh,p,basis_type='dub_orth',volume_quad_1d=12,edge_quad_1d=12)
        space=DGSpace(mesh,p-1,basis_type='dub_orth',volume_quad_1d=12,edge_quad_1d=12)
        if kind=='poly':
            u=lambda x,y:(x+.5*y)**(p+1)
            f=lambda x,y:-1.25*p*(p+1)*(x+.5*y)**(p-1)
            E=lambda x,y:-(p+1)*np.stack(((x+.5*y)**p,.5*(x+.5*y)**p),axis=-1)
        elif kind=='harmonic':
            u=lambda x,y:np.real((x+1j*y)**(p+1))
            f=lambda x,y:0.*x
            E=lambda x,y:-(p+1)*np.stack((np.real((x+1j*y)**p),-np.imag((x+1j*y)**p)),axis=-1)
        else:
            a=2.*np.pi
            u=lambda x,y:np.sin(a*(x+.23))*np.sin(a*(y+.17))
            f=lambda x,y:2*a*a*u(x,y)
            E=lambda x,y:-a*np.stack((np.cos(a*(x+.23))*np.sin(a*(y+.17)),np.sin(a*(x+.23))*np.cos(a*(y+.17))),axis=-1)
        source=out.project_callable(f)
        kw=dict(stabilization=1.,solver='direct',preconditioner=None,assembly_backend='numpy',local_solver_backend='numpy',
                boundary_mode='eliminate',trace_basis='legendre-modal',verbose=False)
        result=solve_diffusion_reaction_hdg(project_same_mesh_field(source,space),0.,u,space,hdg_postprocess='primal',**kw)
        star=result.postprocessed_field
        recovered=np.empty((mesh.num_tri,out.el_dof,2))
        counts=[]
        # Four fixed cores, each with an overlap of physical width 1/4. The
        # number of elements in the overlap increases as h decreases.
        ix=np.repeat(np.tile(np.arange(n),n),2)
        iy=np.repeat(np.repeat(np.arange(n),n),2)
        for bx in range(2):
            for by in range(2):
                core=(ix>=bx*n//2)&(ix<(bx+1)*n//2)&(iy>=by*n//2)&(iy<(by+1)*n//2)
                select=(ix>=max(0,bx*n//2-n//4))&(ix<min(n,(bx+1)*n//2+n//4))
                select&=(iy>=max(0,by*n//2-n//4))&(iy<min(n,(by+1)*n//2+n//4))
                ids=np.flatnonzero(select)
                assert ids.size<mesh.num_tri
                vertices,indices=np.unique(mesh.triangles[ids],return_inverse=True)
                patch=DGMesh.from_arrays(mesh.node_coords[vertices],indices.reshape(-1,3))
                patch_space=DGSpace(patch,p,basis_type='dub_orth',volume_quad_1d=12,edge_quad_1d=12)
                def boundary(x,y):
                    shape=x.shape
                    values=evaluate_field_at_points(star,np.column_stack((x.ravel(),y.ravel()))).reshape(shape)
                    physical=(np.isclose(x,0)|np.isclose(x,1)|np.isclose(y,0)|np.isclose(y,1))
                    return np.where(physical,u(x,y),values)
                local=solve_diffusion_reaction_hdg(patch_space.field(source.coeffs[ids]),0.,boundary,patch_space,**kw)
                keep=core[ids]
                for c in range(2):recovered[ids[keep],:,c]=local.flux.components[c].coeffs[keep]
                counts.append(patch.num_tri)
        ref=out.quad_data.Krf_quads;pts=out.mapped_quads();exact=E(pts[...,0],pts[...,1])
        rec=out.vector_field([out.field(recovered[:,:,c]) for c in range(2)])
        errors={}
        for name,field in [('raw',result.flux),('oversampled',rec)]:
            val=np.stack([c.values_at_ref(ref) for c in field.components],axis=-1)
            errors[name]=float(np.sqrt(np.einsum('kq,q,k->',np.sum((val-exact)**2,axis=-1),out.quad_data.Krf_w,mesh.aff_jacs)))
        slopes={} if not rows else {key:float(np.log2(rows[-1]['errors'][key]/v)) for key,v in errors.items()}
        row=dict(p=p,n=n,kind=kind,jitter=jitter,errors=errors,orders=slopes,patch_elements=counts)
        rows.append(row)
        print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
