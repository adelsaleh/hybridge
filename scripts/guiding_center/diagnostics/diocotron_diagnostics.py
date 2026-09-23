"""Disk modal diagnostics using HDGFEM's cached mapped-grid sampler.

The paper's whole-domain potential perturbation norm is retained by the
standard runner. These additional polar Fourier norms separate mode growth
from axisymmetric equilibrium drift and competing angular modes.
"""
from __future__ import annotations

import math
import numpy as np

from hdgfem.io.raster import RasterGeometry, DeviceRasterSampler


class DiocotronModeDiagnostics:
    """Cache a polar sampling map, equilibrium and radial integration weights."""

    def __init__(self, space, equilibrium_potential, *, mode=9, inner=.45, outer=.50,
                 radial_points=32, angular_points=None, backend="host"):
        if mode < 1 or int(mode)!=mode or radial_points < 2:
            raise ValueError("positive integer mode and at least two radial points required")
        self.space, self.mode, self.backend = space, int(mode), backend
        self.angular_points = int(angular_points or 2**math.ceil(math.log2(max(128, 8*mode))))
        if self.angular_points <= 6*mode:
            raise ValueError("angular points must exceed 6*mode to separate three harmonics")
        # The actual polygon contains this origin-centered circle. Omit its
        # thin outer shell from modal sampling instead of filling missing data.
        mesh = space.mesh
        vertices = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]]
        lengths = np.linalg.norm(vertices[:, 1]-vertices[:, 0], axis=1)
        distances = np.abs(vertices[:,0,0]*vertices[:,1,1]-vertices[:,0,1]*vertices[:,1,0])/lengths
        self.radius = float(distances.min())*(1-1.e-12)
        if not 0 <= inner < outer < self.radius:
            raise ValueError("polar annulus must lie inside the mesh's inscribed circle")
        points,weights=np.polynomial.legendre.leggauss(int(radial_points))
        radii, radial_weights=[],[]
        breaks=[0.,inner,(inner+outer)/2,outer,self.radius]
        for left,right in zip(breaks,breaks[1:]):
            if right<=left:
                continue
            r=(left+right)/2+(right-left)*points/2
            radii.extend(r); radial_weights.extend(r*weights*(right-left)/2)
        self.radii=np.asarray(radii)
        theta=2*np.pi*np.arange(self.angular_points)/self.angular_points
        xy=np.stack((self.radii[:,None]*np.cos(theta), self.radii[:,None]*np.sin(theta)),axis=-1)
        geometry=RasterGeometry.from_points(mesh,xy.reshape(-1,2),width=self.angular_points,height=len(self.radii))
        if len(geometry.valid_pixels)!=xy.shape[0]*xy.shape[1]:
            raise ValueError("polar mode sampling requires a disk mesh without holes")
        # Retain every resolved positive mode below Nyquist. This includes
        # neighboring instabilities outside the seeded mode's first harmonics.
        self.modes=tuple(range(1,self.angular_points//2))
        if backend == "device":
            from hdgfem.backends.cupy import as_cupy_space
            self.sampler=DeviceRasterSampler(space,geometry,device_id=as_cupy_space(space).device_id)
            self.xp=self.sampler.cp
            self.sample=self.sampler.sample
        elif backend == "host":
            self.xp=np
            self.matrix=geometry.sampling_matrix(space)
            self.sample=lambda field: self.matrix @ field.coeffs.ravel()
        else:
            raise ValueError("backend must be host or device")
        self.weights=self.xp.asarray(radial_weights)
        self.equilibrium=self.sample(equilibrium_potential).reshape(-1,self.angular_points).copy()
        self.mode_ids=self.xp.asarray(self.modes)
        # Prime FFT plans and reductions with the same sizes used by every step.
        self.measure(equilibrium_potential)

    def measure(self, potential):
        """Return mode L2 norms; amplitudes grow at gamma, their squares at 2*gamma."""
        if potential.space is not self.space:
            raise ValueError("modal diagnostics require the fixed DGSpace")
        xp=self.xp
        values=self.sample(potential).reshape(-1,self.angular_points)-self.equilibrium
        modes=xp.fft.rfft(values,axis=1)/self.angular_points
        squared=4*np.pi*xp.sum(xp.abs(modes)**2*self.weights[:,None],axis=0)
        squared[0]*=.5
        if self.angular_points%2==0:
            squared[-1]*=.5
        amplitudes=xp.sqrt(xp.maximum(squared[self.mode_ids],0.))
        moment=xp.sum(modes[:,self.mode]*self.weights)
        extras=xp.stack((xp.sqrt(squared[0]),xp.sqrt(squared[1:].sum()),xp.angle(moment)))
        packed=xp.concatenate((amplitudes,extras))
        packed=packed.get() if self.backend=="device" else packed
        result={f"diocotron_phi_mode_{mode}_l2":float(value) for mode,value in zip(self.modes,packed)}
        result.update(diocotron_phi_axisymmetric_l2=float(packed[-3]),
            diocotron_phi_nonaxisymmetric_l2=float(packed[-2]),
            diocotron_phi_mode_phase=float(packed[-1]),
            diocotron_phi_mode_target_l2=result[f"diocotron_phi_mode_{self.mode}_l2"],
            diocotron_modal_radius=self.radius,diocotron_modal_backend=self.backend,
            diocotron_modal_radial_points=len(self.radii),diocotron_modal_angular_points=self.angular_points)
        from hdgfem.diagnostics import modal_activity
        activity = modal_activity(packed[:len(self.modes)], self.modes)
        result["diocotron_active_mode_count"] = int(activity["active_counts"][0])
        for rank, (number, value) in enumerate(zip(
            activity["dominant_modes"][0], activity["dominant_amplitudes"][0]), 1
        ):
            result[f"diocotron_dominant_mode_{rank}"] = int(number) if np.isfinite(number) else None
            result[f"diocotron_dominant_mode_{rank}_l2"] = float(value) if np.isfinite(value) else None
        target=result["diocotron_phi_mode_target_l2"]
        result["diocotron_phi_harmonic_ratio"] = result[f"diocotron_phi_mode_{2*self.mode}_l2"]/max(target,np.finfo(float).tiny)
        return result
