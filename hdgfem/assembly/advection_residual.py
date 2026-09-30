"""Semidiscrete upwind-HDG transport without a global transport solve.

The algebraic trace constraint is eliminated with independent face mass
solves. This is the same flux as advection_reaction, tau*u - (tau-beta.n)*uhat,
including when the two element-side velocities differ. Reusing a trace from
a solve with a different velocity would define a different spatial operator.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from hdgfem.runtime.precision import REAL_DTYPE


class HDGTraceWorkspace:
    """Reuse package trace projection and device data without residual arrays.

    Implicit time schemes need projection and synchronization but no explicit
    flux evaluation. The residual evaluator extends this same workspace.
    """

    def __init__(self, space, *, trace_basis="legacy-lagrange", backend="host"):
        """Bind the fixed space and its existing host/device mirrors."""
        if backend not in {"host", "device"}:
            raise ValueError("workspace backend must be 'host' or 'device'")
        self.space, self.backend = space, backend
        self.trace_host = space.trace_space(trace_basis)
        self.xp, self.cspace = np, None
        if backend == "device":
            from hdgfem.core.device import as_cupy_space
            from hdgfem.runtime.optional import require_cupy
            self.xp = require_cupy()
            self.cspace = as_cupy_space(space)

    def _device_context(self):
        """Enter the device owning the cached arrays, or the host context."""
        return (nullcontext() if self.cspace is None
                else self.xp.cuda.Device(self.cspace.device_id))

    def synchronize(self):
        """Finish pending device work for stage timing boundaries."""
        if self.cspace is not None:
            with self._device_context():
                self.xp.cuda.get_current_stream().synchronize()

    def project_trace(self, density):
        """Project the nearest density predictor into the configured trace basis."""
        from hdgfem.core.field_ops import project_field_to_trace
        with self._device_context():
            return project_field_to_trace(density, trace_basis=self.trace_host.kind, backend=self.backend)


class UpwindHDGTransportResidual(HDGTraceWorkspace):
    """Reusable fixed-space residual, trace projection and device workspace.

    Static geometry, face maps, basis products, mass inverses and work arrays
    are retained. Velocity-dependent face masses are refreshed on each call.
    Returned fields/traces own their storage; later evaluations cannot mutate
    multistep history. Supports prescribed traces and zero boundary flux.
    """

    def __init__(self, space, *, trace_basis="legacy-lagrange", boundary_mode="zero-flux",
                 backend="host", advection_stabilization=None):
        """Bind a fixed space and allocate reusable residual work arrays."""
        if boundary_mode not in {"zero-flux", "eliminate"}:
            raise ValueError("explicit HDG residual requires zero-flux or eliminated boundaries")
        super().__init__(space, trace_basis=trace_basis, backend=backend)
        from hdgfem.solvers.stabilization import upwind_factor
        if upwind_factor(advection_stabilization) != 1.0:
            raise ValueError("residual requires upwind or conflict-averaged-upwind stabilization")
        self.advection_stabilization = advection_stabilization
        self.boundary_mode = boundary_mode
        with self._device_context():
            self._prepare()

    def _prepare(self):
        """Cache the package geometry, trace data and residual-specific work arrays."""
        xp = self.xp
        if self.cspace is None:
            self.mesh, self.q, self.trace = self.space.mesh, self.space.quad_data, self.trace_host
        else:
            from hdgfem.core.device import as_cupy_trace_space
            self.mesh, self.q = self.cspace.mesh, self.cspace.quad_data
            self.trace = as_cupy_trace_space(self.trace_host, device=self.cspace.device_id)
        mesh, trace, q = self.mesh, self.trace, self.q
        self.mu = trace.oriented_basis_table
        self.mu_products = xp.einsum("oaq,obq,q->oqab", self.mu, self.mu, trace.weights)
        self.weighted_mu = self.mu * trace.weights[None, None, :]
        self.side_orientation = (~mesh.orientations).astype(xp.int32)
        # Two side slots per edge permit deterministic gathering, without
        # repeated atomic scatters or a global sparse matrix.
        slots = self.space.mesh.edge_side_indices
        self.side_slots = xp.asarray(np.maximum(slots, 0))
        self.side_present = xp.asarray(slots >= 0)
        self.side_count = self.side_present.sum(axis=1)
        self.active_faces = (mesh.interior_face_mask if self.boundary_mode == "zero-flux"
                             else xp.ones((mesh.num_tri, 3), dtype=bool))
        self.face_basis = trace.bas_of_bd_quads
        self.face_weighted_basis = trace.weighted_bas_of_bd_quads
        self.volume_weighted_gradient = q.dbas_of_quads * q.Krf_w[None, None, :]
        n, nq, nfq, nt = mesh.num_tri, q.Krf_w.size, trace.weights.size, trace.edg_dof
        self.u_volume = xp.empty((n, nq), dtype=REAL_DTYPE)
        self.u_face = xp.empty((n, 3, nfq), dtype=REAL_DTYPE)
        self.beta_volume = xp.empty((n, nq, 2), dtype=REAL_DTYPE)
        self.beta_face = xp.empty((n, 3, nfq, 2), dtype=REAL_DTYPE)
        self.normal_flux = xp.empty_like(self.u_face)
        self.tau = xp.empty_like(self.u_face)
        self.gamma = xp.empty_like(self.u_face)
        self.face_matrix = xp.empty((n, 3, nt, nt), dtype=REAL_DTYPE)
        self.face_rhs = xp.empty((n, 3, nt), dtype=REAL_DTYPE)
        self.full_trace = xp.zeros((mesh.num_edg, nt), dtype=REAL_DTYPE)
        self.identity = xp.eye(nt, dtype=REAL_DTYPE)

    def _coefficients(self, field):
        """Get same-space coefficients without staging device fields through the host."""
        if field.space is not self.space:
            raise ValueError("residual fields must belong to its fixed DGSpace")
        if self.cspace is None:
            return field.coeffs
        from hdgfem.core.device import as_cupy_coefficients
        return as_cupy_coefficients(field, self.cspace)

    def _sum_sides(self, values):
        """Gather and sum contributions using the inverse mesh edge map."""
        values = values.reshape((-1, *values.shape[2:]))
        gathered = values[self.side_slots]
        present = self.side_present.reshape((*self.side_present.shape, *((1,)*(values.ndim-1))))
        return (gathered * present).sum(axis=1)

    def _check_trace_support(self):
        """Reject structurally singular active faces before a dense solve hides it."""
        from hdgfem.linalg.transport_diagnostics import trace_inflow_node_counts, UpwindHDGTraceRankError

        xp, mesh = self.xp, self.mesh
        aligned = xp.where(mesh.orientations[:, :, None], self.normal_flux,
                           self.normal_flux[:, :, ::-1]).reshape(-1, self.normal_flux.shape[-1])
        pairs = aligned[self.side_slots[mesh.int_edges_inds]]
        counts = trace_inflow_node_counts(pairs, xp=xp)
        active = xp.any(pairs != 0, axis=(1, 2))
        deficient = active & (counts < self.trace.edg_dof)
        if bool(xp.any(deficient)):
            positions = xp.flatnonzero(deficient)[:8]
            edges, nodes = mesh.int_edges_inds[positions], counts[positions]
            if self.cspace is not None:
                edges, nodes = edges.get(), nodes.get()
            raise UpwindHDGTraceRankError(edges.tolist(), nodes.tolist(), self.trace.edg_dof)

    def _boundary_trace(self, boundary):
        """Fill prescribed boundary rows using the shared HDG projection helper."""
        if self.boundary_mode == "zero-flux":
            if boundary is not None:
                raise ValueError("zero-flux residual requires boundary=None")
            return
        if boundary is None:
            raise ValueError("eliminated residual requires prescribed boundary data")
        from hdgfem.assembly.hdg import boundary_trace_coefficients
        self.full_trace[self.mesh.bnd_edges_inds] = boundary_trace_coefficients(
            boundary, self.space, trace_space=self.trace_host, backend=self.backend, boundary_only=True,
        )

    def _trace_rhs(self, weighted_values):
        """Project side quadrature values into globally oriented trace test functions."""
        xp = self.xp
        for orientation in (0, 1):
            local = weighted_values @ self.weighted_mu[orientation].T
            if orientation == 0:
                self.face_rhs[:] = local
            else:
                xp.copyto(self.face_rhs, local, where=(self.side_orientation == 1)[:, :, None])

    def evaluate(self, density, beta, boundary=None):
        """Return F(rho) and its algebraically consistent reduced trace.

        On inactive faces the trace is irrelevant and set to zero. A singular
        active face constraint is an error, never silently regularized into a
        different transport operator.
        """
        with self._device_context():
            return self._evaluate(density, beta, boundary)

    def _evaluate(self, density, beta, boundary):
        """Apply the volume and numerical-face fluxes, then the package mass inverse."""
        xp, mesh, q = self.xp, self.mesh, self.q
        rho = self._coefficients(density)
        if beta.dim != 2:
            raise ValueError("transport residual requires two velocity components")
        self.u_volume[:] = rho @ q.bas_of_quads
        self.u_face[:] = xp.einsum("ki,fiq->kfq", rho, self.face_basis)
        for component in (0, 1):
            b = self._coefficients(beta.components[component])
            self.beta_volume[:, :, component] = b @ q.bas_of_quads
            self.beta_face[:, :, :, component] = xp.einsum("ki,fiq->kfq", b, self.face_basis)
        self.normal_flux[:] = xp.einsum("kfqd,kfd->kfq", self.beta_face, mesh.normals)
        from hdgfem.solvers.stabilization import effective_advection_normal_flux
        self.normal_flux[:] = effective_advection_normal_flux(
            self.normal_flux, mesh, self.advection_stabilization, xp=xp)
        self.normal_flux *= self.active_faces[:, :, None]
        xp.abs(self.normal_flux, out=self.tau)
        xp.subtract(self.tau, self.normal_flux, out=self.gamma)
        self._check_trace_support()
        self._trace_rhs(mesh.jacs_el_fc[:, :, None] * self.tau * self.u_face)
        for orientation in (0, 1):
            local = xp.einsum("kfq,qab->kfab", self.gamma, self.mu_products[orientation])
            local *= mesh.jacs_el_fc[:, :, None, None]
            if orientation == 0:
                self.face_matrix[:] = local
            else:
                xp.copyto(self.face_matrix, local, where=(self.side_orientation == 1)[:, :, None, None])
        interior = mesh.int_edges_inds
        matrices = self._sum_sides(self.face_matrix)[interior]
        rhs = self._sum_sides(self.face_rhs)[interior]
        activity = self._sum_sides(self.tau.sum(axis=2)[:, :, None])[interior, 0]
        inactive = activity == 0
        matrices[inactive] = self.identity
        rhs[inactive] = 0
        self.full_trace.fill(0)
        if interior.size:
            # NumPy/CuPy support a stack of small dense face systems. The
            # extra axis is necessary with NumPy 2's batched RHS convention.
            if self.cspace is None:
                solved = xp.linalg.solve(matrices, rhs[:, :, None])[:, :, 0]
            else:
                import cupyx
                with cupyx.errstate(linalg="raise"):
                    solved = xp.linalg.solve(matrices, rhs[:, :, None])[:, :, 0]
            if not bool(xp.all(xp.isfinite(solved))):
                raise FloatingPointError("nonfinite solution of the HDG face trace constraint")
            self.full_trace[interior] = solved
        self._boundary_trace(boundary)
        local_trace = self.full_trace[mesh.loc2glob_edge]
        trace_values = xp.einsum("kfa,aq->kfq", local_trace, self.mu[0])
        reversed_values = xp.einsum("kfa,aq->kfq", local_trace, self.mu[1])
        xp.copyto(trace_values, reversed_values, where=(self.side_orientation == 1)[:, :, None])
        numerical_flux = self.tau*self.u_face - self.gamma*trace_values
        face_load = xp.einsum("kfq,fiq->ki", numerical_flux*mesh.jacs_el_fc[:, :, None],
                              self.face_weighted_basis)
        reference_beta = xp.einsum("kqd,kdD->kqD", self.beta_volume, mesh.inv_aff_mats_t)
        volume_load = xp.einsum("kq,kqD,Diq->ki", self.u_volume, reference_beta,
                                self.volume_weighted_gradient)
        from hdgfem.core.projection import field_from_moments
        moments = volume_load*mesh.aff_jacs[:, None] - face_load
        field = field_from_moments(self.space, moments, name="transport_rhs_h")
        return field, xp.ascontiguousarray(self.full_trace[interior].ravel())
