"""Distributed PETSc corrector for one pseudo-arclength section.

For a fixed predictor z_p=(Phi_p,m_p) and unit tangent t, solve

    R(Phi,m) = 0,
    <(Phi,m)-z_p, t>_H = 0,

with <(u,a),(v,b)>_H = u.T K v / E_T + a*b/T_max**2.
K and E_T are the fixed torsion stiffness and energy. Delta and epsilon never
become unknowns. The assembled Jacobian is [J, R_m; (K t_phi)/E_T, t_m/T_max²].

Crucially, we factor the *whole bordered matrix*, not a Schur complement
requiring J^{-1}. It can remain nonsingular at a simple fold of the fixed-m
problem. PETSc/MUMPS does the sparse algebra; CSR border packing is NumPy-only.
The additional scalar is owned by the last MPI rank, leaving every existing
finite-element global index and ownership range unchanged.
"""
from __future__ import annotations

import time
import uuid
import numpy as np

from .equilibrium import SolveFailure
from .run_diagnostics import ksp_reason_name, snes_reason_name


class BorderedEquilibriumProblem:
    """Long-lived augmented SNES, with rollback-safe field assignment."""

    def __init__(self, solver):
        from petsc4py import PETSc

        self.field, self.comm = solver, solver.comm
        self.n, self.N = solver.owned, solver.K.getSize()[0]
        self.scalar_owner = self.comm.size-1
        self.owns_scalar = self.comm.rank == self.scalar_owner
        self.local_size = self.n+int(self.owns_scalar)
        self.owned_boundary = solver.boundary_dofs[solver.boundary_dofs < self.n]
        self.prefix = "equiband_arc_" + self.comm.bcast(uuid.uuid4().hex if self.comm.rank == 0 else None, root=0) + "_"
        self._build_pattern(PETSc)
        self.x, self.f = self.A.createVecRight(), self.A.createVecLeft()
        self.error_vector = self.x.duplicate()
        self.snes = PETSc.SNES().create(self.comm)
        self.snes.setOptionsPrefix(self.prefix)
        self.snes.setFunction(self._residual, self.f)
        self.snes.setJacobian(self._jacobian, self.A)
        options = {**solver.petsc_options}
        database = PETSc.Options()
        for key, value in options.items():
            database[self.prefix+key] = value
        try:
            self.snes.setFromOptions()
        finally:
            for key in options:
                del database[self.prefix+key]
        self.options = options
        self.predictor = self.tangent = None
        self.tangent_row = None
        self.scalar_row = None

    def _build_pattern(self, PETSc):
        """Preallocate the FE graph plus a dense border without cell/row loops."""
        pointers, columns, _ = self.field.K.getValuesCSR()
        self.field_pointers, self.field_columns = pointers.copy(), columns.copy()
        row_ids = np.repeat(np.arange(self.n), np.diff(pointers))
        self.field_slots = np.arange(len(columns))+row_ids
        self.border_slots = pointers[1:]+np.arange(self.n)
        augmented_pointers = pointers+np.arange(self.n+1, dtype=pointers.dtype)
        self.top_nnz = int(augmented_pointers[-1])
        if self.owns_scalar:
            augmented_pointers = np.append(augmented_pointers, self.top_nnz+self.N+1)
        self.pointers = augmented_pointers.astype(PETSc.IntType)
        self.columns = np.empty(int(self.pointers[-1]), dtype=PETSc.IntType)
        self.columns[self.field_slots] = columns
        self.columns[self.border_slots] = self.N
        if self.owns_scalar:
            self.columns[self.top_nnz:] = np.arange(self.N+1, dtype=PETSc.IntType)
        self.entries = np.zeros(len(self.columns), dtype=PETSc.ScalarType)
        start, end = self.field.K.getOwnershipRange()
        local_end = end+int(self.owns_scalar)
        rows = np.repeat(np.arange(self.local_size), np.diff(self.pointers))
        diagonal = np.bincount(rows[(self.columns >= start) & (self.columns < local_end)],
                               minlength=self.local_size).astype(PETSc.IntType)
        offdiagonal = (np.diff(self.pointers)-diagonal).astype(PETSc.IntType)
        self.A = PETSc.Mat().createAIJ(size=((self.local_size, self.N+1), (self.local_size, self.N+1)),
                                       nnz=(diagonal, offdiagonal), comm=self.comm)
        self.A.setOptionsPrefix(self.prefix)
        self.A.setValuesCSR(self.pointers, self.columns, self.entries)
        self.A.assemble()

    def _scalar(self, vector):
        value = float(vector.getArray(readonly=True)[self.n]) if self.owns_scalar else None
        return self.comm.bcast(value, root=self.scalar_owner)

    def _assign(self, vector):
        # SNES holds its input vector read-locked inside callbacks. Requesting
        # a writable NumPy view here would fail and poison later trial solves.
        self.field.assign(vector.getArray(readonly=True)[:self.n])
        self.field.m.value = self._scalar(vector)
        self.field.lam.value = 1.

    def _constraint(self):
        local = float(np.dot(self.tangent_row, self.field.phi.x.array[:self.n]-self.predictor))
        return self.comm.allreduce(local)+(float(self.field.m.value)-self.predictor_m)*self.tangent_m/self.field.potential_scale**2

    def _assemble_vector(self, form):
        from dolfinx.fem import petsc as fp
        from petsc4py import PETSc
        vector = fp.assemble_vector(form)
        vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fp.set_bc(vector, self.field.bcs)
        return vector

    def _residual(self, snes, x, output):
        self._assign(x)
        residual = self._assemble_vector(self.field.residual_form)
        try:
            output.array[:self.n] = residual.array[:self.n]
            # The augmented vector includes Dirichlet DOFs. Enforce their
            # homogeneous equations explicitly so the identity Jacobian rows
            # are correct even for a diagnostic perturbation off that subspace.
            output.array[self.owned_boundary] = x.getArray(readonly=True)[self.owned_boundary]
        finally:
            residual.destroy()
        constraint = self._constraint()  # all ranks enter the reduction
        if self.owns_scalar:
            output.array[self.n] = constraint

    def _jacobian(self, snes, x, A, P):
        from dolfinx.fem import petsc as fp
        self._assign(x)
        J = self.field.problem.A
        J.zeroEntries()
        fp.assemble_matrix(J, self.field.jacobian_form, bcs=self.field.bcs)
        J.assemble()
        pointers, columns, entries = J.getValuesCSR()
        graph_changed = not (np.array_equal(pointers, self.field_pointers) and np.array_equal(columns, self.field_columns))
        if self.comm.allreduce(int(graph_changed)):
            raise RuntimeError("ARC_JACOBIAN_GRAPH_CHANGED: rebuild the fixed-mesh bordered solver")
        derivative = self._assemble_vector(self.field.midpoint_derivative_form)
        try:
            self.entries[self.field_slots] = entries
            self.entries[self.border_slots] = derivative.array[:self.n]
        finally:
            derivative.destroy()
        if self.owns_scalar:
            self.entries[self.top_nnz:-1] = self.scalar_row
            self.entries[-1] = self.tangent_m/self.field.potential_scale**2
        A.zeroEntries()
        A.setValuesCSR(self.pointers, self.columns, self.entries)
        A.assemble()

    def _monitor(self, snes, iteration, residual):
        self.field.report(f"ARC_SNES m={float(self.field.m.value):.10g} iteration={iteration} "
                          "context=pseudo_arclength_corrector "
                          f"augmented_algebraic_residual={residual:.6e} "
                          f"linear_iterations={snes.getLinearSolveIterations()}", level=2)

    def solve(self, predictor, midpoint, tangent, tangent_m, reference, *, tolerance=1e-10):
        """Return an immutable equilibrium on the given transverse section.

        Call collectively. Predictor/tangent arrays contain this rank's owned
        FE coefficients; their midpoint components are replicated scalars.
        The tangent must have unit branch norm and zero Dirichlet coefficients.
        ``reference`` supplies fixed-physics and branch ancestry, never mutable
        solution storage. Field/midpoint working memory is restored on *every*
        exit; callers still apply geometry and predictor-distance guards.
        """
        from petsc4py import PETSc

        field = self.field
        if reference.delta_fixed != field.config.band.threshold_width_delta or reference.epsilon_fixed != field.config.band.epsilon:
            raise ValueError("PHYSICS_MISMATCH")
        saved, saved_m = field.phi.x.array[:self.n].copy(), float(field.m.value)
        self.predictor, self.predictor_m = np.asarray(predictor), float(midpoint)
        self.tangent, self.tangent_m = np.asarray(tangent), float(tangent_m)
        if not np.isclose(field.state_norm(tangent, tangent_m), 1., atol=1e-8):
            raise ValueError("ARC_TANGENT_NOT_NORMALIZED")
        t, kt = field.K.createVecRight(), field.K.createVecLeft()
        try:
            t.array[:] = tangent
            field.K.mult(t, kt)
            self.tangent_row = kt.array.copy()/field.energy_scale
        finally:
            t.destroy()
            kt.destroy()
        rows = self.comm.gather(self.tangent_row, root=self.scalar_owner)
        self.scalar_row = np.concatenate(rows) if self.owns_scalar else None
        self.x.array[:self.n] = predictor
        if self.owns_scalar:
            self.x.array[self.n] = midpoint
        started = time.perf_counter()
        field.report(f"ARC_CORRECTOR_START predictor_m={midpoint:.10g} tangent_m={tangent_m:.6e}", level=2)
        try:
            field._arm_monitor(self.snes, self._monitor)
            try:
                self.snes.solve(None, self.x)
            except PETSc.Error as error:
                raise SolveFailure("ARC_SNES_NOT_CONVERGED", str(error)) from error
            self._assign(self.x)
            lo, hi = field.config.band.thresholds(float(field.m.value))
            if not 0 < lo < hi < field.potential_scale:
                raise SolveFailure("MIDPOINT_OUTSIDE_SEARCH_BOUND")
            constraint = abs(self._constraint())
            if constraint > tolerance:
                raise SolveFailure("ARC_CONSTRAINT_NOT_CONVERGED", f"residual={constraint:.6e}")
            # Estimate the error with the augmented Jacobian. The fixed-m
            # Newton error estimate is not appropriate exactly at a fold.
            self._residual(self.snes, self.x, self.f)
            self._jacobian(self.snes, self.x, self.A, self.A)
            ksp = self.snes.getKSP()
            ksp.setOperators(self.A)
            try:
                ksp.solve(self.f, self.error_vector)
            except PETSc.Error as error:
                raise SolveFailure("ARC_SINGULAR_BORDER", str(error)) from error
            if ksp.getConvergedReason() <= 0:
                raise SolveFailure("ARC_SINGULAR_BORDER")
            correction = field.state_norm(self.error_vector.array[:self.n], self._scalar(self.error_vector))
            state = field._snapshot(reference.branch_id, reference.state_id, snes=self.snes, newton_error=correction)
            field.report(f"ARC_CORRECTOR_DONE m={state.m:.10g} iterations={state.nonlinear_iterations} "
                         f"snes_reason_code={state.snes_reason} snes_reason={snes_reason_name(state.snes_reason)} "
                         f"ksp_reason_code={state.ksp_reason} ksp_reason={ksp_reason_name(state.ksp_reason)} "
                         f"dual_residual={state.residual_norm:.6e} arc_residual={constraint:.6e} "
                         f"augmented_newton_error={correction:.6e} elapsed={time.perf_counter()-started:.3f}s", level=2)
            return state
        finally:
            field.assign(saved)
            field.m.value, field.lam.value = saved_m, 1.

    def close(self):
        for name in ("snes", "error_vector", "f", "x", "A"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()
                setattr(self, name, None)
