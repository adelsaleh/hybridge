"""DOLFINx 0.11 SNES equilibrium, torsion, and exact midpoint sensitivity.

Energy is recorded, never minimized as a branch selection rule. All physical
equilibria are certified with a stiffness-dual residual independent of SNES's
algebraic stopping norm. DOLFINx is an optional, lazily imported dependency.
"""
from __future__ import annotations

import uuid
import time
import numpy as np

from .geometry import (PackedMesh, build_atlas, interpolation_lattice,
                       monomial_basis)
from .crossings import BandObservableEvaluator
from .models import EquilibriumState
from .nonlinearities import Window
from .reporting import quiet_report
from .run_diagnostics import (compact_json, ksp_reason_name,
                              mesh_resolution_diagnostics, snes_reason_name)


class SolveFailure(RuntimeError):
    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(reason + (": " + detail if detail else ""))


def create_mesh(config, comm, *, report=quiet_report, rebuild_mesh_cache=False):
    """Load an explicit mesh or resolve a validated canonical cache entry.

    Generated geometry never relies on an in-memory mesh hidden from the run
    metadata.  Its exact tagged ``.msh`` file is cached and then imported by
    the same DOLFINx path as user-supplied meshes.
    """
    from dolfinx.io import gmsh as gmshio
    from pathlib import Path

    if config.geometry == "msh":
        from projects.diocotron.paths import resolve_archive_path
        path = resolve_archive_path(config.mesh_file).resolve()
        if not all(comm.allgather(path.is_file())):
            raise ValueError("MESH_FILE_NOT_FOUND_ON_ALL_RANKS")
        report(f"MESH_SOURCE mode=explicit path={path}", level=0)
        source = {"mode": "explicit", "path": str(path)}
    else:
        from .mesh_cache import ensure_cached_mesh
        cached = ensure_cached_mesh(
            config, comm, report=report, rebuild=rebuild_mesh_cache,
        )
        path = cached.path
        source = {
            "mode": "canonical_cache",
            "path": str(path),
            "cache_key": cached.key,
            "cache_status": cached.status,
            "metadata": cached.metadata,
        }
    domain = gmshio.read_from_msh(str(path), comm, rank=0, gdim=2).mesh
    return domain, source


class CellPolynomialEvaluator:
    """Batched FE evaluation on canonical affine or curved audit cells.

    Audit coefficients are replicated using one packed Allgatherv per field.
    This is deliberately a bounded fixed-mesh baseline, not scalable point
    ownership. Rays and reductions themselves are MPI distributed.
    """
    def __init__(self, domain):
        from dolfinx.mesh import CellType
        self.domain, self.comm = domain, domain.comm
        if domain.topology.cell_type != CellType.triangle or domain.geometry.dim != 2:
            raise ValueError("audit evaluator requires two-dimensional triangular geometry")
        self.ncells = domain.topology.index_map(2).size_local
        self.cells = np.arange(self.ncells, dtype=np.int32)
        self.cell_counts = np.array(self.comm.allgather(self.ncells), dtype=np.int32)
        # Evaluate the coordinate element through DOLFINx. Reading geometry
        # dofmaps directly would couple this code to Basix/Gmsh permutations.
        import ufl
        from dolfinx import fem
        geometry_degree = domain.geometry.cmaps[0].degree
        geometry_sites = interpolation_lattice(geometry_degree)
        geometry_values = fem.Expression(ufl.SpatialCoordinate(domain), geometry_sites).eval(
            domain, self.cells).reshape(self.ncells, len(geometry_sites), 2)
        inverse = np.linalg.inv(monomial_basis(geometry_sites, geometry_degree))
        geometry_coefficients = np.einsum("ij,njk->nik", inverse, geometry_values)
        # The lattice ordering above is (i,j); select the three vertices by
        # their coordinates instead of assuming an element dof ordering.
        vertex_sites = np.array([[0., 0.], [1., 0.], [0., 1.]])
        vertices = np.einsum("si,nij->nsj", monomial_basis(vertex_sites, geometry_degree),
                             geometry_coefficients)
        gathered = self.gather(vertices)
        vertex_order = np.lexsort((gathered[:, :, 1], gathered[:, :, 0]), axis=1)
        canonical = np.take_along_axis(gathered, vertex_order[:, :, None], axis=1)
        self.global_order = np.lexsort(canonical.reshape(-1, 6).T[::-1])
        gathered_geometry = self.gather(geometry_coefficients)[self.global_order]
        self.mesh = PackedMesh.from_geometry(gathered_geometry, geometry_degree)

    def gather(self, local):
        from mpi4py import MPI
        width = int(np.prod(local.shape[1:]))
        counts = self.cell_counts*width
        offsets = np.r_[0, np.cumsum(counts[:-1])]
        result = np.empty((int(self.cell_counts.sum()), *local.shape[1:]), dtype=np.float64)
        self.comm.Allgatherv(np.ascontiguousarray(local), [result, counts, offsets, MPI.DOUBLE])
        return result

    def _coefficients(self, function, degree):
        from dolfinx import fem
        function.x.scatter_forward()
        sites = interpolation_lattice(degree)
        values = fem.Expression(function, sites).eval(self.domain, self.cells).reshape(
            self.ncells, len(sites), -1)
        coefficients = np.einsum("ij,njk->nik", np.linalg.inv(monomial_basis(sites, degree)), values)
        return self.gather(coefficients)[self.global_order]

    def scalar(self, function, degree=2):
        coefficients = self._coefficients(function, degree)
        if coefficients.shape[-1] != 1:
            raise ValueError("expected scalar finite-element field")
        return coefficients[..., 0]

    def vector(self, function, degree):
        coefficients = self._coefficients(function, degree)
        if coefficients.shape[-1] != 2:
            raise ValueError("expected two-component finite-element field")
        return coefficients

    def vector_linear(self, function):
        """Compatibility name for the historical CG1 recovery."""
        return self.vector(function, 1)


class EquilibriumSolver:
    def __init__(self, config, comm=None, *, report=None,
                 rebuild_mesh_cache=False):
        import dolfinx
        import ufl
        from dolfinx import fem, mesh
        from dolfinx.fem import petsc as fp
        from mpi4py import MPI
        from petsc4py import PETSc
        from numba import set_num_threads

        if tuple(map(int, dolfinx.__version__.split(".")[:2])) < (0, 11):
            raise RuntimeError("DOLFINx >= 0.11 is required")
        if np.dtype(PETSc.ScalarType).kind != "f":
            raise RuntimeError("equiband requires a real PETSc build")
        self.config, self.comm = config, MPI.COMM_WORLD if comm is None else comm
        self.report = report if report is not None else quiet_report
        self.monitor_enabled = report is not None
        self._arclength_problem = None
        self._snes_context = "setup"
        self.report(f"SETUP phase=mesh geometry={config.geometry} ranks={self.comm.size}")
        set_num_threads(config.threads if config.backend in {"numba", "hybrid"} else 1)
        self.domain, self.mesh_source = create_mesh(
            config,
            self.comm,
            report=self.report,
            rebuild_mesh_cache=rebuild_mesh_cache,
        )
        self.V = fem.functionspace(self.domain, ("Lagrange", config.degree))
        self.owned = self.V.dofmap.index_map.size_local*self.V.dofmap.index_map_bs
        self.report(f"MESH cells={self.domain.topology.index_map(2).size_global} "
                    f"dofs={self.V.dofmap.index_map.size_global} degree={config.degree} "
                    f"geometry_degree={self.domain.geometry.cmaps[0].degree}")
        self.phi = fem.Function(self.V, name="phi")
        self.m = fem.Constant(self.domain, PETSc.ScalarType(0.1))
        self.lam = fem.Constant(self.domain, PETSc.ScalarType(1.))
        self.alpha = fem.Constant(self.domain, PETSc.ScalarType(0.5))
        self.window = Window(config.band)
        self.domain.topology.create_connectivity(1, 2)
        facets = mesh.exterior_facet_indices(self.domain.topology)
        self.boundary_dofs = fem.locate_dofs_topological(self.V, 1, facets)
        self.bcs = [fem.dirichletbc(PETSc.ScalarType(0), self.boundary_dofs, self.V)]
        prefix = "equiband_" + self.comm.bcast(uuid.uuid4().hex if self.comm.rank == 0 else None, root=0) + "_"
        self.linear_options = {"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True}
        if PETSc.Sys.hasExternalPackage("mumps"):
            self.linear_options["pc_factor_mat_solver_type"] = "mumps"
        elif self.comm.size > 1:
            raise RuntimeError("MPI direct verification requires a MUMPS-enabled PETSc build")
        self.dx = ufl.Measure("dx", domain=self.domain, metadata={"quadrature_degree": config.quadrature_degree})
        test, trial = ufl.TestFunction(self.V), ufl.TrialFunction(self.V)
        stiffness = ufl.inner(ufl.grad(trial), ufl.grad(test))*self.dx
        self.metric_problem = fp.LinearProblem(stiffness, fem.Constant(self.domain, PETSc.ScalarType(0))*test*self.dx,
                                               bcs=self.bcs, petsc_options_prefix=prefix+"metric_", petsc_options=self.linear_options)
        # ``LinearProblem.A`` is allocated by the constructor but DOLFINx
        # assembles it on the first solve.  Prime this long-lived Dirichlet
        # Riesz map once so residual certification and pseudo-arclength inner
        # products never see an unassembled PETSc matrix.
        self.metric_problem.solve()
        if self.metric_problem.solver.getConvergedReason() <= 0:
            raise SolveFailure("RESIDUAL_METRIC_NOT_CONVERGED")
        self.K = self.metric_problem.A
        self.VT = fem.functionspace(self.domain, ("Lagrange", config.torsion_degree))
        torsion_dofs = fem.locate_dofs_topological(self.VT, 1, facets)
        torsion_bc = fem.dirichletbc(PETSc.ScalarType(0), torsion_dofs, self.VT)
        torsion_test, torsion_trial = ufl.TestFunction(self.VT), ufl.TrialFunction(self.VT)
        self.torsion_problem = fp.LinearProblem(
            ufl.inner(ufl.grad(torsion_trial), ufl.grad(torsion_test))*self.dx,
            fem.Constant(self.domain, PETSc.ScalarType(1))*torsion_test*self.dx,
            bcs=[torsion_bc], petsc_options_prefix=prefix+"torsion_", petsc_options=self.linear_options)
        self.report("SETUP phase=torsion_solve")
        self.T = self.torsion_problem.solve()
        self.T.name = "T"
        if self.torsion_problem.solver.getConvergedReason() <= 0:
            raise SolveFailure("TORSION_NOT_CONVERGED")
        self.energy_scale = self.integral(ufl.inner(ufl.grad(self.T), ufl.grad(self.T)))
        self.T_equilibrium = fem.Function(self.V, name="T_on_equilibrium_space")
        self.T_equilibrium.interpolate(self.T)
        self.T_equilibrium.x.scatter_forward()
        self.cell_evaluator = CellPolynomialEvaluator(self.domain)
        self.audit_mesh = self.cell_evaluator.mesh
        # The Gmsh ``mesh_size`` is an input control, not the realized cell
        # diameter.  Measure curved physical edges and coordinate-map quality
        # once on rank zero; the packed mesh is identical on every rank.
        self.mesh_diagnostics = self.comm.bcast(
            mesh_resolution_diagnostics(self.audit_mesh, config.mesh_size)
            if self.comm.rank == 0 else None,
            root=0,
        )
        self.report("MESH_RESOLUTION " + compact_json(self.mesh_diagnostics), level=0)
        self.torsion_coefficients = self.cell_evaluator.scalar(self.T, config.torsion_degree)
        self.torsion_plot_coefficients = self.cell_evaluator.scalar(self.T_equilibrium, config.degree)
        self.x_T, self.potential_scale, _ = self.audit_mesh.center(
            self.torsion_coefficients, config.torsion_degree)
        torsion_ksp_reason = int(self.torsion_problem.solver.getConvergedReason())
        self.report(f"TORSION center={self.x_T.tolist()} maximum={self.potential_scale:.10g} "
                    f"degree={config.torsion_degree} ksp_reason_code={torsion_ksp_reason} "
                    f"ksp_reason={ksp_reason_name(torsion_ksp_reason)}")
        vector_space = fem.functionspace(
            self.domain, ("Lagrange", config.recovered_gradient_degree, (2,)))
        p, q = ufl.TrialFunction(vector_space), ufl.TestFunction(vector_space)
        self.gradient_problem = fp.LinearProblem(ufl.inner(p, q)*self.dx, ufl.inner(ufl.grad(self.T), q)*self.dx,
                                                 petsc_options_prefix=prefix+"gradient_", petsc_options=self.linear_options)
        self.report("SETUP phase=gradient_recovery_and_ray_atlas")
        self.gradient = self.gradient_problem.solve()
        self.gradient.name = "recovered_torsion_gradient"
        self.gradient_coefficients = self.cell_evaluator.vector(
            self.gradient, config.recovered_gradient_degree)
        torsion_jump = self.audit_mesh.continuity_error(
            self.torsion_coefficients, config.torsion_degree)
        gradient_jump = self.audit_mesh.continuity_error(
            self.gradient_coefficients, config.recovered_gradient_degree)
        self.report(f"PACKED_FIELD_AUDIT torsion_facet_jump={torsion_jump:.6e} "
                    f"recovered_gradient_facet_jump={gradient_jump:.6e}", level=2)
        self.atlas = build_atlas(self.audit_mesh, self.torsion_coefficients, self.gradient_coefficients, config, self.comm)
        self.report(f"FLOW_ATLAS resolved_for_T_over_Tmax_below="
                    f"{self.atlas.resolved_torsion_fraction:.6f} "
                    f"coordinate_resolution={self.atlas.flow_resolution:.6e} "
                    f"unresolved_neighbor_pairs_at_core={self.atlas.unresolved_neighbor_pairs}", level=0)
        self.observable = BandObservableEvaluator(
            self.audit_mesh, self.atlas, config, self.potential_scale, self.comm,
            torsion_coefficients=self.torsion_coefficients)
        from .audit import ContourAudit
        self.contour_audit = ContourAudit(self.audit_mesh)
        density = self.window.ufl(self.phi, self.m)
        source = (1-self.lam)*self.alpha + self.lam*density
        residual = ufl.inner(ufl.grad(self.phi), ufl.grad(test))*self.dx-source*test*self.dx
        jacobian = ufl.derivative(residual, self.phi, trial)
        options = {**self.linear_options, "snes_type": "newtonls", "snes_linesearch_type": "bt",
                   "snes_rtol": 1e-10, "snes_atol": 1e-12, "snes_max_it": config.maximum_iterations,
                   "snes_error_if_not_converged": True}
        self.problem = fp.NonlinearProblem(residual, self.phi, J=jacobian, bcs=self.bcs,
                                           petsc_options_prefix=prefix+"equilibrium_", petsc_options=options)
        if report is not None:
            self.problem.solver.setMonitor(self._monitor_snes)
        self.residual_form = fem.form(residual)
        self.jacobian_form = fem.form(jacobian)
        # W_m = -W_phi at fixed delta and epsilon.
        variable = ufl.variable(self.phi)
        derivative = ufl.diff(self.window.ufl(variable, self.m), variable)
        self.sensitivity_problem = fp.LinearProblem(jacobian, -derivative*test*self.dx, bcs=self.bcs,
                                                    petsc_options_prefix=prefix+"sensitivity_", petsc_options=self.linear_options)
        # R_m = +W_phi at fixed delta and epsilon. This is the right border of
        # the augmented Jacobian; no inverse of the fixed-m Jacobian is needed.
        self.midpoint_derivative_form = fem.form(derivative*test*self.dx)
        self.difference = fem.Function(self.V)
        self.mass_form = fem.form(density*self.dx)
        self.energy_form = fem.form((0.5*ufl.inner(ufl.grad(self.phi), ufl.grad(self.phi))
                                    - self.window.ufl_primitive(self.phi, self.m))*self.dx)
        self.norm_form = fem.form(ufl.inner(ufl.grad(self.difference), ufl.grad(self.difference))*self.dx)
        self.petsc_options = options
        self.report(f"SETUP phase=ready rays={config.number_of_rays} "
                    f"torsion_degree={config.torsion_degree} "
                    f"gradient_degree={config.recovered_gradient_degree} "
                    f"backend={config.backend} threads_per_rank={config.threads}")

    def _monitor_snes(self, snes, iteration, residual):
        """Report the algebraic Newton norm, not the certified dual PDE norm."""
        self.report(f"SNES m={float(self.m.value):.10g} lambda={float(self.lam.value):.5g} "
                    f"context={self._snes_context} "
                    f"iteration={iteration} algebraic_residual={residual:.6e} "
                    f"linear_iterations={snes.getLinearSolveIterations()}", level=2)

    def _arm_monitor(self, snes=None, callback=None):
        """Reinstall the native monitor before *every* nonlinear solve.

        PETSc 3.25 can clear its native monitor after a failed SNESSolve while
        petsc4py still reports the Python callback in getMonitor(). Cancel then
        install, instead of trusting that cached list or accumulating callbacks.
        This also covers source homotopy and the bordered corrector.
        """
        if self.monitor_enabled:
            snes = self.problem.solver if snes is None else snes
            snes.monitorCancel()
            snes.setMonitor(self._monitor_snes if callback is None else callback)

    def integral(self, integrand):
        from dolfinx import fem
        return float(self.comm.allreduce(fem.assemble_scalar(fem.form(integrand*self.dx))))

    def restore(self, state):
        if state.delta_fixed != self.config.band.threshold_width_delta or state.epsilon_fixed != self.config.band.epsilon:
            raise ValueError("PHYSICS_MISMATCH")
        self.assign(state.values)
        self.m.value = state.m
        self.lam.value = 1.

    def assign(self, values):
        if len(values) != self.owned:
            raise ValueError("PARTITION_MISMATCH")
        self.phi.x.array[:self.owned] = values
        self.phi.x.array[self.boundary_dofs] = 0
        self.phi.x.scatter_forward()

    def residual_norm(self):
        from dolfinx.fem import petsc as fp
        from petsc4py import PETSc
        residual = fp.assemble_vector(self.residual_form)
        residual.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fp.set_bc(residual, self.bcs)
        correction = residual.duplicate()
        try:
            self.metric_problem.solver.solve(residual, correction)
            squared = residual.dot(correction)/self.energy_scale
            return float(np.sqrt(max(0., squared))) if np.isfinite(squared) else np.inf
        finally:
            correction.destroy()
            residual.destroy()

    def norm(self, values):
        from dolfinx import fem
        self.difference.x.array[:self.owned] = values
        self.difference.x.scatter_forward()
        squared = self.comm.allreduce(fem.assemble_scalar(self.norm_form))/self.energy_scale
        return float(np.sqrt(max(0., squared))) if np.isfinite(squared) else np.inf

    def state_norm(self, values, midpoint):
        return float(np.hypot(self.norm(values), midpoint/self.potential_scale))

    def state_inner(self, left, left_m, right, right_m):
        """Distributed H1/torsion-energy product plus normalized midpoint.

        K is the fixed Dirichlet stiffness matrix. Tangents and increments
        have zero boundary coefficients, so its identity boundary rows do not
        alter the physical H1 product. Each owned coefficient is counted once.
        """
        x, y = self.K.createVecRight(), self.K.createVecLeft()
        try:
            x.array[:] = left
            self.K.mult(x, y)
            product = self.comm.allreduce(float(np.dot(right, y.array)))/self.energy_scale
            return product+left_m*right_m/self.potential_scale**2
        finally:
            x.destroy()
            y.destroy()

    def solve_arclength(self, predictor, midpoint, tangent, tangent_m, reference, *, tolerance=1e-10):
        """Correct one augmented predictor without changing accepted snapshots."""
        if self._arclength_problem is None:
            from .bordered import BorderedEquilibriumProblem
            self._arclength_problem = BorderedEquilibriumProblem(self)
        return self._arclength_problem.solve(predictor, midpoint, tangent, tangent_m, reference,
                                            tolerance=tolerance)

    def close(self):
        """Release the custom bordered PETSc objects collectively, if created.

        DOLFINx retains ownership of its standard problem objects. Explicitly
        closing our custom corrector also breaks its back-reference to this
        solver before MPI/PETSc interpreter finalization. Calling twice is safe.
        """
        if self._arclength_problem is not None:
            self._arclength_problem.close()
            self._arclength_problem = None

    def newton_error_estimate(self):
        """H1 size of one final linearized correction, not a rigorous FE bound.

        This exposes amplification by an ill-conditioned Jacobian that the
        residual norm alone cannot see. A singular solve returns infinity;
        the stationary field may still be kept for fold analysis.
        """
        from dolfinx.fem import petsc as fp
        from petsc4py import PETSc
        residual = fp.assemble_vector(self.residual_form)
        residual.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fp.set_bc(residual, self.bcs)
        correction = residual.duplicate()
        try:
            matrix = self.problem.A
            matrix.zeroEntries()
            fp.assemble_matrix(matrix, self.jacobian_form, bcs=self.bcs)
            matrix.assemble()
            ksp = self.problem.solver.getKSP()
            ksp.setOperators(matrix)
            try:
                ksp.solve(residual, correction)
            except PETSc.Error:
                return np.inf
            if ksp.getConvergedReason() <= 0:
                return np.inf
            return self.norm(correction.getArray(readonly=True))
        finally:
            residual.destroy()
            correction.destroy()

    def _snapshot(self, branch_id, parent_id=None, *, snes=None, newton_error=None):
        from dolfinx import fem
        residual = self.residual_norm()
        snes = self.problem.solver if snes is None else snes
        reason = int(snes.getConvergedReason())
        if reason <= 0 or not np.isfinite(residual) or residual > self.config.pde_tolerance:
            raise SolveFailure(
                "PDE_NOT_CONVERGED",
                f"SNES reason={reason} ({snes_reason_name(reason)}), dual residual={residual:.3e}")
        ident = self.comm.bcast(uuid.uuid4().hex if self.comm.rank == 0 else None, root=0)
        return EquilibriumState(
            self.phi.x.array[:self.owned], float(self.m.value),
            self.config.band.threshold_width_delta, self.config.band.epsilon,
            branch_id, residual, reason, int(snes.getIterationNumber()), ident,
            parent_id,
            float(self.comm.allreduce(fem.assemble_scalar(self.energy_form))),
            float(self.comm.allreduce(fem.assemble_scalar(self.mass_form))),
            newton_error=self.newton_error_estimate() if newton_error is None else newton_error,
            ksp_reason=int(snes.getKSP().getConvergedReason()),
        )

    def classify_stability(self, state, tolerance=1e-7):
        """Label the energy Hessian/parabolic linearization, not GC dynamics.

        Negative eigenvalues are retained. The Dirichlet DOFs are eliminated
        before the generalized eigenproblem, so boundary rows cannot supply
        spurious eigenvalues. This optional audit requires slepc4py.
        """
        from dataclasses import replace
        import ufl
        from dolfinx import fem
        from dolfinx.fem import petsc as fp
        from petsc4py import PETSc
        from slepc4py import SLEPc
        self.restore(state)
        a = fp.assemble_matrix(self.jacobian_form, bcs=self.bcs)
        a.assemble()
        trial, test = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        mass = fp.assemble_matrix(fem.form(ufl.inner(trial, test)*self.dx), bcs=self.bcs)
        mass.assemble()
        owned_free = np.setdiff1d(np.arange(self.owned, dtype=np.int32), self.boundary_dofs)
        start, _ = a.getOwnershipRange()
        indices = PETSc.IS().createGeneral((owned_free+start).astype(PETSc.IntType), comm=self.comm)
        aa, mm = a.createSubMatrix(indices, indices), mass.createSubMatrix(indices, indices)
        eps = SLEPc.EPS().create(self.comm)
        try:
            eps.setOperators(aa, mm)
            eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
            eps.setDimensions(1)
            # W_phi <= 1/epsilon for both supported windows. This shift is
            # strictly below the spectrum, so closest-to-shift is the lowest.
            eps.setTarget(-2/self.config.band.epsilon)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.getST().setType(SLEPc.ST.Type.SINVERT)
            ksp = eps.getST().getKSP()
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            if PETSc.Sys.hasExternalPackage("mumps"):
                ksp.getPC().setFactorSolverType("mumps")
            eps.setTolerances(1e-9, 300)
            eps.solve()
            if eps.getConverged() < 1:
                return replace(state, stability="EIGENSOLVE_NOT_CONVERGED")
            eigenvalue = float(eps.getEigenvalue(0).real)
            scaled = eigenvalue*self.config.radius**2
            label = "ENERGY_UNSTABLE" if scaled < -tolerance else "ENERGY_STABLE" if scaled > tolerance else "ENERGY_MARGINAL"
            return replace(state, stability=label, stability_eigenvalue=eigenvalue)
        finally:
            eps.destroy()
            aa.destroy()
            mm.destroy()
            indices.destroy()
            a.destroy()
            mass.destroy()

    def solve(self, m, initial_state=None, *, initial_values=None, branch_id="branch_0"):
        from petsc4py import PETSc
        lo, hi = self.config.band.thresholds(m)
        if not 0 < lo < hi < self.potential_scale:
            raise SolveFailure("MIDPOINT_OUTSIDE_SEARCH_BOUND")
        saved, saved_m = self.phi.x.array[:self.owned].copy(), float(self.m.value)
        previous_context = self._snes_context
        self._snes_context = "fixed_midpoint_equilibrium"
        started = time.perf_counter()
        self.report(f"EQUILIBRIUM_START m={m:.10g} c_minus={lo:.10g} c_plus={hi:.10g}", level=2)
        try:
            if initial_state is not None:
                self.restore(initial_state)
                branch_id = initial_state.branch_id
            if initial_values is not None:
                self.assign(initial_values)
            elif initial_state is None:
                self.assign(0.8*self.T_equilibrium.x.array[:self.owned])
            self.m.value, self.lam.value = m, 1.
            try:
                self._arm_monitor()
                self.problem.solve()
            except PETSc.Error as error:
                raise SolveFailure("SNES_NOT_CONVERGED", str(error)) from error
            state = self._snapshot(branch_id, initial_state.state_id if initial_state is not None else None)
            self.report(f"EQUILIBRIUM_DONE m={m:.10g} "
                        f"snes_reason_code={state.snes_reason} snes_reason={snes_reason_name(state.snes_reason)} "
                        f"ksp_reason_code={state.ksp_reason} ksp_reason={ksp_reason_name(state.ksp_reason)} "
                        f"iterations={state.nonlinear_iterations} dual_residual={state.residual_norm:.6e} "
                        f"newton_error={state.newton_error:.6e} elapsed={time.perf_counter()-started:.3f}s", level=2)
            return state
        except SolveFailure as error:
            self.report(f"EQUILIBRIUM_REJECT m={m:.10g} reason={error} rollback=1", level=2)
            raise
        finally:
            self.assign(saved)
            self.m.value, self.lam.value = saved_m, 1.
            self._snes_context = previous_context

    def homotopy_seed(self, m, alpha=0.8, steps=20, branch_id="branch_0"):
        """Construct a genuine equilibrium with rollback-safe source homotopy.

        ``steps`` sets the initial uniform increment in the source-amplitude
        parameter.  A failed positive-amplitude trial is restored from the
        last converged field and retried with half that increment.  This is
        important for very sharp windows: increasing the Newton budget can
        finish a slowly converging trial, but it cannot repair a line search
        that stagnates after too large a homotopy jump.  Only the exactly
        reached ``lambda=1`` state is returned as an equilibrium.
        """
        from petsc4py import PETSc
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("homotopy steps must be a positive integer")
        self.config.band.thresholds(m)
        saved = self.phi.x.array[:self.owned].copy()
        saved_m, saved_alpha = float(self.m.value), float(self.alpha.value)
        previous_context = self._snes_context
        self._snes_context = "source_homotopy"
        try:
            self.assign(alpha*self.T_equilibrium.x.array[:self.owned])
            self.m.value, self.alpha.value = m, alpha
            self.lam.value = 0.
            self.report(f"SEED_HOMOTOPY m={m:.10g} lambda=0", level=2)
            try:
                self._arm_monitor()
                self.problem.solve()
            except PETSc.Error as error:
                raise SolveFailure("SEED_HOMOTOPY_FAILED", "lambda=0") from error

            accepted_lambda = 0.
            accepted_values = self.phi.x.array[:self.owned].copy()
            initial_step = 1./steps
            step = initial_step
            minimum_step = initial_step/128
            attempts = 0
            maximum_attempts = max(128, 16*steps)
            while accepted_lambda < 1.:
                attempts += 1
                if attempts > maximum_attempts:
                    raise SolveFailure(
                        "SEED_HOMOTOPY_FAILED",
                        f"attempt budget exhausted after lambda={accepted_lambda:.6g}")
                trial_lambda = min(1., accepted_lambda+step)
                self.assign(accepted_values)
                self.lam.value = trial_lambda
                self.report(
                    f"SEED_HOMOTOPY m={m:.10g} lambda={trial_lambda:.5g} "
                    f"delta_lambda={trial_lambda-accepted_lambda:.5g}", level=2)
                try:
                    self._arm_monitor()
                    self.problem.solve()
                except PETSc.Error as error:
                    try:
                        reason = int(self.problem.solver.getConvergedReason())
                    except (AttributeError, TypeError, ValueError):
                        reason = 0
                    self.assign(accepted_values)
                    self.lam.value = accepted_lambda
                    reduced = .5*step
                    self.report(
                        f"SEED_HOMOTOPY_RETRY failed_lambda={trial_lambda:.10g} "
                        f"accepted_lambda={accepted_lambda:.10g} snes_reason_code={reason} "
                        f"snes_reason={snes_reason_name(reason)} "
                        f"next_delta_lambda={reduced:.6g} rollback=1", level=1)
                    if reduced < minimum_step:
                        raise SolveFailure(
                            "SEED_HOMOTOPY_FAILED",
                            f"lambda={trial_lambda:.6g}, last_accepted_lambda="
                            f"{accepted_lambda:.6g}, minimum_delta_lambda={minimum_step:.6g}") from error
                    step = reduced
                    continue
                accepted_lambda = trial_lambda
                accepted_values = self.phi.x.array[:self.owned].copy()
            return self._snapshot(branch_id)
        finally:
            self.assign(saved)
            self.m.value, self.alpha.value, self.lam.value = saved_m, saved_alpha, 1.
            self._snes_context = previous_context

    def evaluate(self, state, target=None):
        self.restore(state)
        coefficients = self.cell_evaluator.scalar(self.phi)
        metrics = self.observable.evaluate(coefficients, state.m, target)
        # Comparison is an audit of the FE field, not a positivity limiter.
        from .audit import quadratic_bounds
        lower, _ = quadratic_bounds(coefficients)
        # Compare in the equilibrium P2 representation. The high-order T field
        # remains authoritative for center/rays; this interpolation is only a
        # coarse numerical comparison-principle audit.
        _, excess = quadratic_bounds(coefficients-self.torsion_plot_coefficients)
        tolerance = max(1e-7, 0.02*(self.config.mesh_size/self.config.radius)**2)*self.potential_scale
        if np.min(lower) < -tolerance or np.max(excess) > tolerance:
            metrics.admissible, metrics.reason = False, "COMPARISON_PRINCIPLE_VIOLATION"
        if metrics.admissible:
            audit = self.contour_audit.check(coefficients, state.m, self.config.crossing_value_tolerance*self.potential_scale)
            if audit != "OK":
                self.report(f"CONTOUR_AUDIT status={audit} "
                            f"diagnostics={self.contour_audit.last_diagnostics}", level=2)
                metrics.admissible, metrics.reason = False, audit
        return metrics

    def sensitivity(self, state):
        self.report(f"SENSITIVITY_START m={state.m:.10g}", level=2)
        self.restore(state)
        from petsc4py import PETSc
        try:
            sensitivity = self.sensitivity_problem.solve()
        except PETSc.Error as error:
            raise SolveFailure("SINGULAR_OR_ILL_CONDITIONED_JACOBIAN") from error
        if self.sensitivity_problem.solver.getConvergedReason() <= 0:
            raise SolveFailure("SENSITIVITY_NOT_CONVERGED")
        values = sensitivity.x.array[:self.owned].copy()
        phi_coefficients = self.cell_evaluator.scalar(self.phi)
        sensitivity_coefficients = self.cell_evaluator.scalar(sensitivity)
        derivative = self.observable.derivative(phi_coefficients, sensitivity_coefficients, state.m)
        sensitivity_reason = int(self.sensitivity_problem.solver.getConvergedReason())
        self.report(f"SENSITIVITY_DONE m={state.m:.10g} dD_dm={derivative:.8g} "
                    f"ksp_reason_code={sensitivity_reason} "
                    f"ksp_reason={ksp_reason_name(sensitivity_reason)}", level=2)
        return values, derivative
