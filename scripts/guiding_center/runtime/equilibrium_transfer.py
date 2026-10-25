"""Consistent transfer of an imported window equilibrium to the HDG evolution, and its residuals.

The projected transfer rho_h = Pi_h W(phi_CG) evolves with the drift of phi(rho_h), the potential the evolution
derives from rho_h (compatible drift: phi_c, the Oswald average of the p+1 postprocessed HDG potential). Its steady
transport residual then contains W'(phi) grad^perp(phi - phi_CG) . grad phi_CG, the mismatch between the HDG and the
source CG potentials, largest at re-entrant wall corners. The consistent density is the root of

    R(rho) = rho - Pi_h W(phi(rho)),

with Pi_h the elementwise L2 projection at the initial-projection quadrature and phi(rho) the evolution's potential of
rho (``compatible``: phi_c; ``primal``: the HDG potential phi_h). At the root the window law holds for the potential
the drift comes from, W(phi) is exactly steady under that drift, and the steady residual left is the transported
projection error of W(phi), which no transfer removes.

Picard iteration rho <- Pi_h W(phi(rho)) diverges on the long edge modes of a band (multipliers near 1/(k d) > 1),
so the root is found by Newton: J = I - Pi_h W'(phi) dphi/drho, solved matrix-free by restarted GMRES in the L2
inner product with one HDG Poisson solve per product (the wall potential is zero, so rho -> phi is linear).

``transfer_residuals`` measures a transferred state: the window-law mismatch in L2 for both potentials, the HDG
semilinear residual -div q_h - W(phi_h) in the dual norm of the flux-jump Gram matrix (hybridge.hdg.gram, primal norm
||grad_h u||^2 + ||q||^2 + sum_F ||u - u_hat||_F^2), the steady upwind-HDG transport residual F_h under the compatible
drift, and the potential rate dphi/dt(0) = -S F_h it drives (energy norm and band modes, the initial rate of the
departure that seeds the band's instability).
"""
from __future__ import annotations

import math
import time

import numpy as np

__all__ = ["WindowLaw", "ConsistentWindowTransfer", "FactoredGramInverse", "transfer_residuals"]

_KERNELS = {}


def _device_kernels():
    if not _KERNELS:
        import cupy as cp
        preamble = """
        __device__ inline double window_sigmoid(double z) {
            double e = exp(-fabs(z));
            return z >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
        }
        """
        _KERNELS["value"] = cp.ElementwiseKernel(
            "float64 q, float64 low, float64 high, float64 mid, float64 eps", "float64 w",
            "double a = (q - low) / eps, b = (q - high) / eps;"
            "w = q <= mid ? window_sigmoid(a) - window_sigmoid(b) : window_sigmoid(-b) - window_sigmoid(-a);",
            "window_value", preamble=preamble)
        _KERNELS["derivative"] = cp.ElementwiseKernel(
            "float64 q, float64 low, float64 high, float64 eps", "float64 d",
            "double ea = exp(-fabs((q - low) / eps)), eb = exp(-fabs((q - high) / eps));"
            "d = (ea / ((1.0 + ea) * (1.0 + ea)) - eb / ((1.0 + eb) * (1.0 + eb))) / eps;",
            "window_derivative")
    return _KERNELS


def _is_device(array) -> bool:
    return hasattr(array, "__cuda_array_interface__")


def _release_device_pool():
    """Return cached CuPy blocks to the device (the evolution may share the device with other runs)."""
    try:
        import cupy as cp
    except ImportError:
        return
    cp.get_default_memory_pool().free_all_blocks()


class WindowLaw:
    """The frozen window W(q) and W'(q) of an imported equilibrium on host or device arrays.

    Same formulas as ``Window.value`` / ``Window.derivative`` (logistic: reflection about the midpoint); other
    window kinds run on the host.
    """

    def __init__(self, window, midpoint):
        self.window, self.midpoint = window, float(midpoint)
        self.low, self.high = (float(v) for v in window.config.thresholds(self.midpoint))
        self.epsilon = float(window.config.epsilon)
        self.logistic = window.config.kind == "logistic"

    def value(self, q):
        if _is_device(q):
            if self.logistic:
                return _device_kernels()["value"](q, self.low, self.high, self.midpoint, self.epsilon)
            import cupy as cp
            return cp.asarray(self.window.value(cp.asnumpy(q), self.midpoint))
        return np.asarray(self.window.value(q, self.midpoint), dtype=float)

    def derivative(self, q):
        if _is_device(q):
            if self.logistic:
                return _device_kernels()["derivative"](q, self.low, self.high, self.epsilon)
            import cupy as cp
            return cp.asarray(self.window.derivative(cp.asnumpy(q), self.midpoint))
        return np.asarray(self.window.derivative(q, self.midpoint), dtype=float)


class ConsistentWindowTransfer:
    """Newton-Krylov root of R(rho) = rho - Pi_h W(phi(rho)) on a fixed HDG space; see the module docstring.

    ``poisson_solver`` is the evolution's reusable Poisson solver (a ``CompatibleFluxPoissonSolver`` for the
    ``compatible`` potential) with zero wall potential; its source is replaced on every product. Coefficient vectors
    live on the device when ``device`` is true. ``log`` receives progress lines.
    """

    def __init__(self, space, poisson_solver, law: WindowLaw, *, quadrature_1d, potential="compatible",
                 device=True, log=None):
        from hybridge.core.space import DGSpace
        if potential not in ("compatible", "primal"):
            raise ValueError("potential must be 'compatible' or 'primal'")
        if potential == "compatible" and not hasattr(poisson_solver, "attach"):
            raise ValueError("the compatible potential needs a CompatibleFluxPoissonSolver")
        self.space, self.poisson, self.law, self.potential = space, poisson_solver, law, potential
        self.pspace = poisson_solver.space
        self.device = bool(device)
        self.log = log or (lambda message: None)
        # the rule of the runner's initial projection (core.projection.project_callable)
        integration = space if quadrature_1d is None else DGSpace(
            space.mesh, space.order, basis_type=space.reference.basis_type, volume_quad_1d=int(quadrature_1d))
        self.quadrature_1d = None if quadrature_1d is None else int(quadrature_1d)
        self.points = np.asarray(integration.quad_data.Krf_quads, dtype=float)
        weights = np.asarray(integration.quad_data.Krf_w, dtype=float)
        projection = (space.basis_at(self.points) * weights[:, None]) @ space.quad_data.MKrf_inv
        if self.device:
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        xp = self.xp
        self.projection = xp.asarray(np.ascontiguousarray(projection))
        self.mass = xp.asarray(np.ascontiguousarray(space.quad_data.MKrf))
        self.jacobians = xp.asarray(np.ascontiguousarray(space.mesh.aff_jacs))[:, None]
        self._potential_table = None
        self.solves, self.solve_seconds = 0, 0.0

    # fields and the L2 inner product
    def field(self, coefficients, *, name="rho_h"):
        if self.device:
            from hybridge.core.device import field_from_cupy_coefficients
            return field_from_cupy_coefficients(self.space, self.xp.ascontiguousarray(coefficients), name=name)
        return self.space.field(np.ascontiguousarray(coefficients), name=name)

    def coefficients(self, field):
        if self.device:
            from hybridge.core.device import as_cupy_coefficients, as_cupy_space
            return self.xp.array(as_cupy_coefficients(field, as_cupy_space(self.space)), copy=True)
        return np.array(field.coeffs, dtype=float, copy=True)

    def inner(self, u, v) -> float:
        return float(self.xp.sum(self.jacobians * (u @ self.mass) * v))

    def norm(self, u) -> float:
        return math.sqrt(max(self.inner(u, u), 0.0))

    # the map rho -> phi(rho) at the projection quadrature
    def solve_potential(self, coefficients):
        from hybridge.core.transfer import project_same_mesh_field
        self.poisson.set_source(project_same_mesh_field(self.field(coefficients), self.pspace))
        start = time.perf_counter()
        result = self.poisson.solve()
        self.solves += 1
        self.solve_seconds += time.perf_counter() - start
        return result

    def potential_values(self, result):
        from hybridge.core.field_ops import field_values_at_ref, solution_field
        if self.potential == "compatible":
            phi_c = result.compatible_potential
            values = phi_c.element_values()
            if self._potential_table is None:
                table = np.ascontiguousarray(phi_c.space.lagrange_basis_at(self.points).T)
                self._potential_table = self.xp.asarray(table)
            if self.device and not _is_device(values):
                values = self.xp.asarray(values)
            return values @ self._potential_table
        return field_values_at_ref(solution_field(result, self.pspace), self.points, device=self.device)

    def residual(self, coefficients):
        """R(rho), W'(phi(rho)) at the quadrature points, and the Poisson result of rho."""
        result = self.solve_potential(coefficients)
        values = self.potential_values(result)
        residual = coefficients - self.law.value(values) @ self.projection
        return residual, self.law.derivative(values), result

    def solve_potential_scaled(self, coefficients):
        """Poisson result of the unit vector along ``coefficients`` and the norm: phi is linear in rho, and the
        solver's absolute tolerance would dominate the solve of a small source."""
        size = self.norm(coefficients)
        return self.solve_potential(coefficients / size if size > 0.0 else coefficients), size

    def jacobian_product(self, vector, slope):
        result, size = self.solve_potential_scaled(vector)
        return vector - size * ((slope * self.potential_values(result)) @ self.projection)

    def gmres(self, apply, rhs, *, rtol, restart, max_products, label):
        """Restarted GMRES (modified Gram-Schmidt, twice) in the L2 inner product; returns (x, info)."""
        xp = self.xp
        solution = xp.zeros_like(rhs)
        rhs_norm = self.norm(rhs)
        residual, residual_norm = rhs.copy(), rhs_norm
        products, history, start = 0, [], time.perf_counter()
        while residual_norm > rtol * rhs_norm and products < max_products:
            basis = [residual / residual_norm]
            hessenberg = np.zeros((restart + 1, restart))
            cosines, sines = np.zeros(restart), np.zeros(restart)
            g = np.zeros(restart + 1)
            g[0] = residual_norm
            size = 0
            for j in range(restart):
                w = apply(basis[j])
                products += 1
                for _ in range(2):
                    for i in range(j + 1):
                        h = self.inner(w, basis[i])
                        hessenberg[i, j] += h
                        w -= h * basis[i]
                hessenberg[j + 1, j] = self.norm(w)
                for i in range(j):
                    a, b = hessenberg[i, j], hessenberg[i + 1, j]
                    hessenberg[i, j], hessenberg[i + 1, j] = cosines[i] * a + sines[i] * b, -sines[i] * a + cosines[i] * b
                a, b = hessenberg[j, j], hessenberg[j + 1, j]
                radius = math.hypot(a, b)
                cosines[j], sines[j] = (1.0, 0.0) if radius == 0.0 else (a / radius, b / radius)
                hessenberg[j, j], hessenberg[j + 1, j] = radius, 0.0
                g[j], g[j + 1] = cosines[j] * g[j], -sines[j] * g[j]
                size = j + 1
                estimate = abs(g[j + 1]) / rhs_norm
                history.append(estimate)
                if products % 10 == 0:
                    self.log(f"[transfer] {label} GMRES {products:4d}: relative residual {estimate:.3e} "
                             f"({time.perf_counter() - start:.1f}s)")
                if estimate <= rtol or products >= max_products or b == 0.0:
                    break
                basis.append(w / b)
            y = np.linalg.solve(np.triu(hessenberg[:size, :size]), g[:size])
            for i in range(size):
                solution += float(y[i]) * basis[i]
            del basis, w
            if self.device:
                _release_device_pool()
            residual = rhs - apply(solution)
            products += 1
            residual_norm = self.norm(residual)
            self.log(f"[transfer] {label} GMRES restart after {products} products: true relative residual "
                     f"{residual_norm / rhs_norm:.3e}")
        return solution, dict(products=products, relative_residual=residual_norm / rhs_norm,
                              history=[float(v) for v in history], seconds=time.perf_counter() - start)

    def solve(self, density, *, rtol=1e-11, max_newton=6, gmres_rtol=1e-6, restart=150, max_products=900,
              max_backtracks=4):
        """Newton from ``density`` (a DGField on the space); returns (field, Poisson result, records)."""
        coefficients = self.coefficients(density)
        density_norm = self.norm(coefficients)
        start = time.perf_counter()
        residual, slope, result = self.residual(coefficients)
        residual_norm = self.norm(residual)
        records = [dict(newton=0, relative_residual=residual_norm / density_norm, step_relative=None,
                        gmres_products=0, gmres_relative_residual=None, seconds=time.perf_counter() - start)]
        self.log(f"[transfer] Newton 0: ||rho - Pi W(phi_{self.potential[0]})|| / ||rho|| = "
                 f"{residual_norm / density_norm:.3e}")
        for k in range(1, max_newton + 1):
            if residual_norm <= rtol * density_norm:
                break
            step, info = self.gmres(lambda v: self.jacobian_product(v, slope), -residual, rtol=gmres_rtol,
                                    restart=restart, max_products=max_products, label=f"Newton {k}")
            scale, accepted = 1.0, False
            for _ in range(max_backtracks + 1):
                trial = coefficients + scale * step
                trial_residual, trial_slope, trial_result = self.residual(trial)
                trial_norm = self.norm(trial_residual)
                if trial_norm < residual_norm:
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                self.log(f"[transfer] Newton {k}: no decrease after {max_backtracks} halvings; stopping")
                break
            coefficients, residual, slope, result, residual_norm = trial, trial_residual, trial_slope, trial_result, trial_norm
            records.append(dict(newton=k, relative_residual=residual_norm / density_norm,
                                step_relative=scale * self.norm(step) / density_norm, step_scale=scale,
                                gmres_products=info["products"], gmres_relative_residual=info["relative_residual"],
                                seconds=time.perf_counter() - start))
            self.log(f"[transfer] Newton {k}: relative residual {residual_norm / density_norm:.3e}, step "
                     f"{records[-1]['step_relative']:.3e} (scale {scale:g}), {info['products']} products, "
                     f"{time.perf_counter() - start:.1f}s")
        if self.device:
            _release_device_pool()
        return self.field(coefficients, name="rho_h"), result, records


class FactoredGramInverse:
    """A condensed HDG Gram inverse whose trace Schur complement is assembled once and factored by PyPardiso.

    Same blocks as ``hdg.gram.CondensedHDGGramInverse`` (reused from ``condensed``): u blocks A_K inverted locally,
    q blocks by the reference mass, and S = C - B^T A^{-1} B (B the element-to-trace blocks) assembled over interior
    edges and Cholesky-factored (PARDISO mtype 2) instead of solved by CG for every residual. The Gram does not depend
    on the state, so every dual norm then costs one pair of triangular solves.
    """

    def __init__(self, condensed, *, threads=4):
        import scipy.sparse as sparse
        from hybridge.linalg.direct import _import_pypardiso, prepare_pypardiso_spd_matrix
        from hybridge.linalg.pardiso_runtime import pardiso_thread_limit
        start = time.perf_counter()
        self.condensed, self.threads = condensed, int(threads)
        g = condensed
        self.local_dofs, self.trace_dofs = g.local_dofs, g.trace_dofs
        self.element_count, self.element_dofs, self.edge_dofs = g.element_count, g.element_dofs, g.edge_dofs
        blocks = g.face_u_trace                                               # (K, 3, n_el, n_tr)
        solved = np.matmul(g.u_block_inverse[:, None], blocks)               # A^{-1} B per face
        schur = -np.matmul(blocks.swapaxes(2, 3)[:, :, None], solved[:, None])   # (K, 3, 3, n_tr, n_tr)
        for face in range(3):
            schur[:, face, face] += g.face_trace_trace[:, face]
        free = g.face_free_edges
        n = g.edge_dofs
        local = np.arange(n)
        rows = np.broadcast_to(free[:, :, None, None, None] * n + local[None, None, None, :, None], schur.shape)
        cols = np.broadcast_to(free[:, None, :, None, None] * n + local[None, None, None, None, :], schur.shape)
        mask = np.broadcast_to((free[:, :, None, None, None] >= 0) & (free[:, None, :, None, None] >= 0), schur.shape)
        rows, cols = rows[mask], cols[mask]
        matrix = sparse.csr_matrix((schur[mask], (rows, cols)), shape=(g.trace_dofs, g.trace_dofs))
        del schur, solved, rows, cols, mask
        self.upper = prepare_pypardiso_spd_matrix(matrix)
        self.solver = _import_pypardiso().PyPardisoSolver(mtype=2)
        self.solver.set_statistical_info_off()
        self._limit = pardiso_thread_limit
        with pardiso_thread_limit(self.threads):
            self.solver.factorize(self.upper)
        self.setup_seconds = time.perf_counter() - start
        self.nnz = int(self.upper.nnz)

    def solve(self, rhs):
        from hybridge.hdg.gram import _apply_local_rhs_inverse, _recover_local_u, _trace_rhs_element_contrib
        g = self.condensed
        rhs = np.asarray(rhs, dtype=np.float64)
        local_rhs = rhs[:g.local_dofs].reshape(g.element_count, 3 * g.element_dofs)
        local_out = np.empty_like(local_rhs)
        tmp_u = np.empty((g.element_count, g.element_dofs), dtype=np.float64)
        _apply_local_rhs_inverse(local_rhs, g.aff_jacs, g.reference_mass_inverse, g.u_block_inverse, local_out, tmp_u)
        trace_rhs = rhs[g.local_dofs:].reshape(-1, g.edge_dofs).copy()
        contribution = _trace_rhs_element_contrib(tmp_u, g.face_u_trace, g.face_free_edges)
        np.add.at(trace_rhs, g.face_free_flat, contribution.reshape(-1, g.edge_dofs)[g.interior_face_flat])
        with self._limit(self.threads):
            trace = np.asarray(self.solver.solve(self.upper, trace_rhs.reshape(-1)), dtype=np.float64)
        _recover_local_u(tmp_u, trace.reshape(-1, g.edge_dofs), g.u_block_inverse, g.face_u_trace,
                         g.face_free_edges, local_out)
        return np.concatenate((local_out.reshape(-1), trace))

    def dual_norm_squared(self, residual):
        """``r^T G^{-1} r`` and diagnostics (relative residual of the Gram solve, checked with the Gram product)."""
        from types import SimpleNamespace
        start = time.perf_counter()
        z = self.solve(residual)
        local_residual, trace_residual = self.condensed.residual(residual, z)
        relative = math.sqrt(float(local_residual @ local_residual + trace_residual @ trace_residual)) / max(
            float(np.linalg.norm(residual)), 1e-300)
        value = float(np.asarray(residual, dtype=np.float64) @ z)
        if value < 0.0:
            raise RuntimeError(f"negative Gram dual norm squared {value:.3e}")
        return value, SimpleNamespace(iterations=0, relative_residual=relative, elapsed=time.perf_counter() - start)


def _host(array):
    if _is_device(array):
        import cupy as cp
        return cp.asnumpy(array)
    return np.asarray(array)


def transfer_residuals(transfer: ConsistentWindowTransfer, density, *, transport_residual=None, band_spectrum=None,
                       gram=None, gram_space=None, cg_potential=None, label=""):
    """Residuals of a transferred state ``density`` (a DGField); see the module docstring. Returns a flat dict.

    ``transport_residual``: an ``UpwindHDGTransportResidual`` on the space (steady residual and dphi/dt(0));
    ``band_spectrum``: the run's ``BandModeSpectrum`` (band modes of dphi/dt(0), normalized as the run's
    ``band_phi_mode_amplitudes``); ``gram``: a flux-jump ``CondensedHDGGramInverse`` on ``gram_space`` (host dual
    norms; the HDG residual is evaluated with the transfer's projection quadrature); ``cg_potential``: the source
    potential phi_CG(x, y) for the HDG-CG potential gap.
    """
    from hybridge.core.field_ops import perpendicular_vector_field, solution_field
    out, start = {}, time.perf_counter()
    reference = None
    if transfer.device:
        _release_device_pool()

    def stage(name):
        transfer.log(f"[residuals{' ' + label if label else ''}] {name} ({time.perf_counter() - start:.1f}s)")
    coefficients = transfer.coefficients(density)
    rho_norm = transfer.norm(coefficients)
    result = transfer.solve_potential(coefficients)
    values_c = transfer.potential_values(result) if transfer.potential == "compatible" else None
    saved = transfer.potential
    try:
        transfer.potential = "primal"
        values_h = transfer.potential_values(result)
    finally:
        transfer.potential = saved
    mismatches = {}
    for name, values in (("compatible", values_c), ("primal", values_h)):
        if values is None:
            continue
        mismatch = coefficients - transfer.law.value(values) @ transfer.projection
        out[f"window_mismatch_{name}_relative_l2"] = transfer.norm(mismatch) / rho_norm
        mismatches[f"window_mismatch_{name}"] = _host(transfer.jacobians * (mismatch @ transfer.mass))
    phi_h = solution_field(result, transfer.pspace)
    if gram is not None:     # dual norm of rho itself, the scale of the relative dual norms
        reference = _rho_gram_dual_squared(gram, gram_space, coefficients)
    out["rho_l2"] = rho_norm
    out["phi_l2"] = float(phi_h.l2_norm())
    stage("window-law mismatch")
    if cg_potential is not None:
        from hybridge.core.projection import project_callable
        cg = project_callable(cg_potential, transfer.pspace, volume_quad_1d=transfer.quadrature_1d, backend="host")
        gap = transfer.pspace.field(_host(phi_h.coeffs if phi_h.coefficients_materialized else
                                          transfer.coefficients(phi_h)) - cg.coeffs)
        out["poisson_cg_gap_relative_l2"] = float(gap.l2_norm() / cg.l2_norm())
        stage("HDG-CG potential gap")
    if transport_residual is not None:
        beta = perpendicular_vector_field(result.flux, 1.0, transfer.space)
        steady, _ = transport_residual.evaluate(density, beta)
        steady_coefficients = transfer.coefficients(steady)
        out["steady_transport_relative_l2"] = transfer.norm(steady_coefficients) / rho_norm
        if transfer.device:
            _release_device_pool()
        stage("steady transport residual")
        rate, size = transfer.solve_potential_scaled(steady_coefficients)   # S F_h / |F_h|; dphi/dt(0) = -S F_h
        rate_field = solution_field(rate, transfer.pspace)
        raw = getattr(rate, "raw_flux", rate.flux)
        reference_flux = getattr(result, "raw_flux", result.flux)
        out["potential_rate_energy_relative"] = float(size * raw.l2_norm() / reference_flux.l2_norm())
        out["potential_rate_relative_l2"] = float(size * rate_field.l2_norm() / phi_h.l2_norm())
        if band_spectrum is not None and band_spectrum.potential_tables is not None:
            tables = band_spectrum.potential_tables
            amplitudes = size * _host(band_spectrum._spectrum(tables, tables.values(rate_field), 0.0,
                                                              band_spectrum.potential_norm))
            out["potential_rate_band_modes"] = [float(v) for v in amplitudes]
            out["potential_rate_band_peak"] = float(np.max(amplitudes[1:]))
            out["potential_rate_band_peak_mode"] = int(np.argmax(amplitudes[1:]) + 1)
        stage("potential rate and its band modes")
        if gram is not None:
            moments = _host(transfer.jacobians * (steady_coefficients @ transfer.mass))
            out.update(_gram_dual_norms(gram, gram_space, coefficients, {"steady_transport": moments}, prefix="",
                                        reference=reference))
            stage("Gram dual norm of the steady transport residual")
    if gram is not None:
        out.update(_gram_dual_norms(gram, gram_space, coefficients, mismatches, prefix="", reference=reference))
        out.update(_hdg_semilinear_dual_norm(transfer, gram, gram_space, result, coefficients, reference))
        stage("HDG residuals and their Gram dual norms")
    if label:
        out = {f"{label}_{k}": v for k, v in out.items()}
    return out


def _u_block_vector(gram, moments):
    vector = np.zeros(gram.local_dofs + gram.trace_dofs)
    local = vector[:gram.local_dofs].reshape(gram.element_count, 3 * gram.element_dofs)
    local[:, :gram.element_dofs] = moments
    return vector


def _rho_gram_dual_squared(gram, space, rho_coefficients):
    jacobians = np.asarray(space.mesh.aff_jacs)[:, None]
    rho_moments = jacobians * (_host(rho_coefficients) @ np.asarray(space.quad_data.MKrf))
    return gram.dual_norm_squared(_u_block_vector(gram, rho_moments))[0]


def _gram_dual_norms(gram, space, rho_coefficients, functionals, *, prefix, reference=None):
    """Dual norms of u-block functionals (moments, shape space.shape), relative to that of rho."""
    if reference is None:
        reference = _rho_gram_dual_squared(gram, space, rho_coefficients)
    out = {f"{prefix}rho_gram_dual": math.sqrt(reference)}
    for name, moments in functionals.items():
        value, diagnostics = gram.dual_norm_squared(_u_block_vector(gram, moments))
        out[f"{prefix}{name}_gram_dual"] = math.sqrt(value)
        out[f"{prefix}{name}_gram_dual_relative"] = math.sqrt(value / reference)
        out[f"{prefix}{name}_gram_cg_iterations"] = int(getattr(diagnostics, "iterations", -1))
    return out


def _hdg_semilinear_dual_norm(transfer, gram, space, result, rho_coefficients, reference=None):
    """HDG residual of -div q_h - W(phi_h) for the solved (phi_h, q_h, u_hat) in the flux-jump Gram dual norm.

    Evaluated with the transfer's projection quadrature. The interior trace equations are those of the linear solve
    (they do not contain W); they are reported in their own basis and left out of the dual norm, which with a zero
    trace block does not depend on the trace basis. The same residual with the source rho_h verifies the solve.
    """
    from hybridge.core.field_ops import solution_field, solution_trace
    from hybridge.core.space import DGSpace
    from hybridge.hdg.condensation import expand_interior_trace
    from hybridge.mixed.local_numpy import hdg_residual
    pspace = transfer.pspace
    integration = DGSpace(pspace.mesh, pspace.order, basis_type=pspace.reference.basis_type,
                          volume_quad_1d=transfer.quadrature_1d, edge_quad_1d=pspace.quad_data.edge_quad_1d)
    phi = solution_field(result, pspace)
    phi_coefficients = _host(phi.coeffs if phi.coefficients_materialized else transfer.coefficients(phi))
    raw_flux = getattr(result, "raw_flux", result.flux)
    flux = np.stack([_host(c.coeffs if c.coefficients_materialized else transfer.coefficients(c))
                     for c in raw_flux.components])
    trace_basis = getattr(transfer.poisson.options, "trace_basis", "legacy-lagrange")
    reduced = _host(solution_trace(result, pspace, reduced=True, prefer_device=False))
    trace = _host(expand_interior_trace(pspace, reduced, 0.0, trace_basis=trace_basis))
    trace_space = integration.trace_space(trace_basis)
    field = integration.field(phi_coefficients, name="phi_h")
    tau = transfer.poisson.options.stabilization
    rho_values = integration.field(_host(rho_coefficients)).values()
    out = {}
    for name, source in (("poisson_solve", rho_values), ("hdg_semilinear", transfer.law.value(field.values()))):
        residual = hdg_residual(field, flux, trace, source_values=np.asarray(source), stabilization=tau,
                                trace_space=trace_space)
        local = residual[:gram.local_dofs].reshape(gram.element_count, 3, gram.element_dofs)
        trace_part = residual[gram.local_dofs:]
        out[f"{name}_u_block_l2"] = float(np.linalg.norm(local[:, 0]))
        out[f"{name}_q_block_l2"] = float(np.linalg.norm(local[:, 1:]))
        out[f"{name}_trace_block_l2"] = float(np.linalg.norm(trace_part))
        vector = residual.copy()
        vector[gram.local_dofs:] = 0.0
        value, diagnostics = gram.dual_norm_squared(vector)
        out[f"{name}_gram_dual"] = math.sqrt(value)
        out[f"{name}_gram_cg_iterations"] = int(getattr(diagnostics, "iterations", -1))
    if reference is None:
        reference = _rho_gram_dual_squared(gram, space, rho_coefficients)
    out["rho_gram_dual"] = math.sqrt(reference)
    for name in ("poisson_solve", "hdg_semilinear"):
        out[f"{name}_gram_dual_relative"] = out[f"{name}_gram_dual"] / math.sqrt(reference)
    return out
