import re
import subprocess
import sys
import textwrap

import pytest


def _pyamgx_runtime_available() -> bool:
    try:
        import cupy as cp
        import pyamgx  # noqa: F401

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_BICGSTAB_LOG_PROGRAM = textwrap.dedent(
    r"""
    import numpy as np
    import scipy.sparse
    import pyamgx

    size = 96
    matrix_host = scipy.sparse.diags(
        (
            -1.25 * np.ones(size - 1),
            3.0 * np.ones(size),
            -0.5 * np.ones(size - 1),
        ),
        offsets=(-1, 0, 1),
        format="csr",
        dtype=np.float64,
    )
    expected = np.sin(np.linspace(0.1, 7.0, size)) + 0.1 * np.cos(np.linspace(0.0, 4.0, size))
    rhs = matrix_host @ expected
    solution = np.zeros(size, dtype=np.float64)

    pyamgx.initialize()
    config = resources = matrix = vector_rhs = vector_solution = solver = None
    try:
        config = pyamgx.Config().create_from_dict(
            {
                "config_version": 2,
                "determinism_flag": 1,
                "exception_handling": 1,
                "solver": {
                    "solver": "BICGSTAB",
                    "preconditioner": {"solver": "NOSOLVER"},
                    "convergence": "RELATIVE_INI_CORE",
                    "norm": "L2",
                    "tolerance": 1.0e-12,
                    "max_iters": 200,
                    "monitor_residual": 1,
                    "store_res_history": 1,
                    "print_solve_stats": 1,
                    "obtain_timings": 0,
                },
            }
        )
        resources = pyamgx.Resources().create_simple(config)
        matrix = pyamgx.Matrix().create(resources, mode="dDDI")
        vector_rhs = pyamgx.Vector().create(resources, mode="dDDI")
        vector_solution = pyamgx.Vector().create(resources, mode="dDDI")
        solver = pyamgx.Solver().create(resources, config)
        matrix.upload_CSR(matrix_host)
        vector_rhs.upload(rhs)
        vector_solution.upload(solution)
        solver.setup(matrix)
        solver.solve(vector_rhs, vector_solution, zero_initial_guess=True)
        vector_solution.download(solution)
        relative_residual = np.linalg.norm(rhs - matrix_host @ solution) / np.linalg.norm(rhs)
        print(
            "PYAMGX_BICGSTAB_RESULT"
            f" iterations={solver.iterations_number}"
            f" relative_residual={relative_residual:.16e}"
        )
    finally:
        for obj in (solver, vector_solution, vector_rhs, matrix, resources, config):
            if obj is not None:
                obj.destroy()
        pyamgx.finalize()
    """
)


@pytest.mark.skipif(not _pyamgx_runtime_available(), reason="PyAMGX runtime is unavailable")
def test_amgx_bicgstab_iteration_log_reports_changing_residuals(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-c", _BICGSTAB_LOG_PROGRAM],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    assert completed.returncode == 0, output
    assert "monitored residual: recursive residual" in output
    assert "stop criterion: RELATIVE_INI_CORE; residual/initial <=" in output
    assert re.search(r"\bused\s+held\s+residual\s+res/initial\s+res/previous\b", output)
    assert "AMGX memory (process, GiB)" in output

    iteration_residuals = [
        float(match.group(1))
        for match in re.finditer(
            r"(?m)^\s*\d+\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+([0-9.eE+-]+)"
            r"\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s*$",
            output,
        )
    ]
    assert len(iteration_residuals) >= 3, output
    assert len(set(iteration_residuals)) >= 3, output
    assert iteration_residuals[-1] < iteration_residuals[0] * 1.0e-6, output

    final_match = re.search(r"Final monitored residual:\s*([0-9.eE+-]+)", output)
    assert final_match is not None, output
    final_residual = float(final_match.group(1))
    assert final_residual < 1.0e-10, output
    assert final_residual == pytest.approx(iteration_residuals[-1], rel=5.0e-5)

    result_match = re.search(
        r"PYAMGX_BICGSTAB_RESULT iterations=(\d+) relative_residual=([0-9.eE+-]+)",
        output,
    )
    assert result_match is not None, output
    assert int(result_match.group(1)) >= 3
    assert float(result_match.group(2)) < 1.0e-10
