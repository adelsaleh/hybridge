#!/usr/bin/env bash
# Sequential AMGX pytest qualification of the installed stack in both repositories.
# --plan is read-only. Runs require a new output directory; rerun into a new one.
set -euo pipefail
repo=${HYBRIDGE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
source "$repo/campaigns/common/hopper_env.sh"
export HYBRIDGE_REPO="$repo" PYTHONPATH="$repo"
export HYBRIDGE_PRECISION=float64 HDGFEM_PRECISION=float64
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTEST_ADDOPTS PYTEST_PLUGINS PYTHONHOME
exec "$HYBRIDGE_PYTHON" -B -u - "$@" <<'PY'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from campaigns.common.lifecycle import stop_on_signals, terminate_group
from campaigns.common.protocol import host_provenance

REPO = Path(os.environ['HYBRIDGE_REPO']).resolve()
WORKSPACE = Path(os.environ.get('HYBRIDGE_WORKSPACE_ROOT', REPO.parent)).resolve()
EXPECTED_COUNTS = {'hybridge': 456, 'research': 356}
HYBRIDGE_NODES = '''
tests/test_amgx_block_config_validation.py
tests/test_amgx_borrowed_csr.py
tests/test_amgx_bsr_retry_preconditioners.py
tests/test_amgx_build_variant.py
tests/test_amgx_capacity.py
tests/test_amgx_divergence_guard.py
tests/test_amgx_fixed_cycles.py
tests/test_pyamgx_logging.py
tests/test_transport_amgx_guard_policy.py
tests/test_solver_convergence_contract.py
tests/test_transport_device_direct_fallback.py
tests/test_convergence_status.py
tests/test_io_config.py
tests/test_precision.py
tests/test_poisson_native_fallback.py
tests/test_adr_solver_reuse_cuda.py
tests/test_adr_tensor_solver_cuda.py
tests/test_adr_tensor_raw_cuda_convergence.py
tests/test_cupy_backend.py::test_pyamgx_solver_matches_direct_small_system
tests/test_cupy_backend.py::test_host_pyamgx_one_shot_keeps_live_device_solver_and_exits_cleanly
tests/test_cupy_backend.py::test_p6_face_bsr_fgmres_dilu_fallback_scalarizes_only_on_device
tests/test_cupy_backend.py::test_advection_reaction_raw_cuda_amgx_solver_smoke
tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_raw_cuda_compressed_amgx_full_solve_stays_device_resident
tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_raw_cuda_schur_lu_cache_reuses_factors_rhs_reconstruction_and_amgx
tests/test_diffusion_reaction_assembly_parity.py::test_fb_hp_mg_reuses_all_fixed_poisson_state_and_supports_periodic_rt_flux
tests/test_diffusion_reaction_assembly_parity.py::test_fb_hp_mg_runtime_gate_falls_back_once_and_reuses_hybrid_amgx
tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_raw_cuda_global_operator_uses_cupy_schur_cholesky_locally
tests/test_diffusion_reaction_assembly_parity.py::test_diffusion_cupy_schur_cholesky_cache_reuses_factors_rhs_reconstruction_and_amgx
tests/test_adr_device_postprocessing.py::test_raw_cuda_result_materialization
tests/test_adr_element_coefficients.py::test_raw_cuda_solve_with_element_coefficients
tests/test_adr_unsteady_reusable.py::test_unsteady_raw_cuda_temporal_order_matches_host
tests/test_advection_diffusion_reaction.py::test_raw_cuda_matches_numpy_with_asymmetric_side_stabilization
tests/test_advection_reaction_conservation.py::test_legacy_non_tangent_raw_cuda_csr_amgx_satisfies_full_global_balance
tests/test_cg_poisson.py::test_amgx_matches_pypardiso
tests/test_diffusion_recovery_cache.py::test_tau_retry_preserves_recovery_but_invalidates_operators
tests/test_hp_mg_packed_kernels.py::test_native_pcg_presets
tests/test_legendre_face_bsr.py::test_scalar_p0_amgx_config_is_independent_of_nodal_preset
tests/test_legendre_face_bsr.py::test_native_pcg_level_three_log_separates_custom_bsr_and_coarse_amgx
tests/test_poisson_high_order_cuda.py::test_public_solver_preserves_format_and_cache_with_changing_boundary
tests/test_solver_conveniences.py::test_a_later_solve_never_overwrites_an_earlier_result
tests/test_adr_case_runner.py::test_block_amg_preset_loads_study_config
tests/test_adr_case_runner.py::test_runner_passes_native_cuda_options
tests/test_backend_capabilities.py::test_advection_raw_cuda_bsr_is_accepted_for_fused_device_amgx
tests/test_backend_capabilities.py::test_advection_raw_cuda_split3_bsr_is_accepted_for_device_amgx
tests/test_backend_capabilities.py::test_advection_raw_cuda_split3_rejects_non_bsr_policy
'''.split()
RESEARCH_NODES = '''
tests/hybridge_guiding_center/test_amgx_asm_cycle.py
tests/hybridge_guiding_center/test_amgx_bsr_dilu.py
tests/hybridge_guiding_center/test_amgx_bsr_interpolation.py
tests/hybridge_guiding_center/test_amgx_bsr_retry_preconditioners.py
tests/hybridge_guiding_center/test_amgx_bsr_smoothing.py
tests/hybridge_guiding_center/test_transport_amgx_guard_policy.py
tests/hybridge_guiding_center/test_solver_convergence_contract.py
tests/hybridge_guiding_center/test_transport_device_direct_fallback.py
tests/hybridge_guiding_center/test_precision.py
tests/hybridge_guiding_center/test_guiding_center_fast_preset.py
tests/hybridge_guiding_center/test_guiding_center_terminal_log.py
tests/hybridge_guiding_center/test_advection_operator_cache.py::test_device_source_refresh_matches_fresh_and_retains_amgx_setup
tests/hybridge_guiding_center/test_diffusion_recovery_cache.py::test_tau_retry_preserves_recovery_but_invalidates_operators
tests/hybridge_guiding_center/test_guiding_center_exact_drift.py::test_device_imex_ark3_raw_cuda_bsr_run_in_both_modes
tests/hybridge_guiding_center/test_guiding_center_h1_bdf3_gpu.py::test_real_gpu_hybrid_matches_host_with_time_dependent_boundaries
tests/hybridge_guiding_center/test_guiding_center_imex_ark3.py::test_real_gpu_ark_matches_host_with_time_dependent_boundaries
tests/hybridge_guiding_center/test_guiding_center_poisson_recovery.py::test_real_poisson_retry_rebuilds_then_reuses_and_records_all_work
tests/hybridge_guiding_center/test_guiding_center_poisson_recovery.py::test_rejected_result_snapshot_preserves_metrics_without_solution_or_preconditioner
tests/hybridge_guiding_center/test_poisson_high_order_cuda.py::test_public_solver_preserves_format_and_cache_with_changing_boundary
tests/hybridge_guiding_center/test_poisson_high_order_cuda.py::test_public_solver_primal_postprocess_on_device
tests/hybridge_guiding_center/test_guiding_center_cases.py::test_guiding_center_bsr_preset_preserves_current_amgx_configuration
tests/hybridge_guiding_center/test_guiding_center_cases.py::test_robust_transport_uses_bsr_preconditioners_before_scaled_fgmres
tests/hybridge_guiding_center/test_guiding_center_time_schemes.py::test_robust_transport_policy_builds_defect_correction_retries
tests/hybridge_guiding_center/test_guiding_center_positive_turbulence.py::test_positive_turbulence_poisson_retry_order_and_conditioning
tests/hybridge_guiding_center/test_guiding_center_positive_turbulence.py::test_non_disk_vortex_gas_retains_robust_poisson_fallback_ladder
tests/hybridge_guiding_center/test_guiding_center_positive_turbulence.py::test_turbulence_poisson_absolute_floor_reaches_all_solver_attempts
tests/hybridge_guiding_center/test_guiding_center_host_solver_benchmark.py::test_strict_amgx_poisson_config_uses_reliable_relative_gmres
tests/hybridge_guiding_center/test_guiding_center_host_solver_benchmark.py::test_transport_factorial_is_bicgstab_ilu_only
tests/hybridge_guiding_center/test_guiding_center_host_solver_benchmark.py::test_poisson_factorial_is_cached_bicgstab_ilu_only
tests/hybridge_guiding_center/test_showcase_positive_recovery.py::test_positive_options_reuse_robust_runner_attempts_and_corrected_upwind
tests/hybridge_guiding_center/test_showcase_positive_recovery.py::test_exhausted_gpu_attempts_use_pardiso_and_require_device_handoff
'''.split()

RUNTIME_CHECK = r'''
import hashlib, json, sys, sysconfig
from pathlib import Path
import cupy as cp
import numpy as np
import pyamgx
from hybridge.linalg.amgx.host import pyamgx_supports_real_dtype

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

extension = Path(pyamgx.__file__).resolve()
installed = Path(sysconfig.get_paths()['platlib']).resolve()
if not extension.is_relative_to(installed):
    raise RuntimeError(f'PyAMGX must be installed in the selected interpreter: {extension}, expected {installed}')
if not pyamgx.has_attach_bsr() or not hasattr(pyamgx.Matrix, 'attach_CSR'):
    raise RuntimeError('Installed PyAMGX lacks the qualified borrowed CSR/BSR APIs')
if cp.cuda.runtime.getDeviceCount() != 1:
    raise RuntimeError('Qualification requires exactly one visible GPU')
properties = cp.cuda.runtime.getDeviceProperties(0)
name = properties['name'].decode() if isinstance(properties['name'], bytes) else properties['name']
if 'H200' not in name:
    raise RuntimeError(f'Expected the allocated H200, found {name}')
cp.zeros(1).sum().item()
fp32_supported = pyamgx_supports_real_dtype(np.float32)
expected = Path(sys.argv[1]).resolve()
mapped = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
          if 'libamgxsh.so' in line}
if mapped != {expected}:
    raise RuntimeError(f'Loaded AMGX differs from selected primary: {mapped}, expected {expected}')
record = dict(python=sys.executable, prefix=sys.prefix, extension=str(extension),
              extension_sha256=sha(extension), library=str(expected), library_sha256=sha(expected),
              fp32_supported=fp32_supported,
              gpu=name, driver=cp.cuda.runtime.driverGetVersion(), cuda=cp.cuda.runtime.runtimeGetVersion())
Path(sys.argv[2]).write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps(record, sort_keys=True))
'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def repository(path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=all'):
        raise RuntimeError(f'Commit source changes before qualification: {path}')
    return dict(root=str(path), commit=git('rev-parse', 'HEAD'), tree=git('rev-parse', 'HEAD^{tree}'))


def run(command, cwd, env, log, timeout):
    started = time.monotonic()
    with log.open('w') as stream:
        stream.write(json.dumps(command) + '\n')
        stream.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stream.write(f'\nRunner timeout after {timeout} seconds\n')
            code = 124
        finally:
            terminate_group(process, grace=5.0)
    print('\n'.join(log.read_text(errors='replace').splitlines()[1:][-12:]), flush=True)
    return dict(returncode=code, wall_seconds=time.monotonic() - started, log_sha256=sha(log))


def junit_counts(path):
    if not path.is_file():
        return dict(tests=0, passed=0, failures=0, errors=0, skipped=0, skip_reasons=[], missing=True)
    cases = list(ET.parse(path).getroot().iter('testcase'))
    skipped = [dict(test='::'.join((case.get('classname', ''), case.get('name', ''))),
                    reason=(case.find('skipped').get('message', '') + '\n' + (case.find('skipped').text or '')).strip())
               for case in cases if case.find('skipped') is not None]
    failures = sum(case.find('failure') is not None for case in cases)
    errors = sum(case.find('error') is not None for case in cases)
    return dict(tests=len(cases), passed=len(cases)-len(skipped)-failures-errors,
                failures=failures, errors=errors, skipped=len(skipped), skip_reasons=skipped,
                junit_sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description='Run installed-stack AMGX tests in both repositories, sequentially on one H200. No resume: choose a fresh output after interruption.')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--only', choices=('both', 'hybridge', 'research'), default='both',
                        help='Rerun only an affected repository; default runs both')
    parser.add_argument('--research-repo', type=Path, default=Path(os.environ.get('HYBRIDGE_RESEARCH_REPO', WORKSPACE / 'guiding-center-cg-hdg')))
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('HYBRIDGE_CAMPAIGN_OUT', REPO / 'run_outputs' / ('fork-pytest-' + os.environ.get('SLURM_JOB_ID', datetime.now().strftime('%Y%m%d-%H%M%S'))))))
    parser.add_argument('--suite-timeout', type=float, default=3600)
    args = parser.parse_args()
    research, output = args.research_repo.resolve(), args.output.resolve()
    suites = [('hybridge', REPO, HYBRIDGE_NODES, str(REPO)),
              ('research', research, RESEARCH_NODES, os.pathsep.join((str(research), str(REPO))))]
    wrapper = ['bash', str(REPO / 'scripts/gpu/run_cuda13.sh'), sys.executable, '-B', '-u']
    plan = []
    for name, root, nodes, pythonpath in suites:
        if args.only != 'both' and name != args.only:
            continue
        if not nodes or any(not (root / node.split('::')[0]).is_file() for node in nodes):
            raise RuntimeError(f'Empty or missing explicit test selection for {name}')
        command = [*wrapper, '-m', 'pytest', '-q', '-ra', '-p', 'no:cacheprovider', '-o', 'addopts=',
                   '--junitxml=' + str(output / name / 'junit.xml'),
                   '--basetemp=' + str(output / name / 'tmp'), *nodes]
        plan.append(dict(name=name, root=str(root), pythonpath=pythonpath, command=command,
                         expected_tests=EXPECTED_COUNTS[name]))
    if args.plan:
        print(json.dumps(dict(output=str(output), precision='float64', resume=False, suites=plan), indent=2))
        return 0
    if args.suite_timeout <= 0:
        raise RuntimeError('--suite-timeout must be positive')
    if output.exists():
        raise RuntimeError(f'Output already exists; choose a fresh output, no resume or overwrite: {output}')
    sources = {name: repository(root) for name, root, _, _ in suites}
    forks = {name: repository(WORKSPACE / name) for name in ('AMGX', 'pyamgx')}
    library = (Path(os.environ['HYBRIDGE_AMGX_BUILD_ROOT']) / 'libamgxsh.so').resolve()
    expected_sha = sha(library)
    output.mkdir(parents=True)
    identity = dict(sources=sources, forks=forks, python=sys.executable, library=str(library), library_sha256=expected_sha,
                    launcher_sha256=sha(REPO / 'campaigns/fork_integration/pytest_repositories.sh'), suites=plan)
    write(output / 'identity.json', identity)
    host_provenance(output, paired_pytest=identity)
    results, interrupted = [], False
    try:
        with stop_on_signals():
            for suite in plan:
                name, root = suite['name'], Path(suite['root'])
                directory = output / name
                directory.mkdir()
                env = dict(os.environ, PYTHONPATH=suite['pythonpath'])
                result = dict(name=name, command=suite['command'])
                results.append(result)
                print(f'Starting {name}: {root}', flush=True)
                preflight = [*wrapper, '-c', RUNTIME_CHECK, str(library), str(directory / 'runtime.json')]
                try:
                    result['preflight'] = run(preflight, root, env, directory / 'preflight.log', 120)
                    if result['preflight']['returncode']:
                        result['status'] = 'preflight_failed'
                    else:
                        result['runtime'] = json.loads((directory / 'runtime.json').read_text())
                        if result['runtime']['library_sha256'] != expected_sha:
                            raise RuntimeError('Selected runtime changed before pytest')
                        result['pytest'] = run(suite['command'], root, env, directory / 'pytest.log', args.suite_timeout)
                        result['counts'] = junit_counts(directory / 'junit.xml')
                        counts = result['counts']
                        good = result['pytest']['returncode'] == 0 and counts['tests'] == suite['expected_tests'] and counts['passed'] > 0
                        result['status'] = ('passed_with_skips' if counts['skipped'] else 'passed') if good else 'failed'
                    if repository(root) != sources[name] or sha(library) != expected_sha:
                        result['status'] = 'inputs_changed'
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, ET.ParseError) as error:
                    result.update(status='runner_error', error=f'{type(error).__name__}: {error}')
                write(directory / 'result.json', result)
                print(json.dumps({key: result[key] for key in ('name', 'status', 'counts', 'error') if key in result}, sort_keys=True), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('Interrupted; descendants stopped. Use a fresh output for the next run.', flush=True)
    finally:
        passed = len(results) == len(plan) and all(row.get('status') in ('passed', 'passed_with_skips') for row in results)
        identity_errors = []
        try:
            for name, source in {**sources, **forks}.items():
                if repository(Path(source['root'])) != source:
                    identity_errors.append(f'Source changed during qualification: {name}')
            if sha(library) != expected_sha:
                identity_errors.append('Selected AMGX library changed during qualification')
            for row in results:
                runtime = row.get('runtime', {})
                if runtime.get('extension') and sha(runtime['extension']) != runtime['extension_sha256']:
                    identity_errors.append('Installed PyAMGX extension changed during qualification')
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            identity_errors.append(str(error))
        passed = passed and not identity_errors
        skipped = sum(row.get('counts', {}).get('skipped', 0) for row in results)
        status = 'passed_with_skips' if skipped else 'passed'
        write(output / 'completion.json', dict(status='interrupted' if interrupted else status if passed else 'failed',
              skipped=skipped, identity_errors=identity_errors,
              completed=datetime.now(timezone.utc).isoformat(), suites=results))
    return 130 if interrupted else 0 if passed else 1


try:
    raise SystemExit(main())
except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, ET.ParseError) as error:
    print(f'error: {error}', file=sys.stderr)
    raise SystemExit(2)
PY
