"""Deterministic Q1 production task registry; inspection only, never starts CUDA.

One bounded process per transport configuration or assembly comparison. Smoke
results, measured memory limits and a campaign runner are required before launch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shlex
import sys

from campaigns.common.production_placement import resources

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Task:
    id: str
    group: str
    arguments: tuple[str, ...]
    artifact: str
    timeout_seconds: int
    precision: str = "float64"
    mode: str = "timing"
    stdout_artifact: bool = False
    order: int | None = None
    stages: tuple[str, ...] = ("assembly",)
    unsupported_reason: str | None = None
    launcher: tuple[str, ...] = ()
    additional_artifacts: tuple[str, ...] = ()

    def command(self, output: Path):
        expand = lambda parts: [part.replace("{output}", str(output)) for part in parts]
        return [*expand(self.launcher), sys.executable, "-u", *expand(self.arguments)]

    def specification_sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def build_tasks():
    tasks = []
    transport = "scripts/gpu/benchmark_advection_tsle_tensor.py"
    for p in range(1, 10):
        for h in ("0.02", "0.0095"):
            for basis in ("legacy-lagrange", "legendre-modal"):
                common = (transport, "--mesh-size", h, "--orders", str(p), "--trace-basis", basis, "--repeats", "7")
                for precision in ("float64", "float32"):
                    key = f"ar-p{p}-h{h}-{basis}-native-{precision}"
                    artifact = f"transport/{key}.json"
                    tasks.append(Task(key, "transport", (*common, "--native", "fused,split3", "--configs", "none",
                        "--output-json", "{output}/" + artifact), artifact, 900, precision,
                        order=p, stages=("assembly", "reconstruction")))
                    for contraction in ("cutensor", "cublas"):
                        for solver in ("coop", "cublas", "magma"):
                            key = f"ar-p{p}-h{h}-{basis}-{precision}-{contraction}-{solver}"
                            artifact = f"transport/{key}.json"
                            tasks.append(Task(key, "transport", (*common, "--configs",
                                f"{precision}/{contraction}/default/{solver}", "--output-json", "{output}/" + artifact),
                                artifact, 900, order=p, stages=("assembly", "reconstruction")))
                            if solver == 'coop':
                                for block in ('32', '64', '128'):
                                    tuned_key = key + '-b' + block
                                    tuned_artifact = f'tensor-layouts/{tuned_key}.json'
                                    tasks.append(Task(tuned_key, 'tensor-layouts', (*common, '--configs',
                                        f'{precision}/{contraction}/default/{solver}', '--block-size', block,
                                        '--output-json', '{output}/' + tuned_artifact), tuned_artifact, 900,
                                        order=p, stages=('assembly', 'reconstruction')))
                            # FP32 tensor arithmetic uses an FP64 process and
                            # compares with an FP64 split3 reference in the driver.
                for block in ("32", "64", "128"):
                    key = f"ar-layout-p{p}-h{h}-{basis}-b{block}"
                    artifact = f"layouts/{key}.json"
                    tasks.append(Task(key, "transport-layouts", ("scripts/gpu/benchmark_advection_tsle_bsr.py",
                        "--case", "disk-tangent", "--mesh-size", h, "--orders", str(p), "--trace-bases", basis,
                        "--fused-block-size", block, "--allow-spills", "--repeats", "7", "--output-json", "{output}/" + artifact),
                        artifact, 900, order=p))
        # The older partially fused path forms local matrices with CuPy and
        # fuses LU/condensation/scatter in CUDA. It needs Dirichlet boundaries;
        # the zero-flux transport case above cannot select it.
        for nx in (16, 128):
            for basis in ("legacy-lagrange", "legendre-modal"):
                for fmt in ("coo",):
                    for block in (1, 32, 64, 128):
                        key = f"ar-partial-p{p}-n{nx}-{basis}-{fmt}-b{block}"
                        artifact = f"partial/{key}.json"
                        tasks.append(Task(key, "partial-transport", ("scripts/gpu/benchmark_fused_raw_assembly_kernels.py",
                            "--case", "transport", "--nx", str(nx), "--order", str(p), "--trace-basis", basis,
                            "--local-assembly", "precomputed", "--matrix-format", fmt, "--block-size", str(block),
                            "--repeats", "7"), artifact, 900, stdout_artifact=True, order=p,
                            unsupported_reason="precomputed transport kernel supports p<=6" if p > 6 else
                            # The harness profiles cooperative transport at 32/64/128 threads only.
                            "benchmark harness supports block sizes 32/64/128 for transport" if block == 1 else None))
                for path in ("fused", "split3"):
                    for fmt in (("coo", "csr", "bsr") if path == "fused" else ("bsr",)):
                        for block in (32, 64, 128):
                            key = f"ar-control-{path}-p{p}-n{nx}-{basis}-{fmt}-b{block}"
                            artifact = f"transport-controls/{key}.json"
                            tasks.append(Task(key, "transport-controls", ("scripts/gpu/benchmark_fused_raw_assembly_kernels.py",
                                "--case", "transport", "--nx", str(nx), "--order", str(p), "--trace-basis", basis,
                                "--local-assembly", path, "--matrix-format", fmt, "--block-size", str(block),
                                "--repeats", "7"), artifact, 900, stdout_artifact=True, order=p))
    for p in range(1, 10):
        for basis in ("legacy-lagrange", "legendre-modal"):
            key = f"dr-p{p}-{basis}"
            artifact = f"poisson/{key}.json"
            tasks.append(Task(key, "poisson", ("scripts/gpu/benchmark_poisson_assembly.py", "--orders", str(p),
                "--nx", "250", "--ny", "200", "--trace-basis", basis, "--warmups", "2", "--repeats", "5",
                "--output", "{output}/" + artifact), artifact, 900, order=p,
                stages=("assembly", "rhs_refresh", "reconstruction")))
            for nx in (16, 128):
                for fmt in ("coo", "csr", "bsr"):
                    for policy in ("none", "schur-lu", "schur-cholesky"):
                        key = f"dr-tune-p{p}-n{nx}-{basis}-{fmt}-{policy}"
                        artifact = f"poisson-kernels/{key}.json"
                        tasks.append(Task(key, "poisson-kernels", ("scripts/gpu/tune_poisson_raw_cuda.py",
                            "--orders", str(p), "--mesh-sizes", str(nx), "--trace-bases", basis, "--formats", fmt,
                            "--cache-policies", policy, "--thorough", "--include-legacy", "--repeats", "7",
                            "--output", "{output}/" + artifact), artifact, 1800, order=p,
                            stages=("assembly", "rhs_refresh", "reconstruction")))
        for nx in (16, 128):
            for coefficient in ("scalar", "constant-full", "variable-full"):
                for policy in ("none", "schur-lu", "schur-lu+mass"):
                    key = f"adr-p{p}-n{nx}-{coefficient}-{policy}"
                    artifact = f"adr/{key}.jsonl"
                    tasks.append(Task(key, "adr", ("scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_raw_cuda.py",
                        "--nx", str(nx), "--orders", str(p), "--coefficients", coefficient,
                        "--bases", "legacy-lagrange", "legendre-modal", "--matrix-formats", "coo", "csr", "bsr",
                        "--block-sizes", "1", "32", "64", "128", "--cache-local-factors", policy,
                        "--warmup", "3", "--repeats", "7", "--reconstruction", "--output", "{output}/" + artifact),
                        artifact, 1800, order=p, stages=("assembly", "reconstruction")))
        for equation in ("poisson", "adr"):
            coefficients = ("scalar",) if equation == "poisson" else ("scalar", "constant-full", "variable-full")
            policies = ("none", "schur-lu", "schur-cholesky") if equation == "poisson" else ("none", "schur-lu", "schur-lu+mass")
            for nx in (16, 128):
                for basis in ("legacy-lagrange", "legendre-modal"):
                    for coefficient in coefficients:
                        for policy in policies:
                            for flux in (("l2_closest",) if equation == "poisson" else ("l2_closest", "RT_projection")):
                                key = f"{equation}-recovery-p{p}-n{nx}-{basis}-{coefficient}-{policy}-{flux}"
                                artifact = f"recovery/{key}.json"
                                tasks.append(Task(key, "recovery", ("-m", "scripts.gpu.benchmark_hdg_recovery",
                                    "--equation", equation, "--order", str(p), "--nx", str(nx), "--trace-basis", basis,
                                    "--coefficient", coefficient, "--cache-policy", policy, "--flux-space", flux,
                                    "--output", "{output}/" + artifact), artifact, 1800, order=p,
                                    stages=(("reconstruction", "primal_postprocessing") if equation == "poisson" else
                                            ("reconstruction", "flux_postprocessing", "primal_postprocessing"))))
        for nx in (16, 128):
            key = f"tensor-primal-p{p}-n{nx}"
            artifact = f"postprocessing/{key}.jsonl"
            tasks.append(Task(key, "postprocessing", ("-m",
                "scripts.advection_diffusion_reaction.diagnostics.benchmark_tensor_postprocessing", "--orders", str(p),
                "--nx", str(nx), "--repeats", "7", "--output", "{output}/" + artifact), artifact, 1800,
                order=p, stages=("primal_postprocessing",)))
    profiler = "scripts/gpu/capture_raw_assembly_profiles.py"
    for p in range(1, 10):
        for phase in ("assembly", "reuse"):
            key = f"profile-p{p}-{phase}"
            options = (("--cases", "poisson", "transport", "adr", "--formats", "bsr", "--cache-policies", "none",
                        "--phases", "assembly", "--coefficient-cases", "constant", "variable") if phase == "assembly" else
                       ("--cases", "poisson", "--formats", "csr", "bsr", "--cache-policies", "schur-lu", "schur-cholesky",
                        "--phases", "cache", "rhs", "reconstruction"))
            artifact = f"profiles/{key}/manifest-both.json"
            tasks.append(Task(key, "profiles", (profiler, "--mode", "both", "--orders", str(p), "--sizes", "128",
                "--bases", "legendre-modal", "--blocks", "128", "--repeats", "5", *options,
                "--output", "{output}/profiles/" + key), artifact, 1800,
                mode="profile_and_separate_baseline", order=p, stages=("assembly", "rhs_refresh", "reconstruction")))
    for p in range(1, 10):
        for equation, fluxes in (('poisson', ('l2_closest',)), ('adr', ('l2_closest', 'RT_projection'))):
            for flux in fluxes:
                key = f'profile-recovery-{equation}-p{p}-{flux}'
                artifact = f'profiles/{key}.json'
                tasks.append(Task(key, 'recovery-profiles', ('-m', 'scripts.gpu.benchmark_hdg_recovery',
                    '--equation', equation, '--order', str(p), '--nx', '32', '--flux-space', flux,
                    '--coefficient', 'scalar' if equation == 'poisson' else 'variable-full', '--cache-policy', 'schur-lu',
                    '--repeats', '1', '--nvtx', '--output', '{output}/' + artifact), artifact, 1800,
                    mode='profile', order=p, stages=('reconstruction', 'postprocessing'),
                    launcher=_ncu(key, 'hybridge_recovery/'), additional_artifacts=(f'profiles/{key}.ncu-rep',)))
        for contraction in ('cutensor', 'cublas'):
            for local in ('coop', 'cublas', 'magma'):
                key = f'profile-tensor-p{p}-{contraction}-{local}'
                artifact = f'profiles/{key}.json'
                tasks.append(Task(key, 'tensor-profiles', (transport, '--mesh-size', '0.02', '--orders', str(p),
                    '--configs', f'float64/{contraction}/default/{local}', '--repeats', '1', '--nvtx',
                    '--output-json', '{output}/' + artifact), artifact, 1800, mode='profile', order=p,
                    stages=('assembly', 'reconstruction'), launcher=_ncu(key, 'hybridge_transport/'),
                    additional_artifacts=(f'profiles/{key}.ncu-rep',)))
    return tasks


def _ncu(key, nvtx):
    return ('ncu', '--target-processes', 'all', '--nvtx', '--nvtx-include', nvtx,
            '--section', 'SpeedOfLight', '--section', 'LaunchStats', '--section', 'Occupancy',
            '--section', 'WarpStateStats', '--section', 'MemoryWorkloadAnalysis_Tables',
            '--export', '{output}/profiles/' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "run_outputs/campaigns/h100_assembly/production")
    parser.add_argument("--group", choices=("transport", "transport-layouts", "partial-transport", "transport-controls", "poisson",
                                          "poisson-kernels", "adr", "recovery", "postprocessing", "profiles",
                                          "tensor-layouts", "tensor-profiles", "recovery-profiles"))
    parser.add_argument("--order", type=int, choices=range(1, 10))
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()
    tasks = [task for task in build_tasks() if (not args.group or task.group == args.group)
             and (args.order is None or task.order == args.order)]
    output = args.output_root.resolve()
    if args.format == "shell":
        for task in tasks:
            print(f"# {task.id}; timeout {task.timeout_seconds}s; artifact {task.artifact}")
            print(shlex.join([f"HYBRIDGE_PRECISION={task.precision}", *task.command(output)]))
    else:
        print(json.dumps(dict(schema_version=1, campaign="h100_assembly", launch_ready=False,
            resources=resources("device"), host_benchmarks_inline=False,
            runner='campaigns.h100_assembly.run',
            required_before_launch=["passing original smoke", "passing expanded-kernel pilot", "live resource checks"],
            output_root=str(output), groups=dict(Counter(task.group for task in tasks)),
            reference_dossier_budget_seconds=10800, scope="expanded H200 study authorized 2026-10-26",
            budget_note="Expanded beyond the dossier matrix. Timeouts are caps, not duration estimates; costs require measurements.",
            tasks=[dict(asdict(task), specification_sha256=task.specification_sha256(), command=task.command(output))
                   for task in tasks]), indent=2))


if __name__ == "__main__":
    main()
