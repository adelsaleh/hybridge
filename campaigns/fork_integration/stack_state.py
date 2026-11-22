"""Source identities, artifact verification and stages for final stack consolidation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from campaigns.common.lifecycle import stop_on_signals
from campaigns.common.protocol import host_provenance
from campaigns.fork_integration.build_variant import run_command, sha256, write_json


def repository(root: Path) -> dict:
    """Record the actual committed checkout, including a frozen Git worktree."""
    def git(*args):
        """Read metadata from this checkout, not the workspace's similarly named repo."""
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"Commit source changes before running: {root}")
    return dict(root=str(root.resolve()), commit=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"))


def inputs(args) -> dict:
    """Bind native/Python source and build recipes, independently of later Hybridge docs."""
    forks = {name: repository(args.workspace / name) for name in ("AMGX", "pyamgx")}
    recipes = {name: sha256(REPO / "campaigns/fork_integration" / name)
               for name in ("build_primary.sh", "stack_state.py", "build_variant.py")}
    return dict(forks=forks, recipes=recipes, build=str(args.build.resolve()),
                install=str(args.install.resolve()), cuda=str(args.cuda.resolve()),
                nvcc_sha256=sha256(args.cuda / "bin/nvcc"),
                nvcc_version=subprocess.check_output([str(args.cuda / "bin/nvcc"), "--version"], text=True),
                host_compiler_sha256=sha256(Path("/usr/bin/c++").resolve()))


def verify_build(args) -> dict:
    """Require the recorded build, wheel and isolated extension to remain unchanged."""
    record = json.loads((args.stack / "BUILD.json").read_text())
    if record["inputs"] != inputs(args):
        raise RuntimeError("Build source/toolchain/recipe identity changed; use a new stack output")
    for name, expected in record["artifacts"].items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"Completed build artifact changed: {name}")
    return record


def baseline_manifest(args) -> dict:
    """Verify copied historical fixtures independently of the live primary runtime."""
    manifest = json.loads((args.baselines / "manifest.json").read_text())
    for row in manifest["copied"]:
        if sha256(args.baselines / row["copy"]) != row["sha256"]:
            raise RuntimeError(f"Historical baseline changed: {row['copy']}")
    return manifest


def prepare(args) -> None:
    """Create or resume the CPU build identity before any build/install operation."""
    current = inputs(args)
    hybridge = repository(REPO)
    args.stack.mkdir(parents=True, exist_ok=True)
    path = args.stack / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != current:
        raise RuntimeError("Stack inputs changed; choose a new HYBRIDGE_STACK_OUT")
    if not path.exists() and any(args.stack.iterdir()):
        raise RuntimeError("Stack output is nonempty without inputs.json")
    write_json(path, current)
    write_json(args.stack / "sources.json", dict(hybridge=hybridge, **current["forks"]))
    host_provenance(args.stack, stack_inputs=current, source_hybridge=hybridge)


def finish_build(args) -> None:
    """Hash every required install artifact and the exact wheel for later promotion."""
    current = inputs(args)
    if current != json.loads((args.stack / "inputs.json").read_text()):
        raise RuntimeError("Sources changed while building")
    artifacts = [args.build / name for name in ("CMakeCache.txt", "libamgx.a", "libamgxsh.so",
                 "src/amgx_tests_launcher", "examples/amgx_capi")]
    artifacts += [args.install / name for name in ("include/amgx_c.h", "include/amgx_config.h",
                  "lib/libamgx.a", "lib/libamgxsh.so", "lib/examples/amgx_capi")]
    wheels = list((args.stack / "wheelhouse").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one freshly built PyAMGX wheel")
    artifacts += wheels + [args.stack / "pyamgx-source.tar"]
    artifacts += [path for path in (args.stack / "python").rglob("*") if path.is_file()]
    extensions = list((args.stack / "python").glob("pyamgx*.so"))
    if len(extensions) != 1:
        raise RuntimeError("Missing isolated PyAMGX extension")
    dynamic = subprocess.check_output(["readelf", "-d", str(extensions[0])], text=True)
    if "(RUNPATH)" not in dynamic or str(args.build.resolve()) not in dynamic or "(RPATH)" in dynamic:
        raise RuntimeError("PyAMGX must use RUNPATH to the canonical primary build")
    for name in ("amgx_c.h", "amgx_config.h"):
        if sha256(args.workspace / "AMGX/include" / name) != sha256(args.install / "include" / name):
            raise RuntimeError(f"Installed header differs from merged source: {name}")
    write_json(args.stack / "BUILD.json", dict(inputs=current, wheel=str(wheels[0].resolve()),
               artifacts={str(path.resolve()): sha256(path) for path in artifacts}, extension_dynamic=dynamic))
    (args.stack / "COMPLETE").write_text("CPU build/install and isolated PyAMGX wheel complete; GPU acceptance pending.\n")


def qualification(args) -> None:
    """Bind resumable GPU stages to current harness, verified build and historical fixtures."""
    record, baseline = verify_build(args), baseline_manifest(args)
    if record["artifacts"][str((args.build / "libamgxsh.so").resolve())] == sha256(args.baselines / "main/libamgxsh.so"):
        raise RuntimeError("Candidate equals historical baseline; refusing self-comparison")
    current = dict(hybridge=repository(REPO), build_manifest_sha256=sha256(args.stack / "BUILD.json"),
                   baseline_manifest_sha256=sha256(args.baselines / "manifest.json"))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "identity.json"
    if path.exists() and json.loads(path.read_text()) != current:
        raise RuntimeError("Qualification inputs changed; choose a new HYBRIDGE_CAMPAIGN_OUT")
    if not path.exists() and any(args.output.iterdir()):
        raise RuntimeError("Qualification output is nonempty without identity.json")
    write_json(path, current)
    write_json(args.output / "sources.json", dict(hybridge=current["hybridge"], **record["inputs"]["forks"]))
    write_json(args.output / "baseline-sources.json", baseline)
    host_provenance(args.output, qualification=current, build_manifest=str(args.stack / "BUILD.json"),
                    gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True))


def stage(args) -> None:
    """Resume command-identical stages only with an intact log; cleanly stop children."""
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    log = args.output / (args.name + ".log")
    marker = args.output / (args.name + ".done.json")
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["command"] != command or previous["log_sha256"] != sha256(log):
            raise RuntimeError(f"Completed stage changed: {args.name}")
        print(f"Verified completed stage: {args.name}", flush=True)
        return
    print(f"Starting: {args.name}", flush=True)
    with stop_on_signals():
        run_command(command, Path.cwd(), log)
    write_json(marker, dict(command=command, log_sha256=sha256(log)))
    print("\n".join(log.read_text().splitlines()[-8:]), flush=True)


def main() -> int:
    """Expose only the small state transitions used by the two shell entry points."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "archive", "finish-build", "verify-build", "qualify", "stage"))
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument("--stack", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--install", type=Path)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda-13.1"))
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name")
    raw = sys.argv[1:]
    separator = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:separator])
    args.command = raw[separator + 1:]
    try:
        if args.action == "archive":
            record = json.loads((args.stack / "inputs.json").read_text())["forks"]["pyamgx"]
            archive = args.stack / "pyamgx-source.tar"
            subprocess.run(["git", "-C", record["root"], "archive", "--format=tar", record["commit"], "-o", str(archive)], check=True)
            destination = args.stack / "pyamgx-source"
            destination.mkdir(exist_ok=True)
            with tarfile.open(archive) as stream:
                stream.extractall(destination, filter="data")
        elif args.action == "verify-build":
            verify_build(args)
        else:
            {"prepare": prepare, "finish-build": finish_build, "qualify": qualification, "stage": stage}[args.action](args)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; resume with the same source/output identity.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
