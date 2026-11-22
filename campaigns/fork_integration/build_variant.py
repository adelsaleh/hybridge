"""Build declared AMGX CUDA replacements against a qualified object baseline.

This deliberately supports the existing Unix Makefiles AMGX library target,
not arbitrary CMake projects. Configuration/header changes require a fresh
CMake build; adding an independent native test is allowed and recorded.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from campaigns.common.lifecycle import stop_on_signals, terminate_group
from campaigns.common.protocol import host_provenance, log


class BuildError(RuntimeError):
    """A source, baseline or resume contract could not be established."""


def sha256(path: Path) -> str:
    """Hash a file without retaining large objects or libraries in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value) -> str:
    """Hash canonical JSON for a reproducible plan identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git(source: Path, *args: str) -> bytes:
    """Read Git data without changing the candidate worktree or index."""
    return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.PIPE)


def clean_commit(source: Path) -> str:
    """Require the exact committed candidate, including no untracked additions."""
    status = git(source, "status", "--porcelain", "--untracked-files=all").decode()
    if status:
        raise BuildError(f"Candidate worktree must be clean and committed:\n{status}")
    return git(source, "rev-parse", "HEAD").decode().strip()


def tree_blobs(source: Path, revision: str) -> dict[str, bytes]:
    """Read a complete committed tree in one cat-file call, retaining blob bytes."""
    entries = []
    for row in git(source, "ls-tree", "-r", "-z", revision).split(b"\0"):
        if row:
            meta, name = row.split(b"\t", 1)
            mode, kind, oid = meta.split()
            if kind != b"blob" or mode == b"120000":
                raise BuildError(f"Source tree contains unsupported symlink/submodule: {name.decode()}")
            entries.append((name.decode(), oid))
    data = subprocess.check_output(["git", "-C", str(source), "cat-file", "--batch"],
                                   input=b"".join(oid + b"\n" for _, oid in entries))
    blobs, offset = {}, 0
    for name, _ in entries:
        end = data.index(b"\n", offset)
        size = int(data[offset:end].split()[2])
        blobs[name] = data[end + 1:end + 1 + size]
        offset = end + size + 2
    return blobs


def source_hashes(blobs: dict[str, bytes]) -> dict[str, str]:
    """Describe every committed file, including headers omitted by old manifests."""
    return {name: hashlib.sha256(value).hexdigest() for name, value in sorted(blobs.items())}


def parse_make_variables(path: Path) -> dict[str, str]:
    """Read simple CMake-generated assignments; unsupported make syntax fails later."""
    return dict(re.findall(r"^(\w+) = (.*)$", path.read_text(), flags=re.MULTILINE))


def dependency_paths(path: Path, baseline: Path) -> list[Path]:
    """Parse nvcc's single-rule dependency file, preserving escaped path spaces."""
    text = path.read_text().replace("\\\n", " ")
    if ":" not in text:
        raise BuildError(f"Invalid dependency file: {path}")
    return [(baseline / value).resolve() for value in shlex.split(text.split(":", 1)[1])]


def tests_only_cmake_change(before: bytes, after: bytes) -> bool:
    """Accept only tests_all list additions/removals under src/tests, not configuration."""
    pattern = r"set\(tests_all\s+\$\{tests_all\}([^)]*)\)"
    old, new = re.search(pattern, before.decode()), re.search(pattern, after.decode())
    if old is None or new is None:
        return False
    old_items, new_items = shlex.split(old[1]), shlex.split(new[1])
    different = set(old_items) ^ set(new_items)
    return (bool(different) and all(re.fullmatch(r"tests/[\w./-]+\.cu", item) for item in different)
            and re.sub(pattern, "TESTS", before.decode(), count=1)
            == re.sub(pattern, "TESTS", after.decode(), count=1))


def audit_changes(before: dict[str, bytes], after: dict[str, bytes], replacements: set[str],
                  units: set[str], dependencies: dict[str, set[str]]) -> list[dict]:
    """Reject unexplained library/build changes and record independent tests/docs."""
    changes = []
    for name in sorted(before.keys() | after.keys()):
        if before.get(name) == after.get(name):
            continue
        affected = sorted(unit for unit, deps in dependencies.items() if name in deps)
        if name in units:
            if name not in replacements or name not in after:
                raise BuildError(f"Undeclared or removed library source: {name}")
            disposition = "replacement"
        elif name == "src/CMakeLists.txt" and tests_only_cmake_change(before.get(name, b""), after.get(name, b"")):
            disposition = "native-test-registration-only"
        elif affected:
            raise BuildError(f"Changed library dependency {name} affects {affected}; use a fresh CMake build")
        elif name.startswith("src/tests/") and Path(name).suffix in {".cu", ".h", ".cuh", ".hpp"}:
            disposition = "independent-native-test"
        elif (name.startswith(("include/", "external/")) or Path(name).suffix in {".h", ".hpp", ".cuh", ".cmake"}
              or Path(name).name == "CMakeLists.txt" or name.startswith("src/")):
            raise BuildError(f"Changed source/header/configuration needs a fresh CMake build: {name}")
        else:
            disposition = "outside-library-target"
        changes.append(dict(path=name, disposition=disposition, dependent_units=affected))
    return changes


def rewrite_source_token(token: str, primary: Path, candidate: Path) -> str:
    """Replace baseline source roots inside nvcc path arguments, never their flags."""
    return re.sub(re.escape(str(primary)) + r"(?=/|$)", lambda _: str(candidate), token)


def expand_options(argv: list[str], baseline: Path, inputs: dict[str, str]) -> list[str]:
    """Inline nvcc/g++ response files so every effective flag is auditable."""
    result, i = [], 0
    while i < len(argv):
        item = argv[i]
        if item in {"--options-file", "-optf"}:
            i += 1
            paths = argv[i].split(",")
        elif item.startswith("@"):
            paths = [item[1:]]
        else:
            result.append(item)
            i += 1
            continue
        for name in paths:
            path = (baseline / name).resolve()
            inputs[str(path)] = sha256(path)
            result.extend(expand_options(shlex.split(path.read_text()), baseline, inputs))
        i += 1
    return result


def tool_record(executable: str) -> dict:
    """Record the actual compiler executable and its reported toolchain version."""
    path = Path(shutil.which(executable) or executable).resolve()
    report = subprocess.check_output([str(path), "--version"], stderr=subprocess.STDOUT, text=True)
    return dict(path=str(path), sha256=sha256(path), version=report)


def make_plan(args) -> dict:
    """Verify the qualified baseline and derive commands without writing any output."""
    source, baseline, output = args.source.resolve(), args.baseline.resolve(), args.output.resolve()
    injected = {name: os.environ[name] for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CPATH",
                "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX")
                if os.environ.get(name)}
    if injected:
        raise BuildError(f"Unset ambient compiler flag/include overrides before building: {sorted(injected)}")
    if output == baseline or baseline in output.parents or source == output or source in output.parents:
        raise BuildError("Output must be separate from the baseline and candidate source trees")
    qualification = json.loads(args.qualification.read_text())
    manifest_path = (args.source_manifest or Path(qualification["source_manifest_local_path"])).resolve()
    if sha256(manifest_path) != qualification["source_manifest_sha256"]:
        raise BuildError("Qualified source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())["AMGX"]
    baseline_commit = qualification["committed_forks"]["AMGX"]
    commit = clean_commit(source)
    before, after = tree_blobs(source, baseline_commit), tree_blobs(source, commit)
    before_hashes, after_hashes = source_hashes(before), source_hashes(after)
    for name, expected in manifest["files"].items():
        if before_hashes.get(name) != expected:
            raise BuildError(f"Qualified baseline commit disagrees with source manifest: {name}")
    for name, expected in after_hashes.items():
        if not (source / name).is_file() or sha256(source / name) != expected:
            raise BuildError(f"Candidate file differs from its committed blob: {name}")
    library = baseline / "libamgxsh.so"
    if sha256(library) != qualification["library_sha256"]:
        raise BuildError("Qualified baseline library hash mismatch")
    cache = baseline / "CMakeCache.txt"
    home = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", cache.read_text(), re.MULTILINE)
    if home is None:
        raise BuildError("Baseline CMake cache has no source root")
    primary = Path(home[1]).resolve()
    target = baseline / "CMakeFiles/amgx_libs.dir"
    flags, recipes = target / "flags.make", target / "build.make"
    link_file = baseline / "CMakeFiles/amgxsh.dir/link.txt"
    inputs = {str(path): sha256(path) for path in (cache, flags, recipes, link_file, manifest_path, args.qualification.resolve())}
    variables = parse_make_variables(flags)
    expanded_link = expand_options(shlex.split(link_file.read_text()), baseline, inputs)
    object_names = [name for name in expanded_link if name.endswith(".o")]
    if not object_names or len(set(object_names)) != len(object_names):
        raise BuildError("Expected a nonempty unique baseline library object list")
    units, dependencies, objects, external = {}, {}, {}, {}
    for name in object_names:
        relative = Path(name)
        prefix = "CMakeFiles/amgx_libs.dir/"
        if not name.startswith(prefix) or not name.endswith(".cu.o"):
            raise BuildError(f"Unsupported baseline library object: {name}")
        unit = name[len(prefix):-2]
        units[unit] = name
        if unit not in manifest["files"]:
            raise BuildError(f"Compiled source absent from qualification manifest: {unit}")
        depfile = baseline / (name + ".d")
        inputs[str(depfile)] = sha256(depfile)
        dependencies[unit] = set()
        for path in dependency_paths(depfile, baseline):
            if path.is_relative_to(primary):
                dep = str(path.relative_to(primary))
                if dep not in before_hashes:
                    raise BuildError(f"Baseline dependency absent from committed source: {dep}")
                if not path.is_file() or sha256(path) != before_hashes[dep]:
                    raise BuildError(f"Baseline dependency changed since qualification: {path}")
                dependencies[unit].add(dep)
            elif str(path) not in external:
                external[str(path)] = sha256(path)
        objects[str((baseline / relative).resolve())] = sha256(baseline / relative)
    replacements = set(args.replace)
    if not replacements or not replacements <= units.keys():
        raise BuildError(f"Declare existing library CUDA units with --replace; unknown={sorted(replacements - units.keys())}")
    changes = audit_changes(before, after, replacements, set(units), dependencies)
    commands = []
    for unit in sorted(replacements):
        object_name = units[unit]
        lines = [line.strip() for line in recipes.read_text().splitlines()
                 if line.startswith("\t") and f" -o {object_name}" in line and " -c " in line]
        if len(lines) != 1:
            raise BuildError(f"Expected one exact baseline compile recipe for {unit}")
        command = lines[0]
        for key, value in variables.items():
            command = command.replace(f"$({key})", value)
        if "$" in command or any(token in command for token in ("&&", ";", "|")):
            raise BuildError(f"Unsupported shell/make syntax in compile recipe for {unit}")
        argv = expand_options(shlex.split(command), baseline, inputs)
        argv = [rewrite_source_token(token, primary, source) for token in argv]
        for index, token in enumerate(argv[:-1]):
            if token in {"-o", "-MF", "-MT"}:
                argv[index + 1] = str(output / argv[index + 1])
        if argv[argv.index("-c") + 1] != str(source / unit):
            raise BuildError(f"Compile command does not use candidate source: {unit}")
        commands.append(dict(unit=unit, argv=argv, output=str(output / object_name),
                             log=str(output / "logs" / (unit + ".log"))))
    replacements_by_object = {units[unit]: str(output / units[unit]) for unit in replacements}
    linked = [replacements_by_object.get(token, str((baseline / token).resolve())) if token in object_names else token
              for token in expanded_link]
    linked[linked.index("-o") + 1] = str(output / "libamgxsh.so")
    tools = {executable: tool_record(executable) for executable in sorted({cmd["argv"][0] for cmd in commands} | {linked[0]})}
    identity = dict(schema=1, helper_sha256=sha256(Path(__file__)), source=str(source), source_commit=commit,
                    source_clean=True, source_hashes=after_hashes, baseline=str(baseline), output=str(output),
                    baseline_commit=baseline_commit, manifest_recorded_commit=manifest["commit"],
                    baseline_source_hashes=before_hashes, baseline_library_sha256=sha256(library),
                    inputs=inputs, baseline_objects=objects, external_dependencies=external,
                    changes=changes, dependency_closure={unit: sorted(deps) for unit, deps in dependencies.items()},
                    compile=commands, link=linked, tools=tools)
    return dict(identity=digest(identity), **identity)


def write_json(path: Path, value) -> None:
    """Atomically replace a small provenance or completion record."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_inputs(plan: dict) -> None:
    """Reject mutable source, baseline, flags or dependencies before and after work."""
    source = Path(plan["source"])
    if clean_commit(source) != plan["source_commit"]:
        raise BuildError("Candidate source commit changed during build")
    inventory = {**plan["inputs"], **plan["baseline_objects"], **plan["external_dependencies"],
                 **{str(source / name): value for name, value in plan["source_hashes"].items()},
                 str(Path(plan["baseline"]) / "libamgxsh.so"): plan["baseline_library_sha256"],
                 **{info["path"]: info["sha256"] for info in plan["tools"].values()}}
    for name, expected in inventory.items():
        if sha256(Path(name)) != expected:
            raise BuildError(f"Build input changed: {name}")


def run_command(argv: list[str], cwd: Path, logfile: Path) -> None:
    """Run a compiler/linker with shared signal cleanup and a persistent command log."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w") as stream:
        stream.write(shlex.join(argv) + "\n")
        stream.flush()
        process = subprocess.Popen(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
        finally:
            terminate_group(process)
    if code:
        raise BuildError(f"Compiler/linker exited {code}; see {logfile}")


def compiled_dependencies(stage: dict, plan: dict) -> dict[str, str]:
    """Verify actual compiler inputs, rejecting primary or uncommitted candidate headers."""
    if stage["unit"] == "link":
        return {}
    candidate = Path(plan["source"])
    primary_cache = Path(plan["baseline"]) / "CMakeCache.txt"
    primary = Path(re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", primary_cache.read_text(), re.MULTILINE)[1]).resolve()
    result = {}
    for path in dependency_paths(Path(stage["output"] + ".d"), Path(plan["baseline"])):
        value = sha256(path)
        if path.is_relative_to(candidate):
            if plan["source_hashes"].get(str(path.relative_to(candidate))) != value:
                raise BuildError(f"Compiler read an uncommitted/changed candidate dependency: {path}")
        elif path.is_relative_to(primary):
            raise BuildError(f"Compiler read primary checkout instead of candidate: {path}")
        result[str(path)] = value
    return result


def build(plan: dict, resume: bool) -> None:
    """Run or resume only hash-verified stages of this exact build identity."""
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a") as lock, stop_on_signals():
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BuildError(f"Another builder holds {output}") from error
        previous = output / "plan.json"
        if previous.exists():
            if not resume or json.loads(previous.read_text())["identity"] != plan["identity"]:
                raise BuildError("Existing output requires --resume with unchanged source, flags and inputs")
        elif any(path.name != ".build.lock" for path in output.iterdir()):
            raise BuildError("Refusing a nonempty output directory without matching plan.json")
        write_json(previous, plan)
        host_provenance(output, build_identity=plan["identity"], amgx_commit=plan["source_commit"],
                        amgx_clean=True, baseline_commit=plan["baseline_commit"], toolchains=plan["tools"])
        # Runtime wrappers read only toolkit selection here; this is not a new CMake tree.
        shutil.copyfile(Path(plan["baseline"]) / "CMakeCache.txt", output / "CMakeCache.txt")
        verify_inputs(plan)
        stages = list(plan["compile"]) + [dict(unit="link", argv=plan["link"], output=str(output / "libamgxsh.so"),
                                              log=str(output / "logs/link.log"))]
        for stage in stages:
            artifact = Path(stage["output"])
            marker = artifact.with_name(artifact.name + ".done.json")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if marker.exists():
                record = json.loads(marker.read_text())
                if (record["identity"] != plan["identity"] or not artifact.is_file()
                        or sha256(artifact) != record["sha256"]):
                    raise BuildError(f"Completed artifact changed: {artifact}; choose a new output")
                for name, expected in record["dependencies"].items():
                    if sha256(Path(name)) != expected:
                        raise BuildError(f"Completed stage dependency changed: {name}")
                log(f"Verified completed stage: {stage['unit']}")
                continue
            log(f"Starting: {stage['unit']}")
            started = time.monotonic()
            run_command(stage["argv"], Path(plan["baseline"]), Path(stage["log"]))
            verify_inputs(plan)
            write_json(marker, dict(identity=plan["identity"], sha256=sha256(artifact),
                                    elapsed_seconds=time.monotonic() - started, argv=stage["argv"],
                                    dependencies=compiled_dependencies(stage, plan)))
        verify_inputs(plan)
        write_json(output / "COMPLETE", dict(identity=plan["identity"], source_commit=plan["source_commit"],
                                             library_sha256=sha256(output / "libamgxsh.so"),
                                             objects={stage["output"]: sha256(Path(stage["output"]))
                                                      for stage in plan["compile"]}))
        log(f"Build complete: {output / 'libamgxsh.so'}; numerical qualification remains separate")


def main(argv=None) -> int:
    """Expose a read-only plan and an explicit resumable incremental build."""
    parser = argparse.ArgumentParser(description=__doc__)
    builds = Path(os.environ.get("HYBRIDGE_BUILDS_ROOT", REPO.parent / "builds"))
    parser.add_argument("--source", type=Path, required=True, help="Clean committed AMGX worktree")
    parser.add_argument("--output", type=Path, required=True, help="New immutable variant build directory")
    parser.add_argument("--baseline", type=Path, default=builds / "AMGX/cuda13.1-integration")
    parser.add_argument("--qualification", type=Path, default=REPO / "docs/campaigns/fork_integration/qualification.json")
    parser.add_argument("--source-manifest", type=Path, help="Relocated sources.json with the qualified hash")
    parser.add_argument("--replace", action="append", required=True, help="Library CUDA unit relative to AMGX root; repeatable")
    parser.add_argument("--plan", action="store_true", help="Validate and print commands/provenance; write nothing")
    parser.add_argument("--resume", action="store_true", help="Resume matching plan and verify completed artifacts")
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args)
        if args.plan:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            build(plan, args.resume)
    except (BuildError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Build interrupted; resume with the same inputs and --resume.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
