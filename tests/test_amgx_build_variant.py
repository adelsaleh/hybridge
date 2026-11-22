"""Small host fixtures for audited incremental AMGX builds; no CUDA compilation."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from campaigns.fork_integration import build_variant as variant


def commit(root):
    """Commit fixture sources without using the developer's Git identity."""
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "fixture"], check=True, capture_output=True)
    return variant.git(root, "rev-parse", "HEAD").decode().strip()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Create a two-object baseline, committed candidate and harmless fake compiler."""
    for name in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
                 "COMPILER_PATH", "GCC_EXEC_PREFIX"):
        monkeypatch.delenv(name, raising=False)
    source, candidate, baseline = [tmp_path / name for name in ("primary", "candidate", "baseline")]
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    for name, content in {"src/a.cu": "a\n", "src/b.cu": "b\n", "include/shared.h": "header\n",
                          "src/CMakeLists.txt": "set(tests_all ${tests_all}\n tests/old.cu)\n"}.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    revision = commit(source)
    subprocess.run(["git", "-C", str(source), "worktree", "add", "-qb", "candidate", str(candidate)], check=True)
    (candidate / "src/a.cu").write_text("new a\n")
    commit(candidate)
    compiler = tmp_path / "compiler"
    compiler.write_text(f'''#!{sys.executable}
from pathlib import Path
import sys
a = sys.argv[1:]
if '--version' in a:
    print('fixture compiler 1')
else:
    out = Path(a[a.index('-o') + 1])
    inputs = [Path(a[a.index('-c') + 1])] if '-c' in a else [Path(x) for x in a if x.endswith('.o')]
    out.write_bytes(b''.join(p.read_bytes() for p in inputs))
    if '-MF' in a:
        Path(a[a.index('-MF') + 1]).write_text(str(out) + ': ' + str(inputs[0]) + '\\n')
''')
    compiler.chmod(0o755)
    target = baseline / "CMakeFiles/amgx_libs.dir"
    target.mkdir(parents=True)
    (target / "flags.make").write_text('CUDA_INCLUDES = --options-file CMakeFiles/amgx_libs.dir/includes_CUDA.rsp\n'
                                       'CUDA_FLAGS = -O3 -std=c++17 "--generate-code=arch=compute_90,code=[compute_90,sm_90]"\n')
    (target / "includes_CUDA.rsp").write_text(f'-I{source}/include\n')
    recipes, objects = [], []
    for unit in ("src/a.cu", "src/b.cu"):
        obj = f"CMakeFiles/amgx_libs.dir/{unit}.o"
        path = baseline / obj
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("baseline " + unit)
        Path(str(path) + ".d").write_text(f"{obj}: {source}/{unit} \\\n {source}/include/shared.h\n")
        recipes.append(f'\t{compiler} $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT {obj} -MF {obj}.d -x cu -c {source}/{unit} -o {obj}\n')
        objects.append(obj)
    (target / "build.make").write_text("".join(recipes))
    link = baseline / "CMakeFiles/amgxsh.dir"
    link.mkdir()
    (link / "objects1.rsp").write_text(" ".join(objects))
    (link / "link.txt").write_text(f'{compiler} -shared -o libamgxsh.so @CMakeFiles/amgxsh.dir/objects1.rsp\n')
    (baseline / "CMakeCache.txt").write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={source}\n')
    (baseline / "libamgxsh.so").write_text("qualified baseline")
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"AMGX": {"commit": revision, "files": variant.source_hashes(variant.tree_blobs(source, revision))}}))
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps(dict(source_manifest_local_path=str(manifest), source_manifest_sha256=variant.sha256(manifest),
                                              library_sha256=variant.sha256(baseline / "libamgxsh.so"), committed_forks={"AMGX": revision})))
    return Namespace(source=candidate, baseline=baseline, output=tmp_path / "output", qualification=qualification,
                     source_manifest=None, replace=["src/a.cu"], primary=source)


def test_plan_rewrites_candidate_includes_preserving_flags_and_other_objects(fixture):
    """Every declared source/header root is candidate-owned and plan writes nothing."""
    plan = variant.make_plan(fixture)
    argv = plan["compile"][0]["argv"]
    assert f"-I{fixture.source}/include" in argv
    assert not any(str(fixture.primary) in token for token in argv)
    assert "--generate-code=arch=compute_90,code=[compute_90,sm_90]" in argv
    assert str(fixture.baseline / "CMakeFiles/amgx_libs.dir/src/b.cu.o") in plan["link"]
    assert str(fixture.output / "CMakeFiles/amgx_libs.dir/src/a.cu.o") in plan["link"]
    assert not fixture.output.exists()


@pytest.mark.parametrize("path,match", [("src/b.cu", "Undeclared"), ("include/shared.h", "Changed library dependency"),
                                       ("src/CMakeLists.txt", "fresh CMake build")])
def test_undeclared_source_header_and_build_changes_are_rejected(fixture, path, match):
    """Object reuse cannot silently miss changed translation units or shared headers."""
    with (fixture.source / path).open("a") as stream:
        stream.write("changed\n")
    commit(fixture.source)
    with pytest.raises(variant.BuildError, match=match):
        variant.make_plan(fixture)


def test_independent_native_test_registration_is_audited(fixture):
    """Adding a separate native oracle does not change the library target."""
    (fixture.source / "src/tests").mkdir()
    (fixture.source / "src/tests/new.h").write_text("oracle\n")
    (fixture.source / "src/CMakeLists.txt").write_text("set(tests_all ${tests_all}\n tests/old.cu\n tests/new.cu)\n")
    commit(fixture.source)
    plan = variant.make_plan(fixture)
    assert {row["disposition"] for row in plan["changes"]} == {
        "replacement", "independent-native-test", "native-test-registration-only"}


@pytest.mark.parametrize("relative", ["include/shared.h", "libamgxsh.so", "src/a.cu"])
def test_dirty_or_changed_baseline_and_candidate_inputs_fail(fixture, relative):
    """Baseline sources/library and uncommitted candidate changes are not trusted."""
    root = fixture.primary if relative.startswith("include") else fixture.baseline if relative.endswith(".so") else fixture.source
    (root / relative).write_text("unexpected\n")
    with pytest.raises(variant.BuildError):
        variant.make_plan(fixture)


def test_build_resumes_only_hash_verified_objects_and_library(fixture):
    """A complete run resumes untouched; changed artifacts and different plans fail."""
    plan = variant.make_plan(fixture)
    variant.build(plan, resume=False)
    output = fixture.output / "libamgxsh.so"
    stamp = output.stat().st_mtime_ns
    variant.build(variant.make_plan(fixture), resume=True)
    assert output.stat().st_mtime_ns == stamp
    assert json.loads((fixture.output / "COMPLETE").read_text())["library_sha256"] == variant.sha256(output)
    with pytest.raises(variant.BuildError, match="--resume"):
        variant.build(plan, resume=False)
    Path(plan["compile"][0]["output"]).write_text("tampered object")
    with pytest.raises(variant.BuildError, match="Completed artifact changed"):
        variant.build(plan, resume=True)


def test_environment_cannot_inject_primary_headers_or_flags(fixture, monkeypatch):
    """The inherited shell cannot silently override the recorded CMake command."""
    monkeypatch.setenv("CPATH", str(fixture.primary / "include"))
    with pytest.raises(variant.BuildError, match="ambient compiler"):
        variant.make_plan(fixture)
