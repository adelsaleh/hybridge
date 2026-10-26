"""Every documented guiding-center command stays valid.

``scripts/guiding_center/example_runs.md`` is the canonical command list; the study and campaign
READMEs repeat the commands of their study. For every ``python ...`` command in their fenced code
blocks this test checks that the module or script exists, that every ``--option`` is accepted (by
the script's parser when it exposes one, else by its ``--help`` text), that ``--preset`` and
``--case`` values of the case CLI exist, and that every ``@run_configs/...`` file exists. Nothing
is solved: the only subprocesses are ``--help`` calls.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GC = ROOT / "scripts" / "guiding_center"
SOURCES = [GC / "example_runs.md", GC / "README.md", *sorted(GC.glob("studies/*/README.md")),
           *sorted((ROOT / "campaigns").glob("*/README.md")), ROOT / "campaigns" / "README.md"]
FENCE = re.compile(r"```(?:bash|sh|shell|console)?\n(.*?)```", re.DOTALL)
PYTHON_NAMES = {"python", "python3", ".venv/bin/python", "../hybridge/.venv/bin/python", "$PY", "${PY}", "$PYTHON"}
ENV_ASSIGNMENT = re.compile(r"^[A-Z_][A-Z0-9_]*=")
PARSER_FACTORIES = ("build_parser", "make_parser", "create_parser", "get_parser")


def _commands():
    """Yield (source, line, env, tokens) for every python command in the documentation sources."""
    for source in SOURCES:
        if not source.exists():
            continue
        for block in FENCE.findall(source.read_text(encoding="utf-8")):
            text = block.replace("\\\n", " ")
            for raw in text.splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    tokens = shlex.split(line, comments=True)
                except ValueError:
                    continue
                for segment in _split_on_dashes(tokens):
                    env = {}
                    while segment and (segment[0] == "env" or ENV_ASSIGNMENT.match(segment[0])):
                        if segment[0] != "env":
                            key, value = segment[0].split("=", 1)
                            env[key] = value
                        segment = segment[1:]
                    if not segment:
                        continue
                    if segment[0] in PYTHON_NAMES or segment[0].endswith("/python"):
                        yield source, line, env, segment[1:]
                    elif segment[0].endswith(".sh") or segment[0].endswith(".py"):
                        yield source, line, env, segment


def _split_on_dashes(tokens):
    """``a -m x -- b -m y`` runs two commands (holoviz_ssh wrapper); split at a standalone ``--``."""
    parts, current = [], []
    for token in tokens:
        if token == "--":
            parts.append(current)
            current = []
        else:
            current.append(token)
    parts.append(current)
    return [part for part in parts if part]


def _target(tokens):
    """(kind, module_or_path, args) of a python command; kind is 'module', 'script' or None."""
    i = 0
    while i < len(tokens) and tokens[i] in {"-u", "-B", "-I", "-O"}:
        i += 1
    if i < len(tokens) and tokens[i] == "-m" and i + 1 < len(tokens):
        return "module", tokens[i + 1], tokens[i + 2:]
    if i < len(tokens) and (tokens[i].endswith(".py") or tokens[i].endswith(".sh")):
        return "script", tokens[i], tokens[i + 1:]
    return None, None, tokens


def _module_of(kind, name):
    if kind == "module":
        path = ROOT / (name.replace(".", "/") + ".py")
        return name, path
    path = ROOT / name
    if path.suffix == ".py":
        return ".".join(path.relative_to(ROOT).with_suffix("").parts), path
    return None, path


def _options(args):
    return sorted({token.split("=", 1)[0] for token in args if token.startswith("--") and len(token) > 2})


_HELP_CACHE: dict[tuple, str] = {}


def _help_text(module, env):
    key = (module, tuple(sorted(env.items())))
    if key not in _HELP_CACHE:
        run_env = {**os.environ, "MPLBACKEND": "Agg", **env}
        for attempt in range(2):  # a second, serial attempt after a failure among the parallel prefetches
            result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, env=run_env,
                                    capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                _HELP_CACHE[key] = result.stdout + result.stderr
                break
            _HELP_CACHE[key] = ""
            _HELP_ERRORS[key] = result.stderr[-2000:]
    return _HELP_CACHE[key]


_HELP_ERRORS: dict[tuple, str] = {}


def _parser(module_name):
    """An ArgumentParser of the module when it exposes a factory, else None (import failures -> None)."""
    import importlib

    try:
        module = importlib.import_module(module_name)
    except BaseException:  # noqa: BLE001 - scripts that refuse to import without NUMBA_DISABLE_JIT, etc.
        return None
    for name in PARSER_FACTORIES:
        factory = getattr(module, name, None)
        if callable(factory):
            try:
                parser = factory()
            except BaseException:  # noqa: BLE001
                return None
            if hasattr(parser, "parse_args"):
                return parser
    return None


def _collect():
    cases = []
    for source, line, env, tokens in _commands():
        kind, name, args = _target(tokens)
        if kind is None:
            continue
        module, path = _module_of(kind, name)
        if not (str(path).startswith(str(GC)) or str(path).startswith(str(ROOT / "campaigns"))):
            continue  # hybridge.io.holoviz_ssh and friends are not guiding-center commands
        cases.append(pytest.param(source, line, env, module, path, args, id=f"{source.name}:{line[:60]}"))
    return cases


CASES = _collect()


@pytest.fixture(scope="module", autouse=True)
def _prefetch_help():
    """Run the --help subprocesses of every module without a parser factory in parallel, once."""
    jobs = {}
    for case in CASES:
        _, _, env, module, path, _ = case.values
        if module and path.suffix == ".py" and str(path).startswith(str(GC)) and _parser(module) is None:
            jobs[(module, tuple(sorted(env.items())))] = env
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: _help_text(item[0][0], item[1]), jobs.items()))


@pytest.mark.parametrize("source, line, env, module, path, args", CASES)
def test_documented_command_is_valid(source, line, env, module, path, args):
    assert path.is_file(), f"{source.name}: {path.relative_to(ROOT)} does not exist"
    for token in args:
        if token.startswith("@"):
            assert (ROOT / token[1:]).is_file(), f"{source.name}: {token} does not exist"
    if path.suffix != ".py":
        return
    if not str(path).startswith(str(GC)):
        return  # campaign scripts: existence only (their subcommands are checked by their own tests)
    parser = _parser(module)
    if parser is not None:
        if "--help" in args:
            return
        if module == "scripts.guiding_center.run_guiding_center_cases":
            parser = _case_cli_parser()
        try:
            namespace = parser.parse_args(args)
        except SystemExit as error:
            pytest.fail(f"{source.name}: the parser of {module} rejects\n  {line}\n  (exit {error.code})")
        if module == "scripts.guiding_center.run_guiding_center_cases":
            _check_case_cli(namespace, source, line)
        return
    help_text = _help_text(module, env)
    assert help_text, (f"{source.name}: `python -m {module} --help` failed:\n"
                       + _HELP_ERRORS.get((module, tuple(sorted(env.items()))), ""))
    for option in _options(args):
        assert re.search(re.escape(option) + r"(?![A-Za-z0-9_-])", help_text), (
            f"{source.name}: {module} does not accept {option}\n  {line}")
    if args and re.fullmatch(r"[a-z][a-z0-9-]*", args[0]) and "{" in help_text:  # leading subcommand
        assert args[0] in help_text, f"{source.name}: {module} has no subcommand {args[0]!r}"


def _case_cli_parser():
    from scripts.guiding_center.runtime.arguments import build_parser

    return build_parser()


def _check_case_cli(namespace, source, line):
    from scripts.guiding_center.cases.guiding_center_cases import CASE_DEFINITIONS
    from scripts.guiding_center.cases.guiding_center_presets import PRESETS

    preset = getattr(namespace, "preset", None)
    case = getattr(namespace, "case", None)
    if preset:
        assert preset in PRESETS, f"{source.name}: unknown preset {preset!r} in\n  {line}"
    if case:
        assert case in CASE_DEFINITIONS, f"{source.name}: unknown case {case!r} in\n  {line}"


def test_example_runs_lists_every_study():
    """example_runs.md has a section for every study directory and for the campaigns."""
    text = (GC / "example_runs.md").read_text(encoding="utf-8")
    for study in sorted(p.name for p in (GC / "studies").iterdir() if p.is_dir() and not p.name.startswith("_")):
        assert f"`studies/{study}/`" in text, f"example_runs.md has no section for studies/{study}"
    assert "## Campaigns" in text
