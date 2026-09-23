"""Display-name checks only; no numerical execution or document compilation."""
from pathlib import Path

import pytest

from scripts.reports.adr_report_labels import PMG_TEX, pmg_names, prose


@pytest.mark.parametrize("legacy", [
    "Native", "native", "Native $hp$-BSR", "native $hp$-BSR",
    "Native $hp$", "native $hp$", "Native hp", "native hp",
    "$hp$-BSR", "native $p$MG--AMG",
])
def test_pmg_aliases_preserve_policy_and_are_idempotent(legacy):
    for policy in ("standard", "robust"):
        expected = f"{PMG_TEX} {policy}"
        assert pmg_names(f"{legacy} {policy}") == expected
        assert pmg_names(expected) == expected
        assert prose(f"{legacy} {policy}") == expected


def test_pmg_names_preserve_internal_identifiers_and_tex_targets():
    identifiers = (
        r"\input{\ADRResultsPath/native_completion/summary.tex} "
        r"\label{sec:adr-native-completion} \ref{tab:adr-native-coverage} "
        r"\path{native/run.json} \url{https://example.org/native} "
        r"\href{native}{Native} native_hp_standard native_hp_robust"
    )
    assert pmg_names(identifiers) == identifiers.replace(
        r"\href{native}{Native}", r"\href{native}{" + PMG_TEX + "}")
    assert pmg_names("ASM+PP and native") == f"ASM+PP and {PMG_TEX}"


def test_report_sources_and_templates_use_canonical_pmg_display_names():
    root = Path(__file__).resolve().parents[1]
    report = root / "docs/research/solver_studies/adr_scaling_2026_09_17"
    sources = list(report.rglob("*.tex"))
    sources.extend(root / "scripts/reports" / name for name in (
        "adr_results_section.tex.in", "adr_synthesis_section.tex.in",
        "oscillatory_scaling_subsection.tex.in", "oscillatory_adr_section.tex.in",
    ))
    assert sources
    for path in sources:
        text = path.read_text()
        assert pmg_names(text) == text, path
