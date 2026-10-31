"""Presentation names for recorded ADR configurations; no numerical execution."""
from __future__ import annotations

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))

import re

PMG = r'$p$MG–AMG'
PMG_TEX = r'$p$MG--AMG'


def polynomial_degree(row):
    value = row.get('pp_degree', row.get('configuration', {}).get('polynomial_degree'))
    if value is None:
        raise ValueError(f"Missing recorded polynomial degree: {row.get('candidate', row.get('method'))}")
    value = int(value)
    assert value > 0
    return value


def polynomial_label(row, family=None):
    name = row.get('candidate', row.get('method', '')).lower()
    if family is None:
        family = 'BJ' if name.startswith('bj') else 'ASM' if name.startswith('asm') else ''
    return (family + '+' if family else '') + f'PP({polynomial_degree(row)})'


def pmg_names(text):
    """Use pMG--AMG display names without changing recorded IDs or TeX links."""
    # These arguments are identifiers, not presentation text. Keep their exact
    # spelling so existing report inclusions and cross-references still resolve.
    parts = re.split(
        r'(\\(?:input|include|includegraphics|label|ref|eqref|pageref|path|url|href)'
        r'\*?(?:\[[^\]]*\])?\{[^}]*\})', text)
    for index in range(0, len(parts), 2):
        part = parts[index]
        for old in ('Native $p$MG--AMG', 'native $p$MG--AMG',
                    'Native $hp$-BSR', 'native $hp$-BSR',
                    'Native $hp$', 'native $hp$', '$hp$-BSR',
                    'Native hp', 'native hp'):
            part = part.replace(old, PMG_TEX)
        parts[index] = re.sub(
            r'(?<![\w/\\:-])native(?![\w/:-])',
            lambda match: PMG_TEX, part, flags=re.IGNORECASE)
    return ''.join(parts)


def prose(text):
    """Remove ambiguous legacy names in prose, keeping paths/labels unchanged."""
    text = pmg_names(text)
    # Bare PP in general discussion has no single configured degree; spell it out.
    text=re.sub(r'ASM\+PP(?!\()', 'polynomially preconditioned ASM', text)
    text=re.sub(r'BJ\+PP(?!\()', 'polynomially preconditioned BJ', text)
    text=re.sub(r'(?<![\w/])PP(?![\w(])', 'polynomial preconditioning', text)
    return text
