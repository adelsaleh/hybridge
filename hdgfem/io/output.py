from __future__ import annotations

from contextlib import contextmanager
import sys
import time


def _stdout_can_encode(text: str) -> bool:
    """Return whether the active stdout encoding can represent ``text``."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def _heavy_rule_char() -> str:
    return "═" if _stdout_can_encode("═") else "="


def _light_rule_char() -> str:
    return "─" if _stdout_can_encode("─") else "-"


def logv(config, level: int, message: str) -> None:
    """Print ``message`` when ``config.verbosity`` is at least ``level``.

    This small helper is intended for scripts and examples that expose an
    argparse-style ``verbosity`` attribute but do not need a full logging setup.
    """
    if int(getattr(config, "verbosity", 1)) >= int(level):
        print(message, flush=True)


@contextmanager
def timed_section(config, level: int, label: str, **fields):
    """Emit ``LABEL_START`` and ``LABEL_DONE time=...`` messages around a block.

    Parameters in ``fields`` are printed on the ``START`` line.  The messages
    are suppressed unless ``config.verbosity >= level``.
    """
    verbose = int(getattr(config, "verbosity", 1)) >= int(level)
    if verbose:
        extras = " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"{label}_START{(' ' + extras) if extras else ''}", flush=True)
    start = time.perf_counter()
    try:
        yield
    finally:
        if verbose:
            print(f"{label}_DONE time={time.perf_counter() - start:.3f}", flush=True)


def pretty_print(items, title="Results", pad_lines=1, default_fmt=".5g"):
    # format all values
    formatted = []
    for label, value, fmt in items:
        spec = fmt or default_fmt
        formatted.append((label, format(value, spec)))

    # column widths
    lw = max(len(lbl) for lbl, _ in formatted)
    vw = max(len(val) for _, val in formatted)

    # print
    print("\n" * pad_lines, end="")
    print(title)
    print(_heavy_rule_char() * (lw + vw + 3))
    for lbl, val in formatted:
        print(f"{lbl:<{lw}} : {val:>{vw}}")
    print(_heavy_rule_char() * (lw + vw + 3))

def pretty_print_2row(items, title="Results", pad_lines=1, default_fmt=".5g", sep="    "):
    """
    items: list of (label, value, fmt) where fmt is like '.5e', '.3f', ',d', or None
    Ensures equal spacing between the two columns across all rows, even with Unicode.
    """
    # Display width (prefer wcwidth; fallback to simple EAW-based width)
    try:
        from wcwidth import wcswidth  # pip install wcwidth
    except Exception:
        from unicodedata import east_asian_width
        def wcswidth(s):
            return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(s))

    def pad_right(s, w):
        s = str(s); d = wcswidth(s); return s + " " * max(0, w - d)
    def pad_left(s, w):
        s = str(s); d = wcswidth(s); return " " * max(0, w - d) + s

    # Format values
    fmted = [(lbl, format(val, fmt or default_fmt)) for lbl, val, fmt in items]
    left, right = fmted[::2], fmted[1::2]

    # Column widths by display width
    lw1 = max((wcswidth(lbl) for lbl,_ in left), default=0)
    vw1 = max((wcswidth(val) for _,val in left), default=0)
    lw2 = max((wcswidth(lbl) for lbl,_ in right), default=0)
    vw2 = max((wcswidth(val) for _,val in right), default=0)

    # Print
    print("\n" * pad_lines, end="")
    print(title)
    rule = _heavy_rule_char() * (lw1 + 3 + vw1 + len(sep) + lw2 + 3 + vw2 if right else lw1 + 3 + vw1)
    print(rule)

    for i in range(0, len(fmted), 2):
        lbl1, val1 = fmted[i]
        left_part = pad_right(lbl1, lw1) + " : " + pad_left(val1, vw1)
        if i + 1 < len(fmted):
            lbl2, val2 = fmted[i+1]
            right_part = pad_right(lbl2, lw2) + " : " + pad_left(val2, vw2)
            print(left_part + sep + right_part)
        else:
            print(left_part)


def pretty_print_2col(items, title="Results", pad_lines=1, default_fmt=".5g", sep="    "):
    """
    items: list of (label, value, fmt)
    Fills the table column-first: the first half of items go in the left column,
    the second half in the right column.
    """

    # Display width (use wcwidth if available, else fallback)
    try:
        from wcwidth import wcswidth
    except Exception:
        from unicodedata import east_asian_width
        def wcswidth(s):
            return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(s))

    def pad_right(s, w): s=str(s); return s + " " * max(0, w - wcswidth(s))
    def pad_left(s, w):  s=str(s); return " " * max(0, w - wcswidth(s)) + s

    # Format values
    fmted = [(lbl, format(val, fmt or default_fmt)) for lbl, val, fmt in items]

    # Split into two columns (column-first filling)
    n = (len(fmted) + 1) // 2
    left, right = fmted[:n], fmted[n:]

    # Widths
    lw1 = max((wcswidth(lbl) for lbl,_ in left), default=0)
    vw1 = max((wcswidth(val) for _,val in left), default=0)
    lw2 = max((wcswidth(lbl) for lbl,_ in right), default=0)
    vw2 = max((wcswidth(val) for _,val in right), default=0)

    # Print
    print("\n" * pad_lines, end="")
    print(title)
    rule = _heavy_rule_char() * (lw1+3+vw1 + (len(sep)+lw2+3+vw2 if right else 0))
    print(rule)

    for i in range(n):
        lbl1, val1 = left[i]
        left_part = pad_right(lbl1, lw1) + " : " + pad_left(val1, vw1)
        if i < len(right):
            lbl2, val2 = right[i]
            right_part = pad_right(lbl2, lw2) + " : " + pad_left(val2, vw2)
            print(left_part + sep + right_part)
        else:
            print(left_part)

def pretty_print_ncol(items, ncols=2, title="Results", pad_lines=1, default_fmt=".5g", sep="    "):
    """
    items: list of (label, value, fmt) where fmt can be None to use default_fmt
    ncols: number of columns (>=1)
    Fills column-first. Aligns using display width so Unicode aligns correctly.
    """
    # Display width helper (prefer wcwidth, else simple fallback)
    try:
        from wcwidth import wcswidth
    except Exception:
        from unicodedata import east_asian_width
        def wcswidth(s): return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(s))

    def pad_right(s, w):
        s = str(s); return s + " " * max(0, w - wcswidth(s))
    def pad_left(s, w):
        s = str(s); return " " * max(0, w - wcswidth(s)) + s

    # 1) format values
    fmted = [(lbl, format(val, fmt or default_fmt)) for lbl, val, fmt in items]

    # 2) determine grid (column-first)
    n = len(fmted)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    # 3) compute per-column widths (label & value each column)
    lab_w = [0]*ncols
    val_w = [0]*ncols
    for c in range(ncols):
        for r in range(nrows):
            i = r + c*nrows
            if i < n:
                lbl, val = fmted[i]
                lab_w[c] = max(lab_w[c], wcswidth(lbl))
                val_w[c] = max(val_w[c], wcswidth(val))

    # 4) print
    print("\n" * pad_lines, end="")
    print(title)

    if n == 0:
        print("(no data)")
        return

    rule_len = sum(lab_w[c] + 3 + val_w[c] for c in range(ncols if n >= ncols else 1))
    rule_len += len(sep) * (max(0, min(ncols, (n + nrows - 1)//nrows) - 1))
    print(_heavy_rule_char() * rule_len)

    for r in range(nrows):
        parts = []
        for c in range(ncols):
            i = r + c*nrows
            if i < n:
                lbl, val = fmted[i]
                parts.append(pad_right(lbl, lab_w[c]) + " : " + pad_left(val, val_w[c]))
        print(sep.join(parts))


def pretty_print_sections(sections, title="Results", pad_lines=1, default_fmt=".5g", sep="    "):
    """
    Print named sections as side-by-side columns.

    sections: list of (section_title, items), where items is a list of
    (label, value, fmt). Uneven section lengths are allowed.
    """
    try:
        from wcwidth import wcswidth
    except Exception:
        from unicodedata import east_asian_width
        def wcswidth(s): return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in str(s))

    def pad_right(s, w):
        s = str(s); return s + " " * max(0, w - wcswidth(s))

    def format_value(value, fmt):
        spec = fmt or default_fmt
        if value is None:
            return "None"
        if spec == "s":
            return str(value)
        return format(value, spec)

    formatted_sections = []
    for section_title, items in sections:
        formatted_items = [(lbl, format_value(val, fmt)) for lbl, val, fmt in items]
        formatted_sections.append((section_title, formatted_items))

    print("\n" * pad_lines, end="")
    print(title)

    if not formatted_sections:
        print("(no data)")
        return

    rendered_sections = [
        [f"{lbl}: {val}" for lbl, val in items]
        for _, items in formatted_sections
    ]
    column_widths = [
        max(wcswidth(section_title), max((wcswidth(item) for item in rendered_sections[i]), default=0))
        for i, (section_title, _) in enumerate(formatted_sections)
    ]

    rule_len = sum(column_widths) + len(sep) * max(0, len(column_widths) - 1)
    print(_heavy_rule_char() * rule_len)
    print(sep.join(pad_right(section_title, width) for (section_title, _), width in zip(formatted_sections, column_widths)))
    print(sep.join(_light_rule_char() * width for width in column_widths))

    max_rows = max((len(items) for _, items in formatted_sections), default=0)

    for row in range(max_rows):
        row_parts = []
        for col, items in enumerate(rendered_sections):
            part = items[row] if row < len(items) else ""
            row_parts.append(pad_right(part, column_widths[col]))
        print(sep.join(row_parts))
