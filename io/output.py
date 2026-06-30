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
    print("═" * (lw + vw + 3))
    for lbl, val in formatted:
        print(f"{lbl:<{lw}} : {val:>{vw}}")
    print("═" * (lw + vw + 3))

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
    rule = "═" * (lw1 + 3 + vw1 + len(sep) + lw2 + 3 + vw2 if right else lw1 + 3 + vw1)
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
    rule = "═" * (lw1+3+vw1 + (len(sep)+lw2+3+vw2 if right else 0))
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
    print("═" * rule_len)

    for r in range(nrows):
        parts = []
        for c in range(ncols):
            i = r + c*nrows
            if i < n:
                lbl, val = fmted[i]
                parts.append(pad_right(lbl, lab_w[c]) + " : " + pad_left(val, val_w[c]))
        print(sep.join(parts))
